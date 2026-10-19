//! **benilla fork.** The render sub-app on its own Web Worker (`BENILLA.md`, "The render worker").
//!
//! # Why
//!
//! On the threaded browser build the page's one thread ran the main schedules, extract and the
//! whole `Render` schedule in series, so a frame cost main + render. Native bevy splits them with
//! `pipelined_rendering`, which this target cannot use: it moves the render `SubApp` between
//! threads twice a frame, and every wgpu handle under `wasm32 + atomics` is a `SendWrapper` around
//! a JS object that belongs to one agent (wasm-bindgen's heap table is per thread, and WebGPU
//! objects do not transfer between agents). Its handshake also blocks, and the page may not.
//!
//! # How
//!
//! The render world never moves. It is built on the worker (`App::finish`/`cleanup` run there, see
//! [`run_blocking`]), its device and surface are created there (from an `OffscreenCanvas` the page
//! transferred), and it stays there. For extract the *main* world visits instead: the page parks a
//! pointer to it, flips [`STATE`] to `EXTRACT`, and spins on the atomic for the few milliseconds the
//! worker takes to run `SubApp::extract` against it. The main world's contents are `Send + Sync` by
//! trait bound (`Resource`, `Component`), so the worker reading them is checked by the type
//! system; a `NonSend` resource keeps bevy's own thread check.
//!
//! The page does not wait for a render at the end of its update. If the worker is `IDLE` it
//! extracts there ([`renderer_extract`]); if not, the extract is *owed*, the page's callback
//! returns to the browser, and the next update pays it before its `First` runs
//! ([`pay_owed_extract`]), waiting only for whatever is left of the render. So every main update is
//! still extracted exactly once and before the next one starts, as on one thread: bevy's messages
//! (`AssetEvent`s that `extract_render_asset` reads, removed components) live two updates, and an
//! update that was never extracted would drop them. The worker renders right after extract and
//! returns to its event loop, which is where the browser presents an `OffscreenCanvas` and settles
//! the promises the renderer awaits (async pipeline creation).
//!
//! # The page-side seams
//!
//! The main world keeps `RenderDevice` and `RenderQueue` (they are `Send`), and benilla's main
//! world creates and writes a few buffers itself. On the page those calls never touch wgpu: a
//! buffer created there is materialised on the worker at first use, a `write_buffer` is queued and
//! applied right after the next extract ([`defer_write`]), and a wrapped handle whose last owner
//! lets go on the page is freed on the worker ([`bury`]).
//!
//! Off this mode (no worker started: `?renderworker=0`, `?nothreads=1`, a page without the bridge)
//! [`is_render_thread`] is always true and none of this runs.

use alloc::{boxed::Box, collections::VecDeque, sync::Arc, vec::Vec};
use core::{
    cell::{Cell, RefCell},
    sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering},
    time::Duration,
};
use std::sync::{MutexGuard, OnceLock, PoisonError};

use bevy_platform::sync::Mutex;

use bevy_app::{App, AppLabel, First, MainScheduleOrder, Plugin, SubApp};
use bevy_ecs::{
    schedule::{Schedule, ScheduleLabel},
    world::World,
};
use bevy_platform::time::Instant;
use wasm_bindgen::JsValue;

use crate::{
    RenderApp,
    render_resource::Buffer,
    renderer::{RenderDevice, RenderQueue},
};

/// The worker has no render app yet (before `cleanup`), or this mode is off.
const NONE: u32 = 0;
/// The worker holds the render app and is waiting for an extract.
const IDLE: u32 = 1;
/// The page has parked its main world in [`MAIN_WORLD`] and is spinning until this changes.
const EXTRACT: u32 = 2;
/// Extract is done and the page has its world back; the worker is rendering.
const RENDERING: u32 = 3;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static KICK: OnceLock<fn()> = OnceLock::new();
static STATE: AtomicU32 = AtomicU32::new(NONE);
static MAIN_WORLD: AtomicPtr<World> = AtomicPtr::new(core::ptr::null_mut());
/// The last main update ended with the worker busy, so its extract is still to run. Page only.
static OWED: AtomicBool = AtomicBool::new(false);

/// How long the page waits on the worker for one extract before calling it dead. A worker that
/// panicked traps in its own agent and never answers; without a ceiling the page would spin
/// forever with its world lent out.
const EXTRACT_DEADLINE: Duration = Duration::from_secs(20);
/// The same for the boot-time jobs [`run_blocking`] waits on (`App::finish`, `App::cleanup`).
const BLOCKING_DEADLINE: Duration = Duration::from_secs(120);

thread_local! {
    static ON_WORKER: Cell<bool> = const { Cell::new(false) };
    static CANVAS: RefCell<Option<JsValue>> = const { RefCell::new(None) };
    static RENDER_APP: RefCell<Option<SubApp>> = const { RefCell::new(None) };
}

/// Whether this session renders on the worker. Set once, on the page, before the app is built.
pub fn active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// Whether wgpu may be touched on this thread: always, unless the render worker is on and this is
/// not it.
pub fn is_render_thread() -> bool {
    !active() || ON_WORKER.with(Cell::get)
}

/// Turn the mode on. The host calls this on the page after it has started the worker and before
/// the `App` is built; `kick` wakes the worker's event loop (a `postMessage`).
pub fn install(kick: fn()) {
    let _ = KICK.set(kick);
    ACTIVE.store(true, Ordering::Release);
}

/// The worker's first call, once its module is instantiated: mark the thread and keep the canvas
/// the page transferred to it. Then drain whatever the page posted while the worker was loading.
pub fn enter(canvas: JsValue) {
    ON_WORKER.with(|w| w.set(true));
    CANVAS.with_borrow_mut(|c| *c = Some(canvas));
    poll();
}

/// The transferred `OffscreenCanvas`, on the worker.
pub(crate) fn canvas() -> Option<JsValue> {
    CANVAS.with_borrow(Clone::clone)
}

fn kick() {
    if let Some(kick) = KICK.get() {
        kick();
    }
}

/// bevy's `Mutex`, which spins on the page (a parked page traps) and parks on the worker
/// (`third_party/bevy_platform/BENILLA.md`). Every critical section here is a push or a swap.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Spin until `done`, reading the clock every few thousand turns; past `deadline`, panic with `what`.
fn spin_until(done: impl Fn() -> bool, deadline: Duration, what: &str) {
    let start = Instant::now();
    let mut turns = 0u32;
    while !done() {
        core::hint::spin_loop();
        turns = turns.wrapping_add(1);
        if turns.is_multiple_of(4096) && start.elapsed() > deadline {
            panic!("render worker: {what} (no answer in {deadline:?})");
        }
    }
}

type Job = Box<dyn FnOnce() + Send>;
static JOBS: Mutex<VecDeque<Job>> = Mutex::new(VecDeque::new());

/// Run `job` on the worker, the next time its event loop wakes.
pub fn post(job: impl FnOnce() + Send + 'static) {
    lock(&JOBS).push_back(Box::new(job));
    kick();
}

/// Lets a closure that is not `Send` ride a job: [`run_blocking`] waits for it, so nothing it
/// borrows is touched by the page meanwhile.
struct AssertSend<F>(F);
// SAFETY: see `run_blocking` — the page does not touch what `F` borrows until the job is done.
unsafe impl<F> Send for AssertSend<F> {}
impl<F: FnOnce()> AssertSend<F> {
    fn call(self) {
        (self.0)();
    }
}

/// Run `f` on the worker and spin until it returns. Boot only: the runner's `App::finish` and
/// `App::cleanup`, so every plugin `finish` that reads the device or builds GPU resources runs on
/// the thread that owns the device.
///
/// # Safety
///
/// `f` runs on another thread with whatever it borrows. The caller must not let anything else
/// touch those borrows until this returns, and what `f` reaches must be sound to use from the
/// worker (the `App`'s worlds are; its `NonSend` resources are thread-checked by bevy).
pub unsafe fn run_blocking<F: FnOnce()>(f: F) {
    let done = Arc::new(AtomicBool::new(false));
    let flag = done.clone();
    let job = AssertSend(move || {
        f();
        flag.store(true, Ordering::Release);
    });
    let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || job.call());
    // SAFETY: the box only outlives `f`'s borrows if this function returns early, and it does not:
    // it waits for the job below, and a job that never finishes ends in the deadline panic.
    let job: Job = unsafe { core::mem::transmute(job) };
    lock(&JOBS).push_back(job);
    kick();
    spin_until(
        || done.load(Ordering::Acquire),
        BLOCKING_DEADLINE,
        "a boot job did not finish",
    );
}

/// What the page measures of the worker, read by `benilla_app::webdiag`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderWorkerStats {
    /// Render updates the worker ran.
    pub renders: u64,
    /// Total and worst worker time of those updates, microseconds (extract excluded).
    pub render_us: u64,
    pub render_max_us: u64,
    /// Updates that ended with the worker busy, their extract owed to the next update.
    pub owed: u64,
    /// Total and worst wait at the start of an update for the worker to finish (owed extracts), µs.
    pub late_us: u64,
    pub late_max_us: u64,
    /// Total and worst page-side wait for an extract (wake + extract + queued writes), µs.
    pub wait_us: u64,
    pub wait_max_us: u64,
}

static RENDERS: AtomicU64 = AtomicU64::new(0);
static RENDER_US: AtomicU64 = AtomicU64::new(0);
static RENDER_MAX_US: AtomicU64 = AtomicU64::new(0);
static OWED_COUNT: AtomicU64 = AtomicU64::new(0);
static LATE_US: AtomicU64 = AtomicU64::new(0);
static LATE_MAX_US: AtomicU64 = AtomicU64::new(0);
static WAIT_US: AtomicU64 = AtomicU64::new(0);
static WAIT_MAX_US: AtomicU64 = AtomicU64::new(0);

/// The counters since the last call (each call starts a new window).
pub fn take_stats() -> RenderWorkerStats {
    RenderWorkerStats {
        renders: RENDERS.swap(0, Ordering::Relaxed),
        render_us: RENDER_US.swap(0, Ordering::Relaxed),
        render_max_us: RENDER_MAX_US.swap(0, Ordering::Relaxed),
        owed: OWED_COUNT.swap(0, Ordering::Relaxed),
        late_us: LATE_US.swap(0, Ordering::Relaxed),
        late_max_us: LATE_MAX_US.swap(0, Ordering::Relaxed),
        wait_us: WAIT_US.swap(0, Ordering::Relaxed),
        wait_max_us: WAIT_MAX_US.swap(0, Ordering::Relaxed),
    }
}

fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// The page's half of the frame handshake, as the extract function of [`RenderWorkerExtractApp`].
///
/// Never waits for a render: a busy worker leaves the extract owed to the start of the next update
/// ([`pay_owed_extract`]), and the page's callback returns. It spins only while the worker
/// extracts.
fn renderer_extract(main_world: &mut World, _: &mut World) {
    if STATE.load(Ordering::Acquire) != IDLE {
        OWED.store(true, Ordering::Relaxed);
        OWED_COUNT.fetch_add(1, Ordering::Relaxed);
        return;
    }
    hand_over(main_world);
}

/// The main schedule's first step: an extract the last update owed, after the rest of the render.
/// Nothing of this update has run yet, so the extract sees the last update's world and messages.
fn pay_owed_extract(world: &mut World) {
    if !OWED.swap(false, Ordering::Relaxed) {
        return;
    }
    let start = Instant::now();
    spin_until(
        || STATE.load(Ordering::Acquire) == IDLE,
        EXTRACT_DEADLINE,
        "the worker did not finish a render",
    );
    let late = micros(start.elapsed());
    LATE_US.fetch_add(late, Ordering::Relaxed);
    LATE_MAX_US.fetch_max(late, Ordering::Relaxed);
    hand_over(world);
}

/// Lend the main world to the worker for one extract, spinning until it is back.
fn hand_over(main_world: &mut World) {
    let start = Instant::now();
    MAIN_WORLD.store(main_world, Ordering::Release);
    STATE.store(EXTRACT, Ordering::Release);
    kick();
    spin_until(
        || STATE.load(Ordering::Acquire) != EXTRACT,
        EXTRACT_DEADLINE,
        "the worker stopped answering an extract",
    );
    MAIN_WORLD.store(core::ptr::null_mut(), Ordering::Release);
    let waited = micros(start.elapsed());
    WAIT_US.fetch_add(waited, Ordering::Relaxed);
    WAIT_MAX_US.fetch_max(waited, Ordering::Relaxed);
}

/// The worker's whole loop body, run each time its event loop wakes: boot jobs first, then an
/// extract and a render if the page asked for one.
pub fn poll() {
    loop {
        let job = lock(&JOBS).pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    if STATE.load(Ordering::Acquire) == EXTRACT {
        RENDER_APP.with_borrow_mut(|app| {
            let app = app
                .as_mut()
                .expect("render worker: an extract arrived before the render app");
            // SAFETY: the page stored this pointer from its `&mut World` and spins, touching
            // nothing, until `STATE` leaves `EXTRACT` below.
            let main_world = unsafe { &mut *MAIN_WORLD.load(Ordering::Acquire) };
            app.extract(main_world);
            // The page's writes from the update this extract closes, before the page moves on.
            if let Some(queue) = app.world().get_resource::<RenderQueue>() {
                drain_writes(queue);
            }
            STATE.store(RENDERING, Ordering::Release);

            let start = Instant::now();
            app.update();
            drain_graveyard();
            let took = micros(start.elapsed());
            RENDERS.fetch_add(1, Ordering::Relaxed);
            RENDER_US.fetch_add(took, Ordering::Relaxed);
            RENDER_MAX_US.fetch_max(took, Ordering::Relaxed);
            STATE.store(IDLE, Ordering::Release);
        });
    } else {
        drain_graveyard();
    }
}

/// The page-side stand-in for the render sub-app: no schedules, only the handshake as its extract.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AppLabel)]
pub struct RenderWorkerExtractApp;

/// The main schedule's step before `First` that pays an owed extract ([`pay_owed_extract`]).
#[derive(ScheduleLabel, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RenderWorkerCatchUp;

/// Moves the render sub-app onto the worker at `cleanup` and leaves the handshake in its place.
/// Added by `RenderPlugin` when [`active`].
pub struct RenderWorkerPlugin;

impl Plugin for RenderWorkerPlugin {
    fn build(&self, app: &mut App) {
        if app.get_sub_app(RenderApp).is_none() {
            return;
        }
        let mut sub_app = SubApp::new();
        sub_app.set_extract(renderer_extract);
        app.insert_sub_app(RenderWorkerExtractApp, sub_app);

        let mut catch_up = Schedule::new(RenderWorkerCatchUp);
        catch_up.add_systems(pay_owed_extract);
        app.add_schedule(catch_up);
        app.world_mut()
            .resource_mut::<MainScheduleOrder>()
            .insert_before(First, RenderWorkerCatchUp);
    }

    fn cleanup(&self, app: &mut App) {
        if app.get_sub_app(RenderWorkerExtractApp).is_none() {
            return;
        }
        // `App::cleanup` runs on the worker in this mode (`run_blocking`), so the render app is
        // parked on the thread that will run it, and never crosses again.
        assert!(
            ON_WORKER.with(Cell::get),
            "render worker: plugins must finish on the worker (bevy_winit's finish hook)"
        );
        let render_app = app
            .remove_sub_app(RenderApp)
            .expect("render worker: no RenderApp to move — another plugin removed it");
        RENDER_APP.with_borrow_mut(|slot| *slot = Some(render_app));
        STATE.store(IDLE, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------------------------
// Page-side GPU calls
// ---------------------------------------------------------------------------------------------

/// The device lazily created buffers are made on. Set wherever the render resources are unpacked,
/// which in this mode is on the worker.
static DEVICE: Mutex<Option<RenderDevice>> = Mutex::new(None);

pub(crate) fn set_device(device: &RenderDevice) {
    *lock(&DEVICE) = Some(device.clone());
}

/// Create a buffer the page asked for, now that it is used on the worker.
pub(crate) fn create_buffer(
    desc: &wgpu::BufferDescriptor,
    contents: Option<&[u8]>,
) -> wgpu::Buffer {
    let device = lock(&DEVICE)
        .clone()
        .expect("render worker: a page-made buffer was used before the device existed");
    match contents {
        Some(contents) => wgpu::util::DeviceExt::create_buffer_init(
            device.wgpu_device(),
            &wgpu::util::BufferInitDescriptor {
                label: desc.label,
                contents,
                usage: desc.usage,
            },
        ),
        None => device.wgpu_device().create_buffer(desc),
    }
}

type Write = (Buffer, wgpu::BufferAddress, Vec<u8>);
static WRITES: Mutex<Vec<Write>> = Mutex::new(Vec::new());

/// Queue a `write_buffer` the page made. Applied on the worker right after the next extract, so it
/// lands before that frame's own writes and submits, as it did when both ran on one thread.
pub(crate) fn defer_write(buffer: Buffer, offset: wgpu::BufferAddress, data: &[u8]) {
    lock(&WRITES).push((buffer, offset, data.to_vec()));
}

fn drain_writes(queue: &RenderQueue) {
    let writes = core::mem::take(&mut *lock(&WRITES));
    for (buffer, offset, data) in writes {
        queue.0.write_buffer(&buffer, offset, &data);
    }
}

/// A wrapped wgpu value whose owner let go of it on the wrong thread.
pub(crate) trait Buried: Send {
    /// Whether this thread may drop it.
    fn owned_here(&self) -> bool;
}

static GRAVE: Mutex<Vec<Box<dyn Buried>>> = Mutex::new(Vec::new());

/// Keep `value` until its own thread can drop it. A `SendWrapper` dropped elsewhere panics, and in
/// this mode the page is the last owner of a handful of handles (a main-world buffer, the
/// main world's device clone at exit).
pub(crate) fn bury(value: Box<dyn Buried>) {
    lock(&GRAVE).push(value);
}

fn drain_graveyard() {
    let mut grave = lock(&GRAVE);
    if grave.is_empty() {
        return;
    }
    let (here, elsewhere): (Vec<_>, Vec<_>) = core::mem::take(&mut *grave)
        .into_iter()
        .partition(|b| b.owned_here());
    *grave = elsewhere;
    drop(grave);
    drop(here);
}
