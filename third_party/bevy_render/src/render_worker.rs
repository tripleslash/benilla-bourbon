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
//! The page never waits for the worker's event loop. A worker that has rendered can be held up
//! between its tasks until the page's thread is free (measured in Chrome: after a task that
//! submitted a frame, no task of the worker ran — no message, no timer — for as long as the page
//! spun, 20 s), so a page that spun on it would wait for itself. The page spins only while the
//! worker is provably inside an extract: it publishes the extract, and the worker *claims* it
//! (`EXTRACT` → `TAKEN`) from inside its poll before touching the world. Unclaimed within
//! [`CLAIM_BUDGET`], the page withdraws it and returns to the browser. After a render the worker is
//! `SETTLING` until a fresh task of its own ([`settle`]) marks it `IDLE`, so the page does not
//! publish into the gap where the worker waits on it.
//!
//! An update whose extract did not happen leaves it *owed*. When the page returns to the browser
//! with an extract owed it *lends* the worker its world ([`lend`], `bevy_winit`'s `set_lend_world`,
//! after every runner callback): the worker takes it as soon as it is free, from its settle or a
//! wake, while the page waits in its event loop, and the page takes it back before its next
//! callback touches it ([`reclaim`]), waiting only if the worker is mid-extract. The runner does not
//! start another update until the extract is paid ([`before_update`], `set_may_update`): if the
//! loan went untaken and the worker is still busy, it skips the update and asks for the next frame. So every main update is still extracted exactly
//! once and before the next one starts, as on one thread: bevy's messages (`AssetEvent`s that
//! `extract_render_asset` reads, removed components) live two updates, and an update that was never
//! extracted would drop them. The worker renders right after extract and returns to its event loop,
//! which is where the browser presents an `OffscreenCanvas` and settles the promises the renderer
//! awaits (async pipeline creation).
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

use bevy_app::{App, AppLabel, Plugin, SubApp};
use bevy_ecs::world::World;
use bevy_platform::time::Instant;
use wasm_bindgen::JsValue;

use crate::{
    RenderApp,
    render_resource::Buffer,
    renderer::{RenderDevice, RenderQueue},
};

/// The worker has no render app yet (before `cleanup`), or this mode is off.
const NONE: u32 = 0;
/// The worker holds the render app, its event loop is free, and it is waiting for an extract.
const IDLE: u32 = 1;
/// The page has parked its main world in [`MAIN_WORLD`] and is waiting for the worker to claim it.
const EXTRACT: u32 = 2;
/// Extract is done and the page has its world back; the worker is rendering.
const RENDERING: u32 = 3;
/// The worker claimed the extract and is running it; the page spins until it is done.
const TAKEN: u32 = 4;
/// The worker has rendered and returned to its event loop; [`settle`] marks it `IDLE` from its
/// next task.
const SETTLING: u32 = 5;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static KICK: OnceLock<fn()> = OnceLock::new();
static WAKE_PAGE: OnceLock<fn()> = OnceLock::new();
static STATE: AtomicU32 = AtomicU32::new(NONE);
static MAIN_WORLD: AtomicPtr<World> = AtomicPtr::new(core::ptr::null_mut());
/// Where the worker is, for the page's deadline panic: a hung worker cannot say so itself.
static WORKER_AT: AtomicU32 = AtomicU32::new(AT_LOOP);
const AT_LOOP: u32 = 0;
const AT_JOBS: u32 = 1;
const AT_EXTRACT: u32 = 2;
const AT_WRITES: u32 = 3;
const AT_RENDER: u32 = 4;
const AT_GRAVE: u32 = 5;

fn worker_at() -> &'static str {
    match WORKER_AT.load(Ordering::Relaxed) {
        AT_LOOP => "outside poll (its event loop: a microtask or another message)",
        AT_JOBS => "running a boot job",
        AT_EXTRACT => "in SubApp::extract",
        AT_WRITES => "applying the page's queued writes",
        AT_RENDER => "in the render update",
        AT_GRAVE => "draining the graveyard",
        _ => "unknown",
    }
}

fn at(stage: u32) {
    WORKER_AT.store(stage, Ordering::Relaxed);
}

/// How far device creation on the worker got: 0 not started, 1 job ran, 2 task spawned, 3 task
/// first polled, 4 `initialize_renderer` returned, 5 resources published.
pub(crate) static DEVICE_AT: AtomicU32 = AtomicU32::new(0);

/// One line of the handshake's state, for a console on the page (`__wenilla_render_state()`): the
/// statics are in shared memory, so the page reads what the worker wrote.
pub fn describe() -> String {
    alloc::format!(
        "state={} worker={} device={} renders={}",
        match STATE.load(Ordering::Acquire) {
            NONE => "none",
            IDLE => "idle",
            EXTRACT => "extract",
            TAKEN => "taken",
            RENDERING => "rendering",
            SETTLING => "settling",
            _ => "?",
        },
        worker_at(),
        DEVICE_AT.load(Ordering::Relaxed),
        RENDERS.load(Ordering::Relaxed),
    )
}

/// The last main update ended without an extract, so it is still to run. Page only.
static OWED: AtomicBool = AtomicBool::new(false);

/// How long a published extract waits for the worker to claim it before the page withdraws it.
/// A free worker answers a wake in well under a millisecond; one that does not is waiting for the
/// page's thread, which spinning longer would only keep from it.
const CLAIM_BUDGET: Duration = Duration::from_millis(4);
/// How long the page waits for a claimed extract before calling the worker dead. A worker that
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

/// How the worker wakes the page's event loop (winit's proxy, through `bevy_winit`). The page
/// waits for the device in `RenderPlugin::ready`, which it checks only when its event loop runs, and
/// a device made here settles no promise there: unwoken, an unfocused page sleeps through it (found
/// live: a black canvas until the first click).
pub fn set_wake_page(wake: fn()) {
    let _ = WAKE_PAGE.set(wake);
}

pub(crate) fn wake_page() {
    if let Some(wake) = WAKE_PAGE.get() {
        wake();
    }
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
            panic!(
                "render worker: {what} (no answer in {deadline:?}; the worker is {})",
                worker_at()
            );
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
    /// Updates that ended without an extract (the worker busy, or a publish withdrawn).
    pub owed: u64,
    /// Times the runner found the last update's extract still owed and could not pay it (it holds
    /// the next update, or asks for another frame).
    pub skipped: u64,
    /// Published extracts the worker did not claim within [`CLAIM_BUDGET`].
    pub withdrawn: u64,
    /// Worlds lent to the worker when the page returned to the browser with an extract owed
    /// ([`lend`]), and those the worker extracted before the page came back ([`reclaim`]).
    pub lent: u64,
    pub lent_paid: u64,
    /// Extracts done, and their total and worst page-side wait (wake + extract + queued writes), µs.
    pub extracts: u64,
    pub wait_us: u64,
    pub wait_max_us: u64,
}

static RENDERS: AtomicU64 = AtomicU64::new(0);
static RENDER_US: AtomicU64 = AtomicU64::new(0);
static RENDER_MAX_US: AtomicU64 = AtomicU64::new(0);
static OWED_COUNT: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);
static WITHDRAWN: AtomicU64 = AtomicU64::new(0);
static LENT_COUNT: AtomicU64 = AtomicU64::new(0);
static LENT_PAID: AtomicU64 = AtomicU64::new(0);
static EXTRACTS: AtomicU64 = AtomicU64::new(0);
static WAIT_US: AtomicU64 = AtomicU64::new(0);
static WAIT_MAX_US: AtomicU64 = AtomicU64::new(0);

/// The counters since the last call (each call starts a new window).
pub fn take_stats() -> RenderWorkerStats {
    RenderWorkerStats {
        renders: RENDERS.swap(0, Ordering::Relaxed),
        render_us: RENDER_US.swap(0, Ordering::Relaxed),
        render_max_us: RENDER_MAX_US.swap(0, Ordering::Relaxed),
        owed: OWED_COUNT.swap(0, Ordering::Relaxed),
        skipped: SKIPPED.swap(0, Ordering::Relaxed),
        withdrawn: WITHDRAWN.swap(0, Ordering::Relaxed),
        lent: LENT_COUNT.swap(0, Ordering::Relaxed),
        lent_paid: LENT_PAID.swap(0, Ordering::Relaxed),
        extracts: EXTRACTS.swap(0, Ordering::Relaxed),
        wait_us: WAIT_US.swap(0, Ordering::Relaxed),
        wait_max_us: WAIT_MAX_US.swap(0, Ordering::Relaxed),
    }
}

fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// `?rwtrace=1`: a timeline of the handshake, one clock for both agents (`timeOrigin + now()`,
/// µs), in a ring the page reads with [`trace`]. Off, a stamp is one relaxed load.
static TRACE_ON: AtomicBool = AtomicBool::new(false);
const TRACE_LEN: usize = 8192;
static TRACE: [AtomicU64; TRACE_LEN] = [const { AtomicU64::new(0) }; TRACE_LEN];
static TRACE_NEXT: AtomicU32 = AtomicU32::new(0);

/// What a trace entry marks. The page's own: an animation frame (`web/threads.js`), the runner's
/// `before_update`, a hand-over tried (with the state it found), done, withdrawn, an extract owed,
/// an update skipped. The worker's: a wake, a claim, render start and end, a settle.
pub mod mark {
    pub const PAGE_FRAME: u8 = 1;
    pub const BEFORE_UPDATE: u8 = 2;
    pub const HAND_TRY: u8 = 3;
    pub const HAND_DONE: u8 = 4;
    pub const WITHDRAWN: u8 = 5;
    pub const OWED: u8 = 6;
    pub const SKIPPED: u8 = 7;
    pub const WORKER_POLL: u8 = 10;
    pub const CLAIM: u8 = 11;
    pub const RENDER_START: u8 = 12;
    pub const RENDER_END: u8 = 13;
    pub const SETTLE: u8 = 14;
    pub const LEND: u8 = 15;
    pub const RECLAIM: u8 = 16;
    pub const TAKE_LENT: u8 = 17;
}

/// Turn the timeline on (page, before the worker starts).
pub fn set_trace(on: bool) {
    TRACE_ON.store(on, Ordering::Release);
}

thread_local! {
    /// This agent's `performance` and its `timeOrigin`, ms.
    static CLOCK: Option<(js_sys::Function, JsValue, f64)> = {
        let perf = js_sys::Reflect::get(&js_sys::global(), &"performance".into()).ok();
        perf.and_then(|perf| {
            let now = js_sys::Reflect::get(&perf, &"now".into()).ok()?;
            let origin = js_sys::Reflect::get(&perf, &"timeOrigin".into()).ok()?.as_f64()?;
            Some((now.into(), perf, origin))
        })
    };
}

/// Stamp `kind` (with an 8-bit `arg`) at this instant, if the timeline is on.
pub fn stamp(kind: u8, arg: u8) {
    if !TRACE_ON.load(Ordering::Relaxed) {
        return;
    }
    let Some(ms) = CLOCK.with(|c| {
        let (now, perf, origin) = c.as_ref()?;
        Some(origin + now.call0(perf).ok()?.as_f64()?)
    }) else {
        return;
    };
    let us = (ms * 1000.0) as u64 & ((1 << 48) - 1);
    let slot = TRACE_NEXT.fetch_add(1, Ordering::Relaxed) as usize % TRACE_LEN;
    TRACE[slot].store(
        (u64::from(kind) << 56) | (u64::from(arg) << 48) | us,
        Ordering::Relaxed,
    );
}

/// The timeline, oldest first, as `kind arg µs` lines, and the ring emptied.
pub fn trace() -> String {
    let next = TRACE_NEXT.swap(0, Ordering::AcqRel) as usize;
    let (start, len) = if next > TRACE_LEN {
        (next % TRACE_LEN, TRACE_LEN)
    } else {
        (0, next)
    };
    let mut out = String::new();
    for i in 0..len {
        let v = TRACE[(start + i) % TRACE_LEN].swap(0, Ordering::Relaxed);
        if v != 0 {
            out.push_str(&alloc::format!(
                "{} {} {}\n",
                v >> 56,
                (v >> 48) & 0xff,
                v & ((1 << 48) - 1)
            ));
        }
    }
    out
}

/// The page's half of the frame handshake, as the extract function of [`RenderWorkerExtractApp`].
///
/// Never waits for a render or for the worker's event loop: if the extract does not happen now
/// ([`try_hand_over`]), it is owed, and the page lends its world to the worker when it returns to
/// the browser ([`lend`]).
fn renderer_extract(main_world: &mut World, _: &mut World) {
    if !try_hand_over(main_world) {
        stamp(mark::OWED, 0);
        OWED.store(true, Ordering::Relaxed);
        OWED_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether the last main update's extract is still owed to the worker. Page only: work that must
/// follow that extract (frame generation's run-ahead `Main`) waits for the next update.
pub fn extract_owed() -> bool {
    OWED.load(Ordering::Relaxed)
}

/// Whether the runner may start the next main update: `bevy_winit`'s `set_may_update` hook, on the
/// page. An extract the last update owed, and the worker did not take while the page was away
/// ([`reclaim`]), is paid first, against the world and messages that update left; if the worker
/// cannot take it yet, the runner skips this update and asks for the next frame.
pub fn before_update(world: &mut World) -> bool {
    stamp(mark::BEFORE_UPDATE, OWED.load(Ordering::Relaxed) as u8);
    if !OWED.load(Ordering::Relaxed) {
        return true;
    }
    if try_hand_over(world) {
        OWED.store(false, Ordering::Relaxed);
        return true;
    }
    SKIPPED.fetch_add(1, Ordering::Relaxed);
    stamp(mark::SKIPPED, 0);
    false
}

/// Lend the main world to the worker for one extract, if it is free to take it now: publish it,
/// wake the worker, and wait at most [`CLAIM_BUDGET`] for its claim. Claimed, spin until the
/// extract is done and return `true`. Unclaimed, withdraw it and return `false`.
fn try_hand_over(main_world: &mut World) -> bool {
    let state = STATE.load(Ordering::Acquire);
    stamp(mark::HAND_TRY, state as u8);
    if state != IDLE {
        return false;
    }
    let start = Instant::now();
    MAIN_WORLD.store(main_world, Ordering::Release);
    STATE.store(EXTRACT, Ordering::Release);
    kick();
    let mut turns = 0u32;
    while STATE.load(Ordering::Acquire) == EXTRACT {
        core::hint::spin_loop();
        turns = turns.wrapping_add(1);
        if turns.is_multiple_of(1024)
            && start.elapsed() > CLAIM_BUDGET
            && STATE
                .compare_exchange(EXTRACT, IDLE, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            MAIN_WORLD.store(core::ptr::null_mut(), Ordering::Release);
            WITHDRAWN.fetch_add(1, Ordering::Relaxed);
            stamp(mark::WITHDRAWN, 0);
            return false;
        }
    }
    // Claimed: the worker is inside `poll`, which needs nothing from this thread.
    spin_until(
        || STATE.load(Ordering::Acquire) != TAKEN,
        EXTRACT_DEADLINE,
        "the worker stopped answering an extract",
    );
    MAIN_WORLD.store(core::ptr::null_mut(), Ordering::Release);
    let waited = micros(start.elapsed());
    stamp(mark::HAND_DONE, 0);
    EXTRACTS.fetch_add(1, Ordering::Relaxed);
    WAIT_US.fetch_add(waited, Ordering::Relaxed);
    WAIT_MAX_US.fetch_max(waited, Ordering::Relaxed);
    true
}

/// Nothing lent.
const LENT_NONE: u32 = 0;
/// The page is waiting in the browser's event loop with its world parked in [`MAIN_WORLD`] and an
/// extract owed; the worker takes it when it is free.
const LENT: u32 = 1;
/// The worker took the lent world and is extracting it.
const LENT_TAKEN: u32 = 2;
/// The worker has extracted the lent world; the page's next callback finds the extract paid.
const LENT_DONE: u32 = 3;
static LEND: AtomicU32 = AtomicU32::new(LENT_NONE);

/// The runner returns to the browser (`bevy_winit`'s `set_lend_world` hook, after each of its
/// callbacks): an extract still owed is paid now if the worker is free, else the world is lent to
/// the worker, which extracts it as soon as its render is done ([`settle`]) while the page waits in
/// its event loop. The page touches nothing of the world until [`reclaim`].
///
/// Without it, an update that ended while the worker was still drawing the last one held the next
/// update for a whole frame and paid at that frame's start, so the worker drew alongside the page's
/// next update and that update's hand-over raced it again (measured: a third of updates owed).
pub fn lend(world: &mut World) {
    if !OWED.load(Ordering::Relaxed) {
        return;
    }
    if try_hand_over(world) {
        OWED.store(false, Ordering::Relaxed);
        return;
    }
    MAIN_WORLD.store(world, Ordering::SeqCst);
    LEND.store(LENT, Ordering::SeqCst);
    LENT_COUNT.fetch_add(1, Ordering::Relaxed);
    stamp(mark::LEND, STATE.load(Ordering::Relaxed) as u8);
    // A worker that went idle before it could see the loan has nothing left to wake it: the loan's
    // store and this load pair with `settle`'s, so one of the two sees the other.
    if STATE.load(Ordering::SeqCst) == IDLE {
        kick();
    }
}

/// The runner is back from the browser (`bevy_winit`'s `set_lend_world` hook, before each of its
/// callbacks): take back a world [`lend`] left with the worker. Untaken, it is withdrawn and the
/// extract stays owed; taken, the worker is inside its task extracting it, which needs nothing from
/// this thread, so the page waits for that to finish.
pub fn reclaim() {
    match LEND.compare_exchange(LENT, LENT_NONE, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => {
            MAIN_WORLD.store(core::ptr::null_mut(), Ordering::Release);
            stamp(mark::RECLAIM, 0);
        }
        Err(LENT_NONE) => {}
        Err(_) => {
            let start = Instant::now();
            spin_until(
                || LEND.load(Ordering::Acquire) == LENT_DONE,
                EXTRACT_DEADLINE,
                "the worker stopped answering a lent extract",
            );
            LEND.store(LENT_NONE, Ordering::Release);
            MAIN_WORLD.store(core::ptr::null_mut(), Ordering::Release);
            OWED.store(false, Ordering::Relaxed);
            let waited = micros(start.elapsed());
            stamp(mark::RECLAIM, 1);
            EXTRACTS.fetch_add(1, Ordering::Relaxed);
            LENT_PAID.fetch_add(1, Ordering::Relaxed);
            WAIT_US.fetch_add(waited, Ordering::Relaxed);
            WAIT_MAX_US.fetch_max(waited, Ordering::Relaxed);
        }
    }
}

/// Extract the main world into the render app and render it. `extracted` runs once the page may
/// have its world back, before the render.
fn extract_and_render(main_world: &mut World, extracted: impl FnOnce()) {
    RENDER_APP.with_borrow_mut(|app| {
        let app = app
            .as_mut()
            .expect("render worker: an extract arrived before the render app");
        at(AT_EXTRACT);
        app.extract(main_world);
        at(AT_WRITES);
        // The page's writes from the update this extract closes, before the page moves on.
        if let Some(queue) = app.world().get_resource::<RenderQueue>() {
            drain_writes(queue);
        }
        STATE.store(RENDERING, Ordering::Release);
        extracted();
        stamp(mark::RENDER_START, 0);

        let start = Instant::now();
        at(AT_RENDER);
        app.update();
        at(AT_GRAVE);
        drain_graveyard();
        let took = micros(start.elapsed());
        RENDERS.fetch_add(1, Ordering::Relaxed);
        RENDER_US.fetch_add(took, Ordering::Relaxed);
        RENDER_MAX_US.fetch_max(took, Ordering::Relaxed);
        STATE.store(SETTLING, Ordering::Release);
        stamp(mark::RENDER_END, 0);
    });
}

/// Take and render a world the page lent ([`lend`]), if the worker is free and there is one.
/// Whether it rendered.
fn take_lent() -> bool {
    if STATE.load(Ordering::SeqCst) != IDLE
        || LEND
            .compare_exchange(LENT, LENT_TAKEN, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
    {
        return false;
    }
    stamp(mark::TAKE_LENT, 0);
    STATE.store(TAKEN, Ordering::Release);
    // SAFETY: the page stored this pointer from its `&mut World` before it lent it and returned to
    // the browser; from the claim above until `LENT_DONE` it does not touch the world (its next
    // callback waits in `reclaim`).
    let main_world = unsafe { &mut *MAIN_WORLD.load(Ordering::Acquire) };
    extract_and_render(main_world, || LEND.store(LENT_DONE, Ordering::Release));
    true
}

/// The worker's whole loop body, run each time its event loop wakes: boot jobs first, then an
/// extract and a render if the page published one or lent its world. Returns whether it rendered,
/// in which case the worker's host must call [`settle`] from a later task.
pub fn poll() -> bool {
    stamp(mark::WORKER_POLL, STATE.load(Ordering::Relaxed) as u8);
    at(AT_JOBS);
    loop {
        let job = lock(&JOBS).pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    let rendered = if STATE
        .compare_exchange(EXTRACT, TAKEN, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        stamp(mark::CLAIM, 0);
        // SAFETY: the page stored this pointer from its `&mut World`, and once the claim above
        // succeeded it spins, touching nothing, until `STATE` leaves `TAKEN`.
        let main_world = unsafe { &mut *MAIN_WORLD.load(Ordering::Acquire) };
        extract_and_render(main_world, || {});
        true
    } else {
        take_lent()
    };
    if !rendered {
        at(AT_GRAVE);
        drain_graveyard();
    }
    at(AT_LOOP);
    rendered
}

/// The worker is free again: called by its host from the first task after a [`poll`] (or a
/// `settle`) that rendered, once the browser has had the worker's event loop back (to present the
/// frame). A world the page lent meanwhile is taken and rendered here; then, as after a poll, it
/// returns `true` and the host settles again from its next task.
pub fn settle() -> bool {
    stamp(mark::SETTLE, STATE.load(Ordering::Relaxed) as u8);
    let _ = STATE.compare_exchange(SETTLING, IDLE, Ordering::SeqCst, Ordering::SeqCst);
    let rendered = take_lent();
    at(AT_LOOP);
    rendered
}

/// The page-side stand-in for the render sub-app: no schedules, only the handshake as its extract.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AppLabel)]
pub struct RenderWorkerExtractApp;

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
