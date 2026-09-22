//! A per-frame CPU budget for asset work, and the reason the browser build needs one.
//!
//! **On the web an asset loader runs on the thread that draws.** Bevy's `IoTaskPool` compiles to
//! `wasm_bindgen_futures::spawn_local` there (`bevy_tasks::Task::wrap_future`), so an
//! [`AssetLoader`](bevy::asset::AssetLoader) is not "off-thread work" at all — it is a chain of
//! microtasks on the thread that draws, and the stretch between two `await`s is a stretch the frame
//! cannot interrupt. That is fine for one loader. It is not fine for the shape this client actually
//! produces: a camera turn resolves a doodad set, thirty fetches settle within a few milliseconds of
//! each other, and the browser then drains **thirty decodes back to back** out of the microtask
//! queue with no paint in between. Nothing is slow; the frame is simply never given back. That is
//! the micro-stutter.
//!
//! So decoding is metered rather than throttled. A loader takes a [`slot`] before its CPU-bound
//! half; the slot is granted immediately while this frame has budget left, and otherwise waits for
//! the next one. The guard charges what the work actually took, so no loader has to estimate its own
//! cost and a texture that turns out to be 8 MB pays 8 MB's worth.
//!
//! Two properties are deliberate:
//!
//! - **The first slot of a frame is always granted**, whatever it costs. A budget that could refuse
//!   every slot would deadlock loading; this one can only overshoot by one item, and the overshoot
//!   is bounded by the largest single asset rather than by the length of the queue.
//! - **Loading is not slowed down where nothing is drawing.** A loading screen wants the whole frame,
//!   so it raises the budget ([`set_budget_ms`]) rather than routing around this module.
//!
//! ## Two lanes, because not every decode is worth the same
//!
//! A frame's budget used to be first-come-first-served, and at a zone boundary the queue that
//! arrives first is the art: one terrain tile's decode competes with the several hundred `.blp`s of
//! the buildings standing on it. Whoever polls first wins, and the tile — the thing that decides
//! whether the ground under the player has a collider — can lose that race for a dozen frames.
//!
//! So a slot names its [`Lane`]. [`Lane::Geometry`] is the work that produces **collision geometry
//! or model bounds** (the ADT tile decode, the WMO root and its groups, the M2 header and hull);
//! [`Lane::Art`] is everything else. Geometry waiters are woken first each frame, and an art waiter
//! yields its turn while geometry is backed up — for at most [`ART_YIELD_FRAMES`] frames, so this
//! is a head start and never a starvation. Nothing about the *budget* changes: the same
//! milliseconds are spent, in a better order.
//!
//! ## The browser has threads now, and a loader still is not on one
//!
//! Since `scripts/web-build-threads.sh` the browser build has real worker threads, so "the web
//! target has one thread" is no longer the flat fact this module was written under. It is still
//! the fact *here*, and for a reason no thread count changes: `TaskPool::spawn` on wasm is the
//! microtask queue by deliberate design, because a load awaits `JsFuture`s and reaches `window`
//! (`benilla_formats::webpack::resident`), and a `JsValue` cannot leave the agent that made it.
//! `bevy_asset` spawns every load that way. So the metering below is exactly as load-bearing as
//! it was — what changed is that the CPU-bound *interior* of a decode, which is plain bytes in
//! linear memory, could now be handed to a worker with `TaskPool::spawn_pooled` and leave the
//! frame entirely. Whether it may is [`pooled_work_runs`]; where it has already happened is
//! `terrain_stream::collider`.
//!
//! **Native builds get none of it.** There the pool is real threads, decoding already runs beside
//! the frame rather than inside it, and a shared counter would only serialise work that is currently
//! parallel — so every item here compiles to nothing on native and the loaders read the same either
//! way.

use bevy::prelude::*;

/// What a frame will spend on asset CPU work before the rest waits for the next one.
///
/// 4 ms against a 16.6 ms frame: a sixth of the budget, which is what a client with a scene to draw
/// can give away without the frame time moving, and enough for a couple of ordinary BLPs.
pub const PLAY_BUDGET_MS: f64 = 4.0;

/// The budget while a loading screen is up — effectively "no limit". Nothing is animating but the
/// screen itself, so the frame has nothing better to do and the player is waiting on exactly this
/// work.
pub const LOADING_BUDGET_MS: f64 = 250.0;

/// How many consecutive frames an [`Lane::Art`] slot will stand aside for backed-up geometry
/// before taking its turn anyway.
///
/// The bound is the whole point. Without one, a world that streams geometry continuously would
/// never decode a texture, and "collision first" would quietly become "art never" — a black city
/// with a floor. Two frames is enough for a tile decode to get ahead of the art of the buildings on
/// it (the case this exists for) and is imperceptible as a delay to the art itself.
pub const ART_YIELD_FRAMES: u32 = 2;

/// Which kind of work a slot is for — see the module doc's "Two lanes".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lane {
    /// Collision geometry and model bounds: the ADT tile decode, WMO roots + groups, M2 headers and
    /// hulls. Woken first each frame; never yields.
    Geometry,
    /// Everything else — textures, the terrain layer array, UI sprites. Stands aside for backed-up
    /// geometry, boundedly ([`ART_YIELD_FRAMES`]).
    Art,
}

#[cfg(target_arch = "wasm32")]
pub use wasm::{begin_frame, set_budget_ms, slot, slot_in, spent_ms, Slot};

#[cfg(not(target_arch = "wasm32"))]
pub use native::{begin_frame, set_budget_ms, slot, slot_in, spent_ms, Slot};

/// Will work handed to `TaskPool::spawn_pooled` actually run — or is this a build where it would
/// sit in the queue forever?
///
/// **The web build has nothing to drive an idle pool.** `bevy_app`'s `TaskPoolPlugin` gates its
/// `tick_global_task_pools` system on `not(all(target_arch = "wasm32", feature = "web"))`, so the
/// browser never schedules it; `TaskPool::scope` covers itself (`third_party/bevy_tasks` forces a
/// tick when `threads.is_empty()`); and nothing in this tree calls `TaskPool::try_tick`. So a
/// pooled task on a page whose workers never started has neither a thread to run it nor a ticker
/// to poll it — and it is not slow, it never completes at all.
///
/// That page is a supported configuration rather than a corner: `?nothreads=1`, an origin that is
/// not cross-origin isolated (no `SharedArrayBuffer`, so no shared memory and no workers), or a
/// browser that simply refuses a `Worker`. `web/threads.js` announces each of them, and
/// `third_party/bevy_tasks/BENILLA.md` calls the resulting zero-worker pool "the designed
/// fallback" — this predicate is what makes it one for `spawn_pooled` too. A caller that gets
/// `false` has to do the work the way the single-threaded build always did: on the frame thread,
/// under a metered [`slot`].
///
/// **Native is unconditionally `true`** and never consults the pool. There a refused thread is
/// still a panic (the fork keeps upstream's `expect` everywhere `std::thread` is real), so a
/// native pool that came up empty is not a state this could report — it is one that already
/// aborted. Asking `thread_num()` there could only invent a native behaviour that does not exist
/// today, and these forks hold to native not moving.
pub fn pooled_work_runs() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    return true;

    // The kill switch, web-only and checked first. See [`set_offload_enabled`].
    #[cfg(target_arch = "wasm32")]
    if !OFFLOAD.load(core::sync::atomic::Ordering::Relaxed) {
        return false;
    }

    // The async-compute pool specifically, because it is the one `spawn_pooled` work goes to on
    // this target (the collider builds, and whatever later joins them). `ComputeTaskPool` is
    // reached through `scope`, which already drives itself at zero workers.
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    return bevy::tasks::AsyncComputeTaskPool::try_get().is_some_and(|p| p.thread_num() > 0);

    // **A wasm build without atomics has no `spawn_pooled` to ask about**, so the answer is a
    // constant rather than a reading. `bevy_tasks` swaps in `single_threaded_task_pool.rs` on this
    // target and that `TaskPool` does not define the method at all — a caller reaching it here
    // would not be slow, it would not compile.
    //
    // It also must not be asked the obvious way, which is the trap this arm exists to close:
    // `single_threaded_task_pool`'s `thread_num()` returns **1**, meaning "the calling thread",
    // not "one worker". A `thread_num() > 0` written without this `cfg` is therefore `true` here
    // — the one answer that is certainly wrong — and it is wrong silently, in the same shape as
    // the `feature = "std"` trap `third_party/bevy_tasks/BENILLA.md` documents.
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    return false;
}

/// The offload kill switch. `true` until a host says otherwise.
#[cfg(target_arch = "wasm32")]
static OFFLOAD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// Turn worker offload off (or back on) for the whole client — `?offload=0` on the page.
///
/// **Why a switch exists for this and not for the rest of the module, and why it earned its keep
/// on the first day.** Every other fallback here is exercised by an ordinary browser: the metered
/// path is what a page without shared memory runs, so it cannot rot unnoticed. The *offloaded*
/// path was the new one, and the first cut of it shipped broken — [`offload`] `.await`ed the
/// pooled `Task`, which makes a worker responsible for waking a future on the page's own thread,
/// and nothing an `AssetLoader` produced ever arrived. `?offload=0` is what a live client was
/// steered onto while that was diagnosed, with no re-upload: it decodes exactly the way it did
/// before any of this, on the frame thread under the budget. The lever stays for the next such
/// day. (The mechanism itself is fixed — [`offload`] polls now, the way
/// `terrain_stream::collider` always did.)
///
/// Off on this switch also parks the collider builds, which is the honest meaning of the lever —
/// it is "no `spawn_pooled` at all", the one state the whole module is already written for.
#[cfg(target_arch = "wasm32")]
pub fn set_offload_enabled(on: bool) {
    OFFLOAD.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Native: nothing to switch, and saying so keeps the call site one text on both targets.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_offload_enabled(_on: bool) {}

/// Run one CPU-bound, self-contained piece of work **off the frame thread if this build can**, and
/// under the frame budget if it cannot.
///
/// This is the whole of the offload story for the loaders, in one call, because every decision it
/// makes is a property of the target rather than of the caller:
///
/// - **Native** runs `work` inline and consults nothing. A loader there is already on an
///   `IoTaskPool` thread, so the frame is not paying for it and a hop to a second pool would only
///   add latency and a channel. This is the same reasoning that makes [`slot`] a no-op on native,
///   and it keeps this function from being a native behaviour change.
/// - **A browser with workers** hands `work` to one and awaits the result. The `Send + 'static`
///   bounds are what make that safe rather than merely intended: a closure that touched a
///   `JsValue`, the DOM or a `LoadContext` does not satisfy them and does not compile, which is
///   exactly the wall that has to exist here (`third_party/bevy_tasks/BENILLA.md` on the
///   `spawn`/`spawn_pooled` split). The await is not a block — the returned `Task` is an
///   `async_channel` receiver, and the worker's send wakes this future through the `+atomics`
///   `js-sys` task queue.
/// - **A browser without them** takes a metered [`slot_in`] and runs `work` on the frame thread:
///   the pre-threads behaviour, unchanged, and the reason `lane` is still a parameter.
///
/// **`lane` matters only in the last case**, and the reason is worth stating rather than glossing:
/// an offloaded decode is not competing for the frame's budget, so there is no budget order for a
/// lane to express. Pass the lane the work would have taken and the fallback stays exactly as
/// correct as it was.
///
/// What that costs is honest to name. The lanes exist because at a zone boundary one tile's decode
/// must not queue behind the several hundred `.blp`s of the buildings standing on it — the tile is
/// where the collider under the player comes from. Offloaded, that ordering is gone: the pool's
/// executor is FIFO and a `Lane::Geometry` closure takes its turn behind whatever was spawned
/// first. The trade is a good one and it is not a wash — the same work now runs on every core
/// instead of 4 ms of one, so the tile lands sooner in wall-clock terms even from further back in
/// the queue, and `terrain_stream::provisional`'s stand-ins hold the body up meanwhile exactly as
/// they always did. But it IS a change of ordering guarantee, not just of venue, and if a
/// stand-on-nothing report ever comes back from a threaded page this paragraph is the first thing
/// to re-read.
#[cfg(not(target_arch = "wasm32"))]
pub async fn offload<T, F>(_lane: Lane, work: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    work()
}

/// See the native definition above for the contract; this is the browser half.
#[cfg(target_arch = "wasm32")]
pub async fn offload<T, F>(lane: Lane, work: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // `cfg`, not just the runtime gate: `spawn_pooled` does not exist on a wasm build without
    // atomics (`bevy_tasks` swaps in `single_threaded_task_pool.rs`), so this call has to be
    // compiled away rather than merely skipped. `pooled_work_runs` answers `false` there too.
    #[cfg(target_feature = "atomics")]
    if pooled_work_runs() {
        // **Polled once a frame, not awaited, and the difference is the whole bug this shape
        // exists to avoid.** Awaiting the `Task` makes the worker's `send` responsible for waking
        // a future that lives on the page's thread — `js-sys`'s `+atomics` task queue and
        // `Atomics.waitAsync`. Shipped that way, nothing an `AssetLoader` produced ever arrived.
        //
        // `terrain_stream::collider` had the answer already and had had it all along: it polls
        // its `Task` from a system every frame and never waits on a waker, which is why the
        // collider offload worked while this did not. Same shape here. `poll_once` still
        // registers the cross-thread waker, so a wake that DOES land is taken immediately and the
        // frame tick is only the floor; a wake that does not land costs one frame and self-heals,
        // which is the property the await did not have. A decode is not on this frame's critical
        // path — that is the premise of the whole module — so a frame of latency buys nothing
        // back and is not worth a cleverer wait.
        let pool = bevy::tasks::AsyncComputeTaskPool::get();
        let _permit = in_flight::acquire(lane, pool.thread_num()).await;
        let mut task = pool.spawn_pooled(async move { work() });
        loop {
            if let Some(done) =
                bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(&mut task))
            {
                return done;
            }
            wasm::next_frame().await;
        }
    }
    let _slot = slot_in(lane).await;
    work()
}

/// How many [`offload`]s may be handed to the pool at once, and the permit that counts one.
///
/// **Without a bound, a streaming burst aborted every worker in the tab.** `offload` used to spawn
/// the moment a loader reached it, so world entry — or a fast run across the map — put several
/// hundred decodes on the pool together. Every one of them lives in the one shared linear memory:
/// its captured input while it waits, the full decoded output (a 1024² BLP is ~5.6 MB of RGBA with
/// mips) once a worker has run it, and that output stays put until the page's next poll collects
/// it. Eight workers producing faster than the page drained took the arena to its `--max-memory`
/// ceiling, and the next `Vec` to grow on each worker hit `rust_oom` — a bare
/// `RuntimeError: unreachable` per thread, no message (`web/threads.js` has why).
///
/// So a loader now waits for a permit before spawning. [`PER_WORKER`] per worker keeps every worker
/// fed through the frame of latency a completion takes to be collected, and bounds the decoded
/// output in flight to a few dozen assets instead of the whole backlog. The inputs of loaders
/// waiting here still sit in memory — they were read before `offload` was called — but those are
/// the compressed bytes, a fraction of what a decode turns them into.
///
/// **Geometry keeps a reserve.** Art may take at most `cap - workers` permits, so however deep the
/// texture backlog, a tile, WMO or M2 decode — the collider under the player — always finds one of
/// the last `workers` free. That is the lane guarantee of this module carried over to the pool,
/// which as the `offload` doc says is otherwise FIFO.
///
/// A counter rather than a queue: every waiter re-checks on the next frame (the same tick the
/// finished tasks are collected on, and what frees their permits), so there is no wake list to keep
/// and nothing to leak if a loader is dropped mid-wait — the permit is a guard, and a dropped
/// loader drops its `Task` and its permit together.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod in_flight {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::Lane;

    /// Permits per pool worker. Two would do if a finished task were collected the instant it
    /// finished; it is collected on the page's next frame, so a worker that finishes early in a
    /// frame needs more queued behind it to stay busy until then.
    pub const PER_WORKER: usize = 4;

    static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

    pub struct Permit;

    impl Drop for Permit {
        fn drop(&mut self) {
            IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Wait (a frame at a time) until `lane` may have a permit with `workers` pool threads.
    pub async fn acquire(lane: Lane, workers: usize) -> Permit {
        let workers = workers.max(1);
        let cap = match lane {
            Lane::Geometry => workers * PER_WORKER,
            Lane::Art => workers * (PER_WORKER - 1),
        };
        loop {
            // A CAS loop rather than `fetch_update`, which nightly — the threads build's toolchain —
            // deprecates under a name stable does not have yet.
            let mut n = IN_FLIGHT.load(Ordering::Acquire);
            while n < cap {
                match IN_FLIGHT.compare_exchange_weak(n, n + 1, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => return Permit,
                    Err(now) => n = now,
                }
            }
            super::wasm::next_frame().await;
        }
    }

    /// Permits held right now — what `benilla_app::webdiag` can print beside linear memory.
    pub fn held() -> usize {
        IN_FLIGHT.load(Ordering::Relaxed)
    }
}

/// Offloaded decodes handed to the pool and not yet collected by the page. Zero on every build
/// that does not offload.
pub fn offloads_in_flight() -> usize {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    return in_flight::held();
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    return 0;
}

/// Resets the budget once per frame, in [`First`] — before any system that could raise it for a
/// loading screen, so such a raise applies to the frame that asked for it.
///
/// On native the system is still registered and still costs nothing, which keeps the schedule
/// identical across targets.
pub fn plugin(app: &mut App) {
    app.add_systems(First, tick);
}

fn tick() {
    begin_frame();
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    /// The guard [`slot`] hands back. Empty here: nothing is metered on a target whose loaders run
    /// on their own threads.
    pub struct Slot;

    /// Immediately granted, always. Kept `async` so the call sites are one text on both targets.
    #[inline]
    pub async fn slot() -> Slot {
        Slot
    }

    /// [`slot`] with a lane, which native has no use for: nothing is metered here, so nothing can
    /// be ordered. Kept so the call sites are one text on both targets.
    #[inline]
    pub async fn slot_in(_lane: super::Lane) -> Slot {
        Slot
    }

    #[inline]
    pub fn begin_frame() {}

    #[inline]
    pub fn set_budget_ms(_ms: f64) {}

    /// Always zero: nothing here is metered, so no frame is ever charged. Kept so the reporters
    /// read the same text on both targets (`benilla_app::webdiag`).
    #[inline]
    pub fn spent_ms() -> f64 {
        0.0
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    use bevy::platform::time::Instant;

    use super::Lane;

    // Single-threaded target, so a `thread_local` is just a global with the right lifetime — the
    // same reasoning that puts the pack bridge's `JsValue`s in one
    // (`benilla_formats::webpack::resident`).
    thread_local! {
        static STATE: RefCell<State> = RefCell::new(State {
            frame: 0,
            spent_ms: 0.0,
            budget_ms: super::PLAY_BUDGET_MS,
            waiting: Vec::new(),
            geometry_parked: 0,
        });
    }

    struct State {
        /// Bumped by [`begin_frame`]. A waiter records the frame it parked in and is done as soon as
        /// this moves past it, which is what makes "next frame" mean the next one rather than "some
        /// frame after a wake that may already have happened".
        frame: u64,
        spent_ms: f64,
        budget_ms: f64,
        /// Parked waiters with the lane they parked in — [`begin_frame`] wakes geometry first.
        waiting: Vec<(Lane, Waker)>,
        /// How many [`Lane::Geometry`] waiters are parked right now. The signal an art slot reads to
        /// decide whether to stand aside: geometry only ever parks because the frame's budget ran
        /// out under it, so a non-zero count means geometry work is genuinely backed up rather than
        /// merely present.
        geometry_parked: usize,
    }

    /// Charges the frame for the work it guarded. Held across the CPU-bound half of a loader and
    /// dropped when that half is done — so the meter records what the decode *cost*, not what the
    /// loader guessed it would.
    ///
    /// Not to be held across an `await` on I/O: a slot held over a network wait would bill the frame
    /// for time the frame never spent, and starve the next asset for nothing.
    pub struct Slot {
        started: Instant,
    }

    impl Drop for Slot {
        fn drop(&mut self) {
            let ms = self.started.elapsed().as_secs_f64() * 1000.0;
            STATE.with(|s| s.borrow_mut().spent_ms += ms);
        }
    }

    /// Wait until this frame can afford some CPU work, then take the meter — [`Lane::Art`], the
    /// lane of everything that is not collision geometry or model bounds.
    pub async fn slot() -> Slot {
        slot_in(Lane::Art).await
    }

    /// [`slot`] in a named lane. See the module doc's "Two lanes".
    ///
    /// A geometry slot waits at most once: a frame that has just begun has spent nothing, so the
    /// check that follows a wake always passes and the loop cannot spin. That is what guarantees
    /// forward progress under any budget — the first geometry slot of a frame is granted however
    /// expensive it turns out to be.
    ///
    /// An art slot can wait up to [`super::ART_YIELD_FRAMES`] frames longer, standing aside while
    /// geometry is backed up. The yield counter lives on this call, not in the shared state, so the
    /// bound is per-await and no amount of geometry can starve a given decode.
    pub async fn slot_in(lane: Lane) -> Slot {
        let mut yielded = 0u32;
        loop {
            let wait = STATE.with(|s| {
                let s = s.borrow();
                if s.spent_ms >= s.budget_ms {
                    return true;
                }
                // The head start: this frame can still afford the work, but geometry is queued for
                // it and geometry is what the floor under the player is made of.
                lane == Lane::Art && s.geometry_parked > 0 && yielded < super::ART_YIELD_FRAMES
            });
            if !wait {
                return Slot {
                    started: Instant::now(),
                };
            }
            yielded += 1;
            NextFrame::park(lane).await;
        }
    }

    /// Open a new frame's budget and release everything that parked in the last one. Called from
    /// [`super::plugin`]'s `First` system.
    ///
    /// The woken continuations do not run here — `spawn_local`'s wakers queue microtasks, which the
    /// browser drains once this frame's animation-frame callback returns. So they land at the tail of
    /// the frame that woke them and are charged to it, which is the accounting this module wants.
    pub fn begin_frame() {
        let woken = STATE.with(|s| {
            let mut s = s.borrow_mut();
            s.frame = s.frame.wrapping_add(1);
            s.spent_ms = 0.0;
            s.geometry_parked = 0;
            std::mem::take(&mut s.waiting)
        });
        // **Geometry first.** `spawn_local`'s wakers queue microtasks, which the browser drains in
        // the order they were queued — so this loop's order IS the order the continuations run in,
        // and therefore the order they take the frame's budget in. Two passes over a list that is
        // empty on almost every frame.
        for (_, waker) in woken.iter().filter(|(lane, _)| *lane == Lane::Geometry) {
            waker.wake_by_ref();
        }
        for (_, waker) in woken.iter().filter(|(lane, _)| *lane != Lane::Geometry) {
            waker.wake_by_ref();
        }
    }

    /// Set what a frame may spend from here on — [`super::LOADING_BUDGET_MS`] while a loading screen
    /// is up, [`super::PLAY_BUDGET_MS`] once there is a world to draw.
    pub fn set_budget_ms(ms: f64) {
        STATE.with(|s| s.borrow_mut().budget_ms = ms.max(0.0));
    }

    /// What the current frame has been charged for decode so far — the running total [`Slot`]'s
    /// `Drop` accumulates, zeroed by [`begin_frame`].
    ///
    /// **Read it before `First`, or it reads zero.** The decodes this counts are microtask
    /// continuations, and the browser drains those only once the frame's animation-frame callback
    /// has returned — so a frame's charge lands *after* its own `Last` and sits in the counter
    /// until the next frame's [`begin_frame`] clears it. The one place that can sample it and
    /// name the right frame is therefore a reader that runs ahead of `First`, which is exactly
    /// where `benilla_app::webdiag` closes its tape.
    pub fn spent_ms() -> f64 {
        STATE.with(|s| s.borrow().spent_ms)
    }

    /// Resolve on the next [`begin_frame`] — the same-thread tick [`super::offload`] polls its
    /// worker `Task` on.
    ///
    /// Parks in [`Lane::Art`] deliberately: this is not a budget waiter and must not be counted as
    /// one. Only a `Lane::Geometry` park bumps `geometry_parked`, which is the signal an art slot
    /// reads to decide whether to stand aside — a poll tick registering there would make real
    /// decodes yield to work that is not competing for the frame at all.
    pub(super) async fn next_frame() {
        NextFrame::park(Lane::Art).await;
    }

    /// Resolves on the first [`begin_frame`] after it was polled, remembering which lane parked so
    /// the wake order can honour it.
    struct NextFrame {
        parked_in: Option<u64>,
        lane: Lane,
    }

    impl NextFrame {
        fn park(lane: Lane) -> Self {
            Self {
                parked_in: None,
                lane,
            }
        }
    }

    impl Future for NextFrame {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            let lane = self.lane;
            STATE.with(|s| {
                let mut s = s.borrow_mut();
                match self.parked_in {
                    Some(frame) if s.frame != frame => Poll::Ready(()),
                    _ => {
                        // Counted once per frame per waiter, not once per poll: `begin_frame` zeroes
                        // it, and a re-poll inside the same frame is the same waiter, not a second
                        // one. (An art slot reads this to decide whether to stand aside; double
                        // counting would make it stand aside for waiters that are not there.)
                        let first_park_this_frame = self.parked_in != Some(s.frame);
                        self.parked_in = Some(s.frame);
                        if lane == Lane::Geometry && first_park_this_frame {
                            s.geometry_parked += 1;
                        }
                        // Re-registered on every poll rather than once: a waker handed out on an
                        // earlier poll may belong to a task the executor has since moved.
                        s.waiting.push((lane, cx.waker().clone()));
                        Poll::Pending
                    }
                }
            })
        }
    }
}
