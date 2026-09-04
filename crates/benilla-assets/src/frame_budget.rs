//! A per-frame CPU budget for asset work, and the reason the browser build needs one.
//!
//! **On the web there is exactly one thread.** Bevy's `IoTaskPool` compiles to
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
pub use wasm::{begin_frame, set_budget_ms, slot, slot_in, Slot};

#[cfg(not(target_arch = "wasm32"))]
pub use native::{begin_frame, set_budget_ms, slot, slot_in, Slot};

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
