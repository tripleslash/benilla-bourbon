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

#[cfg(target_arch = "wasm32")]
pub use wasm::{begin_frame, set_budget_ms, slot, Slot};

#[cfg(not(target_arch = "wasm32"))]
pub use native::{begin_frame, set_budget_ms, slot, Slot};

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

    // Single-threaded target, so a `thread_local` is just a global with the right lifetime — the
    // same reasoning that puts the pack bridge's `JsValue`s in one
    // (`benilla_formats::webpack::resident`).
    thread_local! {
        static STATE: RefCell<State> = RefCell::new(State {
            frame: 0,
            spent_ms: 0.0,
            budget_ms: super::PLAY_BUDGET_MS,
            waiting: Vec::new(),
        });
    }

    struct State {
        /// Bumped by [`begin_frame`]. A waiter records the frame it parked in and is done as soon as
        /// this moves past it, which is what makes "next frame" mean the next one rather than "some
        /// frame after a wake that may already have happened".
        frame: u64,
        spent_ms: f64,
        budget_ms: f64,
        waiting: Vec<Waker>,
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

    /// Wait until this frame can afford some CPU work, then take the meter.
    ///
    /// At most one wait: a frame that has just begun has spent nothing, so the check that follows a
    /// wake always passes and the loop cannot spin. That is also what guarantees forward progress
    /// under any budget — the first slot of a frame is granted however expensive it turns out to be.
    pub async fn slot() -> Slot {
        loop {
            let over = STATE.with(|s| {
                let s = s.borrow();
                s.spent_ms >= s.budget_ms
            });
            if !over {
                return Slot {
                    started: Instant::now(),
                };
            }
            NextFrame::park().await;
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
            std::mem::take(&mut s.waiting)
        });
        for waker in woken {
            waker.wake();
        }
    }

    /// Set what a frame may spend from here on — [`super::LOADING_BUDGET_MS`] while a loading screen
    /// is up, [`super::PLAY_BUDGET_MS`] once there is a world to draw.
    pub fn set_budget_ms(ms: f64) {
        STATE.with(|s| s.borrow_mut().budget_ms = ms.max(0.0));
    }

    /// Resolves on the first [`begin_frame`] after it was polled.
    struct NextFrame {
        parked_in: Option<u64>,
    }

    impl NextFrame {
        fn park() -> Self {
            Self { parked_in: None }
        }
    }

    impl Future for NextFrame {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            STATE.with(|s| {
                let mut s = s.borrow_mut();
                match self.parked_in {
                    Some(frame) if s.frame != frame => Poll::Ready(()),
                    _ => {
                        self.parked_in = Some(s.frame);
                        // Re-registered on every poll rather than once: a waker handed out on an
                        // earlier poll may belong to a task the executor has since moved.
                        s.waiting.push(cx.waker().clone());
                        Poll::Pending
                    }
                }
            })
        }
    }
}
