//! The frame-generation cadence: which frames render the world (W) and which only present (H).
//!
//! With frame generation on, every rendered world frame is followed by a held frame. On H the main
//! world runs no schedule at all — no game update, no extraction of anything new — and the render
//! world skips every camera but the ones that present ([`PresentsHeldFrames`]): the UI camera
//! grounds on the real frame the W before it produced, draws the UI over it, and presents. So every
//! main-world temporal state (particle and transform history, the froxel history, the jitter
//! sequence) advances once per rendered frame for free, and input is not lost: winit's messages
//! accumulate unrotated across the H frame, since `First` does not run.
//!
//! The main loop hook is `app.main_mut().update_schedule`, replaced by [`FgMain`], whose one
//! exclusive system returns at once on H and runs `Main` on W.

use bevy::core_pipeline::schedule::RootNonCameraView;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::ExtractComponent;
use bevy::window::PrimaryWindow;

use super::{FsrHeld, FsrView};

/// The main app's update schedule while frame generation can run: one exclusive system that
/// either holds the frame or runs `Main`.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct FgMain;

/// The empty schedule a held frame gives every camera that does not present.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct HeldFrame;

/// A camera that still draws on a held frame: the player-UI camera and the dev overlay.
/// Extracted, since [`hold_cameras`] reads it in the render world.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub struct PresentsHeldFrames;

/// Whether frame generation is wanted: the `frameGeneration` CVar, which the app writes.
#[derive(Resource, Clone, Copy, Default, Debug, PartialEq)]
pub struct FrameGeneration(pub bool);

/// Frames that must render: a capture run, or anything else that reads every frame's pixels. The
/// app sets it; while set, no frame is held.
#[derive(Resource, Clone, Copy, Default, Debug)]
pub struct FgBlocked(pub bool);

/// The cadence state machine. Pure, so its rules are unit-tested without an app.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct FgCadence {
    /// The last frame was held.
    held: bool,
    /// World frames rendered since the history last reset (FSR's `FrameIndex`); a frame is held
    /// only once the history it would interpolate from is valid.
    since_reset: u32,
    /// The window's physical size at the last world frame; a change forces a world frame.
    window: UVec2,
    /// The FSR epoch at the last world frame.
    epoch: u32,
}

/// What the cadence sees each frame.
#[derive(Clone, Copy, Debug)]
pub struct FgInputs {
    pub wanted: bool,
    pub in_world: bool,
    pub blocked: bool,
    pub window: UVec2,
    pub epoch: u32,
}

impl FgCadence {
    /// Decide this frame: `true` holds it. A held frame is only ever the one after a world frame
    /// that interpolated from valid history; a resize, a new epoch, a capture or leaving the world
    /// forces a world frame and restarts the count.
    pub fn next(&mut self, i: FgInputs) -> bool {
        let reset = i.window != self.window || i.epoch != self.epoch || !i.wanted || !i.in_world;
        if reset {
            self.since_reset = 0;
        }
        // The SDK interpolates only from the second rendered frame of a history on.
        let hold = !self.held && !reset && !i.blocked && self.since_reset >= 2;
        if !hold {
            self.since_reset = self.since_reset.saturating_add(1);
            self.window = i.window;
            self.epoch = i.epoch;
        }
        self.held = hold;
        hold
    }
}

/// `FgMain`'s one system: hold the frame, or run the game.
fn fg_main(world: &mut World) {
    let window = world
        .query_filtered::<&Window, With<PrimaryWindow>>()
        .iter(world)
        .next()
        .map_or(UVec2::ZERO, |w| {
            UVec2::new(w.physical_width(), w.physical_height())
        });
    let (in_world, epoch) = world
        .query::<&FsrView>()
        .iter(world)
        .next()
        .map_or((false, 0), |f| (true, f.epoch));
    let inputs = FgInputs {
        wanted: world.resource::<FrameGeneration>().0,
        in_world,
        blocked: world.resource::<FgBlocked>().0,
        window,
        epoch,
    };
    let held = world.resource_mut::<FgCadence>().next(inputs);
    // Written every frame, so extraction always carries this frame's verdict.
    world.resource_mut::<FsrHeld>().0 = held;
    if held {
        // `First`'s `time_system` does not run, so drain the render thread's frame instant here:
        // its channel holds one, and the next send would panic on a full one.
        if let Some(rx) = world.get_resource::<bevy::time::TimeReceiver>() {
            let _ = rx.0.try_recv();
        }
    } else {
        world.run_schedule(Main);
    }
}

/// Render world, just before the render graph: on a held frame every camera that does not present
/// runs the empty [`HeldFrame`] schedule instead of its own, and so does every auxiliary view (a
/// point or spot light's shadow map). Extraction re-inserts both next frame, so there is nothing
/// to restore.
pub(super) fn hold_cameras(
    held: Res<FsrHeld>,
    mut cameras: Query<&mut ExtractedCamera, Without<PresentsHeldFrames>>,
    mut auxiliary: Query<&mut RootNonCameraView>,
) {
    if !held.0 {
        return;
    }
    for mut camera in &mut cameras {
        camera.schedule = HeldFrame.intern();
    }
    for mut view in &mut auxiliary {
        view.0 = HeldFrame.intern();
    }
}

/// Frames presented since the FPS journal last read it, held ones included; the journal counts
/// rendered frames itself (one per main-world update).
pub static PRESENTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The present cadence since it was last taken ([`take_cadence`]): count, sum and sum of squares
/// of the intervals between presents (s), and the longest, so evenness is read from numbers.
static CADENCE: std::sync::Mutex<(u32, f64, f64, f64)> = std::sync::Mutex::new((0, 0.0, 0.0, 0.0));

/// `(mean interval ms, coefficient of variation, max interval ms)` since the last call.
pub fn take_cadence() -> Option<(f64, f64, f64)> {
    let (n, sum, sq, max) = std::mem::take(
        &mut *CADENCE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    (n > 1).then(|| {
        let mean = sum / f64::from(n);
        let var = (sq / f64::from(n) - mean * mean).max(0.0);
        (mean * 1000.0, var.sqrt() / mean, max * 1000.0)
    })
}

/// The world frames' present times, for [`pace_held_frames`].
#[derive(Resource, Default)]
pub(super) struct FgPacing {
    #[cfg(not(target_arch = "wasm32"))]
    last_world: Option<std::time::Instant>,
    /// The smoothed world-to-world interval, in seconds.
    #[cfg(not(target_arch = "wasm32"))]
    interval: f32,
}

/// Native, just before `render_system`: a held frame presents half a world interval after the
/// world frame before it, so the generated and the real frame split the interval evenly (the SDK's
/// pacer does the same on its present thread). The web presents on rAF, which paces it.
pub(super) fn pace_held_frames(
    held: Res<FsrHeld>,
    mut pacing: ResMut<FgPacing>,
    mut last_present: Local<Option<bevy::platform::time::Instant>>,
) {
    PRESENTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // After this system's own wait: the instant the frame goes to the graph, as near its present
    // as this thread sees.
    let _cadence = CadenceMark(&mut last_present);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let now = std::time::Instant::now();
        if !held.0 {
            if let Some(last) = pacing.last_world {
                let dt = now.duration_since(last).as_secs_f32().min(0.25);
                // A frame with no hold before it (a reset, frame generation off) is not a pair.
                pacing.interval = if pacing.interval == 0.0 {
                    dt
                } else {
                    pacing.interval * 0.9 + dt * 0.1
                };
            }
            pacing.last_world = Some(now);
            return;
        }
        if let Some(last) = pacing.last_world {
            let due = last + std::time::Duration::from_secs_f32(pacing.interval * 0.5);
            if due > now {
                std::thread::sleep(due - now);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (held, pacing);
}

/// Install the cadence: the main app's update schedule becomes [`FgMain`].
pub(super) fn install(app: &mut App) {
    app.init_resource::<FgCadence>()
        .init_resource::<FrameGeneration>()
        .init_resource::<FgBlocked>();
    let mut schedule = Schedule::new(FgMain);
    schedule.add_systems(fg_main);
    app.add_schedule(schedule);
    // `Main` stays the schedule the rest of the app knows; only the entry point moves.
    app.main_mut().update_schedule = Some(FgMain.intern());
}

/// Records the interval since the last present when dropped, after any pacing wait.
struct CadenceMark<'a>(&'a mut Option<bevy::platform::time::Instant>);

impl Drop for CadenceMark<'_> {
    fn drop(&mut self) {
        let now = bevy::platform::time::Instant::now();
        if let Some(last) = self.0.replace(now) {
            let dt = now.duration_since(last).as_secs_f64();
            let mut c = CADENCE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            c.0 += 1;
            c.1 += dt;
            c.2 += dt * dt;
            c.3 = c.3.max(dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(window: UVec2) -> FgInputs {
        FgInputs {
            wanted: true,
            in_world: true,
            blocked: false,
            window,
            epoch: 0,
        }
    }

    /// Two world frames of history, then world and held frames alternate.
    #[test]
    fn world_and_held_frames_alternate_once_history_is_valid() {
        let mut c = FgCadence::default();
        let w = UVec2::new(1600, 900);
        let seq: Vec<bool> = (0..8).map(|_| c.next(on(w))).collect();
        assert_eq!(
            seq,
            [false, false, true, false, true, false, true, false],
            "{seq:?}"
        );
    }

    /// A resize, a capture, leaving the world or a new epoch forces world frames and restarts the
    /// history count.
    #[test]
    fn a_reset_forces_world_frames() {
        let mut c = FgCadence::default();
        let w = UVec2::new(1600, 900);
        for _ in 0..4 {
            c.next(on(w));
        }
        // A resize: a world frame, and a fresh count.
        assert!(!c.next(on(UVec2::new(1280, 720))));
        assert!(!c.next(on(UVec2::new(1280, 720))));
        assert!(c.next(on(UVec2::new(1280, 720))));
        // A capture holds nothing.
        let blocked = FgInputs {
            blocked: true,
            ..on(UVec2::new(1280, 720))
        };
        assert!((0..4).all(|_| !c.next(blocked)));
        // Off, or out of the world: never.
        let off = FgInputs {
            wanted: false,
            ..on(UVec2::new(1280, 720))
        };
        assert!((0..4).all(|_| !c.next(off)));
        let out = FgInputs {
            in_world: false,
            ..on(UVec2::new(1280, 720))
        };
        assert!((0..4).all(|_| !c.next(out)));
        // A new epoch (a loading screen, world entry).
        let w = UVec2::new(1280, 720);
        for _ in 0..3 {
            c.next(on(w));
        }
        let epoch = FgInputs { epoch: 1, ..on(w) };
        assert!(!c.next(epoch));
        assert!(!c.next(epoch));
        assert!(c.next(epoch));
    }
}
