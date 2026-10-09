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
//!
//! **Solo world frames.** A pair only reads as smooth when its two frames are on screen equally
//! long. The web presents each frame on the next `requestAnimationFrame` after its work, so a world
//! frame that costs four display intervals beside a held frame's one shows the generated frame for
//! one interval and the real frame for four (measured in Frostmane Hollow: rAF intervals
//! alternating 6.9 and 27.8 ms). While the world frame outlasts the held frame like that, the
//! cadence stops holding: world frames render back to back ([`FgSolo`]), each presenting the real
//! frame it rendered, until one fits a held frame's interval again. Native has its own pacer
//! ([`pace_held_frames`]) and is not timed.

use bevy::core_pipeline::schedule::RootNonCameraView;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::ecs::system::SystemState;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::ExtractComponent;
use bevy::window::PrimaryWindow;

use super::{FgSolo, FsrHeld, FsrView};

/// A world frame lasting this many held frames' intervals or more splits its pair unevenly; the
/// web's frames last whole display intervals, so an even pair reads 1 and the first uneven one 2.
const UNEVEN_SPLIT: f32 = 1.5;
/// Uneven pairs in a row before the cadence goes solo: a single hitch (a streamed tile, a GC
/// pause) is not a world frame that no longer fits.
const UNEVEN_PAIRS: u32 = 6;
/// Fitting solo frames in a row before holding again, at first; doubled each time holding again
/// went uneven within [`RELAPSE_FRAMES`], so a world frame on the edge does not flap.
const RESUME_FRAMES: u32 = 30;
/// The longest wait before holding again: about four seconds at 144 Hz.
const RESUME_FRAMES_MAX: u32 = 480;
/// World frames after holding again within which going uneven counts as a relapse.
const RELAPSE_FRAMES: u32 = 240;

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

/// Whether frame generation is wanted: the `frameGeneration` CVar, which the app writes. On by
/// default, as the CVar registers.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct FrameGeneration(pub bool);

impl Default for FrameGeneration {
    fn default() -> Self {
        Self(true)
    }
}

/// Frames that must render: a capture run, or anything else that reads every frame's pixels. The
/// app sets it; while set, no frame is held.
#[derive(Resource, Clone, Copy, Default, Debug)]
pub struct FgBlocked(pub bool);

/// The cadence state machine. Pure, so its rules are unit-tested without an app.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct FgCadence {
    /// The last frame was held.
    held: bool,
    /// World frames render back to back: a pair would split unevenly.
    solo: bool,
    /// The last frame was a solo world frame: it ran no chain, so nothing can be held after it.
    ran_solo: bool,
    /// The last world frame's interval, while a held frame follows it.
    paired: Option<f32>,
    /// A held frame's interval, smoothed: about one display interval where frames are timed.
    held_interval: Option<f32>,
    /// Uneven pairs in a row, while holding; fitting solo frames in a row, while solo.
    run: u32,
    /// Fitting solo frames needed before holding again.
    resume_after: u32,
    /// World frames since holding again, counted up to [`RELAPSE_FRAMES`].
    since_resume: u32,
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
    /// Seconds since the last frame began, where frames are timed (the web); `None` never goes solo.
    pub interval: Option<f32>,
}

/// What the cadence decided for this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FgVerdict {
    /// The frame is held.
    pub held: bool,
    /// A world frame no held frame follows: it presents its own real frame.
    pub solo: bool,
}

impl FgCadence {
    /// Decide this frame. A held frame is only ever the one after a world frame that interpolated
    /// from valid history; a resize, a new epoch, a capture or leaving the world forces a world
    /// frame and restarts the count. A timed world frame that outlasts held frames turns the
    /// cadence solo, and a solo one that fits a held frame's interval again turns it back.
    pub fn next(&mut self, i: FgInputs) -> FgVerdict {
        let reset = i.window != self.window || i.epoch != self.epoch || !i.wanted || !i.in_world;
        if reset {
            self.since_reset = 0;
            // A fresh history is a fresh scene: try holding again.
            if self.solo {
                self.solo = false;
                self.run = 0;
                self.since_resume = 0;
            }
        }
        match i.interval.filter(|dt| dt.is_finite() && *dt > 0.0) {
            Some(dt) => self.time_last_frame(dt),
            None => self.paired = None,
        }
        // The SDK interpolates only from the second rendered frame of a history on.
        let hold = !self.held
            && !self.ran_solo
            && !reset
            && !i.blocked
            && !self.solo
            && self.since_reset >= 2;
        if !hold {
            self.since_reset = self.since_reset.saturating_add(1);
            self.window = i.window;
            self.epoch = i.epoch;
            self.since_resume = self.since_resume.saturating_add(1).min(RELAPSE_FRAMES);
        }
        self.held = hold;
        self.ran_solo = !hold && self.solo;
        FgVerdict {
            held: hold,
            solo: self.ran_solo,
        }
    }

    /// Book the last frame's interval: a held frame's sets the reference and closes its pair, a
    /// world frame's waits for the held frame after it or, solo, is checked against the reference.
    fn time_last_frame(&mut self, dt: f32) {
        if self.held {
            self.held_interval = Some(self.held_interval.map_or(dt, |h| h * 0.9 + dt * 0.1));
            if let Some(world) = self.paired.take() {
                // Against this pair's own held frame, so a slow stretch of both reads even.
                self.run = if world >= UNEVEN_SPLIT * dt {
                    self.run + 1
                } else {
                    0
                };
                if self.run >= UNEVEN_PAIRS {
                    self.go_solo();
                }
            }
        } else if self.solo {
            let fits = self.held_interval.is_some_and(|h| dt < UNEVEN_SPLIT * h);
            self.run = if fits { self.run + 1 } else { 0 };
            if self.run >= self.resume_after.max(RESUME_FRAMES) {
                self.solo = false;
                self.run = 0;
                self.since_resume = 0;
            }
        } else {
            self.paired = Some(dt);
        }
    }

    fn go_solo(&mut self) {
        self.solo = true;
        self.run = 0;
        self.paired = None;
        // Uneven again soon after holding again: wait twice as long before the next try.
        self.resume_after = if self.resume_after > 0 && self.since_resume < RELAPSE_FRAMES {
            (self.resume_after.max(RESUME_FRAMES) * 2).min(RESUME_FRAMES_MAX)
        } else {
            RESUME_FRAMES
        };
    }
}

/// What [`fg_main`] reads each frame: the primary window and the FSR view.
type FgLookups = (
    Query<'static, 'static, &'static Window, With<PrimaryWindow>>,
    Query<'static, 'static, &'static FsrView>,
);

/// `FgMain`'s one system: hold the frame, or run the game. With frame generation off it only
/// resets the cadence and runs `Main`; its lookups are a cached system state, not rebuilt. On the
/// web it times each frame from the start of the last: the `requestAnimationFrame` cadence.
fn fg_main(
    world: &mut World,
    lookups: &mut SystemState<FgLookups>,
    mut last: Local<Option<bevy::platform::time::Instant>>,
) {
    let now = bevy::platform::time::Instant::now();
    let interval = last
        .replace(now)
        .map(|at| now.duration_since(at).as_secs_f32())
        .filter(|_| cfg!(target_arch = "wasm32"));
    let wanted = world.resource::<FrameGeneration>().0;
    let verdict = if wanted {
        let (window, in_world, epoch) = {
            let (windows, views) = lookups.get(world).expect("two queries validate");
            let window = windows.iter().next().map_or(UVec2::ZERO, |w| {
                UVec2::new(w.physical_width(), w.physical_height())
            });
            let (in_world, epoch) = views.iter().next().map_or((false, 0), |f| (true, f.epoch));
            (window, in_world, epoch)
        };
        let inputs = FgInputs {
            wanted,
            in_world,
            blocked: world.resource::<FgBlocked>().0,
            window,
            epoch,
            interval,
        };
        world.resource_mut::<FgCadence>().next(inputs)
    } else {
        // Not wanted resets the history count whatever the rest reads.
        world.resource_mut::<FgCadence>().next(FgInputs {
            wanted,
            in_world: false,
            blocked: false,
            window: UVec2::ZERO,
            epoch: 0,
            interval: None,
        })
    };
    let held = verdict.held;
    // Written on change only: the render world keeps the last extracted verdict between changes.
    if world.resource::<FsrHeld>().0 != held {
        world.resource_mut::<FsrHeld>().0 = held;
    }
    if verdict.solo {
        SOLO.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if world.resource::<FgSolo>().0 != verdict.solo {
        world.resource_mut::<FgSolo>().0 = verdict.solo;
    }
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

/// Solo world frames since the world viewer's frame line last read it.
pub static SOLO: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

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
            interval: None,
        }
    }

    /// One display interval at 144 Hz, seconds.
    const VSYNC: f32 = 1.0 / 144.0;

    /// Feed `frames` frames, each lasting `world` display intervals when it renders the world and
    /// one when it is held, as the web's `requestAnimationFrame` paces them; the verdicts back.
    fn run(c: &mut FgCadence, w: UVec2, world: f32, frames: usize) -> Vec<FgVerdict> {
        let mut last: Option<FgVerdict> = None;
        (0..frames)
            .map(|_| {
                let interval = last.map(|v| if v.held { VSYNC } else { world * VSYNC });
                let v = c.next(FgInputs { interval, ..on(w) });
                last = Some(v);
                v
            })
            .collect()
    }

    /// Two world frames of history, then world and held frames alternate.
    #[test]
    fn world_and_held_frames_alternate_once_history_is_valid() {
        let mut c = FgCadence::default();
        let w = UVec2::new(1600, 900);
        let seq: Vec<bool> = (0..8).map(|_| c.next(on(w)).held).collect();
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
        assert!(!c.next(on(UVec2::new(1280, 720))).held);
        assert!(!c.next(on(UVec2::new(1280, 720))).held);
        assert!(c.next(on(UVec2::new(1280, 720))).held);
        // A capture holds nothing.
        let blocked = FgInputs {
            blocked: true,
            ..on(UVec2::new(1280, 720))
        };
        assert!((0..4).all(|_| !c.next(blocked).held));
        // Off, or out of the world: never.
        let off = FgInputs {
            wanted: false,
            ..on(UVec2::new(1280, 720))
        };
        assert!((0..4).all(|_| !c.next(off).held));
        let out = FgInputs {
            in_world: false,
            ..on(UVec2::new(1280, 720))
        };
        assert!((0..4).all(|_| !c.next(out).held));
        // A new epoch (a loading screen, world entry).
        let w = UVec2::new(1280, 720);
        for _ in 0..3 {
            c.next(on(w));
        }
        let epoch = FgInputs { epoch: 1, ..on(w) };
        assert!(!c.next(epoch).held);
        assert!(!c.next(epoch).held);
        assert!(c.next(epoch).held);
    }

    /// A world frame that fits a display interval keeps alternating, timed or not.
    #[test]
    fn an_even_pair_keeps_holding() {
        let mut c = FgCadence::default();
        let seq = run(&mut c, UVec2::new(1600, 900), 1.0, 400);
        assert!(seq.iter().all(|v| !v.solo));
        assert_eq!(seq.iter().filter(|v| v.held).count(), 199);
    }

    /// Four display intervals beside a held frame's one (Frostmane Hollow's bridge): after a run
    /// of uneven pairs the world frames go solo, and none is held while they stay that slow.
    #[test]
    fn an_uneven_pair_goes_solo() {
        let mut c = FgCadence::default();
        let seq = run(&mut c, UVec2::new(1600, 900), 4.0, 400);
        let first = seq.iter().position(|v| v.solo).expect("goes solo");
        assert!(first < 2 + 2 * (UNEVEN_PAIRS as usize + 1), "{first}");
        assert!(seq[first..].iter().all(|v| v.solo && !v.held));
    }

    /// A single hitch is not a world frame that no longer fits.
    #[test]
    fn one_slow_pair_does_not_go_solo() {
        let mut c = FgCadence::default();
        let w = UVec2::new(1600, 900);
        run(&mut c, w, 1.0, 20);
        // The run above ended on a world frame; one slow world frame, then even ones again.
        let mut step = |dt: f32| {
            c.next(FgInputs {
                interval: Some(dt),
                ..on(w)
            })
        };
        assert!(step(VSYNC).held);
        assert!(!step(VSYNC).held);
        assert!(step(5.0 * VSYNC).held);
        assert!(run(&mut c, w, 1.0, 100).iter().all(|v| !v.solo));
    }

    /// Solo world frames that fit a display interval again hold again, through a world frame that
    /// runs the chain first: a held frame never follows a solo one.
    #[test]
    fn a_fitting_solo_frame_holds_again() {
        let mut c = FgCadence::default();
        let w = UVec2::new(1600, 900);
        run(&mut c, w, 4.0, 60);
        assert!(c.solo);
        let seq = run(&mut c, w, 1.0, 200);
        let back = seq.iter().position(|v| !v.solo).expect("holds again");
        assert!(back <= RESUME_FRAMES as usize + 1, "{back}");
        assert!(!seq[back].held, "the first frame back renders the world");
        assert!(seq[back + 1].held);
        for pair in seq.windows(2) {
            assert!(!(pair[0].solo && pair[1].held));
        }
    }

    /// Going uneven again soon after holding again doubles the wait before the next try, up to
    /// its bound; a long even stretch starts the wait over.
    #[test]
    fn a_relapse_waits_longer() {
        let mut c = FgCadence::default();
        let w = UVec2::new(1600, 900);
        run(&mut c, w, 4.0, 60);
        assert_eq!(c.resume_after, RESUME_FRAMES);
        for want in [2, 4, 8, 16, 16].map(|k| (RESUME_FRAMES * k).min(RESUME_FRAMES_MAX)) {
            // Fits solo, holds again, then the pair is uneven at once.
            let wait = c.resume_after as usize + 2;
            run(&mut c, w, 1.0, wait);
            assert!(!c.solo);
            run(&mut c, w, 4.0, 40);
            assert!(c.solo);
            assert_eq!(c.resume_after, want);
        }
        run(
            &mut c,
            w,
            1.0,
            RESUME_FRAMES_MAX as usize + 2 + 2 * RELAPSE_FRAMES as usize,
        );
        assert!(!c.solo);
        run(&mut c, w, 4.0, 40);
        assert_eq!(c.resume_after, RESUME_FRAMES);
    }

    /// A reset tries holding again; untimed frames (native) never go solo.
    #[test]
    fn a_reset_or_no_timing_leaves_solo() {
        let mut c = FgCadence::default();
        let w = UVec2::new(1600, 900);
        run(&mut c, w, 4.0, 60);
        assert!(c.solo);
        assert!(!c.next(on(UVec2::new(1280, 720))).solo);
        assert!(!c.solo);
        let mut n = FgCadence::default();
        assert!((0..400).all(|_| !n.next(on(w)).solo));
    }
}
