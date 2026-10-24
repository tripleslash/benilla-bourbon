//! fork: web — the threaded browser build's memory-pressure valve on the view distance.
//!
//! wasm32 linear memory stops at 4 GiB, and a long view distance over HD art can reach it: the
//! allocator then fails and a worker traps with no message. The view distance decides what stays
//! loaded (`farclip` sets the streamer's residency window, `terrain_stream::window`), so this
//! lowers the effective `farclip` a step at a time while the allocator's [`headroom`] is short, and
//! gives the steps back, one at a time, once it is ample again.
//!
//! The player's setting is never touched: the `farclip` CVar and the saved config keep their value,
//! and only `ViewDistance::farclip`, which every consumer reads, carries the cap. A setting moved
//! while a cap holds is taken as the new setting, under the cap.
//!
//! Deviation: the reference has no such valve; its 32-bit process had the address space the page
//! lacks.
//!
//! [`headroom`]: crate::wasm_alloc::headroom

use benilla_world::view::FARCLIP_RANGE;
use bevy::prelude::*;

/// Below this many bytes of headroom, a step is shed.
const SHED_BELOW: usize = 512 << 20;
/// From this many bytes of headroom, held for the wait, a step comes back.
const RESTORE_ABOVE: usize = 1 << 30;
/// Each step keeps this share of the view distance.
const STEP: f32 = 0.8;
/// Seconds after a change before the next is weighed: the window's tiles release and load over a
/// few frames, and the allocator gets their memory back as their assets drop.
const SETTLE_SECS: f64 = 8.0;
/// Seconds the headroom must stay ample before a step comes back, so a passing burst of frees does
/// not reload what the next one sheds again.
const DWELL_SECS: f64 = 10.0;
/// The longest that wait grows to.
const DWELL_MAX_SECS: f64 = 300.0;
/// A shed this soon after a step came back means the step did not fit: the wait doubles.
const REFIT_SECS: f64 = 60.0;

/// The valve's state, apart from the clock and the headroom it is handed.
#[derive(Resource, Default, Debug)]
pub(crate) struct MemoryPressure {
    /// The player's view distance while a cap holds.
    setting: f32,
    /// The view distance this last wrote; any other value is the player's.
    applied: f32,
    /// The caps in force, tightest last.
    caps: Vec<f32>,
    /// When this last changed the view distance.
    changed_at: f64,
    /// Since when the headroom has been ample.
    ample_since: Option<f64>,
    /// The wait before a step comes back, past [`DWELL_SECS`] after steps that did not fit.
    dwell: f64,
    /// When the last step came back, while a shed may still say it did not fit.
    restored_at: Option<f64>,
}

impl MemoryPressure {
    /// One weighing at `now` seconds, given the allocator's `headroom` and the live `farclip`:
    /// the view distance to write, if it changes.
    fn weigh(&mut self, now: f64, headroom: usize, farclip: f32) -> Option<f32> {
        // The player moved the setting while a cap holds: the new setting, under the cap.
        if let Some(&cap) = self.caps.last() {
            if farclip != self.applied {
                self.setting = farclip;
                self.applied = farclip.min(cap);
                return (self.applied != farclip).then_some(self.applied);
            }
        }
        if now - self.changed_at < SETTLE_SECS {
            return None;
        }
        if headroom < SHED_BELOW {
            self.ample_since = None;
            let cap = (farclip * STEP).max(*FARCLIP_RANGE.start());
            if cap >= farclip {
                return None;
            }
            if self
                .restored_at
                .take()
                .is_some_and(|t| now - t < REFIT_SECS)
            {
                self.dwell = (self.dwell.max(DWELL_SECS) * 2.0).min(DWELL_MAX_SECS);
            }
            if self.caps.is_empty() {
                self.setting = farclip;
            }
            self.caps.push(cap);
            self.applied = cap;
            self.changed_at = now;
            return Some(cap);
        }
        if self.caps.is_empty() || headroom < RESTORE_ABOVE {
            self.ample_since = None;
            return None;
        }
        let since = *self.ample_since.get_or_insert(now);
        if now - since < self.dwell.max(DWELL_SECS) {
            return None;
        }
        self.caps.pop();
        self.ample_since = None;
        self.restored_at = Some(now);
        self.applied = self
            .caps
            .last()
            .map_or(self.setting, |&cap| cap.min(self.setting));
        if self.caps.is_empty() {
            self.dwell = 0.0;
        }
        self.changed_at = now;
        (self.applied != farclip).then_some(self.applied)
    }

    /// The caps in force, for the console line.
    fn depth(&self) -> usize {
        self.caps.len()
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
pub(crate) fn plugin(app: &mut App) {
    app.init_resource::<MemoryPressure>()
        .add_systems(First, relieve);
}

/// Weigh the headroom once a frame and apply the valve's answer, saying so on the console.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
fn relieve(
    mut valve: ResMut<MemoryPressure>,
    mut view: ResMut<benilla_world::view::ViewDistance>,
    time: Res<Time<Real>>,
) {
    let headroom = crate::wasm_alloc::headroom();
    let was = view.farclip;
    if let Some(farclip) = valve.weigh(time.elapsed_secs_f64(), headroom, was) {
        view.farclip = farclip;
        crate::webdiag::out(&format!(
            "wenilla: memory headroom {} MiB, view distance {was:.0} -> {farclip:.0} yd ({} step(s) shed)",
            headroom >> 20,
            valve.depth(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    /// Short headroom sheds a step at once, and another only after the settle.
    #[test]
    fn short_headroom_sheds_a_step_then_waits_for_the_unloads() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 100 * MIB, 1000.0), Some(800.0));
        assert_eq!(v.weigh(15.0, 100 * MIB, 800.0), None, "settling");
        assert_eq!(v.weigh(18.0, 100 * MIB, 800.0), Some(640.0));
        assert_eq!(v.depth(), 2);
    }

    /// Ample headroom never sheds, and with nothing shed there is nothing to give back.
    #[test]
    fn ample_headroom_leaves_the_view_alone() {
        let mut v = MemoryPressure::default();
        for t in 0..100 {
            assert_eq!(v.weigh(f64::from(t), 3000 * MIB, 1000.0), None);
        }
    }

    /// Once the headroom has held above the restore line for the wait, the steps come back one at
    /// a time, each after a settle, ending at the player's setting. The case seen live: shed to
    /// 640 near 500 MiB, then about 1 GiB free.
    #[test]
    fn steps_come_back_one_at_a_time_after_the_wait() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 499 * MIB, 1000.0), Some(800.0));
        assert_eq!(v.weigh(18.0, 511 * MIB, 800.0), Some(640.0));
        assert_eq!(v.weigh(26.0, 1046 * MIB, 640.0), None, "the wait starts");
        assert_eq!(v.weigh(35.0, 1046 * MIB, 640.0), None);
        assert_eq!(v.weigh(36.0, 1046 * MIB, 640.0), Some(800.0));
        assert_eq!(v.weigh(40.0, 1046 * MIB, 800.0), None, "settling");
        assert_eq!(v.weigh(44.0, 1046 * MIB, 800.0), None, "a new wait");
        assert_eq!(v.weigh(54.0, 1046 * MIB, 800.0), Some(1000.0));
        assert_eq!(v.depth(), 0);
        assert_eq!(v.weigh(80.0, 3000 * MIB, 1000.0), None);
    }

    /// A dip below the restore line restarts the wait.
    #[test]
    fn a_dip_restarts_the_wait() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 100 * MIB, 1000.0), Some(800.0));
        assert_eq!(v.weigh(18.0, 2000 * MIB, 800.0), None, "the wait starts");
        assert_eq!(v.weigh(25.0, 900 * MIB, 800.0), None, "the dip");
        assert_eq!(v.weigh(26.0, 2000 * MIB, 800.0), None, "a new wait");
        assert_eq!(v.weigh(35.0, 2000 * MIB, 800.0), None);
        assert_eq!(v.weigh(36.0, 2000 * MIB, 800.0), Some(1000.0));
    }

    /// A step that comes back and is shed again within a minute did not fit: the next wait is
    /// twice as long.
    #[test]
    fn a_step_that_did_not_fit_doubles_the_wait() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 100 * MIB, 1000.0), Some(800.0));
        assert_eq!(v.weigh(18.0, 1100 * MIB, 800.0), None, "the wait starts");
        assert_eq!(v.weigh(28.0, 1100 * MIB, 800.0), Some(1000.0));
        assert_eq!(
            v.weigh(36.0, 400 * MIB, 1000.0),
            Some(800.0),
            "it did not fit"
        );
        assert_eq!(v.weigh(44.0, 1100 * MIB, 800.0), None, "the wait starts");
        assert_eq!(v.weigh(54.0, 1100 * MIB, 800.0), None, "twice as long now");
        assert_eq!(v.weigh(64.0, 1100 * MIB, 800.0), Some(1000.0));
    }

    /// A setting moved under a cap is the new setting: raised, it stays under the cap; lowered
    /// below it, it is written as is; and the steps come back to it, not to the old one.
    #[test]
    fn the_players_setting_is_kept_under_the_cap() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 100 * MIB, 1000.0), Some(800.0));
        assert_eq!(
            v.weigh(12.0, 100 * MIB, 1200.0),
            Some(800.0),
            "raised: capped"
        );
        assert_eq!(
            v.weigh(13.0, 100 * MIB, 600.0),
            None,
            "lowered: below the cap"
        );
        assert_eq!(v.weigh(20.0, 3000 * MIB, 600.0), None, "the wait starts");
        assert_eq!(
            v.weigh(30.0, 3000 * MIB, 600.0),
            None,
            "back to 600, already there"
        );
        assert_eq!(v.depth(), 0);
        assert_eq!(
            v.weigh(40.0, 3000 * MIB, 1000.0),
            None,
            "no cap: the player's own"
        );
    }

    /// At the setting's floor there is nothing left to shed.
    #[test]
    fn the_floor_sheds_nothing() {
        let mut v = MemoryPressure::default();
        let floor = *FARCLIP_RANGE.start();
        assert_eq!(v.weigh(10.0, 100 * MIB, floor), None);
        assert_eq!(v.weigh(10.0, 100 * MIB, 200.0), Some(floor));
        assert_eq!(v.weigh(20.0, 100 * MIB, floor), None);
    }
}
