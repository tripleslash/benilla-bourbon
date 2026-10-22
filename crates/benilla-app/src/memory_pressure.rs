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
/// A step comes back only while this many bytes would remain after its reload.
const RESTORE_ABOVE: usize = 1 << 30;
/// Each step keeps this share of the view distance.
const STEP: f32 = 0.8;
/// Seconds after a change before the next is weighed: the window's tiles release and load over a
/// few frames, and a step's worth is measured once they have.
const SETTLE_SECS: f64 = 4.0;
/// Seconds the headroom must stay ample before a step comes back, so a passing burst of frees does
/// not reload what the next one sheds again.
const DWELL_SECS: f64 = 10.0;

/// One shed step: the cap it put on, and the headroom it bought, measured once settled.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shed {
    cap: f32,
    headroom_before: usize,
    freed: Option<usize>,
}

/// The valve's state, apart from the clock and the headroom it is handed.
#[derive(Resource, Default, Debug)]
pub(crate) struct MemoryPressure {
    /// The player's view distance while a cap holds.
    setting: f32,
    /// The view distance this last wrote; any other value is the player's.
    applied: f32,
    /// The caps in force, tightest last.
    sheds: Vec<Shed>,
    /// When this last changed the view distance.
    changed_at: f64,
    /// Since when the headroom has sufficed for the last step to come back.
    ample_since: Option<f64>,
}

impl MemoryPressure {
    /// One weighing at `now` seconds, given the allocator's `headroom` and the live `farclip`:
    /// the view distance to write, if it changes.
    fn weigh(&mut self, now: f64, headroom: usize, farclip: f32) -> Option<f32> {
        // The player moved the setting while a cap holds: the new setting, under the cap.
        if let Some(cap) = self.sheds.last().map(|s| s.cap) {
            if farclip != self.applied {
                self.setting = farclip;
                self.applied = farclip.min(cap);
                return (self.applied != farclip).then_some(self.applied);
            }
        }
        if now - self.changed_at < SETTLE_SECS {
            return None;
        }
        // A settled step's worth: what its unloads gave back.
        if let Some(last) = self.sheds.last_mut() {
            if last.freed.is_none() {
                last.freed = Some(headroom.saturating_sub(last.headroom_before));
            }
        }
        if headroom < SHED_BELOW {
            self.ample_since = None;
            let cap = (farclip * STEP).max(*FARCLIP_RANGE.start());
            if cap >= farclip {
                return None;
            }
            if self.sheds.is_empty() {
                self.setting = farclip;
            }
            self.sheds.push(Shed {
                cap,
                headroom_before: headroom,
                freed: None,
            });
            self.applied = cap;
            self.changed_at = now;
            return Some(cap);
        }
        let last = *self.sheds.last()?;
        // Bringing the step back loads about what shedding it freed.
        if headroom.saturating_sub(last.freed.unwrap_or(0)) < RESTORE_ABOVE {
            self.ample_since = None;
            return None;
        }
        let since = *self.ample_since.get_or_insert(now);
        if now - since < DWELL_SECS {
            return None;
        }
        self.sheds.pop();
        self.ample_since = None;
        self.applied = self
            .sheds
            .last()
            .map_or(self.setting, |s| s.cap.min(self.setting));
        self.changed_at = now;
        (self.applied != farclip).then_some(self.applied)
    }

    /// The caps in force, for the console line.
    fn depth(&self) -> usize {
        self.sheds.len()
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
        assert_eq!(v.weigh(11.0, 100 * MIB, 800.0), None, "settling");
        assert_eq!(v.weigh(15.0, 100 * MIB, 800.0), Some(640.0));
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

    /// Once the headroom has stayed ample for the dwell, the steps come back one at a time, each
    /// after a settle, ending at the player's setting.
    #[test]
    fn steps_come_back_one_at_a_time_after_the_dwell() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 100 * MIB, 1000.0), Some(800.0));
        assert_eq!(v.weigh(15.0, 100 * MIB, 800.0), Some(640.0));
        // The second step freed 300 MiB, so it comes back only above 1 GiB + 300 MiB.
        assert_eq!(v.weigh(20.0, 400 * MIB, 640.0), None);
        assert_eq!(v.weigh(21.0, 1200 * MIB, 640.0), None, "short of its cost");
        assert_eq!(v.weigh(22.0, 1400 * MIB, 640.0), None, "the dwell starts");
        assert_eq!(v.weigh(31.0, 1400 * MIB, 640.0), None);
        assert_eq!(v.weigh(32.0, 1400 * MIB, 640.0), Some(800.0));
        assert_eq!(v.weigh(33.0, 3000 * MIB, 800.0), None, "settling");
        assert_eq!(v.weigh(40.0, 3000 * MIB, 800.0), None, "a new dwell");
        assert_eq!(v.weigh(50.0, 3000 * MIB, 800.0), Some(1000.0));
        assert_eq!(v.depth(), 0);
        assert_eq!(v.weigh(80.0, 3000 * MIB, 1000.0), None);
    }

    /// A dip below the restore line restarts the dwell.
    #[test]
    fn a_dip_restarts_the_dwell() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 100 * MIB, 1000.0), Some(800.0));
        // Settled: the step freed 500 MiB, so it comes back only above 1 GiB + 500 MiB.
        assert_eq!(v.weigh(15.0, 600 * MIB, 800.0), None);
        assert_eq!(v.weigh(16.0, 2000 * MIB, 800.0), None, "the dwell starts");
        assert_eq!(v.weigh(24.0, 1400 * MIB, 800.0), None, "the dip");
        assert_eq!(v.weigh(25.0, 2000 * MIB, 800.0), None, "a new dwell");
        assert_eq!(v.weigh(34.0, 2000 * MIB, 800.0), None);
        assert_eq!(v.weigh(35.0, 2000 * MIB, 800.0), Some(1000.0));
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
        // Settled at 500 MiB freed; ample from 21, the step comes back at 31, to 600.
        assert_eq!(v.weigh(20.0, 600 * MIB, 600.0), None);
        assert_eq!(v.weigh(21.0, 3000 * MIB, 600.0), None);
        assert_eq!(
            v.weigh(31.0, 3000 * MIB, 600.0),
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
