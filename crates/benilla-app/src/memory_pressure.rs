//! fork: web — the threaded browser build's memory-pressure valve on the view distance.
//!
//! wasm32 linear memory stops at 4 GiB, and a long view distance over HD art can reach it: the
//! allocator then fails and a worker traps with no message. The view distance decides what stays
//! loaded (`farclip` sets the streamer's residency window, `terrain_stream::window`), so this
//! lowers the effective `farclip` a step at a time while the allocator's [`headroom`] or its
//! [`largest_run`] is short, and gives the steps back while there is room for them.
//!
//! What a step costs to take back depends on where the view is, so a step is tried: with the
//! headroom a step's worth above the shed line, the tightest step comes back, and after its settle
//! the next, while the room lasts. Where the shed steps held little (a city whose surroundings are
//! sparse) they all come back a settle apart. A step that takes the headroom under the shed line
//! while it loads is shed again at once; one shed again within a minute did not fit, and the wait
//! before the next try grows.
//!
//! The shed line guards a fall: flown into a city, the headroom drops faster than a shed step
//! unloads. Held still, it wanders, and a view that settles just under the line (Stormwind's full
//! ring, about 500 MiB) would shed and take back its last step without end. So the line holds only
//! while the headroom falls fast; still, a step is shed only a [`BAND`] under it.
//!
//! The player's setting is never touched: the `farclip` CVar and the saved config keep their value,
//! and only `ViewDistance::farclip`, which every consumer reads, carries the cap. A setting moved
//! while a cap holds is taken as the new setting, under the cap.
//!
//! Deviation: the reference has no such valve, because its 32-bit process had the address space
//! the page lacks.
//!
//! [`headroom`]: crate::wasm_alloc::headroom
//! [`largest_run`]: crate::wasm_alloc::largest_run

use benilla_world::view::FARCLIP_RANGE;
use bevy::prelude::*;

/// Below this many bytes of headroom, a step is shed (`?mem_shed_mb=`, a measuring lever). Flying
/// the world viewer into Stormwind at 40 yd/s, about a taxi's speed, the headroom fell 453 to 143
/// MiB in 2 s, faster than a shed step unloads: a line at 256 trapped, and one at 384 trapped in one
/// flight of two, each time with the step shed seconds before.
const SHED_BELOW: usize = 512 << 20;
/// How far under [`SHED_BELOW`] a still headroom may sit before a step is shed: Stormwind's 1017 yd
/// ring settled at 495 to 509 MiB, and the headroom wanders 35 MiB as a window settles.
const BAND: usize = 96 << 20;
/// Bytes a second of fall that make the headroom fast, for [`SHED_BELOW`]: a taxi's flight into
/// Stormwind fell 155 MiB/s, a window settling wanders 35 MiB.
const FALL_PER_SEC: f64 = (64u64 << 20) as f64;
/// The fall is measured across at least this many seconds, so one frame's burst is not a fall.
const FALL_SECS: f64 = 1.0;
/// Below this many bytes in the largest run, a step is shed (`?mem_run_mb=`, a measuring lever):
/// the headroom sums free slices, and a large block needs them in one run. The largest block those
/// flights asked for was 55.5 MiB, which a run from 56 MiB holds; both traps were a block of 16.5
/// and 49.7 MiB finding no run that long with 134 and 143 MiB free.
const RUN_BELOW: usize = 64 << 20;
/// The room above [`SHED_BELOW`] a step needs to come back: the most a step was measured to cost
/// with the view parked, about 200 MiB (Stormwind's 640 to 800 yd ring), and the headroom's 35 MiB
/// wander as a window settles.
const KEEP: usize = 256 << 20;
/// Each step keeps this share of the view distance.
const STEP: f32 = 0.8;
/// Seconds after a change before the next is weighed: the window's tiles release and load over a
/// few frames, and the allocator gets their memory back as their assets drop.
const SETTLE_SECS: f64 = 8.0;
/// Seconds the headroom must have room for a step before the first of a run comes back, so a
/// passing burst of frees does not reload what the next one sheds again. The next of the run
/// follows after a settle.
const DWELL_SECS: f64 = 3.0;
/// The wait after a step that did not fit, doubling with each that does not, to the longest.
const MISFIT_SECS: f64 = 30.0;
const DWELL_MAX_SECS: f64 = 300.0;
/// A shed this soon after a step came back means the step did not fit; a step that has held this
/// long did, and the wait falls back to [`DWELL_SECS`].
const REFIT_SECS: f64 = 60.0;

/// The valve's state, apart from the clock and the gauges it is handed.
#[derive(Resource, Debug)]
pub(crate) struct MemoryPressure {
    /// The headroom and largest run under which a step is shed.
    shed_below: usize,
    run_below: usize,
    /// The player's view distance while a cap holds.
    setting: f32,
    /// The view distance this last wrote; any other value is the player's.
    applied: f32,
    /// The caps in force, tightest last.
    caps: Vec<f32>,
    /// When this last changed the view distance.
    changed_at: f64,
    /// Since when the headroom has had room for a step.
    ample_since: Option<f64>,
    /// The wait before a step comes back, past [`DWELL_SECS`] after steps that did not fit.
    dwell: f64,
    /// When the last step came back, while a shed may still say it did not fit.
    restored_at: Option<f64>,
    /// The headroom sampled at least [`FALL_SECS`] apart, the older first, as (seconds, bytes).
    samples: [(f64, usize); 2],
}

impl Default for MemoryPressure {
    fn default() -> Self {
        Self {
            shed_below: SHED_BELOW,
            run_below: RUN_BELOW,
            setting: 0.0,
            applied: 0.0,
            caps: Vec::new(),
            changed_at: 0.0,
            ample_since: None,
            dwell: 0.0,
            restored_at: None,
            samples: [(0.0, 0); 2],
        }
    }
}

impl MemoryPressure {
    /// One weighing at `now` seconds, given the allocator's `headroom`, its `largest_run` and the
    /// live `farclip`: the view distance to write, if it changes.
    fn weigh(&mut self, now: f64, headroom: usize, run: usize, farclip: f32) -> Option<f32> {
        // The fall since the older sample, between one and two spans of FALL_SECS ago.
        if now - self.samples[1].0 >= FALL_SECS {
            self.samples = [self.samples[1], (now, headroom)];
        }
        let (then, was) = self.samples[0];
        let falling =
            was.saturating_sub(headroom) as f64 > FALL_PER_SEC * (now - then).max(FALL_SECS);
        // The player moved the setting while a cap holds: the new setting, under the cap.
        if let Some(&cap) = self.caps.last() {
            if farclip != self.applied {
                self.setting = farclip;
                self.applied = farclip.min(cap);
                return (self.applied != farclip).then_some(self.applied);
            }
        }
        // Under the band, or a run too short, at any pace; under the line, only while falling.
        let deep = headroom < self.shed_below.saturating_sub(BAND) || run < self.run_below;
        let short = deep || (headroom < self.shed_below && falling);
        // A step that came back is shed again as soon as its loads cross the band, its own loads
        // being the fall; any other change waits out its settle.
        let came_back = self.restored_at == Some(self.changed_at);
        if now - self.changed_at < SETTLE_SECS && !(came_back && deep) {
            return None;
        }
        if self.restored_at.is_some_and(|t| now - t >= REFIT_SECS) {
            self.restored_at = None;
            self.dwell = DWELL_SECS;
        }
        if short {
            self.ample_since = None;
            let cap = (farclip * STEP).max(*FARCLIP_RANGE.start());
            if cap >= farclip {
                return None;
            }
            if self.restored_at.take().is_some() {
                self.dwell = (self.dwell * 2.0).clamp(MISFIT_SECS, DWELL_MAX_SECS);
            }
            if self.caps.is_empty() {
                self.setting = farclip;
            }
            self.caps.push(cap);
            self.applied = cap;
            self.changed_at = now;
            return Some(cap);
        }
        if self.caps.is_empty() || headroom < self.shed_below + KEEP {
            self.ample_since = None;
            return None;
        }
        // Kept across a step that comes back, so the next follows its settle if the room lasts.
        let since = *self.ample_since.get_or_insert(now);
        if now - since < self.dwell.max(DWELL_SECS) {
            return None;
        }
        self.caps.pop();
        self.restored_at = Some(now);
        self.applied = self
            .caps
            .last()
            .map_or(self.setting, |&cap| cap.min(self.setting));
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
    app.insert_resource(MemoryPressure::from_page())
        .insert_resource(Ballast::from_page())
        .add_systems(First, relieve);
}

/// MiB from the page's `name` lever, if it names a number.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
fn page_mib(name: &str) -> Option<usize> {
    crate::webenv::var(name).and_then(|v| v.trim().parse::<usize>().ok())
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
impl MemoryPressure {
    /// The valve with the page's `?mem_shed_mb=` and `?mem_run_mb=` lines, where it sets them.
    fn from_page() -> Self {
        let mut valve = Self::default();
        if let Some(mib) = page_mib("WOW_MEM_SHED_MB") {
            valve.shed_below = mib << 20;
        }
        if let Some(mib) = page_mib("WOW_MEM_RUN_MB") {
            valve.run_below = mib << 20;
        }
        valve
    }
}

/// `?mem_ballast_mb=N`: N MiB held from the start, a measuring lever. The world viewer holds a
/// fraction of what a logged-in client does (no interface, Lua or units), so it meets the valve
/// only with the rest of a client's footprint stood in for.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
#[derive(Resource)]
struct Ballast {
    _held: Vec<Box<[u8]>>,
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
impl Ballast {
    fn from_page() -> Self {
        let mib = page_mib("WOW_MEM_BALLAST_MB").unwrap_or(0);
        // In 16 MiB blocks, so the ballast takes no one run of address space the world could use.
        let _held = (0..mib.div_ceil(16))
            .map(|_| vec![0u8; 16 << 20].into_boxed_slice())
            .collect();
        Self { _held }
    }
}

/// Seconds between the gauge lines `?diag=1` prints, a step or none.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
const GAUGE_SECS: f64 = 2.0;

/// Seconds between the allocator trims short gauges ask for.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
const TRIM_SECS: f64 = 1.0;

/// Weigh the headroom and the largest run once a frame and apply the valve's answer, saying so on
/// the console; under `?diag=1`, the gauges, the largest block asked for since the last line and
/// the steps every [`GAUGE_SECS`] besides.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    not(debug_assertions)
))]
fn relieve(
    mut valve: ResMut<MemoryPressure>,
    mut view: ResMut<benilla_world::view::ViewDistance>,
    time: Res<Time<Real>>,
    mut gauge: Local<Option<(bool, f64)>>,
    mut trimmed_at: Local<f64>,
) {
    let mut headroom = crate::wasm_alloc::headroom();
    let mut run = crate::wasm_alloc::largest_run();
    let now = time.elapsed_secs_f64();
    // Short gauges first give back what the allocator holds freed but unmerged, so a step is
    // never shed for memory a burst of frees left deferred or kept.
    if (headroom < valve.shed_below || run < valve.run_below) && now - *trimmed_at >= TRIM_SECS {
        *trimmed_at = now;
        crate::wasm_alloc::trim();
        headroom = crate::wasm_alloc::headroom();
        run = crate::wasm_alloc::largest_run();
    }
    let (armed, next) = gauge.get_or_insert_with(|| (crate::webdiag::armed(), 0.0));
    if *armed && now >= *next {
        *next = now + GAUGE_SECS;
        crate::webdiag::out(&format!(
            "[mem] t={now:.0}s headroom={} MiB run={} MiB biggest={} KiB linear={} MiB farclip={:.0} shed={}",
            headroom >> 20,
            run >> 20,
            crate::wasm_alloc::take_biggest() >> 10,
            core::arch::wasm32::memory_size(0) >> 4,
            view.farclip,
            valve.depth(),
        ));
    }
    let was = view.farclip;
    if let Some(farclip) = valve.weigh(now, headroom, run, was) {
        view.farclip = farclip;
        crate::webdiag::out(&format!(
            "wow: memory headroom {} MiB, run {} MiB, view distance {was:.0} -> {farclip:.0} yd ({} step(s) shed)",
            headroom >> 20,
            run >> 20,
            valve.depth(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;
    /// A largest run that never sheds.
    const RUN: usize = 4000 * MIB;

    /// Short headroom sheds a step at once, and another only after the settle.
    #[test]
    fn short_headroom_sheds_a_step_then_waits_for_the_unloads() {
        let mut v = MemoryPressure::default();
        assert_eq!(
            v.weigh(10.0, 600 * MIB, RUN, 1000.0),
            None,
            "above the line"
        );
        assert_eq!(v.weigh(10.0, 400 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(v.weigh(15.0, 100 * MIB, RUN, 800.0), None, "settling");
        assert_eq!(v.weigh(18.0, 400 * MIB, RUN, 800.0), Some(640.0));
        assert_eq!(v.depth(), 2);
    }

    /// A fast fall sheds at the line, above the band: the flight into a city.
    #[test]
    fn a_fast_fall_sheds_at_the_line() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 900 * MIB, RUN, 1000.0), None);
        assert_eq!(v.weigh(11.0, 750 * MIB, RUN, 1000.0), None);
        assert_eq!(
            v.weigh(12.0, 600 * MIB, RUN, 1000.0),
            None,
            "above the line"
        );
        assert_eq!(v.weigh(13.0, 500 * MIB, RUN, 1000.0), Some(800.0));
    }

    /// A still headroom in the band sheds nothing.
    #[test]
    fn a_still_headroom_in_the_band_sheds_nothing() {
        let mut v = MemoryPressure::default();
        for t in 10..200 {
            let headroom = if t % 2 == 0 { 495 } else { 509 };
            assert_eq!(v.weigh(f64::from(t), headroom * MIB, RUN, 1000.0), None);
        }
    }

    /// A step whose loads settle in the band stays: Stormwind's last step, 814 to 1017 yd, which
    /// cost 340 MiB and left 495 to 509, no longer goes and comes back without end.
    #[test]
    fn a_step_that_settles_in_the_band_stays() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 400 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(
            v.weigh(18.0, 834 * MIB, RUN, 800.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(21.0, 834 * MIB, RUN, 800.0), Some(1000.0));
        for (t, headroom) in [(22, 700), (23, 600), (24, 520), (25, 509), (26, 495)] {
            assert_eq!(
                v.weigh(f64::from(t), headroom * MIB, RUN, 1000.0),
                None,
                "its own loads"
            );
        }
        for t in 27..400 {
            let headroom = if t % 2 == 0 { 495 } else { 509 };
            assert_eq!(v.weigh(f64::from(t), headroom * MIB, RUN, 1000.0), None);
        }
        assert_eq!(v.depth(), 0);
    }

    /// Ample headroom never sheds, and with nothing shed there is nothing to give back.
    #[test]
    fn ample_headroom_leaves_the_view_alone() {
        let mut v = MemoryPressure::default();
        for t in 0..100 {
            assert_eq!(v.weigh(f64::from(t), 3000 * MIB, RUN, 1000.0), None);
        }
    }

    /// Steps that cost little come back one after another, a settle apart, down to none, though
    /// the headroom never reaches what was once the restore line (1 GiB): the city case.
    #[test]
    fn cheap_steps_come_back_in_a_run_down_to_none() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 400 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(v.weigh(18.0, 400 * MIB, RUN, 800.0), Some(640.0));
        assert_eq!(v.weigh(26.0, 700 * MIB, RUN, 640.0), None, "no room");
        assert_eq!(
            v.weigh(30.0, 850 * MIB, RUN, 640.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(33.0, 850 * MIB, RUN, 640.0), Some(800.0));
        assert_eq!(v.weigh(37.0, 830 * MIB, RUN, 800.0), None, "settling");
        assert_eq!(
            v.weigh(41.0, 830 * MIB, RUN, 800.0),
            Some(1000.0),
            "the run"
        );
        assert_eq!(v.depth(), 0);
        assert_eq!(v.weigh(80.0, 800 * MIB, RUN, 1000.0), None);
    }

    /// A run stops where the room ends, and holds there.
    #[test]
    fn a_run_stops_where_the_room_ends() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 400 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(v.weigh(18.0, 400 * MIB, RUN, 800.0), Some(640.0));
        assert_eq!(
            v.weigh(26.0, 900 * MIB, RUN, 640.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(29.0, 900 * MIB, RUN, 640.0), Some(800.0));
        assert_eq!(v.weigh(37.0, 700 * MIB, RUN, 800.0), None, "it cost 200");
        for t in 38..200 {
            assert_eq!(v.weigh(f64::from(t), 700 * MIB, RUN, 800.0), None);
        }
        assert_eq!(v.depth(), 1);
    }

    /// A dip under the room a step needs restarts the wait.
    #[test]
    fn a_dip_restarts_the_wait() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 400 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(
            v.weigh(18.0, 2000 * MIB, RUN, 800.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(20.0, 700 * MIB, RUN, 800.0), None, "the dip");
        assert_eq!(v.weigh(21.0, 2000 * MIB, RUN, 800.0), None, "a new wait");
        assert_eq!(v.weigh(23.0, 2000 * MIB, RUN, 800.0), None);
        assert_eq!(v.weigh(24.0, 2000 * MIB, RUN, 800.0), Some(1000.0));
    }

    /// A step whose loads take the headroom under the line is shed again at once, without the
    /// settle, and the next try waits half a minute, then a minute: no back and forth a settle
    /// apart. Once a step holds a minute, the wait is short again.
    #[test]
    fn a_step_that_does_not_fit_goes_at_once_and_waits_longer_each_time() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 400 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(
            v.weigh(18.0, 800 * MIB, RUN, 800.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(21.0, 800 * MIB, RUN, 800.0), Some(1000.0));
        assert_eq!(v.weigh(22.0, 600 * MIB, RUN, 1000.0), None, "loading");
        assert_eq!(
            v.weigh(23.0, 400 * MIB, RUN, 1000.0),
            Some(800.0),
            "under the line: shed at once"
        );
        assert_eq!(
            v.weigh(31.0, 800 * MIB, RUN, 800.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(60.0, 800 * MIB, RUN, 800.0), None, "half a minute");
        assert_eq!(v.weigh(61.0, 800 * MIB, RUN, 800.0), Some(1000.0));
        assert_eq!(v.weigh(65.0, 400 * MIB, RUN, 1000.0), Some(800.0), "again");
        assert_eq!(
            v.weigh(73.0, 800 * MIB, RUN, 800.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(132.0, 800 * MIB, RUN, 800.0), None, "a minute now");
        assert_eq!(v.weigh(133.0, 800 * MIB, RUN, 800.0), Some(1000.0));
        assert_eq!(v.weigh(193.0, 700 * MIB, RUN, 1000.0), None, "it held");
        assert_eq!(v.weigh(200.0, 400 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(
            v.weigh(208.0, 800 * MIB, RUN, 800.0),
            None,
            "the wait starts"
        );
        assert_eq!(
            v.weigh(211.0, 800 * MIB, RUN, 800.0),
            Some(1000.0),
            "short again"
        );
    }

    /// A setting moved under a cap is the new setting: raised, it stays under the cap; lowered
    /// below it, it is written as is; and the steps come back to it, not to the old one.
    #[test]
    fn the_players_setting_is_kept_under_the_cap() {
        let mut v = MemoryPressure::default();
        assert_eq!(v.weigh(10.0, 100 * MIB, RUN, 1000.0), Some(800.0));
        assert_eq!(
            v.weigh(12.0, 100 * MIB, RUN, 1200.0),
            Some(800.0),
            "raised: capped"
        );
        assert_eq!(
            v.weigh(13.0, 100 * MIB, RUN, 600.0),
            None,
            "lowered: below the cap"
        );
        assert_eq!(
            v.weigh(20.0, 3000 * MIB, RUN, 600.0),
            None,
            "the wait starts"
        );
        assert_eq!(
            v.weigh(30.0, 3000 * MIB, RUN, 600.0),
            None,
            "back to 600, already there"
        );
        assert_eq!(v.depth(), 0);
        assert_eq!(
            v.weigh(40.0, 3000 * MIB, RUN, 1000.0),
            None,
            "no cap: the player's own"
        );
    }

    /// A short largest run sheds a step though the headroom is ample, and a step that comes back
    /// and shortens it is shed again at once.
    #[test]
    fn a_short_run_sheds_though_the_headroom_is_ample() {
        let mut v = MemoryPressure {
            run_below: 128 * MIB,
            ..MemoryPressure::default()
        };
        assert_eq!(v.weigh(10.0, 2000 * MIB, 200 * MIB, 1000.0), None);
        assert_eq!(v.weigh(11.0, 2000 * MIB, 100 * MIB, 1000.0), Some(800.0));
        assert_eq!(
            v.weigh(19.0, 2000 * MIB, 300 * MIB, 800.0),
            None,
            "the wait starts"
        );
        assert_eq!(v.weigh(22.0, 2000 * MIB, 300 * MIB, 800.0), Some(1000.0));
        assert_eq!(
            v.weigh(23.0, 2000 * MIB, 64 * MIB, 1000.0),
            Some(800.0),
            "did not fit"
        );
    }

    /// At the setting's floor there is nothing left to shed.
    #[test]
    fn the_floor_sheds_nothing() {
        let mut v = MemoryPressure::default();
        let floor = *FARCLIP_RANGE.start();
        assert_eq!(v.weigh(10.0, 100 * MIB, RUN, floor), None);
        assert_eq!(v.weigh(10.0, 100 * MIB, RUN, 200.0), Some(floor));
        assert_eq!(v.weigh(20.0, 100 * MIB, RUN, floor), None);
    }
}
