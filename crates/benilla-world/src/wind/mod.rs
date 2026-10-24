//! Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module
//! `wxl-experimental-wind`, `field/Wind.hpp` and `field/Wind.cpp`.
//!
//! fork (Everwood port): one stateless wind field shared by every visual system. The field is a pure
//! function of its profile, weather and time; consumers read [`WindField`] rather than inventing
//! their own heading or gust clock. [`FoliageWind`] only gates the foliage receivers.

use bevy::prelude::*;

use crate::lighting::EnvFrame;
use crate::weather::{storm_blend, WeatherState, WeatherTick};
use crate::world_unit::{ViewerUnit, WorldUnit};

const TAU: f64 = std::f64::consts::TAU;
const RATIOS: [f64; 3] = [1.0, 0.437, 0.1913];
const WEIGHTS: [f64; 3] = [0.60, 0.28, 0.12];

/// The WarcraftXL wind profile. Units are yards, seconds and degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindProfile {
    pub base_speed: f32,
    pub heading_deg: f32,
    pub gust: f32,
    pub gust_period: f32,
    pub veer_deg: f32,
    pub veer_period: f32,
    pub weather_coupling: f32,
    pub seed: u32,
}

impl Default for WindProfile {
    fn default() -> Self {
        Self {
            base_speed: 9.0,
            heading_deg: 45.0,
            gust: 0.55,
            gust_period: 17.0,
            veer_deg: 22.0,
            veer_period: 53.0,
            weather_coupling: 1.0,
            seed: 0x5749_4E44,
        }
    }
}

/// One resolved instant of the shared field, in Bevy's horizontal XZ plane.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindSample {
    pub dir: Vec2,
    pub speed: f32,
    pub heading_deg: f32,
    /// The shaped gust envelope, 0 at a lull and 1 at a peak.
    pub gust: f32,
}

/// The public shared wind resource. Future water/cloud consumers read [`Self::sample`].
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct WindField {
    pub profile: WindProfile,
    pub sample: WindSample,
}

impl Default for WindField {
    fn default() -> Self {
        let profile = WindProfile::default();
        Self {
            sample: sample(profile, 0.0, 0.0),
            profile,
        }
    }
}

/// Foliage wind quality: 0 Off, 1 grass, 2 grass + trees.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoliageWind(pub u8);

impl FoliageWind {
    /// The registered `foliageWind` default (grass + trees).
    pub const REGISTERED: u8 = 2;

    /// fork: the capture-only A/B pin `WOW_FOLIAGE_WIND`, read only under
    /// `WOW_CAPTURE` (like the other capture levers) so a player run always obeys the saved CVar.
    /// The app adds the dev-build gate on top.
    pub fn capture_override() -> Option<u8> {
        std::env::var_os("WOW_CAPTURE")?;
        std::env::var("WOW_FOLIAGE_WIND")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 2.0) as u8)
    }
}

/// fork: the player's foliage sway strength (`foliageWindStrength`), a plain gain on
/// the sway amplitude: 0.25..3, 1 = the shipped tuning. Parting around units ignores it.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct FoliageWindStrength(pub f32);

impl FoliageWindStrength {
    pub const MIN: f32 = 0.25;
    pub const MAX: f32 = 3.0;
    /// The registered default.
    pub const REGISTERED: f32 = 1.0;

    /// A CVar value into the slider's range (non-finite reads as the default).
    pub fn from_cvar(v: f32) -> Self {
        Self(if v.is_finite() {
            v.clamp(Self::MIN, Self::MAX)
        } else {
            Self::REGISTERED
        })
    }
}

impl Default for FoliageWindStrength {
    fn default() -> Self {
        Self(Self::REGISTERED)
    }
}

/// fork: a storm swings the foliage harder, up to +60 % at a full storm, on top of the
/// faster travel the storm's wind speed already gives.
pub const STORM_SWAY_GAIN: f32 = 0.6;

impl Default for FoliageWind {
    fn default() -> Self {
        // The resource exists before the CVar host. Its ordinary default therefore has to equal
        // the registered default; a default-valued CVar emits no change event at boot. The
        // capture pin lets the A/B harness force the exact zero-displacement path.
        Self(Self::capture_override().unwrap_or(Self::REGISTERED))
    }
}

fn seeded_unit(seed: u32, index: i32) -> f64 {
    let mut h = seed ^ (index as u32).wrapping_mul(0x9E37_79B9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    f64::from(h & 0x00FF_FFFF) / f64::from(0x0100_0000u32)
}

/// WarcraftXL's three-sine band-limited wander in `[-1, 1]`.
fn wander(seed: u32, lane: i32, period: f32, seconds: f64) -> f32 {
    let period = f64::from(period.max(0.05));
    RATIOS
        .into_iter()
        .zip(WEIGHTS)
        .enumerate()
        .map(|(i, (ratio, weight))| {
            let phase = seeded_unit(seed, lane * 97 + i as i32) * TAU;
            let turns = seconds / (period * ratio);
            weight * (turns.fract() * TAU + phase).sin()
        })
        .sum::<f64>() as f32
}

/// Resolves the pure wind field. `storm` is 0 for clear weather and 1 for a full storm.
pub fn sample(profile: WindProfile, seconds: f64, storm: f32) -> WindSample {
    let raw = wander(profile.seed, 0, profile.gust_period, seconds);
    let gust = ((raw + 1.0) * 0.5).powf(1.7);
    let gust_gain = 1.0 + profile.gust * (gust - 0.5) * 2.0;
    let weather_gain = 1.0 + storm.clamp(0.0, 1.0) * 1.1 * profile.weather_coupling;
    let speed = (profile.base_speed * gust_gain * weather_gain).max(0.0);
    let heading_deg = profile.heading_deg
        + profile.veer_deg * wander(profile.seed, 1, profile.veer_period, seconds);
    let heading = heading_deg.to_radians();
    WindSample {
        dir: Vec2::new(heading.cos(), heading.sin()),
        speed,
        heading_deg,
        gust,
    }
}

/// fork: the wrap of the published wind travel, in yards. Every wave rate in
/// `wind_hook.wgsl` is a whole number of cycles per wrap, so the wrap is invisible.
pub const TRAVEL_WRAP: f64 = 4096.0;

/// fork: the integrated travel. The waves' phase must be the integral of the speed,
/// not `speed(t) * t` (whose rate grows with session age and flickered after a few minutes).
#[derive(Default)]
struct WindClock {
    travel: f64,
    last: Option<f64>,
}

/// Capture-only clock offset (`$WOW_CAPTURE_WIND_T`, seconds): photographs the field as it is
/// that long into a session. Ignored outside `WOW_CAPTURE`.
fn capture_wind_offset() -> f64 {
    if std::env::var_os("WOW_CAPTURE").is_none() {
        return 0.0;
    }
    std::env::var("WOW_CAPTURE_WIND_T")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0)
}

/// Integrates the travel from `from` to `to` seconds at a fixed 1/60 s step.
fn integrate_travel(profile: WindProfile, storm: f32, from: f64, to: f64) -> f64 {
    let mut travel = 0.0;
    let mut t = from;
    while t < to {
        let dt = (to - t).min(1.0 / 60.0);
        travel += f64::from(sample(profile, t, storm).speed) * dt;
        t += dt;
    }
    travel
}

/// fork: resolve once at the frame boundary, then publish the same sample and clock to
/// Rust consumers and the `EnvFrame` block. A zero tier clears every visual strength.
fn update_wind(
    time: Res<Time>,
    weather: Res<WeatherState>,
    quality: Res<FoliageWind>,
    strength: Option<Res<FoliageWindStrength>>,
    mut wind: ResMut<WindField>,
    mut frame: ResMut<EnvFrame>,
    mut clock: Local<WindClock>,
    mut offset: Local<Option<f64>>,
    mut logged_tier: Local<Option<u8>>,
) {
    // fork: name the live tier in the log (boot and every change), so a player run
    // shows what the foliage receivers were given.
    if *logged_tier != Some(quality.0) {
        *logged_tier = Some(quality.0);
        info!(
            "wind: foliageWind tier {} (grass {}, trees {})",
            quality.0,
            quality.0 >= 1,
            quality.0 >= 2
        );
    }
    let offset = *offset.get_or_insert_with(capture_wind_offset);
    let seconds = time.elapsed_secs_f64() + offset;
    let storm = storm_blend(weather.sky_density);
    wind.sample = sample(wind.profile, seconds, storm);

    // fork: integrate the travel on the CPU in f64 and publish it wrapped.
    let step = match clock.last {
        // First frame: catch up from session start (only non-zero under a capture offset).
        None => integrate_travel(wind.profile, storm, 0.0, seconds),
        Some(last) => f64::from(wind.sample.speed) * (seconds - last).max(0.0),
    };
    clock.last = Some(seconds);
    clock.travel = (clock.travel + step).rem_euclid(TRAVEL_WRAP);

    frame.wind_dir = wind.sample.dir.to_array();
    frame.wind_base_heading = wind.profile.heading_deg.to_radians();
    frame.wind_gust = wind.sample.gust;
    frame.wind_travel = clock.travel as f32;
    // fork: `sway_strength` carries the player's gain and the storm swing; the tier
    // flags stay 0/1 (the grass parting reads only the grass flag, so it keeps its size).
    let gain = strength.map_or(FoliageWindStrength::REGISTERED, |s| {
        FoliageWindStrength::from_cvar(s.0).0
    });
    frame.sway_strength = if quality.0 > 0 {
        gain * (1.0 + STORM_SWAY_GAIN * storm)
    } else {
        0.0
    };
    frame.grass_strength = if quality.0 >= 1 { 1.0 } else { 0.0 };
    frame.tree_strength = if quality.0 >= 2 { 1.0 } else { 0.0 };
}

/// fork: player first, then the seven nearest streamed units in the 40-yard bubble.
/// EnvFrame stores Bevy world coordinates and a per-body 1.5–2 yard radius.
fn update_benders(
    quality: Res<FoliageWind>,
    viewer: Res<crate::view::Viewer>,
    // fork: `GlobalTransform` (last propagation), not `Transform`, so the scan is
    // not an undeclared order against every unit mover in `Update`.
    viewer_unit: Query<&GlobalTransform, (With<ViewerUnit>, With<WorldUnit>)>,
    units: Query<(&GlobalTransform, &WorldUnit), Without<ViewerUnit>>,
    mut frame: ResMut<EnvFrame>,
) {
    frame.bender_count = 0;
    if quality.0 == 0 {
        return;
    }
    // Capture/world-viewer fixtures have no live Player resource, but can still publish a real
    // ViewerUnit. In play `Viewer::at` remains authoritative; the entity transform is only the
    // no-avatar fallback that lets the parting instrument exercise the same receiver.
    let Some(player) = viewer
        .at
        .or_else(|| viewer_unit.single().ok().map(|t| t.translation()))
    else {
        return;
    };

    frame.benders[0] = [player.x, player.y, player.z, 1.75];
    frame.bender_count = 1;
    // Fixed insertion list: this is a per-frame scan, so do not allocate and sort a Vec of every
    // unit just to retain seven entries.
    let mut nearby = [(f32::INFINITY, Vec3::ZERO, 1.5); 7];
    for (transform, unit) in &units {
        let at = transform.translation();
        let d2 = (at.xz() - player.xz()).length_squared();
        if d2 > 40.0 * 40.0 || d2 >= nearby[6].0 {
            continue;
        }
        let mut slot = 6;
        while slot > 0 && d2 < nearby[slot - 1].0 {
            nearby[slot] = nearby[slot - 1];
            slot -= 1;
        }
        nearby[slot] = (d2, at, (1.5 * unit.scale).clamp(1.5, 2.0));
    }
    for (slot, (_, at, radius)) in nearby
        .into_iter()
        .take_while(|(d2, _, _)| d2.is_finite())
        .enumerate()
    {
        frame.benders[slot + 1] = [at.x, at.y, at.z, radius];
        frame.bender_count += 1;
    }
}

/// fork: the wind writers' set, so other `EnvFrame` writers (wetness) and the
/// viewer publish can order against it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindTick;

/// Installs the shared field after this frame's weather ramp has resolved.
pub struct WindPlugin;

impl Plugin for WindPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WindField>()
            .init_resource::<FoliageWind>()
            .init_resource::<FoliageWindStrength>()
            // fork: one ordered writer run of `EnvFrame`'s wind rows, before
            // the lighting resolve (the fog model writes the same resource there).
            .add_systems(
                Update,
                (update_wind, update_benders)
                    .chain()
                    .in_set(WindTick)
                    .after(WeatherTick)
                    // The camera pose copy also writes `GlobalTransform` (on the camera only).
                    .after(crate::view::publish_camera_pose)
                    .before(crate::lighting::LightingResolveSet),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_warcraft_xl_profile() {
        let p = WindProfile::default();
        assert_eq!(p.base_speed, 9.0);
        assert_eq!(p.heading_deg, 45.0);
        assert_eq!(p.gust, 0.55);
        assert_eq!(p.gust_period, 17.0);
        assert_eq!(p.veer_deg, 22.0);
        assert_eq!(p.veer_period, 53.0);
    }

    #[test]
    fn field_is_stateless_and_weather_only_scales_speed() {
        let p = WindProfile::default();
        let clear = sample(p, 123.456, 0.0);
        assert_eq!(clear, sample(p, 123.456, 0.0));
        let storm = sample(p, 123.456, 1.0);
        assert_eq!(clear.dir, storm.dir);
        assert_eq!(clear.gust, storm.gust);
        assert!((storm.speed / clear.speed - 2.1).abs() < 1.0e-5);
    }

    #[test]
    fn direction_is_unit_length_and_gust_is_bounded() {
        let p = WindProfile::default();
        for t in [0.0, 1.0, 17.0, 53.0, 86_400.0] {
            let s = sample(p, t, 0.0);
            assert!((s.dir.length() - 1.0).abs() < 1.0e-5);
            assert!((0.0..=1.0).contains(&s.gust));
            assert!(s.speed >= 0.0);
        }
    }

    /// fork: the player path — no capture pin, the registered tier — publishes live
    /// grass and tree rows every frame, and the travel keeps moving; tier 0 clears them.
    #[test]
    fn default_tier_publishes_moving_grass_and_tree_rows() {
        use bevy::ecs::system::RunSystemOnce;
        use std::time::Duration;
        assert_eq!(FoliageWind::REGISTERED, 2);
        let mut world = World::new();
        world.insert_resource(Time::<()>::default());
        world.insert_resource(WeatherState::default());
        world.insert_resource(FoliageWind(FoliageWind::REGISTERED));
        world.insert_resource(WindField::default());
        world.insert_resource(EnvFrame::default());
        let mut travel = Vec::new();
        for _ in 0..3 {
            world
                .resource_mut::<Time>()
                .advance_by(Duration::from_millis(500));
            world.run_system_once(update_wind).unwrap();
            let rows = world.resource::<EnvFrame>().pack();
            assert_eq!(rows[1][1..], [1.0, 1.0, 1.0], "sway, grass, tree strength");
            travel.push(rows[1][0]);
        }
        assert!(travel[0] < travel[1] && travel[1] < travel[2], "{travel:?}");
        // The player's strength scales the sway only; a storm swings it harder still.
        world.insert_resource(FoliageWindStrength::from_cvar(2.0));
        world.run_system_once(update_wind).unwrap();
        assert_eq!(world.resource::<EnvFrame>().pack()[1][1..], [2.0, 1.0, 1.0]);
        world.resource_mut::<WeatherState>().sky_density = 1.0;
        world.run_system_once(update_wind).unwrap();
        let sway = world.resource::<EnvFrame>().sway_strength;
        assert!(
            (sway - 2.0 * (1.0 + STORM_SWAY_GAIN)).abs() < 1.0e-6,
            "{sway}"
        );
        world.insert_resource(FoliageWind(0));
        world.run_system_once(update_wind).unwrap();
        let rows = world.resource::<EnvFrame>().pack();
        assert_eq!(rows[1][1..], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn strength_clamps_to_the_slider_range() {
        assert_eq!(FoliageWindStrength::from_cvar(0.0).0, 0.25);
        assert_eq!(FoliageWindStrength::from_cvar(9.0).0, 3.0);
        assert_eq!(FoliageWindStrength::from_cvar(f32::NAN).0, 1.0);
        assert_eq!(FoliageWindStrength::default().0, 1.0);
    }

    #[test]
    fn travel_rate_stays_bounded_late_in_a_session() {
        // fork: the published travel advances at the wind speed, never speed x age.
        let p = WindProfile::default();
        for t0 in [10.0, 600.0, 36_000.0] {
            let d = integrate_travel(p, 0.0, t0, t0 + 1.0);
            assert!(
                d > 0.0 && d < 2.2 * f64::from(p.base_speed),
                "t0={t0} d={d}"
            );
        }
    }
}
