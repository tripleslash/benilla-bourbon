//! **The real sun arc** (fork-owned; `FORK.md` P2).
//!
//! The 1.12.1 lighting sun never sets: a fixed 225° bearing, 20–37° elevation all day. The visible
//! disc (`celestial_sun_direction`) does rise and set, but only along one compass bearing. A
//! modern session keeps the reference's *elevation timing* — so the DBC colours, which are keyed
//! to it, stay in step — and sweeps the bearing east → south → west over the day. The moon takes
//! its own arc by night, from the minute the sun's light goes out to the minute it returns
//! ([`MOONRISE_MINUTE`], [`MOONSET_MINUTE`]); the reference moon rises only at 22:18 and sets at
//! 04:00, which left every evening from 20:36 and every pre-dawn without a key light. The key
//! light (`WowLighting::sun_dir`, which the light buffer, the classic lanes' N·L and the shadow
//! cascades all read) follows whichever body lights more. The authored rows at those hours are
//! still the sun's (Elwynn's diffuse is `[255, 110, 0]` at 20:40 and turns blue only near 21:50),
//! so while the moon keys, the key colour and the celestial tint are the zone's own deep-night
//! rows ([`moonlit`]); the sky, clouds and fog keep the afterglow.

use std::f32::consts::PI;

use bevy::prelude::*;

use benilla_assets::coords::wow_to_bevy;

use crate::lighting::{GameClock, LightingConsumeSet, LightingResolveSet, WowLighting};

/// 06:30 and 21:30 — the reference's dawn and dusk (the sky warp's peaks).
pub const SUNRISE_MINUTE: f32 = 390.0;
pub const SUNSET_MINUTE: f32 = 1290.0;
/// Cap on the arc's elevation, so midday shadows keep some length.
pub const MAX_ELEVATION: f32 = 62.0 * PI / 180.0;
/// Floor on the KEY light's elevation: a body on the horizon would stretch shadows to infinity.
pub const MIN_KEY_ELEVATION: f32 = 4.0 * PI / 180.0;
/// The minutes the reference sun's light reaches zero ([`KEY_ZERO_ELEVATION`] on the elevation
/// table `0xce8d64`): 20:36 on the setting leg, 06:07 on the rising one. The moon's arc spans
/// them, so its light grows as the sun's dies and the key never goes dark.
pub const MOONRISE_MINUTE: f32 = 1236.5;
pub const MOONSET_MINUTE: f32 = 366.5;
/// The moon's culmination: high enough to light a valley floor, low enough to cast shadows.
pub const MOON_PEAK_ELEVATION: f32 = 50.0 * PI / 180.0;
/// Where the moon waits by day: the reference's parked 10° below the horizon.
const MOON_PARKED_ELEVATION: f32 = -10.0 * PI / 180.0;
/// The rise's shape: `sin^k` with `k < 1` clears the horizon in minutes (9° after 11) instead of
/// the half hour a plain sine takes, so dusk dims only briefly between the two bodies.
const MOON_RISE_SHAPE: f32 = 0.6;

/// The key light's strength against its body's TRUE elevation (the key's direction is floored at
/// [`MIN_KEY_ELEVATION`], its energy is not): full from 8° up, nothing once the body is below the
/// horizon. Without it a moon that had already set still lit the night almost horizontally —
/// through mountain ranges no cascade holds, and as a dawn-like glow in the haze.
const KEY_FULL_ELEVATION: f32 = 8.0 * PI / 180.0;
const KEY_ZERO_ELEVATION: f32 = -PI / 180.0;

static KEY_HORIZON: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f80_0000);

static MOON_KEY: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The hours the zone look's noon row ramps over (minutes): in from 09:00 to 11:00, out from 14:30
/// to 16:30, so the morning's and the afternoon's rows keep their authored warmth.
const MIDDAY_IN: (f32, f32) = (540.0, 660.0);
const MIDDAY_OUT: (f32, f32) = (870.0, 990.0);

/// How much of `minute` is midday: 1 from 11:00 to 14:30, 0 before 09:00 and after 16:30.
pub fn midday_at(minute: f32) -> f32 {
    let ramp = |(a, b): (f32, f32)| {
        let t = ((minute - a) / (b - a)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    ramp(MIDDAY_IN) * (1.0 - ramp(MIDDAY_OUT))
}

/// The key light's horizon factor (1 in a classic frame, or with the arc switched off).
pub fn key_horizon() -> f32 {
    f32::from_bits(KEY_HORIZON.load(std::sync::atomic::Ordering::Relaxed))
}

/// How much of the key light is the moon's: its horizon factor while it keys, else 0 (0 in a
/// classic frame). The zone look's moonlit night and the shafts ramp in on it.
pub fn moon_key() -> f32 {
    f32::from_bits(MOON_KEY.load(std::sync::atomic::Ordering::Relaxed))
}

fn horizon_factor(elevation: f32) -> f32 {
    let t = ((elevation - KEY_ZERO_ELEVATION) / (KEY_FULL_ELEVATION - KEY_ZERO_ELEVATION))
        .clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The arc's rewrite of the resolved bodies; every `WowLighting` consumer reads after it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SunArcSet;

pub(super) fn plugin(app: &mut App) {
    app.configure_sets(Update, LightingConsumeSet.after(SunArcSet))
        .add_systems(
            Update,
            apply_sun_arc.in_set(SunArcSet).after(LightingResolveSet),
        );
}

/// Unit vector (Bevy axes) toward a body at `phase` along its arc — 0 rising in the east, 0.5 due
/// south, 1 setting in the west — at `elevation` radians (capped at [`MAX_ELEVATION`]).
pub fn arc_to_body(phase: f32, elevation: f32) -> Vec3 {
    let theta = phase.clamp(0.0, 1.0) * PI;
    let e = elevation.min(MAX_ELEVATION);
    // WoW axes: +X north, +Y west, +Z up. East = −Y, south = −X.
    let (c, s) = (theta.cos(), theta.sin());
    let horizontal = [-s, -c];
    wow_to_bevy([horizontal[0] * e.cos(), horizontal[1] * e.cos(), e.sin()]).normalize()
}

/// The sun's progress along its arc at `minute` (0 at sunrise, 1 at sunset, clamped).
pub fn day_phase(minute: f32) -> f32 {
    ((minute - SUNRISE_MINUTE) / (SUNSET_MINUTE - SUNRISE_MINUTE)).clamp(0.0, 1.0)
}

/// The stars' fade (minutes): in over the hour after the sun's light goes out, out over the hour
/// before it returns. The reference star curve (`0xce9a98`) starts only at 22:30 and is full only
/// at midnight, so under the modern night's moon a `Stars.m2` sky (patch-Y's nebulae) stayed at a
/// fifth of its strength until 23:00.
const STARS_FADE: f32 = 60.0;

/// The star field's alpha in a modern frame at `minute`: never below the reference curve's
/// `reference`, full from an hour after moonrise to an hour before moonset.
pub fn star_alpha(minute: f32, reference: f32) -> f32 {
    reference.max(night_at(minute))
}

/// How much of `minute` is night: 0 by day, in over the [`STARS_FADE`] after moonrise, 1 until
/// the same span before moonset. The stars and the zone look's night rows ramp on it.
pub fn night_at(minute: f32) -> f32 {
    let after_rise = (minute - MOONRISE_MINUTE).rem_euclid(1440.0);
    let before_set = (MOONSET_MINUTE - minute).rem_euclid(1440.0);
    let night = 1440.0 - (MOONRISE_MINUTE - MOONSET_MINUTE);
    if after_rise <= night {
        (after_rise.min(before_set) / STARS_FADE).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Whether `minute` falls between moonrise and moonset.
pub(crate) fn moon_is_up(minute: f32) -> bool {
    minute >= MOONRISE_MINUTE || minute <= MOONSET_MINUTE
}

/// The moon's progress along its arc (0 at moonrise, 1 at moonset, clamped).
pub fn night_phase(minute: f32) -> f32 {
    let night = 1440.0 - (MOONRISE_MINUTE - MOONSET_MINUTE);
    ((minute - MOONRISE_MINUTE).rem_euclid(1440.0) / night).clamp(0.0, 1.0)
}

/// The moon's elevation at `minute` (radians): on the horizon at moonrise and moonset,
/// [`MOON_PEAK_ELEVATION`] at the night's midpoint, parked below the horizon by day.
pub fn moon_elevation(minute: f32) -> f32 {
    if !moon_is_up(minute) {
        return MOON_PARKED_ELEVATION;
    }
    (PI * night_phase(minute))
        .sin()
        .max(0.0)
        .powf(MOON_RISE_SHAPE)
        * MOON_PEAK_ELEVATION
}

/// A colour row under the moon: the row of the hour, moved to the zone's deep-night row by how
/// much of the key is the moon's ([`moon_key`]).
pub fn moonlit(row: [f32; 3], night: [f32; 3], moon_key: f32) -> [f32; 3] {
    let t = moon_key.clamp(0.0, 1.0);
    // Both ends exact: a day frame keeps its row bit for bit, a full moon takes the night's.
    [0, 1, 2].map(|c| row[c] * (1.0 - t) + night[c] * t)
}

fn apply_sun_arc(
    clock: Option<Res<GameClock>>,
    look: Option<Res<super::look::ActiveLook>>,
    mut lighting: ResMut<WowLighting>,
) {
    if !super::frame_is_modern() || !super::sun_arc_enabled() {
        KEY_HORIZON.store(1.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        MOON_KEY.store(0, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    let Some(clock) = clock else {
        return;
    };
    let minute = clock.minute as f32;
    // fork: around noon the key takes the zone look's noon hue; every reader of the row (the
    // surfaces, GI, haze, shafts and clouds) takes it with it.
    if let Some(look) = look {
        lighting.diffuse = look.0.noon_key_row(lighting.diffuse);
    }
    let sun_elev = lighting
        .celestial_dir
        .normalize_or_zero()
        .y
        .clamp(-1.0, 1.0)
        .asin();
    let moon_elev = moon_elevation(minute);
    lighting.star_alpha = star_alpha(minute, lighting.star_alpha);

    let sun = arc_to_body(day_phase(minute), sun_elev);
    let moon = arc_to_body(night_phase(minute), moon_elev);
    lighting.celestial_dir = sun;
    lighting.moon_dir_white = moon;

    // The brighter body keys; at moonrise and moonset both are at zero, so the swap never pops.
    let (sun_light, moon_light) = (horizon_factor(sun_elev), horizon_factor(moon_elev));
    let (to_key, key_light, moon_key) = if sun_light >= moon_light {
        (
            arc_to_body(day_phase(minute), sun_elev.max(MIN_KEY_ELEVATION)),
            sun_light,
            0.0f32,
        )
    } else {
        (
            arc_to_body(night_phase(minute), moon_elev.max(MIN_KEY_ELEVATION)),
            moon_light,
            moon_light,
        )
    };
    MOON_KEY.store(moon_key.to_bits(), std::sync::atomic::Ordering::Relaxed);
    KEY_HORIZON.store(key_light.to_bits(), std::sync::atomic::Ordering::Relaxed);
    // `sun_dir` is the direction light TRAVELS.
    lighting.sun_dir = -to_key;
    // The moon lights with the night's rows, never the sunset's it rises into: the key colour, and
    // the one row behind the specular sheen and the disc tint.
    if moon_key > 0.0 {
        lighting.diffuse = moonlit(lighting.diffuse, lighting.night_key, moon_key);
        lighting.spec = moonlit(lighting.spec, lighting.night_tint, moon_key);
        lighting.celestial_tint = moonlit(lighting.celestial_tint, lighting.night_tint, moon_key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_arc_rises_east_culminates_south_and_sets_west() {
        let e = 30f32.to_radians();
        // Bevy images: WoW east (−Y) → +X, south (−X) → +Z, west (+Y) → −X.
        let rise = arc_to_body(0.0, e);
        let noon = arc_to_body(0.5, e);
        let set = arc_to_body(1.0, e);
        assert!(rise.x > 0.8 && rise.z.abs() < 1e-4, "{rise}");
        assert!(noon.z > 0.8 && noon.x.abs() < 1e-4, "{noon}");
        assert!(set.x < -0.8, "{set}");
        for v in [rise, noon, set] {
            assert!((v.y - e.sin()).abs() < 1e-4);
            assert!((v.length() - 1.0).abs() < 1e-4);
        }
    }

    /// Elwynn at 20:53: the sunset diffuse under a fully risen moon is the midnight row, and a
    /// frame the sun still keys is untouched.
    #[test]
    fn a_risen_moon_lights_with_the_night_row() {
        let dusk = [1.0, 110.0 / 255.0, 0.0];
        let night = [97.0 / 255.0, 130.0 / 255.0, 162.0 / 255.0];
        assert_eq!(moonlit(dusk, night, 0.0), dusk);
        assert_eq!(moonlit(dusk, night, 1.0), night);
        let half = moonlit(dusk, night, 0.5);
        for c in 0..3 {
            assert!((half[c] - (dusk[c] + night[c]) / 2.0).abs() < 1e-6);
        }
        // Full a quarter hour after moonrise, where the authored row is still the sunset's.
        assert!(horizon_factor(moon_elevation(MOONRISE_MINUTE + 15.0)) > 0.99);
    }

    /// The noon key: warm until 09:00, full from 11:00 to 14:30, warm again from 16:30.
    #[test]
    fn midday_ramps_in_by_eleven_and_out_by_half_past_four() {
        for (minute, want) in [
            (0.0, 0.0),
            (540.0, 0.0),
            (660.0, 1.0),
            (690.0, 1.0),
            (870.0, 1.0),
        ] {
            assert_eq!(midday_at(minute), want, "{minute}");
        }
        assert_eq!(midday_at(990.0), 0.0);
        assert_eq!(midday_at(1200.0), 0.0);
        let (rising, falling) = (midday_at(600.0), midday_at(930.0));
        assert!((rising - 0.5).abs() < 1e-6 && (falling - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_set_body_lights_nothing() {
        assert_eq!(horizon_factor(-5f32.to_radians()), 0.0);
        assert_eq!(horizon_factor(20f32.to_radians()), 1.0);
        let low = horizon_factor(3f32.to_radians());
        assert!(low > 0.1 && low < 0.9, "{low}");
    }

    #[test]
    fn elevation_is_capped_and_phases_clamp() {
        assert!((arc_to_body(0.5, 1.5).y - MAX_ELEVATION.sin()).abs() < 1e-4);
        assert_eq!(day_phase(0.0), 0.0);
        assert_eq!(day_phase(1439.0), 1.0);
        assert!((day_phase((SUNRISE_MINUTE + SUNSET_MINUTE) / 2.0) - 0.5).abs() < 1e-6);
        assert!(night_phase(MOONRISE_MINUTE) < 1e-6);
        assert!((night_phase(MOONSET_MINUTE - 1e-3) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn the_stars_come_out_with_the_moon() {
        assert_eq!(star_alpha(MOONRISE_MINUTE, 0.0), 0.0);
        assert!((star_alpha(MOONRISE_MINUTE + 30.0, 0.0) - 0.5).abs() < 1e-4);
        // 22:50, where the reference curve is at a fifth: full.
        assert_eq!(star_alpha(1370.0, 0.22), 1.0);
        assert!((star_alpha(MOONSET_MINUTE - 30.0, 0.0) - 0.5).abs() < 1e-4);
        assert_eq!(star_alpha(720.0, 0.0), 0.0);
        // Never below the reference.
        assert_eq!(star_alpha(720.0, 0.3), 0.3);
    }

    /// The reference sun's elevation (`0xce8d64`: polar 100° at 05:30 and 21:30, 5° across noon).
    fn reference_sun_elevation(minute: f32) -> f32 {
        let dp = minute / 1440.0;
        let polar = if dp < 0.2291667 || dp > 0.8958333 {
            100.0
        } else if dp < 0.4965278 {
            100.0 - 95.0 * (dp - 0.2291667) / (0.4965278 - 0.2291667)
        } else if dp < 0.5034722 {
            5.0
        } else {
            5.0 + 95.0 * (dp - 0.5034722) / (0.8958333 - 0.5034722)
        };
        (90.0f32 - polar).to_radians()
    }

    #[test]
    fn the_moon_rises_as_the_sun_goes_dark_and_sets_as_it_returns() {
        for m in [MOONRISE_MINUTE, MOONSET_MINUTE] {
            let e = reference_sun_elevation(m);
            assert!(
                (e - KEY_ZERO_ELEVATION).abs() < 0.05f32.to_radians(),
                "{m}: {e}"
            );
        }
        // Every minute of the day one of the two bodies lights the key, except at each swap,
        // where both pass through zero together.
        for m in 0..1440 {
            let m = m as f32;
            let sun = horizon_factor(reference_sun_elevation(m));
            let lit = sun.max(horizon_factor(moon_elevation(m)));
            let at_swap = (m - MOONRISE_MINUTE).abs() < 2.0 || (m - MOONSET_MINUTE).abs() < 2.0;
            assert!(at_swap || lit > 0.0, "{m}: key {lit}");
        }
        // The moon is at full light a quarter hour after it rises.
        assert!(horizon_factor(moon_elevation(MOONRISE_MINUTE + 15.0)) > 0.99);
        // The dusk the reference left dark: 21:00 and 22:00 have the moon well up.
        assert!(moon_elevation(1260.0) > 12f32.to_radians());
        assert!(moon_elevation(1320.0) > 25f32.to_radians());
        // Culminates at the night's midpoint, parked by day.
        let mid = (MOONRISE_MINUTE + (1440.0 - MOONRISE_MINUTE + MOONSET_MINUTE) / 2.0) % 1440.0;
        assert!((moon_elevation(mid) - MOON_PEAK_ELEVATION).abs() < 1e-4);
        assert!(moon_elevation(720.0) < 0.0);
    }
}
