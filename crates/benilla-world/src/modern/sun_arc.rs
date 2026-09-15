//! **The real sun arc** (fork-owned; `FORK.md` P2).
//!
//! The 1.12.1 lighting sun never sets: a fixed 225° bearing, 20–37° elevation all day. The visible
//! disc (`celestial_sun_direction`) does rise and set, but only along one compass bearing. A
//! modern session keeps the reference's *elevation timing* — so the DBC colours, which are keyed
//! to it, stay in step — and sweeps the bearing east → south → west over the day. The moon takes
//! the mirrored arc by night. The key light (`WowLighting::sun_dir`, which the light buffer, the
//! classic lanes' N·L and the shadow cascades all read) follows whichever body is up.

use std::f32::consts::PI;

use bevy::prelude::*;

use benilla_assets::coords::wow_to_bevy;

use crate::lighting::{GameClock, LightingResolveSet, WowLighting};

/// 06:30 and 21:30 — the reference's dawn and dusk (the sky warp's peaks).
pub const SUNRISE_MINUTE: f32 = 390.0;
pub const SUNSET_MINUTE: f32 = 1290.0;
/// Cap on the arc's elevation, so midday shadows keep some length.
pub const MAX_ELEVATION: f32 = 62.0 * PI / 180.0;
/// Floor on the KEY light's elevation: a body on the horizon would stretch shadows to infinity.
pub const MIN_KEY_ELEVATION: f32 = 4.0 * PI / 180.0;
/// Below this elevation the sun hands the key to the moon.
const HANDOFF_ELEVATION: f32 = -2.0 * PI / 180.0;

pub(super) fn plugin(app: &mut App) {
    app.add_systems(Update, apply_sun_arc.after(LightingResolveSet));
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

/// The moon's progress along its mirrored arc (0 at sunset, 1 at sunrise, clamped).
pub fn night_phase(minute: f32) -> f32 {
    let night = 1440.0 - (SUNSET_MINUTE - SUNRISE_MINUTE);
    ((minute - SUNSET_MINUTE).rem_euclid(1440.0) / night).clamp(0.0, 1.0)
}

fn apply_sun_arc(clock: Option<Res<GameClock>>, mut lighting: ResMut<WowLighting>) {
    if !super::frame_is_modern() {
        return;
    }
    let Some(clock) = clock else {
        return;
    };
    let minute = clock.minute as f32;
    let sun_elev = lighting.celestial_dir.normalize_or_zero().y.clamp(-1.0, 1.0).asin();
    let moon_elev = lighting.moon_dir_white.normalize_or_zero().y.clamp(-1.0, 1.0).asin();

    let sun = arc_to_body(day_phase(minute), sun_elev);
    let moon = arc_to_body(night_phase(minute), moon_elev);
    lighting.celestial_dir = sun;
    lighting.moon_dir_white = moon;

    let to_key = if sun_elev > HANDOFF_ELEVATION {
        arc_to_body(day_phase(minute), sun_elev.max(MIN_KEY_ELEVATION))
    } else {
        arc_to_body(night_phase(minute), moon_elev.max(MIN_KEY_ELEVATION))
    };
    // `sun_dir` is the direction light TRAVELS.
    lighting.sun_dir = -to_key;
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

    #[test]
    fn elevation_is_capped_and_phases_clamp() {
        assert!((arc_to_body(0.5, 1.5).y - MAX_ELEVATION.sin()).abs() < 1e-4);
        assert_eq!(day_phase(0.0), 0.0);
        assert_eq!(day_phase(1439.0), 1.0);
        assert!((day_phase((SUNRISE_MINUTE + SUNSET_MINUTE) / 2.0) - 0.5).abs() < 1e-6);
        assert!(night_phase(SUNSET_MINUTE) < 1e-6);
        assert!((night_phase(SUNRISE_MINUTE - 1e-3) - 1.0).abs() < 1e-3);
    }
}
