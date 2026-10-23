//! The live pools as points on the shared effect stream: one record per drop, splash and flake,
//! each built into its streak, patter or sprite by the vertex stage (`effect_points.wgsl`), with
//! the shapes' numbers.
//!
//! fork: perf — the triangles and quads were built here, on the CPU, three or four 36-byte
//! vertices a particle every frame.

use crate::particles::buffer::{EffectPoint, EffectShape};

use super::pool::{Drop, Patter};
use super::*;

/// The streak triangle (`0x80ff78`, `0x80ff74`), its apex leaned by the wind's `tilt`; fork: on
/// a modern frame it shrinks away toward the eye (`modern::weather::streak_near`).
pub(super) fn streak_shape(tilt: Quat) -> EffectShape {
    EffectShape::Streak {
        half_w: STREAK_HALF_W,
        tail: STREAK_TAIL,
        near: crate::modern::weather::streak_near(),
        tilt,
    }
}

/// The patter triangle (`0x80e004`, `0x803568`) over its 0.25 s (`0x675280`).
pub(super) const PATTER_SHAPE: EffectShape = EffectShape::Patter {
    right: PATTER_RIGHT,
    up: PATTER_UP,
    life: PATTER_LIFE,
};

/// The flake sprite: `snowpoint.bls`'s point size, in pixels of the era's screen
/// ([`SNOW_PX_REF_HEIGHT`]) under a vertical fov of `fov_y`.
pub(super) fn flake_shape(fov_y: f32) -> EffectShape {
    EffectShape::Flake {
        px_at_eye: SNOW_PX_AT_EYE,
        falloff: SNOW_PX_FALLOFF,
        px_min: SNOW_PX_MIN,
        world_per_px: (fov_y * 0.5).tan() / SNOW_PX_REF_HEIGHT,
    }
}

/// Falling-drop streaks: one point per drop, its velocity the streak's axis.
pub(super) fn push_streaks(out: &mut Vec<EffectPoint>, drops: &[Drop]) {
    out.extend(drops.iter().take(POOL).map(|d| EffectPoint {
        pos: d.pos.to_array(),
        a: 0.0,
        v: d.vel.to_array(),
        b: 0.0,
    }));
}

/// Ground patters: one point per splash, its age and its atlas row.
pub(super) fn push_patters(out: &mut Vec<EffectPoint>, patters: &[Patter]) {
    out.extend(patters.iter().take(GROUND_CAP).map(|p| EffectPoint {
        pos: p.pos.to_array(),
        a: p.age,
        v: [0.0; 3],
        b: f32::from(p.variant),
    }));
}

/// Snow flakes: the point-sprite leg `0x678610`, as quads since wgpu has no point size. Per
/// `snowpoint.bls`: white, alpha `clamp01(t − f1)` while falling and `clamp01(1 − 4·(t − f2))`
/// settled, the whole texture per flake (`GL_COORD_REPLACE`). There is no per-flake size: the
/// 32-byte record has no spare field and none of the 5 spawn draws is one.
pub(super) fn push_flakes(out: &mut Vec<EffectPoint>, drops: &[Drop], settled: &[Patter]) {
    let flake = |pos: Vec3, alpha: f32| EffectPoint {
        pos: pos.to_array(),
        a: alpha,
        v: [0.0; 3],
        b: 0.0,
    };
    out.extend(
        drops
            .iter()
            .take(POOL)
            .map(|d| flake(d.pos, (d.age / SNOW_FADE_IN).clamp(0.0, 1.0))),
    );
    out.extend(
        settled
            .iter()
            .take(GROUND_CAP)
            .map(|s| flake(s.pos, 1.0 - (s.age / SNOW_SETTLE_LIFE).clamp(0.0, 1.0))),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drop_at(pos: Vec3, vel: Vec3, age: f32) -> Drop {
        Drop {
            pos,
            vel,
            land_y: -1000.0,
            cell: (0, 0),
            age,
        }
    }

    /// A drop is one point: where it is and how it falls. The triangle is the shader's.
    #[test]
    fn a_drop_is_one_point_with_its_velocity() {
        let mut out = Vec::new();
        let (pos, vel) = (Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, -30.0, 0.25));
        push_streaks(
            &mut out,
            &[drop_at(pos, vel, 0.0), drop_at(pos * 2.0, vel, 0.0)],
        );
        assert_eq!(out.len(), 2, "one point a drop, where three vertices were");
        assert_eq!(out[0].pos, pos.to_array());
        assert_eq!(out[0].v, vel.to_array());
        assert_eq!(std::mem::size_of::<EffectPoint>(), 32);
    }

    /// A splash carries its age and its atlas row; the frame step is the shader's, over the
    /// shape's 0.25 s.
    #[test]
    fn a_patter_carries_its_age_and_atlas_row() {
        let mut out = Vec::new();
        push_patters(
            &mut out,
            &[Patter {
                pos: Vec3::new(4.0, 5.0, 6.0),
                age: 0.125,
                variant: 3,
            }],
        );
        assert_eq!((out[0].a, out[0].b), (0.125, 3.0));
        assert_eq!(
            PATTER_SHAPE,
            EffectShape::Patter {
                right: 1.0 / 12.0,
                up: 1.0 / 6.0,
                life: 0.25,
            }
        );
    }

    /// Alpha `clamp01(t − f1)` while falling, then `clamp01(1 − 4·(t − f2))` settled.
    #[test]
    fn flake_alpha_fades_in_then_out() {
        let at = Vec3::new(0.0, 0.0, -10.0);
        let alpha_of = |drops: &[Drop], settled: &[Patter]| {
            let mut out = Vec::new();
            push_flakes(&mut out, drops, settled);
            out[0].a
        };
        let falling = |age| [drop_at(at, Vec3::NEG_Y, age)];
        assert!(
            (alpha_of(&falling(0.0), &[]) - 0.0).abs() < 1e-6,
            "born invisible"
        );
        assert!((alpha_of(&falling(0.5), &[]) - 0.5).abs() < 1e-6);
        assert!((alpha_of(&falling(1.0), &[]) - 1.0).abs() < 1e-6);
        assert!(
            (alpha_of(&falling(4.0), &[]) - 1.0).abs() < 1e-6,
            "held, not overshooting"
        );
        let settled = |age| {
            vec![Patter {
                pos: at,
                age,
                variant: 0,
            }]
        };
        assert!((alpha_of(&[], &settled(0.0)) - 1.0).abs() < 1e-6);
        assert!((alpha_of(&[], &settled(0.125)) - 0.5).abs() < 1e-6);
        assert!((alpha_of(&[], &settled(0.25)) - 0.0).abs() < 1e-6);
    }

    /// The flake shape carries `snowpoint.bls`'s `max(1, 14·clamp01(1 − 0.02·d))` and a pixel of
    /// an 800-px-tall screen: at view depth `z` a pixel is `z·tan(fov/2)/800` world units, so a
    /// flake keeps its share of the screen whatever the live resolution.
    #[test]
    fn the_flake_shape_is_the_references_point_size_in_era_pixels() {
        let fov = std::f32::consts::FRAC_PI_4;
        let EffectShape::Flake {
            px_at_eye,
            falloff,
            px_min,
            world_per_px,
        } = flake_shape(fov)
        else {
            panic!("the flake shape");
        };
        assert_eq!((px_at_eye, falloff, px_min), (14.0, 0.02, 1.0));
        // A screen 800 px tall spans `2·z·tan(fov/2)` world units at depth z, and the quad's
        // half-extent is `px·z·world_per_px`: a `px`-pixel flake is `px/800` of the height.
        assert!((world_per_px * 800.0 - (fov * 0.5).tan()).abs() < 1e-6);
        // The law the shader runs with these, at the distances the reference names: 12 px at
        // 7.14 yd, the 1 px floor from 46.43 yd, linear between, not `1/d`.
        let px = |d: f32| (px_at_eye * (1.0 - falloff * d).clamp(0.0, 1.0)).max(px_min);
        assert!((px(0.0) - 14.0).abs() < 1e-6);
        assert!((px(7.142_857) - 12.0).abs() < 1e-4);
        assert!((px(25.0) - 7.0).abs() < 1e-4, "the linear midpoint");
        assert!((px(46.428_57) - 1.0).abs() < 1e-4);
        assert!((px(200.0) - 1.0).abs() < 1e-6, "flat past the floor");
    }

    /// The streak's fixed world size (`0x80ff78`, `0x80ff74`) and the wind's lean ride the shape.
    #[test]
    fn the_streak_shape_carries_the_references_size_and_the_tilt() {
        let tilt = Quat::from_rotation_x(0.3);
        assert_eq!(
            streak_shape(tilt),
            EffectShape::Streak {
                half_w: 0.05,
                tail: 2.0,
                near: 0.0,
                tilt,
            }
        );
    }
}
