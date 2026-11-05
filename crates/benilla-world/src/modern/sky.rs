//! **The modern sky's inputs** (fork-owned; `FORK.md` P4).
//!
//! The dome's stops are already decoded to linear by `sky.rs`'s hook; this adds what an HDR sky needs
//! beyond them — the sun's direction and glow colour for the Mie halo and the analytic disk core
//! (`sky.wgsl`), and what the cloud layer's view-dependent part needs: the key light for the silver
//! lining, the opacity, the horizon haze and fade (`cloud.wgsl`; the rest is baked, `super::cloud_bake`).
//! Classic frames get zeroed lanes, which both shaders read as "classic".

use bevy::prelude::*;

use super::{frame_is_modern, smoothstep, srgb_to_linear, ActiveLighting};
use crate::clouds::CloudMaterial;
use crate::lighting::WowLighting;
use crate::sky::SkyMaterial;

/// The sun glow's radiance against the celestial tint.
const SUN_GLOW: f32 = 0.9;
/// The Mie halo's g.
const MIE_G: f32 = 0.76;
/// The cloud silver lining's strength and lobe g: a thin edge near the key, not every edge.
const CLOUD_SILVER: f32 = 0.5;
const CLOUD_G: f32 = 0.8;

pub(super) fn plugin(app: &mut App) {
    // After the lighting resolve in the same frame, like the dome's own stop writer.
    app.add_systems(
        Update,
        update_modern_sky.in_set(crate::lighting::LightingConsumeSet),
    );
}

/// Quantized to 1/512, so a drifting clock only re-uploads the materials when a lane visibly moves.
fn quantized(v: Vec4) -> Vec4 {
    (v * 512.0).round() / 512.0
}

fn update_modern_sky(
    active: Res<ActiveLighting>,
    light: Res<WowLighting>,
    domes: Query<&MeshMaterial3d<SkyMaterial>>,
    mut skies: ResMut<Assets<SkyMaterial>>,
    layers: Query<&MeshMaterial3d<CloudMaterial>>,
    mut clouds: ResMut<Assets<CloudMaterial>>,
    look: Res<super::look::ActiveLook>,
) {
    let on = active.is_modern() && frame_is_modern() && super::cpu_hooks_enabled();
    let flag = if on { 1.0 } else { 0.0 };

    let sun = light.celestial_dir.normalize_or(Vec3::Y);
    let above = smoothstep(-0.08, 0.1, sun.y);
    let tint = Vec3::from(light.celestial_tint.map(srgb_to_linear));
    let sky_modern = quantized(sun.extend(flag));
    let glow = if on {
        quantized((tint * Vec3::from(look.0.glow) * (SUN_GLOW * above)).extend(MIE_G))
    } else {
        Vec4::ZERO
    };
    for handle in &domes {
        benilla_assets::write_gated(
            &mut skies,
            &handle.0,
            |m| m.extension.modern != sky_modern || m.extension.glow != glow,
            |m| {
                m.extension.modern = sky_modern;
                m.extension.glow = glow;
            },
        );
    }

    // Clouds follow the key light (sun by day, moon by night); the bake lights them, the shader
    // adds the silver lining toward it.
    let params = super::cloud_bake::cloud_params(&light, &look.0);
    let lit = smoothstep(-0.1, 0.45, params.key.y);
    let cloud_modern = quantized(params.key.extend(flag));
    let (cloud_light, cloud_shape, cloud_key, cloud_haze) = if on {
        (
            quantized(Vec4::new(0.0, CLOUD_SILVER * lit, CLOUD_G, 0.0)),
            quantized(Vec4::new(
                params.style.opacity,
                0.0,
                0.0,
                params.style.horizon,
            )),
            quantized(params.key_col.extend(0.0)),
            quantized(params.haze_col.extend(params.style.haze)),
        )
    } else {
        (Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO)
    };
    for handle in &layers {
        benilla_assets::write_gated(
            &mut clouds,
            &handle.0,
            |m| {
                m.extension.modern != cloud_modern
                    || m.extension.light != cloud_light
                    || m.extension.shape != cloud_shape
                    || m.extension.key != cloud_key
                    || m.extension.haze != cloud_haze
            },
            |m| {
                m.extension.modern = cloud_modern;
                m.extension.light = cloud_light;
                m.extension.shape = cloud_shape;
                m.extension.key = cloud_key;
                m.extension.haze = cloud_haze;
            },
        );
    }
}
