//! **The modern sun's shadow map** (fork-owned; `FORK.md` P2a).
//!
//! One Bevy `DirectionalLight`, used only as the *container*: it owns the cascade split, the
//! per-cascade light views and the depth array that the existing receivers (`terrain.wgsl`,
//! `wow_model.wgsl`, `static_gx.wgsl`, via `bevy_pbr::shadows`) already sample. It lives on a
//! private render layer that holds no meshes, so Bevy's own shadow pass queues nothing — the WoW
//! materials never enter its prepass pipelines (the failure `player_shadow.rs` documents). The
//! casters are drawn into its cascade views by our own nodes (`static_gx::render::modern_shadow`).
//!
//! P2b replaces the container with custom cascades that cache static casters.

use bevy::camera::visibility::RenderLayers;
use bevy::light::{CascadeShadowConfigBuilder, DirectionalLight, DirectionalLightShadowMap};
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use super::{ActiveLighting, LightingQuality};
use crate::lighting::WowLighting;
use crate::view::WorldCamera;

/// The private layer the modern sun and the world camera share. Layer 31 is `player_shadow`'s.
pub const MODERN_SHADOW_LAYER: usize = 30;

/// Re-aim only after the key light has turned this far (radians): a cascade re-fit on every
/// sub-texel turn makes the edges crawl.
const SUN_SNAP_RADIANS: f32 = 1.5e-3;

#[derive(Component)]
pub struct ModernSun;

#[derive(Component)]
struct ModernShadowCameraLayer;

pub(super) fn plugin(app: &mut App) {
    app.add_systems(
        PostUpdate,
        sync_modern_sun.before(TransformSystems::Propagate),
    );
}

/// `(cascades, first cascade far bound, maximum distance yd, map size)` per tier.
pub fn cascade_tier(quality: LightingQuality) -> (usize, f32, f32, usize) {
    match quality {
        LightingQuality::Low => (2, 15.0, 60.0, 1024),
        LightingQuality::Medium => (3, 16.0, 180.0, 2048),
        LightingQuality::High => (4, 16.0, 320.0, 2048),
        LightingQuality::Ultra => (4, 20.0, 450.0, 4096),
    }
}

#[allow(clippy::type_complexity)]
fn sync_modern_sun(
    active: Res<ActiveLighting>,
    lighting: Res<WowLighting>,
    mut commands: Commands,
    mut suns: Query<&mut Transform, With<ModernSun>>,
    mut cameras: Query<
        (Entity, Option<&mut RenderLayers>),
        (With<WorldCamera>, Without<ModernShadowCameraLayer>),
    >,
    mut map: ResMut<DirectionalLightShadowMap>,
    mut aimed: Local<Option<Vec3>>,
) {
    if !active.is_modern() {
        return;
    }
    for (entity, layers) in &mut cameras {
        let with = layers
            .as_deref()
            .cloned()
            .unwrap_or_default()
            .with(MODERN_SHADOW_LAYER);
        commands.entity(entity).insert((with, ModernShadowCameraLayer));
    }

    let dir = lighting.sun_dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return;
    }
    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let aim = Transform::IDENTITY.looking_to(dir, up);

    let Ok(mut tf) = suns.single_mut() else {
        let (cascades, first, max, size) = cascade_tier(active.quality);
        if map.size != size {
            map.size = size;
        }
        commands.spawn((
            DirectionalLight {
                // Container only: the WoW lanes light themselves from the shared light buffer.
                illuminance: 0.0,
                shadows_enabled: true,
                shadow_depth_bias: 0.02,
                shadow_normal_bias: 0.8,
                ..default()
            },
            CascadeShadowConfigBuilder {
                num_cascades: cascades,
                minimum_distance: 0.1,
                first_cascade_far_bound: first,
                maximum_distance: max,
                ..default()
            }
            .build(),
            aim,
            Visibility::Visible,
            RenderLayers::layer(MODERN_SHADOW_LAYER),
            ModernSun,
        ));
        *aimed = Some(dir);
        return;
    };
    let turned = aimed.is_none_or(|prev| prev.angle_between(dir) > SUN_SNAP_RADIANS);
    if turned {
        *tf = aim;
        *aimed = Some(dir);
    }
}
