//! **The modern sun's cascade shadow map** (fork-owned; `FORK.md` P2).
//!
//! Entirely our own — no Bevy `DirectionalLight`. Putting one in the world, even on a private
//! layer, drags the WoW materials through Bevy's light and specialization bookkeeping, and that
//! panicked twice (`bevy_pbr` `light.rs:1856`, `material.rs:1061`). Instead:
//!
//! - **Assets** with constant handles (`benilla_assets::materials::MODERN_SHADOW_MAP` /
//!   `MODERN_SHADOW_CASCADES`): a `Depth32Float` 2D array with a comparison sampler, and the cascade
//!   block. Materials name the handles at construction, so creation order does not matter.
//! - **Cascades** fitted here on the CPU every frame: bounding sphere per frustum slice (rotation
//!   stable), centre snapped to the texel grid in light space (translation stable), reverse-Z
//!   orthographic reaching [`CASTER_REACH`] toward the sun.
//! - **Rendering** in `static_gx::render::modern_shadow`, which uploads the block, clears each layer
//!   and draws the casters. Receivers sample through `benilla_modern::sunshadow`.

use bevy::asset::RenderAssetUsages;
use bevy::camera::Projection;
use bevy::image::{
    Image, ImageAddressMode, ImageCompareFunction, ImageFilterMode, ImageSampler,
    ImageSamplerDescriptor,
};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{
    BufferUsages, Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::storage::ShaderStorageBuffer;

use benilla_assets::materials::{MODERN_SHADOW_CASCADES, MODERN_SHADOW_MAP};

use super::{ActiveLighting, LightingQuality};
use crate::lighting::WowLighting;
use crate::view::WorldCamera;

/// Layers in the shadow array. A tier uses its first `n` ([`cascade_splits`]).
pub const MAX_CASCADES: usize = 4;
/// Edge of every cascade, in texels.
pub const SHADOW_MAP_SIZE: u32 = 2048;
/// How far toward the sun (yd) beyond a cascade's sphere casters are still captured.
pub const CASTER_REACH: f32 = 300.0;
/// Constant depth bias, in yards along the light.
const DEPTH_BIAS_YD: f32 = 0.15;
/// Normal offset, in texels.
const NORMAL_OFFSET_TEXELS: f32 = 1.5;
/// Re-aim only after the key light has turned this far (radians).
const SUN_SNAP_RADIANS: f32 = 1.5e-3;
/// Size of [`ModernCascades::std430`] — `sunshadow.wgsl`'s `ModernCascades`.
pub const CASCADES_BYTES: usize = 4 * 64 + 4 * 16;

/// Where each cascade ends (view distance, yd), per tier — the plan's tier table.
pub fn cascade_splits(quality: LightingQuality) -> &'static [f32] {
    match quality {
        LightingQuality::Low => &[15.0, 60.0],
        LightingQuality::Medium => &[16.0, 55.0, 180.0],
        LightingQuality::High => &[16.0, 55.0, 160.0, 320.0],
        LightingQuality::Ultra => &[20.0, 70.0, 200.0, 450.0],
    }
}

/// This frame's cascades, extracted for the render world. `count == 0` = no modern shadow.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct ModernCascades {
    pub count: u32,
    pub clip_from_world: [Mat4; MAX_CASCADES],
    pub splits: [f32; MAX_CASCADES],
    pub texel_world: [f32; MAX_CASCADES],
    pub depth_bias: [f32; MAX_CASCADES],
}

impl ModernCascades {
    /// The GPU block, laid out as `benilla_modern::sunshadow::ModernCascades`.
    pub fn std430(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CASCADES_BYTES);
        for m in &self.clip_from_world {
            out.extend_from_slice(bytemuck::cast_slice(&m.to_cols_array()));
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.splits));
        out.extend_from_slice(bytemuck::cast_slice(&self.texel_world));
        out.extend_from_slice(bytemuck::cast_slice(&self.depth_bias));
        let params = [
            self.count as f32,
            1.0 / SHADOW_MAP_SIZE as f32,
            NORMAL_OFFSET_TEXELS,
            0.0,
        ];
        out.extend_from_slice(bytemuck::cast_slice(&params));
        out
    }
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<ModernCascades>()
        .add_plugins(ExtractResourcePlugin::<ModernCascades>::default())
        .add_systems(Startup, create_shadow_assets)
        // `Last`, never `PostUpdate`: Bevy's `check_entities_needing_specialization` sits in PostUpdate
        // unordered against benilla's material writers (model_fade, straddle, zfill, lazy spawns…), so
        // ANY added PostUpdate system can reshuffle the executor order, put a writer after the check,
        // and panic the render world at `bevy_pbr` `material.rs:1061` (bisected 2026-09-15: P1a clean,
        // P2a's PostUpdate system broke classic too). `Last` still sees propagated transforms.
        .add_systems(Last, update_cascades);
}

/// Insert the two constant-handle assets. Always (classic sessions too): the materials bind them
/// either way, and a missing binding would stall their bind groups.
fn create_shadow_assets(
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let mut image = Image::new_uninit(
        Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: MAX_CASCADES as u32,
        },
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        label: Some("modern_shadow_map"),
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        label: Some("modern_shadow_compare".into()),
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        compare: Some(ImageCompareFunction::GreaterEqual),
        ..default()
    });
    let _ = images.insert(MODERN_SHADOW_MAP.id(), image);

    let mut block =
        ShaderStorageBuffer::with_size(CASCADES_BYTES, RenderAssetUsages::RENDER_WORLD);
    block.buffer_description.usage |= BufferUsages::COPY_DST;
    let _ = buffers.insert(MODERN_SHADOW_CASCADES.id(), block);
}

fn update_cascades(
    active: Res<ActiveLighting>,
    lighting: Res<WowLighting>,
    cameras: Query<(&GlobalTransform, &Projection), With<WorldCamera>>,
    mut out: ResMut<ModernCascades>,
    mut aimed: Local<Option<Vec3>>,
) {
    if !active.is_modern() {
        if out.count != 0 {
            *out = ModernCascades::default();
        }
        return;
    }
    let Some((camera, Projection::Perspective(persp))) = cameras.iter().next() else {
        return;
    };
    let dir = lighting.sun_dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return;
    }
    let dir = match *aimed {
        Some(prev) if prev.angle_between(dir) <= SUN_SNAP_RADIANS => prev,
        _ => {
            *aimed = Some(dir);
            dir
        }
    };
    let world_from_view = camera.to_matrix();
    let splits = cascade_splits(active.quality);
    let mut next = ModernCascades {
        count: splits.len() as u32,
        ..default()
    };
    let mut near = persp.near.max(0.1);
    for (i, &far) in splits.iter().enumerate() {
        let fit = fit_cascade(
            world_from_view,
            persp.fov,
            persp.aspect_ratio,
            near,
            far,
            dir,
            SHADOW_MAP_SIZE,
        );
        next.clip_from_world[i] = fit.clip_from_world;
        next.splits[i] = far;
        next.texel_world[i] = fit.texel_world;
        next.depth_bias[i] = fit.depth_bias;
        // Cascades overlap a little so the PCF footprint never falls off an edge.
        near = far * 0.9;
    }
    *out = next;
}

pub struct CascadeFit {
    pub clip_from_world: Mat4,
    pub texel_world: f32,
    pub depth_bias: f32,
}

/// Fit one cascade around the view-frustum slice `[near, far]` for light travelling along `dir`.
pub fn fit_cascade(
    world_from_view: Mat4,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
    dir: Vec3,
    size: u32,
) -> CascadeFit {
    let tan = (fov_y * 0.5).tan();
    let mut corners = [Vec3::ZERO; 8];
    let mut k = 0;
    for d in [near, far] {
        let (hh, hw) = (d * tan, d * tan * aspect);
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            corners[k] = world_from_view.transform_point3(Vec3::new(sx * hw, sy * hh, -d));
            k += 1;
        }
    }
    let center = corners.iter().copied().sum::<Vec3>() / 8.0;
    let radius = corners
        .iter()
        .map(|c| c.distance(center))
        .fold(0.0f32, f32::max)
        .ceil()
        .max(1.0);

    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let light_rot = Mat4::look_to_rh(Vec3::ZERO, dir, up);
    let texel = 2.0 * radius / size as f32;
    let c = light_rot.transform_point3(center);
    let snapped = Vec3::new((c.x / texel).floor() * texel, (c.y / texel).floor() * texel, c.z);
    let light_view = Mat4::from_translation(-snapped) * light_rot;
    // Reverse-Z: the plane `CASTER_REACH` beyond the sphere toward the sun maps to 1, the sphere's
    // far side to 0.
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, radius, -(radius + CASTER_REACH));
    CascadeFit {
        clip_from_world: proj * light_view,
        texel_world: texel,
        depth_bias: DEPTH_BIAS_YD / (2.0 * radius + CASTER_REACH),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_matches_the_wgsl_layout() {
        assert_eq!(ModernCascades::default().std430().len(), CASCADES_BYTES);
        assert_eq!(CASCADES_BYTES, 320);
    }

    #[test]
    fn a_cascade_is_reverse_z_and_centred() {
        let dir = Vec3::new(0.3, -1.0, 0.2).normalize();
        let fit = fit_cascade(Mat4::IDENTITY, 1.2, 1.5, 0.1, 50.0, dir, 2048);
        // The slice centre lands near the middle of the map (within the snap).
        let mid = Vec3::new(0.0, 0.0, -25.05);
        let p = fit.clip_from_world.project_point3(mid);
        assert!(p.x.abs() < 0.01 && p.y.abs() < 0.01, "{p}");
        assert!(p.z > 0.0 && p.z < 1.0, "{p}");
        // A caster toward the sun is nearer the light: larger reverse-Z depth.
        let caster = fit.clip_from_world.project_point3(mid - dir * 20.0);
        assert!(caster.z > p.z, "{caster} vs {p}");
    }
}
