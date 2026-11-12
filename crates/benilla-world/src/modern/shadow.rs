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
use bevy::camera::{CameraProjection, Projection};
use bevy::image::{
    Image, ImageAddressMode, ImageCompareFunction, ImageFilterMode, ImageSampler,
    ImageSamplerDescriptor,
};
use bevy::math::Vec3A;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{
    BufferUsages, Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::storage::ShaderBuffer;

use benilla_assets::materials::{
    LiquidCataMaterial, LiquidMaterial, TerrainMaterial, WowModelMaterial, MODERN_SHADOW_CASCADES,
    MODERN_SHADOW_MAP,
};
use benilla_formats::ModelBlend;
use bevy::camera::primitives::Aabb;

use super::{ActiveLighting, LightingQuality};
use crate::lighting::WowLighting;
use crate::model_render::{ModelPart, ShadowOccluder};
use crate::rig_palette::{RigPart, RigSkin};
use crate::view::WorldCamera;

/// Layers in the shadow array. A tier uses its first `n` ([`cascade_splits`]).
pub const MAX_CASCADES: usize = 4;
/// The sky-occlusion map rides the array layer after the cascades: a vertical light over the static
/// world.
pub const SKY_LAYER: usize = MAX_CASCADES;
/// fork: modern lighting (P6/P8) — the shadowed local lights' cube atlas rides the layers after the
/// sky map, each a 6×6 tile grid holding the six faces of six lights. This is the **most** any tier
/// asks for (P8's Cinematic); [`local_shadow_layers`] is how many the latched tier actually
/// allocates, and the array is re-created to that depth at the latch
/// (`fit_shadow_map`) — a High session pays for one, a Medium session for none.
pub const MAX_LOCAL_SHADOW_LAYERS: usize = 2;
/// The atlas's first layer (equal to [`MAX_CASCADES`] + 1 always; a tier with no atlas simply never
/// names it, which `locallight::atlas_live` reads as "off").
pub const LOCAL_LAYER: usize = MAX_CASCADES + 1;
/// Layers the array is allocated with at the very largest tier — the size every fixed-size
/// per-layer table here is declared at.
pub const SHADOW_LAYERS: usize = MAX_CASCADES + 1 + MAX_LOCAL_SHADOW_LAYERS;

/// Cube-atlas layers the tier wants. A `Depth32Float` 2048² layer is 16.8 MB in the array **and**
/// 16.8 MB again in the render world's static cache, so a tier that shadows no local light must not
/// allocate one — which is exactly what Low and Medium do (`modern::lights::light_tier`).
///
/// This replaced a `cfg!(target_arch = "wasm32")` const that gave the browser none at all, at any
/// tier. **The browser's existing tiers keep exactly that**: P8 must not make a tier that shipped
/// cost more than it did, and a web High that suddenly allocated a 16.8 MB layer and started
/// rendering 24 cube faces would be precisely that. What the browser gains is the new tier, which a
/// player has to go and pick — and which is defined by those layers.
pub fn local_shadow_layers(quality: LightingQuality) -> usize {
    match quality {
        // fork: P8b — Ultra shadows no local light (it is Cinematic's lighting without them), so it
        // allocates no atlas layer either, natively or in the browser.
        // fork: perf — High is Ultra's frame (`LightingQuality::High`), with no atlas layer either.
        LightingQuality::Low
        | LightingQuality::Medium
        | LightingQuality::High
        | LightingQuality::Ultra => 0,
        LightingQuality::Cinematic => MAX_LOCAL_SHADOW_LAYERS,
    }
}

/// fork: F — **the ray-traced shadows' lane, this frame**: the ray tracing is live
/// (`rt_scene::shadows_live()`: native, capable, `rayTracing` and `rayTracingShadows` on and the scene ready) and the
/// session may trace shadows ([`rt_shadows_capable`]). Then the sun's and the shadowed lamps'
/// visibility through the static world and the terrain is a ray query in every receiver
/// (`benilla_modern::rtshadow`), the cascades draw the entity casters alone (the sky map keeps its
/// static ones) and the lamp atlas holds no static face. A per-frame decision: every reader in the
/// render world sees one value a frame (`rt_scene::update_rt_tier`, `PrepareAssets`), and a switch
/// redraws the cascades and the atlas with or without their static casters at once
/// (`static_gx::render::modern_shadow`). False before render startup.
pub fn rt_shadows() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::static_gx::rt_scene::shadows_live() && rt_shadows_capable()
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// fork: **may this session trace shadows at all?** Capable of ray tracing
/// (`rt_scene::capable()`, fixed at render startup) and `$WOW_RT_SHADOWS=0` did not turn the lane
/// off (read once). Where it holds, every modern receiver's pipeline compiles `RT_SHADOWS` and
/// declares the scene for the whole session, and picks the ray or the map path per frame from the
/// group it is bound ([`rt_shadows`], `RtScene::lane_bind`), so a switch never re-specializes.
pub fn rt_shadows_capable() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        crate::static_gx::rt_scene::capable()
            && *CELL
                .get_or_init(|| crate::launch_env::var("WOW_RT_SHADOWS").as_deref() != Some("0"))
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// Layers the array is created with for a session at `quality`: the cascades, the sky map, and the
/// tier's cube-atlas layers.
pub fn shadow_layers(quality: LightingQuality) -> usize {
    MAX_CASCADES + 1 + local_shadow_layers(quality)
}
/// Half the sky map's footprint (yd): 1024 yd across at 0.5 yd a texel.
pub const SKY_HALF_EXTENT: f32 = 512.0;
/// The sky map re-centres on this grid, so it (and its static cache) holds while the camera moves.
const SKY_SNAP: f32 = 32.0;
/// Edge of every cascade, in texels.
pub const SHADOW_MAP_SIZE: u32 = 2048;
/// How far toward the sun (yd) beyond a cascade's sphere casters are still captured.
/// (Was 300: at a low sun a ridge further than that toward the light cast nothing into a far
/// cascade, and the haze and ground behind it lit up.)
pub const CASTER_REACH: f32 = 1200.0;
/// [`CASTER_REACH`], or the dev A/B override `$WOW_MODERN_CASTERREACH` (yd).
fn caster_reach() -> f32 {
    static CELL: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::launch_env::var("WOW_MODERN_CASTERREACH")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|r| *r > 0.0)
            .unwrap_or(CASTER_REACH)
    })
}
/// `sky_params.w` while no water mirror renders: below any eye, so no view takes the mirror's tier.
pub const NO_MIRROR_PLANE: f32 = -1.0e9;

/// fork: perf — render world: the mirror plane that went into the block this frame (`static_gx`'s
/// `prepare_modern_shadow`, where the cascades are uploaded), `None` in a frame that uploaded none.
/// `modern::lights`' `prepare_lights` hands it to the vertex stages beside the grid's live bit
/// ([`VERTEX_HINT_OFFSET`]).
#[derive(Resource, Default)]
pub(crate) struct UploadedMirrorPlane(pub Option<f32>);

/// fork: perf — where the vertex stages' two facts sit in the shared light buffer: row 20's `.z`
/// (1 = the block's light grid is live this frame, and `.w` is valid) and `.w` (the block's mirror
/// plane, `sky_params.w`). `wow_model.wesl` and `terrain.wesl` bind the block fragment-only, and
/// their vertex stages skip a walk of the light table when these say no fragment will read it.
/// `build_light_data` packs both as 0, which reads as "walk".
pub(crate) const VERTEX_HINT_OFFSET: u64 = 20 * 16 + 8;
/// Constant depth bias, in yards along the light.
const DEPTH_BIAS_YD: f32 = 0.15;
/// Yards a rigged caster's box is widened by for the cascade cull (a pose strays from its bind box).
const RIG_CULL_MARGIN: f32 = 2.0;
/// Normal offset, in texels.
const NORMAL_OFFSET_TEXELS: f32 = 1.5;
/// Re-aim only after the key light has turned this far (radians).
const SUN_SNAP_RADIANS: f32 = 1.5e-3;
/// A cascade's sphere is fitted this much larger than its frustum slice and re-centred only when the
/// slice leaves it, so a moving camera keeps most cascades' matrices — and their static cache — for
/// many frames.
const CASCADE_MARGIN: f32 = 1.25;
/// Size of [`ModernCascades::std430`] — `sunshadow.wesl`'s `ModernCascades`.
pub const CASCADES_BYTES: usize = 5 * 64 + 5 * 16;

/// Where each cascade ends (view distance, yd), per tier — the plan's tier table.
pub fn cascade_splits(quality: LightingQuality) -> &'static [f32] {
    // The last cascade reaches most of the view distance: static casters are cached per cascade, so a
    // far cascade costs a redraw only when it re-centres, and without it the sun's shadows (and so
    // the scene's contrast) ended in a band that walked ahead of the camera. Past it the sky map
    // (`sunshadow::far_sun_visibility`) takes over.
    match quality {
        // fork: perf — the tiers below Ultra moved down one (the maintainer's call, 2026-11-12).
        LightingQuality::Low => &[16.0, 60.0, 200.0, 500.0],
        LightingQuality::Medium => &[16.0, 55.0, 200.0, 600.0],
        // fork: perf — High is Ultra's frame (`LightingQuality::High`).
        LightingQuality::High | LightingQuality::Ultra => &[20.0, 70.0, 250.0, 800.0],
        // fork: P8 — the sun is Ultra's. Cinematic buys local lights, not a sixth cascade: the
        // array's depth is what a new cascade would cost, and P8 spends that depth on the cube
        // atlas instead (`local_shadow_layers`).
        LightingQuality::Cinematic => &[20.0, 70.0, 250.0, 800.0],
    }
}

/// This frame's cascades, extracted for the render world. `count == 0` = no modern shadow.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct ModernCascades {
    pub count: u32,
    pub clip_from_world: [Mat4; MAX_CASCADES],
    /// The entity casters' cull per cascade: `clip_from_world` narrowed to the camera's footprint
    /// in the map ([`narrowed_cull`]), or `clip_from_world` itself where nothing narrows it. CPU
    /// only: not in [`Self::std430`].
    pub cull_from_world: [Mat4; MAX_CASCADES],
    pub splits: [f32; MAX_CASCADES],
    pub texel_world: [f32; MAX_CASCADES],
    pub depth_bias: [f32; MAX_CASCADES],
    /// The debug view mode (`params.w`): 1 = sun visibility, 2 = sky visibility.
    pub debug: u8,
    /// The sky-occlusion map's light clip (straight down) and `[enabled, texel yd, depth bias,
    /// mirror plane]` — the last is the water mirror's plane while it renders, else
    /// [`NO_MIRROR_PLANE`] (`sunshadow::mirror_view`).
    pub sky_clip: Mat4,
    pub sky_params: [f32; 4],
}

/// One terrain caster: mesh, world-space box centre and half extents, LOD epoch.
pub type TerrainCaster = (AssetId<Mesh>, Vec3, Vec3, u32);

/// Every resident terrain cell as a caster: mesh, world-space box centre and half extents, and
/// its LOD epoch (`terrain_stream::TerrainLodEpoch`), which moves when the cell's mesh is rebuilt
/// at a new chunk level under the same handle, so the cached shadow layers and the GI height map
/// that hold it are redrawn with the coarse terrain the camera sees. Collected in `Last`; empty in
/// a classic session.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct ModernTerrainCasters {
    pub cells: Vec<TerrainCaster>,
    /// Bumped whenever `cells` changes, so a reader keyed on the set skips re-hashing it.
    pub rev: u64,
}

/// One entity-lane model part that casts: mesh, world matrix (ignored when skinned — the palette
/// replaces it), a bounding sphere for the cascade cull, the rig slot if skinned, and the base
/// texture if its blend is alpha-tested.
#[derive(Clone)]
pub struct EntityCaster {
    pub mesh: AssetId<Mesh>,
    pub world_from_local: Mat4,
    pub center: Vec3,
    pub radius: f32,
    /// fork: the part's own bounding sphere about `center`, WITHOUT the rig margin `radius` carries
    /// for the cascade cull. The lamp atlas's dynamic phase reads it to recognise the model a light
    /// lives inside (a lantern, a brazier, a torch in a fist), which must not shadow its own light.
    pub tight: f32,
    pub rig_slot: Option<u32>,
    pub cutout: Option<AssetId<Image>>,
    /// fork: whether the part casts into the sun's cascades. False when its box swept along the
    /// sun down to the lowest terrain lies wholly under the camera's horizon (`sun_sweep_hidden`):
    /// every point its shadow can land on is then hidden too. The lamp atlas ignores it.
    pub sun: bool,
}

/// Every casting entity-lane part this frame. Collected in `Last`; empty in a classic session.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct ModernEntityCasters {
    pub parts: Vec<EntityCaster>,
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
            f32::from(self.debug),
        ];
        out.extend_from_slice(bytemuck::cast_slice(&params));
        out.extend_from_slice(bytemuck::cast_slice(&self.sky_clip.to_cols_array()));
        out.extend_from_slice(bytemuck::cast_slice(&self.sky_params));
        out
    }
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<ModernCascades>()
        .init_resource::<ModernTerrainCasters>()
        .init_resource::<ModernEntityCasters>()
        .add_plugins(ExtractResourcePlugin::<ModernEntityCasters>::default())
        .add_plugins(ExtractResourcePlugin::<ModernCascades>::default())
        .add_plugins(ExtractResourcePlugin::<ModernTerrainCasters>::default())
        .add_systems(Startup, create_shadow_assets)
        // `Last`, never `PostUpdate`: Bevy's `check_entities_needing_specialization` sits in PostUpdate
        // unordered against benilla's material writers (model_fade, straddle, zfill, lazy spawns…), so
        // ANY added PostUpdate system can reshuffle the executor order, put a writer after the check,
        // and panic the render world at `bevy_pbr` `material.rs:1061` (bisected 2026-09-15: P1a clean,
        // P2a's PostUpdate system broke classic too). `Last` still sees propagated transforms.
        .add_systems(
            Last,
            (
                fit_shadow_map,
                update_cascades,
                collect_terrain_casters,
                collect_entity_casters,
            ),
        );
}

/// The shadow array as a classic session needs it: one texel per layer. The full array is 2048² ×
/// every layer of Depth32Float (84 MB, plus the render world's equally large static cache), and a
/// classic frame never reads a texel of it — but it paid for it from Startup on. Grown and shrunk at
/// the lighting latch by [`fit_shadow_map`].
const CLASSIC_SHADOW_MAP_SIZE: u32 = 1;

fn shadow_map_image(size: u32, layers: usize) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers as u32,
        },
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // COPY_SRC/COPY_DST: the static cache copies layers in and out (`modern_shadow`).
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::RENDER_ATTACHMENT
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST;
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
    image
}

/// Re-size the constant-handle shadow array whenever the latched model or the live tier changes:
/// full size for a modern session, one texel for a classic one, at the tier's own layer count.
/// Only ever on a real change, since it re-prepares every material that binds the handle.
#[allow(clippy::too_many_arguments)]
fn fit_shadow_map(
    active: Res<ActiveLighting>,
    mut images: ResMut<Assets<Image>>,
    // fork: P8 — the DEPTH is latched too now, not just the edge: a tier's cube-atlas layers are
    // 16.8 MB each here and again in the render world's static cache, so the array is created at
    // the tier's own layer count rather than at the largest one any tier could ask for.
    mut fitted: Local<Option<(bool, usize)>>,
    // The four material families that bind the handle. A re-created texture leaves every prepared
    // bind group holding the OLD view, so the ones that already exist have to be touched — the
    // same duty `volumetrics::resize_volume_image` performs for the froxel volume.
    //
    // This used to be somebody else's job by luck rather than by design: the depth only ever moved
    // on a model latch, and that same edge had `sync_model_pipelines` sweeping models and terrain
    // and `resize_volume_image` sweeping models and both liquids, so between them all four got
    // touched. The tier is live now, so a Medium→High move re-creates this texture on a frame
    // where neither of those systems fires at all, and the sweep has to belong to the system that
    // re-created it.
    models: Option<ResMut<Assets<WowModelMaterial>>>,
    terrain: Option<ResMut<Assets<TerrainMaterial>>>,
    liquids: Option<ResMut<Assets<LiquidMaterial>>>,
    cata: Option<ResMut<Assets<LiquidCataMaterial>>>,
) {
    let modern = active.is_modern();
    // A classic session keeps the full layer COUNT at one texel each (28 bytes): nothing samples
    // it, but a layer index a shader names still has to exist in the view it binds.
    let layers = if modern {
        shadow_layers(active.quality)
    } else {
        SHADOW_LAYERS
    };
    if *fitted == Some((modern, layers)) {
        return;
    }
    *fitted = Some((modern, layers));
    let size = if modern {
        SHADOW_MAP_SIZE
    } else {
        CLASSIC_SHADOW_MAP_SIZE
    };
    let current = images.get(MODERN_SHADOW_MAP.id()).map(|i| {
        (
            i.texture_descriptor.size.width,
            i.texture_descriptor.size.depth_or_array_layers as usize,
        )
    });
    if current == Some((size, layers)) {
        return;
    }
    let _ = images.insert(MODERN_SHADOW_MAP.id(), shadow_map_image(size, layers));
    let mut touched = 0;
    if let Some(mut m) = models {
        touched += m.iter_mut().count();
    }
    if let Some(mut m) = terrain {
        touched += m.iter_mut().count();
    }
    if let Some(mut m) = liquids {
        touched += m.iter_mut().count();
    }
    if let Some(mut m) = cata {
        touched += m.iter_mut().count();
    }
    info!("modern shadow: array {size}² × {layers} allocated, {touched} materials rebound");
}

/// Insert the two constant-handle assets. Always (classic sessions too): the materials bind them
/// either way, and a missing binding would stall their bind groups.
fn create_shadow_assets(
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    // Classic-sized until the lighting latches (`fit_shadow_map`).
    let image = shadow_map_image(CLASSIC_SHADOW_MAP_SIZE, SHADOW_LAYERS);
    let _ = images.insert(MODERN_SHADOW_MAP.id(), image);

    // P4: the froxel fog parameters and the cloud-shadow grid ride behind the cascades
    // (`sunshadow::ModernBlock`).
    // P5: the GI parameters and the probe field ride behind the froxel block (`gi::BLOCK_GI_BYTES`),
    // at a size no quality tier changes — a re-created buffer would re-prepare every material.
    // P6: the local-light grid sits between them (`lights::BLOCK_LIGHT_BYTES`), which is what puts
    // both writable sub-ranges — the grid's cells and the probe field — on 256-byte offsets.
    let mut block = ShaderBuffer::with_size(
        (CASCADES_BYTES
            + super::volumetrics::BLOCK_TAIL_BYTES
            + super::gi::GI_PARAMS_BYTES
            + super::lights::BLOCK_LIGHT_BYTES
            + super::gi::BLOCK_PROBE_BYTES) as u64,
        RenderAssetUsages::RENDER_WORLD,
    );
    block.buffer_usage |= BufferUsages::COPY_DST;
    let _ = buffers.insert(MODERN_SHADOW_CASCADES.id(), block);
}

fn update_cascades(
    active: Res<ActiveLighting>,
    lighting: Res<WowLighting>,
    cameras: Query<(&GlobalTransform, &Projection), With<WorldCamera>>,
    mirror: Option<Res<crate::liquid::ReflectionPlane>>,
    mut out: ResMut<ModernCascades>,
    mut aimed: Local<Option<Vec3>>,
    mut spheres: Local<[Option<(Vec3, f32)>; MAX_CASCADES]>,
    mut cascade_dirs: Local<[Option<Vec3>; MAX_CASCADES]>,
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
        debug: super::shadow_debug_mode(),
        ..default()
    };
    // The camera frustum's corners, for the entity casters' narrowed cull. A water mirror's eye
    // sees past the camera's frustum, so while one renders every cascade keeps its full map.
    let narrow = !super::mirror_tier_everywhere()
        && mirror.as_deref().is_none_or(|m| m.live_plane().is_none());
    let corners = narrow.then(|| {
        persp
            .get_frustum_corners(-persp.near, -persp.far)
            .map(|c| world_from_view.transform_point3a(c))
    });
    let mut near = persp.near.max(0.1);
    // fork (perf): a re-aim redraws a cascade's whole static layer, so it reaches one cascade a
    // frame, nearest first, rather than all four (~12 ms on an Iris Xe) in one frame.
    let mut reaimed = false;
    for (i, &far) in splits.iter().enumerate() {
        let dir = match cascade_dirs[i] {
            Some(d) if d == dir || reaimed => d,
            _ => {
                reaimed = true;
                cascade_dirs[i] = Some(dir);
                // A new sun direction re-fits the cascade.
                spheres[i] = None;
                dir
            }
        };
        let (c, r) = slice_sphere(world_from_view, persp.fov, persp.aspect_ratio, near, far);
        let (center, radius) = match spheres[i] {
            Some((sc, sr)) if sc.distance(c) + r <= sr => (sc, sr),
            _ => {
                let fresh = (c, (r * CASCADE_MARGIN).ceil());
                spheres[i] = Some(fresh);
                fresh
            }
        };
        let fit = cascade_from_sphere(center, radius, dir, SHADOW_MAP_SIZE);
        next.clip_from_world[i] = fit.clip_from_world;
        let pad = (CULL_PAD_TEXELS + COVER_RADIUS_YD / fit.texel_world.max(f32::EPSILON)) * 2.0
            / SHADOW_MAP_SIZE as f32;
        next.cull_from_world[i] = corners
            .as_ref()
            .and_then(|c| narrowed_cull(&fit.clip_from_world, c, pad))
            .unwrap_or(fit.clip_from_world);
        next.splits[i] = far;
        next.texel_world[i] = fit.texel_world;
        next.depth_bias[i] = fit.depth_bias;
        // Cascades overlap a little so the PCF footprint never falls off an edge.
        near = far * 0.9;
    }
    // fork: modern lighting (FORK.md, P3) — the sky-occlusion map: the static world seen from straight
    // above, re-centred on a 32 yd grid.
    let cam = camera.translation();
    let sky_center = (cam / SKY_SNAP).round() * SKY_SNAP;
    let sky = cascade_from_sphere(sky_center, SKY_HALF_EXTENT, Vec3::NEG_Y, SHADOW_MAP_SIZE);
    next.sky_clip = sky.clip_from_world;
    // The mirror's plane rides the spare lane: the lanes shade their reflection tier below it.
    let mirror_plane = if super::mirror_tier_everywhere() {
        -NO_MIRROR_PLANE
    } else {
        mirror
            .and_then(|m| m.live_plane())
            .unwrap_or(NO_MIRROR_PLANE)
    };
    next.sky_params = [1.0, sky.texel_world, sky.depth_bias * 4.0, mirror_plane];
    *out = next;
}

/// Terrain cells are world-space meshes under an identity transform, so their `Aabb` is the box.
#[allow(clippy::type_complexity)] // one Bevy system's full input set
fn collect_terrain_casters(
    active: Res<ActiveLighting>,
    cells: Query<
        (
            &Mesh3d,
            &Aabb,
            &GlobalTransform,
            Option<&crate::terrain_stream::TerrainLodEpoch>,
        ),
        With<MeshMaterial3d<TerrainMaterial>>,
    >,
    mut out: ResMut<ModernTerrainCasters>,
    mut scratch: Local<Vec<TerrainCaster>>,
) {
    if !active.is_modern() {
        if !out.cells.is_empty() {
            out.cells.clear();
            out.rev += 1;
        }
        return;
    }
    scratch.clear();
    for (mesh, aabb, gt, epoch) in &cells {
        let center = gt.transform_point(Vec3::from(aabb.center));
        scratch.push((
            mesh.id(),
            center,
            Vec3::from(aabb.half_extents),
            epoch.map_or(0, |e| e.0),
        ));
    }
    // Written only when it moved: the extract copies it and the render side re-scans it on a
    // change, and the resident terrain stands still on most frames.
    if out.cells != *scratch {
        std::mem::swap(&mut out.cells, &mut *scratch);
        out.rev += 1;
    }
}

/// Opaque and alpha-tested parts whose `ShadowOccluder` says they exist as world content (the
/// camera-independent verdict). Ground clutter stays out at this stage (the plan casts it on High+).
#[allow(clippy::type_complexity)]
fn collect_entity_casters(
    active: Res<ActiveLighting>,
    parts: Query<(
        &Mesh3d,
        &MeshMaterial3d<WowModelMaterial>,
        &GlobalTransform,
        &ModelPart,
        &ShadowOccluder,
        Option<&Aabb>,
        Option<&RigPart>,
    )>,
    rigs: Query<&RigSkin>,
    // Worn gear drawn through a row of its wearer's slot.
    riders: Query<&crate::rig_rider::RigRider>,
    materials: Res<Assets<WowModelMaterial>>,
    lighting: Res<WowLighting>,
    cameras: Query<&GlobalTransform, With<WorldCamera>>,
    horizon: Option<Res<crate::horizon::Horizon>>,
    mirror: Option<Res<crate::liquid::ReflectionPlane>>,
    mut out: ResMut<ModernEntityCasters>,
) {
    if !active.is_modern() {
        if !out.parts.is_empty() {
            out.parts.clear();
        }
        return;
    }
    out.parts.clear();
    // Only parts that can reach a cascade: within the last cascade's sphere of the camera, or along the
    // ray toward the key light that far out. Model art stays resident kilometres away (the art scope),
    // and packing, uploading and cascade-testing every one of those parts each frame made the frame
    // slower and slower the further a flight went.
    let Some(cam) = cameras.iter().next().map(GlobalTransform::translation) else {
        return;
    };
    let reach_radius = cascade_splits(active.quality)
        .last()
        .copied()
        .unwrap_or(0.0)
        * CASCADE_MARGIN
        * 1.2;
    let to_light = (-lighting.sun_dir).normalize_or(Vec3::Y);
    let ray_len = caster_reach();
    let near_cascades = |center: Vec3, radius: f32| {
        let t = (center - cam).dot(to_light).clamp(0.0, ray_len);
        center.distance(cam + to_light * t) <= reach_radius + radius
    };
    // The sun lane's horizon test, only while it is enforced and no water mirror renders: the
    // mirror's eye sees past the camera's horizon.
    let horizon = horizon
        .as_deref()
        .filter(|h| h.enforced())
        .filter(|_| mirror.as_deref().is_none_or(|m| m.live_plane().is_none()));
    let sweep = Sweep {
        dir: -to_light,
        cam,
        reach: reach_radius,
    };
    for (mesh, material, gt, part, occluder, aabb, rig) in &parts {
        if !occluder.0 || !matches!(part.blend, ModelBlend::Opaque | ModelBlend::AlphaTest) {
            continue;
        }
        let Some(m) = materials.get(&material.0) else {
            continue;
        };
        if m.extension.clutter_fade.w > 0.5 {
            continue;
        }
        let cutout = if matches!(part.blend, ModelBlend::AlphaTest) {
            m.base.base_color_texture.as_ref().map(|h| h.id())
        } else {
            None
        };
        let rig_slot = rig
            .and_then(|r| crate::rig_palette::part_slot(r.0, &rigs, &riders))
            .map(u32::from);
        let (scale, _, translation) = gt.to_scale_rotation_translation();
        // The cascade cull's sphere. A rigged part bounds by its box too: an animated doodad's is the
        // authored all-animation box (`terrain_stream::spawn::assemble`), and a posed body strays
        // only a little from its bind pose. A sphere round the ROOT alone culled a swaying tree's
        // canopy out of the small near cascade whenever its trunk stood outside that cascade's box,
        // so the ground just ahead of the camera lit up under the leaves and the shade came back on
        // a line at the next cascade — a line that walked with the camera.
        let (center, radius, tight) = match (aabb, rig_slot) {
            (Some(b), None) => {
                let r = Vec3::from(b.half_extents).length() * scale.max_element();
                (gt.transform_point(Vec3::from(b.center)), r, r)
            }
            (Some(b), Some(_)) => {
                let r = Vec3::from(b.half_extents).length() * scale.max_element();
                (
                    gt.transform_point(Vec3::from(b.center)),
                    r + RIG_CULL_MARGIN,
                    r,
                )
            }
            // No box at all: a generous sphere round the root.
            (None, _) => (translation, 4.0, 4.0),
        };
        if !near_cascades(center, radius) {
            continue;
        }
        let sun = !horizon.is_some_and(|h| sun_sweep_hidden(h, &sweep, center, radius));
        out.parts.push(EntityCaster {
            mesh: mesh.id(),
            world_from_local: gt.to_matrix(),
            center,
            radius,
            tight,
            rig_slot,
            cutout,
            sun,
        });
    }
}

/// fork: the sun's path for [`sun_sweep_hidden`]: the direction light travels, the camera and the
/// last cascade's reach about it.
struct Sweep {
    dir: Vec3,
    cam: Vec3,
    reach: f32,
}

/// fork: a sun shadow lands only along the caster's sweep away from the sun, down to the lowest
/// terrain the horizon walked (less [`SWEEP_FLOOR_MARGIN`] for what stands below it) and no further
/// than the cascades reach. When that whole swept box lies under the camera's horizon, every
/// receiver the shadow could darken is hidden, so the caster is dropped from the cascades. A sun
/// lower than [`SWEEP_MIN_DROP`] sweeps too far to bound, and never culls.
fn sun_sweep_hidden(
    horizon: &crate::horizon::Horizon,
    sweep: &Sweep,
    center: Vec3,
    radius: f32,
) -> bool {
    let drop = -sweep.dir.y;
    if drop < SWEEP_MIN_DROP {
        return false;
    }
    let floor = horizon.floor() - SWEEP_FLOOR_MARGIN;
    let length = ((center.y + radius - floor) / drop)
        .min(center.distance(sweep.cam) + sweep.reach)
        .max(0.0);
    let (lo, hi) = (center - Vec3::splat(radius), center + Vec3::splat(radius));
    let end = sweep.dir * length;
    horizon.hides_box(lo.min(lo + end), hi.max(hi + end), 0)
}

/// fork: how far below the lowest terrain the sun's sweep still looks for a receiver, yards.
const SWEEP_FLOOR_MARGIN: f32 = 50.0;
/// fork: the least downward component of the sun's direction the sweep is bounded for (about 6°).
const SWEEP_MIN_DROP: f32 = 0.1;

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
    let (center, radius) = slice_sphere(world_from_view, fov_y, aspect, near, far);
    cascade_from_sphere(center, radius, dir, size)
}

/// The bounding sphere of the view-frustum slice `[near, far]` (radius rounded up to whole yards).
pub fn slice_sphere(
    world_from_view: Mat4,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> (Vec3, f32) {
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
    (center, radius)
}

/// Texels of padding around a narrowed cull rect: the receivers' filter taps (the 5-tap cross over
/// the hardware 2x2 compare) and their normal offset ([`NORMAL_OFFSET_TEXELS`]).
const CULL_PAD_TEXELS: f32 = 4.0;

/// `sunshadow.wesl`'s `COVER_RADIUS_YD`: a shaded receiver reads the map this far around itself.
const COVER_RADIUS_YD: f32 = 1.5;

/// A cascade's caster cull shrunk to the clip-space XY rect of the camera frustum's `corners`,
/// padded by `pad` and clamped to the map; `None` when the camera's rect misses the map. The
/// reference culls a map's casters this way under `shadowCull` (`0x7bafd0`) and renders with the
/// full matrix: the light clip is orthographic, so a caster shades only receivers on its own clip
/// XY, and a receiver on screen lies inside the frustum's rect. Depth and `w` are untouched.
pub fn narrowed_cull(clip_from_world: &Mat4, corners: &[Vec3A; 8], pad: f32) -> Option<Mat4> {
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for c in corners {
        let p = clip_from_world.project_point3(Vec3::from(*c)).truncate();
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let lo = (lo - pad).max(Vec2::NEG_ONE);
    let hi = (hi + pad).min(Vec2::ONE);
    if hi.x <= lo.x || hi.y <= lo.y {
        return None;
    }
    // Clip XY [lo, hi] onto [-1, 1].
    let scale = 2.0 / (hi - lo);
    let offset = -(hi + lo) / (hi - lo);
    let remap = Mat4::from_cols(
        Vec4::new(scale.x, 0.0, 0.0, 0.0),
        Vec4::new(0.0, scale.y, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(offset.x, offset.y, 0.0, 1.0),
    );
    Some(remap * *clip_from_world)
}

/// The texel-snapped reverse-Z orthographic light clip around a sphere.
pub fn cascade_from_sphere(center: Vec3, radius: f32, dir: Vec3, size: u32) -> CascadeFit {
    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let light_rot = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, dir, up);
    let texel = 2.0 * radius / size as f32;
    let c = light_rot.transform_point3(center);
    let snapped = Vec3::new(
        (c.x / texel).floor() * texel,
        (c.y / texel).floor() * texel,
        c.z,
    );
    let light_view = Mat4::from_translation(-snapped) * light_rot;
    // Reverse-Z: the plane `CASTER_REACH` beyond the sphere toward the sun maps to 1, the sphere's
    // far side to 0.
    let reach = caster_reach();
    let proj =
        bevy::math::proj::orthographic(-radius, radius, -radius, radius, radius, -(radius + reach));
    CascadeFit {
        clip_from_world: proj * light_view,
        texel_world: texel,
        depth_bias: DEPTH_BIAS_YD / (2.0 * radius + reach),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_matches_the_wgsl_layout() {
        assert_eq!(ModernCascades::default().std430().len(), CASCADES_BYTES);
        assert_eq!(CASCADES_BYTES, 400);
    }

    /// The narrowed cull maps the camera's rect, padded, onto the whole clip square, keeps depth,
    /// and gives up when the camera's rect misses the map.
    #[test]
    fn the_narrowed_cull_spans_the_camera_rect() {
        let rect = |x0: f32, y0: f32, x1: f32, y1: f32| -> [Vec3A; 8] {
            std::array::from_fn(|i| {
                let x = if i & 1 == 0 { x0 } else { x1 };
                let y = if i & 2 == 0 { y0 } else { y1 };
                Vec3A::new(x, y, if i & 4 == 0 { 0.2 } else { 0.8 })
            })
        };
        let m = narrowed_cull(&Mat4::IDENTITY, &rect(-0.5, 0.0, 0.25, 0.5), 0.0).unwrap();
        let lo = m.project_point3(Vec3::new(-0.5, 0.0, 0.3));
        let hi = m.project_point3(Vec3::new(0.25, 0.5, 0.3));
        assert!(lo.truncate().abs_diff_eq(Vec2::NEG_ONE, 1e-5), "{lo}");
        assert!(hi.truncate().abs_diff_eq(Vec2::ONE, 1e-5), "{hi}");
        assert!((lo.z - 0.3).abs() < 1e-6, "depth untouched");
        // A rect past the map's edge clamps to it.
        let clamped = narrowed_cull(&Mat4::IDENTITY, &rect(-3.0, -3.0, 3.0, 3.0), 0.0).unwrap();
        assert!(clamped.abs_diff_eq(Mat4::IDENTITY, 1e-6));
        assert!(narrowed_cull(&Mat4::IDENTITY, &rect(2.0, 2.0, 3.0, 3.0), 0.0).is_none());
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
