//! **The modern sun's shadow pass** (fork-owned; `FORK.md` P2).
//!
//! A child of `static_gx::render` so it can reuse that module's GPU state as-is: the baked cell and
//! WMO vertex buffers, the per-cell bind groups (texture array, record table with kill bits) and the
//! draw runs. Each frame, for the primary world view, it:
//!
//! 1. uploads the cascade block (`modern::shadow::ModernCascades`) to its constant-handle buffer;
//! 2. clears every layer of the constant-handle shadow array (reverse-Z: to 0);
//! 3. draws every cached cell, WMO and prop whose world bounds overlap each active cascade,
//!    depth-only — opaque, or alpha-keyed for cutout runs, two-sided.
//!
//! Static casters (terrain, static_gx) are cached per cascade in a second depth array and copied
//! back while the cascade's matrix and the static caster set are unchanged; entity parts are drawn
//! on top every frame.

use bevy::app::SubApp;
use bevy::pbr::{per_view_shadow_pass, LATE_SHADOW_PASS};
use bevy::render::render_resource::binding_types::uniform_buffer;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::texture::GpuImage;
use bevy::shader::Shader;

use benilla_assets::materials::{MODERN_SHADOW_CASCADES, MODERN_SHADOW_MAP};

use super::*;
use crate::modern::lights::{ModernLights, ShadowLight, ATLAS_TILES, PER_ATLAS_LAYER};
use crate::modern::shadow::{
    ModernCascades, ModernEntityCasters, ModernTerrainCasters, LOCAL_LAYER, MAX_CASCADES,
    SHADOW_LAYERS, SHADOW_MAP_SIZE, SKY_LAYER,
};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::platform::collections::HashMap as StrideMap;
use bevy::platform::time::Instant;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, texture_2d, texture_depth_2d,
};
use std::time::Duration;

#[derive(Clone, Copy, ShaderType)]
struct CasterView {
    clip_from_world: Mat4,
}

#[derive(Resource)]
struct GxShadowPipelines {
    view_layout: BindGroupLayoutDescriptor,
    opaque: CachedRenderPipelineId,
    cutout: CachedRenderPipelineId,
    /// fork: P8b — clears one atlas tile by drawing over it (`depth_fill.wgsl`).
    fill: CachedRenderPipelineId,
    /// fork: P8b — restores a rectangle of a cascade from the static cache by drawing over it.
    restore: CachedRenderPipelineId,
    restore_layout: BindGroupLayoutDescriptor,
}

/// fork: P8b — how far past an entity caster's cull sphere its depth may still land, in yards, when
/// sizing the rectangle a restore puts back. The cull sphere already carries the rig margin; this is
/// slack on top, because a texel an entity wrote outside the rectangle would stay written — a shadow
/// left behind where the entity stood — until the cascade next re-centres.
const RESTORE_MARGIN_YD: f32 = 2.0;
/// …and in texels, for the compare's own footprint.
const RESTORE_MARGIN_PX: u32 = 2;
/// fork: P8b — the restore's tile grid per side: a caster marks the tiles its footprint touches, and
/// the restore draws one quad per run of marked tiles in a row. Without it every caster drew its own
/// quad, and at Darnassus hundreds of them overlapped — each fragment writing its depth from the
/// shader, so no early test could drop the repeats — until the "cheap" restore cost 20 ms. With the
/// mask no texel is written twice and the area stays near the casters' real footprint.
const RESTORE_TILES: usize = 32;

/// fork: P8b — what each cube-atlas slot's six tiles currently hold, so a stale slot is redrawn
/// alone. (P6 and P8 keyed the whole LAYER: one light arriving, or a cell streaming in anywhere in
/// the world, redrew all six lights × six faces of it.)
#[derive(Default)]
struct AtlasCache {
    slots: [AtlasSlot; crate::modern::lights::MAX_SHADOWED],
    /// fork: per slot, the entity casters its working faces were last drawn with — what
    /// [`dirty_faces`] holds this frame's list against.
    casters_drawn: [Vec<DrawnCaster>; crate::modern::lights::MAX_SHADOWED],
}

/// fork: one entity caster as a lamp's dynamic phase sees it: which mesh, where it is placed (its
/// world matrix's translation and rotation), and the bounds [`face_sees`] tests — kept for the
/// casters a slot was drawn with, so the faces a caster LEAVES are restored as well as the faces it
/// enters.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DrawnCaster {
    mesh: AssetId<Mesh>,
    at: Vec3,
    turn: Quat,
    center: Vec3,
    radius: f32,
}

impl DrawnCaster {
    fn of(cmd: &EntityDrawCmd) -> Self {
        Self {
            mesh: cmd.mesh,
            at: cmd.pose.0,
            turn: cmd.pose.1,
            center: cmd.center,
            radius: cmd.radius,
        }
    }
}

/// fork: all six faces of a cube, as a [`dirty_faces`] mask.
const ALL_FACES: u8 = 0b11_1111;

/// fork: how many frames a slot's wanted content — its light's anchor and the static casters inside
/// its reach — has to hold still before the slot's static faces are drawn. While a street streams in,
/// every cell that lands changes the caster set; drawing on each change is the redraw storm this
/// waits out, and the lamp is simply unshadowed until then (`modern::lights::built`).
const ATLAS_SETTLE_FRAMES: u32 = 60;
/// fork: a slot drawn while one of its casters was not yet resident is retried this many times, each
/// after another settle, before its partial faces are accepted as built.
const ATLAS_ATTEMPTS: u32 = 3;
/// fork: [`AtlasSlot::dynamic`] — the working tiles hold something not known to be the static map (a
/// fresh texture, or a slot whose static faces were just redrawn into the cache): restore first.
const DYN_UNKNOWN: u64 = 0;
/// fork: [`AtlasSlot::dynamic`] — the working tiles hold exactly the static map.
const DYN_STATIC: u64 = 1;
/// fork: [`AtlasSlot::dynamic`] — the working tiles hold the static map with the casters in
/// [`AtlasCache::casters_drawn`] drawn over it.
const DYN_DRAWN: u64 = 2;
/// fork: how far a dynamic caster's placement may move (yd) before a lamp's faces are redrawn for
/// it — float noise, nothing more. It was a quarter of a yard, which is what made a character's lamp
/// shadow trail it: a walk redrew the shadow only every few frames, a slow step not until the
/// quarter-yard was used up, so the shadow stepped along behind the body like a lower frame rate.
/// Every real movement redraws now; what pays for it is that only the FACES the caster touches are
/// restored and redrawn ([`dirty_faces`]), not all six. An idle creature's animation does not move
/// its placement, so it still costs nothing.
const DYN_MOVE_YD: f32 = 0.02;
/// fork: how far it may turn (radians, ~1°) before the same — turning on the spot is movement too.
const DYN_TURN: f32 = 0.02;
/// fork: an entity part whose own bounds hold the light, and that is no bigger than this (yd), is the
/// light's HOST — the lantern, the brazier, the fist round the torch — and never shadows it. A
/// swinging lantern would otherwise cage its own pool, and be a moving caster every frame it swings.
const HOST_MAX_RADIUS: f32 = 2.5;

#[derive(Clone, Copy, Default)]
struct AtlasSlot {
    /// The light the caster fingerprint below was taken for ([`ShadowLight::map_key`]; 0 = never).
    light: u64,
    /// The global static fingerprint it was taken under — only a change to that re-walks the
    /// caster list, so a settled frame pays nothing here.
    world: u64,
    /// The static casters inside this light's sphere, fingerprinted ([`atlas_casters_key`]). A cell
    /// streaming in 300 yd away changes the world's fingerprint and not this one, so the slot stays.
    casters: u64,
    /// What the six static faces hold: the light and its casters together, or 0 = nothing valid.
    drawn: u64,
    /// fork: the content this slot is waiting to draw, and how many frames it has held still.
    pending: u64,
    waited: u32,
    /// fork: draws of `pending` that came out incomplete.
    attempts: u32,
    /// fork: what the WORKING tiles hold on top of the static map: [`DYN_UNKNOWN`], [`DYN_STATIC`]
    /// or [`DYN_DRAWN`].
    dynamic: u64,
}

impl AtlasSlot {
    fn wanted(&self) -> u64 {
        (self.light ^ self.casters.rotate_left(29)).max(1)
    }
}

#[derive(Resource, Default)]
struct CasterViewBuffer(DynamicUniformBuffer<CasterView>);

#[derive(Clone, Copy, ShaderType)]
struct EntityDraw {
    world_from_local: Mat4,
    rig_slot: u32,
}

/// Entity caster pipelines by (mesh vertex layout, skinned, cutout).
#[derive(Resource)]
struct EntityShadowPipelines {
    view_layout: BindGroupLayoutDescriptor,
    draw_layout: BindGroupLayoutDescriptor,
    skin_layout: BindGroupLayoutDescriptor,
    cut_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    by_key: StrideMap<(MeshVertexBufferLayoutRef, bool, bool), CachedRenderPipelineId>,
}

#[derive(Resource, Default)]
struct EntityDrawBuffer(DynamicUniformBuffer<EntityDraw>);

/// Cutout texture bind groups, rebuilt when the image's view changes.
#[derive(Resource, Default)]
struct CutoutBinds(StrideMap<AssetId<Image>, (TextureViewId, BindGroup)>);

struct EntityDrawCmd {
    mesh: AssetId<Mesh>,
    pipeline: CachedRenderPipelineId,
    offset: u32,
    skinned: bool,
    cutout: Option<BindGroup>,
    center: Vec3,
    radius: f32,
    /// fork: [`crate::modern::shadow::EntityCaster::tight`].
    tight: f32,
    /// fork: [`crate::modern::shadow::EntityCaster::sun`]: drawn into the sun's cascades.
    sun: bool,
    /// fork: where the part is placed — its world matrix's translation and rotation. The lamp
    /// atlas's dynamic phase compares its casters by this ([`dirty_faces`]), not by `center`: an
    /// animated part's bounds are re-fitted as it animates, so a tree swaying beside a lamp moved
    /// its centre every frame while its placement stood still.
    pose: (Vec3, Quat),
}

#[derive(Resource, Default)]
struct EntityShadowFrame {
    draws: Vec<EntityDrawCmd>,
    draw_bind: Option<BindGroup>,
    skin_bind: Option<BindGroup>,
}

/// Terrain caster pipelines, one per vertex stride (position is always attribute 0 at offset 0).
#[derive(Resource)]
struct TerrainShadowPipelines {
    view_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    by_stride: StrideMap<u64, CachedRenderPipelineId>,
}

/// This frame's shadow targets: one depth view per array layer, the caster-view bind group and its
/// per-cascade offsets. `count == 0` = nothing to draw (classic, or not ready).
#[derive(Resource, Default)]
struct ModernShadowFrame {
    layers: Vec<TextureView>,
    /// The working array (the constant-handle image's texture) and the static-only cache.
    texture: Option<Texture>,
    cache: Option<Texture>,
    /// Per cascade: the key its cache layer was last written under (0 = invalid).
    cache_keys: [std::sync::atomic::AtomicU64; SHADOW_LAYERS],
    /// fork (perf): did the dynamic pass actually draw into this layer last frame?
    ///
    /// The whole point of the per-frame restore is to undo the previous frame's entity casters, so
    /// a layer nothing drew into still holds pure static depth and needs no restore at all. A
    /// `Depth32Float` copy is 2048² × 4 B = **16.8 MiB**, and five of them ran every settled frame
    /// — including the sky-occlusion layer, which has no dynamic pass at all (see the `is_sky`
    /// skip below) and so was restored every frame to undo nothing.
    ///
    /// Per LAYER and not per region, because that is the only granularity the API offers:
    /// wgpu-core's `validate_texture_copy_range` refuses a partial copy of a depth format on x/y
    /// ("Multisampled and depth-stencil formats do not support partial copies on x and y
    /// dimensions, but do support copying a subset of layers"), so the tempting
    /// copy-only-the-casters'-rectangle is not expressible. Skipping the layer wholesale is.
    layer_dirty: [std::sync::atomic::AtomicBool; SHADOW_LAYERS],
    /// fork: P8b — per cascade, one texel rectangle per entity caster last frame issued into it:
    /// all a restore has to put back. Empty with the layer dirty means "unknown", and the whole
    /// layer is copied as before.
    dirty_rects: std::sync::Mutex<[Vec<URect>; MAX_CASCADES]>,
    /// fork: P8b — per layer, a view of the static cache's layer for the restore draw.
    cache_views: Vec<TextureView>,
    /// fork: per layer, the static cache's layer as a depth ATTACHMENT — the lamp atlas's static
    /// phase draws a slot's faces straight into the cache, and the working layer takes a copy.
    /// Empty unless [`atlas_restores`].
    cache_targets: Vec<TextureView>,
    /// fork: whether the cache was made for [`atlas_restores`] (all layers, a render attachment).
    cache_restores: bool,
    offsets: Vec<u32>,
    /// fork: modern lighting (P6) — one caster view per cube face of every shadowed local light,
    /// and (P8b) per atlas SLOT what its tiles currently hold.
    local_offsets: Vec<u32>,
    atlas: std::sync::Mutex<AtlasCache>,
    /// The restore draws' rectangle buffers, one per draw in a frame and kept across frames
    /// ([`restore_buffer`]); `restore_next` is this frame's next unused one.
    restore_pool: std::sync::Mutex<Vec<Buffer>>,
    restore_next: std::sync::atomic::AtomicUsize,
    /// Frames prepared so far, the clock [`holds_entities`] alternates the far cascades on.
    frame_no: u32,
    /// fork (perf): per cascade and the sky map, what static content its cached layer holds.
    layer_casters: std::sync::Mutex<[LayerCasters; SKY_LAYER + 1]>,
    bind: Option<BindGroup>,
    count: u32,
}

/// fork (perf): one shadow layer's static casters. The cached layer used to be keyed on the whole
/// world's static set, so any cell streaming in anywhere redrew every cascade and the sky map
/// (~12 ms on an Iris Xe); now only the content inside a layer's own light-space rectangle counts,
/// and a change to it waits for the streaming to settle (the light clip moving still redraws at once).
#[derive(Clone, Copy, Default)]
struct LayerCasters {
    /// The light clip and world fingerprint [`Self::casters`] was taken under (0 = never).
    clip: u64,
    world: u64,
    /// The static casters inside the layer ([`layer_casters_key`]).
    casters: u64,
    /// When (instant, frame) [`Self::casters`] last changed.
    changed_at: Option<(Instant, u32)>,
    /// The casters the cached layer was drawn with.
    drawn: u64,
    /// When (instant, frame) a newer set started waiting, while one does.
    waiting_since: Option<(Instant, u32)>,
}

/// fork (perf): how long a layer's changed caster set must hold still before it is redrawn…
const LAYER_SETTLE: Duration = Duration::from_millis(500);
const LAYER_SETTLE_FRAMES: u32 = 30;
/// …and the longest a changed set waits regardless, so content streaming in for a minute still lands.
/// Both the time and the frame limit must pass: a slow GPU keeps the frame count, a fast one the time.
const LAYER_MAX_DEFER: Duration = Duration::from_secs(2);
const LAYER_MAX_DEFER_FRAMES: u32 = 120;

/// fork (perf): the first cascade whose entity casters are redrawn every other frame.
///
/// Such a cascade starts 60 yd out or more on the tiers that have it and resolves a fifth of a
/// yard a texel at best, so a swaying crown or a runner's step moves its shadow by less than a
/// texel or two between frames. Each such cascade is a few hundred draws a frame at a wooded
/// coast, as many as the near two together.
const HELD_CASCADES_FROM: usize = 2;

/// Does cascade `layer` keep last frame's entity casters this frame? The far cascades take turns,
/// one a frame, so the frame's cost stays even. Only a layer whose static cache holds
/// ([`cascade_key`]) may be held: a re-centred cascade is redrawn whole.
fn holds_entities(layer: usize, frame_no: u32) -> bool {
    layer >= HELD_CASCADES_FROM && (frame_no as usize + layer) % 2 == 1
}

/// A storage buffer holding `rects` for one restore draw. A buffer made per draw is freed in a
/// browser only when its garbage collector runs, and Firefox's ran out of GPU memory first, so the
/// draws of a frame each take the next buffer of a pool that lives as long as the frame resource.
/// Each draw's buffer is its own, so the queue writes all land before the frame's submission reads.
fn restore_buffer(
    frame: &ModernShadowFrame,
    device: &RenderDevice,
    queue: &RenderQueue,
    rects: &[[f32; 4]],
) -> Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(rects);
    let i = frame
        .restore_next
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut pool = frame
        .restore_pool
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if pool.get(i).is_none_or(|b| b.size() < bytes.len() as u64) {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("modern_shadow_restore_rects"),
            size: (bytes.len() as u64).next_power_of_two().max(4096),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if i < pool.len() {
            pool[i] = buffer;
        } else {
            pool.push(buffer);
        }
    }
    let buffer = pool[i.min(pool.len() - 1)].clone();
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn init_shadow_pipelines(
    mut commands: Commands,
    pipes: Res<GxPipelines>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/modern/static_gx_shadow.wgsl");
    let view_layout = BindGroupLayoutDescriptor::new(
        "modern_shadow_caster_view",
        &BindGroupLayoutEntries::single(ShaderStages::VERTEX, uniform_buffer::<CasterView>(true)),
    );
    let make = |label: &'static str, cutout: bool| RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: vec![view_layout.clone(), pipes.cell_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            buffers: vec![vertex_layout()],
        },
        // Opaque casters need no fragment stage; cutout ones alpha-key their texture.
        fragment: cutout.then(|| FragmentState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![],
        }),
        // Two-sided: foliage cards and thin WMO walls must cast from either face.
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        ..default()
    };
    commands.insert_resource(EntityShadowPipelines {
        view_layout: view_layout.clone(),
        draw_layout: BindGroupLayoutDescriptor::new(
            "modern_shadow_entity_draw",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX,
                uniform_buffer::<EntityDraw>(true),
            ),
        ),
        skin_layout: BindGroupLayoutDescriptor::new(
            "modern_shadow_entity_skin",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    uniform_buffer::<EntityDraw>(true),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        ),
        cut_layout: BindGroupLayoutDescriptor::new(
            "modern_shadow_entity_cutout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        ),
        shader: asset_server.load("embedded://benilla_world/shaders/modern/entity_shadow.wgsl"),
        by_key: StrideMap::default(),
    });
    commands.insert_resource(TerrainShadowPipelines {
        view_layout: view_layout.clone(),
        shader: asset_server.load("embedded://benilla_world/shaders/modern/terrain_shadow.wgsl"),
        by_stride: StrideMap::default(),
    });
    // fork: P8b — the atlas's tile clear: one fullscreen triangle at the far plane, depth only.
    let fill = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("modern_shadow_tile_fill".into()),
        layout: vec![],
        vertex: VertexState {
            shader: asset_server.load("embedded://benilla_world/shaders/modern/depth_fill.wgsl"),
            shader_defs: vec![],
            entry_point: Some("fill_vertex".into()),
            buffers: vec![],
        },
        fragment: None,
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        ..default()
    });
    // fork: P8b — the cascade restore: the same triangle, writing the static cache's depth back
    // under a scissor (`depth_fill.wgsl`'s `restore_fragment`).
    let restore_layout = BindGroupLayoutDescriptor::new(
        "modern_shadow_restore",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                texture_depth_2d(),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );
    let fill_shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/modern/depth_fill.wgsl");
    let restore = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("modern_shadow_restore".into()),
        layout: vec![restore_layout.clone()],
        vertex: VertexState {
            shader: fill_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("restore_vertex".into()),
            buffers: vec![],
        },
        fragment: Some(FragmentState {
            shader: fill_shader,
            shader_defs: vec![],
            entry_point: Some("restore_fragment".into()),
            targets: vec![],
        }),
        primitive: PrimitiveState {
            cull_mode: None,
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        ..default()
    });
    commands.insert_resource(GxShadowPipelines {
        opaque: pipeline_cache.queue_render_pipeline(make("modern_shadow_static", false)),
        cutout: pipeline_cache.queue_render_pipeline(make("modern_shadow_static_cutout", true)),
        fill,
        restore,
        restore_layout,
        view_layout,
    });
}

/// fork: does THIS session's tier run the lamp atlas's dynamic phase? Only then does an atlas layer
/// need a static copy to restore from, and only then is the static cache a render attachment (the
/// static phase draws into it). Anything else keeps the cache exactly as the cascades need it:
/// their layers alone, copy and sample only.
fn atlas_restores(lights: Option<&ModernLights>) -> bool {
    lights.is_some_and(|l| l.tier.dynamic_casters && !l.shadowed.is_empty())
}

#[allow(clippy::too_many_arguments)]
fn prepare_modern_shadow(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cascades: Option<Res<ModernCascades>>,
    lights: Option<Res<ModernLights>>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    pipes: Option<Res<GxShadowPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    mut uniforms: ResMut<CasterViewBuffer>,
    mut frame: ResMut<ModernShadowFrame>,
    mut uploaded_plane: Option<ResMut<crate::modern::shadow::UploadedMirrorPlane>>,
) {
    frame.count = 0;
    frame.bind = None;
    frame.frame_no = frame.frame_no.wrapping_add(1);
    *frame.restore_next.get_mut() = 0;
    if let Some(plane) = uploaded_plane.as_deref_mut() {
        plane.0 = None;
    }
    let (Some(cascades), Some(pipes)) = (cascades, pipes) else {
        return;
    };
    let Some(block) = buffers.get(MODERN_SHADOW_CASCADES.id()) else {
        return;
    };
    // Always uploaded, so receivers see `count == 0` in a classic frame.
    queue.write_buffer(&block.buffer, 0, &cascades.std430());
    if let Some(plane) = uploaded_plane.as_deref_mut() {
        plane.0 = Some(cascades.sky_params[3]);
    }
    if cascades.count == 0 {
        return;
    }
    let Some(image) = images.get(MODERN_SHADOW_MAP.id()) else {
        return;
    };
    // The array is re-created when the lighting latch re-sizes it (`modern::shadow::fit_shadow_map`):
    // every view, the static cache and every cached layer key belong to the old texture.
    if frame
        .texture
        .as_ref()
        .is_some_and(|t| t.id() != image.texture.id())
    {
        frame.layers.clear();
        frame.cache = None;
        for key in &frame.cache_keys {
            key.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        *frame
            .atlas
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = AtlasCache::default();
        frame.cache_views.clear();
        frame.cache_targets.clear();
        // fork: the new texture holds no lamp's faces; the main world must stop fading them in.
        for slot in 0..crate::modern::lights::MAX_SHADOWED {
            crate::modern::lights::publish_built(slot, 0);
        }
    }
    frame.texture = Some(image.texture.clone());
    // fork: the cache's shape follows the live tier; a tier change across the dynamic phase makes
    // a new one (the cascades' keys go with it, so they redraw into it once).
    let restores = atlas_restores(lights.as_deref());
    if frame.cache.is_some() && frame.cache_restores != restores {
        frame.cache = None;
        for key in &frame.cache_keys {
            key.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        *frame
            .atlas
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = AtlasCache::default();
        for slot in 0..crate::modern::lights::MAX_SHADOWED {
            crate::modern::lights::publish_built(slot, 0);
        }
    }
    frame.cache_restores = restores;
    if frame.cache.is_none() {
        // fork: P8b — the cache holds only the layers anything is ever RESTORED into: the cascades,
        // whose entity casters are undone each frame. The sky map has no dynamic pass, and the cube
        // atlas has none on any tier (`LightTier::dynamic_casters`), so a cached copy of either was
        // 16.8 MB of memory and a 16.8 MB copy per redraw that nothing read back.
        let layers = if restores {
            image.texture.depth_or_array_layers()
        } else {
            image
                .texture
                .depth_or_array_layers()
                .min(MAX_CASCADES as u32)
        };
        frame.cache = Some(device.create_texture(&TextureDescriptor {
            label: Some("modern_shadow_static_cache"),
            size: Extent3d {
                width: image.texture.width(),
                height: image.texture.height(),
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            // TEXTURE_BINDING: the restore draw reads it (`depth_fill.wgsl`'s `restore_fragment`).
            // RENDER_ATTACHMENT only for the lamp atlas's static phase, which draws into it.
            usage: if restores {
                TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::RENDER_ATTACHMENT
            } else {
                TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING
            },
            view_formats: &[],
        }));
        frame.cache_views.clear();
        frame.cache_targets.clear();
    }
    if frame.cache_targets.is_empty() && restores {
        if let Some(cache) = frame.cache.as_ref() {
            let targets: Vec<TextureView> = (0..cache.depth_or_array_layers())
                .map(|layer| {
                    cache.create_view(&TextureViewDescriptor {
                        label: Some("modern_shadow_cache_target"),
                        dimension: Some(TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..default()
                    })
                })
                .collect();
            frame.cache_targets = targets;
        }
    }
    if frame.cache_views.is_empty() {
        if let Some(cache) = frame.cache.as_ref() {
            let views: Vec<TextureView> = (0..cache.depth_or_array_layers())
                .map(|layer| {
                    cache.create_view(&TextureViewDescriptor {
                        label: Some("modern_shadow_cache_layer"),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::DepthOnly,
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..default()
                    })
                })
                .collect();
            frame.cache_views = views;
        }
    }
    if frame.layers.is_empty() {
        // **The texture's OWN depth, not [`SHADOW_LAYERS`].** P8 latched the array's depth to the
        // tier (`modern::shadow::fit_shadow_map`) so a session that shadows no local light stops
        // paying 16.8 MB a layer for an atlas it never samples — but this loop kept building views
        // for the largest depth any tier could ask for. On every tier below Cinematic, and on the
        // web for High and Ultra, that asked a five-layer texture for views of layers 5 and 6:
        //
        //   Texture view array layer range (baseArrayLayer: 5, arrayLayerCount: 1) exceeds the
        //   texture's array layer count (5).
        //
        // An invalid view invalidates the `modern_local_shadow` pass that attaches it, that pass's
        // command buffer, and the submit carrying it — so `static_gx c0 t0` and the pbr pipelines
        // all came up "invalid due to a previous error" and the screen went black. The readers
        // already index this through `.get()`, so a shorter list simply means "no atlas here",
        // which is what a tier with no local layers means.
        frame.layers = (0..image.texture.depth_or_array_layers())
            .map(|layer| {
                image.texture.create_view(&TextureViewDescriptor {
                    label: Some("modern_shadow_layer"),
                    dimension: Some(TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..default()
                })
            })
            .collect();
    }
    uniforms.0.clear();
    frame.offsets.clear();
    for i in 0..SHADOW_LAYERS {
        let clip_from_world = if i == SKY_LAYER {
            cascades.sky_clip
        } else if i < cascades.count as usize {
            cascades.clip_from_world[i]
        } else {
            Mat4::IDENTITY
        };
        let offset = uniforms.0.push(&CasterView { clip_from_world });
        frame.offsets.push(offset);
    }
    // fork: modern lighting (FORK.md, P6) — the shadowed local lights' cube
    // faces. Six 90° reverse-Z perspectives per light, whose convention `locallight::cube_face`
    // mirrors analytically (`modern::lights::cube_face_clip`).
    frame.local_offsets.clear();
    if let Some(lights) = lights.as_deref() {
        for light in &lights.shadowed {
            // A hole still gets its six offsets, so a slot's tiles are always at `slot * 6 + face`
            // however many lights are actually held — the identity the receivers' nibble depends on.
            let (pos, near) = light.map_or((Vec3::ZERO, crate::modern::lights::CUBE_NEAR), |h| {
                (h.light.pos, h.light.near)
            });
            for face in 0..6 {
                let clip_from_world = crate::modern::lights::cube_face_clip(pos, face, near);
                frame
                    .local_offsets
                    .push(uniforms.0.push(&CasterView { clip_from_world }));
            }
        }
    }
    uniforms.0.write_buffer(&device, &queue);
    let Some(binding) = uniforms.0.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipes.view_layout);
    frame.bind = Some(device.create_bind_group(
        "modern_shadow_caster_view",
        &layout,
        &BindGroupEntries::single(binding),
    ));
    frame.count = cascades.count;
}

/// Queue a terrain caster pipeline for every vertex stride the resident cells use.
fn queue_terrain_shadow_pipelines(
    casters: Option<Res<ModernTerrainCasters>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    pipeline_cache: Res<PipelineCache>,
    pipes: Option<ResMut<TerrainShadowPipelines>>,
    mut unprepared: Local<bool>,
) {
    let (Some(casters), Some(mut pipes)) = (casters, pipes) else {
        return;
    };
    // A new stride arrives with a new cell, or with a cell's mesh finishing its upload.
    if !casters.is_changed() && !*unprepared {
        return;
    }
    *unprepared = false;
    for (mesh_id, _, _, _) in &casters.cells {
        let Some(mesh) = meshes.get(*mesh_id) else {
            *unprepared = true;
            continue;
        };
        let stride = mesh.layout.0.layout().array_stride;
        if pipes.by_stride.contains_key(&stride) {
            continue;
        }
        let descriptor = RenderPipelineDescriptor {
            label: Some("modern_shadow_terrain".into()),
            layout: vec![pipes.view_layout.clone()],
            vertex: VertexState {
                shader: pipes.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![VertexBufferLayout {
                    array_stride: stride,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: None,
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            ..default()
        };
        let id = pipeline_cache.queue_render_pipeline(descriptor);
        pipes.by_stride.insert(stride, id);
    }
}

/// Build this frame's entity caster draws: pipelines (queued on first sight of a layout), one
/// uniform per part, cutout texture bind groups, and the two draw bind groups.
#[allow(clippy::too_many_arguments)]
fn prepare_entity_shadows(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    casters: Option<Res<ModernEntityCasters>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    images: Res<RenderAssets<GpuImage>>,
    light: Option<Res<crate::lighting::SharedLightBuffer>>,
    pipeline_cache: Res<PipelineCache>,
    pipes: Option<ResMut<EntityShadowPipelines>>,
    mut buffer: ResMut<EntityDrawBuffer>,
    mut cut_binds: ResMut<CutoutBinds>,
    mut frame: ResMut<EntityShadowFrame>,
) {
    frame.draws.clear();
    frame.draw_bind = None;
    frame.skin_bind = None;
    let (Some(casters), Some(light), Some(mut pipes)) = (casters, light, pipes) else {
        return;
    };
    if casters.parts.is_empty() {
        return;
    }
    buffer.0.clear();
    for part in &casters.parts {
        let Some(mesh) = meshes.get(part.mesh) else {
            continue;
        };
        let skinned = part.rig_slot.is_some();
        let cut_image = part
            .cutout
            .and_then(|id| images.get(id).map(|gpu| (id, gpu)));
        let mut cutout = cut_image.is_some();
        let key = (mesh.layout.clone(), skinned, cutout);
        let pipeline = match pipes.by_key.get(&key) {
            Some(id) => Some(*id),
            None => {
                let mut attrs = vec![Mesh::ATTRIBUTE_POSITION.at_shader_location(0)];
                if cutout {
                    attrs.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(2));
                }
                if skinned {
                    attrs.push(benilla_assets::ATTRIBUTE_WOW_JOINT_INDEX.at_shader_location(10));
                    attrs.push(benilla_assets::ATTRIBUTE_WOW_JOINT_WEIGHT.at_shader_location(11));
                }
                match mesh.layout.0.get_layout(&attrs) {
                    Ok(vertex_layout) => {
                        let mut defs: Vec<ShaderDefVal> = Vec::new();
                        let mut layout = vec![
                            pipes.view_layout.clone(),
                            if skinned {
                                pipes.skin_layout.clone()
                            } else {
                                pipes.draw_layout.clone()
                            },
                        ];
                        if skinned {
                            defs.push("SKINNED".into());
                        }
                        if cutout {
                            defs.push("CUTOUT".into());
                            layout.push(pipes.cut_layout.clone());
                        }
                        let descriptor = RenderPipelineDescriptor {
                            label: Some("modern_shadow_entity".into()),
                            layout,
                            vertex: VertexState {
                                shader: pipes.shader.clone(),
                                shader_defs: defs.clone(),
                                entry_point: Some("vertex".into()),
                                buffers: vec![vertex_layout],
                            },
                            fragment: cutout.then(|| FragmentState {
                                shader: pipes.shader.clone(),
                                shader_defs: defs.clone(),
                                entry_point: Some("fragment".into()),
                                targets: vec![],
                            }),
                            primitive: PrimitiveState {
                                cull_mode: None,
                                ..default()
                            },
                            depth_stencil: Some(DepthStencilState {
                                format: CORE_3D_DEPTH_FORMAT,
                                depth_write_enabled: Some(true),
                                depth_compare: Some(CompareFunction::GreaterEqual),
                                stencil: StencilState::default(),
                                bias: DepthBiasState::default(),
                            }),
                            ..default()
                        };
                        let id = pipeline_cache.queue_render_pipeline(descriptor);
                        pipes.by_key.insert(key, id);
                        Some(id)
                    }
                    // A cutout part without UVs casts opaque; anything else missing is skipped.
                    Err(_) if cutout => {
                        cutout = false;
                        pipes
                            .by_key
                            .get(&(mesh.layout.clone(), skinned, false))
                            .copied()
                    }
                    Err(_) => None,
                }
            }
        };
        let Some(pipeline) = pipeline else {
            continue;
        };
        let cutout_bind = match (cutout, cut_image) {
            (true, Some((id, gpu))) => {
                let view_id = gpu.texture_view.id();
                let stale = cut_binds.0.get(&id).is_none_or(|(v, _)| *v != view_id);
                if stale {
                    let layout = pipeline_cache.get_bind_group_layout(&pipes.cut_layout);
                    let bind = device.create_bind_group(
                        "modern_shadow_entity_cutout",
                        &layout,
                        &BindGroupEntries::sequential((&gpu.texture_view, &gpu.sampler)),
                    );
                    cut_binds.0.insert(id, (view_id, bind));
                }
                cut_binds.0.get(&id).map(|(_, b)| b.clone())
            }
            _ => None,
        };
        let offset = buffer.0.push(&EntityDraw {
            world_from_local: part.world_from_local,
            rig_slot: part.rig_slot.unwrap_or(0),
        });
        frame.draws.push(EntityDrawCmd {
            mesh: part.mesh,
            pipeline,
            offset,
            skinned,
            cutout: cutout_bind,
            center: part.center,
            radius: part.radius,
            tight: part.tight,
            sun: part.sun,
            pose: {
                let (_, rotation, translation) =
                    part.world_from_local.to_scale_rotation_translation();
                (translation, rotation)
            },
        });
    }
    // Drop the bind groups of cutout textures no caster uses any more: each holds its texture alive.
    if cut_binds.0.len() > 256 {
        let live: std::collections::HashSet<AssetId<Image>> =
            casters.parts.iter().filter_map(|p| p.cutout).collect();
        cut_binds.0.retain(|id, _| live.contains(id));
    }
    buffer.0.write_buffer(&device, &queue);
    let Some(binding) = buffer.0.binding() else {
        frame.draws.clear();
        return;
    };
    let draw_layout = pipeline_cache.get_bind_group_layout(&pipes.draw_layout);
    let skin_layout = pipeline_cache.get_bind_group_layout(&pipes.skin_layout);
    frame.draw_bind = Some(device.create_bind_group(
        "modern_shadow_entity_draw",
        &draw_layout,
        &BindGroupEntries::single(binding.clone()),
    ));
    frame.skin_bind = Some(device.create_bind_group(
        "modern_shadow_entity_skin",
        &skin_layout,
        &BindGroupEntries::sequential((binding, light.0.as_entire_binding())),
    ));
}

pub(crate) fn gx_shadow_pass(
    world: &World,
    view: ViewQuery<&StaticGxView>,
    mut render_context: RenderContext,
) {
    let gx_view = view.into_inner();
    {
        if !gx_view.primary {
            return;
        }
        let frame = world.resource::<ModernShadowFrame>();
        let Some(view_bind) = frame.bind.as_ref() else {
            return;
        };
        // Every pass below opens a `modern_*` span: the journal's `gpu_modern` column
        // (`benilla-app` `perf::journal`) where the device times passes.
        let diagnostics = render_context.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let cascades = world.resource::<ModernCascades>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let ready = world.get_resource::<GxShadowPipelines>().and_then(|pipes| {
            Some((
                pipeline_cache.get_render_pipeline(pipes.opaque)?,
                pipeline_cache.get_render_pipeline(pipes.cutout)?,
            ))
        });

        let gx = world.resource::<GxWorld>();
        let cache = world.resource::<GxGpuCache>();
        let meshes = world.resource::<RenderAssets<RenderMesh>>();
        let allocator = world.resource::<MeshAllocator>();
        let mut casters: Vec<(&GxCellGpu, &GxCellDraw)> = Vec::new();
        for (key, gpu) in &cache.cells {
            if let Some(draw) = gx.cells.get(key) {
                casters.push((gpu, draw));
            }
        }
        for (entity, gpu) in &cache.wmos {
            if let Some(draw) = gx.wmos.get(entity) {
                casters.push((gpu, draw));
            }
        }
        for (entity, gpu) in &cache.props {
            if let Some(draw) = gx.props.get(entity) {
                casters.push((gpu, draw));
            }
        }

        let terrain = world.get_resource::<ModernTerrainCasters>();
        let terrain_pipes = world.get_resource::<TerrainShadowPipelines>();
        // The static caster set's identity: cached layers are reused only while it (and the
        // cascade's matrix) is unchanged. Both halves count their own changes, so a standing frame
        // no longer hashes every caster to learn that nothing moved.
        let static_hash = {
            use std::hash::BuildHasher;
            bevy::platform::hash::FixedHasher
                .hash_one((cache.rev, terrain.map(|t| (t.rev, t.cells.len()))))
        };
        let (Some(working), Some(static_cache)) = (frame.texture.as_ref(), frame.cache.as_ref())
        else {
            return;
        };
        let now = Instant::now();

        // The local-light atlas is the layer after the sky map and is drawn below, not here: it is
        // not a cascade, it is not cleared every frame, and no entity is drawn into it.
        for i in 0..frame.layers.len().min(SKY_LAYER + 1) {
            let layer = &frame.layers[i];
            // The last layer is the sky-occlusion map: static casters only, from straight above.
            let is_sky = i == SKY_LAYER;
            let active = if is_sky {
                cascades.count > 0
            } else {
                i < cascades.count as usize
            };
            let clip_from_world = if is_sky {
                cascades.sky_clip
            } else {
                cascades.clip_from_world[i]
            };
            let (key, layer_casters) = if active {
                layer_key(
                    frame,
                    i,
                    &clip_from_world,
                    static_hash,
                    &casters,
                    terrain,
                    now,
                )
            } else {
                (0, 0)
            };
            let slot = &frame.cache_keys[i];
            let drawn = slot.load(std::sync::atomic::Ordering::Relaxed);
            let cached = active
                && key != 0
                && (drawn == key || defers_redraw(frame, i, &clip_from_world, drawn, now));
            // A held layer keeps what it shows: no restore, no entity pass, and its dirty
            // rectangles stand for the frame that next redraws it.
            if cached && !is_sky && holds_entities(i, frame.frame_no) {
                continue;
            }

            if cached {
                // Only what the previous frame's entity casters dirtied needs putting back.
                if frame.layer_dirty[i].swap(false, std::sync::atomic::Ordering::Relaxed) {
                    // fork: P8b — put back only the rectangles last frame's entities could have
                    // written, by drawing the cached depth over them. A whole-layer copy (16.8 MB)
                    // stays the fallback while the rectangles or the restore pipeline are unknown.
                    let rects = std::mem::take(
                        &mut frame
                            .dirty_rects
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)[i],
                    );
                    let restore = world.get_resource::<GxShadowPipelines>().and_then(|p| {
                        Some((
                            pipeline_cache.get_render_pipeline(p.restore)?,
                            &p.restore_layout,
                        ))
                    });
                    match (restore, frame.cache_views.get(i)) {
                        (Some((restore, layout)), Some(view)) if !rects.is_empty() => {
                            let data: Vec<[f32; 4]> = rects
                                .iter()
                                .map(|r| {
                                    [
                                        r.min.x as f32,
                                        r.min.y as f32,
                                        r.max.x as f32,
                                        r.max.y as f32,
                                    ]
                                })
                                .collect();
                            let device = render_context.render_device();
                            let buffer = restore_buffer(
                                frame,
                                device,
                                world.resource::<RenderQueue>(),
                                &data,
                            );
                            let bind = device.create_bind_group(
                                "modern_shadow_restore",
                                &pipeline_cache.get_bind_group_layout(layout),
                                &BindGroupEntries::sequential((view, buffer.as_entire_binding())),
                            );
                            let mut pass =
                                render_context.begin_tracked_render_pass(RenderPassDescriptor {
                                    label: Some("modern_shadow_restore"),
                                    color_attachments: &[],
                                    depth_stencil_attachment: Some(
                                        RenderPassDepthStencilAttachment {
                                            view: layer,
                                            depth_ops: Some(Operations {
                                                load: LoadOp::Load,
                                                store: StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                });
                            let span = diagnostics.pass_span(&mut pass, "modern_shadow_restore");
                            pass.set_render_pipeline(restore);
                            pass.set_bind_group(0, &bind, &[]);
                            pass.draw(0..6, 0..data.len() as u32);
                            span.end(&mut pass);
                        }
                        _ => {
                            let encoder = render_context.command_encoder();
                            let span = diagnostics.time_span(encoder, "modern_shadow_restore");
                            copy_layer(encoder, static_cache, working, i as u32);
                            span.end(encoder);
                        }
                    }
                }
            } else {
                // Static pass: clear, then terrain and the static world.
                let mut complete = ready.is_some();
                {
                    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("modern_shadow_static"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: layer,
                            depth_ops: Some(Operations {
                                load: LoadOp::Clear(0.0),
                                store: StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let span = diagnostics.pass_span(&mut pass, "modern_shadow_static");
                    if active {
                        pass.set_bind_group(0, view_bind, &[frame.offsets[i]]);
                        if let (Some(t), Some(tp)) = (terrain, terrain_pipes) {
                            for (mesh_id, center, half, _) in &t.cells {
                                if outside_box(&clip_from_world, *center, *half) {
                                    continue;
                                }
                                let Some(mesh) = meshes.get(*mesh_id) else {
                                    complete = false;
                                    continue;
                                };
                                let Some(pipeline) = tp
                                    .by_stride
                                    .get(&mesh.layout.0.layout().array_stride)
                                    .and_then(|id| pipeline_cache.get_render_pipeline(*id))
                                else {
                                    complete = false;
                                    continue;
                                };
                                let (Some(vslice), Some(islice)) = (
                                    allocator.mesh_vertex_slice(mesh_id),
                                    allocator.mesh_index_slice(mesh_id),
                                ) else {
                                    complete = false;
                                    continue;
                                };
                                let bevy::render::mesh::RenderMeshBufferInfo::Indexed {
                                    index_format,
                                    ..
                                } = &mesh.buffer_info
                                else {
                                    continue;
                                };
                                pass.set_render_pipeline(pipeline);
                                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                                pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                                crate::static_gx::count_draw(1);
                                pass.draw_indexed(
                                    islice.range.clone(),
                                    i32::try_from(vslice.range.start).unwrap_or(0),
                                    0..1,
                                );
                            }
                        } else {
                            complete = false;
                        }
                        if let Some((opaque, cutout)) = ready {
                            for (gpu, draw) in &casters {
                                if outside_cascade(&clip_from_world, draw) {
                                    continue;
                                }
                                // The region's own index buffer is bound by `draw_whole`; the mesh gives vertices.
                                let Some(vslice) = allocator.mesh_vertex_slice(&draw.mesh.id())
                                else {
                                    complete = false;
                                    continue;
                                };
                                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                                gpu.draw_whole(
                                    &mut pass,
                                    vslice.range.start,
                                    |cut, _| Some(if cut { cutout } else { opaque }),
                                    1,
                                );
                            }
                        }
                    }
                    span.end(&mut pass);
                }
                // Either way the layer was just cleared and redrawn, so nothing of the previous
                // frame's dynamic casters survives in it.
                frame.layer_dirty[i].store(false, std::sync::atomic::Ordering::Relaxed);
                if let Some(slot) = frame
                    .dirty_rects
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_mut(i)
                {
                    slot.clear();
                }
                if active && complete {
                    // Only a layer something is restored INTO needs a cached copy: the sky map has
                    // no dynamic pass, so its working layer is its cache (and the static cache no
                    // longer allocates it — `atlas_restores`).
                    if !is_sky {
                        copy_layer(
                            render_context.command_encoder(),
                            working,
                            static_cache,
                            i as u32,
                        );
                    }
                    slot.store(key, std::sync::atomic::Ordering::Relaxed);
                    if let Some(state) = frame
                        .layer_casters
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get_mut(i)
                    {
                        state.drawn = layer_casters;
                        state.waiting_since = None;
                    }
                } else {
                    slot.store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // Dynamic pass, every frame: entity parts on top of the (cached or fresh) static layer.
            if !active || is_sky {
                continue;
            }
            let Some(entities) = world.get_resource::<EntityShadowFrame>() else {
                continue;
            };
            let (Some(draw_bind), Some(skin_bind)) = (&entities.draw_bind, &entities.skin_bind)
            else {
                continue;
            };
            if entities.draws.is_empty() {
                continue;
            }
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("modern_shadow_dynamic"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: layer,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let span = diagnostics.pass_span(&mut pass, "modern_shadow_dynamic");
            pass.set_bind_group(0, view_bind, &[frame.offsets[i]]);
            // Whether anything is actually ISSUED, not whether the list was non-empty: every
            // command is still cascade-culled below, so a far cascade can hold a full draw list
            // and take none of it — and a layer nothing wrote needs no restore next frame.
            let mut drew = false;
            // fork: P8b — and WHERE: the tiles the issued casters' footprints touch in this layer,
            // which is all next frame's restore puts back.
            let mut wrote = [0u32; RESTORE_TILES];
            for cmd in &entities.draws {
                if !cmd.sun || outside_box(&clip_from_world, cmd.center, Vec3::splat(cmd.radius)) {
                    continue;
                }
                let footprint = box_texels(
                    &clip_from_world,
                    cmd.center,
                    Vec3::splat(cmd.radius + RESTORE_MARGIN_YD),
                    SHADOW_MAP_SIZE,
                );
                mark_tiles(&mut wrote, footprint, SHADOW_MAP_SIZE);
                let Some(pipeline) = pipeline_cache.get_render_pipeline(cmd.pipeline) else {
                    continue;
                };
                let Some(mesh) = meshes.get(cmd.mesh) else {
                    continue;
                };
                let Some(vslice) = allocator.mesh_vertex_slice(&cmd.mesh) else {
                    continue;
                };
                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(
                    1,
                    if cmd.skinned { skin_bind } else { draw_bind },
                    &[cmd.offset],
                );
                if let Some(cut) = &cmd.cutout {
                    pass.set_bind_group(2, cut, &[]);
                }
                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                match &mesh.buffer_info {
                    bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } => {
                        let Some(islice) = allocator.mesh_index_slice(&cmd.mesh) else {
                            continue;
                        };
                        pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                        crate::static_gx::count_draw(1);
                        pass.draw_indexed(
                            islice.range.clone(),
                            i32::try_from(vslice.range.start).unwrap_or(0),
                            0..1,
                        );
                        drew = true;
                    }
                    bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => {
                        pass.draw(vslice.range.clone(), 0..1);
                        drew = true;
                    }
                }
            }
            span.end(&mut pass);
            if drew {
                frame.layer_dirty[i].store(true, std::sync::atomic::Ordering::Relaxed);
                if let Some(slot) = frame
                    .dirty_rects
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_mut(i)
                {
                    *slot = tile_runs(&wrote, SHADOW_MAP_SIZE);
                }
            }
        }

        // fork: modern lighting (FORK.md, P6) — the shadowed local lights' cube
        // atlas. Every face is static content lit by a light that does not move, so the whole layer
        // is redrawn only when the chosen set or the static world changes; a settled room pays
        // nothing at all after the first frame.
        draw_local_atlas(
            &mut render_context,
            world,
            frame,
            &casters,
            static_hash,
            ready,
            pipeline_cache,
            meshes,
            allocator,
        );
    }
}

/// fork: modern lighting (P6, extended by P8) — the shadowed local lights' cube atlas.
///
/// **P6** drew only the retained static world here, once, and cached the whole layer against the
/// chosen set and the static fingerprint: a torch does not move, a wall does not move, and a
/// settled room paid nothing after its first frame.
///
/// Since the lamp shadows went soft it is **strictly two phases**, per slot:
///
/// - **Static.** A slot's six faces over the retained static world, drawn ONCE — and only after
///   the lamp's anchor and the static casters inside its reach have held still for
///   [`ATLAS_SETTLE_FRAMES`], so a street that is still streaming in is not redrawn cell by cell.
///   On a tier with a dynamic phase they are drawn into the static cache, never the working
///   layer. The main world is told when a slot is built (`modern::lights::publish_built`) and
///   only then fades its shadow in.
/// - **Dynamic** (`LightTier::dynamic_casters`). The entity casters within
///   [`crate::modern::lights::DYNAMIC_REACH`] of the light — and never the model the light lives
///   in ([`HOST_MAX_RADIUS`]) — fingerprinted per slot. Only when that fingerprint changes is the
///   slot's working copy put back from the cache (six tile rectangles, `depth_fill.wgsl`) and the
///   entities drawn over it. A lamp nobody is walking past redraws nothing at all.
///
/// **The culling is per FACE, not per light.** A cube's six faces are six render passes' worth
/// of state, and most of them look at nothing: a sconce against a wall has three faces that see
/// only the wall, and a lamp on open ground has one that sees only sky. A face with no casters
/// is skipped entirely, which is cheaper than the viewport, scissor and bind-group churn of
/// drawing nothing.
#[allow(clippy::too_many_arguments)]
fn draw_local_atlas(
    render_context: &mut RenderContext<'_, '_>,
    world: &World,
    frame: &ModernShadowFrame,
    casters: &[(&GxCellGpu, &GxCellDraw)],
    static_hash: u64,
    ready: Option<(&RenderPipeline, &RenderPipeline)>,
    pipeline_cache: &PipelineCache,
    meshes: &RenderAssets<RenderMesh>,
    allocator: &MeshAllocator,
) {
    let (Some(lights), Some(view_bind)) =
        (world.get_resource::<ModernLights>(), frame.bind.as_ref())
    else {
        return;
    };
    let Some((opaque, cutout)) = ready else {
        return;
    };
    if lights.shadowed.is_empty() || frame.local_offsets.len() < lights.shadowed.len() * 6 {
        return;
    }
    // The tile's stride in the atlas, and the square of it a face is actually drawn into
    // (`LightTier::tile_use`).
    let tile = (
        SHADOW_MAP_SIZE / ATLAS_TILES,
        crate::modern::lights::face_px(&lights.tier, SHADOW_MAP_SIZE),
    );
    let pipes = world.get_resource::<GxShadowPipelines>();
    // fork: P8b — the tile clear that lets one slot be redrawn without the other five. Until it
    // is compiled nothing is drawn: a slot is only ever redrawn alone.
    let Some(fill) = pipes.and_then(|p| pipeline_cache.get_render_pipeline(p.fill)) else {
        return;
    };
    let restore = pipes.and_then(|p| {
        Some((
            pipeline_cache.get_render_pipeline(p.restore)?,
            &p.restore_layout,
        ))
    });
    let (Some(cache), Some(working)) = (frame.cache.as_ref(), frame.texture.as_ref()) else {
        return;
    };
    let entities = lights
        .tier
        .dynamic_casters
        .then(|| world.get_resource::<EntityShadowFrame>())
        .flatten()
        .filter(|e| e.draw_bind.is_some() && e.skin_bind.is_some());
    let mut atlas = frame
        .atlas
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    for layer_index in 0..lights.tier.atlas_layers() {
        let slots = layer_index * PER_ATLAS_LAYER
            ..((layer_index + 1) * PER_ATLAS_LAYER).min(lights.shadowed.len());
        if slots.is_empty() {
            continue;
        }
        let array_layer = LOCAL_LAYER + layer_index;
        let Some(view) = frame.layers.get(array_layer) else {
            continue;
        };
        // Where the static phase draws: into the cache when a dynamic phase copies from it, else
        // the working layer itself.
        let cached = lights
            .tier
            .dynamic_casters
            .then(|| {
                frame
                    .cache_targets
                    .get(array_layer)
                    .zip(frame.cache_views.get(array_layer))
            })
            .flatten();
        let static_target = cached.map_or(view, |(target, _)| target);

        // ---- The static phase: settle, then draw once. -------------------------------------
        let mut build = 0u32;
        for s in slots.clone() {
            let held = lights.shadowed[s].map(|h| h.light);
            let light_key = held.as_ref().map_or(ATLAS_HOLE, ShadowLight::map_key);
            let slot = &mut atlas.slots[s];
            if slot.light != light_key || slot.world != static_hash {
                slot.casters = held.as_ref().map_or(0, |l| atlas_casters_key(casters, l));
                slot.light = light_key;
                slot.world = static_hash;
            }
            let wanted = slot.wanted();
            if slot.drawn == wanted {
                continue;
            }
            if held.is_none() {
                // A hole is never read — its light faded to zero before letting go — so its
                // tiles stay as they are until the next light draws over them.
                slot.drawn = wanted;
                crate::modern::lights::publish_built(s, 0);
                continue;
            }
            if slot.pending != wanted {
                slot.pending = wanted;
                slot.waited = 0;
                slot.attempts = 0;
            }
            slot.waited += 1;
            if slot.waited >= ATLAS_SETTLE_FRAMES {
                build |= 1 << (s - slots.start);
            }
        }
        if build != 0 {
            let complete = draw_atlas_statics(
                render_context,
                frame,
                static_target,
                view_bind,
                &lights.shadowed[slots.clone()],
                slots.start,
                tile,
                casters,
                (opaque, cutout),
                allocator,
                build,
                fill,
            );
            for s in slots.clone() {
                let bit = 1 << (s - slots.start);
                if build & bit == 0 {
                    continue;
                }
                let slot = &mut atlas.slots[s];
                if cached.is_some() {
                    // The working tiles still hold the old faces: they take a fresh copy below.
                    slot.dynamic = DYN_UNKNOWN;
                }
                slot.attempts += 1;
                bevy::log::debug!(
                    "lamp atlas: slot {s} static faces drawn at {:?} (complete {}, attempt {})",
                    lights.shadowed[s].map(|h| h.light.pos),
                    complete & bit != 0,
                    slot.attempts
                );
                if complete & bit != 0 || slot.attempts >= ATLAS_ATTEMPTS {
                    slot.drawn = slot.pending;
                    crate::modern::lights::publish_built(s, slot.light);
                } else {
                    // A caster was not resident yet: wait out another settle and try again.
                    slot.drawn = 0;
                    slot.waited = 0;
                }
            }
        }
        let Some((_, cache_view)) = cached else {
            continue;
        };

        // ---- The dynamic phase: per slot, only what stands right beside its light. ----------
        // fork: per slot, a mask of the six FACES to restore and to redraw — only the faces a
        // caster that arrived, left, moved or turned can be seen in. A body walking past a
        // wall sconce dirties the one or two faces looking its way, not all six.
        let mut near: Vec<Vec<&EntityDrawCmd>> = vec![Vec::new(); slots.len()];
        let mut keys = [DYN_STATIC; PER_ATLAS_LAYER];
        let mut restore_faces = [0u8; PER_ATLAS_LAYER];
        let mut draw_faces = [0u8; PER_ATLAS_LAYER];
        let AtlasCache {
            slots: slot_state,
            casters_drawn,
        } = &mut *atlas;
        for (i, s) in slots.clone().enumerate() {
            let Some(light) = lights.shadowed[s].map(|h| h.light) else {
                continue;
            };
            let slot = &slot_state[s];
            // Nothing to show until this light's static faces exist; nobody reads the slot.
            if crate::modern::lights::built(s) != slot.light {
                continue;
            }
            if let Some(e) = entities {
                near[i] = dynamic_casters_for(&e.draws, &light);
            }
            keys[i] = if near[i].is_empty() {
                DYN_STATIC
            } else {
                DYN_DRAWN
            };
            let now: Vec<DrawnCaster> = near[i].iter().map(|c| DrawnCaster::of(c)).collect();
            let dirty = match slot.dynamic {
                DYN_STATIC => dirty_faces(light.pos, &[], &now),
                DYN_DRAWN => dirty_faces(light.pos, &casters_drawn[s], &now),
                _ => ALL_FACES,
            };
            if dirty == 0 {
                continue;
            }
            if slot.dynamic != DYN_STATIC {
                restore_faces[i] = dirty;
            }
            if keys[i] != DYN_STATIC {
                draw_faces[i] = dirty;
            }
        }
        if restore_faces.iter().chain(&draw_faces).all(|&m| m == 0) {
            continue;
        }
        bevy::log::debug!(
                "lamp atlas: layer {layer_index} dynamic phase — restore {restore_faces:?}, draw {draw_faces:?}"
            );
        if restore_faces.iter().any(|&m| m != 0) {
            match restore {
                Some((pipeline, layout)) => restore_atlas_slots(
                    render_context,
                    frame,
                    world.resource::<RenderQueue>(),
                    pipeline_cache,
                    pipeline,
                    layout,
                    cache_view,
                    view,
                    slots.start,
                    &restore_faces[..slots.len()],
                    tile.0,
                ),
                None => {
                    // No restore draw yet: the whole layer back, and every slot that has
                    // entities beside it draws them again, every face.
                    copy_layer(
                        render_context.command_encoder(),
                        cache,
                        working,
                        array_layer as u32,
                    );
                    for (i, s) in slots.clone().enumerate() {
                        if crate::modern::lights::built(s) != slot_state[s].light {
                            continue;
                        }
                        restore_faces[i] = ALL_FACES;
                        if keys[i] != DYN_STATIC {
                            draw_faces[i] = ALL_FACES;
                        }
                    }
                }
            }
        }
        if let (Some(entities), true) = (entities, draw_faces.iter().any(|&m| m != 0)) {
            draw_atlas_entities(
                render_context,
                frame,
                view,
                view_bind,
                entities,
                &lights.shadowed[slots.clone()],
                &near,
                slots.start,
                &draw_faces[..slots.len()],
                tile,
                pipeline_cache,
                meshes,
                allocator,
            );
        }
        for (i, s) in slots.clone().enumerate() {
            if restore_faces[i] | draw_faces[i] != 0 {
                slot_state[s].dynamic = keys[i];
                let now: Vec<DrawnCaster> = near[i].iter().map(|c| DrawnCaster::of(c)).collect();
                let redrawn = if keys[i] == DYN_STATIC {
                    ALL_FACES
                } else {
                    draw_faces[i]
                };
                let light_pos = lights.shadowed[s].map_or(Vec3::ZERO, |h| h.light.pos);
                casters_drawn[s] = settle_drawn(light_pos, &casters_drawn[s], &now, redrawn);
            }
        }
    }
}

/// The static half of one atlas layer: clear the slots named in `redraw`, then replay the
/// retained world into each of their faces that has anything to see. Returns, as the same kind
/// of mask, the slots whose every caster was resident.
///
/// fork: P8b — only the named slots' tiles are cleared (by drawing over them,
/// `depth_fill.wgsl`) and the rest of the layer is loaded untouched.
#[allow(clippy::too_many_arguments)]
fn draw_atlas_statics(
    render_context: &mut RenderContext<'_, '_>,
    frame: &ModernShadowFrame,
    view: &TextureView,
    view_bind: &BindGroup,
    shadowed: &[Option<crate::modern::lights::HeldSlot>],
    slot_base: usize,
    tile: (u32, u32),
    casters: &[(&GxCellGpu, &GxCellDraw)],
    pipes: (&RenderPipeline, &RenderPipeline),
    allocator: &MeshAllocator,
    redraw: u32,
    fill: &RenderPipeline,
) -> u32 {
    let (opaque, cutout) = pipes;
    let mut complete = 0u32;
    let mut near: Vec<&(&GxCellGpu, &GxCellDraw)> = Vec::new();
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("modern_local_shadow"),
        color_attachments: &[],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let span = diagnostics.pass_span(&mut pass, "modern_local_shadow");
    for (i, light) in shadowed.iter().enumerate() {
        let bit = 1u32 << i;
        if redraw & bit == 0 {
            continue;
        }
        let slot = slot_base + i;
        // This slot's six tiles back to the far plane.
        for face in 0..6usize {
            set_atlas_tile(&mut pass, slot, face, (tile.0, tile.0));
            pass.set_render_pipeline(fill);
            pass.draw(0..3, 0..1);
        }
        // A hole's six tiles keep that clear, which reads as "nothing occludes" — exactly what
        // an unheld slot should mean.
        let Some(light) = light.map(|h| h.light) else {
            complete |= bit;
            continue;
        };
        let mut whole = true;
        // Sphere, not frustum: a cube face's clip is a perspective, and the planar corner test
        // the cascades use flips sign behind the light. A torch reaches a handful of cells, so
        // the six faces share one cheap list, which is then split by face below.
        let reach = light.range;
        near.clear();
        near.extend(casters.iter().filter(|(_, draw)| {
            let center = draw.origin + Vec3::from(draw.aabb.center);
            let radius = Vec3::from(draw.aabb.half_extents).length();
            center.distance(light.pos) <= reach + radius
        }));
        for face in 0..6usize {
            let t = slot * 6 + face;
            let Some(&offset) = frame.local_offsets.get(t) else {
                whole = false;
                continue;
            };
            // fork: P8 — the per-face cull, and the reason the tile is bound LAZILY: a caster
            // reaches this face only if its box touches the face's 90° cone, and a face nothing
            // touches must not pay for a viewport, a scissor and a bind group to draw nothing.
            let mut set_up = false;
            for (gpu, draw) in near.iter() {
                if !face_sees(
                    face,
                    light.pos,
                    draw.origin + Vec3::from(draw.aabb.center),
                    Vec3::from(draw.aabb.half_extents).length(),
                ) {
                    continue;
                }
                // The region's own index buffer is bound by `draw_whole`; the mesh gives vertices.
                let Some(vslice) = allocator.mesh_vertex_slice(&draw.mesh.id()) else {
                    whole = false;
                    continue;
                };
                if !set_up {
                    set_atlas_tile(&mut pass, slot, face, tile);
                    pass.set_bind_group(0, view_bind, &[offset]);
                    set_up = true;
                }
                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                gpu.draw_whole(
                    &mut pass,
                    vslice.range.start,
                    |cut, _| Some(if cut { cutout } else { opaque }),
                    1,
                );
            }
        }
        if whole {
            complete |= bit;
        }
    }
    span.end(&mut pass);
    complete
}

/// fork: the dynamic half — each slot's faces named in `faces` get its own pruned entity list
/// (`near`, from [`dynamic_casters_for`]) drawn over their freshly restored static depth.
#[allow(clippy::too_many_arguments)]
fn draw_atlas_entities(
    render_context: &mut RenderContext<'_, '_>,
    frame: &ModernShadowFrame,
    view: &TextureView,
    view_bind: &BindGroup,
    entities: &EntityShadowFrame,
    shadowed: &[Option<crate::modern::lights::HeldSlot>],
    near: &[Vec<&EntityDrawCmd>],
    slot_base: usize,
    faces: &[u8],
    tile: (u32, u32),
    pipeline_cache: &PipelineCache,
    meshes: &RenderAssets<RenderMesh>,
    allocator: &MeshAllocator,
) {
    let (Some(draw_bind), Some(skin_bind)) = (&entities.draw_bind, &entities.skin_bind) else {
        return;
    };
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("modern_local_shadow_dynamic"),
        color_attachments: &[],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let span = diagnostics.pass_span(&mut pass, "modern_local_shadow_dynamic");
    for (i, light) in shadowed.iter().enumerate() {
        let mask = faces.get(i).copied().unwrap_or(0);
        if mask == 0 {
            continue;
        }
        let (Some(light), Some(list)) = (light.map(|h| h.light), near.get(i)) else {
            continue;
        };
        let slot = slot_base + i;
        for face in 0..6usize {
            if mask & (1 << face) == 0 {
                continue;
            }
            let t = slot * 6 + face;
            let Some(&offset) = frame.local_offsets.get(t) else {
                continue;
            };
            let mut set_up = false;
            for cmd in list {
                if !face_sees(face, light.pos, cmd.center, cmd.radius) {
                    continue;
                }
                let Some(pipeline) = pipeline_cache.get_render_pipeline(cmd.pipeline) else {
                    continue;
                };
                let Some(mesh) = meshes.get(cmd.mesh) else {
                    continue;
                };
                let Some(vslice) = allocator.mesh_vertex_slice(&cmd.mesh) else {
                    continue;
                };
                if !set_up {
                    set_atlas_tile(&mut pass, slot, face, tile);
                    pass.set_bind_group(0, view_bind, &[offset]);
                    set_up = true;
                }
                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(
                    1,
                    if cmd.skinned { skin_bind } else { draw_bind },
                    &[cmd.offset],
                );
                if let Some(cut) = &cmd.cutout {
                    pass.set_bind_group(2, cut, &[]);
                }
                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                match &mesh.buffer_info {
                    bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } => {
                        let Some(islice) = allocator.mesh_index_slice(&cmd.mesh) else {
                            continue;
                        };
                        pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                        crate::static_gx::count_draw(1);
                        pass.draw_indexed(
                            islice.range.clone(),
                            i32::try_from(vslice.range.start).unwrap_or(0),
                            0..1,
                        );
                    }
                    bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => {
                        pass.draw(vslice.range.clone(), 0..1);
                    }
                }
            }
        }
    }
    span.end(&mut pass);
}

/// fork: put the tiles `faces` names (per slot, a mask of its six faces) back from the static cache — the cascade
/// restore's draw (`depth_fill.wgsl`), one rectangle per tile. wgpu refuses a partial copy of a depth
/// format, and a whole-layer copy is 16.8 MB to undo one character walking past one lamp.
#[allow(clippy::too_many_arguments)]
fn restore_atlas_slots(
    render_context: &mut RenderContext<'_, '_>,
    frame: &ModernShadowFrame,
    queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    pipeline: &RenderPipeline,
    layout: &BindGroupLayoutDescriptor,
    cache_view: &TextureView,
    target: &TextureView,
    slot_base: usize,
    faces: &[u8],
    tile: u32,
) {
    let mut rects: Vec<[f32; 4]> = Vec::new();
    for (i, &mask) in faces.iter().enumerate() {
        for face in 0..6usize {
            if mask & (1 << face) == 0 {
                continue;
            }
            let t = ((slot_base + i) % PER_ATLAS_LAYER) * 6 + face;
            let (tx, ty) = (t as u32 % ATLAS_TILES, t as u32 / ATLAS_TILES);
            rects.push([
                (tx * tile) as f32,
                (ty * tile) as f32,
                ((tx + 1) * tile) as f32,
                ((ty + 1) * tile) as f32,
            ]);
        }
    }
    if rects.is_empty() {
        return;
    }
    let device = render_context.render_device();
    let buffer = restore_buffer(frame, device, queue, &rects);
    let bind = device.create_bind_group(
        "modern_local_shadow_restore",
        &pipeline_cache.get_bind_group_layout(layout),
        &BindGroupEntries::sequential((cache_view, buffer.as_entire_binding())),
    );
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("modern_local_shadow_restore"),
        color_attachments: &[],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: target,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let span = diagnostics.pass_span(&mut pass, "modern_local_shadow_restore");
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.draw(0..6, 0..rects.len() as u32);
    span.end(&mut pass);
}

/// fork: the entity casters one lamp's dynamic phase draws: within
/// [`crate::modern::lights::DYNAMIC_REACH`] of the light (or its range, if that is shorter), and not
/// the light's own host — a part no bigger than [`HOST_MAX_RADIUS`] whose bounds hold the light.
fn dynamic_casters_for<'a>(
    draws: &'a [EntityDrawCmd],
    light: &ShadowLight,
) -> Vec<&'a EntityDrawCmd> {
    let reach = light.range.min(crate::modern::lights::DYNAMIC_REACH);
    draws
        .iter()
        .filter(|cmd| {
            let d = cmd.center.distance(light.pos);
            let host = cmd.tight <= HOST_MAX_RADIUS && d < cmd.tight + light.slack;
            d <= reach + cmd.radius && !host
        })
        .collect()
}

/// fork: pair each of this frame's casters beside a lamp with the one its faces were drawn with —
/// the same part, moved no more than [`DYN_MOVE_YD`] and turned no more than [`DYN_TURN`] since.
/// `None` = no such part: it arrived, or it moved. A creature idling beside a lamp, a tree swaying in
/// its bones and a lantern's jiggling ornament all pair, so their shadows are not redrawn every frame
/// they animate. `None` for the whole list when there are too many to pair (a crowd): every face.
fn match_casters(drawn: &[DrawnCaster], now: &[DrawnCaster]) -> Option<Vec<Option<usize>>> {
    let mut taken = [false; 64];
    if drawn.len() > taken.len() || now.len() > taken.len() {
        return None;
    }
    Some(
        now.iter()
            .map(|c| {
                let k = drawn.iter().enumerate().position(|(k, d)| {
                    !taken[k]
                        && d.mesh == c.mesh
                        && d.at.distance(c.at) <= DYN_MOVE_YD
                        && d.turn.angle_between(c.turn) <= DYN_TURN
                })?;
                taken[k] = true;
                Some(k)
            })
            .collect(),
    )
}

/// fork: the faces of `light`'s cube one caster can be seen in, as a six-bit mask.
fn faces_of(light: Vec3, c: &DrawnCaster) -> u8 {
    (0..6usize).fold(0, |m, face| {
        if face_sees(face, light, c.center, c.radius) {
            m | 1 << face
        } else {
            m
        }
    })
}

/// fork: **which faces of a lamp's cube are stale** — the faces a caster that arrived is seen in, the
/// faces one that left was seen in, and both for one that moved or turned (it is at once a caster
/// leaving where it was and one arriving where it is). `0` = the working faces are exactly right.
/// This replaced an all-or-nothing test: a character walking past a sconce redrew all six faces,
/// which is what forced the old quarter-yard tolerance that made its shadow trail it.
fn dirty_faces(light: Vec3, drawn: &[DrawnCaster], now: &[DrawnCaster]) -> u8 {
    let Some(pairs) = match_casters(drawn, now) else {
        return ALL_FACES;
    };
    let mut dirty = 0u8;
    let mut kept = [false; 64];
    for (c, pair) in now.iter().zip(&pairs) {
        match pair {
            Some(k) => kept[*k] = true,
            None => dirty |= faces_of(light, c),
        }
    }
    for (k, d) in drawn.iter().enumerate() {
        if !kept[k] {
            dirty |= faces_of(light, d);
        }
    }
    dirty
}

/// fork: what a slot's faces hold once `redrawn` of them have been drawn with `now` — the record
/// [`dirty_faces`] tests against next frame. A caster whose every face was redrawn is recorded where
/// it is now; one that paired with a caster already drawn and was NOT fully redrawn keeps the pose it
/// was drawn at. Recording its new pose anyway would let a part creeping under [`DYN_MOVE_YD`] a frame
/// drift away from its drawn shadow for as long as some other caster kept dirtying the slot.
fn settle_drawn(
    light: Vec3,
    drawn: &[DrawnCaster],
    now: &[DrawnCaster],
    redrawn: u8,
) -> Vec<DrawnCaster> {
    let pairs = match_casters(drawn, now);
    now.iter()
        .enumerate()
        .map(|(i, c)| {
            let fully = faces_of(light, c) & !redrawn == 0;
            match pairs.as_ref().and_then(|p| p[i]) {
                Some(k) if !fully => drawn[k],
                _ => *c,
            }
        })
        .collect()
}

/// Point one render pass at one atlas tile — the viewport and the scissor together, because the
/// scissor is what keeps a face's filter taps out of its neighbour.
fn set_atlas_tile(
    pass: &mut bevy::render::render_phase::TrackedRenderPass<'_>,
    slot: usize,
    face: usize,
    (tile, drawn): (u32, u32),
) {
    let t = (slot % PER_ATLAS_LAYER) * 6 + face;
    let (tx, ty) = (t as u32 % ATLAS_TILES, t as u32 / ATLAS_TILES);
    // The face is drawn into the tile's top-left `drawn` texels (the whole tile on every tier but
    // Cinematic); the lookup scales its uv by the same fraction (`locallight::tile_use`).
    pass.set_viewport(
        (tx * tile) as f32,
        (ty * tile) as f32,
        drawn as f32,
        drawn as f32,
        0.0,
        1.0,
    );
    pass.set_scissor_rect(tx * tile, ty * tile, drawn, drawn);
}

/// fork: P8 — **does one cube face see this sphere at all?**
///
/// A face is the 90° cone about its major axis, so a sphere is visible from it iff the sphere
/// reaches past the two diagonal planes that bound that cone. Testing the sphere's centre against
/// the plane `axis·d >= |other|` — widened by the radius over √2, the planes' own normalisation —
/// is exact for a sphere and costs three multiplies and a compare.
///
/// This is the cheap half of the aggressive culling: a sconce flat against a wall has three faces
/// that see nothing, and a face that sees nothing is a pass that never binds anything.
fn face_sees(face: usize, light: Vec3, center: Vec3, radius: f32) -> bool {
    let d = center - light;
    let (along, a, b) = match face {
        0 => (d.x, d.y, d.z),
        1 => (-d.x, d.y, d.z),
        2 => (d.y, d.x, d.z),
        3 => (-d.y, d.x, d.z),
        4 => (d.z, d.x, d.y),
        _ => (-d.z, d.x, d.y),
    };
    // The cone's two bounding planes have normals `(axis ∓ other)/√2`; a sphere clears both iff
    // `along + radius·√2 >= |a|` and the same for `b`.
    let slack = radius * std::f32::consts::SQRT_2;
    along + slack >= a.abs() && along + slack >= b.abs()
}

/// An empty atlas slot's light key. Distinct from every real light's (those are forced odd below)
/// and from 0, "never fingerprinted" — a slot emptying still has to be redrawn, or its tiles keep
/// the departed light's depth and the next light to take that slot inherits it.
const ATLAS_HOLE: u64 = 2;

/// fork: P8b — the static casters one light's cube can see, fingerprinted: exactly the set
/// `draw_atlas_statics` replays for it (the same sphere test), each caster hashed the way the
/// world's own fingerprint hashes it, combined order-independently. This is what lets the world's
/// static set change — a cell streaming in across the zone — without a lamp redrawing six faces of
/// geometry it cannot see.
fn atlas_casters_key(
    casters: &[(&GxCellGpu, &GxCellDraw)],
    light: &crate::modern::lights::ShadowLight,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut acc: u64 = 0x2545_f491_4f6c_dd1d;
    for (gpu, draw) in casters {
        let center = draw.origin + Vec3::from(draw.aabb.center);
        let radius = Vec3::from(draw.aabb.half_extents).length();
        if center.distance(light.pos) > light.range + radius {
            continue;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        gpu.mesh.hash(&mut h);
        gpu.killed_applied.hash(&mut h);
        gpu.runs.len().hash(&mut h);
        draw.origin.to_array().map(f32::to_bits).hash(&mut h);
        acc = acc.wrapping_add(h.finish());
    }
    acc
}

/// The cache key of one cascade: its light clip and the static caster fingerprint (never 0).
fn cascade_key(clip_from_world: &Mat4, static_hash: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    clip_from_world
        .to_cols_array()
        .map(f32::to_bits)
        .hash(&mut h);
    static_hash.hash(&mut h);
    h.finish().max(1)
}

/// fork (perf): layer `i`'s cache key this frame and its caster fingerprint. The fingerprint is
/// re-taken only when the light clip or the world's static set moved.
fn layer_key(
    frame: &ModernShadowFrame,
    i: usize,
    clip_from_world: &Mat4,
    static_hash: u64,
    casters: &[(&GxCellGpu, &GxCellDraw)],
    terrain: Option<&ModernTerrainCasters>,
    now: Instant,
) -> (u64, u64) {
    let mut states = frame
        .layer_casters
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = states.get_mut(i) else {
        return (cascade_key(clip_from_world, static_hash), static_hash);
    };
    let clip = cascade_key(clip_from_world, 0);
    if state.clip != clip || state.world != static_hash {
        state.clip = clip;
        state.world = static_hash;
        let fingerprint = layer_casters_key(clip_from_world, casters, terrain);
        if fingerprint != state.casters {
            state.casters = fingerprint;
            state.changed_at = Some((now, frame.frame_no));
        }
    }
    // A set that reverted to the drawn one is no longer waiting.
    if state.casters == state.drawn {
        state.waiting_since = None;
    }
    (cascade_key(clip_from_world, state.casters), state.casters)
}

/// fork (perf): may layer `i` keep showing its cached content although its casters changed? Only
/// while the light clip is the one it was drawn under, and only until the new set has held still
/// for [`LAYER_SETTLE`] or waited [`LAYER_MAX_DEFER`].
fn defers_redraw(
    frame: &ModernShadowFrame,
    i: usize,
    clip_from_world: &Mat4,
    drawn: u64,
    now: Instant,
) -> bool {
    let mut states = frame
        .layer_casters
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = states.get_mut(i) else {
        return false;
    };
    if drawn == 0 || drawn != cascade_key(clip_from_world, state.drawn) {
        return false;
    }
    let at = (now, frame.frame_no);
    let within = |(t, f): (Instant, u32), time: Duration, frames: u32| {
        now.saturating_duration_since(t) < time || at.1.wrapping_sub(f) < frames
    };
    let since = *state.waiting_since.get_or_insert(at);
    state
        .changed_at
        .is_some_and(|c| within(c, LAYER_SETTLE, LAYER_SETTLE_FRAMES))
        && within(since, LAYER_MAX_DEFER, LAYER_MAX_DEFER_FRAMES)
}

/// fork (perf): the static casters inside one layer's light-space rectangle, fingerprinted
/// order-independently — exactly the set the layer's static pass replays.
fn layer_casters_key(
    clip_from_world: &Mat4,
    casters: &[(&GxCellGpu, &GxCellDraw)],
    terrain: Option<&ModernTerrainCasters>,
) -> u64 {
    use std::hash::{BuildHasher, Hash, Hasher};
    let mut acc: u64 = 0x9e37_79b9_7f4a_7c15;
    for (mesh, center, half, lod) in terrain.map(|t| t.cells.as_slice()).unwrap_or_default() {
        if outside_box(clip_from_world, *center, *half) {
            continue;
        }
        let mut h = bevy::platform::hash::FixedHasher.build_hasher();
        mesh.hash(&mut h);
        center.to_array().map(f32::to_bits).hash(&mut h);
        // A cell rebuilt at a new chunk level keeps its handle; its epoch says the mesh moved.
        lod.hash(&mut h);
        acc = acc.wrapping_add(h.finish());
    }
    for (gpu, draw) in casters {
        if outside_cascade(clip_from_world, draw) {
            continue;
        }
        let mut h = bevy::platform::hash::FixedHasher.build_hasher();
        gpu.mesh.hash(&mut h);
        gpu.killed_applied.hash(&mut h);
        gpu.unready.hash(&mut h);
        for run in &gpu.whole {
            (run.index_range.start, run.index_range.end, run.cutout).hash(&mut h);
        }
        draw.origin.to_array().map(f32::to_bits).hash(&mut h);
        acc = acc.wrapping_add(h.finish());
    }
    acc.max(1)
}

/// Copy one whole array layer between two same-sized `Depth32Float` arrays.
fn copy_layer(encoder: &mut CommandEncoder, src: &Texture, dst: &Texture, layer: u32) {
    let size = Extent3d {
        width: src.width(),
        height: src.height(),
        depth_or_array_layers: 1,
    };
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: TextureAspect::All,
        },
        size,
    );
}

/// Does a cell's world box miss the cascade's light-space rectangle? Depth is not tested: a caster
/// between the light and the near plane still casts (the shader pancakes it).
fn outside_cascade(clip_from_world: &Mat4, draw: &GxCellDraw) -> bool {
    // `aabb` is relative to the bake's recentring origin (`bake.rs`).
    outside_box(
        clip_from_world,
        draw.origin + Vec3::from(draw.aabb.center),
        Vec3::from(draw.aabb.half_extents),
    )
}

/// Does a world box miss the cascade's light-space rectangle? (Depth untested, as above.)
fn outside_box(clip_from_world: &Mat4, center: Vec3, half: Vec3) -> bool {
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for i in 0..8u8 {
        let sign = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        let p = clip_from_world.project_point3(center + half * sign);
        lo = lo.min(p.truncate());
        hi = hi.max(p.truncate());
    }
    hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0
}

/// fork: P8b — the texel rectangle a world box covers in a cascade (its light-space footprint, rows
/// running down as a render target's do), widened by [`RESTORE_MARGIN_PX`] and clamped to the layer.
/// Depth is not considered: a caster in front of the near plane still writes (the shader pancakes it).
fn box_texels(clip_from_world: &Mat4, center: Vec3, half: Vec3, size: u32) -> URect {
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for i in 0..8u8 {
        let sign = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        let p = clip_from_world
            .project_point3(center + half * sign)
            .truncate();
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let lo = lo.clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
    let hi = hi.clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
    let s = size as f32;
    let m = RESTORE_MARGIN_PX as f32;
    // NDC y up, texel rows down.
    let x0 = ((lo.x * 0.5 + 0.5) * s - m).floor().clamp(0.0, s);
    let x1 = ((hi.x * 0.5 + 0.5) * s + m).ceil().clamp(0.0, s);
    let y0 = ((0.5 - hi.y * 0.5) * s - m).floor().clamp(0.0, s);
    let y1 = ((0.5 - lo.y * 0.5) * s + m).ceil().clamp(0.0, s);
    URect::new(x0 as u32, y0 as u32, x1 as u32, y1 as u32)
}

/// fork: P8b — mark the restore tiles a texel rectangle touches (one bit per tile, a word per row).
fn mark_tiles(tiles: &mut [u32; RESTORE_TILES], rect: URect, size: u32) {
    if rect.is_empty() {
        return;
    }
    let tile = (size / RESTORE_TILES as u32).max(1);
    let last = RESTORE_TILES as u32 - 1;
    let (x0, x1) = (
        (rect.min.x / tile).min(last),
        ((rect.max.x - 1) / tile).min(last),
    );
    let (y0, y1) = (
        (rect.min.y / tile).min(last),
        ((rect.max.y - 1) / tile).min(last),
    );
    // Bits x0..=x1, built in 64 bits so a full row does not overflow the shift.
    let bits = (((1u64 << (x1 + 1)) - 1) & !((1u64 << x0) - 1)) as u32;
    for row in &mut tiles[y0 as usize..=y1 as usize] {
        *row |= bits;
    }
}

/// fork: P8b — the marked tiles as texel rectangles: one per run of set bits in a row, a run that
/// repeats on the next row extending down instead of starting again. Disjoint by construction.
fn tile_runs(tiles: &[u32; RESTORE_TILES], size: u32) -> Vec<URect> {
    let tile = (size / RESTORE_TILES as u32).max(1);
    let mut out: Vec<URect> = Vec::new();
    // The runs that ended on the previous row, as (x0, x1, index into `out`).
    let mut open: Vec<(u32, u32, usize)> = Vec::new();
    for (y, &row) in tiles.iter().enumerate() {
        let mut next = Vec::new();
        let mut bits = row;
        while bits != 0 {
            let x0 = bits.trailing_zeros();
            let run = (bits >> x0).trailing_ones();
            let x1 = x0 + run;
            bits &= !((((1u64 << x1) - 1) & !((1u64 << x0) - 1)) as u32);
            let y1 = (y as u32 + 1) * tile;
            match open.iter().find(|(a, b, _)| *a == x0 && *b == x1) {
                Some(&(_, _, i)) => {
                    out[i].max.y = y1;
                    next.push((x0, x1, i));
                }
                None => {
                    out.push(URect::new(x0 * tile, y as u32 * tile, x1 * tile, y1));
                    next.push((x0, x1, out.len() - 1));
                }
            }
        }
        open = next;
    }
    out
}

/// Wire the pass before the main pass (called from `render::build`).
pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .init_resource::<CasterViewBuffer>()
        .init_resource::<EntityDrawBuffer>()
        .init_resource::<CutoutBinds>()
        .init_resource::<EntityShadowFrame>()
        .init_resource::<ModernShadowFrame>()
        .add_systems(
            bevy::render::RenderStartup,
            init_shadow_pipelines.after(init_pipelines),
        )
        .add_systems(
            Render,
            (
                prepare_modern_shadow,
                queue_terrain_shadow_pipelines,
                prepare_entity_shadows,
            )
                .in_set(RenderSystems::PrepareResources),
        )
        .add_systems(
            Core3d,
            gx_shadow_pass
                .after(per_view_shadow_pass::<LATE_SHADOW_PASS>)
                .before(Core3dSystems::MainPass),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The restore's rectangles cover every marked tile exactly once — disjoint, so no texel is
    /// written twice — and nothing else.
    /// The near cascades redraw their entities every frame; the far ones alternate, never both in
    /// one frame and never neither.
    #[test]
    fn the_far_cascades_take_their_entities_on_alternate_frames() {
        for frame_no in 0..8 {
            assert!(!holds_entities(0, frame_no) && !holds_entities(1, frame_no));
            assert_ne!(holds_entities(2, frame_no), holds_entities(3, frame_no));
            assert_ne!(holds_entities(2, frame_no), holds_entities(2, frame_no + 1));
        }
    }

    #[test]
    fn restore_runs_cover_the_marked_tiles_once() {
        let size = SHADOW_MAP_SIZE;
        let tile = size / RESTORE_TILES as u32;
        let mut tiles = [0u32; RESTORE_TILES];
        // A caster in the middle, one across the right edge, and one spanning a whole row.
        mark_tiles(&mut tiles, URect::new(700, 700, 900, 1000), size);
        mark_tiles(&mut tiles, URect::new(size - 10, 0, size, 30), size);
        mark_tiles(&mut tiles, URect::new(0, size - tile, size, size), size);
        let runs = tile_runs(&tiles, size);
        let mut covered = [0u32; RESTORE_TILES];
        for r in &runs {
            assert_eq!(r.min.x % tile, 0);
            assert_eq!(r.max.y % tile, 0);
            for y in r.min.y / tile..r.max.y / tile {
                for x in r.min.x / tile..r.max.x / tile {
                    assert_eq!(
                        covered[y as usize] & (1 << x),
                        0,
                        "tile {x},{y} written twice"
                    );
                    covered[y as usize] |= 1 << x;
                }
            }
        }
        assert_eq!(covered, tiles, "the runs are exactly the marked tiles");
        // A caster's footprint always lands inside what is restored.
        let footprint = URect::new(700, 700, 900, 1000);
        assert!(runs.iter().any(|r| r.contains(footprint.min)));
        assert_eq!(tiles[RESTORE_TILES - 1], u32::MAX, "a full row is one run");
    }

    fn body(at: Vec3) -> DrawnCaster {
        DrawnCaster {
            mesh: AssetId::default(),
            at,
            turn: Quat::IDENTITY,
            center: at,
            radius: 0.5,
        }
    }

    /// A caster standing still — or jiggling under the float-noise tolerance — dirties nothing.
    #[test]
    fn a_caster_that_holds_still_dirties_no_face() {
        let light = Vec3::ZERO;
        let b = body(Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(dirty_faces(light, &[b], &[b]), 0);
        let jiggle = body(b.at + Vec3::new(0.0, 0.01, 0.0));
        assert_eq!(
            dirty_faces(
                light,
                &[b],
                &[DrawnCaster {
                    center: b.center,
                    ..jiggle
                }]
            ),
            0
        );
    }

    /// GOLDEN — **a body walking past a lamp redraws its shadow on the step, not a quarter-yard
    /// later.** The old tolerance was 0.25 yd, which is what made the shadow trail the body.
    #[test]
    fn a_small_step_is_redrawn() {
        let light = Vec3::ZERO;
        let b = body(Vec3::new(4.0, 0.0, 0.0));
        let step = body(b.at + Vec3::new(0.0, 0.0, 0.05));
        assert_ne!(
            dirty_faces(light, &[b], &[step]),
            0,
            "a 5 cm step is movement"
        );
        let turned = DrawnCaster {
            turn: Quat::from_rotation_y(0.05),
            ..b
        };
        assert_ne!(
            dirty_faces(light, &[b], &[turned]),
            0,
            "and so is turning on the spot"
        );
    }

    /// Only the faces a caster is seen in are stale: a body on the lamp's +X side dirties the +X
    /// face alone, and one straddling the +X/+Z edge dirties both of those and nothing else.
    #[test]
    fn a_caster_dirties_only_the_faces_it_is_seen_in() {
        let light = Vec3::ZERO;
        let arrived = body(Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(dirty_faces(light, &[], &[arrived]), 1 << 0, "+X only");
        let corner = body(Vec3::new(4.0, 0.0, 4.0));
        let m = dirty_faces(light, &[], &[corner]);
        assert_eq!(m, (1 << 0) | (1 << 4), "+X and +Z, got {m:06b}");
        // Leaving is as stale as arriving: the faces it WAS seen in need their static depth back.
        assert_eq!(dirty_faces(light, &[arrived], &[]), 1 << 0);
        // Moving across the edge dirties where it was and where it is.
        let moved = body(Vec3::new(0.0, 0.0, 4.0));
        assert_eq!(
            dirty_faces(light, &[arrived], &[moved]),
            (1 << 0) | (1 << 4)
        );
    }

    /// A caster that paired with its drawn record, but whose faces were not all redrawn this frame
    /// (another caster dirtied a different face), keeps the pose it was DRAWN at — otherwise a part
    /// creeping under the tolerance each frame would drift away from its shadow forever.
    #[test]
    fn a_caster_not_redrawn_keeps_its_drawn_pose() {
        let light = Vec3::ZERO;
        let a = body(Vec3::new(4.0, 0.0, 0.0));
        let crept = body(a.at + Vec3::new(0.0, 0.015, 0.0));
        let kept = settle_drawn(light, &[a], &[crept], 1 << 4);
        assert_eq!(
            kept,
            vec![a],
            "not redrawn: the record stays where the shadow is"
        );
        let kept = settle_drawn(light, &[a], &[crept], ALL_FACES);
        assert_eq!(kept, vec![crept], "redrawn: the record follows");
    }
}
