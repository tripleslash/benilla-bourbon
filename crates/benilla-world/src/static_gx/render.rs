//! The render-world half of the retained pass: extraction of the published set, per-region GPU
//! assembly against the shared texture pool, the pipelines, and the draw node before the main
//! opaque pass. Texture dims and formats exist only here (BLP images are `RENDER_WORLD`-only), so
//! a region draws once every member's `GpuImage` is resident.

use bevy::camera::primitives::Aabb;
use bevy::image::Image;
use bevy::mesh::VertexBufferLayout;
use bevy::pbr::{MeshPipeline, MeshPipelineViewLayoutKey, MeshViewBindGroup, ViewKeyCache};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d_array, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget};
use bevy::render::{Render, RenderSystems};
use bevy::shader::ShaderDefVal;
use std::ops::Range;

use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, CORE_3D_DEPTH_FORMAT};
use bevy::core_pipeline::{Core3d, Core3dSystems};

/// One baked item's draw facts, in bake order: the vertex word's low bits index its record.
#[derive(Clone)]
pub(crate) struct GxItemDraw {
    pub index_range: Range<u32>,
    pub texture: Option<AssetId<Image>>,
    pub cutout: bool,
    pub two_sided: bool,
    #[allow(dead_code)] // bake-side bookkeeping; the node draws by index range alone
    pub vertex_range: Range<u32>,
    /// The selection key (a WMO group or a prop referrer set); `None` on cell items.
    pub group: Option<u16>,
    /// The authored batch order (the coplanar-MOBA clip-z nudge; 0 on cell items).
    pub order: u16,
    /// The MOMT SIDN night-glow colour (gamma bytes; zero on cell items).
    pub sidn: [u8; 3],
    /// The interior prop's SH-probe slot, record column w bits 1..=13; 0 elsewhere.
    pub slot: u16,
}

/// One baked cell or region, published by the main-world flush.
#[derive(Clone)]
pub(crate) struct GxCellDraw {
    pub mesh: Handle<Mesh>,
    /// The recentring origin: shader world = vertex + origin.
    pub origin: Vec3,
    /// Mesh-local bound (recentred); world bound = origin + this.
    pub aabb: Aabb,
    pub draws: Vec<GxItemDraw>,
    /// The exile kill bitmap, bit i dropping item i; all-zero on regions.
    pub killed: Vec<u64>,
    /// Bumped on every bitmap change; the render side re-syncs the kill column on a new one.
    pub killed_rev: u32,
    /// Mesh-local bounds per selection key (group or referrer set) for the cull; empty on cells.
    pub groups: Vec<(u16, Aabb)>,
    /// A prop region's distinct referrer sets, indexed like `groups`.
    pub sets: Vec<std::sync::Arc<[u16]>>,
    /// fork: the region's meshlets, items in bake order, where the mesh-shader lane can run
    /// (`super::meshlet`; `None` for a region baked without it).
    #[cfg(not(target_arch = "wasm32"))]
    pub meshlets: Option<std::sync::Arc<super::meshlet::GxMeshlets>>,
}

/// Marks a view the retained pass draws into. Without it the node would run for EVERY Core3d
/// view, including the portrait-booth bakes, and paint world cells into a portrait with the
/// booth's view matrices.
///
/// **Two views carry it, and [`primary`](Self::primary) is which.** The world camera is the
/// [`WORLD`](Self::WORLD) one; the water's planar mirror (`liquid::reflect`) is the
/// [`MIRROR`](Self::MIRROR) one. The distinction exists for exactly one consumer —
/// [`prepare_static_gx`] specializes the four pipelines for ONE view's `(samples, format)` and
/// caches that key, so it has to always ask the same view rather than whichever the query
/// yields first.
///
/// **What the mirror gets, and what it does not.** The draw list ([`GxWorld::visible`] /
/// [`GxWorld::visible_wmos`]) is [`super::cull::cull_cells`]'s, and that walk is the world
/// camera's: its frustum, its farclip, its portal flood. Drawing that list under the mirror's
/// matrices is geometrically exact for every cell IN it — the node reads all of its view state
/// out of the `ViewQuery`, including the oblique near plane that clips the mirror below the
/// waterline — so the buildings and props the world camera can see reflect correctly. What it
/// cannot produce is a cell the MIRROR's frustum contains and the world camera's does not,
/// which is the near-vertical shot: pitch straight down at the water and the mirror wants what
/// is overhead and behind you, which the world frustum excludes.
///
/// Closing that needs a second cull list — a second CPU scene walk per frame, or a union
/// frustum that admits the mirror's cells into the world pass too, where they are invisible.
/// Both cost the frame thread this port spends its budget defending, for a half-resolution
/// buffer the water then bends a 20-yard ray through; the trade taken here buys the whole of
/// the common shot — a shore, its trees and its buildings across a lake — for no CPU at all.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub(crate) struct StaticGxView {
    /// Is this the view [`prepare_static_gx`] takes its pipeline key from? True on the world
    /// camera, false on the mirror.
    pub(crate) primary: bool,
}

impl StaticGxView {
    /// The world camera's — the pipeline key's view.
    pub(crate) const WORLD: Self = Self { primary: true };
    /// The water's planar mirror (`liquid::reflect`). Its `(samples, format)` are the world
    /// camera's by construction — that camera copies the `Msaa` at spawn and carries the same
    /// `Hdr` — which is what makes reusing the world's specialization correct rather than
    /// merely convenient.
    pub(crate) const MIRROR: Self = Self { primary: false };
}

/// Mark the world camera, again whenever it respawns.
///
/// The mirror's copy is not inserted here: it rides that camera's own spawn bundle, beside the
/// markers it deliberately does NOT carry (`liquid::reflect`'s header is the list).
fn mark_world_camera(
    mut commands: Commands,
    cam: Query<Entity, (With<crate::view::WorldCamera>, Without<StaticGxView>)>,
) {
    for e in &cam {
        commands.entity(e).insert(StaticGxView::WORLD);
    }
}

/// One entry of the doodad-phase draw list: ADT-doodad cells and WMO-prop regions are one phase
/// in the 1.12 order (the M2 scene, after the WMOs), so they sort near-first together.
#[derive(Clone, PartialEq)]
pub(crate) enum GxDoodadVis {
    Cell((i32, i32)),
    Prop(Entity, GxSel),
}

/// One region's verdicts this frame, indexed by group (WMO) or referrer set (props).
#[derive(Clone, Default, PartialEq)]
pub(crate) struct GxSel {
    /// Drawn this frame: PVS ∧ frustum ∧ farclip ∧ the exterior window gate.
    pub drawn: Vec<bool>,
    /// On the interior fog lane: the client's per-group `[0xca7f00]`, from the portal flood
    /// ([`crate::wmo_portal::GroupPvs::interior_fog`]).
    pub fog: Vec<bool>,
}

/// The published half the render world clones each frame. Regions sit behind `Arc`, so the clones
/// are refcount bumps and the kill scan, the one writer, copies a region on write.
#[derive(Clone, Default, Resource, ExtractResource)]
pub(crate) struct GxWorld {
    pub cells: HashMap<(i32, i32), std::sync::Arc<GxCellDraw>>,
    /// This frame's doodad-phase draw list, near-first across cells and prop regions.
    pub visible: Vec<GxDoodadVis>,
    pub wmos: HashMap<Entity, std::sync::Arc<GxCellDraw>>,
    /// The prop regions, keyed like `wmos` but apart, so a prop arrival never re-bakes a building.
    pub props: HashMap<Entity, std::sync::Arc<GxCellDraw>>,
    /// This frame's per-group admission per WMO region; the node draws only admitted runs.
    pub visible_wmos: Vec<(Entity, GxSel)>,
}

use super::pool::GxTexturePool;
use crate::modern::caps::HwCaps;

/// Record column w, bit 14: the item's interior fog lane, read by `static_gx.wgsl`. Bit 0 is the
/// kill bit and bits 1..=13 the probe slot.
const RECORD_FOG_BIT: u32 = 1 << 14;

/// One coalesced draw run of adjacent live items sharing bind-group slot, pipeline bucket and
/// selection key; a killed item is never in one.
#[derive(Clone)]
struct GxRun {
    /// Index into the region's `bind_groups`, not a pool class.
    slot: usize,
    cutout: bool,
    two_sided: bool,
    index_range: Range<u32>,
    /// The run's selection key; `None` for a cell run, which always draws.
    group: Option<u16>,
}

/// fork: where a region's origin, records and indices live, and what its draws bind.
enum GxRegionBinds {
    /// Its own buffers and one bind group per pool class its items touch: the per-region layout
    /// (the web, `WOW_GX_BINDLESS=0`).
    Own {
        /// (pool class, bind group), indexed by a run's `slot`.
        bind_groups: Vec<(u16, BindGroup)>,
        record_table: Buffer,
        /// Held alive for the bind groups that reference it.
        _cell_uniform: Buffer,
        /// The region's indices, copied from its mesh at assembly in draw order (see
        /// [`GxCellGpu::ranges`]). Every draw a region's runs issue — every pass — reads this
        /// buffer, not the mesh's.
        indices: Buffer,
    },
    /// Its blocks of the shared region table and index arena (`bindless`): one bind group for
    /// every region, and the region's runs offset by `index_base` into the arena.
    Shared(bindless::GxSlots),
}

/// A region's assembled GPU state, cached until its bake (mesh handle) changes.
struct GxCellGpu {
    mesh: AssetId<Mesh>,
    binds: GxRegionBinds,
    /// Per item, its run slot: its index into the `Own` bind groups, kept for kill-driven run
    /// rebuilds; 0 for every item when bindless, where the class rides in the record.
    item_slot: Vec<u16>,
    /// Per item, its index range in the region's indices (relative to `index_base` when bindless).
    /// The indices are copied from the mesh at assembly in draw order: items sorted by
    /// `(cutout, two_sided, slot, group)`, so what binds alike is contiguous and draws as one.
    ranges: Vec<Range<u32>>,
    /// The items in index order.
    order: Vec<u32>,
    /// The draw runs, split at every selection key: what the colour pass draws of a region.
    runs: Vec<GxRun>,
    /// The runs merged across selection keys: what a pass drawing a region whole issues.
    whole: Vec<GxRun>,
    /// CPU copy of the record table, which the kill and fog syncs rewrite and re-upload.
    records: Vec<[u32; 4]>,
    killed_applied: u32,
    /// The [`GxSel::fog`] the records were last written from; empty until the first sync.
    fog_applied: Vec<bool>,
    /// Items held back because their texture was not resident at assembly, as a kill bitmap: a
    /// region never waits whole on one texture, so its other items draw meanwhile.
    unready: Vec<u64>,
    /// The textures behind [`Self::unready`]; the region re-assembles once all are resident.
    waiting: Vec<AssetId<Image>>,
}

#[derive(Resource, Default)]
struct GxGpuCache {
    cells: HashMap<(i32, i32), GxCellGpu>,
    wmos: HashMap<Entity, GxCellGpu>,
    props: HashMap<Entity, GxCellGpu>,
    /// Bumped whenever an entry is added, dropped, or has its runs rebuilt, so a reader that diffs
    /// the cache (`modern_gi`) can skip a frame where nothing moved.
    rev: u64,
}

#[derive(Resource)]
struct GxPipelines {
    view_layout: BindGroupLayoutDescriptor,
    cell_layout: BindGroupLayoutDescriptor,
    light_layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    sampler_clamp: Sampler,
    /// Keyed `(cutout, two_sided)`, re-specialized if the world view's (samples, format) changes.
    pipelines: HashMap<(bool, bool), CachedRenderPipelineId>,
    /// The depth prepass's twins of [`Self::pipelines`], same keys — empty when the pass has no
    /// prepass (a classic frame, or `$WOW_STATIC_GX_PREPASS=0`), and then the colour pipelines
    /// write depth and discard themselves.
    prepass: HashMap<(bool, bool), CachedRenderPipelineId>,
    /// fork: modern lighting — the latched model joins the key, so a latch edge re-specializes.
    /// Without it the four pipelines would keep whichever lighting model the session opened on.
    ///
    /// The view's mesh-view layout key joins it too: Bevy 0.19 builds that layout per view from the
    /// features it has (fog, SSAO, prepass textures…), and ours must be the same layout to bind the
    /// view's group.
    specialized_for: Option<(u32, TextureFormat, bool, MeshPipelineViewLayoutKey)>,
    /// fork: this session draws bindless (`bindless::enabled`, fixed at startup): `cell_layout` is
    /// the shared layout and every static_gx pipeline takes [`Self::layout_defs`].
    bindless: bool,
    /// fork: F — the colour pipelines declare the ray-traced scene as group 3 and, in a frame the
    /// lane is live, trace the sun's and the lamps' visibility (`modern::rt_shadow`,
    /// [`rt_shadow_layout`]); `static_gx_pass` binds the scene or its stand-in. The prepass twins
    /// never do. Always false on wasm32, where nothing reads it.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    rt: bool,
}

/// fork: F — the ray-traced scene's layout for a modern colour pipeline where the session may
/// trace shadows (`modern::shadow::rt_shadows_capable`, fixed for the session), bound as group 3;
/// `None` everywhere else. The pipeline is the same with the ray tracing on or off: the group bound
/// (`RtScene::lane_bind`) picks the path per frame.
fn rt_shadow_layout(modern: bool) -> Option<BindGroupLayoutDescriptor> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        (modern && crate::modern::shadow::rt_shadows_capable()).then(rt_scene::layout)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = modern;
        None
    }
}

/// fork: F — group 3 of [`rt_shadow_layout`]'s pipelines.
const RT_SHADOW_GROUP: u32 = 3;

impl GxPipelines {
    /// fork: the shader defs of group 1's layout: `GX_BINDLESS` when bindless, else none. Every
    /// pipeline that binds `cell_layout` — this pass's, the shadow casters', the voxelizer's —
    /// starts its defs here.
    fn layout_defs(&self) -> Vec<ShaderDefVal> {
        if self.bindless {
            vec![ShaderDefVal::from("GX_BINDLESS")]
        } else {
            Vec::new()
        }
    }
}

fn init_pipelines(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    render_device: Res<RenderDevice>,
    caps: Option<Res<HwCaps>>,
) {
    // Use Bevy's real mesh-view layout for the retained pass. Besides the view and light records,
    // this carries the directional shadow textures and comparison samplers produced for the
    // world camera. The old two-binding layout could render the city, but it had no legal way for
    // `static_gx.wgsl` to receive the shadow map.
    let view_layout = mesh_pipeline
        .get_view_layout(MeshPipelineViewLayoutKey::empty())
        .main_layout
        .clone();
    // The pass's own extra (group 2): the shared light storage.
    let light_layout = BindGroupLayoutDescriptor::new(
        "static_gx_light_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                storage_buffer_read_only_sized(false, None),
                // fork: modern lighting (P2) — the modern sun's shadow array, compare sampler, cascades.
                BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                sampler(SamplerBindingType::Comparison),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );
    // fork: bindless — one layout for every region where the device has binding arrays and a
    // counted multi-draw (`bindless`), else the per-region one.
    let bindless = bindless::enabled(caps.as_deref());
    if bindless {
        commands.insert_resource(bindless::GxBindless::new(caps.as_deref()));
        info!("static_gx: bindless layout (WOW_GX_BINDLESS=0 draws per region)");
    } else {
        info!("static_gx: per-region layout");
    }
    let per_region = BindGroupLayoutDescriptor::new(
        "static_gx_cell_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                // origin (xyz) + pad
                uniform_buffer_sized(false, Some(std::num::NonZero::new(16).unwrap())),
                // the per-item record table
                storage_buffer_read_only_sized(false, None),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let cell_layout = if bindless {
        bindless::layout()
    } else {
        per_region
    };
    // Repeat and clamp samplers, picked by the vertex word's wrap bits since a shared array has no
    // per-layer address mode. Mip filter and anisotropy come from `benilla_assets::tex_filter`,
    // the policy the reference applies to every texture from `[0x835250]`/`[0x835254]`. A
    // mixed-wrap batch takes the repeat sampler and the shader clamps its clamped axis a half
    // texel in; the reference clamps that axis in the sampler.
    let filter = benilla_assets::tex_filter();
    let make = |label: &'static str, mode: AddressMode| {
        render_device.create_sampler(&SamplerDescriptor {
            label: Some(label),
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: filter.gpu_mipmap_filter(),
            address_mode_u: mode,
            address_mode_v: mode,
            anisotropy_clamp: filter.anisotropy_clamp(),
            ..Default::default()
        })
    };
    commands.insert_resource(GxPipelines {
        view_layout,
        cell_layout,
        light_layout,
        sampler: make("static_gx_repeat", AddressMode::Repeat),
        sampler_clamp: make("static_gx_clamp", AddressMode::ClampToEdge),
        pipelines: HashMap::default(),
        prepass: HashMap::default(),
        specialized_for: None,
        bindless,
        rt: false,
    });
}

/// fork: does a modern frame lay the static world's depth before shading it (`static_gx.wgsl`'s
/// prepass)? On by default; `$WOW_STATIC_GX_PREPASS=0` (browser `?static_gx_prepass=0`) is the A/B
/// lever. Classic frames never take it: their fragment is cheap, and their pixels stay the
/// reference's byte for byte.
fn prepass_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_STATIC_GX_PREPASS").as_deref() != Some("0"))
}

/// The world view's pipeline-key inputs.
type GxViewKey = (
    &'static ExtractedView,
    &'static Msaa,
    &'static ViewTarget,
    &'static StaticGxView,
);

/// The bake's interleaved vertex layout, in the attribute-id order Bevy interleaves by: position,
/// normal, uv, colour, then the word and anchor (988_101, 988_102). Keep in sync with `bake_cell`
/// and `static_gx.wgsl`.
fn vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        array_stride: 64,
        step_mode: VertexStepMode::Vertex,
        attributes: vec![
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 24,
                shader_location: 2,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 32,
                shader_location: 5,
            },
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: 48,
                shader_location: 3,
            },
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 52,
                shader_location: 4,
            },
        ],
    }
}

/// Specialize the four pipelines for the world view, and assemble visible regions' GPU state.
fn prepare_static_gx(
    gx: Res<GxWorld>,
    mut cache: ResMut<GxGpuCache>,
    mut pool: ResMut<GxTexturePool>,
    mut pipes: ResMut<GxPipelines>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
    view_keys: Res<ViewKeyCache>,
    images: Res<RenderAssets<GpuImage>>,
    allocator: Res<MeshAllocator>,
    views: Query<GxViewKey>,
    mut shared: Option<ResMut<bindless::GxBindless>>,
) {
    let _t = super::gx_perf_guard(3);
    // The world view's pipeline key (the marker keeps booth views out of it) — and the PRIMARY
    // one specifically: the water's mirror carries the marker too (`StaticGxView::MIRROR`), and
    // a key cached from whichever view the query happened to yield first would flip between
    // them on any frame their `(samples, format)` disagreed, clearing and re-queuing all four
    // pipelines every frame. They cannot disagree today (the mirror copies both at spawn);
    // asking for the one view by name is what keeps that from being load-bearing.
    let Some((view, msaa, _, _)) = views.iter().find(|(_, _, _, marker)| marker.primary) else {
        return;
    };
    let format = view.target_format;
    // fork: modern lighting — the latched model is read here, in the render world, from the same
    // atomic the model lane keys on, so the two lanes cannot disagree about which frame this is.
    let modern = benilla_assets::materials::MODERN_MODEL_PIPELINES
        .load(std::sync::atomic::Ordering::Relaxed);
    // The key Bevy built this view's mesh-view bind group from; absent until the view's first
    // specialization pass, and nothing can draw before it.
    let Some(view_key) = view_keys.get(&view.retained_view_entity) else {
        return;
    };
    let layout_key = MeshPipelineViewLayoutKey::from(*view_key);
    let key = (msaa.samples(), format, modern, layout_key);
    if pipes.specialized_for != Some(key) {
        pipes.view_layout = mesh_pipeline
            .get_view_layout(layout_key)
            .main_layout
            .clone();
        let shader: Handle<Shader> =
            asset_server.load("embedded://benilla_world/shaders/static_gx.wgsl");
        pipes.pipelines.clear();
        pipes.prepass.clear();
        let prepassed = modern && prepass_enabled();
        // fork: F — the colour pipelines' ray-traced scene, where the session may trace shadows.
        let rt_layout = rt_shadow_layout(modern);
        pipes.rt = rt_layout.is_some();
        for cutout in [false, true] {
            for two_sided in [false, true] {
                let mut defs = pipes.layout_defs();
                // The world camera uses Bevy's default Gaussian shadow filtering. The retained
                // pipeline is custom, so it must opt into the same shader branch explicitly;
                // otherwise `shadows::fetch_directional_shadow` falls back to a constant 1.0.
                defs.push(ShaderDefVal::from("SHADOW_FILTER_METHOD_GAUSSIAN"));
                if cutout {
                    defs.push(ShaderDefVal::from("GX_CUTOUT"));
                }
                // fork: modern lighting (P3) — foliage transmission rides the two-sided key.
                if two_sided {
                    defs.push(ShaderDefVal::from("GX_TWO_SIDED"));
                }
                // fork: modern lighting — compile ONE lighting model. Both used to ride every
                // retained pipeline behind a storage read, which is the shape that turned
                // Chrome/Metal's cutout holes black on the model lane (bisected 2026-09-16: not any
                // one modern block, but the fragment's size). This lane has no booth or glue view —
                // it draws the world camera and the water mirror, and both read the same modern
                // light buffer — so the def can decide the path outright, as `MODERN_FRAME`.
                if modern {
                    defs.push(ShaderDefVal::from("WOW_MODERN"));
                    if benilla_assets::materials::modern_vertex_walks() {
                        defs.push(ShaderDefVal::from("MODERN_VERTEX_WALKS"));
                    }
                }
                if prepassed {
                    defs.push(ShaderDefVal::from("GX_PREPASSED"));
                    // The depth-only twin: the same cull mode as the colour pipeline, or a face
                    // only one of them drew would leave depth with no colour on it.
                    let pre = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                        label: Some(
                            format!(
                                "static_gx_prepass c{} t{}",
                                u8::from(cutout),
                                u8::from(two_sided)
                            )
                            .into(),
                        ),
                        layout: vec![
                            pipes.view_layout.clone(),
                            pipes.cell_layout.clone(),
                            pipes.light_layout.clone(),
                        ],
                        vertex: VertexState {
                            shader: shader.clone(),
                            shader_defs: defs.clone(),
                            entry_point: Some("prepass_vertex".into()),
                            buffers: vec![vertex_layout()],
                        },
                        // Every variant keeps a fragment stage: the farclip wall is per pixel.
                        fragment: Some(FragmentState {
                            shader: shader.clone(),
                            shader_defs: defs.clone(),
                            entry_point: Some("prepass_fragment".into()),
                            targets: vec![],
                        }),
                        primitive: PrimitiveState {
                            cull_mode: (!two_sided).then_some(Face::Back),
                            ..Default::default()
                        },
                        depth_stencil: Some(DepthStencilState {
                            format: CORE_3D_DEPTH_FORMAT,
                            depth_write_enabled: Some(true),
                            depth_compare: Some(CompareFunction::GreaterEqual),
                            stencil: StencilState::default(),
                            bias: DepthBiasState::default(),
                        }),
                        multisample: MultisampleState {
                            count: msaa.samples(),
                            ..Default::default()
                        },
                        ..default()
                    });
                    pipes.prepass.insert((cutout, two_sided), pre);
                }
                // fork: F — the colour pipeline's fragment stage alone traces (the scene's layout
                // is visible to the fragment and compute stages only).
                let mut layout = vec![
                    pipes.view_layout.clone(),
                    pipes.cell_layout.clone(),
                    pipes.light_layout.clone(),
                ];
                let vertex_defs = defs.clone();
                if let Some(rt) = &rt_layout {
                    layout.push(rt.clone());
                    defs.push(ShaderDefVal::from("RT_SHADOWS"));
                    defs.push(ShaderDefVal::UInt("RT_GROUP".into(), RT_SHADOW_GROUP));
                }
                let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some(
                        format!("static_gx c{} t{}", u8::from(cutout), u8::from(two_sided)).into(),
                    ),
                    layout,
                    vertex: VertexState {
                        shader: shader.clone(),
                        shader_defs: vertex_defs,
                        entry_point: Some("vertex".into()),
                        buffers: vec![vertex_layout()],
                    },
                    fragment: Some(FragmentState {
                        shader: shader.clone(),
                        shader_defs: defs,
                        entry_point: Some("fragment".into()),
                        targets: vec![Some(ColorTargetState {
                            format,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    primitive: PrimitiveState {
                        cull_mode: (!two_sided).then_some(Face::Back),
                        ..Default::default()
                    },
                    // Behind the prepass: `Equal` against the depth it laid, no write — only the
                    // surface it kept shades, and a hidden fragment is rejected before it runs.
                    depth_stencil: Some(DepthStencilState {
                        format: CORE_3D_DEPTH_FORMAT,
                        depth_write_enabled: Some(!prepassed),
                        depth_compare: Some(if prepassed {
                            CompareFunction::Equal
                        } else {
                            CompareFunction::GreaterEqual
                        }),
                        stencil: StencilState::default(),
                        bias: DepthBiasState::default(),
                    }),
                    multisample: MultisampleState {
                        count: msaa.samples(),
                        ..Default::default()
                    },
                    ..default()
                });
                pipes.pipelines.insert((cutout, two_sided), id);
            }
        }
        pipes.specialized_for = Some(key);
    }

    // Every published map empty is a map change (`StaticGx::clear`): reset the pool and cache.
    if gx.cells.is_empty() && gx.wmos.is_empty() && gx.props.is_empty() {
        if !pool.is_empty() {
            *pool = GxTexturePool::default();
        }
        if !(cache.cells.is_empty() && cache.wmos.is_empty() && cache.props.is_empty()) {
            cache.rev += 1;
        }
        cache.cells.clear();
        cache.wmos.clear();
        cache.props.clear();
        if let Some(shared) = shared.as_deref_mut() {
            shared.reset();
        }
        return;
    }

    // Drop cache entries whose region vanished or re-baked.
    let held = cache.cells.len() + cache.wmos.len() + cache.props.len();
    cache.cells.retain(|c, gpu| {
        let keep = gx.cells.get(c).is_some_and(|d| d.mesh.id() == gpu.mesh);
        if !keep {
            evict(shared.as_deref_mut(), gpu);
        }
        keep
    });
    cache.wmos.retain(|e, gpu| {
        let keep = gx.wmos.get(e).is_some_and(|d| d.mesh.id() == gpu.mesh);
        if !keep {
            evict(shared.as_deref_mut(), gpu);
        }
        keep
    });
    cache.props.retain(|e, gpu| {
        let keep = gx.props.get(e).is_some_and(|d| d.mesh.id() == gpu.mesh);
        if !keep {
            evict(shared.as_deref_mut(), gpu);
        }
        keep
    });
    if cache.cells.len() + cache.wmos.len() + cache.props.len() != held {
        cache.rev += 1;
    }

    // The assemblies' index copies, submitted below, before any pass draws this frame.
    let mut copies = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("static_gx_index_copies"),
    });
    for vis in &gx.visible {
        match vis {
            GxDoodadVis::Cell(cell) => {
                if cache
                    .cells
                    .get(cell)
                    .is_some_and(|g| !g.wants_reassembly(&images))
                {
                    continue;
                }
                let Some(draw) = gx.cells.get(cell) else {
                    continue;
                };
                if let Some(gpu) = assemble_region(
                    draw,
                    &mut pool,
                    shared.as_deref_mut(),
                    &pipes,
                    &pipeline_cache,
                    &render_device,
                    &render_queue,
                    &images,
                    &allocator,
                    &mut copies,
                ) {
                    if let Some(old) = cache.cells.insert(*cell, gpu) {
                        evict(shared.as_deref_mut(), &old);
                    }
                    cache.rev += 1;
                }
            }
            GxDoodadVis::Prop(entity, _) => {
                if cache
                    .props
                    .get(entity)
                    .is_some_and(|g| !g.wants_reassembly(&images))
                {
                    continue;
                }
                let Some(draw) = gx.props.get(entity) else {
                    continue;
                };
                if let Some(gpu) = assemble_region(
                    draw,
                    &mut pool,
                    shared.as_deref_mut(),
                    &pipes,
                    &pipeline_cache,
                    &render_device,
                    &render_queue,
                    &images,
                    &allocator,
                    &mut copies,
                ) {
                    if let Some(old) = cache.props.insert(*entity, gpu) {
                        evict(shared.as_deref_mut(), &old);
                    }
                    cache.rev += 1;
                }
            }
        }
    }
    for (entity, _) in &gx.visible_wmos {
        if cache
            .wmos
            .get(entity)
            .is_some_and(|g| !g.wants_reassembly(&images))
        {
            continue;
        }
        let Some(draw) = gx.wmos.get(entity) else {
            continue;
        };
        if let Some(gpu) = assemble_region(
            draw,
            &mut pool,
            shared.as_deref_mut(),
            &pipes,
            &pipeline_cache,
            &render_device,
            &render_queue,
            &images,
            &allocator,
            &mut copies,
        ) {
            if let Some(old) = cache.wmos.insert(*entity, gpu) {
                evict(shared.as_deref_mut(), &old);
            }
            cache.rev += 1;
        }
    }
    // fork: rt_scene — with the ray tracing enabled, the regions the camera has not drawn are
    // assembled too, a few a frame, so its BLASes hold every streamed region.
    #[cfg(not(target_arch = "wasm32"))]
    if modern && rt_scene::enabled() {
        rt_scene::assemble_unseen(
            &gx,
            &mut cache,
            &mut pool,
            shared.as_deref_mut(),
            &pipes,
            &pipeline_cache,
            &render_device,
            &render_queue,
            &images,
            &allocator,
            &mut copies,
        );
    }
    pool.drain_pending(&render_device, &render_queue);
    // fork: bindless — the one bind group, rebuilt when the table grew or the pool opened a class.
    if let Some(shared) = shared.as_deref_mut() {
        shared.refresh_bind(
            &mut pool,
            &pipeline_cache.get_bind_group_layout(&pipes.cell_layout),
            &pipes.sampler,
            &pipes.sampler_clamp,
            &render_device,
            &render_queue,
        );
    }
    render_queue.submit([copies.finish()]);
    let shared = shared.as_deref();

    // The kill-bit sync: on a new bitmap revision, rewrite the kill column, re-upload and rebuild
    // the runs. A cell that changed out of view syncs on re-entry.
    let mut killed_synced = false;
    for vis in &gx.visible {
        let (Some(gpu), Some(draw)) = (match vis {
            GxDoodadVis::Cell(cell) => (cache.cells.get_mut(cell), gx.cells.get(cell)),
            GxDoodadVis::Prop(entity, _) => (cache.props.get_mut(entity), gx.props.get(entity)),
        }) else {
            continue;
        };
        if gpu.killed_applied == draw.killed_rev {
            continue;
        }
        killed_synced = true;
        let hidden = hidden_bits(&draw.killed, &gpu.unready);
        for (i, rec) in gpu.records.iter_mut().enumerate() {
            // Column w also carries the probe slot and fog bit: rewrite bit 0 alone.
            rec[3] = (rec[3] & !1) | kill_bit(&hidden, i);
        }
        gpu.write_records(&render_queue, shared);
        gpu.runs = build_runs(
            &draw.draws,
            &gpu.order,
            &gpu.ranges,
            &gpu.item_slot,
            &hidden,
        );
        gpu.whole = whole_runs(&gpu.runs);
        gpu.killed_applied = draw.killed_rev;
    }
    if killed_synced {
        cache.rev += 1;
    }

    // The interior-fog sync: the client's per-group `[0xca7f00]` picks a WMO group's and its
    // props' fog triple per frame; rewritten only when the verdict moves, runs untouched.
    let sync_fog = |gpu: &mut GxCellGpu, draw: &GxCellDraw, sel: &GxSel| {
        if gpu.fog_applied == sel.fog {
            return;
        }
        for (rec, item) in gpu.records.iter_mut().zip(&draw.draws) {
            let on = item
                .group
                .is_some_and(|g| sel.fog.get(usize::from(g)).copied().unwrap_or(false));
            rec[3] = (rec[3] & !RECORD_FOG_BIT) | (u32::from(on) * RECORD_FOG_BIT);
        }
        gpu.write_records(&render_queue, shared);
        gpu.fog_applied.clone_from(&sel.fog);
    };
    for (entity, sel) in &gx.visible_wmos {
        if let (Some(gpu), Some(draw)) = (cache.wmos.get_mut(entity), gx.wmos.get(entity)) {
            sync_fog(gpu, draw, sel);
        }
    }
    for vis in &gx.visible {
        let GxDoodadVis::Prop(entity, sel) = vis else {
            continue; // cell items are never on the interior lane
        };
        if let (Some(gpu), Some(draw)) = (cache.props.get_mut(entity), gx.props.get(entity)) {
            sync_fog(gpu, draw, sel);
        }
    }
}

/// Assemble one region's GPU state against the shared pool. An item whose texture is not resident
/// yet is held back ([`GxCellGpu::unready`]) and the rest draw, as an entity-path part draws the
/// moment its own material is ready: one slow or missing texture must not blank a whole cell or
/// building while its colliders and picks stand.
/// A region's GPU state, or `None` while its mesh has no index slice yet. The index copy is
/// recorded into `copies`, which the caller submits before any pass draws.
///
/// fork: with `shared` (bindless) the region takes a block of the shared table and of the index
/// arena instead of buffers and bind groups of its own.
#[allow(clippy::too_many_arguments)]
fn assemble_region(
    draw: &GxCellDraw,
    pool: &mut GxTexturePool,
    mut shared: Option<&mut bindless::GxBindless>,
    pipes: &GxPipelines,
    pipeline_cache: &PipelineCache,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    images: &RenderAssets<GpuImage>,
    allocator: &MeshAllocator,
    copies: &mut CommandEncoder,
) -> Option<GxCellGpu> {
    let islice = allocator.mesh_index_slice(&draw.mesh.id())?;
    // Each item's pool slot, deduped by texture id; untextured items take the white class, and so
    // does a held-back item, whose slot is never sampled.
    let mut white: Option<u16> = None;
    let mut item_class_layer: Vec<(u16, u16)> = Vec::with_capacity(draw.draws.len());
    let mut unready = vec![0u64; draw.draws.len().div_ceil(64)];
    let mut waiting: Vec<AssetId<Image>> = Vec::new();
    for (i, item) in draw.draws.iter().enumerate() {
        let resident = item.texture.map(|tex| (tex, images.get(tex)));
        let assigned = match resident {
            Some((tex, Some(gpu))) => {
                let slot = pool.assign(tex, gpu, render_device);
                // fork: bindless — a class past the binding array is held back for good.
                if shared.as_deref_mut().is_some_and(|s| !s.class_fits(slot.0)) {
                    unready[i / 64] |= 1u64 << (i % 64);
                    None
                } else {
                    Some(slot)
                }
            }
            Some((tex, None)) => {
                unready[i / 64] |= 1u64 << (i % 64);
                if !waiting.contains(&tex) {
                    waiting.push(tex);
                }
                None
            }
            None => None,
        };
        item_class_layer.push(assigned.unwrap_or_else(|| {
            (
                *white.get_or_insert_with(|| pool.white(render_device, render_queue)),
                0,
            )
        }));
    }
    let hidden = hidden_bits(&draw.killed, &unready);
    // The per-item record table, indexed by the word's low bits: [layer, batch-order nudge, SIDN,
    // flags]. Flags: bit 0 the kill bit, re-synced by revision (hence COPY_DST); bits 1..=13 the
    // probe slot, 13 bits being `MAX_PROP_PROBES` exactly. fork: bindless — the pool class rides
    // in the layer word's high half (`bindless`).
    let pack_class = shared.is_some();
    let records: Vec<[u32; 4]> = draw
        .draws
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let (class, layer) = item_class_layer[i];
            [
                u32::from(layer)
                    | if pack_class {
                        u32::from(class) << 16
                    } else {
                        0
                    },
                u32::from(item.order),
                u32::from(item.sidn[0])
                    | (u32::from(item.sidn[1]) << 8)
                    | (u32::from(item.sidn[2]) << 16),
                kill_bit(&hidden, i) | (u32::from(item.slot) << 1),
            ]
        })
        .collect();
    // The per-region layout's buffers and bind groups; bindless, every item is slot 0 (one bind
    // group for every region).
    let (item_slot, own) = if shared.is_some() {
        (vec![0u16; draw.draws.len()], None)
    } else {
        let (item_slot, own) = own_binds(
            draw,
            &records,
            &item_class_layer,
            pool,
            pipes,
            pipeline_cache,
            render_device,
        );
        (item_slot, Some(own))
    };
    // Draw order: what binds alike together, then by selection key, then bake order. Opaque and
    // cutout items only, depth-tested, so the order inside a region is free.
    let mut order: Vec<u32> = (0..draw.draws.len() as u32).collect();
    order.sort_by_key(|&i| {
        let item = &draw.draws[i as usize];
        (
            item.cutout,
            item.two_sided,
            item_slot[i as usize],
            item.group,
            i,
        )
    });
    let mut ranges: Vec<Range<u32>> = vec![0..0; draw.draws.len()];
    let mut at = 0u32;
    for &i in &order {
        let len = draw.draws[i as usize].index_range.len() as u32;
        ranges[i as usize] = at..at + len;
        at += len;
    }
    // Where the indices go: the region's own buffer, or its block of the arena.
    let (binds, indices, index_base) = match (shared, own) {
        (Some(shared), _) => {
            let slots = shared.place(
                draw.origin,
                &draw.aabb,
                &draw.groups,
                &records,
                at,
                render_device,
                render_queue,
                copies,
            );
            let arena = shared.index_buffer().expect("`place` reserves the arena");
            (
                GxRegionBinds::Shared(slots),
                arena.clone(),
                slots.index_base,
            )
        }
        (None, Some((bind_groups, record_table, cell_uniform))) => {
            let indices = render_device.create_buffer(&BufferDescriptor {
                label: Some("static_gx_indices"),
                size: u64::from(at.max(1)) * 4,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let binds = GxRegionBinds::Own {
                bind_groups,
                record_table,
                _cell_uniform: cell_uniform,
                indices: indices.clone(),
            };
            (binds, indices, 0)
        }
        (None, None) => unreachable!("the per-region layout builds its own binds"),
    };
    // One copy per stretch of items that stays contiguous in both buffers (`u32` indices, so
    // every range is copy-aligned).
    let mut stretch: Option<(u32, u32, u32)> = None; // (source start, destination start, length)
    let flush = |stretch: Option<(u32, u32, u32)>, copies: &mut CommandEncoder| {
        if let Some((src, dst, len)) = stretch.filter(|s| s.2 > 0) {
            copies.copy_buffer_to_buffer(
                islice.buffer,
                u64::from(islice.range.start + src) * 4,
                &indices,
                u64::from(index_base + dst) * 4,
                u64::from(len) * 4,
            );
        }
    };
    for &i in &order {
        let src = draw.draws[i as usize].index_range.clone();
        let dst = ranges[i as usize].start;
        stretch = match stretch {
            Some((s, d, len)) if s + len == src.start && d + len == dst => {
                Some((s, d, len + src.len() as u32))
            }
            other => {
                flush(other, copies);
                Some((src.start, dst, src.len() as u32))
            }
        };
    }
    flush(stretch, copies);
    let runs = build_runs(&draw.draws, &order, &ranges, &item_slot, &hidden);
    let whole = whole_runs(&runs);
    Some(GxCellGpu {
        mesh: draw.mesh.id(),
        binds,
        item_slot,
        ranges,
        order,
        runs,
        whole,
        records,
        killed_applied: draw.killed_rev,
        // Empty, so the first fog sync always writes.
        fog_applied: Vec::new(),
        unready,
        waiting,
    })
}

/// The per-region layout's record table, origin uniform and one bind group per distinct pool
/// class the region touches, with each item's index into those bind groups.
#[allow(clippy::type_complexity)]
fn own_binds(
    draw: &GxCellDraw,
    records: &[[u32; 4]],
    item_class_layer: &[(u16, u16)],
    pool: &GxTexturePool,
    pipes: &GxPipelines,
    pipeline_cache: &PipelineCache,
    render_device: &RenderDevice,
) -> (Vec<u16>, (Vec<(u16, BindGroup)>, Buffer, Buffer)) {
    let record_table = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("static_gx_records"),
        contents: bytemuck::cast_slice(records),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let cell_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("static_gx_cell"),
        contents: bytemuck::cast_slice(&[draw.origin.x, draw.origin.y, draw.origin.z, 0.0f32]),
        usage: BufferUsages::UNIFORM,
    });
    // One bind group per distinct pool class the region touches.
    let cell_layout = pipeline_cache.get_bind_group_layout(&pipes.cell_layout);
    let mut bind_groups: Vec<(u16, BindGroup)> = Vec::new();
    let mut item_slot: Vec<u16> = Vec::with_capacity(item_class_layer.len());
    for &(class, _) in item_class_layer {
        let slot = match bind_groups.iter().position(|(c, _)| *c == class) {
            Some(s) => s,
            None => {
                let bg = render_device.create_bind_group(
                    "static_gx_cell",
                    &cell_layout,
                    &BindGroupEntries::sequential((
                        cell_uniform.as_entire_binding(),
                        record_table.as_entire_binding(),
                        pool.view(class),
                        &pipes.sampler,
                        &pipes.sampler_clamp,
                    )),
                );
                bind_groups.push((class, bg));
                bind_groups.len() - 1
            }
        };
        item_slot.push(u16::try_from(slot).expect("gx region under u16 slots"));
    }
    (item_slot, (bind_groups, record_table, cell_uniform))
}

/// fork: bindless — free a region's blocks as it leaves the cache.
fn evict(shared: Option<&mut bindless::GxBindless>, gpu: &GxCellGpu) {
    if let (Some(shared), GxRegionBinds::Shared(slots)) = (shared, &gpu.binds) {
        shared.release(*slots);
    }
}

impl GxCellGpu {
    /// Upload [`Self::records`] where this region's record table lives.
    fn write_records(&self, queue: &RenderQueue, shared: Option<&bindless::GxBindless>) {
        match &self.binds {
            GxRegionBinds::Own { record_table, .. } => {
                queue.write_buffer(record_table, 0, bytemuck::cast_slice(&self.records));
            }
            GxRegionBinds::Shared(slots) => {
                if let Some(shared) = shared {
                    shared.write_records(queue, *slots, &self.records);
                }
            }
        }
    }

    /// Whether every texture this region held items back for is resident now, so a re-assembly
    /// can draw them; a complete region never re-assembles.
    fn wants_reassembly(&self, images: &RenderAssets<GpuImage>) -> bool {
        !self.waiting.is_empty() && self.waiting.iter().all(|&tex| images.get(tex).is_some())
    }
}

/// The items a region does not draw: the fader exiles' kill bits and the held-back ones.
fn hidden_bits(killed: &[u64], unready: &[u64]) -> Vec<u64> {
    let n = killed.len().max(unready.len());
    (0..n)
        .map(|w| killed.get(w).copied().unwrap_or(0) | unready.get(w).copied().unwrap_or(0))
        .collect()
}

/// Coalesce live items adjacent in draw `order` and sharing (slot, bucket, group) into draw runs
/// over their `ranges` in the region's index buffer; a killed item is skipped, so a fully gone
/// cell submits nothing.
fn build_runs(
    draws: &[GxItemDraw],
    order: &[u32],
    ranges: &[Range<u32>],
    item_slot: &[u16],
    killed: &[u64],
) -> Vec<GxRun> {
    let mut runs: Vec<GxRun> = Vec::new();
    for &i in order {
        let i = i as usize;
        if kill_bit(killed, i) != 0 {
            continue;
        }
        let item = &draws[i];
        let slot = usize::from(item_slot[i]);
        match runs.last_mut() {
            Some(r)
                if r.slot == slot
                    && r.cutout == item.cutout
                    && r.two_sided == item.two_sided
                    && r.group == item.group
                    && r.index_range.end == ranges[i].start =>
            {
                r.index_range.end = ranges[i].end;
            }
            _ => runs.push(GxRun {
                slot,
                cutout: item.cutout,
                two_sided: item.two_sided,
                index_range: ranges[i].clone(),
                group: item.group,
            }),
        }
    }
    runs
}

/// The runs merged across selection keys where they stay contiguous: a region drawn whole.
fn whole_runs(runs: &[GxRun]) -> Vec<GxRun> {
    let mut whole: Vec<GxRun> = Vec::new();
    for run in runs {
        match whole.last_mut() {
            Some(w)
                if (w.slot, w.cutout, w.two_sided) == (run.slot, run.cutout, run.two_sided)
                    && w.index_range.end == run.index_range.start =>
            {
                w.index_range.end = run.index_range.end;
            }
            _ => whole.push(GxRun {
                group: None,
                ..run.clone()
            }),
        }
    }
    whole
}

/// Record-table column 3: item `i`'s exile kill bit from the published bitmap.
fn kill_bit(killed: &[u64], i: usize) -> u32 {
    u32::from(
        killed
            .get(i / 64)
            .is_some_and(|w| w & (1u64 << (i % 64)) != 0),
    )
}

/// The retained pass's group 2: the `wow_shared_light` storage every material binds. Group 0 is
/// bevy's standard mesh-view bind group (view matrices, directional lights, shadow textures).
#[derive(Resource)]
struct GxLightBind(BindGroup);

fn prepare_view_bind(
    mut commands: Commands,
    pipes: Res<GxPipelines>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    light: Option<Res<crate::lighting::SharedLightBuffer>>,
    images: Res<RenderAssets<bevy::render::texture::GpuImage>>,
    storage: Res<RenderAssets<bevy::render::storage::GpuShaderBuffer>>,
) {
    let Some(light) = light else {
        return;
    };
    // fork: modern lighting (P2) — the constant-handle shadow assets exist from Startup on.
    let (Some(shadow), Some(cascades)) = (
        images.get(benilla_assets::materials::MODERN_SHADOW_MAP.id()),
        storage.get(benilla_assets::materials::MODERN_SHADOW_CASCADES.id()),
    ) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipes.light_layout);
    commands.insert_resource(GxLightBind(render_device.create_bind_group(
        "static_gx_light",
        &layout,
        &BindGroupEntries::sequential((
            light.0.as_entire_binding(),
            &shadow.texture_view,
            &shadow.sampler,
            cascades.buffer.as_entire_binding(),
        )),
    )));
}

/// The retained static world: its depth prepass (modern) and colour pass, before the opaque pass.
pub(crate) fn static_gx_pass(
    world: &World,
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthTexture,
        &MeshViewBindGroup,
        // The world camera and the water's planar mirror — a booth bake must never receive
        // world cells (see the marker).
        &StaticGxView,
        // fork: GPU cull — the view's matrices and its pyramid's key (`gpu_cull`).
        &ExtractedView,
    )>,
    mut render_context: RenderContext,
) {
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    let (target, depth, view_bind, _marker, extracted) = view.into_inner();
    {
        let _t = super::gx_perf_guard(4);
        let gx = world.resource::<GxWorld>();
        if gx.visible.is_empty() && gx.visible_wmos.is_empty() {
            return;
        }
        let cache = world.resource::<GxGpuCache>();
        let pipes = world.resource::<GxPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(light_bind) = world.get_resource::<GxLightBind>() else {
            return;
        };
        let allocator = world.resource::<MeshAllocator>();
        // All four pipelines or none: a region drawing only its opaque half would flash its
        // cutout content off for a frame.
        let Some(ready) = pipeline_table(pipeline_cache, &pipes.pipelines) else {
            return;
        };
        // …and the prepass with them: a colour pipeline testing `Equal` against depth nobody laid
        // would draw nothing at all.
        let Some(ready_pre) = pipeline_table(pipeline_cache, &pipes.prepass) else {
            return;
        };
        // fork: one draw list for both passes — gathered and state-sorted once, so the prepass
        // and the colour pass cannot disagree about what was drawn (`batch`).
        // WMO regions are phase 0 and the doodad phase 1, the 1.12 client's WMO-then-doodad drain
        // order; inside a phase the draws go by state, not near-first. Cells draw whole; a region
        // draws only its admitted runs.
        let mut batch = GxBatch::new(world);
        for (entity, sel) in &gx.visible_wmos {
            if let (Some(gpu), Some(draw)) = (cache.wmos.get(entity), gx.wmos.get(entity)) {
                if let Some(r) = batch.region(gpu, draw, allocator) {
                    batch.push_admitted(r, 0, sel, pipe_class);
                }
            }
        }
        for vis in &gx.visible {
            match vis {
                GxDoodadVis::Cell(cell) => {
                    if let (Some(gpu), Some(draw)) = (cache.cells.get(cell), gx.cells.get(cell)) {
                        if let Some(r) = batch.region(gpu, draw, allocator) {
                            batch.push_whole(r, 1, pipe_class);
                        }
                    }
                }
                GxDoodadVis::Prop(entity, sel) => {
                    if let (Some(gpu), Some(draw)) = (cache.props.get(entity), gx.props.get(entity))
                    {
                        if let Some(r) = batch.region(gpu, draw, allocator) {
                            batch.push_admitted(r, 1, sel, pipe_class);
                        }
                    }
                }
            }
        }
        if !batch.has_regions() {
            return;
        }
        // fork: GPU cull — a view that can run the two-phase Hi-Z cull hands it the batch; `None`
        // draws as before (`gpu_cull`).
        #[cfg(not(target_arch = "wasm32"))]
        let cull = gpu_cull::GxCullPass::new(world, extracted, depth, !pipes.prepass.is_empty());
        #[cfg(not(target_arch = "wasm32"))]
        batch.set_culled(cull.is_some());
        batch.finish();
        #[cfg(not(target_arch = "wasm32"))]
        let cull = cull.and_then(|c| c.bind(world, &batch));
        let view_offsets = &view_bind.main_offsets;
        // `render/static_gx/elapsed_gpu` on Vulkan and DX12: the perf journal's `gpu_static`.
        // fork: the depth prepass — the same draws in the same order through the depth-only twins,
        // in a pass of its own (it takes the depth clear; the colour pass below then loads it).
        // The attachments are taken here, in pass order, because the first to ask for one takes
        // its clear; the two passes then record as jobs of their own (`crate::par_encode`), in
        // parallel on native and submitted prepass first everywhere.
        let mut jobs = Vec::with_capacity(2);
        if !pipes.prepass.is_empty() {
            jobs.push((None, depth.get_attachment(StoreOp::Store)));
        }
        let depth_attachment = depth.get_attachment(StoreOp::Store);
        jobs.push((Some(target.get_color_attachment()), depth_attachment));
        let batch = &batch;
        #[cfg(not(target_arch = "wasm32"))]
        let (cull, view_depth) = (cull.as_ref(), depth);
        // fork: F — the scene the colour pipelines trace against while the lane is live, else its
        // stand-in (the shader then takes the cascades' path); with neither (before the scene's
        // first prepare) a pipeline that declares it cannot draw.
        #[cfg(not(target_arch = "wasm32"))]
        let rt_bind = if pipes.rt {
            let Some(bind) = world
                .get_resource::<rt_scene::RtScene>()
                .and_then(|s| s.lane_bind(crate::modern::shadow::rt_shadows()))
            else {
                return;
            };
            Some(bind)
        } else {
            None
        };
        crate::par_encode::record(
            &mut render_context,
            jobs,
            |render_context, (color, depth)| {
                let diagnostics = render_context.diagnostic_recorder();
                // fork: GPU cull — the prepass job records phase 1 before its pass, and the
                // pyramid, phase 2 and the late prepass after it (`gpu_cull`).
                #[cfg(not(target_arch = "wasm32"))]
                let cull = cull.filter(|_| color.is_none());
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(cull) = cull {
                    cull.early(render_context);
                }
                let (label, table) = if color.is_some() {
                    ("static_gx", &ready)
                } else {
                    ("static_gx_prepass", &ready_pre)
                };
                // The prepass has no colour target at all, not an empty slot.
                let colors = [color];
                let used = usize::from(colors[0].is_some());
                let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &colors[..used],
                    depth_stencil_attachment: Some(depth),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let span = diagnostics.pass_span(&mut pass, label);
                pass.set_bind_group(0, &view_bind.main, view_offsets);
                pass.set_bind_group(2, &light_bind.0, &[]);
                #[cfg(not(target_arch = "wasm32"))]
                if let (Some(bind), 1) = (rt_bind, used) {
                    pass.set_bind_group(RT_SHADOW_GROUP as usize, bind, &[]);
                }
                batch.emit(&mut pass, table, 0);
                span.end(&mut pass);
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(cull) = cull {
                    drop(pass);
                    cull.pyramid(render_context);
                    cull.late(render_context);
                    // After the early prepass took the clear, the attachment loads.
                    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("static_gx_prepass_late"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(view_depth.get_attachment(StoreOp::Store)),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let span = diagnostics.pass_span(&mut pass, "static_gx_prepass_late");
                    pass.set_bind_group(0, &view_bind.main, view_offsets);
                    pass.set_bind_group(2, &light_bind.0, &[]);
                    batch.emit_late(&mut pass, &ready_pre, 0);
                    span.end(&mut pass);
                }
            },
        );
    }
}

/// The pipelines of `ids` as a table in [`batch::pipe_class`] order, `None` where a key has none;
/// `None` altogether while any of them is still compiling.
fn pipeline_table<'a>(
    pipeline_cache: &'a PipelineCache,
    ids: &HashMap<(bool, bool), CachedRenderPipelineId>,
) -> Option<[Option<&'a RenderPipeline>; 4]> {
    let mut table = [None; 4];
    for (&(cutout, two_sided), id) in ids {
        let class = pipe_class(cutout, two_sided)?;
        table[usize::from(class)] = Some(pipeline_cache.get_render_pipeline(*id)?);
    }
    Some(table)
}

/// Wire the render half (called by the plugin only when armed).
pub(super) fn build(app: &mut App) {
    // The shader registers in `crate::shaders`: `embedded_asset!` prefixes by the calling file.
    // `publish_gx_world` mirrors `StaticGx`'s published half into this resource for extraction.
    app.add_plugins((
        ExtractResourcePlugin::<GxWorld>::default(),
        ExtractComponentPlugin::<StaticGxView>::default(),
    ));
    app.init_resource::<GxWorld>();
    app.add_systems(Update, mark_world_camera);
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
        return;
    };
    render_app
        .init_resource::<GxGpuCache>()
        .init_resource::<GxTexturePool>()
        .init_resource::<batch::GxDrawPool>()
        .add_systems(
            bevy::render::RenderStartup,
            (
                init_pipelines
                    .after(bevy::pbr::MeshPipelineSystems)
                    .after(crate::modern::caps::init),
                batch::init_indirect.after(crate::modern::caps::init),
            ),
        )
        .add_systems(
            Render,
            (
                prepare_static_gx.in_set(RenderSystems::PrepareResources),
                batch::prepare_indirect.in_set(RenderSystems::PrepareResources),
                prepare_view_bind.in_set(RenderSystems::PrepareBindGroups),
            ),
        )
        // Before bevy's opaque pass, as early-z occluders for terrain. This pass takes the depth
        // and colour clear (attachments clear on first use); with nothing visible it returns
        // untouched and the opaque pass clears.
        .add_systems(
            Core3d,
            static_gx_pass
                .in_set(Core3dSystems::MainPass)
                .before(main_opaque_pass_3d),
        );
    // fork: modern lighting (FORK.md) — static casters for the modern sun.
    modern_shadow::build(render_app);
    // fork: modern lighting (FORK.md, P5) — the same casters voxelized for the probe GI.
    modern_gi::build(render_app);
    // fork: the ray-traced scene over the same regions and the terrain, native only.
    #[cfg(not(target_arch = "wasm32"))]
    rt_scene::build(render_app);
    // fork: the two-phase Hi-Z GPU cull of the retained passes, native only.
    #[cfg(not(target_arch = "wasm32"))]
    gpu_cull::build(app);
}

/// Copy the collector's published half into the extractable resource.
pub(super) fn publish_gx_world(gx: Res<super::StaticGx>, mut out: ResMut<GxWorld>) {
    let _t = super::gx_perf_guard(2);
    // Write only what changed: a write through `ResMut` marks the resource changed, which costs a
    // full clone at extract. The maps hold `Arc`s, so identity is pointer identity.
    fn same_arcs<K: std::hash::Hash + Eq>(
        a: &HashMap<K, std::sync::Arc<GxCellDraw>>,
        b: &HashMap<K, std::sync::Arc<GxCellDraw>>,
    ) -> bool {
        a.len() == b.len()
            && a.iter()
                .all(|(k, v)| b.get(k).is_some_and(|w| std::sync::Arc::ptr_eq(v, w)))
    }
    let w = &gx.world;
    if !same_arcs(&out.cells, &w.cells) {
        out.cells.clone_from(&w.cells);
    }
    if out.visible != w.visible {
        out.visible.clone_from(&w.visible);
    }
    if !same_arcs(&out.wmos, &w.wmos) {
        out.wmos.clone_from(&w.wmos);
    }
    if !same_arcs(&out.props, &w.props) {
        out.props.clone_from(&w.props);
    }
    if out.visible_wmos != w.visible_wmos {
        out.visible_wmos.clone_from(&w.visible_wmos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(index_range: Range<u32>, cutout: bool) -> GxItemDraw {
        GxItemDraw {
            index_range,
            texture: None,
            cutout,
            two_sided: false,
            vertex_range: 0..0,
            group: None,
            order: 0,
            sidn: [0; 3],
            slot: 0,
        }
    }

    /// Bake order, each item at its own range: the order before any reorder.
    fn as_baked(draws: &[GxItemDraw]) -> (Vec<u32>, Vec<Range<u32>>) {
        (
            (0..draws.len() as u32).collect(),
            draws.iter().map(|d| d.index_range.clone()).collect(),
        )
    }

    #[test]
    fn runs_fuse_live_items_and_split_at_kills() {
        let draws = vec![
            item(0..3, false),
            item(3..6, false),
            item(6..9, false),
            item(9..12, true), // bucket change
            item(12..15, true),
        ];
        let slots = vec![0, 0, 0, 0, 1]; // the last item binds another pool class
        let (order, ranges) = as_baked(&draws);
        let runs = build_runs(&draws, &order, &ranges, &slots, &[]);
        assert_eq!(runs.len(), 3, "opaque span fused; cutout split by slot");
        assert_eq!(runs[0].index_range, 0..9);
        assert_eq!(runs[1].index_range, 9..12);
        assert!(runs[1].cutout);
        assert_eq!(runs[2].slot, 1);
        // Kill the middle opaque item: the fused run splits around it.
        let runs = build_runs(&draws, &order, &ranges, &slots, &[0b010u64]);
        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0].index_range, 0..3);
        assert_eq!(runs[1].index_range, 6..9);
        // Kill everything: nothing is submitted.
        assert!(build_runs(&draws, &order, &ranges, &slots, &[0b11111u64]).is_empty());
    }

    /// A held-back item (texture not resident) hides alone: the fader kills and the unready bits
    /// both hide, and every other item of the region still draws.
    #[test]
    fn an_unready_texture_hides_its_own_items_not_the_region() {
        let draws = vec![item(0..3, false), item(3..6, false), item(6..9, false)];
        let slots = vec![0, 0, 0];
        let killed = [0b001u64];
        let unready = [0b100u64];
        let hidden = hidden_bits(&killed, &unready);
        assert_eq!(hidden, vec![0b101u64]);
        let (order, ranges) = as_baked(&draws);
        let runs = build_runs(&draws, &order, &ranges, &slots, &hidden);
        assert_eq!(runs.len(), 1, "the one ready, live item draws");
        assert_eq!(runs[0].index_range, 3..6);
        // Bitmaps of different lengths: a region with no faders has an empty kill bitmap.
        assert_eq!(hidden_bits(&[], &[0b1u64, 0b10]), vec![0b1u64, 0b10]);
    }

    /// Items whose pool class alternates in bake order draw as one run per class once reordered,
    /// and the whole-region runs merge across selection keys.
    #[test]
    fn a_reordered_region_draws_one_run_per_class() {
        let mut draws: Vec<GxItemDraw> = (0..6).map(|i| item(i * 3..i * 3 + 3, false)).collect();
        for (i, d) in draws.iter_mut().enumerate() {
            d.group = Some((i / 3) as u16);
        }
        let slots = vec![0, 1, 0, 1, 0, 1];
        let (order, ranges) = as_baked(&draws);
        assert_eq!(build_runs(&draws, &order, &ranges, &slots, &[]).len(), 6);
        // The assembly's order: (cutout, two_sided, slot, group, index), items packed in it.
        let mut order: Vec<u32> = (0..6).collect();
        order.sort_by_key(|&i| (slots[i as usize], draws[i as usize].group, i));
        let mut ranges = vec![0..0; 6];
        let mut at = 0;
        for &i in &order {
            ranges[i as usize] = at..at + 3;
            at += 3;
        }
        let runs = build_runs(&draws, &order, &ranges, &slots, &[]);
        assert_eq!(runs.len(), 4, "one run per (slot, group)");
        let whole = whole_runs(&runs);
        assert_eq!(whole.len(), 2, "one run per slot across groups");
        assert_eq!(whole[0].index_range, 0..9);
        assert_eq!(whole[1].index_range, 9..18);
    }
}

// fork: the shared region table, index arena and bind group of the bindless layout.
mod bindless;
// fork: the state-sorted draw list every retained pass issues through, multi-drawn on native.
mod batch;
use batch::{pipe_class, GxBatch};
// fork: modern lighting (FORK.md) — see `modern/HOOKS.md`.
mod modern_shadow;
pub(crate) use modern_shadow::gx_shadow_pass;
mod modern_gi;
pub(crate) use modern_gi::gi_voxel_pass;
// fork: the ray-traced scene: BLASes, the TLAS and the hit data the ray-traced lanes bind
// (`modern/HOOKS.md`, "Interfaces for the ray-traced lanes"). Native only.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod rt_scene;
// fork: the two-phase Hi-Z GPU cull of the colour pass and its prepass, and the pyramid later lanes
// reuse (`modern/HOOKS.md`, "Interfaces for the Hi-Z pyramid"). Native only.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod gpu_cull;
