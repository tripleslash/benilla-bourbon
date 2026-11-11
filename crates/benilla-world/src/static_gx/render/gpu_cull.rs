//! fork: two-phase Hi-Z occlusion culling of static_gx's bucketed draws on the GPU, native only.
//!
//! # When it runs
//!
//! On where static_gx draws bindless with native multi-draw ([`super::batch::gpu_cull_lane`]:
//! `HwCaps.binding_arrays && mdi_count`, not `WOW_GX_BINDLESS=0`, not `WOW_STATIC_GX_MDI=0`)
//! unless `WOW_GX_GPUCULL=0`; decided once at render startup. It culls a view's colour pass and
//! depth prepass where that view lays a prepass (a modern frame, not `WOW_STATIC_GX_PREPASS=0`)
//! and its depth texture is bindable ([`arm_depth`] opts both static_gx views in). Anywhere else,
//! and with the switch off, the batch draws as B drew it.
//!
//! # What runs, per view, in the prepass job's encoder
//!
//! 1. `static_gx_cull_early` (compute, [`GxCullPass::early`]): one thread per candidate draw of
//!    the view's batch (exactly what the CPU walk handed this pass, so a region `rt_scene` only
//!    assembled off screen is never a candidate). Outside the current frustum: culled. Else
//!    tested against this view's pyramid as last frame built it, under last frame's matrix: not
//!    occluded there, it goes into its bucket (`args`, `counts`); occluded, it is deferred.
//! 2. `static_gx_prepass` draws the buckets: what was visible last frame, the static occluders.
//! 3. `static_gx_hiz` (compute, [`GxCullPass::pyramid`]): this frame's pyramid from that depth.
//! 4. `static_gx_cull_late` (compute, [`GxCullPass::late`]): the deferred draws against the new
//!    pyramid; a survivor is appended to its bucket and written into the late buckets.
//! 5. `static_gx_prepass_late` draws the late buckets (`GxBatch::emit_late`).
//!
//! Where the mesh-shader lane runs (`meshlets.rs`) and a batch maps onto meshlets, the two phases
//! run their `GX_MESHLETS` variant (`static_gx_cull_early_mesh` / `_late_mesh` pipelines, the same
//! spans): a survivor also gets its task dispatch and candidate ref beside its args, and the three
//! passes draw it as meshlets; its args stay valid, so the vertex path can still draw them.
//!
//! The colour pass then draws the buckets, which hold both phases' survivors by then. No CPU work
//! is added past one uniform write and four queue writes a view (`batch.rs`, the module header):
//! the GPU fills the args and counts B's multi-draws already read.
//!
//! A draw's bound is its selection key's (a WMO group's or a prop referrer set's, held in the
//! region table, `bindless.rs`) or, for a cell or a draw fused across keys, its region's.
//!
//! # The pyramid (the interface later lanes reuse)
//!
//! One per static_gx view, [`GxHiZ::pyramid`]: `R32Float`, level 0 the previous power of two of
//! the view's depth texture in each axis, a full mip chain down to 1x1. Each texel holds the
//! farthest reverse-Z depth (the minimum) of the depth texels it covers, footprints taken
//! conservatively (`gx_hiz.wesl`); uv (0, 0) is the depth texture's top left. It holds the static
//! world's early-phase depth only: the regions drawn in step 2, no terrain, no entity, no late
//! draw. It is valid from step 3 of this frame until step 3 of the next, and reads as a
//! `texture_2d<f32>` (unfilterable) with `textureLoad`.
//!
//! # Instruments
//!
//! The four spans above show in `WOW_GPU_PASSES` and `WOW_GPU_STATS`. With `WOW_GX_PERF`, each
//! cull counts its candidates and both phases' survivors on the GPU, read back a few frames late
//! into a `GX_CULL` line every 64 frames; `GX_PERF`'s `draws/frame gx=` counts the multi-draw
//! calls, the late prepass's included.

use super::batch::{GxBatch, GxIndirect};
use super::bindless::GxBindless;
use super::StaticGxView;
use crate::par_encode::Recorder;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_depth_2d,
    texture_depth_2d_multisampled, texture_storage_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::{ExtractedView, RetainedViewEntity, ViewDepthStencilTexture};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::ShaderDefVal;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// The lane's startup decision, for the main world's [`arm_depth`].
static LANE: AtomicBool = AtomicBool::new(false);

/// Threads per cull workgroup (`gx_cull.wesl`).
const CULL_WORKGROUP: u32 = 64;

/// Texels per pyramid workgroup side (`gx_hiz.wesl`).
const HIZ_TILE: u32 = 8;

/// `CullUniform::flags`: the pyramid holds a frame built under `prev_clip_from_world`.
const PREV_VALID: u32 = 1;

/// `CullUniform::flags`: count candidates and survivors into the stats buffer.
const STATS: u32 = 2;

/// `CullUniform::flags`: the view mirrors handedness, so the meshlet cone test is skipped.
const MIRRORED: u32 = 4;

/// Bytes of the stats buffer: candidates, early survivors, late survivors, unused.
pub(super) const STATS_SIZE: u64 = 16;

/// Readback buffers a view keeps for the stats, so a copy never waits on a map.
const READBACKS: usize = 3;

/// Frames a view's pyramid outlives its last frame (a mirror that stands down and comes back).
pub(super) const KEEP_FRAMES: u32 = 300;

/// The cull's uniform. Keep in sync with `CullView` in `gx_occlusion.wesl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullUniform {
    clip_from_world: [f32; 16],
    prev_clip_from_world: [f32; 16],
    pyramid_size: [f32; 2],
    mip_count: u32,
    flags: u32,
    args_start: u32,
    count: u32,
    pad: [u32; 2],
    /// The unjittered eye in world space (`w` 1), for the meshlet cone test.
    eye: [f32; 4],
}

/// Bytes of [`CullUniform`].
pub(super) const CULL_UNIFORM_SIZE: u64 = std::mem::size_of::<CullUniform>() as u64;

/// A readback buffer's state.
const FREE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const MAPPED: u8 = 3;

/// One stats readback buffer and where it is in its copy, map, read cycle.
struct Readback {
    buffer: Buffer,
    state: Arc<AtomicU8>,
}

/// Four GPU counters (`array<atomic<u32>, 4>`) read back a few frames late, so a copy never waits
/// on a map: the cull's `GX_CULL` counters and the meshlet lane's `GX_MESHLET` ones.
pub(super) struct GpuCounters {
    buffer: Buffer,
    readback: [Readback; READBACKS],
    next: AtomicUsize,
    /// Sums since the last [`Self::take`]: readbacks landed, then counters 0, 1 and 2.
    totals: [u64; 4],
}

impl GpuCounters {
    pub(super) fn new(device: &RenderDevice, label: &'static str, readback: &'static str) -> Self {
        let buffer = |label: &'static str, usage: BufferUsages| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: STATS_SIZE,
                usage,
                mapped_at_creation: false,
            })
        };
        let slot = || Readback {
            buffer: buffer(readback, BufferUsages::MAP_READ | BufferUsages::COPY_DST),
            state: Arc::new(AtomicU8::new(FREE)),
        };
        Self {
            buffer: buffer(
                label,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            ),
            readback: [slot(), slot(), slot()],
            next: AtomicUsize::new(0),
            totals: [0; 4],
        }
    }

    /// The counters, for a bind group.
    pub(super) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Zero the counters, before the passes that count.
    pub(super) fn clear(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.buffer, 0, None);
    }

    /// Copy the counters into a free readback buffer, after the passes that count; none free
    /// skips this frame's.
    pub(super) fn copy(&self, encoder: &mut CommandEncoder) {
        let at = self.next.fetch_add(1, Ordering::Relaxed) % self.readback.len();
        let r = &self.readback[at];
        if r.state
            .compare_exchange(FREE, COPIED, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &r.buffer, 0, STATS_SIZE);
        }
    }

    /// Map what the last frames copied and fold what has landed into the totals.
    pub(super) fn poll(&mut self) {
        for r in &self.readback {
            match r.state.load(Ordering::Acquire) {
                COPIED => {
                    r.state.store(MAPPING, Ordering::Release);
                    let state = Arc::clone(&r.state);
                    r.buffer.map_async(MapMode::Read, .., move |result| {
                        let next = if result.is_ok() { MAPPED } else { FREE };
                        state.store(next, Ordering::Release);
                    });
                }
                MAPPED => {
                    let stats: [u32; 4] = {
                        let bytes = r
                            .buffer
                            .get_mapped_range(..)
                            .expect("a mapped stats buffer reads");
                        bytemuck::pod_read_unaligned(&bytes[..STATS_SIZE as usize])
                    };
                    r.buffer.unmap();
                    r.state.store(FREE, Ordering::Release);
                    self.totals[0] += 1;
                    for (sum, n) in self.totals[1..].iter_mut().zip(&stats[..3]) {
                        *sum += u64::from(*n);
                    }
                }
                _ => {}
            }
        }
    }

    /// The sums since the last take: `[readbacks, counter 0, counter 1, counter 2]`.
    pub(super) fn take(&mut self) -> [u64; 4] {
        std::mem::take(&mut self.totals)
    }
}

/// A view's pyramid texture and its views.
struct Pyramid {
    /// Held for the views; nothing reads it directly.
    _texture: Texture,
    /// Every level, for the cull's binding.
    all: TextureView,
    /// One view a level, for the build.
    levels: Vec<TextureView>,
    /// Level 0's size in texels.
    size: UVec2,
}

/// One static_gx view's pyramid and everything its cull binds.
struct GxHiZView {
    /// The world camera's (`StaticGxView::primary`), else the mirror's: the `GX_CULL` label.
    primary: bool,
    pyramid: Option<Pyramid>,
    /// Bumped whenever the pyramid is remade, so the bind groups over it are too.
    generation: u32,
    /// The depth texture is multisampled: level 0 reads every sample.
    multisampled: bool,
    /// Level 0's bind group, for `(depth view, generation)`.
    reduce: Option<(TextureViewId, u32, BindGroup)>,
    /// Level `i + 1`'s bind group, reading level `i`.
    down: Vec<BindGroup>,
    /// The cull's bind group, for `(buffers, generation)`.
    cull: Option<([BufferId; 7], u32, BindGroup)>,
    /// fork: meshlets — the mesh variant's bind group, for `(buffers, generation)`.
    cull_mesh: Option<([BufferId; 10], u32, BindGroup)>,
    uniform: Buffer,
    /// `GX_CULL`: candidates, early and late survivors.
    stats: GpuCounters,
    /// The matrix the pyramid's current contents were built under; `None` before the first build
    /// and after a remake.
    history: Mutex<Option<Mat4>>,
    /// The frame [`prepare_gpu_cull`] last saw the view.
    seen: u32,
}

impl GxHiZView {
    fn new(device: &RenderDevice, primary: bool) -> Self {
        Self {
            primary,
            pyramid: None,
            generation: 0,
            multisampled: false,
            reduce: None,
            down: Vec::new(),
            cull: None,
            cull_mesh: None,
            uniform: device.create_buffer(&BufferDescriptor {
                label: Some("static_gx_cull_view"),
                size: CULL_UNIFORM_SIZE,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stats: GpuCounters::new(device, "static_gx_cull_stats", "static_gx_cull_readback"),
            history: Mutex::new(None),
            seen: 0,
        }
    }

    /// Make the pyramid for a level 0 of `size`, and the build's per-level bind groups.
    fn remake(&mut self, size: UVec2, device: &RenderDevice, down_layout: &BindGroupLayout) {
        let levels = 32 - size.x.max(size.y).leading_zeros();
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("static_gx_hiz"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let all = texture.create_view(&TextureViewDescriptor {
            label: Some("static_gx_hiz"),
            ..default()
        });
        let level_views: Vec<TextureView> = (0..levels)
            .map(|level| {
                texture.create_view(&TextureViewDescriptor {
                    label: Some("static_gx_hiz_level"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..default()
                })
            })
            .collect();
        self.down = level_views
            .windows(2)
            .map(|pair| {
                device.create_bind_group(
                    "static_gx_hiz_down",
                    down_layout,
                    &BindGroupEntries::sequential((&pair[0], &pair[1])),
                )
            })
            .collect();
        self.pyramid = Some(Pyramid {
            _texture: texture,
            all,
            levels: level_views,
            size,
        });
        self.generation = self.generation.wrapping_add(1);
        self.reduce = None;
        self.cull = None;
        self.cull_mesh = None;
        *self
            .history
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// The pipelines and their layouts.
struct GxHiZPipes {
    reduce_layout: BindGroupLayoutDescriptor,
    reduce_ms_layout: BindGroupLayoutDescriptor,
    down_layout: BindGroupLayoutDescriptor,
    cull_layout: BindGroupLayoutDescriptor,
    reduce: CachedComputePipelineId,
    reduce_ms: CachedComputePipelineId,
    down: CachedComputePipelineId,
    early: CachedComputePipelineId,
    late: CachedComputePipelineId,
    /// fork: meshlets — the cull's `GX_MESHLETS` variant, where the mesh-shader lane runs.
    mesh: Option<GxCullMeshPipes>,
}

/// fork: meshlets — the two phases under `GX_MESHLETS`: a survivor also gets its task dispatch.
struct GxCullMeshPipes {
    layout: BindGroupLayoutDescriptor,
    early: CachedComputePipelineId,
    late: CachedComputePipelineId,
}

/// The GPU cull: its pipelines and every static_gx view's pyramid. In the render world exactly
/// while the lane runs.
#[derive(Resource)]
pub(crate) struct GxHiZ {
    pipes: GxHiZPipes,
    views: HashMap<RetainedViewEntity, GxHiZView>,
    /// Frames prepared, the clock [`GxHiZView::seen`] counts on.
    frame: u32,
}

/// A view's Hi-Z pyramid as a later lane binds it (the module header, "The pyramid").
#[allow(dead_code)] // the interface for the Hi-Z SSR and task-shader cull lanes
pub(crate) struct HiZPyramid<'a> {
    /// Every level, `R32Float`.
    pub view: &'a TextureView,
    /// Level 0's size in texels.
    pub size: UVec2,
    pub mip_count: u32,
    /// The `clip_from_world` its contents were built under; `None` before its first build.
    pub clip_from_world: Option<Mat4>,
}

impl GxHiZ {
    /// `view`'s pyramid, once one is made. Read it after `crate::static_gx::static_gx_pass` for
    /// this frame's contents.
    #[allow(dead_code)] // the interface for the Hi-Z SSR and task-shader cull lanes
    pub(crate) fn pyramid(&self, view: &RetainedViewEntity) -> Option<HiZPyramid<'_>> {
        let state = self.views.get(view)?;
        let pyramid = state.pyramid.as_ref()?;
        Some(HiZPyramid {
            view: &pyramid.all,
            size: pyramid.size,
            mip_count: pyramid.levels.len() as u32,
            clip_from_world: *state.history.lock().unwrap_or_else(PoisonError::into_inner),
        })
    }
}

/// fork: meshlets — what the task stage binds of one view's cull: its uniform, its pyramid (every
/// level) and the pyramid's generation, which changes whenever the pyramid is remade.
pub(super) struct GxMeshInputs<'a> {
    pub uniform: &'a Buffer,
    pub pyramid: &'a TextureView,
    pub generation: u32,
}

impl GxHiZ {
    /// fork: meshlets — `view`'s [`GxMeshInputs`], once its pyramid exists (after
    /// [`prepare_gpu_cull`]).
    pub(super) fn mesh_inputs(&self, view: &RetainedViewEntity) -> Option<GxMeshInputs<'_>> {
        let state = self.views.get(view)?;
        Some(GxMeshInputs {
            uniform: &state.uniform,
            pyramid: &state.pyramid.as_ref()?.all,
            generation: state.generation,
        })
    }
}

/// Whether the lane runs this session; false until render startup.
pub(crate) fn lane_active() -> bool {
    LANE.load(Ordering::Relaxed)
}

/// The previous power of two of `n` (at least 1).
fn prev_pow2(n: u32) -> u32 {
    if n <= 1 {
        1
    } else {
        1 << (31 - n.leading_zeros())
    }
}

/// The view's `clip_from_world` without jitter, as the cull projects with it; the cull's one
/// texel of margin covers the jitter the depth was drawn with.
fn clip_from_world(view: &ExtractedView) -> Mat4 {
    view.clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse())
}

/// `RenderStartup`, after `batch::init_indirect`: the lane runs where the indirect buffers carry
/// the cull's twins. Queues the five pipelines.
fn init_gpu_cull(
    mut commands: Commands,
    indirect: Res<GxIndirect>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let active = indirect.cull_lane();
    LANE.store(active, Ordering::Relaxed);
    if !active {
        return;
    }
    let target = || texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly);
    let reduce_layout = BindGroupLayoutDescriptor::new(
        "static_gx_hiz_reduce_layout",
        &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (texture_depth_2d(), target())),
    );
    let reduce_ms_layout = BindGroupLayoutDescriptor::new(
        "static_gx_hiz_reduce_ms_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (texture_depth_2d_multisampled(), target()),
        ),
    );
    let down_layout = BindGroupLayoutDescriptor::new(
        "static_gx_hiz_down_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                target(),
            ),
        ),
    );
    let cull_entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, std::num::NonZero::new(CULL_UNIFORM_SIZE)),
            // The region table and the candidates.
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            // Meta, args, counts, late args, late counts.
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_sized(false, std::num::NonZero::new(STATS_SIZE)),
        ),
    );
    let cull_layout = BindGroupLayoutDescriptor::new("static_gx_cull_layout", &cull_entries);
    // fork: meshlets — the same entries, then the mesh meta (read) and the task dispatches and
    // candidate refs the survivors write (`gx_cull.wesl` under `GX_MESHLETS`).
    let cull_mesh_layout = indirect.meshlet_lane().then(|| {
        let extra = BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (10, storage_buffer_read_only_sized(false, None)),
                (11, storage_buffer_sized(false, None)),
                (12, storage_buffer_sized(false, None)),
            ),
        );
        let entries: Vec<BindGroupLayoutEntry> =
            cull_entries.iter().chain(extra.iter()).cloned().collect();
        BindGroupLayoutDescriptor::new("static_gx_cull_mesh_layout", &entries)
    });
    let hiz: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/modern/gx_hiz.wesl");
    let cull: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/modern/gx_cull.wesl");
    let queue = |label: &'static str,
                 layout: &BindGroupLayoutDescriptor,
                 shader: &Handle<Shader>,
                 defs: &[&'static str],
                 entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            constants: Default::default(),
            label: Some(label.into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: defs.iter().map(|&d| ShaderDefVal::from(d)).collect(),
            entry_point: Some(entry.into()),
            zero_initialize_workgroup_memory: false,
        })
    };
    let pipes = GxHiZPipes {
        reduce: queue(
            "static_gx_hiz_reduce",
            &reduce_layout,
            &hiz,
            &["HIZ_FROM_DEPTH"],
            "downsample",
        ),
        reduce_ms: queue(
            "static_gx_hiz_reduce_ms",
            &reduce_ms_layout,
            &hiz,
            &["HIZ_FROM_DEPTH", "HIZ_MULTISAMPLED"],
            "downsample",
        ),
        down: queue("static_gx_hiz_down", &down_layout, &hiz, &[], "downsample"),
        early: queue(
            "static_gx_cull_early",
            &cull_layout,
            &cull,
            &[],
            "cull_early",
        ),
        late: queue("static_gx_cull_late", &cull_layout, &cull, &[], "cull_late"),
        mesh: cull_mesh_layout.map(|layout| GxCullMeshPipes {
            early: queue(
                "static_gx_cull_early_mesh",
                &layout,
                &cull,
                &["GX_MESHLETS"],
                "cull_early",
            ),
            late: queue(
                "static_gx_cull_late_mesh",
                &layout,
                &cull,
                &["GX_MESHLETS"],
                "cull_late",
            ),
            layout,
        }),
        reduce_layout,
        reduce_ms_layout,
        down_layout,
        cull_layout,
    };
    info!("static_gx: two-phase Hi-Z GPU cull (WOW_GX_GPUCULL=0 turns it off)");
    commands.insert_resource(GxHiZ {
        pipes,
        views: HashMap::default(),
        frame: 0,
    });
}

/// Keep each static_gx view's pyramid sized to its depth and its bind groups current; drop the
/// views gone [`KEEP_FRAMES`]; with `WOW_GX_PERF`, fold the readbacks and print `GX_CULL`. After
/// the depth textures, the indirect buffers and the region table are prepared.
pub(super) fn prepare_gpu_cull(
    mut hiz: ResMut<GxHiZ>,
    views: Query<(&ExtractedView, &ViewDepthStencilTexture, &StaticGxView)>,
    indirect: Res<GxIndirect>,
    shared: Option<Res<GxBindless>>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let GxHiZ {
        pipes,
        views: states,
        frame,
    } = &mut *hiz;
    *frame = frame.wrapping_add(1);
    let now = *frame;
    let buffers = indirect.cull_buffers();
    let table = shared.as_deref().and_then(GxBindless::table_buffer);
    for (view, depth, marker) in &views {
        let texture = depth.texture();
        if !texture.usage().contains(TextureUsages::TEXTURE_BINDING) {
            continue;
        }
        let state = states
            .entry(view.retained_view_entity)
            .or_insert_with(|| GxHiZView::new(&device, marker.primary));
        state.seen = now;
        let size = UVec2::new(prev_pow2(texture.width()), prev_pow2(texture.height()));
        if state.pyramid.as_ref().is_none_or(|p| p.size != size) {
            let layout = pipeline_cache.get_bind_group_layout(&pipes.down_layout);
            state.remake(size, &device, &layout);
        }
        let Some(pyramid) = state.pyramid.as_ref() else {
            continue;
        };
        let generation = state.generation;
        let depth_view = depth.attachment.depth_stencil_views().attachment_view();
        if state
            .reduce
            .as_ref()
            .is_none_or(|(id, g, _)| (*id, *g) != (depth_view.id(), generation))
        {
            let multisampled = texture.sample_count() > 1;
            let layout = pipeline_cache.get_bind_group_layout(if multisampled {
                &pipes.reduce_ms_layout
            } else {
                &pipes.reduce_layout
            });
            let bind = device.create_bind_group(
                "static_gx_hiz_reduce",
                &layout,
                &BindGroupEntries::sequential((depth_view, &pyramid.levels[0])),
            );
            state.multisampled = multisampled;
            state.reduce = Some((depth_view.id(), generation, bind));
        }
        let (Some(b), Some(table)) = (buffers, table) else {
            state.cull = None;
            state.cull_mesh = None;
            continue;
        };
        let key = [
            table.id(),
            b.args.id(),
            b.counts.id(),
            b.cand.id(),
            b.meta.id(),
            b.late_args.id(),
            b.late_counts.id(),
        ];
        if state
            .cull
            .as_ref()
            .is_none_or(|(k, g, _)| (*k, *g) != (key, generation))
        {
            let layout = pipeline_cache.get_bind_group_layout(&pipes.cull_layout);
            let bind = device.create_bind_group(
                "static_gx_cull",
                &layout,
                &BindGroupEntries::sequential((
                    state.uniform.as_entire_binding(),
                    table.as_entire_binding(),
                    b.cand.as_entire_binding(),
                    b.meta.as_entire_binding(),
                    b.args.as_entire_binding(),
                    b.counts.as_entire_binding(),
                    b.late_args.as_entire_binding(),
                    b.late_counts.as_entire_binding(),
                    &pyramid.all,
                    state.stats.buffer().as_entire_binding(),
                )),
            );
            state.cull = Some((key, generation, bind));
        }
        // fork: meshlets — the mesh variant's group: the same, then the mesh meta, task
        // dispatches and candidate refs.
        let (Some(mesh_pipes), Some(m)) = (pipes.mesh.as_ref(), b.mesh) else {
            state.cull_mesh = None;
            continue;
        };
        let mesh_key = [
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            m.meta.id(),
            m.args.id(),
            m.refs.id(),
        ];
        if state
            .cull_mesh
            .as_ref()
            .is_none_or(|(k, g, _)| (*k, *g) != (mesh_key, generation))
        {
            let layout = pipeline_cache.get_bind_group_layout(&mesh_pipes.layout);
            let bind = device.create_bind_group(
                "static_gx_cull_mesh",
                &layout,
                &BindGroupEntries::sequential((
                    state.uniform.as_entire_binding(),
                    table.as_entire_binding(),
                    b.cand.as_entire_binding(),
                    b.meta.as_entire_binding(),
                    b.args.as_entire_binding(),
                    b.counts.as_entire_binding(),
                    b.late_args.as_entire_binding(),
                    b.late_counts.as_entire_binding(),
                    &pyramid.all,
                    state.stats.buffer().as_entire_binding(),
                    m.meta.as_entire_binding(),
                    m.args.as_entire_binding(),
                    m.refs.as_entire_binding(),
                )),
            );
            state.cull_mesh = Some((mesh_key, generation, bind));
        }
    }
    states.retain(|_, s| now.wrapping_sub(s.seen) <= KEEP_FRAMES);
    if crate::static_gx::gx_perf_enabled() {
        for state in states.values_mut() {
            state.stats.poll();
        }
        if now.is_multiple_of(64) {
            for state in states.values_mut() {
                let [frames, candidates, early, late] = state.stats.take();
                let per = |n: u64| n / frames.max(1);
                println!(
                    "GX_CULL {} per frame: candidates={} early={} late={} culled={} ({frames} \
                     frames read back)",
                    if state.primary { "world" } else { "mirror" },
                    per(candidates),
                    per(early),
                    per(late),
                    per(candidates.saturating_sub(early + late)),
                );
            }
        }
    }
}

/// This frame's cull of one view: everything it binds, resolved before the batch is finished.
pub(super) struct GxCullPass<'w> {
    state: &'w GxHiZView,
    reduce: &'w ComputePipeline,
    down: &'w ComputePipeline,
    early: &'w ComputePipeline,
    late: &'w ComputePipeline,
    reduce_bind: &'w BindGroup,
    cull_bind: &'w BindGroup,
    /// fork: meshlets — the `GX_MESHLETS` variant's early and late pipelines and bind group,
    /// where the lane runs and all three are ready.
    mesh: Option<(&'w ComputePipeline, &'w ComputePipeline, &'w BindGroup)>,
    /// [`Self::bind`] took the mesh variant: the survivors carry their task dispatches.
    meshlets: bool,
    clip_from_world: Mat4,
    eye: Vec3,
    /// The view's `world_from_view` mirrors handedness (a reflected camera).
    mirrored: bool,
    /// Workgroups of one phase, set by [`Self::bind`].
    groups: u32,
    stats: bool,
}

impl<'w> GxCullPass<'w> {
    /// The view's cull when it can run this frame: the lane runs, the view lays a static prepass
    /// (`prepassed`), its pyramid and bind groups are made for this depth texture, and the five
    /// pipelines are compiled. `None` leaves the batch to B's path.
    pub(super) fn new(
        world: &'w World,
        view: &ExtractedView,
        depth: &ViewDepthStencilTexture,
        prepassed: bool,
    ) -> Option<Self> {
        if !prepassed {
            return None;
        }
        let hiz = world.get_resource::<GxHiZ>()?;
        let state = hiz.views.get(&view.retained_view_entity)?;
        let (depth_id, _, reduce_bind) = state.reduce.as_ref()?;
        if *depth_id
            != depth
                .attachment
                .depth_stencil_views()
                .attachment_view()
                .id()
        {
            return None;
        }
        let (_, _, cull_bind) = state.cull.as_ref()?;
        let cache = world.resource::<PipelineCache>();
        let p = &hiz.pipes;
        let mesh = p.mesh.as_ref().and_then(|m| {
            Some((
                cache.get_compute_pipeline(m.early)?,
                cache.get_compute_pipeline(m.late)?,
                &state.cull_mesh.as_ref()?.2,
            ))
        });
        Some(Self {
            state,
            reduce: cache.get_compute_pipeline(if state.multisampled {
                p.reduce_ms
            } else {
                p.reduce
            })?,
            down: cache.get_compute_pipeline(p.down)?,
            early: cache.get_compute_pipeline(p.early)?,
            late: cache.get_compute_pipeline(p.late)?,
            reduce_bind,
            cull_bind,
            mesh,
            meshlets: false,
            clip_from_world: clip_from_world(view),
            eye: view.world_from_view.translation(),
            mirrored: view.world_from_view.to_matrix().determinant() < 0.0,
            groups: 0,
            stats: crate::static_gx::gx_perf_enabled(),
        })
    }

    /// Take the finished batch's candidates and write the view's uniform; `None` when the batch
    /// took no cull range (it then draws as B drew it, and nothing of the cull is recorded).
    pub(super) fn bind(mut self, world: &World, batch: &GxBatch<'_>) -> Option<Self> {
        let (args_start, count) = batch.cull_range()?;
        let pyramid = self.state.pyramid.as_ref()?;
        let prev = *self
            .state
            .history
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut flags = 0;
        if prev.is_some() {
            flags |= PREV_VALID;
        }
        if self.stats {
            flags |= STATS;
        }
        if self.mirrored {
            flags |= MIRRORED;
        }
        let uniform = CullUniform {
            clip_from_world: self.clip_from_world.to_cols_array(),
            prev_clip_from_world: prev.unwrap_or(Mat4::IDENTITY).to_cols_array(),
            pyramid_size: pyramid.size.as_vec2().to_array(),
            mip_count: pyramid.levels.len() as u32,
            flags,
            args_start,
            count,
            pad: [0; 2],
            eye: self.eye.extend(1.0).to_array(),
        };
        world.resource::<RenderQueue>().write_buffer(
            &self.state.uniform,
            0,
            bytemuck::bytes_of(&uniform),
        );
        self.groups = count.div_ceil(CULL_WORKGROUP);
        // fork: meshlets — a batch `finish` laid out for meshlets takes the mesh variant.
        if let (true, Some((early, late, bind))) = (batch.meshlets(), self.mesh) {
            self.early = early;
            self.late = late;
            self.cull_bind = bind;
            self.meshlets = true;
        }
        Some(self)
    }

    /// fork: meshlets — the survivors carry their task dispatches and candidate refs this frame
    /// (`gx_cull.wesl` under `GX_MESHLETS`), so the mesh-shader lane can draw them.
    pub(super) fn meshlets(&self) -> bool {
        self.meshlets
    }

    /// Phase 1, `static_gx_cull_early`: before the early prepass.
    pub(super) fn early(&self, rec: &mut Recorder<'_>) {
        let diagnostics = rec.diagnostic_recorder();
        let encoder = rec.command_encoder();
        if self.stats {
            self.state.stats.clear(encoder);
        }
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("static_gx_cull_early"),
            timestamp_writes: None,
        });
        let span = diagnostics.pass_span(&mut pass, "static_gx_cull_early");
        pass.set_pipeline(self.early);
        pass.set_bind_group(0, self.cull_bind, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
        span.end(&mut pass);
    }

    /// `static_gx_hiz`: this frame's pyramid from the early prepass's depth, after it.
    pub(super) fn pyramid(&self, rec: &mut Recorder<'_>) {
        let Some(pyramid) = self.state.pyramid.as_ref() else {
            return;
        };
        let diagnostics = rec.diagnostic_recorder();
        let encoder = rec.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("static_gx_hiz"),
            timestamp_writes: None,
        });
        let span = diagnostics.pass_span(&mut pass, "static_gx_hiz");
        let tiles = |level: u32| {
            let size = (pyramid.size >> level).max(UVec2::ONE);
            UVec2::new(size.x.div_ceil(HIZ_TILE), size.y.div_ceil(HIZ_TILE))
        };
        pass.set_pipeline(self.reduce);
        pass.set_bind_group(0, self.reduce_bind, &[]);
        let t = tiles(0);
        pass.dispatch_workgroups(t.x, t.y, 1);
        pass.set_pipeline(self.down);
        for (level, bind) in (1u32..).zip(&self.state.down) {
            pass.set_bind_group(0, bind, &[]);
            let t = tiles(level);
            pass.dispatch_workgroups(t.x, t.y, 1);
        }
        span.end(&mut pass);
        *self
            .state
            .history
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(self.clip_from_world);
    }

    /// Phase 2, `static_gx_cull_late`: after the pyramid, before the late prepass.
    pub(super) fn late(&self, rec: &mut Recorder<'_>) {
        let diagnostics = rec.diagnostic_recorder();
        let encoder = rec.command_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("static_gx_cull_late"),
                timestamp_writes: None,
            });
            let span = diagnostics.pass_span(&mut pass, "static_gx_cull_late");
            pass.set_pipeline(self.late);
            pass.set_bind_group(0, self.cull_bind, &[]);
            pass.dispatch_workgroups(self.groups, 1, 1);
            span.end(&mut pass);
        }
        if self.stats {
            self.state.stats.copy(encoder);
        }
    }
}

/// Main world: opt every static_gx view's depth into `TEXTURE_BINDING`, the pyramid's source (set
/// once and never cleared, as `ssilvb::arm_depth_binding` does). The world camera usually has it
/// already; the water's mirror does not.
fn arm_depth(mut cams: Query<&mut Camera3d, With<StaticGxView>>) {
    if !lane_active() {
        return;
    }
    for mut cam in &mut cams {
        let usage = TextureUsages::from(cam.depth_texture_usages);
        if usage.contains(TextureUsages::TEXTURE_BINDING) {
            continue; // guarded, so the common frame never flags `Camera3d` as changed
        }
        cam.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
    }
}

/// Wire the cull (called from `render::build`, native only).
pub(super) fn build(app: &mut App) {
    app.add_systems(Update, arm_depth);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(
            RenderStartup,
            init_gpu_cull.after(super::batch::init_indirect),
        )
        .add_systems(
            Render,
            prepare_gpu_cull
                .in_set(RenderSystems::PrepareBindGroups)
                .run_if(resource_exists::<GxHiZ>),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The uniform is the size the WGSL struct is, and the pyramid's level 0 never exceeds the
    /// depth it reads.
    #[test]
    fn the_uniform_and_the_pyramid_agree_with_the_shaders() {
        assert_eq!(
            CULL_UNIFORM_SIZE, 176,
            "CullView in gx_occlusion.wesl is 176 bytes"
        );
        let lib = include_str!("../../shaders/modern/gx_occlusion.wesl");
        assert!(lib.contains(&format!("const PREV_VALID: u32 = {PREV_VALID}u;")));
        assert!(lib.contains(&format!("const STATS: u32 = {STATS}u;")));
        assert!(lib.contains(&format!("const MIRRORED: u32 = {MIRRORED}u;")));
        assert!(lib.contains(&format!(
            "const NO_CONE: i32 = {};",
            crate::static_gx::meshlet::NO_CONE
        )));
        let src = include_str!("../../shaders/modern/gx_cull.wesl");
        assert!(src.contains(&format!("@workgroup_size({CULL_WORKGROUP}, 1, 1)")));
        let hiz = include_str!("../../shaders/modern/gx_hiz.wesl");
        assert!(hiz.contains(&format!("@workgroup_size({HIZ_TILE}, {HIZ_TILE}, 1)")));
        assert_eq!(
            [1, 1919, 1920, 2048, 2049].map(prev_pow2),
            [1, 1024, 1024, 2048, 2048]
        );
    }
}
