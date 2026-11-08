//! fork: static_gx's mesh-shader lane, native only: the GPU cull's survivors drawn as meshlets
//! through a task and a mesh stage, the task stage culling each meshlet.
//!
//! # When it runs
//!
//! Where [`capable`] holds (`HwCaps.mesh_shader`, which only Vulkan reports, since naga emits
//! task and mesh stages to SPIR-V alone; the GPU cull's lane on native multi-draw; not
//! `WOW_GX_MESHLETS=0`), decided once at render startup ([`lane_active`]): every bake from then
//! on cuts its items into meshlets (`crate::static_gx::meshlet`), every region's meshlets live in
//! the meshlet arena (`bindless.rs`), and the colour pass and its prepasses draw a view's culled
//! batch as meshlets wherever the whole batch can be: every draw a run of whole items of regions
//! that have meshlets ([`super::batch::GxBatch::meshlets`]), the cull's mesh variant ready, the
//! mesh pipelines compiled and a mesh group made for every slab the batch draws from. Anything
//! short of that, that view and frame draws as the GPU cull draws (`gpu_cull.rs`), from the same
//! cull: the mesh variant writes the vertex path's args as well.
//!
//! # One mechanism per grain
//!
//! The GPU cull stays the one draw-level cull (a draw outside the frustum or behind the pyramid
//! never launches a task). The task stage adds the meshlet grain on its survivors: each meshlet's
//! sphere against the frustum, its normal cone against the eye, and its box against the same
//! pyramid in the same two phases (`static_gx_mesh.wgsl`, the header). A view keeps one bit per
//! meshlet of the table ([`MeshletView`], `drawn`), written by the two prepasses and read by the
//! colour pass, so the colour pass shades exactly the meshlets whose depth the prepasses laid.
//!
//! # What runs, per view
//!
//! | span | what |
//! |---|---|
//! | `static_gx_cull_early` | C's early phase, the mesh variant: survivors also get their task dispatch |
//! | `static_gx_prepass` | the early survivors as meshlets, mode EARLY (last frame's pyramid) |
//! | `static_gx_hiz` | this frame's pyramid |
//! | `static_gx_cull_late` | C's late phase, the mesh variant |
//! | `static_gx_prepass_late` | every survivor's meshlets not yet drawn, mode LATE (this frame's pyramid) |
//! | `static_gx` | the colour pass, mode COLOUR: the meshlets either prepass drew |
//!
//! Shadow cascades, the lamp atlas and the GI voxelizer stay on the vertex path: they draw regions
//! whole against cached layers, with no occlusion to gain.
//!
//! # Instruments
//!
//! The spans above, unchanged, in `WOW_GPU_PASSES` / `WOW_GPU_STATS` (pipeline statistics count
//! task and mesh invocations there); `GX_PERF`'s draw count counts one multi-draw a bucket as
//! before. With `WOW_GX_PERF`, the task stage counts the meshlets it examined and drew in each
//! phase, read back into a `GX_MESHLET <world|mirror> per frame: meshlets= early= late= culled=`
//! line every 64 frames.

use super::batch::{GxBatch, GxIndirect};
use super::bindless::{GxBindless, GxSlots};
use super::gpu_cull::{
    GpuCounters, GxCullPass, GxHiZ, CULL_UNIFORM_SIZE, KEEP_FRAMES, STATS_SIZE,
};
use super::{GxCellDraw, GxGpuCache, StaticGxView};
use crate::modern::caps::HwCaps;
use crate::par_encode::Recorder;
use crate::static_gx::meshlet::GxMeshlets;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::mesh::allocator::{MeshAllocator, MeshAllocatorSettings};
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::{ExtractedView, RetainedViewEntity};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::{Shader, ShaderDefVal};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

/// The lane's startup decision.
static LANE: AtomicBool = AtomicBool::new(false);

/// The task stage's modes, the immediates' second word (`static_gx_mesh.wgsl`, `GX_MODE_*`).
pub(super) const MODE_EARLY: u32 = 0;
pub(super) const MODE_LATE: u32 = 1;
pub(super) const MODE_COLOUR: u32 = 2;

/// Bytes of the immediates: the bucket's first arg slot, the mode.
const IMMEDIATE_BYTES: u32 = 8;

/// The task entry point of every mesh pipeline.
const TASK_ENTRY: &str = "meshlet_task";

/// The lane's shader: `static_gx.wgsl` under `GX_MESHLETS` with the mesh entries appended
/// (`static_gx_mesh.wgsl`), built from the sources at startup, so a hot reload of either file
/// does not reach it.
pub(super) const MESH_SHADER: Handle<Shader> =
    bevy::asset::uuid_handle!("4b7d2c91-e3a5-4f08-b6d1-9c2e5a7f3d18");

/// The composed source: the mesh-shader and draw-index enables must lead the module, and naga
/// refuses either on a device without it, so they are never in `static_gx.wgsl` itself.
const MESH_SOURCE: &str = concat!(
    "enable wgpu_mesh_shader, draw_index;\n",
    "#import benilla_modern::gx_occlusion\n",
    include_str!("../../shaders/static_gx.wgsl"),
    "\n",
    include_str!("../../shaders/modern/static_gx_mesh.wgsl"),
);

/// `WOW_GX_MESHLETS=0` turns the lane off: static_gx draws through the GPU cull's path.
fn switched_on() -> bool {
    crate::launch_env::var("WOW_GX_MESHLETS").as_deref() != Some("0")
}

/// Whether the lane can run this session: the device has mesh shaders as the lane needs them
/// ([`HwCaps::mesh_shader`]), the GPU cull's lane runs on native multi-draw, and the switch is
/// on. What widens group 1 and group 2 to the task and mesh stages and makes the meshlet arena.
pub(super) fn capable(caps: Option<&HwCaps>) -> bool {
    caps.is_some_and(|c| c.mesh_shader)
        && super::batch::gpu_cull_lane(caps)
        && crate::launch_env::var("WOW_STATIC_GX_MDI").as_deref() != Some("0")
        && switched_on()
}

/// Whether the lane runs this session; false until render startup.
pub(super) fn lane_active() -> bool {
    LANE.load(Ordering::Relaxed)
}

/// A region's meshlet rows in the assembly's item order `order`, and per position `k` of
/// `order` its first meshlet (`starts[order.len()]` is the region's count).
fn arrange(m: &GxMeshlets, order: &[u32]) -> (Vec<[u32; 4]>, Vec<u32>) {
    let mut rows = Vec::with_capacity(m.rows.len());
    let mut starts = Vec::with_capacity(order.len() + 1);
    for &i in order {
        starts.push((rows.len() / 2) as u32);
        let i = i as usize;
        if let (Some(&a), Some(&b)) = (m.items.get(i), m.items.get(i + 1)) {
            rows.extend_from_slice(&m.rows[2 * a as usize..2 * b as usize]);
        }
    }
    starts.push((rows.len() / 2) as u32);
    (rows, starts)
}

/// `assemble_region`'s hook, after `place`: `draw`'s meshlets into the arena in `order` (the
/// index arena's item order), their blocks into `slots`. Returns the per-position meshlet starts
/// [`span`] maps a draw through; empty where the region has none (baked before the lane was
/// wanted, or no arena), and a batch holding such a region draws through the GPU cull's path.
pub(super) fn place(
    shared: &mut GxBindless,
    slots: &mut GxSlots,
    draw: &GxCellDraw,
    order: &[u32],
    device: &RenderDevice,
    queue: &RenderQueue,
    copies: &mut CommandEncoder,
) -> Vec<u32> {
    let Some(m) = draw.meshlets.as_deref() else {
        return Vec::new();
    };
    if m.items.len() != draw.draws.len() + 1 {
        return Vec::new();
    }
    let (rows, starts) = arrange(m, order);
    match shared.place_meshlets(&rows, &m.vertices, &m.triangles, device, queue, copies) {
        Some(placed) => {
            slots.meshlets = placed;
            starts
        }
        None => Vec::new(),
    }
}

/// The meshlets `(first, count)`, relative to the region's block, of a draw over the region's
/// arena indices `first..first + count` (relative to `index_base`): a draw is a run of whole items
/// in `order`, whose ranges (`ranges`, per item) rise along it.
pub(super) fn span(
    ranges: &[Range<u32>],
    order: &[u32],
    starts: &[u32],
    first: u32,
    count: u32,
) -> Option<(u32, u32)> {
    let at = |index: u32| order.partition_point(|&i| ranges[i as usize].start < index);
    let (a, b) = (*starts.get(at(first))?, *starts.get(at(first + count))?);
    Some((a, b.checked_sub(a)?))
}

/// The mesh group (`static_gx_mesh.wgsl`, `GX_MESH_GROUP`): the cull's view uniform, the meshlet
/// table, vertices and triangles, the draw's vertex slab, the candidates, the mesh meta, the
/// candidate refs, the view's drawn bits, its pyramid and its counters. Task and mesh stages.
pub(super) fn mesh_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "static_gx_mesh_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::TASK | ShaderStages::MESH,
            (
                uniform_buffer_sized(false, std::num::NonZero::new(CULL_UNIFORM_SIZE)),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer_sized(false, std::num::NonZero::new(STATS_SIZE)),
            ),
        ),
    )
}

/// Queue the mesh twin of a static_gx pipeline: the same fragment, depth, raster and layout, the
/// mesh group appended (its index is `desc`'s group count), the task and mesh stages
/// (`meshlet_task` and `entry`) of [`MESH_SHADER`] under `desc`'s defs and `GX_MESHLETS`, and the
/// immediates.
pub(super) fn queue_pipeline(
    cache: &PipelineCache,
    desc: &RenderPipelineDescriptor,
    entry: &'static str,
) -> CachedRenderPipelineId {
    let defs = [
        ShaderDefVal::from("GX_MESHLETS"),
        ShaderDefVal::UInt("GX_MESH_GROUP".into(), desc.layout.len() as u32),
    ];
    let mut mesh = desc.clone();
    mesh.label = mesh.label.map(|label| format!("{label} meshlets").into());
    mesh.layout.push(mesh_layout());
    mesh.immediate_size = IMMEDIATE_BYTES;
    mesh.vertex.shader = MESH_SHADER;
    mesh.vertex.entry_point = Some(entry.into());
    mesh.vertex.buffers.clear();
    mesh.vertex.shader_defs.extend(defs.iter().cloned());
    if let Some(fragment) = &mut mesh.fragment {
        fragment.shader = MESH_SHADER;
        fragment.shader_defs.extend(defs);
    }
    cache.queue_mesh_pipeline(mesh, Some(TASK_ENTRY.into()))
}

/// One static_gx view's share of the lane.
struct MeshletView {
    /// The world camera's, else the mirror's: the `GX_MESHLET` label.
    primary: bool,
    /// One bit per meshlet of the table, for [`Self::bits_for`] meshlets; this frame's only.
    drawn: Option<Buffer>,
    bits_for: u32,
    /// `GX_MESHLET`: meshlets examined, drawn early, drawn late.
    stats: GpuCounters,
    /// What the slab groups were made over besides their slab: the buffers' ids and the
    /// pyramid's generation.
    base: Option<([BufferId; 7], u32)>,
    /// The mesh group per vertex slab.
    groups: HashMap<BufferId, BindGroup>,
    /// The frame [`prepare_meshlets`] last saw the view.
    seen: u32,
}

/// The lane's per-view state. In the render world exactly while the lane runs.
#[derive(Resource, Default)]
pub(super) struct GxMeshletLane {
    views: HashMap<RetainedViewEntity, MeshletView>,
    /// Frames prepared, the clock [`MeshletView::seen`] counts on.
    frame: u32,
}

/// `RenderStartup`, after `batch::init_indirect`: the lane runs where the indirect buffers carry
/// its twins; from here on every bake builds meshlets.
fn init_meshlets(mut commands: Commands, indirect: Res<GxIndirect>) {
    let on = indirect.meshlet_lane();
    LANE.store(on, Ordering::Relaxed);
    crate::static_gx::meshlet::set_wanted(on);
    if on {
        info!(
            "static_gx: meshlets drawn by task and mesh shaders (WOW_GX_MESHLETS=0 turns it off)"
        );
        commands.init_resource::<GxMeshletLane>();
    }
}

/// Keep every static_gx view's drawn bits sized to the meshlet table and a mesh group made for
/// every vertex slab a cached region lives in; with `WOW_GX_PERF`, fold the readbacks and print
/// `GX_MESHLET`. After the GPU cull's bind groups (the pyramid is sized there).
#[allow(clippy::too_many_arguments)]
fn prepare_meshlets(
    mut lane: ResMut<GxMeshletLane>,
    views: Query<(&ExtractedView, &StaticGxView)>,
    hiz: Option<Res<GxHiZ>>,
    indirect: Res<GxIndirect>,
    shared: Option<Res<GxBindless>>,
    cache: Res<GxGpuCache>,
    allocator: Res<MeshAllocator>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let GxMeshletLane {
        views: states,
        frame,
    } = &mut *lane;
    *frame = frame.wrapping_add(1);
    let now = *frame;
    let meshlets = shared.as_deref().and_then(GxBindless::meshlet_buffers);
    let cull = indirect.cull_buffers();
    let (Some(hiz), Some(m), Some(c), Some(mesh)) = (
        hiz.as_deref(),
        meshlets,
        cull,
        cull.and_then(|c| c.mesh),
    ) else {
        return;
    };
    // Every slab a cached region draws from, once; a slab past the storage binding limit gets no
    // group, and a batch drawing from it takes the GPU cull's path.
    let limit = device.limits().max_storage_buffer_binding_size;
    let mut slabs: HashMap<BufferId, &Buffer> = HashMap::default();
    for gpu in cache
        .cells
        .values()
        .chain(cache.wmos.values())
        .chain(cache.props.values())
    {
        if let Some(slice) = allocator.mesh_vertex_slice(&gpu.mesh) {
            if slice.buffer.size() <= limit {
                slabs.insert(slice.buffer.id(), slice.buffer);
            }
        }
    }
    let layout = pipeline_cache.get_bind_group_layout(&mesh_layout());
    for (view, marker) in &views {
        let Some(inputs) = hiz.mesh_inputs(&view.retained_view_entity) else {
            continue;
        };
        let state = states
            .entry(view.retained_view_entity)
            .or_insert_with(|| MeshletView {
                primary: marker.primary,
                drawn: None,
                bits_for: 0,
                stats: GpuCounters::new(
                    &device,
                    "static_gx_meshlet_stats",
                    "static_gx_meshlet_readback",
                ),
                base: None,
                groups: HashMap::default(),
                seen: 0,
            });
        state.seen = now;
        if state.drawn.is_none() || state.bits_for < m.capacity {
            state.drawn = Some(device.create_buffer(&BufferDescriptor {
                label: Some("static_gx_meshlet_drawn"),
                size: u64::from(m.capacity.div_ceil(32).max(1)) * 4,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            state.bits_for = m.capacity;
        }
        let Some(drawn) = state.drawn.as_ref() else {
            continue;
        };
        let base = (
            [
                m.table.id(),
                m.vertices.id(),
                m.triangles.id(),
                c.cand.id(),
                mesh.meta.id(),
                mesh.refs.id(),
                drawn.id(),
            ],
            inputs.generation,
        );
        if state.base != Some(base) {
            state.groups.clear();
            state.base = Some(base);
        }
        state.groups.retain(|slab, _| slabs.contains_key(slab));
        for (&id, slab) in &slabs {
            if state.groups.contains_key(&id) {
                continue;
            }
            let bind = device.create_bind_group(
                "static_gx_mesh",
                &layout,
                &BindGroupEntries::sequential((
                    inputs.uniform.as_entire_binding(),
                    m.table.as_entire_binding(),
                    m.vertices.as_entire_binding(),
                    m.triangles.as_entire_binding(),
                    slab.as_entire_binding(),
                    c.cand.as_entire_binding(),
                    mesh.meta.as_entire_binding(),
                    mesh.refs.as_entire_binding(),
                    drawn.as_entire_binding(),
                    inputs.pyramid,
                    state.stats.buffer().as_entire_binding(),
                )),
            );
            state.groups.insert(id, bind);
        }
    }
    states.retain(|_, s| now.wrapping_sub(s.seen) <= KEEP_FRAMES);
    if crate::static_gx::gx_perf_enabled() {
        for state in states.values_mut() {
            state.stats.poll();
        }
        if now.is_multiple_of(64) {
            for state in states.values_mut() {
                let [frames, examined, early, late] = state.stats.take();
                let per = |n: u64| n / frames.max(1);
                println!(
                    "GX_MESHLET {} per frame: meshlets={} early={} late={} culled={} ({frames} \
                     frames read back)",
                    if state.primary { "world" } else { "mirror" },
                    per(examined),
                    per(early),
                    per(late),
                    per(examined.saturating_sub(early + late)),
                );
            }
        }
    }
}

/// This frame's meshlet draws of one view: present only where its whole culled batch draws as
/// meshlets (the module header, "When it runs").
pub(super) struct GxMeshletPass<'w> {
    colour: [Option<&'w RenderPipeline>; 4],
    prepass: [Option<&'w RenderPipeline>; 4],
    view: &'w MeshletView,
    stats: bool,
}

impl<'w> GxMeshletPass<'w> {
    /// The view's meshlet pass, after the cull took the batch (`GxCullPass::bind`): the cull ran
    /// its mesh variant, every slab of the batch has a mesh group and every mesh pipeline of
    /// `colour` and `prepass` is compiled.
    pub(super) fn new(
        world: &'w World,
        view: &ExtractedView,
        batch: &GxBatch<'_>,
        cull: &GxCullPass<'_>,
        colour: &HashMap<(bool, bool), CachedRenderPipelineId>,
        prepass: &HashMap<(bool, bool), CachedRenderPipelineId>,
    ) -> Option<Self> {
        if !(cull.meshlets() && batch.meshlets()) || colour.is_empty() || prepass.is_empty() {
            return None;
        }
        let lane = world.get_resource::<GxMeshletLane>()?;
        let state = lane.views.get(&view.retained_view_entity)?;
        if batch.slabs().any(|slab| !state.groups.contains_key(&slab)) {
            return None;
        }
        let cache = world.resource::<PipelineCache>();
        Some(Self {
            colour: super::pipeline_table(cache, colour)?,
            prepass: super::pipeline_table(cache, prepass)?,
            view: state,
            stats: crate::static_gx::gx_perf_enabled(),
        })
    }

    /// Zero the view's counters: in the prepass job, after the early cull, before its pass.
    pub(super) fn begin(&self, rec: &mut Recorder<'_>) {
        if self.stats {
            self.view.stats.clear(rec.command_encoder());
        }
    }

    /// Read the counters back: in the prepass job, after the late prepass.
    pub(super) fn end(&self, rec: &mut Recorder<'_>) {
        if self.stats {
            self.view.stats.copy(rec.command_encoder());
        }
    }

    /// Draw `batch` as meshlets in `mode` ([`MODE_EARLY`] and [`MODE_LATE`] through the prepass
    /// twins, [`MODE_COLOUR`] through the colour pipelines), the mesh group at `mesh_group`.
    pub(super) fn emit<'b, 'p>(
        &self,
        batch: &GxBatch<'b>,
        pass: &mut TrackedRenderPass<'p>,
        mode: u32,
        mesh_group: usize,
    ) where
        'w: 'p,
        'b: 'p,
    {
        let view: &'w MeshletView = self.view;
        let pipes = if mode == MODE_COLOUR {
            &self.colour
        } else {
            &self.prepass
        };
        batch.emit_mesh(
            pass,
            pipes,
            mesh_group,
            |slab| view.groups.get(&slab),
            mode,
            0,
        );
    }
}

/// Wire the lane (called from `render::build`, native only).
pub(super) fn build(app: &mut App) {
    if let Some(mut shaders) = app.world_mut().get_resource_mut::<Assets<Shader>>() {
        // A UUID id never fails to insert.
        let _ = shaders.insert(
            MESH_SHADER.id(),
            Shader::from_wgsl(MESH_SOURCE, "benilla_world/shaders/static_gx_mesh.wgsl"),
        );
    }
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    // The mesh stage reads the vertex slabs as storage. The allocator takes its usages when it is
    // made, in its plugin's `finish`, before the device is known, so every slab carries it
    // wherever the switch leaves the lane possible.
    if switched_on() {
        render_app
            .world_mut()
            .get_resource_or_init::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::STORAGE;
    }
    render_app
        .add_systems(
            RenderStartup,
            init_meshlets.after(super::batch::init_indirect),
        )
        .add_systems(
            Render,
            prepare_meshlets
                .in_set(RenderSystems::PrepareBindGroups)
                .after(super::gpu_cull::prepare_gpu_cull)
                .run_if(resource_exists::<GxMeshletLane>),
        );
}
