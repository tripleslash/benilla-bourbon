//! fork: **the ray-traced scene** — a BLAS per static_gx region and per terrain cell, one TLAS over
//! them, and the bind group the ray-traced lanes (probe GI, shadows, water reflections) trace
//! through with `benilla_modern::rt_scene` (`shaders/modern/rt_scene.wgsl`). Native only.
//!
//! A child of `static_gx::render` because its scene is this pass's: the regions the GPU cache has
//! assembled, read through their blocks of the bindless region table and index arena
//! (`bindless.rs`), plus the resident terrain cells (`modern::shadow::ModernTerrainCasters`).
//!
//! # The tier: capable, enabled, live
//!
//! Three levels, each implying the one before ([`RtTier`]):
//!
//! - **capable** ([`capable`]): [`HwCaps::ray_query`] holds, static_gx draws bindless
//!   (`bindless::enabled`), and `WOW_RT=0` did not turn it off. Decided once at render startup;
//!   [`RtScene`] exists exactly where it holds, and every pipeline that can trace declares the
//!   scene for the whole session, so nothing re-specializes on a switch.
//! - **enabled** ([`enabled`]): capable, and the `rayTracing` option
//!   ([`crate::modern::RayTracingSetting`], extracted every frame) is on. Read each frame. While it
//!   holds, the scene is built and kept; when it drops, every BLAS, the TLAS, the arenas and the
//!   scratch are freed ([`RtScene::free_all`]).
//! - **live** ([`live`]): enabled, and the scene is ready ([`RtScene::update_ready`]): every region
//!   static_gx holds and every resident terrain cell has its BLAS, nothing the camera has not drawn
//!   is still waiting to be assembled, and the TLAS holds them, for [`READY_FRAMES`] frames in a
//!   row. Once live, streaming does not drop it: only an empty scene, or more entries waiting than
//!   built, sends the lanes back to raster. The ray-traced lanes (`modern::gi::rt_gi`,
//!   `modern::shadow::rt_shadows`, `modern::volumetrics::rt_froxel`) key on it, per frame.
//!
//! The three are decided once a frame, in `PrepareAssets` ([`update_rt_tier`]), before anything
//! reads them, so every pass of a frame agrees. A pass that declares the scene binds
//! [`RtScene::lane_bind`]: the scene while its lane is live, else the stand-in, whose marked
//! instance row tells the WGSL library it is not (`rt_scene::live`), so the receivers take their
//! raster path in the same pipeline.
//!
//! The scene is kept only in a modern-lighting frame: a classic frame drops it, so the classic path
//! pays nothing. While enabled, the regions the camera has not drawn are assembled too, a few a
//! frame ([`assemble_unseen`]), so the scene holds every streamed region, not only those seen so
//! far.
//!
//! # What is built, and when
//!
//! Every frame [`prepare_rt_scene`] diffs the regions and cells against the scene: a new or changed
//! one takes a slot and waits; one that left frees its BLAS, its blocks and its slot. Waiting
//! entries are built nearest the camera first, up to `WOW_RT_BUDGET` triangles a frame (default
//! [`DEFAULT_BUDGET`]). A build is, in [`rt_scene_pass`]: the entry's vertices copied raw out of the
//! mesh slab into a scratch buffer (the slabs are `COPY_SRC`, not `BLAS_INPUT` or `STORAGE`), a
//! terrain cell's indices copied into the terrain index arena, the attribute repack
//! (`rt_repack.wgsl`) into the attribute arena, then the BLAS built from the scratch copy. The TLAS
//! is rebuilt in a frame where an instance came or went, and kept otherwise: nothing in it moves.
//!
//! A static region's indices sort opaque items before cutout ones (`assemble_region`), so its BLAS
//! has up to two geometries: the opaque items, `OPAQUE` (a ray never stops to ask), then the cutout
//! items, non-opaque (each candidate is alpha-tested by the library). A region with a hidden item
//! (an exile kill or a texture not yet resident: record bit 0) makes both non-opaque, so the test
//! lets those items through, and a kill revision rebuilds the region. Terrain is opaque.
//!
//! # The instances
//!
//! An entry's slot is its TLAS instance index and its custom index (24 bits). Its mask is
//! `1 << kind` (cell 0, WMO 1, prop 2, terrain 3); its transform is the region's origin (terrain
//! is in world space already). The instance table holds two rows a slot (the WGSL header has
//! them): kind, table row, first attribute row, first index; the geometries' first triangles, the
//! cutout geometry, the vertex count.
//!
//! # The bind group ([`layout`], [`RtScene::bind_group`])
//!
//! | binding | resource |
//! |---|---|
//! | 0 | the TLAS |
//! | 1 | the instance table, `array<vec4<u32>>` |
//! | 2 | static_gx's region table (`GxBindless`) |
//! | 3 | static_gx's index arena |
//! | 4 | the terrain index arena |
//! | 5 | the attribute arena, [`ATTRIBUTE_WORDS`] `u32` a vertex |
//! | 6 | static_gx's texture classes, `binding_array<texture_2d_array<f32>, 1024>` |
//! | 7, 8 | static_gx's repeat and clamp samplers |
//!
//! Visible to the compute and fragment stages. The group index is the importer's
//! ([`shader_defs`]).

use std::num::NonZero;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use bevy::app::SubApp;
use bevy::mesh::MeshVertexAttributeId;
use bevy::render::mesh::RenderMeshBufferInfo;
use bevy::render::render_resource::binding_types::{
    acceleration_structure, storage_buffer_sized, uniform_buffer,
};

use super::bindless::{GrowBuffer, RangeAlloc, GX_CLASS_SLOTS};
use super::*;
use crate::modern::shadow::ModernTerrainCasters;
use crate::static_gx::ATTRIBUTE_GX_WORD;

/// `u32` words of a vertex's attribute row. Keep in sync with `rt_scene.wgsl` and `rt_repack.wgsl`.
const ATTRIBUTE_WORDS: u32 = 5;
/// Instance-table rows per slot. Keep in sync with `rt_scene.wgsl`.
const INSTANCE_ROWS: usize = 2;
/// An attribute word offset the repack reads as "the mesh has none".
const NONE: u32 = u32::MAX;
/// Instance kinds; the TLAS mask bit is `1 << kind`. Keep in sync with `rt_scene.wgsl`.
const KIND_CELL: u32 = 0;
const KIND_WMO: u32 = 1;
const KIND_PROP: u32 = 2;
const KIND_TERRAIN: u32 = 3;
/// Instance row 1 z: no geometry is the cutout one.
const NO_GEOMETRY: u32 = 2;
/// Triangles built a frame unless `WOW_RT_BUDGET` says otherwise.
const DEFAULT_BUDGET: u32 = 1 << 18;
/// Builds a frame at most, whatever their size.
const MAX_BUILDS: usize = 64;
/// Regions the camera has not drawn, assembled a frame ([`assemble_unseen`]).
const UNSEEN_BUDGET: u32 = 4;
/// The arenas' floors: attribute rows (5 MiB), terrain indices (4 MiB), scratch bytes.
const MIN_ATTRIBUTES: u32 = 1 << 18;
const MIN_TERRAIN_INDICES: u32 = 1 << 20;
const MIN_SCRATCH: u64 = 16 << 20;
/// The TLAS's floor, in instances.
const MIN_INSTANCES: u32 = 1024;
/// A custom index has 24 bits.
const MAX_SLOTS: u32 = 1 << 24;
/// A repack workgroup's threads, and the most workgroups in a dispatch row. Keep in sync with
/// `rt_repack.wgsl`.
const REPACK_GROUP: u32 = 64;
const REPACK_ROW: u32 = 65535;

/// Frames in a row the scene must be complete before the lanes go live (the hysteresis's rising
/// edge; the falling one is [`RtScene::update_ready`]'s loose test).
const READY_FRAMES: u32 = 8;
/// Frames in a row with every entry built but regions still unassembled after which the scene
/// counts as ready anyway: a region whose mesh never reaches the slab must not hold the lanes off.
const READY_GRACE: u32 = 300;
/// The stand-in's instance rows (`rt_scene.wgsl`'s `STUB_MARK`): a kind no entry has, which the
/// library reads as "this is not the scene" (`rt_scene::live`). Keep in sync.
const STUB_MARK: u32 = u32::MAX;

/// The levels' decisions, for readers in any world ([`capable`], [`enabled`], [`live`]); written once
/// a frame by [`update_rt_tier`] (and `CAPABLE` once at startup).
static CAPABLE: AtomicBool = AtomicBool::new(false);
static ENABLED: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicBool = AtomicBool::new(false);
/// Regions the camera has not drawn still waiting to be assembled, as [`assemble_unseen`] left
/// them this frame: part of the scene's readiness.
static UNSEEN_BACKLOG: AtomicU32 = AtomicU32::new(0);

/// Whether this session can ray-trace at all (the module header's first level). Fixed once render
/// startup ran; false before.
pub(crate) fn capable() -> bool {
    CAPABLE.load(Ordering::Relaxed)
}

/// Capable, and the `rayTracing` option on this frame: the scene is (being) built.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Enabled, and the scene ready this frame: the ray-traced lanes trace. Per frame; outside the
/// render world, the render world's latest decision.
pub(crate) fn live() -> bool {
    LIVE.load(Ordering::Relaxed)
}

/// The device can (ray queries, and static_gx bindless, whose table and arena the scene reads),
/// and `WOW_RT=0` did not turn it off. Never on wasm32 (the module is native only).
fn capable_on(caps: Option<&HwCaps>) -> bool {
    caps.is_some_and(|c| c.ray_query)
        && bindless::enabled(caps)
        && crate::launch_env::var("WOW_RT").as_deref() != Some("0")
}

/// The ray-traced tier's three levels (the module header), in the render world from render
/// startup on: `capable` fixed, `enabled` and `live` decided each frame by [`update_rt_tier`].
/// [`RtScene`] is present exactly when `capable`.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct RtTier {
    pub capable: bool,
    pub enabled: bool,
    pub live: bool,
    /// Render frames since startup: the switch lines' clock.
    pub frame: u32,
    /// When the lanes started waiting for a ready scene (the option came on, or a live scene fell
    /// back), as (instant, frame): what the line at the next going-live measures.
    waiting_since: Option<(Instant, u32)>,
}

/// The scene's bind group layout (the module header has the table).
pub(crate) fn layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "rt_scene_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                acceleration_structure(),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_2d_array(TextureSampleType::Float { filterable: true })
                    .count(NonZero::new(GX_CLASS_SLOTS).unwrap()),
                sampler(SamplerBindingType::Filtering),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    )
}

/// The defs a pipeline importing `benilla_modern::rt_scene` passes: the bind group its bindings
/// sit in.
#[allow(dead_code)] // the ray-traced lanes' interface
pub(crate) fn shader_defs(group: u32) -> Vec<ShaderDefVal> {
    vec![ShaderDefVal::UInt("RT_GROUP".into(), group)]
}

/// One repack dispatch, `rt_repack.wgsl`'s `Job`.
#[derive(Clone, Copy, ShaderType)]
struct RepackJob {
    src_word: u32,
    stride: u32,
    count: u32,
    dst: u32,
    normal: u32,
    uv: u32,
    word: u32,
    color: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum RtKey {
    Cell(i32, i32),
    Wmo(Entity),
    Prop(Entity),
    Terrain(AssetId<Mesh>),
}

/// What an entry is built from; any change rebuilds it.
#[derive(Clone, Copy, PartialEq)]
struct Source {
    kind: u32,
    mesh: AssetId<Mesh>,
    /// The vertices in the mesh's slab: first element, count.
    vertices: (u32, u32),
    /// Static: the region's block of the index arena; terrain: the mesh's index slab range.
    indices: (u32, u32),
    /// Static: the region's table row; terrain: 0.
    row: u32,
    /// Static: the kill revision; terrain: the LOD epoch.
    rev: u32,
    /// The instance's translation: the region's origin, or zero.
    translation: Vec3,
}

/// Word offsets of the attributes the repack reads, and the stride.
#[derive(Clone, Copy, PartialEq, Eq)]
struct VertexLayout {
    stride: u64,
    normal: u32,
    uv: u32,
    word: u32,
    color: u32,
}

struct RtEntry {
    source: Source,
    layout: VertexLayout,
    slot: u32,
    center: Vec3,
    /// Indices before the first cutout item (the count when there is none).
    split: u32,
    /// An item is hidden (record bit 0): both geometries non-opaque.
    hidden: bool,
    /// The BLAS, once built (or building this frame); `None` while waiting.
    blas: Option<Blas>,
    /// The first attribute row of its block (`source.vertices.1` rows), once planned.
    attributes: Option<u32>,
    /// Terrain: the first index of its block of the terrain arena, once planned.
    terrain_indices: Option<u32>,
    /// The frame this entry's source was last seen.
    seen: u32,
}

/// One BLAS build, planned in [`prepare_rt_scene`] and recorded by [`rt_scene_pass`].
struct RtBuild {
    key: RtKey,
    blas: Blas,
    /// Each geometry's size and first index.
    geometries: Vec<(BlasTriangleGeometrySizeDescriptor, u32)>,
    /// The index buffer the geometries read: the gx arena or the terrain arena.
    index: Buffer,
    stride: u64,
    /// The entry's first vertex in the scratch buffer.
    scratch_vertex: u32,
    /// The raw vertex copy into the scratch buffer.
    vertex_copy: BufferCopy,
    /// Terrain: the index copy into `index`.
    index_copy: Option<BufferCopy>,
    repack_offset: u32,
    repack_groups: (u32, u32),
}

/// A copy out of a mesh slab: the slab, its byte offset, the bytes, the destination byte offset.
struct BufferCopy {
    src: Buffer,
    at: u64,
    bytes: u64,
    dst: u64,
}

/// A waiting entry picked for this frame: its scratch byte offset and its slabs' buffers and byte
/// offsets (vertex; terrain index).
struct Pick {
    key: RtKey,
    scratch_at: u64,
    vertex: (Buffer, u64),
    index: Option<(Buffer, u64)>,
}

/// This frame's work for [`rt_scene_pass`].
#[derive(Default)]
struct RtFrame {
    builds: Vec<RtBuild>,
    repack_bind: Option<BindGroup>,
    build_tlas: bool,
    /// Recorded already: the pass runs in the first view of the frame that reaches it.
    recorded: AtomicBool,
}

/// The `(TLAS, buffers, pool classes)` the bind group was built over.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BindKey {
    tlas: u32,
    instances: BufferId,
    table: BufferId,
    arena: BufferId,
    terrain: BufferId,
    attributes: BufferId,
    classes: u32,
}

/// The ray-traced scene. Present in the render world exactly while [`RtTier::capable`]; holds
/// entries, structures and arenas only while [`RtTier::enabled`].
#[derive(Resource)]
pub(crate) struct RtScene {
    layout: BindGroupLayoutDescriptor,
    repack_layout: BindGroupLayoutDescriptor,
    repack: CachedComputePipelineId,
    /// Triangles built a frame.
    budget: u32,
    entries: HashMap<RtKey, RtEntry>,
    free_slots: Vec<u32>,
    next_slot: u32,
    tlas: Tlas,
    /// Bumped whenever `tlas` is replaced by a larger one.
    tlas_generation: u32,
    /// Entries whose instance is in the TLAS.
    built: u32,
    /// CPU copy of the instance table, [`INSTANCE_ROWS`] a slot.
    instances: Vec<[u32; 4]>,
    instance_buffer: Option<Buffer>,
    dirty_slots: Vec<u32>,
    attributes: GrowBuffer,
    attribute_alloc: RangeAlloc,
    terrain_indices: GrowBuffer,
    terrain_alloc: RangeAlloc,
    scratch: Option<Buffer>,
    scratch_bytes: u64,
    repack_jobs: DynamicUniformBuffer<RepackJob>,
    frame: RtFrame,
    frame_no: u32,
    bind: Option<BindGroup>,
    bound: Option<BindKey>,
    /// Entries were waiting at the last frame's end: the settled line is due when none are.
    was_pending: bool,
    /// fork: F — the stand-in bind group ([`Self::lane_bind`]) over an empty TLAS, built
    /// once, and what it holds (kept beside it, never read).
    stub: Option<BindGroup>,
    #[allow(dead_code)]
    stub_parts: Option<(Tlas, Buffer)>,
    /// The scene is ready for the lanes ([`Self::update_ready`]): what makes [`RtTier::live`].
    ready: bool,
    /// Frames in a row every entry has been built ([`Self::update_ready`]'s rising edge).
    complete_frames: u32,
}

/// The hit attributes' arena, empty.
fn attribute_arena() -> GrowBuffer {
    GrowBuffer::new(
        "rt_attributes",
        BufferUsages::STORAGE,
        u64::from(ATTRIBUTE_WORDS) * 4,
        MIN_ATTRIBUTES,
    )
}

/// The terrain cells' index arena, empty.
fn terrain_arena() -> GrowBuffer {
    GrowBuffer::new(
        "rt_terrain_indices",
        BufferUsages::STORAGE | BufferUsages::BLAS_INPUT,
        4,
        MIN_TERRAIN_INDICES,
    )
}

fn create_tlas(device: &RenderDevice, max_instances: u32) -> Tlas {
    device.wgpu_device().create_tlas(&CreateTlasDescriptor {
        label: Some("rt_tlas"),
        max_instances,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    })
}

impl RtScene {
    fn new(
        device: &RenderDevice,
        repack_layout: BindGroupLayoutDescriptor,
        repack: CachedComputePipelineId,
        budget: u32,
    ) -> Self {
        Self {
            layout: layout(),
            repack_layout,
            repack,
            budget,
            entries: HashMap::default(),
            free_slots: Vec::new(),
            next_slot: 0,
            // One instance until the first build grows it (`ensure_tlas`): a session that starts
            // with the option off holds no TLAS memory to speak of.
            tlas: create_tlas(device, 1),
            tlas_generation: 0,
            built: 0,
            instances: Vec::new(),
            instance_buffer: None,
            dirty_slots: Vec::new(),
            attributes: attribute_arena(),
            attribute_alloc: RangeAlloc::default(),
            terrain_indices: terrain_arena(),
            terrain_alloc: RangeAlloc::default(),
            scratch: None,
            scratch_bytes: 0,
            repack_jobs: DynamicUniformBuffer::default(),
            frame: RtFrame::default(),
            frame_no: 0,
            bind: None,
            bound: None,
            was_pending: false,
            stub: None,
            stub_parts: None,
            ready: false,
            complete_frames: 0,
        }
    }

    /// The scene's bind group ([`layout`]), once the TLAS holds an instance. [`rt_scene_pass`]
    /// (re)builds that TLAS in the same frame, so a pass binding it orders after that pass. A lane
    /// that may only trace a ready scene takes [`Self::lane_bind`] instead.
    #[allow(dead_code)] // the ray-traced lanes' interface
    pub(crate) fn bind_group(&self) -> Option<&BindGroup> {
        self.bind.as_ref()
    }

    /// The group a pipeline that declares the scene for the whole session binds this frame: the
    /// scene while its lane is `live` (and the scene's group exists), else the stand-in — an empty
    /// TLAS every ray misses, one-row buffers holding [`STUB_MARK`] (so `rt_scene::live()` is false
    /// in the shader and the receivers take their raster path) and the pool's white class in every
    /// texture slot. Binding never changes a pipeline, so a switch costs no specialization. `None`
    /// only before the first [`prepare_rt_scene`].
    pub(crate) fn lane_bind(&self, live: bool) -> Option<&BindGroup> {
        if live {
            self.bind.as_ref().or(self.stub.as_ref())
        } else {
            self.stub.as_ref()
        }
    }

    /// fork: F — build the stand-in once. Its TLAS is built here, empty, and submitted at once, so
    /// any pass that binds it later is recorded after its build (wgpu refuses an unbuilt TLAS). Its
    /// rows hold [`STUB_MARK`]: the library's `live()` reads instance row 0.
    fn ensure_stub(
        &mut self,
        pool: &mut GxTexturePool,
        pipes: &GxPipelines,
        pipeline_cache: &PipelineCache,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        if self.stub.is_some() {
            return;
        }
        let tlas = create_tlas(device, 1);
        // One row is all a shader's runtime-sized arrays need to bind; no ray hits, so only
        // `rt_scene::live()` reads it: instance row 0's kind, the mark.
        let rows = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("rt_stub_rows"),
            contents: bytemuck::cast_slice(&[STUB_MARK; 16]),
            usage: BufferUsages::STORAGE,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("rt_stub_tlas"),
        });
        encoder.build_acceleration_structures(
            std::iter::empty::<&BlasBuildEntry<'_>>(),
            std::iter::once(&tlas),
        );
        queue.submit([encoder.finish()]);
        let white = pool.white(device, queue);
        let white_view: &WgpuTextureView = &**pool.view(white);
        let views: Vec<&WgpuTextureView> = (0..GX_CLASS_SLOTS).map(|_| white_view).collect();
        let layout = pipeline_cache.get_bind_group_layout(&self.layout);
        self.stub = Some(device.create_bind_group(
            "rt_scene_stub",
            &layout,
            &BindGroupEntries::sequential((
                tlas.as_binding(),
                rows.as_entire_binding(),
                rows.as_entire_binding(),
                rows.as_entire_binding(),
                rows.as_entire_binding(),
                rows.as_entire_binding(),
                &views[..],
                &pipes.sampler,
                &pipes.sampler_clamp,
            )),
        ));
        self.stub_parts = Some((tlas, rows));
    }

    /// Entries waiting for their BLAS; the scene holds every resident region and cell at 0.
    #[allow(dead_code)] // the ray-traced lanes' interface
    pub(crate) fn pending(&self) -> usize {
        self.entries.values().filter(|e| e.blas.is_none()).count()
    }

    /// Drop every entry: a classic frame, or a map change. The buffers and the TLAS stay. The lanes
    /// go back to raster until the scene is ready again.
    fn clear(&mut self) {
        self.ready = false;
        self.complete_frames = 0;
        if self.entries.is_empty() && self.built == 0 {
            return;
        }
        for e in self.entries.values().filter(|e| e.blas.is_some()) {
            if let Some(instance) = self.tlas.get_mut_single(e.slot as usize) {
                *instance = None;
            }
        }
        self.entries.clear();
        self.free_slots.clear();
        self.next_slot = 0;
        self.built = 0;
        self.attribute_alloc = RangeAlloc::default();
        self.terrain_alloc = RangeAlloc::default();
        self.dirty_slots.clear();
        self.bind = None;
        self.bound = None;
        self.was_pending = false;
    }

    /// The `rayTracing` option is off: drop every entry and free the memory the scene holds — every
    /// BLAS (with its entry and its TLAS instance), the TLAS (replaced by a one-instance one, grown
    /// again on the first build), the instance table, the attribute and terrain-index arenas, the
    /// scratch and the repack jobs. Only the stand-in stays. Dropping a handle while a submitted
    /// frame still uses the object is safe: wgpu-core keeps it alive until that submission is done.
    fn free_all(&mut self, device: &RenderDevice) {
        self.clear();
        let holds = self.tlas.get().len() > 1
            || self.instance_buffer.is_some()
            || self.scratch.is_some()
            || self.attributes.buffer().is_some()
            || self.terrain_indices.buffer().is_some();
        if !holds {
            return;
        }
        let freed = self.attributes.bytes() + self.scratch_bytes;
        self.tlas = create_tlas(device, 1);
        self.tlas_generation += 1;
        self.instances = Vec::new();
        self.instance_buffer = None;
        self.attributes = attribute_arena();
        self.terrain_indices = terrain_arena();
        self.scratch = None;
        self.scratch_bytes = 0;
        self.repack_jobs = DynamicUniformBuffer::default();
        self.frame = RtFrame::default();
        info!(
            "rt_scene: freed every BLAS, the TLAS, the arenas and the scratch ({:.1} MiB of \
             attributes and scratch besides the structures)",
            freed as f64 / (1024.0 * 1024.0)
        );
    }

    /// The scene's readiness for the lanes, once a frame after the plan, with hysteresis. Rising:
    /// no entry waits for its BLAS, no region the camera has not drawn waits to be assembled
    /// ([`UNSEEN_BACKLOG`]; after [`READY_GRACE`] frames of nothing else missing it is let go), the
    /// TLAS holds an instance and its bind group exists — for [`READY_FRAMES`] frames in a row, so
    /// the builds the last plan made have been recorded. Falling (while ready): the scene is empty,
    /// or more entries wait than are built (a teleport, a burst far larger than the scene). A cell
    /// streaming in while ready only waits its turn: the lanes stay on rays, as the far geometry a
    /// ray misses for a frame or two is what the raster path's own streaming shows too.
    fn update_ready(&mut self) {
        let pending = self.entries.values().filter(|e| e.blas.is_none()).count();
        let built = self.built as usize;
        if self.ready {
            if built == 0 || self.bind.is_none() || pending > built {
                self.ready = false;
                self.complete_frames = 0;
            }
            return;
        }
        if pending > 0 || built == 0 || self.bind.is_none() {
            self.complete_frames = 0;
            return;
        }
        self.complete_frames = self.complete_frames.saturating_add(1);
        let assembled = UNSEEN_BACKLOG.load(Ordering::Relaxed) == 0;
        self.ready = (assembled && self.complete_frames >= READY_FRAMES)
            || self.complete_frames >= READY_GRACE;
    }

    /// Keep `key` if its source and layout stand; else free the old entry and add a waiting one,
    /// returned so the caller can fill its geometry split. `None` also when the slots ran out.
    fn visit(
        &mut self,
        key: RtKey,
        source: Source,
        layout: VertexLayout,
        center: Vec3,
        now: u32,
    ) -> Option<&mut RtEntry> {
        if let Some(entry) = self.entries.get_mut(&key) {
            if entry.source == source && entry.layout == layout {
                entry.seen = now;
                return None;
            }
        }
        if let Some(old) = self.entries.remove(&key) {
            self.release(old);
        }
        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None if self.next_slot < MAX_SLOTS => {
                self.next_slot += 1;
                self.next_slot - 1
            }
            None => return None,
        };
        self.entries.insert(
            key,
            RtEntry {
                source,
                layout,
                slot,
                center,
                split: source.indices.1,
                hidden: false,
                blas: None,
                attributes: None,
                terrain_indices: None,
                seen: now,
            },
        );
        self.entries.get_mut(&key)
    }

    /// Free an entry's instance, blocks and slot.
    fn release(&mut self, entry: RtEntry) {
        if entry.blas.is_some() {
            if let Some(instance) = self.tlas.get_mut_single(entry.slot as usize) {
                *instance = None;
            }
            self.built -= 1;
            self.frame.build_tlas = true;
        }
        if let Some(at) = entry.attributes {
            self.attribute_alloc.free(at, entry.source.vertices.1);
        }
        if let Some(at) = entry.terrain_indices {
            self.terrain_alloc.free(at, entry.source.indices.1);
        }
        self.free_slots.push(entry.slot);
    }

    /// The last frame's work never reached an encoder (no static_gx view rendered): its builds
    /// wait again, and its TLAS build carries over.
    fn requeue(&mut self, last: RtFrame) {
        self.frame.build_tlas |= last.build_tlas;
        for build in last.builds {
            let Some(e) = self.entries.get_mut(&build.key) else {
                continue;
            };
            if e.blas.as_ref() != Some(&build.blas) {
                continue;
            }
            e.blas = None;
            let (slot, vertices, indices) = (e.slot, e.source.vertices.1, e.source.indices.1);
            if let Some(at) = e.attributes.take() {
                self.attribute_alloc.free(at, vertices);
            }
            if let Some(at) = e.terrain_indices.take() {
                self.terrain_alloc.free(at, indices);
            }
            if let Some(instance) = self.tlas.get_mut_single(slot as usize) {
                *instance = None;
            }
            self.built -= 1;
            self.frame.build_tlas = true;
        }
    }

    /// Release every entry not seen this frame.
    fn sweep(&mut self, now: u32) {
        let gone: Vec<RtKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.seen != now)
            .map(|(k, _)| *k)
            .collect();
        for key in gone {
            if let Some(entry) = self.entries.remove(&key) {
                self.release(entry);
            }
        }
    }

    /// Hold instance `slot`, growing the TLAS into a larger one if it cannot.
    fn ensure_tlas(&mut self, slot: u32, device: &RenderDevice) {
        if (slot as usize) < self.tlas.get().len() {
            return;
        }
        let mut tlas = create_tlas(device, (slot + 1).next_power_of_two().max(MIN_INSTANCES));
        for (i, instance) in self.tlas.get().iter().enumerate() {
            if instance.is_some() {
                tlas[i] = instance.clone();
            }
        }
        self.tlas = tlas;
        self.tlas_generation += 1;
        self.frame.build_tlas = true;
    }

    fn write_instance(&mut self, slot: u32, rows: [[u32; 4]; INSTANCE_ROWS]) {
        let at = slot as usize * INSTANCE_ROWS;
        if self.instances.len() < at + INSTANCE_ROWS {
            self.instances.resize(at + INSTANCE_ROWS, [0; 4]);
        }
        self.instances[at..at + INSTANCE_ROWS].copy_from_slice(&rows);
        self.dirty_slots.push(slot);
    }

    /// Upload the instance rows written this frame; a table that outgrew its buffer goes up whole
    /// into a larger one.
    fn upload_instances(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        let bytes = (self.instances.len() * 16) as u64;
        if bytes == 0 {
            return;
        }
        if self
            .instance_buffer
            .as_ref()
            .is_none_or(|b| b.size() < bytes)
        {
            let rows = self
                .instances
                .len()
                .next_power_of_two()
                .max(MIN_INSTANCES as usize * INSTANCE_ROWS);
            self.instances.resize(rows, [0; 4]);
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("rt_instances"),
                size: (rows * 16) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.instances));
            self.instance_buffer = Some(buffer);
            self.dirty_slots.clear();
            return;
        }
        let Some(buffer) = &self.instance_buffer else {
            return;
        };
        for slot in std::mem::take(&mut self.dirty_slots) {
            let at = slot as usize * INSTANCE_ROWS;
            queue.write_buffer(
                buffer,
                (at * 16) as u64,
                bytemuck::cast_slice(&self.instances[at..at + INSTANCE_ROWS]),
            );
        }
    }

    /// Pick the waiting entries this frame builds, nearest `camera` first within the budget,
    /// place their blocks, create their BLASes and instances, and leave the work in `frame`.
    #[allow(clippy::too_many_arguments)]
    fn plan(
        &mut self,
        camera: Vec3,
        arena: &Buffer,
        allocator: &MeshAllocator,
        pipeline_cache: &PipelineCache,
        device: &RenderDevice,
        queue: &RenderQueue,
        copies: &mut CommandEncoder,
    ) {
        let mut waiting: Vec<(f32, RtKey)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.blas.is_none())
            .map(|(k, e)| (e.center.distance_squared(camera), *k))
            .collect();
        if waiting.is_empty() {
            return;
        }
        waiting.sort_by(|a, b| a.0.total_cmp(&b.0));

        // The picks, each with its scratch offset and its copies' sources.
        let mut picks: Vec<Pick> = Vec::new();
        let (mut triangles, mut cursor) = (0u32, 0u64);
        for (_, key) in waiting {
            if picks.len() >= MAX_BUILDS {
                break;
            }
            let Some(e) = self.entries.get_mut(&key) else {
                continue;
            };
            let tris = e.source.indices.1 / 3;
            if !picks.is_empty() && triangles.saturating_add(tris) > self.budget {
                break;
            }
            let Some(vslice) = allocator.mesh_vertex_slice(&e.source.mesh) else {
                continue;
            };
            let index = if e.source.kind == KIND_TERRAIN {
                let Some(islice) = allocator.mesh_index_slice(&e.source.mesh) else {
                    continue;
                };
                Some((islice.buffer.clone(), u64::from(islice.range.start) * 4))
            } else {
                None
            };
            let stride = e.layout.stride;
            cursor = cursor.next_multiple_of(stride);
            let at = cursor;
            cursor += u64::from(e.source.vertices.1) * stride;
            e.attributes = Some(self.attribute_alloc.alloc(e.source.vertices.1));
            if index.is_some() {
                e.terrain_indices = Some(self.terrain_alloc.alloc(e.source.indices.1));
            }
            triangles = triangles.saturating_add(tris);
            let vertex = (
                vslice.buffer.clone(),
                u64::from(vslice.range.start) * stride,
            );
            picks.push(Pick {
                key,
                scratch_at: at,
                vertex,
                index,
            });
        }
        if picks.is_empty() {
            return;
        }

        // The arenas hold every block now, and the scratch every pick.
        self.attributes
            .reserve(self.attribute_alloc.end(), device, queue, copies);
        self.terrain_indices
            .reserve(self.terrain_alloc.end(), device, queue, copies);
        if cursor > self.scratch_bytes {
            self.scratch_bytes = cursor.next_power_of_two().max(MIN_SCRATCH);
            self.scratch = Some(device.create_buffer(&BufferDescriptor {
                label: Some("rt_scratch"),
                size: self.scratch_bytes,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::BLAS_INPUT,
                mapped_at_creation: false,
            }));
        }
        let attributes = self.attributes.buffer().expect("reserved above").clone();
        let terrain = self
            .terrain_indices
            .buffer()
            .expect("reserved above")
            .clone();
        let scratch = self.scratch.clone().expect("created above");

        self.repack_jobs.clear();
        let mut builds = Vec::with_capacity(picks.len());
        for Pick {
            key,
            scratch_at,
            vertex,
            index,
        } in picks
        {
            let Some(e) = self.entries.get_mut(&key) else {
                continue;
            };
            let first_attribute = e.attributes.unwrap_or(0);
            let count = e.source.vertices.1;
            let stride = e.layout.stride;
            let (index_buffer, first_index, index_copy) = match (index, e.terrain_indices) {
                (Some((src, at)), Some(block)) => (
                    terrain.clone(),
                    block,
                    Some(BufferCopy {
                        src,
                        at,
                        bytes: u64::from(e.source.indices.1) * 4,
                        dst: u64::from(block) * 4,
                    }),
                ),
                _ => (arena.clone(), e.source.indices.0, None),
            };
            let (geometries, bases, cutout) = blas_geometries(e, first_index);
            let blas = device.wgpu_device().create_blas(
                &CreateBlasDescriptor {
                    label: Some("rt_blas"),
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: geometries.iter().map(|g| g.0.clone()).collect(),
                },
            );
            let repack_offset = self.repack_jobs.push(&RepackJob {
                src_word: (scratch_at / 4) as u32,
                stride: (stride / 4) as u32,
                count,
                dst: first_attribute,
                normal: e.layout.normal,
                uv: e.layout.uv,
                word: e.layout.word,
                color: e.layout.color,
            });
            let groups = count.div_ceil(REPACK_GROUP);
            e.blas = Some(blas.clone());
            let (slot, kind, t) = (e.slot, e.source.kind, e.source.translation);
            let rows = [
                [kind, e.source.row, first_attribute, first_index],
                [bases[0], bases[1], cutout, count],
            ];
            builds.push(RtBuild {
                key,
                blas: blas.clone(),
                geometries,
                index: index_buffer,
                stride,
                scratch_vertex: (scratch_at / stride) as u32,
                vertex_copy: BufferCopy {
                    src: vertex.0,
                    at: vertex.1,
                    bytes: u64::from(count) * stride,
                    dst: scratch_at,
                },
                index_copy,
                repack_offset,
                repack_groups: (groups.min(REPACK_ROW), groups.div_ceil(REPACK_ROW)),
            });
            self.ensure_tlas(slot, device);
            let transform = [1.0, 0.0, 0.0, t.x, 0.0, 1.0, 0.0, t.y, 0.0, 0.0, 1.0, t.z];
            let instance = TlasInstance::new(&blas, transform, slot, 1u8 << kind);
            self.tlas[slot as usize] = Some(instance);
            self.write_instance(slot, rows);
            self.built += 1;
        }
        self.repack_jobs.write_buffer(device, queue);
        let layout = pipeline_cache.get_bind_group_layout(&self.repack_layout);
        self.frame.repack_bind = self.repack_jobs.binding().map(|job| {
            device.create_bind_group(
                "rt_repack",
                &layout,
                &BindGroupEntries::sequential((
                    scratch.as_entire_binding(),
                    attributes.as_entire_binding(),
                    job,
                )),
            )
        });
        self.frame.builds = builds;
        self.frame.build_tlas = true;
    }

    /// Rebuild the bind group when the TLAS, a buffer or the pool's class count moved; none while
    /// the TLAS holds nothing.
    fn refresh_bind(
        &mut self,
        shared: &bindless::GxBindless,
        pool: &mut GxTexturePool,
        pipes: &GxPipelines,
        pipeline_cache: &PipelineCache,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        let (Some(table), Some(arena), Some(terrain), Some(attributes), Some(instances)) = (
            shared.table_buffer(),
            shared.index_buffer(),
            self.terrain_indices.buffer(),
            self.attributes.buffer(),
            self.instance_buffer.as_ref(),
        ) else {
            self.bind = None;
            return;
        };
        if self.built == 0 {
            self.bind = None;
            return;
        }
        let white = pool.white(device, queue);
        let classes = pool.class_count().min(GX_CLASS_SLOTS as usize) as u32;
        let key = BindKey {
            tlas: self.tlas_generation,
            instances: instances.id(),
            table: table.id(),
            arena: arena.id(),
            terrain: terrain.id(),
            attributes: attributes.id(),
            classes,
        };
        if self.bound == Some(key) && self.bind.is_some() {
            return;
        }
        let views: Vec<&WgpuTextureView> = (0..GX_CLASS_SLOTS)
            .map(|c| {
                let class = if c < classes { c as u16 } else { white };
                &**pool.view(class)
            })
            .collect();
        let layout = pipeline_cache.get_bind_group_layout(&self.layout);
        self.bind = Some(device.create_bind_group(
            "rt_scene",
            &layout,
            &BindGroupEntries::sequential((
                self.tlas.as_binding(),
                instances.as_entire_binding(),
                table.as_entire_binding(),
                arena.as_entire_binding(),
                terrain.as_entire_binding(),
                attributes.as_entire_binding(),
                &views[..],
                &pipes.sampler,
                &pipes.sampler_clamp,
            )),
        ));
        self.bound = Some(key);
    }

    /// One line when a streaming burst's last build is planned: the BLAS count and the memory.
    fn log_settled(&mut self, device: &RenderDevice) {
        let waiting = self.entries.values().any(|e| e.blas.is_none());
        if !std::mem::replace(&mut self.was_pending, waiting) || waiting {
            return;
        }
        let (mut terrain, mut triangles, mut input) = (0usize, 0u64, 0u64);
        for e in self.entries.values() {
            terrain += usize::from(e.source.kind == KIND_TERRAIN);
            triangles += u64::from(e.source.indices.1 / 3);
            input += u64::from(e.source.vertices.1) * 12 + u64::from(e.source.indices.1) * 4;
        }
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        // Zero unless wgpu is built with its `counters` feature.
        let counted = device
            .wgpu_device()
            .get_internal_counters()
            .hal
            .acceleration_structure_memory
            .read();
        let structures = if counted > 0 {
            format!("{:.1} MiB", mib(counted as u64))
        } else {
            "uncounted".to_string()
        };
        info!(
            "rt_scene: {} BLAS ({} static, {terrain} terrain), {triangles} triangles; {:.1} MiB of \
             positions and indices in, {:.1} MiB hit attributes, {:.1} MiB scratch, acceleration \
             structures {structures}; TLAS {} slots",
            self.entries.len(),
            self.entries.len() - terrain,
            mib(input),
            mib(self.attributes.bytes()),
            mib(self.scratch_bytes),
            self.tlas.get().len(),
        );
    }
}

fn geometry_size(
    vertex_count: u32,
    index_count: u32,
    flags: AccelerationStructureGeometryFlags,
) -> BlasTriangleGeometrySizeDescriptor {
    BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_count),
        flags,
    }
}

/// An entry's geometries (size, first index): its opaque indices, then its cutout ones, each where
/// it has a triangle. Also each geometry's first triangle in the entry and which one is cutout.
fn blas_geometries(
    e: &RtEntry,
    first_index: u32,
) -> (Vec<(BlasTriangleGeometrySizeDescriptor, u32)>, [u32; 2], u32) {
    let (len, split, vertices) = (e.source.indices.1, e.split, e.source.vertices.1);
    let opaque = if e.hidden {
        AccelerationStructureGeometryFlags::empty()
    } else {
        AccelerationStructureGeometryFlags::OPAQUE
    };
    let mut out = Vec::with_capacity(2);
    let mut bases = [0u32; 2];
    let mut cutout = NO_GEOMETRY;
    if split >= 3 {
        out.push((geometry_size(vertices, split, opaque), first_index));
    }
    if len - split >= 3 {
        bases[out.len()] = split / 3;
        cutout = out.len() as u32;
        out.push((
            geometry_size(
                vertices,
                len - split,
                AccelerationStructureGeometryFlags::empty(),
            ),
            first_index + split,
        ));
    }
    (out, bases, cutout)
}

/// A region's index count before its first cutout item (`assemble_region` sorts opaque items
/// first), and whether an item is hidden (record bit 0: killed, or its texture not resident).
fn static_geometry(gpu: &GxCellGpu, draw: &GxCellDraw, len: u32) -> (u32, bool) {
    let mut split = len;
    let mut opaque_end = 0;
    for (range, item) in gpu.ranges.iter().zip(&draw.draws) {
        if item.cutout {
            split = split.min(range.start);
        } else {
            opaque_end = opaque_end.max(range.end);
        }
    }
    let hidden = gpu.records.iter().any(|r| r[3] & 1 != 0);
    // Not opaque-first after all: every item opaque, rather than every item alpha-tested.
    if opaque_end > split {
        return (len, hidden);
    }
    (split, hidden)
}

/// The word offsets the repack reads: position must lead (`Float32x3` at 0, as every BLAS input
/// here is read), and an attribute counts only in the format it is read as. The vertex colour is
/// taken only beside a static word: terrain's `COLOR` is its splat layers.
fn mesh_vertex_words(mesh: &RenderMesh) -> Option<VertexLayout> {
    let layout = &mesh.layout.0;
    let ids = layout.attribute_ids();
    let attributes = &layout.layout().attributes;
    let stride = layout.layout().array_stride;
    let first = attributes.first()?;
    if ids.first() != Some(&Mesh::ATTRIBUTE_POSITION.id)
        || first.offset != 0
        || first.format != VertexFormat::Float32x3
        || !stride.is_multiple_of(4)
    {
        return None;
    }
    let find = |id: MeshVertexAttributeId, format: VertexFormat| {
        ids.iter()
            .position(|i| *i == id)
            .and_then(|i| attributes.get(i))
            .filter(|a| a.format == format)
            .map_or(NONE, |a| (a.offset / 4) as u32)
    };
    let word = find(ATTRIBUTE_GX_WORD.id, VertexFormat::Uint32);
    let color = find(Mesh::ATTRIBUTE_COLOR.id, VertexFormat::Float32x4);
    Some(VertexLayout {
        stride,
        normal: find(Mesh::ATTRIBUTE_NORMAL.id, VertexFormat::Float32x3),
        uv: find(Mesh::ATTRIBUTE_UV_0.id, VertexFormat::Float32x2),
        word,
        color: if word == NONE { NONE } else { color },
    })
}

/// `RenderStartup`: decide whether the session is capable, and set the scene up where it is. A
/// lane's startup system that reads [`RtTier`] or [`capable`] orders after this.
pub(crate) fn init_rt_scene(
    mut commands: Commands,
    caps: Option<Res<HwCaps>>,
    device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let capable = capable_on(caps.as_deref());
    CAPABLE.store(capable, Ordering::Relaxed);
    commands.insert_resource(RtTier {
        capable,
        ..default()
    });
    if !capable {
        return;
    }
    let repack_layout = BindGroupLayoutDescriptor::new(
        "rt_repack_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                uniform_buffer::<RepackJob>(true),
            ),
        ),
    );
    let repack = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("modern_rt_repack".into()),
        layout: vec![repack_layout.clone()],
        immediate_size: 0,
        shader: asset_server.load("embedded://benilla_world/shaders/modern/rt_repack.wgsl"),
        shader_defs: vec![],
        entry_point: Some("repack".into()),
        zero_initialize_workgroup_memory: false,
    });
    let budget = crate::launch_env::var("WOW_RT_BUDGET")
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(DEFAULT_BUDGET);
    info!(
        "rt_scene: ray tracing capable, {budget} triangles a frame; the rayTracing option switches \
         it at run time (WOW_RT=0 turns it off for the session)"
    );
    commands.insert_resource(RtScene::new(&device, repack_layout, repack, budget));
}

/// `PrepareAssets`, first thing each frame: decide [`RtTier::enabled`] (capable and the
/// `rayTracing` option) and [`RtTier::live`] (enabled and the scene ready as the last frame left
/// it), publish both for every world ([`enabled`], [`live`]), and log one line at each switch: the
/// requested and the effective state, the frame, and at a going-live how long the scene took.
fn update_rt_tier(
    setting: Option<Res<crate::modern::RayTracingSetting>>,
    tier: Option<ResMut<RtTier>>,
    scene: Option<Res<RtScene>>,
) {
    let Some(mut tier) = tier else {
        return;
    };
    tier.frame = tier.frame.wrapping_add(1);
    let frame = tier.frame;
    let requested = setting.is_none_or(|s| s.0);
    let enabled = tier.capable && requested;
    let scene = scene.as_deref();
    let live = enabled && scene.is_some_and(|s| s.ready);
    let on_off = |b: bool| if b { "on" } else { "off" };
    if enabled != tier.enabled && !enabled {
        tier.waiting_since = None;
        info!(
            "rt: switch at frame {frame}: requested off, effective off (was {}); the lanes are on \
             raster and the scene is freed",
            on_off(tier.live)
        );
    } else if enabled != tier.enabled {
        tier.waiting_since = Some((Instant::now(), frame));
        info!(
            "rt: switch at frame {frame}: requested on, effective off until the scene is ready \
             (built nearest the camera first, WOW_RT_BUDGET triangles a frame)"
        );
    } else if live != tier.live && live {
        let (ms, frames) = tier.waiting_since.take().map_or((0.0, 0), |(at, from)| {
            (at.elapsed().as_secs_f64() * 1000.0, frame.wrapping_sub(from))
        });
        info!(
            "rt: switch at frame {frame}: requested on, effective on; the scene was ready {ms:.0} \
             ms ({frames} frames) after the lanes asked for it, {} entries",
            scene.map_or(0, |s| s.entries.len())
        );
    } else if live != tier.live {
        tier.waiting_since = Some((Instant::now(), frame));
        info!(
            "rt: switch at frame {frame}: requested on, effective off; the scene is not ready (a \
             classic frame, a map change, or more waiting than built), the lanes are on raster \
             until it is"
        );
    }
    tier.enabled = enabled;
    tier.live = live;
    ENABLED.store(enabled, Ordering::Relaxed);
    LIVE.store(live, Ordering::Relaxed);
}

/// Diff the scene against the regions and cells, plan this frame's builds, keep the bind group
/// current and judge the scene's readiness. With the option off, free it all instead. After
/// `prepare_static_gx`, whose assemblies and evictions it reads.
#[allow(clippy::too_many_arguments)]
fn prepare_rt_scene(
    tier: Res<RtTier>,
    scene: ResMut<RtScene>,
    gx: Res<GxWorld>,
    cache: Res<GxGpuCache>,
    shared: Option<Res<bindless::GxBindless>>,
    mut pool: ResMut<GxTexturePool>,
    pipes: Res<GxPipelines>,
    terrain: Option<Res<ModernTerrainCasters>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<(&ExtractedView, &StaticGxView)>,
) {
    // `WOW_GX_PERF`: counted in static_gx's `prepare=`, the phase it extends.
    let _t = crate::static_gx::gx_perf_guard(3);
    let scene = scene.into_inner();
    // fork: F — before any early return: the stand-in is what a classic frame binds too, and every
    // frame the lanes are not live.
    scene.ensure_stub(&mut pool, &pipes, &pipeline_cache, &device, &queue);
    let last = std::mem::take(&mut scene.frame);
    // The option is off: nothing is built or kept, and `assemble_unseen` does not run.
    if !tier.enabled {
        scene.free_all(&device);
        return;
    }
    if !last.recorded.load(Ordering::Relaxed) {
        scene.requeue(last);
    }
    let modern = benilla_assets::materials::MODERN_MODEL_PIPELINES
        .load(Ordering::Relaxed);
    let map_clear = gx.cells.is_empty() && gx.wmos.is_empty() && gx.props.is_empty();
    let shared = match shared {
        Some(shared) if modern && !map_clear => shared,
        _ => {
            scene.clear();
            return;
        }
    };
    // No region placed yet this map: there is no arena to bind, and nothing ready.
    let Some(arena) = shared.index_buffer().cloned() else {
        scene.ready = false;
        scene.complete_frames = 0;
        return;
    };
    scene.frame_no = scene.frame_no.wrapping_add(1);
    let now = scene.frame_no;

    let regions = cache
        .cells
        .iter()
        .map(|(k, g)| (RtKey::Cell(k.0, k.1), KIND_CELL, g, gx.cells.get(k)))
        .chain(
            cache
                .wmos
                .iter()
                .map(|(e, g)| (RtKey::Wmo(*e), KIND_WMO, g, gx.wmos.get(e))),
        )
        .chain(
            cache
                .props
                .iter()
                .map(|(e, g)| (RtKey::Prop(*e), KIND_PROP, g, gx.props.get(e))),
        );
    for (key, kind, gpu, draw) in regions {
        let (Some(draw), GxRegionBinds::Shared(slots)) = (draw, &gpu.binds) else {
            continue;
        };
        let (Some(mesh), Some(vslice)) = (
            meshes.get(gpu.mesh),
            allocator.mesh_vertex_slice(&gpu.mesh),
        ) else {
            continue;
        };
        let Some(layout) = mesh_vertex_words(mesh) else {
            continue;
        };
        if slots.index_len < 3 || vslice.range.is_empty() {
            continue;
        }
        let source = Source {
            kind,
            mesh: gpu.mesh,
            vertices: (vslice.range.start, vslice.range.len() as u32),
            indices: (slots.index_base, slots.index_len),
            row: slots.row,
            rev: gpu.killed_applied,
            translation: draw.origin,
        };
        let center = draw.origin + Vec3::from(draw.aabb.center);
        if let Some(entry) = scene.visit(key, source, layout, center, now) {
            (entry.split, entry.hidden) = static_geometry(gpu, draw, slots.index_len);
        }
    }
    for &(mesh_id, center, _, epoch) in terrain.iter().flat_map(|t| t.cells.iter()) {
        let Some(mesh) = meshes.get(mesh_id) else {
            continue;
        };
        let RenderMeshBufferInfo::Indexed {
            count,
            index_format: IndexFormat::Uint32,
        } = mesh.buffer_info
        else {
            continue;
        };
        let (Some(vslice), Some(islice)) = (
            allocator.mesh_vertex_slice(&mesh_id),
            allocator.mesh_index_slice(&mesh_id),
        ) else {
            continue;
        };
        let Some(layout) = mesh_vertex_words(mesh) else {
            continue;
        };
        if count < 3 || vslice.range.is_empty() {
            continue;
        }
        let source = Source {
            kind: KIND_TERRAIN,
            mesh: mesh_id,
            vertices: (vslice.range.start, vslice.range.len() as u32),
            indices: (islice.range.start, count),
            row: 0,
            rev: epoch,
            translation: Vec3::ZERO,
        };
        scene.visit(RtKey::Terrain(mesh_id), source, layout, center, now);
    }
    scene.sweep(now);

    let camera = views
        .iter()
        .find(|(_, marker)| marker.primary)
        .map_or(Vec3::ZERO, |(view, _)| view.world_from_view.translation());
    let mut copies = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("rt_scene_copies"),
    });
    // The repack writes what a hit reads: nothing is built before its pipeline is.
    if pipeline_cache.get_compute_pipeline(scene.repack).is_some() {
        scene.plan(
            camera,
            &arena,
            &allocator,
            &pipeline_cache,
            &device,
            &queue,
            &mut copies,
        );
    }
    scene.upload_instances(&device, &queue);
    queue.submit([copies.finish()]);
    scene.refresh_bind(&shared, &mut pool, &pipes, &pipeline_cache, &device, &queue);
    scene.log_settled(&device);
    scene.update_ready();
}

/// Record this frame's builds: the raw copies, the attribute repack (span `modern_rt_repack`), the
/// BLAS builds (`modern_rt_blas`) and the TLAS build (`modern_rt_tlas`). Runs once a frame, in the
/// first static_gx view to reach it; a pass that traces orders after it.
pub(crate) fn rt_scene_pass(
    world: &World,
    _view: ViewQuery<&StaticGxView>,
    mut render_context: RenderContext,
) {
    let Some(scene) = world.get_resource::<RtScene>() else {
        return;
    };
    let frame = &scene.frame;
    if frame.builds.is_empty() && !(frame.build_tlas && scene.built > 0) {
        return;
    }
    if frame.recorded.swap(true, Ordering::Relaxed) {
        return;
    }
    let pipeline_cache = world.resource::<PipelineCache>();
    let diagnostics = render_context.diagnostic_recorder();
    let encoder = render_context.command_encoder();

    if let Some(scratch) = scene.scratch.as_ref().filter(|_| !frame.builds.is_empty()) {
        for b in &frame.builds {
            let v = &b.vertex_copy;
            encoder.copy_buffer_to_buffer(&v.src, v.at, scratch, v.dst, v.bytes);
            if let Some(i) = &b.index_copy {
                encoder.copy_buffer_to_buffer(&i.src, i.at, &b.index, i.dst, i.bytes);
            }
        }
        if let (Some(repack), Some(bind)) = (
            pipeline_cache.get_compute_pipeline(scene.repack),
            frame.repack_bind.as_ref(),
        ) {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_rt_repack"),
                timestamp_writes: None,
            });
            let span = diagnostics.pass_span(&mut pass, "modern_rt_repack");
            pass.set_pipeline(repack);
            for b in &frame.builds {
                pass.set_bind_group(0, bind, &[b.repack_offset]);
                pass.dispatch_workgroups(b.repack_groups.0, b.repack_groups.1, 1);
            }
            span.end(&mut pass);
        }
        let entries: Vec<BlasBuildEntry<'_>> = frame
            .builds
            .iter()
            .map(|b| BlasBuildEntry {
                blas: &b.blas,
                geometry: BlasGeometries::TriangleGeometries(
                    b.geometries
                        .iter()
                        .map(|(size, first)| BlasTriangleGeometry {
                            size,
                            vertex_buffer: &**scratch,
                            first_vertex: b.scratch_vertex,
                            vertex_stride: b.stride,
                            index_buffer: Some(&*b.index),
                            first_index: Some(*first),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        })
                        .collect(),
                ),
            })
            .collect();
        let span = diagnostics.time_span(encoder, "modern_rt_blas");
        encoder.build_acceleration_structures(entries.iter(), std::iter::empty::<&Tlas>());
        span.end(encoder);
    }

    if frame.build_tlas && scene.built > 0 {
        let span = diagnostics.time_span(encoder, "modern_rt_tlas");
        encoder.build_acceleration_structures(
            std::iter::empty::<&BlasBuildEntry<'_>>(),
            std::iter::once(&scene.tlas),
        );
        span.end(encoder);
    }
}

/// While the tier is enabled, assemble up to [`UNSEEN_BUDGET`] regions a frame that the camera has
/// not drawn (or whose held-back textures arrived out of view), so the scene holds every streamed
/// region. Called by `prepare_static_gx` after the visible regions, into the same `copies`. The
/// regions still not held after it ([`UNSEEN_BACKLOG`]) hold the scene's readiness back.
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_unseen(
    gx: &GxWorld,
    cache: &mut GxGpuCache,
    pool: &mut GxTexturePool,
    shared: Option<&mut bindless::GxBindless>,
    pipes: &GxPipelines,
    pipeline_cache: &PipelineCache,
    device: &RenderDevice,
    queue: &RenderQueue,
    images: &RenderAssets<GpuImage>,
    allocator: &MeshAllocator,
    copies: &mut CommandEncoder,
) {
    let mut assembly = Assembly {
        pool,
        shared,
        pipes,
        pipeline_cache,
        device,
        queue,
        images,
        allocator,
        copies,
        budget: UNSEEN_BUDGET,
        placed: false,
        backlog: 0,
    };
    assembly.unseen(&gx.cells, &mut cache.cells);
    assembly.unseen(&gx.wmos, &mut cache.wmos);
    assembly.unseen(&gx.props, &mut cache.props);
    if assembly.placed {
        cache.rev += 1;
    }
    UNSEEN_BACKLOG.store(assembly.backlog, Ordering::Relaxed);
}

/// [`assemble_region`]'s inputs, carried across the three region maps.
struct Assembly<'a> {
    pool: &'a mut GxTexturePool,
    shared: Option<&'a mut bindless::GxBindless>,
    pipes: &'a GxPipelines,
    pipeline_cache: &'a PipelineCache,
    device: &'a RenderDevice,
    queue: &'a RenderQueue,
    images: &'a RenderAssets<GpuImage>,
    allocator: &'a MeshAllocator,
    copies: &'a mut CommandEncoder,
    budget: u32,
    placed: bool,
    /// Regions not held at all and left for a later frame (past the budget, or not assemblable
    /// yet): what [`UNSEEN_BACKLOG`] publishes.
    backlog: u32,
}

impl Assembly<'_> {
    fn unseen<K: Copy + Eq + std::hash::Hash>(
        &mut self,
        drawn: &HashMap<K, std::sync::Arc<GxCellDraw>>,
        held: &mut HashMap<K, GxCellGpu>,
    ) {
        for (key, draw) in drawn {
            let current = held.get(key);
            if current.is_some_and(|g| !g.wants_reassembly(self.images)) {
                continue;
            }
            // A held region waiting for a texture is in the scene already; only a missing one
            // counts against readiness.
            let missing = current.is_none();
            if self.budget == 0 {
                self.backlog += u32::from(missing);
                continue;
            }
            self.budget -= 1;
            let Some(gpu) = assemble_region(
                draw,
                self.pool,
                self.shared.as_deref_mut(),
                self.pipes,
                self.pipeline_cache,
                self.device,
                self.queue,
                self.images,
                self.allocator,
                self.copies,
            ) else {
                self.backlog += u32::from(missing);
                continue;
            };
            if let Some(old) = held.insert(*key, gpu) {
                evict(self.shared.as_deref_mut(), &old);
            }
            self.placed = true;
        }
    }
}

/// Wire the scene (called from `render::build`, native only).
pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .add_systems(
            bevy::render::RenderStartup,
            init_rt_scene.after(crate::modern::caps::init),
        )
        // The levels are decided before anything in the frame reads them.
        .add_systems(
            Render,
            update_rt_tier.in_set(RenderSystems::PrepareAssets),
        )
        .add_systems(
            Render,
            prepare_rt_scene
                .in_set(RenderSystems::PrepareResources)
                .after(prepare_static_gx)
                .run_if(|tier: Option<Res<RtTier>>| tier.is_some_and(|t| t.capable)),
        )
        .add_systems(Core3d, rt_scene_pass.before(Core3dSystems::MainPass));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL agrees with the layout here: the array size, the row counts, the kinds.
    #[test]
    fn the_shaders_agree_with_the_scene() {
        let lib = include_str!("../../shaders/modern/rt_scene.wgsl");
        assert!(lib.contains(&format!(
            "binding_array<texture_2d_array<f32>, {GX_CLASS_SLOTS}>"
        )));
        assert!(lib.contains(&format!("const ATTRIBUTE_WORDS: u32 = {ATTRIBUTE_WORDS}u;")));
        assert!(lib.contains(&format!("const NO_GEOMETRY: u32 = {NO_GEOMETRY}u;")));
        assert!(lib.contains(&format!("const STUB_MARK: u32 = {STUB_MARK:#x}u;")));
        assert!(lib.contains(&format!(
            "const HEADER_ROWS: u32 = {}u;",
            bindless::HEADER_ROWS
        )));
        assert!(lib.contains(&format!("custom * {INSTANCE_ROWS}u")));
        for (name, kind) in [
            ("KIND_CELL", KIND_CELL),
            ("KIND_WMO", KIND_WMO),
            ("KIND_PROP", KIND_PROP),
            ("KIND_TERRAIN", KIND_TERRAIN),
        ] {
            assert!(
                lib.contains(&format!("const {name}: u32 = {kind}u;")),
                "{name}"
            );
        }
        let repack = include_str!("../../shaders/modern/rt_repack.wgsl");
        assert!(repack.contains(&format!("const ATTRIBUTE_WORDS: u32 = {ATTRIBUTE_WORDS}u;")));
        assert!(repack.contains(&format!("@workgroup_size({REPACK_GROUP})")));
        assert!(repack.contains(&format!(
            "const ROW: u32 = {}u;",
            REPACK_ROW * REPACK_GROUP
        )));
    }

    fn entry(len: u32, split: u32, hidden: bool) -> RtEntry {
        RtEntry {
            source: Source {
                kind: KIND_WMO,
                mesh: AssetId::default(),
                vertices: (0, 100),
                indices: (40, len),
                row: 0,
                rev: 0,
                translation: Vec3::ZERO,
            },
            layout: VertexLayout {
                stride: 64,
                normal: 3,
                uv: 6,
                word: 12,
                color: 8,
            },
            slot: 0,
            center: Vec3::ZERO,
            split,
            hidden,
            blas: None,
            attributes: None,
            terrain_indices: None,
            seen: 0,
        }
    }

    /// Opaque indices first and `OPAQUE`, cutout ones after and non-opaque; either alone is
    /// geometry 0; a hidden item makes both non-opaque.
    #[test]
    fn a_region_splits_into_opaque_then_cutout_geometry() {
        let opaque = AccelerationStructureGeometryFlags::OPAQUE;
        let none = AccelerationStructureGeometryFlags::empty();
        let (g, bases, cutout) = blas_geometries(&entry(30, 12, false), 40);
        assert_eq!(g.len(), 2);
        assert_eq!(
            (g[0].0.index_count, g[0].0.flags, g[0].1),
            (Some(12), opaque, 40)
        );
        assert_eq!(
            (g[1].0.index_count, g[1].0.flags, g[1].1),
            (Some(18), none, 52)
        );
        assert_eq!((bases, cutout), ([0, 4], 1));

        let (g, bases, cutout) = blas_geometries(&entry(30, 0, false), 40);
        assert_eq!((g.len(), g[0].1, bases, cutout), (1, 40, [0, 0], 0));

        let (g, _, cutout) = blas_geometries(&entry(30, 30, false), 40);
        assert_eq!((g.len(), g[0].0.flags, cutout), (1, opaque, NO_GEOMETRY));

        let (g, _, _) = blas_geometries(&entry(30, 12, true), 40);
        assert!(g.iter().all(|(s, _)| s.flags.is_empty()));
    }
}
