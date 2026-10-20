//! **The GI scene** (fork-owned; `FORK.md` P5).
//!
//! A child of `static_gx::render` for the same reason as `modern_shadow`: the world the probes trace
//! against is exactly the world this pass already holds on the GPU — the baked cell, WMO and prop
//! buffers, their texture arrays and their record tables.
//!
//! Each frame it keeps two things current for `modern::gi`:
//!
//! 1. **The voxel clipmap.** Levels of `N³` voxels at growing scales, addressed toroidally, holding
//!    the static world's albedo and surface normal. A level is rasterized three times per box — once
//!    down each axis, so a triangle edge-on to one sweep is caught by another — at several samples
//!    per voxel. Only *boxes* are ever redone: the slabs the camera scrolled in, and the bounds of
//!    content that streamed in or out, one union box per level a frame. Nothing else is touched,
//!    so a standing camera pays nothing.
//! 2. **The terrain heightfield.** The resident terrain cells drawn top-down into a depth map (the
//!    plan's `gi/heightfield.rs`), which is what terrain hits and every ray past the clipmap use.
//!    Terrain is not voxelized: it is a height per column by construction, and a 2 yd map holds it
//!    more finely than a 12 yd voxel could.

use bevy::app::SubApp;
use bevy::platform::collections::HashMap as GiMap;
use bevy::render::render_resource::binding_types::{texture_storage_3d, uniform_buffer};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::shader::Shader;

use super::*;
use crate::modern::gi::{self, GiClip, GiTier, ModernGi, ModernGiScene};
use crate::modern::shadow::ModernTerrainCasters;
use crate::modern::ActiveLighting;

/// Samples per voxel edge in the voxelizing raster: a wall thinner than a voxel still fills it.
const SUPERSAMPLE: u32 = 3;
/// The heightfield's edge, in texels, and half its world footprint (yd) — about 1.5 yd a texel.
const HEIGHT_SIZE: u32 = 512;
const HEIGHT_HALF: f32 = 384.0;
/// The heightfield re-centres on this grid, so it (and its cache) holds while the camera moves.
const HEIGHT_SNAP: f32 = 32.0;
/// How far above and below the camera the heightfield can hold a surface.
const HEIGHT_REACH: f32 = 600.0;
/// A clipmap level re-centres when the camera comes this close to its edge (in voxels).
const SCROLL_MARGIN: i32 = 8;
/// Casters re-voxelized per frame because they arrived — each costs one box per level, and each box
/// costs three sweeps, so the budget is small and the backlog drains over the streaming burst.
/// Removals and scrolls are never deferred: stale geometry in the clipmap is a light leak, and a
/// scrolled-in slab is unwritten memory.
const ARRIVAL_BUDGET: usize = 3;
/// The alpha a cutout texel needs to fill its voxel. Lower than the draw lane's key on purpose: a
/// canopy should voxelize as a canopy, not as the handful of texels that survive an 0.88 key.
const VOXEL_ALPHA_KEY: f32 = 0.45;
/// The mip the voxel albedo is sampled from — a texel's worth of detail is meaningless at 1.5 yd.
const VOXEL_ALBEDO_MIP: f32 = 3.0;

#[derive(Clone, Copy, ShaderType)]
struct VoxelView {
    clip_from_world: Mat4,
    box_lo: Vec4,
    box_hi: Vec4,
    info: Vec4,
}

#[derive(Clone, Copy, ShaderType)]
struct HeightView {
    clip_from_world: Mat4,
}

#[derive(Resource)]
struct GiVoxelPipelines {
    view_layout: BindGroupLayoutDescriptor,
    store_layout: BindGroupLayoutDescriptor,
    height_layout: BindGroupLayoutDescriptor,
    voxel_shader: Handle<Shader>,
    height_shader: Handle<Shader>,
    /// Keyed `(cutout, two_sided)`.
    raster: GiMap<(bool, bool), CachedRenderPipelineId>,
    clear: CachedComputePipelineId,
    /// Terrain casters carry position at offset 0 whatever their stride (`terrain_shadow.wgsl`).
    height: GiMap<u64, CachedRenderPipelineId>,
}

/// One box of one level to empty and redraw.
#[derive(Clone, Copy)]
struct VoxelJob {
    level: usize,
    lo: IVec3,
    hi: IVec3,
    /// Uniform offsets: one per raster axis; the clear reuses the first.
    axis: [u32; 3],
}

/// What a caster contributed the last time it was voxelized: enough to notice a re-bake, a kill-bit
/// change or a move, and its world bounds so its old box can be cleared when it leaves.
#[derive(Clone, Copy, PartialEq)]
struct VoxelCaster {
    fingerprint: u64,
    center: Vec3,
    half: Vec3,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CasterKey {
    Cell(i32, i32),
    Wmo(Entity),
    Prop(Entity),
}

#[derive(Resource, Default)]
struct GiVoxelState {
    tier: Option<GiTier>,
    /// The base voxel coord each level currently holds.
    base: [IVec3; gi::MAX_LEVELS],
    /// The static casters the clipmap currently holds.
    casters: GiMap<CasterKey, VoxelCaster>,
    /// The `GxGpuCache::rev` [`Self::casters`] was last diffed against; `None` forces a diff.
    diffed_rev: Option<u64>,
    /// Arrivals were left past the per-frame budget, so the next frame diffs again.
    backlog: bool,
    /// The top of every caster in [`Self::casters`] (`clip.bounds.y` before its rounding margin).
    caster_top: f32,
    jobs: Vec<VoxelJob>,
    albedo: Option<(Texture, TextureView)>,
    normal: Option<(Texture, TextureView)>,
    depth: Option<TextureView>,
    height: Option<(Texture, TextureView)>,
    /// The key the heightfield was last drawn under (0 = redraw).
    height_key: u64,
    height_offset: u32,
    height_pending: bool,
    /// A terrain cell's mesh was not prepared at the last stride scan, so the next frame rescans.
    height_unprepared: bool,
    view_bind: Option<BindGroup>,
    store_bind: Option<BindGroup>,
    height_bind: Option<BindGroup>,
}

#[derive(Resource, Default)]
struct VoxelViewBuffer(DynamicUniformBuffer<VoxelView>);

#[derive(Resource, Default)]
struct HeightViewBuffer(DynamicUniformBuffer<HeightView>);

fn init_gi_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "modern_gi_voxel_view",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
            uniform_buffer::<VoxelView>(true),
        ),
    );
    let store_layout = BindGroupLayoutDescriptor::new(
        "modern_gi_voxel_store",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
            (
                texture_storage_3d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                texture_storage_3d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let height_layout = BindGroupLayoutDescriptor::new(
        "modern_gi_height_view",
        &BindGroupLayoutEntries::single(ShaderStages::VERTEX, uniform_buffer::<HeightView>(true)),
    );
    let clear = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("modern_gi_voxel_clear".into()),
        layout: vec![view_layout.clone(), store_layout.clone()],
        immediate_size: 0,
        shader: asset_server.load("embedded://benilla_world/shaders/modern/gi_voxel_clear.wgsl"),
        shader_defs: vec![],
        entry_point: Some("clear_box".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(GiVoxelPipelines {
        view_layout,
        store_layout,
        height_layout,
        voxel_shader: asset_server
            .load("embedded://benilla_world/shaders/modern/static_gx_voxel.wgsl"),
        height_shader: asset_server
            .load("embedded://benilla_world/shaders/modern/terrain_shadow.wgsl"),
        raster: GiMap::default(),
        clear,
        height: GiMap::default(),
    });
}

/// What the voxel clipmap's textures and contents depend on: voxels per side, the level count and
/// each level's voxel size.
fn clipmap_shape(tier: &GiTier) -> (u32, usize, [f32; gi::MAX_LEVELS]) {
    (tier.voxels, tier.levels, tier.voxel)
}

fn voxel_texture(
    device: &RenderDevice,
    label: &'static str,
    n: u32,
    levels: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: n * levels,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

/// The orthographic box camera that sweeps `axis` over one job's box.
fn box_clip(lo: Vec3, hi: Vec3, axis: usize) -> Mat4 {
    let span = (hi - lo).max(Vec3::splat(1e-3));
    // (u, v) are the two axes across the sweep, `d` the one along it.
    let (u, v, d) = match axis {
        0 => (2usize, 1, 0),
        1 => (0usize, 2, 1),
        _ => (0usize, 1, 2),
    };
    let mut rows = [Vec4::ZERO; 4];
    rows[0][u] = 2.0 / span[u];
    rows[0][3] = -1.0 - 2.0 * lo[u] / span[u];
    // The raster's v axis points down the texture, which is what the sampling convention expects.
    rows[1][v] = -2.0 / span[v];
    rows[1][3] = 1.0 + 2.0 * lo[v] / span[v];
    rows[2][d] = 1.0 / span[d];
    rows[2][3] = -lo[d] / span[d];
    rows[3][3] = 1.0;
    Mat4::from_cols(
        Vec4::new(rows[0].x, rows[1].x, rows[2].x, rows[3].x),
        Vec4::new(rows[0].y, rows[1].y, rows[2].y, rows[3].y),
        Vec4::new(rows[0].z, rows[1].z, rows[2].z, rows[3].z),
        Vec4::new(rows[0].w, rows[1].w, rows[2].w, rows[3].w),
    )
}

/// The top-down heightfield's clip, and the rows that read it back (`probegi::terrain_height`).
fn height_clip(center: Vec3) -> (Mat4, GiClip) {
    let h = HEIGHT_HALF;
    let y_min = center.y - HEIGHT_REACH;
    let span = 2.0 * HEIGHT_REACH;
    let rows = [
        Vec4::new(1.0 / h, 0.0, 0.0, -center.x / h),
        Vec4::new(0.0, 0.0, -1.0 / h, center.z / h),
        Vec4::new(0.0, 1.0 / span, 0.0, -y_min / span),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    ];
    let clip = Mat4::from_cols(
        Vec4::new(rows[0].x, rows[1].x, rows[2].x, rows[3].x),
        Vec4::new(rows[0].y, rows[1].y, rows[2].y, rows[3].y),
        Vec4::new(rows[0].z, rows[1].z, rows[2].z, rows[3].z),
        Vec4::new(rows[0].w, rows[1].w, rows[2].w, rows[3].w),
    );
    let out = GiClip {
        // uv, not ndc: the map is read with `textureLoad`, which takes texels.
        hf_row_x: Vec4::new(0.5 / h, 0.0, 0.0, 0.5 - 0.5 * center.x / h),
        hf_row_y: Vec4::new(0.0, 0.0, 0.5 / h, 0.5 - 0.5 * center.z / h),
        hf_row_z: rows[2],
        hf: Vec4::new(1.0, y_min, span, HEIGHT_SIZE as f32),
        ..GiClip::default()
    };
    (clip, out)
}

/// The world box of a level's voxel range.
fn level_box(lo: IVec3, hi: IVec3, voxel: f32) -> (Vec3, Vec3) {
    (lo.as_vec3() * voxel, hi.as_vec3() * voxel)
}

fn overlaps(center: Vec3, half: Vec3, lo: Vec3, hi: Vec3) -> bool {
    (center - half).cmple(hi).all() && (center + half).cmpge(lo).all()
}

/// The voxel range a world box covers in one level, clamped to that level's window.
fn box_voxels(center: Vec3, half: Vec3, voxel: f32, base: IVec3, n: i32) -> Option<(IVec3, IVec3)> {
    let lo = ((center - half) / voxel).floor().as_ivec3().max(base);
    let hi = ((center + half) / voxel)
        .ceil()
        .as_ivec3()
        .min(base + IVec3::splat(n));
    (lo.cmplt(hi).all()).then_some((lo, hi))
}

/// Each level's boxes as their one bounding box, levels in first-seen order. Every input box lies
/// in its level's window, so the union does too.
fn merge_per_level(boxes: Vec<(usize, IVec3, IVec3)>) -> Vec<(usize, IVec3, IVec3)> {
    let mut merged: Vec<(usize, IVec3, IVec3)> = Vec::new();
    for (level, lo, hi) in boxes {
        match merged.iter_mut().find(|m| m.0 == level) {
            Some(m) => {
                m.1 = m.1.min(lo);
                m.2 = m.2.max(hi);
            }
            None => merged.push((level, lo, hi)),
        }
    }
    merged
}

/// The slabs a level gains when its window moves from `from` to `to` (up to one per axis).
fn scroll_jobs(from: IVec3, to: IVec3, n: i32, out: &mut Vec<(IVec3, IVec3)>) {
    let delta = to - from;
    if delta == IVec3::ZERO {
        return;
    }
    if delta.abs().max_element() >= n {
        out.push((to, to + IVec3::splat(n)));
        return;
    }
    for axis in 0..3 {
        let d = delta[axis];
        if d == 0 {
            continue;
        }
        let mut lo = to;
        let mut hi = to + IVec3::splat(n);
        if d > 0 {
            lo[axis] = from[axis] + n;
        } else {
            hi[axis] = from[axis];
        }
        out.push((lo, hi));
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_gi_voxels(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    active: Option<Res<ActiveLighting>>,
    gi: Option<Res<ModernGi>>,
    gx: Option<Res<GxWorld>>,
    cache: Option<Res<GxGpuCache>>,
    terrain: Option<Res<ModernTerrainCasters>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    pipeline_cache: Res<PipelineCache>,
    pipes: Option<ResMut<GiVoxelPipelines>>,
    gx_pipes: Option<Res<GxPipelines>>,
    mut state: ResMut<GiVoxelState>,
    mut views: ResMut<VoxelViewBuffer>,
    mut heights: ResMut<HeightViewBuffer>,
    mut scene: ResMut<ModernGiScene>,
) {
    scene.ready = false;
    state.jobs.clear();
    state.height_pending = false;
    state.view_bind = None;
    state.height_bind = None;
    let (Some(gi), Some(gx), Some(cache), Some(mut pipes), Some(gx_pipes)) =
        (gi, gx, cache, pipes, gx_pipes)
    else {
        return;
    };
    if !active.is_some_and(|a| a.is_modern()) || !gi.on {
        if state.tier.is_some() {
            *state = GiVoxelState::default();
        }
        return;
    }
    let tier = gi.tier;
    let n = tier.voxels as i32;

    // A new clipmap shape (or the first modern frame) re-creates the clipmap, and everything in it is
    // new. The probe cascades are not part of it: the interior sub-grid a room adds
    // (`GiTier::with_interior`) leaves the voxels as they are, so a camera passing in and out of
    // rooms keeps its textures instead of reallocating three 3D textures at every crossing.
    let fresh = state.tier.map(|t| clipmap_shape(&t)) != Some(clipmap_shape(&tier));
    if fresh {
        state.albedo = Some(voxel_texture(
            &device,
            "modern_gi_voxel_albedo",
            tier.voxels,
            tier.levels as u32,
        ));
        state.normal = Some(voxel_texture(
            &device,
            "modern_gi_voxel_normal",
            tier.voxels,
            tier.levels as u32,
        ));
        let raster = tier.voxels * SUPERSAMPLE;
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("modern_gi_voxel_depth"),
            size: Extent3d {
                width: raster,
                height: raster,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CORE_3D_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        state.depth = Some(depth.create_view(&TextureViewDescriptor::default()));
        state.casters.clear();
        state.diffed_rev = None;
        state.store_bind = None;
        state.height_key = 0;
        info!(
            "modern GI: voxel clipmap {}^3 x {} levels ({:?} yd), heightfield {HEIGHT_SIZE}^2 over {} yd",
            tier.voxels,
            tier.levels,
            &tier.voxel[..tier.levels],
            2.0 * HEIGHT_HALF,
        );
    }
    state.tier = Some(tier);
    if state.height.is_none() {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("modern_gi_heightfield"),
            size: Extent3d {
                width: HEIGHT_SIZE,
                height: HEIGHT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        state.height = Some((texture, view));
        state.height_key = 0;
    }
    let (Some(albedo), Some(normal), Some(height)) = (
        state.albedo.clone(),
        state.normal.clone(),
        state.height.clone(),
    ) else {
        return;
    };
    if state.store_bind.is_none() {
        state.store_bind = Some(device.create_bind_group(
            "modern_gi_voxel_store",
            &pipeline_cache.get_bind_group_layout(&pipes.store_layout),
            &BindGroupEntries::sequential((&albedo.1, &normal.1)),
        ));
    }

    // --- the clipmap's windows -----------------------------------------------------------------
    let mut boxes: Vec<(usize, IVec3, IVec3)> = Vec::new();
    for level in 0..tier.levels {
        let voxel = tier.voxel[level];
        let want = (gi.camera / voxel).floor().as_ivec3() - IVec3::splat(n / 2);
        let current = state.base[level];
        let cam = (gi.camera / voxel).floor().as_ivec3();
        let inside = cam.cmpge(current + IVec3::splat(SCROLL_MARGIN)).all()
            && cam.cmplt(current + IVec3::splat(n - SCROLL_MARGIN)).all();
        if fresh {
            state.base[level] = want;
            boxes.push((level, want, want + IVec3::splat(n)));
        } else if !inside {
            let mut slabs = Vec::new();
            scroll_jobs(current, want, n, &mut slabs);
            state.base[level] = want;
            for (lo, hi) in slabs {
                boxes.push((level, lo, hi));
            }
        }
    }

    // --- what the static world has, against what the clipmap holds -----------------------------
    // Only when the cache moved, a backlog is draining, or the clipmap is new: on a standing
    // camera the diff below finds nothing, and walking every cached region to learn that was the
    // largest single cost of the browser's frame.
    let mut current_len = state.casters.len();
    if fresh || state.backlog || state.diffed_rev != Some(cache.rev) {
        let mut current: GiMap<CasterKey, VoxelCaster> = GiMap::default();
        let drawable = |gpu: &GxCellGpu, draw: &GxCellDraw| -> Option<VoxelCaster> {
            if gpu.runs.is_empty() {
                return None;
            }
            meshes.get(draw.mesh.id())?;
            allocator.mesh_vertex_slice(&draw.mesh.id())?;
            allocator.mesh_index_slice(&draw.mesh.id())?;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            {
                use std::hash::Hash;
                gpu.mesh.hash(&mut hash);
                gpu.killed_applied.hash(&mut hash);
                gpu.runs.len().hash(&mut hash);
                draw.origin.to_array().map(f32::to_bits).hash(&mut hash);
            }
            Some(VoxelCaster {
                fingerprint: {
                    use std::hash::Hasher;
                    hash.finish()
                },
                center: draw.origin + Vec3::from(draw.aabb.center),
                half: Vec3::from(draw.aabb.half_extents),
            })
        };
        for (key, gpu) in &cache.cells {
            if let Some(draw) = gx.cells.get(key) {
                if let Some(v) = drawable(gpu, draw) {
                    current.insert(CasterKey::Cell(key.0, key.1), v);
                }
            }
        }
        for (entity, gpu) in &cache.wmos {
            if let Some(draw) = gx.wmos.get(entity) {
                if let Some(v) = drawable(gpu, draw) {
                    current.insert(CasterKey::Wmo(*entity), v);
                }
            }
        }
        for (entity, gpu) in &cache.props {
            if let Some(draw) = gx.props.get(entity) {
                if let Some(v) = drawable(gpu, draw) {
                    current.insert(CasterKey::Prop(*entity), v);
                }
            }
        }

        // Gone or changed: their old box is emptied and rebuilt from whatever still overlaps it.
        let mut departed: Vec<VoxelCaster> = Vec::new();
        state.casters.retain(|key, had| match current.get(key) {
            Some(now) if now == had => true,
            _ => {
                departed.push(*had);
                false
            }
        });
        let mut arrived: Vec<(CasterKey, VoxelCaster)> = current
            .iter()
            .filter(|(key, _)| !state.casters.contains_key(*key))
            .map(|(key, v)| (*key, *v))
            .collect();
        let arrived_len = arrived.len();
        // Near the camera first: the finest cascade is what the player is standing in.
        arrived.sort_by(|a, b| {
            let d = |v: &VoxelCaster| v.center.distance_squared(gi.camera);
            d(&a.1).total_cmp(&d(&b.1))
        });
        let admitted = if fresh {
            0
        } else {
            arrived.len().min(ARRIVAL_BUDGET)
        };
        current_len = current.len();

        for v in departed
            .iter()
            .chain(arrived.iter().take(admitted).map(|(_, v)| v))
        {
            for level in 0..tier.levels {
                let voxel = tier.voxel[level];
                if let Some((lo, hi)) = box_voxels(v.center, v.half, voxel, state.base[level], n) {
                    boxes.push((level, lo, hi));
                }
            }
        }
        for (key, v) in arrived.into_iter().take(admitted) {
            state.casters.insert(key, v);
        }
        if fresh {
            // The full-level boxes queued above cover every caster there is.
            state.casters = current;
        }

        state.backlog = !fresh && arrived_len > admitted;
        state.diffed_rev = Some(cache.rev);
        state.caster_top = state
            .casters
            .values()
            .map(|c| c.center.y + c.half.y)
            .fold(crate::modern::gi::NO_TERRAIN, f32::max);
    }

    // --- the jobs' uniforms ---------------------------------------------------------------------
    // One box per level a frame, the union of its slabs and caster bounds: every box costs three
    // raster passes over each caster it touches, and a streaming flight queues a dozen small ones
    // whose casters overlap. Re-voxelizing voxels already right inside the union is GPU work only.
    let boxes = if std::env::var("WOW_GI_MERGE").is_ok_and(|v| v == "0") {
        boxes
    } else {
        merge_per_level(boxes)
    };
    views.0.clear();
    for (level, lo, hi) in boxes {
        let voxel = tier.voxel[level];
        let (world_lo, world_hi) = level_box(lo, hi, voxel);
        let mut axis = [0u32; 3];
        for (a, slot) in axis.iter_mut().enumerate() {
            *slot = views.0.push(&VoxelView {
                clip_from_world: box_clip(world_lo, world_hi, a),
                box_lo: lo.as_vec3().extend(voxel),
                box_hi: hi.as_vec3().extend(tier.voxels as f32),
                info: Vec4::new(
                    (level as u32 * tier.voxels) as f32,
                    VOXEL_ALPHA_KEY,
                    VOXEL_ALBEDO_MIP,
                    0.0,
                ),
            });
        }
        state.jobs.push(VoxelJob {
            level,
            lo,
            hi,
            axis,
        });
    }
    if !state.jobs.is_empty() {
        bevy::log::info_once!(
            "modern GI: first voxelization — {} box(es) this frame over {} static casters",
            state.jobs.len(),
            current_len,
        );
    }
    views.0.write_buffer(&device, &queue);
    if let Some(binding) = views.0.binding() {
        state.view_bind = Some(device.create_bind_group(
            "modern_gi_voxel_view",
            &pipeline_cache.get_bind_group_layout(&pipes.view_layout),
            &BindGroupEntries::single(binding),
        ));
    } else {
        state.jobs.clear();
    }

    // --- the raster pipelines --------------------------------------------------------------------
    for cutout in [false, true] {
        for two_sided in [false, true] {
            if pipes.raster.contains_key(&(cutout, two_sided)) {
                continue;
            }
            let mut defs: Vec<ShaderDefVal> = Vec::new();
            if cutout {
                defs.push("CUTOUT".into());
            }
            if two_sided {
                defs.push("TWO_SIDED".into());
            }
            let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("modern_gi_voxelize".into()),
                layout: vec![
                    pipes.view_layout.clone(),
                    // static_gx's own cell layout (group 1), bound unchanged.
                    gx_pipes.cell_layout.clone(),
                    pipes.store_layout.clone(),
                ],
                vertex: VertexState {
                    shader: pipes.voxel_shader.clone(),
                    shader_defs: defs.clone(),
                    entry_point: Some("vertex".into()),
                    buffers: vec![vertex_layout()],
                },
                fragment: Some(FragmentState {
                    shader: pipes.voxel_shader.clone(),
                    shader_defs: defs,
                    entry_point: Some("fragment".into()),
                    targets: vec![],
                }),
                primitive: PrimitiveState {
                    cull_mode: None,
                    ..default()
                },
                // The sweep must see every surface along the axis, so nothing is depth-tested away.
                depth_stencil: Some(DepthStencilState {
                    format: CORE_3D_DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(CompareFunction::Always),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                ..default()
            });
            pipes.raster.insert((cutout, two_sided), id);
        }
    }

    // --- the terrain heightfield -----------------------------------------------------------------
    let centre = (gi.camera / HEIGHT_SNAP).round() * HEIGHT_SNAP;
    let (clip_from_world, mut clip) = height_clip(centre);
    for level in 0..tier.levels {
        clip.level[level] = state.base[level].as_vec3().extend(tier.voxel[level]);
    }
    clip.info = Vec4::new(
        tier.levels as f32,
        tier.voxels as f32,
        tier.steps as f32,
        tier.reach,
    );
    // fork: P8b — the highest ground the map can hold: the top of every resident terrain cell whose
    // box reaches into its window. Every cell the map draws is one of these (a cell whose mesh is not
    // resident yet is simply not drawn), so no texel stands above it, and the trace skips the map
    // for any ray point higher than this (`probegi::march`). The half-yard covers the depth
    // encoding's rounding.
    clip.bounds.x = terrain
        .as_deref()
        .map_or(crate::modern::gi::NO_TERRAIN, |t| {
            t.cells
                .iter()
                .filter(|(_, c, h, _)| {
                    (c.x - centre.x).abs() <= HEIGHT_HALF + h.x
                        && (c.z - centre.z).abs() <= HEIGHT_HALF + h.z
                })
                .map(|(_, c, h, _)| c.y + h.y + 0.5)
                .fold(crate::modern::gi::NO_TERRAIN, f32::max)
        });
    let mut key = 0u64;
    if let Some(t) = terrain.as_deref() {
        // Order-free over the cells, and a fast hasher: this runs every frame over every resident
        // terrain cell.
        use std::hash::BuildHasher;
        let hasher = bevy::platform::hash::FixedHasher;
        // The LOD epoch too: a cell rebuilt at a new chunk level keeps its mesh handle.
        for (mesh_id, center, _, lod) in &t.cells {
            key = key.wrapping_add(hasher.hash_one((
                mesh_id,
                center.to_array().map(f32::to_bits),
                lod,
            )));
        }
        key = key
            .wrapping_add(hasher.hash_one(clip_from_world.to_cols_array().map(f32::to_bits)))
            .max(1);
        // A new stride arrives with a new cell, which moves the key, or with a mesh finishing its
        // upload.
        let rescan = key != state.height_key || state.height_unprepared;
        state.height_unprepared = false;
        let cells = if rescan { &t.cells[..] } else { &[][..] };
        for (mesh_id, _, _, _) in cells {
            let Some(mesh) = meshes.get(*mesh_id) else {
                state.height_unprepared = true;
                continue;
            };
            let stride = mesh.layout.0.layout().array_stride;
            if pipes.height.contains_key(&stride) {
                continue;
            }
            let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("modern_gi_heightfield".into()),
                layout: vec![pipes.height_layout.clone()],
                vertex: VertexState {
                    shader: pipes.height_shader.clone(),
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
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::GreaterEqual),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                ..default()
            });
            pipes.height.insert(stride, id);
        }
    }
    if key != 0 && key != state.height_key {
        heights.0.clear();
        state.height_offset = heights.0.push(&HeightView { clip_from_world });
        heights.0.write_buffer(&device, &queue);
        if let Some(binding) = heights.0.binding() {
            state.height_bind = Some(device.create_bind_group(
                "modern_gi_height_view",
                &pipeline_cache.get_bind_group_layout(&pipes.height_layout),
                &BindGroupEntries::single(binding),
            ));
            state.height_pending = true;
            state.height_key = key;
        }
    }

    scene.albedo = Some(albedo);
    scene.normal = Some(normal);
    scene.height = Some(height);
    // fork: the highest voxel the clipmap can hold — the top of every caster it holds, plus two of
    // the coarsest voxels for a box's rounding out to whole voxels. A ray rising above it and the
    // ground (`probegi::march`) can hit nothing more, and ends as the sky it would have reached.
    clip.bounds.y = state.caster_top + 2.0 * tier.voxel[tier.levels - 1];
    scene.clip = clip;
    // The probes trace as soon as there is a clipmap to trace against — an empty one only means an
    // open sky, which is what the sky-occlusion map says there too.
    scene.ready = true;
}

pub(crate) fn gi_voxel_pass(
    world: &World,
    view: ViewQuery<&StaticGxView>,
    mut render_context: RenderContext,
) {
    let gx_view = view.into_inner();
    {
        if !gx_view.primary {
            return;
        }
        let state = world.resource::<GiVoxelState>();
        let Some(tier) = state.tier else {
            return;
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipes = world.resource::<GiVoxelPipelines>();
        let gx = world.resource::<GxWorld>();
        let cache = world.resource::<GxGpuCache>();
        let meshes = world.resource::<RenderAssets<RenderMesh>>();
        let allocator = world.resource::<MeshAllocator>();
        // Every pass below opens a `modern_*` span: the journal's `gpu_modern` column.
        let diagnostics = render_context.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();

        // The terrain heightfield: redrawn only when the terrain or its window changed.
        if state.height_pending {
            if let (Some(height), Some(bind)) = (&state.height, &state.height_bind) {
                let terrain = world.get_resource::<ModernTerrainCasters>();
                let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                    label: Some("modern_gi_heightfield"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &height.1,
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
                let timing = diagnostics.pass_span(&mut pass, "modern_gi_heightfield");
                pass.set_bind_group(0, bind, &[state.height_offset]);
                for (mesh_id, _, _, _) in terrain.iter().flat_map(|t| t.cells.iter()) {
                    let Some(mesh) = meshes.get(*mesh_id) else {
                        continue;
                    };
                    let Some(pipeline) = pipes
                        .height
                        .get(&mesh.layout.0.layout().array_stride)
                        .and_then(|id| pipeline_cache.get_render_pipeline(*id))
                    else {
                        continue;
                    };
                    let (Some(vslice), Some(islice)) = (
                        allocator.mesh_vertex_slice(mesh_id),
                        allocator.mesh_index_slice(mesh_id),
                    ) else {
                        continue;
                    };
                    let bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } =
                        &mesh.buffer_info
                    else {
                        continue;
                    };
                    pass.set_render_pipeline(pipeline);
                    pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                    pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                    crate::static_gx::count_draw(2);
                    pass.draw_indexed(
                        islice.range.clone(),
                        i32::try_from(vslice.range.start).unwrap_or(0),
                        0..1,
                    );
                }
                timing.end(&mut pass);
            }
        }

        if state.jobs.is_empty() {
            return;
        }
        let (Some(view_bind), Some(store_bind), Some(depth)) =
            (&state.view_bind, &state.store_bind, &state.depth)
        else {
            return;
        };
        let Some(clear) = pipeline_cache.get_compute_pipeline(pipes.clear) else {
            return;
        };
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

        for job in &state.jobs {
            let voxel = tier.voxel[job.level];
            let (world_lo, world_hi) = level_box(job.lo, job.hi, voxel);
            let span = (job.hi - job.lo).max(IVec3::ONE);
            {
                let encoder = render_context.command_encoder();
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("modern_gi_voxel_clear"),
                    timestamp_writes: None,
                });
                let timing = diagnostics.pass_span(&mut pass, "modern_gi_voxel_clear");
                pass.set_pipeline(clear);
                pass.set_bind_group(0, view_bind, &[job.axis[0]]);
                pass.set_bind_group(1, store_bind, &[]);
                pass.dispatch_workgroups(
                    (span.x as u32).div_ceil(4),
                    (span.y as u32).div_ceil(4),
                    (span.z as u32).div_ceil(4),
                );
                timing.end(&mut pass);
            }
            for offset in &job.axis {
                let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                    label: Some("modern_gi_voxelize"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: depth,
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
                let timing = diagnostics.pass_span(&mut pass, "modern_gi_voxelize");
                pass.set_bind_group(0, view_bind, &[*offset]);
                pass.set_bind_group(2, store_bind, &[]);
                for (gpu, draw) in &casters {
                    let center = draw.origin + Vec3::from(draw.aabb.center);
                    if !overlaps(
                        center,
                        Vec3::from(draw.aabb.half_extents),
                        world_lo,
                        world_hi,
                    ) {
                        continue;
                    }
                    let Some(vslice) = allocator.mesh_vertex_slice(&draw.mesh.id()) else {
                        continue;
                    };
                    pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                    gpu.draw_whole(
                        &mut pass,
                        vslice.range.start,
                        |cut, two| {
                            pipes
                                .raster
                                .get(&(cut, two))
                                .and_then(|id| pipeline_cache.get_render_pipeline(*id))
                        },
                        2,
                    );
                }
                timing.end(&mut pass);
            }
        }
    }
}

/// Wire the pass between the modern shadow pass and the probe trace (called from `render::build`).
pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .init_resource::<GiVoxelState>()
        .init_resource::<VoxelViewBuffer>()
        .init_resource::<HeightViewBuffer>()
        .add_systems(
            bevy::render::RenderStartup,
            init_gi_pipelines.after(init_pipelines),
        )
        .add_systems(
            Render,
            prepare_gi_voxels.in_set(RenderSystems::PrepareResources),
        )
        .add_systems(Core3d, gi_voxel_pass.before(Core3dSystems::MainPass));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A room adds a probe cascade (`GiTier::with_interior`) and nothing the clipmap is built from,
    /// so stepping in and out of rooms keeps its textures on every tier.
    #[test]
    fn the_interior_cascade_keeps_the_clipmap() {
        use crate::modern::LightingQuality as Q;
        for quality in [Q::Low, Q::Medium, Q::High, Q::Ultra, Q::Cinematic] {
            let tier = gi::gi_tier(quality);
            assert_eq!(
                clipmap_shape(&tier),
                clipmap_shape(&tier.with_interior()),
                "{quality:?}"
            );
        }
    }

    #[test]
    fn a_frames_boxes_merge_to_one_union_per_level() {
        let boxes = vec![
            (1, IVec3::new(0, 0, 0), IVec3::new(4, 4, 4)),
            (0, IVec3::new(10, 0, 0), IVec3::new(12, 64, 64)),
            (1, IVec3::new(8, -2, 1), IVec3::new(9, 3, 2)),
            (0, IVec3::new(0, 60, 0), IVec3::new(64, 62, 64)),
        ];
        assert_eq!(
            merge_per_level(boxes),
            vec![
                (1, IVec3::new(0, -2, 0), IVec3::new(9, 4, 4)),
                (0, IVec3::new(0, 0, 0), IVec3::new(64, 64, 64)),
            ]
        );
        assert!(merge_per_level(Vec::new()).is_empty());
    }

    #[test]
    fn a_scrolled_window_only_redraws_what_it_gained() {
        let mut out = Vec::new();
        scroll_jobs(IVec3::ZERO, IVec3::new(8, 0, 0), 64, &mut out);
        assert_eq!(out.len(), 1);
        let (lo, hi) = out[0];
        assert_eq!(lo, IVec3::new(64, 0, 0));
        assert_eq!(hi, IVec3::new(72, 64, 64));
        // A jump further than the window is the whole window.
        out.clear();
        scroll_jobs(IVec3::ZERO, IVec3::new(400, 0, 0), 64, &mut out);
        assert_eq!(out, vec![(IVec3::new(400, 0, 0), IVec3::new(464, 64, 64))]);
        // Two axes at once gain one slab each (their corner is redone twice, which is free).
        out.clear();
        scroll_jobs(IVec3::ZERO, IVec3::new(-8, 0, 8), 64, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_box_camera_maps_its_box_onto_the_whole_target() {
        let lo = Vec3::new(-96.0, 12.0, 30.0);
        let hi = Vec3::new(0.0, 108.0, 126.0);
        for axis in 0..3 {
            let clip = box_clip(lo, hi, axis);
            let a = clip.project_point3(lo);
            let b = clip.project_point3(hi);
            // The box's corners land on the clip volume's corners, depth included.
            assert!(
                (a.x.abs() - 1.0).abs() < 1e-4 && (b.x.abs() - 1.0).abs() < 1e-4,
                "{axis}: {a} {b}"
            );
            assert!(
                (a.y.abs() - 1.0).abs() < 1e-4 && (b.y.abs() - 1.0).abs() < 1e-4,
                "{axis}: {a} {b}"
            );
            let depth = [a.z, b.z];
            assert!(
                depth.contains(&0.0) || depth.iter().any(|z| z.abs() < 1e-4),
                "{axis}: {a} {b}"
            );
            assert!(
                depth.iter().any(|z| (z - 1.0).abs() < 1e-4),
                "{axis}: {a} {b}"
            );
            // The centre stays in the middle of the target.
            let mid = clip.project_point3(0.5 * (lo + hi));
            assert!(mid.x.abs() < 1e-4 && mid.y.abs() < 1e-4, "{axis}: {mid}");
        }
    }

    #[test]
    fn the_heightfield_reads_back_the_height_it_wrote() {
        let (clip_from_world, clip) = height_clip(Vec3::new(64.0, 50.0, -32.0));
        let p = Vec3::new(64.0 + 100.0, 73.0, -32.0 - 40.0);
        let ndc = clip_from_world.project_point3(p);
        // The rendering clip and the sampling rows agree on where the point lands.
        let uv_from_ndc = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let q = p.extend(1.0);
        let uv_from_rows = Vec2::new(clip.hf_row_x.dot(q), clip.hf_row_y.dot(q));
        assert!(
            (uv_from_ndc - uv_from_rows).abs().max_element() < 1e-5,
            "{uv_from_ndc} {uv_from_rows}"
        );
        // And the stored depth decodes back to the height.
        let height = clip.hf.y + ndc.z * clip.hf.z;
        assert!((height - p.y).abs() < 1e-2, "{height} vs {}", p.y);
    }
}
