//! The render-world half of the probe GI: the block's GI parameters, the trace and blend compute
//! passes, and their place in the graph (after the modern shadow pass, whose cascades the trace
//! samples, and before the froxels, which read the probes).
//!
//! fork: perf — with the async compute queue (`modern::async_compute`; native, not
//! `$WOW_ASYNC_GI=0`) the classify, trace and blend passes record on it instead
//! ([`gi_trace_pass_async`]), and the field they keep is a buffer of their own
//! ([`GiFrame::field`]), copied into the block's probe region at the next frame's start
//! ([`copy_gi_field`]): every reader of the probes then reads the field one frame late, and nothing on
//! the main queue waits for the trace. The block's probe region is the whole field state either way
//! (the trace's "previous field" reads it, classify and blend the field), so the probes evolve
//! exactly as on the main queue, one frame later where they are read.

use bevy::app::SubApp;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::diagnostic::DiagnosticsRecorder;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d_array, texture_3d,
    texture_depth_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy::render::renderer::{AsyncComputeContext, RenderAsyncComputeQueue};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderStartup, RenderSystems};

use benilla_assets::materials::{MODERN_SHADOW_CASCADES, MODERN_SHADOW_MAP};

use super::{ModernGi, ModernGiScene, GI_PARAMS_BYTES, PROBE_BYTES};
use crate::lighting::SharedLightBuffer;
use crate::modern::shadow::CASCADES_BYTES;
use crate::modern::volumetrics::BLOCK_TAIL_BYTES;
use crate::modern::ActiveLighting;

/// Where the probe array starts inside the modern storage block. 256-aligned by construction (the
/// `GiParams` block and the P6 light grid in front of it are both sized for it), which is what lets
/// the blend pass bind the probes alone.
pub const PROBE_OFFSET: u64 =
    crate::modern::lights::node::LIGHT_OFFSET + crate::modern::lights::BLOCK_LIGHT_BYTES as u64;
/// Where the block's `GiParams` sit.
const PARAMS_OFFSET: u64 = (CASCADES_BYTES + BLOCK_TAIL_BYTES) as u64;

/// fork: perf — [`GiPipelines::async_trace`] for the session, any world; false before render
/// startup and always on wasm32.
static ASYNC_TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// fork: perf — whether the probe trace records on the async compute queue this session: the perf
/// probe's `PERF_OVERLAP` then counts its spans as async ones.
pub(crate) fn async_trace() -> bool {
    cfg!(not(target_arch = "wasm32")) && ASYNC_TRACE.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Resource)]
struct GiPipelines {
    classify_layout: BindGroupLayoutDescriptor,
    trace_layout: BindGroupLayoutDescriptor,
    blend_layout: BindGroupLayoutDescriptor,
    classify: CachedComputePipelineId,
    /// The trace's three stages (`gi_trace.wgsl`): per probe, per ray, per probe again.
    setup: CachedComputePipelineId,
    trace: CachedComputePipelineId,
    sum: CachedComputePipelineId,
    blend: CachedComputePipelineId,
    /// fork: the ray stage as a ray query against the ray-traced scene (`gi_trace_rt.wgsl`), queued
    /// where the session may trace probes ([`super::rt_gi_capable`]) and used in place of `trace`
    /// in a frame [`super::rt_gi`] holds; `None` otherwise.
    rt_trace: Option<CachedComputePipelineId>,
    /// fork: F — the voxel path's `trace` declares the scene at group 1 for its hits' sun term
    /// (`RT_SHADOWS`) wherever the session may trace shadows
    /// (`modern::shadow::rt_shadows_capable`): the voxel trace runs whenever the probe rays are not
    /// live, and it is bound the scene while the shadows are (`$WOW_RT_GI=0`; the cascades then hold
    /// no static world) and the stand-in otherwise, on which `rtshadow` reads the cascades alone.
    voxel_rt: bool,
    /// fork: perf — the classify, trace and blend passes record on the async compute queue
    /// ([`gi_trace_pass_async`]): the queue exists (`modern::async_compute`) and `$WOW_ASYNC_GI` is
    /// not "0". Decided at render startup, for the session.
    #[cfg(not(target_arch = "wasm32"))]
    async_trace: bool,
}

/// The trace's dispatch is rows of this many workgroups (`gi_classify.wgsl`'s `GROUP_ROW`): one
/// axis of a dispatch stops at 65535 workgroups, and a frame listing every probe of Ultra's field at
/// one a workgroup is 65536 — dropped whole, which left every probe cold for good.
const GROUP_ROW: u32 = 256;
/// The indirect dispatch's reset value: rows of [`GROUP_ROW`], none of them yet, `z` one.
const DISPATCH_RESET: [u32; 3] = [GROUP_ROW, 0, 1];

#[derive(Resource, Default)]
struct GiFrame {
    uniform: Option<Buffer>,
    scratch: Option<Buffer>,
    scratch_probes: usize,
    /// The trace's per-probe setup records and per-ray results, sized for a frame's listed cap.
    setups: Option<Buffer>,
    rays: Option<Buffer>,
    /// The work list `gi_classify.wgsl` packs — a count, then one probe index per listed probe —
    /// and the indirect dispatch it grows. Both are reset every frame before the passes run.
    work: Option<Buffer>,
    dispatch: Option<Buffer>,
    classify_bind: Option<BindGroup>,
    trace_bind: Option<BindGroup>,
    blend_bind: Option<BindGroup>,
    /// fork: the ray-traced scene's bind group (group 1 of the ray-query stage) where this frame's
    /// rays are ray queries ([`super::rt_gi`]); `None` on the voxel path.
    rt_bind: Option<BindGroup>,
    /// fork: F — the scene, or its stand-in, for the voxel path's `trace` where
    /// [`GiPipelines::voxel_rt`]; `None` otherwise.
    shadow_bind: Option<BindGroup>,
    /// fork: a 1x1x1 stand-in for the voxel clipmap's two textures, which the ray-traced tier never
    /// builds: the trace's group 0 still has their slots, and only the voxel stage reads them.
    vox_stub: Option<TextureView>,
    /// fork: last frame's [`super::rt_gi`]; a switch cuts the field ([`Self::cut_frames`]), so no
    /// probe keeps light the other path traced.
    rt_was: Option<bool>,
    /// fork: frames still to trace as a cut (every probe traced and replaced, as after a teleport):
    /// one probe period from a switch of the ray stage.
    cut_frames: u32,
    /// Workgroups of the classify pass: one thread per probe of the tier.
    groups: u32,
    /// fork: perf — the probe field of the async-queue trace ([`GiPipelines::async_trace`]), the
    /// tier's probes at [`PROBE_BYTES`] each: classify reads it and the blend writes it, in place
    /// of the block's probe region, and [`copy_gi_field`] copies it into that region at the next
    /// frame's start, before the probes' readers. `None` on the main-queue trace.
    #[cfg(not(target_arch = "wasm32"))]
    field: Option<Buffer>,
    /// fork: perf — a trace was recorded into [`Self::field`] since [`copy_gi_field`] last copied
    /// it into the block.
    #[cfg(not(target_arch = "wasm32"))]
    field_ahead: bool,
    /// fork: perf — [`Self::field`] is new (the first traced frame, or a tier change resized it):
    /// [`copy_gi_field`] fills it from the block's probe region before its first trace.
    #[cfg(not(target_arch = "wasm32"))]
    field_seed: bool,
    /// fork: perf — a field a resize replaced while it was ahead of the block: copied into the block
    /// before the new field is filled from it.
    #[cfg(not(target_arch = "wasm32"))]
    field_retired: Option<Buffer>,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    #[cfg(not(target_arch = "wasm32"))] async_queue: Option<Res<RenderAsyncComputeQueue>>,
) {
    let trace_layout = BindGroupLayoutDescriptor::new(
        "modern_gi_trace",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                texture_2d_array(TextureSampleType::Depth),
                sampler(SamplerBindingType::Comparison),
                storage_buffer_read_only_sized(false, None),
                texture_3d(TextureSampleType::Float { filterable: false }),
                texture_3d(TextureSampleType::Float { filterable: false }),
                texture_depth_2d(),
                uniform_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                // The work list.
                storage_buffer_read_only_sized(false, None),
                // The setup records and the ray results.
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let blend_layout = BindGroupLayoutDescriptor::new(
        "modern_gi_blend",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                // The work list.
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );
    // The probe region (read), the trace's parameters, the work list and its dispatch (written).
    let classify_layout = BindGroupLayoutDescriptor::new(
        "modern_gi_classify",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                uniform_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let compute = |label: &'static str,
                   layout: &BindGroupLayoutDescriptor,
                   path: &'static str,
                   entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            constants: Default::default(),
            label: Some(label.into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(path),
            shader_defs: vec![],
            entry_point: Some(entry.into()),
            zero_initialize_workgroup_memory: false,
        })
    };
    let classify = compute(
        "modern_gi_classify",
        &classify_layout,
        "embedded://benilla_world/shaders/modern/gi_classify.wgsl",
        "classify_probes",
    );
    let setup = compute(
        "modern_gi_setup",
        &trace_layout,
        "embedded://benilla_world/shaders/modern/gi_trace.wgsl",
        "setup_probes",
    );
    // fork: F — the voxel path's hits take their sun through the scene, where the session may trace
    // shadows: a frame whose probe rays are not ray queries runs this trace, live shadows or not
    // (native only).
    #[cfg(not(target_arch = "wasm32"))]
    let voxel_rt = crate::modern::shadow::rt_shadows_capable();
    #[cfg(target_arch = "wasm32")]
    let voxel_rt = false;
    #[cfg(not(target_arch = "wasm32"))]
    let trace = if voxel_rt {
        let mut shader_defs = crate::static_gx::rt_scene::shader_defs(1);
        shader_defs.push(bevy::shader::ShaderDefVal::from("RT_SHADOWS"));
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            constants: Default::default(),
            label: Some("modern_gi_trace".into()),
            layout: vec![trace_layout.clone(), crate::static_gx::rt_scene::layout()],
            immediate_size: 0,
            shader: asset_server.load("embedded://benilla_world/shaders/modern/gi_trace.wgsl"),
            shader_defs,
            entry_point: Some("trace_rays".into()),
            zero_initialize_workgroup_memory: false,
        })
    } else {
        compute(
            "modern_gi_trace",
            &trace_layout,
            "embedded://benilla_world/shaders/modern/gi_trace.wgsl",
            "trace_rays",
        )
    };
    #[cfg(target_arch = "wasm32")]
    let trace = compute(
        "modern_gi_trace",
        &trace_layout,
        "embedded://benilla_world/shaders/modern/gi_trace.wgsl",
        "trace_rays",
    );
    let sum = compute(
        "modern_gi_sum",
        &trace_layout,
        "embedded://benilla_world/shaders/modern/gi_trace.wgsl",
        "sum_probes",
    );
    let blend = compute(
        "modern_gi_blend",
        &blend_layout,
        "embedded://benilla_world/shaders/modern/gi_blend.wgsl",
        "blend_probes",
    );
    // fork: the ray-query stage, over the same group 0 plus the scene at group 1, queued beside the
    // voxel one so a switch picks between two compiled pipelines (native only).
    #[cfg(not(target_arch = "wasm32"))]
    let rt_trace = super::rt_gi_capable().then(|| {
        let mut shader_defs = crate::static_gx::rt_scene::shader_defs(1);
        // fork: F — the hits' sun through the scene too, where the cascades hold entities alone
        // (this pipeline runs only on a live scene, so the shadows are live with it).
        if crate::modern::shadow::rt_shadows_capable() {
            shader_defs.push(bevy::shader::ShaderDefVal::from("RT_SHADOWS"));
        }
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            constants: Default::default(),
            label: Some("modern_gi_trace_rt".into()),
            layout: vec![trace_layout.clone(), crate::static_gx::rt_scene::layout()],
            immediate_size: 0,
            shader: asset_server.load("embedded://benilla_world/shaders/modern/gi_trace_rt.wgsl"),
            shader_defs,
            entry_point: Some("trace_rays".into()),
            zero_initialize_workgroup_memory: false,
        })
    });
    #[cfg(target_arch = "wasm32")]
    let rt_trace: Option<CachedComputePipelineId> = None;
    if rt_trace.is_some() {
        info!(
            "modern GI: probe rays are ray queries against the TLAS while ray tracing is live \
             (WOW_RT_GI=0 turns it off)"
        );
    }
    // fork: perf — the trace on the async compute queue wherever there is one (`async_compute::init`
    // runs first), unless `$WOW_ASYNC_GI=0` keeps it on the main queue.
    #[cfg(not(target_arch = "wasm32"))]
    let async_trace =
        async_queue.is_some() && crate::launch_env::var("WOW_ASYNC_GI").as_deref() != Some("0");
    #[cfg(not(target_arch = "wasm32"))]
    {
        ASYNC_TRACE.store(async_trace, std::sync::atomic::Ordering::Relaxed);
        if async_trace {
            info!(
                "modern GI: the probe trace runs on the async compute queue, its field read a \
                 frame late (WOW_ASYNC_GI=0 keeps it on the main queue)"
            );
        }
    }
    commands.insert_resource(GiPipelines {
        classify_layout,
        trace_layout,
        blend_layout,
        classify,
        setup,
        trace,
        sum,
        blend,
        rt_trace,
        voxel_rt,
        #[cfg(not(target_arch = "wasm32"))]
        async_trace,
    });
}

/// Write the block's GI parameters, and get the two compute passes' buffers and bind groups ready.
/// The parameters are written every modern frame, GI on or off: a lane reads `params.x` to decide
/// whether a probe field exists at all, so it has to be current even when it does not.
#[allow(clippy::too_many_arguments)]
fn prepare_gi(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    active: Option<Res<ActiveLighting>>,
    gi: Option<Res<ModernGi>>,
    scene: Res<ModernGiScene>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    light: Option<Res<SharedLightBuffer>>,
    pipes: Option<Res<GiPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    mut frame: ResMut<GiFrame>,
    #[cfg(not(target_arch = "wasm32"))] rt_scene: Option<Res<crate::static_gx::rt_scene::RtScene>>,
) {
    frame.classify_bind = None;
    frame.trace_bind = None;
    frame.blend_bind = None;
    frame.rt_bind = None;
    frame.shadow_bind = None;
    frame.groups = 0;
    let (Some(gi), Some(block)) = (gi, buffers.get(MODERN_SHADOW_CASCADES.id())) else {
        return;
    };
    let modern = active.is_some_and(|a| a.is_modern());
    // fork: the ray-traced trace ([`super::rt_gi`], this frame's decision) casts against the
    // scene's TLAS, whose bind group exists once the TLAS holds an instance (`RtScene::bind_group`).
    let rt = super::rt_gi();
    // fork: a switch of the ray stage, either way, cuts the field for a probe period: every probe
    // is traced again on the new path and replaces its history, as after a teleport.
    if frame.rt_was.is_some_and(|was| was != rt) {
        frame.cut_frames = gi.tier.period.max(1);
    }
    frame.rt_was = Some(rt);
    #[cfg(not(target_arch = "wasm32"))]
    let rt_bind = rt
        .then(|| rt_scene.as_deref().and_then(|s| s.bind_group().cloned()))
        .flatten();
    #[cfg(target_arch = "wasm32")]
    let rt_bind: Option<BindGroup> = None;
    // Until then there is nothing to trace against (this tier builds no clipmap): the field is
    // off and the lanes keep the sky map, as on a frame with no voxel model.
    let on = modern && gi.on && scene.ready && (!rt || rt_bind.is_some());

    // The probes' own sky is the shared light's ambient row; the CPU copy here only has to agree
    // with it closely enough for the froxels' reference (the trace reads the row itself).
    let params = super::params(
        &gi,
        gi.camera,
        Vec3::ZERO,
        f32::from(crate::modern::shadow_debug_mode()),
    );
    let mut params = params;
    if !on {
        params.params.x = 0.0;
    }
    let mut bytes = Vec::with_capacity(GI_PARAMS_BYTES);
    params.std430(&mut bytes);
    if block.buffer.size() >= PARAMS_OFFSET + bytes.len() as u64 {
        queue.write_buffer(&block.buffer, PARAMS_OFFSET, &bytes);
    }
    if !on {
        return;
    }

    let probes = gi.tier.probes();
    let probe_bytes = (probes * PROBE_BYTES) as u64;
    if block.buffer.size() < PROBE_OFFSET + probe_bytes {
        warn_once!("modern GI: the storage block is too small for {probes} probes");
        return;
    }
    // fork: the voxel clipmap's two textures, or at the ray-traced tier, which builds none, the
    // stand-in that fills their slots.
    // Borrowed on the voxel path, exactly as before; the stand-in is touched only at the tier.
    let stub = rt.then(|| {
        frame
            .vox_stub
            .get_or_insert_with(|| vox_stub(&device))
            .clone()
    });
    let voxels = match &stub {
        Some(stub) => Some((stub, stub)),
        None => scene
            .albedo
            .as_ref()
            .zip(scene.normal.as_ref())
            .map(|(a, n)| (&a.1, &n.1)),
    };
    let inputs = (
        pipes.as_deref(),
        light.as_deref(),
        images.get(MODERN_SHADOW_MAP.id()),
        voxels,
        scene.height.as_ref(),
    );
    let (Some(pipes), Some(light), Some(shadow), Some((albedo, normal)), Some(height)) = inputs
    else {
        return;
    };
    let ray_stage = if rt {
        pipes.rt_trace
    } else {
        Some(pipes.trace)
    };
    if pipeline_cache
        .get_compute_pipeline(pipes.classify)
        .is_none()
        || ray_stage
            .and_then(|id| pipeline_cache.get_compute_pipeline(id))
            .is_none()
        || pipeline_cache.get_compute_pipeline(pipes.blend).is_none()
    {
        return;
    }

    // fork: the switch's cut rides the teleport's (`ModernGi::cut`).
    let mut traced = *gi;
    traced.cut |= frame.cut_frames > 0;
    frame.cut_frames = frame.cut_frames.saturating_sub(1);
    let uniform_bytes = super::trace_bytes(&traced, &params, &scene.clip);
    let uniform = match &frame.uniform {
        Some(b) if b.size() >= uniform_bytes.len() as u64 => b.clone(),
        _ => {
            let b = device.create_buffer(&BufferDescriptor {
                label: Some("modern_gi_params"),
                size: uniform_bytes.len() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            frame.uniform = Some(b.clone());
            b
        }
    };
    queue.write_buffer(&uniform, 0, &uniform_bytes);

    if frame.scratch.is_none() || frame.scratch_probes != probes {
        frame.scratch = Some(device.create_buffer(&BufferDescriptor {
            label: Some("modern_gi_scratch"),
            size: probe_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        frame.scratch_probes = probes;
    }
    let Some(scratch) = frame.scratch.clone() else {
        return;
    };
    // A frame lists at most `total / period` probes (`probegi::listed_cap`): one setup record each
    // (64 bytes, `GiSetup`) and two rows a ray (32 bytes).
    let cap = (probes / gi.tier.period.max(1) as usize).max(1) as u64;
    let sized = |b: &Option<Buffer>, bytes: u64| b.as_ref().is_some_and(|b| b.size() == bytes);
    let setup_bytes = cap * 64;
    if !sized(&frame.setups, setup_bytes) {
        frame.setups = Some(device.create_buffer(&BufferDescriptor {
            label: Some("modern_gi_setups"),
            size: setup_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    let ray_bytes = cap * u64::from(gi.tier.rays) * 32;
    if !sized(&frame.rays, ray_bytes) {
        frame.rays = Some(device.create_buffer(&BufferDescriptor {
            label: Some("modern_gi_rays"),
            size: ray_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    let (Some(setups), Some(ray_out)) = (frame.setups.clone(), frame.rays.clone()) else {
        return;
    };
    // The work list holds at most every probe of the allocation, so it is sized once and never
    // re-created by a tier change; the dispatch is three words. Both are reset here, every frame the
    // passes run: `write_buffer` lands before the frame's command buffers, so the classify pass
    // always starts from an empty list.
    let work = frame
        .work
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("modern_gi_work"),
                size: 4 + 4 * super::PROBE_ALLOC as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    let dispatch = frame
        .dispatch
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("modern_gi_dispatch"),
                size: 12,
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    queue.write_buffer(&work, 0, &0u32.to_le_bytes());
    queue.write_buffer(&dispatch, 0, bytemuck::cast_slice(&DISPATCH_RESET));
    // The probe region alone — the same range the blend pass binds writable.
    let probe_region = || {
        BindingResource::Buffer(BufferBinding {
            buffer: &block.buffer,
            offset: PROBE_OFFSET,
            size: Some(std::num::NonZero::new(probe_bytes).expect("probe region is never empty")),
        })
    };
    // fork: perf — on the async queue, classify and the blend keep the trace's own field: a write
    // into the block there made every main submission that binds the block (every world material,
    // the froxels) wait for the trace, since wgpu orders the two queues per buffer.
    #[cfg(not(target_arch = "wasm32"))]
    let field = pipes.async_trace.then(|| {
        if frame.field.as_ref().is_none_or(|f| f.size() != probe_bytes) {
            let old = frame.field.replace(device.create_buffer(&BufferDescriptor {
                label: Some("modern_gi_field"),
                size: probe_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            if std::mem::take(&mut frame.field_ahead) {
                frame.field_retired = old;
            }
            frame.field_seed = true;
        }
        frame.field.clone().expect("made above")
    });
    #[cfg(target_arch = "wasm32")]
    let field: Option<Buffer> = None;
    let probes_binding = || match &field {
        Some(field) => field.as_entire_binding(),
        None => probe_region(),
    };

    frame.classify_bind = Some(device.create_bind_group(
        "modern_gi_classify",
        &pipeline_cache.get_bind_group_layout(&pipes.classify_layout),
        &BindGroupEntries::sequential((
            probes_binding(),
            uniform.as_entire_binding(),
            work.as_entire_binding(),
            dispatch.as_entire_binding(),
        )),
    ));
    frame.trace_bind = Some(device.create_bind_group(
        "modern_gi_trace",
        &pipeline_cache.get_bind_group_layout(&pipes.trace_layout),
        &BindGroupEntries::sequential((
            block.buffer.as_entire_binding(),
            &shadow.texture_view,
            &shadow.sampler,
            light.0.as_entire_binding(),
            albedo,
            normal,
            &height.1,
            uniform.as_entire_binding(),
            scratch.as_entire_binding(),
            work.as_entire_binding(),
            setups.as_entire_binding(),
            ray_out.as_entire_binding(),
        )),
    ));
    frame.blend_bind = Some(device.create_bind_group(
        "modern_gi_blend",
        &pipeline_cache.get_bind_group_layout(&pipes.blend_layout),
        &BindGroupEntries::sequential((
            uniform.as_entire_binding(),
            scratch.as_entire_binding(),
            // The probe region of the block alone: the rest of it is read-only storage in the same
            // frame's render passes, and a writable binding is exclusive per buffer, not per range.
            // fork: perf — the trace's own field on the async queue.
            probes_binding(),
            work.as_entire_binding(),
        )),
    ));
    frame.rt_bind = rt_bind;
    // fork: F — the voxel path's scene for its hits' sun term while the shadows are live, else the
    // stand-in, on which `rtshadow` reads the cascades alone (they hold the static world again).
    #[cfg(not(target_arch = "wasm32"))]
    if pipes.voxel_rt {
        frame.shadow_bind = rt_scene
            .as_deref()
            .and_then(|s| s.lane_bind(crate::modern::shadow::rt_shadows()).cloned());
    }
    frame.groups = (probes as u32).div_ceil(64);
    bevy::log::info_once!(
        "modern GI: probes live — {} in {} cascade(s), {} rays, 1/{} traced a frame",
        probes,
        gi.tier.cascades,
        gi.tier.rays,
        gi.tier.period,
    );
}

/// fork: the stand-in view for the voxel clipmap's slots in the trace's group 0 at the ray-traced
/// tier: a 1x1x1 `Rgba8Unorm` 3D texture, the clipmap's own format and dimension, never read.
fn vox_stub(device: &RenderDevice) -> TextureView {
    device
        .create_texture(&TextureDescriptor {
            label: Some("modern_gi_voxel_stub"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default())
}

/// What a frame's trace records, from [`prepare_gi`]'s bind groups: `None` when there is nothing
/// to trace or a pipeline is not ready yet, so nothing is recorded.
struct TracePlan<'a> {
    classify_bind: &'a BindGroup,
    trace_bind: &'a BindGroup,
    blend_bind: &'a BindGroup,
    dispatch: &'a Buffer,
    groups: u32,
    classify: &'a ComputePipeline,
    setup: &'a ComputePipeline,
    trace: &'a ComputePipeline,
    sum: &'a ComputePipeline,
    blend: &'a ComputePipeline,
    /// fork: at the ray-traced tier the ray stage is the ray query, with the scene at group 1.
    rt_stage: Option<(&'a BindGroup, &'a ComputePipeline)>,
    /// fork: F — the voxel path's scene for its hits' sun term (`GiPipelines::voxel_rt`).
    shadow_bind: Option<&'a BindGroup>,
}

impl<'a> TracePlan<'a> {
    fn new(
        frame: &'a GiFrame,
        pipes: &GiPipelines,
        pipeline_cache: &'a PipelineCache,
    ) -> Option<Self> {
        let (Some(classify_bind), Some(trace_bind), Some(blend_bind), Some(dispatch)) = (
            &frame.classify_bind,
            &frame.trace_bind,
            &frame.blend_bind,
            &frame.dispatch,
        ) else {
            return None;
        };
        if frame.groups == 0 {
            return None;
        }
        let (Some(classify), Some(setup), Some(trace), Some(sum), Some(blend)) = (
            pipeline_cache.get_compute_pipeline(pipes.classify),
            pipeline_cache.get_compute_pipeline(pipes.setup),
            pipeline_cache.get_compute_pipeline(pipes.trace),
            pipeline_cache.get_compute_pipeline(pipes.sum),
            pipeline_cache.get_compute_pipeline(pipes.blend),
        ) else {
            return None;
        };
        // fork: F — a voxel trace that declares the scene cannot run without it.
        if pipes.voxel_rt && frame.shadow_bind.is_none() && frame.rt_bind.is_none() {
            return None;
        }
        // fork: at the ray-traced tier the ray stage is the ray query, with the scene at group 1.
        let rt_stage = match &frame.rt_bind {
            Some(bind) => Some((
                bind,
                pipes
                    .rt_trace
                    .and_then(|id| pipeline_cache.get_compute_pipeline(id))?,
            )),
            None => None,
        };
        Some(Self {
            classify_bind,
            trace_bind,
            blend_bind,
            dispatch,
            groups: frame.groups,
            classify,
            setup,
            trace,
            sum,
            blend,
            rt_stage,
            shadow_bind: frame.shadow_bind.as_ref(),
        })
    }

    /// The classify, trace and blend passes into `encoder`, either queue's, each timed under its
    /// span (the journal's `gpu_modern` column, where the device times passes).
    fn record(self, encoder: &mut CommandEncoder, diagnostics: Option<&DiagnosticsRecorder>) {
        // Classify sweeps every probe and packs the ones due a trace; the trace and the blend then
        // launch one thread per packed probe (`gi_classify.wgsl` says why that is the whole win).
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_gi_classify"),
                timestamp_writes: None,
            });
            let span = diagnostics.pass_span(&mut pass, "modern_gi_classify");
            pass.set_pipeline(self.classify);
            pass.set_bind_group(0, self.classify_bind, &[]);
            pass.dispatch_workgroups(self.groups, 1, 1);
            span.end(&mut pass);
        }
        {
            // fork: a span of its own at the ray-traced tier, so the two paths compare by name.
            let label = if self.rt_stage.is_some() {
                "modern_gi_trace_rt"
            } else {
                "modern_gi_trace"
            };
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            let span = diagnostics.pass_span(&mut pass, label);
            pass.set_bind_group(0, self.trace_bind, &[]);
            // Setup, rays, sums — one dispatch each over the same rows (`gi_trace.wgsl`).
            pass.set_pipeline(self.setup);
            pass.dispatch_workgroups_indirect(self.dispatch, 0);
            match self.rt_stage {
                Some((rt_bind, rt_trace)) => {
                    pass.set_pipeline(rt_trace);
                    pass.set_bind_group(1, rt_bind, &[]);
                }
                None => {
                    pass.set_pipeline(self.trace);
                    // fork: F — the voxel path's hits trace their sun (`GiPipelines::voxel_rt`).
                    if let Some(bind) = self.shadow_bind {
                        pass.set_bind_group(1, bind, &[]);
                    }
                }
            }
            pass.dispatch_workgroups_indirect(self.dispatch, 0);
            pass.set_pipeline(self.sum);
            pass.dispatch_workgroups_indirect(self.dispatch, 0);
            span.end(&mut pass);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_gi_blend"),
                timestamp_writes: None,
            });
            let span = diagnostics.pass_span(&mut pass, "modern_gi_blend");
            pass.set_pipeline(self.blend);
            pass.set_bind_group(0, self.blend_bind, &[]);
            pass.dispatch_workgroups_indirect(self.dispatch, 0);
            span.end(&mut pass);
        }
    }
}

/// The probe trace, on the world camera's view alone: the field is world-space, and the water
/// mirror and the booths, which read none of it, each ran it again ([`world_view_only`]).
///
/// [`world_view_only`]: crate::modern::world_view_only
pub(crate) fn gi_trace_pass(
    world: &World,
    view: ViewQuery<Has<crate::modern::volumetrics::ModernFogView>>,
    mut render_context: RenderContext,
) {
    if !crate::modern::world_view_only(view.into_inner()) {
        return;
    }
    let Some(pipes) = world.get_resource::<GiPipelines>() else {
        return;
    };
    // fork: perf — the async queue's pass records it instead (`gi_trace_pass_async`).
    #[cfg(not(target_arch = "wasm32"))]
    if pipes.async_trace {
        return;
    }
    let pipeline_cache = world.resource::<PipelineCache>();
    let Some(plan) = TracePlan::new(world.resource::<GiFrame>(), pipes, pipeline_cache) else {
        return;
    };
    let diagnostics = render_context.diagnostic_recorder();
    plan.record(render_context.command_encoder(), diagnostics.as_deref());
}

/// fork: perf — [`gi_trace_pass`] on the async compute queue ([`GiPipelines::async_trace`]), in
/// `MainPass` after `static_gx_pass` and before bevy's opaque pass, so it runs beside the rest of
/// the frame's raster and post-processing: every input is ready by then (this frame's cascades and
/// lamp atlas from `gx_shadow_pass`, the voxels and heightfield from `gi_voxel_pass`, the TLAS from
/// `rt_scene_pass`, the light grid from `copy_light_grid`), the async submission waits for every
/// main submission before it, and nothing after it on the main queue writes what it reads or reads
/// what it writes (it writes only [`GiFrame::field`] and its own scratch). After `static_gx_pass`
/// rather than in front of it: the ray-traced scene's group binds static_gx's index arena as
/// storage, which static_gx's draws use as an index buffer, a different state, so a trace submitted
/// in front of them made them wait for it. The frame's last submission waits for it
/// (`bevy_render`'s `wait_for_async_compute`).
#[cfg(not(target_arch = "wasm32"))]
fn gi_trace_pass_async(
    mut frame: ResMut<GiFrame>,
    pipes: Res<GiPipelines>,
    pipeline_cache: Res<PipelineCache>,
    view: ViewQuery<Has<crate::modern::volumetrics::ModernFogView>>,
    mut async_compute: AsyncComputeContext,
) {
    if !pipes.async_trace
        || !async_compute.available()
        || !crate::modern::world_view_only(view.into_inner())
    {
        return;
    }
    let Some(plan) = TracePlan::new(&frame, &pipes, &pipeline_cache) else {
        return;
    };
    let diagnostics = async_compute.diagnostic_recorder();
    let Some(encoder) = async_compute.command_encoder() else {
        return;
    };
    plan.record(encoder, diagnostics.as_deref());
    frame.field_ahead = true;
}

/// fork: perf — with the trace on the async queue, its field into the block's probe region on the
/// main queue, at the start of the world view's frame (before the froxels and the main pass, the
/// probes' readers, and before this frame's trace): the field as the last frame's trace left it.
/// The last frame's final submission already waited for that trace, so this copy waits for nothing.
/// A new field is filled from the block first ([`GiFrame::field_seed`]), after a replaced one that
/// was ahead is copied in ([`GiFrame::field_retired`]). It runs only with the queue (`build`'s run
/// condition).
#[cfg(not(target_arch = "wasm32"))]
fn copy_gi_field(
    mut frame: ResMut<GiFrame>,
    pipes: Res<GiPipelines>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    view: ViewQuery<Has<crate::modern::volumetrics::ModernFogView>>,
    mut render_context: RenderContext,
) {
    if !pipes.async_trace || !crate::modern::world_view_only(view.into_inner()) {
        return;
    }
    let Some(block) = buffers.get(MODERN_SHADOW_CASCADES.id()) else {
        return;
    };
    let retired = frame.field_retired.take();
    let ahead = std::mem::take(&mut frame.field_ahead);
    let seed = std::mem::take(&mut frame.field_seed);
    if retired.is_none() && !ahead && !seed {
        return;
    }
    // The probe region is the block's tail; a field that would overrun it is never copied.
    let fits = |b: &Buffer| PROBE_OFFSET + b.size() <= block.buffer.size();
    let field = frame.field.clone().filter(|b| fits(b));
    let encoder = render_context.command_encoder();
    if let Some(old) = retired.filter(|b| fits(b)) {
        encoder.copy_buffer_to_buffer(&old, 0, &block.buffer, PROBE_OFFSET, old.size());
    }
    match field {
        Some(field) if seed => {
            encoder.copy_buffer_to_buffer(&block.buffer, PROBE_OFFSET, &field, 0, field.size());
        }
        Some(field) if ahead => {
            encoder.copy_buffer_to_buffer(&field, 0, &block.buffer, PROBE_OFFSET, field.size());
        }
        _ => {}
    }
}

pub(super) fn build(render_app: &mut SubApp) {
    // The trace samples this frame's cascades and this frame's voxels, and the froxels' ambient
    // reads the probes.
    let trace = gi_trace_pass
        .after(crate::static_gx::gx_shadow_pass)
        .after(crate::static_gx::gi_voxel_pass)
        .before(crate::modern::volumetrics::froxel_pass)
        .before(Core3dSystems::MainPass);
    // fork: native only — the pipelines read the ray-traced tier's decision, and the ray-query
    // stage casts against the TLAS `rt_scene_pass` (re)builds this frame.
    #[cfg(not(target_arch = "wasm32"))]
    let (init, trace) = (
        init_pipelines.after(crate::static_gx::rt_scene::init_rt_scene),
        trace.after(crate::static_gx::rt_scene::rt_scene_pass),
    );
    #[cfg(target_arch = "wasm32")]
    let init = init_pipelines;
    // fork: perf — whether the trace goes to the async queue is decided once that queue is made.
    #[cfg(not(target_arch = "wasm32"))]
    let init = init.after(crate::modern::async_compute::init);
    render_app
        .init_resource::<GiFrame>()
        .add_systems(RenderStartup, init)
        // After the voxel pass, which owns the scene the trace marches through.
        .add_systems(Render, prepare_gi.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(Core3d, trace);
    // fork: perf — the trace on the async queue, and its field copied in at the next frame's start
    // (see `gi_trace_pass_async` for the place). Both skipped without the queue: neither they nor
    // their buffers run.
    #[cfg(not(target_arch = "wasm32"))]
    render_app.add_systems(
        Core3d,
        (
            copy_gi_field
                .run_if(resource_exists::<RenderAsyncComputeQueue>)
                .before(crate::modern::volumetrics::froxel_pass)
                .before(Core3dSystems::MainPass),
            gi_trace_pass_async
                .run_if(resource_exists::<RenderAsyncComputeQueue>)
                .in_set(Core3dSystems::MainPass)
                .after(copy_gi_field)
                .after(crate::static_gx::gx_shadow_pass)
                .after(crate::static_gx::gi_voxel_pass)
                .after(crate::static_gx::rt_scene::rt_scene_pass)
                .after(crate::static_gx::static_gx_pass)
                .before(bevy::core_pipeline::core_3d::main_opaque_pass_3d),
        ),
    );
}
