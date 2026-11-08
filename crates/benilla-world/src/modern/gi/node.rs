//! The render-world half of the probe GI: the block's GI parameters, the trace and blend compute
//! passes, and their place in the graph (after the modern shadow pass, whose cascades the trace
//! samples, and before the froxels, which read the probes).

use bevy::app::SubApp;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d_array, texture_3d,
    texture_depth_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
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
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
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

    frame.classify_bind = Some(device.create_bind_group(
        "modern_gi_classify",
        &pipeline_cache.get_bind_group_layout(&pipes.classify_layout),
        &BindGroupEntries::sequential((
            probe_region(),
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
            probe_region(),
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
    let frame = world.resource::<GiFrame>();
    let (Some(classify_bind), Some(trace_bind), Some(blend_bind), Some(dispatch)) = (
        &frame.classify_bind,
        &frame.trace_bind,
        &frame.blend_bind,
        &frame.dispatch,
    ) else {
        return;
    };
    if frame.groups == 0 {
        return;
    }
    let pipes = world.resource::<GiPipelines>();
    let pipeline_cache = world.resource::<PipelineCache>();
    let (Some(classify), Some(setup), Some(trace), Some(sum), Some(blend)) = (
        pipeline_cache.get_compute_pipeline(pipes.classify),
        pipeline_cache.get_compute_pipeline(pipes.setup),
        pipeline_cache.get_compute_pipeline(pipes.trace),
        pipeline_cache.get_compute_pipeline(pipes.sum),
        pipeline_cache.get_compute_pipeline(pipes.blend),
    ) else {
        return;
    };
    // fork: F — a voxel trace that declares the scene cannot run without it.
    if pipes.voxel_rt && frame.shadow_bind.is_none() && frame.rt_bind.is_none() {
        return;
    }
    // fork: at the ray-traced tier the ray stage is the ray query, with the scene at group 1.
    let rt_stage = match &frame.rt_bind {
        Some(bind) => match pipes
            .rt_trace
            .and_then(|id| pipeline_cache.get_compute_pipeline(id))
        {
            Some(pipeline) => Some((bind, pipeline)),
            None => return,
        },
        None => None,
    };
    // The journal's `gpu_modern` column, where the device times passes.
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = render_context.command_encoder();
    // Classify sweeps every probe and packs the ones due a trace; the trace and the blend then
    // launch one thread per packed probe (`gi_classify.wgsl` says why that is the whole win).
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("modern_gi_classify"),
            timestamp_writes: None,
        });
        let span = diagnostics.pass_span(&mut pass, "modern_gi_classify");
        pass.set_pipeline(classify);
        pass.set_bind_group(0, classify_bind, &[]);
        pass.dispatch_workgroups(frame.groups, 1, 1);
        span.end(&mut pass);
    }
    {
        // fork: a span of its own at the ray-traced tier, so the two paths compare by name.
        let label = if rt_stage.is_some() {
            "modern_gi_trace_rt"
        } else {
            "modern_gi_trace"
        };
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        let span = diagnostics.pass_span(&mut pass, label);
        pass.set_bind_group(0, trace_bind, &[]);
        // Setup, rays, sums — one dispatch each over the same rows (`gi_trace.wgsl`).
        pass.set_pipeline(setup);
        pass.dispatch_workgroups_indirect(dispatch, 0);
        match rt_stage {
            Some((rt_bind, rt_trace)) => {
                pass.set_pipeline(rt_trace);
                pass.set_bind_group(1, rt_bind, &[]);
            }
            None => {
                pass.set_pipeline(trace);
                // fork: F — the voxel path's hits trace their sun (`GiPipelines::voxel_rt`).
                if let Some(bind) = &frame.shadow_bind {
                    pass.set_bind_group(1, bind, &[]);
                }
            }
        }
        pass.dispatch_workgroups_indirect(dispatch, 0);
        pass.set_pipeline(sum);
        pass.dispatch_workgroups_indirect(dispatch, 0);
        span.end(&mut pass);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("modern_gi_blend"),
            timestamp_writes: None,
        });
        let span = diagnostics.pass_span(&mut pass, "modern_gi_blend");
        pass.set_pipeline(blend);
        pass.set_bind_group(0, blend_bind, &[]);
        pass.dispatch_workgroups_indirect(dispatch, 0);
        span.end(&mut pass);
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
    render_app
        .init_resource::<GiFrame>()
        .add_systems(RenderStartup, init)
        // After the voxel pass, which owns the scene the trace marches through.
        .add_systems(Render, prepare_gi.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(Core3d, trace);
}
