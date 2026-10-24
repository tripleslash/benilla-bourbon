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
    commands.insert_resource(GiPipelines {
        classify_layout,
        trace_layout,
        blend_layout,
        classify,
        setup,
        trace,
        sum,
        blend,
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
) {
    frame.classify_bind = None;
    frame.trace_bind = None;
    frame.blend_bind = None;
    frame.groups = 0;
    let (Some(gi), Some(block)) = (gi, buffers.get(MODERN_SHADOW_CASCADES.id())) else {
        return;
    };
    let modern = active.is_some_and(|a| a.is_modern());
    let on = modern && gi.on && scene.ready;

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
    let inputs = (
        pipes.as_deref(),
        light.as_deref(),
        images.get(MODERN_SHADOW_MAP.id()),
        scene.albedo.as_ref(),
        scene.normal.as_ref(),
        scene.height.as_ref(),
    );
    let (Some(pipes), Some(light), Some(shadow), Some(albedo), Some(normal), Some(height)) = inputs
    else {
        return;
    };
    if pipeline_cache
        .get_compute_pipeline(pipes.classify)
        .is_none()
        || pipeline_cache.get_compute_pipeline(pipes.trace).is_none()
        || pipeline_cache.get_compute_pipeline(pipes.blend).is_none()
    {
        return;
    }

    let uniform_bytes = super::trace_bytes(&gi, &params, &scene.clip);
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
            &albedo.1,
            &normal.1,
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
    frame.groups = (probes as u32).div_ceil(64);
    bevy::log::info_once!(
        "modern GI: probes live — {} in {} cascade(s), {} rays, 1/{} traced a frame",
        probes,
        gi.tier.cascades,
        gi.tier.rays,
        gi.tier.period,
    );
}

/// The probe trace, on the world camera's view alone: the field is world-space, and the water
/// mirror and the booths, which read none of it, each ran it again ([`world_view_only`]).
///
/// [`world_view_only`]: crate::modern::world_view_only
fn gi_trace_pass(
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
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("modern_gi_trace"),
            timestamp_writes: None,
        });
        let span = diagnostics.pass_span(&mut pass, "modern_gi_trace");
        pass.set_bind_group(0, trace_bind, &[]);
        // Setup, rays, sums — one dispatch each over the same rows (`gi_trace.wgsl`).
        for stage in [setup, trace, sum] {
            pass.set_pipeline(stage);
            pass.dispatch_workgroups_indirect(dispatch, 0);
        }
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
    render_app
        .init_resource::<GiFrame>()
        .add_systems(RenderStartup, init_pipelines)
        // After the voxel pass, which owns the scene the trace marches through.
        .add_systems(Render, prepare_gi.in_set(RenderSystems::PrepareBindGroups))
        // The trace samples this frame's cascades and this frame's voxels, and the froxels'
        // ambient reads the probes.
        .add_systems(
            Core3d,
            gi_trace_pass
                .after(crate::static_gx::gx_shadow_pass)
                .after(crate::static_gx::gi_voxel_pass)
                .before(crate::modern::volumetrics::froxel_pass)
                .before(Core3dSystems::MainPass),
        );
}
