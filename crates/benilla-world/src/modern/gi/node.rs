//! The render-world half of the probe GI: the block's GI parameters, the trace and blend compute
//! passes, and their place in the graph (after the modern shadow pass, whose cascades the trace
//! samples, and before the froxels, which read the probes).

use bevy::app::SubApp;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel};
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d_array,
    texture_3d, texture_depth_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::storage::GpuShaderStorageBuffer;
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
    trace_layout: BindGroupLayoutDescriptor,
    blend_layout: BindGroupLayoutDescriptor,
    trace: CachedComputePipelineId,
    blend: CachedComputePipelineId,
}

#[derive(Resource, Default)]
struct GiFrame {
    uniform: Option<Buffer>,
    scratch: Option<Buffer>,
    scratch_probes: usize,
    trace_bind: Option<BindGroup>,
    blend_bind: Option<BindGroup>,
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
            push_constant_ranges: vec![],
            shader: asset_server.load(path),
            shader_defs: vec![],
            entry_point: Some(entry.into()),
            zero_initialize_workgroup_memory: false,
        })
    };
    let trace = compute(
        "modern_gi_trace",
        &trace_layout,
        "embedded://benilla_world/shaders/modern/gi_trace.wgsl",
        "trace_probes",
    );
    let blend = compute(
        "modern_gi_blend",
        &blend_layout,
        "embedded://benilla_world/shaders/modern/gi_blend.wgsl",
        "blend_probes",
    );
    commands.insert_resource(GiPipelines {
        trace_layout,
        blend_layout,
        trace,
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
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    light: Option<Res<SharedLightBuffer>>,
    pipes: Option<Res<GiPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    mut frame: ResMut<GiFrame>,
) {
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
    if pipeline_cache.get_compute_pipeline(pipes.trace).is_none()
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
            BindingResource::Buffer(BufferBinding {
                buffer: &block.buffer,
                offset: PROBE_OFFSET,
                size: Some(std::num::NonZero::new(probe_bytes).expect("probe region is never empty")),
            }),
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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct GiTraceLabel;

#[derive(Default)]
struct GiTraceNode;

impl Node for GiTraceNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let frame = world.resource::<GiFrame>();
        let (Some(trace_bind), Some(blend_bind)) = (&frame.trace_bind, &frame.blend_bind) else {
            return Ok(());
        };
        if frame.groups == 0 {
            return Ok(());
        }
        let pipes = world.resource::<GiPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (Some(trace), Some(blend)) = (
            pipeline_cache.get_compute_pipeline(pipes.trace),
            pipeline_cache.get_compute_pipeline(pipes.blend),
        ) else {
            return Ok(());
        };
        let encoder = render_context.command_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_gi_trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(trace);
            pass.set_bind_group(0, trace_bind, &[]);
            pass.dispatch_workgroups(frame.groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_gi_blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blend);
            pass.set_bind_group(0, blend_bind, &[]);
            pass.dispatch_workgroups(frame.groups, 1, 1);
        }
        Ok(())
    }
}

pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .init_resource::<GiFrame>()
        .add_systems(RenderStartup, init_pipelines)
        // After the voxel pass, which owns the scene the trace marches through.
        .add_systems(Render, prepare_gi.in_set(RenderSystems::PrepareBindGroups))
        .add_render_graph_node::<GiTraceNode>(Core3d, GiTraceLabel)
        .add_render_graph_edges(Core3d, (GiTraceLabel, Node3d::StartMainPass))
        // The trace samples this frame's cascades and this frame's voxels.
        .add_render_graph_edge(Core3d, crate::static_gx::GxShadowLabel, GiTraceLabel)
        .add_render_graph_edge(Core3d, crate::static_gx::GxVoxelLabel, GiTraceLabel)
        // The froxels' ambient reads the probes.
        .add_render_graph_edge(Core3d, GiTraceLabel, crate::modern::volumetrics::FroxelLabel);
}
