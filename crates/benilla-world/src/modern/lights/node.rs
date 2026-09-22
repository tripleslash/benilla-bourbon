//! The render-world half of the local lights: the grid's parameters in the shared block, the
//! binning compute pass, and its place in the graph (before the main pass, whose every lane reads
//! the cells it writes).

use bevy::app::SubApp;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel};
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderQueue};
use bevy::render::storage::GpuShaderStorageBuffer;
use bevy::render::{Render, RenderStartup, RenderSystems};

use benilla_assets::materials::MODERN_SHADOW_CASCADES;

use super::{ModernLights, BLOCK_LIGHT_BYTES, LIGHT_PARAM_BYTES};
use crate::lighting::SharedLightBuffer;
use crate::modern::gi::GI_PARAMS_BYTES;
use crate::modern::shadow::{CASCADES_BYTES, SHADOW_MAP_SIZE};
use crate::modern::volumetrics::BLOCK_TAIL_BYTES;
use crate::modern::ActiveLighting;

/// Where the light grid starts inside the modern storage block. 256-aligned by construction (the
/// cascades, the froxel tail and the GI parameters in front of it add to exactly 5120), which is
/// what lets the binning pass bind that region alone as read-write.
pub const LIGHT_OFFSET: u64 = (CASCADES_BYTES + BLOCK_TAIL_BYTES + GI_PARAMS_BYTES) as u64;

#[derive(Resource)]
struct LightPipelines {
    layout: BindGroupLayoutDescriptor,
    bin: CachedComputePipelineId,
}

#[derive(Resource, Default)]
struct LightFrame {
    bind: Option<BindGroup>,
    groups: u32,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "modern_light_bin",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let bin = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("modern_light_bin".into()),
        layout: vec![layout.clone()],
        push_constant_ranges: vec![],
        shader: asset_server.load("embedded://benilla_world/shaders/modern/light_bin.wgsl"),
        shader_defs: vec![],
        entry_point: Some("bin_lights".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(LightPipelines { layout, bin });
}

/// Write the grid's parameters and get the binning pass ready. The parameters go out every modern
/// frame, grid on or off: a lane reads the live bit to decide whether to walk a cell at all, so it
/// has to be current even when there is nothing to walk.
#[allow(clippy::too_many_arguments)]
fn prepare_lights(
    queue: Res<RenderQueue>,
    device: Res<bevy::render::renderer::RenderDevice>,
    active: Option<Res<ActiveLighting>>,
    lights: Option<Res<ModernLights>>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    light_buffer: Option<Res<SharedLightBuffer>>,
    pipes: Option<Res<LightPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    mut frame: ResMut<LightFrame>,
) {
    frame.bind = None;
    frame.groups = 0;
    let (Some(lights), Some(block)) = (lights, buffers.get(MODERN_SHADOW_CASCADES.id())) else {
        return;
    };
    let modern = active.is_some_and(|a| a.is_modern());
    let on = modern && lights.on;

    let mut params = super::params(&lights, SHADOW_MAP_SIZE);
    if !on {
        // Keep the flags word alone: a lane that reads "not live" never touches a cell.
        params.base.w = 0.0;
        params.misc.x = 0.0;
    }
    let mut bytes = Vec::with_capacity(LIGHT_PARAM_BYTES);
    params.std430(&mut bytes);
    if block.buffer.size() < LIGHT_OFFSET + BLOCK_LIGHT_BYTES as u64 {
        warn_once!("modern lights: the storage block is too small for the light grid");
        return;
    }
    queue.write_buffer(&block.buffer, LIGHT_OFFSET, &bytes);
    if !on {
        return;
    }
    let (Some(pipes), Some(light_buffer)) = (pipes.as_deref(), light_buffer.as_deref()) else {
        return;
    };
    if pipeline_cache.get_compute_pipeline(pipes.bin).is_none() {
        return;
    }
    frame.bind = Some(device.create_bind_group(
        "modern_light_bin",
        &pipeline_cache.get_bind_group_layout(&pipes.layout),
        &BindGroupEntries::sequential((
            light_buffer.0.as_entire_binding(),
            // The light region of the block alone: the rest of it is read-only storage in the same
            // frame's render passes, and a writable binding is exclusive per buffer, not per range.
            BindingResource::Buffer(BufferBinding {
                buffer: &block.buffer,
                offset: LIGHT_OFFSET,
                size: Some(
                    std::num::NonZero::new(BLOCK_LIGHT_BYTES as u64)
                        .expect("the light region is never empty"),
                ),
            }),
        )),
    ));
    frame.groups = (lights.tier.cells() as u32).div_ceil(64);
    bevy::log::info_once!(
        "modern lights: grid live — {} cells of {} yd, {} shadowed light(s)",
        lights.tier.cells(),
        lights.tier.cell,
        lights.shadowed.len(),
    );
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct LightBinLabel;

#[derive(Default)]
struct LightBinNode;

impl Node for LightBinNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let frame = world.resource::<LightFrame>();
        let (Some(bind), true) = (&frame.bind, frame.groups > 0) else {
            return Ok(());
        };
        let pipes = world.resource::<LightPipelines>();
        let Some(pipeline) = world
            .resource::<PipelineCache>()
            .get_compute_pipeline(pipes.bin)
        else {
            return Ok(());
        };
        // The journal's `gpu_modern` column (`benilla-app` `perf::journal`), where the device times
        // passes.
        let diagnostics = render_context.diagnostic_recorder();
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("modern_light_bin"),
                timestamp_writes: None,
            });
        let span = diagnostics.pass_span(&mut pass, "modern_light_bin");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(frame.groups, 1, 1);
        span.end(&mut pass);
        Ok(())
    }
}

pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .init_resource::<LightFrame>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare_lights.in_set(RenderSystems::PrepareBindGroups))
        .add_render_graph_node::<LightBinNode>(Core3d, LightBinLabel)
        .add_render_graph_edges(Core3d, (LightBinLabel, Node3d::StartMainPass))
        // The atlas slots the cells name are the ones the shadow pass just drew.
        .add_render_graph_edge(Core3d, crate::static_gx::GxShadowLabel, LightBinLabel)
        // fork: P8b — the froxels take their point lights from this frame's cells.
        .add_render_graph_edge(Core3d, LightBinLabel, crate::modern::volumetrics::FroxelLabel);
}
