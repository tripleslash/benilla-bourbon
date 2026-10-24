//! The render-world half of the local lights: the grid's parameters in the shared block, the
//! binning compute pass, and its place in the graph (before the main pass, whose every lane reads
//! the cells it writes).

use bevy::app::SubApp;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::{Render, RenderStartup, RenderSystems};

use benilla_assets::materials::MODERN_SHADOW_CASCADES;

use super::{ModernLights, BLOCK_LIGHT_BYTES, LIGHT_PARAM_BYTES};
use crate::lighting::{LightRank, SharedLightBuffer};
use crate::modern::gi::GI_PARAMS_BYTES;
use crate::modern::shadow::{CASCADES_BYTES, SHADOW_MAP_SIZE};
use crate::modern::volumetrics::BLOCK_TAIL_BYTES;
use crate::modern::ActiveLighting;

/// Where the light grid starts inside the modern storage block. 256-aligned by construction (the
/// cascades, the froxel tail and the GI parameters in front of it add to exactly 5376), which is
/// what lets the binning pass bind that region alone as read-write.
pub const LIGHT_OFFSET: u64 = (CASCADES_BYTES + BLOCK_TAIL_BYTES + GI_PARAMS_BYTES) as u64;

#[derive(Resource)]
struct LightPipelines {
    layout: BindGroupLayoutDescriptor,
    bin: CachedComputePipelineId,
    /// The table's steady strengths ([`LightRank`]), which the binning pass ranks a cell by.
    rank: Buffer,
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
    device: Res<bevy::render::renderer::RenderDevice>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "modern_light_bin",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );
    let rank = device.create_buffer(&BufferDescriptor {
        label: Some("modern_light_rank"),
        size: std::mem::size_of::<LightRank>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bin = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("modern_light_bin".into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        shader: asset_server.load("embedded://benilla_world/shaders/modern/light_bin.wgsl"),
        shader_defs: vec![],
        entry_point: Some("bin_lights".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(LightPipelines { layout, bin, rank });
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
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    light_buffer: Option<Res<SharedLightBuffer>>,
    rank: Option<Res<LightRank>>,
    pipes: Option<Res<LightPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    uploaded_plane: Option<Res<crate::modern::shadow::UploadedMirrorPlane>>,
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
    // fork: perf — the vertex stages' copy of the live bit just written and of the mirror plane
    // the block took this frame (`shadow::VERTEX_HINT_OFFSET`). Written only when both went into
    // the block, after `upload_light` has packed the row (this is `PrepareBindGroups`), so a
    // vertex stage that reads "skip" agrees with what its fragments read from the block.
    let live = (params.base.w as u32) & 1 != 0;
    if let (true, Some(light_buffer), Some(plane)) = (
        live && crate::modern::vertex_hints_enabled(),
        light_buffer.as_deref(),
        uploaded_plane.and_then(|p| p.0),
    ) {
        queue.write_buffer(
            &light_buffer.0,
            crate::modern::shadow::VERTEX_HINT_OFFSET,
            bytemuck::cast_slice(&[1.0f32, plane]),
        );
    }
    if !on {
        return;
    }
    let (Some(pipes), Some(light_buffer)) = (pipes.as_deref(), light_buffer.as_deref()) else {
        return;
    };
    if pipeline_cache.get_compute_pipeline(pipes.bin).is_none() {
        return;
    }
    // The strengths of the table `upload_light` wrote this frame: both are extracted from the one
    // pack, so entry `i` here is entry `i` there.
    let Some(rank) = rank else {
        return;
    };
    queue.write_buffer(&pipes.rank, 0, bytemuck::cast_slice(&rank.0));
    frame.bind = Some(
        device.create_bind_group(
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
                pipes.rank.as_entire_binding(),
            )),
        ),
    );
    frame.groups = (lights.tier.cells() as u32).div_ceil(64);
    bevy::log::info_once!(
        "modern lights: grid live — {} cells of {} yd, {} shadowed light(s)",
        lights.tier.cells(),
        lights.tier.cell,
        lights.shadowed.len(),
    );
}

/// The binning pass, on the world camera's view alone: the grid is world-space, and binning it
/// again for the water mirror and the booths rewrote the same cells
/// ([`world_view_only`](crate::modern::world_view_only)).
fn light_bin_pass(
    frame: Res<LightFrame>,
    pipes: Res<LightPipelines>,
    pipeline_cache: Res<PipelineCache>,
    view: ViewQuery<Has<crate::modern::volumetrics::ModernFogView>>,
    mut render_context: RenderContext,
) {
    if !crate::modern::world_view_only(view.into_inner()) {
        return;
    }
    let (Some(bind), true) = (&frame.bind, frame.groups > 0) else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipes.bin) else {
        return;
    };
    // The journal's `gpu_modern` column (`benilla-app` `perf::journal`), where the device times
    // passes.
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
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
}

pub(super) fn build(render_app: &mut SubApp) {
    render_app
        .init_resource::<LightFrame>()
        .init_resource::<crate::modern::shadow::UploadedMirrorPlane>()
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(
            Render,
            prepare_lights.in_set(RenderSystems::PrepareBindGroups),
        )
        // The atlas slots the cells name are the ones the shadow pass just drew; fork: P8b — the
        // froxels take their point lights from this frame's cells.
        .add_systems(
            Core3d,
            light_bin_pass
                .after(crate::static_gx::gx_shadow_pass)
                .before(crate::modern::volumetrics::froxel_pass)
                .before(Core3dSystems::MainPass),
        );
}
