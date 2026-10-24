//! fork: modern lighting — the auto-exposure's screen meter. A compute pass on the world view reads
//! the scene-referred frame, before the combine exposes it, into a 64×36 grid of log₂ luminance
//! (`shaders/modern/meter.wgsl`); the grid is read back and its mean published as the scene key
//! ([`super::look::publish_scene_key`]), which [`super::look`] turns into the exposure's lift.
//!
//! One readback in flight at a time: a frame whose buffer is still mapping meters nothing, and the
//! lift eases over seconds, so a reading every few frames is enough. The pass stands down while the
//! auto-exposure is off or the frame is classic ([`super::look::meter_wanted`]).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::render_resource::binding_types::{sampler, storage_buffer_sized, texture_2d};
use bevy::render::render_resource::*;
use bevy::render::renderer::{render_system, RenderContext, RenderDevice, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::ffx_glow::{FfxGlow, FfxState};

/// The grid, as `meter.wgsl` declares it.
const GRID: UVec2 = UVec2::new(64, 36);
const CELLS_BYTES: u64 = (GRID.x * GRID.y * 4) as u64;

/// The readback's states: free, copied this frame, mapping, mapped.
const IDLE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;

/// The meter's shader, built in the main app and handed to the render app.
#[derive(Resource, Clone)]
struct MeterShader(Handle<Shader>);

#[derive(Resource)]
struct Meter {
    pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    cells: Buffer,
    staging: Buffer,
    state: Arc<AtomicU8>,
}

/// `RenderStartup`: the pipeline and the two buffers.
fn init(
    mut commands: Commands,
    shader: Option<Res<MeterShader>>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(shader) = shader else {
        return;
    };
    let layout = BindGroupLayoutDescriptor::new(
        "modern_meter_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_sized(false, std::num::NonZero::new(CELLS_BYTES)),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("modern_meter".into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        shader: shader.0.clone(),
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
    });
    let buffer = |label, usage| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: CELLS_BYTES,
            usage,
            mapped_at_creation: false,
        })
    };
    commands.insert_resource(Meter {
        pipeline,
        layout,
        sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("modern_meter_linear"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        }),
        cells: buffer(
            "modern_meter_cells",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ),
        staging: buffer(
            "modern_meter_staging",
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        state: Arc::new(AtomicU8::new(IDLE)),
    });
}

/// `PostProcess`, before the FFXGlow combine exposes the frame: meter the world view's scene.
fn meter_pass(
    view: ViewQuery<(&ViewTarget, &FfxGlow)>,
    meter: Option<Res<Meter>>,
    pipeline_cache: Res<PipelineCache>,
    mut render_context: RenderContext,
) {
    let Some(meter) = meter else {
        return;
    };
    let (target, glow) = view.into_inner();
    if glow.state != FfxState::Player || !super::look::meter_wanted() {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(meter.pipeline) else {
        return; // still compiling: the authored rows stand in
    };
    if meter.state.load(Ordering::Acquire) != IDLE {
        return;
    }
    let bind = render_context.render_device().create_bind_group(
        "modern_meter",
        &pipeline_cache.get_bind_group_layout(&meter.layout),
        &BindGroupEntries::sequential((
            target.main_texture_view(),
            &meter.sampler,
            meter.cells.as_entire_binding(),
        )),
    );
    let encoder = render_context.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("modern_meter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(GRID.x.div_ceil(8), GRID.y.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&meter.cells, 0, &meter.staging, 0, CELLS_BYTES);
    meter.state.store(COPIED, Ordering::Release);
}

/// After the render graph submitted: map what this frame copied, and publish what landed.
fn readback(meter: Option<Res<Meter>>, device: Res<RenderDevice>) {
    let Some(m) = meter else {
        return;
    };
    let _ = device.poll(wgpu::PollType::Poll);
    match m.state.load(Ordering::Acquire) {
        COPIED => {
            m.state.store(MAPPING, Ordering::Release);
            let state = m.state.clone();
            m.staging.slice(..).map_async(MapMode::Read, move |result| {
                state.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
            });
        }
        READY => {
            let mean = {
                let view = m.staging.slice(..).get_mapped_range();
                let cells: &[f32] = bytemuck::cast_slice(&view);
                cells.iter().sum::<f32>() / cells.len().max(1) as f32
            };
            m.staging.unmap();
            m.state.store(IDLE, Ordering::Release);
            if mean.is_finite() {
                super::look::publish_scene_key(mean.exp2());
            }
        }
        _ => {}
    }
}

pub(super) struct MeterPlugin;

impl Plugin for MeterPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init)
            .add_systems(
                Core3d,
                meter_pass
                    .in_set(Core3dSystems::PostProcess)
                    .before(crate::ffx_glow::ffx_glow_pass),
            )
            .add_systems(
                Render,
                readback.in_set(RenderSystems::Render).after(render_system),
            );
    }

    /// The shader asset store exists once the plugin chain has built.
    fn finish(&self, app: &mut App) {
        let Some(mut shaders) = app.world_mut().get_resource_mut::<Assets<Shader>>() else {
            return;
        };
        let shader = MeterShader(shaders.add(Shader::from_wgsl(
            include_str!("../shaders/modern/meter.wgsl"),
            "benilla_modern/meter.wgsl",
        )));
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(shader);
        }
    }
}
