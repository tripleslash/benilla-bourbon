//! fork: native HDR output — `WOW_HDR_PROBE=1`, the measurement of what an HDR frame actually
//! carries: once a second, every pixel of the frame a backdrop camera hands the window (the
//! player-UI camera's target in the client, the FSR backdrop's in the worldviewer), in nits, read
//! back and logged as the peak and the share of pixels above paper white and above twice it. It
//! runs after the world's combine, so a highlight the HDR exit failed to keep reads as paper white.
//! Unarmed, nothing is built.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::core_pipeline::{Core2d, Core2dSystems};
use bevy::prelude::*;
use bevy::render::render_resource::binding_types::{
    storage_buffer_sized, texture_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{render_system, RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::{Msaa, ViewTarget};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::ffx_glow::FfxBackdrop;
use crate::hdr_output::HdrDisplay;

/// `Totals` in `hdr_probe.wgsl`: the peak in tenths of a nit, then three counts.
const TOTALS_BYTES: u64 = 16;
const PARAMS_BYTES: u64 = 16;
/// How often a frame is measured.
const EVERY: Duration = Duration::from_secs(1);

const IDLE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;

fn armed() -> bool {
    crate::launch_env::var("WOW_HDR_PROBE").is_some_and(|v| v.trim() != "0")
}

#[derive(Resource, Clone)]
struct ProbeShader(Handle<Shader>);

#[derive(Resource)]
struct Probe {
    pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
    params: Buffer,
    totals: Buffer,
    staging: Buffer,
    state: Arc<AtomicU8>,
    /// When the last frame was measured, and the paper white it was read against.
    last: Mutex<(Option<Instant>, f32)>,
}

fn init(
    mut commands: Commands,
    shader: Option<Res<ProbeShader>>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(shader) = shader else {
        return;
    };
    let layout = BindGroupLayoutDescriptor::new(
        "hdr_probe_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer_sized(false, std::num::NonZero::new(PARAMS_BYTES)),
                storage_buffer_sized(false, std::num::NonZero::new(TOTALS_BYTES)),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        constants: Default::default(),
        label: Some("hdr_probe".into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        shader: shader.0.clone(),
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: true,
    });
    let buffer = |label, size, usage| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    commands.insert_resource(Probe {
        pipeline,
        layout,
        params: buffer(
            "hdr_probe_params",
            PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ),
        totals: buffer(
            "hdr_probe_totals",
            TOTALS_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ),
        staging: buffer(
            "hdr_probe_staging",
            TOTALS_BYTES,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        state: Arc::new(AtomicU8::new(IDLE)),
        last: Mutex::new((None, 0.0)),
    });
}

/// After the backdrop's main pass: measure the frame its target holds, once a second.
fn probe_pass(
    view: ViewQuery<(&ViewTarget, &FfxBackdrop, &Msaa)>,
    probe: Option<Res<Probe>>,
    hdr: Option<Res<HdrDisplay>>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
    mut render_context: RenderContext,
) {
    let (Some(probe), Some(hdr)) = (probe, hdr) else {
        return;
    };
    let (target, backdrop, msaa) = view.into_inner();
    if !hdr.active
        || *msaa != Msaa::Off
        || backdrop.source.is_none()
        || target.main_texture_format() != TextureFormat::Rgba16Float
    {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(probe.pipeline) else {
        return;
    };
    if probe.state.load(Ordering::Acquire) != IDLE {
        return;
    }
    {
        let mut last = probe.last.lock().unwrap();
        if last.0.is_some_and(|t| t.elapsed() < EVERY) {
            return;
        }
        *last = (Some(Instant::now()), hdr.paper_nits);
    }
    let gamma_lane = u32::from(!backdrop.decode);
    let params: [u32; 4] = [hdr.paper_nits.to_bits(), gamma_lane, 0, 0];
    queue.write_buffer(&probe.params, 0, bytemuck::cast_slice(&params));
    let bind = render_context.render_device().create_bind_group(
        "hdr_probe",
        &pipeline_cache.get_bind_group_layout(&probe.layout),
        &BindGroupEntries::sequential((
            target.main_texture_view(),
            probe.params.as_entire_binding(),
            probe.totals.as_entire_binding(),
        )),
    );
    let size = target.main_texture().size();
    let encoder = render_context.command_encoder();
    encoder.clear_buffer(&probe.totals, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("hdr_probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&probe.totals, 0, &probe.staging, 0, TOTALS_BYTES);
    probe.state.store(COPIED, Ordering::Release);
}

/// After the render graph submitted: map what was copied, and log what landed.
fn readback(probe: Option<Res<Probe>>, hdr: Option<Res<HdrDisplay>>, device: Res<RenderDevice>) {
    let Some(p) = probe else {
        return;
    };
    let _ = device.poll(wgpu::PollType::Poll);
    match p.state.load(Ordering::Acquire) {
        COPIED => {
            p.state.store(MAPPING, Ordering::Release);
            let state = p.state.clone();
            p.staging.slice(..).map_async(MapMode::Read, move |result| {
                state.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
            });
        }
        READY => {
            let totals: [u32; 4] = {
                let view = p
                    .staging
                    .slice(..)
                    .get_mapped_range()
                    .expect("a mapped staging buffer reads");
                let words: &[u32] = bytemuck::cast_slice(&view);
                [words[0], words[1], words[2], words[3]]
            };
            p.staging.unmap();
            p.state.store(IDLE, Ordering::Release);
            let paper = p.last.lock().unwrap().1;
            let cap = hdr.map_or(0.0, |h| h.peak_nits);
            let [peak, above, above2, pixels] = totals;
            let share = |n: u32| 100.0 * n as f64 / pixels.max(1) as f64;
            info!(
                "HDR_PROBE peak={:.0} nits paper={paper:.0} cap={cap:.0} above_paper={:.3}% \
                 above_2x_paper={:.3}% pixels={pixels}",
                peak as f32 / 10.0,
                share(above),
                share(above2),
            );
        }
        _ => {}
    }
}

pub(crate) struct HdrProbePlugin;

impl Plugin for HdrProbePlugin {
    fn build(&self, app: &mut App) {
        if !armed() {
            return;
        }
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init)
            .add_systems(
                Core2d,
                probe_pass
                    .in_set(Core2dSystems::MainPass)
                    .after(crate::ffx_glow::ffx_transparent_pass_2d),
            )
            .add_systems(
                Render,
                readback.in_set(RenderSystems::Render).after(render_system),
            );
    }

    fn finish(&self, app: &mut App) {
        if !armed() {
            return;
        }
        let Some(mut shaders) = app.world_mut().get_resource_mut::<Assets<Shader>>() else {
            return;
        };
        let shader = ProbeShader(shaders.add(Shader::from_wgsl(
            include_str!("shaders/hdr_probe.wgsl"),
            "benilla_hdr/hdr_probe.wgsl",
        )));
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(shader);
        }
    }
}
