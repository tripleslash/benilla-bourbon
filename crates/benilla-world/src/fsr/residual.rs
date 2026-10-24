//! `WOW_FSR_DEBUG=residual` — the plan's motion-vector check (`FORK.md`, FSR 3.1.5):
//! over the pixels a moving draw wrote, the mean luma difference between each rendered frame and
//! the last one read back through the frame's own motion vectors, against the camera's
//! reprojection alone (`shaders/fsr/residual.wgsl`). Vectors that carry the movement make the
//! first smaller than the second. Every two seconds:
//! `FSR_RESIDUAL mv=<mean> camera=<mean> pixels=<per frame>`.
//!
//! One readback in flight at a time: a frame whose buffer is still mapping records nothing, so the
//! line averages the frames it caught. Inert, and allocates nothing, without the knob.

use std::num::NonZero;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_sized, texture_2d, texture_depth_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;

use super::FsrShaders;

/// Whether the knob is set; read once.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::launch_env::live_var("WOW_FSR_DEBUG").is_some_and(|v| v.trim() == "residual")
    })
}

/// The readback's states: free, copied this frame, mapping, mapped.
const IDLE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;

/// The sums: through the motion vectors, through the camera, pixels.
const SUMS_BYTES: u64 = 3 * 4;

#[derive(Resource)]
pub(super) struct Residual {
    pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    sums: Buffer,
    staging: Buffer,
    state: Arc<AtomicU8>,
    /// Last rendered frame's colour at render resolution, reallocated with the size.
    prev: Mutex<Option<(UVec2, Texture, TextureView)>>,
    acc: Mutex<Acc>,
}

#[derive(Default)]
struct Acc {
    through_mv: u64,
    through_camera: u64,
    pixels: u64,
    frames: u32,
    since: Option<Instant>,
}

/// `RenderStartup`: the pipeline and buffers, when the knob is set.
pub(super) fn init(
    mut commands: Commands,
    shaders: Option<Res<FsrShaders>>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let (true, Some(shaders)) = (enabled(), shaders) else {
        return;
    };
    let layout = BindGroupLayoutDescriptor::new(
        "fsr_residual_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, NonZero::new(super::upscaler::PARAMS_SIZE)),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_sized(false, NonZero::new(SUMS_BYTES)),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("fsr_residual".into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        shader: shaders.residual.clone(),
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
    });
    let buffer = |label, usage| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: SUMS_BYTES,
            usage,
            mapped_at_creation: false,
        })
    };
    commands.insert_resource(Residual {
        pipeline,
        layout,
        sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("fsr_residual_linear"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        }),
        sums: buffer(
            "fsr_residual_sums",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ),
        staging: buffer(
            "fsr_residual_staging",
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        state: Arc::new(AtomicU8::new(IDLE)),
        prev: Mutex::new(None),
        acc: Mutex::new(Acc::default()),
    });
}

/// What one frame's measurement reads.
pub(super) struct Inputs<'a> {
    pub params: BindingResource<'a>,
    pub depth: &'a TextureView,
    pub velocity: &'a TextureView,
    pub color: &'a Texture,
    pub color_view: &'a TextureView,
    pub render: UVec2,
}

impl Residual {
    /// At the end of the upscale: measure this frame against the last one, then keep this one.
    pub(super) fn record(
        &self,
        device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        encoder: &mut CommandEncoder,
        i: Inputs<'_>,
    ) {
        let mut prev = self
            .prev
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fresh = prev.as_ref().is_none_or(|(size, ..)| *size != i.render);
        if fresh {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("fsr_residual_prev"),
                size: Extent3d {
                    width: i.render.x,
                    height: i.render.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            *prev = Some((i.render, texture, view));
        }
        let Some((_, prev_texture, prev_view)) = prev.as_ref() else {
            return;
        };
        let ready = pipeline_cache.get_compute_pipeline(self.pipeline);
        if let (false, Some(pipeline), IDLE) = (fresh, ready, self.state.load(Ordering::Acquire)) {
            let bind = device.create_bind_group(
                "fsr_residual",
                &pipeline_cache.get_bind_group_layout(&self.layout),
                &BindGroupEntries::sequential((
                    i.params,
                    i.depth,
                    i.velocity,
                    i.color_view,
                    prev_view,
                    &self.sampler,
                    self.sums.as_entire_binding(),
                )),
            );
            encoder.clear_buffer(&self.sums, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("fsr_residual"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(i.render.x.div_ceil(8), i.render.y.div_ceil(8), 1);
            }
            encoder.copy_buffer_to_buffer(&self.sums, 0, &self.staging, 0, SUMS_BYTES);
            self.state.store(COPIED, Ordering::Release);
        }
        encoder.copy_texture_to_texture(
            i.color.as_image_copy(),
            prev_texture.as_image_copy(),
            Extent3d {
                width: i.render.x,
                height: i.render.y,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// After the render graph submitted: map what this frame copied, fold what landed, and log.
pub(super) fn readback(residual: Option<Res<Residual>>, device: Res<RenderDevice>) {
    let Some(r) = residual else {
        return;
    };
    let _ = device.poll(wgpu::PollType::Poll);
    match r.state.load(Ordering::Acquire) {
        COPIED => {
            r.state.store(MAPPING, Ordering::Release);
            let state = r.state.clone();
            r.staging.slice(..).map_async(MapMode::Read, move |result| {
                state.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
            });
        }
        READY => {
            let sums: [u32; 3] = {
                let view = r.staging.slice(..).get_mapped_range();
                let words: &[u32] = bytemuck::cast_slice(&view);
                [words[0], words[1], words[2]]
            };
            r.staging.unmap();
            r.state.store(IDLE, Ordering::Release);
            let mut acc = r
                .acc
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            acc.through_mv += u64::from(sums[0]);
            acc.through_camera += u64::from(sums[1]);
            acc.pixels += u64::from(sums[2]);
            acc.frames += 1;
            let since = *acc.since.get_or_insert_with(Instant::now);
            if since.elapsed().as_secs_f32() >= 2.0 {
                let px = acc.pixels.max(1) as f64 * 1024.0;
                info!(
                    "FSR_RESIDUAL mv={:.4} camera={:.4} pixels={:.0}",
                    acc.through_mv as f64 / px,
                    acc.through_camera as f64 / px,
                    acc.pixels as f64 / f64::from(acc.frames.max(1)),
                );
                *acc = Acc {
                    since: Some(Instant::now()),
                    ..default()
                };
            }
        }
        _ => {}
    }
}
