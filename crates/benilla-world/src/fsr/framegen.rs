//! FSR 3.1.5 frame generation on the world camera: the last two HUD-less finished frames
//! ([`WorldFinal`]) in, one interpolated frame out.
//!
//! A port of physics-rs's `render/fsr3_fg.rs` (FidelityFX SDK 3.1.5, MIT; `LICENSE-FIDELITYFX.txt`)
//! to Bevy's render world. The chain, in the SDK's order (`shaders/fsr/fg_*.wgsl`):
//!
//! 1. `fg_reconstruct_depth` — the upscaler's depths walked half a motion vector forward.
//! 2. `fg_game_vector_field` — its motion vectors scattered the same way, contested cells resolved
//!    by a priority packed above the vector.
//! 3. `fg_mv_pyramid_seed` + `fg_pyramid_reduce` — a pyramid over that field for the holes.
//! 4. `fg_disocclusion` — whether each cell's surface was visible in the previous and next frame.
//! 5. `fg_interpolate` — the fetch and blend at display resolution.
//! 6. `fg_color_pyramid_seed` + `fg_pyramid_reduce` — a pyramid over the result.
//! 7. `fg_inpaint` — the pixels neither source could supply.
//!
//! Steps 1–4 share the upscaler's dilated motion, depths and reconstructed previous depth
//! ([`FsrState::shared`]) and are skipped on a reset frame, where step 5 passes the current frame
//! through. Optical flow is omitted, as physics-rs omits it (the SDK's `opticalFlowScale <= 0`).
//!
//! **Sources.** [`WorldFinal`] is the FFXGlow combine of the upscaled world, in the UI camera's
//! target format and encoding: the compose writes what the UI camera's ground draw would have
//! written, and the chain reads those stored values back through a non-sRGB view, the SDK's SDR
//! back-buffer case (`BackBufferTransferFunction == 0`).
//!
//! **Presentation.** On a rendered world frame the UI camera grounds on the generated frame between
//! the last real frame and this one; on the held frame after it, on `WorldFinal[p]`, the real frame
//! just rendered ([`FgPresent`]). So the pair is monotonic in time and costs one frame of latency,
//! as in the SDK. `fg_present.wgsl` copies either into the UI target, decoding first when that
//! target is sRGB so its write re-encodes the stored value unchanged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bevy::core_pipeline::core_2d::CORE_2D_DEPTH_FORMAT;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_storage_2d,
    uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, Msaa, ViewTarget};
use bytemuck::{Pod, Zeroable};

use super::caps::Formats;
use super::upscaler::{FsrState, VIEW_SPACE_TO_METERS};
use super::{FgSolo, FsrHeld, FsrShaders, FsrView};
use crate::ffx_glow::FfxBackdrop;

/// Compute workgroup edge; `@workgroup_size(8, 8, 1)` in every pass.
const WORKGROUP: u32 = 8;
/// Levels of the vector field's inpainting pyramid, at most (`LoadInpaintedGameFieldMv`).
const MV_PYRAMID_MAX_LEVELS: u32 = 11;
/// Levels of the interpolated image's inpainting pyramid, at most (`ComputeInpainting`).
const COLOR_PYRAMID_MAX_LEVELS: u32 = 10;
/// Both pyramids: `(mv.xy, high, low)` and `(rgb, weight)`.
const PYRAMID: TextureFormat = TextureFormat::Rgba16Float;
/// The interpolated image and the inpainted result: what the sources carry, bytes.
const IMAGE: TextureFormat = TextureFormat::Rgba8Unorm;

/// Per-frame constants; `FgParams` in `shaders/fsr/fg_common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FgParams {
    render_size: [f32; 2],
    display_size: [f32; 2],
    inv_render_size: [f32; 2],
    inv_display_size: [f32; 2],
    device_to_view_depth: [f32; 4],
    tan_half_fov: f32,
    frame_index_since_reset: f32,
    srgb_output: f32,
    mv_pyramid_mips: f32,
    color_pyramid_mips: f32,
    _pad: [f32; 3],
}

const PARAMS_SIZE: u64 = std::mem::size_of::<FgParams>() as u64;

/// `DeviceToViewSpaceTransformFactors` for Bevy's reverse-Z infinite projection, and
/// `TanHalfFoV`: `view_space_depth(d) = y / (d − x)` is `near·0.9144 / d`, the eye distance in
/// metres, with `x = −ε` keeping a cleared 0 finite; `zw` scale an NDC into eye space at unit
/// distance.
fn depth_factors(proj: Mat4) -> ([f32; 4], f32) {
    let tan_x = 1.0 / proj.x_axis.x;
    let tan_y = 1.0 / proj.y_axis.y;
    (
        [
            -f32::EPSILON,
            proj.w_axis.z * VIEW_SPACE_TO_METERS,
            tan_x,
            tan_y,
        ],
        tan_x,
    )
}

/// A pyramid's level sizes over a `source`-sized image: the source halved, then each level the
/// one below halved, stopping before a side would round to zero — wgpu clamps mip extents at one
/// and the shaders' halving does not, and the two must agree.
fn pyramid_sizes(source: UVec2, max_levels: u32) -> Vec<UVec2> {
    let mut sizes = vec![(source / 2).max(UVec2::ONE)];
    while (sizes.len() as u32) < max_levels {
        let s = sizes[sizes.len() - 1];
        if s.x < 2 || s.y < 2 {
            break;
        }
        sizes.push(s / 2);
    }
    sizes
}

/// A pyramid built one level per dispatch: every level for the searches, and single-mip views for
/// the reduce (WebGPU forbids overlapping read and write views of one texture in a bind group).
struct Pyramid {
    sampled: TextureView,
    levels: Vec<TextureView>,
    sizes: Vec<UVec2>,
}

impl Pyramid {
    fn new(device: &RenderDevice, label: &'static str, source: UVec2, max_levels: u32) -> Self {
        let sizes = pyramid_sizes(source, max_levels);
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: sizes[0].x,
                height: sizes[0].y,
                depth_or_array_layers: 1,
            },
            mip_level_count: sizes.len() as u32,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: PYRAMID,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let levels = (0..sizes.len() as u32)
            .map(|base_mip_level| {
                texture.create_view(&TextureViewDescriptor {
                    label: Some(label),
                    base_mip_level,
                    mip_level_count: Some(1),
                    ..default()
                })
            })
            .collect();
        Self {
            sampled: texture.create_view(&TextureViewDescriptor::default()),
            levels,
            sizes,
        }
    }
}

/// One of the two finished frames the interpolator works between: the compose writes it through
/// its own format's view, the chain reads it back through the non-sRGB view.
pub struct WorldFinal {
    pub attachment: TextureView,
    pub sampled: TextureView,
}

struct Targets {
    render: UVec2,
    display: UVec2,
    format: TextureFormat,
    finals: [WorldFinal; 2],
    interpolated_depth: Buffer,
    field_x: Buffer,
    field_y: Buffer,
    disocclusion: TextureView,
    mv_pyramid: Pyramid,
    interp: TextureView,
    color_pyramid: Pyramid,
    output: TextureView,
}

impl Targets {
    fn new(
        device: &RenderDevice,
        formats: Formats,
        render: UVec2,
        display: UVec2,
        format: TextureFormat,
    ) -> Self {
        let sampled_format = format.remove_srgb_suffix();
        let view_formats: &[TextureFormat] = if sampled_format == format {
            &[]
        } else {
            std::slice::from_ref(&sampled_format)
        };
        let final_image = |label: &'static str| {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: display.x,
                    height: display.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats,
            });
            WorldFinal {
                attachment: texture.create_view(&TextureViewDescriptor::default()),
                sampled: texture.create_view(&TextureViewDescriptor {
                    format: Some(sampled_format),
                    ..default()
                }),
            }
        };
        let image = |label: &'static str, format: TextureFormat, size: UVec2| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        let scatter = |label: &'static str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: u64::from(render.x) * u64::from(render.y) * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            render,
            display,
            format,
            finals: [
                final_image("fsr_world_final_0"),
                final_image("fsr_world_final_1"),
            ],
            interpolated_depth: scatter("fsr_fg_interpolated_depth"),
            field_x: scatter("fsr_fg_vector_field_x"),
            field_y: scatter("fsr_fg_vector_field_y"),
            disocclusion: image("fsr_fg_disocclusion", formats.rg8, render),
            mv_pyramid: Pyramid::new(device, "fsr_fg_mv_pyramid", render, MV_PYRAMID_MAX_LEVELS),
            interp: image("fsr_fg_interpolated", IMAGE, display),
            color_pyramid: Pyramid::new(
                device,
                "fsr_fg_color_pyramid",
                display,
                COLOR_PYRAMID_MAX_LEVELS,
            ),
            output: image("fsr_fg_output", IMAGE, display),
        }
    }
}

/// One view's frame-generation state, advanced once per rendered world frame by
/// [`prepare_framegen`]; the two flags are set by [`fsr_framegen_pass`] as the chain lands.
#[derive(Component)]
pub struct FgState {
    targets: Targets,
    params: Buffer,
    /// The slot this frame's `WorldFinal` occupies.
    parity: usize,
    /// Rendered frames since the last reset; 0 passes the current frame through.
    frame_index_since_reset: u32,
    /// `WorldFinal[parity]` holds this world frame's composed image.
    composed: Arc<AtomicBool>,
    /// The generated frame of this world frame was written.
    generated: Arc<AtomicBool>,
    /// The claiming backdrop decodes at its exit ([`FfxBackdrop::decode`]), so the compose must
    /// too: `WorldFinal` holds what that backdrop's combine would have drawn.
    decode: bool,
}

impl FgState {
    /// The format the compose renders `WorldFinal` in: the UI camera's target's.
    pub fn format(&self) -> TextureFormat {
        self.targets.format
    }

    /// Whether the compose decodes at its exit, as the claiming backdrop's combine does.
    pub fn decode(&self) -> bool {
        self.decode
    }

    /// This frame's finished image, where the compose writes.
    pub fn final_attachment(&self) -> &TextureView {
        &self.targets.finals[self.parity].attachment
    }
}

/// The ground a claiming backdrop camera draws while frame generation runs: the generated frame on
/// a rendered world frame, the real one on the held frame after it. Present only on a world view
/// with [`FgState`]; drawn only once [`Self::ready`].
#[derive(Component, Clone)]
pub struct FgPresent {
    pub pipeline: CachedRenderPipelineId,
    pub bind: BindGroup,
    ready: Arc<AtomicBool>,
}

impl FgPresent {
    /// Whether the image this binds was written: the chain ran this frame, or on a held frame, the
    /// compose ran on the world frame before it.
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

/// The pipelines and layouts, built at startup.
#[derive(Resource)]
pub struct FgPipelines {
    layouts: FgLayouts,
    reconstruct_depth: CachedComputePipelineId,
    game_vector_field: CachedComputePipelineId,
    mv_pyramid_seed: CachedComputePipelineId,
    mv_pyramid_reduce: CachedComputePipelineId,
    disocclusion: CachedComputePipelineId,
    interpolate: CachedComputePipelineId,
    color_pyramid_seed: CachedComputePipelineId,
    color_pyramid_reduce: CachedComputePipelineId,
    inpaint: CachedComputePipelineId,
    sampler: Sampler,
}

impl FgPipelines {
    /// Every compute pipeline, for the warm-up to wait on.
    pub fn ids(&self) -> [CachedComputePipelineId; 9] {
        [
            self.reconstruct_depth,
            self.game_vector_field,
            self.mv_pyramid_seed,
            self.mv_pyramid_reduce,
            self.disocclusion,
            self.interpolate,
            self.color_pyramid_seed,
            self.color_pyramid_reduce,
            self.inpaint,
        ]
    }
}

struct FgLayouts {
    reconstruct_depth: BindGroupLayoutDescriptor,
    game_vector_field: BindGroupLayoutDescriptor,
    mv_pyramid_seed: BindGroupLayoutDescriptor,
    pyramid_reduce: BindGroupLayoutDescriptor,
    disocclusion: BindGroupLayoutDescriptor,
    interpolate: BindGroupLayoutDescriptor,
    color_pyramid_seed: BindGroupLayoutDescriptor,
    inpaint: BindGroupLayoutDescriptor,
}

impl FgLayouts {
    fn new(formats: Formats) -> Self {
        use StorageTextureAccess::WriteOnly;
        let uniform = || uniform_buffer_sized(false, std::num::NonZero::new(PARAMS_SIZE));
        // The upscaler's depths are `rg32float`, not filterable in WebGPU's core set; everything
        // else here is, and the passes that sample need it to be.
        let tex = |filterable| texture_2d(TextureSampleType::Float { filterable });
        let out = |format| texture_storage_2d(format, WriteOnly);
        let lin = || sampler(SamplerBindingType::Filtering);
        let rw = || storage_buffer_sized(false, None);
        let ro = || storage_buffer_read_only_sized(false, None);
        let c = ShaderStages::COMPUTE;
        Self {
            reconstruct_depth: BindGroupLayoutDescriptor::new(
                "fsr_fg_reconstruct_depth_layout",
                &BindGroupLayoutEntries::sequential(c, (uniform(), tex(true), tex(false), rw())),
            ),
            game_vector_field: BindGroupLayoutDescriptor::new(
                "fsr_fg_game_vector_field_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(true),
                        tex(false),
                        tex(true),
                        tex(true),
                        lin(),
                        rw(),
                        rw(),
                    ),
                ),
            ),
            mv_pyramid_seed: BindGroupLayoutDescriptor::new(
                "fsr_fg_mv_pyramid_seed_layout",
                &BindGroupLayoutEntries::sequential(c, (uniform(), ro(), ro(), out(PYRAMID))),
            ),
            pyramid_reduce: BindGroupLayoutDescriptor::new(
                "fsr_fg_pyramid_reduce_layout",
                &BindGroupLayoutEntries::sequential(c, (uniform(), tex(true), out(PYRAMID))),
            ),
            disocclusion: BindGroupLayoutDescriptor::new(
                "fsr_fg_disocclusion_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(false),
                        ro(),
                        ro(),
                        ro(),
                        ro(),
                        tex(true),
                        out(formats.rg8),
                    ),
                ),
            ),
            interpolate: BindGroupLayoutDescriptor::new(
                "fsr_fg_interpolate_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(true),
                        tex(true),
                        tex(true),
                        ro(),
                        ro(),
                        tex(true),
                        lin(),
                        out(IMAGE),
                    ),
                ),
            ),
            color_pyramid_seed: BindGroupLayoutDescriptor::new(
                "fsr_fg_color_pyramid_seed_layout",
                &BindGroupLayoutEntries::sequential(c, (uniform(), tex(true), out(PYRAMID))),
            ),
            inpaint: BindGroupLayoutDescriptor::new(
                "fsr_fg_inpaint_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (uniform(), tex(true), tex(true), out(IMAGE)),
                ),
            ),
        }
    }
}

/// The present copy (`fg_present.wgsl`), drawn inside the backdrop camera's 2D main pass as its
/// ground, specialised on that pass's target format and sample count.
#[derive(Resource)]
pub struct FgPresentPipeline {
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

impl SpecializedRenderPipeline for FgPresentPipeline {
    type Key = (TextureFormat, u32);

    fn specialize(&self, (format, samples): Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("fsr_fg_present".into()),
            layout: vec![self.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vs_main".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fs_main".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            // The 2D pass's depth attachment, never tested or written: the ground is under all,
            // as the combine it stands in for (`ffx_glow.rs`, `FfxCombineKey::inside_2d`).
            depth_stencil: Some(DepthStencilState {
                format: CORE_2D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: samples,
                ..default()
            },
            ..default()
        }
    }
}

/// `RenderStartup`: queue the nine compute pipelines and set up the present copy.
pub(super) fn init_pipelines(
    mut commands: Commands,
    shaders: Option<Res<FsrShaders>>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    mut present_pipelines: ResMut<SpecializedRenderPipelines<FgPresentPipeline>>,
) {
    let Some(shaders) = shaders else {
        return;
    };
    let layouts = FgLayouts::new(shaders.formats);
    let queue =
        |label: &'static str, layout: &BindGroupLayoutDescriptor, shader: &Handle<Shader>| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: vec![layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some("main".into()),
                zero_initialize_workgroup_memory: true,
            })
        };
    let s = &shaders.framegen;
    let l = &layouts;
    let pipelines = FgPipelines {
        reconstruct_depth: queue("fsr_fg_reconstruct_depth", &l.reconstruct_depth, &s[0]),
        game_vector_field: queue("fsr_fg_game_vector_field", &l.game_vector_field, &s[1]),
        mv_pyramid_seed: queue("fsr_fg_mv_pyramid_seed", &l.mv_pyramid_seed, &s[2]),
        mv_pyramid_reduce: queue("fsr_fg_mv_pyramid_reduce", &l.pyramid_reduce, &s[3]),
        disocclusion: queue("fsr_fg_disocclusion", &l.disocclusion, &s[4]),
        interpolate: queue("fsr_fg_interpolate", &l.interpolate, &s[5]),
        color_pyramid_seed: queue("fsr_fg_color_pyramid_seed", &l.color_pyramid_seed, &s[6]),
        color_pyramid_reduce: queue("fsr_fg_color_pyramid_reduce", &l.pyramid_reduce, &s[7]),
        inpaint: queue("fsr_fg_inpaint", &l.inpaint, &s[8]),
        layouts,
        sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("fsr_fg_linear_clamp"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        }),
    };
    commands.insert_resource(pipelines);
    let present = FgPresentPipeline {
        layout: BindGroupLayoutDescriptor::new(
            "fsr_fg_present_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer_sized(false, std::num::NonZero::new(PARAMS_SIZE)),
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
            ),
        ),
        shader: s[9].clone(),
    };
    // Warm: the player-UI camera's target (a non-HDR `Camera2d`, single-sampled) is the one ground
    // the client presents on, so its copy compiles at startup, behind the entry cover.
    present_pipelines.specialize(
        &pipeline_cache,
        &present,
        (TextureFormat::Rgba8UnormSrgb, 1),
    );
    commands.insert_resource(present);
}

/// `PrepareResources`, after [`super::upscaler::prepare_fsr`]: for a world view with frame
/// generation on and a backdrop camera claiming it, allocate the chain in that camera's format,
/// advance it by one rendered frame (parity, reset, parameters), and publish [`FgPresent`]. A held
/// frame advances nothing and presents the real frame the last world frame composed; so does a
/// world frame no held frame follows ([`FgSolo`]), with the one it composes itself.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare_framegen(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    held: Res<FsrHeld>,
    solo: Res<FgSolo>,
    shaders: Option<Res<FsrShaders>>,
    present: Option<Res<FgPresentPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mut present_pipelines: ResMut<SpecializedRenderPipelines<FgPresentPipeline>>,
    mut views: Query<(
        Entity,
        &MainEntity,
        &FsrView,
        &ExtractedView,
        Option<&FsrState>,
        Option<&mut FgState>,
    )>,
    backdrops: Query<(&FfxBackdrop, &ViewTarget, &Msaa)>,
) {
    let (Some(shaders), Some(present)) = (shaders, present) else {
        return;
    };
    for (entity, main, fsr, view, fsr_state, fg) in &mut views {
        // The claiming backdrop's target: the format the compose writes and the present reads.
        let claim = backdrops
            .iter()
            .find(|(b, _, _)| b.source == Some(main.id()))
            .map(|(b, target, msaa)| (target.main_texture_format(), msaa.samples(), b.decode));
        let (Some(fsr_state), Some((format, samples, decode)), true) =
            (fsr_state, claim, fsr.frame_generation)
        else {
            if fg.is_some() {
                commands.entity(entity).remove::<(FgState, FgPresent)>();
            }
            continue;
        };
        let (render, display) = (fsr_state.render_size(), fsr_state.display_size());
        let keep = fg.as_ref().is_some_and(|s| {
            s.targets.render == render
                && s.targets.display == display
                && s.targets.format == format
                && s.decode == decode
        });
        let mut created = None;
        let state: &mut FgState = match fg {
            Some(s) if keep => s.into_inner(),
            _ => created.insert(FgState {
                targets: Targets::new(&device, shaders.formats, render, display, format),
                params: device.create_buffer(&BufferDescriptor {
                    label: Some("fsr_fg_params"),
                    size: PARAMS_SIZE,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                parity: 0,
                frame_index_since_reset: 0,
                composed: Arc::new(AtomicBool::new(false)),
                generated: Arc::new(AtomicBool::new(false)),
                decode,
            }),
        };

        if !held.0 {
            // The other slot holds the last world frame's image only if it was composed; the
            // upscaler's reset (a cut, a new epoch) resets the pair too.
            let prev_composed = state.composed.load(Ordering::Acquire);
            if !prev_composed || fsr_state.frame_index() == 0 {
                state.frame_index_since_reset = 0;
            }
            state.parity ^= 1;
            // New flags each world frame: the last frame's present keeps its own.
            state.composed = Arc::new(AtomicBool::new(false));
            state.generated = Arc::new(AtomicBool::new(false));
            let (factors, tan_half_fov) = depth_factors(view.clip_from_view);
            let (r, d) = (render.as_vec2(), display.as_vec2());
            let params = FgParams {
                render_size: r.to_array(),
                display_size: d.to_array(),
                inv_render_size: (1.0 / r).to_array(),
                inv_display_size: (1.0 / d).to_array(),
                device_to_view_depth: factors,
                tan_half_fov,
                frame_index_since_reset: state.frame_index_since_reset as f32,
                srgb_output: if format.is_srgb() { 1.0 } else { 0.0 },
                mv_pyramid_mips: state.targets.mv_pyramid.sizes.len() as f32,
                color_pyramid_mips: state.targets.color_pyramid.sizes.len() as f32,
                _pad: [0.0; 3],
            };
            queue.write_buffer(&state.params, 0, bytemuck::bytes_of(&params));
            state.frame_index_since_reset = state.frame_index_since_reset.saturating_add(1);
        }

        // A world frame grounds on the generated frame, the held frame after it on the real one,
        // and a solo world frame on the real one it composes.
        let (source, ready) = if held.0 || solo.0 {
            (
                &state.targets.finals[state.parity].sampled,
                state.composed.clone(),
            )
        } else {
            (&state.targets.output, state.generated.clone())
        };
        let bind = device.create_bind_group(
            "fsr_fg_present",
            &pipeline_cache.get_bind_group_layout(&present.layout),
            &BindGroupEntries::sequential((state.params.as_entire_binding(), source)),
        );
        let pipeline = present_pipelines.specialize(&pipeline_cache, &present, (format, samples));
        let mut e = commands.entity(entity);
        e.insert(FgPresent {
            pipeline,
            bind,
            ready,
        });
        if let Some(created) = created {
            e.insert(created);
        }
    }
}

/// `PostProcess`, after the FFXGlow filter passes: compose this world frame into `WorldFinal[p]`
/// (the FFXGlow combine of the upscaled world, `ffx_glow::compose_world_final`), then run the
/// chain from `WorldFinal[1 − p]` and `WorldFinal[p]` to the generated frame. A solo world frame
/// ([`FgSolo`]) composes and stops: nothing presents a generated frame.
pub(super) fn fsr_framegen_pass(
    world: &World,
    view: ViewQuery<(&FgState, &FsrState)>,
    held: Res<FsrHeld>,
    solo: Res<FgSolo>,
    pipelines: Option<Res<FgPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    mut render_context: RenderContext,
) {
    if held.0 {
        return;
    }
    let Some(pipelines) = pipelines else {
        return;
    };
    let entity = view.entity();
    let (state, fsr) = view.into_inner();
    let Some(upscaled) = fsr.output() else {
        return; // the upscaler did not run: nothing to compose
    };
    let get = |id| pipeline_cache.get_compute_pipeline(id);
    let (
        Some(reconstruct_depth),
        Some(game_vector_field),
        Some(mv_seed),
        Some(mv_reduce),
        Some(disocclusion),
        Some(interpolate),
        Some(color_seed),
        Some(color_reduce),
        Some(inpaint),
    ) = (
        get(pipelines.reconstruct_depth),
        get(pipelines.game_vector_field),
        get(pipelines.mv_pyramid_seed),
        get(pipelines.mv_pyramid_reduce),
        get(pipelines.disocclusion),
        get(pipelines.interpolate),
        get(pipelines.color_pyramid_seed),
        get(pipelines.color_pyramid_reduce),
        get(pipelines.inpaint),
    )
    else {
        return; // still compiling: the backdrop falls back to the combine this frame
    };
    if !crate::ffx_glow::compose_world_final(
        world,
        entity,
        upscaled,
        state.final_attachment(),
        &mut render_context,
    ) {
        return;
    }
    state.composed.store(true, Ordering::Release);
    if solo.0 {
        return;
    }

    let t = &state.targets;
    let p = state.parity;
    let (dilated_mv, depths, prev_depth) = fsr.shared();
    let device = render_context.render_device().clone();
    let l = &pipelines.layouts;
    let layout = |d: &BindGroupLayoutDescriptor| pipeline_cache.get_bind_group_layout(d);
    let params = state.params.as_entire_binding();
    let smp = &pipelines.sampler;
    let (prev_final, cur_final) = (&t.finals[1 - p].sampled, &t.finals[p].sampled);

    let g_reconstruct = device.create_bind_group(
        "fsr_fg_reconstruct_depth",
        &layout(&l.reconstruct_depth),
        &BindGroupEntries::sequential((
            params.clone(),
            dilated_mv,
            depths,
            t.interpolated_depth.as_entire_binding(),
        )),
    );
    let g_field = device.create_bind_group(
        "fsr_fg_game_vector_field",
        &layout(&l.game_vector_field),
        &BindGroupEntries::sequential((
            params.clone(),
            dilated_mv,
            depths,
            prev_final,
            cur_final,
            smp,
            t.field_x.as_entire_binding(),
            t.field_y.as_entire_binding(),
        )),
    );
    let g_mv_seed = device.create_bind_group(
        "fsr_fg_mv_pyramid_seed",
        &layout(&l.mv_pyramid_seed),
        &BindGroupEntries::sequential((
            params.clone(),
            t.field_x.as_entire_binding(),
            t.field_y.as_entire_binding(),
            &t.mv_pyramid.levels[0],
        )),
    );
    let g_disocclusion = device.create_bind_group(
        "fsr_fg_disocclusion",
        &layout(&l.disocclusion),
        &BindGroupEntries::sequential((
            params.clone(),
            depths,
            t.interpolated_depth.as_entire_binding(),
            prev_depth.as_entire_binding(),
            t.field_x.as_entire_binding(),
            t.field_y.as_entire_binding(),
            &t.mv_pyramid.sampled,
            &t.disocclusion,
        )),
    );
    let g_interpolate = device.create_bind_group(
        "fsr_fg_interpolate",
        &layout(&l.interpolate),
        &BindGroupEntries::sequential((
            params.clone(),
            prev_final,
            cur_final,
            &t.disocclusion,
            t.field_x.as_entire_binding(),
            t.field_y.as_entire_binding(),
            &t.mv_pyramid.sampled,
            smp,
            &t.interp,
        )),
    );
    let g_color_seed = device.create_bind_group(
        "fsr_fg_color_pyramid_seed",
        &layout(&l.color_pyramid_seed),
        &BindGroupEntries::sequential((params.clone(), &t.interp, &t.color_pyramid.levels[0])),
    );
    let g_inpaint = device.create_bind_group(
        "fsr_fg_inpaint",
        &layout(&l.inpaint),
        &BindGroupEntries::sequential((
            params.clone(),
            &t.interp,
            &t.color_pyramid.sampled,
            &t.output,
        )),
    );
    let reduce_groups = |pyramid: &Pyramid| -> Vec<BindGroup> {
        (1..pyramid.levels.len())
            .map(|i| {
                device.create_bind_group(
                    "fsr_fg_pyramid_reduce",
                    &layout(&l.pyramid_reduce),
                    &BindGroupEntries::sequential((
                        params.clone(),
                        &pyramid.levels[i - 1],
                        &pyramid.levels[i],
                    )),
                )
            })
            .collect()
    };
    let (mv_reduce_groups, color_reduce_groups) = (
        reduce_groups(&t.mv_pyramid),
        reduce_groups(&t.color_pyramid),
    );
    // The params block records whether this frame interpolates; a reset frame (index 0) skips the
    // preparation passes and the interpolation copies the current frame (`bExecutePreparationPasses`).
    let prepare = state.frame_index_since_reset > 1;

    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = render_context.command_encoder();
    // All three scatter targets accumulate from "nothing here".
    encoder.clear_buffer(&t.interpolated_depth, 0, None);
    encoder.clear_buffer(&t.field_x, 0, None);
    encoder.clear_buffer(&t.field_y, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fsr_framegen"),
            timestamp_writes: None,
        });
        // One span for the chain: the journal's `gpu_fg` column.
        let span = diagnostics.pass_span(&mut pass, "fsr_framegen");
        let mut dispatch = |pipeline: &ComputePipeline, group: &BindGroup, size: UVec2| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(size.x.div_ceil(WORKGROUP), size.y.div_ceil(WORKGROUP), 1);
        };
        if prepare {
            dispatch(reconstruct_depth, &g_reconstruct, t.render);
            dispatch(game_vector_field, &g_field, t.render);
            dispatch(mv_seed, &g_mv_seed, t.mv_pyramid.sizes[0]);
            for (i, group) in mv_reduce_groups.iter().enumerate() {
                dispatch(mv_reduce, group, t.mv_pyramid.sizes[i + 1]);
            }
            dispatch(disocclusion, &g_disocclusion, t.render);
        }
        dispatch(interpolate, &g_interpolate, t.display);
        dispatch(color_seed, &g_color_seed, t.color_pyramid.sizes[0]);
        for (i, group) in color_reduce_groups.iter().enumerate() {
            dispatch(color_reduce, group, t.color_pyramid.sizes[i + 1]);
        }
        dispatch(inpaint, &g_inpaint, t.display);
        span.end(&mut pass);
    }
    state.generated.store(true, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter block is the size and alignment `fg_common.wgsl` declares.
    #[test]
    fn the_params_block_matches_the_shader() {
        assert_eq!(PARAMS_SIZE, 8 * 4 + 16 + 8 * 4);
        assert_eq!(PARAMS_SIZE % 16, 0);
    }

    /// `view_space_depth(d) = y / (d − x)` is the eye distance in metres on a reverse-Z infinite
    /// projection, and a cleared 0 stays finite.
    #[test]
    fn the_depth_factors_invert_reverse_z() {
        let proj = Mat4::perspective_infinite_reverse_rh(1.0, 1.6, 0.2);
        let ([x, y, tan_x, tan_y], tan_half_fov) = depth_factors(proj);
        for dist in [0.5_f32, 7.0, 300.0] {
            let clip = proj * Vec4::new(0.0, 0.0, -dist, 1.0);
            let d = clip.z / clip.w;
            let metres = y / (d - x);
            assert!(
                (metres - dist * VIEW_SPACE_TO_METERS).abs() / dist < 1e-3,
                "{dist}"
            );
        }
        assert!((y / (0.0 - x)).is_finite());
        assert!((tan_y - 0.5_f32.tan()).abs() < 1e-5);
        assert!((tan_x - 1.6 * 0.5_f32.tan()).abs() < 1e-5);
        assert_eq!(tan_half_fov, tan_x);
    }

    /// Every level halves the one below, the chain stops before a side would reach zero, and the
    /// level count never passes the reference's bound.
    #[test]
    fn the_pyramid_sizes_halve_down_to_one() {
        let sizes = pyramid_sizes(UVec2::new(1920, 1080), COLOR_PYRAMID_MAX_LEVELS);
        assert_eq!(sizes[0], UVec2::new(960, 540));
        assert_eq!(sizes.len(), COLOR_PYRAMID_MAX_LEVELS as usize);
        for pair in sizes.windows(2) {
            assert_eq!(pair[1], pair[0] / 2);
        }
        // A thin source stops early rather than clamping a side at one.
        let thin = pyramid_sizes(UVec2::new(64, 8), MV_PYRAMID_MAX_LEVELS);
        assert_eq!(
            thin,
            [UVec2::new(32, 4), UVec2::new(16, 2), UVec2::new(8, 1)]
        );
    }
}
