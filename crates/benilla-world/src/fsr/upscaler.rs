//! The FSR 3.1.5 temporal upscaler on the world camera, as `Core3d` systems.
//!
//! A port of physics-rs's `render/fsr3.rs` (FidelityFX SDK 3.1.5, MIT; `LICENSE-FIDELITYFX.txt`)
//! to Bevy's render world. The world renders at render resolution; this chain reconstructs a
//! display-resolution image, [`FsrOutput`], with the main texture's own colour semantics, which the
//! FFXGlow filter and the UI camera's combine read in place of the main texture.
//!
//! The passes, in the SDK's order (`shaders/fsr/*.wgsl`):
//!
//! 0. `motion` — benilla's input adapter: camera motion by depth reprojection, moving content from
//!    the velocity target, and the generated reactive mask.
//! 1. `prepare_inputs` — depth extents, dilated motion, luma, the reconstructed previous depth.
//! 2. `shading_change_pyramid` and 3. `shading_change` — the shading-change detector.
//! 4. `prepare_reactivity` — disocclusion, reactivity, shading change, accumulation, new locks.
//! 5. `luma_instability`.
//! 6. `accumulate` — the upsample and blend, on the display grid.
//! 7. `rcas` — the sharpening, into [`FsrOutput`].
//!
//! Deviations from the SDK are physics-rs's (no SPD, a storage-buffer depth scatter, the widened
//! formats without tier 1; see its module doc) plus: the HDR permutation is chosen per frame by a
//! uniform, and the reactive mask is generated inside the adapter.

use bevy::prelude::*;
use bevy::render::camera::{ExtractedCamera, TemporalJitter};
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_depth_2d,
    texture_storage_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget};
use bytemuck::{Pod, Zeroable};

use super::caps::Formats;
use super::{FsrHeld, FsrShaders, FsrView, RcasSharpness};

/// Whether shading-change detection is compiled in (passes 2 and 3's detector half): the one
/// optional part of FSR 3.1, and its most expensive. On, as the SDK ships it.
pub const SHADING_CHANGE_DETECTION: bool = true;

/// `ffxFsr3UpscalerSetConfiguration`'s three constants, at their SDK defaults.
const VELOCITY_FACTOR: f32 = 1.0;
const ACCUMULATION_ADDED_PER_FRAME: f32 = 1.0 / 3.0;
const MIN_DISOCCLUSION_ACCUMULATION: f32 = -1.0 / 3.0;

/// `ViewSpaceToMetersFactor`: a world unit is one yard.
pub const VIEW_SPACE_TO_METERS: f32 = 0.9144;

/// `GenReactiveScale` for the generated reactive mask: the SDK sample's 1.0.
const REACTIVE_SCALE: f32 = 1.0;

/// A camera move this far in one rendered frame, in yards, is a cut: the history describes another
/// place.
const CUT_DISTANCE: f32 = 50.0;
/// …or this far round, in radians (60°).
const CUT_ANGLE: f32 = std::f32::consts::FRAC_PI_3;

/// `WOW_FSR_DEBUG=views`'s counters of [`fsr_upscale_pass`]: `[ran, pipelines compiling, depth not
/// sampleable]`, so a chain that never lands says why from numbers.
pub(super) static UPSCALE_COUNTS: [std::sync::atomic::AtomicU32; 3] =
    [const { std::sync::atomic::AtomicU32::new(0) }; 3];

fn count(slot: usize) {
    UPSCALE_COUNTS[slot].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Compute workgroup edge; `@workgroup_size(8, 8, 1)` in every pass.
const WORKGROUP: u32 = 8;

/// The half-resolution grid the shading-change detector runs on (`ShadingChangeRenderSize`).
fn half_size(render: UVec2) -> UVec2 {
    (render / 2).max(UVec2::ONE)
}

/// Per-frame constants shared by every pass; `Fsr3Params` in `shaders/fsr/common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Fsr3Params {
    cur_rel_from_clip: [[f32; 4]; 4],
    clip_from_cur_rel: [[f32; 4]; 4],
    prev_clip_from_cur_rel: [[f32; 4]; 4],
    render_size: [f32; 2],
    display_size: [f32; 2],
    inv_render_size: [f32; 2],
    inv_display_size: [f32; 2],
    jitter: [f32; 2],
    prev_jitter: [f32; 2],
    downscale_factor: [f32; 2],
    depth_linear: [f32; 2],
    exposure: f32,
    frame_index: f32,
    jitter_sequence_length: f32,
    view_space_to_meters: f32,
    velocity_factor: f32,
    accumulation_added_per_frame: f32,
    min_disocclusion_accumulation: f32,
    sharpness: f32,
    hdr: f32,
    reactive_scale: f32,
    _pad: [f32; 2],
}

pub(super) const PARAMS_SIZE: u64 = std::mem::size_of::<Fsr3Params>() as u64;

/// The chain's display-resolution result: the main texture's colour semantics at the display size.
/// On the world view while FSR runs; the FFXGlow filter and the UI camera's combine read it.
#[derive(Component, Clone)]
pub struct FsrOutput {
    pub view: TextureView,
    pub size: UVec2,
}

/// The velocity target `velocity.rs` clears and draws moving content into: uv motion in `xy`,
/// the written flag in `w`, at render resolution.
#[derive(Component, Clone)]
pub struct FsrVelocityTarget {
    pub view: TextureView,
}

/// Everything the chain allocates for one view. The `[_; 2]` pairs carry state between frames and
/// ping-pong on one parity: a frame writing slot `p` reads slot `1 - p`.
struct Targets {
    render: UVec2,
    display: UVec2,
    /// The colour before the transparent pass (`fsr_opaque_copy`), for the reactive mask.
    opaque: Texture,
    opaque_view: TextureView,
    velocity: TextureView,
    mv: TextureView,
    dilated_mv: TextureView,
    /// `x` = dilated (nearest) device depth, `y` = farthest depth in yards·0.9144.
    depths: TextureView,
    luma: [TextureView; 2],
    luma_history: [TextureView; 2],
    luma_instability: TextureView,
    masks: TextureView,
    accumulation: [TextureView; 2],
    /// Reconstructed previous-frame depth: a storage buffer, since the scatter needs an atomic.
    prev_depth: Buffer,
    shading_change_mips: [TextureView; 3],
    farthest_mip1: TextureView,
    shading_change: TextureView,
    history: [TextureView; 2],
    new_locks: TextureView,
    output: TextureView,
}

impl Targets {
    fn new(device: &RenderDevice, formats: Formats, render: UVec2, display: UVec2) -> Self {
        let half = half_size(render);
        // Each pyramid level is the one above halved, rounded up.
        let m1 = (half + UVec2::ONE) / 2;
        let m2 = (m1 + UVec2::ONE) / 2;
        let make =
            |label: &'static str, format: TextureFormat, usage: TextureUsages, size: UVec2| {
                device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: size.x.max(1),
                        height: size.y.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
        let view = |t: Texture| t.create_view(&TextureViewDescriptor::default());
        // Written by a compute pass, read by a later one.
        const INTERNAL: TextureUsages =
            TextureUsages::STORAGE_BINDING.union(TextureUsages::TEXTURE_BINDING);
        // Also cleared at the start of a frame, which without `CLEAR_TEXTURE` is an empty render
        // pass.
        const CLEARABLE: TextureUsages = INTERNAL.union(TextureUsages::RENDER_ATTACHMENT);
        let internal = |label, format, size| view(make(label, format, INTERNAL, size));
        let clearable = |label, format, size| view(make(label, format, CLEARABLE, size));
        let (f16, masks) = (TextureFormat::Rgba16Float, TextureFormat::Rgba8Unorm);

        let opaque = make(
            "fsr_opaque_only",
            TextureFormat::Rgba16Float,
            TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            render,
        );
        Self {
            render,
            display,
            opaque_view: opaque.create_view(&TextureViewDescriptor::default()),
            opaque,
            velocity: view(make(
                "fsr_velocity",
                TextureFormat::Rgba16Float,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                render,
            )),
            mv: internal("fsr_motion_vectors", f16, render),
            dilated_mv: internal("fsr_dilated_motion", formats.rg16, render),
            depths: internal("fsr_depths", TextureFormat::Rg32Float, render),
            luma: [
                clearable("fsr_luma_0", formats.r16, render),
                clearable("fsr_luma_1", formats.r16, render),
            ],
            luma_history: [
                clearable("fsr_luma_history_0", f16, render),
                clearable("fsr_luma_history_1", f16, render),
            ],
            luma_instability: internal("fsr_luma_instability", formats.r16, render),
            masks: internal("fsr_dilated_reactive_masks", masks, render),
            accumulation: [
                clearable("fsr_accumulation_0", formats.r8, render),
                clearable("fsr_accumulation_1", formats.r8, render),
            ],
            prev_depth: device.create_buffer(&BufferDescriptor {
                label: Some("fsr_reconstructed_depth"),
                size: u64::from(render.x.max(1)) * u64::from(render.y.max(1)) * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            shading_change_mips: [
                internal("fsr_shading_change_mip_0", formats.rg16, half),
                internal("fsr_shading_change_mip_1", formats.rg16, m1),
                internal("fsr_shading_change_mip_2", formats.rg16, m2),
            ],
            farthest_mip1: internal("fsr_farthest_depth_mip_1", formats.r16, half),
            shading_change: internal("fsr_shading_change", formats.r8, half),
            history: [
                internal("fsr_history_0", f16, display),
                internal("fsr_history_1", f16, display),
            ],
            new_locks: clearable("fsr_new_locks", formats.r8, display),
            output: internal("fsr_output", f16, display),
        }
    }
}

/// One view's FSR state in the render world: its targets, its parameter block, and the history
/// bookkeeping. Advanced once per rendered world frame by [`prepare_fsr`].
#[derive(Component)]
pub struct FsrState {
    targets: Targets,
    params: Buffer,
    /// The slot this frame writes.
    parity: usize,
    /// Frames accumulated since the last reset; 0 makes every pixel a new sample.
    frame_index: u32,
    /// This frame must zero the slots it reads before running.
    reset_pending: bool,
    /// Last frame's eye, rotation and unjittered projection.
    prev: Option<(Vec3, Quat, Mat4)>,
    epoch: u32,
    modern: bool,
    /// [`FsrOutput`] holds this rendered frame's upscale: false from `prepare_fsr` until the chain
    /// lands, and kept through a held frame.
    output_ready: bool,
    /// This frame's camera-relative unjittered clip matrices, current and previous
    /// ([`frame_matrices`]), for the effect lane's motion.
    rel: [Mat4; 2],
    /// This frame's clip-from-world matrices: jittered (the raster), unjittered, and last frame's
    /// unjittered, for the model parts' motion.
    clip: [Mat4; 3],
}

impl FsrState {
    /// The upscaled image, once the chain has written it this rendered frame (or, on a held frame,
    /// the last one).
    pub fn output(&self) -> Option<&TextureView> {
        self.output_ready.then_some(&self.targets.output)
    }

    /// Rendered frames accumulated since the last reset; 0 on a reset frame.
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// `[clip_from_cur_rel, prev_clip_from_cur_rel]`, unjittered and camera-relative.
    pub fn rel_matrices(&self) -> [Mat4; 2] {
        self.rel
    }

    /// `[raster, current, previous]` clip-from-world: the jittered matrix the frame rasterized
    /// with, and the unjittered current and previous ones.
    pub fn clip_from_world(&self) -> [Mat4; 3] {
        self.clip
    }

    /// The dilated motion, dilated depths and reconstructed previous depth, which frame generation
    /// shares rather than derives twice (`ffxFrameInterpolationPrepare`).
    pub fn shared(&self) -> (&TextureView, &TextureView, &Buffer) {
        (
            &self.targets.dilated_mv,
            &self.targets.depths,
            &self.targets.prev_depth,
        )
    }

    pub fn render_size(&self) -> UVec2 {
        self.targets.render
    }

    pub fn display_size(&self) -> UVec2 {
        self.targets.display
    }

    /// Whether the history this frame accumulated onto was valid.
    pub fn history_valid(&self) -> bool {
        self.frame_index > 1
    }

    pub fn params(&self) -> &Buffer {
        &self.params
    }
}

/// The pipelines and layouts, built at startup from [`FsrShaders`].
#[derive(Resource)]
pub struct FsrPipelines {
    pub formats: Formats,
    layouts: Layouts,
    motion: CachedComputePipelineId,
    prepare_inputs: CachedComputePipelineId,
    shading_change_pyramid: CachedComputePipelineId,
    shading_change: CachedComputePipelineId,
    prepare_reactivity: CachedComputePipelineId,
    luma_instability: CachedComputePipelineId,
    accumulate: CachedComputePipelineId,
    rcas: CachedComputePipelineId,
    sampler: Sampler,
}

impl FsrPipelines {
    /// Every pipeline, for the warm-up to wait on.
    pub fn ids(&self) -> [CachedComputePipelineId; 8] {
        [
            self.motion,
            self.prepare_inputs,
            self.shading_change_pyramid,
            self.shading_change,
            self.prepare_reactivity,
            self.luma_instability,
            self.accumulate,
            self.rcas,
        ]
    }
}

struct Layouts {
    motion: BindGroupLayoutDescriptor,
    prepare_inputs: BindGroupLayoutDescriptor,
    shading_change_pyramid: BindGroupLayoutDescriptor,
    shading_change: BindGroupLayoutDescriptor,
    prepare_reactivity: BindGroupLayoutDescriptor,
    luma_instability: BindGroupLayoutDescriptor,
    accumulate: BindGroupLayoutDescriptor,
    rcas: BindGroupLayoutDescriptor,
}

impl Layouts {
    fn new(formats: Formats) -> Self {
        use StorageTextureAccess::WriteOnly;
        let uniform = || uniform_buffer_sized(false, std::num::NonZero::new(PARAMS_SIZE));
        // Loaded textures bind non-filterable; sampled ones need a filterable format and are
        // allocated in one.
        let tex = |filterable| texture_2d(TextureSampleType::Float { filterable });
        let out = |format| texture_storage_2d(format, WriteOnly);
        let lin = || sampler(SamplerBindingType::Filtering);
        let (f16, rg32, masks) = (
            TextureFormat::Rgba16Float,
            TextureFormat::Rg32Float,
            TextureFormat::Rgba8Unorm,
        );
        let c = ShaderStages::COMPUTE;
        Self {
            motion: BindGroupLayoutDescriptor::new(
                "fsr_motion_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        texture_depth_2d(),
                        tex(false),
                        tex(false),
                        tex(false),
                        out(f16),
                    ),
                ),
            ),
            prepare_inputs: BindGroupLayoutDescriptor::new(
                "fsr_prepare_inputs_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        texture_depth_2d(),
                        tex(false),
                        tex(false),
                        out(formats.rg16),
                        out(rg32),
                        out(formats.r16),
                        storage_buffer_sized(false, None),
                    ),
                ),
            ),
            shading_change_pyramid: BindGroupLayoutDescriptor::new(
                "fsr_shading_change_pyramid_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(false),
                        tex(false),
                        tex(false),
                        tex(false),
                        out(formats.rg16),
                        out(formats.rg16),
                        out(formats.rg16),
                        out(formats.r16),
                    ),
                ),
            ),
            shading_change: BindGroupLayoutDescriptor::new(
                "fsr_shading_change_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(true),
                        tex(true),
                        tex(true),
                        lin(),
                        out(formats.r8),
                    ),
                ),
            ),
            prepare_reactivity: BindGroupLayoutDescriptor::new(
                "fsr_prepare_reactivity_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(false),
                        tex(false),
                        storage_buffer_read_only_sized(false, None),
                        tex(false),
                        tex(true),
                        tex(false),
                        tex(true),
                        lin(),
                        out(masks),
                        out(formats.r8),
                        out(formats.r8),
                    ),
                ),
            ),
            luma_instability: BindGroupLayoutDescriptor::new(
                "fsr_luma_instability_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(false),
                        tex(true),
                        tex(true),
                        tex(true),
                        lin(),
                        out(f16),
                        out(formats.r16),
                    ),
                ),
            ),
            accumulate: BindGroupLayoutDescriptor::new(
                "fsr_accumulate_layout",
                &BindGroupLayoutEntries::sequential(
                    c,
                    (
                        uniform(),
                        tex(false),
                        tex(false),
                        tex(true),
                        tex(true),
                        tex(true),
                        tex(false),
                        tex(false),
                        lin(),
                        out(f16),
                    ),
                ),
            ),
            rcas: BindGroupLayoutDescriptor::new(
                "fsr_rcas_layout",
                &BindGroupLayoutEntries::sequential(c, (uniform(), tex(false), out(f16))),
            ),
        }
    }
}

/// `RenderStartup`: queue the eight compute pipelines against the shaders the plugin built for
/// this device's formats.
pub(super) fn init_pipelines(
    mut commands: Commands,
    shaders: Option<Res<FsrShaders>>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(shaders) = shaders else {
        return;
    };
    let layouts = Layouts::new(shaders.formats);
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
    let s = &shaders.upscaler;
    let pipelines = FsrPipelines {
        formats: shaders.formats,
        motion: queue("fsr_motion", &layouts.motion, &s[0]),
        prepare_inputs: queue("fsr_prepare_inputs", &layouts.prepare_inputs, &s[1]),
        shading_change_pyramid: queue(
            "fsr_shading_change_pyramid",
            &layouts.shading_change_pyramid,
            &s[2],
        ),
        shading_change: queue("fsr_shading_change", &layouts.shading_change, &s[3]),
        prepare_reactivity: queue("fsr_prepare_reactivity", &layouts.prepare_reactivity, &s[4]),
        luma_instability: queue("fsr_luma_instability", &layouts.luma_instability, &s[5]),
        accumulate: queue("fsr_accumulate", &layouts.accumulate, &s[6]),
        rcas: queue("fsr_rcas", &layouts.rcas, &s[7]),
        layouts,
        sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("fsr_linear_clamp"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        }),
    };
    commands.insert_resource(pipelines);
}

/// The camera-relative matrices of one frame (`common.wgsl`'s first three members), from the
/// eye, rotation and projections of this frame and the last. No world-space magnitude enters a
/// matrix: the eye delta is the only translation, so the reprojection is exact far from the origin.
pub(crate) fn frame_matrices(
    eye: Vec3,
    rot: Quat,
    proj: Mat4,
    jittered: Mat4,
    prev: (Vec3, Quat, Mat4),
) -> [Mat4; 3] {
    let (prev_eye, prev_rot, prev_proj) = prev;
    let view_rot = Mat4::from_quat(rot);
    let cur_rel_from_clip = view_rot * jittered.inverse();
    let clip_from_cur_rel = proj * Mat4::from_quat(rot.inverse());
    let prev_clip_from_cur_rel =
        prev_proj * Mat4::from_quat(prev_rot.inverse()) * Mat4::from_translation(eye - prev_eye);
    [cur_rel_from_clip, clip_from_cur_rel, prev_clip_from_cur_rel]
}

/// `PrepareResources`: allocate or resize each FSR view's targets and advance its history by one
/// rendered frame — parity, reset, matrices, the parameter block — and publish [`FsrOutput`].
#[allow(clippy::type_complexity)]
pub(super) fn prepare_fsr(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipelines: Option<Res<FsrPipelines>>,
    held: Res<FsrHeld>,
    rcas: Option<Res<RcasSharpness>>,
    mut views: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        &FsrView,
        &Msaa,
        &ViewTarget,
        Option<&mut FsrState>,
    )>,
    mut warned_msaa: Local<bool>,
) {
    // A held frame renders no world view: its history must not advance.
    if held.0 {
        return;
    }
    let Some(pipelines) = pipelines else {
        return;
    };
    for (entity, view, camera, fsr, msaa, target, state) in &mut views {
        let Some(render) = camera.physical_viewport_size else {
            continue;
        };
        // The chain reads single-sampled depth and colour, and copies the main texture.
        if msaa.samples() > 1 || target.main_texture_format() != TextureFormat::Rgba16Float {
            if !*warned_msaa {
                warn!("fsr: the world view is multisampled or not HDR — the upscaler stays off");
                *warned_msaa = true;
            }
            commands
                .entity(entity)
                .remove::<(FsrState, FsrOutput, FsrVelocityTarget)>();
            continue;
        }
        let display = fsr.display.max(UVec2::ONE);
        let modern = crate::modern::frame_is_modern();
        let (eye, rot) = (
            view.world_from_view.translation(),
            view.world_from_view.rotation(),
        );
        let proj = view.clip_from_view;

        let keep = state
            .as_ref()
            .is_some_and(|s| s.targets.render == render && s.targets.display == display);
        let mut created = None;
        let state: &mut FsrState = match state {
            Some(s) if keep => s.into_inner(),
            _ => created.insert(FsrState {
                targets: Targets::new(&device, pipelines.formats, render, display),
                params: device.create_buffer(&BufferDescriptor {
                    label: Some("fsr_params"),
                    size: PARAMS_SIZE,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                parity: 0,
                frame_index: 0,
                // Fresh targets are zeroed.
                reset_pending: false,
                prev: None,
                epoch: fsr.epoch,
                modern,
                output_ready: false,
                rel: [Mat4::IDENTITY; 2],
                clip: [Mat4::IDENTITY; 3],
            }),
        };

        // History reset: a new lighting model or epoch (world entry, a loading screen), or a cut.
        let cut = state.prev.is_some_and(|(prev_eye, prev_rot, _)| {
            eye.distance(prev_eye) > CUT_DISTANCE || rot.angle_between(prev_rot) > CUT_ANGLE
        });
        if state.epoch != fsr.epoch || state.modern != modern || cut {
            state.epoch = fsr.epoch;
            state.modern = modern;
            state.frame_index = 0;
            state.reset_pending = true;
            state.prev = None;
        }

        state.parity ^= 1;
        let prev = state.prev.unwrap_or((eye, rot, proj));
        state.prev = Some((eye, rot, proj));

        let mut jittered = proj;
        TemporalJitter {
            offset: super::bevy_offset(fsr.jitter),
        }
        .jitter_projection(&mut jittered, render.as_vec2());
        let [cur_rel_from_clip, clip_from_cur_rel, prev_clip_from_cur_rel] =
            frame_matrices(eye, rot, proj, jittered, prev);
        state.output_ready = false;
        state.rel = [clip_from_cur_rel, prev_clip_from_cur_rel];
        let view_from_world = view.world_from_view.to_matrix().inverse();
        let prev_view_from_world = Mat4::from_rotation_translation(prev.1, prev.0).inverse();
        state.clip = [
            jittered * view_from_world,
            proj * view_from_world,
            prev.2 * prev_view_from_world,
        ];

        let (r, d) = (render.as_vec2(), display.as_vec2());
        let exposure = if modern {
            crate::modern::look::exposure().max(1e-6)
        } else {
            1.0
        };
        let params = Fsr3Params {
            cur_rel_from_clip: cur_rel_from_clip.to_cols_array_2d(),
            clip_from_cur_rel: clip_from_cur_rel.to_cols_array_2d(),
            prev_clip_from_cur_rel: prev_clip_from_cur_rel.to_cols_array_2d(),
            render_size: r.to_array(),
            display_size: d.to_array(),
            inv_render_size: (1.0 / r).to_array(),
            inv_display_size: (1.0 / d).to_array(),
            jitter: fsr.jitter.to_array(),
            prev_jitter: fsr.prev_jitter.to_array(),
            downscale_factor: (r / d).to_array(),
            // The SDK's reversed-infinite `DeviceToViewSpaceTransformFactors`, x negated: the
            // near plane is `proj[3][2]` on Bevy's reverse-Z infinite projection.
            depth_linear: [f32::EPSILON, proj.w_axis.z],
            exposure,
            frame_index: state.frame_index as f32,
            jitter_sequence_length: fsr.period as f32,
            view_space_to_meters: VIEW_SPACE_TO_METERS,
            velocity_factor: VELOCITY_FACTOR,
            accumulation_added_per_frame: ACCUMULATION_ADDED_PER_FRAME,
            min_disocclusion_accumulation: MIN_DISOCCLUSION_ACCUMULATION,
            // RCAS's sharpness in stops, where 0 is the most; the shader takes `exp2(-stops)`.
            sharpness: (-rcas.as_deref().copied().unwrap_or_default().stops()).exp2(),
            hdr: if modern { 1.0 } else { 0.0 },
            reactive_scale: REACTIVE_SCALE,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&state.params, 0, bytemuck::bytes_of(&params));

        let published = (
            FsrOutput {
                view: state.targets.output.clone(),
                size: display,
            },
            FsrVelocityTarget {
                view: state.targets.velocity.clone(),
            },
        );
        let mut e = commands.entity(entity);
        e.insert(published);
        if let Some(created) = created {
            e.insert(created);
        }
    }
}

/// `MainPass`, after the fog composite and before the transparent pass: the colour with every
/// opaque surface in it, the generated reactive mask's "before".
pub(super) fn fsr_opaque_copy(
    view: ViewQuery<(&ViewTarget, &FsrState)>,
    held: Res<FsrHeld>,
    mut render_context: RenderContext,
) {
    if held.0 {
        return;
    }
    let (target, state) = view.into_inner();
    let size = state.targets.render;
    render_context.command_encoder().copy_texture_to_texture(
        target.main_texture().as_image_copy(),
        state.targets.opaque.as_image_copy(),
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
}

/// `EarlyPostProcess`: the whole chain, main texture and depth in, [`FsrOutput`] out.
pub(super) fn fsr_upscale_pass(
    view: ViewQuery<(&ViewTarget, &ViewDepthTexture, &mut FsrState)>,
    pipelines: Option<Res<FsrPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    held: Res<FsrHeld>,
    residual: Option<Res<super::residual::Residual>>,
    mut render_context: RenderContext,
) {
    if held.0 {
        return;
    }
    let Some(pipelines) = pipelines else {
        return;
    };
    let (target, depth, mut state) = view.into_inner();
    let get = |id| pipeline_cache.get_compute_pipeline(id);
    let (
        Some(motion),
        Some(prepare_inputs),
        Some(pyramid),
        Some(shading_change),
        Some(reactivity),
        Some(instability),
        Some(accumulate),
        Some(rcas),
    ) = (
        get(pipelines.motion),
        get(pipelines.prepare_inputs),
        get(pipelines.shading_change_pyramid),
        get(pipelines.shading_change),
        get(pipelines.prepare_reactivity),
        get(pipelines.luma_instability),
        get(pipelines.accumulate),
        get(pipelines.rcas),
    )
    else {
        count(1);
        return; // still compiling: the backdrop falls back to the bilinear read this frame
    };
    // The depth is read by two passes, so it must be sampleable.
    if !depth
        .texture
        .usage()
        .contains(TextureUsages::TEXTURE_BINDING)
    {
        count(2);
        return;
    }

    let device = render_context.render_device().clone();
    let t = &state.targets;
    let p = state.parity;
    let q = 1 - p;
    let (render, half, display) = (t.render, half_size(t.render), t.display);
    let params = state.params.as_entire_binding();
    let layout = |d: &BindGroupLayoutDescriptor| pipeline_cache.get_bind_group_layout(d);
    let l = &pipelines.layouts;
    let smp = &pipelines.sampler;
    let color = target.main_texture_view();

    let g_motion = device.create_bind_group(
        "fsr_motion",
        &layout(&l.motion),
        &BindGroupEntries::sequential((
            params.clone(),
            depth.view(),
            &t.velocity,
            color,
            &t.opaque_view,
            &t.mv,
        )),
    );
    let g_prepare = device.create_bind_group(
        "fsr_prepare_inputs",
        &layout(&l.prepare_inputs),
        &BindGroupEntries::sequential((
            params.clone(),
            depth.view(),
            &t.mv,
            color,
            &t.dilated_mv,
            &t.depths,
            &t.luma[p],
            t.prev_depth.as_entire_binding(),
        )),
    );
    let g_pyramid = device.create_bind_group(
        "fsr_shading_change_pyramid",
        &layout(&l.shading_change_pyramid),
        &BindGroupEntries::sequential((
            params.clone(),
            &t.luma[p],
            &t.luma[q],
            &t.dilated_mv,
            &t.depths,
            &t.shading_change_mips[0],
            &t.shading_change_mips[1],
            &t.shading_change_mips[2],
            &t.farthest_mip1,
        )),
    );
    let g_shading = device.create_bind_group(
        "fsr_shading_change",
        &layout(&l.shading_change),
        &BindGroupEntries::sequential((
            params.clone(),
            &t.shading_change_mips[0],
            &t.shading_change_mips[1],
            &t.shading_change_mips[2],
            smp,
            &t.shading_change,
        )),
    );
    let g_reactivity = device.create_bind_group(
        "fsr_prepare_reactivity",
        &layout(&l.prepare_reactivity),
        &BindGroupEntries::sequential((
            params.clone(),
            &t.dilated_mv,
            &t.depths,
            t.prev_depth.as_entire_binding(),
            &t.mv,
            &t.shading_change,
            &t.luma[p],
            &t.accumulation[q],
            smp,
            &t.masks,
            &t.accumulation[p],
            &t.new_locks,
        )),
    );
    let g_instability = device.create_bind_group(
        "fsr_luma_instability",
        &layout(&l.luma_instability),
        &BindGroupEntries::sequential((
            params.clone(),
            &t.dilated_mv,
            &t.masks,
            &t.luma[p],
            &t.luma_history[q],
            smp,
            &t.luma_history[p],
            &t.luma_instability,
        )),
    );
    let g_accumulate = device.create_bind_group(
        "fsr_accumulate",
        &layout(&l.accumulate),
        &BindGroupEntries::sequential((
            params.clone(),
            color,
            &t.dilated_mv,
            &t.masks,
            &t.luma_instability,
            &t.farthest_mip1,
            &t.new_locks,
            &t.history[q],
            smp,
            &t.history[p],
        )),
    );
    let g_rcas = device.create_bind_group(
        "fsr_rcas",
        &layout(&l.rcas),
        &BindGroupEntries::sequential((params, &t.history[p], &t.output)),
    );

    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = render_context.command_encoder();

    // A load-clear render pass with no draws zeroes a storage texture without `CLEAR_TEXTURE`.
    let clear = |encoder: &mut CommandEncoder, label: &'static str, view: &TextureView| {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    };
    // The scatter keeps a maximum, so it starts from 0, the far plane.
    encoder.clear_buffer(&t.prev_depth, 0, None);
    // Locks are per-frame candidates; last frame's must not leak into this one.
    clear(encoder, "fsr_new_locks_clear", &t.new_locks);
    // A reset zeroes what this frame reads, the slots the previous frame wrote.
    if state.reset_pending {
        clear(encoder, "fsr_accumulation_clear", &t.accumulation[q]);
        clear(encoder, "fsr_luma_clear", &t.luma[q]);
        clear(encoder, "fsr_luma_history_clear", &t.luma_history[q]);
    }

    let groups = |size: UVec2| (size.x.div_ceil(WORKGROUP), size.y.div_ceil(WORKGROUP));
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fsr_upscale"),
            timestamp_writes: None,
        });
        // One span for the whole chain: the journal's `gpu_fsr` column.
        let span = diagnostics.pass_span(&mut pass, "fsr_upscale");
        let mut dispatch = |pipeline: &ComputePipeline, group: &BindGroup, size: UVec2| {
            let (x, y) = groups(size);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        };
        dispatch(motion, &g_motion, render);
        dispatch(prepare_inputs, &g_prepare, render);
        // Runs either way: with the detector off it is the farthest-depth downsample alone.
        dispatch(pyramid, &g_pyramid, half);
        if SHADING_CHANGE_DETECTION {
            dispatch(shading_change, &g_shading, half);
        }
        dispatch(reactivity, &g_reactivity, render);
        dispatch(instability, &g_instability, render);
        dispatch(accumulate, &g_accumulate, display);
        dispatch(rcas, &g_rcas, display);
        span.end(&mut pass);
    }
    // `WOW_FSR_DEBUG=residual`: this frame's motion vectors against the last frame.
    if let Some(residual) = residual.as_deref() {
        residual.record(
            &device,
            &pipeline_cache,
            encoder,
            super::residual::Inputs {
                params: state.params.as_entire_binding(),
                depth: depth.view(),
                velocity: &t.velocity,
                color: target.main_texture(),
                color_view: target.main_texture_view(),
                render,
            },
        );
    }

    count(0);
    state.reset_pending = false;
    state.output_ready = true;
    // The history this frame wrote is usable from the next one.
    state.frame_index = state.frame_index.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reverse_z(aspect: f32) -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(1.0, aspect, 0.2)
    }

    /// A static point reprojects to where it was: through the camera-relative matrices, a world
    /// point seen from two eyes 9000 yd from the origin lands within 0.01 px of the previous
    /// frame's direct projection.
    #[test]
    fn camera_relative_reprojection_is_exact_far_from_the_origin() {
        let size = Vec2::new(1920.0, 1080.0);
        let proj = reverse_z(size.x / size.y);
        let prev_eye = Vec3::new(9000.0, 120.0, -9000.0);
        let eye = prev_eye + Vec3::new(0.37, 0.02, -0.51);
        let prev_rot = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.1);
        let rot = Quat::from_rotation_y(0.41) * Quat::from_rotation_x(-0.1);
        let point = eye + rot * Vec3::new(3.0, -1.0, -40.0);

        let [cur_rel_from_clip, _, prev_clip_from_cur_rel] =
            frame_matrices(eye, rot, proj, proj, (prev_eye, prev_rot, proj));
        // The point's clip position this frame, then back through the matrices as the shader does.
        let view = rot.inverse() * (point - eye);
        let clip = proj * view.extend(1.0);
        let ndc = clip / clip.w;
        let rel = cur_rel_from_clip * ndc;
        let prev_clip = prev_clip_from_cur_rel * rel;
        let reprojected = prev_clip.truncate().truncate() / prev_clip.w;

        let prev_view = prev_rot.inverse() * (point - prev_eye);
        let direct = proj * prev_view.extend(1.0);
        let direct = direct.truncate().truncate() / direct.w;
        let px = (reprojected - direct) * size * 0.5;
        assert!(px.length() < 0.01, "{px}");
    }

    /// The sky, at reverse-Z depth 0, reprojects by rotation alone: an eye translation moves it
    /// nowhere.
    #[test]
    fn the_sky_moves_with_the_rotation_only() {
        let proj = reverse_z(16.0 / 9.0);
        let rot = Quat::from_rotation_y(0.3);
        let [cur_rel_from_clip, clip_from_cur_rel, prev_clip_from_cur_rel] = frame_matrices(
            Vec3::new(10.0, 0.0, 0.0),
            rot,
            proj,
            proj,
            (Vec3::ZERO, rot, proj),
        );
        let rel = cur_rel_from_clip * Vec4::new(0.25, -0.1, 0.0, 1.0);
        let cur = clip_from_cur_rel * rel;
        let prev = prev_clip_from_cur_rel * rel;
        let (cur, prev) = (cur.xy() / cur.w, prev.xy() / prev.w);
        assert!((cur - prev).length() < 1e-5, "{cur} vs {prev}");
    }

    /// Reverse-Z linearisation: `near / (d + ε)` is the eye distance for every depth in front,
    /// and a cleared 0 is large and finite.
    #[test]
    fn reverse_z_depth_linearises_to_eye_distance() {
        let proj = reverse_z(1.5);
        let near = proj.w_axis.z;
        for dist in [0.5_f32, 3.0, 40.0, 900.0] {
            let clip = proj * Vec4::new(0.0, 0.0, -dist, 1.0);
            let d = clip.z / clip.w;
            let linear = near / (d + f32::EPSILON);
            assert!((linear - dist).abs() / dist < 1e-3, "{dist}: {linear}");
        }
        assert!((near / f32::EPSILON).is_finite());
    }

    /// The parameter block is the size and alignment `common.wgsl` declares.
    #[test]
    fn the_params_block_matches_the_shader() {
        assert_eq!(PARAMS_SIZE, 3 * 64 + 8 * 8 + 12 * 4);
        assert_eq!(PARAMS_SIZE % 16, 0);
    }
}
