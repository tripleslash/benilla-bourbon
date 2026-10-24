//! The player-UI gamma lane's one decode. [`ui_quad.wgsl`](shaders/ui_quad.wgsl) composites the
//! UI in gamma bytes, as the reference's fixed-function device does into its 8-bit backbuffer:
//! blends are gamma arithmetic clamped at each write, and `alphaMode="ADD"` is `dst + texel·α`
//! (EGxBlend 3, `glBlendFunc(GL_SRC_ALPHA, GL_ONE)`; factor tables `0x85c1f8`/`0x85c224`). This
//! node decodes the finished image to linear once, straight into the swapchain (output mode
//! `Skip`, no blit), whose sRGB write re-encodes the client's byte. Without it the UI presents
//! about 2.2 times too bright. [`UiGammaLane`] gates it off every other `Camera2d` on `Core2d`.
//!
//! Bevy UI (the glue and loading screens) converts in its own shaders ([`use_gamma_ui_shaders`])
//! and lands on this camera, which [`crate::ui_pass`] makes `IsDefaultUiCamera`.

use bevy::core_pipeline::upscaling::upscaling;
use bevy::core_pipeline::{Core2d, FullscreenShader};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, uniform_buffer_sized};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::ui_render::ui_pass;
use bevy::ui_render::ui_texture_slice_pipeline::{
    init_ui_texture_slice_pipeline, UiTextureSlicePipeline,
};
use bevy::ui_render::{init_ui_pipeline, UiPipeline};

use benilla_world::final_pass::FinalPassTarget;

/// Marks the camera whose target holds the gamma-composited UI, with the [`DisplayGamma`]
/// exponent and the [`DisplayBrightness`] lift the decode applies.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub(crate) struct UiGammaLane {
    /// The clamped `gamma` CVar; `1.0` is the identity ramp.
    pub(crate) gamma: f32,
    /// The signed level shift of [`DisplayBrightness::lift`]; `0.0` changes no byte.
    pub(crate) lift: f32,
}

impl Default for UiGammaLane {
    fn default() -> Self {
        Self {
            gamma: DEFAULT_GAMMA,
            lift: 0.0,
        }
    }
}

/// The `gamma` CVar (registered at `0x402d70`, name `0x82e924`, default `"1.0"` at `0x82e92c`,
/// flags 0). The reference's callback `0x4034d0` hands `SetDeviceGammaRamp` the ramp
/// `__ftol(pow(i / 255, gamma) · 65535)` (`0x591680`); there is no other whole-frame grade and
/// no `Brightness` or `Contrast` CVar.
///
/// Deviation: the reference skips that upload when windowed (`byte[dev+0x20b]` is `gxWindow`), and
/// every benilla mode is windowed, so the same curve applies in the compositor, where the slider
/// can move pixels: [`UiGammaNode`] raises the gamma byte the RAMDAC would have read to `gamma`
/// before its decode, the continuous form of the 256-entry ramp.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub(crate) struct DisplayGamma(pub(crate) f32);

/// The registered `"1.0"`: the identity ramp, at which the pass changes no byte.
pub(crate) const DEFAULT_GAMMA: f32 = 1.0;

/// Deviation: the reference never clamps (`SetGamma` `0x4891f0` writes `-4.000000` for 5), but its
/// ramp is a fullscreen OS call and ours is the image, where an extreme exponent blacks out the
/// panel that would undo it. The range spans the stock slider's `[0.5, 1.5]` four times over; the
/// CVar keeps the value written and only its consumers clamp.
pub(crate) const GAMMA_RANGE: std::ops::RangeInclusive<f32> = 0.25..=4.0;

impl Default for DisplayGamma {
    fn default() -> Self {
        Self(DEFAULT_GAMMA)
    }
}

/// benilla's `brightness` CVar, 0 to 1. The reference has no such CVar: its one whole-frame
/// grade is the `gamma` ramp ([`DisplayGamma`]), which bends the midtones and leaves black and
/// white where they are. This is the level control beside it, applied after the ramp: above the
/// default it lifts the blacks toward [`BRIGHTNESS_SPAN`] with white held, below it dims the
/// whites by up to that much with black held. The default is the identity.
pub(crate) const CVAR_BRIGHTNESS: &str = "brightness";

/// The registered `"0.5"`, the slider's centre, at which the pass changes no byte.
pub(crate) const DEFAULT_BRIGHTNESS: f32 = 0.5;

/// How far either end of the slider moves the image: black up to this byte share at 1, white down
/// by it at 0.
pub(crate) const BRIGHTNESS_SPAN: f32 = 0.25;

/// The clamped `brightness` CVar.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub(crate) struct DisplayBrightness(pub(crate) f32);

impl Default for DisplayBrightness {
    fn default() -> Self {
        Self(DEFAULT_BRIGHTNESS)
    }
}

impl DisplayBrightness {
    /// The signed shift the decode applies: `0` at the default, `±BRIGHTNESS_SPAN` at the ends.
    pub(crate) fn lift(self) -> f32 {
        (self.0.clamp(0.0, 1.0) - DEFAULT_BRIGHTNESS) * 2.0 * BRIGHTNESS_SPAN
    }
}

/// Copies the clamped gamma and the brightness lift onto the lane camera when either changes.
fn stamp_lane_gamma(
    gamma: Res<DisplayGamma>,
    brightness: Res<DisplayBrightness>,
    mut lanes: Query<&mut UiGammaLane>,
) {
    if !gamma.is_changed() && !brightness.is_changed() {
        return;
    }
    for mut lane in &mut lanes {
        lane.gamma = gamma.0.clamp(*GAMMA_RANGE.start(), *GAMMA_RANGE.end());
        lane.lift = brightness.lift();
    }
}

/// The decode's shared parts; the pipeline is specialised per view ([`ViewUiGammaPipeline`]).
#[derive(Resource)]
struct UiGammaPipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
    /// The exponent and lift's 16-byte uniform, queue-written each frame, which lands before the graph's
    /// submit. One buffer, not one per view: [`UiGammaLane`] is on exactly one camera.
    ramp: Buffer,
}

impl SpecializedRenderPipeline for UiGammaPipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("ui_gamma_decode".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fs_decode".into()),
                // An 8-bit sRGB target either way (the swapchain, or a `Write` camera's non-`Hdr`
                // main texture), so the lane clamps at every blend like the reference.
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        }
    }
}

/// The decode pipeline for one view, specialised on where the decode lands
/// ([`benilla_world::final_pass`]); stamped every frame, as Bevy stamps `ViewUpscalingPipeline`.
#[derive(Component)]
struct ViewUiGammaPipeline(CachedRenderPipelineId);

fn prepare_view_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<UiGammaPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiGammaPipeline>>,
    views: Query<(Entity, &ExtractedCamera, &ViewTarget), With<UiGammaLane>>,
) {
    for (entity, camera, target) in &views {
        let Some(format) = FinalPassTarget::format(&camera.output_mode, target) else {
            continue;
        };
        let id = pipelines.specialize(&pipeline_cache, &pipeline, format);
        commands.entity(entity).insert(ViewUiGammaPipeline(id));
    }
}

fn init_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "ui_gamma_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                // The exponent and lift as a `vec4<f32>`: a uniform is 16-byte aligned, so no
                // padding field.
                uniform_buffer_sized(false, Some(std::num::NonZero::new(16).unwrap())),
            ),
        ),
    );
    // Point sampling: this is a 1:1 full-screen resolve, never a resample.
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        min_filter: FilterMode::Nearest,
        mag_filter: FilterMode::Nearest,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..Default::default()
    });
    let ramp = render_device.create_buffer(&BufferDescriptor {
        label: Some("ui_gamma_ramp"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(UiGammaPipeline {
        layout,
        sampler,
        shader: asset_server.load("embedded://benilla_app/shaders/ui_gamma.wgsl"),
        fullscreen: fullscreen_shader.clone(),
        ramp,
    });
}

/// The decode: the view's finished UI target, gamma-decoded once into its output.
fn ui_gamma_decode(
    view: ViewQuery<(
        &ViewTarget,
        &UiGammaLane,
        &ExtractedCamera,
        &ViewUiGammaPipeline,
    )>,
    pipelines: Res<UiGammaPipeline>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
    let (view_target, lane, camera, pipeline) = view.into_inner();
    let Some(decode) = pipeline_cache.get_render_pipeline(pipeline.0) else {
        // Still compiling: the UI stays undecoded, over-bright, for a frame or two.
        return;
    };
    // This frame's exponent and lift as `[g, lift, 0, 0]`; the shader reads `.x` and `.y`.
    queue.write_buffer(
        &pipelines.ramp,
        0,
        bytemuck::cast_slice(&[lane.gamma, lane.lift, 0.0, 0.0]),
    );
    // The swapchain for the `Skip` player-UI camera, the ping-pong target for a `Write` one.
    let Some(out) = FinalPassTarget::resolve(camera, view_target) else {
        return; // no output surface this frame (an occluded swapchain)
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipelines.layout);
    let bind = ctx.render_device().create_bind_group(
        "ui_gamma_decode",
        &layout,
        &BindGroupEntries::sequential((
            out.source,
            &pipelines.sampler,
            pipelines.ramp.as_entire_binding(),
        )),
    );
    // The perf journal buckets this span's name into `gpu_ui`.
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let scissor = out.scissor_rect();
    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("ui_gamma_decode"),
            color_attachments: &[Some(out.destination)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    if let Some((x, y, w, h)) = scissor {
        pass.set_scissor_rect(x, y, w, h);
    }
    let span = diagnostics.pass_span(&mut pass, "ui_gamma_decode");
    pass.set_pipeline(decode);
    pass.set_bind_group(0, &bind, &[]);
    pass.draw(0..3, 0..1);
    span.end(&mut pass);
}

/// Puts Bevy UI on the gamma lane: the glue and loading screens are Bevy UI trees, not quads, and
/// would otherwise composite in linear. Bevy UI has no colour-space hook, so this swaps the
/// `shader` handle [`UiPipeline`] and [`UiTextureSlicePipeline`] clone into every specialisation
/// for vendored gamma-emitting copies, once, after their init systems.
fn use_gamma_ui_shaders(
    asset_server: Res<AssetServer>,
    mut node: ResMut<UiPipeline>,
    mut slice: ResMut<UiTextureSlicePipeline>,
) {
    node.shader = asset_server.load("embedded://benilla_app/shaders/ui_node_gamma.wgsl");
    slice.shader = asset_server.load("embedded://benilla_app/shaders/ui_slice_gamma.wgsl");
}

pub(crate) struct UiGammaPlugin;

/// The `gamma` CVar's change callback, clamped to [`GAMMA_RANGE`], and `brightness`'s, clamped to
/// its slider's 0 to 1.
pub(crate) fn on_cvar(
    ev: On<crate::cvars::CvarChanged>,
    mut gamma: ResMut<DisplayGamma>,
    mut brightness: ResMut<DisplayBrightness>,
) {
    if ev.is(benilla_ui::script::CVAR_GAMMA) {
        gamma.0 = ev.num().clamp(*GAMMA_RANGE.start(), *GAMMA_RANGE.end());
    } else if ev.is(CVAR_BRIGHTNESS) {
        brightness.0 = ev.num().clamp(0.0, 1.0);
    }
}

impl Plugin for UiGammaPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(on_cvar);
        app.init_resource::<DisplayGamma>()
            .init_resource::<DisplayBrightness>()
            .add_plugins(ExtractComponentPlugin::<UiGammaLane>::default())
            .add_systems(Update, stamp_lane_gamma);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SpecializedRenderPipelines<UiGammaPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare_view_pipelines.in_set(RenderSystems::Prepare),
            )
            .add_systems(
                RenderStartup,
                use_gamma_ui_shaders
                    .after(init_ui_pipeline)
                    .after(init_ui_texture_slice_pipeline),
            )
            // After the quads and after Bevy UI, which writes gamma into the same target, and
            // before the output blit, which a `Skip` camera skips: the decode is the output write.
            .add_systems(
                Core2d,
                ui_gamma_decode
                    .after(benilla_world::ffx_glow::ffx_transparent_pass_2d)
                    .after(ui_pass)
                    .before(upscaling),
            );
    }
}
