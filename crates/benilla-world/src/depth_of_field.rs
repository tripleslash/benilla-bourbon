//! fork: depth of field — a subtle far-field blur, after WoW Forever's (1.12 has none).
//!
//! The mechanism is the reference-era client's own three DoF pixel shaders as WoW Forever ships
//! them (build 1.60.1.70938, CASC fileIDs 3523455–3523457 for D3D11, 3523459–3523461 for D3D12,
//! cbuffers `DOF1Const` and `DepthOfField2`): a half-res downsample that keeps only the texels past
//! the focus distance, a 5×5 binomial blur weighted by that mask, and a composite that blends the
//! blur over the frame by `saturate(depth·scale + bias)·strength` with a partial desaturation.
//! `shaders/depth_of_field.wesl` carries the passes. The constants are the client's run-time
//! cbuffer values, which its shaders do not hold, so ours are chosen for a quiet effect: the
//! blur starts past the player and reaches its strength toward the horizon.
//!
//! The chain runs on the world camera only, at render resolution, between the main pass and the
//! upscaler, and draws its composite over the resolved main texture with an alpha blend, so the
//! main texture stays the main texture. Off (the `depthOfField` CVar, "0" by default) it runs
//! nothing and the frame is the frame a build without it renders.

use bevy::core_pipeline::{Core3d, Core3dSystems, FullscreenShader};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_depth_2d, texture_depth_2d_multisampled, uniform_buffer,
    uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::view::{
    Msaa, ViewDepthStencilTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms,
};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::ShaderDefVal;

use crate::view::WorldCamera;

/// The `depthOfField` CVar: the far-field blur on or off. Off by default, as 1.12 has none.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DepthOfField(pub bool);

impl Default for DepthOfField {
    /// `$WOW_DOF` (`1`/`0`) overrides the shipped off, session-only: the A/B lever, and the way
    /// `benilla-worldview`, which has no CVar host, turns it on.
    fn default() -> Self {
        Self(crate::launch_env::var("WOW_DOF").is_some_and(|v| v.trim() == "1"))
    }
}

/// Where the blur begins, in yards from the camera: past a fully zoomed-out player.
const START_YD: f32 = 45.0;
/// Where the blur reaches [`STRENGTH`], in yards.
const END_YD: f32 = 400.0;
/// The share of the blur in the farthest pixel: well short of whole, so distance softens
/// rather than smears.
const STRENGTH: f32 = 0.6;
/// The composite's desaturation of the blur (`DepthOfField2`'s third term), a touch of haze.
const DESATURATION: f32 = 0.1;
/// The render height the reference's texel offsets are taken at; a taller frame spreads the
/// kernel to cover the same share of the screen.
const REFERENCE_HEIGHT: f32 = 1080.0;

/// `(start, end, strength, desaturation)`, each overridable for tuning, session-only, as
/// `$WOW_DOF_START`, `$WOW_DOF_END`, `$WOW_DOF_STRENGTH` and `$WOW_DOF_DESATURATION`.
fn focus() -> [f32; 4] {
    static F: std::sync::OnceLock<[f32; 4]> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let start = env_f32("WOW_DOF_START", START_YD, 0.0, 5000.0);
        [
            start,
            env_f32("WOW_DOF_END", END_YD, 0.0, 10000.0).max(start + 1.0),
            env_f32("WOW_DOF_STRENGTH", STRENGTH, 0.0, 1.0),
            env_f32("WOW_DOF_DESATURATION", DESATURATION, 0.0, 1.0),
        ]
    })
}

fn env_f32(key: &str, fallback: f32, lo: f32, hi: f32) -> f32 {
    crate::launch_env::var(key)
        .and_then(|v| v.parse::<f32>().ok())
        .map_or(fallback, |v| v.clamp(lo, hi))
}

/// The kernel's spread at a render height: 1 at [`REFERENCE_HEIGHT`] and below, wider above.
fn spread(height: u32) -> f32 {
    (height as f32 / REFERENCE_HEIGHT).clamp(1.0, 3.0)
}

/// `DofParams` in the shader: two `vec4<f32>`.
const PARAMS_SIZE: u64 = 32;

/// Marks the one view the chain runs on, the world camera: a portrait bake or the minimap never
/// takes it.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
#[extract_app(bevy::render::RenderApp)]
pub struct DofView;

fn mark_world_camera(
    mut commands: Commands,
    cam: Query<Entity, (With<WorldCamera>, Without<DofView>)>,
) {
    for e in &cam {
        commands.entity(e).insert(DofView);
    }
}

/// Opt the world camera's depth into `TEXTURE_BINDING` once the blur is first wanted, as
/// `ssilvb::arm_depth_binding` does; set once and never cleared, since clearing reallocates it.
fn arm_depth_binding(dof: Res<DepthOfField>, mut cams: Query<&mut Camera3d, With<WorldCamera>>) {
    if !dof.0 {
        return;
    }
    for mut cam in &mut cams {
        let usage = TextureUsages::from(cam.depth_texture_usages);
        if usage.contains(TextureUsages::TEXTURE_BINDING) {
            continue;
        }
        cam.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
    }
}

// ---------------------------------------------------------------- render world

/// The pipelines per (MSAA sample count, main texture format): the depth binding's type follows
/// the first, the composite's target the second.
#[derive(Resource)]
struct DofPipelines {
    shader: Handle<Shader>,
    sampler: Sampler,
    variants: bevy::platform::collections::HashMap<(u32, TextureFormat), DofVariant>,
}

struct DofVariant {
    layout: BindGroupLayoutDescriptor,
    down: CachedRenderPipelineId,
    blur: CachedRenderPipelineId,
    composite: CachedRenderPipelineId,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    device: Res<RenderDevice>,
) {
    commands.insert_resource(DofPipelines {
        shader: asset_server.load("embedded://benilla_world/shaders/depth_of_field.wesl"),
        sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("dof_linear"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        }),
        variants: Default::default(),
    });
}

fn specialize_pipelines(
    mut pipelines: ResMut<DofPipelines>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    dof: Res<DepthOfField>,
    views: Query<(&Msaa, &ViewTarget), With<DofView>>,
) {
    if !dof.0 {
        return;
    }
    for (msaa, target) in &views {
        let key = (msaa.samples(), target.main_texture_format());
        if pipelines.variants.contains_key(&key) {
            continue;
        }
        let variant = build_variant(&pipelines.shader, &fullscreen_shader, &pipeline_cache, key);
        pipelines.variants.insert(key, variant);
    }
}

fn build_variant(
    shader: &Handle<Shader>,
    fullscreen_shader: &FullscreenShader,
    pipeline_cache: &PipelineCache,
    (samples, main_format): (u32, TextureFormat),
) -> DofVariant {
    let ms = samples > 1;
    let layout = BindGroupLayoutDescriptor::new(
        "dof_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                if ms {
                    texture_depth_2d_multisampled()
                } else {
                    texture_depth_2d()
                },
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer_sized(false, std::num::NonZero::new(PARAMS_SIZE)),
            ),
        ),
    );
    let defs: Vec<ShaderDefVal> = if ms { vec!["DOF_MSAA".into()] } else { vec![] };
    let pipeline = |label: &'static str,
                    entry: &'static str,
                    format: TextureFormat,
                    blend: Option<BlendState>| {
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                constants: Default::default(),
                shader: shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some(entry.into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        })
    };
    // The composite lays the blur over the frame by its `t` and leaves the frame's alpha alone.
    let over = Some(BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    });
    DofVariant {
        down: pipeline("dof_down", "fs_down", TextureFormat::Rgba16Float, None),
        blur: pipeline("dof_blur", "fs_blur", TextureFormat::Rgba16Float, None),
        composite: pipeline("dof_composite", "fs_composite", main_format, over),
        layout,
    }
}

/// The half-res mask target and its blur, plus the params uniform.
#[derive(Component)]
pub(crate) struct DofTextures {
    half_a: CachedTexture,
    half_b: CachedTexture,
    params: Buffer,
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    dof: Res<DepthOfField>,
    views: Query<(Entity, &ExtractedCamera, Option<&DofTextures>), With<DofView>>,
) {
    if !dof.0 {
        return;
    }
    for (entity, camera, existing) in &views {
        let Some(vp) = camera.physical_viewport_size else {
            continue;
        };
        let mut tex = |label: &'static str| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: vp.x.div_ceil(2).max(1),
                        height: vp.y.div_ceil(2).max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let half_a = tex("dof_half_a");
        let half_b = tex("dof_half_b");
        if existing.is_some_and(|t| {
            t.half_a.texture.id() == half_a.texture.id()
                && t.half_b.texture.id() == half_b.texture.id()
        }) {
            continue;
        }
        let params = render_device.create_buffer(&BufferDescriptor {
            label: Some("dof_params"),
            size: PARAMS_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        commands.entity(entity).insert(DofTextures {
            half_a,
            half_b,
            params,
        });
    }
}

/// `EarlyPostProcess`, before the upscaler: the three passes on the world view.
pub(crate) fn dof_pass(
    world: &World,
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ViewUniformOffset,
        &DofTextures,
        &Msaa,
        &DofView,
    )>,
    mut render_context: RenderContext,
) {
    let (view_target, depth, view_offset, textures, msaa, _marker) = view.into_inner();
    if !world.resource::<DepthOfField>().0 {
        return;
    }
    // The arming reallocates the depth a frame after the enable; binding the old one is a
    // validation error, so that frame draws sharp.
    if !depth
        .texture()
        .usage()
        .contains(TextureUsages::TEXTURE_BINDING)
    {
        return;
    }
    let pipelines = world.resource::<DofPipelines>();
    let Some(variant) = pipelines
        .variants
        .get(&(msaa.samples(), view_target.main_texture_format()))
    else {
        return; // `specialize_pipelines` builds it this frame
    };
    let pipeline_cache = world.resource::<PipelineCache>();
    let (Some(down), Some(blur), Some(composite)) = (
        pipeline_cache.get_render_pipeline(variant.down),
        pipeline_cache.get_render_pipeline(variant.blur),
        pipeline_cache.get_render_pipeline(variant.composite),
    ) else {
        return; // still compiling: the frame draws sharp
    };
    let Some(view_binding) = world.resource::<ViewUniforms>().uniforms.binding() else {
        return;
    };
    let [start, end, strength, desaturation] = focus();
    let height = depth.texture().size().height;
    world.resource::<RenderQueue>().write_buffer(
        &textures.params,
        0,
        bytemuck::cast_slice(&[
            start,
            end,
            strength,
            desaturation,
            spread(height),
            0.0,
            0.0,
            0.0,
        ]),
    );

    let render_device = render_context.render_device().clone();
    let layout = pipeline_cache.get_bind_group_layout(&variant.layout);
    let bind = |label: &'static str, src: &TextureView| {
        render_device.create_bind_group(
            label,
            &layout,
            &BindGroupEntries::sequential((
                view_binding.clone(),
                depth.attachment.depth_stencil_views().attachment_view(),
                src,
                &pipelines.sampler,
                textures.params.as_entire_binding(),
            )),
        )
    };

    // scene → half a (masked), half a → half b (blur), half b → the scene (blended over).
    let unsampled = view_target.get_unsampled_color_attachment();
    let chain: [(&str, &RenderPipeline, BindGroup, RenderPassColorAttachment); 3] = [
        (
            "dof_down",
            down,
            bind("dof_down_bind", view_target.main_texture_view()),
            plain_attachment(&textures.half_a.default_view),
        ),
        (
            "dof_blur",
            blur,
            bind("dof_blur_bind", &textures.half_a.default_view),
            plain_attachment(&textures.half_b.default_view),
        ),
        (
            "dof_composite",
            composite,
            bind("dof_composite_bind", &textures.half_b.default_view),
            unsampled,
        ),
    ];
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    for (label, pipeline, bind, attachment) in chain {
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        let span = diagnostics.pass_span(&mut pass, label);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[view_offset.offset]);
        pass.draw(0..3, 0..1);
        span.end(&mut pass);
    }
}

fn plain_attachment(view: &TextureView) -> RenderPassColorAttachment<'_> {
    RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: Operations::default(),
    }
}

pub struct DepthOfFieldPlugin;

impl Plugin for DepthOfFieldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DepthOfField>()
            .add_plugins((
                ExtractResourcePlugin::<DepthOfField>::default(),
                ExtractComponentPlugin::<DofView>::default(),
            ))
            .add_systems(
                Update,
                (
                    mark_world_camera,
                    // After the frame's script calls, whose `depthOfField` callback writes the knob.
                    arm_depth_binding.after(crate::schedule::WorldStage::Present),
                ),
            );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_pipelines)
            .add_systems(
                Render,
                (prepare_textures, specialize_pipelines).in_set(RenderSystems::PrepareResources),
            )
            // After every main-pass draw; the upscaler orders itself after it (`fsr::FsrPlugin`).
            .add_systems(Core3d, dof_pass.in_set(Core3dSystems::EarlyPostProcess));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off out of the box: 1.12 has no depth of field.
    #[test]
    fn the_blur_ships_off() {
        if crate::launch_env::var("WOW_DOF").is_none() {
            assert!(!DepthOfField::default().0);
        }
    }

    /// The shipped focus is a quiet one: it starts past the player, ramps over hundreds of yards
    /// and never takes a pixel wholly.
    #[test]
    fn the_shipped_focus_is_subtle() {
        const { assert!(START_YD >= 30.0) };
        const { assert!(END_YD - START_YD >= 200.0) };
        const { assert!(STRENGTH > 0.0 && STRENGTH < 1.0) };
        const { assert!(DESATURATION < 0.25) };
    }

    /// The kernel keeps the reference's texel offsets up to 1080 rows and widens above them.
    #[test]
    fn the_spread_follows_the_render_height() {
        assert_eq!(spread(720), 1.0);
        assert_eq!(spread(1080), 1.0);
        assert_eq!(spread(2160), 2.0);
        assert_eq!(spread(100_000), 3.0);
    }
}
