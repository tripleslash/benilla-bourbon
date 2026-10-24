//! fork: **the wet screen-space reflection** — rain-soaked ground and puddles reflect what is on
//! screen instead of a flat sky colour. No 1.12 counterpart: wet surfaces are the fork's own.
//!
//! Cheap by construction. It adds no target, no copy and no prepass: it reads the three pictures
//! the water already takes of the opaque frame (`liquid::screen`: the snapshot, its quarter-size
//! blur, and linear depth), and runs one full-screen draw that leaves every untagged pixel at the
//! first texture read.
//!
//! **The tag.** While [`EnvFrame::wet_ssr`] is set, terrain writes `1 + reflectance` into its
//! output alpha (`wet_hook::wet_reflectance`, fog applied) instead of adding `fog colour ×
//! reflectance` itself. Every other opaque write keeps alpha at or below 1, the main texture is
//! `Rgba16Float`, and the scene copy carries alpha through, so a snapshot texel above 1 is wet
//! ground and how much it reflects. Under MSAA the resolve averages the tag, which fades the
//! reflection across a silhouette rather than cutting it.
//!
//! **The pass.** After the snapshot and before the transparent pass, so the water still refracts
//! the ground without it. A short march along the view ray reflected about world up, against the
//! linear depth; a hit takes the snapshot there, a miss the sky colour. It adds into the view
//! target and puts the tagged pixel's alpha back to 1, so nothing past this pass sees a tag.
//!
//! **Off is free.** `rainSurfaces 0` zeroes the wetness row, so the flag never rises: terrain
//! never enters its wet branch, the pipeline is never queued, and the node returns before any
//! GPU work. Dry weather with the feature on is the same.

use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::{Core3d, Core3dSystems, FullscreenShader};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, uniform_buffer, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::lighting::{EnvFrame, LightingConsumeSet, WowLighting};
use crate::liquid::{LiquidScreenTargets, LiquidScreenView};

/// What the pass needs from the main world: whether it runs, and the colour a ray that finds
/// nothing reflects.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub(crate) struct WetReflections {
    /// Mirrors [`EnvFrame::wet_ssr`]: terrain tags exactly while this is set.
    live: bool,
    /// The sky colour terrain would have added, in the frame's own space (gamma in a classic
    /// frame, linear in a modern one); `w` unused.
    sky: [f32; 4],
}

/// The fog colour row 4 carries (`global_light`), in the space the frame holds.
fn publish(
    frame: Res<EnvFrame>,
    light: Res<WowLighting>,
    active: Option<Res<crate::modern::ActiveLighting>>,
    mut out: ResMut<WetReflections>,
) {
    let live = frame.wet_ssr > 0.5;
    if !live && !out.live {
        return; // dry, or `rainSurfaces` off: nothing to publish
    }
    let sky = if active.is_some_and(|a| a.is_modern()) {
        let tint = crate::modern::look::fog_tint();
        let c = |i: usize| crate::modern::srgb_to_linear(light.fog_color[i] * tint[i]);
        [c(0), c(1), c(2), 0.0]
    } else {
        let c = light.fog_color;
        [c[0], c[1], c[2], 0.0]
    };
    let next = WetReflections { live, sky };
    // Guarded, so the extract does not see a change every frame of dry weather.
    if out.live != next.live || (live && out.sky != next.sky) {
        *out = next;
    }
}

// ---------------------------------------------------------------- render world

/// The layout and one pipeline per MSAA sample count: the pass writes the view target, so its
/// multisample state has to match it. Its reads are the liquid snapshots, single-sampled always.
#[derive(Resource)]
struct WetSsrPipelines {
    shader: Handle<Shader>,
    layout: BindGroupLayoutDescriptor,
    variants: bevy::platform::collections::HashMap<u32, CachedRenderPipelineId>,
    /// `sky.rgb`, rewritten each frame the pass runs.
    params: Buffer,
}

fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "wet_ssr_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, uniform_buffer::<ViewUniform>(true)),
                (1, texture_2d(TextureSampleType::Float { filterable: true })),
                (2, texture_2d(TextureSampleType::Float { filterable: true })),
                // `R32Float`, read with `textureLoad` only.
                (
                    3,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (4, sampler(SamplerBindingType::Filtering)),
                (
                    5,
                    uniform_buffer_sized(false, Some(std::num::NonZero::new(16).unwrap())),
                ),
            ),
        ),
    );
    let params = render_device.create_buffer(&BufferDescriptor {
        label: Some("wet_ssr_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(WetSsrPipelines {
        shader: asset_server.load("embedded://benilla_world/shaders/wet_ssr.wgsl"),
        layout,
        variants: Default::default(),
        params,
    });
}

/// Queued on the first wet frame, never before: with `rainSurfaces` off (or in dry weather from
/// boot) the pass compiles nothing and draws nothing.
fn specialize_pipelines(
    mut pipelines: ResMut<WetSsrPipelines>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    wet: Option<Res<WetReflections>>,
    views: Query<&Msaa, With<LiquidScreenView>>,
) {
    if !wet.is_some_and(|w| w.live) {
        return;
    }
    for msaa in &views {
        let samples = msaa.samples();
        if pipelines.variants.contains_key(&samples) {
            continue;
        }
        // Colour adds; alpha is replaced by the shader's 1, which clears the tag. An untagged
        // pixel discards and keeps both.
        let blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Add,
            },
        };
        let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("wet_ssr".into()),
            layout: vec![pipelines.layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: pipelines.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fs_reflect".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(blend),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multisample: MultisampleState {
                count: samples,
                ..default()
            },
            ..default()
        });
        pipelines.variants.insert(samples, id);
    }
}

pub(crate) fn wet_ssr_pass(
    world: &World,
    view: ViewQuery<(
        &ViewTarget,
        &ViewUniformOffset,
        &Msaa,
        // The world camera only: the snapshots are its frame.
        &LiquidScreenView,
    )>,
    mut render_context: RenderContext,
) {
    let (view_target, view_offset, msaa, _marker) = view.into_inner();
    let Some(wet) = world.get_resource::<WetReflections>() else {
        return;
    };
    if !wet.live {
        return;
    }
    let Some(targets) = world.get_resource::<LiquidScreenTargets>() else {
        return;
    };
    let pipelines = world.resource::<WetSsrPipelines>();
    let Some(&id) = pipelines.variants.get(&msaa.samples()) else {
        return; // `specialize_pipelines` queues it this frame
    };
    let cache = world.resource::<PipelineCache>();
    let Some(pipeline) = cache.get_render_pipeline(id) else {
        return; // still compiling
    };
    let Some(view_binding) = world.resource::<ViewUniforms>().uniforms.binding() else {
        return;
    };
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let (Some(color), Some(blur), Some(depth)) = (
        gpu_images.get(&targets.color),
        gpu_images.get(&targets.blur),
        gpu_images.get(&targets.depth),
    ) else {
        return;
    };
    world.resource::<RenderQueue>().write_buffer(
        &pipelines.params,
        0,
        bytemuck::cast_slice(&wet.sky),
    );

    let device = render_context.render_device().clone();
    let bind = device.create_bind_group(
        "wet_ssr_bind",
        &cache.get_bind_group_layout(&pipelines.layout),
        &BindGroupEntries::with_indices((
            (0, view_binding),
            (1, &color.texture_view),
            (2, &blur.texture_view),
            (3, &depth.texture_view),
            (4, &color.sampler),
            (5, pipelines.params.as_entire_binding()),
        )),
    );
    // Into the view's own attachment, as `ssilvb`'s composite: the transparent pass behind finds
    // the frame it expects.
    let mut pass = render_context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("wet_ssr"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[view_offset.offset]);
    pass.draw(0..3, 0..1);
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<WetReflections>()
        .add_plugins(ExtractResourcePlugin::<WetReflections>::default())
        // After the resolve (this frame's fog) and the wetness tick (this frame's flag).
        .add_systems(
            Update,
            publish
                .in_set(LightingConsumeSet)
                .after(super::wetness::wetness_tick),
        );
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(
            Render,
            specialize_pipelines.in_set(RenderSystems::PrepareResources),
        )
        // After the snapshot it reads (which is itself after every opaque lane), before the water.
        .add_systems(
            Core3d,
            wet_ssr_pass
                .in_set(Core3dSystems::MainPass)
                .after(crate::liquid::liquid_screen_pass)
                .before(main_transparent_pass_3d),
        );
}
