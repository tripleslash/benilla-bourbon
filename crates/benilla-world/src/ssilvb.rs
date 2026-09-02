//! **SSILVB** — screen-space indirect lighting with a visibility bitmask.
//!
//! The world has never had indirect lighting. Every surface takes `ambient + sun·N·L` from the
//! authored `LightParams` rows, so a doorway, the underside of Goldshire's eaves and the base of a
//! tree are lit exactly like open ground — the one thing that reads as "flat" in a frame that is
//! otherwise faithful. This pass adds the two terms a screen-space gather can honestly recover:
//! **ambient occlusion** (how much of a fragment's hemisphere is shut) and **one bounce of indirect
//! colour** (what shut it, and in what colour). `shaders/ssilvb.wgsl` carries the method; this file
//! carries the wiring, and the two facts the wiring exists for.
//!
//! **Fact one: it reads the MAIN depth buffer, not a prepass.** Bevy's prepass cannot serve here
//! twice over. The custom WoW materials break when pulled into it (that is the pipeline-validation
//! failure `benilla-app`'s `player_shadow` module records from the first shadow experiment), and
//! the retained `static_gx` pass is a hand-rolled node that would never populate one at all — so
//! every Stormwind building would simply be absent from the occlusion. The main depth buffer has no
//! such gaps: terrain, the opaque and alpha-masked model lanes and the retained cell pass all write
//! into it. The cost of that choice is that view-space normals have to be reconstructed from depth,
//! which the shader does.
//!
//! **Fact two: it runs mid-graph, between the retained static pass and the transparent pass.** By
//! then the depth buffer holds the whole opaque world *including* the alpha-masked lane, which is
//! what puts Elwynn's tree canopies in the occluder set. Water, particles and ribbons draw
//! afterwards and are correctly left out of both terms — a campfire should not be dimmed by the
//! ground behind it, and its additive plume should not bounce green off the grass.
//!
//! Two player sliders drive it ([`IndirectLighting`], the CVars `ssaoStrength` / `ssgiStrength`).
//! Either at zero returns its half to an exact no-op; both at zero and this node never runs, which
//! is the state a visual A/B is taken against.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::FullscreenShader;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    texture_2d, texture_depth_2d, texture_depth_2d_multisampled, uniform_buffer,
    uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::shader::ShaderDefVal;
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::view::{
    Msaa, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms,
};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::view::WorldCamera;

/// **The two player knobs**, in one resource because they are two halves of one gather: the same
/// swept horizon produces both, and turning either to zero costs nothing extra.
///
/// Written by `benilla-app`'s CVar host (`ssaoStrength` / `ssgiStrength`); read by the render
/// node. Both are clamped to [`SSILVB_RANGE`] at every writer's edge.
#[derive(Resource, Clone, Copy, PartialEq, Debug, ExtractResource)]
pub struct IndirectLighting {
    /// Ambient-occlusion strength. `0` = no occlusion; `1` = the gather's own visibility applied
    /// whole.
    pub ao: f32,
    /// Indirect / colour-bleed strength. `0` = no colour bleeding.
    pub gi: f32,
}

/// The settable range of both strengths. The Graphics page offers `0–1`; the extra headroom above
/// it is for the hand-typed CVar and for looking at an exaggerated term while tuning — the same
/// narrower-row / wider-clamp split `renderScale` uses.
pub const SSILVB_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;

/// The shipped AO strength. Non-zero, unlike `worldShadows` and `renderScale`, on the director's
/// call: this is meant to be what the game looks like, not an opt-in experiment.
///
/// **It is the row's midpoint, and that is the point.** The shader carries a gain (`AO_GAIN` /
/// `GI_GAIN` in `shaders/ssilvb.wgsl`) sized so that 0.5 lands on the intended look with the top
/// half of the slider left as headroom. Turn either half up here and the slider stops being
/// centred on the shipped look; turn the strength up in the shader gain instead.
pub const AO_DEFAULT: f32 = 0.5;

/// The shipped indirect strength. See [`AO_DEFAULT`].
pub const GI_DEFAULT: f32 = 0.5;

impl Default for IndirectLighting {
    /// `$WOW_SSAO` / `$WOW_SSGI` override the shipped defaults, **session-only** — the A/B levers,
    /// the same posture as `$WOW_RENDER_SCALE` and `$WOW_MSAA`, and the only way to drive this pass
    /// from `benilla-worldview`, which has no CVar host at all.
    fn default() -> Self {
        Self {
            ao: env_strength("WOW_SSAO", AO_DEFAULT),
            gi: env_strength("WOW_SSGI", GI_DEFAULT),
        }
    }
}

impl IndirectLighting {
    /// Is there anything to compute? Both halves at zero and the whole chain — textures, four
    /// draws, the depth read — is skipped, leaving the frame bit-identical to a build without this
    /// module in it.
    pub fn enabled(self) -> bool {
        self.ao > 0.0 || self.gi > 0.0
    }
}

fn env_strength(key: &str, fallback: f32) -> f32 {
    match std::env::var(key).ok().and_then(|v| v.parse::<f32>().ok()) {
        Some(v) => v.clamp(*SSILVB_RANGE.start(), *SSILVB_RANGE.end()),
        None => fallback,
    }
}

/// The world-space sweep radius, in yards (`$WOW_SSILVB_RADIUS`). Four yards is about a doorway:
/// wide enough to darken the corner of a room and the base of a trunk, narrow enough that a hill
/// does not occlude the field in front of it. Not a player knob — a player asks "how much", not
/// "over what distance", and every value here trades noise against reach at fixed sample count.
fn sample_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| env_f32("WOW_SSILVB_RADIUS", 4.0, 0.25, 64.0))
}

/// The thickness a depth sample is assumed to have, in yards (`$WOW_SSILVB_THICKNESS`) — the
/// method's one real heuristic. A depth buffer says where a surface starts and never where it ends,
/// so the bitmask needs a back face to stop every occluder from being a wall running to infinity.
fn occluder_thickness() -> f32 {
    static T: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *T.get_or_init(|| env_f32("WOW_SSILVB_THICKNESS", 0.6, 0.01, 16.0))
}

fn env_f32(key: &str, fallback: f32, lo: f32, hi: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map_or(fallback, |v| v.clamp(lo, hi))
}

/// Marks the ONE view this pass runs on — the world camera.
///
/// Same reason `static_gx` carries its own marker: without it the node would run for every `Core3d`
/// view in the tree, including the portrait-booth bakes and the minimap, occluding a character's
/// face against the booth backdrop and costing four full-screen passes per pane.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub struct SsilvbView;

fn mark_world_camera(
    mut commands: Commands,
    cam: Query<Entity, (With<WorldCamera>, Without<SsilvbView>)>,
) {
    for e in &cam {
        commands.entity(e).insert(SsilvbView);
    }
}

/// Opt the world camera's depth texture into `TEXTURE_BINDING` so the gather can read it.
///
/// `Camera3d::depth_texture_usages` defaults to `RENDER_ATTACHMENT` alone — Bevy allocates a depth
/// buffer you cannot sample unless something asks — so this is the same move
/// `capture::depth_probe` makes for `COPY_SRC`, one flag over.
///
/// **Set once and never cleared.** Changing the usage reallocates the texture, and clearing it when
/// a slider crossed zero would rebuild the depth buffer on a drag tick. The flag costs nothing to
/// leave on: it can inhibit depth compression on some tilers, which is a price this pass is paying
/// anyway the moment it is enabled.
fn arm_depth_binding(
    lighting: Res<IndirectLighting>,
    mut cams: Query<&mut Camera3d, With<WorldCamera>>,
) {
    if !lighting.enabled() {
        return;
    }
    for mut cam in &mut cams {
        let usage = TextureUsages::from(cam.depth_texture_usages);
        if usage.contains(TextureUsages::TEXTURE_BINDING) {
            continue; // guarded, so the common frame never flags `Camera3d` as changed
        }
        cam.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
    }
}

// ---------------------------------------------------------------- render world

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct SsilvbLabel;

/// The two layouts and the four pipelines, **per MSAA sample count**.
///
/// **Why two layouts.** Every pass but the last reads (view, depth, source, params); the composite
/// reads only the gathered term and the params, and needs neither the view nor the depth. One WGSL
/// module cannot declare two different bindings at the same index, so the shader numbers its four
/// slots once and each layout claims the subset its entry point uses.
///
/// **Why keyed on sample count.** Two things change with it, and neither is expressible as a
/// runtime flag: the *binding type* of the view's depth (which has no resolve, unlike its colour),
/// and the composite's own `multisample` state — that pass writes into the view target, which is
/// multisampled when the rest of the frame is. `MsaaSetting` is latched at camera spawn, so in
/// practice one variant is built per session.
#[derive(Resource)]
struct SsilvbPipelines {
    shader: Handle<Shader>,
    variants: bevy::platform::collections::HashMap<u32, SsilvbVariant>,
}

struct SsilvbVariant {
    layout_pass: BindGroupLayoutDescriptor,
    layout_composite: BindGroupLayoutDescriptor,
    gather: CachedRenderPipelineId,
    blur_h: CachedRenderPipelineId,
    blur_v: CachedRenderPipelineId,
    composite: CachedRenderPipelineId,
}

fn init_pipelines(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(SsilvbPipelines {
        shader: asset_server.load("embedded://benilla_world/shaders/ssilvb.wgsl"),
        variants: Default::default(),
    });
}

/// Build the variant this frame's views need, once each. See [`SsilvbPipelines`] for why the sample
/// count cannot be a runtime flag, and `liquid::screen::specialize_pipelines` for the twin of this.
fn specialize_pipelines(
    mut pipelines: ResMut<SsilvbPipelines>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    lighting: Res<IndirectLighting>,
    views: Query<&Msaa, With<SsilvbView>>,
) {
    if !lighting.enabled() {
        return;
    }
    for msaa in &views {
        let samples = msaa.samples();
        if pipelines.variants.contains_key(&samples) {
            continue;
        }
        let variant =
            build_variant(&pipelines.shader, &fullscreen_shader, &pipeline_cache, samples);
        pipelines.variants.insert(samples, variant);
    }
}

fn build_variant(
    shader: &Handle<Shader>,
    fullscreen_shader: &FullscreenShader,
    pipeline_cache: &PipelineCache,
    samples: u32,
) -> SsilvbVariant {
    let ms = samples > 1;
    let params = || uniform_buffer_sized(false, Some(std::num::NonZero::new(16).unwrap()));
    let term = || texture_2d(TextureSampleType::Float { filterable: true });
    let layout_pass = BindGroupLayoutDescriptor::new(
        "ssilvb_pass_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, uniform_buffer::<ViewUniform>(true)),
                (
                    1,
                    if ms {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    },
                ),
                (2, term()),
                (3, params()),
            ),
        ),
    );
    let layout_composite = BindGroupLayoutDescriptor::new(
        "ssilvb_composite_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((2, term()), (3, params())),
        ),
    );
    let defs: Vec<ShaderDefVal> = if ms {
        vec!["SSILVB_MSAA".into()]
    } else {
        vec![]
    };
    let pipeline = |label: &'static str,
                    layout: &BindGroupLayoutDescriptor,
                    entry: &'static str,
                    blend: Option<BlendState>,
                    target_samples: u32|
     -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some(entry.into()),
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            // The gather and the two blurs write this pass's OWN single-sampled ping-pong, so they
            // stay at one sample whatever the frame is doing. Only the composite writes the view
            // target, and it has to match it.
            multisample: MultisampleState {
                count: target_samples,
                ..default()
            },
            ..default()
        }
    };
    // **The composite multiplies rather than writes** — `result = Dst·Src + Zero·Dst`. That is
    // what lets this pass sit mid-graph at all: the main texture stays the main texture, so
    // `MainTransparentPass` keeps drawing into the same attachment with the load flags the opaque
    // pass left on it. Flipping the view target's ping-pong here would instead hand that pass a
    // texture it had not used yet, and Bevy clears an attachment on first use — the whole opaque
    // frame, gone. Alpha is passed through untouched (`Zero·Src + One·Dst`).
    //
    // Under MSAA the multiply lands on every sample of the pixel, which is right: the occlusion is
    // a property of the pixel's neighbourhood, not of which triangle happened to cover which
    // sample, and darkening the samples unequally would show up as a fringe on every silhouette.
    let multiply = Some(BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::Dst,
            dst_factor: BlendFactor::Zero,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    });
    SsilvbVariant {
        gather: pipeline_cache.queue_render_pipeline(pipeline(
            "ssilvb_gather",
            &layout_pass,
            "fs_gather",
            None,
            1,
        )),
        blur_h: pipeline_cache.queue_render_pipeline(pipeline(
            "ssilvb_blur_h",
            &layout_pass,
            "fs_blur_h",
            None,
            1,
        )),
        blur_v: pipeline_cache.queue_render_pipeline(pipeline(
            "ssilvb_blur_v",
            &layout_pass,
            "fs_blur_v",
            None,
            1,
        )),
        composite: pipeline_cache.queue_render_pipeline(pipeline(
            "ssilvb_composite",
            &layout_composite,
            "fs_composite",
            multiply,
            samples,
        )),
        layout_pass,
        layout_composite,
    }
}

/// The gather target and its blur ping-pong, plus the 16-byte params uniform.
#[derive(Component)]
struct SsilvbTextures {
    /// The gather writes here, and the vertical blur writes back into it — so this is what the
    /// composite reads.
    a: CachedTexture,
    /// The horizontal blur's output, read by the vertical one.
    b: CachedTexture,
    /// `(ao, gi, radius, thickness)`, rewritten each frame with a queue write (queue writes land
    /// before the graph's submit executes).
    params: Buffer,
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    lighting: Res<IndirectLighting>,
    views: Query<(Entity, &ExtractedCamera, &Msaa, Option<&SsilvbTextures>), With<SsilvbView>>,
) {
    if !lighting.enabled() {
        return;
    }
    for (entity, camera, _msaa, existing) in &views {
        let Some(vp) = camera.physical_viewport_size else {
            continue;
        };
        // Full resolution: the term is read through a depth-aware blur that already refuses to
        // cross a silhouette, and a half-res gather would need a min/max depth pyramid of its own
        // to sample against. Revisit off a measurement, not off a hunch.
        let mut tex = |label: &'static str| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: vp.x.max(1),
                        height: vp.y.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let a = tex("ssilvb_gather_a");
        let b = tex("ssilvb_gather_b");
        // `TextureCache::get` must be called every frame to keep these alive, but it hands the same
        // textures back while the viewport holds — so only a genuine resize rebuilds the component.
        if existing.is_some_and(|t| t.a.texture.id() == a.texture.id() && t.b.texture.id() == b.texture.id())
        {
            continue;
        }
        let params = render_device.create_buffer(&BufferDescriptor {
            label: Some("ssilvb_params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        commands.entity(entity).insert(SsilvbTextures { a, b, params });
    }
}

#[derive(Default)]
struct SsilvbNode;

impl ViewNode for SsilvbNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static SsilvbTextures,
        &'static Msaa,
        // The world camera only — a booth bake must never be occluded against its own backdrop.
        &'static SsilvbView,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, depth, view_offset, textures, msaa, _marker): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let lighting = *world.resource::<IndirectLighting>();
        if !lighting.enabled() {
            return Ok(());
        }
        // **Multisampling no longer stands this pass down.** A multisampled depth and colour are
        // different binding types, not missing information: `build_variant` compiles a second set
        // of pipelines that read them per-sample, and the composite matches the view target's own
        // sample count so the multiply lands on every sample. Nothing is resolved to make it work,
        // and the single-sampled build is unchanged — it never sees the MSAA spelling at all.
        // The depth texture has to have been ARMED for sampling before it can be bound, and the
        // arming is a main-world write that reallocates it — so on the first frame after an enable
        // (and on the frame a re-created camera has not been marked yet) the buffer in hand is
        // still render-attachment-only. Binding it anyway is a wgpu validation error, which is not
        // a degrade: it kills the render thread. Skip the frame instead; the next one is armed.
        // (`benilla-app`'s `capture::depth_probe` guards its own readback the same way, and for
        // the same reason: its `COPY_SRC` only lands on the texture allocated after it asked.)
        if !depth
            .texture
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
        {
            return Ok(());
        }
        let pipelines = world.resource::<SsilvbPipelines>();
        let Some(variant) = pipelines.variants.get(&msaa.samples()) else {
            return Ok(()); // `specialize_pipelines` builds it this frame; gathered next
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let (Some(gather), Some(blur_h), Some(blur_v), Some(composite)) = (
            pipeline_cache.get_render_pipeline(variant.gather),
            pipeline_cache.get_render_pipeline(variant.blur_h),
            pipeline_cache.get_render_pipeline(variant.blur_v),
            pipeline_cache.get_render_pipeline(variant.composite),
        ) else {
            return Ok(()); // pipelines still compiling — draw the frame un-occluded
        };
        let Some(view_binding) = world.resource::<ViewUniforms>().uniforms.binding() else {
            return Ok(());
        };
        world.resource::<RenderQueue>().write_buffer(
            &textures.params,
            0,
            bytemuck::cast_slice(&[
                lighting.ao,
                lighting.gi,
                sample_radius(),
                occluder_thickness(),
            ]),
        );

        let render_device = render_context.render_device().clone();
        let layout_pass = pipeline_cache.get_bind_group_layout(&variant.layout_pass);
        let layout_composite = pipeline_cache.get_bind_group_layout(&variant.layout_composite);

        // Every bind group here is built per frame. The gather binds the view target's live main
        // texture, which flips whenever anything ahead of this node post-processes, and the rest
        // bind the depth view, which the arming above reallocates once — four `create_bind_group`
        // calls a frame is not worth the invalidation bookkeeping that would buy back.
        let pass_bind = |label: &'static str, src: &TextureView| {
            render_device.create_bind_group(
                label,
                &layout_pass,
                &BindGroupEntries::with_indices((
                    (0, view_binding.clone()),
                    (1, depth.view()),
                    (2, src),
                    (3, textures.params.as_entire_binding()),
                )),
            )
        };

        // gather → a, blur H → b, blur V → a. The vertical blur lands back in `a`, which is what
        // the composite reads.
        let chain: [(&str, &RenderPipeline, BindGroup, &TextureView); 3] = [
            (
                "ssilvb_gather",
                gather,
                // The opaque world as it stands: terrain, both model lanes, and the retained
                // cells. Read as a plain texture — Bevy resolved it at the end of the opaque pass,
                // and no pass has it bound as an attachment here.
                pass_bind("ssilvb_gather_bind", view_target.main_texture_view()),
                &textures.a.default_view,
            ),
            (
                "ssilvb_blur_h",
                blur_h,
                pass_bind("ssilvb_blur_h_bind", &textures.a.default_view),
                &textures.b.default_view,
            ),
            (
                "ssilvb_blur_v",
                blur_v,
                pass_bind("ssilvb_blur_v_bind", &textures.b.default_view),
                &textures.a.default_view,
            ),
        ];
        for (label, pipeline, bind, dst) in chain {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some(label),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: dst,
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations::default(),
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[view_offset.offset]);
            pass.draw(0..3, 0..1);
        }

        let bind = render_device.create_bind_group(
            "ssilvb_composite_bind",
            &layout_composite,
            &BindGroupEntries::with_indices((
                (2, &textures.a.default_view),
                (3, textures.params.as_entire_binding()),
            )),
        );
        // Straight back into the view's OWN attachment, multiplied in by the pipeline's blend.
        // `get_color_attachment` hands back the same texture the opaque pass drew into, already
        // past its clear-on-first-use — so the transparent pass behind us finds the frame it
        // expects, only darker.
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("ssilvb_composite"),
                color_attachments: &[Some(view_target.get_color_attachment())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        pass.set_pipeline(composite);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

pub struct SsilvbPlugin;

impl Plugin for SsilvbPlugin {
    fn build(&self, app: &mut App) {
        // The knob is initialised HERE rather than in `WorldFoundation` with the other player
        // settings: `MsaaSetting` and `ViewDistance` live there because the camera spawn and the
        // cull read them and no one plugin owns them, while this one is read by exactly this pass.
        app.init_resource::<IndirectLighting>()
            .add_plugins((
                ExtractResourcePlugin::<IndirectLighting>::default(),
                ExtractComponentPlugin::<SsilvbView>::default(),
            ))
            .add_systems(Update, (mark_world_camera, arm_depth_binding));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_pipelines)
            .add_systems(
                Render,
                (prepare_textures, specialize_pipelines).in_set(RenderSystems::PrepareResources),
            )
            .add_render_graph_node::<ViewNodeRunner<SsilvbNode>>(Core3d, SsilvbLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    SsilvbLabel,
                    Node3d::MainTransparentPass,
                ),
            )
            // …and after the retained cell pass, which draws between those two: without this edge
            // the gather could run before Stormwind's walls had written any depth at all.
            .add_render_graph_edge(Core3d, crate::static_gx::StaticGxLabel, SsilvbLabel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both halves at zero is the OFF state the whole chain checks for — the state a visual A/B is
    /// taken against, and the one a player who dislikes the look lands on. Either half alone still
    /// runs the gather, because the gather is what produces both.
    #[test]
    fn either_half_alone_still_runs_the_pass_and_both_zero_does_not() {
        assert!(!IndirectLighting { ao: 0.0, gi: 0.0 }.enabled());
        assert!(IndirectLighting { ao: 0.5, gi: 0.0 }.enabled());
        assert!(IndirectLighting { ao: 0.0, gi: 0.5 }.enabled());
        assert!(IndirectLighting { ao: 1.0, gi: 1.0 }.enabled());
    }

    /// The shipped defaults are non-zero — deliberately unlike `worldShadows` and `renderScale`,
    /// whose defaults are chosen so no frame moves. Pinned here so the CVar registration in
    /// `benilla-app` (which welds its registered strings to these) cannot drift from the engine.
    #[test]
    fn the_shipped_defaults_are_the_ones_the_cvars_register() {
        assert_eq!(AO_DEFAULT, 0.5);
        assert_eq!(GI_DEFAULT, 0.5);
        assert!(SSILVB_RANGE.contains(&AO_DEFAULT));
        assert!(SSILVB_RANGE.contains(&GI_DEFAULT));
        // The row offers 0–1; the clamp leaves headroom above it for a hand-typed CVar.
        assert_eq!(*SSILVB_RANGE.start(), 0.0);
        assert!(*SSILVB_RANGE.end() >= 1.0);
    }

    /// The env levers clamp, and an unparseable value is not an error that costs a run — it falls
    /// back to the shipped default, like every other `$WOW_*` knob in the engine.
    #[test]
    fn the_env_levers_clamp_and_fall_back() {
        // No such variable: the shipped default, untouched.
        assert_eq!(env_strength("WOW_SSILVB_NO_SUCH_VAR", 0.5), 0.5);
        assert_eq!(env_f32("WOW_SSILVB_NO_SUCH_VAR", 4.0, 0.25, 64.0), 4.0);
    }
}
