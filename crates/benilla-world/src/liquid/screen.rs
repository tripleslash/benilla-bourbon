//! **The cata liquid path's screen-space inputs.**
//!
//! `ProcWaterAbove` perm 4 — the permutation this client ports — reads five samplers, and three of
//! them are pictures of the frame so far:
//!
//! | reference stage | reference name | here |
//! |---|---|---|
//! | `samp0` | `"LiquidProc BackBufferTex"` | [`LiquidScreenTargets::color`] — refraction |
//! | `samp3` | `"blurredBack"` | [`LiquidScreenTargets::blur`] — the screen-space reflection |
//! | `samp6` | `DepthCache::GetDepthSR()` | [`LiquidScreenTargets::depth`] — linear view depth |
//!
//! (`samp2`, the 8×64 depth ramp, is evaluated analytically from the shared light instead of being
//! built as a texture; `samp5` is the bump map in [`super::art`]. Neither needs a pass.)
//!
//! **Why perm 4 and not perm 2.** Perm 2 binds a real reflection render target — a second scene
//! view rendered per frame, which the reference only fills at all because it has a whole
//! `CWorldScene::RenderReflection` path. Perm 4 is the shipped variant for `reflectionMode 0`: the
//! reflection is a screen-space ray traced against the *blurred back buffer*, with a constant
//! horizon colour where the ray leaves the screen or points at the sky. Same program family, same
//! author, one less render pass and no second camera — and it is what a Cataclysm player with
//! reflections off actually saw.
//!
//! ## Where this runs, and why there
//!
//! Between `Ssilvb` and `MainTransparentPass`, the slot `ssilvb` already established. By then the
//! main texture holds terrain, both model lanes, the retained static cells and the indirect-lighting
//! multiply — the whole opaque world — and nothing transparent has drawn yet. Water is the first
//! thing in the transparent pass, so this is the last moment the snapshot is both complete and
//! water-free. It is also why water does not refract other water, particles or ribbons: the
//! reference has exactly the same hole, for exactly the same reason.
//!
//! ## The resize discipline
//!
//! These are `Image` assets, so the water material can bind them through `AsBindGroup` like any
//! other texture. That makes a resize the one genuinely sharp edge: mutating an `Image` behind its
//! handle allocates a **new** `GpuImage`, and a material bind group built before that prepare still
//! points at the old one — `world_backdrop`'s header (decision 1647) is the full account. So a
//! resize here publishes fresh `AssetId`s and re-points every liquid material, which is cheap
//! because there are only ever a couple of dozen of them.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::FullscreenShader;
use bevy::ecs::query::QueryItem;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::asset::RenderAssetUsages;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_depth_2d, uniform_buffer,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderContext;
use bevy::render::texture::GpuImage;
use bevy::render::view::{
    Msaa, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms,
};
use bevy::render::{RenderApp, RenderStartup};

use crate::view::WorldCamera;

/// How much smaller the reflection source is than the frame. The reflection is a blurred,
/// bump-scattered ray hit — a quarter of the linear resolution is invisible in it, and the
/// reference blurs its own copy into a fixed 512² for the same reason.
const BLUR_SHIFT: u32 = 2;

/// The three pictures of the frame the cata water shader reads.
///
/// Handles rather than render-world `CachedTexture`s, because the consumer is a `Material`: it
/// binds them through `AsBindGroup`, which resolves `Handle<Image>` against
/// `RenderAssets<GpuImage>`. The node below renders *into* those same `GpuImage`s.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct LiquidScreenTargets {
    /// The opaque frame, viewport-sized. Refraction (`samp0`).
    pub(crate) color: Handle<Image>,
    /// A four-tap blur of the same, quarter-sized. The screen-space reflection (`samp3`).
    pub(crate) blur: Handle<Image>,
    /// Linear view depth in yards, `R32Float`, viewport-sized. The refraction guard and the
    /// depth tint (`samp6`).
    pub(crate) depth: Handle<Image>,
    /// The size `color`/`depth` were allocated at. A change here is what triggers the republish.
    pub(crate) size: UVec2,
}

impl LiquidScreenTargets {
    /// Allocate a fresh set at `size`, dropping nothing — the caller owns retiring the old ones.
    pub(crate) fn allocate(images: &mut Assets<Image>, size: UVec2) -> Self {
        let size = size.max(UVec2::ONE);
        let blur_size = (size >> UVec2::splat(BLUR_SHIFT)).max(UVec2::ONE);
        LiquidScreenTargets {
            color: images.add(target_image(
                "liquid_scene_color",
                size,
                ViewTarget::TEXTURE_FORMAT_HDR,
            )),
            blur: images.add(target_image(
                "liquid_scene_blur",
                blur_size,
                ViewTarget::TEXTURE_FORMAT_HDR,
            )),
            depth: images.add(target_image(
                "liquid_scene_depth",
                size,
                TextureFormat::R32Float,
            )),
            size,
        }
    }
}

/// One render target as an `Image`: no CPU data, no main-world copy, clamped and linear.
///
/// Clamp, not repeat, on every one of them. All three are addressed by screen UV, and the water
/// shader's refraction and reflection offsets both push that UV off the edge on purpose — under
/// `Repeat` a wave at the left margin would refract the right-hand side of the screen.
fn target_image(label: &'static str, size: UVec2, format: TextureFormat) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        label: Some(label.into()),
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        // R32Float is not filterable without a device feature, and the depth copy is compared
        // rather than blended — so it takes nearest and the colour copies take linear.
        mag_filter: filter_for(format),
        min_filter: filter_for(format),
        mipmap_filter: ImageFilterMode::Nearest,
        ..default()
    });
    image
}

fn filter_for(format: TextureFormat) -> ImageFilterMode {
    if format == TextureFormat::R32Float {
        ImageFilterMode::Nearest
    } else {
        ImageFilterMode::Linear
    }
}

/// Marks the ONE view these passes run for — the world camera.
///
/// Same reason `ssilvb` and `static_gx` carry markers: without it the node runs for every `Core3d`
/// view in the tree, including the portrait booth's bake cameras, and the last one to run would
/// leave *its* frame in the shared targets for the water to refract.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub(crate) struct LiquidScreenView;

fn mark_world_camera(
    mut commands: Commands,
    cam: Query<Entity, (With<WorldCamera>, Without<LiquidScreenView>)>,
) {
    for e in &cam {
        commands.entity(e).insert(LiquidScreenView);
    }
}

/// Opt the world camera's depth texture into `TEXTURE_BINDING` so the linearise pass can read it.
///
/// `Camera3d::depth_texture_usages` defaults to `RENDER_ATTACHMENT` alone. `ssilvb::arm_depth_binding`
/// does the same thing for the same reason, and the two compose: the flag is set once, guarded, and
/// never cleared, so whichever of them runs first pays for it and the other finds it already on.
fn arm_depth_binding(mut cams: Query<&mut Camera3d, With<WorldCamera>>) {
    for mut cam in &mut cams {
        let usage = TextureUsages::from(cam.depth_texture_usages);
        if usage.contains(TextureUsages::TEXTURE_BINDING) {
            continue; // guarded, so the common frame never flags `Camera3d` as changed
        }
        cam.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
    }
}

/// Republish the targets at the new size when the window resizes, and re-point the materials.
///
/// **Fresh `AssetId`s, not a `resize` through the old handle.** A mutated `Image` becomes a new
/// `GpuImage` at the next `prepare_assets`, and a material bind group prepared earlier in that same
/// frame keeps the old one — `world_backdrop`'s header records the whole failure. A new handle
/// cannot land in that state: `as_bind_group` either finds the new `GpuImage` or retries next
/// frame.
pub(crate) fn resize_targets(
    mut images: ResMut<Assets<Image>>,
    // OPTION, because `setup_liquid` returns early when there is no client data and never inserts
    // it. A plain `ResMut` would fail parameter validation on every frame of a dataless run, which
    // Bevy's default handler turns into a panic — `benilla-worldview` and every headless test boot
    // exactly that way.
    targets: Option<ResMut<LiquidScreenTargets>>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidCataMaterial>>,
    windows: Query<&Window>,
) {
    let (Some(mut targets), Some(size)) = (
        targets,
        windows.iter().next().map(|w| w.physical_size()),
    ) else {
        return;
    };
    if size.x == 0 || size.y == 0 || size == targets.size {
        return;
    }
    let retired = [
        targets.color.clone(),
        targets.blur.clone(),
        targets.depth.clone(),
    ];
    *targets = LiquidScreenTargets::allocate(&mut images, size);
    for (_, material) in materials.iter_mut() {
        material.extension.scene_color = targets.color.clone();
        material.extension.scene_blur = targets.blur.clone();
        material.extension.scene_depth = targets.depth.clone();
    }
    for handle in retired {
        images.remove(&handle);
    }
}

// ---------------------------------------------------------------- render world

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct LiquidScreenLabel;

#[derive(Resource)]
struct LiquidScreenPipelines {
    layout: BindGroupLayoutDescriptor,
    copy: CachedRenderPipelineId,
    blur: CachedRenderPipelineId,
    linear_depth: CachedRenderPipelineId,
}

fn init_pipelines(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/liquid_screen.wgsl");
    // One layout for all three passes. Two of them leave a binding unread, which is legal — the
    // pipeline layout only has to be a superset of what the entry point touches, and `ssilvb`
    // shares one layout across four entry points on the same reasoning.
    let layout = BindGroupLayoutDescriptor::new(
        "liquid_screen_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, uniform_buffer::<ViewUniform>(true)),
                (1, texture_depth_2d()),
                (2, texture_2d(TextureSampleType::Float { filterable: true })),
                (3, sampler(SamplerBindingType::Filtering)),
            ),
        ),
    );
    let pipeline = |label: &'static str, entry: &'static str, format: TextureFormat| {
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        }
    };
    commands.insert_resource(LiquidScreenPipelines {
        copy: pipeline_cache.queue_render_pipeline(pipeline(
            "liquid_screen_copy",
            "fs_copy",
            ViewTarget::TEXTURE_FORMAT_HDR,
        )),
        blur: pipeline_cache.queue_render_pipeline(pipeline(
            "liquid_screen_blur",
            "fs_blur",
            ViewTarget::TEXTURE_FORMAT_HDR,
        )),
        linear_depth: pipeline_cache.queue_render_pipeline(pipeline(
            "liquid_screen_depth",
            "fs_linear_depth",
            TextureFormat::R32Float,
        )),
        layout,
    });
}

#[derive(Default)]
struct LiquidScreenNode;

impl ViewNode for LiquidScreenNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static Msaa,
        &'static LiquidScreenView,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, depth, view_offset, msaa, _marker): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // Multisampling stands the whole thing down, and the water materials stand down with it
        // (`surface::cata_available`). A multisampled colour or depth texture is a different
        // binding type with no single value per pixel, and resolving both mid-graph to feed a
        // refraction is a cost nobody asked for. `ssilvb` makes the same trade.
        if *msaa != Msaa::Off {
            return Ok(());
        }
        let Some(targets) = world.get_resource::<LiquidScreenTargets>() else {
            return Ok(());
        };
        // The depth arming is a main-world write that REALLOCATES the texture, so the frame after
        // an enable still holds a render-attachment-only buffer. Binding it would be a wgpu
        // validation error, which kills the render thread rather than degrading. Skip; the next
        // frame is armed.
        if !depth
            .texture
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
        {
            return Ok(());
        }
        let pipelines = world.resource::<LiquidScreenPipelines>();
        let cache = world.resource::<PipelineCache>();
        let (Some(copy), Some(blur), Some(linear_depth)) = (
            cache.get_render_pipeline(pipelines.copy),
            cache.get_render_pipeline(pipelines.blur),
            cache.get_render_pipeline(pipelines.linear_depth),
        ) else {
            return Ok(()); // still compiling — the water falls back to its flat lane this frame
        };
        let Some(view_binding) = world.resource::<ViewUniforms>().uniforms.binding() else {
            return Ok(());
        };
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let (Some(color), Some(blurred), Some(depth_out)) = (
            gpu_images.get(&targets.color),
            gpu_images.get(&targets.blur),
            gpu_images.get(&targets.depth),
        ) else {
            return Ok(()); // the images have not been prepared yet
        };

        let device = render_context.render_device().clone();
        let layout = cache.get_bind_group_layout(&pipelines.layout);
        let bind = |src: &TextureView, samp: &Sampler| {
            device.create_bind_group(
                "liquid_screen_bind",
                &layout,
                &BindGroupEntries::with_indices((
                    (0, view_binding.clone()),
                    (1, depth.view()),
                    (2, src),
                    (3, samp),
                )),
            )
        };

        // colour ← the live main texture; blur ← that copy; depth ← the view's depth buffer.
        // The blur reads `color` rather than the main texture again purely so the two colour reads
        // agree bit for bit — a reflection sampling a different frame from the refraction beside it
        // is the kind of mismatch that only shows up in motion.
        let passes: [(&str, &RenderPipeline, BindGroup, &TextureView); 3] = [
            (
                "liquid_screen_copy",
                copy,
                bind(view_target.main_texture_view(), &color.sampler),
                &color.texture_view,
            ),
            (
                "liquid_screen_blur",
                blur,
                bind(&color.texture_view, &color.sampler),
                &blurred.texture_view,
            ),
            (
                "liquid_screen_depth",
                linear_depth,
                bind(&color.texture_view, &color.sampler),
                &depth_out.texture_view,
            ),
        ];
        for (label, pipeline, group, dst) in passes {
            let mut pass = render_context
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
            pass.set_bind_group(0, &group, &[view_offset.offset]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }
}

/// Wire the capture passes into the app. Called from [`super::LiquidPlugin`].
pub(crate) fn register(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<LiquidScreenTargets>::default())
        .add_plugins(ExtractComponentPlugin::<LiquidScreenView>::default())
        .add_systems(Update, (mark_world_camera, arm_depth_binding));

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(RenderStartup, init_pipelines)
        .add_render_graph_node::<ViewNodeRunner<LiquidScreenNode>>(Core3d, LiquidScreenLabel)
        // After everything opaque — including the retained static pass and the indirect-lighting
        // multiply — and before the transparent pass water itself draws in.
        .add_render_graph_edges(
            Core3d,
            (
                Node3d::MainOpaquePass,
                LiquidScreenLabel,
                Node3d::MainTransparentPass,
            ),
        )
        // After the retained cell pass: Stormwind's walls are opaque world and belong in the
        // snapshot. `StaticGxPlugin` is already built by the time `LiquidPlugin` is, so this edge
        // can be declared here; the `Ssilvb` one cannot — see `register_late`.
        .add_render_graph_edge(Core3d, crate::static_gx::StaticGxLabel, LiquidScreenLabel);
}

/// The one graph edge that cannot be declared in `build`.
///
/// `add_render_graph_edge` requires BOTH nodes to already exist, and `SsilvbPlugin` is added after
/// `LiquidPlugin` in `world_plugins` — so declaring this edge during `build` panics with
/// `InvalidNode(SsilvbLabel)` on every boot. `Plugin::finish` runs once every plugin has built,
/// which is exactly the phase an inter-plugin edge belongs in.
///
/// The edge itself is not optional bookkeeping. Without it the two passes are only *both* ordered
/// between the opaque and transparent passes, with nothing between them — so the scene copy could
/// be taken before the indirect-lighting multiply, and the riverbed seen THROUGH the water would
/// be lit differently from the riverbed seen beside it.
pub(crate) fn register_late(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.add_render_graph_edge(Core3d, crate::ssilvb::SsilvbLabel, LiquidScreenLabel);
}
