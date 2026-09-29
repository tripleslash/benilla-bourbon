//! **The cata liquid path's screen-space inputs.**
//!
//! `ProcWaterAbove` perm 0 — the permutation this client ports — reads six samplers, and four of
//! them are pictures of the frame:
//!
//! | reference stage | reference name | here |
//! |---|---|---|
//! | `samp0` | `"LiquidProc BackBufferTex"` | [`LiquidScreenTargets::color`] — refraction |
//! | `samp1` | `GetFakeReflectionBuffer()` | [`LiquidScreenTargets::reflect`] — the planar mirror |
//! | `samp3` | `"blurredBack"` | [`LiquidScreenTargets::blur`] — the screen-space reflection |
//! | `samp6` | `DepthCache::GetDepthSR()` | [`LiquidScreenTargets::depth`] — linear view depth |
//!
//! (`samp2`, the 8×64 depth ramp, is evaluated analytically from the shared light instead of being
//! built as a texture; `samp5` is the bump map in [`super::art`]. Neither needs a pass.)
//!
//! Three of the four are filled by the passes in THIS file, which are blits. The fourth is a scene
//! render and belongs to a camera, not to a node — [`super::reflect`] owns it; this file only
//! allocates its target, because the resize discipline below is the same for all four and having
//! one of them keep its own copy of it is how the four drift apart.
//!
//! ## Multisampling
//!
//! These passes run under MSAA and cost nothing extra to do it. They were already copying the frame
//! into single-sampled targets, and a copy that reads N samples and writes one value *is* a
//! resolve — so the resolve is the copy, in the pass that was happening anyway, with no second
//! target and no `resolve_target` chain. The single-sampled build is untouched: it keeps the plain
//! binding and the plain filtered fetch, with no per-sample loop to skip. `liquid_screen.wgsl`
//! carries which half resolves by averaging (colour) and which by taking sample 0 (depth), and why.
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

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems, FullscreenShader};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_depth_2d, texture_depth_2d_multisampled, uniform_buffer,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::view::{
    Msaa, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms,
};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::ShaderDefVal;

use crate::view::WorldCamera;

/// How much smaller the reflection source is than the frame. The reflection is a blurred,
/// bump-scattered ray hit — a quarter of the linear resolution is invisible in it, and the
/// reference blurs its own copy into a fixed 512² for the same reason.
const BLUR_SHIFT: u32 = 2;

/// How much smaller the **planar reflection** render is than the frame — the reference's
/// `reflectionDownscale` CVar, whose `CreateReflectionBuffer` sizes the target
/// `viewport / (1 << downscale)`.
///
/// The CVar registers at `"0"` (full frame) and `SetReflectionDownscale` accepts up to 4. One is
/// taken here rather than zero because this buffer is never read sharp: every fetch of it is a
/// projected read 20 yards along a bent ray, and the term it feeds is then lerped against a
/// quarter-resolution blur. A second full-resolution scene render to be read that softly is the one
/// cost in this feature with nothing to show for it.
const REFLECT_SHIFT: u32 = 1;

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
    /// The **planar mirror** — a second, mirrored render of the world about the world plane
    /// `z = 0`, at [`REFLECT_SHIFT`]. `ProcWaterAbove` perm 0's `samp1`, and the base its
    /// screen-space trace is laid over. Filled by a camera (`liquid::reflect`), not by a node —
    /// it is a scene render, not a blit.
    pub(crate) reflect: Handle<Image>,
    /// The size `color`/`depth` were allocated at. A change here is what triggers the republish.
    pub(crate) size: UVec2,
}

impl LiquidScreenTargets {
    /// Allocate a fresh set at `size`, dropping nothing — the caller owns retiring the old ones.
    pub(crate) fn allocate(images: &mut Assets<Image>, size: UVec2) -> Self {
        let size = size.max(UVec2::ONE);
        let blur_size = (size >> UVec2::splat(BLUR_SHIFT)).max(UVec2::ONE);
        let reflect_size = (size >> UVec2::splat(REFLECT_SHIFT)).max(UVec2::ONE);
        LiquidScreenTargets {
            color: images.add(target_image(
                "liquid_scene_color",
                size,
                TextureFormat::Rgba16Float,
            )),
            blur: images.add(target_image(
                "liquid_scene_blur",
                blur_size,
                TextureFormat::Rgba16Float,
            )),
            depth: images.add(target_image(
                "liquid_scene_depth",
                size,
                TextureFormat::R32Float,
            )),
            reflect: images.add(target_image(
                "liquid_reflection",
                reflect_size,
                TextureFormat::Rgba16Float,
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
    let (Some(mut targets), Some(size)) =
        (targets, windows.iter().next().map(|w| w.physical_size()))
    else {
        return;
    };
    if size.x == 0 || size.y == 0 || size == targets.size {
        return;
    }
    let retired = [
        targets.color.clone(),
        targets.blur.clone(),
        targets.depth.clone(),
        targets.reflect.clone(),
    ];
    *targets = LiquidScreenTargets::allocate(&mut images, size);
    for (_, material) in materials.iter_mut() {
        material.extension.scene_color = targets.color.clone();
        material.extension.scene_blur = targets.blur.clone();
        material.extension.scene_depth = targets.depth.clone();
        material.extension.reflect_map = targets.reflect.clone();
    }
    for handle in retired {
        images.remove(&handle);
    }
}

// ---------------------------------------------------------------- render world

/// The three pipelines, per MSAA sample count.
///
/// Multisampling changes the *binding type* of the view's colour and depth, not just a flag, so the
/// layout and all three pipelines are per-count. `MsaaSetting` is latched at camera spawn and never
/// moves mid-session, so in practice exactly one variant is ever built — but keying it means a
/// second view at a different count (a booth pane, a future split target) cannot quietly bind a
/// multisampled texture into a single-sampled slot, which is a validation error that kills the
/// render thread rather than degrading.
#[derive(Resource)]
struct LiquidScreenPipelines {
    shader: Handle<Shader>,
    variants: bevy::platform::collections::HashMap<u32, ScreenVariant>,
}

struct ScreenVariant {
    layout: BindGroupLayoutDescriptor,
    copy: CachedRenderPipelineId,
    blur: CachedRenderPipelineId,
    linear_depth: CachedRenderPipelineId,
}

fn init_pipelines(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(LiquidScreenPipelines {
        shader: asset_server.load("embedded://benilla_world/shaders/liquid_screen.wgsl"),
        variants: Default::default(),
    });
}

/// Build the variant this frame's views actually need, once each.
///
/// A `Render`-schedule system rather than `RenderStartup`, because the sample count is not known
/// until a view exists — the same reason `static_gx` specialises its four pipelines against a
/// `(samples, format)` key instead of queueing them at startup.
fn specialize_pipelines(
    mut pipelines: ResMut<LiquidScreenPipelines>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<&Msaa, With<LiquidScreenView>>,
) {
    for msaa in &views {
        let samples = msaa.samples();
        if pipelines.variants.contains_key(&samples) {
            continue;
        }
        let variant = build_variant(
            &pipelines.shader,
            &fullscreen_shader,
            &pipeline_cache,
            samples,
        );
        pipelines.variants.insert(samples, variant);
    }
}

fn build_variant(
    shader: &Handle<Shader>,
    fullscreen_shader: &FullscreenShader,
    pipeline_cache: &PipelineCache,
    samples: u32,
) -> ScreenVariant {
    let ms = samples > 1;
    // Only binding 1 varies: the view's DEPTH is multisampled under MSAA and has no resolve.
    // Binding 2 is the view's colour, which Bevy already resolved at the end of the opaque pass, and
    // binding 4 is the copy this file makes — both single-sampled on every path.
    let layout = BindGroupLayoutDescriptor::new(
        "liquid_screen_layout",
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
                (2, texture_2d(TextureSampleType::Float { filterable: true })),
                (3, sampler(SamplerBindingType::Filtering)),
                (4, texture_2d(TextureSampleType::Float { filterable: true })),
            ),
        ),
    );
    let defs: Vec<ShaderDefVal> = if ms {
        vec!["LIQUID_SCREEN_MSAA".into()]
    } else {
        vec![]
    };
    let pipeline = |label: &'static str, entry: &'static str, format: TextureFormat| {
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some(entry.into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            // The TARGETS are single-sampled in every build — that is the whole point: these
            // passes write resolved images, so the resolve is this copy and there is no second one.
            ..default()
        }
    };
    ScreenVariant {
        copy: pipeline_cache.queue_render_pipeline(pipeline(
            "liquid_screen_copy",
            "fs_copy",
            TextureFormat::Rgba16Float,
        )),
        blur: pipeline_cache.queue_render_pipeline(pipeline(
            "liquid_screen_blur",
            "fs_blur",
            TextureFormat::Rgba16Float,
        )),
        linear_depth: pipeline_cache.queue_render_pipeline(pipeline(
            "liquid_screen_depth",
            "fs_linear_depth",
            TextureFormat::R32Float,
        )),
        layout,
    }
}

pub(crate) fn liquid_screen_pass(
    world: &World,
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthTexture,
        &ViewUniformOffset,
        &Msaa,
        &LiquidScreenView,
    )>,
    mut render_context: RenderContext,
) {
    let (view_target, depth, view_offset, msaa, _marker) = view.into_inner();
    {
        let Some(targets) = world.get_resource::<LiquidScreenTargets>() else {
            return;
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
            return;
        }
        // The variant for THIS view's sample count. Multisampling does not stand these passes
        // down; it changes one binding type and nothing else (see `liquid_screen.wgsl`).
        let pipelines = world.resource::<LiquidScreenPipelines>();
        let Some(variant) = pipelines.variants.get(&msaa.samples()) else {
            return; // `specialize_pipelines` builds it this frame; drawn next
        };
        let cache = world.resource::<PipelineCache>();
        let (Some(copy), Some(blur), Some(linear_depth)) = (
            cache.get_render_pipeline(variant.copy),
            cache.get_render_pipeline(variant.blur),
            cache.get_render_pipeline(variant.linear_depth),
        ) else {
            return; // still compiling — the water falls back to its flat lane this frame
        };
        let Some(view_binding) = world.resource::<ViewUniforms>().uniforms.binding() else {
            return;
        };
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let (Some(color), Some(blurred), Some(depth_out)) = (
            gpu_images.get(&targets.color),
            gpu_images.get(&targets.blur),
            gpu_images.get(&targets.depth),
        ) else {
            return; // the images have not been prepared yet
        };

        let device = render_context.render_device().clone();
        let layout = cache.get_bind_group_layout(&variant.layout);
        // Bindings 0-3 are the same for every pass: the view uniform, the view's depth (1) and its
        // resolved colour (2), and a sampler. Binding 4 is the blur's source — and it has to VARY,
        // because a texture cannot be a bound resource and a colour target in the same pass. The
        // copy writes `color`, so its group must not also read `color`; the blur reads `color` and
        // writes `blurred`, so its group must not read `blurred`. Two groups, swapped.
        let bind = |label: &'static str, blur_src: &TextureView| {
            device.create_bind_group(
                label,
                &layout,
                &BindGroupEntries::with_indices((
                    (0, view_binding.clone()),
                    (1, depth.view()),
                    (2, view_target.main_texture_view()),
                    (3, &color.sampler),
                    (4, blur_src),
                )),
            )
        };
        // Whatever is bound at 4 for these two is unread by their entry points — `blurred` is
        // simply the texture neither of them writes.
        let bind_write_color = bind("liquid_screen_bind_write", &blurred.texture_view);
        let bind_blur = bind("liquid_screen_bind_blur", &color.texture_view);

        // colour ← the live main texture; blur ← that copy; depth ← the view's depth buffer.
        // The blur reads `color` rather than the main texture again purely so the two colour reads
        // agree bit for bit — a reflection sampling a different frame from the refraction beside it
        // is the kind of mismatch that only shows up in motion.
        let passes: [(&str, &RenderPipeline, &BindGroup, &TextureView); 3] = [
            (
                "liquid_screen_copy",
                copy,
                &bind_write_color,
                &color.texture_view,
            ),
            (
                "liquid_screen_blur",
                blur,
                &bind_blur,
                &blurred.texture_view,
            ),
            (
                "liquid_screen_depth",
                linear_depth,
                &bind_write_color,
                &depth_out.texture_view,
            ),
        ];
        for (label, pipeline, group, dst) in passes {
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
                        multiview_mask: None,
                    });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[view_offset.offset]);
            pass.draw(0..3, 0..1);
        }
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
        .add_systems(
            Render,
            specialize_pipelines.in_set(RenderSystems::PrepareResources),
        )
        // After everything opaque — the retained static pass and the indirect-lighting multiply
        // included, so the riverbed seen THROUGH the water is lit like the riverbed beside it — and
        // before the transparent pass water itself draws in.
        .add_systems(
            Core3d,
            liquid_screen_pass
                .in_set(Core3dSystems::MainPass)
                .after(main_opaque_pass_3d)
                .after(crate::static_gx::static_gx_pass)
                .after(crate::ssilvb::ssilvb_pass)
                .before(main_transparent_pass_3d),
        );
}
