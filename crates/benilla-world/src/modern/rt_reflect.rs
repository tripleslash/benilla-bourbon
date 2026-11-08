//! fork: G — **ray-traced water reflections with a Hi-Z screen-space march in front** (native only).
//!
//! Where [`rt_reflections`] holds this frame, the cata water's planar mirror
//! (`liquid::reflect`'s second scene render, about 4 ms of render-thread CPU at a waterside town)
//! stands down, and the water takes the reflection it would have read from the mirror along the same
//! ray instead: first a march through this frame's full-depth Hi-Z pyramid (the screen-space
//! reflection), then, for a ray that leaves the screen or passes behind what the screen shows, one
//! ray query against the ray-traced scene (`static_gx::rt_scene`), its hit shaded as the probe trace
//! shades one (`benilla_modern::rtreflect`). A ray that finds nothing reads the sky the water
//! already falls back to.
//!
//! The lane replaces the mirror only where the mirror would have run: the `waterDetail` tiers with
//! a planar reflection (2, the reference's, and 3, the experimental), above the water. The SSR tier
//! (1) keeps the reference's perm-4 trace, a single projected point 128 yd along the ray; at tier 2
//! that point is still laid over the base the lane now supplies, as the reference lays it over the
//! mirror.
//!
//! Where the session may trace reflections ([`rt_reflections_capable`]), the water declares the
//! scene in the one group it already declares for the ray-traced shadows
//! (`benilla_assets::materials::RT_REFLECT_LAYOUT`, `push_rt_water`), and bevy_pbr's
//! `MaterialExtraBindGroup` binds the scene while either lane is live, else the stand-in
//! (`modern::rt_shadow`). The per-frame choice is the shader's: a lane row in the modern block
//! (`light_spare[12]`, `x` shadows, `y` reflections, written in the render world with the frame's
//! decisions) and the water's own flag (`LiquidParams.anim.x`, written with the mirror's).

use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::pbr::MaterialExtraBindGroup;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{texture_2d, texture_storage_2d};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::liquid::{LiquidScreenTargets, LiquidScreenView};
use crate::static_gx::rt_scene;

/// Texels per pyramid workgroup side (`gx_hiz.wgsl`).
const HIZ_TILE: u32 = 8;

/// **The water's reflection this frame**, decided in the main world with the mirror camera's
/// (`liquid::reflect`): `on` while the lane stands in for the mirror. Extracted every frame; the
/// render world builds the pyramid on it, so the march has its levels whenever the water asks.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, Debug, ExtractResource)]
pub(crate) struct RtReflectLane {
    pub(crate) on: bool,
}

impl RtReflectLane {
    /// The water's `LiquidParams.anim.x`: 0 the mirror's path, 1 the march then a ray, 2 a ray
    /// alone (`WOW_SSR=0`).
    pub(crate) fn shader_flag(self) -> f32 {
        match (self.on, ssr_enabled()) {
            (false, _) => 0.0,
            (true, true) => 1.0,
            (true, false) => 2.0,
        }
    }
}

/// Main world, `Update`, after the frame's script calls (the `rayTracing` and `waterDetail`
/// callbacks) and the submersion verdict, before `liquid::reflect`'s mirror sync, which stands the
/// mirror down while [`RtReflectLane::on`] holds and writes it to the water with the mirror's own
/// flag. On: the lane is live ([`rt_reflections`]), the option is on and the frame modern (both
/// read here, this frame), the tier has a mirror to replace and the eye is above the water.
///
/// **The one-frame lag, and the case taken.** `rt_reflections()` here is the render world's latest
/// decision, a frame behind the frame this one extracts into. Going live, the mirror stays on for
/// that frame (the lane is not yet decided here), so the switch never shows a frame without a
/// reflection. Falling back, the option (`RayTracingSetting`) and the lighting model are read here,
/// this frame, so turning ray tracing off at run time takes the lane down in the frame the render
/// world frees the scene, whenever the option's callback ran before this system (the scripts of
/// `WorldStage::Present`). Two cases reach the render world first: the callback ran later in the
/// frame (a click the UI handles after this), or the render world found the scene not ready (a
/// teleport, a map change). For that one frame the water keeps the lane over the stand-in: the
/// march still runs (its pyramid follows this decision) and the rays it would have handed on read
/// the sky, since the stand-in's lane row is clear. The mirror is back the next frame; neither path
/// keeps history, so nothing is cut.
pub(crate) fn decide_water_lane(
    detail: Res<crate::liquid::WaterDetail>,
    underwater: Res<crate::liquid::Underwater>,
    setting: Option<Res<super::RayTracingSetting>>,
    active: Option<Res<super::ActiveLighting>>,
    frames: Option<Res<bevy::diagnostic::FrameCount>>,
    mut lane: ResMut<RtReflectLane>,
    mut logged: Local<Option<bool>>,
) {
    let option = setting.is_none_or(|s| s.0);
    let modern = active.is_some_and(|a| a.is_modern());
    let traced = rt_reflections() && option && modern;
    let wanted = traced && detail.has_planar_reflection();
    let on = wanted && !underwater.0.any();
    if lane.on != on {
        lane.on = on;
    }
    // One line a switch of the lane itself; a dive's mirror stands down on its own.
    if *logged != Some(wanted) && (logged.is_some() || wanted) {
        let frame = frames.map_or(0, |f| f.0);
        if wanted {
            info!(
                "rt reflect: switch at frame {frame}: on; the water reflects by {}, the mirror \
                 camera is off",
                if ssr_enabled() {
                    "the Hi-Z march, then a ray"
                } else {
                    "a ray alone (WOW_SSR=0)"
                }
            );
        } else {
            let why = if !rt_reflections() {
                "ray tracing is not live"
            } else if !option {
                "the rayTracing option is off"
            } else if !modern {
                "classic lighting"
            } else {
                "the water tier has no mirror"
            };
            info!(
                "rt reflect: switch at frame {frame}: off ({why}); the water reflects by the mirror \
                 camera where its tier has one"
            );
        }
    }
    *logged = Some(wanted);
}

/// **The ray-traced reflections' lane, this frame**: the ray tracing is live (`rt_scene::live()`:
/// capable, the `rayTracing` option on and the scene ready) and the session may trace reflections
/// ([`rt_reflections_capable`]). Per frame, any world (outside the render world, its latest
/// decision); false before render startup.
pub(crate) fn rt_reflections() -> bool {
    rt_scene::live() && rt_reflections_capable()
}

/// **May this session trace reflections at all?** Capable of ray tracing (`rt_scene::capable()`,
/// fixed at render startup) and `$WOW_RT_REFLECT=0` did not turn the lane off (read once). Where it
/// holds the water compiles `RT_REFLECT` and declares the scene for the whole session.
pub(crate) fn rt_reflections_capable() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    rt_scene::capable()
        && *CELL.get_or_init(|| crate::launch_env::var("WOW_RT_REFLECT").as_deref() != Some("0"))
}

/// `$WOW_SSR=0` (read once): the lane keeps the mirror off but skips the screen-space march, so every
/// reflected ray goes to the TLAS. The A/B lever for the march's share of the water's cost.
pub(crate) fn ssr_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_SSR").as_deref() != Some("0"))
}

/// `RenderStartup`, after the tier is decided: where the session may trace reflections, publish the
/// scene's layout to the water and give bevy_pbr's draw command the resource it binds.
fn init_rt_reflect(mut commands: Commands, extra: Option<Res<MaterialExtraBindGroup>>) {
    if !rt_reflections_capable() {
        return;
    }
    // A second startup (a test app, a renderer rebuilt) finds it set to the same layout.
    let _ = benilla_assets::materials::RT_REFLECT_LAYOUT.set(rt_scene::layout());
    // One resource for both lanes: the shadows' startup may have inserted it already.
    if extra.is_none() {
        commands.insert_resource(MaterialExtraBindGroup::default());
    }
    info!(
        "rt reflect: the water declares the scene; while ray tracing is live it reflects by a Hi-Z \
         screen-space march, then by ray, with the mirror camera off (WOW_RT_REFLECT=0 keeps the \
         mirror, WOW_SSR=0 skips the march)"
    );
}

/// **The march's Hi-Z pyramid** — the water's own linear depth copy (`LiquidScreenTargets::depth`,
/// bound at `liquid_cata.wgsl`'s 108 as `scene_depth`) given a full mip chain, each level the
/// nearest depth over its footprint in the level below.
///
/// - **Level 0** is the copy itself: the world camera's planar view depth in yards, `R32Float`,
///   viewport-sized, the sky at 1e9, written by `liquid_screen_pass` after the opaque pass (terrain,
///   the entities and static_gx all in), MSAA resolved there by taking sample 0. One depth read for
///   the refraction and the march alike, in the binding the water already has.
/// - **Levels 1..** (`modern_reflect_hiz`, one dispatch a level of `gx_hiz.wgsl`'s `downsample`
///   under `HIZ_LINEAR`) take the minimum: on linear depth that is the NEAREST surface over a
///   conservative footprint (no texel skipped). A march skips a cell only while the ray stays in
///   front of everything in it, so the nearest is what keeps it from stepping over a hit. The
///   farthest (C's occlusion pyramid) is conservative the other way: a cell is skipped past a
///   pillar standing in front of distant ground. Linear depth rather than reverse-Z makes the
///   march's thickness test a difference in yards.
/// - Valid from this pass until the next frame's, in a frame [`RtReflectLane::on`] holds; the view
///   is the world camera's (`LiquidScreenView`). A level `L` texel covers uv
///   `[x / size_L, (x + 1) / size_L)` of the screen at least.
#[derive(Resource)]
struct ReflectHiZ {
    layout: BindGroupLayoutDescriptor,
    down: CachedComputePipelineId,
    /// The depth copy's view the levels were made for, levels 1.. sizes, and level `i + 1`'s bind
    /// group reading level `i`.
    levels: Option<(TextureViewId, Vec<UVec2>, Vec<BindGroup>)>,
}

/// `RenderStartup`: queue the pyramid's reduction where the lane may march.
fn init_reflect_hiz(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    if !(rt_reflections_capable() && ssr_enabled()) {
        return;
    }
    let layout = BindGroupLayoutDescriptor::new(
        "modern_reflect_hiz_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let down = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("modern_reflect_hiz".into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        shader: asset_server.load("embedded://benilla_world/shaders/modern/gx_hiz.wgsl"),
        shader_defs: vec!["HIZ_LINEAR".into()],
        entry_point: Some("downsample".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(ReflectHiZ {
        layout,
        down,
        levels: None,
    });
}

/// `PrepareBindGroups`: one bind group a level over the depth copy, remade when the copy is (a
/// resize republishes it). Kept as they are in a frame the lane is off.
fn prepare_reflect_hiz(
    mut hiz: ResMut<ReflectHiZ>,
    lane: Option<Res<RtReflectLane>>,
    targets: Option<Res<LiquidScreenTargets>>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    if !lane.is_some_and(|l| l.on) {
        return;
    }
    let Some(depth) = targets.as_deref().and_then(|t| images.get(&t.depth)) else {
        hiz.levels = None;
        return;
    };
    let count = depth.texture_descriptor.mip_level_count;
    if count < 2 {
        hiz.levels = None;
        return;
    }
    let id = depth.texture_view.id();
    if hiz.levels.as_ref().is_some_and(|(made, ..)| *made == id) {
        return;
    }
    let layout = pipeline_cache.get_bind_group_layout(&hiz.layout);
    let views: Vec<TextureView> = (0..count)
        .map(|level| {
            depth.texture.create_view(&TextureViewDescriptor {
                label: Some("modern_reflect_hiz_level"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..default()
            })
        })
        .collect();
    let size0 = UVec2::new(
        depth.texture_descriptor.size.width,
        depth.texture_descriptor.size.height,
    );
    let sizes = (1..count)
        .map(|level| (size0 >> level).max(UVec2::ONE))
        .collect();
    let binds = views
        .windows(2)
        .map(|pair| {
            device.create_bind_group(
                "modern_reflect_hiz",
                &layout,
                &BindGroupEntries::sequential((&pair[0], &pair[1])),
            )
        })
        .collect();
    hiz.levels = Some((id, sizes, binds));
}

/// `modern_reflect_hiz`: the pyramid's levels above the depth copy, after `liquid_screen_pass`
/// writes it and before the transparent pass, where the water marches it. World camera only.
pub(crate) fn reflect_hiz_pass(
    world: &World,
    view: ViewQuery<&LiquidScreenView>,
    mut render_context: RenderContext,
) {
    let _world_camera = view.into_inner();
    if !world
        .get_resource::<RtReflectLane>()
        .is_some_and(|l| l.on)
    {
        return;
    }
    let Some(hiz) = world.get_resource::<ReflectHiZ>() else {
        return;
    };
    let Some((_, sizes, binds)) = hiz.levels.as_ref() else {
        return;
    };
    let Some(down) = world
        .resource::<PipelineCache>()
        .get_compute_pipeline(hiz.down)
    else {
        return;
    };
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = render_context.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("modern_reflect_hiz"),
        timestamp_writes: None,
    });
    let span = diagnostics.pass_span(&mut pass, "modern_reflect_hiz");
    pass.set_pipeline(down);
    for (size, bind) in sizes.iter().zip(binds) {
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(size.x.div_ceil(HIZ_TILE), size.y.div_ceil(HIZ_TILE), 1);
    }
    span.end(&mut pass);
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<RtReflectLane>()
        .add_plugins(ExtractResourcePlugin::<RtReflectLane>::default());
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .add_systems(
                RenderStartup,
                (
                    init_rt_reflect
                        .after(rt_scene::init_rt_scene)
                        .after(super::rt_shadow::init_rt_shadows),
                    init_reflect_hiz.after(rt_scene::init_rt_scene),
                ),
            )
            .add_systems(
                Render,
                prepare_reflect_hiz
                    .in_set(RenderSystems::PrepareBindGroups)
                    .run_if(resource_exists::<ReflectHiZ>),
            )
            .add_systems(
                Core3d,
                reflect_hiz_pass
                    .in_set(Core3dSystems::MainPass)
                    .after(crate::liquid::liquid_screen_pass)
                    .before(main_transparent_pass_3d),
            );
    }
}
