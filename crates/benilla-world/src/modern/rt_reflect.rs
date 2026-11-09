//! fork: G — **ray-traced water reflections** (native only), the `waterDetail` 4 tier
//! ("Experimental (RT)", [`crate::liquid::WATER_DETAIL_RT`]).
//!
//! Where [`rt_reflections`] holds this frame, the cata water's planar mirror
//! (`liquid::reflect`'s second scene render, about 4 ms of render-thread CPU at a waterside town)
//! stands down, and the water takes the reflection it would have read from the mirror along the same
//! ray instead: one ray query against the ray-traced scene (`static_gx::rt_scene`), its hit lit as
//! the mirror's own render lights that surface (`liquid_cata.wgsl`'s `lane_shade`, over
//! `benilla_modern::rtreflect`) and fogged as the mirror fogs it. A ray that finds nothing reads the
//! sky the water already falls back to.
//!
//! **The tier asks, nothing else.** The lane runs at the RT tier alone, whatever the ray-tracing
//! options say: the tier asks for the scene ([`RtWaterTier`], extracted; `rt_scene` builds it while
//! an option's lane or this asks), and with only the water asking, the shadows, the probe GI and
//! the froxels stay on raster (`rt_scene::gi_live` and its siblings). At every other tier the
//! water is what that tier always was, the options on or off. Where the session cannot trace reflections (no ray
//! queries, `WOW_RT=0`, `WOW_RT_REFLECT=0`, the web) the RT tier renders as the experimental tier,
//! mirror and all, with one log line.
//!
//! Where the session may trace reflections ([`rt_reflections_capable`]), the water declares the
//! scene in the one group it already declares for the ray-traced shadows
//! (`benilla_assets::materials::RT_REFLECT_LAYOUT`, `push_rt_water`), and bevy_pbr's
//! `MaterialExtraBindGroup` binds the scene while either lane is live, else the stand-in
//! (`modern::rt_shadow`). The per-frame choice is the shader's: a lane row in the modern block
//! (`light_spare[12]`, `x` shadows, `y` reflections, written in the render world with the frame's
//! decisions) and the water's own flag (`LiquidParams.anim.x`, written with the mirror's).

use bevy::pbr::MaterialExtraBindGroup;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::{RenderApp, RenderStartup};

use crate::static_gx::rt_scene;

/// **The water's reflection this frame**, decided in the main world with the mirror camera's
/// (`liquid::reflect`): `on` while the lane stands in for the mirror. Extracted every frame.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, Debug, ExtractResource)]
pub(crate) struct RtReflectLane {
    pub(crate) on: bool,
}

impl RtReflectLane {
    /// The water's `LiquidParams.anim.x`: 0 the mirror's path, 1 the ray.
    pub(crate) fn shader_flag(self) -> f32 {
        if self.on {
            1.0
        } else {
            0.0
        }
    }
}

/// **Does the water's tier ask for the ray-traced scene?** `waterDetail` 4
/// ([`crate::liquid::WaterDetail::wants_ray_traced_reflection`]), written by [`decide_water_lane`]
/// and extracted, where `rt_scene::update_rt_tier` reads it beside the ray-tracing options.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, Debug, ExtractResource)]
pub(crate) struct RtWaterTier(pub(crate) bool);

/// Main world, `Update`, after the frame's script calls (the `waterDetail` callback) and the
/// submersion verdict, before `liquid::reflect`'s mirror sync, which stands the mirror down while
/// [`RtReflectLane::on`] holds and writes it to the water with the mirror's own flag. Asks for the
/// scene at the RT tier ([`RtWaterTier`]). On: the lane is live ([`rt_reflections`]), the tier is
/// the RT one and the frame modern (both read here, this frame), and the eye is above the water.
///
/// **The one-frame lag, and the case taken.** `rt_reflections()` here is the render world's latest
/// decision, a frame behind the frame this one extracts into. Going live, the mirror stays on for
/// that frame (the lane is not yet decided here), so the switch never shows a frame without a
/// reflection. Falling back, the tier and the lighting model are read here, this frame, so leaving
/// the RT tier at run time brings the mirror back the same frame, whenever the tier's callback ran
/// before this system (the scripts of `WorldStage::Present`). Two cases reach the render world
/// first: the callback ran later in the frame (a click the UI handles after this), or the render
/// world found the scene not ready (a teleport, a map change). For that one frame the water keeps
/// the lane over the stand-in, whose rays read the sky, since the stand-in's lane row is clear. The
/// mirror is back the next frame; neither path keeps history, so nothing is cut.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decide_water_lane(
    detail: Res<crate::liquid::WaterDetail>,
    underwater: Res<crate::liquid::Underwater>,
    active: Option<Res<super::ActiveLighting>>,
    frames: Option<Res<bevy::diagnostic::FrameCount>>,
    mut ask: ResMut<RtWaterTier>,
    mut lane: ResMut<RtReflectLane>,
    mut logged: Local<Option<bool>>,
    mut fallback_logged: Local<bool>,
) {
    let tier = detail.wants_ray_traced_reflection();
    if ask.0 != tier {
        ask.0 = tier;
    }
    // Where the session cannot trace reflections, the RT tier is the experimental tier: one line
    // once render startup has decided the capability.
    if tier && rt_scene::decided() && !rt_reflections_capable() {
        if !*fallback_logged {
            *fallback_logged = true;
            info!(
                "rt reflect: waterDetail 4 (Experimental (RT)) needs ray tracing, which this session \
                 cannot do (no ray queries, WOW_RT=0 or WOW_RT_REFLECT=0); the water renders as the \
                 experimental tier, by the planar mirror"
            );
        }
    } else if !tier {
        *fallback_logged = false;
    }
    let modern = active.is_some_and(|a| a.is_modern());
    let wanted = tier && rt_reflections() && modern;
    let on = wanted && !underwater.0.any();
    if lane.on != on {
        lane.on = on;
    }
    // One line a switch of the lane itself; a dive's mirror stands down on its own.
    if *logged != Some(wanted) && (logged.is_some() || wanted) {
        let frame = frames.map_or(0, |f| f.0);
        if wanted {
            info!(
                "rt reflect: switch at frame {frame}: on; the water reflects by ray, the mirror \
                 camera is off"
            );
        } else {
            let why = if !tier {
                "the water tier is not the RT one"
            } else if !modern {
                "classic lighting"
            } else {
                "the ray-traced scene is not live"
            };
            info!(
                "rt reflect: switch at frame {frame}: off ({why}); the water reflects by the mirror \
                 camera where its tier has one"
            );
        }
    }
    *logged = Some(wanted);
}

/// **The ray-traced reflections' lane, this frame**: the scene is live with the water's tier asking
/// for it (`rt_scene::water_live()`) and the session may trace reflections
/// ([`rt_reflections_capable`]). Independent of the ray-tracing options. Per frame, any world
/// (outside the render world, its latest decision); false before render startup.
pub(crate) fn rt_reflections() -> bool {
    rt_scene::water_live() && rt_reflections_capable()
}

/// **May this session trace reflections at all?** Capable of ray tracing (`rt_scene::capable()`,
/// fixed at render startup) and `$WOW_RT_REFLECT=0` did not turn the lane off (read once). Where it
/// holds the water compiles `RT_REFLECT` and declares the scene for the whole session.
pub(crate) fn rt_reflections_capable() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    rt_scene::capable()
        && *CELL.get_or_init(|| crate::launch_env::var("WOW_RT_REFLECT").as_deref() != Some("0"))
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
        "rt reflect: the water declares the scene; at waterDetail 4 it reflects by ray with the \
         mirror camera off, whatever the ray-tracing options say (WOW_RT_REFLECT=0 keeps the mirror)"
    );
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<RtReflectLane>()
        .init_resource::<RtWaterTier>()
        .add_plugins((
            ExtractResourcePlugin::<RtReflectLane>::default(),
            ExtractResourcePlugin::<RtWaterTier>::default(),
        ));
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            RenderStartup,
            init_rt_reflect
                .after(rt_scene::init_rt_scene)
                .after(super::rt_shadow::init_rt_shadows),
        );
    }
}
