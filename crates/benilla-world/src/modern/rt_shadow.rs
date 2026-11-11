//! fork: F — **ray-traced shadows**, the render-world plumbing (native only).
//!
//! Where [`super::shadow::rt_shadows`] holds this frame, every modern receiver of the sun and of
//! the shadowed lamps takes their visibility through the static world and the terrain as a ray
//! query against the ray-traced scene (`static_gx::rt_scene`), through `benilla_modern::rtshadow`,
//! and the maps
//! keep only what the scene does not hold: the cascades draw the entity casters alone and the lamp
//! atlas no static face (`static_gx::render::modern_shadow`). The sky map keeps its static casters
//! (the ambient's open-sky read and the debug views use it).
//!
//! Where the session may trace shadows ([`super::shadow::rt_shadows_capable`]), the receivers
//! declare the scene for the whole session, and are bound the scene while the lane is live or the
//! stand-in otherwise (`RtScene::lane_bind`: an empty TLAS whose marked rows make
//! `rt_scene::live()` false, so each `rtshadow` twin returns its map-only original). The ray
//! tracing then switches at run time with no pipeline re-specialized:
//!
//! - the Bevy material lanes (`wow_model.wesl` under `WOW_MODERN`, `terrain.wesl` under
//!   `WOW_MODERN`, `liquid_cata.wesl`) declare it as the group after the material's
//!   (`benilla_assets::materials::RT_SHADOW_LAYOUT`, set here at render startup), and bevy_pbr's
//!   `DrawMaterial` binds `MaterialExtraBindGroup` there (`third_party/bevy_pbr/BENILLA.md`, the
//!   fourth delta), which this module fills every frame;
//! - static_gx's colour pipelines declare it as group 3 and `static_gx_pass` binds it;
//! - the voxel probe trace binds it at group 1 (`modern::gi`, where it compiles `RT_SHADOWS`);
//!   the ray-traced probe trace and the froxel rays have pipelines of their own, picked per frame
//!   ([`super::gi::rt_gi`], [`super::volumetrics::rt_froxel`]).
//!
//! No pass of its own: the rays run inside the receivers' fragment stages, so their cost is read
//! as the colour passes' (`static_gx`, `main_opaque_pass_3d`) against a `WOW_RT_SHADOWS=0` run,
//! and the static shadow caster draws `WOW_GX_PERF` counts fall to the sky map's alone.

use bevy::pbr::MaterialExtraBindGroup;
use bevy::prelude::*;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::static_gx::rt_scene::{self, RtScene};

/// `RenderStartup`, after the tier is decided: where the session may trace shadows, publish the
/// scene's layout to the material lanes and give bevy_pbr's draw command the resource it binds.
pub(super) fn init_rt_shadows(mut commands: Commands) {
    if !super::shadow::rt_shadows_capable() {
        return;
    }
    // A second startup (a test app, a renderer rebuilt) finds it set to the same layout.
    let _ = benilla_assets::materials::RT_SHADOW_LAYOUT.set(rt_scene::layout());
    commands.insert_resource(MaterialExtraBindGroup::default());
    info!(
        "rt shadows: the receivers declare the scene; while ray tracing is live, sun and lamp \
         visibility by ray, cascades and lamp atlas for entities only (WOW_RT_SHADOWS=0 turns it \
         off)"
    );
}

/// Every frame: the scene's bind group while a lane that binds it here is live (these shadows, or
/// the water's reflections, `super::rt_reflect`), else its stand-in, where the material lanes' draws
/// bind it. With the reflections alone live (the water's RT tier, the shadows' option off) the
/// group is `RtScene::water_bind`, whose lane row keeps every `rtshadow` twin on raster. The water's
/// shader code also asks its own row of the block (`light_spare[12]`) besides `rt_scene::live()`,
/// and each lane compiles only where it may run, so neither traces for the other.
/// After `prepare_rt_scene` (`PrepareResources`), which builds both.
fn bind_rt_shadows(scene: Option<Res<RtScene>>, extra: Option<ResMut<MaterialExtraBindGroup>>) {
    let (Some(scene), Some(mut extra)) = (scene, extra) else {
        return;
    };
    // The shadow lane's group serves the reflections too; the reflections alone take the scene
    // with the shadow lane's row clear, so the receivers keep their raster shadows.
    let next = if super::shadow::rt_shadows() {
        scene.lane_bind(true)
    } else if super::rt_reflect::rt_reflections() {
        scene.water_bind()
    } else {
        scene.lane_bind(false)
    };
    if extra.0.as_ref().map(|b| b.id()) != next.map(|b| b.id()) {
        extra.0 = next.cloned();
    }
}

pub(super) fn plugin(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .add_systems(
                RenderStartup,
                init_rt_shadows.after(rt_scene::init_rt_scene),
            )
            .add_systems(
                Render,
                bind_rt_shadows.in_set(RenderSystems::PrepareBindGroups),
            );
    }
}
