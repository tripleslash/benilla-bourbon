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

use bevy::pbr::MaterialExtraBindGroup;
use bevy::prelude::*;
use bevy::render::{RenderApp, RenderStartup};

use crate::static_gx::rt_scene;

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

pub(super) fn plugin(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            RenderStartup,
            init_rt_reflect
                .after(rt_scene::init_rt_scene)
                .after(super::rt_shadow::init_rt_shadows),
        );
    }
}
