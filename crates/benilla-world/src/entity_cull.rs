//! fork: perf — GPU-driven drawing and two-phase GPU occlusion culling for the entity lane, the
//! meshes Bevy draws through the world camera (models, terrain, clutter, rigs).
//!
//! The reference culls on the CPU only: frustum and portal walks, then every elected object draws.
//! What draws here is still decided that way first — benilla's own visibility writes `Visibility`
//! in `Update` and `PostUpdate` ahead of `VisibilitySystems::CheckVisibility` (`exterior_cull`,
//! `model_render`'s `ModelVisSet`, `lazy::realize_bound`), and only what that leaves visible is
//! extracted. On top of it the world camera, opted in, draws its binned phases through Bevy's
//! indirect lane, whose compute preprocessing frustum-tests each instance and then tests it
//! against a depth pyramid. It hides only what the frame's own depth already hides, so it changes
//! the cost of a frame and never its picture.
//!
//! The two phases (`bevy_pbr`'s `mesh_preprocess.wgsl`): the early phase draws, in the depth
//! prepass, every instance that passes against LAST frame's pyramid (built after the main pass, so
//! it holds everything that wrote depth); the rest are re-tested against a pyramid of THIS frame's
//! early prepass and the survivors draw in the late prepass. The late test is the authority, so the
//! prepass must write exactly the depth the main pass would: every opaque or alpha-masked entity
//! material carries a prepass twin of its own vertex and discards (`wow_model.wesl`,
//! `terrain.wesl`), and every other material keeps out of the prepass (`enable_prepass`). A
//! `NoFrustumCulling` mesh — dress, equipment, particles, the player-shadow proxy, whose boxes are
//! a bind pose or absent — is never occluded (`third_party/bevy_pbr/BENILLA.md`).
//!
//! On by default on native, where the world camera draws through the indirect lane with the depth
//! prepass; `$WOW_ENTITY_GPU_CULL=0` turns it off (every camera on direct draws, no prepass), and it
//! is never on on wasm32. Off, nothing here runs and no pipeline changes: the prepass twins compile
//! only into prepass pipelines, and the shader cuts are inert without `PREPASS_PIPELINE` and
//! `DEPTH_PREPASS`.
//!
//! The prepass is not optional under this scheme: Bevy 0.19's two phases ARE prepass draws, and the
//! late test reads this frame's prepass depth. Culling against last frame's pyramid alone would pop
//! what a frame uncovers; splitting the main opaque pass in two instead needs a
//! `bevy_core_pipeline` rework. Opaque and alpha-mask phases are binned, not sorted, so there is no
//! sort for the culling to save either.
//!
//! The static world (`static_gx`) lays its depth in [`WorldOccluders`], after Bevy's early
//! prepass and before the early pyramid is built, so its buildings occlude in this frame's late
//! test as well as the next frame's early one: an entity behind a wall is culled by both. In a
//! classic frame (no static depth prepass) the buildings draw in the main pass and reach only the
//! next frame's pyramid; see [`WorldOccluders`].

use bevy::camera::Camera;
use bevy::core_pipeline::mip_generation::experimental::depth::early_downsample_depth;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::schedule::SystemSet;
use bevy::prelude::*;
use bevy::render::occlusion_culling::OcclusionCulling;
use bevy::render::view::NoIndirectDrawing;

/// Is the world camera on the GPU-driven lane? Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let lever = std::env::var("WOW_ENTITY_GPU_CULL").ok();
        decide(lever.as_deref())
    })
}

/// The gate: on unless `$WOW_ENTITY_GPU_CULL` is `0`, and never on wasm32.
fn decide(lever: Option<&str>) -> bool {
    !cfg!(target_arch = "wasm32") && lever != Some("0")
}

/// Where, in the world camera's render schedule, a depth writer that is not one of Bevy's binned
/// phases lays depth that THIS frame's occlusion test sees: after the early prepass, before the
/// early pyramid is built from the depth texture (`early_downsample_depth`), in the prepass set.
///
/// `static_gx` is here (`static_gx::render::static_gx_occluders`), through the depth texture
/// alone:
///
/// 1. The world camera's depth-only job (the GPU cull's two phases, `static_gx_prepass` and
///    `static_gx_prepass_late`, the `GX_PREPASSED` twins) records and is submitted from this set,
///    on the same `ViewDepthStencilTexture` with `StoreOp::Store`. The early prepass has already
///    taken the depth clear, so the pass loads. Its pipelines keep `GreaterEqual` and write; the
///    compare and the reverse-Z target are Bevy's prepass's own.
/// 2. The colour job records with it, from the same batch (the cull wrote that batch's argument
///    ranges), and is submitted in its own slot in `MainPass` by `static_gx_pass`: its `Equal`
///    test against the depth the twins laid is unchanged, and entity depth already in the buffer
///    only rejects static fragments an entity covers, as the opaque pass would have.
/// 3. In a classic frame (no `GX_PREPASSED`, colour pipelines writing depth themselves) nothing
///    changes: `static_gx_pass` records both in `MainPass`, and the buildings reach the pyramid
///    one frame late.
/// 4. The water mirror never takes this set's work: only the world camera has a pyramid.
///
/// Off the lane (the default) the system is not registered and `static_gx_pass` runs as it always
/// did.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorldOccluders;

/// Put the world camera on the GPU-driven lane, and give the water mirror the layout it shares.
pub(crate) struct EntityCullPlugin;

impl Plugin for EntityCullPlugin {
    fn build(&self, app: &mut App) {
        // Every camera draws direct: `NoIndirectDrawing` must ride a camera's spawn (the work-item
        // buffers latch the first time a view is seen, and a later insert panics in wgpu), so it is
        // a required component, and the world camera's observer takes it back off in the same
        // command flush, before the render world first sees the view.
        app.register_required_components::<Camera, NoIndirectDrawing>();
        if enabled() {
            app.add_observer(arm_world_camera)
                .add_observer(match_static_views);
        }
        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };
        render_app.configure_sets(
            Core3d,
            WorldOccluders
                .in_set(Core3dSystems::Prepass)
                .after(bevy::core_pipeline::prepass::node::early_prepass)
                .before(early_downsample_depth),
        );
    }
}

/// The world camera: indirect draws, the depth prepass, and the occlusion test. Only the world
/// camera — the water mirror, the portrait booths and the glue camera keep direct draws and no
/// occlusion test, and only the mirror a prepass ([`match_static_views`]).
fn arm_world_camera(
    add: On<Add<crate::view::WorldCamera>>,
    cameras: Query<(), With<Camera>>,
    mut commands: Commands,
) {
    if cameras.contains(add.entity) {
        commands
            .entity(add.entity)
            .remove::<NoIndirectDrawing>()
            .insert((DepthPrepass, OcclusionCulling));
    }
}

/// A secondary static-world view (the water mirror) takes the depth prepass too, and nothing else:
/// no occlusion test, direct draws. `static_gx` specializes its pipelines from the world camera's
/// mesh-view layout and binds them in every `StaticGxView`, and `DepthPrepass` is a layout bit
/// (the prepass depth at binding 20), so a mirror without it cannot bind them. It also keeps the
/// mirror's material pipelines on the world camera's view key, the one the warm pass compiles.
fn match_static_views(
    add: On<Add<crate::static_gx::StaticGxView>>,
    views: Query<&crate::static_gx::StaticGxView, With<Camera>>,
    mut commands: Commands,
) {
    if views.get(add.entity).is_ok_and(|v| !v.primary) {
        commands.entity(add.entity).insert(DepthPrepass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gate_is_on_unless_the_lever_says_zero() {
        let native = !cfg!(target_arch = "wasm32");
        assert_eq!(decide(None), native);
        assert_eq!(decide(Some("")), native);
        assert!(!decide(Some("0")));
        assert_eq!(decide(Some("1")), native);
    }

    /// The world camera leaves its spawn command on the GPU lane, with no `NoIndirectDrawing`
    /// for the render world to latch; any other camera keeps direct draws and no prepass.
    #[test]
    fn only_the_world_camera_is_armed() {
        let mut app = App::new();
        app.register_required_components::<Camera, NoIndirectDrawing>();
        app.add_observer(arm_world_camera);
        let world_cam = app
            .world_mut()
            .spawn((Camera::default(), crate::view::WorldCamera))
            .id();
        let booth = app.world_mut().spawn(Camera::default()).id();
        // A `WorldCamera` marker on a non-camera (the nameplate tests spawn one) is left alone.
        let bare = app.world_mut().spawn(crate::view::WorldCamera).id();
        app.world_mut().flush();
        let w = app.world();
        assert!(!w.entity(world_cam).contains::<NoIndirectDrawing>());
        assert!(w.entity(world_cam).contains::<DepthPrepass>());
        assert!(w.entity(world_cam).contains::<OcclusionCulling>());
        assert!(w.entity(booth).contains::<NoIndirectDrawing>());
        assert!(!w.entity(booth).contains::<DepthPrepass>());
        assert!(!w.entity(bare).contains::<DepthPrepass>());
    }
}
