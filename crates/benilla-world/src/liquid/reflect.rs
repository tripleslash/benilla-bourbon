//! **The planar reflection** — the second, mirrored scene render `ProcWaterAbove` perm 0 samples on
//! `samp1`, and the reason a lake keeps reflecting the shore when you pitch the camera down.
//!
//! ## Why there is a second render at all
//!
//! The port's first cut ran perm 4, whose reflection is a screen-space trace and nothing else: 128
//! yards along the reflected ray, projected, read out of the blurred back buffer, and replaced by a
//! flat horizon constant wherever that lands off-frame. `saturate(0.75 − r.up²)` is the program's
//! own statement that the trace dies as the ray tips skyward — which is precisely what pitching the
//! camera down does — so the water went flat on exactly the shot a player takes of a lake.
//!
//! Perm 4 is not what a `waterDetail 2` player runs. `Liquid::CRenderer::Flush` chooses the
//! program's `reflectMode` as
//!
//! ```text
//! v29 = renderer[+0x30];                                        // 0 = ordinary, 1 = real-reflection
//! if (!v29 && instance->settings[+0x24] == 2) v29 = 2;
//! ```
//!
//! and `CMaterialWater::RenderAbove` indexes the container by `depthFmt + 2·v29 + 6·dynDisplMapCnt
//! + 2·(9·depthMode − 9)`. So mode 0 (perms 0/1) is the default arm, mode 1 (perms 2/3) is the
//! `waterDetail > 2` tier with the *real* buffer, and mode 2 (perms 4/5) is a minority arm gated on
//! one liquid class. The sampler census of the shipped `ProcWaterAbove.bls` separates them cleanly:
//! mode 0 reads `{0,1,2,3,5,6}`, mode 1 `{0,1,2,5,6}`, mode 2 `{0,2,3,5,6}` — mode 0 is the only one
//! that reads both the reflection buffer and the blurred back buffer, and it lerps the second over
//! the first.
//!
//! And the buffer mode 0 reads is not a stand-in. `Liquid::HasFakeReflection()` is `waterDetail !=
//! 0` outright — it does not consult `reflectionMode` at all — and it drives
//! `CWorldScene::RenderReflection(s_defaultReflectionView2, FakeReflectionFB, 0, 0.0)`, which
//! mirrors the camera with `C4Plane::Mirror` about the plane `(0, 0, 1, −0)` and renders the scene
//! again. "Fake" names the *plane*, not the render: one fixed mirror at world `z = 0` for every
//! liquid surface on the map, rather than one per surface height.
//!
//! ## The mirror, and the flip that pays for it
//!
//! WoW's `z = 0` is Bevy's `y = 0`, so the mirror is `M = diag(1, −1, 1)`. `M · C` is a reflection —
//! determinant −1 — and a Bevy `Transform` stores a `Quat`, which cannot hold one. So the camera is
//! built as the nearest proper rotation instead: `Transform::looking_to(M·forward, M·up)`, which
//! derives `right = (M·up) × (M·back) = −M·right`. That is `M · C · diag(−1, 1, 1)` — a true
//! rotation, correct winding, correct back-face culling — whose picture is the mirror's, flipped
//! left-to-right. `liquid_cata.wgsl` undoes it with one `1 − u` at the read.
//!
//! The alternative was flipping the cull mode of every material in the scene for one view, which
//! Bevy does not expose per-view at all.
//!
//! ## What this camera deliberately is not
//!
//! It carries **no** `LiquidScreenView`, no `Ssilvb`, no `FfxGlow` and no
//! `RenderLayers`. The first three are the markers the client's own graph nodes key on, so each of
//! them stands down for this view exactly as it already does for a portrait booth. The absence of
//! `RenderLayers` is load-bearing in the other direction: `particles::sim` resolves an emitter's
//! camera by *finding the first camera whose layers intersect* and its booth query takes
//! `&RenderLayers`, so a reflection camera carrying one would start answering for world emitters
//! (decision 0775's failure, from the other end). Bevy reads a camera with no `RenderLayers` as
//! layer 0, which is what this wants.
//!
//! Water is kept out of its own mirror by the opposite move: liquid surfaces ride
//! [`LIQUID_RENDER_LAYER`] and this camera does not. That is the reference's own arrangement — its
//! reflection view never sets the `0x200` water-pass bit, so `RenderAlphaWater` simply does not run
//! for it.
//!
//! ## What the mirror does NOT contain, and why each is left out
//!
//! Standing a node down for this view is free; teaching one to run for it is not, because several
//! of them carry a cull list that is the world camera's by construction. Three are out:
//!
//! * **The retained static pass** (`static_gx`) — the merged ADT-doodad cells and WMO prop regions.
//!   Its marker exists precisely because "the cull list is the world camera's", so running it here
//!   would paint the world camera's visible set with the mirror's matrices. **This is the one gap a
//!   player can see**: where a zone's trees and props are baked into the retained pass rather than
//!   drawn as entities, they are missing from the mirror. The screen-space trace still carries them
//!   wherever it lands, which is most of the frame at the shallow angles a reflection reads
//!   strongest at — so the symptom is a lake that reflects its shore and its buildings but thins
//!   out on the props as the camera pitches down, not one that goes flat. Closing it means a second
//!   cull list for that pass, which is its own piece of work.
//! * **The indirect-lighting multiply** (`ssilvb`) and **the glow** (`ffx_glow`), for the same
//!   marker reason. Both are whole-frame filters on the world view, and the reference clears a
//!   comparable pile of view flags on its own reflection view before rendering it.
//!
//! And one thing the mirror contains that the reference's does not: **everything below the water**.
//! The reference clips its reflection view against the mirror plane; `Projection` has no oblique
//! near-plane hook here, so the riverbed is reflected too. It is nearly harmless in this program,
//! because every read of this buffer is 20 yards along a bent ray and then lerped against the
//! screen-space trace — but it is a real difference and it is why this file names the plane.

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::view::Hdr;

use super::screen::LiquidScreenTargets;
use super::{Underwater, WaterDetail};
use crate::view::WorldCamera;
use benilla_assets::materials::LiquidCataMaterial;

/// The render layer every liquid surface rides, so the reflection camera can be the one world view
/// that does not draw water.
///
/// **31 is the player shadow's and 2.. is the portrait ladder's**; this is the one layer claimed
/// from `benilla-world` rather than from `benilla-app`'s ladder, because the entities on it are
/// spawned here and the camera that must not see them is spawned here. It is deliberately at the
/// far end of the range from the app's growing ladder, and it is asserted distinct from the one
/// fixed app layer this crate can see.
///
/// Unlike a booth layer this is ADDITIVE: the world camera renders layer 0 *and* this one
/// ([`keep_water_on_the_world_camera`]), because water is part of the world. Only the mirror is
/// missing it.
pub(crate) const LIQUID_RENDER_LAYER: usize = 30;

/// The mirror's camera. One per world camera, spawned once the targets exist.
#[derive(Component)]
pub(crate) struct LiquidReflectionCamera;

/// Add [`LIQUID_RENDER_LAYER`] to the world camera, so it still draws the water the mirror does not.
///
/// Guarded and additive, and it has to be both: `player_shadow` writes the same component with the
/// same shape, and its teardown removes the whole component only when it was the one that inserted
/// it. Running this from `Startup` means the component already exists by the time the shadow lane
/// first touches it, so that branch never fires — and if it ever did, the next frame repairs it.
fn keep_water_on_the_world_camera(
    mut commands: Commands,
    mut cams: Query<(Entity, Option<&mut RenderLayers>), With<WorldCamera>>,
) {
    for (entity, layers) in &mut cams {
        match layers {
            Some(mut layers) => {
                if layers.intersects(&RenderLayers::layer(LIQUID_RENDER_LAYER)) {
                    continue; // guarded — the steady frame never flags the component changed
                }
                *layers = layers.clone().with(LIQUID_RENDER_LAYER);
            }
            None => {
                commands
                    .entity(entity)
                    .insert(RenderLayers::default().with(LIQUID_RENDER_LAYER));
            }
        }
    }
}

/// Spawn the mirror's camera once — after the targets exist and the world camera does.
///
/// Its `order` is below the world camera's so it renders first: the water reads the buffer in the
/// same frame it was filled, and a frame-old mirror on a turning camera reads as the reflection
/// sliding against the world.
fn spawn_reflection_camera(
    mut commands: Commands,
    targets: Option<Res<LiquidScreenTargets>>,
    world_cam: Query<(&Projection, &Msaa), With<WorldCamera>>,
    existing: Query<(), With<LiquidReflectionCamera>>,
) {
    let (Some(targets), Ok((projection, msaa))) = (targets, world_cam.single()) else {
        return;
    };
    if !existing.is_empty() {
        return;
    }
    commands.spawn((
        Camera3d::default(),
        LiquidReflectionCamera,
        Camera {
            // Before the world camera (which is 0), and inactive until `sync` says the mirror is
            // wanted — the first frame has no pose for it yet.
            order: -1,
            is_active: false,
            ..default()
        },
        // `RenderTarget` is its own component in 0.18, and `world_backdrop` re-points the world
        // camera's the same way on a resize — see `sync_reflection_camera` for this one's.
        RenderTarget::Image(ImageRenderTarget::from(targets.reflect.clone())),
        // The same colour space as the frame the water is drawn into. The reflection is mixed with
        // `scene_blur`, which IS the world camera's target, so the two have to agree or the mirror
        // reads as a differently-exposed picture pasted into the water.
        Hdr,
        Tonemapping::None,
        // **The world camera's sample count, not `Off`** — and it is a pipeline-count decision,
        // not a quality one. Bevy specializes every material pipeline per view sample count, so a
        // mirror at a different count would compile a SECOND copy of every pipeline in the scene.
        // On the browser backend those compiles are the frame-time cliff this client already
        // watches; on any backend it doubles the cache for a buffer nobody can see an edge in.
        // `gxMultisample` is latched at camera spawn and never moves (see `player::setup`), so
        // copying it once here is copying it for good.
        *msaa,
        projection.clone(),
        Transform::IDENTITY,
    ));
}

/// Mirror the world camera into it, and decide whether the mirror runs at all.
///
/// The three conditions are the reference's own: `HasFakeReflection()` is `waterDetail != 0`, and
/// `RenderReflection` is skipped outright while `Liquid::CameraSubmerged()` — from underneath there
/// is nothing above the surface to mirror, and `ProcWaterBelow` does not sample `samp1` in any
/// permutation.
fn sync_reflection_camera(
    detail: Res<WaterDetail>,
    underwater: Res<Underwater>,
    targets: Option<Res<LiquidScreenTargets>>,
    world_cam: Query<
        (&GlobalTransform, &Projection, &Camera),
        (With<WorldCamera>, Without<LiquidReflectionCamera>),
    >,
    mut mirror: Query<
        (&mut Transform, &mut Projection, &mut Camera, &mut RenderTarget),
        (With<LiquidReflectionCamera>, Without<WorldCamera>),
    >,
) {
    let Ok((mut pose, mut projection, mut camera, mut target)) = mirror.single_mut() else {
        return;
    };
    let Ok((world_pose, world_projection, world_camera)) = world_cam.single() else {
        camera.is_active = false;
        return;
    };
    let wanted = detail.is_cata() && !underwater.0.any();
    if camera.is_active != wanted {
        camera.is_active = wanted;
    }
    if !wanted {
        return;
    }

    // Re-point at a resize (`screen::resize_targets` publishes fresh `AssetId`s rather than
    // mutating behind the handle — see its header for why). Guarded on the asset id, because a
    // blind write every frame is a `Changed<RenderTarget>` every frame, and the target's own
    // bookkeeping keys off that.
    if let Some(targets) = targets {
        let stale = match &*target {
            RenderTarget::Image(image) => image.handle.id() != targets.reflect.id(),
            _ => true,
        };
        if stale {
            *target = RenderTarget::Image(ImageRenderTarget::from(targets.reflect.clone()));
        }
    }
    // The frame the mirror clears to is the frame the world clears to: whatever the sky does not
    // cover reads as the same horizon on both sides of the surface. Written blind — neither
    // `ClearColorConfig` nor `Projection` implements `PartialEq`, and both are one component on
    // one entity that nothing else watches for changes (the `scene_still` gate in
    // `model_render::visibility` reads the WORLD camera's).
    camera.clear_color = world_camera.clear_color.clone();
    *projection = world_projection.clone();

    // `M = diag(1, −1, 1)` — WoW `z = 0` is Bevy `y = 0`. See the header for why this is
    // `looking_to` and not a reflection matrix, and what the shader owes it.
    let mirrored = |v: Vec3| Vec3::new(v.x, -v.y, v.z);
    let world = world_pose.compute_transform();
    let want = Transform::from_translation(mirrored(world.translation))
        .looking_to(mirrored(*world.forward()), mirrored(*world.up()));
    if *pose != want {
        *pose = want;
    }
}

/// Tell the water whether the mirror is live this frame — `LiquidParams.path.z`.
///
/// Value-watched, the shape `surface::sync_below_water` already uses: a write here marks every cata
/// material changed and rebuilds its bind group, so the steady frame must not make one. With the
/// flag clear the shader falls back to the flat horizon colour, which is not a guess — it is what
/// perm 4 has in the mirror's place.
fn sync_reflection_flag(
    mut materials: ResMut<Assets<LiquidCataMaterial>>,
    mirror: Query<&Camera, With<LiquidReflectionCamera>>,
    mut last: Local<Option<bool>>,
) {
    let live = mirror.single().is_ok_and(|c| c.is_active);
    if *last == Some(live) {
        return;
    }
    *last = Some(live);
    let flag = if live { 1.0 } else { 0.0 };
    for (_, material) in materials.iter_mut() {
        material.extension.path.z = flag;
    }
}

/// Wire the mirror into the app. Called from [`super::LiquidPlugin`].
pub(super) fn register(app: &mut App) {
    app.add_systems(
        Startup,
        keep_water_on_the_world_camera.after(super::surface::setup_liquid),
    )
    .add_systems(
        Update,
        (
            // The world camera can spawn after `Startup` (the fallback free-fly path spawns a
            // different one), so both the layer patch and the spawn stay in `Update`. Both are
            // guarded, so the steady frame is one query and a compare.
            keep_water_on_the_world_camera,
            spawn_reflection_camera,
            sync_reflection_camera,
            sync_reflection_flag,
        )
            .chain()
            // AFTER the submersion verdict: the mirror stands down while the eye is under water,
            // and a frame-old verdict would show one frame of the above-water mirror through the
            // surface on every dive.
            .after(super::SubmersionVerdict),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The mirror is a rotation, and its picture is flipped left-to-right.** Both halves are the
    /// contract `liquid_cata.wgsl`'s `1 − u` depends on, and neither is visible at the call site.
    #[test]
    fn the_mirrored_pose_is_a_rotation_with_its_right_hand_flipped() {
        // Looking north-ish and down, from above the mirror plane — the pose the report was taken
        // at.
        let world = Transform::from_translation(Vec3::new(3.0, 10.0, -7.0))
            .looking_to(Vec3::new(0.0, -0.5, -1.0).normalize(), Vec3::Y);
        let m = |v: Vec3| Vec3::new(v.x, -v.y, v.z);
        let mirror = Transform::from_translation(m(world.translation))
            .looking_to(m(*world.forward()), m(*world.up()));

        // Under the plane, by as much as the world camera is over it.
        assert!((mirror.translation - Vec3::new(3.0, -10.0, -7.0)).length() < 1e-5);
        // The view direction is the reflected one — this is what makes it a mirror at all.
        assert!((*mirror.forward() - m(*world.forward())).length() < 1e-5);
        // And the right hand is the NEGATED reflection, which is the whole of the shader's debt.
        assert!((*mirror.right() + m(*world.right())).length() < 1e-5);
    }

    /// A camera looking straight down mirrors to one looking straight up from the same depth — the
    /// degenerate case for `looking_to`, whose `up` is parallel to nothing it is handed.
    #[test]
    fn straight_down_mirrors_to_straight_up() {
        let world =
            Transform::from_translation(Vec3::new(0.0, 25.0, 0.0)).looking_to(Vec3::NEG_Y, Vec3::Z);
        let m = |v: Vec3| Vec3::new(v.x, -v.y, v.z);
        let mirror = Transform::from_translation(m(world.translation))
            .looking_to(m(*world.forward()), m(*world.up()));
        assert!((mirror.translation - Vec3::new(0.0, -25.0, 0.0)).length() < 1e-5);
        assert!((*mirror.forward() - Vec3::Y).length() < 1e-5);
    }

    /// The layer this crate claims must not be the one app layer it can name. The rest of the
    /// ladder is asserted in `portrait`, where it lives.
    #[test]
    fn the_water_layer_is_clear_of_the_player_shadows() {
        // `benilla_app::player_shadow::PLAYER_SHADOW_LAYER` — restated rather than imported,
        // because the dependency runs the other way.
        const PLAYER_SHADOW_LAYER: usize = 31;
        assert_ne!(LIQUID_RENDER_LAYER, PLAYER_SHADOW_LAYER);
    }
}
