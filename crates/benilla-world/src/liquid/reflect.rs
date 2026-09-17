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
//! [`UNMIRRORED_RENDER_LAYER`] and this camera does not. That is the reference's own arrangement — its
//! reflection view never sets the `0x200` water-pass bit, so `RenderAlphaWater` simply does not run
//! for it.
//!
//! ## The retained static pass, which this view DOES carry
//!
//! The buildings are not entities. Every WMO group's geometry, every merged ADT-doodad cell and
//! every WMO prop region leaves bevy_pbr for `static_gx`'s retained buffers, drawn by one custom
//! node keyed on its own view marker — and while that marker sat on the world camera alone, the
//! mirror rendered a world with no buildings and no trees in it. A lake reflected its terrain, its
//! creatures and its animated doodads, and *not* the town on its shore, which is the one thing a
//! player looks at a reflection to see. The screen-space trace over the top carried them wherever
//! it landed, so the symptom was a shore that faded out of the water as the camera pitched down
//! rather than one that was never there — which is exactly why it read as a water bug and not as a
//! missing pass.
//!
//! So this camera spawns with [`StaticGxView::MIRROR`](crate::static_gx::StaticGxView::MIRROR),
//! and the retained node runs for it. Every
//! piece of view state that node uses — the target, the depth, the view uniform, and with it the
//! oblique near plane below — comes out of its `ViewQuery`, so the mirror's draws are the mirror's
//! matrices with no second copy of anything.
//!
//! What it reuses rather than rebuilds is the **cull list**, which is the world camera's frustum,
//! farclip and portal flood. For every cell in that list the mirror's picture is exact; a cell the
//! mirror's frustum contains and the world camera's does not is missing, which is the near-vertical
//! shot (pitch straight down and the mirror wants what is overhead and behind you). The marker's
//! own doc carries why a second cull list is not the trade this port takes.
//!
//! ## What the mirror does NOT contain, and why each is left out
//!
//! Standing a node down for this view is free; teaching one to run for it is not. Two stay out:
//!
//! * **The indirect-lighting multiply** (`ssilvb`) and **the glow** (`ffx_glow`), for the marker
//!   reason. Both are whole-frame filters on the world view, and the reference clears a
//!   comparable pile of view flags on its own reflection view before rendering it.
//!
//! ## Only what is above the water, and only where the mirror is about this water
//!
//! The first cut of this file rendered the mirror unclipped, and the riverbed came back up onto the
//! surface as a reflection of the world *below* it. Three things fix that, and only the first is
//! the reference's:
//!
//! 1. **The oblique clip plane.** `RenderReflection` transforms the mirror plane into view space
//!    and wraps `CWorldView::Render` in `CShaderEffect::SetClipPlane` / `SetClipEnabled(1)`, so
//!    every one of its shaders discards the far side. benilla has no per-view clip term in its
//!    material shaders and could not gain one without editing every world shader it has — so it
//!    takes the other standard route to the same clip, Lengyel's oblique near plane, which lives
//!    entirely in the projection matrix and therefore applies to every draw with no shader change
//!    at all. Bevy ships it as `PerspectiveProjection::near_clip_plane`, already adapted to its
//!    reversed-Z convention. [`view_space_clip_plane`] builds the plane; the tests check a point
//!    above the water renders and one below it does not.
//! 2. **The plane follows the water, not sea level.** The reference passes `0.0` for the fake
//!    buffer and keeps its tracked `GetClosestReflectionHeight` for the real one. Sea level is the
//!    right plane for an ocean and badly wrong for anything else — mirroring a mountain lake about
//!    `z = 0` reflects a slab of world from hundreds of yards below it — and, worse for the bug
//!    above, clipping at `z = 0` for a lake at `z = 200` clips nothing that lake can see. So the
//!    plane is the height of the water the eye is over, resolved through the same per-subject
//!    liquid query everything else asks.
//! 3. **A height guard in the shader.** One mirrored render is about one height, and a surface at
//!    some other height would wear a picture of the world reflected about someone else's
//!    waterline. `liquid_cata.wgsl` fades the mirror out over `MIRROR_PLANE_FADE_START..END` yards
//!    of disagreement, back to the flat horizon colour perm 4 uses. It carries a second guard for
//!    the same family of ghost: the bent-ray read leaves the mirror's frame at the margins, and
//!    every screen target here is ClampToEdge, so unguarded that smears one border column across
//!    yards of water.
//!
//! Guards 2 and 3 are this port's, not the reference's, and they exist because this renders **one**
//! mirror per frame where the reference re-renders per instance group.

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::view::Hdr;

use super::screen::LiquidScreenTargets;
use super::{Underwater, WaterDetail};
use crate::view::WorldCamera;
use crate::world_point::{Subject, WorldPoint};
use benilla_assets::coords::bevy_to_wow;
use benilla_assets::materials::LiquidCataMaterial;

/// **The mirror plane this frame, and whether it is usable** — the reference's
/// `Liquid::GetClosestReflectionHeight`, resolved for the fake buffer too.
///
/// The reference passes `0.0` as `RenderReflection`'s plane for the fake buffer and reserves the
/// tracked closest height for the real one. Sea level is the right plane for an ocean and the wrong
/// one for everything else, and "the wrong plane" is not a subtle error: mirroring a mountain lake
/// about `z = 0` reflects a slab of world from four hundred yards below it onto the surface. So the
/// plane here is the height of the water the eye is actually over, and [`plane`] is published to
/// the shader so a surface at some *other* height can fade the mirror out rather than wear a
/// picture that was never about it.
#[derive(Resource, Default, Clone, Copy)]
pub(crate) struct ReflectionPlane {
    /// World Y (Bevy) of the surface the mirror is about. Meaningless while `live` is false.
    plane: f32,
    /// Is the mirror rendering this frame at all?
    live: bool,
}

impl ReflectionPlane {
    /// The plane (world Y) while the mirror renders this frame — what the modern lanes compare the
    /// eye against to know they are drawing for the mirror (`sunshadow::mirror_view`).
    pub(crate) fn live_plane(&self) -> Option<f32> {
        self.live.then_some(self.plane)
    }
}

/// **The layer for things the world draws and the mirror must not** — the one render layer whose
/// meaning is a negation.
///
/// Unlike a booth layer this is ADDITIVE: the world camera renders layer 0 *and* this one
/// ([`keep_unmirrored_on_the_world_camera`]), so everything on it is a full citizen of the frame.
/// Only [`LiquidReflectionCamera`], which carries no `RenderLayers` at all and therefore sees layer
/// 0 alone, is missing it.
///
/// Two kinds of thing belong here, and the second is why this is not called the water's layer:
///
/// * **Liquid surfaces.** Water must not appear in its own mirror. That is the reference's own
///   arrangement — its reflection view never sets the `0x200` water-pass bit, so `RenderAlphaWater`
///   simply does not run for it.
/// * **World-pass billboards that face the WORLD camera.** An overhead unit name
///   (`benilla_app::nameplates`) is a real depth-tested mesh in the 3-D pass, not an overlay, and
///   it is oriented every frame to the world camera's basis. Seen from anywhere else that basis is
///   wrong by construction, so the mirror rendered a plate that was aimed at a viewer it does not
///   have — legible text, upside down, floating in the lake. A reflection is not a second opinion
///   about a billboard; a billboard has exactly one viewer, and this layer is how the frame says so.
///
/// **31 is the player shadow's and 2.. is the portrait ladder's**; this is the one layer claimed
/// from `benilla-world` rather than from `benilla-app`'s ladder, because the camera that must not
/// see it is spawned here. It is deliberately at the far end of the range from the app's growing
/// ladder, and it is asserted distinct from the one fixed app layer this crate can see.
pub const UNMIRRORED_RENDER_LAYER: usize = 30;

/// The mirror's camera. One per world camera, spawned once the targets exist.
#[derive(Component)]
pub(crate) struct LiquidReflectionCamera;

/// Add [`UNMIRRORED_RENDER_LAYER`] to the world camera, so it still draws everything the mirror
/// does not.
///
/// Guarded and additive, and it has to be both: `player_shadow` writes the same component with the
/// same shape, and its teardown removes the whole component only when it was the one that inserted
/// it. Running this from `Startup` means the component already exists by the time the shadow lane
/// first touches it, so that branch never fires — and if it ever did, the next frame repairs it.
fn keep_unmirrored_on_the_world_camera(
    mut commands: Commands,
    mut cams: Query<(Entity, Option<&mut RenderLayers>), With<WorldCamera>>,
) {
    for (entity, layers) in &mut cams {
        match layers {
            Some(mut layers) => {
                if layers.intersects(&RenderLayers::layer(UNMIRRORED_RENDER_LAYER)) {
                    continue; // guarded — the steady frame never flags the component changed
                }
                *layers = layers.clone().with(UNMIRRORED_RENDER_LAYER);
            }
            None => {
                commands
                    .entity(entity)
                    .insert(RenderLayers::default().with(UNMIRRORED_RENDER_LAYER));
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
        // **The retained static pass runs for this view** — the buildings, the merged doodad
        // cells and the WMO props, none of which are entities and all of which were absent from
        // the mirror while this marker lived on the world camera alone. See the header.
        crate::static_gx::StaticGxView::MIRROR,
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

/// The minimum height the eye must sit above the mirror plane for the mirror to run.
///
/// At zero separation the mirrored camera coincides with the real one and the oblique clip plane
/// passes through the eye: `near_clip_plane`'s `w` goes to zero, Lengyel's scale factor `a =
/// −Qz / (C · Q)` divides by ~0, and the projection comes out as garbage rather than as a
/// degenerate-but-safe matrix. Standing the mirror down for the half yard either side of the
/// waterline costs nothing visible — at that height the surface is edge-on and the Fresnel term
/// that weights the reflection is at its own floor.
const MIN_EYE_HEIGHT: f32 = 0.5;

/// Mirror the world camera into it, and decide whether the mirror runs at all.
///
/// The conditions are the reference's own plus one: `HasFakeReflection()` is `waterDetail != 0`,
/// and `RenderReflection` is skipped outright while `Liquid::CameraSubmerged()` — from underneath
/// there is nothing above the surface to mirror, and `ProcWaterBelow` does not sample `samp1` in
/// any permutation. The one addition is [`MIN_EYE_HEIGHT`].
#[allow(clippy::too_many_arguments)]
fn sync_reflection_camera(
    detail: Res<WaterDetail>,
    underwater: Res<Underwater>,
    targets: Option<Res<LiquidScreenTargets>>,
    world: WorldPoint,
    mut plane_out: ResMut<ReflectionPlane>,
    mut held: Local<f32>,
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
        plane_out.live = false;
        return;
    };

    // **The plane, and it is the water the eye is over — not sea level.** `water_surface_at` is the
    // same per-subject query every other liquid consumer asks, so an interior pool answers for a
    // camera inside its building and the ADT's MCLQ answers outdoors. Where the eye is over no
    // water at all the last plane is HELD rather than snapped to zero: stepping off a shore would
    // otherwise move the mirror four hundred yards in one frame, and the height guard in the shader
    // already fades a surface that disagrees with the plane. Zero is only the value before the eye
    // has ever been over water.
    let eye = world_pose.translation();
    let plane = match world.water_surface_at(Subject::Eye, bevy_to_wow(eye)) {
        Some(surface) => {
            *held = surface;
            surface
        }
        None => *held,
    };

    // fork: `waterDetail 1` (SSR only) never renders the mirror; the shader's flag stays clear and
    // it reflects through the screen-space trace over the horizon colour.
    let wanted = detail.has_planar_reflection() && !underwater.0.any() && eye.y - plane > MIN_EYE_HEIGHT;
    if camera.is_active != wanted {
        camera.is_active = wanted;
    }
    *plane_out = ReflectionPlane {
        plane,
        live: wanted,
    };
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

    // The mirror about `y = plane`. A DIRECTION reflects the same whatever the plane's height —
    // only the position carries it — which is why the two halves are spelled differently.
    // See the header for why this is `looking_to` and not a reflection matrix, and what the shader
    // owes it.
    let want = mirror_pose(&world_pose.compute_transform(), plane);
    if *pose != want {
        *pose = want;
    }

    // **The oblique clip plane — the whole of the "only things ABOVE the water" rule.**
    //
    // The reference does this with a user clip plane: `RenderReflection` transforms the mirror
    // plane into view space and spends `CShaderEffect::SetClipPlane` / `SetClipEnabled(1)` around
    // `CWorldView::Render`, so every one of its shaders discards the far side. benilla has no such
    // per-view term in its material shaders and could not gain one without touching every world
    // shader it has — so it takes the other standard route to the same clip, Lengyel's oblique
    // near plane, which lives entirely in the projection matrix and therefore applies to every
    // draw the mirror makes with no shader changes at all. Bevy ships it as
    // `PerspectiveProjection::near_clip_plane`, already adapted to its reversed-Z convention.
    //
    // Without it the mirror renders the riverbed, the lake floor and every rock under the surface,
    // and the water shader reads them back out as a reflection of the world *below* it — which is
    // the ghost this exists to remove.
    if let Projection::Perspective(perspective) = &mut *projection {
        perspective.near_clip_plane = view_space_clip_plane(&want, plane);
    }
}

/// The mirrored pose of `world` about the horizontal plane `y = plane`.
///
/// Split out so the two things a test can actually pin — that this is a rotation, and that its
/// right hand is negated (the shader's `1 − u`) — are pinned against the function the camera uses
/// rather than against a copy of it.
fn mirror_pose(world: &Transform, plane: f32) -> Transform {
    let flip = |v: Vec3| Vec3::new(v.x, -v.y, v.z);
    let position = Vec3::new(
        world.translation.x,
        2.0 * plane - world.translation.y,
        world.translation.z,
    );
    Transform::from_translation(position).looking_to(flip(*world.forward()), flip(*world.up()))
}

/// The world plane `y = plane` in the VIEW space of `pose`, in the form
/// `PerspectiveProjection::near_clip_plane` wants: a unit normal pointing at the side that survives,
/// and `w = −(N · Q)` for any `Q` on the plane.
///
/// The surviving side is world `+Y` — everything above the water. The mirror's camera sits *below*
/// the plane looking up, so `+Y` also points away from it, which is the orientation the field
/// documents.
fn view_space_clip_plane(pose: &Transform, plane: f32) -> Vec4 {
    let view_from_world = pose.to_matrix().inverse();
    // A direction, so `w = 0` — the translation must not reach it.
    let normal = (view_from_world * Vec4::new(0.0, 1.0, 0.0, 0.0))
        .truncate()
        .normalize_or_zero();
    let point = (view_from_world * Vec4::new(0.0, plane, 0.0, 1.0)).truncate();
    normal.extend(-normal.dot(point))
}

/// Tell the water whether the mirror is live this frame and what plane it is about —
/// `LiquidParams.path.z` and `.w`.
///
/// Value-watched, the shape `surface::sync_below_water` already uses: a write here marks every cata
/// material changed and rebuilds its bind group, so the steady frame must not make one. The plane
/// moves continuously as the eye crosses water, so the watch has a dead band rather than an
/// equality test — a tenth of a yard is well under the scale the shader's height guard fades over,
/// and it is the difference between rebuilding thirty bind groups on the frames the plane really
/// moved and rebuilding them on every frame a boat rocks.
///
/// With the flag clear the shader falls back to the flat horizon colour, which is not a guess — it
/// is what perm 4 has in the mirror's place.
fn sync_reflection_flag(
    mut materials: ResMut<Assets<LiquidCataMaterial>>,
    reflection: Res<ReflectionPlane>,
    mut last: Local<Option<(bool, f32)>>,
) {
    /// How far the plane may drift before the materials are told. See the doc above.
    const PLANE_DEAD_BAND: f32 = 0.1;

    let want = (reflection.live, reflection.plane);
    if let Some((live, plane)) = *last {
        if live == want.0 && (plane - want.1).abs() < PLANE_DEAD_BAND {
            return;
        }
    }
    *last = Some(want);
    let flag = if want.0 { 1.0 } else { 0.0 };
    for (_, material) in materials.iter_mut() {
        material.extension.path.z = flag;
        material.extension.path.w = want.1;
    }
}

/// Wire the mirror into the app. Called from [`super::LiquidPlugin`].
pub(super) fn register(app: &mut App) {
    app.init_resource::<ReflectionPlane>()
        .add_systems(
            Startup,
            keep_unmirrored_on_the_world_camera.after(super::surface::setup_liquid),
        )
        .add_systems(
        Update,
        (
            // The world camera can spawn after `Startup` (the fallback free-fly path spawns a
            // different one), so both the layer patch and the spawn stay in `Update`. Both are
            // guarded, so the steady frame is one query and a compare.
            keep_unmirrored_on_the_world_camera,
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
    use bevy::camera::CameraProjection;
    use bevy::render::view::Msaa;

    /// The clip test wgpu applies, on a world point: kept iff `0 ≤ clip.z ≤ clip.w`.
    ///
    /// Written against the real `PerspectiveProjection` — Bevy's own Lengyel implementation for its
    /// reversed-Z convention — rather than against a restatement of it, because the whole value of
    /// these tests is that they fail if that convention ever moves.
    fn is_rendered(pose: &Transform, plane: f32, world_point: Vec3) -> bool {
        let projection = PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_4,
            aspect_ratio: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
            near_clip_plane: view_space_clip_plane(pose, plane),
        };
        let clip = projection.get_clip_from_view()
            * pose.to_matrix().inverse()
            * world_point.extend(1.0);
        clip.z >= 0.0 && clip.z <= clip.w
    }

    /// A camera over water, looking down at it — the pose the report was taken at.
    fn eye_over_water() -> Transform {
        Transform::from_translation(Vec3::new(0.0, 10.0, 0.0))
            .looking_to(Vec3::new(0.0, -0.5, -1.0).normalize(), Vec3::Y)
    }

    /// **The mirror is a rotation, and its picture is flipped left-to-right.** Both halves are the
    /// contract `liquid_cata.wgsl`'s `1 − u` depends on, and neither is visible at the call site.
    #[test]
    fn the_mirrored_pose_is_a_rotation_with_its_right_hand_flipped() {
        let world = eye_over_water();
        let mirror = mirror_pose(&world, 0.0);
        let flip = |v: Vec3| Vec3::new(v.x, -v.y, v.z);

        // Under the plane, by as much as the world camera is over it.
        assert!((mirror.translation - Vec3::new(0.0, -10.0, 0.0)).length() < 1e-5);
        // The view direction is the reflected one — this is what makes it a mirror at all.
        assert!((*mirror.forward() - flip(*world.forward())).length() < 1e-5);
        // And the right hand is the NEGATED reflection, which is the whole of the shader's debt.
        assert!((*mirror.right() + flip(*world.right())).length() < 1e-5);
    }

    /// The plane is not always sea level, and the mirror has to follow it — a camera 10 yards over
    /// a lake at 200 mirrors to 10 yards under **that** lake, not to −10.
    #[test]
    fn the_mirror_hangs_below_the_plane_it_is_given() {
        let world = Transform::from_translation(Vec3::new(4.0, 210.0, -3.0));
        let mirror = mirror_pose(&world, 200.0);
        assert!((mirror.translation - Vec3::new(4.0, 190.0, -3.0)).length() < 1e-4);
    }

    /// A camera looking straight down mirrors to one looking straight up from the same depth — the
    /// degenerate case for `looking_to`, whose `up` is parallel to nothing it is handed.
    #[test]
    fn straight_down_mirrors_to_straight_up() {
        let world =
            Transform::from_translation(Vec3::new(0.0, 25.0, 0.0)).looking_to(Vec3::NEG_Y, Vec3::Z);
        let mirror = mirror_pose(&world, 0.0);
        assert!((mirror.translation - Vec3::new(0.0, -25.0, 0.0)).length() < 1e-5);
        assert!((*mirror.forward() - Vec3::Y).length() < 1e-5);
    }

    /// **The reported bug: the mirror rendered what was UNDER the water.** The oblique clip plane
    /// is the whole fix, and this is the test that says so — a shoreline tree survives, the
    /// riverbed a yard under the same spot does not.
    #[test]
    fn the_mirror_clips_everything_below_the_water_plane() {
        let mirror = mirror_pose(&eye_over_water(), 0.0);
        // Well inside the frustum, straight ahead of the mirrored eye.
        let above = Vec3::new(0.0, 6.0, -40.0);
        let below = Vec3::new(0.0, -6.0, -40.0);
        assert!(
            is_rendered(&mirror, 0.0, above),
            "geometry above the water must reach the mirror"
        );
        assert!(
            !is_rendered(&mirror, 0.0, below),
            "geometry below the water must not — this is the ghost reflection"
        );
    }

    /// And the plane it clips at is the one it was handed, not zero. A lake at 200 keeps its shore
    /// and drops its bed; sea level is 200 yards of world *below* that bed and must be gone too.
    #[test]
    fn the_clip_plane_follows_the_water_height() {
        let eye = Transform::from_translation(Vec3::new(0.0, 210.0, 0.0))
            .looking_to(Vec3::new(0.0, -0.5, -1.0).normalize(), Vec3::Y);
        let mirror = mirror_pose(&eye, 200.0);
        assert!(is_rendered(&mirror, 200.0, Vec3::new(0.0, 206.0, -40.0)));
        assert!(!is_rendered(&mirror, 200.0, Vec3::new(0.0, 194.0, -40.0)));
        assert!(!is_rendered(&mirror, 200.0, Vec3::new(0.0, 0.0, -40.0)));
    }

    /// A point ON the plane is the boundary case the water surface itself sits at, and it must not
    /// land on the clipped side — the shoreline where land meets water is exactly here, and losing
    /// it would put a bald ring around every lake in the mirror.
    #[test]
    fn the_waterline_itself_survives_the_clip() {
        let mirror = mirror_pose(&eye_over_water(), 0.0);
        assert!(is_rendered(&mirror, 0.0, Vec3::new(0.0, 0.0, -40.0)));
    }

    /// **The layer has to be a layer the mirror is missing, and layer 0 is the one it has.**
    /// Setting this constant to 0 would compile, would look like an ordinary layer assignment at
    /// every call site, and would silently put water and every unit name straight back into the
    /// reflection — the mirror carries no `RenderLayers` at all, which Bevy reads as layer 0.
    #[test]
    fn the_unmirrored_layer_is_not_the_layer_the_mirror_renders() {
        assert_ne!(
            UNMIRRORED_RENDER_LAYER, 0,
            "the mirror renders layer 0; the unmirrored layer must be a different one"
        );
        assert!(!RenderLayers::default().intersects(&RenderLayers::layer(UNMIRRORED_RENDER_LAYER)));
    }

    /// **The buildings ride the retained pass, and the mirror has to be a view that pass draws
    /// into** — the bug this marker closes: WMO group geometry, merged doodad cells and WMO props
    /// are not entities, so no render layer and no visibility term reaches them. They appear in
    /// the mirror if and only if `static_gx`'s own view marker is on this camera, and the failure
    /// is silent (a lake that reflects its terrain and its creatures and none of its town).
    ///
    /// Asserted through the real spawn rather than by reading the bundle, because the marker is
    /// one line in a ten-element tuple and deleting it compiles.
    #[test]
    fn the_mirror_is_a_view_the_retained_static_pass_draws_into() {
        let mut app = App::new();
        app.insert_resource(LiquidScreenTargets {
            color: Handle::default(),
            blur: Handle::default(),
            depth: Handle::default(),
            reflect: Handle::default(),
            size: UVec2::ONE,
        });
        app.world_mut().spawn((
            WorldCamera,
            Projection::from(PerspectiveProjection::default()),
            Msaa::Off,
        ));
        app.add_systems(Update, spawn_reflection_camera);
        app.update();

        let mut mirrors = app
            .world_mut()
            .query_filtered::<&crate::static_gx::StaticGxView, With<LiquidReflectionCamera>>();
        let marker = mirrors
            .iter(app.world())
            .next()
            .copied()
            .expect("the mirror camera must carry the retained pass's view marker");
        // And it must be the SECONDARY one: `prepare_static_gx` caches the four pipelines against
        // one view's `(samples, format)`, and two primaries would let that key flip per frame.
        assert!(
            !marker.primary,
            "the mirror is not the view the retained pass takes its pipeline key from"
        );
    }

    /// The layer this crate claims must not be the one app layer it can name. The rest of the
    /// ladder is asserted in `portrait`, where it lives.
    #[test]
    fn the_water_layer_is_clear_of_the_player_shadows() {
        // `benilla_app::player_shadow::PLAYER_SHADOW_LAYER` — restated rather than imported,
        // because the dependency runs the other way.
        const PLAYER_SHADOW_LAYER: usize = 31;
        assert_ne!(UNMIRRORED_RENDER_LAYER, PLAYER_SHADOW_LAYER);
    }
}
