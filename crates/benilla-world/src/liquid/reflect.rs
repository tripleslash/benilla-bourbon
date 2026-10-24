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
//! and `CMaterialWater::RenderAbove` indexes the container by
//! `depthFmt + 2·v29 + 6·dynDisplMapCnt + 2·(9·depthMode − 9)`. So mode 0 (perms 0/1) is the default arm, mode 1 (perms 2/3) is the
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
//!    plane is the real buffer's: the closest drawn liquid the eye is above
//!    ([`closest_reflection_height`]).
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

use bevy::camera::primitives::{Aabb, Frustum};
use bevy::camera::visibility::RenderLayers;
use bevy::camera::CameraProjection;
use bevy::camera::Hdr;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;

use super::screen::LiquidScreenTargets;
use super::{Underwater, WaterDetail};
use crate::view::WorldCamera;
use benilla_assets::materials::LiquidCataMaterial;

/// **The mirror plane this frame, and whether it is usable** — the reference's
/// `Liquid::GetClosestReflectionHeight`, resolved for the fake buffer too.
///
/// The reference passes `0.0` as `RenderReflection`'s plane for the fake buffer and reserves the
/// tracked closest height for the real one. Sea level is the right plane for an ocean and the wrong
/// one for everything else, and "the wrong plane" is not a subtle error: mirroring a mountain lake
/// about `z = 0` reflects a slab of world from four hundred yards below it onto the surface. So the
/// plane here is the closest drawn water the eye is above, and [`plane`] is published to
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
/// **63 is the player shadow's and 2.. is the portrait ladder's**, whose character panes grow one
/// layer each from 29; this is the one layer claimed from `benilla-world` rather than from
/// `benilla-app`'s ladder, because the camera that must not see it is spawned here. It sits at the
/// far end of `RenderLayers`' 64-layer inline block, clear of the growing panes (at 30 it was the
/// second pane's, whose camera then drew the nameplates), and it is asserted distinct from the one
/// fixed app layer this crate can see.
pub const UNMIRRORED_RENDER_LAYER: usize = 62;

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
        // One cluster, as the world camera: nothing here reads Bevy's light clusters, and the
        // default grid is a 55 KB buffer allocated afresh every frame (`view::cpu_light_clusters`).
        bevy::light::cluster::ClusterConfig::Single,
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

/// fork (perf): how far past the view distance a liquid surface still arms the mirror (yd).
///
/// A surface's transform is its origin, not its extent, and an MCLQ surface is a chunk wide — so a
/// lake whose origin sits just past the far clip can still have a corner inside it. One chunk
/// (33.3 yd) would do; this is four, because the whole point of the margin is that being wrong
/// costs a visible unreflected frame and being generous costs nothing but the frames where water is
/// nearly in range anyway.
const WATER_REACH_MARGIN: f32 = 133.0;

/// fork (perf): how long the mirror stays up after the last frame water was in reach (seconds).
const RELEASE_SECS: f32 = 1.0;

/// fork (perf): how far outside the world camera's view a liquid surface still arms the mirror
/// (radians, added to each side of the frustum). The test runs on this frame's pose, so the only
/// way to see unmirrored water is a turn wider than this in one frame.
const VIEW_MARGIN: f32 = 0.175;

/// fork (perf): how far the eye may move (yd) before the terrain sight lines to open-world water are
/// re-tested ([`WaterSight`]).
const SIGHT_MOVE_YD: f32 = 3.0;
/// fork (perf): water boxes sight-tested per frame (five sight lines each), oldest verdict first; a
/// box not yet tested counts as seen, a stale verdict stands until its box is re-tested.
const SIGHT_BOXES_PER_FRAME: usize = 9;
/// fork (perf): water this close to the eye (yd) is never hidden by the terrain test.
const SIGHT_NEAR_YD: f32 = 48.0;
/// fork (perf): the end of a sight line left unsampled (yd): a shore box's corners sit over dry bank.
const SIGHT_END_YD: f32 = 8.0;
/// fork (perf): ground must stand this far over a sight line (yd) to block it.
const SIGHT_CLEAR_YD: f32 = 1.0;

/// fork (perf): which open-world water boxes the terrain hides from the eye. The mirror is a second
/// render of the whole scene, and a lake behind a ridge, or the sea under a plateau, armed it on
/// distance and frustum alone.
#[derive(Default)]
struct WaterSight {
    /// Where the current generation of verdicts is taken from.
    eye: Option<Vec3>,
    generation: u32,
    /// Per box centre: (seen, the generation it was tested in).
    verdicts: std::collections::HashMap<[u32; 3], (bool, u32)>,
    /// This frame's boxes without a current verdict: (age in generations, key, box).
    pending: Vec<(u32, [u32; 3], (Vec3, Vec3))>,
}

/// Can the eye see any of an ADT water box's top face past the terrain? Five sight lines (the
/// centre and four inset corners); a column with no resident ground never blocks.
fn terrain_hides(
    eye: Vec3,
    (centre, half): (Vec3, Vec3),
    mut ground: impl FnMut(Vec3) -> Option<f32>,
) -> bool {
    let top = centre.y + half.y;
    let targets = [
        Vec3::new(centre.x, top, centre.z),
        Vec3::new(centre.x - half.x * 0.9, top, centre.z - half.z * 0.9),
        Vec3::new(centre.x + half.x * 0.9, top, centre.z - half.z * 0.9),
        Vec3::new(centre.x - half.x * 0.9, top, centre.z + half.z * 0.9),
        Vec3::new(centre.x + half.x * 0.9, top, centre.z + half.z * 0.9),
    ];
    targets.into_iter().all(|target| {
        let ray = target - eye;
        let len = ray.length();
        if len < SIGHT_NEAR_YD {
            return false;
        }
        let steps = (len / 8.0).clamp(4.0, 32.0) as u32;
        (1..steps).any(|k| {
            let t = k as f32 / steps as f32;
            if t * len > len - SIGHT_END_YD {
                return false;
            }
            let p = eye + ray * t;
            ground(p).is_some_and(|h| h > p.y + SIGHT_CLEAR_YD)
        })
    })
}

/// fork (perf): the world camera's frustum widened by [`VIEW_MARGIN`] on every side and cut at
/// `far`. `None` for a projection that is not perspective, which arms on distance alone.
fn mirror_frustum(pose: &GlobalTransform, projection: &Projection, far: f32) -> Option<Frustum> {
    let Projection::Perspective(p) = projection else {
        return None;
    };
    let half_v = (p.fov * 0.5 + VIEW_MARGIN).min(1.5);
    let half_h = ((p.fov * 0.5).tan() * p.aspect_ratio).atan() + VIEW_MARGIN;
    let wide = PerspectiveProjection {
        fov: 2.0 * half_v,
        aspect_ratio: half_h.min(1.5).tan() / half_v.tan(),
        far,
        ..p.clone()
    };
    Some(wide.compute_frustum(pose))
}

/// Mirror the world camera into it, and decide whether the mirror runs at all.
///
/// The conditions are the reference's own plus two: [`MIN_EYE_HEIGHT`], and — fork (perf) — that
/// there is water anywhere near enough to reflect. `HasFakeReflection()` is `waterDetail != 0`, and
/// `RenderReflection` is skipped outright while `Liquid::CameraSubmerged()` — from underneath there
/// is nothing above the surface to mirror, and `ProcWaterBelow` does not sample `samp1` in any
/// permutation.
///
/// **Why the second addition exists.** The plane is deliberately HELD when the eye leaves water
/// (see the `held` binding below) so that stepping off a shore does not move the mirror four
/// hundred yards in one frame. The cost of that kindness was never bounded: a camera in the middle
/// of a dry zone still rendered a second, half-resolution pass over the whole scene, into a texture
/// no fragment would sample, because the only liquid in the session was one the player walked past
/// ten minutes ago. That is the single largest thing in the frame with nothing at all to show for
/// it.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sync_reflection_camera(
    detail: Res<WaterDetail>,
    underwater: Res<Underwater>,
    targets: Option<Res<LiquidScreenTargets>>,
    view: Res<crate::view::ViewDistance>,
    time: Res<Time>,
    liquids: Query<
        (
            &GlobalTransform,
            Option<&Aabb>,
            &ViewVisibility,
            Option<&super::surface::LiquidLayerBoxes>,
        ),
        With<super::surface::LiquidSurface>,
    >,
    mut plane_out: ResMut<ReflectionPlane>,
    mut water_seen_at: Local<f32>,
    mut held: Local<f32>,
    streamer: Option<Res<crate::terrain_stream::TerrainStreamer>>,
    adt_tiles: Option<Res<Assets<benilla_assets::AdtTile>>>,
    mut sight: Local<WaterSight>,
    world_cam: Query<
        (&GlobalTransform, &Projection, &Camera),
        (With<WorldCamera>, Without<LiquidReflectionCamera>),
    >,
    mut mirror: Query<
        (
            &mut Transform,
            &mut Projection,
            &mut Camera,
            &mut RenderTarget,
        ),
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

    // **The plane is the closest drawn water the eye is above — not sea level**
    // ([`closest_reflection_height`]). Where no water is drawn the last plane is HELD rather than
    // snapped to zero, and the height guard in the shader fades a surface that disagrees with it.
    // Zero is only the value before any water has been drawn.
    let eye = world_pose.translation();
    // fork (perf): an ADT cell answers for each MCNK layer it draws; a WMO pool for its own box.
    let boxes = || {
        liquids.iter().flat_map(|(gt, aabb, seen, layers)| {
            let own = match layers {
                Some(_) => None,
                None => aabb.map(|a| world_box(gt, a)),
            };
            // The flag: an ADT layer, the only kind the terrain sight test may hide.
            own.into_iter()
                .map(|b| (b, false))
                .chain(
                    layers
                        .into_iter()
                        .flat_map(|l| l.0.iter().copied().map(|b| (b, true))),
                )
                .map(move |(b, adt)| (b, seen.get(), adt))
        })
    };
    let plane = match closest_reflection_height(eye, boxes().map(|(b, seen, _)| (b, seen))) {
        Some(surface) => {
            *held = surface;
            surface
        }
        None => *held,
    };

    // fork (perf): is there water near enough to be worth a second scene render?
    //
    // Not a test on last frame's visibility, which would arm the mirror one frame after the water
    // is already on screen, a visible pop. Distance first, then the frustum widened on this
    // frame's pose. `RELEASE_SECS` then keeps it a moment longer, so a shoreline at
    // the edge of either cannot toggle a whole scene pass on and off frame to frame.
    // The box, not the transform: an ADT surface's transform is IDENTITY (MCLQ is absolute), so its
    // translation is the world origin and every open-world lake read as out of reach.
    //
    // fork (perf): and inside this frame's view, widened ([`mirror_frustum`]): water off screen
    // has nothing to reflect into. The frustum is the pose this frame renders with, so a turn
    // arms the mirror before the water is on screen.
    let reach = view.farclip + WATER_REACH_MARGIN;
    let frustum = mirror_frustum(world_pose, world_projection, reach);
    // fork (perf): open-world water the terrain hides from the eye does not arm it either.
    let mut tile_cache = None;
    let mut ground = |p: Vec3| {
        let (s, t) = (streamer.as_deref()?, adt_tiles.as_deref()?);
        crate::terrain_stream::terrain_height_under_cached(s, t, p, &mut tile_cache)
    };
    // An eye under the ground (a cave, a tunnel) is not judged by it at all.
    let sight_live = ground(eye).is_none_or(|h| h <= eye.y + SIGHT_CLEAR_YD);
    if sight.eye.is_none_or(|at| at.distance(eye) > SIGHT_MOVE_YD) {
        sight.eye = Some(eye);
        sight.generation = sight.generation.wrapping_add(1);
        if sight.verdicts.len() > 4096 {
            sight.verdicts.clear();
        }
    }
    let sight = &mut *sight;
    let generation = sight.generation;
    sight.pending.clear();
    let mut near_water = false;
    for ((centre, half), _, adt) in boxes() {
        let in_view = ((eye - centre).abs() - half)
            .max(Vec3::ZERO)
            .length_squared()
            <= reach * reach
            && frustum.as_ref().is_none_or(|f| {
                f.intersects_obb_identity(&Aabb {
                    center: centre.into(),
                    half_extents: half.into(),
                })
            });
        if !in_view {
            continue;
        }
        if !adt || !sight_live {
            near_water = true;
            break;
        }
        let key = centre.to_array().map(f32::to_bits);
        match sight.verdicts.get(&key) {
            Some(&(seen, g)) if g == generation => {
                if seen {
                    near_water = true;
                    break;
                }
            }
            last => sight.pending.push((
                last.map_or(u32::MAX, |&(_, g)| generation.wrapping_sub(g)),
                key,
                (centre, half),
            )),
        }
    }
    if !near_water {
        // Oldest first, so a long walk cannot starve the boxes late in the query order.
        sight
            .pending
            .sort_unstable_by_key(|&(age, ..)| std::cmp::Reverse(age));
        for (k, &(_, key, b)) in sight.pending.iter().enumerate() {
            let seen = if k < SIGHT_BOXES_PER_FRAME {
                let seen = !terrain_hides(eye, b, &mut ground);
                sight.verdicts.insert(key, (seen, generation));
                seen
            } else {
                sight.verdicts.get(&key).is_none_or(|&(s, _)| s)
            };
            if seen {
                near_water = true;
                break;
            }
        }
    }
    if near_water {
        *water_seen_at = time.elapsed_secs();
    }
    let water_recent = near_water || time.elapsed_secs() - *water_seen_at < RELEASE_SECS;
    // fork: `waterDetail 1` (SSR only) never renders the mirror; the shader's flag stays clear and
    // it reflects through the screen-space trace over the horizon colour.
    let wanted = detail.has_planar_reflection()
        && !underwater.0.any()
        && eye.y - plane > MIN_EYE_HEIGHT
        && water_recent;
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
    camera.clear_color = world_camera.clear_color;
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

/// A surface's world-space box: centre and half extents.
fn world_box(gt: &GlobalTransform, aabb: &Aabb) -> (Vec3, Vec3) {
    let m = gt.affine();
    let centre = m.transform_point3a(aabb.center);
    let half = m.matrix3.abs() * aabb.half_extents;
    (centre.into(), half.into())
}

/// `Liquid::RendererDraw`'s tracked instance: of the liquid drawn this frame, the one whose centre is
/// closest to the camera and whose plane the camera is above; its height is the mirror's plane
/// (`GetClosestReflectionHeight`). `None` while no such water is drawn.
///
/// Not "the water under the eye": a camera on the shore of Lake Nazferiti is over grass, so that rule
/// held a stale plane and the shader's height guard faded the whole lake to the trace.
fn closest_reflection_height(
    eye: Vec3,
    surfaces: impl Iterator<Item = ((Vec3, Vec3), bool)>,
) -> Option<f32> {
    surfaces
        .filter(|((centre, _), drawn)| *drawn && eye.y > centre.y)
        .map(|((centre, _), _)| (centre.distance_squared(eye), centre.y))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, height)| height)
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

/// fork (perf): a doodad whose world bounds reach less than this from their centre (yd) is left out
/// of the mirror. A barrel, a crate, a bush or a lamp post is a few pixels in a rippled reflection;
/// a tree, a rock or a building is what the water shows, and stays.
const MIRROR_MIN_DOODAD_RADIUS: f32 = 2.5;

/// fork (perf): `$WOW_MIRROR_THIN=0` (`?mirror_thin=0`) keeps everything in the mirror, the A/B
/// lever for [`thin_mirror_view`].
fn mirror_thinning() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MIRROR_THIN").as_deref() != Some("0"))
}

/// fork (perf): `$WOW_MIRROR_PLANE_CULL=0` (`?mirror_plane_cull=0`) keeps what lies under the water
/// in the mirror's list, the A/B lever for the plane test in [`thin_mirror_view`].
fn mirror_plane_cull() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MIRROR_PLANE_CULL").as_deref() != Some("0"))
}

/// fork (perf): how far below the mirror plane an entity's bounds must top out (yd) before the
/// mirror drops it — float slack on a test that is otherwise exact.
const MIRROR_PLANE_SLACK: f32 = 0.05;

/// What the mirror's visible list keeps of one entity: `kind` its [`ModelPart`] kind (a twin's is
/// its source part's), `radius` its world bounds' half diagonal.
///
/// [`ModelPart`]: crate::model_render::ModelPart
fn kept_in_mirror(
    clutter: bool,
    kind: Option<crate::model_render::ModelKind>,
    radius: f32,
) -> bool {
    use crate::model_render::ModelKind;
    if clutter {
        return false;
    }
    match kind {
        Some(ModelKind::Creature) => false,
        // A transport's hull is a WMO-display GameObject, so the kind alone cannot drop it.
        Some(ModelKind::Doodad | ModelKind::GameObject) => radius >= MIRROR_MIN_DOODAD_RADIUS,
        Some(ModelKind::Wmo) | None => true,
    }
}

/// fork (perf): leave ground clutter, units, and small GameObjects and doodads out of the mirror.
///
/// The mirror re-queues and re-encodes every entity its frustum holds, and at a waterside town that
/// second pass was ~4 ms of render-thread CPU a frame (Auberdine, 2026-10-18), which the browser
/// pays on the one thread it draws from. What it holds that the rippled, fogged reflection cannot
/// show is the small and the moving: grass tufts, crates, the people on the dock. They are removed
/// from the mirror camera's own visible list after Bevy's check, and from nothing else: the world
/// camera, the shadow casters and every other reader of the entity are unchanged. Buildings, trees
/// and the terrain are in the retained pass or large, and still reflect.
///
/// **And whatever lies wholly under the water.** The mirror's oblique near plane discards every
/// fragment below the plane, but its frustum test does not know that, so the seabed, the shore's
/// drowned shelf and anything standing in deep water were queued, sorted and encoded for a view that
/// clips all of it. An entity whose world bounds top out below the plane is dropped, which changes
/// no pixel.
///
/// Removing entries keeps each per-class list sorted, which Bevy's visibility diff requires.
#[allow(clippy::type_complexity)]
fn thin_mirror_view(
    mut mirror: Query<&mut bevy::camera::visibility::VisibleEntities, With<LiquidReflectionCamera>>,
    plane: Res<ReflectionPlane>,
    parts: Query<(
        Has<crate::clutter::ClutterMesh>,
        Option<&crate::model_render::ModelPart>,
        Option<&crate::straddle::StraddleTwinOf>,
        Option<&crate::zfill::ZfillTwinOf>,
        Option<&Aabb>,
        &GlobalTransform,
    )>,
    sources: Query<&crate::model_render::ModelPart>,
) {
    let (thin, below) = (
        mirror_thinning(),
        mirror_plane_cull().then(|| plane.live_plane()).flatten(),
    );
    if !thin && below.is_none() {
        return;
    }
    for mut visible in &mut mirror {
        for list in visible.entities.values_mut() {
            list.retain(|&e| {
                let Ok((clutter, part, straddle, zfill, aabb, gt)) = parts.get(e) else {
                    return true;
                };
                if let (Some(plane), Some(b)) = (below, aabb) {
                    if under_plane(gt, b, plane) {
                        return false;
                    }
                }
                if !thin {
                    return true;
                }
                let kind = part.map(|p| p.kind).or_else(|| {
                    let source = straddle.map(|t| t.0).or(zfill.map(|t| t.0))?;
                    sources.get(source).ok().map(|p| p.kind)
                });
                let radius = aabb.map_or(f32::INFINITY, |b| {
                    (gt.affine().matrix3 * Vec3::from(b.half_extents)).length()
                });
                kept_in_mirror(clutter, kind, radius)
            });
        }
    }
}

/// Do these local bounds, placed by `gt`, top out below the mirror plane (world Y)?
fn under_plane(gt: &GlobalTransform, b: &Aabb, plane: f32) -> bool {
    let m = gt.affine();
    let centre = m.transform_point3a(b.center);
    let half_y = m.matrix3.row(1).abs().dot(b.half_extents);
    centre.y + half_y < plane - MIRROR_PLANE_SLACK
}

/// Wire the mirror into the app. Called from [`super::LiquidPlugin`].
pub(super) fn register(app: &mut App) {
    app.init_resource::<ReflectionPlane>()
        .add_systems(
            PostUpdate,
            thin_mirror_view.after(bevy::camera::visibility::VisibilitySystems::CheckVisibility),
        )
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
    /// fork (perf): bounds that top out under the plane leave the mirror; bounds that reach above
    /// it stay, scaled and rotated placements included.
    #[test]
    fn what_lies_wholly_under_the_water_leaves_the_mirror() {
        use super::under_plane;
        use bevy::camera::primitives::Aabb;
        use bevy::prelude::*;
        let b = Aabb::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let at = |y: f32| GlobalTransform::from(Transform::from_xyz(0.0, y, 0.0));
        assert!(under_plane(&at(-2.0), &b, 0.0), "a seabed chunk");
        assert!(
            !under_plane(&at(-0.5), &b, 0.0),
            "a rock breaking the surface"
        );
        let tipped = GlobalTransform::from(
            Transform::from_xyz(0.0, -2.5, 0.0)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4))
                .with_scale(Vec3::splat(2.0)),
        );
        // Tipped 45° and doubled, its top is 2·√2 ≈ 2.83 above its centre: above the plane.
        assert!(!under_plane(&tipped, &b, 0.0));
    }

    /// fork (perf): the mirror keeps the terrain, WMOs and large doodads, and drops clutter, units,
    /// GameObjects and small doodads.
    #[test]
    fn the_mirror_keeps_the_large_and_drops_the_small_and_the_moving() {
        use super::{kept_in_mirror, MIRROR_MIN_DOODAD_RADIUS};
        use crate::model_render::ModelKind;
        assert!(
            kept_in_mirror(false, None, 1.0),
            "terrain and unclassified meshes"
        );
        assert!(kept_in_mirror(false, Some(ModelKind::Wmo), 0.5));
        assert!(kept_in_mirror(
            false,
            Some(ModelKind::Doodad),
            MIRROR_MIN_DOODAD_RADIUS
        ));
        assert!(
            !kept_in_mirror(false, Some(ModelKind::Doodad), 1.0),
            "a crate"
        );
        assert!(!kept_in_mirror(false, Some(ModelKind::Creature), 50.0));
        assert!(
            kept_in_mirror(false, Some(ModelKind::GameObject), 50.0),
            "a transport's hull"
        );
        assert!(
            !kept_in_mirror(false, Some(ModelKind::GameObject), 1.0),
            "a chest"
        );
        assert!(!kept_in_mirror(true, None, f32::INFINITY), "clutter");
    }

    /// fork (perf): the mirror's water-reach margin must clear one MCLQ surface's own extent.
    ///
    /// A `LiquidSurface`'s transform is its ORIGIN and an MCLQ surface spans a chunk, so a lake
    /// whose origin sits just past the far clip can still have a corner inside the frame. A margin
    /// under one chunk would stand the mirror down on water the player can see, which is the one
    /// way this optimization is allowed to be wrong and must not be.
    #[test]
    fn the_water_reach_margin_clears_a_whole_chunk() {
        const {
            assert!(
                super::WATER_REACH_MARGIN >= benilla_formats::CHUNK_SIZE,
                "a surface's origin can be a chunk away from the part of it that is on screen"
            );
            assert!(
                super::RELEASE_SECS > 0.0,
                "without a release delay a shoreline at the edge of the radius toggles a whole \
                 scene pass on and off frame to frame"
            );
        }
    }

    use super::*;
    use bevy::camera::CameraProjection;
    use bevy::render::view::Msaa;

    /// fork (perf): the viewer's 70° camera at 16:9, looking down -Z from the origin.
    fn frustum_ahead() -> Frustum {
        let pose = GlobalTransform::from(Transform::IDENTITY);
        let projection = Projection::Perspective(PerspectiveProjection {
            fov: 1.221_730_5,
            aspect_ratio: 16.0 / 9.0,
            ..default()
        });
        mirror_frustum(&pose, &projection, 1133.0).unwrap()
    }

    /// Does a one-chunk water box `dist` yards out at `deg` degrees right of the view arm it?
    fn arms(deg: f32, dist: f32) -> bool {
        let (s, c) = deg.to_radians().sin_cos();
        frustum_ahead().intersects_obb_identity(&Aabb {
            center: Vec3A::new(s * dist, 0.0, -c * dist),
            half_extents: Vec3A::new(16.6, 0.5, 16.6),
        })
    }

    #[test]
    fn water_ahead_arms_the_mirror_and_water_behind_does_not() {
        assert!(arms(0.0, 400.0));
        assert!(!arms(180.0, 400.0));
        assert!(!arms(0.0, 1400.0), "past the reach");
    }

    #[test]
    fn water_just_off_screen_still_arms_the_mirror() {
        // The frame's edge is ~51° right of axis at 16:9; the margin carries ten more.
        assert!(arms(58.0, 400.0));
        assert!(!arms(80.0, 400.0));
    }

    #[test]
    fn water_around_the_eye_arms_the_mirror_whatever_the_heading() {
        assert!(arms(180.0, 5.0));
    }

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
        let clip =
            projection.get_clip_from_view() * pose.to_matrix().inverse() * world_point.extend(1.0);
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

    /// **The reported bug: a camera on a lake's shore got no mirror.** It is over grass, so the
    /// plane must come from the closest drawn water it is above — not from what is under it, and
    /// never from water it is not above or that is not drawn.
    #[test]
    fn the_plane_is_the_closest_drawn_water_the_eye_is_above() {
        let eye = Vec3::new(0.0, 40.0, 0.0);
        let flat = |x: f32, y: f32| (Vec3::new(x, y, 0.0), Vec3::new(16.7, 0.0, 16.7));
        let surfaces = [
            (flat(30.0, 33.0), true), // the lake beside the shore
            (flat(90.0, 0.0), true),  // the sea, farther out
            (flat(10.0, 60.0), true), // a pool on a cliff above the eye
            (flat(5.0, 38.0), false), // culled this frame
        ];
        assert_eq!(
            closest_reflection_height(eye, surfaces.into_iter()),
            Some(33.0)
        );
        assert_eq!(closest_reflection_height(eye, std::iter::empty()), None);
    }

    /// An ADT surface's transform is IDENTITY (MCLQ is absolute), so its world box is its mesh box
    /// — not a point at the world origin.
    #[test]
    fn an_identity_surface_is_placed_by_its_mesh_box() {
        let aabb = Aabb::from_min_max(
            Vec3::new(9000.0, 30.0, -200.0),
            Vec3::new(9033.3, 31.0, -166.7),
        );
        let (centre, half) = world_box(&GlobalTransform::IDENTITY, &aabb);
        assert!((centre - Vec3::new(9016.65, 30.5, -183.35)).length() < 1e-2);
        assert!((half - Vec3::new(16.65, 0.5, 16.65)).length() < 1e-2);
        let turned = GlobalTransform::from(Transform::from_rotation(Quat::from_rotation_y(
            std::f32::consts::FRAC_PI_2,
        )));
        let (_, half) = world_box(
            &turned,
            &Aabb::from_min_max(Vec3::ZERO, Vec3::new(4.0, 1.0, 2.0)),
        );
        assert!((half - Vec3::new(1.0, 0.5, 2.0)).length() < 1e-4);
    }

    /// The layer this crate claims must not be the one app layer it can name. The rest of the
    /// ladder is asserted in `portrait`, where it lives.
    #[test]
    fn the_water_layer_is_clear_of_the_player_shadows() {
        // `benilla_app::player_shadow::PLAYER_SHADOW_LAYER` — restated rather than imported,
        // because the dependency runs the other way.
        const PLAYER_SHADOW_LAYER: usize = 63;
        assert_ne!(UNMIRRORED_RENDER_LAYER, PLAYER_SHADOW_LAYER);
    }
}
