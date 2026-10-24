//! WMO portal culling: each frame, which of a building's groups are reachable through portals from
//! the camera's current group.
//!
//! The 1.12 client's recursion (`0x6b41c0`) floods the portal graph from the camera's group with a
//! screen rect as its frustum: per portal it tests that the camera is on the front side, projects
//! the portal to a screen rect, intersects that with the carried rect and recurses with the result,
//! ending a branch when the rect collapses. That is why the Stormwind cathedral culls from the
//! Trade District but draws from the gates.
//!
//! From inside, exterior shells draw only through deferred windows, doorways the flood reached onto
//! an `0x8` group (none skips the pass, `0x6b3c87`): per window, every `0x8` group inside a frustum
//! clipped to that doorway becomes a flood root (`0x6b3d39 call 0x6b41c0`), carrying its rect on.
//!
//! This module computes each group's visible bit ([`WmoPortalInstance::visible`]); the one
//! `Visibility` authority, `crate::model_render`'s `apply_model_visibility`, ANDs it with the rest.
//! The seed is the down-ray in [`seed`]. The epsilons here (the side test at exactly `0.0`, rect
//! collapse `0.001`, the `w`-clamp pair, the `0.1` near-parallel snap) are the client's own: do
//! not loosen them into tolerances.

use std::sync::Arc;

mod fog;
mod interior;
mod probe;
mod room_vis;
mod seed;

use fog::select_wmo_fog;
pub use fog::{CameraRoomExtent, CameraWmoFog, WmoFogTarget};
pub use interior::{
    indoor_verdict_at, indoors_at, terrain_z_local, IndoorVerdict, LightAttach,
    INTERIOR_PROBE_HEIGHT,
};
pub(crate) use interior::{surface_terrain_sample, WmoAreas, POSITION_PROBE_LIFT};
use interior::{track_area_interior, track_current_interior, track_unit_interiors};
pub use interior::{
    CurrentAreaInterior, CurrentWmoInterior, EntityOutdoors, PlayerWmoRoom, UnitWmoRoom,
    WmoInteriorKeys,
};
use probe::TraceLog;
pub use probe::WmoCullProbe;
pub use room_vis::room_pvs_visible;
use seed::dominant_axes;
pub use seed::{down_ray_seeds, floor_z_at, DownRaySeeds};

use crate::terrain_stream::{terrain_height_under, TerrainStreamer};
use crate::view::WorldCamera;
use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};
use benilla_assets::DataLoadAppExt;
use benilla_assets::{AdtTile, WmoGroupNav, WmoModel, WmoPortalInfo};
use bevy::camera::primitives::{Aabb, Frustum};
use bevy::camera::Projection;
use bevy::math::{Affine3A, Vec4};
use bevy::prelude::*;

/// MOGP EXTERIOR, an outdoor group: it seeds the outside leg, and a portal onto one opens a window.
const EXTERIOR: u32 = 0x8;

/// MOGP EXTERIOR_LIT, an interior-graph group lit as outdoors: the unit light classify (`0x6a87f0`)
/// takes the exterior leg on `MOGI & 0x48`, the zone-text bit on `0x8` alone. Stormwind's streets
/// (115 of 306 groups) and Orgrimmar's valleys are `0x40` without `0x8`.
const EXTERIOR_LIT: u32 = 0x40;

/// Eye-on-portal-plane band (yd): an eye this close to a portal's plane and inside its polygon gets
/// the full-screen rect (`0x6b46f0`, `|d| <= 0.01` at `0x8029d0`). The side test still runs: the
/// recorded skip flag is dead code (`0x6b43c9`).
const ON_PLANE_EPS: f32 = 0.01;
/// The `|w|` band below which a clip-space vertex's `w` is substituted before the divide
/// (`0x801360`, `0.001`, strict `<`).
const W_CLAMP_BAND: f32 = 0.001;
/// The substituted `w`, positive whatever the vertex's sign (the `1e-5` immediate `0x3727c5ac`); a
/// vertex with `w <= -0.001` divides by its real negative `w`.
const W_CLAMP_SUB: f32 = 1.0e-5;
/// Minimum NDC extent of a narrowed rect (`0x801360`, `0.001`); a collapse kills the branch.
const RECT_EPS: f32 = 0.001;
/// Near-parallel threshold on the ray-plane denominator (the f64 `0x811658`, `0.0001`): below it
/// the down-ray crosses a portal only through the snap.
const PORTAL_NEAR_PARALLEL: f32 = 1.0e-4;
/// The snap window (yd) for a near-parallel portal (the `0.1` pushed at `0x6a40c0`): a vertical
/// doorway counts as crossed only while the eye is this close to its plane.
const PORTAL_PLANE_SNAP: f32 = 0.1;
/// Nearest-crossing accept for the portal leg (`0x6be421`-`0x6be434`): a crossing whose fraction
/// is less than `0x80c4f4`'s `1e-4` past the best hit so far still wins, so a floor-hole portal
/// flips the room. The fraction is of the 1760 yd ray ([`MAX_FLOOR_DROP`]): about 0.176 yd.
const NEAREST_TIE_YD: f32 = 1.0e-4 * MAX_FLOOR_DROP;
/// MOGI/MOGP `0x10000`, the callback-pass group: drawn in a third pass (`0x6b3d6f..0x6b3dbc`)
/// against the full camera frustum, gated on nothing, and skipped by Pass 2 (`0x6b3d0b`). Both
/// passes read MOGI; [`WmoGroupNav::flags`] is MOGP, which differs only by
/// `0x1|0x4|0x200|0x800|0x1000` in all 5219 shipped groups (the loader `0x6c4530`), so the two
/// agree here and on [`EXTERIOR`]. 24 groups carry it: 14 in `stormwind.wmo`, 6 in Dire Maul and
/// the 4 Razorfen enclosing shells.
const CALLBACK_PASS: u32 = 0x10000;
/// The client's deferred-window worklist holds a fixed 16 records with no bound check
/// (`0x6b37b0`). Shipped data never nears it, so a flood past it is flagged as our bug.
const WORKLIST_CAP: usize = 16;
/// Recursion depth cap, the client's own: `[0xcb004c]` seeded from `[0x86b6b8]` = 10 at every
/// depth-0 entry, so each Pass-2 root gets ten fresh hops. A cycle backstop, not a visibility term.
const DEPTH_CAP: u32 = 10;
/// Hard cap on flood iterations, a runaway-graph backstop; real WMOs settle far below it.
const MAX_ITERS: u32 = 1 << 16;

/// An NDC rectangle `[min_x, min_y, max_x, max_y]`, the screen frustum the flood carries.
pub(crate) type Rect = [f32; 4];
const FULL_SCREEN: Rect = [-1.0, -1.0, 1.0, 1.0];

/// The deferred exterior-window worklist: the doorways the outdoor world may draw through this
/// frame, as NDC rects. The client's `[0xcbe324]` list (count `[0xcbe320]`), which the scene
/// driver `0x681070` copies to `0xc7cb7c` and walks once per window, the frustum narrowed to it;
/// `0x682fa0` is the only exterior producer, so no window means no exterior content (`0x681199`).
#[derive(Resource, Default, Clone, PartialEq, Debug)]
pub enum ExteriorWindows {
    /// The driver's outside leg (`0x6811ca`): no containing WMO, the whole frustum is the window.
    #[default]
    Unrestricted,
    /// The camera is inside: exterior content draws only where it survives one of these, and an
    /// empty list, the sealed room, draws none.
    Windows(Vec<Rect>),
}

/// Each exterior window's portal reach (`0x6b4680`), the fifth float of its record: the farthest
/// view depth of the portal's vertices, never below 0. The scene driver feeds the horizon from it
/// plus 33.333332 (`0x6811a9`), so terrain short of the doorway never occludes through it. Parallel
/// to [`ExteriorWindows::Windows`], empty outdoors.
#[derive(Resource, Default, Clone, PartialEq, Debug)]
pub struct ExteriorWindowReach(pub Vec<f32>);

/// Tag on every portal-culled WMO piece (a group submesh, a group's MLIQ surface, a MODD prop): its
/// placement instance and the absolute groups that can draw it. Geometry and liquid belong to one
/// group; a prop can be named by many, as the reference admits a prop from any visited group whose
/// MODR names it. Built once per group or prop at spawn.
#[derive(Component, Clone)]
pub struct WmoGroupVis {
    pub(crate) instance: Entity,
    pub(crate) groups: Arc<[u16]>,
}

impl WmoGroupVis {
    /// Drawn iff any referencing group is in this frame's PVS; an index past the visible set reads
    /// visible, so a lookup miss never blanks a building.
    pub fn drawn_by(&self, inst: &WmoPortalInstance) -> bool {
        self.groups
            .iter()
            .any(|&g| inst.visible.get(g as usize).copied().unwrap_or(true))
    }

    /// Takes the building's MFOG triple iff any naming group does (the client's `[0xca7f00]`,
    /// [`WmoPortalInstance::interior_fog`]). A miss reads false, unlike `drawn_by`: a missing bit
    /// must not paint a room's fog onto the open world.
    /// The LIGHTING verdict: lit iff any referencing group is reachable through portals from the
    /// camera's room ([`WmoPortalInstance::lit`]). The twin of [`Self::drawn_by`], on the same
    /// fail-open, and deliberately NOT the same set — see that field.
    pub fn lit_by(&self, inst: &WmoPortalInstance) -> bool {
        self.groups
            .iter()
            .any(|&g| inst.lit.get(g as usize).copied().unwrap_or(true))
    }

    pub fn interior_fogged_by(&self, inst: &WmoPortalInstance) -> bool {
        self.groups.iter().any(|&g| inst.fogs_group(g))
    }
}

/// Is a room-bound rider drawn this frame? [`WmoGroupVis::drawn_by`] for lanes that carry their
/// rooms by value: a prop's particles and ribbons
/// ([`crate::particles::EmitterFade::room_admitted`]) and a building's point lights
/// ([`crate::lighting::LightRooms`]). The reference admits a WMO prop only in frames the portal
/// walk visits its group (`0x6838f0`, from the visit callback `0x685d70`), so a prop in a culled
/// room is neither drawn, animated nor particle-ticked.
///
/// - No rooms: admitted (an ADT doodad, a creature, a held item).
/// - Rooms, placement resident: `drawn_by`, the predicate the owner's submeshes are culled by.
/// - Rooms, placement gone: refused, where `drawn_by` fails open. A rider outlives its building by
///   a frame, and a stray bright strip or torch is an artefact where a missing one is nothing.
pub fn room_admits(room: Option<&WmoGroupVis>, instance: Option<&WmoPortalInstance>) -> bool {
    match (room, instance) {
        (None, _) => true,
        (Some(r), Some(inst)) => r.drawn_by(inst),
        (Some(_), None) => false,
    }
}

/// [`room_admits`] for a LIGHT. Same shape, different set: a light is admitted by portal
/// reachability ([`WmoPortalInstance::lit`]) rather than by this frame's frustum-clipped cull, so
/// tilting the camera cannot switch a lamp on and off.
pub fn room_admits_light(room: Option<&WmoGroupVis>, instance: Option<&WmoPortalInstance>) -> bool {
    match (room, instance) {
        (None, _) => true,
        (Some(r), Some(inst)) => r.lit_by(inst),
        (Some(_), None) => false,
    }
}

/// **Is a room-bound light genuinely INDOORS?** — the light's own half of the surface lanes'
/// indoor/outdoor class test, answered from the groups that name it.
///
/// Not the same question as "does it name rooms". A sconce bolted to the outside of Northshire
/// Abbey names rooms too: it belongs to the building's `0x8` shell. Packing `w` from
/// `rooms.is_some()` therefore read "is a WMO light", and a class test built on that would put
/// every porch lamp and courtyard brazier in the game out the moment it tried to reject indoor
/// light on outdoor ground.
///
/// The discriminator is the one the render path already uses for the receiver
/// (`RenderSubmesh::interior`, the client's `groupFlags & 0x48` branch at `0x6b3f90`): a group is a
/// true interior with neither `0x8` nor `0x40` set. ALL naming groups must be interior for the
/// light to count as indoor — a fixture a courtyard also names can legitimately light that
/// courtyard, and the error directions are lopsided (1276): wrongly "outdoor" leaves a light doing
/// what it does today, wrongly "indoor" switches a working lamp off.
pub fn rooms_are_indoor(vis: &WmoGroupVis, model: &WmoModel) -> bool {
    !vis.groups.is_empty()
        && vis.groups.iter().all(|&g| {
            model
                .group_nav
                .get(g as usize)
                .is_some_and(|n| n.flags & (EXTERIOR | EXTERIOR_LIT) == 0)
        })
}

/// One placed WMO's portal-cull state, spawned with its geometry and despawned with the placement.
#[derive(Component)]
pub struct WmoPortalInstance {
    /// The building asset (portal graph and per-group nav).
    pub handle: Handle<WmoModel>,
    /// Placement transform, Bevy model→world: a WoW-space portal vertex reaches world as
    /// `world_from_local * wow_to_bevy(v)`, the path the rendered geometry takes.
    pub world_from_local: Affine3A,
    /// The placement's `MODF.nameSet`, the `WMOAreaTable.NameSetID` variant (one abbey model
    /// serves Northshire and Tyr's Hand by name set).
    pub name_set: u16,
    /// Per-group PVS bit this frame, by absolute group index; seeded all-true, which a portal-less
    /// prop keeps.
    pub visible: Vec<bool>,
    /// **Per-group LIGHTING reachability** — which rooms this frame's camera can be lit from.
    ///
    /// Not [`Self::visible`], and the difference is the whole point. `visible` is the reference's
    /// portal cull, and that cull clips the frustum to each doorway: a room drops out of it the
    /// moment the doorway leaves the screen. Gating LIGHTS on it made a lamp's light blink as the
    /// player tilted the camera or stepped back — Northshire Abbey's entrance lamp, which sits
    /// right beside the portal that decides it.
    ///
    /// This is the same graph walked without the frustum: which groups are reachable through
    /// portals from the room the camera stands in, whatever is on screen. Turning the camera cannot
    /// change it, which is exactly the property a light needs. Seeded all-`true`, like `visible`.
    pub lit: Vec<bool>,
    /// Per-group interior-fog gate this frame (the client's `[0xca7f00]`, see [`GroupPvs`]): `true`
    /// draws the group and its doodads with the building's MFOG triple, `false` with the scene fog.
    /// Seeded all-false, which a portal-less prop keeps.
    pub interior_fog: Vec<bool>,
    /// Per-group ever-visited latch: once the flood visits a group, its MLIQ surface draws for the
    /// rest of the placement's residency, whatever this frame's PVS. Latched by
    /// [`compute_wmo_pvs`]; starts all-false.
    pub liquid_visited: Vec<bool>,
    /// Per-group whole-group submersion from MOGP `groupLiquid`: `Some(kind)` means anywhere in the
    /// group is under liquid, at every z. Baked at spawn; empty on a placement spawned before its
    /// model resolved.
    pub flooded: Vec<Option<benilla_formats::LiquidKind>>,
}

impl WmoPortalInstance {
    /// The `[0xca7f00]` bit for one group, the grain every consumer asks at: a WMO piece ORs it
    /// over its groups, a unit asks it of the room its light node attached to. A group past the set
    /// reads false.
    pub fn fogs_group(&self, group: u16) -> bool {
        self.interior_fog
            .get(group as usize)
            .copied()
            .unwrap_or(false)
    }
}

/// One WMO room: a placement's instance entity and an absolute group index in it, the client's
/// `[0xc7b748]` (the camera's containing placed map object) with the group its containing-group
/// set `0xc7cd88` carries. It is also the liquid query's scope key: a liquid footprint has no
/// floor, so an unowned pool would claim every position under it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WmoRoom {
    /// The placement's [`WmoPortalInstance`] entity.
    pub instance: Entity,
    /// Absolute group index within the building.
    pub group: u16,
}

/// The camera's interior claim this frame: `None` over the open world or an EXTERIOR (`0x8`)
/// group's floor, while an EXTERIOR_LIT-only porch still claims (the `[0xc7b748]` writer rejects on
/// `0x8` alone, `0x6be451`). Written by [`compute_wmo_pvs`] from the flood's own seed; the
/// reference's environment probe `0x6809c0` picks the WMO group's MLIQ over the ADT liquid on it.
#[derive(Resource, Default, Clone, Copy, PartialEq)]
pub struct CameraInteriorClaim(pub Option<InteriorClaim>);

/// See [`CameraInteriorClaim`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InteriorClaim {
    /// The room the camera eye is in: the placement and its current group.
    pub room: WmoRoom,
    /// A group with any of `0x148` is in the placement's PVS: the weather flag `[0xca80c4]`
    /// (`0x6b42d9`), so rain shows through a doorway the camera can see.
    pub(crate) exterior_visible: bool,
}

/// Ordering handle so the `Visibility` authority runs after this frame's PVS.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WmoPvsSet;

/// Registers the per-frame PVS compute; the apply side is `crate::model_render`'s.
pub struct WmoPortalPlugin;

impl Plugin for WmoPortalPlugin {
    fn build(&self, app: &mut App) {
        app.add_data_loaders(interior::load_wmo_areas)
            .init_resource::<CurrentWmoInterior>()
            .init_resource::<PlayerWmoRoom>()
            .init_resource::<CurrentAreaInterior>()
            .init_resource::<CameraWmoFog>()
            .init_resource::<CameraRoomExtent>()
            .init_resource::<CameraInteriorClaim>()
            .init_resource::<ExteriorWindows>()
            .init_resource::<ExteriorWindowReach>()
            .init_resource::<WmoCullProbe>()
            .add_systems(
                Update,
                (
                    compute_wmo_pvs,
                    track_current_interior,
                    track_area_interior,
                    track_unit_interiors,
                )
                    .in_set(WmoPvsSet)
                    // On this frame's camera pose: the flood, the claim and the windows key on the eye
                    // and decide what the frame may draw.
                    .after(crate::view::CameraPoseSet),
            );
    }
}

/// Max drop (yd) for the current-group down-ray, the client's ray length: a floor further below
/// is not the room you are in.
const MAX_FLOOR_DROP: f32 = 1760.0;

/// Recompute each resident WMO's per-group visible set from the camera; a portal-less prop stays
/// all-visible.
fn compute_wmo_pvs(
    wmos: Res<Assets<WmoModel>>,
    cam: Query<(&GlobalTransform, &Projection), With<WorldCamera>>,
    mut instances: Query<(Entity, &mut WmoPortalInstance)>,
    streamer: Res<TerrainStreamer>,
    adt_tiles: Res<Assets<AdtTile>>,
    mut probe: ResMut<WmoCullProbe>,
    mut camera_fog: ResMut<CameraWmoFog>,
    mut camera_room: ResMut<CameraRoomExtent>,
    mut camera_claim: ResMut<CameraInteriorClaim>,
    mut camera_windows: ResMut<ExteriorWindows>,
    mut camera_reach: ResMut<ExteriorWindowReach>,
) {
    // No world camera yet: keep last frame's sets, all visible.
    let Some((cam_t, proj)) = cam.iter().next() else {
        return;
    };
    let clip_from_world = proj.get_clip_from_view() * cam_t.to_matrix().inverse();
    let eye_world = cam_t.translation();
    probe.eye = eye_world;
    // The seed's terrain leg, sampled once for the camera's column (`0x6821f0`).
    let terrain = terrain_height_under(&streamer, &adt_tiles, eye_world);
    // `WOW_CULLDUMP=<path>` re-requests the dump every frame, last writer wins: a headless run has
    // no panel button, and a one-shot request at startup would catch an empty world.
    static ENV_DUMP: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let dump_to = probe.dump_to.take().or_else(|| {
        ENV_DUMP
            .get_or_init(|| std::env::var_os("WOW_CULLDUMP").map(std::path::PathBuf::from))
            .clone()
    });
    // The full camera pose: replaying a dump needs forward, fov and aspect, not just the eye.
    let mut dump_text = dump_to.is_some().then(|| {
        let fwd = cam_t.forward();
        let (fovy, aspect) = match proj {
            Projection::Perspective(p) => (p.fov, p.aspect_ratio),
            _ => (f32::NAN, f32::NAN),
        };
        format!("eye world: {eye_world:?}\nforward: {fwd:?} fovy {fovy:.4} aspect {aspect:.4}\n")
    });

    // The fog target and the claim come from the flood's own seed; the first placement whose seed
    // claims a non-exterior group wins both.
    let mut fog_target: Option<WmoFogTarget> = None;
    // fork: modern lighting — the claiming room's world sphere, beside its fog (see
    // [`CameraRoomExtent`]).
    let mut room_extent: Option<(Vec3, f32)> = None;
    let mut claim: Option<InteriorClaim> = None;
    // `None` is the driver's outside leg: no containing map object, one full-screen walk.
    let mut windows: Option<(Vec<Rect>, Vec<f32>)> = None;
    for (entity, mut inst) in &mut instances {
        let Some(model) = wmos.get(&inst.handle) else {
            continue;
        };
        let groups = model.group_nav.len();
        // A WMO with no portal graph (single-group props, doors) is never culled, and never on the
        // interior-fog lane. Written through the change gate: marking every prop changed each frame
        // would defeat `Changed<WmoPortalInstance>`.
        if model.portal_refs.is_empty() || model.portal_infos.is_empty() {
            let held = inst.bypass_change_detection();
            let mut changed = false;
            if held.visible.len() != groups || held.visible.iter().any(|v| !*v) {
                held.visible = vec![true; groups];
                changed = true;
            }
            if held.lit.len() != groups || held.lit.iter().any(|v| !*v) {
                held.lit = vec![true; groups];
                changed = true;
            }
            if held.interior_fog.len() != groups || held.interior_fog.iter().any(|v| *v) {
                held.interior_fog = vec![false; groups];
                changed = true;
            }
            if changed {
                inst.set_changed();
            }
            continue;
        }
        let world_from_local = inst.world_from_local;
        // The eye in model space (WoW axes): the down-ray origin and the front-side test point.
        let local_from_world = world_from_local.inverse();
        let eye_local = bevy_to_wow(local_from_world.transform_point3(eye_world));
        let terrain_local = terrain.map(|z| terrain_z_local(&local_from_world, eye_world, z));
        // One flood with every recorder attached: the dump must never change the answer it records.
        let mut tap = SeedTap::default();
        let mut log = dump_text
            .is_some()
            .then(|| TraceLog::new(model, eye_local, terrain_local));
        // Compare, then write: a parked camera's flood repeats its answer, and a blind write would
        // mark every instance changed.
        let mut fresh = compute_pvs_traced(
            model,
            eye_local,
            terrain_local,
            &clip_from_world,
            &world_from_local,
            &mut (&mut tap, &mut log),
        );
        let lit = lighting_reach(model, eye_local, terrain_local);
        if inst.bypass_change_detection().lit != lit {
            inst.bypass_change_detection().lit = lit;
            inst.set_changed();
        }
        if inst.bypass_change_detection().visible != fresh.visible {
            inst.bypass_change_detection().visible = fresh.visible;
            inst.set_changed();
        }
        if inst.bypass_change_detection().interior_fog != fresh.interior_fog {
            inst.bypass_change_detection().interior_fog = fresh.interior_fog;
            inst.set_changed();
        }
        if let (Some(text), Some(log)) = (&mut dump_text, &log) {
            // The origin tells apart two placements of one model in the dump.
            let o = world_from_local.translation;
            text.push_str(&format!(
                "placement @ world ({:.1}, {:.1}, {:.1}) — {} deferred window(s) {:?}, {} walk steps\n",
                o.x,
                o.y,
                o.z,
                fresh.windows.len(),
                fresh.windows,
                fresh.iters
            ));
            text.push_str(&log.text);
            text.push_str(&format!("visible: {:?}\n\n", inst.visible));
        }
        // The claim rejects on `0x8` alone (the `[0xc7b748]` writer, `0x6be451`/`0x6be477`): a
        // `0x40`-only porch still claims the camera.
        if claim.is_none() {
            if let Some(gi) = tap.in_group {
                if model
                    .group_nav
                    .get(gi)
                    .is_some_and(|n| n.flags & EXTERIOR == 0)
                {
                    // The MFOG target exists only when one of the camera's containing groups (the
                    // in-group and across-group, `0xc7cd88`) is a true interior, with no `0x48`
                    // (the byteOut loop `0x69de5f`–`0x69dea0`). Its candidates come from the
                    // first such group's fog indices.
                    fog_target = [Some(gi), tap.across]
                        .into_iter()
                        .flatten()
                        .find_map(|g| {
                            model
                                .group_nav
                                .get(g)
                                .filter(|n| n.flags & (EXTERIOR | EXTERIOR_LIT) == 0)
                        })
                        .and_then(|n| select_wmo_fog(&model.fogs, n.fog_indices, eye_local));
                    // fork: modern lighting — how far the room's own air reaches, for the
                    // medium ([`CameraRoomExtent`]): the MOGI box of the group whose fog just
                    // engaged, as a world sphere, and nothing when none did. Never the in-group
                    // as such: an exterior-lit courtyard claims the camera without carrying a
                    // fog, and Orgrimmar's entry canyon (g133, `0x0a841`) is a 170 yd sphere —
                    // stepping out of the gate tunnel spread the tunnel's teal over the whole
                    // city for the 4 s fade-out. A placement is a rotation and a translation,
                    // so the box's half-diagonal is its radius unchanged.
                    room_extent = fog_target
                        .and(fog_room(&model.group_nav, [Some(gi), tap.across]))
                        .and_then(|g| model.group_nav.get(g))
                        .map(|n| {
                            let box_local = group_aabb(n);
                            (
                                world_from_local.transform_point3(Vec3::from(box_local.center)),
                                Vec3::from(box_local.half_extents).length(),
                            )
                        });
                    claim = Some(InteriorClaim {
                        room: WmoRoom {
                            instance: entity,
                            group: gi as u16,
                        },
                        exterior_visible: model
                            .group_nav
                            .iter()
                            .zip(&inst.visible)
                            .any(|(n, v)| *v && n.flags & 0x148 != 0),
                    });
                    // The exterior windows are the claiming placement's alone: the reference floods
                    // the containing map object (`[0xc7b748]`) and defers that flood's windows,
                    // the list Pass 2 already used. Its push test is `0x8` alone
                    // (`opens_a_window`), not the `0x148` of `exterior_visible` above.
                    windows = Some((
                        std::mem::take(&mut fresh.windows),
                        std::mem::take(&mut fresh.reaches),
                    ));
                }
            }
        }
        // The `liquid_visited` latch, through the change gate: only a slot that flips marks the
        // instance changed.
        let held = inst.bypass_change_detection();
        let mut latched = false;
        if held.liquid_visited.len() != groups {
            held.liquid_visited = vec![false; groups];
            latched = true;
        }
        for g in 0..groups {
            if held.visible.get(g).copied().unwrap_or(false) && !held.liquid_visited[g] {
                held.liquid_visited[g] = true;
                latched = true;
            }
        }
        if latched {
            inst.set_changed();
        }
    }
    if camera_fog.0 != fog_target {
        camera_fog.0 = fog_target;
    }
    if camera_room.0 != room_extent {
        camera_room.0 = room_extent;
    }
    if camera_claim.0 != claim {
        camera_claim.0 = claim;
    }
    let (want_windows, want_reach) = match windows {
        Some((rects, reaches)) => (ExteriorWindows::Windows(rects), reaches),
        None => (ExteriorWindows::Unrestricted, Vec::new()),
    };
    if *camera_windows != want_windows {
        *camera_windows = want_windows;
    }
    if camera_reach.0 != want_reach {
        camera_reach.0 = want_reach;
    }
    if let Some((mut text, path)) = dump_text.zip(dump_to) {
        // The frame's published verdict, after every placement: the claim and the windows.
        text.push_str(&format!("CAMERA CLAIM: {:?}\n", camera_claim.0));
        text.push_str(&format!("EXTERIOR WINDOWS: {:?}\n", *camera_windows));
        // The folder is made only when there is a trace to put in it.
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&path, &text) {
            Ok(()) => info!("wmo cull trace written to {}", path.display()),
            Err(e) => warn!("wmo cull trace: cannot write {}: {e}", path.display()),
        }
    }
}

/// The per-frame [`FloodTrace`] tap: the down-ray's seeds, for the claim and the fog without a
/// second down-ray.
#[derive(Default)]
struct SeedTap {
    in_group: Option<usize>,
    /// The across-portal seed; with [`Self::in_group`], the client's `0xc7cd88`, the camera's
    /// containing groups (not the flood's visible set).
    across: Option<usize>,
}
impl FloodTrace for SeedTap {
    fn seed(&mut self, seeds: DownRaySeeds) {
        self.in_group = seeds.in_group;
        self.across = seeds.across;
    }
}

/// The deferred-window push test: a flood step opens a window onto the open world exactly when its
/// destination group carries `0x8` (`0x6b44f8 test byte [eax+0x10],0x8` before the push into
/// `[0xcbe324]`). Not [`EXTERIOR_LIT`], and not the weather flag's `0x148` (`0x6b42d0`, driving
/// `[0xca80c4]`), a correlated but different signal.
fn opens_a_window(nav: &[WmoGroupNav], to: usize) -> bool {
    nav.get(to).is_some_and(|n| n.flags & EXTERIOR != 0)
}

/// A tap on the flood's per-portal decisions: `()` is the per-frame no-op, and the probe dump and
/// the audit harness pass a recorder, so there is one flood and no traced copy.
pub(crate) trait FloodTrace {
    fn seed(&mut self, _seeds: DownRaySeeds) {}
    fn side_fail(&mut self, _from: usize, _portal: u16, _to: usize, _d: f32) {}
    fn rect_none(&mut self, _from: usize, _portal: u16, _to: usize) {}
    fn rect_collapse(&mut self, _from: usize, _portal: u16, _to: usize) {}
    fn entered(&mut self, _from: usize, _portal: u16, _to: usize, _rect: Rect, _on_plane: bool) {}
    /// One Pass-2 verdict: an EXTERIOR group against one deferred window.
    fn pass2(&mut self, _group: usize, _flags: u32, _window: usize, _admitted: bool) {}
    /// One Pass-3 verdict: a `0x10000` callback-pass group against the full frustum.
    fn pass3(&mut self, _group: usize, _flags: u32, _admitted: bool) {}
    /// More deferred windows than the reference's worklist holds: a tripwire on our flood
    /// ([`WORKLIST_CAP`]).
    fn worklist_over_cap(&mut self, _windows: usize) {}
}
impl FloodTrace for () {}

/// Two recorders on one flood: the per-frame [`SeedTap`] and the dump's [`TraceLog`].
impl<A: FloodTrace, B: FloodTrace> FloodTrace for (&mut A, &mut B) {
    fn seed(&mut self, seeds: DownRaySeeds) {
        self.0.seed(seeds);
        self.1.seed(seeds);
    }
    fn side_fail(&mut self, from: usize, portal: u16, to: usize, d: f32) {
        self.0.side_fail(from, portal, to, d);
        self.1.side_fail(from, portal, to, d);
    }
    fn rect_none(&mut self, from: usize, portal: u16, to: usize) {
        self.0.rect_none(from, portal, to);
        self.1.rect_none(from, portal, to);
    }
    fn rect_collapse(&mut self, from: usize, portal: u16, to: usize) {
        self.0.rect_collapse(from, portal, to);
        self.1.rect_collapse(from, portal, to);
    }
    fn entered(&mut self, from: usize, portal: u16, to: usize, rect: Rect, on_plane: bool) {
        self.0.entered(from, portal, to, rect, on_plane);
        self.1.entered(from, portal, to, rect, on_plane);
    }
    fn pass2(&mut self, group: usize, flags: u32, window: usize, admitted: bool) {
        self.0.pass2(group, flags, window, admitted);
        self.1.pass2(group, flags, window, admitted);
    }
    fn pass3(&mut self, group: usize, flags: u32, admitted: bool) {
        self.0.pass3(group, flags, admitted);
        self.1.pass3(group, flags, admitted);
    }
    fn worklist_over_cap(&mut self, windows: usize) {
        self.0.worklist_over_cap(windows);
        self.1.worklist_over_cap(windows);
    }
}

/// An absent recorder records nothing, so the dump costs no second flood.
impl<T: FloodTrace> FloodTrace for Option<T> {
    fn seed(&mut self, seeds: DownRaySeeds) {
        if let Some(t) = self {
            t.seed(seeds);
        }
    }
    fn side_fail(&mut self, from: usize, portal: u16, to: usize, d: f32) {
        if let Some(t) = self {
            t.side_fail(from, portal, to, d);
        }
    }
    fn rect_none(&mut self, from: usize, portal: u16, to: usize) {
        if let Some(t) = self {
            t.rect_none(from, portal, to);
        }
    }
    fn rect_collapse(&mut self, from: usize, portal: u16, to: usize) {
        if let Some(t) = self {
            t.rect_collapse(from, portal, to);
        }
    }
    fn entered(&mut self, from: usize, portal: u16, to: usize, rect: Rect, on_plane: bool) {
        if let Some(t) = self {
            t.entered(from, portal, to, rect, on_plane);
        }
    }
    fn pass2(&mut self, group: usize, flags: u32, window: usize, admitted: bool) {
        if let Some(t) = self {
            t.pass2(group, flags, window, admitted);
        }
    }
    fn pass3(&mut self, group: usize, flags: u32, admitted: bool) {
        if let Some(t) = self {
            t.pass3(group, flags, admitted);
        }
    }
    fn worklist_over_cap(&mut self, windows: usize) {
        if let Some(t) = self {
            t.worklist_over_cap(windows);
        }
    }
}

/// **Which rooms the camera can be LIT from** — the portal graph walked with no frustum at all.
///
/// Deliberately not [`compute_pvs_traced`], and deliberately much simpler. That function is a
/// faithful carve of the reference's cull, down to clipping the frustum to each doorway and
/// dropping a branch whose window closes; several decisions rest on it and it must not be bent to
/// serve lighting. But that same frustum clipping is what made a lamp's light blink when the player
/// tilted the camera — the room the lamp stands in left the *screen*, so the cull dropped it, so its
/// light left the table.
///
/// A light's reach is a property of the building, not of where the player is looking. So this walks
/// the same graph on connectivity alone: from the group the camera stands in, out through every
/// portal, as far as the graph goes. No frustum, no windows, no front-side test — a lamp behind you
/// lights the wall in front of you, and a lamp through a doorway you are not looking at still lights
/// its own room.
///
/// With no containing group (the camera outside the building) every group is lit: the building's
/// own lights are then gated by range alone, which is what the open world already does.
fn lighting_reach(model: &WmoModel, eye_local: [f32; 3], terrain_z: Option<f32>) -> Vec<bool> {
    let nav = &model.group_nav;
    let mut lit = vec![false; nav.len()];
    let seeds = down_ray_seeds(model, eye_local, terrain_z);
    let mut stack: Vec<usize> = Vec::new();
    match seeds.in_group {
        Some(c) => {
            stack.push(c);
            if let Some(a) = seeds.across {
                stack.push(a);
            }
        }
        // **Outside the building: no room gate at all**, and the narrower answers tried here do
        // not work. `lit` is ONE set for the whole frame, so anything tight enough to stop an
        // interior torch washing the road outside is also tight enough to darken that torch when
        // the player looks straight at it through the doorway — a hop cap did exactly that, and
        // the room you could plainly see through the abbey door went black.
        //
        // The two cases differ by the RECEIVER, not by the light, so they cannot be separated
        // here at all: that belongs to the indoor/outdoor class test the surface lanes already
        // carry (`static_gx.wgsl`'s `modern_local_lights`, which rejects an outdoor light on an
        // interior surface and wants the mirror-image rejection for the bleed).
        None => return vec![true; nav.len()],
    }
    while let Some(gi) = stack.pop() {
        if lit.get(gi).copied().unwrap_or(true) {
            continue;
        }
        lit[gi] = true;
        let Some(g) = nav.get(gi) else { continue };
        let lo = g.ref_start as usize;
        let hi = lo + g.ref_count as usize;
        for r in model.portal_refs.get(lo..hi).unwrap_or(&[]) {
            let to = r.group as usize;
            if to < lit.len() && !lit[to] {
                stack.push(to);
            }
        }
    }
    lit
}

/// The portal flood's per-group visible set, for the audit harness. `eye_local` is the camera in
/// WMO model space (WoW axes), the seed and the front-side test point; a portal vertex reaches clip
/// space as `clip_from_world * world_from_local * wow_to_bevy(v)`.
#[cfg(test)]
fn compute_pvs(
    model: &WmoModel,
    eye_local: [f32; 3],
    terrain_z: Option<f32>,
    clip_from_world: &Mat4,
    world_from_local: &Affine3A,
) -> Vec<bool> {
    compute_pvs_traced(
        model,
        eye_local,
        terrain_z,
        clip_from_world,
        world_from_local,
        &mut (),
    )
    .visible
}

/// What one flood publishes per group: this frame's PVS bit and the interior-fog bit, the client's
/// `[0xca7f00]`, which gates both interior-fog pushes (the group drawer `0x6b5190` and the group
/// liquid drawer `0x6b62e0`). A group takes the fog when the walk reaches it from one of the
/// camera's containing groups through true interiors only: the recursion's interior-chain anchor
/// clears at any `flags & 0x48` hop (`0x6b424f` → `0x6b42c6`). A `0x48` group is drawn by the
/// exterior drawer `0x6b4f10`, which pushes no fog, so the break decides the rooms beyond it.
pub(crate) struct GroupPvs {
    /// Per absolute group index: in this frame's portal PVS.
    pub(crate) visible: Vec<bool>,
    /// Per absolute group index: draws with the interior fog triple.
    pub(crate) interior_fog: Vec<bool>,
    /// The deferred exterior-window worklist this flood left, the client's `[0xcbe324]` records:
    /// one clipped NDC rect per doorway reached onto an `0x8` group. Pass 2 and [`ExteriorWindows`]
    /// both read it, as the reference feeds both from one list (`0x681070` copies it).
    pub(crate) windows: Vec<Rect>,
    /// Each window's portal reach ([`ExteriorWindowReach`]), parallel to `windows`.
    pub(crate) reaches: Vec<f32>,
    /// Stack steps across every run of the walk this frame, Pass 1 and each Pass-2 replay; the dump
    /// prints it.
    pub(crate) iters: u32,
}

/// One step on the walk's stack: the group to enter, the group it was entered from (so the entry
/// portal is never re-crossed), the screen rect, the depth, and whether the path kept an unbroken
/// interior chain from the seed.
type Step = (usize, usize, Rect, u32, bool);

/// The portal walk `0x6b41c0` and the per-frame state its runs share. The inside leg runs it from
/// the camera's groups (Pass 1, `0x6b3c05`) and again from every exterior group a deferred window
/// admits (Pass 2, `0x6b3d39`, the same recursion); the visible set, the fog lane, the window
/// stamps and the backstop are per frame, not per pass.
struct Flood<'a> {
    model: &'a WmoModel,
    eye_local: [f32; 3],
    clip_from_world: &'a Mat4,
    world_from_local: &'a Affine3A,
    visible: Vec<bool>,
    interior_fog: Vec<bool>,
    /// The deferred exterior-window worklist Pass 1 leaves: the client's `[0xcbe324]` records,
    /// reset at `0x6b3b51` and filled at `0x6b4541`.
    windows: Vec<Rect>,
    /// Each window's portal reach, stored beside its rect (`0x6b45ab`).
    reaches: Vec<f32>,
    /// The client's per-frame stamp on the portal record (`0x6b4505`, `[esi+0x18]` vs
    /// `[0xca8014]`): a portal adds at most one window per frame, however many paths reach it. It
    /// guards the store only, never the recursion; the first push wins.
    portal_pushed: Vec<bool>,
    /// Runaway backstop, counted across every run of the walk this frame ([`MAX_ITERS`]).
    iters: u32,
}

impl Flood<'_> {
    /// Drain `stack`, entering each step's group and crossing its portals. `record_windows` is the
    /// client's inside-leg flag `[0xcde5ac]`, set for Pass 1 only (`0x6b3b47`, cleared `0x6b3c1a`),
    /// and the push is guarded on it (`0x6b4511`).
    fn walk<T: FloodTrace>(&mut self, stack: &mut Vec<Step>, record_windows: bool, trace: &mut T) {
        let model = self.model;
        let nav = &model.group_nav;
        let eye_local = self.eye_local;
        while let Some((g, came, rect, depth, chain)) = stack.pop() {
            self.iters += 1;
            if self.iters > MAX_ITERS || g >= nav.len() {
                continue;
            }
            self.visible[g] = true;
            // The interior fog, ORed over paths: an unbroken interior chain, and a true interior
            // group, since the drawer router (`0x6b3fe8`, `flags & 0x48`) sends any other to
            // `0x6b4f10`, which pushes no fog, the seed's own group included.
            self.interior_fog[g] |= chain && nav[g].flags & (EXTERIOR | EXTERIOR_LIT) == 0;
            if depth >= DEPTH_CAP {
                continue;
            }
            let nav_g = &nav[g];
            // The chain breaks past an exterior or exterior-lit group, where the anchor clears
            // (`0x6b424f` → `0x6b42c6`).
            let chain_on = chain && nav_g.flags & (EXTERIOR | EXTERIOR_LIT) == 0;
            let start = nav_g.ref_start as usize;
            let end = (start + nav_g.ref_count as usize).min(model.portal_refs.len());
            for r in &model.portal_refs[start..end] {
                let neighbour = r.group as usize;
                // Never re-cross the entry portal, and skip the "no group" sentinel.
                if r.group == u16::MAX || neighbour == came {
                    continue;
                }
                let Some(info) = model.portal_infos.get(r.portal as usize) else {
                    continue;
                };
                // Front-side test on the `side`-oriented half-space: skip only on `d < 0`, enter on
                // exactly 0 (the client's `0x7ffd74`, 0.0); it always runs, as the on-plane skip
                // flag is dead code (`0x6b43c9`).
                let mut d = info.plane[0] * eye_local[0]
                    + info.plane[1] * eye_local[1]
                    + info.plane[2] * eye_local[2]
                    + info.plane[3];
                if r.side < 0 {
                    d = -d;
                }
                if d < 0.0 {
                    trace.side_fail(g, r.portal, neighbour, d);
                    continue;
                }
                // An eye in the portal gets the full screen (`0x6b46f0`), or a camera crossing a
                // doorway would clip the room ahead to nothing for a frame.
                let on_plane = eye_on_portal(&model.portal_vertices, info, eye_local);
                // The portal's screen rect, intersected with the carried one; a collapse kills the
                // branch.
                let prect = if on_plane {
                    FULL_SCREEN
                } else {
                    match portal_screen_rect(
                        model,
                        info,
                        self.clip_from_world,
                        self.world_from_local,
                    ) {
                        Some(p) => p,
                        None => {
                            trace.rect_none(g, r.portal, neighbour);
                            continue;
                        }
                    }
                };
                let Some(inter) = intersect_rect(rect, prect) else {
                    trace.rect_collapse(g, r.portal, neighbour);
                    continue;
                };
                trace.entered(g, r.portal, neighbour, inter, on_plane);
                // A portal onto an exterior group records a deferred window (`0x6b44f8` tests
                // `0x8`, `0x6b4539` counts, `0x6b4541` stores the rect).
                if record_windows && opens_a_window(nav, neighbour) {
                    if let Some(stamp) = self.portal_pushed.get_mut(r.portal as usize) {
                        if !*stamp {
                            *stamp = true;
                            self.windows.push(inter);
                            self.reaches.push(portal_reach(
                                model,
                                info,
                                self.clip_from_world,
                                self.world_from_local,
                            ));
                        }
                    }
                }
                stack.push((neighbour, g, inter, depth + 1, chain_on));
            }
        }
    }
}

/// The portal flood with a [`FloodTrace`] tap on every decision, the flood the frame runs.
fn compute_pvs_traced<T: FloodTrace>(
    model: &WmoModel,
    eye_local: [f32; 3],
    terrain_z: Option<f32>,
    clip_from_world: &Mat4,
    world_from_local: &Affine3A,
    trace: &mut T,
) -> GroupPvs {
    let nav = &model.group_nav;
    let mut flood = Flood {
        model,
        eye_local,
        clip_from_world,
        world_from_local,
        visible: vec![false; nav.len()],
        interior_fog: vec![false; nav.len()],
        windows: Vec::new(),
        reaches: Vec::new(),
        portal_pushed: vec![false; model.portal_infos.len()],
        iters: 0,
    };

    // Seed: the down-ray's set, each an independent full-screen root (the client appends both to
    // the containing-group set `0xc7cd88` and visits each, `0x6b3bd4`–`0x6b3c10`). Over no surface
    // or an exterior one, the outside leg seeds every EXTERIOR group full-screen.
    let mut stack: Vec<Step> = Vec::new();
    let seeds = down_ray_seeds(model, eye_local, terrain_z);
    trace.seed(seeds);
    // The inside-leg flag `[0xcde5ac]`: Pass 1 of `0x6b3b20` alone records windows, and
    // `0x6b3b20` runs only with a containing map object, `[0xc7b748]`, whose writer rejects an
    // `0x8` group (`0x6be451`). An eye over a building's own `0x8` porch is the driver's outside
    // leg (`0x6811ca`): no worklist, no Pass 2.
    let inside_leg = seeds
        .in_group
        .and_then(|gi| nav.get(gi))
        .is_some_and(|n| n.flags & EXTERIOR == 0);
    match seeds.in_group {
        Some(c) => {
            stack.push((c, usize::MAX, FULL_SCREEN, 0, true));
            if let Some(a) = seeds.across {
                stack.push((a, usize::MAX, FULL_SCREEN, 0, true));
            }
        }
        None => {
            // The outside leg: no containing map object, so no interior fog (the selector
            // `0x69de20` bails on `[0xc7b748] == 0`).
            for (gi, g) in nav.iter().enumerate() {
                if g.flags & EXTERIOR != 0 {
                    stack.push((gi, usize::MAX, FULL_SCREEN, 0, false));
                }
            }
        }
    }
    // Pass 1, the interior flood (`0x6b3bd4..0x6b3c10`), the only run that records windows.
    flood.walk(&mut stack, inside_leg, trace);

    // Pass 2, the deferred-window replay (`0x6b3c73..0x6b3d6f`): skipped with an empty worklist
    // (`0x6b3c87`), so a sealed room draws no exterior shell. Per window, each `0x8` group inside a
    // frustum clipped to that window (`0x6b3d13`/`0x6b3d22`) is flooded from, not merely marked
    // (`0x6b3d39 call 0x6b41c0`), the walk carrying the window's rect on. The sub-frustum is
    // `crate::exterior_cull::window_frustum` without its narrowness reject, which gates the scene
    // walk `0x682fa0` and not the volume builder `0x682930`: a doorway too narrow to admit a
    // hillside still admits a wall.
    if flood.windows.len() > WORKLIST_CAP {
        trace.worklist_over_cap(flood.windows.len());
    }
    // The worklist is fixed once Pass 1 ends, so it is replayed by value.
    let worklist = std::mem::take(&mut flood.windows);
    for (wi, rect) in worklist.iter().enumerate() {
        let frustum = crate::exterior_cull::window_frustum(*rect, clip_from_world);
        // Every group is a candidate against every window: the reference has no already-drawn
        // skip, and skipping a visible group would also skip the walk through it.
        let mut roots: Vec<Step> = Vec::new();
        for (gi, g) in nav.iter().enumerate() {
            // `¬0x10000 ∧ 0x8` (`0x6b3d0b`, `0x6b3d13`): a callback-pass group is Pass 3's alone.
            if g.flags & (EXTERIOR | CALLBACK_PASS) != EXTERIOR {
                continue;
            }
            let admitted = frustum.intersects_obb(&group_aabb(g), world_from_local, true, true);
            trace.pass2(gi, g.flags, wi, admitted);
            if admitted {
                roots.push((gi, usize::MAX, *rect, 0, false));
            }
        }
        flood.walk(&mut roots, false, trace);
    }
    flood.windows = worklist;

    // Pass 3, the callback pass (`0x6b3d6f..0x6b3dbc`): each `0x10000` group against the full
    // camera frustum (no `0x682930` here), and the `jbe` fall-through, so it runs with an empty
    // worklist too and a callback group draws even from a sealed room (`0x6b4160` → `0x685d70`,
    // the flood's per-group callback without the recursion). Both legs run it: the camera-outside
    // seed `0x6b3dd0` forks on the same bit. Every arm frustum-tests the MOGI AABB first
    // (`0x6b3d9d`, `0x6b3f2d`, `call 0x682f60`).
    let base = Frustum(bevy::math::primitives::ViewFrustum::from_clip_from_world(
        clip_from_world,
    ));
    for (gi, g) in nav.iter().enumerate() {
        if flood.visible[gi] || g.flags & CALLBACK_PASS == 0 {
            continue;
        }
        let aabb = group_aabb(g);
        let admitted = base.intersects_obb(&aabb, world_from_local, true, true);
        trace.pass3(gi, g.flags, admitted);
        flood.visible[gi] |= admitted;
    }
    GroupPvs {
        visible: flood.visible,
        interior_fog: flood.interior_fog,
        windows: flood.windows,
        reaches: flood.reaches,
        iters: flood.iters,
    }
}

/// `0x6b4680`: the largest view depth of the portal's vertices, from 0 and replaced only by a
/// strictly larger one (`0x6b46cf`). A perspective clip `w` is the depth along the camera's forward.
fn portal_reach(
    model: &WmoModel,
    info: &WmoPortalInfo,
    clip_from_world: &Mat4,
    world_from_local: &Affine3A,
) -> f32 {
    portal_poly(&model.portal_vertices, info)
        .into_iter()
        .flatten()
        .map(|v| {
            let world = world_from_local.transform_point3(wow_to_bevy(*v));
            (*clip_from_world * world.extend(1.0)).w
        })
        .fold(0.0, |m, w| if w > m { w } else { m })
}

/// A group's MOGI bounding box as a Bevy-space [`Aabb`] in the placement's local frame, the bound
/// the frustum tests take (the client's `0x682f60` takes the same box). `wow_to_bevy` is a signed
/// axis permutation, so the two corners' min and max are the whole conversion.
fn group_aabb(nav: &WmoGroupNav) -> Aabb {
    let a = wow_to_bevy(nav.bbox_min);
    let b = wow_to_bevy(nav.bbox_max);
    Aabb::from_min_max(a.min(b), a.max(b))
}

/// fork: modern lighting — the group of the camera's containing set (the down-ray's in-group,
/// then its across-portal partner) whose MFOG the interior fog engages: the first TRUE interior,
/// `flags & (EXTERIOR | EXTERIOR_LIT) == 0` — the conjunct the fog target is selected under.
fn fog_room(nav: &[WmoGroupNav], seeds: [Option<usize>; 2]) -> Option<usize> {
    seeds.into_iter().flatten().find(|&g| {
        nav.get(g)
            .is_some_and(|n| n.flags & (EXTERIOR | EXTERIOR_LIT) == 0)
    })
}

/// The portal's polygon (WoW model space), or `None` if the span is out of range or under 3.
fn portal_poly<'a>(
    portal_vertices: &'a [[f32; 3]],
    info: &WmoPortalInfo,
) -> Option<&'a [[f32; 3]]> {
    let start = info.start_vertex as usize;
    let verts = portal_vertices.get(start..start + info.count as usize)?;
    (verts.len() >= 3).then_some(verts)
}

/// Does the eye lie in this portal: within [`ON_PLANE_EPS`] of its plane, inclusive, and inside its
/// polygon (dominant-axis projection, `0x7c23e0`)? The flood then gives it the full screen.
fn eye_on_portal(portal_vertices: &[[f32; 3]], info: &WmoPortalInfo, eye: [f32; 3]) -> bool {
    let [nx, ny, nz, d] = info.plane;
    if (nx * eye[0] + ny * eye[1] + nz * eye[2] + d).abs() > ON_PLANE_EPS {
        return false;
    }
    let Some(verts) = portal_poly(portal_vertices, info) else {
        return false;
    };
    // The eye is on the plane, so projecting out the dominant axis keeps it in place.
    let (u, v) = dominant_axes(info.plane);
    point_in_poly_2d(verts.iter().map(|p| (p[u], p[v])), (eye[u], eye[v]))
}

/// Even-odd point-in-polygon on 2-D projected vertices.
fn point_in_poly_2d(pts: impl Iterator<Item = (f32, f32)> + Clone, p: (f32, f32)) -> bool {
    let mut inside = false;
    let mut prev = pts.clone().last();
    for cur in pts {
        if let Some(pr) = prev {
            if (cur.1 > p.1) != (pr.1 > p.1)
                && p.0 < (pr.0 - cur.0) * (p.1 - cur.1) / (pr.1 - cur.1) + cur.0
            {
                inside = !inside;
            }
        }
        prev = Some(cur);
    }
    inside
}

/// A portal polygon's NDC bounding rect, the client's `0x6b46f0`: to clip space, clipped against
/// the four side planes only (no near plane), then the per-vertex NDC min and max, where
/// `|w| < 0.001` becomes `+1e-5` and a vertex with `w <= -0.001` divides by its own negative `w`,
/// its mirrored NDC entering the rect. So a doorway the eye straddles blows open instead of
/// collapsing. Unclamped; `None` when fully clipped (the client's `rc.flags |= 0x1` skip).
fn portal_screen_rect(
    model: &WmoModel,
    info: &WmoPortalInfo,
    clip_from_world: &Mat4,
    world_from_local: &Affine3A,
) -> Option<Rect> {
    let verts = portal_poly(&model.portal_vertices, info)?;
    // WoW model vertex → world → clip, the path the rendered geometry takes.
    let clip: Vec<Vec4> = verts
        .iter()
        .map(|v| {
            let world = world_from_local.transform_point3(wow_to_bevy(*v));
            *clip_from_world * world.extend(1.0)
        })
        .collect();
    let clipped = clip_side_planes(&clip)?;
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for c in &clipped {
        let w = if c.w.abs() < W_CLAMP_BAND {
            W_CLAMP_SUB
        } else {
            c.w
        };
        let inv = 1.0 / w;
        let (x, y) = (c.x * inv, c.y * inv);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    Some([min_x, min_y, max_x, max_y])
}

/// Sutherland–Hodgman clip of a clip-space polygon against the four side planes (`x ≥ −w`, `x ≤ w`,
/// `y ≥ −w`, `y ≤ w`; the client's guard pyramid `0xc7d0dc` fed to `0x6b4a80`). No near plane: a
/// polygon spanning the eye survives as points near `w = 0` for the caller's `w`-clamp. `None`
/// under 3 vertices.
fn clip_side_planes(poly: &[Vec4]) -> Option<Vec<Vec4>> {
    const PLANES: [fn(&Vec4) -> f32; 4] =
        [|v| v.w + v.x, |v| v.w - v.x, |v| v.w + v.y, |v| v.w - v.y];
    let mut cur = poly.to_vec();
    for f in PLANES {
        let n = cur.len();
        let mut out: Vec<Vec4> = Vec::with_capacity(n + 2);
        for i in 0..n {
            let a = cur[i];
            let b = cur[(i + 1) % n];
            let (fa, fb) = (f(&a), f(&b));
            if fa >= 0.0 {
                out.push(a);
            }
            if (fa >= 0.0) != (fb >= 0.0) {
                let t = fa / (fa - fb);
                out.push(a + (b - a) * t);
            }
        }
        cur = out;
        if cur.len() < 3 {
            return None;
        }
    }
    Some(cur)
}

/// Intersect two NDC rects; `None` unless the overlap spans [`RECT_EPS`] on both axes.
fn intersect_rect(a: Rect, b: Rect) -> Option<Rect> {
    let rect = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    (rect[2] - rect[0] >= RECT_EPS && rect[3] - rect[1] >= RECT_EPS).then_some(rect)
}

#[cfg(test)]
mod audit;

#[cfg(test)]
mod tests {
    use super::*;

    use benilla_assets::WmoPortalRef;

    /// Any visible referrer draws a prop; an index past the visible set fails open.
    #[test]
    fn a_prop_is_drawn_while_any_of_its_rooms_is_visible() {
        let inst = WmoPortalInstance {
            handle: Handle::default(),
            world_from_local: Affine3A::IDENTITY,
            name_set: 0,
            visible: vec![false, true, false],
            lit: vec![false, true, false],
            interior_fog: vec![false; 3],
            liquid_visited: vec![false; 3],
            flooded: vec![None; 3],
        };
        let key = |groups: &[u16]| WmoGroupVis {
            instance: Entity::PLACEHOLDER,
            groups: Arc::from(groups),
        };
        assert!(key(&[1]).drawn_by(&inst), "its own room is visible");
        assert!(!key(&[0]).drawn_by(&inst), "its own room is culled");
        assert!(
            key(&[0, 2, 1]).drawn_by(&inst),
            "one visible referrer out of three draws the fall"
        );
        assert!(!key(&[0, 2]).drawn_by(&inst), "no referrer is visible");
        assert!(
            !key(&[]).drawn_by(&inst),
            "no referrers at all: nothing ORs"
        );
        assert!(key(&[9]).drawn_by(&inst), "index past the set fails open");
    }

    /// The interior-fog key ORs like `drawn_by`, but a lookup miss reads false.
    #[test]
    fn the_interior_fog_key_ors_over_rooms_and_fails_closed() {
        let inst = WmoPortalInstance {
            handle: Handle::default(),
            world_from_local: Affine3A::IDENTITY,
            name_set: 0,
            visible: vec![true; 3],
            lit: vec![true; 3],
            interior_fog: vec![false, true, false],
            liquid_visited: vec![false; 3],
            flooded: vec![None; 3],
        };
        let key = |groups: &[u16]| WmoGroupVis {
            instance: Entity::PLACEHOLDER,
            groups: Arc::from(groups),
        };
        assert!(key(&[1]).interior_fogged_by(&inst), "its own room is on it");
        assert!(
            !key(&[0]).interior_fogged_by(&inst),
            "its own room is off it"
        );
        assert!(
            key(&[0, 2, 1]).interior_fogged_by(&inst),
            "one referrer on the lane is enough"
        );
        assert!(
            !key(&[0, 2]).interior_fogged_by(&inst),
            "no referrer is on it"
        );
        assert!(
            !key(&[]).interior_fogged_by(&inst),
            "no referrers: nothing ORs"
        );
        assert!(
            !key(&[9]).interior_fogged_by(&inst),
            "index past the set fails CLOSED — the opposite of drawn_by"
        );
    }

    /// Only `0x8` opens a window (`0x6b44f8`). The flags are Stormwind's own, from a live cull
    /// trace: of its 306 groups only `g46` (`0x08809`) carries `0x8`; its streets and rooms are
    /// `0x40` without `0x8`.
    #[test]
    fn only_an_exterior_group_opens_a_window_onto_the_open_world() {
        let g = |flags: u32| WmoGroupNav {
            flags,
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
            ref_start: 0,
            ref_count: 0,
            area_table_id: 0,
            fog_indices: [0; 4],
            group_liquid: 0xf,
        };
        // Stormwind rooms and streets…
        let nav = [g(0x0b841), g(0x0a841), g(0x0ba41), g(0x08809), g(0x02a05)];
        for to in [0, 1, 2] {
            assert!(
                !opens_a_window(&nav, to),
                "g{to} is EXTERIOR_LIT (0x40) without EXTERIOR (0x8) — a lit indoor room, not a \
                 doorway onto Elwynn"
            );
        }
        assert!(!opens_a_window(&nav, 4), "0x02a05 carries neither bit");
        // …and g46, the city's one exterior shell.
        assert!(
            opens_a_window(&nav, 3),
            "0x08809 carries 0x8 — this one IS the open world"
        );
        // A destination past the nav (a corrupt MOPR ref) fails closed: no doorway.
        assert!(
            !opens_a_window(&nav, 99),
            "an out-of-range group is no window"
        );
    }

    /// **A room's air is the room whose fog engaged, never an exterior-lit courtyard's.** The flag
    /// values are Orgrimmar's own (`ogrimmar.wmo`): g006, the gate tunnel's last segment
    /// (`0x0a805`, a true interior carrying the building's teal MFOG), opens through a doorway
    /// portal onto g133, the entry canyon (`0x0a841`, exterior-lit, a 170 yd box reaching the
    /// Valley of Strength). The modern medium took the courtyard's sphere for the tunnel's fog, and
    /// the whole city fogged over for the 4 s fade-out whenever the camera stepped out.
    #[test]
    fn only_a_true_interior_carries_the_rooms_air() {
        let g = |flags: u32| WmoGroupNav {
            flags,
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
            ref_start: 0,
            ref_count: 0,
            area_table_id: 0,
            fog_indices: [0; 4],
            group_liquid: 0xf,
        };
        let (tunnel, canyon, outside) = (0, 1, 2);
        let nav = [g(0x0a805), g(0x0a841), g(0x08009)];
        assert_eq!(fog_room(&nav, [Some(tunnel), None]), Some(tunnel));
        assert_eq!(
            fog_room(&nav, [Some(canyon), None]),
            None,
            "the canyon claims the camera but carries no room"
        );
        assert_eq!(
            fog_room(&nav, [Some(canyon), Some(tunnel)]),
            Some(tunnel),
            "on a crossing into the tunnel the room is the tunnel, not the courtyard"
        );
        assert_eq!(fog_room(&nav, [Some(outside), None]), None);
        assert_eq!(
            fog_room(&nav, [Some(99), None]),
            None,
            "an unknown group is no room"
        );
    }

    #[test]
    fn eye_on_portal_only_within_the_band_and_polygon() {
        // A vertical doorway in the x=0 plane, spanning y ∈ [-2,2], z ∈ [0,4].
        let verts = vec![
            [0.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 4.0],
            [0.0, -2.0, 4.0],
        ];
        let info = WmoPortalInfo {
            start_vertex: 0,
            count: 4,
            plane: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(eye_on_portal(&verts, &info, [0.0, 0.0, 2.0]));
        assert!(eye_on_portal(&verts, &info, [0.005, 1.5, 3.5]));
        // In the plane but above the door.
        assert!(!eye_on_portal(&verts, &info, [0.0, 0.0, 5.0]));
        // Within the door's span but off the plane.
        assert!(!eye_on_portal(&verts, &info, [0.5, 0.0, 2.0]));
    }

    #[test]
    fn point_in_poly_2d_hits_inside_and_misses_outside() {
        let sq = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        assert!(point_in_poly_2d(sq.iter().copied(), (0.0, 0.0)));
        assert!(point_in_poly_2d(sq.iter().copied(), (-0.9, 0.9)));
        assert!(!point_in_poly_2d(sq.iter().copied(), (1.5, 0.0)));
        assert!(!point_in_poly_2d(sq.iter().copied(), (0.0, -1.5)));
    }

    #[test]
    fn intersect_rect_collapses_to_none() {
        assert_eq!(
            intersect_rect([-1.0, -1.0, 0.0, 0.0], [-0.5, -0.5, 1.0, 1.0]),
            Some([-0.5, -0.5, 0.0, 0.0])
        );
        // Disjoint rects: the window collapses.
        assert_eq!(
            intersect_rect([-1.0, -1.0, -0.5, -0.5], [0.5, 0.5, 1.0, 1.0]),
            None
        );
        // A sliver thinner than the client's 0.001 zero-area eps also collapses.
        assert_eq!(
            intersect_rect([-1.0, -1.0, 1.0, 1.0], [0.0, -1.0, 0.0005, 1.0]),
            None
        );
    }

    #[test]
    fn clip_side_planes_drops_behind_and_off_screen_keeps_straddling() {
        // Fully behind the eye (w < 0, inside the mirror cone): outside every side half-space.
        let behind = [
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.1, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.1, 0.0, -1.0),
        ];
        assert!(clip_side_planes(&behind).is_none());
        let front = [
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.5, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 0.5, 0.0, 1.0),
        ];
        assert_eq!(clip_side_planes(&front).map(|p| p.len()), Some(3));
        // Off-screen laterally, x > w for every vertex.
        let off = [
            Vec4::new(2.0, 0.0, 0.0, 1.0),
            Vec4::new(3.0, 0.0, 0.0, 1.0),
            Vec4::new(2.0, 1.0, 0.0, 1.0),
        ];
        assert!(clip_side_planes(&off).is_none());
        // Straddling the eye plane, one vertex behind: survives, with boundary points near w = 0.
        let straddle = [
            Vec4::new(0.0, -0.5, 0.0, 1.0),
            Vec4::new(0.0, 0.5, 0.0, 1.0),
            Vec4::new(0.0, 0.0, 0.0, -0.5),
        ];
        assert!(clip_side_planes(&straddle).is_some());
    }

    #[test]
    fn portal_rect_explodes_open_on_the_doorway_approach() {
        // The camera 0.05 yd before the doorway plane, outside the 0.01 on-plane band, looking
        // through it: with no near-plane clip the rect blows out to the whole screen.
        let model = portal_model();
        let info = &model.portal_infos[0];
        // WoW (−0.05, 0, 0) → Bevy (0, 0, 0.05); looking along WoW +x <=> Bevy −z.
        let view = Mat4::look_at_rh(
            bevy::math::Vec3::new(0.0, 0.0, 0.05),
            bevy::math::Vec3::new(0.0, 0.0, -10.0),
            bevy::math::Vec3::Y,
        );
        let clip = Mat4::perspective_rh(0.9, 16.0 / 9.0, 0.1, 1000.0) * view;
        let rect = portal_screen_rect(&model, info, &clip, &Affine3A::IDENTITY)
            .expect("a doorway ahead must keep a rect");
        let inter = intersect_rect(FULL_SCREEN, rect).expect("must overlap the screen");
        assert!(inter[0] <= -0.99 && inter[1] <= -0.99 && inter[2] >= 0.99 && inter[3] >= 0.99);
    }

    #[test]
    fn portal_rect_is_none_when_fully_behind_the_camera() {
        let model = portal_model();
        let info = &model.portal_infos[0];
        // Camera 5 yd past the doorway (WoW +x is Bevy −z), looking on along −z.
        let view = Mat4::look_at_rh(
            bevy::math::Vec3::new(0.0, 0.0, -5.0),
            bevy::math::Vec3::new(0.0, 0.0, -10.0),
            bevy::math::Vec3::Y,
        );
        let clip = Mat4::perspective_rh(0.9, 16.0 / 9.0, 0.1, 1000.0) * view;
        assert!(portal_screen_rect(&model, info, &clip, &Affine3A::IDENTITY).is_none());
    }

    /// Pass 2 is a per-window test (`0x6b3d13`/`0x6b3d22`): of two portal-disconnected shells, the
    /// one through the doorway draws and the one 30 yd aside does not.
    #[test]
    fn pass_two_admits_only_the_exterior_shells_a_doorway_shows() {
        let model = doorway_model();
        // Eye 3 yd back from the doorway, WoW (-3, 0, 1) = Bevy (0, 1, 3), looking along WoW +x
        // (Bevy −z), straight through the opening.
        let eye_local = [-3.0, 0.0, 1.0];
        let view = Mat4::look_at_rh(
            bevy::math::Vec3::new(0.0, 1.0, 3.0),
            bevy::math::Vec3::new(0.0, 1.0, -10.0),
            bevy::math::Vec3::Y,
        );
        let clip = Mat4::perspective_rh(0.9, 16.0 / 9.0, 0.1, 1000.0) * view;
        let pvs = compute_pvs_traced(&model, eye_local, None, &clip, &Affine3A::IDENTITY, &mut ());

        assert_eq!(
            pvs.windows.len(),
            1,
            "the one portal onto an exterior group is the one deferred window"
        );
        assert!(pvs.visible[0], "the camera's own room");
        assert!(pvs.visible[1], "the exterior group the portal opens onto");
        assert!(
            pvs.visible[2],
            "a portal-disconnected shell straight through the doorway is admitted by its window \
             — this is the half the flood alone can never reach"
        );
        assert!(
            !pvs.visible[3],
            "a portal-disconnected shell 30 yd off the doorway's axis is culled: no window shows \
             it, and drawing it would be pure over-draw"
        );
        assert!(
            pvs.visible[4],
            "an admitted shell is FLOODED FROM, not merely marked — g4 is an interior behind \
             g2 with no edge to the seed's half of the graph, so the walk through g2's own back \
             door is its only route into the PVS (`0x6b3d39 call 0x6b41c0`). A marking Pass 2 \
             leaves it culled, which from a Darnassus shop is most of the city"
        );

        // The control: standing in the doorway takes the full-screen branch, and the side shell
        // comes back, so the cull above is the window's.
        let in_door = compute_pvs_traced(
            &model,
            [0.0, 0.0, 1.0],
            None,
            &clip,
            &Affine3A::IDENTITY,
            &mut (),
        );
        assert!(
            in_door.visible[3],
            "with the doorway full-screen, every shell in front of the camera draws"
        );
    }

    /// Pass 3 draws a `0x10000` group from a sealed room (the `jbe` fall-through at `0x6b3d6f`):
    /// with an empty worklist every `0x8` group is dark, yet the callback group draws, as
    /// Razorfen's enclosing shells (`0x8` and `0x10000`) and Stormwind's canals do.
    #[test]
    fn pass_three_draws_a_callback_group_from_a_sealed_room() {
        // The doorway model, sealed: no portals, no windows. g1..g3 keep `0x8`; g3 gains `0x10000`.
        let mut model = doorway_model();
        model.portal_refs.clear();
        model.group_nav[0].ref_count = 0;
        model.group_nav[1].ref_count = 0;
        model.group_nav[3].flags |= CALLBACK_PASS;
        // Aim at g3, 30 yd off-axis, so the full frustum holds it: WoW (2..6, 30..34) is Bevy
        // x ∈ [-34,-30], z ∈ [-6,-2].
        let view = Mat4::look_at_rh(
            bevy::math::Vec3::new(0.0, 1.0, 3.0),
            bevy::math::Vec3::new(-32.0, 1.0, -4.0),
            bevy::math::Vec3::Y,
        );
        let clip = Mat4::perspective_rh(0.9, 16.0 / 9.0, 0.1, 1000.0) * view;
        let pvs = compute_pvs_traced(
            &model,
            [-3.0, 0.0, 1.0],
            None,
            &clip,
            &Affine3A::IDENTITY,
            &mut (),
        );

        assert!(
            pvs.windows.is_empty(),
            "a sealed room opens no doorway onto the outdoors"
        );
        assert!(pvs.visible[0], "the camera's own room");
        assert!(
            !pvs.visible[1] && !pvs.visible[2],
            "plain 0x8 shells stay dark with an empty worklist — Pass 2 never runs"
        );
        assert!(
            pvs.visible[3],
            "the 0x10000 group draws anyway: Pass 3 is the jbe fall-through, full frustum, ungated"
        );

        // The control: Pass 3 is still a frustum test, so facing away darkens g3.
        let away = Mat4::perspective_rh(0.9, 16.0 / 9.0, 0.1, 1000.0)
            * Mat4::look_at_rh(
                bevy::math::Vec3::new(0.0, 1.0, 3.0),
                bevy::math::Vec3::new(32.0, 1.0, 10.0),
                bevy::math::Vec3::Y,
            );
        let behind = compute_pvs_traced(
            &model,
            [-3.0, 0.0, 1.0],
            None,
            &away,
            &Affine3A::IDENTITY,
            &mut (),
        );
        assert!(
            !behind.visible[3],
            "a callback group behind the camera is still frustum-culled"
        );

        // The outside leg draws it too (`0x6b3dd0` forks on the same bit): an eye with no floor
        // under it seeds only `0x8` groups, and g3, with no portal refs, is Pass 3's alone.
        let outside = compute_pvs_traced(
            &model,
            [-3.0, 0.0, 500.0],
            None,
            &clip,
            &Affine3A::IDENTITY,
            &mut (),
        );
        assert!(
            outside.visible[3],
            "the callback arm belongs to BOTH legs — gating it on the inside leg left 20 of the 24 \
             real groups undrawable from anywhere"
        );
    }

    /// An interior room `g0` (floor at z=0, the seed) whose doorway `p0` (WoW x=0, y ∈ [-0.5,0.5],
    /// z ∈ [0,2]) opens onto exterior `g1`, two portal-disconnected exterior shells, `g2` ahead
    /// through the opening and `g3` 30 yd aside, and `g4`, reached only through `g2`.
    fn doorway_model() -> WmoModel {
        let nav = |flags: u32, min: [f32; 3], max: [f32; 3], ref_start: u16, ref_count: u16| {
            WmoGroupNav {
                flags,
                bbox_min: min,
                bbox_max: max,
                ref_start,
                ref_count,
                area_table_id: 0,
                fog_indices: [0; 4],
                group_liquid: benilla_formats::NO_GROUP_LIQUID,
            }
        };
        let floor = |z: f32| {
            vec![
                [[-10.0, -10.0, z], [0.0, -10.0, z], [0.0, 10.0, z]],
                [[-10.0, -10.0, z], [0.0, 10.0, z], [-10.0, 10.0, z]],
            ]
        };
        WmoModel {
            portal_vertices: vec![
                [0.0, -0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 2.0],
                [0.0, -0.5, 2.0],
                // p1, the back door of g2 and g4's only way in.
                [6.0, -0.5, 0.0],
                [6.0, 0.5, 0.0],
                [6.0, 0.5, 2.0],
                [6.0, -0.5, 2.0],
            ],
            portal_infos: vec![
                WmoPortalInfo {
                    start_vertex: 0,
                    count: 4,
                    plane: [1.0, 0.0, 0.0, 0.0],
                },
                WmoPortalInfo {
                    start_vertex: 4,
                    count: 4,
                    plane: [1.0, 0.0, 0.0, -6.0],
                },
            ],
            // g0 ↔ g1, `side = -1` putting the eye at x < 0 on the entering half-space; then
            // g2 ↔ g4 through p1, reachable only by Pass 2 admitting g2 and walking on.
            portal_refs: vec![
                WmoPortalRef {
                    portal: 0,
                    group: 1,
                    side: -1,
                },
                WmoPortalRef {
                    portal: 0,
                    group: 0,
                    side: 1,
                },
                WmoPortalRef {
                    portal: 1,
                    group: 4,
                    side: -1,
                },
                WmoPortalRef {
                    portal: 1,
                    group: 2,
                    side: 1,
                },
            ],
            group_nav: vec![
                nav(0x2000, [-10.0, -10.0, 0.0], [0.0, 10.0, 3.0], 0, 1),
                nav(EXTERIOR, [1.0, -0.5, 0.0], [5.0, 0.5, 2.0], 1, 1),
                nav(EXTERIOR, [2.0, -0.5, 0.0], [6.0, 0.5, 2.0], 2, 1),
                nav(EXTERIOR, [2.0, 30.0, 0.0], [6.0, 34.0, 2.0], 0, 0),
                nav(0x2000, [7.0, -0.5, 0.0], [11.0, 0.5, 2.0], 3, 1),
            ],
            group_collision_tris: vec![floor(0.0), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            ..portal_model()
        }
    }

    /// A bare model with one vertical doorway portal in the WoW x=0 plane, y and z in [-1,1].
    /// A three-room chain: 0 —portal— 1 —portal— 2, with group 2 nowhere near the camera's view.
    fn chain_model() -> WmoModel {
        let mut m = portal_model();
        // An interior room spanning the origin, so the down-ray's seed lands in group 0.
        let room = |ref_start: u16, ref_count: u16| WmoGroupNav {
            flags: 0,
            bbox_min: [-8.0, -8.0, -8.0],
            bbox_max: [8.0, 8.0, 8.0],
            ref_start,
            ref_count,
            area_table_id: 0,
            fog_indices: [0; 4],
            group_liquid: benilla_formats::NO_GROUP_LIQUID,
        };
        m.group_nav = vec![room(0, 1), room(1, 2), room(3, 1)];
        m.portal_refs = vec![
            WmoPortalRef {
                portal: 0,
                group: 1,
                side: 1,
            },
            WmoPortalRef {
                portal: 0,
                group: 0,
                side: -1,
            },
            WmoPortalRef {
                portal: 0,
                group: 2,
                side: 1,
            },
            WmoPortalRef {
                portal: 0,
                group: 1,
                side: -1,
            },
        ];
        m
    }

    /// Two triangles of walkable floor spanning `x0..x1 × y0..y1` at height `z` (WoW model space).
    fn floor_quad(x0: f32, x1: f32, y0: f32, y1: f32, z: f32) -> Vec<[[f32; 3]; 3]> {
        vec![
            [[x0, y0, z], [x1, y0, z], [x1, y1, z]],
            [[x0, y0, z], [x1, y1, z], [x0, y1, z]],
        ]
    }

    /// [`chain_model`] with the three rooms laid out side by side in X and **given floors**, so the
    /// down-ray actually resolves a containing group.
    ///
    /// `chain_model` alone carries no collision triangles at all, so `down_ray_seeds` answers
    /// `in_group: None` for every eye — which meant the golden below was, until this model existed,
    /// silently measuring the OUTSIDE leg's old fail-open ("return every group lit") and never once
    /// walking the portal chain it claims to pin.
    fn floored_chain_model() -> WmoModel {
        let mut m = chain_model();
        let span = [(-9.0f32, -3.0f32), (-3.0, 3.0), (3.0, 9.0)];
        for (gi, (x0, x1)) in span.iter().enumerate() {
            m.group_nav[gi].bbox_min = [*x0, -3.0, -1.0];
            m.group_nav[gi].bbox_max = [*x1, 3.0, 5.0];
        }
        m.group_collision_tris = span
            .iter()
            .map(|(x0, x1)| floor_quad(*x0, *x1, -3.0, 3.0, -1.0))
            .collect();
        m.group_collision_grids = vec![None; 3];
        m.group_camera_only_tris = vec![Vec::new(); 3];
        m
    }

    /// GOLDEN — **a light's room gate is reachability, and reachability cannot see the camera.**
    ///
    /// The lamp at Northshire Abbey's entrance blinked as the player tilted the camera or backed
    /// away. Lights were gated on [`WmoPortalInstance::visible`] — the reference's portal cull,
    /// which clips the frustum to each doorway — so looking away from the doorway dropped the room
    /// and the lamp's light left the table. A lamp beside a portal re-rolls that verdict on every
    /// small camera move.
    ///
    /// [`lighting_reach`] takes no orientation and no projection **at all**: its signature is the
    /// guarantee, and this pins the behaviour behind it — every room on the portal chain is lit
    /// from the camera's own room, however far down the chain and whatever is on screen.
    #[test]
    fn lighting_reach_floods_the_whole_portal_chain_and_ignores_the_view() {
        let m = floored_chain_model();
        // Standing in room 0. Room 2 is two doorways away and cannot be on screen through both.
        let lit = lighting_reach(&m, [-6.0, 0.0, 1.0], None);
        assert_eq!(lit.len(), 3, "one verdict per group");
        assert!(
            lit.iter().all(|v| *v),
            "every room on the chain is lit: {lit:?}"
        );
    }

    /// GOLDEN — **"names rooms" is not "is indoors", and the porch lamp is the difference.**
    ///
    /// The surface lanes' class test runs both ways off the packed `w`: an interior surface takes
    /// only indoor lights, an exterior one only outdoor lights. That is only safe if `w` means
    /// what it says. Packed from `rooms.is_some()` it meant "is a WMO light", and a sconce bolted
    /// to the outside of the abbey names rooms too — so the two-way test would have switched off
    /// every porch lamp and courtyard brazier in the game.
    #[test]
    fn a_lamp_on_a_buildings_outside_is_not_an_indoor_light() {
        let mut m = floored_chain_model();
        m.group_nav[0].flags |= EXTERIOR; // group 0 is the outdoor shell
        let e = Entity::from_raw_u32(1).expect("a valid test entity id");
        let vis = |groups: &[u16]| WmoGroupVis {
            instance: e,
            groups: groups.into(),
        };
        // A candle in the nave: only interior groups name it.
        assert!(rooms_are_indoor(&vis(&[1, 2]), &m));
        // A sconce on the shell — it lights the open world, and must survive the test.
        assert!(!rooms_are_indoor(&vis(&[0]), &m));
        // A fixture a courtyard ALSO names keeps its outdoor half: all naming groups must be
        // interior, because the error directions are lopsided (1276).
        assert!(!rooms_are_indoor(&vis(&[0, 1]), &m));
        // `0x40`-only is an interior GRAPH group lit as outdoors — Stormwind's streets (decision
        // 0475). A street lamp there is not an indoor light.
        let mut streets = floored_chain_model();
        streets.group_nav[1].flags |= EXTERIOR_LIT;
        assert!(!rooms_are_indoor(&vis(&[1]), &streets));
        // Naming nothing at all is the empty claim, not an indoor one.
        assert!(!rooms_are_indoor(&vis(&[]), &m));
        // A group index past the model answers the same way — a lookup miss is not "indoors".
        assert!(!rooms_are_indoor(&vis(&[99]), &m));
    }

    /// Outside the building there is no room gate at all — the building's lights are then bounded
    /// by range alone, which is what the open world already does.
    ///
    /// **A narrower answer was tried here and reverted.** Seeding the outdoor shell and flooding a
    /// single portal hop inward stopped an interior torch washing the road, but it also darkened
    /// the room you can see through the abbey's open door, because that room is more than one
    /// portal from the shell. `lit` is one set for the whole frame and the two cases differ by the
    /// receiver, so no verdict here can tell them apart — the bleed belongs to the surface lanes'
    /// indoor/outdoor class test, not to this function. Keep that in mind before narrowing it
    /// again.
    #[test]
    fn lighting_reach_outside_a_building_gates_nothing() {
        let m = floored_chain_model();
        // Standing on the road beside the building — outside every group's own column, which is
        // what actually makes the down-ray answer "no containing group" (being merely high above
        // it does not: the ray still lands on a floor).
        let lit = lighting_reach(&m, [100.0, 0.0, 1.0], None);
        assert!(lit.iter().all(|v| *v), "{lit:?}");
    }

    fn portal_model() -> WmoModel {
        WmoModel {
            wmo_id: 0,
            submeshes: Vec::new(),
            submesh_group: Vec::new(),
            portal_vertices: vec![
                [0.0, -1.0, -1.0],
                [0.0, 1.0, -1.0],
                [0.0, 1.0, 1.0],
                [0.0, -1.0, 1.0],
            ],
            portal_infos: vec![WmoPortalInfo {
                start_vertex: 0,
                count: 4,
                plane: [1.0, 0.0, 0.0, 0.0],
            }],
            portal_refs: Vec::new(),
            group_nav: Vec::new(),
            ambient: [0.0; 3],
            fogs: Vec::new(),
            skybox: None,
            group_collision_tris: Vec::new(),
            group_camera_only_tris: Vec::new(),
            group_collision_bounds: Vec::new(),
            group_collision_grids: Vec::new(),
            collision_bounds: None,
            collision: None,
            collision_camera: None,
            doodads: Vec::new(),
            doodad_sets: Vec::new(),
            lights: Vec::new(),
            group_bounds: Vec::new(),
            group_footprints: Vec::new(),
            material_ground_type: Vec::new(),
            material_diff_color: Vec::new(),
            group_footprint_bounds: Vec::new(),
            group_footprint_grids: Vec::new(),
            group_light_refs: Vec::new(),
            group_liquids: Vec::new(),
            doodad_base: Default::default(),
            doodad_owner: Default::default(),
            doodad_groups: Default::default(),
            convex_volume: Vec::new(),
        }
    }
}
