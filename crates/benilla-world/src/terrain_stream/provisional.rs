//! **Stand-in collision for geometry that has not arrived yet** — the floor a body gets while the
//! real thing is still on the wire.
//!
//! Every other part of the streamer answers "is it here?" with yes or no, and the movement code was
//! built on the second answer meaning *wait*: the post-snap settle hold freezes the body until the
//! destination is resident (decision 0737), and that is correct for the case it covers — an arrival.
//! It cannot cover the other case, because there is nothing to freeze: **a body that is already
//! walking, into ground that has not landed.** Nothing holds it, and if the collider under the next
//! tile is a fetch away it walks off the edge of the world.
//!
//! On native that gap is a few milliseconds of local `pread` and effectively never observable. On
//! the web the same gap is a network — the BENPACK container's objects come off an origin — and
//! "sometimes falling through the map" is what a few hundred milliseconds of it looks like from
//! inside the game.
//!
//! So the streamer stops answering "not yet" with *nothing*. Two stand-ins, one per kind of thing
//! that can hold a body up, and each is built from a description the client **already has** before
//! the geometry itself is fetched:
//!
//! - **Ground** ([`ground_floor`]) — the map's WDL. It is one 770 KB file, loaded whole at startup
//!   for the horizon band, and it carries a coarse height sample per MCNK for all 4,096 tiles. So
//!   the shape of every tile in the world is known before any `.adt` is asked for, and a tile whose
//!   own heightfield has not landed gets that shape as a **parry heightfield**: 289 samples, exact
//!   to about a chunk.
//! - **Buildings** ([`hull_shape`]) — the best box the client already holds. Before the `.wmo` has
//!   arrived that is the MODF bounding box, authored in the *tile* rather than in the model, so a
//!   building's mass is known one fetch before its geometry is. Once the root has loaded it is the
//!   per-group collision bounds instead — `WmoModel::group_collision_bounds`, already computed at
//!   load — which is a courtyard-shaped answer rather than a brick.
//!
//! Both are **temporary by construction**: a stand-in is retired the frame its real collider
//! attaches, and the two never coexist. Both sit on [`walk_layers`] — the body's audience only — so
//! the camera boom never rides a phantom and the mouse pick never clamps against one, and neither
//! carries `GroundDecalSurface` or `PickOccluder`. A stand-in is something to stand on, not scenery.
//!
//! ## Nothing here may stall the frame, ever
//!
//! This is the law the module is built around, and the first version broke it. Standing in for a
//! tile meant a parry **trimesh** off the WDL mesh — 3,072 triangles and a QBVH — queued on the
//! async compute pool, which on `wasm32` is `spawn_local` on the thread that draws. Crossing a tile
//! boundary at a run raised up to nine of those in one frame, and the fix for a fall became a
//! hitch. Two rules came out of it:
//!
//! 1. **Only what is already in memory.** Nothing here reads the chain, awaits an asset, or waits
//!    on a load. A stand-in that cannot be built from what the client already holds is simply not
//!    built, and the real geometry arrives on its own schedule.
//! 2. **Only shapes that cost nothing to build.** A heightfield stores a lattice and builds no
//!    acceleration structure (a regular grid already is one); a cuboid is six planes; a compound of
//!    a few dozen cuboids is those, plus a list. All three are built **inline** rather than through
//!    `PendingCollider` — there is nothing to defer, and going through the attach queue would put a
//!    phantom into `colliders_pending`, where it would hold up the very reveal it exists to make
//!    safe.
//!
//! ## What this deliberately does not do
//!
//! It does not make an unloaded world *playable*. What it makes safe is the one decision that
//! cannot wait: whether a body may be simulated where it stands. `WorldLoadProgress::focus_collidable`
//! is satisfied by a stand-in, so the settle release and the loading cover stop keying on a fetch —
//! but every other residency term still means what it meant (`scene_ready` is still the real tiles,
//! furnished, with their placements up), so a *reveal* still shows the world rather than a coarse
//! shape of it. The distinction is the whole design: nothing waits for collision, and nothing is
//! ever unsupported.
//!
//! Native builds get all of it too, and that is on purpose rather than by accident: the failure it
//! prevents is a race, and a race that is merely improbable on a fast disk is still a race. The
//! machinery costs one map walk per frame over sets that are empty in a settled world.

use std::collections::HashMap;

use avian3d::prelude::{Collider, RigidBody};
use bevy::platform::time::Instant;
use bevy::prelude::*;

use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};
use benilla_assets::{AdtTile, WmoModel};

use super::collider::PendingCollider;
use super::{ModelHandle, Placements, StreamActivity, TerrainStreamer, WorldLoadProgress};
use crate::collision::{walk_layers, ColliderEpoch};
use crate::view::Viewer;
use crate::wdl::{CoarseFloor, WdlStreamer};

/// Chebyshev tile radius around the view focus that gets a stand-in ground floor.
///
/// One tile — the focus tile and its eight neighbours, the same neighbourhood the scene term
/// already calls "around you". A tile is 533 yards and the fastest thing in the game crosses one in
/// well over half a minute, so a body can never reach the far edge of this set before the streamer
/// has had every chance to fill it; anything wider would be building floors for ground nobody can
/// walk to.
const GROUND_RADIUS: i32 = 1;

/// How far from the body a not-yet-loaded building gets a stand-in box, in yards.
///
/// Bounded because the alternative is a city of phantom boxes: the stream window is several tiles
/// across and a zone load registers thousands of placements at once, all of them "not spawned yet"
/// for a moment. What matters is the handful a body could actually walk into, and at run speed
/// nothing outside 300 yards is reachable inside the time a placement takes to spawn.
const HULL_RADIUS: f32 = 300.0;

/// How near the body a WMO's MODF box must be, in yards, for the building to spawn ahead of the
/// rest of the window: about 14 s at the 7 yd/s run speed.
pub(super) const URGENT_REACH: f32 = 100.0;

/// How far a stand-in box is inflated when testing whether it already contains the body — see the
/// occupancy guard in [`retire_and_raise_hulls`].
const HULL_CLEARANCE: f32 = 2.0;

/// How far under its tile's terrain surface a body must be, in yards, to count as underground: a
/// body on the ground stands at the surface, so a slope's interpolation never reads as buried.
const UNDERGROUND_DEPTH: f32 = 2.0;

/// The floor a degenerate box gets, in yards. `benilla_formats` already refuses a zero-volume MODF
/// entry, so this is the second line rather than the first: a zero-extent cuboid is a shape parry
/// has no good answer for, and one that is merely thin is harmless.
const MIN_HULL_EXTENT: f32 = 0.1;

/// How many in-flight placement loads are worth *counting* before the answer stops changing.
///
/// The consumer of `WorldLoadProgress::loads_in_flight` asks one question — "is anything still
/// coming" — and every additional query past the first is an `AssetServer::load_state` lock and
/// hash lookup. A city arrival has ten thousand unspawned placements, so an honest full count would
/// be ten thousand of those per frame to decide a boolean. Eight is enough to be robust against a
/// handful settling in the same frame, and the published number reads "at least this many".
const LOADS_SAMPLE_CAP: usize = 8;

/// `WOW_NO_PROVISIONAL=1` — put up no stand-ins at all, the whole lane's ablation switch (the same
/// shape as `WOW_NO_DOODAD_BODIES` and `WOW_NO_TILE_CELLS` one file over).
///
/// It is how the raw behaviour stays observable: this module exists to make a race *stop being
/// visible*, and a fix like that is exactly the kind that can rot into doing nothing without anyone
/// noticing. With it set, a body walking into unloaded ground falls through it again, which is the
/// only way to check that the thing being prevented is still preventable.
///
/// Read once. A stand-in already up when the flag is read is not a concern — the flag is read on the
/// first frame of the first world.
fn disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("WOW_NO_PROVISIONAL").is_some())
}

/// Marks a stand-in collider, so anything that needs to tell one from real geometry can.
///
/// Nothing in the collision queries reads it — the separation there is [`walk_layers`], which is a
/// property of the *audience* rather than of the collider's provenance. This is for the instruments,
/// and so a despawn can never take a real collider by mistake.
#[derive(Component)]
pub struct ProvisionalCollider;

/// Which description a building's stand-in was built from — the thing that decides whether a live
/// stand-in is still the best one available.
///
/// A placement's stand-in starts as [`Self::Modf`], because before the `.wmo` is fetched that is
/// the only thing there is. When the root lands, a strictly better answer appears in memory and the
/// stand-in is rebuilt once as [`Self::Groups`]. Recording which one is up is what makes that a
/// single upgrade rather than a per-frame comparison of two colliders.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HullKind {
    /// The tile's authored MODF box: one brick around the whole building.
    Modf,
    /// `WmoModel::group_collision_bounds`: one box per group that has collision faces, so doorways,
    /// courtyards and the space under an arch are open.
    Groups,
}

/// The live stand-ins, by what they stand in for. Both maps are empty in a settled world.
#[derive(Resource, Default)]
pub(super) struct ProvisionalCollision {
    /// Tile coords -> the WDL floor entity holding that tile's place.
    ground: HashMap<(i32, i32), Entity>,
    /// WMO placement uniqueId -> the box entity holding that building's place, and what it was
    /// built from (see [`HullKind`]).
    hulls: HashMap<u32, (Entity, HullKind)>,
}

impl ProvisionalCollision {
    /// The tiles a WDL floor stands in for right now.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(super) fn floored_tiles(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.ground.keys().copied()
    }

    /// Drop every stand-in — the world they stood in for is going away (`release_world`).
    pub(super) fn release(&mut self, commands: &mut Commands) {
        for entity in self
            .ground
            .values()
            .copied()
            .chain(self.hulls.values().map(|(e, _)| *e))
        {
            commands.entity(entity).try_despawn();
        }
        self.ground.clear();
        self.hulls.clear();
    }
}

/// Put up the stand-ins the world currently needs and take down the ones it does not.
///
/// Runs at the **tail** of the streaming chain, which is the only position that works: the head is
/// `finish_colliders`, so a collider that attached this frame is already visible here and its
/// stand-in goes in the same frame rather than one late. Everything it reads was written earlier in
/// the same chain.
#[allow(clippy::too_many_arguments)] // three read-only worlds, two published facts, one instrument
pub(super) fn maintain_provisional_collision(
    mut commands: Commands,
    mut prov: ResMut<ProvisionalCollision>,
    streamer: Res<TerrainStreamer>,
    placements: Res<Placements>,
    mut progress: ResMut<WorldLoadProgress>,
    wdl: Option<Res<WdlStreamer>>,
    viewer: Res<Viewer>,
    server: Res<AssetServer>,
    // The loaded WMO roots, for the per-group upgrade — read, never waited on (rule 1 in the module
    // doc): a root that is not here yet simply leaves its placement on the MODF box.
    wmos: Res<Assets<WmoModel>>,
    mut epoch: ResMut<ColliderEpoch>,
    mut activity: ResMut<StreamActivity>,
    // Colliders still building or waiting for their attach budget — the query that answers "has the
    // real thing landed", for a tile root and for a placement's own collider entities alike.
    building: Query<(), With<PendingCollider>>,
    // The decoded tiles, for the terrain height over the body — read, never waited on.
    adt_tiles: Res<Assets<AdtTile>>,
) {
    let t0 = Instant::now();
    let mut changed = 0usize;
    // The ablation switch takes the stand-ins and nothing else: the two published facts below are
    // about the world's own state and are what the loading screen and the settle release read, so a
    // run with the lever set must still load and reveal exactly as it does without it.
    if !disabled() {
        changed += retire_and_raise_ground(
            &mut commands,
            &mut prov,
            &streamer,
            &progress,
            wdl.as_deref(),
            &building,
        );
    }
    let (placements_loading, hull_changes) = retire_and_raise_hulls(
        &mut commands,
        &mut prov,
        &placements,
        &viewer,
        &server,
        &wmos,
        &building,
    );
    changed += hull_changes;
    // A collider appeared or vanished, so every cached collision answer is dated against a world
    // that no longer exists (`ColliderEpoch`). The streamer's own attach queue stamps itself and
    // `track_collider_removals` stamps despawns, but this lane inserts outside both — it is exactly
    // the "any lane that inserts a collider outside the attach queue" the epoch's doc names.
    if changed > 0 {
        epoch.bump();
    }

    // **The facts the settle release reads**, published here because this is the one system that
    // has the focus tile, the body and the query that says whether a collider attached.
    let in_unsolid_wmo = viewer.at.is_some_and(|body| {
        // Nothing waiting to spawn and no collider in the attach queue: every building is solid.
        (placements.pending_spawns > 0 || progress.colliders_pending > 0)
            && inside_unsolid_wmo(&placements, body, |e| building.contains(e))
    });
    progress.focus_collidable =
        focus_is_collidable(&streamer, &prov, &progress, &building) && !in_unsolid_wmo;
    progress.support_pending = in_unsolid_wmo
        || viewer
            .at
            .is_some_and(|body| ground_pending(&streamer, &server, &adt_tiles, body, &building));
    progress.loads_in_flight = placements_loading + tiles_loading(&streamer, &server);

    activity.provisional_live = (prov.ground.len() + prov.hulls.len()) as u32;
    activity.provisional_ms += t0.elapsed().as_secs_f32() * 1000.0;
}

/// Is there collision under the view focus — see `WorldLoadProgress::focus_collidable`.
///
/// **A stand-in counts.** It did not, once: the fact was defined as "the *real* collider has
/// attached", so a body was held until the tile's own heightfield had been fetched, decoded, built
/// and attached. That is a hard wait on a network, and holding for it is the same mistake as
/// falling through — one is a stall, the other is a fall, and both come from treating "not here
/// yet" as something to sit still for. What the consumers of this actually need is that the body is
/// not released over *nothing*, and a stand-in floor is not nothing.
fn focus_is_collidable(
    streamer: &TerrainStreamer,
    prov: &ProvisionalCollision,
    progress: &WorldLoadProgress,
    building: &Query<(), With<PendingCollider>>,
) -> bool {
    let Some(tile) = progress.focus_tile else {
        return false; // pre-world: nothing is under anything yet
    };
    if prov.ground.contains_key(&tile) {
        return true;
    }
    match streamer.tiles.contains_key(&tile) {
        true => tile_is_collidable(streamer, tile, building),
        // No entry at all is two different things — a tile the map does not author (open ocean, a
        // map edge, a WMO-only map) and a tile whose request has not landed — and the streamer told
        // them apart half a chain ago when it computed residency. Borrow its answer rather than
        // deriving a second, differently-wrong one: `focus_resident` carries the vacuous arm, the
        // WDT gate and the global WMO's spawn all folded in.
        false => progress.focus_resident,
    }
}

/// Is `body` inside the MODF box of a WMO placement whose real collision has not attached — not
/// spawned yet, or a collider of its still building?
///
/// The box is the [`HULL_CLEARANCE`]-inflated one the stand-in's occupancy guard tests: the guard
/// puts up no stand-in around a body already inside, so this is what keeps that body from being
/// taken for one standing on something.
fn inside_unsolid_wmo(
    placements: &Placements,
    body: Vec3,
    building: impl Fn(Entity) -> bool,
) -> bool {
    placements.by_id.values().any(|p| {
        wmo_box_within(p, body, HULL_CLEARANCE)
            && (!p.spawned || p.entities.iter().any(|e| building(*e)))
    })
}

/// Is `body` within `reach` yards of a WMO placement's MODF box (inside it is 0)? `false` for an
/// M2 doodad, whose MDDF authors no box.
pub(super) fn wmo_box_within(p: &super::Placement, body: Vec3, reach: f32) -> bool {
    let Some(bounds) = p.bounds.filter(|_| matches!(p.model, ModelHandle::Wmo(_))) else {
        return false;
    };
    let (min, max) = bevy_aabb(&bounds);
    (min - body).max(body - max).max(Vec3::ZERO).length() <= reach
}

/// Is the terrain-side ground under `body` still arriving: its own tile has no real collider yet,
/// or the body is under that tile's terrain while a neighbouring tile is still loading?
///
/// The second arm covers a WMO no registered placement describes yet. A tile lists the buildings it
/// authors, and Turtle WoW's Ironforge Airfields tunnel lies in `Azeroth_34_41` but is listed by
/// `Azeroth_35_41` alone, so until that tile lands the tunnel is not a placement at all. A body
/// under the terrain is inside some building, and it may be the one still coming.
fn ground_pending(
    streamer: &TerrainStreamer,
    server: &AssetServer,
    adt_tiles: &Assets<AdtTile>,
    body: Vec3,
    building: &Query<(), With<PendingCollider>>,
) -> bool {
    let wow = bevy_to_wow(body);
    let (tx, ty) = benilla_formats::world_to_tile(wow[0], wow[1]);
    let tile = (tx as i32, ty as i32);
    let arriving = streamer
        .tiles
        .get(&tile)
        .is_some_and(|s| !server.load_state(&s.handle).is_failed());
    if arriving && !tile_is_collidable(streamer, tile, building) {
        return true;
    }
    buried(
        body.y,
        super::terrain_height_under(streamer, adt_tiles, body),
    ) && neighbour_tile_unspawned(streamer, server, tile)
}

/// A body at height `y` is under the terrain `surface` of its column; `None` is a hole or a tile
/// not decoded, and neither says the body is buried.
fn buried(y: f32, surface: Option<f32>) -> bool {
    surface.is_some_and(|s| y < s - UNDERGROUND_DEPTH)
}

/// Is a tile of `tile`'s 3x3 neighbourhood requested and not spawned, its placements (registered at
/// spawn) still unknown? A failed load is not coming.
fn neighbour_tile_unspawned(
    streamer: &TerrainStreamer,
    server: &AssetServer,
    tile: (i32, i32),
) -> bool {
    (-1..=1).any(|dx| {
        (-1..=1).any(|dy| {
            streamer
                .tiles
                .get(&(tile.0 + dx, tile.1 + dy))
                .is_some_and(|s| s.entity.is_none() && !server.load_state(&s.handle).is_failed())
        })
    })
}

/// Desired tiles whose `AdtTile` is still on the wire — half of
/// `WorldLoadProgress::loads_in_flight`.
///
/// The whole streamed set, not just the focus neighbourhood: the question is "is this stream
/// alive", and a tile arriving two tiles out answers it exactly as well as one arriving underfoot.
/// Uncapped, unlike the placement half, because the set it walks is the stream window — a few dozen
/// entries, of which only the unspawned ones are asked about at all.
fn tiles_loading(streamer: &TerrainStreamer, server: &AssetServer) -> usize {
    streamer
        .tiles
        .values()
        .filter(|t| t.entity.is_none() && server.load_state(&t.handle).is_loading())
        .count()
}

/// Has this tile got real collision under it — or has it been settled that it never will?
///
/// Both halves matter. A tile whose root is spawned and whose `PendingCollider` has attached is
/// collidable, obviously. A tile whose root is spawned and which never *queued* a collider is
/// collidable too, in the only sense that counts here: `terrain_collider_data` returned `None`
/// because the tile authors no triangles, and there is nothing left to wait for. Treating that as
/// "still coming" would leave a WDL floor up over open ocean forever.
pub(super) fn tile_is_collidable(
    streamer: &TerrainStreamer,
    tile: (i32, i32),
    building: &Query<(), With<PendingCollider>>,
) -> bool {
    streamer
        .tiles
        .get(&tile)
        .and_then(|t| t.entity)
        .is_some_and(|e| !building.contains(e))
}

/// The ground lane: a WDL floor under every tile of the focus neighbourhood that has none of its
/// own yet, and nothing anywhere else. Returns how many floors went up or came down.
fn retire_and_raise_ground(
    commands: &mut Commands,
    prov: &mut ProvisionalCollision,
    streamer: &TerrainStreamer,
    progress: &WorldLoadProgress,
    wdl: Option<&WdlStreamer>,
    building: &Query<(), With<PendingCollider>>,
) -> usize {
    // No focus yet (pre-world, or just after `release_world`) means no neighbourhood to floor — and
    // every floor that exists is over ground nobody is standing on any more.
    let focus = progress.focus_tile;
    let wanted = |tile: (i32, i32)| {
        focus.is_some_and(|(cx, cy)| {
            (tile.0 - cx).abs() <= GROUND_RADIUS && (tile.1 - cy).abs() <= GROUND_RADIUS
        }) && !tile_is_collidable(streamer, tile, building)
    };

    let mut changed = 0usize;
    prov.ground.retain(|tile, entity| {
        if wanted(*tile) {
            return true;
        }
        commands.entity(*entity).try_despawn();
        changed += 1;
        false
    });

    let (Some((cx, cy)), Some(wdl)) = (focus, wdl) else {
        return changed;
    };
    for dx in -GROUND_RADIUS..=GROUND_RADIUS {
        for dy in -GROUND_RADIUS..=GROUND_RADIUS {
            let tile = (cx + dx, cy + dy);
            if prov.ground.contains_key(&tile) || !wanted(tile) {
                continue;
            }
            // Off the 64x64 grid, or a tile this map does not author: nothing to stand on and
            // nothing coming, so no floor. (The WDL's present-set is the map's own; a tile absent
            // there is open ocean or off the edge.)
            let (Ok(tx), Ok(ty)) = (u32::try_from(tile.0), u32::try_from(tile.1)) else {
                continue;
            };
            let Some(floor) = wdl.coarse_floor(tx, ty) else {
                continue;
            };
            let at = floor.at;
            let entity = commands
                .spawn((
                    ground_floor(floor),
                    ProvisionalCollider,
                    Transform::from_translation(at),
                ))
                .id();
            prov.ground.insert(tile, entity);
            changed += 1;
        }
    }
    changed
}

/// One tile's stand-in floor, as the components that make it solid.
///
/// Built **inline**, not through `PendingCollider`: a heightfield is a lattice with no acceleration
/// structure to compute, so there is nothing to defer, and the attach queue is the wrong place for
/// a phantom besides (module doc, rule 2).
///
/// [`walk_layers`] rather than the default layer: only the body's own trace may see it. A camera
/// boom riding a coarse horizon surface would swing the shot out over a hillside that is not there,
/// and a mouse pick clamping against one would stop the cursor at a surface with nothing drawn on
/// it.
fn ground_floor(floor: CoarseFloor) -> impl Bundle {
    (
        Collider::heightfield(floor.heights, floor.scale),
        RigidBody::Static,
        walk_layers(),
    )
}

/// The building lane: the best box the client already holds for every near placement whose real
/// collision has not attached, retired the moment it does.
///
/// Returns `(placements still fetching their model, stand-ins raised or dropped)` — the first is
/// the other half of `WorldLoadProgress::loads_in_flight`, counted here because this is already the
/// walk over `by_id` and a second one would be the same cost twice.
fn retire_and_raise_hulls(
    commands: &mut Commands,
    prov: &mut ProvisionalCollision,
    placements: &Placements,
    viewer: &Viewer,
    server: &AssetServer,
    wmos: &Assets<WmoModel>,
    building: &Query<(), With<PendingCollider>>,
) -> (usize, usize) {
    // A placement is done with its stand-in once it has spawned AND every collider it spawned has
    // attached — `spawned` alone is a frame or more too early, which is exactly the window the
    // attach budget opens (decision 0610). A placement that has been released outright (streamed
    // out from under us) is done with it too.
    let settled = |uid: &u32| {
        placements
            .by_id
            .get(uid)
            .is_none_or(|p| p.spawned && !p.entities.iter().any(|e| building.contains(*e)))
    };
    let mut changed = 0usize;
    prov.hulls.retain(|uid, (entity, _)| {
        if !settled(uid) {
            return true;
        }
        commands.entity(*entity).try_despawn();
        changed += 1;
        false
    });

    // Nothing waiting to spawn means nothing can want a new box and nothing can be in flight — the
    // same gate, for the same reason, that lets `spawn_loaded_placements` skip its own full-map
    // walk in a settled world. A placement whose colliders are merely still *attaching* keeps the
    // stand-in it already has; the walk below is only about raising new ones.
    if placements.pending_spawns == 0 {
        return (0, changed);
    }
    // The ablation switch is applied to the box, not to the walk: `loads_in_flight` is a fact about
    // the world's own state and must read the same either way.
    let boxes_wanted = !disabled();
    let mut loading = 0usize;
    for (uid, placement) in &placements.by_id {
        let live = prov.hulls.get(uid).copied();
        if placement.spawned && live.is_none() {
            // Spawned, and it never needed a stand-in (or its stand-in is already gone). Its real
            // colliders are the streamer's business from here.
            continue;
        }
        if !placement.spawned && loading < LOADS_SAMPLE_CAP && placement.model.is_loading(server) {
            loading += 1;
        }
        // No body means nothing to keep from falling (a capture run, the world viewer) — but the
        // count above is still worth having, so this is a `continue` on the box rather than an
        // early return from the walk.
        let Some(body) = viewer.at.filter(|_| boxes_wanted) else {
            continue;
        };
        // **The best description already in memory**, and never a reason to fetch one: the root is
        // read if it happens to be loaded and the MODF box is used if it is not (module doc, rule
        // 1). A live stand-in of the same kind is left exactly where it is.
        let root = hull_root(placement, wmos);
        let kind = match root {
            Some(_) => HullKind::Groups,
            None => HullKind::Modf,
        };
        if live.is_some_and(|(_, up)| up == kind) {
            continue;
        }
        let Some(bounds) = placement.bounds else {
            continue; // an M2 doodad, or a MODF entry whose box is degenerate
        };
        let (min, max) = bevy_aabb(&bounds);
        if ((min + max) * 0.5).distance(body) > HULL_RADIUS {
            continue;
        }
        // **Never box in a body that is already inside.** A building's authored AABB is bigger than
        // its walls, so a body standing in a doorway — or under an arch, or on the far side of a
        // wall the box swallows — would be inside a solid the mover has to be pushed out of, and
        // being shoved through the world is a worse bug than the one this prevents. The box is a
        // barrier for something approaching from outside; if we are already inside it, waiting is
        // the right answer, and [`inside_unsolid_wmo`] is what makes the body wait. Re-tested
        // every frame, so stepping clear puts the box up.
        //
        // The MODF box is the test either way, including for a `Groups` stand-in: the question is
        // whether the body is somewhere inside this building's footprint, and the per-group answer
        // to that is the union, which is what MODF already is.
        let clearance = Vec3::splat(HULL_CLEARANCE);
        if body.cmpge(min - clearance).all() && body.cmple(max + clearance).all() {
            continue;
        }
        // An upgrade replaces rather than adds: two stand-ins for one building would leave the
        // coarse one solid behind the fine one, which is the brick this exists to stop being.
        if let Some((entity, _)) = live {
            commands.entity(entity).try_despawn();
        }
        let (shape, at) = root
            .and_then(|model| group_hull(model, placement))
            .unwrap_or_else(|| modf_hull(min, max));
        let entity = commands
            .spawn((
                (shape, RigidBody::Static, walk_layers()),
                ProvisionalCollider,
                at,
            ))
            .id();
        prov.hulls.insert(*uid, (entity, kind));
        changed += 1;
    }
    (loading, changed)
}

/// This placement's WMO root **if it is already in memory** — never a load, never a wait.
///
/// `None` for an M2 doodad and for a building whose root has not arrived, which are the same answer
/// from here: there is no finer description to use, so the MODF box stands.
fn hull_root<'a>(placement: &super::Placement, wmos: &'a Assets<WmoModel>) -> Option<&'a WmoModel> {
    match &placement.model {
        ModelHandle::Wmo(handle) => wmos.get(handle),
        ModelHandle::M2(_) => None,
    }
}

/// The coarse stand-in: one cuboid the size of the placement's authored MODF bounds, and where to
/// put it.
///
/// It is a coarser answer than the geometry it stands in for and it is *supposed* to be — an
/// over-solid building for a few hundred milliseconds is a bumped shoulder; an under-solid one is a
/// fall through the world.
///
/// The transform carries **no rotation**, and that is not an oversight: a MODF box is authored
/// world-axis-aligned around the already-rotated building, so turning it by the placement rotation
/// would swing a box that is only correct where it stands.
fn modf_hull(min: Vec3, max: Vec3) -> (Collider, Transform) {
    let size = (max - min).max(Vec3::splat(MIN_HULL_EXTENT));
    (
        Collider::cuboid(size.x, size.y, size.z),
        Transform::from_translation((min + max) * 0.5),
    )
}

/// The fine stand-in: one cuboid per group that actually carries collision faces, as a compound in
/// the model's own space — so the courtyard is a courtyard and the gate is a gate, rather than the
/// whole building being one brick.
///
/// `WmoModel::group_collision_bounds` is the input and it is already computed at load: it is the
/// AABB of each group's **walking-collision triangles**, derived from the faces themselves rather
/// than from an authored MOGI box (which can float above its own floor). Nothing is fetched, parsed
/// or built beyond a few dozen cuboids and a list.
///
/// `None` when the root carries no collision bounds at all — a decorative building with no walking
/// faces, where the MODF brick would be wrong rather than merely coarse.
fn group_hull(model: &WmoModel, placement: &super::Placement) -> Option<(Collider, Transform)> {
    let parts: Vec<_> = model
        .group_collision_bounds
        .iter()
        .flatten()
        .map(|(lo, hi)| {
            let (min, max) = bevy_aabb(&[*lo, *hi]);
            let size = (max - min).max(Vec3::splat(MIN_HULL_EXTENT));
            (
                (min + max) * 0.5,
                Quat::IDENTITY,
                Collider::cuboid(size.x, size.y, size.z),
            )
        })
        .collect();
    // One group is not worth a compound — it IS the cuboid, and parry would wrap it in a tree of
    // one. Zero groups means there is nothing to stand on.
    // The bounds are in the model's OWN space (they are the AABBs of the model-space collision
    // triangles), so unlike the MODF box this one takes the placement transform whole — origin,
    // rotation and all.
    (!parts.is_empty()).then(|| (Collider::compound(parts), placement.transform))
}

/// A WoW-space `[min, max]` pair as a **Bevy-space** `(min, max)`.
///
/// `wow_to_bevy` permutes the axes and flips two of their signs, so a mapped "min" corner is not the
/// mapped box's minimum — both corners go through and the extremes are taken afterwards. Same
/// reasoning, and same shape, as `benilla_formats`' MODF-to-world mapping one layer down.
fn bevy_aabb(bounds: &[[f32; 3]; 2]) -> (Vec3, Vec3) {
    let (a, b) = (wow_to_bevy(bounds[0]), wow_to_bevy(bounds[1]));
    (a.min(b), a.max(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AABB mapping survives the axis flips: a WoW box maps to the box that contains both of
    /// its mapped corners, whichever way round they come out. (Two of three axes are negated, so a
    /// mapping that took the corners in order would produce an inverted box — a "min" above its
    /// "max" — and a cuboid built from `max - min` would have negative extents.)
    #[test]
    fn a_wow_box_maps_to_a_bevy_box_with_its_corners_in_order() {
        let wow = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]];
        let (min, max) = bevy_aabb(&wow);
        assert!(min.cmple(max).all(), "min {min:?} is not below max {max:?}");
        for corner in wow.map(wow_to_bevy) {
            assert!(
                corner.cmpge(min).all() && corner.cmple(max).all(),
                "{corner:?} fell outside the mapped box"
            );
        }
    }

    /// The box's extents are the box's, not half of it: `Collider::cuboid` takes full lengths, and
    /// halving them here would put a body through the wall of every stand-in it met.
    #[test]
    fn a_hull_box_is_as_big_as_its_bounds() {
        let (min, max) = bevy_aabb(&[[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]]);
        let size = max - min;
        assert_eq!(size.max(Vec3::splat(MIN_HULL_EXTENT)), size);
        assert!(size.min_element() >= 10.0, "unexpected extents {size:?}");
    }

    /// Turtle WoW's Ironforge Airfields tunnel (`MediumTunnel.wmo` #881972, listed by
    /// `Azeroth_35_41` only): its MODF box in WoW coords, and a point on its floor 14 yd in from
    /// the Ironforge end, 141 yd under the mountain's terrain (672 there).
    const TUNNEL_BOX: [[f32; 3]; 2] = [[-4888.0, -1630.0, 471.0], [-4846.0, -1426.0, 578.0]];
    const IN_TUNNEL: [f32; 3] = [-4866.0, -1440.0, 531.0];

    /// The streamer chain's tail alone, around a body at `body` (WoW coords) whose own tile has
    /// collidable ground and whose neighbourhood reads fully ready, the latched state of a world
    /// a body is running through.
    fn tail_app(body: [f32; 3]) -> App {
        let (tx, ty) = benilla_formats::world_to_tile(body[0], body[1]);
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()));
        app.init_asset::<WmoModel>()
            .init_asset::<benilla_assets::AdtTile>()
            .init_resource::<ProvisionalCollision>()
            .init_resource::<TerrainStreamer>()
            .init_resource::<Placements>()
            .init_resource::<ColliderEpoch>()
            .init_resource::<StreamActivity>()
            .insert_resource(Viewer {
                at: Some(wow_to_bevy(body)),
                ..Viewer::default()
            })
            .insert_resource(WorldLoadProgress {
                focus_tile: Some((tx as i32, ty as i32)),
                focus_resident: true,
                scene_ready: true,
                total: 1,
                ready: 1,
                complete: true,
                ..WorldLoadProgress::default()
            })
            .add_systems(Update, maintain_provisional_collision);
        app
    }

    /// The tunnel as a registered WMO placement whose model is still on the wire.
    fn tunnel() -> super::super::Placement {
        super::super::Placement {
            uid: 881_972,
            ident: super::super::PlacementIdent::new(
                "World\\wmo\\Dungeon\\MD_Dwarven Tunnels\\MediumTunnel.wmo",
                [-4868.0, -1629.0, 505.0],
                [-0.33, 179.5, -8.67],
                1.0,
                0,
            ),
            model: ModelHandle::Wmo(Handle::default()),
            transform: Transform::IDENTITY,
            entities: Vec::new(),
            spawned: false,
            doodad_set: 0,
            name_set: 0,
            doodads: Vec::new(),
            portal_instance: None,
            refs: 1,
            owner: (0, 0),
            bounds: Some(TUNNEL_BOX),
        }
    }

    fn file(app: &mut App, placement: super::super::Placement) {
        let mut pl = app.world_mut().resource_mut::<Placements>();
        pl.pending_spawns = usize::from(!placement.spawned);
        pl.by_id.insert(placement.uid, placement);
    }

    fn presentable(app: &App) -> bool {
        app.world().resource::<WorldLoadProgress>().presentable()
    }

    /// A body inside a tunnel whose WMO has not landed stands on nothing: the terrain above it is
    /// collidable, the tunnel floor is not there, and no stand-in boxes in a body already inside.
    /// The world must not read as safe to release a body into.
    #[test]
    fn a_body_inside_a_wmo_still_on_the_wire_is_not_supported() {
        let mut app = tail_app(IN_TUNNEL);
        file(&mut app, tunnel());
        app.update();
        assert!(
            !presentable(&app),
            "the body inside the unloaded tunnel read as presentable: it would fall"
        );
        assert!(app.world().resource::<WorldLoadProgress>().support_pending);
    }

    /// Spawned is not solid: until the tunnel's collider attaches, the body still has nothing.
    #[test]
    fn a_wmo_whose_collider_is_building_still_holds_nothing_up() {
        let mut app = tail_app(IN_TUNNEL);
        let task = bevy::tasks::AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new)
            .spawn(async { Collider::sphere(1.0) });
        let collider = app
            .world_mut()
            .spawn(PendingCollider::new(task, Some(walk_layers()), true))
            .id();
        let mut solid_later = tunnel();
        solid_later.spawned = true;
        solid_later.entities = vec![collider];
        file(&mut app, solid_later);
        app.world_mut()
            .resource_mut::<WorldLoadProgress>()
            .colliders_pending = 1;
        app.update();
        let progress = app.world().resource::<WorldLoadProgress>();
        assert!(!progress.focus_collidable && progress.support_pending);

        // The collider attaches: the tunnel floor is real and nothing is pending.
        app.world_mut()
            .entity_mut(collider)
            .remove::<PendingCollider>();
        app.world_mut()
            .resource_mut::<WorldLoadProgress>()
            .colliders_pending = 0;
        app.update();
        assert!(presentable(&app));
        assert!(!app.world().resource::<WorldLoadProgress>().support_pending);
    }

    /// Outside the box the stand-in hull is the answer, so a body walking past an unloaded
    /// building is neither unsupported nor held.
    #[test]
    fn a_body_outside_the_unloaded_wmo_is_not_held_for_it() {
        let mut app = tail_app([-4866.0, -1380.0, 505.0]);
        file(&mut app, tunnel());
        app.update();
        assert!(presentable(&app));
        assert!(!app.world().resource::<WorldLoadProgress>().support_pending);
    }

    /// Buried is under the surface by more than a slope's error; a hole or an undecoded tile is no
    /// surface at all.
    #[test]
    fn only_a_body_well_under_the_terrain_is_buried() {
        assert!(
            buried(531.0, Some(672.0)),
            "the tunnel floor under the mountain"
        );
        assert!(!buried(683.0, Some(683.0)), "standing on the ground");
        assert!(!buried(682.0, Some(683.0)), "a slope's interpolation");
        assert!(!buried(505.0, None), "a cave mouth's hole");
    }
}
