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
//!   own heightfield has not landed gets that shape as its collider: 3,072 triangles against the
//!   real ~145k, exact to about a chunk.
//! - **Buildings** ([`hull_box`]) — the MODF bounding box. Each WMO placement's world-space AABB is
//!   authored in the *tile*, not in the `.wmo`, so a building's mass is known one fetch before its
//!   geometry is. A placement whose model is still loading gets that box.
//!
//! Both are **temporary by construction**: a stand-in is retired the frame its real collider
//! attaches, and the two never coexist. Both sit on [`walk_layers`] — the body's audience only — so
//! the camera boom never rides a phantom and the mouse pick never clamps against one, and neither
//! carries `GroundDecalSurface` or `PickOccluder`. A stand-in is something to stand on, not scenery.
//!
//! ## What this deliberately does not do
//!
//! It does not make an unloaded world *playable*, and it must not be read as permission to release
//! a body onto one. The loading-screen clear and the settle release still key on the real thing
//! (`WorldLoadProgress::focus_collidable`, which counts attached terrain collision and ignores every
//! stand-in) — an arrival still waits for the ground it is actually going to stand on. This is the
//! net under the cases an arrival cannot cover: walking across a boundary, a stall backstop that
//! fires on a dead stream, a cinematic that flies the stream off the body's own ground.
//!
//! Native builds get all of it too, and that is on purpose rather than by accident: the failure it
//! prevents is a race, and a race that is merely improbable on a fast disk is still a race. The
//! machinery costs one map walk per frame over sets that are empty in a settled world.

use std::collections::HashMap;

use avian3d::prelude::Collider;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;

use benilla_assets::coords::wow_to_bevy;

use super::collider::{build_collider_task, PendingCollider};
use super::{Placements, StreamActivity, TerrainStreamer, WorldLoadProgress};
use crate::collision::walk_layers;
use crate::view::Viewer;
use crate::wdl::WdlStreamer;

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

/// How far a stand-in box is inflated when testing whether it already contains the body — see the
/// occupancy guard in [`retire_and_raise_hulls`].
const HULL_CLEARANCE: f32 = 2.0;

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

/// The live stand-ins, by what they stand in for. Both maps are empty in a settled world.
#[derive(Resource, Default)]
pub(super) struct ProvisionalCollision {
    /// Tile coords -> the WDL floor entity holding that tile's place.
    ground: HashMap<(i32, i32), Entity>,
    /// WMO placement uniqueId -> the MODF box entity holding that building's place.
    hulls: HashMap<u32, Entity>,
}

impl ProvisionalCollision {
    /// Drop every stand-in — the world they stood in for is going away (`release_world`).
    pub(super) fn release(&mut self, commands: &mut Commands) {
        for entity in self.ground.values().chain(self.hulls.values()) {
            commands.entity(*entity).try_despawn();
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
pub(super) fn maintain_provisional_collision(
    mut commands: Commands,
    mut prov: ResMut<ProvisionalCollision>,
    streamer: Res<TerrainStreamer>,
    placements: Res<Placements>,
    mut progress: ResMut<WorldLoadProgress>,
    wdl: Option<Res<WdlStreamer>>,
    viewer: Res<Viewer>,
    server: Res<AssetServer>,
    mut activity: ResMut<StreamActivity>,
    // Colliders still building or waiting for their attach budget — the query that answers "has the
    // real thing landed", for a tile root and for a placement's own collider entities alike.
    building: Query<(), With<PendingCollider>>,
) {
    let t0 = Instant::now();
    // The ablation switch takes the stand-ins and nothing else: the two published facts below are
    // about the REAL world and are what the loading screen and the settle release read, so a run
    // with the lever set must still load and reveal exactly as it does without it.
    if !disabled() {
        retire_and_raise_ground(
            &mut commands,
            &mut prov,
            &streamer,
            &progress,
            wdl.as_deref(),
            &building,
        );
    }
    let placements_loading = retire_and_raise_hulls(
        &mut commands,
        &mut prov,
        &placements,
        &viewer,
        &server,
        &building,
    );

    // **The two facts the settle release reads**, published here because this is the one system
    // that has both the focus tile and the query that says whether its collider attached.
    progress.focus_collidable = focus_has_real_collision(&streamer, &progress, &building);
    progress.loads_in_flight = placements_loading + tiles_loading(&streamer, &server);

    activity.provisional_live = (prov.ground.len() + prov.hulls.len()) as u32;
    activity.provisional_ms += t0.elapsed().as_secs_f32() * 1000.0;
}

/// Is the tile under the view focus really collidable — see `WorldLoadProgress::focus_collidable`.
fn focus_has_real_collision(
    streamer: &TerrainStreamer,
    progress: &WorldLoadProgress,
    building: &Query<(), With<PendingCollider>>,
) -> bool {
    let Some(tile) = progress.focus_tile else {
        return false; // pre-world: nothing is under anything yet
    };
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
/// own yet, and nothing anywhere else.
fn retire_and_raise_ground(
    commands: &mut Commands,
    prov: &mut ProvisionalCollision,
    streamer: &TerrainStreamer,
    progress: &WorldLoadProgress,
    wdl: Option<&WdlStreamer>,
    building: &Query<(), With<PendingCollider>>,
) {
    // No focus yet (pre-world, or just after `release_world`) means no neighbourhood to floor — and
    // every floor that exists is over ground nobody is standing on any more.
    let focus = progress.focus_tile;
    let wanted = |tile: (i32, i32)| {
        focus.is_some_and(|(cx, cy)| {
            (tile.0 - cx).abs() <= GROUND_RADIUS && (tile.1 - cy).abs() <= GROUND_RADIUS
        }) && !tile_is_collidable(streamer, tile, building)
    };

    prov.ground.retain(|tile, entity| {
        if wanted(*tile) {
            return true;
        }
        commands.entity(*entity).try_despawn();
        false
    });

    let (Some((cx, cy)), Some(wdl)) = (focus, wdl) else {
        return;
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
            let Some((verts, tris)) = wdl.coarse_collision(tx, ty) else {
                continue;
            };
            let entity = commands
                .spawn((
                    ground_floor(verts, tris),
                    ProvisionalCollider,
                    Transform::IDENTITY,
                ))
                .id();
            prov.ground.insert(tile, entity);
        }
    }
}

/// One tile's stand-in floor, built off-thread like every other streamed collider.
///
/// [`walk_layers`] rather than the default layer: only the body's own trace may see it. A camera
/// boom riding a coarse horizon mesh would swing the shot out over a hillside that is not there,
/// and a mouse pick clamping against one would stop the cursor at a surface with nothing drawn on
/// it.
fn ground_floor(verts: Vec<Vec3>, tris: Vec<[u32; 3]>) -> PendingCollider {
    PendingCollider::new(build_collider_task(verts, tris), Some(walk_layers()), true)
}

/// The building lane: a MODF box for every near placement whose model has not spawned, retired the
/// moment that placement's own colliders are attached.
///
/// Returns how many unspawned placements are still fetching their model — the other half of
/// `WorldLoadProgress::loads_in_flight`, counted here because this is already the walk over
/// `by_id` and a second one would be the same cost twice.
fn retire_and_raise_hulls(
    commands: &mut Commands,
    prov: &mut ProvisionalCollision,
    placements: &Placements,
    viewer: &Viewer,
    server: &AssetServer,
    building: &Query<(), With<PendingCollider>>,
) -> usize {
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
    prov.hulls.retain(|uid, entity| {
        if !settled(uid) {
            return true;
        }
        commands.entity(*entity).try_despawn();
        false
    });

    // Nothing waiting to spawn means nothing can want a new box and nothing can be in flight — the
    // same gate, for the same reason, that lets `spawn_loaded_placements` skip its own full-map
    // walk in a settled world.
    if placements.pending_spawns == 0 {
        return 0;
    }
    // The ablation switch is applied to the box, not to the walk: `loads_in_flight` is a fact about
    // the real world and must read the same either way.
    let boxes_wanted = !disabled();
    let mut loading = 0usize;
    for (uid, placement) in &placements.by_id {
        if placement.spawned || prov.hulls.contains_key(uid) {
            continue;
        }
        if loading < LOADS_SAMPLE_CAP && placement.model.is_loading(server) {
            loading += 1;
        }
        // No body means nothing to keep from falling (a capture run, the world viewer) — but the
        // count above is still worth having, so this is a `continue` on the box rather than an
        // early return from the walk.
        let Some(body) = viewer.at.filter(|_| boxes_wanted) else {
            continue;
        };
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
        // the right answer. Re-tested every frame, so stepping clear puts the box up.
        let clearance = Vec3::splat(HULL_CLEARANCE);
        if body.cmpge(min - clearance).all() && body.cmple(max + clearance).all() {
            continue;
        }
        let entity = commands
            .spawn((
                hull_box(max - min),
                ProvisionalCollider,
                Transform::from_translation((min + max) * 0.5),
            ))
            .id();
        prov.hulls.insert(*uid, entity);
    }
    loading
}

/// One building's stand-in box: a cuboid the size of its authored MODF bounds, on the body's layer
/// only for the same reasons [`ground_floor`] is.
///
/// A box, not the model's real hull, because the whole point is that the hull has not been fetched.
/// It is a coarser answer than the geometry it stands in for and it is *supposed* to be — an
/// over-solid building for a few hundred milliseconds is a bumped shoulder; an under-solid one is a
/// fall through the world.
fn hull_box(size: Vec3) -> PendingCollider {
    let size = size.max(Vec3::splat(MIN_HULL_EXTENT));
    // Built through the same off-thread path as every other streamed collider rather than inserted
    // directly: a cuboid costs nothing to build either way, and going through `PendingCollider`
    // means one attach budget, one epoch bump, and one shape of "a collider this streamer owns".
    PendingCollider::new(
        AsyncComputeTaskPool::get().spawn(async move { Collider::cuboid(size.x, size.y, size.z) }),
        Some(walk_layers()),
        true,
    )
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
}
