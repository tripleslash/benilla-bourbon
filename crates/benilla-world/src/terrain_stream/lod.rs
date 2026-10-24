//! Terrain chunk LOD (`systems/terrain/lod.md`, "Chunk levels" on): each chunk draws at level 3, or
//! at level 2 past `lodDist` while `lod` is on ([`HorizonCulling::chunk_level`], `0x6afad0`). Its
//! mesh is keyed by its level and the level of each neighbour that is strictly finer (`0x6beb50`),
//! across cell and tile edges; a level 2 chunk splits the edge it shares with a finer one, so no
//! T-junction cracks (`0x6c51b0`, [`chunk_lod_indices`]). A neighbour tile not resident is not
//! finer.
//!
//! A cell's 4×4 chunks share one mesh, so a key change rebuilds that cell's mesh in place with the
//! same vertices and the new triangle lists ([`chunks_to_mesh_lod`]); a cell whose keys held keeps
//! its mesh untouched, as the reference keeps a chunk's buffers while its key at `+0xbc` holds. The
//! draw only: collision, `height_at` and the pick read the decoded chunks, never these indices.
//!
//! Deviation: the reference rebuilds a changed chunk's indices the frame it draws; a cell rebuild
//! here costs a vertex upload too, so at most [`REBUILDS_PER_FRAME`] cells rebuild a frame, the
//! nearest first, and a far cell may draw its old level for a few frames after a turn.

use std::sync::Arc;

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::{chunks_to_mesh_lod, AdtTile};
use benilla_formats::{chunk_lod_indices, ChunkMesh, CHUNK_LEVEL_NEAR};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::furnish::CELL_SPAN;
use super::{TerrainCell, TerrainStreamer};
use crate::horizon::HorizonCulling;
use crate::view::WorldCamera;

/// Cells rebuilt per frame at most.
const REBUILDS_PER_FRAME: usize = 4;

/// The key of a chunk at the near level with no finer neighbour: the furnished full fan.
const KEY_FULL: u32 = (CHUNK_LEVEL_NEAR as u32) << 16;

/// Each chunk's bounding sphere in Bevy space, `None` for a chunk with no vertices.
type Spheres = Arc<[Option<(Vec3, f32)>]>;

/// How many times a cell's mesh has been rebuilt at a new LOD key under the same handle: the
/// shadow and GI caches, which key a terrain caster on its mesh, read it to know the mesh moved.
#[derive(Component, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct TerrainLodEpoch(pub(crate) u32);

/// What the gate keeps between frames.
#[derive(Default)]
pub(super) struct LodState {
    /// Per resident tile asset, its chunks' spheres, built once.
    spheres: HashMap<AssetId<AdtTile>, Spheres>,
    /// Per cell entity, the key each of its 16 chunks was last built with.
    built: HashMap<Entity, [u32; 16]>,
}

/// A chunk's mesh key (`0x6beb50`): the level at bit 16, and each strictly finer neighbour's
/// level at 0, 4, 8 and 12 in the order row − 1, column + 1, row + 1, column − 1.
fn key_of(level: u8, caps: [u8; 4]) -> u32 {
    caps.iter()
        .enumerate()
        .fold(u32::from(level) << 16, |k, (i, c)| {
            k | (u32::from(*c) << (4 * i))
        })
}

/// The inverse of [`key_of`].
fn unkey(key: u32) -> (u8, [u8; 4]) {
    let level = ((key >> 16) & 0xf) as u8;
    let caps = [0, 1, 2, 3].map(|i| ((key >> (4 * i)) & 0xf) as u8);
    (level, caps)
}

fn spheres_of(tile: &AdtTile) -> Spheres {
    tile.chunks
        .iter()
        .map(|c| {
            let mut pts = c.positions.iter().map(|p| wow_to_bevy(*p));
            let first = pts.next()?;
            let (min, max) = pts.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
            Some(((min + max) * 0.5, (max - min).length() * 0.5))
        })
        .collect()
}

/// The chunk `(row, col)` steps from a tile's chunk, across the tile edge: tile x runs with the
/// chunk column (east), tile y with the row (south).
fn step(tile: (i32, i32), row: i32, col: i32) -> ((i32, i32), usize) {
    let (dy, r) = (row.div_euclid(16), row.rem_euclid(16));
    let (dx, c) = (col.div_euclid(16), col.rem_euclid(16));
    ((tile.0 + dx, tile.1 + dy), (r * 16 + c) as usize)
}

/// Each chunk of a cell's mesh key from the frame's levels.
fn cell_keys(
    levels: &HashMap<(i32, i32), [u8; 256]>,
    cell: &TerrainCell,
    lod_on: bool,
) -> Option<[u32; 16]> {
    let own = levels.get(&cell.tile)?;
    let span = CELL_SPAN as i32;
    let (cy, cx) = (i32::from(cell.cell) / 4, i32::from(cell.cell) % 4);
    let mut keys = [KEY_FULL; 16];
    for (k, key) in keys.iter_mut().enumerate() {
        let (row, col) = (cy * span + k as i32 / span, cx * span + k as i32 % span);
        let level = own[(row * 16 + col) as usize];
        let mut caps = [0u8; 4];
        // The neighbour links are world flag 4's alone (`0x6beb6d`).
        if lod_on {
            for (side, (dr, dc)) in [(-1, 0), (0, 1), (1, 0), (0, -1)].into_iter().enumerate() {
                let (tile, i) = step(cell.tile, row + dr, col + dc);
                let theirs = levels.get(&tile).map_or(0, |l| l[i]);
                if theirs > level {
                    caps[side] = theirs;
                }
            }
        }
        *key = key_of(level, caps);
    }
    Some(keys)
}

/// The cell's chunks in the furnished order, with their triangle lists at `keys`.
fn cell_lists<'a>(adt: &'a AdtTile, cell: &TerrainCell, keys: &[u32; 16]) -> Vec<CellPart<'a>> {
    cell_chunks(cell)
        .enumerate()
        .filter_map(|(k, i)| {
            let chunk = adt.chunks.get(i)?;
            let shading = adt.shading.get(i)?;
            let indices = if keys[k] == KEY_FULL {
                chunk.indices.clone()
            } else {
                let (level, caps) = unkey(keys[k]);
                chunk_lod_indices(level, caps, chunk.holes)
            };
            Some((chunk, shading, indices))
        })
        .collect()
}

type CellPart<'a> = (&'a ChunkMesh, &'a benilla_assets::ChunkShading, Vec<u32>);

/// A cell whose keys moved: its nearest chunk's distance, the entity, the cell, its mesh and the
/// keys to build.
type Due = (f32, Entity, TerrainCell, AssetId<Mesh>, [u32; 16]);

/// Keys every furnished cell's chunks from this frame's levels and rebuilds, nearest first and
/// [`REBUILDS_PER_FRAME`] a frame, the cells whose keys moved. The mesh is replaced under its own
/// handle by a queued command, so the entity, its `Aabb` and every caster list keep it.
pub(super) fn update_terrain_lod(
    mut commands: Commands,
    culling: Option<Res<HorizonCulling>>,
    streamer: Res<TerrainStreamer>,
    tiles: Res<Assets<AdtTile>>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    cells: Query<(Entity, &TerrainCell, &Mesh3d)>,
    mut state: Local<LodState>,
) {
    let Some(cam_t) = cam.iter().next() else {
        return;
    };
    let culling = culling.as_deref().copied().unwrap_or_default();
    let (eye, forward) = (cam_t.translation(), Vec3::from(cam_t.forward()));
    let state = &mut *state;
    // The frame's level of every resident chunk (`0x6afad0`), by its sphere's view depth.
    let mut levels: HashMap<(i32, i32), [u8; 256]> = HashMap::default();
    let mut resident: HashMap<(i32, i32), (&AdtTile, Spheres)> = HashMap::default();
    let mut live = Vec::new();
    for (key, handle) in streamer.spawned_tiles() {
        let Some(adt) = tiles.get(handle) else {
            continue;
        };
        live.push(handle.id());
        let spheres = state
            .spheres
            .entry(handle.id())
            .or_insert_with(|| spheres_of(adt))
            .clone();
        resident.insert(key, (adt, spheres.clone()));
        let mut lv = [CHUNK_LEVEL_NEAR; 256];
        for (l, s) in lv.iter_mut().zip(spheres.iter()) {
            if let Some((center, radius)) = s {
                *l = culling.chunk_level(forward.dot(*center - eye) - radius);
            }
        }
        levels.insert(key, lv);
    }
    state.spheres.retain(|id, _| live.contains(id));
    state.built.retain(|e, _| cells.contains(*e));

    let mut due: Vec<Due> = Vec::new();
    for (entity, cell, mesh) in &cells {
        let Some(keys) = cell_keys(&levels, cell, culling.lod) else {
            continue;
        };
        let built = state.built.get(&entity).copied().unwrap_or([KEY_FULL; 16]);
        if built == keys {
            continue;
        }
        let Some((_, spheres)) = resident.get(&cell.tile) else {
            continue;
        };
        // The cell's nearest chunk sphere orders the rebuilds.
        let near = cell_chunks(cell)
            .filter_map(|i| *spheres.get(i)?)
            .map(|(c, r)| (c.distance(eye) - r).max(0.0))
            .fold(f32::INFINITY, f32::min);
        due.push((near, entity, *cell, mesh.id(), keys));
    }
    due.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    for (_, entity, cell, mesh_id, keys) in due.into_iter().take(REBUILDS_PER_FRAME) {
        let Some((adt, _)) = resident.get(&cell.tile) else {
            continue;
        };
        let lists = cell_lists(adt, &cell, &keys);
        let parts: Vec<_> = lists
            .iter()
            .map(|(c, s, i)| (*c, *s, i.as_slice()))
            .collect();
        state.built.insert(entity, keys);
        let Some(mesh) = chunks_to_mesh_lod(&parts) else {
            continue;
        };
        commands.queue(move |world: &mut World| {
            if let Some(mut meshes) = world.get_resource_mut::<Assets<Mesh>>() {
                // A cell that unloaded since leaves an index nothing holds: nothing to replace.
                if meshes.insert(mesh_id, mesh).is_err() {
                    return;
                }
            }
            // The casters' caches see the new mesh through the cell's epoch.
            if let Ok(mut cell) = world.get_entity_mut(entity) {
                let next = cell
                    .get::<TerrainLodEpoch>()
                    .map_or(1, |e| e.0.wrapping_add(1));
                cell.insert(TerrainLodEpoch(next));
            }
        });
    }
}

/// A cell's chunk indices in its tile, in the furnished order (MCIN row-major, row·16 + column).
fn cell_chunks(cell: &TerrainCell) -> impl Iterator<Item = usize> {
    let (cy, cx) = (usize::from(cell.cell) / 4, usize::from(cell.cell) % 4);
    (0..CELL_SPAN * CELL_SPAN)
        .map(move |k| (cy * CELL_SPAN + k / CELL_SPAN) * 16 + cx * CELL_SPAN + k % CELL_SPAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_holds_the_level_and_the_finer_neighbours() {
        assert_eq!(key_of(CHUNK_LEVEL_NEAR, [0; 4]), KEY_FULL);
        let k = key_of(2, [3, 0, 3, 0]);
        assert_eq!(k, (2 << 16) | 3 | (3 << 8));
        assert_eq!(unkey(k), (2, [3, 0, 3, 0]));
    }

    #[test]
    fn a_step_off_the_tile_lands_in_its_neighbour() {
        assert_eq!(step((10, 20), -1, 5), ((10, 19), 15 * 16 + 5));
        assert_eq!(step((10, 20), 3, 16), ((11, 20), 3 * 16));
        assert_eq!(step((10, 20), 16, 0), ((10, 21), 0));
        assert_eq!(step((10, 20), 2, -1), ((9, 20), 2 * 16 + 15));
        assert_eq!(step((10, 20), 2, 3), ((10, 20), 35));
    }

    /// A far chunk beside a near one in the next tile takes the near one's level in its caps; the
    /// near one, finer, takes none. With `lod` off no chunk links a neighbour.
    #[test]
    fn a_coarse_chunk_caps_its_finer_neighbour_across_the_tile_edge() {
        let mut levels: HashMap<(i32, i32), [u8; 256]> = HashMap::default();
        levels.insert((0, 0), [2; 256]);
        levels.insert((1, 0), [3; 256]);
        // Cell 3 is the tile's north-east cell: its column-3 chunks touch tile (1, 0).
        let cell = TerrainCell {
            tile: (0, 0),
            cell: 3,
        };
        let keys = cell_keys(&levels, &cell, true).expect("a resident tile");
        // Chunk k = 3 is row 0, the cell's east column: east finer.
        assert_eq!(unkey(keys[3]), (2, [0, 3, 0, 0]));
        // Chunk k = 0 is inside the tile: nothing finer.
        assert_eq!(unkey(keys[0]), (2, [0; 4]));
        let east = TerrainCell {
            tile: (1, 0),
            cell: 0,
        };
        let keys = cell_keys(&levels, &east, true).expect("a resident tile");
        assert!(keys.iter().all(|k| *k == KEY_FULL));
        let off = cell_keys(&levels, &cell, false).expect("a resident tile");
        assert!(off.iter().all(|k| unkey(*k).1 == [0; 4]));
        // A tile with no resident neighbour: its edge is not finer.
        assert_eq!(
            unkey(cell_keys(&levels, &cell, true).unwrap()[0]).1,
            [0; 4],
            "row 0 faces tile (0, -1), not resident"
        );
    }
}
