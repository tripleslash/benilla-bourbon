//! ADT terrain to per-chunk triangle meshes, in raw WoW world coords (+X north, +Y west, +Z up).
//!
//! Each MCNK is the reference's 145-vertex centre fan: 81 outer corner verts and 64 inner verts at
//! the cell centres, each inner one with its own authored MCVT height, and four triangles per cell
//! fanning from the centre. The arrays keep MCVT's stride-17 order: outer `(r,c)` at `r·17+c`,
//! inner at `r·17+9+c`. A hole (`MCNK+0x3C`) omits a 2×2-cell block's triangles; its verts stay.

use std::io::Cursor;

use crate::Chain;
use anyhow::{Context, Result};
use benilla_adt::{parse_adt, CombinedAlphaMap, ParsedAdt};
use benilla_wdt::{version::WowVersion, WdtFile, WdtReader};

/// Edge length (texels) of a chunk's combined alpha map (always 64×64 in vanilla).
pub const ALPHA_MAP_SIZE: u32 = 64;

/// Edge length (texels) of a chunk's MCSH shadow map (always 64×64 in vanilla).
pub const SHADOW_MAP_SIZE: u32 = 64;

/// Yards per ADT tile (1/64 of a map edge).
pub const TILE_SIZE: f32 = 533.333_3;
/// Yards per MCNK chunk (16×16 per tile), also the grid the impassable-chunk wall runs on.
pub const CHUNK_SIZE: f32 = TILE_SIZE / 16.0;
/// Yards between adjacent outer-grid vertices; the liquid mesher shares the lattice.
pub(crate) const UNIT_SIZE: f32 = CHUNK_SIZE / 8.0;

/// World (x, y) of Stormwind, from `TaxiNodes.dbc`.
pub const STORMWIND_XY: (f32, f32) = (-8840.56, 489.7);

/// Ground-texture repeats per chunk, one per cell: the reference's fixed 8, not a setting.
pub const TERRAIN_LAYER_TILES: f32 = 8.0;

/// One MCNK as an indexed triangle mesh in raw WoW coords, plus its texturing data.
#[derive(Debug, Clone)]
pub struct ChunkMesh {
    /// Vertex positions in WoW yards, 145 in stride-17 order.
    pub positions: Vec<[f32; 3]>,
    /// Unit normals from MCNR, authored continuous across chunk borders; empty without MCNR.
    pub normals: Vec<[f32; 3]>,
    /// Texture coords, 0..1 across the chunk; the layers scale them by [`TERRAIN_LAYER_TILES`].
    pub uvs: Vec<[f32; 2]>,
    /// Triangle-list indices: four fan triangles per cell, holed blocks omitted.
    pub indices: Vec<u32>,
    /// Low-res hole bitmap (`MCNK+0x3C`): bit `(cellY>>1)·4 + (cellX>>1)` drops that 2×2 block.
    pub holes: u16,
    pub base_texture: Option<String>,
    /// MTEX filenames of the layers (up to 4), layer 0 first.
    pub layer_textures: Vec<String>,
    /// Per-layer MCLY `effectId`, a `GroundEffectTexture.dbc` id (`0xFFFFFFFF` for none).
    pub layer_effect_ids: Vec<u32>,
    /// RGBA alpha map (`ALPHA_MAP_SIZE`²), R/G/B for layers 1/2/3; `None` with a single layer.
    pub alpha_map: Option<Vec<u8>>,
    /// The baked MCSH shadow, a byte per texel over `SHADOW_MAP_SIZE`² (255 shadowed, 0 lit).
    pub shadow: Option<Vec<u8>>,
    /// `predominantTexture` (MCNK `0x40`, 2 bits per cell): the layer whose `effectId` picks each
    /// cell's clutter, `k = row*8 + col`.
    pub pred_tex: [u8; 64],
    /// `noEffectDoodad` (MCNK `0x50`, 1 bit per cell): no clutter on that cell. The ADT header's
    /// names for these two grids mislead; `adt_to_tile_mesh` reassembles them.
    pub no_effect_doodad: [bool; 64],
    /// The chunk's `(IndexX, IndexY)` in its tile, which seeds the reference's clutter PRNG.
    pub index_x: u32,
    pub index_y: u32,
    /// The chunk's `AreaTable.dbc` id (MCNK `+0x34`), the zone or subzone underfoot.
    pub area_id: u32,
    /// MCNK flag bit 1 ([`benilla_adt::MCNK_IMPASSABLE`]): an invisible wall over the whole chunk
    /// at every height, where the reference stops a mover.
    pub impassable: bool,
    /// The chunk's MCLQ liquid surfaces, one per block; a river mouth has a stream and a sea.
    pub liquids: Vec<crate::liquid::LiquidMesh>,
}

impl ChunkMesh {
    /// Whether a raw WoW position is in this chunk's MCSH shadow, by the texel the terrain shader
    /// samples; `None` off the chunk's footprint, and a chunk with no MCSH is lit.
    pub fn mcsh_shadowed_at(&self, wow: [f32; 3]) -> Option<bool> {
        let nw = *self.positions.first()?;
        let south = (nw[0] - wow[0]) / TILE_SIZE * 16.0;
        let east = (nw[1] - wow[1]) / TILE_SIZE * 16.0;
        if !(0.0..=1.0).contains(&south) || !(0.0..=1.0).contains(&east) {
            return None;
        }
        let Some(shadow) = self.shadow.as_ref() else {
            return Some(false); // inside this chunk, no MCSH: lit
        };
        let n = SHADOW_MAP_SIZE as usize;
        let row = ((south * n as f32) as usize).min(n - 1);
        let col = ((east * n as f32) as usize).min(n - 1);
        Some(shadow.get(row * n + col).copied().unwrap_or(0) >= 128)
    }

    /// The `GroundEffectTexture` id under a raw WoW position: the `effectId` of the cell's
    /// `predominantTexture` layer, the grid the clutter scatters by.
    pub fn ground_effect_at(&self, wow: [f32; 3]) -> Option<u32> {
        let nw = *self.positions.first()?;
        let south = (nw[0] - wow[0]) / TILE_SIZE * 16.0;
        let east = (nw[1] - wow[1]) / TILE_SIZE * 16.0;
        if !(0.0..=1.0).contains(&south) || !(0.0..=1.0).contains(&east) {
            return None;
        }
        let row = ((south * 8.0) as usize).min(7);
        let col = ((east * 8.0) as usize).min(7);
        let layer = self.pred_tex[row * 8 + col] as usize;
        let effect = self.layer_effect_ids.get(layer).copied()?;
        (effect != u32::MAX).then_some(effect)
    }

    /// The chunk's `AreaTable` id if the raw WoW position is on its footprint.
    pub fn area_at(&self, wow: [f32; 3]) -> Option<u32> {
        let nw = *self.positions.first()?;
        let south = (nw[0] - wow[0]) / TILE_SIZE * 16.0;
        let east = (nw[1] - wow[1]) / TILE_SIZE * 16.0;
        if (0.0..=1.0).contains(&south) && (0.0..=1.0).contains(&east) {
            Some(self.area_id)
        } else {
            None
        }
    }

    /// Whether the chunk is impassable, `None` off its footprint; the flag has no height.
    pub fn impassable_at(&self, wow: [f32; 3]) -> Option<bool> {
        let nw = *self.positions.first()?;
        let south = (nw[0] - wow[0]) / TILE_SIZE * 16.0;
        let east = (nw[1] - wow[1]) / TILE_SIZE * 16.0;
        ((0.0..=1.0).contains(&south) && (0.0..=1.0).contains(&east)).then_some(self.impassable)
    }

    /// fork: per vertex, how far the ground lies under the chunk's own MCLQ surface, in yards:
    /// positive under water, negative above it, `f32::NEG_INFINITY` where no wet cell touches the
    /// vertex. An outer vertex reads the liquid vertex it shares the lattice with; an inner one the
    /// mean of its cell's four. Linear over a fan triangle, as the ground and the surface both are,
    /// so the interpolated value is the depth at every point of a wet cell.
    pub fn water_depths(&self) -> Vec<f32> {
        let mut depths = vec![f32::NEG_INFINITY; self.positions.len()];
        if self.positions.len() != 145 {
            return depths;
        }
        for liquid in &self.liquids {
            if liquid.grid != [9, 9] || liquid.positions.len() != 81 || liquid.wet.len() != 64 {
                continue;
            }
            let wet = |r: usize, c: usize| r < 8 && c < 8 && liquid.wet[r * 8 + c];
            let surface = |r: usize, c: usize| liquid.positions[r * 9 + c][2];
            for r in 0..9usize {
                for c in 0..9usize {
                    // The (up to) four cells meeting at this corner; `wrapping_sub` sends the
                    // row or column before the first out of range.
                    let (rn, cw) = (r.wrapping_sub(1), c.wrapping_sub(1));
                    if wet(rn, cw) || wet(rn, c) || wet(r, cw) || wet(r, c) {
                        let i = r * 17 + c;
                        depths[i] = depths[i].max(surface(r, c) - self.positions[i][2]);
                    }
                    if wet(r, c) {
                        let i = r * 17 + 9 + c;
                        let mean = 0.25
                            * (surface(r, c)
                                + surface(r, c + 1)
                                + surface(r + 1, c)
                                + surface(r + 1, c + 1));
                        depths[i] = depths[i].max(mean - self.positions[i][2]);
                    }
                }
            }
        }
        depths
    }

    /// The terrain height under a raw WoW column, interpolated on the covering fan triangle. `None`
    /// off the footprint or in a hole: a cave mouth's column goes to the WMO below it, where the
    /// reference's terrain probe misses too.
    pub fn height_at(&self, wow: [f32; 3]) -> Option<f32> {
        let nw = *self.positions.first()?;
        let south = (nw[0] - wow[0]) / TILE_SIZE * 16.0;
        let east = (nw[1] - wow[1]) / TILE_SIZE * 16.0;
        if !(0.0..=1.0).contains(&south) || !(0.0..=1.0).contains(&east) {
            return None;
        }
        // Only the covering cell's four fan triangles, with the corners, order and hole rule
        // `adt_to_tile_mesh` indexes, so the answer is the full walk's.
        if self.positions.len() == 145 {
            let row = ((south * 8.0) as u32).min(7);
            let col = ((east * 8.0) as u32).min(7);
            if self.holes & (1u16 << ((row >> 1) * 4 + (col >> 1))) != 0 {
                return None;
            }
            let tl = (row * 17 + col) as usize;
            let (tr, bl, br, ctr) = (tl + 1, tl + 17, tl + 18, (row * 17 + 9 + col) as usize);
            let p = |i: usize| self.positions[i];
            return [
                [p(ctr), p(tl), p(bl)],
                [p(ctr), p(bl), p(br)],
                [p(ctr), p(br), p(tr)],
                [p(ctr), p(tr), p(tl)],
            ]
            .iter()
            .find_map(|tri| triangle_z_at(tri, wow[0], wow[1]));
        }
        self.indices.as_chunks::<3>().0.iter().find_map(|t| {
            let tri = [
                *self.positions.get(t[0] as usize)?,
                *self.positions.get(t[1] as usize)?,
                *self.positions.get(t[2] as usize)?,
            ];
            triangle_z_at(&tri, wow[0], wow[1])
        })
    }
}

/// A chunk's near level (`[0x867964]`, `maxLOD`): the full 145-vertex fan, 256 triangles.
pub const CHUNK_LEVEL_NEAR: u8 = 3;
/// A chunk's far level (`[0x867968]`): 41 of the 145 vertices, 64 triangles.
pub const CHUNK_LEVEL_FAR: u8 = 2;

/// A chunk's triangle list at `level` over its own 145 vertices (stride-17 order), the index
/// builder `0x6c51b0`. A leaf is a cell of four triangles around its centre, (centre, from, to) on
/// the west, south, east and north edges, the winding of the full fan. At level 3 a leaf is one
/// quad around its inner vertex; at level 2 it is a 2×2 block around the outer vertex at its
/// middle, its corners the centre ± 17 ± 1, and a set hole bit (`(row>>1)·4 + (col>>1)`, a 2×2
/// block) drops it at either level. `caps` is the four neighbours' levels where finer, else 0, in
/// the order row − 1, column + 1, row + 1, column − 1 (`+0xc4` to `+0xd0`): a border leaf whose
/// neighbour on that side is finer splits the edge at its middle (`0x6c5304` and its three
/// siblings), so the finer chunk's edge vertices are all met and no crack opens. A level 3 chunk has
/// no finer neighbour and never splits. Leaves are emitted row by row rather than in the
/// quadtree's order, which draws the same triangles.
pub fn chunk_lod_indices(level: u8, caps: [u8; 4], holes: u16) -> Vec<u32> {
    let (cells, step) = if level >= CHUNK_LEVEL_NEAR {
        (8u32, 1u32)
    } else {
        (4u32, 2u32)
    };
    let finer = |side: usize| caps[side] > level;
    let mut out = Vec::with_capacity((cells * cells * 12) as usize + 4 * cells as usize * 3);
    for row in 0..cells {
        for col in 0..cells {
            let hole_bit = if step == 1 {
                (row >> 1) * 4 + (col >> 1)
            } else {
                row * 4 + col
            };
            if holes & (1u16 << hole_bit) != 0 {
                continue;
            }
            // The leaf's corners and centre: level 3 around the inner vertex 9 past the corner,
            // level 2 around the outer vertex at the block's middle.
            let (tl, ctr) = if step == 1 {
                let tl = row * 17 + col;
                (tl, tl + 9)
            } else {
                let ctr = (2 * row + 1) * 17 + 2 * col + 1;
                (ctr - 18, ctr)
            };
            let tr = tl + step;
            let bl = tl + 17 * step;
            let br = bl + step;
            // The four edges and the caps side each faces when the leaf is on the border.
            let edges = [
                (tl, bl, col == 0 && finer(3)),
                (bl, br, row == cells - 1 && finer(2)),
                (br, tr, col == cells - 1 && finer(1)),
                (tr, tl, row == 0 && finer(0)),
            ];
            for (from, to, split) in edges {
                if split {
                    let mid = (from + to) / 2;
                    out.extend_from_slice(&[ctr, from, mid, ctr, mid, to]);
                } else {
                    out.extend_from_slice(&[ctr, from, to]);
                }
            }
        }
    }
    out
}

/// The `z` of `tri` at column `(x, y)`, `None` outside its XY projection or for a vertical face.
/// Shared by the WMO down-ray and the terrain probe, the two legs of the reference's arbitration.
pub fn triangle_z_at(tri: &[[f32; 3]; 3], x: f32, y: f32) -> Option<f32> {
    let (a, b, c) = (tri[0], tri[1], tri[2]);
    let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if det.abs() < 1.0e-9 {
        return None; // vertical / degenerate in XY
    }
    let l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det;
    let l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det;
    let l3 = 1.0 - l1 - l2;
    if l1 < 0.0 || l2 < 0.0 || l3 < 0.0 {
        return None;
    }
    Some(l1 * a[2] + l2 * b[2] + l3 * c[2])
}

/// The `AreaTable` id under a raw WoW position, from one tile's chunks; `None` off the tile.
pub fn area_id_at(chunks: &[ChunkMesh], wow: [f32; 3]) -> Option<u32> {
    chunks.iter().find_map(|c| c.area_at(wow))
}

/// Whether a raw WoW position is over an impassable MCNK of one tile. `None` off the tile means
/// unknown, never passable: walls are authored right across tile seams.
pub fn impassable_at(chunks: &[ChunkMesh], wow: [f32; 3]) -> Option<bool> {
    chunks.iter().find_map(|c| c.impassable_at(wow))
}

/// The `GroundEffectTexture` id under a raw WoW position, from one tile's chunks.
pub fn ground_effect_at(chunks: &[ChunkMesh], wow: [f32; 3]) -> Option<u32> {
    chunks.iter().find_map(|c| c.ground_effect_at(wow))
}

/// The terrain height under a raw WoW column from one tile's chunks, `None` off the tile or in a
/// hole: the terrain leg of the reference's down-ray arbitration.
pub fn terrain_height_at(chunks: &[ChunkMesh], wow: [f32; 3]) -> Option<f32> {
    match chunk_at(chunks, wow) {
        Some(c) => c.height_at(wow),
        None => chunks.iter().find_map(|c| c.height_at(wow)),
    }
}

/// The MCNK of a full tile that covers `wow`'s column, by arithmetic: chunks are in MCIN order
/// (rows step south along x, columns east along y), and chunk 0's first vertex is the tile's
/// north-west corner. `None` sends the caller to the linear walk.
fn chunk_at(chunks: &[ChunkMesh], wow: [f32; 3]) -> Option<&ChunkMesh> {
    if chunks.len() != 256 {
        return None;
    }
    let nw = *chunks[0].positions.first()?;
    let row = (nw[0] - wow[0]) / CHUNK_SIZE;
    let col = (nw[1] - wow[1]) / CHUNK_SIZE;
    if !(0.0..16.0).contains(&row) || !(0.0..16.0).contains(&col) {
        return None;
    }
    chunks.get(row as usize * 16 + col as usize)
}

/// Whether a raw WoW position is in MCSH shadow, from one tile's chunks; `None` off the tile,
/// never "lit": a doodad straddling tiles is listed in each, and only its origin's tile answers,
/// as the reference's global world-to-chunk lookup (`0x69b350`) does. It picks a map doodad's
/// static shade, 1.0 lit or 0.5 shadowed; the 2.5 lit value belongs to an entity's light node.
pub fn mcsh_shadowed_at(chunks: &[ChunkMesh], wow: [f32; 3]) -> Option<bool> {
    match chunk_at(chunks, wow) {
        Some(c) => c.mcsh_shadowed_at(wow),
        None => chunks.iter().find_map(|c| c.mcsh_shadowed_at(wow)),
    }
}

/// A placed M2 doodad (tree, bush, prop): which model, and where, in raw WoW world coords.
#[derive(Debug, Clone)]
pub struct Doodad {
    /// MMDX model path (`.mdx`); `load_m2_mesh` normalizes it to `.m2`.
    pub model: String,
    /// Position in WoW world coords (X north, Y west, Z up), converted from MDDF.
    pub position: [f32; 3],
    /// Euler rotation in degrees (X, Y, Z); Y is the heading about the up axis.
    pub rotation: [f32; 3],
    /// Uniform scale (MDDF stores it ×1024).
    pub scale: f32,
    /// MDDF uniqueId, shared by every tile a doodad straddles; the reference dedups by it
    /// (`0x694bc0`).
    pub unique_id: u32,
}

/// A placed WMO (building): which model, and where, in raw WoW world coords.
#[derive(Debug, Clone)]
pub struct WmoInstance {
    /// MWMO root path (`.wmo`); `load_wmo` loads its group files too.
    pub model: String,
    /// Position in WoW world coords (X north, Y west, Z up), converted from MODF.
    pub position: [f32; 3],
    /// Euler rotation in degrees (X, Y, Z); Y is the heading about the up axis.
    pub rotation: [f32; 3],
    /// MODF uniqueId, the same in every tile a building straddles, deduped by it (`0x694bc0`).
    pub unique_id: u32,
    /// MODF doodad-set index: the set shown beside the always-on set 0.
    pub doodad_set: u16,
    /// MODF name-set index, the placement's `WMOAreaTable.NameSetID`.
    pub name_set: u16,
    /// The placement's authored **world-space bounding box** in WoW coords — `[min, max]`, mapped
    /// through the same MODF→world transform as [`Self::position`]. `None` where the entry authors
    /// a degenerate box (nothing to bound).
    ///
    /// Known from the *tile*, so it exists before the `.wmo` has been fetched — which is what makes
    /// it useful: a building's mass can be stood on while its real hull is still on the wire
    /// (`benilla_world::terrain_stream::provisional`).
    pub bounds: Option<[[f32; 3]; 2]>,
}

/// One ADT tile: its chunk meshes, placed doodads and buildings.
#[derive(Debug, Clone)]
pub struct TileMesh {
    pub chunks: Vec<ChunkMesh>,
    pub doodads: Vec<Doodad>,
    pub wmos: Vec<WmoInstance>,
}

impl TileMesh {
    pub fn vertex_count(&self) -> usize {
        self.chunks.iter().map(|c| c.positions.len()).sum()
    }
}

/// Snap an X/Y world coordinate to the global vertex lattice (`MAP_CENTER − n·UNIT_SIZE`): each
/// MCNK's `f32` corner is off by a millimetre or two, which cracks and z-fights shared edges.
pub(crate) fn snap_to_lattice(coord: f32) -> f32 {
    let map_center = 32.0_f64 * 1600.0 / 3.0; // 64-tile map; TILE_SIZE = 1600/3 yd
    let unit = (1600.0_f64 / 3.0) / 128.0; // 128 vertex-units per tile edge
    let idx = ((map_center - f64::from(coord)) / unit).round();
    (map_center - idx * unit) as f32
}

/// Build per-chunk meshes from one vanilla (monolithic) ADT file's bytes.
pub fn adt_to_tile_mesh(adt_bytes: &[u8]) -> Result<TileMesh> {
    let mut cursor = Cursor::new(adt_bytes);
    let parsed = parse_adt(&mut cursor).map_err(|e| anyhow::anyhow!("parsing ADT: {e}"))?;
    // A vanilla ADT is always one monolithic root.
    let ParsedAdt::Root(root) = parsed;

    let mut chunks = Vec::new();
    for mcnk in &root.mcnk_chunks {
        let Some(mcvt) = &mcnk.heights else {
            continue; // no MCVT height data for this chunk
        };
        if mcvt.heights.len() < 145 {
            continue;
        }
        // The header's `position` is already WoW `[X north, Y west, Z up]`.
        let [wx, wy, wz] = mcnk.header.position;

        // MCNR is one normal per MCVT vertex, its bytes WoW's `[X, Y, Z]` (`0x6aff10`).
        let mcnr = mcnk.normals.as_ref().filter(|n| n.normals.len() >= 145);

        // Each row's 9 outer verts, then its 8 inner ones half a cell in, horizontally only; the
        // last row has no inner verts.
        let mut positions = Vec::with_capacity(145);
        let mut normals = Vec::with_capacity(if mcnr.is_some() { 145 } else { 0 });
        let mut uvs = Vec::with_capacity(145);
        for row in 0..9u32 {
            for col in 0..9u32 {
                let idx = (row * 17 + col) as usize;
                let h = mcvt.heights[idx];
                positions.push([
                    // X/Y snap so shared edge verts coincide; Z stays the MCVT height.
                    snap_to_lattice(wx - row as f32 * UNIT_SIZE), // WoW X (north): rows step south
                    snap_to_lattice(wy - col as f32 * UNIT_SIZE), // WoW Y (west): columns step east
                    wz + h,                                       // WoW Z (up): height over base
                ]);
                if let Some(mcnr) = mcnr {
                    normals.push(mcnr.normals[idx].to_normalized());
                }
                uvs.push([col as f32 / 8.0, row as f32 / 8.0]);
            }
            if row < 8 {
                for col in 0..8u32 {
                    let idx = (row * 17 + 9 + col) as usize;
                    let h = mcvt.heights[idx];
                    // Not snapped: no other chunk shares an inner vert.
                    positions.push([
                        wx - (row as f32 * UNIT_SIZE + UNIT_SIZE / 2.0),
                        wy - (col as f32 * UNIT_SIZE + UNIT_SIZE / 2.0),
                        wz + h,
                    ]);
                    if let Some(mcnr) = mcnr {
                        normals.push(mcnr.normals[idx].to_normalized());
                    }
                    uvs.push([(col as f32 + 0.5) / 8.0, (row as f32 + 0.5) / 8.0]);
                }
            }
        }

        // Four fan triangles per cell, CCW from above (front face up in Bevy under the det +1
        // transform); a set hole bit drops its 2×2 block, and 1.12 has no high-res holes.
        let holes = mcnk.header.holes_low_res;
        let indices = chunk_lod_indices(CHUNK_LEVEL_NEAR, [0; 4], holes);

        // Name and effect id together, so the two vecs stay aligned when a layer is dropped.
        let layers: Vec<(String, u32)> = mcnk
            .layers
            .as_ref()
            .map(|mcly| {
                mcly.layers
                    .iter()
                    .filter_map(|layer| {
                        root.textures
                            .get(layer.texture_id as usize)
                            .cloned()
                            .map(|t| (t, layer.effect_id))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let layer_textures: Vec<String> = layers.iter().map(|(t, _)| t.clone()).collect();
        let layer_effect_ids: Vec<u32> = layers.iter().map(|(_, e)| *e).collect();
        let base_texture = layer_textures.first().cloned();

        // Layers 1..3 into one RGBA map; 1.12 alpha is 4-bit, last row and column duplicated.
        let alpha_map = (layer_textures.len() > 1).then(|| {
            CombinedAlphaMap::new(
                mcnk, /* has_big_alpha */ false, /* fix_alpha */ true,
            )
            .as_slice()
            .to_vec()
        });

        let shadow = mcnk.shadow.as_ref().map(|sh| {
            let n = SHADOW_MAP_SIZE as usize;
            let mut out = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    if sh.is_shadowed(x, y) {
                        out[y * n + x] = 255;
                    }
                }
            }
            out
        });

        // predominantTexture (`0x40..0x4F`) is the header's `pred_tex` then `no_effect_doodad`,
        // and noEffectDoodad (`0x50..0x57`) its `unknown_8bytes`: 2 and 1 bits per cell, LSB first.
        let h = &mcnk.header;
        let mut pt16 = [0u8; 16];
        pt16[..8].copy_from_slice(&h.pred_tex);
        pt16[8..].copy_from_slice(&h.no_effect_doodad);
        let ned8 = h.unknown_8bytes;
        let mut pred_tex = [0u8; 64];
        let mut no_effect_doodad = [false; 64];
        for k in 0..64 {
            pred_tex[k] = (pt16[k / 4] >> (2 * (k % 4))) & 0x3;
            no_effect_doodad[k] = (ned8[k / 8] >> (k % 8)) & 0x1 == 1;
        }

        let liquids: Vec<_> = mcnk
            .liquids
            .iter()
            .filter_map(|mclq| crate::liquid::build_liquid_mesh(mclq, h.position))
            .collect();

        chunks.push(ChunkMesh {
            positions,
            normals,
            uvs,
            indices,
            holes,
            base_texture,
            layer_textures,
            layer_effect_ids,
            alpha_map,
            shadow,
            pred_tex,
            no_effect_doodad,
            index_x: h.index_x,
            index_y: h.index_y,
            area_id: h.area_id,
            impassable: h.impassable(),
            liquids,
        });
    }

    if chunks.is_empty() {
        anyhow::bail!("ADT produced no terrain chunks (no MCVT height data?)");
    }

    // MDDF positions are corner-origin: X = 32·TILE − mddf.z, Y = 32·TILE − mddf.x, Z = mddf.y.
    const MAP_CENTER: f32 = 32.0 * TILE_SIZE;
    let doodads = root
        .doodad_placements
        .iter()
        .filter_map(|d| {
            let model = root.models.get(d.name_id as usize)?.clone();
            Some(Doodad {
                model,
                position: [
                    MAP_CENTER - d.position[2],
                    MAP_CENTER - d.position[0],
                    d.position[1],
                ],
                rotation: d.rotation,
                scale: d.scale as f32 / 1024.0,
                unique_id: d.unique_id,
            })
        })
        .collect();

    // MODF takes the same transform; its scale is unused in 1.12.
    let wmos = root
        .wmo_placements
        .iter()
        .filter_map(|w| {
            let model = root.wmos.get(w.name_id as usize)?.clone();
            Some(WmoInstance {
                model,
                position: [
                    MAP_CENTER - w.position[2],
                    MAP_CENTER - w.position[0],
                    w.position[1],
                ],
                rotation: w.rotation,
                unique_id: w.unique_id,
                doodad_set: w.doodad_set,
                name_set: w.name_set,
                bounds: modf_bounds_to_world(&w.bounds),
            })
        })
        .collect();

    Ok(TileMesh {
        chunks,
        doodads,
        wmos,
    })
}

/// A MODF bounding box (raw placement frame, `[min, max]`) as a **WoW world-space** AABB.
///
/// The corner remap is [`parse_tile_mesh`]'s own — `(x, y, z) -> (MAP_CENTER - z, MAP_CENTER - y', y)`
/// with the axes it uses for a position — and it flips two axes, so a mapped "min" corner is not
/// the mapped box's min. Both corners go through the transform and the extremes are taken
/// afterwards, which is the only form of this that stays correct if the remap ever changes.
///
/// `None` for a degenerate box: several MODF entries in the shipped corpus author `0,0,0 .. 0,0,0`,
/// and a zero-volume hull is worse than no hull — it is an invisible point the mover can hit.
fn modf_bounds_to_world(bounds: &[[f32; 3]; 2]) -> Option<[[f32; 3]; 2]> {
    // The same corner origin `parse_tile_mesh` uses for a placement position.
    const MAP_CENTER: f32 = 32.0 * TILE_SIZE;
    let corner = |p: &[f32; 3]| [MAP_CENTER - p[2], MAP_CENTER - p[0], p[1]];
    let (a, b) = (corner(&bounds[0]), corner(&bounds[1]));
    let min = [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])];
    let max = [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])];
    let solid =
        (0..3).all(|i| max[i] - min[i] > 0.01) && min.iter().chain(&max).all(|v| v.is_finite());
    solid.then_some([min, max])
}

/// Read a map's WDT from the chain and parse it (vanilla/Classic layout).
fn read_wdt(chain: &mut Chain, map: &str) -> Result<WdtFile> {
    let path = format!("World\\Maps\\{map}\\{map}.wdt");
    let bytes = chain
        .read_file(&path)
        .with_context(|| format!("reading {path}"))?;
    WdtReader::new(Cursor::new(bytes), WowVersion::Classic)
        .read()
        .map_err(|e| anyhow::anyhow!("parsing WDT {path}: {e}"))
}

fn tile_exists(wdt: &WdtFile, x: u32, y: u32) -> bool {
    wdt.get_tile(x as usize, y as usize)
        .is_some_and(|t| t.has_adt)
}

/// The existing ADT tile at a world (x, y) on `map`, else the first one spiralling out 4 tiles.
pub fn find_tile_near(
    chain: &mut Chain,
    map: &str,
    world_x: f32,
    world_y: f32,
) -> Result<(u32, u32)> {
    let wdt = read_wdt(chain, map)?;
    let (tx, ty) = benilla_wdt::world_to_tile(world_x, world_y);
    if tile_exists(&wdt, tx, ty) {
        return Ok((tx, ty));
    }
    for r in 1..=4i32 {
        for dy in -r..=r {
            for dx in -r..=r {
                let (nx, ny) = (tx as i32 + dx, ty as i32 + dy);
                if (0..64).contains(&nx)
                    && (0..64).contains(&ny)
                    && tile_exists(&wdt, nx as u32, ny as u32)
                {
                    return Ok((nx as u32, ny as u32));
                }
            }
        }
    }
    anyhow::bail!("no existing ADT tile near world ({world_x}, {world_y}) on map {map}")
}

/// A map's tile index (its WDT), parsed once for streaming.
pub struct MapTiles {
    map: String,
    wdt: WdtFile,
}

impl MapTiles {
    /// Parse the map's WDT from the chain.
    pub fn load(chain: &mut Chain, map: &str) -> Result<Self> {
        let wdt = read_wdt(chain, map)?;
        Ok(Self {
            map: map.to_string(),
            wdt,
        })
    }

    /// The map name (for `load_tile_mesh`).
    pub fn map(&self) -> &str {
        &self.map
    }

    /// The tile `(x, y)` containing a world `(x, y)`.
    pub fn tile_at(&self, world_x: f32, world_y: f32) -> (u32, u32) {
        benilla_wdt::world_to_tile(world_x, world_y)
    }

    /// Every existing ADT tile within `radius` tiles of a world `(x, y)`, nearest first.
    pub fn existing_in_radius(&self, world_x: f32, world_y: f32, radius: u32) -> Vec<(u32, u32)> {
        let (cx, cy) = benilla_wdt::world_to_tile(world_x, world_y);
        let r = radius as i32;
        let mut out: Vec<(i32, (u32, u32))> = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                let (tx, ty) = (cx as i32 + dx, cy as i32 + dy);
                if (0..64).contains(&tx)
                    && (0..64).contains(&ty)
                    && tile_exists(&self.wdt, tx as u32, ty as u32)
                {
                    out.push((dx * dx + dy * dy, (tx as u32, ty as u32)));
                }
            }
        }
        out.sort_by_key(|(d, _)| *d);
        out.into_iter().map(|(_, t)| t).collect()
    }
}

/// Read and mesh one ADT tile from the chain.
pub fn load_tile_mesh(chain: &mut Chain, map: &str, tile_x: u32, tile_y: u32) -> Result<TileMesh> {
    let path = format!("World\\Maps\\{map}\\{map}_{tile_x}_{tile_y}.adt");
    let bytes = chain
        .read_file(&path)
        .with_context(|| format!("reading {path}"))?;
    adt_to_tile_mesh(&bytes)
}

/// Load every existing ADT tile within `radius` tiles of the world (x, y) on `map`.
pub fn load_tiles_around(
    chain: &mut Chain,
    map: &str,
    world_x: f32,
    world_y: f32,
    radius: u32,
) -> Result<Vec<((u32, u32), TileMesh)>> {
    let wdt = read_wdt(chain, map)?;
    let (cx, cy) = benilla_wdt::world_to_tile(world_x, world_y);
    let r = radius as i32;

    let mut out = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let (tx, ty) = (cx as i32 + dx, cy as i32 + dy);
            if !(0..64).contains(&tx) || !(0..64).contains(&ty) {
                continue;
            }
            let (tx, ty) = (tx as u32, ty as u32);
            if !tile_exists(&wdt, tx, ty) {
                continue;
            }
            // An unparseable tile is skipped, not fatal.
            if let Ok(mesh) = load_tile_mesh(chain, map, tx, ty) {
                out.push(((tx, ty), mesh));
            }
        }
    }

    if out.is_empty() {
        anyhow::bail!("no loadable tiles within {radius} of world ({world_x}, {world_y}) on {map}");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk with only `positions` and `indices` set.
    fn chunk(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> ChunkMesh {
        ChunkMesh {
            positions,
            normals: Vec::new(),
            uvs: Vec::new(),
            indices,
            holes: 0,
            base_texture: None,
            layer_textures: Vec::new(),
            layer_effect_ids: Vec::new(),
            alpha_map: None,
            shadow: None,
            pred_tex: [0; 64],
            no_effect_doodad: [false; 64],
            index_x: 0,
            index_y: 0,
            area_id: 0,
            impassable: false,
            liquids: Vec::new(),
        }
    }

    /// One tile of a single flat MCNK (heights 0 over a base of 50) whose every MCNR normal is the
    /// bytes `n`.
    fn flat_tile(n: [u8; 3]) -> Vec<u8> {
        let chunk = |magic: &[u8; 4], payload: &[u8]| {
            [&magic[..], &(payload.len() as u32).to_le_bytes(), payload].concat()
        };
        let mcvt = chunk(b"TVCM", &[0u8; 145 * 4]);
        let mcnr = chunk(b"RNCM", &[n.repeat(145), vec![0; 13]].concat());
        let mut h = vec![0u8; 128];
        h[0x14..0x18].copy_from_slice(&(8u32 + 128).to_le_bytes()); // MCVT, from the MCNK's magic
        h[0x18..0x1c].copy_from_slice(&(8 + 128 + mcvt.len() as u32).to_le_bytes()); // MCNR
        h[0x70..0x74].copy_from_slice(&50f32.to_le_bytes()); // base height
        chunk(b"KNCM", &[h, mcvt, mcnr].concat())
    }

    /// `CMapChunk::LoadNormals` (`0x6aff10`) keeps the MCNR bytes in file order, X, Y, Z, so the
    /// mesh's normal is the bytes over 127 with no swap.
    #[test]
    fn the_mesh_normals_are_the_mcnr_bytes_in_file_order() {
        let mesh = adt_to_tile_mesh(&flat_tile([0, 0, 127])).unwrap();
        assert_eq!(mesh.chunks[0].normals[0], [0.0, 0.0, 1.0]);
        let mesh = adt_to_tile_mesh(&flat_tile([127, 64, 0])).unwrap();
        assert_eq!(mesh.chunks[0].normals[0], [1.0, 64.0 / 127.0, 0.0]);
        assert_eq!(mesh.chunks[0].normals.len(), 145);
    }

    #[test]
    fn height_at_interpolates_inside_and_misses_holes_and_off_footprint() {
        // The north half is one quad rising to z=10 at the middle; the south half is a hole.
        let nw = [0.0, 0.0, 0.0];
        let mid_x = nw[0] - CHUNK_SIZE * 0.5; // half a chunk south
        let far_x = nw[0] - CHUNK_SIZE; // the chunk's south edge
        let far_y = nw[1] - CHUNK_SIZE; // the chunk's east edge
        let positions = vec![
            nw,                   // 0: NW
            [nw[0], far_y, 0.0],  // 1: NE
            [mid_x, nw[1], 10.0], // 2: mid-W (ridge)
            [mid_x, far_y, 10.0], // 3: mid-E (ridge)
            [far_x, nw[1], 0.0],  // 4: SW
            [far_x, far_y, 0.0],  // 5: SE
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let c = chunk(positions, indices);

        // A column in the meshed north half interpolates up the ridge.
        let quarter_x = nw[0] - CHUNK_SIZE * 0.25;
        let z = c
            .height_at([quarter_x, nw[1] - CHUNK_SIZE * 0.5, 0.0])
            .unwrap();
        assert!((z - 5.0).abs() < 0.01, "ridge midpoint ≈ 5, got {z}");
        // A column over the hole has no surface.
        assert_eq!(
            c.height_at([far_x + 1.0, nw[1] - CHUNK_SIZE * 0.5, 0.0]),
            None
        );
        // A column past the chunk's east edge is off the footprint entirely.
        assert_eq!(c.height_at([quarter_x, far_y - 5.0, 0.0]), None);
    }

    /// A chunk sloping from z 2 at its north edge to z −6 at its south, under a pond at z 0 whose
    /// westmost column of cells is dry: the depth is the water line less the ground wherever a wet
    /// cell touches the vertex, and nothing where none does.
    #[test]
    fn water_depths_follow_the_surface_over_wet_cells_only() {
        let ground = |row: f32| 2.0 - row;
        let mut positions = Vec::with_capacity(145);
        for row in 0..9 {
            for col in 0..9 {
                positions.push([-(row as f32), -(col as f32), ground(row as f32)]);
            }
            if row < 8 {
                for col in 0..8 {
                    let r = row as f32 + 0.5;
                    positions.push([-r, -(col as f32 + 0.5), ground(r)]);
                }
            }
        }
        let mut c = chunk(positions, Vec::new());
        assert!(c.water_depths().iter().all(|d| *d == f32::NEG_INFINITY));
        c.liquids.push(crate::liquid::LiquidMesh {
            grid: [9, 9],
            wet: (0..64).map(|cell| cell % 8 != 0).collect(),
            shared: vec![false; 64],
            positions: (0..81)
                .map(|n| [-((n / 9) as f32), -((n % 9) as f32), 0.0])
                .collect(),
            uvs: Vec::new(),
            depths: Vec::new(),
            indices: Vec::new(),
            sound_nibble: 0,
            material_id: None,
            kind: crate::liquid::LiquidKind::Still,
        });
        let d = c.water_depths();
        // Outer row 5, column 4: ground −3 under the line. Row 1: a yard above it.
        assert_eq!(d[5 * 17 + 4], 3.0);
        assert_eq!(d[17 + 4], -1.0);
        // The inner vertex of wet cell (5, 4) sits half a row further down.
        assert_eq!(d[5 * 17 + 9 + 4], 3.5);
        // Column 0 touches only dry cells, and so does the dry column's inner vertex; column 1
        // borders a wet cell and carries its depth.
        assert_eq!(d[5 * 17], f32::NEG_INFINITY);
        assert_eq!(d[5 * 17 + 9], f32::NEG_INFINITY);
        assert_eq!(d[5 * 17 + 1], 3.0);
    }

    /// 20 of the 43 shipped WDTs author no ADT tiles: their global WMO is the whole map.
    #[test]
    fn real_wdts_split_into_adt_maps_and_wmo_only_maps() {
        let data = crate::wow_data_or_skip!();
        let mut chain = crate::open_chain(&data).expect("open chain");

        let jail = read_wdt(&mut chain, "StormwindJail").expect("StormwindJail.wdt");
        let g = jail
            .global_wmo()
            .expect("the Stockade is one WMO, no terrain");
        assert_eq!(
            g.model.to_ascii_lowercase(),
            "world\\wmo\\dungeon\\az_stormwindprisons\\stormwindjail.wmo"
        );
        assert_eq!(g.position, [0.0, 0.0, 0.0]);
        assert_eq!(g.rotation, [0.0, 0.0, 0.0]);
        assert!(
            (0..64).all(|y| (0..64).all(|x| !tile_exists(&jail, x, y))),
            "a WMO-only map authors no ADT tiles at all"
        );

        // Deadmines, the control, ships real terrain and no global WMO.
        let deadmines = read_wdt(&mut chain, "DeadminesInstance").expect("DeadminesInstance.wdt");
        assert!(deadmines.global_wmo().is_none());
        let tiles = (0..64)
            .flat_map(|y| (0..64).map(move |x| (x, y)))
            .filter(|&(x, y)| tile_exists(&deadmines, x, y))
            .count();
        assert_eq!(tiles, 36, "Deadmines ships 36 ADT tiles");

        // Dire Maul is the only shipped map placed at a heading, 180° about the origin.
        let dm = read_wdt(&mut chain, "DireMaul").expect("DireMaul.wdt");
        assert_eq!(
            dm.global_wmo().expect("WMO-only").rotation,
            [0.0, 180.0, 0.0]
        );
    }

    /// The reference stops a mover 1.46 yd east of `.go xyz -6601.98 -531.87 335.60 0`: the MCNK
    /// there is flagged [`benilla_adt::MCNK_IMPASSABLE`] and the pin's own is not.
    #[test]
    fn the_pin_stands_one_chunk_west_of_an_impassable_band() {
        let data = crate::wow_data_or_skip!();
        let mut chain = crate::open_chain(&data).expect("open chain");

        // A step east crosses the chunk boundary at y = -533.333, a tile seam too.
        let pin = [-6601.98, -531.87, 335.60];
        let east = [-6601.98, -535.0, 335.60];
        assert_eq!(crate::world_to_tile(pin[0], pin[1]), (32, 44));
        assert_eq!(crate::world_to_tile(east[0], east[1]), (33, 44));

        let here = load_tile_mesh(&mut chain, "Azeroth", 32, 44).expect("Azeroth_32_44.adt");
        let next = load_tile_mesh(&mut chain, "Azeroth", 33, 44).expect("Azeroth_33_44.adt");

        assert_eq!(
            impassable_at(&here.chunks, pin),
            Some(false),
            "the pin's own chunk is walkable — the reporter was standing on it"
        );
        assert_eq!(
            impassable_at(&next.chunks, east),
            Some(true),
            "the chunk 1.46 yd east is the wall"
        );
        // A tile answers only for its own footprint.
        assert_eq!(impassable_at(&here.chunks, east), None);
        assert_eq!(impassable_at(&next.chunks, pin), None);

        // Searing Gorge's rim (area 51) crosses this tile as a ribbon; the pin's tile has none.
        assert_eq!(
            next.chunks.iter().filter(|c| c.impassable).count(),
            48,
            "Azeroth_33_44 authors 48 impassable chunks"
        );
        assert_eq!(here.chunks.iter().filter(|c| c.impassable).count(), 0);
    }
    /// Height and MCSH alike, sampled off-border across four Elwynn and Stormwind tiles.
    #[test]
    fn the_chunk_index_answers_exactly_like_the_walk() {
        let data = crate::wow_data_or_skip!();
        let mut chain = crate::open_chain(&data).expect("open chain");
        let walk = |chunks: &[ChunkMesh], wow: [f32; 3]| {
            chunks.iter().find_map(|c| {
                c.indices.as_chunks::<3>().0.iter().find_map(|t| {
                    let tri = [
                        c.positions[t[0] as usize],
                        c.positions[t[1] as usize],
                        c.positions[t[2] as usize],
                    ];
                    triangle_z_at(&tri, wow[0], wow[1])
                })
            })
        };
        let mut hits = 0;
        for (x, y) in [(32u32, 48u32), (31, 48), (30, 48), (31, 47)] {
            let tile = load_tile_mesh(&mut chain, "Azeroth", x, y).expect("Elwynn/Stormwind tile");
            assert_eq!(tile.chunks.len(), 256);
            let nw = tile.chunks[0].positions[0];
            let step = TILE_SIZE / 64.0;
            for i in 0..64 {
                for j in 0..64 {
                    let wow = [
                        nw[0] - (i as f32 + 0.37) * step,
                        nw[1] - (j as f32 + 0.61) * step,
                        0.0,
                    ];
                    let walk_h = walk(&tile.chunks, wow);
                    assert_eq!(
                        terrain_height_at(&tile.chunks, wow),
                        walk_h,
                        "height at {wow:?}"
                    );
                    let walk_s = tile.chunks.iter().find_map(|c| c.mcsh_shadowed_at(wow));
                    assert_eq!(
                        mcsh_shadowed_at(&tile.chunks, wow),
                        walk_s,
                        "shadow at {wow:?}"
                    );
                    hits += usize::from(walk_h.is_some());
                }
            }
        }
        assert!(
            hits > 12000,
            "the sample grid should land on terrain almost everywhere: {hits}"
        );
    }

    /// A vertex's place on the chunk's flat grid, (column, row) in quads: outer on the corners,
    /// inner at the quad centres.
    fn grid_of(i: u32) -> (f32, f32) {
        let (r, c) = (i / 17, i % 17);
        if c < 9 {
            (c as f32, r as f32)
        } else {
            ((c - 9) as f32 + 0.5, r as f32 + 0.5)
        }
    }

    /// Twice the signed area of each triangle on the flat grid.
    fn windings(indices: &[u32]) -> Vec<f32> {
        indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let (a, b, c) = (grid_of(t[0]), grid_of(t[1]), grid_of(t[2]));
                (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
            })
            .collect()
    }

    /// Level 3 is the full fan, 256 triangles; level 2 takes 41 of the 145 vertices, 192 indices,
    /// 64 triangles (`0x86b820`, `0x86b8c8`), all wound as the full fan is.
    #[test]
    fn the_level_meshes_have_the_reference_counts_and_winding() {
        let near = chunk_lod_indices(CHUNK_LEVEL_NEAR, [0; 4], 0);
        assert_eq!(near.len(), 768);
        let far = chunk_lod_indices(CHUNK_LEVEL_FAR, [0; 4], 0);
        assert_eq!(far.len(), 192);
        let distinct: std::collections::BTreeSet<u32> = far.iter().copied().collect();
        assert_eq!(distinct.len(), 41);
        // Only outer vertices: the 5×5 corners and the 4×4 block centres.
        assert!(distinct.iter().all(|&i| i % 17 < 9));
        let sign = windings(&near)[0].signum();
        assert!(windings(&near)
            .iter()
            .all(|w| w.signum() == sign && *w != 0.0));
        assert!(windings(&far)
            .iter()
            .all(|w| w.signum() == sign && *w != 0.0));
    }

    /// A level 2 chunk beside a finer one splits its shared edge at every level 3 vertex on it,
    /// and the finer chunk's own list does not change.
    #[test]
    fn a_coarse_chunk_meets_every_vertex_of_a_finer_neighbours_edge() {
        // Each side: the caps byte and the outer vertices on that edge.
        let sides: [(usize, Vec<u32>); 4] = [
            (0, (0..9).collect()),
            (1, (0..9).map(|r| r * 17 + 8).collect()),
            (2, (0..9).map(|c| 8 * 17 + c).collect()),
            (3, (0..9).map(|r| r * 17).collect()),
        ];
        for (side, edge) in sides {
            let mut caps = [0; 4];
            caps[side] = CHUNK_LEVEL_NEAR;
            let stitched = chunk_lod_indices(CHUNK_LEVEL_FAR, caps, 0);
            assert_eq!(
                stitched.len(),
                192 + 4 * 3,
                "one triangle more per border cell"
            );
            assert!(
                edge.iter().all(|v| stitched.contains(v)),
                "side {side} misses an edge vertex"
            );
            let sign = windings(&stitched)[0].signum();
            assert!(windings(&stitched).iter().all(|w| w.signum() == sign));
        }
        let plain = chunk_lod_indices(CHUNK_LEVEL_NEAR, [0; 4], 0);
        assert_eq!(
            chunk_lod_indices(CHUNK_LEVEL_NEAR, [CHUNK_LEVEL_NEAR; 4], 0),
            plain
        );
    }

    /// A hole bit is one 2×2 block: one level 2 leaf, four level 3 leaves.
    #[test]
    fn a_hole_drops_one_block_at_either_level() {
        assert_eq!(
            chunk_lod_indices(CHUNK_LEVEL_FAR, [0; 4], 1).len(),
            192 - 12
        );
        assert_eq!(
            chunk_lod_indices(CHUNK_LEVEL_NEAR, [0; 4], 1).len(),
            768 - 48
        );
        assert!(chunk_lod_indices(CHUNK_LEVEL_FAR, [0; 4], 0xffff).is_empty());
        // The dropped level 2 leaf is the one around outer vertex (1, 1).
        assert!(!chunk_lod_indices(CHUNK_LEVEL_FAR, [0; 4], 1).contains(&18));
    }
}
