//! WDL, the low-detail distant terrain: loader and coarse mesher. One `.wdl` per map holds the
//! whole 64×64-tile world at one height per MCNK, the horizon the reference draws beyond the
//! streamed tiles out to `horizonfarclip`, unlit, untextured, vertex-white and fogged.
//!
//! Layout: `MVER` (version 18), `MAOF` (4096 `u32` tile offsets, `[tile_y][tile_x]`), then per
//! present tile a `MARE` of 17×17 outer and 16×16 inner `int16` heights; vanilla Azeroth has no
//! `MWMO`/`MWID`/`MODF` or `MAHO`. The mesh is raw WoW world coordinates (X north, Y west, Z up).

use std::io::{Cursor, Read, Seek, SeekFrom};

use crate::Chain;
use anyhow::{bail, Context, Result};

use crate::terrain::TILE_SIZE;

/// 16 cells a tile edge: 17×17 corners and 16×16 cell centres, an MCNK's grid at tile scale.
const OUTER_EDGE: usize = 17;
const INNER_EDGE: usize = 16;
const OUTER_N: usize = OUTER_EDGE * OUTER_EDGE;
const INNER_N: usize = INNER_EDGE * INNER_EDGE;
const MARE_BYTES: usize = (OUTER_N + INNER_N) * 2;
const CHUNK_SIZE: f32 = TILE_SIZE / 16.0;
/// World coordinate of tile (0, 0)'s max-X, max-Y corner; the 64-tile map is centred on the origin.
const MAP_OFFSET: f64 = 32.0 * (TILE_SIZE as f64);

/// A map's parsed WDL: the present tiles' low-detail heightmaps, indexed by MAOF position.
pub struct WdlFile {
    /// 64×64 in MAOF order, `[tile_y * 64 + tile_x]`, with `tile_x`/`tile_y` as in
    /// `Map_<tile_x>_<tile_y>.adt` and `benilla_wdt::world_to_tile`; `None` where the map has none.
    tiles: Vec<Option<MareTile>>,
}

/// One tile's absolute heights in yards: 17×17 corners, then 16×16 cell centres, row-major.
struct MareTile {
    outer: [i16; OUTER_N],
    inner: [i16; INNER_N],
}

/// One tile's outer height grid, as a **regular lattice** rather than a mesh — the shape a
/// heightfield collider wants.
///
/// [`WdlTileMesh`] describes the same surface, but as 545 vertices and 3,072 indices, and building a
/// parry trimesh out of that costs a QBVH. This is the same data with nothing derived from it: 289
/// heights, a centre and an edge length, which a heightfield consumes directly and stores as a grid
/// (no acceleration structure, because a lattice *is* one). That difference is why the streamer's
/// stand-in floors are free enough to raise while the player is running
/// (`benilla_world::terrain_stream::provisional`).
///
/// The inner 16×16 cell-centre samples are dropped: a heightfield's cell is two triangles from its
/// four corners, and the centre-fan's fifth sample has nowhere to go. It costs a little fidelity
/// against the drawn horizon, on a surface that exists to be replaced.
pub struct WdlTileGrid {
    /// 17×17 corner heights in absolute WoW `z` yards, `[row][col]` in the on-disk MARE order: row
    /// steps along **−X** (north → south), column along **−Y** (east → west).
    pub heights: Vec<Vec<f32>>,
    /// The tile's centre in raw WoW `(x, y)`. The heights are absolute, so there is no `z`.
    pub centre: [f32; 2],
    /// The tile's edge length in yards — the same on both planar axes.
    pub extent: f32,
}

/// A coarse WDL tile mesh in raw WoW world coordinates, with no normals or UVs: the reference
/// draws it unlit, vertex-white and fogged.
pub struct WdlTileMesh {
    /// 545 verts: 289 outer (`[r*17+c]`) then 256 inner (`[289 + r*16+c]`).
    pub positions: Vec<[f32; 3]>,
    /// 3072 indices = 16×16 cells × 4 triangles.
    pub indices: Vec<u32>,
}

/// The MAOF index of a tile, row-major over `tile_y` as on disk and in `benilla_wdt`.
fn tile_index(tile_x: u32, tile_y: u32) -> usize {
    tile_y as usize * 64 + tile_x as usize
}

impl WdlFile {
    /// Read and parse the map's `.wdl` off the patch chain (`World\Maps\<map>\<map>.wdl`).
    pub fn load(chain: &mut Chain, map: &str) -> Result<Self> {
        let path = format!("World\\Maps\\{map}\\{map}.wdl");
        let bytes = chain
            .read_file(&path)
            .with_context(|| format!("reading {path}"))?;
        parse_wdl(&bytes).with_context(|| format!("parsing {path}"))
    }

    /// Number of present (non-empty) tiles.
    pub fn present_count(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_some()).count()
    }

    /// Is tile `(tile_x, tile_y)` present in this WDL?
    pub fn is_present(&self, tile_x: u32, tile_y: u32) -> bool {
        tile_x < 64 && tile_y < 64 && self.tiles[tile_index(tile_x, tile_y)].is_some()
    }

    /// Iterate every present tile's `(tile_x, tile_y)`.
    pub fn present_tiles(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..4096u32).filter_map(move |i| self.tiles[i as usize].as_ref().map(|_| (i % 64, i / 64)))
    }

    /// Present tiles within a Chebyshev `radius` of world `(x, y)`, the centre tile included, as
    /// the reference's far walk (`0x683040`) visits a ±3-tile window. Below a tile of view distance
    /// (vanilla goes down to 177) the camera's own tile draws the near horizon, and without it sky
    /// shows through; the far band's near plane in `wdl.wgsl` alone bounds that side.
    pub fn tiles_in_ring(&self, world_x: f32, world_y: f32, radius: u32) -> Vec<(u32, u32)> {
        let (cx, cy) = benilla_wdt::world_to_tile(world_x, world_y);
        let r = radius as i32;
        let mut out = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                let (tx, ty) = (cx as i32 + dx, cy as i32 + dy);
                if (0..64).contains(&tx)
                    && (0..64).contains(&ty)
                    && self.is_present(tx as u32, ty as u32)
                {
                    out.push((tx as u32, ty as u32));
                }
            }
        }
        out
    }

    /// Tile `(tile_x, tile_y)`'s outer height lattice, or `None` if the map has no such tile.
    ///
    /// The lattice positions are [`Self::tile_mesh`]'s own, derived the same way from the **global**
    /// index so adjacent tiles share an edge exactly; only the form differs (see [`WdlTileGrid`]).
    pub fn tile_height_grid(&self, tile_x: u32, tile_y: u32) -> Option<WdlTileGrid> {
        if tile_x >= 64 || tile_y >= 64 {
            return None;
        }
        let tile = self.tiles[tile_index(tile_x, tile_y)].as_ref()?;
        let cs = CHUNK_SIZE as f64;
        let lat = |global: f64| (MAP_OFFSET - global * cs) as f32;
        let heights = (0..OUTER_EDGE)
            .map(|r| {
                (0..OUTER_EDGE)
                    .map(|c| f32::from(tile.outer[r * OUTER_EDGE + c]))
                    .collect()
            })
            .collect();
        Some(WdlTileGrid {
            // The lattice midpoint: 8 chunks in from the tile's own origin corner on both axes.
            centre: [
                lat(tile_y as f64 * 16.0 + 8.0),
                lat(tile_x as f64 * 16.0 + 8.0),
            ],
            heights,
            extent: TILE_SIZE,
        })
    }

    /// The coarse centre-fan mesh of tile `(tile_x, tile_y)`. Vertex X/Y come from the global
    /// lattice index in `f64`, so neighbouring tiles share bit-identical edges.
    pub fn tile_mesh(&self, tile_x: u32, tile_y: u32) -> Option<WdlTileMesh> {
        if tile_x >= 64 || tile_y >= 64 {
            return None;
        }
        let tile = self.tiles[tile_index(tile_x, tile_y)].as_ref()?;
        let cs = CHUNK_SIZE as f64;
        // Lattice position of a global row/col index (X north falls with row, Y west with col).
        let lat = |global: f64| (MAP_OFFSET - global * cs) as f32;

        let mut positions = Vec::with_capacity(OUTER_N + INNER_N);
        for r in 0..OUTER_EDGE {
            for c in 0..OUTER_EDGE {
                positions.push([
                    lat((tile_y as usize * 16 + r) as f64),
                    lat((tile_x as usize * 16 + c) as f64),
                    f32::from(tile.outer[r * OUTER_EDGE + c]),
                ]);
            }
        }
        for r in 0..INNER_EDGE {
            for c in 0..INNER_EDGE {
                positions.push([
                    lat(tile_y as usize as f64 * 16.0 + r as f64 + 0.5),
                    lat(tile_x as usize as f64 * 16.0 + c as f64 + 0.5),
                    f32::from(tile.inner[r * INNER_EDGE + c]),
                ]);
            }
        }

        // Four-triangle centre fan per cell, wound CCW from above like the terrain so it faces up
        // after the WoW-to-Bevy transform.
        let inner_base = OUTER_N as u32;
        let mut indices = Vec::with_capacity(INNER_EDGE * INNER_EDGE * 12);
        for r in 0..INNER_EDGE as u32 {
            for c in 0..INNER_EDGE as u32 {
                let tl = r * OUTER_EDGE as u32 + c;
                let tr = tl + 1;
                let bl = tl + OUTER_EDGE as u32;
                let br = bl + 1;
                let ctr = inner_base + r * INNER_EDGE as u32 + c;
                indices.extend_from_slice(&[
                    ctr, tl, bl, // west fan
                    ctr, bl, br, // south fan
                    ctr, br, tr, // east fan
                    ctr, tr, tl, // north fan
                ]);
            }
        }
        Some(WdlTileMesh { positions, indices })
    }

    /// A **coarser** mesh of tile `(tile_x, tile_y)` for the far horizon: the outer lattice sampled
    /// every `step` corners (1, 2, 4, 8 or 16), two triangles per cell, no centre fan. `None` if the
    /// tile is absent or `step` does not divide the 16-cell edge.
    ///
    /// At `step == 1` the border vertices and edges are exactly [`Self::tile_mesh`]'s, so the two LODs
    /// meet without a crack. A coarser step leaves T-junctions against a finer neighbour, so it hangs a
    /// `skirt`-yard curtain straight down from every border edge to cover them.
    pub fn tile_mesh_coarse(
        &self,
        tile_x: u32,
        tile_y: u32,
        step: usize,
        skirt: f32,
    ) -> Option<WdlTileMesh> {
        if tile_x >= 64 || tile_y >= 64 || step == 0 || !INNER_EDGE.is_multiple_of(step) {
            return None;
        }
        let tile = self.tiles[tile_index(tile_x, tile_y)].as_ref()?;
        let cs = CHUNK_SIZE as f64;
        let lat = |global: f64| (MAP_OFFSET - global * cs) as f32;
        let edge = INNER_EDGE / step + 1;
        let mut positions = Vec::with_capacity(edge * edge + 4 * edge * 2);
        for r in 0..edge {
            for c in 0..edge {
                let (or, oc) = (r * step, c * step);
                positions.push([
                    lat((tile_y as usize * 16 + or) as f64),
                    lat((tile_x as usize * 16 + oc) as f64),
                    f32::from(tile.outer[or * OUTER_EDGE + oc]),
                ]);
            }
        }
        let e = edge as u32;
        let mut indices = Vec::with_capacity((edge - 1) * (edge - 1) * 6 + 4 * (edge - 1) * 6);
        for r in 0..e - 1 {
            for c in 0..e - 1 {
                let tl = r * e + c;
                let (tr, bl) = (tl + 1, tl + e);
                let br = bl + 1;
                // The centre fan's winding with the centre moved onto a corner.
                indices.extend_from_slice(&[tl, bl, br, tl, br, tr]);
            }
        }
        if step > 1 && skirt > 0.0 {
            let border: Vec<u32> = (0..e)
                .chain((1..e).map(|r| r * e + e - 1))
                .chain((0..e - 1).rev().map(|c| (e - 1) * e + c))
                .chain((1..e - 1).rev().map(|r| r * e))
                .collect();
            for (k, &top) in border.iter().enumerate() {
                let next = border[(k + 1) % border.len()];
                let [x, y, z] = positions[top as usize];
                let base = positions.len() as u32;
                positions.push([x, y, z - skirt]);
                let [nx, ny, nz] = positions[next as usize];
                positions.push([nx, ny, nz - skirt]);
                indices.extend_from_slice(&[top, base, base + 1, top, base + 1, next]);
            }
        }
        Some(WdlTileMesh { positions, indices })
    }

    /// The drawn WDL height (absolute WoW `z`) under world `(x, y)`, interpolated over the same
    /// centre fan [`Self::tile_mesh`] builds, so it matches the rendered horizon exactly.
    pub fn height_at(&self, world_x: f32, world_y: f32) -> Option<f32> {
        let cs = CHUNK_SIZE as f64;
        // Global lattice coordinates, the inverse of `tile_mesh`'s `lat`: 1024 cells per axis.
        let gr = (MAP_OFFSET - f64::from(world_x)) / cs;
        let gc = (MAP_OFFSET - f64::from(world_y)) / cs;
        if !(0.0..1024.0).contains(&gr) || !(0.0..1024.0).contains(&gc) {
            return None;
        }
        let (cell_r, cell_c) = (gr as usize, gc as usize);
        let tile = self.tiles[tile_index((cell_c / 16) as u32, (cell_r / 16) as u32)].as_ref()?;
        let (r, c) = (cell_r % 16, cell_c % 16);
        let h = |rr: usize, cc: usize| f64::from(tile.outer[rr * OUTER_EDGE + cc]);
        let (tl, tr) = (h(r, c), h(r, c + 1));
        let (bl, br) = (h(r + 1, c), h(r + 1, c + 1));
        let ctr = f64::from(tile.inner[r * INNER_EDGE + c]);
        // `v` runs down the rows, `u` across; the cell's diagonals pick the fan triangle.
        let (v, u) = (gr - cell_r as f64, gc - cell_c as f64);
        let (a, b, t, s) = if v <= u && v <= 1.0 - u {
            (tl, tr, u, v) // north fan (CTR, TR, TL): edge TL→TR at v = 0
        } else if v >= u && v >= 1.0 - u {
            (bl, br, u, 1.0 - v) // south fan: edge BL→BR at v = 1
        } else if u < v {
            (tl, bl, v, u) // west fan: edge TL→BL at u = 0
        } else {
            (tr, br, v, 1.0 - u) // east fan: edge TR→BR at u = 1
        };
        // `s` runs 0 at the outer edge to 0.5 at the centre, where the edge parameter degenerates.
        let height = if s >= 0.5 {
            ctr
        } else {
            let edge = a + (b - a) * ((t - s) / (1.0 - 2.0 * s));
            edge + (ctr - edge) * (s * 2.0)
        };
        Some(height as f32)
    }
}

/// Read one IFF chunk header: the magic, stored reversed on disk, and the `u32` LE body size.
fn read_chunk_header(cur: &mut Cursor<&[u8]>) -> Result<([u8; 4], u32)> {
    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    magic.reverse();
    let mut size = [0u8; 4];
    cur.read_exact(&mut size)?;
    Ok((magic, u32::from_le_bytes(size)))
}

fn parse_wdl(bytes: &[u8]) -> Result<WdlFile> {
    let mut cur = Cursor::new(bytes);

    let (magic, size) = read_chunk_header(&mut cur)?;
    if &magic != b"MVER" {
        bail!("expected MVER first, got {:?}", magic);
    }
    let mut ver = [0u8; 4];
    cur.read_exact(&mut ver)?;
    let version = u32::from_le_bytes(ver);
    if version != 18 {
        bail!("unexpected WDL version {version} (expected 18)");
    }
    cur.seek(SeekFrom::Current(size as i64 - 4))?; // skip any MVER tail (none in vanilla)

    // Skip to MAOF past any low-detail WMO chunks.
    let offsets = loop {
        let (magic, size) = read_chunk_header(&mut cur)?;
        if &magic == b"MAOF" {
            if size as usize != 4096 * 4 {
                bail!("MAOF is {size} B, expected {} (4096 u32)", 4096 * 4);
            }
            let mut raw = vec![0u8; size as usize];
            cur.read_exact(&mut raw)?;
            let mut offs = vec![0u32; 4096];
            for (i, o) in offs.iter_mut().enumerate() {
                *o = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            }
            break offs;
        }
        cur.seek(SeekFrom::Current(size as i64))?;
    };

    // Each non-zero MAOF entry points at that tile's MARE chunk header.
    let mut tiles: Vec<Option<MareTile>> = (0..4096).map(|_| None).collect();
    for (i, &off) in offsets.iter().enumerate() {
        if off == 0 {
            continue;
        }
        cur.seek(SeekFrom::Start(off as u64))?;
        let (magic, size) = read_chunk_header(&mut cur)?;
        if &magic != b"MARE" {
            bail!("MAOF[{i}] offset {off} points at {magic:?}, expected MARE");
        }
        if (size as usize) < MARE_BYTES {
            bail!("MARE[{i}] is {size} B, expected ≥ {MARE_BYTES}");
        }
        let mut body = vec![0u8; MARE_BYTES];
        cur.read_exact(&mut body)?;
        let mut outer = [0i16; OUTER_N];
        let mut inner = [0i16; INNER_N];
        for (j, h) in outer.iter_mut().enumerate() {
            *h = i16::from_le_bytes([body[j * 2], body[j * 2 + 1]]);
        }
        let base = OUTER_N * 2;
        for (j, h) in inner.iter_mut().enumerate() {
            *h = i16::from_le_bytes([body[base + j * 2], body[base + j * 2 + 1]]);
        }
        tiles[i] = Some(MareTile { outer, inner });
    }

    Ok(WdlFile { tiles })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_index_is_row_major_over_y() {
        // A tile from a trace of the reference: (tile_x=34, tile_y=48) -> 48*64+34.
        assert_eq!(tile_index(34, 48), 48 * 64 + 34);
        assert_eq!(tile_index(0, 0), 0);
        assert_eq!(tile_index(63, 63), 4095);
    }

    #[test]
    fn mare_geometry_constants() {
        assert_eq!(OUTER_N, 289);
        assert_eq!(INNER_N, 256);
        assert_eq!(MARE_BYTES, 1090);
        assert_eq!(OUTER_N + INNER_N, 545); // verts per tile
                                            // 16×16 cells × 4 tris × 3: the reference's draw count.
        assert_eq!(INNER_EDGE * INNER_EDGE * 12, 3072);
        assert!((CHUNK_SIZE - TILE_SIZE / 16.0).abs() < 1e-3);
    }

    /// **The grid and the mesh describe the same lattice** — the test that keeps a heightfield
    /// collider built from [`WdlFile::tile_height_grid`] on the ground rather than a tile away or
    /// transposed across it.
    ///
    /// Two independent things are pinned per corner: the height at `[row][col]` is the height of
    /// the mesh vertex at the same index, and the world `(x, y)` that `centre`/`extent` imply for
    /// that index is that vertex's own. A transposition passes the first and fails the second,
    /// which is the whole reason the second is here.
    #[test]
    fn the_height_grid_and_the_mesh_are_the_same_lattice() {
        let mut tiles: Vec<Option<MareTile>> = (0..4096).map(|_| None).collect();
        let mut outer = [0i16; OUTER_N];
        // Distinct per (row, col) and NOT symmetric under transposition — the whole point.
        for r in 0..OUTER_EDGE {
            for c in 0..OUTER_EDGE {
                outer[r * OUTER_EDGE + c] = (r * 100 + c) as i16;
            }
        }
        let (tx, ty) = (30u32, 41u32);
        tiles[tile_index(tx, ty)] = Some(MareTile {
            outer,
            inner: [0i16; INNER_N],
        });
        let wdl = WdlFile { tiles };

        let grid = wdl.tile_height_grid(tx, ty).expect("the tile is present");
        let mesh = wdl.tile_mesh(tx, ty).expect("the tile is present");
        assert_eq!(grid.heights.len(), OUTER_EDGE);
        for (r, row) in grid.heights.iter().enumerate() {
            assert_eq!(row.len(), OUTER_EDGE);
            for (c, height) in row.iter().enumerate() {
                let vertex = mesh.positions[r * OUTER_EDGE + c];
                assert_eq!(*height, vertex[2], "height at [{r}][{c}]");
                // `centre` is the lattice midpoint and `extent` its full span, so index `i` of 17
                // sits at `centre + (0.5 - i/16) * extent` — row along −X, column along −Y.
                let at = |i: usize, mid: f32| mid + (0.5 - i as f32 / 16.0) * grid.extent;
                assert!(
                    (at(r, grid.centre[0]) - vertex[0]).abs() < 0.01,
                    "row {r} sits at x {} not {}",
                    at(r, grid.centre[0]),
                    vertex[0]
                );
                assert!(
                    (at(c, grid.centre[1]) - vertex[1]).abs() < 0.01,
                    "column {c} sits at y {} not {}",
                    at(c, grid.centre[1]),
                    vertex[1]
                );
            }
        }
    }

    /// The tile's last corner row and column belong to the absent neighbour tile in this synthetic
    /// file, so they are skipped.
    #[test]
    fn height_at_matches_the_drawn_mesh_exactly() {
        let mut tiles: Vec<Option<MareTile>> = (0..4096).map(|_| None).collect();
        let mut outer = [0i16; OUTER_N];
        let mut inner = [0i16; INNER_N];
        for (i, o) in outer.iter_mut().enumerate() {
            *o = ((i * 37) % 251) as i16 - 100;
        }
        for (i, o) in inner.iter_mut().enumerate() {
            *o = ((i * 53) % 211) as i16 - 60;
        }
        let (tx, ty) = (30u32, 41u32);
        tiles[tile_index(tx, ty)] = Some(MareTile { outer, inner });
        let wdl = WdlFile { tiles };
        let mesh = wdl.tile_mesh(tx, ty).unwrap();
        for (i, p) in mesh.positions.iter().enumerate() {
            if i < OUTER_N {
                let (r, c) = (i / OUTER_EDGE, i % OUTER_EDGE);
                if r == OUTER_EDGE - 1 || c == OUTER_EDGE - 1 {
                    continue; // neighbour-tile lattice line
                }
            }
            let h = wdl.height_at(p[0], p[1]).unwrap();
            assert!(
                (h - p[2]).abs() < 0.01,
                "vertex {i} at ({}, {}): height_at {h} vs mesh {}",
                p[0],
                p[1],
                p[2]
            );
        }
        assert_eq!(wdl.height_at(0.0, 0.0), None);
    }
}
