//! The residency window, derived from `farclip` (177..777) as the reference derives it:
//! `SetFarClip` (`0x6725d0`) sets `r = 1 + trunc(farclip / 33.333)` chunks, and the world tick
//! (`0x672730`) centres two windows on the viewer's chunk, the inner `idx ± r` (must be resident)
//! and the outer `idx ± max(r + 2, 8)` (requested ahead, built under 5 ms a frame). Eviction is
//! membership in the outer window, by tile (`[area+0x88]` vs `bounds >> 4`).
//!
//! - Deviation: whole tiles, because benilla loads, spawns and releases ADT tiles, not chunks: a
//!   tile is wanted when any of its chunks lies in the outer window, and waited for when any lies
//!   in the inner one, so each edge is a tile line where the reference's is a chunk line.
//! - Deviation: a one-chunk keep band ([`KEEP_BAND_CHUNKS`]), because a tile reload re-decodes and
//!   re-spawns where the reference's is a cheap read: a tile releases only past `outer + 1`.

use benilla_formats::{world_to_chunk, CHUNK_SIZE, TILE_SIZE};

/// Chunks per tile edge; `chunk >> 4` is the tile.
const CHUNKS_PER_TILE: i32 = 16;
/// Chunk indices run `0..=1023` (64 tiles × 16), matching the reference's `[0, 0x3ff]` clamp.
const CHUNK_MAX: i32 = 64 * CHUNKS_PER_TILE - 1;
/// The outer window's floor in chunks: `max(r + 2, 8)` (`0x672966`–`0x67297b`).
const OUTER_FLOOR_CHUNKS: i32 = 8;
/// Chunks the outer window grows by for the release test only.
pub const KEEP_BAND_CHUNKS: i32 = 1;

/// The two nested windows around one focus chunk, in chunk units on the tile-grid axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamWindow {
    /// The focus's chunk index `(chunk_x, chunk_y)`, on the tile grid's axes.
    pub focus: (i32, i32),
    /// Inner half-width in chunks: `1 + trunc(farclip / CHUNK_SIZE)`.
    pub inner: i32,
    /// Outer half-width in chunks: `max(inner + 2, 8)`.
    pub outer: i32,
}

impl StreamWindow {
    /// The windows `farclip` yards of view distance put around the WoW-space focus `(x, y)`.
    pub fn at(farclip: f32, wow_x: f32, wow_y: f32) -> Self {
        let (cx, cy) = world_to_chunk(wow_x, wow_y);
        let inner = inner_radius(farclip);
        Self {
            focus: (cx as i32, cy as i32),
            inner,
            outer: outer_radius(inner),
        }
    }

    /// The tile the focus stands on.
    pub fn focus_tile(&self) -> (i32, i32) {
        (
            self.focus.0 / CHUNKS_PER_TILE,
            self.focus.1 / CHUNKS_PER_TILE,
        )
    }

    /// **The neighbouring tile the focus is nearest to** — the one a body walking this way is
    /// likeliest to step onto next.
    ///
    /// It exists for the warm queue's terrain tiers (`benilla_formats::web::warm_priority`), which
    /// order the eight neighbours and need one of them named as the urgent one. The answer is the neighbour whose edge is
    /// fewest chunks away, resolved per axis and then between the two axes — and it is always an
    /// **axis** neighbour, never a diagonal, because a corner tile's nearest point is
    /// `√(dx² + dy²)` away and that is never closer than `min(dx, dy)`. A tie between the axes goes
    /// to x, arbitrarily: at a tie the two are equally close and both are in the near ring anyway.
    ///
    /// Clamped to the grid, so at a map edge this reports the focus tile itself rather than a tile
    /// that does not exist. That is harmless — the focus tile shares the same tier.
    pub fn lean_tile(&self) -> (i32, i32) {
        // Per axis: which way the nearer edge is, and how many chunks of this tile lie before it.
        let toward_edge = |chunk: i32| {
            let within = chunk.rem_euclid(CHUNKS_PER_TILE);
            let back = within;
            let forward = CHUNKS_PER_TILE - 1 - within;
            if forward < back {
                (1, forward)
            } else {
                (-1, back)
            }
        };
        let (step_x, reach_x) = toward_edge(self.focus.0);
        let (step_y, reach_y) = toward_edge(self.focus.1);
        let (tx, ty) = self.focus_tile();
        let (ox, oy) = if reach_x <= reach_y {
            (step_x, 0)
        } else {
            (0, step_y)
        };
        let last = CHUNK_MAX / CHUNKS_PER_TILE;
        ((tx + ox).clamp(0, last), (ty + oy).clamp(0, last))
    }

    /// The tile range the outer window grown by `band` chunks touches, clamped to the grid.
    fn tile_range(&self, band: i32) -> ((i32, i32), (i32, i32)) {
        self.tile_span(self.outer + band)
    }

    /// The tile range a window of `half` chunks around the focus touches, clamped to the grid.
    fn tile_span(&self, half: i32) -> ((i32, i32), (i32, i32)) {
        let lo = |c: i32| (c - half).clamp(0, CHUNK_MAX) / CHUNKS_PER_TILE;
        let hi = |c: i32| (c + half).clamp(0, CHUNK_MAX) / CHUNKS_PER_TILE;
        (
            (lo(self.focus.0), lo(self.focus.1)),
            (hi(self.focus.0), hi(self.focus.1)),
        )
    }

    /// Every tile the outer window touches, in grid order: the set the streamer wants resident.
    pub fn wanted_tiles(&self) -> Vec<(i32, i32)> {
        tiles_in(self.tile_range(0))
    }

    /// Every tile the inner window touches, in grid order: the set the tile pass waits for and
    /// builds without a limit (`0x698644`–`0x6986a5`).
    pub fn inner_tiles(&self) -> Vec<(i32, i32)> {
        tiles_in(self.tile_span(self.inner))
    }

    /// Whether a tile is inside the outer window plus the keep band.
    pub fn keeps(&self, tile: (i32, i32)) -> bool {
        let ((x0, y0), (x1, y1)) = self.tile_range(KEEP_BAND_CHUNKS);
        (x0..=x1).contains(&tile.0) && (y0..=y1).contains(&tile.1)
    }
}

/// The tiles of an inclusive tile range, in grid order.
fn tiles_in(((x0, y0), (x1, y1)): ((i32, i32), (i32, i32))) -> Vec<(i32, i32)> {
    let mut out = Vec::with_capacity(((x1 - x0 + 1) * (y1 - y0 + 1)) as usize);
    for tx in x0..=x1 {
        for ty in y0..=y1 {
            out.push((tx, ty));
        }
    }
    out
}

/// The inner half-width in chunks, `1 + trunc(farclip / 33.333)` (`0x6725e9`: `fmul −0.03`,
/// `__ftol`, `1 − eax`): 777 ⇒ 24, 350 ⇒ 11, 177 ⇒ 6.
pub fn inner_radius(farclip: f32) -> i32 {
    1 + (farclip / CHUNK_SIZE).trunc() as i32
}

/// The outer half-width in chunks: `max(inner + 2, 8)`.
pub fn outer_radius(inner: i32) -> i32 {
    (inner + 2).max(OUTER_FLOOR_CHUNKS)
}

/// The farthest, in yards, a resident tile's corner can lie from the focus: the floor an art sweep
/// must clear (`art_scope::radius_floor`) so it never evicts what the streamer still holds.
pub fn max_resident_reach_yd(farclip: f32) -> f32 {
    let half = outer_radius(inner_radius(farclip)) + KEEP_BAND_CHUNKS + CHUNKS_PER_TILE;
    half as f32 * CHUNK_SIZE * std::f32::consts::SQRT_2 + TILE_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_formats::tile_to_world;

    /// 350 is the reference's registered default (this fork ships 1000); at 177 the outer window
    /// sits on its 8-chunk floor.
    #[test]
    fn the_radii_are_the_references() {
        assert_eq!(inner_radius(777.0), 24);
        assert_eq!(outer_radius(24), 26);
        assert_eq!(inner_radius(350.0), 11);
        assert_eq!(outer_radius(11), 13);
        assert_eq!(inner_radius(177.0), 6);
        assert_eq!(outer_radius(6), 8);
        // Truncation, not rounding: a hair under a chunk multiple stays on the lower ring.
        assert_eq!(inner_radius(CHUNK_SIZE * 3.0 - 0.01), 3);
        assert_eq!(inner_radius(CHUNK_SIZE * 3.0 + 0.01), 4);
    }

    /// The window at the centre of `chunk` in `tile`.
    fn at_chunk(farclip: f32, tile: (u32, u32), chunk: (i32, i32)) -> StreamWindow {
        let (ox, oy) = tile_to_world(tile.0, tile.1);
        let wy = oy - (chunk.0 as f32 + 0.5) * CHUNK_SIZE;
        let wx = ox - (chunk.1 as f32 + 0.5) * CHUNK_SIZE;
        StreamWindow::at(farclip, wx, wy)
    }

    /// At 777 the outer window reaches 26 chunks: two tiles each way from the middle band (5×5);
    /// from a corner chunk, two behind and one ahead (4×4).
    #[test]
    fn the_wanted_block_follows_the_focus_chunk() {
        let w = at_chunk(777.0, (32, 44), (7, 7));
        assert_eq!(w.focus, (32 * 16 + 7, 44 * 16 + 7));
        assert_eq!(w.focus_tile(), (32, 44));
        let tiles = w.wanted_tiles();
        assert_eq!(tiles.len(), 25);
        assert!(tiles.contains(&(30, 42)) && tiles.contains(&(34, 46)));

        let w = at_chunk(777.0, (32, 44), (0, 0));
        let tiles = w.wanted_tiles();
        // Back: chunk 0 − 26 reaches 26 chunks into the previous tiles ⇒ two tiles. Forward:
        // chunk 0 + 26 = chunk 10 of the next tile ⇒ one tile.
        assert_eq!(tiles.len(), 16);
        assert!(tiles.contains(&(30, 42)) && tiles.contains(&(33, 45)));
        assert!(!tiles.contains(&(34, 44)));
    }

    /// At 350 the outer window is 13 chunks (433 yd): a 2×2 to 3×3 block.
    #[test]
    fn a_shorter_view_distance_wants_fewer_tiles() {
        assert_eq!(at_chunk(350.0, (32, 44), (7, 7)).wanted_tiles().len(), 9);
        assert_eq!(at_chunk(350.0, (32, 44), (15, 15)).wanted_tiles().len(), 4);
    }

    #[test]
    fn the_keep_band_is_one_chunk_of_hysteresis() {
        // From chunk 11 the outer window (26) reaches forward to chunk 37 = tile +2, chunk 5.
        let w = at_chunk(777.0, (32, 44), (11, 11));
        assert!(w.wanted_tiles().contains(&(34, 46)));
        // From chunk 5 it reaches chunk 31 = tile +1's last chunk: tile +2 is not wanted…
        let w = at_chunk(777.0, (32, 44), (5, 5));
        assert!(!w.wanted_tiles().contains(&(34, 46)));
        // …but the band (chunk 32 = tile +2, chunk 0) still keeps it.
        assert!(w.keeps((34, 46)));
        // From chunk 4 even the band falls short: the tile releases.
        let w = at_chunk(777.0, (32, 44), (4, 4));
        assert!(!w.keeps((34, 46)));
        // What is wanted is always kept.
        for t in w.wanted_tiles() {
            assert!(w.keeps(t));
        }
    }

    /// **The focus TILE is not a key for this window, and treating it as one loses a tile.**
    ///
    /// The window is centred on the focus *chunk*, so walking across the tile it is standing on
    /// grows [`Self::wanted_tiles`] well before the tile line is reached — at the registrar
    /// default that is 13 chunks, 433 yd, of warning. `stream_terrain`'s settled skip used to
    /// compare the focus tile and so read every one of those frames as "nothing changed": the
    /// neighbouring ADT was never requested until the body stepped onto it and the focus tile
    /// finally moved. This is that fact, stated where the window law lives.
    #[test]
    fn crossing_a_chunk_line_can_want_a_tile_the_focus_tile_never_moved_for() {
        // Two stands inside one tile: at its low edge, and thirteen chunks along — the first
        // chunk from which the outer window (13) no longer reaches back over the low tile line.
        let at_low_edge = at_chunk(350.0, (32, 44), (0, 8));
        let across_the_tile = at_chunk(350.0, (32, 44), (13, 8));
        // The same tile, both of them, so the coarse key cannot tell them apart…
        assert_eq!(at_low_edge.focus_tile(), across_the_tile.focus_tile());
        assert_eq!(at_low_edge.focus_tile(), (32, 44));
        // …and they are not the same window.
        assert_ne!(at_low_edge, across_the_tile);
        // The tile BEHIND is wanted from the low edge and released from across the tile…
        assert!(at_low_edge.wanted_tiles().contains(&(31, 44)));
        assert!(!across_the_tile.wanted_tiles().contains(&(31, 44)));
        // …and the tile AHEAD the other way round. Neither crossing moves the focus tile, so a
        // skip keyed on the tile never asks for the ground the walk just brought into range.
        assert!(!at_low_edge.wanted_tiles().contains(&(33, 44)));
        assert!(across_the_tile.wanted_tiles().contains(&(33, 44)));
    }

    /// The lean tile is the neighbour whose edge is nearest, on the nearer axis alone — never a
    /// diagonal, since a corner tile is `√2` further away than the axis tile beside it.
    #[test]
    fn the_lean_tile_is_the_nearest_neighbour() {
        // Chunk 15 of 16 on x, mid-tile on y: the +x neighbour is 0 chunks away and wins.
        assert_eq!(at_chunk(777.0, (32, 44), (15, 7)).lean_tile(), (33, 44));
        // Chunk 0 on x: the −x neighbour, same reasoning from the other side.
        assert_eq!(at_chunk(777.0, (32, 44), (0, 7)).lean_tile(), (31, 44));
        // Mid-tile on x, chunk 15 on y: now y is the nearer axis.
        assert_eq!(at_chunk(777.0, (32, 44), (7, 15)).lean_tile(), (32, 45));
        assert_eq!(at_chunk(777.0, (32, 44), (7, 0)).lean_tile(), (32, 43));
        // In the corner chunk both axes read 0 — the tie goes to x, not to the diagonal, because
        // the diagonal tile's nearest point is further than either axis tile's.
        assert_eq!(at_chunk(777.0, (32, 44), (15, 15)).lean_tile(), (33, 44));
        // Dead centre still names a nearest neighbour; it is just further off.
        let lean = at_chunk(777.0, (32, 44), (7, 7)).lean_tile();
        assert_eq!((lean.0 - 32).abs() + (lean.1 - 44).abs(), 1);
        // Whatever it is, it is always one of the eight — and always a tile the window wants.
        for cx in 0..16 {
            for cy in 0..16 {
                let w = at_chunk(777.0, (32, 44), (cx, cy));
                let (lx, ly) = w.lean_tile();
                assert!((lx - 32).abs() <= 1 && (ly - 44).abs() <= 1, "{cx},{cy}");
                assert!(w.wanted_tiles().contains(&(lx, ly)), "{cx},{cy}");
            }
        }
    }

    /// At the grid's edge there is no neighbour to name on that side, so the clamp reports the
    /// focus tile itself rather than a tile the map does not author.
    #[test]
    fn the_lean_tile_clamps_at_the_map_edge() {
        assert_eq!(at_chunk(777.0, (0, 63), (0, 15)).lean_tile(), (0, 63));
    }

    /// The inner window is `±r` chunks: at 177 (r = 6) a mid-tile focus waits on its own tile
    /// alone, at 350 (r = 11) on its 3×3, and at 777 (r = 24, 49 chunks) always on a 4×4.
    #[test]
    fn the_inner_tiles_are_the_inner_windows() {
        let mid = at_chunk(177.0, (32, 44), (7, 7));
        assert_eq!(mid.inner_tiles(), vec![(32, 44)]);
        // From chunk 0, six chunks back is the previous tile's chunk 10.
        let corner = at_chunk(177.0, (32, 44), (0, 0));
        assert_eq!(
            corner.inner_tiles(),
            vec![(31, 43), (31, 44), (32, 43), (32, 44)]
        );

        let mid = at_chunk(350.0, (32, 44), (7, 7));
        let tiles = mid.inner_tiles();
        assert_eq!(tiles.len(), 9);
        assert!(tiles.contains(&(31, 43)) && tiles.contains(&(33, 45)));
        let corner = at_chunk(350.0, (32, 44), (15, 15));
        assert_eq!(
            corner.inner_tiles(),
            vec![(32, 44), (32, 45), (33, 44), (33, 45)]
        );

        for chunk in [(7, 7), (0, 0), (15, 15)] {
            let w = at_chunk(777.0, (32, 44), chunk);
            let tiles = w.inner_tiles();
            assert_eq!(tiles.len(), 16, "from chunk {chunk:?}");
            // Always inside what the streamer wants.
            let wanted = w.wanted_tiles();
            assert!(tiles.iter().all(|t| wanted.contains(t)));
        }
        let mid = at_chunk(777.0, (32, 44), (7, 7));
        // Chunk 519 ± 24 runs 495..=543: tiles 30..=33.
        assert!(mid.inner_tiles().contains(&(30, 42)) && mid.inner_tiles().contains(&(33, 45)));
        assert!(!mid.inner_tiles().contains(&(34, 44)));
    }

    #[test]
    fn the_window_clamps_to_the_grid() {
        let w = at_chunk(777.0, (0, 63), (0, 15));
        let tiles = w.wanted_tiles();
        assert!(tiles
            .iter()
            .all(|&(x, y)| (0..64).contains(&x) && (0..64).contains(&y)));
        assert!(tiles.contains(&(0, 63)) && tiles.contains(&(1, 62)));
        assert!(w.keeps((0, 63)) && !w.keeps((-1, 63)) && !w.keeps((0, 64)));
        assert!(w
            .inner_tiles()
            .iter()
            .all(|&(x, y)| (0..64).contains(&x) && (0..64).contains(&y)));
    }

    #[test]
    fn the_resident_reach_bounds_the_block() {
        let reach = max_resident_reach_yd(777.0);
        // (26 + 1 + 16) chunks = 43 × 33⅓ = 1433 yd per axis; the corner is √2 of that, plus a tile.
        assert!((reach - (43.0 * CHUNK_SIZE * std::f32::consts::SQRT_2 + TILE_SIZE)).abs() < 0.01);
        assert!(max_resident_reach_yd(350.0) < reach);
        // A wanted tile's far corner lies inside the reach.
        let w = at_chunk(777.0, (32, 44), (9, 9));
        let (fx, fy) = (w.focus.0 as f32, w.focus.1 as f32);
        let far = w
            .wanted_tiles()
            .into_iter()
            .map(|(tx, ty)| {
                let cx = ((tx + 1) * CHUNKS_PER_TILE) as f32 - fx;
                let cy = ((ty + 1) * CHUNKS_PER_TILE) as f32 - fy;
                (cx * cx + cy * cy).sqrt() * CHUNK_SIZE
            })
            .fold(0.0f32, f32::max);
        assert!(
            far < reach,
            "far corner {far} must sit inside the reach {reach}"
        );
    }
}
