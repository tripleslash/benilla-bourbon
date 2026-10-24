//! fork (Everwood port): the rain-occlusion height map. A camera-centred top-down grid
//! ([`GRID`]² cells of [`CELL`] yd, ~96 yd across, recentred in [`STEP`]-cell steps) that says,
//! per cell, how high the topmost surface is and whether open air lies under it, so the wet lane
//! (`wet_hook.wgsl`) keeps ground and walls under porches, bridges and roofs dry.
//!
//! Source: CPU vertical rays against the collision world, the same `SpatialQuery` and
//! [`WorldCollision::body_filter`](crate::collision::WorldCollision::body_filter) the precip
//! ground oracle lands drops on roofs with (`precip/pool.rs`, `HeightCache`). It is the cheapest
//! robust source: the colliders already exist (terrain, WMO groups, collidable doodads), cost
//! nothing to build, and see exactly the roofs the rain itself sees. Each cell casts down from
//! [`LIFT`] yd above the camera: the first hit is the top surface; then it keeps casting from just
//! under each hit (at most [`MAX_HOPS`]) until it finds an air gap of at least [`GAP`] yd (a roof,
//! a deck, a floor over a room) or runs out of surfaces (open ground). A roof's own shell (its
//! outer and ceiling faces a hand apart) is thinner than the gap, so it does not count as shelter.
//! Doodads without a canopy collider (most trees) do not shelter; that is a limit of the source.
//! fork: a liquid surface counts as a roof too ([`under_liquid`]): rain does not wet a riverbed,
//! so ground a yard or more under the water line is dry, while the shore stays wet. That is the
//! models' answer near the camera; terrain carries its own depth under the water per vertex
//! (`benilla_assets::adt::pack_water_depth`), which holds past the grid and in the shallows.
//!
//! Per cell the GPU gets one `u32` ([`pack_cell`]): the low half the top height relative to the
//! header's `base_y` in 1/64 yd (±512 yd), the high half the soft cover 0..1 as unorm16. The cover is the
//! raw 0/1 gap flag eroded by two cells (the drip margin: ground just inside an eave stays wet)
//! then box-blurred over 3×3 (the soft edge), so it ramps 1/3, 2/3, 1 inward over ~2 yd.
//! A fragment is sheltered by `cover × smoothstep(0.5, 1.5, top − y)`: the roof itself (at `top`)
//! stays wet, everything a yard or more under it is dry.
//!
//! The grid rides the shared light buffer as its own region, right after the `EnvFrame` block
//! ([`region_offset`]), so no receiver needs a new binding. The region is two header rows
//! (`[origin_x, origin_z, 1/CELL, active]`, `[base_y, GRID, 0, 0]`) and `GRID²` words.
//!
//! Inert unless it is wet: with `rainSurfaces 0` or a zero wetness (`EnvFrame::wetness`) the
//! system casts nothing, the header's `active` is 0 and every reader skips the map. It stays live
//! while the ground dries after the rain, so a porch does not turn wet when the rain stops.
//! The precip splashes do not read the map: they already land on the topmost surface through the
//! reference's own ground oracle (`0x67c760`), which is the right answer for a drop.

use std::sync::Arc;

use avian3d::prelude::{SpatialQuery, SpatialQueryFilter};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::RenderQueue;
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::lighting::{EnvFrame, SharedLightBuffer};
use crate::view::WorldCamera;

/// Cells per side.
pub const GRID: usize = crate::lighting::SHELTER_GRID;
/// Cell size in yd (the grid spans `GRID × CELL` = 96 yd).
pub const CELL: f32 = 0.75;
/// The recentre step in cells (12 yd): the camera stays at least 48 cells from every edge.
pub const STEP: i32 = 16;
/// Ray start above the camera, yd.
pub const LIFT: f32 = 50.0;
/// Camera height drift (yd) that restarts the grid, since the ray start decides which roofs count.
const RECAST_DRIFT: f32 = 20.0;
/// How far a column is searched below its ray start, yd.
const REACH: f32 = 250.0;
/// The smallest air gap under a surface that counts as shelter, yd.
pub const GAP: f32 = 1.2;
/// Surfaces followed down a column before it is called open.
pub const MAX_HOPS: usize = 4;
/// Cells cast per frame while cells are unsampled (nearest the camera first).
pub const FILL_BUDGET: usize = 192;
/// Cells re-cast per frame once full, so streamed-in colliders show up within a few seconds.
pub const REFRESH_BUDGET: usize = 32;
/// The drip margin: covered cells this many cells from open sky count as open (1.5 yd), so a
/// column sampled just under an eave never leaks shelter outside it.
const ERODE: i32 = 2;
/// Header rows before the cells.
pub const HEADER_ROWS: usize = crate::lighting::SHELTER_HEADER_ROWS;
/// The region's size in the shared light buffer: the header and one word a cell.
pub const REGION_BYTES: u64 = crate::lighting::SHELTER_REGION_BYTES;
/// A missing top: far below anything, so `top − y` never shelters.
const NO_TOP: f32 = -60_000.0;

/// Byte offset of the region: right after the per-frame blob (and before the prop probes).
pub fn region_offset() -> u64 {
    crate::lighting::shelter_region_offset()
}

/// One column's answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column {
    /// The topmost surface, world Y; `None` when the column hits nothing.
    pub top: Option<f32>,
    /// Open air of at least [`GAP`] under some surface of the column.
    pub covered: bool,
}

/// Walks one column down from `from_y` with `cast(origin_y, max_dist, first) -> hit distance`.
pub fn probe_column(from_y: f32, mut cast: impl FnMut(f32, f32, bool) -> Option<f32>) -> Column {
    let Some(d) = cast(from_y, REACH, true) else {
        return Column {
            top: None,
            covered: false,
        };
    };
    let top = from_y - d;
    let mut y = top;
    for _ in 0..MAX_HOPS {
        let start = y - 0.02;
        let reach = start - (from_y - REACH);
        if reach <= 0.0 {
            break;
        }
        let Some(d) = cast(start, reach, false) else {
            break;
        };
        let next = start - d;
        if y - next >= GAP {
            return Column {
                top: Some(top),
                covered: true,
            };
        }
        y = next;
    }
    Column {
        top: Some(top),
        covered: false,
    }
}

/// fork: a liquid surface over open ground covers it: the column's top becomes the water line, so
/// the bed under it is sheltered like the floor under a roof. A column already covered keeps its
/// own (higher) roof; water at or under the top (a shore, a dock) changes nothing.
pub fn under_liquid(col: Column, surface_y: Option<f32>) -> Column {
    match (col.top, surface_y) {
        (Some(top), Some(s)) if !col.covered && s > top => Column {
            top: Some(s),
            covered: true,
        },
        _ => col,
    }
}

/// Height steps a yard in a packed cell (1/64 yd over ±512 yd around `base_y`).
pub const HEIGHT_SCALE: f32 = 64.0;

/// Packs a cell: the low half the top relative to `base_y` as `rel × 64 + 32768` (clamped), the
/// high half the cover as unorm16. Plain integers, so the shaders need no float16 capability.
pub fn pack_cell(top_rel: f32, cover: f32) -> u32 {
    let h = (top_rel * HEIGHT_SCALE + 32_768.0)
        .round()
        .clamp(0.0, 65_535.0) as u32;
    let c = (cover.clamp(0.0, 1.0) * 65_535.0).round() as u32;
    h | (c << 16)
}

/// The inverse of [`pack_cell`]: `(top_rel, cover)`, as `wet_hook::shelter_cell` decodes it.
pub fn unpack_cell(w: u32) -> (f32, f32) {
    (
        ((w & 0xffff) as f32 - 32_768.0) / HEIGHT_SCALE,
        (w >> 16) as f32 / 65_535.0,
    )
}

/// The nearest-first cast order: cell indices by distance from the grid centre.
fn fill_order() -> Vec<u16> {
    let c = GRID as f32 / 2.0 - 0.5;
    let mut order: Vec<u16> = (0..GRID * GRID).map(|i| i as u16).collect();
    order.sort_by(|&a, &b| {
        let d = |i: u16| {
            let (x, z) = (f32::from(i) % GRID as f32, (usize::from(i) / GRID) as f32);
            (x - c).powi(2) + (z - c).powi(2)
        };
        d(a).total_cmp(&d(b))
    });
    order
}

/// The grid: raw columns, the derived cover and the packed words the GPU reads.
#[derive(Resource)]
pub struct ShelterGrid {
    /// World cell coordinates of cell (0, 0) (`floor(world / CELL)`).
    pub origin: IVec2,
    /// World Y the rays start from.
    pub cast_y: f32,
    top: Vec<f32>,
    raw: Vec<f32>,
    eroded: Vec<f32>,
    sampled: Vec<bool>,
    pending: usize,
    refresh: usize,
    order: Vec<u16>,
    /// The packed words, shared with the render world.
    pub cells: Arc<Vec<u32>>,
    /// Grid rows changed since the last publish, `[lo, hi)`.
    dirty: Option<(usize, usize)>,
    /// Whether the map is live (it was wet this frame).
    pub active: bool,
    started: bool,
}

impl Default for ShelterGrid {
    fn default() -> Self {
        let n = GRID * GRID;
        Self {
            origin: IVec2::ZERO,
            cast_y: 0.0,
            top: vec![NO_TOP; n],
            raw: vec![0.0; n],
            eroded: vec![0.0; n],
            sampled: vec![false; n],
            pending: n,
            refresh: 0,
            order: fill_order(),
            cells: Arc::new(vec![pack_cell(NO_TOP, 0.0); n]),
            dirty: None,
            active: false,
            started: false,
        }
    }
}

impl ShelterGrid {
    /// The origin a camera at `cam_xz` wants: its step, less half the grid.
    pub fn origin_for(cam_xz: Vec2) -> IVec2 {
        let cell = (cam_xz / CELL).floor().as_ivec2();
        (cell.div_euclid(IVec2::splat(STEP)) - IVec2::splat(GRID as i32 / STEP / 2)) * STEP
    }

    /// The Y relative heights are stored against.
    pub fn base_y(&self) -> f32 {
        self.cast_y - LIFT
    }

    /// The cell holding a world XZ, if inside.
    pub fn cell_of(&self, xz: Vec2) -> Option<(usize, usize)> {
        let c = (xz / CELL).floor().as_ivec2() - self.origin;
        let n = GRID as i32;
        (c.x >= 0 && c.y >= 0 && c.x < n && c.y < n).then_some((c.x as usize, c.y as usize))
    }

    /// A cell's centre in world XZ.
    pub fn cell_centre(&self, x: usize, z: usize) -> Vec2 {
        (self.origin.as_vec2() + Vec2::new(x as f32 + 0.5, z as f32 + 0.5)) * CELL
    }

    /// Cells still to cast before the grid is whole.
    pub fn pending(&self) -> usize {
        self.pending
    }

    /// Forgets every column.
    pub fn clear(&mut self) {
        self.top.fill(NO_TOP);
        self.raw.fill(0.0);
        self.sampled.fill(false);
        self.pending = GRID * GRID;
        self.refresh = 0;
        self.rederive_all();
    }

    /// Follows the camera: shifts the grid by whole steps (keeping every column still inside)
    /// and restarts it when the camera has climbed or dropped past [`RECAST_DRIFT`]. Returns
    /// whether anything moved.
    pub fn follow(&mut self, cam: Vec3) -> bool {
        let want_y = cam.y + LIFT;
        if !self.started || (want_y - self.cast_y).abs() > RECAST_DRIFT {
            self.started = true;
            self.cast_y = want_y;
            self.origin = Self::origin_for(cam.xz());
            self.clear();
            return true;
        }
        let want = Self::origin_for(cam.xz());
        if want == self.origin {
            return false;
        }
        let d = want - self.origin;
        self.origin = want;
        let n = GRID as i32;
        let mut top = vec![NO_TOP; GRID * GRID];
        let mut raw = vec![0.0; GRID * GRID];
        let mut sampled = vec![false; GRID * GRID];
        for z in 0..n {
            for x in 0..n {
                let (ox, oz) = (x + d.x, z + d.y);
                if ox < 0 || oz < 0 || ox >= n || oz >= n {
                    continue;
                }
                let (i, o) = ((z * n + x) as usize, (oz * n + ox) as usize);
                top[i] = self.top[o];
                raw[i] = self.raw[o];
                sampled[i] = self.sampled[o];
            }
        }
        self.top = top;
        self.raw = raw;
        self.pending = sampled.iter().filter(|s| !**s).count();
        self.sampled = sampled;
        self.rederive_all();
        true
    }

    /// Casts up to a frame's budget of columns with `probe(xz)`, nearest unsampled first, then a
    /// slow refresh sweep. Returns how many columns were cast.
    pub fn step(&mut self, mut probe: impl FnMut(Vec2) -> Column) -> usize {
        let mut cast = 0;
        if self.pending > 0 {
            for k in 0..self.order.len() {
                if cast == FILL_BUDGET {
                    break;
                }
                let i = usize::from(self.order[k]);
                if self.sampled[i] {
                    continue;
                }
                self.cast_cell(i, &mut probe);
                cast += 1;
            }
        } else {
            for _ in 0..REFRESH_BUDGET {
                let i = usize::from(self.order[self.refresh]);
                self.refresh = (self.refresh + 1) % self.order.len();
                self.cast_cell(i, &mut probe);
                cast += 1;
            }
        }
        cast
    }

    fn cast_cell(&mut self, i: usize, probe: &mut impl FnMut(Vec2) -> Column) {
        let (x, z) = (i % GRID, i / GRID);
        let col = probe(self.cell_centre(x, z));
        if !self.sampled[i] {
            self.sampled[i] = true;
            self.pending -= 1;
        }
        let top = col.top.unwrap_or(NO_TOP);
        let raw = if col.covered { 1.0 } else { 0.0 };
        if top != self.top[i] || raw != self.raw[i] {
            self.top[i] = top;
            self.raw[i] = raw;
            self.rederive(x, z);
        }
    }

    fn rederive_all(&mut self) {
        self.rederive_rect(0, 0, GRID - 1, GRID - 1);
    }

    /// A raw change reaches the cover `ERODE + 1` cells out (erode, then blur).
    fn rederive(&mut self, x: usize, z: usize) {
        let r = ERODE as usize + 1;
        self.rederive_rect(
            x.saturating_sub(r),
            z.saturating_sub(r),
            (x + r).min(GRID - 1),
            (z + r).min(GRID - 1),
        );
    }

    /// Recomputes the cover and the packed words over an inclusive rect.
    fn rederive_rect(&mut self, x0: usize, z0: usize, x1: usize, z1: usize) {
        let n = GRID as i32;
        let at = |v: &Vec<f32>, x: i32, z: i32| {
            if x < 0 || z < 0 || x >= n || z >= n {
                0.0
            } else {
                v[(z * n + x) as usize]
            }
        };
        // Erosion over the rect grown by one (the blur reads it).
        let (ex0, ez0) = (x0.saturating_sub(1), z0.saturating_sub(1));
        let (ex1, ez1) = ((x1 + 1).min(GRID - 1), (z1 + 1).min(GRID - 1));
        for z in ez0..=ez1 {
            for x in ex0..=ex1 {
                let mut m = 1.0f32;
                for dz in -ERODE..=ERODE {
                    for dx in -ERODE..=ERODE {
                        m = m.min(at(&self.raw, x as i32 + dx, z as i32 + dz));
                    }
                }
                self.eroded[z * GRID + x] = m;
            }
        }
        let base = self.base_y();
        let cells = Arc::make_mut(&mut self.cells);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let mut s = 0.0;
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        s += at(&self.eroded, x as i32 + dx, z as i32 + dz);
                    }
                }
                let i = z * GRID + x;
                cells[i] = pack_cell(self.top[i] - base, s / 9.0);
            }
        }
        self.dirty = Some(match self.dirty {
            Some((lo, hi)) => (lo.min(z0), hi.max(z1 + 1)),
            None => (z0, z1 + 1),
        });
    }

    /// The two header rows.
    pub fn header(&self) -> [[f32; 4]; HEADER_ROWS] {
        let o = self.origin.as_vec2() * CELL;
        [
            [o.x, o.y, 1.0 / CELL, if self.active { 1.0 } else { 0.0 }],
            [self.base_y(), GRID as f32, 0.0, 0.0],
        ]
    }

    /// The sheltered amount a fragment at `p` gets: the CPU twin of `wet_hook::shelter_amount`
    /// (bilinear over the four nearest cell centres), for tests and probes.
    pub fn shelter_at(&self, p: Vec3) -> f32 {
        if !self.active {
            return 0.0;
        }
        let g = (p.xz() / CELL - self.origin.as_vec2()) - 0.5;
        let gf = g.floor();
        if gf.x < 0.0
            || gf.y < 0.0
            || gf.x + 1.0 > (GRID - 1) as f32
            || gf.y + 1.0 > (GRID - 1) as f32
        {
            return 0.0;
        }
        let f = g - gf;
        let (x, z) = (gf.x as usize, gf.y as usize);
        let y_rel = p.y - self.base_y();
        let tap = |x: usize, z: usize| {
            let (top, cover) = unpack_cell(self.cells[z * GRID + x]);
            let t = (top - y_rel - 0.5).clamp(0.0, 1.0);
            cover * t * t * (3.0 - 2.0 * t)
        };
        let a = tap(x, z) + (tap(x + 1, z) - tap(x, z)) * f.x;
        let b = tap(x, z + 1) + (tap(x + 1, z + 1) - tap(x, z + 1)) * f.x;
        (a + (b - a) * f.y).clamp(0.0, 1.0)
    }
}

/// The render-world copy: the header every frame, the cells when they changed.
#[derive(Resource, Clone, ExtractResource)]
pub struct ShelterExtract {
    header: [[f32; 4]; HEADER_ROWS],
    cells: Arc<Vec<u32>>,
    generation: u64,
    /// Rows this generation changed, `[lo, hi)`.
    rows: Option<(usize, usize)>,
}

impl Default for ShelterExtract {
    fn default() -> Self {
        Self {
            header: [[0.0; 4]; HEADER_ROWS],
            cells: Arc::new(Vec::new()),
            generation: 0,
            rows: None,
        }
    }
}

/// Main world: follows the camera and casts this frame's columns while it is wet; idle (no rays,
/// the header marked off once) otherwise.
fn tick_shelter(
    frame: Res<EnvFrame>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    spatial: SpatialQuery,
    // fork: the liquid surfaces, through the spatial index (a dry cell costs one hash miss).
    // Optional: a world without the liquid plugin simply has no water to shelter under.
    water_index: Option<Res<crate::liquid::WaterIndex>>,
    water: Query<&crate::liquid::WaterChunkInfo>,
    mut grid: ResMut<ShelterGrid>,
    mut out: ResMut<ShelterExtract>,
) {
    let wet = frame.wetness > 0.0;
    let cam = cam.single().ok().map(GlobalTransform::translation);
    let live = wet && cam.is_some();
    if !live {
        if grid.active {
            grid.active = false;
            // Stale columns would lie on the next shower; forget them.
            grid.started = false;
            out.header = grid.header();
        }
        return;
    }
    let cam = cam.unwrap_or_default();
    grid.active = true;
    grid.follow(cam);
    let filter = crate::collision::WorldCollision::body_filter();
    let x0 = grid.cast_y;
    grid.step(|xz| {
        let col = cast_column(&spatial, &filter, xz, x0);
        let surface = water_index
            .as_deref()
            .and_then(|index| water_surface_y(index, &water, xz));
        under_liquid(col, surface)
    });
    let header = grid.header();
    if out.header != header {
        out.header = header;
    }
    if let Some(rows) = grid.dirty.take() {
        out.cells = Arc::clone(&grid.cells);
        out.generation = out.generation.wrapping_add(1);
        out.rows = Some(rows);
    }
}

fn cast_column(
    spatial: &SpatialQuery,
    filter: &SpatialQueryFilter,
    xz: Vec2,
    from_y: f32,
) -> Column {
    probe_column(from_y, |y, max, first| {
        spatial
            .cast_ray(Vec3::new(xz.x, y, xz.y), Dir3::NEG_Y, max, first, filter)
            .map(|hit| hit.distance)
    })
}

/// The highest liquid surface over a Bevy XZ, as Bevy Y (WoW z): any kind, since nothing under
/// lava or slime is rained on either.
fn water_surface_y(
    index: &crate::liquid::WaterIndex,
    water: &Query<&crate::liquid::WaterChunkInfo>,
    xz: Vec2,
) -> Option<f32> {
    let wow = benilla_assets::coords::bevy_to_wow(Vec3::new(xz.x, 0.0, xz.y));
    index
        .over(wow[0], wow[1])
        .iter()
        .filter_map(|&e| water.get(e).ok()?.surface_z_at(wow[0], wow[1]))
        .max_by(f32::total_cmp)
}

/// Render world: writes the header each frame it moved and the changed cell rows.
fn upload_shelter(
    queue: Res<RenderQueue>,
    buffer: Option<Res<SharedLightBuffer>>,
    data: Option<Res<ShelterExtract>>,
    mut last: Local<(u64, [[f32; 4]; HEADER_ROWS])>,
) {
    let (Some(buffer), Some(data)) = (buffer, data) else {
        return;
    };
    if last.1 != data.header {
        last.1 = data.header;
        queue.write_buffer(
            &buffer.0,
            region_offset(),
            bytemuck::cast_slice(&data.header),
        );
    }
    if last.0 == data.generation || data.cells.is_empty() {
        return;
    }
    // Only the changed rows after a consecutive generation; everything after a skipped one.
    let (lo, hi) = match data.rows {
        Some(r) if last.0.wrapping_add(1) == data.generation => r,
        _ => (0, GRID),
    };
    last.0 = data.generation;
    let words = &data.cells[lo * GRID..hi * GRID];
    queue.write_buffer(
        &buffer.0,
        region_offset() + (HEADER_ROWS * 16 + lo * GRID * 4) as u64,
        bytemuck::cast_slice(words),
    );
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<ShelterGrid>()
        .init_resource::<ShelterExtract>()
        .add_plugins(ExtractResourcePlugin::<ShelterExtract>::default())
        // fork: it reads EnvFrame.wetness, which the resolve set's fog model also
        // writes; run it with wetness, before that set (an undeclared order otherwise).
        .add_systems(
            Update,
            tick_shelter
                .after(super::wetness::wetness_tick)
                .before(crate::lighting::LightingResolveSet),
        );
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            upload_shelter.in_set(RenderSystems::PrepareResources),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat ground at y 0 with a 6×6 yd porch roof (outer face 3.2, ceiling 3.0) at x, z 10..16.
    fn porch(xz: Vec2) -> impl FnMut(f32, f32, bool) -> Option<f32> {
        move |y: f32, max: f32, _first: bool| {
            let under = (10.0..16.0).contains(&xz.x) && (10.0..16.0).contains(&xz.y);
            let surfaces: &[f32] = if under { &[3.2, 3.0, 0.0] } else { &[0.0] };
            surfaces
                .iter()
                .find(|&&s| s <= y && y - s <= max)
                .map(|&s| y - s)
        }
    }

    fn filled(cam: Vec3) -> ShelterGrid {
        let mut g = ShelterGrid {
            active: true,
            ..Default::default()
        };
        g.follow(cam);
        let from = g.cast_y;
        while g.pending() > 0 {
            g.step(|xz| probe_column(from, porch(xz)));
        }
        g
    }

    #[test]
    fn a_column_under_a_roof_is_covered_and_open_ground_is_not() {
        let roof = probe_column(50.0, porch(Vec2::new(12.0, 12.0)));
        assert!(
            roof.covered && (roof.top.unwrap() - 3.2).abs() < 1e-4,
            "{roof:?}"
        );
        let open = probe_column(50.0, porch(Vec2::new(0.0, 0.0)));
        assert_eq!(
            open,
            Column {
                top: Some(0.0),
                covered: false
            }
        );
        let none = probe_column(50.0, |_, _, _| None);
        assert_eq!(
            none,
            Column {
                top: None,
                covered: false
            }
        );
    }

    /// A riverbed is covered at the water line; the shore above it, a roof over the river and dry
    /// ground keep their own answer.
    #[test]
    fn water_over_open_ground_covers_it_at_the_water_line() {
        let bed = Column {
            top: Some(-2.0),
            covered: false,
        };
        assert_eq!(
            under_liquid(bed, Some(0.0)),
            Column {
                top: Some(0.0),
                covered: true
            }
        );
        let shore = Column {
            top: Some(0.5),
            covered: false,
        };
        assert_eq!(under_liquid(shore, Some(0.0)), shore);
        let bridge = Column {
            top: Some(6.0),
            covered: true,
        };
        assert_eq!(under_liquid(bridge, Some(0.0)), bridge);
        assert_eq!(under_liquid(bed, None), bed);
    }

    #[test]
    fn a_thin_shell_is_not_shelter() {
        // Two faces 0.3 yd apart and nothing under them: a slab lying on the ground.
        let c = probe_column(10.0, |y, _, _| {
            [1.3f32, 1.0].iter().find(|&&s| s <= y).map(|&s| y - s)
        });
        assert!(!c.covered);
    }

    #[test]
    fn the_transform_round_trips_and_the_camera_sits_mid_grid() {
        for cam in [
            Vec2::new(0.0, 0.0),
            Vec2::new(-8_532.4, 671.9),
            Vec2::new(123.4, -9_876.5),
        ] {
            let mut g = ShelterGrid::default();
            g.follow(cam.extend(20.0).xzy());
            let (x, z) = g.cell_of(cam).expect("the camera is inside its grid");
            for c in [x, z] {
                assert!((48..80).contains(&c), "camera cell {c} too near an edge");
            }
            let centre = g.cell_centre(x, z);
            assert!((centre - cam).abs().max_element() <= CELL / 2.0 + 1e-3);
            assert_eq!(g.cell_of(centre), Some((x, z)));
        }
    }

    #[test]
    fn the_grid_recentres_only_in_whole_steps_and_keeps_its_columns() {
        let mut g = filled(Vec3::new(13.0, 1.0, 13.0));
        let origin = g.origin;
        // Within a step: nothing moves.
        assert!(!g.follow(Vec3::new(13.0 + 5.0, 1.0, 13.0)));
        assert_eq!(g.origin, origin);
        // Past it: one step, and only the new strip is pending.
        assert!(g.follow(Vec3::new(13.0 + STEP as f32 * CELL, 1.0, 13.0)));
        assert_eq!(g.origin - origin, IVec2::new(STEP, 0));
        assert_eq!(g.pending(), STEP as usize * GRID);
        // The porch kept its shelter across the shift.
        assert!(g.shelter_at(Vec3::new(13.0, 0.0, 13.0)) > 0.99);
        // A big climb restarts the grid.
        assert!(g.follow(Vec3::new(
            13.0 + STEP as f32 * CELL,
            1.0 + RECAST_DRIFT + 1.0,
            13.0
        )));
        assert_eq!(g.pending(), GRID * GRID);
    }

    #[test]
    fn a_frame_casts_at_most_its_budget_nearest_first() {
        let mut g = ShelterGrid {
            active: true,
            ..Default::default()
        };
        g.follow(Vec3::new(13.0, 1.0, 13.0));
        let mut calls = 0;
        let mut far = 0.0f32;
        let n = g.step(|xz| {
            calls += 1;
            far = far.max((xz - Vec2::new(13.0, 13.0)).length());
            Column {
                top: Some(0.0),
                covered: false,
            }
        });
        assert_eq!((n, calls), (FILL_BUDGET, FILL_BUDGET));
        // The first frame covers the camera's own neighbourhood (grid centre is ≤ a step away).
        assert!(far < 20.0, "first cells reach {far} yd");
        assert_eq!(g.pending(), GRID * GRID - FILL_BUDGET);
        while g.pending() > 0 {
            g.step(|_| Column {
                top: Some(0.0),
                covered: false,
            });
        }
        // Full: only the refresh trickle.
        assert_eq!(
            g.step(|_| Column {
                top: Some(0.0),
                covered: false
            }),
            REFRESH_BUDGET
        );
    }

    #[test]
    fn under_the_porch_is_dry_the_roof_and_the_drip_line_are_wet() {
        let g = filled(Vec3::new(13.0, 1.0, 13.0));
        // Ground under the middle of the porch: sheltered.
        assert!(g.shelter_at(Vec3::new(13.0, 0.0, 13.0)) > 0.99);
        // The roof's own top: open.
        assert_eq!(g.shelter_at(Vec3::new(13.0, 3.2, 13.0)), 0.0);
        // Open ground: open.
        assert_eq!(g.shelter_at(Vec3::new(5.0, 0.0, 5.0)), 0.0);
        // Just inside the eave (the drip margin): mostly wet; a yard in: mostly dry.
        let edge = g.shelter_at(Vec3::new(10.2, 0.0, 13.0));
        let inner = g.shelter_at(Vec3::new(11.8, 0.0, 13.0));
        assert!(edge < 0.5, "drip line {edge}");
        assert!(inner > edge && inner > 0.6, "inner {inner}");
        // At and just outside the eave: fully wet.
        assert_eq!(g.shelter_at(Vec3::new(9.6, 0.0, 13.0)), 0.0);
        assert_eq!(g.shelter_at(Vec3::new(10.1, 0.0, 13.0)), 0.0);
        // Walls and floors a story under the roof are dry too; a fragment above it is not.
        assert!(g.shelter_at(Vec3::new(13.0, 1.5, 13.0)) > 0.99);
        assert_eq!(g.shelter_at(Vec3::new(13.0, 4.0, 13.0)), 0.0);
    }

    #[test]
    fn an_inactive_grid_shelters_nothing_and_packs_off() {
        let mut g = filled(Vec3::new(13.0, 1.0, 13.0));
        g.active = false;
        assert_eq!(g.shelter_at(Vec3::new(13.0, 0.0, 13.0)), 0.0);
        assert_eq!(g.header()[0][3], 0.0);
    }

    #[test]
    fn packed_cells_round_trip_to_a_sixty_fourth_of_a_yard() {
        for (top, cover) in [
            (0.0f32, 0.0f32),
            (1.0, 1.0),
            (-2.5, 0.5),
            (3.21, 1.0 / 3.0),
            (-60.0, 0.0),
        ] {
            let (t, c) = unpack_cell(pack_cell(top, cover));
            assert!((t - top).abs() <= 0.5 / HEIGHT_SCALE, "{top} -> {t}");
            assert!((c - cover).abs() <= 1e-4, "{cover} -> {c}");
        }
        // Out of range clamps: a missing top reads 512 yd under the base.
        assert_eq!(unpack_cell(pack_cell(NO_TOP, 2.0)), (-512.0, 1.0));
    }

    #[test]
    fn the_region_is_aligned_for_the_probe_rows_after_it() {
        assert_eq!(REGION_BYTES % 16, 0);
        assert_eq!(region_offset() % 16, 0);
        assert_eq!(
            crate::lighting::prop_probe_region_offset(),
            region_offset() + REGION_BYTES
        );
    }
}
