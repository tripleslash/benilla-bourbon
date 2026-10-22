//! The terrain horizon (`systems/terrain/cull.md#horizon-buffer`): 320 screen columns, each the
//! highest projected `y/z` a nearer terrain chunk reached in it, and the tests that hide a chunk,
//! an object or a far tile lying below it in every column it spans (`0x686000`, `0x686180`).
//!
//! The reference fills it during the band walk (`0x682fa0`): 32 bands of 33⅓ yd front to back,
//! each testing its objects and chunks, then adding its visible chunks' rows (`0x683bf0`). Here the
//! chunk walk runs once a frame and keeps the buffer as each band found it, so a test anywhere in
//! the frame reads what the walk order gave its band. World flag 0x20 (`showCull`) and a camera
//! forward within ±0.9 of level gate everything (`0x681b79`, `0x681b90`).
//!
//! Inside a building the reference walks once per portal window (`0x681130`): the buffer reset, the
//! frustum narrowed to the window and the occluder feed starting at the window's reach plus
//! 33.333332 (`0x6811a9`), so terrain short of the doorway never occludes. Here each window keeps
//! its own walk, and an object is hidden only when every window that admits it hides it.
//!
//! A chunk's occluder rows (`CMapChunk::AddHorizonRows`, `0x6c15d0`) are its outer row and column
//! away from the eye: row 8 when the camera target's x is below the eye's (`[0xc7d118] <
//! [0xc7cf20]`), column 8 when its y is (`[0xc7d11c] < [0xc7cf24]`); and the chunk files into its
//! band by the corner `0x86b580[q]` = {0, 8, 0x88, 0x90}, `q` = `[0xc7f294]`, built from the same
//! two compares (`0x681026` to `0x681059`). Each row takes `[0x810f60 + 4·level]` = {2, 3, 5, 9}
//! vertices at stride `[0x810f70 + 4·level]` = {8, 4, 2, 1}; the level is 3, or 2 when the `lod`
//! CVar (world flag 4) is on and the chunk's view depth is above `lodDist` (`0x6afb06` to
//! `0x6afb2a`).

use std::sync::Arc;

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::AdtTile;
use bevy::camera::primitives::Aabb;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use bevy::camera::primitives::Frustum;

use crate::exterior_cull::ExteriorCullSet;
use crate::terrain_stream::{TerrainCell, TerrainStreamer};
use crate::view::{ViewDistance, WorldCamera};
use crate::wmo_portal::{ExteriorWindowReach, ExteriorWindows, Rect};
use crate::world_flags::WorldFlags;

/// The buffer's columns, 0x140.
pub const HORIZON_COLS: usize = 320;
type Columns = [f32; HORIZON_COLS];

/// The depth bands, `0xc7bd40`.
const BANDS: usize = 32;
/// `0x685fe0`'s fill, 0xc9742400.
const RESET: f32 = -1_000_000.0;
/// `[0x810170]`.
const COL_SCALE: f32 = 64.0;
/// Added after the rounding; the tests' span ends one further.
const COL_BIAS: i32 = 0xa0;
const COL_MAX: i32 = 0x13f;
/// `[0x810180]`, 0x4031c71c: the least z of an occluder vertex.
const Z_GATE_INSERT: f32 = f32::from_bits(0x4031_c71c);
/// `[0x80fed4]`: the least z of a tested point.
const Z_GATE_TEST: f32 = 50.0;
/// `[0x8029d4]`, 2^-22: the least radius `0x686000` tests.
const RADIUS_GATE: f32 = f32::from_bits(0x3480_0000);
/// `0x681b79`, `0x681b90`: the camera's forward vertical component, both ends included.
const FORWARD_LIMIT: f32 = 0.9;
/// `0x6810d8`: the occluder feed's far end is the far clip less this.
const OCCLUDER_FAR_MARGIN: f32 = 33.333_332;
/// `[0x810174]`: bands per yard.
const BAND_SCALE: f32 = 0.03;
/// Test flag bit 3: a point nearer than z = 50 does not make the box visible (the far tiles).
pub const FLAG_NEAR_OK: u32 = 8;
/// A holes word with every block cut: the chunk is dropped (`0x683c28`).
const ALL_HOLES: u16 = 0xffff;

/// How the horizon culls while world flag 0x20 ([`WorldFlags::CULL`], `showCull`) is set.
/// `$WOW_HORIZON=count` walks and counts without hiding.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct HorizonCulling {
    /// Off under `count`: every verdict is computed and counted, none applied.
    pub enforce: bool,
    /// The `lod` CVar, world flag 4 (`0x688920`): a chunk past [`Self::lod_dist`] adds its rows at
    /// level 2. Registered "1".
    pub lod: bool,
    /// The `lodDist` CVar (`0x672680`, `[0x867960]`), registered "100.0", shipped
    /// [`LOD_DIST_SHIPPED`]. The drawn chunk level and the horizon rows read this one knob.
    pub lod_dist: f32,
}

/// `lodDist`'s registered value (`0x688524`).
pub const LOD_DIST_DEFAULT: f32 = 100.0;
/// The `lodDist` benilla ships (the CVar table's `Deviates` row): with the far clip at 1000 yd,
/// the reference's 100 would draw most visible relief at level 2.
pub const LOD_DIST_SHIPPED: f32 = 500.0;
/// The range `lodDist`'s callback (`0x688ab0`) accepts; outside it, it prints its line and stores
/// [`LOD_DIST_FALLBACK`] (`0x688af4`).
///
/// Deviation: the reference accepts 50 to 250; the top is raised to the fork's 1000 yd far clip so
/// the shipped [`LOD_DIST_SHIPPED`] passes its own check.
pub const LOD_DIST_RANGE: std::ops::RangeInclusive<f32> = 50.0..=1000.0;
/// `0x429a0000`, what an out-of-range `lodDist` stores.
pub const LOD_DIST_FALLBACK: f32 = 77.0;
/// `[0x867964]`, `maxLOD`: a chunk's level.
const LEVEL_NEAR: usize = 3;
/// `[0x867968]`: a chunk's level past `lodDist` while `lod` is on.
const LEVEL_FAR: usize = 2;
/// `[0x810f60]`: the vertices a level's row takes.
const ROW_COUNT: [usize; 4] = [2, 3, 5, 9];
/// `[0x810f70]`: their stride along the outer grid.
const ROW_STRIDE: [usize; 4] = [8, 4, 2, 1];

impl HorizonCulling {
    /// As the reference starts: culling enforced, terrain LOD on, at the shipped `lodDist`.
    const REFERENCE: Self = Self {
        enforce: true,
        lod: true,
        lod_dist: LOD_DIST_SHIPPED,
    };

    /// The depth past which a chunk adds its rows at level 2, `None` while `lod` is off.
    fn lod_depth(&self) -> Option<f32> {
        self.lod.then_some(self.lod_dist)
    }
}

impl Default for HorizonCulling {
    fn default() -> Self {
        match crate::launch_env::var("WOW_HORIZON").as_deref() {
            Some("count") => Self {
                enforce: false,
                ..Self::REFERENCE
            },
            _ => Self::REFERENCE,
        }
    }
}

/// What the horizon hid last frame, per consumer, for `$WOW_HORIZON_CENSUS`.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct HorizonCensus {
    pub cells: usize,
    pub liquid: usize,
    pub far: usize,
    pub bodies: usize,
    /// Entity-path model parts, of those the visibility walk re-evaluated.
    pub parts: usize,
    pub gx_cells: usize,
    /// Retained WMO groups and prop sets.
    pub gx_groups: usize,
}

/// The chunk walk's counts, summed over the frame's walks.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct WalkStats {
    /// Chunks in view and not all holes.
    pub chunks: usize,
    pub chunks_hidden: usize,
    pub occluders: usize,
    pub cells_hidden: usize,
}

/// The level camera the horizon projects through (`0xc7b700`, `cull.md#horizon-view`): the eye
/// looking along the camera's horizontal heading, the real lens.
#[derive(Clone, Copy, PartialEq, Debug)]
struct HorizonView {
    eye: Vec3,
    /// The horizontal heading, also the band plane's normal (`0xc7cfb8`).
    heading: Vec3,
    right: Vec3,
    /// The camera's own forward, the view plane's normal (`0xc7bcb0`).
    forward: Vec3,
    xscale: f32,
    yscale: f32,
    q: f32,
    near: f32,
}

impl HorizonView {
    /// `None` for a heading too near vertical (`0x680e50`), which the forward gate refuses anyway.
    fn new(eye: Vec3, forward: Vec3, fov: f32, aspect: f32, near: f32, far: f32) -> Option<Self> {
        let flat = Vec3::new(forward.x, 0.0, forward.z);
        if flat.length_squared() <= 1e-4 {
            return None;
        }
        let heading = flat.normalize();
        let yscale = 1.0 / (fov * 0.5).tan();
        Some(Self {
            eye,
            heading,
            right: heading.cross(Vec3::Y),
            forward,
            xscale: yscale / aspect,
            yscale,
            q: far / (far - near),
            near,
        })
    }

    /// A point through the matrix, before any divide: `TransformPoint` keeps x, y and z.
    fn project(&self, p: Vec3) -> [f32; 3] {
        let d = p - self.eye;
        [
            d.dot(self.right) * self.xscale,
            d.y * self.yscale,
            self.q * (d.dot(self.heading) - self.near),
        ]
    }

    /// `(x/z, y/z, z)`, through one reciprocal as `0x681c0d` and `0x686250` store them.
    fn screen(&self, p: Vec3) -> [f32; 3] {
        let [x, y, z] = self.project(p);
        let inv = 1.0 / f64::from(z);
        [(inv * f64::from(x)) as f32, (inv * f64::from(y)) as f32, z]
    }

    /// Distance along the heading, the band key's depth (`cull.md#depth-buckets`).
    fn band_depth(&self, p: Vec3) -> f32 {
        (p - self.eye).dot(self.heading)
    }

    /// Distance along the camera's forward, the chunk's `+0x78`.
    fn view_depth(&self, p: Vec3) -> f32 {
        (p - self.eye).dot(self.forward)
    }
}

/// x87 `fistp` in round-to-nearest: out of range or NaN stores the integer indefinite.
fn fistp(v: f64) -> i32 {
    let r = v.round_ties_even();
    if (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&r) {
        r as i32
    } else {
        i32::MIN
    }
}

/// A column from `x/z`: `fistp(f32(64·x) − 0.5) + bias`, wrapping.
fn col(x: f32, bias: i32) -> i32 {
    fistp(f64::from(COL_SCALE * x) - 0.5).wrapping_add(bias)
}

/// The band of a depth: round-to-nearest of `depth·0.03 − 0.5`; negative is band 0, and a band of
/// 32 or more is not filed.
fn band_of(depth: f32) -> Option<usize> {
    let k = fistp(f64::from((f64::from(depth) * f64::from(BAND_SCALE)) as f32) - 0.5);
    let k = usize::try_from(k.max(0)).unwrap_or(0);
    (k < BANDS).then_some(k)
}

/// `AddHorizonOccluder` (`0x681b50`) on projected points: each edge whose ends both have z of at
/// least 2.7777777 raises every column between its ends' to the lower end's `y/z`.
fn insert(buf: &mut Columns, pts: &[[f32; 3]]) {
    for e in pts.windows(2) {
        let ([x0, y0, z0], [x1, y1, z1]) = (e[0], e[1]);
        if z0 < Z_GATE_INSERT || z1 < Z_GATE_INSERT {
            continue;
        }
        // A tie or NaN takes the second (`0x681c69`).
        let ymin = if y0 < y1 { y0 } else { y1 };
        let (c0, c1) = (col(x0, COL_BIAS), col(x1, COL_BIAS));
        let (lo, hi) = if c1 < c0 { (c1, c0) } else { (c0, c1) };
        for c in lo.max(0)..=hi.min(COL_MAX) {
            let v = &mut buf[c as usize];
            if *v < ymin {
                *v = ymin;
            }
        }
    }
}

/// The tests' shared tail: a span starting at 320 or ending below 0 is visible, else hidden when
/// no clamped column is below `top` (a NaN column is not).
fn below(buf: &Columns, lo: i32, hi: i32, top: f32) -> bool {
    if lo >= HORIZON_COLS as i32 || hi < 0 {
        return false;
    }
    (lo.max(0)..=hi.min(COL_MAX))
        .all(|c| buf[c as usize].partial_cmp(&top).is_none_or(|o| o.is_ge()))
}

/// `TestHorizonBox` (`0x686180`): the eight corners' least and greatest `x/z` and greatest `y/z`.
fn box_hidden(view: &HorizonView, buf: &Columns, min: Vec3, max: Vec3, flags: u32) -> bool {
    let (mut min_x, mut max_x, mut max_y) = (f32::MAX, f32::MIN, f32::MIN);
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        let [sx, sy, z] = view.screen(corner);
        if flags & FLAG_NEAR_OK == 0 && z < Z_GATE_TEST {
            return false;
        }
        if sx < min_x {
            min_x = sx;
        }
        if max_x < sx {
            max_x = sx;
        }
        if max_y < sy {
            max_y = sy;
        }
    }
    below(buf, col(min_x, COL_BIAS), col(max_x, COL_BIAS + 1), max_y)
}

/// `TestHorizonSphere` (`0x686000`). The top adds the radius scaled by the lens's x term and the
/// span uses its y term, as the reference crosses them (`0x686086`).
fn sphere_hidden(view: &HorizonView, buf: &Columns, center: Vec3, radius: f32) -> bool {
    if radius.abs() < RADIUS_GATE {
        return false;
    }
    let [x, y, z] = view.project(center);
    if z < Z_GATE_TEST {
        return false;
    }
    let inv = 1.0 / f64::from(z);
    let cx = (inv * f64::from(x)) as f32;
    let rx = (inv * f64::from(radius * view.xscale)) as f32;
    let ry = (inv * f64::from(radius * view.yscale)) as f32;
    let top = (f64::from(y) * inv + f64::from(rx)) as f32;
    below(buf, col(cx - ry, COL_BIAS), col(ry + cx, COL_BIAS + 1), top)
}

/// A portal window's sub-frustum, compared by the rect and lens that built it.
#[derive(Clone)]
struct WindowVolume {
    rect: Rect,
    clip_from_world: Mat4,
    frustum: Frustum,
}

impl PartialEq for WindowVolume {
    fn eq(&self, other: &Self) -> bool {
        self.rect == other.rect && self.clip_from_world == other.clip_from_world
    }
}

/// One scene walk's horizon (`0x6811b6`): the buffer as each band found it.
#[derive(Clone, PartialEq)]
struct WindowWalk {
    view: HorizonView,
    /// The portal window the walk ran through; `None` outdoors, where the window is the view.
    window: Option<WindowVolume>,
    /// The buffer before band `b`'s chunks went in, then after the last band (the far tiles').
    snapshots: Vec<Columns>,
}

impl WindowWalk {
    /// Whether this walk's window admits a world box: a walk tests only what it reaches.
    fn admits(&self, min: Vec3, max: Vec3) -> bool {
        self.window.as_ref().is_none_or(|w| {
            w.frustum.intersects_obb(
                &Aabb::from_min_max(min, max),
                &bevy::math::Affine3A::IDENTITY,
                true,
                true,
            )
        })
    }
}

/// The frame's horizon: one walk outdoors, one per portal window inside, and the terrain cells
/// they hid.
#[derive(Resource, Default, PartialEq)]
pub struct Horizon {
    /// Empty while the gates are shut.
    walks: Vec<WindowWalk>,
    enforce: bool,
    hidden_cells: HashSet<((i32, i32), u8)>,
    /// The lowest resident chunk's floor, Bevy y: no terrain the walk saw lies under it.
    floor: f32,
    pub stats: WalkStats,
}

impl Horizon {
    /// Whether a walk ran this frame; when not, every test answers visible.
    pub fn active(&self) -> bool {
        !self.walks.is_empty()
    }

    /// Whether a hidden verdict is applied, not only counted.
    pub fn enforced(&self) -> bool {
        self.enforce && self.active()
    }

    /// Hidden when at least one walk's window admits the box and every such walk hides it: the
    /// reference draws an object from any window's walk that passes it.
    fn every_walk_hides(&self, min: Vec3, max: Vec3, hides: impl Fn(&WindowWalk) -> bool) -> bool {
        let mut reached = false;
        for w in self.walks.iter().filter(|w| w.admits(min, max)) {
            if !hides(w) {
                return false;
            }
            reached = true;
        }
        reached
    }

    /// A world bounding sphere, filed by its nearest point (`0x6816f0`, `0x681a40`).
    pub fn hides_sphere(&self, center: Vec3, radius: f32) -> bool {
        let r = Vec3::splat(radius.abs());
        self.every_walk_hides(center - r, center + r, |w| {
            band_of(w.view.band_depth(center) - radius)
                .is_some_and(|band| sphere_hidden(&w.view, &w.snapshots[band], center, radius))
        })
    }

    /// A world box, filed by its bounding sphere's nearest point. A liquid slot files by its
    /// centre (`0x681970`), a later band; the nearer band never hides more than the reference.
    pub fn hides_box(&self, min: Vec3, max: Vec3, flags: u32) -> bool {
        let center = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        self.every_walk_hides(min, max, |w| {
            band_of(w.view.band_depth(center) - radius)
                .is_some_and(|band| box_hidden(&w.view, &w.snapshots[band], min, max, flags))
        })
    }

    /// A far-band tile's box, tested after every band with flag 8 (`0x6832c8`).
    pub fn hides_far_box(&self, min: Vec3, max: Vec3) -> bool {
        self.every_walk_hides(min, max, |w| {
            box_hidden(&w.view, &w.snapshots[BANDS], min, max, FLAG_NEAR_OK)
        })
    }

    /// A terrain cell none of whose chunks in any window's view survived the horizon.
    pub fn hides_cell(&self, cell: &TerrainCell) -> bool {
        self.hidden_cells.contains(&(cell.tile, cell.cell))
    }

    /// The lowest resident terrain, Bevy y, for a test that sweeps a box down to the ground.
    pub fn floor(&self) -> f32 {
        self.floor
    }

    /// A level wall at `slope` (`y/z`) in every column of every band, seen from `eye` along
    /// `heading` through a square 90° lens, enforced. Built by tests outside this crate, which
    /// have no terrain to walk; the frame never builds one.
    pub fn level_wall(eye: Vec3, heading: Vec3, slope: f32) -> Self {
        let Some(view) =
            HorizonView::new(eye, heading, std::f32::consts::FRAC_PI_2, 1.0, 0.0, 1000.0)
        else {
            return Self::default();
        };
        Self {
            walks: vec![WindowWalk {
                view,
                window: None,
                snapshots: vec![[slope; HORIZON_COLS]; BANDS + 1],
            }],
            enforce: true,
            hidden_cells: HashSet::default(),
            floor: -1000.0,
            stats: WalkStats::default(),
        }
    }
}

/// An entity's world-space box from its transform and local `Aabb`.
pub fn world_box(gt: &GlobalTransform, aabb: &Aabb) -> (Vec3, Vec3) {
    let affine = gt.affine();
    let center = affine.transform_point3a(aabb.center);
    let m = affine.matrix3;
    let he = aabb.half_extents;
    let half = m.x_axis.abs() * he.x + m.y_axis.abs() * he.y + m.z_axis.abs() * he.z;
    (Vec3::from(center - half), Vec3::from(center + half))
}

/// A chunk's AABB over its 145 vertices (the box copy `+0x94`), in Bevy space.
#[derive(Clone, Copy)]
struct ChunkBox {
    min: Vec3,
    max: Vec3,
    holes: u16,
}

/// Each resident tile's chunk boxes, built once per asset.
type ChunkBoxCache = HashMap<AssetId<AdtTile>, Arc<[Option<ChunkBox>]>>;

fn chunk_boxes(tile: &AdtTile) -> Arc<[Option<ChunkBox>]> {
    tile.chunks
        .iter()
        .map(|c| {
            let mut pts = c.positions.iter().map(|p| wow_to_bevy(*p));
            let first = pts.next()?;
            let (min, max) = pts.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
            Some(ChunkBox {
                min,
                max,
                holes: c.holes,
            })
        })
        .collect()
}

/// One chunk as the walk reads it.
#[derive(Clone, Copy)]
struct Chunk<'a> {
    cell: ((i32, i32), u8),
    b: ChunkBox,
    /// The 145 vertices in WoW yards, stride-17 order.
    positions: &'a [[f32; 3]],
}

struct Walk {
    snapshots: Vec<Columns>,
    /// Cells with a chunk hidden and none drawn.
    hidden_cells: HashSet<((i32, i32), u8)>,
    /// Cells with a chunk drawn, or in view and not filed: another window's walk cannot hide them.
    shown_cells: HashSet<((i32, i32), u8)>,
    stats: WalkStats,
}

/// [`walk_from`] with the outdoor feed, which starts at the eye (`0x6811ca` stores 0).
#[cfg(test)]
fn walk<'a>(
    view: &HorizonView,
    occluder_far: f32,
    lod_depth: Option<f32>,
    chunks: impl Iterator<Item = Chunk<'a>>,
    in_view: impl Fn(Vec3, Vec3) -> bool,
) -> Walk {
    walk_from(view, 0.0, occluder_far, lod_depth, chunks, in_view)
}

/// `CullSceneChunks` (`0x683bf0`) over every band, front to back: a chunk all holes or out of view
/// is dropped; one the horizon hides is not drawn; a visible one without holes whose view depth
/// lies in `[occluder_near, occluder_far)` adds its rows after its band's tests (`0x683d53`,
/// `0x683d67`).
fn walk_from<'a>(
    view: &HorizonView,
    occluder_near: f32,
    occluder_far: f32,
    lod_depth: Option<f32>,
    chunks: impl Iterator<Item = Chunk<'a>>,
    in_view: impl Fn(Vec3, Vec3) -> bool,
) -> Walk {
    // Row 8 is south (WoW x falls, Bevy z rises), column 8 east (WoW y falls, Bevy x rises).
    let far_row = if view.heading.z > 0.0 { 8 } else { 0 };
    let far_col = if view.heading.x > 0.0 { 8 } else { 0 };
    let mut stats = WalkStats::default();
    let mut bands: Vec<Vec<Chunk>> = (0..BANDS).map(|_| Vec::new()).collect();
    // Per cell: a chunk drawn (or not filed, so not this walk's to hide), a chunk hidden.
    let mut cells: HashMap<((i32, i32), u8), (bool, bool)> = HashMap::default();
    for ch in chunks {
        if ch.b.holes == ALL_HOLES || ch.positions.len() < 145 || !in_view(ch.b.min, ch.b.max) {
            continue;
        }
        stats.chunks += 1;
        let corner = wow_to_bevy(ch.positions[far_row * 17 + far_col]);
        match band_of(view.band_depth(corner)) {
            Some(b) => bands[b].push(ch),
            None => cells.entry(ch.cell).or_default().0 = true,
        }
    }
    let mut buf: Columns = [RESET; HORIZON_COLS];
    let mut snapshots = Vec::with_capacity(BANDS + 1);
    for band in &bands {
        snapshots.push(buf);
        let mut occluders: Vec<(&Chunk, usize)> = Vec::new();
        for ch in band {
            let seen = cells.entry(ch.cell).or_default();
            if box_hidden(view, &buf, ch.b.min, ch.b.max, 0) {
                stats.chunks_hidden += 1;
                seen.1 = true;
                continue;
            }
            seen.0 = true;
            if ch.b.holes != 0 {
                continue;
            }
            // The chunk's `+0x78`: its sphere's view depth, nearest point.
            let center = (ch.b.min + ch.b.max) * 0.5;
            let depth = view.view_depth(center) - (ch.b.max - ch.b.min).length() * 0.5;
            if (occluder_near..occluder_far).contains(&depth) {
                occluders.push((ch, level_of(depth, lod_depth)));
            }
        }
        for (ch, level) in occluders {
            stats.occluders += 1;
            let (count, stride) = (ROW_COUNT[level], ROW_STRIDE[level]);
            let pick = |i: usize| view.screen(wow_to_bevy(ch.positions[i]));
            let row: Vec<[f32; 3]> = (0..count)
                .map(|k| pick(far_row * 17 + k * stride))
                .collect();
            insert(&mut buf, &row);
            let column: Vec<[f32; 3]> = (0..count)
                .map(|k| pick(k * stride * 17 + far_col))
                .collect();
            insert(&mut buf, &column);
        }
    }
    snapshots.push(buf);
    let (mut hidden_cells, mut shown_cells) = (HashSet::default(), HashSet::default());
    for (k, (shown, hidden)) in cells {
        if shown {
            shown_cells.insert(k);
        } else if hidden {
            hidden_cells.insert(k);
        }
    }
    stats.cells_hidden = hidden_cells.len();
    Walk {
        snapshots,
        hidden_cells,
        shown_cells,
        stats,
    }
}

/// A chunk's level by its view depth (`0x6afb06` to `0x6afb2a`): [`LEVEL_FAR`] only when terrain
/// LOD is on and the depth is strictly above `lodDist`, a NaN staying near.
fn level_of(depth: f32, lod_depth: Option<f32>) -> usize {
    match lod_depth {
        Some(dist) if depth > dist => LEVEL_FAR,
        _ => LEVEL_NEAR,
    }
}

impl HorizonCulling {
    /// The level a chunk draws at (`0x6afad0`): 2 past `lodDist` while `lod` is on, else 3, by its
    /// sphere's view depth `f·(centre − eye) − radius`. The horizon's occluder rows take the same
    /// level, so the draw and the feed always agree.
    pub fn chunk_level(&self, view_depth: f32) -> u8 {
        level_of(view_depth, self.lod_depth()) as u8
    }
}

/// Everything the walk reads: when it matches the last frame's, the walk would rebuild the same
/// [`Horizon`]. A tile's chunks are fixed per asset id, which [`ChunkBoxCache`] already relies on.
#[derive(PartialEq)]
struct WalkInputs {
    culling: HorizonCulling,
    /// World flag 0x20.
    cull: bool,
    windows: ExteriorWindows,
    reach: Vec<f32>,
    farclip: f32,
    eye: Option<(Mat4, Option<[f32; 4]>)>,
    /// Resident tiles, and whether each one's asset is loaded.
    resident: Vec<((i32, i32), AssetId<AdtTile>, bool)>,
}

/// Builds [`Horizon`] from the world camera and the resident tiles' chunks; writes only on change,
/// so a still scene leaves the model-visibility skip armed, and walks only when an input moved.
fn build_horizon(
    culling: Res<HorizonCulling>,
    flags: Option<Res<WorldFlags>>,
    windows: Res<ExteriorWindows>,
    reach: Option<Res<ExteriorWindowReach>>,
    view_dist: Res<ViewDistance>,
    cam: Query<(&GlobalTransform, &Projection), With<WorldCamera>>,
    streamer: Option<Res<TerrainStreamer>>,
    tiles: Option<Res<Assets<AdtTile>>>,
    mut boxes: Local<ChunkBoxCache>,
    mut last: Local<Option<WalkInputs>>,
    mut horizon: ResMut<Horizon>,
) {
    let inputs = WalkInputs {
        culling: *culling,
        cull: flags
            .as_deref()
            .copied()
            .unwrap_or_default()
            .has(WorldFlags::CULL),
        windows: windows.clone(),
        reach: reach.as_deref().map_or_else(Vec::new, |r| r.0.clone()),
        farclip: view_dist.farclip,
        eye: cam.iter().next().map(|(t, p)| {
            let lens = match p {
                Projection::Perspective(p) => Some([p.fov, p.aspect_ratio, p.near, p.far]),
                _ => None,
            };
            (t.to_matrix(), lens)
        }),
        resident: streamer.as_deref().map_or_else(Vec::new, |s| {
            s.spawned_tiles()
                .map(|(key, h)| (key, h.id(), tiles.as_deref().is_some_and(|t| t.contains(h))))
                .collect()
        }),
    };
    if last.as_ref() == Some(&inputs) {
        return;
    }
    let next = (|| {
        if !inputs.cull {
            return None;
        }
        let (cam_t, proj) = cam.iter().next()?;
        let Projection::Perspective(p) = proj else {
            return None;
        };
        let forward = Vec3::from(cam_t.forward());
        // NaN passes, as `test ah,5; jnp` and `test ah,0x41; je` leave it.
        if forward.y < -FORWARD_LIMIT || forward.y > FORWARD_LIMIT {
            return None;
        }
        let view = HorizonView::new(
            cam_t.translation(),
            forward,
            p.fov,
            p.aspect_ratio,
            p.near,
            p.far,
        )?;
        let (streamer, tiles) = (streamer.as_deref()?, tiles.as_deref()?);
        let clip_from_world = proj.get_clip_from_view() * cam_t.to_matrix().inverse();
        // Outdoors one walk from the eye on the full rect; inside, one per window from its reach
        // plus the margin, none in a sealed room (`0x681199`). A window under the scene walk's
        // width gate walks nothing.
        let walks: Vec<(Option<WindowVolume>, f32)> = match &*windows {
            ExteriorWindows::Unrestricted => vec![(None, 0.0)],
            ExteriorWindows::Windows(rects) => rects
                .iter()
                .enumerate()
                .filter_map(|(i, rect)| {
                    let frustum =
                        crate::exterior_cull::scene_window_frustum(*rect, &clip_from_world)?;
                    let near = inputs.reach.get(i).copied().unwrap_or(0.0) + OCCLUDER_FAR_MARGIN;
                    let volume = WindowVolume {
                        rect: *rect,
                        clip_from_world,
                        frustum,
                    };
                    Some((Some(volume), near))
                })
                .collect(),
        };
        if walks.is_empty() {
            return None;
        }
        let full =
            crate::exterior_cull::scene_window_frustum([-1.0, -1.0, 1.0, 1.0], &clip_from_world)?;
        let resident: Vec<((i32, i32), &AdtTile, AssetId<AdtTile>)> = streamer
            .spawned_tiles()
            .filter_map(|(key, h)| Some((key, tiles.get(h)?, h.id())))
            .collect();
        boxes.retain(|id, _| resident.iter().any(|(_, _, r)| r == id));
        for (_, tile, id) in &resident {
            boxes.entry(*id).or_insert_with(|| chunk_boxes(tile));
        }
        let chunks: Vec<Chunk> = resident
            .iter()
            .flat_map(|(key, tile, id)| {
                let b = boxes.get(id).cloned();
                tile.chunks.iter().enumerate().filter_map(move |(i, c)| {
                    let cell = ((i / 16) / 4 * 4 + (i % 16) / 4) as u8;
                    Some(Chunk {
                        cell: (*key, cell),
                        b: (*b.as_ref()?.get(i)?)?,
                        positions: &c.positions,
                    })
                })
            })
            .collect();
        let mut out = Horizon {
            enforce: culling.enforce,
            floor: chunks
                .iter()
                .map(|c| c.b.min.y)
                .fold(f32::INFINITY, f32::min),
            ..Horizon::default()
        };
        let mut shown: HashSet<((i32, i32), u8)> = HashSet::default();
        for (window, near) in walks {
            let volume = window.as_ref().map_or(&full, |w| &w.frustum);
            let in_view = |min: Vec3, max: Vec3| {
                volume.intersects_obb(
                    &Aabb::from_min_max(min, max),
                    &bevy::math::Affine3A::IDENTITY,
                    true,
                    true,
                )
            };
            let w = walk_from(
                &view,
                near,
                view_dist.farclip - OCCLUDER_FAR_MARGIN,
                culling.lod_depth(),
                chunks.iter().copied(),
                in_view,
            );
            out.stats.chunks += w.stats.chunks;
            out.stats.chunks_hidden += w.stats.chunks_hidden;
            out.stats.occluders += w.stats.occluders;
            out.hidden_cells.extend(w.hidden_cells);
            shown.extend(w.shown_cells);
            out.walks.push(WindowWalk {
                view,
                window,
                snapshots: w.snapshots,
            });
        }
        // A cell another window's walk drew is drawn.
        out.hidden_cells.retain(|k| !shown.contains(k));
        out.stats.cells_hidden = out.hidden_cells.len();
        Some(out)
    })()
    .unwrap_or_default();
    *last = Some(inputs);
    horizon.set_if_neq(next);
}

/// `$WOW_HORIZON_CENSUS=1`: one line every 64 frames.
fn print_census(horizon: Res<Horizon>, census: Res<HorizonCensus>, mut frame: Local<u32>) {
    *frame = frame.wrapping_add(1);
    if !frame.is_multiple_of(64) {
        return;
    }
    let (s, c) = (horizon.stats, *census);
    info!(
        "HORIZON active={} enforce={} chunks={} chunks_hidden={} occluders={} cells={} \
         liquid={} far={} bodies={} parts={} gx_cells={} gx_groups={}",
        horizon.active(),
        horizon.enforced(),
        s.chunks,
        s.chunks_hidden,
        s.occluders,
        c.cells,
        c.liquid,
        c.far,
        c.bodies,
        c.parts,
        c.gx_cells,
        c.gx_groups,
    );
}

/// Ordering handle: [`Horizon`] is this frame's after this set.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HorizonSet;

pub(crate) struct HorizonPlugin;

impl Plugin for HorizonPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HorizonCulling>()
            .init_resource::<Horizon>()
            .init_resource::<HorizonCensus>()
            .add_systems(
                PostUpdate,
                build_horizon
                    .in_set(HorizonSet)
                    .after(bevy::transform::TransformSystems::Propagate)
                    // The lens's aspect ratio is this frame's.
                    .after(bevy::camera::CameraUpdateSystems)
                    // It reads the world camera's transform; the particle models write their own.
                    .ambiguous_with(crate::billboard::BillboardPlace)
                    .before(ExteriorCullSet),
            );
        if crate::launch_env::var("WOW_HORIZON_CENSUS").is_some() {
            app.add_systems(Last, print_census);
        }
    }
}

#[cfg(test)]
mod tests;
