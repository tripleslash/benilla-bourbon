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
//! Outdoors only: inside a building the reference walks once per portal window with the occluder
//! feed at the window's plane (`0x6811a9`); a window frame here tests nothing.
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

use crate::exterior_cull::ExteriorCullSet;
use crate::terrain_stream::{TerrainCell, TerrainStreamer};
use crate::view::{ViewDistance, WorldCamera};
use crate::wmo_portal::ExteriorWindows;

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

/// World flag 0x20, the `showCull` toggle: on by default, as `0x66f6cb` starts it.
/// `$WOW_HORIZON=0` starts it off; `$WOW_HORIZON=count` walks and counts without hiding.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct HorizonCulling {
    pub enabled: bool,
    /// Off under `count`: every verdict is computed and counted, none applied.
    pub enforce: bool,
    /// The `lod` CVar, world flag 4 (`0x688920`): a chunk past [`Self::lod_dist`] adds its rows at
    /// level 2. Registered "1".
    pub lod: bool,
    /// The `lodDist` CVar (`0x672680`, `[0x867960]`), registered "100.0".
    pub lod_dist: f32,
}

/// `lodDist`'s registered value (`0x688524`).
pub const LOD_DIST_DEFAULT: f32 = 100.0;
/// The range `lodDist`'s callback (`0x688ab0`) accepts; outside it, it prints its line and stores
/// [`LOD_DIST_FALLBACK`] (`0x688af4`).
pub const LOD_DIST_RANGE: std::ops::RangeInclusive<f32> = 50.0..=250.0;
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
    /// As the reference starts: culling on and enforced, terrain LOD on at `lodDist` 100.
    const REFERENCE: Self = Self {
        enabled: true,
        enforce: true,
        lod: true,
        lod_dist: LOD_DIST_DEFAULT,
    };

    /// The depth past which a chunk adds its rows at level 2, `None` while `lod` is off.
    fn lod_depth(&self) -> Option<f32> {
        self.lod.then_some(self.lod_dist)
    }
}

impl Default for HorizonCulling {
    fn default() -> Self {
        match crate::launch_env::var("WOW_HORIZON").as_deref() {
            Some("0") => Self {
                enabled: false,
                ..Self::REFERENCE
            },
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

/// The chunk walk's counts.
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

/// The frame's horizon: the buffer as each band found it, and the terrain cells it hid.
#[derive(Resource, Default, PartialEq)]
pub struct Horizon {
    /// `Some` while the gates are open.
    view: Option<HorizonView>,
    enforce: bool,
    /// The buffer before band `b`'s chunks went in, then after the last band (the far tiles').
    snapshots: Vec<Columns>,
    hidden_cells: HashSet<((i32, i32), u8)>,
    pub stats: WalkStats,
}

impl Horizon {
    /// Whether the walk ran this frame; when not, every test answers visible.
    pub fn active(&self) -> bool {
        self.view.is_some()
    }

    /// Whether a hidden verdict is applied, not only counted.
    pub fn enforced(&self) -> bool {
        self.enforce && self.active()
    }

    /// A world bounding sphere, filed by its nearest point (`0x6816f0`, `0x681a40`).
    pub fn hides_sphere(&self, center: Vec3, radius: f32) -> bool {
        let Some(view) = &self.view else {
            return false;
        };
        let Some(band) = band_of(view.band_depth(center) - radius) else {
            return false;
        };
        sphere_hidden(view, &self.snapshots[band], center, radius)
    }

    /// A world box, filed by its bounding sphere's nearest point. A liquid slot files by its
    /// centre (`0x681970`), a later band; the nearer band never hides more than the reference.
    pub fn hides_box(&self, min: Vec3, max: Vec3, flags: u32) -> bool {
        let Some(view) = &self.view else {
            return false;
        };
        let center = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        let Some(band) = band_of(view.band_depth(center) - radius) else {
            return false;
        };
        box_hidden(view, &self.snapshots[band], min, max, flags)
    }

    /// A far-band tile's box, tested after every band with flag 8 (`0x6832c8`).
    pub fn hides_far_box(&self, min: Vec3, max: Vec3) -> bool {
        let Some(view) = &self.view else {
            return false;
        };
        box_hidden(view, &self.snapshots[BANDS], min, max, FLAG_NEAR_OK)
    }

    /// A terrain cell none of whose chunks in view survived the horizon.
    pub fn hides_cell(&self, cell: &TerrainCell) -> bool {
        self.hidden_cells.contains(&(cell.tile, cell.cell))
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
struct Chunk<'a> {
    cell: ((i32, i32), u8),
    b: ChunkBox,
    /// The 145 vertices in WoW yards, stride-17 order.
    positions: &'a [[f32; 3]],
}

struct Walk {
    snapshots: Vec<Columns>,
    hidden_cells: HashSet<((i32, i32), u8)>,
    stats: WalkStats,
}

/// `CullSceneChunks` (`0x683bf0`) over every band, front to back: a chunk all holes or out of view
/// is dropped; one the horizon hides is not drawn; a visible one without holes whose view depth
/// lies in `[0, occluder_far)` adds its rows after its band's tests (`0x683d53`, `0x683d67`).
fn walk<'a>(
    view: &HorizonView,
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
            if (0.0..occluder_far).contains(&depth) {
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
    let hidden_cells: HashSet<_> = cells
        .into_iter()
        .filter(|(_, (shown, hidden))| *hidden && !*shown)
        .map(|(k, _)| k)
        .collect();
    stats.cells_hidden = hidden_cells.len();
    Walk {
        snapshots,
        hidden_cells,
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

/// Everything the walk reads: when it matches the last frame's, the walk would rebuild the same
/// [`Horizon`]. A tile's chunks are fixed per asset id, which [`ChunkBoxCache`] already relies on.
#[derive(PartialEq)]
struct WalkInputs {
    culling: HorizonCulling,
    unrestricted: bool,
    farclip: f32,
    eye: Option<(Mat4, Option<[f32; 4]>)>,
    /// Resident tiles, and whether each one's asset is loaded.
    resident: Vec<((i32, i32), AssetId<AdtTile>, bool)>,
}

/// Builds [`Horizon`] from the world camera and the resident tiles' chunks; writes only on change,
/// so a still scene leaves the model-visibility skip armed, and walks only when an input moved.
fn build_horizon(
    culling: Res<HorizonCulling>,
    windows: Res<ExteriorWindows>,
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
        unrestricted: matches!(*windows, ExteriorWindows::Unrestricted),
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
    *last = Some(inputs);
    let next = (|| {
        if !culling.enabled || !matches!(*windows, ExteriorWindows::Unrestricted) {
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
        let frustum =
            crate::exterior_cull::scene_window_frustum([-1.0, -1.0, 1.0, 1.0], &clip_from_world)?;
        let resident: Vec<((i32, i32), &AdtTile, AssetId<AdtTile>)> = streamer
            .spawned_tiles()
            .filter_map(|(key, h)| Some((key, tiles.get(h)?, h.id())))
            .collect();
        boxes.retain(|id, _| resident.iter().any(|(_, _, r)| r == id));
        for (_, tile, id) in &resident {
            boxes.entry(*id).or_insert_with(|| chunk_boxes(tile));
        }
        let chunks = resident.iter().flat_map(|(key, tile, id)| {
            let b = boxes.get(id).cloned();
            tile.chunks.iter().enumerate().filter_map(move |(i, c)| {
                let cell = ((i / 16) / 4 * 4 + (i % 16) / 4) as u8;
                Some(Chunk {
                    cell: (*key, cell),
                    b: (*b.as_ref()?.get(i)?)?,
                    positions: &c.positions,
                })
            })
        });
        let in_view = |min: Vec3, max: Vec3| {
            frustum.intersects_obb(
                &Aabb::from_min_max(min, max),
                &bevy::math::Affine3A::IDENTITY,
                true,
                true,
            )
        };
        let w = walk(
            &view,
            view_dist.farclip - OCCLUDER_FAR_MARGIN,
            culling.lod_depth(),
            chunks,
            in_view,
        );
        Some(Horizon {
            view: Some(view),
            enforce: culling.enforce,
            snapshots: w.snapshots,
            hidden_cells: w.hidden_cells,
            stats: w.stats,
        })
    })()
    .unwrap_or_default();
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
