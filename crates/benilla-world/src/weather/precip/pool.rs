//! The precip simulation: the drop and patter records, the packet delay line, the ground height
//! cache and the per-kind frame.

use bevy::platform::collections::HashMap;

use super::*;

/// One falling particle, in Bevy world space (y up).
pub(super) struct Drop {
    pub(super) pos: Vec3,
    pub(super) vel: Vec3,
    /// Ground height (Bevy y) of the drop's current grid cell, re-sampled on each new cell; the
    /// drop dies and may patter there. The reference instead bakes the death once at birth, where
    /// the velocity ray meets the grid (`0x675174` rain, `0x677c2f` snow, both `0x67cb60`).
    pub(super) land_y: f32,
    /// The grid cell `land_y` was sampled for.
    pub(super) cell: (i32, i32),
    /// Seconds since visible, the reference's `t − f1`; a late-replayed record starts at its lag.
    pub(super) age: f32,
}

/// A recorded drop: replays at `at` (`baseTime` plus its record offset) once its packet seals.
struct Pending {
    at: f32,
    drop: Drop,
}

/// One drop packet (`Packet<Drop>`): `baseTime = now + 6144/rate`, stamped at open only
/// (`0x67598c`), is when the batch replays. On the shader leg it draws only from a buffer baked
/// at close (`0x6752b0`, called only from `0x675a97`). A sparse first packet stamps ~100 s out
/// and still replays: the 60 s guard (`0x80ff6c`) bounds the replay span, not the delay.
struct Packet {
    /// The eye at open, set once (`0x678598`, from `0xc7cf20`); [`RETIRE_DIST`] measures from it.
    anchor: Vec3,
    /// `buildTime`, when the packet opened; record offsets are relative to it.
    opened: f32,
    /// `baseTime`, `opened + 6144/rate` at open: the instant replay starts.
    visible_at: f32,
    records: Vec<Pending>,
    /// Records ever pushed; `records` drains on activation, the cap counts this.
    count: u32,
}

/// One ground splash: a camera-facing triangle animated across the 4×4 splash atlas.
pub(super) struct Patter {
    pub(super) pos: Vec3,
    pub(super) age: f32,
    /// Atlas row (0..4 splash variants).
    pub(super) variant: u8,
}

/// One kind's falling and ground layers; its geometry rides the shared effect stream.
#[derive(Default)]
pub(super) struct Pool {
    /// Active drops: falling, drawn, landing.
    pub(super) drops: Vec<Drop>,
    /// The open packet, recording; it draws nothing.
    open: Option<Packet>,
    /// Sealed packets, replaying or waiting for their `baseTime`.
    sealed: Vec<Packet>,
    pub(super) patters: Vec<Patter>,
}

impl Pool {
    /// Records still in the pipeline, open and sealed: the state log's pipe count.
    pub(super) fn pending_len(&self) -> usize {
        self.open.as_ref().map_or(0, |p| p.records.len())
            + self.sealed.iter().map(|p| p.records.len()).sum::<usize>()
    }

    /// Seals the open packet when full, `P/6144` old or past `baseTime` (`0x675a6d`), opens one at
    /// the live rate, and returns its free space and the replay instant of a record made now.
    fn open_for(&mut self, now: f32, rate: f32, close_age: f32, cam: Vec3) -> (usize, f32) {
        if let Some(pk) = &self.open {
            if pk.count >= PACKET_CAP as u32 || now - pk.opened >= close_age || now >= pk.visible_at
            {
                self.seal();
            }
        }
        let pk = self.open.get_or_insert_with(|| Packet {
            anchor: cam,
            opened: now,
            visible_at: now + PACKET_CAP as f32 / rate.max(1.0),
            records: Vec::new(),
            count: 0,
        });
        (
            PACKET_CAP - pk.count as usize,
            pk.visible_at + (now - pk.opened),
        )
    }

    fn seal(&mut self) {
        if let Some(pk) = self.open.take() {
            if !pk.records.is_empty() {
                self.sealed.push(pk);
            }
        }
    }

    /// The type-change cut (`0x67be40`): the open packet is discarded unbaked (`0x6756b2`) and
    /// every sealed one not yet replaying is unlinked (`0x67575a`); replaying ones finish.
    pub(super) fn cut(&mut self, now: f32) {
        self.open = None;
        self.sealed.retain(|p| now > p.visible_at);
    }

    /// The anchor cull ([`RETIRE_DIST`]), run before emission like the active-list walk
    /// (`0x677ff0`): sealed packets by their anchor in 3-D, active drops by their own distance
    /// ([`RETIRE_DROP_SLACK`]). The open packet is not in that list. A discard: no patter is left.
    pub(super) fn retire_far(&mut self, cam: Vec3, kind: WeatherKind) {
        let anchor_r2 = RETIRE_DIST * RETIRE_DIST;
        self.sealed
            .retain(|pk| pk.anchor.distance_squared(cam) <= anchor_r2);
        let (half_xy, z_off) = spawn_box(kind);
        let corner = 2.0f32.mul_add(half_xy * half_xy, z_off * z_off).sqrt();
        let drop_r = RETIRE_DIST + corner + RETIRE_DROP_SLACK;
        self.drops
            .retain(|d| d.pos.distance_squared(cam) <= drop_r * drop_r);
    }
}

/// A lazy cache of the weather ground oracle (`0x67c760` → `0x6b7070`), the max of terrain and
/// the chunk's WMO and doodad hits (`CMapObj::IntersectSegment` `0x6a37b0`, `0x6b7237`), so drops
/// land on roofs: the first surface below the spawn plane, or 200 below it on a miss.
///
/// fork: perf — cached as each cell's COLUMN of surfaces, cast once from above any spawn plane
/// the camera is near, so every plane reads its answer from the column. The cache used to hold one
/// answer per cell for one plane and was dropped whole whenever the plane moved 8 yd: a flying
/// camera re-cast every cell of the spawn area several times a second, thousands of rays a frame
/// (`PRECIP_PROFILE`, Duskwood in rain). Its bound covers the band of cells a flight leaves drops
/// in (packets replay up to [`RETIRE_DIST`] behind the camera), and past it the cells far from the
/// camera go, once a frame ([`HeightCache::evict_far`]).
///
/// fork: a column answers for the colliders it was cast against. A zone streams them in over
/// seconds, and until a tile's or a building's own lands the rays meet its stand-in (the far
/// band's coarse ground, a bounding box: `terrain_stream::provisional`) or nothing at all; a
/// column kept past that had the rain landing on a plane yards off the floor, or falling through
/// it, for as long as the camera stayed. So the cache holds only the cells round the camera
/// ([`Self::evict_far`]), and when a collider comes or goes within [`NEAR_REACH`] of it
/// (`crate::collision::ColliderChanges`) every column is stale: re-cast when next asked for,
/// [`STALE_RECASTS`] a frame, and answering as it is until its turn.
#[derive(Resource, Default)]
pub(super) struct HeightCache {
    cells: HashMap<(i32, i32), Column>,
    /// How many times the colliders near the camera have changed ([`Self::begin_frame`]).
    epoch: u64,
    /// Frames since the cells far from the camera last went.
    since_trim: u32,
    /// Stale columns this frame may still re-cast.
    recasts_left: u32,
    /// fork: perf — rays cast, evictions and stale columns re-cast since the profile last read
    /// them (`WOW_PRECIP_PROFILE`).
    pub(super) rays: u32,
    pub(super) clears: u32,
    pub(super) recasts: u32,
}

/// One cell's surfaces, highest first, from `top` down to `bottom` (the known span).
struct Column {
    top: f32,
    bottom: f32,
    hits: [f32; COLUMN_HITS],
    len: u8,
    /// The cache's stamp when it was cast.
    epoch: u64,
}

/// The reference's cell stride (`0x80ff98` = 1.0416666).
pub(super) const CELL: f32 = 1.041_666_6;
/// A probe that hits nothing reports ground 200 below the query (`0x67c812`: `refZ − 200`).
const MISS_DEPTH: f32 = 200.0;
/// How far below the plane a probe reaches before it counts as a miss.
const PROBE_REACH: f32 = MISS_DEPTH + 50.0;
/// A column is cast from this far above the plane that asked for it, so a climbing camera reuses
/// it until it has risen this far.
const COLUMN_HEADROOM: f32 = 64.0;
/// Surfaces kept per column: a roof or a canopy over the ground, a bridge over both.
const COLUMN_HITS: usize = 4;
/// The cache's bound (about 8 MB), and the square half-width (yd) kept around the camera past it.
const CACHE_CELLS: usize = 1 << 17;
const KEEP_HALF: f32 = 160.0;
/// Stale columns re-cast per frame: the rain's ~16 000 cells follow a change in about two
/// seconds, at a few hundred rays a frame while they do.
const STALE_RECASTS: u32 = 128;
/// Frames between trims to [`KEEP_HALF`], and how far from the camera (yd, XZ) a collider change
/// still dates the columns: the kept square's half-width plus what a flight covers between trims.
const TRIM_FRAMES: u32 = 120;
pub(super) const NEAR_REACH: f32 = KEEP_HALF + 100.0;

impl Column {
    /// The oracle's answer for a probe from `plane`: the first surface at or below it within
    /// [`PROBE_REACH`], or the miss depth; `None` when this column does not cover that span.
    fn ground(&self, plane: f32) -> Option<f32> {
        if plane > self.top {
            return None;
        }
        let floor = plane - PROBE_REACH;
        match self.hits[..usize::from(self.len)]
            .iter()
            .find(|&&y| y <= plane)
        {
            Some(&y) if y >= floor => Some(y),
            // The first surface under the plane is past the reach: a miss.
            Some(_) => Some(plane - MISS_DEPTH),
            None if self.bottom <= floor => Some(plane - MISS_DEPTH),
            None => None,
        }
    }
}

impl HeightCache {
    /// Opens a frame's lookups; `changed` when a collider came or went within [`NEAR_REACH`].
    pub(super) fn begin_frame(&mut self, changed: bool) {
        if changed {
            self.epoch += 1;
        }
        self.recasts_left = STALE_RECASTS;
    }

    /// Does a column cast at stamp `cast_epoch` answer this lookup as it is? One from before the
    /// last nearby change is re-cast instead while the frame's budget lasts.
    fn serves(&mut self, cast_epoch: u64) -> bool {
        if cast_epoch == self.epoch || self.recasts_left == 0 {
            return true;
        }
        self.recasts_left -= 1;
        self.recasts += 1;
        false
    }

    pub(super) fn ground_y(
        &mut self,
        x: f32,
        z: f32,
        cast_from_y: f32,
        spatial: &SpatialQuery,
        filter: &SpatialQueryFilter,
    ) -> f32 {
        let key = Self::key(x, z);
        let cached = self
            .cells
            .get(&key)
            .and_then(|c| Some((c.ground(cast_from_y)?, c.epoch)));
        if let Some((y, cast_epoch)) = cached {
            if self.serves(cast_epoch) {
                return y;
            }
        }
        let cx = (key.0 as f32 + 0.5) * CELL;
        let cz = (key.1 as f32 + 0.5) * CELL;
        let column = self.cast_column(cx, cz, cast_from_y, spatial, filter);
        let y = column
            .ground(cast_from_y)
            .unwrap_or(cast_from_y - MISS_DEPTH);
        self.cells.insert(key, column);
        y
    }

    /// Keep only the cells within [`KEEP_HALF`] of the camera: past the bound, and every
    /// [`TRIM_FRAMES`], so no column outlives the reach its staleness is judged over.
    pub(super) fn evict_far(&mut self, cam: Vec3) {
        self.since_trim += 1;
        if self.cells.len() < CACHE_CELLS && self.since_trim < TRIM_FRAMES {
            return;
        }
        self.since_trim = 0;
        let (kx, kz) = Self::key(cam.x, cam.z);
        let keep = (KEEP_HALF / CELL).ceil() as i32;
        self.cells
            .retain(|&(cx, cz), _| (cx - kx).abs() <= keep && (cz - kz).abs() <= keep);
        self.clears += 1;
    }

    /// Walk down from [`COLUMN_HEADROOM`] above `plane` to [`PROBE_REACH`] below it, one ray per
    /// surface. Not solid: a ray starting inside a doodad's hull reports where it leaves it, so the
    /// walk goes on below.
    fn cast_column(
        &mut self,
        x: f32,
        z: f32,
        plane: f32,
        spatial: &SpatialQuery,
        filter: &SpatialQueryFilter,
    ) -> Column {
        let top = plane + COLUMN_HEADROOM;
        let mut column = Column {
            top,
            bottom: plane - PROBE_REACH,
            hits: [0.0; COLUMN_HITS],
            len: 0,
            epoch: self.epoch,
        };
        let mut from = top;
        while usize::from(column.len) < COLUMN_HITS {
            let reach = from - column.bottom;
            if reach <= 0.0 {
                break;
            }
            self.rays += 1;
            let Some(hit) =
                spatial.cast_ray(Vec3::new(x, from, z), Dir3::NEG_Y, reach, false, filter)
            else {
                break;
            };
            let y = from - hit.distance;
            column.hits[usize::from(column.len)] = y;
            column.len += 1;
            from = y - 0.05;
        }
        // Full: the span below the last surface is unknown.
        if usize::from(column.len) == COLUMN_HITS {
            column.bottom = column.bottom.max(from);
        }
        column
    }

    /// [`Self::ground_y`] without the ray: the cached answer, or `None` where `ground_y` would
    /// cast or re-cast (no column, one that does not cover `cast_from_y`, or one from before the
    /// last nearby collider change). Read-only, so the drift asks it from every worker at once.
    fn peek(&self, key: (i32, i32), cast_from_y: f32) -> Option<f32> {
        let column = self.cells.get(&key)?;
        if column.epoch != self.epoch {
            return None;
        }
        column.ground(cast_from_y)
    }

    /// The grid cell owning an XZ position.
    pub(super) fn key(x: f32, z: f32) -> (i32, i32) {
        ((x / CELL).floor() as i32, (z / CELL).floor() as i32)
    }
}

/// Drops per parallel task in the drift: a step is a few nanoseconds, so a chunk this size keeps
/// the dispatch cost small against the work (heavy rain holds ~17 k drops).
const DRIFT_CHUNK: usize = 2048;

/// Move every drop by `dt` and give each that changed cell its new cell's ground from the cache.
/// Returns the drops whose new cell [`HeightCache::peek`] could not answer, in pool order: each
/// moves alone and only reads the cache, so the chunks run in parallel on the compute pool, and
/// the misses take their rays after, serially, as rays fill the shared cache and spend the
/// frame's recast budget, in the order the serial walk spent it.
fn drift_drops(drops: &mut [Drop], heights: &HeightCache, cast_plane: f32, dt: f32) -> Vec<usize> {
    let step = |base: usize, chunk: &mut [Drop]| {
        let mut miss = Vec::new();
        for (i, d) in chunk.iter_mut().enumerate() {
            d.pos += d.vel * dt;
            d.age += dt;
            let cell = HeightCache::key(d.pos.x, d.pos.z);
            if cell != d.cell {
                d.cell = cell;
                match heights.peek(cell, cast_plane) {
                    Some(y) => d.land_y = y,
                    None => miss.push(base + i),
                }
            }
        }
        miss
    };
    // Below a chunk, or with no compute pool (a unit test), the walk stays on this thread.
    let pool = bevy::tasks::ComputeTaskPool::try_get().filter(|p| p.thread_num() > 1);
    match pool {
        Some(pool) if drops.len() > DRIFT_CHUNK => {
            let step = &step;
            pool.scope(|s| {
                for (c, chunk) in drops.chunks_mut(DRIFT_CHUNK).enumerate() {
                    s.spawn(async move { step(c * DRIFT_CHUNK, chunk) });
                }
            })
            .into_iter()
            .flatten()
            .collect()
        }
        _ => step(0, drops),
    }
}

/// The per-frame record count, `fistp(min(space, quota) − 0.5)` under x87 round-to-nearest-even
/// (`0x6754cc`). The parity is load-bearing: at `space = 1` it is `RNE(0.5) = 0` forever, so a
/// 6143-record packet never fills and seals only at the `P/6144` age.
fn frame_count(space: usize, quota: f32) -> usize {
    ((space as f32).min(quota) - 0.5).round_ties_even().max(0.0) as usize
}

/// One kind's spawn box: the scatter plane's half-extent and the slab's local lift above it.
pub(super) const fn spawn_box(kind: WeatherKind) -> (f32, f32) {
    match kind {
        WeatherKind::Rain => (RAIN_HALF_XY, RAIN_Z_OFF),
        _ => (SNOW_HALF_XY, SNOW_Z_OFF),
    }
}

/// `(spawn, velocity)` from five random draws (`0x677750` snow, `0x674c50` rain), placed by:
///
/// ```text
/// pos = R(α, ĥ×ŷ)·(O − T·V)  +  1.75·W  +  C
/// ```
///
/// `O` is the scatter on the eye plane where the particle arrives (its vertical draw is dead at
/// `0x6777ce`), backed up its velocity by `T = z_off/|V.y|`. `R` ([`WeatherWind::slab`]) leans
/// the offset but not the velocity (`0x677a41` copies position only), so drift is world-fixed.
pub(super) fn spawn_particle(
    kind: WeatherKind,
    w: f32,
    origin: Vec3,
    slab: Quat,
    r: [f32; 5],
) -> (Vec3, Vec3) {
    let (half_xy, z_off) = spawn_box(kind);
    let scatter = Vec3::new(
        (r[0] - 0.5) * 2.0 * half_xy,
        0.0,
        (r[1] - 0.5) * 2.0 * half_xy,
    );
    // `w` is the density. The heading centres on the world-fixed −1.57 (`0x80ffbc`), spread by
    // grade over its full width (`r − 0.5` halves it): rain ±7.5° at grade 1, calm snow 2π.
    let (vy, drift_mag, spread) = match kind {
        WeatherKind::Rain => (
            -(RAIN_VZ_BASE + RAIN_VZ_W * w + RAIN_VZ_RNG * w * r[2]),
            ((2.0 * r[3] - 1.0) + RAIN_DRIFT_BASE) * w + RAIN_DRIFT_EPS,
            RAIN_SPREAD_W * w + RAIN_SPREAD_BIAS,
        ),
        _ => (
            -(SNOW_VZ_BASE + SNOW_VZ_W * w + w * r[2]),
            ((r[3] - 0.5) + SNOW_DRIFT_OFF) * w + SNOW_DRIFT_EPS,
            std::f32::consts::TAU - SNOW_SPREAD_W * w,
        ),
    };
    let heading = DRIFT_AZ_CENTER + (r[4] - 0.5) * spread;
    // `vy` is strictly negative for both kinds, so `T` is positive and finite.
    let vel = wow_azimuth_to_bevy(heading) * drift_mag + Vec3::Y * vy;
    (origin + slab * (scatter - vel * (z_off / -vy)), vel)
}

/// One kind's frame: retire, record, activate sealed packets, integrate, land.
pub(super) fn run_kind(
    pool: &mut Pool,
    kind: WeatherKind,
    weather: &WeatherState,
    wind: &WeatherWind,
    heights: &mut HeightCache,
    spatial: &SpatialQuery,
    filter: &SpatialQueryFilter,
    rng: &mut u32,
    now: f32,
    dt: f32,
    cam_pos: Vec3,
) {
    let density = weather.density_for(kind);
    let p = match kind {
        WeatherKind::Rain => RAIN_P,
        _ => SNOW_P,
    };
    let z_off = spawn_box(kind).1;
    // Every probe this frame casts from one plane: the tilted slab spreads spawn heights over
    // ~8..46 yd, and the height cache clears whenever the cast height moves 8 yd.
    let cast_plane = cam_pos.y + z_off;

    // ===== retire =====
    pool.retire_far(cam_pos, kind);

    // ===== record =====
    // The emit gate (`0x6754a0`, `0x8015b8` = 1.0): a frame records only when its quota is over
    // 1, so below ~60 drops/s nothing spawns. `REF_FPS_GAIN` is this path's one deviation.
    let rate = weather.density_gain() * p * density * REF_FPS_GAIN;
    let quota = rate * dt.min(DT_CAP);
    if quota > 1.0 {
        let close_age = p / PACKET_CAP as f32;
        let (space, replay_at) = pool.open_for(now, rate, close_age, cam_pos);
        let n = frame_count(space, quota)
            .min(pool_bound(kind).saturating_sub(pool.drops.len() + pool.pending_len()));
        // The volume's origin: the eye led by the wind (`1.75·W + C`, added after the tilt).
        let anchor = cam_pos + wind.vel * WIND_LEAD;
        let origin = Vec3::new(anchor.x, cam_pos.y, anchor.z);
        let w = density;
        for _ in 0..n {
            let r = [
                rand01(rng),
                rand01(rng),
                rand01(rng),
                rand01(rng),
                rand01(rng),
            ];
            let (spawn, vel) = spawn_particle(kind, w, origin, wind.slab, r);
            let land_y = heights.ground_y(spawn.x, spawn.z, cast_plane, spatial, filter);
            // Ground at or above the spawn kills it at birth (`0x675051`).
            if land_y >= spawn.y {
                continue;
            }
            let drop = Drop {
                pos: spawn,
                vel,
                land_y,
                cell: HeightCache::key(spawn.x, spawn.z),
                age: 0.0,
            };
            if let Some(pk) = &mut pool.open {
                pk.count += 1;
                pk.records.push(Pending {
                    at: replay_at,
                    drop,
                });
            }
        }
    }

    // ===== activate: sealed packets replay from `baseTime` =====
    // A record whose instant passed while its packet was open appears mid-fall, on its new
    // cell's ground; one that already landed is skipped with no splash.
    let drops = &mut pool.drops;
    for pk in &mut pool.sealed {
        let mut i = 0;
        while i < pk.records.len() {
            if pk.records[i].at <= now {
                let rec = pk.records.swap_remove(i);
                let lag = now - rec.at;
                let mut drop = rec.drop;
                drop.pos += drop.vel * lag;
                drop.age = lag;
                let cell = HeightCache::key(drop.pos.x, drop.pos.z);
                if cell != drop.cell {
                    drop.cell = cell;
                    drop.land_y =
                        heights.ground_y(drop.pos.x, drop.pos.z, cast_plane, spatial, filter);
                }
                if drop.pos.y > drop.land_y {
                    drops.push(drop);
                }
            } else {
                i += 1;
            }
        }
    }
    pool.sealed.retain(|pk| !pk.records.is_empty());

    // Rain leaves one patter per landing drop (`0x6754a0`). The reference skips it when the
    // ridden transport's `|v|² > 2` (`0x6755c2`, `mgr+0x5c`), never for running; here the gate
    // always passes, with no transport speed wired. Snow settles and fades over 0.25 s.
    let ground_life = match kind {
        WeatherKind::Rain => PATTER_LIFE,
        _ => SNOW_SETTLE_LIFE,
    };
    let ground_gate = true;

    // ===== integrate + land =====
    // Landing tests the current cell's ground, a max from above: a drop drifting under cover
    // lands on the roof and splashes there, never on the floor below.
    for i in drift_drops(&mut pool.drops, heights, cast_plane, dt) {
        let d = &mut pool.drops[i];
        d.land_y = heights.ground_y(d.pos.x, d.pos.z, cast_plane, spatial, filter);
    }
    let mut landed: Vec<Patter> = Vec::new();
    {
        pool.drops.retain_mut(|d| {
            if d.pos.y > d.land_y {
                return true;
            }
            if ground_gate {
                landed.push(Patter {
                    pos: Vec3::new(d.pos.x, d.land_y + 0.02, d.pos.z),
                    age: 0.0,
                    variant: 0, // assigned below (needs the rng)
                });
            }
            false
        });
    }
    for l in &mut landed {
        l.variant = (rand01(rng) * 4.0) as u8 & 3;
    }
    landed.truncate(GROUND_CAP.saturating_sub(pool.patters.len()));
    pool.patters.append(&mut landed);
    pool.patters.retain_mut(|l| {
        l.age += dt;
        l.age < ground_life
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A column answers any plane under its top as one ray from that plane would: the first
    /// surface at or below it within reach, the miss depth past it, and nothing it did not cover.
    #[test]
    fn a_column_answers_every_plane_under_its_top() {
        // A roof at 30 over ground at 10, cast for a plane at 40.
        let mut hits = [0.0; COLUMN_HITS];
        hits[..2].copy_from_slice(&[30.0, 10.0]);
        let column = Column {
            top: 40.0 + COLUMN_HEADROOM,
            bottom: 40.0 - PROBE_REACH,
            hits,
            len: 2,
            epoch: 0,
        };
        assert_eq!(column.ground(40.0), Some(30.0));
        assert_eq!(column.ground(100.0), Some(30.0));
        // Under the roof, the ground.
        assert_eq!(column.ground(20.0), Some(10.0));
        // Above the column's top: re-cast.
        assert_eq!(column.ground(40.0 + COLUMN_HEADROOM + 1.0), None);
        // A plane whose first surface lies past the reach: a miss.
        let low = Column {
            top: 400.0,
            bottom: 400.0 - COLUMN_HEADROOM - PROBE_REACH,
            hits,
            len: 2,
            epoch: 0,
        };
        assert_eq!(low.ground(300.0), Some(300.0 - MISS_DEPTH));
        // Below every surface and past what the column covers: re-cast.
        assert_eq!(column.ground(5.0), None);
    }

    /// A column cast before the last nearby collider change is re-cast when asked for, a bounded
    /// number a frame; past the budget it answers as it is, and the next frame takes up where
    /// this one stopped. A frame with no nearby change dates nothing.
    #[test]
    fn a_stale_column_is_recast_within_the_frames_budget() {
        let mut heights = HeightCache::default();
        heights.begin_frame(false);
        assert!(heights.serves(0), "nothing changed: a column answers");
        heights.begin_frame(true);
        for _ in 0..STALE_RECASTS {
            assert!(!heights.serves(0), "cast before the change: re-cast");
        }
        assert!(
            heights.serves(0),
            "the budget is spent: it answers as it is"
        );
        assert!(heights.serves(1), "cast since the change");
        assert_eq!(heights.recasts, STALE_RECASTS);
        heights.begin_frame(false);
        assert!(!heights.serves(0), "a new frame, a new budget");
        assert!(heights.serves(1));
    }

    /// The drift's read-only lookup answers only what `ground_y` would answer without a ray: a
    /// current column over the plane. A missing cell, an uncovered plane or a column from before
    /// the last nearby collider change is a miss, left to the serial pass and its recast budget.
    #[test]
    fn the_drift_peeks_only_a_current_column() {
        let mut heights = HeightCache::default();
        let mut hits = [0.0; COLUMN_HITS];
        hits[0] = 5.0;
        heights.cells.insert(
            (0, 0),
            Column {
                top: 40.0,
                bottom: -300.0,
                hits,
                len: 1,
                epoch: 0,
            },
        );
        assert_eq!(
            heights.peek((0, 0), 20.0),
            Some(5.0),
            "a current column answers"
        );
        assert_eq!(
            heights.peek((0, 0), 60.0),
            None,
            "a plane over its top casts"
        );
        assert_eq!(heights.peek((1, 0), 20.0), None, "no column casts");
        heights.begin_frame(true);
        assert_eq!(heights.peek((0, 0), 20.0), None, "a stale column re-casts");
    }

    /// Drifting in chunks moves every drop and lists the misses in pool order, as the serial walk
    /// met them.
    #[test]
    fn the_drift_moves_every_drop_and_lists_misses_in_order() {
        let heights = HeightCache::default();
        let mut drops: Vec<Drop> = (0..DRIFT_CHUNK * 2 + 7)
            .map(|i| Drop {
                pos: Vec3::new(i as f32 * 0.01, 30.0, 0.0),
                vel: Vec3::new(2.0, -10.0, 0.0),
                land_y: 0.0,
                cell: (i32::MIN, 0),
                age: 0.0,
            })
            .collect();
        let misses = drift_drops(&mut drops, &heights, 40.0, 0.1);
        assert_eq!(misses, (0..drops.len()).collect::<Vec<_>>());
        assert!(drops
            .iter()
            .all(|d| (d.pos.y - 29.0).abs() < 1e-5 && d.age > 0.0));
    }

    /// The cells far from the camera go on a clock as well as at the bound, so every column the
    /// cache holds lies inside the reach a collider change is judged over.
    #[test]
    fn the_cache_keeps_only_the_cells_round_the_camera() {
        let mut heights = HeightCache::default();
        let column = |epoch| Column {
            top: 0.0,
            bottom: 0.0,
            hits: [0.0; COLUMN_HITS],
            len: 0,
            epoch,
        };
        heights.cells.insert(HeightCache::key(0.0, 0.0), column(0));
        heights
            .cells
            .insert(HeightCache::key(KEEP_HALF + 20.0, 0.0), column(0));
        for _ in 0..TRIM_FRAMES - 1 {
            heights.evict_far(Vec3::ZERO);
        }
        assert_eq!(heights.cells.len(), 2, "not yet due");
        heights.evict_far(Vec3::ZERO);
        assert_eq!(heights.cells.len(), 1, "the far cell went with the trim");
        assert!(heights.cells.contains_key(&HeightCache::key(0.0, 0.0)));
        const { assert!(NEAR_REACH > KEEP_HALF) };
    }

    /// A converged [`WeatherWind`] for a run along Bevy +X at `speed`, through the real window.
    fn wind_at(speed: f32) -> WeatherWind {
        let mut wind = WeatherWind::default();
        let dt = 1.0 / 60.0;
        for i in 0..30 {
            wind.update(Vec3::X * (speed * dt * i as f32), Vec3::X, speed, dt);
        }
        wind
    }

    /// A wire grade through the knee (`0x67bcc8`): the density the spawn kernel sees.
    fn grade(wire: f32) -> f32 {
        ((wire - 0.25) * (4.0 / 3.0)).max(0.0)
    }

    /// At a 7 yd/s run the leading corner is born ~8 yd up and ~53 ahead, the trailing ~46 up.
    #[test]
    fn the_spawn_slab_leans_into_the_run() {
        let wind = wind_at(7.0);
        let (half_xy, z_off) = spawn_box(WeatherKind::Snow);
        let lead = wind.slab * Vec3::new(half_xy, z_off, 0.0);
        let trail = wind.slab * Vec3::new(-half_xy, z_off, 0.0);
        assert!(
            (lead.x - 53.5).abs() < 0.5 && (lead.y - 7.9).abs() < 0.5,
            "leading corner {lead:?} — the worked value is (53.5, 7.9)"
        );
        assert!(
            (trail.x + 27.9).abs() < 0.5 && (trail.y - 46.3).abs() < 0.5,
            "trailing corner {trail:?} — the worked value is (−27.9, 46.3)"
        );
        assert_eq!(wind_at(0.0).slab, Quat::IDENTITY);
    }

    /// At 18 yd/s the slab has saturated at 65° while the streak apex is at 27°.
    #[test]
    fn the_slab_and_streak_tilts_are_distinct_ramps() {
        let wind = wind_at(18.0);
        let deg = |q: Quat| q.angle_between(Quat::IDENTITY).to_degrees();
        assert!(
            (deg(wind.slab) - 65.0).abs() < 0.5,
            "slab {}",
            deg(wind.slab)
        );
        assert!(
            (deg(wind.tilt) - 27.0).abs() < 0.5,
            "streak {}",
            deg(wind.tilt)
        );
    }

    /// Each particle's eye-height crossing against where the runner is then: untilted, a flake
    /// at grade 0.6 falls ~7.8 s, long enough for a 7 yd/s runner to outrun nearly all of them.
    #[test]
    fn a_running_player_still_meets_snow_head_on() {
        let ahead_share = |speed: f32, slab: Quat| {
            let m = grade(0.6);
            let origin = Vec3::X * (speed * WIND_LEAD);
            let mut rng = 0x9E37_79B9_u32;
            let (mut ahead, mut total) = (0u32, 0u32);
            for _ in 0..40_000 {
                let r = [
                    rand01(&mut rng),
                    rand01(&mut rng),
                    rand01(&mut rng),
                    rand01(&mut rng),
                    rand01(&mut rng),
                ];
                let (spawn, vel) = spawn_particle(WeatherKind::Snow, m, origin, slab, r);
                // Born at or below the eye: it never crosses the plane, so it is not an arrival.
                if spawn.y <= 0.0 {
                    continue;
                }
                // The camera runs +X from the origin; eye height is y = 0 and the ground is flat.
                let tau = spawn.y / -vel.y;
                total += 1;
                if (spawn.x + vel.x * tau) - speed * tau > 0.0 {
                    ahead += 1;
                }
            }
            f64::from(ahead) / f64::from(total)
        };

        let standing = ahead_share(0.0, Quat::IDENTITY);
        assert!(
            (standing - 0.5).abs() < 0.05,
            "a standing player's arrivals should split ~50/50, got {standing:.3}"
        );

        let running = wind_at(7.0);
        let tilted = ahead_share(7.0, running.slab);
        let flat = ahead_share(7.0, Quat::IDENTITY);
        eprintln!(
            "snow arrivals ahead of the player: standing {standing:.3}, \
             running tilted {tilted:.3}, running flat (untilted slab) {flat:.3}"
        );
        assert!(
            flat < 0.10,
            "the untilted slab is supposed to reproduce the symptom (nearly nothing arrives \
             ahead of a runner); got {flat:.3} — if this rose, the symptom's cause moved"
        );
        assert!(
            tilted > 0.25,
            "with the slab tilt a runner should still meet a quarter or more of the arrivals \
             head-on (the worked value is ~0.34); got {tilted:.3}"
        );
        assert!(
            tilted > flat * 3.0,
            "the tilt must dominate, not nudge: tilted {tilted:.3} vs flat {flat:.3}"
        );
    }

    /// The tilt moves rain's arrival band from −61..+69 to −45..+86 yd, emptying neither side.
    #[test]
    fn the_tilt_leaves_rain_balanced() {
        let m = grade(0.6);
        let speed = 7.0;
        let slab = wind_at(speed).slab;
        let origin = Vec3::X * (speed * WIND_LEAD);
        let mut rng = 0x1234_5678_u32;
        let (mut ahead, mut total) = (0u32, 0u32);
        for _ in 0..40_000 {
            let r = [
                rand01(&mut rng),
                rand01(&mut rng),
                rand01(&mut rng),
                rand01(&mut rng),
                rand01(&mut rng),
            ];
            let (spawn, vel) = spawn_particle(WeatherKind::Rain, m, origin, slab, r);
            if spawn.y <= 0.0 {
                continue;
            }
            let tau = spawn.y / -vel.y;
            total += 1;
            if (spawn.x + vel.x * tau) - speed * tau > 0.0 {
                ahead += 1;
            }
        }
        let share = f64::from(ahead) / f64::from(total);
        assert!(
            (0.35..0.75).contains(&share),
            "rain should stay balanced fore/aft under the tilt, got {share:.3}"
        );
    }

    /// At 60 fps with the gain, benilla emits what a reference capped at 30 fps does.
    #[test]
    fn the_frame_cap_gain_matches_the_reference_installs_throughput() {
        // Nominal snow at wire grade 1.0 and weatherDensity 3 (K = 1): 14000/s.
        let raw = SNOW_P * grade(1.0);
        // The reference's budget `min(dt, 1/60)·rate`, its remainder discarded.
        let per_second = |rate: f32, fps: f32| {
            frame_count(PACKET_CAP, rate * (1.0 / fps).min(DT_CAP)) as f32 * fps
        };

        let reference = per_second(raw, REF_MAXFPS);
        assert!(
            (reference / (raw * 0.5) - 1.0).abs() < 0.01,
            "a 30-capped reference emits half its nominal rate, got {reference} of {raw}"
        );
        let ours = per_second(raw * REF_FPS_GAIN, 60.0);
        assert!(
            (ours / reference - 1.0).abs() < 0.01,
            "ours {ours} must match the reference's {reference}"
        );
        // Without the gain it runs at twice the reference.
        assert!((per_second(raw, 60.0) / reference - 2.0).abs() < 0.01);
        // Above 60 fps the reference's own `min(dt, 1/60)` already flattens throughput, so the
        // gain does not compound with a fast machine.
        assert!((per_second(raw * REF_FPS_GAIN, 144.0) / ours - 1.0).abs() < 0.02);
    }

    #[test]
    fn frame_count_rounds_nearest_even() {
        assert_eq!(frame_count(6144, 583.3), 583); // RNE(582.8)
        assert_eq!(frame_count(300, 583.3), 300); // space-clamped: RNE(299.5) → 300 (even)
        assert_eq!(frame_count(1, 583.3), 0); // the parity stick: RNE(0.5) → 0
        assert_eq!(frame_count(2, 583.3), 2); // RNE(1.5) → 2 (even)
        assert_eq!(frame_count(6144, 1.2), 1); // RNE(0.7) → 1
    }

    /// `baseTime` is stamped at open alone; past it, the packet reopens at the live rate.
    #[test]
    fn packet_seals_before_replaying() {
        let mut pool = Pool::default();
        let close_age = RAIN_P / PACKET_CAP as f32; // 35000/6144 ≈ 5.7 s
        let (_, at0) = pool.open_for(0.0, RAIN_P, close_age, Vec3::ZERO);
        assert!((at0 - PACKET_CAP as f32 / RAIN_P).abs() < 1e-3);
        // 0.1 s later, same packet: the stamp holds and the record offset advances.
        let (_, at1) = pool.open_for(0.1, RAIN_P * 2.0, close_age, Vec3::ZERO);
        assert!((at1 - (at0 + 0.1)).abs() < 1e-3);
        let (_, at2) = pool.open_for(1.0, RAIN_P * 2.0, close_age, Vec3::ZERO);
        assert!((at2 - (1.0 + PACKET_CAP as f32 / (RAIN_P * 2.0))).abs() < 1e-3);
    }

    #[test]
    fn cut_discards_the_unreplayed_pipeline() {
        let d = || Drop {
            pos: Vec3::ZERO,
            vel: Vec3::NEG_Y,
            land_y: -100.0,
            cell: (0, 0),
            age: 0.0,
        };
        let mut pool = Pool {
            open: Some(Packet {
                anchor: Vec3::ZERO,
                opened: 9.0,
                visible_at: 12.0,
                records: vec![Pending {
                    at: 12.0,
                    drop: d(),
                }],
                count: 1,
            }),
            ..Default::default()
        };
        pool.sealed.push(Packet {
            anchor: Vec3::ZERO,
            opened: 0.0,
            visible_at: 1.0, // replaying since t=1
            records: vec![Pending { at: 4.0, drop: d() }],
            count: 1,
        });
        pool.sealed.push(Packet {
            anchor: Vec3::ZERO,
            opened: 8.0,
            visible_at: 60.0, // never started replaying
            records: vec![Pending {
                at: 60.5,
                drop: d(),
            }],
            count: 1,
        });
        pool.cut(10.0);
        assert!(pool.open.is_none());
        assert_eq!(pool.sealed.len(), 1, "only the replaying packet survives");
        assert!((pool.sealed[0].visible_at - 1.0).abs() < 1e-6);
    }

    /// A 10 s upswing's first visible rain lands in the reference's 4.5–15 s onset band.
    #[test]
    fn upswing_first_visible_rain_is_gated() {
        let mut pool = Pool::default();
        let close_age = RAIN_P / PACKET_CAP as f32;
        let dt = 1.0 / 60.0;
        let mut first_visible = f32::MAX;
        let mut t = 0.0f32;
        while t < 30.0 {
            let a = (t / 10.0).min(1.0);
            let intensity = ((a - 0.25) * (4.0 / 3.0)).max(0.0);
            let rate = RAIN_P * intensity; // K = 1 (weatherDensity 3)
            let quota = rate * dt;
            if quota > 1.0 {
                let (space, replay_at) = pool.open_for(t, rate, close_age, Vec3::ZERO);
                let n = frame_count(space, quota);
                if let Some(pk) = &mut pool.open {
                    pk.count += n as u32;
                    for _ in 0..n {
                        pk.records.push(Pending {
                            at: replay_at,
                            drop: Drop {
                                pos: Vec3::ZERO,
                                vel: Vec3::NEG_Y,
                                land_y: -100.0,
                                cell: (0, 0),
                                age: 0.0,
                            },
                        });
                    }
                }
            }
            // Replay needs the packet sealed and the record's instant past.
            for pk in &pool.sealed {
                for r in &pk.records {
                    if r.at <= t {
                        first_visible = first_visible.min(r.at.max(pk.visible_at));
                    }
                }
            }
            t += dt;
        }
        assert!(
            (4.5..15.0).contains(&first_visible),
            "first visible rain at {first_visible:.2} s — the reference band is ~5/8.5–14 s"
        );
    }

    /// A teleport discards the field and its queue; a 150 yd walk touches nothing.
    #[test]
    fn a_teleport_retires_the_field_and_its_pipeline() {
        let field = || {
            let d = |x: f32| Drop {
                pos: Vec3::new(x, 0.0, 0.0),
                vel: Vec3::NEG_Y,
                land_y: -100.0,
                cell: (0, 0),
                age: 0.0,
            };
            Pool {
                drops: vec![d(0.0), d(40.0)],
                sealed: vec![Packet {
                    anchor: Vec3::ZERO,
                    opened: 0.0,
                    visible_at: 1.0,
                    records: vec![Pending {
                        at: 4.0,
                        drop: d(0.0),
                    }],
                    count: 1,
                }],
                ..Default::default()
            }
        };
        let mut walked = field();
        walked.retire_far(Vec3::new(150.0, 0.0, 0.0), WeatherKind::Snow);
        assert_eq!(walked.drops.len(), 2, "150 yd is inside the guard");
        assert_eq!(walked.sealed.len(), 1);

        let mut jumped = field();
        jumped.retire_far(Vec3::new(500.0, 0.0, 0.0), WeatherKind::Snow);
        assert!(
            jumped.drops.is_empty(),
            "the stranded field is discarded, not drained"
        );
        assert!(
            jumped.sealed.is_empty(),
            "and so is everything queued behind it"
        );
    }

    /// The drop cull is strictly weaker than the packet cull, which only a teleport can trip.
    #[test]
    fn the_drop_cull_never_beats_the_reference_to_a_flake() {
        /// Ground speed at +100% (the epic mount) in yd/s, the fastest continuous eye motion.
        const EPIC_MOUNT: f32 = 14.0;
        for kind in [WeatherKind::Rain, WeatherKind::Snow] {
            let (half_xy, z_off) = spawn_box(kind);
            let corner = 2.0f32.mul_add(half_xy * half_xy, z_off * z_off).sqrt();
            // The farthest a drawn flake can be: anchor within 200, flake within the box corner of
            // the origin, origin leading the eye by `1.75·W`.
            let widest_drawn = RETIRE_DIST + corner + WIND_LEAD * EPIC_MOUNT;
            let ours = RETIRE_DIST + corner + RETIRE_DROP_SLACK;
            assert!(
                ours >= widest_drawn,
                "{kind:?}: culling at {ours:.1} yd would beat the reference's {widest_drawn:.1} yd"
            );
        }
        // A snow packet's life at mount speed covers under half of `RETIRE_DIST`.
        let packet_life = SNOW_P / PACKET_CAP as f32 + SNOW_Z_OFF / (SNOW_VZ_BASE + SNOW_VZ_W);
        assert!(
            packet_life * EPIC_MOUNT < RETIRE_DIST,
            "a packet outlives {:.0} yd of running — the anchor cull is no longer teleport-only",
            packet_life * EPIC_MOUNT
        );
    }
}
