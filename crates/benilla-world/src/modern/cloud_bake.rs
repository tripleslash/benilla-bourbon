//! **The modern cloud layer, baked off the render thread** (fork-owned; `FORK.md`
//! P4).
//!
//! After Schneider's cloud model (Guerrilla), on the 2D dome: the reference's 128² coverage tile is
//! the low-resolution field, and a worker bakes it up to a [`RES`]² image. An inverted-Worley billow
//! noise thresholded against the coverage gives the shape — a thin field leaves a few small round
//! puffs, a thick one merges them — and a light march toward the key over that density map gives
//! the self-shadowing (Beer's law: sun-facing edges glow, cores and bellies darken), with a powder
//! term on the thin edges. The lit side takes the key's colour, the shade the desaturated authored
//! cloud colour. In a modern frame the image replaces the tile on the cloud material's own binding;
//! `cloud.wgsl` adds only the view-dependent silver lining and the horizon fade. Classic frames never
//! see it.
//!
//! The bake runs on the async compute pool's workers. With none running (a page without threads),
//! it advances [`FALLBACK_ROWS`] rows a frame on the main thread instead, so no frame pays for a
//! whole bake.

use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureFormat};
use bevy::tasks::{block_on, futures_lite::future, Task};

use super::look::ZoneLook;
use super::{frame_is_modern, smoothstep, srgb_to_linear, ActiveLighting};
use crate::clouds::{CloudCoverage, CloudLayer, CloudMaterial};
use crate::lighting::WowLighting;

/// The baked layer's side, in texels.
pub(super) const RES: usize = 512;
/// Seconds between bakes: the coverage field evolves over seconds, not frames, and each bake's
/// upload rebuilds the layer's GPU texture.
const PERIOD: f32 = 1.0;
/// Light-march taps toward the key.
const TAPS: usize = 6;
/// Rows a frame on the main-thread fallback (density and light passes each take [`RES`] rows).
const FALLBACK_ROWS: usize = 24;
/// The share of the sky's hue the lit side takes.
const SKY_HUE: f32 = 0.2;
/// The billow noise's drift, in noise cells a second.
const DRIFT: Vec2 = Vec2::new(0.011, 0.007);

/// The modern cloud layer's look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CloudStyle {
    /// Peak opacity.
    pub opacity: f32,
    /// The puffs' edge softness: the width of the threshold the billow noise is cut at.
    pub soft: f32,
    /// The billow noise's cells across the dome's UV (higher = smaller puffs).
    pub detail: f32,
    /// The dome height (camera-relative direction `y`) where the alpha's horizon fade ends.
    pub horizon: f32,
    /// How far the authored cloud colour desaturates under a high key (1 = grey). A low key keeps
    /// it: a sunset's clouds take the authored colour.
    pub desat: f32,
    /// How far the key light's colour is whitened under a high key (1 = white).
    pub whiten: f32,
    /// A gain on the lit side's brightness. The key colour is taken at full brightness whatever
    /// its level, so this sets how far a cloud stands out of the sky behind it, the dim moon's as
    /// the noon sun's.
    pub bright: f32,
    /// The coverage's optical depth at full cover (Beer's law: opacity `1 − e^−τ`).
    pub density: f32,
    /// The light march's length toward the key, in dome UV.
    pub step: f32,
    /// The light path's extinction: how dark a cloud's core and belly go.
    pub sigma: f32,
    /// The shade's ambient level: the skylight the shadowed side is filled with.
    pub ambient: f32,
    /// The powder term's share: how far thin edges darken.
    pub powder: f32,
    /// The dome height below which the cloud takes the horizon haze's colour (aerial perspective).
    pub haze: f32,
    /// fork: the density field's floor (0 = none, 1 = a whole deck): the gaps between the puffs
    /// close over ([`storm_deck`]).
    pub deck: f32,
}

pub(super) const CLOUD_STYLE: CloudStyle = CloudStyle {
    opacity: 0.4,
    soft: 0.2,
    detail: 24.0,
    horizon: 0.12,
    desat: 0.9,
    whiten: 0.3,
    bright: 0.5,
    density: 3.0,
    step: 0.08,
    sigma: 0.6,
    ambient: 0.7,
    powder: 0.4,
    haze: 0.3,
    deck: 0.0,
};

/// fork: **the weather thickens the clouds, and a thunderstorm closes the sky.** With the
/// weather's intensity (`weather::intensity_now`, its effect density) the layer's peak opacity
/// rises to [`STORM_OPACITY`] and its optical depth to [`STORM_DENSITY`]; under heavy rain
/// (`weather::heavy_now`) the density field is floored at the heaviness, so the gaps between the
/// puffs fill in. At [`CLOUD_STYLE`]'s 0.4 the coverage tile's thickest storm still showed the sky,
/// stars and nebulae through every cloud.
const STORM_OPACITY: f32 = 1.0;
const STORM_DENSITY: f32 = 5.0;

/// The style at weather intensity `intensity` and rain heaviness `heavy`.
fn storm_deck(s: CloudStyle, intensity: f32, heavy: f32) -> CloudStyle {
    let i = intensity.clamp(0.0, 1.0);
    CloudStyle {
        opacity: s.opacity + (STORM_OPACITY - s.opacity) * i,
        density: s.density + (STORM_DENSITY - s.density) * i,
        deck: s.deck.max(heavy.clamp(0.0, 1.0)),
        ..s
    }
}

/// Dev tuning (`$WOW_MODERN_CLOUDS`, browser `?modern_clouds=`): comma-separated overrides in field
/// order — `opacity,soft,detail,horizon,desat,whiten,bright,density,step,sigma,ambient,powder,haze,deck`.
/// An empty or unparsable entry keeps the default.
pub(super) fn cloud_style() -> CloudStyle {
    let Some(v) = crate::launch_env::live_var("WOW_MODERN_CLOUDS") else {
        return CLOUD_STYLE;
    };
    let o: Vec<Option<f32>> = v.split(',').map(|p| p.trim().parse().ok()).collect();
    let get = |i: usize, d: f32| o.get(i).copied().flatten().unwrap_or(d);
    let d = CLOUD_STYLE;
    CloudStyle {
        opacity: get(0, d.opacity),
        soft: get(1, d.soft).max(1e-3),
        detail: get(2, d.detail),
        horizon: get(3, d.horizon).max(1e-3),
        desat: get(4, d.desat),
        whiten: get(5, d.whiten),
        bright: get(6, d.bright),
        density: get(7, d.density),
        step: get(8, d.step),
        sigma: get(9, d.sigma),
        ambient: get(10, d.ambient),
        powder: get(11, d.powder),
        haze: get(12, d.haze).max(1e-3),
        deck: get(13, d.deck).clamp(0.0, 1.0),
    }
}

/// Dev lever (`$WOW_MODERN_CLOUD_PERIOD`, browser `?modern_cloud_period=`): seconds between bakes;
/// 0 bakes once and keeps that layer, the A/B for what rebaking costs.
fn bake_period() -> f32 {
    crate::launch_env::live_var("WOW_MODERN_CLOUD_PERIOD")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(PERIOD)
}

/// What a bake and the cloud shader share for this frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct CloudParams {
    pub style: CloudStyle,
    /// Toward the key light (Bevy frame).
    pub key: Vec3,
    /// The key's colour (linear), whitened under a high key and scaled by `bright`.
    pub key_col: Vec3,
    /// The authored colour's desaturation this frame.
    pub desat: f32,
    /// The key elevation's brightness: a low key lights the deck from below rather than leaving it
    /// a grey slab.
    pub elev: f32,
    /// The skylight (linear): the dome's upper stops, what fills a cloud's shade.
    pub sky_col: Vec3,
    /// The horizon haze's colour (linear): the dome's horizon stop, what a low cloud dissolves into.
    pub haze_col: Vec3,
}

pub(super) fn cloud_params(light: &WowLighting, look: &ZoneLook) -> CloudParams {
    let style = storm_deck(
        cloud_style(),
        super::weather::intensity_now(),
        super::weather::heavy_now(),
    );
    // Clouds follow the key light (sun by day, moon by night).
    let key = (-light.sun_dir).normalize_or(Vec3::Y);
    // Neutral under a high key, coloured as it nears the horizon: a 15:00 sun-glow row's pink is
    // desaturated away, a sunset's red is kept.
    let high = smoothstep(0.05, 0.4, key.y);
    let sky_col =
        (sky_stop(light, look, 0) + sky_stop(light, look, 1) + sky_stop(light, look, 2)) / 3.0;
    // The lit side takes a fifth of the sky's hue too: the air between the cloud and the eye
    // tints it, so a green-skied zone's clouds are not a neutral white pasted over it.
    let sky_hue = sky_col / sky_col.max_element().max(1e-3);
    let key_col =
        Vec3::from(light.diffuse.map(srgb_to_linear)) * Vec3::from(look.key_hue(look.key_tint));
    let key_col = (key_col / key_col.max_element().max(1e-3)).lerp(Vec3::ONE, style.whiten * high)
        * Vec3::ONE.lerp(sky_hue, SKY_HUE)
        * style.bright;
    CloudParams {
        style,
        key,
        key_col,
        desat: style.desat * high,
        elev: 0.8 + 0.3 * smoothstep(-0.1, 0.45, key.y),
        sky_col,
        haze_col: sky_stop(light, look, 4),
    }
}

/// A sky-dome stop (zenith 0 to horizon 4) as the modern dome draws it: the authored gamma row
/// under the zone look's tint, decoded.
fn sky_stop(light: &WowLighting, look: &ZoneLook, i: usize) -> Vec3 {
    Vec3::new(
        srgb_to_linear(light.sky[i][0] * look.sky_tint[0]),
        srgb_to_linear(light.sky[i][1] * look.sky_tint[1]),
        srgb_to_linear(light.sky[i][2] * look.sky_tint[2]),
    )
}

/// One bake: the inputs it snapshotted and its two passes' buffers.
pub(super) struct CloudBake {
    /// The coverage tile, linear rgb and coverage in [0, 1].
    tile: Vec<Vec4>,
    cols: usize,
    params: CloudParams,
    drift: Vec2,
    density: Vec<f32>,
    out: Vec<[u8; 4]>,
    /// Rows done across both passes (`0..2·RES`).
    row: usize,
}

impl CloudBake {
    pub(super) fn new(tile: &[[u8; 4]], params: CloudParams, time: f32) -> Self {
        let cols = (tile.len() as f32).sqrt() as usize;
        let lut: [f32; 256] = std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0));
        let tile = tile
            .iter()
            .map(|t| {
                Vec4::new(
                    lut[t[0] as usize],
                    lut[t[1] as usize],
                    lut[t[2] as usize],
                    f32::from(t[3]) / 255.0,
                )
            })
            .collect();
        Self {
            tile,
            cols,
            params,
            drift: DRIFT * time,
            density: vec![0.0; RES * RES],
            out: vec![[0; 4]; RES * RES],
            row: 0,
        }
    }

    /// Advance `rows` rows; true once the image is complete.
    pub(super) fn step(&mut self, rows: usize) -> bool {
        let end = (self.row + rows).min(2 * RES);
        for r in self.row..end {
            if r < RES {
                self.density_row(r);
            } else {
                self.light_row(r - RES);
            }
        }
        self.row = end;
        self.row == 2 * RES
    }

    pub(super) fn run(&mut self) {
        self.step(2 * RES);
    }

    pub(super) fn image(&self) -> &[[u8; 4]] {
        &self.out
    }

    /// The tile at dome UV, bilinear and clamped to the edge as the GPU sampler reads it.
    fn tile_at(&self, uv: Vec2) -> Vec4 {
        let n = self.cols;
        let p = (uv * n as f32 - 0.5).clamp(Vec2::ZERO, Vec2::splat((n - 1) as f32));
        let i = p.floor();
        let f = p - i;
        let (x0, y0) = (i.x as usize, i.y as usize);
        let (x1, y1) = ((x0 + 1).min(n - 1), (y0 + 1).min(n - 1));
        let t = |x: usize, y: usize| self.tile[y * n + x];
        t(x0, y0)
            .lerp(t(x1, y0), f.x)
            .lerp(t(x0, y1).lerp(t(x1, y1), f.x), f.y)
    }

    fn density_row(&mut self, y: usize) {
        let s = self.params.style;
        for x in 0..RES {
            let uv = texel_uv(x, y);
            let a = self.tile_at(uv).w;
            let puffs = if a <= 0.0 {
                0.0
            } else {
                let p = uv * s.detail + self.drift;
                // Two octaves: a third would be finer than the layer's texels and alias.
                let n = 0.65 * billow(p) + 0.35 * billow(p * 2.03 + Vec2::new(5.2, 1.3));
                smoothstep(1.0 - a - s.soft, 1.0 - a + s.soft, n) * (0.5 + 0.5 * a)
            };
            self.density[y * RES + x] = puffs.max(s.deck);
        }
    }

    fn density_at(&self, uv: Vec2) -> f32 {
        let p = (uv * RES as f32).clamp(Vec2::ZERO, Vec2::splat((RES - 1) as f32));
        self.density[p.y as usize * RES + p.x as usize]
    }

    fn light_row(&mut self, y: usize) {
        let CloudParams {
            style: s,
            key,
            key_col,
            desat,
            elev,
            sky_col,
            ..
        } = self.params;
        // UV u follows world x and v world z (`clouds::layer`), so the march toward the key is its
        // horizontal direction; a key overhead marches nowhere and the cloud's own thickness shades
        // its belly.
        let march = Vec2::new(key.x, key.z) * s.step;
        for x in 0..RES {
            let uv = texel_uv(x, y);
            let d = self.density[y * RES + x];
            let mut path = 0.0;
            if d > 0.0 {
                for k in 1..=TAPS {
                    path += self.density_at(uv + march * (k as f32 / TAPS as f32));
                }
            }
            let to_key = (-s.sigma * (0.5 * d + 1.5 * path / TAPS as f32)).exp();
            let thin = (-d * s.density).exp();
            let powder = 1.0 - thin * thin;
            let lit = elev * to_key * (1.0 + (powder - 1.0) * s.powder);
            let authored = self.tile_at(uv).truncate();
            let luma = authored.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            // The shade is filled by the sky above it, with a quarter of the authored colour kept
            // for the zone's own cast.
            let ambient = sky_col.lerp(authored.lerp(Vec3::splat(luma), desat), 0.25) * s.ambient;
            let c = key_col * lit + ambient;
            let alpha = (1.0 - thin) * s.opacity;
            self.out[y * RES + x] = [
                linear_to_srgb8(c.x),
                linear_to_srgb8(c.y),
                linear_to_srgb8(c.z),
                (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
            ];
        }
    }
}

fn texel_uv(x: usize, y: usize) -> Vec2 {
    (Vec2::new(x as f32, y as f32) + 0.5) / RES as f32
}

fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// A cell's feature point in [0, 1)², from an integer hash.
fn feature(cx: i32, cy: i32) -> Vec2 {
    let mut h = (cx as u32).wrapping_mul(0x8da6_b343) ^ (cy as u32).wrapping_mul(0xd816_3841);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    let g = h.wrapping_mul(0x297a_2d39) ^ (h >> 15);
    Vec2::new((h & 0xffff) as f32 / 65536.0, (g & 0xffff) as f32 / 65536.0)
}

/// How sharply [`billow`]'s smooth minimum picks the nearest feature point.
const BILLOW_K: f32 = 9.0;

/// Inverted smooth Worley: a round dome over each cell's feature point, blended into its
/// neighbours by a smooth minimum (`−ln Σ e^{−k·d} / k`), so merging billows meet in rounded
/// creases rather than the straight Voronoi edges a hard F1 cuts.
fn billow(p: Vec2) -> f32 {
    let c = p.floor();
    let f = p - c;
    let (cx, cy) = (c.x as i32, c.y as i32);
    let mut sum = 0.0;
    for oy in -1..=1 {
        for ox in -1..=1 {
            let o = Vec2::new(ox as f32, oy as f32);
            let d = (o + feature(cx + ox, cy + oy) - f).length();
            sum += (-BILLOW_K * d).exp();
        }
    }
    1.0 - (-sum.max(1e-12).ln() / BILLOW_K).clamp(0.0, 1.0)
}

/// The bake in flight, how long since the last one began, and what it was baked from.
#[derive(Default)]
struct Baker {
    task: Option<Task<(CloudBake, f32)>>,
    local: Option<CloudBake>,
    since: Option<f32>,
    inputs: Option<u64>,
    logged: bool,
}

/// What a bake depends on, hashed: an unchanged field under an unchanged light is not rebaked.
fn inputs_hash(tile: &[[u8; 4]], params: &CloudParams) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tile.hash(&mut h);
    let s = params.style;
    for v in [
        s.opacity,
        s.soft,
        s.detail,
        s.desat,
        s.whiten,
        s.bright,
        s.density,
        s.step,
        s.sigma,
        s.ambient,
        s.powder,
        s.deck,
        params.desat,
        params.elev,
    ]
    .into_iter()
    .chain(params.key.to_array())
    .chain(params.key_col.to_array())
    .chain(params.sky_col.to_array())
    {
        // Quantized, so a drifting clock does not rebake a layer nobody could tell apart.
        ((v * 256.0).round() as i32).hash(&mut h);
    }
    h.finish()
}

pub(super) fn plugin(app: &mut App) {
    // `Last`: after `clouds::tick_clouds` advanced the coverage this frame.
    app.add_systems(Last, bake_clouds);
}

#[allow(clippy::too_many_arguments)]
fn bake_clouds(
    active: Res<ActiveLighting>,
    light: Res<WowLighting>,
    look: Res<super::look::ActiveLook>,
    coverage: Option<Res<CloudCoverage>>,
    layer: Option<Res<CloudLayer>>,
    time: Res<Time>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<CloudMaterial>>,
    mut baker: Local<Baker>,
) {
    let on = active.is_modern() && frame_is_modern() && super::cpu_hooks_enabled();
    let (Some(coverage), Some(layer), true) = (coverage, layer, on) else {
        // Dropping a task cancels it; the next modern frame starts over.
        *baker = Baker {
            logged: baker.logged,
            ..default()
        };
        return;
    };

    let mut done = None;
    if let Some(task) = baker.task.as_mut() {
        if let Some(result) = block_on(future::poll_once(task)) {
            baker.task = None;
            done = Some(result);
        }
    }
    if let Some(local) = baker.local.as_mut() {
        if local.step(FALLBACK_ROWS) {
            done = baker.local.take().map(|b| (b, f32::NAN));
        }
    }
    if let Some((bake, ms)) = done {
        if !baker.logged {
            baker.logged = true;
            if ms.is_nan() {
                info!("modern clouds: {RES}² layer baked on the main thread, {FALLBACK_ROWS} rows a frame");
            } else {
                info!("modern clouds: {RES}² layer baked on a worker in {ms:.1} ms");
            }
        }
        upload(&bake, &layer, &mut images, &mut materials);
    }

    let period = bake_period();
    let since = baker.since.map_or(f32::INFINITY, |s| s + time.delta_secs());
    baker.since = Some(since);
    let due = since >= period && (period > 0.0 || baker.inputs.is_none());
    if baker.task.is_none() && baker.local.is_none() && due {
        baker.since = Some(0.0);
        let params = cloud_params(&light, &look.0);
        let inputs = inputs_hash(coverage.tile(), &params);
        if baker.inputs == Some(inputs) {
            return;
        }
        baker.inputs = Some(inputs);
        // The billows drift with the bake's clock: they move once a bake, slowly.
        let bake = CloudBake::new(coverage.tile(), params, time.elapsed_secs_wrapped());
        // `spawn_pooled` does not exist on a wasm build without atomics; `pooled_work_runs`
        // answers false there too (`terrain_stream::collider`).
        #[cfg(not(all(target_arch = "wasm32", not(target_feature = "atomics"))))]
        if benilla_assets::frame_budget::pooled_work_runs() {
            baker.task = Some(
                bevy::tasks::AsyncComputeTaskPool::get().spawn_pooled(async move {
                    let start = bevy::platform::time::Instant::now();
                    let mut bake = bake;
                    bake.run();
                    (bake, start.elapsed().as_secs_f32() * 1000.0)
                }),
            );
            return;
        }
        baker.local = Some(bake);
    }
}

fn upload(
    bake: &CloudBake,
    layer: &CloudLayer,
    images: &mut Assets<Image>,
    materials: &mut Assets<CloudMaterial>,
) {
    let Some(mut image) = images.get_mut(&layer.image) else {
        return;
    };
    let size = Extent3d {
        width: RES as u32,
        height: RES as u32,
        depth_or_array_layers: 1,
    };
    if image.texture_descriptor.size != size {
        image.resize(size);
    }
    // Gamma bytes of a linear colour: sampled through an sRGB view, the shader reads linear.
    image.texture_descriptor.format = TextureFormat::Rgba8UnormSrgb;
    image
        .data
        .as_mut()
        .expect("cpu-side cloud image")
        .copy_from_slice(bake.image().as_flattened());
    // A modified Image gets a new GPU texture; touch the material so its bind group follows.
    // `into_inner`: a bare `AssetMut` emits nothing.
    if let Some(material) = materials.get_mut(&layer.material) {
        material.into_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(key: Vec3) -> CloudParams {
        CloudParams {
            style: CLOUD_STYLE,
            key: key.normalize(),
            key_col: Vec3::ONE,
            desat: 1.0,
            elev: 1.0,
            sky_col: Vec3::new(0.3, 0.45, 0.7),
            haze_col: Vec3::ONE,
        }
    }

    fn baked(tile: &[[u8; 4]], key: Vec3) -> CloudBake {
        let mut bake = CloudBake::new(tile, params(key), 0.0);
        bake.run();
        bake
    }

    #[test]
    fn a_clear_tile_bakes_a_clear_layer() {
        let bake = baked(&vec![[200, 200, 200, 0]; 128 * 128], Vec3::Y);
        assert!(bake.image().iter().all(|t| t[3] == 0));
    }

    /// A thin field leaves fewer, smaller puffs than a thick one: the billows are cut against the
    /// coverage, not the coverage blurred.
    #[test]
    fn thicker_coverage_merges_more_puffs() {
        let cover = |a: u8| {
            let bake = baked(&vec![[200, 200, 200, a]; 128 * 128], Vec3::Y);
            bake.image().iter().filter(|t| t[3] > 32).count() as f32 / (RES * RES) as f32
        };
        let (thin, thick) = (cover(40), cover(200));
        assert!(
            thin > 0.0 && thin < 0.5 * thick,
            "thin {thin:.3} thick {thick:.3}"
        );
    }

    /// The light march: with the key low toward +x over a cloud filling the tile's −x half, the
    /// sun-facing edge is lit and the core, whose path to the key runs through the cloud, is dark.
    #[test]
    fn the_sun_facing_edge_is_brighter_than_the_core() {
        let mut tile = vec![[200, 200, 200, 0]; 128 * 128];
        for y in 0..128 {
            for x in 0..64 {
                tile[y * 128 + x][3] = 255;
            }
        }
        let bake = baked(&tile, Vec3::new(1.0, 0.3, 0.0));
        let mean = |x0: usize, x1: usize| {
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for y in 0..RES {
                for x in x0..x1 {
                    let t = bake.image()[y * RES + x];
                    // Inside the cloud: past half the layer's peak opacity.
                    if f32::from(t[3]) > 0.5 * 255.0 * CLOUD_STYLE.opacity {
                        sum += f32::from(t[1]);
                        n += 1.0;
                    }
                }
            }
            sum / n.max(1.0)
        };
        let (core, edge) = (mean(64, 160), mean(240, 252));
        assert!(edge > core + 20.0, "edge {edge:.0} core {core:.0}");
    }

    /// Dev: `BENILLA_CLOUD_PNG=<path> cargo test -p benilla-world dump_a_bake -- --ignored` writes
    /// a bake of a synthetic field over a mid-blue sky, for judging the look without a web build.
    #[test]
    #[ignore]
    fn dump_a_bake() {
        let Ok(path) = std::env::var("BENILLA_CLOUD_PNG") else {
            return;
        };
        let tile: Vec<[u8; 4]> = (0..128 * 128)
            .map(|i| {
                let p = Vec2::new((i % 128) as f32, (i / 128) as f32) / 128.0;
                let a = 0.5
                    + 0.3 * (p.x * 9.0).sin() * (p.y * 7.0 + 1.0).cos()
                    + 0.2 * (p.x * 23.0 + p.y * 17.0).sin()
                    - 0.25;
                [210, 200, 205, (a.clamp(0.0, 1.0) * 255.0) as u8]
            })
            .collect();
        let key = std::env::var("BENILLA_CLOUD_KEY")
            .ok()
            .and_then(|v| {
                let c: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
            })
            .unwrap_or(Vec3::new(0.6, 0.5, 0.2));
        // `BENILLA_CLOUD_SKY=r,g,b` (linear): the skylight the shade takes and the backdrop.
        let sky = std::env::var("BENILLA_CLOUD_SKY")
            .ok()
            .and_then(|v| {
                let c: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
            })
            .unwrap_or(Vec3::new(0.18, 0.42, 0.78));
        let mut p = params(key);
        p.style = cloud_style();
        p.sky_col = sky;
        p.key_col = Vec3::new(1.0, 0.97, 0.92)
            * Vec3::ONE.lerp(sky / sky.max_element(), SKY_HUE)
            * p.style.bright;
        p.desat = 0.9;
        let mut bake = CloudBake::new(&tile, p, 0.0);
        bake.run();
        let sky = sky.to_array();
        let px: Vec<u8> = bake
            .image()
            .iter()
            .flat_map(|t| {
                let a = f32::from(t[3]) / 255.0;
                (0..3).map(move |c| {
                    let lin = srgb_to_linear(f32::from(t[c]) / 255.0);
                    linear_to_srgb8(lin * a + sky[c] * (1.0 - a))
                })
            })
            .collect();
        image::RgbImage::from_raw(RES as u32, RES as u32, px)
            .expect("rgb buffer")
            .save(&path)
            .expect("png");
    }

    /// Fallback and worker bakes are the same image.
    #[test]
    fn stepping_a_bake_matches_running_it() {
        let tile: Vec<[u8; 4]> = (0..128 * 128)
            .map(|i| [180, 170, 160, (i % 251) as u8])
            .collect();
        let whole = baked(&tile, Vec3::new(0.3, 0.6, 0.2));
        let mut stepped = CloudBake::new(&tile, params(Vec3::new(0.3, 0.6, 0.2)), 0.0);
        while !stepped.step(FALLBACK_ROWS) {}
        assert!(whole.image() == stepped.image());
    }
}
