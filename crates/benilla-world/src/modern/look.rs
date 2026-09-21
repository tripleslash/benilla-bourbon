//! **Per-zone look** (fork-owned; `FORK.md` P4, amendments A1–A3).
//!
//! The planned techniques reproduce the reference shots' effects; what sells them is calibration. One
//! row per zone sets:
//!
//! - `key` / `fill` — the key-to-fill ratio (A3): gains on the shared light's diffuse and ambient rows
//!   in a modern frame (`global_light::build_light_data`), aimed at 6–10:1 under forest canopies;
//! - `exposure` — the post's exposure (the FFXGlow combine's `wave.w`);
//! - `haze` — the shaft haze extinction, never near-cleared, so shafts start at the camera (A1);
//! - `forward` — the dual-lobe phase's forward share: warm toward the sun, the zone's hue away (A2);
//! - `key_scatter` — how strongly the key light scatters into the fog;
//! - `glow` — the colour of the light scattered toward the sun (the froxel key in-scatter and the sky's
//!   Mie halo), and `haze_tint` — the colour of the ambient haze away from it. Classic Light.dbc fog
//!   is often grey where the reference grade is warm: Darkshore's salmon dawn is these two rows.
//!
//! Zones are `AreaTable.dbc` top-level rows; a subzone resolves through its parents. The area comes
//! from the character's [`CurrentArea`], or — with no avatar (the engine-only world view) — from the
//! terrain chunk under the camera. `$WOW_MODERN_ZONE` (browser `?modern_zone=`) pins one for A/B.
//! Looks cross-fade over about two seconds.

use std::sync::atomic::{AtomicU32, Ordering};

use bevy::prelude::*;

use benilla_assets::{AdtTile, AssetSet, LockRecover, WorldAssets};
use benilla_formats::AreaTableCatalog;

use super::ActiveLighting;
use crate::terrain_stream::{area_id_under, CurrentArea, TerrainStreamer};
use crate::view::WorldCamera;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneLook {
    pub key: f32,
    pub fill: f32,
    pub exposure: f32,
    pub haze: f32,
    pub forward: f32,
    pub key_scatter: f32,
    pub glow: [f32; 3],
    pub haze_tint: [f32; 3],
    /// How far the long-range height fog reaches against the zone's authored fog end (>1 = clearer).
    pub fog_reach: f32,
    /// A gamma multiplier on the authored fog colour the medium is solved toward.
    pub fog_tint: [f32; 3],
    /// Yards over which the fog's density falls by `e` above the regional ground.
    pub fog_falloff: f32,
    /// A gamma multiplier on the sky dome's five authored stops.
    pub sky_tint: [f32; 3],
    /// The distant (WDL) silhouette's shade against the fog colour.
    pub silhouette: f32,
    /// The fog is clear at least this far out (planar yd): the scene around the camera stays crisp
    /// while the long range still fades into the ground fog.
    pub fog_clear: f32,
    /// Gain on the key light scattered by the shaft haze — how hard light cones read.
    pub shafts: f32,
    /// The post's cinematic grade: contrast, saturation, vignette strength, and toe (gamma > 1 deepens
    /// the shadows).
    pub contrast: f32,
    pub saturation: f32,
    pub vignette: f32,
    pub toe: f32,
}

impl ZoneLook {
    /// Every zone without a row: the P3 balance, a light haze.
    pub const NEUTRAL: Self = Self {
        key: 1.0,
        fill: 1.0,
        // The Durotar calibration (the modern client's gate shot) is the default everywhere: a crisp
        // scene out to 450 yd, a dense ground fog that hugs the terrain (so distant ranges stand as
        // silhouette layers over bright haze at their feet), and a cinematic grade (eased off the gate
        // shot's darker one for everyday play). Zone rows
        // override the colour.
        exposure: 1.1,
        haze: 0.0002,
        forward: 0.7,
        key_scatter: 0.6,
        glow: [1.0, 1.0, 1.0],
        haze_tint: [1.0, 1.0, 1.0],
        fog_reach: 0.5,
        fog_tint: [1.0, 1.0, 1.0],
        fog_falloff: 70.0,
        sky_tint: [1.0, 1.0, 1.0],
        silhouette: 0.55,
        fog_clear: 450.0,
        shafts: 0.3,
        contrast: 1.12,
        saturation: 0.86,
        vignette: 0.3,
        toe: 1.03,
    };

    fn lerp(self, to: Self, t: f32) -> Self {
        let m = |a: f32, b: f32| a + (b - a) * t;
        Self {
            key: m(self.key, to.key),
            fill: m(self.fill, to.fill),
            exposure: m(self.exposure, to.exposure),
            haze: m(self.haze, to.haze),
            forward: m(self.forward, to.forward),
            key_scatter: m(self.key_scatter, to.key_scatter),
            glow: [0, 1, 2].map(|i| m(self.glow[i], to.glow[i])),
            haze_tint: [0, 1, 2].map(|i| m(self.haze_tint[i], to.haze_tint[i])),
            fog_reach: m(self.fog_reach, to.fog_reach),
            fog_tint: [0, 1, 2].map(|i| m(self.fog_tint[i], to.fog_tint[i])),
            fog_falloff: m(self.fog_falloff, to.fog_falloff),
            sky_tint: [0, 1, 2].map(|i| m(self.sky_tint[i], to.sky_tint[i])),
            silhouette: m(self.silhouette, to.silhouette),
            fog_clear: m(self.fog_clear, to.fog_clear),
            shafts: m(self.shafts, to.shafts),
            contrast: m(self.contrast, to.contrast),
            saturation: m(self.saturation, to.saturation),
            vignette: m(self.vignette, to.vignette),
            toe: m(self.toe, to.toe),
        }
    }

    fn close_to(self, other: Self) -> bool {
        let d = |a: f32, b: f32| (a - b).abs() < 1e-4;
        d(self.key, other.key)
            && d(self.fill, other.fill)
            && d(self.exposure, other.exposure)
            && d(self.haze, other.haze)
            && d(self.forward, other.forward)
            && d(self.key_scatter, other.key_scatter)
            && d(self.fog_reach, other.fog_reach)
            && d(self.fog_falloff, other.fog_falloff)
            && d(self.silhouette, other.silhouette)
            && d(self.fog_clear, other.fog_clear)
            && d(self.shafts, other.shafts)
            && d(self.contrast, other.contrast)
            && d(self.saturation, other.saturation)
            && d(self.vignette, other.vignette)
            && d(self.toe, other.toe)
            && (0..3).all(|i| {
                d(self.glow[i], other.glow[i])
                    && d(self.haze_tint[i], other.haze_tint[i])
                    && d(self.fog_tint[i], other.fog_tint[i])
                    && d(self.sky_tint[i], other.sky_tint[i])
            })
    }
}

/// The look table, by `AreaTable.dbc` zone id.
const ZONES: &[(u32, ZoneLook)] = &[
    // Teldrassil: dense purple canopy haze, a strong forward glow through the openings.
    (141, ZoneLook { key: 1.1, fill: 0.92, exposure: 1.08, haze: 0.0030, forward: 0.78, key_scatter: 0.85, ..ZoneLook::NEUTRAL }),
    // Darnassus: Teldrassil's light, a little clearer. Its plazas are open sky, so the canopy boost the
    // forest's beams ride on does not reach them: the shaft gain carries the city's own.
    (1657, ZoneLook { key: 1.1, fill: 0.95, exposure: 1.0, haze: 0.0026, forward: 0.75, key_scatter: 0.75, shafts: 0.6, ..ZoneLook::NEUTRAL }),
    // Ashenvale: cyan-teal volumetric haze and deep contrast under the trees.
    (331, ZoneLook { key: 1.15, fill: 0.88, exposure: 1.0, haze: 0.0039, forward: 0.68, key_scatter: 0.75, shafts: 1.5, ..ZoneLook::NEUTRAL }),
    // Darkshore: pink forward-scatter toward a low sun over dense coastal fog.
    (148, ZoneLook {
        key: 1.05,
        fill: 0.95,
        exposure: 1.0,
        haze: 0.0028,
        forward: 0.82,
        key_scatter: 0.95,
        glow: [1.0, 0.56, 0.42],
        haze_tint: [0.92, 0.74, 0.74],
        ..ZoneLook::NEUTRAL
    }),
    // Durotar: the calibration zone for the defaults above (the gate shot, `worldview_pose=930,-4440,85,
    // -5,9` at 17:30); its amber colour and darker grade are its own.
    (14, ZoneLook {
        fog_tint: [1.7, 2.6, 1.8],
        sky_tint: [1.05, 1.15, 0.8],
        // The gate shot's own darker grade (the defaults ease it for everyday play).
        exposure: 0.95,
        contrast: 1.2,
        saturation: 0.82,
        vignette: 0.4,
        toe: 1.2,
        ..ZoneLook::NEUTRAL
    }),
    // Mulgore: warm plains haze, a peach band on the horizon at dusk behind the tan mesa layers, and the
    // bluffs crisp out to their far rises (checked against the modern client's Thunder Bluff sunset,
    // `worldview_pose=-1635,75,198,13,-3`, 19:00).
    (215, ZoneLook {
        exposure: 1.05,
        haze: 0.0004,
        glow: [1.4, 0.85, 0.6],
        haze_tint: [1.1, 0.85, 0.8],
        // Mulgore's authored daytime fog end is short (~890 yd): at reach 1 the plains past the clear
        // range went milky by noon.
        fog_reach: 1.6,
        fog_tint: [2.3, 1.25, 0.85],
        fog_falloff: 70.0,
        sky_tint: [0.85, 1.0, 1.25],
        silhouette: 0.75,
        fog_clear: 400.0,
        shafts: 0.3,
        ..ZoneLook::NEUTRAL
    }),
    // Elwynn Forest: treeline shafts at dusk.
    (12, ZoneLook { key: 1.1, fill: 0.9, exposure: 1.0, haze: 0.0024, forward: 0.72, key_scatter: 0.7, ..ZoneLook::NEUTRAL }),
    // Duskwood: gloom — a weak key, a darker exposure.
    (10, ZoneLook { key: 1.0, fill: 0.95, exposure: 0.95, haze: 0.0030, forward: 0.6, key_scatter: 0.45, ..ZoneLook::NEUTRAL }),
    // Felwood: a sickly canopy.
    (361, ZoneLook { key: 1.1, fill: 0.88, exposure: 1.0, haze: 0.0036, forward: 0.65, key_scatter: 0.6, ..ZoneLook::NEUTRAL }),
    // Stormwind City: a light street haze, enough for the sun to fall between the rooftops as beams —
    // a city has no canopy boost, so the shaft gain stands in for it (the streets are open sky).
    (1519, ZoneLook { key: 1.0, fill: 1.0, exposure: 1.0, haze: 0.0018, forward: 0.7, key_scatter: 0.6, shafts: 0.6, ..ZoneLook::NEUTRAL }),
    // Orgrimmar: Durotar's colour and grade, with the dusty canyon haze the plains around it do not
    // need — the canyon walls and the spiked roofs are what cut the beams.
    (1637, ZoneLook {
        fog_tint: [1.7, 2.6, 1.8],
        sky_tint: [1.05, 1.15, 0.8],
        exposure: 0.95,
        contrast: 1.2,
        saturation: 0.82,
        vignette: 0.4,
        toe: 1.2,
        haze: 0.0018,
        glow: [1.25, 0.95, 0.75],
        key_scatter: 0.7,
        shafts: 0.6,
        ..ZoneLook::NEUTRAL
    }),
    // Ironforge: an interior city, so its haze stays thin.

    (1537, ZoneLook { key: 1.0, fill: 1.0, exposure: 1.0, haze: 0.0009, forward: 0.6, key_scatter: 0.4, ..ZoneLook::NEUTRAL }),
];

pub fn look_for_zone(zone: u32) -> ZoneLook {
    ZONES
        .iter()
        .find(|(id, _)| *id == zone)
        .map_or(ZoneLook::NEUTRAL, |(_, look)| *look)
}

/// The look this frame renders with (cross-faded toward the current zone's row).
#[derive(Resource, Clone, Copy, Debug)]
pub struct ActiveLook(pub ZoneLook);

impl Default for ActiveLook {
    fn default() -> Self {
        Self(ZoneLook::NEUTRAL)
    }
}

/// `AreaTable.dbc`, for walking a subzone up to its zone.
#[derive(Resource)]
struct ZoneCatalog(AreaTableCatalog);

const ONE_BITS: u32 = 0x3f80_0000;
static KEY: AtomicU32 = AtomicU32::new(ONE_BITS);
static FILL: AtomicU32 = AtomicU32::new(ONE_BITS);
static EXPOSURE: AtomicU32 = AtomicU32::new(ONE_BITS);
static FOG_TINT: [AtomicU32; 3] = [const { AtomicU32::new(ONE_BITS) }; 3];
static SKY_TINT: [AtomicU32; 3] = [const { AtomicU32::new(ONE_BITS) }; 3];
static SILHOUETTE: AtomicU32 = AtomicU32::new(0x3f33_3333); // 0.7
static GRADE: [AtomicU32; 4] = [const { AtomicU32::new(0) }; 4];

/// The zone look's cinematic grade for the post: `[contrast, saturation, vignette, toe]`.
pub fn grade() -> [f32; 4] {
    [0, 1, 2, 3].map(|i| f32::from_bits(GRADE[i].load(Ordering::Relaxed)))
}

/// The zone look's distant-silhouette shade (the WDL band, a modern frame only).
pub fn silhouette() -> f32 {
    f32::from_bits(SILHOUETTE.load(Ordering::Relaxed))
}

/// The zone look's gamma multiplier on the authored fog colour (1 in classic).
pub fn fog_tint() -> [f32; 3] {
    [0, 1, 2].map(|i| f32::from_bits(FOG_TINT[i].load(Ordering::Relaxed)))
}

/// The zone look's gamma multiplier on the sky dome's stops (1 in classic).
pub fn sky_tint() -> [f32; 3] {
    [0, 1, 2].map(|i| f32::from_bits(SKY_TINT[i].load(Ordering::Relaxed)))
}

/// The key and fill gains for the shared light's rows (1, 1 in a classic session).
pub fn key_fill() -> (f32, f32) {
    (
        // The key also fades with its body's elevation (`sun_arc::key_horizon`, 1 in classic).
        f32::from_bits(KEY.load(Ordering::Relaxed)) * super::sun_arc::key_horizon(),
        f32::from_bits(FILL.load(Ordering::Relaxed)),
    )
}

/// The post's exposure for a modern frame.
pub fn exposure() -> f32 {
    f32::from_bits(EXPOSURE.load(Ordering::Relaxed))
}

fn publish(look: ZoneLook) {
    KEY.store(look.key.to_bits(), Ordering::Relaxed);
    FILL.store(look.fill.to_bits(), Ordering::Relaxed);
    EXPOSURE.store((super::MODERN_EXPOSURE * look.exposure).to_bits(), Ordering::Relaxed);
    SILHOUETTE.store(look.silhouette.to_bits(), Ordering::Relaxed);
    for (i, v) in [look.contrast, look.saturation, look.vignette, look.toe].into_iter().enumerate() {
        GRADE[i].store(v.to_bits(), Ordering::Relaxed);
    }
    for i in 0..3 {
        FOG_TINT[i].store(look.fog_tint[i].to_bits(), Ordering::Relaxed);
        SKY_TINT[i].store(look.sky_tint[i].to_bits(), Ordering::Relaxed);
    }
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<ActiveLook>()
        .add_systems(Startup, load_zone_catalog.after(AssetSet::Open))
        .add_systems(Last, update_look);
}

fn load_zone_catalog(mut commands: Commands, assets: Option<Res<WorldAssets>>) {
    let Some(assets) = assets else {
        return;
    };
    let loaded = {
        let mut chain = assets.chain.lock_recover();
        benilla_formats::load_area_table_catalog(&mut chain)
    };
    match loaded {
        Ok(catalog) => commands.insert_resource(ZoneCatalog(catalog)),
        Err(e) => warn!("modern look: AreaTable failed to load, zone looks stay neutral: {e:#}"),
    }
}

/// Walk an area up to its top-level zone.
fn zone_of(catalog: &AreaTableCatalog, area: u32) -> u32 {
    let mut id = area;
    for _ in 0..4 {
        match catalog.get(id) {
            Some(row) if row.zone_id != 0 && row.zone_id != id => id = row.zone_id,
            _ => break,
        }
    }
    id
}

/// Dev tuning (`$WOW_MODERN_LOOK`, browser `?modern_look=`): comma-separated overrides of the active
/// row, in field order — `key,fill,exposure,haze,forward,key_scatter,glow r,g,b,haze_tint r,g,b,
/// fog_reach,fog_tint r,g,b,fog_falloff,sky_tint r,g,b,silhouette,fog_clear,shafts,contrast,saturation,vignette,toe`. An empty or unparsable entry keeps the zone's value.
fn look_override() -> Vec<Option<f32>> {
    crate::launch_env::live_var("WOW_MODERN_LOOK")
        .map(|v| v.split(',').map(|p| p.trim().parse().ok()).collect())
        .unwrap_or_default()
}

fn apply_override(mut look: ZoneLook, o: &[Option<f32>]) -> ZoneLook {
    let get = |i: usize| o.get(i).copied().flatten();
    let set = |v: &mut f32, i: usize| {
        if let Some(x) = get(i) {
            *v = x;
        }
    };
    set(&mut look.key, 0);
    set(&mut look.fill, 1);
    set(&mut look.exposure, 2);
    set(&mut look.haze, 3);
    set(&mut look.forward, 4);
    set(&mut look.key_scatter, 5);
    for c in 0..3 {
        set(&mut look.glow[c], 6 + c);
        set(&mut look.haze_tint[c], 9 + c);
        set(&mut look.fog_tint[c], 13 + c);
        set(&mut look.sky_tint[c], 17 + c);
    }
    set(&mut look.fog_reach, 12);
    set(&mut look.fog_falloff, 16);
    set(&mut look.silhouette, 20);
    set(&mut look.fog_clear, 21);
    set(&mut look.shafts, 22);
    set(&mut look.contrast, 23);
    set(&mut look.saturation, 24);
    set(&mut look.vignette, 25);
    set(&mut look.toe, 26);
    look
}

fn pinned_zone() -> Option<u32> {
    static CELL: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MODERN_ZONE").and_then(|v| v.trim().parse().ok()))
}

/// Seconds between the no-avatar terrain probes.
const PROBE_INTERVAL: f32 = 0.5;

#[allow(clippy::too_many_arguments)]
fn update_look(
    active: Res<ActiveLighting>,
    area: Option<Res<CurrentArea>>,
    catalog: Option<Res<ZoneCatalog>>,
    streamer: Option<Res<TerrainStreamer>>,
    tiles: Option<Res<Assets<AdtTile>>>,
    cameras: Query<&GlobalTransform, With<WorldCamera>>,
    time: Res<Time>,
    mut look: ResMut<ActiveLook>,
    mut probe: Local<(f32, Option<u32>)>,
) {
    if !active.is_modern() {
        if look.0 != ZoneLook::NEUTRAL {
            look.0 = ZoneLook::NEUTRAL;
        }
        publish(ZoneLook::NEUTRAL);
        return;
    }
    let from_avatar = area.and_then(|a| a.0);
    let area_id = pinned_zone().or(from_avatar).or_else(|| {
        probe.0 -= time.delta_secs();
        if probe.0 <= 0.0 {
            probe.0 = PROBE_INTERVAL;
            probe.1 = match (streamer.as_deref(), tiles.as_deref(), cameras.iter().next()) {
                (Some(s), Some(t), Some(cam)) => area_id_under(s, t, cam.translation()),
                _ => None,
            };
        }
        probe.1
    });
    let zone = match (area_id, catalog.as_deref()) {
        (Some(id), Some(c)) => Some(zone_of(&c.0, id)),
        (id, None) => id,
        (None, _) => None,
    };
    let target = apply_override(zone.map_or(ZoneLook::NEUTRAL, look_for_zone), &look_override());
    let next = look.0.lerp(target, (time.delta_secs() / 2.0).min(1.0));
    let next = if next.close_to(target) { target } else { next };
    if next != look.0 {
        look.0 = next;
    }
    publish(look.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modern::srgb_to_linear;

    /// Lit-over-shadowed luminance for a typical forest noon (ambient 0.42, diffuse 0.95, gamma): the
    /// lanes sum in gamma, clamp, and the frame decodes.
    fn key_to_fill(look: ZoneLook) -> f32 {
        let fill = 0.42 * look.fill;
        let lit = (fill + 0.95 * look.key).min(1.0);
        srgb_to_linear(lit) / srgb_to_linear(fill)
    }

    /// Amendment A3's target: 6–10:1 under a canopy, and no row outside that band either.
    #[test]
    fn every_row_keeps_the_key_to_fill_ratio_in_band() {
        for (zone, look) in ZONES.iter().chain(std::iter::once(&(0, ZoneLook::NEUTRAL))) {
            let r = key_to_fill(*look);
            assert!((6.0..=10.0).contains(&r), "zone {zone}: {r:.2}:1");
        }
    }

    #[test]
    fn an_unlisted_zone_is_neutral_and_looks_cross_fade() {
        assert_eq!(look_for_zone(999_999), ZoneLook::NEUTRAL);
        let teldrassil = look_for_zone(141);
        let half = ZoneLook::NEUTRAL.lerp(teldrassil, 0.5);
        assert!((half.haze - (ZoneLook::NEUTRAL.haze + teldrassil.haze) / 2.0).abs() < 1e-6);
    }
}
