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

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use bevy::prelude::*;

use benilla_assets::DataLoadAppExt;
use benilla_assets::{AdtTile, LockRecover, WorldAssets};
use benilla_formats::AreaTableCatalog;

use super::{srgb_to_linear, ActiveLighting};
use crate::lighting::{GameClock, WowLighting};
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
    /// A multiplier on the key light's colour, per channel — for a zone whose authored sun is a hue
    /// the reference grade does not carry (Durotar's noon sun is a deep orange where the reference
    /// shot's is near white).
    pub key_tint: [f32; 3],
    /// fork: the ground-fog banks by day, as a share of the night's (`volumetrics`): 0 at night only,
    /// 1 as thick at noon as at midnight.
    pub ground_fog: f32,
    /// fork: the night's ground-fog banks as a share of the full bank (`volumetrics`): an ordinary
    /// zone lies clear at night, and only the dark zones (eerie, Tirisfal, Darkshore) keep banks, and
    /// the misty ones a share on the default look (`MIST_ZONES`).
    pub night_banks: f32,
    /// fork: the night air's glow (`volumetrics::NIGHT_GLOW`) as a share: an ordinary zone's night
    /// stays clear, a dark zone's air glows grey.
    pub night_glow: f32,
    /// fork: how far a building's own fog (its MFOG) reaches against its authored span (>1 = clearer).
    pub room_reach: f32,
    /// fork: the noon sun's hue (gamma), taken at the authored key's luminance
    /// ([`ZoneLook::noon_key_row`]). Elwynn's authored noon row is `[255, 136, 0]`, a sunset orange
    /// with no blue for a multiplier to lift, where the reference shot's noon sun is golden.
    pub noon_key: [f32; 3],
    /// fork: how far the key moves to [`Self::noon_key`] (0 keeps the authored row).
    pub noon_mix: f32,
    /// fork: the sea fog's extinction at sea level (1/yd): a bank lying on the ocean along the
    /// whole coast, clear over the beach (`froxel_scatter.wgsl`'s `sea_fog`). 0 = none.
    pub sea_fog: f32,
    /// fork: the share of the fog's forward lobe kept while the moon keys (`sun_arc::moon_key`):
    /// in a dense zone fog the lobe that glows salmon around the sun made a second sun of the moon.
    /// 1 = as by day.
    pub moon_glow: f32,
    /// fork: how much of the look's sun hues (`glow`, `key_tint`) the moonlight keeps
    /// ([`ZoneLook::key_hue`]): 0 lights the moon in its authored colour, never warm like the sun.
    pub moon_hue: f32,
    /// fork: a gain on [`MOON_KEY`] in this zone. The moon lights each zone with its own night
    /// rows, and over a bright authored night diffuse (Elwynn's) twice the key read as dusk, where
    /// snow and the dark zones need it to read at all. 1 = the default look's night.
    pub moon_key: f32,
    /// fork: how much the ground banks stand out from the haze under the moon: 0 lights them as
    /// the shaft haze is lit ([`MOON_HAZE`] on their isotropic half) and lays only a hazy zone's
    /// share of the night's banks (`volumetrics::added_air`); 1 lights them by the whole moon and
    /// lays the whole night bank, dense fog lying on the ground.
    pub bank_moon: f32,
}

impl ZoneLook {
    /// fork: **Elwynn's look, in every zone.** The Elwynn Forest calibration (treeline shafts at dusk)
    /// is the one look: every zone gets its shaft haze, key-to-fill and grade, so Stranglethorn's
    /// jungle, Durotar, the Barrens and Mulgore carry the same light cones Elwynn does. A zone that
    /// needs its own tuning takes a look built on this ([`ZoneLook::EERIE`]).
    pub const NEUTRAL: Self = Self {
        key: 1.1,
        fill: 0.9,
        exposure: 1.0,
        haze: 0.0024,
        forward: 0.72,
        key_scatter: 0.7,
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
        key_tint: [1.0, 1.0, 1.0],
        ground_fog: 0.0,
        night_banks: 0.0,
        night_glow: 0.33,
        room_reach: 1.0,
        noon_key: [1.0, 0.8, 0.4],
        noon_mix: 0.0,
        sea_fog: 0.0,
        moon_glow: 1.0,
        moon_hue: 1.0,
        moon_key: 1.0,
        bank_moon: 0.0,
    };

    /// fork: **the default look at noon**, which [`look_for_zone_at`] ramps [`Self::NEUTRAL`] into
    /// from 09:00 to 11:00 and out of from 14:30 to 16:30 (`sun_arc::midday_at`). Against the
    /// Goldshire-to-Stormwind reference shot: a golden sun at the authored key's brightness, a
    /// cooler haze in the distance, and a richer, harder grade; the morning's and the evening's warm
    /// rows keep the default look.
    pub const NOON: Self = Self {
        fill: 0.85,
        haze_tint: [0.8, 1.0, 1.05],
        contrast: 1.28,
        saturation: 1.12,
        toe: 1.07,
        noon_mix: 1.0,
        ..Self::NEUTRAL
    };

    /// The key's colour row moved `noon_mix` of the way to [`Self::noon_key`] at the row's own
    /// linear luminance, so the noon sun changes hue and never brightness.
    pub fn noon_key_row(&self, row: [f32; 3]) -> [f32; 3] {
        let t = self.noon_mix.clamp(0.0, 1.0);
        if t == 0.0 {
            return row;
        }
        let lin = |c: [f32; 3]| c.map(srgb_to_linear);
        let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let hue = lin(self.noon_key);
        let scale = lum(lin(row)) / lum(hue).max(1e-4);
        let noon = hue.map(|c| linear_to_srgb(c * scale));
        [0, 1, 2].map(|i| row[i] * (1.0 - t) + noon[i] * t)
    }

    /// One of the look's sun hues (`glow`, `key_tint`) as the key light takes it now: faded toward
    /// white by `1 - moon_hue` while the moon keys (`sun_arc::moon_key`).
    pub fn key_hue(&self, hue: [f32; 3]) -> [f32; 3] {
        let t = (1.0 - self.moon_hue.clamp(0.0, 1.0)) * super::sun_arc::moon_key();
        hue.map(|c| c + (1.0 - c) * t)
    }

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
            key_tint: [0, 1, 2].map(|i| m(self.key_tint[i], to.key_tint[i])),
            ground_fog: m(self.ground_fog, to.ground_fog),
            night_banks: m(self.night_banks, to.night_banks),
            night_glow: m(self.night_glow, to.night_glow),
            room_reach: m(self.room_reach, to.room_reach),
            noon_key: [0, 1, 2].map(|i| m(self.noon_key[i], to.noon_key[i])),
            noon_mix: m(self.noon_mix, to.noon_mix),
            sea_fog: m(self.sea_fog, to.sea_fog),
            moon_glow: m(self.moon_glow, to.moon_glow),
            moon_hue: m(self.moon_hue, to.moon_hue),
            moon_key: m(self.moon_key, to.moon_key),
            bank_moon: m(self.bank_moon, to.bank_moon),
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
            && d(self.ground_fog, other.ground_fog)
            && d(self.night_banks, other.night_banks)
            && d(self.night_glow, other.night_glow)
            && d(self.room_reach, other.room_reach)
            && d(self.noon_mix, other.noon_mix)
            && d(self.sea_fog, other.sea_fog)
            && d(self.moon_glow, other.moon_glow)
            && d(self.moon_hue, other.moon_hue)
            && d(self.moon_key, other.moon_key)
            && d(self.bank_moon, other.bank_moon)
            && (0..3).all(|i| {
                d(self.glow[i], other.glow[i])
                    && d(self.haze_tint[i], other.haze_tint[i])
                    && d(self.fog_tint[i], other.fog_tint[i])
                    && d(self.sky_tint[i], other.sky_tint[i])
                    && d(self.key_tint[i], other.key_tint[i])
                    && d(self.noon_key[i], other.noon_key[i])
            })
    }
}

impl ZoneLook {
    /// fork: **the eerie look.** Brill's calibration (Tirisfal at 14:00 against a reference shot),
    /// for every dark zone: a dim green sun over a deep fill, so the dapple reads under the canopy;
    /// the shafts lit green through a haze whose shadowed air stays dark; green fog and sky, a lower
    /// exposure and a saturated, harder grade; and the night's full ground-fog banks and glowing
    /// air, kept through the day.
    pub const EERIE: Self = Self {
        key: 0.85,
        fill: 0.55,
        exposure: 0.6,
        glow: [0.65, 1.0, 0.7],
        haze_tint: [0.22, 0.34, 0.26],
        fog_tint: [0.8, 1.0, 0.8],
        sky_tint: [0.6, 0.7, 0.6],
        key_tint: [0.78, 1.0, 0.8],
        saturation: 1.15,
        contrast: 1.25,
        haze: 0.0034,
        shafts: 0.4,
        vignette: 0.38,
        ground_fog: 0.7,
        night_banks: 1.0,
        night_glow: 1.0,
        noon_mix: 0.0,
        // The dark zones need the doubled moon to read at night (Duskwood, approved in the client).
        moon_key: 2.0,
        // Their night banks are ground fog, not more haze: whole, and lit by the whole moon.
        bank_moon: 1.0,
        ..Self::NEUTRAL
    };

    /// fork: **Tirisfal Glades**, the eerie look tinted greener (asked in the viewer): the sun,
    /// shafts, fog and sky lean further from red and blue toward green.
    pub const TIRISFAL: Self = Self {
        glow: [0.58, 1.0, 0.62],
        haze_tint: [0.19, 0.36, 0.22],
        fog_tint: [0.7, 1.05, 0.7],
        sky_tint: [0.52, 0.72, 0.52],
        key_tint: [0.72, 1.0, 0.74],
        ..Self::EERIE
    };
}

impl ZoneLook {
    /// fork: **the cold look.** Dun Morogh's calibration (Kharanos at noon, approved in the viewer),
    /// for the snow zones: a blue key, fog, sky and haze over the snow, the far ridges set in a
    /// blue haze while the near field stays crisp, and a brighter exposure for the snow.
    pub const COLD: Self = Self {
        fill: 1.0,
        exposure: 1.22,
        haze: 0.004,
        forward: 0.6,
        key_scatter: 0.6,
        glow: [0.72, 0.88, 1.18],
        haze_tint: [0.42, 0.68, 1.28],
        fog_reach: 0.38,
        fog_tint: [0.58, 0.82, 1.32],
        sky_tint: [0.6, 0.84, 1.32],
        fog_clear: 250.0,
        shafts: 0.25,
        contrast: 1.1,
        saturation: 1.0,
        vignette: 0.28,
        key_tint: [0.72, 0.9, 1.2],
        noon_mix: 0.0,
        ..Self::NEUTRAL
    };

    /// fork: **the cold look at night**, which [`look_for_zone_at`] ramps [`Self::COLD`] into
    /// across `sun_arc::night_at`. Dun Morogh's night fog ends at 417 yd, and the day's thick blue
    /// haze over it veiled the moonlit snow in a grey wash brighter than the snow itself: the night
    /// air is clear and the long-range fog reaches past the authored end (Gates of Ironforge at
    /// 02:00, approved in the viewer).
    pub const COLD_NIGHT: Self = Self {
        haze: 0.0012,
        fog_reach: 1.2,
        // The moonlit snow reads as snow (Gates of Ironforge at 02:00, approved in the viewer).
        moon_key: 2.0,
        ..Self::COLD
    };
}

impl ZoneLook {
    /// fork: **Ironforge.** The city is one hall under the mountain, whose MFOG filled it with a blue
    /// wash at Cinematic: the room fog reaches three times its authored span, the shaft haze is near
    /// cleared with a faint forge-warm tint, and the exposure lifts the hall (approved in the viewer).
    pub const IRONFORGE: Self = Self {
        fill: 1.05,
        exposure: 1.3,
        haze: 0.0008,
        key_scatter: 0.4,
        haze_tint: [1.0, 0.92, 0.82],
        fog_reach: 1.5,
        fog_clear: 900.0,
        shafts: 0.15,
        contrast: 1.08,
        saturation: 0.95,
        vignette: 0.2,
        toe: 1.0,
        night_banks: 0.0,
        night_glow: 0.0,
        room_reach: 3.0,
        noon_mix: 0.0,
        ..Self::NEUTRAL
    };
}

impl ZoneLook {
    /// fork: **Darkshore**, at every hour. Against a modern-client reference at the Long Wash
    /// looking up the coast to Auberdine (`worldview_pose=6187.7,520.7,20,21.5,-4,5`, map 1, 17:00,
    /// approved in the viewer): the zone's authored
    /// `Light.dbc` rows are grey all day (fog `[60 60 59]`, sun `[224 224 224]` at noon), where the
    /// reference is a mauve haze glowing salmon toward the sun. The fog tint lifts the grey into
    /// that mauve, the glow warms the sun side, the piers stand as silhouettes, and a thick sea fog
    /// lies along the whole coast.
    pub const DARKSHORE: Self = Self {
        key: 0.85,
        fill: 0.95,
        exposure: 0.95,
        haze: 0.0025,
        forward: 0.82,
        key_scatter: 0.9,
        glow: [1.15, 0.7, 0.6],
        haze_tint: [0.85, 0.76, 0.78],
        fog_reach: 0.7,
        fog_tint: [2.1, 1.85, 1.85],
        fog_falloff: 100.0,
        sky_tint: [1.15, 1.0, 1.0],
        silhouette: 0.75,
        fog_clear: 80.0,
        contrast: 1.12,
        saturation: 0.85,
        vignette: 0.32,
        key_tint: [1.0, 0.88, 0.85],
        ground_fog: 0.3,
        night_banks: 0.6,
        night_glow: 0.5,
        sea_fog: 0.06,
        // The moon lights the haze from every side, not as a second sun.
        moon_glow: 0.1,
        // The salmon glow and warm key are the sun's: the moon lights in its own colour.
        moon_hue: 0.0,
        // A dark zone's moon, as the eerie look's.
        moon_key: 2.0,
        ..Self::NEUTRAL
    };

    /// fork: **Darkshore at midday**, which [`look_for_zone_at`] ramps [`Self::DARKSHORE`] into
    /// across `sun_arc::midday_at`, as the default look takes [`Self::NOON`]. The salmon glow and
    /// warm fog read golden under the white noon sun (asked in the viewer): the noon sun, glow, fog,
    /// haze and sky go grey at the same fog brightness, and the grade is less saturated.
    pub const DARKSHORE_NOON: Self = Self {
        glow: [1.0, 0.96, 0.95],
        haze_tint: [0.8, 0.8, 0.82],
        fog_tint: [1.9, 1.9, 1.93],
        sky_tint: [1.0, 1.0, 1.04],
        saturation: 0.72,
        key_tint: [0.97, 0.97, 1.0],
        ..Self::DARKSHORE
    };
}

/// `AreaTable.dbc` zone id that takes [`ZoneLook::DARKSHORE`].
const DARKSHORE_ZONE: u32 = 148;

/// `AreaTable.dbc` zone id that takes [`ZoneLook::IRONFORGE`].
const IRONFORGE_ZONE: u32 = 1537;

/// `AreaTable.dbc` zone ids that take [`ZoneLook::COLD`]: Dun Morogh, Winterspring, Alterac Valley and the Alterac Mountains.
const COLD_ZONES: &[u32] = &[1, 618, 2597, 36];

/// `AreaTable.dbc` zone id that takes [`ZoneLook::TIRISFAL`].
const TIRISFAL_ZONE: u32 = 85;

/// `AreaTable.dbc` zone ids that take [`ZoneLook::EERIE`]: Duskwood, Silverpine Forest, Deadwind
/// Pass, the Western and Eastern Plaguelands, Swamp of Sorrows and Felwood, and Turtle WoW's
/// Gilneas and Gilneas City.
const EERIE_ZONES: &[u32] = &[10, 130, 41, 28, 139, 8, 361, 5179, 5208];

/// `AreaTable.dbc` zone ids that wear the default look but keep [`MIST_BANKS`] of the night's
/// ground banks: Wetlands, Dustwallow Marsh, Ashenvale, Teldrassil and Feralas, and Turtle WoW's
/// Moonwhisper Coast.
const MIST_ZONES: &[u32] = &[11, 15, 331, 141, 357, 5642];

/// fork: the night banks a misty zone keeps: the share every ordinary zone had before only the
/// dark zones kept theirs.
const MIST_BANKS: f32 = 0.25;

/// The zone's look at `minute` of the day: the default look takes [`ZoneLook::NOON`] across midday
/// (`sun_arc::midday_at`) and Darkshore [`ZoneLook::DARKSHORE_NOON`]; the snow zones take
/// [`ZoneLook::COLD_NIGHT`] across the night (`sun_arc::night_at`); any other zone with its own
/// look keeps it at every hour. A misty zone keeps its banks on top, at every hour.
pub fn look_for_zone_at(zone: u32, minute: f32) -> ZoneLook {
    let look = look_for_zone(zone);
    let (to, t) = if look == ZoneLook::NEUTRAL {
        (ZoneLook::NOON, super::sun_arc::midday_at(minute))
    } else if look == ZoneLook::DARKSHORE {
        (ZoneLook::DARKSHORE_NOON, super::sun_arc::midday_at(minute))
    } else if look == ZoneLook::COLD {
        (ZoneLook::COLD_NIGHT, super::sun_arc::night_at(minute))
    } else {
        return look;
    };
    let look = look.lerp(to, t);
    if MIST_ZONES.contains(&zone) {
        ZoneLook {
            night_banks: MIST_BANKS,
            ..look
        }
    } else {
        look
    }
}

pub fn look_for_zone(zone: u32) -> ZoneLook {
    if zone == TIRISFAL_ZONE {
        ZoneLook::TIRISFAL
    } else if EERIE_ZONES.contains(&zone) {
        ZoneLook::EERIE
    } else if zone == IRONFORGE_ZONE {
        ZoneLook::IRONFORGE
    } else if zone == DARKSHORE_ZONE {
        ZoneLook::DARKSHORE
    } else if COLD_ZONES.contains(&zone) {
        ZoneLook::COLD
    } else {
        ZoneLook::NEUTRAL
    }
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
static MOON_KEY_GAIN: AtomicU32 = AtomicU32::new(ONE_BITS);
static FILL: AtomicU32 = AtomicU32::new(ONE_BITS);
static EXPOSURE: AtomicU32 = AtomicU32::new(ONE_BITS);
static FOG_TINT: [AtomicU32; 3] = [const { AtomicU32::new(ONE_BITS) }; 3];
static SKY_TINT: [AtomicU32; 3] = [const { AtomicU32::new(ONE_BITS) }; 3];
static KEY_TINT: [AtomicU32; 3] = [const { AtomicU32::new(ONE_BITS) }; 3];
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

/// The zone look's per-channel multiplier on the key light's colour (the shared light's diffuse
/// row; 1 in classic, where the row is never tinted).
pub fn key_tint() -> [f32; 3] {
    [0, 1, 2].map(|i| f32::from_bits(KEY_TINT[i].load(Ordering::Relaxed)))
}

/// fork: **a moonlit night.** While the moon keys, every zone takes one night key (times its look's
/// `moon_key`) and a thinner fill instead of its day ratio: the authored night rows sit key and
/// fill close together, which under a risen moon read as a flat blue wash with no shadow in it.
/// Ramped by [`super::sun_arc::moon_key`], so it comes in with the moonrise and leaves with the
/// moonset. Twice this in every zone put Goldshire at 02:00 near dusk (the frame's brightest fifth
/// at sRGB 111 against Westfall's 56), so only the looks that need it double it.
pub const MOON_KEY: f32 = 0.85;
/// The fill's multiplier under a full moon: the shadowed side falls toward the night floor.
pub const MOON_FILL: f32 = 0.45;

/// The exact sRGB encode, the inverse of [`srgb_to_linear`].
fn linear_to_srgb(c: f32) -> f32 {
    let x = c.max(0.0);
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// The key and fill gains for the shared light's rows (1, 1 in a classic session).
pub fn key_fill() -> (f32, f32) {
    let moon = super::sun_arc::moon_key();
    (
        // The key also fades with its body's elevation (`sun_arc::key_horizon`, 1 in classic).
        lerp(
            f32::from_bits(KEY.load(Ordering::Relaxed)),
            MOON_KEY * f32::from_bits(MOON_KEY_GAIN.load(Ordering::Relaxed)),
            moon,
        ) * super::sun_arc::key_horizon(),
        f32::from_bits(FILL.load(Ordering::Relaxed)) * lerp(1.0, MOON_FILL, moon),
    )
}

/// fork: the share of the moon's key the shaft haze and banks scatter on their isotropic half
/// (`froxel_scatter.wgsl`'s `shaft_phase`); their phase lobe, where a beam reads, takes it whole as
/// it takes the sun's. The authored night diffuse rows are bright enough to light the ground
/// (Darkshore's is 0.58), and haze lit by them evenly read as a pale dusk out to the horizon.
pub const MOON_HAZE: f32 = 0.4;

/// Dev tuning (`$WOW_MODERN_MOON_HAZE`, browser `?modern_moon_haze=`): the share in place of
/// [`MOON_HAZE`].
fn moon_haze() -> f32 {
    crate::launch_env::live_var("WOW_MODERN_MOON_HAZE")
        .and_then(|v| v.trim().parse::<f32>().ok())
        .map_or(MOON_HAZE, |v| v.clamp(0.0, 1.0))
}

/// The key light's share on the air's isotropic half (1 by day, [`MOON_HAZE`] under the moon).
pub fn key_air_share() -> f32 {
    lerp(1.0, moon_haze(), super::sun_arc::moon_key())
}

/// The post's exposure for a modern frame.
pub fn exposure() -> f32 {
    f32::from_bits(EXPOSURE.load(Ordering::Relaxed))
}

/// **The auto-exposure**: it lifts a dark picture into visible brightness and does nothing else.
/// It meters the screen: the scene's log-average luminance ([`super::meter`]) as the zone look
/// exposes it, so a dark zone, a night and deep shade all lift as the eye would adapt. Below
/// [`AUTO_EXPOSURE_TARGET`] it makes up about two thirds of the shortfall, in stops; at and above
/// it, an open day, it is exactly 1. It only ever brightens, by at most [`AUTO_EXPOSURE_MAX`]: a
/// night still reads as night. Until the meter has a reading (boot, its pipeline compiling, no
/// world view) the AUTHORED rows stand in (the ambient with its night floor, and the sun or moon
/// diffuse at a typical incidence, against [`AUTO_EXPOSURE_DARK`]).
///
/// fork: 0.65 since 2026-10-12 (the director's call), from 0.5, with the threshold and the ceiling
/// below raised beside it: at the first numbers the lift was too slight to notice.
const AUTO_EXPOSURE_STRENGTH: f32 = 0.65;
/// The light rows (sRGB ambient, diffuse) below which the lift begins: a dusk, about a fifth of a
/// forest noon's key (from `[0.25]`, `[0.3]`, a deep dusk).
const AUTO_EXPOSURE_DARK: ([f32; 3], [f32; 3]) = ([0.3; 3], [0.36; 3]);
/// The ceiling on the auto-exposure's lift, ~1.2 stop (from 1.75): a moonless night under the
/// night floor sits at it.
const AUTO_EXPOSURE_MAX: f32 = 2.25;
/// The auto-exposure's time constant, in seconds — an eye adapting, not a meter snapping.
const AUTO_EXPOSURE_SECONDS: f32 = 1.5;
/// The exposed scene key (log-average linear luminance) below which the lift begins: under an
/// open daytime frame's, so a day is left alone. `$WOW_AUTO_EXPOSURE_TARGET` (browser
/// `?auto_exposure_target=`) overrides it live, for tuning.
const AUTO_EXPOSURE_TARGET: f32 = 0.06;
/// Seconds without a new meter reading before the authored rows stand in again.
const METER_STALE_SECONDS: f32 = 1.0;

/// The screen meter's last scene key (f32 bits) and how many it has published.
static SCENE_KEY: AtomicU32 = AtomicU32::new(0);
static SCENE_KEY_SEQ: AtomicU32 = AtomicU32::new(0);
/// Whether this frame wants the meter: a modern frame with the auto-exposure on.
static METER_WANTED: AtomicBool = AtomicBool::new(false);

/// The meter's reading: the scene's log-average linear luminance, before exposure.
pub(super) fn publish_scene_key(key: f32) {
    SCENE_KEY.store(key.to_bits(), Ordering::Relaxed);
    SCENE_KEY_SEQ.fetch_add(1, Ordering::Relaxed);
}

/// Whether the meter's pass should run this frame.
pub(super) fn meter_wanted() -> bool {
    METER_WANTED.load(Ordering::Relaxed)
}

/// The exposure lift a metered scene asks for under the look's own exposure: 1 at `target` and
/// above, rising toward [`AUTO_EXPOSURE_MAX`] as the picture darkens.
fn screen_lift(scene_key: f32, look_exposure: f32, target: f32) -> f32 {
    (target / (scene_key * look_exposure).max(1e-5))
        .powf(AUTO_EXPOSURE_STRENGTH)
        .clamp(1.0, AUTO_EXPOSURE_MAX)
}

fn exposure_target() -> f32 {
    crate::launch_env::live_var("WOW_AUTO_EXPOSURE_TARGET")
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(AUTO_EXPOSURE_TARGET)
}

/// The screen meter as [`update_look`] follows it: the last reading's sequence number, seconds
/// since it moved, and the debug line's clock.
#[derive(Default)]
struct MeterWatch {
    seq: u32,
    age: f32,
    seen: bool,
    log: f32,
}

/// The linear luminance a pair of sRGB light rows lights a surface with: the ambient whole, the
/// diffuse at half (a typical N·L).
fn light_key(ambient: [f32; 3], diffuse: [f32; 3]) -> f32 {
    let lum = |c: [f32; 3]| {
        0.2126 * srgb_to_linear(c[0])
            + 0.7152 * srgb_to_linear(c[1])
            + 0.0722 * srgb_to_linear(c[2])
    };
    lum(ambient) + 0.5 * lum(diffuse)
}

/// The `autoExposure` cvar: the light auto-exposure on (the default) or off. benilla-app's
/// `video::on_cvar` writes it; `$WOW_AUTO_EXPOSURE=0` boots it off for a session. A modern frame's
/// only — a classic one has no exposure to lift.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoExposure(pub bool);

impl Default for AutoExposure {
    fn default() -> Self {
        Self(crate::launch_env::var("WOW_AUTO_EXPOSURE").is_none_or(|v| v.trim() != "0"))
    }
}

/// The exposure lift this frame's light asks for: 1 at [`AUTO_EXPOSURE_DARK`] and above, rising
/// toward [`AUTO_EXPOSURE_MAX`] as the light falls.
fn auto_exposure(l: &WowLighting) -> f32 {
    let reference = light_key(AUTO_EXPOSURE_DARK.0, AUTO_EXPOSURE_DARK.1);
    let key = light_key(super::night_floor(l.ambient, l.sidn_night), l.diffuse);
    (reference / key.max(1e-4))
        .powf(AUTO_EXPOSURE_STRENGTH)
        .clamp(1.0, AUTO_EXPOSURE_MAX)
}

fn publish(look: ZoneLook, auto_exposure: f32) {
    KEY.store(look.key.to_bits(), Ordering::Relaxed);
    MOON_KEY_GAIN.store(look.moon_key.to_bits(), Ordering::Relaxed);
    FILL.store(look.fill.to_bits(), Ordering::Relaxed);
    EXPOSURE.store(
        (super::MODERN_EXPOSURE * look.exposure * auto_exposure).to_bits(),
        Ordering::Relaxed,
    );
    SILHOUETTE.store(look.silhouette.to_bits(), Ordering::Relaxed);
    for (i, v) in [look.contrast, look.saturation, look.vignette, look.toe]
        .into_iter()
        .enumerate()
    {
        GRADE[i].store(v.to_bits(), Ordering::Relaxed);
    }
    for i in 0..3 {
        FOG_TINT[i].store(look.fog_tint[i].to_bits(), Ordering::Relaxed);
        SKY_TINT[i].store(look.sky_tint[i].to_bits(), Ordering::Relaxed);
        KEY_TINT[i].store(look.key_hue(look.key_tint)[i].to_bits(), Ordering::Relaxed);
    }
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<ActiveLook>()
        .init_resource::<AutoExposure>()
        .add_data_loaders(load_zone_catalog)
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
/// fog_reach,fog_tint r,g,b,fog_falloff,sky_tint r,g,b,silhouette,fog_clear,shafts,contrast,saturation,vignette,toe,
/// key_tint r,g,b,ground_fog,night_banks,night_glow,room_reach,noon_key r,g,b,noon_mix,sea_fog,moon_glow,moon_hue,moon_key,bank_moon`. An empty or unparsable entry keeps the zone's value.
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
    for c in 0..3 {
        set(&mut look.key_tint[c], 27 + c);
    }
    set(&mut look.ground_fog, 30);
    set(&mut look.night_banks, 31);
    set(&mut look.night_glow, 32);
    set(&mut look.room_reach, 33);
    for c in 0..3 {
        set(&mut look.noon_key[c], 34 + c);
    }
    set(&mut look.noon_mix, 37);
    set(&mut look.sea_fog, 38);
    set(&mut look.moon_glow, 39);
    set(&mut look.moon_hue, 40);
    set(&mut look.moon_key, 41);
    set(&mut look.bank_moon, 42);
    look
}

fn pinned_zone() -> Option<u32> {
    static CELL: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::launch_env::var("WOW_MODERN_ZONE").and_then(|v| v.trim().parse().ok())
    })
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
    clock: Option<Res<GameClock>>,
    lighting: Option<Res<WowLighting>>,
    auto: Option<Res<AutoExposure>>,
    mut look: ResMut<ActiveLook>,
    mut probe: Local<(f32, Option<u32>)>,
    mut adapted: Local<Option<f32>>,
    mut watch: Local<MeterWatch>,
) {
    let auto_on = auto.is_none_or(|a| a.0);
    METER_WANTED.store(active.is_modern() && auto_on, Ordering::Relaxed);
    if !active.is_modern() {
        if look.0 != ZoneLook::NEUTRAL {
            look.0 = ZoneLook::NEUTRAL;
        }
        *adapted = None;
        *watch = MeterWatch::default();
        publish(ZoneLook::NEUTRAL, 1.0);
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
    let minute = clock.map_or(0.0, |c| c.minute as f32);
    let target = apply_override(
        look_for_zone_at(zone.unwrap_or(0), minute),
        &look_override(),
    );
    let next = look.0.lerp(target, (time.delta_secs() / 2.0).min(1.0));
    let next = if next.close_to(target) { target } else { next };
    if next != look.0 {
        look.0 = next;
    }
    // The first modern frame takes the lift whole — entering the world at night should not open on
    // a second and a half of the eye adjusting.
    let authored = lighting.as_deref().map_or(1.0, auto_exposure);
    // The meter's reading while it is fresh; the authored rows until then.
    let seq = SCENE_KEY_SEQ.load(Ordering::Relaxed);
    if seq != watch.seq {
        watch.seq = seq;
        watch.age = 0.0;
        watch.seen = true;
    } else {
        watch.age += time.delta_secs();
    }
    let scene_key = (watch.seen && watch.age < METER_STALE_SECONDS)
        .then(|| f32::from_bits(SCENE_KEY.load(Ordering::Relaxed)));
    let lift = match (auto_on, scene_key) {
        (false, _) => 1.0,
        (true, Some(key)) => screen_lift(key, look.0.exposure, exposure_target()),
        (true, None) => authored,
    };
    let eye = adapted.get_or_insert(lift);
    *eye += (lift - *eye) * (1.0 - (-time.delta_secs() / AUTO_EXPOSURE_SECONDS).exp());
    // `WOW_LOOK_DEBUG=meter` (browser `?look_debug=meter`): the meter and the lift, every 2 s.
    watch.log += time.delta_secs();
    if watch.log >= 2.0 {
        watch.log = 0.0;
        if crate::launch_env::live_var("WOW_LOOK_DEBUG").is_some_and(|v| v.trim() == "meter") {
            info!(
                "LOOK_METER zone={:?} scene_key={:?} look_exposure={:.2} authored_lift={authored:.2} lift={lift:.2} eye={:.2}",
                zone, scene_key, look.0.exposure, *eye
            );
        }
    }
    publish(look.0, *eye);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The screen lift leaves an open day alone, rises as the exposed picture darkens, counts the
    /// look's own exposure, and never passes its ceiling.
    #[test]
    fn the_screen_lift_only_brightens_a_dark_picture() {
        let t = AUTO_EXPOSURE_TARGET;
        assert_eq!(screen_lift(t, 1.0, t), 1.0);
        assert_eq!(screen_lift(4.0 * t, 1.0, t), 1.0);
        let dusk = screen_lift(0.5 * t, 1.0, t);
        let dark = screen_lift(0.25 * t, 1.0, t);
        assert!(dusk > 1.0 && dark > dusk, "{dusk} {dark}");
        // A look that exposes the same scene lower asks for more.
        assert!(screen_lift(t, ZoneLook::EERIE.exposure, t) > 1.0);
        assert_eq!(screen_lift(0.0, 1.0, t), AUTO_EXPOSURE_MAX);
    }
    use crate::modern::srgb_to_linear;

    /// Lit-over-shadowed luminance for a typical forest noon (ambient 0.42, diffuse 0.95, gamma): the
    /// lanes sum in gamma, clamp, and the frame decodes.
    fn key_to_fill(look: ZoneLook) -> f32 {
        let fill = 0.42 * look.fill;
        let lit = (fill + 0.95 * look.key).min(1.0);
        srgb_to_linear(lit) / srgb_to_linear(fill)
    }

    /// Amendment A3's target: 6–10:1 under a canopy. The dark zones' look runs deeper, up to 25:1:
    /// its fill was taken down against the Brill reference shot, whose shade is near black.
    #[test]
    fn every_row_keeps_the_key_to_fill_ratio_in_band() {
        for (zone, look, band) in [
            (0, ZoneLook::NEUTRAL, 6.0..=10.0),
            // The noon row's shade is deeper, as the Goldshire reference shot's is.
            (0, ZoneLook::NOON, 6.0..=18.0),
            (10, ZoneLook::EERIE, 6.0..=25.0),
            (1, ZoneLook::COLD, 6.0..=10.0),
            (1, ZoneLook::COLD_NIGHT, 6.0..=10.0),
            (1537, ZoneLook::IRONFORGE, 6.0..=10.0),
            (148, ZoneLook::DARKSHORE, 6.0..=10.0),
            (148, ZoneLook::DARKSHORE_NOON, 6.0..=10.0),
        ] {
            let r = key_to_fill(look);
            assert!(band.contains(&r), "zone {zone}: {r:.2}:1");
        }
    }

    /// Elwynn's authored noon row, `[255, 136, 0]`: kept by the default look and the zones' own,
    /// golden at noon and no brighter (linear luminance).
    #[test]
    fn the_noon_key_turns_the_sunset_orange_golden_at_the_same_brightness() {
        let row = [1.0, 136.0 / 255.0, 0.0];
        for look in [
            ZoneLook::NEUTRAL,
            ZoneLook::EERIE,
            ZoneLook::COLD,
            ZoneLook::IRONFORGE,
        ] {
            assert_eq!(look.noon_key_row(row), row);
        }
        let noon = ZoneLook::NOON.noon_key_row(row);
        let lum = |c: [f32; 3]| {
            let l = c.map(srgb_to_linear);
            0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]
        };
        assert!((lum(noon) - lum(row)).abs() < 1e-4, "{noon:?}");
        assert!(
            noon[2] > 0.1 && noon[1] / noon[0] > row[1] / row[0],
            "{noon:?}"
        );
    }

    /// The default look takes the noon row across midday only; a zone with its own look keeps it.
    #[test]
    fn only_the_default_look_ramps_into_noon() {
        assert_eq!(look_for_zone_at(999_999, 0.0), ZoneLook::NEUTRAL);
        assert_eq!(look_for_zone_at(999_999, 720.0), ZoneLook::NOON);
        assert_eq!(look_for_zone_at(1, 720.0), ZoneLook::COLD);
        assert_eq!(look_for_zone_at(10, 720.0), ZoneLook::EERIE);
    }

    /// Every snow zone's night air is clearer than its day's, and the day's look comes back by
    /// dawn.
    #[test]
    fn the_snow_zones_nights_are_clear() {
        for zone in COLD_ZONES {
            let night = look_for_zone_at(*zone, 120.0);
            assert_eq!(night, ZoneLook::COLD_NIGHT, "zone {zone}");
            assert!(night.moon_key > 1.0, "zone {zone}");
            assert!(night.haze < ZoneLook::COLD.haze && night.fog_reach > ZoneLook::COLD.fog_reach);
            assert_eq!(
                look_for_zone_at(*zone, 720.0),
                ZoneLook::COLD,
                "zone {zone}"
            );
        }
        let dusk = look_for_zone_at(1, super::super::sun_arc::MOONRISE_MINUTE + 30.0);
        assert!(dusk.haze < ZoneLook::COLD.haze && dusk.haze > ZoneLook::COLD_NIGHT.haze);
    }

    #[test]
    fn an_unlisted_zone_is_neutral_and_looks_cross_fade() {
        assert_eq!(look_for_zone(999_999), ZoneLook::NEUTRAL);
        let hazier = ZoneLook {
            haze: 0.004,
            ..ZoneLook::NEUTRAL
        };
        let half = ZoneLook::NEUTRAL.lerp(hazier, 0.5);
        assert!((half.haze - (ZoneLook::NEUTRAL.haze + hazier.haze) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn the_dark_zones_days_are_darker_than_elwynns() {
        for zone in EERIE_ZONES.iter().chain([&TIRISFAL_ZONE]) {
            let d = look_for_zone(*zone);
            assert!(d.key < ZoneLook::NEUTRAL.key && d.fill < ZoneLook::NEUTRAL.fill);
            assert!(d.exposure < ZoneLook::NEUTRAL.exposure);
        }
    }

    /// The snow zones' key, fog and sky lean blue.
    #[test]
    fn the_snow_zones_are_cold() {
        for zone in COLD_ZONES {
            let c = look_for_zone(*zone);
            assert_eq!(c, ZoneLook::COLD, "zone {zone}");
            for tint in [c.key_tint, c.fog_tint, c.sky_tint, c.haze_tint] {
                assert!(
                    tint[2] > 1.0 && tint[0] < tint[1] && tint[1] < tint[2],
                    "zone {zone}"
                );
            }
        }
    }

    /// An ordinary zone's night is clearer than a dark zone's: no banks and a dim air glow.
    #[test]
    fn only_the_dark_zones_nights_are_eerie() {
        for zone in EERIE_ZONES.iter().chain([&TIRISFAL_ZONE]) {
            let d = look_for_zone(*zone);
            assert_eq!((d.night_banks, d.night_glow), (1.0, 1.0), "zone {zone}");
        }
        for zone in [12, 33, 14, 1637, 17, 215] {
            let n = look_for_zone(zone);
            assert_eq!(n.night_banks, 0.0, "zone {zone}");
            assert!(n.night_glow < ZoneLook::EERIE.night_glow, "zone {zone}");
        }
    }

    /// Only the dark looks (eerie, Tirisfal, Darkshore) lay ground banks at night; every other
    /// look lies clear, by day as by night.
    #[test]
    fn only_the_dark_looks_carry_night_banks() {
        for dark in [
            ZoneLook::EERIE,
            ZoneLook::TIRISFAL,
            ZoneLook::DARKSHORE,
            ZoneLook::DARKSHORE_NOON,
        ] {
            assert!(dark.night_banks > 0.0);
        }
        for clear in [
            ZoneLook::NEUTRAL,
            ZoneLook::NOON,
            ZoneLook::COLD,
            ZoneLook::COLD_NIGHT,
            ZoneLook::IRONFORGE,
        ] {
            assert_eq!((clear.night_banks, clear.ground_fog), (0.0, 0.0));
        }
        assert_eq!(look_for_zone(DARKSHORE_ZONE), ZoneLook::DARKSHORE);
    }

    /// Tirisfal is the eerie look, its sun, shafts, fog and sky leaning further green.
    #[test]
    fn tirisfal_is_greener_than_the_other_dark_zones() {
        let (t, e) = (look_for_zone(TIRISFAL_ZONE), ZoneLook::EERIE);
        let green = |c: [f32; 3]| c[1] / c[0].max(c[2]);
        for (a, b) in [
            (t.glow, e.glow),
            (t.haze_tint, e.haze_tint),
            (t.fog_tint, e.fog_tint),
            (t.sky_tint, e.sky_tint),
            (t.key_tint, e.key_tint),
        ] {
            assert!(green(a) > green(b), "{a:?} against {b:?}");
        }
        assert_eq!((t.key, t.fill, t.exposure), (e.key, e.fill, e.exposure));
    }

    /// Darkshore wears its own look outside midday and its grey noon row across it.
    #[test]
    fn darkshore_greys_across_midday() {
        for minute in [0.0, 390.0, 1020.0, 1080.0, 1260.0] {
            assert_eq!(
                look_for_zone_at(DARKSHORE_ZONE, minute),
                ZoneLook::DARKSHORE,
                "{minute}"
            );
        }
        for minute in [660.0, 720.0, 870.0] {
            assert_eq!(
                look_for_zone_at(DARKSHORE_ZONE, minute),
                ZoneLook::DARKSHORE_NOON,
                "{minute}"
            );
        }
        // Every warm hue is nearer grey at noon: a smaller spread between its channels.
        let spread = |c: [f32; 3]| c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2]);
        let (d, n) = (ZoneLook::DARKSHORE, ZoneLook::DARKSHORE_NOON);
        for (a, b) in [
            (n.glow, d.glow),
            (n.haze_tint, d.haze_tint),
            (n.fog_tint, d.fog_tint),
            (n.sky_tint, d.sky_tint),
            (n.key_tint, d.key_tint),
        ] {
            assert!(spread(a) < spread(b), "{a:?} against {b:?}");
        }
        assert!(n.saturation < d.saturation);
    }

    /// Darkshore's sea fog is its own: no other look lays one, and its grey authored rows are
    /// tinted warm, toward the reference's mauve.
    #[test]
    fn only_darkshore_lays_a_sea_fog() {
        let d = look_for_zone(DARKSHORE_ZONE);
        assert!(d.sea_fog > 0.0);
        assert!(d.fog_tint[0] > d.fog_tint[1] && d.glow[0] > d.glow[2]);
        for look in [
            ZoneLook::NEUTRAL,
            ZoneLook::NOON,
            ZoneLook::EERIE,
            ZoneLook::COLD,
            ZoneLook::IRONFORGE,
        ] {
            assert_eq!(look.sea_fog, 0.0);
        }
    }

    /// The misty zones wear the default look and its noon ramp, and keep night banks; an ordinary
    /// zone keeps none.
    #[test]
    fn the_misty_zones_keep_night_banks() {
        for zone in MIST_ZONES {
            assert_eq!(look_for_zone(*zone), ZoneLook::NEUTRAL, "zone {zone}");
            let noon = look_for_zone_at(*zone, 720.0);
            assert_eq!(
                ZoneLook {
                    night_banks: ZoneLook::NOON.night_banks,
                    ..noon
                },
                ZoneLook::NOON,
                "zone {zone}"
            );
            assert!(
                look_for_zone_at(*zone, 120.0).night_banks > 0.0,
                "zone {zone}"
            );
        }
        assert_eq!(look_for_zone_at(12, 120.0).night_banks, 0.0);
    }

    /// Every zone the user named reads Elwynn's look (Elwynn 12, Stranglethorn 33, Durotar 14,
    /// Orgrimmar 1637, the Barrens 17, Mulgore 215).
    #[test]
    fn every_zone_wears_elwynns_look() {
        for zone in [12, 33, 14, 1637, 17, 215] {
            assert_eq!(look_for_zone(zone), ZoneLook::NEUTRAL, "zone {zone}");
        }
    }
}
