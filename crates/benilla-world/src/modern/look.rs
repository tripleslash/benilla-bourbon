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
}

impl ZoneLook {
    /// Every zone without a row: the P3 balance, a light haze.
    pub const NEUTRAL: Self = Self {
        key: 1.0,
        fill: 1.0,
        exposure: 1.0,
        haze: 0.0018,
        forward: 0.7,
        key_scatter: 0.6,
        glow: [1.0, 1.0, 1.0],
        haze_tint: [1.0, 1.0, 1.0],
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
            && (0..3).all(|i| d(self.glow[i], other.glow[i]) && d(self.haze_tint[i], other.haze_tint[i]))
    }
}

/// The look table, by `AreaTable.dbc` zone id.
const ZONES: &[(u32, ZoneLook)] = &[
    // Teldrassil: dense purple canopy haze, a strong forward glow through the openings.
    (141, ZoneLook { key: 1.1, fill: 0.92, exposure: 1.08, haze: 0.0030, forward: 0.78, key_scatter: 0.85, ..ZoneLook::NEUTRAL }),
    // Darnassus: Teldrassil's light, a little clearer.
    (1657, ZoneLook { key: 1.1, fill: 0.95, exposure: 1.0, haze: 0.0026, forward: 0.75, key_scatter: 0.75, ..ZoneLook::NEUTRAL }),
    // Ashenvale: cyan-teal volumetric haze and deep contrast under the trees.
    (331, ZoneLook { key: 1.15, fill: 0.88, exposure: 1.0, haze: 0.0039, forward: 0.68, key_scatter: 0.75, ..ZoneLook::NEUTRAL }),
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
    }),
    // Elwynn Forest: treeline shafts at dusk.
    (12, ZoneLook { key: 1.1, fill: 0.9, exposure: 1.0, haze: 0.0024, forward: 0.72, key_scatter: 0.7, ..ZoneLook::NEUTRAL }),
    // Duskwood: gloom — a weak key, a darker exposure.
    (10, ZoneLook { key: 1.0, fill: 0.95, exposure: 0.95, haze: 0.0030, forward: 0.6, key_scatter: 0.45, ..ZoneLook::NEUTRAL }),
    // Felwood: a sickly canopy.
    (361, ZoneLook { key: 1.1, fill: 0.88, exposure: 1.0, haze: 0.0036, forward: 0.65, key_scatter: 0.6, ..ZoneLook::NEUTRAL }),
    // Stormwind City, Ironforge: cities keep the haze thin.
    (1519, ZoneLook { key: 1.0, fill: 1.0, exposure: 1.0, haze: 0.0012, forward: 0.7, key_scatter: 0.5, ..ZoneLook::NEUTRAL }),
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
    let target = zone.map_or(ZoneLook::NEUTRAL, look_for_zone);
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
