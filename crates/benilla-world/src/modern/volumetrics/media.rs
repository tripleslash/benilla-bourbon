//! **The fog media** (`FORK.md` P4): the zone's authored fog turned into a
//! participating medium.
//!
//! - **Extinction** is a long-range height fog: `σ(y) = density · e^(−(y − ref) / falloff)`. The
//!   density is calibrated on the zone's fog end BEFORE the farclip clamp ([`FAR_TAU`] of optical
//!   depth over it at the reference height), so the horizon is not a wall at the view distance: the
//!   ground fades into the haze while high ground — the distant (WDL) ranges — rises above it. The
//!   air is the same at every distance from the camera: walking toward a bank never clears it.
//! - **Height falloff** of [`HEIGHT_FALLOFF`] above a reference height that follows the regional
//!   ground (the WDL heights around the camera), not the camera, so flying up does not drag the
//!   fog layer along.
//! - **Single scattering.** The air scatters [`ALBEDO`] of the light that reaches it, in the fog
//!   colour's hue (held subtle), and nothing else: the sky's radiance from above and the ground's
//!   from below, as far as the sky reaches the air, and the key the surfaces take, through the phase
//!   function and the key's shadow. Deep air seen level converges on the sky at the horizon. It never glows of its own and never sends back more than arrives, so a fogged surface
//!   always lands between its own colour and the air's, and the air under a canopy or at night is
//!   as dark as the light there.
//!
//! The variations the plan names ride the same solve: a storm thickens the air, underwater drops
//! the height falloff and the haze, and an interior (the WMO crossfade `t`) flattens the falloff
//! toward a uniform room haze.

/// Optical depth at the reference height over the zone's (unclamped) fog end: transmittance there is
/// `e^−FAR_TAU`.
pub const FAR_TAU: f32 = 1.6;
/// Underwater, and inside a room, the authored (clamped) end is a real wall: this much optical depth
/// over it.
pub const WALL_TAU: f32 = 3.0;
/// The single-scattering albedo of the air's droplets and dust at the fog colour's brightest
/// channel: haze and fog scatter nearly all the light they intercept.
pub const ALBEDO: f32 = 0.9;
/// Density drops by `e` every this many yards above the reference height.
pub const HEIGHT_FALLOFF: f32 = 70.0;
/// The dual-lobe Henyey-Greenstein pair.
pub const FORWARD_G: f32 = 0.75;
pub const BACKWARD_G: f32 = -0.2;
/// The shortest fog span the solve accepts (yd); a degenerate DBC pair must not read as a wall.
const MIN_SPAN: f32 = 8.0;
/// The fog colour's saturation `(max - min) / max` never exceeds this (see [`subtle`]).
pub const MAX_FOG_SATURATION: f32 = 0.3;
/// Nor its brightest channel (gamma) this.
pub const MAX_FOG_LEVEL: f32 = 0.78;
/// fork: **a room's air is never a wall.** The most extinction (1/yd) a building's MFOG is solved to:
/// [`WALL_TAU`] over a cavern's short record put the Cleft of Shadows in a purple wall inside 50 yd.
/// At this cap 50 yd keeps 86 % of the scene and 100 yd 74 %. Underwater keeps its wall.
pub const ROOM_MAX_DENSITY: f32 = 0.003;
/// The haze a full storm adds (1/yd): a uniform humid veil. 0.006 alone took 45 % of the scene at
/// 100 yd and 0.002 a third at 200; this keeps 79 % at 200.
pub const STORM_HAZE: f32 = 0.0012;

/// What one frame's media is solved from. Colours are the authored gamma values.
#[derive(Clone, Copy, Debug)]
pub struct MediaInputs {
    pub fog_color: [f32; 3],
    /// The pushed (farclip-clamped) pair: the room's and the water's wall.
    pub fog_start: f32,
    pub fog_end: f32,
    /// The zone's fog end before the farclip clamp (yd): the open air's calibration.
    pub zone_fog_end: f32,
    /// The light that reaches the air (linear): the mean radiance arriving from every direction
    /// (`volumetrics::air_ambient`), and the key light the surfaces take, at full incidence.
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    /// The weather storm blend `bcc` (0 clear, 1 full storm).
    pub storm: f32,
    pub underwater: bool,
    /// The camera-in-WMO crossfade weight.
    pub interior: f32,
    /// The zone look's shaft haze extinction (1/yd).
    pub haze: f32,
    /// The zone look's forward-lobe share.
    pub forward: f32,
    /// The zone look's fog reach (>1 = clearer), colour multiplier and height falloff (yd).
    pub fog_reach: f32,
    pub fog_tint: [f32; 3],
    pub fog_falloff: f32,
    /// fork: the zone look's room-fog reach (>1 = clearer): divides a building's MFOG extinction.
    pub room_reach: f32,
    /// fork: the zone look's `added_air`: the share of the storm's humid veil a zone takes on top
    /// of its own haze.
    pub added_air: f32,
}

/// The solved medium, in the units `froxelfog.wesl` reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Media {
    /// Fog extinction at the reference height (1/yd along the ray).
    pub density: f32,
    /// `1 / height falloff`; 0 = uniform.
    pub inv_height: f32,
    /// Shaft haze extinction (1/yd).
    pub haze: f32,
    /// Density noise amount (0 = smooth).
    pub noise: f32,
    /// The single-scattering albedo per channel (at most [`ALBEDO`]).
    pub albedo: [f32; 3],
    /// Linear in-scatter radiance per unit of scattering from the ambient term: albedo x ambient.
    pub ambient_in: [f32; 3],
    /// Linear in-scatter radiance from the key light, before the phase function and its shadow:
    /// albedo x key.
    pub key_in: [f32; 3],
    pub forward_share: f32,
    pub g_forward: f32,
    pub g_backward: f32,
}

/// fork: **the haze is always subtle.** The tinted fog colour (gamma) is held to at most
/// [`MAX_FOG_SATURATION`] saturation and [`MAX_FOG_LEVEL`] on its brightest channel, hue kept: the
/// Forever tint on Westfall's and Stranglethorn's authored fog made a saturated cyan wall over half
/// the screen. Every zone's air passes through here.
pub(crate) fn subtle(c: [f32; 3]) -> [f32; 3] {
    let c = c.map(|v| v.max(0.0));
    let hi = c[0].max(c[1]).max(c[2]);
    if hi <= 1e-5 {
        return c;
    }
    let lo = c[0].min(c[1]).min(c[2]);
    let keep = (MAX_FOG_SATURATION * hi / (hi - lo).max(1e-5)).min(1.0);
    let level = (MAX_FOG_LEVEL / hi).min(1.0);
    c.map(|v| (hi - (hi - v) * keep) * level)
}

pub fn solve(i: &MediaInputs) -> Media {
    let decode = |c: [f32; 3]| c.map(crate::modern::srgb_to_linear);
    // The zone's fog tint is calibrated on the zone's own `Light.dbc` fog, so it fades out with the
    // room's: Durotar's (green x2.6, the amber its grey fog wants) on Orgrimmar's teal MFOG record
    // was most of what made the gate hall green.
    let fog = decode(subtle([0, 1, 2].map(|c| {
        i.fog_color[c] * (1.0 + (i.fog_tint[c] - 1.0) * (1.0 - i.interior.clamp(0.0, 1.0)))
    })));
    let ambient = i.ambient.map(|c| c.max(0.0));
    let key = i.diffuse.map(|c| c.max(0.0));
    let storm = i.storm.clamp(0.0, 1.0);
    let interior = i.interior.clamp(0.0, 1.0);

    // The open air calibrates on the zone's own end; a room (and the water) on the pushed wall.
    let wall_span = (i.fog_end - i.fog_start.max(0.0)).max(MIN_SPAN);
    let open_end = i.zone_fog_end.max(i.fog_end).max(MIN_SPAN);
    let open_density = FAR_TAU / (open_end * i.fog_reach.max(0.05));
    let wall_density = if i.underwater {
        WALL_TAU / wall_span
    } else {
        (WALL_TAU / wall_span).min(ROOM_MAX_DENSITY) / i.room_reach.max(0.05)
    };
    let walled = if i.underwater { 1.0 } else { interior };
    let density = open_density + (wall_density - open_density) * walled;

    // The fog colour is the medium's hue, never its brightness: its brightest channel scatters
    // `ALBEDO` of the light, the others in proportion.
    let peak = fog[0].max(fog[1]).max(fog[2]);
    let albedo = if peak > 1e-6 {
        fog.map(|c| ALBEDO * c / peak)
    } else {
        [ALBEDO; 3]
    };
    let ambient_in = [0, 1, 2].map(|c| albedo[c] * ambient[c]);
    let key_in = [0, 1, 2].map(|c| albedo[c] * key[c]);

    if i.underwater {
        return Media {
            density,
            inv_height: 0.0,
            haze: 0.0,
            noise: 0.1,
            albedo,
            ambient_in,
            key_in,
            forward_share: 0.55,
            g_forward: 0.5,
            g_backward: BACKWARD_G,
        };
    }
    Media {
        density,
        inv_height: (1.0 - interior) / i.fog_falloff.max(1.0),
        haze: i.haze.max(0.0) + STORM_HAZE * storm * i.added_air.clamp(0.0, 1.0),
        // Faint: drifting density wisps over the ground read as cloud shadows sliding past the player.
        // A storm barely raises them: it thickens the air, not the ground.
        noise: 0.15 + 0.1 * storm,
        albedo,
        ambient_in,
        key_in,
        forward_share: i.forward.clamp(0.0, 1.0) * (1.0 - 0.5 * storm),
        g_forward: FORWARD_G,
        g_backward: BACKWARD_G,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elwynn_noon() -> MediaInputs {
        MediaInputs {
            fog_color: [0.55, 0.68, 0.80],
            fog_start: 87.5,
            fog_end: 350.0,
            zone_fog_end: 500.0,
            ambient: [0.40, 0.40, 0.45],
            diffuse: [0.90, 0.85, 0.70],
            storm: 0.0,
            underwater: false,
            interior: 0.0,
            haze: 0.0015,
            forward: 0.7,
            fog_reach: 1.0,
            fog_tint: [1.0; 3],
            fog_falloff: HEIGHT_FALLOFF,
            room_reach: 1.0,
            added_air: 1.0,
        }
    }

    /// The air sends back at most `ALBEDO` of what reaches it, in every channel: no fog colour,
    /// tint or dark light row makes it glow brighter than a white surface under the same light.
    #[test]
    fn the_air_never_scatters_more_than_reaches_it() {
        for (fog, amb, key) in [
            ([0.55, 0.68, 0.80], [0.40, 0.40, 0.45], [0.90, 0.85, 0.70]),
            // A night: a bright authored fog over near-black light rows.
            ([0.6, 0.6, 0.6], [0.02, 0.03, 0.05], [0.05, 0.05, 0.08]),
            // Ashenvale's teal over a dark blue ambient.
            ([0.012, 0.286, 0.42], [0.28, 0.39, 0.65], [0.75, 0.96, 1.0]),
        ] {
            let m = solve(&MediaInputs {
                fog_color: fog,
                ambient: amb,
                diffuse: key,
                ..elwynn_noon()
            });
            for c in 0..3 {
                let (a, k) = (amb[c], key[c]);
                assert!(m.albedo[c] <= ALBEDO + 1e-6, "{c}: {:?}", m.albedo);
                assert!(m.ambient_in[c] <= ALBEDO * a + 1e-6);
                assert!(m.key_in[c] <= ALBEDO * k + 1e-6);
            }
        }
    }

    /// The fog colour sets the air's hue, never its brightness: however dark or light the authored
    /// colour, its brightest channel scatters `ALBEDO` of the light.
    #[test]
    fn the_fog_colour_is_the_hue_not_the_brightness() {
        let bright = solve(&elwynn_noon());
        let dim = solve(&MediaInputs {
            fog_color: [0.55 * 0.5, 0.68 * 0.5, 0.80 * 0.5],
            ..elwynn_noon()
        });
        for m in [bright, dim] {
            let peak = m.albedo[0].max(m.albedo[1]).max(m.albedo[2]);
            assert!((peak - ALBEDO).abs() < 1e-6, "{:?}", m.albedo);
        }
        assert!((bright.ambient_in[2] - dim.ambient_in[2]).abs() < 1e-6);
        // Bluer fog, bluer air.
        assert!(bright.albedo[2] > bright.albedo[0]);
        assert!((bright.albedo[2] - ALBEDO).abs() < 1e-6);
    }

    #[test]
    fn darkness_scatters_nothing() {
        let mut i = elwynn_noon();
        i.ambient = [0.0, 0.0, 0.0];
        i.diffuse = [0.0, 0.0, 0.0];
        let m = solve(&i);
        assert!(m
            .ambient_in
            .iter()
            .chain(&m.key_in)
            .all(|v| v.is_finite() && *v == 0.0));
    }

    /// The open air calibrates on the zone's end, not the farclip wall: a lowered view distance does
    /// not thicken the fog, and the zone's end keeps `e^−FAR_TAU` of the scene at the reference height.
    #[test]
    fn the_horizon_is_not_a_wall_at_the_view_distance() {
        let i = elwynn_noon();
        let mut short = i;
        short.fog_start = 44.25;
        short.fog_end = 177.0;
        let (a, b) = (solve(&i), solve(&short));
        assert!((a.density - b.density).abs() < 1e-6);
        assert!((a.density * 500.0 - FAR_TAU).abs() < 1e-4);
        assert!((-a.density * 500.0).exp() > 0.15);
    }

    #[test]
    fn a_tinted_fog_is_never_a_saturated_wall() {
        let c = subtle([0.35 * 1.9, 0.6 * 2.5, 0.7 * 2.25]);
        let (hi, lo) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        assert!(hi <= MAX_FOG_LEVEL + 1e-5);
        assert!((hi - lo) / hi <= MAX_FOG_SATURATION + 1e-4);
        assert!(c[2] >= c[1] && c[1] >= c[0], "hue kept");
        assert_eq!(subtle([0.5, 0.52, 0.55]), [0.5, 0.52, 0.55]);
    }

    #[test]
    fn a_storm_thickens_the_air() {
        let mut i = elwynn_noon();
        i.fog_start = -60.0;
        i.fog_end = 278.0;
        i.zone_fog_end = 278.0;
        i.storm = 1.0;
        let m = solve(&i);
        assert!(m.haze > elwynn_noon().haze);
        assert!(m.density > solve(&elwynn_noon()).density);
        assert!(m.forward_share < 0.5);
    }

    #[test]
    fn underwater_is_uniform_hazeless_and_walled() {
        let mut i = elwynn_noon();
        i.underwater = true;
        i.fog_start = 0.0;
        i.fog_end = 60.0;
        let m = solve(&i);
        assert_eq!(m.inv_height, 0.0);
        assert_eq!(m.haze, 0.0);
        assert!((m.density - WALL_TAU / 60.0).abs() < 1e-5);
    }

    #[test]
    fn an_interior_flattens_the_height_falloff_and_takes_the_room_wall() {
        let mut i = elwynn_noon();
        i.interior = 1.0;
        let m = solve(&i);
        assert_eq!(m.inv_height, 0.0);
        assert!((m.density - ROOM_MAX_DENSITY).abs() < 1e-6);
    }

    /// A cavern's short MFOG record (the Cleft of Shadows) must not solve to a wall: the room keeps
    /// most of the scene at 50 yd, while underwater keeps its wall on the same record.
    #[test]
    fn a_rooms_air_is_never_a_wall() {
        let mut i = elwynn_noon();
        i.interior = 1.0;
        i.fog_start = 0.0;
        i.fog_end = 60.0;
        let room = solve(&i);
        assert!((-room.density * 50.0).exp() > 0.85, "{}", room.density);
        i.underwater = true;
        assert!((solve(&i).density - WALL_TAU / 60.0).abs() < 1e-5);
    }

    /// The zone's fog tint belongs to the zone's own `Light.dbc` fog, not to a building's authored
    /// MFOG: Orgrimmar's gate hall wore Durotar's tint (green x2.6) over the WMO's teal record, and
    /// the hall went green. The tint fades out with the room's fog and is untouched in the open air.
    #[test]
    fn a_rooms_own_fog_does_not_take_the_zones_tint() {
        let tint = [1.7, 2.6, 1.8];
        let teal = [15.0 / 255.0, 52.0 / 255.0, 53.0 / 255.0];
        let hue = |interior: f32| {
            let mut i = elwynn_noon();
            i.fog_color = teal;
            i.fog_tint = tint;
            i.interior = interior;
            solve(&i).albedo
        };
        let room = hue(1.0);
        let target = subtle(teal).map(crate::modern::srgb_to_linear);
        let peak = target[0].max(target[1]).max(target[2]);
        for c in 0..3 {
            assert!((room[c] - ALBEDO * target[c] / peak).abs() < 1e-4, "{c}");
        }
        let open = hue(0.0);
        assert!(
            open[1] / open[2] > room[1] / room[2],
            "{open:?} vs {room:?}"
        );
    }
}
