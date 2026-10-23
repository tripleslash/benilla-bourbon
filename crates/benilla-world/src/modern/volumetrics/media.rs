//! **The fog media** (`FORK.md` P4): the zone's authored fog turned into a
//! participating medium.
//!
//! - **Extinction** is a long-range height fog: `σ(y) = density · e^(−(y − ref) / falloff)`, clear
//!   before the authored start. The density is calibrated on the zone's fog end BEFORE the farclip
//!   clamp ([`FAR_TAU`] of optical depth over it at the reference height), so the horizon is not a
//!   wall at the view distance: the ground fades into a bright haze while high ground — the distant
//!   (WDL) ranges — rises above it as a silhouette, the way the modern client draws Durotar with
//!   Hyjal behind it. (It used to follow the classic linear ramp in planar depth, which put 95 % of
//!   the scene into the fog colour at `min(zone end, farclip)`: nothing beyond 350 yd showed at all.)
//! - **Height falloff** of [`HEIGHT_FALLOFF`] above a reference height that follows the regional
//!   ground (the WDL heights around the camera), not the camera, so flying up does not drag the
//!   fog layer along.
//! - **Albedo** solved per channel so that the in-scatter of an ambient-plus-key medium at infinity
//!   equals the DBC fog colour, decoded to linear — the colour the sky dome paints at its horizon, so
//!   a silhouette's foot and the sky behind it meet. The zone's hue survives because the fog colour
//!   is the target, not an input to a physical sky: Teldrassil stays purple, Darkshore pink. The
//!   ambient share keeps at least [`SHADED_SHARE`] of it, so fog the key light cannot reach (under a
//!   canopy) is not several times darker than the colour the zone authored.
//!
//! The variations the plan names ride the same solve: a storm thickens the haze, dims the key and
//! draws the clear range in to [`STORM_CLEAR`], underwater drops the height falloff and the shaft
//! haze, and an interior (the WMO crossfade `t`) flattens the falloff toward a uniform room haze.

/// Optical depth at the reference height over the zone's (unclamped) fog span: transmittance there is
/// `e^−FAR_TAU`.
pub const FAR_TAU: f32 = 1.6;
/// Underwater, and inside a room, the authored (clamped) end is a real wall: this much optical depth
/// over it.
pub const WALL_TAU: f32 = 3.0;
/// The share of the fog colour the ambient term alone scatters. With a key-heavy albedo (a bright noon
/// sun over a dark ambient row) fog in the key's shadow kept a fifth of the authored colour or less.
pub const SHADED_SHARE: f32 = 0.7;
/// Density drops by `e` every this many yards above the reference height.
pub const HEIGHT_FALLOFF: f32 = 70.0;
/// The phase value the albedo solve assumes for the key light — roughly what a horizontal view ray
/// sees with the dual lobe below. Toward the sun the fog reads warmer and brighter, away from it closer
/// to the zone's own hue (amendment A2).
pub const REFERENCE_PHASE: f32 = 0.5;
/// The dual-lobe Henyey-Greenstein pair.
pub const FORWARD_G: f32 = 0.75;
pub const BACKWARD_G: f32 = -0.2;
/// The shortest fog span the solve accepts (yd); a degenerate DBC pair must not read as a wall.
const MIN_SPAN: f32 = 8.0;
/// The near-clear range never reaches past this share of the span, however far the zone's start is.
const MAX_CLEAR_SHARE: f32 = 0.35;
/// Albedo is a calibration here, not a physical ratio, but a black ambient channel must not explode.
const MAX_ALBEDO: f32 = 16.0;
/// The fog colour's saturation `(max - min) / max` never exceeds this (see [`subtle`]).
pub const MAX_FOG_SATURATION: f32 = 0.3;
/// Nor its brightest channel (gamma) this.
pub const MAX_FOG_LEVEL: f32 = 0.78;
/// fork: **a room's air is never a wall.** The most extinction (1/yd) a building's MFOG is solved to:
/// [`WALL_TAU`] over a cavern's short record put the Cleft of Shadows in a purple wall inside 50 yd.
/// At this cap 50 yd keeps 86 % of the scene and 100 yd 74 %. Underwater keeps its wall.
pub const ROOM_MAX_DENSITY: f32 = 0.003;
/// fork: **a storm closes the horizon, not the street.** At full storm the zone look's clear range
/// draws in to this (planar yd) instead of to the eye, so the storm's short authored end still
/// walls the distance while the scene around the camera stays readable. With no clear range at
/// all the modern storm kept 15 % of the scene at 100 yd in Elwynn, under the reference's 43 %.
pub const STORM_CLEAR: f32 = 120.0;
/// The shaft haze a full storm adds (1/yd): a uniform humid veil, never near-cleared. 0.006 alone
/// took 45 % of the scene at 100 yd and 0.002 a third at 200; this keeps 79 % at 200.
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
    /// The zone look's key-light scattering weight.
    pub key_scatter: f32,
    /// The zone look's fog reach (>1 = clearer), colour multiplier and height falloff (yd).
    pub fog_reach: f32,
    pub fog_tint: [f32; 3],
    pub fog_falloff: f32,
    /// The zone look's minimum clear range (planar yd), open air only.
    pub fog_clear: f32,
    /// fork: the zone look's room-fog reach (>1 = clearer): divides a building's MFOG extinction.
    pub room_reach: f32,
    /// fork: the zone look's `added_air`: the share of the storm's humid veil a zone takes on top
    /// of its own haze.
    pub added_air: f32,
}

/// The solved medium, in the units `froxelfog.wgsl` reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Media {
    /// Fog extinction at the reference height (1/yd along the ray).
    pub density: f32,
    /// `1 / height falloff`; 0 = uniform.
    pub inv_height: f32,
    /// No fog nearer than this (planar yd); the shaft haze ignores it.
    pub near_clear: f32,
    /// Shaft haze extinction (1/yd).
    pub haze: f32,
    /// Density noise amount (0 = smooth).
    pub noise: f32,
    /// Linear in-scatter radiance from the ambient term.
    pub ambient_in: [f32; 3],
    /// Linear in-scatter radiance from the key light, before the phase function.
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
    let ambient = decode(i.ambient);
    let key = decode(i.diffuse);
    let storm = i.storm.clamp(0.0, 1.0);
    let interior = i.interior.clamp(0.0, 1.0);

    // The open air calibrates on the zone's own end; a room (and the water) on the pushed wall.
    let wall_span = (i.fog_end - i.fog_start.max(0.0)).max(MIN_SPAN);
    let open_end = i.zone_fog_end.max(i.fog_end);
    let open_start = if i.fog_end > 1e-3 {
        i.fog_start / i.fog_end * open_end
    } else {
        0.0
    };
    let open_span = (open_end - open_start.max(0.0)).max(MIN_SPAN);
    let open_density = FAR_TAU / (open_span * i.fog_reach.max(0.05));
    let wall_density = if i.underwater {
        WALL_TAU / wall_span
    } else {
        (WALL_TAU / wall_span).min(ROOM_MAX_DENSITY) / i.room_reach.max(0.05)
    };
    let walled = if i.underwater { 1.0 } else { interior };
    let density = open_density + (wall_density - open_density) * walled;
    // The look's clear range, drawn in to `STORM_CLEAR` by a storm.
    let look_clear = i.fog_clear + (i.fog_clear.min(STORM_CLEAR) - i.fog_clear) * storm;
    let near_clear = (open_start + (i.fog_start - open_start) * walled)
        .max(0.0)
        .min(MAX_CLEAR_SHARE * (open_span + (wall_span - open_span) * walled))
        .max(look_clear * (1.0 - walled));
    // An overcast sky scatters less direct light into the fog.
    let key_scatter = i.key_scatter.max(0.0) * (1.0 - 0.75 * storm);

    // In-scatter reaching the eye from infinitely far is the medium's radiance `L`; pick the albedo
    // that makes it the fog colour for a ray of average phase.
    let mut ambient_in = [0.0; 3];
    let mut key_in = [0.0; 3];
    for c in 0..3 {
        let lit = ambient[c] + key[c] * key_scatter * REFERENCE_PHASE;
        let albedo = if lit > 1e-5 {
            (fog[c] / lit).min(MAX_ALBEDO)
        } else {
            0.0
        };
        ambient_in[c] = albedo * ambient[c];
        key_in[c] = albedo * key[c] * key_scatter;
        // Move in-scatter from the key to the ambient term until the shaded fog keeps its share; the
        // sum at the reference phase (the fog colour) is unchanged.
        let target = ambient_in[c] + key_in[c] * REFERENCE_PHASE;
        let floor = target * SHADED_SHARE;
        if ambient_in[c] < floor {
            let key_left = target - ambient_in[c];
            key_in[c] *= (target - floor) / key_left;
            ambient_in[c] = floor;
        }
    }

    if i.underwater {
        return Media {
            density,
            inv_height: 0.0,
            near_clear,
            haze: 0.0,
            noise: 0.1,
            ambient_in,
            key_in,
            forward_share: 0.55,
            g_forward: 0.5,
            g_backward: BACKWARD_G,
        };
    }
    Media {
        // A storm closes the horizon in: its (shorter) authored end already raises the density.
        density,
        inv_height: (1.0 - interior) / i.fog_falloff.max(1.0),
        near_clear,
        haze: i.haze.max(0.0) + STORM_HAZE * storm * i.added_air.clamp(0.0, 1.0),
        // Faint: drifting density wisps over the ground read as cloud shadows sliding past the player.
        // A storm barely raises them: it thickens the air, not the ground.
        noise: 0.15 + 0.1 * storm,
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
            key_scatter: 0.6,
            fog_reach: 1.0,
            fog_tint: [1.0; 3],
            fog_falloff: HEIGHT_FALLOFF,
            fog_clear: 0.0,
            room_reach: 1.0,
            added_air: 1.0,
        }
    }

    /// CIE L*a*b* of a linear sRGB colour (D65).
    fn lab(c: [f32; 3]) -> [f32; 3] {
        let x = 0.4124 * c[0] + 0.3576 * c[1] + 0.1805 * c[2];
        let y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let z = 0.0193 * c[0] + 0.1192 * c[1] + 0.9505 * c[2];
        let f = |t: f32| {
            if t > 0.008_856 {
                t.cbrt()
            } else {
                7.787 * t + 16.0 / 116.0
            }
        };
        let (fx, fy, fz) = (f(x / 0.9505), f(y), f(z / 1.089));
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
    }

    fn delta_e(a: [f32; 3], b: [f32; 3]) -> f32 {
        let (a, b) = (lab(a), lab(b));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The plan's exit: distant fog within ΔE 5 of the DBC colour (held subtle). A horizontal ray to infinity at the
    /// reference phase lands on the colour exactly; side-on to the sun it stays close.
    #[test]
    fn distant_fog_lands_on_the_dbc_colour() {
        let i = elwynn_noon();
        let m = solve(&i);
        let target = subtle(i.fog_color).map(crate::modern::srgb_to_linear);
        let far = |phase: f32| [0, 1, 2].map(|c| m.ambient_in[c] + m.key_in[c] * phase);
        assert!(
            delta_e(far(REFERENCE_PHASE), target) < 0.5,
            "{:?} vs {target:?}",
            far(REFERENCE_PHASE)
        );
        // Side-on through the dual lobe (μ = 0): 0.7 · hg(0.75) + 0.3 · hg(−0.2).
        let hg = |g: f32, mu: f32| (1.0 - g * g) / (1.0 + g * g - 2.0 * g * mu).powf(1.5);
        let side =
            m.forward_share * hg(FORWARD_G, 0.0) + (1.0 - m.forward_share) * hg(BACKWARD_G, 0.0);
        assert!(
            delta_e(far(side), target) < 5.0,
            "ΔE {}",
            delta_e(far(side), target)
        );
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
        assert!(
            (a.density - b.density).abs() < 1e-6,
            "{} vs {}",
            a.density,
            b.density
        );
        assert!((a.near_clear - b.near_clear).abs() < 1e-3);
        let span = 500.0 - 125.0;
        assert!((a.density * span - FAR_TAU).abs() < 1e-4);
        assert!(
            (-a.density * span).exp() > 0.15,
            "the zone end must still show the scene"
        );
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
    fn a_storm_veil_has_no_clear_range_and_more_haze() {
        let mut i = elwynn_noon();
        i.fog_start = -60.0;
        i.fog_end = 278.0;
        i.zone_fog_end = 278.0;
        i.storm = 1.0;
        let m = solve(&i);
        assert_eq!(m.near_clear, 0.0);
        assert!(m.haze > elwynn_noon().haze);
        assert!(m.density > solve(&elwynn_noon()).density);
        assert!(m.forward_share < 0.5);
    }

    /// The scene a horizontal ray at the reference height keeps over `d` yards: the haze from the
    /// eye, the fog past `froxelfog::fog_clear`'s ease (`smoothstep(0.8·clear, clear, d)`).
    fn kept(m: &Media, d: f32) -> f32 {
        let (lo, hi) = (0.8 * m.near_clear, m.near_clear);
        let eased = |x: f32| {
            if hi - lo < 1e-3 {
                return (x - hi).max(0.0);
            }
            // ∫ smoothstep = (hi − lo)·(t³ − t⁴/2), then the full medium past `hi`.
            let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            (hi - lo) * (t.powi(3) - 0.5 * t.powi(4)) + (x - hi).max(0.0)
        };
        (-(m.haze * d + m.density * eased(d))).exp()
    }

    /// Elwynn under [`crate::modern::look::ZoneLook::NEUTRAL`] at full storm: the committed
    /// −139/278 pair (`lighting::resolve`). The storm draws the look's clear range in to
    /// [`STORM_CLEAR`], so the near field stays readable (the reference's linear fog keeps 43 % of
    /// the scene at 100 yd) while the storm's authored end still walls the distance.
    #[test]
    fn a_storm_closes_the_horizon_and_not_the_street() {
        let look = crate::modern::look::ZoneLook::NEUTRAL;
        let clear = MediaInputs {
            haze: look.haze * super::super::HAZE_SCALE,
            fog_reach: look.fog_reach,
            fog_clear: look.fog_clear,
            ..elwynn_noon()
        };
        let storm = MediaInputs {
            fog_start: -139.0,
            fog_end: 278.0,
            zone_fog_end: 278.0,
            storm: 1.0,
            ..clear
        };
        let (c, s) = (solve(&clear), solve(&storm));
        assert_eq!(c.near_clear, look.fog_clear);
        assert_eq!(s.near_clear, STORM_CLEAR);
        assert!((s.haze - c.haze - STORM_HAZE).abs() < 1e-6);
        assert!(kept(&s, 50.0) > 0.8, "{}", kept(&s, 50.0));
        assert!(kept(&s, 100.0) > 0.6, "{}", kept(&s, 100.0));
        // The horizon: the storm's end keeps a tenth of what the clear day does there.
        assert!(kept(&s, 278.0) < 0.1, "{}", kept(&s, 278.0));
        assert!(kept(&c, 278.0) > 0.6, "{}", kept(&c, 278.0));
        // Half-way up the ramp the clear range is half-way in.
        let half = solve(&MediaInputs {
            storm: 0.5,
            ..storm
        });
        assert!((half.near_clear - 0.5 * (look.fog_clear + STORM_CLEAR)).abs() < 1e-3);
    }

    /// A look with a shorter clear range than the storm's keeps its own: the storm never clears air.
    #[test]
    fn a_storm_never_lengthens_the_clear_range() {
        let mut i = elwynn_noon();
        i.fog_clear = 40.0;
        i.fog_start = -139.0;
        i.fog_end = 278.0;
        i.zone_fog_end = 278.0;
        i.storm = 1.0;
        assert_eq!(solve(&i).near_clear, 40.0);
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

    /// Ashenvale noon: a teal fog over a dark blue ambient row under a bright key. Fog the key cannot
    /// reach keeps the shaded share of the colour, and the reference-phase sum is still the colour.
    #[test]
    fn shaded_fog_keeps_its_share_of_the_colour() {
        let i = MediaInputs {
            fog_color: [0.012, 0.286, 0.42],
            fog_start: 35.0,
            fog_end: 350.0,
            zone_fog_end: 350.0,
            ambient: [0.28, 0.39, 0.65],
            diffuse: [0.75, 0.96, 1.0],
            storm: 0.0,
            underwater: false,
            interior: 0.0,
            haze: 0.0039,
            forward: 0.68,
            key_scatter: 0.75,
            fog_reach: 1.0,
            fog_tint: [1.0; 3],
            fog_falloff: HEIGHT_FALLOFF,
            fog_clear: 0.0,
            room_reach: 1.0,
            added_air: 1.0,
        };
        let m = solve(&i);
        for c in 0..3 {
            let sum = m.ambient_in[c] + m.key_in[c] * REFERENCE_PHASE;
            assert!(m.ambient_in[c] >= sum * SHADED_SHARE - 1e-6, "channel {c}");
            let target = crate::modern::srgb_to_linear(subtle(i.fog_color)[c]);
            assert!(
                (sum - target).abs() < 1e-4,
                "channel {c}: {sum} vs {target}"
            );
        }
    }

    #[test]
    fn a_black_ambient_channel_stays_finite() {
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
    /// MFOG: Orgrimmar's gate hall wore Durotar's tint (green x2.6, which its grey fog wants as
    /// amber) over the WMO's teal record, and the hall went green. The tint fades out with the
    /// room's fog and is untouched in the open air.
    #[test]
    fn a_rooms_own_fog_does_not_take_the_zones_tint() {
        let tint = [1.7, 2.6, 1.8];
        let teal = [15.0 / 255.0, 52.0 / 255.0, 53.0 / 255.0];
        let colour = |interior: f32| {
            let mut i = elwynn_noon();
            i.fog_color = teal;
            i.fog_tint = tint;
            i.interior = interior;
            let m = solve(&i);
            [0, 1, 2].map(|c| m.ambient_in[c] + m.key_in[c] * REFERENCE_PHASE)
        };
        let room = colour(1.0);
        let target = subtle(teal).map(crate::modern::srgb_to_linear);
        for c in 0..3 {
            assert!(
                (room[c] - target[c]).abs() < 1e-4,
                "channel {c}: {} vs {}",
                room[c],
                target[c]
            );
        }
        // In the open the tint still does its work — the green lift is the whole point of the row.
        let open = colour(0.0);
        assert!(
            open[1] > room[1] * 2.0,
            "open {:?} vs room {:?}",
            open,
            room
        );
    }
}
