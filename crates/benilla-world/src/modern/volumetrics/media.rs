//! **The fog media** (`FORK.md` P4): the zone's authored linear GL fog turned into a
//! participating medium that reproduces it.
//!
//! - **Extinction** `σt = 3.9 / (end − max(start, 0))`, so a horizontal ray keeps ≈2 % at the fog end.
//! - **Height falloff** of 60 yd above a reference height that trails the camera; the classic
//!   near-clear range (the fog's start) is kept.
//! - **Albedo** solved per channel so that the in-scatter of an ambient-plus-key medium at the fog end
//!   equals the DBC fog colour, decoded to linear. The zone's hue survives because the fog colour is
//!   the target, not an input to a physical sky: Teldrassil stays purple, Darkshore pink.
//!
//! The variations the plan names ride the same solve: a storm thickens the haze and dims the key,
//! underwater drops the height falloff and the shaft haze, and an interior (the WMO crossfade `t`)
//! flattens the falloff toward a uniform room haze.

/// Optical depth at the authored fog end: `exp(−3.9)` ≈ 2 % transmittance.
pub const OPTICAL_DEPTH_AT_END: f32 = 3.9;
/// Density drops by `e` every this many yards above the reference height.
pub const HEIGHT_FALLOFF: f32 = 60.0;
/// The phase value the albedo solve assumes for the key light — roughly what a horizontal view ray
/// sees with the dual lobe below. Toward the sun the fog reads warmer and brighter, away from it closer
/// to the zone's own hue (amendment A2).
pub const REFERENCE_PHASE: f32 = 0.5;
/// The dual-lobe Henyey-Greenstein pair.
pub const FORWARD_G: f32 = 0.75;
pub const BACKWARD_G: f32 = -0.2;
/// A sky pixel keeps this share of the volume's optical depth (the dome already paints its horizon).
pub const SKY_AERIAL: f32 = 0.35;
/// The shortest fog span the solve accepts (yd); a degenerate DBC pair must not read as a wall.
const MIN_SPAN: f32 = 8.0;
/// Albedo is a calibration here, not a physical ratio, but a black ambient channel must not explode.
const MAX_ALBEDO: f32 = 16.0;

/// What one frame's media is solved from. Colours are the authored gamma values.
#[derive(Clone, Copy, Debug)]
pub struct MediaInputs {
    pub fog_color: [f32; 3],
    pub fog_start: f32,
    pub fog_end: f32,
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
}

/// The solved medium, in the units `froxelfog.wgsl` reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Media {
    /// Fog extinction (1/yd) at the reference height.
    pub sigma: f32,
    /// `1 / height falloff`; 0 = uniform.
    pub inv_height: f32,
    /// No fog nearer than this (yd); the shaft haze ignores it.
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

pub fn solve(i: &MediaInputs) -> Media {
    let decode = |c: [f32; 3]| c.map(crate::modern::srgb_to_linear);
    let fog = decode(i.fog_color);
    let ambient = decode(i.ambient);
    let key = decode(i.diffuse);
    let storm = i.storm.clamp(0.0, 1.0);
    let interior = i.interior.clamp(0.0, 1.0);

    let span = (i.fog_end - i.fog_start.max(0.0)).max(MIN_SPAN);
    let sigma = OPTICAL_DEPTH_AT_END / span;
    // An overcast sky scatters less direct light into the fog.
    let key_scatter = i.key_scatter.max(0.0) * (1.0 - 0.75 * storm);

    // In-scatter reaching the eye from the fog end is `L · (1 − T_end)`; pick the albedo that makes it
    // the fog colour for a ray of average phase.
    let reach = 1.0 - (-OPTICAL_DEPTH_AT_END).exp();
    let mut ambient_in = [0.0; 3];
    let mut key_in = [0.0; 3];
    for c in 0..3 {
        let lit = ambient[c] + key[c] * key_scatter * REFERENCE_PHASE;
        let albedo = if lit > 1e-5 {
            (fog[c] / reach / lit).min(MAX_ALBEDO)
        } else {
            0.0
        };
        ambient_in[c] = albedo * ambient[c];
        key_in[c] = albedo * key[c] * key_scatter;
    }

    if i.underwater {
        return Media {
            sigma,
            inv_height: 0.0,
            near_clear: 0.0,
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
        sigma,
        inv_height: (1.0 - interior) / HEIGHT_FALLOFF,
        // A storm's negative start is the reference's near veil: no clear range at all.
        near_clear: i.fog_start.max(0.0),
        haze: i.haze.max(0.0) + 0.006 * storm,
        noise: 0.6 + 0.3 * storm,
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
            fog_start: 120.0,
            fog_end: 400.0,
            ambient: [0.40, 0.40, 0.45],
            diffuse: [0.90, 0.85, 0.70],
            storm: 0.0,
            underwater: false,
            interior: 0.0,
            haze: 0.0015,
            forward: 0.7,
            key_scatter: 0.6,
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

    /// The plan's exit: distant fog within ΔE 5 of the DBC colour. A horizontal ray past the fog end
    /// at the reference phase lands on the colour exactly; side-on to the sun it stays close.
    #[test]
    fn distant_fog_lands_on_the_dbc_colour() {
        let i = elwynn_noon();
        let m = solve(&i);
        let target = i.fog_color.map(crate::modern::srgb_to_linear);
        let tau = m.sigma * (i.fog_end - i.fog_start);
        assert!((tau - OPTICAL_DEPTH_AT_END).abs() < 1e-4);
        let far = |phase: f32| {
            let t = (-tau).exp();
            [0, 1, 2].map(|c| (m.ambient_in[c] + m.key_in[c] * phase) * (1.0 - t))
        };
        assert!(delta_e(far(REFERENCE_PHASE), target) < 0.5, "{:?} vs {target:?}", far(REFERENCE_PHASE));
        // Side-on through the dual lobe (μ = 0): 0.7 · hg(0.75) + 0.3 · hg(−0.2).
        let hg = |g: f32, mu: f32| (1.0 - g * g) / (1.0 + g * g - 2.0 * g * mu).powf(1.5);
        let side = m.forward_share * hg(FORWARD_G, 0.0) + (1.0 - m.forward_share) * hg(BACKWARD_G, 0.0);
        assert!(delta_e(far(side), target) < 5.0, "ΔE {}", delta_e(far(side), target));
    }

    #[test]
    fn a_storm_veil_has_no_clear_range_and_more_haze() {
        let mut i = elwynn_noon();
        i.fog_start = -60.0;
        i.storm = 1.0;
        let m = solve(&i);
        assert_eq!(m.near_clear, 0.0);
        assert!(m.haze > elwynn_noon().haze);
        assert!(m.forward_share < 0.5);
    }

    #[test]
    fn underwater_is_uniform_and_hazeless() {
        let mut i = elwynn_noon();
        i.underwater = true;
        let m = solve(&i);
        assert_eq!(m.inv_height, 0.0);
        assert_eq!(m.haze, 0.0);
        assert_eq!(m.near_clear, 0.0);
    }

    #[test]
    fn a_black_ambient_channel_stays_finite() {
        let mut i = elwynn_noon();
        i.ambient = [0.0, 0.0, 0.0];
        i.diffuse = [0.0, 0.0, 0.0];
        let m = solve(&i);
        assert!(m.ambient_in.iter().chain(&m.key_in).all(|v| v.is_finite() && *v == 0.0));
    }

    #[test]
    fn an_interior_flattens_the_height_falloff() {
        let mut i = elwynn_noon();
        i.interior = 1.0;
        assert_eq!(solve(&i).inv_height, 0.0);
    }
}
