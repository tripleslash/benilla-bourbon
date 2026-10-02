//! **Modern weather** (fork-owned): how a modern frame draws the reference's weather.
//!
//! The simulation is upstream's and untouched: the same mist puffs are born, streamed and retired
//! by the same laws (`weather::precip::mist`). Only a puff's look differs. The reference's puff is
//! a 12 yd sheet of the fog colour, opaque from 18 yd, and up to 128 of them stream through the
//! camera: against the classic fog wall they are the wall's near edge, but a modern frame's air is
//! a thin haze the scene shows through, and there the sheets alone hid it. A modern puff is a
//! smaller, fainter wisp of that same haze. The storm's hold on the distance is the fog medium's
//! (`volumetrics::media`, `STORM_CLEAR`).

use super::volumetrics::media;

/// What one mist puff looks like (`weather::precip::mist::push_mist`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuffLook {
    /// Half the puff quad's edge (yd).
    pub half: f32,
    /// The per-corner distance fade: nothing nearer than `near`, whole from `far` (yd to the eye).
    pub near: f32,
    pub far: f32,
    /// The fade-in and fade-out ramps of a puff's life (s).
    pub fade: f32,
    /// The most a puff covers, over its fades and its texture.
    pub alpha: f32,
}

/// A modern frame's puff: 10 yd (0.69 of the reference's area), at most 0.6 opaque, clear within
/// 8 yd and whole from 20, on 0.7 s ramps, so nothing solid crosses the camera. Together about a
/// third of the reference's veil.
pub const PUFF: PuffLook = PuffLook {
    half: 5.0,
    near: 8.0,
    far: 20.0,
    fade: 0.7,
    alpha: 0.6,
};

/// The puff a modern frame draws; `None` in a classic one, which keeps the reference's.
pub fn puff_look() -> Option<PuffLook> {
    super::frame_is_modern().then_some(PUFF)
}

/// The colour (gamma) a modern puff takes for the scene's fog colour: the haze's own, the zone
/// look's tint under the same saturation and level caps as the medium (`media::subtle`).
pub fn puff_color(fog_color: [f32; 3]) -> [f32; 3] {
    let tint = super::look::fog_tint();
    media::subtle([0, 1, 2].map(|c| fog_color[c] * tint[c]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bright, saturated authored fog (a sandstorm's) is held to the haze's caps, hue kept; a
    /// grey storm fog passes through.
    #[test]
    fn a_puff_is_never_brighter_or_more_saturated_than_the_haze() {
        let sand = puff_color([0.95, 0.80, 0.45]);
        let (hi, lo) = (sand[0], sand[2]);
        assert!(hi <= media::MAX_FOG_LEVEL + 1e-5);
        assert!((hi - lo) / hi <= media::MAX_FOG_SATURATION + 1e-4);
        assert!(sand[0] >= sand[1] && sand[1] >= sand[2], "hue kept");
        assert_eq!(puff_color([0.4, 0.42, 0.45]), [0.4, 0.42, 0.45]);
    }
}
