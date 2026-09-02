//! **The cata liquid bump map** — the one asset the Cataclysm water path needs and no install has.
//!
//! `ProcWaterAbove`/`ProcWaterBelow` sample a per-liquid-type normal map on stage 5 four times per
//! fragment, at two tiling scales and two scroll phases, and blend the two pairs by view distance.
//! In the reference that texture is authored Cataclysm art, reached through
//! `Liquid::CMaterialSettings::GetTexture(mat, 2, 0)` — i.e. a `LiquidType.dbc` `Texture[]` column
//! that 1.12.1's `LiquidType.dbc` does not have. It is also not among the files under
//! the Cataclysm client this port was read from, whose files carry the shader programs and `Ripples/perturbTex.blp` and nothing else.
//!
//! So it is **synthesised**, not shipped: [`generate`] builds a tileable four-octave value-noise
//! height field and stores its central-difference normal. The bytes are generated once and
//! committed next to this file, then `include_bytes!`d — the binary answers for its own content
//! with no runtime cost and no dependency on a data directory, and benilla keeps its standing
//! promise to bundle no game assets (`README.md`), because nothing here came from Blizzard.
//!
//! The test below is the seam: it re-runs [`generate`] and compares against the committed bytes.
//! Change the algorithm and it fails until the blob is regenerated, which
//! `WOW_REGEN_LIQUID_ART=1 cargo test -p benilla-world liquid::art` does in place.
//!
//! **Why a full RGBA normal and not a two-channel derivative.** The *below-water* program reads
//! three components off this texture (`ProcWaterBelow` perm 0: `add r1.xyz, r1, r2` … `dp3 r0.y,
//! r1, r1`), so the tap's `z` is load-bearing — a two-channel map would hand `normalize` a
//! zero-length vector the moment the camera goes under. `x`/`y`/`z` are stored `n · 0.5 + 0.5`, so
//! a flat texel is `(0.5, 0.5, 1.0)` and the programs' `tap_a + tap_b − 1` recovers a sum centred
//! on `+Z`.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

/// Edge length of the bump map, in texels. Square and power-of-two: it is tiled at four different
/// scales at once, so a non-power-of-two period would beat against itself.
pub(super) const BUMP_SIZE: u32 = 256;

/// The committed bytes — `BUMP_SIZE² × RGBA8`, produced by [`generate`].
const BUMP_BYTES: &[u8] = include_bytes!("../../assets/liquid/water_bump.rgba");

/// Build the `Image` the cata liquid materials bind on stage 5.
///
/// `RENDER_WORLD` only, like every other world texture: nothing on the CPU side reads it back, and
/// the main-world copy would be a quarter-megabyte held for nothing. The sampler repeats on both
/// axes — the whole point is tiling — and takes linear filtering.
pub(super) fn bump_image() -> Image {
    let mut image = Image::new(
        Extent3d {
            width: BUMP_SIZE,
            height: BUMP_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        BUMP_BYTES.to_vec(),
        // NOT sRGB: this is a normal, not a colour. The gamma lane (`world_assets::images`) is
        // about colour bytes; pushing a direction through a transfer curve would bend it.
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        label: Some("liquid_bump".into()),
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

/// Deterministically synthesise the bump map: four octaves of tileable value noise summed into a
/// height field, then stored as its central-difference normal.
///
/// `#[cfg(test)]`, because the *product* is what ships: the blob is committed and `include_bytes!`d,
/// and the only caller is the test that verifies it still matches (or, under
/// `WOW_REGEN_LIQUID_ART`, rewrites it). Compiling the generator into the client would be dead code
/// in every shipped binary; deleting it would leave a blob with no stated provenance. This is the
/// third option.
///
/// Tileability is the only subtle part, and it is bought at the lattice: each octave's integer grid
/// wraps modulo its own cell count, and every period divides [`BUMP_SIZE`], so the field is
/// continuous across the seam by construction rather than by blending a mirrored copy over it.
#[cfg(test)]
fn generate() -> Vec<u8> {
    let n = BUMP_SIZE as usize;
    // (period in texels, amplitude). The coarse octaves carry the swell, the fine ones the chop;
    // amplitudes halve so the summed slope stays inside what the programs' `c23.w` expects to
    // scale.
    const OCTAVES: [(usize, f32); 4] = [(64, 1.0), (32, 0.5), (16, 0.25), (8, 0.125)];
    let mut height = vec![0.0f32; n * n];
    for (period, amp) in OCTAVES {
        let cells = n / period; // lattice points across the texture — wraps exactly
        for y in 0..n {
            for x in 0..n {
                height[y * n + x] += amp * value_noise(x, y, period, cells);
            }
        }
    }
    // Normalise to 0..1 so the relief scale below is independent of how many octaves were summed.
    let (lo, hi) = height
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
    let span = (hi - lo).max(1e-6);
    for h in &mut height {
        *h = (*h - lo) / span;
    }

    // Central differences on the wrapped lattice → a slope, then a normal. `RELIEF` is the height,
    // in texel widths, that a full 0..1 excursion stands for; it sets how steep the encoded normals
    // get. 6.0 puts the steepest texels near 45°, which is about where the reference's taps sit
    // once `c23.w` has scaled them.
    const RELIEF: f32 = 6.0;
    let mut out = vec![0u8; n * n * 4];
    let at = |hs: &[f32], x: usize, y: usize| hs[(y % n) * n + (x % n)];
    for y in 0..n {
        for x in 0..n {
            let dx = (at(&height, x + 1, y) - at(&height, x + n - 1, y)) * 0.5 * RELIEF;
            let dy = (at(&height, x, y + 1) - at(&height, x, y + n - 1)) * 0.5 * RELIEF;
            // The normal of a height field z = h(x, y) is (−dh/dx, −dh/dy, 1), normalised.
            let inv = 1.0 / (dx * dx + dy * dy + 1.0).sqrt();
            let (nx, ny, nz) = (-dx * inv, -dy * inv, inv);
            let i = (y * n + x) * 4;
            out[i] = encode(nx);
            out[i + 1] = encode(ny);
            out[i + 2] = encode(nz);
            out[i + 3] = 255;
        }
    }
    out
}

/// `n ∈ [−1, 1]` → a byte, the standard `n · 0.5 + 0.5` encoding the programs undo with their
/// `tap + tap − 1`.
#[cfg(test)]
fn encode(v: f32) -> u8 {
    ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

/// One octave of tileable value noise: smoothstep-interpolated hashes on a `cells × cells` lattice
/// that wraps at the texture edge.
#[cfg(test)]
fn value_noise(x: usize, y: usize, period: usize, cells: usize) -> f32 {
    let fx = (x % period) as f32 / period as f32;
    let fy = (y % period) as f32 / period as f32;
    let (cx, cy) = (x / period, y / period);
    let corner =
        |ox: usize, oy: usize| hash(((cx + ox) % cells) as u32, ((cy + oy) % cells) as u32);
    let (sx, sy) = (smoothstep(fx), smoothstep(fy));
    let top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * sx;
    let bottom = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * sx;
    top + (bottom - top) * sy
}

#[cfg(test)]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// A cheap integer hash into `[0, 1)`. Fixed constants and no RNG: the point is that two machines
/// (and the test below) produce byte-identical output.
#[cfg(test)]
fn hash(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 13;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    /// The committed bytes ARE [`super::generate`]'s output.
    ///
    /// This is the whole provenance story for a binary blob in the tree: the function above is the
    /// source, the file is its build product, and this test is the link between them. If it fails,
    /// either the generator changed on purpose — `WOW_REGEN_LIQUID_ART=1 cargo test -p
    /// benilla-world liquid::art` rewrites the file — or something edited the blob by hand, which
    /// is exactly the thing worth failing over.
    #[test]
    fn the_committed_bump_map_matches_the_generator() {
        let fresh = super::generate();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/liquid/water_bump.rgba");
        if std::env::var_os("WOW_REGEN_LIQUID_ART").is_some() {
            std::fs::write(path, &fresh).unwrap();
            return;
        }
        assert_eq!(
            fresh.len(),
            super::BUMP_BYTES.len(),
            "committed bump map is the wrong size"
        );
        assert!(
            fresh == super::BUMP_BYTES,
            "assets/liquid/water_bump.rgba is stale — regenerate with \
             WOW_REGEN_LIQUID_ART=1 cargo test -p benilla-world liquid::art"
        );
    }

    /// The encoded field is unit normals with a positive `z`, and it TILES.
    ///
    /// The second half is the property the four-tap blend actually depends on: a seam would read as
    /// a hard line swimming across the water at four different scales at once.
    #[test]
    fn the_bump_map_is_unit_normals_and_wraps() {
        let n = super::BUMP_SIZE as usize;
        let b = super::BUMP_BYTES;
        let nrm = |x: usize, y: usize| {
            let i = (y * n + x) * 4;
            let d = |c: usize| b[i + c] as f32 / 255.0 * 2.0 - 1.0;
            (d(0), d(1), d(2))
        };
        for (x, y) in [(0usize, 0usize), (7, 91), (128, 128), (255, 255)] {
            let (a, c, e) = nrm(x, y);
            let len = (a * a + c * c + e * e).sqrt();
            assert!((len - 1.0).abs() < 0.02, "({x},{y}) is not unit: {len}");
            assert!(e > 0.0, "({x},{y}) points into the surface");
        }
        // Column 0 continues column 255, and row 0 continues row 255 — within a slope step of what
        // the interior gradient does, not exactly equal (they are different texels).
        for i in 0..n {
            let (l, r) = (nrm(0, i), nrm(n - 1, i));
            let (t, bo) = (nrm(i, 0), nrm(i, n - 1));
            assert!((l.0 - r.0).abs() < 0.35, "vertical seam at row {i}");
            assert!((t.1 - bo.1).abs() < 0.35, "horizontal seam at column {i}");
        }
    }
}
