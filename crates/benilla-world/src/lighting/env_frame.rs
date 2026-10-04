//! fork: the environment block, ported from the Everwood fork's `MonkeyFrame` and cut down to the
//! lanes we take (wind, rain wetness, benders).
//!
//! One resource, [`EnvFrame`], that the wind and wetness systems write. The light packer
//! (`global_light::build_light_data`) copies it every frame into [`ENV_FRAME_ROWS`] rows appended
//! to the shared light buffer right after the point-light table, so no earlier offset moves. The
//! WGSL mirror is `benilla::env_frame::EnvFrame` (`benilla-assets/src/shaders/env_frame.wgsl`).
//!
//! All zero (the default) is "every feature off": every reader treats a zero row as identity.
//!
//! Layout (12 × `vec4<f32>`, 192 B):
//!
//! | row | xyz | w |
//! |---|---|---|
//! | 0 `wind_a` | dir_x, dir_y, base_heading (rad) | gust |
//! | 1 `wind_b` | travel (yd, wrapped), sway_strength, grass_strength | tree_strength |
//! | 2 `wet_a` | rain_rate, wetness, ripple_time_s | snow |
//! | 3 `misc` | bender_count, wet_ssr, 0 | 0 |
//! | 4-11 `benders` | world x, y, z | radius |
//!
//! Coordinates are Bevy world space (Y up, 1 unit = 1 yd), the receivers' `world_position` space.
//! The wind direction is a unit vector in the world XZ plane: `dir_x` = world x, `dir_y` = world z.
//!
//! The rain-shelter map (`weather::shelter`) is its own region right behind the per-frame blob
//! ([`shelter_region_offset`]), ahead of the prop probes.

use bevy::prelude::*;

/// Rows in the appended block.
pub const ENV_FRAME_ROWS: usize = 12;

/// How many benders (grass pushers: the player and nearby units) the block carries.
pub const MAX_BENDERS: usize = 8;

/// Cells per side of the rain-shelter map.
pub const SHELTER_GRID: usize = 128;
/// Header rows before the shelter cells.
pub const SHELTER_HEADER_ROWS: usize = 2;
/// The shelter region's size in the shared light buffer: the header and one word a cell.
pub const SHELTER_REGION_BYTES: u64 =
    (SHELTER_HEADER_ROWS * 16 + SHELTER_GRID * SHELTER_GRID * 4) as u64;

/// Byte offset of the shelter region: right after the per-frame blob, before the prop probes.
pub fn shelter_region_offset() -> u64 {
    super::global_light::per_frame_blob_bytes()
}

/// The per-frame environment parameters. Each lane writes the fields it owns through
/// `ResMut<EnvFrame>`; see the module docs for the packed layout.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct EnvFrame {
    // Wind (rows 0-1).
    /// Unit vector in the world XZ plane (`[x, z]`).
    pub wind_dir: [f32; 2],
    /// The profile's fixed heading in radians: the waves' spatial term uses it, so the veer never
    /// rotates the phase field about the world origin.
    pub wind_base_heading: f32,
    pub wind_gust: f32,
    /// The integrated wind travel in yards, wrapped at `wind::TRAVEL_WRAP`; the shader's wave
    /// rates are whole cycles per wrap, so it is seamless.
    pub wind_travel: f32,
    pub sway_strength: f32,
    pub grass_strength: f32,
    pub tree_strength: f32,
    // Wetness (row 2).
    pub rain_rate: f32,
    pub wetness: f32,
    pub ripple_time_s: f32,
    pub snow: f32,
    /// 1 while the wet screen-space reflection pass composites (`weather::wet_ssr`): the terrain
    /// then tags its reflection weight in alpha instead of adding the sky colour itself.
    pub wet_ssr: f32,
    // Benders (row 3 `.x`, rows 4-11).
    /// Only the first `min(bender_count, MAX_BENDERS)` are packed; `[x, y, z, radius]`.
    pub benders: [[f32; 4]; MAX_BENDERS],
    pub bender_count: u32,
}

impl EnvFrame {
    /// The packed rows.
    pub fn pack(&self) -> [[f32; 4]; ENV_FRAME_ROWS] {
        let mut rows = [[0.0f32; 4]; ENV_FRAME_ROWS];
        rows[0] = [
            self.wind_dir[0],
            self.wind_dir[1],
            self.wind_base_heading,
            self.wind_gust,
        ];
        rows[1] = [
            self.wind_travel,
            self.sway_strength,
            self.grass_strength,
            self.tree_strength,
        ];
        rows[2] = [self.rain_rate, self.wetness, self.ripple_time_s, self.snow];
        let n = (self.bender_count as usize).min(MAX_BENDERS);
        rows[3] = [n as f32, self.wet_ssr, 0.0, 0.0];
        rows[4..4 + n].copy_from_slice(&self.benders[..n]);
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_frame_packs_to_zero() {
        let rows = EnvFrame::default().pack();
        assert!(rows.iter().flatten().all(|v| *v == 0.0));
    }

    #[test]
    fn fields_land_in_their_documented_rows() {
        let f = EnvFrame {
            wind_dir: [1.0, 2.0],
            wind_base_heading: 3.0,
            wind_gust: 4.0,
            wind_travel: 5.0,
            sway_strength: 6.0,
            grass_strength: 7.0,
            tree_strength: 8.0,
            rain_rate: 9.0,
            wetness: 10.0,
            ripple_time_s: 11.0,
            snow: 12.0,
            wet_ssr: 17.0,
            benders: [[13.0, 14.0, 15.0, 16.0]; MAX_BENDERS],
            bender_count: 20, // clamped to MAX_BENDERS
        };
        let r = f.pack();
        assert_eq!(r[0], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r[1], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(r[2], [9.0, 10.0, 11.0, 12.0]);
        assert_eq!(r[3], [8.0, 17.0, 0.0, 0.0]);
        assert_eq!(r[11], [13.0, 14.0, 15.0, 16.0]);
    }

    /// The WGSL mirror declares the same rows; a member added on one side only would shift every
    /// bender and every region behind the block.
    #[test]
    fn the_wgsl_mirror_has_the_same_rows() {
        let src = include_str!("../../../benilla-assets/src/shaders/env_frame.wgsl")
            .replace("\r\n", "\n");
        let body = src.split_once("struct EnvFrame {").unwrap().1;
        let body = body.split_once("\n}").unwrap().0;
        let mut rows = 0;
        for line in body
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
        {
            if line.contains("array<vec4<f32>, 8>") {
                rows += 8;
            } else if line.contains("vec4<f32>") {
                rows += 1;
            }
        }
        assert_eq!(rows, ENV_FRAME_ROWS);
    }

    /// Every shader that declares the buffer past the point table declares the block and the
    /// shelter region in the same order, or every region behind them reads the wrong bytes.
    #[test]
    fn the_tail_mirrors_declare_the_block_and_the_shelter_region() {
        let cells = SHELTER_GRID * SHELTER_GRID;
        for (name, src) in [
            (
                "wow_model.wgsl",
                include_str!("../../../benilla-assets/src/shaders/wow_model.wgsl"),
            ),
            ("static_gx.wgsl", include_str!("../shaders/static_gx.wgsl")),
            (
                "fsr/velocity.wgsl",
                include_str!("../shaders/fsr/velocity.wgsl"),
            ),
            (
                "modern/entity_shadow.wgsl",
                include_str!("../shaders/modern/entity_shadow.wgsl"),
            ),
        ] {
            let points = src.find("points: array<vec4<f32>, 512>").expect(name);
            let probes = src
                .find("prop_probes: array<vec4<f32>, 57344>")
                .expect(name);
            let between = &src[points..probes];
            assert!(
                between.contains("env: env_frame::EnvFrame")
                    || between.contains(&format!("_env: array<vec4<f32>, {ENV_FRAME_ROWS}>")),
                "{name}: the environment block sits between the point table and the probes"
            );
            assert!(
                between.contains(&format!("shelter: array<u32, {cells}>")),
                "{name}: the shelter cells sit between the block and the probes"
            );
        }
        assert_eq!(SHELTER_REGION_BYTES % 16, 0);
        assert_eq!(shelter_region_offset() % 16, 0);
        assert_eq!(
            super::super::prop_probe_region_offset(),
            shelter_region_offset() + SHELTER_REGION_BYTES
        );
    }
}
