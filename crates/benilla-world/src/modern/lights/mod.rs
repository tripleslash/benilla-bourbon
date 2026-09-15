//! **Local lights** (fork-owned; `FORK.md` P6): the light grid that replaces the
//! reference's ≤3-nearest selection in a modern frame, and the cube-shadow atlas a few of those
//! lights get on High and above.
//!
//! **Why a world grid and not the plan's view clusters.** P6 asked for a froxel-shaped cluster grid
//! binned in view space. Every other grid this port grew — the froxel volume, the voxel clipmap, the
//! probe cascades — is world space and camera-centred, for the same two reasons that apply here: the
//! water mirror is a second view that reads all of them (a view-space cluster list would be the main
//! camera's, and wrong on the mirror), and a world cell can be looked up from a world position
//! alone, which is all a hooked lane has. So the bins are `cell` yards on a side around the camera,
//! and the lookup is three floors and a multiply.
//!
//! **What a cell holds.** Up to [`PER_CELL`] indices into the shared light table (already
//! `room_admits`-gated and capped at 256 by `lighting::global_light`), chosen by their strength at
//! the cell's centre, plus a nibble each naming the light's atlas slot. One `vec4<u32>` a cell — the
//! whole grid is one 864 KB region of the shared modern storage block, so P6 needed no new binding
//! either.
//!
//! **The shadowed lights.** A hanging lamp that lights a room through its floor is the interior look
//! P6 is after, so the strongest few local lights get a real cube shadow. Their faces live in the
//! last layer of the modern shadow array as a 6×6 tile grid (six faces × six lights), which is why
//! the receivers needed no new texture: they already bind that array for the sun. The faces are
//! rendered by `static_gx::render::modern_shadow` from the same retained draws the cascades replay,
//! and only when the chosen set or the static world changes — a torch does not move.

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::RenderApp;

use super::{ActiveLighting, LightingQuality};
use crate::lighting::{LightRooms, WorldPointLight};
use crate::view::WorldCamera;

pub mod node;

/// Lights one cell can name. Mirrors `locallight.wgsl`'s `PER_CELL`.
pub const PER_CELL: usize = 8;
/// Shadowed lights the atlas layer holds: [`ATLAS_TILES`]² tiles of six faces each.
pub const MAX_SHADOWED: usize = 6;
/// Tiles per side in the atlas layer.
pub const ATLAS_TILES: u32 = 6;
/// Cells allocated in the block, whatever the session's tier — the largest any tier uses. Held at
/// one size so a quality change never re-creates the buffer every material binds.
pub const CELL_ALLOC: usize = 48 * 24 * 48;
/// Bytes of one cell (`vec4<u32>`).
pub const CELL_BYTES: usize = 16;
/// Size of [`LightParams::std430`] — `locallight.wgsl`'s `LightGrid` prefix. A multiple of 256, so
/// the GI block behind it keeps the 256-aligned probe offset the blend pass binds.
pub const LIGHT_PARAM_BYTES: usize = 32 * 16;
/// Everything P6 adds to the modern storage block.
pub const BLOCK_LIGHT_BYTES: usize = LIGHT_PARAM_BYTES + CELL_ALLOC * CELL_BYTES;

/// The cube faces' near plane (yd). A fixture light is a point, so this only has to clear the
/// emitter's own geometry.
pub const CUBE_NEAR: f32 = 0.25;
/// Shadow bias, in yards back toward the light.
const CUBE_BIAS_YD: f32 = 0.12;
/// Tile border kept clear of filter taps, in texels.
const TILE_INSET_TEXELS: f32 = 2.0;
/// How near the camera a light has to be to be worth a cube.
const SHADOW_RADIUS: f32 = 70.0;

/// What a quality tier bins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightTier {
    pub on: bool,
    /// Cells per axis.
    pub dims: UVec3,
    /// Cell edge (yd).
    pub cell: f32,
    /// Lights that get a cube shadow (0 below High — the plan's tier table).
    pub shadowed: usize,
}

impl LightTier {
    pub const OFF: Self = Self {
        on: false,
        dims: UVec3::ONE,
        cell: 1.0,
        shadowed: 0,
    };

    pub fn cells(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize
    }
}

/// The plan's cluster tier table, read as world cells. Low keeps P3's ≤3-nearest per-pixel path:
/// with no probe GI and no volumetrics there is nothing for a richer selection to sit beside.
pub fn light_tier(quality: LightingQuality) -> LightTier {
    match quality {
        LightingQuality::Low => LightTier::OFF,
        LightingQuality::Medium => LightTier {
            on: true,
            dims: UVec3::new(32, 16, 32),
            cell: 8.0,
            shadowed: 0,
        },
        LightingQuality::High => LightTier {
            on: true,
            dims: UVec3::new(48, 24, 48),
            cell: 6.0,
            shadowed: 4,
        },
        LightingQuality::Ultra => LightTier {
            on: true,
            dims: UVec3::new(48, 24, 48),
            cell: 4.5,
            shadowed: MAX_SHADOWED,
        },
    }
}

/// Is the local-light grid on in this session (`$WOW_MODERN_LIGHTS`, browser `?modern_lights=0`)?
pub fn lights_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MODERN_LIGHTS").as_deref() != Some("0"))
}

/// One light that owns six atlas tiles this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShadowLight {
    pub pos: Vec3,
    pub range: f32,
}

/// This frame's grid placement and shadowed set, extracted for the render world (which owns the
/// binning pass and the atlas render).
#[derive(Resource, Clone, ExtractResource, Default)]
pub struct ModernLights {
    pub on: bool,
    pub tier: LightTier,
    /// The base cell coord — the grid's low corner in cell units.
    pub base: IVec3,
    pub shadowed: Vec<ShadowLight>,
}

impl Default for LightTier {
    fn default() -> Self {
        Self::OFF
    }
}

/// The CPU mirror of `locallight.wgsl`'s `LightGrid` prefix (field order documented there).
#[derive(Clone, Copy, Debug, Default)]
pub struct LightParams {
    pub dims: Vec4,
    pub base: Vec4,
    pub misc: Vec4,
    pub shadow_misc: Vec4,
    pub shadow: [Vec4; MAX_SHADOWED],
}

impl LightParams {
    pub fn std430(&self, out: &mut Vec<u8>) {
        let start = out.len();
        let mut row = |v: Vec4| out.extend_from_slice(bytemuck::cast_slice(&v.to_array()));
        row(self.dims);
        row(self.base);
        row(self.misc);
        row(self.shadow_misc);
        for s in &self.shadow {
            row(*s);
        }
        // `spare`: P7's per-zone local-light grading, and the padding that keeps the GI block's
        // probe offset 256-aligned.
        while out.len() - start < LIGHT_PARAM_BYTES {
            out.extend_from_slice(&[0u8; 16]);
        }
    }
}

/// This frame's parameters for the grid at `camera`.
pub fn params(lights: &ModernLights, shadow_map_size: u32) -> LightParams {
    let tier = &lights.tier;
    let atlas = !lights.shadowed.is_empty();
    let mut flags = 0u32;
    if lights.on {
        flags |= 1;
    }
    if atlas {
        flags |= 2;
    }
    let mut shadow = [Vec4::ZERO; MAX_SHADOWED];
    for (slot, light) in lights.shadowed.iter().enumerate().take(MAX_SHADOWED) {
        shadow[slot] = light.pos.extend(light.range);
    }
    LightParams {
        dims: tier.dims.as_vec3().extend(tier.cell),
        base: lights.base.as_vec3().extend(flags as f32),
        misc: Vec4::new(
            tier.cells() as f32,
            lights.shadowed.len() as f32,
            crate::modern::shadow::LOCAL_LAYER as f32,
            ATLAS_TILES as f32,
        ),
        shadow_misc: Vec4::new(
            CUBE_NEAR,
            CUBE_BIAS_YD,
            TILE_INSET_TEXELS,
            1.0 / shadow_map_size as f32,
        ),
        shadow,
    }
}

/// The grid's base cell for a camera — deterministic, so a cell that scrolls out is the cell that
/// scrolls back in.
pub fn grid_base(tier: &LightTier, camera: Vec3) -> IVec3 {
    (camera / tier.cell).floor().as_ivec3() - (tier.dims.as_ivec3() / 2)
}

/// The reverse-Z clip of one cube face of a light — the caster view the atlas tile is drawn under.
///
/// The convention is the shader's, not the other way round: `locallight::cube_face` picks the face
/// and its uv analytically (no matrix reaches the fragment stage), and this builds the matrix that
/// lands a caster exactly where that lookup will go looking for it. [`tests`] pins the pair.
pub fn cube_face_clip(light_pos: Vec3, face: usize) -> Mat4 {
    let (axis, s, t) = face_basis(face);
    // View space looks down -Z with +Y up, and the shader's `t` runs *down* the face, so up = -t.
    let up = -t;
    let rot = Mat3::from_cols(
        Vec3::new(s.x, up.x, -axis.x),
        Vec3::new(s.y, up.y, -axis.y),
        Vec3::new(s.z, up.z, -axis.z),
    );
    let view = Mat4::from_mat3(rot) * Mat4::from_translation(-light_pos);
    Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 1.0, CUBE_NEAR) * view
}

/// The major axis and the face's `s`/`t` directions — the twin of `locallight::cube_face`.
fn face_basis(face: usize) -> (Vec3, Vec3, Vec3) {
    match face {
        0 => (Vec3::X, Vec3::NEG_Z, Vec3::NEG_Y),
        1 => (Vec3::NEG_X, Vec3::Z, Vec3::NEG_Y),
        2 => (Vec3::Y, Vec3::X, Vec3::Z),
        3 => (Vec3::NEG_Y, Vec3::X, Vec3::NEG_Z),
        4 => (Vec3::Z, Vec3::X, Vec3::NEG_Y),
        _ => (Vec3::NEG_Z, Vec3::NEG_X, Vec3::NEG_Y),
    }
}

/// The Rust twin of `locallight::cube_face`: `(face, uv, major)` for a direction from the light.
pub fn cube_face_of(v: Vec3) -> (usize, Vec2, f32) {
    let a = v.abs();
    let (face, major, st) = if a.x >= a.y && a.x >= a.z {
        if v.x > 0.0 {
            (0, a.x, Vec2::new(-v.z, -v.y))
        } else {
            (1, a.x, Vec2::new(v.z, -v.y))
        }
    } else if a.y >= a.z {
        if v.y > 0.0 {
            (2, a.y, Vec2::new(v.x, v.z))
        } else {
            (3, a.y, Vec2::new(v.x, -v.z))
        }
    } else if v.z > 0.0 {
        (4, a.z, Vec2::new(v.x, -v.y))
    } else {
        (5, a.z, Vec2::new(-v.x, -v.y))
    };
    (face, st / major.max(1e-6) * 0.5 + Vec2::splat(0.5), major)
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<ModernLights>()
        .add_plugins(ExtractResourcePlugin::<ModernLights>::default())
        // `Last`, never `PostUpdate` (see `HOOKS.md`, "Scheduling rule").
        .add_systems(Last, update_lights);
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        node::build(render_app);
    }
}

fn update_lights(
    active: Res<ActiveLighting>,
    cameras: Query<&GlobalTransform, With<WorldCamera>>,
    lights_q: Query<(&WorldPointLight, &GlobalTransform, Option<&LightRooms>)>,
    portals: Query<&crate::wmo_portal::WmoPortalInstance>,
    mut out: ResMut<ModernLights>,
    mut held: Local<Vec<ShadowLight>>,
) {
    let tier = light_tier(active.quality);
    if !active.is_modern() || !tier.on || !lights_enabled() {
        if out.on {
            *out = ModernLights::default();
            held.clear();
        }
        return;
    }
    let Some(camera) = cameras.iter().next() else {
        return;
    };
    let cam = camera.translation();

    // The shadowed set: the strongest few lights near the camera, under the same room gate the
    // packed table itself runs (`global_light::build_light_data`) — a torch in a culled room is not
    // in the table, so it must not hold an atlas slot either.
    let mut ranked: Vec<(f32, ShadowLight)> = Vec::new();
    if crate::modern::shadow::LOCAL_SHADOW_LAYERS > 0 && tier.shadowed > 0 {
        for (light, gt, rooms) in &lights_q {
            if !crate::wmo_portal::room_admits(
                rooms.map(|r| &r.0),
                rooms.and_then(|r| portals.get(r.0.instance).ok()),
            ) {
                continue;
            }
            let pos = gt.translation();
            let d = pos.distance(cam);
            if d > SHADOW_RADIUS || light.range <= 0.0 {
                continue;
            }
            // The light's reach as the frame sees it: its authored energy over the distance the
            // camera stands at, so a bright chandelier two rooms away loses to the candle beside it.
            let lum = light.intensity * (0.2126 * light.color[0] + 0.7152 * light.color[1] + 0.0722 * light.color[2]);
            ranked.push((lum / (1.0 + d), ShadowLight { pos, range: light.range }));
        }
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    }
    // **Sticky.** Re-ranking every frame would reshuffle the atlas as the camera walks, and a
    // reshuffle redraws all six faces of all six lights — a building's worth of geometry each. A
    // light that still qualifies keeps its slot until it leaves the radius outright, so a walk
    // through a lit room pays for the atlas once.
    let want = tier.shadowed.min(MAX_SHADOWED);
    let keep: Vec<ShadowLight> = held
        .iter()
        .copied()
        .filter(|h| ranked.iter().any(|(_, c)| *c == *h))
        .take(want)
        .collect();
    let mut chosen = keep;
    for (_, candidate) in &ranked {
        if chosen.len() >= want {
            break;
        }
        if !chosen.contains(candidate) {
            chosen.push(*candidate);
        }
    }
    *held = chosen.clone();

    *out = ModernLights {
        on: true,
        tier,
        base: grid_base(&tier, cam),
        shadowed: chosen,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_matches_the_wgsl_layout() {
        let mut bytes = Vec::new();
        LightParams::default().std430(&mut bytes);
        assert_eq!(bytes.len(), LIGHT_PARAM_BYTES);
        assert_eq!(LIGHT_PARAM_BYTES % 256, 0);
        assert_eq!(BLOCK_LIGHT_BYTES % 256, 0);
        // The probe region still starts 256-aligned with the light grid in front of it.
        let offset = crate::modern::shadow::CASCADES_BYTES
            + crate::modern::volumetrics::BLOCK_TAIL_BYTES
            + BLOCK_LIGHT_BYTES
            + crate::modern::gi::GI_PARAMS_BYTES;
        assert_eq!(offset % 256, 0, "probe offset {offset}");
    }

    #[test]
    fn every_tier_fits_the_allocation_and_the_atlas() {
        for q in [
            LightingQuality::Low,
            LightingQuality::Medium,
            LightingQuality::High,
            LightingQuality::Ultra,
        ] {
            let t = light_tier(q);
            assert!(t.cells() <= CELL_ALLOC, "{q:?}: {}", t.cells());
            assert!(t.shadowed <= MAX_SHADOWED, "{q:?}");
            assert!(
                t.shadowed as u32 * 6 <= ATLAS_TILES * ATLAS_TILES,
                "{q:?}: the atlas holds {} tiles",
                ATLAS_TILES * ATLAS_TILES
            );
        }
        assert!(!light_tier(LightingQuality::Low).on);
        assert_eq!(light_tier(LightingQuality::Medium).shadowed, 0);
    }

    /// The whole cube lookup rests on this: the matrix a face is *rendered* under and the analytic
    /// mapping a fragment is *sampled* with have to agree, or a lamp's shadow lands on the wrong
    /// wall. Nothing in the frame would say so — the depths still compare.
    #[test]
    fn a_face_matrix_lands_where_the_shader_looks() {
        let light = Vec3::new(12.0, 3.5, -40.0);
        for v in [
            Vec3::new(3.0, 0.4, 0.2),
            Vec3::new(-5.0, 1.0, -2.0),
            Vec3::new(0.3, 4.0, 1.1),
            Vec3::new(0.1, -2.0, 0.9),
            Vec3::new(1.0, 0.2, 6.0),
            Vec3::new(-0.7, 0.3, -3.0),
            Vec3::new(2.0, 1.9, 1.8),
        ] {
            let (face, uv, major) = cube_face_of(v);
            let clip = cube_face_clip(light, face);
            let p = clip * (light + v).extend(1.0);
            let ndc = p.xyz() / p.w;
            let from_matrix = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            assert!(
                (from_matrix - uv).abs().max_element() < 1e-5,
                "face {face}: matrix {from_matrix} vs shader {uv}"
            );
            // Reverse-Z: the stored depth is `near / major`, which is what the lookup compares.
            assert!((ndc.z - CUBE_NEAR / major).abs() < 1e-5, "face {face}: {ndc}");
            // Inside the face, never off it.
            assert!(uv.min_element() >= -1e-5 && uv.max_element() <= 1.0 + 1e-5, "{uv}");
        }
    }

    #[test]
    fn the_grid_follows_the_camera_by_whole_cells() {
        let tier = light_tier(LightingQuality::Medium);
        let base = grid_base(&tier, Vec3::new(100.0, 20.0, -40.0));
        assert_eq!(base, IVec3::new(12 - 16, 2 - 8, -5 - 16));
        let moved = grid_base(&tier, Vec3::new(104.0, 20.0, -40.0));
        assert!((moved - base).abs().max_element() <= 1);
    }
}
