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

pub mod emissive;
pub mod node;

/// Lights one cell can name. Mirrors `locallight.wgsl`'s `PER_CELL`.
///
/// Eight, and not more, because the cell is one `vec4<u32>`: two words of eight byte indices and
/// one of eight nibble slots. Widening it widens the grid's 864 KB region by the same factor for
/// every tier, so P8 buys its extra lights with a **finer** grid instead (`light_tier`) — more
/// cells of eight, not fewer cells of sixteen.
pub const PER_CELL: usize = 8;
/// Shadowed lights the atlas holds: six per layer ([`ATLAS_TILES`]² tiles of six faces each), and
/// at most [`crate::modern::shadow::MAX_LOCAL_SHADOW_LAYERS`] layers.
pub const MAX_SHADOWED: usize = 6 * crate::modern::shadow::MAX_LOCAL_SHADOW_LAYERS;
/// Shadowed lights one atlas layer holds.
pub const PER_ATLAS_LAYER: usize = 6;
/// Tiles per side in an atlas layer.
pub const ATLAS_TILES: u32 = 6;
/// Cells allocated in the block, whatever the session's tier — the largest any tier uses. Held at
/// one size so a quality change never re-creates the buffer every material binds.
pub const CELL_ALLOC: usize = 48 * 24 * 48;
/// Bytes of one cell (`vec4<u32>`).
pub const CELL_BYTES: usize = 16;
/// Size of [`LightParams::std430`] — `locallight.wgsl`'s `LightGrid` prefix. A multiple of 256, so
/// the GI block behind it keeps the 256-aligned probe offset the blend pass binds. P8 doubled the
/// shadowed set (6 → 12 rows) and added a filter row; the region was 512 B with 22 spare rows, so
/// it absorbed both without moving the probe offset a byte.
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
/// fork: P8 — how fast a cube's penumbra opens: face-uv texels of blur per yard between the light
/// and the receiver, at a nominal one-yard emitter. A brazier is about that wide; a sword's glow is
/// narrower and a chandelier wider, but the light table carries no emitter size, so one opening
/// stands for all of them and the tier's tap count decides how smooth it reads.
const PENUMBRA_OPENING: f32 = 0.9;
/// fork: P8 — the aggressive cull's screen test: a light whose lit sphere subtends less than this
/// fraction of the viewport's height gets no cube, however bright it is. A lamp 60 yd off across a
/// square is a dozen pixels of shading; six faces of depth for it is the cost of a building.
const MIN_SCREEN_FRACTION: f32 = 0.035;
/// fork: P8 — and how far outside the frustum a light may still sit and keep its cube. Not zero:
/// a lamp just off the left edge still lights the wall the camera IS looking at, and a hard frustum
/// test would pop its shadow on and off as the player turned. Measured in the light's own ranges.
const FRUSTUM_SLACK: f32 = 0.35;

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
    /// fork: P8 — filter taps on a cube lookup. `1` is P6's single hardware compare; more taps are
    /// a Vogel disc whose radius opens with the receiver's distance from the light (the penumbra a
    /// finite emitter would cast), which is what "soft" means here.
    pub taps: u32,
    /// fork: P8 — how much of a local light a shadow may take away. Below `1` the shadow is
    /// **faint**: a lamp's occluded side keeps some of it. This is not a cheat for a bad filter —
    /// a WoW point light stands in for a whole fixture (a brazier's fire, a sconce's flame and its
    /// bowl's bounce), so a hard zero behind a chair reads as a hole the reference never had. A
    /// glowing sword is the extreme case the user named: its shadow should be *there* and should
    /// not be a silhouette.
    pub shade: f32,
    /// fork: P8 — do entity casters (characters, creatures, the weapon in a hand) draw into the
    /// cube atlas? Costs the atlas its full cache: the layer must be restored and redrawn each
    /// frame, the same deal the cascades already make.
    pub dynamic_casters: bool,
    /// fork: P8 — run the geometric cull ([`ShadowCull`]) over the candidates for an atlas slot.
    ///
    /// It only ever *removes* work, but it also removes shadows a tier used to draw, so it is a
    /// tier row and not a global: P8 must not change what High and Ultra look like any more than it
    /// changes what they cost. Cinematic needs it because it is the tier with twelve slots to fill
    /// and a world full of newly-emitting lamps to fill them from.
    pub aggressive_cull: bool,
}

impl LightTier {
    pub const OFF: Self = Self {
        on: false,
        dims: UVec3::ONE,
        cell: 1.0,
        shadowed: 0,
        taps: 1,
        shade: 1.0,
        dynamic_casters: false,
        aggressive_cull: false,
    };

    pub fn cells(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize
    }

    /// Atlas layers the tier's shadowed set spans.
    pub fn atlas_layers(&self) -> usize {
        self.shadowed.div_ceil(PER_ATLAS_LAYER)
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
            ..LightTier::OFF
        },
        LightingQuality::High => LightTier {
            on: true,
            dims: UVec3::new(48, 24, 48),
            cell: 6.0,
            shadowed: 4,
            ..LightTier::OFF
        },
        // Untouched by P8, deliberately: one hardware compare, no penumbra, no shade floor, no
        // entity casters and no cull — the row P6 shipped, so an Ultra frame costs and looks exactly
        // what it did before this tier existed.
        LightingQuality::Ultra => LightTier {
            on: true,
            dims: UVec3::new(48, 24, 48),
            cell: 4.5,
            shadowed: PER_ATLAS_LAYER,
            ..LightTier::OFF
        },
        // fork: P8. The grid is the finest a 48³ allocation allows — 3 yd cells reach 144 yd across,
        // which is a city block, and at Cinematic that block holds every lamp post, every window
        // candle and every drawn weapon rather than the handful the reference commits. Twelve lights
        // shadow, over both atlas layers, with a 12-tap Vogel disc and entity casters in the cubes.
        LightingQuality::Cinematic => LightTier {
            on: true,
            dims: UVec3::new(48, 24, 48),
            cell: 3.0,
            shadowed: MAX_SHADOWED,
            taps: 12,
            shade: 0.90,
            dynamic_casters: true,
            aggressive_cull: true,
        },
    }
}

/// Is the local-light grid on in this session (`$WOW_MODERN_LIGHTS`, browser `?modern_lights=0`)?
pub fn lights_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MODERN_LIGHTS").as_deref() != Some("0"))
}

/// fork: P8 — **the aggressive cull**: the two geometric reasons to refuse a light its six faces of
/// depth, asked before the light is even ranked.
///
/// Both are about the *lit sphere*, not the light's point: a lamp behind the camera still lights the
/// wall in front of it, so the test is whether that sphere is visible, and it is deliberately
/// generous ([`FRUSTUM_SLACK`]) so a shadow does not pop as the player turns. What it does refuse,
/// hard, is a lamp on the far side of a square — [`MIN_SCREEN_FRACTION`] of the viewport is a dozen
/// pixels, and at that size a cube shadow is six 341² depth renders spent on something the eye
/// cannot resolve.
#[derive(Clone, Copy, Debug)]
pub struct ShadowCull {
    eye: Vec3,
    forward: Vec3,
    /// `tan(vertical fov / 2)` — how many yards of sphere one viewport height spans at one yard.
    tan_half_fov: f32,
}

impl ShadowCull {
    /// From the world camera. A camera with no `Projection` (the world view's bare setups, and every
    /// unit test) reads as Bevy's default perspective, whose vertical fov is π/4.
    pub fn new(camera: &GlobalTransform, projection: Option<&Projection>) -> Self {
        let fov = match projection {
            Some(Projection::Perspective(p)) => p.fov,
            // An orthographic or custom projection has no single fov; π/4 keeps the screen test's
            // scale honest rather than turning it off, and the frustum half still applies.
            _ => std::f32::consts::FRAC_PI_4,
        };
        Self {
            eye: camera.translation(),
            forward: camera.forward().as_vec3(),
            tan_half_fov: (fov * 0.5).tan().max(1e-3),
        }
    }

    /// Is a light at `pos` reaching `range` yards worth an atlas slot?
    pub fn worth_a_cube(&self, pos: Vec3, range: f32) -> bool {
        let to = pos - self.eye;
        let along = to.dot(self.forward);
        // Behind the eye by more than the slack the lit sphere allows: nothing it lights can be in
        // frame either, because the sphere itself is behind the near plane.
        if along < -(range * (1.0 + FRUSTUM_SLACK)) {
            return false;
        }
        // The screen test, measured at the sphere — a light's apparent size is the size of what it
        // lights, not of the emitter. In front of the eye the sphere spans `2·range` yards at depth
        // `along`, over a viewport spanning `2·along·tan(fov/2)`; behind it, `along` is small or
        // negative and the sphere is by definition large on screen, so the test passes.
        let depth = along.max(1e-3);
        let fraction = range / (depth * self.tan_half_fov);
        fraction >= MIN_SCREEN_FRACTION
    }
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
    /// **The atlas slots, by index, holes included.** Not a compacted list of the lights that got
    /// one: a light's slot NAMES the six atlas tiles its shadow was drawn into, and a fragment
    /// reaches those tiles through a nibble the binning pass wrote. So a slot index has to mean the
    /// same light from one frame to the next, or every fragment that reads slot `n` is handed some
    /// other lamp's depth and shades against the wrong room.
    ///
    /// That is exactly what a compacted `Vec` could not promise: one light leaving the radius
    /// shifted every light behind it up a slot, and the scene flipped between lit and black as the
    /// player turned. `None` is a slot no light holds — the atlas tiles for it are cleared, which
    /// reads as "nothing occludes" (reverse-Z clears to the far plane), so a hole is harmless.
    pub shadowed: Vec<Option<ShadowLight>>,
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
    /// fork: P8 — the soft-shadow filter: `x` taps, `y` the penumbra opening (texels of face uv per
    /// yard of receiver distance), `z` the shade floor (how much of a light a shadow may take),
    /// `w` the atlas's first layer. Named `soft` in the WGSL, where `filter` is a reserved keyword.
    pub soft: Vec4,
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
        row(self.soft);
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
    // The atlas is live when the tier HAS slots — a frame in which every slot happens to be empty
    // still binds the same layers, and each empty slot's tiles read as "nothing occludes".
    let atlas = lights.shadowed.iter().any(Option::is_some);
    let mut flags = 0u32;
    if lights.on {
        flags |= 1;
    }
    if atlas {
        flags |= 2;
    }
    // Written BY SLOT, holes and all: a hole stays `Vec4::ZERO`, whose `w` (the range) is 0, and
    // `light_bin.wgsl`'s `shadow_slot` skips a zero-range slot rather than matching a light that
    // happens to stand at the world origin against it.
    let mut shadow = [Vec4::ZERO; MAX_SHADOWED];
    for (slot, light) in lights.shadowed.iter().enumerate().take(MAX_SHADOWED) {
        if let Some(light) = light {
            shadow[slot] = light.pos.extend(light.range);
        }
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
        soft: Vec4::new(
            tier.taps as f32,
            PENUMBRA_OPENING,
            tier.shade,
            crate::modern::shadow::LOCAL_LAYER as f32,
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
    // fork: P8 — the camera's `Projection` as well as its transform: the aggressive cull needs the
    // frustum and the vertical field of view to answer "is this lamp worth six faces of depth".
    cameras: Query<(&GlobalTransform, Option<&Projection>), With<WorldCamera>>,
    lights_q: Query<(&WorldPointLight, &GlobalTransform, Option<&LightRooms>)>,
    portals: Query<&crate::wmo_portal::WmoPortalInstance>,
    mut out: ResMut<ModernLights>,
    mut held: Local<Vec<Option<ShadowLight>>>,
) {
    let tier = light_tier(active.quality);
    if !active.is_modern() || !tier.on || !lights_enabled() {
        if out.on {
            *out = ModernLights::default();
            held.clear();
        }
        return;
    }
    let Some((camera, projection)) = cameras.iter().next() else {
        return;
    };
    let cam = camera.translation();
    let cull = tier
        .aggressive_cull
        .then(|| ShadowCull::new(camera, projection));

    // The shadowed set: the strongest few lights near the camera, under the same room gate the
    // packed table itself runs (`global_light::build_light_data`) — a torch in a culled room is not
    // in the table, so it must not hold an atlas slot either.
    let mut ranked: Vec<(f32, ShadowLight)> = Vec::new();
    if crate::modern::shadow::local_shadow_layers(active.quality) > 0 && tier.shadowed > 0 {
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
    held.resize(want, None);
    held.truncate(want);
    assign_slots(&mut held, &ranked, &cull);
    let chosen = held.clone();

    *out = ModernLights {
        on: true,
        tier,
        base: grid_base(&tier, cam),
        shadowed: chosen,
    };
}

/// **Slots are stable.** A light keeps its index for as long as it is still a candidate at all —
/// still in the table, still admitted by its room, still inside [`SHADOW_RADIUS`]. Nothing else
/// evicts it: not a better light arriving, and (since the flicker this caused) not the P8 cull.
///
/// The cull's job is to stop a lamp ACQUIRING six faces of depth it cannot pay back, so it is asked
/// once, when a slot is being filled. Asking it every frame of a light's tenure instead made the
/// whole set view-dependent — turn away from a lamp and it lost its slot, turn back and it took a
/// different one — and since a slot NAMES six atlas tiles, and a fragment reaches those tiles
/// through a nibble the binning pass wrote, every reshuffle pointed live fragments at another
/// lamp's shadow. In a room with several lamps that reads as the scene flipping between lit and
/// pitch black as the player turns, which is exactly what it did.
fn assign_slots(
    held: &mut [Option<ShadowLight>],
    ranked: &[(f32, ShadowLight)],
    cull: &Option<ShadowCull>,
) {
    for slot in held.iter_mut() {
        if let Some(light) = *slot {
            if !ranked.iter().any(|(_, c)| *c == light) {
                *slot = None;
            }
        }
    }
    // Fill the holes, strongest candidate first, and only with lights the cull says are worth it.
    for (_, candidate) in ranked
        .iter()
        .filter(|(_, c)| cull.is_none_or(|k| k.worth_a_cube(c.pos, c.range)))
    {
        if held.iter().flatten().any(|h| h == candidate) {
            continue;
        }
        let Some(hole) = held.iter().position(Option::is_none) else {
            break;
        };
        held[hole] = Some(*candidate);
    }
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
        for q in LightingQuality::ALL {
            let t = light_tier(q);
            assert!(t.cells() <= CELL_ALLOC, "{q:?}: {}", t.cells());
            assert!(t.shadowed <= MAX_SHADOWED, "{q:?}");
            // Every shadowed light must have six tiles somewhere in the layers the tier allocates.
            let layers = crate::modern::shadow::local_shadow_layers(q);
            assert!(
                t.atlas_layers() <= layers,
                "{q:?}: {} shadowed lights need {} layer(s), the tier allocates {layers}",
                t.shadowed,
                t.atlas_layers(),
            );
            assert!(
                t.shadowed.div_ceil(PER_ATLAS_LAYER) * PER_ATLAS_LAYER
                    <= layers * PER_ATLAS_LAYER,
                "{q:?}"
            );
            assert!(t.taps <= 12, "{q:?}: the shader's MAX_TAPS is 12");
            assert!((0.0..=1.0).contains(&t.shade), "{q:?}");
        }
        assert!(!light_tier(LightingQuality::Low).on);
        assert_eq!(light_tier(LightingQuality::Medium).shadowed, 0);
        assert_eq!(PER_ATLAS_LAYER as u32 * 6, ATLAS_TILES * ATLAS_TILES);
    }

    /// fork: P8 — **the no-regression gate.** The tier rows that existed before P8 must be the rows
    /// they were: one hardware compare, no penumbra, no shade floor, no entity casters in the cubes
    /// and no cull. Everything P8 costs has to be spent on P8's own tier, and this is what says so —
    /// a `taps: 2` typed into the Ultra row would otherwise double a lookup every lit fragment in
    /// every lamp-lit room makes, and nothing in the frame would announce it.
    #[test]
    fn p8_costs_nothing_on_the_tiers_that_shipped_before_it() {
        for q in [
            LightingQuality::Low,
            LightingQuality::Medium,
            LightingQuality::High,
            LightingQuality::Ultra,
        ] {
            let t = light_tier(q);
            assert_eq!(t.taps, 1, "{q:?} must take P6's single hardware compare");
            assert_eq!(t.shade, 1.0, "{q:?} must have no shade floor");
            assert!(!t.dynamic_casters, "{q:?} must keep its fully cached atlas");
            assert!(!t.aggressive_cull, "{q:?} must choose its lights as it did");
            assert!(t.shadowed <= PER_ATLAS_LAYER, "{q:?}: one atlas layer at most");
            // And no tier that shipped may allocate a layer it did not allocate before: one
            // natively above Medium, none at all in the browser.
            let want = match (q, cfg!(target_arch = "wasm32")) {
                (LightingQuality::Low | LightingQuality::Medium, _) | (_, true) => 0,
                _ => 1,
            };
            assert_eq!(
                crate::modern::shadow::local_shadow_layers(q),
                want,
                "{q:?}"
            );
        }
        // …and the tier that is new is allowed to cost what it says it costs.
        let cine = light_tier(LightingQuality::Cinematic);
        assert!(cine.taps > 1 && cine.dynamic_casters && cine.aggressive_cull);
        assert_eq!(
            crate::modern::shadow::local_shadow_layers(LightingQuality::Cinematic),
            crate::modern::shadow::MAX_LOCAL_SHADOW_LAYERS
        );
    }

    fn lamp(x: f32) -> ShadowLight {
        ShadowLight {
            pos: Vec3::new(x, 0.0, 0.0),
            range: 20.0,
        }
    }

    fn ranked(lamps: &[ShadowLight]) -> Vec<(f32, ShadowLight)> {
        lamps
            .iter()
            .enumerate()
            .map(|(i, l)| (100.0 - i as f32, *l))
            .collect()
    }

    /// GOLDEN — **a light keeps its slot.** This is the fix for the bug the user saw: in Northshire
    /// Abbey the scene flipped between lit and pitch black depending on which way they looked.
    ///
    /// A slot NAMES the six atlas tiles a lamp's shadow was drawn into, and every lit fragment
    /// reaches those tiles through a nibble the binning pass wrote from the slot table. The set used
    /// to be a compacted `Vec`, so one light dropping out shifted every light behind it up a slot —
    /// and every fragment that had been reading slot 3 was suddenly handed slot 3's *new* occupant's
    /// depth, shading a room against a lamp somewhere else entirely.
    #[test]
    fn a_light_keeps_its_slot_when_another_one_leaves() {
        let (a, b, c) = (lamp(0.0), lamp(10.0), lamp(20.0));
        let mut held = vec![None; 4];
        assign_slots(&mut held, &ranked(&[a, b, c]), &None);
        assert_eq!(held, vec![Some(a), Some(b), Some(c), None]);

        // `b` leaves the radius — it is simply not a candidate any more.
        assign_slots(&mut held, &ranked(&[a, c]), &None);
        assert_eq!(
            held,
            vec![Some(a), None, Some(c), None],
            "a and c must not move; b's slot becomes a hole"
        );

        // A new lamp takes the HOLE, not the end, and still nothing else moves.
        let d = lamp(30.0);
        assign_slots(&mut held, &ranked(&[a, c, d]), &None);
        assert_eq!(held, vec![Some(a), Some(d), Some(c), None]);
    }

    /// The cull gates ACQUISITION only. A held light that the camera turns away from keeps its
    /// slot — asking the cull again every frame is what made the set view-dependent, and a
    /// view-dependent set of slots is the flicker above.
    #[test]
    fn the_cull_never_evicts_a_light_it_already_let_in() {
        let facing = ShadowCull {
            eye: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            tan_half_fov: (std::f32::consts::FRAC_PI_4 * 0.5).tan(),
        };
        // Far enough down the view axis that turning around puts its whole lit sphere well behind
        // the eye — a lamp merely a few yards behind you keeps its slot on the cull's own terms
        // (`FRUSTUM_SLACK`), which is a different rule and is tested below.
        let near = ShadowLight {
            pos: Vec3::new(0.0, 0.0, -40.0),
            range: 12.0,
        };
        let mut held = vec![None; 2];
        assign_slots(&mut held, &ranked(&[near]), &Some(facing));
        assert_eq!(held[0], Some(near), "in front and close: it gets a slot");

        // Now the player turns right around. The lamp is behind them and would never be admitted
        // fresh — but it is still in range, so it keeps what it has.
        let turned = ShadowCull {
            forward: Vec3::Z,
            ..facing
        };
        assert!(
            !turned.worth_a_cube(near.pos, near.range),
            "precondition: the cull would refuse it now"
        );
        assign_slots(&mut held, &ranked(&[near]), &Some(turned));
        assert_eq!(held[0], Some(near), "but it keeps the slot it already holds");

        // It only goes when it stops being a candidate at all (out of range, or room-culled).
        assign_slots(&mut held, &[], &Some(turned));
        assert_eq!(held[0], None);
    }

    /// fork: P8 — the aggressive cull's two refusals, and the one thing it must NOT refuse.
    #[test]
    fn the_cull_refuses_the_small_and_the_far_behind() {
        let cull = ShadowCull {
            eye: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            tan_half_fov: (std::f32::consts::FRAC_PI_4 * 0.5).tan(),
        };
        // A lamp right in front, close: obviously worth it.
        assert!(cull.worth_a_cube(Vec3::new(0.0, 0.0, -8.0), 12.0));
        // The same lamp far enough away that its whole lit sphere is a sliver of the viewport.
        assert!(!cull.worth_a_cube(Vec3::new(0.0, 0.0, -900.0), 12.0));
        // Well behind the eye, and its sphere with it: nothing it lights can be in frame.
        assert!(!cull.worth_a_cube(Vec3::new(0.0, 0.0, 40.0), 12.0));
        // But a lamp just behind the eye keeps its cube — it still lights the wall the camera IS
        // looking at, and a hard test here pops a shadow on and off as the player turns.
        assert!(cull.worth_a_cube(Vec3::new(0.0, 0.0, 6.0), 12.0));
        // Off to the side, well within its own reach of the view axis: kept, same reason.
        assert!(cull.worth_a_cube(Vec3::new(9.0, 0.0, -10.0), 12.0));
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
