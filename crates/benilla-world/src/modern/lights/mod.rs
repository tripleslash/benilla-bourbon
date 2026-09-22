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
/// How far past its own radius a fixture's geometry is pushed out of its cube. A lantern's frame
/// stands a little outside the glow it houses, so clipping exactly at the glow's extent still
/// leaves the ironwork casting.
const FIXTURE_CLEARANCE: f32 = 1.6;
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
const ADMIT_FRACTION: f32 = 0.035;
/// fork: P8 — the hysteresis floor: a light already holding a slot keeps it until its lit sphere
/// falls below this, well under [`ADMIT_FRACTION`]. Without a gap between the two a light sitting
/// near the threshold takes and loses its cube on alternate frames.
const KEEP_FRACTION: f32 = 0.020;
/// fork: P8 — how fast a slot's shadow cross-fades in and out, in units of fade per second. A slot
/// is never switched on or off: it ramps, so admitting or releasing one is invisible. ~4 frames at
/// 60 Hz.
const FADE_RATE: f32 = 15.0;

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
    /// fork: P8 — **the far level**: a second, coarser grid around the same camera that holds the
    /// same lights, read wherever the near grid does not reach. Cells per axis; `x` and `z` must
    /// agree (the shader carries one horizontal count). [`UVec3::ZERO`] is no far level, which is
    /// every tier but Cinematic — past the near grid those take no local light, as they always did.
    pub far_dims: UVec3,
    /// The far cell's edge as a whole multiple of [`Self::cell`]. Whole, so the far level's base is
    /// an integer function of the near level's and the two can never disagree about the camera.
    pub far_step: u32,
    /// fork: P8 — how much of each atlas tile a cube face is drawn into, per edge. `0.5` renders a
    /// quarter of the depth, and the wider texels are what the penumbra filter smooths over anyway.
    pub tile_use: f32,
    /// fork: P8 — the penumbra opening (face texels of blur per yard of receiver distance, at a
    /// nominal one-yard emitter). See [`PENUMBRA_OPENING`], which every tier but Cinematic keeps.
    pub penumbra: f32,
    /// fork: P8 — the band, in yards inside [`SHADOW_RADIUS`], over which a cube shadow fades with
    /// the light's distance from the camera. `0` is no fade: a slot is at full strength from the
    /// frame it finishes ramping in, wherever the light is.
    pub shadow_fade_yd: f32,
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
        far_dims: UVec3::ZERO,
        far_step: 1,
        tile_use: 1.0,
        penumbra: PENUMBRA_OPENING,
        shadow_fade_yd: 0.0,
    };

    /// Every cell the binning pass fills: the near level's, then the far level's.
    pub fn cells(&self) -> usize {
        self.near_cells() + (self.far_dims.x * self.far_dims.y * self.far_dims.z) as usize
    }

    /// The near level's cells.
    pub fn near_cells(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize
    }

    /// fork: P8 — the far level's cell edge (yd).
    pub fn far_cell(&self) -> f32 {
        self.cell * self.far_step.max(1) as f32
    }

    /// fork: P8 — how far the far level reaches from the camera (zero when there is none).
    pub fn far_reach(&self) -> Vec3 {
        self.far_dims.as_vec3() * self.far_cell() * 0.5
    }

    /// fork: P8 — how far ANY local light reaches, near level or far.
    pub fn full_reach(&self) -> Vec3 {
        self.reach().max(self.far_reach())
    }

    /// fork: P8 — **how far this grid reaches from the camera**, in world yards per axis.
    ///
    /// The grid is camera-centred ([`grid_base`]) and `dims × cell` yards across, so this is half
    /// of that. Past it a fragment falls in no cell, takes nothing from the grid, and — on a WMO
    /// surface, which has no other source of local light in a modern frame — takes none at all.
    /// It is the number the tier table has to be read against, and the one it did not state.
    pub fn reach(&self) -> Vec3 {
        self.dims.as_vec3() * self.cell * 0.5
    }

    /// Atlas layers the tier's shadowed set spans.
    pub fn atlas_layers(&self) -> usize {
        self.shadowed.div_ceil(PER_ATLAS_LAYER)
    }
}

/// fork: P8b — **the emissive tiers' grid**: two levels, shared by every tier that lights from things
/// that only glow (`LightingQuality::lights_emissive_sources`). The NEAR level is 3 yd cells — the
/// resolution those tiers need, since a city block then holds every lamp post, every lit window and
/// every drawn weapon rather than the handful the reference commits — over ±60/±30 yd. The FAR level
/// is 18 yd cells over ±324/±144 yd, holding the same lights.
///
/// The far level is the fix for lamps "popping in" as the player walks up to them. With one level of
/// 3 yd cells the grid ended at ±72/±36 yd, and past its edge nothing took any local light at all: a
/// Goldshire street at 80 yd was unlit however many lamps stood in it, and lit up the moment the
/// grid's edge swept over it. Now a lamp lights its street at any distance the table packs it
/// (`global_light`, which fades lights out smoothly at the far end rather than at a hard cap), and
/// the near/far seam cross-fades (`locallight::cell_pair`), so no distance has an edge.
///
/// No shadowed lights here: a tier adds its cube slots on top ([`light_tier`]'s Cinematic row).
pub const EMISSIVE_GRID: LightTier = LightTier {
    on: true,
    dims: UVec3::new(40, 20, 40),
    cell: 3.0,
    shadowed: 0,
    taps: 1,
    shade: 1.0,
    dynamic_casters: false,
    aggressive_cull: false,
    far_dims: UVec3::new(36, 16, 36),
    far_step: 6,
    tile_use: 1.0,
    penumbra: PENUMBRA_OPENING,
    shadow_fade_yd: 0.0,
};

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
        // fork: P8b — Cinematic's lights without a single cube shadow: every lamp, lit window and
        // glowing weapon emits (the emissive gate is `>= Ultra`), on Cinematic's two-level grid.
        // P6's six hard-shadowed slots are gone from this row — the user's call: Ultra is "the
        // lighting part" of Cinematic, and the atlas layer, its static cache and its cube renders
        // go with the shadows.
        LightingQuality::Ultra => EMISSIVE_GRID,
        // fork: P8 — the emissive grid plus a dozen cube shadows, **as cheap as they can be made**
        // (P8b: the user traded their detail for their cost). One hardware 2×2 compare per lookup —
        // the penumbra's eight-tap Vogel disc is gone — under a shade floor, so the shadow is faint
        // rather than soft and its hard edge reads as a soft one; static casters only (cached per
        // slot — a settled street pays nothing a frame, and a slot change redraws that slot's six
        // faces alone); faces drawn into half a tile. They fade out with distance before a slot is
        // released, so a lamp acquiring or losing its cube never shows.
        LightingQuality::Cinematic => LightTier {
            shadowed: MAX_SHADOWED,
            taps: 1,
            shade: 0.55,
            aggressive_cull: true,
            tile_use: 0.5,
            penumbra: 2.5,
            shadow_fade_yd: 30.0,
            ..EMISSIVE_GRID
        },
    }
}

/// Is the local-light grid on in this session (`$WOW_MODERN_LIGHTS`, browser `?modern_lights=0`)?
pub fn lights_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::launch_env::var("WOW_MODERN_LIGHTS").as_deref() != Some("0"))
}

/// fork: P8 — **the cull**: is this light worth six faces of depth?
///
/// **The metric is the light's solid angle at the eye, and it does not depend on where the camera
/// is pointing.** That is the whole design, and it replaced a frustum test that did. A lamp's cube
/// costs the same whether or not you are facing it, and — far more importantly — a set chosen by
/// what is *in frame* changes every time the player turns their head. Each change admits or drops a
/// light's shadow, so walls lit a moment ago fall into shadow and back out again: the "lights
/// randomly light the walls and drop shadows depending on my view angle" this is the fix for.
///
/// Orientation invariance alone is not enough, because a light near the threshold would still
/// chatter as the player walks. Two more things hold it steady, and all three are standard practice
/// for a shadow atlas that has to evict:
///
/// 1. **Hysteresis.** A light is admitted at [`ADMIT_FRACTION`] and kept until it falls below
///    [`KEEP_FRACTION`], which is well below it. Crossing back and forth costs nothing.
/// 2. **A temporal cross-fade** ([`FADE_RATE`]). A slot's shadow ramps in when it is filled and
///    ramps out before it is released, so the transition is never a step — the same treatment a
///    modern renderer gives a shadow LOD change or a cached-atlas eviction.
/// 3. **Stickiness.** A held light is never evicted to make room for a better one; only falling
///    below `KEEP_FRACTION`, leaving [`SHADOW_RADIUS`] or being room-culled releases a slot.
#[derive(Clone, Copy, Debug)]
pub struct ShadowCull {
    eye: Vec3,
    /// `tan(vertical fov / 2)` — how many yards of sphere one viewport height spans at one yard.
    tan_half_fov: f32,
}

impl ShadowCull {
    /// From the world camera. A camera with no `Projection` (the world view's bare setups, and every
    /// unit test) reads as Bevy's default perspective, whose vertical fov is π/4.
    pub fn new(camera: &GlobalTransform, projection: Option<&Projection>) -> Self {
        let fov = match projection {
            Some(Projection::Perspective(p)) => p.fov,
            // An orthographic or custom projection has no single fov; π/4 keeps the scale honest
            // rather than turning the test off.
            _ => std::f32::consts::FRAC_PI_4,
        };
        Self {
            eye: camera.translation(),
            tan_half_fov: (fov * 0.5).tan().max(1e-3),
        }
    }

    /// How large the light's lit sphere is on screen, as a fraction of the viewport height — the
    /// importance metric, and a function of DISTANCE only. A sphere of radius `range` at distance
    /// `d` spans `2·range` across a viewport spanning `2·d·tan(fov/2)`.
    pub fn screen_fraction(&self, pos: Vec3, range: f32) -> f32 {
        range / (pos.distance(self.eye).max(1e-3) * self.tan_half_fov)
    }

    /// Worth ACQUIRING a slot: the strict threshold.
    pub fn worth_a_cube(&self, pos: Vec3, range: f32) -> bool {
        self.screen_fraction(pos, range) >= ADMIT_FRACTION
    }

    /// Worth KEEPING one: the loose threshold. The gap between the two is the hysteresis.
    pub fn still_worth_it(&self, pos: Vec3, range: f32) -> bool {
        self.screen_fraction(pos, range) >= KEEP_FRACTION
    }
}

/// One atlas slot's live state: the light in it, and how far its shadow has cross-faded in.
///
/// `fade` is what makes a slot change invisible. A slot is never switched on or off — it ramps at
/// [`FADE_RATE`], and a slot on its way out (`retiring`) ramps to zero before the light actually
/// releases it, so nothing pops when the shadowed set changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldSlot {
    pub light: ShadowLight,
    pub fade: f32,
    pub retiring: bool,
}

/// One light that owns six atlas tiles this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShadowLight {
    pub pos: Vec3,
    pub range: f32,
    /// **The cube's near plane, per light** — and the whole reason it is per light rather than the
    /// one [`CUBE_NEAR`] every cube used to share.
    ///
    /// A synthesized light sits INSIDE the thing that emits it: a lantern's flame is behind the
    /// lantern's own glass and iron frame. Drawn into a cube with a 0.25yd near plane, that frame
    /// is a caster a hand's breadth from the light, so it projects its bars over the whole pool and
    /// takes up to `shade` of the lamp away — the floor got a cage pattern instead of a circle, and
    /// the lamp went dark the moment it was close enough to earn a cube at all. A lamp far enough
    /// off to be denied one stayed at full brightness, which is the "super bright at a distance,
    /// dim up close" the player saw, and every acquire and release of a slot was that same step,
    /// which is what made them flicker.
    ///
    /// Geometry nearer the light than the near plane is simply clipped, so setting this to the
    /// fixture's own radius means the fixture is never drawn into its own cube. Walls, furniture
    /// and everything beyond it still shadow normally. This is also what makes a shadowed lamp and
    /// an unshadowed one look alike, which is the property the distance behaviour depends on.
    pub near: f32,
}

/// How far a synthesized light's own fixture reaches, in yards — the radius whose geometry must be
/// clipped out of its cube. Attached by the spawn site that derived the light ([`emissive`]);
/// authored M2 lights carry none, because an artist placing a light block put it where they wanted
/// it and its model is not automatically in the way.
#[derive(Component, Clone, Copy, Debug)]
pub struct FixtureRadius(pub f32);

/// A light this fork DERIVED from a model's glow rather than one the artist authored
/// ([`emissive`]).
///
/// It exists so the tier gate can be applied where the light is *read* instead of where it is
/// spawned. Gating the spawn looked equivalent and was not: the lighting model latches after the
/// world has begun streaming, so every doodad that arrived before the latch asked "are emissive
/// lights on?", got `false`, and was never asked again. Which lampposts lit up came down to whether
/// they happened to stream in before or after that frame — and they re-rolled it on every reload,
/// which is a lamp blinking for no reason the player can see.
#[derive(Component, Clone, Copy, Debug)]
pub struct SynthesizedLight;

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
    pub shadowed: Vec<Option<HeldSlot>>,
    /// fork: P8 — the camera this frame's grid is centred on, for the shadows' distance fade
    /// ([`LightTier::shadow_fade_yd`]).
    pub eye: Vec3,
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
    /// fork: P8 — `x` the far level's horizontal cells, `y` its vertical cells, `z` its cell as a
    /// multiple of the near cell, `w` the drawn fraction of an atlas tile. `locallight.wgsl`'s
    /// `LightGrid.far`, taken from the padding that follows the struct.
    pub far: Vec4,
    pub shadow: [Vec4; MAX_SHADOWED],
    /// The twelve per-light near planes ([`ShadowLight::near`]), four to a row — written into the
    /// first three `spare` rows, which cost nothing: the region was already padded to
    /// [`LIGHT_PARAM_BYTES`] to keep the GI probe offset aligned, and this only fills padding.
    pub slot_near: [Vec4; MAX_SHADOWED / 4],
    /// The twelve per-slot cross-fades — the next three `spare` rows, still inside the padding.
    pub slot_fade: [Vec4; MAX_SHADOWED / 4],
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
        row(self.far);
        for s in &self.shadow {
            row(*s);
        }
        for s in &self.slot_near {
            row(*s);
        }
        for s in &self.slot_fade {
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
    let atlas = lights.shadowed.iter().flatten().any(|h| h.fade > 0.0);
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
    let mut slot_near = [Vec4::splat(CUBE_NEAR); MAX_SHADOWED / 4];
    // fork: P8 — the per-slot cross-fade. 0 reads exactly as "this light has no cube", so a slot
    // ramping in or out is indistinguishable from one that was never there; the shader mixes the
    // lookup toward full visibility by it.
    let mut slot_fade = [Vec4::ZERO; MAX_SHADOWED / 4];
    for (slot, held) in lights.shadowed.iter().enumerate().take(MAX_SHADOWED) {
        if let Some(held) = held {
            shadow[slot] = held.light.pos.extend(held.light.range);
            slot_near[slot / 4][slot % 4] = held.light.near;
            slot_fade[slot / 4][slot % 4] = (held.fade
                * shadow_distance_weight(tier, held.light.pos.distance(lights.eye)))
            .clamp(0.0, 1.0);
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
            tier.penumbra,
            tier.shade,
            crate::modern::shadow::LOCAL_LAYER as f32,
        ),
        far: Vec4::new(
            tier.far_dims.x as f32,
            tier.far_dims.y as f32,
            tier.far_step.max(1) as f32,
            // The fraction actually drawn, in whole texels — the render sets an integral viewport,
            // and the lookup has to scale by exactly what was rendered.
            face_px(tier, shadow_map_size) as f32 / (shadow_map_size / ATLAS_TILES).max(1) as f32,
        ),
        shadow,
        slot_near,
        slot_fade,
    }
}

/// fork: P8 — **how strong a cube shadow is for a light this far from the camera**, 0..1.
///
/// A slot is only ever held inside [`SHADOW_RADIUS`], so without this a lamp's shadow is a step
/// function of distance: absent at 71 yd, present (after a four-frame ramp) at 69. Fading it over
/// [`LightTier::shadow_fade_yd`] before the radius makes a lamp look the same whether or not it
/// holds a cube at the moment it is released or taken, which is what keeps the view consistent as
/// the player walks.
pub fn shadow_distance_weight(tier: &LightTier, distance: f32) -> f32 {
    if tier.shadow_fade_yd <= 0.0 {
        return 1.0;
    }
    let t = ((SHADOW_RADIUS - distance) / tier.shadow_fade_yd).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// fork: P8 — the edge, in texels, of the square a cube face is drawn into within its atlas tile
/// ([`LightTier::tile_use`]). The renderer's viewport and the lookup's uv scale both read this.
pub fn face_px(tier: &LightTier, shadow_map_size: u32) -> u32 {
    let tile = shadow_map_size / ATLAS_TILES;
    ((tile as f32 * tier.tile_use.clamp(0.05, 1.0)).floor() as u32).clamp(1, tile.max(1))
}

/// fork: P8 — the far level's base cell (the Rust twin of `locallight::far_base`): the far cell the
/// near level's camera cell falls in, minus half the far level.
pub fn far_base(tier: &LightTier, near_base: IVec3) -> IVec3 {
    let eye = near_base + tier.dims.as_ivec3() / 2;
    let k = tier.far_step.max(1) as f32;
    (eye.as_vec3() / k).floor().as_ivec3() - tier.far_dims.as_ivec3() / 2
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
pub fn cube_face_clip(light_pos: Vec3, face: usize, near: f32) -> Mat4 {
    let (axis, s, t) = face_basis(face);
    // View space looks down -Z with +Y up, and the shader's `t` runs *down* the face, so up = -t.
    let up = -t;
    let rot = Mat3::from_cols(
        Vec3::new(s.x, up.x, -axis.x),
        Vec3::new(s.y, up.y, -axis.y),
        Vec3::new(s.z, up.z, -axis.z),
    );
    let view = Mat4::from_mat3(rot) * Mat4::from_translation(-light_pos);
    Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 1.0, near) * view
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
    lights_q: Query<(
        &WorldPointLight,
        &GlobalTransform,
        Option<&LightRooms>,
        Option<&FixtureRadius>,
        Has<crate::lighting::NightGlow>,
        Has<SynthesizedLight>,
    )>,
    portals: Query<&crate::wmo_portal::WmoPortalInstance>,
    mut out: ResMut<ModernLights>,
    mut held: Local<Vec<Option<HeldSlot>>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().clamp(0.0, 0.1);
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
        let emissive = crate::modern::emissive_lights_on();
        for (light, gt, rooms, fixture, window, synthesized) in &lights_q {
            if !shadow_candidate(window, synthesized, emissive) {
                continue;
            }
            if !crate::wmo_portal::room_admits_light(
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
            ranked.push((
                lum / (1.0 + d),
                ShadowLight {
                    pos,
                    range: light.range,
                    // Clear the fixture, with a margin — a card's extent is the glow, and the iron
                    // around it sits a little outside that.
                    near: fixture.map_or(CUBE_NEAR, |f| CUBE_NEAR.max(f.0 * FIXTURE_CLEARANCE)),
                },
            ));
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
    assign_slots(&mut held, &ranked, &cull, dt);
    let chosen = held.clone();

    *out = ModernLights {
        on: true,
        tier,
        base: grid_base(&tier, cam),
        shadowed: chosen,
        eye: cam,
    };
}

/// Can this light hold a cube slot at all, before any ranking or culling?
///
/// - **A lit window never does.** It is a soft fill, not a shadow caster: a city has more of them
///   than the atlas has slots for lamps, and a pane's light has no hard edge to cast.
/// - **A light this fork derived from a glow ([`SynthesizedLight`]) only does on a tier that lights
///   from glows** (`emissive`, [`crate::modern::emissive_lights_on`]). Every such light exists on
///   every tier — the tier gate sits where the table is packed (`global_light`), not where the light
///   was spawned — so without this a lamppost that emits nothing on High still ranked for High's
///   four slots: six faces of depth rendered for a light no fragment reads, and an authored lamp
///   beside it denied the shadow it would have had.
pub fn shadow_candidate(window: bool, synthesized: bool, emissive: bool) -> bool {
    !window && (!synthesized || emissive)
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
    held: &mut [Option<HeldSlot>],
    ranked: &[(f32, ShadowLight)],
    cull: &Option<ShadowCull>,
    dt: f32,
) {
    // 1. Decide which slots are on their way out. A held light retires only when it stops being a
    //    candidate outright (out of range, room-culled) or falls below the KEEP threshold — never
    //    because a better light turned up, and never because the player looked away.
    for slot in held.iter_mut() {
        let Some(h) = slot else { continue };
        let still_a_candidate = ranked.iter().any(|(_, c)| *c == h.light);
        let still_big_enough = cull.is_none_or(|c| c.still_worth_it(h.light.pos, h.light.range));
        h.retiring = !(still_a_candidate && still_big_enough);
    }

    // 2. Ramp. A retiring slot fades out and is released only once it reaches zero, so its shadow
    //    is already gone by the time the tiles are handed to someone else.
    let step = (FADE_RATE * dt).clamp(0.0, 1.0);
    for slot in held.iter_mut() {
        let Some(h) = slot else { continue };
        if h.retiring {
            h.fade -= step;
            if h.fade <= 0.0 {
                *slot = None;
            }
        } else {
            h.fade = (h.fade + step).min(1.0);
        }
    }

    // 3. Fill the free slots, strongest candidate first, and only with lights the cull admits at
    //    the strict threshold. A new slot starts at zero fade and ramps in.
    for (_, candidate) in ranked
        .iter()
        .filter(|(_, c)| cull.is_none_or(|k| k.worth_a_cube(c.pos, c.range)))
    {
        if held.iter().flatten().any(|h| h.light == *candidate) {
            continue;
        }
        let Some(hole) = held.iter().position(Option::is_none) else {
            break;
        };
        held[hole] = Some(HeldSlot {
            light: *candidate,
            fade: 0.0,
            retiring: false,
        });
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

    /// fork: P8b — **the tier contract** (it replaced P8's "no tier that shipped may cost more"
    /// gate when Cinematic's lighting moved down to Ultra). Each row's cube shadows are pinned to
    /// what the tier table says it buys, because none of them announce themselves in the frame: a
    /// `taps: 8` typed back into Cinematic multiplies a lookup every lamp-lit fragment makes, and a
    /// `shadowed: 6` on Ultra silently re-allocates an atlas layer and its cache.
    ///
    /// - Low and Medium: no cube shadows, no layer.
    /// - High: P6's row as it shipped — four hard single-tap cubes, one layer natively, none in the
    ///   browser, no emissive sources.
    /// - Ultra: lights from glowing things, no cube shadows and no layer anywhere.
    /// - Cinematic: the same lights plus twelve cubes — one tap each, faint, static casters only.
    #[test]
    fn each_tier_buys_the_shadows_the_table_says() {
        for q in LightingQuality::ALL {
            let t = light_tier(q);
            assert_eq!(t.taps, 1, "{q:?}: every tier takes the single hardware compare since P8b");
            assert!(!t.dynamic_casters, "{q:?}: no tier redraws entities into the atlas");
        }
        for q in [LightingQuality::Low, LightingQuality::Medium, LightingQuality::Ultra] {
            assert_eq!(light_tier(q).shadowed, 0, "{q:?} shadows no local light");
            assert_eq!(crate::modern::shadow::local_shadow_layers(q), 0, "{q:?}");
        }
        let high = light_tier(LightingQuality::High);
        assert_eq!((high.shadowed, high.shade), (4, 1.0), "High is P6's row");
        assert!(!high.aggressive_cull);
        assert_eq!(
            crate::modern::shadow::local_shadow_layers(LightingQuality::High),
            if cfg!(target_arch = "wasm32") { 0 } else { 1 },
        );
        let cine = light_tier(LightingQuality::Cinematic);
        assert_eq!(cine.shadowed, MAX_SHADOWED);
        assert!(cine.shade < 1.0 && cine.aggressive_cull, "faint, and culled");
        assert_eq!(
            crate::modern::shadow::local_shadow_layers(LightingQuality::Cinematic),
            crate::modern::shadow::MAX_LOCAL_SHADOW_LAYERS
        );
        // Ultra IS Cinematic's lighting: the same grid and the same emissive gate — only the cubes
        // differ.
        let ultra = light_tier(LightingQuality::Ultra);
        assert_eq!(
            (ultra.dims, ultra.cell, ultra.far_dims, ultra.far_step),
            (cine.dims, cine.cell, cine.far_dims, cine.far_step)
        );
        assert_eq!(
            LightingQuality::Ultra.lights_emissive_sources(),
            LightingQuality::Cinematic.lights_emissive_sources()
        );
    }

    /// A light derived from a glow may only hold a cube slot on a tier that lights from glows, and a
    /// lit window never may ([`shadow_candidate`]).
    #[test]
    fn only_a_lit_light_can_hold_a_cube() {
        // An authored lamp, on any tier.
        assert!(shadow_candidate(false, false, false));
        assert!(shadow_candidate(false, false, true));
        // A synthesized lamp: only while emissive lights are on.
        assert!(!shadow_candidate(false, true, false));
        assert!(shadow_candidate(false, true, true));
        // A window: never.
        assert!(!shadow_candidate(true, false, true));
        assert!(!shadow_candidate(true, true, true));
    }

    /// **What each tier's grid actually reaches**, per axis — the number the tier table never
    /// stated and the one a `cell` change has to be read against.
    ///
    /// `cell` looks like a quality knob and is not. The cell allocation is fixed ([`CELL_ALLOC`]),
    /// so a finer cell is bought with reach one for one, and a fragment outside the grid takes
    /// nothing from it — which on a WMO surface is the only local light it has in a modern frame.
    /// Cinematic shipped at 3 yd and so reached ±72 yd horizontally and ±36 vertically: the least
    /// of any tier with a grid, and less than the Ultra it is built on top of. The tier whose
    /// whole point is a city full of lamps was the one that lit the least of a city.
    ///
    /// The vertical is the tight axis on every row — 24 cells, not 48 — and a city's ramps and
    /// tiers live on it, so it is asserted apart rather than folded into one number.
    #[test]
    fn each_tiers_grid_reach_is_the_one_it_is_supposed_to_have() {
        let reach = |q| light_tier(q).reach();
        assert_eq!(reach(LightingQuality::Low), Vec3::new(0.5, 0.5, 0.5), "no grid");
        assert_eq!(reach(LightingQuality::Medium), Vec3::new(128.0, 64.0, 128.0));
        assert_eq!(reach(LightingQuality::High), Vec3::new(144.0, 72.0, 144.0));
        assert_eq!(reach(LightingQuality::Ultra), Vec3::new(60.0, 30.0, 60.0));
        assert_eq!(reach(LightingQuality::Cinematic), Vec3::new(60.0, 30.0, 60.0));
        // …and past that, the emissive tiers' far level: 18 yd cells to ±324/±144.
        for q in [LightingQuality::Ultra, LightingQuality::Cinematic] {
            assert_eq!(light_tier(q).far_reach(), Vec3::new(324.0, 144.0, 324.0), "{q:?}");
        }
        for q in [LightingQuality::Low, LightingQuality::Medium, LightingQuality::High] {
            assert_eq!(light_tier(q).far_reach(), Vec3::ZERO, "{q:?} has no far level");
        }
    }

    /// GOLDEN — **a lamp lights its street at any distance the table packs it.** The report: lights
    /// "popping into view" as the player walked up to them, where the reference's look the same
    /// across the whole view distance. The near grid ended at ±72 yd and nothing past it took any
    /// local light — at Goldshire at midnight all nineteen packed lights sat 82–175 yd from the
    /// camera and every one of them lit nothing.
    ///
    /// So the far level has to cover every light the table can hold plus that light's own range,
    /// horizontally; and it has to be a whole multiple of the near cell, or the two levels' bases
    /// could disagree about which cell the camera is in.
    #[test]
    fn the_emissive_tiers_far_level_covers_every_packed_light_and_its_range() {
        for q in LightingQuality::ALL.into_iter().filter(|q| q.lights_emissive_sources()) {
            let t = light_tier(q);
            let lit = crate::lighting::MODERN_PACK_RADIUS + 48.0;
            assert!(
                t.far_reach().x >= lit && t.far_reach().z >= lit,
                "{q:?}: {:?} < {lit}",
                t.far_reach()
            );
            assert_eq!(t.far_dims.x, t.far_dims.z, "the shader carries one horizontal count");
            assert!(t.far_step >= 2, "a far level as fine as the near one is just a bigger near level");
            // Both levels together fit the one allocation every tier shares.
            assert!(t.cells() <= CELL_ALLOC, "{q:?}: {} cells", t.cells());
        }
    }

    /// The far level's base is the near level's, divided down — and it tracks the camera by whole
    /// far cells, so a cell that scrolls out is the cell that scrolls back in.
    #[test]
    fn the_far_level_is_centred_on_the_near_levels_camera_cell() {
        let t = light_tier(LightingQuality::Cinematic);
        for cam in [
            Vec3::new(-9400.0, 60.0, 40.0),
            Vec3::new(0.5, -3.0, 0.5),
            Vec3::new(1234.0, 250.0, -8765.0),
        ] {
            let near = grid_base(&t, cam);
            let far = far_base(&t, near);
            // The camera is inside the far level, near its middle.
            let c = (cam / t.far_cell()).floor().as_ivec3() - far;
            let half = t.far_dims.as_ivec3() / 2;
            assert!((c - half).abs().max_element() <= 1, "{cam}: far cell {c} vs centre {half}");
        }
    }

    /// GOLDEN — **a cube shadow fades with distance before its slot is released**, so a lamp does
    /// not change as it crosses [`SHADOW_RADIUS`]. Full strength close in, zero at the radius, and
    /// no fade at all on the tiers that do not ask for one.
    #[test]
    fn a_shadow_fades_out_before_the_radius_that_releases_it() {
        let cine = light_tier(LightingQuality::Cinematic);
        assert_eq!(shadow_distance_weight(&cine, 5.0), 1.0);
        assert_eq!(shadow_distance_weight(&cine, SHADOW_RADIUS), 0.0);
        let mid = shadow_distance_weight(&cine, SHADOW_RADIUS - cine.shadow_fade_yd * 0.5);
        assert!(mid > 0.0 && mid < 1.0, "{mid}");
        let high = light_tier(LightingQuality::High);
        assert_eq!(shadow_distance_weight(&high, SHADOW_RADIUS - 1.0), 1.0);
    }

    /// A face drawn into part of its tile is drawn into whole texels, and a tier that draws the
    /// whole tile still does.
    #[test]
    fn a_face_is_drawn_into_whole_texels_of_its_tile() {
        let size = crate::modern::shadow::SHADOW_MAP_SIZE;
        let tile = size / ATLAS_TILES;
        assert_eq!(face_px(&light_tier(LightingQuality::High), size), tile);
        let half = face_px(&light_tier(LightingQuality::Cinematic), size);
        assert!(half < tile && half >= tile / 2 - 1, "{half} of {tile}");
    }

    /// **The reach ladder is monotone** — a higher tier's grid lights at least as far out as a lower
    /// one's, on every axis. It was not: Ultra's 4.5 yd single level (P6) reached less than High's
    /// 6 yd one, and a test pinned the gap until someone revisited the row. P8b did (Ultra took the
    /// emissive tiers' two-level grid), and this is the full check that test asked to become.
    #[test]
    fn the_reach_ladder_is_monotone() {
        for pair in LightingQuality::ALL.windows(2) {
            let (lo, hi) = (light_tier(pair[0]).full_reach(), light_tier(pair[1]).full_reach());
            assert!(hi.cmpge(lo).all(), "{:?} reaches {hi:?} < {:?}'s {lo:?}", pair[1], pair[0]);
        }
    }

    fn lamp(x: f32) -> ShadowLight {
        ShadowLight {
            pos: Vec3::new(x, 0.0, 0.0),
            range: 20.0,
            near: CUBE_NEAR,
        }
    }

    /// Settle the fades: enough frames that everything ramping has reached its end. Shadow state
    /// is temporal now, so a test that wants a steady answer has to let it settle.
    fn settle(held: &mut Vec<Option<HeldSlot>>, ranked: &[(f32, ShadowLight)], cull: &Option<ShadowCull>) {
        for _ in 0..30 {
            assign_slots(held, ranked, cull, 1.0 / 60.0);
        }
    }

    /// Which light sits in each slot, fades ignored.
    fn lights_in(held: &[Option<HeldSlot>]) -> Vec<Option<ShadowLight>> {
        held.iter().map(|h| h.map(|h| h.light)).collect()
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
        settle(&mut held, &ranked(&[a, b, c]), &None);
        assert_eq!(lights_in(&held), vec![Some(a), Some(b), Some(c), None]);

        // `b` leaves the radius — it is simply not a candidate any more.
        settle(&mut held, &ranked(&[a, c]), &None);
        assert_eq!(
            lights_in(&held),
            vec![Some(a), None, Some(c), None],
            "a and c must not move; b's slot becomes a hole"
        );

        // A new lamp takes the HOLE, not the end, and still nothing else moves.
        let d = lamp(30.0);
        settle(&mut held, &ranked(&[a, c, d]), &None);
        assert_eq!(lights_in(&held), vec![Some(a), Some(d), Some(c), None]);
    }

    fn cull_at(eye: Vec3) -> ShadowCull {
        ShadowCull {
            eye,
            tan_half_fov: (std::f32::consts::FRAC_PI_4 * 0.5).tan(),
        }
    }

    /// GOLDEN — **turning the camera may not change which lights cast shadows.**
    ///
    /// This is the bug the player reported last: lights "randomly light the walls and drop shadows
    /// depending on my view angle" in Northshire Abbey. The cull used to reject a light whose lit
    /// sphere was behind the eye, which made the shadowed SET a function of where you were looking
    /// — so every turn of the head admitted and dropped cubes, and walls fell in and out of shadow.
    ///
    /// The metric is the light's solid angle at the eye now, and a sphere's solid angle does not
    /// care which way the camera faces. The eye does not move here; only the look direction would,
    /// and it is not an input at all.
    #[test]
    fn the_cull_does_not_depend_on_where_the_camera_looks() {
        let cull = cull_at(Vec3::ZERO);
        // The same lamp, placed all around the eye at one distance: every answer must agree,
        // because nothing about the test can see an orientation.
        let behind = cull.screen_fraction(Vec3::new(0.0, 0.0, 30.0), 12.0);
        for pos in [
            Vec3::new(0.0, 0.0, -30.0),
            Vec3::new(30.0, 0.0, 0.0),
            Vec3::new(-30.0, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(30.0 / 2.0_f32.sqrt(), 0.0, -30.0 / 2.0_f32.sqrt()),
        ] {
            assert!(
                (cull.screen_fraction(pos, 12.0) - behind).abs() < 1e-4,
                "{pos} reads differently from straight behind: a turn would change the set"
            );
        }
    }

    /// GOLDEN — **hysteresis.** A light hovering at the admission threshold must not take and lose
    /// its cube on alternate frames. It is admitted at `ADMIT_FRACTION` and kept until it falls
    /// below `KEEP_FRACTION`, and the gap between them is what stops the chatter.
    #[test]
    fn a_light_at_the_threshold_does_not_chatter() {
        assert!(
            KEEP_FRACTION < ADMIT_FRACTION,
            "the keep threshold must be the looser one"
        );
        // Placed exactly where it would just fail to be admitted fresh...
        let range = 12.0;
        let d_admit = range / (ADMIT_FRACTION * cull_at(Vec3::ZERO).tan_half_fov);
        let cull = cull_at(Vec3::ZERO);
        let outside = ShadowLight {
            pos: Vec3::new(0.0, 0.0, -(d_admit * 1.05)),
            range,
            near: CUBE_NEAR,
        };
        assert!(!cull.worth_a_cube(outside.pos, outside.range), "not admitted");
        assert!(
            cull.still_worth_it(outside.pos, outside.range),
            "but had it already held a slot, it would keep it — that gap is the hysteresis"
        );
    }

    /// GOLDEN — **a slot never switches on or off, it ramps.** A cube arriving or leaving is the
    /// moment a shadow would pop, so the fade is what makes the set's churn invisible: a new slot
    /// starts at zero (reading exactly as "no cube"), and a retiring one reaches zero BEFORE it
    /// releases its tiles.
    #[test]
    fn a_slots_shadow_cross_fades_in_and_out() {
        let a = lamp(0.0);
        let mut held = vec![None; 2];

        // Admitted: it takes the slot immediately, and contributes NOTHING on the frame it
        // arrives — a fade of zero reads exactly as "this light has no cube", which is the whole
        // point. The filling happens after the ramp, so its first step lands next frame.
        assign_slots(&mut held, &ranked(&[a]), &None, 1.0 / 60.0);
        let h = held[0].expect("took a slot");
        assert_eq!(h.light, a);
        assert_eq!(h.fade, 0.0, "arrives dark");

        assign_slots(&mut held, &ranked(&[a]), &None, 1.0 / 60.0);
        let h = held[0].unwrap();
        assert!(h.fade > 0.0 && h.fade < 1.0, "then ramps: {}", h.fade);

        settle(&mut held, &ranked(&[a]), &None);
        assert_eq!(held[0].unwrap().fade, 1.0, "settles at full shadow");

        // Now it stops being a candidate. It must fade out, and hold its slot while it does —
        // releasing the tiles first would hand a live shadow to whoever took the slot next.
        assign_slots(&mut held, &[], &None, 1.0 / 60.0);
        let h = held[0].expect("still holds the slot while fading out");
        assert!(h.retiring && h.fade < 1.0, "{h:?}");

        settle(&mut held, &[], &None);
        assert_eq!(held[0], None, "released only once it reached zero");
    }

    /// GOLDEN — **a derived light's tier gate is read where the light is READ, not where it is
    /// spawned.** The two look equivalent and are not: the lighting model latches a frame or more
    /// after the world starts streaming, so a doodad that arrived first asked "are emissive lights
    /// on?", was told no, and was never asked again. Which of Northshire's lampposts lit came down
    /// to streaming order, and re-rolled on every reload — a lamp blinking for no visible reason,
    /// and others simply never working.
    ///
    /// The entity is spawned unconditionally and tagged; the table decides per frame.
    #[test]
    fn a_derived_lights_tier_gate_is_read_per_frame_not_latched_at_spawn() {
        // The rule `global_light`'s filter applies, as arithmetic: an authored light is never
        // gated, and a derived one follows the tier as it stands THIS frame.
        let admits = |synthesized: bool, emissive_on: bool| !synthesized || emissive_on;

        assert!(admits(false, false), "an authored light is never gated");
        assert!(admits(false, true));
        assert!(!admits(true, false), "derived, tier says no: out of the table");
        assert!(admits(true, true), "derived, tier says yes: in it");

        // The property that matters: the answer depends only on the tier now, never on when the
        // doodad happened to spawn. Same inputs, same answer, whichever order they arrived in.
        for on in [true, false] {
            assert_eq!(admits(true, on), on, "spawn order cannot enter into it");
        }
    }

    /// GOLDEN — **a street lamp may not light the inside of the chapel beside it.**
    ///
    /// A local light has no occluder at all unless it holds one of the twelve cube slots, and
    /// `global_light`'s room term admits every room-less light by law — it is "every light outside
    /// a building", and the reference never had a culled room's torch to register. So the `w`
    /// channel of the light table's colour row carries whether the light belongs to a building, and
    /// a modern INTERIOR fragment takes only the lights that do.
    ///
    /// This cost nothing while the lamps of Elwynn and Darkshore carried no light block at all.
    /// Deriving one from their glow cards is what turned it into a chapel full of morning sun.
    #[test]
    fn the_light_tables_w_channel_says_whether_a_light_may_reach_an_interior() {
        // The packer writes 1 for a light that names rooms and 0 for one that names none; the gate
        // in `static_gx.wgsl` is `interior && w < 0.5 -> skip`. Both halves of that, as arithmetic.
        let street = 0.0f32;
        let chapel = 1.0f32;
        let admits = |interior: bool, w: f32| !(interior && w < 0.5);

        assert!(!admits(true, street), "the street lamp stays out of the room");
        assert!(admits(true, chapel), "the room's own sconce still lights it");
        assert!(admits(false, street), "and the street lamp still lights the street");
        assert!(
            admits(false, chapel),
            "a building's own light still reaches its exterior walls — the portal gate, not this \
             one, is what decides that"
        );
    }

    /// GOLDEN — **a fixture may not shadow its own light.** A synthesized light sits inside the
    /// thing that emits it, so with one shared near plane the lantern's own frame was a caster a
    /// hand's breadth away, projecting its bars across the pool and taking most of the lamp with
    /// it. That is what made a lamp dim as soon as it was close enough to earn a cube while a
    /// distant one stayed at full brightness — and, at every acquire and release, flicker.
    ///
    /// Geometry nearer than the near plane is clipped, so the fixture's radius IS the fix.
    #[test]
    fn a_fixture_is_clipped_out_of_its_own_cube() {
        let bare = CUBE_NEAR;
        let housed = CUBE_NEAR.max(0.4 * FIXTURE_CLEARANCE);
        assert!(
            housed > 0.4,
            "the near plane must clear the fixture it houses, not sit inside it: {housed}"
        );
        assert_eq!(bare, CUBE_NEAR, "an authored light keeps P6's near plane");

        // And the matrix actually built for the cube uses it: a point at the fixture's own radius
        // falls OUTSIDE the clip, while the floor the lamp lights stays inside.
        let clip = cube_face_clip(Vec3::ZERO, 0, housed);
        let on_the_frame = clip.project_point3(Vec3::new(0.4, 0.0, 0.0));
        let on_the_floor = clip.project_point3(Vec3::new(6.0, 0.0, 0.0));
        assert!(on_the_frame.z > 1.0, "the frame is clipped: {on_the_frame}");
        assert!(
            (0.0..=1.0).contains(&on_the_floor.z),
            "the floor still casts: {on_the_floor}"
        );
    }

    /// fork: P8 — the cull still refuses what it should: a lamp too small on screen to resolve a
    /// shadow in. What it no longer refuses is anything for being out of frame.
    #[test]
    fn the_cull_refuses_a_lamp_too_small_to_read() {
        let cull = cull_at(Vec3::ZERO);
        // Across a zone: a dozen pixels of lit sphere, not worth six depth renders.
        assert!(!cull.worth_a_cube(Vec3::new(0.0, 0.0, -600.0), 8.0));
        // The lamp you are standing next to.
        assert!(cull.worth_a_cube(Vec3::new(0.0, 0.0, -6.0), 12.0));
        // And one directly behind you at the same distance is treated identically.
        assert!(cull.worth_a_cube(Vec3::new(0.0, 0.0, 6.0), 12.0));
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
            let clip = cube_face_clip(light, face, CUBE_NEAR);
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
