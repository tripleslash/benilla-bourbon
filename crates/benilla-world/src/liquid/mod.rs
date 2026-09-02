//! Liquid (water) rendering: the animated lake/river/ocean surfaces the reference draws over MCLQ
//! geometry. The parse + flat mesh live in `benilla_formats::liquid` (built into each `ChunkMesh.liquid`
//! by the terrain loader); this is the Bevy render glue — one shared [`LiquidMaterial`] per
//! [`LiquidKind`], a `texture_2d_array` of its animated frames, and a 24 fps frame-index cycler.
//!
//! **This is the ADT path, and we render WMO liquid through it too — which is wrong.** The reference
//! has three liquid renderers, not one: ADT MCLQ (below), WMO MLIQ water (`0x6b62e0` category 0,
//! split again into exterior `0x6b6630` / interior `0x6b6420` on the group's `MOGP.flags & 0x48` —
//! one texture stage, no depth ramp, alpha from a per-vertex authored byte), and magma/slime. The
//! `wmo_liquid_arms` census sizes them: 30 exterior water groups (Stormwind's canals) against 134
//! interior (every dungeon pool, Blackfathom's included). Neither is what this file implements.
//! Recorded in wow-re `terrain/scratch/water-shading-law.md`.
//!
//! Faithful ADT model — the combiner is an **asset**, extracted from `patch.MPQ` and read, not
//! inferred: `Shaders\Pixel\ocean0_s.bls` is `rgb = primary·colorTex.rgb + detailTex.rgb +
//! (secondary+0.25)·detailTex.a`, `alpha = colorTex.a`, with `0.25` its own scalar `PARAM`. The
//! formula this module has always carried is right verbatim; its **provenance was not** — the
//! citation here used to name an "apitrace WoW.17 program 159" and a `docs/knowledge/terrain.md`,
//! neither of which exists, and wow-re had the program attributed to a character draw. The body
//! colour is **`primary · waterTint`**, where:
//! - **`waterTint`** is a plain **2-endpoint linear lerp** of the zone's dedicated `Light.dbc` water
//!   rows, RAW (no ×0.711): IntBand rows 16/17 (river/lake) or 14/15 (ocean), shallow→deep, by the
//!   per-vertex depth `V` (river/lake `V = clamp(byte/42)`, VERIFIED `c81768`/`FUN_0068d790`; saturates
//!   ~5 yd so the channel middle reaches the deep/teal row). The swatch is rebuilt **per world frame**,
//!   not baked once at bootstrap: the dirty flags `[0xc8117c]`/`[0xc81b70]` clear at
//!   `0x680b90`/`0x680b97` and refill via `0x58acd0`, so colour and opacity track the zone and the
//!   clock. (The earlier "reflected sky × 0.711 via `FUN_0068c250`" model fingered the WRONG builder —
//!   a separate grey edge texture never bound on the water unit; and `byte/255` was the wrong LUT →
//!   river never went teal. The `FUN_0068a830` attribution this line used to carry was also wrong:
//!   wow-re reads `0x68a830` as a sky-band row fill.)
//! - **`primary`** is the lit vertex colour `clamp(ambient + N·L·sun)`.
//! - the animated `lake_a`/`ocean_h` frame is the **`detailTex`** (near-black RGB + ripple alpha): a
//!   faint flat lift + an achromatic shimmer on crests — NOT the body colour. Mipped + 16× aniso so the
//!   ripple averages out at distance (near-field samples mip 0, so near sparkle is the term itself).
//! - **opacity** = the SAME `V` indexes both colour and alpha (one swatch row → RGB + A): a ramp between
//!   the LightParams shallow/deep alphas — river 0.5→1.0, ocean 0.75→1.0. The river channel reaches
//!   α=1.0 (opaque, deep teal) by byte 42 ≈ 5 yd; the shore stays see-through (the pale edge band —
//!   the bottom shows). **OPEN:** we ramp it linearly (`127+2·row`), but wow-re reads the byte-verified
//!   ADT water alpha as the `0xc7fbc0` LUT's `1.6·(i/63)^8` — a curve that stays near-transparent far
//!   longer and then climbs hard. Unchanged so far: it is a look change to every ADT water surface in
//!   the game, so it belongs in one A/B with the WMO arms rather than landing on its own.
//!
//! River/lake `V = clamp(byte/42)` (steep `c81768` LUT, `FUN_0068d790`) — NOT `byte/255` (the `c7fcd8`
//! LUT, a different draw list the from-above river path doesn't use; it left the river middle stuck on
//! the shallow green row). Ocean uses a non-LUT UV path → placeholder `/255` pending its own RE+A/B.
//! (Earlier cuts: ripple-as-colour → black; `×8` → "deep too early"; FLAT colour → "completely gone";
//! sky × 0.711 → wrong builder; `byte/255` → wrong LUT, no teal centre. Faithful = rows 14–17 raw lerp
//! + the /42 V. 2026-05-31.)
//!
//! Two-sided, alpha-blended, depth-write off (Bevy's transparent pass = the verified MCLQ water render
//! state).
//!
//! The frame-flip is the client's first render animation — a deliberate **one-off** (a frame-index
//! uniform off Bevy real `Time`), NOT a general animation system. Two clocks: animation =
//! wall-clock; day/night = server game-time.

//! ## Layout
//!
//! Three concerns, so three files behind this face:
//!
//! * [`query`] — **where the liquid is, and whether you are in it.** The world-space grid every
//!   surface publishes ([`WaterChunkInfo`]), the position queries the swim/sound/foam/lighting
//!   systems ask it ([`liquid_at`], [`water_surface_at`]), and the per-frame camera submersion
//!   verdict ([`Underwater`]). Nothing here renders.
//! * [`spatial`] — **which surfaces are near an XY**, without walking them all: the grid-hash
//!   [`WaterIndex`] the per-draw consumers (the water-plane interleave's three lanes) pre-filter
//!   through. The once-a-frame askers keep the plain walk.
//! * [`surface`] — **the Bevy render glue.** The per-kind animated materials, the two spawn paths
//!   (MCLQ and WMO MLIQ), the flat mesh build, and the 24 fps frame cycler.
//! * [`drift`] — **the underwater drift cloud**: the 4000-mote field the reference draws while the
//!   camera eye is inside a liquid (decision 1814). It is here rather than under `weather` because
//!   the reference keeps it that way too — the pool is CWorld's, not the weather manager's, and
//!   the two share no state and no code — and because [`Underwater`] is its whole trigger.
//!
//! The one cross-feed runs query → lighting: `detect_submersion` publishes WHICH liquid the eye is
//! in, and `lighting::update_time_lighting` selects the whole submerged atmosphere from it.

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use benilla_assets::materials::{LiquidCataMaterial, LiquidMaterial};
use benilla_assets::AssetSet;

mod art;
mod drift;
mod query;
mod ripple;
mod screen;
#[cfg(test)]
mod real_data;
mod spatial;
mod surface; // the against-real-client-files tests — they span both halves

// The submodules are private, so this list IS the subsystem's face: everything the rest of the
// client may name. Most of this list is now reached through `crate::world_point::WorldPoint`
// rather than directly (decision 1164).
//
// `LiquidSurface` used to be deliberately absent — "spawned, never named from outside; add it here
// the day something needs it". Decision 1652 is that day: the exterior-window cull counts liquid
// apart from the rest of the scene, because a few dozen surfaces summed into tens of thousands of
// terrain cells is a leg that could reach nothing at all and never show it.
pub use query::{
    camera_claim, describe_at, liquid_at, player_claim, surfaces_at, unit_claim, water_surface_at,
    EyeLiquid, FoamPatch, LiquidClaim, LiquidHit, LiquidSource, RoomPlacements, SubmergedEye,
    Underwater, WaterChunkInfo, WmoPool,
};
pub(crate) use spatial::{maintain_water_index, WaterIndex};
pub(crate) use surface::{
    spawn_liquids, spawn_wmo_liquids, LiquidAssets, LiquidSoundSource, LiquidSurface,
};

/// **Which liquid renderer the client is running** — the `waterDetail` CVar, in the reference's own
/// value space.
///
/// `0` is the 1.12.1 path (`liquid.wgsl`, a port of `Water.bls` and the fixed-function combine
/// around it) and `2` is the Cataclysm path (`liquid_cata.wgsl`, `ProcWater` + `ProcWaterAbove`).
/// Those are the reference's numbers, not ours: `Liquid::CMaterialWater::InitResources` switches on
/// exactly this cvar, and its `1` is the *medium* tier (`MediumWater`), which benilla does not
/// implement. Leaving the gap rather than renumbering means a medium tier can land later without
/// re-teaching the options page, the config files players already have, or this type.
///
/// Cata is the default. `$WOW_WATER_DETAIL` overrides for a session.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub struct WaterDetail(pub u8);

/// The 1.12.1 renderer.
pub const WATER_DETAIL_LEGACY: u8 = 0;
/// The Cataclysm renderer — `ProcWaterAbove` perm 4 / `ProcWaterBelow` perm 0.
pub const WATER_DETAIL_CATA: u8 = 2;

impl WaterDetail {
    /// Is this the Cataclysm renderer? Written as "not legacy" rather than "== 2" so that a medium
    /// tier landing at `1` joins the cata lane by default instead of silently falling back.
    pub fn is_cata(self) -> bool {
        self.0 != WATER_DETAIL_LEGACY
    }

    /// Clamp an arbitrary cvar value onto a tier this build actually has. Anything that is not a
    /// known tier reads as cata, which is the default and the better-looking answer; a player who
    /// hand-edits `waterDetail 7` into their config gets water, not a black lake.
    pub fn from_cvar(v: f32) -> Self {
        if v <= 0.0 {
            WaterDetail(WATER_DETAIL_LEGACY)
        } else {
            WaterDetail(WATER_DETAIL_CATA)
        }
    }
}

impl Default for WaterDetail {
    fn default() -> Self {
        match std::env::var("WOW_WATER_DETAIL")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
        {
            Some(v) => Self::from_cvar(v),
            None => WaterDetail(WATER_DETAIL_CATA),
        }
    }
}

/// The frame slot where [`Underwater`] is written — the label every consumer of the submersion
/// verdict orders itself `.after(..)`. The submerged view is a whole-screen swap (atmosphere, clear
/// colour, and the sky-pass suppression all flip on it), so a consumer reading a frame-old verdict
/// shows one mixed frame on every surface crossing — the murk with the sun still up, or clear water
/// with the sky already gone. One label, so the "same frame" contract is written once instead of
/// re-derived per consumer.
#[derive(bevy::ecs::schedule::SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmersionVerdict;

/// The water subsystem: load the per-kind frame arrays + shared materials at startup, then cycle the
/// animation frame each update. Spawning the per-chunk surfaces happens in the terrain streamer (via
/// [`spawn_liquids`], water lives *with* its tile), reading [`LiquidAssets`].
pub(crate) struct LiquidPlugin;

impl Plugin for LiquidPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<LiquidMaterial>::default())
            .add_plugins(MaterialPlugin::<LiquidCataMaterial>::default())
            .init_resource::<WaterDetail>()
            .init_resource::<Underwater>()
            .init_resource::<SubmergedEye>()
            .init_resource::<WaterIndex>()
            // PreUpdate: surfaces stream in/out via Update-side commands, so the edge is visible
            // here the frame after — before any of that frame's consumers ask. A despawn's stale
            // entry in between self-filters at the consumer (`Query::get` misses).
            .add_systems(PreUpdate, maintain_water_index)
            .add_systems(Startup, surface::setup_liquid.after(AssetSet::Open))
            // The renderer swap and the above/below flip. Both are value-watched (see their
            // headers) so the steady state is one compare, and both run in `Update` because their
            // effect is a material the NEXT frame draws with — there is no ordering to get right
            // beyond being before the extract.
            .add_systems(
                Update,
                (
                    surface::sync_below_water.after(SubmersionVerdict),
                    screen::resize_targets,
                ),
            )
            // `PostUpdate`, so the tiles the streamer spawned this tick have already had their
            // commands applied and show up in its `Added` query — a surface is never drawn
            // material-less, and never dressed twice.
            .add_systems(PostUpdate, surface::apply_water_detail)
            .add_systems(
                Update,
                (
                    // AFTER the portal pass, which publishes the camera eye's room this frame
                    // (`CameraInteriorClaim`): the submersion verdict is the eye's room applied to
                    // the eye's position, and a frame-old room would flash the wrong atmosphere for
                    // one frame on every doorway crossing — which, for a whole-screen filter, is
                    // exactly the artefact that reads worst.
                    query::detect_submersion
                        .after(crate::wmo_portal::WmoPvsSet)
                        .in_set(SubmersionVerdict),
                ),
            )
            // The surface-render kill-switch (see [`hide_liquid_surfaces`]) — inert without the env
            // var, so it costs nothing when it isn't being used.
            //
            // `PostUpdate`, after the two systems that now own a liquid surface's `Visibility`
            // every frame — the exterior-window cull (ADT surfaces, decision 1652) and the
            // model-visibility authority (WMO pools, 0689/0784) — and before Bevy consumes the
            // result. An *override* that runs last, deliberately, rather than a second writer
            // trying to compose with them (decision 0025's law, and 0784's reasoning for why
            // ordering is the wrong tool for two real terms but the right one for a kill-switch:
            // there is nothing to AND here, the switch simply wins).
            .add_systems(
                PostUpdate,
                surface::hide_liquid_surfaces
                    .after(crate::exterior_cull::ExteriorCullSet)
                    .before(bevy::camera::visibility::VisibilitySystems::VisibilityPropagate)
                    .run_if(|| std::env::var_os("WOW_NO_LIQUID").is_some()),
            );
        // The underwater drift cloud — the one thing in this subsystem that RENDERS because the
        // eye is submerged, rather than answering where the liquid is (see the layout note above).
        drift::register(app);
        // The three pictures of the frame the cata renderer reads, and the graph node that fills
        // them. Registered unconditionally: the node stands itself down when the targets are
        // absent, and the materials that read them exist either way. Multisampling is NOT a
        // stand-down — the node compiles a per-sample-count variant and resolves in the copy it was
        // already making.
        screen::register(app);
        // The dynamic ripple displacement map: the wave a wading body pushes into the water.
        ripple::register(app);
    }

    /// The liquid screen pass's edge to `Ssilvb`, which can only be declared once every plugin has
    /// built — see [`screen::register_late`].
    fn finish(&self, app: &mut App) {
        screen::register_late(app);
    }
}
