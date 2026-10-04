//! Liquid: the animated surfaces (`surface`), where the liquid is and whether a subject or the
//! camera eye is in it (`query`, indexed by `spatial`), and the underwater drift cloud (`drift`).
//! The MCLQ parse and flat mesh are `benilla_formats::liquid`; the shading is `liquid.wgsl`.
//!
//! The reference has three liquid renderers, each with its own arm in the shader: ADT MCLQ, WMO
//! MLIQ water (`0x6b62e0` category 0, exterior `0x6b6630` or interior `0x6b6420` by the group's
//! `MOGP.flags & 0x48`: one texture stage, no depth ramp, alpha from a per-vertex authored byte)
//! and magma/slime.
//!
//! The ADT combiner is the asset `Shaders\Pixel\ocean0_s.bls` in `patch.MPQ`:
//! `rgb = primary·colorTex.rgb + detailTex.rgb + (secondary + 0.25)·detailTex.a`,
//! `alpha = colorTex.a`. `primary` is the lit vertex colour `clamp(ambient + N·L·sun)`;
//! `detailTex` is the animated `lake_a`/`ocean_h` frame, near-black RGB and a ripple alpha.
//! `colorTex` is a 64-row swatch, RGB between the zone's raw `Light.dbc` water rows (IntBand 16/17
//! river/lake, 14/15 ocean) and alpha between its `LightParams` shallow and deep alphas, filled by
//! `0x68a830` as `row(i) = c0 + floor(i·(c1−c0)/64)`, so it never reaches the deep endpoint. The
//! ocean's last row alone is darkened by an HSV `V ×= 0.9` and forced opaque; about 80% of ocean
//! vertices sample it. The dirty flags `[0xc8117c]`/`[0xc81b70]` clear every world frame
//! (`0x680b90`/`0x680b97`, refill `0x58acd0`), so colour and opacity track the zone and the clock.
//!
//! One per-vertex depth `V` indexes colour and alpha alike: `min(byte/42, 1)` on river/lake
//! (`0xc81768`, read at `0x68d818` by `0x68d790`), `min(byte/255, 1)` on ocean (`0xc7fcd8`, read
//! at `0x68d718` by `0x68d690`), both tables built by `0x68c4c0`. `0xc7fbc0`'s `1.6·(i/63)^8` is no
//! water alpha: it fills the sky glare texture `[0xc7fbb8]` (`0x68c250` via `0x68c4a0`), whose one
//! liquid binding, `0x685257`, sits in a branch that never runs (`[0xc800ec]` is only ever 0).

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use benilla_assets::materials::{LiquidCataMaterial, LiquidMaterial};
use benilla_assets::AssetSet;

mod art;
mod drift;
mod query;
#[cfg(test)]
mod real_data;
mod reflect;
mod ripple;
mod screen;
mod spatial;
mod surface;

// The submodules are private: this list is everything the rest of the client may name.
pub use query::{
    camera_claim, describe_at, liquid_at, player_claim, surfaces_at, unit_claim, water_surface_at,
    EyeLiquid, FoamPatch, LiquidClaim, LiquidHit, LiquidSource, RoomPlacements, SubmergedEye,
    Underwater, WaterChunkInfo, WmoPool,
};
pub use reflect::UNMIRRORED_RENDER_LAYER;
// fork: modern lighting (FORK.md, P4) — the fog composite runs on the mirror and orders before the scene copy.
pub(crate) use reflect::{LiquidReflectionCamera, ReflectionPlane};
pub(crate) use screen::{liquid_screen_pass, LiquidScreenTargets, LiquidScreenView};
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
/// fork: `1` is taken since by the SSR-only cata tier ([`WATER_DETAIL_SSR`]) and `3` by the
/// experimental modern combine ([`WATER_DETAIL_EXPERIMENTAL`]).
///
/// fork: Modern (Experimental) is the default. `$WOW_WATER_DETAIL` overrides for a session.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub struct WaterDetail(pub u8);

/// The 1.12.1 renderer.
pub const WATER_DETAIL_LEGACY: u8 = 0;
/// fork: the Cataclysm renderer WITHOUT its planar mirror — the screen-space trace alone over the
/// horizon colour (what perm 4 runs). Skips the second, mirrored scene render (and every modern
/// lighting pass that render drags along) for about half the water's cost. It sits in the
/// reference's medium-tier slot, which benilla never implemented.
pub const WATER_DETAIL_SSR: u8 = 1;
/// The Cataclysm renderer — `ProcWaterAbove` perm 4 / `ProcWaterBelow` perm 0.
pub const WATER_DETAIL_CATA: u8 = 2;
/// fork: modern lighting — the Cataclysm renderer, mirror and all, whose above-water combine is
/// the fork's stylised clear water (`liquid_cata.wgsl`'s `modern_water`, tuned against WoW Forever's
/// rivers) instead of the reference's: the bed seen through the water and filtered toward its hue,
/// Beer–Lambert toward the zone's deep colour, a calm surface whose refraction scales with the
/// water crossed, and a soft, capped reflection. "Modern (Experimental)" on the options page, under
/// either lighting model; a modern frame adds its wind sea, cascades and glitter. It takes the slot
/// after the reference's highest tier rather than one of its numbers, so `2` keeps meaning the
/// reference's water.
pub const WATER_DETAIL_EXPERIMENTAL: u8 = 3;

impl WaterDetail {
    /// Is this the Cataclysm renderer? Written as "not legacy" rather than "== 2" so that a medium
    /// tier landing at `1` joins the cata lane by default instead of silently falling back.
    pub fn is_cata(self) -> bool {
        self.0 != WATER_DETAIL_LEGACY
    }

    /// Clamp an arbitrary cvar value onto a tier this build actually has. Anything that is not a
    /// known tier reads as cata, which is the default and the better-looking answer; a player who
    /// hand-edits `waterDetail 7` into their config gets water, not a black lake — and not the
    /// experimental tier, which only a `3` asks for.
    pub fn from_cvar(v: f32) -> Self {
        if v <= 0.0 {
            WaterDetail(WATER_DETAIL_LEGACY)
        } else if v < 1.5 {
            WaterDetail(WATER_DETAIL_SSR)
        } else if (2.5..3.5).contains(&v) {
            WaterDetail(WATER_DETAIL_EXPERIMENTAL)
        } else {
            WaterDetail(WATER_DETAIL_CATA)
        }
    }

    /// Does the water run the Modern (Experimental) combine ([`WATER_DETAIL_EXPERIMENTAL`])?
    pub fn is_experimental(self) -> bool {
        self.0 == WATER_DETAIL_EXPERIMENTAL
    }

    /// Does the cata water get its planar mirror render? Only the full tiers do (cata and the
    /// experimental tier above it); the SSR tier reflects through the screen-space trace alone.
    pub fn has_planar_reflection(self) -> bool {
        self.0 >= WATER_DETAIL_CATA
    }
}

impl Default for WaterDetail {
    fn default() -> Self {
        match crate::launch_env::var("WOW_WATER_DETAIL").and_then(|v| v.parse::<f32>().ok()) {
            Some(v) => Self::from_cvar(v),
            None => WaterDetail(WATER_DETAIL_EXPERIMENTAL),
        }
    }
}

/// `WOW_FORCE_SUB=<frames>`: hold the camera-eye verdict submerged for the first `<frames>` frames,
/// then release it, so the wet-to-dry crossing lands on a known frame with no water or server.
fn forced_submersion_frames() -> u32 {
    std::env::var("WOW_FORCE_SUB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The [`forced_submersion_frames`] override, run after the real probe so it overwrites the
/// verdict rather than racing it; logs the release frame.
fn force_submersion(mut underwater: ResMut<Underwater>, mut frame: Local<u32>) {
    let hold = forced_submersion_frames();
    *frame += 1;
    if *frame <= hold {
        underwater.0 = benilla_formats::Submersion::Water;
    } else if *frame == hold + 1 {
        info!(
            "WOW_FORCE_SUB: frame {} — eye verdict released to Dry",
            *frame
        );
    }
}

/// The set that writes [`Underwater`]; every consumer of the verdict runs after it, since a
/// frame-old verdict shows one mixed frame (murk with the sky still up) on every surface crossing.
#[derive(bevy::ecs::schedule::SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmersionVerdict;

/// The liquid subsystem: the shared materials at startup, the submersion verdict and the drift
/// cloud each frame. The terrain streamer spawns the surfaces with their tile ([`spawn_liquids`]).
pub(crate) struct LiquidPlugin;

impl Plugin for LiquidPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<LiquidMaterial>::default())
            .add_plugins(MaterialPlugin::<LiquidCataMaterial>::default())
            .init_resource::<WaterDetail>()
            .init_resource::<Underwater>()
            .init_resource::<SubmergedEye>()
            .init_resource::<WaterIndex>()
            // PreUpdate: a surface streamed by last frame's commands is indexed before anyone asks.
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
                    surface::sync_water_tier,
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
                    // After the portal pass publishes this frame's `CameraInteriorClaim`: a
                    // frame-old room flashes the wrong atmosphere on every doorway crossing.
                    query::detect_submersion
                        .after(crate::wmo_portal::WmoPvsSet)
                        .in_set(SubmersionVerdict),
                ),
            )
            // The `WOW_NO_LIQUID` override: after both per-frame `Visibility` owners of a surface
            // (the exterior cull, the model-visibility authority) and before Bevy reads it.
            .add_systems(
                PostUpdate,
                surface::hide_liquid_surfaces
                    .after(crate::exterior_cull::ExteriorCullSet)
                    .before(bevy::camera::visibility::VisibilitySystems::VisibilityPropagate)
                    .run_if(|| std::env::var_os("WOW_NO_LIQUID").is_some()),
            );
        // Added only when `WOW_FORCE_SUB` names a hold: an ordinary run carries no system for it.
        if forced_submersion_frames() > 0 {
            app.add_systems(
                Update,
                force_submersion
                    .after(query::detect_submersion)
                    .in_set(SubmersionVerdict),
            );
        }
        // fork: modern lighting — the water look's live override (see [`surface::sync_water_look`])
        // likewise exists only in a session launched with it: a production run carries neither the
        // system nor its per-frame read of the knob.
        if crate::launch_env::var("WOW_WATER_LOOK").is_some() {
            app.add_systems(Update, surface::sync_water_look);
        }
        drift::register(app);
        // The three pictures of the frame the cata renderer reads, and the graph node that fills
        // them. Registered unconditionally: the node stands itself down when the targets are
        // absent, and the materials that read them exist either way. Multisampling is NOT a
        // stand-down — the node compiles a per-sample-count variant and resolves in the copy it was
        // already making.
        screen::register(app);
        // The dynamic ripple displacement map: the wave a wading body pushes into the water.
        ripple::register(app);
        // The planar mirror `ProcWaterAbove` perm 0 reads on `samp1` — the one input of the cata
        // water that is a scene render rather than a blit, and therefore a camera rather than a
        // node. Registered unconditionally: its camera stands itself down off `waterDetail` and
        // the submersion verdict, and the material binds the target either way.
        reflect::register(app);
    }
}
