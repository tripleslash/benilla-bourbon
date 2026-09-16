//! **Modern lighting** — the fork's parallel lighting model (`FORK.md`).
//!
//! Everything under this module is fork-owned and runs *beside* upstream's classic lanes, never in
//! place of them: `lightingModel = 0` (classic, the default) leaves every frame byte-identical to a
//! build without this module. The hook points it needs inside upstream-owned files are listed in
//! `HOOKS.md` next to this file, so an upstream merge can be checked against them.
//!
//! P0 scope: the settings, their per-world latch, and the shared WGSL libraries
//! (`benilla_modern::{color, frame}`). Nothing here draws yet.

use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};

use crate::view::WorldCamera;

pub mod gi;
pub mod lights;
pub mod look;
pub mod shadow;
mod sky;
pub mod sun_arc;
mod tick_heal;
pub mod volumetrics;

/// P1b's fixed exposure (the FFXGlow combine's `wave.w`); auto-exposure lands in P7.
pub const MODERN_EXPOSURE: f32 = 1.0;

/// Which lighting model the world renders with. CVar `lightingModel`: `0` classic, `1` modern.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightingModel {
    /// Upstream's 1.12.1 reproduction: gamma lane, baked shadows, linear fog.
    #[default]
    Classic,
    /// Linear HDR, all-caster shadows, probe GI, froxel volumetrics.
    Modern,
}

impl LightingModel {
    /// The CVar's numeric value space. Anything at or above one half reads as modern.
    pub fn from_cvar(v: f32) -> Self {
        if v >= 0.5 {
            Self::Modern
        } else {
            Self::Classic
        }
    }

    pub fn as_cvar(self) -> u8 {
        match self {
            Self::Classic => 0,
            Self::Modern => 1,
        }
    }

    /// `$WOW_LIGHTING` accepts the names as well as the numbers.
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "0" | "classic" => Some(Self::Classic),
            "1" | "modern" => Some(Self::Modern),
            _ => None,
        }
    }
}

/// The modern model's quality tier (the plan's tier table). CVar `lightingQuality`: `0`–`3`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LightingQuality {
    Low = 0,
    Medium = 1,
    High = 2,
    Ultra = 3,
}

impl LightingQuality {
    /// Medium in the browser, High natively — the plan's budget split.
    pub const PLATFORM_DEFAULT: Self = if cfg!(target_arch = "wasm32") {
        Self::Medium
    } else {
        Self::High
    };

    /// Rounded and clamped into the tier range; a non-number reads as the platform default.
    pub fn from_cvar(v: f32) -> Self {
        if !v.is_finite() {
            return Self::PLATFORM_DEFAULT;
        }
        match v.round().clamp(0.0, 3.0) as u8 {
            0 => Self::Low,
            1 => Self::Medium,
            2 => Self::High,
            _ => Self::Ultra,
        }
    }

    pub fn as_cvar(self) -> u8 {
        self as u8
    }
}

/// What the player asked for. Written by `benilla-app`'s CVar host; takes effect at the next
/// world entry (see [`ActiveLighting`]).
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct LightingSettings {
    pub model: LightingModel,
    pub quality: LightingQuality,
}

impl Default for LightingSettings {
    /// `$WOW_LIGHTING` / `$WOW_LIGHTING_QUALITY` override the shipped defaults, session-only — the
    /// A/B levers, and the only way to drive this from `benilla-worldview`, which has no CVar host.
    fn default() -> Self {
        let model = crate::launch_env::var("WOW_LIGHTING")
            .and_then(|v| LightingModel::parse(&v))
            .unwrap_or_default();
        let quality = crate::launch_env::var("WOW_LIGHTING_QUALITY")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .map_or(LightingQuality::PLATFORM_DEFAULT, LightingQuality::from_cvar);
        Self { model, quality }
    }
}

/// What this world session actually renders with: [`LightingSettings`] copied once, when a
/// [`WorldCamera`] appears. Latched rather than live because switching models swaps materials,
/// render-graph nodes and a few hundred pipelines — work that belongs behind the loading cover,
/// the same posture as `gxMultisample`.
#[derive(Resource, Clone, Copy, PartialEq, Debug, ExtractResource)]
pub struct ActiveLighting {
    pub model: LightingModel,
    pub quality: LightingQuality,
}

impl Default for ActiveLighting {
    fn default() -> Self {
        Self {
            model: LightingModel::Classic,
            quality: LightingQuality::PLATFORM_DEFAULT,
        }
    }
}

impl ActiveLighting {
    pub fn is_modern(self) -> bool {
        self.model == LightingModel::Modern
    }
}

/// The latched model as a process-wide bit, for the Rust-side hooks inside upstream-owned systems
/// (sky colours, celestial and cloud materials, the FFXGlow combine) whose signatures stay as
/// upstream wrote them. Written only by [`latch_on_world_entry`].
static MODERN_FRAME: AtomicBool = AtomicBool::new(false);

/// Is the world frame linear (a modern session)?
pub fn frame_is_modern() -> bool {
    MODERN_FRAME.load(Ordering::Relaxed)
}

/// Bisect switches (`$WOW_MODERN_SUNARC=0`, `$WOW_MODERN_CPUHOOKS=0`; browser `?modern_sunarc=0`,
/// `?modern_cpuhooks=0`): turn one modern main-world behaviour off without a rebuild. Read once.
fn switch_on(cell: &'static std::sync::OnceLock<bool>, key: &str) -> bool {
    *cell.get_or_init(|| crate::launch_env::var(key).as_deref() != Some("0"))
}

/// The sun arc's direction overrides (`sun_arc`).
pub fn sun_arc_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch_on(&CELL, "WOW_MODERN_SUNARC")
}

/// The shadow debug view (`$WOW_MODERN_SHADOWDEBUG`, browser `?modern_shadowdebug=`): `1` = the modern
/// receivers output raw sun visibility as grey, `2` = sky visibility, `3` = the probe field's
/// irradiance (P5), `4` = how much of it the fragment's probe cell answered, `5` = what the local
/// light grid put on the fragment, cube shadows included (P6). `0` (default) = off.
pub fn shadow_debug_mode() -> u8 {
    static CELL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::launch_env::var("WOW_MODERN_SHADOWDEBUG")
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(0)
    })
}

/// The CPU-side modern hooks: sky stop decode, the cloud texture's sRGB view, the disc `span.z` flag.
pub fn cpu_hooks_enabled() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch_on(&CELL, "WOW_MODERN_CPUHOOKS")
}

/// The exact sRGB transfer, CPU side — the twin of `benilla_modern::color::srgb_to_linear_channel`.
pub fn srgb_to_linear(c: f32) -> f32 {
    let x = c.max(0.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Hermite smoothstep, as WGSL defines it.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// An authored gamma colour as the current frame wants it: untouched in a classic frame, decoded
/// (rgb only; `w` rides through) in a modern one.
pub fn decode_if_modern(v: Vec4) -> Vec4 {
    if frame_is_modern() && cpu_hooks_enabled() {
        Vec4::new(srgb_to_linear(v.x), srgb_to_linear(v.y), srgb_to_linear(v.z), v.w)
    } else {
        v
    }
}

/// Registers the settings, the latch, and the shared WGSL libraries.
pub struct ModernLightingPlugin;

impl Plugin for ModernLightingPlugin {
    fn build(&self, app: &mut App) {
        crate::shaders::modern_libraries(app);
        app.init_resource::<LightingSettings>()
            .init_resource::<ActiveLighting>()
            .add_plugins(ExtractResourcePlugin::<ActiveLighting>::default())
            .add_systems(PreUpdate, (latch_on_world_entry, sync_model_pipelines).chain())
            .add_plugins((
                sun_arc::plugin,
                shadow::plugin,
                tick_heal::plugin,
                look::plugin,
                sky::plugin,
                volumetrics::plugin,
                gi::plugin,
                lights::plugin,
            ));
    }
}

fn latch_on_world_entry(
    settings: Res<LightingSettings>,
    mut active: ResMut<ActiveLighting>,
    entered: Query<(), Added<WorldCamera>>,
) {
    if entered.is_empty() {
        return;
    }
    let next = ActiveLighting {
        model: settings.model,
        quality: settings.quality,
    };
    if *active != next {
        info!("lighting latched: {:?} / {:?}", next.model, next.quality);
        *active = next;
    }
    MODERN_FRAME.store(active.is_modern(), Ordering::Relaxed);
}

/// Model pipelines compile the modern code only in a modern session
/// (`benilla_assets::materials::MODERN_MODEL_PIPELINES`). On a latch edge every model material is
/// touched, so it re-prepares under the new pipeline key; the runtime flag in the shader stays, so
/// a material not yet re-prepared still renders the right model.
fn sync_model_pipelines(
    active: Res<ActiveLighting>,
    mut materials: ResMut<Assets<benilla_assets::materials::WowModelMaterial>>,
    mut last: Local<Option<bool>>,
) {
    let modern = active.is_modern();
    if *last == Some(modern) {
        return;
    }
    let first = last.is_none();
    *last = Some(modern);
    benilla_assets::materials::MODERN_MODEL_PIPELINES.store(modern, Ordering::Relaxed);
    if first && !modern {
        return;
    }
    let ids: Vec<_> = materials.ids().collect();
    for id in ids {
        let _ = materials.get_mut(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_cvar_round_trips_and_names_parse() {
        for m in [LightingModel::Classic, LightingModel::Modern] {
            assert_eq!(LightingModel::from_cvar(m.as_cvar() as f32), m);
        }
        assert_eq!(LightingModel::parse(" Modern "), Some(LightingModel::Modern));
        assert_eq!(LightingModel::parse("classic"), Some(LightingModel::Classic));
        assert_eq!(LightingModel::parse("2x"), None);
    }

    #[test]
    fn srgb_decode_matches_the_curve() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-5);
        // Mod2x's neutral grey stays neutral after the shader's 2.295 source scale.
        assert!((2.0 * srgb_to_linear(0.5) * 2.295 - 0.982).abs() < 0.01);
    }

    #[test]
    fn quality_rounds_and_clamps() {
        assert_eq!(LightingQuality::from_cvar(-4.0), LightingQuality::Low);
        assert_eq!(LightingQuality::from_cvar(1.4), LightingQuality::Medium);
        assert_eq!(LightingQuality::from_cvar(2.6), LightingQuality::Ultra);
        assert_eq!(LightingQuality::from_cvar(99.0), LightingQuality::Ultra);
        assert_eq!(
            LightingQuality::from_cvar(f32::NAN),
            LightingQuality::PLATFORM_DEFAULT
        );
    }
}
