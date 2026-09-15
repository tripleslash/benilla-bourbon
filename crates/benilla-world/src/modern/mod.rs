//! **Modern lighting** — the fork's parallel lighting model (`FORK.md`).
//!
//! Everything under this module is fork-owned and runs *beside* upstream's classic lanes, never in
//! place of them: `lightingModel = 0` (classic, the default) leaves every frame byte-identical to a
//! build without this module. The hook points it needs inside upstream-owned files are listed in
//! `HOOKS.md` next to this file, so an upstream merge can be checked against them.
//!
//! P0 scope: the settings, their per-world latch, and the shared WGSL libraries
//! (`benilla_modern::{color, frame}`). Nothing here draws yet.

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};

use crate::view::WorldCamera;

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

/// Registers the settings, the latch, and the shared WGSL libraries.
pub struct ModernLightingPlugin;

impl Plugin for ModernLightingPlugin {
    fn build(&self, app: &mut App) {
        crate::shaders::modern_libraries(app);
        app.init_resource::<LightingSettings>()
            .init_resource::<ActiveLighting>()
            .add_plugins(ExtractResourcePlugin::<ActiveLighting>::default())
            .add_systems(PreUpdate, latch_on_world_entry);
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
