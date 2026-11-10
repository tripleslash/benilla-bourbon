//! fork: dlss (FORK.md) — NVIDIA DLSS 4.5 in FSR's upscaler slot. Not a 1.12 feature;
//! benilla's own, behind the `upscaler` and `frameGeneration` settings.
//!
//! - **Super Resolution** (`sr`) runs inside the FSR chain's slot (`fsr::upscaler`): the same
//!   jittered render, the same motion vectors (benilla's `motion` adapter, from the velocity target
//!   and a depth reprojection), the same exposure (`modern::look::exposure`, which `modern::meter`
//!   drives), into the same [`crate::fsr::FsrOutput`] every consumer already reads. NGX through the
//!   vendored `third_party/dlss_wgpu`, on Vulkan (upstream's path) and on D3D12 (its `d3d12.rs`).
//! - **Frame Generation** (`streamline`, Windows) is NVIDIA's closed `sl.dlss_g` plugin, which
//!   only Streamline drives: loaded at boot, it sits on the swap chain through wgpu-hal's
//!   interposer hooks and generates frames at present, with Reflex markers around submit and
//!   present. Where it is not available (no Streamline runtime, an RTX 20/30 card, Vulkan off
//!   Windows), DLSS frames feed FSR 3.1.5's frame generation instead, which reads only the
//!   upscaled output and the shared motion inputs.
//!
//! The runtime libraries are the player's own: `nvngx_dlss.dll` (and for frame generation the
//! Streamline set) in [`DlssRuntime::dir`], never shipped and never committed. Missing or refused,
//! DLSS falls back to FSR with one log line.
//!
//! Neither path costs the other: with DLSS chosen the FSR chain dispatches only its motion adapter
//! (and its input preparation while FSR frame generation stands in); with FSR chosen no DLSS
//! feature exists and Streamline is not loaded unless DLSS frame generation was chosen at boot,
//! where it stays on its pass-through path (`DLSSGMode::eOff`).
//!
//! **Ray Reconstruction hook.** [`DlssSupport::ray_reconstruction`] is read from NGX at startup and
//! `sr::DlssSdkRes` is the one NGX instance; a ray-traced shadow and GI pass that wants DLSS-RR as
//! its denoiser creates a `dlss_wgpu::ray_reconstruction::DlssRayReconstruction` on it and replaces
//! the Super Resolution evaluate in `fsr::upscaler::fsr_upscale_pass`'s DLSS branch (one call,
//! `sr::encode`) with its own, fed the guide buffers it renders. Not built here.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
#[cfg(not(target_arch = "wasm32"))]
use bevy::render::{Render, RenderApp, RenderSystems};

#[cfg(not(target_arch = "wasm32"))]
pub mod sr;
#[cfg(windows)]
pub mod streamline;

/// benilla's NGX project ID: the `CUSTOM` engine type takes any GUID-like string (DLSS
/// Programming Guide 5.2.1).
pub const PROJECT_ID: &str = "6f1e2b7c-3a94-4d8e-b1c5-0d7a9e4f2c61";

/// The NVIDIA runtime files the client loads from [`DlssRuntime::dir`]: DLSS Super Resolution and
/// Frame Generation's NGX snippets, then Streamline's (`streamline::RUNTIME_FILES`), Reflex's
/// Vulkan library among them. The standalone desktop build fetches whichever its site serves
/// (`benilla-desktop/src/dlss.rs`).
pub const RUNTIME_FILES: &[&str] = &[
    "nvngx_dlss.dll",
    "nvngx_dlssg.dll",
    "sl.interposer.dll",
    "sl.common.dll",
    "sl.dlss_g.dll",
    "sl.reflex.dll",
    "sl.pcl.dll",
    "NvLowLatencyVk.dll",
];

/// The temporal upscaler the player picked, the `upscaler` CVar.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Upscaler {
    /// DLSS 4.5 Super Resolution, FSR where NGX reports no support. Native only: the web build
    /// has no NGX, so there it is never chosen ([`Upscaler::from_cvar`]).
    Dlss,
    /// FSR 3.1.5.
    Fsr,
    /// Neither: below 1:1 the bilinear stretch, and no frame generation.
    Off,
}

impl Default for Upscaler {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Upscaler {
    /// DLSS on a native client, FSR on the web build, which has no NGX.
    pub const DEFAULT: Self = if cfg!(target_arch = "wasm32") {
        Self::Fsr
    } else {
        Self::Dlss
    };

    /// The CVar's value, case-blind; anything else is the default. `dlss` on the web build is
    /// FSR, what it would fall back to there.
    pub fn from_cvar(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "fsr" => Self::Fsr,
            "off" | "none" | "0" => Self::Off,
            "dlss" if !cfg!(target_arch = "wasm32") => Self::Dlss,
            _ => Self::DEFAULT,
        }
    }

    pub fn as_cvar(self) -> &'static str {
        match self {
            Self::Dlss => "dlss",
            Self::Fsr => "fsr",
            Self::Off => "off",
        }
    }
}

/// The player's choice; the app's `upscaler` callback writes it.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct UpscalerChoice(pub Upscaler);

/// What this machine runs, resolved once: Super Resolution when the render device opened
/// (`DlssPlugin::finish`), frame generation at boot (Streamline loaded and DLSS-G supported).
#[derive(Resource, Clone, Copy, Default, Debug, PartialEq, Eq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DlssSupport {
    pub super_resolution: bool,
    pub frame_generation: bool,
    /// The largest frame-generation multiplier DLSS-G reaches here (`slDLSSGGetState`'s
    /// `numFramesToGenerateMax` plus one: 2 on an RTX 40, 4 on an RTX 50); 0 without DLSS-G.
    pub frame_generation_max: u32,
    /// NVIDIA Reflex low-latency mode runs here (Streamline's `sl.reflex` reports it available).
    pub reflex: bool,
    /// The RR hook's gate (module doc); nothing reads it yet.
    pub ray_reconstruction: bool,
}

/// DLSS Frame Generation on this frame: the app's resolution of `upscaler`, `frameGeneration` and
/// [`DlssSupport::frame_generation`]. Streamline's options follow it in the render world.
#[derive(Resource, Clone, Copy, Default, Debug, PartialEq, Eq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DlssFrameGeneration(pub bool);

/// The frames DLSS-G generates per rendered frame while it runs (`numFramesToGenerate`): the
/// `frameGenerationMultiplier` less one, clamped to [`DlssSupport::frame_generation_max`]. The app
/// resolves it.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct DlssgFrames(pub u32);

impl Default for DlssgFrames {
    fn default() -> Self {
        Self(1)
    }
}

/// The `nvidiaReflex` setting: `auto` (the default: on while either frame generation runs, which
/// is when the held frames' latency is worth buying back), `0` off, `1` on, `2` on with boost.
/// Where Reflex does not run (no NVIDIA card, no Streamline) every value is off.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReflexSetting {
    #[default]
    Auto,
    Off,
    On,
    Boost,
}

impl ReflexSetting {
    /// The CVar's value, case-blind; anything else is `auto`.
    pub fn from_cvar(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "0" | "off" => Self::Off,
            "1" | "on" => Self::On,
            "2" | "boost" => Self::Boost,
            _ => Self::Auto,
        }
    }

    pub fn as_cvar(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "0",
            Self::On => "1",
            Self::Boost => "2",
        }
    }

    /// Reflex's mode for this setting, given whether it runs here and whether frame generation
    /// of either kind runs. DLSS-G forces it on regardless (`streamline`).
    pub fn resolve(self, available: bool, frame_generation: bool) -> Reflex {
        match (available, self) {
            (false, _) | (_, Self::Off) => Reflex::Off,
            (true, Self::Auto) if frame_generation => Reflex::On,
            (true, Self::Auto) => Reflex::Off,
            (true, Self::On) => Reflex::On,
            (true, Self::Boost) => Reflex::Boost,
        }
    }
}

/// Reflex's latency over its last reports, where Streamline's Reflex runs: the mean milliseconds
/// from the simulation's start to the GPU's end of the frame, from the present's start to it, and
/// the number of reports (`streamline::latency_report`).
pub fn reflex_latency() -> Option<(f32, f32, usize)> {
    #[cfg(windows)]
    return streamline::latency_report();
    #[cfg(not(windows))]
    None
}

/// NVIDIA Reflex's low-latency mode as the app resolved it from `nvidiaReflex`: off, on, or on
/// with boost (`sl::ReflexMode`). DLSS-G runs it on whatever this says (DLSS-G guide 8.0).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reflex {
    #[default]
    Off,
    On,
    Boost,
}

/// Where the player keeps the DLSS runtime, and where NGX and Streamline may write their logs.
#[derive(Resource, Clone, Debug)]
pub struct DlssRuntime {
    pub dir: PathBuf,
    pub data: PathBuf,
}

impl DlssRuntime {
    /// `$WOW_DLSS_PATH`, else `<state>/dlss/`; NGX's data under `<state>/dlss/ngx/`.
    pub fn resolve(state: Option<&Path>) -> Option<Self> {
        let dir = crate::launch_env::live_var("WOW_DLSS_PATH")
            .map(|v| PathBuf::from(v.trim()))
            .or_else(|| state.map(|s| s.join("dlss")))?;
        let data = state.map_or_else(|| dir.join("ngx"), |s| s.join("dlss").join("ngx"));
        Some(Self { dir, data })
    }

    /// The DLSS Super Resolution runtime NGX loads from [`Self::dir`].
    pub fn super_resolution_library(&self) -> Option<PathBuf> {
        if cfg!(windows) {
            Some(self.dir.join("nvngx_dlss.dll")).filter(|p| p.is_file())
        } else {
            std::fs::read_dir(&self.dir)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("libnvidia-ngx-dlss.so"))
                })
        }
    }
}

/// The boot half, before `RenderPlugin` builds: NGX's search path, its Vulkan instance and device
/// extensions on a Vulkan-only boot, and Streamline (`streamline`) when `streamline` says so: an
/// NVIDIA adapter, where it brings Reflex under any frame generation and DLSS-G where the card
/// runs it. The render device does not exist yet, so nothing here can know whether DLSS runs; it
/// only makes it possible.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_boot(
    app: &mut App,
    runtime: Option<DlssRuntime>,
    backends: wgpu::Backends,
    streamline: bool,
) {
    let Some(runtime) = runtime else {
        return;
    };
    let has_sr = runtime.super_resolution_library().is_some();
    if dlss_wgpu::NGX_LINKED && has_sr {
        let _ = std::fs::create_dir_all(&runtime.data);
        dlss_wgpu::set_runtime_paths(vec![runtime.dir.clone()], runtime.data.clone());
        if backends == wgpu::Backends::VULKAN {
            sr::register_vulkan_extensions(app);
        }
    }
    #[cfg(windows)]
    if streamline {
        streamline::boot(app, &runtime, backends);
    }
    #[cfg(not(windows))]
    let _ = streamline;
    app.insert_resource(runtime);
}

/// The HUD-less compose's key on a world view while DLSS-G runs (`streamline`): the claiming
/// backdrop camera's format and decode, which `ffx_glow` specializes its compose pair on.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub struct HudlessCompose {
    pub format: bevy::render::render_resource::TextureFormat,
    pub decode: bool,
}

/// Set when NGX refused to create a feature: the app then puts FSR in the slot for the session.
static REFUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether DLSS was refused at run time; the app's upscaler resolution reads it.
pub fn refused() -> bool {
    REFUSED.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn refuse() {
    REFUSED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// One world view's DLSS feature, held in `FsrState` so it lives and dies with the chain. Opaque,
/// and never built on the web.
pub struct DlssViewState {
    #[cfg(not(target_arch = "wasm32"))]
    inner: sr::ViewDlss,
}

#[cfg(not(target_arch = "wasm32"))]
impl DlssViewState {
    pub(crate) fn new(inner: sr::ViewDlss) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> Option<&sr::ViewDlss> {
        Some(&self.inner)
    }

    pub(crate) fn inner_mut(&mut self) -> Option<&mut sr::ViewDlss> {
        Some(&mut self.inner)
    }
}

pub struct DlssPlugin;

impl Plugin for DlssPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DlssSupport>()
            .init_resource::<UpscalerChoice>()
            .init_resource::<DlssFrameGeneration>()
            .init_resource::<DlssgFrames>()
            .init_resource::<Reflex>()
            .init_resource::<ReflexSetting>()
            .add_plugins((
                ExtractResourcePlugin::<DlssSupport>::default(),
                ExtractResourcePlugin::<DlssFrameGeneration>::default(),
                ExtractResourcePlugin::<DlssgFrames>::default(),
            ));
        #[cfg(windows)]
        streamline::build(app);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                sr::prepare_dlss
                    .in_set(RenderSystems::PrepareResources)
                    .after(crate::fsr::upscaler::prepare_fsr),
            );
        }
    }

    /// NGX opens on the render device, which exists once the renderer has.
    fn finish(&self, app: &mut App) {
        // Streamline takes the device first, so NGX's line can say whether frame generation runs.
        #[cfg(windows)]
        streamline::finish(app);
        #[cfg(target_arch = "wasm32")]
        let _ = app;
        #[cfg(not(target_arch = "wasm32"))]
        sr::init_sdk(app);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cvar_round_trips_and_defaults_to_dlss() {
        for u in [Upscaler::Dlss, Upscaler::Fsr, Upscaler::Off] {
            assert_eq!(Upscaler::from_cvar(u.as_cvar()), u);
        }
        assert_eq!(Upscaler::from_cvar("FSR"), Upscaler::Fsr);
        assert_eq!(Upscaler::from_cvar("garbage"), Upscaler::Dlss);
        assert_eq!(Upscaler::default(), Upscaler::Dlss);
    }

    /// `nvidiaReflex` round-trips; `auto` is on only while frame generation runs, and nothing is
    /// on where Reflex does not run.
    #[test]
    fn reflex_resolves_auto_to_frame_generation() {
        for r in [
            ReflexSetting::Auto,
            ReflexSetting::Off,
            ReflexSetting::On,
            ReflexSetting::Boost,
        ] {
            assert_eq!(ReflexSetting::from_cvar(r.as_cvar()), r);
        }
        assert_eq!(ReflexSetting::from_cvar("garbage"), ReflexSetting::Auto);
        assert_eq!(ReflexSetting::Auto.resolve(true, true), Reflex::On);
        assert_eq!(ReflexSetting::Auto.resolve(true, false), Reflex::Off);
        assert_eq!(ReflexSetting::On.resolve(true, false), Reflex::On);
        assert_eq!(ReflexSetting::Boost.resolve(true, true), Reflex::Boost);
        assert_eq!(ReflexSetting::Off.resolve(true, true), Reflex::Off);
        assert_eq!(ReflexSetting::Boost.resolve(false, true), Reflex::Off);
    }

    /// Every row of `HOOKS.md` names a file that still carries the marker.
    #[test]
    fn every_hook_row_carries_its_marker() {
        const MARKER: &str = "fork: dlss (FORK.md)";
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let rows: Vec<&str> = include_str!("HOOKS.md")
            .lines()
            .filter_map(|l| l.strip_prefix("| `"))
            .filter_map(|l| l.split('`').next())
            .collect();
        assert!(rows.len() >= 8, "the hook table parses: {rows:?}");
        let missing: Vec<&str> = rows
            .iter()
            .copied()
            .filter(|file| {
                std::fs::read_to_string(crates.join(file)).map_or(true, |t| !t.contains(MARKER))
            })
            .collect();
        assert!(
            missing.is_empty(),
            "HOOKS.md rows that lost `{MARKER}`: {missing:?}"
        );
    }
}
