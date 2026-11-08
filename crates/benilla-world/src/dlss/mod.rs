//! fork: dlss (FORK.md) — NVIDIA DLSS 4.5 in FSR's upscaler slot. Not a 1.12 feature;
//! benilla's own, behind the `upscaler` and `frameGeneration` settings.
//!
//! - **Super Resolution** (`sr`) runs inside the FSR chain's slot (`fsr::upscaler`): the same
//!   jittered render, the same motion vectors (benilla's `motion` adapter, from the velocity target
//!   and a depth reprojection), the same exposure (`modern::look::exposure`, which `modern::meter`
//!   drives), into the same [`crate::fsr::FsrOutput`] every consumer already reads. NGX through the
//!   vendored `third_party/dlss_wgpu`, on Vulkan (upstream's path) and on D3D12 (its `d3d12.rs`).
//! - **Frame Generation** (`fg`) is NGX's DLSS-G (`nvngx_dlssg`), driven directly inside FSR frame
//!   generation's own cadence: the same held-frame pacing, the same HUD-less `WorldFinal` pair and
//!   the same present, with DLSS-G making the generated frame in place of FSR's chain. No swap-chain
//!   hooks and no Streamline. Where NGX reports no DLSS-G (an RTX 20/30 card, an old driver) or
//!   refuses the feature, FSR's chain generates instead, on whichever upscaler is in the slot.
//!
//! The runtime ([`RUNTIME_FILES`]) is loaded from [`DlssRuntime::dir`]: the standalone desktop exe
//! fetches it there, a dev build has it copied there. Missing or refused, DLSS falls back to FSR
//! with one log line.
//!
//! Neither path costs the other: with DLSS chosen the FSR chain dispatches only its motion adapter
//! (and its input preparation while FSR frame generation stands in), and under DLSS-G FSR frame
//! generation allocates none of its chain; with FSR chosen no DLSS feature exists.
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
pub mod fg;
#[cfg(not(target_arch = "wasm32"))]
pub mod sr;

/// benilla's NGX project ID: the `CUSTOM` engine type takes any GUID-like string (DLSS
/// Programming Guide 5.2.1).
pub const PROJECT_ID: &str = "6f1e2b7c-3a94-4d8e-b1c5-0d7a9e4f2c61";

/// The NVIDIA runtime files the client loads from [`DlssRuntime::dir`]: NGX's DLSS Super
/// Resolution and Frame Generation snippets. The standalone desktop build fetches them
/// (`benilla-desktop/src/dlss.rs`).
pub const RUNTIME_FILES: &[&str] = &["nvngx_dlss.dll", "nvngx_dlssg.dll"];

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

/// What this machine runs, resolved once the render device opened (`DlssPlugin::finish`): Super
/// Resolution, and Frame Generation where NGX reports it too.
#[derive(Resource, Clone, Copy, Default, Debug, PartialEq, Eq, ExtractResource)]
pub struct DlssSupport {
    pub super_resolution: bool,
    pub frame_generation: bool,
    /// The RR hook's gate (module doc); nothing reads it yet.
    pub ray_reconstruction: bool,
}

/// DLSS Frame Generation on this frame: the app's resolution of `upscaler`, `frameGeneration` and
/// [`DlssSupport::frame_generation`]. FSR frame generation's cadence runs either way; this picks
/// DLSS-G as its generator (`fsr::framegen`, `fg`).
#[derive(Resource, Clone, Copy, Default, Debug, PartialEq, Eq, ExtractResource)]
pub struct DlssFrameGeneration(pub bool);

/// Where the DLSS runtime lives, and where NGX may write its logs.
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

/// The boot half, before `RenderPlugin` builds: NGX's search path, and its Vulkan instance and
/// device extensions (Super Resolution's and Frame Generation's) on a Vulkan-only boot. The render
/// device does not exist yet, so nothing here can know whether DLSS runs; it only makes it
/// possible.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_boot(app: &mut App, runtime: Option<DlssRuntime>, backends: wgpu::Backends) {
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
    app.insert_resource(runtime);
}

/// The format DLSS-G writes its output in for a `WorldFinal` of `format`: the linear twin, as a
/// storage image. `None` where this device cannot store it (BGRA without
/// `BGRA8UNORM_STORAGE`) or NGX takes no such backbuffer; FSR's chain then generates.
pub fn fg_output_format(
    format: bevy::render::render_resource::TextureFormat,
    features: wgpu::Features,
) -> Option<bevy::render::render_resource::TextureFormat> {
    use bevy::render::render_resource::TextureFormat as F;
    let linear = format.remove_srgb_suffix();
    match linear {
        F::Rgba8Unorm | F::Rgb10a2Unorm | F::Rgba16Float => Some(linear),
        F::Bgra8Unorm => features
            .contains(wgpu::Features::BGRA8UNORM_STORAGE)
            .then_some(linear),
        _ => None,
    }
    .filter(|_| cfg!(not(target_arch = "wasm32")))
}

/// One world view's DLSS-G feature (`fg`), beside FSR frame generation's state, which it lives and
/// dies with. Opaque, and never built on the web.
#[derive(Component)]
pub struct DlssFgView {
    #[cfg(not(target_arch = "wasm32"))]
    inner: fg::ViewFg,
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

/// Set when NGX refused DLSS-G: the app then lets FSR's chain generate for the session.
static FG_REFUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether DLSS Frame Generation was refused at run time; the app's slot resolution reads it.
pub fn frame_generation_refused() -> bool {
    FG_REFUSED.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn refuse_frame_generation() {
    FG_REFUSED.store(true, std::sync::atomic::Ordering::Relaxed);
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
            .add_plugins((
                ExtractResourcePlugin::<DlssSupport>::default(),
                ExtractResourcePlugin::<DlssFrameGeneration>::default(),
            ));
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            use bevy::core_pipeline::{Core3d, Core3dSystems};
            render_app
                .add_systems(
                    Render,
                    (
                        sr::prepare_dlss.after(crate::fsr::upscaler::prepare_fsr),
                        fg::prepare_dlss_fg.after(crate::fsr::framegen::prepare_framegen),
                    )
                        .in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    Core3d,
                    fg::dlss_framegen_pass
                        .in_set(Core3dSystems::PostProcess)
                        .after(crate::fsr::framegen::fsr_framegen_pass),
                );
        }
    }

    /// NGX opens on the render device, which exists once the renderer has.
    fn finish(&self, app: &mut App) {
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
