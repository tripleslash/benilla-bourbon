//! The world as the UI's backdrop: the world's final pass, the FFXGlow combine ([`FfxBackdrop`]),
//! is the first draw of the UI camera's main pass, straight into the UI's byte target.
//!
//! The reference's fixed-function device blends in gamma bytes. The world lane emits gamma bytes,
//! and the UI lane blends in an `Rgba8UnormSrgb` target, so every UI blend is arithmetic on the
//! gamma value (`alphaMode="ADD"` is `dst + texel·α`, clamped, as EGxBlend 3). Drawing the world
//! inside that target keeps the UI-over-world blend in bytes too; composited through the sRGB
//! swapchain it would run in linear and lighten every translucent UI pixel over the world (a docked
//! chat tab, black at α 102/255, must leave the scene at 60 %; a linear blend leaves 77.5 %).
//!
//! The world camera's own target is a size-carrier: a one-byte image at the world's render size
//! that nothing writes, kept because Bevy sizes a view's main texture from its target and the
//! camera's logical viewport rides it ([`retarget_world_camera`]).
//!
//! [`RenderScale`] renders the world at `window × scale` while the UI stays native. At exactly 2.0
//! the combine's bilinear read is a 2×2 box average, so supersampling needs no filter of its own.
//! The camera's logical viewport must not move with it ([`render_target_for`]).
//!
//! fork: fsr (FORK.md) — below 1.0, or with frame generation on, FSR 3.1.5 reconstructs the
//! display-size image instead of the bilinear stretch ([`sync_fsr_view`]); frame generation caps
//! the scale at 1.0 ([`effective_scale`]).
//!
//! fork: dlss (FORK.md) — the `upscaler` CVar picks what fills that slot: DLSS 4.5 (the
//! default, FSR where NGX reports no support), FSR, or neither; frame generation is DLSS's
//! (Streamline) where it runs and FSR's otherwise ([`resolve_upscaler`]).

use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::camera::MipBias;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::PrimaryWindow;

use crate::ui_pass::PlayerUiCamera;
use benilla_world::dlss::{
    DlssFrameGeneration, DlssSupport, DlssgFrames, Reflex, ReflexSetting, Upscaler, UpscalerChoice,
};
use benilla_world::ffx_glow::FfxBackdrop;
use benilla_world::fsr::{FgBlocked, FrameGeneration, FsrView, UpscalerKind};
use benilla_world::view::WorldCamera;

/// The render scale, the `renderScale` CVar, not a 1.12 CVar: it scales the 3D and keeps the
/// interface native. At the default 1.0 [`render_target_for`] returns the window's own numbers.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub(crate) struct RenderScale(pub(crate) f32);

/// The settable range of [`RenderScale`], shared by the CVar and `$WOW_RENDER_SCALE`. Past 2 on
/// purpose: supersampling prices a pixel on a machine whose present is railed at vsync.
pub(crate) const RENDER_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.25..=4.0;

/// The per-axis ceiling in physical px, `wgpu::Limits::default()`'s `max_texture_dimension_2d`,
/// applied to the ratio ([`render_target_for`]) so the picture shrinks rather than reshapes.
const MAX_RENDER_AXIS: u32 = 8192;

impl Default for RenderScale {
    /// `$WOW_RENDER_SCALE` overrides the default for the session only, never into `config.toml`.
    fn default() -> Self {
        // Through `webenv` so the browser build has the same lever as `?render_scale=0.75` — the
        // one knob that matters there: the world is GPU-bound at a 2560×1600 DPR-scaled canvas on
        // an iGPU (87 fps → 154 fps at ~40 % of the pixels, measured 2026-08-29).
        let scale = crate::webenv::var("WOW_RENDER_SCALE")
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .map_or(1.0, |v| {
                v.clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end())
            });
        Self(scale)
    }
}

/// The backdrop's size and the world camera's target scale factor, for a window of `px` physical
/// pixels at `window_factor`, rendered at `scale`.
///
/// 1. The ratio, never one axis, clamps to [`MAX_RENDER_AXIS`], so the image keeps the window's
///    aspect.
/// 2. `size.x / factor`, the camera's logical viewport width, is the window's logical width at any
///    `scale`: the factor derives from the size built, after the `round()`, since every pick ray is
///    denominated in it.
///
/// At `scale == 1.0` both are exact in IEEE, so the window's own numbers come back bit for bit.
fn render_target_for(px: UVec2, window_factor: f32, scale: f32) -> (UVec2, f32) {
    let px = px.max(UVec2::ONE);
    let ceiling = |axis: u32| MAX_RENDER_AXIS as f32 / axis as f32;
    let scale = scale
        .clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end())
        .min(ceiling(px.x))
        .min(ceiling(px.y));
    let size = (px.as_vec2() * scale)
        .round()
        .as_uvec2()
        .clamp(UVec2::ONE, UVec2::splat(MAX_RENDER_AXIS));
    (size, window_factor * size.x as f32 / px.x as f32)
}

/// The texture LOD bias a render at `effective` scale owes its mipmapped textures: `log2(scale)`,
/// clamped at 0 so supersampling keeps its sharper mips.
///
/// WebGPU samplers carry no LOD bias, so it rides [`MipBias`] into the view uniform, which the PBR
/// lane applies and our own shaders must too (`terrain.wgsl`, `static_gx.wgsl`, `liquid.wgsl`, the
/// coverage re-sample in `wow_model.wgsl`).
fn mip_bias(effective: f32) -> f32 {
    effective.log2().min(0.0)
}

/// fork: fsr (FORK.md) — FSR's texture LOD bias while it runs: one level sharper than the render
/// scale owes, `log2(scale) − 1`, as the SDK prescribes for its jittered input.
fn fsr_mip_bias(effective: f32) -> f32 {
    effective.log2() - 1.0
}

/// fork: fsr (FORK.md) — the scale the world renders at: frame generation needs FSR's 1:1 inputs at
/// most, so it caps the scale at 1.0.
fn effective_scale(scale: f32, frame_generation: bool) -> f32 {
    if frame_generation {
        scale.min(1.0)
    } else {
        scale
    }
}

/// fork: fsr (FORK.md) — FSR runs below 1:1, and at 1:1 under frame generation; it needs a
/// single-sampled world.
fn fsr_active(effective: f32, frame_generation: bool, msaa: Msaa) -> bool {
    (effective < 1.0 || frame_generation) && msaa == Msaa::Off
}

/// fork: dlss (FORK.md) — what the slot runs this frame, from the `upscaler` choice, the
/// `frameGeneration` wish and what this machine supports: the upscaler (`None`: the bilinear
/// stretch), DLSS frame generation, and FSR frame generation. DLSS falls back to FSR where NGX
/// reports no support or refused a feature; DLSS frame generation where Streamline does not run,
/// to FSR's on whichever upscaler is in the slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Slot {
    pub(crate) upscaler: Option<UpscalerKind>,
    pub(crate) dlss_frame_generation: bool,
    pub(crate) fsr_frame_generation: bool,
}

pub(crate) fn resolve_upscaler(
    choice: Upscaler,
    support: DlssSupport,
    refused: bool,
    frame_generation: bool,
) -> Slot {
    match choice {
        Upscaler::Off => Slot {
            upscaler: None,
            dlss_frame_generation: false,
            fsr_frame_generation: false,
        },
        Upscaler::Fsr => Slot {
            upscaler: Some(UpscalerKind::Fsr),
            dlss_frame_generation: false,
            fsr_frame_generation: frame_generation,
        },
        Upscaler::Dlss => {
            let sr = support.super_resolution && !refused;
            let dlss_fg = frame_generation && support.frame_generation;
            Slot {
                upscaler: Some(if sr {
                    UpscalerKind::Dlss
                } else {
                    UpscalerKind::Fsr
                }),
                dlss_frame_generation: dlss_fg,
                fsr_frame_generation: frame_generation && !dlss_fg,
            }
        }
    }
}

/// fork: dlss (FORK.md) — frame generation of either kind is wanted, with an upscaler in the
/// slot (`upscaler = off` runs neither).
fn frame_generation_wanted(choice: Option<&UpscalerChoice>, fg: Option<&FrameGeneration>) -> bool {
    fg.is_some_and(|f| f.0) && choice.is_none_or(|c| c.0 != Upscaler::Off)
}

/// The world camera's target, the size-carrier, in physical px × [`RenderScale`]; at 1.0 the
/// combine's read of the world is an identity resample.
#[derive(Resource)]
pub(crate) struct WorldBackdrop {
    /// The image the world camera targets. Nothing writes or samples it (the camera's final pass
    /// is the UI camera's, [`FfxBackdrop`]); it gives the view its size and logical viewport.
    pub(crate) image: Handle<Image>,
    /// The size the image was last built at, in physical px: the resize gate.
    size: UVec2,
    /// The image and scale factor the camera was last stamped with, and whether FSR ran. All gate
    /// the re-stamp: a rebuilt backdrop is a new asset ([`track_render_size`]), and FSR moves the
    /// mip bias. `None` until the first stamp.
    stamped: Option<(AssetId<Image>, f32, bool)>,
}

impl WorldBackdrop {
    /// The world's render size in physical px, after the rounding and the axis ceiling.
    pub(crate) fn render_size(&self) -> UVec2 {
        self.size
    }
}

/// A fresh size-carrier at `size` physical px: one byte a pixel, a render attachment only, its CPU
/// copy dropped once the render world has it.
fn new_world_target(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0],
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::RENDER_ATTACHMENT;
    image
}

fn window_physical_size(window: &Window) -> UVec2 {
    UVec2::new(
        window.physical_width().max(1),
        window.physical_height().max(1),
    )
}

fn setup_backdrop(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    scale: Res<RenderScale>,
    fg: Option<Res<FrameGeneration>>,
) {
    let scale = effective_scale(scale.0, fg.is_some_and(|f| f.0));
    let (size, _) = windows.single().map_or((UVec2::new(1280, 720), 1.0), |w| {
        render_target_for(window_physical_size(w), w.resolution.scale_factor(), scale)
    });
    let image = images.add(new_world_target(size));
    commands.insert_resource(WorldBackdrop {
        image,
        size,
        // Nothing stamped yet, so `retarget_world_camera`'s first run always fires.
        stamped: None,
    });
}

/// Keep the target at `window × `[`RenderScale`]; runs before the stamp, whose factor must pair
/// with this size.
///
/// A rebuild publishes a new asset and removes the old one, never writing through the old handle:
/// `AssetId<Image>` names a GPU texture, and anything keyed on it would keep the retired texture.
fn track_render_size(
    mut backdrop: ResMut<WorldBackdrop>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    scale: Res<RenderScale>,
    fg: Option<Res<FrameGeneration>>,
    // fork: dlss (FORK.md)
    choice: Option<Res<UpscalerChoice>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let fg = frame_generation_wanted(choice.as_deref(), fg.as_deref());
    let scale = RenderScale(effective_scale(scale.0, fg));
    let (size, _) = render_target_for(
        window_physical_size(window),
        window.resolution.scale_factor(),
        scale.0,
    );
    if size == backdrop.size {
        return;
    }
    let retired = std::mem::replace(&mut backdrop.image, images.add(new_world_target(size)));
    images.remove(&retired);
    backdrop.size = size;
    // Logged on every change: the rendered pixel count is the one term a probe cannot infer.
    info!(
        "render scale {:.3}: world renders at {}x{} into a {}x{} window",
        scale.0,
        size.x,
        size.y,
        window.physical_width(),
        window.physical_height()
    );
}

/// Point the world camera at the backdrop instead of the swapchain, carrying the window's scale
/// factor. A system, as the camera has two spawn sites; `benilla-worldview` keeps the swapchain.
///
/// `From<Handle<Image>>` stamps `scale_factor: 1.0`, which on a physical-px image makes the logical
/// viewport the physical size and throws every pick ray and `world_to_viewport` off by the display
/// scale. With the window's factor the logical viewport stays the window's, and render scale moves
/// size and factor together, so projection and picking never see it. Re-stamped when the image or
/// the factor changes, with the [`MipBias`] ([`mip_bias`]) inserted on every stamp.
fn retarget_world_camera(
    mut commands: Commands,
    mut backdrop: ResMut<WorldBackdrop>,
    windows: Query<&Window, With<PrimaryWindow>>,
    added: Query<(), Added<WorldCamera>>,
    fg: Option<Res<FrameGeneration>>,
    // fork: dlss (FORK.md) — `upscaler = off` keeps the bilinear stretch's bias.
    choice: Option<Res<UpscalerChoice>>,
    mut cameras: Query<(Entity, &mut RenderTarget, Option<&Msaa>), With<WorldCamera>>,
) {
    let (px, window_factor) = windows.single().map_or((UVec2::ONE, 1.0), |w| {
        (window_physical_size(w), w.resolution.scale_factor())
    });
    // The built size, not the requested one: `track_render_size` may have clamped.
    let scale_factor = window_factor * backdrop.size.x as f32 / px.x.max(1) as f32;
    // fork: fsr (FORK.md) — FSR's own bias while it runs.
    let effective = scale_factor / window_factor;
    let msaa = cameras
        .iter()
        .next()
        .and_then(|(_, _, m)| m.copied())
        .unwrap_or_default();
    let slot_on = choice.as_deref().is_none_or(|c| c.0 != Upscaler::Off);
    let fsr = slot_on
        && fsr_active(
            effective,
            frame_generation_wanted(choice.as_deref(), fg.as_deref()),
            msaa,
        );
    // The image as well as the factor: a pure window resize rebuilds without moving the factor.
    let want = (backdrop.image.id(), scale_factor, fsr);
    let moved = backdrop.stamped != Some(want);
    if !moved && added.is_empty() {
        return;
    }
    backdrop.stamped = Some(want);
    let bias = MipBias(if fsr {
        fsr_mip_bias(effective)
    } else {
        mip_bias(effective)
    });
    for (entity, mut current, _) in &mut cameras {
        *current = RenderTarget::Image(bevy::camera::ImageRenderTarget {
            handle: backdrop.image.clone(),
            scale_factor,
        });
        commands.entity(entity).insert(bias.clone());
    }
}

/// Point the player-UI camera's ground pass ([`FfxBackdrop`]) at the active world camera; with
/// none it is `None`, and the UI camera clears to transparent rather than show a stale world frame.
fn claim_backdrop(
    cameras: Query<(Entity, &Camera), With<WorldCamera>>,
    mut ui: Query<&mut FfxBackdrop, With<PlayerUiCamera>>,
) {
    let source = cameras
        .iter()
        .find_map(|(entity, camera)| camera.is_active.then_some(entity));
    for mut backdrop in &mut ui {
        // Compare first: a write would trip change detection for nothing.
        if backdrop.source != source {
            backdrop.source = source;
        }
    }
}

/// fork: fsr (FORK.md) — FSR on the active world camera: an [`FsrView`] at the window's size while
/// it runs ([`fsr_active`]), none otherwise. Its epoch moves when the camera comes back from being
/// gated (world entry, a loading screen), which resets the history. A capture run holds no frame.
#[allow(clippy::type_complexity)]
fn sync_fsr_view(
    mut commands: Commands,
    backdrop: Res<WorldBackdrop>,
    windows: Query<&Window, With<PrimaryWindow>>,
    fg: Option<Res<FrameGeneration>>,
    blocked: Option<ResMut<FgBlocked>>,
    capture: Option<Res<crate::run_mode::CaptureMode>>,
    // fork: dlss (FORK.md) — the slot's choice, what runs here, and DLSS-G's switch.
    choice: Option<Res<UpscalerChoice>>,
    support: Option<Res<DlssSupport>>,
    dlss_fg: Option<ResMut<DlssFrameGeneration>>,
    mut cameras: Query<(Entity, &Camera, Option<&Msaa>, Option<&mut FsrView>), With<WorldCamera>>,
    mut was_active: Local<bevy::ecs::entity::EntityHashMap<bool>>,
) {
    let slot = resolve_upscaler(
        choice.as_deref().map_or_else(Upscaler::default, |c| c.0),
        support.as_deref().copied().unwrap_or_default(),
        benilla_world::dlss::refused(),
        fg.is_some_and(|f| f.0),
    );
    // FSR's held-frame cadence stands down while DLSS frame generation paces the present.
    if let Some(mut blocked) = blocked {
        let want = capture.is_some() || slot.dlss_frame_generation;
        if blocked.0 != want {
            blocked.0 = want;
        }
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let display = window_physical_size(window);
    let frame_generation = slot.fsr_frame_generation;
    let any_frame_generation = slot.fsr_frame_generation || slot.dlss_frame_generation;
    let effective = backdrop.size.x as f32 / display.x as f32;
    let mut dlss_fg_on = false;
    for (entity, camera, msaa, view) in &mut cameras {
        let returned =
            camera.is_active && !was_active.insert(entity, camera.is_active).unwrap_or(false);
        let kind = slot.upscaler;
        let on = kind.is_some()
            && fsr_active(
                effective,
                any_frame_generation,
                msaa.copied().unwrap_or_default(),
            );
        let kind = kind.unwrap_or_default();
        // A capture reads every frame's pixels: no generated frame among them.
        dlss_fg_on |= on && camera.is_active && slot.dlss_frame_generation && capture.is_none();
        match (on, view) {
            (true, Some(mut view)) => {
                if view.display != display
                    || view.frame_generation != frame_generation
                    || view.upscaler != kind
                    || returned
                {
                    view.display = display;
                    view.frame_generation = frame_generation;
                    view.upscaler = kind;
                    if returned {
                        view.epoch = view.epoch.wrapping_add(1);
                    }
                }
            }
            (true, None) => {
                commands
                    .entity(entity)
                    .insert(FsrView::new(display, 0, frame_generation).with_upscaler(kind));
            }
            (false, Some(_)) => {
                commands.entity(entity).remove::<FsrView>();
            }
            (false, None) => {}
        }
    }
    if let Some(mut dlss_fg) = dlss_fg {
        if dlss_fg.0 != dlss_fg_on {
            dlss_fg.0 = dlss_fg_on;
        }
    }
}

/// fork: dlss (FORK.md) — the frame-generation multiplier that runs, from the one asked for
/// (`frameGenerationMultiplier`), the slot and what this machine supports, and the reason it is
/// what it is: DLSS-G reaches [`DlssSupport::frame_generation_max`], FSR's chain one generated
/// frame (2x), and 1 is no frame generation.
pub(crate) fn effective_multiplier(
    asked: u32,
    slot: Slot,
    support: DlssSupport,
) -> (u32, &'static str) {
    let asked = asked.max(2);
    if slot.dlss_frame_generation {
        let max = support.frame_generation_max.max(2);
        if asked > max {
            (max, "DLSS frame generation reaches no further on this card")
        } else {
            (asked, "DLSS frame generation")
        }
    } else if slot.fsr_frame_generation {
        let fsr = benilla_world::fsr::cadence::FgMultiplier::FSR_MAX;
        if asked > fsr {
            (
                fsr,
                "FSR frame generation makes one frame per rendered frame; more takes DLSS multi-frame generation (RTX 50)",
            )
        } else {
            (fsr, "FSR frame generation")
        }
    } else {
        (1, "frame generation off")
    }
}

/// fork: dlss (FORK.md) — after [`sync_fsr_view`]: DLSS-G's generated frames per rendered
/// frame ([`DlssgFrames`]) from the multiplier that runs, one log line when that moves, and
/// Reflex's mode from `nvidiaReflex` ([`ReflexSetting::resolve`]): with `auto`, on while either
/// frame generation runs.
#[allow(clippy::too_many_arguments)]
fn sync_frame_generation_extras(
    fg: Option<Res<FrameGeneration>>,
    choice: Option<Res<UpscalerChoice>>,
    support: Option<Res<DlssSupport>>,
    multiplier: Option<Res<benilla_world::fsr::cadence::FgMultiplier>>,
    dlssg_frames: Option<ResMut<DlssgFrames>>,
    setting: Option<Res<ReflexSetting>>,
    reflex: Option<ResMut<Reflex>>,
    mut last: Local<Option<(u32, &'static str)>>,
) {
    let support = support.as_deref().copied().unwrap_or_default();
    let slot = resolve_upscaler(
        choice.as_deref().map_or_else(Upscaler::default, |c| c.0),
        support,
        benilla_world::dlss::refused(),
        fg.is_some_and(|f| f.0),
    );
    let asked = multiplier.map_or(2, |m| m.0);
    let effective = effective_multiplier(asked, slot, support);
    if *last != Some(effective) {
        *last = Some(effective);
        info!(
            "frame generation: {}x (asked {asked}x): {}",
            effective.0, effective.1
        );
    }
    if let Some(mut frames) = dlssg_frames {
        let want = DlssgFrames(effective.0.saturating_sub(1).max(1));
        if *frames != want {
            *frames = want;
        }
    }
    if let Some(mut reflex) = reflex {
        let want = setting.as_deref().copied().unwrap_or_default().resolve(
            support.reflex,
            slot.fsr_frame_generation || slot.dlss_frame_generation,
        );
        if *reflex != want {
            *reflex = want;
        }
    }
}

/// Owns the world camera's target (the size-carrier) and the UI camera's claim on the world.
pub(crate) struct WorldBackdropPlugin;

/// Render scale's change callback, clamped to [`RENDER_SCALE_RANGE`]; the backdrop and the
/// camera's factor follow together on the next frame. fork: fsr (FORK.md) — and `frameGeneration`'s,
/// a flag, and `rcasSharpness`'s, clamped to the SDK's `[0, 1]`. fork: dlss (FORK.md) — and
/// `upscaler`'s, `dlss`, `fsr` or `off`, applied to the next frame.
pub(crate) fn on_cvar(
    ev: &crate::cvars::CvarChanged,
    mut scale: bevy::prelude::Mut<'_, RenderScale>,
    mut fg: bevy::prelude::Mut<'_, FrameGeneration>,
    mut rcas: bevy::prelude::Mut<'_, benilla_world::fsr::RcasSharpness>,
    upscaler: Option<bevy::prelude::Mut<'_, UpscalerChoice>>,
) {
    if ev.is("upscaler") {
        if let Some(mut upscaler) = upscaler {
            upscaler.0 = Upscaler::from_cvar(&ev.new);
        }
    }
    if ev.is("rcasSharpness") {
        let v = ev.num();
        rcas.0 = if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            benilla_world::fsr::RCAS_SHARPNESS_DEFAULT
        };
    }
    if ev.is("renderScale") {
        scale.0 = ev
            .num()
            .clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end());
    }
    if ev.is("frameGeneration") {
        fg.0 = ev.flag();
    }
}

/// fork: dlss (FORK.md) — `frameGenerationMultiplier`'s callback, presented frames per
/// rendered frame clamped to 2..=4 (what runs is [`effective_multiplier`]'s), and `nvidiaReflex`'s,
/// `auto`, `0`, `1` or `2`. Both apply to the next frame.
pub(crate) fn on_frame_generation_cvar(
    ev: &crate::cvars::CvarChanged,
    multiplier: Option<bevy::prelude::Mut<'_, benilla_world::fsr::cadence::FgMultiplier>>,
    reflex: Option<bevy::prelude::Mut<'_, ReflexSetting>>,
) {
    if ev.is("frameGenerationMultiplier") {
        if let Some(mut m) = multiplier {
            let v = ev.num();
            m.0 = if v.is_finite() {
                (v.round() as u32).clamp(2, 4)
            } else {
                2
            };
        }
    }
    if ev.is("nvidiaReflex") {
        if let Some(mut r) = reflex {
            *r = ReflexSetting::from_cvar(&ev.new);
        }
    }
}

impl Plugin for WorldBackdropPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderScale>()
            // fork: fsr (FORK.md) — the settings' knobs; `FsrPlugin` installs them too.
            .init_resource::<FrameGeneration>()
            .init_resource::<benilla_world::fsr::RcasSharpness>()
            // fork: dlss (FORK.md) — the slot's knobs; `DlssPlugin` installs them too.
            .init_resource::<UpscalerChoice>()
            .init_resource::<DlssFrameGeneration>()
            .init_resource::<benilla_world::fsr::cadence::FgMultiplier>()
            .init_resource::<DlssgFrames>()
            .init_resource::<ReflexSetting>()
            .init_resource::<Reflex>()
            .add_systems(Startup, setup_backdrop)
            .add_systems(
                Update,
                (
                    track_render_size,
                    retarget_world_camera,
                    claim_backdrop,
                    sync_fsr_view,
                    sync_frame_generation_extras,
                )
                    .chain()
                    // After the calls, whose `renderScale` callback writes the knob.
                    .after(crate::vm_pass::VmPass::End),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::image::TextureFormatPixelInfo as _;

    /// One byte a pixel, a render attachment, never sampled.
    #[test]
    fn the_world_target_is_a_size_carrier_only() {
        let image = new_world_target(UVec2::new(320, 200));
        assert_eq!(image.texture_descriptor.format, TextureFormat::R8Unorm);
        let usage = image.texture_descriptor.usage;
        assert!(usage.contains(TextureUsages::RENDER_ATTACHMENT));
        assert!(
            !usage.contains(TextureUsages::TEXTURE_BINDING),
            "nothing samples the world's target: the UI camera's ground pass reads the view's \
             main texture, not this image"
        );
        assert_eq!(
            image.asset_usage,
            RenderAssetUsages::RENDER_WORLD,
            "the CPU copy of a texture nothing reads is dropped once the render world has it"
        );
    }

    #[test]
    fn the_ui_camera_claims_the_drawing_world_camera() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_systems(Update, claim_backdrop);
        let ui = app
            .world_mut()
            .spawn((PlayerUiCamera, FfxBackdrop::default()))
            .id();
        let world_cam = app
            .world_mut()
            .spawn((
                WorldCamera,
                Camera {
                    is_active: true,
                    ..default()
                },
            ))
            .id();
        let source = |app: &mut App| app.world().get::<FfxBackdrop>(ui).unwrap().source;
        app.update();
        assert_eq!(source(&mut app), Some(world_cam), "active: claimed");
        app.world_mut()
            .get_mut::<Camera>(world_cam)
            .unwrap()
            .is_active = false;
        app.update();
        assert_eq!(source(&mut app), None, "gated: the claim goes with it");
    }

    /// The window's factor, not `1.0`, re-stamped when the window moves to another display.
    #[test]
    fn the_world_cameras_target_carries_the_windows_scale_factor() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
            .init_asset::<Image>();
        let window = |sf: f32| {
            let mut w = Window::default();
            w.resolution.set_scale_factor(sf);
            w
        };
        app.world_mut().spawn((window(2.0), PrimaryWindow));
        app.init_resource::<RenderScale>()
            .add_systems(Startup, setup_backdrop)
            // Chained as the plugin chains them: the stamp reads the size the resize pass settled.
            .add_systems(Update, (track_render_size, retarget_world_camera).chain());
        let cam = app
            .world_mut()
            .spawn((WorldCamera, RenderTarget::default()))
            .id();
        app.update();

        let stamped = |app: &App, e: Entity| match app.world().entity(e).get::<RenderTarget>() {
            Some(RenderTarget::Image(t)) => t.scale_factor,
            _ => panic!("the world camera must target the backdrop image"),
        };
        assert_eq!(
            stamped(&app, cam),
            2.0,
            "a 2× display: the target must report the window's scale factor, or every logical \
             cursor position is read as a physical one and the pick ray misses by 2×"
        );
        // …and it follows the window across displays.
        let mut w = app.world_mut().query::<&mut Window>();
        w.single_mut(app.world_mut())
            .unwrap()
            .resolution
            .set_scale_factor(1.0);
        app.update();
        assert_eq!(
            stamped(&app, cam),
            1.0,
            "moving to a 1× display re-stamps: a stale factor is the same defect, inverted"
        );
    }

    /// A zero-size window (minimised on some platforms) still builds a legal texture.
    #[test]
    fn a_degenerate_size_still_builds_a_legal_texture() {
        let image = new_world_target(UVec2::ZERO);
        assert_eq!(image.texture_descriptor.size.width, 1);
        assert_eq!(image.texture_descriptor.size.height, 1);
        assert_eq!(
            image.data.as_ref().map(Vec::len),
            TextureFormat::R8Unorm.pixel_size().ok()
        );
    }

    /// Scale 1.0 gives the window's own numbers exactly, which every visual golden rests on.
    #[test]
    fn scale_one_reproduces_the_windows_own_numbers_exactly() {
        for px in [
            UVec2::new(1280, 720),
            UVec2::new(3200, 1800),
            UVec2::new(1601, 901), // odd on both axes, the case a `/ 2` would round
        ] {
            for sf in [1.0, 1.5, 2.0] {
                assert_eq!(render_target_for(px, sf, 1.0), (px, sf), "{px} at {sf}×");
            }
        }
    }

    /// `logical = image_size / scale_factor` (`bevy_camera`'s `to_logical`) stays the window's at
    /// every scale: x exact, y within one `round()`.
    #[test]
    fn every_scale_keeps_the_logical_viewport_the_windows_own() {
        for px in [UVec2::new(1280, 720), UVec2::new(3200, 1800)] {
            for sf in [1.0, 2.0] {
                let want = px.as_vec2() / sf;
                for scale in [0.25, 0.5, 0.6667, 0.75, 1.0, 1.25, 2.0, 4.0] {
                    let (size, factor) = render_target_for(px, sf, scale);
                    let logical = size.as_vec2() / factor;
                    assert!(
                        (logical.x - want.x).abs() < 0.001,
                        "{px} at {sf}× scaled {scale}: logical width {logical:?}, want {want:?}"
                    );
                    assert!(
                        (logical.y - want.y).abs() < 1.0,
                        "{px} at {sf}× scaled {scale}: logical height {logical:?}, want {want:?}"
                    );
                }
            }
        }
    }

    /// The ceiling clamps the ratio, keeping the aspect, and binds on a huge window at 1.0 too.
    #[test]
    fn the_axis_ceiling_shrinks_the_picture_it_does_not_reshape_it() {
        let px = UVec2::new(3840, 2160);
        let (size, _) = render_target_for(px, 1.0, 4.0);
        assert_eq!(size.x, MAX_RENDER_AXIS);
        let aspect = |v: UVec2| v.x as f32 / v.y as f32;
        assert!(
            (aspect(size) - aspect(px)).abs() < 0.001,
            "aspect kept: {size}"
        );
        // …and the clamp is not render scale's alone.
        let huge = UVec2::new(10240, 4320);
        let (size, _) = render_target_for(huge, 1.0, 1.0);
        assert!(
            size.x <= MAX_RENDER_AXIS && size.y <= MAX_RENDER_AXIS,
            "{size}"
        );
        assert!(
            (aspect(size) - aspect(huge)).abs() < 0.001,
            "aspect kept: {size}"
        );
    }

    /// fork: fsr (FORK.md) — FSR runs below 1:1 or under frame generation, single-sampled
    /// only; frame generation caps the scale, and its bias is one level past the scale's.
    #[test]
    fn fsr_runs_below_one_or_with_frame_generation_and_caps_the_scale() {
        assert!(fsr_active(0.67, false, Msaa::Off));
        assert!(
            !fsr_active(1.0, false, Msaa::Off),
            "1:1 stays byte-identical"
        );
        assert!(!fsr_active(1.5, false, Msaa::Off));
        assert!(
            fsr_active(1.0, true, Msaa::Off),
            "frame generation runs FSR at 1:1"
        );
        assert!(
            !fsr_active(0.67, true, Msaa::Sample4),
            "never over a multisampled world"
        );
        assert_eq!(effective_scale(1.5, true), 1.0);
        assert_eq!(effective_scale(0.75, true), 0.75);
        assert_eq!(effective_scale(1.5, false), 1.5);
        assert!((fsr_mip_bias(0.5) - -2.0).abs() < 1e-6);
        assert!((fsr_mip_bias(1.0) - -1.0).abs() < 1e-6);
    }

    /// fork: dlss (FORK.md) — the multiplier that runs: DLSS-G up to the card's reach, FSR's
    /// chain 2x whatever is asked, 1 with no frame generation.
    #[test]
    fn the_multiplier_clamps_to_what_generates() {
        let rtx50 = DlssSupport {
            super_resolution: true,
            frame_generation: true,
            frame_generation_max: 4,
            ..default()
        };
        let rtx40 = DlssSupport {
            frame_generation_max: 2,
            ..rtx50
        };
        let dlss = resolve_upscaler(Upscaler::Dlss, rtx50, false, true);
        assert_eq!(effective_multiplier(3, dlss, rtx50).0, 3);
        assert_eq!(effective_multiplier(4, dlss, rtx50).0, 4);
        assert_eq!(effective_multiplier(4, dlss, rtx40).0, 2);
        let fsr = resolve_upscaler(Upscaler::Fsr, rtx50, false, true);
        assert_eq!(effective_multiplier(4, fsr, rtx50).0, 2);
        let off = resolve_upscaler(Upscaler::Dlss, rtx50, false, false);
        assert_eq!(effective_multiplier(3, off, rtx50).0, 1);
    }

    /// fork: dlss (FORK.md) — DLSS where supported, FSR where not or when refused; DLSS
    /// frame generation where Streamline runs, FSR's otherwise; `off` runs neither.
    #[test]
    fn the_slot_resolves_dlss_with_fsr_as_its_fallback() {
        let none = DlssSupport::default();
        let sr = DlssSupport {
            super_resolution: true,
            ..default()
        };
        let all = DlssSupport {
            super_resolution: true,
            frame_generation: true,
            frame_generation_max: 2,
            ..default()
        };
        let s = resolve_upscaler(Upscaler::Dlss, all, false, true);
        assert_eq!(s.upscaler, Some(UpscalerKind::Dlss));
        assert!(s.dlss_frame_generation && !s.fsr_frame_generation);
        let s = resolve_upscaler(Upscaler::Dlss, sr, false, true);
        assert_eq!(s.upscaler, Some(UpscalerKind::Dlss));
        assert!(
            !s.dlss_frame_generation && s.fsr_frame_generation,
            "RTX 20/30: FSR FG on DLSS"
        );
        let s = resolve_upscaler(Upscaler::Dlss, none, false, true);
        assert_eq!(s.upscaler, Some(UpscalerKind::Fsr), "no NGX: FSR");
        let s = resolve_upscaler(Upscaler::Dlss, all, true, false);
        assert_eq!(s.upscaler, Some(UpscalerKind::Fsr), "refused: FSR");
        let s = resolve_upscaler(Upscaler::Fsr, all, false, true);
        assert_eq!(s.upscaler, Some(UpscalerKind::Fsr));
        assert!(!s.dlss_frame_generation && s.fsr_frame_generation);
        let s = resolve_upscaler(Upscaler::Off, all, false, true);
        assert_eq!(s.upscaler, None);
        assert!(!s.dlss_frame_generation && !s.fsr_frame_generation);
    }

    #[test]
    fn the_mip_bias_is_log2_below_one_and_nothing_above_it() {
        assert!((mip_bias(0.5) - -1.0).abs() < 1e-6);
        assert!((mip_bias(0.25) - -2.0).abs() < 1e-6);
        assert_eq!(
            mip_bias(1.0),
            0.0,
            "off must be exactly off — 0.0, not -0.0's cousin"
        );
        assert_eq!(mip_bias(2.0), 0.0);
        assert_eq!(mip_bias(4.0), 0.0);
    }

    /// The id moves, the old asset is removed, and the camera is re-stamped onto the new image.
    #[test]
    fn a_rebuilt_backdrop_is_a_new_asset_and_the_camera_follows_it() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
            .init_asset::<Image>();
        let mut w = Window::default();
        w.resolution.set_scale_factor(2.0);
        app.world_mut().spawn((w, PrimaryWindow));
        app.insert_resource(RenderScale(1.0))
            .add_systems(Startup, setup_backdrop)
            .add_systems(Update, (track_render_size, retarget_world_camera).chain());
        let cam = app
            .world_mut()
            .spawn((WorldCamera, RenderTarget::default()))
            .id();
        app.update();
        let first = app.world().resource::<WorldBackdrop>().image.id();

        // Move the render scale while the world is up.
        app.insert_resource(RenderScale(0.5));
        app.update();

        let second = app.world().resource::<WorldBackdrop>().image.id();
        assert_ne!(
            first, second,
            "a rebuilt backdrop is a NEW GPU texture, so it must be a new AssetId — \
             the same id hands `ui_pass` back a bind group pointing at the retired texture"
        );
        assert!(
            !app.world().resource::<Assets<Image>>().contains(first),
            "the retired image must be removed, not merely dropped: a cached material still \
             holds a strong handle to it"
        );
        match app.world().entity(cam).get::<RenderTarget>() {
            Some(RenderTarget::Image(t)) => assert_eq!(
                t.handle.id(),
                second,
                "the camera must be re-stamped onto the new image"
            ),
            _ => panic!("the world camera must target the backdrop image"),
        }
    }

    /// A pure window resize moves the image without moving the scale factor.
    #[test]
    fn a_window_resize_restamps_the_camera_even_though_the_factor_does_not_move() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
            .init_asset::<Image>();
        let mut w = Window::default();
        w.resolution.set_scale_factor(2.0);
        let win = app.world_mut().spawn((w, PrimaryWindow)).id();
        app.insert_resource(RenderScale(1.0))
            .add_systems(Startup, setup_backdrop)
            .add_systems(Update, (track_render_size, retarget_world_camera).chain());
        let cam = app
            .world_mut()
            .spawn((WorldCamera, RenderTarget::default()))
            .id();
        app.update();
        let before = match app.world().entity(cam).get::<RenderTarget>() {
            Some(RenderTarget::Image(t)) => (t.handle.id(), t.scale_factor),
            _ => panic!("the world camera must target the backdrop image"),
        };

        app.world_mut()
            .entity_mut(win)
            .get_mut::<Window>()
            .expect("the primary window")
            .resolution
            .set(640.0, 400.0);
        app.update();

        let after = match app.world().entity(cam).get::<RenderTarget>() {
            Some(RenderTarget::Image(t)) => (t.handle.id(), t.scale_factor),
            _ => panic!("the world camera must target the backdrop image"),
        };
        assert_ne!(before.0, after.0, "the resize must re-point the camera");
        assert!(
            (before.1 - after.1).abs() < 1e-6,
            "at scale 1.0 the factor is the window's own and does not move on a resize — \
             which is exactly why the re-stamp cannot be gated on it alone: {} vs {}",
            before.1,
            after.1
        );
        assert!(
            app.world().resource::<Assets<Image>>().contains(after.0),
            "the camera must point at an image that still exists"
        );
    }

    /// A half-scale world on a 2× display: a half-size image, a factor of 1.0, `MipBias(-1)`.
    #[test]
    fn a_half_scale_world_halves_the_image_the_factor_and_the_mip() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
            .init_asset::<Image>();
        let mut w = Window::default();
        w.resolution.set_scale_factor(2.0);
        let px = window_physical_size(&w);
        app.world_mut().spawn((w, PrimaryWindow));
        app.insert_resource(RenderScale(0.5))
            .add_systems(Startup, setup_backdrop)
            .add_systems(Update, (track_render_size, retarget_world_camera).chain());
        let cam = app
            .world_mut()
            .spawn((WorldCamera, RenderTarget::default()))
            .id();
        app.update();

        let backdrop = app.world().resource::<WorldBackdrop>();
        assert_eq!(backdrop.size, UVec2::new(px.x / 2, px.y / 2));
        match app.world().entity(cam).get::<RenderTarget>() {
            Some(RenderTarget::Image(t)) => assert!(
                (t.scale_factor - 1.0).abs() < 1e-6,
                "half the pixels at half the factor is the same logical viewport: {}",
                t.scale_factor
            ),
            _ => panic!("the world camera must target the backdrop image"),
        }
        let bias = app.world().entity(cam).get::<MipBias>().expect("a bias");
        assert!(
            (bias.0 - -1.0).abs() < 1e-6,
            "half scale owes one mip level: {}",
            bias.0
        );
    }
}
