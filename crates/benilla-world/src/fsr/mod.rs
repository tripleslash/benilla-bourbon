//! fork: fsr (FORK.md) — AMD FidelityFX Super Resolution 3.1.5 on the world camera:
//! temporal upscaling below a render scale of 1, motion vectors for what moves, and frame
//! generation. Not a 1.12 feature; benilla's own, behind the `renderScale` and `frameGeneration`
//! settings, and inert unless the app puts an [`FsrView`] on the world camera.
//!
//! - [`caps`]: the device's narrow storage formats.
//! - [`upscaler`]: the upscaler chain, `Core3d` systems on views carrying [`FsrView`].
//! - [`velocity`]: the velocity target moving content draws its motion into.
//!
//! The upstream files this module touches are listed in `HOOKS.md`, each marked
//! `// fork: fsr (FORK.md)`.

use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::camera::TemporalJitter;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::{RenderAdapter, RenderDevice};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

pub mod caps;
pub mod upscaler;
pub mod velocity;

pub use upscaler::{FsrOutput, FsrState, FsrVelocityTarget};

/// Base jitter sequence length at 1:1, before the upscale ratio lengthens it.
const JITTER_BASE_PERIOD: u32 = 8;

/// `ffxFsr3UpscalerGetJitterPhaseCount`: the base period times the *width* ratio squared, so every
/// output pixel is eventually sampled directly.
pub fn jitter_period(render: UVec2, display: UVec2) -> u32 {
    let ratio = display.x.max(1) as f32 / render.x.max(1) as f32;
    ((JITTER_BASE_PERIOD as f32 * ratio * ratio) as u32).max(1)
}

/// The `i`-th radical-inverse term in `base`, the Halton sequence's building block.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f32;
    while index > 0 {
        result += (index % base) as f32 * fraction;
        index /= base;
        fraction /= base as f32;
    }
    result
}

/// The jitter of phase `phase`, as FSR's sample offset in render pixels: Halton(2, 3), 1-based so
/// the first sample is not the pixel centre, recentred on the pixel (`ffxFsr3UpscalerGetJitterOffset`).
pub fn jitter_at(phase: u32, period: u32) -> Vec2 {
    let index = phase % period.max(1) + 1;
    Vec2::new(halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}

/// FSR's jitter (pixel `p` sampled the scene at `p + 0.5 − jitter`) as Bevy's
/// [`TemporalJitter::offset`], which shifts the image the other way: pixel `p` samples
/// `p + 0.5 + offset`.
pub fn bevy_offset(fsr_jitter: Vec2) -> Vec2 {
    -fsr_jitter
}

/// FSR on this camera: its display size and history epoch, which the app sets, and the jitter
/// sequence, which [`advance_jitter`] runs. Extracted every frame; its removal takes the render
/// world's FSR state with it.
#[derive(Component, Clone, Copy, Debug, ExtractComponent)]
#[extract_component_sync_target((Self, FsrState, FsrOutput, FsrVelocityTarget))]
pub struct FsrView {
    /// The display resolution the chain reconstructs, in physical px.
    pub display: UVec2,
    /// Bumped by the app to reset the history: world entry, a loading screen.
    pub epoch: u32,
    pub(crate) phase: u32,
    pub(crate) period: u32,
    pub(crate) jitter: Vec2,
    pub(crate) prev_jitter: Vec2,
}

impl FsrView {
    pub fn new(display: UVec2, epoch: u32) -> Self {
        Self {
            display,
            epoch,
            phase: 0,
            period: 1,
            jitter: Vec2::ZERO,
            prev_jitter: Vec2::ZERO,
        }
    }
}

/// Render-world flag: this frame is a held frame of frame generation, and no world view renders.
/// The FSR systems stand down on it so the history advances once per rendered world frame.
#[derive(Resource, Default, Clone, Copy, ExtractResource)]
pub struct FsrHeld(pub bool);

/// The upscaler's shaders, built once for this device's formats (`caps`).
#[derive(Resource, Clone)]
pub struct FsrShaders {
    pub formats: caps::Formats,
    /// `motion`, `prepare_inputs`, `shading_change_pyramid`, `shading_change`,
    /// `prepare_reactivity`, `luma_instability`, `accumulate`, `rcas`.
    pub upscaler: [Handle<Shader>; 8],
}

const COMMON: &str = include_str!("../shaders/fsr/common.wgsl");
const PASSES: [(&str, &str); 8] = [
    ("motion", include_str!("../shaders/fsr/motion.wgsl")),
    ("prepare_inputs", include_str!("../shaders/fsr/prepare_inputs.wgsl")),
    (
        "shading_change_pyramid",
        include_str!("../shaders/fsr/shading_change_pyramid.wgsl"),
    ),
    ("shading_change", include_str!("../shaders/fsr/shading_change.wgsl")),
    (
        "prepare_reactivity",
        include_str!("../shaders/fsr/prepare_reactivity.wgsl"),
    ),
    ("luma_instability", include_str!("../shaders/fsr/luma_instability.wgsl")),
    ("accumulate", include_str!("../shaders/fsr/accumulate.wgsl")),
    ("rcas", include_str!("../shaders/fsr/rcas.wgsl")),
];

/// One pass's full source: the shared prelude, then the pass with its placeholders resolved.
pub fn pass_source(pass: &str, formats: caps::Formats) -> String {
    format!("{COMMON}\n{}", formats.preprocess(pass))
}

/// Advance each FSR camera's jitter by one phase, once per world frame, and put it on the camera
/// as [`TemporalJitter`], which every lane's projection then carries. A held frame runs no main
/// world, so the sequence advances with rendered frames only.
pub fn advance_jitter(
    mut commands: Commands,
    mut cameras: Query<(Entity, &Camera, &mut FsrView)>,
    mut removed: RemovedComponents<FsrView>,
) {
    for (entity, camera, mut fsr) in &mut cameras {
        let Some(render) = camera.physical_viewport_size() else {
            continue;
        };
        fsr.period = jitter_period(render, fsr.display);
        fsr.prev_jitter = fsr.jitter;
        fsr.jitter = jitter_at(fsr.phase, fsr.period);
        fsr.phase = fsr.phase.wrapping_add(1);
        commands.entity(entity).insert(TemporalJitter {
            offset: bevy_offset(fsr.jitter),
        });
    }
    for entity in removed.read() {
        if let Ok(mut e) = commands.get_entity(entity) {
            e.remove::<TemporalJitter>();
        }
    }
}

pub struct FsrPlugin;

impl Plugin for FsrPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FsrHeld>()
            .add_plugins((
                ExtractComponentPlugin::<FsrView>::default(),
                ExtractResourcePlugin::<FsrHeld>::default(),
            ))
            // `Last`, not `PostUpdate` (`modern/HOOKS.md`, "Scheduling rule"): after the camera's
            // viewport is computed, before extraction.
            .add_systems(Last, advance_jitter);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, upscaler::init_pipelines)
            .add_systems(
                Render,
                upscaler::prepare_fsr.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                (
                    // The colour with every opaque surface in it, fog included, before anything
                    // blends over it.
                    upscaler::fsr_opaque_copy
                        .in_set(Core3dSystems::MainPass)
                        .after(crate::modern::volumetrics::fog_composite_pass)
                        .after(crate::liquid::liquid_screen_pass)
                        .before(main_transparent_pass_3d),
                    velocity::fsr_velocity_pass
                        .in_set(Core3dSystems::MainPass)
                        .after(main_transparent_pass_3d)
                        .after(crate::modern::volumetrics::lamp_halo_pass),
                    upscaler::fsr_upscale_pass.in_set(Core3dSystems::EarlyPostProcess),
                ),
            );
    }

    /// The shaders depend on the device's formats, which exist once the renderer has.
    fn finish(&self, app: &mut App) {
        let world = app.world();
        let (Some(device), Some(adapter)) = (
            world.get_resource::<RenderDevice>(),
            world.get_resource::<RenderAdapter>(),
        ) else {
            return;
        };
        let formats = caps::Formats::new(caps::narrow_storage(device, adapter));
        info!(
            "fsr: storage formats {}",
            if formats.r16 == bevy::render::render_resource::TextureFormat::R16Float {
                "narrow (the reference's)"
            } else {
                "widened (no texture-formats-tier1)"
            }
        );
        let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
        let upscaler = PASSES.map(|(name, pass)| {
            shaders.add(Shader::from_wgsl(
                pass_source(pass, formats),
                format!("benilla_fsr/{name}.wgsl"),
            ))
        });
        let built = FsrShaders { formats, upscaler };
        app.insert_resource(built.clone());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(built);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bevy's jitter applied to a reverse-Z projection puts pixel `p`'s sample at
    /// `p + 0.5 − jitter`, FSR's convention, in both axes.
    #[test]
    fn the_jitter_sign_matches_fsr() {
        let size = Vec2::new(1280.0, 720.0);
        let proj = Mat4::perspective_infinite_reverse_rh(1.0, size.x / size.y, 0.2);
        let jitter = Vec2::new(0.3, -0.2);
        let mut jittered = proj;
        TemporalJitter {
            offset: bevy_offset(jitter),
        }
        .jitter_projection(&mut jittered, size);
        // A point that projects to the centre of pixel (640, 360) unjittered…
        let point = Vec4::new(0.0, 0.0, -10.0, 1.0);
        let px = |m: Mat4| {
            let c = m * point;
            let ndc = c.xy() / c.w;
            Vec2::new((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y)
        };
        let (plain, moved) = (px(proj), px(jittered));
        // …is drawn `jitter` further along, so the pixel centre that shows it sampled
        // `centre − jitter`.
        assert!((moved - plain - jitter).length() < 1e-3, "{plain} → {moved}");
    }

    /// The period is the base 8 at 1:1 and grows with the area ratio; the sequence repeats with it
    /// and never sits on the pixel centre.
    #[test]
    fn the_jitter_period_scales_with_the_ratio() {
        assert_eq!(jitter_period(UVec2::new(1920, 1080), UVec2::new(1920, 1080)), 8);
        assert_eq!(jitter_period(UVec2::new(1280, 720), UVec2::new(1920, 1080)), 18);
        assert_eq!(jitter_period(UVec2::new(960, 540), UVec2::new(1920, 1080)), 32);
        let period = 18;
        for phase in 0..period {
            let j = jitter_at(phase, period);
            assert_eq!(j, jitter_at(phase + period, period));
            assert!(j.x.abs() <= 0.5 && j.y.abs() <= 0.5);
            assert_ne!(j, Vec2::ZERO);
        }
    }
}
