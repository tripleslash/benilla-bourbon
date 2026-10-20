//! fork: fsr (FORK.md) — AMD FidelityFX Super Resolution 3.1.5 on the world camera:
//! temporal upscaling below a render scale of 1, motion vectors for what moves, and frame
//! generation. Not a 1.12 feature; benilla's own, behind the `renderScale` and `frameGeneration`
//! settings, and inert unless the app puts an [`FsrView`] on the world camera.
//!
//! - [`caps`]: the device's narrow storage formats.
//! - [`upscaler`]: the upscaler chain, `Core3d` systems on views carrying [`FsrView`].
//! - [`velocity`]: the velocity target moving content draws its motion into.
//! - [`framegen`]: frame generation, the interpolated frame between two rendered ones.
//! - [`cadence`]: which frames render the world and which only present.
//!
//! The upstream files this module touches are listed in `HOOKS.md`, each marked
//! `// fork: fsr (FORK.md)`.

use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::camera::TemporalJitter;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::SpecializedRenderPipelines;
use bevy::render::renderer::{render_system, RenderAdapter, RenderDevice};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

pub mod cadence;
pub mod caps;
pub mod framegen;
mod residual;
pub mod upscaler;
pub mod velocity;

pub use cadence::{FgBlocked, FrameGeneration, PresentsHeldFrames};
pub use framegen::{FgPresent, FgState};
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

/// FSR on this camera: its display size, history epoch and whether frame generation runs, which
/// the app sets, and the jitter sequence, which [`advance_jitter`] runs. Extracted every frame; its
/// removal takes the render world's FSR state with it.
#[derive(Component, Clone, Copy, Debug, ExtractComponent)]
#[extract_component_sync_target((
    Self,
    FsrState,
    FsrOutput,
    FsrVelocityTarget,
    FgState,
    FgPresent
))]
pub struct FsrView {
    /// The display resolution the chain reconstructs, in physical px.
    pub display: UVec2,
    /// Bumped by the app to reset the history: world entry, a loading screen.
    pub epoch: u32,
    /// Frame generation runs on this view (`frameGeneration`), given a backdrop camera to present.
    pub frame_generation: bool,
    pub(crate) phase: u32,
    pub(crate) period: u32,
    pub(crate) jitter: Vec2,
    pub(crate) prev_jitter: Vec2,
}

impl FsrView {
    pub fn new(display: UVec2, epoch: u32, frame_generation: bool) -> Self {
        Self {
            display,
            epoch,
            frame_generation,
            phase: 0,
            period: 1,
            jitter: Vec2::ZERO,
            prev_jitter: Vec2::ZERO,
        }
    }
}

/// fork: the `rcasSharpness` CVar, the RCAS pass's sharpness on the SDK's `[0, 1]` scale, which
/// `ffx_fsr3upscaler.cpp` maps to `2 − 2·sharpness` stops; [`RCAS_SHARPNESS_DEFAULT`] as shipped.
#[derive(Resource, Clone, Copy, PartialEq, Debug, ExtractResource)]
pub struct RcasSharpness(pub f32);

/// The sharpness the upscaler shipped with, physics-rs's 0.875: a quarter stop.
pub const RCAS_SHARPNESS_DEFAULT: f32 = 0.875;

impl Default for RcasSharpness {
    fn default() -> Self {
        Self(RCAS_SHARPNESS_DEFAULT)
    }
}

impl RcasSharpness {
    /// The sharpness in stops, where 0 is the most, the SDK's mapping of its `[0, 1]` dial.
    pub fn stops(self) -> f32 {
        2.0 - 2.0 * self.0.clamp(0.0, 1.0)
    }
}

/// Render-world flag: this frame is a held frame of frame generation, and no world view renders.
/// The FSR systems stand down on it so the history advances once per rendered world frame.
#[derive(Resource, Default, Clone, Copy, ExtractResource)]
pub struct FsrHeld(pub bool);

/// Render-world flag: this world frame is not followed by a held frame, because the world frame
/// alone outlasts a held one and the pair would split the interval unevenly ([`cadence`]). It
/// presents the real frame it rendered and skips the interpolation chain.
#[derive(Resource, Default, Clone, Copy, ExtractResource)]
pub struct FgSolo(pub bool);

/// The chain's shaders, built once for this device's formats (`caps`).
#[derive(Resource, Clone)]
pub struct FsrShaders {
    pub formats: caps::Formats,
    /// `motion`, `prepare_inputs`, `shading_change_pyramid`, `shading_change`,
    /// `prepare_reactivity`, `luma_instability`, `accumulate`, `rcas`.
    pub upscaler: [Handle<Shader>; 8],
    /// `reconstruct_depth`, `game_vector_field`, `mv_pyramid_seed`, the vector pyramid's reduce,
    /// `disocclusion`, `interpolate`, `color_pyramid_seed`, the colour pyramid's reduce, `inpaint`,
    /// and the present copy.
    pub framegen: [Handle<Shader>; 10],
    /// Model parts' motion (`velocity.wgsl`).
    pub velocity: Handle<Shader>,
    /// The effect lane's motion (`effect_velocity.wgsl`).
    pub effect_velocity: Handle<Shader>,
    /// `WOW_FSR_DEBUG=residual`'s measurement (`residual.wgsl`).
    pub residual: Handle<Shader>,
}

const COMMON: &str = include_str!("../shaders/fsr/common.wgsl");
const PASSES: [(&str, &str); 8] = [
    ("motion", include_str!("../shaders/fsr/motion.wgsl")),
    (
        "prepare_inputs",
        include_str!("../shaders/fsr/prepare_inputs.wgsl"),
    ),
    (
        "shading_change_pyramid",
        include_str!("../shaders/fsr/shading_change_pyramid.wgsl"),
    ),
    (
        "shading_change",
        include_str!("../shaders/fsr/shading_change.wgsl"),
    ),
    (
        "prepare_reactivity",
        include_str!("../shaders/fsr/prepare_reactivity.wgsl"),
    ),
    (
        "luma_instability",
        include_str!("../shaders/fsr/luma_instability.wgsl"),
    ),
    ("accumulate", include_str!("../shaders/fsr/accumulate.wgsl")),
    ("rcas", include_str!("../shaders/fsr/rcas.wgsl")),
];

const FG_COMMON: &str = include_str!("../shaders/fsr/fg_common.wgsl");
const FG_REDUCE: &str = include_str!("../shaders/fsr/fg_pyramid_reduce.wgsl");
/// The frame-generation passes in [`FsrShaders::framegen`] order; `None` is the pyramid reduce,
/// built twice with `FG_PYRAMID_REDUCE_MV` resolved each way.
const FG_PASSES: [(&str, Option<&str>); 10] = [
    (
        "fg_reconstruct_depth",
        Some(include_str!("../shaders/fsr/fg_reconstruct_depth.wgsl")),
    ),
    (
        "fg_game_vector_field",
        Some(include_str!("../shaders/fsr/fg_game_vector_field.wgsl")),
    ),
    (
        "fg_mv_pyramid_seed",
        Some(include_str!("../shaders/fsr/fg_mv_pyramid_seed.wgsl")),
    ),
    ("fg_mv_pyramid_reduce", None),
    (
        "fg_disocclusion",
        Some(include_str!("../shaders/fsr/fg_disocclusion.wgsl")),
    ),
    (
        "fg_interpolate",
        Some(include_str!("../shaders/fsr/fg_interpolate.wgsl")),
    ),
    (
        "fg_color_pyramid_seed",
        Some(include_str!("../shaders/fsr/fg_color_pyramid_seed.wgsl")),
    ),
    ("fg_color_pyramid_reduce", None),
    (
        "fg_inpaint",
        Some(include_str!("../shaders/fsr/fg_inpaint.wgsl")),
    ),
    (
        "fg_present",
        Some(include_str!("../shaders/fsr/fg_present.wgsl")),
    ),
];
const VELOCITY: &str = include_str!("../shaders/fsr/velocity.wgsl");
const EFFECT_VELOCITY: &str = include_str!("../shaders/fsr/effect_velocity.wgsl");
const RESIDUAL: &str = include_str!("../shaders/fsr/residual.wgsl");

/// One upscaler pass's full source: the shared prelude, then the pass with its placeholders
/// resolved.
pub fn pass_source(pass: &str, formats: caps::Formats) -> String {
    format!("{COMMON}\n{}", formats.preprocess(pass))
}

/// One frame-generation pass's full source, by its [`FG_PASSES`] name: the frame-generation
/// prelude, then the pass, every placeholder resolved.
pub fn fg_pass_source(name: &str, formats: caps::Formats) -> String {
    let (_, pass) = FG_PASSES
        .iter()
        .find(|(n, _)| *n == name)
        .expect("a frame-generation pass");
    let pass = pass.unwrap_or(FG_REDUCE).replace(
        "FG_PYRAMID_REDUCE_MV",
        if name == "fg_mv_pyramid_reduce" {
            "true"
        } else {
            "false"
        },
    );
    formats.preprocess(&format!("{FG_COMMON}\n{pass}"))
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

/// `WOW_FSR_DEBUG=views`: every two seconds, after the render graph, one line per render view saying which of the parts
/// FSR and the backdrop need it has, so a black frame is read from numbers, not guessed at.
#[allow(clippy::type_complexity)]
fn debug_views(
    views: Query<(
        Entity,
        &bevy::render::sync_world::MainEntity,
        Option<&bevy::render::camera::ExtractedCamera>,
        Option<&bevy::render::view::ViewTarget>,
        Has<crate::ffx_glow::FfxGlowTextures>,
        Option<&FsrState>,
        Has<FgState>,
        Has<crate::ffx_glow::FfxBackdrop>,
        Option<&crate::ffx_glow::FfxBackdropView>,
        Option<&bevy::render::view::ViewDepthTexture>,
    )>,
    pipelines: Option<Res<upscaler::FsrPipelines>>,
    pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
    held: Res<FsrHeld>,
    ready: Query<&FgPresent>,
    mut tally: Local<[[u32; 4]; 2]>,
    mut last: Local<Option<bevy::platform::time::Instant>>,
) {
    // Every frame, by world (0) or held (1): frames, frames whose window was written (it presents),
    // frames with a claiming backdrop, and those whose claim had frame generation's image ready.
    let row = &mut tally[usize::from(held.0)];
    row[0] += 1;
    let written = views.iter().any(|(.., cam, target, _, _, _, _, _, _)| {
        cam.is_some_and(|c| {
            matches!(
                c.target,
                Some(bevy::camera::NormalizedRenderTarget::Window(_))
            )
        }) && target.is_some_and(bevy::render::view::ViewTarget::needs_present)
    });
    row[1] += u32::from(written);
    let source = views
        .iter()
        .find_map(|(.., claim, _)| claim.and_then(|c| c.source_entity()));
    row[2] += u32::from(source.is_some());
    row[3] += u32::from(source.is_some_and(|s| ready.get(s).is_ok_and(FgPresent::ready)));
    if last.is_some_and(|t| t.elapsed().as_secs_f32() < 2.0) {
        return;
    }
    *last = Some(bevy::platform::time::Instant::now());
    let [w, h] = std::mem::take(&mut *tally);
    info!(
        "FSR_PRESENT world frames={} written={} claimed={} fg_ready={} | held frames={} written={} claimed={} fg_ready={}",
        w[0], w[1], w[2], w[3], h[0], h[1], h[2], h[3]
    );
    for (e, main, cam, target, glow, fsr, fg, backdrop, claim, depth) in &views {
        let target = target.is_some();
        // After the render graph: `out` says whether this frame's chain landed.
        info!(
            "FSR_VIEW {e} main={} order={:?} size={:?} target={target} glow={glow} fsr={} out={} fg={fg} backdrop={backdrop} claim={:?} depth={:?}",
            main.id(),
            cam.map(|c| c.order),
            cam.and_then(|c| c.physical_target_size),
            fsr.is_some(),
            fsr.is_some_and(|f| f.output().is_some()),
            claim.map(|c| c.source_entity()),
            depth.map(|d| d.texture.usage()),
        );
    }
    let [ran, compiling, unbindable] = upscaler::UPSCALE_COUNTS
        .each_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed));
    let pending: Vec<String> = pipelines
        .map(|p| {
            p.ids()
                .iter()
                .map(|&id| pipeline_cache.get_compute_pipeline_state(id))
                .filter(|state| {
                    !matches!(
                        state,
                        bevy::render::render_resource::CachedPipelineState::Ok(_)
                    )
                })
                .map(|state| format!("{state:?}"))
                .collect()
        })
        .unwrap_or_else(|| vec!["no FsrPipelines".into()]);
    info!(
        "FSR_UPSCALE ran={ran} compiling={compiling} depth_unbindable={unbindable} pending={pending:?}"
    );
}

pub struct FsrPlugin;

impl Plugin for FsrPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FsrHeld>()
            .init_resource::<FgSolo>()
            .init_resource::<RcasSharpness>()
            .init_resource::<velocity::FsrMotionParts>()
            .add_plugins((
                ExtractComponentPlugin::<FsrView>::default(),
                ExtractComponentPlugin::<PresentsHeldFrames>::default(),
                ExtractResourcePlugin::<FsrHeld>::default(),
                ExtractResourcePlugin::<FgSolo>::default(),
                ExtractResourcePlugin::<RcasSharpness>::default(),
                ExtractResourcePlugin::<velocity::FsrMotionParts>::default(),
            ))
            // `Last`, not `PostUpdate` (`modern/HOOKS.md`, "Scheduling rule"): after the camera's
            // viewport and every transform are final, before extraction.
            .add_systems(Last, (advance_jitter, velocity::collect_motion_parts));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_schedule(Schedule::new(cadence::HeldFrame))
            .init_resource::<velocity::RigPalettePrev>()
            .init_resource::<velocity::VelocityFrame>()
            .init_resource::<velocity::EffectVelocityDraws>()
            .init_resource::<cadence::FgPacing>()
            .init_resource::<SpecializedRenderPipelines<framegen::FgPresentPipeline>>()
            .init_resource::<SpecializedRenderPipelines<velocity::EffectVelocityPipeline>>()
            .add_systems(
                RenderStartup,
                (
                    upscaler::init_pipelines,
                    velocity::init_pipelines,
                    velocity::init_effect_pipeline
                        .after(crate::particles::render::init_effect_pipeline),
                    framegen::init_pipelines,
                    residual::init,
                ),
            )
            .add_systems(
                Render,
                (
                    upscaler::prepare_fsr.in_set(RenderSystems::PrepareResources),
                    (velocity::prepare_velocity, framegen::prepare_framegen)
                        .in_set(RenderSystems::PrepareResources)
                        .after(upscaler::prepare_fsr),
                    velocity::prepare_effect_velocity.in_set(RenderSystems::PrepareBindGroups),
                    // Just before the render graph: the held frame's schedule swap, then its pacing.
                    (cadence::hold_cameras, cadence::pace_held_frames)
                        .chain()
                        .in_set(RenderSystems::Render)
                        .before(render_system),
                ),
            )
            .add_systems(
                Render,
                residual::readback
                    .in_set(RenderSystems::Render)
                    .after(render_system),
            )
            .add_systems(
                Render,
                debug_views
                    .in_set(RenderSystems::Render)
                    .after(render_system)
                    .run_if(|| {
                        crate::launch_env::live_var("WOW_FSR_DEBUG").is_some_and(|v| v == "views")
                    }),
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
                    // After the FFXGlow filter passes, whose blur the compose reads.
                    framegen::fsr_framegen_pass
                        .in_set(Core3dSystems::PostProcess)
                        .after(crate::ffx_glow::ffx_glow_pass),
                ),
            );
    }

    /// The shaders depend on the device's formats, which exist once the renderer has; the cadence
    /// replaces the main app's update schedule, which only `finish` may.
    fn finish(&self, app: &mut App) {
        cadence::install(app);
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
        let framegen = FG_PASSES.map(|(name, _)| {
            shaders.add(Shader::from_wgsl(
                fg_pass_source(name, formats),
                format!("benilla_fsr/{name}.wgsl"),
            ))
        });
        let velocity = shaders.add(Shader::from_wgsl(VELOCITY, "benilla_fsr/velocity.wgsl"));
        let effect_velocity = shaders.add(Shader::from_wgsl(
            EFFECT_VELOCITY,
            "benilla_fsr/effect_velocity.wgsl",
        ));
        let residual = shaders.add(Shader::from_wgsl(
            pass_source(RESIDUAL, formats),
            "benilla_fsr/residual.wgsl",
        ));
        let built = FsrShaders {
            formats,
            upscaler,
            framegen,
            velocity,
            effect_velocity,
            residual,
        };
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
        assert!(
            (moved - plain - jitter).length() < 1e-3,
            "{plain} → {moved}"
        );
    }

    /// The period is the base 8 at 1:1 and grows with the area ratio; the sequence repeats with it
    /// and never sits on the pixel centre.
    #[test]
    fn the_jitter_period_scales_with_the_ratio() {
        assert_eq!(
            jitter_period(UVec2::new(1920, 1080), UVec2::new(1920, 1080)),
            8
        );
        assert_eq!(
            jitter_period(UVec2::new(1280, 720), UVec2::new(1920, 1080)),
            18
        );
        assert_eq!(
            jitter_period(UVec2::new(960, 540), UVec2::new(1920, 1080)),
            32
        );
        let period = 18;
        for phase in 0..period {
            let j = jitter_at(phase, period);
            assert_eq!(j, jitter_at(phase + period, period));
            assert!(j.x.abs() <= 0.5 && j.y.abs() <= 0.5);
            assert_ne!(j, Vec2::ZERO);
        }
    }

    /// Every row of `HOOKS.md` names a file that still carries the marker: an upstream merge that
    /// drops a hook fails here, not in a frame.
    #[test]
    fn every_hook_row_carries_its_marker() {
        const MARKER: &str = "fork: fsr (FORK.md)";
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let hooks = include_str!("HOOKS.md");
        let rows: Vec<&str> = hooks
            .lines()
            .filter_map(|l| l.strip_prefix("| `"))
            .filter_map(|l| l.split('`').next())
            .collect();
        assert!(rows.len() >= 20, "the hook table parses: {rows:?}");
        let missing: Vec<&str> = rows
            .iter()
            .copied()
            .filter(|file| {
                std::fs::read_to_string(crates.join(file))
                    .map_or(true, |text| !text.contains(MARKER))
            })
            .collect();
        assert!(
            missing.is_empty(),
            "HOOKS.md rows whose file lost the `{MARKER}` marker: {missing:?}"
        );
    }

    /// Every FSR shader, in both format sets, parses and validates with naga: no placeholder
    /// survives, and each binds what its Rust layout declares.
    #[test]
    fn every_fsr_shader_validates_in_both_format_sets() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let validate = |name: &str, source: &str| {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        };
        for narrow in [false, true] {
            let formats = caps::Formats::new(narrow);
            for (name, pass) in PASSES {
                validate(name, &pass_source(pass, formats));
            }
            for (name, _) in FG_PASSES {
                validate(name, &fg_pass_source(name, formats));
            }
            validate("residual", &pass_source(RESIDUAL, formats));
        }
    }
}
