//! fork: fsr (FORK.md) — FSR in the engine binary. `WOW_RENDER_SCALE=<s>` renders the world
//! at `window × s` and `WOW_FRAME_GENERATION=1` turns frame generation on. Either one moves the
//! world camera onto a size-carrier image and adds a backdrop camera that grounds on it through the
//! FFXGlow combine, the shape of the client's player-UI camera (`benilla-app`'s
//! `world_backdrop.rs`). So the upscaler and frame generation run here with no login. `WOW_FSR=0`
//! keeps the scale and drops FSR, the bilinear stretch the client had before it, for comparison.
//! With neither knob set, the viewer keeps its swapchain path unchanged.

use bevy::asset::RenderAssetUsages;
use bevy::camera::{CameraOutputMode, ImageRenderTarget, RenderTarget};
use bevy::prelude::*;
use bevy::render::camera::MipBias;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::PrimaryWindow;

use crate::ffx_glow::FfxBackdrop;
use crate::fsr::{FrameGeneration, FsrView, PresentsHeldFrames};
use crate::view::WorldCamera;

/// The mode's two knobs.
#[derive(Resource, Clone, Copy, Debug)]
struct FsrViewConfig {
    scale: f32,
    frame_generation: bool,
    /// `WOW_FSR=0`: the bilinear stretch alone.
    no_fsr: bool,
}

impl FsrViewConfig {
    /// `None` when neither knob is set: the viewer's own path.
    fn from_env() -> Option<Self> {
        let scale = crate::launch_env::live_var("WOW_RENDER_SCALE")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite());
        let frame_generation =
            crate::launch_env::live_var("WOW_FRAME_GENERATION").is_some_and(|v| v.trim() != "0");
        if scale.is_none() && !frame_generation {
            return None;
        }
        let scale = scale.unwrap_or(1.0).clamp(0.25, 1.0);
        let no_fsr = crate::launch_env::live_var("WOW_FSR").is_some_and(|v| v.trim() == "0");
        Some(Self {
            // Frame generation caps the scale at 1:1, as the client does.
            scale,
            frame_generation: frame_generation && !no_fsr,
            no_fsr,
        })
    }

    /// FSR runs below 1:1, and at 1:1 under frame generation.
    fn fsr(self) -> bool {
        !self.no_fsr && (self.scale < 1.0 || self.frame_generation)
    }
}

/// The size-carrier the world camera targets and the size it was built at.
#[derive(Resource)]
struct Carrier {
    image: Handle<Image>,
    size: UVec2,
}

fn carrier_image(size: UVec2) -> Image {
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

fn render_size(px: UVec2, scale: f32) -> UVec2 {
    (px.as_vec2() * scale).round().as_uvec2().max(UVec2::ONE)
}

/// Spawn the backdrop camera over the world camera, and turn frame generation on if asked.
fn spawn_backdrop(
    mut commands: Commands,
    config: Res<FsrViewConfig>,
    mut fg: ResMut<FrameGeneration>,
    world: Query<Entity, With<WorldCamera>>,
) {
    fg.0 = config.frame_generation;
    let source = world.iter().next();
    commands.spawn((
        Name::new("worldview FSR backdrop"),
        Camera2d,
        Msaa::Off,
        // The combine decodes itself: no UI gamma lane follows it here.
        FfxBackdrop {
            source,
            decode: true,
        },
        PresentsHeldFrames,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
    ));
    info!(
        "worldview: FSR mode, render scale {:.3}, FSR {}, frame generation {}",
        config.scale,
        if config.fsr() { "on" } else { "off" },
        if config.frame_generation { "on" } else { "off" }
    );
}

/// Keep the carrier at `window × scale` and the world camera on it, with the window's logical
/// viewport, FSR's view and its mip bias.
#[allow(clippy::type_complexity)]
fn track(
    mut commands: Commands,
    config: Res<FsrViewConfig>,
    mut images: ResMut<Assets<Image>>,
    carrier: Option<ResMut<Carrier>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<
        (Entity, &mut Camera, &mut RenderTarget, Option<&FsrView>),
        With<WorldCamera>,
    >,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let px = UVec2::new(
        window.physical_width().max(1),
        window.physical_height().max(1),
    );
    let size = render_size(px, config.scale);
    let rebuilt = match carrier {
        Some(c) if c.size == size => None,
        Some(mut c) => {
            let old = std::mem::replace(&mut c.image, images.add(carrier_image(size)));
            images.remove(&old);
            c.size = size;
            Some(c.image.clone())
        }
        None => {
            let image = images.add(carrier_image(size));
            commands.insert_resource(Carrier {
                image: image.clone(),
                size,
            });
            Some(image)
        }
    };
    let factor = window.resolution.scale_factor() * size.x as f32 / px.x as f32;
    for (entity, mut camera, mut target, fsr) in &mut cameras {
        // No upscaling blit into the size-carrier: the backdrop's ground reads the main texture,
        // as the client's world camera does (`benilla-app`'s `player/setup.rs`).
        if !matches!(camera.output_mode, CameraOutputMode::Skip) {
            camera.output_mode = CameraOutputMode::Skip;
        }
        if let Some(image) = &rebuilt {
            *target = RenderTarget::Image(ImageRenderTarget {
                handle: image.clone(),
                scale_factor: factor,
            });
        }
        let fresh = fsr.is_none_or(|f| f.display != px);
        if !config.fsr() {
            if rebuilt.is_some() {
                // The bilinear stretch's own bias (`world_backdrop.rs::mip_bias`).
                commands
                    .entity(entity)
                    .insert(MipBias(config.scale.log2().min(0.0)));
            }
        } else if fresh {
            let epoch = fsr.map_or(0, |f| f.epoch.wrapping_add(1));
            commands.entity(entity).insert((
                FsrView::new(px, epoch, config.frame_generation),
                // FSR's texture LOD bias: one level sharper than the render scale owes.
                MipBias(config.scale.log2() - 1.0),
            ));
        }
    }
}

/// Every two seconds of real time, `FSR_FRAMES rendered=<n> presented=<m> solo=<s>`: world frames
/// (one per main-world update, which a held frame skips) against frames presented, held ones
/// included, and the world frames no held frame followed.
fn log_frames(time: Res<Time<bevy::time::Real>>, mut acc: Local<(f32, u32)>) {
    acc.0 += time.delta_secs();
    acc.1 += 1;
    if acc.0 >= 2.0 {
        let presented =
            crate::fsr::cadence::PRESENTED.swap(0, std::sync::atomic::Ordering::Relaxed);
        let solo = crate::fsr::cadence::SOLO.swap(0, std::sync::atomic::Ordering::Relaxed);
        let (mean, cv, max) = crate::fsr::cadence::take_cadence().unwrap_or_default();
        info!(
            "FSR_FRAMES rendered={:.1}/s presented={:.1}/s solo={:.1}/s present_ms={mean:.1} cv={cv:.2} max_ms={max:.1}",
            acc.1 as f32 / acc.0,
            presented as f32 / acc.0,
            solo as f32 / acc.0
        );
        *acc = (0.0, 0);
    }
}

/// The mode, when its knobs are set.
pub(super) fn plugin(app: &mut App) {
    let Some(config) = FsrViewConfig::from_env() else {
        return;
    };
    app.insert_resource(config)
        .add_systems(Startup, spawn_backdrop.after(super::spawn_view_camera))
        .add_systems(Update, track)
        .add_systems(Last, log_frames);
}
