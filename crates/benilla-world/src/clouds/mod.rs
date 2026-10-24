//! The procedural sky clouds: as in the reference, one coverage field feeds both the glare's cloud
//! occlusion and the visible dome, so the cloud over the sun is the cloud that dims it.
//!
//! Capture mode freezes the field's clock and rebuilds only when the density changes, so captures
//! stay byte-identical whatever the frame timing.

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use crate::lighting::WowLighting;
use benilla_assets::AssetSet;

mod kernel;
mod layer;

pub use layer::CloudMaterial;
// fork: modern lighting — the modern bake (`modern::cloud_bake`) writes the layer's image.
pub(crate) use layer::CloudLayer;
mod tables;

pub use kernel::{occ1_moon, occ1_sun};

/// The live cloud coverage field.
#[derive(Resource)]
pub struct CloudCoverage {
    kernel: kernel::CloudKernel,
    /// Whether the init rebuild ran (`0x6cff90`); the bands scroll after it (`0x6cffc0`).
    primed: bool,
    frozen: bool,
    last_density: f32,
    last_frame: Option<kernel::CloudFrame>,
}

impl Default for CloudCoverage {
    fn default() -> Self {
        CloudCoverage {
            kernel: kernel::CloudKernel::default(),
            primed: false,
            frozen: std::env::var_os("WOW_CAPTURE").is_some(),
            last_density: 0.0,
            last_frame: None,
        }
    }
}

impl CloudCoverage {
    /// Coverage in [0, 1] toward camera-relative `d` (Bevy frame, +Y up).
    pub fn coverage(&self, d: Vec3) -> f32 {
        self.kernel.coverage(d)
    }

    /// fork: modern lighting — the coloured tile, the modern bake's low-resolution field.
    pub(crate) fn tile(&self) -> &[[u8; 4]] {
        self.kernel.rgba()
    }
}

pub struct CloudsPlugin;

impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<layer::CloudMaterial>::default())
            .init_resource::<CloudCoverage>()
            .add_systems(Startup, layer::setup_cloud_layer.after(AssetSet::Open))
            .add_systems(
                Update,
                (
                    // After the submersion verdict, so surfacing rebuilds the tile the frame the
                    // dome reappears, and after the lighting resolve, so that rebuild takes the
                    // air's density and palette rather than the water's.
                    tick_clouds
                        .after(crate::liquid::SubmersionVerdict)
                        .in_set(crate::lighting::LightingConsumeSet),
                    // After the skybox resolve and the submersion verdict, so the dome hides in
                    // the same frame as the rest of the sky.
                    layer::apply_cloud_visibility
                        .after(crate::skybox::SkyboxResolve)
                        .after(crate::liquid::SubmersionVerdict),
                ),
            )
            // Camera-anchored after transform propagation, like the sky dome.
            .add_systems(
                PostUpdate,
                layer::follow_cloud_dome.in_set(crate::billboard::BillboardPlace),
            );
    }
}

/// Advances the coverage field (a full rebuild first, then the 10 Hz band scroll) and re-uploads
/// its texels when they change, as the reference does per regen (`0x58ac70`).
fn tick_clouds(
    mut cov: ResMut<CloudCoverage>,
    light: Res<WowLighting>,
    time: Res<Time>,
    clock: Option<Res<layer::CloudLayer>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<layer::CloudMaterial>>,
    underwater: Res<crate::liquid::Underwater>,
    mut was_submerged: Local<bool>,
) {
    // Surfacing forces a full rebuild: the reference's wet→dry edge (`0x680ac3`) calls
    // `0x6d2210(1)`, which selects `0x6cff90`; a band scroll would bring the clouds back over
    // 0.4 s. Going under needs nothing, since the sky pass is skipped while submerged.
    let submerged = underwater.0.any();
    let surfaced = *was_submerged && !submerged;
    *was_submerged = submerged;
    let density = light.cloud_density;
    let frame = kernel::CloudFrame {
        sun: light.cloud_colors[0],
        slope: light.cloud_colors[1],
        gbase: light.cloud_colors[2],
        bcc: light.storm_bcc,
        glow_dir: light.cloud_glow_dir,
        glow_track: light.cloud_glow_track,
    };
    static CLOUD_DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let cloud_dump = *CLOUD_DUMP.get_or_init(|| std::env::var_os("WOW_CLOUD_DUMP").is_some());
    if surfaced && cloud_dump {
        eprintln!("[cloud] surfaced -> full rebuild (C {density:.3})");
    }
    if cloud_dump && cov.last_frame != Some(frame) {
        eprintln!(
            "[cloud] C {density:.3} sun {:?} slope {:?} gbase {:?} bcc {:.2} glow_dir {:?} track {:.2}",
            frame.sun, frame.slope, frame.gbase, frame.bcc, frame.glow_dir, frame.glow_track
        );
    }
    let changed = if !cov.primed || surfaced || (cov.frozen && density != cov.last_density) {
        cov.primed = true;
        cov.last_density = density;
        cov.last_frame = Some(frame);
        cov.kernel.rebuild(density, &frame);
        true
    } else if cov.frozen {
        // Frozen clock: coverage stays put; recolor only when the color inputs move.
        if cov.last_frame != Some(frame) {
            cov.last_frame = Some(frame);
            cov.kernel.recolor(&frame);
            true
        } else {
            false
        }
    } else {
        cov.kernel.tick(time.delta_secs(), density, &frame)
    };
    // fork: modern lighting (FORK.md) — a modern frame's image is the worker-baked
    // layer (`modern::cloud_bake`); the tile is uploaded only in a classic frame, and the first
    // classic frame after a modern one restores its size and format.
    if crate::modern::frame_is_modern() && crate::modern::cpu_hooks_enabled() {
        return;
    }
    let format = bevy::render::render_resource::TextureFormat::Rgba8Unorm;
    let size = bevy::render::render_resource::Extent3d {
        width: kernel::COLS as u32,
        height: kernel::COLS as u32,
        depth_or_array_layers: 1,
    };
    let reformat = clock.as_ref().is_some_and(|layer| {
        images.get(&layer.image).is_some_and(|i| {
            i.texture_descriptor.format != format || i.texture_descriptor.size != size
        })
    });
    if changed || reformat {
        if let Some(layer) = clock.as_ref() {
            if let Some(mut image) = images.get_mut(&layer.image) {
                if image.texture_descriptor.size != size {
                    image.resize(size);
                }
                if image.texture_descriptor.format != format {
                    image.texture_descriptor.format = format;
                }
                image
                    .data
                    .as_mut()
                    .expect("cpu-side cloud image")
                    .copy_from_slice(cov.kernel.rgba().as_flattened());
                // Touch the material: a modified Image gets a new GPU texture, and a stale bind
                // group keeps sampling the first upload. `into_inner`: a bare `AssetMut` emits nothing.
                if let Some(material) = materials.get_mut(&layer.material) {
                    material.into_inner();
                }
            }
        }
    }
}
