//! fork: native HDR output — the window presents in scRGB (`Rgba16Float`, linear, `1.0` = 80 nits)
//! instead of 8-bit sRGB, so the modern model's highlights reach the display's peak. 1.12 has no
//! HDR: it presents an 8-bit backbuffer. Off (the default) changes nothing: no request reaches the
//! surface (`bevy_render`'s twentieth delta), no camera turns `Hdr`, and every HDR pipeline variant
//! stays unbuilt. The web build never asks: a WebGPU canvas has no linear extended space and no
//! luminance report to place white or peak with.
//!
//! On, three numbers place the picture: the OS's SDR white level for the interface (read from the
//! display, [`WindowHdrStatus::sdr_white_nits`]), [`HdrOutputSetting::mid_grey_nits`] for the world's
//! 18% grey, and [`HdrOutputSetting::peak_nits`] for the brightest highlight the modern tonemap rolls
//! toward. The FFXGlow combine places the world ([`HdrDisplay::combine_params`]) and the UI gamma
//! decode scales the whole frame onto the swapchain ([`HdrDisplay::decode_params`]).

use bevy::camera::Hdr;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::view::window::WindowHdrOutput;

#[cfg(doc)]
use bevy::render::view::window::WindowHdrStatus;

/// The `hdrOutput`, `hdrPeakNits` and `hdrMidGreyNits` CVars, as their callbacks write them.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct HdrOutputSetting {
    pub enabled: bool,
    /// The brightest a highlight is drawn, in nits.
    pub peak_nits: f32,
    /// Where the world's 18% grey lands, in nits.
    pub mid_grey_nits: f32,
}

/// `hdrPeakNits`'s registered default.
pub const DEFAULT_PEAK_NITS: f32 = 1000.0;
/// `hdrMidGreyNits`'s registered default: 18% of BT.2408's 203-nit HDR reference white.
pub const DEFAULT_MID_GREY_NITS: f32 = 37.0;
/// The bounds the options rows offer and the consumers clamp to.
pub const PEAK_NITS_RANGE: std::ops::RangeInclusive<f32> = 400.0..=4000.0;
pub const MID_GREY_NITS_RANGE: std::ops::RangeInclusive<f32> = 10.0..=100.0;

impl Default for HdrOutputSetting {
    fn default() -> Self {
        Self {
            enabled: false,
            peak_nits: DEFAULT_PEAK_NITS,
            mid_grey_nits: DEFAULT_MID_GREY_NITS,
        }
    }
}

/// scRGB's unit: `1.0` is 80 nits.
const SCRGB_NITS: f32 = 80.0;
/// Scene-linear middle grey.
const MID_GREY: f32 = 0.18;
/// The interface's white where the display reports no SDR white level: BT.2408's reference white.
const FALLBACK_PAPER_NITS: f32 = 203.0;

/// What the frame is placed with this frame; `active` only while the window presents in scRGB.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct HdrDisplay {
    pub active: bool,
    /// The interface's white, in nits: the OS's SDR white level.
    pub paper_nits: f32,
    pub peak_nits: f32,
    pub mid_grey_nits: f32,
}

impl HdrDisplay {
    /// The combine's HDR row: `x` the world's gain over paper white (the nits of scene-linear 1.0
    /// over the paper's), `y` the peak over paper white, `z` the tonemap's white in scene units
    /// (the peak over scene 1.0's nits, at least 1), `w` paper white in scRGB units, for an exit
    /// that writes the window's values itself.
    pub fn combine_params(&self) -> [f32; 4] {
        let scene_white = self.mid_grey_nits / MID_GREY;
        [
            scene_white / self.paper_nits,
            (self.peak_nits / self.paper_nits).max(1.0),
            (self.peak_nits / scene_white).max(1.0),
            self.paper_nits / SCRGB_NITS,
        ]
    }

    /// The decode's HDR pair: paper white and the peak, each in scRGB units.
    pub fn decode_params(&self) -> [f32; 2] {
        [
            self.paper_nits / SCRGB_NITS,
            self.peak_nits.max(self.paper_nits) / SCRGB_NITS,
        ]
    }
}

/// A camera that turns `Hdr` while the window presents in scRGB, so its target keeps values above
/// white: the player-UI camera, whose target the world's combine and the interface share.
#[derive(Component, Clone, Copy, Default)]
pub struct HdrOutputTarget;

/// Asks the window for scRGB while the setting is on (native only), and turns what the surface
/// answered into [`HdrDisplay`] and the targets' `Hdr`.
fn sync_hdr_output(
    mut commands: Commands,
    setting: Res<HdrOutputSetting>,
    mut window: ResMut<WindowHdrOutput>,
    mut display: ResMut<HdrDisplay>,
    targets: Query<(Entity, Has<Hdr>), With<HdrOutputTarget>>,
) {
    let requested = setting.enabled && cfg!(not(target_arch = "wasm32"));
    if window.requested != requested {
        window.requested = requested;
    }
    let active = requested && window.status.active();
    let next = if active {
        let paper = window
            .status
            .sdr_white_nits()
            .unwrap_or(FALLBACK_PAPER_NITS);
        HdrDisplay {
            active,
            paper_nits: paper,
            peak_nits: setting
                .peak_nits
                .clamp(*PEAK_NITS_RANGE.start(), *PEAK_NITS_RANGE.end()),
            mid_grey_nits: setting
                .mid_grey_nits
                .clamp(*MID_GREY_NITS_RANGE.start(), *MID_GREY_NITS_RANGE.end()),
        }
    } else {
        HdrDisplay::default()
    };
    if *display != next {
        if next.active != display.active || next.paper_nits != display.paper_nits {
            info!(
                "hdr: window {}, paper white {} nits (display peak {:?} nits), peak {} nits, \
                 mid grey {} nits",
                if next.active {
                    "presents in scRGB"
                } else {
                    "presents in sRGB"
                },
                next.paper_nits,
                window.status.max_nits(),
                next.peak_nits,
                next.mid_grey_nits,
            );
        }
        *display = next;
    }
    for (entity, hdr) in &targets {
        match (active, hdr) {
            (true, false) => {
                commands.entity(entity).insert(Hdr);
            }
            (false, true) => {
                commands.entity(entity).remove::<Hdr>();
            }
            _ => {}
        }
    }
}

/// Runs while HDR is asked for, and once more after it is let go.
fn hdr_in_play(
    setting: Res<HdrOutputSetting>,
    display: Res<HdrDisplay>,
    window: Res<WindowHdrOutput>,
) -> bool {
    setting.enabled || display.active || window.requested
}

pub struct HdrOutputPlugin;

impl Plugin for HdrOutputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HdrOutputSetting>()
            .init_resource::<HdrDisplay>()
            .init_resource::<WindowHdrOutput>()
            .add_plugins(ExtractResourcePlugin::<HdrDisplay>::default())
            .add_systems(PostUpdate, sync_hdr_output.run_if(hdr_in_play));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display(paper: f32, peak: f32, mid: f32) -> HdrDisplay {
        HdrDisplay {
            active: true,
            paper_nits: paper,
            peak_nits: peak,
            mid_grey_nits: mid,
        }
    }

    /// At the defaults on a 203-nit paper white, scene 1.0 lands on the paper: the world sits
    /// against the interface as it does in SDR.
    #[test]
    fn the_defaults_put_scene_white_on_paper_white() {
        let [gain, peak, white, paper] = display(203.0, 1000.0, 37.0).combine_params();
        assert!((paper - 203.0 / 80.0).abs() < 1e-5);
        assert!((gain - 1.0).abs() < 0.02, "{gain}");
        assert!((peak - 1000.0 / 203.0).abs() < 1e-4);
        assert!((white - 1000.0 / (37.0 / 0.18)).abs() < 1e-3);
    }

    /// A peak below the paper never asks the tonemap to roll under white.
    #[test]
    fn a_dim_peak_holds_white() {
        let [_, peak, white, _] = display(400.0, 300.0, 80.0).combine_params();
        assert_eq!(peak, 1.0);
        assert_eq!(white, 1.0);
        assert_eq!(display(400.0, 300.0, 80.0).decode_params(), [5.0, 5.0]);
    }

    #[test]
    fn scrgb_is_eighty_nits_a_unit() {
        assert_eq!(display(240.0, 1000.0, 37.0).decode_params(), [3.0, 12.5]);
    }
}
