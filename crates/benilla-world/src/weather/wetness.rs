//! fork (Everwood port): surface wetness from the zone weather, published in the `EnvFrame`
//! `wet_a` row
//! (`rain_rate, wetness, ripple_time_s, snow`).
//!
//! - `rain_rate` is channel A's effect density while the effect is rain (already ramped ~10 s by
//!   the weather channel), else 0.
//! - `wetness` rises while it rains (full in about 90 s at full rain, slower in a drizzle) and
//!   dries after it stops (about 4 min from soaked). It runs on real time, like the channels.
//! - `ripple_time_s` is the rain-ripple clock (wraps every [`RIPPLE_WRAP_S`] s; the shader's ring
//!   rates are chosen so the wrap is seamless).
//! - `snow` is reserved (0).
//!
//! The `rainSurfaces` cvar ([`RainSurfaces`]) gates the whole row: off writes zeros, and every
//! shader reader treats a zero row as "dry", so the image is unchanged.
//!
//! Capture knobs: `WOW_RAIN_SURFACES=0|1` overrides the cvar; `WOW_WETNESS=<0..1>` pins the
//! wetness (a capture cannot wait minutes for the ramp); `WOW_WET_T=<s>` pins the ripple clock;
//! `WOW_WET_SSR=0` turns the screen-space reflection off (the sky colour stands in for it).

use bevy::prelude::*;

use super::{WeatherKind, WeatherState, WeatherTick};
use crate::lighting::EnvFrame;

/// The `rainSurfaces` cvar: 0 Off (no wet surfaces, no rain ripples), 1 On. benilla-app's
/// `video::on_cvar` writes it.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RainSurfaces(pub bool);

impl Default for RainSurfaces {
    fn default() -> Self {
        Self(true)
    }
}

/// Seconds from dry to soaked under full rain.
const WET_UP_S: f32 = 90.0;
/// Seconds from soaked to dry once the rain stops.
const DRY_S: f32 = 240.0;
/// The ripple clock's wrap. The shader's ring rates are multiples of 1/`RIPPLE_WRAP_S`, so every
/// ring's phase is continuous across the wrap.
pub const RIPPLE_WRAP_S: f32 = 1000.0;

/// The wetness integrator's state.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct Wetness {
    pub wetness: f32,
    pub ripple_time_s: f32,
}

impl Wetness {
    /// One real-time step. `rain` is the rain rate 0..1 (0 when not raining).
    fn step(&mut self, rain: f32, dt: f32) {
        if rain > 0.0 {
            // A drizzle wets slower and never quite soaks.
            let target = 0.55 + 0.45 * rain.min(1.0);
            let rate = (0.35 + 0.65 * rain.min(1.0)) / WET_UP_S;
            if self.wetness < target {
                self.wetness = (self.wetness + rate * dt).min(target);
            } else {
                self.wetness = (self.wetness - dt / DRY_S).max(target);
            }
        } else {
            self.wetness = (self.wetness - dt / DRY_S).max(0.0);
        }
        self.ripple_time_s = (self.ripple_time_s + dt) % RIPPLE_WRAP_S;
    }
}

fn env_f32(key: &str) -> Option<f32> {
    crate::launch_env::var(key)?.trim().parse().ok()
}

/// The session-only overrides, read once.
struct EnvOverrides {
    on: Option<bool>,
    wetness: Option<f32>,
    ripple_t: Option<f32>,
    ssr: Option<bool>,
}

fn env_overrides() -> &'static EnvOverrides {
    static ENV: std::sync::OnceLock<EnvOverrides> = std::sync::OnceLock::new();
    ENV.get_or_init(|| EnvOverrides {
        on: env_f32("WOW_RAIN_SURFACES").map(|v| v >= 0.5),
        wetness: env_f32("WOW_WETNESS").map(|v| v.clamp(0.0, 1.0)),
        ripple_t: env_f32("WOW_WET_T"),
        ssr: env_f32("WOW_WET_SSR").map(|v| v >= 0.5),
    })
}

pub(super) fn wetness_tick(
    time: Res<Time<Real>>,
    state: Res<WeatherState>,
    cvar: Res<RainSurfaces>,
    mut wet: ResMut<Wetness>,
    mut frame: ResMut<EnvFrame>,
) {
    let env = env_overrides();
    let rain = if state.effect_kind == WeatherKind::Rain {
        state.effect_density.clamp(0.0, 1.0)
    } else {
        0.0
    };
    wet.step(rain, time.delta_secs());
    let on = env.on.unwrap_or(cvar.0);
    let (rate, wetness, ripple) = if on {
        (
            rain,
            env.wetness.unwrap_or(wet.wetness),
            // fork: the rings read the clock only while it rains; publishing it when
            // dry rewrote the shared light buffer every frame in clear weather.
            env.ripple_t
                .unwrap_or(if rain > 0.0 { wet.ripple_time_s } else { 0.0 }),
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    // The wet reflection pass runs while anything is wet (`super::wet_ssr`).
    let ssr = if wetness > 0.0 && env.ssr.unwrap_or(true) {
        1.0
    } else {
        0.0
    };
    // Write only on change, so change detection on the frame stays meaningful.
    if frame.rain_rate != rate
        || frame.wetness != wetness
        || frame.ripple_time_s != ripple
        || frame.wet_ssr != ssr
    {
        frame.rain_rate = rate;
        frame.wetness = wetness;
        frame.ripple_time_s = ripple;
        frame.snow = 0.0;
        frame.wet_ssr = ssr;
    }
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<RainSurfaces>()
        .init_resource::<Wetness>()
        // After the wind rows and before the resolve, so the `EnvFrame` writers run in one
        // declared order.
        .add_systems(
            Update,
            wetness_tick
                .after(WeatherTick)
                .after(crate::wind::WindTick)
                .before(crate::lighting::LightingResolveSet),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_rain_soaks_in_about_ninety_seconds_and_dries_in_four_minutes() {
        let mut w = Wetness::default();
        for _ in 0..(89 * 10) {
            w.step(1.0, 0.1);
        }
        assert!(w.wetness > 0.95 && w.wetness < 1.0, "{}", w.wetness);
        for _ in 0..(20 * 10) {
            w.step(1.0, 0.1);
        }
        assert_eq!(w.wetness, 1.0);
        for _ in 0..(120 * 10) {
            w.step(0.0, 0.1);
        }
        assert!(
            (w.wetness - 0.5).abs() < 0.01,
            "half dry at 2 min: {}",
            w.wetness
        );
        for _ in 0..(121 * 10) {
            w.step(0.0, 0.1);
        }
        assert_eq!(w.wetness, 0.0);
    }

    #[test]
    fn a_drizzle_never_quite_soaks() {
        let mut w = Wetness::default();
        for _ in 0..(600 * 10) {
            w.step(0.2, 0.1);
        }
        assert!((w.wetness - 0.64).abs() < 1e-4, "{}", w.wetness);
    }

    #[test]
    fn the_ripple_clock_wraps() {
        let mut w = Wetness {
            ripple_time_s: RIPPLE_WRAP_S - 0.05,
            ..Default::default()
        };
        w.step(0.0, 0.1);
        assert!(w.ripple_time_s < 0.1);
    }
}
