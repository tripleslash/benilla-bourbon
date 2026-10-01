// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module
// wxl-experimental-wind, grass/GrassWind.hpp and grass/GrassWind.cpp.
#define_import_path benilla::wind_hook

#import benilla::env_frame::EnvFrame

const WIND_TAU: f32 = 6.28318530718;
// fork: `wind_b.x` is the integrated wind travel in yards, wrapped at 4096 on the CPU
// (`wind::TRAVEL_WRAP`). Each wave rate is a whole number of cycles per wrap, so the wrap is
// seamless; the phase advances at k * speed, never k * speed * age.
const WIND_WRAP: f32 = 4096.0;

// The phase (radians) of a wave doing `cycles` whole cycles per wrap of travel.
fn wind_travel_phase(travel: f32, cycles: f32) -> f32 {
    return fract(cycles * (travel / WIND_WRAP)) * WIND_TAU;
}

// The fixed profile heading: the waves' spatial term uses it, so the veer (which only turns the
// bend direction) never rotates the phase field about the world origin.
fn wind_base_dir(frame: EnvFrame) -> vec2<f32> {
    return vec2<f32>(cos(frame.wind_a.z), sin(frame.wind_a.z));
}

// fork: gusts rolling across a field: a 55 yd band travelling downwind (68 cycles
// per wrap, about 8 yd/s at the base speed) that swells the sway to 1.2x and lulls it to 0.55x.
fn wind_gust_roll(frame: EnvFrame, xz: vec2<f32>) -> f32 {
    let band = sin(dot(wind_base_dir(frame), xz) * (WIND_TAU / 55.0)
        - wind_travel_phase(frame.wind_b.x, 68.0));
    return mix(0.55, 1.2, 0.5 + 0.5 * band);
}

fn wind_cross_dir(dir: vec2<f32>) -> vec2<f32> {
    // WarcraftXL's secondary wave is 35 degrees off the primary.
    return vec2<f32>(
        dir.x * 0.819152 - dir.y * 0.573576,
        dir.x * 0.573576 + dir.y * 0.819152,
    );
}

// fork: WarcraftXL's two travelling waves, gust response, downwind lean, per-tuft
// phase/variance and distance fade, plus the same radial parting extended from the player to the
// eight EnvFrame benders. `bend_height` is authored from local mesh height, never texture V.
fn grass_offset(
    world: vec3<f32>,
    bend_height: f32,
    tuft_phase: f32,
    camera: vec3<f32>,
    frame: EnvFrame,
) -> vec3<f32> {
    let anchored = clamp((bend_height - 0.30) / 0.70, 0.0, 1.0);
    let weight = anchored * anchored;
    if (weight <= 0.0 || frame.wind_b.z <= 0.0) {
        return vec3<f32>(0.0);
    }

    let dir_primary = normalize(frame.wind_a.xy + vec2<f32>(1.0e-6, 0.0));
    let dir_secondary = wind_cross_dir(dir_primary);
    let base_primary = wind_base_dir(frame);
    let base_secondary = wind_cross_dir(base_primary);
    let wave_primary = WIND_TAU / 18.0;
    let wave_secondary = WIND_TAU / 6.5;
    // fork: visible tuning. At the 9 yd/s base speed the primary wave runs a 2.5 s
    // period (182 cycles per wrap) and the cross flutter 1.1 s (410); both speed up with gusts.
    let phase_primary = wave_primary * dot(base_primary, world.xz)
        - wind_travel_phase(frame.wind_b.x, 182.0) + tuft_phase;
    let phase_secondary = wave_secondary * dot(base_secondary, world.xz)
        - wind_travel_phase(frame.wind_b.x, 410.0) + tuft_phase * 1.7;
    let gust = 1.0 + 0.5 * (frame.wind_a.w - 0.5) * 2.0;
    let variance = mix(0.7, 1.3, fract(sin(tuft_phase * 12.9898) * 43758.5453));
    let distance_fade = 1.0 / (1.0 + distance(camera, world) * 0.015);
    var xz = (
        dir_primary * 0.18 * (sin(phase_primary) + 0.35)
        + dir_secondary * 0.06 * sin(phase_secondary)
    ) * gust * variance * wind_gust_roll(frame, world.xz) * weight * distance_fade
        * frame.wind_b.y * frame.wind_b.z;

    let count = min(u32(frame.misc.x), 8u);
    for (var i = 0u; i < 8u; i += 1u) {
        if (i < count) {
            let bender = frame.benders[i];
            let away = world.xz - bender.xz;
            let r = length(away);
            let radial = away / max(r, 1.0e-3);
            let edge = 1.0 - smoothstep(0.0, max(bender.w, 0.01), r);
            let cone = smoothstep(0.30, 0.80, world.y - bender.y);
            xz += radial * 0.50 * edge * cone * weight * frame.wind_b.z;
        }
    }
    // fork: a bent tip dips, so a blade reads as bending rather than stretching.
    return vec3<f32>(xz.x, -min(0.6 * dot(xz, xz), 0.25), xz.y);
}

// fork: our tree/bush design (WarcraftXL has no tree wind). Only leaf-card batches call
// this: a low-frequency crown sway, small leaf flutter, stable anchor phase and distance fade.
fn tree_offset(
    world: vec3<f32>,
    anchor: vec3<f32>,
    camera: vec3<f32>,
    frame: EnvFrame,
) -> vec3<f32> {
    if (frame.wind_b.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = max(world.y - anchor.y, 0.0);
    // A one-yard bush still needs a crown; tree leaf cards generally begin above this range and
    // therefore reach full weight, while the first 0.15 yd remains a planted base.
    let crown = smoothstep(0.15, 1.5, h);
    // fork: a taller crown swings further (lever arm): 0.43x at 1.5 yd, 1x at 6 yd,
    // capped at 1.3x from 8.5 yd, so a bush nods while a canopy sways.
    let weight = crown * crown * clamp(0.25 + h * 0.12, 0.25, 1.3);
    let dir = normalize(frame.wind_a.xy + vec2<f32>(1.0e-6, 0.0));
    let cross = vec2<f32>(-dir.y, dir.x);
    let base = wind_base_dir(frame);
    let base_cross = vec2<f32>(-base.y, base.x);
    let phase = fract(sin(dot(anchor.xz, vec2<f32>(0.173, 0.317))) * 43758.5453) * WIND_TAU;
    // fork: visible tuning. At the 9 yd/s base speed: a 4 s crown sway (114 cycles
    // per wrap), a 5.5 s side wave (82) and a 0.8 s leaf flutter (569).
    let slow = sin(dot(base, world.xz) * (WIND_TAU / 34.0)
        - wind_travel_phase(frame.wind_b.x, 114.0) + phase);
    let side = sin(dot(base_cross, world.xz) * (WIND_TAU / 21.0)
        - wind_travel_phase(frame.wind_b.x, 82.0) + phase * 1.7);
    let flutter = sin(wind_travel_phase(frame.wind_b.x, 569.0)
        + dot(world.xz, vec2<f32>(1.7, 2.1)) + phase * 2.0);
    let gust = 0.75 + frame.wind_a.w * 0.5;
    let fade = 1.0 / (1.0 + distance(camera, world) * 0.006);
    let xz = (
        dir * (0.08 + slow * 0.20)
        + cross * side * 0.08
        + dir * flutter * 0.03
    ) * weight * gust * mix(0.75, 1.05, (wind_gust_roll(frame, anchor.xz) - 0.55) / 0.65)
        * fade * frame.wind_b.y * frame.wind_b.w;
    return vec3<f32>(xz.x, -0.15 * dot(xz, xz), xz.y);
}
