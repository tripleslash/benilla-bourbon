// FSR 3.1.5 pass 7 — RCAS (Robust Contrast Adaptive Sharpening), and the chain's only write to its
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// display-resolution result, `FsrOutput` (`fsr/upscaler.rs`).
//
// A port of AMD's `FsrRcasF` (`fsr1/ffx_fsr1.h`, which the FSR 3 upscaler's RCAS pass includes
// verbatim): a 5-tap cross at output resolution with a per-pixel limiter derived from the ring's
// min/max, so it never pushes the centre past the local peak range and cannot clip or halo. It follows
// the temporal accumulation, whose softness comes from the history resample and the reconstruction
// kernel.

@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

/// FSR's luma proxy: green weighted 2x against red and blue (`FsrRcasF` uses
/// `g + 0.5 * (b + r)`, i.e. luma times two — only ratios of it matter).
fn tap_luma(c: vec3<f32>) -> f32 {
    return c.g + 0.5 * (c.b + c.r);
}

/// Source texel at integer coordinate `p`, clamped to the extent so the cross
/// stays in bounds at the border.
fn load_src(p: vec2<i32>) -> vec3<f32> {
    let hi = vec2<i32>(textureDimensions(src_tex, 0)) - vec2<i32>(1, 1);
    // `FsrRcasLoadF`: the input scaled by the exposure, divided back out of the output.
    return textureLoad(src_tex, clamp(p, vec2<i32>(0, 0), hi), 0).rgb * params.exposure;
}

/// Ceiling on the sharpening lobe (`FSR_RCAS_LIMIT`, one bit of headroom under
/// the 0.25 that would push a tap to the exact peak).
const RCAS_LIMIT: f32 = 0.25 - 1.0 / 16.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sp = vec2<i32>(gid.xy);
    if (any(sp >= display_size_i())) {
        return;
    }
    // Minimal 3x3 cross:
    //
    //     b
    //   d e f
    //     h
    let c_b = load_src(sp + vec2<i32>(0, -1));
    let c_d = load_src(sp + vec2<i32>(-1, 0));
    let c_e = load_src(sp);
    let c_f = load_src(sp + vec2<i32>(1, 0));
    let c_h = load_src(sp + vec2<i32>(0, 1));

    let l_b = tap_luma(c_b);
    let l_d = tap_luma(c_d);
    let l_e = tap_luma(c_e);
    let l_f = tap_luma(c_f);
    let l_h = tap_luma(c_h);

    // Noise detection: how far the centre sits off the mean of its ring,
    // relative to the ring's spread. Flat-but-noisy pixels get less sharpening.
    let ring_mean = 0.25 * (l_b + l_d + l_f + l_h);
    let ring_max = max(max(max(l_b, l_d), max(l_e, l_f)), l_h);
    let ring_min = min(min(min(l_b, l_d), min(l_e, l_f)), l_h);
    var nz = clamp(abs(ring_mean - l_e) * rcp_safe(ring_max - ring_min), 0.0, 1.0);
    nz = 1.0 - 0.5 * nz;

    // Per-channel limiter: the largest lobe that keeps the sharpened centre
    // inside the ring's peak range. Both divisors are guarded away from zero
    // keeping their sign: the first is non-negative, the second non-positive.
    let mn4 = min(min(c_b, c_d), min(c_f, c_h));
    let mx4 = max(max(c_b, c_d), max(c_f, c_h));
    // Holds back the *darkening* half of the lobe on a centre already darker
    // than everything around it — a thin dark feature that a symmetric limiter
    // would drive toward black. Added to RCAS after the original FSR 1.0
    // release; it is in the copy the FSR 3 upscaler builds against.
    let lower_limiter = clamp(l_e * rcp_safe(min(min(l_b, l_d), min(l_f, l_h))), 0.0, 1.0);
    let hit_min = mn4 / max(4.0 * mx4, vec3<f32>(1.0 / 65536.0)) * lower_limiter;
    let hit_max = (vec3<f32>(1.0) - mx4)
        / min(4.0 * mn4 - vec3<f32>(4.0), vec3<f32>(-1.0 / 65536.0));
    let lobe_rgb = max(-hit_min, hit_max);
    var lobe = max(-RCAS_LIMIT, min(max(max(lobe_rgb.r, lobe_rgb.g), lobe_rgb.b), 0.0))
        * params.sharpness;
    lobe = lobe * nz;

    let rcp_l = rcp_safe(4.0 * lobe + 1.0);
    let pix = (lobe * c_b + lobe * c_d + lobe * c_h + lobe * c_f + c_e) * rcp_l;

    // `FsrOutput` carries the main texture's own colour semantics: gamma values on a classic
    // frame, linear HDR on a modern one.
    textureStore(output, sp, vec4<f32>(pix / params.exposure, 1.0));
}
