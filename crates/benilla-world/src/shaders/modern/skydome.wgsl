// The sky dome's dawn/dusk azimuthal warp (`0x6d0f50`), shared by the dome itself (`sky.wgsl`) and
// the water's reflection of it (`liquid_cata.wgsl`'s `sky_dome`), so the sky the water mirrors is the
// sky the player sees: one table, one warp, nothing to drift apart.
#define_import_path benilla_modern::skydome

// Glow `g` for a sun-relative azimuth phase, sampled linearly like `0x6d0f50` from the reference's
// six-keyframe wrap-around table at `0xce9af8` (written by `0x6ce210`); 0.125 is the sun bearing.
fn azimuth_glow(phase: f32) -> f32 {
    let p = fract(phase);
    if (p < 0.125) { return mix(0.0, 1.0, (p + 0.125) / 0.25); } // wrap 0.875(g0)→1.125(g1)
    else if (p < 0.375) { return mix(1.0, 0.0, (p - 0.125) / 0.25); }
    else if (p < 0.5) { return mix(0.0, -0.5, (p - 0.375) / 0.125); }
    else if (p < 0.625) { return mix(-0.5, -0.7, (p - 0.5) / 0.125); }
    else if (p < 0.75) { return mix(-0.7, -0.5, (p - 0.625) / 0.125); }
    else if (p < 0.875) { return mix(-0.5, 0.0, (p - 0.75) / 0.125); }
    else { return mix(0.0, 1.0, (p - 0.875) / 0.25); } // wrap 0.875(g0)→1.125(g1)
}

// One mid-ring's warped colour for glow `g` (`0x6d0f50`): `S^2` is the prepass S nested in the
// per-segment S, and 0.7 is the constant at `0x7ffd7c`. `sky0` is the zenith row, `sky1` the next.
fn warp_one(base: vec3<f32>, g: f32, s: f32, sky0: vec3<f32>, sky1: vec3<f32>) -> vec3<f32> {
    let s2 = s * s;
    if (g >= 0.0) {
        return mix(base, sky1, (1.0 - g) * s2);
    }
    return mix(mix(base, sky1, s), sky0, 0.7 * (-g) * s2);
}

// The glow pair a direction's azimuth falls between and its blend: the reference bakes the warp per
// vertex at 24 azimuth segments and interpolates, hence two segments' glows and a lerp. `sun_az` is
// the sun's `atan2(z, x)`; `+ 0.125` puts its bearing at glow phase 0.125.
fn warp_glows(dir: vec3<f32>, sun_az: f32) -> vec3<f32> {
    let az = fract((atan2(dir.z, dir.x) - sun_az) / 6.2831853 + 0.125);
    let seg = az * 24.0;
    let s0 = floor(seg);
    return vec3<f32>(azimuth_glow(s0 / 24.0), azimuth_glow((s0 + 1.0) / 24.0), seg - s0);
}

// A mid ring warped along a direction whose glows are `gl` ([`warp_glows`]).
fn warp_ring(base: vec3<f32>, gl: vec3<f32>, s: f32, sky0: vec3<f32>, sky1: vec3<f32>) -> vec3<f32> {
    return mix(warp_one(base, gl.x, s, sky0, sky1), warp_one(base, gl.y, s, sky0, sky1), gl.z);
}
