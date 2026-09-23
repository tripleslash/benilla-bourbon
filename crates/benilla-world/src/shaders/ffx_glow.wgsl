// FFXGlow, the reference's full-screen glow, from the shipped Shaders\Pixel\FFXGlow.bls and
// FFXGauss4.bls:
//   blur = Gauss4(Gauss4(Box4(scene → ¼), horizontal), vertical)     weights ⅛ ⅜ ⅜ ⅛
//   out  = lerp(screen, blur, z) + w · blur²
// `w` is the zone's LightParams glow weight, `z` the drunk/underwater haze mix (0 sober and dry).
// Tap offsets are exact source-texel counts from texel-centre UVs, with no half-texel shift.
// The math runs on gamma bytes; the float scene RT does not saturate per draw as the reference's
// byte RT does, so every scene read clamps to 1.0 first.

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_samp: sampler;
// Combine-pass bindings, unbound in the downsample and blur passes.
@group(0) @binding(2) var blur_tex: texture_2d<f32>;
struct FfxCombine {
    // x = zone glow weight `w` (both combines), y = FFXDeath gate (1 while a ghost, no ramp),
    // z = haze mix, w = deband dither on.
    lane: vec4<f32>,
    // xy = the GlowWave phases `(t mod 3174)/3174` and `(t mod 2805)/2805`; z = the fork's modern-lighting
    // flag (FORK.md); w = its exposure. Written on
    // every frame, read only by `fs_combine_wave`.
    wave: vec4<f32>,
    // fork: modern lighting — the zone look's cinematic grade: x = contrast, y = saturation,
    // z = vignette, w = toe (gamma on the encoded picture; > 1 deepens the shadows). All 0 = off.
    grade: vec4<f32>,
}
@group(0) @binding(3) var<uniform> ffx: FfxCombine;
// The 128×128 wave LUT and its repeat sampler: bound on every combine, sampled only underwater.
@group(0) @binding(4) var wave_tex: texture_2d<f32>;
@group(0) @binding(5) var wave_samp: sampler;



// Box4, the reference's one downsample, full → ¼: 4 bilinear taps at source-texel offsets
// {−1.5, +0.5}², a 4×4 footprint (table 0xce89cc). Each tap clamps to 1.0 before the average,
// as the byte RT saturated; an unclamped super-white would square into a hard glow disc.
@fragment
fn fs_downsample(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(in_tex));
    let one = vec4<f32>(1.0);
    return (min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, -1.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, -1.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, 0.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, 0.5) * texel), one))
        * 0.25;
}

// fork: modern lighting (FORK.md) — the HDR twin of `fs_downsample`. A modern
// frame is linear radiance, so the taps keep their energy above 1 (capped only against inf); the
// modern combine turns that surplus into bloom.
@fragment
fn fs_downsample_hdr(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(in_tex));
    let cap = vec4<f32>(64.0);
    return (min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, -1.5) * texel), cap)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, -1.5) * texel), cap)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, 0.5) * texel), cap)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, 0.5) * texel), cap))
        * 0.25;
}

// Gauss4: taps at ±0.5 (weight 0.375) and ±2.5 (weight 0.125) source texels — byte-pinned
// (0x6cad10 tap arrays; corrects the earlier INFERRED ±1.5 outer tap). Weights are the shipped
// FFXGauss4.bls constants.
fn gauss4(uv: vec2<f32>, axis: vec2<f32>) -> vec4<f32> {
    let texel = axis / vec2<f32>(textureDimensions(in_tex));
    return textureSample(in_tex, in_samp, uv - 2.5 * texel) * 0.125
        + textureSample(in_tex, in_samp, uv - 0.5 * texel) * 0.375
        + textureSample(in_tex, in_samp, uv + 0.5 * texel) * 0.375
        + textureSample(in_tex, in_samp, uv + 2.5 * texel) * 0.125;
}

@fragment
fn fs_gauss_h(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return gauss4(in.uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_gauss_v(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return gauss4(in.uv, vec2<f32>(0.0, 1.0));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// Deviation: an opt-in deband dither (`WOW_DITHER=1`, `ffx.lane.w`); the reference's 8-bit
// framebuffer is undithered. The hash is Bevy's `screen_space_dither`, a pure function of the
// pixel; Bevy applies it only in its tonemapping pass, which `Tonemapping::None` skips.
fn screen_space_dither(frag_coord: vec2<f32>) -> vec3<f32> {
    var dither = vec3<f32>(dot(vec2<f32>(171.0, 231.0), frag_coord)).xxx;
    dither = fract(dither.rgb / vec3<f32>(103.0, 71.0, 97.0));
    return (dither - vec3<f32>(0.5)) / 255.0;
}

// fork: modern lighting — the P1b tonemap: identity below the knee (so ordinary pixels keep their
// classic value), an exponential shoulder above it that approaches white without clipping.
const MODERN_BLOOM: f32 = 0.35;
const MODERN_KNEE: f32 = 0.8;
fn modern_tonemap(x: vec3<f32>) -> vec3<f32> {
    let k = MODERN_KNEE;
    let over = max(x - vec3<f32>(k), vec3<f32>(0.0));
    let rolled = vec3<f32>(k) + (1.0 - k) * (vec3<f32>(1.0) - exp(-over / (1.0 - k)));
    return select(x, rolled, x > vec3<f32>(k));
}

// fork: modern lighting — the cinematic grade on the encoded, tonemapped picture: saturation about
// the luma, contrast about mid grey, a toe that deepens the shadows, and a soft vignette.
fn cinematic_grade(g: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let luma = dot(g, vec3<f32>(0.2126, 0.7152, 0.0722));
    var c = mix(vec3<f32>(luma), g, ffx.grade.y);
    // Contrast as an S-curve, NOT a scale about mid grey: the latter maps everything below 0.5 toward
    // zero, so a night exterior (whose pixels all sit near 0.08) went black on screen while a daylit
    // one only gained a little snap. The curve fixes 0 and 1 and shapes the midtones between them.
    let s = clamp(c * c * (vec3<f32>(3.0) - 2.0 * c), vec3<f32>(0.0), vec3<f32>(1.0));
    c = mix(c, s, clamp((ffx.grade.x - 1.0) * 2.0, 0.0, 1.0));
    c = pow(c, vec3<f32>(max(ffx.grade.w, 0.01)));
    let edge = length((uv - vec2<f32>(0.5)) * vec2<f32>(1.25, 1.0)) * 1.3;
    return c * (1.0 - ffx.grade.z * smoothstep(0.35, 1.0, edge));
}

// The combine's one exit: dither (when armed) in GAMMA space — the space the present-encode
// rounds in — then the frame's single decode, or, under `GAMMA_OUT`, no decode at all.
fn combine_out(outg: vec3<f32>, alpha: f32, frag_coord: vec2<f32>) -> vec4<f32> {
    let dithered = clamp(
        outg + screen_space_dither(frag_coord) * step(0.5, ffx.lane.w),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
#ifdef GAMMA_OUT
    // As the UI camera's ground pass, store the byte as `ui_quad.wgsl` does (colour premultiplied
    // by alpha, alpha as coverage); the lane's end decodes the frame.
    return vec4<f32>(dithered * alpha, alpha);
#else
    return vec4<f32>(srgb_to_linear(dithered), alpha);
#endif
}

// The combine both entries run, at the fragment's UV or the warped one; `frag_coord` stays
// unwarped because the dither belongs to the destination pixel.
fn combine_body(uv: vec2<f32>, frag_coord: vec2<f32>) -> vec4<f32> {
    let scene = textureSample(in_tex, in_samp, uv);
    // Clamped to [0, 1] like the downsample's taps: the byte framebuffer this term stands for
    // could never exceed 255, and an unclamped super-white here keeps the whole additive stack
    // hard-white after the min() below — the screen term must saturate BEFORE the glow add.
    // fork: modern lighting (FORK.md) — in a modern frame (`wave.z`) the scene
    // and its blur are linear HDR radiance: expose (`wave.w`), add the energy above white as bloom,
    // roll the shoulder into [0, 1], then run the combine's byte math on the gamma encode — the
    // glow, haze and death programs keep their classic behaviour on a tonemapped image.
    let blur_raw = max(textureSample(blur_tex, in_samp, uv).rgb, vec3<f32>(0.0));
    var sg = clamp(scene.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var bg = blur_raw;
    if (ffx.wave.z > 0.5) {
        let exposure = ffx.wave.w;
        let blur_hdr = blur_raw * exposure;
        let bloom = max(blur_hdr - vec3<f32>(1.0), vec3<f32>(0.0)) * MODERN_BLOOM;
        sg = colorspace::linear_to_srgb(modern_tonemap(max(scene.rgb, vec3<f32>(0.0)) * exposure + bloom));
        if (ffx.grade.x > 0.0) {
            sg = cinematic_grade(sg, uv);
        }
        bg = colorspace::linear_to_srgb(modern_tonemap(blur_hdr));
    }
    // The scene alpha rides through untouched: the window surface ignores it and the opaque-clear
    // booths hold 1 everywhere — only the transparent-clear create booth reads it, to composite
    // its baked model straight over the UI page.
    if ffx.lane.y > 0.0 {
        // FFXDeath replaces the glow while a ghost (one active-pass slot `[0xce8bb4]`), unhazed:
        // luma = sat(dot(screen + w·blur², (0.299, 0.587, 0.144))), 0.144 as shipped, and
        // out = luma + tint · sat(4·luma·(1 − luma)), tint 0x5393A8 from the pack at `0x6cb930`.
        let glowed = sg + ffx.lane.x * bg * bg;
        let luma = clamp(dot(glowed, vec3<f32>(0.299, 0.587, 0.144)), 0.0, 1.0);
        let p = clamp(4.0 * luma * (1.0 - luma), 0.0, 1.0);
        let ghost_tint = vec3<f32>(83.0, 147.0, 168.0) / 255.0; // 0x5393A8
        let outg = min(vec3<f32>(luma) + ghost_tint * p, vec3<f32>(1.0));
        return combine_out(outg, scene.a, frag_coord);
    }
    // The haze cross-fades toward the blur before the glow add; z = max(drunk fraction, 84/255
    // while the eye is submerged), 1 at 100 inebriation. The blur term clamps like the screen.
    let hazed = mix(sg, min(bg, vec3<f32>(1.0)), ffx.lane.z);
    let outg = min(hazed + ffx.lane.x * bg * bg, vec3<f32>(1.0));
    return combine_out(outg, scene.a, frag_coord);
}

@fragment
fn fs_combine(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return combine_body(in.uv, in.position.xy);
}

// FFXGlowWave, the underwater screen warp: with the camera eye in any liquid the reference runs it
// in place of FFXGlow (`CFFXGlow::Render` `0x6cc630`, GlowWave render `0x6cb310`). It is the same
// combine with the screen and blur texcoords displaced by a 128×128 two-channel LUT holding
// `du = sin(2πx/128)`, `dv = sin(2πy/128)`. The reference builds the wave texcoord per vertex
// (`0x7bca80`); it is affine in the screen UV, so evaluating it per fragment is exact:
//     uv' = uv · R(10°) · S(W/128, 0.88·H/128) · T(p1, p2)
// The phases run off one millisecond clock with periods 3.174 s and 2.805 s. The scale is in
// destination pixels, so the pattern is screen-fixed: 128 px across, 145.45 px down at any
// resolution. Amplitude is 3.0 full-res pixels on both samples (0.75 texels of the ¼ RT).
const WAVE_ROT_RAD: f32 = 0.174532925; // 10°, the reference's rotation row
const WAVE_LUT_EDGE: f32 = 128.0; // the LUT is 128×128 texels ([0xce89a0])
const WAVE_V_SCALE: f32 = 0.88; // [0xce89c0]: the v axis only; u's scale is 1.0
const WAVE_AMPLITUDE_PX: f32 = 3.0; // full-res pixels, both samples

// The warp's screen-UV offset in UV units; `screen` is the full-res scene size in pixels.
fn wave_offset(uv: vec2<f32>, screen: vec2<f32>) -> vec2<f32> {
    let c = cos(WAVE_ROT_RAD);
    let s = sin(WAVE_ROT_RAD);
    let scale = vec2<f32>(screen.x, WAVE_V_SCALE * screen.y) / WAVE_LUT_EDGE;
    let tex = vec2<f32>(c * uv.x - s * uv.y, s * uv.x + c * uv.y) * scale + ffx.wave.xy;
    // Sampled, not evaluated: the reference's sine is the 128-texel LUT under linear filtering.
    // The sampler must repeat: the texcoord spans about 10 cycles, and a clamp pins all but one.
    // The unsigned 8-bit LUT is unbiased by (x − 0.5)·2, as the shipped ps_2_0 permutation does.
    let d = (textureSample(wave_tex, wave_samp, tex).rg - vec2<f32>(0.5)) * 2.0;
    return d * WAVE_AMPLITUDE_PX / screen;
}

@fragment
fn fs_combine_wave(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let screen = vec2<f32>(textureDimensions(in_tex));
    return combine_body(in.uv + wave_offset(in.uv, screen), in.position.xy);
}
