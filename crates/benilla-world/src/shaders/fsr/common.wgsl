// Shared declarations for the FSR 3.1.5 upscaler passes (`fsr/upscaler.rs`), after
// `ffx_fsr3upscaler_common.h` and `ffx_fsr3upscaler_callbacks_hlsl.h` (FidelityFX SDK 3.1.5,
// MIT; see `fsr/LICENSE-FIDELITYFX.txt`). Ported through physics-rs's WGSL.
//
// Prepended to every upscaler pass, so each pass file holds only its own bindings and its step of
// the algorithm. The parameter block is one uniform bound by all of them at `@group(0) @binding(0)`.
// Names follow the SDK's in snake_case.
//
// ## Depth
//
// Bevy's depth is reverse-Z with an infinite far plane: the SDK's `FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH`
// with `FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE`. Nearer is larger, a cleared 0 is the far plane, and
// `depth_linear` holds the SDK's reversed-infinite `DeviceToViewSpaceTransformFactors`.
//
// ## Colour
//
// A classic frame is gamma values in [0, 1], the SDK's SDR permutation (`exposure` 1). A modern
// frame is linear HDR: `hdr` is 1 and `exposure` is the zone look's, the SDK's `HDR_COLOR_INPUT`
// permutation taken at run time rather than compiled twice.
//
// ## Compile-time placeholders
//
// `fsr/upscaler.rs` resolves these tokens in each pass's source when it builds the shader:
//
// | token           | with tier 1 | without       |
// |-----------------|-------------|---------------|
// | `FSR3_FMT_R16`  | `r16float`  | `rgba16float` |
// | `FSR3_FMT_RG16` | `rg16float` | `rgba16float` |
// | `FSR3_FMT_R8`   | `r8unorm`   | `rgba8unorm`  |
// | `FSR3_FMT_RG8`  | `rg8unorm`  | `rgba8unorm`  |
//
// Only the leading channels are ever written or read, so the two differ in footprint alone.
// `FSR3_SHADING_CHANGE` becomes `true` or `false` and compiles the shading-change detector in or out.
// The naga validation test covers every combination.

struct Fsr3Params {
    // This frame's **jittered** clip → camera-relative world (world axes, origin at the eye): the
    // one matrix that matches the depth buffer, so unprojecting a depth sample lands on the point
    // the fragment shaded.
    cur_rel_from_clip: mat4x4<f32>,
    // This frame's **unjittered** camera-relative world → clip: half of a camera motion vector.
    clip_from_cur_rel: mat4x4<f32>,
    // The previous frame's unjittered clip, from this frame's camera-relative world:
    // `prev_clip_from_view × prev_view_rot × T(eye_now − eye_prev)`. Built on the CPU so no
    // world-space magnitude ever reaches the GPU, which keeps it exact 9000 yd from the origin.
    prev_clip_from_cur_rel: mat4x4<f32>,

    // Render (input) and display (output) resolutions, with reciprocals.
    render_size: vec2<f32>,
    display_size: vec2<f32>,
    inv_render_size: vec2<f32>,
    inv_display_size: vec2<f32>,

    // `Jitter()`: the sample offset in render pixels — pixel `p` of the render target sampled
    // the scene at `p + 0.5 - jitter` in unjittered screen space.
    jitter: vec2<f32>,
    // `PreviousFrameJitter()`.
    prev_jitter: vec2<f32>,
    // `DownscaleFactor()` = render_size / display_size.
    downscale_factor: vec2<f32>,
    // `DeviceToViewSpaceTransformFactors().xy`, negated in x: `view_space_depth` below.
    depth_linear: vec2<f32>,

    // `Exposure()`: 1 on a classic frame, the zone look's on a modern one.
    exposure: f32,
    // `FrameIndex()`: 0 on a frame whose history means nothing.
    frame_index: f32,
    // `JitterSequenceLength()`.
    jitter_sequence_length: f32,
    // `ViewSpaceToMetersFactor()`: one yard.
    view_space_to_meters: f32,

    // `VelocityFactor()`, `AccumulationAddedPerFrame()`, `MinDisocclusionAccumulation()`.
    velocity_factor: f32,
    accumulation_added_per_frame: f32,
    min_disocclusion_accumulation: f32,
    // RCAS sharpness as a linear multiplier, `exp2(-stops)`.
    sharpness: f32,

    // 1 on a modern (linear HDR) frame: the `HDR_COLOR_INPUT` permutation.
    hdr: f32,
    // `GenReactiveScale()` of the generated reactive mask (`motion.wgsl`).
    reactive_scale: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Fsr3Params;

// --- constants ----------------------------------------------------------------
//
// `ffx_fsr3upscaler_common.h`.

const FSR3_FP16_MIN: f32 = 6.10e-05;
const FSR3_FP16_MAX: f32 = 65504.0;
const FSR3_EPSILON: f32 = 6.10e-05;
const FSR3_FP32_MAX: f32 = 3.402823466e+38;
const FSR3_FP32_MIN: f32 = 1.175494351e-38;

/// A bilinear tap below this weight contributes nothing worth scattering into,
/// or reading back out of, the reconstructed depth buffer.
const RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD: f32 = FSR3_EPSILON * 10.0;

/// The Lanczos kernel's weights are not normalised in the accumulation — the
/// running total *is* the history weight — so they carry a fixed scale, and the
/// current frame's contribution is the average weight one jittered sample set
/// lands with.
const UPSAMPLE_LANCZOS_WEIGHT_SCALE: f32 = 1.0 / 16.0;
const AVERAGE_LANCZOS_WEIGHT_PER_FRAME: f32 = 0.74 * UPSAMPLE_LANCZOS_WEIGHT_SCALE;

/// Lock lifetime: a lock only starts protecting its feature once it is past
/// `LOCK_THRESHOLD`, and saturates at `LOCK_MAX`.
const LOCK_THRESHOLD: f32 = 1.0;
const LOCK_MAX: f32 = 2.0;

/// Levels of the shading-change pyramid the detector reads, and the level it
/// starts from (`s_MipLevelsToUse` / `iShadingChangeMipStart`).
const SHADING_CHANGE_MIP_LEVELS: i32 = 3;
const SHADING_CHANGE_MIP_START: i32 = 0;
/// Taps in one shading-change sample set (`SHADING_CHANGE_SET_SIZE`).
const SHADING_CHANGE_SET_SIZE: i32 = 5;

/// Channels of the dilated reactive-masks texture.
const REACTIVE: i32 = 0;
const DISOCCLUSION: i32 = 1;
const SHADING_CHANGE: i32 = 2;
const ACCUMULATION: i32 = 3;

// --- basic helpers ------------------------------------------------------------

/// Reciprocal with the divisor floored away from zero. Every division in these
/// passes goes through this or an explicit `max`: a single NaN in a temporal
/// accumulation is permanent, since it feeds back through the history.
fn rcp_safe(x: f32) -> f32 {
    return 1.0 / max(x, 1.0 / 65536.0);
}

/// Render resolution as integers, and the half-resolution grid the shading-change
/// detector runs on (`ShadingChangeRenderSize()`, truncated exactly as there).
fn render_size_i() -> vec2<i32> {
    return vec2<i32>(params.render_size);
}

fn display_size_i() -> vec2<i32> {
    return vec2<i32>(params.display_size);
}

fn shading_change_size_i() -> vec2<i32> {
    return vec2<i32>(params.render_size * 0.5);
}

/// Eye-space distance (positive) for a device depth: `GetViewSpaceDepth`,
/// `factors.y / (device - factors.x)`. For reverse-Z with an infinite far plane the SDK's factors
/// are `(-FLT_EPSILON, near)`, so the far plane (a cleared 0) is a large finite distance rather than
/// a division by zero; `depth_linear` holds `(FLT_EPSILON, near)`.
fn view_space_depth(device_depth: f32) -> f32 {
    let denom = device_depth + params.depth_linear.x;
    return params.depth_linear.y / select(denom, FSR3_FP32_MIN, abs(denom) < FSR3_FP32_MIN);
}

fn view_space_depth_meters(device_depth: f32) -> f32 {
    return view_space_depth(device_depth) * params.view_space_to_meters;
}

/// Motion magnitude expressed in the pixels it would cover on a 4K screen, which
/// is the scale every velocity threshold in FSR is written against — so those
/// thresholds mean the same thing at any render resolution.
fn get_4k_velocity(mv: vec2<f32>) -> f32 {
    return length(mv * vec2<f32>(3840.0, 2160.0));
}

/// `min(a, b) / max(a, b)`: 1 when the two agree, falling to 0 as they diverge.
fn min_divided_by_max(v0: f32, v1: f32, on_zero: f32) -> f32 {
    let m = max(v0, v1);
    return select(on_zero, min(v0, v1) / m, m != 0.0);
}

/// Luminance, Rec. 709 weights (`RGBToLuma`). Linear on a modern frame; on a classic frame the
/// input is gamma-encoded, which makes this a perceptual luma, as the port's SDR path always was.
fn rgb_to_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// The `HDR_COLOR_INPUT` permutation, chosen per frame.
fn is_hdr() -> bool {
    return params.hdr > 0.5;
}

/// `Tonemap`: the reversible max-channel tonemap the HDR path blends in.
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    return c / (max(max(0.0, c.r), max(c.g, c.b)) + 1.0);
}

/// `InverseTonemap`, with `FSR3UPSCALER_TONEMAP_EPSILON` = 1 / FP16_MAX.
fn inverse_tonemap(c: vec3<f32>) -> vec3<f32> {
    return c / max(1.0 / FSR3_FP16_MAX, 1.0 - max(c.r, max(c.g, c.b)));
}

/// YCoCg, the space the rectification box is built in: one luma axis and two
/// chroma axes, so a colour box tracks brightness separately from hue and clips
/// far less aggressively than an RGB box on the same neighbourhood.
fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

/// uv -> normalized device coordinates (y flips: NDC is up, uv is down).
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

/// The inverse of [`uv_to_ndc`], for a clip position that has already been
/// divided through by w.
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
}

/// `IsUvInside`. Anything outside has no history — the surface was off-screen
/// last frame.
fn is_uv_inside(uv: vec2<f32>) -> bool {
    return all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
}

/// `IsOnScreen`.
fn is_on_screen(p: vec2<i32>, size: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < size);
}

/// `ClampLoad`: an offset neighbour, kept inside the image.
fn clamp_load(p: vec2<i32>, offset: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    return clamp(p + offset, vec2<i32>(0), size - vec2<i32>(1));
}

/// `ClampUv`: pulls a uv onto the half-texel border so a linear fetch cannot
/// reach past the edge of the region actually written. Every target here is
/// allocated at exactly the size it is used at, so the reference's separate
/// texture/resource sizes collapse into this one argument.
fn clamp_uv(uv: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let loc = clamp(uv * size, vec2<f32>(0.5), size - vec2<f32>(0.5));
    return loc / size;
}

// --- kernels ------------------------------------------------------------------

/// FSR's polynomial stand-in for a windowed Lanczos-2 kernel, evaluated on the
/// **squared** sample distance (`Lanczos2ApproxSq`):
/// `(25/16 (2/5 x^2 - 1)^2 - 9/16) (1/4 x^2 - 1)^2`. The first factor is the
/// base lobe including its negative ring, the second the window that takes it to
/// zero at x = 2.
fn lanczos2_approx_sq(d2: f32) -> f32 {
    let x2 = min(d2, 4.0);
    let a = (2.0 / 5.0) * x2 - 1.0;
    let b = (1.0 / 4.0) * x2 - 1.0;
    return ((25.0 / 16.0) * (a * a) - (25.0 / 16.0 - 1.0)) * (b * b);
}

/// The real windowed Lanczos-2 kernel, `sinc(x) sinc(x/2)`, which the history
/// resample uses (`FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE 0`) where
/// the approximation's error would compound frame over frame.
fn lanczos2(x_in: f32) -> f32 {
    let x = min(abs(x_in), 2.0);
    if (x < FSR3_EPSILON) {
        return 1.0;
    }
    let pi = 3.141592653589793;
    return (sin(pi * x) / (pi * x)) * (sin(0.5 * pi * x) / (0.5 * pi * x));
}

/// The four taps and weights a bilinear fetch at `uv` would use, as integer
/// coordinates — for the passes that have to reason about *which* texels a
/// reprojection lands on rather than just their blend.
struct BilinearSamplingData {
    base_pos: vec2<i32>,
    weights: vec4<f32>,
};

fn bilinear_sampling_data(uv: vec2<f32>, size: vec2<f32>) -> BilinearSamplingData {
    let px = uv * size - vec2<f32>(0.5);
    var data: BilinearSamplingData;
    data.base_pos = vec2<i32>(floor(px));
    let f = fract(px);
    data.weights = vec4<f32>(
        (1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y,
        f.x * f.y,
    );
    return data;
}

/// The offset of bilinear tap `i`, matching `BilinearSamplingData::weights`.
fn bilinear_offset(i: i32) -> vec2<i32> {
    return vec2<i32>(i & 1, i >> 1);
}

// --- rectification box --------------------------------------------------------
//
// The neighbourhood statistics the history is clipped against, accumulated one
// tap at a time: a weighted mean and second moment (which become the centre and
// standard deviation), plus the unweighted extremes the upsample's deringing
// clamp uses.

struct RectificationBox {
    box_center: vec3<f32>,
    box_vec: vec3<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    weight: f32,
};

fn rectification_box_add_sample(
    box_in: RectificationBox,
    initial: bool,
    color: vec3<f32>,
    weight: f32,
) -> RectificationBox {
    var b = box_in;
    let weighted = color * weight;
    if (initial) {
        b.aabb_min = color;
        b.aabb_max = color;
        b.box_center = weighted;
        b.box_vec = color * weighted;
        b.weight = weight;
    } else {
        b.aabb_min = min(b.aabb_min, color);
        b.aabb_max = max(b.aabb_max, color);
        b.box_center += weighted;
        b.box_vec += color * weighted;
        b.weight += weight;
    }
    return b;
}

/// Turns the running sums into a mean and a standard deviation.
fn rectification_box_compute_variance(box_in: RectificationBox) -> RectificationBox {
    var b = box_in;
    b.weight = select(1.0, b.weight, abs(b.weight) > FSR3_FP32_MIN);
    b.box_center /= b.weight;
    b.box_vec /= b.weight;
    b.box_vec = sqrt(abs(b.box_vec - b.box_center * b.box_center));
    return b;
}
