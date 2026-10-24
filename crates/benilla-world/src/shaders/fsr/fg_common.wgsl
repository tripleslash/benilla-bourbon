// Shared declarations for the FSR 3.1.5 *frame generation* passes
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`fsr/framegen.rs`).
//
// Prepended to every `fsr3_fg_*.wgsl` module, exactly as `common.wgsl` is
// prepended to the upscaler's. The two preludes are deliberately separate: the
// upscaler and the interpolator are distinct effects in the SDK
// (`ffx_fsr3upscaler` and `ffx_frameinterpolation`), with their own constant
// buffers, and a pass belongs to one or the other. A handful of small geometric
// helpers therefore appear in both; that is the price of each pass compiling
// against exactly the constants its reference counterpart had.
//
// Names follow `ffx_frameinterpolation_common.h` in snake_case.
//
// ## Compile-time placeholders
//
// Resolved by `preprocess` in `fsr/upscaler.rs`, shared with the upscaler:
//
// | token           | with tier 1 | without       |
// |-----------------|-------------|---------------|
// | `FSR3_FMT_RG8`  | `rg8unorm`  | `rgba8unorm`  |
//
// See `common.wgsl` for the full table and why it exists. One more,
// `FG_PYRAMID_REDUCE_MV`, becomes `true` or `false` and picks which pyramid
// `fg_pyramid_reduce.wgsl` is building.
//
// A placeholder resolves to a reserved word or a WGSL enumerant, never to a name
// declared in one of these files.

struct FgParams {
    // Render (input) resolution — the grid the motion vector field, the depth
    // estimates and the disocclusion mask live on — and the display resolution
    // the interpolated frame is produced at. `MaxRenderSize` and
    // `InterpolationRectSize` in the reference are these same two values: the
    // engine has no dynamic resolution and no letterboxing, so every rect the
    // reference carries around collapses to the full image and the letterbox
    // scale factors it multiplies through collapse to one.
    render_size: vec2<f32>,
    display_size: vec2<f32>,
    inv_render_size: vec2<f32>,
    inv_display_size: vec2<f32>,

    // `DeviceToViewSpaceTransformFactors()`. `xy` turn a device depth into an
    // eye-space distance *in metres* (the metres factor is folded into `y`);
    // `zw` scale a normalized device coordinate into eye-space x and y at unit
    // distance, i.e. `tan(fovH/2)` and `tan(fovV/2)`.
    device_to_view_depth: vec4<f32>,

    // `TanHalfFoV()`: the horizontal half-angle's tangent, which is
    // `device_to_view_depth.z`, kept separately because the reference does.
    tan_half_fov: f32,
    // `FrameIndexSinceLastReset()`. Zero means there is no previous frame to
    // interpolate against and the pass falls back to copying the current one;
    // below ten it distrusts the game vector field's colour match, which needs a
    // few frames of settled history before it means anything.
    frame_index_since_reset: f32,
    // 1 when the presentation blit writes through an sRGB view and its stored
    // (already encoded) values have to be decoded first, since the hardware
    // re-encodes on write.
    srgb_output: f32,

    // How many levels each inpainting pyramid actually has. The reference walks a
    // fixed 11 (vectors) and 10 (colour) levels and lets the out-of-bounds tests
    // absorb the ones that do not exist; here the chain is built level by level
    // (see `fsr/framegen.rs` on the single-pass downsampler) so the count is
    // known exactly, and the search loops stop at it.
    mv_pyramid_mips: f32,
    color_pyramid_mips: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: FgParams;

// --- constants ----------------------------------------------------------------

const FG_EPSILON: f32 = 1e-03;
const FG_FLT_MAX: f32 = 3.402823466e+38;

/// A bilinear tap below this weight is not worth scattering a depth into
/// (`fReconstructedDepthBilinearWeightThreshold`).
const RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD: f32 = FG_EPSILON;

// --- basic helpers ------------------------------------------------------------

fn render_size_i() -> vec2<i32> {
    return vec2<i32>(params.render_size);
}

fn display_size_i() -> vec2<i32> {
    return vec2<i32>(params.display_size);
}

/// `IsOnScreen`.
fn is_on_screen(p: vec2<i32>, size: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < size);
}

/// `IsUvInside`. The reference's bounds are strict on both sides, and the
/// interpolation leans on that: a vector landing exactly on the border is
/// treated as having left the frame.
fn is_uv_inside(uv: vec2<f32>) -> bool {
    return uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0;
}

/// `MinDividedByMax`: 1 when the two agree, falling to 0 as they diverge.
fn min_divided_by_max(v0: f32, v1: f32) -> f32 {
    let m = max(v0, v1);
    return select(0.0, min(v0, v1) / m, m != 0.0);
}

/// `NormalizedDot3`, scale-invariant colour similarity: both vectors are divided
/// by the longer one's length before the dot, so it answers "do these point the
/// same way" without a brightness difference dominating the answer.
fn normalized_dot3(v0: vec3<f32>, v1: vec3<f32>) -> f32 {
    let max_length = max(length(v0), length(v1));
    if (max_length > 0.0) {
        return dot(v0 / max_length, v1 / max_length);
    }
    return 1.0;
}

/// Rec. 709 luma of a *linear* colour (`RGBToLuma`).
fn rgb_to_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// The sRGB electro-optical transfer function, i.e. `ffxLinearFromSrgb`.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow(max(c + vec3<f32>(0.055), vec3<f32>(0.0)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

/// `RawRGBToLinear` at `BackBufferTransferFunction() == 0`. The interpolation
/// sources are the engine's sRGB-encoded 8-bit presentation images — exactly the
/// SDR back buffer the reference's first case describes — so the switch over
/// transfer functions collapses to this one branch and the PQ/scRGB paths, along
/// with the min/max luminance constants that drive them, are omitted.
fn raw_rgb_to_linear(raw: vec3<f32>) -> vec3<f32> {
    return srgb_to_linear(raw);
}

/// `RawRGBToLuminance`.
fn raw_rgb_to_luminance(raw: vec3<f32>) -> f32 {
    return rgb_to_luma(raw_rgb_to_linear(raw));
}

// --- projection ---------------------------------------------------------------

/// `ConvertFromDeviceDepthToViewSpace`: eye-space distance in metres.
/// `device_to_view_depth.x` is `-proj[2][2]` and `.y` is `proj[3][2]` scaled by
/// the view-space-to-metres factor, so this inverts the projection's depth row.
fn view_space_depth(device_depth: f32) -> f32 {
    let denom = device_depth - params.device_to_view_depth.x;
    // The reference divides unguarded. A depth sitting exactly on the far plane
    // makes the denominator vanish, and the infinity that produces would
    // propagate through every comparison downstream.
    return params.device_to_view_depth.y / select(denom, 1.175494351e-38, abs(denom) < 1.175494351e-38);
}

/// `ComputeNdc`: pixel position to normalized device coordinates.
fn compute_ndc(px: vec2<f32>, size: vec2<i32>) -> vec2<f32> {
    return px / vec2<f32>(size) * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
}

/// `GetViewSpacePosition`: the eye-space point a device depth at a given pixel
/// unprojects to.
fn view_space_position(viewport_pos: vec2<i32>, viewport_size: vec2<i32>, device_depth: f32) -> vec3<f32> {
    let z = view_space_depth(device_depth);
    let ndc = compute_ndc(vec2<f32>(viewport_pos), viewport_size);
    return vec3<f32>(
        params.device_to_view_depth.z * ndc.x * z,
        params.device_to_view_depth.w * ndc.y * z,
        z,
    );
}

// --- bilinear sampling --------------------------------------------------------

/// The four taps and weights a bilinear fetch at `uv` would use, as integer
/// coordinates (`GetBilinearSamplingData`). Frame generation *scatters* far more
/// than it gathers — depths and motion vectors are pushed forward along their own
/// motion — so this, not a sampler, is how most of the chain reprojects.
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

// --- the motion vector field --------------------------------------------------
//
// The heart of the algorithm. Both halves of the interpolated frame's motion —
// where each of its pixels came from in frame N-1 and where it goes in frame N —
// are resolved on one grid, and that grid is built by *scattering*: every render
// pixel pushes its own half-motion into the cell it will be halfway through
// moving to. Many pixels land on the same cell, and the one that should win is
// the one in front, so the resolution has to happen in the atomic itself.
//
// The trick is that a single `atomicMax` over a packed 32-bit word does it. The
// priority occupies the high bits and the vector the low, so comparing the words
// as integers compares the priorities first and only falls back to the vector's
// bit pattern on a tie. From the top:
//
// | bit  | field                                                 |
// |------|-------------------------------------------------------|
// | 31   | primary (a real vector) rather than secondary (a trail)|
// | 21-30| high priority — nearness, so the foreground wins       |
// | 16-20| low priority — how well the two frames' colours agree  |
// | 0-15 | the vector component, as an IEEE half                  |
//
// X and Y are stored in separate images that share the priority bits, so a tie
// on priority can in principle take x from one contributor and y from another.
// That is true of the reference as well.

const MOTION_VECTOR_FIELD_PRIMARY_VECTOR_INDICATION_BIT: u32 = 0x80000000u;
const PRIORITY_LOW_MAX: u32 = 31u;
const PRIORITY_HIGH_MAX: u32 = 1023u;
const PRIORITY_LOW_OFFSET: u32 = 16u;
const PRIORITY_HIGH_OFFSET: u32 = 21u;

struct VectorFieldEntry {
    motion_vector: vec2<f32>,
    high_priority_factor: f32,
    low_priority_factor: f32,
    valid: bool,
    primary: bool,
    secondary: bool,
    in_painted: bool,
    velocity: f32,
    neg_outside: bool,
    pos_outside: bool,
};

fn new_vector_field_entry() -> VectorFieldEntry {
    var e: VectorFieldEntry;
    e.motion_vector = vec2<f32>(0.0);
    e.high_priority_factor = 0.0;
    e.low_priority_factor = 0.0;
    e.valid = false;
    e.primary = false;
    e.secondary = false;
    e.in_painted = false;
    e.velocity = 0.0;
    e.neg_outside = false;
    e.pos_outside = false;
    return e;
}

fn packed_vector_field_entry_is_primary(packed: u32) -> bool {
    return (packed & MOTION_VECTOR_FIELD_PRIMARY_VECTOR_INDICATION_BIT) != 0u;
}

/// `ffxF32ToF16` — the half's bit pattern in the low 16 bits.
fn f32_to_f16_bits(v: f32) -> u32 {
    return pack2x16float(vec2<f32>(v, 0.0)) & 0xFFFFu;
}

/// The inverse. The priority bits are masked off first: left in place they would
/// decode as the packed pair's second half, and an exponent of all ones there
/// would hand back a NaN.
fn f16_bits_to_f32(packed: u32) -> f32 {
    return unpack2x16float(packed & 0xFFFFu).x;
}

fn pack_vector_field_entries(
    is_primary: bool,
    high_priority_factor: u32,
    low_priority_factor: u32,
    motion_vector: vec2<f32>,
) -> vec2<u32> {
    let priority = (u32(is_primary) * MOTION_VECTOR_FIELD_PRIMARY_VECTOR_INDICATION_BIT)
        | ((high_priority_factor & PRIORITY_HIGH_MAX) << PRIORITY_HIGH_OFFSET)
        | ((low_priority_factor & PRIORITY_LOW_MAX) << PRIORITY_LOW_OFFSET);

    return vec2<u32>(
        priority | f32_to_f16_bits(motion_vector.x),
        priority | f32_to_f16_bits(motion_vector.y),
    );
}

fn unpack_vector_field_entries(packed: vec2<u32>) -> VectorFieldEntry {
    var e = new_vector_field_entry();
    e.high_priority_factor =
        f32((packed.x >> PRIORITY_HIGH_OFFSET) & PRIORITY_HIGH_MAX) / f32(PRIORITY_HIGH_MAX);
    e.low_priority_factor =
        f32((packed.x >> PRIORITY_LOW_OFFSET) & PRIORITY_LOW_MAX) / f32(PRIORITY_LOW_MAX);

    e.primary = packed_vector_field_entry_is_primary(packed.x);
    e.valid = e.high_priority_factor > 0.0;
    e.secondary = e.valid && !e.primary;

    // Secondary vectors pack an inverted depth priority — the point of a trail is
    // to be *behind* what drew it — so the factor has to be flipped back on the
    // way out, where the inpainting pyramid reads it as a plain confidence.
    if (e.secondary) {
        e.high_priority_factor = 1.0 - e.high_priority_factor;
    }

    e.motion_vector = vec2<f32>(f16_bits_to_f32(packed.x), f16_bits_to_f32(packed.y));
    e.in_painted = false;
    return e;
}

/// Fills in the three derived fields every consumer wants: whether following the
/// vector backwards (towards frame N-1) or forwards (towards frame N) leaves the
/// image, and how long it is.
fn finish_vector_field_entry(e_in: VectorFieldEntry, uv: vec2<f32>) -> VectorFieldEntry {
    var e = e_in;
    e.neg_outside = !is_uv_inside(uv - e.motion_vector);
    e.pos_outside = !is_uv_inside(uv + e.motion_vector);
    e.velocity = length(e.motion_vector);
    return e;
}

// --- pyramid reduction --------------------------------------------------------
//
// Both inpainting pyramids exist to answer "what is nearby that I could copy?",
// and the answer has to skip whatever is missing. A plain box filter would
// average holes into their neighbours and spread the hole up the chain instead of
// closing it, so each level is a *weighted* mean whose weight is the sample's own
// validity: an invalid sample contributes nothing and does not dilute the ones
// that do. These are the two `SpdReduce4` overrides, one per pyramid.

/// The motion vector field's reduction: `(mv.xy, high, low)`, valid where the
/// high-priority factor is non-zero.
fn reduce4_mv(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    let w = vec4<f32>(
        f32(v0.z > 0.0),
        f32(v1.z > 0.0),
        f32(v2.z > 0.0),
        f32(v3.z > 0.0),
    );
    let sum = w.x + w.y + w.z + w.w;
    let acc = v0 * w.x + v1 * w.y + v2 * w.z + v3 * w.w;
    return acc / select(1.0, sum, sum > FG_EPSILON);
}

/// The interpolated image's reduction: `(rgb, weight)`, where the weight is how
/// much of a *real* pixel this is — the complement of the inpainting weight, so a
/// fully inpainted pixel is exactly the thing that must not be inpainted from.
fn reduce4_color(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    let sum = v0.w + v1.w + v2.w + v3.w;
    if (sum == 0.0) {
        return vec4<f32>(0.0);
    }
    return (v0 * v0.w + v1 * v1.w + v2 * v2.w + v3 * v3.w) / sum;
}

// --- vector inpainting --------------------------------------------------------
//
// A cell nothing scattered into has no vector, and the interpolation needs one
// for every cell. The pyramid built from the field is the search structure: level
// by level, ask whether any *valid* vector exists in the neighbourhood this cell
// falls into, and stop at the first level that answers yes. Coarse levels cover
// wide holes; the loop reaching them at all means everything nearer was empty
// too.

/// `ComputeMvInpaintingLevel`: one level's answer, as a bilinear blend restricted
/// to the taps that hold a vector and additionally weighted by how much each one
/// is trusted.
fn compute_mv_inpainting_level(
    pyramid: texture_2d<f32>,
    uv: vec2<f32>,
    level: i32,
    tex_size: vec2<i32>,
) -> vec4<f32> {
    let info = bilinear_sampling_data(uv, vec2<f32>(tex_size));

    var sum = 0.0;
    var color = vec4<f32>(0.0);

    for (var i = 0; i < 4; i = i + 1) {
        let sample_pos = info.base_pos + bilinear_offset(i);
        if (!is_on_screen(sample_pos, tex_size)) {
            continue;
        }
        let s = textureLoad(pyramid, sample_pos, level);
        // `z` is the high-priority factor: zero means the cell held nothing, and
        // a low value means it held something distant or merely inferred.
        let weight = info.weights[i] * f32(s.z > 0.0) * s.z;
        sum += weight;
        color += s * weight;
    }

    return color / select(1.0, sum, sum > 0.0);
}

/// The mip walk itself: the coarsest-to-finest search short-circuits on the first
/// level that returns a vector with any confidence at all.
fn inpaint_game_field_mv(pyramid: texture_2d<f32>, uv: vec2<f32>) -> VectorFieldEntry {
    var tex_size = render_size_i();
    var inpainted = vec4<f32>(0.0);
    let mips = i32(params.mv_pyramid_mips);

    for (var level = 0; level < mips && inpainted.w == 0.0; level = level + 1) {
        tex_size = tex_size / 2;
        inpainted = compute_mv_inpainting_level(pyramid, uv, level, tex_size);
    }

    var e = new_vector_field_entry();
    e.motion_vector = inpainted.xy;
    e.high_priority_factor = inpainted.z;
    e.low_priority_factor = inpainted.w;
    e.in_painted = true;
    return e;
}

/// The optical flow vector field, which this port does not compute — see the
/// module docs in `fsr/framegen.rs` for why, and for what is lost. The
/// reference's own "no optical flow" path is exactly this: the field is cleared
/// and never written, so every sample of it reads back as an entry with no
/// vector and no priority, and the interpolation's optical-flow branch degrades
/// into an unwarped blend of the two source frames.
fn sample_optical_flow_motion_vector_field(uv: vec2<f32>) -> VectorFieldEntry {
    return finish_vector_field_entry(new_vector_field_entry(), uv);
}
