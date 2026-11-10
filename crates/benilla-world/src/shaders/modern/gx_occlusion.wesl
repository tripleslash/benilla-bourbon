// fork: static_gx's Hi-Z tests (`static_gx/render/gpu_cull.rs`): the view uniform, the frustum
// test and the pyramid test a box goes through, shared by the draw cull (`gx_cull.wgsl`) and the
// meshlet cull in the task stage (`static_gx_mesh.wgsl`). Native only.
//
// Depth is reverse-Z: larger is nearer, and the pyramid keeps each texel's farthest (minimum).

// Keep in sync with `CullUniform` in `gpu_cull.rs` (176 bytes).
struct CullView {
    clip_from_world: mat4x4<f32>,
    // The matrix the pyramid was last built under; valid with `PREV_VALID`.
    prev_clip_from_world: mat4x4<f32>,
    // The pyramid's level-0 size in texels.
    pyramid_size: vec2<f32>,
    mip_count: u32,
    flags: u32,
    // This view's candidates: `count` draws from `args_start` in the frame's args buffers.
    args_start: u32,
    count: u32,
    pad: vec2<u32>,
    // The unjittered eye in world space, for the meshlet cone test.
    eye: vec4<f32>,
}

const PREV_VALID: u32 = 1u;
const STATS: u32 = 2u;
// The view mirrors handedness (a reflected camera): the cone test is skipped.
const MIRRORED: u32 = 4u;

struct Bound {
    center: vec3<f32>,
    extent: vec3<f32>,
}

fn corner(b: Bound, i: u32) -> vec4<f32> {
    let s = vec3<f32>(
        select(-1.0, 1.0, (i & 1u) != 0u),
        select(-1.0, 1.0, (i & 2u) != 0u),
        select(-1.0, 1.0, (i & 4u) != 0u),
    );
    return vec4<f32>(b.center + b.extent * s, 1.0);
}

// Every corner beyond one clip plane: left, right, bottom, top, or in front of the near plane
// (reverse-Z, `z > w`). The far plane is at infinity. Exact for an oblique near plane too.
fn outside(clip_from_world: mat4x4<f32>, b: Bound) -> bool {
    var left = true;
    var right = true;
    var bottom = true;
    var top = true;
    var near = true;
    for (var i = 0u; i < 8u; i += 1u) {
        let p = clip_from_world * corner(b, i);
        left = left && p.x < -p.w;
        right = right && p.x > p.w;
        bottom = bottom && p.y < -p.w;
        top = top && p.y > p.w;
        near = near && p.z > p.w;
    }
    return left || right || bottom || top || near;
}

// Is the box behind everything `pyramid` (level 0 `size` texels, `mip_count` levels) holds over
// its screen rectangle? A box that reaches the eye or the near plane, or lies off the pyramid's
// screen, is never occluded.
fn occluded(
    clip_from_world: mat4x4<f32>,
    b: Bound,
    pyramid: texture_2d<f32>,
    size: vec2<f32>,
    mip_count: u32,
) -> bool {
    var lo = vec2<f32>(1.0e30);
    var hi = vec2<f32>(-1.0e30);
    var nearest = 0.0;
    for (var i = 0u; i < 8u; i += 1u) {
        let p = clip_from_world * corner(b, i);
        if (p.w <= 0.0) {
            return false;
        }
        let n = p.xyz / p.w;
        lo = min(lo, n.xy);
        hi = max(hi, n.xy);
        nearest = max(nearest, n.z);
    }
    if (nearest >= 1.0) {
        return false;
    }
    // NDC to uv (v down), one level-0 texel wider each side: the jitter and the rounding.
    let margin = 1.0 / size;
    let uv_lo = clamp(vec2<f32>(lo.x, -hi.y) * 0.5 + 0.5 - margin, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_hi = clamp(vec2<f32>(hi.x, -lo.y) * 0.5 + 0.5 + margin, vec2<f32>(0.0), vec2<f32>(1.0));
    if (any(uv_hi <= uv_lo)) {
        return false;
    }
    // The level where the rectangle spans at most one texel, so it touches at most 2x2 of them.
    let texels = (uv_hi - uv_lo) * size;
    let level = min(u32(ceil(log2(max(max(texels.x, texels.y), 1.0)))), mip_count - 1u);
    let dims = vec2<f32>(textureDimensions(pyramid, level));
    let a = vec2<u32>(min(uv_lo * dims, dims - 1.0));
    let z = vec2<u32>(min(uv_hi * dims, dims - 1.0));
    let far = min(
        min(textureLoad(pyramid, a, level).r, textureLoad(pyramid, vec2<u32>(z.x, a.y), level).r),
        min(textureLoad(pyramid, vec2<u32>(a.x, z.y), level).r, textureLoad(pyramid, z, level).r),
    );
    return nearest < far;
}

// The cone byte that disables the cone test (`meshlet::NO_CONE`).
const NO_CONE: i32 = 127;

// One snorm8 byte of a packed cone, as -1..1.
fn cone_byte(cone: u32, shift: u32) -> f32 {
    return f32(extractBits(i32(cone), shift, 8u)) / 127.0;
}

// meshoptimizer's cone test (`meshlet::backfacing`): every triangle of the meshlet faces away
// from `eye`. `cone` packs the snorm8 axis in bytes 0..3 and the cutoff in byte 3.
fn backfacing(cone: u32, center: vec3<f32>, radius: f32, eye: vec3<f32>) -> bool {
    if (extractBits(i32(cone), 24u, 8u) == NO_CONE) {
        return false;
    }
    let axis = vec3<f32>(cone_byte(cone, 0u), cone_byte(cone, 8u), cone_byte(cone, 16u));
    let v = center - eye;
    return dot(v, axis) >= cone_byte(cone, 24u) * length(v) + radius;
}
