// fork: GPU cull (`static_gx/render/gpu_cull.rs`) — the two phases of static_gx's Hi-Z occlusion
// cull over one view's bucketed draws (`static_gx/render/batch.rs`, "The indirect layout").
//
// One thread per candidate draw. A draw's bound is its region's (region table rows `row + 1` and
// `row + 2`), or its selection key's (`row + 3 + items + 2 * key`) when the CPU named one.
// - `cull_early`: outside the current frustum, culled for good; else tested against last frame's
//   pyramid under last frame's matrix. Not occluded there: written into its bucket (`args`,
//   `counts`) for the early prepass and the colour pass. Occluded: marked deferred.
// - `cull_late`: a deferred draw tested against this frame's pyramid, built from the early depth.
//   Not occluded: appended to its bucket (the colour pass draws it) and written into the late
//   buckets (`late_args`, `late_counts`), which the late prepass draws.
// Depth is reverse-Z: larger is nearer, and the pyramid keeps each texel's farthest (minimum).

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// Keep in sync with `CullUniform` in `gpu_cull.rs`.
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
}

const PREV_VALID: u32 = 1u;
const STATS: u32 = 2u;
const NO_KEY: u32 = 0xffffu;
const HEADER_ROWS: u32 = 3u;
const DRAWN: u32 = 0u;
const DEFERRED: u32 = 1u;

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<storage, read> table: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> cand: array<DrawArgs>;
// Per candidate: count slot, its bucket's first arg, selection key, phase-1 verdict.
@group(0) @binding(3) var<storage, read_write> draw_meta: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> args: array<DrawArgs>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> late_args: array<DrawArgs>;
@group(0) @binding(7) var<storage, read_write> late_counts: array<atomic<u32>>;
@group(0) @binding(8) var pyramid: texture_2d<f32>;
// `WOW_GX_PERF`: candidates, early survivors, late survivors.
@group(0) @binding(9) var<storage, read_write> stats: array<atomic<u32>, 4>;

struct Bound {
    center: vec3<f32>,
    extent: vec3<f32>,
}

// The world box of region `row`, or of its selection key `key`.
fn bound(row: u32, key: u32) -> Bound {
    let head = table[row];
    var c = table[row + 1u];
    var h = table[row + 2u];
    if (key != NO_KEY) {
        let at = row + HEADER_ROWS + head.w + 2u * key;
        c = table[at];
        h = table[at + 1u];
    }
    return Bound(bitcast<vec3<f32>>(head.xyz) + bitcast<vec3<f32>>(c.xyz), bitcast<vec3<f32>>(h.xyz));
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

// Is the box behind everything the pyramid holds over its screen rectangle? A box that reaches the
// eye or the near plane, or lies off the pyramid's screen, is never occluded.
fn occluded(clip_from_world: mat4x4<f32>, b: Bound) -> bool {
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
    let margin = 1.0 / view.pyramid_size;
    let uv_lo = clamp(vec2<f32>(lo.x, -hi.y) * 0.5 + 0.5 - margin, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_hi = clamp(vec2<f32>(hi.x, -lo.y) * 0.5 + 0.5 + margin, vec2<f32>(0.0), vec2<f32>(1.0));
    if (any(uv_hi <= uv_lo)) {
        return false;
    }
    // The level where the rectangle spans at most one texel, so it touches at most 2x2 of them.
    let texels = (uv_hi - uv_lo) * view.pyramid_size;
    let level = min(u32(ceil(log2(max(max(texels.x, texels.y), 1.0)))), view.mip_count - 1u);
    let dims = vec2<f32>(textureDimensions(pyramid, level));
    let a = vec2<u32>(min(uv_lo * dims, dims - 1.0));
    let z = vec2<u32>(min(uv_hi * dims, dims - 1.0));
    let far = min(
        min(textureLoad(pyramid, a, level).r, textureLoad(pyramid, vec2<u32>(z.x, a.y), level).r),
        min(textureLoad(pyramid, vec2<u32>(a.x, z.y), level).r, textureLoad(pyramid, z, level).r),
    );
    return nearest < far;
}

// Append the draw to its bucket; the count it bumps is the one the multi-draw reads.
fn draw_into_bucket(a: DrawArgs, m: vec4<u32>) {
    let slot = atomicAdd(&counts[m.x], 1u);
    args[m.y + slot] = a;
}

@compute @workgroup_size(64, 1, 1)
fn cull_early(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= view.count) {
        return;
    }
    let d = view.args_start + id.x;
    let a = cand[d];
    let m = draw_meta[d];
    let b = bound(a.first_instance, m.z);
    var verdict = DRAWN;
    if (!outside(view.clip_from_world, b)) {
        if ((view.flags & PREV_VALID) != 0u && occluded(view.prev_clip_from_world, b)) {
            verdict = DEFERRED;
        } else {
            draw_into_bucket(a, m);
            if ((view.flags & STATS) != 0u) {
                atomicAdd(&stats[1], 1u);
            }
        }
    }
    draw_meta[d].w = verdict;
    if ((view.flags & STATS) != 0u) {
        atomicAdd(&stats[0], 1u);
    }
}

@compute @workgroup_size(64, 1, 1)
fn cull_late(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= view.count) {
        return;
    }
    let d = view.args_start + id.x;
    let m = draw_meta[d];
    if (m.w != DEFERRED) {
        return;
    }
    let a = cand[d];
    if (occluded(view.clip_from_world, bound(a.first_instance, m.z))) {
        return;
    }
    draw_into_bucket(a, m);
    let slot = atomicAdd(&late_counts[m.x], 1u);
    late_args[m.y + slot] = a;
    if ((view.flags & STATS) != 0u) {
        atomicAdd(&stats[2], 1u);
    }
}
