// FSR 3.1.5 pass 2 — the shading-change pyramid
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_fsr3upscaler_shading_change_pyramid.h`).
//
// The single largest thing FSR 3.1 added over the 2.x lineage. A pixel's history
// can be invalid for two quite different reasons, and the older algorithm only
// had machinery for the first:
//
// * The *geometry* moved — the surface was hidden last frame, or its motion
//   vector points somewhere else. Depth and motion vectors detect that, and the
//   disocclusion test in `prepare_reactivity.wgsl` still does.
// * The *shading* changed while the geometry stood still — a light turned on, a
//   shadow swept across, a reflection slid over a surface that never moved. No
//   depth or motion test can see this. Accumulating through it is what makes a
//   temporal upscaler lag behind lighting changes, and the neighbourhood clamp
//   cannot catch it because every value involved is locally valid.
//
// The detector compares this frame's luminance against the previous frame's,
// reprojected — but a raw per-pixel difference is dominated by the jitter (every
// pixel's sample lands somewhere else each frame, so *every* pixel differs). So
// the comparison is between two five-tap **sets** around each position, and what
// it reports is the *smallest* difference any pairing of the two sets admits
// (`compute_minimum_difference`): if some sample in the neighbourhood can explain
// this frame's value, nothing changed and the jitter is the whole story.
//
// That per-pixel signal is then reduced into a pyramid of (mean difference, mean
// sign) pairs. Both halves matter, and the product of them is what
// `shading_change.wgsl` reads: a region whose pixels all moved the same way
// is a real shading change and keeps its magnitude, while a region of
// disagreeing differences is noise and cancels itself to nothing.
//
// The reference builds all of this with SPD (single-pass downsampler) over a
// mip chain. Only the first three levels are ever read, and three levels fit
// inside one workgroup's tile, so the cross-workgroup atomic counter SPD needs
// to reach further has nothing to do here: this is one dispatch on the
// half-resolution grid, reducing 8x8 -> 4x4 -> 2x2 in workgroup memory.
//
// The farthest-depth mip the accumulation samples rides along. In the reference
// it falls out of the *auto-exposure* pyramid's first reduction; the engine has
// no auto-exposure (see `fsr/upscaler.rs` on colour space) and so no such pyramid, but
// the reduction is the same 2x2 average over the same grid, so it is produced
// here instead.

@group(0) @binding(1) var luma_tex: texture_2d<f32>;
@group(0) @binding(2) var prev_luma_tex: texture_2d<f32>;
@group(0) @binding(3) var dilated_mv_tex: texture_2d<f32>;
@group(0) @binding(4) var depths_tex: texture_2d<f32>;
@group(0) @binding(5) var mip0_out: texture_storage_2d<FSR3_FMT_RG16, write>;
@group(0) @binding(6) var mip1_out: texture_storage_2d<FSR3_FMT_RG16, write>;
@group(0) @binding(7) var mip2_out: texture_storage_2d<FSR3_FMT_RG16, write>;
@group(0) @binding(8) var farthest_mip1_out: texture_storage_2d<FSR3_FMT_R16, write>;

/// Whether the detector is compiled in at all (`SHADING_CHANGE_DETECTION` in
/// `fsr/upscaler.rs`). When false this pass keeps only the farthest-depth
/// downsample it carries alongside, `shading_change.wgsl` is never
/// dispatched, and every consumer reads the zero its target was allocated with.
const SHADING_CHANGE_ENABLED: bool = FSR3_SHADING_CHANGE;

/// The five-tap set: centre plus its four edge neighbours.
const SET_OFFSETS: array<vec2<i32>, 5> = array<vec2<i32>, 5>(
    vec2<i32>(0, 0),
    vec2<i32>(-1, 0),
    vec2<i32>(1, 0),
    vec2<i32>(0, -1),
    vec2<i32>(0, 1),
);

/// Sorting network for five elements, in the reference's order.
fn sort_set(set_in: array<f32, 5>) -> array<f32, 5> {
    var s = set_in;
    // Each step is a compare-and-swap of one pair.
    var lo = 0.0;
    lo = min(s[0], s[3]); s[3] = max(s[0], s[3]); s[0] = lo;
    lo = min(s[1], s[4]); s[4] = max(s[1], s[4]); s[1] = lo;
    lo = min(s[0], s[2]); s[2] = max(s[0], s[2]); s[0] = lo;
    lo = min(s[1], s[3]); s[3] = max(s[1], s[3]); s[1] = lo;
    lo = min(s[0], s[1]); s[1] = max(s[0], s[1]); s[0] = lo;
    lo = min(s[2], s[4]); s[4] = max(s[2], s[4]); s[2] = lo;
    lo = min(s[1], s[2]); s[2] = max(s[1], s[2]); s[1] = lo;
    lo = min(s[3], s[4]); s[4] = max(s[3], s[4]); s[3] = lo;
    lo = min(s[2], s[3]); s[3] = max(s[2], s[3]); s[2] = lo;
    return s;
}

/// The smallest relative difference any pairing of the two sorted sets admits,
/// signed by which set was brighter, walking them in the manner of a merge.
///
/// The differences are *relative* (`1 - min/max`) rather than absolute, so the
/// same fractional change counts the same in shadow and in highlight. A pair that
/// matches to within floating-point noise short-circuits the whole thing to
/// "no change": if any sample explains any other exactly, the neighbourhood is
/// the same neighbourhood.
fn compute_minimum_difference(set0_in: array<f32, 5>, set1_in: array<f32, 5>) -> f32 {
    let set0 = sort_set(set0_in);
    let set1 = sort_set(set1_in);

    var min_diff = FSR3_FP16_MAX - 1.0;
    var a = 0;
    var b = 0;

    let fmax = min(set0[SHADING_CHANGE_SET_SIZE - 1], set1[SHADING_CHANGE_SET_SIZE - 1]);
    if (fmax > FSR3_FP32_MIN) {
        for (var i = 0; i < SHADING_CHANGE_SET_SIZE; i = i + 1) {
            if (min_diff >= FSR3_FP16_MAX) {
                break;
            }
            let va = set0[a];
            let vb = set1[b];
            var diff = va - vb;
            if (abs(diff) > FSR3_FP16_MIN) {
                diff = sign(diff) * (1.0 - min_divided_by_max(va, vb, 0.0));
                min_diff = select(min_diff, diff, abs(diff) < abs(min_diff));
                // The reference advances `a` first and then tests the *updated*
                // `a` when advancing `b`; the dependency is deliberate.
                a = min(a + i32(va < vb), SHADING_CHANGE_SET_SIZE - 1);
                b = min(b + i32(set0[a] >= vb), SHADING_CHANGE_SET_SIZE - 1);
            } else {
                min_diff = FSR3_FP16_MAX;
            }
        }
    }

    return min_diff * f32(min_diff < (FSR3_FP16_MAX - 1.0));
}

/// This frame's five luminance taps around the pixel whose jittered sample landed
/// at `uv`.
fn current_luma_set(uv: vec2<f32>, size: vec2<i32>) -> array<f32, 5> {
    let uv_jittered = uv + params.jitter * params.inv_render_size;
    let base = vec2<i32>(floor(uv_jittered * params.render_size));

    var s: array<f32, 5>;
    for (var i = 0; i < SHADING_CHANGE_SET_SIZE; i = i + 1) {
        let q = clamp_load(base, SET_OFFSETS[i], size);
        s[i] = max(textureLoad(luma_tex, q, 0).x * params.exposure, FSR3_EPSILON);
    }
    return s;
}

struct PreviousLumaSet {
    samples: array<f32, 5>,
    exists: bool,
};

/// The same five taps a frame ago, followed along the motion vector — and aligned
/// to *that* frame's jitter, since it is a different sample grid.
fn previous_luma_set(uv: vec2<f32>, mv: vec2<f32>, size: vec2<i32>) -> PreviousLumaSet {
    var data: PreviousLumaSet;
    let uv_jittered = uv + params.prev_jitter * params.inv_render_size;
    let reprojected = uv_jittered + mv;
    data.exists = is_uv_inside(reprojected);

    if (data.exists) {
        let base = vec2<i32>(floor(reprojected * params.render_size));
        for (var i = 0; i < SHADING_CHANGE_SET_SIZE; i = i + 1) {
            let q = clamp_load(base, SET_OFFSETS[i], size);
            data.samples[i] = max(textureLoad(prev_luma_tex, q, 0).x * params.exposure, FSR3_EPSILON);
        }
    }
    return data;
}

/// One render-resolution pixel's contribution: the signed relative luminance
/// change, and its sign on its own, so the reduction can tell a coherent change
/// from cancelling noise.
fn shading_change_source(p_in: vec2<i32>, size: vec2<i32>) -> vec2<f32> {
    let p = clamp_load(p_in, vec2<i32>(0), size);
    let mv = textureLoad(dilated_mv_tex, p, 0).xy;
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;

    let previous = previous_luma_set(uv, mv, size);
    var diff = 0.0;
    if (previous.exists) {
        diff = compute_minimum_difference(current_luma_set(uv, size), previous.samples);
    }

    return vec2<f32>(diff, select(0.0, sign(diff), diff != 0.0));
}

/// The 8x8 tile of level 0 this workgroup produced, and the 4x4 of level 1 built
/// from it — enough for the three levels the detector reads.
var<workgroup> tile0: array<vec2<f32>, 64>;
var<workgroup> tile1: array<vec2<f32>, 16>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let render = render_size_i();
    let q = vec2<i32>(gid.xy);
    let local = vec2<i32>(lid.xy);

    // Level 0: the 2x2 average of the render-resolution signal. Threads outside
    // the level's extent still compute one — their loads clamp to the edge, as
    // the reference's do, and the reduction below reads them.
    var sum = vec2<f32>(0.0);
    var farthest = 0.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let src = q * 2 + vec2<i32>(dx, dy);
            if (SHADING_CHANGE_ENABLED) {
                sum += shading_change_source(src, render);
            }
            farthest += textureLoad(depths_tex, clamp_load(src, vec2<i32>(0), render), 0).y;
        }
    }

    // The farthest-depth mip is not part of the detector — the accumulation's
    // rectification box needs it either way — so it is written first and
    // unconditionally.
    if (all(q < vec2<i32>(textureDimensions(farthest_mip1_out)))) {
        textureStore(farthest_mip1_out, q, vec4<f32>(farthest * 0.25, 0.0, 0.0, 0.0));
    }

    // Everything below is the detector proper. Compiled out entirely when it is
    // switched off, leaving the downsample above as the whole pass.
    if (!SHADING_CHANGE_ENABLED) {
        return;
    }

    let level0 = sum * 0.25;
    tile0[lid.y * 8u + lid.x] = level0;

    if (all(q < vec2<i32>(textureDimensions(mip0_out)))) {
        textureStore(mip0_out, q, vec4<f32>(level0, 0.0, 0.0));
    }

    workgroupBarrier();

    // Level 1: 4x4 per workgroup.
    if (local.x < 4 && local.y < 4) {
        var acc = vec2<f32>(0.0);
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            for (var dx = 0u; dx < 2u; dx = dx + 1u) {
                acc += tile0[(lid.y * 2u + dy) * 8u + (lid.x * 2u + dx)];
            }
        }
        let level1 = acc * 0.25;
        tile1[lid.y * 4u + lid.x] = level1;
        let dst = vec2<i32>(wid.xy) * 4 + local;
        if (all(dst < vec2<i32>(textureDimensions(mip1_out)))) {
            textureStore(mip1_out, dst, vec4<f32>(level1, 0.0, 0.0));
        }
    }

    workgroupBarrier();

    // Level 2: 2x2 per workgroup.
    if (local.x < 2 && local.y < 2) {
        var acc = vec2<f32>(0.0);
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            for (var dx = 0u; dx < 2u; dx = dx + 1u) {
                acc += tile1[(lid.y * 2u + dy) * 4u + (lid.x * 2u + dx)];
            }
        }
        let dst = vec2<i32>(wid.xy) * 2 + local;
        if (all(dst < vec2<i32>(textureDimensions(mip2_out)))) {
            textureStore(mip2_out, dst, vec4<f32>(acc * 0.25, 0.0, 0.0));
        }
    }
}
