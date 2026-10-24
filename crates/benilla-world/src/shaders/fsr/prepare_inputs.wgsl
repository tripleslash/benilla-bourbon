// FSR 3.1.5 pass 1 — prepare inputs (`ffx_fsr3upscaler_prepare_inputs.h`), inverted depth.
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
//
// Everything the rest of the chain reads about *geometry* is derived here, at
// render resolution, in one sweep of a 3x3 neighbourhood:
//
// * **Depth extents.** The nearest sample of the 3x3 and (in the reference's
//   peculiar sense, reproduced below) a farthest one. The nearest drives
//   dilation; the farthest becomes a coarse per-region distance that the
//   accumulation uses to widen its rectification box on close-up geometry.
//
// * **Dilate.** Each pixel adopts the motion vector of the *nearest* sample in
//   its neighbourhood. Silhouettes are the whole reason: a pixel sitting on the
//   edge between a moving object and the background gets its colour from both,
//   and following the background's vector there drags the object's edge into a
//   smear. Taking the nearest surface's vector biases the edge toward the
//   foreground, which is the one that will still be there next frame.
//
// * **Reconstruct.** Every pixel projects its own depth *backwards* along that
//   motion vector and scatters it into the pixels it came from. What that builds
//   is an estimate of the previous frame's depth buffer — not from history, which
//   would have to be stored and would be a frame stale, but from this frame's
//   geometry running the clock backwards. `prepare_reactivity.wgsl` then
//   asks: at the place my history is about to be read from, was there something
//   *in front* of where I would have been? If so I was hidden, and that history
//   is somebody else's surface.
//
// * **Luma.** The frame's luminance, which the shading-change pyramid, the thin
//   feature detector and the instability test all consume.
//
// The scatter is a race — many pixels can land on the same previous pixel — and the winner has to
// be the nearest, so it needs an atomic. WebGPU has no atomics on storage *textures*, so the target
// is a plain storage buffer, one u32 per render pixel, addressed by hand.
//
// Depth is reverse-Z (`FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH`): nearer is larger, and non-negative
// IEEE bit patterns order like the unsigned integers, so the nearest is a plain `atomicMax` of the
// depth's bits — the SDK's `InterlockedMax` for inverted depth. Zero, what `clear_buffer` leaves, is
// the far plane, which is exactly the reference's clear.

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var mv_tex: texture_2d<f32>;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var dilated_mv_out: texture_storage_2d<FSR3_FMT_RG16, write>;
// x = dilated (nearest) device depth, y = farthest depth in metres.
@group(0) @binding(5) var depths_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(6) var luma_out: texture_storage_2d<FSR3_FMT_R16, write>;
@group(0) @binding(7) var<storage, read_write> prev_depth: array<atomic<u32>>;

/// The 3x3 in the reference's order. `FindDepthExtents` only updates its
/// "farthest" when it finds a new nearest, so which sample is visited when is
/// load-bearing and this order is part of the algorithm.
const SAMPLE_OFFSETS: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(0, 1),
    vec2<i32>(0, -1),
    vec2<i32>(-1, 0),
    vec2<i32>(-1, 1),
    vec2<i32>(1, 1),
    vec2<i32>(-1, -1),
    vec2<i32>(1, -1),
);

struct DepthExtents {
    nearest: f32,
    nearest_coord: vec2<i32>,
    farthest: f32,
};

fn find_depth_extents(p: vec2<i32>, size: vec2<i32>) -> DepthExtents {
    var e: DepthExtents;
    // Reverse-Z: nearer is larger.
    let center = textureLoad(depth_tex, clamp_load(p, vec2<i32>(0), size), 0);
    e.nearest = center;
    e.farthest = center;
    e.nearest_coord = p;

    for (var i = 1; i < 9; i = i + 1) {
        let q = p + SAMPLE_OFFSETS[i];
        if (!is_on_screen(q, size)) {
            continue;
        }
        let d = textureLoad(depth_tex, q, 0);
        if (d > e.nearest) {
            e.farthest = min(e.farthest, d);
            e.nearest_coord = q;
            e.nearest = d;
        }
    }
    return e;
}

/// Scatters this pixel's depth into the four texels a bilinear fetch of its
/// history would touch, so the disocclusion test sees a filled surface rather
/// than the holes a nearest-texel scatter would leave.
fn reconstruct_prev_depth(p: vec2<i32>, size: vec2<i32>, depth: f32, mv_in: vec2<f32>) {
    let nearest_meters = min(view_space_depth_meters(depth), FSR3_FP16_MAX);
    // Sub-pixel motion is indistinguishable from none, and reprojecting on it
    // only spreads a surface over texels it never left.
    let threshold = mix(0.25, 0.75, saturate(nearest_meters / 100.0));
    let mv = mv_in * f32(get_4k_velocity(mv_in) > threshold);

    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;
    let info = bilinear_sampling_data(uv + mv, params.render_size);

    for (var i = 0; i < 4; i = i + 1) {
        if (info.weights[i] <= RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD) {
            continue;
        }
        let dst = info.base_pos + bilinear_offset(i);
        if (!is_on_screen(dst, size)) {
            continue;
        }
        let index = u32(dst.y * size.x + dst.x);
        atomicMax(&prev_depth[index], bitcast<u32>(depth));
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    let extents = find_depth_extents(p, size);
    // The engine supplies render-resolution motion vectors
    // (`FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS`), so the dilated
    // vector is simply the one at the nearest sample.
    let dilated_mv = textureLoad(mv_tex, extents.nearest_coord, 0);

    reconstruct_prev_depth(p, size, extents.nearest, dilated_mv.xy);

    textureStore(dilated_mv_out, p, dilated_mv);
    textureStore(
        depths_out,
        p,
        vec4<f32>(
            extents.nearest,
            min(view_space_depth_meters(extents.farthest), FSR3_FP16_MAX),
            0.0,
            0.0,
        ),
    );

    let rgb = max(vec3<f32>(0.0), textureLoad(color_tex, p, 0).rgb);
    textureStore(luma_out, p, vec4<f32>(rgb_to_luma(rgb), 0.0, 0.0, 0.0));
}
