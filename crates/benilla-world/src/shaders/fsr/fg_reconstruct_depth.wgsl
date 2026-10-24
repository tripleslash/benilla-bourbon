// FSR 3.1.5 frame generation, pass 1 — the interpolated frame's depth buffer
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_frameinterpolation_reconstruct_previous_depth.h`).
//
// The frame being generated sits halfway between the two rendered ones, and it
// was never rasterized, so it has no depth buffer of its own. This pass builds
// one the same way the upscaler builds its estimate of the *previous* frame's:
// every render pixel walks its own depth along its own motion vector and drops it
// where it lands — except that here the walk stops halfway, at `mv * 0.5`, which
// is where the surface is at the instant the generated frame depicts.
//
// What that buys is the only depth-ordered view of the in-between moment there
// is, and `fg_disocclusion.wgsl` needs exactly that: to ask, for each pixel
// of a frame that does not exist, whether the surface it shows was visible in the
// frame before and in the frame after.
//
// The scatter is a race — many pixels can reach the same destination, and the one
// in front has to win — so it needs an atomic. WebGPU has no atomics on storage
// textures, so the target is a plain storage buffer indexed by hand, in the same
// encoding `prepare_inputs.wgsl` uses: depth is reverse-Z, nearer is larger, and
// non-negative IEEE bit patterns order like unsigned integers, so the nearest is
// a plain `atomicMax` (the SDK's inverted-depth `InterlockedMax`). A cleared
// (zero) word is the far plane, the reference's inverted-depth clear.

@group(0) @binding(1) var dilated_mv_tex: texture_2d<f32>;
@group(0) @binding(2) var depths_tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> interpolated_depth: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    // Both inputs are the upscaler's, already dilated onto the nearest surface in
    // each 3x3 (see `prepare_inputs.wgsl`) — the resources FSR 3.1 shares
    // between the two effects rather than deriving twice.
    let mv = textureLoad(dilated_mv_tex, p, 0).xy;
    let depth = textureLoad(depths_tex, p, 0).x;

    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;
    // Half the motion: the generated frame is the midpoint of the interval.
    let info = bilinear_sampling_data(uv + mv * 0.5, params.render_size);

    for (var i = 0; i < 4; i = i + 1) {
        if (info.weights[i] <= RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD) {
            continue;
        }
        let dst = info.base_pos + bilinear_offset(i);
        if (!is_on_screen(dst, size)) {
            continue;
        }
        let index = u32(dst.y * size.x + dst.x);
        atomicMax(&interpolated_depth[index], bitcast<u32>(depth));
    }
}
