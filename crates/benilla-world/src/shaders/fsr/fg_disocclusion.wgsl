// FSR 3.1.5 frame generation, pass 4 — the disocclusion mask
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_frameinterpolation_disocclusion_mask.h`).
//
// The generated frame is assembled from two others, and every one of its pixels
// has to be fetched from at least one of them. This pass decides which: for each
// cell of the in-between image, was the surface it shows actually *visible* in
// frame N-1, and was it visible in frame N?
//
// Both questions are asked the same way. Follow the cell's motion vector back (or
// forward) to where the surface would have been, and compare the depth this pass
// estimated for the in-between moment against the depth estimated for that frame.
// If something was measurably nearer there, the surface was behind it: hidden,
// and that frame has nothing to offer. "Measurably" is the point — a surface
// seen at a glancing angle spans a large depth range within one pixel, so the
// separation that counts as an occlusion has to scale with distance and with the
// field of view, which is what `Ksep`/`Kfov` below are for. The idea comes from
// "Minimum Triangle Separation for Correct Z-Buffer Occlusion".
//
// The two answers, one per channel, are then hard-binarised: partial visibility
// is not a useful thing to blend, and the interpolation wants a decision.
//
// The two depth estimates come from opposite directions and neither is a
// rendered buffer:
//
// * `interpolated_depth` — this frame's geometry walked half a motion vector
//   forward (`fg_reconstruct_depth.wgsl`).
// * `prev_depth` — the *upscaler's* estimate of the previous frame's depth, this
//   frame's geometry walked a whole vector backwards
//   (`prepare_inputs.wgsl`). One of the three resources FSR 3.1 shares
//   between upscaling and frame generation rather than deriving twice.

@group(0) @binding(1) var depths_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> interpolated_depth: array<u32>;
@group(0) @binding(3) var<storage, read> prev_depth: array<u32>;
@group(0) @binding(4) var<storage, read> field_x: array<u32>;
@group(0) @binding(5) var<storage, read> field_y: array<u32>;
@group(0) @binding(6) var mv_pyramid: texture_2d<f32>;
@group(0) @binding(7) var disocclusion_out: texture_storage_2d<FSR3_FMT_RG8, write>;

/// How much depth separation, per unit of distance and half-viewport, counts as
/// "in front of" rather than "the same surface" (`Ksep`).
const K_SEP: f32 = 1.37e-05;

/// Decodes one of the two scattered depth buffers: the depth's own bits (reverse-Z, so the
/// nearest is the largest and the scatter's `atomicMax` keeps it), and a zero word — the cleared
/// state — reads as 0, the far plane, which is the reference's inverted-depth clear.
fn decode_scattered_depth(packed: u32) -> f32 {
    return bitcast<f32>(packed);
}

/// `LoadInpaintedGameFieldMv`: the field's own vector where it has one, and the
/// pyramid's best guess where it does not.
fn load_inpainted_game_field_mv(uv: vec2<f32>) -> VectorFieldEntry {
    let size = render_size_i();
    let p = clamp(vec2<i32>(uv * params.render_size), vec2<i32>(0), size - vec2<i32>(1));
    let i = u32(p.y * size.x + p.x);

    var e = unpack_vector_field_entries(vec2<u32>(field_x[i], field_y[i]));
    if (!e.valid) {
        e = inpaint_game_field_mv(mv_pyramid, uv);
    }
    return finish_vector_field_entry(e, uv);
}

/// `LoadEstimatedDepth`. Index 0 is the previous frame's reconstructed depth,
/// index 1 the current frame's dilated depth.
fn load_estimated_depth(estimated_index: u32, p: vec2<i32>, size: vec2<i32>) -> f32 {
    if (estimated_index == 0u) {
        return decode_scattered_depth(prev_depth[u32(p.y * size.x + p.x)]);
    }
    return textureLoad(depths_tex, p, 0).x;
}

/// `ComputeDepthClip`: the fraction of the bilinear footprint at `uv_sample`
/// whose depth sits far enough in front of `current_depth` to have hidden it.
fn compute_depth_clip(estimated_index: u32, uv_sample: vec2<f32>, current_depth: f32) -> f32 {
    let size = render_size_i();
    let current_depth_vs = view_space_depth(current_depth);
    let info = bilinear_sampling_data(uv_sample, params.render_size);

    var occluded = 0.0;
    var weight_sum = 0.0;

    for (var i = 0; i < 4; i = i + 1) {
        let sample_pos = info.base_pos + bilinear_offset(i);
        if (!is_on_screen(sample_pos, size)) {
            continue;
        }
        let weight = info.weights[i];
        if (weight <= RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD) {
            continue;
        }

        let other_depth = load_estimated_depth(estimated_index, sample_pos, size);
        let other_depth_vs = view_space_depth(other_depth);
        let depth_diff = current_depth_vs - other_depth_vs;

        // Only something *nearer* than this surface can have hidden it.
        if (depth_diff <= 0.0) {
            continue;
        }

        // The separation test is evaluated on the plane of whichever of the two
        // is further away, so a shallow grazing surface is not mistaken for an
        // occluder of itself. Reverse-Z (`FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH`), so that
        // is the smaller device depth.
        let plane_depth = min(other_depth, current_depth);
        let center = view_space_position(vec2<i32>(params.render_size * 0.5), size, plane_depth);
        let corner = view_space_position(vec2<i32>(0, 0), size, plane_depth);

        let viewport_extent = length(params.render_size);
        let depth_threshold = min(current_depth_vs, other_depth_vs);
        // How much wider the frustum is at its corner than along its axis: the
        // tolerance has to grow with the field of view.
        let k_fov = length(corner) / length(center);
        let required_separation = K_SEP * k_fov * viewport_extent * depth_threshold;

        // Below the required separation the two are the same surface seen twice,
        // and this tap votes "visible"; above it, "occluded".
        occluded += f32((required_separation / depth_diff) >= 1.0) * weight;
        weight_sum += weight;
    }

    // No tap found anything in front — nothing to clip against.
    if (weight_sum <= 0.0) {
        return 0.0;
    }
    return saturate(1.0 - occluded / weight_sum);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    let depth = decode_scattered_depth(interpolated_depth[u32(p.y * size.x + p.x)]);
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;

    let game_mv = load_inpainted_game_field_mv(uv);

    // `+mv` walks towards frame N-1, `-mv` towards frame N — the field stores
    // half-vectors, so each is half the interval away.
    let to_previous = 1.0 - compute_depth_clip(0u, uv + game_mv.motion_vector, depth);
    let to_current = 1.0 - compute_depth_clip(1u, uv - game_mv.motion_vector, depth);

    // Binarise. A partial answer here would show up as a partial cross-fade
    // between two frames that disagree, which reads as a ghost rather than as a
    // softer edge.
    var mask = vec2<f32>(
        f32(to_previous >= FG_EPSILON),
        f32(to_current >= FG_EPSILON),
    );

    // A vector that leaves the image entirely has no surface to be occluded by;
    // the depth test above would have found nothing there and called it hidden.
    // Treat "off-screen" as visible instead and let the interpolation's own
    // out-of-bounds handling take it from there.
    let src_motion_vector = game_mv.motion_vector * 2.0;
    let sample_previous = vec2<i32>((uv + src_motion_vector) * params.render_size);
    mask.x = saturate(mask.x + f32(!is_on_screen(sample_previous, size)));

    let sample_current = vec2<i32>((uv - src_motion_vector) * params.render_size);
    mask.y = saturate(mask.y + f32(!is_on_screen(sample_current, size)));

    textureStore(disocclusion_out, p, vec4<f32>(mask, 0.0, 0.0));
}
