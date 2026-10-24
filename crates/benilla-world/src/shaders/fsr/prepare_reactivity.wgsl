// FSR 3.1.5 pass 4 — prepare reactivity
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_fsr3upscaler_prepare_reactivity.h`).
//
// Gathers everything that argues for or against trusting a pixel's history into
// the four channels of one render-resolution texture, plus this frame's thin
// feature locks. The accumulation reads only this; it does no analysis of its
// own.
//
// * **Disocclusion.** Was this surface visible last frame? A pixel that has just
//   emerged from behind something — the far side of a moving object, the wall a
//   falling body has slid off — has a perfectly valid motion vector pointing at a
//   perfectly valid patch of history that belongs to a completely different
//   surface. Accepting it is what produces trails. The test compares this pixel's
//   depth against `prepare_inputs.wgsl`'s reconstruction of the previous
//   depth buffer, at the place the history is about to be read from.
//
// * **Reactivity.** Pixels whose *colour* does not follow their own motion
//   vector, so history along that vector is the wrong history even though the
//   geometry is right. Two sources: the transparency-and-composition mask
//   (`motion.wgsl`'s `w`), and motion divergence, which catches the case where
//   the vector points into a region that is itself moving somewhere else
//   entirely. The generated reactive mask (`motion.wgsl`'s `z`) joins the
//   shading change, as the reference's `DilateReactiveMasks` does.
//
// * **Shading change**, sampled back from `shading_change.wgsl`.
//
// * **Accumulation.** The one piece of state that survives between frames here:
//   how converged this pixel is, on a 0..1 scale, advanced by a third each frame
//   and knocked back down by the two tests above. This is FSR 3.1's replacement
//   for the 2.x lineage's raw "number of accumulated frames" — a bounded,
//   reprojected quantity rather than a counter, so a pixel recovers from a
//   disocclusion on a fixed schedule instead of at a rate that depends on how
//   long it had been converged.
//
// The pass also detects the thin features the rectification box would otherwise
// erase — a rope against the floor, a rigid body's silhouette edge, the rim of a
// splash. Such a feature is present in some jittered frames and absent in others,
// so on the frames it is missing the box sees a neighbourhood without it and
// clips away the accumulated evidence that it exists; the result is a thin line
// that crawls no matter how long the history gets. A lock marks the pixel and
// tells the accumulation to hold its history through those frames. This pass only
// *creates* locks; `accumulate.wgsl` ages and breaks them.

@group(0) @binding(1) var dilated_mv_tex: texture_2d<f32>;
@group(0) @binding(2) var depths_tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> prev_depth: array<u32>;
// The generated reactive mask rides in z, the transparency-and-composition mask in w
// (`motion.wgsl`).
@group(0) @binding(4) var mv_tex: texture_2d<f32>;
@group(0) @binding(5) var shading_change_tex: texture_2d<f32>;
@group(0) @binding(6) var luma_tex: texture_2d<f32>;
@group(0) @binding(7) var prev_accumulation_tex: texture_2d<f32>;
@group(0) @binding(8) var linear_sampler: sampler;
@group(0) @binding(9) var masks_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(10) var accumulation_out: texture_storage_2d<FSR3_FMT_R8, write>;
@group(0) @binding(11) var new_locks_out: texture_storage_2d<FSR3_FMT_R8, write>;

/// How much depth separation, per unit of distance and half-viewport, counts as
/// "in front of" rather than "the same surface" (`Ksep`).
const K_SEP: f32 = 1.37e-05;

/// Reads the scattered previous-frame depth: the depth's own bits (`prepare_inputs.wgsl`), so a
/// word nothing landed on reads 0, the reverse-Z far plane.
fn reconstructed_prev_depth(p: vec2<i32>, size: vec2<i32>) -> f32 {
    return bitcast<f32>(prev_depth[u32(p.y * size.x + p.x)]);
}

/// A texture load that reads zero outside the image, matching the HLSL the
/// reference's motion-divergence test relies on at the screen border.
fn load_or_zero(tex: texture_2d<f32>, p: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
    if (!is_on_screen(p, size)) {
        return vec4<f32>(0.0);
    }
    return textureLoad(tex, p, 0);
}

fn compute_disocclusions(
    uv: vec2<f32>,
    mv_in: vec2<f32>,
    current_depth_vs: f32,
    size: vec2<i32>,
) -> f32 {
    let nearest_meters = min(current_depth_vs * params.view_space_to_meters, FSR3_FP16_MAX);
    let threshold = mix(0.25, 0.75, saturate(nearest_meters / 100.0));
    // Sub-pixel motion is indistinguishable from none; the same discard the
    // scatter applied, so the two agree on where to look.
    let mv = mv_in * f32(get_4k_velocity(mv_in) > threshold);

    let info = bilinear_sampling_data(uv + mv, params.render_size);

    var disocclusion = 0.0;
    var weight_sum = 0.0;
    var potential = true;

    for (var i = 0; i < 4; i = i + 1) {
        if (!potential) {
            break;
        }
        let sample_pos = clamp_load(info.base_pos, bilinear_offset(i), size);
        let weight = info.weights[i];
        if (weight <= RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD) {
            continue;
        }

        let prev_vs = view_space_depth(reconstructed_prev_depth(sample_pos, size));
        let depth_difference = current_depth_vs - prev_vs;

        // Nothing reprojected in front of this surface — one tap that agrees is
        // enough to call the whole footprint visible.
        potential = potential && (depth_difference > FSR3_FP32_MIN);
        if (!potential) {
            continue;
        }

        // The tolerance scales with distance and with the viewport, so one
        // constant covers the near geometry and the far wall alike.
        let half_viewport_width = length(params.render_size * 0.5);
        let depth_threshold = max(current_depth_vs, prev_vs);
        let required_separation = K_SEP * half_viewport_width * depth_threshold;

        disocclusion += saturate(required_separation / depth_difference) * weight;
        weight_sum += weight;
    }

    if (potential && weight_sum > 0.0) {
        return saturate(1.0 - disocclusion / weight_sum);
    }
    return 0.0;
}

/// How much this pixel's motion vector disagrees with the motion of whatever it
/// points at. A vector landing in a region that is itself moving elsewhere is
/// describing a surface that will not be there — the classic case being a fast
/// object crossing a background, where the background's own history is about to
/// be handed to the object's pixels.
fn compute_motion_divergence(uv: vec2<f32>, mv: vec2<f32>, depth: f32, size: vec2<i32>) -> f32 {
    let velocity_4k = get_4k_velocity(mv);
    if (velocity_4k <= 0.0) {
        return 0.0;
    }

    let reprojected_pos = vec2<i32>((uv + mv) * params.render_size);
    let reprojected_depth = load_or_zero(depths_tex, reprojected_pos, size).x;
    let reprojected_mv = load_or_zero(dilated_mv_tex, reprojected_pos, size).xy;
    let reprojected_velocity = get_4k_velocity(reprojected_mv);

    let nucleus_meters = view_space_depth_meters(reprojected_depth);
    let current_meters = view_space_depth_meters(depth);

    // Only trust the disagreement when the two are at a similar distance —
    // otherwise it is just parallax — and only once the motion is large enough
    // for a mismatch to mean anything.
    let distance_factor = min_divided_by_max(nucleus_meters, current_meters, 0.0);
    let velocity_factor = saturate(velocity_4k / 10.0);

    return (1.0 - saturate(reprojected_velocity / velocity_4k)) * distance_factor * velocity_factor;
}

/// Is this pixel a luminance ridge — a feature the sampling grid is too coarse to
/// hold on to?
///
/// The test is structural rather than a simple extremum: the eight neighbours are
/// split into those close in luminance to the centre and those that are not, and
/// the pixel qualifies only if it is outside the range of the dissimilar ones
/// *and* no complete 2x2 quadrant of similar neighbours surrounds it. That last
/// rejection is what separates a thin feature from the interior of an ordinary
/// edge: an edge always has one quadrant lying entirely on its own side.
fn compute_thin_feature_confidence(p: vec2<i32>, size: vec2<i32>) -> f32 {
    // 1 2 3
    // 4 0 5
    // 6 7 8
    var offsets: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
        vec2<i32>(0, 0),
        vec2<i32>(-1, -1),
        vec2<i32>(0, -1),
        vec2<i32>(1, -1),
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(-1, 1),
        vec2<i32>(0, 1),
        vec2<i32>(1, 1),
    );

    var samples: array<f32, 9>;
    var luma_min = FSR3_FP32_MAX;
    var luma_max = FSR3_FP32_MIN;
    for (var i = 0; i < 9; i = i + 1) {
        let q = clamp_load(p, offsets[i], size);
        let v = textureLoad(luma_tex, q, 0).x * params.exposure;
        samples[i] = v;
        luma_min = min(luma_min, v);
        luma_max = max(luma_max, v);
    }

    let spread = luma_max - luma_min;
    if (spread <= 0.0) {
        return 0.0;
    }

    let threshold = 0.9;
    var dissimilar_min = FSR3_FP32_MAX;
    var dissimilar_max = 0.0;
    // Bit 0 flags the nucleus as similar to itself.
    var pattern_mask = 1u;

    for (var i = 1; i < 9; i = i + 1) {
        let difference = abs(samples[i] - samples[0]) / spread;
        if (difference < threshold) {
            pattern_mask |= 1u << u32(i);
        } else {
            dissimilar_min = min(dissimilar_min, samples[i]);
            dissimilar_max = max(dissimilar_max, samples[i]);
        }
    }

    let is_ridge = samples[0] > dissimilar_max || samples[0] < dissimilar_min;
    if (!is_ridge) {
        return 0.0;
    }

    // Upper left, upper right, lower left, lower right — each with the nucleus.
    var rejection_masks: array<u32, 4> = array<u32, 4>(
        (1u << 1u) | (1u << 2u) | (1u << 4u) | 1u,
        (1u << 2u) | (1u << 3u) | (1u << 5u) | 1u,
        (1u << 4u) | (1u << 6u) | (1u << 7u) | 1u,
        (1u << 5u) | (1u << 7u) | (1u << 8u) | 1u,
    );
    for (var i = 0; i < 4; i = i + 1) {
        if ((pattern_mask & rejection_masks[i]) == rejection_masks[i]) {
            return 0.0;
        }
    }

    return 1.0 - luma_min / luma_max;
}

/// Advances this pixel's convergence, and hands the *pre*-advance value to the
/// rest of the frame.
///
/// Reprojecting the previous value and adding a third per frame is what gives the
/// recovery its fixed schedule: a fully disoccluded pixel is pushed to -1/3, so
/// it spends one frame at zero weight and reaches full convergence four frames
/// later regardless of what it was before.
fn update_accumulation(
    p: vec2<i32>,
    uv: vec2<f32>,
    mv: vec2<f32>,
    disocclusion: f32,
    shading_change: f32,
) -> f32 {
    let reprojected_uv = uv + mv;
    var accumulation = 0.0;
    if (is_uv_inside(reprojected_uv)) {
        let uv_hw = clamp_uv(reprojected_uv, params.render_size);
        accumulation = saturate(
            textureSampleLevel(prev_accumulation_tex, linear_sampler, uv_hw, 0.0).x
        );
    }

    accumulation = mix(accumulation, 0.0, shading_change);
    accumulation = mix(
        accumulation,
        min(params.min_disocclusion_accumulation, accumulation),
        disocclusion,
    );
    // Anything under one part in a hundred is not convergence worth claiming.
    accumulation *= f32(round(accumulation * 100.0) > 1.0);

    // Stored normalised for the next frame to reproject; returned raw, since the
    // negative range is exactly what encodes the recovery delay.
    let to_store = saturate(accumulation + params.accumulation_added_per_frame);
    textureStore(accumulation_out, p, vec4<f32>(to_store, 0.0, 0.0, 0.0));

    return accumulation;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;
    let mv = textureLoad(dilated_mv_tex, p, 0).xy;
    let dilated_depth = textureLoad(depths_tex, p, 0).x;

    let disocclusion = compute_disocclusions(uv, mv, view_space_depth(dilated_depth), size);

    // `DilateReactiveMasks`: the reactive mask's 3x3 maximum, maxed with the detector. The
    // detector's grid is half resolution and carries the jitter, which is removed here.
    let shading_change_uv = clamp_uv(
        uv - params.jitter * params.inv_render_size,
        vec2<f32>(shading_change_size_i()),
    );
    var dilated_reactive = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let q = clamp_load(p, vec2<i32>(x, y), size);
            dilated_reactive = max(dilated_reactive, textureLoad(mv_tex, q, 0).z);
        }
    }
    let shading_change = max(
        dilated_reactive,
        saturate(textureSampleLevel(shading_change_tex, linear_sampler, shading_change_uv, 0.0).x),
    );

    let motion_divergence = compute_motion_divergence(uv, mv, dilated_depth, size);
    // The reference dilates the application's transparency-and-composition mask
    // with a bilinear fetch at the pixel centre, which on a render-resolution
    // mask is the texel itself.
    let transparency = textureLoad(mv_tex, p, 0).w;
    let reactiveness = max(motion_divergence, transparency);

    let accumulation = update_accumulation(p, uv, mv, disocclusion, shading_change);

    var masks: vec4<f32>;
    masks[REACTIVE] = reactiveness;
    masks[DISOCCLUSION] = disocclusion;
    masks[SHADING_CHANGE] = shading_change;
    masks[ACCUMULATION] = accumulation;
    textureStore(masks_out, p, masks);

    let lock_strength = compute_thin_feature_confidence(p, size);
    if (lock_strength > 1.0 / 100.0) {
        // The display pixel this render pixel's jittered sample fell in
        // (`ComputeHrPosFromLrPos`): `jitter` is the sample offset, so subtracting
        // it recovers the unjittered position the sample represents.
        let src_jittered = vec2<f32>(p) + vec2<f32>(0.5) - params.jitter;
        let dst = vec2<i32>(floor(src_jittered * params.inv_render_size * params.display_size));
        if (is_on_screen(dst, display_size_i())) {
            textureStore(new_locks_out, dst, vec4<f32>(lock_strength, 0.0, 0.0, 0.0));
        }
    }
}
