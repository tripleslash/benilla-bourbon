// FSR 3.1.5 pass 5 — luminance instability
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_fsr3upscaler_luma_instability.h`).
//
// Some pixels never settle: a specular highlight sliding over animated water, a
// sub-pixel feature the jitter alternately hits and misses on a surface that is
// itself changing. Accumulating them does not converge on an answer, and the
// rectification box cannot catch it because every value involved is locally
// valid.
//
// The signature this looks for is oscillation: the current luminance differs from
// last frame's, but *matches* one of the three frames before it more closely than
// last frame did, on the same side. A signal that keeps returning to values it
// has already visited is not converging on anything, and its history is an
// average of a flicker rather than a better estimate.
//
// Where that holds, the pixel is marked, and the accumulation lets its history be
// clipped rather than protecting it — the opposite of what a lock does. The
// factor is scaled down by velocity (a fast-moving pixel's luminance is expected
// to change) and by the other three masks, since a disoccluded or reactive or
// shading-changed pixel already has its own reason to distrust history and does
// not need this one stacked on top.
//
// The test only runs on pixels that are already well converged (`accumulation >
// 0.9`). Anything less is still filling in, and its luminance is expected to move.

@group(0) @binding(1) var dilated_mv_tex: texture_2d<f32>;
@group(0) @binding(2) var masks_tex: texture_2d<f32>;
@group(0) @binding(3) var luma_tex: texture_2d<f32>;
@group(0) @binding(4) var prev_luma_history_tex: texture_2d<f32>;
@group(0) @binding(5) var linear_sampler: sampler;
@group(0) @binding(6) var luma_history_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var luma_instability_out: texture_storage_2d<FSR3_FMT_R16, write>;

struct LumaInstabilityData {
    history: vec4<f32>,
    factor: f32,
};

fn compute_luma_instability_factor(
    history_in: vec4<f32>,
    current_luma: f32,
) -> LumaInstabilityData {
    // The four slots are frames N-1 .. N-4.
    var history = history_in;
    var instability = 0.0;

    let diff0 = current_luma - history[0];
    let similarity0 = min_divided_by_max(current_luma, history[0], 1.0);
    var max_similarity = similarity0;

    if (similarity0 < 1.0) {
        for (var i = 1; i < 4; i = i + 1) {
            let diff1 = current_luma - history[i];
            let similarity1 = min_divided_by_max(current_luma, history[i], 0.0);
            // Same side of the current value: an oscillation returns to where it
            // has been, it does not merely pass through.
            if (sign(diff0) == sign(diff1)) {
                max_similarity = max(max_similarity, similarity1);
            }
        }
        instability = f32(max_similarity > similarity0);
    }

    history = vec4<f32>(current_luma, history[0], history[1], history[2]);

    var data: LumaInstabilityData;
    data.history = history;
    // Needs a full four frames before the pattern means anything; the oldest slot
    // being zero says the history is not that long yet.
    data.factor = instability * f32(history[3] != 0.0);
    return data;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    var data: LumaInstabilityData;
    data.history = vec4<f32>(0.0);
    data.factor = 0.0;

    let mv = textureLoad(dilated_mv_tex, p, 0).xy;
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;
    // Each frame's samples sit on its own jittered grid, so the two ends of the
    // reprojection are expressed against their own frame's jitter.
    let uv_current_jittered = uv + params.jitter * params.inv_render_size;
    let uv_prev_jittered = uv + params.prev_jitter * params.inv_render_size;
    let reprojected_uv = uv_prev_jittered + mv;

    if (is_uv_inside(reprojected_uv)) {
        let uv_masks = clamp_uv(uv_current_jittered, params.render_size);
        let masks = textureSampleLevel(masks_tex, linear_sampler, uv_masks, 0.0);
        let reactive = saturate(masks[REACTIVE]);
        let disocclusion = saturate(masks[DISOCCLUSION]);
        let shading_change = saturate(masks[SHADING_CHANGE]);
        let accumulation = saturate(masks[ACCUMULATION]);

        if (accumulation > 0.9) {
            // The history is stored exposure-free (`LumaInstability`), both ends scaled here.
            let current_luma =
                textureSampleLevel(luma_tex, linear_sampler, uv_masks, 0.0).x * params.exposure;
            let history = textureSampleLevel(
                prev_luma_history_tex,
                linear_sampler,
                clamp_uv(reprojected_uv, params.render_size),
                0.0,
            ) * params.exposure;

            data = compute_luma_instability_factor(history, current_luma);
            data.history /= params.exposure;

            let velocity_weight = 1.0 - saturate(get_4k_velocity(mv) / 20.0);
            data.factor *= velocity_weight
                * (1.0 - disocclusion)
                * (1.0 - reactive)
                * (1.0 - shading_change);
        }
    }

    // A pixel that failed either gate stores a zeroed history: it has to
    // re-establish four consecutive frames before it can be judged again.
    textureStore(luma_history_out, p, data.history);
    textureStore(luma_instability_out, p, vec4<f32>(data.factor, 0.0, 0.0, 0.0));
}
