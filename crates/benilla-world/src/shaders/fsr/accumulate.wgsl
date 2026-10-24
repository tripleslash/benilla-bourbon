// FSR 3.1.5 pass 6 — accumulation and upsampling
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_fsr3upscaler_accumulate.h`, `_upsample.h`, `_reproject.h`).
//
// This is the pass that makes the technique an upscaler rather than an
// anti-aliaser bolted to one. It runs on the **display** grid, and for each
// output pixel it does two things and blends them:
//
// * **Upsample.** The render-resolution samples around this output pixel are
//   resolved with a Lanczos-2 reconstruction kernel, weighted by where each
//   sample actually landed — including this frame's sub-pixel jitter. An output
//   pixel whose centre a jittered sample fell exactly on is reconstructed from
//   that sample; one that falls between them is interpolated. This is why the
//   jitter buys *resolution* here and not merely edge convergence: the samples
//   are treated as an irregular sampling of the display grid, not as repeated
//   measurements of the render grid.
//
// * **Accumulate.** The previous output frame is reprojected along the dilated
//   motion vectors and blended in. The blend weight is not a frame counter: the
//   history carries a weight of its own and the current frame contributes the
//   Lanczos kernel's own total, so a pixel whose samples land far from its centre
//   contributes less than one that lands on it. `prepare_reactivity.wgsl`'s
//   accumulation channel sets the ceiling.
//
// The three things that argue against history — disocclusion, reactivity,
// shading change — do not gate the blend directly. They enter through the
// rectification box: the box tightens toward the current neighbourhood's actual
// spread when any of them fires (`rectify_history`), and loosens to three
// standard deviations when none does. Locks and the instability factor then
// decide how much of that clipping is applied. That indirection is the 3.1
// design: rejecting history outright makes an output that tracks the input, and
// the input is what was noisy in the first place.
//
// Colour statistics are in YCoCg throughout — one luma axis and two chroma axes,
// so the box tracks brightness separately from hue and clips far less
// aggressively than an RGB box on the same neighbourhood.

@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var dilated_mv_tex: texture_2d<f32>;
@group(0) @binding(3) var masks_tex: texture_2d<f32>;
@group(0) @binding(4) var luma_instability_tex: texture_2d<f32>;
@group(0) @binding(5) var farthest_mip1_tex: texture_2d<f32>;
@group(0) @binding(6) var new_locks_tex: texture_2d<f32>;
@group(0) @binding(7) var history_tex: texture_2d<f32>;
@group(0) @binding(8) var linear_sampler: sampler;
// rgb = the accumulated image, a = the lock this pixel carries into next frame.
@group(0) @binding(9) var history_out: texture_storage_2d<rgba16float, write>;

/// Everything about this output pixel that the steps below share, gathered once
/// (`AccumulationPassCommonParams`).
struct CommonParams {
    px_hr: vec2<i32>,
    hr_uv: vec2<f32>,
    lr_uv_jittered: vec2<f32>,
    lr_uv_hw: vec2<f32>,
    mv: vec2<f32>,
    reprojected_hr_uv: vec2<f32>,
    velocity_4k: f32,
    disocclusion: f32,
    reactive: f32,
    shading_change: f32,
    accumulation: f32,
    luma_instability: f32,
    farthest_depth_meters: f32,
    is_existing_sample: bool,
    is_new_sample: bool,
};

/// The running state the steps hand between each other (`AccumulationPassData`).
struct PassData {
    box: RectificationBox,
    upsampled_color: vec3<f32>,
    upsampled_weight: f32,
    history_color: vec3<f32>,
    history_weight: f32,
    lock: f32,
    lock_contribution: f32,
};

// --- history resample ---------------------------------------------------------

/// One axis of the bicubic fetch: four taps weighted by a true windowed Lanczos-2
/// kernel, renormalised.
fn lanczos2_row(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, t: f32) -> vec4<f32> {
    let w0 = lanczos2(-1.0 - t);
    let w1 = lanczos2(-t);
    let w2 = lanczos2(1.0 - t);
    let w3 = lanczos2(2.0 - t);
    return (w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3) * rcp_safe(w0 + w1 + w2 + w3);
}

/// Lanczos-2 history fetch, as a 4x4 of point loads.
///
/// A bilinear reprojection would resample the history every frame, and since the
/// history is itself the previous resample, that blur compounds without bound —
/// the reason plain TAA goes soft under sustained motion. The negative lobe
/// cancels most of it; the deringing clamp at the end bounds the overshoot it
/// costs, against the inner four taps only, since those are the ones the
/// reprojection actually landed between.
fn history_sample(uv: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
    var px = uv * size - vec2<f32>(0.5);
    let frac = fract(px);
    px = clamp(px, vec2<f32>(0.0), size - vec2<f32>(1.0));
    let base = vec2<i32>(floor(px));
    let last = vec2<i32>(size) - vec2<i32>(1);

    var taps: array<vec4<f32>, 16>;
    for (var y = 0; y < 4; y = y + 1) {
        for (var x = 0; x < 4; x = x + 1) {
            let q = clamp(base + vec2<i32>(x - 1, y - 1), vec2<i32>(0), last);
            taps[y * 4 + x] = textureLoad(history_tex, q, 0);
        }
    }

    let row0 = lanczos2_row(taps[0], taps[1], taps[2], taps[3], frac.x);
    let row1 = lanczos2_row(taps[4], taps[5], taps[6], taps[7], frac.x);
    let row2 = lanczos2_row(taps[8], taps[9], taps[10], taps[11], frac.x);
    let row3 = lanczos2_row(taps[12], taps[13], taps[14], taps[15], frac.x);
    var color = lanczos2_row(row0, row1, row2, row3, frac.y);

    let dering_min = min(min(taps[5], taps[6]), min(taps[9], taps[10]));
    let dering_max = max(max(taps[5], taps[6]), max(taps[9], taps[10]));
    return clamp(color, dering_min, dering_max);
}

// --- locks --------------------------------------------------------------------

/// Ages this pixel's lock, breaks it where the history it protects is no longer
/// worth protecting, and folds in whatever `prepare_reactivity.wgsl` found
/// this frame.
///
/// A lock only starts contributing past `LOCK_THRESHOLD`, so a feature has to be
/// detected on more than one frame before its history is held through a frame the
/// jitter misses it on. That hysteresis is what keeps a lock from forming on
/// noise.
fn update_lock_status(p: CommonParams, data_in: PassData) -> PassData {
    var data = data_in;

    data.lock *= f32(!p.is_new_sample);

    let lifetime_decrease_factor = max(saturate(p.shading_change), max(p.reactive, p.disocclusion));
    data.lock = max(0.0, data.lock - lifetime_decrease_factor * LOCK_MAX);

    data.lock_contribution = saturate(saturate(data.lock - LOCK_THRESHOLD) * (LOCK_MAX - LOCK_THRESHOLD));

    let new_lock_intensity = textureLoad(new_locks_tex, p.px_hr, 0).x * (1.0 - p.reactive);
    data.lock = max(0.0, min(data.lock + new_lock_intensity, LOCK_MAX));

    // Bleeds over the jitter sequence rather than over frames: a lock should
    // survive exactly as long as it takes the sequence to sample the feature
    // again.
    let lifetime_decrease = (0.1 / max(params.jitter_sequence_length, 1.0)) * (1.0 - lifetime_decrease_factor);
    data.lock = max(0.0, data.lock - lifetime_decrease);

    // Assuming the motion holds, this lock is about to leave the screen; letting
    // it live would pin it to the border.
    let estimated_uv_next_frame = p.hr_uv - p.mv;
    data.lock *= f32(is_uv_inside(estimated_uv_next_frame));

    return data;
}

// --- upsample -----------------------------------------------------------------

/// Widens the reconstruction kernel as confidence in the history grows.
///
/// A narrow kernel resolves a single frame's samples sharply but sees only the
/// three taps nearest the output pixel; a wide one reaches further and is
/// smoother. Where the history is trusted the sharpness comes from accumulation
/// instead, so the kernel can afford to be wide; where it is not — a
/// disocclusion, a shading change, a pixel still filling in — the current frame
/// has to carry the image on its own and the kernel tightens.
fn compute_kernel_bias(p: CommonParams, history_weight: f32) -> f32 {
    let kernel_bias_max = min(1.99, 1.0 / params.downscale_factor.x);
    let kernel_bias_min = max(1.0, (1.0 + kernel_bias_max) * 0.3);

    let kernel_bias_weight = min(
        1.0 - p.disocclusion * 0.5,
        min(1.0 - p.shading_change, saturate(history_weight * 5.0)),
    );

    return mix(kernel_bias_min, kernel_bias_max, kernel_bias_weight);
}

/// `LoadPreparedColor`: the input scaled by the exposure, in YCoCg.
fn load_prepared_color(q: vec2<i32>) -> vec3<f32> {
    return rgb_to_ycocg(max(vec3<f32>(0.0), textureLoad(color_tex, q, 0).rgb) * params.exposure);
}

/// Resolves the 3x3 of render-resolution samples bracketing this output pixel,
/// and builds the rectification box from the same taps.
fn compute_upsampled_color_and_weight(p: CommonParams, data_in: PassData) -> PassData {
    var data = data_in;
    let render = render_size_i();

    let dst_output_pos = vec2<f32>(p.px_hr) + vec2<f32>(0.5);
    let src_output_pos = dst_output_pos * params.downscale_factor;
    let src_input_pos = vec2<i32>(floor(src_output_pos));
    // Where the sample at offset (0,0) actually landed, and how far that is from
    // the output pixel's centre — the origin every tap's kernel distance is
    // measured from.
    let src_unjittered_pos = vec2<f32>(src_input_pos) + vec2<f32>(0.5) - params.jitter;
    let base_sample_offset = src_unjittered_pos - src_output_pos;

    // The 3x3 is not centred: it is shifted so the sample positions bracket the
    // output position rather than the input pixel that happens to contain it.
    let offset_tl = select(vec2<i32>(-1), vec2<i32>(-2), src_unjittered_pos > src_output_pos);

    // A pixel with no accumulation at all has no history to reconstruct against,
    // so the kernel would only sharpen noise; it takes the box centre instead.
    let is_initial_sample = p.accumulation == 0.0;
    let kernel_bias = compute_kernel_bias(p, data.history_weight);

    for (var row = 0; row < 3; row = row + 1) {
        for (var col = 0; col < 3; col = col + 1) {
            let step = vec2<i32>(col, row);
            let src_sample_offset = base_sample_offset + vec2<f32>(offset_tl + step);
            let src_sample_pos = src_input_pos + offset_tl + step;
            let on_screen = f32(is_on_screen(src_sample_pos, render));
            var sample = load_prepared_color(clamp(src_sample_pos, vec2<i32>(0), render - vec2<i32>(1)));
            // `FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT`: an initial sample's box is built on
            // tonemapped colour, through RGB so the tonemap does not desaturate.
            if (is_initial_sample && is_hdr()) {
                sample = rgb_to_ycocg(tonemap(ycocg_to_rgb(sample)));
            }

            if (!is_initial_sample) {
                let biased = src_sample_offset * kernel_bias;
                let weight = on_screen * lanczos2_approx_sq(dot(biased, biased));
                data.upsampled_color += sample * weight;
                data.upsampled_weight += weight;
            }

            // The box is built from the same taps but on a Gaussian falloff and
            // with no negative ring: it describes what colours are *present*
            // around this pixel, which the kernel's negative lobe has no bearing
            // on.
            let rectification_curve_bias = -2.3;
            let offset_sq = dot(src_sample_offset, src_sample_offset);
            let box_weight = exp(rectification_curve_bias * offset_sq) * on_screen;
            data.box = rectification_box_add_sample(
                data.box,
                row == 0 && col == 0,
                sample,
                box_weight,
            );
        }
    }

    data.box = rectification_box_compute_variance(data.box);

    data.upsampled_weight *= f32(data.upsampled_weight > FSR3_EPSILON);
    if (data.upsampled_weight > FSR3_EPSILON) {
        // Normalised so the deringing clamp can compare colours; the weight then
        // becomes this frame's share of the running accumulation.
        data.upsampled_color = data.upsampled_color / data.upsampled_weight;
        data.upsampled_weight *= AVERAGE_LANCZOS_WEIGHT_PER_FRAME;
        // The kernel's negative lobe can overshoot past every sample it was built
        // from.
        data.upsampled_color = clamp(data.upsampled_color, data.box.aabb_min, data.box.aabb_max);
    }

    if (is_initial_sample) {
        if (is_hdr()) {
            data.upsampled_color = rgb_to_ycocg(inverse_tonemap(ycocg_to_rgb(data.box.box_center)));
        } else {
            data.upsampled_color = data.box.box_center;
        }
        data.upsampled_weight = 1.0;
        data.history_weight = 0.0;
    }

    return data;
}

// --- rectify and blend --------------------------------------------------------

/// Clips the history into an ellipsoid around the current neighbourhood, sized by
/// how much this pixel is entitled to trust its history.
///
/// The box is an ellipsoid rather than a per-channel clamp so accepted history
/// keeps its hue, where a box clamp would bend it toward a corner no neighbouring
/// pixel actually has. Luma gets 1.7x the chroma axes' extent: a temporally
/// stable pixel varies more in brightness than in colour, and clipping luma as
/// tightly as chroma is what makes a clamped TAA look flat under motion.
fn rectify_history(p: CommonParams, data_in: PassData) -> PassData {
    var data = data_in;

    let velocity_factor = saturate(p.velocity_4k / 20.0);
    let distance_factor = saturate(0.75 - p.farthest_depth_meters / 20.0);
    let accumulation_factor = 1.0 - p.accumulation;
    let reactive_factor = sqrt(p.reactive);
    let shading_change_factor = p.shading_change;
    // Any one reason to distrust the history tightens the box on its own.
    let box_scale_t = max(
        velocity_factor,
        max(distance_factor, max(accumulation_factor, max(reactive_factor, shading_change_factor))),
    );

    let box_scale = mix(3.0, 1.0, box_scale_t);
    let scaled_box_vec = data.box.box_vec * vec3<f32>(1.7, 1.0, 1.0) * box_scale;

    let clamped_box_vec = max(scaled_box_vec, vec3<f32>(1.193e-7));
    let transformed = (data.history_color - data.box.box_center) / clamped_box_vec;

    if (length(transformed) > 1.0) {
        let clamped = normalize(transformed);
        let final_clamped = clamped * scaled_box_vec + data.box.box_center;

        // Locks and instability pull in opposite directions and both are gated on
        // the pixel actually having history worth arguing about.
        let history_contribution =
            max(p.luma_instability, data.lock_contribution) * p.accumulation * (1.0 - p.disocclusion);
        data.history_color = mix(final_clamped, data.history_color, saturate(history_contribution));
    }

    return data;
}

fn accumulate(data_in: PassData) -> PassData {
    var data = data_in;

    // Guards the blend when both weights have collapsed.
    data.history_weight *= f32(data.history_weight > FSR3_FP16_MIN);
    data.history_weight = max(FSR3_EPSILON, data.history_weight + data.upsampled_weight);

    // HDR input blends tonemapped, through RGB so the tonemap does not desaturate.
    if (is_hdr()) {
        data.upsampled_color = rgb_to_ycocg(tonemap(ycocg_to_rgb(data.upsampled_color)));
        data.history_color = rgb_to_ycocg(tonemap(ycocg_to_rgb(data.history_color)));
    }

    let alpha = saturate(data.upsampled_weight / data.history_weight);
    data.history_color = mix(data.history_color, data.upsampled_color, alpha);
    data.history_color = ycocg_to_rgb(data.history_color);

    if (is_hdr()) {
        data.history_color = inverse_tonemap(data.history_color);
    }

    return data;
}

// --- entry point --------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let display = display_size_i();
    let px_hr = vec2<i32>(gid.xy);
    if (any(px_hr >= display)) {
        return;
    }

    var p: CommonParams;
    p.px_hr = px_hr;
    p.hr_uv = (vec2<f32>(px_hr) + vec2<f32>(0.5)) * params.inv_display_size;
    // The render-resolution pixel whose jittered sample covers this output pixel.
    p.lr_uv_jittered = p.hr_uv + params.jitter * params.inv_render_size;
    p.lr_uv_hw = clamp_uv(p.lr_uv_jittered, params.render_size);

    // The engine supplies render-resolution motion vectors, so the vector for
    // this output pixel is the dilated one covering it.
    let mv_pos = clamp(
        vec2<i32>(p.hr_uv * params.render_size),
        vec2<i32>(0),
        render_size_i() - vec2<i32>(1),
    );
    p.mv = textureLoad(dilated_mv_tex, mv_pos, 0).xy;
    p.velocity_4k = get_4k_velocity(p.mv);

    p.reprojected_hr_uv = p.hr_uv + p.mv;
    p.is_existing_sample = is_uv_inside(p.reprojected_hr_uv);

    p.luma_instability = textureSampleLevel(
        luma_instability_tex,
        linear_sampler,
        clamp_uv(p.hr_uv, params.render_size),
        0.0,
    ).x;
    p.farthest_depth_meters = textureSampleLevel(
        farthest_mip1_tex,
        linear_sampler,
        clamp_uv(p.lr_uv_jittered, vec2<f32>(shading_change_size_i())),
        0.0,
    ).x;
    p.is_new_sample = !p.is_existing_sample || params.frame_index == 0.0;

    let masks = textureSampleLevel(masks_tex, linear_sampler, p.lr_uv_hw, 0.0);
    p.reactive = saturate(masks[REACTIVE]);
    p.disocclusion = saturate(masks[DISOCCLUSION]);
    p.shading_change = saturate(masks[SHADING_CHANGE]);
    p.accumulation = saturate(masks[ACCUMULATION]);
    p.accumulation *= f32(round(p.accumulation * 100.0) > 1.0);

    var data: PassData;
    data.upsampled_color = vec3<f32>(0.0);
    data.history_color = vec3<f32>(0.0);
    data.history_weight = 1.0;
    data.upsampled_weight = 0.0;
    data.lock = 0.0;
    data.lock_contribution = 0.0;

    if (p.is_existing_sample && !p.is_new_sample) {
        let reprojected = history_sample(p.reprojected_hr_uv, params.display_size);
        // `ReprojectHistoryColor`: the history is stored exposure-free.
        data.history_color = rgb_to_ycocg(reprojected.rgb * params.exposure);
        data.lock = reprojected.w;
    }

    data = update_lock_status(p, data);

    // The ceiling on the history's weight: a pixel that is not converged cannot
    // hold as much of its past, and fast motion caps it harder still.
    var base_accumulation = p.accumulation;
    base_accumulation = min(
        base_accumulation,
        mix(base_accumulation, 0.15, saturate(max(0.0, (p.velocity_4k * params.velocity_factor) / 0.5))),
    );
    data.history_weight = base_accumulation;

    data = compute_upsampled_color_and_weight(p, data);
    data = rectify_history(p, data);
    data = accumulate(data);

    let color = max(data.history_color / params.exposure, vec3<f32>(0.0));
    textureStore(history_out, px_hr, vec4<f32>(color, data.lock));
}
