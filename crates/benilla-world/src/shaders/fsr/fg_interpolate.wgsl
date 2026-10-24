// FSR 3.1.5 frame generation, pass 5 — the interpolated frame
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_frameinterpolation.h`).
//
// Everything before this ran at render resolution and produced *decisions*; this
// pass runs at display resolution and produces pixels. For each one it fetches
// the same surface out of both source frames — backwards along the motion vector
// into frame N-1, forwards into frame N — and blends the two according to which
// of them could actually see it.
//
// ## Where the colour comes from
//
// The disocclusion mask says whether each source frame saw this surface at all,
// and the blend follows directly: seen by both, take the midpoint; seen by only
// one, take that one whole; seen by neither, give up and mark the pixel for
// inpainting. Nothing here fades gradually between those cases, because the mask
// was binarised for exactly that reason.
//
// A pixel whose reprojection leaves the image is the same situation arriving by a
// different route, and is handled the same way: a source with no bilinear weight
// left inside the frame contributes nothing.
//
// ## The optical-flow arbitration
//
// The second half of the pass is a sanity check on the first. It builds a second
// candidate — the same two frames blended along the *optical flow* field instead
// of the game's motion vectors — and compares how well each candidate's two
// fetches agree with each other. Motion vectors that describe what actually
// happened land on the same surface in both frames, so the two fetches match;
// vectors that are wrong land on different things, and they do not. The ratio of
// the two agreements decides how far to fall back.
//
// This port computes no optical flow (see `fsr/framegen.rs`), which is the
// SDK's own no-optical-flow configuration: the field reads back empty, so the
// second candidate is the two frames blended with *no* warping at all. That is
// still a useful thing to arbitrate against — it is what the frame would look
// like if nothing moved — and the check keeps its shape: where the game vectors
// produce a good match the result is untouched, and where they produce a bad one
// it softens towards the unwarped average rather than committing to a smear.
//
// The first ten frames after a reset skip the arbitration outright: the history
// the vectors are built from has not settled, and the comparison would be
// measuring that rather than the vectors.

@group(0) @binding(1) var prev_source_tex: texture_2d<f32>;
@group(0) @binding(2) var cur_source_tex: texture_2d<f32>;
@group(0) @binding(3) var disocclusion_tex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> field_x: array<u32>;
@group(0) @binding(5) var<storage, read> field_y: array<u32>;
@group(0) @binding(6) var mv_pyramid: texture_2d<f32>;
@group(0) @binding(7) var linear_clamp: sampler;
@group(0) @binding(8) var interp_out: texture_storage_2d<rgba8unorm, write>;

/// One of the two source fetches: the colour, and how much of the bilinear
/// footprint stayed inside the image. That second number is what tells the blend
/// a source has nothing to offer, as distinct from offering black.
struct InterpolationSourceColor {
    raw: vec3<f32>,
    bilinear_weight_sum: f32,
};

/// `SampleTextureBilinear`. Gathered by hand rather than through the sampler
/// because the out-of-bounds taps have to be dropped *and counted*, which a
/// clamped sampler would silently paper over by repeating the edge.
fn sample_source_bilinear(is_current: bool, uv: vec2<f32>, mv: vec2<f32>) -> InterpolationSourceColor {
    let size = display_size_i();
    let info = bilinear_sampling_data(uv + mv, params.display_size);

    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var i = 0; i < 4; i = i + 1) {
        let sample_pos = info.base_pos + bilinear_offset(i);
        if (!is_on_screen(sample_pos, size)) {
            continue;
        }
        let weight = info.weights[i];
        if (is_current) {
            color += textureLoad(cur_source_tex, sample_pos, 0).rgb * weight;
        } else {
            color += textureLoad(prev_source_tex, sample_pos, 0).rgb * weight;
        }
        weight_sum += weight;
    }

    var result: InterpolationSourceColor;
    result.raw = select(vec3<f32>(0.0), color / weight_sum, weight_sum != 0.0);
    result.bilinear_weight_sum = weight_sum;
    return result;
}

/// `LoadInpaintedGameFieldMv`, indexed in the render-resolution field by a
/// display-resolution uv.
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

/// `computeInterpolatedColor`. Returns the colour in `xyz` and the inpainting
/// weight in `w`.
fn compute_interpolated_color(p: vec2<i32>) -> vec4<f32> {
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_display_size;

    let game_mv = load_inpainted_game_field_mv(uv);
    let of_mv = sample_optical_flow_motion_vector_field(uv);

    // Only a footprint that is *entirely* visible counts as visible; the mask was
    // binarised per render pixel and this is the display-resolution read of it,
    // so anything less than one means the filter straddled a disocclusion edge.
    let mask = saturate(textureSampleLevel(disocclusion_tex, linear_clamp, uv, 0.0).xy);
    var disocclusion_factor = vec2<f32>(mask == vec2<f32>(1.0, 1.0));

    let prev_game = sample_source_bilinear(false, uv, game_mv.motion_vector);
    let cur_game = sample_source_bilinear(true, uv, -game_mv.motion_vector);
    let prev_of = sample_source_bilinear(false, uv, of_mv.motion_vector);
    let cur_of = sample_source_bilinear(true, uv, -of_mv.motion_vector);

    var interpolated = vec3<f32>(0.0);
    var inpainting_weight = 0.0;
    var disoccluded_factor = 0.0;

    // --- pick between the two sources ----------------------------------------
    {
        // A vector pointing off-screen is a source that cannot be sampled, which
        // is the same thing as a disocclusion as far as the blend is concerned.
        disocclusion_factor.x *= f32(!game_mv.pos_outside);
        disocclusion_factor.y *= f32(!game_mv.neg_outside);

        // Neither frame saw it: there is no correct colour, only a plausible one,
        // and finding that is the inpainting pass's job.
        inpainting_weight = saturate(max(
            inpainting_weight,
            f32(length(disocclusion_factor) <= FG_EPSILON),
        ));

        // 0.5 with both visible; pulled to 0 (all previous) or 1 (all current)
        // as one of them drops out.
        var t = 0.5;
        t += 0.5 * (1.0 - disocclusion_factor.x);
        t -= 0.5 * (1.0 - disocclusion_factor.y);
        interpolated = mix(prev_game.raw, cur_game.raw, saturate(t));
        disoccluded_factor = saturate(1.0 - min(disocclusion_factor.x, disocclusion_factor.y));

        // The reprojection left the image rather than being occluded within it.
        if (prev_game.bilinear_weight_sum == 0.0) {
            interpolated = cur_game.raw;
        } else if (cur_game.bilinear_weight_sum == 0.0) {
            interpolated = prev_game.raw;
        }
        if (prev_game.bilinear_weight_sum == 0.0 && cur_game.bilinear_weight_sum == 0.0) {
            inpainting_weight = 1.0;
        }
    }

    // --- arbitrate against the optical-flow candidate -------------------------
    {
        var of_t = 0.5;
        if (prev_of.bilinear_weight_sum > 0.0 && cur_of.bilinear_weight_sum > 0.0) {
            of_t = 0.5;
        } else if (prev_of.bilinear_weight_sum > 0.0) {
            of_t = 0.0;
        } else {
            of_t = 1.0;
        }
        let of_color = mix(prev_of.raw, cur_of.raw, of_t);

        // How well each candidate's two fetches agree with each other.
        let of_similarity = normalized_dot3(prev_of.raw, cur_of.raw);
        var game_similarity = normalized_dot3(prev_game.raw, cur_game.raw);

        // Where the surface was genuinely disoccluded the two fetches *should*
        // disagree, and holding that against the game vectors would be punishing
        // them for being right.
        game_similarity = mix(max(FG_EPSILON, game_similarity), 1.0, saturate(disoccluded_factor));

        var game_mv_bias = saturate(game_similarity / max(FG_EPSILON, of_similarity));
        // Before the history has settled the comparison means nothing; trust the
        // game vectors outright.
        let frame_index_factor = f32(params.frame_index_since_reset < 10.0);
        game_mv_bias = mix(game_mv_bias, 1.0, frame_index_factor);

        interpolated = mix(of_color, interpolated, saturate(game_mv_bias));
    }

    return vec4<f32>(interpolated, inpainting_weight);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = display_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    var result: vec4<f32>;
    if (params.frame_index_since_reset == 0.0) {
        // Nothing to interpolate from. Passing the current frame through keeps the
        // present cadence uniform through a scene switch rather than stalling it.
        result = vec4<f32>(textureLoad(cur_source_tex, p, 0).rgb, 0.0);
    } else {
        result = compute_interpolated_color(p);
    }

    textureStore(interp_out, p, result);
}
