// FSR 3.1.5 frame generation, pass 2 — the game motion vector field
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_frameinterpolation_game_motion_vector_field.h`).
//
// Turns the renderer's per-pixel motion vectors, which describe where each pixel
// of *this* frame came from, into a field indexed by the *interpolated* frame:
// for every cell of the in-between image, the motion that carries it to frame N
// and back to frame N-1. Each render pixel scatters its half-motion into the four
// cells a bilinear fetch at its halfway point would touch, and the packed
// priority in the high bits of the atomic (see `fg_common.wgsl`) decides who
// wins a contested cell.
//
// Priority has two parts, and both matter:
//
// * **High** — nearness, from the eye-space depth. Where a foreground object
//   passes in front of a background, both scatter into the same cells; taking the
//   nearer surface's vector is what stops the background's motion from tearing a
//   hole through the object.
// * **Low** — how well the two source frames agree in luminance along the vector.
//   A vector that lands somewhere with a completely different brightness is
//   describing a surface that was occluded, or lit differently, or simply wrong;
//   it loses to one that lands on a match.
//
// ## Trails
//
// A moving object leaves behind cells that nothing scatters into: the object has
// moved off them and the background behind it never carried the object's motion.
// Those cells are holes in the field, and the inpainting pyramid has to guess
// them. To make that guess a good one, each moving pixel also walks *backwards*
// along its own vector and lays down a trail of **secondary** vectors, marked as
// such by the top bit so they always lose to a real one. Their depth priority is
// inverted — a trail belongs behind whatever drew it — and their low priority
// rewards vectors pointing towards the centre of the image over ones leaving it.
//
// The walk stops early once it has run over four primary vectors, which is the
// reference's way of saying it has re-entered an area that is properly covered
// and has nothing left to contribute.

@group(0) @binding(1) var dilated_mv_tex: texture_2d<f32>;
@group(0) @binding(2) var depths_tex: texture_2d<f32>;
@group(0) @binding(3) var prev_source_tex: texture_2d<f32>;
@group(0) @binding(4) var cur_source_tex: texture_2d<f32>;
@group(0) @binding(5) var linear_clamp: sampler;
@group(0) @binding(6) var<storage, read_write> field_x: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> field_y: array<atomic<u32>>;

/// `getPriorityFactorFromViewSpaceDepth`. The cube root compresses the huge
/// dynamic range of a depth buffer into something the ten available bits can
/// resolve across a whole scene, and `d/(1+d)` maps that onto `[0, 1)` with no far
/// clamp; the result is inverted so nearer means larger. Never zero: zero is what
/// marks a cell as empty.
fn priority_factor_from_view_space_depth(view_space_depth_meters: f32) -> u32 {
    let d = pow(max(view_space_depth_meters, 0.0), 0.33);
    let factor = u32((1.0 - (d * (1.0 / (1.0 + d)))) * f32(PRIORITY_HIGH_MAX));
    return max(1u, factor);
}

/// The colour-agreement half of the priority: how close the two frames' luminance
/// is at the two ends of the vector, zeroed outright if the far end left the
/// image.
fn low_priority_factor(uv: vec2<f32>, mv: vec2<f32>) -> u32 {
    let prev_col = textureSampleLevel(prev_source_tex, linear_clamp, uv + mv, 0.0).rgb;
    let cur_col = textureSampleLevel(cur_source_tex, linear_clamp, uv, 0.0).rgb;
    // The floor keeps a black pixel from making the ratio meaningless.
    let prev_luma = 0.001 + raw_rgb_to_luminance(prev_col);
    let cur_luma = 0.001 + raw_rgb_to_luminance(cur_col);

    let agreement = round(min_divided_by_max(prev_luma, cur_luma) * f32(PRIORITY_LOW_MAX));
    return u32(agreement) * u32(is_uv_inside(uv + mv));
}

fn field_index(p: vec2<i32>, size: vec2<i32>) -> u32 {
    return u32(p.y * size.x + p.x);
}

/// `UpdateGameMotionVectorField`.
fn update_field(p: vec2<i32>, size: vec2<i32>, packed: vec2<u32>) {
    let i = field_index(p, size);
    atomicMax(&field_x[i], packed.x);
    atomicMax(&field_y[i], packed.y);
}

/// `UpdateGameMotionVectorFieldEx`: the same, handing back whatever was already
/// there so the trail can tell when it has walked back into covered ground.
fn update_field_ex(p: vec2<i32>, size: vec2<i32>, packed: vec2<u32>) -> u32 {
    let i = field_index(p, size);
    let previous_x = atomicMax(&field_x[i], packed.x);
    let previous_y = atomicMax(&field_y[i], packed.y);
    return max(previous_x, previous_y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;
    let depth = textureLoad(depths_tex, p, 0).x;
    let mv = textureLoad(dilated_mv_tex, p, 0).xy;
    let mv_half = mv * 0.5;
    let interpolated_location_uv = uv + mv_half;

    let high_priority_primary = priority_factor_from_view_space_depth(view_space_depth(depth));
    let low_priority_primary = low_priority_factor(uv, mv);

    // --- primary vectors ------------------------------------------------------
    {
        let packed = pack_vector_field_entries(true, high_priority_primary, low_priority_primary, mv_half);
        let info = bilinear_sampling_data(interpolated_location_uv, params.render_size);
        for (var i = 0; i < 4; i = i + 1) {
            let dst = info.base_pos + bilinear_offset(i);
            if (is_on_screen(dst, size)) {
                update_field(dst, size, packed);
            }
        }
    }

    // --- secondary vectors (the trail) ----------------------------------------
    //
    // Guarded on sub-pixel motion, which also keeps `normalize` off a zero vector.
    if (length(mv_half * params.render_size) <= FG_EPSILON) {
        return;
    }

    var write_secondary = true;
    var primary_hits = 0u;
    // One pixel's worth of uv per step, along the diagonal.
    let step_scale = length(params.inv_render_size);
    let step_mv = normalize(mv);
    // Never trail further than the motion itself, and never across more than half
    // the image however fast the object moves.
    let break_dist = min(length(mv_half), length(vec2<f32>(0.5, 0.5)));

    var mv_scale = step_scale;
    while (mv_scale <= break_dist && write_secondary) {
        let secondary_uv = interpolated_location_uv - step_mv * mv_scale;
        let info = bilinear_sampling_data(secondary_uv, params.render_size);

        // Reversed depth priority: a trail sits behind the surface that laid it.
        let high_priority_secondary = max(1u, PRIORITY_HIGH_MAX - high_priority_primary);
        // A trail heading into the image is more useful to inpaint from than one
        // heading out of it.
        let to_center = normalize(vec2<f32>(0.5, 0.5) - secondary_uv);
        let low_priority_secondary = u32(max(0.0, dot(to_center, step_mv)) * f32(PRIORITY_LOW_MAX));
        let packed = pack_vector_field_entries(
            false,
            high_priority_secondary,
            low_priority_secondary,
            mv_half,
        );

        // Unlike the primary, only the nearest bilinear tap is written: a trail is
        // a hint, not a measurement, and spreading it would swamp the field.
        let dst = info.base_pos + bilinear_offset(0);
        write_secondary = write_secondary && is_on_screen(dst, size);
        if (write_secondary) {
            let existing = update_field_ex(dst, size, packed);
            primary_hits += u32(packed_vector_field_entry_is_primary(existing));
            write_secondary = write_secondary && (primary_hits <= 3u);
        }

        mv_scale += step_scale;
    }
}
