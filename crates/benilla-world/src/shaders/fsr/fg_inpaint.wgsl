// FSR 3.1.5 frame generation, pass 7 — inpainting
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// (`ffx_frameinterpolation_inpainting.h`), and the last write before
// presentation.
//
// Some pixels of the interpolated frame were visible in neither source: a surface
// that moved out from behind something and is genuinely new in the second frame
// has no earlier image, and the first frame's copy of it is whatever was in
// front. The interpolation marked those and left them; this pass invents them.
//
// It does so by reading *every* level of the pyramid built from the interpolated
// frame, not the first one that answers. Each level contributes its own weighted
// average of the real pixels near this one, and the levels are combined with a
// cubic falloff — a fine level is a much better guess than a coarse one, but a
// coarse one still beats nothing when the hole is wide enough that the fine
// levels are empty too. The result is a plausible continuation of the
// surroundings rather than a hard patch, which is what a hole a few pixels across
// at the trailing edge of a moving object needs.
//
// The reference's HUD-less composition and debug overlays are omitted: the engine
// composites no separate UI layer, and the debug views have no equivalent here.

@group(0) @binding(1) var interp_tex: texture_2d<f32>;
@group(0) @binding(2) var color_pyramid: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

/// The number of levels the reference's falloff is written against. Kept at the
/// reference's constant rather than tied to the pyramid's actual height, so the
/// weight a given level carries does not change with the display resolution.
const INPAINTING_FALLOFF_LEVELS: f32 = 10.0;

/// `ComputeInpaintingLevel`: one level's bilinear neighbourhood, restricted to the
/// taps that hold real pixels. Returns the *unnormalised* sum in `rgb` and the
/// weight it was accumulated with in `w`, since the caller needs both.
fn compute_inpainting_level(uv: vec2<f32>, level: i32, tex_size: vec2<i32>) -> vec4<f32> {
    let info = bilinear_sampling_data(uv, vec2<f32>(tex_size));

    var color = vec4<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        let sample_pos = info.base_pos + bilinear_offset(i);
        if (!is_on_screen(sample_pos, tex_size)) {
            continue;
        }
        let s = textureLoad(color_pyramid, sample_pos, level);
        let weight = info.weights[i] * f32(s.w > 0.0);
        color += vec4<f32>(s.rgb * weight, weight);
    }
    return color;
}

/// `ComputeInpainting`: every level, weighted towards the fine ones.
fn compute_inpainting(p: vec2<i32>, fallback: vec3<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_display_size;

    var color = vec4<f32>(0.0);
    var tex_size = display_size_i();
    let mips = i32(params.color_pyramid_mips);

    for (var level = 0; level < mips; level = level + 1) {
        tex_size = tex_size / 2;

        let mip_color = compute_inpainting_level(uv, level, tex_size);
        if (mip_color.w <= 0.0) {
            continue;
        }
        let normalized = mip_color.rgb / mip_color.w;
        let mip_weight = pow(1.0 - f32(level) / INPAINTING_FALLOFF_LEVELS, 3.0) * mip_color.w;
        color += vec4<f32>(normalized, 1.0) * mip_weight;
    }

    // Every level empty means the whole frame was inpainted, which only a
    // degenerate input produces — but the reference's unguarded divide would turn
    // it into a NaN, and a NaN reaching the swapchain is permanent on screen.
    if (color.w <= 0.0) {
        return fallback;
    }
    return color.rgb / color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = display_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    let interpolated = textureLoad(interp_tex, p, 0);
    var rgb = interpolated.rgb;

    let inpainting_weight = interpolated.a;
    if (inpainting_weight > FG_EPSILON) {
        rgb = mix(rgb, compute_inpainting(p, rgb), inpainting_weight);
    }

    textureStore(output, p, vec4<f32>(rgb, 1.0));
}
