// FSR 3.1.5 frame generation, pass 3a — the base level of the motion vector
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// field's inpainting pyramid
// (`ffx_frameinterpolation_compute_game_vector_field_inpainting_pyramid.h`,
// `SpdLoadSourceImage` plus SPD's first reduction).
//
// Unpacks the scattered field into plain `(mv.xy, high, low)` and reduces each
// 2x2 of it into one texel of the pyramid's level 0, dropping the cells nothing
// landed in rather than averaging their emptiness in. See `reduce4_mv` in
// `fg_common.wgsl`.

@group(0) @binding(1) var<storage, read> field_x: array<u32>;
@group(0) @binding(2) var<storage, read> field_y: array<u32>;
@group(0) @binding(3) var mip_out: texture_storage_2d<rgba16float, write>;

/// One source cell, in the pyramid's `(mv.xy, high, low)` form. Out-of-range
/// reads come back with a zero high-priority factor, which is exactly the
/// "nothing here" the reduction already skips — so an odd source dimension needs
/// no special case.
fn load_source(p: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
    if (!is_on_screen(p, size)) {
        return vec4<f32>(0.0);
    }
    let i = u32(p.y * size.x + p.x);
    let e = unpack_vector_field_entries(vec2<u32>(field_x[i], field_y[i]));
    return vec4<f32>(e.motion_vector, e.high_priority_factor, e.low_priority_factor);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = vec2<i32>(textureDimensions(mip_out));
    let p = vec2<i32>(gid.xy);
    if (any(p >= dst_size)) {
        return;
    }

    let src_size = render_size_i();
    let base = p * 2;
    let reduced = reduce4_mv(
        load_source(base, src_size),
        load_source(base + vec2<i32>(1, 0), src_size),
        load_source(base + vec2<i32>(0, 1), src_size),
        load_source(base + vec2<i32>(1, 1), src_size),
    );

    textureStore(mip_out, p, reduced);
}
