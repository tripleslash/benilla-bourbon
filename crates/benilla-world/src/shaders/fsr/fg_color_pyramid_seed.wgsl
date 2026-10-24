// FSR 3.1.5 frame generation, pass 6a — the base level of the interpolated
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
// image's inpainting pyramid
// (`ffx_frameinterpolation_compute_inpainting_pyramid.h`, `SpdLoadSourceImage`
// plus SPD's first reduction).
//
// Reduces the interpolated frame into a pyramid whose weight is *confidence*
// rather than coverage: the interpolation stored how much of each pixel it had to
// give up on, and this inverts that into how much of it is real. Pixels that were
// given up on therefore contribute nothing to the levels above, which is what
// stops a hole from propagating up the chain and being inpainted with itself.

@group(0) @binding(1) var interp_tex: texture_2d<f32>;
@group(0) @binding(2) var mip_out: texture_storage_2d<rgba16float, write>;

fn load_source(p: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
    if (!is_on_screen(p, size)) {
        return vec4<f32>(0.0);
    }
    let c = textureLoad(interp_tex, p, 0);
    // Alpha arrives as the inpainting weight; reverse it into a sample weight.
    return vec4<f32>(c.rgb, saturate(1.0 - c.a));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = vec2<i32>(textureDimensions(mip_out));
    let p = vec2<i32>(gid.xy);
    if (any(p >= dst_size)) {
        return;
    }

    let src_size = display_size_i();
    let base = p * 2;
    let reduced = reduce4_color(
        load_source(base, src_size),
        load_source(base + vec2<i32>(1, 0), src_size),
        load_source(base + vec2<i32>(0, 1), src_size),
        load_source(base + vec2<i32>(1, 1), src_size),
    );

    textureStore(mip_out, p, reduced);
}
