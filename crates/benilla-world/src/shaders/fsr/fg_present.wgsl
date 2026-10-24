// FSR 3.1.5 frame generation — the presentation blit.
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
//
// With frame generation on, the upscaler no longer writes the swapchain: RCAS
// resolves into an off-screen presentation image so that the interpolator has
// both of the last two of them to work from. Two images then have to reach the
// screen per simulated frame — the generated one first, then the rendered one —
// and each present needs its own swapchain texture, so each is copied across by
// this pass.
//
// It is a copy and nothing else. Values are stored already encoded, exactly as
// RCAS left them, so the only work is undoing that encode for the sRGB swapchain
// view, which re-applies it on write.

@group(0) @binding(1) var src_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
};

// Single oversized triangle covering the target; the source is addressed by
// integer pixel coordinate, so no varyings are needed.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VsOut;
    out.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hi = vec2<i32>(textureDimensions(src_tex, 0)) - vec2<i32>(1, 1);
    let p = clamp(vec2<i32>(in.clip.xy), vec2<i32>(0, 0), hi);

    var rgb = textureLoad(src_tex, p, 0).rgb;
    if (params.srgb_output > 0.5) {
        rgb = srgb_to_linear(rgb);
    }
    return vec4<f32>(rgb, 1.0);
}
