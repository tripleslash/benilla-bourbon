// Night-sky stars (`Stars.m2`): each batch's texel under the vertex colour, white with the fade
// alpha, times a second stage's texel where the batch has one (`GL_COMBINE` MODULATE on RGB and
// alpha), multiplied in gamma space and blended premultiplied as the reference blends onto gamma
// bytes: alpha blend, or SRC_ALPHA, ONE for an additive batch. Depth is the far pin in
// `sky_vertex.wgsl`.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    forward_io::VertexOutput,
}

// `StarExt::params`: `.x` additive, `.y` stage 2 bound, `.z` the frame is linear.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> star: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var stage2_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var stage2_sampler: sampler;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let pbr_input = pbr_input_from_standard_material(in, is_front);
    // Stage 0: texel × `base_color` (white, alpha the fade × the batch weight). `rgb` arrives
    // linear (Bevy decodes the sRGB texture); the stages multiply the gamma bytes.
    let c0 = pbr_input.material.base_color;
    var rgb = linear_to_srgb(c0.rgb);
    var a = c0.a;
#ifdef VERTEX_UVS_B
    if star.y > 0.5 {
        let c1 = textureSample(stage2_texture, stage2_sampler, in.uv_b);
        rgb *= linear_to_srgb(c1.rgb);
        a *= c1.a;
    }
#endif
    // fork: modern lighting (FORK.md) — a linear frame takes linear colour.
    if star.z > 0.5 {
        rgb = srgb_to_linear(rgb);
    }
    // Premultiplied; alpha 0 leaves `dst` whole, the additive blend.
    return vec4<f32>(rgb * a, select(a, 0.0, star.x > 0.5));
}
