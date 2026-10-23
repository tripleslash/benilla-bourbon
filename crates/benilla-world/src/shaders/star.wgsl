// Night-sky stars (`Stars.m2`): each batch's texel under the vertex colour, white with the fade
// alpha, times a second stage's texel where the batch has one (`GL_COMBINE` MODULATE on RGB and
// alpha), multiplied in gamma space and blended premultiplied as the reference blends onto gamma
// bytes: alpha blend, or SRC_ALPHA, ONE for an additive batch. Depth is the far pin in
// `sky_vertex.wgsl`.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

// `StarExt::params`: `.x` additive, `.y` stage 2 bound, `.z` the frame is linear, `.w` the
// reciprocal of the modern post's exposure (1 on a classic frame).
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> star: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var stage2_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var stage2_sampler: sampler;
// fork: `StarExt::horizon`: gone below the view elevation (sine) `.x`, whole above `.y`; `.y` 0 = off.
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var<uniform> horizon: vec4<f32>;
// fork: `StarExt::tint`: the sky's hue at luma 1 `.rgb`, and the share of the dome's colour that
// takes it `.w` (0 = the authored colour).
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var<uniform> tint: vec4<f32>;

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
    // fork: modern lighting (FORK.md) — a linear frame takes linear colour. The
    // alpha folds in gamma first, then the product decodes, as `wow_model.wgsl` does: a linear
    // `texel × a` is far brighter than the reference's gamma one. `dst`'s share decodes likewise,
    // `(1 − a)` on gamma bytes being `(1 − a)^2.2` on linear ones.
    // fork: and the colour leaves pre-divided by the post's exposure, which then multiplies it back
    // to the classic level instead of lifting the night sky with the lit world.
    if star.z > 0.5 {
        // fork: the horizon fade: the dome draws after the fog composite, so near the
        // horizon it gives way to the hazed sky band under it.
        if horizon.y > 0.0 {
            let up = normalize(in.world_position.xyz - view.world_position).y;
            a *= smoothstep(horizon.x, horizon.y, up);
        }
        // fork: the dome's colour gives way to the sky's hue at its own luma (gamma bytes, as the
        // stages multiply them), so the nebulae read as the zone's night rather than patch-Y's teal.
        let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        rgb = mix(rgb, min(tint.rgb * luma, vec3<f32>(1.0)), tint.w);
        let keep = pow(1.0 - a, 2.2);
        return vec4<f32>(srgb_to_linear(rgb * a) * star.w, select(1.0 - keep, 0.0, star.x > 0.5));
    }
    // Premultiplied; alpha 0 leaves `dst` whole, the additive blend.
    return vec4<f32>(rgb * a, select(a, 0.0, star.x > 0.5));
}
