// The cloud dome. Its colour is CPU-side as in the reference: the kernel's `0x6cfb00` port builds
// the image the reference binds as its texture (`0x58ac70`). This stage applies the dome's rim
// fade (`0x6d0530`) and blends the raw gamma texels premultiplied. Depth is the far pin in
// `sky_vertex.wgsl`.

#import bevy_pbr::forward_io::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var cloud_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var cloud_samp: sampler;

// fork: modern lighting (FORK.md, P4) — the key light for the modern cloud layer
// (`modern::sky`); zero in a classic frame.
#import bevy_pbr::mesh_view_bindings::view
struct CloudLight {
    // xyz = toward the key light; w = modern flag.
    modern: vec4<f32>,
    // x = the key elevation's brightness; y = silver-lining strength; z = its lobe g.
    light: vec4<f32>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> cloud_light: CloudLight;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(cloud_tex, cloud_samp, in.uv);
    var a = texel.a;
#ifdef VERTEX_COLORS
    a *= in.color.a; // the dome's rim fade (ring alphas)
#endif
    var rgb = texel.rgb;
    // fork: modern lighting (P4) — clouds take the key light's elevation, and thin edges toward the
    // light carry a silver lining (HDR, so the bloom picks it up).
    if (cloud_light.modern.w > 0.5) {
        let dir = normalize(in.world_position.xyz - view.world_position.xyz);
        let g = cloud_light.light.z;
        let denom = max(1.0 + g * g - 2.0 * g * dot(dir, normalize(cloud_light.modern.xyz)), 1e-4);
        let lobe = (1.0 - g * g) / (denom * sqrt(denom));
        rgb = rgb * (cloud_light.light.x + cloud_light.light.y * 0.05 * lobe * (1.0 - a));
    }
    // Premultiplied gamma blend; the RGB is already the reference's byte math.
    return vec4<f32>(rgb * a, a);
}
