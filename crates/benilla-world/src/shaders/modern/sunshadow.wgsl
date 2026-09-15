// Modern lighting (FORK.md, P2): sampling the modern sun's cascade shadow map.
//
// Imported as `benilla_modern::sunshadow` — a module name no hooked shader uses as an identifier
// (naga_oil substitutes a module alias for every bare identifier of that name).
//
// No bindings are declared here: terrain and wow_model bind the map in the material group,
// static_gx in its light group, so each receiver passes its own texture, sampler and cascade block.
#define_import_path benilla_modern::sunshadow

// Mirrors `benilla_world::modern::shadow::ModernCascades::std430` (a layout test pins the size).
struct ModernCascades {
    // Reverse-Z orthographic light clip per cascade.
    clip_from_world: array<mat4x4<f32>, 4>,
    // View distance (yd) at which each cascade ends.
    splits: vec4<f32>,
    // World size (yd) of one texel per cascade — the unit of the normal offset.
    texel_world: vec4<f32>,
    // Constant depth bias per cascade, in that cascade's depth units.
    depth_bias: vec4<f32>,
    // x = cascade count (0 = no modern shadow this frame), y = 1 / map size, z = normal offset in
    // texels, w = debug view (receivers output raw visibility).
    params: vec4<f32>,
}

fn cascade_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    i: u32,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
) -> f32 {
    let offset_pos = world_pos + normal * (c.texel_world[i] * c.params.z);
    let clip = c.clip_from_world[i] * vec4<f32>(offset_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    // Reverse-Z: a receiver is lit when it is at least as close to the light as the stored caster.
    let depth = ndc.z + c.depth_bias[i];
    // 5-tap rotated cross over the hardware 2x2 compare — soft edges at a web-friendly cost.
    var taps = array<vec2<f32>, 5>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.2, 0.4),
        vec2<f32>(-0.4, 1.2),
        vec2<f32>(-1.2, -0.4),
        vec2<f32>(0.4, -1.2),
    );
    var lit = 0.0;
    for (var t = 0u; t < 5u; t = t + 1u) {
        lit = lit + textureSampleCompareLevel(map, samp, uv + taps[t] * c.params.y, i32(i), depth);
    }
    return lit / 5.0;
}

// Sun visibility (1 = lit) at a world position seen `view_distance` yd in front of the camera.
fn sun_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_distance: f32,
) -> f32 {
    let count = u32(c.params.x);
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_distance < c.splits[i]) {
            let v = cascade_visibility(map, samp, c, i, world_pos, normal);
            if (i + 1u == count) {
                // The last cascade fades to lit instead of ending on a hard line.
                let fade = smoothstep(c.splits[i] * 0.85, c.splits[i], view_distance);
                return mix(v, 1.0, fade);
            }
            return v;
        }
    }
    return 1.0;
}
