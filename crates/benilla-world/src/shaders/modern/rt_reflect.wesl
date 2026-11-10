import benilla_world::shaders::modern::rt_scene;
import benilla_world::shaders::modern::rt_shadow as rtshadow;

enable wgpu_ray_query;

// fork: G — the ray-traced half of the water's reflection (`modern::rt_reflect`, native only): one
// reflected ray as a ray query against `benilla_modern::rt_scene`'s TLAS (the static world and the
// terrain, cutout texels alpha-tested), and what the water needs to light its hit as the mirror's
// own render lights that surface: the vertex normal, the texel in the authored value space, the
// vertex colour and the item's word and SIDN, and the sun by one ray through the scene
// (`rtshadow::sun_ray`). A terrain hit leaves its albedo to the caller, the splat not being in the
// scene.
//
// Every value this module hands back is a plain struct, so the importer declares no ray-query type
// and needs no `enable` of its own (as with `rtshadow`). Imported as `benilla_modern::rtreflect`,
// under `RT_REFLECT`, which only a pipeline that declares the scene at `RT_GROUP` carries.


// The texel level a reflected hit reads: the mirror renders at a reduced size and the water reads
// it through a blurred chain, and a hit two levels down is about as soft.
const ALBEDO_MIP: f32 = 2.0;
// The sun ray's start off the hit, along its normal (yd).
const SUN_LIFT_YD: f32 = 0.05;

// What a reflected ray found.
struct ReflectHit {
    // 0 = nothing within the reach (the sky), 1 = the static world, 2 = terrain.
    kind: u32,
    t: f32,
    position: vec3<f32>,
    // The interpolated vertex normal in world space, turned to face the ray.
    normal: vec3<f32>,
    // A static hit's texel as static_gx samples it: the pool holds the BLP's bytes in a non-sRGB
    // format (`world_assets::color_texture_format`), so the sample IS the authored (gamma) value
    // the world lights; white where the item is untextured, as static_gx's `base`. Zero on terrain.
    albedo: vec3<f32>,
    // The interpolated vertex colour (MOCV or the baked tint); white where the vertex has none.
    color: vec4<f32>,
    // The static item's vertex word (`static_gx.wgsl`'s WORD_* bits); 0 on terrain.
    word: u32,
    // The item's SIDN night glow colour, 0..1 (its record's z); zero on terrain.
    sidn: vec3<f32>,
    // The sun's reach at the hit through the scene: 1 lit, 0 shaded (0 where the face turns away).
    sun: f32,
}

// The nearest blocking surface along `dir` from `origin` in [t_min, reach]. Nothing on the stand-in
// (`rt_scene::live()` false), where the caller keeps the sky.
fn trace(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, reach: f32, to_sun: vec3<f32>) -> ReflectHit {
    var out: ReflectHit;
    out.kind = 0u;
    out.t = reach;
    out.position = origin + dir * reach;
    out.normal = -dir;
    out.albedo = vec3<f32>(0.0);
    out.color = vec4<f32>(1.0);
    out.word = 0u;
    out.sidn = vec3<f32>(0.0);
    out.sun = 1.0;
    if (!rt_scene::live()) {
        return out;
    }
    let hit = rt_scene::trace(origin, dir, t_min, reach, rt_scene::MASK_ALL);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return out;
    }
    let s = rt_scene::surface(origin, dir, hit);
    out.t = s.t;
    out.position = s.position;
    out.normal = s.normal;
    out.color = s.color;
    if (s.kind == rt_scene::KIND_TERRAIN) {
        out.kind = 2u;
    } else {
        out.kind = 1u;
        out.word = s.word;
        out.albedo = rt_scene::texel(s.row, s.word, s.uv, ALBEDO_MIP).rgb;
        let rec = rt_scene::record(s.row, s.word);
        out.sidn = vec3<f32>(
            f32(rec.z & 0xffu),
            f32((rec.z >> 8u) & 0xffu),
            f32((rec.z >> 16u) & 0xffu),
        ) / 255.0;
    }
    if (dot(out.normal, to_sun) > 0.0) {
        out.sun = rtshadow::sun_ray(out.position + out.normal * SUN_LIFT_YD, to_sun);
    } else {
        out.sun = 0.0;
    }
    return out;
}
