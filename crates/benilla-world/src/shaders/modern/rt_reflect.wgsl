enable wgpu_ray_query;

// fork: G — the ray-traced half of the water's reflection (`modern::rt_reflect`, native only): one
// reflected ray as a ray query against `benilla_modern::rt_scene`'s TLAS (the static world and the
// terrain, cutout texels alpha-tested), its hit taken as far as the probe trace takes one
// (`gi_trace_rt.wgsl`'s `cast`): the vertex normal, the texel at a mip in the authored value space,
// and the sun by one ray through the scene (`rtshadow::sun_ray`). The water lights it with its own
// rows and probes; a terrain hit leaves its albedo to the caller, the splat not being in the scene.
//
// Every value this module hands back is a plain struct, so the importer declares no ray-query type
// and needs no `enable` of its own (as with `rtshadow`). Imported as `benilla_modern::rtreflect`,
// under `RT_REFLECT`, which only a pipeline that declares the scene at `RT_GROUP` carries.
#define_import_path benilla_modern::rtreflect

#import benilla_modern::rt_scene
#import benilla_modern::rtshadow
#import benilla_modern::colorspace

// The texel level a reflected hit reads: the water reads the mirror and the screen through a
// quarter-resolution blur, and a hit two levels down is about as soft.
const ALBEDO_MIP: f32 = 2.0;
// `static_gx_voxel.wgsl`'s untextured albedo, as the probe trace takes it.
const UNTEXTURED: vec3<f32> = vec3<f32>(0.5);
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
    // A static hit's texel colour in the authored value space the world lights in (mid grey
    // untextured); zero on terrain.
    albedo: vec3<f32>,
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
    if (s.kind == rt_scene::KIND_TERRAIN) {
        out.kind = 2u;
    } else {
        out.kind = 1u;
        var albedo = UNTEXTURED;
        if ((s.word & rt_scene::WORD_TEXTURED) != 0u) {
            // The pool's arrays are sRGB views: the sample arrives linear, and the world lights in
            // the authored value space.
            albedo = colorspace::linear_to_srgb(rt_scene::texel(s.row, s.word, s.uv, ALBEDO_MIP).rgb);
        }
        out.albedo = albedo;
    }
    if (dot(out.normal, to_sun) > 0.0) {
        out.sun = rtshadow::sun_ray(out.position + out.normal * SUN_LIFT_YD, to_sun);
    } else {
        out.sun = 0.0;
    }
    return out;
}
