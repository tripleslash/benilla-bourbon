enable wgpu_ray_query;

// fork: the probe trace's ray stage at the ray-traced tier (`modern::gi::rt_gi`, native only): the
// twin of `gi_trace.wgsl`'s `trace_rays` that casts each ray as a ray query against the TLAS of
// `benilla_modern::rt_scene` (the static world and the terrain, cutout texels alpha-tested) instead
// of marching the voxel clipmap and the heightfield. `setup_probes` and `sum_probes` are
// `gi_trace.wgsl`'s own, over the same group 0, so a probe is set up, summed and blended exactly as
// on the voxel path; only what a ray finds differs.
//
// A hit is shaded as the voxel path shades a voxel: the surface's albedo (its texel at the voxel
// albedo's mip, in the lanes' value space; mid grey untextured; the ground albedo on terrain) times
// the key light under the cascades, the probe's near lights and the previous frame's field.
//
// Group 0 is `modern_gi_trace` (node.rs), of which this stage reads the block, the cascades, the
// light rows, the uniform, the work list, the setups and the ray results; group 1 is the scene
// (`rt_scene::shader_defs(1)`).
#import benilla_modern::probegi
#import benilla_modern::sunshadow
#import benilla_modern::locallight
#import benilla_modern::colorspace
#import benilla_modern::rt_scene
// fork: F — the ray-traced shadows, where they run.
#ifdef RT_SHADOWS
#import benilla_modern::rtshadow
#endif

// `gi_trace.wgsl`'s `LightRows`.
struct LightRows {
    rows: array<vec4<f32>, 21>,
    points: array<vec4<f32>, 512>,
}

// `gi_trace.wgsl`'s `GiTrace`; `clip.info.w` is the ray's reach (yd).
struct GiTrace {
    gi: probegi::GiParams,
    clip: probegi::GiClip,
    cam: vec4<f32>,
    fwd: vec4<f32>,
    rot_x: vec4<f32>,
    rot_y: vec4<f32>,
    rot_z: vec4<f32>,
    // x = backface share that invalidates a probe, y = the sun bias (yd), z = ray start offset (yd),
    // w = the bounce's gain.
    tune: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> modern_block: sunshadow::ModernBlock;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
@group(0) @binding(2) var shadow_samp: sampler_comparison;
@group(0) @binding(3) var<storage, read> light: LightRows;
@group(0) @binding(7) var<uniform> trace: GiTrace;

struct GiWork {
    count: u32,
    items: array<u32>,
}
@group(0) @binding(9) var<storage, read> work: GiWork;

// `gi_trace.wgsl`'s `GiSetup`, as `setup_probes` leaves it.
struct GiSetup {
    pos: vec4<f32>,
    info: vec4<u32>,
    near: vec4<u32>,
    cls: vec4<f32>,
}
@group(0) @binding(10) var<storage, read_write> setups: array<GiSetup>;
@group(0) @binding(11) var<storage, read_write> ray_out: array<vec4<f32>>;

const LANES: u32 = 64u;
const NEAR_LIGHTS: u32 = 4u;
// `static_gx_voxel.wgsl`'s albedo mip (`modern_gi::VOXEL_ALBEDO_MIP`): a texel's worth of detail is
// noise to a probe, and the two paths have to bounce the same colour.
const ALBEDO_MIP: f32 = 3.0;
// `static_gx_voxel.wgsl`'s untextured albedo.
const UNTEXTURED: vec3<f32> = vec3<f32>(0.5);
var<private> near_light: array<u32, 4>;
var<private> near_count: u32;

// `gi_trace.wgsl`'s `local_light_at`.
fn local_light_at(p: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var k = 0u; k < near_count; k = k + 1u) {
        let i = near_light[k];
        let pos_range = light.points[2u * i];
        let to_light = pos_range.xyz - p;
        let d = length(to_light);
        let nl = max(dot(nrm, to_light / max(d, 1e-4)), 0.0);
        sum = sum + light.points[2u * i + 1u].rgb * (locallight::falloff(d, pos_range.w) * nl);
    }
    return sum;
}

// `gi_trace.wgsl`'s `fib_dir`, `rotate`, `bounce_at`, `listed_count` and `ray_slot`, unchanged: the
// sum stage recomputes the same directions to project these rays.
fn fib_dir(i: u32, n: f32) -> vec3<f32> {
    let k = f32(i) + 0.5;
    let z = 1.0 - 2.0 * k / n;
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = k * 2.399963;
    return vec3<f32>(r * cos(phi), z, r * sin(phi));
}

fn rotate(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(trace.rot_x.xyz, d), dot(trace.rot_y.xyz, d), dot(trace.rot_z.xyz, d));
}

fn bounce_at(g: probegi::GiParams, p: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
    let found = probegi::nearest_index(g, p);
    if (found.y < 0.5) {
        return vec3<f32>(0.0);
    }
    let probe = modern_block.probes[u32(found.x)];
    if (probe.dc.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let e = probe.dc.rgb + probe.sh_x.rgb * nrm.x + probe.sh_y.rgb * nrm.y + probe.sh_z.rgb * nrm.z;
    return max(e, vec3<f32>(0.0)) * trace.tune.w;
}

fn listed_count() -> u32 {
    return min(work.count, probegi::listed_cap(trace.gi));
}

fn ray_slot(r: u32, item: u32, count: u32) -> u32 {
    return 2u * (r * count + item);
}

// What a probe ray found, in `probegi::GiHit`'s terms.
struct RtHit {
    // 0 = the ray reached the sky, 1 = it hit the static world, 2 = it hit terrain.
    kind: f32,
    dist: f32,
    pos: vec3<f32>,
    albedo: vec3<f32>,
    // The interpolated vertex normal in world space, as authored (not turned to the ray): the voxel
    // normal is the same vertex normal, so a ray that leaves a surface by its back counts as one.
    nrm: vec3<f32>,
    // 1 on the region's cutout geometry (foliage, leaf cards): lit through, never back-faced.
    two_sided: f32,
}

// The nearest blocking surface along the ray, shaded as far as the voxel path shades a voxel.
fn cast_ray(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, reach: f32) -> RtHit {
    let hit = rt_scene::trace(origin, dir, t_min, reach, rt_scene::MASK_ALL);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return RtHit(0.0, reach, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 0.0);
    }
    let inst = rt_scene::instance(hit.instance_custom_data);
    let v = rt_scene::triangle(inst, hit.geometry_index, hit.primitive_index);
    let w = rt_scene::weights(hit.barycentrics);
    let local_n = rt_scene::vertex_normal(v.x) * w.x
        + rt_scene::vertex_normal(v.y) * w.y
        + rt_scene::vertex_normal(v.z) * w.z;
    // object_to_world is mat4x3: the product is the world-space vector.
    var nrm = hit.object_to_world * vec4<f32>(local_n, 0.0);
    nrm = select(-dir, normalize(nrm), dot(nrm, nrm) > 1e-8);
    let pos = origin + dir * hit.t;
    if (inst.kind == rt_scene::KIND_TERRAIN) {
        // The splat set is not in the scene, nor in the clipmap: the ground albedo stands in for it
        // on both paths (`trace_rays` reads `g.ground`).
        return RtHit(2.0, hit.t, pos, vec3<f32>(0.0), nrm, 0.0);
    }
    let word = rt_scene::vertex_word(v.x);
    var albedo = UNTEXTURED;
    if ((word & rt_scene::WORD_TEXTURED) != 0u) {
        let uv = rt_scene::vertex_uv(v.x) * w.x + rt_scene::vertex_uv(v.y) * w.y + rt_scene::vertex_uv(v.z) * w.z;
        // The pool's arrays are sRGB views, so the sample arrives linear; the probes light in the
        // authored value space, as `static_gx_voxel.wgsl` writes the voxel.
        albedo = colorspace::linear_to_srgb(rt_scene::texel(inst.row, word, uv, ALBEDO_MIP).rgb);
    }
    let two_sided = select(0.0, 1.0, hit.geometry_index == inst.cutout_geometry);
    return RtHit(1.0, hit.t, pos, albedo, nrm, two_sided);
}

// One ray per lane, ray-major across consecutive probes: `gi_trace.wgsl`'s `trace_rays`, with the
// ray query in place of `probegi::march`. Its output rows are the voxel stage's, so `sum_probes`
// and the blend read them unchanged.
@compute @workgroup_size(64)
fn trace_rays(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let g0 = trace.gi;
    if (g0.params.x < 0.5) {
        return;
    }
    let rays = clamp(u32(g0.misc.x), 1u, LANES);
    let count = listed_count();
    let global = (group_id.y * groups.x + group_id.x) * LANES + lane;
    if (count == 0u || global >= count * rays) {
        return;
    }
    let ray = global / count;
    let item = global % count;
    let setup = setups[item];
    var g = g0;
    g.sky = vec4<f32>(light.rows[0].xyz, g.sky.w);
    near_count = setup.info.z;
    near_light = array<u32, 4>(setup.near.x, setup.near.y, setup.near.z, setup.near.w);
    let p = setup.pos.xyz;
    let valid = setup.pos.w;
    let interior_class = setup.cls.x;
    let to_sun = -normalize(light.rows[2].xyz);
    let key = light.rows[1].xyz * light.rows[2].w;

    let dir = rotate(fib_dir(ray, f32(rays)));
    var radiance = vec3<f32>(0.0);
    var sky_seen = 0.0;
    var backface = 0.0;
    var hit_dist = 0.0;
    if (valid > 0.0) {
        // The same start offset the march takes, as the near end of the query.
        let hit = cast_ray(p, dir, trace.tune.z, max(trace.clip.info.w, trace.tune.z + 1.0));
        hit_dist = hit.dist;
        if (hit.kind < 0.5) {
            radiance = probegi::miss_radiance(g, dir, interior_class);
            sky_seen = 1.0;
        } else {
            let albedo = select(hit.albedo, g.ground.xyz, hit.kind > 1.5);
            var nrm = hit.nrm;
            let facing = dot(nrm, -dir);
            if (hit.two_sided > 0.5) {
                // Foliage is lit through: its shading normal turns to meet the ray. No gap share
                // as the voxel path's `FOLIAGE_GAP`: a ray here passes between the leaves where
                // their texels do, and what it hit is a leaf.
                nrm = select(nrm, -nrm, facing < 0.0);
            } else if (facing < 0.0) {
                backface = 1.0;
            }
            let lift = hit.pos + nrm * trace.tune.y;
#ifdef RT_SHADOWS
            // fork: F — the cascades hold the entities alone; one ray takes the static world and the
            // terrain (`benilla_modern::rtshadow`, the scene at group 1).
            let sun_vis = rtshadow::sun_visibility_anywhere(
                shadow_map,
                shadow_samp,
                modern_block.cascades,
                lift,
                to_sun,
            );
#else
            let sun_vis = sunshadow::sun_visibility_anywhere(
                shadow_map,
                shadow_samp,
                modern_block.cascades,
                lift,
            );
#endif
            let direct = key * (max(dot(nrm, to_sun), 0.0) * sun_vis);
            radiance = albedo * (direct + local_light_at(lift, nrm) + bounce_at(g, lift, nrm));
        }
    }
    let at = ray_slot(ray, item, count);
    ray_out[at] = vec4<f32>(radiance, sky_seen);
    ray_out[at + 1u] = vec4<f32>(backface, hit_dist, 0.0, 0.0);
}
