// Modern lighting (FORK.md, P5): the probe trace — a Fibonacci sphere of rays per
// probe (one lane each since P8b, see `trace_probes`) marched through the voxel clipmap and the
// terrain heightfield, projected onto an order-1 SH lobe and left in the scratch buffer for
// `gi_blend.wgsl` to fold in.
//
// A ray's radiance is the sky where it reaches it, and at a surface the voxel's albedo times the key
// light under the cascades plus the previous frame's probe field (the second bounce). The whole
// trace runs in the shared light's own value space, so a probe's result substitutes for the ambient
// row exactly where the lanes read it.
#import benilla_modern::probegi
#import benilla_modern::sunshadow
#import benilla_modern::locallight

// The shared light buffer as this pass reads it (`lighting::global_light`): header rows 0 ambient,
// 1 diffuse, 2 sun, 20 `.x` the live point count — then the point table, which P6's bounce needs so
// that a lamp-lit room bounces its lamps and not only the sun it cannot see.
struct LightRows {
    rows: array<vec4<f32>, 21>,
    points: array<vec4<f32>, 512>,
}

struct GiTrace {
    gi: probegi::GiParams,
    clip: probegi::GiClip,
    // xyz = the camera, w = reset (1 = every probe is traced and replaced this frame).
    cam: vec4<f32>,
    // xyz = the camera's forward (the cascade lookup's planar depth), w = the history weight.
    fwd: vec4<f32>,
    // This frame's ray rotation, row by row.
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
@group(0) @binding(4) var vox_albedo: texture_3d<f32>;
@group(0) @binding(5) var vox_normal: texture_3d<f32>;
@group(0) @binding(6) var hf_map: texture_depth_2d;
@group(0) @binding(7) var<uniform> trace: GiTrace;
@group(0) @binding(8) var<storage, read_write> scratch: array<probegi::GiProbe>;

// This frame's work list (`gi_classify.wgsl`): every probe due a trace, packed. The dispatch is
// indirect — one workgroup per `group_probes(rays)` listed probes, one lane per ray.
struct GiWork {
    count: u32,
    items: array<u32>,
}
@group(0) @binding(9) var<storage, read> work: GiWork;

// fork: what the setup stage leaves for a listed probe (`setup_probes`), by its place in the list.
struct GiSetup {
    // xyz = the probe's position, w = its validity before the backface test.
    pos: vec4<f32>,
    // x = its probe index, y = the level its rays start in, z = its near lights' count, w = 1 when
    // it is fresh (scrolled in, never traced, or a cut).
    info: vec4<u32>,
    near: vec4<u32>,
    // x = its interior class.
    cls: vec4<f32>,
}
@group(0) @binding(10) var<storage, read_write> setups: array<GiSetup>;
// fork: every ray's contribution (`trace_rays`), two rows a ray, ray-major: `rgb` = the radiance
// and `w` = the share that reached the sky; then x = whether it left a surface by its back and y =
// its hit distance.
@group(0) @binding(11) var<storage, read_write> ray_out: array<vec4<f32>>;

// P6: the local lights near a probe, picked once and then evaluated at every ray's hit — walking the
// whole table per hit would be 24 rays x 256 lights a probe.
const NEAR_LIGHTS: u32 = 4u;
// The share of a ray that passes a foliage voxel (two-sided, alpha-tested) to whatever lies beyond.
const FOLIAGE_GAP: f32 = 0.45;
var<private> near_light: array<u32, 4>;
var<private> near_count: u32;

// fork: P8b — **one thread per RAY, not one per probe**, and since then **a warp per ray
// DIRECTION, not per probe.**
//
// The trace used to give each probe one thread and loop it over all of that probe's rays; P8b gave
// each ray a lane, a probe a workgroup. But a warp runs as long as its longest lane, and a probe's
// rays span the sphere: in nearly every probe one ray runs level and walks its whole step budget
// while its neighbours hit the ground in a handful, so every warp paid for its slowest direction.
// Measured at the worldview's pose, a quarter of the step budget took the pass from 6.0 to 2.7 ms,
// and early-outs that shortened the average ray moved nothing.
//
// So the work is three stages over the same list and the same indirect dispatch:
//
// - `setup_probes`: per probe, as before — its lights (every lane of a probe testing its share of
//   the table, the first lane taking the four nearest in table order), its validity and its class,
//   left in `setups`.
// - `trace_rays`: ONE RAY INDEX ACROSS CONSECUTIVE PROBES per warp — the same direction (one
//   rotation for the frame), so lanes that start together mostly end together.
// - `sum_probes`: per probe, its rays added **in ray order**, exactly as the old loop added them, so
//   a probe comes out as it did. The SH weightings are the ray's radiance times its direction,
//   recomputed from the same direction with the same arithmetic.
const LANES: u32 = 64u;
// Probes one workgroup can hold — the per-probe shared arrays below are sized for it, and a tier
// with fewer than 16 rays simply leaves lanes idle rather than overflow them.
const MAX_GROUP_PROBES: u32 = 4u;
// The shared light table's capacity (`lighting::global_light::MAX_POINT_LIGHTS`).
const MAX_LIGHTS: u32 = 256u;
// Per probe of the group: every table light's squared distance, or -1 when out of reach.
var<workgroup> wg_light_d2: array<f32, 1024>;

// Probes per workgroup for `rays` rays a probe. The classify pass sizes the indirect dispatch with
// the same arithmetic (`gi_classify.wgsl`), so the two can never disagree.
fn group_probes(rays: u32) -> u32 {
    return clamp(LANES / clamp(rays, 1u, LANES), 1u, MAX_GROUP_PROBES);
}

// The four nearest lights whose reach covers the probe, from the distances its lanes left in
// `wg_light_d2` — the old per-probe walk's selection, made in the same table order with the same
// strict comparisons, so it picks the same four.
fn pick_near_lights(slot: u32, count: u32) {
    near_count = 0u;
    var far = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    for (var i = 0u; i < count; i = i + 1u) {
        let d2 = wg_light_d2[slot * MAX_LIGHTS + i];
        if (d2 < 0.0) {
            continue;
        }
        var at = near_count;
        for (var k = 0u; k < near_count; k = k + 1u) {
            if (d2 < far[k]) {
                at = k;
                break;
            }
        }
        if (at >= NEAR_LIGHTS) {
            continue;
        }
        var k = min(near_count, NEAR_LIGHTS - 1u);
        while (k > at) {
            far[k] = far[k - 1u];
            near_light[k] = near_light[k - 1u];
            k = k - 1u;
        }
        far[at] = d2;
        near_light[at] = i;
        near_count = min(near_count + 1u, NEAR_LIGHTS);
    }
}

// What the probe's nearby lights put on a surface. Unshadowed on purpose: the cube atlas is a
// High-and-above luxury and the bounce it feeds is a soft term, so paying for it here would cost the
// trace six lookups a hit to move a second-order quantity.
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

// The golden-angle sphere: `n` rays as evenly spread as a closed form gets them.
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

// The previous frame's field at a hit, one tap: enough for a second bounce, and it costs one probe
// read instead of eight.
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

// This frame's listed probes, as far as the cap lets the trace read (`probegi::listed_cap`).
fn listed_count() -> u32 {
    return min(work.count, probegi::listed_cap(trace.gi));
}

// The ray a stage-two lane walks: its index in the list's ray-major order.
fn ray_slot(r: u32, item: u32, count: u32) -> u32 {
    return 2u * (r * count + item);
}

@compute @workgroup_size(64)
fn setup_probes(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Read straight off the uniform, so the early-out is uniform across the workgroup (the barrier
    // below requires it).
    if (trace.gi.params.x < 0.5) {
        return;
    }
    let g = trace.gi;
    let rays = clamp(u32(g.misc.x), 1u, LANES);
    let per_group = group_probes(rays);
    // Which of this workgroup's probes this lane serves, and which share of its light table.
    let slot = lane / rays;
    let ray = lane % rays;
    // The dispatch is rows of workgroups (`gi_classify.wgsl`'s `GROUP_ROW`).
    let item = (group_id.y * groups.x + group_id.x) * per_group + slot;
    let busy = slot < per_group && item < listed_count();

    var idx = 0u;
    var cascade = 0u;
    var w = vec3<i32>(0);
    var p = vec3<f32>(0.0);
    if (busy) {
        idx = work.items[item] & 0x7fffffffu;
        let per_cascade = u32(g.params.y);
        cascade = probegi::logical_cascade(g, idx / per_cascade);
        // The world probe this slot currently holds (the grid is toroidal).
        w = probegi::slot_probe(g, cascade, idx % per_cascade);
        p = probegi::probe_pos(g, cascade, w);
    }
    let count = min(u32(light.rows[20].x), MAX_LIGHTS);
    if (busy) {
        // This lane's share of the light table.
        for (var i = ray; i < count; i = i + rays) {
            let pos_range = light.points[2u * i];
            let dv = pos_range.xyz - p;
            let d2 = dot(dv, dv);
            // A probe gathers from a light whose reach covers it with a little slack, so a lamp
            // just outside the probe still lights the wall the probe is measuring.
            let reach = pos_range.w + 8.0;
            wg_light_d2[slot * MAX_LIGHTS + i] = select(d2, -1.0, d2 > reach * reach);
        }
    }
    workgroupBarrier();
    if (!busy || ray != 0u) {
        return;
    }
    let previous = modern_block.probes[idx];
    let scrolled = previous.sh_y.w != probegi::key_lo(w) || previous.sh_z.w != probegi::key_hi(w);
    // Never traced — not "invalid": an invalid probe used to count as cold, so it was re-traced
    // every frame under a new ray rotation, flipping between valid and invalid (and between classes)
    // frame to frame. That flip was the blinking on trunks, canopies and interior walls.
    let cold = previous.depth.w < 0.5;
    let reset = trace.cam.w > 0.5;
    let fresh = scrolled || cold || reset;
    // A probe under the terrain, or inside a wall, lights nothing: it is marked invalid and the
    // receivers around it fall back on their neighbours (and, with none, on the sky map).
    var valid = 1.0;
    if (p.y < probegi::terrain_height(hf_map, trace.clip, p) - 0.25) {
        valid = 0.0;
    }
    // P6: how enclosed this probe is. It decides two things — what a ray that reaches nothing sees
    // (the room's own ambient, not the sky), and which receivers may read this probe at all
    // (`probegi::accum`'s class match).
    //
    // TWO signals, because the top-down map alone is not enough: a forest canopy shades it as hard
    // as a roof does, so the map by itself would classify a wood as indoors and the class match
    // would then throw P5's canopy bounce away. What separates them is sideways: a probe under
    // leaves still gets rays out to the sky, a probe in a room does not, and the probe's own last
    // sky share is exactly that number. A probe with no history — cold, or just scrolled in —
    // counts as outdoors, which is the direction that self-corrects invisibly: a room reads one
    // update period too bright on entry, where the other way round a moving dark patch would follow
    // the camera through a wood.
    let sky_here = sunshadow::sky_visibility(shadow_map, shadow_samp, modern_block.cascades, p);
    let seen_before = select(previous.sh_x.w, 1.0, fresh);
    let interior_class = (1.0 - smoothstep(0.0, 0.25, sky_here))
        * (1.0 - smoothstep(0.02, 0.20, seen_before));
    pick_near_lights(slot, count);
    var out: GiSetup;
    out.pos = vec4<f32>(p, valid);
    out.info = vec4<u32>(
        idx,
        min(cascade, u32(trace.clip.info.x) - 1u),
        near_count,
        select(0u, 1u, fresh),
    );
    out.near = vec4<u32>(near_light[0], near_light[1], near_light[2], near_light[3]);
    out.cls = vec4<f32>(interior_class, 0.0, 0.0, 0.0);
    setups[item] = out;
}

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
    // Rows of workgroups (`gi_classify.wgsl`), every lane one ray: the same indirect dispatch the
    // setup stage ran, whose `group_probes` rows always hold at least `count * rays` lanes.
    let global = (group_id.y * groups.x + group_id.x) * LANES + lane;
    if (count == 0u || global >= count * rays) {
        return;
    }
    // Ray-major: consecutive lanes are consecutive probes under the same ray direction.
    let ray = global / count;
    let item = global % count;
    let setup = setups[item];
    var g = g0;
    // The sky's radiance and the key are read from the rows the surface lanes light with, never from
    // a second copy of them: a probe has to substitute for the ambient row exactly.
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
        let hit = probegi::march(
            vox_albedo,
            vox_normal,
            hf_map,
            trace.clip,
            p + dir * trace.tune.z,
            dir,
            setup.info.y,
        );
        hit_dist = hit.dist;
        if (hit.kind < 0.5) {
            radiance = probegi::miss_radiance(g, dir, interior_class);
            sky_seen = 1.0;
        } else {
            let albedo = select(hit.albedo, g.ground.xyz, hit.kind > 1.5);
            var nrm = hit.nrm;
            let facing = dot(nrm, -dir);
            var gap = 0.0;
            if (hit.two_sided > 0.5) {
                // Foliage is lit through: its shading normal turns to meet the ray.
                nrm = select(nrm, -nrm, facing < 0.0);
                // A leaf voxel is not a solid wall: a share of the ray slips between the leaves to
                // the sky. Without it a canopy reads as a roof and the ground under a tree goes
                // black.
                gap = FOLIAGE_GAP;
            } else if (facing < 0.0) {
                backface = 1.0;
            }
            let lift = hit.pos + nrm * trace.tune.y;
            // Placed by the map that holds it, never by where the camera looks: a hit behind the
            // player is as shaded as one in front (`sun_visibility_anywhere`).
            let sun_vis = sunshadow::sun_visibility_anywhere(
                shadow_map,
                shadow_samp,
                modern_block.cascades,
                lift,
            );
            let direct = key * (max(dot(nrm, to_sun), 0.0) * sun_vis);
            // P6: the local lights too, so a room whose only light is a hearth still bounces.
            radiance = albedo * (direct + local_light_at(lift, nrm) + bounce_at(g, lift, nrm));
            radiance = mix(radiance, probegi::miss_radiance(g, dir, interior_class), gap);
            sky_seen = gap;
        }
    }
    let at = ray_slot(ray, item, count);
    ray_out[at] = vec4<f32>(radiance, sky_seen);
    ray_out[at + 1u] = vec4<f32>(backface, hit_dist, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn sum_probes(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    if (trace.gi.params.x < 0.5) {
        return;
    }
    let rays = clamp(u32(trace.gi.misc.x), 1u, LANES);
    let count = listed_count();
    // Rows of workgroups, as the blend reads the same dispatch: one thread per listed probe.
    let item = id.y * (groups.x * LANES) + id.x;
    if (item >= count) {
        return;
    }
    let setup = setups[item];
    var dc = vec3<f32>(0.0);
    var sh_x = vec3<f32>(0.0);
    var sh_y = vec3<f32>(0.0);
    var sh_z = vec3<f32>(0.0);
    var sky_seen = 0.0;
    var backfaces = 0.0;
    var hit_dist = 0.0;
    var hit_dist_sq = 0.0;
    for (var r = 0u; r < rays; r = r + 1u) {
        let at = ray_slot(r, item, count);
        let a = ray_out[at];
        let b = ray_out[at + 1u];
        let dir = rotate(fib_dir(r, f32(rays)));
        dc = dc + a.rgb;
        sh_x = sh_x + a.rgb * dir.x;
        sh_y = sh_y + a.rgb * dir.y;
        sh_z = sh_z + a.rgb * dir.z;
        sky_seen = sky_seen + a.w;
        backfaces = backfaces + b.x;
        hit_dist = hit_dist + b.y;
        hit_dist_sq = hit_dist_sq + b.y * b.y;
    }

    let inv_rays = 1.0 / f32(rays);
    // A probe most of whose rays leave a surface by its back is inside geometry. Soft around the
    // limit: with 24 rotating rays the share jitters, and a hard cut turned that jitter into a probe
    // popping in and out of the field.
    let back_share = backfaces * inv_rays;
    let valid = setup.pos.w * (1.0 - smoothstep(trace.tune.x * 0.7, trace.tune.x * 1.3, back_share));
    var out: probegi::GiProbe;
    // E(n) / pi = dc + sh . n, from a uniform-sphere estimate of the radiance field.
    out.dc = vec4<f32>(dc * inv_rays, valid);
    out.sh_x = vec4<f32>(sh_x * (2.0 * inv_rays), sky_seen * inv_rays);
    out.sh_y = vec4<f32>(sh_y * (2.0 * inv_rays), select(1.0, 2.0, setup.info.w == 1u));
    out.sh_z = vec4<f32>(sh_z * (2.0 * inv_rays), 0.0);
    // P6: the two distance moments the Chebyshev test reads, and the class the receivers match on.
    out.depth = vec4<f32>(hit_dist * inv_rays, hit_dist_sq * inv_rays, setup.cls.x, 1.0);
    scratch[setup.info.x] = out;
}
