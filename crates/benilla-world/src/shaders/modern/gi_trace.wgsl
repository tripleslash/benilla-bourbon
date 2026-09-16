// Modern lighting (FORK.md, P5): the probe trace — one invocation per probe, a
// Fibonacci sphere of rays marched through the voxel clipmap and the terrain heightfield, projected
// onto an order-1 SH lobe and left in the scratch buffer for `gi_blend.wgsl` to fold in.
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

// P6: the local lights near a probe, picked once and then evaluated at every ray's hit — walking the
// whole table per hit would be 24 rays x 256 lights a probe.
const NEAR_LIGHTS: u32 = 4u;
// The share of a ray that passes a foliage voxel (two-sided, alpha-tested) to whatever lies beyond.
const FOLIAGE_GAP: f32 = 0.45;
var<private> near_light: array<u32, 4>;
var<private> near_count: u32;

fn pick_near_lights(p: vec3<f32>) {
    near_count = 0u;
    var far = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    let count = min(u32(light.rows[20].x), 256u);
    for (var i = 0u; i < count; i = i + 1u) {
        let pos_range = light.points[2u * i];
        let dv = pos_range.xyz - p;
        let d2 = dot(dv, dv);
        // A probe gathers from a light whose reach covers it with a little slack, so a lamp just
        // outside the probe still lights the wall the probe is measuring.
        let reach = pos_range.w + 8.0;
        if (d2 > reach * reach) {
            continue;
        }
        var slot = near_count;
        for (var k = 0u; k < near_count; k = k + 1u) {
            if (d2 < far[k]) {
                slot = k;
                break;
            }
        }
        if (slot >= NEAR_LIGHTS) {
            continue;
        }
        var k = min(near_count, NEAR_LIGHTS - 1u);
        while (k > slot) {
            far[k] = far[k - 1u];
            near_light[k] = near_light[k - 1u];
            k = k - 1u;
        }
        far[slot] = d2;
        near_light[slot] = i;
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

@compute @workgroup_size(64)
fn trace_probes(@builtin(global_invocation_id) id: vec3<u32>) {
    var g = trace.gi;
    if (g.params.x < 0.5) {
        return;
    }
    // The sky's radiance and the key are read from the rows the surface lanes light with, never from
    // a second copy of them: a probe has to substitute for the ambient row exactly.
    g.sky = vec4<f32>(light.rows[0].xyz, g.sky.w);
    let per_cascade = u32(g.params.y);
    let total = per_cascade * u32(g.dims.w);
    let idx = id.x;
    if (idx >= total) {
        return;
    }
    let cascade = probegi::logical_cascade(g, idx / per_cascade);
    let slot = i32(idx % per_cascade);
    let n = vec3<i32>(g.dims.xyz);
    let texel = vec3<i32>(slot % n.x, (slot / n.x) % n.y, slot / (n.x * n.y));
    // The world probe this slot currently holds (the grid is toroidal).
    let base = vec3<i32>(g.grid[cascade].xyz);
    let w = base + vec3<i32>(
        probegi::wrap(texel.x - base.x, n.x),
        probegi::wrap(texel.y - base.y, n.y),
        probegi::wrap(texel.z - base.z, n.z),
    );

    let previous = modern_block.probes[idx];
    let scrolled = previous.sh_y.w != probegi::key_lo(w) || previous.sh_z.w != probegi::key_hi(w);
    // Never traced — not "invalid": an invalid probe used to count as cold, so it was re-traced
    // every frame under a new ray rotation, flipping between valid and invalid (and between classes)
    // frame to frame. That flip was the blinking on trunks, canopies and interior walls.
    let cold = previous.depth.w < 0.5;
    let reset = trace.cam.w > 0.5;
    let period = max(u32(g.misc.y), 1u);
    let scheduled = ((idx * 7u + u32(g.misc.z)) % period) == 0u;
    if (!scrolled && !cold && !reset && !scheduled) {
        scratch[idx].sh_y.w = 0.0;
        return;
    }

    let p = probegi::probe_pos(g, cascade, w);
    let start_level = min(cascade, u32(trace.clip.info.x) - 1u);
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
    let seen_before = select(previous.sh_x.w, 1.0, cold || scrolled || reset);
    let interior_class = (1.0 - smoothstep(0.0, 0.25, sky_here))
        * (1.0 - smoothstep(0.02, 0.20, seen_before));
    pick_near_lights(p);

    let to_sun = -normalize(light.rows[2].xyz);
    let key = light.rows[1].xyz * light.rows[2].w;
    let rays = max(u32(g.misc.x), 1u);
    let inv_rays = 1.0 / f32(rays);
    var dc = vec3<f32>(0.0);
    var sh_x = vec3<f32>(0.0);
    var sh_y = vec3<f32>(0.0);
    var sh_z = vec3<f32>(0.0);
    var sky_seen = 0.0;
    var backfaces = 0.0;
    var hit_dist = 0.0;
    var hit_dist_sq = 0.0;

    for (var i = 0u; i < rays; i = i + 1u) {
        let dir = rotate(fib_dir(i, f32(rays)));
        var radiance = vec3<f32>(0.0);
        if (valid > 0.0) {
            let hit = probegi::march(
                vox_albedo,
                vox_normal,
                hf_map,
                trace.clip,
                p + dir * trace.tune.z,
                dir,
                start_level,
            );
            hit_dist = hit_dist + hit.dist;
            hit_dist_sq = hit_dist_sq + hit.dist * hit.dist;
            if (hit.kind < 0.5) {
                radiance = probegi::miss_radiance(g, dir, interior_class);
                sky_seen = sky_seen + 1.0;
            } else {
                let albedo = select(hit.albedo, g.ground.xyz, hit.kind > 1.5);
                var nrm = hit.nrm;
                let facing = dot(nrm, -dir);
                var gap = 0.0;
                if (hit.two_sided > 0.5) {
                    // Foliage is lit through: its shading normal turns to meet the ray.
                    nrm = select(nrm, -nrm, facing < 0.0);
                    // A leaf voxel is not a solid wall: a share of the ray slips between the
                    // leaves to the sky. Without it a canopy reads as a roof and the ground
                    // under a tree goes black.
                    gap = FOLIAGE_GAP;
                } else if (facing < 0.0) {
                    backfaces = backfaces + 1.0;
                }
                let lift = hit.pos + nrm * trace.tune.y;
                let depth = dot(lift - trace.cam.xyz, trace.fwd.xyz);
                let sun_vis = sunshadow::sun_visibility_fast(
                    shadow_map,
                    shadow_samp,
                    modern_block.cascades,
                    lift,
                    depth,
                );
                let direct = key * (max(dot(nrm, to_sun), 0.0) * sun_vis);
                // P6: the local lights too, so a room whose only light is a hearth still bounces.
                radiance = albedo * (direct + local_light_at(lift, nrm) + bounce_at(g, lift, nrm));
                radiance = mix(radiance, probegi::miss_radiance(g, dir, interior_class), gap);
                sky_seen = sky_seen + gap;
            }
        }
        dc = dc + radiance;
        sh_x = sh_x + radiance * dir.x;
        sh_y = sh_y + radiance * dir.y;
        sh_z = sh_z + radiance * dir.z;
    }

    // A probe most of whose rays leave a surface by its back is inside geometry. Soft around the
    // limit: with 24 rotating rays the share jitters, and a hard cut turned that jitter into a probe
    // popping in and out of the field.
    let back_share = backfaces * inv_rays;
    valid = valid * (1.0 - smoothstep(trace.tune.x * 0.7, trace.tune.x * 1.3, back_share));
    var out: probegi::GiProbe;
    // E(n) / pi = dc + sh . n, from a uniform-sphere estimate of the radiance field.
    out.dc = vec4<f32>(dc * inv_rays, valid);
    out.sh_x = vec4<f32>(sh_x * (2.0 * inv_rays), sky_seen * inv_rays);
    out.sh_y = vec4<f32>(sh_y * (2.0 * inv_rays), select(1.0, 2.0, scrolled || cold || reset));
    out.sh_z = vec4<f32>(sh_z * (2.0 * inv_rays), 0.0);
    // P6: the two distance moments the Chebyshev test reads, and the class the receivers match on.
    out.depth = vec4<f32>(hit_dist * inv_rays, hit_dist_sq * inv_rays, interior_class, 1.0);
    scratch[idx] = out;
}
