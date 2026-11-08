enable wgpu_ray_query;

// fork: ray-traced shadows (`modern::rt_shadow`, native only): the sun's and the shadowed lamps'
// visibility through the static world and the terrain, as ray queries against
// `benilla_modern::rt_scene`'s TLAS with its cutout alpha test. At this lane the cascades and the
// lamp atlas hold the entity casters alone (`static_gx::render::modern_shadow`), so each entry here
// is the product of two terms: rays for what the TLAS holds, the maps for what it does not.
//
// A receiver calls the twin of the `sunshadow` / `locallight` function it called before, under
// `#ifdef RT_SHADOWS`; its pipeline binds the scene at `RT_GROUP` (`rt_scene::shader_defs`). The
// pipeline is the same whether the ray tracing is on or off: each twin asks `rt_scene::live()`
// first, and on the stand-in (the option off, or the scene not ready yet: `RtScene::lane_bind`) it
// returns exactly what the original returns, with the maps holding the static casters again. The
// importer declares no ray-query type itself, so it needs no `enable` of its own: the composer's
// header for this module carries the scene's acceleration structure, and the directive with it.
//
// Imported as `benilla_modern::rtshadow`, a name no hooked shader uses as an identifier, and with no
// identifier in it ending in a digit (a composable module may not carry those).
#define_import_path benilla_modern::rtshadow

#import benilla_modern::rt_scene
#import benilla_modern::sunshadow
#import benilla_modern::locallight

const TAU: f32 = 6.2831853;
// The angular radius (radians) of the disc the sun's penumbra rays aim at: twice the sun's own
// (0.27 degrees), the softness the cascades' five-tap filter gave a crown's shadow at mid range,
// and like a real sun's it widens with the distance to the occluder (contact-hardening).
const SUN_CONE: f32 = 0.0093;
// How far toward the sun a ray looks (yd): `modern::shadow::CASTER_REACH`, the cascades' reach.
const SUN_REACH: f32 = 1200.0;
// The receiver's lift off its own surface (yd), and its growth per yard of view distance: a far
// fragment's position is coarser, and a self-hit there reads as acne.
const BIAS_YD: f32 = 0.04;
const BIAS_PER_YD: f32 = 0.001;
// The cover ring's extra lift (yd) off the receiver's tangent plane, so the ground beside a point on
// a curved slope is not taken for cover.
const COVER_LIFT_YD: f32 = 0.25;

// A per-receiver rotation in [0, TAU): the penumbra rays' pattern turns from one point to the next,
// so four rays read as a dithered gradient and not as bands. `locallight::local_visibility`'s hash,
// wrapped first for the same precision reason, so the lamps' noise and the sun's agree.
fn spin(p: vec3<f32>) -> f32 {
    let wrapped = fract(p * 0.1);
    return fract(dot(wrapped, vec3<f32>(12.9898, 78.233, 37.719))) * TAU;
}

// Two unit vectors spanning the plane across `axis` (columns).
fn across(axis: vec3<f32>) -> mat2x3<f32> {
    let side = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
    let ua = normalize(cross(axis, side));
    let ub = cross(axis, ua);
    return mat2x3<f32>(ua, ub);
}

// The receiver's normal turned to the side `toward` lies on: a ray then never starts behind a
// two-sided leaf that faces away, and a wall facing away starts inside itself and is shaded, as its
// N.L already says.
fn facing(n: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    return select(n, -n, dot(n, toward) < 0.0);
}

// The receiver lifted off its own surface toward `toward`. A zero normal (a point in the air) stays.
fn lift(p: vec3<f32>, n: vec3<f32>, toward: vec3<f32>, view_distance: f32) -> vec3<f32> {
    return p + facing(n, toward) * (BIAS_YD + view_distance * BIAS_PER_YD);
}

// 1 where the sun reaches `origin` along `dir` past everything the scene holds, else 0.
fn sun_ray(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    return select(1.0, 0.0, rt_scene::occluded(origin, dir, 0.0, SUN_REACH, rt_scene::MASK_ALL));
}

// The sun's visibility at a receiver through the static world and the terrain. `soft`: four rays
// toward points of a ring at 0.7 of `SUN_CONE` (a uniform disc's mean radius) around the sun, the
// two opposite ones first; where those agree the point is taken as wholly lit or wholly shaded, so
// only a penumbra pays for the other two. Not `soft`: the centre ray alone.
fn sun_rays(p: vec3<f32>, n: vec3<f32>, to_sun: vec3<f32>, view_distance: f32, soft: bool) -> f32 {
    let origin = lift(p, n, to_sun, view_distance);
    if (!soft) {
        return sun_ray(origin, to_sun);
    }
    let basis = across(to_sun);
    let turn = spin(p);
    let oa = vec2<f32>(cos(turn), sin(turn)) * (SUN_CONE * 0.7);
    let ob = vec2<f32>(-oa.y, oa.x);
    let va = sun_ray(origin, normalize(to_sun + basis * oa));
    let vb = sun_ray(origin, normalize(to_sun - basis * oa));
    if (va == vb) {
        return va;
    }
    let vc = sun_ray(origin, normalize(to_sun + basis * ob));
    let vd = sun_ray(origin, normalize(to_sun - basis * ob));
    return 0.25 * (va + vb + vc + vd);
}

// How open the cover toward the sun is around a receiver (1 = nothing within
// `sunshadow::COVER_RADIUS_YD` shades it): `sunshadow::sun_cover_open`'s ring as three rays toward
// the sun from a ring that wide in the receiver's own tangent plane, turned per point.
fn cover_rays(p: vec3<f32>, n: vec3<f32>, to_sun: vec3<f32>, view_distance: f32) -> f32 {
    let face = facing(n, to_sun);
    let base = p + face * (BIAS_YD + view_distance * BIAS_PER_YD + COVER_LIFT_YD);
    let basis = across(face);
    let turn = spin(p) + 0.7;
    var open = 0.0;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let a = turn + f32(k) * (TAU / 3.0);
        let at = base + basis * (vec2<f32>(cos(a), sin(a)) * sunshadow::COVER_RADIUS_YD);
        open = open + sun_ray(at, to_sun);
    }
    return open / 3.0;
}

// The cascades' share, at this lane the entity casters alone, with no far term past them: the static
// cover the sky map's far term stood in for is the rays' now. The last cascade fades to lit instead
// of into the far field.
fn entity_cascades(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: sunshadow::ModernCascades,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_distance: f32,
    lite: bool,
) -> f32 {
    let count = u32(c.params.x);
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_distance < c.splits[i]) {
            let v = sunshadow::cascade_visibility(map, samp, c, i, world_pos, normal, lite);
            if (i + 1u == count) {
                return mix(v, 1.0, smoothstep(c.splits[i] * 0.7, c.splits[i], view_distance));
            }
            return v;
        }
    }
    return 1.0;
}

// `sunshadow::sun_visibility` at this lane: rays times the entity cascades, under the same shade
// floor, whose cover read takes the rays' ring for the static world and the cascades' ring for the
// entities. `to_sun` is the unit direction toward the key light.
fn sun_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: sunshadow::ModernCascades,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_distance: f32,
    lite: bool,
    to_sun: vec3<f32>,
) -> f32 {
    if (!rt_scene::live()) {
        return sunshadow::sun_visibility(map, samp, c, world_pos, normal, view_distance, lite);
    }
    // No cascades this frame is no modern shadow this frame, as `sunshadow` reads it.
    if (u32(c.params.x) == 0u) {
        return 1.0;
    }
    let ent = entity_cascades(map, samp, c, world_pos, normal, view_distance, lite);
    let ray = sun_rays(world_pos, normal, to_sun, view_distance, !lite);
    let v = ent * ray;
    var open = 0.5;
    if (v < 1.0 && !lite) {
        open = 1.0;
        if (ray < 1.0) {
            open = cover_rays(world_pos, normal, to_sun, view_distance);
        }
        if (ent < 1.0) {
            open = open * sunshadow::sun_cover_open(map, samp, c, world_pos, normal, view_distance);
        }
    }
    return mix(mix(sunshadow::SUN_SHADE_DEEP, sunshadow::SUN_SHADE_OPEN, open), 1.0, v);
}

// `sunshadow::sun_visibility_anywhere` at this lane (the probe trace's hits): the finest cascade
// whose map holds the point for the entities, times one ray. `world_pos` is already lifted.
fn sun_visibility_anywhere(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: sunshadow::ModernCascades,
    world_pos: vec3<f32>,
    to_sun: vec3<f32>,
) -> f32 {
    if (!rt_scene::live()) {
        return sunshadow::sun_visibility_anywhere(map, samp, c, world_pos);
    }
    let count = u32(c.params.x);
    if (count == 0u) {
        return 1.0;
    }
    var ent = 1.0;
    let margin = c.params.y;
    for (var i = 0u; i < count; i = i + 1u) {
        let clip = sunshadow::cascade_clip(c, i) * vec4<f32>(world_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (all(uv >= vec2<f32>(margin)) && all(uv <= vec2<f32>(1.0 - margin)) && ndc.z >= 0.0) {
            ent = textureSampleCompareLevel(map, samp, uv, i32(i), ndc.z + c.depth_bias[i] * 2.0);
            break;
        }
    }
    return ent * sun_ray(world_pos, to_sun);
}

// `sunshadow::sun_visibility_fast` at this lane (the froxels): the cascade tap, which holds the
// entities (and the static world too where the cascades still draw it), times one ray from the
// point in the air. No far term: past the cascades the ray alone answers.
fn sun_visibility_fast(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: sunshadow::ModernCascades,
    world_pos: vec3<f32>,
    view_distance: f32,
    to_sun: vec3<f32>,
) -> f32 {
    if (!rt_scene::live()) {
        return sunshadow::sun_visibility_fast(map, samp, c, world_pos, view_distance);
    }
    let count = u32(c.params.x);
    if (count == 0u) {
        return 1.0;
    }
    var ent = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_distance < c.splits[i]) {
            let clip = sunshadow::cascade_clip(c, i) * vec4<f32>(world_pos, 1.0);
            let ndc = clip.xyz / clip.w;
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
                ent = textureSampleCompareLevel(map, samp, uv, i32(i), ndc.z + c.depth_bias[i] * 2.0);
            }
            break;
        }
    }
    return ent * sun_ray(world_pos, to_sun);
}

// One ray's share alone (the water, whose cascade read `sunshadow::sun_visibility_load` already
// fades to lit past the cascades): 1 with no cascades this frame, as every `sunshadow` read, and 1
// off the scene (the cascades then hold the static casters themselves). `world_pos` is already
// lifted.
fn sun_visibility_ray(c: sunshadow::ModernCascades, world_pos: vec3<f32>, to_sun: vec3<f32>) -> f32 {
    if (!rt_scene::live() || u32(c.params.x) == 0u) {
        return 1.0;
    }
    return sun_ray(world_pos, to_sun);
}

// Where a ray from `origin` along `dir` toward a point `dist` away by a lamp at `light` stops: short
// of the lamp's own fixture box (`ShadowLight::clear`, half-extent `clear` about the light, 0 =
// none), whose glass and frame the atlas never drew either (`FIXTURE_CLIP`), and never nearer the
// light than the cube's near plane `near`, inside which the atlas saw nothing.
fn fixture_entry(
    origin: vec3<f32>,
    dir: vec3<f32>,
    dist: f32,
    light: vec3<f32>,
    clear: f32,
    near: f32,
) -> f32 {
    var stop = dist - near;
    if (clear > 0.0) {
        let safe = select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
        let inv = vec3<f32>(1.0) / safe;
        let ta = (light - vec3<f32>(clear) - origin) * inv;
        let tb = (light + vec3<f32>(clear) - origin) * inv;
        let lo = min(ta, tb);
        stop = min(stop, max(max(lo.x, lo.y), lo.z));
    }
    return stop;
}

// 1 where nothing the scene holds stands between `origin` and `aim` (a point on a lamp at
// `light`), else 0.
fn lamp_ray(origin: vec3<f32>, aim: vec3<f32>, light: vec3<f32>, clear: f32, near: f32) -> f32 {
    let to = aim - origin;
    let dist = length(to);
    if (dist <= 1e-4) {
        return 1.0;
    }
    let dir = to / dist;
    let stop = fixture_entry(origin, dir, dist, light, clear, near);
    if (stop <= 0.0) {
        return 1.0;
    }
    return select(1.0, 0.0, rt_scene::occluded(origin, dir, 0.0, stop, rt_scene::MASK_ALL));
}

// A lamp's visibility at a receiver through the static world and the terrain. `taps > 1` (the
// tier's soft filter): four rays toward a ring at 0.7 of the emitter's radius `emitter` across the
// light, two first as the sun's; else the one ray to the light. A zero `normal` (a froxel) starts at
// the point itself. 1 off the scene.
fn lamp_rays(
    p: vec3<f32>,
    normal: vec3<f32>,
    light: vec3<f32>,
    clear: f32,
    near: f32,
    emitter: f32,
    taps: u32,
) -> f32 {
    if (!rt_scene::live()) {
        return 1.0;
    }
    let to = light - p;
    let dist = length(to);
    if (dist <= near) {
        return 1.0;
    }
    let dir = to / dist;
    let origin = lift(p, normal, dir, 0.0);
    if (taps <= 1u || emitter <= 0.0) {
        return lamp_ray(origin, light, light, clear, near);
    }
    let basis = across(dir);
    let turn = spin(p) + 1.3;
    let oa = vec2<f32>(cos(turn), sin(turn)) * (emitter * 0.7);
    let ob = vec2<f32>(-oa.y, oa.x);
    let va = lamp_ray(origin, light + basis * oa, light, clear, near);
    let vb = lamp_ray(origin, light - basis * oa, light, clear, near);
    if (va == vb) {
        return va;
    }
    let vc = lamp_ray(origin, light + basis * ob, light, clear, near);
    let vd = lamp_ray(origin, light - basis * ob, light, clear, near);
    return 0.25 * (va + vb + vc + vd);
}

// `locallight::local_visibility` at this lane: the atlas (the entity casters alone) times the rays,
// under the tier's shade floor and the slot's cross-fade. `anchor` is where the slot's cube was
// rendered (`light_shadow[slot]`), `light` where the light is this frame, `clear` the fixture box
// (`light_spare` rows 9-11).
fn local_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    g: locallight::LightGrid,
    slot: u32,
    anchor: vec3<f32>,
    light: vec3<f32>,
    clear: f32,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    near: f32,
    fade: f32,
) -> f32 {
    if (!rt_scene::live()) {
        return locallight::local_visibility(map, samp, g, slot, anchor, world_pos, normal, near, fade);
    }
    if (slot >= locallight::MAX_SHADOWED || !locallight::atlas_live(g) || fade <= 0.0) {
        return 1.0;
    }
    // Negative: the atlas's blocker search found nothing in front of the receiver (wholly lit).
    let raw = locallight::local_visibility_raw(map, samp, g, slot, anchor, world_pos, normal, near);
    let atlas = select(raw, 1.0, raw < 0.0);
    let rays = lamp_rays(world_pos, normal, light, clear, near, g.soft.y, u32(g.soft.x));
    let shaded = mix(1.0 - g.soft.z, 1.0, atlas * rays);
    return mix(1.0, shaded, fade);
}
