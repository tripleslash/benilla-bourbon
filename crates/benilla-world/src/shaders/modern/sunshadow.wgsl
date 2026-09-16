// Modern lighting (FORK.md, P2): sampling the modern sun's cascade shadow map.
//
// Imported as `benilla_modern::sunshadow` — a module name no hooked shader uses as an identifier
// (naga_oil substitutes a module alias for every bare identifier of that name).
//
// No bindings are declared here: terrain and wow_model bind the map in the material group,
// static_gx in its light group, so each receiver passes its own texture, sampler and cascade block.
#define_import_path benilla_modern::sunshadow

#import benilla_modern::froxelfog
#import benilla_modern::probegi
#import benilla_modern::locallight

// P4: the whole storage block every modern lane binds — the cascades, then the froxel fog parameters
// (`volumetrics::ModernVolume::std430`), then the cloud-shadow grid. Pass `.cascades` or `.fog` to the
// helpers, never the block: a by-value copy of the grid per call would be 4 KB.
struct ModernBlock {
    cascades: ModernCascades,
    fog: froxelfog::FroxelFog,
    // Sun visibility under the clouds per grid cell (1 = clear); `fog.cloud` places the grid.
    cloud: array<f32, 1024>,
    // P5: the probe field's parameters. `GiParams` is sized so the light grid behind it starts
    // 256-aligned, which is what lets the binning pass bind that region alone as writable.
    gi: probegi::GiParams,
    // P6: the local-light grid — parameters, the padding that keeps the probe array behind it
    // 256-aligned (the GI blend pass binds that one), then one cell per world bin. The cells are a
    // member of the BLOCK and not of `LightGrid`: a lane copies the parameters by value and must
    // never copy 864 KB of cells with them.
    lights: locallight::LightGrid,
    light_spare: array<vec4<f32>, 22>,
    light_cells: array<vec4<u32>, 55296>,
    // P5: the probe field itself, last because it is the block's only runtime-sized member.
    probes: array<probegi::GiProbe>,
}

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
    // texels, w = debug view (1 sun visibility, 2 sky visibility).
    params: vec4<f32>,
    // The sky-occlusion map (P3): the static world from straight above, in array layer 4.
    sky_clip: mat4x4<f32>,
    // x = enabled, y = texel world size, z = depth bias, w unused.
    sky_params: vec4<f32>,
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

// Past the last cascade the sun is not simply "lit": the top-down sky-occlusion map (layer 4, ±512 yd)
// still knows where a canopy or a roof stands, so a forest keeps its shade to the horizon instead of
// lighting up in a band that walks ahead of the camera. Soft — leaves let some sun through.
const FAR_SUN_UNDER_COVER: f32 = 0.3;

fn far_sun_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
) -> f32 {
    return mix(FAR_SUN_UNDER_COVER, 1.0, sky_visibility_fast(map, samp, c, world_pos));
}

// A surface in full shade still keeps this share of the key: leaves and edges scatter sunlight
// around, and a zero floor made a canopy floor read as night next to classic's 0.7 (which keeps
// nearly all of it). The froxels do NOT take it — the contrast between lit and shadowed air is the
// shafts.
const SUN_SHADE_FLOOR: f32 = 0.35;

fn sun_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_distance: f32,
) -> f32 {
    return mix(SUN_SHADE_FLOOR, 1.0, sun_visibility_raw(map, samp, c, world_pos, normal, view_distance));
}

// Sun visibility (1 = lit) at a world position seen `view_distance` yd in front of the camera.
fn sun_visibility_raw(
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
                // The last cascade fades into the far field instead of ending on a hard line.
                let fade = smoothstep(c.splits[i] * 0.7, c.splits[i], view_distance);
                if (fade <= 0.0) {
                    return v;
                }
                return mix(v, far_sun_visibility(map, samp, c, world_pos), fade);
            }
            return v;
        }
    }
    if (count == 0u) {
        return 1.0;
    }
    return far_sun_visibility(map, samp, c, world_pos);
}

// P3: the open-sky fraction above a point (1 = nothing overhead), from the top-down static map in
// array layer 4. A wide 3x3 compare, so a canopy dims the ground softly rather than as a stencil.
fn sky_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
) -> f32 {
    if (c.sky_params.x < 0.5) {
        return 1.0;
    }
    // Lifted a little, so the receiver's own surface never counts as its roof.
    let clip = c.sky_clip * vec4<f32>(world_pos + vec3<f32>(0.0, 0.75, 0.0), 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    let depth = ndc.z + c.sky_params.z;
    var open = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let o = vec2<f32>(f32(x), f32(y)) * (4.0 * c.params.y);
            open = open + textureSampleCompareLevel(map, samp, uv + o, 4, depth);
        }
    }
    return open / 9.0;
}

// P4: one hardware-compare tap per cascade — the froxels' visibility, where the volume's own blur
// replaces PCF.
fn sun_visibility_fast(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
    view_distance: f32,
) -> f32 {
    let count = u32(c.params.x);
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_distance < c.splits[i]) {
            let clip = c.clip_from_world[i] * vec4<f32>(world_pos, 1.0);
            let ndc = clip.xyz / clip.w;
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
                return 1.0;
            }
            return textureSampleCompareLevel(map, samp, uv, i32(i), ndc.z + c.depth_bias[i] * 2.0);
        }
    }
    if (count == 0u) {
        return 1.0;
    }
    return far_sun_visibility(map, samp, c, world_pos);
}

// Sun visibility with no comparison sampler: a manual bilinear compare over `textureLoad`, four
// offset footprints. For a layout that is already at the stage's sampler limit — the cata water.
fn load_compare(map: texture_depth_2d_array, layer: i32, uv: vec2<f32>, depth: f32) -> f32 {
    let size = vec2<f32>(textureDimensions(map));
    let st = uv * size - vec2<f32>(0.5);
    let base = floor(st);
    let fr = st - base;
    let lo = vec2<i32>(base);
    let hi = vec2<i32>(size) - vec2<i32>(1);
    let ta = clamp(lo, vec2<i32>(0), hi);
    let tb = clamp(lo + vec2<i32>(1, 0), vec2<i32>(0), hi);
    let tc = clamp(lo + vec2<i32>(0, 1), vec2<i32>(0), hi);
    let td = clamp(lo + vec2<i32>(1, 1), vec2<i32>(0), hi);
    // Reverse-Z: lit when the receiver is at least as near the light as the caster.
    let la = select(0.0, 1.0, depth >= textureLoad(map, ta, layer, 0));
    let lb = select(0.0, 1.0, depth >= textureLoad(map, tb, layer, 0));
    let lc = select(0.0, 1.0, depth >= textureLoad(map, tc, layer, 0));
    let ld = select(0.0, 1.0, depth >= textureLoad(map, td, layer, 0));
    return mix(mix(la, lb, fr.x), mix(lc, ld, fr.x), fr.y);
}

fn sun_visibility_load(
    map: texture_depth_2d_array,
    c: ModernCascades,
    world_pos: vec3<f32>,
    view_distance: f32,
) -> f32 {
    let count = u32(c.params.x);
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_distance < c.splits[i]) {
            let clip = c.clip_from_world[i] * vec4<f32>(world_pos, 1.0);
            let ndc = clip.xyz / clip.w;
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
                return 1.0;
            }
            let depth = ndc.z + c.depth_bias[i] * 2.0;
            let o = c.params.y * 1.25;
            var v = load_compare(map, i32(i), uv + vec2<f32>(o, o * 0.3), depth);
            v = v + load_compare(map, i32(i), uv + vec2<f32>(-o * 0.3, o), depth);
            v = v + load_compare(map, i32(i), uv + vec2<f32>(-o, -o * 0.3), depth);
            v = v + load_compare(map, i32(i), uv + vec2<f32>(o * 0.3, -o), depth);
            v = v * 0.25;
            if (i + 1u == count) {
                return mix(v, 1.0, smoothstep(c.splits[i] * 0.85, c.splits[i], view_distance));
            }
            return v;
        }
    }
    return 1.0;
}

// The wind over open water: the direction the froxel haze drifts (`volumetrics::update`'s wind).
fn wind_dir() -> vec2<f32> {
    return normalize(vec2<f32>(0.9, 0.35));
}

// A wind-driven swell as a surface SLOPE (d height / d xz), for the modern water: five directional
// deep-water waves spread about the wind, each fading out before it would alias at its distance, with
// slow gusts rolling across them so the surface has patches of chop and calm.
fn wind_swell(xz: vec2<f32>, time: f32, dist: f32) -> vec2<f32> {
    let wd = wind_dir();
    let across = vec2<f32>(-wd.y, wd.x);
    // Gusts: a large, slowly advected pattern (cat's paws), 0.35..1.
    let g = xz * 0.021 - wd * time * 0.09;
    let gust = 0.675 + 0.325 * sin(g.x * 1.7 + sin(g.y * 1.3 + time * 0.05) * 2.0) * cos(g.y * 1.1 - g.x * 0.4);
    var slope = vec2<f32>(0.0);
    // wavelength (yd), direction offset across the wind, steepness
    let waves = array<vec3<f32>, 5>(
        vec3<f32>(11.0, 0.0, 0.10),
        vec3<f32>(6.3, 0.45, 0.09),
        vec3<f32>(3.7, -0.35, 0.08),
        vec3<f32>(2.1, 0.8, 0.07),
        vec3<f32>(1.3, -0.65, 0.06),
    );
    for (var k = 0u; k < 5u; k = k + 1u) {
        let wl = waves[k].x;
        let dir = normalize(wd + across * waves[k].y);
        let kw = 6.2831853 / wl;
        // Deep water: omega = sqrt(g k), g = 9.8 m/s^2 in yards.
        let omega = sqrt(10.7 * kw);
        let phase = dot(dir, xz) * kw - omega * time + f32(k) * 1.7;
        let fade = 1.0 - smoothstep(wl * 18.0, wl * 45.0, dist);
        slope = slope + dir * (cos(phase) * waves[k].z * fade);
    }
    return slope * gust;
}

// P4: `sky_visibility` in one tap.
fn sky_visibility_fast(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
) -> f32 {
    if (c.sky_params.x < 0.5) {
        return 1.0;
    }
    let clip = c.sky_clip * vec4<f32>(world_pos + vec3<f32>(0.0, 0.75, 0.0), 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    return textureSampleCompareLevel(map, samp, uv, 4, ndc.z + c.sky_params.z);
}

// P4: the cloud-shadow grid's bilinear footprint around a world point — the four cell indices (00, 10,
// 01, 11) and the weights. The caller reads the four cells from its own block and hands them to
// `cloud_mix`; `grid` is `ModernBlock.fog.cloud`.
struct CloudTaps {
    idx: vec4<u32>,
    w: vec2<f32>,
}

fn cloud_taps(grid: vec4<f32>, world_pos: vec3<f32>) -> CloudTaps {
    let n = clamp(grid.w, 1.0, 32.0);
    let g = (world_pos.xz - grid.xy) / max(grid.z, 1e-3) - vec2<f32>(0.5);
    let lo = clamp(floor(g), vec2<f32>(0.0), vec2<f32>(n - 1.0));
    let hi = min(lo + vec2<f32>(1.0), vec2<f32>(n - 1.0));
    let row = u32(n);
    let xa = u32(lo.x);
    let xb = u32(hi.x);
    let ya = u32(lo.y) * row;
    let yb = u32(hi.y) * row;
    return CloudTaps(vec4<u32>(ya + xa, ya + xb, yb + xa, yb + xb), clamp(g - lo, vec2<f32>(0.0), vec2<f32>(1.0)));
}

fn cloud_mix(t: CloudTaps, ca: f32, cb: f32, cc: f32, cd: f32) -> f32 {
    return mix(mix(ca, cb, t.w.x), mix(cc, cd, t.w.x), t.w.y);
}

// Ambient light's share under cover never drops below this. The lanes sum in gamma and the frame
// decodes, so a share `s` here is roughly `s^2.2` of the open ambient on screen: 0.35 put the ground
// under a single tree at ~10 % — pitch black next to a sunlit patch.
const SKY_AMBIENT_FLOOR: f32 = 0.8;
// The same floor for the exterior ambient after the probe field has replaced the sky map: probes
// under a canopy legitimately read dark, but skylight still arrives sideways and off the ground.
const SHADE_FLOOR: f32 = 0.8;

// The exterior ambient (sky map or probe field) held above the open-sky row's floor.
fn shade_floor(ambient: vec3<f32>, open_sky: vec3<f32>) -> vec3<f32> {
    return max(ambient, open_sky * SHADE_FLOOR);
}

fn sky_ambient(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
) -> f32 {
    return mix(SKY_AMBIENT_FLOOR, 1.0, sky_visibility(map, samp, c, world_pos));
}
