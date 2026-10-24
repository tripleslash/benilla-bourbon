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
    // fork: P8 — the twelve shadowed lights' positions and ranges, moved OUT of `LightGrid` so the
    // struct every lit fragment copies by value shrank from ten rows to five (see the note beside
    // `LightGrid`). Only the binning compute pass reads them.
    light_shadow: array<vec4<f32>, 12>,
    // The padding that keeps the probe array behind this region 256-aligned. The REGION is 512 B
    // whatever P8 moved around inside it, which is why the probe offset did not shift a byte (the
    // far-level row `LightGrid.far` came out of this padding: 15 rows became 14).
    light_spare: array<vec4<f32>, 14>,
    light_cells: array<vec4<u32>, 55296>,
    // P5: the probe field itself, last because it is the block's only runtime-sized member.
    probes: array<probegi::GiProbe>,
}

// Mirrors `benilla_world::modern::shadow::ModernCascades::std430` (a layout test pins the size).
struct ModernCascades {
    // Reverse-Z orthographic light clip per cascade, as four columns each: read through
    // `cascade_clip`. Not `array<mat4x4<f32>, 4>`: Firefox (155, D3D12) transposes the matrices of
    // such an array when the struct holding it is copied out of a storage buffer, and every helper
    // here takes this struct by value. Same bytes, same loads.
    clip_cols: array<vec4<f32>, 16>,
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
    // x = enabled, y = texel world size, z = depth bias, w = the water mirror's plane (world Y) while
    // the mirror renders, and `NO_MIRROR_PLANE` otherwise — see `mirror_view`.
    sky_params: vec4<f32>,
}

// **Is this fragment being drawn for the water mirror** (`liquid::reflect`)? The mirror renders the
// whole scene a second time, into a buffer the waves then bend and blur, so it runs none of the
// modern lanes: terrain, the models and the retained statics skip their cascades, cloud shadow, probe
// field, sky map and light grid for it and keep the plain ambient and N·L (and the fog composite does
// not run on it at all). The helpers' `lite` arms below are what it took before that.
//
// No view flag reaches a material, so the eye's HEIGHT answers: the mirror camera always hangs at
// least `MIN_EYE_HEIGHT` (0.5 yd) under the plane it mirrors about, and runs only while the world
// camera is at least that far above it. The quarter yard of margin covers a frame's drift between the
// plane's verdict and the eye. Booths and UI panes never reach here: their light rows are classic.
fn mirror_view(c: ModernCascades, eye: vec3<f32>) -> bool {
    return eye.y < c.sky_params.w - 0.25;
}

// Cascade `i`'s light clip.
//
// The columns are picked by CONSTANT index. `c` is a by-value copy, and a computed index into an
// array inside a value has no SPIR-V form: naga writes the array to a function-local variable
// first, on every call, and the lanes that did so ran several times slower for it (the froxel
// scatter 10.9 -> 0.9 ms; RTX 3080 Laptop, Vulkan). `MODERN_ARRAY_INDEX` is the A/B lever
// (`$WOW_MODERN_ARRAYINDEX=1`): the computed index again.
fn cascade_clip(c: ModernCascades, i: u32) -> mat4x4<f32> {
#ifdef MODERN_ARRAY_INDEX
    let k = i * 4u;
    return mat4x4<f32>(c.clip_cols[k], c.clip_cols[k + 1u], c.clip_cols[k + 2u], c.clip_cols[k + 3u]);
#else
    switch i {
        case 0u: {
            return mat4x4<f32>(c.clip_cols[0], c.clip_cols[1], c.clip_cols[2], c.clip_cols[3]);
        }
        case 1u: {
            return mat4x4<f32>(c.clip_cols[4], c.clip_cols[5], c.clip_cols[6], c.clip_cols[7]);
        }
        case 2u: {
            return mat4x4<f32>(c.clip_cols[8], c.clip_cols[9], c.clip_cols[10], c.clip_cols[11]);
        }
        default: {
            return mat4x4<f32>(c.clip_cols[12], c.clip_cols[13], c.clip_cols[14], c.clip_cols[15]);
        }
    }
#endif
}

fn cascade_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    i: u32,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    lite: bool,
) -> f32 {
    let offset_pos = world_pos + normal * (c.texel_world[i] * c.params.z);
    let clip = cascade_clip(c, i) * vec4<f32>(offset_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    // Reverse-Z: a receiver is lit when it is at least as close to the light as the stored caster.
    let depth = ndc.z + c.depth_bias[i];
    if (lite) {
        // The mirror's tier: the hardware 2x2 compare alone.
        return textureSampleCompareLevel(map, samp, uv, i32(i), depth);
    }
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

// A surface in full shade still keeps a share of the key: sunlight scattered in through the leaves
// and around the edges of whatever shades it. How much depends on how open that cover is toward the
// sun — under one thin crown the shade stays light, under layered canopy it goes deep — so the share
// runs from `SUN_SHADE_DEEP` (every cover tap blocked) to `SUN_SHADE_OPEN` (none), and a half-open
// cover keeps the 0.35 one floor used to give everywhere. A zero floor made a canopy floor read as
// night next to classic's 0.7 (which keeps nearly all of it). The froxels do NOT take it — the
// contrast between lit and shadowed air is the shafts.
const SUN_SHADE_DEEP: f32 = 0.12;
const SUN_SHADE_OPEN: f32 = 0.58;
// The radius (yd) of the cover read: about a crown's depth of leaves, wide enough that one crown's
// holes count and narrow enough that the open ground past a neighbouring tree does not.
const COVER_RADIUS_YD: f32 = 1.5;

fn sun_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_distance: f32,
    lite: bool,
) -> f32 {
    let v = sun_visibility_raw(map, samp, c, world_pos, normal, view_distance, lite);
    // Only a fragment the sun does not fully reach pays for the cover read.
    var open = 0.5;
    if (v < 1.0 && !lite) {
        open = sun_cover_open(map, samp, c, world_pos, normal, view_distance);
    }
    return mix(mix(SUN_SHADE_DEEP, SUN_SHADE_OPEN, open), 1.0, v);
}

// How open the cover toward the sun is around a point (1 = no caster within `COVER_RADIUS_YD`): eight
// compares on two rings in the receiver's own tangent plane, so a slope or an oblique sun never
// counts the receiver's own ground as cover. Past the cascades 0.5: the far field has its own floor.
fn sun_cover_open(
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
            let clip_from_world = cascade_clip(c, i);
            // The light clip is orthographic (w = 1), so a tangent step is a fixed step in clip.
            let base = (clip_from_world * vec4<f32>(world_pos + normal * (c.texel_world[i] * c.params.z), 1.0)).xyz;
            let side = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.x) > 0.9);
            let t1 = normalize(cross(normal, side));
            let t2 = cross(normal, t1);
            let a = (clip_from_world * vec4<f32>(t1 * COVER_RADIUS_YD, 0.0)).xyz;
            let b = (clip_from_world * vec4<f32>(t2 * COVER_RADIUS_YD, 0.0)).xyz;
            var ring = array<vec2<f32>, 8>(
                vec2<f32>(1.0, 0.0),
                vec2<f32>(0.0, 1.0),
                vec2<f32>(-1.0, 0.0),
                vec2<f32>(0.0, -1.0),
                vec2<f32>(0.35, 0.35),
                vec2<f32>(-0.35, 0.35),
                vec2<f32>(-0.35, -0.35),
                vec2<f32>(0.35, -0.35),
            );
            var open = 0.0;
            for (var t = 0u; t < 8u; t = t + 1u) {
                let p = base + a * ring[t].x + b * ring[t].y;
                let uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
                open = open + textureSampleCompareLevel(map, samp, uv, i32(i), p.z + c.depth_bias[i] * 2.0);
            }
            return open / 8.0;
        }
    }
    return 0.5;
}

// Sun visibility (1 = lit) at a world position seen `view_distance` yd in front of the camera.
fn sun_visibility_raw(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_distance: f32,
    lite: bool,
) -> f32 {
    let count = u32(c.params.x);
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_distance < c.splits[i]) {
            let v = cascade_visibility(map, samp, c, i, world_pos, normal, lite);
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
            let clip = cascade_clip(c, i) * vec4<f32>(world_pos, 1.0);
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

// Sun visibility at a world point that has no place in the view — the probe trace's ray hits, which
// lie all around the camera, behind it as often as in front. `sun_visibility_fast` picks its cascade
// by view depth and reads anything outside that cascade's map as LIT, which is right for a fragment
// on screen and wrong for a wall behind the player: every probe near it took the shaded street
// behind the camera for a sunlit one, bounced that light, and dimmed again once the player turned
// or walked on and the wall came into a cascade. So the probes' brightness followed the camera.
// Here the finest cascade whose map actually holds the point answers, and past all of them the
// camera-independent far term does.
fn sun_visibility_anywhere(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
) -> f32 {
    let count = u32(c.params.x);
    if (count == 0u) {
        return 1.0;
    }
    // A texel in from the edge, so the compare's footprint never reads past the map.
    let margin = c.params.y;
    for (var i = 0u; i < count; i = i + 1u) {
        let clip = cascade_clip(c, i) * vec4<f32>(world_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (all(uv >= vec2<f32>(margin)) && all(uv <= vec2<f32>(1.0 - margin)) && ndc.z >= 0.0) {
            return textureSampleCompareLevel(map, samp, uv, i32(i), ndc.z + c.depth_bias[i] * 2.0);
        }
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
            let clip = cascade_clip(c, i) * vec4<f32>(world_pos, 1.0);
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

// A wind-driven sea as a surface SLOPE (d height / d xz), for the modern water. Nine directional
// deep-water trains from long swell to short chop, their directions spread irregularly about the wind
// (not a fan), their phases pushed around by a caller-supplied noise `warp` so crests never line up
// into a visible lattice, and their energy gathered into gust patches by `gust` (0..1, also noise).
// Each train fades out before it would alias at this distance.
fn wind_swell(xz: vec2<f32>, time: f32, dist: f32, warp: vec2<f32>, gust: f32) -> vec2<f32> {
    let wd = wind_dir();
    let across = vec2<f32>(-wd.y, wd.x);
    // wavelength (yd), direction offset across the wind, steepness
    let waves = array<vec3<f32>, 9>(
        vec3<f32>(17.3, 0.07, 0.055),
        vec3<f32>(11.1, -0.31, 0.06),
        vec3<f32>(7.9, 0.46, 0.058),
        vec3<f32>(5.3, -0.12, 0.055),
        vec3<f32>(3.7, 0.83, 0.05),
        vec3<f32>(2.6, -0.58, 0.046),
        vec3<f32>(1.9, 0.24, 0.042),
        vec3<f32>(1.3, -0.97, 0.036),
        vec3<f32>(0.9, 0.62, 0.03),
    );
    // Chop (short trains) follows the gusts; swell (long trains) barely does.
    var slope = vec2<f32>(0.0);
    for (var k = 0u; k < 9u; k = k + 1u) {
        let wl = waves[k].x;
        let dir = normalize(wd + across * waves[k].y);
        let kw = 6.2831853 / wl;
        // Deep water: omega = sqrt(g k), g = 9.8 m/s^2 in yards.
        let omega = sqrt(10.7 * kw);
        let fk = f32(k);
        // The warp displaces each train by a different share of its own wavelength.
        let shift = dot(warp, vec2<f32>(cos(fk * 2.4), sin(fk * 2.4))) * wl * 0.6;
        let phase = (dot(dir, xz) + shift) * kw - omega * time + fk * 1.37;
        let fade = 1.0 - smoothstep(wl * 18.0, wl * 45.0, dist);
        let chop = mix(0.85, 0.25 + 1.1 * gust, clamp(1.0 - wl / 8.0, 0.0, 1.0));
        // A sharpened crest: slope of a trochoid-like profile, steeper just before the crest.
        let c = cos(phase);
        let crest = c * (1.0 + 0.35 * sin(phase));
        slope = slope + dir * (crest * waves[k].z * fade * chop);
    }
    return slope;
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

// The exterior ambient before its floor: the probe field over the sky map, by how much of the
// fragment's probe cell answered (`gi.w`). Where the field answers whole, the sky map's term is
// multiplied by zero — so the lanes read the 3x3 sky map ([`sky_ambient`]) only when `gi.w < 1`
// and pass `sky = 1` otherwise, and this returns the field's value exactly there rather than
// through a `mix` whose arithmetic form a backend is free to pick.
fn probe_or_sky(amb: vec3<f32>, sky: f32, gi: vec4<f32>) -> vec3<f32> {
    return select(mix(amb * sky, gi.rgb, gi.w), gi.rgb, gi.w >= 1.0);
}

fn sky_ambient(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    c: ModernCascades,
    world_pos: vec3<f32>,
    lite: bool,
) -> f32 {
    if (lite) {
        return mix(SKY_AMBIENT_FLOOR, 1.0, sky_visibility_fast(map, samp, c, world_pos));
    }
    return mix(SKY_AMBIENT_FLOOR, 1.0, sky_visibility(map, samp, c, world_pos));
}
