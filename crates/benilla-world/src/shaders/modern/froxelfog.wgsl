// Modern lighting (FORK.md, P4): the froxel fog volume's parameters, its lookup, and
// the analytic height fog that stands in for it beyond its far plane and on the Low tier.
//
// Imported as `benilla_modern::froxelfog` — a module name no hooked shader uses as an identifier
// (naga_oil substitutes a module alias for every bare identifier of that name).
//
// No bindings are declared here: each lane binds the integrated volume and its sampler itself and
// passes them in, beside the parameter block (`sunshadow::ModernBlock.fog`).
#define_import_path benilla_modern::froxelfog

// Mirrors `benilla_world::modern::volumetrics::FroxelFog::std430` (a layout test pins the size).
struct FroxelFog {
    // The main world camera's unjittered clip-from-world. Every lane looks the volume up through it,
    // the water mirror included: a reflected point is fogged as the main camera sees it.
    clip_from_world: mat4x4<f32>,
    // Its inverse: the compute pass walks froxels back into the world.
    world_from_clip: mat4x4<f32>,
    // Last frame's clip-from-world, for the scatter history's reprojection.
    prev_clip_from_world: mat4x4<f32>,
    // xyz = the main camera's world position; w = seconds (the noise clock).
    camera: vec4<f32>,
    // The main camera's view-from-world third row: planar view depth = -(dot(xyz, p) + w).
    view_row: vec4<f32>,
    // xyz = this tier's froxel grid; w = its far plane (planar yd).
    grid: vec4<f32>,
    // xyz = the integrated volume texture's allocated size; w = mode (0 off, 1 analytic, 2 froxels).
    alloc: vec4<f32>,
    // x = fog extinction (1/yd); y = 1 / height falloff (yd); z = reference height; w = near-clear distance.
    media: vec4<f32>,
    // x = shaft haze extinction (1/yd, never near-cleared); y = noise amount; z = storm blend;
    // w = scatter history weight (0 on a cut).
    medium: vec4<f32>,
    // rgb = ambient in-scatter radiance (albedo x ambient); w = a sky pixel's share of optical depth.
    ambient: vec4<f32>,
    // rgb = key-light in-scatter radiance (albedo x key); w = the forward lobe's share.
    sun: vec4<f32>,
    // xyz = unit direction TOWARD the key light; w = the forward lobe's g.
    sun_dir: vec4<f32>,
    // x = the backward lobe's g; y = cloud shadow on; z = this frame's slice jitter [0, 1); w = frame index.
    phase: vec4<f32>,
    // xy = the cloud-shadow grid's origin (world x, z); z = cell size (yd); w = cells per side.
    cloud: vec4<f32>,
    // x = near plane; y = 1 / clip_from_view[0][0]; z = 1 / clip_from_view[1][1]; w = screen-shaft strength.
    proj: vec4<f32>,
    // xy = the key light's uv in the main view; z = its on-screen weight; w = froxel point-light count.
    screen: vec4<f32>,
    // xyz = noise scroll (yd); w = point-light in-scatter gain.
    wind: vec4<f32>,
}

const MODE_ANALYTIC: f32 = 1.0;
const MODE_FROXEL: f32 = 2.0;

// Planar view depth (yd in front of the main camera) of a world point.
fn view_depth(f: FroxelFog, p: vec3<f32>) -> f32 {
    return -(dot(f.view_row.xyz, p) + f.view_row.w);
}

// Slices are quadratic in depth — dense near the eye, where shafts start.
fn slice_of_depth(f: FroxelFog, d: f32) -> f32 {
    return sqrt(clamp(d / f.grid.w, 0.0, 1.0));
}

fn depth_of_slice(f: FroxelFog, s: f32) -> f32 {
    return f.grid.w * s * s;
}

// Density scale at height `y`: exponential falloff above the reference height, at most x3 below it.
fn height_density(f: FroxelFog, y: f32) -> f32 {
    return exp(clamp(-(y - f.media.z) * f.media.y, -12.0, 1.1));
}

// The mean of `height_density` along a straight segment between heights `y_from` and `y_to`.
fn height_average(f: FroxelFog, y_from: f32, y_to: f32) -> f32 {
    let k = f.media.y;
    let dy = y_to - y_from;
    if (abs(dy * k) < 1e-3) {
        return height_density(f, 0.5 * (y_from + y_to));
    }
    let a = clamp(-(y_from - f.media.z) * k, -12.0, 1.1);
    let b = clamp(-(y_to - f.media.z) * k, -12.0, 1.1);
    return clamp((exp(a) - exp(b)) / (dy * k), 0.0, 3.0);
}

// Henyey-Greenstein, normalized so that its average over the sphere is 1.
fn hg(g: f32, mu: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * mu, 1e-4);
    return (1.0 - g2) / (denom * sqrt(denom));
}

// Amendment A2: a forward lobe (warm toward the sun) blended with a backward one (the zone's hue away
// from it). `mu` = cos of the angle between the view ray and the direction toward the light.
fn dual_phase(f: FroxelFog, mu: f32) -> f32 {
    return mix(hg(f.phase.x, mu), hg(f.sun_dir.w, mu), f.sun.w);
}

// Analytic height fog along the main camera's ray to `p`, from fraction `t_start` of the way there to `p`
// itself: rgb = in-scattered radiance, a = transmittance. The fog starts at the near-clear distance;
// the shaft haze does not.
fn analytic_path(f: FroxelFog, p: vec3<f32>, t_start: f32) -> vec4<f32> {
    return analytic_path_keyed(f, p, t_start, 1.0);
}

// Past the froxel grid nothing knows where the key light is shadowed, so the tail keeps only this
// share of its key in-scatter: a long view ray past a ridge or a canopy otherwise lit up the whole
// distance behind it as if the sun reached every yard of it.
const TAIL_KEY_SHARE: f32 = 0.35;

fn analytic_path_keyed(f: FroxelFog, p: vec3<f32>, t_start: f32, key_share: f32) -> vec4<f32> {
    let ray = p - f.camera.xyz;
    let len = length(ray);
    let t = max(t_start, 0.0);
    if (len < 1e-3 || t >= 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let start = len * t;
    let fog_len = max(len - max(f.media.w, start), 0.0);
    let h = height_average(f, f.camera.y + ray.y * t, p.y);
    let tau = (f.media.x * fog_len + f.medium.x * (len - start)) * h;
    let tr = exp(-tau);
    let lum = f.ambient.rgb + f.sun.rgb * (dual_phase(f, dot(ray / len, f.sun_dir.xyz)) * key_share);
    return vec4<f32>(lum * (1.0 - tr), tr);
}

// Where a world point falls in the froxel grid: xy = screen uv, z = slice coordinate, w = 1 when inside
// the main camera's frustum.
fn froxel_uvw(f: FroxelFog, p: vec3<f32>) -> vec4<f32> {
    let clip = f.clip_from_world * vec4<f32>(p, 1.0);
    if (clip.w <= 1e-4) {
        return vec4<f32>(0.0);
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let inside = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    return vec4<f32>(uv, slice_of_depth(f, clip.w), select(0.0, 1.0, inside));
}

// The integrated volume at a screen uv and slice coordinate. Texel k holds the accumulation to the FAR
// edge of slice k, so a continuous slice position c reads texel coordinate c - 0.5; inside the first
// slice the accumulation ramps up from nothing.
fn integrated(vol: texture_3d<f32>, samp: sampler, f: FroxelFog, uv: vec2<f32>, s: f32) -> vec4<f32> {
    let n = f.grid.xyz;
    let c = s * n.z;
    let texel = clamp(vec3<f32>(uv * n.xy, c - 0.5), vec3<f32>(0.5), n - vec3<f32>(0.5));
    let v = textureSampleLevel(vol, samp, texel / f.alloc.xyz, 0.0);
    let ramp = clamp(c, 0.0, 1.0);
    return vec4<f32>(v.rgb * ramp, mix(1.0, v.a, ramp));
}

// The fog between the main camera and world point `p`: rgb = in-scatter, a = transmittance. A surface
// takes `rgb * a + in-scatter`. Froxels inside the grid, the analytic tail beyond its far plane, and the
// analytic path alone outside the main frustum (the mirror's view) or on the Low tier.
fn fog_at(vol: texture_3d<f32>, samp: sampler, f: FroxelFog, p: vec3<f32>) -> vec4<f32> {
    if (f.alloc.w < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let d = view_depth(f, p);
    if (f.alloc.w < MODE_FROXEL - 0.5 || d <= f.proj.x) {
        return analytic_path(f, p, 0.0);
    }
    let q = froxel_uvw(f, p);
    if (q.w < 0.5) {
        return analytic_path(f, p, 0.0);
    }
    let near = integrated(vol, samp, f, q.xy, q.z);
    if (d <= f.grid.w) {
        return near;
    }
    let tail = analytic_path_keyed(f, p, f.grid.w / d, TAIL_KEY_SHARE);
    return vec4<f32>(near.rgb + near.a * tail.rgb, near.a * tail.a);
}
