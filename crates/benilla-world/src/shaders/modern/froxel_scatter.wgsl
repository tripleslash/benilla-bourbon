// Modern lighting (FORK.md, P4): froxel scattering — the Hillaire 2015 V-buffer and
// in-scatter in one compute pass, blended with last frame's reprojected result.
//
// Per froxel, jittered in depth: extinction from the fog media (height falloff, near-clear, drifting
// noise) plus the constant shaft haze; in-scatter from the key light (dual-lobe HG x cascade visibility
// x cloud shadow), the sky (ambient x sky visibility) and the nearest point lights. Written as
// (sigma_s * L, sigma_t) for `froxel_integrate.wgsl` to accumulate front to back.
#import benilla_modern::sunshadow
#import benilla_modern::froxelfog
#import benilla_modern::probegi
#import benilla_modern::locallight

// The shared light buffer's prefix (`lighting::global_light`): the header rows and the point table,
// packed nearest-to-camera first.
struct LightTable {
    rows: array<vec4<f32>, 21>,
    points: array<vec4<f32>, 512>,
}

@group(0) @binding(0) var<storage, read> modern_block: sunshadow::ModernBlock;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
@group(0) @binding(2) var shadow_samp: sampler_comparison;
@group(0) @binding(3) var<storage, read> light_table: LightTable;
@group(0) @binding(4) var noise_tex: texture_3d<f32>;
@group(0) @binding(5) var repeat_samp: sampler;
@group(0) @binding(6) var history_tex: texture_3d<f32>;
@group(0) @binding(7) var scatter_out: texture_storage_3d<rgba16float, write>;

// Under full cover the ambient in-scatter keeps this share: skylight still arrives sideways. It
// matches the surfaces' linear shade floor (`sunshadow::SHADE_FLOOR` 0.72 in gamma ≈ 0.49 on
// screen); at 0.25 the haze under a canopy went darker than the ground it hangs over, and a dense
// zone fog (Teldrassil) turned the whole forest murky.
const SKY_FLOOR: f32 = 0.65;
// The noise's feature size (yd).
const NOISE_SCALE: f32 = 48.0;
// The shaft haze scatters the key this much harder than the fog medium does, and the sky this much
// less: light cones read from the CONTRAST between sunlit and shadowed air, and an ambient-heavy
// haze fills the shadowed air in until the cones vanish. A calibration, not a physical albedo.
const SHAFT_KEY_GAIN: f32 = 1.4;
const SHAFT_SKY_SHARE: f32 = 0.55;
// The haze's phase is half isotropic, so a cone seen side-on under a canopy still reads; the fog's
// dual lobe alone passes ~0.4 at 90 degrees from the sun.
const SHAFT_ISOTROPIC: f32 = 0.5;

// Under full cover the shaft haze is this much denser, and scatters the key this much harder on top.
const CANOPY_HAZE: f32 = 1.5;
const CANOPY_SHAFT: f32 = 1.0;
// Point lights' in-scatter against their surface lighting.
const POINT_GAIN: f32 = 0.35;
// fork: a lamp's glow in the haze leans a little toward the eye (smoke scatters forward): the share
// of a Henyey-Greenstein lobe of `POINT_G` mixed over the isotropic term.
const POINT_G: f32 = 0.5;
const POINT_FORWARD: f32 = 0.3;
// fork: **the glow sits at the flame.** A light's in-scatter keeps `POINT_FAR_SHARE` of itself past
// its halo and all of it within `POINT_HALO_YD` of the source, so a brazier stands in a small bright
// halo while a street of lit windows (each a light, 48 yd of range) no longer fills the air around
// the camera with a grey glow.
const POINT_HALO_YD: f32 = 4.0;
const POINT_FAR_SHARE: f32 = 0.35;
// fork: the night air's near clear (yd of planar depth): none of it nearer than `NEAR`, all of it past
// `FAR`. The street the player stands in stays clear; the distance layers into haze.
const NIGHT_CLEAR_NEAR: f32 = 30.0;
const NIGHT_CLEAR_FAR: f32 = 150.0;

// The key light's own radiance for the shaft haze (the zone's glow and shaft gain applied): the
// medium's key term is normalized to the fog colour, so a beam would take the fog's hue.
fn shaft_key(f: froxelfog::FroxelFog) -> vec3<f32> {
    return vec3<f32>(f.medium.z, f.ambient.w, f.wind.w);
}

// How far around a froxel the canopy cover is gathered (yd).
const CANOPY_REACH: f32 = 7.0;

// The share of the sky hidden around `p` (0 open, 1 under a closed canopy): four sky-map taps a few
// yards out, lifted a little so the ground itself never counts.
fn canopy_cover(p: vec3<f32>) -> f32 {
    let c = modern_block.cascades;
    var open = 0.0;
    open = open + sunshadow::sky_visibility_fast(shadow_map, shadow_samp, c, p + vec3<f32>(CANOPY_REACH, 0.0, 0.0));
    open = open + sunshadow::sky_visibility_fast(shadow_map, shadow_samp, c, p + vec3<f32>(-CANOPY_REACH, 0.0, 0.0));
    open = open + sunshadow::sky_visibility_fast(shadow_map, shadow_samp, c, p + vec3<f32>(0.0, 0.0, CANOPY_REACH));
    open = open + sunshadow::sky_visibility_fast(shadow_map, shadow_samp, c, p + vec3<f32>(0.0, 0.0, -CANOPY_REACH));
    return 1.0 - open * 0.25;
}

// One table light's in-scatter at a froxel: the surface lanes' WoW falloff under the same range
// window, zero past the range. `to_eye` is the unit direction from the froxel to the camera.
//
// fork: under a forward lobe (`POINT_G`, `POINT_FORWARD`), relative to the isotropic term so the
// light's total in-scatter is unchanged: brighter looking toward the lamp, dimmer looking away.
fn point_in_scatter(i: u32, p: vec3<f32>, to_eye: vec3<f32>) -> vec3<f32> {
    let pos_range = light_table.points[2u * i];
    let dl = length(pos_range.xyz - p);
    if (dl >= pos_range.w) {
        return vec3<f32>(0.0);
    }
    let q = dl / max(pos_range.w, 1e-3);
    let window = clamp(1.0 - q * q * q * q, 0.0, 1.0);
    // +0.5 keeps a froxel inside the flame finite.
    let atten = 1.0 / (0.7 * dl + 0.03 * dl * dl + 0.5);
    let mu = dot((p - pos_range.xyz) / max(dl, 1e-3), to_eye);
    let g2 = POINT_G * POINT_G;
    let hg = (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * POINT_G * mu, 1e-4), 1.5);
    let phase = mix(1.0, hg, POINT_FORWARD);
    let halo = mix(POINT_FAR_SHARE, 1.0, exp(-dl / POINT_HALO_YD));
    return light_table.points[2u * i + 1u].rgb * (atten * window * window * POINT_GAIN * phase * halo);
}

// Every light one grid cell names, at a froxel.
fn cell_in_scatter(cell: vec4<u32>, p: vec3<f32>, to_eye: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let count = locallight::cell_count(cell);
    for (var i = 0u; i < count; i = i + 1u) {
        sum = sum + point_in_scatter(locallight::cell_light(cell, i), p, to_eye);
    }
    return sum;
}

// A per-froxel, per-frame offset in [-0.5, 0.5]: the one-tap shadow lookup then covers the froxel's
// footprint over the history instead of its centre alone, so a shaft's edge is not a froxel staircase.
fn froxel_jitter(id: vec3<u32>, frame: f32) -> vec2<f32> {
    let h = fract(sin(dot(vec3<f32>(id), vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
    let r = fract(vec2<f32>(h + frame * 0.618034, h * 1.7 + frame * 0.754878));
    return r - vec2<f32>(0.5);
}

@compute @workgroup_size(4, 4, 4)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = modern_block.fog;
    if (any(id >= vec3<u32>(f.grid.xyz))) {
        return;
    }
    let n = f.grid.xyz;
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5) + froxel_jitter(id, f.phase.w) * 0.5) / n.xy;
    let s = (f32(id.z) + f.phase.z) / n.z;
    let d = max(froxelfog::depth_of_slice(f, s), f.proj.x);
    // Reverse-Z, infinite perspective: planar depth `d` sits at ndc z = near / d.
    let h = f.world_from_clip * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, f.proj.x / d, 1.0);
    let p = h.xyz / h.w;

    // Extinction. The fog is a height fog, clear before its start; the shaft haze (amendment A1) never
    // clears, so shafts can begin at the camera. Both are per yard of the view ray.
    let noise = textureSampleLevel(noise_tex, repeat_samp, (p + f.wind.xyz) / NOISE_SCALE, 0.0).r;
    let wisps = mix(1.0, 0.35 + 1.3 * noise, f.medium.y);
    let density = froxelfog::height_density(f, p.y);
    // fork: modern lighting — a room's medium reaches as far as the room does (`froxelfog::room`):
    // inside its walls the froxel breathes the building's MFOG (uniform, no near-clear range), and
    // past them the zone's own air, so a doorway's view is fogged by what is on the far side of it.
    let room = froxelfog::room_weight(f, p);
    let open_fog = f.media.x * froxelfog::fog_clear(f, d) * density;
    let sigma_fog = mix(open_fog, f.room_media.x, room) * wisps;
    // Under a canopy the air is thick with dust and damp: the haze gets denser and scatters the key
    // much harder there, so the light through a gap in the leaves reads as a beam. The cover is the
    // neighbourhood's (the sky map around the froxel), not the froxel's own column — a point inside a
    // beam has open sky straight above it by definition.
    let cover = canopy_cover(p);
    let sigma_haze = f.medium.x * density * (1.0 + CANOPY_HAZE * cover)
        + f.room_media.z * density * smoothstep(NIGHT_CLEAR_NEAR, NIGHT_CLEAR_FAR, d);
    let sigma = sigma_fog + sigma_haze;

    // In-scatter.
    let ray = p - f.camera.xyz;
    let mu = dot(ray / max(length(ray), 1e-3), f.sun_dir.xyz);
    let ct = sunshadow::cloud_taps(f.cloud, p);
    let cloud = sunshadow::cloud_mix(
        ct,
        modern_block.cloud[ct.idx.x],
        modern_block.cloud[ct.idx.y],
        modern_block.cloud[ct.idx.z],
        modern_block.cloud[ct.idx.w],
    );
    // Two taps a quarter froxel apart along the view ray: a beam's edge crossing a froxel then reads as
    // a partial value instead of flipping on and off with the per-frame jitter.
    // Slices are quadratic in depth (d = far * s^2), so a slice is 2 * sqrt(far * d) / n deep.
    let along = normalize(p - f.camera.xyz) * max(0.5 * sqrt(f.grid.w * d) / n.z, 0.1);
    let sun_vis = 0.5 * cloud * (
        sunshadow::sun_visibility_fast(shadow_map, shadow_samp, modern_block.cascades, p - along, d)
            + sunshadow::sun_visibility_fast(shadow_map, shadow_samp, modern_block.cascades, p + along, d)
    );
    let sky = sunshadow::sky_visibility_fast(shadow_map, shadow_samp, modern_block.cascades, p);
    // fork: modern lighting (FORK.md, P5) — where the probe field reaches this
    // froxel, its mean radiance against an open-field probe's is what shades the ambient in-scatter:
    // the haze goes dark under a canopy and keeps the zone's own colour in the open. One tap, because
    // the volume's own blur and its history are a better filter than eight reads would be.
    var sky_term = mix(SKY_FLOOR, 1.0, sky);
    let g = modern_block.gi;
    if (g.params.x > 0.5 && g.sky.w > 1e-4) {
        let found = probegi::nearest_index(g, p);
        if (found.y > 0.5) {
            let probe = modern_block.probes[u32(found.x)];
            if (probe.dc.w > 0.0) {
                let mean = dot(probe.dc.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
                sky_term = clamp(mean / g.sky.w, SKY_FLOOR, 1.25);
            }
        }
    }
    let phase = froxelfog::dual_phase(f, mu);
    let lum_fog = mix(
        f.ambient.rgb * sky_term + f.sun.rgb * (phase * sun_vis),
        f.room_ambient.rgb * sky_term + f.room_sun.rgb * (phase * sun_vis),
        room,
    );
    let lum_haze = f.ambient.rgb * (sky_term * SHAFT_SKY_SHARE)
        + shaft_key(f) * (mix(phase, 1.0, SHAFT_ISOTROPIC) * sun_vis * SHAFT_KEY_GAIN * (1.0 + CANOPY_SHAFT * cover));
    var points = vec3<f32>(0.0);

    // The point lights, isotropic, with the surface lanes' WoW falloff and range window.
    //
    // fork: P8b — **from the light grid wherever it is live**, not from the first N lights of the
    // camera-sorted table. That loop cost every froxel N iterations (16, 32 or 64 by tier, over up
    // to two million froxels), and nearly all of them were lights that could not reach it — and it
    // could not see a lamp that sat past the camera's N nearest even when the froxel stood in its
    // pool. The grid already holds, per cell, exactly the lights that reach that cell (the strongest
    // eight where more do), so a froxel now reads one cell — two in the near/far seam, cross-faded
    // as the surfaces do (`locallight::cell_pair`) — and the haze over a lamp's pool is lit by the
    // same lights as the ground under it. Without a grid (`?modern_lights=0`) the old walk stands.
    let lg = modern_block.lights;
    let to_eye = normalize(f.camera.xyz - p);
    if (locallight::grid_live(lg)) {
        let pair = locallight::cell_pair(lg, p);
        if (pair.w > 0u) {
            let far_w = bitcast<f32>(pair.z);
            if (far_w < 1.0) {
                points = cell_in_scatter(modern_block.light_cells[pair.x], p, to_eye) * (1.0 - far_w);
            }
            if (far_w > 0.0) {
                points = points + cell_in_scatter(modern_block.light_cells[pair.y], p, to_eye) * far_w;
            }
        }
    } else {
        let count = min(u32(f.screen.w), u32(light_table.rows[20].x));
        for (var i = 0u; i < count; i = i + 1u) {
            points = points + point_in_scatter(i, p, to_eye);
        }
    }

    var cur = vec4<f32>((lum_fog + points) * sigma_fog + (lum_haze + points) * sigma_haze, sigma);

    // History: this froxel's world point in last frame's grid. Clip w is the planar depth.
    if (f.medium.w > 0.0) {
        let prev = f.prev_clip_from_world * vec4<f32>(p, 1.0);
        if (prev.w > f.proj.x && prev.w < f.grid.w) {
            let pn = prev.xy / prev.w;
            let puv = vec2<f32>(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
            if (all(puv > vec2<f32>(0.0)) && all(puv < vec2<f32>(1.0))) {
                let half_texel = vec3<f32>(0.5) / n;
                let uvw = clamp(
                    vec3<f32>(puv, froxelfog::slice_of_depth(f, prev.w)),
                    half_texel,
                    vec3<f32>(1.0) - half_texel,
                );
                cur = mix(cur, textureSampleLevel(history_tex, repeat_samp, uvw, 0.0), f.medium.w);
            }
        }
    }
    textureStore(scatter_out, id, cur);
}
