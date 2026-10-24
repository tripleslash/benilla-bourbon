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
// fork: the GI terrain heightfield (a 1-texel stand-in when GI is off), for the ground-fog banks.
@group(0) @binding(8) var ground_map: texture_depth_2d;

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

// The key light's own radiance for the shaft haze (the zone's glow and shaft gain applied): the
// medium's key term is normalized to the fog colour, so a beam would take the fog's hue.
fn shaft_key(f: froxelfog::FroxelFog) -> vec3<f32> {
    return vec3<f32>(f.medium.z, f.ambient.w, f.wind.w);
}

// fork: the shaft haze's and the banks' phase: half the fog's dual lobe, where a beam reads, and half
// isotropic at the key's share on the air (`halo_air.w`: 1 by day, the moon's `look::MOON_HAZE`).
// Lit evenly by a bright night diffuse, every moonlit froxel out to the horizon read pale.
fn shaft_phase(f: froxelfog::FroxelFog, phase: f32) -> f32 {
    return mix(phase, f.halo_air.w, SHAFT_ISOTROPIC);
}

// fork: **ground fog.** Overnight (and by day in the dark zones) the air near the ground gathers into
// banks (`fog.night`): dense within a few yards of the terrain under each froxel, broken into
// patches by the drifting noise at a coarse scale, and lit by the key as hard as the shaft haze, so
// a beam reads brightest where it crosses a bank.
// The patches' feature size (yd), and the noise band that is bank: below the low edge is clear air.
const BANK_SCALE: f32 = 150.0;
// 1 / the banks' height falloff (yd) above the terrain: dense within a few yards of it.
const BANK_INV_FALLOFF: f32 = 0.25;
const BANK_EDGE: vec2<f32> = vec2<f32>(0.45, 0.72);
// A bank's in-scatter: the key as the shaft haze takes it, and this share of the sky.
const BANK_KEY_GAIN: f32 = 1.6;
const BANK_SKY_SHARE: f32 = 0.5;

// The terrain under `p` from the heightfield (`probegi::terrain_height`'s read), read once per froxel
// for the ground banks and the sea fog: x = the height at the texel `p` clamps to (`-1e9` where
// nothing is drawn), y = 1 when `p` lies inside the heightfield over drawn terrain. The caller
// checks `ground.x` (the heightfield is live).
fn terrain_under(f: froxelfog::FroxelFog, p: vec3<f32>) -> vec2<f32> {
    let q = vec4<f32>(p, 1.0);
    let raw = vec2<f32>(dot(f.ground_x, q), dot(f.ground_y, q));
    let uv = clamp(raw, vec2<f32>(0.0), vec2<f32>(1.0));
    let n = f.ground.w;
    let d = textureLoad(ground_map, vec2<i32>(clamp(uv * n, vec2<f32>(0.0), vec2<f32>(n - 1.0))), 0);
    if (d <= 0.0) {
        return vec2<f32>(-1e9, 0.0);
    }
    return vec2<f32>(f.ground.y + d * f.ground.z, select(0.0, 1.0, all(raw == uv)));
}

// The ground banks' extinction at `p` (1/yd), over `terrain` (`terrain_under`; off or outside it,
// the ground under the camera).
fn ground_banks(f: froxelfog::FroxelFog, p: vec3<f32>, terrain: vec2<f32>) -> f32 {
    let core = f.night.x;
    if (core <= 0.0) {
        return 0.0;
    }
    let ground = select(f.night.y, terrain.x, terrain.y > 0.5);
    let above = max(p.y - ground, 0.0);
    let height = exp(-min(above * BANK_INV_FALLOFF, 24.0));
    let q = vec3<f32>(p.x, p.y * 0.25, p.z) + f.wind.xyz * 0.5;
    let n = textureSampleLevel(noise_tex, repeat_samp, q / BANK_SCALE + vec3<f32>(0.37, 0.11, 0.73), 0.0).r;
    return core * height * smoothstep(BANK_EDGE.x, BANK_EDGE.y, n);
}

// fork: **sea fog.** A zone look's `sea_fog` (`fog.night.w`) lays a bank on the ocean along the
// whole coast: dense at sea level, rolling up to a top that swells and sinks across the water, broken
// into billows, and coming in off the shore over the first few yards of depth, so the beach stays
// clear. The sea is wherever the terrain heightfield's bed lies under sea level; past the
// heightfield's edge the edge row stands in, and an undrawn texel (open ocean, no tile) is sea.
// WoW's oceans lie at height 0.
const SEA_LEVEL: f32 = 0.0;
// The bank's mean top (yd of e-folding above the sea), and the depth band that is the shore.
const SEA_HEIGHT: f32 = 16.0;
const SEA_SHORE: vec2<f32> = vec2<f32>(1.5, 10.0);
// The swell's feature size (yd). The billows are the froxel's own drifting noise (`NOISE_SCALE`,
// 48 yd), already sampled for the fog's wisps: a third tap per froxel over the whole sea cost
// frames walking through the banks.
const SEA_SCALE: f32 = 220.0;
// The swell's drift, in tiles per wrapped hour (`camera.w` wraps at 3600 s): whole numbers, so the
// noise does not jump when the clock wraps (~0.5 yd/s).
const SEA_DRIFT: vec3<f32> = vec3<f32>(8.0, 0.0, 3.0);

// Whether the sea fog can reach `p` at all: on in this zone, the heightfield live, and `p` between
// just under the surface and the height past which the tallest bank has thinned to nothing.
fn sea_reaches(f: froxelfog::FroxelFog, p: vec3<f32>) -> bool {
    let above = p.y - SEA_LEVEL;
    return f.night.w > 0.0 && f.ground.x > 0.5 && above <= SEA_HEIGHT * 10.0 && above >= -2.0;
}

// The sea fog's extinction at `p` (1/yd), over the bed `seabed` (`terrain_under`'s x), broken by
// `billow` (the froxel's noise).
fn sea_fog(f: froxelfog::FroxelFog, p: vec3<f32>, seabed: f32, billow: f32) -> f32 {
    if (!sea_reaches(f, p)) {
        return 0.0;
    }
    let core = f.night.w;
    let above = p.y - SEA_LEVEL;
    let shore = smoothstep(SEA_SHORE.x, SEA_SHORE.y, SEA_LEVEL - seabed);
    if (shore <= 0.0) {
        return 0.0;
    }
    let drift = SEA_DRIFT * (f.camera.w / 3600.0);
    let swell = textureSampleLevel(noise_tex, repeat_samp, vec3<f32>(p.x, 0.0, p.z) / SEA_SCALE + drift + vec3<f32>(0.61, 0.29, 0.17), 0.0).r;
    let top = SEA_HEIGHT * mix(0.3, 2.2, smoothstep(0.25, 0.75, swell));
    let body = smoothstep(0.3, 0.6, billow + 0.25 * (swell - 0.5));
    // Under the surface the water carries its own fog.
    let height = exp(-max(above, 0.0) / top) * smoothstep(-2.0, 0.0, above);
    return core * shore * height * body;
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

// A light's in-scatter at a froxel: `medium` is radiance still to scale by the froxel's extinction,
// `lamp` the halo's share already scattered by the light's own lamp air (`froxelfog::lamp_air`).
struct PointScatter {
    medium: vec3<f32>,
    lamp: vec3<f32>,
}

// One table light's in-scatter at a froxel (`froxelfog`'s point law). `to_eye` is the unit direction
// from the froxel to the camera. A halo light (`froxelfog::is_halo_light`) scatters only its far
// share here: the halo pass integrates its halo per pixel.
fn point_in_scatter(i: u32, p: vec3<f32>, to_eye: vec3<f32>, room: f32, halo_count: u32) -> PointScatter {
    let pos_range = light_table.points[2u * i];
    let dl = length(pos_range.xyz - p);
    if (dl >= pos_range.w) {
        return PointScatter(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let mu = dot((p - pos_range.xyz) / max(dl, 1e-3), to_eye);
    let far = froxelfog::point_far_share(room);
    let rgb_class = light_table.points[2u * i + 1u];
    let share = froxelfog::indoor_air_share(rgb_class.w, room);
    if (share <= 0.0) {
        return PointScatter(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let excess = select(
        (froxelfog::point_halo_gain(room) - far) * exp(-dl / froxelfog::point_halo_yd(room)),
        0.0,
        froxelfog::is_halo_light(i, halo_count, rgb_class.w) || !froxelfog::has_halo(rgb_class.w),
    );
    let lit = rgb_class.rgb * share
        * (froxelfog::point_falloff(dl, pos_range.w) * froxelfog::POINT_GAIN * froxelfog::point_phase(mu));
    let lamp = froxelfog::lamp_air(modern_block.fog, rgb_class.w) * exp(-dl / froxelfog::LAMP_AIR_YD);
    return PointScatter(lit * (far + excess), lit * (excess * lamp));
}

// Every light one grid cell names, at a froxel.
fn cell_in_scatter(cell: vec4<u32>, p: vec3<f32>, to_eye: vec3<f32>, room: f32, halo_count: u32) -> PointScatter {
    var sum = PointScatter(vec3<f32>(0.0), vec3<f32>(0.0));
    let count = locallight::cell_count(cell);
    for (var i = 0u; i < count; i = i + 1u) {
        let one = point_in_scatter(locallight::cell_light(cell, i), p, to_eye, room, halo_count);
        sum.medium = sum.medium + one.medium;
        sum.lamp = sum.lamp + one.lamp;
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
    // fork: modern lighting — a room's medium reaches as far as the room does (`froxelfog::room`):
    // inside its walls the froxel breathes the building's MFOG (uniform, no near-clear range), and
    // past them the zone's own air, so a doorway's view is fogged by what is on the far side of it.
    let room = froxelfog::room_weight(f, p);
    // Under a canopy the air is thick with dust and damp: the haze gets denser and scatters the key
    // much harder there, so the light through a gap in the leaves reads as a beam. The cover is the
    // neighbourhood's (the sky map around the froxel), not the froxel's own column — a point inside a
    // beam has open sky straight above it by definition.
    let cover = canopy_cover(p);
    let ext = froxelfog::extinction(f, p, d, room, wisps, cover);
    let sigma_fog = ext.x;
    let sigma_haze = ext.y;
    let sigma_night = ext.z;
    // fork: the terrain under the froxel, read once for the ground banks and the sea fog, and only
    // when one of them can reach it.
    var terrain = vec2<f32>(-1e9, 0.0);
    if (f.ground.x > 0.5 && (f.night.x > 0.0 || sea_reaches(f, p))) {
        terrain = terrain_under(f, p);
    }
    // fork: the ground banks (outside rooms: a building's air is its own).
    let sigma_bank = ground_banks(f, p, terrain) * (1.0 - room);
    // fork: the sea fog, lit as the zone's own fog (`lum_fog`'s open-air term).
    let sigma_sea = sea_fog(f, p, terrain.x, noise) * (1.0 - room);
    let sigma = sigma_fog + sigma_haze + sigma_night + sigma_bank + sigma_sea;
    // fork: the lamps scatter off the clear-weather air (`froxelfog::halo_fog`), so rain does not
    // swell their glow, and as it is beside the eye (`froxelfog::LAMP_EYE_DEPTH`), so distance does
    // not either; the storm's own medium still dims it on the way to the eye.
    let lamp_ext = froxelfog::extinction(froxelfog::halo_fog(f), p, froxelfog::LAMP_EYE_DEPTH, room, 1.0, cover);
    let sigma_lamps = (lamp_ext.x + lamp_ext.y + lamp_ext.z) * froxelfog::LAMP_EYE_GAIN + sigma_bank + sigma_sea;

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
    // The zone's open air; the sea fog is lit as it is.
    let lum_sea = f.ambient.rgb * sky_term + f.sun.rgb * (phase * sun_vis);
    let lum_fog = mix(
        lum_sea,
        f.room_ambient.rgb * sky_term + f.room_sun.rgb * (phase * sun_vis),
        room,
    );
    // The key as the shaft haze and the banks take it; under full cover the haze scatters it twice
    // as hard (its density: `froxelfog::CANOPY_HAZE`).
    let key_lit = shaft_phase(f, phase) * sun_vis;
    let lum_haze = f.ambient.rgb * (sky_term * SHAFT_SKY_SHARE)
        + shaft_key(f) * (key_lit * SHAFT_KEY_GAIN * (1.0 + cover));
    // A bank's isotropic half takes its own share of the key (`fog.night.z`): the haze's, or the
    // whole moon where the zone's banks are ground fog.
    let bank_lit = mix(phase, f.night.z, SHAFT_ISOTROPIC) * sun_vis;
    let lum_bank = f.ambient.rgb * (sky_term * BANK_SKY_SHARE)
        + shaft_key(f) * (bank_lit * BANK_KEY_GAIN);
    var points = PointScatter(vec3<f32>(0.0), vec3<f32>(0.0));

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
    let halo_count = froxelfog::halo_lights(f, u32(light_table.rows[20].x));
    if (locallight::grid_live(lg)) {
        let pair = locallight::cell_pair(lg, p);
        if (pair.w > 0u) {
            let far_w = bitcast<f32>(pair.z);
            if (far_w < 1.0) {
                let near = cell_in_scatter(modern_block.light_cells[pair.x], p, to_eye, room, halo_count);
                points = PointScatter(near.medium * (1.0 - far_w), near.lamp * (1.0 - far_w));
            }
            if (far_w > 0.0) {
                let far = cell_in_scatter(modern_block.light_cells[pair.y], p, to_eye, room, halo_count);
                points.medium = points.medium + far.medium * far_w;
                points.lamp = points.lamp + far.lamp * far_w;
            }
        }
    } else {
        let count = min(u32(f.screen.w), u32(light_table.rows[20].x));
        for (var i = 0u; i < count; i = i + 1u) {
            let one = point_in_scatter(i, p, to_eye, room, halo_count);
            points.medium = points.medium + one.medium;
            points.lamp = points.lamp + one.lamp;
        }
    }

    var cur = vec4<f32>(
        lum_fog * sigma_fog + lum_haze * sigma_haze + (lum_haze + froxelfog::night_glow(f)) * sigma_night
            + lum_bank * sigma_bank + lum_sea * sigma_sea + points.medium * sigma_lamps + points.lamp,
        sigma,
    );

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
                // fork: fetched at its texel, not filtered: a history re-sampled trilinearly every frame
                // is blurred again every frame, so a lamp's glow diffused out of its froxels (the mean
                // drifted with the history weight) and the jitter's beat survived the blend — the
                // Darkshire town hall pulsed at the Halton period.
                let texel = min(vec3<u32>(uvw * n), vec3<u32>(f.grid.xyz) - vec3<u32>(1u));
                cur = mix(cur, textureLoad(history_tex, texel, 0), f.medium.w);
            }
        }
    }
    textureStore(scatter_out, id, cur);
}
