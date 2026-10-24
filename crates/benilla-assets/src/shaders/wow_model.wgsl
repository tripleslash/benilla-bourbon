// The model pass for M2, WMO and ground-clutter meshes, lit and fogged in gamma space:
//   M2:           color = clamp01(A + D·I·(4/17)(0.375 + 2μ + 1.875μ²)) × tex × tint, μ = N·L
//   clutter, WMO: color = clamp(ambient + diffuse·max(N·L, 0)) × tex × tint
//   stage 1:      color = combine(color, tex1), tex1 at the generated coordinate (presets below)
//   fog:          color = mix(fog_color, color, fog_factor); out = color, raw gamma
// The M2 law is the order-2 SH lobe of `Shaders\Vertex\Model2.bls` (cvar `M2UseShaders`, default
// 1); M2's one FFP light site (`70bdf6`) runs only with that cvar off. Clutter and WMO are the FFP
// light (GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL); clutter's normal is the terrain normal under
// the tuft, which the reference writes onto the clutter vertex.
// Not built: a specular term (the M2 per-material shininess is only inferred) and the WMO
// per-group authored colour.

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import benilla_modern::sunshadow
#import benilla_modern::froxelfog
#import benilla_modern::probegi
#import benilla_modern::locallight
#import benilla::env_frame
// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module
// wxl-experimental-wind, grass/GrassWind.cpp (grass vertex displacement seam).
#import benilla::wind_hook
// fork: rain on surfaces (the Everwood port).
#import benilla::wet_hook
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    pbr_bindings,
    forward_io::VertexOutput,
    mesh_view_bindings::{lights, view},
    shadows,
    mesh_functions,
}

// bevy_pbr 0.18.1's `forward_io::FragmentOutput`. No depth output: a fragment depth write costs
// the pipeline early-Z, so the sky lane pins its depth in the vertex stage.
struct WowFragOut {
    @location(0) color: vec4<f32>,
}

// Per-material uniforms at binding 100, in `WowModelExt`'s field order (materials.rs).
//   clutter_fade: x = plateau-end view depth (yd, 0.75·far), y = ramp-zero view depth (the ~70 yd
//     detail-doodad horizon `[0x867958]`), z = the batch marker bits, w = clutter
//   model_flags: x = WMO, y = fade blend twin, z = interior (a WMO interior group, or an interior
//     M2 lit by its SH probe), w = unlit fullbright (M2 UNLIT 0x01 or Mod/Mod2x, or WMO UNLIT on
//     an exterior-group batch; the interior drawer ignores it)
struct ModelParams {
    clutter_fade: vec4<f32>,
    model_flags: vec4<f32>,
    // x = the terrain-shade selector (see the doodad sun), y = the WMO batch order, zw = the
    // UV-scroll seed.
    sun_scale: vec4<f32>,
    // xyz = the animated M2Color tint (identity when static); w = the WMO interior batch class:
    // 0 exterior law, 1 INT, 2 TRANS.
    tint: vec4<f32>,
    // WMO glass, 0 on M2: xyz = the MOMT SIDN (0x10) emissive (gamma /255), w = MOMT WINDOW (0x20).
    sidn: vec4<f32>,
    // Rows of `wow_light.matanim`, 0 = identity: x = UV scroll, y = tint, z = texture-transform
    // affine, w = the UI tile's cell clip.
    anim_slots: vec4<f32>,
    // The second texture stage: x = bound (1) or not (0), y = its combine preset 0-5; zw unused.
    stage1: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> m: ModelParams;
// Stage 1's texture, sampled at the generated coordinate `env_uv` when `m.stage1.x` is set.
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var stage1_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var stage1_sampler: sampler;

// **How dark a fully covered character shadow gets** — the same number `terrain.wgsl` carries and
// `benilla-world`'s `static_gx.wgsl` carries; move all three together. Here (and in `static_gx`)
// it scales the SUN term only — the subtract-and-restore below keeps the authored ambient — so at
// one value a model reads slightly lighter than the ground under it, which is the intended
// reading. 0.70, raised from the 0.45 this first shipped at (a near-black silhouette).
const SHADOW_SUN_FLOOR: f32 = 0.70;

// fork: modern lighting (FORK.md, P6) — the interior look's four numbers.
// MOCV.a stops being a multiplier on a bake and becomes what it was authored as, an emissive mask;
// SIDN stops being a night decal and becomes a real emitter, still ramped by the night fraction but
// never off, because a lit pane reads as a light source from a dark room at noon; and a room whose
// probes have not answered yet falls back on its own MOHD ambient rather than going black.
const MODERN_MOCV_EMISSIVE: f32 = 2.0;
const MODERN_SIDN_GAIN: f32 = 2.5;
const MODERN_SIDN_DAY: f32 = 0.35;
const MODERN_ROOM_FLOOR: f32 = 0.30;
// fork: a modern interior surface keeps at least this share of its classic light (the batch class's
// bake, MOCV included); `static_gx.wgsl` carries the same number.
const MODERN_BAKE_FLOOR: f32 = 0.9;

// The shared global light (lighting::global_light): ONE storage buffer every material reads, updated
// once/frame in place — replaces the per-material light/fog uniforms the old apply_wow_lighting re-pushed
// each frame. The model reads rows 0-2 (ambient/diffuse/sun) + fog, plus rows 6-12: the disassembled
// `Model2.bls` probe of the day/night scene light at intensity 1 — the reference's own exterior M2
// response, and since 0803 the live law on the doodad/entity lane.
// Shared with terrain.wgsl — a tree and the dirt under it fade into the same haze.
// (light_spec is part of the prefix layout but the model path never reads it.)
struct WowLight {
    light_ambient: vec4<f32>, // rgb ambient; w = Mod2x scale
    light_diffuse: vec4<f32>, // rgb sun diffuse; w = clamp-light flag (>0.5 ⇒ saturate)
    light_sun: vec4<f32>,     // xyz sun travel dir (to-light = −xyz); w = directional enable
    light_spec: vec4<f32>,    // terrain's specular (w = its shininess); unread here
    fog_color: vec4<f32>,     // rgb row-7 fog (gamma); w = enable (>0.5)
    fog_params: vec4<f32>,    // x = start, y = end, w = farclip wall; z unused
    // Rows 6-12: the reference's `Model2.bls` order-2 SH probe of the scene light at intensity 1:
    // ambient DC in the c10 `.w` lanes, the sun's bands in c10.xyz, c13 and c16.xyz. Every sun band
    // is linear in the committed colour, so all scale by the intensity (never I²).
    sh_c10_r: vec4<f32>,
    sh_c10_g: vec4<f32>,
    sh_c10_b: vec4<f32>,
    sh_c13_r: vec4<f32>,
    sh_c13_g: vec4<f32>,
    sh_c13_b: vec4<f32>,
    // xyz = the c16 quadratic band (x²−y²) per channel; w unused.
    sh_c16: vec4<f32>,
    _water: array<vec4<f32>, 4>, // rows 13-16: the liquid swatches, unread here
    // x = the SIDN night fraction (1 overnight, 0 by day); yzw = the sun's SH DC at intensity 1.
    grade: vec4<f32>,
    // Rows 18-19: the interior fog, the 4 s camera-in-WMO MFOG crossfade (the scene fog outdoors).
    wmo_fog_color: vec4<f32>,    // rgb interior fog (gamma); w = enable (mirrors fog_color.w)
    wmo_fog_params: vec4<f32>,   // x = start yd, y = end yd; zw unused
    // Row 20 `.x` = the point-light count, then per light `[pos.xyz, range]`, `[rgb, 0]`. Here, not
    // in Bevy's clusterables, which the view layout gives the fragment stage only.
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
    // fork: the environment block (`benilla::env_frame`: wind, wetness, benders), then the
    // rain-shelter region (`weather::shelter`): `[origin_x, origin_z, 1 / cell, active]`,
    // `[base_y, cells per side, 0, 0]` and one packed word a cell.
    env: env_frame::EnvFrame,
    shelter_hdr: vec4<f32>,
    shelter_cfg: vec4<f32>,
    shelter: array<u32, 16384>,
    // The interior-prop SH probes, 7 rows per slot (`MAX_PROP_PROBES` = 8192 slots). Only this
    // shader declares this tail; the other shaders bind the same buffer by its prefix.
    prop_probes: array<vec4<f32>, 57344>,
    // Per rig slot (sizes mirrored in rig_palette.rs): the base bone index into `palettes`, whose
    // 3 rows per bone are `rig_from_joint × inverse_bindpose` from the rig's own origin.
    rig_table: array<u32, 2048>,
    // Per rig slot: the CM2 body tint (`model+0x184/188/18c`) packed `0xFFRRGGBB` like the
    // reference's node value (`0x60d840`: `param | 0xff000000`); 0 is identity.
    rig_tint: array<u32, 2048>,
    // Per rig slot: the world origin its palette rows are measured from.
    rig_origin: array<vec4<f32>, 2048>,
    // The mat-anim rows (size mirrored in mat_anim_table.rs), row 0 zero: a UV-scroll delta (xy),
    // a tint delta (xyz), a texture-transform affine `[cos − 1, sin, sx − 1, sy − 1]` or a UI cell.
    matanim: array<vec4<f32>, 2048>,
    // Per rig slot, the straddle waterline (size mirrored in straddle.rs): x = its world height
    // (Bevy Y), y = the side the near copy keeps (+1 above, −1 below, 0 not straddling).
    water_clip: array<vec2<f32>, 2048>,
    palettes: array<vec4<f32>>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;
// fork: modern lighting (FORK.md, P2)
@group(#{MATERIAL_BIND_GROUP}) @binding(121) var modern_shadow_map: texture_depth_2d_array;
@group(#{MATERIAL_BIND_GROUP}) @binding(122) var modern_shadow_samp: sampler_comparison;
@group(#{MATERIAL_BIND_GROUP}) @binding(123) var<storage, read> modern_block: sunshadow::ModernBlock;
// fork: modern lighting (FORK.md, P4) — the integrated froxel fog volume.
@group(#{MATERIAL_BIND_GROUP}) @binding(124) var modern_fog_volume: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(125) var modern_fog_samp: sampler;

// fork: modern lighting (FORK.md, P5) — the probe field's irradiance at this
// fragment: the eight probes around it in the finest cascade that owns the point, and whatever a
// coarser one can still answer beyond that. `rgb` is in the ambient row's own units (it replaces
// that row); `w` is how much of the cell answered, which is 0 wherever the field has nothing to say
// and the sky-occlusion map stays in charge.
// `lite` (the water mirror, `sunshadow::mirror_view`): the finest cascade that owns the point answers
// alone, without the Chebyshev visibility.
fn modern_gi_ambient(p: vec3<f32>, nrm: vec3<f32>, recv_interior: f32, lite: bool) -> vec4<f32> {
    var g = modern_block.gi;
    if (lite) {
        g.interior.w = f32(u32(g.interior.w) & ~2u);
    }
    if (g.params.x < 0.5 || g.params.z <= 0.0) {
        return vec4<f32>(0.0);
    }
    var sum = vec3<f32>(0.0);
    var answered = 0.0;
    var left = 1.0;
    let count = u32(g.dims.w);
    for (var c = 0u; c < count; c = c + 1u) {
        let pk = probegi::pick(g, c, p, nrm, recv_interior);
        if (pk.inside <= 0.0) {
            continue;
        }
        var acc = probegi::start();
        for (var k = 0u; k < 8u; k = k + 1u) {
            acc = probegi::accum(acc, g, pk, k, modern_block.probes[probegi::tap_index(g, pk, k)], p, nrm);
        }
        let cell_gi = probegi::finish(acc, pk.recv);
        let share = select(left * pk.inside, 1.0, lite);
        sum = sum + cell_gi.rgb * (cell_gi.w * share);
        answered = answered + cell_gi.w * share;
        // Only the share that ANSWERED is spent: a cell the fine cascade is unsure of (freshly
        // scrolled probes, probes inside walls) hands the rest to the coarser cascade rather than
        // to the sky map. From afar that coarser cascade answered this point whole, so falling back
        // to it keeps the point from changing as the player walks up to it. The mirror still
        // takes the finest cascade alone.
        left = left - select(share * cell_gi.w, 1.0, lite);
        if (left <= 0.001) {
            break;
        }
    }
    if (answered <= 0.001) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(sum * (g.params.z / answered), clamp(answered, 0.0, 1.0));
}

// fork: modern lighting (FORK.md, P6) — the local lights that reach this
// fragment, read out of the world grid the binning pass fills: the WoW falloff per light, under its
// cube shadow when it holds an atlas slot. The walk lives here rather than in
// `benilla_modern::locallight` for the same reason `modern_gi_ambient` does — the light table is
// this lane's own binding, which a WGSL module cannot reach. `lite` (the water mirror) skips the cube
// shadows.
// fork: modern lighting (P6, gated in P8) — `interior` is the receiver's own INTERIOR bit. A light
// that names no WMO rooms may not light an interior surface: a local light has no occluder unless
// it holds one of the twelve cube slots, and `global_light`'s room term admits every room-less
// light by law, so without this the street lamp outside lights the abbey's furniture through the
// wall. The same rule `static_gx.wgsl` applies to the room's own surfaces, so a chair and the floor
// it stands on agree.
fn modern_local_lights(p: vec3<f32>, nrm: vec3<f32>, interior: bool, lite: bool) -> vec3<f32> {
    // fork: P8 — the near cell, the far cell, and how far into the seam between the two levels
    // this fragment stands (`locallight::cell_pair`). Almost every fragment reads one cell; only
    // the few near cells at the near level's edge read both and cross-fade.
    let pair = locallight::cell_pair(modern_block.lights, p);
    if (pair.w < 1u) {
        return vec3<f32>(0.0);
    }
    let far_w = bitcast<f32>(pair.z);
    var sum = vec3<f32>(0.0);
    if (far_w < 1.0) {
        let near_cell = modern_block.light_cells[pair.x];
        sum = modern_cell_lights(near_cell, p, nrm, interior, lite) * (1.0 - far_w);
    }
    if (far_w > 0.0) {
        let far_cell = modern_block.light_cells[pair.y];
        sum = sum + modern_cell_lights(far_cell, p, nrm, interior, lite) * far_w;
    }
    return sum;
}

// One grid cell's lights on one fragment — the walk `modern_local_lights` runs once per cell it
// reads.
fn modern_cell_lights(
    cell: vec4<u32>,
    p: vec3<f32>,
    nrm: vec3<f32>,
    interior: bool,
    lite: bool,
) -> vec3<f32> {
    let g = modern_block.lights;
    let count = locallight::cell_count(cell);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let id = locallight::cell_light(cell, i);
        let pos_range = wow_light.points[2u * id];
        // `w` is 1 for a light that belongs to a building (`global_light`'s packer).
        if (interior && wow_light.points[2u * id + 1u].w < 0.5) {
            continue;
        }
        let to_light = pos_range.xyz - p;
        let d = length(to_light);
        // fork: P8b — past its range a light's window is exactly zero (`locallight::falloff`), so
        // it adds nothing here and its cube lookup below would be wasted. The binning pass lists a
        // light in every cell its sphere touches, so much of a listed light's cell lies outside it.
        if (d >= pos_range.w) {
            continue;
        }
        let nl = max(dot(nrm, to_light / max(d, 1e-4)), 0.0);
        if (nl <= 0.0) {
            continue;
        }
        var vis = 1.0;
        // fork: P8b — only a light that holds a cube slot has a shadow to look up. The rest (every
        // light on Ultra, most of them everywhere) skip the slot's two parameter rows and the lookup
        // call, which would only have returned 1 (`locallight::local_visibility`'s first test).
        let slot = locallight::cell_slot(cell, i);
        if (!lite && slot < locallight::MAX_SHADOWED) {
            // The slot's own near plane and cross-fade, from the block's spare rows (rows 0-2 and
            // 3-5, four slots a row). A synthesized light's near plane is its fixture's radius, so
            // a lantern is clipped out of its own cube instead of casting its frame across the
            // floor. Each ROW is loaded to a value before a component is picked: indexing a vector
            // component dynamically straight off a storage buffer is not something every backend
            // takes, and what it costs is the whole module.
            let row = modern_block.light_spare[slot / 4u];
            let frow = modern_block.light_spare[3u + slot / 4u];
            vis = locallight::local_visibility(
                modern_shadow_map,
                modern_shadow_samp,
                g,
                slot,
                modern_block.light_shadow[slot].xyz,
                p,
                nrm,
                row[slot % 4u],
                frow[slot % 4u],
            );
        }
        sum = sum + wow_light.points[2u * id + 1u].rgb * (locallight::falloff(d, pos_range.w) * nl * vis);
    }
    return sum;
}

// The M2 cutout alpha-test reference, 224/255; in sync with `materials::VANILLA_ALPHA_KEY_REF`.
const VANILLA_ALPHA_KEY: f32 = 0.8784314;

// fork: modern lighting — how far toward the key light a skinned body's shadow lookup starts (yd):
// clear of its own limbs (the body is at most ~1 yd thick along any sun ray), well under a canopy.
const RIG_SHADOW_LIFT: f32 = 1.5;

// The detail-doodad (ground clutter) texture LOD bias — the reference sets
// `D3DSAMP_MIPMAPLODBIAS = +0.25` on stage 0 for its detail-doodad pass alone (`0x6813f4`,
// capture-confirmed on all 29 doodad batches). It is what makes the distance cut resolve per pixel
// rather than per leaf; the full derivation is at the use site in `fragment`, and in wow-re
// `terrain/scratch/doodad-fade-pixel-granularity.md`.
const DETAIL_DOODAD_LOD_BIAS: f32 = 0.25;

// Bevy's `VertexOutput` (same fields, locations and defs; no tangents, morphs or visibility
// ranges) plus our interpolants; the fragment rebuilds a `VertexOutput` from it.
struct WowVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
    // The point-light sum `Σ att·sat(N·L)·colour`, per vertex like the reference FFP.
    @location(8) point_lit: vec3<f32>,
    // fork: modern lighting (FORK.md, P3) — the vertex's point-light selection (flat).
    @location(11) @interpolate(flat) point_sel: vec3<u32>,
#ifdef WOW_MERGED_FADE
    // A merged blob's per-placement fade alpha, constant over the placement.
    @location(9) merged_fade: f32,
#endif
#ifdef WOW_MERGED_SLOT
    @location(10) @interpolate(flat) merged_slot: u32,
#endif
    // Stage 1's generated coordinate, (0, 0) when the material binds no stage 1. Location 12: the
    // fork's `point_sel` holds 11.
    @location(12) env_uv: vec2<f32>,
}

// The generated environment coordinate of `Model2.bls` (its programs for a stage whose coordinate
// lookup is above 2, `0x70b8bd`): in view space R = normalize(P − 2(P·N)N), uv = R.xy·0.5 + 0.5,
// per vertex. The reference view basis (`0x5c3e70`) and Bevy's −Z-forward one differ only in z,
// so R.xy needs no fixup.
fn wow_env_uv(p_view: vec3<f32>, n_view: vec3<f32>) -> vec2<f32> {
    let refl = normalize(p_view - 2.0 * dot(p_view, n_view) * n_view);
    return refl.xy * 0.5 + vec2<f32>(0.5, 0.5);
}

// Stage 1's colour combine on `prev`, the stage-0 result, by gx preset as the default device,
// Direct3D, maps it (`0x80a25c` = 4, 4, 13, 7, 5, 12 as `D3DTOP`, args TEXTURE and CURRENT, which
// the device never changes): 0 and 1 modulate, 2 BLENDTEXTUREALPHA interpolates by the texel's
// alpha, 3 adds, 4 modulates ×2, 5 BLENDDIFFUSEALPHA interpolates by `diffuse_a`, the vertex
// colour's alpha. Each stage's result saturates.
fn stage1_rgb(preset: u32, prev: vec3<f32>, tex: vec4<f32>, diffuse_a: f32) -> vec3<f32> {
    var c = prev * tex.rgb;
    if (preset == 2u) {
        c = mix(prev, tex.rgb, tex.a);
    } else if (preset == 3u) {
        c = prev + tex.rgb;
    } else if (preset == 4u) {
        c = 2.0 * prev * tex.rgb;
    } else if (preset == 5u) {
        c = mix(prev, tex.rgb, diffuse_a);
    }
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// A stage's alpha on `prev`, the previous stage's (the diffuse alpha, for stage 0), by gx preset
// (`0x80a274`, args TEXTURE and CURRENT): 1 modulates, 3 adds, 4 modulates ×2, the rest
// (SELECTARG2) pass the previous.
fn stage_alpha(preset: u32, prev: f32, tex: f32) -> f32 {
    var a = prev;
    if (preset == 1u) {
        a = prev * tex;
    } else if (preset == 3u) {
        a = prev + tex;
    } else if (preset == 4u) {
        a = 2.0 * prev * tex;
    }
    return clamp(a, 0.0, 1.0);
}

// Zero-safe normalize. M2s author `(0,0,0)` normals (`Creature\QuirajProphet`) and the reference
// lights them: `Model2.bls` normalizes with `RSQ`/`MUL`, where `0 × INF` is 0, leaving the SH DC
// term. `normalize(0)` is NaN, which clamps to 0 and blacks the batch. Wrapping `normalize`
// rather than `v · inverseSqrt(l2)` keeps every nonzero normal's bits.
fn wow_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    return select(vec3<f32>(0.0), normalize(v), l2 > 1e-12);
}

// The point lights at P: the reference commits at most three per draw, the nearest to the
// receiving unit's own position `anchor` (`0x71bf90` gathers by squared distance, `0x71c730` seats
// slots 1-3). Range bounds candidacy only; a selected light falls off as `1/(0.7·d + 0.03·d²)`,
// diffuse only, on the submitted normal (no GL_LIGHT_MODEL_TWO_SIDE). Mirrored in terrain.wgsl.
fn point_light_sum(P: vec3<f32>, N: vec3<f32>, anchor: vec3<f32>) -> vec3<f32> {
    let count = u32(wow_light.point_count.x);
    var sel = array<u32, 3>(0u, 0u, 0u);
    var sd = array<f32, 3>(1e30, 1e30, 1e30);
    for (var i = 0u; i < count; i = i + 1u) {
        let pos_range = wow_light.points[2u * i];
        let dv = pos_range.xyz - anchor;
        let d2 = dot(dv, dv);
        if (d2 > pos_range.w * pos_range.w) {
            continue;
        }
        if (d2 < sd[0]) {
            sd[2] = sd[1]; sel[2] = sel[1];
            sd[1] = sd[0]; sel[1] = sel[0];
            sd[0] = d2; sel[0] = i;
        } else if (d2 < sd[1]) {
            sd[2] = sd[1]; sel[2] = sel[1];
            sd[1] = d2; sel[1] = i;
        } else if (d2 < sd[2]) {
            sd[2] = d2; sel[2] = i;
        }
    }
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < 3u; s = s + 1u) {
        if (sd[s] > 9.9e29) {
            break;
        }
        let pos_range = wow_light.points[2u * sel[s]];
        let to_light = pos_range.xyz - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        sum += wow_light.points[2u * sel[s] + 1u].rgb * (atten * nl);
    }
    return sum;
}

// fork: modern lighting (FORK.md, P3) — the same ≤3-nearest selection as `point_light_sum`,
// returned as table indices + 1 (0 = empty) so the fragment can light per pixel.
fn point_light_select(anchor: vec3<f32>) -> vec3<u32> {
    let count = u32(wow_light.point_count.x);
    var sel = vec3<u32>(0u);
    var sd = vec3<f32>(1e30);
    for (var i = 0u; i < count; i = i + 1u) {
        let pos_range = wow_light.points[2u * i];
        let dv = pos_range.xyz - anchor;
        let d2 = dot(dv, dv);
        if (d2 > pos_range.w * pos_range.w) {
            continue;
        }
        if (d2 < sd.x) {
            sd.z = sd.y; sel.z = sel.y;
            sd.y = sd.x; sel.y = sel.x;
            sd.x = d2; sel.x = i + 1u;
        } else if (d2 < sd.y) {
            sd.z = sd.y; sel.z = sel.y;
            sd.y = d2; sel.y = i + 1u;
        } else if (d2 < sd.z) {
            sd.z = d2; sel.z = i + 1u;
        }
    }
    return sel;
}

// fork: modern lighting (FORK.md, P3) — the selected lights evaluated per pixel: the WoW falloff
// `1/(0.7d + 0.03d²)` × N·L × a smooth window that reaches zero at the light's range.
fn point_light_eval(sel: vec3<u32>, P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < 3u; s = s + 1u) {
        let id = sel[s];
        if (id == 0u) {
            continue;
        }
        let pos_range = wow_light.points[2u * (id - 1u)];
        let to_light = pos_range.xyz - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let q = d / max(pos_range.w, 1e-3);
        let window = clamp(1.0 - q * q * q * q, 0.0, 1.0);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        sum = sum + wow_light.points[2u * (id - 1u) + 1u].rgb * (atten * nl * window * window);
    }
    return sum;
}

// The centre of the MCNK chunk under P, the light anchor for world-merged clutter, since the
// reference gathers lights per terrain chunk (mirrored in terrain.wgsl). WoW x/y are Bevy −z/−x
// and the grid is symmetric, so snapping Bevy x/z lands on the same cells. The height is the
// vertex's own, not the chunk record's: lights sit near the surface.
fn mcnk_cell_anchor(P: vec3<f32>) -> vec3<f32> {
    let cell = 533.33333 / 16.0;
    let half = 32.0 * 533.33333;
    let ix = floor((half + P.x) / cell);
    let iz = floor((half + P.z) / cell);
    return vec3<f32>((ix + 0.5) * cell - half, P.y, (iz + 0.5) * cell - half);
}

#ifdef WOW_MERGED_FADE
// The doodad fade curve (`0x683f80`, in sync with `model_fade::doodad_fade_alpha`): alpha =
// 1 − (d − start)/range, d = horizontal distance − radius, over a size-bucketed band.
fn merged_fade_alpha(radius: f32, horiz_dist: f32) -> f32 {
    if (radius > 7.0) {
        return 1.0;
    }
    var start = 150.0;
    var range = 50.0;
    if (radius <= 0.5) {
        start = 40.0;
        range = 10.0;
    } else if (radius <= 2.5) {
        start = 100.0;
        range = 25.0;
    }
    let d = horiz_dist - radius;
    return clamp(1.0 - (d - start) / range, 0.0, 1.0);
}
#endif

// Bevy 0.18's `forward_io::Vertex` at Bevy's locations, plus the palette joints at 10/11 (appended
// by `WowModelExt::specialize` for a mesh with `ATTRIBUTE_WOW_JOINT_INDEX`) and the merged-blob
// attributes at 12/13.
struct WowVertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef WOW_RIG_SKIN
    @location(10) joint_indices: vec4<u32>,
    @location(11) joint_weights: vec4<f32>,
#endif
#ifdef WOW_MERGED_FADE
    // The placement fade sphere: xyz = world centre, w = fade radius.
    @location(12) fade_sphere: vec4<f32>,
#endif
#ifdef WOW_MERGED_SLOT
    // The interior-prop SH-probe slot, in place of the per-entity MeshTag payload.
    @location(13) merged_slot: u32,
#endif
}

#ifdef WOW_RIG_SKIN
fn wow_rig_slot(instance_index: u32) -> u32 {
    return (mesh_functions::get_tag(instance_index) >> 19u) & 0x7ffu;
}

// Blends the four weighted bones' palette rows into `rig_from_local`, which replaces the mesh's
// world matrix like Bevy's `skin_model`; its translation is relative to `rig_origin[slot]`.
fn wow_skin_model(instance_index: u32, indices: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    let base = wow_light.rig_table[wow_rig_slot(instance_index)];
    let b0 = 3u * (base + indices.x);
    let b1 = 3u * (base + indices.y);
    let b2 = 3u * (base + indices.z);
    let b3 = 3u * (base + indices.w);
    let r0 = weights.x * wow_light.palettes[b0]
        + weights.y * wow_light.palettes[b1]
        + weights.z * wow_light.palettes[b2]
        + weights.w * wow_light.palettes[b3];
    let r1 = weights.x * wow_light.palettes[b0 + 1u]
        + weights.y * wow_light.palettes[b1 + 1u]
        + weights.z * wow_light.palettes[b2 + 1u]
        + weights.w * wow_light.palettes[b3 + 1u];
    let r2 = weights.x * wow_light.palettes[b0 + 2u]
        + weights.y * wow_light.palettes[b1 + 2u]
        + weights.z * wow_light.palettes[b2 + 2u]
        + weights.w * wow_light.palettes[b3 + 2u];
    // r0/r1/r2 are the affine's rows; a WGSL matrix is column-major.
    return mat4x4<f32>(
        vec4<f32>(r0.x, r1.x, r2.x, 0.0),
        vec4<f32>(r0.y, r1.y, r2.y, 0.0),
        vec4<f32>(r0.z, r1.z, r2.z, 0.0),
        vec4<f32>(r0.w, r1.w, r2.w, 1.0),
    );
}

// bevy_pbr::skinning's inverse-transpose via the adjugate, verbatim.
fn inverse_transpose_3x3m(in: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(in[1], in[2]);
    let y = cross(in[2], in[0]);
    let z = cross(in[0], in[1]);
    let det = dot(in[2], z);
    return mat3x3<f32>(x / det, y / det, z / det);
}

fn wow_skin_normals(frame_from_local: mat4x4<f32>, normal: vec3<f32>) -> vec3<f32> {
    return wow_normalize(
        inverse_transpose_3x3m(mat3x3<f32>(
            frame_from_local[0].xyz,
            frame_from_local[1].xyz,
            frame_from_local[2].xyz
        )) * normal
    );
}
#endif

// Bevy 0.18's `mesh.wgsl` vertex stage plus our skinning and point lights. A `MaterialExtension`
// replaces the whole stage, so this must track Bevy's on upgrades.
@vertex
fn vertex(vertex: WowVertex) -> WowVsOut {
    var out: WowVsOut;

    let mesh_world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    // Precision: `frame_from_local` holds the orientation and a small translation, `frame_origin`
    // the ~9 k yd world position, so no f32 product mixes the two (an f32 ULP at 9 k is ~1 mm).
#ifdef WOW_RIG_SKIN
    var frame_from_local = wow_skin_model(
        vertex.instance_index,
        vertex.joint_indices,
        vertex.joint_weights
    );
    let frame_origin = wow_light.rig_origin[wow_rig_slot(vertex.instance_index)].xyz;
#else
    var frame_from_local = mesh_world_from_local;
    let frame_origin = mesh_world_from_local[3].xyz;
    frame_from_local[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
#endif

#ifdef VERTEX_NORMALS
#ifdef WOW_RIG_SKIN
    out.world_normal = wow_skin_normals(frame_from_local, vertex.normal);
#else
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index
    );
#endif
#endif

#ifdef VERTEX_POSITIONS
    // Precision: camera-relative to clip space; `clip_from_world × p_world` cancels
    // catastrophically with camera and geometry near 9 k yd. `world_position` is absolute again:
    // lighting and fog need no such precision.
    var p_cam = (frame_from_local * vec4<f32>(vertex.position, 1.0)).xyz
        + (frame_origin - view.world_position);
    // fork: foliage wind. Clutter UV_B is `(height from base, per-tuft phase)`, authored by
    // `clutter.rs`; UV_0 and the vertex colour keep their texture and tint meanings. The
    // depth-prime twin runs this same stage, so both passes displace alike.
#ifdef VERTEX_UVS_B
    if (m.clutter_fade.w > 0.5) {
        p_cam += wind_hook::grass_offset(
            p_cam + view.world_position, vertex.uv_b.x, vertex.uv_b.y, view.world_position,
            wow_light.env
        );
    }
#endif
    // fork: a retained tree inside its fade band draws here with the same anchor and field maths,
    // so it does not pop. The leaf marker is material bit 14 (`model_render::FOLIAGE_WIND_MARKER`).
    if ((u32(m.clutter_fade.z) & 16384u) != 0u && m.clutter_fade.w <= 0.5) {
        p_cam += wind_hook::tree_offset(
            p_cam + view.world_position, frame_origin, view.world_position, wow_light.env
        );
    }
    out.world_position = vec4<f32>(p_cam + view.world_position, 1.0);
    let view_rot = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    );
    out.position = view.clip_from_view * vec4<f32>(view_rot * p_cam, 1.0);
    // WMO batch order (`sun_scale.y`, 0 off WMO): the reference layers coplanar batches by MOBA
    // draw order under depth-write + LEQUAL; Bevy reorders draws, so a later batch must win the
    // reverse-Z GreaterEqual test. Scaling clip z by (1 + n·2⁻²³) raises z/w by n ULPs. Uniform
    // data, not a `DepthBiasState`, which would make every batch index its own pipeline.
    out.position.z *= 1.0 + m.sun_scale.y * 1.1920929e-7;
#ifdef WOW_SKY_DEPTH
    // The WMO-skybox lane (`clutter_fade.z` bit 13): clip z = 0 is reverse-Z infinitely far, so
    // the world always draws over the sky shell (`benilla_world::sky_order`).
    out.position.z = 0.0;
#endif
#ifdef WOW_MERGED_FADE
    // A fully faded placement leaves the clip volume, so its triangles never rasterize.
    let fade_d = distance(view.world_position.xz, vertex.fade_sphere.xz);
    out.merged_fade = merged_fade_alpha(vertex.fade_sphere.w, fade_d);
    if (out.merged_fade <= 0.0) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
#ifdef WOW_MERGED_SLOT
    out.merged_slot = vertex.merged_slot;
#endif
#endif
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
    // Env-mapped batches (`clutter_fade.z` bit 12; `texture_unit_lookup[texCoordSet] > 2` at
    // `0x70b8bd`) generate their texcoord per vertex (`wow_env_uv`).
#ifdef VERTEX_POSITIONS
#ifdef VERTEX_NORMALS
    if ((u32(m.clutter_fade.z) & 4096u) != 0u) {
        out.uv = wow_env_uv(view_rot * p_cam, normalize(view_rot * out.world_normal));
    }
#endif
#endif
#endif
    // Stage 1 takes the same generated coordinate in its own interpolant: a two-texture unit's
    // second lookup is the environment one in every stock model (`0xffff`).
    out.env_uv = vec2<f32>(0.0);
#ifdef VERTEX_POSITIONS
#ifdef VERTEX_NORMALS
    if (m.stage1.x > 0.5) {
        out.env_uv = wow_env_uv(view_rot * p_cam, normalize(view_rot * out.world_normal));
    }
#endif
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
    // MONKEY (glue scenes): a stage 1 on its own UV set (`stage1.w`, the OctoWoW HD login scene)
    // reads UV set B instead of the generated coordinate, combined by the same presets.
    if (m.stage1.w > 0.5) {
        out.env_uv = vertex.uv_b;
    }
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

    // Point lights: none on WMO surfaces, none here on interior M2 props (their group's MOLR lights
    // are in the SH probe), else the ≤3 nearest to the instance origin (clutter: its MCNK chunk).
    if (m.model_flags.x > 0.5 || m.model_flags.z > 0.5) {
        out.point_lit = vec3<f32>(0.0);
    } else {
        var anchor = mesh_world_from_local[3].xyz;
        if (m.clutter_fade.w > 0.5) {
            anchor = mcnk_cell_anchor(out.world_position.xyz);
        }
#ifdef WOW_MODERN
#ifdef MODERN_VERTEX_WALKS
        out.point_lit = point_light_sum(out.world_position.xyz, out.world_normal, anchor);
        out.point_sel = point_light_select(anchor);
#else
        // fork: perf — each is a walk of the whole light table, and a fragment reads only one of
        // them (`fragment`'s `point_diffuse`): the three-nearest selection in a modern frame, the
        // Gouraud sum in a classic one (a booth through this pipeline). So a vertex makes that
        // one. `MODERN_VERTEX_WALKS` is the A/B lever (`$WOW_MODERN_VERTEXWALKS=1`): both, as
        // before.
        //
        // And neither where the light grid answers the fragment instead (its own test, in
        // `fragment`: the grid live, not a rig, not the water mirror, not an unlit batch). The
        // block is bound fragment-only here, so the grid's live bit and the mirror's plane reach
        // this stage in the light row's spare lanes (`modern::shadow::VERTEX_HINT_OFFSET`): `.z`
        // is 1 only in a frame both were written to the block, and `.w` is then the plane
        // `sunshadow::mirror_view` tests.
        let grid_answers = wow_light.point_count.z > 0.5
            && view.world_position.y >= wow_light.point_count.w - 0.25
            && m.sun_scale.x < 1.5
            && m.model_flags.w <= 0.5;
        if (wow_light.point_count.y > 0.5) {
            out.point_lit = vec3<f32>(0.0);
            if (grid_answers) {
                out.point_sel = vec3<u32>(0u);
            } else {
                out.point_sel = point_light_select(anchor);
            }
        } else {
            out.point_lit = point_light_sum(out.world_position.xyz, out.world_normal, anchor);
            out.point_sel = vec3<u32>(0u);
        }
#endif
#else
        out.point_lit = point_light_sum(out.world_position.xyz, out.world_normal, anchor);
#endif
    }
    return out;
}

@fragment
fn fragment(in: WowVsOut, @builtin(front_facing) is_front: bool) -> WowFragOut {
    // The UI model tile's cell clip: the reference gives each `<Model>` pane its widget rect as the
    // viewport; our panes share an atlas, so the tile passes its cell as a mat-anim row
    // (`anim_slots.w`, `[min.x, min.y, max.x, max.y]` in atlas texels) and this is the scissor.
    if (m.anim_slots.w > 0.5) {
        let r = wow_light.matanim[u32(m.anim_slots.w)];
        if (in.position.x < r.x || in.position.y < r.y
            || in.position.x > r.z || in.position.y > r.w) {
            discard;
        }
    }
    // The far-clip wall: discard past `farclip` (`fog_params.w`, 0 = off) by planar eye depth, as
    // terrain.wgsl does.
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }
#ifdef WOW_WATER_CLIP
    // The straddle split: a translucent model crossing its water plane draws on each side of the
    // water pass (the far copy has `clutter_fade.z` bit 11) and each copy keeps its half, the
    // reference's `M2UseClipPlanes` plane at the waterline. In sync with `straddle::keeps`.
    let water_clip = wow_light.water_clip[(mesh_functions::get_tag(in.instance_index) >> 19u) & 0x7ffu];
    if (water_clip.y != 0.0) {
        let far_copy = (u32(m.clutter_fade.z) & 2048u) != 0u;
        let keep_side = select(water_clip.y, -water_clip.y, far_copy);
        if (keep_side * (in.world_position.y - water_clip.x) < 0.0) {
            discard;
        }
    }
#endif
    // Rebuild Bevy's `VertexOutput` with the M2 texture transform (in sync with
    // `tex_anim::uv_transform`): uv' = R((uv + t − p) ⊙ s) + p, p = (½, ½), t = `sun_scale.zw`
    // plus its mat-anim delta, R and s from the affine row `[cos − 1, sin, sx − 1, sy − 1]`.
    var vo: VertexOutput;
    vo.position = in.position;
    vo.world_position = in.world_position;
    vo.world_normal = in.world_normal;
#ifdef VERTEX_UVS_A
    // An env-mapped coordinate takes no UV animation: the reference excludes an env stage from
    // `textureTransform`.
    if ((u32(m.clutter_fade.z) & 4096u) != 0u) {
        vo.uv = in.uv;
    } else {
        let uv_t = in.uv + m.sun_scale.zw + wow_light.matanim[u32(m.anim_slots.x)].xy;
        let affine = wow_light.matanim[u32(m.anim_slots.z)];
        let d = (uv_t - vec2<f32>(0.5, 0.5)) * vec2<f32>(1.0 + affine.z, 1.0 + affine.w);
        let c = 1.0 + affine.x;
        vo.uv = vec2<f32>(0.5 + d.x * c - d.y * affine.y, 0.5 + d.x * affine.y + d.y * c);
    }
#endif
#ifdef VERTEX_UVS_B
    vo.uv_b = in.uv_b;
#endif
#ifdef VERTEX_COLORS
    vo.color = in.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    vo.instance_index = in.instance_index;
#endif
    var pbr_input = pbr_input_from_standard_material(vo, is_front);
    var base_color = pbr_input.material.base_color;
    // WMO MOCV alpha is lighting data (the TRANS lerp, the INT glow mask), and the reference's WMO
    // output alpha is tex.a alone. Bevy folds the vertex alpha into base_color, so re-sample the
    // texel alpha: dividing the fold back out loses everything where MOCV.a ≈ 0.
#ifdef VERTEX_COLORS
    if (m.model_flags.x > 0.5) {
#ifdef VERTEX_UVS_A
        base_color.a = textureSampleBias(
            pbr_bindings::base_color_texture,
            pbr_bindings::base_color_sampler,
            vo.uv,
            // The colour's LOD: `pbr_input_from_standard_material` applies `view.mip_bias`, and
            // coverage from another mip erodes out of step with the art.
            view.mip_bias,
        ).a;
#else
        base_color.a = 1.0;
#endif
    }
#endif
    // The detail-doodad LOD bias (`0x6813f4`): on the atlases whose alpha pyramid is binary below
    // mip 0 it keeps every fragment below full alpha, so the fade erodes per pixel, not per leaf.
    // wgpu has no sampler LOD bias and the sampler is shared with unbiased batches, so it goes on
    // this sample. Clutter's tint is white and its vertex colour the MCSH grey, so `* vo.color`
    // reproduces Bevy's fold.
    if (m.clutter_fade.w > 0.5) {
#ifdef VERTEX_UVS_A
        var biased = textureSampleBias(
            pbr_bindings::base_color_texture,
            pbr_bindings::base_color_sampler,
            vo.uv,
            view.mip_bias + DETAIL_DOODAD_LOD_BIAS,
        );
#ifdef VERTEX_COLORS
        biased = biased * vo.color;
#endif
        base_color = biased;
#endif
    }
    // The clutter distance fade, the reference's stage-1 ramp: u = (z_eye − 52.5)/17.5 by a
    // camera-space texgen (`0x6b2b80`), so the boundary is a view plane, not a sphere; the ramp is
    // a bilinear read of the 64-texel table `4·(63 − col)` (`0x6b2320`). It multiplies the alpha
    // the cutout tests (ALPHAREF `detailDoodadAlpha` = 128).
    if (m.clutter_fade.w > 0.5) {
        let z_eye = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        let u = (z_eye - m.clutter_fade.x) / max(m.clutter_fade.y - m.clutter_fade.x, 0.001);
        let ramp = clamp((254.0 - 256.0 * u) / 255.0, 0.0, 252.0 / 255.0);
        base_color.a = base_color.a * ramp;
    }

    // Stage 1 (`0x70b740` binds a stage per texture of the unit): its alpha joins the stage-0
    // alpha the cutout reads (the blend's is `faded_alpha`, below); its colour combines after the light (below).
    let stage1_on = m.stage1.x > 0.5;
    let stage1_preset = u32(m.stage1.y);
    let tex0_alpha = base_color.a;
    var tex1 = vec4<f32>(1.0);
    if (stage1_on) {
        // tortoise: a stage on a UV set (`stage1.w` = 1 + its scroll row) moves by its own
        // texture transform; row 0 is the identity.
        var st1_uv = in.env_uv;
        if (m.stage1.w > 0.5) {
            st1_uv = st1_uv + wow_light.matanim[u32(m.stage1.w) - 1u].xy;
        }
        tex1 = textureSampleBias(stage1_texture, stage1_sampler, st1_uv, view.mip_bias);
        base_color.a = stage_alpha(stage1_preset, base_color.a, tex1.a);
    }

    // The MeshTag (mesh_tag.rs): bit 31 = highlight, bit 30 = interior fog; payload bits 0-5 = the
    // fade alpha (a zero payload is untagged, opaque), 19-29 = the rig slot, 6-13 = the ground
    // shade (0 lit, 255 MCSH-shadowed) or, on an interior-prop material, 6-18 = the SH-probe slot.
    let interior_prop = m.model_flags.z > 0.5 && m.model_flags.x < 0.5;
    let raw_tag = mesh_functions::get_tag(in.instance_index);
    let highlighted = (raw_tag & 0x80000000u) != 0u;
    let interior_fogged = (raw_tag & 0x40000000u) != 0u;
    let fade_tag = raw_tag & 0x3fffffffu;
    let alpha6 = f32(fade_tag & 0x3fu) / 63.0;
    var obj_fade = select(alpha6, 1.0, fade_tag == 0u);
#ifdef WOW_MERGED_FADE
    // A merged blob's per-vertex fade composes where the tag fade does, so it feathers the same.
    obj_fade = obj_fade * in.merged_fade;
#endif
    // The body tint (an aura colouring the whole model), by rig slot, 0 = identity: the material
    // ambient+diffuse colour (gx SetState(1), GL_COLOR_MATERIAL), so it multiplies the light sum
    // inside the clamp, never the emission.
    let tint_word = wow_light.rig_tint[(fade_tag >> 19u) & 0x7ffu];
    let inst_tint = select(
        vec3<f32>(
            f32((tint_word >> 16u) & 0xffu),
            f32((tint_word >> 8u) & 0xffu),
            f32(tint_word & 0xffu),
        ) * (1.0 / 255.0),
        vec3<f32>(1.0),
        tint_word == 0u,
    );
    // The blend twin re-applies its source's cutout: the reference scales ALPHAREF with the fade,
    // so the cutoff stays tex.a < 224/255 on the unfaded alpha. Only for a source batch that
    // alpha-tests (bit 10): ALPHAREF keys on the stored blend mode.
    if ((u32(m.clutter_fade.z) & 1024u) != 0u && base_color.a < VANILLA_ALPHA_KEY) {
        discard;
    }
    // fork: the alpha-mask cutout, spelled out in the entry point as well as in bevy's
    // `alpha_discard` below. Past a certain fragment size (the modern lighting stack) Chrome/Metal
    // stopped honouring the discard inside that helper, and alpha-tested WMO canopies and tree cards
    // drew as solid polygons. Bits: ALPHA_MODE_RESERVED_BITS (7 << 29), MASK (1 << 29).
    if ((pbr_input.material.flags & (7u << 29u)) == (1u << 29u)
        && base_color.a < pbr_input.material.alpha_cutoff) {
        discard;
    }
    // fork: the depth-prime twin stops here, and the colour lane below is not COMPILED into it at
    // all (a preprocessor cut, not an early return, so no dead tail reaches naga's MSL writer). On
    // Chrome/Metal a colour-masked twin carrying the whole lighting stack lost its cutout: it wrote
    // depth over the entire card, and a fading tree's holes blocked the sky behind them (black).
    // Declared ahead of the cut: both lanes write it, and the twin's lane has nothing below.
    var out: WowFragOut;
#ifdef WOW_ZFILL
    out.color = vec4<f32>(0.0);
#else
    // DEPTH-PRIME TWIN (the zfill pipeline variant — terrain.rs `specialize`, the reference's
    // M2UseZFill clone command, wow-re m2-blend-promotion-zfill.md §4): colour writes are masked
    // off at the pipeline, so on that variant only the discards above (farclip wall, the
    // bit-10 twin cutout) shape what depth gets written; the lighting/fog below is computed and
    // thrown away. (A shader-def early return here would be the natural spelling, but naga's MSL
    // backend miscompiles the dead tail — "redefinition of '_tmp'" — so the twin pays the colour
    // math instead. Episodes are transient; the cost is bounded.)
    // The blended alpha is the stage chain on the instance alpha, the diffuse (`0x70c73a`, c28.w):
    // stage 0 combines it with the texel by its preset's alpha op, so an opaque batch's preset 0
    // (`0x811f8c`) blends by the fade alone and the texel's alpha never reaches the frame. Stage 1
    // follows on that. `m.stage1.z` is the stage-0 preset + 1, 0 for a batch with none.
    var faded_alpha = base_color.a * obj_fade;
    if (m.stage1.z > 0.5) {
        faded_alpha = stage_alpha(u32(m.stage1.z) - 1u, obj_fade, tex0_alpha);
        if (stage1_on) {
            faded_alpha = stage_alpha(stage1_preset, faded_alpha, tex1.a);
        }
    }
    let base = alpha_discard(pbr_input.material, base_color);

    // --- Lighting --------------------------------------------------------------------------------
    let is_clutter = m.clutter_fade.w > 0.5;
    let L = -normalize(wow_light.light_sun.xyz);
    // `wow_normalize`: an authored zero normal reaches here on the unskinned lane.
    let n_m2 = wow_normalize(pbr_input.world_normal);
    // Bevy negates `world_normal` on back faces of double-sided materials (foliage, every WMO
    // face); the reference never enables GL_LIGHT_MODEL_TWO_SIDE (`0x59ce30`), so undo it.
    let n_lit = select(-n_m2, n_m2, is_front);
    let ndotl = max(dot(n_lit, L), 0.0);
    let lit_nl = clamp(wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl, vec3<f32>(0.0), vec3<f32>(1.0));
    // The order-2 SH basis, shared by the exterior lobe and the interior probes.
    let quad = vec4<f32>(n_lit.x * n_lit.y, n_lit.y * n_lit.z, n_lit.z * n_lit.z, n_lit.x * n_lit.z);
    let n1 = vec4<f32>(n_lit, 1.0);
    let x2y2 = n_lit.x * n_lit.x - n_lit.y * n_lit.y;
    // Exterior doodads and entities: the header's `Model2.bls` lobe at the per-instance intensity
    // I, `[node+0xa4]` (animator targets 2.5 lit and 0.5 shadowed, `0x69e4ad`/`0x69e280`; 1.0
    // indoors, `0x69e36b`). `sun_scale.x` picks the family:
    //   ≥ 0.85: an entity M2, 2.5 mixed toward 0.5 by the tag's shade byte (ramped like `0x69e770`)
    //   0.5..0.85: a doodad (ADT MDDF or WMO MODD, both `CMapDoodadDef`): 1.0, never 2.5
    //   < 0.5: a doodad on MCSH-shadowed ground: 0.5
    // Deviation: `min(I, 1)`. The reference bakes I into the SH unclamped (`0x71c4e0`), so a lit
    // entity commits ×2.5 there and ×1.0 here; lifting the cap takes sun-facing surfaces past 1.0.
    // `entity_shade::LIT_T` aims the lit ramp at 1.0 because of it: lift the cap and set `LIT_T`
    // back to 0.0 together.
    let inst_shade = select(f32((fade_tag >> 6u) & 0xffu) / 255.0, 0.0, interior_prop);
    let mat_shade = select(0.0, 1.0, m.sun_scale.x < 0.5);
    // fork: modern lighting (FORK.md, P3) — the baked MCSH doodad shade retires; the modern cascades darken instead.
    let shade_t = select(max(mat_shade, inst_shade), 0.0, wow_light.point_count.y > 0.5);
    let mid_band = m.sun_scale.x >= 0.5 && m.sun_scale.x < 0.85;
    let intensity = min(select(mix(2.5, 0.5, shade_t), 1.0, mid_band), 1.0);
    // One `intensity` multiply covers every sun band (never I²); the c10 `.w` ambient does not
    // scale. `pack_model_core_rows` packs the same closed form.
    let sun_dc = wow_light.grade.yzw * intensity;
    let sun_lobe = vec3<f32>(
        wow_light.sh_c10_r.w + sun_dc.x
            + intensity
                * (dot(wow_light.sh_c10_r.xyz, n_lit) + dot(wow_light.sh_c13_r, quad)
                    + wow_light.sh_c16.x * x2y2),
        wow_light.sh_c10_g.w + sun_dc.y
            + intensity
                * (dot(wow_light.sh_c10_g.xyz, n_lit) + dot(wow_light.sh_c13_g, quad)
                    + wow_light.sh_c16.y * x2y2),
        wow_light.sh_c10_b.w + sun_dc.z
            + intensity
                * (dot(wow_light.sh_c10_b.xyz, n_lit) + dot(wow_light.sh_c13_b, quad)
                    + wow_light.sh_c16.z * x2y2),
    );
    // Clamp the sum, never a term: the lobe's own ringing dips to −0.037·C near μ ≈ −0.53, and a
    // per-term clamp would erase it.
    let lit_doodad = clamp(sun_lobe, vec3<f32>(0.0), vec3<f32>(1.0));
    // WMO and clutter take the FFP N·L light with no terrain shade; the lobe needs the directional
    // light on (`light_sun.w`).
    let is_wmo = m.model_flags.x > 0.5;
    let is_interior = m.model_flags.z > 0.5;
    // Unlit fullbright (model_flags.w): M2 UNLIT (0x01) glass/glow cards, or WMO UNLIT on an
    // exterior-group batch (`tex × white` — the inn's always-lit outside panes). Wins over the lit
    // path with no emission terms (lighting is off, so GL_EMISSION is dead there) — only the
    // highlight is added (below). Hoisted: the lit sum is dropped whole for it
    // (`select(lit_rgb, unlit_rgb, is_emissive)`), so the shadow, probe-field, sky and point-light
    // lookups that only feed that sum are skipped for it. A light shaft is stacked unlit additive
    // planes yards across, and standing in one paid every lookup per layer per pixel.
    let is_emissive = m.model_flags.w > 0.5;
    // fork: modern lighting (FORK.md, P6) — the AUTHORED-RIG lane's flag, hoisted
    // from its own lane below (documented there) because the modern interior light has to exclude a
    // booth: a glue scene's light is entirely its authored rig and there is no room around it.
    let is_rig = m.sun_scale.x >= 1.5;
    let use_doodad_shade = (wow_light.light_sun.w > 0.5) && !is_clutter && !is_wmo;
    let lit_exterior = select(lit_nl, lit_doodad, use_doodad_shade);
    // WMO interior groups (`groupFlags & 0x48 == 0`) by batch class (`tint.w`): INT (1) is unlit
    // `tex × MOCV`; TRANS (2) is the reference's two passes, lit × MOCV.a + unlit × (1 − MOCV.a),
    // as one lerp; EXT (0) is `lit_nl`. A WINDOW batch (MOMT 0x20) on the interior drawer lights
    // with ambient = diffuse = the Direct/Ambient midpoint, ambient +16/255 (`0x6d37e0`).
    var trans_a = 1.0;
#ifdef VERTEX_COLORS
    trans_a = in.color.a;
#endif
    let window_mid = 0.5 * (wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb);
    let lit_window = clamp(
        window_mid + vec3<f32>(16.0 / 255.0) + window_mid * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let lit_int_base = select(lit_nl, lit_window, m.sidn.w > 0.5);
    // fork: modern lighting (FORK.md, P6) — interiors light like everything else
    // in a modern frame: the probe field's 1.5 yd interior sub-grid for the indirect, the binned
    // local lights for the fixtures, and the authored MOCV kept only as the emissive mask it is.
    // The gate is the probe field itself (`gi.params.x`), which is exactly the tier split the plan
    // asked for: Low traces no probes, so Low keeps the bake and its `prop_probes`.
#ifdef WOW_MODERN
    // The water mirror takes none of the modern lanes (`sunshadow::mirror_view`): no cascades,
    // cloud shadow, probe field, sky map, light grid or foliage transmission — the plain ambient and
    // N·L, and an interior keeps its bake. A reflection is bent and blurred by the waves; the second
    // render of the whole world is what the frame pays for, so it pays for the plain one.
    let modern_lite = sunshadow::mirror_view(modern_block.cascades, view.world_position);
    let modern_interior = wow_light.point_count.y > 0.5 && modern_block.gi.params.x > 0.5 && !modern_lite;
    var lit_modern_room = vec3<f32>(0.0);
    if (modern_interior && is_interior && !is_rig && !is_emissive) {
        let rn = wow_normalize(in.world_normal);
        let room_gi = modern_gi_ambient(in.world_position.xyz, rn, 1.0, modern_lite);
        // The building's own `MOHD.ambColor` is the floor: where the field has nothing to say yet
        // (a probe slice not traced this frame, a cupboard no probe sits in) the room still reads as
        // the room and not as a hole.
        let room = max(modern_block.gi.interior.rgb, wow_light.light_ambient.rgb * MODERN_ROOM_FLOOR);
        lit_modern_room = mix(room, room_gi.rgb, room_gi.w);
    }
#else
    let modern_interior = false;
    let lit_modern_room = vec3<f32>(0.0);
#endif
    // The classic interior light by batch class, before MOCV: what a modern room is floored on.
    var bake_interior = vec3<f32>(1.0);
    if (m.tint.w > 1.5) {
        bake_interior = mix(vec3<f32>(1.0), lit_int_base, trans_a);
    } else if (m.tint.w < 0.5) {
        bake_interior = lit_int_base;
    }
    var lit_wmo_interior = vec3<f32>(1.0);
    if (modern_interior) {
        // Every batch class takes the same light now: INT is no longer "unlit, the bake IS the
        // light", TRANS has no bake left to lerp toward, and the WINDOW lane's brighter GL_LIGHT0
        // swap is replaced by the pane's own SIDN emission below.
        lit_wmo_interior = lit_modern_room;
    } else if (m.tint.w > 1.5) {
        lit_wmo_interior = mix(vec3<f32>(1.0), lit_int_base, trans_a);
    } else if (m.tint.w < 0.5) {
        lit_wmo_interior = lit_int_base;
    }
    // Interior M2 props (WMO MODD doodads): the reference commits the prop's MODD colour through a
    // fixed-axis diffuse lobe plus its group's MOLR lights as an order-2 SH probe, folded at spawn
    // by `lighting::prop_probe_coeffs` and evaluated here per fragment (the reference: per vertex).
    // Its soft wrap (≈ 0.088·C side-on) is the reference's response, not a max(N·L, 0).
#ifdef WOW_MERGED_SLOT
    // A merged blob bakes the slot per vertex; its tag carries only fog and alpha.
    let probe = 7u * in.merged_slot;
#else
    let probe = 7u * ((fade_tag >> 6u) & 0x1fffu);
#endif
    let lit_m2_interior = clamp(
        vec3<f32>(
            dot(wow_light.prop_probes[probe + 0u], n1)
                + dot(wow_light.prop_probes[probe + 3u], quad)
                + wow_light.prop_probes[probe + 6u].x * x2y2,
            dot(wow_light.prop_probes[probe + 1u], n1)
                + dot(wow_light.prop_probes[probe + 4u], quad)
                + wow_light.prop_probes[probe + 6u].y * x2y2,
            dot(wow_light.prop_probes[probe + 2u], n1)
                + dot(wow_light.prop_probes[probe + 5u], quad)
                + wow_light.prop_probes[probe + 6u].z * x2y2,
        ),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    // fork: modern lighting (FORK.md, P6) — an interior MODD prop reads the room
    // the same way its walls do; its folded `prop_probes` slot survives on Low alone (and on the
    // booth lane below, whose light is entirely its authored rig).
    // It keeps its classic light as a floor, as the walls do (`MODERN_BAKE_FLOOR`), or a prop goes
    // dark against a lit wall.
    let lit_m2_room = select(
        lit_m2_interior,
        max(lit_modern_room, lit_m2_interior * MODERN_BAKE_FLOOR),
        modern_interior,
    );
    let lit_interior = select(lit_m2_room, lit_wmo_interior, is_wmo);
    // AUTHORED-RIG lane (`ShadeSel::Rig`, sun_scale.x = 2.0 — the glue create booth, decision
    // 0429): the lit value is the probe-slot SH eval — the scene M2's ambient + directional
    // lights, folded into slot 0 of the material's OWN buffer (booth instances carry tag 0, so
    // `lit_m2_interior` above already evaluated exactly that probe). No sun, no intensity family,
    // no day/night — a glue scene's light is entirely its authored rig. Rig materials are neither
    // WMO nor interior, so the per-vertex point term (the rig's authored point lights) flows in
    // through `point_diffuse` below like any exterior entity.
    let lit = select(select(lit_exterior, lit_interior, is_interior), lit_m2_interior, is_rig);
    var player_shadow = 1.0;
    if (lights.n_directional_lights > 0u && !is_emissive) {
        let view_z = (view.view_from_world * in.world_position).z;
        for (var light_id = 0u; light_id < lights.n_directional_lights; light_id = light_id + 1u) {
            if ((lights.directional_lights[light_id].flags & 1u) != 0u) {
#ifdef WOW_RIG_SKIN
                // A skinned UNIT samples the map ONCE at its rig origin (between the feet),
                // nudged 2.5 units TOWARD the sun, and the whole unit dims uniformly — the
                // reference's own per-unit response. Per-fragment sampling is wrong here: the
                // caster proxy holds a CPU-skinned copy of this very body, so a body fragment
                // reads its own limbs/weapon as occluders (self-shadow speckle no bias fixes on
                // animated silhouettes). Points along the sun ray share a shadow-map texel, so
                // the nudge only excludes occluders within ~2.5 units of the feet ALONG the ray
                // (a fence hugging the unit) — a building or canopy still lands. Gated on the
                // directional-enable flag so an interior-lit unit never takes an exterior dim.
                if (wow_light.light_sun.w > 0.5) {
                    let sun_ray = normalize(wow_light.light_sun.xyz);
                    let anchor = wow_light.rig_origin[(fade_tag >> 19u) & 0x7ffu].xyz
                        - 2.5 * sun_ray;
                    player_shadow = shadows::fetch_directional_shadow(
                        light_id,
                        vec4<f32>(anchor, 1.0),
                        vec3<f32>(0.0, 1.0, 0.0),
                        view_z,
                        // Bevy 0.19: the pixel seeds the filter's per-fragment rotation.
                        in.position.xy,
                    );
                }
#else
                player_shadow = shadows::fetch_directional_shadow(
                    light_id,
                    in.world_position,
                    wow_normalize(in.world_normal),
                    view_z,
                    // Bevy 0.19: the pixel seeds the filter's per-fragment rotation.
                    in.position.xy,
                );
#endif
                break;
            }
        }
    }
    // The realtime map blocks only the directional sun. Preserve the authored ambient/probe
    // contribution instead of multiplying the whole lighting result; the latter makes interiors,
    // point-lit props, and shadow-side characters globally too dark.
    //
    // `is_rig` (the glue create-booth lane) is excluded DELIBERATELY: a booth scene's light is
    // entirely its authored rig — there is no world sun to block. World units are `ShadeSel::Lit`
    // and receive through the WOW_RIG_SKIN ground sample above. The `player_shadow < 0.999`
    // arm keeps `worldShadows 0` (and a fully sunlit fragment) byte-identical: `ambient +
    // (lit − ambient) × 1.0` is not guaranteed to round back to `lit`.
    // fork: modern lighting (FORK.md, P2) — per-fragment modern cascades for
    // every lane (units included); a shadowed fragment keeps only its ambient.
#ifdef WOW_MODERN
    let modern_frame = wow_light.point_count.y > 0.5;
    if (modern_frame && !modern_lite && !is_emissive) {
        let mview_z = (view.view_from_world * in.world_position).z;
        var shadow_p = in.world_position.xyz;
#ifdef WOW_RIG_SKIN
        // A skinned body is its own caster in the cascades (the entity lane draws the posed rig), so a
        // body fragment looked up where it stands reads its own arms, head and cloak as occluders and
        // stays shaded wherever it walks. Look it up from a point a little way toward the key light
        // instead: past its own limbs, but still under any canopy, roof or cliff — so a patch of sun
        // lights the part of the body that stands in it, per fragment.
        shadow_p = shadow_p - normalize(wow_light.light_sun.xyz) * RIG_SHADOW_LIFT;
#endif
        player_shadow = sunshadow::sun_visibility(modern_shadow_map, modern_shadow_samp, modern_block.cascades, shadow_p, wow_normalize(in.world_normal), -mview_z, modern_lite);
        // fork: modern lighting (FORK.md, P4) — cloud shadows.
        let cloud_ct = sunshadow::cloud_taps(modern_block.fog.cloud, in.world_position.xyz);
        player_shadow = player_shadow * sunshadow::cloud_mix(cloud_ct, modern_block.cloud[cloud_ct.idx.x], modern_block.cloud[cloud_ct.idx.y], modern_block.cloud[cloud_ct.idx.z], modern_block.cloud[cloud_ct.idx.w]);
    }
#else
    let modern_frame = false;
#endif
    let shadow_term = select(mix(SHADOW_SUN_FLOOR, 1.0, player_shadow), player_shadow, modern_frame);
    let lit_with_shadow_base = select(
        lit,
        clamp(
            wow_light.light_ambient.rgb + (lit - wow_light.light_ambient.rgb) * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ),
        !is_interior && !is_rig && player_shadow < 0.999,
    );
    // fork: modern lighting (FORK.md, P3) — foliage (a two-sided, alpha-masked material) lets
    // the sun through: wrap diffuse past the terminator plus back-lit transmission toward the eye,
    // both under the cascades' visibility. Bits: STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT (1<<4),
    // ALPHA_MODE_RESERVED_BITS (7<<29) == ALPHA_MODE_MASK (1<<29).
#ifdef WOW_MODERN
    let foliage_flags = pbr_input.material.flags;
    let is_foliage = wow_light.point_count.y > 0.5 && !is_interior && !is_rig && !modern_lite
        && (foliage_flags & (1u << 4u)) != 0u
        && (foliage_flags & (7u << 29u)) == (1u << 29u);
    var foliage_light = vec3<f32>(0.0);
    if (is_foliage) {
        let fn_ = wow_normalize(in.world_normal);
        let to_sun = -normalize(wow_light.light_sun.xyz);
        let to_eye = normalize(view.world_position.xyz - in.world_position.xyz);
        let nl = dot(fn_, to_sun);
        let wrap = max((nl + 0.5) / 1.5, 0.0) - max(nl, 0.0);
        let back = pow(max(dot(to_eye, -to_sun), 0.0), 6.0);
        foliage_light = wow_light.light_diffuse.rgb * player_shadow * (0.5 * max(wrap, 0.0) + 0.6 * back);
    }
#else
    let foliage_light = vec3<f32>(0.0);
#endif
    // fork: modern lighting (FORK.md, P3) — modern exterior: ambient under the sky-occlusion map.
#ifdef WOW_MODERN
    var lit_with_shadow_sky = lit_with_shadow_base;
    if (wow_light.point_count.y > 0.5 && !is_interior && !is_rig && !modern_lite && !is_emissive) {
        let amb = wow_light.light_ambient.rgb;
        // fork: modern lighting (FORK.md, P5) — the probe field over the sky map.
        let gi = modern_gi_ambient(in.world_position.xyz, wow_normalize(in.world_normal), 0.0, modern_lite);
        // The sky map only where the field does not answer whole (`sunshadow::probe_or_sky`).
        var sky = 1.0;
        if (gi.w < 1.0) {
            sky = sunshadow::sky_ambient(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz, modern_lite);
        }
        let amb_gi = sunshadow::shade_floor(sunshadow::probe_or_sky(amb, sky, gi), amb);
        lit_with_shadow_sky = clamp(amb_gi + max(lit - amb, vec3<f32>(0.0)) * player_shadow, vec3<f32>(0.0), vec3<f32>(1.0));
    }
#else
    let lit_with_shadow_sky = lit_with_shadow_base;
#endif
    let lit_with_shadow = lit_with_shadow_sky + foliage_light;
    // Gamma-space albedo — the lane (0161): the buffer holds bytes, lighting math runs on the
    // authored values. (The old fog_params.z linear-space A/B is dead — settled by the lane.)
    // `m.tint` is the animated M2Color RGB (identity 1 for static batches) — the same per-batch
    // tint the vertex colours carry for constant tracks, so it folds at the same point.
    let anim_tint = m.tint.rgb + wow_light.matanim[u32(m.anim_slots.y)].xyz;
    let albedo = base.rgb * anim_tint;
    // SIDN night glow: the emissive × the night fraction (ramping 20:30→21:30, 06:00→07:00), a
    // GL_EMISSION term inside the clamped lit sum, on lit lanes only.
    var sidn_w = 1.0;
    if (is_interior && is_wmo) {
        if (m.tint.w > 1.5) {
            sidn_w = trans_a; // TRANS: weighted by its lit pass
        } else if (m.tint.w > 0.5) {
            sidn_w = 0.0; // INT: unlit, so no emission
        }
    }
    var sidn_e = m.sidn.rgb * (wow_light.grade.x * sidn_w);
    // fork: modern lighting (FORK.md, P6) — a modern window is an emitter: the
    // authored colour at an HDR gain, on every batch class (the FFP's dead emissive write on the
    // unlit INT lane was a consequence of lighting being off there, and lighting is on now), and
    // with a daylight floor so a pane still glows seen from inside.
#ifdef WOW_MODERN
    if (modern_interior && is_wmo && is_interior) {
        sidn_e = m.sidn.rgb * (MODERN_SIDN_GAIN * mix(MODERN_SIDN_DAY, 1.0, wow_light.grade.x));
    }
#endif
    // WoW dynamic point lights (decisions 0016/0273/0278, selection 0285) — exterior doodads,
    // entities, clutter, and terrain receive their unit's committed lights: the ≤3 NEAREST to the
    // receiving unit's own position, never the whole scene (`point_light_sum`). WMO surfaces take
    // ZERO (observed on every WMO surface batch in the abbey capture) and interior props fold their
    // group-MOLR lobes into the SH probe instead — both zeroed in the VERTEX stage. The term
    // arrives GOURAUD-INTERPOLATED — per-vertex like the reference FFP, whose tessellation-scale
    // smoothing is the authored look. Diffuse-only (committed ambient/specular are zero).
    // fork: modern lighting (FORK.md, P3) — per-pixel point lights in a modern frame.
#ifdef WOW_MODERN
    var point_diffuse = in.point_lit;
    if (wow_light.point_count.y > 0.5 && !is_emissive) {
        point_diffuse = point_light_eval(in.point_sel, in.world_position.xyz, wow_normalize(in.world_normal));
    }
    // fork: modern lighting (FORK.md, P6) — the binned grid replaces the
    // reference's three-nearest selection where it is live, and it reaches the lanes the reference
    // gives no point light at all: a WMO surface and an interior prop are lit by the room's fixtures
    // in a modern frame, because the bake that used to stand in for them is gone. An unlit batch
    // skips the walk and its cube lookups, as it skips every lookup that only feeds the lit sum.
    if (locallight::grid_live(modern_block.lights) && !is_rig && !modern_lite && !is_emissive) {
        point_diffuse = modern_local_lights(
            in.world_position.xyz,
            wow_normalize(in.world_normal),
            is_interior,
            modern_lite,
        );
    }

#else
    let point_diffuse = in.point_lit;
#endif
    // The hover/target highlight (tag bit 31): the scene's committed ambient
    // (`0x614576`-`0x6145bd`), added to the batch colour inside the final clamp, lit or unlit
    // (`c29`). Sampled live, where the reference holds the value sampled when the highlight
    // began: a unit's tag has no slot to hold a colour, and the ambient moves slowly.
    let highlight = select(vec3<f32>(0.0), wow_light.light_ambient.rgb, highlighted);
    // FFP combine (byte + trace verified, decision 0273): the LIGHT SUM — matte base + point lights +
    // the highlight emission — saturates per fragment FIRST, and the texture modulates the clamped
    // result, so a surface never exceeds its own fully-lit texture (a fixture light at zero distance
    // drives the prop to tex×1, not past it — the old `clamp(albedo·sum)` order blew emitter props to
    // saturated gold). WMO surfaces fold their vertex colour INSIDE the clamp (GL_COLOR_MATERIAL:
    // MOCV is the material ambient+diffuse — `tex × clamp(MOCV·sum + emission)`), so a strong fixture
    // light overdrives a dim bake toward the full texture exactly like the reference. Bevy pre-folds
    // ATTRIBUTE_COLOR into `base`, so un-fold it with a guarded divide (a dim channel's product is ~0
    // either way). At zero point contribution every factor is ≤1 and both forms collapse to the old
    // product — the approved interior/exterior base looks are preserved bit-for-bit.
    var lit_rgb: vec3<f32>;
#ifdef VERTEX_COLORS
    if (is_wmo) {
        let vc = in.color.rgb;
        // fork: modern lighting (FORK.md, P3/P6) — a modern WMO surface drops its
        // baked MOCV light: an exterior one since P3 (the cascades and the sky map shade it), an
        // interior one since P6 (the probe field and the binned fixtures light it). What survives
        // indoors is MOCV's ALPHA, the artists' emissive mask, promoted from a multiplier on the
        // bake to an emission term of its own — and left UNCLAMPED, so a hearth carries real energy
        // into the bloom instead of saturating at white like the byte framebuffer it was authored
        // for.
        let drop_mocv = wow_light.point_count.y > 0.5 && (!is_interior || modern_interior);
        // MOCV multiplies the lit terms (GL_COLOR_MATERIAL) but NOT the emission terms — SIDN and
        // the highlight add alongside, exactly the FFP's material-emission placement. This sum is
        // left EXACTLY as upstream wrote it, term for term: float addition does not associate, and a
        // regrouping here would move the classic frame by an ulp and break its identity gate. The
        // modern interior is a separate statement below for the same reason.
        var primary = clamp(
            select(vc, vec3<f32>(1.0), drop_mocv) * (lit_with_shadow + point_diffuse) + sidn_e + highlight,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        if (modern_interior && is_interior) {
            // Emission OUTSIDE the clamp: a hearth or a pane carries real energy into the bloom
            // instead of saturating at white like the byte framebuffer it was authored for. The
            // MOCV-alpha term rides the INT class alone, the one class whose alpha is an emissive
            // mask rather than the lit-pass weight.
            var emissive = sidn_e + highlight;
            if (m.tint.w > 0.5 && m.tint.w < 1.5) {
                emissive = emissive + vec3<f32>(MODERN_MOCV_EMISSIVE * trans_a);
            }
            let lit_sum = max(
                select(vc, vec3<f32>(1.0), drop_mocv) * (lit_with_shadow + point_diffuse),
                vc * bake_interior * MODERN_BAKE_FLOOR,
            );
            primary = clamp(lit_sum, vec3<f32>(0.0), vec3<f32>(1.0)) + emissive;
        }
        let tex_rgb = base.rgb / max(vc, vec3<f32>(1.0 / 255.0));
        lit_rgb = tex_rgb * m.tint.rgb * primary;
        if (is_interior && m.tint.w > 0.5 && m.tint.w < 1.5 && !modern_interior) {
            // INT: the MOCV-ALPHA SELF-ILLUMINATION term. The reference's interior pixel shader is
            // `tex·MOCV.rgb·(1 + 4·MOCV.a)` with only the framebuffer's final [0,1] clamp — read
            // off the client's own D3D pixel shader in the Goldshire-inn trace (literal 4.0 in the
            // source, no lights referenced), so the glow multiplies the FULL product and may
            // overdrive it to white, never pre-clamped like the FFP light sum above. The alpha
            // channel is an authored emissive mask: the inn fireplace surround bakes α≈100 (×2.6),
            // hearths glow, and the FixColorVertexAlpha 255 at interior↔exterior portal seams
            // lifts doorways to full brightness. Near-zero everywhere unpainted (the abbey rooms),
            // where this collapses to the plain tex×MOCV it replaces.
            lit_rgb = clamp(
                tex_rgb * m.tint.rgb * vc * (1.0 + 4.0 * trans_a),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        }
    } else {
        // An M2: the body tint multiplies the light terms as MOCV does above. A WMO surface has
        // no tint slot, so the WMO branch leaves it out.
        let primary = clamp(
            inst_tint * (lit_with_shadow + point_diffuse) + sidn_e + highlight,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        lit_rgb = albedo * primary;
    }
#else
    // A WMO batch without MOCV lands here too; its tint slot is the identity slot 0.
    let primary = clamp(
        inst_tint * (lit_with_shadow + point_diffuse) + sidn_e + highlight,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    lit_rgb = albedo * primary;
#endif
    // The unlit path: an M2's unlit program outputs `c28 + c29` (`0x70c663`-`0x70c693` fold the
    // tint·M2Color term into `c29` beside the highlight), so the texel modulates
    // `clamp(C·tint + highlight)`, C the M2Color. A WMO keeps the plain modulate.
    var unlit_rgb = albedo * inst_tint;
    if (!is_wmo) {
#ifdef VERTEX_COLORS
        // The constant M2Color rides the vertex colour, which Bevy folded into `base`.
        let unlit_c = in.color.rgb * anim_tint;
        let unlit_tex = base.rgb / max(in.color.rgb, vec3<f32>(1.0 / 255.0));
#else
        let unlit_c = anim_tint;
        let unlit_tex = base.rgb;
#endif
        let unlit_sum = unlit_c * inst_tint + highlight;
        unlit_rgb = unlit_tex * clamp(unlit_sum, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    var rgb = select(lit_rgb, unlit_rgb, is_emissive);
    // An M2 Mod or Mod2x batch draws the bare texel: the reference zeroes its tint·M2Color term and
    // forces the primary colour to the blend identity (`0x70c507`/`0x70c5b8`), so neither the
    // animated M2Color nor the body tint reaches it. Its alpha still does, through the lerp below.
    let is_mod = (u32(m.clutter_fade.z) & 128u) != 0u;
    let is_mod2x = (u32(m.clutter_fade.z) & 256u) != 0u;
    if ((is_mod || is_mod2x) && !is_wmo) {
        rgb = base.rgb;
    }
    // Stage 1 combines with the stage-0 result, the lit texel, before the fog.
    if (stage1_on) {
        // The diffuse alpha is the batch and instance alpha, the shader's `obj_fade`.
        rgb = stage1_rgb(stage1_preset, rgb, tex1, obj_fade);
    }

    // fork: rain on surfaces (`benilla::wet_hook`). Nothing gets wet under a roof, a porch or a
    // bridge: the shelter grid's open share, read only while something is wet.
    var wet_open = 1.0;
    if (wow_light.env.wet_a.y > 0.0) {
        let wet_st = wet_hook::shelter_taps(in.world_position.xyz, wow_light.shelter_hdr,
            wow_light.shelter_cfg);
        if (wet_st.live > 0.0) {
            wet_open = 1.0 - wet_hook::shelter_amount(wet_st, wow_light.shelter[wet_st.idx.x],
                wow_light.shelter[wet_st.idx.y], wow_light.shelter[wet_st.idx.z],
                wow_light.shelter[wet_st.idx.w], in.world_position.y, wow_light.shelter_cfg);
        }
    }
    // Interior, booth-rig, unlit, Mod, Mod2x and additive batches stay dry; a dry frame returns
    // `rgb` untouched. A booth's own light blob carries a zero row, so it never gets wet.
    let wet_dry = is_interior || interior_fogged || is_rig || is_emissive || is_mod || is_mod2x
        || (u32(m.clutter_fade.z) & 4u) != 0u;
    let wet = wet_hook::wet_surface(rgb, n_lit, in.world_position.xyz,
        select(1.0, 0.0, wet_dry) * wet_open, wow_light.env);
    rgb = wet.albedo;
    if (wet.boost > 0.0) {
        rgb = min(rgb + wet_hook::wet_sheen(wet.boost, n_lit,
            normalize(view.world_position - in.world_position.xyz), L,
            wow_light.light_diffuse.rgb, wow_light.fog_color.rgb, player_shadow),
            vec3<f32>(1.0));
    }

    // Linear fog by planar eye depth, as in terrain.wgsl. Per-batch colour policy (`clutter_fade.z`
    // bits 4-6, the M2 state setter `0x70baf0`): 0 scene, 1 black (additive), 2 white (Mod),
    // 3 grey (Mod2x), 4 unfogged (render flag 0x02). Tag bit 30, not the static `model_flags.z`,
    // selects the interior triple: WMO content by the per-group `[0xca7f00]` gate on the pushes
    // `0x6b5190`/`0x6b62e0`, an entity M2 by its node's classification (`0x71c110`, `[node+0xc]`).
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if (interior_fogged) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    let fog_policy = (u32(m.clutter_fade.z) >> 4u) & 7u;
    // fork: modern lighting (FORK.md, P4) — a modern frame fogs from the froxel volume (below).
    if (fog_color.w > 0.5 && fog_policy != 4u && wow_light.point_count.y < 0.5) {
        let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        let denom = max(fog_span.y - fog_span.x, 0.001);
        let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
        var fog_rgb = fog_color.xyz;
        if (fog_policy == 1u) { fog_rgb = vec3<f32>(0.0); }
        else if (fog_policy == 2u) { fog_rgb = vec3<f32>(1.0); }
        else if (fog_policy == 3u) { fog_rgb = vec3<f32>(0.50196078); }
        rgb = mix(fog_rgb, rgb, factor);
    }


    // Output alpha is `faded_alpha`. Opaque intent (`clutter_fade.z` bit 3) pins it to 1: a no-op
    // under correct pipeline state, and a guard for macOS/Metal with an extra camera, where an
    // opaque draw can bind a blending pipeline and show the BLP's garbage alpha.
    let opaque_intent = (u32(m.clutter_fade.z) & 8u) != 0u;
    // Additive (`clutter_fade.z` bit 2, the bit `specialize` keys the (ONE, ONE) blend on): the
    // alpha weight folds into the colour here, in gamma, as the reference weights its source.
    let is_additive = (u32(m.clutter_fade.z) & 4u) != 0u;
    var out_rgb = rgb;
    if (is_additive) {
        out_rgb = out_rgb * faded_alpha;
    }
    // Mod (bit 7) and Mod2x (bit 8) read no source alpha, so the fade rides the colour as in the
    // reference: texenv preset 5, `mix(prev.rgb, tex.rgb, prev.a)`, with the primary colour forced
    // to the blend identity (white, or 0.5 grey for Mod2x) and prev.a the instance alpha. The fog
    // above commutes with this because its white and grey are that identity.
    if (is_mod || is_mod2x) {
        let identity = select(vec3<f32>(1.0), vec3<f32>(0.5), is_mod2x);
        out_rgb = mix(identity, out_rgb, obj_fade);
    }
    // GAMMA LANE (0161): raw gamma out — blending (alpha AND additive) happens in gamma like
    // the reference's byte framebuffer; the frame decodes once in the FFXGlow combine. (The old
    // `lin` A/B emitted linear for the sRGB encode — subsumed by the lane.)
    // fork: modern lighting (FORK.md) — the frame is linear, so decode AFTER the
    // premultiply (additive) and the identity mix (Mod/Mod2x). Mod stays exact in linear
    // ((ab)^γ = a^γ·b^γ); Mod2x's (Dst, Src) doubling becomes 2·a·b in linear, so the source is
    // scaled by 2.295 (= 0.5 / srgb_to_linear(0.5) / 2 · 2) to keep its 0.5-grey identity neutral.
#ifdef WOW_MODERN
    if (wow_light.point_count.y > 0.5) {
        var lin = colorspace::srgb_to_linear(rgb);
        // Additive: the alpha weight folds in BEFORE the decode, like the classic lane's gamma fold.
        // Weighting after it is the α^(1/2.2) inflation 0160 removed — a 30% halo edge reads ~58%,
        // so every soft glow card around a lamp swells into a bright disc.
        if (is_additive) {
            lin = colorspace::srgb_to_linear(rgb * faded_alpha);
        }
        if (is_mod || is_mod2x) {
            let identity = select(vec3<f32>(1.0), vec3<f32>(0.5), is_mod2x);
            lin = colorspace::srgb_to_linear(mix(identity, rgb, obj_fade));
            if (is_mod2x) {
                lin = lin * 2.295;
            }
        }
        // fork: modern lighting (FORK.md, P4) — an opaque-intent batch is fogged by the
        // composite; a blended one draws after it, so it fogs itself from the froxel volume under the
        // classic per-blend policy: scene fog for alpha, fade to black for additive, the blend identity
        // for Mod (1) and Mod2x (0.5, after its 2.295 source scale).
        if (!opaque_intent && fog_policy != 4u && wow_light.fog_color.w > 0.5) {
            let fog = froxelfog::fog_at(modern_fog_volume, modern_fog_samp, modern_block.fog, in.world_position.xyz);
            if (is_additive || fog_policy == 1u) {
                lin = lin * fog.a;
            } else if (is_mod || fog_policy == 2u) {
                lin = mix(vec3<f32>(1.0), lin, fog.a);
            } else if (is_mod2x || fog_policy == 3u) {
                lin = mix(vec3<f32>(0.5), lin, fog.a);
            } else {
                lin = lin * fog.a + fog.rgb;
            }
        }
        out_rgb = lin;
    }
    if (modern_frame && modern_block.cascades.params.w > 0.5) {
        let dbg = modern_block.cascades.params.w;
        if (dbg > 4.5) {
            // fork: modern lighting (P6) — mode 5: what the local-light grid put here.
            out_rgb = modern_local_lights(
                in.world_position.xyz,
                wow_normalize(in.world_normal),
                is_interior,
                false,
            );
        } else if (dbg > 2.5) {
            let gdbg = modern_gi_ambient(in.world_position.xyz, wow_normalize(in.world_normal), 0.0, false); // modern GI debug view
            out_rgb = select(gdbg.rgb, vec3<f32>(gdbg.w), dbg > 3.5);
        } else {
            out_rgb = vec3<f32>(select(player_shadow, sunshadow::sky_visibility(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz), dbg > 1.5)); // modern shadow debug view
        }
    }
#endif
    out.color = vec4<f32>(out_rgb, select(faded_alpha, 1.0, opaque_intent));
#endif
    return out;
}
