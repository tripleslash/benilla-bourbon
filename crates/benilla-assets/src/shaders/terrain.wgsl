// Terrain splat shader: a custom vertex and fragment stage on Bevy's StandardMaterial. Blends up to
// four tiled layers by a per-chunk alpha map, lit as the reference's fixed-function terrain:
//   diffuse  = clamp(ambient (row 1) + diffuse (row 0)·max(N·L, 0) + Σ point·att·N·L), evaluated
//     and clamped per vertex (GL T&L), Gouraud-interpolated, modulated 1× into the texture;
//   specular = clamp(row 9 · max(N·H, 0)^20) per vertex (local viewer), times the sheen mask (the
//     `_s` texture's per-texel alpha, blended like the colour), added after the modulate.
// Evaluated per pixel, or unmasked, the near-white row 9 would wash lit ground to cream.
//
// Bevy's `VertexOutput` has no slot for the interpolated specular, so the vertex transform and IO
// struct are our own and `main_pass_post_lighting_processing` is dropped; fog is in-shader.

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import benilla_modern::sunshadow
#import benilla_modern::probegi
#import benilla_modern::locallight
#import benilla::env_frame
// fork: rain on surfaces (the Everwood port).
#import benilla::wet_hook
#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{lights, view},
    shadows,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var layer_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var alpha_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var splat_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var shadow_array: texture_2d_array<f32>;
// fork: modern lighting (FORK.md, P2)
@group(#{MATERIAL_BIND_GROUP}) @binding(121) var modern_shadow_map: texture_depth_2d_array;
@group(#{MATERIAL_BIND_GROUP}) @binding(122) var modern_shadow_samp: sampler_comparison;
@group(#{MATERIAL_BIND_GROUP}) @binding(123) var<storage, read> modern_block: sunshadow::ModernBlock;

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

// **How dark a fully covered character shadow gets.** One number, carried by value in all three
// receivers — `wow_model.wgsl` beside this file, and `benilla-world`'s `static_gx.wgsl`. Move the
// three together; they are separate files in two crates only because WGSL has no shared header.
//
// Terrain's copy reads DEEPEST of the three at the same value, because here the factor multiplies
// the whole `tex · primary · (0.3·mcsh + 0.7)` modulate rather than the sun term alone (the
// neutral greyscale attenuation the combine below explains). That is where it should read hardest:
// the ground is what a character's shadow is actually cast onto.
//
// **0.70, raised from the 0.45 this first shipped at.** At 0.45 a character standing on Elwynn's
// morning grass wore a hole in the ground rather than a silhouette.
const SHADOW_SUN_FLOOR: f32 = 0.70;

// fork: rain on surfaces — how far a soaked face lifts the authored `_s` sheen (×1.2 at most), the
// wet sun highlight's share on the ground, and the ground's share of the wet reflectance
// (`wet_hook::wet_reflectance`, the screen-space pass's weight or the sky term's). Well under what
// the first cut shipped: at full strength soaked ground read as lacquer, and grass at 0.2/0.28/0.8
// still shone like a wet road.
const WET_SPEC_GAIN: f32 = 0.12;
const WET_SUN_GAIN: f32 = 0.16;
const WET_REFLECT_GAIN: f32 = 0.55;
// fork: yards per unit of the `UV1.x` fraction: `adt::WATER_DEPTH_RANGE / WATER_DEPTH_SPAN`.
const WATER_DEPTH_UNPACK: f32 = 40.0;

// Per-tile Vec4 uniforms, packed into one buffer (binding 106); the field order must match the Rust
// `TerrainExtension`. params.x = layer tiling factor; yzw unused.
struct TerrainParams {
    params: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var<uniform> t: TerrainParams;

// The shared global light (`lighting::global_light`), updated in place once a frame; this mirrors
// its row layout, which must match.
struct WowLight {
    light_ambient: vec4<f32>, // rgb = row 1 ambient; w = Mod2x scale (×1).
    light_diffuse: vec4<f32>, // rgb = row 0 sun diffuse; w = clamp-light flag (>0.5 ⇒ saturate).
    light_sun: vec4<f32>,     // xyz = world-space sun travel direction (to-light = −xyz); w unused.
    light_spec: vec4<f32>,    // rgb = row 9 specular color; w = shininess (20). rgb == 0 disables.
    fog_color: vec4<f32>,     // rgb = row 7 fog (raw, gamma 0..1); w = enable (>0.5 ⇒ blend).
    fog_params: vec4<f32>,    // x = fog_start yd; y = fog_end yd; z unused; w = farclip wall.
    _sh: array<vec4<f32>, 6>, // rows 6-11: model SH coefficients, unread by terrain.
    sh_c16: vec4<f32>,        // row 12: xyz = the models' c16 quad band; w free.
    _water: array<vec4<f32>, 4>, // rows 13-16: liquid swatches, unread by terrain.
    grade: vec4<f32>,         // row 17: reserved, layout only.
    _wmo_fog: array<vec4<f32>, 2>, // rows 18-19: interior fog, unread by terrain.
    // The dynamic point-light table (`global_light::build_light_data`): row 20 `.x` = live count,
    // then two rows per light, `[pos.xyz, range]` and `[rgb, 0]`.
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
    // fork: the environment block (`benilla::env_frame`: wind, wetness, benders), then the
    // rain-shelter region (`weather::shelter`): `[origin_x, origin_z, 1 / cell, active]`,
    // `[base_y, cells per side, 0, 0]` and one packed word a cell.
    env: env_frame::EnvFrame,
    shelter_hdr: vec4<f32>,
    shelter_cfg: vec4<f32>,
    shelter: array<u32, 16384>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

// fork: modern lighting (FORK.md, P6) — the local lights that reach this
// fragment, read out of the world grid the binning pass fills: the WoW falloff per light, under its
// cube shadow when it holds an atlas slot. The walk lives here rather than in
// `benilla_modern::locallight` for the same reason `modern_gi_ambient` does — the light table is
// this lane's own binding, which a WGSL module cannot reach. `lite` (the water mirror) skips the cube
// shadows.
fn modern_local_lights(p: vec3<f32>, nrm: vec3<f32>, lite: bool) -> vec3<f32> {
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
        sum = modern_cell_lights(near_cell, p, nrm, lite) * (1.0 - far_w);
    }
    if (far_w > 0.0) {
        let far_cell = modern_block.light_cells[pair.y];
        sum = sum + modern_cell_lights(far_cell, p, nrm, lite) * far_w;
    }
    return sum;
}

// One grid cell's lights on one fragment — the walk `modern_local_lights` runs once per cell it
// reads.
fn modern_cell_lights(
    cell: vec4<u32>,
    p: vec3<f32>,
    nrm: vec3<f32>,
    lite: bool,
) -> vec3<f32> {
    let g = modern_block.lights;
    let count = locallight::cell_count(cell);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let id = locallight::cell_light(cell, i);
        let pos_range = wow_light.points[2u * id];
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

// Custom vertex→fragment payload: the standard fields the splat fragment needs, PLUS `specular` — the
// per-vertex-clamped sun glint that must be Gouraud-interpolated (Q14). Same struct on both stages, so
// `@builtin(position)` is clip-position out / frag-coord in (`world_position` is unused until Step 5 fog).
struct TerrainVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv_b: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) specular: vec3<f32>,
    // The Gouraud diffuse, summed and clamped per vertex as GL T&L does: an over-gamut light (a
    // carried torch is (1.4, 0.87, 0.40)) saturates one vertex and falls off linearly from it,
    // where clamping per pixel would pin a wide plateau at white.
    @location(6) primary: vec3<f32>,
    // The receiver normal is carried so the real cascaded shadow lookup can apply a slope-aware
    // bias. This is forward-pass data only; the terrain shadow caster is a stock Bevy proxy.
    @location(7) world_normal: vec3<f32>,
    // fork: modern lighting (FORK.md, P3) — the vertex's point-light selection (flat).
    @location(8) @interpolate(flat) point_sel: vec3<u32>,
}

// Terrain's point-light candidacy half-width (yd): `w + 10`, `w` being the chunk's bounding-sphere
// radius `sqrt(2·16.666666² + (zExtent/2)²)`, 23.570166 on flat ground (constant at `0x68dfac`).
// This takes the flat `w` everywhere, so steep chunks gather slightly narrower than the reference;
// the lights that differ are 33 yd out or more, where att ≈ 0.02.
const TERRAIN_REACH: f32 = 33.570166;

// The dynamic point-light term. The reference commits at most three point lights per draw, the
// nearest by squared distance to the receiving unit, into GL slots 1-3; terrain's unit is the MCNK
// chunk, so `anchor` is its cell centre. Candidacy is a 20-yd spatial-hash sweep over
// `[floor((c - w - 10)/20), floor((c + w + 10)/20)]`, a Chebyshev box (`w` is `chunk+0x68`,
// copied at `0x71bc47`) of half-width `w + 10`, at least `TERRAIN_REACH`. A committed light is
// diffuse only (the ambient and specular slots are zeroed at `0x71c7e3`), falls off as
// `1/(0.7·d + 0.03·d²)` on the vertex normal, and has no distance cutoff. The M2 lane in
// wow_model.wgsl uses the packed per-light range: the lanes differ on purpose.
fn point_light_sum(P: vec3<f32>, N: vec3<f32>, anchor: vec3<f32>) -> vec3<f32> {
    let count = u32(wow_light.point_count.x);
    var sel = array<u32, 3>(0u, 0u, 0u);
    var sd = array<f32, 3>(1e30, 1e30, 1e30);
    for (var i = 0u; i < count; i = i + 1u) {
        let pos_range = wow_light.points[2u * i];
        let dv = pos_range.xyz - anchor;
        let d2 = dot(dv, dv);
        // Ground is always an EXTERIOR receiver, so an INDOOR light (`w` = 1) is refused outright.
        // This is not a cull bolted onto a faithful selection — it is what makes it faithful: the
        // reference's register walk has no such term because it never has a culled room's torch to
        // register in the first place, while we create every light once and keep it. Without it
        // Northshire Abbey's candles lit the road through the abbey wall.
        if (wow_light.points[2u * i + 1u].w >= 0.5) {
            continue;
        }
        // The hash sweep is horizontal (the grid keys on WoW x/y = Bevy z/x) and has no vertical
        // bound; ranking below is the full 3-D distance, as `0x71bf90` does.
        if (max(abs(dv.x), abs(dv.z)) > TERRAIN_REACH) {
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
        // Ground is always an EXTERIOR receiver, so an INDOOR light (`w` = 1) is refused outright.
        // This is not a cull bolted onto a faithful selection — it is what makes it faithful: the
        // reference's register walk has no such term because it never has a culled room's torch to
        // register in the first place, while we create every light once and keep it. Without it
        // Northshire Abbey's candles lit the road through the abbey wall.
        if (wow_light.points[2u * i + 1u].w >= 0.5) {
            continue;
        }
        if (max(abs(dv.x), abs(dv.z)) > TERRAIN_REACH) {
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

// fork: perf — `point_light_sum`'s own evaluation of a selection `point_light_select` has already
// made: the two choose by one rule, so this is that sum without its walk of the light table.
fn point_light_gouraud(sel: vec3<u32>, P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < 3u; s = s + 1u) {
        let id = sel[s];
        if (id == 0u) {
            break;
        }
        let pos_range = wow_light.points[2u * (id - 1u)];
        let to_light = pos_range.xyz - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        sum += wow_light.points[2u * (id - 1u) + 1u].rgb * (atten * nl);
    }
    return sum;
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

// The MCNK chunk cell centre under a world point: the reference's light anchor, the chunk's world
// AABB centre `CMapChunk+0x5c..0x64` (written by `0x6b0e50`). The grid is fixed world-wide (chunk
// 533.33333/16 yd, half-extent 32 tiles) and symmetric, and WoW x/y are Bevy −z/−x, so snapping
// Bevy x/z lands on the same cells; wow_model.wgsl mirrors it for clutter.
// The anchor height is the vertex's own y, not the chunk's mid-height `(minH + maxH)/2`, so on
// relief one chunk's vertices can select different lights: the per-chunk constant is not plumbed
// to the vertex stage. On flat ground the two agree.
fn mcnk_cell_anchor(P: vec3<f32>) -> vec3<f32> {
    let cell = 533.33333 / 16.0;
    let half = 32.0 * 533.33333;
    let ix = floor((half + P.x) / cell);
    let iz = floor((half + P.z) / cell);
    return vec3<f32>((ix + 0.5) * cell - half, P.y, (iz + 0.5) * cell - half);
}

@vertex
fn vertex(in: Vertex) -> TerrainVsOut {
    var out: TerrainVsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    let world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.world_normal = world_normal;
    out.uv = in.uv;
    out.uv_b = in.uv_b;
    out.color = in.color;

    // Sun specular per vertex: white material, row-9 colour, shininess `light_spec.w` (20), local
    // viewer; clamped here, then Gouraud-interpolated.
    let n = normalize(world_normal);
    let l = -normalize(wow_light.light_sun.xyz); // to-light (the sun travels along +light_sun)
    let v = normalize(view.world_position.xyz - out.world_position.xyz); // to-eye (local viewer)
    let h = normalize(l + v);
    let ndoth = max(dot(n, h), 0.0);
    out.specular = clamp(wow_light.light_spec.rgb * pow(ndoth, wow_light.light_spec.w), vec3<f32>(0.0), vec3<f32>(1.0));

    // The whole diffuse sum on the MCNR normal, clamped per vertex: ambient, sun N·L, and the
    // chunk's three nearest point lights with their over-gamut colours raw.
    let ndotl = max(dot(n, l), 0.0);
#ifdef WOW_MODERN
#ifdef MODERN_VERTEX_WALKS
    let points = point_light_sum(out.world_position.xyz, n, mcnk_cell_anchor(out.world_position.xyz));
    out.point_sel = point_light_select(mcnk_cell_anchor(out.world_position.xyz));
#else
    // fork: perf — one walk of the light table per vertex, not two: the selection, and the Gouraud
    // sum evaluated from it (`point_light_gouraud`). `MODERN_VERTEX_WALKS` is the A/B lever
    // (`$WOW_MODERN_VERTEXWALKS=1`): the two walks, as before.
    //
    // And none where the fragment reads neither: outside the water mirror it rebuilds the primary
    // per pixel, and with the light grid live it takes its lights from the grid (`fragment`). The
    // block is bound fragment-only here, so the grid's live bit and the mirror's plane reach this
    // stage in the light row's spare lanes (`modern::shadow::VERTEX_HINT_OFFSET`): `.z` is 1 only
    // in a frame both were written to the block, and `.w` is then the plane
    // `sunshadow::mirror_view` tests.
    var points = vec3<f32>(0.0);
    out.point_sel = vec3<u32>(0u);
    if (wow_light.point_count.z <= 0.5 || view.world_position.y < wow_light.point_count.w - 0.25) {
        out.point_sel = point_light_select(mcnk_cell_anchor(out.world_position.xyz));
        points = point_light_gouraud(out.point_sel, out.world_position.xyz, n);
    }
#endif
#else
    let points = point_light_sum(out.world_position.xyz, n, mcnk_cell_anchor(out.world_position.xyz));
    out.point_sel = point_light_select(mcnk_cell_anchor(out.world_position.xyz));
#endif
    let sun_lighting = wow_light.light_diffuse.rgb * ndotl;
    out.primary = clamp(
        wow_light.light_ambient.rgb + sun_lighting + points,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return out;
}

@fragment
fn fragment(in: TerrainVsOut) -> @location(0) vec4<f32> {
    // The far-clip wall: the reference clips the detailed world per pixel at its far plane
    // (`farclip`, about 777 yd). Discard beyond `fog_params.w` (0 disables) on planar eye-Z.
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }

    // Per-chunk array indices baked per vertex, COLOR the 4 layers and UV1.x the alpha; constant
    // across a chunk, so they interpolate back to the integer. (fork: UV1.x's fraction is the
    // depth under water, read by the wet hook below.)
    let li = vec4<i32>(round(in.color));
    let ai = i32(round(in.uv_b.x));

    let tiled = in.uv * t.params.x;
    // `.a` is the `_s` texture's sheen mask, 0 without an `_s`. `view.mip_bias` undoes the coarser
    // mip a render scale below 1 picks; the alpha and shadow arrays have one mip, so only the
    // layers take it.
    let s0 = textureSampleBias(layer_array, splat_samp, tiled, li.x, view.mip_bias);
    let s1 = textureSampleBias(layer_array, splat_samp, tiled, li.y, view.mip_bias);
    let s2 = textureSampleBias(layer_array, splat_samp, tiled, li.z, view.mip_bias);
    let s3 = textureSampleBias(layer_array, splat_samp, tiled, li.w, view.mip_bias);
    // The alpha and shadow maps (one untiled 64² grid per chunk) share the layers' repeat sampler,
    // so a linear footprint at a chunk edge would wrap to the far edge; a half-texel inset clamps.
    let auv = clamp(in.uv, vec2<f32>(0.5 / 64.0), vec2<f32>(1.0 - 0.5 / 64.0));
    let a = textureSample(alpha_array, splat_samp, auv, ai);

    var color = s0.rgb;
    color = mix(color, s1.rgb, a.r);
    color = mix(color, s2.rgb, a.g);
    color = mix(color, s3.rgb, a.b);

    // The sheen mask, blended with the colour's splat weights (the fragment program's
    // `secondary · texture[0].w`).
    var specmask = s0.a;
    specmask = mix(specmask, s1.a, a.r);
    specmask = mix(specmask, s2.a, a.g);
    specmask = mix(specmask, s3.a, a.b);

    // STEP 3 (diffuse) arrives from the vertex stage, ALREADY summed (ambient + sun N·L + the
    // chunk's ≤3 committed point lights) and ALREADY clamped — Gouraud of the clamped vertex
    // colour, byte-faithful to GL T&L (see the `primary` struct note; the MCNR normal and the
    // DayNight sun share a space per the Phase-0 validation). The MCSH ×(0.3·s + 0.7) factor below
    // still scales the whole modulate, as the traced combine does.
    // Fetch the character coverage here, but apply it as a NEUTRAL scalar after the terrain's
    // authored lighting below. Subtracting the warm sun RGB left only Elwynn's green ambient and
    // produced a green silhouette on yellow morning ground. The era-style projected shadow is a
    // greyscale attenuation: it darkens the existing hue instead of changing it.
    // Keep the raw fetch for the specular gate below. The realtime map now contains CHARACTERS
    // only; MCSH contains the static world's baked blockers. They are independent occluders, so
    // the character map must never replace (and thereby brighten) the authored terrain bake.
    var world_shadow = 1.0;
    if (lights.n_directional_lights > 0u) {
        let view_z = (view.view_from_world * in.world_position).z;
        for (var light_id = 0u; light_id < lights.n_directional_lights; light_id = light_id + 1u) {
            if ((lights.directional_lights[light_id].flags & 1u) != 0u) {
                world_shadow = shadows::fetch_directional_shadow(
                    light_id,
                    in.world_position,
                    normalize(in.world_normal),
                    view_z,
                    // Bevy 0.19: the pixel seeds the filter's per-fragment rotation.
                    in.clip_position.xy,
                );
                break;
            }
        }
    }
    // fork: modern lighting (FORK.md, P2) — the modern sun's own cascades replace
    // the realtime fetch; shadow removes the sun's share of the Gouraud primary, never the ambient.
    // fork: modern lighting — a shader def, not a storage read. Both models used to be compiled
    // into every terrain pipeline and chosen at runtime; see `materials::TerrainKey` for why that
    // is the shape that broke Metal's cutout discard on the model lane.
#ifdef WOW_MODERN
    let modern = true;
#else
    let modern = false;
#endif
    var primary = in.primary;
#ifdef WOW_MODERN
    // The water mirror takes none of this (`sunshadow::mirror_view`): it keeps the Gouraud primary —
    // ambient, the sun's N·L and the three nearest lights, per vertex — with no cascades, cloud
    // shadow, probe field, sky map or light grid. A reflection is bent and blurred by the waves; the
    // second render of the whole world is what the frame pays for, so it pays for the plain one.
    let lite = sunshadow::mirror_view(modern_block.cascades, view.world_position);
    if (!lite) {
        let mview_z = (view.view_from_world * in.world_position).z;
        let mn = normalize(in.world_normal);
        // fork: modern lighting (FORK.md, P3) — the sun is evaluated per pixel under the cascades
        // (Lambert on the MCNR normal) instead of Gouraud per vertex. The primary's point-light share
        // rides along, recovered as what the vertex sum holds beyond ambient + sun.
        let ml = -normalize(wow_light.light_sun.xyz);
        let mndotl = max(dot(mn, ml), 0.0);
        // fork (perf): a slope facing away from the sun takes none of its light, so it skips the
        // cascade and cloud reads; only the sheen gate reads 0 there, under a vanishing N·H^20.
        // The shadow debug views keep the reads, so they still show what the cascades hold.
        world_shadow = 0.0;
        if (mndotl > 0.0 || modern_block.cascades.params.w > 0.5) {
            world_shadow = sunshadow::sun_visibility(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz, mn, -mview_z, lite);
            // fork: modern lighting (FORK.md, P4) — cloud shadows sweep over the ground.
            let cloud_ct = sunshadow::cloud_taps(modern_block.fog.cloud, in.world_position.xyz);
            world_shadow = world_shadow * sunshadow::cloud_mix(cloud_ct, modern_block.cloud[cloud_ct.idx.x], modern_block.cloud[cloud_ct.idx.y], modern_block.cloud[cloud_ct.idx.z], modern_block.cloud[cloud_ct.idx.w]);
        }
        // fork: modern lighting (FORK.md, P6) — the binned local-light grid
        // replaces the reference's three-nearest selection wherever it is live; Low has no grid and
        // keeps P3's per-pixel evaluation of that selection.
        var points = point_light_eval(in.point_sel, in.world_position.xyz, mn);
        if (locallight::grid_live(modern_block.lights)) {
            points = modern_local_lights(in.world_position.xyz, mn, lite);
        }
        // fork: modern lighting (FORK.md, P5) — the probe field replaces the ambient
        // row wherever it has an answer; the sky map is what is left where it does not.
        // Terrain is never an interior receiver: an ADT chunk is the open world by construction.
        let gi = modern_gi_ambient(in.world_position.xyz, mn, 0.0, lite);
        // fork: modern lighting (FORK.md, P3) — ambient dimmed by the sky-occlusion
        // map, read only where the probe field does not answer whole (`sunshadow::probe_or_sky`).
        var sky = 1.0;
        if (gi.w < 1.0) {
            sky = sunshadow::sky_ambient(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz, lite);
        }
        let ambient = sunshadow::shade_floor(sunshadow::probe_or_sky(wow_light.light_ambient.rgb, sky, gi), wow_light.light_ambient.rgb);
        primary = ambient + wow_light.light_diffuse.rgb * (mndotl * world_shadow) + points;
    }
#endif

    // MCSH baked shadow. With `pixelShaders` and `specular` on, terrain is one pass through
    // `terrainp_s.bls`, which reads the MCSH bit from the blend texture's alpha (1 lit, 0 shadowed)
    // and uses it twice, below; with them off the reference draws a separate ambient-tint overlay.
    // Our `shadow_array` is 255 shadowed, 0 lit; a chunk with no MCSH map (`uv_b.y < 0`) is lit.
    // fork: modern lighting — MCSH RETIRES in a modern frame (P3), and the three `select`s below
    // used to prove it by sampling the map and then throwing the sample away. The sample is a live
    // texture read and `shadow_array` is a live binding, on the lane that sits at the 16-texture
    // cap; behind the def, a modern pipeline reads neither.
#ifndef WOW_MODERN
    var shadow_lit = 1.0;
    let si = in.uv_b.y;
    if (si >= 0.0) {
        // `textureSampleLevel`, not `textureSample`: this sits in a per-fragment branch, and WGSL's
        // uniformity rule (enforced by the browser's compiler, not by naga) forbids the implicit
        // derivative there. The shadow grid has one mip anyway, so level 0 is the same sample.
        let mcsh = textureSampleLevel(shadow_array, splat_samp, auv, i32(round(si)), 0.0).r;
        shadow_lit = 1.0 - mcsh;
    }
    // MCSH always remains the authored base. Realtime character coverage applies a neutral
    // attenuation and closes the sheen gate. This makes the result monotonic and hue-preserving:
    // a character can only darken the terrain already underneath it.
    let shadow_lit_eff = shadow_lit;
    let spec_gate = min(shadow_lit, world_shadow);
    let character_shadow_term = mix(SHADOW_SUN_FLOOR, 1.0, world_shadow);
#else
    // The same three values the `select`s produced on the modern arm, with nothing sampled to
    // reach them.
    let shadow_lit_eff = 1.0;
    let spec_gate = world_shadow;
    let character_shadow_term = 1.0;
#endif

    // The faithful `terrainp_s` combine (Q15):
    //   diffuse  = tex · primary · (0.3·shadow + 0.7)   → a flat −30% in shadow, NO colour tint
    //   specular = per-vertex sheen · gloss_mask · shadow → gated to ZERO in shadow (no sheen in shade)
    // (`tex·primary` is the MODULATE-1× diffuse; the sheen is added after, separate-specular.) Then
    // LDR-clamp; gamma/byte throughout; raw gamma out (GAMMA LANE, 0161).
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
        // Nor under water: the depth below the chunk's own liquid rides the alpha index's
        // fraction (`adt::pack_water_depth`).
        wet_open = wet_open
            * wet_hook::above_water((in.uv_b.x - round(in.uv_b.x)) * WATER_DEPTH_UNPACK);
    }
    // Rain-darkened ground and puddles; a dry frame returns `color` untouched. The combine below is
    // gamma math in either lighting model (a modern frame decodes at the output), so one hook
    // serves both.
    let wet_n = normalize(in.world_normal);
    let wet = wet_hook::wet_puddles(wet_hook::wet_surface(color, wet_n, in.world_position.xyz,
        wet_open, wow_light.env), wet_n, in.world_position.xyz, wet_open, wow_light.env);
    color = wet.albedo;
    let diffuse_term = color * primary * (0.3 * shadow_lit_eff + 0.7) * character_shadow_term;
    let spec_term = in.specular * specmask * spec_gate;
    var tuned = clamp(diffuse_term + spec_term, vec3<f32>(0.0), vec3<f32>(1.0));
    // The wet sheen, and the authored `_s` sheen brightens a little with it. Its shadow term is the
    // sheen gate: MCSH and the character map in a classic frame, the sun cascades in a modern one.
    // The reflection is the screen-space pass's while it runs (`weather::wet_ssr`): the weight
    // rides out in alpha, `1 + weight`, and the pass adds what the reflected ray finds. A modern
    // frame's water mirror keeps the sky term, since the pass never sees that target; a classic
    // mirror cannot tell itself apart and reflects wet ground without it.
    var wet_reflect = 0.0;
    if (wet.boost > 0.0) {
        let wet_v = normalize(view.world_position - in.world_position.xyz);
        var sheen = spec_term * (WET_SPEC_GAIN * min(wet.boost, 1.0))
            + wet_hook::wet_sun(wet.boost, wet_n, wet_v, -normalize(wow_light.light_sun.xyz),
                wow_light.light_diffuse.rgb, spec_gate, WET_SUN_GAIN);
        let reflectance = WET_REFLECT_GAIN * wet_hook::wet_reflectance(wet.boost, wet_n, wet_v);
        var tag = wet_hook::ssr_live(wow_light.env);
#ifdef WOW_MODERN
        tag = tag && !lite;
#endif
        if (tag) {
            wet_reflect = reflectance;
        } else {
            sheen = sheen + wow_light.fog_color.rgb * reflectance;
        }
        tuned = clamp(tuned + sheen, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // Gamma-space GL_LINEAR fog on planar eye-Z, as the reference computes `fogcoord`, not radial
    // distance. Per pixel matches its per-vertex fog, since a linear factor interpolates exactly.
    // fork: modern lighting (FORK.md, P4) — a modern frame is fogged by the composite.
    if (wow_light.fog_color.w > 0.5 && !modern) {
        let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        let denom = max(wow_light.fog_params.y - wow_light.fog_params.x, 0.001);
        let factor = clamp((wow_light.fog_params.y - eye_z) / denom, 0.0, 1.0);
        tuned = mix(wow_light.fog_color.xyz, tuned, factor);
        // The reflection fogs with the ground it lies on.
        wet_reflect = wet_reflect * factor;
    }

    // GAMMA LANE (decision 0161): the framebuffer HOLDS gamma bytes — all blending happens in
    // gamma exactly like the reference's byte framebuffer; the frame's ONE decode lives in the
    // FFXGlow combine (the last node). Output raw.
    // fork: modern lighting (FORK.md) — the frame is linear.
    if (modern && modern_block.cascades.params.w > 0.5) {
        let dbg = modern_block.cascades.params.w;
        if (dbg > 4.5) {
            // fork: modern lighting (P6) — mode 5: what the local-light grid put here.
            tuned = modern_local_lights(in.world_position.xyz, normalize(in.world_normal), false);
        } else if (dbg > 2.5) {
            let gdbg = modern_gi_ambient(in.world_position.xyz, normalize(in.world_normal), 0.0, false); // modern GI debug view
            tuned = select(gdbg.rgb, vec3<f32>(gdbg.w), dbg > 3.5);
        } else {
            tuned = vec3<f32>(select(world_shadow, sunshadow::sky_visibility(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz), dbg > 1.5)); // modern shadow debug view
        }
        wet_reflect = 0.0;
    }
    // fork: alpha above 1 is the wet reflection tag; every other opaque write stays at or below 1.
    return vec4<f32>(colorspace::modern_out(tuned, wow_light.point_count.y), 1.0 + wet_reflect);
}
