// Phase 6 terrain splat shader — extends Bevy's StandardMaterial with a CUSTOM vertex + fragment stage.
//
// Blends up to 4 tiled layer textures by a per-chunk alpha map, then lights the result with WoW's
// faithful terrain lighting (fixed-function GL_LIGHTING, reproduced):
//   • DIFFUSE — `clamp(ambient_row1 + diffuse_row0·max(N·L,0) + Σ point·att·N·L)`, modulated into
//     the texture (MOD 1×). Evaluated AND CLAMPED per VERTEX (the GL T&L locus), then
//     Gouraud-interpolated — the clamp position matters once an over-gamut point light is in the
//     sum (see the `primary` varying note).
//   • SPECULAR (the bare-ground "sheen") — `clamp(row9 · pow(max(N·H,0),20)) · sheen_mask`, added
//     AFTER the texture modulate (separate-specular, LOCAL_VIEWER). The Blinn term is computed
//     PER-VERTEX (clamped per vertex, Gouraud-interpolated); the `sheen_mask` is the `_s` texture's
//     per-texel ALPHA, blended like the colour (so the sheen rides only shiny texels — stone, not
//     grass). See docs/knowledge/lighting.md
//     ⚠️ Two things bound it, BOTH needed (each was a separate blow-out): (a) per-vertex undersampling
//     of the sharp lobe (sampled only at the ~4 yd MCVT vertices, clamped, linearly smeared) vs a
//     per-PIXEL eval that saturates; (b) the `_s` alpha sheen-mask (ref mean ≈ 0.1) vs applying it at
//     a uniform 1.0 — row 9 ≈ near-white, so unmasked it washes the whole lit face to cream.
//
// Because Bevy's `VertexOutput` has no slot for the interpolated specular, we do our own (minimal)
// vertex transform and a custom IO struct, and drop `main_pass_post_lighting_processing` (fog returns
// IN-SHADER at Step 5, per docs/plans/lighting-rebuild.md — not Bevy's DistanceFog).
//
// One material serves a whole ADT tile (see terrain.rs): layer textures live in a `texture_2d_array`
// (binding 100), per-chunk alpha maps in another (104). Each merged-mesh vertex carries its chunk's 4
// layer indices in `color` (vertex COLOR) and its alpha-layer index in `uv_b.x` (UV1) — constant across
// the chunk, so interpolation lands back on the integer index.

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import benilla_modern::sunshadow
#import benilla_modern::probegi
#import benilla_modern::locallight
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

// Per-tile Vec4 uniforms packed into ONE buffer (binding 106) — the field order here MUST match the
// Rust `TerrainExtension` struct. Light + fog live in the shared global-light storage buffer (below);
// what's left is just the per-tile layer tiling factor.
//   params — x = layer tiling factor; yzw unused.
struct TerrainParams {
    params: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var<uniform> t: TerrainParams;

// The shared global light (lighting::global_light): ONE storage buffer every material reads, updated
// once/frame in place — replaces the per-material light/fog uniforms `apply_wow_lighting` re-pushed
// each frame. Rows 0-5 (of the 18-row LightStd430) are the light + fog terrain needs, plus the grade
// row at the end (rows 6-16 — model SH + water swatches — are skipped as a pad). Read in BOTH stages
// (the per-vertex sheen needs light_sun/spec in the vertex stage).
struct WowLight {
    light_ambient: vec4<f32>, // rgb = row 1 ambient; w = Mod2x scale (×1).
    light_diffuse: vec4<f32>, // rgb = row 0 sun diffuse; w = clamp-light flag (>0.5 ⇒ saturate).
    light_sun: vec4<f32>,     // xyz = world-space sun TRAVEL direction (to-light = −xyz); w unused.
    light_spec: vec4<f32>,    // rgb = row 9 specular color; w = shininess (20). rgb == 0 disables.
    fog_color: vec4<f32>,     // rgb = row 7 fog (raw, gamma 0..1); w = enable (>0.5 ⇒ blend).
    fog_params: vec4<f32>,    // x = fog_start yd; y = fog_end yd; z unused; w = farclip wall.
    _sh: array<vec4<f32>, 6>, // rows 6-11: the model SH coeffs (live in wow_model.wgsl, 0354) — unread by terrain.
    sh_c16: vec4<f32>,        // row 12: xyz the models' c16 quad band; .w a FREE lane (the 0273 point gain is retired).
    _water: array<vec4<f32>, 4>, // rows 13-16: the liquid swatches — unread by terrain.
    grade: vec4<f32>,         // reserved (the 0282 interior A/B retired; 0163). Layout only.
    _wmo_fog: array<vec4<f32>, 2>, // rows 18-19: the interior fog triple — unread by terrain.
    // The dynamic point-light table (decision 0278), packed by `global_light::build_light_data`:
    // row 20 `.x` = live entry count; then TWO rows per light — `[pos.xyz, range]`, `[rgb, 0]`.
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
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
                pos_range.xyz,
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
    // The full Gouraud diffuse — `clamp01(ambient + sun·max(N·L,0) + Σ point·att·N·L)`, evaluated
    // AND CLAMPED per VERTEX: the GL T&L locus. The clamp position is load-bearing for a hot
    // carried light (the raw committed torch is (1.4, 0.87, 0.40)): GL clamps the summed vertex
    // colour BEFORE interpolation, so an over-gamut torch saturates one vertex and Gouraud falls
    // linearly away from it. Interpolating the raw sum and clamping per pixel — what we did
    // before — keeps pixels pinned at white far from the hot vertex: a wide flat plateau with a
    // hard knee, the director's "hard square". Sun N·L moving per-pixel → per-vertex is the
    // faithful direction (Step 3's own note: indistinguishable on the smooth term).
    @location(6) primary: vec3<f32>,
    // The receiver normal is carried so the real cascaded shadow lookup can apply a slope-aware
    // bias. This is forward-pass data only; the terrain shadow caster is a stock Bevy proxy.
    @location(7) world_normal: vec3<f32>,
    // fork: modern lighting (FORK.md, P3) — the vertex's point-light selection (flat).
    @location(8) @interpolate(flat) point_sel: vec3<u32>,
}

// Terrain's point-light **candidacy half-width** (yd): the reference's guaranteed covered box is
// `w + 10` where `w` is the chunk's bounding-sphere radius — `sqrt(2·16.666666² + (zExtent/2)²)`,
// i.e. 23.570166 on flat ground (`0x68df70` hard-codes that same constant at `68dfac`). Relief
// widens a chunk's own `w`; we use the flat value, so on steep ground we gather very slightly
// narrower than the reference — the lights that differ are ≥33 yd out, where `att ≈ 0.02`.
const TERRAIN_REACH: f32 = 33.570166;

// The dynamic point-light term at a world-space point (decisions 0016/0273/0278, selection 0285) —
// the reference commits AT MOST THREE point lights per draw, the nearest to the receiving unit's own
// position (the gather sorts by squared distance from the unit, the commit seats GL slots 1-3 —
// wow-re `wmo-surface-dynamic-light` §4/§6). Terrain's unit is the 33.33-yd MCNK chunk, so the anchor
// is the chunk cell center under the vertex (our tiles merge chunks into one world-space mesh —
// `mcnk_cell_anchor` recovers the chunk analytically). Pass 1 selects; pass 2 evaluates only the
// selected lights at the vertex — the byte-verified falloff `1/(0.7·d + 0.03·d²)`, diffuse-only
// (committed ambient/specular are zero — VERIFIED: the staging struct's ambient/specular slots are
// explicitly zeroed at `71c7e3`-`71c7ec` right before the point loop), on the submitted vertex
// normal, no distance cutoff on the EVALUATION (a committed GL light has none; candidacy alone
// bounds the set).
//
// **Candidacy is the reference's spatial-hash sweep, not a radius** (VERIFIED, wow-re
// `terrain/scratch/terrain-dynamic-light-gather.md`): the gather walks a 20-yd-cell hash over a cell
// span `[floor((c − w − 10)/20), floor((c + w + 10)/20)]` — a **Chebyshev box**, no circular reject —
// where terrain's own `w` is the chunk's bounding-SPHERE radius `chunk+0x68`
// (`sqrt(2·16.666666² + (zExtent/2)²)` = 23.570166 on flat ground, `0x71bc30` copies it at
// `71bc47`-`71bc5f`). That gives a guaranteed covered half-width of `w + 10` = [`TERRAIN_REACH`].
// We used a 48-yd SPHERE here — an INFERRED stand-in from decision 0285, before anyone had read the
// terrain side. It over-gathered: with 29 live point lights measured around Darkshire against three
// committed slots, every extra candidate is another chance for a chunk's ≤3 set to flip as an
// emitter moves, which is chunk-shaped popping the reference never has.
//
// The M2 lane in wow_model.wgsl still uses the packed per-light range — its own `w` is 0 (a ±10 yd
// box, a *different* byte-pinned number), a change that would move the approved interior look, so it
// is the director's to weigh. The two lanes diverge on purpose; do not "fix" them back into one.
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

// The MCNK chunk cell center under a world point — terrain's light-selection anchor (the terrain
// draw unit; the grid is fixed world-wide, chunk = 533.3333/16 yd, half-extent 32 tiles). WoW x/y
// are Bevy −z/−x and the grid is symmetric, so snapping Bevy x/z directly lands on the same cells.
//
// **The XY snap is VERIFIED** (wow-re `terrain-dynamic-light-gather.md`: `CMapChunk+0x5c..0x64` is
// the chunk's world AABB centre, written by the MCVT world-vert builder `0x6b0e50` at
// `6b106c`/`6b10ec`, and the 33.33-yd chunk really is the draw unit — gather → commit → draw pair
// 1:1 per record at `68478a`/`68479c`/`6847bb`/`6847c4`).
//
// **The height is KNOWN-WRONG and deliberately not fixed here.** The reference's anchor Z is the
// chunk's own mid-height `(minH + maxH)/2` over its 145 verts — one constant per chunk. We keep the
// vertex's own y, which means the anchor VARIES within a chunk, so on relief two vertices of the
// same chunk can select different light sets and seam mid-chunk — the reference cannot do that.
// Fixing it needs a per-chunk constant plumbed to the vertex stage (a custom mesh attribute + a
// `specialize` on the terrain material, which has none today), so it is scoped as its own change;
// on flat ground the two anchors coincide and nothing differs. Mirrored in wow_model.wgsl (clutter
// snaps to the same cells).
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

    // STEP 3b: terrain SUN SPECULAR, computed PER-VERTEX (Q14). Material spec = white, shininess =
    // wow_light.light_spec.w (20), spec color = row 9 (wow_light.light_spec.rgb); H = halfway(to-light, to-eye) with
    // the LOCAL viewer (per-vertex eye dir). Clamp HERE, per vertex — rasterizer Gouraud-interpolates.
    let n = normalize(world_normal);
    let l = -normalize(wow_light.light_sun.xyz); // to-light (the sun travels along +light_sun)
    let v = normalize(view.world_position.xyz - out.world_position.xyz); // to-eye (local viewer)
    let h = normalize(l + v);
    let ndoth = max(dot(n, h), 0.0);
    out.specular = clamp(wow_light.light_spec.rgb * pow(ndoth, wow_light.light_spec.w), vec3<f32>(0.0), vec3<f32>(1.0));

    // STEP 3 at the faithful locus — the WHOLE diffuse sum, clamped HERE per vertex (see the
    // struct note). Sun: `ambient + diffuse·max(N·L,0)` on the MCNR normal; points: the ≤3-nearest
    // committed lights of this vertex's MCNK chunk cell — the terrain draw unit (0285) — at the
    // byte-verified `1/(0.7d + 0.03d²)`, raw over-gamut colours in.
    let ndotl = max(dot(n, l), 0.0);
    let points = point_light_sum(out.world_position.xyz, n, mcnk_cell_anchor(out.world_position.xyz));
    out.point_sel = point_light_select(mcnk_cell_anchor(out.world_position.xyz));
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
    // HARD FAR-CLIP WALL (faithful `farclip`): the reference clips the detailed world at its projection
    // far plane (~777 yd). Geometry beyond is GPU-clipped per-pixel, so a tall object reveals
    // closest-part-first as you approach and the sky/WDL shows behind it. We reproduce that with a
    // per-fragment discard on PLANAR eye-Z (same coordinate as the fog) beyond `fog_params.w` = farclip
    // (0 ⇒ disabled). WDL/sky use other shaders, so they still render to the horizon behind the wall.
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }

    // Per-chunk array indices, baked into the merged mesh: COLOR = 4 layer indices, UV1.x = alpha.
    let li = vec4<i32>(round(in.color));
    let ai = i32(round(in.uv_b.x));

    let tiled = in.uv * t.params.x;
    // Full RGBA: `.rgb` = diffuse, `.a` = the `_s` texture's per-texel SHEEN MASK (see assets.rs /
    // terrain.rs — the base `.blp` has no alpha, so we load the `_s` variant; matte where no mask).
    // `view.mip_bias` (decision 1639): a render scale below 1 doubles every derivative and so
    // picks a coarser mip, and that blur is then stretched back up by the resolve. The bias undoes
    // exactly that shift and is 0.0 at native and above, so this is an identity at scale 1.
    // Only the LAYER array takes it: `alpha_array`/`shadow_array` are one 64x64 grid per chunk,
    // built with a single mip level (terrain.rs `array_image`), so no bias could move them.
    let s0 = textureSampleBias(layer_array, splat_samp, tiled, li.x, view.mip_bias);
    let s1 = textureSampleBias(layer_array, splat_samp, tiled, li.y, view.mip_bias);
    let s2 = textureSampleBias(layer_array, splat_samp, tiled, li.z, view.mip_bias);
    let s3 = textureSampleBias(layer_array, splat_samp, tiled, li.w, view.mip_bias);
    // Alpha + shadow maps are one 64² grid per chunk in 0..1 (NOT tiled), yet they share the layer
    // textures' Repeat sampler. The old "repeat == clamp here" assumption is FALSE under LINEAR
    // filtering: at a chunk edge (uv→0/1) the bilinear footprint WRAPS to the chunk map's opposite
    // edge, blending unrelated weights → a thin seam at every chunk border (the creases; introduced
    // when 228d336 switched Nearest→Linear). Inset by half a texel so the footprint clamps instead.
    let auv = clamp(in.uv, vec2<f32>(0.5 / 64.0), vec2<f32>(1.0 - 0.5 / 64.0));
    let a = textureSample(alpha_array, splat_samp, auv, ai);

    var color = s0.rgb;
    color = mix(color, s1.rgb, a.r);
    color = mix(color, s2.rgb, a.g);
    color = mix(color, s3.rgb, a.b);

    // Sheen mask: blend the layers' `_s` alpha with the SAME splat weights as the colour, so the sun
    // specular rides only the shiny texels (stone/pebbles), not matte grass/dirt. This per-pixel mask
    // is what bounds the highlight — the real client's mask averages ≈ 0.1 (Q14); applying the sheen at
    // a uniform 1.0 is what blew out the floor. (FP: `secondary · texture[0].w`.)
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
    {
        let mview_z = (view.view_from_world * in.world_position).z;
        let mn = normalize(in.world_normal);
        // The water mirror shades its cheaper tier (`sunshadow::mirror_view`).
        let lite = sunshadow::mirror_view(modern_block.cascades, view.world_position);
        world_shadow = sunshadow::sun_visibility(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz, mn, -mview_z, lite);
        // fork: modern lighting (FORK.md, P4) — cloud shadows sweep over the ground.
        let cloud_ct = sunshadow::cloud_taps(modern_block.fog.cloud, in.world_position.xyz);
        world_shadow = world_shadow * sunshadow::cloud_mix(cloud_ct, modern_block.cloud[cloud_ct.idx.x], modern_block.cloud[cloud_ct.idx.y], modern_block.cloud[cloud_ct.idx.z], modern_block.cloud[cloud_ct.idx.w]);
        // fork: modern lighting (FORK.md, P3) — the sun is evaluated per pixel under the cascades
        // (Lambert on the MCNR normal) instead of Gouraud per vertex. The primary's point-light share
        // rides along, recovered as what the vertex sum holds beyond ambient + sun.
        let ml = -normalize(wow_light.light_sun.xyz);
        let mndotl = max(dot(mn, ml), 0.0);
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

    // STEP 4: MCSH baked shadow. On the reference path (pixelShaders+specular) terrain is ONE pass
    // through `terrainp_s.bls`, which carries the MCSH bit in the blend texture's alpha (1.0 lit / 0.0
    // shadowed) and uses it TWO ways (Q15) — NOT the Q11 separate ambient-tint overlay (that's the
    // legacy/non-pixelShaders path). Our `shadow_array` R8 is 255=shadowed/0=lit, so `shadow_lit =
    // 1 - mcsh`. Chunks with no MCSH map (`uv_b.y < 0`) are fully lit.
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
    let diffuse_term = color * primary * (0.3 * shadow_lit_eff + 0.7) * character_shadow_term;
    let spec_term = in.specular * specmask * spec_gate;
    var tuned = clamp(diffuse_term + spec_term, vec3<f32>(0.0), vec3<f32>(1.0));

    // STEP 5: gamma-space linear fog (q6). Applied AFTER tone/curve (so the diagnostic knobs see
    // the unfogged lit pixel) and BEFORE the viz bypass + raw output. The fog coordinate is
    // PLANAR eye-Z (view-space depth along the camera forward axis), NOT radial distance: the
    // reference VS computes `fogcoord` as a linear function of a single axial coordinate (eye-Z) —
    // VERIFIED from the host GLSL fog source in two apitraces (WoW.5/WoW.8). Radial `length()` is
    // ≥ planar everywhere off-axis, so it OVER-fogs the screen edges (worst in short-fog zones like
    // Duskwood, +0.26–0.32 fog factor at the corners). The per-vertex T&L original and this
    // per-pixel form land on the same byte under LINEAR fog (the factor interpolates exactly), and
    // the per-pixel form is what survives Bevy's mesh interpolators without a custom slot.
    // fork: modern lighting (FORK.md, P4) — a modern frame is fogged by the composite.
    if (wow_light.fog_color.w > 0.5 && !modern) {
        let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        let denom = max(wow_light.fog_params.y - wow_light.fog_params.x, 0.001);
        let factor = clamp((wow_light.fog_params.y - eye_z) / denom, 0.0, 1.0);
        tuned = mix(wow_light.fog_color.xyz, tuned, factor);
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
    }
    return vec4<f32>(colorspace::modern_out(tuned, wow_light.point_count.y), 1.0);
}
