// The B1 retained static pass (decision 1429; `static_gx/mod.rs`): the never-fade order-free
// static-world law of `wow_model.wgsl`, re-expressed over retained buffers — per-vertex
// texture-array layer + flag bits (+ a per-item record) where the entity path had one
// material per batch. Slice 1 = exterior ADT doodads; slice 2 adds WMO group geometry.
//
// **The law here MIRRORS `benilla_assets/shaders/wow_model.wgsl` and must track it** (the
// terrain.wgsl precedent: naga_oil cannot import functions that reference another module's
// bindings, so the shared pieces are copied with keep-in-sync notes at both ends). Mirrored
// pieces, in order: the `WowLight` per-frame PREFIX (the buffer is bigger — probe/palette
// regions ride the tail; binding a prefix-shaped struct against the same buffer is exactly
// what terrain.wgsl does), `wow_normalize`, `point_light_sum`, the exterior-doodad sun family
// (the Model2.bls order-2 lobe, 0803 — `min(I,1)` cap included, with its recorded caveat),
// the WMO surface lanes (slice 2: MOCV inside the clamp under GL_COLOR_MATERIAL, the
// INT/TRANS/EXT batch-class lanes, the WINDOW midpoint light, SIDN × night on lit lanes,
// ZERO point lights, the authored batch-order clip-z nudge, the interior fog triple), the
// interior-PROP probe lane (B4, decision 1433: the per-item slot rides the record table and
// the probe rows live in the TAIL of the same shared buffer — zero live point lights, the
// group-MOLR lobes are folded into the probe at spawn), the exterior-prop Matte fixed-1.0
// family (B4), and the step-5 fog. Deliberately ABSENT (this lane never carries them): the
// clutter/rig/fade/env-map/highlight/mat-anim lanes and per-instance tint (identity for
// statics).
//
// Population contract (enforced by the collector's divert — `static_gx::StaticGx::divert`):
// never-fade order-free non-animated statics (interior props are never-fade by law),
// Opaque/AlphaTest only, no env-map, no depth flags; cells additionally
// `ShadeSel::Lit`/`Shaded` only (the WMO lane never reads the selector; the prop lane adds
// `Matte`). Output alpha is pinned 1.0 — every draw here is opaque-intent by admission
// (wow_model.wgsl's bit-3 armor, as a constant).

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import benilla_modern::sunshadow
#import benilla_modern::probegi
#import benilla_modern::locallight
#import bevy_pbr::{
    mesh_view_bindings::{lights, view},
    shadows,
}

// Group 0 is Bevy's standard mesh-view bind group (view matrices, directional-light records and
// the shadow textures the retained pass reads).

// **How dark a fully covered character shadow gets** — the same number `benilla-assets`'
// `terrain.wgsl` and `wow_model.wgsl` carry; move all three together. Scales the SUN term only:
// the subtract-and-restore below keeps the authored ambient, and with it the retained pass's
// fixture/probe lighting. 0.70, raised from the 0.45 this first shipped at.
const SHADOW_SUN_FLOOR: f32 = 0.70;

// fork: modern lighting (FORK.md, P6) — the interior look's numbers, carried by
// value beside `wow_model.wgsl`'s copies; move the two together.
const MODERN_MOCV_EMISSIVE: f32 = 2.0;
const MODERN_SIDN_GAIN: f32 = 2.5;
const MODERN_SIDN_DAY: f32 = 0.35;
const MODERN_ROOM_FLOOR: f32 = 0.30;

// The shared global light's per-frame PREFIX (lighting::global_light; the full layout lives in
// wow_model.wgsl — keep field order in sync with BOTH).
struct WowLight {
    light_ambient: vec4<f32>,
    light_diffuse: vec4<f32>,
    light_sun: vec4<f32>,
    light_spec: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    sh_c10_r: vec4<f32>,
    sh_c10_g: vec4<f32>,
    sh_c10_b: vec4<f32>,
    sh_c13_r: vec4<f32>,
    sh_c13_g: vec4<f32>,
    sh_c13_b: vec4<f32>,
    sh_c16: vec4<f32>,
    _water: array<vec4<f32>, 4>,
    grade: vec4<f32>,
    wmo_fog_color: vec4<f32>,
    wmo_fog_params: vec4<f32>,
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
    // The interior-prop SH probe table (B4 — lighting::prop_probes, 7 rows per slot, 8192
    // slots; wow_model.wgsl owns the layout note). Sits immediately after the point table in
    // the shared buffer, so extending the mirrored prefix by one region is all it takes; the
    // rig/tint/mat-anim regions beyond it stay unmirrored (statics never read them).
    prop_probes: array<vec4<f32>, 57344>,
}
@group(2) @binding(0) var<storage, read> wow_light: WowLight;
// fork: modern lighting — the latched lighting model as a COMPILE-TIME fact
// (`static_gx::render`'s `WOW_MODERN` def, keyed on `materials::MODERN_MODEL_PIPELINES`).
//
// It replaces the `wow_light.point_count.y > 0.5` storage read this lane used to gate on. That read
// is still the right question for the MODEL lane, whose booths and glue scene pack their own
// classic light blob inside a modern session; this lane has no such view — it draws the world
// camera and the water mirror, and both read the modern buffer — so here the model is known when
// the pipeline is built. The heavy branches are `#ifdef`-removed outright; this constant is for the
// handful of cheap predicates where a `select` reads better than a preprocessor block.
#ifdef WOW_MODERN
const MODERN_FRAME: bool = true;
#else
const MODERN_FRAME: bool = false;
#endif

// fork: modern lighting (FORK.md, P2)
@group(2) @binding(1) var modern_shadow_map: texture_depth_2d_array;
@group(2) @binding(2) var modern_shadow_samp: sampler_comparison;
@group(2) @binding(3) var<storage, read> modern_block: sunshadow::ModernBlock;

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
// fork: modern lighting (P6, gated in P8) — the local lights reaching one fragment.
//
// `interior` is the receiver's own INTERIOR bit, and it is a gate, not a grading term: a light that
// names no WMO rooms may not light an interior surface at all. A local light has no occluder unless
// it holds one of the twelve cube slots, and `global_light`'s room term admits every room-less light
// by law, so without this a street lamp lights the inside of the chapel beside it through the wall.
// Lights that DO name rooms already went through the portal gate and are not touched here.
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
        // `w` is 1 for a light that is INDOORS (`global_light`'s packer), and the test runs BOTH
        // ways: an interior surface takes only indoor lights, an exterior one only outdoor lights.
        // A local light has no occluder unless it holds a cube slot, so without this a lamp in the
        // street lights the inside of the chapel beside it — and Northshire Abbey's candles light
        // the road outside it, which is the same bug with the receiver swapped.
        let indoor_light = wow_light.points[2u * id + 1u].w >= 0.5;
        if (interior != indoor_light) {
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

// Per-cell state (static_gx/render.rs `cell_layout`).
struct GxCell {
    origin: vec4<f32>, // xyz = the bake's recentring origin (0974): world = vertex + origin
}
@group(1) @binding(0) var<uniform> cell: GxCell;
// The per-ITEM record table, indexed by the vertex word's low bits:
// x = texture-array layer, y = the WMO authored batch order (the clip-z nudge; 0 on cells —
// an exact no-op), z = the MOMT SIDN colour packed r|g<<8|b<<16 (gamma bytes), w = flags:
// bit 0 the exile kill bit (B2), bits 1..=13 the interior-prop probe slot (B4), bit 14 the
// per-frame INTERIOR FOG lane (1787 — the client's per-group `[0xca7f00]`).
@group(1) @binding(1) var<storage, read> recs: array<vec4<u32>>;
@group(1) @binding(2) var tex_array: texture_2d_array<f32>;
// Repeat and clamp variants of the BLP model-albedo sampler (linear tri-filtered, aniso 8 —
// exact parity with the entity path's per-image samplers; render.rs owns the rationale).
@group(1) @binding(3) var samp_repeat: sampler;
@group(1) @binding(4) var samp_clamp: sampler;

// Vertex word bits — keep in sync with static_gx/mod.rs WORD_*.
const WORD_WRAP_X: u32 = 65536u;    // 1 << 16
const WORD_WRAP_Y: u32 = 131072u;   // 1 << 17
const WORD_UNLIT: u32 = 262144u;    // 1 << 18
const WORD_FOG_OFF: u32 = 524288u;  // 1 << 19
const WORD_SHADE_LIT: u32 = 1048576u; // 1 << 20 — no producer since 2050 (entities never divert here); kept, retiring it is a word-layout round
const WORD_TEXTURED: u32 = 2097152u;  // 1 << 21
// The WMO lane (slice 2) — the entity path's per-material facts as bits:
const WORD_WMO: u32 = 4194304u;        // 1 << 22 — model_flags.x (the WMO surface laws)
const WORD_INTERIOR: u32 = 8388608u;   // 1 << 23 — model_flags.z (interior group)
const WORD_CLASS_INT: u32 = 16777216u; // 1 << 24 — tint.w == 1 (INT batch)
const WORD_CLASS_TRANS: u32 = 33554432u; // 1 << 25 — tint.w == 2 (TRANS batch)
const WORD_WINDOW: u32 = 67108864u;    // 1 << 26 — sidn.w (WINDOW midpoint light)
const WORD_HAS_VC: u32 = 134217728u;   // 1 << 27 — the batch AUTHORED vertex colours
                                       //           (the entity shader's VERTEX_COLORS def)
// The prop lane (B4). INTERIOR without WMO = an interior M2 prop — the entity shader's own
// `interior_prop = flags.z && !flags.x` split: probe lighting, interior fog, zero live
// point lights.
const WORD_MATTE: u32 = 268435456u;    // 1 << 28 — EITHER doodad class (ADT MDDF or exterior MODD prop): intensity FIXED 1.0 (2050)

// Vanilla cutout ref (224/255) — wow_model.wgsl's VANILLA_ALPHA_KEY.
const VANILLA_ALPHA_KEY: f32 = 0.8784314;

// ---- mirrored law (wow_model.wgsl) ----

fn wow_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    return select(vec3<f32>(0.0), normalize(v), l2 > 1e-12);
}

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

// ---- the pass ----

struct GxVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) word: u32,
    @location(4) anchor: vec3<f32>,
    // MOCV / the baked constant tint (white where the batch authors none — WORD_HAS_VC says
    // which; the entity path's ATTRIBUTE_COLOR, interpolated exactly like it).
    @location(5) color: vec4<f32>,
}

struct GxVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) word: u32,
    @location(4) point_lit: vec3<f32>,
    @location(5) color: vec4<f32>,
    // fork: modern lighting (FORK.md, P3) — the vertex's point-light selection (flat).
    @location(6) @interpolate(flat) point_sel: vec3<u32>,
}

@vertex
fn vertex(v: GxVertex) -> GxVsOut {
    var out: GxVsOut;
    // The exile kill bit (B2, 1431 — record column w, bit 0): a punched-out item's vertices
    // all collapse to one constant point, so every triangle is zero-area and the rasterizer
    // drops it — no fragments, no state change, one u32 write flipped it. Triangles never
    // span items (the bake is item-contiguous), so a partial collapse cannot exist.
    if ((recs[v.word & 0xffffu].w & 1u) != 0u) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.world_position = vec4<f32>(0.0);
        out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
        out.uv = vec2<f32>(0.0);
        out.word = v.word;
        out.point_lit = vec3<f32>(0.0);
        out.color = vec4<f32>(1.0);
        return out;
    }
    // Camera-relative clip (0974/wow_model.wgsl): both addends are small — the recentred
    // vertex and (cell origin − camera).
    let p_cam = v.position + (cell.origin.xyz - view.world_position);
    let world = v.position + cell.origin.xyz;
    out.world_position = vec4<f32>(world, 1.0);
    let view_rot = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    );
    out.position = view.clip_from_view * vec4<f32>(view_rot * p_cam, 1.0);
    // WMO authored batch order (the per-item record's y; wow_model.wgsl's `m.sun_scale.y`
    // nudge verbatim): a later coplanar MOBA batch must WIN the reverse-Z GreaterEqual test
    // in any draw order — load-bearing here, where the (bucket, texture, group) sort erases
    // authored order by design. 0 on every cell item ⇒ ×1.0, an exact no-op.
    out.position.z *= 1.0 + f32(recs[v.word & 0xffffu].y) * 1.1920929e-7;
    out.world_normal = v.normal;
    out.uv = v.uv;
    out.word = v.word;
    out.color = v.color;
    // The ≤3-nearest FFP selection, anchored at the PLACEMENT origin exactly like the entity
    // path (the baked per-vertex anchor — 1429's parity note; the blob path coarsened this).
    // WMO surfaces take ZERO point lights — the entity path zeroes them in the vertex stage
    // (wow-re trace-forensics-abbey-interior-d3d §2: zero on every observed WMO surface) —
    // and so do interior M2 props (B4): their group-MOLR point lobes are folded into the
    // per-item SH probe at spawn, the entity path's own vertex-stage zeroing.
    if ((v.word & (WORD_WMO | WORD_INTERIOR)) != 0u) {
        out.point_lit = vec3<f32>(0.0);
        // …and the per-pixel selection with it. It used to be left unwritten, which was harmless
        // only because the grid overwrote the fragment's answer on every modern frame. Now that
        // the fragment keeps this selection outside the grid, an unwritten output would be an
        // undefined one — and index 0 is `point_light_eval`'s own "no light", so spelling it is
        // both the defined value and the reference's answer for these lanes.
        out.point_sel = vec3<u32>(0u);
    } else {
        out.point_lit = point_light_sum(world, v.normal, v.anchor);
        out.point_sel = point_light_select(v.anchor);
    }
    return out;
}

@fragment
fn fragment(in: GxVsOut) -> @location(0) vec4<f32> {
    // The hard farclip wall — per-pixel planar eye-Z, same plane as the entity path.
    let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
    if (wow_light.fog_params.w > 0.0 && eye_z > wow_light.fog_params.w) {
        discard;
    }
    // Sample: the item's array layer through the sampler matching its wrap flags. Both-repeat
    // and both-clamp ride real samplers (exact parity); the rare MIXED batch keeps the repeat
    // sampler plus the half-texel inset clamp on its clamped axis (0763's silhouette clamp,
    // per axis). The derivatives are taken unconditionally (see below) and the select
    // is per fragment.
    var base = vec4<f32>(1.0);
    let wrap_x = (in.word & WORD_WRAP_X) != 0u;
    let wrap_y = (in.word & WORD_WRAP_Y) != 0u;
    // The samples below sit inside a per-fragment branch (`WORD_TEXTURED` is per vertex), where
    // WGSL's uniformity rule forbids implicit derivatives — naga lets it through, the browser's
    // compiler does not. So the derivatives are taken HERE, in uniform flow, and the samples use
    // `textureSampleGrad`; `mip_bias` folds in as a scale of 2^bias on the footprint, which is
    // exactly what the bias form does to the LOD.
    let grad_scale = exp2(view.mip_bias);
    let uv_dx = dpdx(in.uv) * grad_scale;
    let uv_dy = dpdy(in.uv) * grad_scale;
    if ((in.word & WORD_TEXTURED) != 0u) {
        let layer = i32(recs[in.word & 0xffffu].x);
        let dims = vec2<f32>(textureDimensions(tex_array).xy);
        let inset = 0.5 / dims;
        var uv_mixed = in.uv;
        if (!wrap_x) {
            uv_mixed.x = clamp(uv_mixed.x, inset.x, 1.0 - inset.x);
        }
        if (!wrap_y) {
            uv_mixed.y = clamp(uv_mixed.y, inset.y, 1.0 - inset.y);
        }
        // `view.mip_bias` (decision 1639) — the render-scale LOD compensation, 0.0 at native.
        // Carried here as well as in wow_model.wgsl because this lane draws the same statics
        // through its own sampler rather than through `pbr_input_from_standard_material`, which
        // applies the bias for free; a lane that skipped it would blur out of step with the
        // entity path drawing the identical art.
        let c_repeat = textureSampleGrad(tex_array, samp_repeat, in.uv, layer, uv_dx, uv_dy);
        let c_clamp = textureSampleGrad(tex_array, samp_clamp, in.uv, layer, uv_dx, uv_dy);
        let c_mixed = textureSampleGrad(tex_array, samp_repeat, uv_mixed, layer, uv_dx, uv_dy);
        if (wrap_x && wrap_y) {
            base = c_repeat;
        } else if (!wrap_x && !wrap_y) {
            base = c_clamp;
        } else {
            base = c_mixed;
        }
    }
#ifdef GX_CUTOUT
    if (base.a < VANILLA_ALPHA_KEY) {
        discard;
    }
#endif
    // Lighting — the exterior-doodad lane verbatim (wow_model.wgsl steps 7+, 0803's lobe):
    // both faces light from the SAME submitted normal (the FFP never enables
    // GL_LIGHT_MODEL_TWO_SIDE). The entity shader's `select(-n, n, is_front)` exists to UNDO
    // bevy_pbr's double-sided back-face negation; this pipeline never negated, so the raw
    // baked normal IS the submitted normal and any select here would INTRODUCE the flip the
    // entity path removes (the first capture A/B lit every back-facing canopy card from the
    // wrong side — the felwood heatmap's bright clusters).
    let n_lit = wow_normalize(in.world_normal);
    // fork: modern lighting — the water mirror takes none of the modern lanes
    // (`sunshadow::mirror_view`): no cascades, cloud shadow, probe field, sky map or light grid — the
    // plain ambient and N·L, and an interior keeps its bake. See `terrain.wgsl` for why.
#ifdef WOW_MODERN
    let modern_lite = sunshadow::mirror_view(modern_block.cascades, view.world_position);
#else
    let modern_lite = false;
#endif
    // fork: modern lighting (FORK.md, P3) — per-pixel point lights in a modern frame.
    // Was a `select`, which in WGSL evaluates BOTH arms: a classic pipeline ran the per-pixel
    // selection loop and then threw it away.
    var point_lit_px = in.point_lit;
#ifdef WOW_MODERN
    point_lit_px = point_light_eval(in.point_sel, in.world_position.xyz, n_lit);
    // fork: modern lighting (FORK.md, P6) — the binned local-light grid replaces
    // that three-nearest selection where it is live, and reaches the WMO and interior-prop lanes,
    // whose vertex stage deliberately zeroes the selection (the reference commits no point light to
    // a WMO surface). It has to: P6 retires the bake that used to be those rooms' light.
    if (locallight::grid_live(modern_block.lights) && !modern_lite) {
        point_lit_px = modern_local_lights(
            in.world_position.xyz,
            n_lit,
            (in.word & WORD_INTERIOR) != 0u,
            modern_lite,
        );
    }
#endif
    // fork: modern lighting (P6) — interiors run the modern lanes whenever the probe field is live,
    // which is the plan's tier split: Low traces no probes and keeps the bake and `prop_probes`.
#ifdef WOW_MODERN
    let modern_interior = modern_block.gi.params.x > 0.5 && !modern_lite;
#else
    let modern_interior = false;
#endif
    var lit_modern_room = vec3<f32>(0.0);
#ifdef WOW_MODERN
    if (modern_interior && (in.word & WORD_INTERIOR) != 0u) {
        let room_gi = modern_gi_ambient(in.world_position.xyz, n_lit, 1.0, modern_lite);
        let room = max(modern_block.gi.interior.rgb, wow_light.light_ambient.rgb * MODERN_ROOM_FLOOR);
        lit_modern_room = mix(room, room_gi.rgb, room_gi.w);
    }
#endif
    let L = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(n_lit, L), 0.0);
    let lit_nl = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    // The retained path uses the same Bevy directional shadow map as terrain and entity models.
    // `static_gx` used to have no mesh-view shadow bindings, which made every Stormwind WMO act
    // as if shadows were disabled even though the caster map was populated.
    var world_shadow = 1.0;
    if (lights.n_directional_lights > 0u && (in.word & WORD_INTERIOR) == 0u) {
        let view_z = (view.view_from_world * in.world_position).z;
        for (var light_id = 0u; light_id < lights.n_directional_lights; light_id = light_id + 1u) {
            if ((lights.directional_lights[light_id].flags & 1u) != 0u) {
                world_shadow = shadows::fetch_directional_shadow(
                    light_id,
                    in.world_position,
                    n_lit,
                    view_z,
                );
                break;
            }
        }
    }
    // fork: modern lighting (FORK.md, P2) — the modern sun's own cascades.
#ifdef WOW_MODERN
    if ((in.word & WORD_INTERIOR) == 0u && !modern_lite) {
        let mview_z = (view.view_from_world * in.world_position).z;
        world_shadow = sunshadow::sun_visibility(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz, n_lit, -mview_z, modern_lite);
        // fork: modern lighting (FORK.md, P4) — cloud shadows.
        let cloud_ct = sunshadow::cloud_taps(modern_block.fog.cloud, in.world_position.xyz);
        world_shadow = world_shadow * sunshadow::cloud_mix(cloud_ct, modern_block.cloud[cloud_ct.idx.x], modern_block.cloud[cloud_ct.idx.y], modern_block.cloud[cloud_ct.idx.z], modern_block.cloud[cloud_ct.idx.w]);
    }
    let shadow_term = world_shadow;
#else
    let shadow_term = mix(SHADOW_SUN_FLOOR, 1.0, world_shadow);
#endif
    // Keep ambient energy when the realtime map blocks the sun. The retained pass also carries
    // authored fixture/probe lighting, which must not be darkened by a directional shadow. The
    // `world_shadow < 0.999` arm keeps `worldShadows 0` (no shadow-mapped light) and a fully
    // sunlit fragment byte-identical: `ambient + (lit − ambient) × 1.0` is not guaranteed to
    // round back to `lit`.
    let lit_nl_shadowed_classic = select(
        lit_nl,
        clamp(
            wow_light.light_ambient.rgb + (lit_nl - wow_light.light_ambient.rgb) * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ),
        world_shadow < 0.999,
    );
    // fork: modern lighting (FORK.md, P3) — modern exterior: sun share under the cascades, ambient
    // under the sky-occlusion map.
    var lit_nl_shadowed = lit_nl_shadowed_classic;
    // The sky map and the probe field at this fragment, read ONCE: the exterior doodad lane below
    // wants the same two for its lobe (same point, same normal), and both lanes are evaluated before
    // `select` keeps one.
    var modern_sky = 1.0;
    var modern_gi = vec4<f32>(0.0);
#ifdef WOW_MODERN
    if ((in.word & WORD_INTERIOR) == 0u && !modern_lite) {
        let amb = wow_light.light_ambient.rgb;
        // fork: modern lighting (FORK.md, P5) — the probe field over the sky map.
        modern_gi = modern_gi_ambient(in.world_position.xyz, n_lit, 0.0, modern_lite);
        // The sky map only where the field does not answer whole (`sunshadow::probe_or_sky`); both
        // blends below take `modern_sky` through that same helper, so 1 is exact for both.
        if (modern_gi.w < 1.0) {
            modern_sky = sunshadow::sky_ambient(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz, modern_lite);
        }
        lit_nl_shadowed = clamp(sunshadow::shade_floor(sunshadow::probe_or_sky(amb, modern_sky, modern_gi), amb) + max(lit_nl - amb, vec3<f32>(0.0)) * world_shadow, vec3<f32>(0.0), vec3<f32>(1.0));
    }
#endif
    // The order-2 SH basis products over the fragment normal — shared by the exterior
    // doodad lobe and the interior-prop probe lane (wow_model.wgsl computes them once too).
    let quad = vec4<f32>(n_lit.x * n_lit.y, n_lit.y * n_lit.z, n_lit.z * n_lit.z, n_lit.x * n_lit.z);
    let x2y2 = n_lit.x * n_lit.x - n_lit.y * n_lit.y;
    // ATTRIBUTE_COLOR, pre-folded into the sampled base exactly where bevy_pbr folds it
    // (`base_color *= in.color` before the texture multiply — commutative, so tex×vc is the
    // same bits). **A colour-less batch takes the CONSTANT 1.0, never the interpolated
    // white**: the entity path's fold literally does not exist without its VERTEX_COLORS
    // shader-def, and GPU interpolation of a constant attribute is NOT exact — a barycentric
    // sum lands at 1.0±ε, and `base × 0.99999994` re-rounds ~half of all textured pixels one
    // byte down. That was the ±1/255 film over 300k pixels of house-south with the WMO lane
    // OFF — the attribution sweep's cells-only leg is what separated it from the WMO bake's
    // own (f64-fixed) quantization film.
    let has_vc = (in.word & WORD_HAS_VC) != 0u;
    let vc = select(vec4<f32>(1.0), in.color, has_vc);
    let folded = select(base.rgb, base.rgb * vc.rgb, has_vc);
    var rgb: vec3<f32>;
    if ((in.word & WORD_WMO) != 0u) {
        // ---- the WMO surface lanes (slice 2) — wow_model.wgsl's is_wmo branch verbatim ----
        let interior = (in.word & WORD_INTERIOR) != 0u;
        let class_int = (in.word & WORD_CLASS_INT) != 0u;
        let class_trans = (in.word & WORD_CLASS_TRANS) != 0u;
        let trans_a = vc.a; // 1.0 where no MOCV authored — the entity path's default
        // WINDOW (MOMT 0x20) — interior drawer only: GL_LIGHT0 swapped to the brighter
        // Direct/Ambient midpoint pair, ambient +16/255 saturating (0x6d37e0).
        let window_mid = 0.5 * (wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb);
        // The window's directional half rides the sun, so it takes the shadow term (×1.0 exact
        // when the caster is off — no identity guard needed on a pure multiply).
        let lit_window_shadowed = clamp(
            window_mid + vec3<f32>(16.0 / 255.0) + window_mid * ndotl * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let lit_int_base = select(lit_nl_shadowed, lit_window_shadowed, (in.word & WORD_WINDOW) != 0u);
        // The interior BATCH-CLASS lanes (trace-forensics-abbey-interior-d3d §2): INT = unlit
        // (the bake IS the room's light), TRANS = the per-vertex MOCV-alpha lit↔bake lerp,
        // EXT = plain lit_nl. Exterior groups take lit_nl at sun-scale 1 (prog 198/VS 151 —
        // no terrain shade, no SH lobe).
        var lit_wmo_interior = vec3<f32>(1.0);
        if (modern_interior) {
            // fork: modern lighting (P6) — every batch class takes the room's own light: INT is no
            // longer unlit, TRANS has no bake left to lerp toward, and the WINDOW swap is replaced
            // by the pane's SIDN emission below.
            lit_wmo_interior = lit_modern_room;
        } else if (class_trans) {
            lit_wmo_interior = mix(vec3<f32>(1.0), lit_int_base, trans_a);
        } else if (!class_int) {
            lit_wmo_interior = lit_int_base;
        }
        // EXTERIOR surfaces are the ones the realtime map can shadow (the fetch above is
        // gated on !interior, so `lit_nl_shadowed` collapses to `lit_nl` for interior
        // fragments anyway). Taking plain `lit_nl` here left every Stormwind street and
        // wall unshadowed while the caster map was fully populated.
        let lit_wmo = select(lit_nl_shadowed, lit_wmo_interior, interior);
        // SIDN night glow (MOMT 0x10): the authored emissive × the live night fraction, an
        // EMISSION term — inside the clamped sum, never MOCV-multiplied; dead on the unlit
        // INT lane, TRANS-weighted by the lit-pass alpha (wmo-interior-night-light §4).
        let rec = recs[in.word & 0xffffu];
        let sidn_rgb = vec3<f32>(
            f32(rec.z & 0xffu),
            f32((rec.z >> 8u) & 0xffu),
            f32((rec.z >> 16u) & 0xffu),
        ) / 255.0;
        var sidn_w = 1.0;
        if (interior) {
            if (class_trans) {
                sidn_w = trans_a;
            } else if (class_int) {
                sidn_w = 0.0;
            }
        }
        var sidn_e = sidn_rgb * (wow_light.grade.x * sidn_w);
        // fork: modern lighting (P6) — a modern window is an emitter, on every class and with a
        // daylight floor (`wow_model.wgsl` carries the same three numbers).
        if (modern_interior && interior) {
            sidn_e = sidn_rgb * (MODERN_SIDN_GAIN * mix(MODERN_SIDN_DAY, 1.0, wow_light.grade.x));
        }
        // GL_COLOR_MATERIAL: MOCV multiplies the lit terms INSIDE the clamp (point term is
        // zero on WMO surfaces, kept in the sum for exact mirroring), emission adds beside.
        // fork: modern lighting (FORK.md, P3/P6) — a modern WMO surface drops its
        // baked MOCV light (exterior since P3, interior since P6); MOCV's ALPHA survives indoors as
        // the emissive mask it was authored as, added unclamped so a hearth reaches the bloom.
        // The classic sum is left term for term as upstream wrote it (float addition does not
        // associate, and the classic frame is held byte-identical); the modern interior is its own
        // statement, with the emission OUTSIDE the clamp so a hearth reaches the bloom.
        let drop_mocv = MODERN_FRAME && (!interior || modern_interior);
        var primary = clamp(
            select(vc.rgb, vec3<f32>(1.0), drop_mocv) * (lit_wmo + point_lit_px) + sidn_e,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        if (modern_interior && interior) {
            var emissive = sidn_e;
            if (class_int && has_vc) {
                emissive = emissive + vec3<f32>(MODERN_MOCV_EMISSIVE * trans_a);
            }
            let lit_sum = select(vc.rgb, vec3<f32>(1.0), drop_mocv) * (lit_wmo + point_lit_px);
            primary = clamp(lit_sum, vec3<f32>(0.0), vec3<f32>(1.0)) + emissive;
        }
        // The entity algebra EXACTLY: bevy pre-folds vc into base and the shader divides it
        // back out — (tex×vc)/max(vc, 1/255) is not tex in floats, so simplifying would
        // break the pixel-identity bar.
        let tex_rgb = folded / max(vc.rgb, vec3<f32>(1.0 / 255.0));
        rgb = tex_rgb * primary;
        if (interior && class_int && has_vc && !modern_interior) {
            // INT self-illumination: tex·MOCV·(1 + 4·MOCV.a), only the final [0,1] clamp —
            // the client's own interior pixel shader (literal 4.0). Gated on AUTHORED
            // colours: a colour-less INT batch takes the entity path's no-COLORS combine
            // (plain tex × lit), which the white-vc algebra above already reproduces.
            rgb = clamp(
                tex_rgb * vc.rgb * (1.0 + 4.0 * trans_a),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        }
    } else if ((in.word & WORD_INTERIOR) != 0u) {
        // ---- the interior M2-PROP lane (B4, decision 1433) — wow_model.wgsl's
        // interior-prop branch verbatim: the folded per-instance SH probe (MODD ambient +
        // the fixed-axis diffuse lobe + the owning group's MOLR lobes, folded ONCE at
        // spawn), evaluated over the fragment normal. The slot rides the record table
        // (w bits 1..14) where the entity path rode MeshTag bits 6..19. The soft SH wrap
        // is the reference's authored response — deliberately NOT a hard max(N·L,0).
        // point_lit is zero (vertex stage); inst_tint identity, no highlight, SIDN zero on
        // every M2 batch — the combine collapses to folded × clamp(probe eval).
        let probe = 7u * ((recs[in.word & 0xffffu].w >> 1u) & 0x1fffu);
        let n1 = vec4<f32>(n_lit, 1.0);
        let lit_prop = clamp(
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
        // fork: modern lighting (FORK.md, P6) — an interior MODD prop reads the
        // room the way its walls do; the folded `prop_probes` slot survives on Low alone.
        let lit_prop_room = select(lit_prop, lit_modern_room, modern_interior);
        let primary = clamp(lit_prop_room + point_lit_px, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb = folded * primary;
    } else {
        // ---- the exterior ADT-doodad lane (slice 1) + the exterior MODD-prop family (B4) --
        // The intensity family: statics are `mat_shade` only (no per-instance ramp byte —
        // that is an entity concern): lit ground ⇒ mix(2.5,0.5,0)=2.5, MCSH-shadowed ⇒ 0.5,
        // and the Matte mid-band (an exterior WMO MODD prop) FIXED 1.0 — the 2.5 site is one
        // a MODD prop never reaches (§8b); then the recorded `min(I,1)` cap (unfaithful,
        // kept in exact sync — 0803 §3; today it makes Matte and Lit read identically, and
        // the distinct bit exists so lifting the cap cannot silently split this lane).
        // fork: modern lighting (FORK.md, P3) — the baked MCSH doodad shade retires in a modern frame.
        let shade_t = select(select(1.0, 0.0, (in.word & WORD_SHADE_LIT) != 0u), 0.0, MODERN_FRAME);
        let intensity = min(
            select(mix(2.5, 0.5, shade_t), 1.0, (in.word & WORD_MATTE) != 0u),
            1.0,
        );
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
        // The SH lobe's own ambient rows (the .w column) are the doodad lane's ambient share:
        // subtract-and-restore around the shadow term so only the directional part of the lobe
        // dims. Guarded like `lit_nl_shadowed` so the no-shadow state stays byte-identical.
        let sun_ambient = vec3<f32>(
            wow_light.sh_c10_r.w,
            wow_light.sh_c10_g.w,
            wow_light.sh_c10_b.w,
        );
        let lit_doodad_plain = clamp(sun_lobe, vec3<f32>(0.0), vec3<f32>(1.0));
        let lit_doodad_classic = select(
            lit_doodad_plain,
            clamp(
                sun_ambient + (sun_lobe - sun_ambient) * shadow_term,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            ),
            world_shadow < 0.999,
        );
        // fork: modern lighting (FORK.md, P3) — modern: the lobe's ambient under the sky map.
        var lit_doodad = lit_doodad_classic;
#ifdef WOW_MODERN
        {
            // fork: modern lighting (FORK.md, P5) — the doodad lobe's own ambient
            // share comes from the probes, scaled to keep the lobe's authored colour balance. This lane
            // is exterior by construction, so `modern_sky` / `modern_gi` were read above.
            let amb_gi = sunshadow::shade_floor(sunshadow::probe_or_sky(sun_ambient, modern_sky, modern_gi), sun_ambient);
            lit_doodad = clamp(amb_gi + max(sun_lobe - sun_ambient, vec3<f32>(0.0)) * world_shadow, vec3<f32>(0.0), vec3<f32>(1.0));
        }
#endif
        // Sun disabled (light_sun.w) falls back to the FFP matte, like the entity path.
        var lit = select(lit_nl_shadowed, lit_doodad, wow_light.light_sun.w > 0.5);
        // FFP combine: the light sum saturates FIRST, the texture (× the baked constant
        // tint, when authored) modulates the clamped result. Statics: inst_tint identity,
        // no highlight, no SIDN.
#ifdef GX_TWO_SIDED
#ifdef GX_CUTOUT
        // fork: modern lighting (FORK.md, P3) — foliage: wrap diffuse + back-lit
        // transmission under the cascades (see wow_model.wgsl).
#ifdef WOW_MODERN
        {
            let to_eye = normalize(view.world_position.xyz - in.world_position.xyz);
            let nl = dot(n_lit, L);
            let wrap = max((nl + 0.5) / 1.5, 0.0) - max(nl, 0.0);
            let back = pow(max(dot(to_eye, -L), 0.0), 6.0);
            lit = lit + wow_light.light_diffuse.rgb * world_shadow * (0.5 * max(wrap, 0.0) + 0.6 * back);
        }
#endif
#endif
#endif
        let primary = clamp(lit + point_lit_px, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb = folded * primary;
    }
    // Unlit fullbright (M2 UNLIT 0x01 / WMO UNLIT on an exterior-group batch): the albedo —
    // texture × the folded vertex colour — with lighting off; no emission terms reach it.
    if ((in.word & WORD_UNLIT) != 0u) {
        rgb = folded;
    }
    // Step-5 fog (planar eye-Z; policy: Scene, or Off via the render-flag bit — the
    // Black/White/Grey families belong to blends this lane excludes by admission). Interior
    // WMO surfaces fog with the INTERIOR triple — the room keeps its warm MFOG haze
    // (round-6 Q-I: one flag gates walls, pools and props alike).
    // The interior fog lane is the per-frame record bit, NOT the baked `WORD_INTERIOR`: the
    // client pushes the interior triple per GROUP under `[0xca7f00]` (`0x6b5190` for surfaces,
    // `0x6b62e0` for the group's doodads — wow-re round-6 Q-I), and the flood decides it each
    // frame (decision 1787, `wow_model.wgsl` carries the same law on its tag bit 30). Keyed on
    // the batch's static interior flag instead, every true-interior group of a building wore the
    // building's MFOG the moment the camera stood anywhere in it — B335.
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if ((recs[in.word & 0xffffu].w & 16384u) != 0u) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    // fork: modern lighting (FORK.md, P4) — a modern frame is fogged by the composite.
    if (fog_color.w > 0.5 && (in.word & WORD_FOG_OFF) == 0u && !MODERN_FRAME) {
        let denom = max(fog_span.y - fog_span.x, 0.001);
        let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
        rgb = mix(fog_color.xyz, rgb, factor);
    }
    // Raw gamma out (0161's lane); alpha pinned 1.0 — opaque-intent by admission.
    // fork: modern lighting (FORK.md) — the frame is linear.
#ifdef WOW_MODERN
    if (modern_block.cascades.params.w > 0.5) {
        let dbg = modern_block.cascades.params.w;
        if (dbg > 5.5) {
            // fork: P8 — mode 6: **cell occupancy**, the probe for "why is that lamp dark?".
            // The binning pass records how many lights REACHED each cell alongside how many it
            // could keep (`PER_CELL`), so this shows where the per-cell budget is the thing
            // deciding a light's fate rather than the light's own falloff:
            //   green  = the cell kept everything that reached it
            //   yellow = full, nothing dropped
            //   red    = over budget, and how far over
            //   blue   = OUTSIDE the grid — no cell, and so no local light from it at all
            // That last one is not a budget verdict but the absence of any, and it used to read
            // as cell 0's: `cell_at` answers `(0, 0)` off the grid and this took the `.x` without
            // the `.y`, painting the whole far world with whatever cell 0 happened to hold. It is
            // its own colour now, because the grid's edge is exactly the thing this probe is most
            // often asked about — a lamp is dark out there whatever its budget says.
            let dgrid = modern_block.lights;
            let dfound = locallight::cell_at(dgrid, in.world_position.xyz);
            if (dfound.y < 1u) {
                rgb = vec3<f32>(0.0, 0.2, 1.0);
            } else {
                let dcell = modern_block.light_cells[dfound.x];
                let kept = f32(locallight::cell_count(dcell));
                let reached = f32((dcell.x >> 8u) & 0xffu);
                let over = max(reached - kept, 0.0);
                rgb = select(
                    vec3<f32>(0.0, kept / 8.0, 0.0),
                    mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), min(over / 8.0, 1.0)),
                    over > 0.0,
                );
            }
        } else if (dbg > 4.5) {
            // fork: modern lighting (P6) — mode 5: what the local-light grid put on this fragment.
            rgb = modern_local_lights(
                in.world_position.xyz,
                n_lit,
                (in.word & WORD_INTERIOR) != 0u,
                false,
            );
        } else if (dbg > 2.5) {
            let gdbg = modern_gi_ambient(in.world_position.xyz, wow_normalize(in.world_normal), 0.0, false); // modern GI debug view
            rgb = select(gdbg.rgb, vec3<f32>(gdbg.w), dbg > 3.5);
        } else {
            rgb = vec3<f32>(select(world_shadow, sunshadow::sky_visibility(modern_shadow_map, modern_shadow_samp, modern_block.cascades, in.world_position.xyz), dbg > 1.5)); // modern shadow debug view
        }
    }
#endif
    return vec4<f32>(colorspace::modern_out(rgb, wow_light.point_count.y), 1.0);
}
