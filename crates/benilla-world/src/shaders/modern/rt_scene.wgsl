// fork: the ray-traced scene's hit data (`static_gx/render/rt_scene.rs` has the contract): one TLAS
// over a BLAS per static_gx region and per terrain cell, and what a hit needs to become a surface.
// Imported by the ray-traced lanes (probe GI, shadows, water reflections) on native only.
//
// An importer starts with `enable wgpu_ray_query;` itself (naga_oil keeps each module's enables)
// and passes `rt_scene::shader_defs(group)`: `RT_GROUP` is the bind group this library's bindings
// sit in, laid out by `rt_scene::layout()`.
enable wgpu_ray_query;

#define_import_path benilla_modern::rt_scene

// The scene: instances carry their entry's slot as the custom index.
@group(#{RT_GROUP}) @binding(0) var rt_tlas: acceleration_structure;
// Two rows per slot:
//   row 0: x kind | flags (bits 8..), y the region's table row (static), z first attribute row,
//          w first index (static: the gx arena; terrain: the terrain arena)
//   row 1: x geometry 0's first triangle, y geometry 1's, z the cutout geometry (2 = none),
//          w vertex count
@group(#{RT_GROUP}) @binding(1) var<storage, read> rt_instances: array<vec4<u32>>;
// static_gx's region table (`gx_bindless`): header rows, then one record per item.
@group(#{RT_GROUP}) @binding(2) var<storage, read> rt_gx_table: array<vec4<u32>>;
// static_gx's index arena: a region's indices, relative to its first vertex.
@group(#{RT_GROUP}) @binding(3) var<storage, read> rt_gx_indices: array<u32>;
// The terrain cells' indices, relative to each cell's first vertex.
@group(#{RT_GROUP}) @binding(4) var<storage, read> rt_terrain_indices: array<u32>;
// Per vertex, ATTRIBUTE_WORDS words: uv.x, uv.y (f32 bits), the normal (octahedral, snorm16x2),
// the static vertex word (0 on terrain), the vertex colour (unorm8x4; white where none).
@group(#{RT_GROUP}) @binding(5) var<storage, read> rt_attributes: array<u32>;
// static_gx's texture classes: pool class c at element c (`bindless::GX_CLASS_SLOTS`).
@group(#{RT_GROUP}) @binding(6) var rt_textures: binding_array<texture_2d_array<f32>, 1024>;
@group(#{RT_GROUP}) @binding(7) var rt_repeat: sampler;
@group(#{RT_GROUP}) @binding(8) var rt_clamp: sampler;

// Instance kinds (row 0 x, low byte) and the TLAS instance mask bit each carries (1 << kind).
const KIND_CELL: u32 = 0u;
const KIND_WMO: u32 = 1u;
const KIND_PROP: u32 = 2u;
const KIND_TERRAIN: u32 = 3u;
const MASK_CELL: u32 = 1u;
const MASK_WMO: u32 = 2u;
const MASK_PROP: u32 = 4u;
const MASK_TERRAIN: u32 = 8u;
const MASK_STATIC: u32 = 7u;
const MASK_ALL: u32 = 15u;

const NO_GEOMETRY: u32 = 2u;
const HEADER_ROWS: u32 = 3u;
const ATTRIBUTE_WORDS: u32 = 5u;
// The stand-in's instance rows (`rt_scene.rs` STUB_MARK): bound in place of the scene whenever the
// importer's lane is not live this frame.
const STUB_MARK: u32 = 0xffffffffu;

// Is the bound group the scene, live this frame (`RtScene::lane_bind`)? False on the stand-in, whose
// instance row 0 carries the mark; a real scene's row 0 is a slot's kind (or never written, 0). One
// uniform load: a receiver that declares the scene for the whole session takes its raster path when
// this is false, so the ray tracing switches at run time without a pipeline of its own.
fn live() -> bool {
    return rt_instances[0].x != STUB_MARK;
}

// static_gx's vertex word bits (`static_gx/mod.rs` WORD_*).
const WORD_WRAP_X: u32 = 65536u;
const WORD_WRAP_Y: u32 = 131072u;
const WORD_TEXTURED: u32 = 2097152u;

// The alpha-test cutout reference 224/255, as static_gx.wgsl's VANILLA_ALPHA_KEY.
const ALPHA_KEY: f32 = 0.8784314;
// The mip a candidate's alpha is tested at: the draw lane keys the full-size texel.
const ALPHA_LEVEL: f32 = 0.0;

struct Instance {
    kind: u32,
    row: u32,
    attribute_base: u32,
    index_base: u32,
    // Geometry 0's and geometry 1's first triangle in the instance's index range.
    triangle_base: vec2<u32>,
    cutout_geometry: u32,
    vertex_count: u32,
}

// The instance a hit's `instance_custom_data` names.
fn instance(custom: u32) -> Instance {
    let a = rt_instances[custom * 2u];
    let b = rt_instances[custom * 2u + 1u];
    return Instance(a.x & 0xffu, a.y, a.z, a.w, b.xy, b.z, b.w);
}

// The attribute rows of triangle `primitive` of geometry `geometry`.
fn triangle(inst: Instance, geometry: u32, primitive: u32) -> vec3<u32> {
    let tri = select(inst.triangle_base.x, inst.triangle_base.y, geometry == 1u) + primitive;
    let at = inst.index_base + tri * 3u;
    var i: vec3<u32>;
    if (inst.kind == KIND_TERRAIN) {
        i = vec3<u32>(rt_terrain_indices[at], rt_terrain_indices[at + 1u], rt_terrain_indices[at + 2u]);
    } else {
        i = vec3<u32>(rt_gx_indices[at], rt_gx_indices[at + 1u], rt_gx_indices[at + 2u]);
    }
    return i + vec3<u32>(inst.attribute_base);
}

fn vertex_uv(v: u32) -> vec2<f32> {
    let at = v * ATTRIBUTE_WORDS;
    return vec2<f32>(bitcast<f32>(rt_attributes[at]), bitcast<f32>(rt_attributes[at + 1u]));
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x = n.x + select(t, -t, n.x >= 0.0);
    n.y = n.y + select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn vertex_normal(v: u32) -> vec3<f32> {
    return oct_decode(unpack2x16snorm(rt_attributes[v * ATTRIBUTE_WORDS + 2u]));
}

fn vertex_word(v: u32) -> u32 {
    return rt_attributes[v * ATTRIBUTE_WORDS + 3u];
}

fn vertex_color(v: u32) -> vec4<f32> {
    return unpack4x8unorm(rt_attributes[v * ATTRIBUTE_WORDS + 4u]);
}

// A hit's corner weights from its barycentrics.
fn weights(barycentrics: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
}

// The record of the item a static vertex word names (`gx_bindless::rec`): x layer | class << 16,
// y batch order, z SIDN, w flags (bit 0 hidden: killed or its texture not resident).
fn record(row: u32, word: u32) -> vec4<u32> {
    return rt_gx_table[row + HEADER_ROWS + (word & 0xffffu)];
}

// A static item's texel at an explicit level, through the sampler its wrap bits pick (as
// static_gx.wgsl's `gx_sample`); white where the item is untextured.
fn texel(row: u32, word: u32, uv: vec2<f32>, level: f32) -> vec4<f32> {
    if ((word & WORD_TEXTURED) == 0u) {
        return vec4<f32>(1.0);
    }
    let r = record(row, word);
    let c = r.x >> 16u;
    let l = i32(r.x & 0xffffu);
    let wrap_x = (word & WORD_WRAP_X) != 0u;
    let wrap_y = (word & WORD_WRAP_Y) != 0u;
    if (wrap_x && wrap_y) {
        return textureSampleLevel(rt_textures[c], rt_repeat, uv, l, level);
    }
    if (!wrap_x && !wrap_y) {
        return textureSampleLevel(rt_textures[c], rt_clamp, uv, l, level);
    }
    let inset = 0.5 / vec2<f32>(textureDimensions(rt_textures[c]).xy);
    var uv_mixed = uv;
    if (!wrap_x) {
        uv_mixed.x = clamp(uv_mixed.x, inset.x, 1.0 - inset.x);
    }
    if (!wrap_y) {
        uv_mixed.y = clamp(uv_mixed.y, inset.y, 1.0 - inset.y);
    }
    return textureSampleLevel(rt_textures[c], rt_repeat, uv_mixed, l, level);
}

// Whether a candidate triangle stops the ray. Only non-opaque geometry reaches here: an item its
// record hides never does; a cutout item does where its texel reaches the alpha key, as the draw
// lane's discard keeps it.
fn candidate_blocks(c: RayIntersection) -> bool {
    let inst = instance(c.instance_custom_data);
    if (inst.kind == KIND_TERRAIN) {
        return true;
    }
    let v = triangle(inst, c.geometry_index, c.primitive_index);
    let word = vertex_word(v.x);
    if ((record(inst.row, word).w & 1u) != 0u) {
        return false;
    }
    if (c.geometry_index != inst.cutout_geometry || (word & WORD_TEXTURED) == 0u) {
        return true;
    }
    let w = weights(c.barycentrics);
    let uv = vertex_uv(v.x) * w.x + vertex_uv(v.y) * w.y + vertex_uv(v.z) * w.z;
    return texel(inst.row, word, uv, ALPHA_LEVEL).a >= ALPHA_KEY;
}

// The nearest surface along `dir` in [t_min, t_max] among the instances `mask` admits (MASK_*),
// cutout texels and hidden items let through. `kind == RAY_QUERY_INTERSECTION_NONE` on a miss.
fn trace(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32, mask: u32) -> RayIntersection {
    var rq: ray_query;
    rayQueryInitialize(&rq, rt_tlas, RayDesc(RAY_FLAG_NONE, mask, t_min, t_max, origin, dir));
    loop {
        if (!rayQueryProceed(&rq)) {
            break;
        }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_TRIANGLE && candidate_blocks(c)) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    return rayQueryGetCommittedIntersection(&rq);
}

// Whether anything the mask admits blocks the segment: the first blocking hit ends the query.
fn occluded(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32, mask: u32) -> bool {
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        rt_tlas,
        RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, mask, t_min, t_max, origin, dir),
    );
    loop {
        if (!rayQueryProceed(&rq)) {
            break;
        }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_TRIANGLE && candidate_blocks(c)) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    return rayQueryGetCommittedIntersection(&rq).kind != RAY_QUERY_INTERSECTION_NONE;
}

// A committed hit as a surface.
struct Surface {
    // KIND_*.
    kind: u32,
    // The static region's table row; 0 on terrain.
    row: u32,
    // The static item's vertex word (its record's index in the low 16 bits); 0 on terrain.
    word: u32,
    t: f32,
    position: vec3<f32>,
    // The interpolated vertex normal in world space, turned to face the ray's origin.
    normal: vec3<f32>,
    uv: vec2<f32>,
    // The vertex colour (MOCV or the baked tint); white where the vertex has none.
    color: vec4<f32>,
    front_face: bool,
}

// The surface of a committed hit (`kind != RAY_QUERY_INTERSECTION_NONE`) of the ray `trace` cast.
fn surface(origin: vec3<f32>, dir: vec3<f32>, hit: RayIntersection) -> Surface {
    let inst = instance(hit.instance_custom_data);
    let v = triangle(inst, hit.geometry_index, hit.primitive_index);
    let w = weights(hit.barycentrics);
    let local_n = vertex_normal(v.x) * w.x + vertex_normal(v.y) * w.y + vertex_normal(v.z) * w.z;
    // object_to_world is mat4x3: the product is the world-space vector.
    var n = normalize(hit.object_to_world * vec4<f32>(local_n, 0.0));
    if (dot(n, dir) > 0.0) {
        n = -n;
    }
    var s: Surface;
    s.kind = inst.kind;
    s.row = inst.row;
    s.word = vertex_word(v.x);
    s.t = hit.t;
    s.position = origin + dir * hit.t;
    s.normal = n;
    s.uv = vertex_uv(v.x) * w.x + vertex_uv(v.y) * w.y + vertex_uv(v.z) * w.z;
    s.color = vertex_color(v.x) * w.x + vertex_color(v.y) * w.y + vertex_color(v.z) * w.z;
    s.front_face = hit.front_face;
    return s;
}

// A static surface's texel at `level` (white on terrain, whose splat this library does not hold).
fn surface_texel(s: Surface, level: f32) -> vec4<f32> {
    if (s.kind == KIND_TERRAIN) {
        return vec4<f32>(1.0);
    }
    return texel(s.row, s.word, s.uv, level);
}
