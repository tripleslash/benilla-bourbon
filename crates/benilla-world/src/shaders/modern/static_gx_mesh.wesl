// fork: static_gx's mesh-shader lane (`static_gx/render/meshlets.rs`), native only, composed under
// `GX_MESHLETS` with `static_gx.wesl`'s defs: the task and mesh stages run static_gx's vertex
// bodies (`gx_vertex`, `gx_prepass_vertex`) per output vertex, and the fragments are static_gx's
// bodies behind this module's own entries. The enables are here alone: naga refuses either on a
// device without it, so `static_gx.wesl` never says them.

import benilla_world::shaders::static_gx::{
    GxVertex, GxPrepassVertex, GxVsOut, GxPrepassOut,
    gx_vertex, gx_prepass_vertex, gx_fragment, gx_prepass_fragment,
};
import benilla_world::shaders::modern::gx_occlusion;
import benilla_world::shaders::modern::gx_bindless;

enable wgpu_mesh_shader, draw_index;

// One multi-draw per bucket, one draw per survivor of C's draw cull (`gx_cull.wesl`): the task
// stage runs one workgroup per `TASK_MESHLETS` of the draw's meshlets, one thread a meshlet, and
// launches one mesh workgroup per meshlet it keeps; the mesh stage writes the meshlet's vertices
// through the vertex path and its triangles from the meshlet table.
//
// The task stage's modes (the immediates' `mode`), against the view's `drawn` bits, one per
// meshlet of the meshlet table:
// - EARLY (the early prepass, the draws C's early phase kept): a meshlet inside the frustum, not
//   back-facing by its cone and not occluded in last frame's pyramid (under last frame's matrix)
//   is drawn and its bit set; any other's bit is cleared.
// - LATE (the late prepass, over every draw of the bucket once C's late phase ran): an early
//   draw's meshlets with the bit clear, and every meshlet of a late draw, against this frame's
//   pyramid; the drawn ones' bits set, a late draw's others cleared.
// - COLOUR (the colour pass): the meshlets whose bit is set, which is exactly what the two
//   prepasses drew, so the colour pass's `Equal` depth test sees the same surfaces.

// Threads of a task workgroup (`meshlet::TASK_MESHLETS`) and of a mesh workgroup
// (`meshlet::MESH_THREADS`); the output arrays below hold `MESHLET_VERTICES` and
// `MESHLET_TRIANGLES`.
const GX_TASK_MESHLETS: u32 = 32u;
const GX_MESH_THREADS: u32 = 64u;

// The immediates' modes (`meshlets.rs`, `MODE_*`).
const GX_MODE_EARLY: u32 = 0u;
const GX_MODE_LATE: u32 = 1u;
const GX_MODE_COLOUR: u32 = 2u;

// A mesh ref's bit for a draw C's late phase found (`gx_cull.wesl`).
const GX_LATE_DRAW: u32 = 0x80000000u;

// Words of one static vertex in the slab (`render.rs`, `vertex_layout`: 64 bytes).
const GX_VERTEX_WORDS: u32 = 16u;

struct GxMeshDrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// 8 bytes: this bucket's first arg slot in the frame's args, the task stage's mode.
struct GxMeshImmediates {
    first_arg: u32,
    mode: u32,
}
var<immediate> gx_mesh_imm: GxMeshImmediates;

// `meshlet::TASK_PAYLOAD_BYTES` (144): the draw's region row and slab offset, the mode, then the
// kept meshlets.
struct GxTaskPayload {
    row: u32,
    base_vertex: i32,
    mode: u32,
    pad: u32,
    meshlets: array<u32, 32>,
}
var<task_payload> gx_payload: GxTaskPayload;

// The mesh group (`meshlets.rs`, `mesh_layout`).
@group(constants::GX_MESH_GROUP) @binding(0) var<uniform> gx_mesh_view: gx_occlusion::CullView;
// Two rows a meshlet: `[first vertex, first triangle, vertex count | triangle count << 8, cone]`
// (offsets into bindings 2 and 3) and `[mesh-local centre xyz (f32 bits), radius (f32 bits)]`.
@group(constants::GX_MESH_GROUP) @binding(1) var<storage, read> gx_meshlets: array<vec4<u32>>;
// Per meshlet vertex, its index in the region's mesh.
@group(constants::GX_MESH_GROUP) @binding(2) var<storage, read> gx_meshlet_vertices: array<u32>;
// Per triangle, its meshlet-local corners `a | b << 8 | c << 16`.
@group(constants::GX_MESH_GROUP) @binding(3) var<storage, read> gx_meshlet_triangles: array<u32>;
// The draw's mesh-allocator vertex slab, read as words.
@group(constants::GX_MESH_GROUP) @binding(4) var<storage, read> gx_slab: array<u32>;
@group(constants::GX_MESH_GROUP) @binding(5) var<storage, read> gx_mesh_cand: array<GxMeshDrawArgs>;
// Per candidate: its first meshlet, its meshlet count.
@group(constants::GX_MESH_GROUP) @binding(6) var<storage, read> gx_mesh_meta: array<vec2<u32>>;
// Per arg slot: the survivor's candidate index (`GX_LATE_DRAW` on for a late draw).
@group(constants::GX_MESH_GROUP) @binding(7) var<storage, read> gx_mesh_refs: array<u32>;
// One bit per meshlet of the table: drawn by this view's prepasses this frame.
@group(constants::GX_MESH_GROUP) @binding(8) var<storage, read_write> gx_drawn: array<atomic<u32>>;
@group(constants::GX_MESH_GROUP) @binding(9) var gx_mesh_pyramid: texture_2d<f32>;
// `WOW_GX_PERF`: meshlets examined, drawn early, drawn late.
@group(constants::GX_MESH_GROUP) @binding(10) var<storage, read_write> gx_mesh_stats: array<atomic<u32>, 4>;

var<workgroup> gx_task_kept: atomic<u32>;

// Meshlet `m` of the region at `origin` passes the frustum, its cone, and the pyramid: last
// frame's under last frame's matrix (`current` false, only where one was built), else this
// frame's under this frame's.
fn gx_meshlet_visible(m: u32, origin: vec3<f32>, current: bool) -> bool {
    let head = gx_meshlets[2u * m];
    let sphere = gx_meshlets[2u * m + 1u];
    let center = origin + bitcast<vec3<f32>>(sphere.xyz);
    let radius = bitcast<f32>(sphere.w);
    let b = gx_occlusion::Bound(center, vec3<f32>(radius));
    let v = gx_mesh_view;
    if (gx_occlusion::outside(v.clip_from_world, b)) {
        return false;
    }
    if ((v.flags & gx_occlusion::MIRRORED) == 0u
        && gx_occlusion::backfacing(head.w, center, radius, v.eye.xyz)) {
        return false;
    }
    if (current) {
        return !gx_occlusion::occluded(
            v.clip_from_world, b, gx_mesh_pyramid, v.pyramid_size, v.mip_count
        );
    }
    if ((v.flags & gx_occlusion::PREV_VALID) == 0u) {
        return true;
    }
    return !gx_occlusion::occluded(
        v.prev_clip_from_world, b, gx_mesh_pyramid, v.pyramid_size, v.mip_count
    );
}

fn gx_count(i: u32) {
    if ((gx_mesh_view.flags & gx_occlusion::STATS) != 0u) {
        atomicAdd(&gx_mesh_stats[i], 1u);
    }
}

// Whether this pass draws meshlet `m` of a draw (the header's modes), its bit kept current.
fn gx_meshlet_kept(m: u32, origin: vec3<f32>, mode: u32, late: bool) -> bool {
    let word = m >> 5u;
    let bit = 1u << (m & 31u);
    if (mode == GX_MODE_COLOUR) {
        return (atomicLoad(&gx_drawn[word]) & bit) != 0u;
    }
    if (mode == GX_MODE_EARLY) {
        gx_count(0u);
        if (gx_meshlet_visible(m, origin, false)) {
            atomicOr(&gx_drawn[word], bit);
            gx_count(1u);
            return true;
        }
        atomicAnd(&gx_drawn[word], ~bit);
        return false;
    }
    // LATE: an early draw retests only what its early pass did not draw.
    if (late) {
        gx_count(0u);
    } else if ((atomicLoad(&gx_drawn[word]) & bit) != 0u) {
        return false;
    }
    if (gx_meshlet_visible(m, origin, true)) {
        atomicOr(&gx_drawn[word], bit);
        gx_count(2u);
        return true;
    }
    if (late) {
        atomicAnd(&gx_drawn[word], ~bit);
    }
    return false;
}

@task @payload(gx_payload) @workgroup_size(32, 1, 1)
fn meshlet_task(
    @builtin(draw_index) draw: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) -> @builtin(mesh_task_size) vec3<u32> {
    if (lane == 0u) {
        atomicStore(&gx_task_kept, 0u);
    }
    workgroupBarrier();
    let r = gx_mesh_refs[gx_mesh_imm.first_arg + draw];
    let d = r & ~GX_LATE_DRAW;
    let span = gx_mesh_meta[d];
    let a = gx_mesh_cand[d];
    let mode = gx_mesh_imm.mode;
    if (lane == 0u) {
        gx_payload.row = a.first_instance;
        gx_payload.base_vertex = a.base_vertex;
        gx_payload.mode = mode;
    }
    let i = group.x * GX_TASK_MESHLETS + lane;
    if (i < span.y) {
        gx_bindless::enter(a.first_instance);
        let m = span.x + i;
        if (gx_meshlet_kept(m, gx_bindless::origin(), mode, (r & GX_LATE_DRAW) != 0u)) {
            gx_payload.meshlets[atomicAdd(&gx_task_kept, 1u)] = m;
        }
    }
    workgroupBarrier();
    return vec3<u32>(atomicLoad(&gx_task_kept), 1u, 1u);
}

// Vertex `local` of the region's mesh, from the slab at the draw's offset.
fn gx_slab_word(local: u32, word: u32) -> u32 {
    return gx_slab[(u32(gx_payload.base_vertex) + local) * GX_VERTEX_WORDS + word];
}

fn gx_slab_vec3(local: u32, word: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(vec3<u32>(
        gx_slab_word(local, word),
        gx_slab_word(local, word + 1u),
        gx_slab_word(local, word + 2u),
    ));
}

fn gx_slab_vec2(local: u32, word: u32) -> vec2<f32> {
    return bitcast<vec2<f32>>(vec2<u32>(gx_slab_word(local, word), gx_slab_word(local, word + 1u)));
}

// Meshlet triangle `t`'s corners.
fn gx_triangle(t: u32) -> vec3<u32> {
    let c = gx_meshlet_triangles[t];
    return vec3<u32>(c & 0xffu, (c >> 8u) & 0xffu, (c >> 16u) & 0xffu);
}

struct GxMeshPrimitive {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct GxMeshOutput {
    @builtin(vertices) vertices: array<GxVsOut, 64>,
    @builtin(primitives) primitives: array<GxMeshPrimitive, 124>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}
var<workgroup> gx_mesh_out: GxMeshOutput;

struct GxPrepassMeshOutput {
    @builtin(vertices) vertices: array<GxPrepassOut, 64>,
    @builtin(primitives) primitives: array<GxMeshPrimitive, 124>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}
var<workgroup> gx_prepass_out: GxPrepassMeshOutput;

// The colour pass's mesh stage: one workgroup per kept meshlet, one thread per vertex, the
// vertex path's `vertex` body per vertex (`gx_vertex`).
@mesh(gx_mesh_out) @payload(gx_payload) @workgroup_size(64, 1, 1)
fn meshlet_mesh(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let head = gx_meshlets[2u * gx_payload.meshlets[group.x]];
    let vertices = head.z & 0xffu;
    let triangles = head.z >> 8u;
    if (lane == 0u) {
        gx_mesh_out.vertex_count = vertices;
        gx_mesh_out.primitive_count = triangles;
    }
    if (lane < vertices) {
        let k = gx_meshlet_vertices[head.x + lane];
        var v: GxVertex;
        v.position = gx_slab_vec3(k, 0u);
        v.normal = gx_slab_vec3(k, 3u);
        v.uv = gx_slab_vec2(k, 6u);
        v.color = bitcast<vec4<f32>>(vec4<u32>(
            gx_slab_word(k, 8u),
            gx_slab_word(k, 9u),
            gx_slab_word(k, 10u),
            gx_slab_word(k, 11u),
        ));
        v.word = gx_slab_word(k, 12u);
        v.anchor = gx_slab_vec3(k, 13u);
        v.instance = gx_payload.row;
        gx_mesh_out.vertices[lane] = gx_vertex(v);
    }
    for (var t = lane; t < triangles; t += GX_MESH_THREADS) {
        gx_mesh_out.primitives[t].indices = gx_triangle(head.y + t);
    }
}

// The prepasses' mesh stage: the same meshlets through `prepass_vertex`'s body.
@mesh(gx_prepass_out) @payload(gx_payload) @workgroup_size(64, 1, 1)
fn meshlet_mesh_prepass(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let head = gx_meshlets[2u * gx_payload.meshlets[group.x]];
    let vertices = head.z & 0xffu;
    let triangles = head.z >> 8u;
    if (lane == 0u) {
        gx_prepass_out.vertex_count = vertices;
        gx_prepass_out.primitive_count = triangles;
    }
    if (lane < vertices) {
        let k = gx_meshlet_vertices[head.x + lane];
        var v: GxPrepassVertex;
        v.position = gx_slab_vec3(k, 0u);
        v.uv = gx_slab_vec2(k, 6u);
        v.word = gx_slab_word(k, 12u);
        v.anchor = gx_slab_vec3(k, 13u);
        v.instance = gx_payload.row;
        gx_prepass_out.vertices[lane] = gx_prepass_vertex(v);
    }
    for (var t = lane; t < triangles; t += GX_MESH_THREADS) {
        gx_prepass_out.primitives[t].indices = gx_triangle(head.y + t);
    }
}

// The fragment entries: static_gx's, whose module's own entries an import does not carry.
@fragment
fn fragment(in: GxVsOut) -> @location(0) vec4<f32> {
    return gx_fragment(in);
}

@fragment
fn prepass_fragment(in: GxPrepassOut) {
    gx_prepass_fragment(in);
}
