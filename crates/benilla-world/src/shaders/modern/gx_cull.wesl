// fork: GPU cull (`static_gx/render/gpu_cull.rs`) — the two phases of static_gx's Hi-Z occlusion
// cull over one view's bucketed draws (`static_gx/render/batch.rs`, "The indirect layout").
//
// One thread per candidate draw. A draw's bound is its region's (region table rows `row + 1` and
// `row + 2`), or its selection key's (`row + 3 + items + 2 * key`) when the CPU named one.
// - `cull_early`: outside the current frustum, culled for good; else tested against last frame's
//   pyramid under last frame's matrix. Not occluded: written into its bucket (`args`, `counts`)
//   for the early prepass and the colour pass. Occluded: marked deferred.
// - `cull_late`: a deferred draw tested against this frame's pyramid, built from the early depth.
//   Not occluded: appended to its bucket (the colour pass draws it) and written into the late
//   buckets (`late_args`, `late_counts`), which the late prepass draws.
// Under `GX_MESHLETS` (`render/meshlets.rs`) a survivor also gets its task dispatch beside its
// args: one task workgroup per `TASK_MESHLETS` of its meshlets, and its candidate index, the late
// phase's in bit 31, for the task stage's meshlet cull.
// The tests themselves are `benilla_modern::gx_occlusion`'s.
import benilla_world::shaders::modern::gx_occlusion;

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

const NO_KEY: u32 = 0xffffu;
const HEADER_ROWS: u32 = 3u;
const DRAWN: u32 = 0u;
const DEFERRED: u32 = 1u;

@group(0) @binding(0) var<uniform> view: gx_occlusion::CullView;
@group(0) @binding(1) var<storage, read> table: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> cand: array<DrawArgs>;
// Per candidate: count slot, its bucket's first arg, selection key, phase-1 verdict.
@group(0) @binding(3) var<storage, read_write> draw_meta: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> args: array<DrawArgs>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> late_args: array<DrawArgs>;
@group(0) @binding(7) var<storage, read_write> late_counts: array<atomic<u32>>;
@group(0) @binding(8) var pyramid: texture_2d<f32>;
// `WOW_GX_PERF`: candidates, early survivors, late survivors.
@group(0) @binding(9) var<storage, read_write> stats: array<atomic<u32>, 4>;

@if(GX_MESHLETS) {
// `DispatchIndirectArgs`: 12 bytes, the stride `multi_draw_mesh_tasks_indirect_count` reads.
struct MeshArgs {
    x: u32,
    y: u32,
    z: u32,
}

// Threads of a task workgroup, one meshlet each (`meshlet::TASK_MESHLETS`).
const TASK_MESHLETS: u32 = 32u;
// A mesh ref's bit for a draw the late phase found (`meshlets.rs`, `LATE_DRAW`).
const LATE_DRAW: u32 = 0x80000000u;

// Per candidate: its first meshlet in the meshlet table, its meshlet count.
@group(0) @binding(10) var<storage, read> mesh_meta: array<vec2<u32>>;
// At each arg slot: the survivor's task dispatch and its candidate index (`LATE_DRAW` on).
@group(0) @binding(11) var<storage, read_write> mesh_args: array<MeshArgs>;
@group(0) @binding(12) var<storage, read_write> mesh_refs: array<u32>;
}

// The world box of region `row`, or of its selection key `key`.
fn bound(row: u32, key: u32) -> gx_occlusion::Bound {
    let head = table[row];
    var c = table[row + 1u];
    var h = table[row + 2u];
    if (key != NO_KEY) {
        let at = row + HEADER_ROWS + head.w + 2u * key;
        c = table[at];
        h = table[at + 1u];
    }
    return gx_occlusion::Bound(
        bitcast<vec3<f32>>(head.xyz) + bitcast<vec3<f32>>(c.xyz),
        bitcast<vec3<f32>>(h.xyz),
    );
}

fn occluded(clip_from_world: mat4x4<f32>, b: gx_occlusion::Bound) -> bool {
    return gx_occlusion::occluded(clip_from_world, b, pyramid, view.pyramid_size, view.mip_count);
}

// Append candidate `d` to its bucket; the count it bumps is the one the multi-draw reads.
fn draw_into_bucket(d: u32, a: DrawArgs, m: vec4<u32>, late: bool) {
    let slot = atomicAdd(&counts[m.x], 1u);
    args[m.y + slot] = a;
@if(GX_MESHLETS) {
    let groups = (mesh_meta[d].y + TASK_MESHLETS - 1u) / TASK_MESHLETS;
    mesh_args[m.y + slot] = MeshArgs(groups, 1u, 1u);
    mesh_refs[m.y + slot] = d | select(0u, LATE_DRAW, late);
}
}

@compute @workgroup_size(64, 1, 1)
fn cull_early(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= view.count) {
        return;
    }
    let d = view.args_start + id.x;
    let a = cand[d];
    let m = draw_meta[d];
    let b = bound(a.first_instance, m.z);
    var verdict = DRAWN;
    if (!gx_occlusion::outside(view.clip_from_world, b)) {
        if ((view.flags & gx_occlusion::PREV_VALID) != 0u && occluded(view.prev_clip_from_world, b)) {
            verdict = DEFERRED;
        } else {
            draw_into_bucket(d, a, m, false);
            if ((view.flags & gx_occlusion::STATS) != 0u) {
                atomicAdd(&stats[1], 1u);
            }
        }
    }
    draw_meta[d].w = verdict;
    if ((view.flags & gx_occlusion::STATS) != 0u) {
        atomicAdd(&stats[0], 1u);
    }
}

@compute @workgroup_size(64, 1, 1)
fn cull_late(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= view.count) {
        return;
    }
    let d = view.args_start + id.x;
    let m = draw_meta[d];
    if (m.w != DEFERRED) {
        return;
    }
    let a = cand[d];
    if (occluded(view.clip_from_world, bound(a.first_instance, m.z))) {
        return;
    }
    draw_into_bucket(d, a, m, true);
    let slot = atomicAdd(&late_counts[m.x], 1u);
    late_args[m.y + slot] = a;
    if ((view.flags & gx_occlusion::STATS) != 0u) {
        atomicAdd(&stats[2], 1u);
    }
}
