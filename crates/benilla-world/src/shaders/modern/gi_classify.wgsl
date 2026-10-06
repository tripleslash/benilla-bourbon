// Modern lighting (FORK.md, P5; compacted in P8b): which probes the trace takes
// this frame, as a packed work list.
//
// **Why a list, and not the trace testing each probe itself.** The trace used to dispatch one thread
// per probe and return early from every probe it was not due to trace. The schedule is
// `(idx * 7 + frame) % period == 0`, and consecutive probes cycle through every residue of it, so
// exactly one lane in `period` did any work: High's period of 8 left seven of every eight lanes of
// the trace idle, each wave running as long as its one live probe's rays took. A camera scroll was
// worse still — an x-slab of scrolled probes is one probe per `dims.x`, so one live lane per wave.
//
// So this pass asks the question once per probe — scrolled in, never traced, a cut, or scheduled —
// and appends every probe that says yes to `work.items`. The trace (one lane per ray of each listed
// probe) and the blend (one thread per listed probe) then run through an indirect dispatch, so
// nothing they launch is idle for want of a due probe. Nothing about any probe's result changes: it
// is traced by the same code with the same rays, and each probe is owned by one workgroup, so the
// order the atomics hand out slots in is invisible.
#import benilla_modern::probegi

// The trace's parameters, as `gi_trace.wgsl` declares them (the layout is `gi::trace_bytes`).
struct GiTrace {
    gi: probegi::GiParams,
    clip: probegi::GiClip,
    cam: vec4<f32>,
    fwd: vec4<f32>,
    rot_x: vec4<f32>,
    rot_y: vec4<f32>,
    rot_z: vec4<f32>,
    tune: vec4<f32>,
}

// The work list: how many probes are listed, then their indices. Reset to zero by the CPU every
// frame (`gi::node::prepare_gi`).
struct GiWork {
    count: atomic<u32>,
    items: array<u32>,
}

// The trace's and the blend's indirect dispatch, laid out in ROWS of `GROUP_ROW` workgroups of 64
// lanes: `x` is a row's length (set by the CPU), `y` the rows, grown by one with the first workgroup of each
// row. Rows because one axis of a dispatch stops at 65535 workgroups, and a frame that lists every
// probe (the first one, a cut — every probe cold) at one probe a workgroup is 65536 of them in Ultra's
// field alone: a single-axis indirect dispatch of that size is simply dropped, the probes stay cold,
// and the next frame lists them all again. Reset to (GROUP_ROW, 0, 1) by the CPU every frame.
const GROUP_ROW: u32 = 256u;
struct GiDispatch {
    x: u32,
    y: atomic<u32>,
    z: u32,
}

// The probe region of the shared block alone (the blend pass binds the same range writable).
@group(0) @binding(0) var<storage, read> probes: array<probegi::GiProbe>;
@group(0) @binding(1) var<uniform> trace: GiTrace;
@group(0) @binding(2) var<storage, read_write> work: GiWork;
@group(0) @binding(3) var<storage, read_write> dispatch: GiDispatch;

@compute @workgroup_size(64)
fn classify_probes(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = trace.gi;
    if (g.params.x < 0.5) {
        return;
    }
    let per_cascade = u32(g.params.y);
    let total = per_cascade * u32(g.dims.w);
    let idx = id.x;
    if (idx >= total) {
        return;
    }
    let cascade = probegi::logical_cascade(g, idx / per_cascade);
    let w = probegi::slot_probe(g, cascade, idx % per_cascade);
    let previous = probes[idx];
    // The trace's own four reasons, verbatim (`gi_trace.wgsl` documents each).
    let scrolled = previous.sh_y.w != probegi::key_lo(w) || previous.sh_z.w != probegi::key_hi(w);
    let cold = previous.depth.w < 0.5;
    let period = max(u32(g.misc.y), 1u);
    // fork: a probe no receiver on screen can read keeps a quarter of the rate. The field is an
    // ease with a 19.8 s time constant (`gi::SETTLE_SECS`), so one updated every four periods is
    // a few seconds behind it when the camera turns onto it, and on its full schedule again from
    // then; a cut is never throttled. "On screen" is the view cone (`trace.rot_x.w`, its
    // half-diagonal) widened by two probe spacings — the eight probes around any receiver, and the
    // normal bias it samples along, are inside it — and every probe within those two spacings of
    // the camera. Four periods stay coprime with the 7 (every tier's period is 3·2^k·…), so the
    // schedule still visits every probe.
    let spacing = probegi::grid_row(g, cascade).w;
    let d = probegi::probe_pos(g, cascade, w) - trace.cam.xyz;
    let dist = length(d);
    let margin = 2.0 * spacing;
    var seen = trace.cam.w > 0.5 || dist <= margin;
    if (!seen) {
        let off_axis = acos(clamp(dot(d, trace.fwd.xyz) / dist, -1.0, 1.0));
        seen = off_axis <= trace.rot_x.w + asin(min(margin / dist, 1.0));
    }
    let scheduled = ((idx * 7u + u32(g.misc.z)) % select(4u * period, period, seen)) == 0u;
    // fork: a cut is no longer a reason to list a probe — it lists the whole field at once. The CPU
    // holds the cut for `period` frames instead (`gi::update_gi`), and the trace REPLACES every
    // probe it takes during that window, so each probe's scheduled turn in it is its reset.
    if (!scrolled && !cold && !scheduled) {
        return;
    }
    let slot = atomicAdd(&work.count, 1u);
    // fork: past the frame's cap (`probegi::listed_cap`) a probe waits: still cold or scrolled next
    // frame, so it is listed again, and the ones listed this frame are not.
    if (slot >= probegi::listed_cap(g)) {
        return;
    }
    // Bit 31 marks a throttled probe, whose blend eases four updates' worth at once.
    work.items[slot] = idx | select(0x80000000u, 0u, seen);
    // The trace spends one workgroup per `group_probes(rays)` listed probes — one lane per ray
    // (`gi_trace.wgsl`, whose arithmetic this repeats) — so every that-many-th probe listed adds a
    // workgroup, and every `GROUP_ROW`-th workgroup opens a new row of the dispatch. The blend reuses
    // the dispatch, which then launches at least one thread per probe.
    let rays = clamp(u32(g.misc.x), 1u, 64u);
    let per_group = clamp(64u / rays, 1u, 4u);
    if ((slot % per_group) == 0u && ((slot / per_group) % GROUP_ROW) == 0u) {
        atomicAdd(&dispatch.y, 1u);
    }
}
