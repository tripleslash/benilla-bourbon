// Modern lighting (FORK.md, P5): the probe blend — this frame's traced probes
// folded into the live field with hysteresis, written straight into the probe region of the shared
// modern storage block (`sunshadow::ModernBlock.probes`), which is what every lane samples.
//
// A probe the toroidal grid just scrolled in, or one that has never been traced, is replaced whole
// rather than blended: its slot held another world probe a frame ago.
#import benilla_modern::probegi

struct GiBlend {
    gi: probegi::GiParams,
    clip: probegi::GiClip,
    cam: vec4<f32>,
    fwd: vec4<f32>,
    rot_x: vec4<f32>,
    rot_y: vec4<f32>,
    rot_z: vec4<f32>,
    tune: vec4<f32>,
}

// A reseeded probe's validity share on its first trace.
const RESEED_VALIDITY: f32 = 0.2;

@group(0) @binding(0) var<uniform> blend: GiBlend;
@group(0) @binding(1) var<storage, read> scratch: array<probegi::GiProbe>;
@group(0) @binding(2) var<storage, read_write> probes: array<probegi::GiProbe>;

// The same work list the trace ran over (`gi_classify.wgsl`), dispatched the same way: only the
// probes traced this frame are folded in, and every one of them has a fresh scratch entry.
struct GiWork {
    count: u32,
    items: array<u32>,
}
@group(0) @binding(3) var<storage, read> work: GiWork;

@compute @workgroup_size(64)
fn blend_probes(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let g = blend.gi;
    if (g.params.x < 0.5) {
        return;
    }
    // Rows of workgroups, as the trace dispatches them (`gi_classify.wgsl`'s `GROUP_ROW`).
    let t = id.y * (groups.x * 64u) + id.x;
    // fork: no further than the classify pass's cap (`probegi::listed_cap`).
    if (t >= min(work.count, probegi::listed_cap(g))) {
        return;
    }
    let item = work.items[t];
    let idx = item & 0x7fffffffu;
    let per_cascade = u32(g.params.y);
    let fresh = scratch[idx];
    let cascade = probegi::logical_cascade(g, idx / per_cascade);
    let w = probegi::slot_probe(g, cascade, idx % per_cascade);

    let old = probes[idx];
    // 2 = replace (scrolled in, cold, or a cut); otherwise ease toward the new value. A large step
    // eases faster, so a sunrise or a door opening does not lag behind for seconds.
    //
    // The ease-off has a dead band and a floor measured against the open sky. 24 rays under a
    // rotating pattern are noisy, and in the dark (under a canopy, in a room) that noise is a large
    // RELATIVE change every update — which used to throw the history away and let each probe jump to
    // its latest noisy estimate every period: the field sparkled.
    var history = blend.fwd.w;
    // A probe off screen is traced every fourth period (`gi_classify.wgsl`), so each of its updates
    // eases as far as four would: the same time constant, on or off screen.
    if ((item & 0x80000000u) != 0u) {
        history = history * history;
        history = history * history;
    }
    let reseed = fresh.sh_y.w > 1.5 || old.depth.w < 0.5;
    if (reseed) {
        history = 0.0;
    } else {
        let before = dot(old.dc.xyz, vec3<f32>(0.3333));
        let after = dot(fresh.dc.xyz, vec3<f32>(0.3333));
        let scale = max(max(before, after), max(g.sky.w * 0.15, 1e-3));
        let change = abs(after - before) / scale;
        history = history * (1.0 - 0.6 * smoothstep(0.35, 0.9, change));
    }
    // Validity and class ease too (a little faster than the lobe), so a probe near a wall or inside
    // a canopy fades in and out of its neighbours' answers instead of popping.
    let soft = select(max(history, 0.5), 0.0, reseed);
    var out: probegi::GiProbe;
    // A reseeded probe (scrolled in, cold, or a cut) holds one noisy 24-ray trace: it enters at a
    // fraction of its validity and earns the rest over the next updates, while the neighbours and the
    // coarser cascade keep answering. Taking it at full weight at once made each newly scrolled cell
    // snap to that first estimate and then drift — patches of ground blinking as the player walked.
    let validity = select(mix(fresh.dc.w, old.dc.w, soft), fresh.dc.w * RESEED_VALIDITY, reseed);
    out.dc = vec4<f32>(mix(fresh.dc.xyz, old.dc.xyz, history), validity);
    out.sh_x = vec4<f32>(
        mix(fresh.sh_x.xyz, old.sh_x.xyz, history),
        mix(fresh.sh_x.w, old.sh_x.w, history),
    );
    out.sh_y = vec4<f32>(mix(fresh.sh_y.xyz, old.sh_y.xyz, history), probegi::key_lo(w));
    out.sh_z = vec4<f32>(mix(fresh.sh_z.xyz, old.sh_z.xyz, history), probegi::key_hi(w));
    // P6: the distance moments ease like the lobe; the class eases faster (a door opening still
    // reaches the receivers within a few updates) but never snaps on one noisy trace.
    out.depth = vec4<f32>(
        mix(fresh.depth.x, old.depth.x, history),
        mix(fresh.depth.y, old.depth.y, history),
        mix(fresh.depth.z, old.depth.z, min(soft, 0.6)),
        1.0,
    );
    probes[idx] = out;
}
