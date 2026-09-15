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

@group(0) @binding(0) var<uniform> blend: GiBlend;
@group(0) @binding(1) var<storage, read> scratch: array<probegi::GiProbe>;
@group(0) @binding(2) var<storage, read_write> probes: array<probegi::GiProbe>;

@compute @workgroup_size(64)
fn blend_probes(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = blend.gi;
    if (g.params.x < 0.5) {
        return;
    }
    let per_cascade = u32(g.params.y);
    let total = per_cascade * u32(g.dims.w);
    let idx = id.x;
    if (idx >= total) {
        return;
    }
    let fresh = scratch[idx];
    // 0 = this probe was not traced this frame.
    if (fresh.sh_y.w < 0.5) {
        return;
    }
    let cascade = idx / per_cascade;
    let slot = i32(idx % per_cascade);
    let n = vec3<i32>(g.dims.xyz);
    let texel = vec3<i32>(slot % n.x, (slot / n.x) % n.y, slot / (n.x * n.y));
    let base = vec3<i32>(g.grid[cascade].xyz);
    let w = base + vec3<i32>(
        probegi::wrap(texel.x - base.x, n.x),
        probegi::wrap(texel.y - base.y, n.y),
        probegi::wrap(texel.z - base.z, n.z),
    );

    let old = probes[idx];
    // 2 = replace (scrolled in, cold, or a cut); otherwise ease toward the new value. A large step
    // eases faster, so a sunrise or a door opening does not lag behind for seconds.
    var history = blend.fwd.w;
    if (fresh.sh_y.w > 1.5 || old.dc.w <= 0.0) {
        history = 0.0;
    } else {
        let before = max(dot(old.dc.xyz, vec3<f32>(0.3333)), 1e-4);
        let after = max(dot(fresh.dc.xyz, vec3<f32>(0.3333)), 1e-4);
        let change = abs(after - before) / max(before, after);
        history = history * (1.0 - clamp(change, 0.0, 1.0));
    }
    var out: probegi::GiProbe;
    out.dc = vec4<f32>(mix(fresh.dc.xyz, old.dc.xyz, history), fresh.dc.w);
    out.sh_x = vec4<f32>(
        mix(fresh.sh_x.xyz, old.sh_x.xyz, history),
        mix(fresh.sh_x.w, old.sh_x.w, history),
    );
    out.sh_y = vec4<f32>(mix(fresh.sh_y.xyz, old.sh_y.xyz, history), probegi::key_lo(w));
    out.sh_z = vec4<f32>(mix(fresh.sh_z.xyz, old.sh_z.xyz, history), probegi::key_hi(w));
    probes[idx] = out;
}
