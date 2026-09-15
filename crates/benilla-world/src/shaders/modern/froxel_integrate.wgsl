// Modern lighting (FORK.md, P4): front-to-back integration of the froxel scatter
// volume into the volume every lane samples — rgb = in-scatter accumulated to each slice's far edge,
// a = transmittance to it.
//
// Energy-conserving per slice (Hillaire 2015): with sigma_s = sigma_t (albedo lives in the radiance),
// a slice of optical depth sigma * ds adds L * (1 - exp(-sigma * ds)) behind the transmittance so far.
#import benilla_modern::sunshadow
#import benilla_modern::froxelfog

@group(0) @binding(0) var<storage, read> modern_block: sunshadow::ModernBlock;
@group(0) @binding(1) var scatter_tex: texture_3d<f32>;
@group(0) @binding(2) var integrated_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = modern_block.fog;
    let dims = vec3<u32>(f.grid.xyz);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    // Path length per yard of planar depth along this column's view ray.
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / f.grid.xy;
    let vx = (uv.x * 2.0 - 1.0) * f.proj.y;
    let vy = (1.0 - uv.y * 2.0) * f.proj.z;
    let stretch = sqrt(1.0 + vx * vx + vy * vy);

    var acc = vec3<f32>(0.0);
    var tr = 1.0;
    var prev_d = 0.0;
    for (var k = 0u; k < dims.z; k = k + 1u) {
        let dk = froxelfog::depth_of_slice(f, f32(k + 1u) / f.grid.z);
        let ds = (dk - prev_d) * stretch;
        prev_d = dk;
        let texel = vec3<i32>(i32(id.x), i32(id.y), i32(k));
        let s = textureLoad(scatter_tex, texel, 0);
        let sigma = max(s.a, 1e-6);
        let t = exp(-sigma * ds);
        acc = acc + tr * (s.rgb / sigma) * (1.0 - t);
        tr = tr * t;
        textureStore(integrated_out, texel, vec4<f32>(acc, tr));
    }
}
