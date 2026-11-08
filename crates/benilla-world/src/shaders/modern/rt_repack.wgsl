// fork: the ray-traced scene's attribute repack (`static_gx/render/rt_scene.rs`): one entry's
// vertices, copied raw out of the mesh slab into the scratch buffer, become its rows of the
// attribute arena `benilla_modern::rt_scene` reads hits through — ATTRIBUTE_WORDS words a vertex:
// uv.x, uv.y (f32 bits), the normal (octahedral, snorm16x2), the static vertex word, the vertex
// colour (unorm8x4).

struct Job {
    // The entry's first vertex in the scratch buffer, in words.
    src_word: u32,
    // The vertex stride, in words.
    stride: u32,
    count: u32,
    // The entry's first attribute row.
    dst: u32,
    // Each attribute's word offset in the vertex; NONE where the mesh has none.
    normal: u32,
    uv: u32,
    word: u32,
    color: u32,
}

@group(0) @binding(0) var<storage, read> scratch: array<u32>;
@group(0) @binding(1) var<storage, read_write> attributes: array<u32>;
@group(0) @binding(2) var<uniform> job: Job;

const NONE: u32 = 0xffffffffu;
const ATTRIBUTE_WORDS: u32 = 5u;
// Invocations a dispatch row holds: 65535 workgroups of 64.
const ROW: u32 = 4194240u;

fn sign_not_zero(v: vec2<f32>) -> vec2<f32> {
    return select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
}

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    if (n.z < 0.0) {
        return (vec2<f32>(1.0) - abs(p.yx)) * sign_not_zero(p);
    }
    return p;
}

fn load_f32(at: u32) -> f32 {
    return bitcast<f32>(scratch[at]);
}

@compute @workgroup_size(64)
fn repack(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = id.x + id.y * ROW;
    if (v >= job.count) {
        return;
    }
    let s = job.src_word + v * job.stride;
    var uv = vec2<u32>(0u);
    if (job.uv != NONE) {
        uv = vec2<u32>(scratch[s + job.uv], scratch[s + job.uv + 1u]);
    }
    var n = vec3<f32>(0.0, 1.0, 0.0);
    if (job.normal != NONE) {
        n = vec3<f32>(load_f32(s + job.normal), load_f32(s + job.normal + 1u), load_f32(s + job.normal + 2u));
    }
    var word = 0u;
    if (job.word != NONE) {
        word = scratch[s + job.word];
    }
    var color = 0xffffffffu;
    if (job.color != NONE) {
        color = pack4x8unorm(vec4<f32>(
            load_f32(s + job.color),
            load_f32(s + job.color + 1u),
            load_f32(s + job.color + 2u),
            load_f32(s + job.color + 3u),
        ));
    }
    let d = (job.dst + v) * ATTRIBUTE_WORDS;
    attributes[d] = uv.x;
    attributes[d + 1u] = uv.y;
    attributes[d + 2u] = pack2x16snorm(oct_encode(n));
    attributes[d + 3u] = word;
    attributes[d + 4u] = color;
}
