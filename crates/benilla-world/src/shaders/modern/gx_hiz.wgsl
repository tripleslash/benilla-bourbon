// fork: GPU cull (`static_gx/render/gpu_cull.rs`) — one level of a view's Hi-Z pyramid.
//
// Each texel keeps the farthest depth of the source texels it covers: depth is reverse-Z, so the
// farthest is the minimum. Level 0 is the previous power of two of the depth texture in each axis
// and reads the depth itself (`HIZ_FROM_DEPTH`, every sample with `HIZ_MULTISAMPLED`); every later
// level reads the one before it. A texel covers the source rectangle
// `[floor(x * S / D), ceil((x + 1) * S / D))`, so level 0 (ratio in [1, 2)) takes up to 3x3 texels
// and every later level exactly 2x2: no source texel is ever skipped, and the pyramid stays
// conservative.
//
// fork: G — `HIZ_LINEAR`: the levels above the water's linear view depth copy (`modern::rt_reflect`,
// yards, the sky at 1e9). The same minimum is then the NEAREST depth over the footprint, which is
// what a reflection march skips cells by; only the empty value differs.

#ifdef HIZ_FROM_DEPTH
#ifdef HIZ_MULTISAMPLED
@group(0) @binding(0) var src: texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var src: texture_depth_2d;
#endif
#else
@group(0) @binding(0) var src: texture_2d<f32>;
#endif
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

// What a texel reduces from: reverse-Z's far plane, or past any linear depth.
#ifdef HIZ_LINEAR
const HIZ_EMPTY: f32 = 3.0e38;
#else
const HIZ_EMPTY: f32 = 1.0;
#endif

// The farthest depth at one source texel.
fn load_src(p: vec2<u32>) -> f32 {
#ifdef HIZ_FROM_DEPTH
#ifdef HIZ_MULTISAMPLED
    var d = 1.0;
    let samples = textureNumSamples(src);
    for (var s = 0u; s < samples; s += 1u) {
        d = min(d, textureLoad(src, p, s));
    }
    return d;
#else
    return textureLoad(src, p, 0);
#endif
#else
    return textureLoad(src, p, 0).r;
#endif
}

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let out = textureDimensions(dst);
    if (id.x >= out.x || id.y >= out.y) {
        return;
    }
    let size = textureDimensions(src);
    let lo = (id.xy * size) / out;
    let hi = min(((id.xy + 1u) * size + out - 1u) / out, size);
    var d = HIZ_EMPTY;
    for (var y = lo.y; y < hi.y; y += 1u) {
        for (var x = lo.x; x < hi.x; x += 1u) {
            d = min(d, load_src(vec2<u32>(x, y)));
        }
    }
    textureStore(dst, id.xy, vec4<f32>(d, 0.0, 0.0, 0.0));
}
