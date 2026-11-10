// fork: GPU cull (`static_gx/render/gpu_cull.rs`) — one level of a view's Hi-Z pyramid.
//
// Each texel keeps the farthest depth of the source texels it covers: depth is reverse-Z, so the
// farthest is the minimum. Level 0 is the previous power of two of the depth texture in each axis
// and reads the depth itself (`HIZ_FROM_DEPTH`, every sample with `HIZ_MULTISAMPLED`); every later
// level reads the one before it. A texel covers the source rectangle
// `[floor(x * S / D), ceil((x + 1) * S / D))`, so level 0 (ratio in [1, 2)) takes up to 3x3 texels
// and every later level exactly 2x2: no source texel is ever skipped, and the pyramid stays
// conservative.

@if(HIZ_FROM_DEPTH) {
@if(HIZ_MULTISAMPLED) {
@group(0) @binding(0) var src: texture_depth_multisampled_2d;
} @else {
@group(0) @binding(0) var src: texture_depth_2d;
}
} @else {
@group(0) @binding(0) var src: texture_2d<f32>;
}
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

// The farthest depth at one source texel.
fn load_src(p: vec2<u32>) -> f32 {
@if(HIZ_FROM_DEPTH) {
@if(HIZ_MULTISAMPLED) {
    var d = 1.0;
    let samples = textureNumSamples(src);
    for (var s = 0u; s < samples; s += 1u) {
        d = min(d, textureLoad(src, p, s));
    }
    return d;
} @else {
    return textureLoad(src, p, 0);
}
} @else {
    return textureLoad(src, p, 0).r;
}
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
    var d = 1.0;
    for (var y = lo.y; y < hi.y; y += 1u) {
        for (var x = lo.x; x < hi.x; x += 1u) {
            d = min(d, load_src(vec2<u32>(x, y)));
        }
    }
    textureStore(dst, id.xy, vec4<f32>(d, 0.0, 0.0, 0.0));
}
