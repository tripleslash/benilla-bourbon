// Modern lighting (FORK.md, P8b): clear part of a depth target by drawing over it.
//
// A render pass can only clear its WHOLE attachment, and the cube atlas now redraws one light's six
// tiles at a time rather than a layer's thirty-six (`static_gx::render::modern_shadow`). So a stale
// slot's tiles are cleared with this instead: one fullscreen triangle at depth 0 — reverse-Z's far
// plane, what a clear leaves and what the lookup reads as "nothing occludes" — drawn under a viewport
// and scissor that cover exactly one tile, with depth compare Always. Depth only, no fragment stage.

@vertex
fn fill_vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
}

// And to restore the RECTANGLES of a cascade last frame's entity casters could have written, from
// its static copy: one quad per caster (instanced, `rects`), each fragment writing back the cached
// depth at its own texel (depth compare Always; overlapping quads write the same value twice). A
// whole-layer copy was 16.8 MB a cascade, every frame an entity cast into it — which in game is every
// frame — and wgpu refuses a partial copy of a depth format, so a restore of just the dirty part has
// to be a draw.
@group(0) @binding(0) var cached: texture_depth_2d;
// Per caster: xy = the rectangle's low texel corner, zw = its high one (exclusive), rows down.
@group(0) @binding(1) var<storage, read> rects: array<vec4<f32>>;

@vertex
fn restore_vertex(
    @builtin(vertex_index) v: u32,
    @builtin(instance_index) k: u32,
) -> @builtin(position) vec4<f32> {
    // Two triangles, corners 0 1 2 / 2 1 3 — x from bit 0, y from bit 1.
    let corner = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[v];
    let r = rects[k];
    let texel = vec2<f32>(
        select(r.x, r.z, (corner & 1u) != 0u),
        select(r.y, r.w, (corner & 2u) != 0u),
    );
    let size = vec2<f32>(textureDimensions(cached));
    let uv = texel / size;
    return vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

@fragment
fn restore_fragment(@builtin(position) at: vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(cached, vec2<i32>(at.xy), 0);
}
