// **The dynamic ripple displacement map** — the height field a wading body pushes into the water,
// ported from the reference's `PropagateDisplMap` / `PerturbDisplMap` pair.
//
// The reference runs four programs over a set of `proceduralDisplMap%d` render targets:
// `ClearDisplMap` zeroes one, `PerturbDisplMap` stamps `XTextures\Ripples\perturbTex` quads into it
// where bodies move, `PerturbDisplMapMesh` does the same for a mesh-shaped disturbance, and
// `PropagateDisplMap` integrates the wave. `ProcWaterAbove`'s samp4/samp8 permutations then read the
// result as a height field and fold its gradient into the surface normal.
//
// **This is all four in one pass**, and the collapse is not a shortcut — it removes a hazard. The
// reference stamps with textured quads because that is how a fixed-function pipeline adds a shape to
// a render target; a fragment shader that already knows its own world position can simply evaluate
// the same radial kernel and add it, with no second draw, no blend state, and no chance of the stamp
// pass and the propagate pass disagreeing about which frame they are in.
//
// ## The integration
//
// The map packs two frames of the simulation into one texel — `.x` the current height, `.y` the
// previous one — which is what lets a single texture carry the second-order wave equation:
//
//     h' = damp · edge · (½·(L + R + U + D) − h_prev)
//
// Verbatim from `PropagateDisplMap`, damping constant included (`0.97`, its `c0.z`). `edge` is that
// program's `saturate(16 · distance-to-border)`: waves are killed at the map's rim rather than
// allowed to reflect off it, because the rim is not a wall, it is the edge of what the client
// happens to be simulating.
//
// ## Reprojection
//
// The map covers a fixed square of world that FOLLOWS THE CAMERA, so every frame it has moved and
// last frame's texels are no longer over this frame's world positions. Each fragment therefore
// converts its own texel to a world point and samples the previous state where that point *was* —
// `prev` carries the previous placement for exactly this. Without it the field would smear across
// the ground as the player walked, which is also why the placement is snapped to whole texels on
// the way in (`ripple.rs`): a sub-texel origin would resample the whole field every frame and
// diffuse it into nothing.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

/// The most disturbances the field integrates in one step. **This is one ABI with `ripple.rs`'s
/// `MAX_STAMPS`, which sizes the buffer** — nothing checks the two, and WGSL clamps an out-of-range
/// constant index rather than faulting, so a shader smaller than the buffer does not fail, it
/// silently restamps its last slot for every disturbance past the end. Keep them equal.
const MAX_STAMPS: u32 = 48u;

struct Ripple {
    /// `xy` = the map's min corner in Bevy world XZ; `z` = 1 / extent in yards; `w` = the
    /// interpolation between the packed previous and current heights, for the water shader.
    placement: vec4<f32>,
    /// `x` = one texel in UV; `y` = the slope strength the water shader applies; `z` = enabled;
    /// `w` = the damping constant.
    params: vec4<f32>,
    /// The PREVIOUS frame's `placement`, for the reprojection above.
    prev: vec4<f32>,
    /// `x` = how many of `stamps` are live.
    stamp_count: vec4<f32>,
    /// `xy` = world XZ of the disturbance, `z` = its radius in yards, `w` = its amplitude.
    stamps: array<vec4<f32>, MAX_STAMPS>,
};

@group(0) @binding(0) var<storage, read> r: Ripple;
@group(0) @binding(1) var prev_map: texture_2d<f32>;
@group(0) @binding(2) var prev_samp: sampler;

/// The disturbance kernel — a radial falloff, evaluated rather than sampled.
///
/// The reference stamps a 64² texture (`perturbTex.blp`) for this. Decoded, that file is a plain
/// symmetric radial falloff: 255 at the centre, 0 at the rim, with the profile
/// `1.0, 0.976, 0.804, 0.514, 0.192, 0.031, 0.008, 0` at eighths of the radius. A smoothstep applied
/// twice reproduces it to within a couple of levels, so this evaluates the curve instead of shipping
/// a texture of it — no fetch, no blob, and the same shape.
fn kernel(dist: f32, radius: f32) -> f32 {
    let t = clamp(1.0 - dist / max(radius, 0.001), 0.0, 1.0);
    let a = t * t * (3.0 - 2.0 * t);
    return a * a * (3.0 - 2.0 * a);
}

/// The previous state at a UV, or zero outside the map.
///
/// Zero rather than the clamped edge texel: `ClampToEdge` would drag the border value inward and
/// every wave that reached the rim would pile up against a wall of its own last value.
fn prev_at(uv: vec2<f32>) -> vec2<f32> {
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec2<f32>(0.0);
    }
    return textureSampleLevel(prev_map, prev_samp, uv, 0.0).xy;
}

@fragment
fn fs_propagate(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel = r.params.x;
    let extent = 1.0 / max(r.placement.z, 1.0e-6);

    // This texel's world point, then where that point sat in the previous frame's map.
    let world = r.placement.xy + in.uv * extent;
    let puv = (world - r.prev.xy) * r.prev.z;

    // The four neighbours' CURRENT heights and this texel's PREVIOUS one — the two halves the
    // packed layout exists to carry.
    let sum = prev_at(puv + vec2<f32>(-texel, 0.0)).x
        + prev_at(puv + vec2<f32>(texel, 0.0)).x
        + prev_at(puv + vec2<f32>(0.0, -texel)).x
        + prev_at(puv + vec2<f32>(0.0, texel)).x;
    let centre = prev_at(puv);

    // `saturate(16 · distance to the nearest border)`, in THIS frame's UV — the fade belongs to the
    // map's own rim, not to wherever the previous map happened to be.
    let edge = clamp(
        min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y)) * 16.0,
        0.0,
        1.0,
    );
    var h = (sum * 0.5 - centre.y) * r.params.w * edge;

    // The disturbances, added straight into the new height. In world units, so a stamp does not
    // change size or shape as the map slides under it.
    let count = u32(r.stamp_count.x);
    for (var i = 0u; i < count; i = i + 1u) {
        let s = r.stamps[i];
        h += kernel(distance(world, s.xy), s.z) * s.w;
    }

    // `.x` the new height, `.y` what was current and is now previous. `.z`/`.w` are unread; they
    // carry 0 and 1 as the reference's own program does.
    return vec4<f32>(h, centre.x, 0.0, 1.0);
}
