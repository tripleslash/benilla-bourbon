// fork: depth of field — the far-field blur of WoW Forever's three DoF pixel shaders (benilla's
// `depth_of_field` module names them), as one chain on the world view:
//
//   · fs_down  — full → half resolution. Each half-res texel averages only the four source texels
//                that lie past the focus start, and keeps the share of them that did in alpha. A
//                texel with none keeps its own colour at alpha 0. So a near silhouette never
//                leaks into the blurred background.
//   · fs_blur  — a 5×5 binomial kernel ([1 2 3 2 1]²) at ±1.2 and ±2.4 half-res texels, each tap
//                weighted by its alpha: only far colour is spread.
//   · fs_composite — `t = saturate((d − start) / (end − start)) · strength`, the blur partly
//                desaturated, and blended over the frame by `t` (the pipeline's alpha blend), so a
//                pixel nearer than `start` is left exactly as it was.
//
// `d` is the view-space distance to the depth buffer's surface, in yards; the sky (reverse-Z 0)
// is as far as anything gets. The frame's own encoding is kept: a classic frame blends gamma
// bytes as the reference blends its framebuffer, a modern frame blends linear radiance.

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_render::view::{View, position_ndc_to_view, uv_to_ndc};

@group(0) @binding(0) var<uniform> view: View;

// The main depth buffer, sample 0 under multisampling: a depth is a set of alternatives, not a
// quantity to average.
@if(DOF_MSAA) {
@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;
} @else {
@group(0) @binding(1) var depth_tex: texture_depth_2d;
}

// The pass input: the scene for fs_down, the masked half-res colour for fs_blur, the blur for
// fs_composite.
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var src_samp: sampler;

// focus: x = start (yd), y = end (yd), z = strength, w = desaturation.
// shape: x = the kernel's spread, in half-res texels per reference texel.
struct DofParams {
    focus: vec4<f32>,
    shape: vec4<f32>,
}
@group(0) @binding(4) var<uniform> params: DofParams;

// The distance in yards behind a depth-buffer texel; the infinite far plane reads as very far.
fn distance_at(px: vec2<i32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let clamped = clamp(px, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let depth = textureLoad(depth_tex, clamped, 0);
    if depth <= 0.0 {
        return 1.0e6;
    }
    let uv = (vec2<f32>(clamped) + vec2<f32>(0.5)) / dims;
    return length(position_ndc_to_view(vec3<f32>(uv_to_ndc(uv), depth), view.view_from_clip));
}

// The depth texel under a colour texel of a texture `color_dims` wide.
fn depth_px(color_px: vec2<i32>, color_dims: vec2<f32>) -> vec2<i32> {
    let scale = vec2<f32>(textureDimensions(depth_tex)) / color_dims;
    return vec2<i32>((vec2<f32>(color_px) + vec2<f32>(0.5)) * scale);
}

@fragment
fn fs_down(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let src_dims = vec2<f32>(textureDimensions(src_tex));
    let base = vec2<i32>(in.position.xy) * 2;
    let last = vec2<i32>(src_dims) - vec2<i32>(1);
    var sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var i = 0; i < 4; i++) {
        let px = min(base + vec2<i32>(i & 1, i >> 1), last);
        if distance_at(depth_px(px, src_dims)) > params.focus.x {
            sum += textureLoad(src_tex, px, 0).rgb;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return vec4<f32>(textureLoad(src_tex, min(base, last), 0).rgb, 0.0);
    }
    return vec4<f32>(sum / count, count * 0.25);
}

@fragment
fn fs_blur(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let uv = in.uv;
    let centre = textureSampleLevel(src_tex, src_samp, uv, 0.0);
    if centre.a == 0.0 {
        return vec4<f32>(centre.rgb, 0.0);
    }
    let offsets = array<f32, 5>(-2.4, -1.2, 0.0, 1.2, 2.4);
    let weights = array<f32, 5>(1.0, 2.0, 3.0, 2.0, 1.0);
    let step = texel * params.shape.x;
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var y = 0; y < 5; y++) {
        for (var x = 0; x < 5; x++) {
            let at = uv + vec2<f32>(offsets[x], offsets[y]) * step;
            let tap = textureSampleLevel(src_tex, src_samp, at, 0.0);
            let w = weights[x] * weights[y] * tap.a;
            sum += tap.rgb * w;
            total += w;
        }
    }
    return vec4<f32>(sum / max(total, 1.0e-4), 1.0);
}

@fragment
fn fs_composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let out_dims = vec2<f32>(textureDimensions(depth_tex));
    let d = distance_at(vec2<i32>(in.uv * out_dims));
    let start = params.focus.x;
    let ramp = max(params.focus.y - start, 1.0e-3);
    let t = saturate((d - start) / ramp) * params.focus.z;
    let blur = textureSampleLevel(src_tex, src_samp, in.uv, 0.0).rgb;
    let luma = dot(blur, vec3<f32>(0.2126, 0.7152, 0.0722));
    return vec4<f32>(mix(blur, vec3<f32>(luma), params.focus.w), t);
}
