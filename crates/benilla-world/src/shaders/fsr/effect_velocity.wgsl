// FSR motion vectors for the effect lane (`fsr/velocity.rs`): each merged effect draw of the world
// view drawn a second time into the velocity target, with the lane's own raster (`wow_effect.wgsl`'s
// vertex path, the same depth state) so the depth test meets the depth the lane drew against, and
// its motion from each vertex's world displacement since the last world frame (`EffectMotion`).
//
// The motion goes through the camera-relative UNJITTERED matrices of the upscaler
// (`upscaler.rs::frame_matrices`): no world-space magnitude enters, so it is exact far from the
// origin. Coverage is the lane's own verdict: an alpha test at 224/255, a blend where alpha is at
// least one half, an add where the premultiplied colour is.

#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var effect_texture: texture_2d<f32>;
@group(1) @binding(1) var effect_sampler: sampler;

struct EffectMotionMatrices {
    clip_from_cur_rel: mat4x4<f32>,
    prev_clip_from_cur_rel: mat4x4<f32>,
}
@group(2) @binding(0) var<uniform> motion: EffectMotionMatrices;

struct Vertex {
    // Camera-relative, absolute for decals under DECAL_WORLD_CLIP (`particles/render.rs`).
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    // World displacement since the last world frame: current minus previous.
    @location(3) delta: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) cur_clip: vec4<f32>,
    @location(3) prev_clip: vec4<f32>,
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
#ifdef DECAL_WORLD_CLIP
    out.clip_position = view.clip_from_world * vec4<f32>(v.position, 1.0);
    let rel = v.position - view.world_position.xyz;
#else
    let view_pos = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    ) * v.position;
    out.clip_position = view.clip_from_view * vec4<f32>(view_pos, 1.0);
    let rel = v.position;
#endif
    out.cur_clip = motion.clip_from_cur_rel * vec4<f32>(rel, 1.0);
    out.prev_clip = motion.prev_clip_from_cur_rel * vec4<f32>(rel - v.delta, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSampleBias(effect_texture, effect_sampler, in.uv, view.mip_bias) * in.color;
#ifdef BLEND_ADD
    if (max(c.r, max(c.g, c.b)) * c.a < 0.5) {
        discard;
    }
#endif
#ifdef BLEND_ALPHA
    if (c.a < 0.5) {
        discard;
    }
#endif
#ifdef BLEND_ALPHAKEY
    if (c.a < 224.0 / 255.0) {
        discard;
    }
#endif
    if (in.cur_clip.w <= 1.0e-6 || in.prev_clip.w <= 1.0e-6) {
        discard;
    }
    let cur = ndc_to_uv(in.cur_clip.xy / in.cur_clip.w);
    let prev = ndc_to_uv(in.prev_clip.xy / in.prev_clip.w);
    return vec4<f32>(prev - cur, 0.0, 1.0);
}
