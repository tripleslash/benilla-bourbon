// FSR 3.1.5 input adapter — screen-space motion vectors and the generated reactive mask.
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
//
// Not a pass of the reference: FidelityFX takes both as *inputs*, and this is where benilla makes
// them. Everything downstream is the SDK's chain, from `prepare_inputs.wgsl` on.
//
// **Motion.** FSR wants a jitter-free uv displacement pointing from this frame to the previous one.
// Static geometry moves only with the camera, so its vector is a reprojection of the depth buffer:
//
//   1. Unproject the depth sample through the **jittered** inverse the frame was rasterized with,
//      into camera-relative world space, and keep it homogeneous.
//   2. Project it with this frame's and last frame's **unjittered** matrices, and subtract.
//
// Keeping the point homogeneous is what makes the sky right: reverse-Z with an infinite far plane
// puts it at depth 0, which unprojects to `w = 0`, a direction; the translation between the two eyes
// drops out and the vector is the camera's rotation alone. Moving content (`velocity.wgsl`) writes its
// own vector into the velocity target, and where that target is written its `xy` wins.
//
// **Reactive.** The SDK's `GenerateReactiveMask` (`ffx_fsr3upscaler_autogen_reactive_pass.hlsl`), with
// `USE_COMPONENTS_MAX` and, on a modern frame, `APPLY_TONEMAP`: how much the transparent pass changed
// this pixel, `min(1, scale · max|final − opaque|)`. It rides in `z`; `w` is the
// transparency-and-composition mask, which benilla does not supply.

@group(0) @binding(1) var depth_tex: texture_depth_2d;
// `xy` = uv motion, previous minus current, jitter-free; `w` = 1 where a moving draw wrote it.
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
// The finished colour, and the copy of it taken before the transparent pass.
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var opaque_tex: texture_2d<f32>;
// xy = motion, z = reactive, w = transparency and composition.
@group(0) @binding(5) var mv_out: texture_storage_2d<rgba16float, write>;

fn reactive(p: vec2<i32>) -> f32 {
    var pre = textureLoad(opaque_tex, p, 0).rgb;
    var post = textureLoad(color_tex, p, 0).rgb;
    if (is_hdr()) {
        pre = tonemap(pre * params.exposure);
        post = tonemap(post * params.exposure);
    }
    let delta = abs(post - pre);
    return min(1.0, max(delta.x, max(delta.y, delta.z)) * params.reactive_scale);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }

    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;
    let device_depth = textureLoad(depth_tex, p, 0);

    // The point this pixel shaded, camera-relative and homogeneous.
    let rel = params.cur_rel_from_clip * vec4<f32>(uv_to_ndc(uv), device_depth, 1.0);
    let cur_clip = params.clip_from_cur_rel * rel;
    let prev_clip = params.prev_clip_from_cur_rel * rel;

    var mv = vec2<f32>(0.0);
    // Behind the eye in either frame there is no screen position to difference; a zero vector
    // sends the reprojection to this pixel's own history, which the disocclusion test then judges.
    if (cur_clip.w > 1.0e-6 && prev_clip.w > 1.0e-6) {
        mv = ndc_to_uv(prev_clip.xy / prev_clip.w) - ndc_to_uv(cur_clip.xy / cur_clip.w);
    }
    let velocity = textureLoad(velocity_tex, p, 0);
    if (velocity.w > 0.5) {
        mv = velocity.xy;
    }

    textureStore(mv_out, p, vec4<f32>(mv, reactive(p), 0.0));
}
