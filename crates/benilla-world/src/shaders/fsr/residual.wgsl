// benilla's own FSR instrument (`WOW_FSR_DEBUG=residual`, `fsr/residual.rs`), not a pass of the
// reference: how well a frame's motion vectors predict it from the last one. Over the pixels a
// moving draw wrote (the velocity target's `w`), the luma difference between this frame and the
// last one read back through (0) the frame's own motion and (1) the camera's reprojection alone,
// summed ×1024 with the pixel count (2). Motion vectors that carry the movement make (0) smaller
// than (1); both carry the same jitter noise.

@group(0) @binding(1) var depth_tex: texture_depth_2d;
// `xy` = uv motion, previous minus current; `w` = 1 where a moving draw wrote it.
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
// Last rendered frame's finished colour, at render resolution.
@group(0) @binding(4) var prev_tex: texture_2d<f32>;
@group(0) @binding(5) var lin: sampler;
@group(0) @binding(6) var<storage, read_write> sums: array<atomic<u32>, 3>;

fn display_luma(c: vec3<f32>) -> f32 {
    var x = c;
    if (is_hdr()) {
        x = tonemap(x * params.exposure);
    }
    return rgb_to_luma(clamp(x, vec3<f32>(0.0), vec3<f32>(1.0)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = render_size_i();
    let p = vec2<i32>(gid.xy);
    if (any(p >= size)) {
        return;
    }
    let velocity = textureLoad(velocity_tex, p, 0);
    if (velocity.w <= 0.5) {
        return;
    }
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) * params.inv_render_size;

    // The camera's reprojection, as `motion.wgsl` computes it for static geometry.
    let device_depth = textureLoad(depth_tex, p, 0);
    let rel = params.cur_rel_from_clip * vec4<f32>(uv_to_ndc(uv), device_depth, 1.0);
    let cur_clip = params.clip_from_cur_rel * rel;
    let prev_clip = params.prev_clip_from_cur_rel * rel;
    var camera = vec2<f32>(0.0);
    if (cur_clip.w > 1.0e-6 && prev_clip.w > 1.0e-6) {
        camera = ndc_to_uv(prev_clip.xy / prev_clip.w) - ndc_to_uv(cur_clip.xy / cur_clip.w);
    }

    let cur = display_luma(textureLoad(color_tex, p, 0).rgb);
    let through_mv = display_luma(textureSampleLevel(prev_tex, lin, uv + velocity.xy, 0.0).rgb);
    let through_camera = display_luma(textureSampleLevel(prev_tex, lin, uv + camera, 0.0).rgb);
    atomicAdd(&sums[0], u32(abs(through_mv - cur) * 1024.0));
    atomicAdd(&sums[1], u32(abs(through_camera - cur) * 1024.0));
    atomicAdd(&sums[2], 1u);
}
