// FSR 3.1.5 pass 3 — shading change (`ffx_fsr3upscaler_shading_change.h`).
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
//
// Collapses the three pyramid levels `shading_change_pyramid.wgsl` built
// into one factor per half-resolution pixel: how much this neighbourhood's
// *shading* changed for reasons no motion vector can explain.
//
// Each level contributes `|mean difference * mean sign|`. The product is the
// point of the pair: a level whose pixels all moved the same way keeps its
// magnitude, while one whose differences disagree has a mean sign near zero and
// contributes nothing. Taking the maximum across levels then means a change is
// reported at whatever scale it is coherent on — a small bright highlight sliding
// across a surface shows up in level 0, a whole wall changing brightness in
// level 2 — without the fine levels' per-pixel noise being able to fake either.
//
// Runs on the half-resolution grid the pyramid's level 0 lives on; the
// accumulation samples the result back through a linear filter.

@group(0) @binding(1) var mip0_tex: texture_2d<f32>;
@group(0) @binding(2) var mip1_tex: texture_2d<f32>;
@group(0) @binding(3) var mip2_tex: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var shading_change_out: texture_storage_2d<FSR3_FMT_R8, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = shading_change_size_i();
    let p = vec2<i32>(gid.xy);
    if (!is_on_screen(p, size)) {
        return;
    }

    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / vec2<f32>(size);
    // Re-applies the jitter, which the accumulation reverses again when it
    // samples this texture — so the two agree on which scene position a value
    // describes rather than on which pixel index holds it.
    let uv_jittered = uv + params.jitter * params.inv_render_size;
    let mip_uv = clamp_uv(uv_jittered, vec2<f32>(textureDimensions(mip0_tex)));

    var samples: array<f32, 3>;
    let s0 = textureSampleLevel(mip0_tex, linear_sampler, mip_uv, 0.0).xy;
    let s1 = textureSampleLevel(mip1_tex, linear_sampler, mip_uv, 0.0).xy;
    let s2 = textureSampleLevel(mip2_tex, linear_sampler, mip_uv, 0.0).xy;
    samples[0] = abs(s0.x * s0.y);
    samples[1] = abs(s1.x * s1.y);
    samples[2] = abs(s2.x * s2.y);

    // `1 + iShadingChangeMipStart / s_MipLevelsToUse`, which is 1 while the
    // detector starts at level 0.
    let scale = 1.0 + f32(SHADING_CHANGE_MIP_START) / f32(SHADING_CHANGE_MIP_LEVELS);

    var shading_change = 0.0;
    for (var i = SHADING_CHANGE_MIP_START; i < SHADING_CHANGE_MIP_LEVELS; i = i + 1) {
        if (samples[i] > 0.0) {
            shading_change = max(shading_change, samples[i]) * scale;
        }
    }

    textureStore(shading_change_out, p, vec4<f32>(saturate(shading_change), 0.0, 0.0, 0.0));
}
