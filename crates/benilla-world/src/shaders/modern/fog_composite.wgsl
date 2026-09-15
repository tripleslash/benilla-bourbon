// Modern lighting (FORK.md, P4): the fog composite — the froxel volume (or, on the
// Low tier, analytic height fog with screen-space shafts) applied to the opaque frame and the sky.
//
// Runs between the opaque and transparent passes, on the world camera and the water mirror. The
// pipeline blends `src + dst * src.a`, so the fragment returns (in-scatter, transmittance) and the
// frame keeps its own attachment (the SSILVB composite's reason, see `ssilvb.rs`). Transparents fog
// themselves in their own shaders from the same volume.
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View
#import benilla_modern::sunshadow
#import benilla_modern::froxelfog

@group(0) @binding(0) var<uniform> view: View;
#ifdef FOG_MSAA
@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var depth_tex: texture_depth_2d;
#endif
@group(0) @binding(2) var fog_volume: texture_3d<f32>;
@group(0) @binding(3) var fog_samp: sampler;
@group(0) @binding(4) var<storage, read> modern_block: sunshadow::ModernBlock;

const SHAFT_TAPS: u32 = 16u;

fn depth_at(px: vec2<i32>) -> f32 {
    return textureLoad(depth_tex, px, 0);
}

// 1 where the depth buffer holds nothing (sky) at a screen uv.
fn open_sky(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth_tex));
    let px = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.9999)) * size);
    return select(0.0, 1.0, depth_at(px) <= 0.0);
}

@fragment
fn fs_composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let f = modern_block.fog;
    let depth = depth_at(vec2<i32>(in.position.xy));
    let sky = depth <= 0.0;
    // A sky pixel is fogged as a point on its view ray at the grid's far plane.
    let near = view.clip_from_view[3][2];
    let ndc_z = select(depth, near / max(f.grid.w, 1.0), sky);
    let h = view.world_from_clip * vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, ndc_z, 1.0);
    let p = h.xyz / h.w;
    var fog = froxelfog::fog_at(fog_volume, fog_samp, f, p);

    // Low tier: the analytic in-scatter assumes an unoccluded key light. Re-weight its key share by how
    // much open sky lies between this pixel and the light on screen — radial shafts out of the gaps.
    if (f.alloc.w < froxelfog::MODE_FROXEL - 0.5 && f.proj.w > 0.0 && f.screen.z > 0.0) {
        let to_light = f.screen.xy - in.uv;
        var open = 0.0;
        var total = 0.0;
        var weight = 1.0;
        for (var i = 0u; i < SHAFT_TAPS; i = i + 1u) {
            let t = (f32(i) + 0.5) / f32(SHAFT_TAPS);
            open = open + open_sky(in.uv + to_light * t) * weight;
            total = total + weight;
            weight = weight * 0.9;
        }
        let ray = normalize(p - f.camera.xyz);
        let key = f.sun.rgb * froxelfog::dual_phase(f, dot(ray, f.sun_dir.xyz)) * (1.0 - fog.a);
        let shaft = (open / total) * 2.0 - 1.0;
        fog = vec4<f32>(max(fog.rgb + key * (shaft * f.proj.w * f.screen.z), vec3<f32>(0.0)), fog.a);
    }

    // The dome already paints the horizon haze, so the sky keeps only a share of the optical depth —
    // aerial perspective and shafts in front of it, not a second horizon.
    if (sky) {
        let tr = pow(max(fog.a, 1e-4), f.ambient.w);
        fog = vec4<f32>(fog.rgb * ((1.0 - tr) / max(1.0 - fog.a, 1e-4)), tr);
    }
    return fog;
}
