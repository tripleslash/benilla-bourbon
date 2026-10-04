// fork: the wet screen-space reflection. `weather/wet_ssr.rs` carries the tag and the wiring.
//
// Per tagged pixel: the view ray reflected about world up (wet ground reads as a flat mirror; the
// slope error disappears in the blur), marched a few steps against the snapshot's linear depth,
// refined, and the snapshot fetched where it lands: sharp in a puddle, blurred on soaked ground.
// A ray that leaves the frame, or turns back toward the camera, reflects the sky colour.
// Everything comes out in the frame's own space, so it adds straight into the view target.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::{View, position_ndc_to_view, uv_to_ndc}

@group(0) @binding(0) var<uniform> view: View;
// The opaque frame (`liquid::screen`), alpha carrying the tag.
@group(0) @binding(1) var scene_color: texture_2d<f32>;
// Its quarter-size blur.
@group(0) @binding(2) var scene_blur: texture_2d<f32>;
// Linear view depth in yards, `R32Float`; the sky reads `liquid_screen.wgsl`'s `NO_DEPTH` (1e9).
@group(0) @binding(3) var scene_depth: texture_2d<f32>;
@group(0) @binding(4) var scene_samp: sampler;
// rgb = the sky colour a ray that finds nothing reflects.
@group(0) @binding(5) var<uniform> sky: vec4<f32>;

/// Anything deeper is the sky.
const SKY_DEPTH: f32 = 1.0e8;
/// March steps, quadratically spaced so the near field gets most of them, then bisections.
const STEPS: i32 = 10;
const REFINE: i32 = 3;
/// Tags below this are not worth a march; the pixel only has its alpha put back.
const MIN_TAG: f32 = 1.0 / 256.0;

fn depth_at(uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
    return textureLoad(scene_depth, px, 0).x;
}

// A view-space point's screen UV in `xy`; `z` < 0 when it lies behind the camera or off the frame.
fn view_to_uv(p: vec3<f32>) -> vec3<f32> {
    let clip = view.clip_from_view * vec4<f32>(p, 1.0);
    if (clip.w <= 1e-4) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5;
    let inside = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    return vec3<f32>(uv, select(-1.0, 1.0, inside));
}

@fragment
fn fs_reflect(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Nearest, so a tag never bleeds into the pixel beside it.
    let cdims = vec2<f32>(textureDimensions(scene_color));
    let cpx = vec2<i32>(clamp(in.uv * cdims, vec2<f32>(0.0), cdims - 1.0));
    let tag = textureLoad(scene_color, cpx, 0).a - 1.0;
    if (tag <= 0.0) {
        discard;
    }
    // From here on the pixel is tagged: whatever it adds, its alpha goes back to 1.
    let d = depth_at(in.uv);
    if (tag < MIN_TAG || d > SKY_DEPTH) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // The fragment in view space: the near-plane point under this pixel (reverse-Z puts it at
    // NDC z = 1), scaled out to the planar depth.
    let near = position_ndc_to_view(vec3<f32>(uv_to_ndc(in.uv), 1.0), view.view_from_clip);
    let p = near * (d / max(-near.z, 1e-6));
    let n = normalize((view.view_from_world * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    let r = reflect(normalize(p), n);

    // A ray heading back at the camera has nothing on screen to find.
    let away = smoothstep(0.0, 0.25, -r.z);
    if (away <= 0.0) {
        return vec4<f32>(sky.rgb * tag, 1.0);
    }

    // Interleaved gradient noise on the step offset: banding becomes grain, which the blur eats.
    let jitter = fract(52.9829189 * fract(dot(in.position.xy, vec2<f32>(0.06711056, 0.00583715))));
    let reach = clamp(d * 2.0, 12.0, 400.0);
    var prev_t = 0.0;
    var t = 0.0;
    var hit = false;
    var landed = false;
    var uv = in.uv;
    for (var i = 0; i < STEPS; i += 1) {
        let s = (f32(i) + jitter) / f32(STEPS);
        t = 0.3 + reach * s * s;
        let q = p + r * t;
        let suv = view_to_uv(q);
        if (suv.z < 0.0) {
            break;
        }
        uv = suv.xy;
        landed = true;
        let sd = depth_at(uv);
        let behind = -q.z - sd;
        // Behind a surface by no more than this step could have crossed: the ray went through it.
        // Deeper than that, it passed behind something thin and goes on.
        let span = (t - prev_t) * max(abs(r.z), 0.1) + 0.5 + 0.01 * sd;
        if (sd < SKY_DEPTH && behind > 0.0 && behind < span) {
            hit = true;
            break;
        }
        prev_t = t;
    }
    if (hit) {
        var lo = prev_t;
        var hi = t;
        for (var k = 0; k < REFINE; k += 1) {
            let mid = 0.5 * (lo + hi);
            let q = p + r * mid;
            let suv = view_to_uv(q);
            if (suv.z >= 0.0 && -q.z > depth_at(suv.xy)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let suv = view_to_uv(p + r * hi);
        if (suv.z >= 0.0) {
            uv = suv.xy;
        }
    }

    var refl = sky.rgb;
    // A hit, or the sky the ray ran out over: what the frame shows there, faded to the sky colour
    // toward the frame's edge (the quartic the water's trace uses) so nothing pops as it leaves.
    if (landed) {
        let puddle = smoothstep(0.18, 0.35, tag);
        let fetched = mix(
            textureSampleLevel(scene_blur, scene_samp, uv, 0.0).rgb,
            textureSampleLevel(scene_color, scene_samp, uv, 0.0).rgb,
            puddle,
        );
        let e = uv * 2.0 - 1.0;
        let e4 = e * e * e * e;
        let edge = clamp(1.0 - e4.x, 0.0, 1.0) * clamp(1.0 - e4.y, 0.0, 1.0);
        refl = mix(sky.rgb, max(fetched, vec3<f32>(0.0)), edge * away);
    }
    return vec4<f32>(refl * tag, 1.0);
}
