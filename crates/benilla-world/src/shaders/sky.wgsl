// The sky-dome gradient (`SkyExt`): the five Light.dbc SkyColor stops at the reference's ring
// elevations, then the fog colour at and below the horizon (`0x6d0d10`, `0x6d0f50`), interpolated
// per fragment by elevation. Depth is the far pin in `sky_vertex.wgsl`; this writes colour only.
#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

struct SkyColors {
    sky0: vec4<f32>, // zenith (90°)
    sky1: vec4<f32>, // 16.8°
    sky2: vec4<f32>, // 9.8°
    sky3: vec4<f32>, // 3.7°
    sky4: vec4<f32>, // 1.8°
    fog: vec4<f32>,  // horizon (0°) and below: LightIntBand row 7
    warp: vec4<f32>, // x = dawn/dusk warp strength S (0 = off), y = sun azimuth (rad), zw reserved
    // fork: modern lighting (FORK.md, P4) — xyz = camera→sun, w = modern flag.
    modern: vec4<f32>,
    // fork: modern lighting (P4) — rgb = the sun glow's linear radiance, w = its Mie g.
    glow: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> sky: SkyColors;

// Glow `g` for a sun-relative azimuth phase, sampled linearly like `0x6d0f50` from the reference's
// six-keyframe wrap-around table at `0xce9af8` (written by `0x6ce210`); 0.125 is the sun bearing.
fn azimuth_glow(phase: f32) -> f32 {
    let p = fract(phase);
    if (p < 0.125) { return mix(0.0, 1.0, (p + 0.125) / 0.25); } // wrap 0.875(g0)→1.125(g1)
    else if (p < 0.375) { return mix(1.0, 0.0, (p - 0.125) / 0.25); }
    else if (p < 0.5) { return mix(0.0, -0.5, (p - 0.375) / 0.125); }
    else if (p < 0.625) { return mix(-0.5, -0.7, (p - 0.5) / 0.125); }
    else if (p < 0.75) { return mix(-0.7, -0.5, (p - 0.625) / 0.125); }
    else if (p < 0.875) { return mix(-0.5, 0.0, (p - 0.75) / 0.125); }
    else { return mix(0.0, 1.0, (p - 0.875) / 0.25); } // wrap 0.875(g0)→1.125(g1)
}

// One mid-ring's warped colour for glow `g` (`0x6d0f50`): `S^2` is the prepass S nested in the
// per-segment S, and 0.7 is the constant at `0x7ffd7c`.
fn warp_one(base: vec3<f32>, g: f32, s: f32) -> vec3<f32> {
    let s2 = s * s;
    if (g >= 0.0) {
        return mix(base, sky.sky1.rgb, (1.0 - g) * s2);
    }
    return mix(mix(base, sky.sky1.rgb, s), sky.sky0.rgb, 0.7 * (-g) * s2);
}

// fork (Everwood port): the smooth gradient of a modern frame.
// The reference's ring elevations (degrees), horizon to zenith.
const STOP_ELEV = array<f32, 6>(0.0, 1.8, 3.7, 9.8, 16.8, 90.0);

// Fritsch–Carlson tangent at an interior stop: 0 at a local extremum, else the weighted harmonic
// mean of the two secants, so the curve never overshoots a stop (no halos, no new colours).
fn pchip_tangent(first_delta: vec3<f32>, second_delta: vec3<f32>, first_span: f32, second_span: f32) -> vec3<f32> {
    let left_weight = 2.0 * second_span + first_span;
    let right_weight = second_span + 2.0 * first_span;
    let left_safe = select(first_delta, vec3<f32>(1.0), first_delta == vec3<f32>(0.0));
    let right_safe = select(second_delta, vec3<f32>(1.0), second_delta == vec3<f32>(0.0));
    let m = (left_weight + right_weight) / (left_weight / left_safe + right_weight / right_safe);
    return select(vec3<f32>(0.0), m, first_delta * second_delta > vec3<f32>(0.0));
}

// Monotone cubic through the six stops (`y[0]` = fog at 0°, `y[5]` = zenith), linear light. The
// end tangents are 0: flat into the fog band below the horizon and flat over the zenith, so the
// curve is C1 everywhere and the rings leave no Mach bands.
fn smooth_gradient(elev: f32, y: array<vec3<f32>, 6>) -> vec3<f32> {
    if (elev <= 0.0) {
        return y[0];
    }
    if (elev >= 90.0) {
        return y[5];
    }
    var yy = y;
    var se = STOP_ELEV;
    var d: array<vec3<f32>, 5>;
    var h: array<f32, 5>;
    for (var i = 0; i < 5; i++) {
        h[i] = se[i + 1] - se[i];
        d[i] = (yy[i + 1] - yy[i]) / h[i];
    }
    var k = 0;
    for (var i = 1; i < 5; i++) {
        if (elev >= se[i]) {
            k = i;
        }
    }
    var tangent_start = vec3<f32>(0.0);
    var tangent_end = vec3<f32>(0.0);
    if (k > 0) {
        tangent_start = pchip_tangent(d[k - 1], d[k], h[k - 1], h[k]);
    }
    if (k < 4) {
        tangent_end = pchip_tangent(d[k], d[k + 1], h[k], h[k + 1]);
    }
    let hk = h[k];
    let t = (elev - se[k]) / hk;
    let t_sq = t * t;
    let t_cube = t_sq * t;
    let blend_start = 2.0 * t_cube - 3.0 * t_sq + 1.0;
    let blend_start_tangent = t_cube - 2.0 * t_sq + t;
    let blend_end = -2.0 * t_cube + 3.0 * t_sq;
    let blend_end_tangent = t_cube - t_sq;
    return blend_start * yy[k] + blend_start_tangent * hk * tangent_start
        + blend_end * yy[k + 1] + blend_end_tangent * hk * tangent_end;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_position.xyz - view.world_position.xyz);
    let elev = degrees(asin(clamp(dir.y, -1.0, 1.0))); // −90..90, 0 = horizon

    // Dawn/dusk warp (`0x6d0f50`): only the four mid rings, never the apex or the fog rim, and
    // never brighter. The reference bakes it per vertex at 24 azimuth segments and interpolates,
    // hence the lerp of two segments; `+ 0.125` puts the sun bearing at glow phase 0.125.
    var s1 = sky.sky1.rgb;
    var s2c = sky.sky2.rgb;
    var s3 = sky.sky3.rgb;
    var s4 = sky.sky4.rgb;
    let warp_s = sky.warp.x;
    if (warp_s > 0.0) {
        let az = fract((atan2(dir.z, dir.x) - sky.warp.y) / 6.2831853 + 0.125);
        let seg = az * 24.0; // 24 azimuth segments, like the dome vertices
        let s0 = floor(seg);
        let f = seg - s0;
        let g0 = azimuth_glow(s0 / 24.0);
        let g1 = azimuth_glow((s0 + 1.0) / 24.0);
        s1 = mix(warp_one(sky.sky1.rgb, g0, warp_s), warp_one(sky.sky1.rgb, g1, warp_s), f);
        s2c = mix(warp_one(sky.sky2.rgb, g0, warp_s), warp_one(sky.sky2.rgb, g1, warp_s), f);
        s3 = mix(warp_one(sky.sky3.rgb, g0, warp_s), warp_one(sky.sky3.rgb, g1, warp_s), f);
        s4 = mix(warp_one(sky.sky4.rgb, g0, warp_s), warp_one(sky.sky4.rgb, g1, warp_s), f);
    }

    // Elevation gradient, linear between rings like the reference's Gouraud-shaded dome.
    var col: vec3<f32>;
    if (elev <= 0.0) {
        col = sky.fog.rgb; // horizon and below = fog colour (row 7), unwarped
    } else if (elev < 1.8) {
        col = mix(sky.fog.rgb, s4, elev / 1.8);
    } else if (elev < 3.7) {
        col = mix(s4, s3, (elev - 1.8) / (3.7 - 1.8));
    } else if (elev < 9.8) {
        col = mix(s3, s2c, (elev - 3.7) / (9.8 - 3.7));
    } else if (elev < 16.8) {
        col = mix(s2c, s1, (elev - 9.8) / (16.8 - 9.8));
    } else {
        col = mix(s1, sky.sky0.rgb, (elev - 16.8) / (90.0 - 16.8)); // warped ring1 to the raw apex
    }
    // fork (Everwood port): a modern frame runs a monotone cubic through the same six stops (which
    // are already linear there, `sky.rs`'s decode hook), so the rings leave no bands. Classic keeps
    // the reference's linear rings above.
    if (sky.modern.w > 0.5) {
        col = smooth_gradient(
            elev, array<vec3<f32>, 6>(sky.fog.rgb, s4, s3, s2c, s1, sky.sky0.rgb)
        );
    }

    // Raw gamma out: the reference draws the sky as raw DBC bytes, sRGB off (`0x6d4940`).
    // fork: modern lighting (FORK.md, P4) — the HDR sky: a Mie halo around the sun
    // that hugs the horizon, and an analytic disk core for the bloom (the textured disc draws over it).
    if (sky.modern.w > 0.5) {
        let mu = dot(dir, normalize(sky.modern.xyz));
        let g = sky.glow.w;
        let denom = max(1.0 + g * g - 2.0 * g * mu, 1e-4);
        let mie = (1.0 - g * g) / (denom * sqrt(denom));
        let horizon = 1.0 - 0.6 * smoothstep(0.0, 0.5, dir.y);
        let disk = smoothstep(0.9999, 0.99996, mu);
        col = col + sky.glow.rgb * (0.02 * mie * horizon + 24.0 * disk);
    }
    let rgb = col;
    return vec4<f32>(rgb, 1.0);
}
