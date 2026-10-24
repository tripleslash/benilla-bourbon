// Modern lighting (FORK.md, P4): the fog composite — the froxel volume (or, on the
// Low tier, analytic height fog with screen-space shafts) applied to the opaque frame and the sky.
//
// Runs between the opaque and transparent passes, on the world camera and the water mirror. The
// pipeline blends `src + dst * src.a`, so the fragment returns (in-scatter, transmittance) and the
// frame keeps its own attachment (the SSILVB composite's reason, see `ssilvb.rs`). Transparents fog
// themselves in their own shaders from the same volume. `fs_halo`, the nearest lamps' halos, runs
// after the transparent pass, blended additively.
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

// The shared light buffer's prefix (`lighting::global_light`): the header rows and the point table,
// packed nearest-to-camera first (`froxel_scatter.wgsl` binds the same).
struct LightTable {
    rows: array<vec4<f32>, 21>,
    points: array<vec4<f32>, 512>,
}
@group(0) @binding(5) var<storage, read> light_table: LightTable;

const SHAFT_TAPS: u32 = 16u;
// Where, as a share of the view distance, the detailed world starts dissolving into the silhouette.
const SEAM_FROM: f32 = 0.7;
// How far a sky pixel's ray is fogged (yd).
const SKY_FOG_DISTANCE: f32 = 16000.0;

// fork: the per-pixel halo reaches this many e-folding lengths from its light; past it the kernel is
// under 0.2 % of its peak.
const HALO_REACH: f32 = 3.0;
// Taps along the ray. The substitution `t = tp + H tan(theta)` absorbs the kernel's inverse-square
// peak, so eight even steps in theta hold the integral within a few percent at any miss distance.
const HALO_TAPS: u32 = 8u;

// fork: **the halo along this pixel's own ray.** For each halo light (`froxelfog::is_halo_light`),
// the halo share the froxels leave out (`froxel_scatter.wgsl`'s `point_in_scatter`):
// `(gain - far) · exp(-dl / halo_yd)` over the same falloff and phase, scattered by the medium and by
// the light's own lamp air (`froxelfog::lamp_air`), integrated from the eye to the surface `seg`
// yards out — so a surface in front of a lamp cuts its halo, and a lamp in the open stands in a glow
// as sharp as the pixel grid. Seen through the fog in front of it at the ray's closest point.
fn point_halos(f: froxelfog::FroxelFog, eye: vec3<f32>, ray: vec3<f32>, seg: f32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    // fork: the clear-weather air (`froxelfog::halo_fog`): the weather does not swell a halo.
    let clear_air = froxelfog::halo_fog(f);
    let count = froxelfog::halo_lights(f, u32(light_table.rows[20].x));
    for (var i = 0u; i < count; i = i + 1u) {
        let rgb_class = light_table.points[2u * i + 1u];
        if (!froxelfog::is_halo_light(i, count, rgb_class.w)) {
            continue;
        }
        let pos_range = light_table.points[2u * i];
        let to_light = pos_range.xyz - eye;
        let tp = dot(to_light, ray);
        let miss2 = max(dot(to_light, to_light) - tp * tp, 0.0);
        let room = froxelfog::room_weight(f, pos_range.xyz);
        let share = froxelfog::indoor_air_share(rgb_class.w, room);
        if (share <= 0.0) {
            continue;
        }
        let halo_yd = froxelfog::point_halo_yd(room);
        let reach = min(HALO_REACH * halo_yd, pos_range.w);
        if (miss2 >= reach * reach) {
            continue;
        }
        let half_span = sqrt(reach * reach - miss2);
        let t0 = max(tp - half_span, 0.0);
        let t1 = min(tp + half_span, seg);
        if (t1 <= t0) {
            continue;
        }
        // dt / (1 + dl^2) = dtheta / h, with h = sqrt(1 + miss^2).
        let h = sqrt(1.0 + miss2);
        let th0 = atan((t0 - tp) / h);
        let th1 = atan((t1 - tp) / h);
        // x = the halo kernel, y = the same under the lamp air's own falloff.
        var acc = vec2<f32>(0.0);
        for (var k = 0u; k < HALO_TAPS; k = k + 1u) {
            let th = mix(th0, th1, (f32(k) + 0.5) / f32(HALO_TAPS));
            let s = h * tan(th);
            let dl2 = s * s + miss2;
            let dl = sqrt(dl2);
            // The light-to-point direction against the point-to-eye one.
            let mu = dot(eye + ray * (tp + s) - pos_range.xyz, -ray) / max(dl, 1e-3);
            let kernel = exp(-dl / halo_yd) * froxelfog::point_falloff(dl, pos_range.w)
                * froxelfog::point_phase(mu) * (1.0 + dl2);
            acc = acc + kernel * vec2<f32>(1.0, exp(-dl / froxelfog::LAMP_AIR_YD));
        }
        let integral = acc * ((th1 - th0) / (f32(HALO_TAPS) * h));
        let pc = eye + ray * clamp(tp, t0, t1);
        let ext = froxelfog::extinction(clear_air, pc, froxelfog::LAMP_EYE_DEPTH, room, 1.0, 0.0);
        let through = froxelfog::fog_at(fog_volume, fog_samp, f, pc).a;
        let excess = froxelfog::point_halo_gain(room) - froxelfog::point_far_share(room);
        let scattered = (ext.x + ext.y + ext.z) * froxelfog::LAMP_EYE_GAIN * integral.x
            + froxelfog::lamp_air(f, rgb_class.w) * integral.y;
        sum = sum + rgb_class.rgb * (froxelfog::POINT_GAIN * excess * scattered * through * share);
    }
    return sum;
}

// The world point behind a pixel, and whether it is sky: a sky pixel stands at the grid's far plane
// along its view ray, pushed out to `SKY_FOG_DISTANCE`.
fn pixel_point(f: froxelfog::FroxelFog, in: FullscreenVertexOutput) -> vec4<f32> {
    let depth = depth_at(vec2<i32>(in.position.xy));
    let sky = depth <= 0.0;
    let near = view.clip_from_view[3][2];
    let ndc_z = select(depth, near / max(f.grid.w, 1.0), sky);
    let h = view.world_from_clip * vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, ndc_z, 1.0);
    var p = h.xyz / h.w;
    if (sky) {
        // The sky is fogged along its whole ray: the height fog saturates toward the horizon, where the
        // silhouettes' feet meet it, and thins overhead on its own.
        p = view.world_position + normalize(p - view.world_position) * SKY_FOG_DISTANCE;
    }
    return vec4<f32>(p, select(0.0, 1.0, sky));
}

// fork: **the lamp halos, after the transparent pass**, added over the finished frame. Drawn before
// it, as the composite is, an alpha-blended tree or fence behind a lamp painted over the halo and
// fogged itself from the volume, which holds none of these halos: a dark silhouette in the glow.
@fragment
fn fs_halo(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let f = modern_block.fog;
    let p = pixel_point(f, in).xyz;
    let seg = length(p - view.world_position);
    let halo = point_halos(f, view.world_position, (p - view.world_position) / max(seg, 1e-3), seg);
    return vec4<f32>(halo, 0.0);
}

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
    let pixel = pixel_point(f, in);
    let p = pixel.xyz;
    let sky = pixel.w > 0.5;
    var fog = froxelfog::fog_at(fog_volume, fog_samp, f, p);

    // Low tier: the analytic in-scatter assumes an unoccluded key light. Re-weight its key share by how
    // much open sky lies between this pixel and the light on screen — radial shafts out of the gaps.
    // Not in the water mirror: the light's screen position is the main view's, so the taps would march
    // toward the wrong point of the mirror's frame, sixteen of them per reflected pixel.
    let mirror = sunshadow::mirror_view(modern_block.cascades, view.world_position);

    // The seam: over the last stretch before the view distance the detailed world dissolves into the
    // distant band's own silhouette colour, so where the detailed terrain ends and the WDL band begins
    // there is no line. (The band itself is drawn behind the wall — its depth sits past `farclip`.)
    let farclip = f.phase.y;
    if (!sky && farclip > 0.0 && !mirror) {
        let d = froxelfog::view_depth(f, p);
        if (d <= farclip) {
            let k = smoothstep(SEAM_FROM * farclip, farclip, d);
            if (k > 0.0) {
                let fog_lin = f.ambient.rgb + f.sun.rgb * 0.5;
                let silhouette = fog_lin * pow(max(f.sun_dir.w, 0.0), 2.2);
                // Composite as `rgb + dst * a`: the pixel keeps (1 − k) of itself and gains k of the
                // silhouette, both seen through this pixel's own fog.
                fog = vec4<f32>(fog.rgb + silhouette * (k * fog.a), fog.a * (1.0 - k));
            }
        }
    }

    if (f.alloc.w < froxelfog::MODE_FROXEL - 0.5 && f.proj.w > 0.0 && f.screen.z > 0.0 && !mirror) {
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

    return fog;
}
