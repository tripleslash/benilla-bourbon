#define_import_path benilla_world::effect_points

// fork: perf — the effect lane's point draws (`particles::buffer::EffectShape`): one 32-byte point
// per drop, splash, flake or puff, built into its triangle or quad here by vertex index. A triangle
// is vertices 0..3, a quad six vertices over the perimeter corners (bl, br, tr, tl) as [0, 1, 2]
// and [0, 2, 3]. Positions come back camera-relative: the point arrives absolute and the eye comes
// off before any corner is added, so a millimetre extent survives a world coordinate in the
// thousands.
//
// `shape_a.x` names the shape; the other lanes of `shape_a` and `shape_b` are its numbers, in the
// order `EffectShape::rows` packs them.

struct Expanded {
    // World position less the eye.
    rel: vec3<f32>,
    uv: vec2<f32>,
    // Raw gamma RGBA, as a stream vertex carries it.
    color: vec4<f32>,
}

// `v` normalized, or `fallback` for a zero vector.
fn unit_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > 1.0e-12) {
        return v * inverseSqrt(len2);
    }
    return fallback;
}

// `v` turned by the unit quaternion `q` (xyz, w).
fn turn(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// A quad vertex's corner signs: x right, y up.
fn quad_corner(index: u32) -> vec2<f32> {
    var corners = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let c = corners[index % 6u];
    return vec2<f32>(
        select(-1.0, 1.0, c == 1u || c == 2u),
        select(-1.0, 1.0, c >= 2u),
    );
}

// fork: modern weather — a procedural puff's coverage at `uv`: `(1 − r²)³` over the inscribed disc,
// whose value and slope both reach zero at the rim, so stacked puffs leave no contour.
fn puff_falloff(uv: vec2<f32>) -> f32 {
    let p = uv * 2.0 - 1.0;
    let k = clamp(1.0 - dot(p, p), 0.0, 1.0);
    return k * k * k;
}

// `point_a` = (pos, a), `point_b` = (v, b). `eye`, `right`, `up` and `forward` are the view's
// world position and axes.
fn expand(
    point_a: vec4<f32>,
    point_b: vec4<f32>,
    index: u32,
    shape_a: vec4<f32>,
    shape_b: vec4<f32>,
    eye: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    forward: vec3<f32>,
) -> Expanded {
    var out: Expanded;
    let rel = point_a.xyz - eye;
    out.rel = rel;
    out.uv = vec2<f32>(0.0);
    out.color = vec4<f32>(1.0);
    if (shape_a.x < 1.5) {
        // Streak (half_w, tail, near; tilt): the base straddles the eye line across the fall, the apex
        // trails up the fall, leaned by the wind (`0x80ff78`, `0x80ff74`, `0x674a70`). fork: with a
        // `near` (modern frames) the triangle shrinks away toward the eye, whole from `near` yards.
        let anti_vel = -unit_or(point_b.xyz, vec3<f32>(0.0, -1.0, 0.0));
        let size = select(1.0, smoothstep(0.25 * shape_a.w, shape_a.w, length(rel)), shape_a.w > 0.0);
        if (index % 3u == 2u) {
            out.rel = rel + turn(shape_b, anti_vel * (shape_a.z * size));
            out.uv = vec2<f32>(0.5, 0.0);
        } else {
            let to_eye = unit_or(-rel, vec3<f32>(1.0, 0.0, 0.0));
            let across = unit_or(cross(to_eye, anti_vel), vec3<f32>(1.0, 0.0, 0.0)) * (shape_a.y * size);
            let side = select(-1.0, 1.0, index % 3u == 1u);
            out.rel = rel + across * side;
            out.uv = vec2<f32>(0.5 + 0.5 * side, 1.0);
        }
    } else if (shape_a.x < 2.5) {
        // Patter (right, up, life): base-left, apex, base-right over a 4×4 atlas, a frame a
        // quarter of its life (`0x675ac0`).
        let frame = min(floor(clamp(point_a.w / shape_a.w, 0.0, 1.0) * 4.0), 3.0);
        let uv0 = vec2<f32>(frame, point_b.w) * 0.25;
        let k = index % 3u;
        if (k == 0u) {
            out.rel = rel - right * shape_a.y;
            out.uv = uv0 + vec2<f32>(0.0, 0.25);
        } else if (k == 1u) {
            out.rel = rel + up * shape_a.z;
            out.uv = uv0 + vec2<f32>(0.125, 0.043);
        } else {
            out.rel = rel + right * shape_a.y;
            out.uv = uv0 + vec2<f32>(0.25, 0.25);
        }
    } else if (shape_a.x < 3.5) {
        // Flake (px_at_eye, falloff, px_min; world_per_px): `snowpoint.bls`'s point size off the
        // radial distance, then pixels to world at this view depth. Behind the eye it has no size.
        let corner = quad_corner(index);
        let px = max(shape_a.y * clamp(1.0 - shape_a.z * length(rel), 0.0, 1.0), shape_a.w);
        let half = px * max(dot(rel, forward), 0.0) * shape_b.x;
        out.rel = rel + (right * corner.x + up * corner.y) * half;
        out.uv = vec2<f32>(0.5 + 0.5 * corner.x, 0.5 - 0.5 * corner.y);
        out.color = vec4<f32>(1.0, 1.0, 1.0, point_a.w);
    } else {
        // Puff (half, near, far; colour): alpha per corner by its own distance, so a quad beside
        // the eye shows its far corners (`0x67add0`).
        let corner = quad_corner(index);
        out.rel = rel + (right * corner.x + up * corner.y) * shape_a.y;
        out.uv = vec2<f32>(0.5 + 0.5 * corner.x, 0.5 - 0.5 * corner.y);
        let ramp = clamp((length(out.rel) - shape_a.z) / (shape_a.w - shape_a.z), 0.0, 1.0);
        out.color = vec4<f32>(shape_b.xyz, point_a.w * ramp);
    }
    return out;
}
