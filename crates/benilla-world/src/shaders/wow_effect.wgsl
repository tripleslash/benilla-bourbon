// The effect-lane pass (particles, ribbons, decals, water foam, precipitation): texture bytes,
// never decoded on sample, times the authored track colour in gamma space, as the reference
// multiplies them. The output stays gamma; the FFXGlow combine decodes the frame once.
//
// Blend variants, one shader def per EffectBlend:
// - BLEND_ADD:      (rgb·α, 0) under (One, 1−srcα), the reference's `src·α` byte add.
// - BLEND_ALPHA:    straight (rgb, α) under standard alpha blending.
// - BLEND_OPAQUE:   (rgb, 1) with blending off.
// - BLEND_ALPHAKEY: BLEND_OPAQUE behind the alpha test (EGxRs id 0x08), a discard below 224/255.
// - BLEND_MULTIPLY: (rgb·α, α) under (Dst, 1−srcα) = `dst·lerp(1, rgb, α)`, the blob shadow's
//   modulate-with-fade and ModelBlend::Mod at α = 1.
// - BLEND_MOD2X:    (rgb, 1) under (Dst, Src) = `2·src·dst`, rain's state; reads no alpha.

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import benilla_modern::sunshadow
#import benilla_modern::froxelfog
#import bevy_render::view::View
#ifdef EFFECT_POINTS
#import benilla_world::effect_points
#endif

// Prefix of `lighting::global_light`'s buffer; keep in sync with wow_model.wgsl's copy.
struct WowLight {
    light_ambient: vec4<f32>,
    light_diffuse: vec4<f32>,
    light_sun: vec4<f32>,
    light_spec: vec4<f32>,
    fog_color: vec4<f32>,  // rgb row-7 fog (gamma); w = enable (>0.5)
    fog_params: vec4<f32>, // x=start y=end z=linear-lighting A/B flag w=farclip wall
    sh_c10_r: vec4<f32>,
    sh_c10_g: vec4<f32>,
    sh_c10_b: vec4<f32>,
    sh_c13_r: vec4<f32>,
    sh_c13_g: vec4<f32>,
    sh_c13_b: vec4<f32>,
    sh_c16: vec4<f32>,
    _water: array<vec4<f32>, 4>,
    grade: vec4<f32>,
    _wmo_fog: array<vec4<f32>, 2>, // 18-19
    point_count: vec4<f32>,        // 20 // fork: modern lighting (FORK.md) — y = modern flag
};

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var effect_texture: texture_2d<f32>;
@group(1) @binding(1) var effect_sampler: sampler;
@group(1) @binding(2) var<storage, read> wow_light: WowLight;
// Per-draw params. `fog`: x the fog colour policy, `0x70baf0`'s per-blend table (0 off for file
// flag 0x8, 1 scene, 2 black for Add, 3 white for Mod, 4 grey for Mod2x); y rain's forced fog, zw
// its start and end. `clip`: the target rect in pixels (min.xy, max.xy), the whole target when
// z <= x; the reference clips a UI model to its widget's viewport, and model panes share an atlas.
// `shape_a`, `shape_b`: a point draw's shape (`effect_points.wgsl`), zero on every other draw.
struct EffectParams {
    fog: vec4<f32>,
    clip: vec4<f32>,
    shape_a: vec4<f32>,
    shape_b: vec4<f32>,
};
@group(1) @binding(3) var<uniform> wow_params: EffectParams;
// fork: modern lighting (FORK.md, P4) — the froxel fog, sampled in the vertex stage.
@group(1) @binding(4) var<storage, read> modern_block: sunshadow::ModernBlock;
@group(1) @binding(5) var modern_fog_volume: texture_3d<f32>;
@group(1) @binding(6) var modern_fog_samp: sampler;

#ifdef EFFECT_PARTICLES
// fork: perf — a GPU-expanded particle pool's record (`particles::buffer::ParticlePool`): the CPU
// writer's per-pool inputs (`particles::quads::expand_quads`), one 256-byte row by dynamic offset.
struct ParticlePool {
    center_x: vec4<f32>, // the stored frame to the world, row-major; w the translation
    center_y: vec4<f32>,
    center_z: vec4<f32>,
    cam_right: vec4<f32>, // w: the size unit
    cam_up: vec4<f32>,    // w: the model's render alpha
    plane_r: vec4<f32>,   // the XY-quad basis; w: the spin
    plane_u: vec4<f32>,   // w: the tail time
    color0: vec4<f32>,
    color1: vec4<f32>,
    color2: vec4<f32>,
    scale: vec4<f32>,     // three size keys, then mid
    head: vec4<f32>,      // the head ramps (base, span) for segments A and B
    tail: vec4<f32>,      // the tail ramps likewise
    atlas: vec4<f32>,     // the segment repeats, then the atlas columns and rows
    mode: vec4<f32>,      // x: head_tail | xy << 2 | tail_clamps << 3 | committed << 4; yzw: light
    spare: vec4<f32>,
};
@group(1) @binding(7) var<uniform> particle_pool: ParticlePool;
#endif

// The rain pass's forced fog colour: 0x80808080 → grey (render-state 0x0d).
const RAIN_FOG_GREY: vec3<f32> = vec3<f32>(0.50196078, 0.50196078, 0.50196078);

#ifdef EFFECT_POINTS
// fork: perf — a point draw: one instance a point, its corner picked by vertex index
// (`effect_points.wgsl`).
struct Vertex {
    @builtin(vertex_index) index: u32,
    @location(0) point_a: vec4<f32>, // world position, the shape's scalar
    @location(1) point_b: vec4<f32>, // the shape's vector, its second scalar
};
#else
#ifdef EFFECT_PARTICLES
// fork: perf — a pool draw: one instance a particle (`particles::buffer::GpuParticle`), six
// vertices a quad, its head first.
struct Vertex {
    @builtin(vertex_index) index: u32,
    @location(0) pos_age: vec4<f32>, // stored-frame position, age
    @location(1) vel_u: vec4<f32>,   // stored-frame velocity, normalized age
    @location(2) twinkle: f32,       // the gated twinkle's size multiplier
    @location(3) phase: u32,
};
#else
struct Vertex {
    // Camera-relative for f32 precision (absolute verts through `clip_from_world` shear thin
    // geometry far from the origin), except decals under DECAL_WORLD_CLIP, which arrive absolute.
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,    // raw authored gamma RGBA; α is the blend weight
};
#endif
#endif

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    // Planar eye-Z for fog and the farclip wall, positive in front of the camera.
    @location(2) view_z: f32,
    // fork: modern lighting (FORK.md, P4) — rgb = in-scatter, a = transmittance (1 in a classic frame).
    @location(3) fog: vec4<f32>,
};

#ifdef EFFECT_PARTICLES
// One corner of a particle's quad, its world position relative to the camera and its texel.
struct ParticleCorner {
    rel: vec3<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
};

// `CellRamp::sample`: `floor(base + span·t) & 0xFF`.
fn particle_cell(base: f32, span: f32, t: f32) -> u32 {
    return u32(i32(floor(base + span * t)) & 0xFF);
}

// The CPU writer's law, `particles::quads::expand_quads`, for corner `index` of the particle.
fn particle_corner(v: Vertex) -> ParticleCorner {
    let p = particle_pool;
    let bits = u32(p.mode.x);
    let head_tail = bits & 3u;
    let xy_quad = (bits & 4u) != 0u;
    let tail_clamps = (bits & 8u) != 0u;
    let committed = (bits & 16u) != 0u;
    let has_head = head_tail != 1u;
    let quad = v.index / 6u;
    // Two triangles over the perimeter-order corners, as the lane's `[0,1,2, 0,2,3]` indices.
    var order = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let corner = order[v.index % 6u];
    let is_head = has_head && quad == 0u;

    // The over-life ramps (`OverLife::sample`).
    let u = clamp(v.vel_u.w, 0.0, 1.0);
    let mid = clamp(p.scale.w, 1e-3, 1.0);
    var t: f32;
    var seg: u32;
    var c0: vec4<f32>;
    var c1: vec4<f32>;
    var s0: f32;
    var s1: f32;
    if (u <= mid) {
        t = u / mid;
        seg = 0u;
        c0 = p.color0;
        c1 = p.color1;
        s0 = p.scale.x;
        s1 = p.scale.y;
    } else {
        t = (u - mid) / max(1.0 - mid, 1e-3);
        seg = 1u;
        c0 = p.color1;
        c1 = p.color2;
        s0 = p.scale.y;
        s1 = p.scale.z;
    }
    t = clamp(t, 0.0, 1.0) * 0.99 + 0.005;
    var rgba = c0 + (c1 - c0) * t;
    let size = s0 + (s1 - s0) * t;
    let repeat = select(p.atlas.y, p.atlas.x, seg == 0u);
    var ct = t;
    if (repeat != 1.0) {
        ct = fract(t * repeat);
    }
    var cell: u32;
    if (is_head) {
        cell = select(
            particle_cell(p.head.z, p.head.w, ct),
            particle_cell(p.head.x, p.head.y, ct),
            seg == 0u,
        );
    } else {
        cell = select(
            particle_cell(p.tail.z, p.tail.w, ct),
            particle_cell(p.tail.x, p.tail.y, ct),
            seg == 0u,
        );
    }
    // The model's render alpha scales alpha alone (`0x7b9b10`).
    rgba.a = rgba.a * p.cam_up.w;
    // A committed light folds into the RGB, the product clamped (`fold_committed_light`).
    if (committed) {
        rgba = vec4<f32>(clamp(rgba.rgb * p.mode.yzw, vec3<f32>(0.0), vec3<f32>(1.0)), rgba.a);
    }

    let pos = v.pos_age.xyz;
    let age = v.pos_age.w;
    let world = vec3<f32>(
        dot(p.center_x.xyz, pos) + p.center_x.w,
        dot(p.center_y.xyz, pos) + p.center_y.w,
        dot(p.center_z.xyz, pos) + p.center_z.w,
    );
    let center = world - view.world_position.xyz;
    let half = size * v.twinkle * p.cam_right.w;
    let cam_right = p.cam_right.xyz;
    let cam_up = p.cam_up.xyz;

    // The atlas cell: `col = idx & (cols − 1)`, `row = idx >> log2(cols)`.
    let cols = u32(p.atlas.z);
    let inv_cols = 1.0 / p.atlas.z;
    let inv_rows = 1.0 / p.atlas.w;
    let cx = f32(cell & (cols - 1u));
    let cy = f32(cell >> countTrailingZeros(cols));
    let u0 = cx * inv_cols;
    let u1 = (cx + 1.0) * inv_cols;
    let v0 = cy * inv_rows;
    let v1 = (cy + 1.0) * inv_rows;

    var out: ParticleCorner;
    out.color = rgba;
    var offsets: array<vec3<f32>, 4>;
    var uvs: array<vec2<f32>, 4>;
    if (is_head) {
        var base_r = cam_right;
        var base_u = cam_up;
        if (xy_quad) {
            base_r = p.plane_r.xyz;
            base_u = p.plane_u.xyz;
        }
        var r = base_r * half;
        var up = base_u * half;
        let spin = p.plane_r.w;
        if (spin != 0.0) {
            // `spin_angle`: a negative angle negated on a bit-5 particle.
            var angle = spin * age;
            if (angle < 0.0 && (v.phase & 0x20u) != 0u) {
                angle = -angle;
            }
            let sa = sin(angle);
            let ca = cos(angle);
            r = (base_r * ca + base_u * sa) * half;
            up = (base_u * ca - base_r * sa) * half;
        }
        offsets = array<vec3<f32>, 4>(-r - up, r - up, r + up, -r + up);
        uvs = array<vec2<f32>, 4>(vec2(u0, v1), vec2(u1, v1), vec2(u1, v0), vec2(u0, v0));
    } else {
        // The tail streak: |velocity|·tailTime back along the motion, 2·half wide on screen.
        let vel_world = vec3<f32>(
            dot(p.center_x.xyz, v.vel_u.xyz),
            dot(p.center_y.xyz, v.vel_u.xyz),
            dot(p.center_z.xyz, v.vel_u.xyz),
        );
        var t_eff = p.plane_u.w;
        if (tail_clamps) {
            t_eff = min(t_eff, age);
        }
        let tail = -vel_world * t_eff;
        let tr = dot(tail, cam_right);
        let tu = dot(tail, cam_up);
        let l2 = tr * tr + tu * tu;
        if (l2 < 7.7e-4) {
            // Degenerate: the plain-billboard fallback (`0x7b33fa`).
            let r = cam_right * half;
            let up = cam_up * half;
            offsets = array<vec3<f32>, 4>(-r - up, r - up, r + up, -r + up);
            uvs = array<vec2<f32>, 4>(vec2(u0, v1), vec2(u1, v1), vec2(u1, v0), vec2(u0, v0));
        } else {
            let inv_l = half / sqrt(l2);
            let perp = (cam_up * tr - cam_right * tu) * inv_l;
            offsets = array<vec3<f32>, 4>(-perp, perp, tail + perp, tail - perp);
            uvs = array<vec2<f32>, 4>(vec2(u0, v1), vec2(u0, v0), vec2(u1, v0), vec2(u1, v1));
        }
    }
    out.rel = center + offsets[corner];
    out.uv = uvs[corner];
    return out;
}
#endif

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
#ifdef EFFECT_POINTS
    let built = effect_points::expand(
        v.point_a,
        v.point_b,
        v.index,
        wow_params.shape_a,
        wow_params.shape_b,
        view.world_position.xyz,
        view.world_from_view[0].xyz,
        view.world_from_view[1].xyz,
        -view.world_from_view[2].xyz,
    );
    let position = built.rel;
    let uv = built.uv;
    let color = built.color;
#else
#ifdef EFFECT_PARTICLES
    let built = particle_corner(v);
    let position = built.rel;
    let uv = built.uv;
    let color = built.color;
#else
    let position = v.position;
    let uv = v.uv;
    let color = v.color;
#endif
#endif
#ifdef DECAL_WORLD_CLIP
    // Decals (raster_bias ≠ 0) take absolute verts through the world meshes' `clip_from_world`:
    // their depth must tie the ground within the raster bias, and the cam-relative route rounds
    // differently by more than that at world-scale coordinates.
    out.clip_position = view.clip_from_world * vec4<f32>(position, 1.0);
    // Eye-Z via the full affine transform; its rounding is yard-scale and harmless.
    out.view_z = -(view.view_from_world * vec4<f32>(position, 1.0)).z;
#else
    // Cam-relative verts: view_from_world is [R | −R·cam] and the rebase already subtracted cam,
    // so only the rotation applies.
    let view_pos = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    ) * position;
    out.clip_position = view.clip_from_view * vec4<f32>(view_pos, 1.0);
    out.view_z = -view_pos.z;
#endif
    // fork: modern lighting (FORK.md, P4) — the froxel fog per vertex, like the
    // reference's fog on this path (an off block returns no fog).
#ifdef DECAL_WORLD_CLIP
    let fog_world = position;
#else
    let fog_world = view.world_position.xyz + position;
#endif
    out.fog = froxelfog::fog_at(modern_fog_volume, modern_fog_samp, modern_block.fog, fog_world);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // The target-rect clip; `@builtin(position)` is the framebuffer pixel, a UI tile's atlas texel.
    if (wow_params.clip.z > wow_params.clip.x) {
        let p = in.clip_position.xy;
        if (p.x < wow_params.clip.x || p.y < wow_params.clip.y
            || p.x > wow_params.clip.z || p.y > wow_params.clip.w) {
            discard;
        }
    }
    // `$WOW_PARTICLE_FLAT` debug instrument: solid magenta, no inputs.
#ifdef WOW_PARTICLE_FLAT
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
#else
    // The hard farclip wall (the live `farclip`, 177 to 777 yd): the reference's far plane clips
    // effects and ours is farther, so every world shader discards. The fog ends by the farclip, so
    // this cuts only fog.
    if (wow_light.fog_params.w > 0.0 && in.view_z > wow_light.fog_params.w) {
        discard;
    }
    // Texture bytes × authored vertex colour in gamma space; `view.mip_bias` is the render-scale
    // LOD compensation, 0.0 at native and above.
    var c = textureSampleBias(effect_texture, effect_sampler, in.uv, view.mip_bias) * in.color;
#ifdef EFFECT_POINTS
    // fork: modern weather — a procedural puff (shape_b.w = 1) takes an analytic falloff in place
    // of `snowmist01.blp`, whose 4-bit DXT3 alpha bands into rings.
    if (wow_params.shape_a.x > 3.5 && wow_params.shape_b.w > 0.5) {
        c = vec4<f32>(in.color.rgb, in.color.a * effect_points::puff_falloff(in.uv));
    }
#endif
#ifdef BLEND_ALPHAKEY
    // The fixed-function alpha test (EGxRs id 0x08, `glAlphaFunc(GL_GEQUAL, ref/255)`): `0x70c256`
    // sets ref = round(instanceAlpha × 224), 224 at full alpha, and `c.a` is the reference's
    // fragment alpha (texture α × track α, GL_MODULATE). GEQUAL passes at the ref.
    if (c.a < 224.0 / 255.0) {
        discard;
    }
#endif
    var rgb = c.rgb;
#ifdef EFFECT_LIT
    // Scene lighting (EGxRs id 0x0e, `0x70d8b0`) when the emitter clears the unlit flag 0x1 and
    // its blend is not Mod or Mod2x (`0x70bb00`): the fixed-function matte with N = world up for
    // the whole draw, as `0x7b3fd0` takes the normal from view row 2 and `0x71bce0` moves the
    // light into that frame, so N·L holds as the camera orbits. Bevy +Y is WoW +Z.
    let L = -normalize(wow_light.light_sun.xyz);
    let N = vec3<f32>(0.0, 1.0, 0.0);
    let lit = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * max(dot(N, L), 0.0),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    rgb = rgb * lit;
#endif
    // Scene fog: the world's linear fog (same start and end, planar eye-Z) in gamma space before
    // the blend; only its colour follows the per-blend policy in `wow_params.fog.x`.
    // fork: modern lighting (FORK.md, P4) — a modern frame takes the froxel fog below.
    if (wow_light.fog_color.w > 0.5 && wow_params.fog.x > 0.5 && wow_light.point_count.y < 0.5) {
        let denom = max(wow_light.fog_params.y - wow_light.fog_params.x, 0.001);
        let factor = clamp((wow_light.fog_params.y - in.view_z) / denom, 0.0, 1.0);
        var fog_rgb = wow_light.fog_color.xyz;
        if (wow_params.fog.x > 1.5 && wow_params.fog.x < 2.5) { fog_rgb = vec3<f32>(0.0); }
        else if (wow_params.fog.x > 2.5 && wow_params.fog.x < 3.5) { fog_rgb = vec3<f32>(1.0); }
        else if (wow_params.fog.x > 3.5) { fog_rgb = RAIN_FOG_GREY; }
        rgb = mix(fog_rgb, rgb, factor);
    }
    // Rain's forced fog: grey over the draw's own start/end, whatever the scene fog; grey is
    // neutral under Mod2x, so this is the streaks' distance fade.
    if (wow_params.fog.y > 0.5) {
        let denom = max(wow_params.fog.w - wow_params.fog.z, 0.001);
        let factor = clamp((wow_params.fog.w - in.view_z) / denom, 0.0, 1.0);
        rgb = mix(RAIN_FOG_GREY, rgb, factor);
    }
    // fork: modern lighting (FORK.md) — decode after the additive premultiply (in
    // gamma, like the classic fold: decoding first inflates a soft halo by α^(1/2.2)); Mod2x keeps
    // its 0.5-grey identity neutral in linear (see wow_model.wgsl).
    var add_alpha = c.a;
    if (wow_light.point_count.y > 0.5) {
#ifdef BLEND_ADD
        rgb = colorspace::srgb_to_linear(rgb * c.a);
        add_alpha = 1.0;
#else
        rgb = colorspace::srgb_to_linear(rgb);
#endif
#ifdef BLEND_MOD2X
        rgb = rgb * 2.295;
#endif
        // P4: the per-vertex froxel fog, by the draw's fog policy — scene, black (fade out), white and the
        // rain grey (the Mod/Mod2x identities).
        if (wow_light.fog_color.w > 0.5 && wow_params.fog.x > 0.5) {
            if (wow_params.fog.x > 1.5 && wow_params.fog.x < 2.5) {
                rgb = rgb * in.fog.a;
            } else if (wow_params.fog.x > 2.5 && wow_params.fog.x < 3.5) {
                rgb = mix(vec3<f32>(1.0), rgb, in.fog.a);
            } else if (wow_params.fog.x > 3.5) {
                rgb = mix(vec3<f32>(0.5), rgb, in.fog.a);
            } else {
                rgb = rgb * in.fog.a + in.fog.rgb;
            }
        }
    }
#ifdef BLEND_ADD
    // Premultiplied in gamma, so stacked quads sum like the reference's bytes.
    return vec4<f32>(rgb * add_alpha, 0.0);
#else
#ifdef BLEND_OPAQUE
    return vec4<f32>(rgb, 1.0);
#else
#ifdef BLEND_ALPHAKEY
    // Blending is off (EGxBlend 1 → `glDisable(GL_BLEND)`); the surviving fragments are solid.
    return vec4<f32>(rgb, 1.0);
#else
#ifdef BLEND_MULTIPLY
    return vec4<f32>(rgb * c.a, c.a);
#else
#ifdef BLEND_MOD2X
    return vec4<f32>(rgb, 1.0);
#else
    return vec4<f32>(rgb, c.a);
#endif
#endif
#endif
#endif
#endif
#endif
}
