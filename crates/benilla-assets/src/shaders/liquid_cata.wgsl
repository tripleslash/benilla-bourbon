// **The Cataclysm liquid renderer** — a port of `waterDetail 2`'s `ProcWater` + `ProcWaterAbove` /
// `ProcWaterBelow` program family to WGSL. The 1.12.1 renderer is `liquid.wgsl` and is untouched;
// which one a surface wears is the `waterDetail` CVar, and `liquid/surface.rs` owns the swap.
//
// ## Provenance
//
// Everything below was read out of the Cataclysm client's shipped `Shaders/` programs, which are BLS
// containers of D3D SM3 bytecode (`"HSXG"`, version, count, then `{key, _, _, len, code}` records).
// The tier map is `Liquid::CMaterialWater::InitResources`:
//
//     waterDetail 0 → vs Water      + ps Water                 (the 1.12-era look)
//     waterDetail 1 → vs ProcWater  + ps MediumWater/…Below
//     waterDetail 2 → vs ProcWater  + ps ProcWaterAbove/…Below  ← THIS FILE
//
// `ProcWaterAbove` ships **54 permutations** over
// `depthFmt + 2·reflectMode + 6·dynDisplMapCnt + 18·(depthMode−1)`. This is **perm 4**:
// `samp {0, 2, 3, 5, 6}`, 102 instructions — the variant for `reflectionMode 0` and no dynamic
// displacement maps. Perm 2 differs only in taking a real planar-reflection render target on
// `samp1`; perm 4 traces the reflection in screen space against the blurred back buffer instead,
// with a flat horizon colour where the ray leaves the screen. Choosing perm 4 is not a shortcut —
// it is the program a Cataclysm player with reflections off actually ran, and it costs one fewer
// scene render.
//
// The below-water half is `ProcWaterBelow` **perm 0**: `samp {0, 2, 5, 6}`, 61 instructions — the
// same surface with the reflection dropped, which is right, because from underneath there is
// nothing above the surface to reflect but the sky the fog already stands for.
//
// ## The five stages, and what became of each
//
// | stage | reference | here |
// |---|---|---|
// | `samp0` | `"LiquidProc BackBufferTex"` | `scene_color` — the opaque frame |
// | `samp2` | `proceduralRiverDepthTex`, an 8×64 CPU gradient | **evaluated analytically** (see `swatch`) |
// | `samp3` | `"blurredBack"` | `scene_blur` |
// | `samp5` | authored Cataclysm normal map | `bump`, synthesised (`liquid::art`) |
// | `samp6` | `DepthCache::GetDepthSR()` | `scene_depth`, already linearised to yards |
//
// The depth ramp is the one deliberate departure, and it is a *gain*: the reference bakes eight
// 64-tall gradients on the CPU and re-uploads them when the zone changes, while benilla already has
// their two endpoints live on the shared global-light buffer — `Light.dbc` `LightIntBand` rows
// 16/17 (river/lake) and 14/15 (ocean), repacked every frame by `lighting::global_light`. Lerping
// those directly is the same function with one fewer copy, and it keeps cata water tracking the
// day/night clock and the zone for free, exactly as `liquid.wgsl` does.
//
// ## Axes
//
// The reference is Z-up and its programs read the surface normal's `z` as "up". benilla is Bevy's
// Y-up, and every liquid surface is a horizontal plane with normal `(0, 1, 0)`, so the port keeps
// the bump gradient in a surface-local frame and lifts it onto `(x, z)` with `y` up. The two are
// the same rotation applied consistently; naming it here is cheaper than a sign hunt later.
//
// ## Gamma
//
// GAMMA LANE (0161), unchanged: the scene copy is the frame's own gamma bytes, the swatch rows are
// gamma, and the output is raw. The reference's water math is gamma-space too, so nothing here has
// to be un-done — a refraction is a copy of neighbouring pixels, and copying is transfer-agnostic.

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{view, globals},
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var frames: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var frames_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var bump_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var bump_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var scene_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var scene_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var scene_blur: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var scene_depth: texture_2d<f32>;

struct LiquidParams {
    // x = fullbright (magma/slime); y = ocean swatch; z = interior fog; w = sun-sheen shininess.
    kind: vec4<f32>,
    // x = which renderer (0 = ADT MCLQ, 1 = WMO exterior, 2 = WMO interior);
    // y = BELOW-WATER (1 = the eye is under this surface → the `ProcWaterBelow` combine).
    path: vec4<f32>,
    // x = reserved; y = frame count; z = scroll flag; w = clock enable.
    anim: vec4<f32>,
    // x/y = the two bump tiling scales; z = scroll rate; w = bump strength (the reference's c23.w).
    bump_params: vec4<f32>,
    // x = view-perturb (c23.y); y = specular power (c4.w); z = specular scale (c27.w);
    // w = sun-glare scale (c27.z).
    shade_params: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> w: LiquidParams;

// The shared global light — the same buffer and the same row layout `liquid.wgsl` reads. See that
// file's header for what each row is and who packs it.
struct WowLight {
    light_ambient: vec4<f32>,
    light_diffuse: vec4<f32>,
    light_sun: vec4<f32>,
    light_spec: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    _sh: array<vec4<f32>, 6>,
    _sh_c16: vec4<f32>,
    water_river: array<vec4<f32>, 2>,
    water_ocean: array<vec4<f32>, 2>,
    _grade: vec4<f32>,
    wmo_fog_color: vec4<f32>,
    wmo_fog_params: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    // The four bump UV sets, packed as two `vec4`s exactly as the reference packs TEXCOORD0/1.
    // `.xy` and `.zw` of each are the same scale at two scroll phases; the two locations are two
    // different scales, and the fragment stage blends between them by view distance.
    @location(1) bump_uv_a: vec4<f32>,
    @location(2) bump_uv_b: vec4<f32>,
    // Camera-relative world offset (the reference's TEXCOORD4). Its LENGTH is the view distance the
    // detail blend keys off, so it is deliberately not normalised here.
    @location(3) view_vec: vec3<f32>,
    // The swatch coordinate: MCLQ depth for ADT, the authored opacity byte for WMO.
    @location(4) depth_v: f32,
    // `ambient + diffuse·N·L`, times the mesh's vertex colour (the reference's COLOR0).
    @location(5) lit_color: vec3<f32>,
    @location(6) uv: vec2<f32>,
}

// ─────────────────────────────────────────────────────────────────────── shared with liquid.wgsl

fn anim_time() -> f32 {
    return w.anim.w * globals.time;
}

fn frame_layer() -> i32 {
    return i32(floor(anim_time() * 24.0) % max(w.anim.y, 1.0));
}

fn apply_scroll(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x, uv.y + w.anim.z * fract(anim_time() / 10.0));
}

// Distance fog — planar eye-Z, GL_LINEAR, gamma space, and the interior-vs-scene block choice on
// `kind.z`. Identical to `liquid.wgsl`'s, deliberately: the fog a surface takes is a property of
// the surface and its room, not of which renderer is drawing it, and the two paths must agree at
// the toggle or every A/B is confounded by a fog change.
fn apply_fog(rgb: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if (w.kind.z > 0.5) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    if (fog_color.w <= 0.5) {
        return rgb;
    }
    let eye_z = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    let denom = max(fog_span.y - fog_span.x, 0.001);
    let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
    return mix(fog_color.xyz, rgb, factor);
}

// ─────────────────────────────────────────────────────────────────────────────────── the swatch

/// The depth ramp the reference samples on `samp2`, evaluated instead of sampled.
///
/// `proceduralRiverDepthTex` / `proceduralOceanDepthTex` / `proceduralWmoWaterTex` are 8×64
/// gradients built by `Liquid::ExtWaterDepthTexCallback` from the zone's `Light.dbc` water rows and
/// re-uploaded when the zone changes. Those rows are already on the shared light buffer here, so
/// this is the same two-endpoint lerp with the texture taken out of the middle. `rgb` is the body
/// colour, `a` is the opacity the refraction is tinted by.
fn swatch(v: f32) -> vec4<f32> {
    var shallow = wow_light.water_river[0];
    var deep = wow_light.water_river[1];
    if (w.kind.y > 0.5) {
        shallow = wow_light.water_ocean[0];
        deep = wow_light.water_ocean[1];
    }
    let t = clamp(v, 0.0, 1.0);
    return vec4<f32>(mix(shallow.rgb, deep.rgb, t), mix(shallow.w, deep.w, t));
}

// ─────────────────────────────────────────────────────────────────────────────────── the vertex

@vertex
fn vertex(in: Vertex) -> VsOut {
    var out: VsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    out.uv = in.uv;
    out.depth_v = in.uv_b.x;
    out.view_vec = out.world_position.xyz - view.world_position.xyz;

    // **The four bump UV sets.** The reference builds these from two 2×3 texture matrices — a
    // rotation by `GetFloat(mat, 5)` degrees scaled by `GetFloat(mat, 3)` and `(mat, 4)` — and
    // offsets each pair by a scroll phase driven off `c50`. Anchoring them to WORLD xz rather than
    // the mesh UV is what makes a lake read as one body of water: MCLQ UVs restart per chunk, and
    // a normal map keyed to them would put a visible seam on every 33-yard boundary.
    let world_uv = out.world_position.xz;
    let phase = anim_time() * w.bump_params.z;
    let s0 = w.bump_params.x;
    let s1 = w.bump_params.y;
    // The two phases of a pair scroll in different directions, so the crests interfere rather than
    // sliding as one sheet — the reference gets the same effect from the sign pattern in its two
    // matrices.
    out.bump_uv_a = vec4<f32>(
        world_uv * s0 + vec2<f32>(phase, phase * 0.37),
        world_uv * s0 * 1.31 + vec2<f32>(-phase * 0.63, phase * 0.81),
    );
    out.bump_uv_b = vec4<f32>(
        world_uv * s1 + vec2<f32>(-phase * 0.21, -phase * 0.44),
        world_uv * s1 * 0.87 + vec2<f32>(phase * 0.52, -phase * 0.29),
    );

    // COLOR0: `ambient + diffuse · saturate(N·L)`, times the mesh's vertex colour. The reference
    // computes it per vertex over the WORLD normal for exactly this reason — the surface is flat,
    // so there is nothing a per-fragment evaluation would add before the bump gets involved.
    let world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    let to_light = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(normalize(world_normal), to_light), 0.0);
    var vcolor = vec3<f32>(1.0);
#ifdef VERTEX_COLORS
    vcolor = in.color.rgb;
#endif
    out.lit_color = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    ) * vcolor;
    return out;
}

// ───────────────────────────────────────────────────────────────────────────────── the fragment

/// The four-tap bump gradient, blended between its two scales by view distance.
///
/// Verbatim from the program, constants included: the two `.xy` pairs are summed and re-centred
/// (`tap_a + tap_b − 1`), the coarse pair is scaled by `1.4`, the fine pair by `2.0` with the
/// coarse subtracted back out, and the mix runs on `saturate(0.001·d + 0.2)`. That ramp is the
/// reason cata water does not shimmer into noise at range: at the camera the coarse pair dominates
/// (`t = 0.2`), and past ~800 yards the fine pair has taken over entirely — which sounds backwards
/// until you notice the fine pair is what carries the *large* features once it is scaled by 2.
fn bump_gradient(in: VsOut) -> vec3<f32> {
    let a = textureSample(bump_tex, bump_samp, in.bump_uv_a.xy).xyz;
    let b = textureSample(bump_tex, bump_samp, in.bump_uv_a.zw).xyz;
    let c = textureSample(bump_tex, bump_samp, in.bump_uv_b.xy).xyz;
    let d = textureSample(bump_tex, bump_samp, in.bump_uv_b.zw).xyz;
    let fine = (a + b) - 1.0;
    let coarse = ((c + d) - 1.0) * 1.4;
    let dist = length(in.view_vec);
    let t = clamp(dist * 0.001 + 0.2, 0.0, 1.0);
    let g = (fine * 2.0 - coarse) * t + coarse;
    // `bump.z = 1` before the normalise, exactly as the program sets it: the stored `z` carries the
    // per-tap slope, but the SUM of four taps is a gradient, and a gradient's surface normal takes
    // a unit up-axis.
    return normalize(vec3<f32>(g.xy * w.bump_params.w, 1.0));
}

/// Screen UV for this fragment, from the rasteriser's own pixel position.
///
/// The viewport ORIGIN is subtracted, not assumed zero: `@builtin(position)` is in framebuffer
/// pixels, and a view that does not start at the corner (a booth pane, a split target) would
/// otherwise read the scene copy at an offset — which reads as water refracting a shifted copy of
/// the world, the most confusing possible symptom for the smallest possible cause.
///
/// The result is a 0..1 UV, which is also why the capture targets do not have to match the
/// viewport's resolution: `liquid_screen.wgsl` writes them by UV too, so render scale can put any
/// ratio between the two.
fn screen_uv(position: vec4<f32>) -> vec2<f32> {
    return (position.xy - view.viewport.xy) / view.viewport.zw;
}

/// Linear view depth of the opaque scene at a screen UV, in yards.
///
/// `textureLoad`, because `R32Float` takes no filtering sampler and because a depth used for an
/// in-front-of test wants a point fetch: interpolating across a silhouette invents a surface that
/// is in neither place, and every such invention here becomes a haloed edge in the refraction.
fn scene_depth_at(uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
    return textureLoad(scene_depth, px, 0).x;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    // HARD FAR-CLIP WALL — the same one terrain, the models and `liquid.wgsl` take.
    let frag_view_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
    if (wow_light.fog_params.w > 0.0 && frag_view_z > wow_light.fog_params.w) {
        discard;
    }

    // ---- Magma and slime -----------------------------------------------------------------------
    //
    // The fullbright kinds do NOT take this program in the reference either: `LiquidType.dbc`'s
    // material id routes them to `CMaterialMagma` and `CMaterialMercury`, which are separate
    // classes with separate shaders (`ProcMagma`, `ProcMercury`). `ProcMagma` samples a 128×128×16
    // noise VOLUME on stage 0 — art this client does not have and cannot synthesise without
    // guessing at lava — so magma keeps the combine it already had, which is `Magma.bls`'s own
    // (`sheet · vcolor`, fogged) and therefore already at parity with the cata *low* tier.
    // Porting `ProcMagma`/`ProcMercury` is a separate piece of work with its own art question.
    if (w.kind.x > 0.5) {
        let sheet = textureSampleBias(
            frames, frames_samp, apply_scroll(in.uv), frame_layer(), view.mip_bias);
        return vec4<f32>(apply_fog(sheet.rgb, in.world_position.xyz), 1.0);
    }

    let grad = bump_gradient(in);
    // The shading normal: the vertex normal's horizontal components pushed by the bump gradient,
    // `c23.y` deep. Y-up here where the reference is Z-up — see the header.
    let n = normalize(vec3<f32>(
        grad.x * w.shade_params.x,
        1.0,
        grad.y * w.shade_params.x,
    ));
    let v = normalize(in.view_vec);      // camera → fragment
    let to_light = -normalize(wow_light.light_sun.xyz);

    // Fresnel. `1 − saturate(N · −V)`, squared — the program keeps both, using the plain value for
    // the reflection blend's base and the square for the specular and refraction weights.
    let facing = clamp(dot(-v, n), 0.0, 1.0);
    let fres = 1.0 - facing;
    let fres2 = fres * fres;

    // Specular: Blinn against the sun, plus the program's `saturate(N.up − 0.8)` sheen floor — a
    // term that is ~0.2 on flat water and falls to zero on the crests, which is what keeps the
    // highlight sitting in the troughs rather than riding the peaks.
    let h = normalize(to_light - v);
    let sheen_floor = clamp(n.y - 0.8, 0.0, 1.0);
    let spec = pow(clamp(dot(h, n), 0.0, 1.0), max(w.shade_params.y, 1.0));
    let spec_color = wow_light.light_spec.rgb * (sheen_floor + spec);

    // The body: the depth swatch, its coordinate nudged by the bump so the shoreline band wobbles
    // with the waves instead of sitting on the geometry. The program offsets the ramp UV by
    // `abs(bump.xy · 0.1)` — absolute, so the nudge only ever reads DEEPER, never back past the
    // shore into colours the surface does not have.
    let ramp = swatch(in.depth_v + abs(grad.y) * 0.1);
    let body = ramp.rgb * in.lit_color;

    let uv = screen_uv(in.clip_position);

    // ---- Refraction ----------------------------------------------------------------------------
    //
    // Offset the screen read by the bump, scaled by how much water is actually in front of what is
    // behind it, then GUARD the result: if the offset lands on something nearer than this water
    // surface, fall back to the unoffset read. Without that guard a wave at the shoreline bends the
    // riverbank *over* the water — the classic refraction bleed, and the reference spends a `cmp`
    // on it for the same reason.
    let scene_z = scene_depth_at(uv);
    let depth_ratio = clamp((scene_z - frag_view_z) / max(frag_view_z, 0.001) * 100.0, 0.0, 1.0);
    let offset_amount = clamp((1.2 - fres2) * depth_ratio, 0.0, 1.0);
    var refract_uv = uv + grad.xy * 0.1 * offset_amount;
    if (scene_depth_at(refract_uv) < frag_view_z) {
        refract_uv = uv;
    }
    var refracted = textureSampleLevel(scene_color, scene_samp, refract_uv, 0.0).rgb;
    // Tint what shows through by the swatch's own opacity — this is where the water gets its
    // "you can see the bottom near the shore" behaviour, and it is why the surface can output
    // alpha 1.0 and still read as transparent.
    refracted = mix(refracted, body, ramp.a * (1.0 - 0.1 * fres2));

    let below = w.path.y > 0.5;
    var rgb: vec3<f32>;
    if (below) {
        // ---- ProcWaterBelow perm 0 -------------------------------------------------------------
        //
        // No reflection: from underneath there is nothing above the surface but sky, and the
        // underwater fog block already stands for it. What is left is the refraction — which from
        // below is the world seen up through the surface — carrying the specular and the glare.
        rgb = refracted;
    } else {
        // ---- ProcWaterAbove perm 4: the screen-space reflection --------------------------------
        //
        // Reflect the view about a FLATTENED normal (`N · (0.3, 0.3, 1)`, renormalised). Flattening
        // is not an approximation to apologise for: a full-strength normal throws the ray so far
        // off screen that the trace misses on most fragments, and the reference flattens by exactly
        // this before tracing.
        let flat_n = normalize(n * vec3<f32>(0.3, 1.0, 0.3));
        let r = reflect(v, flat_n);
        // A point 128 yards along the reflected ray, projected. One fixed distance rather than a
        // march: this is a reflection off a horizontal plane under a sky, so the ray's screen
        // trajectory is nearly straight and the hit is dominated by what is far away.
        let hit_world = in.world_position.xyz + r * 128.0;
        let clip = view.clip_from_world * vec4<f32>(hit_world, 1.0);
        let ndc = clip.xy / max(clip.w, 0.0001);
        // Two independent validity terms, both from the program:
        //   * the screen-edge fade `saturate(1 − ndc.y²) · (1 − ndc.x²)`, so a reflection does not
        //     pop as its source slides off the frame;
        //   * `0.75 − r.up²`, which kills the trace as the ray tips toward the sky — where there is
        //     nothing in the back buffer to find.
        let q = ndc * ndc;
        let edge = clamp(1.0 - q.y, 0.0, 1.0) * (1.0 - q.x);
        let edge_fade = select(0.0, edge, (1.0 - q.x) > 0.0);
        let skyward = 0.75 - r.y * r.y;
        let validity = select(0.0, skyward * 1.33333 * edge_fade, skyward > 0.0);
        let refl_uv = (ndc * vec2<f32>(1.0, -1.0) + 1.0) * 0.5;
        let hit = textureSampleLevel(scene_blur, scene_samp, refl_uv, 0.0).rgb;
        // Where the trace fails, the horizon. `c21` in the program is a constant the client uploads
        // per zone; benilla's equivalent is the fog colour, which in WoW IS the horizon — the sky
        // dome converges on it by construction, so a reflection that falls back to it falls back to
        // something the player can see is right.
        var reflected = mix(wow_light.fog_color.rgb, hit, clamp(validity, 0.0, 1.0));
        // Then 60% of the way back toward the body colour. Water is not a mirror; this is the
        // program's own `c11.y`, and it is what stops a lake from reading as chrome at noon.
        reflected = mix(reflected, body, 0.6);
        rgb = mix(refracted, reflected, fres2 * 0.5 + 0.1);
    }

    // The sun's broad glare on the surface, modulated by the body so it cannot brighten water that
    // is already dark, and the specular, weighted by the Fresnel square so it only fires at grazing
    // angles.
    let glare = clamp(dot(to_light, n), 0.0, 1.0) * w.shade_params.w;
    rgb += glare * wow_light.light_spec.rgb * body;
    rgb += w.shade_params.z * spec_color * fres2;

    rgb = apply_fog(rgb, in.world_position.xyz);
    // **Alpha 1.0, and that is the program's own answer** (`mov ocol0.w, 1`). Cata water is opaque
    // at the blend stage because it does its own transparency: what is behind the surface arrives
    // through the refraction read, tinted by the swatch. It still draws in the transparent pass, on
    // the water rung, so the sort order the rest of the frame depends on is unchanged.
    return vec4<f32>(rgb, 1.0);
}
