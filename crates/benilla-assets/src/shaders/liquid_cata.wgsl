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
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var noise_vol: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(112) var noise_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var foam_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var foam_samp: sampler;

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
    // x = foam tiling (repeats/yard); y = foam depth scale (yards); z = foam strength (c28.y);
    // w = how far the wave normal drags the second foam tap.
    foam_params: vec4<f32>,
    // ProcMagma: x/y = the two noise-volume sample scales, z/w = layer 1 and 2 scales.
    magma_scales: vec4<f32>,
    // ProcMagma: x = layer 3 scale, y = wobble frequency, z = clock rate, w = tap drift rate.
    magma_params: vec4<f32>,
    // ProcMagma: the base colour under every layer (the reference's c17.rgb).
    magma_color: vec4<f32>,
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

@group(#{MATERIAL_BIND_GROUP}) @binding(113) var ripple_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(114) var ripple_samp: sampler;

/// The dynamic ripple field's placement — see `benilla_world`'s `liquid::ripple`, which owns the
/// simulation and writes this every frame. Only the first two rows are read here; the rest is the
/// wave step's own state and is declared so the two shaders agree on the layout.
struct Ripple {
    // xy = the map's min corner in world XZ; z = 1/extent; w = interpolation between the two
    // packed heights.
    placement: vec4<f32>,
    // x = one texel in UV; y = the slope scale; z = enabled; w = damping.
    params: vec4<f32>,
    prev: vec4<f32>,
    stamp_count: vec4<f32>,
    stamps: array<vec4<f32>, 16>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(91) var<storage, read> ripple: Ripple;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    // **These three carry two different things, and the reference does the same.** On water they
    // are the four bump UV sets and the two foam sets — `bump_uv_*` packed as two `vec4`s exactly
    // as `ProcWater` packs TEXCOORD0/1, `.xy` and `.zw` of each being one scale at two scroll
    // phases, the two locations being two scales the fragment stage blends by view distance.
    //
    // On MAGMA the same interpolator slots carry `ProcMagma`'s: the two noise-volume taps, the
    // first two lava layers, the third layer and the wobble. That is not a trick to save space —
    // `ProcMagma` and `ProcWater` are two vertex programs writing the same TEXCOORD0/1/2 registers,
    // and a `MaterialExtension` gets one vertex entry point for both, so the vertex stage branches
    // on `kind.x` and fills whichever set its kind will read. Each branch is a uniform branch and
    // costs nothing; adding three magma-only varyings would instead cost interpolator bandwidth on
    // every water fragment in the world, which is the common case by a wide margin.
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
    // The two shoreline-foam UV sets (the reference's TEXCOORD2), a high-frequency pair with a
    // slow sinusoidal wobble so the surf churns rather than slides.
    @location(7) foam_uv: vec4<f32>,
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

    if (w.kind.x > 0.5) {
        // ---- ProcMagma's vertex half -------------------------------------------------------
        //
        // Two noise-volume taps that drift apart with the clock (so the two never agree and the
        // blend between them keeps moving), three layer UV sets at three scales, and the wobble.
        let drift = anim_time() * w.magma_params.w;
        out.bump_uv_a = vec4<f32>(
            world_uv * w.magma_scales.x + vec2<f32>(drift, drift * 0.4),
            world_uv * w.magma_scales.y - vec2<f32>(drift * 0.7, drift * 0.9),
        );
        out.bump_uv_b = vec4<f32>(
            world_uv * w.magma_scales.z,
            world_uv * w.magma_scales.w + vec2<f32>(drift * 0.15, -drift * 0.11),
        );
        // **The wobble**, and it is worth spelling out because the bytecode hides it behind six
        // `sincos` and a pile of range wrapping. `ProcMagma`'s vertex program builds
        // `0.5 + 0.15 · 0.5 · Σ` over three harmonics of a transformed UV in both axes — the
        // `frac(x/2π + 0.5)·2π − π` around each is only bringing the angle into `sincos`'s domain,
        // so the whole of it is `sin(k·u) + sin(k·v)` for k = 1, 2, 3. It is what decides, per
        // vertex, which of the two volume taps a patch of crust follows, and three harmonics is
        // what keeps that from reading as a grid.
        let mw = world_uv * w.magma_params.y + drift * 0.05;
        let harmonics =
            (sin(mw.x) + sin(mw.y))
            + (sin(2.0 * mw.x) + sin(2.0 * mw.y))
            + (sin(3.0 * mw.x) + sin(3.0 * mw.y));
        out.foam_uv = vec4<f32>(
            world_uv * w.magma_params.x,
            0.5 + 0.075 * harmonics,
            0.0,
        );
        // COLOR0 is unread on this lane — magma is fullbright, which is the reference's own state
        // (`GL_LIGHTING` off at both magma passes) and not a simplification.
        out.lit_color = vec3<f32>(1.0);
        return out;
    }

    // The foam UV pair. The reference builds these from the bump coordinate scaled by 228 plus a
    // `sincos` wobble driven off the clock and the vertex position — a high frequency with a slow
    // drift, which is what keeps surf from reading as a sliding decal. Same shape here, anchored to
    // world XZ for the same reason the bump sets are.
    let fs = w.foam_params.x;
    let wob = vec2<f32>(
        sin(phase * 4.0 + world_uv.x * 0.05),
        cos(phase * 2.0 + world_uv.y * 0.05),
    ) * 0.1;
    out.foam_uv = vec4<f32>(world_uv * fs + wob, world_uv * fs * 0.73 - wob * 0.7);

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
fn bump_gradient(in: VsOut) -> vec2<f32> {
    let a = textureSample(bump_tex, bump_samp, in.bump_uv_a.xy).xyz;
    let b = textureSample(bump_tex, bump_samp, in.bump_uv_a.zw).xyz;
    let c = textureSample(bump_tex, bump_samp, in.bump_uv_b.xy).xyz;
    let d = textureSample(bump_tex, bump_samp, in.bump_uv_b.zw).xyz;
    let fine = (a + b) - 1.0;
    let coarse = ((c + d) - 1.0) * 1.4;
    let dist = length(in.view_vec);
    let t = clamp(dist * 0.001 + 0.2, 0.0, 1.0);
    let g = (fine * 2.0 - coarse) * t + coarse;
    // The raw SLOPE, not a normal. The reference normalises here (`bump.z = 1`, then `nrm`), but the
    // ripple field has its own slope to add and the two have to be summed before either is turned
    // into a direction — normalising twice would let whichever is larger swallow the other.
    return g.xy * w.bump_params.w;
}

/// The dynamic ripple field's contribution to the surface slope.
///
/// Three taps rather than four: the centre and one step in each axis is a forward difference, which
/// is what `ProcWaterAbove` takes (it reads centre, +x and +y and subtracts). The lookup is nudged
/// by the bump gradient first — the reference's own `−bump · 0.02` — so a wake shears with the swell
/// crossing it instead of lying flat under it.
///
/// Zero outside the map, and zero when the field is off. The map covers a square that follows the
/// camera, so "outside" is most of the world's water at any moment; the early-out is the common
/// case, not the exception.
fn ripple_slope(world_xz: vec2<f32>, bump: vec2<f32>) -> vec2<f32> {
    if (ripple.params.z < 0.5) {
        return vec2<f32>(0.0);
    }
    let uv = (world_xz - ripple.placement.xy) * ripple.placement.z - bump * 0.02;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec2<f32>(0.0);
    }
    let e = ripple.params.x;
    let c = ripple_height(uv);
    return vec2<f32>(ripple_height(uv + vec2<f32>(e, 0.0)) - c,
                     ripple_height(uv + vec2<f32>(0.0, e)) - c) * ripple.params.y;
}

/// One height sample, interpolated between the two fixed steps the map packs.
///
/// `mix(previous, current, t)` — the reference's `lrp` against its `c31.x`, and the reason a 60 Hz
/// simulation does not stutter on a 144 Hz display.
fn ripple_height(uv: vec2<f32>) -> f32 {
    let h = textureSampleLevel(ripple_map, ripple_samp, uv, 0.0);
    return mix(h.y, h.x, ripple.placement.w);
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

/// **`ProcMagma`** — the Cataclysm magma/slime program, ported whole.
///
/// The reference routes these kinds away from the water material entirely: `LiquidType.dbc`'s
/// material id sends them to `CMaterialMagma` and `CMaterialMercury`, separate classes with
/// separate shaders. What `ProcMagma` does is composite three layers over a base colour, each
/// layer masked by one channel of an animated noise volume:
///
/// ```text
/// mask = mix(volume(uvA, t), volume(uvB, t), wobble)   // three independent channels
/// rgb  = mix(base,  layer1, mask.x)
/// rgb  = mix(rgb,   layer2, mask.y · layer2.a)
/// rgb  = mix(rgb,   layer3, mask.z · layer3.a)
/// ```
///
/// The volume is `benilla_world`'s synthesised stand-in for `proceduralNoiseVolTex`; its third axis
/// is a clock, so the masks crawl. The three LAYERS are where this departs from the reference and
/// says so: it binds three authored Cataclysm textures through `LiquidType.dbc` columns 1.12.1 does
/// not have, so the port samples the kind's own animated sheet at three different scales instead.
/// The structure is the reference's and the art is the one this client actually ships — which is
/// what turns a flat scrolling sheet into a crust with hot cracks moving under it.
///
/// **Slime takes this too.** `CMaterialMercury` is genuinely a different program (`ProcMercury` is
/// bump and Fresnel over a reflection, closer to water than to lava) and is NOT ported here; slime
/// rides the magma composite because it is the same shape of problem — a body whose sheet is its
/// colour — and because the alternative is leaving it flat. Named rather than hidden.
fn magma_sheet(uv: vec2<f32>, layer: i32) -> vec4<f32> {
    return textureSampleBias(frames, frames_samp, uv, layer, view.mip_bias);
}

fn magma(in: VsOut) -> vec4<f32> {
    let layer = frame_layer();
    // The clock walks the volume's third axis. `fract` is belt and braces — the sampler repeats on
    // `w` — but it keeps the coordinate small enough that f32 does not start losing the low bits of
    // it after an hour of uptime.
    let slice = fract(anim_time() * w.magma_params.z);
    let na = textureSample(noise_vol, noise_samp, vec3<f32>(in.bump_uv_a.xy, slice)).rgb;
    let nb = textureSample(noise_vol, noise_samp, vec3<f32>(in.bump_uv_a.zw, slice)).rgb;
    let mask = mix(na, nb, in.foam_uv.z);

    let l1 = magma_sheet(in.bump_uv_b.xy, layer);
    let l2 = magma_sheet(in.bump_uv_b.zw, layer);
    let l3 = magma_sheet(in.foam_uv.xy, layer);

    var rgb = mix(w.magma_color.rgb, l1.rgb, mask.x);
    rgb = mix(rgb, l2.rgb, clamp(mask.y * l2.a, 0.0, 1.0));
    rgb = mix(rgb, l3.rgb, clamp(mask.z * l3.a, 0.0, 1.0));
    // Fogged and opaque, exactly as before: the reference fogs every liquid batch, magma included,
    // and `ocol0.w` is a hard 1 in the program.
    return vec4<f32>(apply_fog(rgb, in.world_position.xyz), 1.0);
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
        return magma(in);
    }

    // The swell, plus whatever a body wading nearby has pushed into the water. The two are summed
    // as SLOPES and normalised once — see `bump_gradient`.
    let swell = bump_gradient(in);
    let grad = normalize(vec3<f32>(
        swell + ripple_slope(in.world_position.xz, swell),
        1.0,
    ));
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

    // ---- Shoreline foam ------------------------------------------------------------------------
    //
    // `ProcWaterAbove`'s samp7 permutations, and the whole of it is one idea: **the water column**.
    // `scene_z − frag_view_z` is how far the bottom sits behind the surface at this pixel, and
    // where that collapses, the surface is meeting something. Surf is what a water surface does
    // where it meets something.
    //
    // That is also why this one term covers both of the things foam is asked for. A shoreline is
    // where the column goes to zero because the bed rises into the surface; a character standing in
    // the water is where it goes to zero because a body does. The mask cannot tell them apart and
    // does not need to — it foams around rocks, piers, tree trunks and players for free, which is
    // exactly the behaviour the reference gets from the same subtraction.
    //
    // The column is scaled by `1 − V.up` before the ramp, which is the reference's own slant
    // correction: at a grazing view you are looking through more water for the same depth, so the
    // surf band reads wider from the shore than from directly above it.
    let column = max(scene_z - frag_view_z, 0.0);
    let slant = 1.0 - v.y;
    // The band WIDENS at grazing angles. The reference divides by `(fres² · c9.w + c9.z) · 2.5`,
    // i.e. an authored depth scale with a Fresnel-driven term added to it; the shape is what
    // matters and the coefficient is folded to one constant here, the two registers behind it
    // being `LiquidType.dbc` columns that 1.12.1 has no equivalent of.
    let foam_depth = max(w.foam_params.y, 0.01) * (1.0 + fres2 * 1.5);
    let t = clamp(column * slant / foam_depth, 0.0, 1.0);
    // `1 − smoothstep`: full at the waterline, gone by the depth scale. Hermite rather than a
    // linear ramp because a linear one puts a visible straight edge along the contour where it
    // reaches zero, and a shoreline is exactly where that edge would be looked at.
    let shore = 1.0 - (t * t * (3.0 - 2.0 * t));
    // **The two attenuators that make this subtle rather than a white sheet**, both the reference's
    // and both easy to miss in the bytecode because they are single `mul`s a long way from the
    // rest of the term:
    //
    //   * the **view-distance blend** — the very same `saturate(0.001·d + 0.2)` the bump taps use.
    //     At the camera that is 0.2, so the surf you are standing over is at a fifth strength and
    //     only reaches full out near the horizon. Without it the shoreline nearest the player is
    //     the brightest thing in the frame, which is exactly backwards.
    //   * **N.up** — foam lies flat. A wave face turned toward you carries less of it.
    let detail_blend = clamp(length(in.view_vec) * 0.001 + 0.2, 0.0, 1.0);
    // Two taps, ADDED, the second dragged by the wave normal so the two layers beat against each
    // other instead of sliding as one sheet. The sheet is sparse and near-black by construction
    // (`liquid::art`) precisely so this addition is foam rather than a lift.
    let foam_a = textureSample(foam_tex, foam_samp, in.foam_uv.xy).rgb;
    let foam_b = textureSample(
        foam_tex,
        foam_samp,
        in.foam_uv.zw + n.zx * w.foam_params.w,
    ).rgb;
    let foam = (foam_a + foam_b) * (shore * w.foam_params.z * detail_blend * n.y);

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
    // Foam last, and additively: it sits ON the water rather than being tinted by it, and it is
    // the one term that should survive a dark shoreline at dusk.
    rgb += foam;

    rgb = apply_fog(rgb, in.world_position.xyz);
    // **Alpha 1.0, and that is the program's own answer** (`mov ocol0.w, 1`). Cata water is opaque
    // at the blend stage because it does its own transparency: what is behind the surface arrives
    // through the refraction read, tinted by the swatch. It still draws in the transparent pass, on
    // the water rung, so the sort order the rest of the frame depends on is unchanged.
    return vec4<f32>(rgb, 1.0);
}
