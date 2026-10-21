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
// `depthFmt + 2·reflectMode + 6·dynDisplMapCnt + 18·(depthMode−1)` — the formula read straight off
// `CMaterialWater::RenderAbove`'s `v6 + 2*a2 + 6*dynDisplMapCnt + 2*(9*depthMode - 9)`, and the
// count it implies (54, and 27 for `ProcWaterBelow`) is exactly what the shipped containers hold.
// This is **perm 0**: `samp {0, 1, 2, 3, 5, 6}`, `reflectMode 0` with no dynamic displacement maps,
// and it is the arm a `waterDetail 2` player actually runs. `CRenderer::Flush` picks the mode as
// `v29 = renderer[+0x30]` — 0 for the ordinary renderer, 1 for the real-reflection one — and only
// then `if (!v29 && instance->settings[+0x24] == 2) v29 = 2`. So mode 1 (perms 2/3) is the
// `waterDetail > 2` tier and mode 2 (perms 4/5) is a minority arm on one liquid class; mode 0 is
// the default, and the only one that reads BOTH the reflection buffer on `samp1` and the blurred
// back buffer on `samp3`, lerping the second over the first.
//
// **This file used to port perm 4, and that is the bug it was reported for.** Perm 4 has no
// `samp1`: its reflection is the screen-space trace alone, with a flat horizon constant where the
// trace leaves the frame — and `saturate(0.75 − r.up²)` says outright that the trace dies as the
// ray tips skyward, which is exactly what pitching the camera down does. Water went flat on the
// pitch a player uses to look at a lake. Perm 0 has a real mirrored render underneath the trace, so
// that same pitch reveals the mirror instead of a constant. `benilla_world`'s `liquid::reflect`
// owns the camera that fills it, and carries the cost and the gaps.
//
// The `fp40` sibling of each container is the same program as **plain ARB text**, and it is the
// better read: `PARAM c[36] = { program.local[0..31], … }` names the client constants outright,
// where the SM3 build scatters its literals through `def`s at c2/c9/c10/c11/c12 (registers the
// client also writes and this program does not read). Both were traced for this port and they
// agree instruction for instruction.
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

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import benilla_modern::sunshadow
#import benilla_modern::froxelfog
#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{view, globals},
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var frames: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var frames_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var bump_tex: texture_2d<f32>;
// **One sampler for all three synthesised sheets** — the bump map, the foam and the magma noise
// volume. `liquid::art` builds them with byte-identical descriptors (Repeat on u/v/w, Linear mag,
// min and mipmap), a sampler is pipeline STATE rather than a property of a texture, and WebGPU
// samplers carry no dimensionality — so the 3D volume takes the same one the 2D sheets do.
//
// This is not tidying, it is the fix for a black screen. WebGPU caps FRAGMENT samplers at **16 per
// stage** and counts them across every bind group in the pipeline layout, Bevy's view groups
// included; three separate ones here put this material at **17**, and `CreatePipelineLayout`
// refused it. On the browser backend that refusal is silent (wgpu forwards and never reads the
// result), and an invalid pipeline poisons the command buffer that binds it, so `submit` dropped
// the WHOLE frame — every other pass with it — and the canvas presented as black wherever liquid
// drew. Both liquid pipelines fell to it, `AlphaMode::Opaque` magma and `AlphaMode::Blend` water
// alike, because they share this layout. Native never sees any of it: Vulkan and D3D12 report the
// limit in the hundreds of thousands, and Bevy asks the adapter for whatever it has.
//
// So: a FOURTH sampler on this material costs the frame, not the sheet. The remaining three are
// 101 (`frames`, its own filter policy), 106 (`scene_*`, clamped) and 114 (`ripple_map`); 106 and
// 114 are themselves identical and can merge the same way if this ever needs another.
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var art_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var scene_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var scene_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var scene_blur: texture_2d<f32>;
/// **The planar reflection buffer** — `ProcWaterAbove`'s `samp1`, filled by a second, mirrored
/// scene render (`benilla_world`'s `liquid::reflect`). Takes `scene_samp` (106): it is a projected
/// screen read like the other two, and a fourth sampler on this material costs the frame — see the
/// note at 104.
@group(#{MATERIAL_BIND_GROUP}) @binding(115) var reflect_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var scene_depth: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var noise_vol: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var foam_tex: texture_2d<f32>;

struct LiquidParams {
    // x = fullbright (magma/slime); y = ocean swatch; z = interior fog; w = sun-sheen shininess.
    kind: vec4<f32>,
    // x = which renderer (0 = ADT MCLQ, 1 = WMO exterior, 2 = WMO interior);
    // y = BELOW-WATER (1 = the eye is under this surface → the `ProcWaterBelow` combine);
    // z = is `reflect_map` live this frame (the reference's `HasFakeReflection`);
    // w = the world Y the mirror was rendered about (`liquid::reflect`'s plane).
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
    // fork: modern lighting — xyz = absorption per yard of water (r, g, b) (`modern_water`);
    // w = the Modern (Experimental) water tier (`waterDetail 3`): 1 runs `modern_water`, 0 keeps
    // the classic combine on a modern frame too.
    optics: vec4<f32>,
    // fork: modern lighting — the Modern (Experimental) tier's stylised clear water (`modern_water`):
    // x = how far the bed is filtered toward the water's hue, y = the depth (yd) that filter takes to
    // come in, z = the head-on reflectance, w = how much of the reflection is pulled to the body.
    look_a: vec4<f32>,
    // x = the wind swell a river or lake keeps (the sea keeps all of it), y = the swell left over
    // water with no depth under it, z = the refraction's bend per yard of water, w = the glitter
    // spike's gain.
    look_b: vec4<f32>,
    // x = the reflectance at grazing (the Fresnel's cap), y = the power of its climb, z = the share
    // of the reference's combine blended over the clear surface, w = the wind ripples' RMS slope
    // (`wind_ripples`).
    look_c: vec4<f32>,
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
    point_count: vec4<f32>,        // 20 // fork: modern lighting (FORK.md) — y = modern flag
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;
// fork: modern lighting (FORK.md, P4) — water fogs itself from the froxel volume.
// fork: modern lighting — the sun shadow on water: how much of the lit colour the shade keeps.
const WATER_SHADE_KEEP: f32 = 0.8;
// fork: modern lighting — where the water's shadow is read: on the bed down the view ray, stopped
// `BED_LIFT` yd short of it (so the bed does not shadow itself), handing over to the surface as the
// column under the pixel deepens from `PROBE` to twice that.
const WATER_SHADOW_PROBE: f32 = 3.0;
const WATER_SHADOW_BED_LIFT: f32 = 0.15;

@group(#{MATERIAL_BIND_GROUP}) @binding(123) var<storage, read> modern_block: sunshadow::ModernBlock;
@group(#{MATERIAL_BIND_GROUP}) @binding(124) var modern_fog_volume: texture_3d<f32>;
// fork: modern lighting — the sun cascades, read without a sampler (the layout has none left).
@group(#{MATERIAL_BIND_GROUP}) @binding(121) var modern_shadow_map: texture_depth_2d_array;
// fork: modern lighting — the HDR sun glitter `fragment_classic` leaves for the linear wrapper, which
// adds it after the decode so it can carry energy above white into the bloom.
var<private> modern_glint: vec3<f32>;

@group(#{MATERIAL_BIND_GROUP}) @binding(113) var ripple_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(114) var ripple_samp: sampler;

/// The dynamic ripple field's placement — see `benilla_world`'s `liquid::ripple`, which owns the
/// simulation and writes this every frame. Only the first two rows are read here; the rest is the
/// wave step's own state and is declared so the two shaders agree on the layout.
///
/// **`stamps` is one ABI with `ripple.rs`'s `MAX_STAMPS` and `liquid_ripple.wgsl`'s — THREE copies
/// of one number.** This one once sat at 16 while the other two moved to 48, survivable then only
/// because this program read `placement` and `params` and stopped — a declared struct SMALLER than
/// the buffer binds and runs, so the mismatch produced no error anywhere. fork: it now reads the sky
/// rows PAST `stamps`, where a drift would silently misread them, so `ripple.rs`'s
/// `both_shaders_declare_the_blob_this_packs` pins all three copies.
struct Ripple {
    // xy = the map's min corner in world XZ; z = 1/extent; w = interpolation between the two
    // packed heights.
    placement: vec4<f32>,
    // x = one texel in UV; y = the slope scale; z = enabled; w = damping.
    params: vec4<f32>,
    prev: vec4<f32>,
    stamp_count: vec4<f32>,
    stamps: array<vec4<f32>, 48>,
    // fork: modern lighting — the sky dome's five stops (zenith → 1.8°) and its horizon colour,
    // exactly as the dome draws them (`sky::dome_stops`: linear on a modern frame). **This program
    // now reads past `stamps`, so the 48 above is load-bearing here**; `ripple.rs`'s tests pin it.
    sky: array<vec4<f32>, 6>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(91) var<storage, read> ripple: Ripple;

/// fork: modern lighting — the sky dome's colour along a direction: `sky.wgsl`'s elevation gradient,
/// piecewise-linear through the five stops at their geometric elevations and the fog colour at and
/// below the horizon (the dawn/dusk azimuth warp and the sun's glow are left to the dome — the sun
/// itself reaches the water as the glitter). In the reflection's lane: gamma, like the mirror read.
///
/// The dome is camera-centred and world-aligned, so this is exact for the backdrop whatever the
/// mirror is doing — the environment the water reflects when the mirror has no picture of its own.
fn sky_dome(dir: vec3<f32>) -> vec3<f32> {
    let elev = degrees(asin(clamp(dir.y, -1.0, 1.0)));
    let s = ripple.sky;
    var col: vec3<f32>;
    if (elev <= 0.0) {
        col = s[5].rgb;
    } else if (elev < 1.8) {
        col = mix(s[5].rgb, s[4].rgb, elev / 1.8);
    } else if (elev < 3.7) {
        col = mix(s[4].rgb, s[3].rgb, (elev - 1.8) / (3.7 - 1.8));
    } else if (elev < 9.8) {
        col = mix(s[3].rgb, s[2].rgb, (elev - 3.7) / (9.8 - 3.7));
    } else if (elev < 16.8) {
        col = mix(s[2].rgb, s[1].rgb, (elev - 9.8) / (16.8 - 9.8));
    } else {
        col = mix(s[1].rgb, s[0].rgb, (elev - 16.8) / (90.0 - 16.8));
    }
    return colorspace::modern_in(col, wow_light.point_count.y);
}

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
    // fork: modern lighting (FORK.md, P4) — a modern frame fogs from the volume (`fragment`).
    if (fog_color.w <= 0.5 || wow_light.point_count.y > 0.5) {
        return rgb;
    }
    let eye_z = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    let denom = max(fog_span.y - fog_span.x, 0.001);
    let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
    return mix(fog_color.xyz, rgb, factor);
}

/// Where the mirror starts and stops being about this surface — the fade in guard 2 above, in
/// yards of height between the fragment and the plane `liquid::reflect` mirrored about.
///
/// **Both are ours**, with no reference value to take: the reference has no single-plane mirror to
/// guard, because it re-renders per instance group. They are chosen against what the mirror is
/// actually for — a lake's own surface is flat to well under a yard, so the start is far enough out
/// that no real body of water fades against its own ripple, and the end is inside the height at
/// which a mirror about the wrong plane stops resembling the world at all.
const MIRROR_PLANE_FADE_START: f32 = 2.0;
const MIRROR_PLANE_FADE_END: f32 = 12.0;

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

    // The foam UV pair — `ProcWater`'s `result.texcoord[2]`, read off the shipped `arbvp1` program:
    //
    //     texcoord[2].xy = base · 228 + sincos · 0.1
    //     texcoord[2].zw = base · 228 + sincos · 0.05 + 0.5
    //
    // **One scale, not two.** Both taps sample the sheet at the same frequency; what separates them
    // is a half-texture offset and a wobble at HALF the amplitude, so the pair beats against itself
    // at the wobble's own slow rate. This used to give the second tap 0.73× the first's scale, which
    // is a different thing entirely: two frequencies 27 % apart sum into a coarse moiré whose period
    // is far longer than either, and that beat — not the sheet — was the size of benilla's surf.
    // (The 228 is against the reference's own `attrib[6]/32` liquid coordinate, whose world scale is
    // authored data this client has no copy of; `foam_params.x` is that number's stand-in, anchored
    // to world XZ like the bump sets.)
    let fs = w.foam_params.x;
    let wob = vec2<f32>(
        sin(phase * 4.0 + world_uv.x * 0.05),
        cos(phase * 2.0 + world_uv.y * 0.05),
    ) * 0.1;
    out.foam_uv = vec4<f32>(world_uv * fs + wob, world_uv * fs + wob * 0.5 + 0.5);

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
    let a = textureSample(bump_tex, art_samp, in.bump_uv_a.xy).xyz;
    let b = textureSample(bump_tex, art_samp, in.bump_uv_a.zw).xyz;
    let c = textureSample(bump_tex, art_samp, in.bump_uv_b.xy).xyz;
    let d = textureSample(bump_tex, art_samp, in.bump_uv_b.zw).xyz;
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
    let na = textureSample(noise_vol, art_samp, vec3<f32>(in.bump_uv_a.xy, slice)).rgb;
    let nb = textureSample(noise_vol, art_samp, vec3<f32>(in.bump_uv_a.zw, slice)).rgb;
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

// fork: modern lighting — the waterline: the water (yd along the view ray) the reflection takes to
// come in, so a surface meets its bank as the bed seen through nothing rather than as a drawn edge.
const MODERN_SHORE_FADE: f32 = 1.0;
// fork: modern lighting — the surf on the experimental tier: a soft bright band along the waterline,
// the reference shot's (a faint lift at its peak, fading out over under a yard of water, no sheet
// detail in it). `WIDTH` scales the classic foam depth; `MOTTLE` is the sheet's frequency when it is
// read only to keep the band from being a uniform stripe — low enough that magnification blurs it.
const MODERN_FOAM_GAIN: f32 = 0.04;
const MODERN_FOAM_WIDTH: f32 = 1.6;
const MODERN_FOAM_MOTTLE: f32 = 0.04;
// fork: modern lighting — the experimental tier's waves: the vertical depth (yd) by which a surface
// has its full swell back, and the most water (yd along the ray) the refraction bends through.
const MODERN_WAVE_FULL_DEPTH: f32 = 3.0;
const MODERN_REFRACT_MAX_PATH: f32 = 6.0;
// fork: modern lighting — the experimental tier's wind ripples: how many trains (only the long, high-
// amplitude end of the ladder, so water seen up close keeps the Modern swell), the rungs the ladder
// was cut from, the longest wavelength (yd), the step down to the next, how much the long ones outweigh the short (slope ∝ wavelength to
// this power), the share of deep-water dispersion they travel at, and the phase change per pixel
// (radians) over which a train fades. The fade is the reference shot's: at every distance the finest
// octave left has a ~7-13 px period along the view, which puts its start near 2π/12 and its end near
// 2π/5.
const RIPPLE_TRAINS: u32 = 4u;
const RIPPLE_LADDER: f32 = 12.0;
const RIPPLE_LONGEST: f32 = 12.0;
const RIPPLE_STEP: f32 = 0.78;
const RIPPLE_TILT: f32 = 0.35;
const RIPPLE_PACE: f32 = 0.6;
const RIPPLE_ALIAS_START: f32 = 0.5;
const RIPPLE_ALIAS_END: f32 = 1.2;

/// fork: modern lighting — **the wind ripples**, a slope layer under the swell, normalised to a unit
/// RMS slope (the caller scales it by `look_c.w`). Nothing shades them: they tilt the normal, and the
/// Fresnel, the reflection and the glint the surface already has turn a face toward the eye into the
/// water's own dark and a face away into the sky it reflects — at a grazing look, foreshortened into
/// the reference shot's streaks laid across the view.
///
/// The long end of a sea's spectrum, sampled: `RIPPLE_TRAINS` trains a geometric step apart from
/// `RIPPLE_LONGEST` down (~12 to ~6 yd; the finer rungs were too busy up close), the long ones a little steeper (`RIPPLE_TILT`). Their headings scatter about the wind by the
/// golden angle, the long ones within ~50° of it and the short ones from every side. Two noise taps
/// (~16 yd and ~5 yd) bend each train's crests and gather it into wave groups of its own, so no two
/// patches of water carry the same pattern.
///
/// Each train fades on the pixel's LARGER world footprint (`foot_x`/`foot_y`), whatever its heading,
/// as a mip level is chosen: at a grazing look that is the foreshortened extent along the view, so
/// the octaves finer than it go across the view too, and the ones that stay are drawn out into the
/// reference shot's long streaks (~90 % of its ripple power has crests within 20° of across the
/// view, 6-15 times longer than they are deep). The many octaves are what keeps that true at every
/// distance: there is always one at the edge of what the pixels resolve.
fn wind_ripples(xz: vec2<f32>, foot_x: vec2<f32>, foot_y: vec2<f32>) -> vec2<f32> {
    let t = anim_time();
    let coarse = textureSampleLevel(noise_vol, art_samp, vec3<f32>(xz * 0.015 + vec2<f32>(t * 0.004, 0.0), fract(t * 0.003)), 0.0).rgb - vec3<f32>(0.5);
    let fine = textureSampleLevel(noise_vol, art_samp, vec3<f32>(xz * 0.05 - vec2<f32>(0.0, t * 0.01), fract(t * 0.006 + 0.5)), 0.0).rgb - vec3<f32>(0.5);
    let wd = sunshadow::wind_dir();
    var slope = vec2<f32>(0.0);
    var energy = 0.0;
    let foot = max(length(foot_x), length(foot_y));
    for (var i = 0u; i < RIPPLE_TRAINS; i = i + 1u) {
        let fi = f32(i);
        let octave = fi / RIPPLE_LADDER;
        let wl = RIPPLE_LONGEST * pow(RIPPLE_STEP, fi) * (1.0 + 0.08 * sin(fi * 5.1));
        let turn = mix(0.55, 1.0, octave) * 1.5707963 * (fract(fi * 0.618034 + 0.13) * 2.0 - 1.0);
        let dir = vec2<f32>(wd.x * cos(turn) - wd.y * sin(turn), wd.x * sin(turn) + wd.y * cos(turn));
        let kw = 6.2831853 / wl;
        let omega = sqrt(10.7 * kw) * RIPPLE_PACE;
        // The long trains follow the coarse field, the short ones the fine one.
        let n = mix(coarse, fine, octave);
        let basis = vec3<f32>(cos(fi * 2.4), sin(fi * 2.4), cos(fi * 3.7));
        let shift = dot(n.rg, basis.xy) * 3.2;
        let group = clamp(1.0 + 2.6 * dot(n, basis.yzx), 0.1, 1.9);
        let phase = dot(dir, xz) * kw + shift - omega * t + fi * 1.37;
        let fade = 1.0 - smoothstep(RIPPLE_ALIAS_START, RIPPLE_ALIAS_END, kw * foot);
        let weight = pow(wl / RIPPLE_LONGEST, RIPPLE_TILT);
        // Crest sharpened as `sunshadow::wind_swell`'s.
        let c = cos(phase);
        slope = slope + dir * (weight * group * fade * c * (1.0 + 0.35 * sin(phase)));
        energy = energy + weight * weight;
    }
    // The unfaded layer's RMS slope (a half per train from the cosine), so a faded train only calms the
    // water rather than being renormalised back up.
    return slope / sqrt(energy * 0.5);
}

/// fork: modern lighting — **stylised clear water**, above the water, in linear radiance. Replaces
/// the classic COMBINE on a modern frame at the Modern (Experimental) water tier (`waterDetail 3`,
/// `w.optics.w`; plain Modern keeps the combine) — not its reflection, which arrives here as `refl`
/// exactly as the classic lane reads it (the planar mirror on its short bent ray, the trace over it
/// on the flattened normal).
///
/// The target is WoW Forever's river: a bed you can see from the bank out, taking the water's hue
/// from the first inches, with the bank's shadows carried across it unbroken, and hardly any mirror
/// in it. Two of the classic combine's terms stand between the frame and that — the body is the
/// swatch painted over the refraction at the swatch's own opacity (0.5 at a river's waterline, so
/// every shore ends milky, and a zone whose shallow row is brown — the Southfury's 124, 91, 70 —
/// reads as tea), and its depth is the MCLQ vertex byte, which knows nothing of the bed. A purely
/// physical surface over the same bed goes too far the other way: a yard of river is ~93 % clear,
/// so the water all but vanishes, and its Fresnel mirror reaches ~100 % at a grazing angle and reads
/// as chrome.
///
/// So, per channel over the water actually in front of the bed (the depth copy knows how much):
///
/// * the bed is **filtered toward the water's hue** — the DEEP row, normalised to its brightest
///   channel — by `look_a.x`, coming in over the first `look_a.y` yards so the waterline stays soft.
///   A filter multiplies: the bed keeps its contrast, and its shadows show through as the water's
///   own dark colour rather than being washed out by an overlay;
/// * then **Beer–Lambert** toward the DEEP row under the surface's light (the colour the classic
///   swatch reaches at depth), so a channel darkens into the zone's water and the clock keeps moving
///   it;
/// * the reflection goes over that at a stylised Fresnel — a raised head-on reflectance
///   (`look_a.z`) climbing to a capped grazing one (`look_c.x`) — and is itself pulled `look_a.w` of
///   the way to the body — the reference combine's `c11.y` idea, which is what keeps
///   water painterly rather than glossy.
fn modern_water(
    in: VsOut,
    n: vec3<f32>,
    v: vec3<f32>,
    bed_raw: vec3<f32>,
    bed_z: f32,
    frag_view_z: f32,
    deep: vec3<f32>,
    lit: vec3<f32>,
    refl: vec3<f32>,
) -> vec3<f32> {
    // `scene_depth` is planar view depth, so the bed's distance along this pixel's ray scales off
    // the surface's by the same ratio. The light that lit the bed crossed the column on its way
    // down as well, so the vertical depth is added once more.
    let path = max(bed_z - frag_view_z, 0.0) * length(in.view_vec) / max(frag_view_z, 0.001);
    let column = path * clamp(-v.y, 0.0, 1.0);
    let trans = exp(-w.optics.xyz * min(path + column, 1000.0));
    // The scene copy is linear on a modern frame and the frame's own gamma bytes on a classic one;
    // this combine runs in linear either way. The bed and its depth are the refraction's own read
    // (`fragment_classic`), shared with the reference half of the blend rather than taken twice.
    let bed = select(colorspace::srgb_to_linear(bed_raw), bed_raw, wow_light.point_count.y > 0.5);
    let scatter = colorspace::srgb_to_linear(deep * lit);
    let deep_lin = colorspace::srgb_to_linear(deep);
    let hue = deep_lin / max(max(deep_lin.r, max(deep_lin.g, deep_lin.b)), 1e-4);
    let tint = mix(vec3<f32>(1.0), hue, w.look_a.x * smoothstep(0.0, max(w.look_a.y, 0.01), column));
    let under = bed * tint * trans + scatter * (vec3<f32>(1.0) - trans);

    // A stylised Fresnel, the reference combine's shape rather than Schlick's: a raised head-on
    // reflectance (`look_a.z`) climbing only to a cap (`look_c.x`) at grazing, along `(1 − cos)` to
    // the `look_c.y` power, and faded in over the first few inches of water so the surface meets
    // the bank as the bed rather than as a line of reflection laid along it. Water's real curve is
    // the wrong one to draw: 2 % head-on leaves a river looked down on with no sky in it at all,
    // and its climb to ~100 % at grazing turns every stretch of water toward the horizon into a
    // solid sheet — mercury, not water. The reference combine runs `0.1 + 0.5 · fres²`.
    let f0 = clamp(w.look_a.z, 0.0, 1.0);
    let cos_i = clamp(dot(-v, n), 0.0, 1.0);
    let rise = pow(1.0 - cos_i, max(w.look_c.y, 0.5));
    let fres = mix(f0, clamp(w.look_c.x, f0, 1.0), rise) * smoothstep(0.0, MODERN_SHORE_FADE, path);
    return mix(under, mix(refl, scatter, w.look_a.w), fres);
}

fn fragment_classic(in: VsOut) -> vec4<f32> {
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

    // The water column under this pixel, read before the surface has a normal — `textureLoad`, so no
    // uniform control flow is asked of it. `ray_len` is the water crossed along the unrefracted ray
    // (the depth copy is planar, so the bed's distance scales off the surface's by their ratio) and
    // `column_v` roughly how deep that is.
    let uv = screen_uv(in.clip_position);
    let scene_z = scene_depth_at(uv);
    let v = normalize(in.view_vec);      // camera → fragment
    let view_len = length(in.view_vec);
    let ray_len = max(scene_z - frag_view_z, 0.0) * view_len / max(frag_view_z, 0.001);
    let column_v = ray_len * clamp(-v.y, 0.0, 1.0);
    let below = w.path.y > 0.5;
    let modern = wow_light.point_count.y > 0.5;
    let modern_above = modern && !below;
    // fork: modern lighting — above the water at the Modern (Experimental) tier the combine is
    // `modern_water`'s, under either lighting model: the tier is the water's, not the frame's. What
    // only a modern frame has (the wind sea, the cascades, the HDR glitter, the froxel fog) it takes
    // there; a classic frame keeps its own fog and the reference's sun specular.
    let experimental = !below && w.optics.w > 0.5;
    // fork: modern lighting — the experimental tier's wind ripples, read here in uniform control flow
    // for the pixel footprint's derivatives. Only the sea carries them; a river, a lake and a WMO's
    // liquid keep the cata bump's own octaves.
    var ripples = vec2<f32>(0.0);
    if (experimental && w.kind.y > 0.5 && w.look_c.w > 0.0) {
        ripples = w.look_c.w * wind_ripples(
            in.world_position.xz, dpdx(in.world_position.xz), dpdy(in.world_position.xz),
        );
    }

    // The swell, plus whatever a body wading nearby has pushed into the water. The two are summed
    // as SLOPES and normalised once — see `bump_gradient`.
    var swell = bump_gradient(in);
    // fork: modern lighting — a modern frame adds a wind-driven swell under the authored bump.
    var wind = vec2<f32>(0.0);
    if (modern) {
        // Two slow drifting taps of the noise volume: one warps the wave phases, one is the gust
        // field, so no two stretches of water repeat the same pattern.
        let wn = in.world_position.xz;
        let wt = globals.time;
        let warp_n = textureSampleLevel(noise_vol, art_samp, vec3<f32>(wn * 0.011 + vec2<f32>(wt * 0.004, 0.0), fract(wt * 0.003)), 0.0).rgb;
        let gust_n = textureSampleLevel(noise_vol, art_samp, vec3<f32>(wn * 0.0045 - sunshadow::wind_dir() * wt * 0.0012, fract(wt * 0.002 + 0.5)), 0.0).r;
        let warp = warp_n.rg * 2.0 - vec2<f32>(1.0);
        let gust = smoothstep(0.25, 0.75, gust_n);
        wind = sunshadow::wind_swell(wn, wt, view_len, warp, gust);
        if (!experimental) {
            swell = swell + wind;
        }
    }
    if (experimental) {
        // The experimental tier's water is calm: the sea keeps its whole sea, a river or lake
        // `look_b.x` of it (its authored bump is its own) with the wind's ripples over it, and water
        // with little under it goes near glassy — a hand-deep stream does not carry a 17-yard swell.
        let keep = select(w.look_b.x, 1.0, w.kind.y > 0.5);
        swell = (swell + wind * keep + ripples) * mix(w.look_b.y, 1.0, smoothstep(0.0, MODERN_WAVE_FULL_DEPTH, column_v));
    }
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
    // fork: modern lighting — the sun's share of the water's light falls under the cascades and the
    // cloud shadow, so water in a tree's or a cliff's shade reads shaded. `water_sun` is the sun's
    // reach at the surface, which the highlights take; `water_lit` the body's shaded light.
    var water_sun = 1.0;
    var water_lit = in.lit_color;
    if (modern) {
        // **The shadow is read where it falls — on the bed, not on the surface.** A shadow is a
        // property of what the light lands on, and the bed seen through the water already carries its
        // own in the scene copy; a second lookup at the surface puts a second edge on the water,
        // offset from the bed's by the depth over the sun's slope (and, while it rode the waves, a
        // wobbling one). So the probe walks down the unrefracted view ray to the bed and the body's
        // shade lands on the bed's shadow: one edge, carried across the water. The bed probe stops
        // `BED_LIFT` yd short of the bed (so the bed does not shadow itself) and is world-fixed ground,
        // so the shade holds still as the camera moves.
        //
        // Where the bed lies too deep to show (a column of `WATER_SHADOW_PROBE` to twice that), the
        // shade hands over to the surface point itself, again world-fixed. Every point under the
        // surface was the wrong one: a sun ray from it can be cut by what is UNDER the water — a
        // swimmer's own legs put a crisp copy of the body on the water, yards off it — and one dropped
        // by the column over a shelving beach landed in the sand, a dark band along the shore. From
        // the surface the ray only rises, so only what stands out of the water shades it. From below
        // there is no bed along the ray; the surface is the probe there too. The fade goes by the
        // vertical column, never the ray, which is long over a hand-deep shore.
        //
        // The sun's highlights (glare, specular, glint) are the sun mirrored off the surface itself, so
        // they take the surface point's visibility whatever the depth, and none of it in full shade.
        let lifted = in.world_position.xyz + vec3<f32>(0.0, 0.05, 0.0);
        let sun_surface = sunshadow::sun_visibility_load(modern_shadow_map, modern_block.cascades, lifted, frag_view_z);
        let deep = select(smoothstep(WATER_SHADOW_PROBE, 2.0 * WATER_SHADOW_PROBE, column_v), 1.0, below);
        var body_sun = sun_surface;
        if (deep < 1.0) {
            let probe_d = max(ray_len - WATER_SHADOW_BED_LIFT, 0.0);
            // The probe sits on this pixel's ray, so its planar depth scales with the distance along it.
            let probe_z = frag_view_z * (1.0 + probe_d / max(view_len, 0.001));
            let sun_bed = sunshadow::sun_visibility_load(modern_shadow_map, modern_block.cascades, lifted + v * probe_d, probe_z);
            body_sun = mix(sun_bed, sun_surface, deep);
        }
        let wct = sunshadow::cloud_taps(modern_block.fog.cloud, lifted);
        let cloud = sunshadow::cloud_mix(wct, modern_block.cloud[wct.idx.x], modern_block.cloud[wct.idx.y], modern_block.cloud[wct.idx.z], modern_block.cloud[wct.idx.w]);
        // The shade keeps some of the light: skylight still reaches open water from every side.
        // Water is not a solid receiver: light scatters in the body under the shadow too, so the shade
        // keeps most of the lit colour.
        let shade = max(min(in.lit_color, wow_light.light_ambient.rgb), in.lit_color * WATER_SHADE_KEEP);
        water_lit = mix(shade, in.lit_color, body_sun * cloud);
        water_sun = sun_surface * cloud;
    }
    let body = ramp.rgb * water_lit;

    // ---- Refraction ----------------------------------------------------------------------------
    //
    // Offset the screen read by the bump, scaled by how much water is actually in front of what is
    // behind it, then GUARD the result: if the offset lands on something nearer than this water
    // surface, fall back to the unoffset read. Without that guard a wave at the shoreline bends the
    // riverbank *over* the water — the classic refraction bleed, and the reference spends a `cmp`
    // on it for the same reason.
    let depth_ratio = clamp((scene_z - frag_view_z) / max(frag_view_z, 0.001) * 100.0, 0.0, 1.0);
    let offset_amount = clamp((1.2 - fres2) * depth_ratio, 0.0, 1.0);
    // **The NORMAL bends the read, not the slope.** `MULR R3.zw, R2.xyxy, R3.z` then
    // `MULR R5.zw, R3, c[33].z` — `R2` is the normalised shading normal, `c33.z` is 0.1. Driving
    // this off `grad` instead skips the `c23.y` perturb and the renormalise, which at the shipped
    // 0.55 perturb is very nearly a doubling: the water bent what was behind it about twice as hard
    // as the reference does, and disagreed with the surface's own lighting while doing it.
    var refract_uv = uv + n.xz * 0.1 * offset_amount;
    if (experimental) {
        // fork: modern lighting — the experimental tier bends the bed by what the water actually
        // does: the bed's sideways shift is the water crossed times the tilt the surface puts on the
        // ray (`look_b.z` per yard; ~0.25 is glass-flat physics for water), projected to the screen
        // at this depth. The reference's `depth_ratio` saturates at a hundredth of the view distance,
        // so a hand-deep stream bent its bed as hard as a lake — the wobble that makes a riverbed
        // look like glass. Never more than the reference's own bend.
        let per_yd = 0.5 * view.clip_from_view[1][1] / max(frag_view_z, 0.5);
        let bend = min(w.look_b.z * min(ray_len, MODERN_REFRACT_MAX_PATH) * per_yd, 0.1 * offset_amount);
        refract_uv = uv + n.xz * bend;
    }
    var bed_z = scene_depth_at(refract_uv);
    if (bed_z < frag_view_z) {
        refract_uv = uv;
        bed_z = scene_z;
    }
    let bed_raw = textureSampleLevel(scene_color, scene_samp, refract_uv, 0.0).rgb;
    var refracted = colorspace::modern_in(bed_raw, wow_light.point_count.y);
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
    //   * the **view-distance fade**, `1.1 − saturate(d / 150)` — its own term, NOT the bump taps'
    //     `saturate(0.001·d + 0.2)`. Surf is loudest where you are standing in it and gone by 150
    //     yards. This carried the bump's blend for a while, which is the same law read backwards:
    //     a fifth strength underfoot climbing to full at the horizon, i.e. a faint smear at the
    //     shore you are standing on and a bright rim on every lake in the distance.
    //   * **N.up** — foam lies flat. A wave face turned toward you carries less of it.
    let detail_blend = 1.1 - clamp(length(in.view_vec) * (1.0 / 150.0), 0.0, 1.0);
    // Two taps, ADDED, and **both dragged by the wave normal** — which is where the dynamic ripple
    // reaches the surf, the normal being the one term the displacement field feeds. The reference's
    // two drags are deliberately unequal: `−N.xy` at full strength on the first and `+N.yx · 0.2` on
    // the second, opposed and swizzled, so a wave pulls the two layers apart instead of sliding them
    // together. Only the second was dragged here, so half the foam sat still while the water moved
    // under it — a raindrop or a wading leg perturbed the water and left the surf where it was.
    let foam_a = textureSample(foam_tex, art_samp, in.foam_uv.xy - n.xz).rgb;
    let foam_b = textureSample(
        foam_tex,
        art_samp,
        in.foam_uv.zw + n.zx * w.foam_params.w,
    ).rgb;
    var foam = (foam_a + foam_b) * (shore * w.foam_params.z * detail_blend * n.y);

    if (experimental) {
        // fork: modern lighting — the surf as a soft band rather than the sheet's broken water: the
        // same column mask, `MODERN_FOAM_WIDTH` times as wide, lit like the surface it sits on, and
        // mottled only by the sheet read at a frequency its texels magnify into blur.
        let soft_t = clamp(column * slant / (foam_depth * MODERN_FOAM_WIDTH), 0.0, 1.0);
        let band = 1.0 - soft_t * soft_t * (3.0 - 2.0 * soft_t);
        let mottle = textureSampleLevel(foam_tex, art_samp, in.foam_uv.xy * MODERN_FOAM_MOTTLE, 0.0).r;
        foam = water_lit * (band * MODERN_FOAM_GAIN * (0.6 + 0.8 * mottle) * detail_blend);
    }
    var rgb: vec3<f32>;
    if (below) {
        // ---- ProcWaterBelow perm 0 -------------------------------------------------------------
        //
        // No reflection: from underneath there is nothing above the surface but sky, and the
        // underwater fog block already stands for it. What is left is the refraction — which from
        // below is the world seen up through the surface — carrying the specular and the glare.
        rgb = refracted;
    } else {
        // ---- ProcWaterAbove perm 0: the planar mirror, with the screen-space trace over it -----
        //
        // **Perm 0, not perm 4, is what a `waterDetail 2` player runs.** `CRenderer::Flush` picks
        // the program's `reflectMode` as `v29 = renderer[+0x30]` — 0 for the ordinary renderer, 1
        // for the real-reflection one — and only then `if (!v29 && instance->settings[+0x24] == 2)
        // v29 = 2`. So mode 2 (perm 4/5) is a minority arm gated on one liquid class, mode 1 (perm
        // 2/3) is the `waterDetail > 2` tier, and mode 0 (perm 0/1) is the default. The three are
        // told apart by their sampler sets in the shipped container: 0 takes `{0,1,2,3,5,6}`, 1
        // takes `{0,1,2,5,6}`, 2 takes `{0,2,3,5,6}` — mode 0 is the only one that reads BOTH the
        // reflection buffer (`samp1`) and the blurred back buffer (`samp3`).
        //
        // What that buys, and why this port's first cut lost reflections whenever the camera
        // pitched down: perm 4's reflection is the screen-space trace ALONE, with a flat horizon
        // constant where the trace leaves the frame. Pan down and the trace always leaves the frame
        // — `saturate(0.75 − r.up²)` is exactly the term that says so — so the water went flat.
        // Perm 0 has the mirror underneath it and the trace only layered on top, so the pitch that
        // kills the trace reveals the render rather than a constant.
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
        //   * the screen-edge fade `saturate(1 − ndc.x⁴) · saturate(1 − ndc.y⁴)`, so a reflection
        //     does not pop as its source slides off the frame;
        //   * `saturate(0.75 − r.up²)`, which kills the trace as the ray tips toward the sky —
        //     where there is nothing in the back buffer to find.
        //
        // **The fade is QUARTIC, and the exponent is the whole of it.** The program squares the NDC
        // pair, multiplies that by the pair again, and only then subtracts from one — `MULR R0.xy,
        // R0.zwzw, R0.zwzw` / `MULR R0.xy, R0, R0.zwzw` / `MADR_SAT R1.w, -R0.y, R0, 1`, and the
        // D3D build spells the same thing with `mad r4.xy, r4.yxzw, -r4.yxzw, 1` over an already
        // squared register. A quadratic fade instead is 1 − x² where the reference has 1 − x⁴, so it
        // is at or below the real curve everywhere and worst in between: two thirds of the way to the
        // screen edge it passes 0.56 of the reflection where the reference passes 0.80, and at half
        // way 0.75 against 0.94. The reflection was there, correctly weighted, and then thrown away
        // over most of the frame.
        let q = ndc * ndc;
        let q2 = q * q;
        let edge = clamp(1.0 - q2.x, 0.0, 1.0) * clamp(1.0 - q2.y, 0.0, 1.0);
        let skyward = clamp(0.75 - r.y * r.y, 0.0, 1.0);
        let validity = clamp(skyward * edge * 1.33333, 0.0, 1.0);
        let refl_uv = (ndc * vec2<f32>(1.0, -1.0) + 1.0) * 0.5;
        let hit = colorspace::modern_in(textureSampleLevel(scene_blur, scene_samp, refl_uv, 0.0).rgb, wow_light.point_count.y);
        // **The mirror's own lookup — a SECOND ray, and a much shorter one.** The program does not
        // read `samp1` at the fragment's own screen position, which is what a textbook planar
        // reflection would do; it projects a point 20 yards along
        //
        //     ray = V + saturate(V·N · 0.45 + 0.55) · (−N_horizontal − V)
        //
        // (`MADR R4.xyz, -R0, c[36].xxyw, -R1` builds `−N_h − V` with the UP component zeroed by
        // `c36.y = 0`; `MADR_SAT R1.w, R1.w, c[35].y, c[35].z` is the weight; `c35.w = 20` is the
        // distance). Head-on the weight is ~0.55 and the ray leans off the view direction toward
        // the wave's own downhill; at a grazing angle it drops toward 0.45·V. That lean IS the
        // mirror's distortion — the buffer is a flat render about `z = 0`, so a surface that is not
        // at `z = 0`, or not flat, has to be paid for by displacing the read rather than by
        // re-rendering per surface.
        //
        // **`1 − u`, and it is ours, not the reference's.** The reference mirrors its camera with a
        // plain `C4Plane::Mirror`, which is a reflection matrix; a Bevy `Transform` holds a `Quat`
        // and cannot, so `liquid::reflect` builds the mirrored camera as a proper rotation — which
        // costs a flip of the camera's local X, and therefore a horizontal flip of the picture.
        // Undoing it here is one subtract, against a second camera that could not exist otherwise.
        let bend = clamp(dot(v, n) * 0.45 + 0.55, 0.0, 1.0);
        let mirror_ray = v + bend * (-vec3<f32>(n.x, 0.0, n.z) - v);
        let mirror_clip = view.clip_from_world * vec4<f32>(in.world_position.xyz + mirror_ray * 20.0, 1.0);
        let mirror_ndc = mirror_clip.xy / max(mirror_clip.w, 0.0001);
        var mirror_uv = (mirror_ndc * vec2<f32>(1.0, -1.0) + 1.0) * 0.5;
        mirror_uv.x = 1.0 - mirror_uv.x;

        // ---- The two guards on the mirror read, and BOTH are ours ------------------------------
        //
        // The reference needs neither, because it re-renders its mirror per liquid instance group
        // and clamps at the border of a buffer whose frustum is the water's own. This port renders
        // ONE mirror per frame about ONE plane, so it has two ways to be handed a picture that was
        // never about this fragment — and a planar reflection that is confidently wrong is worse
        // than none, because it moves with the camera and reads as a ghost sliding under the
        // surface rather than as a flat sheen.
        //
        // **Guard 1: the read left the mirror's frame.** `mirror_uv` is a projection of a bent ray
        // and lands outside `[0, 1]` at the margins. The sampler is ClampToEdge (every screen
        // target here is — see `liquid/screen.rs`), so out of range does not read black, it reads
        // the border texel smeared across the whole margin: one column of the mirror painted over
        // yards of water. The fade is the same quartic shape the program already spends on the
        // screen-space trace, for the same reason, on this read's own NDC.
        let mq = mirror_ndc * mirror_ndc;
        let mq2 = mq * mq;
        let mirror_edge =
            clamp(1.0 - mq2.x, 0.0, 1.0) * clamp(1.0 - mq2.y, 0.0, 1.0)
            * step(0.0, mirror_clip.w);

        // **Guard 2: this surface is not at the mirror's plane.** One mirrored render can only be
        // about one height, and `liquid::reflect` puts it at the water the eye is over. A surface
        // somewhere else — the lake below the waterfall, a fountain two storeys up, the ocean seen
        // from an inland pond — would wear a picture of the world reflected about someone else's
        // waterline, which is exactly the ghost this guard exists for. It fades over yards rather
        // than cutting, so a stepped pool does not show a hard line between its tiers.
        let plane_gap = abs(in.world_position.y - w.path.w);
        let mirror_height = 1.0 - smoothstep(MIRROR_PLANE_FADE_START, MIRROR_PLANE_FADE_END, plane_gap);
        let mirror_valid = mirror_edge * mirror_height;
        // Where the mirror is standing down — the eye submerged, or the legacy tier, the
        // reference's own `HasFakeReflection` conditions — the horizon takes its place. `c21` in
        // the program is a constant the client uploads per zone; benilla's equivalent is the fog
        // colour, which in WoW IS the horizon: the sky dome converges on it by construction, so a
        // reflection that falls back to it falls back to something the player can see is right.
        // This is also exactly what perm 4 has in place of the mirror, so the fallback is not a
        // guess — it is the other shipped program.
        var base = wow_light.fog_color.rgb;
        if (experimental) {
            // fork: modern lighting — the experimental tier keeps this whole lookup and puts the sky
            // dome along the reflected ray in the horizon constant's place: one colour for every
            // direction (in a jungle zone an olive haze) read as goo over a lake whose mirror had not
            // rendered yet; the dome is the backdrop the player can see.
            base = sky_dome(r);
        }
        if (w.path.z > 0.5) {
            let mirror = colorspace::modern_in(textureSampleLevel(reflect_map, scene_samp, mirror_uv, 0.0).rgb, wow_light.point_count.y);
            base = mix(base, mirror, mirror_valid);
        }
        // `ADDR R3.xyz, R3, -R5` / `MADR R3.xyz, R1.w, R3, R5` — the mirror is the BASE and the
        // screen-space hit is lerped over it by the same validity perm 4 lerps the horizon by.
        var reflected = mix(base, hit, validity);
        if (experimental) {
            // fork: modern lighting — the same reflection, softened toward the body and laid over
            // `modern_water`'s tinted bed at its stylised Fresnel instead of the two steps below.
            // Linear out, re-encoded so the wrapper's decode hands it back unchanged.
            let clear = colorspace::linear_to_srgb(modern_water(
                in, n, v, bed_raw, bed_z, frag_view_z, swatch(1.0).rgb, water_lit,
                colorspace::srgb_to_linear(reflected),
            ));
            // The tier sits between the clear surface and the reference's combine: `look_c.z` of
            // the reference's own (the swatch over the refraction, the reflection pulled to the
            // body), over the same calm normal, the same reflection, and the same bed read.
            let cata = mix(refracted, mix(reflected, body, 0.6), fres2 * 0.5 + 0.1);
            rgb = mix(clear, cata, w.look_c.z);
        } else {
            // Then 60% of the way back toward the body colour. Water is not a mirror; this is the
            // program's own `c11.y`, and it is what stops a lake from reading as chrome at noon.
            reflected = mix(reflected, body, 0.6);
            rgb = mix(refracted, reflected, fres2 * 0.5 + 0.1);
        }
    }

    // The sun's broad glare on the surface, modulated by the body so it cannot brighten water that
    // is already dark, and the specular, weighted by the Fresnel square so it only fires at grazing
    // angles.
    //
    // **The glare wears the DIFFUSE colour and the specular wears the specular one** — `MULR R2.xyz,
    // R0.w, c[3]` against `MULR R1.xyz, R1.x, c[4]`, two different registers. `SetShaderLighting`
    // fills the pixel block straight off the light record: `c2` ambient, `c3` diffuse, `c4` specular
    // with the Blinn exponent in `.w`. Running both terms off the specular row made the broad sun
    // track on the water take a colour the sun itself never has at that hour.
    //
    // fork: modern lighting — neither is on `modern_water`'s surface (the experimental tier): the
    // glare is the body colour lit a second time (the brown wash over a Durotar river at noon), and
    // the specular's sheen floor is a flat lift at every grazing angle that the Fresnel reflection
    // carries for real there. The sun itself is the glint below.
    if (!experimental) {
        let glare = clamp(dot(to_light, n), 0.0, 1.0) * w.shade_params.w;
        rgb += glare * wow_light.light_diffuse.rgb * body * water_sun;
        // Below the surface the specular is weighted by the Fresnel square TWICE and by the swatch's
        // opacity — `ProcWaterBelow` spends `MULR R1.xyz, R1, R3.w` (fres²·ramp.a) and then
        // `MULR R1.xyz, R1, R2.w` (fres² again) where `ProcWaterAbove` spends one `MULR R1.xyz, R1,
        // R1.w`. From underneath the sheen is the sun read back down through the water, so it dims
        // with how much water is in the way and survives only at the shallowest grazing angle.
        let spec_weight = select(fres2, fres2 * fres2 * ramp.a, below);
        rgb += w.shade_params.z * spec_color * spec_weight * water_sun;
    } else {
        // fork: modern lighting — the experimental tier takes the reference's glare and specular
        // in the same share it takes the reference's combine (`look_c.z`).
        let k = w.look_c.z;
        let glare = clamp(dot(to_light, n), 0.0, 1.0) * w.shade_params.w;
        rgb += k * glare * wow_light.light_diffuse.rgb * body * water_sun;
        rgb += k * w.shade_params.z * spec_color * fres2 * water_sun;
        if (!modern) {
            // A classic frame has no glitter path to carry the sun (that is HDR, added by the
            // linear wrapper), so the clear share keeps the reference's Blinn track there —
            // without the sheen floor, whose flat grazing lift the reflection carries.
            rgb += (1.0 - k) * w.shade_params.z * wow_light.light_spec.rgb * spec * fres2;
        }
    }
    // fork: modern lighting — the sun's glitter path: a tight HDR lobe on the reflected ray over a
    // broad sheen, only where the sun reaches the surface. Added in linear by the wrapper.
    if (modern_above && to_light.y > -0.05) {
        let rv = reflect(v, n);
        let sd = clamp(dot(rv, to_light), 0.0, 1.0);
        // Warmed by the zone's scatter colour (the froxel key in-scatter, normalised), so a dawn
        // glitter path takes the haze's salmon instead of the key row's white.
        let scatter = modern_block.fog.sun.rgb;
        let warm = scatter / max(max(scatter.r, max(scatter.g, scatter.b)), 1e-4);
        let key = pow(wow_light.light_diffuse.rgb, vec3<f32>(2.2)) * mix(vec3<f32>(1.0), warm, 0.6);
        let horizon = smoothstep(-0.05, 0.08, to_light.y);
        // The experimental tier's calm water keeps the broad path and only `look_b.w` of the spike —
        // the pin-point sparkle field is the glossy half of the look it steps away from.
        let spike = select(9.0, 9.0 * w.look_b.w, experimental);
        modern_glint = key * ((pow(sd, 900.0) * spike + pow(sd, 160.0) * 0.8 + pow(sd, 24.0) * 0.12) * water_sun * horizon * (0.35 + 0.65 * fres));
    }
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

// fork: modern lighting (FORK.md) — the entry point wraps the classic body and
// decodes its output when the frame is linear; every return path above stays untouched.
@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    modern_glint = vec3<f32>(0.0);
    let c = fragment_classic(in);
    var rgb = colorspace::modern_out(c.rgb, wow_light.point_count.y) + modern_glint;
    // P4: transparent water draws after the fog composite, so it fogs itself; opaque magma and slime
    // (`kind.x`) draw before it and are fogged there.
    if (wow_light.point_count.y > 0.5 && wow_light.fog_color.w > 0.5 && w.kind.x < 0.5) {
        let fog = froxelfog::fog_at(modern_fog_volume, scene_samp, modern_block.fog, in.world_position.xyz);
        rgb = rgb * fog.a + fog.rgb;
    }
    return vec4<f32>(rgb, c.a);
}
