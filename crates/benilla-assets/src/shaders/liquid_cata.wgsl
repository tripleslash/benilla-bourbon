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
#import benilla_modern::skydome
// fork: F — the ray-traced shadows (`benilla_modern::rtshadow`, native only), where the lane runs.
#ifdef RT_SHADOWS
#import benilla_modern::rtshadow
#endif
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
    // water with no depth under it, z = the refraction's bend per yard of water, w = the sun/moon
    // highlight's GGX roughness `α` (`disk_highlight`).
    look_b: vec4<f32>,
    // x = the far water's mean reflectance ceiling (`lively_fresnel`), y = the gain on the facets'
    // spread the Fresnel averages over, z = the share
    // of the reference's combine blended over the clear surface, w = the wind ripples' RMS slope
    // (`wind_ripples`).
    look_c: vec4<f32>,
    // The highlight: x = the light disk's angular radius (radians), y = the share of the kind's own
    // swell on the highlight's own normal, z = the wind ripples' RMS slope on it (the sea's alone),
    // w = its gain over the microfacet value.
    look_d: vec4<f32>,
    // x = the turbidity (grey extinction per yard on top of the absorption), y = the wind speed
    // (m/s) whose sub-pixel ripples floor the roughness (Cox–Munk), z = the caustics' gain, w = the
    // roughness `α` at which the surface counts as wholly unresolved.
    look_e: vec4<f32>,
    // x/y = the share of the reflection's contrast lost near and far (the grade), z = the
    // displacement's gain over the sea's own height, w = reserved.
    look_f: vec4<f32>,
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
// fork: fog once — the share of the experimental tier's output, linear, that already carries its own
// fog (the bed seen through the water, the reflected world): `fragment_classic` leaves it out of its
// gamma output and the wrapper adds it after the surface's fog. Zero on every other lane.
var<private> modern_prefogged: vec3<f32>;

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
///
/// fork: the dome's dawn/dusk warp too (`benilla_modern::skydome`, its strength and the sun's azimuth
/// on the rows' `w` — `liquid::ripple`'s `pack_sky_rows`), so a sunset sky's glow toward the sun is in
/// the water under it.
fn sky_dome(dir: vec3<f32>) -> vec3<f32> {
    return sky_dome_at(degrees(asin(clamp(dir.y, -1.0, 1.0))), dir);
}

/// [`sky_dome`] at elevation `elev` (degrees) on the azimuth of `dir`.
fn sky_dome_at(elev: f32, dir: vec3<f32>) -> vec3<f32> {
    let s = ripple.sky;
    var s1 = s[1].rgb;
    var s2 = s[2].rgb;
    var s3 = s[3].rgb;
    var s4 = s[4].rgb;
    let warp_s = s[0].w;
    if (warp_s > 0.0) {
        let gl = skydome::warp_glows(dir, s[1].w);
        s1 = skydome::warp_ring(s[1].rgb, gl, warp_s, s[0].rgb, s[1].rgb);
        s2 = skydome::warp_ring(s[2].rgb, gl, warp_s, s[0].rgb, s[1].rgb);
        s3 = skydome::warp_ring(s[3].rgb, gl, warp_s, s[0].rgb, s[1].rgb);
        s4 = skydome::warp_ring(s[4].rgb, gl, warp_s, s[0].rgb, s[1].rgb);
    }
    var col: vec3<f32>;
    if (elev <= 0.0) {
        col = s[5].rgb;
    } else if (elev < 1.8) {
        col = mix(s[5].rgb, s4, elev / 1.8);
    } else if (elev < 3.7) {
        col = mix(s4, s3, (elev - 1.8) / (3.7 - 1.8));
    } else if (elev < 9.8) {
        col = mix(s3, s2, (elev - 3.7) / (9.8 - 3.7));
    } else if (elev < 16.8) {
        col = mix(s2, s1, (elev - 9.8) / (16.8 - 9.8));
    } else {
        col = mix(s1, s[0].rgb, (elev - 16.8) / (90.0 - 16.8));
    }
    return colorspace::modern_in(col, wow_light.point_count.y);
}

/// fork: the sky a rough surface reflects along `dir`: the dome blurred along elevation by the lobe
/// of the slope the surface no longer draws (`alpha_lost`, the roughness above the calm floor), three
/// taps at `±σ`, `σ = √2·α`. The spread is capped at half the ray's elevation, so the band at and
/// below the horizon (the fog colour) never bleeds up into the reflected sky.
fn sky_dome_rough(dir: vec3<f32>, alpha_lost: f32) -> vec3<f32> {
    let elev = asin(clamp(dir.y, -1.0, 1.0));
    let sigma = min(1.41421356 * alpha_lost, 0.5 * max(elev, 0.0));
    if (sigma < 0.002) {
        return sky_dome_at(degrees(elev), dir);
    }
    return 0.25 * sky_dome_at(degrees(elev - sigma), dir)
        + 0.5 * sky_dome_at(degrees(elev), dir)
        + 0.25 * sky_dome_at(degrees(elev + sigma), dir);
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

// fork: the experimental tier's mirror read. How far (yd) the reflected world is assumed to stand
// behind the water — the reference's own 20, there being no per-pixel hit distance — and the share
// of it the wave's lean moves the read by.
const MIRROR_HIT_YD: f32 = 20.0;
const MIRROR_WAVE_REACH: f32 = 0.4;
// fork: the mirror's mip chain (`liquid::screen`'s `REFLECT_MIPS`, which a test pins against this):
// its level count, the blur of its level 0 in its own texels (the four-tap tent it is filtered
// with), and the level whose texels are the local mean the grade pivots on.
const MIRROR_MIPS: f32 = 5.0;
const MIRROR_CHAIN_SIGMA0: f32 = 1.2;
const MIRROR_MEAN_LOD: f32 = 4.0;
// fork: a three-tap column at gap `g` spreads as far as one tap of `σ·√(1 + 8/3)` would: the column's
// reach over the narrow axis's blur.
const MIRROR_TAP_REACH: f32 = 1.915;
// fork: the view distances (yd) over which the reflection's grade goes from near to far.
const GRADE_NEAR_YD: f32 = 30.0;
const GRADE_FAR_YD: f32 = 300.0;
// fork: the vertical column (yd) over which the experimental tier's water comes in at the shore.
const SHORE_FADE_YD: f32 = 0.3;
// fork: the steepest bed (rise per yard) whose column change across a pixel the soft shore widens
// its fade for; anything steeper is a depth discontinuity behind the water.
const SHORE_MAX_BED_SLOPE: f32 = 2.0;

/// fork: a reflected ray that a wave has turned under the surface, folded back just above it: a facet
/// that would reflect into the water sends its light up off the next wave instead.
fn fold_above(r: vec3<f32>) -> vec3<f32> {
    return normalize(r + vec3<f32>(0.0, max(r.y, 0.01) - r.y, 0.0));
}

/// fork: **the mirror through a rough surface** (the benilla-watershader fork's lobe): the reflected
/// rays of the facets the pixel no longer draws spread `2·σ` radians along the view's bearing (`up`)
/// and `2·σ·μ` across it (`side`), the across axis foreshortened at grazing — the tall streaks a
/// rough lake draws under a far bank. Seen from the eye that spread is `h = D/(D + d_eye)` of itself,
/// so a reflection stays sharp near its contact. The mirror's mips carry the narrow axis's blur and a
/// three-tap column the rest of the tall one; under half a texel it is the plain read.
///
/// Then **graded**, luma-only, toward the mirror's own local mean (a high mip): rough and far water
/// keeps `1 − (look_f.x + look_f.y·ρ)` of its reflection's contrast. The pivot is the world the mirror
/// drew, never the sky, so water under a dark bank darkens toward the bank; luma-only, so a grey post
/// does not take the pivot's hue.
fn mirror_read_rough(uv: vec2<f32>, lost: vec3<f32>, v: vec3<f32>, d_eye: f32, rho: f32) -> vec3<f32> {
    let flat_len = length(v.xz);
    let b = v.xz / max(flat_len, 1e-4);
    let c = vec2<f32>(-b.y, b.x);
    let l_along = b.x * b.x * lost.x + b.y * b.y * lost.y + 2.0 * b.x * b.y * lost.z;
    let l_across = c.x * c.x * lost.x + c.y * c.y * lost.y + 2.0 * c.x * c.y * lost.z;
    let mu = clamp(-v.y, 0.0, 1.0);
    let h = MIRROR_HIT_YD / (MIRROR_HIT_YD + d_eye);
    let dims = vec2<f32>(textureDimensions(reflect_map, 0));
    // Mirror texels per radian at the centre of the view, vertically and across alike.
    let per_rad = dims.y * 0.5 * view.clip_from_view[1][1];
    let sv = 2.0 * sqrt(max(l_along, 0.0)) * h * per_rad;
    let ss = 2.0 * sqrt(max(l_across, 0.0)) * mu * h * per_rad;
    var col: vec3<f32>;
    if (max(sv, ss) < 0.5) {
        col = textureSampleLevel(reflect_map, scene_samp, uv, 0.0).rgb;
    } else {
        let lod = clamp(log2(max(max(ss, sv / MIRROR_TAP_REACH), 1e-3) / MIRROR_CHAIN_SIGMA0), 0.0, MIRROR_MIPS - 1.0);
        let sig = MIRROR_CHAIN_SIGMA0 * exp2(lod);
        // Three taps at `±g` spread `g·√(2/3)`; the gap makes up what the level does not carry, at
        // most two of its own blurs apart so the column never breaks into copies. Taps step ±¼σ
        // sideways so the filter's bilinear facets do not line up.
        let gap = min(sqrt(max(sv * sv - sig * sig, 0.0) * 1.5), 2.0 * sig);
        let du = vec2<f32>(0.25 * sig, gap) / dims;
        col = (textureSampleLevel(reflect_map, scene_samp, uv + vec2<f32>(du.x, -du.y), lod).rgb
            + textureSampleLevel(reflect_map, scene_samp, uv, lod).rgb
            + textureSampleLevel(reflect_map, scene_samp, uv + vec2<f32>(-du.x, du.y), lod).rgb) / 3.0;
    }
    let mean = textureSampleLevel(reflect_map, scene_samp, uv, MIRROR_MEAN_LOD).rgb;
    let k = clamp(1.0 - (w.look_f.x + w.look_f.y * rho), 0.0, 1.0);
    let y = colorspace::luminance(col);
    let y_mean = colorspace::luminance(mean);
    let y_graded = y_mean + k * (y - y_mean);
    return col * clamp(y_graded / max(y, 1e-4), 0.0, 4.0);
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

    // fork: **the surface rises and falls** — the Modern (Experimental) tier on a modern frame lifts
    // an ADT surface by the height of the same long wind trains whose slope the fragment draws
    // (`sunshadow::wind_sea_height`, the slope's own integral, over the same noise warp), so the
    // displaced surface and its normal are one sea. It keeps the kind's share of the wind sea (all of
    // it on the sea), goes flat toward the shore (the depth byte `V` is 0 at the waterline), full to
    // `HEIGHT_FULL_YD` and gone by `HEIGHT_GONE_YD` — inside the fine ring (`liquid::surface`), so
    // the coarse lattice past it is always flat — and never more than `HEIGHT_MAX_YD` either way.
    // WMO pools stay flat: a fountain does not carry a sea.
    if (w.optics.w > 0.5 && wow_light.point_count.y > 0.5 && w.path.x < 0.5 && w.look_f.z > 0.0) {
        let dist = length(out.view_vec);
        let fade = 1.0 - smoothstep(HEIGHT_FULL_YD, HEIGHT_GONE_YD, dist);
        if (fade > 0.0) {
            let xz = out.world_position.xz;
            let noise = wind_noise(xz, globals.time);
            let keep = select(w.look_b.x, 1.0, w.kind.y > 0.5);
            let shore = smoothstep(HEIGHT_SHORE_V.x, HEIGHT_SHORE_V.y, out.depth_v);
            let h = sunshadow::wind_sea_height(xz, globals.time, dist, noise.xy, noise.z, WAVE_HEIGHT_FROM)
                * keep * shore * fade * w.look_f.z;
            out.world_position.y = out.world_position.y + clamp(h, -HEIGHT_MAX_YD, HEIGHT_MAX_YD);
            out.clip_position = position_world_to_clip(out.world_position.xyz);
            out.view_vec = out.world_position.xyz - view.world_position.xyz;
        }
    }
    return out;
}

// fork: the displacement's reach: full to `FULL`, gone by `GONE` yd from the eye (inside the fine
// ring's 64), and at most `MAX` yd either way (`liquid::surface`'s `HEIGHT_MAX_YD`, which a test holds
// to this). `SHORE_V` is the depth `V` over which it comes in from the waterline.
const HEIGHT_FULL_YD: f32 = 34.0;
const HEIGHT_GONE_YD: f32 = 56.0;
const HEIGHT_MAX_YD: f32 = 0.75;
const HEIGHT_SHORE_V: vec2<f32> = vec2<f32>(0.05, 0.4);

/// fork: the wind sea's two slow noise taps at world `xz`: `xy` the phase warp, `z` the gust field —
/// one read for the vertex stage's height and the fragment's slope alike, so the two follow one sea.
/// An explicit level, so either stage may call it.
fn wind_noise(xz: vec2<f32>, t: f32) -> vec3<f32> {
    let warp_n = textureSampleLevel(noise_vol, art_samp, vec3<f32>(xz * 0.011 + vec2<f32>(t * 0.004, 0.0), fract(t * 0.003)), 0.0).rgb;
    let gust_n = textureSampleLevel(noise_vol, art_samp, vec3<f32>(xz * 0.0045 - sunshadow::wind_dir() * t * 0.0012, fract(t * 0.002 + 0.5)), 0.0).r;
    return vec3<f32>(warp_n.rg * 2.0 - vec2<f32>(1.0), smoothstep(0.25, 0.75, gust_n));
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
///
/// fork: also the slope variance per axis the sheet's mips have averaged out of the four taps
/// (`lost`), stored in each mip texel's alpha (`bump_lost`) and weighted as the taps are summed — the far water's ripples become its roughness instead of vanishing.
struct Bump {
    slope: vec2<f32>,
    lost: f32,
}

/// fork: the slope variance a tap's alpha can carry (`liquid::art`'s `BUMP_VAR_MAX`, which its test
/// holds to this): each mip texel stores the variance per axis its footprint averaged away as
/// `sqrt(σ² / BUMP_VAR_MAX)`, 0 at level 0, so a filtered tap's `a² · BUMP_VAR_MAX` is that variance
/// back, trilinear like the normal beside it.
const BUMP_VAR_MAX: f32 = 0.05;

fn bump_lost(tap: vec4<f32>) -> f32 {
    return tap.a * tap.a * BUMP_VAR_MAX;
}

fn bump_gradient(in: VsOut) -> Bump {
    let ta = textureSample(bump_tex, art_samp, in.bump_uv_a.xy);
    let tb = textureSample(bump_tex, art_samp, in.bump_uv_a.zw);
    let tc = textureSample(bump_tex, art_samp, in.bump_uv_b.xy);
    let td = textureSample(bump_tex, art_samp, in.bump_uv_b.zw);
    let a = ta.xyz;
    let b = tb.xyz;
    let c = tc.xyz;
    let d = td.xyz;
    let fine = (a + b) - 1.0;
    let coarse = ((c + d) - 1.0) * 1.4;
    let dist = length(in.view_vec);
    let t = clamp(dist * 0.001 + 0.2, 0.0, 1.0);
    let g = (fine * 2.0 - coarse) * t + coarse;
    // The raw SLOPE, not a normal. The reference normalises here (`bump.z = 1`, then `nrm`), but the
    // ripple field has its own slope to add and the two have to be summed before either is turned
    // into a direction — normalising twice would let whichever is larger swallow the other.
    //
    // fork: `g = t·(n_a + n_b) + 0.7·(1 − t)·(n_c + n_d)` in decoded normals, so each tap's lost
    // variance enters at its weight squared.
    let s = w.bump_params.w;
    let lost = t * t * (bump_lost(ta) + bump_lost(tb)) + 0.49 * (1.0 - t) * (1.0 - t) * (bump_lost(tc) + bump_lost(td));
    return Bump(g.xy * s, lost * s * s);
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

/// fork: the world point the screen shows at `uv` at planar view depth `z` (yd), through the view's
/// own projection, its jitter included: `ndc = (P₀₀·x − P₂₀·z) / z`, solved for `x` (and `y`).
fn view_point_at(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p = view.clip_from_view;
    let vp = vec3<f32>((ndc.x + p[2][0]) * z / p[0][0], (ndc.y + p[2][1]) * z / p[1][1], -z);
    return (view.world_from_view * vec4<f32>(vp, 1.0)).xyz;
}

/// fork: how far a refracted read at view depth `read_z` can be trusted to show what this pixel's
/// own ray sees (`own_z`), through water whose surface is at `surface_z`: 1 while the water in front
/// of the read is within [`REFRACT_SAME_SURFACE`] doublings of the water in front of this pixel's
/// background, 0 by twice that. A bed reads as itself; a body just under the surface, read from
/// water over a deep bed (or the bed read from water over the body), is a different surface, and so
/// is anything nearer than the surface, which has no water in front of it at all.
fn refract_trust(read_z: f32, own_z: f32, surface_z: f32) -> f32 {
    let ratio = max(read_z - surface_z, 1e-4) / max(own_z - surface_z, 1e-4);
    return 1.0 - smoothstep(REFRACT_SAME_SURFACE, 2.0 * REFRACT_SAME_SURFACE, abs(log2(ratio)));
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
// fork: the shortest wind train (yd) whose height the surface carries — the crest shading's and the
// displacement's: the long swell rises and falls, the chop only tilts.
const WAVE_HEIGHT_FROM: f32 = 4.0;
// fork: the crest shading: how far the body's light swings, and the swell height (yd) that takes it
// all the way.
const CREST_SHADE: f32 = 0.15;
const CREST_HEIGHT_YD: f32 = 0.3;
const MODERN_REFRACT_MAX_PATH: f32 = 6.0;
// fork: how many doublings apart the water in front of a refracted read and in front of this
// pixel's own background may be before the read stops being trusted (whole within one, none by two).
const REFRACT_SAME_SURFACE: f32 = 1.0;
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
///
/// fork: the slope a train's fade takes out is not thrown away: `lost` carries it, `(xx, zz, xz)` in
/// the same unit RMS, for the caller to turn into roughness (`surface_roughness`) — the fork's
/// energy-conserving filter, so a calmed-out octave widens the gloss it would have broken up.
struct Ripples {
    slope: vec2<f32>,
    lost: vec3<f32>,
}

fn wind_ripples(xz: vec2<f32>, foot_x: vec2<f32>, foot_y: vec2<f32>) -> Ripples {
    let t = anim_time();
    let coarse = textureSampleLevel(noise_vol, art_samp, vec3<f32>(xz * 0.015 + vec2<f32>(t * 0.004, 0.0), fract(t * 0.003)), 0.0).rgb - vec3<f32>(0.5);
    let fine = textureSampleLevel(noise_vol, art_samp, vec3<f32>(xz * 0.05 - vec2<f32>(0.0, t * 0.01), fract(t * 0.006 + 0.5)), 0.0).rgb - vec3<f32>(0.5);
    let wd = sunshadow::wind_dir();
    var slope = vec2<f32>(0.0);
    var lost = vec3<f32>(0.0);
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
        let amp = weight * group;
        lost = lost + vec3<f32>(dir.x * dir.x, dir.y * dir.y, dir.x * dir.y)
            * (amp * amp * sunshadow::WIND_CREST_VAR * (1.0 - fade * fade));
        energy = energy + weight * weight;
    }
    // The unfaded layer's RMS slope (a half per train from the cosine), so a faded train only calms the
    // water rather than being renormalised back up.
    let norm = energy * 0.5;
    return Ripples(slope / sqrt(norm), lost / norm);
}

/// fork: the GGX `α` of a roughness tensor `(xx, zz, xz)` (`α²` per axis): the isotropic lobe of the
/// same footprint, `det^¼`.
fn tensor_alpha(a: vec3<f32>) -> f32 {
    return sqrt(sqrt(max(a.x * a.y - a.z * a.z, 1e-12)));
}

/// fork: the roughness tensor `α²` of a surface whose resolved facets are `α0` rough, over the sub-
/// pixel wind ripples of a `wind` m/s wind (Cox–Munk: a mean square slope of `0.00512·U`, of which the
/// share under the finest drawn octave is about an eighth) and the slope covariance `lost` its
/// octaves no longer draw — `α²` being twice the slope variance per axis (Beckmann).
fn surface_roughness(alpha0: f32, wind: f32, lost: vec3<f32>) -> vec3<f32> {
    let floor2 = alpha0 * alpha0 + COX_MUNK_SHARE * 0.00512 * max(wind, 0.0);
    return vec3<f32>(floor2, floor2, 0.0) + 2.0 * lost;
}

/// fork: the share of the Cox–Munk mean square slope that sits below the finest octave drawn.
const COX_MUNK_SHARE: f32 = 0.12;

/// fork: the glint's soft knee, on its brightest channel so its hue holds: linear to `GLINT_KNEE`,
/// then rolling off toward `GLINT_KNEE + GLINT_TOP` — HDR enough to bloom, never a single-pixel
/// firefly a hundred times the sky.
const GLINT_KNEE: f32 = 4.0;
const GLINT_TOP: f32 = 12.0;

fn glint_knee(c: vec3<f32>) -> vec3<f32> {
    let m = max(c.r, max(c.g, c.b));
    if (m <= GLINT_KNEE) {
        return c;
    }
    let over = m - GLINT_KNEE;
    return c * ((GLINT_KNEE + GLINT_TOP * over / (GLINT_TOP + over)) / m);
}

// fork: modern lighting — the experimental tier's highlight: water's own head-on reflectance (the
// sun's Fresnel is the physical one, so a sunset path climbs to full at grazing where the stylised
// reflection caps), and the geometric specular AA's screen-space variance and clamp (Tokuyoshi &
// Kaplanyan 2019's 0.25 and 0.18).
const WATER_F0: f32 = 0.02;
const SPEC_AA_VARIANCE: f32 = 0.25;
const SPEC_AA_CLAMP: f32 = 0.18;
const HIGHLIGHT_PI: f32 = 3.14159265;

/// fork: modern lighting — **geometric specular antialiasing**: `α` widened by how far the normal
/// turns across the pixel, so a lobe narrower than the pixel's spread of facets is drawn as the
/// average of them — a steady sheet of glitter in the distance instead of a sparkle that crawls. The
/// same idea as `wind_ripples`' footprint fade, on the lobe instead of the slope. Called in uniform
/// control flow (the derivatives).
fn specular_aa(n: vec3<f32>, alpha: f32) -> f32 {
    let dx = dpdx(n);
    let dy = dpdy(n);
    let variance = SPEC_AA_VARIANCE * (dot(dx, dx) + dot(dy, dy));
    let kernel = min(2.0 * variance, SPEC_AA_CLAMP);
    return sqrt(clamp(alpha * alpha + kernel, 0.0, 1.0));
}

/// fork: modern lighting — **the sun or moon off the water**, per unit of the key's light: GGX with
/// the height-correlated Smith visibility and Schlick's Fresnel, lit by a DISK of angular radius
/// `radius` rather than a point. The disk is Karis's representative point (2013): the light direction
/// moved to the point of the disk nearest the reflected ray, and the lobe's peak rescaled by
/// `(α / α')²`, `α' = α + tan(radius) / 2`, so the disk's energy stays what a point's would be.
///
/// With `α` under the radius every facet whose reflected ray falls inside the disk sees all of it and
/// the rest see nothing: a highlight broken into flat, sharp-edged islands as wide as the swell spreads
/// the facets, each one as bright as the disk itself. `π` because the frame's lit colour is
/// `albedo · light` with no `1/π`, so the key here is illuminance over `π`.
fn disk_highlight(n: vec3<f32>, to_eye: vec3<f32>, to_light: vec3<f32>, alpha: f32, radius: f32) -> f32 {
    let nv = dot(n, to_eye);
    if (nv <= 1e-4) {
        return 0.0;
    }
    let r = reflect(-to_eye, n);
    let tan_r = tan(radius);
    let to_ray = dot(to_light, r) * r - to_light;
    let lp = normalize(to_light + to_ray * clamp(tan_r / max(length(to_ray), 1e-5), 0.0, 1.0));
    let nl = clamp(dot(n, lp), 0.0, 1.0);
    let h = normalize(lp + to_eye);
    let nh = clamp(dot(n, h), 0.0, 1.0);
    let vh = clamp(dot(to_eye, h), 0.0, 1.0);
    let a = max(alpha, 0.002);
    let a2 = a * a;
    let a_disk = clamp(a + 0.5 * tan_r, 0.0, 1.0);
    let energy = a2 / (a_disk * a_disk);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (HIGHLIGHT_PI * dd * dd);
    let vis = 0.5 / max(nl * (nv * (1.0 - a) + a) + nv * (nl * (1.0 - a) + a), 1e-5);
    let f = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - vh, 5.0);
    return HIGHLIGHT_PI * d * vis * f * nl * energy;
}

/// fork: Karis's 2014 fit of the split-sum environment BRDF for a GGX lobe of roughness `r` (`√α`):
/// the share of a lobe's light a reflectance `f0` returns at `n·v`.
fn env_brdf_approx(f0: f32, n_dot_v: f32, r: f32) -> f32 {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let q = r * c0 + c1;
    let a004 = min(q.x * q.x, exp2(-9.28 * n_dot_v)) * q.x + q.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + q.zw;
    return f0 * ab.x + ab.y;
}

/// fork: **water's Fresnel over a rough lobe**: Schlick at `μ = n·v`, times the share of a lobe of
/// roughness `α` that still reflects against a mirror's (the env-BRDF fit's rough-over-smooth ratio).
/// Exactly Schlick at `α = 0`; a rough surface reflects less at grazing, where its facets turn away.
/// Below `μ_kink` the fit's `min` changes branch and the ratio dips into a dark band under the
/// horizon, so the ratio is read no lower than it: reflectance never rises as the view leaves grazing.
fn water_fresnel(mu: f32, alpha: f32) -> f32 {
    let m = clamp(mu, 0.0, 1.0);
    let schlick = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - m, 5.0);
    let r = sqrt(clamp(alpha, 0.0, 1.0));
    let kink = -log2(max((1.0 - r) * (1.0 - r), 1e-6)) / 9.28;
    let me = max(m, kink);
    let ratio = env_brdf_approx(WATER_F0, me, r) / max(env_brdf_approx(WATER_F0, me, 0.0), 1e-4);
    return max(schlick * min(ratio, 1.0), WATER_F0);
}

/// fork: the stylised reflectance: water's own curve ([`water_fresnel`]) raised to a head-on floor
/// `f0`, so a river looked straight down on keeps some sky.
fn styled_fresnel(mu: f32, alpha: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * water_fresnel(mu, alpha);
}

/// fork: **the lively Fresnel** (the benilla-watershader fork's): the reflectance averaged over the
/// facets the pixel no longer draws — two-point Gauss–Hermite at `μ ± σ` — on the pixel's own wave
/// normal (`mu_px`), then RESCALED by how far the flat water's mean (`mu_flat`) overshoots the
/// `ceiling`. A hard cap flattened all far water to one coat; the rescale keeps each wave's own
/// contrast — a face turned to the eye shows the body, its back the sky — while the mean still sits at
/// the ceiling. `σ` is what makes far water average smoothly instead of flickering facet by facet.
fn lively_fresnel(mu_px: f32, mu_flat: f32, sigma: f32, alpha: f32, f0: f32, ceiling: f32) -> f32 {
    let f_px = 0.5 * (styled_fresnel(mu_px - sigma, alpha, f0) + styled_fresnel(mu_px + sigma, alpha, f0));
    let f_mean = 0.5 * (styled_fresnel(mu_flat - sigma, alpha, f0) + styled_fresnel(mu_flat + sigma, alpha, f0));
    return clamp(f_px * min(1.0, ceiling / max(f_mean, 1e-4)), f0, 0.95);
}

/// fork: the cosine of a ray of cosine `mu` (against the surface's up) once refracted into water
/// (`n = 1.333`), floored at the critical angle's 0.6612: however grazing the ray in air, it crosses at
/// most ~1.5 yards of water per yard of depth.
fn refracted_cos(mu: f32) -> f32 {
    let m = clamp(mu, 0.0, 1.0);
    return max(sqrt(max(1.0 - (1.0 - m * m) / (WATER_IOR * WATER_IOR), 0.0)), WATER_CRITICAL_COS);
}

const WATER_IOR: f32 = 1.333;
const WATER_CRITICAL_COS: f32 = 0.6612;
/// fork: the share of the bed the authored depth's swatch blend leaves showing that the clear water
/// keeps at the least ([`modern_water`]'s transmittance floor), so a bed the 1.12 water showed at that
/// depth never sinks out of sight under the clear combine.
const BED_FLOOR_SHARE: f32 = 0.6;
/// fork: how much the body's light swings with the sun's bearing across a wave's face.
const BODY_RIPPLE_SHADE: f32 = 0.2;

// ─────────────────────────────────────────────────────────────────────────────────── the caustics
//
// fork: **the caustic net** (the benilla-watershader fork's lens estimate). A ray of the sun entering
// the surface at `x` lands on a bed `D` deep along its refracted path at `x + D·κ·∇h(x)`, κ = 1 − 1/n,
// so the bed's irradiance is the sun's over |det(I + D·κ·H)|, `H` the surface's Hessian. The lenses
// are two sets of eight analytic ripples, a fine one (1.6-0.65 yd) sharp in the first yard or two and a
// coarse one (3.8-1.8 yd) focusing `CAUSTIC_COARSE` times deeper, laid on the wind's headings at the
// swell's deep-water pace, alternate ripples running opposite ways so the net shimmers in place
// rather than reading as a current. A few plane waves alone tile the bed with
// a crystal lattice, so each ripple's wavefronts wander along its heading and its strength sways, both
// by its own mix of the slow noise volume (`caustic_wobble`) over cells under two wavelengths wide;
// the wander's slope stays near a quarter, so the Hessian keeps the unwarped heading. The intensity is regularised
// (`CAUSTIC_EPS`) and divided by its mean at that depth, so only the sun's direct share is
// redistributed and the bed keeps its energy; neither set is drawn past its sharpest, where it would
// break into speckle, and each ripple fades as the bed's pixel footprint nears its wavelength.

/// The path along the refracted sun (yd) at which a ripple's curvature of one focuses.
const CAUSTIC_FOCUS_YD: f32 = 0.72;
/// How much deeper the coarse set focuses than the fine one.
const CAUSTIC_COARSE: f32 = 3.3;
/// The regularisation of `1/|det|`: the brightest line is `1/ε` over the mean.
const CAUSTIC_EPS: f32 = 0.15;
/// The deepest focus ratio either set is drawn at.
const CAUSTIC_RHO_MAX: f32 = 1.6;
/// Each ripple's curvature per unit focus.
const CAUSTIC_CURV: f32 = 0.57735;
/// The path (yd) over which the net's contrast fades out.
const CAUSTIC_FADE_YD: vec2<f32> = vec2<f32>(6.0, 20.0);
/// How fast turbidity scatters the net away, per unit of turbidity per yard.
const CAUSTIC_TURB: f32 = 0.6;
/// A ripple's wavelength in bed pixels between which it fades out of the net: its lines are a tenth
/// of it wide, so under 8 px it would alias, and at 16 it draws whole.
const CAUSTIC_RIPPLE_PX: vec2<f32> = vec2<f32>(8.0, 16.0);
/// The noise volume's frequency (repeats per yd) under the fine set's wander, and the coarse set's:
/// its coarsest cells are a quarter of a repeat, 1.8 and 4.2 yd.
const CAUSTIC_WOBBLE_FREQ: vec2<f32> = vec2<f32>(0.14, 0.06);
/// How far (yd) a ripple's wavefronts wander along its heading per unit of its noise mix, fine and
/// coarse: several radians of phase on the shortest ripple of each set.
const CAUSTIC_WANDER_YD: vec2<f32> = vec2<f32>(0.5, 1.0);
/// How far a ripple's strength sways per unit of its noise mix, about `0.85`, capped at 1 so no
/// patch is drawn past the set's sharpest.
const CAUSTIC_SWAY: f32 = 0.4;
/// Ripples per set; each one's curvature is scaled by `sqrt(6 / N)` so the set's rms curvature, and
/// with it the fit `caustic_mean` reads, stays the six-ripple set's.
const CAUSTIC_RIPPLES: u32 = 8u;

/// The signed noise (about ±1 at its extremes) that bends one set's ripples at world `p`: the
/// volume's three independent channels, fixed on the bed and morphing slowly through its slices (a
/// drift would carry the whole net one way).
fn caustic_wobble(p: vec2<f32>, t: f32, coarse: bool) -> vec3<f32> {
    let f = select(CAUSTIC_WOBBLE_FREQ.x, CAUSTIC_WOBBLE_FREQ.y, coarse);
    let uv = p * f + select(vec2<f32>(0.0), vec2<f32>(0.37, 0.71), coarse);
    let slice = fract(t * 0.004 + select(0.0, 0.5, coarse));
    let n = textureSampleLevel(noise_vol, art_samp, vec3<f32>(uv, slice), 0.0).rgb;
    return (n - vec3<f32>(0.5)) * 4.0;
}

/// One set's Hessian at world `p` (`h_xx, h_zz, h_xz` per unit focus) and, in `w`, the mean square
/// of what the footprint `foot` (yd per pixel on the bed) and the sway `s` keep of its ripples.
fn caustic_hessian(p: vec2<f32>, t: f32, foot: f32, coarse: bool, s: vec3<f32>) -> vec4<f32> {
    let wd = sunshadow::wind_dir();
    let wander_yd = select(CAUSTIC_WANDER_YD.x, CAUSTIC_WANDER_YD.y, coarse);
    var h = vec4<f32>(0.0);
    let n = f32(CAUSTIC_RIPPLES);
    let curv = CAUSTIC_CURV * sqrt(6.0 / n);
    for (var i = 0u; i < CAUSTIC_RIPPLES; i = i + 1u) {
        let fi = f32(i);
        // Geometric steps with an irrational jitter, so no two ripples beat in a simple ratio.
        let jitter = 1.0 + 0.09 * sin(fi * 5.1 + select(0.7, 2.3, coarse));
        let wl = select(1.6 * exp2(-fi * 0.186), 3.8 * exp2(-fi * 0.154), coarse) * jitter;
        // Headings scattered by the golden ratio over the half circle downwind: a ripple's Hessian is
        // the same either way along it, so over the full circle the eight would pair up.
        let turn = 3.14159265 * (fract(fi * 0.618034 + select(0.13, 0.41, coarse)) - 0.5);
        let d = vec2<f32>(wd.x * cos(turn) - wd.y * sin(turn), wd.x * sin(turn) + wd.y * cos(turn));
        let k = 6.2831853 / wl;
        let omega = sqrt(10.7 * k) * RIPPLE_PACE;
        // Each ripple's own mix of the three noise channels: one for its wander, one for its sway.
        let b = vec3<f32>(sin(fi * 2.1 + 0.4), sin(fi * 3.7 + 1.9), sin(fi * 5.3 + 3.1));
        let wander = dot(s, b) * wander_yd;
        let sway = clamp(0.85 + CAUSTIC_SWAY * dot(s, b.zxy), 0.35, 1.0);
        let keep = smoothstep(CAUSTIC_RIPPLE_PX.x, CAUSTIC_RIPPLE_PX.y, wl / max(foot, 1e-5)) * sway;
        // Odd ripples run upwind: the golden-ratio headings split the half circle evenly between the
        // two parities, so the net's travel cancels.
        let run = select(1.0, -1.0, (i & 1u) == 1u);
        let phase = (dot(d, p) + wander) * k - run * omega * t + fi * 2.39996 + select(0.0, 1.1, coarse);
        let c = -sin(phase) * curv * keep;
        h = h + vec4<f32>(c * d.x * d.x, c * d.y * d.y, c * d.x * d.y, keep * keep);
    }
    return vec4<f32>(h.xyz, h.w / n);
}

/// The regularised intensity's mean over the pattern at focus ratio `rho`: a rational fit, within 11 %
/// from 0 to 6.
fn caustic_mean(rho: f32) -> f32 {
    let r = min(rho, 6.0);
    let r2 = r * r;
    let r3 = r2 * r;
    return (0.989 + 2.1614 * r2 - 0.1102 * r3) / (1.0 - 0.1993 * r2 + 0.6322 * r3);
}

/// The net of Hessian `h` at focus ratio `rho`, mean 1. A set whose ripples the footprint has faded
/// is the whole set at a shallower focus (`rho` times the rms it keeps), so its mean is read there.
fn caustic_lens(h: vec4<f32>, rho: f32) -> f32 {
    let jxx = 1.0 + rho * h.x;
    let jzz = 1.0 + rho * h.y;
    let jxz = rho * h.z;
    let det = jxx * jzz - jxz * jxz;
    return inverseSqrt(det * det + CAUSTIC_EPS * CAUSTIC_EPS) / caustic_mean(rho * sqrt(h.w));
}

/// Both sets' net, mean 1, on the bed point whose refracted sun ray entered the surface at `entry`
/// (world XZ) and ran `path` yards to it, seen through `foot` yards a pixel on the bed: the fine net
/// sharp in the first yard or two, handing over to the coarse one as the bed falls away.
fn caustic_net(entry: vec2<f32>, path: f32, t: f32, foot: f32) -> f32 {
    let wf = 1.0 - smoothstep(1.0, 3.5, path);
    let wc = smoothstep(0.4, 2.0, path);
    var c = 1.0;
    if (wf > 0.0) {
        let rho = min(path / CAUSTIC_FOCUS_YD, CAUSTIC_RHO_MAX);
        let s = caustic_wobble(entry, t, false);
        c = c + wf * (caustic_lens(caustic_hessian(entry, t, foot, false, s), rho) - 1.0);
    }
    if (wc > 0.0) {
        let rho = min(path / (CAUSTIC_FOCUS_YD * CAUSTIC_COARSE), CAUSTIC_RHO_MAX);
        let s = caustic_wobble(entry, t, true);
        c = c + wc * (caustic_lens(caustic_hessian(entry, t, foot, true, s), rho) - 1.0);
    }
    return c;
}

/// fork: the clear water's output, linear: all of it (`rgb`), and the share of it that already
/// carries its own fog (`fogged`) — the bed seen through the water and the world in the reflection,
/// each fogged where it was drawn. A modern frame's surface fog then falls on the rest alone ("fog
/// once"): the body's own light and the sky.
struct ClearWater {
    rgb: vec3<f32>,
    fogged: vec3<f32>,
}

/// What [`modern_water`] reads beyond the surface's own frame: the reflection, how much of it is
/// already fogged, the calm surface's roughness tensor and lost slope, the bed's caustic gain, the
/// crest shading, the sun's direction and reach, and the swatch's opacity at the authored depth.
struct ClearInputs {
    refl: vec3<f32>,
    refl_fogged: f32,
    rough: vec3<f32>,
    lost: vec3<f32>,
    caustic: f32,
    crest: f32,
    to_light: vec3<f32>,
    sun: f32,
    auth_alpha: f32,
}

/// fork: modern lighting — **stylised clear water**, above the water, in linear radiance. Replaces
/// the classic COMBINE on a modern frame at the Modern (Experimental) water tier (`waterDetail 3`,
/// `w.optics.w`; plain Modern keeps the combine) — not its reflection, which arrives here as `refl`
/// exactly as the classic lane reads it (the planar mirror on its bent ray, the trace over it where
/// the mirror is invalid).
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
/// * the bed carries the **caustics** the surface focuses onto it (`x.caustic`, mean 1);
/// * then **Beer–Lambert** over the optical path a refracted ray really runs — down to the bed along
///   the view refracted into the water, and the sun's light down to it along its own refracted ray,
///   both floored at the critical angle — through the absorption and the grey turbidity, toward the
///   DEEP row under the surface's light (the colour the classic swatch reaches at depth). The deep
///   row stands for the water's single-scatter albedo, so what the column takes from the bed it gives
///   back in the zone's colour; a channel darkens into the zone's water and the clock keeps moving it.
///   The body's light leans with the sun's bearing across each wave and lifts on the crests, where
///   the water is thinnest. The bed keeps at least `BED_FLOOR_SHARE` of what the 1.12 swatch blend
///   left of it at the authored depth;
/// * the reflection goes over that at the **lively Fresnel** ([`lively_fresnel`]): water's curve over
///   the surface's roughness raised to a head-on floor (`look_a.z`), its far-water mean rescaled to a
///   ceiling (`look_c.x`), the facets' spread `look_c.y` times what the roughness says — and is itself
///   pulled `look_a.w` of the way to the body, the reference combine's `c11.y` idea, which is what
///   keeps water painterly rather than glossy.
fn modern_water(
    in: VsOut,
    n: vec3<f32>,
    v: vec3<f32>,
    bed_raw: vec3<f32>,
    bed_z: f32,
    frag_view_z: f32,
    deep: vec3<f32>,
    lit: vec3<f32>,
    x: ClearInputs,
) -> ClearWater {
    // `scene_depth` is planar view depth, so the bed's distance along this pixel's ray scales off
    // the surface's by the same ratio; `z` is how deep the bed lies under the surface there.
    let path = max(bed_z - frag_view_z, 0.0) * length(in.view_vec) / max(frag_view_z, 0.001);
    let mu_v = clamp(-v.y, 0.0, 1.0);
    let z = path * mu_v;
    let optical = min(z / refracted_cos(mu_v) + z / refracted_cos(x.to_light.y), 1000.0);
    var trans = exp(-(w.optics.xyz + vec3<f32>(max(w.look_e.x, 0.0))) * optical);
    trans = max(trans, vec3<f32>((1.0 - pow(clamp(x.auth_alpha, 0.0, 1.0), 2.2)) * BED_FLOOR_SHARE));
    // The scene copy is linear on a modern frame and the frame's own gamma bytes on a classic one;
    // this combine runs in linear either way. The bed and its depth are the refraction's own read
    // (`fragment_classic`), shared with the reference half of the blend rather than taken twice.
    let bed = select(colorspace::srgb_to_linear(bed_raw), bed_raw, wow_light.point_count.y > 0.5);
    let bearing = x.to_light.xz / max(length(x.to_light.xz), 1e-4);
    let lean = 1.0 + BODY_RIPPLE_SHADE * dot(n.xz, bearing) / max(n.y, 0.2) * x.sun;
    let scatter = colorspace::srgb_to_linear(deep * lit) * max(lean * x.crest, 0.0);
    let deep_lin = colorspace::srgb_to_linear(deep);
    let hue = deep_lin / max(max(deep_lin.r, max(deep_lin.g, deep_lin.b)), 1e-4);
    let tint = mix(vec3<f32>(1.0), hue, w.look_a.x * smoothstep(0.0, max(w.look_a.y, 0.01), z));
    let seen_bed = bed * tint * x.caustic * trans;
    let body = scatter * (vec3<f32>(1.0) - trans);

    let f0 = clamp(w.look_a.z, 0.0, 1.0);
    let ceiling = clamp(w.look_c.x, f0, 1.0);
    let flat_len = length(v.xz);
    let b = v.xz / max(flat_len, 1e-4);
    let along = b.x * b.x * x.lost.x + b.y * b.y * x.lost.y + 2.0 * b.x * b.y * x.lost.z;
    let sigma = flat_len * sqrt(max(along, 0.0)) * max(w.look_c.y, 0.0);
    let fres = lively_fresnel(dot(-v, n), mu_v, sigma, tensor_alpha(x.rough), f0, ceiling);

    let a = clamp(w.look_a.w, 0.0, 1.0);
    var out: ClearWater;
    out.rgb = (1.0 - fres) * (seen_bed + body) + fres * ((1.0 - a) * x.refl + a * scatter);
    out.fogged = (1.0 - fres) * seen_bed + fres * (1.0 - a) * x.refl * clamp(x.refl_fogged, 0.0, 1.0);
    return out;
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
    // fork: the pixel's world footprint and how fast the water column changes across it, taken here
    // at the top, in uniform control flow, for everything below that filters by them — the ripples'
    // fade, the caustics' and the soft shore's width. A browser's compiler refuses a derivative
    // under a branch on anything per-pixel, and naga does not check it.
    let foot_x = dpdx(in.world_position.xz);
    let foot_y = dpdy(in.world_position.xz);
    let foot = max(length(foot_x), length(foot_y));
    // A bed can change the column across a pixel by its footprint times its slope at most; a bigger
    // jump is a silhouette in the depth copy (a ridge in front of a deeper bed), not a shore, and
    // left unbounded it would fade the water to clear along every such edge.
    let column_rate = min(fwidth(column_v), foot * SHORE_MAX_BED_SLOPE);
    // fork: modern lighting — the experimental tier's wind ripples, read here in uniform control flow
    // for the pixel footprint's derivatives. Only the sea carries them, on its calm surface
    // (`look_c.w`) and on its highlight's normal (`look_d.z`); a river, a lake and a WMO's liquid keep
    // the cata bump's own octaves on both.
    var ripples = vec2<f32>(0.0);
    var spec_ripples = vec2<f32>(0.0);
    var ripples_lost = vec3<f32>(0.0);
    if (experimental && w.kind.y > 0.5 && (w.look_c.w > 0.0 || w.look_d.z > 0.0)) {
        let unit = wind_ripples(in.world_position.xz, foot_x, foot_y);
        ripples = w.look_c.w * unit.slope;
        spec_ripples = w.look_d.z * unit.slope;
        ripples_lost = unit.lost;
    }

    // The swell, plus whatever a body wading nearby has pushed into the water. The two are summed
    // as SLOPES and normalised once — see `bump_gradient`.
    let bump_tap = bump_gradient(in);
    let bump = bump_tap.slope;
    var swell = bump;
    // fork: modern lighting — a modern frame adds a wind-driven swell under the authored bump.
    var wind = vec2<f32>(0.0);
    var wind_lost = vec3<f32>(0.0);
    var sea_height = 0.0;
    if (modern) {
        // Two slow drifting taps of the noise volume: one warps the wave phases, one is the gust
        // field, so no two stretches of water repeat the same pattern.
        let wn = in.world_position.xz;
        let wt = globals.time;
        let noise = wind_noise(wn, wt);
        let sea = sunshadow::wind_sea(wn, wt, view_len, noise.xy, noise.z, WAVE_HEIGHT_FROM);
        wind = sea.slope;
        wind_lost = sea.lost;
        sea_height = sea.height;
        if (!experimental) {
            swell = swell + wind;
        }
    }
    // fork: the experimental tier's roughness — the slope variance its normal no longer draws, as an
    // `α²` tensor `(xx, zz, xz)` in the shading normal's own slope units: the bump sheet's mips, the
    // wind sea's faded trains and the ripples' faded octaves, each at the share of it the surface
    // keeps, under the Cox–Munk floor. `rough_s` is the calm surface's (the Fresnel and the
    // reflection read it), `rough_h` the highlight's own normal's.
    var rough_s = vec3<f32>(0.0);
    var rough_h = vec3<f32>(0.0);
    var lost_s = vec3<f32>(0.0);
    let keep = select(w.look_b.x, 1.0, w.kind.y > 0.5);
    if (experimental) {
        // The experimental tier's water is calm: the sea keeps its whole sea, a river or lake
        // `look_b.x` of it (its authored bump is its own) with the wind's ripples over it, and water
        // with little under it goes near glassy — a hand-deep stream does not carry a 17-yard swell.
        let calm = mix(w.look_b.y, 1.0, smoothstep(0.0, MODERN_WAVE_FULL_DEPTH, column_v));
        swell = (swell + wind * keep + ripples) * calm;
        let sp2 = w.shade_params.x * w.shade_params.x;
        let shared_lost = vec3<f32>(bump_tap.lost, bump_tap.lost, 0.0) + wind_lost * (keep * keep);
        lost_s = (shared_lost + ripples_lost * (w.look_c.w * w.look_c.w)) * (calm * calm * sp2);
        let lost_h = (shared_lost + ripples_lost * (w.look_d.z * w.look_d.z)) * (w.look_d.y * w.look_d.y * sp2);
        rough_s = surface_roughness(w.look_b.w, w.look_e.y, lost_s);
        rough_h = surface_roughness(w.look_b.w, w.look_e.y, lost_h);
    }
    // fork: the crests' lift on the body's light: the long swell's height (the trains the surface
    // carries as height), at the share the calm surface keeps — thin water at a crest lets more light
    // through than the deeper column in a trough.
    let crest = 1.0 + CREST_SHADE * clamp(sea_height * keep / CREST_HEIGHT_YD, -1.0, 1.0);
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
    // fork: modern lighting — the experimental tier's highlight has a normal of its own: the calm
    // surface is what the bed and the sky are read through, but the sun's path needs the facets the
    // shallows' calm (`look_b.y`) takes away to spread its islands wide. It keeps each kind's own
    // octaves — the authored bump, the wind sea at the share the calm surface keeps (all of it on the
    // sea, `look_b.x` on a river or lake), and the wind ripples on the sea alone — `look_d.y` of them,
    // whatever the depth. Its roughness is widened here, in uniform control flow, by how far that
    // normal turns per pixel.
    var spec_n = n;
    var spec_alpha = w.look_b.w;
    if (experimental && modern) {
        let full = (bump + wind * keep + spec_ripples) * w.look_d.y;
        let spec_grad = normalize(vec3<f32>(full + ripple_slope(in.world_position.xz, full), 1.0));
        spec_n = normalize(vec3<f32>(spec_grad.x * w.shade_params.x, 1.0, spec_grad.y * w.shade_params.x));
        // fork: the lobe is as wide as the facets the normal no longer draws (`rough_h`), and then
        // widened again by how far the drawn normal turns across the pixel.
        spec_alpha = specular_aa(spec_n, tensor_alpha(rough_h));
    }
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
    // fork: the sun's reach on the bed (its shadow there, under the cloud), for the caustics.
    var bed_sun = 1.0;
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
#ifdef RT_SHADOWS
        // fork: F — the cascades hold the entities alone; one ray each takes the static world and
        // the terrain, at the surface and at the bed probe.
        let to_key = -normalize(wow_light.light_sun.xyz);
        let sun_surface = sunshadow::sun_visibility_load(modern_shadow_map, modern_block.cascades, lifted, frag_view_z)
            * rtshadow::sun_visibility_ray(modern_block.cascades, lifted, to_key);
#else
        let sun_surface = sunshadow::sun_visibility_load(modern_shadow_map, modern_block.cascades, lifted, frag_view_z);
#endif
        let deep = select(smoothstep(WATER_SHADOW_PROBE, 2.0 * WATER_SHADOW_PROBE, column_v), 1.0, below);
        var body_sun = sun_surface;
        if (deep < 1.0) {
            let probe_d = max(ray_len - WATER_SHADOW_BED_LIFT, 0.0);
            // The probe sits on this pixel's ray, so its planar depth scales with the distance along it.
            let probe_z = frag_view_z * (1.0 + probe_d / max(view_len, 0.001));
#ifdef RT_SHADOWS
            let sun_bed = sunshadow::sun_visibility_load(modern_shadow_map, modern_block.cascades, lifted + v * probe_d, probe_z)
                * rtshadow::sun_visibility_ray(modern_block.cascades, lifted + v * probe_d, to_key);
#else
            let sun_bed = sunshadow::sun_visibility_load(modern_shadow_map, modern_block.cascades, lifted + v * probe_d, probe_z);
#endif
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
        bed_sun = body_sun * cloud;
    }
    let body = ramp.rgb * water_lit;

    // ---- Refraction ----------------------------------------------------------------------------
    //
    // Offset the screen read by the bump, scaled by how much water is actually in front of what is
    // behind it, then GUARD the result: if the offset lands on something other than what this
    // pixel looks at (`refract_trust`), fall back to the unoffset read. Without that guard a wave
    // at the shoreline bends the riverbank *over* the water — the classic refraction bleed, and the
    // reference spends a `cmp` on it for the same reason.
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
        // fork: and no further than the water lets the bed be seen: past `ln 10 / σ̄` less than a
        // tenth of it survives, and bending what cannot be seen only smears the body.
        let sigma_mean = dot(w.optics.xyz, vec3<f32>(1.0 / 3.0)) + max(w.look_e.x, 0.0);
        let reach = min(min(ray_len, MODERN_REFRACT_MAX_PATH), 2.302585 / max(sigma_mean, 1e-3));
        let bend = min(w.look_b.z * reach * per_yd, 0.1 * offset_amount);
        refract_uv = uv + n.xz * bend;
    }
    // Deviation: a bent read is kept only where it lands on the surface this pixel's own ray sees
    // (`refract_trust`); the reference rejects it only when it lands nearer than the water (`cmp`
    // against the surface depth), because under that guard alone a swimmer showed twice: the water beside
    // the body, over a deep bed, bent its read onto the body, and the water over the body bent its
    // read off onto the bed, so the submerged half drew as a shifted copy. A read nearer than the
    // surface has no water in front of it, so the reference's own case is the trust's zero.
    let trust = refract_trust(scene_depth_at(refract_uv), scene_z, frag_view_z);
    refract_uv = uv + (refract_uv - uv) * trust;
    var bed_z = scene_depth_at(refract_uv);
    if (refract_trust(bed_z, scene_z, frag_view_z) < 0.5) {
        refract_uv = uv;
        bed_z = scene_z;
    }
    if (experimental) {
        // fork: and only where it lands on BED: a read whose point stands more than its own
        // sideways distance (45°) above the bed under this pixel is a reed, a post or a rock face
        // seen through the water, not the floor, and it fades back to the unbent read. The depth goes
        // with it, so the column `modern_water` filters is the one under what it shows.
        let bed_here = in.world_position.xyz + v * ray_len;
        let seen = view_point_at(refract_uv, bed_z);
        let rise = length(seen.xz - bed_here.xz);
        let accept = 1.0 - smoothstep(rise, rise + 0.15, seen.y - bed_here.y);
        refract_uv = uv + (refract_uv - uv) * accept;
        bed_z = mix(scene_z, bed_z, accept);
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
    // fork: the soft shore, the experimental tier's: the whole water layer comes in over the first
    // `SHORE_FADE_YD` of vertical column as `t·(2 − t)` (three quarters there at half way, so a
    // shallow shelf still reads as water), never narrower than a few pixels' worth of the column's
    // own change, so a distant bank fades rather than stair-steps. Past ~half a yard of fade whole
    // lagoons go clear; the fork that measured that settled on a third of a yard.
    let shore_t = clamp(column_v / max(SHORE_FADE_YD, 3.0 * column_rate), 0.0, 1.0);
    let shore_cover = select(1.0, shore_t * (2.0 - shore_t), experimental);

    // fork: the caustics on the bed (`caustic_net`), the experimental tier's on a modern frame: the
    // bed under this pixel, reached along the unrefracted ray, is lit through the point where the
    // sun's refracted ray entered the surface above it. Only the sun's direct share of the bed's light
    // is redistributed — none in the bed's shadow — and the net fades with the path and turbidity.
    var caustic = 1.0;
    if (experimental && modern && w.look_e.z > 0.0 && to_light.y > 0.0) {
        let t_ref = refract(-to_light, vec3<f32>(0.0, 1.0, 0.0), 1.0 / WATER_IOR);
        let path_c = column_v / max(-t_ref.y, 0.2);
        let fade = (1.0 - smoothstep(CAUSTIC_FADE_YD.x, CAUSTIC_FADE_YD.y, path_c))
            * exp(-CAUSTIC_TURB * max(w.look_e.x, 0.0) * path_c);
        let sun_l = colorspace::luminance(wow_light.light_diffuse.rgb) * to_light.y * bed_sun;
        let sky_l = colorspace::luminance(wow_light.light_ambient.rgb);
        let share = sun_l / max(sky_l + sun_l, 1e-4);
        let amp = fade * share * w.look_e.z;
        if (amp > 1.0 / 256.0) {
            let bed_p = in.world_position.xyz + v * ray_len;
            let entry = bed_p.xz - t_ref.xz * path_c;
            let foot_bed = foot * (view_len + ray_len) / max(view_len, 1e-3);
            caustic = max(1.0 + amp * (caustic_net(entry, path_c, anim_time(), foot_bed) - 1.0), 0.0);
        }
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
        var mirror_point = in.world_position.xyz + mirror_ray * 20.0;
        if (experimental) {
            // fork: the experimental tier reads the mirror where a planar reflection is: the mirror
            // image of a point `MIRROR_HIT_YD` along the flat reflected ray, which lies on this
            // pixel's own view ray, moved by how far the WAVE's reflected ray (folded back above the
            // surface) leans off the flat one, a share of the way to the hit — far banks wobble,
            // contact points stay put. The world point moves rather than the UV: at grazing a UV shift
            // shows the same content twice. A surface off the mirror's plane reads the image of its
            // own height (`2·(plane − y)`).
            let r_flat = reflect(v, vec3<f32>(0.0, 1.0, 0.0));
            let dr = fold_above(reflect(v, n)) - r_flat;
            mirror_point = in.world_position.xyz
                + (v + vec3<f32>(dr.x, -dr.y, dr.z) * MIRROR_WAVE_REACH) * MIRROR_HIT_YD
                + vec3<f32>(0.0, 2.0 * (w.path.w - in.world_position.y), 0.0);
        }
        let mirror_clip = view.clip_from_world * vec4<f32>(mirror_point, 1.0);
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
        // fork: the roughness above the calm floor, which blurs what the surface reflects.
        var alpha_lost = 0.0;
        if (experimental) {
            // fork: modern lighting — the experimental tier keeps this whole lookup and puts the sky
            // dome along the reflected ray in the horizon constant's place: one colour for every
            // direction (in a jungle zone an olive haze) read as goo over a lake whose mirror had not
            // rendered yet; the dome is the backdrop the player can see. fork: blurred by the
            // surface's unresolved roughness, along the ray folded back above the horizon.
            let a_s = tensor_alpha(rough_s);
            alpha_lost = sqrt(max(a_s * a_s - w.look_b.w * w.look_b.w, 0.0));
            base = sky_dome_rough(fold_above(r), alpha_lost);
        }
        if (w.path.z > 0.5) {
            var mirror_raw: vec3<f32>;
            if (experimental) {
                // fork: through the rough surface, graded toward the mirror's own local mean.
                let a_s = tensor_alpha(rough_s);
                let rho_rough = smoothstep(w.look_b.w, max(w.look_e.w, w.look_b.w + 1e-3), a_s);
                let rho_far = smoothstep(GRADE_NEAR_YD, GRADE_FAR_YD, view_len);
                let rho = 1.0 - (1.0 - rho_rough) * (1.0 - rho_far);
                mirror_raw = mirror_read_rough(mirror_uv, lost_s, v, view_len, rho);
            } else {
                mirror_raw = textureSampleLevel(reflect_map, scene_samp, mirror_uv, 0.0).rgb;
            }
            let mirror = colorspace::modern_in(mirror_raw, wow_light.point_count.y);
            base = mix(base, mirror, mirror_valid);
        }
        // `ADDR R3.xyz, R3, -R5` / `MADR R3.xyz, R1.w, R3, R5` — the mirror is the BASE and the
        // screen-space hit is lerped over it by the same validity perm 4 lerps the horizon by.
        //
        // fork: the experimental tier lays the hit only where the mirror is not valid: a read off the
        // screen changes as its source leaves the frame (the same island graded about grey on screen
        // and about the sky past its edge), and the mirror under it never does.
        var ssr_share = validity;
        if (experimental) {
            ssr_share = validity * (1.0 - mirror_valid * w.path.z);
        }
        var reflected = mix(base, hit, ssr_share);
        if (experimental) {
            // fork: modern lighting — the same reflection, softened toward the body and laid over
            // `modern_water`'s tinted bed at its lively Fresnel instead of the two steps below.
            var x: ClearInputs;
            x.refl = colorspace::srgb_to_linear(reflected);
            // The mirror's world and the screen's hit were fogged where they were drawn; the dome
            // under them was not.
            x.refl_fogged = 1.0 - (1.0 - mirror_valid * w.path.z) * (1.0 - ssr_share);
            x.rough = rough_s;
            x.lost = lost_s;
            x.caustic = caustic;
            x.crest = crest;
            x.to_light = to_light;
            x.sun = water_sun;
            x.auth_alpha = swatch(in.depth_v).a;
            let clear = modern_water(in, n, v, bed_raw, bed_z, frag_view_z, swatch(1.0).rgb, water_lit, x);
            // The tier sits between the clear surface and the reference's combine: `look_c.z` of
            // the reference's own (the swatch over the refraction, the reflection pulled to the
            // body), over the same calm normal, the same reflection, and the same bed read.
            let cata = mix(refracted, mix(reflected, body, 0.6), fres2 * 0.5 + 0.1);
            let k = w.look_c.z;
            // fork: the soft shore — the whole water layer comes in over the first fraction of a yard
            // of column (`shore_cover`), over the bed seen straight through.
            let here_raw = textureSampleLevel(scene_color, scene_samp, uv, 0.0).rgb;
            let here = select(colorspace::srgb_to_linear(here_raw), here_raw, modern);
            let total = mix(here, mix(clear.rgb, colorspace::srgb_to_linear(cata), k), shore_cover);
            // fork: fog once — on a modern frame what already carries its fog (the bed, the
            // reflected world, the bank at the waterline) leaves here linear and goes round the
            // wrapper's surface fog; the rest is fogged at the surface as before.
            if (modern) {
                modern_prefogged = mix(here, (1.0 - k) * clear.fogged, shore_cover);
            }
            // Linear out, re-encoded so the wrapper's decode hands it back unchanged.
            rgb = colorspace::linear_to_srgb(max(total - modern_prefogged, vec3<f32>(0.0)));
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
        // fork: all of it at the soft shore's coverage, like the surface it sits on.
        let k = w.look_c.z * shore_cover;
        let glare = clamp(dot(to_light, n), 0.0, 1.0) * w.shade_params.w;
        rgb += k * glare * wow_light.light_diffuse.rgb * body * water_sun;
        rgb += k * w.shade_params.z * spec_color * fres2 * water_sun;
        if (!modern) {
            // A classic frame has no glitter path to carry the sun (that is HDR, added by the
            // linear wrapper), so the clear share keeps the reference's Blinn track there —
            // without the sheen floor, whose flat grazing lift the reflection carries.
            rgb += (1.0 - w.look_c.z) * shore_cover * w.shade_params.z * wow_light.light_spec.rgb * spec * fres2;
        }
    }
    // fork: modern lighting — the sun's glitter path: a tight HDR lobe on the reflected ray over a
    // broad sheen, only where the sun (or the moon, which keys the light by night) reaches the
    // surface. Added in linear by the wrapper.
    if (modern_above && to_light.y > -0.05) {
        // Warmed by the zone's scatter colour (the froxel key in-scatter, normalised), so a dawn
        // glitter path takes the haze's salmon instead of the key row's white.
        let scatter = modern_block.fog.sun.rgb;
        let warm = scatter / max(max(scatter.r, max(scatter.g, scatter.b)), 1e-4);
        let key = pow(wow_light.light_diffuse.rgb, vec3<f32>(2.2)) * mix(vec3<f32>(1.0), warm, 0.6);
        let horizon = smoothstep(-0.05, 0.08, to_light.y);
        if (experimental) {
            // The experimental tier's highlight is a microfacet lobe off the light's disk on its own
            // rough normal (`disk_highlight`): Forever's PBR water, its path broken into flat-topped
            // islands as bright as the disk, rather than a smooth glow on a calm surface.
            modern_glint = glint_knee(key * (disk_highlight(spec_n, -v, to_light, spec_alpha, w.look_d.x) * w.look_d.w * water_sun * horizon)) * shore_cover;
        } else {
            let rv = reflect(v, n);
            let sd = clamp(dot(rv, to_light), 0.0, 1.0);
            modern_glint = key * ((pow(sd, 900.0) * 9.0 + pow(sd, 160.0) * 0.8 + pow(sd, 24.0) * 0.12) * water_sun * horizon * (0.35 + 0.65 * fres));
        }
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
    modern_prefogged = vec3<f32>(0.0);
    let c = fragment_classic(in);
    var rgb = colorspace::modern_out(c.rgb, wow_light.point_count.y) + modern_glint;
    // P4: transparent water draws after the fog composite, so it fogs itself; opaque magma and slime
    // (`kind.x`) draw before it and are fogged there.
    if (wow_light.point_count.y > 0.5 && wow_light.fog_color.w > 0.5 && w.kind.x < 0.5) {
        let air = modern_block.fog;
        let fog = froxelfog::fog_at(modern_fog_volume, scene_samp, air, in.world_position.xyz);
        rgb = rgb * fog.a + fog.rgb;
    }
    // fork: fog once — what already carried its fog when it was drawn (the experimental tier's bed and
    // reflected world, `fragment_classic`) joins after the surface's fog, not under it.
    rgb = rgb + modern_prefogged;
    return vec4<f32>(rgb, c.a);
}
