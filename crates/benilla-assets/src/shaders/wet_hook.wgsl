#define_import_path benilla::wet_hook

// fork (Everwood port): rain on surfaces, the one place the wet look is computed. Receivers
// (terrain, static_gx, wow_model) call it with one-line hooks and pass the EnvFrame (its `wet_a`
// row: `rain_rate, wetness, ripple_time_s, snow`, written by
// `benilla-world/src/weather/wetness.rs`). A zero wetness (dry weather, or `rainSurfaces 0`)
// returns the inputs untouched, so the dry image is bit-identical.
//
// Sky exposure: the receiver passes `exposure` = 0 for interior-class, unlit and emissive
// surfaces and the shelter map's open share otherwise; the normal then takes undersides out (dry),
// walls half wet, tops fully wet and glossy.

#import benilla::env_frame::EnvFrame

// fork: the four cells around a fragment in the camera-centred shelter grid.
// `hdr` = `[origin_x, origin_z, 1 / cell, active]`, `cfg` = `[base_y, cells per side, 0, 0]`;
// `live` is 0 when the map is off or the fragment lies outside it (open sky).
struct ShelterTaps {
    idx: vec4<u32>,
    frac: vec2<f32>,
    live: f32,
}

fn shelter_taps(world_pos: vec3<f32>, hdr: vec4<f32>, cfg: vec4<f32>) -> ShelterTaps {
    var t: ShelterTaps;
    t.idx = vec4<u32>(0u);
    t.frac = vec2<f32>(0.0);
    t.live = 0.0;
    if (hdr.w < 0.5) {
        return t;
    }
    let g = (world_pos.xz - hdr.xy) * hdr.z - 0.5;
    let gf = floor(g);
    if (gf.x < 0.0 || gf.y < 0.0 || gf.x + 1.0 > cfg.y - 1.0 || gf.y + 1.0 > cfg.y - 1.0) {
        return t;
    }
    let side = u32(cfg.y);
    let first = u32(gf.y) * side + u32(gf.x);
    t.idx = vec4<u32>(first, first + 1u, first + side, first + side + 1u);
    t.frac = g - gf;
    t.live = 1.0;
    return t;
}

// One cell: low half `(top − base_y) × 64 + 32768`, high half the soft cover as unorm16
// (`shelter::pack_cell`). Sheltered when a covered top lies a yard or more above the fragment; the
// roof's own surface (at `top`) stays open.
fn shelter_cell(word: u32, y_rel: f32) -> f32 {
    let top = (f32(word & 0xffffu) - 32768.0) / 64.0;
    let cover = f32(word >> 16u) / 65535.0;
    return cover * smoothstep(0.5, 1.5, top - y_rel);
}

// How sheltered the fragment is, 0 open .. 1 under cover, bilinear over the four cells.
fn shelter_amount(t: ShelterTaps, wa: u32, wb: u32, wc: u32, wd: u32, world_y: f32,
    cfg: vec4<f32>) -> f32 {
    if (t.live <= 0.0) {
        return 0.0;
    }
    let y_rel = world_y - cfg.x;
    let near = mix(shelter_cell(wa, y_rel), shelter_cell(wb, y_rel), t.frac.x);
    let far = mix(shelter_cell(wc, y_rel), shelter_cell(wd, y_rel), t.frac.x);
    return clamp(mix(near, far, t.frac.y), 0.0, 1.0);
}

// fork: the open share of ground that carries its own depth under the water line (yards, positive
// submerged; terrain packs it per vertex, `benilla_assets::adt::pack_water_depth`): the shore is
// wet to the water's edge, and nothing under the surface darkens, shines or holds a puddle. The
// shelter grid's water roof covers only the yards around the camera and only a bed a yard down;
// this reaches every drawn chunk and the shallows.
fn above_water(depth: f32) -> f32 {
    return 1.0 - smoothstep(0.0, 0.15, depth);
}

struct WetSurface {
    albedo: vec3<f32>,
    // Gloss weight for `wet_sheen`: 0 dry, ~1 a soaked top face, up to 2 in a terrain puddle (the
    // part above 1 is standing water: a flat mirror of the sky).
    boost: f32,
}

// How wet a surface is from its facing: none below, half on walls, full on tops.
fn wet_amount(n: vec3<f32>, exposure: f32, wetness: f32) -> f32 {
    let facing = smoothstep(-0.2, 0.25, n.y);
    return clamp(wetness, 0.0, 1.0) * clamp(exposure, 0.0, 1.0) * facing
        * mix(0.5, 1.0, smoothstep(0.3, 0.8, n.y));
}

// Wet albedo: darker (water fills the micro-pores) and a touch more saturated; the gloss weight
// is kept for up-facing faces only.
fn wet_surface(albedo: vec3<f32>, n: vec3<f32>, world_pos: vec3<f32>, exposure: f32,
    ef: EnvFrame) -> WetSurface {
    let wet_row = ef.wet_a;
    var o: WetSurface;
    o.albedo = albedo;
    o.boost = 0.0;
    if (wet_row.y <= 0.0 || exposure <= 0.0) {
        return o;
    }
    let w = wet_amount(n, exposure, wet_row.y);
    let luma = dot(albedo, vec3<f32>(0.299, 0.587, 0.114));
    let sat = max(mix(vec3<f32>(luma), albedo, 1.0 + 0.3 * w), vec3<f32>(0.0));
    o.albedo = sat * (1.0 - 0.36 * w);
    o.boost = w * smoothstep(0.35, 0.9, n.y);
    return o;
}

fn wet_hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn wet_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(wet_hash(i), wet_hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(wet_hash(i + vec2<f32>(0.0, 1.0)), wet_hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// Terrain bonus: puddle patches on flat ground once it is well soaked — darker, and a stronger
// gloss weight so they read as standing water catching the sky. fork: `exposure`
// as for `wet_surface`, so no puddle forms under cover.
fn wet_puddles(s: WetSurface, n: vec3<f32>, world_pos: vec3<f32>, exposure: f32,
    ef: EnvFrame) -> WetSurface {
    let wet_row = ef.wet_a;
    var o = s;
    if (wet_row.y <= 0.0 || exposure <= 0.0) {
        return o;
    }
    let xz = world_pos.xz;
    let field = 0.65 * wet_noise(xz * 0.17) + 0.35 * wet_noise(xz * 0.53 + 17.0);
    let level = smoothstep(0.94, 0.99, n.y);
    // Puddles grow with the wetness: the threshold falls as the ground soaks.
    let edge = mix(0.8, 0.6, smoothstep(0.4, 1.0, wet_row.y));
    let puddle = smoothstep(edge, edge + 0.06, field) * level * smoothstep(0.35, 0.8, wet_row.y)
        * clamp(exposure, 0.0, 1.0);
    o.albedo = o.albedo * (1.0 - 0.45 * puddle);
    o.boost = max(o.boost, 2.0 * puddle);
    return o;
}

// The wet sun highlight: tight, shadowed, `gain` × the sheen's own strength. Zero when `boost` is.
fn wet_sun(boost: f32, n: vec3<f32>, to_view: vec3<f32>, to_light: vec3<f32>,
    sun_rgb: vec3<f32>, shadow: f32, gain: f32) -> vec3<f32> {
    if (boost <= 0.0) {
        return vec3<f32>(0.0);
    }
    // fork: nudged so opposite light/view vectors never normalize a zero vector (NaN).
    let h = normalize(to_light + to_view + vec3<f32>(0.0, 1.0e-4, 0.0));
    let nl = max(dot(n, to_light), 0.0);
    let spec = pow(max(dot(n, h), 0.0), 80.0) * smoothstep(0.0, 0.15, nl);
    let gloss = min(boost, 1.0);
    let mirror = max(boost - 1.0, 0.0);
    return sun_rgb * ((0.9 * gloss + 1.2 * mirror) * spec * shadow * gain);
}

// How much of the surroundings a wet surface reflects, Fresnel-weighted: a faint grazing gloss on
// soaked faces, a mirror in a puddle. The sky colour times this, or the screen-space reflection
// (`benilla-world/src/weather/wet_ssr.rs`) where the receiver tags it. Below 0.75 always.
fn wet_reflectance(boost: f32, n: vec3<f32>, to_view: vec3<f32>) -> f32 {
    if (boost <= 0.0) {
        return 0.0;
    }
    let fres = 0.03 + 0.97 * pow(1.0 - clamp(dot(n, to_view), 0.0, 1.0), 5.0);
    let gloss = min(boost, 1.0);
    let mirror = max(boost - 1.0, 0.0);
    return gloss * min(0.25 * fres, 0.2) + mirror * (0.22 + 0.3 * fres);
}

// Whether the wet screen-space reflection pass composites this frame (`EnvFrame.misc.y`). A
// receiver that tags its output alpha must only do so while this is set.
fn ssr_live(ef: EnvFrame) -> bool {
    return ef.misc.y > 0.5;
}

// The models' share of the wet sun highlight and of the sky reflection. A model's smooth-shaded
// normals spread the highlight into a broad disc that clips to white at full strength; wet stone
// and bark should only darken and catch a faint sheen.
const MODEL_SUN_GAIN: f32 = 0.12;
const MODEL_SKY_GAIN: f32 = 0.5;

// The wet sheen on models (static_gx, wow_model), added to the lit colour (gamma in a classic
// frame, linear in a modern one): a faint sun highlight (shadowed) and a Fresnel reflection of the
// sky/fog colour. Zero when `boost` is zero.
fn wet_sheen(boost: f32, n: vec3<f32>, to_view: vec3<f32>, to_light: vec3<f32>,
    sun_rgb: vec3<f32>, sky_rgb: vec3<f32>, shadow: f32) -> vec3<f32> {
    return wet_sun(boost, n, to_view, to_light, sun_rgb, shadow, MODEL_SUN_GAIN)
        + sky_rgb * (MODEL_SKY_GAIN * wet_reflectance(boost, n, to_view));
}
