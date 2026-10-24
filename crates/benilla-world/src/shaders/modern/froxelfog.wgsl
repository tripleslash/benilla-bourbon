// Modern lighting (FORK.md, P4): the froxel fog volume's parameters, its lookup, and
// the analytic height fog that stands in for it beyond its far plane and on the Low tier.
//
// Imported as `benilla_modern::froxelfog` — a module name no hooked shader uses as an identifier
// (naga_oil substitutes a module alias for every bare identifier of that name).
//
// No bindings are declared here: each lane binds the integrated volume and its sampler itself and
// passes them in, beside the parameter block (`sunshadow::ModernBlock.fog`).
#define_import_path benilla_modern::froxelfog

// Mirrors `benilla_world::modern::volumetrics::FroxelFog::std430` (a layout test pins the size).
struct FroxelFog {
    // The main world camera's unjittered clip-from-world. Every lane looks the volume up through it,
    // the water mirror included: a reflected point is fogged as the main camera sees it.
    clip_from_world: mat4x4<f32>,
    // Its inverse: the compute pass walks froxels back into the world.
    world_from_clip: mat4x4<f32>,
    // Last frame's clip-from-world, for the scatter history's reprojection.
    prev_clip_from_world: mat4x4<f32>,
    // xyz = the main camera's world position; w = seconds (the noise clock).
    camera: vec4<f32>,
    // The main camera's view-from-world third row: planar view depth = -(dot(xyz, p) + w).
    view_row: vec4<f32>,
    // xyz = this tier's froxel grid; w = its far plane (planar yd).
    grid: vec4<f32>,
    // xyz = the integrated volume texture's allocated size; w = mode (0 off, 1 analytic, 2 froxels).
    alloc: vec4<f32>,
    // x = fog extinction at the reference height (1/yd); y = 1 / height falloff (yd); z = reference
    // height (the regional ground); w = fog start (planar yd).
    media: vec4<f32>,
    // x = shaft haze extinction (1/yd, never near-cleared); y = noise amount; z = shaft key radiance (r);
    // w = scatter history weight (0 on a cut).
    medium: vec4<f32>,
    // rgb = ambient in-scatter radiance (albedo x ambient); w = shaft key radiance (g).
    ambient: vec4<f32>,
    // rgb = key-light in-scatter radiance (albedo x key); w = the forward lobe's share.
    sun: vec4<f32>,
    // xyz = unit direction TOWARD the key light; w = the distant silhouette's shade (the WDL band's).
    sun_dir: vec4<f32>,
    // x = the backward lobe's g; y = the view distance (farclip, yd); z = this frame's slice jitter [0, 1); w = frame index.
    phase: vec4<f32>,
    // xy = the cloud-shadow grid's origin (world x, z); z = cell size (yd); w = cells per side.
    cloud: vec4<f32>,
    // x = near plane; y = 1 / clip_from_view[0][0]; z = 1 / clip_from_view[1][1]; w = screen-shaft strength.
    proj: vec4<f32>,
    // xy = the key light's uv in the main view; z = its on-screen weight; w = froxel point-light count.
    screen: vec4<f32>,
    // xyz = noise scroll (yd); w = shaft key radiance (b).
    wind: vec4<f32>,
    // fork: modern lighting — THE ROOM THE CAMERA STANDS IN. xyz = the centre of its MOGI box in
    // world space, w = its radius (yd); w = 0 means the camera is in no room and the whole lane is
    // off. A room's MFOG is a room-scale wall (`volumetrics::media` spends WALL_TAU over the
    // record's own end, uniformly and with no near-clear range), and a room's air ends at its
    // walls: blended into one medium it followed the view out of the doorway, and Orgrimmar's gate
    // hall — a 30 yd tunnel carrying a 292 yd record — wrapped the city in the building's teal.
    room: vec4<f32>,
    // x = the room medium's extinction (1/yd, uniform); y = its weight (the classic 4 s interior
    // crossfade); z = the night air's extinction (1/yd); w = the night air's glow (r).
    room_media: vec4<f32>,
    // rgb = the room's ambient in-scatter radiance; w = the night air's glow (g).
    room_ambient: vec4<f32>,
    // rgb = the room's key-light in-scatter radiance; w = the night air's glow (b).
    room_sun: vec4<f32>,
    // fork: the night's ground banks. x = extinction at the ground (1/yd, 0 by day); y = the ground
    // under the camera (world y); z = the key's share on their isotropic half (`halo_air.w`, up to 1
    // where the zone look's banks are ground fog: `volumetrics::bank_air_share`); w = the sea fog's
    // extinction at sea level (1/yd, the zone look's `sea_fog`; `froxel_scatter::sea_fog`).
    night: vec4<f32>,
    // fork: the terrain heightfield the banks lie on (`probegi::GiClip`'s `hf_row_x`, `hf_row_y` and
    // `hf`: x = enabled, y = lowest height, z = span, w = texels), written in the render world.
    ground_x: vec4<f32>,
    ground_y: vec4<f32>,
    ground: vec4<f32>,
    // fork: the lamp halos' air — the zone's medium solved in clear weather (`volumetrics::halo_air`):
    // x = fog extinction (1/yd), y = near-clear range (planar yd), z = haze extinction; w = the key's
    // share on the shaft haze's and banks' isotropic half (`look::key_air_share`).
    halo_air: vec4<f32>,
}

// fork: **the air a lamp's halo stands in** — `f` with the open air's medium swapped for its
// clear-weather solve. Rain and storms thicken the medium (no near-clear range, more haze, a
// shorter fog end), and a halo scattered by that medium swelled with the weather; scattered by
// this one it stays the lamp's own, and the storm only dims it through the fog in front of it.
fn halo_fog(f: FroxelFog) -> FroxelFog {
    var g = f;
    g.media.x = f.halo_air.x;
    g.media.w = f.halo_air.y;
    g.medium.x = f.halo_air.z;
    return g;
}

// fork: **a lamp's halo is the same from any distance.** The lamps scatter off the air as it is
// `LAMP_EYE_DEPTH` yards from the eye (inside the near-clear range, at the night air's near share),
// not at their own depth: there the fog thickens past the near-clear range and the night air
// triples by 80 yd, so a camp's torch stood in a wide bright bloom from afar and shrank to its
// glow a step closer. 1, not 0: with no near-clear range `fog_clear`'s smoothstep has equal edges.
// That air scatters `LAMP_EYE_GAIN` times harder, a little of the glow the near look gave up, far
// short of the distant bloom (up to ~2.9 times the air).
const LAMP_EYE_DEPTH: f32 = 1.0;
const LAMP_EYE_GAIN: f32 = 1.25;

// Under full cover the shaft haze is this much denser (`canopy_cover` in `froxel_scatter.wgsl`).
const CANOPY_HAZE: f32 = 1.5;
// fork: **a room's air stays clear.** Inside a building (the room weight) the open air's shaft haze
// keeps `ROOM_HAZE_SHARE`: an inn's candles over the zone's haze put a grey veil on the common room.
const ROOM_HAZE_SHARE: f32 = 0.15;
// fork: the night air keeps `NIGHT_NEAR_SHARE` of itself at the camera, ramping to all of it between
// `NIGHT_CLEAR_NEAR` and `NIGHT_CLEAR_FAR` yards of planar depth: the street stays readable, the
// distance layers into the haze, and the lamps beside the player still have air to light.
const NIGHT_NEAR_SHARE: f32 = 0.35;
const NIGHT_CLEAR_NEAR: f32 = 10.0;
const NIGHT_CLEAR_FAR: f32 = 80.0;

// The medium's extinction at world point `p`, `d` yards of planar depth from the main camera, per
// yard of ray: x = the fog, y = the shaft haze, z = the night air. `room` is `room_weight` at `p`,
// `wisps` the drifting noise's scale on the fog (1 = its mean), `cover` the canopy cover (0 = open).
fn extinction(f: FroxelFog, p: vec3<f32>, d: f32, room: f32, wisps: f32, cover: f32) -> vec3<f32> {
    let density = height_density(f, p.y);
    // fork: modern lighting — a room's medium reaches as far as the room does: inside its walls the
    // building's MFOG (uniform, no near-clear range), past them the zone's own air.
    let open_fog = f.media.x * fog_clear(f, d) * density;
    let fog = mix(open_fog, f.room_media.x, room) * wisps;
    let haze = f.medium.x * density * (1.0 + CANOPY_HAZE * cover) * mix(1.0, ROOM_HAZE_SHARE, room);
    let night = f.room_media.z * density
        * mix(NIGHT_NEAR_SHARE, 1.0, smoothstep(NIGHT_CLEAR_NEAR, NIGHT_CLEAR_FAR, d));
    return vec3<f32>(fog, haze, night);
}

// ── The point lights' in-scatter, shared by the froxels and the composite's per-pixel halo ──────
// Point lights' in-scatter against their surface lighting.
const POINT_GAIN: f32 = 0.35;
// fork: a lamp's glow in the haze leans a little toward the eye (smoke scatters forward): the share
// of a Henyey-Greenstein lobe of `POINT_G` mixed over the isotropic term.
const POINT_G: f32 = 0.5;
const POINT_FORWARD: f32 = 0.3;
// fork: **the glow sits at the flame.** A light's in-scatter keeps `POINT_FAR_SHARE` of itself past
// its halo and all of it within `POINT_HALO_YD` of the source, so a brazier stands in a small bright
// halo while a street of lit windows (each a light, 48 yd of range) no longer fills the air around
// the camera with a grey glow. At the flame it scatters `POINT_HALO_GAIN` times its far-field share,
// so a torch or a brazier stands in a visible glow (the Forever Orgrimmar night shot's lamps). At
// 4.0 the halo read far brighter than the lamp it stands around, 2.5 vanished in clear air; 3.0
// keeps 0.75 of it.
const POINT_HALO_YD: f32 = 4.0;
const POINT_FAR_SHARE: f32 = 0.35;
const POINT_HALO_GAIN: f32 = 3.0;
// fork: inside a building a light keeps nothing past its halo and its halo shrinks to
// `ROOM_POINT_HALO_YD` at `ROOM_POINT_HALO_GAIN`: an inn's dozen candles each filling their range
// put a grey veil on the whole common room.
const ROOM_POINT_HALO_YD: f32 = 1.5;
const ROOM_POINT_HALO_GAIN: f32 = 1.5;

// fork: **a lamp stands in its own air.** A light's halo excess also scatters in a small pocket of
// air around it: `LAMP_AIR` times the night air's extinction (`room_media.z`, zero by day, full at
// the SIDN night) at the light, falling off over `LAMP_AIR_YD`. The zone's own air is a few
// thousandths per yard, and a flame's halo lit in it alone came out at a few hundredths of the
// night scene's radiance — no halo at all; the pocket gives a flame a core glow about two yards
// across instead of a lit fog filling its whole range. 15, not 20: 0.75 of the core, with the halo.
const LAMP_AIR: f32 = 15.0;
const LAMP_AIR_YD: f32 = 1.0;

// How much lamp air a table light stands in, from its colour row's `w` (0/1 the indoor class, plus
// `global_light::WINDOW_TAG` 0.25 for a window's night glow): a lit pane is not a flame, and a
// facade of window lights each standing in lamp air put a grey fog over the whole house.
fn lamp_air_share(class_w: f32) -> f32 {
    return select(1.0, 0.0, fract(class_w) > 0.125);
}

// A window's light (`global_light::WINDOW_TAG` 0.25 over the indoor class).
fn is_window(class_w: f32) -> bool {
    return abs(fract(class_w) - 0.25) < 0.0625;
}

// fork: **only a flame stands in a halo.** A window's light (a pane is not a flame) and a light
// nothing on screen draws (`global_light::GLOWLESS_TAG` 0.375, a WMO's MOLT light) scatter their
// far share like any light and no halo: theirs floated a glow in the air a yard off the facade,
// under the window, where no lamp hangs.
fn has_halo(class_w: f32) -> bool {
    return fract(class_w) <= 0.125;
}

// The lamp air's extinction at its peak (1/yd) for a light of colour-row `w`: scale it by
// `exp(-dl / LAMP_AIR_YD)` at distance `dl` from the light.
fn lamp_air(f: FroxelFog, class_w: f32) -> f32 {
    return LAMP_AIR * f.room_media.z * lamp_air_share(class_w);
}

// fork: **the halo is integrated per pixel.** A lamp among the table's first `halo_lights` (the table
// is packed nearest-to-camera first) scatters only its far share into the froxels, and the halo pass
// adds its halo along each pixel's own ray: a 4 yd halo is two or three froxel slices deep at street
// range, and its lamp air a yard, so in the volume it smeared into a faint blob or vanished. Window
// lights stay in the froxels (`is_halo_light`) with no halo at all (`has_halo`): they take no lamp
// air, and a city's panes would otherwise take the slots of the lamps past them.
const HALO_LIGHTS: u32 = 32u;

// How many of the table's first lights are scanned for per-pixel halos: the tier's point-light count
// (`screen.w`, 0 on the analytic tier, which scatters no point light), at most `HALO_LIGHTS`.
fn halo_lights(f: FroxelFog, table_count: u32) -> u32 {
    return min(min(u32(f.screen.w), HALO_LIGHTS), table_count);
}

// fork: **an indoor light lights only the room's air.** The volume has no occlusion for a table
// light, so a lamp inside a building scattered into the air outside its walls: a glow through the
// masonry, pulsing with the froxels' slice jitter. An indoor-class light (the table's `w` integer
// part, `global_light`'s `rooms_are_indoor`) scatters in proportion to how much of the air is the
// camera's room (`room_weight`), so seen from outside it scatters nothing. A window's light keeps
// its glow: it stands in front of the glass, outside.
fn indoor_air_share(class_w: f32, room: f32) -> f32 {
    let indoor = floor(class_w) >= 1.0 && !is_window(class_w);
    return select(1.0, room, indoor);
}

// Whether table light `i`, of colour-row `w`, takes its halo per pixel rather than in the froxels.
fn is_halo_light(i: u32, halo_count: u32, class_w: f32) -> bool {
    return i < halo_count && lamp_air_share(class_w) > 0.5;
}

// The surface lanes' WoW falloff under the same range window, zero past the range. +0.5 keeps a
// point inside the flame finite.
fn point_falloff(dl: f32, range: f32) -> f32 {
    let q = dl / max(range, 1e-3);
    let window = clamp(1.0 - q * q * q * q, 0.0, 1.0);
    return window * window / (0.7 * dl + 0.03 * dl * dl + 0.5);
}

// The forward lobe relative to the isotropic term, so a light's total in-scatter is unchanged:
// brighter looking toward the lamp, dimmer looking away. `mu` = cos(light→point, point→eye).
fn point_phase(mu: f32) -> f32 {
    let g2 = POINT_G * POINT_G;
    let hg = (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * POINT_G * mu, 1e-4), 1.5);
    return mix(1.0, hg, POINT_FORWARD);
}

// The halo's e-folding distance, its far share and its peak gain, by the room weight at the light.
fn point_halo_yd(room: f32) -> f32 {
    return mix(POINT_HALO_YD, ROOM_POINT_HALO_YD, room);
}

fn point_far_share(room: f32) -> f32 {
    return POINT_FAR_SHARE * (1.0 - room);
}

fn point_halo_gain(room: f32) -> f32 {
    return mix(POINT_HALO_GAIN, ROOM_POINT_HALO_GAIN, room);
}

// Yards over which the room's medium fades out at its own bound, so a doorway is a soft edge in the
// air rather than a sphere the eye can find.
const ROOM_EDGE: f32 = 8.0;

// How much of the air at world point `p` is the room's own (0 = the open air's, 1 = all the room's).
fn room_weight(f: FroxelFog, p: vec3<f32>) -> f32 {
    if (f.room.w <= 0.0) {
        return 0.0;
    }
    let edge = max(f.room.w - ROOM_EDGE, 0.0);
    return f.room_media.y * (1.0 - smoothstep(edge, f.room.w, distance(p, f.room.xyz)));
}

// How far along `dir` the camera's own room reaches (yd, 0 = no room, or the camera is outside it):
// the far root of the ray against the room's sphere.
fn room_reach(f: FroxelFog, dir: vec3<f32>) -> f32 {
    if (f.room.w <= 0.0) {
        return 0.0;
    }
    let oc = f.camera.xyz - f.room.xyz;
    // fork: a room the camera has just left keeps its fading air (`volumetrics::fading_room`); the
    // path from outside it is the open air's, not the room's far root.
    if (dot(oc, oc) >= f.room.w * f.room.w) {
        return 0.0;
    }
    let b = dot(oc, dir);
    let disc = b * b - (dot(oc, oc) - f.room.w * f.room.w);
    if (disc <= 0.0) {
        return 0.0;
    }
    return max(sqrt(disc) - b, 0.0);
}

// fork: the night air's own glow, scattered on top of the haze's (`volumetrics::NIGHT_GLOW`).
fn night_glow(f: FroxelFog) -> vec3<f32> {
    return vec3<f32>(f.room_media.w, f.room_ambient.w, f.room_sun.w);
}

const MODE_ANALYTIC: f32 = 1.0;
const MODE_FROXEL: f32 = 2.0;

// Planar view depth (yd in front of the main camera) of a world point.
fn view_depth(f: FroxelFog, p: vec3<f32>) -> f32 {
    return -(dot(f.view_row.xyz, p) + f.view_row.w);
}

// Slices are quadratic in depth — dense near the eye, where shafts start.
fn slice_of_depth(f: FroxelFog, d: f32) -> f32 {
    return sqrt(clamp(d / f.grid.w, 0.0, 1.0));
}

fn depth_of_slice(f: FroxelFog, s: f32) -> f32 {
    return f.grid.w * s * s;
}

// Density scale at height `y`: exponential falloff above the reference height, at most x3 below it.
fn height_density(f: FroxelFog, y: f32) -> f32 {
    return exp(clamp(-(y - f.media.z) * f.media.y, -24.0, 1.1));
}

// The mean of `height_density` along a straight segment between heights `y_from` and `y_to`.
fn height_average(f: FroxelFog, y_from: f32, y_to: f32) -> f32 {
    let k = f.media.y;
    let dy = y_to - y_from;
    if (abs(dy * k) < 1e-3) {
        return height_density(f, 0.5 * (y_from + y_to));
    }
    let a = clamp(-(y_from - f.media.z) * k, -24.0, 1.1);
    let b = clamp(-(y_to - f.media.z) * k, -24.0, 1.1);
    return clamp((exp(a) - exp(b)) / (dy * k), 0.0, 3.0);
}

// The fog is a long-range height fog (`volumetrics::media`): `media.x` per yard of ray at the reference
// height, clear before the fog start. The start eases in over a fifth of itself so the froxels do not
// band on it.
fn fog_clear(f: FroxelFog, d: f32) -> f32 {
    return smoothstep(0.8 * f.media.w, f.media.w, d);
}

// The forward lobe's g (mirrors `media::FORWARD_G`).
const FORWARD_G: f32 = 0.75;

// Henyey-Greenstein, normalized so that its average over the sphere is 1.
fn hg(g: f32, mu: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * mu, 1e-4);
    return (1.0 - g2) / (denom * sqrt(denom));
}

// Amendment A2: a forward lobe (warm toward the sun) blended with a backward one (the zone's hue away
// from it). `mu` = cos of the angle between the view ray and the direction toward the light.
fn dual_phase(f: FroxelFog, mu: f32) -> f32 {
    return mix(hg(f.phase.x, mu), hg(FORWARD_G, mu), f.sun.w);
}

// Analytic height fog along the main camera's ray to `p`, from fraction `t_start` of the way there to `p`
// itself: rgb = in-scattered radiance, a = transmittance. The fog starts at the near-clear distance;
// the shaft haze does not.
fn analytic_path(f: FroxelFog, p: vec3<f32>, t_start: f32) -> vec4<f32> {
    return analytic_path_keyed(f, p, t_start, 1.0);
}

// Distance fog on top of the height fog: the long range fades out completely before the distant
// (WDL) ring ends, so far terrain never appears at an edge. Counted only through the lower
// atmosphere — a ray climbing out of it stops accruing — so the sky overhead stays clear.
const FAR_FOG_START: f32 = 2200.0;
const FAR_FOG_END: f32 = 6000.0;
const FAR_FOG_TAU: f32 = 4.5;
const FAR_FOG_CEILING: f32 = 900.0;

// Optical depth of the distance fog over the first `len` yards of a ray with vertical direction `dy`.
fn distance_fog(f: FroxelFog, len: f32, dy: f32) -> f32 {
    let reach = low_reach(f, len, dy);
    let t = clamp((reach - FAR_FOG_START) / (FAR_FOG_END - FAR_FOG_START), 0.0, 1.0);
    return FAR_FOG_TAU * t * t;
}

// The ray length (yd) through the lower atmosphere for a ray of vertical direction `dy`.
fn low_reach(f: FroxelFog, len: f32, dy: f32) -> f32 {
    if (dy > 1e-4) {
        let climb = max(f.media.z + FAR_FOG_CEILING - f.camera.y, 0.0);
        return min(len, climb / dy);
    }
    return len;
}

// Past the froxel grid nothing knows where the key light is shadowed, so the tail keeps only this
// share of its key in-scatter: a long view ray past a ridge or a canopy otherwise lit up the whole
// distance behind it as if the sun reached every yard of it.
const TAIL_KEY_SHARE: f32 = 0.35;

fn analytic_path_keyed(f: FroxelFog, p: vec3<f32>, t_start: f32, key_share: f32) -> vec4<f32> {
    let ray = p - f.camera.xyz;
    let len = length(ray);
    let t = max(t_start, 0.0);
    if (len < 1e-3 || t >= 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // Planar depth grows linearly along the ray, so the fog start is a fixed fraction of it.
    let d = max(view_depth(f, p), 1e-3);
    let t_fog = max(t, clamp(0.9 * f.media.w / d, 0.0, 1.0));
    let dir = ray / len;
    let phase = dual_phase(f, dot(dir, f.sun_dir.xyz));
    // The room's share of THIS ray: the air the camera is breathing, as far as the room's walls.
    // Beyond them the open air takes over, which is why the two are integrated one after the other
    // instead of mixed — the fog over a city seen from a doorway is the city's, not the doorway's.
    let room_len = clamp(room_reach(f, dir), 0.0, len) * (1.0 - t);
    let room_w = f.room_media.y * select(0.0, 1.0, f.room.w > 0.0);
    let room_tau = f.room_media.x * room_len * room_w;
    let room_tr = exp(-room_tau);
    let room_lum = f.room_ambient.rgb + f.room_sun.rgb * (phase * key_share);
    // The open air keeps the ray the room did not take.
    let open_share = max(1.0 - (room_len / len) * room_w, 0.0);
    let fog_tau = f.media.x * len * (1.0 - t_fog) * height_average(f, f.camera.y + ray.y * t_fog, p.y) * open_share;
    // fork: plus the night air (`room_media.z`), which glows on top (`night_glow`).
    let haze_path = len * (1.0 - t) * height_average(f, f.camera.y + ray.y * t, p.y);
    let night_tau = f.room_media.z * haze_path;
    let haze_tau = f.medium.x * haze_path + night_tau;
    let far_tau = distance_fog(f, len, ray.y / len) - distance_fog(f, len * t, ray.y / len);
    let total_tau = fog_tau + haze_tau + far_tau;
    let tr = exp(-total_tau);
    let lum = f.ambient.rgb + f.sun.rgb * (phase * key_share)
        + night_glow(f) * (night_tau / max(total_tau, 1e-6));
    // Front to back: the room, then everything past it.
    return vec4<f32>(room_lum * (1.0 - room_tr) + room_tr * lum * (1.0 - tr), room_tr * tr);
}

// Where a world point falls in the froxel grid: xy = screen uv, z = slice coordinate, w = 1 when inside
// the main camera's frustum.
fn froxel_uvw(f: FroxelFog, p: vec3<f32>) -> vec4<f32> {
    let clip = f.clip_from_world * vec4<f32>(p, 1.0);
    if (clip.w <= 1e-4) {
        return vec4<f32>(0.0);
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let inside = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    return vec4<f32>(uv, slice_of_depth(f, clip.w), select(0.0, 1.0, inside));
}

// The integrated volume at a screen uv and slice coordinate. Texel k holds the accumulation to the FAR
// edge of slice k, so a continuous slice position c reads texel coordinate c - 0.5; inside the first
// slice the accumulation ramps up from nothing.
fn integrated(vol: texture_3d<f32>, samp: sampler, f: FroxelFog, uv: vec2<f32>, s: f32) -> vec4<f32> {
    let n = f.grid.xyz;
    let c = s * n.z;
    let texel = clamp(vec3<f32>(uv * n.xy, c - 0.5), vec3<f32>(0.5), n - vec3<f32>(0.5));
    let v = textureSampleLevel(vol, samp, texel / f.alloc.xyz, 0.0);
    let ramp = clamp(c, 0.0, 1.0);
    return vec4<f32>(v.rgb * ramp, mix(1.0, v.a, ramp));
}

// The fog between the main camera and world point `p`: rgb = in-scatter, a = transmittance. A surface
// takes `rgb * a + in-scatter`. Froxels inside the grid, the analytic tail beyond its far plane, and the
// analytic path alone outside the main frustum (the mirror's view) or on the Low tier.
fn fog_at(vol: texture_3d<f32>, samp: sampler, f: FroxelFog, p: vec3<f32>) -> vec4<f32> {
    if (f.alloc.w < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let d = view_depth(f, p);
    if (f.alloc.w < MODE_FROXEL - 0.5 || d <= f.proj.x) {
        return analytic_path(f, p, 0.0);
    }
    let q = froxel_uvw(f, p);
    if (q.w < 0.5) {
        return analytic_path(f, p, 0.0);
    }
    let near = integrated(vol, samp, f, q.xy, q.z);
    if (d <= f.grid.w) {
        return near;
    }
    let tail = analytic_path_keyed(f, p, f.grid.w / d, TAIL_KEY_SHARE);
    return vec4<f32>(near.rgb + near.a * tail.rgb, near.a * tail.a);
}
