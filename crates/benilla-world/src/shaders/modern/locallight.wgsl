// Modern lighting (FORK.md, P6): the local-light grid and the shadowed-light
// cube atlas — how a fragment finds the lights that reach it, and how much of each one it sees.
//
// Imported as `benilla_modern::locallight` — a module name no hooked shader uses as an identifier
// (naga_oil substitutes a module alias for every bare identifier of that name), and with no
// identifier in it ending in a digit (a composable module may not carry those).
//
// No bindings are declared here. The grid rides the shared modern storage block
// (`sunshadow::ModernBlock`) and the cube atlas rides the last layer of the modern shadow array,
// so every lane that already binds those reads both; the light table itself stays where it is (the
// shared light buffer's `points`), which is why each hooked shader keeps its own small walk over a
// cell — a WGSL module cannot reach its caller's bindings.
#define_import_path benilla_modern::locallight

// Lights one cell can hold. Mirrors `benilla_world::modern::lights::PER_CELL`.
const PER_CELL: u32 = 8u;
// Shadowed lights the cube atlas holds (6 faces each, 6x6 tiles in one array layer).
const MAX_SHADOWED: u32 = 6u;

// The grid's parameters — the part a lane copies by value, so it is kept small for the same reason
// `sunshadow::ModernCascades` is (160 B, ten rows). The cells themselves are a separate member of
// the block (`light_cells`), never a member of this struct: an 864 KB array inside a by-value
// parameter is a copy per call.
//
// Mirrors the head of `benilla_world::modern::lights::LightParams::std430`, whose padding to 512 B
// is the block's `light_spare` and is what keeps the probe array behind it 256-aligned.
struct LightGrid {
    // xyz = cells per axis, w = cell size (yd).
    dims: vec4<f32>,
    // xyz = the base cell coord (world / cell size, integral), w = flags: 1 = the grid is live,
    // 2 = the shadowed-light atlas is live.
    base: vec4<f32>,
    // x = live cells (the tier's, which is what the binning pass dispatches over), y = shadowed
    // lights, z = the atlas's array layer, w = atlas tiles per side.
    misc: vec4<f32>,
    // x = the cube faces' near plane (yd), y = the depth bias (yd), z = the tile inset (texels),
    // w = 1 / the shadow map's edge in texels.
    shadow_misc: vec4<f32>,
    // Per shadowed light: xyz = world position, w = range (yd). Slot `s` owns atlas tiles
    // `6*s .. 6*s+5`.
    shadow: array<vec4<f32>, 6>,
}

// One cell of the grid, as the block stores it (`light_cells`; the allocation is one size on every
// platform and every tier, so a quality change never re-creates the buffer every material binds —
// `modern::lights::CELL_ALLOC`): x = the light count in bits 0-7 and the candidates seen in bits
// 8-15, y and z = the table indices one byte each, w = each entry's shadow slot as a nibble
// (0 = none).

fn grid_live(g: LightGrid) -> bool {
    return (u32(g.base.w) & 1u) != 0u;
}

fn atlas_live(g: LightGrid) -> bool {
    return (u32(g.base.w) & 2u) != 0u;
}

// The cell a world point falls in, or `count.y == 0` when the point is outside the grid (which is
// every point more than the grid's half extent from the camera — nothing there is near a light).
fn cell_at(g: LightGrid, p: vec3<f32>) -> vec2<u32> {
    if (!grid_live(g)) {
        return vec2<u32>(0u, 0u);
    }
    let n = vec3<i32>(g.dims.xyz);
    let c = vec3<i32>(floor(p / g.dims.w)) - vec3<i32>(g.base.xyz);
    if (any(c < vec3<i32>(0)) || any(c >= n)) {
        return vec2<u32>(0u, 0u);
    }
    let index = (c.z * n.y + c.y) * n.x + c.x;
    return vec2<u32>(u32(index), 1u);
}

// How many lights a cell holds.
fn cell_count(cell: vec4<u32>) -> u32 {
    return min(cell.x & 0xffu, PER_CELL);
}

// The `i`-th light's index in the shared table.
fn cell_light(cell: vec4<u32>, i: u32) -> u32 {
    let word = select(cell.z, cell.y, i < 4u);
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

// The `i`-th light's shadow slot, or `MAX_SHADOWED` when it casts none.
fn cell_slot(cell: vec4<u32>, i: u32) -> u32 {
    let nibble = (cell.w >> (i * 4u)) & 0xfu;
    return select(nibble - 1u, MAX_SHADOWED, nibble == 0u);
}

// The WoW point falloff `1/(0.7d + 0.03d^2)` under a smooth window that reaches zero at the light's
// range — the same curve P3 gave the per-pixel `point_light_eval`, so a light's look does not change
// when the grid takes over its selection.
fn falloff(d: f32, range: f32) -> f32 {
    let atten = 1.0 / (0.7 * d + 0.03 * d * d);
    let q = d / max(range, 1e-3);
    let window = clamp(1.0 - q * q * q * q, 0.0, 1.0);
    return atten * window * window;
}

// The cube face a direction leaves through, and where it lands on that face. `xy` = the face uv in
// `[0,1]` (y measured downward, like a render target), `z` = the face index, `w` = the distance
// along the face's major axis. Pinned against the CPU's own face matrices by a unit test in
// `benilla_world::modern::lights`.
fn cube_face(v: vec3<f32>) -> vec4<f32> {
    let a = abs(v);
    var face = 0.0;
    var major = a.x;
    var st = vec2<f32>(0.0);
    if (a.x >= a.y && a.x >= a.z) {
        major = a.x;
        if (v.x > 0.0) {
            face = 0.0;
            st = vec2<f32>(-v.z, -v.y);
        } else {
            face = 1.0;
            st = vec2<f32>(v.z, -v.y);
        }
    } else if (a.y >= a.z) {
        major = a.y;
        if (v.y > 0.0) {
            face = 2.0;
            st = vec2<f32>(v.x, v.z);
        } else {
            face = 3.0;
            st = vec2<f32>(v.x, -v.z);
        }
    } else {
        major = a.z;
        if (v.z > 0.0) {
            face = 4.0;
            st = vec2<f32>(v.x, -v.y);
        } else {
            face = 5.0;
            st = vec2<f32>(-v.x, -v.y);
        }
    }
    let uv = st / max(major, 1e-6) * 0.5 + vec2<f32>(0.5);
    return vec4<f32>(uv, face, major);
}

// How much of a shadowed light a point sees (1 = fully lit). `slot >= MAX_SHADOWED` — a light with
// no tiles of its own — is fully lit by definition, which is what the tier table means by "the rest
// of the lights are unshadowed".
fn local_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    g: LightGrid,
    slot: u32,
    light_pos: vec3<f32>,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
) -> f32 {
    if (slot >= MAX_SHADOWED || !atlas_live(g)) {
        return 1.0;
    }
    let near = g.shadow_misc.x;
    // Offset along the normal by a texel of the face, so a lit surface never shadows itself at a
    // grazing angle. One texel is `2 * major / tile` yards across at distance `major`.
    let raw = cube_face(world_pos - light_pos);
    let tiles = g.misc.w;
    let texel_yd = 2.0 * raw.w * (g.shadow_misc.w * tiles);
    let v = (world_pos + normal * texel_yd) - light_pos;
    let ct = cube_face(v);
    // Reverse-Z infinite perspective: depth is `near / major`, so the bias is a step back toward
    // the light in yards.
    let depth = near / max(ct.w - g.shadow_misc.y, near);
    // Inset the face uv by the tile border, so a filter tap can never reach the neighbouring face.
    let inset = g.shadow_misc.z * g.shadow_misc.w * tiles;
    let face_uv = clamp(ct.xy, vec2<f32>(inset), vec2<f32>(1.0 - inset));
    let tile = u32(ct.z) + slot * 6u;
    let per = u32(tiles);
    let origin = vec2<f32>(f32(tile % per), f32(tile / per));
    let uv = (origin + face_uv) / tiles;
    return textureSampleCompareLevel(map, samp, uv, i32(g.misc.z), depth);
}
