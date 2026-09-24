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
// Shadowed lights the cube atlas holds (6 faces each, 6x6 tiles per array layer, up to two layers
// at P8's Cinematic tier). Mirrors `benilla_world::modern::lights::MAX_SHADOWED`.
const MAX_SHADOWED: u32 = 12u;
// Shadowed lights one atlas layer holds.
const PER_ATLAS_LAYER: u32 = 6u;
// Filter taps the soft path can take. The tier asks for at most this many
// (`benilla_world::modern::lights::LightTier::taps`); the loop is bounded here so it unrolls.
const MAX_TAPS: u32 = 12u;
// fork: the soft path's blocker search — raw depth reads around the receiver that find how far the
// occluder stands from the light, which is what sizes the penumbra.
const BLOCKER_TAPS: u32 = 6u;
// fork: the widest penumbra (and blocker search) the soft path opens, in face texels. Past it the
// taps would reach across the face into the next tile's inset.
const MAX_PENUMBRA_TEXELS: f32 = 12.0;
// fork: the narrowest, so a contact shadow still takes the hardware 2x2 of a texel.
const MIN_PENUMBRA_TEXELS: f32 = 1.0;

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
    // fork: P8 — the soft filter. Named `soft` because `filter` is a WGSL RESERVED KEYWORD: naga
    // rejects it as a struct member, and the only thing that says so is the composer.
    // x = taps (1 = P6's single hardware compare), y = the emitter radius the penumbra opens from
    // (yd), z = the shade floor (how much of a light a shadow may take away; below 1 the shadow is
    // faint), w = the atlas's first array layer.
    soft: vec4<f32>,
    // fork: P8 — the FAR level and the atlas's tile use. x = far cells per horizontal axis (0 = the
    // tier has no far level, and the grid is the near level alone, exactly as before), y = far
    // cells vertically, z = the far cell's edge as a whole multiple of the near cell's, w = how
    // much of each atlas tile a cube face is drawn into (1 = all of it).
    far: vec4<f32>,
}

// fork: P8 — **the shadowed lights' positions are NOT a member of `LightGrid`**, for exactly the
// reason `light_cells` is not: three fragment lanes copy the parameters by value
// (`let g = modern_block.lights;`) and none of them reads a shadowed light's position — the cube
// lookup is addressed by slot, and the light's own position comes from the shared light table the
// lane is already walking. P6 kept six rows of them inside the struct; P8 needed twelve, which
// would have doubled a copy every lit fragment makes. They live in the block instead
// (`sunshadow::ModernBlock.light_shadow`), where only the binning compute pass reads them — so the
// struct every lane copies went from ten rows to five, and every modern tier got *cheaper* than it
// was before P8.
//
// Per shadowed light: xyz = world position, w = range (yd). Slot `s` lives on atlas layer
// `soft.w + s / PER_ATLAS_LAYER` and owns the six tiles from `6 * (s % PER_ATLAS_LAYER)`.

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

// Cells in the near level — the far level's cells follow them in the block.
fn near_cells(g: LightGrid) -> u32 {
    let n = vec3<u32>(g.dims.xyz);
    return n.x * n.y * n.z;
}

// fork: P8 — the far level's cells per axis (zero when the tier has none).
fn far_dims(g: LightGrid) -> vec3<i32> {
    return vec3<i32>(i32(g.far.x), i32(g.far.y), i32(g.far.x));
}

// The far level's cell edge in yards.
fn far_size(g: LightGrid) -> f32 {
    return g.dims.w * max(g.far.z, 1.0);
}

// The far level's base cell, derived from the near level's and not sent separately, so the two can
// never disagree about where the camera is: the near base is `floor(camera / cell) - dims / 2`,
// which puts the camera's own cell at `base + dims / 2`, and the far level is centred on the far
// cell that one falls in. The Rust twin is `modern::lights::far_base`.
fn far_base(g: LightGrid) -> vec3<i32> {
    let eye = vec3<i32>(g.base.xyz) + vec3<i32>(g.dims.xyz) / 2;
    let k = max(g.far.z, 1.0);
    return vec3<i32>(floor(vec3<f32>(eye) / k)) - far_dims(g) / 2;
}

// The near-level cell a world point falls in, or `.y == 0` when the point is outside it.
fn near_cell_at(g: LightGrid, p: vec3<f32>) -> vec2<u32> {
    let n = vec3<i32>(g.dims.xyz);
    let c = vec3<i32>(floor(p / g.dims.w)) - vec3<i32>(g.base.xyz);
    if (any(c < vec3<i32>(0)) || any(c >= n)) {
        return vec2<u32>(0u, 0u);
    }
    let index = (c.z * n.y + c.y) * n.x + c.x;
    return vec2<u32>(u32(index), 1u);
}

// The far-level cell a world point falls in (an index into the block, past the near cells), or
// `.y == 0` when the tier has no far level or the point is outside it too.
fn far_cell_at(g: LightGrid, p: vec3<f32>) -> vec2<u32> {
    let n = far_dims(g);
    if (n.x <= 0) {
        return vec2<u32>(0u, 0u);
    }
    let c = vec3<i32>(floor(p / far_size(g))) - far_base(g);
    if (any(c < vec3<i32>(0)) || any(c >= n)) {
        return vec2<u32>(0u, 0u);
    }
    let index = (c.z * n.y + c.y) * n.x + c.x;
    return vec2<u32>(near_cells(g) + u32(index), 1u);
}

// The cell a world point falls in — the near level's where it reaches, else the far level's — or
// `.y == 0` when the point is outside both (nothing there is near a light).
fn cell_at(g: LightGrid, p: vec3<f32>) -> vec2<u32> {
    if (!grid_live(g)) {
        return vec2<u32>(0u, 0u);
    }
    let near = near_cell_at(g, p);
    if (near.y > 0u) {
        return near;
    }
    return far_cell_at(g, p);
}

// fork: P8 — **the two cells a fragment reads, and how much of the far one.** `x` = the near cell
// (or the far one, when the point is past the near level), `y` = the far cell, `z` = the far cell's
// weight as `f32` bits, `w` = 1 when there is anything to read at all.
//
// The far level exists so a lamp lights the street it stands in at 200 yd exactly as it does at
// 20 — the near level alone ran out at ±72 yd, and everything past it took no local light at all,
// which read as lamps switching on as the player walked toward them. The two levels hold the same
// lights under the same falloff, so they agree everywhere a far cell kept every light that reached
// it; where one overflowed they can differ, and the last `BLEND_CELLS` near cells cross-fade into
// the far level instead of switching, so the seam is never an edge either.
const BLEND_CELLS: f32 = 4.0;

fn cell_pair(g: LightGrid, p: vec3<f32>) -> vec4<u32> {
    if (!grid_live(g)) {
        return vec4<u32>(0u);
    }
    let near = near_cell_at(g, p);
    let far = far_cell_at(g, p);
    if (near.y == 0u) {
        return vec4<u32>(far.x, far.x, bitcast<u32>(0.0), far.y);
    }
    if (far.y == 0u) {
        return vec4<u32>(near.x, near.x, bitcast<u32>(0.0), 1u);
    }
    // How deep inside the near level the point is, in near cells, on its tightest axis.
    let lo = vec3<f32>(g.base.xyz) * g.dims.w;
    let hi = lo + g.dims.xyz * g.dims.w;
    let inset = min(p - lo, hi - p) / g.dims.w;
    let depth = min(inset.x, min(inset.y, inset.z));
    let w = 1.0 - smoothstep(0.0, BLEND_CELLS, depth);
    return vec4<u32>(near.x, far.x, bitcast<u32>(w), 1u);
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
// when the grid takes over its selection. fork: `+ CORE` softens the pool under the light: bare, a
// lantern held a yard off the floor painted a hot spot there at ~1.2x its colour.
const CORE: f32 = 0.8;
fn falloff(d: f32, range: f32) -> f32 {
    let atten = 1.0 / (0.7 * d + 0.03 * d * d + CORE);
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

// Where one cube face of one slot lands in the atlas: `xy` = the tile's origin in uv, `z` = the
// array layer, `w` = the tile's edge in uv. Two layers of 6x6 tiles hold twelve lights, and which
// layer a slot is on is the only thing P8 added to P6's addressing.
fn atlas_tile(g: LightGrid, slot: u32, face: u32) -> vec4<f32> {
    let tiles = g.misc.w;
    let layer = g.soft.w + f32(slot / PER_ATLAS_LAYER);
    let tile = (slot % PER_ATLAS_LAYER) * 6u + face;
    let per = u32(tiles);
    let origin = vec2<f32>(f32(tile % per), f32(tile / per)) / tiles;
    return vec4<f32>(origin, layer, tile_use(g) / tiles);
}

// fork: P8 — how much of its tile a cube face was drawn into (`LightTier::tile_use`): a face drawn
// at half the tile's edge is a quarter of the depth to render, and its texels are twice as wide.
// Zero (a block written before this row existed) reads as the whole tile.
fn tile_use(g: LightGrid) -> f32 {
    return select(g.far.w, 1.0, g.far.w <= 0.0);
}

// A Vogel-disc offset — the same spiral the sun's filter uses, so a lamp's penumbra and the sun's
// are made of the same noise and do not read as two different renderers. `rotation` decorrelates
// neighbouring fragments.
fn vogel_offset(index: u32, count: u32, rotation: f32) -> vec2<f32> {
    let golden = 2.399963229728653;
    let r = sqrt((f32(index) + 0.5) / f32(count));
    let theta = f32(index) * golden + rotation;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// How much of a shadowed light a point sees (1 = fully lit). `slot >= MAX_SHADOWED` — a light with
// no tiles of its own — is fully lit by definition, which is what the tier table means by "the rest
// of the lights are unshadowed".
//
// fork: the lookup is a blocker search and then a Vogel disc of `soft.x` taps whose radius is the
// penumbra a flame of `soft.y` yards would cast — contact-hardening, crisp where the occluder
// touches the receiver — lifted by the tier's shade floor. At `taps == 1` it is P6's single hardware
// compare under the floor.
//
// `light_pos` is the slot's ANCHOR (`light_shadow[slot]`), where its cube was rendered — the middle
// of a swaying light's sway, not where the light is this frame.
fn local_visibility(
    map: texture_depth_2d_array,
    samp: sampler_comparison,
    g: LightGrid,
    slot: u32,
    light_pos: vec3<f32>,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    // The cube's near plane FOR THIS SLOT (`ShadowLight::near`), which the caller reads from the
    // block's spare rows. It has to be the same number the face matrix was built with, or the
    // depth this compares is not the depth that was stored.
    near: f32,
    // fork: P8 — this slot's CROSS-FADE, 0..1. A slot is never switched on or off: it ramps in when
    // it is filled and out before it is released, so the shadowed set can change without anything
    // popping. At 0 this returns exactly what a light with no cube returns, which is what makes a
    // slot's arrival and departure invisible.
    fade: f32,
) -> f32 {
    if (slot >= MAX_SHADOWED || !atlas_live(g) || fade <= 0.0) {
        return 1.0;
    }
    // Offset along the normal by a texel of the face, so a lit surface never shadows itself at a
    // grazing angle. One texel is `2 * major / tile` yards across at distance `major`.
    let raw = cube_face(world_pos - light_pos);
    // Face texels are `shadow_misc.w * tiles / tile_use` of face uv each: a face drawn into part
    // of its tile has fewer, wider texels, and every texel-sized term below scales with it.
    let tiles = g.misc.w / tile_use(g);
    let texel_yd = 2.0 * raw.w * (g.shadow_misc.w * tiles);
    let v = (world_pos + normal * texel_yd) - light_pos;
    let ct = cube_face(v);
    // Reverse-Z infinite perspective: depth is `near / major`, so the bias is a step back toward
    // the light in yards.
    let depth = near / max(ct.w - g.shadow_misc.y, near);
    // Inset the face uv by the tile border, so a filter tap can never reach the neighbouring face.
    let inset = g.shadow_misc.z * g.shadow_misc.w * tiles;
    let tile = atlas_tile(g, slot, u32(ct.z));

    // **The single-tap fast path is P6's.** A tier that asks for one tap takes one hardware compare
    // and no spiral, no rotation hash and no blend — the soft path below must not cost it a single
    // instruction, and a `select` would not do: WGSL evaluates both of its arms, so the Vogel spiral
    // would be computed on every tap of every tier.
    //
    // fork: P8b — it takes the shade floor too, which is what lets Cinematic run on this path: its
    // shadow stays faint (a lamp's occluded side keeps `1 - soft.z` of it) without the eight taps.
    // At `soft.z == 1` (High) the floor is `mix(0, 1, one)`, which is `one` exactly — P6's value.
    let taps = max(u32(g.soft.x), 1u);
    if (taps <= 1u) {
        let face_uv = clamp(ct.xy, vec2<f32>(inset), vec2<f32>(1.0 - inset));
        let one = textureSampleCompareLevel(map, samp, tile.xy + face_uv * tile.w, i32(tile.z), depth);
        return mix(1.0, mix(1.0 - g.soft.z, 1.0, one), fade);
    }

    // fork: **the penumbra of a finite emitter**, contact-hardening. A point light stands in for a
    // flame about `soft.y` yards across, so the blur at the receiver is that radius scaled by how far
    // the receiver stands behind its occluder: `radius × (receiver − blocker) / blocker`. A chair leg
    // where it meets the floor is crisp and its shadow softens along its length, which is what makes
    // a lamp's shadow read as a lamp's and not as a blurred stencil.
    //
    // The blocker's distance comes from a few raw depth reads around the receiver (the stored depth
    // is `near / major`, so a read turns straight back into yards along the face's axis). No blocker
    // among them is a lit receiver, and the filter is skipped.
    let texel_uv = g.shadow_misc.w * tiles;
    let dims = vec2<f32>(textureDimensions(map));
    let search = MAX_PENUMBRA_TEXELS * texel_uv;
    // Rotate the disc per fragment, from the receiver's own position: a fixed spiral banded every
    // lit floor with visible arms.
    //
    // **Wrapped before it is hashed, and that is not cosmetic.** A WoW world coordinate runs to
    // ±17066, and hashing it raw put `dot` up around 1.3e6 — where an f32 ulp is about an eighth,
    // so `fract` collapsed to a handful of distinct values and the whole disc took the same few
    // rotations across large patches of floor. Taking the fraction first keeps the hash's input
    // inside [0,1) with precision to spare, and wrapping every ten yards is invisible in a rotation.
    let wrapped = fract(world_pos * 0.1);
    let rotation = fract(dot(wrapped, vec3<f32>(12.9898, 78.233, 37.719))) * 6.2831853;
    var blocker = 0.0;
    var blockers = 0.0;
    for (var i = 0u; i < BLOCKER_TAPS; i = i + 1u) {
        let offset = vogel_offset(i, BLOCKER_TAPS, rotation + 1.3) * search;
        let face_uv = clamp(ct.xy + offset, vec2<f32>(inset), vec2<f32>(1.0 - inset));
        let texel = vec2<i32>((tile.xy + face_uv * tile.w) * dims);
        let stored = textureLoad(map, texel, i32(tile.z), 0);
        // Reverse-Z: a stored depth above the receiver's is nearer the light.
        if (stored > depth) {
            blocker = blocker + near / max(stored, 1e-6);
            blockers = blockers + 1.0;
        }
    }
    if (blockers <= 0.0) {
        return 1.0;
    }
    let occluder = max(blocker / blockers, near);
    let penumbra_yd = g.soft.y * max(ct.w - occluder, 0.0) / occluder;
    // A yard at distance `major` is `0.5 / major` of face uv.
    let spread = clamp(
        penumbra_yd * 0.5 / max(ct.w, 1e-3),
        MIN_PENUMBRA_TEXELS * texel_uv,
        MAX_PENUMBRA_TEXELS * texel_uv,
    );

    var lit = 0.0;
    var taken = 0u;
    for (var i = 0u; i < MAX_TAPS; i = i + 1u) {
        if (i >= taps) {
            break;
        }
        let offset = vogel_offset(i, taps, rotation) * spread;
        let face_uv = clamp(ct.xy + offset, vec2<f32>(inset), vec2<f32>(1.0 - inset));
        let uv = tile.xy + face_uv * tile.w;
        lit = lit + textureSampleCompareLevel(map, samp, uv, i32(tile.z), depth);
        taken = taken + 1u;
    }
    let visibility = lit / f32(max(taken, 1u));
    // Faint: the shadow may only take `shade` of the light away. `shade == 1` is P6's behaviour.
    let shaded = mix(1.0 - g.soft.z, 1.0, visibility);
    return mix(1.0, shaded, fade);
}
