// Modern lighting (FORK.md, P5): the runtime probe GI — the probe field's layout
// and sampling, and the ray march through the voxel clipmap and the terrain heightfield.
//
// Imported as `benilla_modern::probegi` — a module name no hooked shader uses as an identifier
// (naga_oil substitutes a module alias for every bare identifier of that name), and with no
// identifier in it ending in a digit (a composable module may not carry those).
//
// No bindings are declared here. The probe field rides the shared modern storage block
// (`sunshadow::ModernBlock`), so every lane that already binds the block can read it; the march's
// textures are function parameters, like `sunshadow`'s shadow map.
#define_import_path benilla_modern::probegi

// Mirrors `benilla_world::modern::gi::GiParams::std430` (a layout test pins the size at 208 B, which
// is what puts the probe array at a 256-aligned offset inside the block).
struct GiParams {
    // x = mode (0 off, 1 probes), y = probes per cascade, z = GI gain (0 = the sky map alone),
    // w = normal bias in probe spacings.
    params: vec4<f32>,
    // Per cascade: xyz = the base probe coord (world coord / spacing, integral), w = spacing (yd).
    // Cascade 0 is the P6 interior sub-grid when `interior.w` says so — a finer grid that exists
    // only while the camera stands in a WMO, with the exterior ladder shifted up behind it.
    grid: array<vec4<f32>, 5>,
    // xyz = probes per axis, w = cascade count.
    dims: vec4<f32>,
    // rgb = the sky's reference radiance (the shared light's ambient row), w = the open-field mean
    // radiance's luminance — what a probe under nothing reads, the froxels' reference.
    sky: vec4<f32>,
    // rgb = ground albedo (the heightfield's surface), w = the sky gradient's horizon-to-zenith span.
    ground: vec4<f32>,
    // x = rays per probe, y = update period (frames), z = frame index, w = debug view.
    misc: vec4<f32>,
    // P6: rgb = the interior ambient (the camera's WMO `MOHD.ambColor`, decoded) — what an interior
    // probe's escaped rays see instead of the sky. w = flags: 1 = cascade 0 is the interior
    // sub-grid, 2 = Chebyshev probe visibility (High and above).
    interior: vec4<f32>,
    // Room for P7's grading — also the padding that aligns the light grid behind this block. 9
    // rows since the froxel block grew by nine (the room fog, the night banks, the ground rows,
    // the lamp halos' air): the offset has to land on 256 again.
    spare: array<vec4<f32>, 9>,
}

// Cascade `c`'s grid row, picked by constant index: `g` is a by-value copy, and a computed index
// into an array inside a value is a copy of the array per call (`sunshadow::cascade_clip`, whose
// A/B lever `MODERN_ARRAY_INDEX` this shares).
fn grid_row(g: GiParams, c: u32) -> vec4<f32> {
#ifdef MODERN_ARRAY_INDEX
    return g.grid[c];
#else
    switch c {
        case 0u: {
            return g.grid[0];
        }
        case 1u: {
            return g.grid[1];
        }
        case 2u: {
            return g.grid[2];
        }
        case 3u: {
            return g.grid[3];
        }
        default: {
            return g.grid[4];
        }
    }
#endif
}

fn interior_grid(g: GiParams) -> bool {
    return (u32(g.interior.w) & 1u) != 0u;
}

fn chebyshev_on(g: GiParams) -> bool {
    return (u32(g.interior.w) & 2u) != 0u;
}

// One probe: irradiance over pi as an order-1 SH lobe, plus the bookkeeping the toroidal grid needs.
// `E(n) / pi = dc.rgb + sh_x.rgb * n.x + sh_y.rgb * n.y + sh_z.rgb * n.z`, which is exactly the units
// the shared light's ambient row is in, so a probe substitutes for it directly.
struct GiProbe {
    // rgb = the lobe's DC term, w = validity (0 = never traced, or traced from inside geometry).
    dc: vec4<f32>,
    // rgb = the lobe's x term, w = sky visibility (the share of rays that reached the sky).
    sh_x: vec4<f32>,
    // rgb = the lobe's y term, w = the world-grid key's low half (x and z).
    sh_y: vec4<f32>,
    // rgb = the lobe's z term, w = the world-grid key's high half (y).
    sh_z: vec4<f32>,
    // P6: x = the mean ray hit distance (yd), y = its mean square — the two moments a Chebyshev
    // visibility test needs; z = the probe's interior class (1 = nothing overhead reaches the sky),
    // which is what keeps a room's probes out of the street and the street's out of the room;
    // w = 1 once the slot has been traced (validity cannot say so: an invalid probe is traced too).
    depth: vec4<f32>,
}

// The toroidal grid's identity for a probe coord: which world probe this texel currently holds. A
// scrolled-in probe reads a key that is not its own and is re-traced whole instead of blended.
fn key_lo(w: vec3<i32>) -> f32 {
    return f32((w.x & 4095) + (w.z & 4095) * 4096);
}

fn key_hi(w: vec3<i32>) -> f32 {
    return f32(w.y & 4095);
}

// Positive modulo — a probe coord's slot in its cascade's toroidal grid.
fn wrap(v: i32, n: i32) -> i32 {
    return ((v % n) + n) % n;
}

// Where cascade `c`'s probes live in the block. The interior sub-grid is cascade 0 for every reader
// (finest first) but is STORED behind the exterior ladder, so stepping into or out of a WMO leaves
// every exterior probe in the slot it already had, with its history. Before this, the whole ladder
// shifted one slot and the field re-seeded from single noisy traces: rooms and canopies blinked.
fn storage_cascade(g: GiParams, c: u32) -> u32 {
    if (!interior_grid(g)) {
        return c;
    }
    return select(c - 1u, u32(g.dims.w) - 1u, c == 0u);
}

// The inverse: which cascade a storage block holds.
// fork: **the most probes one frame may trace** — the steady state's own share, `total / period`.
// A frame in which every probe is due — the first after entering the world (every probe cold, the
// clipmap already full of the city that streamed in behind the loading cover) — used to list the
// whole field for ONE dispatch: 65536 probes x 64 rays of up to 160 DDA steps on Cinematic, seconds
// of GPU on a laptop, and Windows resets the driver past two (`DXGI_ERROR_DEVICE_HUNG`, the browser
// build's world-entry crash). Capped, a cold field fills over `period` frames and no frame's trace
// costs more than every steady frame's already does. `gi_classify` lists no more than this, and the
// trace and the blend read no further.
fn listed_cap(g: GiParams) -> u32 {
    let total = u32(g.params.y) * u32(g.dims.w);
    return max(total / max(u32(g.misc.y), 1u), 1u);
}

fn logical_cascade(g: GiParams, s: u32) -> u32 {
    if (!interior_grid(g)) {
        return s;
    }
    return select(s + 1u, 0u, s == u32(g.dims.w) - 1u);
}

fn probe_index(g: GiParams, c: u32, w: vec3<i32>) -> u32 {
    let n = vec3<i32>(g.dims.xyz);
    let t = vec3<i32>(wrap(w.x, n.x), wrap(w.y, n.y), wrap(w.z, n.z));
    let inside = (t.z * n.y + t.y) * n.x + t.x;
    return u32(inside) + storage_cascade(g, c) * u32(g.params.y);
}

// The world probe a storage slot currently holds: slot `slot` of cascade `c`'s toroidal grid, placed
// against that cascade's base. The classify, trace and blend passes all ask this of the same index,
// so it is written once.
fn slot_probe(g: GiParams, c: u32, slot: u32) -> vec3<i32> {
    let s = i32(slot);
    let n = vec3<i32>(g.dims.xyz);
    let texel = vec3<i32>(s % n.x, (s / n.x) % n.y, s / (n.x * n.y));
    let base = vec3<i32>(grid_row(g, c).xyz);
    return base + vec3<i32>(
        wrap(texel.x - base.x, n.x),
        wrap(texel.y - base.y, n.y),
        wrap(texel.z - base.z, n.z),
    );
}

fn probe_pos(g: GiParams, c: u32, w: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(w) * grid_row(g, c).w;
}

// Where a world point falls in one cascade: the low corner of its probe cell, the trilinear
// fractions, and how strongly this cascade owns the point (1 inside, fading to 0 at its edge, so a
// coarser cascade takes over without a seam).
struct GiPick {
    base: vec3<i32>,
    frac: vec3<f32>,
    cascade: u32,
    inside: f32,
    spacing: f32,
    // P6: the RECEIVER's interior class (1 = an interior WMO surface or an interior prop).
    recv: f32,
}

fn pick(g: GiParams, c: u32, p: vec3<f32>, nrm: vec3<f32>, recv_interior: f32) -> GiPick {
    let spacing = grid_row(g, c).w;
    let n = g.dims.xyz;
    // Sampled a little along the normal, so a surface never interpolates the probe behind it.
    let biased = p + nrm * (spacing * g.params.w);
    let f = biased / spacing - grid_row(g, c).xyz;
    let lo = floor(f);
    // One cell of margin on each side keeps the trilinear footprint inside the cascade, and the
    // outermost cell fades so the next cascade can take the remainder.
    let edge = min(min(f.x, f.y), min(f.z, min(min(n.x - 1.0 - f.x, n.y - 1.0 - f.y), n.z - 1.0 - f.z)));
    var out: GiPick;
    out.cascade = c;
    out.spacing = spacing;
    out.recv = clamp(recv_interior, 0.0, 1.0);
    out.base = vec3<i32>(clamp(lo, vec3<f32>(0.0), n - vec3<f32>(2.0))) + vec3<i32>(grid_row(g, c).xyz);
    out.frac = clamp(f - lo, vec3<f32>(0.0), vec3<f32>(1.0));
    out.inside = clamp((edge - 0.5) / CASCADE_FADE, 0.0, 1.0);
    return out;
}

// How many probe cells a cascade's ownership fades over at its edge. The cascades are camera
// centred, so the edge sweeps across the world as the player walks, and every point it crosses
// swaps a coarse answer for a fine one. The two genuinely differ in a shaded street (a 6 yd lattice
// cannot see an 8 yd alley's walls that a 2 yd one can), so a one-cell fade read as a darker band
// creeping ahead of the player. Spread over three cells the same swap is a slow drift instead.
const CASCADE_FADE: f32 = 3.0;

fn tap_coord(pk: GiPick, k: u32) -> vec3<i32> {
    return pk.base + vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
}

fn tap_index(g: GiParams, pk: GiPick, k: u32) -> u32 {
    return probe_index(g, pk.cascade, tap_coord(pk, k));
}

// The running sum over a cell's eight probes.
struct GiAccum {
    sum: vec3<f32>,
    weight: f32,
    sky: f32,
    /// The same weight **before** the class match — the denominator that says how much of this cell
    /// the class test threw away. See [`finish`].
    base: f32,
}

fn start() -> GiAccum {
    return GiAccum(vec3<f32>(0.0), 0.0, 0.0, 0.0);
}

fn accum(a: GiAccum, g: GiParams, pk: GiPick, k: u32, probe: GiProbe, p: vec3<f32>, nrm: vec3<f32>) -> GiAccum {
    let coord = tap_coord(pk, k);
    // A probe whose slot holds another world probe (the grid scrolled) or that sits inside geometry
    // contributes nothing.
    var w = probe.dc.w;
    if (probe.sh_y.w != key_lo(coord) || probe.sh_z.w != key_hi(coord)) {
        w = 0.0;
    }
    if (w <= 0.0) {
        return a;
    }
    let trilinear = mix(1.0 - pk.frac, pk.frac, vec3<f32>(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u)));
    w = w * max(trilinear.x * trilinear.y * trilinear.z, 1e-4);
    // Backface weight: a probe on the far side of the surface is nearly ignored.
    let to_probe = probe_pos(g, pk.cascade, coord) - p;
    let facing = dot(normalize(to_probe + nrm * 1e-3), nrm) * 0.5 + 0.5;
    w = w * (facing * facing + 0.2);
    // P6 class match — deliberately ASYMMETRIC. An interior receiver rejects a probe that sees the
    // sky: that is the doorway leak the plan names, and the receiver's own class is authored (the
    // lane's interior flag) rather than inferred, so the rejection is safe. An exterior receiver
    // only mildly prefers exterior probes, because the probe's side of the test IS inferred and a
    // probe deep under a canopy is legitimately dark and legitimately outdoors — a symmetric test
    // would throw P5's canopy bounce away along with the room's.
    let base_w = w;
    let leak = pk.recv * (1.0 - probe.depth.z);
    let spill = (1.0 - pk.recv) * probe.depth.z;
    w = w * clamp(1.0 - leak * 0.92 - spill * 0.35, 0.05, 1.0);
    // P6 (High and above) Chebyshev visibility: a probe whose rays typically stop nearer than the
    // receiver is behind something, and its weight decays with how much nearer.
    if (chebyshev_on(g)) {
        let reach = length(to_probe);
        let mean = probe.depth.x;
        if (mean > 0.0 && reach > mean) {
            let variance = max(probe.depth.y - mean * mean, 1e-3);
            let over = reach - mean;
            w = w * clamp(variance / (variance + over * over), 0.0, 1.0);
        }
    }
    let e = max(probe.dc.rgb + probe.sh_x.rgb * nrm.x + probe.sh_y.rgb * nrm.y + probe.sh_z.rgb * nrm.z, vec3<f32>(0.0));
    return GiAccum(a.sum + e * w, a.weight + w, a.sky + probe.sh_x.w * w, a.base + base_w);
}

// rgb = the irradiance over pi the ambient row is replaced by; w = how much of the cell answered.
// `recv` is the receiver's interior class, as [`pick`] took it.
fn finish(a: GiAccum, recv: f32) -> vec4<f32> {
    if (a.weight <= 1e-5 || a.base <= 1e-5) {
        return vec4<f32>(0.0);
    }
    // **The class match has to reach the CONFIDENCE, not just the weights.** It scales each probe's
    // weight, but the colour is normalised by the total weight — so a cell in which the class test
    // rejected *every* probe still returns their irradiance at full strength, merely with all the
    // weights small together. Interior receivers are where that shows: the cascades are camera
    // centred, so from across the room only the coarse ones cover a point, and a 54 yd probe
    // spacing has no probe inside the building at all. Northshire Abbey lit up with the daylight
    // outside it, and got correctly dark as the player walked close enough for the fine cascade to
    // answer.
    //
    // `weight / base` is the share of this cell that survived the class test. Folding it into the
    // confidence lets the lane fall back to the room's authored ambient instead of the sky.
    // Applied for an INTERIOR receiver only (`recv`), because the exterior half of the class test
    // is a mild preference over an inferred class (see `accum`) and must not start dimming the
    // canopy bounce it was deliberately kept gentle for.
    // Curved, not linear, and that is the difference between "much better" and done. The class
    // factor bottoms out at the 0.05..0.08 clamp in `accum` rather than at zero, so a cell whose
    // every probe was rejected still survives at ~0.08 — about a hundredth of full daylight once
    // the confidence curve has it, which is exactly the faint bleed left after the first fix.
    //
    // A cell the class test almost entirely rejected holds no information an interior receiver can
    // use, and reporting a hundredth of the sky is worse than reporting nothing: the fallback is
    // the room's OWN authored ambient, which is right. So the share is pushed through a smoothstep
    // that reaches zero well above the clamp's floor. A doorway cell, genuinely half interior and
    // half exterior probes, still lands mid-curve and still spills its light inward — which is a
    // real doorway doing a real thing, and must survive.
    let survived = smoothstep(0.20, 0.60, clamp(a.weight / a.base, 0.0, 1.0));
    let confidence = clamp(a.weight * 3.0, 0.0, 1.0) * mix(1.0, survived, clamp(recv, 0.0, 1.0));
    return vec4<f32>(a.sum / a.weight, confidence);
}

// The nearest probe's slot — the froxels' one-tap lookup.
fn nearest_index(g: GiParams, p: vec3<f32>) -> vec2<f32> {
    let n = g.dims.xyz;
    let count = u32(g.dims.w);
    for (var c = 0u; c < count; c = c + 1u) {
        let f = p / grid_row(g, c).w - grid_row(g, c).xyz;
        if (all(f >= vec3<f32>(0.0)) && all(f <= n - vec3<f32>(1.0))) {
            let w = vec3<i32>(round(f)) + vec3<i32>(grid_row(g, c).xyz);
            return vec2<f32>(f32(probe_index(g, c, w)), 1.0);
        }
    }
    return vec2<f32>(0.0, 0.0);
}

// ---------------------------------------------------------------------------------------------
// The scene the probes trace against: a toroidal voxel clipmap of the retained static world, and a
// top-down terrain heightfield. Both are traced by the GI compute passes, which own the textures.

// Mirrors `benilla_world::modern::gi::GiClip`.
struct GiClip {
    // Per level: xyz = the base voxel coord, w = voxel size (yd).
    level: array<vec4<f32>, 4>,
    // x = level count, y = voxels per side, z = step budget, w = the march's reach (yd).
    info: vec4<f32>,
    // The heightfield's world-to-clip, packed as rows so the struct stays a plain vec4 block.
    hf_row_x: vec4<f32>,
    hf_row_y: vec4<f32>,
    hf_row_z: vec4<f32>,
    // x = enabled, y = the lowest height the map can hold, z = its span (yd), w = its texel count.
    hf: vec4<f32>,
    // fork: P8b — x = the highest ground the heightfield holds (world Y, yd): the top of every
    // resident terrain cell's box, so no texel of the map can stand above it. A ray point above it
    // cannot be under the terrain, so `march` skips the map read there — and a ray climbing above it
    // past the clipmap can never come down to it, so the far walk ends. Exact, not a shortcut.
    // y = the highest voxel the clipmap can hold (world Y, yd): the top of every caster in it, with
    // a margin for whole voxels. A ray rising above both can hit nothing; `march` ends it there as
    // the miss it would have walked to. zw spare.
    bounds: vec4<f32>,
}

// The terrain's height under a world point, or far below everything when the map does not reach it.
fn terrain_height(hf_map: texture_depth_2d, clip: GiClip, p: vec3<f32>) -> f32 {
    if (clip.hf.x < 0.5) {
        return -1e9;
    }
    let q = vec4<f32>(p.x, p.y, p.z, 1.0);
    let uv = vec2<f32>(dot(clip.hf_row_x, q), dot(clip.hf_row_y, q));
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return -1e9;
    }
    let n = clip.hf.w;
    let texel = vec2<i32>(clamp(uv * n, vec2<f32>(0.0), vec2<f32>(n - 1.0)));
    let d = textureLoad(hf_map, texel, 0);
    if (d <= 0.0) {
        return -1e9;
    }
    return clip.hf.y + d * clip.hf.z;
}

// What a ray found.
struct GiHit {
    // 0 = the ray reached the sky, 1 = it hit the static world, 2 = it hit terrain.
    kind: f32,
    dist: f32,
    pos: vec3<f32>,
    albedo: vec3<f32>,
    nrm: vec3<f32>,
    // 1 when the surface is two-sided (foliage): it is lit through, never back-faced.
    two_sided: f32,
}

// The terrain's normal from its heightfield, by central difference.
fn terrain_normal(hf_map: texture_depth_2d, clip: GiClip, p: vec3<f32>) -> vec3<f32> {
    let e = max(2.0 * clip.hf.z / max(clip.hf.w, 1.0), 1.0);
    let hx = terrain_height(hf_map, clip, p + vec3<f32>(e, 0.0, 0.0))
        - terrain_height(hf_map, clip, p - vec3<f32>(e, 0.0, 0.0));
    let hz = terrain_height(hf_map, clip, p + vec3<f32>(0.0, 0.0, e))
        - terrain_height(hf_map, clip, p - vec3<f32>(0.0, 0.0, e));
    if (abs(hx) > 1e6 || abs(hz) > 1e6) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(vec3<f32>(-hx, 2.0 * e, -hz));
}

// fork: P8b — one toroidal step: a wrapped coordinate moved by exactly one, wrapped again. Equal to
// `wrap(v, n)` for any `v` in `-1..=n`, which is all a DDA step can produce.
fn wrap_step(v: i32, n: i32) -> i32 {
    return select(select(v, v - n, v >= n), v + n, v < 0);
}

fn voxel_texel(clip: GiClip, lvl: u32, w: vec3<i32>) -> vec3<i32> {
    let n = i32(clip.info.y);
    return vec3<i32>(wrap(w.x, n), wrap(w.y, n), wrap(w.z, n) + i32(lvl) * n);
}

fn miss(dist: f32) -> GiHit {
    return GiHit(0.0, dist, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 0.0);
}

// The ray march: voxel DDA inside each clipmap level, coarsening as the ray leaves one, with the
// terrain heightfield tested along the way and taking over past the coarsest level.
fn march(
    vox_albedo: texture_3d<f32>,
    vox_normal: texture_3d<f32>,
    hf_map: texture_depth_2d,
    clip: GiClip,
    origin: vec3<f32>,
    dir: vec3<f32>,
    start_level: u32,
) -> GiHit {
    let reach = clip.info.w;
    let levels = u32(clip.info.x);
    let n = clip.info.y;
    var budget = i32(clip.info.z);
    var t = 0.0;
    var lvl = start_level;
    // A zero component would make the slab distances infinite; nudge it instead of branching.
    let d = select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let stepv = sign(d);
    // fork: P8b — the step as integers, and the highest ground the terrain map holds
    // (`GiClip::bounds`): a point above it cannot be under the terrain, so the map is not read there.
    let istep = vec3<i32>(stepv);
    let ni = i32(n);
    let ground_top = clip.bounds.x;
    // Above this, going up, nothing is left to hit (`GiClip::bounds`).
    let ceiling = max(clip.bounds.y, ground_top);
    let rising = dir.y >= 0.0;

    while (lvl < levels && t < reach && budget > 0) {
        if (rising && origin.y + dir.y * t >= ceiling) {
            return miss(reach);
        }
        let vs = clip.level[lvl].w;
        let base = clip.level[lvl].xyz;
        var p = origin + dir * t;
        var w = floor(p / vs);
        if (any(w < base) || any(w >= base + n)) {
            // Outside this level: take the next one, which covers it.
            lvl = lvl + 1u;
            continue;
        }
        let tdelta = abs(vec3<f32>(vs) / d);
        var tmax = ((w + max(stepv, vec3<f32>(0.0))) * vs - p) / d;
        // fork: P8b — the voxel's slot in the level's toroidal texture, wrapped ONCE here and then
        // carried along with `w`: a DDA step moves one axis by exactly one voxel, so the wrap is a
        // compare, where `voxel_texel`'s six integer modulos a step were most of the loop's ALU.
        var wt = voxel_texel(clip, lvl, vec3<i32>(w)) - vec3<i32>(0, 0, i32(lvl) * ni);
        let layer = i32(lvl) * ni;
        // fork: the walk loads each voxel one step AHEAD of testing it. The next voxel's address is
        // pure DDA arithmetic, independent of what this one holds, so its load is issued before
        // this voxel's hit and terrain tests run and lands while they do — two loads in flight a
        // lane instead of one, on a loop that is otherwise waiting on memory. The walk, its tests
        // and their order are unchanged; a load past a hit is simply never read.
        var a = textureLoad(vox_albedo, vec3<i32>(wt.x, wt.y, wt.z + layer), 0);
        while (budget > 0) {
            budget = budget - 1;
            let texel = vec3<i32>(wt.x, wt.y, wt.z + layer);
            // The next voxel along the axis whose slab boundary is nearest.
            let seg = min(min(tmax.x, tmax.y), tmax.z);
            var w_next = w;
            var tmax_next = tmax;
            var wt_next = wt;
            if (tmax.x <= tmax.y && tmax.x <= tmax.z) {
                w_next.x = w.x + stepv.x;
                tmax_next.x = tmax.x + tdelta.x;
                wt_next.x = wrap_step(wt.x + istep.x, ni);
            } else if (tmax.y <= tmax.z) {
                w_next.y = w.y + stepv.y;
                tmax_next.y = tmax.y + tdelta.y;
                wt_next.y = wrap_step(wt.y + istep.y, ni);
            } else {
                w_next.z = w.z + stepv.z;
                tmax_next.z = tmax.z + tdelta.z;
                wt_next.z = wrap_step(wt.z + istep.z, ni);
            }
            // Always a texel of this level's slab (the wrap keeps it in range), read or not.
            let a_next = textureLoad(vox_albedo, vec3<i32>(wt_next.x, wt_next.y, wt_next.z + layer), 0);
            if (a.a > 0.5) {
                let packed = textureLoad(vox_normal, texel, 0);
                var face = packed.xyz * 2.0 - vec3<f32>(1.0);
                if (dot(face, face) < 0.2) {
                    face = -dir;
                }
                let hit_pos = origin + dir * t;
                return GiHit(1.0, t, hit_pos, a.rgb, normalize(face), 1.0 - packed.w);
            }
            // The terrain under this voxel, sampled at the segment's midpoint.
            let mid = origin + dir * (t + seg * 0.5);
            // Above the highest ground there is nothing to find in the map (`GiClip::bounds`).
            var h = -1e9;
            if (mid.y < ground_top) {
                h = terrain_height(hf_map, clip, mid);
            }
            if (mid.y < h) {
                // Bisect the last segment so the hit lands on the surface rather than a step edge.
                var lo = t;
                var hi = t + seg;
                for (var i = 0u; i < 4u; i = i + 1u) {
                    let m = 0.5 * (lo + hi);
                    let q = origin + dir * m;
                    if (q.y < terrain_height(hf_map, clip, q)) {
                        hi = m;
                    } else {
                        lo = m;
                    }
                }
                let hit_pos = origin + dir * hi;
                return GiHit(2.0, hi, hit_pos, vec3<f32>(0.0), terrain_normal(hf_map, clip, hit_pos), 0.0);
            }
            // Advance one voxel, onto the one already loaded.
            t = t + seg;
            if (t >= reach || (rising && origin.y + dir.y * t >= ceiling)) {
                return miss(reach);
            }
            w = w_next;
            tmax = tmax_next - vec3<f32>(seg);
            wt = wt_next;
            if (any(w < base) || any(w >= base + n)) {
                lvl = lvl + 1u;
                break;
            }
            a = a_next;
        }
    }

    // Past the clipmap the terrain map is all that is left; walk it in coarse steps.
    if (clip.hf.x > 0.5) {
        var t_far = t;
        let step = max(clip.level[levels - 1u].w, 4.0);
        while (t_far < reach && budget > 0) {
            // fork: P8b — a ray climbing above the highest ground can never come back down to it,
            // and past the clipmap the ground is all that is left: it misses, exactly as the rest
            // of this walk would have found.
            let at = origin + dir * t_far;
            if (dir.y >= 0.0 && at.y >= ground_top) {
                break;
            }
            budget = budget - 1;
            let q = origin + dir * (t_far + step * 0.5);
            var h = -1e9;
            if (q.y < ground_top) {
                h = terrain_height(hf_map, clip, q);
            }
            if (q.y < h) {
                let hit_pos = origin + dir * t_far;
                return GiHit(2.0, t_far, hit_pos, vec3<f32>(0.0), terrain_normal(hf_map, clip, hit_pos), 0.0);
            }
            t_far = t_far + step;
        }
    }
    return miss(reach);
}

// The sky's radiance in a direction, in the shared light's own units: the ambient row shaped by a
// gentle zenith gradient whose cosine-weighted mean over the hemisphere is exactly the row, so an
// open-field probe reproduces the classic ambient.
fn sky_radiance(g: GiParams, dir: vec3<f32>) -> vec3<f32> {
    let k = 1.0 - g.ground.w * (2.0 / 3.0) + g.ground.w * clamp(dir.y, 0.0, 1.0);
    return g.sky.rgb * k;
}

// P6: what a ray that reached nothing sees, given how enclosed its probe is. An open-field probe
// still reads the classic ambient exactly (`interior_class` is 0 there); a probe under a roof reads
// the building's own `MOHD.ambColor` instead, so a ray that escapes through an unvoxelized wall
// cannot flood the room with daylight.
fn miss_radiance(g: GiParams, dir: vec3<f32>, interior_class: f32) -> vec3<f32> {
    return mix(sky_radiance(g, dir), g.interior.rgb, clamp(interior_class, 0.0, 1.0));
}
