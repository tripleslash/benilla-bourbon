// Modern lighting (FORK.md, P6): the local-light binning pass — one invocation
// per world cell, writing the cell's light list straight into the light-grid region of the shared
// modern storage block (`sunshadow::ModernBlock.lights`), which is what every lane reads.
//
// The candidate set is the shared light table the reference itself commits (already room-gated and
// capped at 256 by `lighting::global_light`), so a modern frame lights from exactly the lights a
// classic frame would — it just stops being limited to the three nearest the receiving unit.
#import benilla_modern::locallight

// The shared light buffer, as far as this pass reads it: the header rows (row 20 `.x` is the live
// light count) and the point table behind them.
struct LightTable {
    rows: array<vec4<f32>, 21>,
    points: array<vec4<f32>, 512>,
}

// The light-grid region of the shared modern storage block, bound alone and read-write. Same bytes
// as `sunshadow::ModernBlock`'s `lights` + `light_spare` + `light_cells`.
struct LightBlock {
    params: locallight::LightGrid,
    shadow: array<vec4<f32>, 12>,
    spare: array<vec4<f32>, 14>,
    cells: array<vec4<u32>, 55296>,
}

// fork: each table light's steady strength (`lighting::LightRank`): the luminance of its committed
// colour with no flame flicker on it, in the table's order.
struct LightRank {
    lum: array<f32, 256>,
}

@group(0) @binding(0) var<storage, read> light: LightTable;
@group(0) @binding(1) var<storage, read_write> grid: LightBlock;
@group(0) @binding(2) var<storage, read> light_rank: LightRank;

// Which atlas slot a light holds, or `MAX_SHADOWED` when it holds none. Matched on position rather
// than on table order: the shadowed set is chosen on the CPU a frame's worth of systems before the
// table is packed, and nothing keeps the two orders in step.
//
// The slots themselves are STABLE across frames (`modern::lights::update_lights`) — a light keeps
// its index until it leaves the radius — so the nibble this writes still names the same six atlas
// tiles next frame. It did not always: a compacted set shifted every slot when one light dropped
// out, and a fragment then shaded against another lamp's shadow.
//
// fork: a slot's position is the MIDDLE of its light's sway (a lantern on a swinging bone), and the
// table carries where the light is this frame — so the match is within the slot's own slack
// (`ShadowLight::slack`, `spare` rows 6-8), not exact.
fn shadow_slot(pos: vec3<f32>) -> u32 {
    let n = u32(grid.params.misc.y);
    for (var s = 0u; s < locallight::MAX_SHADOWED; s = s + 1u) {
        if (s >= n) {
            break;
        }
        // A slot no light holds is all zeroes, and its `w` (the range) is what says so. Without
        // this a lamp standing at the world origin would match every empty slot.
        if (grid.shadow[s].w <= 0.0) {
            continue;
        }
        let d = grid.shadow[s].xyz - pos;
        // The row is loaded before a component is picked (see the lanes' `light_spare` reads).
        let row = grid.spare[6u + s / 4u];
        let slack = max(row[s % 4u], 0.01);
        if (dot(d, d) < slack * slack) {
            return s;
        }
    }
    return locallight::MAX_SHADOWED;
}

@compute @workgroup_size(64)
fn bin_lights(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = grid.params;
    let total = u32(p.misc.x);
    let cell = id.x;
    if (cell >= total) {
        return;
    }
    // fork: P8 — cells past the near level's are the FAR level's (`locallight::far_cell_at`): the
    // same binning at a coarser cell, centred on the same camera, so a lamp 200 yd off is in a list
    // a fragment can read instead of in no list at all.
    let near_total = locallight::near_cells(p);
    var n = vec3<u32>(p.dims.xyz);
    var base = vec3<f32>(p.base.xyz);
    var size = p.dims.w;
    var within = cell;
    if (cell >= near_total) {
        n = vec3<u32>(locallight::far_dims(p));
        base = vec3<f32>(locallight::far_base(p));
        size = locallight::far_size(p);
        within = cell - near_total;
    }
    let c = vec3<i32>(
        i32(within % n.x),
        i32((within / n.x) % n.y),
        i32(within / (n.x * n.y)),
    );
    let lo = (base + vec3<f32>(c)) * size;
    let hi = lo + vec3<f32>(size);
    let mid = 0.5 * (lo + hi);

    var best = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var score = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var held = 0u;
    var seen = 0u;
    let count = min(u32(light.rows[20].x), 256u);
    for (var i = 0u; i < count; i = i + 1u) {
        let pos_range = light.points[2u * i];
        // The cell's nearest point to the light: a light reaches the cell iff that point is inside
        // its range, which is the test a per-pixel lookup would make anywhere in the cell.
        let near = clamp(pos_range.xyz, lo, hi);
        let dv = near - pos_range.xyz;
        if (dot(dv, dv) > pos_range.w * pos_range.w) {
            continue;
        }
        seen = seen + 1u;
        // Ranked by what the cell's centre would actually receive, so an overflowing cell drops the
        // lights that were about to contribute least. fork: by the light's STEADY strength, never
        // the table's colour: that carries the flame flicker, and two flames near the cut of a full
        // cell then traded places as they breathed, stepping the cell's light each time.
        let w = light_rank.lum[i] * locallight::falloff(length(mid - pos_range.xyz), pos_range.w);
        // Insertion into a descending list of eight.
        var slot = held;
        for (var k = 0u; k < held; k = k + 1u) {
            if (w > score[k]) {
                slot = k;
                break;
            }
        }
        if (slot >= 8u) {
            continue;
        }
        var k = min(held, 7u);
        while (k > slot) {
            score[k] = score[k - 1u];
            best[k] = best[k - 1u];
            k = k - 1u;
        }
        score[slot] = w;
        best[slot] = i;
        held = min(held + 1u, 8u);
    }

    var words = vec4<u32>(0u);
    words.x = held | (min(seen, 255u) << 8u);
    for (var k = 0u; k < held; k = k + 1u) {
        let i = best[k];
        let byte = (i & 0xffu) << ((k & 3u) * 8u);
        if (k < 4u) {
            words.y = words.y | byte;
        } else {
            words.z = words.z | byte;
        }
        // The nibble holds `slot + 1` (0 = "no cube"), so it addresses up to fifteen slots and P8's
        // twelve fit it unchanged — the cell is still one `vec4<u32>`.
        let slot = shadow_slot(light.points[2u * i].xyz);
        if (slot < locallight::MAX_SHADOWED) {
            words.w = words.w | ((slot + 1u) << (k * 4u));
        }
    }
    grid.cells[cell] = words;
}
