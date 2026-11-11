//! fork: static_gx's meshlets, native only — each baked item's triangles cut into meshlets of at
//! most [`MESHLET_VERTICES`] vertices and [`MESHLET_TRIANGLES`] triangles, each with a bounding
//! sphere and a normal cone, built beside the bake's mesh for the mesh-shader lane
//! (`render/meshlets.rs`). Nothing is built unless that lane can run ([`set_wanted`]).
//!
//! # The sizes
//!
//! 64 vertices and 124 triangles. NVIDIA's mesh-shader guidance is 64 vertices and up to 126
//! primitives an output block; AMD's RDNA guidance allows up to 128 and 256 with one thread a
//! vertex. 124 is the largest multiple of four under 126, meshoptimizer's own mesh-shader
//! default, so the mesh stage's [`MESH_THREADS`] (64) threads each write one vertex and at most two
//! triangles, and the format sits inside wgpu's recommended minimum mesh limits (256 and 256)
//! on every device that has the feature.
//!
//! # The builder
//!
//! A scan in index order, meshoptimizer's `buildMeshletsScan`: a triangle joins the open meshlet
//! unless its new vertices or the triangle itself would overflow it, which closes it. A meshlet
//! never spans two items, so a run of items (what a draw covers) is a contiguous run of meshlets.
//! It is linear in the indices (a per-vertex stamp names the open meshlet), so the bake's worker
//! threads take it alongside the mesh; the baked index order (each batch's authored order) keeps
//! the vertices local.
//!
//! # The bounds
//!
//! The sphere is the vertices' box centre with the farthest vertex's distance, in the region's
//! mesh-local space; a foliage-wind meshlet adds [`WIND_REACH_YD`], what the sway moves a vertex.
//! The cone is meshoptimizer's `computeClusterBounds`: the axis the mean of the triangles' unit
//! normals (`(b - a) × (c - a)`, front faces counter-clockwise as the pipelines cull), `mindp` the
//! least of their dots with it, and the cutoff `sqrt(1 - mindp²)`. A meshlet is wholly back-facing
//! from an eye when `dot(c - eye, axis) >= cutoff * |c - eye| + radius`. A two-sided or swaying
//! meshlet, or one whose normals spread past `mindp = 0.1`, takes no cone (cutoff byte 127).

use bevy::math::Vec3;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

/// Vertices a meshlet holds at most (the mesh stage's output vertices).
pub(crate) const MESHLET_VERTICES: u32 = 64;

/// Triangles a meshlet holds at most (the mesh stage's output primitives).
pub(crate) const MESHLET_TRIANGLES: u32 = 124;

/// Threads of a task workgroup: the meshlets it tests, one each.
pub(crate) const TASK_MESHLETS: u32 = 32;

/// Threads of a mesh workgroup: one per output vertex.
pub(crate) const MESH_THREADS: u32 = 64;

/// Bytes of the task payload (`GxTaskPayload` in `static_gx_mesh.wesl`): four words, then one
/// meshlet index per task thread.
pub(crate) const TASK_PAYLOAD_BYTES: u32 = 16 + 4 * TASK_MESHLETS;

/// Task workgroups one draw dispatches at most; a draw over more meshlets than this covers
/// ([`MAX_DRAW_MESHLETS`]) is drawn by the GPU cull's path instead.
pub(crate) const MAX_DRAW_TASKS: u32 = 65535;

/// Meshlets one draw covers at most.
pub(crate) const MAX_DRAW_MESHLETS: u32 = MAX_DRAW_TASKS * TASK_MESHLETS;

/// Yards a foliage-wind vertex moves at most (`wind_hook::tree_offset` at the default strength and
/// a full gust: the crown's weight 1.3 times the summed swings 0.39 is about 0.67 yd), with margin.
pub(crate) const WIND_REACH_YD: f32 = 1.0;

/// The cone cutoff byte that disables the cone test.
pub(crate) const NO_CONE: i8 = 127;

/// Whether bakes build meshlets: set once at render startup where the mesh-shader lane is
/// capable. A region baked before it has none and draws through the GPU cull's path.
static WANTED: AtomicBool = AtomicBool::new(false);

/// Have every later bake build meshlets.
pub(crate) fn set_wanted(on: bool) {
    WANTED.store(on, Ordering::Relaxed);
}

/// Whether a bake builds meshlets.
pub(crate) fn wanted() -> bool {
    WANTED.load(Ordering::Relaxed)
}

/// One region's meshlets, items in bake order.
#[derive(Default, Debug, PartialEq)]
pub(crate) struct GxMeshlets {
    /// Two table rows a meshlet, as `render/meshlets.rs` uploads them ("The meshlet table"), with
    /// the vertex and triangle offsets relative to this region's own lists:
    /// `[vertex offset, triangle offset, vertex count | triangle count << 8, cone]` and
    /// `[centre.x, centre.y, centre.z, radius]` (f32 bits, mesh-local).
    pub rows: Vec<[u32; 4]>,
    /// Per meshlet vertex, its index in the region's mesh (the slab offset is the draw's).
    pub vertices: Vec<u32>,
    /// Per triangle, its meshlet-local corners `a | b << 8 | c << 16`.
    pub triangles: Vec<u32>,
    /// Item `i`'s meshlets are `items[i]..items[i + 1]`.
    pub items: Vec<u32>,
}

impl GxMeshlets {
    /// Meshlets held.
    pub(crate) fn len(&self) -> u32 {
        (self.rows.len() / 2) as u32
    }
}

/// One item's share of the bake: its index range and what its meshlets' bounds must know.
pub(crate) struct ItemSpan {
    pub indices: Range<u32>,
    pub two_sided: bool,
    pub sway: bool,
}

/// The meshlet the scan is filling.
struct Open {
    vertices: Vec<u32>,
    triangles: Vec<[u8; 3]>,
}

/// Cut every item of a region into meshlets (the module header). `positions` are the mesh's
/// recentred positions and `indices` its whole index list.
pub(crate) fn build(positions: &[[f32; 3]], indices: &[u32], items: &[ItemSpan]) -> GxMeshlets {
    let mut out = GxMeshlets {
        items: Vec::with_capacity(items.len() + 1),
        ..GxMeshlets::default()
    };
    // `stamp[v]` is the meshlet `v` was last added to, `local[v]` its corner there.
    let mut stamp = vec![u32::MAX; positions.len()];
    let mut local = vec![0u8; positions.len()];
    let mut id = 0u32;
    let mut open = Open {
        vertices: Vec::with_capacity(MESHLET_VERTICES as usize),
        triangles: Vec::with_capacity(MESHLET_TRIANGLES as usize),
    };
    for item in items {
        out.items.push(out.len());
        let range = item.indices.start as usize..item.indices.end as usize;
        let Some(tris) = indices.get(range) else {
            continue;
        };
        for tri in tris.as_chunks::<3>().0 {
            let [a, b, c] = *tri;
            if [a, b, c].iter().any(|&v| v as usize >= positions.len()) {
                continue;
            }
            let fresh = |v: u32| stamp[v as usize] != id;
            let new = u32::from(fresh(a))
                + u32::from(fresh(b) && b != a)
                + u32::from(fresh(c) && c != a && c != b);
            if open.vertices.len() as u32 + new > MESHLET_VERTICES
                || open.triangles.len() as u32 + 1 > MESHLET_TRIANGLES
            {
                close(&mut open, positions, item, &mut out);
                id += 1;
            }
            let mut corner = |v: u32| {
                if stamp[v as usize] != id {
                    stamp[v as usize] = id;
                    local[v as usize] = open.vertices.len() as u8;
                    open.vertices.push(v);
                }
                local[v as usize]
            };
            let t = [corner(a), corner(b), corner(c)];
            open.triangles.push(t);
        }
        if !open.triangles.is_empty() {
            close(&mut open, positions, item, &mut out);
            id += 1;
        }
    }
    out.items.push(out.len());
    out
}

/// Append the open meshlet with its bounds and start an empty one.
fn close(open: &mut Open, positions: &[[f32; 3]], item: &ItemSpan, out: &mut GxMeshlets) {
    let p = |corner: u8| Vec3::from(positions[open.vertices[usize::from(corner)] as usize]);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for &v in &open.vertices {
        let q = Vec3::from(positions[v as usize]);
        lo = lo.min(q);
        hi = hi.max(q);
    }
    let centre = (lo + hi) * 0.5;
    let mut radius = open
        .vertices
        .iter()
        .map(|&v| Vec3::from(positions[v as usize]).distance(centre))
        .fold(0.0f32, f32::max);
    if item.sway {
        radius += WIND_REACH_YD;
    }
    let cone = if item.two_sided || item.sway {
        pack_cone(Vec3::ZERO, NO_CONE)
    } else {
        let normals: Vec<Vec3> = open
            .triangles
            .iter()
            .filter_map(|&[a, b, c]| {
                let n = (p(b) - p(a)).cross(p(c) - p(a));
                (n.length_squared() > 0.0).then(|| n.normalize())
            })
            .collect();
        cone_of(&normals)
    };
    out.rows.push([
        out.vertices.len() as u32,
        out.triangles.len() as u32,
        open.vertices.len() as u32 | ((open.triangles.len() as u32) << 8),
        cone,
    ]);
    out.rows.push([
        centre.x.to_bits(),
        centre.y.to_bits(),
        centre.z.to_bits(),
        radius.to_bits(),
    ]);
    out.vertices.append(&mut open.vertices);
    out.triangles.extend(
        open.triangles
            .drain(..)
            .map(|[a, b, c]| u32::from(a) | (u32::from(b) << 8) | (u32::from(c) << 16)),
    );
}

/// The packed cone of unit `normals`: the snorm8 axis in bytes 0..3 and the snorm8 cutoff in
/// byte 3, the cutoff raised by the axis's rounding so the test stays conservative.
fn cone_of(normals: &[Vec3]) -> u32 {
    let sum: Vec3 = normals.iter().copied().sum();
    if normals.is_empty() || sum.length() < 1e-4 {
        return pack_cone(Vec3::ZERO, NO_CONE);
    }
    let axis = sum.normalize();
    let mindp = normals.iter().map(|n| n.dot(axis)).fold(1.0f32, f32::min);
    if mindp <= 0.1 {
        return pack_cone(Vec3::ZERO, NO_CONE);
    }
    let cutoff = (1.0 - mindp * mindp).max(0.0).sqrt();
    let q = (axis * 127.0)
        .round()
        .clamp(Vec3::splat(-127.0), Vec3::splat(127.0));
    let err = (q / 127.0 - axis).abs();
    let cutoff = (127.0 * (cutoff + err.x + err.y + err.z)).ceil().min(127.0) as i8;
    pack_cone(q, cutoff)
}

fn pack_cone(axis: Vec3, cutoff: i8) -> u32 {
    let byte = |x: f32| u32::from((x as i8) as u8);
    byte(axis.x) | (byte(axis.y) << 8) | (byte(axis.z) << 16) | (u32::from(cutoff as u8) << 24)
}

/// The cone test the task shader runs (`static_gx_mesh.wesl`, `backfacing`), for the tests.
#[cfg(test)]
fn backfacing(cone: u32, centre: Vec3, radius: f32, eye: Vec3) -> bool {
    let s = |shift: u32| f32::from(((cone >> shift) & 0xff) as u8 as i8) / 127.0;
    if ((cone >> 24) & 0xff) as u8 as i8 == NO_CONE {
        return false;
    }
    let axis = Vec3::new(s(0), s(8), s(16));
    let v = centre - eye;
    v.dot(axis) >= s(24) * v.length() + radius
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `n` x `n` quad grid in the y = 0 plane, its triangles counter-clockwise from +y.
    fn grid(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                positions.push([x as f32, 0.0, z as f32]);
            }
        }
        let at = |x: u32, z: u32| z * (n + 1) + x;
        let mut indices = Vec::new();
        for z in 0..n {
            for x in 0..n {
                indices.extend([at(x, z), at(x, z + 1), at(x + 1, z)]);
                indices.extend([at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)]);
            }
        }
        (positions, indices)
    }

    /// Every meshlet stays inside the limits, never spans an item, and its triangles are the
    /// item's in order.
    #[test]
    fn meshlets_hold_the_limits_and_every_triangle_in_order() {
        let (positions, indices) = grid(24);
        let half = (indices.len() / 2) as u32 / 3 * 3;
        let items = [
            ItemSpan {
                indices: 0..half,
                two_sided: false,
                sway: false,
            },
            ItemSpan {
                indices: half..indices.len() as u32,
                two_sided: true,
                sway: false,
            },
        ];
        let m = build(&positions, &indices, &items);
        assert_eq!(m.items.len(), 3);
        assert_eq!(*m.items.last().unwrap(), m.len());
        let mut rebuilt = Vec::new();
        for k in 0..m.len() as usize {
            let [vo, to, counts, _] = m.rows[2 * k];
            let (vc, tc) = (counts & 0xff, counts >> 8);
            assert!(vc <= MESHLET_VERTICES && tc <= MESHLET_TRIANGLES && tc > 0);
            for t in &m.triangles[to as usize..(to + tc) as usize] {
                for shift in [0, 8, 16] {
                    let corner = (t >> shift) & 0xff;
                    assert!(corner < vc);
                    rebuilt.push(m.vertices[(vo + corner) as usize]);
                }
            }
        }
        assert_eq!(
            rebuilt, indices,
            "the meshlets' triangles are the index list"
        );
        // The second item's meshlets start after the first's: no meshlet spans the two.
        let first_end = m.items[1] as usize;
        let tris_before: u32 = (0..first_end).map(|k| m.rows[2 * k][2] >> 8).sum();
        assert_eq!(tris_before * 3, half);
    }

    /// A flat floor's cone faces up: culled from below, kept from above; a two-sided item has none.
    #[test]
    fn the_cone_culls_a_floor_from_below_only() {
        let (positions, indices) = grid(4);
        let span = |two_sided| ItemSpan {
            indices: 0..indices.len() as u32,
            two_sided,
            sway: false,
        };
        let m = build(&positions, &indices, &[span(false)]);
        let [_, _, _, cone] = m.rows[0];
        let [cx, cy, cz, r] = m.rows[1].map(f32::from_bits);
        let centre = Vec3::new(cx, cy, cz);
        assert!(backfacing(cone, centre, r, centre - Vec3::Y * 50.0));
        assert!(!backfacing(cone, centre, r, centre + Vec3::Y * 50.0));
        let m = build(&positions, &indices, &[span(true)]);
        assert!(!backfacing(
            m.rows[0][3],
            centre,
            r,
            centre - Vec3::Y * 50.0
        ));
    }

    /// A swaying meshlet's sphere grows by the wind's reach and drops its cone.
    #[test]
    fn a_swaying_meshlet_is_wider_and_coneless() {
        let (positions, indices) = grid(2);
        let span = |sway| ItemSpan {
            indices: 0..indices.len() as u32,
            two_sided: false,
            sway,
        };
        let still = build(&positions, &indices, &[span(false)]);
        let sway = build(&positions, &indices, &[span(true)]);
        let r = |m: &GxMeshlets| f32::from_bits(m.rows[1][3]);
        assert!((r(&sway) - r(&still) - WIND_REACH_YD).abs() < 1e-5);
        assert_eq!((sway.rows[0][3] >> 24) as u8 as i8, NO_CONE);
    }
}
