//! The mouseover pick's two passes over one skinning. The reference's pick tests each candidate's
//! posed mesh exactly and, when nothing hit, again with every vertex moved by its skinned normal
//! (the halo). [`super::ray_posed_mesh`] is one such pass, blending a matrix per vertex each call;
//! [`PosedPickScratch`] keeps the exact pass's skinned positions and each vertex's halo position, so
//! the halo pass is triangle tests alone. The arithmetic is `ray_posed_mesh`'s, operation for
//! operation, so both passes answer what its two calls answer.

use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;

use super::pick::ray_triangle;

/// One skinned part in the arena: its mesh and its vertex range. `halo` is false for a mesh
/// without normals, which the halo pass cannot hit.
struct Span {
    mesh: AssetId<Mesh>,
    start: usize,
    len: usize,
    halo: bool,
}

/// One pick's skinned candidate parts, reused across frames so a warm pick allocates nothing.
#[derive(Default)]
pub struct PosedPickScratch {
    pos: Vec<Vec3>,
    /// `pos[i]` moved by its skinned normal, index-aligned; padding for a span without normals.
    halo: Vec<Vec3>,
    spans: Vec<Span>,
}

impl PosedPickScratch {
    /// Forget the last pick's parts.
    pub fn clear(&mut self) {
        self.pos.clear();
        self.halo.clear();
        self.spans.clear();
    }

    /// Skin `mesh_id` through `palette`, keep it for [`Self::ray_halo`], and return the exact
    /// pass's hit, as `ray_posed_mesh(.., false)` does. A mesh that cannot be skinned still takes
    /// a span, so span indices follow the calls; that span never hits.
    pub fn ray_exact(
        &mut self,
        mesh_assets: &Assets<Mesh>,
        mesh_id: AssetId<Mesh>,
        palette: &[Mat4],
        origin: Vec3,
        dir: Vec3,
    ) -> Option<f32> {
        let mut span = Span {
            mesh: mesh_id,
            start: self.pos.len(),
            len: 0,
            halo: false,
        };
        let hit = self.skin_and_cast(mesh_assets, &mut span, palette, origin, dir);
        self.spans.push(span);
        hit
    }

    fn skin_and_cast(
        &mut self,
        mesh_assets: &Assets<Mesh>,
        span: &mut Span,
        palette: &[Mat4],
        origin: Vec3,
        dir: Vec3,
    ) -> Option<f32> {
        let mesh = mesh_assets.get(span.mesh)?;
        let VertexAttributeValues::Float32x3(positions) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)?
        else {
            return None;
        };
        // The joints ride the WOW attributes, as in `ray_posed_mesh`.
        let (
            Some(VertexAttributeValues::Uint16x4(joints)),
            Some(VertexAttributeValues::Float32x4(weights)),
        ) = (
            mesh.attribute(benilla_assets::ATTRIBUTE_WOW_JOINT_INDEX),
            mesh.attribute(benilla_assets::ATTRIBUTE_WOW_JOINT_WEIGHT),
        )
        else {
            warn_once!(
                "a skinned part's mesh lacks the WOW joint attributes — the posed pick cannot hit it"
            );
            return None;
        };
        let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(n)) => Some(n),
            _ => None,
        };
        span.halo = normals.is_some();
        for (i, (p, (j, w))) in positions
            .iter()
            .zip(joints.iter().zip(weights.iter()))
            .enumerate()
        {
            let mut m = Mat4::ZERO;
            for k in 0..4 {
                if w[k] > 0.0 {
                    if let Some(mk) = palette.get(j[k] as usize) {
                        m += *mk * w[k];
                    }
                }
            }
            let out = (m * Vec4::new(p[0], p[1], p[2], 1.0)).truncate();
            self.pos.push(out);
            self.halo.push(match normals {
                Some(ns) => {
                    let n = ns[i];
                    out + m.transform_vector3(Vec3::new(n[0], n[1], n[2]))
                }
                None => Vec3::ZERO,
            });
        }
        span.len = self.pos.len() - span.start;
        cast(mesh, &self.pos[span.start..], origin, dir)
    }

    /// The halo pass for the `index`-th [`Self::ray_exact`] call since the last clear, as
    /// `ray_posed_mesh(.., true)` answers it, from the vertices that call skinned.
    pub fn ray_halo(
        &self,
        mesh_assets: &Assets<Mesh>,
        index: usize,
        origin: Vec3,
        dir: Vec3,
    ) -> Option<f32> {
        let span = self.spans.get(index)?;
        if !span.halo {
            return None;
        }
        let mesh = mesh_assets.get(span.mesh)?;
        cast(
            mesh,
            &self.halo[span.start..span.start + span.len],
            origin,
            dir,
        )
    }
}

/// The nearest triangle hit over `world`, in index order, as `ray_posed_mesh` folds it.
fn cast(mesh: &Mesh, world: &[Vec3], origin: Vec3, dir: Vec3) -> Option<f32> {
    let tri = |a: usize, b: usize, c: usize| -> Option<f32> {
        ray_triangle(origin, dir, &[world[a], world[b], world[c]])
    };
    match mesh.indices()? {
        Indices::U16(ix) => ix
            .as_chunks::<3>()
            .0
            .iter()
            .filter_map(|c| tri(c[0] as usize, c[1] as usize, c[2] as usize))
            .fold(None::<f32>, |acc, t| Some(acc.map_or(t, |a| a.min(t)))),
        Indices::U32(ix) => ix
            .as_chunks::<3>()
            .0
            .iter()
            .filter_map(|c| tri(c[0] as usize, c[1] as usize, c[2] as usize))
            .fold(None::<f32>, |acc, t| Some(acc.map_or(t, |a| a.min(t)))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::PrimitiveTopology;

    /// A two-joint quad facing +Z, its right half on joint 1.
    fn quad() -> Mesh {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 4]);
        mesh.insert_attribute(
            benilla_assets::ATTRIBUTE_WOW_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        );
        mesh.insert_attribute(
            benilla_assets::ATTRIBUTE_WOW_JOINT_WEIGHT,
            VertexAttributeValues::Float32x4(vec![[1.0, 0.0, 0.0, 0.0]; 4]),
        );
        mesh.insert_indices(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
        mesh
    }

    #[test]
    fn both_passes_answer_what_two_ray_posed_mesh_calls_answer() {
        let mut assets = Assets::<Mesh>::default();
        let id = assets.add(quad()).id();
        let palette = [
            Mat4::from_translation(Vec3::new(0.3, 0.0, -2.0)),
            Mat4::from_scale_rotation_translation(
                Vec3::splat(1.5),
                Quat::from_rotation_y(0.4),
                Vec3::new(0.1, 0.2, -2.5),
            ),
        ];
        let mut scratch = PosedPickScratch::default();
        let mut hits = 0;
        // A ray through each joint's half, one past the edge, one wide of everything.
        for (o, d) in [
            (Vec3::new(0.2, 0.1, 5.0), Vec3::NEG_Z),
            (Vec3::new(-0.5, 0.1, 5.0), Vec3::NEG_Z),
            (Vec3::new(-1.6, 0.0, 5.0), Vec3::NEG_Z),
            (Vec3::new(9.0, 9.0, 5.0), Vec3::NEG_Z),
        ] {
            scratch.clear();
            let exact = scratch.ray_exact(&assets, id, &palette, o, d);
            let halo = scratch.ray_halo(&assets, 0, o, d);
            assert_eq!(
                exact,
                super::super::ray_posed_mesh(&assets, id, &palette, o, d, false)
            );
            assert_eq!(
                halo,
                super::super::ray_posed_mesh(&assets, id, &palette, o, d, true)
            );
            hits += usize::from(exact.is_some()) + usize::from(halo.is_some());
        }
        // Not a vacuous None == None: the two rays on the quad hit in both passes.
        assert_eq!(hits, 4);
    }
}
