//! The animated liquid surfaces: one shared material per (kind, [`LiquidPath`], scroll) over a
//! `texture_2d_array` of the kind's frames, the two spawn paths (an ADT chunk's MCLQ, a WMO
//! group's MLIQ) and the flat mesh. The 24 fps frame flip and the lava scroll run in the shader
//! off the wall clock, so no CPU mutates a material; a deterministic run bakes the clock off
//! (frame 0, scroll 0). Day and night follow server time.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::MeshAabb;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;

use super::art;
use super::query::{wet_footprint, FoamPatch, LiquidSource, WmoPool};
use super::ripple;
use super::screen::LiquidScreenTargets;
use super::WaterDetail;
use crate::collision::liquid_layers;
use crate::lighting::{CATA_WATER_SHININESS, WATER_SHININESS};
use crate::terrain_stream::{build_collider_task, PendingCollider};
use avian3d::prelude::{Collider, RigidBody};
use benilla_assets::coords::wow_to_bevy;
use benilla_assets::materials::{LiquidCataExt, LiquidCataMaterial, LiquidExt, LiquidMaterial};
use benilla_assets::LockRecover;
use benilla_assets::{liquid_frame_array, RenderConfig, WorldAssets};
use benilla_formats::{read_texture_mip_chain, BlpMipChain, LiquidKind, LiquidMesh};

/// The shared liquid materials by [`LiquidKey`]; absent without client data.
#[derive(Resource, Default)]
pub(crate) struct LiquidAssets {
    materials: HashMap<LiquidKey, LiquidEntry>,
}

/// Which of the reference's three liquid renderers draws a surface: `0x6b62e0` sends the type
/// nibble to a category, and category 0 (nibbles 0/4/8) splits on the group's `MOGP.flags & 0x48`.
/// Magma and slime share `0x6b68f0` (WMO) / `0x68dca0` (ADT) whatever the flags, so for them the
/// path only picks the fog block. Fog is per pass: the WMO liquid pass re-submits the smoothed
/// interior fog block (`0x6b6323`–`0x6b6342`) under the same `[0xca7f00]` gate as the WMO geometry,
/// so a pool fogs like its walls, while ADT liquid submits none and draws under the scene block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LiquidPath {
    /// An ADT chunk's MCLQ (`0x6851b0` river, `0x685010` ocean): two stages, the depth swatch at
    /// `tc0 = (0.5, LUT[depthByte])` and the sheet at `tc1 = (col·¼, row·¼)`; no vertex colour.
    Adt,
    /// A WMO group's MLIQ, exterior or exterior-lit (`MOGP.flags & 0x48 != 0`), `0x6b6630`: one
    /// stage, a 9-float vertex with an up normal and a colour dword, `MapObjExtWater0.bls` bound at
    /// `0x6b6654`.
    WmoExterior,
    /// A WMO group's MLIQ, a true interior (`MOGP.flags & 0x48 == 0`), `0x6b6420`: one stage, a
    /// 6-float vertex with no normal, lighting off, no pixel program.
    WmoInterior,
}

impl LiquidPath {
    /// A WMO group's own class, as the reference's `[owner+0x10] & 0x48` test reads it.
    pub(crate) fn wmo(interior: bool) -> Self {
        if interior {
            Self::WmoInterior
        } else {
            Self::WmoExterior
        }
    }

    fn interior_fog(self) -> bool {
        matches!(self, Self::WmoInterior)
    }

    /// The renderer selector `liquid.wgsl` branches on (`LiquidParams.path.x`).
    fn shader_id(self) -> f32 {
        match self {
            Self::Adt => 0.0,
            Self::WmoExterior => 1.0,
            Self::WmoInterior => 2.0,
        }
    }

    /// Does this arm's pool take its body colour from its own MOMT `diffColor`? Only the interior
    /// one: the exterior reads a live `Light.dbc` band, and ADT liquid has no MOMT.
    fn takes_material_body_color(self) -> bool {
        matches!(self, Self::WmoInterior)
    }
}

/// Which shared material a surface takes. The scroll is per nibble and per path, which
/// [`LiquidKind`] collapses (2 and 6 are both `Magma`), so it keys the material too.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct LiquidKey {
    kind: LiquidKind,
    path: LiquidPath,
    /// This surface takes the animated stage-0 texture matrix ([`scrolls`]).
    scroll: bool,
}

/// Does a WMO MLIQ surface of this type nibble take the reference's animated stage-0 texture
/// matrix, the lava/slime scroll? Only 6 and 7: the WMO magma/slime kernel `0x6b68f0` gates on
/// `and esi,0xc; cmp esi,4` over the `0x6ba970` nibble and builds an identity but for element 13,
/// the v-translate. Nibbles 2 and 3 reach the same kernel and stay still (the Great Forge is 2,
/// Blackrock 6). ADT liquid never scrolls, though open-world lava is nibble 6: its kernel
/// `0x68dca0` zeroes element 13 at `0x68dda1`, so the caller gates the path. The phase runs off
/// the reference's uptime clock; only its rate and period are reproducible.
fn scrolls(nibble: u8) -> bool {
    matches!(nibble, 6 | 7)
}

/// The two materials a liquid key can wear — one per renderer, built side by side at startup so
/// that flipping `waterDetail` is a component swap and never an asset build. Both share the kind's
/// one decoded frame array; the pair costs a few hundred bytes, not a second 30 × 256² decode.
struct LiquidEntry {
    /// The 1.12.1 renderer (`liquid.wgsl`).
    material: Handle<LiquidMaterial>,
    /// The Cataclysm renderer (`liquid_cata.wgsl`).
    cata: Handle<LiquidCataMaterial>,
}

impl LiquidAssets {
    /// The shared material for a kind, renderer and scroll, if one was built and its frames loaded.
    pub(crate) fn material(
        &self,
        kind: LiquidKind,
        path: LiquidPath,
        scroll: bool,
    ) -> Option<Handle<LiquidMaterial>> {
        self.materials
            .get(&LiquidKey { kind, path, scroll })
            .map(|e| e.material.clone())
    }

    /// The same surface's **Cataclysm** material. Misses exactly where [`Self::material`] misses,
    /// the two being built in the same pass off the same frame array.
    fn cata_material(
        &self,
        kind: LiquidKind,
        path: LiquidPath,
        scroll: bool,
    ) -> Option<Handle<LiquidCataMaterial>> {
        self.materials
            .get(&LiquidKey { kind, path, scroll })
            .map(|e| e.cata.clone())
    }
}

/// Which shared material a spawned surface should be wearing.
///
/// Carried on the entity because the *renderer* can change under it at runtime and the spawn site
/// is long gone by then: [`apply_water_detail`] re-materials every liquid surface in the world when
/// `waterDetail` moves, and this is how it knows which of the ~30 shared materials each one wants.
#[derive(Component, Clone, Copy)]
pub(crate) struct LiquidLane {
    kind: LiquidKind,
    path: LiquidPath,
    scroll: bool,
}

/// Record which shared material a freshly-spawned surface wants.
///
/// The material itself is attached by [`apply_water_detail`], not here, and that is the whole point:
/// *which component type* it is differs between the two renderers
/// (`MeshMaterial3d<LiquidMaterial>` vs `MeshMaterial3d<LiquidCataMaterial>`, the price of the two
/// paths not sharing a bind group), and the streamer's two spawn sites are already at Bevy's
/// 16-param ceiling with no room to take the setting. One system owning the decision also means
/// there is no second place that can disagree with it — a tile streaming in on the same frame the
/// player flips the toggle cannot land on the losing side of a race, because it is the same sweep
/// that dresses both.
fn record_lane(entity: &mut EntityCommands, lane: LiquidLane) {
    entity.insert(lane);
}

/// Marks a spawned liquid surface: one per MCNK liquid layer or WMO group pool, and the cell entity
/// that draws an ADT tile's layers ([`spawn_liquids`]).
#[derive(Component)]
pub(crate) struct LiquidSurface;

/// fork: perf — on an ADT draw entity: the world box (centre, half extents) of each MCNK layer
/// it draws. The mirror tracks its plane per liquid instance (`reflect`), and a cell is several.
#[derive(Component)]
pub(crate) struct LiquidLayerBoxes(pub(crate) Vec<(Vec3, Vec3)>);

/// fork: perf — a tile's liquid layers draw in cells of 4×4 chunks, the span its terrain cells
/// take (`terrain_stream::furnish`).
const CELL_SPAN: usize = 4;

/// fork: perf — what ADT layers must share to draw as one mesh.
#[derive(PartialEq)]
enum DrawGroup {
    /// Layers of one kind lying flat at one height (`level`, the height's bits) in one cell.
    Cell {
        cell: (usize, usize),
        kind: LiquidKind,
        level: u32,
    },
    /// A layer with any slope draws alone, the `n`th of its tile.
    Alone(usize),
}

impl DrawGroup {
    /// The group of the tile's `n`th layer, which sits in MCIN chunk `chunk` (row·16 + column).
    fn of(n: usize, chunk: usize, lq: &LiquidMesh) -> Self {
        let level = lq.positions.first().map_or(0.0, |p| p[2]);
        if lq.positions.iter().all(|p| p[2] == level) {
            Self::Cell {
                cell: (chunk / 16 / CELL_SPAN, chunk % 16 / CELL_SPAN),
                kind: lq.kind,
                level: level.to_bits(),
            }
        } else {
            Self::Alone(n)
        }
    }
}

/// A layer's world box, as its own mesh's `Aabb` was: over every grid vertex, wet or dry.
fn layer_box(lq: &LiquidMesh) -> Option<(Vec3, Vec3)> {
    let mut points = lq.positions.iter().map(|p| wow_to_bevy(*p));
    let first = points.next()?;
    let (lo, hi) = points.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
    Some(((lo + hi) * 0.5, (hi - lo) * 0.5))
}

/// `WOW_NO_LIQUID`: hide every liquid surface and only the surface, since the swim grid, foam and
/// sound ride sibling components. An override run after both per-frame `Visibility` owners (the
/// exterior cull for ADT surfaces, `apply_model_visibility` for WMO pools), so it wins the frame.
pub(super) fn hide_liquid_surfaces(mut surfaces: Query<&mut Visibility, With<LiquidSurface>>) {
    for mut vis in &mut surfaces {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
}

/// The ambient loop's sound-class nibble, resolved through `SoundWaterType.dbc` (`0x54e0a0`), on
/// every liquid surface; the driver reads the geometry off the same entity's `WaterChunkInfo`.
#[derive(Component)]
pub(crate) struct LiquidSoundSource {
    /// The surface's sound-class nibble (`class = n & 3`, `FluidSpeed = n & 0xc`).
    pub(crate) nibble: u8,
}

/// Spawn an ADT tile's liquid surfaces into `entities`, to despawn with the tile. `liquids` pairs
/// each layer with its MCIN chunk index.
///
/// fork: perf — each MCNK layer is an entity carrying its facts: the swim grid, the sound class,
/// the foam mark and the waterline collider. The drawing is per cell ([`DrawGroup`]): the
/// reference queues one batch per layer (`0x683ab0`), and a coast is a thousand of them in view,
/// each a draw of the same material. Every per-layer fact the shader reads is per vertex, so a
/// cell's layers are one mesh.
pub(crate) fn spawn_liquids<'a>(
    commands: &mut Commands,
    liquids: impl Iterator<Item = (usize, &'a LiquidMesh)>,
    liquid_assets: Option<&LiquidAssets>,
    meshes: &mut Assets<Mesh>,
    entities: &mut Vec<Entity>,
) {
    let Some(liquid) = liquid_assets else {
        return;
    };
    // The draw groups in first-seen order: a tile holds a few dozen at most.
    let mut groups: Vec<(DrawGroup, LiquidLane, Vec<&LiquidMesh>)> = Vec::new();
    for (n, (chunk, lq)) in liquids.enumerate() {
        // The scene fog (the ADT passes submit none and draw under the scene submit `0x66ff20`),
        // and never a scroll, though ADT magma is nibble 6: its kernel zeroes the v-translate.
        let lane = LiquidLane {
            kind: lq.kind,
            path: LiquidPath::Adt,
            scroll: false,
        };
        if liquid.material(lq.kind, LiquidPath::Adt, false).is_none() {
            continue; // this kind's frames failed to load (warned at setup)
        }
        let mut facts = commands.spawn((
            Transform::IDENTITY,
            LiquidSurface,
            wet_footprint(lq, &Transform::IDENTITY, LiquidSource::AdtChunk),
            LiquidSoundSource {
                nibble: lq.sound_nibble,
            },
        ));
        if !lq.kind.is_fullbright() {
            facts.insert(FoamPatch); // white surf is a water thing
        }
        // The waterline for the camera sweep under `cameraWaterCollision`; nothing else queries it.
        // Built off-thread like the terrain's own trimesh: a tile carries up to 256 layers.
        if let Some((verts, tris)) = liquid_collider_data(lq) {
            facts.insert(PendingCollider::new(
                build_collider_task(verts, tris),
                Some(liquid_layers()),
                true,
            ));
        }
        entities.push(facts.id());

        let group = DrawGroup::of(n, chunk, lq);
        match groups.iter_mut().find(|(g, ..)| *g == group) {
            Some((.., layers)) => layers.push(lq),
            None => groups.push((group, lane, vec![lq])),
        }
    }
    for (_, lane, layers) in groups {
        let mesh = liquid_bevy_mesh(&layers, None);
        // Computed here, so the exterior cull tests the cell from its first frame. fork: as tall as
        // the surface's displacement reaches (`liquid_cata.wgsl`'s `HEIGHT_MAX_YD`).
        let aabb = mesh.compute_aabb().map(|mut b| {
            b.half_extents.y += HEIGHT_MAX_YD;
            b
        });
        let coarse = meshes.add(mesh);
        let fine = (!lane.kind.is_fullbright())
            .then(|| LiquidFine::new(&layers, coarse.clone()))
            .flatten();
        let mut ec = commands.spawn((
            Mesh3d(coarse),
            Transform::IDENTITY,
            LiquidSurface,
            LiquidLayerBoxes(layers.iter().filter_map(|lq| layer_box(lq)).collect()),
            // Exterior scene: the ADT liquid producer `0x683ab0` is called only from the
            // per-window populate `0x682fa0`, like ADT terrain (`0x683bf0`) and doodads
            // (`0x683700`). The exterior cull is its only `Visibility` writer.
            crate::exterior_cull::ExteriorScene,
            // Water is world geometry to every camera EXCEPT the mirror that reflects it
            // — see `reflect`'s header, and the reference's own reflection view, which
            // never sets the `0x200` water-pass bit.
            bevy::camera::visibility::RenderLayers::layer(super::UNMIRRORED_RENDER_LAYER),
        ));
        if let Some(aabb) = aabb {
            ec.insert(aabb);
        }
        if let Some(fine) = fine {
            ec.insert(fine);
        }
        record_lane(&mut ec, lane);
        entities.push(ec.id());
    }
}

/// fork: how far the Modern (Experimental) water's surface rises or falls at most, yards
/// (`liquid_cata.wgsl`'s `HEIGHT_MAX_YD`, which a test holds to this): the cells' bounds grow by it.
pub(super) const HEIGHT_MAX_YD: f32 = 0.75;

/// fork: the fine ring's split: each wet cell of a surface near the eye is drawn as this many quads a
/// side, so the Modern (Experimental) water's displacement (`liquid_cata.wgsl`'s vertex stage), which
/// the 4.17-yard lattice cannot carry, has a vertex about every yard.
const FINE_SPLIT: usize = 4;
/// A surface takes its fine mesh when its nearest point comes this near the eye, horizontally, and
/// gives it back past [`FINE_RING_OUT`]. The shader's height is gone by 56 yd (`HEIGHT_GONE_YD`), so
/// a surface drawn coarse is always flat there and the two meshes meet without a crack.
const FINE_RING_IN: f32 = 64.0;
const FINE_RING_OUT: f32 = 80.0;
/// The fine meshes built in one frame at most: a camera cut spreads its ring over a few frames.
const FINE_RING_BUILDS: usize = 6;

/// fork: what an ADT water surface needs to build its fine mesh on demand ([`swap_fine_liquids`]):
/// its layers, its coarse mesh, the fine one while it is near, and its wet cells' bounds in Bevy XZ.
#[derive(Component)]
pub(crate) struct LiquidFine {
    layers: Vec<LiquidMesh>,
    coarse: Handle<Mesh>,
    fine: Option<Handle<Mesh>>,
    lo: Vec2,
    hi: Vec2,
}

impl LiquidFine {
    /// `None` for a surface with no real vertex.
    fn new(layers: &[&LiquidMesh], coarse: Handle<Mesh>) -> Option<Self> {
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for p in layers
            .iter()
            .flat_map(|lq| lq.positions.iter())
            .filter(|p| p[2].abs() < 1.0e8)
        {
            let b = wow_to_bevy(*p);
            lo = lo.min(Vec2::new(b.x, b.z));
            hi = hi.max(Vec2::new(b.x, b.z));
        }
        (lo.x <= hi.x).then(|| Self {
            layers: layers.iter().map(|lq| (*lq).clone()).collect(),
            coarse,
            fine: None,
            lo,
            hi,
        })
    }
}

/// fork: **the fine ring** — while the Modern (Experimental) water runs on a modern frame (the tier
/// whose vertex stage displaces the surface), each ADT water surface within [`FINE_RING_IN`] of the
/// eye draws its fine mesh ([`liquid_bevy_mesh_split`]), built on demand, and one past
/// [`FINE_RING_OUT`] its coarse one again, the fine one dropped.
pub(super) fn swap_fine_liquids(
    detail: Res<WaterDetail>,
    eye: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    mut surfaces: Query<(&mut Mesh3d, &mut LiquidFine)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let on = detail.is_experimental() && crate::modern::frame_is_modern();
    let eye = eye.iter().next().map(|t| {
        let p = t.translation();
        Vec2::new(p.x, p.z)
    });
    let mut builds = 0;
    for (mut mesh, mut fine) in &mut surfaces {
        let d = eye.map_or(f32::MAX, |e| (e.clamp(fine.lo, fine.hi) - e).length());
        let want = on && (d < FINE_RING_IN || (fine.fine.is_some() && d < FINE_RING_OUT));
        if want {
            if fine.fine.is_none() {
                if builds >= FINE_RING_BUILDS {
                    continue;
                }
                builds += 1;
                let layers: Vec<&LiquidMesh> = fine.layers.iter().collect();
                fine.fine = Some(meshes.add(liquid_bevy_mesh_split(&layers, FINE_SPLIT)));
            }
            if let Some(handle) = &fine.fine {
                if mesh.0 != *handle {
                    mesh.0 = handle.clone();
                }
            }
        } else {
            if mesh.0 != fine.coarse {
                mesh.0 = fine.coarse.clone();
            }
            if let Some(handle) = fine.fine.take() {
                meshes.remove(&handle);
            }
        }
    }
}

/// fork: [`liquid_bevy_mesh`] with every wet cell split `split` ways a side, bilinear over the cell's
/// four corners — position, UV and the depth `V` alike. An edge two cells share is interpolated from
/// the same two corners by `a·(1 − t) + b·t`, exact at both ends, so the cells' vertices along it agree
/// bit for bit and the displaced surface opens no crack.
fn liquid_bevy_mesh_split(layers: &[&LiquidMesh], split: usize) -> Mesh {
    let split = split.max(1);
    let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for lq in layers {
        let cols = lq.grid[0] as usize;
        let cells_x = cols.saturating_sub(1);
        if cells_x == 0 {
            continue;
        }
        for (cell, _) in lq.wet.iter().enumerate().filter(|(_, wet)| **wet) {
            let (cx, cy) = (cell % cells_x, cell / cells_x);
            let at = |dx: usize, dy: usize| (cy + dy) * cols + cx + dx;
            let (c00, c10, c01, c11) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
            let base = positions.len() as u32;
            for j in 0..=split {
                let t = j as f32 / split as f32;
                for i in 0..=split {
                    let u = i as f32 / split as f32;
                    let bil =
                        |a: f32, b: f32, c: f32, d: f32| lerp(lerp(a, b, u), lerp(c, d, u), t);
                    let (p00, p10, p01, p11) = (
                        lq.positions[c00],
                        lq.positions[c10],
                        lq.positions[c01],
                        lq.positions[c11],
                    );
                    let p = [
                        bil(p00[0], p10[0], p01[0], p11[0]),
                        bil(p00[1], p10[1], p01[1], p11[1]),
                        bil(p00[2], p10[2], p01[2], p11[2]),
                    ];
                    positions.push(wow_to_bevy(p).to_array());
                    let (q00, q10, q01, q11) = (lq.uvs[c00], lq.uvs[c10], lq.uvs[c01], lq.uvs[c11]);
                    uvs.push([
                        bil(q00[0], q10[0], q01[0], q11[0]),
                        bil(q00[1], q10[1], q01[1], q11[1]),
                    ]);
                    let d = bil(
                        lq.depths[c00],
                        lq.depths[c10],
                        lq.depths[c01],
                        lq.depths[c11],
                    );
                    uv1.push([d, 0.0]);
                }
            }
            let row = (split + 1) as u32;
            for j in 0..split as u32 {
                for i in 0..split as u32 {
                    let a = base + j * row + i;
                    let (b, c, d) = (a + 1, a + row, a + row + 1);
                    // Two-sided like every reference liquid pass, so the winding is free.
                    indices.extend_from_slice(&[a, c, b, b, c, d]);
                }
            }
        }
    }
    let n = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; n]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// The collision trimesh input for one [`LiquidMesh`]: the render mesh's own wet-cell triangles,
/// `None` when every cell is dry.
fn liquid_collider_data(lq: &LiquidMesh) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    let tris: Vec<[u32; 3]> = lq
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    if tris.is_empty() {
        return None;
    }
    let verts: Vec<Vec3> = lq.positions.iter().map(|p| wow_to_bevy(*p)).collect();
    Some((verts, tris))
}

/// [`liquid_collider_data`] built inline, for a WMO group's few pools.
fn liquid_collider(lq: &LiquidMesh) -> Option<Collider> {
    liquid_collider_data(lq).map(|(verts, tris)| Collider::trimesh(verts, tris))
}

/// The render mesh for `layers` in their own space (absolute for MCLQ, model-local for WMO
/// liquid), one after another: a flat up normal, the tiling UVs, and the per-vertex depth `V` in
/// UV1.x.
fn liquid_bevy_mesh(layers: &[&LiquidMesh], body_color: Option<[f32; 3]>) -> Mesh {
    let n: usize = layers.iter().map(|lq| lq.positions.len()).sum();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n);
    let mut uv1: Vec<[f32; 2]> = Vec::with_capacity(n);
    let mut indices: Vec<u32> = Vec::with_capacity(layers.iter().map(|lq| lq.indices.len()).sum());
    for lq in layers {
        let base = positions.len() as u32;
        positions.extend(lq.positions.iter().map(|p| wow_to_bevy(*p).to_array()));
        uvs.extend_from_slice(&lq.uvs);
        uv1.extend(lq.depths.iter().map(|&d| [d, 0.0]));
        indices.extend(lq.indices.iter().map(|i| base + i));
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    // WoW up (0, 0, 1) is Bevy up (0, 1, 0).
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; n]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    // An interior pool's `MOMT.diffColor` rides the vertex colour, where the reference's interior
    // vertex carries it, keeping one material per lane; other lanes take the shader's white.
    if let Some([red, green, blue]) = body_color {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[red, green, blue, 1.0]; n]);
    }
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Spawn a WMO group's liquid surfaces at the building's placement `transform`, pushed onto
/// `entities` to despawn with the placement. `interior` is the group's `MOGI & 0x48 == 0` class,
/// which picks the renderer and the fog block; `pool` scopes each surface to its room and floor.
pub(crate) fn spawn_wmo_liquids<'a>(
    commands: &mut Commands,
    liquids: impl Iterator<Item = &'a LiquidMesh>,
    liquid_assets: Option<&LiquidAssets>,
    meshes: &mut Assets<Mesh>,
    transform: Transform,
    interior: bool,
    pool: WmoPool,
    // The owning root's MOMT `diffColor` table: MOMT lives in the root, not the group file.
    material_diff_color: &[[f32; 3]],
    entities: &mut Vec<Entity>,
) {
    let Some(liquid) = liquid_assets else {
        return;
    };
    let path = LiquidPath::wmo(interior);
    for lq in liquids {
        // The one path that can scroll, decided by the nibble, not the kind (`scrolls`).
        let scroll = scrolls(lq.sound_nibble);
        // The interior arm's body colour via the pool's MLIQ `materialId`; none when fullbright.
        let body_color = (path.takes_material_body_color() && !lq.kind.is_fullbright())
            .then(|| {
                lq.material_id
                    .and_then(|id| material_diff_color.get(usize::from(id)))
                    .copied()
            })
            .flatten();
        let lane = LiquidLane {
            kind: lq.kind,
            path,
            scroll,
        };
        if liquid.material(lq.kind, path, scroll).is_none() {
            continue; // this kind's frames failed to load (warned at setup)
        }
        if scroll {
            debug!(
                "liquid: {:?} nibble {} takes the scroll lane",
                lq.kind, lq.sound_nibble
            );
        }
        let surface = {
            let mut ec = commands.spawn((
                Mesh3d(meshes.add(liquid_bevy_mesh(&[lq], body_color))),
                transform,
                LiquidSurface,
                // Bit 30 is the interior-fog lane (`liquid.wgsl`'s `room_fog`), written each frame
                // by the `Visibility` authority off the pool's room; spawned clear.
                bevy::mesh::MeshTag(0),
                // Every kind, the lava and slime hum too.
                LiquidSoundSource {
                    nibble: lq.sound_nibble,
                },
                // Water is world geometry to every camera EXCEPT the mirror that reflects it —
                // see `reflect`'s header, and the reference's own reflection view, which never
                // sets the `0x200` water-pass bit.
                bevy::camera::visibility::RenderLayers::layer(super::UNMIRRORED_RENDER_LAYER),
            ));
            record_lane(&mut ec, lane);
            ec.id()
        };
        // Every kind carries the swim grid, so lava and slime swim; their damage is not modelled.
        commands.entity(surface).insert(wet_footprint(
            lq,
            &transform,
            LiquidSource::WmoGroup(pool),
        ));
        if !lq.kind.is_fullbright() {
            commands.entity(surface).insert(FoamPatch);
        }
        // The camera's waterline, model-local under the entity's placement `transform`.
        if let Some(collider) = liquid_collider(lq) {
            commands
                .entity(surface)
                .insert((collider, RigidBody::Static, liquid_layers()));
        }
        entities.push(surface);
    }
}

/// Each kind's frames, `XTextures\<dir>\<stem>.<1..=count>.blp`, as `(kind, dir, stem, count)`.
const FRAME_SETS: &[(LiquidKind, &str, &str, u32)] = &[
    (LiquidKind::Still, "river", "lake_a", 30),
    (LiquidKind::Rapids, "river", "fast_a", 16),
    (LiquidKind::Ocean, "ocean", "ocean_h", 30),
    // Fullbright: the sheet is the opaque, unlit, fogged body (`0x6b68f0`). Magma comes from both
    // the WMO pools and the ADT magma queue; the reference has no ADT queue for slime.
    (LiquidKind::Magma, "lava", "lava", 30),
    (LiquidKind::Slime, "slime", "slime", 30),
];

pub(super) fn setup_liquid(
    mut commands: Commands,
    config: Option<Res<RenderConfig>>,
    world_assets: Option<ResMut<WorldAssets>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
    mut cata_materials: ResMut<Assets<LiquidCataMaterial>>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
) {
    let (Some(_config), Some(mut world_assets)) = (config, world_assets) else {
        return; // no client data → no terrain, so no water either
    };
    // The cata renderer's two non-light inputs, both created here so that a material can never be
    // built before its bindings exist:
    //
    // * the synthesised wave normal map (`art`), which is a compile-time blob and cannot fail;
    // * the three screen-space targets (`screen`), allocated at a provisional size because the
    //   window may not have reported one yet at `Startup`. `screen::resize_targets` corrects that on
    //   the first `Update` with a real size, republishing the handles and re-pointing the materials
    //   — the same path a genuine resize takes, so the first frame is not a special case anybody
    //   has to maintain.
    let bump = images.add(art::bump_image());
    let foam = images.add(art::foam_image());
    let noise_vol = images.add(art::noise_volume_image());
    let targets = LiquidScreenTargets::allocate(&mut images, UVec2::new(1280, 720));
    let ripples = ripple::setup(&mut images);
    // The ripple field's placement buffer, on the same terms as the shared light: created here so
    // the material can clone it, written in place by the render world every frame.
    let ripple_buf = ripple::new_ripple_buffer(&render_device);

    // Light, fog and the water swatches come off the shared global-light buffer, as for terrain.
    let mut assets = LiquidAssets::default();
    for &(kind, dir, stem, count) in FRAME_SETS {
        let Some((frames, frame_count)) =
            load_frame_array(&mut world_assets, &mut images, kind, dir, stem, count)
        else {
            warn!("liquid: no frames for {stem} — {kind:?} water will not render");
            continue;
        };
        // Blend per kind, over the device defaults (`0x593bf0`) every reference setter pushes and
        // pops against. Water and ocean: EGxBlend 2 (`SRC_ALPHA / INV_SRC_ALPHA`), depth write off
        // (both under the fancy-water CVar `[0xc9a324]`, which is not modelled). Magma and slime:
        // blend stays 0 (disabled) with depth test and write on (ids `0x10`/`0x12`), as the ADT
        // lava pass sets (`0x6855e2`) and the WMO arm leaves it, so they are opaque. All four
        // liquid passes turn culling off at entry (`0x59d7d8`); `glFrontFace` is never imported.
        let alpha_mode = if kind.is_fullbright() {
            AlphaMode::Opaque
        } else {
            AlphaMode::Blend
        };
        // One material per renderer and scroll the kind can take, all on one frame array; only
        // magma and slime (nibbles 6 and 7) can scroll.
        let scroll_lanes: &[bool] = if kind.is_fullbright() {
            &[false, true]
        } else {
            &[false]
        };
        for (path, &scroll) in [
            LiquidPath::Adt,
            LiquidPath::WmoExterior,
            LiquidPath::WmoInterior,
        ]
        .into_iter()
        .flat_map(|p| scroll_lanes.iter().map(move |s| (p, s)))
        {
            let material = materials.add(ExtendedMaterial {
                base: StandardMaterial {
                    // The shader does the WoW lighting.
                    unlit: true,
                    alpha_mode,
                    cull_mode: None,
                    double_sided: true,
                    // Water takes the water-pass slot of the reference's `0x483460` interleave
                    // (`sky_order::WATER_BIAS`); the opaque kinds take no rung.
                    depth_bias: if kind.is_fullbright() {
                        0.0
                    } else {
                        crate::sky_order::WATER_BIAS
                    },
                    ..default()
                },
                extension: LiquidExt {
                    frames: frames.clone(),
                    // x = fullbright (the sheet as body, still fogged); y = the ocean swatch;
                    // z = the interior fog block; w = water's sun-sheen exponent.
                    kind: Vec4::new(
                        if kind.is_fullbright() { 1.0 } else { 0.0 },
                        if kind == LiquidKind::Ocean { 1.0 } else { 0.0 },
                        if path.interior_fog() { 1.0 } else { 0.0 },
                        WATER_SHININESS,
                    ),
                    // Which of the reference's three liquid renderers `liquid.wgsl` runs.
                    path: Vec4::new(path.shader_id(), 0.0, 0.0, 0.0),
                    // x reserved; y = frame count; z = scroll; w = clock enable (0 on a
                    // deterministic run: frame 0, scroll 0).
                    anim: Vec4::new(
                        0.0,
                        frame_count as f32,
                        if scroll { 1.0 } else { 0.0 },
                        if crate::dev_state::deterministic_run() {
                            0.0
                        } else {
                            1.0
                        },
                    ),
                    light_buf: world_assets.shared_light.clone(),
                    modern_cascades: benilla_assets::materials::MODERN_SHADOW_CASCADES, // fork: modern lighting (FORK.md, P4)
                    modern_fog: benilla_assets::materials::MODERN_FOG_VOLUME,
                },
            });
            // The same surface on the Cataclysm renderer. Everything that is a *property of the
            // surface* — kind lanes, renderer id, animation, blend, cull, the sort rung — is
            // identical by construction: the toggle changes how water is SHADED, not where it sits
            // in the frame or what it is. Only the shading inputs are new.
            let cata = cata_materials.add(ExtendedMaterial {
                base: StandardMaterial {
                    unlit: true,
                    alpha_mode,
                    cull_mode: None,
                    double_sided: true,
                    depth_bias: if kind.is_fullbright() {
                        0.0
                    } else {
                        crate::sky_order::WATER_BIAS
                    },
                    ..default()
                },
                extension: LiquidCataExt {
                    frames: frames.clone(),
                    bump: bump.clone(),
                    foam: foam.clone(),
                    noise_vol: noise_vol.clone(),
                    scene_color: targets.color.clone(),
                    scene_blur: targets.blur.clone(),
                    scene_depth: targets.depth.clone(),
                    reflect_map: targets.reflect_blur.clone(),
                    ripple_map: ripples.map.clone(),
                    kind: Vec4::new(
                        if kind.is_fullbright() { 1.0 } else { 0.0 },
                        if kind == LiquidKind::Ocean { 1.0 } else { 0.0 },
                        if path.interior_fog() { 1.0 } else { 0.0 },
                        WATER_SHININESS,
                    ),
                    // `y` starts above-water; `sync_below_water` flips it with the eye. `z`
                    // starts with the mirror DOWN — `reflect::sync_reflection_flag` raises it once
                    // that camera has a pose, so no frame ever reads an uninitialised target.
                    path: Vec4::new(path.shader_id(), 0.0, 0.0, 0.0),
                    anim: Vec4::new(
                        0.0,
                        frame_count as f32,
                        if scroll { 1.0 } else { 0.0 },
                        if crate::dev_state::deterministic_run() {
                            0.0
                        } else {
                            1.0
                        },
                    ),
                    bump_params: cata_bump_params(kind),
                    shade_params: cata_shade_params(kind),
                    foam_params: cata_foam_params(kind),
                    magma_scales: cata_magma_scales(kind),
                    magma_params: cata_magma_params(kind),
                    magma_color: cata_magma_color(kind),
                    optics: cata_optics(kind),
                    look_a: cata_water_look(kind)[0],
                    look_b: cata_water_look(kind)[1],
                    look_c: cata_water_look(kind)[2],
                    look_d: cata_water_look(kind)[3],
                    look_e: cata_water_look(kind)[4],
                    look_f: cata_water_look(kind)[5],
                    light_buf: world_assets.shared_light.clone(),
                    modern_cascades: benilla_assets::materials::MODERN_SHADOW_CASCADES, // fork: modern lighting (FORK.md, P4)
                    modern_fog: benilla_assets::materials::MODERN_FOG_VOLUME,
                    modern_shadow: benilla_assets::materials::MODERN_SHADOW_MAP,
                    ripple_buf: ripple_buf.0.clone(),
                },
            });
            assets.materials.insert(
                LiquidKey { kind, path, scroll },
                LiquidEntry { material, cata },
            );
        }
    }
    // Frame sets, not materials.
    info!(
        "liquid: loaded {} water frame set(s)",
        FRAME_SETS
            .iter()
            .filter(|(k, ..)| assets.material(*k, LiquidPath::Adt, false).is_some())
            .count()
    );
    commands.insert_resource(assets);
    commands.insert_resource(targets);
    commands.insert_resource(ripples);
    commands.insert_resource(ripple_buf);
}

/// The bump lane for a kind: the two tiling scales, the scroll rate, and the bump strength.
///
/// The reference reads these off `LiquidType.dbc`'s `Float[18]` block —
/// `CMaterialSettings::GetFloat(mat, 3)` and `(mat, 4)` for the scales (each multiplied by
/// `Liquid::WorldCoordScale`), `(mat, 7) × 0.35` for the strength. 1.12.1's `LiquidType.dbc` has no
/// such columns, so the table is here instead, seeded from the constants the shipped programs
/// carry and tuned per kind. Values are in repeats per yard: water wants a long swell, rapids a
/// short chop.
fn cata_bump_params(kind: LiquidKind) -> Vec4 {
    match kind {
        // Open ocean: the largest features and the slowest drift.
        LiquidKind::Ocean => Vec4::new(0.018, 0.055, 0.05, 0.9),
        // A river running: tighter than a lake and noticeably quicker.
        LiquidKind::Rapids => Vec4::new(0.075, 0.21, 0.30, 1.25),
        // Lakes and rivers at rest.
        _ => Vec4::new(0.035, 0.11, 0.09, 1.0),
    }
}

/// `ProcMagma`'s tiling lane: the two noise-volume scales and the first two layer scales, all in
/// repeats per yard.
///
/// The two volume scales are deliberately close but not equal — the whole point of the pair is that
/// they disagree, and two taps at the same scale would blend between two views of the same
/// structure. The layer scales spread wider, so the composite has a large crust, a mid crack
/// pattern and a fine shimmer rather than three copies of one thing.
fn cata_magma_scales(kind: LiquidKind) -> Vec4 {
    match kind {
        // Slime is a thicker, slower body: everything larger and lazier than lava.
        LiquidKind::Slime => Vec4::new(0.020, 0.029, 0.050, 0.11),
        _ => Vec4::new(0.028, 0.041, 0.075, 0.16),
    }
}

/// `ProcMagma`'s motion lane: third layer scale, wobble frequency, clock rate, tap drift.
fn cata_magma_params(kind: LiquidKind) -> Vec4 {
    match kind {
        LiquidKind::Slime => Vec4::new(0.24, 0.09, 0.020, 0.004),
        // Lava crawls, and the clock rate is what that is: a full trip round the volume's third
        // axis every ~28 s. Fast enough to see moving, slow enough not to boil.
        _ => Vec4::new(0.33, 0.13, 0.035, 0.007),
    }
}

/// `ProcMagma`'s base colour — what the surface is where every mask reads zero.
///
/// The cold crust, not a tint: it shows through only where all three layers are masked out, so it
/// wants to be darker than anything in the sheet rather than an average of it.
fn cata_magma_color(kind: LiquidKind) -> Vec4 {
    match kind {
        LiquidKind::Slime => Vec4::new(0.04, 0.09, 0.03, 1.0),
        _ => Vec4::new(0.10, 0.028, 0.012, 1.0),
    }
}

/// fork: modern lighting — the water's optics on the Modern (Experimental) tier: absorption per yard
/// of water crossed (r, g, b). `w` starts at 0 (plain Modern); [`sync_water_tier`] raises it on
/// `waterDetail 3`.
///
/// Red goes first, which cools a sandy bed as the water deepens and leaves the channel the zone's
/// deep colour. The values are read off the reference shots rather than off a textbook: a river bank
/// there keeps ~0.94 of its red under the first yard of water, and the bed still shows mid-channel,
/// so a river is clear — most of the blue in open water is the sky, through the Fresnel term. Magma
/// and slime are opaque and never reach the modern water lane.
fn cata_optics(kind: LiquidKind) -> Vec4 {
    match kind {
        LiquidKind::Ocean => Vec4::new(0.08, 0.04, 0.032, 0.0),
        LiquidKind::Rapids => Vec4::new(0.09, 0.045, 0.036, 0.0),
        LiquidKind::Magma | LiquidKind::Slime => Vec4::ZERO,
        _ => Vec4::new(0.07, 0.035, 0.03, 0.0),
    }
}

/// fork: modern lighting — the Modern (Experimental) tier's look, the target being WoW Forever's
/// water: a clear bed that takes the water's hue from the first inches, the bank's shadows carried
/// across it, a calm surface and a sky in it that never turns to chrome. `[look_a, look_b, look_c,
/// look_d]` on the material:
///
/// * `look_a` = (bed filter, its onset depth in yd, head-on reflectance, reflection pulled to the body);
/// * `look_b` = (wind swell a river keeps, swell over no depth, refraction bend per yd, highlight
///   roughness `α`);
/// * `look_c` = (the far water's reflectance ceiling, the gain on the facets' spread the Fresnel
///   averages over, share of the reference
///   combine blended over the clear surface, the wind ripples' RMS slope);
/// * `look_d` = (the sun/moon disk's angular radius in radians, the share of the kind's own swell
///   on the highlight's normal, the wind ripples' RMS slope on it — the sea's alone, a river or lake
///   having no ripple octaves — the highlight's gain).
///
/// The Fresnel is water's own over the surface's roughness (the benilla-watershader fork's lively
/// Fresnel), raised to a floor that keeps the sky in a river looked down on, with the far water's mean
/// rescaled to a ceiling that stops the water toward the horizon becoming a solid sheet while each
/// wave keeps its own contrast. Rapids keep more of their chop in the shallows (white water is the point
/// of them) and the sea keeps its whole wind sea whatever `look_b.x` says. The ripples are a light
/// wind's: a real surface's RMS slope runs ~0.1-0.2 at a few yards a second of wind. Magma and slime
/// never reach the lane.
///
/// The highlight is Forever's PBR water: a lobe narrower than the light's disk (`α` below the
/// radius), so each facet that sees the disk reflects all of it and the path breaks into flat,
/// sharp-edged islands, spread wide by a normal that keeps the swell the shallows' calm gives up —
/// each kind's own octaves, never the sea's on a lake. The disk is a stylised ~2.6°, five times the real sun's, which is what makes the islands
/// big enough to read; rapids take a broader lobe, white water being rough.
///
/// * `look_e` = (turbidity per yd, wind speed in m/s for the Cox–Munk roughness floor, caustic gain,
///   the roughness `α` at which the surface counts as wholly unresolved);
/// * `look_f` = (the reflection's contrast lost near, and far, the displacement's gain, reserved).
///
/// The roughness the far water takes is the slope its faded octaves no longer draw (the
/// benilla-watershader fork's energy-conserving filter), floored by a light wind's sub-pixel ripples:
/// 1.5 m/s on a river or lake keeps the near highlight's islands crisp (`α` stays under the disk), the
/// sea's 3 and the rapids' 4 roughen theirs. Turbidity is small on clear water (the bed must show
/// mid-channel) and larger on rapids, whose bubbles scatter; the caustics follow it the other way.
fn cata_water_look(kind: LiquidKind) -> [Vec4; 6] {
    let fresnel = Vec4::new(0.35, 1.0, 0.5, 0.2);
    match kind {
        LiquidKind::Ocean => [
            Vec4::new(0.35, 1.0, 0.08, 0.3),
            Vec4::new(1.0, 0.5, 0.25, 0.04),
            fresnel,
            Vec4::new(0.045, 1.0, 0.2, 1.0),
            Vec4::new(0.02, 3.0, 0.55, 0.25),
            Vec4::new(0.1, 0.5, 1.0, 0.0),
        ],
        LiquidKind::Rapids => [
            Vec4::new(0.5, 0.6, 0.08, 0.3),
            Vec4::new(0.35, 0.6, 0.25, 0.07),
            fresnel,
            Vec4::new(0.045, 1.0, 0.0, 1.0),
            Vec4::new(0.08, 4.0, 0.28, 0.3),
            Vec4::new(0.1, 0.5, 0.6, 0.0),
        ],
        LiquidKind::Magma | LiquidKind::Slime => [Vec4::ZERO; 6],
        _ => [
            Vec4::new(0.5, 0.6, 0.08, 0.3),
            Vec4::new(0.35, 0.25, 0.25, 0.03),
            fresnel,
            Vec4::new(0.045, 1.0, 0.0, 1.0),
            Vec4::new(0.03, 1.5, 0.6, 0.25),
            Vec4::new(0.1, 0.5, 0.6, 0.0),
        ],
    }
}

/// The shoreline-foam lane: tiling, depth scale, strength, and the second tap's normal drag.
///
/// The **tiling** is the sheet's frequency in repeats per yard. The reference's own number is
/// `228 ×` a liquid coordinate whose world scale is authored data this client cannot read, so this
/// is a stand-in — but its ORDER is not free: at the 0.6 it carried, one foam cell spanned nearly
/// two yards and the surf read as a few big soft lumps rather than as broken water. Against the
/// reference client's shoreline it wants to be a few times finer than that.
///
/// The **depth scale** is the one number with a physical meaning: how deep the water gets before the
/// surf stops, in yards. Sub-yard on a river, because a shoreline band is a band and not a shallows
/// — the first cut used two yards and painted every sandbar in Elwynn. The ocean gets a little more,
/// a beach shelving more slowly than a bank.
///
/// The **strength** is smaller than it looks like it should be, and that is the distance fade being
/// the reference's own way round (`1.1 − d/150`, `liquid_cata.wgsl`): the surf you are standing in
/// gets the full term rather than the fifth the bump blend used to give it, so the authored number
/// has to come down by about the same factor to land on the same shoreline.
///
/// The **drag** is the reference's `0.2` outright — the second tap's `+N.yx · 0.2`; the first tap's
/// drag is a full-strength `−N.xy` and takes no parameter.
///
/// Magma and slime never reach this — the fullbright branch returns before the foam term — but they
/// are given a zero strength anyway rather than a plausible number, so that a future port of
/// `ProcMagma` starts from "off" instead of inheriting water's surf.
fn cata_foam_params(kind: LiquidKind) -> Vec4 {
    match kind {
        LiquidKind::Ocean => Vec4::new(0.40, 0.58, 0.082, 0.2),
        // Rapids: tighter and brighter, and reaching less far out — white water is the point, and
        // a river is narrow enough that a wide band would simply fill it.
        LiquidKind::Rapids => Vec4::new(0.75, 0.30, 0.128, 0.2),
        LiquidKind::Magma | LiquidKind::Slime => Vec4::ZERO,
        _ => Vec4::new(0.52, 0.40, 0.098, 0.2),
    }
}

/// The shading lane: view perturb (`c23.y`), specular power (`c4.w`), specular scale (`c27.w`) and
/// sun-glare scale (`c27.z`).
///
/// The specular power is [`CATA_WATER_SHININESS`], **not** the 1.12 path's [`WATER_SHININESS`].
/// The exponent is not a property of the surface after all: `Liquid::SetShaderLighting` writes 6.0
/// into the vertex block's `c4.w` and 50.0 into the pixel block's, out of the same light record, and
/// `ProcWaterAbove` reads the pixel one. Carrying 6 here gave cata water the fixed-function pipe's
/// broad wash under a program written for a tight track.
fn cata_shade_params(kind: LiquidKind) -> Vec4 {
    let glare = match kind {
        // Ocean carries the strongest sun track; it is the one surface with an unbroken horizon.
        LiquidKind::Ocean => 0.35,
        LiquidKind::Rapids => 0.22,
        _ => 0.25,
    };
    Vec4::new(0.55, CATA_WATER_SHININESS, 0.6, glare)
}

/// Decode frames `1..=count` with their authored mips into one repeating, anisotropic array,
/// stopping at the first missing, non-square or mis-sized frame; returns it and the count loaded.
fn load_frame_array(
    world_assets: &mut WorldAssets,
    images: &mut Assets<Image>,
    kind: LiquidKind,
    dir: &str,
    stem: &str,
    count: u32,
) -> Option<(Handle<Image>, u32)> {
    let mut frames: Vec<BlpMipChain> = Vec::new();
    let mut size = 0u32;
    for i in 1..=count {
        let path = format!("XTextures\\{dir}\\{stem}.{i}.blp");
        let Ok(chain) = read_texture_mip_chain(&mut world_assets.chain.lock_recover(), &path)
        else {
            break;
        };
        if chain.width != chain.height {
            break; // water frames are square; bail rather than build a ragged array
        }
        if size == 0 {
            size = chain.width;
        } else if chain.width != size {
            break; // a frame at a different resolution can't share the array
        }
        frames.push(chain);
    }
    if frames.is_empty() {
        return None;
    }
    let loaded = frames.len() as u32;
    // Deviation: water frames have their per-frame mean flattened (`flatten_frame_dc`), because
    // mipping turns the shipped frames' drifting means into a flicker on distant water; magma and
    // slime keep theirs, the body's intended pulse.
    let normalize_dc = !kind.is_fullbright();
    Some((images.add(liquid_frame_array(frames, normalize_dc)), loaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_nibbles_six_and_seven_scroll() {
        // Blackrock's lava and Stratholme's slime.
        assert!(scrolls(6), "magma variant 6 takes the animated matrix");
        assert!(scrolls(7), "slime variant 7 takes the animated matrix");

        // Same kernel, same sheet, still static: the Great Forge (2) and Undercity's slime (3).
        assert!(!scrolls(2), "the Great Forge's lava is static");
        assert!(!scrolls(3), "nibble-3 slime is static");

        // 4 passes a bare `nibble & 0xc == 4` test but is a river variant, never the kernel's.
        assert!(!scrolls(4), "nibble 4 is lake_a, not a hazard liquid");

        // Everything else in the table, including the hole nibble.
        for n in [0u8, 1, 5, 8, 0xf] {
            assert!(!scrolls(n), "nibble {n} must not scroll");
        }
    }

    /// Only the fullbright kinds have nibbles that scroll, so only they get a scroll lane.
    #[test]
    fn only_fullbright_kinds_can_take_a_scroll_lane() {
        for &(kind, ..) in FRAME_SETS {
            let wants_lane = kind.is_fullbright();
            assert_eq!(
                wants_lane,
                [2u8, 3, 6, 7]
                    .iter()
                    .any(|&n| scrolls(n) && LiquidKind::from_nibble(n) == Some(kind)),
                "{kind:?}: scroll lane and the nibbles that can ask for one must agree"
            );
        }
    }

    /// The smallest MCLQ sheet `spawn_liquids` builds: 2×2 vertices at `at`.
    fn one_sheet(at: [f32; 3]) -> LiquidMesh {
        let [x, y, z] = at;
        LiquidMesh {
            grid: [2, 2],
            wet: vec![true],
            shared: vec![false],
            positions: vec![
                [x, y, z],
                [x + 8.0, y, z],
                [x, y + 8.0, z],
                [x + 8.0, y + 8.0, z],
            ],
            uvs: vec![[0.0, 0.0]; 4],
            depths: vec![1.0; 4],
            indices: vec![0, 1, 2, 1, 3, 2],
            sound_nibble: 0,
            material_id: None,
            kind: LiquidKind::Still,
        }
    }

    /// fork: the displacement's bound is one number in two places: the shader clamps the surface to
    /// it and the cells' bounds grow by it ([`HEIGHT_MAX_YD`]). A shader past the bound draws a wave
    /// the exterior cull can drop.
    #[test]
    fn the_cells_bounds_reach_as_high_as_the_shader_displaces() {
        let cata = include_str!("../../../benilla-assets/src/shaders/liquid_cata.wgsl");
        let want = format!("const HEIGHT_MAX_YD: f32 = {HEIGHT_MAX_YD};");
        assert!(
            cata.contains(&want),
            "liquid_cata.wgsl must declare `{want}`"
        );
    }

    /// fork: the fine ring's mesh splits each wet cell `split²` ways with no crack: two cells that
    /// share an edge put their vertices along it at the same positions bit for bit, the corners are
    /// the lattice's own, and a dry cell gets nothing.
    #[test]
    fn the_fine_mesh_splits_wet_cells_and_shares_their_edges_exactly() {
        use bevy::mesh::VertexAttributeValues;
        // A 3×2 grid of vertices (two cells side by side, the left one wet, then both), sloped so
        // interpolation is exercised on every axis.
        let mut lq = one_sheet([0.0, 0.0, 0.0]);
        lq.grid = [3, 2];
        lq.positions = vec![
            [0.0, 0.0, 1.0],
            [4.1666, 0.0, 1.3],
            [8.3333, 0.0, 0.7],
            [0.0, 4.1666, 1.1],
            [4.1666, 4.1666, 0.9],
            [8.3333, 4.1666, 1.7],
        ];
        lq.uvs = vec![[0.0, 0.0]; 6];
        lq.depths = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        lq.wet = vec![true, false];
        let split = 4;
        let one = super::liquid_bevy_mesh_split(&[&lq], split);
        assert_eq!(
            one.count_vertices(),
            (split + 1) * (split + 1),
            "one wet cell, the dry one skipped"
        );

        lq.wet = vec![true, true];
        let two = super::liquid_bevy_mesh_split(&[&lq], split);
        let Some(VertexAttributeValues::Float32x3(p)) = two.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let row = split + 1;
        let (left, right) = (&p[..row * row], &p[row * row..]);
        for j in 0..row {
            assert_eq!(
                left[j * row + split],
                right[j * row],
                "the shared edge's vertex {j} differs between the two cells"
            );
        }
        // The corners are the lattice's own vertices, carried to Bevy space untouched.
        assert_eq!(left[0], wow_to_bevy(lq.positions[0]).to_array());
        assert_eq!(
            right[row * row - 1],
            wow_to_bevy(lq.positions[5]).to_array()
        );
    }

    /// Only the ADT still-water material, without which `spawn_liquids` spawns nothing.
    fn adt_assets() -> LiquidAssets {
        LiquidAssets {
            materials: HashMap::from([(
                LiquidKey {
                    kind: LiquidKind::Still,
                    path: LiquidPath::Adt,
                    scroll: false,
                },
                LiquidEntry {
                    material: Handle::default(),
                    cata: Handle::default(),
                },
            )]),
        }
    }

    /// `spawn_liquids` over `sheets`, each with its MCIN chunk index: the app and what it spawned.
    fn spawn_adt(sheets: Vec<(usize, LiquidMesh)>) -> (App, Vec<Entity>) {
        use bevy::ecs::system::RunSystemOnce;
        // The waterline collider builds on the async pool, which no plugin here starts.
        bevy::tasks::AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
        let mut app = App::new();
        app.add_plugins(bevy::asset::AssetPlugin::default())
            .init_asset::<Mesh>()
            .init_asset::<LiquidMaterial>();
        let assets = adt_assets();
        let spawned = app
            .world_mut()
            .run_system_once(
                move |mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>| {
                    let mut ents = Vec::new();
                    spawn_liquids(
                        &mut commands,
                        sheets.iter().map(|(chunk, lq)| (*chunk, lq)),
                        Some(&assets),
                        &mut meshes,
                        &mut ents,
                    );
                    ents
                },
            )
            .expect("the spawn system ran");
        (app, spawned)
    }

    /// The spawned entities that draw, with the index count of each one's mesh.
    fn draws(app: &App, spawned: &[Entity]) -> Vec<(Entity, usize)> {
        let meshes = app.world().resource::<Assets<Mesh>>();
        spawned
            .iter()
            .filter_map(|&e| {
                let mesh = app.world().entity(e).get::<Mesh3d>()?;
                Some((e, meshes.get(&mesh.0)?.indices()?.len()))
            })
            .collect()
    }

    /// Flat layers of one kind at one height in one 4×4-chunk cell are one draw; another height,
    /// another cell or a slope is its own. Every layer keeps its own facts entity.
    #[test]
    fn a_cells_flat_layers_at_one_height_draw_as_one() {
        let mut sloped = one_sheet([300.0, 100.0, 5.0]);
        sloped.positions[3][2] = 6.0;
        let (app, spawned) = spawn_adt(vec![
            // Chunks 0, 1 and 17 share cell (0, 0); chunk 4 opens cell (0, 1).
            (0, one_sheet([100.0, 100.0, 5.0])),
            (1, one_sheet([100.0, 108.0, 5.0])),
            (17, one_sheet([108.0, 108.0, 5.0])),
            (1, one_sheet([100.0, 108.0, 9.0])),
            (4, one_sheet([100.0, 132.0, 5.0])),
            (0, sloped),
        ]);
        let mut counts: Vec<usize> = draws(&app, &spawned).iter().map(|d| d.1).collect();
        counts.sort_unstable();
        assert_eq!(
            counts,
            [6, 6, 6, 18],
            "three layers at one height share a mesh; the higher layer, the next cell and the \
             slope each draw alone"
        );
        let facts = spawned
            .iter()
            .filter(|&&e| {
                app.world()
                    .entity(e)
                    .contains::<crate::liquid::WaterChunkInfo>()
            })
            .count();
        assert_eq!(facts, 6, "every layer keeps its swim grid");
        let boxes: usize = spawned
            .iter()
            .filter_map(|&e| app.world().entity(e).get::<LiquidLayerBoxes>())
            .map(|b| b.0.len())
            .sum();
        assert_eq!(boxes, 6, "…and its own box for the mirror's plane");
    }

    /// An ADT surface carries `ExteriorScene` and none of the components `UnownedSceneFilter`
    /// excludes, so the exterior cull reaches it.
    #[test]
    fn an_adt_liquid_surface_is_tagged_exterior_scene() {
        let (app, spawned) = spawn_adt(vec![(0, one_sheet([100.0, 100.0, 5.0]))]);
        let drawn = draws(&app, &spawned);
        assert_eq!(drawn.len(), 1, "one sheet in, one surface drawn");
        let e = app.world().entity(drawn[0].0);
        assert!(
            e.contains::<bevy::camera::primitives::Aabb>(),
            "the cull tests the surface's box from its first frame"
        );
        assert!(
            e.contains::<crate::exterior_cull::ExteriorScene>(),
            "an open-world liquid surface is exterior scene — untagged, the window cull never \
             queries it and the lake draws through a sealed ceiling"
        );
        assert!(
            e.contains::<LiquidSurface>(),
            "…and is still a liquid surface"
        );
        // The three exclusions in `UnownedSceneFilter`.
        assert!(!e.contains::<crate::model_render::ModelPart>());
        assert!(!e.contains::<crate::wmo_portal::WmoGroupVis>());
        assert!(!e.contains::<crate::world_unit::WorldUnit>());
    }
}

/// The scroll and fog-block rules against the real client files.
#[cfg(test)]
mod real_data {
    use super::LiquidKind;
    use benilla_formats::{parse_wmo_root, wmo_group_liquid_mesh};
    use std::collections::HashMap;

    /// The scroll gate over the shipped pools, where both behaviours share buildings: Ironforge has
    /// 9 static nibble-2 and 2 scrolling nibble-6 magma groups, Blackrock is nibble 6 throughout,
    /// Undercity's 38 slime canals are nibble 3 and still, and Stratholme's are nibble 7 and flow.
    #[test]
    fn only_the_nibble_six_and_seven_pools_scroll_in_the_shipped_data() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open chain");
        // Every liquid-bearing group of a WMO, as `(nibble, kind)` → count.
        let mut census = |root_path: &str| -> HashMap<(u8, LiquidKind), usize> {
            let bytes = chain.read_file(root_path).expect("root readable");
            let root = parse_wmo_root(&bytes).expect("parse root");
            let stem = root_path
                .strip_suffix(".wmo")
                .unwrap_or(root_path)
                .to_string();
            let mut tally = HashMap::new();
            for gi in 0..root.group_count() as usize {
                let Ok(gb) = chain.read_file(&format!("{stem}_{gi:03}.wmo")) else {
                    continue;
                };
                if let Some(m) = wmo_group_liquid_mesh(&gb) {
                    *tally.entry((m.sound_nibble, m.kind)).or_default() += 1;
                }
            }
            tally
        };

        let ironforge = census("world\\wmo\\khazmodan\\cities\\ironforge\\ironforge.wmo");
        assert_eq!(
            ironforge,
            HashMap::from([
                ((2, LiquidKind::Magma), 9),
                ((4, LiquidKind::Still), 2),
                ((6, LiquidKind::Magma), 2),
            ]),
            "Ironforge's liquid census moved",
        );

        let blackrock = census("world\\wmo\\dungeon\\az_blackrock\\blackrock.wmo");
        assert_eq!(
            blackrock,
            HashMap::from([((6, LiquidKind::Magma), 2)]),
            "Blackrock Mountain's lava — the `.go xyz -7531.21 -1123.64 172.58` pin — is all nibble 6",
        );

        let undercity = census("world\\wmo\\lorderon\\undercity\\undercity.wmo");
        assert_eq!(
            undercity,
            HashMap::from([((3, LiquidKind::Slime), 38)]),
            "Undercity's slime is nibble 3 throughout",
        );

        let stratholme = census("world\\wmo\\dungeon\\ld_stratholme\\stratholme.wmo");
        assert_eq!(
            stratholme,
            HashMap::from([((7, LiquidKind::Slime), 3)]),
            "Stratholme's slime is nibble 7 throughout",
        );

        // The gate's verdict over all of it.
        assert_eq!(
            ironforge
                .iter()
                .filter(|((n, _), _)| super::scrolls(*n))
                .map(|(_, c)| c)
                .sum::<usize>(),
            2,
            "exactly 2 of Ironforge's 11 magma pools scroll",
        );
        for (label, tally) in [("blackrock", &blackrock), ("stratholme", &stratholme)] {
            assert!(
                tally.keys().all(|(n, _)| super::scrolls(*n)),
                "{label}: every pool scrolls",
            );
        }
        assert!(
            undercity.keys().all(|(n, _)| !super::scrolls(*n)),
            "Undercity's slime stands still",
        );
    }

    /// A WMO pool's fog block follows its group's `MOGI & 0x48 == 0` class: Undercity's 38 liquid
    /// groups are interior but for group 7, and Stormwind's 22 are all exterior.
    #[test]
    fn a_wmo_pools_fog_block_follows_its_groups_interior_class() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open chain");
        // Which groups of a WMO carry liquid, and whether each is an interior group.
        let liquid_groups = |chain: &mut benilla_formats::Chain, root_path: &str| {
            let bytes = chain.read_file(root_path).expect("root readable");
            let root = parse_wmo_root(&bytes).expect("parse root");
            let stem = root_path
                .strip_suffix(".wmo")
                .unwrap_or(root_path)
                .to_string();
            (0..root.group_count() as usize)
                .filter_map(|gi| {
                    let gb = chain.read_file(&format!("{stem}_{gi:03}.wmo")).ok()?;
                    wmo_group_liquid_mesh(&gb)?;
                    Some((gi, root.group_infos().get(gi).is_some_and(|g| g.interior)))
                })
                .collect::<Vec<_>>()
        };

        let uc = liquid_groups(&mut chain, "world\\wmo\\lorderon\\undercity\\undercity.wmo");
        let exterior: Vec<usize> = uc.iter().filter(|(_, i)| !i).map(|(g, _)| *g).collect();
        assert_eq!(uc.len(), 38, "Undercity's liquid group count moved: {uc:?}");
        assert_eq!(
            exterior,
            vec![7],
            "exactly one Undercity liquid group is exterior (the flag is per group, not per building)",
        );

        let sw = liquid_groups(
            &mut chain,
            "world\\wmo\\azeroth\\buildings\\stormwind\\stormwind.wmo",
        );
        assert_eq!(sw.len(), 22, "Stormwind's liquid group count moved: {sw:?}");
        assert!(
            sw.iter().all(|(_, interior)| !interior),
            "Stormwind's canals and fountains are open to the sky — none takes the interior fog: {sw:?}",
        );
    }
}

/// **Re-material every liquid surface when `waterDetail` moves.**
///
/// The two renderers are different material *types*, so the swap is a component swap: drop
/// `MeshMaterial3d<LiquidMaterial>`, add `MeshMaterial3d<LiquidCataMaterial>`, or the reverse. Both
/// handles already exist ([`LiquidEntry`] builds them side by side at startup), so nothing is
/// loaded, decoded or compiled here — the cost is one archetype move per surface, paid once per
/// flip of a settings row.
///
/// **Watches the VALUE, not `is_changed()`.** The CVar host constructs its whole `Knobs` bundle on
/// any cvar write, which deref-muts every knob resource — so `Res::is_changed()` fires when the
/// player adjusts the volume slider. `terrain_stream`'s `rescatter_clutter` documents the same trap
/// and takes the same `Local` shape; here the consequence of getting it wrong would be re-materialing
/// every lake in the world on an unrelated setting.
pub(super) fn apply_water_detail(
    mut commands: Commands,
    detail: Res<WaterDetail>,
    assets: Option<Res<LiquidAssets>>,
    all: Query<(Entity, &LiquidLane)>,
    fresh: Query<(Entity, &LiquidLane), Added<LiquidLane>>,
    mut last: Local<Option<WaterDetail>>,
) {
    let Some(assets) = assets else {
        return; // no client data - nothing spawned to material
    };
    let previous = last.replace(*detail);
    let changed = previous.is_some_and(|p| p != *detail);

    let dress = |commands: &mut Commands, entity: Entity, lane: &LiquidLane| {
        let mut e = commands.entity(entity);
        if detail.is_cata() {
            if let Some(m) = assets.cata_material(lane.kind, lane.path, lane.scroll) {
                e.remove::<MeshMaterial3d<LiquidMaterial>>();
                e.insert(MeshMaterial3d(m));
            }
        } else if let Some(m) = assets.material(lane.kind, lane.path, lane.scroll) {
            e.remove::<MeshMaterial3d<LiquidCataMaterial>>();
            e.insert(MeshMaterial3d(m));
        }
    };

    if changed {
        // The setting moved: re-dress the whole world. `fresh` is a subset of `all`, so the second
        // sweep is skipped rather than run twice.
        let mut swapped = 0usize;
        for (entity, lane) in &all {
            dress(&mut commands, entity, lane);
            swapped += 1;
        }
        info!(
            "liquid: waterDetail {} to {} ({swapped} surface(s) re-materialed)",
            previous.map_or(u8::MAX, |p| p.0),
            detail.0
        );
        return;
    }
    // The steady state: dress whatever streamed in this tick. Usually empty, and an empty `Added`
    // query is a bitmask test rather than a walk.
    for (entity, lane) in &fresh {
        dress(&mut commands, entity, lane);
    }
}

/// Flip the cata materials' below-water lane with the camera eye.
///
/// The reference chooses `ProcWaterAbove` or `ProcWaterBelow` **per draw**
/// (`CMaterialWater::RenderAbove` / `RenderBelow`), against that surface's own plane. benilla's
/// submersion verdict is a single global resource — one eye is in one liquid — and every other
/// consumer of it (the atmosphere swap, the sky suppression, the clear colour, the drift cloud)
/// already treats it that way. Flipping one uniform on the shared materials keeps this renderer in
/// step with the rest of the frame rather than inventing a second, per-surface answer that would
/// disagree with the sky above it.
///
/// Value-watched for the same reason as [`apply_water_detail`].
pub(super) fn sync_below_water(
    submerged: Res<super::Underwater>,
    mut materials: ResMut<Assets<LiquidCataMaterial>>,
    mut last: Local<Option<bool>>,
) {
    let below = submerged.0.is_water();
    if last.replace(below) == Some(below) {
        return;
    }
    let flag = if below { 1.0 } else { 0.0 };
    for (_, material) in materials.iter_mut() {
        material.extension.path.y = flag;
    }
}

/// fork: modern lighting — tell the cata materials whether the water runs the Modern
/// (Experimental) combine (`optics.w`). Modern and Experimental are the same material, mirror and
/// cost, so a move between them is one uniform on the shared materials rather than the component
/// swap [`apply_water_detail`] makes between renderers.
///
/// Value-watched for the same reason as [`apply_water_detail`].
pub(super) fn sync_water_tier(
    detail: Res<WaterDetail>,
    mut materials: ResMut<Assets<LiquidCataMaterial>>,
    mut last: Local<Option<bool>>,
) {
    let experimental = detail.is_experimental();
    if last.replace(experimental) == Some(experimental) {
        return;
    }
    let flag = if experimental { 1.0 } else { 0.0 };
    for (_, material) in materials.iter_mut() {
        material.extension.optics.w = flag;
    }
}

/// fork: modern lighting — the live override of the Modern (Experimental) tier's look
/// (`$WOW_WATER_LOOK`, browser `?water_look=`): up to twenty-four comma-separated values over
/// [`cata_water_look`]'s lanes in order — `bed_filter, filter_depth, reflect_f0, reflect_body,
/// wind_keep, shallow_keep, refract_bend, highlight_roughness, reflect_grazing, facet_spread,
/// cata_share, ripple_slope, disk_radius, highlight_swell, highlight_ripples, highlight_gain,
/// turbidity, wind_speed, caustic_gain, alpha_full, grade_near, grade_far, height_gain, -`. An
/// empty or unparsable entry keeps the kind's own value, so `,,0.3` moves only
/// the reflection. For calibrating against a reference shot without a rebuild; the settled numbers
/// belong in [`cata_water_look`].
///
/// Registered only when the session is launched with the knob set (`liquid`'s plugin), so a production
/// run pays nothing for it; within such a session it is value-watched on the string, and the
/// materials (and so their bind groups) are touched only on the frame it changes.
pub(super) fn sync_water_look(
    assets: Option<Res<LiquidAssets>>,
    mut materials: ResMut<Assets<LiquidCataMaterial>>,
    mut last: Local<Option<String>>,
) {
    let Some(assets) = assets else {
        return;
    };
    let now = crate::launch_env::live_var("WOW_WATER_LOOK").unwrap_or_default();
    // The first sight of an empty knob needs nothing: the materials were built with the defaults.
    if last.as_deref().unwrap_or("") == now {
        *last = Some(now);
        return;
    }
    let over: Vec<Option<f32>> = now.split(',').map(|p| p.trim().parse().ok()).collect();
    for (key, entry) in &assets.materials {
        if key.kind.is_fullbright() {
            continue;
        }
        let Some(mut material) = materials.get_mut(&entry.cata) else {
            continue;
        };
        let mut look = cata_water_look(key.kind);
        for (i, v) in over.iter().enumerate().take(24) {
            if let Some(v) = v {
                look[i / 4][i % 4] = *v;
            }
        }
        material.extension.look_a = look[0];
        material.extension.look_b = look[1];
        material.extension.look_c = look[2];
        material.extension.look_d = look[3];
        material.extension.look_e = look[4];
        material.extension.look_f = look[5];
    }
    info!("liquid: water look {now:?}");
    *last = Some(now);
}
