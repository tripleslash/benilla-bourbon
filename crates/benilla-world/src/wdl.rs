//! Distant low-detail terrain (WDL): streams the map's coarse horizon heightmap around the view —
//! beyond the detailed ADT ring — drawn unlit-white under a fog pair of its own that saturates it
//! into the flat scene-fog colour (a "fog hull", decision 1521), the hills the reference shows on the
//! horizon where ours used to fade to void. The parse + coarse mesh live in `benilla_formats::wdl`; this
//! is just the Bevy streaming + render glue (one shared [`WdlMaterial`], one mesh per ring tile).
//!
//! Mechanism + RE (geometry/shading/fog/depth all VERIFIED from apitrace WoW.8 + the real `.wdl`).
//! How it partitions against the detailed world — the reference's far-band **backdrop** law, and why a
//! shared clip plane could not work — is `wdl.wgsl`'s header (decision 0684).

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;

use crate::view::WorldCamera;
use crate::world_map::CurrentMap;
use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};
use benilla_assets::materials::{WdlExt, WdlMaterial};
use benilla_assets::LockRecover;
use benilla_assets::MapCatalogRes;
use benilla_assets::{AssetSet, RenderConfig, WorldAssets};
use benilla_formats::WdlFile;

/// Chebyshev tile radius the WDL ring covers around the view. WDL is flat haze at every distance (its
/// own saturated fog pair, decision 1521) — what it contributes is silhouette, and hills 2–4 tiles out
/// still rise above the horizon as fog-coloured silhouettes, so we extend toward the reference's
/// `horizonfarclip` (~2112 yd ≈ 4 tiles). Drawn as a full ring; the
/// shader makes it a **backdrop** — near plane at `farclip − 33`, depth clamped behind everything the
/// detailed world can draw (`wdl.wgsl`'s header, decision 0684) — so it fills whatever the detailed
/// world leaves empty and can never overlap it. (The reference's own far walk is a ±3-tile window.)
const WDL_RADIUS: u32 = 5;

/// fork: modern lighting (FORK.md) — the ring a modern frame streams. Its height fog
/// lets the far ranges rise above the haze kilometres out (Hyjal behind Orgrimmar), so the band reaches
/// ~6.4 km, on coarser meshes the further out a tile lies ([`modern_lod`]).
const WDL_RADIUS_MODERN: u32 = 12;

/// The mesh a modern ring tile `ring` tiles (Chebyshev) from the view's own takes: 0 = the full centre
/// fan, otherwise the outer lattice every `n` corners (`WdlFile::tile_mesh_coarse`).
fn modern_lod(ring: u32) -> usize {
    match ring {
        0..=2 => 0,
        3..=5 => 1,
        _ => 2,
    }
}

/// The curtain (yd) a coarse ring tile hangs from its border over the T-junctions against a finer one.
const WDL_SKIRT: f32 = 60.0;

/// New WDL tiles built per frame — coarse meshes are cheap (545 verts), but a whole ring at once
/// would hitch on zone load, so spread it.
const WDL_LOADS_PER_FRAME: usize = 8;

/// The WDL subsystem: load the map's `.wdl` + a shared fog material at startup, then stream the
/// coarse horizon ring in/out around the view each frame.
pub(crate) struct WdlPlugin;

impl Plugin for WdlPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WdlMaterial>::default())
            .add_systems(Startup, setup_wdl.after(AssetSet::Open))
            // The distant ring is world, and follows the world's lifecycle (decision 0777): it
            // spawns only while a character is in one, and its meshes go when they leave.
            .add_systems(Update, stream_wdl.run_if(crate::schedule::world_is_live))
            .add_systems(
                Update,
                release_wdl_ring
                    .run_if(crate::schedule::world_left)
                    .after(crate::schedule::WorldStage::Net)
                    .before(crate::schedule::WorldStage::Stream),
            );
    }
}

/// Streams the coarse WDL ring: the parsed file, the shared material every ring tile uses, and which
/// ring tiles are currently spawned. (The material holds no fog of its own — it reads the shared
/// global-light buffer, same as terrain and the models.)
#[derive(Resource)]
pub(crate) struct WdlStreamer {
    wdl: WdlFile,
    /// `mapId` the loaded `wdl` is for; a cross-map teleport (≠ [`CurrentMap`]) reloads it.
    map_id: u32,
    material: Handle<WdlMaterial>,
    /// Spawned ring tiles and the LOD each was built at (always 0 in a classic frame).
    loaded: HashMap<(u32, u32), (Entity, usize)>,
}

impl WdlStreamer {
    /// The drawn WDL surface height (raw WoW `z`) under a **Bevy-space** position — whole-map
    /// coverage from the coarse horizon heightmap, exact to the rendered mesh
    /// (`WdlFile::height_at`). The far leg of the lens-flare occlusion march
    /// (`sun::follow::FlareGate`), beyond the resident detailed-terrain ring. `None` off the map
    /// or where no WDL tile is authored (open ocean) — never an occluder there.
    pub(crate) fn height_under(&self, bevy_pos: Vec3) -> Option<f32> {
        let wow = bevy_to_wow(bevy_pos);
        self.wdl.height_at(wow[0], wow[1])
    }

    /// One tile's coarse surface as **parry heightfield inputs in Bevy space**: the 17×17 height
    /// lattice, where to put it, and how far it spans. `None` where this map authors no such tile.
    ///
    /// The horizon data, borrowed for a second job: it is the only description of a tile's ground
    /// that exists *before* the tile's own `.adt` has been fetched, because the whole map's copy of
    /// it arrived in one 770 KB file at startup. `terrain_stream::provisional` stands a body on it
    /// while the real heightfield is still on the wire.
    ///
    /// **A lattice, not a mesh, and that is the whole point.** The obvious form is
    /// `WdlFile::tile_mesh` fed to a trimesh, and it is what this returned first — 3,072 triangles
    /// per tile, and a parry trimesh means a QBVH build. Nine of those at a tile-boundary crossing
    /// is a visible hitch on a target whose "async compute pool" is `spawn_local` on the thread that
    /// draws. A heightfield stores the grid and builds no acceleration structure at all, because a
    /// regular lattice already is one: 289 floats copied, and the cost stops being worth pacing.
    ///
    /// The index order passes through untouched, which is worth stating because it looks like it
    /// should not. `WdlTileGrid` is `[row][col]` with row along −X and column along −Y (WoW), and
    /// parry's heightfield is `[row][col]` with row along +Z and column along +X (its own space).
    /// `wow_to_bevy` maps +X to −Z and +Y to −X, so both flips cancel and each index keeps its axis
    /// and its direction.
    pub(crate) fn coarse_floor(&self, tile_x: u32, tile_y: u32) -> Option<CoarseFloor> {
        self.wdl.tile_height_grid(tile_x, tile_y).map(coarse_floor)
    }
}

/// One WDL tile's height lattice as parry heightfield inputs — [`WdlStreamer::coarse_floor`]'s
/// whole body, split out because it is the part with an axis convention in it and therefore the
/// part worth testing on its own.
fn coarse_floor(grid: benilla_formats::WdlTileGrid) -> CoarseFloor {
    CoarseFloor {
        // `centre` carries no z: the heights are absolute WoW z and parry adds them to the
        // collider's own origin, so the collider sits at y = 0 and the lattice supplies the height.
        // (`scale.y` is 1 for the same reason.)
        at: wow_to_bevy([grid.centre[0], grid.centre[1], 0.0]),
        scale: Vec3::new(grid.extent, 1.0, grid.extent),
        heights: grid.heights,
    }
}

/// What [`WdlStreamer::coarse_floor`] hands back — a parry heightfield, unbuilt.
pub(crate) struct CoarseFloor {
    /// 17×17 absolute heights, `[row][col]`; row indexes Bevy **Z**, column indexes Bevy **X**.
    pub(crate) heights: Vec<Vec<f32>>,
    /// Where the lattice's centre sits in Bevy space (`y = 0` — see [`WdlStreamer::coarse_floor`]).
    pub(crate) at: Vec3,
    /// Per-axis scale: the tile's edge on X and Z, and `1.0` on Y since the heights are absolute.
    pub(crate) scale: Vec3,
}

/// Marks a spawned WDL tile (so a future system could query/cull them as a group if needed).
#[derive(Component)]
struct WdlTile;

fn setup_wdl(
    mut commands: Commands,
    config: Option<Res<RenderConfig>>,
    world_assets: Option<ResMut<WorldAssets>>,
    mut materials: ResMut<Assets<WdlMaterial>>,
) {
    let (Some(_config), Some(world_assets)) = (config, world_assets) else {
        return; // no client data → no terrain at all, so no WDL
    };
    // Default map = Azeroth (Eastern Kingdoms), matching world_map's DEFAULT_MAP_ID.
    let wdl = match WdlFile::load(&mut world_assets.chain.lock_recover(), "Azeroth") {
        Ok(w) => {
            info!(
                "WDL: Azeroth.wdl loaded ({} distant tiles)",
                w.present_count()
            );
            w
        }
        Err(e) => {
            warn!("WDL unavailable, no distant terrain: {e:#}");
            return;
        }
    };
    // One shared material, reading the shared global light like terrain does — no seed and no
    // per-frame push: `build_light_data` has already packed the fog by the first draw. Opaque ⇒
    // depth-LEQUAL + depth-write, no blend (the verified WoW.8 state).
    let material = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            cull_mode: None,
            ..default()
        },
        extension: WdlExt {
            light_buf: world_assets.shared_light.clone(),
        },
    });
    commands.insert_resource(WdlStreamer {
        wdl,
        map_id: 0,
        material,
        loaded: HashMap::new(),
    });
}

fn stream_wdl(
    mut commands: Commands,
    streamer: Option<ResMut<WdlStreamer>>,
    assets: Option<ResMut<WorldAssets>>,
    focus: Res<crate::terrain_stream::ViewFocus>,
    camera: Query<&Transform, With<WorldCamera>>,
    current_map: Option<Res<CurrentMap>>,
    map_catalog: Option<Res<MapCatalogRes>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // All four are absent together — there is no client data, so `setup_wdl` and `load_world_map`
    // both bailed. `Option` rather than a hard `Res` because a missing resource is a *validation*
    // failure, not a `None`: the system would never run, and Bevy's default handler panics the
    // client (decision 1451). This one is gated on `world_is_live`, so it takes a world with no
    // install to reach — which a player build in the wrong folder can still do.
    let (Some(mut streamer), Some(assets), Some(current_map), Some(map_catalog)) =
        (streamer, assets, current_map, map_catalog)
    else {
        return;
    };
    // The map the *focus* is on — `CurrentMap` once the snap has landed, the picked character's
    // own map during world entry, so the ring is never built for the map we are leaving.
    let map_id = focus.map(Some(current_map.0));

    // Cross-map teleport: reload the new map's `.wdl` and drop the old ring.
    if map_id != streamer.map_id {
        if let Some(dir) = map_catalog.0.directory(map_id) {
            match WdlFile::load(&mut assets.chain.lock_recover(), dir) {
                Ok(wdl) => {
                    for (_, (e, _)) in streamer.loaded.drain() {
                        commands.entity(e).despawn();
                    }
                    streamer.wdl = wdl;
                    streamer.map_id = map_id;
                }
                // No `.wdl` for this map (instances often lack one) → just clear the ring.
                Err(_) => {
                    for (_, (e, _)) in streamer.loaded.drain() {
                        commands.entity(e).despawn();
                    }
                    streamer.map_id = map_id;
                }
            }
        }
    }

    // The same view focus the detailed streamer uses — literally the same resource now, rather
    // than a copy under a comment claiming they agree (decision 0777).
    let center = focus.resolve(camera.single().ok().map(|c| c.translation));
    // The FULL window, the camera's own tile INCLUDED (`tiles_in_ring`'s doc is the why — at a
    // lowered view distance the own tile *is* the near horizon, and dropping it leaves a gap the sky
    // pours through). What bounds the band on the near side is the shader's near plane, never the
    // streamed set, so this cannot toggle against the detailed streaming radius — a constant backdrop.
    let modern = crate::modern::frame_is_modern();
    let radius = if modern {
        WDL_RADIUS_MODERN
    } else {
        WDL_RADIUS
    };
    let (ctx, cty) = benilla_formats::world_to_tile(center[0], center[1]);
    let lod_of = |c: (u32, u32)| {
        if modern {
            modern_lod(
                (c.0 as i32 - ctx as i32)
                    .unsigned_abs()
                    .max((c.1 as i32 - cty as i32).unsigned_abs()),
            )
        } else {
            0
        }
    };
    let desired = streamer.wdl.tiles_in_ring(center[0], center[1], radius);

    // Unload ring tiles no longer wanted, or wanted at another LOD.
    let stale: Vec<(u32, u32)> = streamer
        .loaded
        .iter()
        .filter(|(c, (_, lod))| !desired.contains(c) || *lod != lod_of(**c))
        .map(|(c, _)| *c)
        .collect();
    for coords in stale {
        if let Some((e, _)) = streamer.loaded.remove(&coords) {
            commands.entity(e).despawn();
        }
    }

    // Load a few missing tiles this frame, nearest first.
    let mut missing: Vec<(u32, u32)> = desired
        .into_iter()
        .filter(|c| !streamer.loaded.contains_key(c))
        .collect();
    missing.sort_by_key(|c| {
        (c.0 as i32 - ctx as i32)
            .abs()
            .max((c.1 as i32 - cty as i32).abs())
    });
    missing.truncate(WDL_LOADS_PER_FRAME);
    for coords in missing {
        let lod = lod_of(coords);
        let tile = if lod == 0 {
            streamer.wdl.tile_mesh(coords.0, coords.1)
        } else {
            streamer
                .wdl
                .tile_mesh_coarse(coords.0, coords.1, lod, WDL_SKIRT)
        };
        let Some(tile) = tile else {
            continue;
        };
        let positions: Vec<[f32; 3]> = tile
            .positions
            .iter()
            .map(|p| wow_to_bevy(*p).to_array())
            .collect();
        // Unlit + untextured: a constant up-normal and zero UV satisfy the StandardMaterial pipeline;
        // the custom WDL shader reads neither (it outputs the flat fog colour).
        let n = positions.len();
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; n]);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; n]);
        mesh.insert_indices(Indices::U32(tile.indices));
        let e = commands
            .spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(streamer.material.clone()),
                Transform::IDENTITY,
                WdlTile,
                // The far band is exterior scene like the detailed tiles are: the reference's far
                // walk `0x683040` is fed by the SAME per-window populate `0x682fa0`, so from a sealed
                // room the horizon is not drawn either (`crate::exterior_cull`). One ring tile is one
                // drawn object here, which is already the reference's far-tier granularity — this
                // band never needed the chunk split the near tiles did (decision 0780).
                crate::exterior_cull::ExteriorScene,
            ))
            .id();
        streamer.loaded.insert(coords, (e, lod));
    }
}

/// Leaving the world drops the distant ring (decision 0777). The parsed `.wdl` itself stays — it is
/// one small file, the resource `stream_wdl` needs to exist at all, and `height_under` is read by
/// the sun's flare-occlusion march; what costs per frame is the spawned mesh set, and that goes.
fn release_wdl_ring(mut commands: Commands, streamer: Option<ResMut<WdlStreamer>>) {
    let Some(mut streamer) = streamer else { return };
    for (_, (e, _)) in streamer.loaded.drain() {
        commands.entity(e).despawn();
    }
}

/// The backdrop law (`wdl.wgsl`'s header, decision 0684) is a property of the **shader**, so it is
/// checked there — the same shape as `sky_order.rs`'s depth-law test, and for the same reason: the
/// failure it guards is invisible except at a ridge crest on a fogged horizon, and the obvious "tidy-up"
/// (go back to one shared clip plane, drop the frag-depth write) silently reintroduces it.
#[test]
fn the_far_band_stays_a_depth_pushed_backdrop() {
    let src = benilla_assets::materials::WDL_WGSL;
    assert!(
        src.contains("const WDL_OVERLAP: f32 = 33.0;"),
        "wdl.wgsl: the far band's 33 yd overlap into the wall is gone — the coarse-vs-fine seam \
         reopens as a hole at the horizon (the reference's far-band near plane, [0x8101b0])"
    );
    assert!(
        src.contains("out.depth = depth;") && src.contains("view_z_to_depth_ndc(-farclip)"),
        "wdl.wgsl: the far band no longer clamps its depth behind the far-clip wall — its overlap \
         now pokes THROUGH the detailed terrain (the reference's compressed far-band depth range)"
    );
}

/// The hull is a **fog hull**, not a fogged surface (decision 1521): the reference's far-band emitter
/// submits its own fog pair — start `0`, end `1.0` (`0x6bd7ae`–`0x6bd7c8` inside `0x6bd780`) — so the
/// hull is the flat fog colour at every distance. Reading the SCENE fog distances here is the bug this
/// pins: the 33 yd overlap then sits inside the fog ramp, up to `33 / (end − start)` white (25% at
/// farclip 177), a pale band above the fine terrain's fogged-out silhouette that grows as the view
/// distance drops — and it is invisible at 777, so it can sit in the open for months.
#[test]
fn the_far_band_is_a_fog_hull_not_a_fogged_surface() {
    let src = benilla_assets::materials::WDL_WGSL;
    assert!(
        src.contains("rgb = w.fog_color.xyz;"),
        "wdl.wgsl: the hull no longer paints the flat fog colour (the reference's own start-0 / \
         end-1.0 fog pair, saturated beyond one yard)"
    );
    assert!(
        !src.contains("w.fog_params.x") && !src.contains("w.fog_params.y"),
        "wdl.wgsl: the hull reads the SCENE fog distances again — the 33 yd overlap goes partly \
         white at low view distances (the band decision 1521 closed)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The heights matrix passes through untransposed, and that is a claim worth proving.**
    ///
    /// `WdlTileGrid` is `[row][col]` with row along −X and column along −Y in WoW; parry's
    /// heightfield is `[row][col]` with row along +Z and column along +X in its own space.
    /// `wow_to_bevy` maps +X to −Z and +Y to −X, so both flips cancel — which is exactly the kind of
    /// reasoning that is right until it is not, and silently puts a tile's floor sideways when it is
    /// not.
    ///
    /// So: for each corner, where the WDL says that sample is (mapped into Bevy space) against
    /// where parry will place `heights[i][j]` (its own documented `x_at(j)` / `z_at(i)`). Two
    /// derivations, one answer.
    #[test]
    fn the_height_lattice_lands_where_the_wdl_says_it_does() {
        const EDGE: usize = 17;
        let extent = 533.333_3_f32;
        // A lattice whose rows and columns cannot be confused for one another.
        let heights: Vec<Vec<f32>> = (0..EDGE)
            .map(|r| (0..EDGE).map(|c| (r * 100 + c) as f32).collect())
            .collect();
        let centre = [1234.5_f32, -678.25_f32];
        let floor = coarse_floor(benilla_formats::WdlTileGrid {
            heights: heights.clone(),
            centre,
            extent,
        });

        for (r, row) in heights.iter().enumerate() {
            for (c, height) in row.iter().enumerate() {
                // Where the WDL puts this sample: row along −X, column along −Y, then into Bevy.
                let wow = [
                    centre[0] + (0.5 - r as f32 / 16.0) * extent,
                    centre[1] + (0.5 - c as f32 / 16.0) * extent,
                    *height,
                ];
                let want = wow_to_bevy(wow);
                // Where parry puts `heights[i][j]`: `x_at(j) = (-0.5 + j/16) * scale.x`,
                // `z_at(i) = (-0.5 + i/16) * scale.z`, plus the collider's own origin.
                let got = floor.at
                    + Vec3::new(
                        (-0.5 + c as f32 / 16.0) * floor.scale.x,
                        floor.heights[r][c] * floor.scale.y,
                        (-0.5 + r as f32 / 16.0) * floor.scale.z,
                    );
                assert!(
                    want.distance(got) < 0.01,
                    "sample [{r}][{c}]: the WDL says {want:?}, parry would place it at {got:?}"
                );
            }
        }
    }
}
