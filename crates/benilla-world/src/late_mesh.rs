//! fork: web — re-specialize an entity whose mesh the upload budget deferred.
//!
//! Bevy 0.19 specializes a mesh entity's pipeline only on the frame its `DirtySpecializations`
//! mark is live, and re-tries later only what its pending queue holds. `specialize_material_meshes`
//! queues an entity whose material or mesh instance is not ready, but one whose `RenderMesh` is
//! not prepared yet it simply skips, and nothing marks it again. With
//! [`RenderAssetBytesPerFrame`](bevy::render::render_asset::RenderAssetBytesPerFrame) set (the
//! browser's default, `WEB_UPLOAD_BUDGET_MB`) a large mesh prepares a frame or more after its
//! entity spawns, so the entity never gets a pipeline and never draws: every terrain cell in the
//! browser, while the small doodad meshes beside them drew.
//!
//! So around the mesh prepare: note what was extracted, keep what the budget deferred, and on the
//! frame a deferred mesh lands, mark every entity drawing it as changed, which that frame's
//! specialization then picks up. Registered only where a budget is.

use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::{prepare_assets, ExtractedAssets, RenderAssets};
use bevy::render::sync_world::MainEntity;
use bevy::render::{Render, RenderApp, RenderSystems};

/// Meshes extracted this frame, and meshes the budget deferred that have not prepared yet.
#[derive(Resource, Default)]
struct LateMeshes {
    extracted: Vec<AssetId<Mesh>>,
    waiting: HashSet<AssetId<Mesh>>,
}

/// Before the prepare: which meshes arrive this frame (`prepare_assets` drains the list).
fn note_extracted(extracted: Res<ExtractedAssets<RenderMesh>>, mut late: ResMut<LateMeshes>) {
    late.extracted.clear();
    late.extracted
        .extend(extracted.extracted.iter().map(|(id, _)| *id));
    for id in &extracted.removed {
        late.waiting.remove(id);
    }
}

/// After the prepare: the arrivals the budget deferred wait; the waiting ones that prepared now
/// have their entities marked for specialization.
fn mark_landed(
    meshes: Res<RenderAssets<RenderMesh>>,
    instances: Res<bevy::pbr::RenderMeshInstances>,
    mut dirty: ResMut<bevy::render::camera::DirtySpecializations>,
    mut late: ResMut<LateMeshes>,
) {
    let LateMeshes { extracted, waiting } = &mut *late;
    let landed: HashSet<AssetId<Mesh>> = waiting
        .iter()
        .copied()
        .filter(|id| meshes.get(*id).is_some())
        .collect();
    for id in extracted.drain(..) {
        if meshes.get(id).is_none() {
            waiting.insert(id);
        }
    }
    if landed.is_empty() {
        return;
    }
    waiting.retain(|id| !landed.contains(id));
    let entities: Vec<MainEntity> = match &*instances {
        bevy::pbr::RenderMeshInstances::CpuBuilding(m) => m.keys().copied().collect(),
        bevy::pbr::RenderMeshInstances::GpuBuilding(m) => m.keys().copied().collect(),
    };
    for main in entities {
        if instances
            .mesh_asset_id(main)
            .is_some_and(|id| landed.contains(&id))
        {
            dirty.changed_renderables.insert(main);
        }
    }
}

/// Install the pair in the render world.
pub(crate) fn plugin(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.init_resource::<LateMeshes>().add_systems(
        Render,
        (
            note_extracted.before(prepare_assets::<RenderMesh>),
            mark_landed.after(prepare_assets::<RenderMesh>),
        )
            .in_set(RenderSystems::PrepareAssets),
    );
}
