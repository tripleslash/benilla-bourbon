//! **Healing Bevy 0.18's material-specialization tick race** (fork-owned; `modern/HOOKS.md`).
//!
//! `bevy_pbr`'s `specialize_material_meshes` unwraps `EntitySpecializationTicks[entity]` for every
//! visible mesh (`material.rs:1061`). That table is filled by `extract_entities_needs_specialization`
//! from the main world's `check_entities_needing_specialization`, which runs in `PostUpdate` with no
//! ordering against benilla's own material writers. When a writer lands after the check, the same
//! frame extracts the material instance but no tick, and the render world panics the moment the
//! mesh is visible. Bisected 2026-09-15: any added `PostUpdate` system, or just the sun arc moving
//! the sky, was enough to trip it — in classic as well as modern.
//!
//! This system runs just before specialization and inserts, for each visible mesh that has a
//! material instance but no tick, exactly the tick extraction would have written one frame later.
//! It is a no-op when the race does not occur.

use bevy::ecs::system::SystemChangeTick;
use bevy::pbr::{EntitySpecializationTickPair, EntitySpecializationTicks, RenderMaterialInstances};
use bevy::prelude::*;
use bevy::render::view::RenderVisibleEntities;
use bevy::render::{Render, RenderApp, RenderSystems};

pub(super) fn plugin(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app.add_systems(
        Render,
        heal_missing_specialization_ticks
            .in_set(RenderSystems::PrepareMeshes)
            .before(bevy::pbr::specialize_material_meshes),
    );
}

fn heal_missing_specialization_ticks(
    views: Query<&RenderVisibleEntities>,
    instances: Res<RenderMaterialInstances>,
    mut ticks: ResMut<EntitySpecializationTicks>,
    change: SystemChangeTick,
    mut healed: Local<u64>,
) {
    for visible in &views {
        for (_, main) in visible.iter::<Mesh3d>() {
            if instances.instances.contains_key(main) && !ticks.contains_key(main) {
                ticks.insert(
                    *main,
                    EntitySpecializationTickPair {
                        system_tick: change.this_run(),
                        material_instances_tick: instances.current_change_tick,
                    },
                );
                *healed += 1;
                if *healed == 1 {
                    warn!(
                        "healed a missing material-specialization tick (bevy_pbr 0.18 PostUpdate race; \
                         benilla_world::modern::tick_heal)"
                    );
                }
            }
        }
    }
}
