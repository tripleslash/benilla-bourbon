//! Extra unit tokens read the same held-unit projection as stock target chains.

use benilla_ui::script::{pfui_host::units, UiScript};
use bevy::prelude::*;

pub(crate) fn register(app: &mut App) {
    app.add_systems(Update, feed.in_set(crate::pfui::InputSnapshots));
}

fn feed(
    script: Option<NonSend<UiScript>>,
    tables: super::SnapshotTables,
    stores: Query<(&crate::net::Guid, Ref<crate::net::ObjectStore>)>,
    self_q: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
    names: Res<crate::names::NameCache>,
    commands: Res<crate::net::NetCommands>,
    relations: crate::target::RelationWorld,
    group: Res<crate::ui_party::GroupState>,
    guild: Res<crate::ui_guild::GuildState>,
    mut removed: RemovedComponents<crate::net::ObjectStore>,
    mut memo: Local<crate::ui_script::VmMemo<bool>>,
) {
    let Some(script) = script else { return };
    // Cleared against a new VM, which is told everything once.
    let memo = memo.get(&script);
    let despawned = removed.read().count() != 0;
    if *memo
        && !despawned
        && !names.is_changed()
        && !group.is_changed()
        && !guild.is_changed()
        && !stores.iter().any(|(_, store)| store.is_changed())
    {
        return;
    }
    let by_guid = |guid: u64| {
        stores
            .iter()
            .find(|(id, _)| id.0 == guid)
            .map(|(_, store)| store.into_inner())
    };
    // What a held unit's snapshot reads, as the live unit reads build it.
    let held = super::held::HeldUnits {
        names: &names,
        group: &group,
        self_store: self_q.single().ok(),
        unit_names: tables.unit_names(),
        types: tables.types(&names),
        rel: relations.get(),
        me: relations.me(),
        tooltip: super::tooltip_facts::TooltipSources {
            names: &names,
            catalog: tables.catalog(),
            object: &by_guid,
            study: tables.study(),
        },
        guild: &guild,
    };
    units::set_traits(
        script.lua(),
        stores
            .iter()
            .filter(|(_, store)| store.is_unit())
            .map(|(guid, store)| {
                (
                    guid.0,
                    units::Traits {
                        creature_type: held.types.of(&store),
                        created_by_spell: store.0.unit_created_by_spell().unwrap_or(0),
                    },
                )
            }),
    );
    units::set_unit_states(
        script.lua(),
        stores
            .iter()
            .filter(|(_, store)| store.is_unit())
            .map(|(guid, store)| {
                // The name `GetUnitName` reads, a miss asked as the stock getter asks it.
                let name = names
                    .resolve_unit(
                        guid.0,
                        Some(&store),
                        crate::query_cache::Ask::New(&commands),
                    )
                    .map(str::to_string);
                (
                    guid.0,
                    held.state_named(&store, guid.0, name),
                    store.0.unit_target().unwrap_or(0),
                )
            }),
    );
    *memo = true;
}
