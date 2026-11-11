//! Changes for additional tokens are measured against their actual GUID snapshots.

use super::{resolve, state, UnitState};
use mlua::Lua;
use std::collections::BTreeMap;

pub(super) type Snapshot = BTreeMap<String, (Option<UnitState>, Option<u64>)>;

pub(super) fn snapshot(lua: &Lua) -> Snapshot {
    super::super::query_tokens(lua)
        .into_iter()
        .map(|token| {
            let guid = resolve(lua, &token).flatten();
            let value = guid.and_then(|guid| state(lua, guid));
            let target = lua
                .app_data_ref::<super::Metadata>()
                .and_then(|data| guid.and_then(|guid| data.targets.get(&guid).copied()));
            (token, (value, target))
        })
        .collect()
}

pub(super) fn dispatch(lua: &Lua, before: Snapshot, after: Snapshot) {
    for (token, (after, target)) in after {
        let (before, old_target) = before.get(&token).cloned().unwrap_or_default();
        let before = before.as_ref();
        let after = after.as_ref();
        let mut events = Vec::new();
        if old_target != target {
            events.push("UNIT_TARGET");
        }
        macro_rules! changed {
            ($field:ident, $event:literal) => {
                if before.map(|state| &state.$field) != after.map(|state| &state.$field) {
                    events.push($event);
                }
            };
        }
        changed!(name, "UNIT_NAME_UPDATE");
        changed!(health, "UNIT_HEALTH");
        changed!(max_health, "UNIT_MAXHEALTH");
        changed!(level, "UNIT_LEVEL");
        changed!(flags, "UNIT_FLAGS");
        changed!(dynamic_flags, "UNIT_DYNAMIC_FLAGS");
        changed!(player_flags, "PLAYER_FLAGS_CHANGED");
        changed!(is_connected, "UNIT_CONNECTION");
        changed!(quest_log, "UNIT_QUEST_LOG_CHANGED");
        if before.map(|state| {
            (
                state.reaction,
                state.pvp,
                state.tapped,
                state.tapped_by_player,
                &state.faction_group,
            )
        }) != after.map(|state| {
            (
                state.reaction,
                state.pvp,
                state.tapped,
                state.tapped_by_player,
                &state.faction_group,
            )
        }) {
            events.push("UNIT_FACTION");
        }
        for event in events {
            crate::script::pfui_host::fire_pfui_event(
                lua,
                event,
                vec![crate::script::ScriptValue::Str(token.clone())],
            );
        }
    }
}
