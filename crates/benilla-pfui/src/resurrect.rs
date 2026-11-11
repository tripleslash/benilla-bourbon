//! `UnitHasIncomingResurrection`: whether a resurrection cast at the unit (or the corpse of a
//! released player) is under way, as the client sees the casts ([`set`]).

use benilla_ui::script::{pfui_host, ScriptValue};
use mlua::{Lua, Value};
use std::collections::HashSet;

#[derive(Default)]
struct Incoming(HashSet<u64>);

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Incoming::default());
    lua.globals().set(
        "UnitHasIncomingResurrection",
        lua.create_function(|lua, unit: Value| {
            let Value::String(unit) = unit else {
                return Ok(false);
            };
            let guid = pfui_host::units::guid(lua, &unit.to_string_lossy())
                .ok()
                .flatten();
            Ok(guid.is_some_and(|guid| {
                lua.app_data_ref::<Incoming>()
                    .is_some_and(|incoming| incoming.0.contains(&guid))
            }))
        })?,
    )
}

/// Whether `guid` has a resurrection coming; a change raises `INCOMING_RESURRECT_CHANGED` once for
/// each of `tokens`, the unit tokens that name it.
pub fn set(lua: &Lua, guid: u64, incoming: bool, tokens: &[&str]) {
    let changed = lua.app_data_mut::<Incoming>().is_some_and(|mut set| {
        if incoming {
            set.0.insert(guid)
        } else {
            set.0.remove(&guid)
        }
    });
    if changed {
        for token in tokens {
            pfui_host::fire_pfui_event(
                lua,
                "INCOMING_RESURRECT_CHANGED",
                vec![ScriptValue::Str((*token).to_string())],
            );
        }
    }
}

/// Forget every cast: a logout.
pub fn clear(lua: &Lua) {
    if let Some(mut set) = lua.app_data_mut::<Incoming>() {
        set.0.clear();
    }
}
