//! Real social GUIDs and the list-update event used by extension-owned friend notes.

use mlua::{Lua, Table};

const FRIEND_GUIDS: &str = "benilla.pfui.friend_guids";

pub fn set_friend_guids<'a>(
    lua: &Lua,
    rows: impl IntoIterator<Item = (&'a str, u64)>,
) -> mlua::Result<()> {
    let table = lua.create_table()?;
    for (name, guid) in rows {
        if !name.is_empty() && guid != 0 {
            table.set(name.to_ascii_lowercase(), format!("0x{guid:016X}"))?;
        }
    }
    lua.set_named_registry_value(FRIEND_GUIDS, table)
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_named_registry_value(FRIEND_GUIDS, lua.create_table()?)?;
    let bridge = lua.create_table()?;
    bridge.set(
        "guid",
        lua.create_function(|lua, name: String| {
            let guids: Table = lua.named_registry_value(FRIEND_GUIDS)?;
            guids.get::<Option<String>>(name.to_ascii_lowercase())
        })?,
    )?;
    bridge.set(
        "updated",
        lua.create_function(|lua, ()| {
            crate::script::pfui_host::fire_pfui_event(lua, "FRIENDLIST_UPDATE", Vec::new());
            Ok(())
        })?,
    )?;
    lua.set_named_registry_value("benilla.pfui.social_bridge", bridge)
}
