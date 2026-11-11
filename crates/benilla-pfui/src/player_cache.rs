//! Mirror real name-query replies only while the persistent cache is enabled.

use mlua::{Function, Lua, Table};

/// Class tokens come from the host's class catalog; race IDs resolve through the localized DBC.
/// Wire gender is 0/1, unlike UnitSex's 2/3. The private writer can correct a female to male.
pub fn observe_name(
    lua: &Lua,
    guid: u64,
    name: &str,
    class: Option<&str>,
    race: u32,
    gender: u32,
) -> mlua::Result<bool> {
    if guid == 0 || guid >> 48 != 0 || name.is_empty() {
        return Ok(false);
    }
    let bridge: Option<Table> = lua.named_registry_value("benilla.pfui.social_bridge")?;
    let Some(writer) = bridge
        .map(|bridge| bridge.get::<Option<Function>>("nameReply"))
        .transpose()?
        .flatten()
    else {
        return Ok(false);
    };
    writer.call((
        format!("0x{guid:016X}"),
        name,
        class,
        race,
        (gender <= 1).then_some(gender),
    ))
}
