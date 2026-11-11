//! `C_Reputation` and `GetFactionInfoByID`: the stock getters' rows and reads ([`faction_at`],
//! [`read_entry`], [`read_faction`]), answered as ClassicAPI's tables, with the faction id the
//! modern struct adds.

use super::{
    binding_abi, faction_at, is_header_key, read_entry, read_faction, FactionEntry, FactionQuery,
    Model,
};
use mlua::{Lua, MultiValue, Table, Value};

fn data(
    lua: &Lua,
    entry: &FactionEntry,
    is_header: bool,
    collapsed: bool,
    watched: Option<u32>,
) -> mlua::Result<Table> {
    let data = lua.create_table()?;
    data.set("factionID", entry.faction_id)?;
    data.set("name", entry.name.as_str())?;
    data.set("description", entry.description.as_str())?;
    data.set("reaction", entry.standing_id)?;
    data.set("currentReactionThreshold", entry.bar_min)?;
    data.set("nextReactionThreshold", entry.bar_max)?;
    data.set("currentStanding", entry.standing)?;
    data.set("atWarWith", entry.at_war)?;
    data.set("canToggleAtWar", entry.can_toggle_at_war)?;
    data.set("isHeader", is_header)?;
    data.set("isCollapsed", collapsed)?;
    data.set("isWatched", watched == Some(entry.rep_list_id))?;
    data.set("canSetInactive", !is_header)?;
    for name in [
        "isHeaderWithRep",
        "isChild",
        "hasBonusRepGain",
        "isAccountWide",
    ] {
        data.set(name, false)?;
    }
    Ok(data)
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let namespace = lua.create_table()?;
    namespace.set(
        "GetFactionDataByIndex",
        lua.create_function(|lua, index: i64| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let faction = usize::try_from(index).map_or(0, |i| faction_at(&model, i));
            let is_header = is_header_key(&model, faction);
            let collapsed = is_header && model.reputation_collapsed.contains(&faction);
            let read = read_entry(lua, &model, faction);
            read.entry
                .map(|e| data(lua, &e, is_header, collapsed, read.watched))
                .transpose()
        })?,
    )?;
    namespace.set(
        "GetWatchedFactionData",
        lua.create_function(|lua, ()| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let read = read_faction(lua, &model, FactionQuery::Watched);
            read.entry
                .map(|e| data(lua, &e, e.is_header, false, read.watched))
                .transpose()
        })?,
    )?;
    // The eleven GetFactionInfo values by faction ID. A faction the player has not met answers
    // its name with the documented neutral rank; one without a reputation slot answers nothing.
    lua.globals().set(
        "GetFactionInfoByID",
        lua.create_function(|lua, id: Value| {
            use Value::{Integer, Nil};
            let id = lua
                .coerce_number(id)?
                .filter(|n| n.fract() == 0.0)
                .and_then(|n| u32::try_from(n as i64).ok());
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let Some(read) = id.map(|id| read_faction(lua, &model, FactionQuery::Of(id))) else {
                return Ok(MultiValue::new());
            };
            let Some(entry) = read.entry.filter(|e| !e.is_header) else {
                return Ok(MultiValue::new());
            };
            let mut values = vec![
                Value::String(lua.create_string(&entry.name)?),
                Value::String(lua.create_string(&entry.description)?),
            ];
            if entry.visible {
                values.extend([
                    Integer(i64::from(entry.standing_id)),
                    Integer(i64::from(entry.bar_min)),
                    Integer(i64::from(entry.bar_max)),
                    Integer(i64::from(entry.standing)),
                    binding_abi::flag(entry.at_war),
                    binding_abi::flag(entry.can_toggle_at_war),
                    Nil,
                    Nil,
                    binding_abi::flag(read.watched == Some(entry.rep_list_id)),
                ]);
            } else {
                values.extend([Integer(4), Integer(0), Integer(3000), Integer(0)]);
                values.extend([Nil, Nil, Nil, Nil, Nil]);
            }
            Ok(MultiValue::from_vec(values))
        })?,
    )?;
    lua.globals().set("C_Reputation", namespace)
}
