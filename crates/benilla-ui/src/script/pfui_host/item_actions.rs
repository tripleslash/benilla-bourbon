//! Item references bind a carried instance at request time, then the engine re-resolves its slot.

use super::{id, Extras, Instance};
use crate::script::{pfui_host::attributes::ClickAction, ScriptCall, EQUIPMENT_BAG};
use mlua::{Lua, Table, Value};

fn carried(bag: i64) -> bool {
    (0..=4).contains(&bag) || bag == -2 || bag == EQUIPMENT_BAG
}

/// The carried item `input` names (a location, an item guid, an id or link, or a name), the
/// first in bag order; `None` when none is carried or it is locked.
pub(in crate::script::pfui_host) fn instance(lua: &Lua, input: Value) -> mlua::Result<Option<u64>> {
    let location = if let Value::Table(location) = &input {
        if let Some(slot) = location
            .get::<Option<u32>>("equipmentSlotIndex")
            .ok()
            .flatten()
        {
            Some((EQUIPMENT_BAG, slot))
        } else {
            location
                .get::<Option<i64>>("bagID")
                .ok()
                .flatten()
                .zip(location.get::<Option<u32>>("slotIndex").ok().flatten())
        }
    } else {
        None
    };
    let text = match &input {
        Value::String(text) => Some(text.to_string_lossy().to_string()),
        _ => None,
    };
    let guid = text
        .as_deref()
        .and_then(|s| s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")))
        .and_then(|s| u64::from_str_radix(s, 16).ok());
    let entry = id(lua, input)?;
    let extras = lua.app_data_ref::<Extras>().expect("items");
    let model = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model");
    let matches = |item: &Instance| {
        if let Some(guid) = guid {
            return item.guid == guid;
        }
        if let Some(entry) = entry {
            return item.item_id == entry;
        }
        let name = item.name.as_deref().or_else(|| {
            model
                .item_templates
                .get(&item.item_id)
                .map(|t| t.name.as_str())
        });
        text.as_deref()
            .zip(name)
            .is_some_and(|(wanted, name)| wanted.to_lowercase() == name.to_lowercase())
    };
    Ok(extras
        .instances
        .iter()
        .filter(|((bag, slot), item)| {
            carried(*bag)
                && *slot > 0
                && item.guid != 0
                && location.map_or_else(|| matches(item), |at| at == (*bag, *slot))
        })
        .min_by_key(|((bag, slot), _)| {
            (
                if *bag == EQUIPMENT_BAG {
                    6
                } else if *bag == -2 {
                    5
                } else {
                    *bag
                },
                *slot,
            )
        })
        .and_then(|(&(bag, slot), item)| {
            (!super::equipment::locked(&model, bag, slot)).then_some(item.guid)
        }))
}

pub(in crate::script::pfui_host) fn request_use(
    lua: &Lua,
    item: Value,
    unit: Option<String>,
) -> mlua::Result<()> {
    let Some(guid) = instance(lua, item)? else {
        return Ok(());
    };
    let target = if let Some(unit) = unit {
        let Some(guid) = super::super::units::guid(lua, &unit)? else {
            return Ok(());
        };
        Some(guid)
    } else {
        None
    };
    crate::script::calls::submit(
        lua,
        ScriptCall::PfuiClick(ClickAction::Item { guid, target }),
    )?;
    Ok(())
}

pub(super) fn install(lua: &Lua, api: &Table) -> mlua::Result<()> {
    api.set(
        "UseItemByName",
        lua.create_function(|lua, (item, unit): (Value, Option<String>)| {
            if let Some(unit) = &unit {
                if let crate::script::UnitTokenParse::Unit {
                    base: crate::script::UnitBase::Target,
                    hops,
                } = crate::script::parse_unit_token(unit)
                {
                    if let Some(guid) = instance(lua, item)? {
                        crate::script::calls::submit(
                            lua,
                            ScriptCall::PfuiClick(ClickAction::TargetItem { guid, hops }),
                        )?;
                    }
                    return Ok(());
                }
            }
            request_use(lua, item, unit)
        })?,
    )
}
