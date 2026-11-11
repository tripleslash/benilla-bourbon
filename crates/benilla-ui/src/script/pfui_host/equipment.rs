//! Equipment operations select live item GUIDs and enter the existing atomic swap queue.

use super::{Extras, Model};
use crate::script::{ContainerMove, ScriptValue, EQUIPMENT_BAG};
use mlua::{Lua, MultiValue, Value};

pub(super) fn locked(model: &Model, bag: i64, slot: u32) -> bool {
    if bag == EQUIPMENT_BAG {
        model
            .inventory_slots
            .get(slot as usize)
            .and_then(|s| s.as_ref())
            .is_some_and(|s| crate::script::cursor::item_locked(model, s.guid) || s.flags & 4 != 0)
    } else {
        model
            .containers
            .get(&bag)
            .and_then(|b| b.slots.get(&slot))
            .is_some_and(|s| crate::script::cursor::item_locked(model, s.guid) || s.flags & 4 != 0)
    }
}

fn location(bag: i64, slot: u32) -> Option<u32> {
    if slot == 0 || slot > 255 {
        return None;
    }
    Some(match bag {
        EQUIPMENT_BAG if slot <= 19 => 0x0010_0000 | slot,
        0..=4 => 0x0030_0000 | (bag as u32 * 256) | slot,
        -1 => 0x0040_0000 | slot,
        5..=10 => 0x0060_0000 | ((bag as u32 - 4) * 256) | slot,
        _ => return None,
    })
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let bridge = lua.create_table()?;
    bridge.set(
        "tooltips",
        lua.named_registry_value::<mlua::Table>(crate::script::tooltip::REG_TOOLTIP_METHODS)?,
    )?;
    bridge.set(
        "snapshot",
        lua.create_function(|lua, ()| {
            let extras = lua.app_data_ref::<Extras>().expect("items");
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let snapshot = lua.create_table()?;
            for (&(bag, slot), item) in &extras.instances {
                let Some(code) = location(bag, slot) else {
                    continue;
                };
                if item.guid == 0 {
                    continue;
                }
                let row = lua.create_table()?;
                row.set("guid", format!("0x{:016X}", item.guid))?;
                row.set("itemID", item.item_id)?;
                row.set(
                    "name",
                    item.name.as_deref().or_else(|| {
                        model
                            .item_templates
                            .get(&item.item_id)
                            .map(|item| item.name.as_str())
                    }),
                )?;
                row.set("bag", bag)?;
                row.set("slot", slot)?;
                row.set("location", code)?;
                row.set("locked", locked(&model, bag, slot))?;
                snapshot.push(row)?;
            }
            Ok(snapshot)
        })?,
    )?;
    bridge.set(
        "swap",
        lua.create_function(|lua, (guid, target): (String, u32)| {
            if !(1..=19).contains(&target) {
                return Ok(false);
            }
            let guid = guid
                .strip_prefix("0x")
                .and_then(|text| u64::from_str_radix(text, 16).ok());
            let Some(guid) = guid.filter(|guid| *guid != 0) else {
                return Ok(false);
            };
            let source = {
                let extras = lua.app_data_ref::<Extras>().expect("items");
                extras
                    .instances
                    .iter()
                    .find(|(_, item)| item.guid == guid)
                    .map(|(&(bag, slot), _)| (bag, slot))
            };
            let Some((bag, slot)) = source else {
                return Ok(false);
            };
            if !(bag == EQUIPMENT_BAG || (0..=4).contains(&bag))
                || (bag == EQUIPMENT_BAG && slot == target)
            {
                return Ok(false);
            }
            {
                let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
                if locked(&model, bag, slot) || locked(&model, EQUIPMENT_BAG, target) {
                    return Ok(false);
                }
            }
            // The stock move, applied in the call.
            crate::script::calls::submit(
                lua,
                crate::script::ScriptCall::Item(crate::script::ItemCall::Move(ContainerMove {
                    src_bag: bag,
                    src_slot: slot,
                    dst_bag: EQUIPMENT_BAG,
                    dst_slot: target,
                    count: None,
                })),
            )?;
            Ok(true)
        })?,
    )?;
    bridge.set(
        "event",
        lua.create_function(|lua, args: MultiValue| {
            let mut args = args.into_iter();
            let Some(Value::String(event)) = args.next() else {
                return Err(mlua::Error::runtime("equipment event requires a name"));
            };
            let mut payload = Vec::new();
            for value in args {
                payload.push(match value {
                    Value::Integer(value) => ScriptValue::Int(value),
                    Value::Number(value) => ScriptValue::Number(value),
                    Value::String(value) => ScriptValue::Str(value.to_string_lossy()),
                    Value::Boolean(value) => ScriptValue::Bool(value),
                    Value::Nil => ScriptValue::Nil,
                    _ => return Err(mlua::Error::runtime("invalid equipment event payload")),
                });
            }
            crate::script::pfui_host::fire_pfui_event(lua, &event.to_string_lossy(), payload);
            Ok(())
        })?,
    )?;
    super::super::equipment_actions::install(lua, &bridge)?;
    lua.set_named_registry_value("benilla.pfui.equipment_bridge", bridge)
}
