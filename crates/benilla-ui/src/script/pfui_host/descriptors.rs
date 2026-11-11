//! Small extension queries over action and equipment snapshots.

use crate::script::{action, Model};
use mlua::{Lua, MultiValue, Value};

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let globals = lua.globals();
    globals.set(
        "GetActionInfo",
        lua.create_function(|lua, slot: i64| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let action = u32::try_from(slot)
                .ok()
                .filter(|slot| (1..=120).contains(slot))
                .and_then(|slot| model.actions.get(&slot));
            let Some(action) = action.filter(|action| action.action != 0) else {
                return Ok(MultiValue::new());
            };
            let (kind, id, subtype) = match action.kind & 0xf0 {
                action::ACTION_KIND_SPELL => ("spell", Some(action.action), Some("spell")),
                action::ACTION_KIND_ITEM => ("item", Some(action.action), None),
                action::ACTION_KIND_MACRO => {
                    // The host stores macro payloads as 1-based slots, not client hash IDs.
                    if model.macros.get(action.action as usize).is_some() {
                        ("macro", Some(action.action), None)
                    } else {
                        ("item", None, None)
                    }
                }
                _ => return Ok(MultiValue::new()),
            };
            let mut values = vec![Value::String(lua.create_string(kind)?)];
            values.push(id.map_or(Value::Nil, |id| Value::Integer(i64::from(id))));
            if let Some(subtype) = subtype {
                values.push(Value::String(lua.create_string(subtype)?));
            }
            Ok(MultiValue::from_vec(values))
        })?,
    )?;
    globals.set(
        "GetInventoryItemID",
        lua.create_function(|lua, (token, slot): (String, i64)| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let player = crate::script::unit::guid_of(lua, &model, "player")?;
            let resolved = crate::script::unit::guid_of(lua, &model, &token)
                .ok()
                .flatten();
            let token = if player.is_some() && resolved == player {
                "player"
            } else {
                &token
            };
            Ok(usize::try_from(slot)
                .ok()
                .and_then(|slot| super::inv_slot(lua, &model, token, slot))
                .map(|slot| slot.item_id)
                .filter(|id| *id != 0))
        })?,
    )?;
    globals.set(
        "GetInventoryItemDurability",
        lua.create_function(|lua, slot: i64| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let durability = (1..=19)
                .contains(&slot)
                .then(|| model.inventory_slots[slot as usize].as_ref())
                .flatten()
                .and_then(|slot| slot.durability)
                .filter(|(_, max)| *max > 0);
            Ok(match durability {
                Some((current, max)) => MultiValue::from_vec(vec![
                    Value::Integer(i64::from(current)),
                    Value::Integer(i64::from(max)),
                ]),
                None => MultiValue::new(),
            })
        })?,
    )?;
    // The stock repair feed's cost for a worn slot, the tooltip's third value; 0 with nothing
    // to repair or outside the nineteen equipment slots.
    globals.set(
        "GetInventoryItemRepairCost",
        lua.create_function(|lua, slot: i64| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            Ok(if (1..=19).contains(&slot) {
                model.inv_repair_cost(lua, slot as usize)
            } else {
                0
            })
        })?,
    )?;
    // The inbox row's item as the basic entry link the stock builder writes, with its entry.
    globals.set(
        "GetInboxItemLink",
        lua.create_function(|lua, (index, attachment): (i64, Option<i64>)| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let row = usize::try_from(index)
                .ok()
                .and_then(|index| index.checked_sub(1))
                .filter(|_| attachment.is_none_or(|n| n == 1))
                .and_then(|index| model.mail.as_ref()?.inbox.get(index))
                .filter(|row| row.item_id != 0);
            let Some((row, name)) = row.and_then(|row| Some((row, row.item_name.as_deref()?)))
            else {
                return Ok((None, None));
            };
            Ok((
                Some(crate::script::item_stats::item_link(
                    row.item_id,
                    name,
                    row.item_quality.unwrap_or(1),
                )),
                Some(row.item_id),
            ))
        })?,
    )?;
    globals.set(
        "GetInboxItemID",
        lua.create_function(|lua, index: i64| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            Ok(usize::try_from(index)
                .ok()
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| model.mail.as_ref()?.inbox.get(index))
                .map(|row| row.item_id)
                .filter(|id| *id != 0))
        })?,
    )?;
    // The attached item is still the carried instance, so its own link is returned.
    globals.set(
        "GetSendMailItemLink",
        lua.create_function(|lua, attachment: Option<i64>| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let item = model
                .mail_send_item
                .as_ref()
                .filter(|_| attachment.is_none_or(|n| n == 1));
            Ok(
                match item.and_then(|item| Some((item.link.clone()?, item.item_id))) {
                    Some((link, id)) => (Some(link), Some(id)),
                    None => (None, None),
                },
            )
        })?,
    )?;
    Ok(())
}
