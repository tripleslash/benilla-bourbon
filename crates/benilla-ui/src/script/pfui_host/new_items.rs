//! Bag glow follows instance GUIDs; a login baseline excludes already-owned items.

use super::{Extras, Model};
use mlua::{Lua, Value};
use std::collections::HashSet;

#[derive(Default)]
struct NewItems {
    owner: u64,
    baseline_until: f64,
    resident: HashSet<u64>,
    flagged: HashSet<u64>,
}

fn notify(lua: &Lua) {
    crate::script::pfui_host::fire_pfui_event(lua, "BAG_NEW_ITEMS_UPDATED", vec![]);
}

pub(super) fn update(lua: &Lua) {
    if lua.app_data_ref::<NewItems>().is_none() {
        return;
    }
    let (owner, now) = {
        let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
        (model.unit_guids.player, model.now)
    };
    let resident: HashSet<_> = lua
        .app_data_ref::<Extras>()
        .expect("items")
        .instances
        .iter()
        .filter(|((bag, _), item)| (0..=4).contains(bag) && item.guid != 0)
        .map(|(_, item)| item.guid)
        .collect();
    let changed = {
        let mut items = lua.app_data_mut::<NewItems>().expect("new items");
        if items.owner != owner {
            items.owner = owner;
            items.baseline_until = now + 1.5;
            items.resident.clear();
            items.flagged.clear();
        }
        if owner == 0 || now < items.baseline_until {
            items.resident = resident;
            return;
        }
        let old = items.flagged.clone();
        items.flagged.retain(|guid| resident.contains(guid));
        let acquired: Vec<_> = resident.difference(&items.resident).copied().collect();
        items.flagged.extend(acquired);
        items.resident = resident;
        old != items.flagged
    };
    if changed {
        notify(lua);
    }
}

fn guid(lua: &Lua, bag: Value, slot: Value) -> Option<u64> {
    let bag = lua.coerce_number(bag).ok().flatten()?;
    let slot = lua.coerce_number(slot).ok().flatten()?;
    if !bag.is_finite()
        || !slot.is_finite()
        || !(0.0..=4.0).contains(&bag)
        || slot < 1.0
        || slot > f64::from(u32::MAX)
    {
        return None;
    }
    lua.app_data_ref::<Extras>()?
        .instances
        .get(&(bag as i64, slot as u32))
        .map(|item| item.guid)
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(NewItems::default());
    let api = lua.create_table()?;
    api.set(
        "IsNewItem",
        lua.create_function(|lua, (bag, slot): (Value, Value)| {
            Ok(guid(lua, bag, slot).is_some_and(|guid| {
                lua.app_data_ref::<NewItems>()
                    .expect("new items")
                    .flagged
                    .contains(&guid)
            }))
        })?,
    )?;
    api.set(
        "RemoveNewItem",
        lua.create_function(|lua, (bag, slot): (Value, Value)| {
            let changed = guid(lua, bag, slot).is_some_and(|guid| {
                lua.app_data_mut::<NewItems>()
                    .expect("new items")
                    .flagged
                    .remove(&guid)
            });
            if changed {
                notify(lua);
            }
            Ok(())
        })?,
    )?;
    api.set(
        "ClearAll",
        lua.create_function(|lua, ()| {
            let changed = {
                let mut items = lua.app_data_mut::<NewItems>().expect("new items");
                let changed = !items.flagged.is_empty();
                items.flagged.clear();
                changed
            };
            if changed {
                notify(lua);
            }
            Ok(())
        })?,
    )?;
    lua.globals().set("C_NewItems", api)
}
