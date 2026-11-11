//! Queries and operations over the stock item cache and inventory-operation queue.

use crate::script::{ContainerMove, ItemTemplateView, Model};
use mlua::{Lua, MultiValue, Value};
use std::collections::HashMap;

#[path = "equipment.rs"]
mod equipment;

#[path = "new_items.rs"]
mod new_items;

#[path = "inventory_events.rs"]
mod events;

#[path = "state_events.rs"]
mod state_events;

#[path = "item_actions.rs"]
mod actions;
pub(super) use actions::{instance, request_use};

/// The host calls this once after the stock inventory feeds have delivered their events.
pub fn dispatch_inventory_events(lua: &Lua) {
    events::dispatch(lua);
    state_events::dispatch(lua);
}

/// Public-slot GUIDs come directly from the player descriptor, even before item objects arrive.
pub fn observe_equipment_guids(lua: &Lua, guids: [u64; 19]) {
    events::observe_equipment(lua, guids);
}

/// Modern atomic bag operations are paced independently of the stock cursor queue.
pub fn take_bag_moves(lua: &Lua) -> Vec<ContainerMove> {
    lua.app_data_mut::<Extras>()
        .map(|mut extras| std::mem::take(&mut extras.moves))
        .unwrap_or_default()
}

pub fn set_bag_busy(lua: &Lua, busy: bool) {
    events::set_bag_busy(lua, busy);
}

pub fn bag_position_available(lua: &Lua, bag: i64, slot: u32) -> bool {
    let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
    if (bag == -1 || (5..=10).contains(&bag)) && model.bank.is_none() {
        return false;
    }
    model
        .containers
        .get(&bag)
        .is_some_and(|bag| slot > 0 && slot <= bag.num_slots)
}

pub fn bag_failure(lua: &Lua, guid: u64) {
    if let Some(mut extras) = lua.app_data_mut::<Extras>() {
        extras.failures.push(guid);
    }
}

pub fn take_bag_failures(lua: &Lua) -> Vec<u64> {
    lua.app_data_mut::<Extras>()
        .map(|mut extras| std::mem::take(&mut extras.failures))
        .unwrap_or_default()
}

#[derive(Default)]
pub(super) struct Extras {
    metadata: HashMap<u32, (u32, u32)>,
    /// Each template's first on-use spell (`GetItemSpell`).
    use_spells: HashMap<u32, u32>,
    pub(super) instances: HashMap<(i64, u32), Instance>,
    moves: Vec<ContainerMove>,
    failures: Vec<u64>,
}

/// One real inventory object, including bank objects received at login.
pub struct Instance {
    pub guid: u64,
    pub item_id: u32,
    pub count: u32,
    pub uses: u32,
    pub name: Option<String>,
}

/// Publish wire-only metadata before the stock template-arrival event.
pub fn set_metadata(lua: &Lua, id: u32, level: u32, family: u32) {
    if let Some(mut extras) = lua.app_data_mut::<Extras>() {
        extras.metadata.insert(id, (level, family));
    }
}

/// Replace the instance snapshot before inventory events. Slots use the Lua bag numbering.
pub fn set_instances(lua: &Lua, slots: impl IntoIterator<Item = ((i64, u32), Instance)>) {
    if let Some(mut extras) = lua.app_data_mut::<Extras>() {
        extras.instances.clear();
        extras.instances.extend(slots);
    }
    new_items::update(lua);
}

/// Publish an item template's first on-use (trigger 0) spell, which `GetItemSpell` answers.
pub fn set_use_spell(lua: &Lua, id: u32, spell_id: u32) {
    if let Some(mut extras) = lua.app_data_mut::<Extras>() {
        extras.use_spells.insert(id, spell_id);
    }
}

/// The on-use spell of `item`: an item id, an `item:` string or link, or the name of an item the
/// player carries; `None` for an item without one or not cached.
pub fn use_spell(lua: &Lua, item: Value) -> mlua::Result<Option<u32>> {
    let name = match &item {
        Value::String(text) => Some(text.to_string_lossy().to_lowercase()),
        _ => None,
    };
    let extras = lua.app_data_ref::<Extras>().expect("pfUI item state");
    let id = id(lua, item)?.or_else(|| {
        extras
            .instances
            .values()
            .find(|instance| {
                instance
                    .name
                    .as_ref()
                    .zip(name.as_ref())
                    .is_some_and(|(have, wanted)| have.to_lowercase() == *wanted)
            })
            .map(|instance| instance.item_id)
    });
    Ok(id.and_then(|id| extras.use_spells.get(&id).copied()))
}

fn metadata(lua: &Lua, id: u32) -> Option<(u32, u32)> {
    lua.app_data_ref::<Extras>()?.metadata.get(&id).copied()
}

fn id(lua: &Lua, value: Value) -> mlua::Result<Option<u32>> {
    if let Some(number) = lua.coerce_number(value.clone())? {
        return Ok(
            (number.is_finite() && number >= 1.0 && number <= f64::from(u32::MAX))
                .then_some(number as u32),
        );
    }
    if let Value::String(text) = value {
        let text = text.to_string_lossy().to_ascii_lowercase();
        if let Some((_, tail)) = text.split_once("item:") {
            return Ok(tail
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .and_then(|n| n.parse::<u32>().ok())
                .filter(|n| *n > 0));
        }
    }
    Ok(None)
}

fn location_id(lua: &Lua, value: Value) -> mlua::Result<Option<u32>> {
    if let Value::Table(location) = value {
        let bag: Option<i64> = location.get("bagID")?;
        let slot: Option<u32> = location.get("slotIndex")?;
        let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
        if let (Some(bag), Some(slot)) = (bag, slot) {
            return Ok(model
                .containers
                .get(&bag)
                .and_then(|bag| bag.slots.get(&slot))
                .map(|s| s.item_id)
                .filter(|id| *id != 0));
        }
        let equip: Option<usize> = location.get("equipmentSlotIndex")?;
        return Ok(equip
            .and_then(|i| model.inventory_slots.get(i))
            .and_then(|v| v.as_ref())
            .map(|v| v.item_id));
    }
    id(lua, value)
}

/// The VM's copy of item `id`'s record; a miss with `request` asks for it inside the getter's call,
/// and its landing signals `GET_ITEM_INFO_RECEIVED`.
fn cached(lua: &Lua, id: Option<u32>, request: bool) -> Option<ItemTemplateView> {
    let id = id?;
    let result = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model")
        .item_templates
        .get(&id)
        .cloned();
    if result.is_none() && request {
        let _ = super::super::calls::ask(
            lua,
            super::super::RecordAsk::Item {
                entry: id,
                owner: 0,
                callback: super::super::ItemCallback::Extension,
            },
        );
    }
    result
}

fn equip_token(value: u32) -> &'static str {
    [
        "",
        "INVTYPE_HEAD",
        "INVTYPE_NECK",
        "INVTYPE_SHOULDER",
        "INVTYPE_BODY",
        "INVTYPE_CHEST",
        "INVTYPE_WAIST",
        "INVTYPE_LEGS",
        "INVTYPE_FEET",
        "INVTYPE_WRIST",
        "INVTYPE_HAND",
        "INVTYPE_FINGER",
        "INVTYPE_TRINKET",
        "INVTYPE_WEAPON",
        "INVTYPE_SHIELD",
        "INVTYPE_RANGED",
        "INVTYPE_CLOAK",
        "INVTYPE_2HWEAPON",
        "INVTYPE_BAG",
        "INVTYPE_TABARD",
        "INVTYPE_ROBE",
        "INVTYPE_WEAPONMAINHAND",
        "INVTYPE_WEAPONOFFHAND",
        "INVTYPE_HOLDABLE",
        "INVTYPE_AMMO",
        "INVTYPE_THROWN",
        "INVTYPE_RANGEDRIGHT",
        "INVTYPE_QUIVER",
        "INVTYPE_RELIC",
    ]
    .get(value as usize)
    .copied()
    .unwrap_or("")
}

fn string(lua: &Lua, value: Option<&str>) -> mlua::Result<Value> {
    value
        .map(|s| lua.create_string(s).map(Value::String))
        .unwrap_or(Ok(Value::Nil))
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Extras::default());
    new_items::install(lua)?;
    equipment::install(lua)?;
    events::install(lua)?;
    state_events::install(lua);
    let api = lua.create_table()?;
    actions::install(lua, &api)?;
    api.set("GetItemID", lua.create_function(location_id)?)?;
    for name in [
        "GetItemName",
        "GetItemNameByID",
        "GetItemQuality",
        "GetItemQualityByID",
        "GetItemInventoryType",
        "GetItemInventoryTypeByID",
        "GetItemMaxStackSizeByID",
        "GetItemSellPriceByID",
        "GetCurrentItemLevel",
        "GetItemFamily",
    ] {
        api.set(
            name,
            lua.create_function(move |lua, input: Value| {
                let id = location_id(lua, input)?;
                let Some(item) = cached(lua, id, true) else {
                    return Ok(Value::Nil);
                };
                let value = match name {
                    "GetItemName" | "GetItemNameByID" => return string(lua, Some(&item.name)),
                    "GetItemQuality" | "GetItemQualityByID" => item.quality,
                    "GetItemInventoryType" | "GetItemInventoryTypeByID" => item.inventory_type,
                    "GetItemMaxStackSizeByID" => item.stackable,
                    "GetItemSellPriceByID" => item.sell_price,
                    "GetCurrentItemLevel" => match id.and_then(|id| metadata(lua, id)) {
                        Some((level, _)) => level,
                        None => return Ok(Value::Nil),
                    },
                    "GetItemFamily" => {
                        let Some((_, family)) = id.and_then(|id| metadata(lua, id)) else {
                            return Ok(Value::Nil);
                        };
                        if family == 0 {
                            0
                        } else {
                            1u32.checked_shl(family - 1).unwrap_or(0)
                        }
                    }
                    _ => unreachable!(),
                };
                Ok(Value::Integer(i64::from(value)))
            })?,
        )?;
    }
    api.set(
        "GetItemCount",
        lua.create_function(
            |lua, (input, bank, uses): (Value, Option<bool>, Option<bool>)| {
                let item_id = id(lua, input.clone())?;
                let name = match input {
                    Value::String(s) if item_id.is_none() => {
                        Some(s.to_string_lossy().to_lowercase())
                    }
                    _ => None,
                };
                let extras = lua.app_data_ref::<Extras>().expect("pfUI item state");
                Ok(extras
                    .instances
                    .iter()
                    .filter(|((bag, _), item)| {
                        (bank == Some(true) || !(*bag == -1 || (5..=10).contains(bag)))
                            && (item_id == Some(item.item_id)
                                || name.as_ref().is_some_and(|n| {
                                    item.name.as_ref().is_some_and(|s| s.to_lowercase() == *n)
                                }))
                    })
                    .map(|(_, item)| {
                        if uses == Some(true) {
                            u64::from(item.uses)
                        } else {
                            u64::from(item.count)
                        }
                    })
                    .sum::<u64>())
            },
        )?,
    )?;
    api.set(
        "IsItemDataCachedByID",
        lua.create_function(|lua, input: Value| Ok(cached(lua, id(lua, input)?, false).is_some()))?,
    )?;
    api.set(
        "IsEquippedItem",
        lua.create_function(|lua, input: Value| {
            let id = id(lua, input)?;
            Ok(id.is_some_and(|id| {
                lua.app_data_ref::<crate::script::ModelCell>().expect("model").inventory_slots[1..20]
                    .iter()
                    .flatten()
                    .any(|slot| slot.item_id == id)
            }))
        })?,
    )?;
    api.set(
        "GetItemUniquenessByID",
        lua.create_function(|lua, input: Value| {
            let Some(item) = cached(lua, id(lua, input)?, true) else {
                return Ok(MultiValue::new());
            };
            Ok(MultiValue::from_vec(vec![
                Value::Boolean(item.max_count > 0),
                Value::Nil,
                if item.max_count > 0 {
                    Value::Integer(i64::from(item.max_count))
                } else {
                    Value::Nil
                },
                Value::Nil,
            ]))
        })?,
    )?;
    api.set(
        "GetItemInfoInstant",
        lua.create_function(|lua, input: Value| {
            let Some(id) = id(lua, input)? else {
                return Ok(MultiValue::new());
            };
            let item = cached(lua, Some(id), false);
            let mut result = vec![Value::Integer(i64::from(id))];
            if let Some(item) = item {
                result.extend([
                    string(lua, item.item_type.as_deref())?,
                    string(lua, item.item_sub_type.as_deref())?,
                    string(lua, Some(equip_token(item.inventory_type)))?,
                    string(lua, item.icon.as_deref())?,
                    Value::Integer(i64::from(item.class)),
                    Value::Integer(i64::from(item.subclass)),
                ]);
            } else {
                result.resize(7, Value::Nil);
            }
            Ok(MultiValue::from_vec(result))
        })?,
    )?;
    api.set(
        "GetItemInfo",
        lua.create_function(|lua, input: Value| {
            let Some(id) = id(lua, input)? else {
                return Ok(MultiValue::new());
            };
            let Some(v) = cached(lua, Some(id), true) else {
                return Ok(MultiValue::new());
            };
            let link = format!(
                "|c{}|Hitem:{id}:0:0:0|h[{}]|h|r",
                quality_color(v.quality),
                v.name
            );
            Ok(MultiValue::from_vec(vec![
                string(lua, Some(&v.name))?,
                string(lua, Some(&link))?,
                Value::Integer(i64::from(v.quality)),
                metadata(lua, id).map_or(Value::Nil, |(level, _)| Value::Integer(i64::from(level))),
                Value::Integer(i64::from(v.required_level)),
                string(lua, v.item_type.as_deref())?,
                string(lua, v.item_sub_type.as_deref())?,
                Value::Integer(i64::from(v.stackable)),
                string(lua, Some(equip_token(v.inventory_type)))?,
                string(lua, v.icon.as_deref())?,
                Value::Integer(i64::from(v.sell_price)),
                Value::Integer(i64::from(v.class)),
                Value::Integer(i64::from(v.subclass)),
                Value::Integer(i64::from(v.bonding)),
                Value::Integer(0),
                if v.item_set == 0 {
                    Value::Nil
                } else {
                    Value::Integer(i64::from(v.item_set))
                },
                Value::Boolean(false),
                string(lua, Some(&v.description))?,
            ]))
        })?,
    )?;
    lua.globals().set("C_Item", api)?;
    let player = lua.create_table()?;
    player.set(
        "CanUseItem",
        lua.create_function(|lua, input: Value| {
            let Some(item) = cached(lua, id(lua, input)?, true) else {
                return Ok(false);
            };
            let book = super::player_book(lua);
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            Ok(crate::script::item_stats::item_usable(
                &item,
                &model.player_req,
                |id| book.slots.iter().any(|s| s.spell_id == id),
            ))
        })?,
    )?;
    lua.globals().set("C_PlayerInfo", player)?;
    let container = lua.create_table()?;
    container.set(
        "GetContainerItemID",
        lua.create_function(|lua, (bag, slot): (i64, u32)| {
            Ok(lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model")
                .containers
                .get(&bag)
                .and_then(|bag| bag.slots.get(&slot))
                .map(|slot| slot.item_id)
                .filter(|id| *id > 0))
        })?,
    )?;
    container.set(
        "IsContainerItemOpenable",
        lua.create_function(|lua, (bag, slot): (i64, u32)| {
            let instance = lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model")
                .containers
                .get(&bag)
                .and_then(|b| b.slots.get(&slot))
                .cloned();
            let Some(instance) = instance else {
                return Ok((None, None));
            };
            let Some(item) = cached(lua, Some(instance.item_id), true) else {
                return Ok((None, None));
            };
            let openable = item.flags & 4 != 0;
            Ok((
                Some(openable),
                Some(openable && (item.lock_id == 0 || instance.flags & 4 != 0)),
            ))
        })?,
    )?;
    container.set(
        "GetContainerItemCharges",
        lua.create_function(|lua, (bag, slot): (i64, u32)| {
            let uses = lua
                .app_data_ref::<Extras>()
                .and_then(|extras| extras.instances.get(&(bag, slot)).map(|item| item.uses));
            if uses.is_some() {
                return Ok(uses);
            }
            // nil for an empty slot; 0 for an item whose instance has not been fed yet, which the
            // stock getter already shows.
            let info: mlua::Function = lua.globals().get("GetContainerItemInfo")?;
            let texture: Value = info.call((bag, slot))?;
            Ok((!texture.is_nil()).then_some(0))
        })?,
    )?;
    for name in ["SwapItems", "MoveItem"] {
        container.set(
            name,
            lua.create_function(
                move |lua,
                      (src_bag, src_slot, dst_bag, dst_slot, count): (
                    i64,
                    u32,
                    i64,
                    u32,
                    Option<u32>,
                )| {
                    if name == "MoveItem" && !count.is_some_and(|c| c > 0 && c <= 255) {
                        return Ok(false);
                    }
                    let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
                    if (src_bag == -1
                        || dst_bag == -1
                        || (5..=10).contains(&src_bag)
                        || (5..=10).contains(&dst_bag))
                        && model.bank.is_none()
                    {
                        return Ok(false);
                    }
                    let Some(src) = model
                        .containers
                        .get(&src_bag)
                        .and_then(|b| b.slots.get(&src_slot))
                    else {
                        return Ok(false);
                    };
                    if crate::script::cursor::item_locked(&model, src.guid) || src.count == 0 {
                        return Ok(false);
                    }
                    let Some(dst) = model.containers.get(&dst_bag) else {
                        return Ok(false);
                    };
                    if dst_slot == 0
                        || dst_slot > dst.num_slots
                        || dst
                            .slots
                            .get(&dst_slot)
                            .is_some_and(|s| crate::script::cursor::item_locked(&model, s.guid))
                    {
                        return Ok(false);
                    }
                    if src_bag == dst_bag && src_slot == dst_slot {
                        return Ok(false);
                    }
                    // A sort models later swaps over its virtual grid, so validate positions
                    // here and re-read counts/locks when each operation actually sends.
                    drop(model);
                    let mut extras = lua.app_data_mut::<Extras>().expect("item extras");
                    if extras.moves.len() >= 512 {
                        return Err(mlua::Error::runtime("pfUI bag operation queue is full"));
                    }
                    extras.moves.push(ContainerMove {
                        src_bag,
                        src_slot,
                        dst_bag,
                        dst_slot,
                        count: if name == "MoveItem" { count } else { None },
                    });
                    drop(extras);
                    events::set_bag_busy(lua, true);
                    Ok(true)
                },
            )?,
        )?;
    }
    lua.globals().set("C_Container", container)
}

pub(super) fn quality_color(quality: u32) -> &'static str {
    [
        "ff9d9d9d", "ffffffff", "ff1eff00", "ff0070dd", "ffa335ee", "ffff8000", "ffe6cc80",
    ]
    .get(quality as usize)
    .copied()
    .unwrap_or("ffffffff")
}
