//! Item subjects follow the stock builders and clear/hide lifecycle, outside stock widget state.

use crate::script::{object::frame_handle_of, tooltip::REG_TOOLTIP_METHODS, Model};
use crate::widget::{FrameHandle, KindState};
use mlua::{Function, Lua, MultiValue, Table, Value};
use std::collections::HashMap;

#[derive(Clone)]
struct Item {
    id: u32,
    name: Option<String>,
    link: Option<String>,
}

#[derive(Default)]
struct Subjects {
    items: HashMap<FrameHandle, Item>,
    spells: HashMap<FrameHandle, u32>,
    pending: Vec<(FrameHandle, Option<Item>)>,
    epoch: u64,
}

pub(crate) fn clear(lua: &Lua, handle: FrameHandle) -> u64 {
    let Some(mut subjects) = lua.app_data_mut::<Subjects>() else {
        return 0;
    };
    subjects.items.remove(&handle);
    subjects.spells.remove(&handle);
    subjects.epoch = subjects.epoch.wrapping_add(1);
    subjects.epoch
}

/// Spell builders publish after their cleared callbacks, before showing the actual lines.
pub(crate) fn set_spell(lua: &Lua, handle: FrameHandle, id: u32) {
    if let Some(mut subjects) = lua.app_data_mut::<Subjects>() {
        subjects.items.remove(&handle);
        if id == 0 {
            subjects.spells.remove(&handle);
        } else {
            subjects.spells.insert(handle, id);
        }
    }
}

/// Restore the builder's subject after cleared callbacks, before its OnShow. A nested builder
/// supersedes its caller. A failed/empty builder hides and HasItem remains false.
pub(crate) fn resume(lua: &Lua, handle: FrameHandle, epoch: u64) {
    let Some(mut subjects) = lua.app_data_mut::<Subjects>() else {
        return;
    };
    if subjects.epoch != epoch {
        return;
    }
    let item = subjects
        .pending
        .last()
        .filter(|(h, _)| *h == handle)
        .and_then(|(_, item)| item.clone());
    if let Some(item) = item {
        subjects.items.insert(handle, item);
    }
}

fn shown_item(lua: &Lua, handle: FrameHandle) -> Option<Item> {
    let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
    if !matches!(model.arena.frame(handle).map(|f| &f.kind_state), Some(KindState::Tooltip(t)) if t.num_lines > 0)
    {
        return None;
    }
    lua.app_data_ref::<Subjects>()
        .and_then(|s| s.items.get(&handle).cloned())
}

fn number(args: &MultiValue, index: usize) -> Option<usize> {
    match args.get(index)? {
        Value::Integer(n) => usize::try_from(*n).ok(),
        Value::Number(n) if n.is_finite() && *n >= 0.0 => Some(*n as usize),
        _ => None,
    }
}

fn text(args: &MultiValue, index: usize) -> Option<String> {
    match args.get(index)? {
        Value::String(s) => Some(s.to_string_lossy()),
        _ => None,
    }
}

fn subject(lua: &Lua, verb: &str, args: &MultiValue) -> mlua::Result<Option<Item>> {
    let getter = match verb {
        "SetBagItem" => Some("GetContainerItemLink"),
        "SetInventoryItem" => Some("GetInventoryItemLink"),
        "SetQuestItem" => Some("GetQuestItemLink"),
        "SetQuestLogItem" => Some("GetQuestLogItemLink"),
        "SetTradePlayerItem" => Some("GetTradePlayerItemLink"),
        "SetTradeTargetItem" => Some("GetTradeTargetItemLink"),
        "SetLootItem" => Some("GetLootSlotLink"),
        "SetLootRollItem" => Some("GetLootRollItemLink"),
        "SetMerchantItem" => Some("GetMerchantItemLink"),
        "SetAuctionItem" => Some("GetAuctionItemLink"),
        _ => None,
    };
    let link: Option<String> = if verb == "SetHyperlink" {
        text(args, 1)
    } else if let Some(getter) = getter {
        let getter: Function = lua.globals().get(getter)?;
        // These link getters take at most two arguments; extra builder flags stay private.
        getter.call(MultiValue::from_vec(
            args.iter().skip(1).take(2).cloned().collect(),
        ))?
    } else {
        None
    };
    let id_from_link = link.as_deref().and_then(|link| {
        let raw = link
            .strip_prefix("item:")
            .or_else(|| link.split_once("|Hitem:").map(|(_, tail)| tail))?;
        raw.split([':', '|']).next()?.parse::<u32>().ok()
    });
    let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
    let index = number(args, 1);
    let row = index.and_then(|i| i.checked_sub(1));
    let fallback = match verb {
        "SetBagItem" => {
            let bag = args
                .get(1)
                .cloned()
                .and_then(|v| lua.coerce_integer(v).ok().flatten());
            bag.and_then(|bag| model.containers.get(&bag))
                .and_then(|bag| {
                    number(args, 2)
                        .and_then(|i| u32::try_from(i).ok())
                        .and_then(|i| bag.slots.get(&i))
                })
                .map(|s| s.item_id)
        }
        "SetInventoryItem" => text(args, 1)
            .and_then(|unit| number(args, 2).and_then(|i| super::inv_slot(lua, &model, &unit, i)))
            .map(|s| s.item_id),
        "SetQuestItem" => model
            .quest
            .as_ref()
            .and_then(|q| text(args, 1).and_then(|kind| crate::script::quest::item_vec(q, &kind)))
            .and_then(|items| {
                number(args, 2)
                    .and_then(|i| i.checked_sub(1))
                    .and_then(|i| items.get(i))
            })
            .map(|item| item.item_id),
        // The selected quest's record lists its `(item id, count)` pairs.
        "SetQuestLogItem" => model
            .selected_quest_record()
            .and_then(|q| match text(args, 1).as_deref() {
                Some("choice") => Some(&q.choices),
                Some("reward") => Some(&q.rewards),
                _ => None,
            })
            .and_then(|items| {
                number(args, 2)
                    .and_then(|i| i.checked_sub(1))
                    .and_then(|i| items.get(i))
            })
            .map(|&(item_id, _)| item_id),
        "SetMerchantItem" | "SetBuybackItem" => model.merchant.as_ref().and_then(|m| {
            let rows = if verb == "SetBuybackItem" {
                &m.buyback
            } else {
                &m.items
            };
            row.and_then(|i| rows.get(i)).map(|s| s.item_id)
        }),
        "SetLootItem" => model
            .loot
            .as_ref()
            .and_then(|l| row.and_then(|i| l.rows.get(i)))
            .and_then(|r| r.as_ref())
            .filter(|r| !r.is_coin)
            .map(|r| r.item_id),
        "SetLootRollItem" => index
            .and_then(|id| {
                model
                    .loot_rolls
                    .rolls
                    .iter()
                    .find(|r| r.roll_id as usize == id)
            })
            .map(|r| r.item_id),
        "SetTradePlayerItem" | "SetTradeTargetItem" => model.trade.as_ref().and_then(|t| {
            let side = if verb == "SetTradePlayerItem" {
                &t.player
            } else {
                &t.target
            };
            row.and_then(|i| side.slots.get(i))
                .and_then(|i| i.as_ref())
                .map(|i| i.item_id)
        }),
        "SetInboxItem" => model
            .mail
            .as_ref()
            .and_then(|m| row.and_then(|i| m.inbox.get(i)))
            .map(|m| m.item_id),
        "SetAuctionItem" => model
            .auction
            .as_ref()
            .and_then(|a| {
                text(args, 1)
                    .and_then(|kind| crate::script::auction::list_index_of(&kind))
                    .map(|i| &a.lists[i])
            })
            .and_then(|list| {
                number(args, 2)
                    .and_then(|i| i.checked_sub(1))
                    .and_then(|i| list.rows.get(i))
            })
            .map(|item| item.item_id),
        "SetAuctionSellItem" => model.auction_sell_item.as_ref().map(|i| i.item_id),
        "SetSendMailItem" => model.mail_send_item.as_ref().map(|i| i.item_id),
        "SetAction" => index
            .and_then(|i| u32::try_from(i).ok())
            .and_then(|i| model.actions.get(&i))
            .filter(|a| a.kind == 0x80)
            .map(|a| a.action),
        "SetTradeSkillItem" => index
            .and_then(|i| crate::script::tradeskill::recipe_at(&model, i))
            .and_then(|recipe| match number(args, 2) {
                Some(i) => i
                    .checked_sub(1)
                    .and_then(|i| recipe.reagents.get(i))
                    .map(|r| r.item),
                None => Some(recipe.product_item),
            }),
        "SetCraftItem" => model
            .craft
            .as_ref()
            .and_then(|c| row.and_then(|i| c.recipes.get(i)))
            .and_then(|r| {
                number(args, 2)
                    .and_then(|i| i.checked_sub(1))
                    .and_then(|i| r.reagents.get(i))
            })
            .map(|r| r.item),
        "SetTrainerService" => index
            .and_then(|i| crate::script::trainer::service(&model, i))
            .and_then(|service| match service.tooltip {
                crate::script::trainer::TrainerTooltip::Item(id) => Some(id),
                _ => None,
            }),
        "SetCraftSpell" => model
            .craft
            .as_ref()
            .and_then(|c| row.and_then(|i| c.recipes.get(i)))
            .and_then(|recipe| match recipe.tooltip {
                crate::script::craft::CraftTooltip::Item(id) => Some(id),
                _ => None,
            }),
        _ => None,
    };
    let Some(id) = id_from_link.or(fallback).filter(|id| *id != 0) else {
        return Ok(None);
    };
    let view = model.item_templates.get(&id);
    let name = link
        .as_deref()
        .and_then(|link| {
            link.split_once("|h[")?
                .1
                .split_once("]|h")
                .map(|(name, _)| name.to_owned())
        })
        .or_else(|| view.map(|view| view.name.clone()));
    // Instance links retain enchant/suffix/unique fields. Bare-ID sources use the cached
    // template's name; no cold item name is invented.
    let link = link.or_else(|| {
        view.map(|view| {
            format!(
                "|c{}|Hitem:{id}:0:0:0|h[{}]|h|r",
                super::items::quality_color(view.quality),
                view.name
            )
        })
    });
    Ok(Some(Item { id, name, link }))
}

/// Fire one of ClassicAPI's tooltip scripts on `handle`, its error recorded.
pub(super) fn fire_set(lua: &Lua, handle: FrameHandle, script: &str) {
    let id = lua.app_data_mut::<crate::script::ModelCell>().expect("model").frame_id(handle);
    if let Err(e) = crate::script::event::fire_widget_handler(lua, id, script, Vec::new()) {
        lua.app_data_mut::<crate::script::ModelCell>()
            .expect("model")
            .record_lua_error(&e);
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Subjects::default());
    let methods: Table = lua.named_registry_value(REG_TOOLTIP_METHODS)?;
    let original: Function = methods.get("SetUnit")?;
    methods.set(
        "SetUnit",
        lua.create_function(move |lua, args: MultiValue| {
            let this = args.front().cloned();
            let result = super::units::call(lua, &original, args)?;
            // ClassicAPI's `OnTooltipSetUnit`, once the unit is drawn.
            if let Some(Value::Table(this)) = this {
                fire_set(lua, frame_handle_of(lua, &this)?, "OnTooltipSetUnit");
            }
            Ok(result)
        })?,
    )?;
    for verb in [
        "SetBagItem",
        "SetInventoryItem",
        "SetHyperlink",
        "SetQuestItem",
        "SetQuestLogItem",
        "SetTradePlayerItem",
        "SetTradeTargetItem",
        "SetLootItem",
        "SetLootRollItem",
        "SetMerchantItem",
        "SetBuybackItem",
        "SetInboxItem",
        "SetAuctionItem",
        "SetAuctionSellItem",
        "SetSendMailItem",
        "SetAction",
        "SetTradeSkillItem",
        "SetCraftItem",
        "SetTrainerService",
        "SetCraftSpell",
    ] {
        let original: Function = methods.get(verb)?;
        methods.set(
            verb,
            lua.create_function(move |lua, args: MultiValue| {
                let this: Table =
                    mlua::FromLua::from_lua(args.front().cloned().unwrap_or(Value::Nil), lua)?;
                // A ClassicAPI client opens a profession link in its window instead of drawing it.
                if verb == "SetHyperlink" {
                    if let Some(link) = text(&args, 1).filter(|link| link.starts_with("trade:")) {
                        super::professions::open_link(lua, &link)?;
                        this.get::<Function>("Hide")?.call::<()>(this.clone())?;
                        return Ok(MultiValue::new());
                    }
                }
                let handle = frame_handle_of(lua, &this)?;
                let item = subject(lua, verb, &args)?;
                lua.app_data_mut::<Subjects>()
                    .expect("subjects")
                    .pending
                    .push((handle, item));
                let result = original.call::<MultiValue>(args);
                lua.app_data_mut::<Subjects>()
                    .expect("subjects")
                    .pending
                    .pop();
                // ClassicAPI's `OnTooltipSetItem`, once the setter drew an item.
                if result.is_ok() && shown_item(lua, handle).is_some() {
                    fire_set(lua, handle, "OnTooltipSetItem");
                }
                result
            })?,
        )?;
    }
    for name in ["HasItem", "GetItem"] {
        methods.set(
            name,
            lua.create_function(move |lua, this: Table| {
                let item = shown_item(lua, frame_handle_of(lua, &this)?);
                if name == "HasItem" {
                    return Ok(MultiValue::from_vec(vec![Value::Boolean(item.is_some())]));
                }
                let Some(item) = item else {
                    return Ok(MultiValue::new());
                };
                let value = |s: Option<String>| {
                    s.map(|s| lua.create_string(s).map(Value::String))
                        .unwrap_or(Ok(Value::Nil))
                };
                Ok(MultiValue::from_vec(vec![
                    value(item.name)?,
                    value(item.link)?,
                    Value::Integer(i64::from(item.id)),
                ]))
            })?,
        )?;
    }
    for name in ["HasSpell", "GetSpell"] {
        methods.set(
            name,
            lua.create_function(move |lua, this: Table| {
                let handle = frame_handle_of(lua, &this)?;
                let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
                let has_lines = matches!(model.arena.frame(handle).map(|f| &f.kind_state),
                    Some(KindState::Tooltip(t)) if t.num_lines > 0);
                let id = lua
                    .app_data_ref::<Subjects>()
                    .and_then(|s| s.spells.get(&handle).copied())
                    .filter(|_| has_lines);
                if name == "HasSpell" {
                    return Ok(MultiValue::from_vec(vec![Value::Boolean(id.is_some())]));
                }
                let Some(id) = id else {
                    return Ok(MultiValue::new());
                };
                let view = model
                    .spell_tooltips
                    .get(&id)
                    .or_else(|| model.pet_spell_tooltips.get(&id));
                let fallback = view.map(|view| (view.name.clone(), view.rank.clone()));
                drop(model);
                let metadata: Option<Table> = lua.globals().get("C_Spell")?;
                let info = metadata
                    .and_then(|api| api.get::<Option<Function>>("GetSpellInfo").ok().flatten())
                    .map(|get| get.call::<Option<Table>>(id))
                    .transpose()?
                    .flatten();
                let (spell_name, rank) = if let Some(info) = info {
                    (
                        info.get::<String>("name")?,
                        info.get::<Option<String>>("rank")?.unwrap_or_default(),
                    )
                } else if let Some((name, rank)) = fallback {
                    (name, rank.unwrap_or_default())
                } else {
                    return Ok(MultiValue::new());
                };
                Ok(MultiValue::from_vec(vec![
                    Value::String(lua.create_string(spell_name)?),
                    Value::String(lua.create_string(rank)?),
                    Value::Integer(i64::from(id)),
                ]))
            })?,
        )?;
    }
    for name in ["HasUnit", "GetUnitGUID"] {
        methods.set(
            name,
            lua.create_function(move |lua, this: Table| {
                let handle = frame_handle_of(lua, &this)?;
                let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
                let tooltip = match model.arena.frame(handle).map(|f| &f.kind_state) {
                    Some(KindState::Tooltip(t)) => Some(t),
                    _ => None,
                };
                let guid = tooltip.and_then(|t| t.unit_guid).filter(|guid| *guid != 0);
                if name == "HasUnit" {
                    return Ok(MultiValue::from_vec(vec![Value::Boolean(guid.is_some())]));
                }
                let Some(guid) = guid else {
                    return Ok(MultiValue::new());
                };
                let name = tooltip
                    .and_then(|t| t.left_lines.first())
                    .and_then(|h| model.region_data.get(h))
                    .and_then(|r| r.text.as_deref());
                Ok(MultiValue::from_vec(vec![
                    name.map(|name| lua.create_string(name).map(Value::String))
                        .unwrap_or(Ok(Value::Nil))?,
                    Value::String(lua.create_string(format!("0x{guid:016X}"))?),
                ]))
            })?,
        )?;
    }
    Ok(())
}
