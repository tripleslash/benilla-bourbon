//! Quest details are a pure cache probe over server templates, separate from the visible log.

use crate::script::ScriptValue;
use mlua::{Lua, Table};
use std::collections::HashMap;

#[derive(Clone, Default)]
pub struct Requirement {
    pub kind: String,
    pub id: u32,
    pub count: u32,
    pub text: String,
}

#[derive(Clone, Default)]
pub struct Detail {
    pub quest_id: u32,
    pub title: String,
    pub level: u32,
    pub tag: Option<String>,
    pub money: i32,
    pub money_max_level: u32,
    pub reward_spell: u32,
    pub src_item: u32,
    pub flags: u32,
    pub description: String,
    pub objectives: String,
    pub completion: String,
    pub poi: Option<(u32, f32, f32, u32)>,
    pub rewards: Vec<(u32, u32)>,
    pub choices: Vec<(u32, u32)>,
    pub requirements: Vec<Requirement>,
}

#[derive(Default)]
struct Cache(HashMap<u32, Detail>);

#[derive(Default)]
struct Lifecycle {
    slots: Option<Vec<(u32, u8)>>,
    accepted: Vec<u32>,
    removed: Vec<u32>,
}

/// Compare authoritative descriptor slots, including quests under collapsed headers.
/// Initial login and reload snapshots are silent; a delayed template is not a new accept.
pub fn observe_log_slots(lua: &Lua, slots: Vec<(u32, u8)>) {
    let Some(mut life) = lua.app_data_mut::<Lifecycle>() else {
        return;
    };
    if let Some(prior) = life.slots.take() {
        for &(id, _) in &prior {
            if id != 0 && !slots.iter().any(|&(now, _)| now == id) {
                life.accepted.retain(|pending| *pending != id);
                life.removed.push(id);
            }
        }
        for &(id, flags) in &slots {
            let previous = prior.iter().find(|&&(was, _)| was == id);
            // ClassicAPI also announces a return from complete to incomplete. The stock
            // slot watcher announces cleared FAIL; both are actual state transitions.
            let accepted = previous.is_none()
                || previous.is_some_and(|&(_, old)| old & 3 != 0 && flags & 3 == 0);
            if id != 0 && accepted && !life.accepted.contains(&id) {
                life.accepted.push(id);
            }
        }
    }
    life.slots = Some(slots);
}

/// The host calls this after the stock quest-log and acceptance-message feeds. Pending accepts
/// wait for a cached row; folding alone never removes a quest. A folded row's index is 0 here
/// because the stock public log has no index for it; its questID remains available to pfUI.
pub fn dispatch_log_events(lua: &Lua) {
    let mut events = Vec::new();
    {
        let Some(mut life) = lua.app_data_mut::<Lifecycle>() else {
            return;
        };
        for id in std::mem::take(&mut life.removed) {
            events.push(("QUEST_REMOVED", vec![ScriptValue::Int(i64::from(id))]));
        }
        let model = lua
            .app_data_ref::<crate::script::ModelCell>()
            .expect("model");
        life.accepted.retain(|id| {
            let index = model
                .quest_log
                .entries
                .iter()
                .position(|row| !row.is_header && row.quest_id == *id)
                .map(|index| index as i64 + 1)
                .or_else(|| model.quest_log.hidden_quest_ids.contains(id).then_some(0));
            let Some(index) = index else { return true };
            events.push((
                "QUEST_ACCEPTED",
                vec![ScriptValue::Int(index), ScriptValue::Int(i64::from(*id))],
            ));
            false
        });
    }
    for (name, args) in events {
        super::fire_pfui_event(lua, name, args);
    }
}

/// A successful SMSG_QUESTGIVER_QUEST_COMPLETE, never a dialog close or reward request.
pub fn turned_in(lua: &Lua, id: u32, xp: u32, copper: u32) {
    super::fire_pfui_event(
        lua,
        "QUEST_TURNED_IN",
        vec![id, xp, copper]
            .into_iter()
            .map(|value| ScriptValue::Int(i64::from(value)))
            .collect(),
    );
}

/// Replace the current realm's cache; the host re-feeds it after a VM reload.
pub fn set_details(lua: &Lua, details: impl IntoIterator<Item = Detail>) {
    if let Some(mut cache) = lua.app_data_mut::<Cache>() {
        cache.0 = details
            .into_iter()
            .map(|detail| (detail.quest_id, detail))
            .collect();
    }
}

fn items(lua: &Lua, items: &[(u32, u32)]) -> mlua::Result<Table> {
    let array = lua.create_table()?;
    for &(id, count) in items.iter().filter(|(id, _)| *id != 0) {
        let item = lua.create_table()?;
        item.set("id", id)?;
        item.set("count", count)?;
        array.push(item)?;
    }
    Ok(array)
}

fn data(lua: &Lua, detail: &Detail) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set("questID", detail.quest_id)?;
    table.set("title", detail.title.as_str())?;
    table.set("level", detail.level)?;
    table.set("questType", detail.tag.as_deref())?;
    table.set("rewardMoney", detail.money.max(0))?;
    table.set(
        "requiredMoney",
        if detail.money < 0 {
            detail.money.unsigned_abs()
        } else {
            0
        },
    )?;
    table.set("rewardMoneyAtMaxLevel", detail.money_max_level)?;
    table.set("rewardSpellID", detail.reward_spell)?;
    table.set("srcItemID", detail.src_item)?;
    table.set("questFlags", detail.flags)?;
    table.set("isSharable", detail.flags & 8 != 0)?;
    table.set("description", detail.description.as_str())?;
    table.set("objectives", detail.objectives.as_str())?;
    table.set("completionText", detail.completion.as_str())?;
    if let Some((map, x, y, opt)) = detail.poi.filter(|(map, _, _, _)| *map != 0) {
        let poi = lua.create_table()?;
        poi.set("mapID", map)?;
        poi.set("x", x)?;
        poi.set("y", y)?;
        poi.set("opt", opt)?;
        table.set("poi", poi)?;
    }
    table.set("rewardItems", items(lua, &detail.rewards)?)?;
    table.set("choiceItems", items(lua, &detail.choices)?)?;
    let requirements = lua.create_table()?;
    for req in &detail.requirements {
        let row = lua.create_table()?;
        row.set("kind", req.kind.as_str())?;
        row.set("id", req.id)?;
        row.set("count", req.count)?;
        row.set("text", req.text.as_str())?;
        requirements.push(row)?;
    }
    table.set("requirements", requirements)?;
    Ok(table)
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Cache::default());
    lua.set_app_data(Lifecycle::default());
    let namespace = lua.create_table()?;
    namespace.set(
        "GetQuestIDForLogIndex",
        lua.create_function(|lua, index: i64| {
            let model = lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model");
            Ok(usize::try_from(index)
                .ok()
                .and_then(|i| i.checked_sub(1))
                .and_then(|i| model.quest_log.entries.get(i))
                .map(|entry| if entry.is_header { 0 } else { entry.quest_id }))
        })?,
    )?;
    namespace.set(
        "GetQuestDetails",
        lua.create_function(|lua, id: i64| {
            let cache = lua.app_data_ref::<Cache>().expect("quest cache");
            u32::try_from(id)
                .ok()
                .and_then(|id| cache.0.get(&id))
                .map(|detail| data(lua, detail))
                .transpose()
        })?,
    )?;
    lua.globals().set("C_QuestLog", namespace)
}
