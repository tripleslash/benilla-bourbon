//! In-memory group-roll history, reconstructed exclusively from received roll packets.

use benilla_ui::script::{pfui_host, ScriptValue};
use mlua::{Lua, MultiValue, Value};
use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub target: u64,
    pub slot: u32,
}

#[derive(Clone)]
pub struct Player {
    pub guid: u64,
    pub name: Option<String>,
    pub class: Option<String>,
    pub kind: u8,
    pub roll: u8,
    pub is_me: bool,
}

pub enum Event {
    Start,
    Roll(Player),
    Won(Player),
    Passed,
}

struct Item {
    key: Key,
    id: u64,
    link: String,
    timestamp: f64,
    players: Vec<Player>,
    winner: Option<u64>,
    done: bool,
}

#[derive(Default)]
struct History {
    serial: u64,
    items: VecDeque<Item>,
}

/// One packet's values, captured at receipt, including cached names/classes and its UI timestamp.
pub fn observe(lua: &Lua, key: Key, item: u32, suffix: i32, timestamp: f64, event: Event) {
    let mut notifications = Vec::new();
    {
        let Some(mut history) = lua.app_data_mut::<History>() else {
            return;
        };
        let starting = matches!(event, Event::Start);
        let link = format!("item:{item}:0:{suffix}:0");
        let existing = history
            .items
            .iter()
            .rposition(|item| item.key == key && item.link == link && (!starting || !item.done));
        // A duplicate start on reconnect keeps the same stable roll identity.
        let index = existing.unwrap_or_else(|| {
            if history.items.len() == 128 {
                history.items.pop_front();
            }
            history.serial = history.serial.wrapping_add(1);
            let id = history.serial;
            history.items.push_back(Item {
                key,
                id,
                link,
                timestamp,
                players: vec![],
                winner: None,
                done: false,
            });
            notifications.push(("LOOT_HISTORY_FULL_UPDATE", vec![]));
            history.items.len() - 1
        });
        let item = &mut history.items[index];
        let won = matches!(event, Event::Won(_));
        let was_done = item.done;
        match event {
            Event::Start => {}
            Event::Roll(mut player) | Event::Won(mut player) => {
                let player_index = item.players.iter().position(|old| old.guid == player.guid);
                if let Some(index) = player_index {
                    let old = &item.players[index];
                    // A later dice result may land after a cache eviction; retain captured identity.
                    if player.name.is_none() {
                        player.name = old.name.clone();
                    }
                    if player.class.is_none() {
                        player.class = old.class.clone();
                    }
                    // Replayed votes carry zero and cannot erase an observed dice result.
                    if player.roll == 0 && old.roll != 0 {
                        player.roll = old.roll;
                        player.kind = old.kind;
                    }
                    item.players[index] = player;
                } else {
                    item.players.push(player);
                }
                let player_index = player_index.unwrap_or(item.players.len() - 1);
                if won {
                    item.winner = Some(item.players[player_index].guid);
                    item.done = true;
                }
                notifications.push((
                    "LOOT_HISTORY_ROLL_CHANGED",
                    vec![
                        ScriptValue::Int(index as i64 + 1),
                        ScriptValue::Int(player_index as i64 + 1),
                    ],
                ));
                if won && !was_done {
                    notifications.push((
                        "LOOT_HISTORY_ROLL_COMPLETE",
                        vec![ScriptValue::Int(index as i64 + 1)],
                    ));
                }
            }
            Event::Passed => {
                item.done = true;
                if !was_done {
                    notifications.push((
                        "LOOT_HISTORY_ROLL_COMPLETE",
                        vec![ScriptValue::Int(index as i64 + 1)],
                    ));
                }
            }
        }
    }
    for (event, args) in notifications {
        pfui_host::fire_pfui_event(lua, event, args);
    }
}

fn text(lua: &Lua, text: Option<&str>) -> mlua::Result<Value> {
    Ok(text
        .map(|text| lua.create_string(text))
        .transpose()?
        .map_or(Value::Nil, Value::String))
}

/// A real name-query answer may arrive after the last roll packet, including after completion.
/// Fill only missing identity fields; preserve captured names, votes, dice and winner indices.
pub fn resolve_player(lua: &Lua, guid: u64, name: &str, class: Option<&str>) {
    if guid == 0 || name.is_empty() {
        return;
    }
    let mut changed = Vec::new();
    {
        let Some(mut history) = lua.app_data_mut::<History>() else {
            return;
        };
        for (item_index, item) in history.items.iter_mut().enumerate() {
            for (player_index, player) in item.players.iter_mut().enumerate() {
                if player.guid != guid {
                    continue;
                }
                let mut dirty = false;
                if player.name.is_none() {
                    player.name = Some(name.into());
                    dirty = true;
                }
                if player.class.is_none() && class.is_some() {
                    player.class = class.map(str::to_owned);
                    dirty = true;
                }
                if dirty {
                    changed.push((item_index + 1, player_index + 1));
                }
            }
        }
    }
    for (item, player) in changed {
        pfui_host::fire_pfui_event(
            lua,
            "LOOT_HISTORY_ROLL_CHANGED",
            vec![
                ScriptValue::Int(item as i64),
                ScriptValue::Int(player as i64),
            ],
        );
    }
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(History::default());
    let api = lua.create_table()?;
    api.set(
        "GetNumItems",
        lua.create_function(|lua, ()| {
            Ok(lua
                .app_data_ref::<History>()
                .expect("loot history")
                .items
                .len())
        })?,
    )?;
    api.set(
        "GetItem",
        lua.create_function(|lua, index: i64| {
            let history = lua.app_data_ref::<History>().expect("loot history");
            let Some(item) = usize::try_from(index)
                .ok()
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| history.items.get(index))
            else {
                return Ok(MultiValue::new());
            };
            let winner = item
                .winner
                .and_then(|guid| item.players.iter().position(|player| player.guid == guid));
            Ok(MultiValue::from_vec(vec![
                Value::Integer(item.id as i64),
                text(lua, Some(&item.link))?,
                Value::Integer(item.players.len() as i64),
                Value::Boolean(item.done),
                winner.map_or(Value::Nil, |index| Value::Integer(index as i64 + 1)),
                Value::Number(item.timestamp),
            ]))
        })?,
    )?;
    api.set(
        "GetPlayerInfo",
        lua.create_function(|lua, (item, player): (i64, i64)| {
            let history = lua.app_data_ref::<History>().expect("loot history");
            let Some(item) = usize::try_from(item)
                .ok()
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| history.items.get(index))
            else {
                return Ok(MultiValue::new());
            };
            let Some(player) = usize::try_from(player)
                .ok()
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| item.players.get(index))
            else {
                return Ok(MultiValue::new());
            };
            Ok(MultiValue::from_vec(vec![
                text(lua, player.name.as_deref())?,
                text(lua, player.class.as_deref())?,
                Value::Integer(i64::from(player.kind)),
                Value::Integer(i64::from(player.roll)),
                Value::Boolean(item.winner == Some(player.guid)),
                Value::Boolean(player.is_me),
            ]))
        })?,
    )?;
    api.set(
        "Clear",
        lua.create_function(|lua, ()| {
            lua.app_data_mut::<History>()
                .expect("loot history")
                .items
                .clear();
            pfui_host::fire_pfui_event(lua, "LOOT_HISTORY_FULL_UPDATE", vec![]);
            Ok(())
        })?,
    )?;
    lua.globals().set("C_LootHistory", api)
}
