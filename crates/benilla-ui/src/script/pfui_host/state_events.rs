//! Edge events for player state the stock client changes without an event of its own: the
//! shapeshift form, worn durability and reputation standing. Each compares the stock snapshots
//! after their feeds; a new VM or another character sets a baseline and announces nothing.

use crate::script::ScriptValue;
use mlua::Lua;
use std::collections::HashMap;

/// A worn slot's item entry and `(current, max)` durability.
type Worn = [Option<(u32, Option<(u32, u32)>)>; 19];

#[derive(Default)]
struct State {
    owner: u64,
    form: Option<u8>,
    worn: Option<Worn>,
    standings: HashMap<u32, i32>,
}

pub(super) fn install(lua: &Lua) {
    lua.set_app_data(State::default());
}

pub(super) fn dispatch(lua: &Lua) {
    if lua.app_data_ref::<State>().is_none() {
        return;
    }
    let form = super::super::units::player_raw(lua).map(|unit| unit.form);
    let mut events = Vec::new();
    {
        let model = lua
            .app_data_ref::<crate::script::ModelCell>()
            .expect("model");
        let mut state = lua.app_data_mut::<State>().expect("state events");
        let owner = model.unit_guids.player;
        let same = owner != 0 && state.owner == owner;
        if !same {
            *state = State {
                owner,
                ..Default::default()
            };
        }
        if let Some(form) = form {
            if state.form.replace(form).is_some_and(|was| was != form) {
                events.push(("UPDATE_SHAPESHIFT_FORM", Vec::new()));
            }
        }
        if owner != 0 {
            let worn: Worn = std::array::from_fn(|slot| {
                model.inventory_slots[slot + 1]
                    .as_ref()
                    .map(|item| (item.item_id, item.durability))
            });
            if state.worn.replace(worn).is_some_and(|was| was != worn) {
                events.push(("UPDATE_INVENTORY_DURABILITY", Vec::new()));
            }
            let standings: HashMap<u32, i32> = model
                .reputation
                .entries
                .iter()
                .filter(|entry| !entry.is_header)
                .map(|entry| (entry.faction_id, entry.standing))
                .collect();
            // The login sync and a relisted table are a baseline; only a standing that moved
            // inside an unchanged list is a gain or a loss.
            if !state.standings.is_empty() && state.standings.len() == standings.len() {
                let mut moved: Vec<_> = standings
                    .iter()
                    .filter_map(|(id, now)| {
                        let was = state.standings.get(id)?;
                        (was != now).then_some((*id, *now, now - was))
                    })
                    .collect();
                moved.sort_unstable();
                for (id, now, delta) in moved {
                    events.push((
                        "FACTION_STANDING_CHANGED",
                        vec![
                            ScriptValue::Int(i64::from(id)),
                            ScriptValue::Int(i64::from(now)),
                            ScriptValue::Int(i64::from(delta)),
                        ],
                    ));
                }
            }
            if !standings.is_empty() {
                state.standings = standings;
            }
        }
    }
    for (event, payload) in events {
        super::super::fire_pfui_event(lua, event, payload);
    }
}
