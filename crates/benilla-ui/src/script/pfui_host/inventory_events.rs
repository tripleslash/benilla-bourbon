//! Aggregate stock bag events and compare actual equipment instance GUIDs after the stock feeds.

use super::Model;
use crate::script::ScriptValue;
use mlua::Lua;

#[derive(Default)]
struct State {
    owner: u64,
    worn: Option<[u64; 19]>,
    current: Option<[u64; 19]>,
    bags: bool,
    bag_busy: bool,
}

pub(super) fn set_bag_busy(lua: &Lua, busy: bool) {
    if let Some(mut state) = lua.app_data_mut::<State>() {
        state.bag_busy = busy;
    }
}

pub(super) fn observe_equipment(lua: &Lua, guids: [u64; 19]) {
    if let Some(mut state) = lua.app_data_mut::<State>() {
        state.current = Some(guids);
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(State::default());
    // Mark only: the host drains this after all ordinary inventory handlers this frame.
    lua.load(
        "local mark=... local f=CreateFrame('Frame') f:RegisterEvent('BAG_UPDATE') \
         f:SetScript('OnEvent',function() mark() end)",
    )
    .set_name("@benilla-pfui/inventory-events.lua")
    .call(lua.create_function(|lua, ()| {
        lua.app_data_mut::<State>().expect("inventory events").bags = true;
        Ok(())
    })?)
}

/// Called after the stock bag and paperdoll snapshots. One bag event per frame; equipment
/// payloads follow ClassicAPI's `(slot, 1-or-nil)`, including same-entry instance swaps.
pub(super) fn dispatch(lua: &Lua) {
    // VM creation precedes the optional adapter's staged installation, including at login.
    if lua.app_data_ref::<State>().is_none() {
        return;
    }
    let owner = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model")
        .unit_guids
        .player;
    let (bags, changed) = {
        let mut state = lua.app_data_mut::<State>().expect("inventory events");
        let Some(current) = state.current else { return };
        let prior = state.worn.replace(current);
        let mut changed = Vec::new();
        if state.owner == owner && owner != 0 {
            if let Some(prior) = prior {
                for (slot, (&was, &now)) in prior.iter().zip(&current).enumerate() {
                    if was != now {
                        changed.push((slot + 1, now != 0));
                    }
                }
            }
        }
        state.owner = owner;
        let bags = !state.bag_busy && std::mem::take(&mut state.bags);
        (bags, changed)
    };
    for (slot, equipped) in changed {
        let present = if equipped {
            ScriptValue::Int(1)
        } else {
            ScriptValue::Nil
        };
        super::super::fire_pfui_event(
            lua,
            "PLAYER_EQUIPMENT_CHANGED",
            vec![ScriptValue::Int(slot as i64), present],
        );
    }
    if bags {
        super::super::fire_pfui_event(lua, "BAG_UPDATE_DELAYED", vec![]);
    }
}
