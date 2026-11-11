//! Expose the engine's actual plate pool with stable slots and canonical frame wrappers.

use super::PlateState;
use crate::{script::ScriptValue, widget::FrameHandle};
use mlua::{Lua, Table};
use std::collections::HashSet;

#[derive(Default)]
struct Slots {
    live: Vec<Option<(u64, FrameHandle)>>,
    created: HashSet<FrameHandle>,
}

pub(crate) fn guid(lua: &Lua, index: usize) -> Option<u64> {
    lua.app_data_ref::<Slots>()?
        .live
        .get(index.checked_sub(1)?)?
        .map(|(guid, _)| guid)
}

pub(crate) fn all_tokens(lua: &Lua) -> Vec<String> {
    lua.app_data_ref::<Slots>()
        .map(|slots| {
            slots
                .live
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.is_some())
                .map(|(index, _)| format!("nameplate{}", index + 1))
                .collect()
        })
        .unwrap_or_default()
}

fn wrapper(lua: &Lua, handle: FrameHandle) -> mlua::Result<Table> {
    let id = lua
        .app_data_mut::<crate::script::ModelCell>()
        .expect("model")
        .frame_id(handle);
    crate::script::object::frame_wrapper(lua, id)
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Slots::default());
    let api = lua.create_table()?;
    api.set(
        "GetNamePlateForUnit",
        lua.create_function(|lua, token: String| {
            let guid = crate::script::pfui_host::auras::unit_guid(lua, &token);
            let frame = lua.app_data_ref::<Slots>().and_then(|slots| {
                slots
                    .live
                    .iter()
                    .flatten()
                    .find(|(id, _)| Some(*id) == guid)
                    .map(|(_, frame)| *frame)
            });
            frame.map(|frame| wrapper(lua, frame)).transpose()
        })?,
    )?;
    lua.globals().set("C_NamePlate", api)
}

pub(crate) fn before_sync(lua: &Lua, states: &[PlateState]) {
    let Some(slots) = lua.app_data_ref::<Slots>() else {
        return;
    };
    let present: HashSet<_> = states.iter().map(|state| state.key).collect();
    let removed: Vec<_> = slots
        .live
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| {
            slot.filter(|(guid, _)| !present.contains(guid))
                .map(|_| index)
        })
        .collect();
    drop(slots);
    for index in removed {
        // Keep the leaving unit and frame readable during the removal callback.
        crate::script::pfui_host::fire_pfui_event(
            lua,
            "NAME_PLATE_UNIT_REMOVED",
            vec![ScriptValue::Str(format!("nameplate{}", index + 1))],
        );
        lua.app_data_mut::<Slots>().expect("slots").live[index] = None;
    }
}

pub(crate) fn after_sync(lua: &Lua) {
    if lua.app_data_ref::<Slots>().is_none() {
        return;
    }
    let pool: Vec<_> = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model")
        .nameplates
        .plates
        .iter()
        .map(|plate| (plate.key(), plate.frame))
        .collect();
    let mut created = vec![];
    let mut added = vec![];
    {
        let mut slots = lua.app_data_mut::<Slots>().expect("slots");
        slots.live.resize(pool.len(), None);
        for (index, (guid, frame)) in pool.into_iter().enumerate() {
            if slots.created.insert(frame) {
                created.push(frame);
            }
            let next = guid.map(|guid| (guid, frame));
            if slots.live[index] != next && next.is_some() {
                added.push(index);
            }
            slots.live[index] = next;
        }
    }
    for frame in created {
        match wrapper(lua, frame) {
            Ok(frame) => crate::script::event::pfui::fire(lua, "NAME_PLATE_CREATED", frame),
            Err(error) => lua
                .app_data_mut::<crate::script::ModelCell>()
                .expect("model")
                .record_error(error.to_string()),
        }
    }
    for index in added {
        crate::script::pfui_host::fire_pfui_event(
            lua,
            "NAME_PLATE_UNIT_ADDED",
            vec![ScriptValue::Str(format!("nameplate{}", index + 1))],
        );
    }
}
