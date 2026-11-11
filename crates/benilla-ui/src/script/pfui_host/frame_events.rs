//! A frame-valued extension event (`NAME_PLATE_CREATED`): the stock event walk
//! ([`crate::script::tick::fire_event_into`]) with a frame table as `arg1`, which a
//! [`crate::script::ScriptValue`] cannot carry.

use mlua::{Lua, Table, Value};

/// Fire `event` to its listeners with `frame` as `arg1`: `event` is set once around the walk, and
/// each listener's successor is read before its call, as the stock walk does. A raise of the
/// tick's, as every pfUI raise is ([`crate::script::pfui_host::fire_pfui_event`]).
pub(crate) fn fire(lua: &Lua, event: &str, frame: Table) {
    crate::script::origin::within(crate::script::origin::Origin::Tick, || {
        walk(lua, event, frame)
    });
}

fn walk(lua: &Lua, event: &str, frame: Table) {
    crate::script::origin::check(event, Some(lua));
    let model_mut = || {
        lua.app_data_mut::<crate::script::ModelCell>()
            .expect("model")
    };
    let mut at = model_mut()
        .event_to_frames
        .get(event)
        .and_then(|l| l.first().copied());
    if at.is_none() {
        return;
    }
    let globals = lua.globals();
    let saved = globals.get::<Value>("event").ok();
    if let Ok(name) = lua.create_string(event) {
        let _ = globals.set("event", name);
    }
    while let Some(h) = at {
        let mut model = model_mut();
        let next = model.event_to_frames.get(event).and_then(|l| {
            let pos = l.iter().position(|&x| x == h)?;
            l.get(pos + 1).copied()
        });
        let id = model.frame_id(h);
        drop(model);
        if let Err(e) = super::fire(lua, id, "OnEvent", vec![Value::Table(frame.clone())]) {
            model_mut().record_lua_error(&e);
        }
        at = next;
    }
    if let Some(saved) = saved {
        let _ = globals.set("event", saved);
    }
}
