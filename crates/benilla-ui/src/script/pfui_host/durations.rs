//! The engine half of ClassicAPI's timer bars and eased fills (`statusBar:SetTimerDuration`,
//! `SetValue(value, ExponentialEaseOut)`): the fill a bar shows apart from its value. The duration
//! objects, text bindings and the per-frame pass are Lua (`benilla-pfui`'s `durations.lua`).

use crate::script::object::frame_handle_of;
use crate::script::Model;
use crate::widget::KindState;
use mlua::{Lua, Table, Value};

/// Registry key of the table `durations.lua` takes: `set_display(bar, fraction or nil)`.
pub const DURATION_ENGINE: &str = "benilla.pfui.duration_engine";

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let engine = lua.create_table()?;
    engine.set(
        "set_display",
        lua.create_function(|lua, (bar, fraction): (Table, Value)| {
            let fraction = match fraction {
                Value::Number(n) => Some(n as f32),
                Value::Integer(n) => Some(n as f32),
                _ => None,
            };
            let handle = frame_handle_of(lua, &bar)?;
            let mut model = lua.app_data_mut::<Model>().expect("model app_data");
            if let Some(KindState::StatusBar(state)) = model
                .arena
                .frame_mut(handle)
                .map(|frame| &mut frame.kind_state)
            {
                state.set_display(fraction);
            }
            // The pass runs after the update walk wrote the bars' fills: write this one now, so
            // the frame being drawn shows it.
            crate::script::statusbar::update(&mut model, handle);
            Ok(())
        })?,
    )?;
    lua.set_named_registry_value(DURATION_ENGINE, engine)
}
