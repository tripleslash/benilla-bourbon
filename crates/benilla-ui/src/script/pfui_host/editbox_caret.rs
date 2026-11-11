//! The EditBox caret methods a ClassicAPI client backports: `SetCursorPosition(position)` and
//! `GetCursorPosition()`, in characters from the start of the text.

use crate::script::object::frame_handle_of;
use crate::widget::KindState;
use mlua::{Lua, Table, Value};

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let methods: Table = lua.named_registry_value(crate::script::editbox::REG_EDITBOX_METHODS)?;
    methods.set(
        "SetCursorPosition",
        lua.create_function(|lua, (this, position): (Table, Value)| {
            let h = frame_handle_of(lua, &this)?;
            let position = lua
                .coerce_number(position)?
                .filter(|n| n.is_finite())
                .map_or(0, |n| n.max(0.0) as usize);
            let mut model = lua
                .app_data_mut::<crate::script::ModelCell>()
                .expect("model");
            if let Some(KindState::EditBox(eb)) =
                model.arena.frame_mut(h).map(|f| &mut f.kind_state)
            {
                let byte = eb
                    .text
                    .char_indices()
                    .nth(position)
                    .map_or(eb.text.len(), |(byte, _)| byte);
                eb.move_caret_to(byte, false);
            }
            Ok(())
        })?,
    )?;
    methods.set(
        "GetCursorPosition",
        lua.create_function(|lua, this: Table| {
            let h = frame_handle_of(lua, &this)?;
            let model = lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model");
            Ok(match model.arena.frame(h).map(|f| &f.kind_state) {
                Some(KindState::EditBox(eb)) => {
                    eb.text[..eb.cursor.min(eb.text.len())].chars().count()
                }
                _ => 0,
            })
        })?,
    )
}
