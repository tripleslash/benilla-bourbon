//! Read the stock cursor payload, without introducing a second cursor state.

use crate::script::cursor::{CursorPayload, EQUIPMENT_BAG};
use mlua::{Lua, MultiValue, Value};

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.globals().set(
        "GetCursorInfo",
        lua.create_function(|lua, ()| {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let Some(payload) = &model.cursor else {
                return Ok(MultiValue::new());
            };
            let mut result = Vec::new();
            let (kind, id) = match payload {
                CursorPayload::Item(item) => ("item", item.item_id),
                // The cursor holds the spell alone now; its book slot is looked up, 0 when the
                // spell has left the book.
                CursorPayload::Spell(spell) => (
                    "spell",
                    super::player_book(lua)
                        .slots
                        .iter()
                        .position(|slot| slot.spell_id == spell.spell_id)
                        .map_or(0, |at| at as u32 + 1),
                ),
                // The cursor holds the macro's id; its place in the table is answered.
                CursorPayload::Macro(macro_) => (
                    "macro",
                    model
                        .macros
                        .index_of_id(macro_.id)
                        .map_or(0, |at| at as u32),
                ),
                CursorPayload::Merchant(item) => ("merchant", item.row + 1),
                CursorPayload::Money(money) => ("money", money.copper),
                _ => return Ok(MultiValue::new()),
            };
            result.push(Value::String(lua.create_string(kind)?));
            result.push(Value::Integer(i64::from(id)));
            match payload {
                CursorPayload::Item(item) if item.bag != EQUIPMENT_BAG => {
                    if let Some(link) = &item.link {
                        result.push(Value::String(lua.create_string(link)?));
                    }
                }
                CursorPayload::Spell(spell) => {
                    result.push(Value::String(lua.create_string("spell")?));
                    result.push(Value::Integer(i64::from(spell.spell_id)));
                }
                _ => {}
            }
            Ok(MultiValue::from_vec(result))
        })?,
    )
}
