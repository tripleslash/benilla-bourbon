//! The ten-value metadata tuple used by pfUI's spell library, including book-slot overloads.

use super::Spell;
use mlua::{Lua, MultiValue, Value};
use std::{collections::BTreeMap, rc::Rc};

pub(super) fn install(lua: &Lua, records: Rc<BTreeMap<u32, Spell>>) -> mlua::Result<()> {
    lua.globals().set(
        "GetSpellInfo",
        lua.create_function(move |lua, (input, book): (Value, Option<String>)| {
            let Some((id, spell)) =
                benilla_ui::script::pfui_host::units::legacy_spell_identifier(lua, input, book)?
                    .and_then(|id| records.get(&id).map(|spell| (id, spell)))
            else {
                return Ok(MultiValue::from_vec(vec![Value::Nil]));
            };
            Ok(MultiValue::from_vec(vec![
                Value::String(lua.create_string(&spell.name)?),
                Value::String(lua.create_string(&spell.rank)?),
                spell
                    .icon
                    .as_ref()
                    .map(|icon| lua.create_string(icon))
                    .transpose()?
                    .map_or(Value::Nil, Value::String),
                Value::Integer(i64::from(spell.cost)),
                Value::Boolean(spell.funnel),
                Value::Integer(i64::from(spell.power_type)),
                Value::Integer(i64::from(spell.cast_time)),
                Value::Number(f64::from(spell.min_range)),
                Value::Number(f64::from(spell.max_range)),
                Value::Integer(i64::from(id)),
            ]))
        })?,
    )
}
