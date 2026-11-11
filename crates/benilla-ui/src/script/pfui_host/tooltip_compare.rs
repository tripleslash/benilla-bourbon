//! The hyperlink form of the stock compare bindings: the offered item is named by a link, an
//! entry or another tooltip's item, and the worn candidate is chosen by the same walk.

use super::compare::{compare_against_worn, offered_view};
use crate::script::object::frame_handle_of;
use mlua::{Lua, Table, Value};

fn entry(lua: &Lua, value: &Value) -> mlua::Result<Option<u32>> {
    if let Value::String(text) = value {
        let text = text.to_string_lossy();
        if let Some((_, tail)) = text.split_once("item:") {
            return Ok(tail
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .and_then(|n| n.parse().ok())
                .filter(|n| *n != 0));
        }
    }
    Ok(lua
        .coerce_number(value.clone())?
        .filter(|n| n.fract() == 0.0 && *n >= 1.0 && *n <= f64::from(u32::MAX))
        .map(|n| n as u32))
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let methods: Table = lua.named_registry_value(crate::script::tooltip::REG_TOOLTIP_METHODS)?;
    // One value: 1 once the worn item is drawn under its header, nil when the offered item is
    // unknown, not equippable or has no worn counterpart at this offset. The per-stat change
    // lines of the modern binding are not drawn, so `shift` selects nothing here.
    methods.set(
        "SetHyperlinkCompareItem",
        lua.create_function(
            |lua,
             (this, link, offset, _shift, other): (
                Table,
                Value,
                Option<Value>,
                Value,
                Option<Table>,
            )| {
                let handle = frame_handle_of(lua, &this)?;
                let mut offered = entry(lua, &link)?;
                if offered.is_none() && link.is_nil() {
                    if let Some(other) = other {
                        if let Some(get) = other.get::<Option<mlua::Function>>("GetItem")? {
                            let (_, shown): (Value, Value) = get.call(other)?;
                            offered = entry(lua, &shown)?;
                        }
                    }
                }
                let left = match offset {
                    None | Some(Value::Nil) => 0.0,
                    Some(value) => match lua.coerce_number(value)? {
                        Some(n) if n.trunc() >= 1.0 => n.trunc() - 1.0,
                        _ => return Ok(Value::Nil),
                    },
                };
                let Some(view) = offered.and_then(|id| offered_view(lua, handle, id)) else {
                    return Ok(Value::Nil);
                };
                compare_against_worn(lua, &this, handle, &view, left)
            },
        )?,
    )
}
