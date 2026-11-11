//! `GetCurrentChatEvent` and `GetCurrentChatGUID`: the `CHAT_MSG_*` event whose handlers run now,
//! with the arguments they got. An event raised inside another one's handlers is the current one
//! until its own handlers are done.

use crate::script::ScriptValue;
use mlua::{Lua, MultiValue, Table, Value};

const REG_STACK: &str = "benilla.pfui.chat_events";

/// The event in progress while it lives: [`enter`] pushes it, the drop pops it.
pub(crate) struct ChatEvent<'a> {
    lua: &'a Lua,
    pushed: bool,
}

impl Drop for ChatEvent<'_> {
    fn drop(&mut self) {
        if !self.pushed {
            return;
        }
        if let Ok(stack) = self.lua.named_registry_value::<Table>(REG_STACK) {
            let _ = stack.raw_remove(stack.raw_len());
        }
    }
}

pub(crate) fn enter<'a>(lua: &'a Lua, event: &str, args: &[ScriptValue]) -> ChatEvent<'a> {
    let pushed = event.starts_with("CHAT_MSG_")
        && super::installed(lua).unwrap_or(false)
        && push(lua, event, args).is_ok();
    ChatEvent { lua, pushed }
}

fn push(lua: &Lua, event: &str, args: &[ScriptValue]) -> mlua::Result<()> {
    let stack: Table = lua.named_registry_value(REG_STACK)?;
    let entry = lua.create_table()?;
    entry.raw_set(1, event)?;
    for (i, arg) in args.iter().enumerate() {
        entry.raw_set(i + 2, arg.clone().into_lua(lua)?)?;
    }
    entry.raw_set("n", args.len() + 1)?;
    stack.raw_set(stack.raw_len() + 1, entry)
}

fn current(lua: &Lua) -> mlua::Result<Option<Table>> {
    let stack: Table = lua.named_registry_value(REG_STACK)?;
    match stack.raw_get::<Value>(stack.raw_len())? {
        Value::Table(entry) => Ok(Some(entry)),
        _ => Ok(None),
    }
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_named_registry_value(REG_STACK, lua.create_table()?)?;
    let globals = lua.globals();
    globals.set(
        "GetCurrentChatEvent",
        lua.create_function(|lua, ()| {
            let Some(entry) = current(lua)? else {
                return Ok(MultiValue::new());
            };
            let n: usize = entry.raw_get("n")?;
            let mut values = Vec::with_capacity(n);
            for i in 1..=n {
                values.push(entry.raw_get::<Value>(i)?);
            }
            Ok(MultiValue::from_vec(values))
        })?,
    )?;
    // The sender's GUID, the event's `arg12` (position 13 after the name), `nil` for `""`.
    globals.set(
        "GetCurrentChatGUID",
        lua.create_function(|lua, ()| {
            let Some(entry) = current(lua)? else {
                return Ok(Value::Nil);
            };
            match entry.raw_get::<Value>(13)? {
                Value::String(s) if !s.as_bytes().is_empty() => Ok(Value::String(s)),
                _ => Ok(Value::Nil),
            }
        })?,
    )
}
