//! `C_Macro.CreateMacro` and `C_Macro.EditMacro` through the stock verbs, which already take an
//! icon by file name ([`super::icon_by_name`]), so the table, caps, ids and events stay theirs.
//! The modern signatures differ in their arguments (`CreateMacro(name, icon, body, perCharacter)`
//! has no `local` flag and an optional icon, `EditMacro` also names its macro by name) and in
//! their failures, which answer nil: a full table, or a macro that is not there.

use mlua::{Function, Lua, MultiValue, Value};

/// An icon given as a path: its bare file name, which the stock verbs put under
/// `Interface\Icons\`. A number, or anything else, goes through as given.
fn icon(lua: &Lua, value: Value) -> mlua::Result<Value> {
    let Value::String(texture) = &value else {
        return Ok(value);
    };
    let texture = texture.to_string_lossy().replace('/', "\\");
    match texture.rsplit('\\').next() {
        Some(base) if !base.is_empty() && base.parse::<f64>().is_err() => {
            Ok(Value::String(lua.create_string(base)?))
        }
        _ => Ok(value),
    }
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let g = lua.globals();
    let create: Function = g.get("CreateMacro")?;
    let edit: Function = g.get("EditMacro")?;
    let index_by_name: Function = g.get("GetMacroIndexByName")?;
    let info: Function = g.get("GetMacroInfo")?;
    let namespace = lua.create_table()?;
    namespace.set(
        "CreateMacro",
        lua.create_function(move |lua, args: MultiValue| {
            let mut a: Vec<Value> = args.into_iter().collect();
            a.resize(4, Value::Nil);
            let per_character = a.pop().unwrap_or(Value::Nil);
            a[1] = match std::mem::replace(&mut a[1], Value::Nil) {
                Value::Nil => Value::String(lua.create_string("INV_Misc_QuestionMark")?),
                given => icon(lua, given)?,
            };
            a.push(Value::Nil);
            a.push(per_character);
            // The stock verb raises on a full table; the modern one answers nil.
            Ok(create
                .call::<MultiValue>(MultiValue::from_vec(a))
                .unwrap_or_default())
        })?,
    )?;
    namespace.set(
        "EditMacro",
        lua.create_function(move |lua, args: MultiValue| {
            let mut a: Vec<Value> = args.into_iter().collect();
            a.resize(4, Value::Nil);
            if let Value::String(_) = a[0] {
                a[0] = index_by_name.call(a[0].clone())?;
            }
            // A macro that is not there answers nil, where the stock verb answers the first empty
            // slot's index.
            if info.call::<Value>(a[0].clone())?.is_nil() {
                return Ok(MultiValue::new());
            }
            a[2] = icon(lua, std::mem::replace(&mut a[2], Value::Nil))?;
            edit.call::<MultiValue>(MultiValue::from_vec(a))
        })?,
    )?;
    g.set("C_Macro", namespace)
}
