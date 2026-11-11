//! The engine half of ClassicAPI functions that read the VM's own tables.

use mlua::{Lua, Value};

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let utils = lua.create_table()?;
    // An event `RegisterEvent` accepts: the 1.12 table's, or one the extension fires
    // ([`super::extension_event`]).
    utils.set(
        "IsEventValid",
        lua.create_function(|lua, name: Value| {
            let Value::String(name) = name else {
                return Ok(false);
            };
            let name = name.to_string_lossy();
            let model = lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model");
            Ok(model.vm.event_key(&name).is_some()
                || super::extension_event(&model, &name).is_some())
        })?,
    )?;
    lua.globals().set("C_EventUtils", utils)
}
