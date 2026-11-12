//! ClassicAPI 1.20's base-library additions that read the VM's stack.

use mlua::Lua;

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    // `GetCallstackHeight()`: the call levels on the stack, the caller being level 1 and a native
    // function a level, as `debugstack` counts them.
    lua.globals().set(
        "GetCallstackHeight",
        lua.create_function(|lua, ()| {
            let mut height = 0usize;
            while lua.inspect_stack(height + 1, |_| ()).is_some() {
                height += 1;
            }
            Ok(height)
        })?,
    )?;
    lua.load(include_str!("baselib.lua"))
        .set_name("@benilla-pfui/baselib.lua")
        .exec()
}
