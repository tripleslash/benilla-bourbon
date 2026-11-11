use mlua::{Function, Lua, Table};

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let frames: Table = lua.named_registry_value("__benilla_frame_methods")?;
    let invoke: Function = lua.named_registry_value("benilla.pfui.attribute_invoke")?;
    let actions: Table = lua.named_registry_value("benilla.pfui.attribute_actions")?;
    let methods: Table = lua
        .load(include_str!("attributes.lua"))
        .set_name("@benilla-pfui/attributes.lua")
        .call((frames.clone(), invoke, actions))?;
    for name in [
        "SetAttribute",
        "SetAttributeNoHandler",
        "GetAttribute",
        "ClearAttribute",
        "SetScript",
        "GetScript",
        "HasScript",
    ] {
        frames.set(name, methods.get::<Function>(name)?)?;
    }
    lua.set_named_registry_value(
        "benilla.pfui.attribute_hover",
        methods.get::<Function>("hoveredUnit")?,
    )
}
