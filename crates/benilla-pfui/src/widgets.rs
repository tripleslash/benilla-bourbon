use mlua::{Function, Lua, MultiValue, Table, Value};

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let frames: Table = lua.named_registry_value("__benilla_frame_methods")?;
    let methods: Table = lua
        .load(include_str!("widgets.lua"))
        .set_name("@benilla-pfui/widgets.lua")
        .call(frames.clone())?;
    lua.set_named_registry_value("benilla.pfui.unit_events", methods.get::<Table>("filters")?)?;
    // These are the pinned host's shared method tables: extending them reaches existing,
    // anonymous, XML-created and future objects without replacing CreateFrame.
    for key in [
        "__benilla_frame_methods",
        "__benilla_region_methods",
        "__benilla_texture_methods",
        "__benilla_fontstring_methods",
        "__benilla_title_methods",
    ] {
        let target: Table = lua.named_registry_value(key)?;
        for name in ["SetSize", "GetSize"] {
            target.set(name, methods.get::<mlua::Function>(name)?)?;
        }
    }
    for key in [
        "__benilla_frame_methods",
        "__benilla_texture_methods",
        "__benilla_fontstring_methods",
    ] {
        let target: Table = lua.named_registry_value(key)?;
        target.set("SetShown", methods.get::<mlua::Function>("SetShown")?)?;
    }
    frames.set("HookScript", methods.get::<mlua::Function>("HookScript")?)?;
    for name in [
        "RegisterUnitEvent",
        "UnregisterEvent",
        "UnregisterAllEvents",
    ] {
        frames.set(name, methods.get::<mlua::Function>(name)?)?;
    }
    let text: Table = lua.named_registry_value("__benilla_fontstring_methods")?;
    text.set(
        "SetFormattedText",
        methods.get::<mlua::Function>("SetFormattedText")?,
    )?;
    // Reference colour channels use lua_tonumber. The pinned host's shape-C helper drops
    // numeric strings, including the strings pfUI's GetStringColor returns for map reveal.
    for key in [
        "__benilla_frame_methods",
        "__benilla_region_methods",
        "__benilla_texture_methods",
        "__benilla_fontstring_methods",
        "__benilla_statusbar_methods",
        "__benilla_colorselect_methods",
    ] {
        let target: Table = lua.named_registry_value(key)?;
        for (name, first, count) in [
            ("SetVertexColor", 1, 3),
            ("SetTextColor", 1, 3),
            ("SetShadowColor", 1, 3),
            ("SetStatusBarColor", 1, 3),
            ("SetColorRGB", 1, 3),
            ("SetColorHSV", 1, 3),
            ("SetGradient", 2, 6),
            ("SetGradientAlpha", 2, 8),
        ] {
            let Some(native) = target.get::<Option<Function>>(name)? else {
                continue;
            };
            target.set(
                name,
                lua.create_function(move |lua, mut args: MultiValue| {
                    for value in args.iter_mut().skip(first).take(count) {
                        if matches!(value, Value::String(_)) {
                            if let Some(number) = lua.coerce_number(value.clone())? {
                                *value = Value::Number(number);
                            }
                        }
                    }
                    native.call::<MultiValue>(args)
                })?,
            )?;
        }
    }
    Ok(())
}
