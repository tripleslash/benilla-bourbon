//! ClassicAPI's `GetUnscaledWidth`, `GetUnscaledHeight` and `GetUnscaledSize([ignoreRect])`: a
//! region's size in the units of `GetTop` and `GetBottom`, installed with the layer, so a stock VM
//! keeps 1.12's surface.

use crate::script::object::{classicapi_size_read, frame_handle_of};
use crate::script::region::classicapi_region_size_read;
use mlua::{Lua, Table};

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let frames: Table = lua.named_registry_value(super::super::REG_FRAME_METHODS)?;
    for (name, width) in [("GetUnscaledWidth", true), ("GetUnscaledHeight", false)] {
        frames.set(
            name,
            lua.create_function(move |lua, (this, ignore): (Table, Option<bool>)| {
                let h = frame_handle_of(lua, &this)?;
                Ok(classicapi_size_read(lua, h, width, ignore == Some(true)))
            })?,
        )?;
    }
    frames.set(
        "GetUnscaledSize",
        lua.create_function(|lua, (this, ignore): (Table, Option<bool>)| {
            let h = frame_handle_of(lua, &this)?;
            let ignore = ignore == Some(true);
            Ok((
                classicapi_size_read(lua, h, true, ignore),
                classicapi_size_read(lua, h, false, ignore),
            ))
        })?,
    )?;
    for key in [
        super::super::REG_REGION_METHODS,
        super::super::REG_TEXTURE_METHODS,
        super::super::REG_FONTSTRING_METHODS,
        super::super::REG_TITLE_METHODS,
    ] {
        let methods: Table = lua.named_registry_value(key)?;
        for (name, width) in [("GetUnscaledWidth", true), ("GetUnscaledHeight", false)] {
            methods.set(
                name,
                lua.create_function(move |lua, (this, ignore): (Table, Option<bool>)| {
                    classicapi_region_size_read(lua, &this, width, ignore == Some(true))
                })?,
            )?;
        }
        methods.set(
            "GetUnscaledSize",
            lua.create_function(|lua, (this, ignore): (Table, Option<bool>)| {
                let ignore = ignore == Some(true);
                Ok((
                    classicapi_region_size_read(lua, &this, true, ignore)?,
                    classicapi_region_size_read(lua, &this, false, ignore)?,
                ))
            })?,
        )?;
    }
    Ok(())
}
