//! The window's size in device pixels, for `gxResolution`.
//!
//! On the reference client `gxResolution` is the mode the game draws at, and pfUI derives the
//! size of one device pixel from it (`768 / height / uiScale`) for its borders. The host keeps
//! that CVar for the windowed size only and draws fullscreen at the monitor's own, so the stored
//! string is not what is on screen: pfUI's borders came out two pixels thick on a 1800-line
//! display. With the extension, `GetCVar("gxResolution")` answers what the window really is.

use mlua::Lua;

const PIXELS: &str = "benilla.pfui.window_pixels";

/// The host feeds the window's physical size whenever it knows it; zero is no window yet.
pub fn set_window_pixels(lua: &Lua, width: u32, height: u32) -> mlua::Result<()> {
    let value = (width > 0 && height > 0).then(|| format!("{width}x{height}"));
    lua.set_named_registry_value(PIXELS, value)
}

/// `bridge.windowPixels()`: `"WxH"`, or nil before the host has a window.
pub(crate) fn install(lua: &Lua, bridge: &mlua::Table) -> mlua::Result<()> {
    bridge.set(
        "windowPixels",
        lua.create_function(|lua, ()| lua.named_registry_value::<Option<String>>(PIXELS))?,
    )
}
