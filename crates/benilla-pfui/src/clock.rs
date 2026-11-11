//! Server-calendar display time. This deliberately differs from UTC unix timestamps.

use mlua::Lua;
const CLOCK: &str = "benilla.pfui.server_local_time";

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let api = lua.create_table()?;
    api.set(
        "GetServerTimeLocal",
        lua.create_function(|lua, ()| {
            lua.named_registry_value::<Option<f64>>(CLOCK)
                .map(|time| time.map(|time| time.floor()))
        })?,
    )?;
    lua.globals().set("C_DateAndTime", api)
}

/// The host feeds the realm's current calendar epoch, advanced from LOGIN_SETTIMESPEED.
pub fn update(lua: &Lua, epoch: Option<f64>) -> mlua::Result<()> {
    if epoch.is_some_and(|n| !n.is_finite()) {
        return Err(mlua::Error::runtime("invalid server calendar time"));
    }
    lua.set_named_registry_value(CLOCK, epoch)
}

/// Convert the packet's zero-based 31-day serial to its actual Gregorian calendar date. Advance
/// elapsed game minutes from this anchor, rather than interpreting a synthetic February 31.
pub fn epoch(day_serial: u32, minute: u32) -> Option<f64> {
    if minute >= 1440 {
        return None;
    }
    let year = 2000 + i64::from(day_serial / 372);
    let month = i64::from(day_serial % 372 / 31) + 1;
    let day = i64::from(day_serial % 31) + 1;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if month > 12 || day > days[(month - 1) as usize] {
        return None;
    }
    let adjusted_year = year - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let yoe = adjusted_year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * shifted_month + 2) / 5 + day - 1;
    let serial = era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
    Some((serial * 86400 + i64::from(minute) * 60) as f64)
}
