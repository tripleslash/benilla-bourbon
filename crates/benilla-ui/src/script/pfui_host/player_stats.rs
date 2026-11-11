//! Caster displays read streamed school modifiers and each unit's casting-speed descriptor.

use mlua::{Lua, Value};
use std::collections::HashMap;

#[derive(Default)]
struct Stats {
    damage: [i64; 7],
    cast_speeds: HashMap<u64, f32>,
    speeds: HashMap<u64, [f32; 3]>,
}

pub fn set_damage(lua: &Lua, damage: [i64; 7]) {
    if let Some(mut stats) = lua.app_data_mut::<Stats>() {
        stats.damage = damage;
    }
}

pub fn set_cast_speeds(lua: &Lua, speeds: impl IntoIterator<Item = (u64, f32)>) {
    if let Some(mut stats) = lua.app_data_mut::<Stats>() {
        stats.cast_speeds.clear();
        stats.cast_speeds.extend(speeds);
    }
}

/// Each visible unit's `[current, run, swim]` speed in yards per second.
pub fn set_speeds(lua: &Lua, speeds: impl IntoIterator<Item = (u64, [f32; 3])>) {
    if let Some(mut stats) = lua.app_data_mut::<Stats>() {
        stats.speeds.clear();
        stats.speeds.extend(speeds);
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Stats::default());
    // current, run, flight, swim; 1.12 has no flight, and a unit out of view answers zeros.
    lua.globals().set(
        "GetUnitSpeed",
        lua.create_function(|lua, unit: Value| {
            let guid = lua.coerce_string(unit)?.and_then(|unit| {
                super::units::guid(lua, &unit.to_string_lossy())
                    .ok()
                    .flatten()
            });
            let stats = lua.app_data_ref::<Stats>().expect("player stats");
            let [current, run, swim] = guid
                .and_then(|guid| stats.speeds.get(&guid))
                .copied()
                .unwrap_or_default();
            Ok((f64::from(current), f64::from(run), 0.0, f64::from(swim)))
        })?,
    )?;
    lua.globals().set(
        "GetSpellBonusDamage",
        lua.create_function(|lua, school: Value| {
            let number = lua
                .coerce_number(school)?
                .filter(|n| n.is_finite() && n.fract() == 0.0 && (1.0..=7.0).contains(n))
                .ok_or_else(|| {
                    mlua::Error::runtime("Usage: GetSpellBonusDamage(school), school must be 1..7")
                })?;
            Ok(lua.app_data_ref::<Stats>().expect("player stats").damage[number as usize - 1])
        })?,
    )?;
    lua.globals().set(
        "UnitSpellHaste",
        lua.create_function(|lua, unit: Value| {
            let guid = lua.coerce_string(unit)?.and_then(|unit| {
                super::units::guid(lua, &unit.to_string_lossy())
                    .ok()
                    .flatten()
            });
            let stats = lua.app_data_ref::<Stats>().expect("player stats");
            let speed = guid
                .and_then(|guid| stats.cast_speeds.get(&guid))
                .copied()
                .filter(|speed| speed.is_finite() && *speed > 0.0);
            Ok(speed.map_or(0.0, |speed| (1.0 / f64::from(speed) - 1.0) * 100.0))
        })?,
    )?;
    Ok(())
}
