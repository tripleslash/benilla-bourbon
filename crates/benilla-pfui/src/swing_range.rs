//! `C_SwingTimer`: whether the current target is in range for each `Enum.PlayerSwingType`, as the
//! client measures it each frame ([`set`]), and `PLAYER_SWING_RANGE_UPDATE` when an enabled
//! type's answer changes.

use benilla_ui::script::{pfui_host, ScriptValue};
use mlua::{Lua, Value};

/// Each swing type's answer, `None` where none is possible (no target, an unattackable one, no
/// weapon for the type), and whether its range check is on.
#[derive(Default)]
struct Ranges {
    now: [Option<bool>; 3],
    enabled: [bool; 3],
}

fn index(swing_type: Value) -> Option<usize> {
    match swing_type {
        Value::Integer(n) => usize::try_from(n).ok(),
        Value::Number(n) if n.fract() == 0.0 && n >= 0.0 => Some(n as usize),
        _ => None,
    }
    .filter(|i| *i < 3)
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Ranges::default());
    let api = lua.create_table()?;
    api.set(
        "EnableRangeCheck",
        lua.create_function(|lua, (swing_type, enable): (Value, Value)| {
            if let (Some(i), Some(mut ranges)) = (index(swing_type), lua.app_data_mut::<Ranges>()) {
                ranges.enabled[i] = !matches!(enable, Value::Nil | Value::Boolean(false));
            }
            Ok(())
        })?,
    )?;
    api.set(
        "IsTargetWithinSwingRange",
        lua.create_function(|lua, swing_type: Value| {
            Ok(index(swing_type).and_then(|i| lua.app_data_ref::<Ranges>().and_then(|r| r.now[i])))
        })?,
    )?;
    lua.globals().set("C_SwingTimer", api)
}

/// This frame's answers; an enabled type whose answer changed fires
/// `PLAYER_SWING_RANGE_UPDATE(swingType, isInRange, checksRange)`.
pub fn set(lua: &Lua, now: [Option<bool>; 3]) {
    let changed: Vec<(usize, Option<bool>)> = {
        let Some(mut ranges) = lua.app_data_mut::<Ranges>() else {
            return;
        };
        let changed = (0..3)
            .filter(|&i| ranges.enabled[i] && ranges.now[i] != now[i])
            .map(|i| (i, now[i]))
            .collect();
        ranges.now = now;
        changed
    };
    let flag = |on: bool| {
        if on {
            ScriptValue::Int(1)
        } else {
            ScriptValue::Nil
        }
    };
    for (swing_type, answer) in changed {
        pfui_host::fire_pfui_event(
            lua,
            "PLAYER_SWING_RANGE_UPDATE",
            vec![
                ScriptValue::Int(swing_type as i64),
                flag(answer == Some(true)),
                flag(answer.is_some()),
            ],
        );
    }
}
