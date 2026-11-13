//! ClassicAPI's mirror-timer readers, `GetMirrorTimerInfo(index)` and
//! `GetMirrorTimerProgress(timer)` (ClassicAPI `src/unit/MirrorTimer.cpp`). 1.12's handler
//! (`0x5e7990`) keeps no state; these are 3.3.5's slots, one per wire type: a start writes its
//! slot and stamps the time, a pause writes nothing, a stop clears it. The app records each edge
//! before its `MIRROR_TIMER_*` event fires, so a handler of the event reads the new state.

use mlua::{Lua, MultiValue, Value};

/// The timer names by wire type (`0x460520`, `0x46052c`, `0x460534`); an empty slot is
/// `"UNKNOWN"` (`0x5e7ae0`).
const NAMES: [&str; 3] = ["EXHAUSTION", "BREATH", "FEIGNDEATH"];

/// One slot, as 3.3.5's `MirrorTimerInfo`: values in ms, `scale` in ms of value per ms.
#[derive(Clone, Default)]
struct Timer {
    value: i64,
    max_value: i64,
    scale: i64,
    paused: bool,
    spell_id: u32,
    label: String,
    /// `GetTime()` at the start, in seconds.
    start: f64,
}

#[derive(Default)]
struct State([Option<Timer>; 3]);

/// A `SMSG_START_MIRROR_TIMER` of wire type `kind`, captioned `label` (the event's arg6).
#[allow(clippy::too_many_arguments)]
pub fn start(
    lua: &Lua,
    kind: u32,
    value: i64,
    max_value: i64,
    scale: i64,
    paused: bool,
    spell_id: u32,
    label: &str,
) {
    let start = crate::script::clock::now(lua);
    let Some(mut state) = lua.app_data_mut::<State>() else {
        return;
    };
    if let Some(slot) = state.0.get_mut(kind as usize) {
        *slot = Some(Timer {
            value,
            max_value,
            scale,
            paused,
            spell_id,
            label: label.to_owned(),
            start,
        });
    }
}

/// A `SMSG_STOP_MIRROR_TIMER`, or `StopMirrorTimers` on release: the slot reads empty.
pub fn stop(lua: &Lua, kind: u32) {
    if let Some(mut state) = lua.app_data_mut::<State>() {
        if let Some(slot) = state.0.get_mut(kind as usize) {
            *slot = None;
        }
    }
}

/// 3.3.5's `value + scale·(now − start)` in whole ms, except that a timer started paused holds
/// its value (ClassicAPI's reading; 3.3.5 keeps extrapolating). An empty slot reads 0.
fn progress(timer: Option<&Timer>, now: f64) -> i64 {
    match timer {
        None => 0,
        Some(t) if t.paused => t.value,
        Some(t) => {
            let elapsed_ms = ((now - t.start) * 1000.0).floor().max(0.0) as i64;
            t.value + t.scale * elapsed_ms
        }
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(State::default());
    let globals = lua.globals();
    // Retail's seven values; an index outside 1..3 returns nothing.
    globals.set(
        "GetMirrorTimerInfo",
        lua.create_function(|lua, index: Value| {
            let index = match index {
                Value::Integer(n) => n,
                Value::Number(n) => n as i64,
                _ => return Err(mlua::Error::runtime("Usage: GetMirrorTimerInfo(index)")),
            };
            let Some(slot) = index.checked_sub(1).filter(|i| (0..3).contains(i)) else {
                return Ok(MultiValue::new());
            };
            let state = lua.app_data_ref::<State>().expect("mirror timer state");
            let timer = state.0[slot as usize].as_ref();
            let name = timer.map_or("UNKNOWN", |_| NAMES[slot as usize]);
            Ok(MultiValue::from_vec(vec![
                Value::String(lua.create_string(name)?),
                Value::Integer(timer.map_or(0, |t| t.value)),
                Value::Integer(timer.map_or(0, |t| t.max_value)),
                Value::Integer(timer.map_or(0, |t| t.scale)),
                Value::Integer(timer.map_or(0, |t| i64::from(t.paused))),
                Value::String(lua.create_string(timer.map_or("", |t| t.label.as_str()))?),
                Value::Integer(timer.map_or(0, |t| i64::from(t.spell_id))),
            ]))
        })?,
    )?;
    // The name matches case-insensitively, as 3.3.5's `MirrorTimerToIndex`; another name returns
    // nil.
    globals.set(
        "GetMirrorTimerProgress",
        lua.create_function(|lua, name: Value| {
            let name = match &name {
                Value::String(s) => s.to_string_lossy(),
                Value::Integer(n) => n.to_string(),
                Value::Number(n) => n.to_string(),
                _ => {
                    return Err(mlua::Error::runtime(
                        "Usage: GetMirrorTimerProgress(\"timer\")",
                    ))
                }
            };
            let Some(slot) = NAMES.iter().position(|n| n.eq_ignore_ascii_case(&name)) else {
                return Ok(None);
            };
            let now = crate::script::clock::now(lua);
            let state = lua.app_data_ref::<State>().expect("mirror timer state");
            Ok(Some(progress(state.0[slot].as_ref(), now)))
        })?,
    )
}
