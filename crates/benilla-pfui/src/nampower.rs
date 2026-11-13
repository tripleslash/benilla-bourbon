//! Nampower 3's pfUI event contract. No Windows code or DLL is loaded.

use benilla_ui::script::{pfui_host, ScriptValue};
use mlua::Lua;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aura {
    pub slot: u8,
    pub spell: u32,
    pub stacks: u8,
    pub level: u8,
    pub visible: bool,
}

#[derive(Clone, Copy)]
struct Application {
    caster: u64,
    duration: f64,
    expires: f64,
    observed: f64,
}

#[derive(Default)]
struct State {
    auras: HashMap<u64, Vec<Aura>>,
    applications: HashMap<(u64, u32), Application>,
    targets: HashMap<u64, u64>,
    gcd_end: f64,
    health: HashMap<u64, u32>,
    durations: HashMap<u8, (u32, f64)>,
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(State::default());
    let bridge = lua.create_table()?;
    let actions = lua.named_registry_value::<mlua::Table>("benilla.pfui.attribute_actions")?;
    bridge.set("namedCast", actions.get::<mlua::Function>("namedCast")?)?;
    bridge.set("bookCast", actions.get::<mlua::Function>("bookCast")?)?;
    bridge.set("queueScript", actions.get::<mlua::Function>("queueScript")?)?;
    bridge.set("cancelQueue", actions.get::<mlua::Function>("cancelQueue")?)?;
    bridge.set(
        "auraDuration",
        lua.create_function(|lua, slot: i64| {
            if !(0..48).contains(&slot) || player(lua) == 0 {
                return Ok(None);
            }
            let slot = slot as u8;
            let state = lua.app_data_ref::<State>().expect("Nampower state");
            let spell = state
                .auras
                .get(&player(lua))
                .and_then(|auras| auras.iter().find(|a| a.slot == slot))
                .map_or(0, |a| a.spell);
            let end = state
                .durations
                .get(&slot)
                .filter(|(id, _)| spell != 0 && (*id == 0 || *id == spell))
                .map_or(0.0, |(_, end)| *end);
            let result = lua.create_table()?;
            result.raw_set(1, spell)?;
            result.raw_set(2, ((end - now(lua)).max(0.0) * 1000.0).floor())?;
            result.raw_set(3, (end * 1000.0).floor())?;
            Ok(Some(result))
        })?,
    )?;
    bridge.set(
        "castInfo",
        lua.create_function(|lua, ()| {
            let Some(cast) = pfui_host::casts::player_cast(lua) else {
                return Ok(None);
            };
            let now = now(lua);
            let info = lua.create_table()?;
            let target = lua
                .app_data_ref::<State>()
                .and_then(|s| s.targets.get(&player(lua)).copied())
                .unwrap_or(0);
            info.set(
                "castId",
                cast.cast_guid
                    .rsplit('-')
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0),
            )?;
            info.set("spellId", cast.spell_id)?;
            info.set("guid", format!("0x{target:016X}"))?;
            info.set("castType", if cast.channel { 3 } else { 0 })?;
            info.set("castStartS", cast.start_ms / 1000.0)?;
            info.set("castEndS", cast.end_ms / 1000.0)?;
            info.set("castRemainingMs", (cast.end_ms - now * 1000.0).max(0.0))?;
            info.set("castDurationMs", (cast.end_ms - cast.start_ms).max(0.0))?;
            let end = lua.app_data_ref::<State>().map_or(0.0, |s| s.gcd_end);
            info.set("gcdEndS", end)?;
            info.set("gcdRemainingMs", ((end - now) * 1000.0).max(0.0))?;
            Ok(Some(info))
        })?,
    )?;
    lua.load(include_str!("nampower.lua"))
        .set_name("@benilla-pfui/nampower.lua")
        .call::<()>(bridge)
}

pub fn installed(lua: &Lua) -> bool {
    lua.app_data_ref::<State>().is_some()
}

pub fn set_gcd_end(lua: &Lua, end: f64) {
    if let Some(mut state) = lua.app_data_mut::<State>() {
        state.gcd_end = end;
    }
}

pub fn update_duration(lua: &Lua, slot: u8, spell: u32, remaining_ms: u32) {
    let end = if remaining_ms == 0 {
        0.0
    } else {
        now(lua) + f64::from(remaining_ms) / 1000.0
    };
    if let Some(mut state) = lua.app_data_mut::<State>() {
        state.durations.insert(slot, (spell, end));
    }
}

pub fn health(lua: &Lua, guid: u64, health: u32) {
    let previous = lua
        .app_data_mut::<State>()
        .and_then(|mut state| state.health.insert(guid, health));
    if health == 0 && previous.is_some_and(|health| health > 0) {
        fire(
            lua,
            None,
            "UNIT_DIED",
            vec![ScriptValue::Str(format!("0x{guid:016X}"))],
        );
    }
}

pub fn player(lua: &Lua) -> u64 {
    lua.globals()
        .get::<mlua::Function>("GetPlayerGuid")
        .and_then(|f| f.call::<Option<String>>(()))
        .ok()
        .flatten()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0)
}

pub fn now(lua: &Lua) -> f64 {
    lua.globals()
        .get::<mlua::Function>("GetTime")
        .and_then(|f| f.call::<f64>(()))
        .unwrap_or(0.0)
}

pub fn cvar(lua: &Lua, name: &str) -> f64 {
    lua.globals()
        .get::<mlua::Function>("GetCVar")
        .and_then(|f| f.call::<String>(name))
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

pub fn fire(lua: &Lua, gate: Option<&str>, event: &str, args: Vec<ScriptValue>) {
    if gate.is_none_or(|name| cvar(lua, name) != 0.0) {
        pfui_host::fire_pfui_event(lua, event, args);
    }
}

pub fn target(lua: &Lua, caster: u64, target: u64) {
    if let Some(mut state) = lua.app_data_mut::<State>() {
        state.targets.insert(caster, target);
    }
}

/// A successful spell-go names its hit targets; missed targets never acquire ownership.
pub fn application(lua: &Lua, caster: u64, target: u64, spell: u32, duration_ms: i32) {
    let now = now(lua);
    if let Some(mut state) = lua.app_data_mut::<State>() {
        state.applications.insert(
            (target, spell),
            Application {
                caster,
                duration: f64::from(duration_ms.max(0)) / 1000.0,
                expires: if duration_ms > 0 {
                    now + f64::from(duration_ms) / 1000.0
                } else {
                    0.0
                },
                observed: now,
            },
        );
    }
}

/// Descriptor edges use raw slots; Lua indices skip hidden auras and holes in each half.
pub fn auras(lua: &Lua, guid: u64, incoming: Vec<Aura>) {
    auras_at(lua, guid, incoming, now(lua), player(lua));
}

/// [`auras`] with [`now`] and [`player`] given: each is a call into the VM, which the host makes
/// once for a frame's units.
pub fn auras_at(lua: &Lua, guid: u64, mut incoming: Vec<Aura>, now: f64, me: u64) {
    incoming.sort_by_key(|a| a.slot);
    let (previous, annotations) = {
        let Some(mut state) = lua.app_data_mut::<State>() else {
            return;
        };
        let previous = state
            .auras
            .insert(guid, incoming.clone())
            .unwrap_or_default();
        if guid == me {
            for old in &previous {
                if !incoming.iter().any(|a| a.slot == old.slot) {
                    state.durations.remove(&old.slot);
                }
            }
            for aura in &incoming {
                if let Some((id, _)) = state.durations.get_mut(&aura.slot) {
                    // Duration packets can precede the descriptor that names a new aura.
                    if *id == 0 {
                        *id = aura.spell;
                    }
                }
            }
        }
        let annotations: Vec<_> = incoming
            .iter()
            .filter_map(|a| {
                let applied = *state.applications.get(&(guid, a.spell))?;
                let existing = previous
                    .iter()
                    .any(|old| old.slot == a.slot && old.spell == a.spell);
                // Do not assign an old cast to an unrelated application in a recycled slot.
                (existing || now - applied.observed < 5.0).then_some((a.spell, applied))
            })
            .collect();
        for old in &previous {
            if !incoming.iter().any(|a| a.spell == old.spell) {
                state.applications.remove(&(guid, old.spell));
            }
        }
        (previous, annotations)
    };
    for old in &previous {
        if !incoming
            .iter()
            .any(|aura| aura.slot == old.slot && aura.spell == old.spell)
        {
            pfui_host::auras::forget_annotation(lua, guid, old.slot, old.spell);
        }
    }
    for (spell, app) in annotations {
        pfui_host::auras::annotate(lua, guid, spell, app.caster, app.duration, app.expires);
    }
    let index = |list: &[Aura], aura: &Aura| -> i64 {
        if !aura.visible {
            return 0;
        }
        list.iter()
            .filter(|a| a.visible && (a.slot < 32) == (aura.slot < 32))
            .position(|a| a.slot == aura.slot)
            .map_or(0, |i| i as i64 + 1)
    };
    let mut events = Vec::new();
    for old in &previous {
        if !incoming
            .iter()
            .any(|a| a.slot == old.slot && a.spell == old.spell)
        {
            events.push((
                Aura {
                    stacks: 0,
                    level: 0,
                    ..*old
                },
                index(&previous, old),
                false,
                1,
            ));
        }
    }
    for new in &incoming {
        match previous
            .iter()
            .find(|a| a.slot == new.slot && a.spell == new.spell)
        {
            None => events.push((*new, index(&incoming, new), true, 0)),
            Some(old) if old.stacks != new.stacks => {
                events.push((*new, index(&incoming, new), new.stacks > old.stacks, 2))
            }
            _ => (),
        }
    }
    for (aura, index, added, state) in events {
        let event = format!(
            "{}_{}_{}",
            if aura.slot < 32 { "BUFF" } else { "DEBUFF" },
            if added { "ADDED" } else { "REMOVED" },
            if guid == me { "SELF" } else { "OTHER" }
        );
        fire(
            lua,
            None,
            &event,
            vec![
                ScriptValue::Str(format!("0x{guid:016X}")),
                ScriptValue::Int(index),
                ScriptValue::Int(i64::from(aura.spell)),
                ScriptValue::Int(i64::from(aura.stacks)),
                ScriptValue::Int(i64::from(aura.level)),
                ScriptValue::Int(i64::from(aura.slot)),
                ScriptValue::Int(state),
            ],
        );
    }
}

pub fn retain_units(lua: &Lua, present: &HashSet<u64>) {
    if let Some(mut state) = lua.app_data_mut::<State>() {
        state.auras.retain(|guid, _| present.contains(guid));
        state
            .applications
            .retain(|(guid, _), _| present.contains(guid));
        state.targets.retain(|guid, _| present.contains(guid));
        state.health.retain(|guid, _| present.contains(guid));
    }
}

pub fn reset(lua: &Lua) {
    lua.set_app_data(State::default());
}

#[cfg(test)]
#[path = "nampower_tests.rs"]
mod tests;
