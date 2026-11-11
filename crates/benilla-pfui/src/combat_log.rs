//! The combat log a ClassicAPI client gives addons: `COMBAT_LOG_EVENT_UNFILTERED` for every combat
//! event the client sees ([`record`]), `COMBAT_LOG_EVENT` for those its filters pass, the event's
//! values while a handler runs (`CombatLogGetCurrentEventInfo`), and the kept entries.

use benilla_ui::script::{pfui_host, ScriptValue};
use mlua::{Lua, MultiValue, Value};
use std::collections::VecDeque;

/// `COMBATLOG_OBJECT_NONE`: the flags of an event's missing source or target.
pub const NONE: u32 = 0x8000_0000;
pub const AFFILIATION_MINE: u32 = 0x1;
pub const AFFILIATION_PARTY: u32 = 0x2;
pub const AFFILIATION_RAID: u32 = 0x4;
pub const AFFILIATION_OUTSIDER: u32 = 0x8;
pub const REACTION_FRIENDLY: u32 = 0x10;
pub const REACTION_NEUTRAL: u32 = 0x20;
pub const REACTION_HOSTILE: u32 = 0x40;
pub const CONTROL_PLAYER: u32 = 0x100;
pub const CONTROL_NPC: u32 = 0x200;
pub const TYPE_PLAYER: u32 = 0x400;
pub const TYPE_NPC: u32 = 0x800;
pub const TYPE_PET: u32 = 0x1000;
pub const TYPE_GUARDIAN: u32 = 0x2000;
pub const TYPE_OBJECT: u32 = 0x4000;
pub const TARGET: u32 = 0x1_0000;
pub const FOCUS: u32 = 0x2_0000;
const SPECIAL: u32 = TARGET | FOCUS | 0x4_0000 | 0x8_0000;
const GROUPS: [u32; 4] = [0xF, 0xF0, 0x300, 0xFC00];

/// A source or target as the log names it.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub guid: u64,
    /// `None` reads as `UNKNOWNOBJECT`.
    pub name: Option<String>,
    pub flags: u32,
    pub raid_flags: u32,
}

/// One combat event: its subevent, its two ends, the spell it names and the values after the
/// eleven common ones (the prefix's, then the suffix's).
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub subevent: &'static str,
    pub source: Option<Unit>,
    pub dest: Option<Unit>,
    pub spell: Option<(u32, Option<String>)>,
    pub values: Vec<ScriptValue>,
}

struct Entry {
    at: f64,
    event: Event,
    args: Vec<ScriptValue>,
}

enum Who {
    Mask(u32),
    Guid(String),
}

enum Spell {
    Id(u32),
    Name(String),
}

struct Filter {
    events: Option<Vec<String>>,
    source: Option<Who>,
    dest: Option<Who>,
    spell: Option<Spell>,
}

struct Log {
    entries: VecDeque<Entry>,
    filters: Vec<Filter>,
    /// The selected entry, an index into `entries`.
    current: Option<usize>,
    retention: f64,
    /// The events whose handlers run now, innermost last.
    dispatching: Vec<Vec<ScriptValue>>,
}

impl Default for Log {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            filters: Vec::new(),
            current: None,
            retention: 300.0,
            dispatching: Vec::new(),
        }
    }
}

/// Whether `flags` match `mask` (`CombatLog_Object_IsA`): a target, focus or main-tank/assist bit
/// they share, else one bit of each of the four groups; a missing unit by the none bit.
pub fn object_is_a(flags: u32, mask: u32) -> bool {
    if flags & NONE != 0 {
        return mask & NONE != 0;
    }
    if flags & mask & SPECIAL != 0 {
        return true;
    }
    GROUPS.iter().all(|group| flags & mask & group != 0)
}

fn complete(mask: u32) -> bool {
    mask & (SPECIAL | NONE) != 0 || GROUPS.iter().all(|group| mask & group != 0)
}

fn guid_text(guid: u64) -> String {
    format!("0x{guid:016X}")
}

impl Who {
    fn matches(&self, unit: Option<&Unit>) -> bool {
        match self {
            Who::Mask(mask) => object_is_a(unit.map_or(NONE, |u| u.flags), *mask),
            Who::Guid(guid) => unit.is_some_and(|u| guid_text(u.guid).eq_ignore_ascii_case(guid)),
        }
    }
}

impl Filter {
    fn matches(&self, event: &Event) -> bool {
        self.events.as_ref().is_none_or(|names| {
            names
                .iter()
                .any(|name| name.eq_ignore_ascii_case(event.subevent))
        }) && self
            .source
            .as_ref()
            .is_none_or(|w| w.matches(event.source.as_ref()))
            && self
                .dest
                .as_ref()
                .is_none_or(|w| w.matches(event.dest.as_ref()))
            && self.spell.as_ref().is_none_or(|s| match (s, &event.spell) {
                (Spell::Id(id), Some((spell, _))) => id == spell,
                (Spell::Name(name), Some((_, Some(spell)))) => spell.eq_ignore_ascii_case(name),
                _ => false,
            })
    }
}

impl Log {
    fn passes(&self, event: &Event) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| f.matches(event))
    }

    /// The indices of the entries a call counts, oldest first.
    fn counted(&self, ignore_filter: bool) -> Vec<usize> {
        (0..self.entries.len())
            .filter(|&i| ignore_filter || self.passes(&self.entries[i].event))
            .collect()
    }
}

fn args_of(timestamp: f64, event: &Event) -> Vec<ScriptValue> {
    let end = |unit: Option<&Unit>| -> [ScriptValue; 4] {
        match unit {
            Some(u) => [
                ScriptValue::Str(guid_text(u.guid)),
                u.name
                    .clone()
                    .map_or(ScriptValue::Str("Unknown".into()), ScriptValue::Str),
                ScriptValue::Int(i64::from(u.flags)),
                ScriptValue::Int(i64::from(u.raid_flags)),
            ],
            None => [
                ScriptValue::Str(String::new()),
                ScriptValue::Nil,
                ScriptValue::Int(i64::from(NONE)),
                ScriptValue::Int(0),
            ],
        }
    };
    let mut args = vec![
        ScriptValue::Number(timestamp),
        ScriptValue::Str(event.subevent.to_string()),
        ScriptValue::Bool(event.source.is_none()),
    ];
    args.extend(end(event.source.as_ref()));
    args.extend(end(event.dest.as_ref()));
    args.extend(event.values.iter().cloned());
    args
}

fn now(lua: &Lua) -> f64 {
    lua.globals()
        .get::<mlua::Function>("GetTime")
        .and_then(|f| f.call::<f64>(()))
        .unwrap_or(0.0)
}

/// The wall clock through `web-time`: std's on native, `Date.now()` on wasm32, where std's panics.
fn unix_time() -> f64 {
    web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .map_or(0.0, |d| (d.as_millis() as f64) / 1000.0)
}

/// Keep `event` and raise `COMBAT_LOG_EVENT_UNFILTERED` with its values, then `COMBAT_LOG_EVENT`
/// when the filters pass it.
pub fn record(lua: &Lua, event: Event) {
    let at = now(lua);
    let args = args_of(unix_time(), &event);
    let passes = {
        let Some(mut log) = lua.app_data_mut::<Log>() else {
            return;
        };
        while log
            .entries
            .front()
            .is_some_and(|e| at - e.at > log.retention)
        {
            log.entries.pop_front();
            // The selection keeps its entry; when its own goes, it moves to the next one, which
            // takes the front.
            log.current = log.current.map(|c| c.saturating_sub(1));
        }
        let passes = log.passes(&event);
        log.entries.push_back(Entry {
            at,
            event,
            args: args.clone(),
        });
        passes
    };
    // The event's values stay current around its raises, which wait for the host when none is
    // lent.
    pfui_host::run_hosted(lua, move |lua| {
        if let Some(mut log) = lua.app_data_mut::<Log>() {
            log.dispatching.push(args.clone());
        }
        pfui_host::fire_pfui_event(lua, "COMBAT_LOG_EVENT_UNFILTERED", args.clone());
        if passes {
            pfui_host::fire_pfui_event(lua, "COMBAT_LOG_EVENT", args);
        }
        if let Some(mut log) = lua.app_data_mut::<Log>() {
            log.dispatching.pop();
        }
    });
}

/// Forget the entries and filters: a logout.
pub fn clear(lua: &Lua) {
    if let Some(mut log) = lua.app_data_mut::<Log>() {
        *log = Log::default();
    }
}

fn lua_value(lua: &Lua, v: &ScriptValue) -> mlua::Result<Value> {
    Ok(match v {
        ScriptValue::Nil => Value::Nil,
        ScriptValue::Bool(b) => Value::Boolean(*b),
        ScriptValue::Int(i) => Value::Integer(*i),
        ScriptValue::Number(n) => Value::Number(*n),
        ScriptValue::Str(s) => Value::String(lua.create_string(s)?),
    })
}

fn multi(lua: &Lua, args: &[ScriptValue]) -> mlua::Result<MultiValue> {
    args.iter()
        .map(|v| lua_value(lua, v))
        .collect::<mlua::Result<Vec<_>>>()
        .map(MultiValue::from_vec)
}

fn truthy(v: &Value) -> bool {
    !matches!(v, Value::Nil | Value::Boolean(false))
}

fn who(lua: &Lua, v: Value, which: &str) -> mlua::Result<Option<Who>> {
    match v {
        Value::Nil => Ok(None),
        Value::String(s) => Ok(Some(Who::Guid(s.to_string_lossy()))),
        other => {
            let mask = lua.coerce_number(other)?.ok_or_else(|| {
                mlua::Error::runtime("Usage: CombatLogAddFilter(events, source, dest, spell)")
            })? as i64 as u32;
            if !complete(mask) {
                return Err(mlua::Error::runtime(format!(
                    "CombatLogAddFilter(): incomplete filter for {which}"
                )));
            }
            Ok(Some(Who::Mask(mask)))
        }
    }
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Log::default());
    let g = lua.globals();
    g.set(
        "CombatLogGetCurrentEventInfo",
        lua.create_function(|lua, ()| {
            let top = lua
                .app_data_ref::<Log>()
                .and_then(|log| log.dispatching.last().cloned());
            multi(lua, &top.unwrap_or_default())
        })?,
    )?;
    g.set(
        "CombatLogAddFilter",
        lua.create_function(
            |lua, (events, source, dest, spell): (Value, Value, Value, Value)| {
                let events = match events {
                    Value::String(s) => Some(
                        s.to_string_lossy()
                            .split([' ', ','])
                            .filter(|e| !e.is_empty())
                            .map(str::to_string)
                            .collect(),
                    ),
                    _ => None,
                };
                let source = who(lua, source, "srcMask")?;
                let dest = who(lua, dest, "dstMask")?;
                let spell = match spell {
                    Value::Nil => None,
                    Value::String(s) => Some(Spell::Name(s.to_string_lossy())),
                    other => lua.coerce_number(other)?.map(|n| Spell::Id(n as u32)),
                };
                if let Some(mut log) = lua.app_data_mut::<Log>() {
                    log.filters.push(Filter {
                        events,
                        source,
                        dest,
                        spell,
                    });
                }
                Ok(())
            },
        )?,
    )?;
    g.set(
        "CombatLogResetFilter",
        lua.create_function(|lua, ()| {
            if let Some(mut log) = lua.app_data_mut::<Log>() {
                log.filters.clear();
            }
            Ok(())
        })?,
    )?;
    g.set(
        "CombatLog_Object_IsA",
        lua.create_function(|lua, (flags, mask): (Value, Value)| {
            let flags = lua.coerce_number(flags)?.unwrap_or(0.0) as i64 as u32;
            let mask = lua.coerce_number(mask)?.unwrap_or(0.0) as i64 as u32;
            Ok(object_is_a(flags, mask).then_some(1))
        })?,
    )?;
    g.set(
        "CombatLogGetRetentionTime",
        lua.create_function(|lua, ()| {
            Ok(lua.app_data_ref::<Log>().map_or(300.0, |log| log.retention))
        })?,
    )?;
    g.set(
        "CombatLogSetRetentionTime",
        lua.create_function(|lua, seconds: f64| {
            if let Some(mut log) = lua.app_data_mut::<Log>() {
                log.retention = seconds.max(0.0);
            }
            Ok(())
        })?,
    )?;
    g.set(
        "CombatLogGetNumEntries",
        lua.create_function(|lua, ignore: Value| {
            Ok(lua
                .app_data_ref::<Log>()
                .map_or(0, |log| log.counted(truthy(&ignore)).len()))
        })?,
    )?;
    // Index 1 is the oldest entry the filter counts, 0 the newest and -1 the one before it.
    g.set(
        "CombatLogSetCurrentEntry",
        lua.create_function(|lua, (index, ignore): (i64, Value)| {
            let Some(mut log) = lua.app_data_mut::<Log>() else {
                return Ok(None);
            };
            let counted = log.counted(truthy(&ignore));
            let at = if index >= 1 {
                usize::try_from(index - 1).ok()
            } else {
                usize::try_from(counted.len() as i64 - 1 + index).ok()
            };
            log.current = at.and_then(|i| counted.get(i).copied());
            Ok(log.current.map(|_| 1))
        })?,
    )?;
    g.set(
        "CombatLogAdvanceEntry",
        lua.create_function(|lua, (count, ignore): (i64, Value)| {
            let Some(mut log) = lua.app_data_mut::<Log>() else {
                return Ok(None);
            };
            let counted = log.counted(truthy(&ignore));
            let Some(current) = log.current else {
                return Ok(None);
            };
            let position = counted
                .iter()
                .position(|&i| i >= current)
                .unwrap_or(counted.len());
            log.current = usize::try_from(position as i64 + count)
                .ok()
                .and_then(|p| counted.get(p).copied());
            Ok(log.current.map(|_| 1))
        })?,
    )?;
    g.set(
        "CombatLogGetCurrentEntry",
        lua.create_function(|lua, ()| {
            let args = lua.app_data_ref::<Log>().and_then(|log| {
                log.current
                    .and_then(|i| log.entries.get(i))
                    .map(|e| e.args.clone())
            });
            multi(lua, &args.unwrap_or_default())
        })?,
    )?;
    g.set(
        "CombatLogClearEntries",
        lua.create_function(|lua, ()| {
            if let Some(mut log) = lua.app_data_mut::<Log>() {
                log.entries.clear();
                log.current = None;
            }
            Ok(())
        })?,
    )?;
    lua.load(CONSTANTS)
        .set_name("@benilla-pfui/combat_log.lua")
        .exec()
}

/// The flag constants, which the ClassicAPI addon defines again with the same values.
const CONSTANTS: &str = r#"
local function set(name, value) if getglobal(name) == nil then setglobal(name, value) end end
set("COMBATLOG_OBJECT_AFFILIATION_MINE", 1) set("COMBATLOG_OBJECT_AFFILIATION_PARTY", 2)
set("COMBATLOG_OBJECT_AFFILIATION_RAID", 4) set("COMBATLOG_OBJECT_AFFILIATION_OUTSIDER", 8)
set("COMBATLOG_OBJECT_AFFILIATION_MASK", 15)
set("COMBATLOG_OBJECT_REACTION_FRIENDLY", 16) set("COMBATLOG_OBJECT_REACTION_NEUTRAL", 32)
set("COMBATLOG_OBJECT_REACTION_HOSTILE", 64) set("COMBATLOG_OBJECT_REACTION_MASK", 240)
set("COMBATLOG_OBJECT_CONTROL_PLAYER", 256) set("COMBATLOG_OBJECT_CONTROL_NPC", 512)
set("COMBATLOG_OBJECT_CONTROL_MASK", 768)
set("COMBATLOG_OBJECT_TYPE_PLAYER", 1024) set("COMBATLOG_OBJECT_TYPE_NPC", 2048)
set("COMBATLOG_OBJECT_TYPE_PET", 4096) set("COMBATLOG_OBJECT_TYPE_GUARDIAN", 8192)
set("COMBATLOG_OBJECT_TYPE_OBJECT", 16384) set("COMBATLOG_OBJECT_TYPE_MASK", 64512)
set("COMBATLOG_OBJECT_TARGET", 65536) set("COMBATLOG_OBJECT_FOCUS", 131072)
set("COMBATLOG_OBJECT_MAINTANK", 262144) set("COMBATLOG_OBJECT_MAINASSIST", 524288)
set("COMBATLOG_OBJECT_SPECIAL_MASK", 983040) set("COMBATLOG_OBJECT_NONE", 2147483648)
local bits = {1, 2, 4, 8, 16, 32, 64, 128}
for i = 1, 8 do set("COMBATLOG_OBJECT_RAIDTARGET" .. i, bits[i]) end
set("COMBATLOG_OBJECT_RAIDTARGET_MASK", 255)
local others = 2 + 4 + 8
local types = 1024 + 2048 + 4096 + 8192 + 16384
set("COMBATLOG_FILTER_ME", 1 + 16 + 256 + 1024 + 16384)
set("COMBATLOG_FILTER_MINE", 1 + 16 + 256 + 1024 + 16384 + 8192 + 4096)
set("COMBATLOG_FILTER_MY_PET", 1 + 16 + 256 + 8192 + 4096)
set("COMBATLOG_FILTER_FRIENDLY_UNITS", others + 16 + 256 + 512 + types)
set("COMBATLOG_FILTER_HOSTILE_PLAYERS", others + 64 + 256 + types)
set("COMBATLOG_FILTER_HOSTILE_UNITS", others + 64 + 512 + types)
set("COMBATLOG_FILTER_NEUTRAL_UNITS", others + 32 + 256 + 512 + types)
set("COMBATLOG_FILTER_UNKNOWN_UNITS", 2147483648)
set("COMBATLOG_FILTER_EVERYTHING", 4294967295)
"#;
