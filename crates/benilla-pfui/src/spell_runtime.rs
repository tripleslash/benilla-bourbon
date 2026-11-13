//! Gameplay queries read snapshots built with the engine's stock spell calculations.

use benilla_ui::script::pfui_host::units;
use benilla_ui::script::CooldownQuery;
use mlua::{Lua, Table, Value};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Cost {
    pub kind: i32,
    pub amount: u32,
    pub percent: u32,
}

pub type Cooldown = (i64, u32, bool);
pub type RangeQuery = Box<dyn Fn(u32, u64) -> Option<bool>>;

#[derive(Default)]
struct Runtime {
    costs: HashMap<u32, Cost>,
    cooldowns: HashMap<u32, Cooldown>,
    /// How often [`set_cooldowns`] replaced the table: it holds a row per catalog spell, so the
    /// host's feed must not rebuild it every frame.
    cooldown_feeds: u64,
    usable: HashMap<u32, (bool, bool)>,
    range: Option<RangeQuery>,
}

/// Whether this VM carries the spell getters: without them nothing reads the snapshots, and the
/// host's feed skips building them.
pub fn installed(lua: &Lua) -> bool {
    lua.app_data_ref::<Runtime>().is_some()
}

pub fn set_costs(lua: &Lua, costs: HashMap<u32, Cost>) {
    if let Some(mut state) = lua.app_data_mut::<Runtime>() {
        state.costs = costs;
    }
}
/// The rows `GetSpellCooldown` answers from where no host is lent to ask, and its list of the
/// spells it knows: with a host it reads the list as of the call, so the rows need not follow it
/// frame by frame.
pub fn set_cooldowns(lua: &Lua, cooldowns: HashMap<u32, Cooldown>) {
    if let Some(mut state) = lua.app_data_mut::<Runtime>() {
        state.cooldowns = cooldowns;
        state.cooldown_feeds += 1;
    }
}
/// [`Runtime::cooldown_feeds`], for the host's tests.
pub fn cooldown_feeds(lua: &Lua) -> u64 {
    lua.app_data_ref::<Runtime>()
        .map_or(0, |state| state.cooldown_feeds)
}
pub fn set_usable(lua: &Lua, usable: HashMap<u32, (bool, bool)>) {
    if let Some(mut state) = lua.app_data_mut::<Runtime>() {
        state.usable = usable;
    }
}
pub fn set_range_query(lua: &Lua, range: Option<RangeQuery>) {
    if let Some(mut state) = lua.app_data_mut::<Runtime>() {
        state.range = range;
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Runtime::default());
    let api: Table = lua.globals().get("C_Spell")?;
    api.set(
        "GetSpellPowerCost",
        lua.create_function(|lua, input: Value| {
            let Some(id) = units::spell_identifier(lua, input)? else {
                return Ok(None);
            };
            let state = lua.app_data_ref::<Runtime>().expect("spell runtime");
            let Some(cost) = state.costs.get(&id) else {
                return Ok(None);
            };
            if cost.amount == 0 && cost.percent == 0 {
                return Ok(None);
            }
            let entry = lua.create_table()?;
            entry.set("type", cost.kind)?;
            let name = match cost.kind {
                -2 => "HEALTH",
                0 => "MANA",
                1 => "RAGE",
                2 => "FOCUS",
                3 => "ENERGY",
                4 => "HAPPINESS",
                _ => "NONE",
            };
            entry.set("name", name)?;
            entry.set("cost", cost.amount)?;
            entry.set("minCost", cost.amount)?;
            entry.set("costPercent", cost.percent)?;
            entry.set("costPerSec", 0)?;
            entry.set("requiredAuraID", 0)?;
            entry.set("hasRequiredAura", false)?;
            let result = lua.create_table()?;
            result.set(1, entry)?;
            Ok(Some(result))
        })?,
    )?;
    api.set(
        "GetSpellCooldown",
        lua.create_function(|lua, input: Value| {
            let Some(id) = units::spell_identifier(lua, input)? else {
                return Ok(None);
            };
            let fed = lua
                .app_data_ref::<Runtime>()
                .expect("spell runtime")
                .cooldowns
                .get(&id)
                .copied();
            // No row: a spell the catalog lacks.
            let Some(fed) = fed else {
                return Ok(None);
            };
            // The player's list as of this call, as the stock `GetSpellCooldown` reads it
            // (`CooldownQuery::Spell`); the fed row where no host is lent.
            let (start, duration, enabled) =
                benilla_ui::script::live::ask(lua, &CooldownQuery::Spell(id))
                    .map_or(fed, |live| live.unwrap_or((0, 0, true)));
            let now: f64 = lua.globals().get::<mlua::Function>("GetTime")?.call(())?;
            let (start, duration) =
                if enabled && now >= (start as f64 + f64::from(duration)) / 1000.0 {
                    (0.0, 0.0)
                } else {
                    (start as f64 / 1000.0, f64::from(duration) / 1000.0)
                };
            let result = lua.create_table()?;
            result.set("startTime", start)?;
            result.set("duration", duration)?;
            result.set("isEnabled", enabled)?;
            result.set("modRate", 1.0)?;
            Ok(Some(result))
        })?,
    )?;
    api.set(
        "IsSpellUsable",
        lua.create_function(|lua, input: Value| {
            let id = units::spell_identifier(lua, input)?;
            Ok(id
                .and_then(|id| {
                    lua.app_data_ref::<Runtime>()
                        .expect("spell runtime")
                        .usable
                        .get(&id)
                        .copied()
                })
                .unwrap_or((false, false)))
        })?,
    )?;
    api.set(
        "IsSpellInRange",
        lua.create_function(|lua, (input, token): (Value, String)| {
            let guid = units::guid(lua, &token)?;
            let id = units::spell_identifier(lua, input)?;
            let state = lua.app_data_ref::<Runtime>().expect("spell runtime");
            Ok(id
                .zip(guid)
                .and_then(|(id, guid)| state.range.as_ref()?.as_ref()(id, guid)))
        })?,
    )
}
