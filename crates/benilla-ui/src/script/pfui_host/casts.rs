//! Observed cast/channel snapshots and the modern lifecycle event projection.

use crate::script::ScriptValue;
use mlua::{Lua, MultiValue, Table, Value};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Cast {
    pub spell_id: u32,
    pub name: String,
    pub rank: String,
    pub icon: Option<String>,
    pub start_ms: f64,
    pub end_ms: f64,
    pub tradeskill: bool,
    pub cast_guid: String,
    pub delay_ms: u32,
    pub channel: bool,
    /// An interrupt or a silence can stop it (`PreventionType` silence).
    pub silenceable: bool,
}

#[derive(Default)]
struct Extras(HashMap<u64, Cast>);

/// Each caster's current cast's unit target, as its `SMSG_SPELL_START` named it.
#[derive(Default)]
struct Targets(HashMap<u64, u64>);

/// Replace the casts' unit targets, beside [`set_casts`].
pub fn set_cast_targets(lua: &Lua, targets: impl IntoIterator<Item = (u64, u64)>) {
    lua.set_app_data(Targets(targets.into_iter().collect()));
}

pub fn player_cast(lua: &Lua) -> Option<Cast> {
    let me = lua
        .app_data_ref::<crate::script::ModelCell>()?
        .unit_guids
        .player;
    lua.app_data_ref::<Extras>()?.0.get(&me).cloned()
}

pub fn set_casts(lua: &Lua, casts: impl IntoIterator<Item = (u64, Cast)>) {
    if let Some(mut extras) = lua.app_data_mut::<Extras>() {
        extras.0.clear();
        extras.0.extend(casts);
    }
}

fn info(lua: &Lua, token: &str, channel: bool) -> mlua::Result<MultiValue> {
    let guid = super::auras::unit_guid(lua, token);
    let Some(extras) = lua.app_data_ref::<Extras>() else {
        return Ok(MultiValue::new());
    };
    let Some((guid, cast)) = guid
        .and_then(|guid| extras.0.get(&guid).map(|cast| (guid, cast)))
        .filter(|(_, cast)| cast.channel == channel)
    else {
        return Ok(MultiValue::new());
    };
    let mut values = vec![
        Value::String(lua.create_string(&cast.name)?),
        Value::String(lua.create_string(&cast.name)?),
        cast.icon
            .as_ref()
            .map(|icon| lua.create_string(icon))
            .transpose()?
            .map_or(Value::Nil, Value::String),
        Value::Number(cast.start_ms),
        Value::Number(cast.end_ms),
        Value::Boolean(cast.tradeskill),
    ];
    if !channel {
        values.push(Value::String(lua.create_string(&cast.cast_guid)?));
    }
    values.push(Value::Boolean(super::interrupts::not_interruptible(
        lua, guid, cast,
    )));
    values.push(Value::Integer(i64::from(cast.spell_id)));
    if !channel {
        values.extend([Value::Nil, Value::Integer(i64::from(cast.delay_ms))]);
    }
    Ok(MultiValue::from_vec(values))
}

pub fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Extras::default());
    // `UnitSpellTargetName(unit)`: the name of the unit the unit's cast is aimed at; nil without
    // a cast, without a unit target (self, ground, item) or with nobody to name.
    lua.globals().set(
        "UnitSpellTargetName",
        lua.create_function(|lua, unit: String| {
            let Some(caster) = super::units::guid(lua, &unit)? else {
                return Ok(None);
            };
            let casting = lua
                .app_data_ref::<Extras>()
                .is_some_and(|extras| extras.0.contains_key(&caster));
            let target = lua
                .app_data_ref::<Targets>()
                .and_then(|targets| targets.0.get(&caster).copied());
            Ok(target
                .filter(|_| casting)
                .and_then(|target| super::units::name_of(lua, target)))
        })?,
    )?;
    let api: Table = lua.globals().get("C_Spell")?;
    api.set(
        "UnitCastingInfo",
        lua.create_function(|lua, unit: String| info(lua, &unit, false))?,
    )?;
    api.set(
        "UnitChannelInfo",
        lua.create_function(|lua, unit: String| info(lua, &unit, true))?,
    )?;
    api.set(
        "ChannelInfo",
        lua.create_function(|lua, ()| info(lua, "player", true))?,
    )?;
    Ok(())
}

/// Fire once for each current stock token naming the caster, preserving legacy argN values.
pub fn event(lua: &Lua, caster: u64, kind: &str, cast: &Cast) {
    for token in super::units::event_tokens(lua, caster) {
        crate::script::pfui_host::fire_pfui_event(
            lua,
            kind,
            vec![
                ScriptValue::Str(token),
                ScriptValue::Str(cast.cast_guid.clone()),
                ScriptValue::Int(i64::from(cast.spell_id)),
                ScriptValue::Str(cast.name.clone()),
                ScriptValue::Str(cast.rank.clone()),
            ],
        );
    }
}
