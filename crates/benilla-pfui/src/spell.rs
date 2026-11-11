//! pfUI's immutable spell lookups; gameplay-dependent queries live in the host bridge.

use crate::creature::{localized, number, numeric_id, rows, string};
use mlua::{Lua, Value};
use std::{collections::BTreeMap, rc::Rc};

mod legacy;
#[path = "spell_runtime.rs"]
pub mod runtime;
mod states;

#[derive(Clone, Debug)]
pub struct Spell {
    pub name: String,
    pub rank: String,
    pub icon: Option<String>,
    pub dispel: u32,
    pub attributes: u32,
    pub reagents: Vec<(u32, u32)>,
    pub effects: Vec<BTreeMap<&'static str, f64>>,
    pub cast_time: i32,
    pub min_range: f32,
    pub max_range: f32,
    pub cost: u32,
    pub power_type: i32,
    pub funnel: bool,
    /// How the spell moves the swing clock when it goes off.
    pub swing: Option<crate::swing::Kind>,
}

#[derive(Clone, Debug)]
pub struct Catalog(pub BTreeMap<u32, Spell>);

impl Catalog {
    pub fn read(
        read: &dyn Fn(&str) -> Option<Vec<u8>>,
        locale: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let slots = [
            "enUS", "koKR", "frFR", "deDE", "zhCN", "zhTW", "esES", "esMX",
        ];
        let locale = slots.iter().position(|name| *name == locale).unwrap_or(0);
        let spells = rows(read, "Spell", 173)?;
        let icons = rows(read, "SpellIcon", 2)?;
        let cast_times = rows(read, "SpellCastTimes", 4)?;
        let ranges = rows(read, "SpellRange", 22)?;
        let mut times = BTreeMap::new();
        for row in cast_times.records() {
            times.insert(number(row, 0)?, (number(row, 1)? as i32).max(0));
        }
        let mut limits = BTreeMap::new();
        for row in ranges.records() {
            limits.insert(
                number(row, 0)?,
                (
                    f32::from_bits(number(row, 1)?),
                    f32::from_bits(number(row, 2)?),
                ),
            );
        }
        let mut paths = BTreeMap::new();
        for icon in icons.records() {
            paths.insert(number(icon, 0)?, string(&icons, icon, 1)?);
        }
        let mut result = BTreeMap::new();
        for row in spells.records() {
            let mut reagents = Vec::new();
            for i in 0..8 {
                let id = number(row, 42 + i)? as i32;
                if id > 0 {
                    reagents.push((id as u32, number(row, 50 + i)?));
                }
            }
            let mut effects = Vec::new();
            for i in 0..3 {
                let mut effect = BTreeMap::new();
                for (name, col, signed, float) in [
                    ("effect", 61, false, false),
                    ("auraName", 91, false, false),
                    ("basePoints", 76, true, false),
                    ("baseDice", 67, true, false),
                    ("dieSides", 64, true, false),
                    ("pointsPerLevel", 73, false, true),
                    ("dicePerLevel", 70, false, true),
                    ("miscValue", 106, true, false),
                ] {
                    let value = number(row, col + i)?;
                    let value = if float {
                        f64::from(f32::from_bits(value))
                    } else if signed {
                        f64::from(value as i32)
                    } else {
                        f64::from(value)
                    };
                    effect.insert(name, value);
                }
                effects.push(effect);
            }
            result.insert(
                number(row, 0)?,
                Spell {
                    name: localized(&spells, row, 120, locale)?,
                    rank: localized(&spells, row, 129, locale)?,
                    icon: paths
                        .get(&number(row, 117)?)
                        .filter(|s| !s.is_empty())
                        .cloned(),
                    dispel: number(row, 4)?,
                    attributes: number(row, 6)?,
                    reagents,
                    effects,
                    cast_time: times.get(&number(row, 18)?).copied().unwrap_or(0),
                    min_range: limits.get(&number(row, 36)?).map_or(0.0, |range| range.0),
                    max_range: limits.get(&number(row, 36)?).map_or(0.0, |range| range.1),
                    cost: number(row, 32)?,
                    power_type: number(row, 31)? as i32,
                    funnel: number(row, 8)? & 0x800 != 0,
                    swing: crate::swing::Kind::of(
                        number(row, 6)?,
                        number(row, 8)?,
                        number(row, 21)?,
                    ),
                },
            );
        }
        Ok(Self(result))
    }

    pub fn install(&self, lua: &Lua) -> mlua::Result<()> {
        if !lua.globals().get::<Value>("C_Spell")?.is_nil() {
            return Err(mlua::Error::runtime("C_Spell already installed"));
        }
        let api = lua.create_table()?;
        crate::swing::install_kinds(
            lua,
            self.0
                .iter()
                .filter_map(|(id, spell)| Some((*id, spell.swing?))),
        );
        let records = Rc::new(self.0.clone());
        legacy::install(lua, records.clone())?;
        let r = records.clone();
        api.set(
            "GetSpellInfo",
            lua.create_function(move |lua, input: Value| {
                let Some((id, spell)) =
                    numeric_id(lua, input)?.and_then(|id| r.get(&id).map(|spell| (id, spell)))
                else {
                    return Ok(None);
                };
                let data = lua.create_table()?;
                data.set("name", spell.name.as_str())?;
                data.set("rank", spell.rank.as_str())?;
                data.set("spellID", id)?;
                data.set("iconID", spell.icon.as_deref())?;
                data.set("castTime", spell.cast_time)?;
                data.set("minRange", spell.min_range)?;
                data.set("maxRange", spell.max_range)?;
                data.set("cost", spell.cost)?;
                data.set("powerType", spell.power_type)?;
                data.set("isFunnel", spell.funnel)?;
                Ok(Some(data))
            })?,
        )?;
        let r = records.clone();
        let book = lua.create_table()?;
        book.set(
            "GetPlayerSpellsByAura",
            lua.create_function(move |lua, aura: i64| {
                let result = lua.create_table()?;
                if aura <= 0 {
                    return Ok(result);
                }
                for id in benilla_ui::script::pfui_host::units::known_spells(lua) {
                    if r.get(&id).is_some_and(|spell| {
                        spell.effects.iter().any(|effect| {
                            effect.get("auraName") == Some(&(aura as f64))
                                && effect.get("effect").is_some_and(|effect| *effect != 0.0)
                        })
                    }) {
                        result.push(id)?;
                    }
                }
                Ok(result)
            })?,
        )?;
        lua.globals().set("C_SpellBook", book)?;
        for name in ["GetSpellName", "GetSpellSubtext", "GetSpellTexture"] {
            let records = records.clone();
            api.set(
                name,
                lua.create_function(move |lua, id: Value| {
                    let id = numeric_id(lua, id)?;
                    let spell = id.and_then(|id| records.get(&id));
                    Ok(spell
                        .and_then(|s| match name {
                            "GetSpellName" => Some(s.name.as_str()).filter(|s| !s.is_empty()),
                            "GetSpellSubtext" => Some(s.rank.as_str()),
                            _ => s.icon.as_deref(),
                        })
                        .map(str::to_owned))
                })?,
            )?;
        }
        let r = records.clone();
        api.set(
            "GetSpellDispelType",
            lua.create_function(move |lua, id: Value| {
                Ok(numeric_id(lua, id)?
                    .and_then(|id| r.get(&id))
                    .map_or(0, |s| s.dispel))
            })?,
        )?;
        let r = records.clone();
        api.set(
            "IsSpellPassive",
            lua.create_function(move |lua, id: Value| {
                Ok(numeric_id(lua, id)?
                    .and_then(|id| r.get(&id))
                    .is_some_and(|s| s.attributes & 0x40 != 0))
            })?,
        )?;
        let r = records.clone();
        api.set(
            "IsNextMeleeSpell",
            lua.create_function(move |lua, id: Value| {
                Ok(numeric_id(lua, id)?
                    .and_then(|id| r.get(&id))
                    .is_some_and(|s| s.attributes & 0x404 != 0))
            })?,
        )?;
        let r = records.clone();
        api.set(
            "GetSpellReagents",
            lua.create_function(move |lua, id: Value| {
                let Some(spell) = numeric_id(lua, id)?.and_then(|id| r.get(&id)) else {
                    return Ok(None);
                };
                let result = lua.create_table()?;
                for (index, (id, count)) in spell.reagents.iter().enumerate() {
                    let reagent = lua.create_table()?;
                    reagent.set("itemID", *id)?;
                    reagent.set("count", *count)?;
                    result.set(index + 1, reagent)?;
                }
                Ok(Some(result))
            })?,
        )?;
        states::install(lua, records.clone())?;
        let r = records;
        api.set(
            "GetSpellEffectInfo",
            lua.create_function(move |lua, id: Value| {
                let Some(spell) = numeric_id(lua, id)?.and_then(|id| r.get(&id)) else {
                    return Ok(None);
                };
                let result = lua.create_table()?;
                for (index, fields) in spell.effects.iter().enumerate() {
                    let effect = lua.create_table()?;
                    for (name, value) in fields {
                        effect.set(*name, *value)?;
                    }
                    result.set(index + 1, effect)?;
                }
                Ok(Some(result))
            })?,
        )?;
        lua.globals().set("C_Spell", api)?;
        benilla_ui::script::pfui_host::auras::install_spell_cancel(lua)?;
        benilla_ui::script::pfui_host::macro_verbs::install_spell_verbs(lua)?;
        benilla_ui::script::pfui_host::casts::install(lua)?;
        lua.load(SPELL_ENGINE)
            .set_name("@benilla-pfui/spell_engine.lua")
            .exec()?;
        runtime::install(lua)
    }
}

/// The ClassicAPI calls built on the catalog's own `C_Spell`.
const SPELL_ENGINE: &str = r#"
function C_Spell.CastingInfo() return C_Spell.UnitCastingInfo("player") end

-- A single school's name; a mask of more than one school has none here.
function C_Spell.GetSchoolString(mask)
  mask = tonumber(mask) or 0
  local masks = {1, 2, 4, 8, 16, 32, 64}
  for school = 0, 6 do
    if mask == masks[school + 1] then return getglobal("SPELL_SCHOOL" .. school .. "_CAP") or "Unknown" end
  end
  return "Unknown"
end

-- Modern's link alone; the global `GetSpellLink` adds the id.
function C_Spell.GetSpellLink(spell) return (GetSpellLink(spell)) end
"#;
