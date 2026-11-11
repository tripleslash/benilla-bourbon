//! Form and mount cancellation selects real buff spells by the player's own DBC effects.

use super::Spell;
use benilla_ui::script::pfui_host::{auras, units};
use mlua::Lua;
use std::{collections::BTreeMap, rc::Rc};

fn has_effect(spell: &Spell, aura: u32, form: Option<u8>) -> bool {
    spell.effects.iter().any(|effect| {
        effect.get("effect").is_some_and(|effect| *effect != 0.0)
            && effect.get("auraName") == Some(&f64::from(aura))
            && form.is_none_or(|form| effect.get("miscValue") == Some(&f64::from(form)))
    })
}

pub(super) fn install(lua: &Lua, spells: Rc<BTreeMap<u32, Spell>>) -> mlua::Result<()> {
    let records = spells.clone();
    lua.globals().set(
        "IsStealthed",
        lua.create_function(move |lua, ()| {
            // Deviation: scan actual stealth auras instead of the reference DLL's visibility bit,
            // because that bit also describes mounts and some druid forms.
            Ok(units::player_raw(lua).is_some_and(|unit| {
                unit.mount_display == 0
                    && unit.auras.iter().take(32).any(|spell| {
                        records
                            .get(spell)
                            .is_some_and(|spell| has_effect(spell, 16, None))
                    })
            }))
        })?,
    )?;
    for (name, effect) in [("Dismount", 78), ("CancelShapeshiftForm", 36)] {
        let records = spells.clone();
        lua.globals().set(
            name,
            lua.create_function(move |lua, ()| {
                let Some(unit) = units::player_raw(lua) else {
                    return Ok(());
                };
                if effect == 36 && (unit.form == 0 || units::player_is_warrior(lua)) {
                    return Ok(());
                }
                let form = (effect == 36).then_some(unit.form);
                if let Some(spell) = unit.auras.iter().take(32).find(|id| {
                    records
                        .get(id)
                        .is_some_and(|spell| has_effect(spell, effect, form))
                }) {
                    auras::cancel_player_spell(lua, *spell)?;
                }
                Ok(())
            })?,
        )?;
    }
    Ok(())
}
