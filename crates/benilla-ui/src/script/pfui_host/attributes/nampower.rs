//! Resolve Nampower's spellbook overloads without changing the stock Lua bindings.

use super::{ClickAction, QueuePolicy, ScriptCall};
use crate::script::{
    parse_unit_token, resolve_spell_in_books, PetBookState, SpellBookState, UnitBase,
    UnitTokenParse,
};
use mlua::{Lua, Table, Value};

fn queue(lua: &Lua, id: u32, pet: bool, unit: Value, policy: &str) -> mlua::Result<()> {
    let policy = match policy {
        "force" => QueuePolicy::Force,
        "never" => QueuePolicy::Never,
        _ => QueuePolicy::Automatic,
    };
    let call = if let Value::String(unit) = unit {
        if let UnitTokenParse::Unit {
            base: UnitBase::Target,
            hops,
        } = parse_unit_token(&unit.to_str()?)
        {
            ScriptCall::PfuiClick(ClickAction::TargetSpell {
                spell_id: id,
                pet,
                hops,
            })
        } else {
            let Some(guid) = super::super::units::guid(lua, &unit.to_str()?)? else {
                return Ok(());
            };
            ScriptCall::PfuiClick(if pet {
                ClickAction::PetSpell {
                    spell_id: id,
                    target: guid,
                }
            } else {
                ClickAction::Spell {
                    spell_id: id,
                    target: Some(guid),
                }
            })
        }
    } else {
        let on_self = crate::script::binding_abi::bool_or_default(Some(&unit), false);
        if pet {
            ScriptCall::CastPetSpell {
                spell_id: id,
                on_self,
            }
        } else {
            // The protected-action gate's tag 2, as the stock `CastSpell` verb asks it.
            ScriptCall::CastSpell {
                spell_id: id,
                on_self,
                may_cast: crate::script::protected::gate(lua, crate::script::protected::Tag::CAST),
            }
        }
    };
    let call = if policy == QueuePolicy::Automatic {
        call
    } else {
        ScriptCall::PfuiClick(ClickAction::Cast {
            call: Box::new(call),
            policy,
        })
    };
    crate::script::calls::submit(lua, call)?;
    Ok(())
}

pub(super) fn install(lua: &Lua, actions: &Table) -> mlua::Result<()> {
    actions.set(
        "cancelQueue",
        lua.create_function(|lua, ground_only: bool| {
            crate::script::calls::submit(
                lua,
                ScriptCall::PfuiClick(ClickAction::CancelQueue { ground_only }),
            )?;
            Ok(())
        })?,
    )?;
    actions.set(
        "queueScript",
        lua.create_function(|lua, (body, priority): (String, Option<u8>)| {
            let priority = priority.unwrap_or(1);
            if !(1..=3).contains(&priority) {
                return Err(mlua::Error::runtime(
                    "QueueScript priority must be 1, 2 or 3",
                ));
            }
            if !body.is_empty() {
                crate::script::calls::submit(
                    lua,
                    ScriptCall::PfuiClick(ClickAction::QueueScript { body, priority }),
                )?;
            }
            Ok(())
        })?,
    )?;
    actions.set(
        "namedCast",
        lua.create_function(|lua, (name, unit, policy): (String, Value, String)| {
            let resolved =
                crate::script::pfui_host::spell_named(lua, &name).map(|s| (s.spell_id, s.pet));
            if let Some((id, pet)) = resolved {
                queue(lua, id, pet, unit, &policy)?;
            }
            Ok(())
        })?,
    )?;
    actions.set(
        "bookCast",
        lua.create_function(
            |lua, (input, book, unit, policy): (Value, Value, Value, String)| {
                let pet = match book {
                    Value::String(book) => book.to_str()?.eq_ignore_ascii_case("pet"),
                    Value::Integer(_) | Value::Number(_) => false,
                    _ => {
                        return Err(mlua::Error::runtime(
                            "Invalid spell slot in CastSpellNoQueue",
                        ))
                    }
                };
                let resolved = {
                    let book = if pet {
                        crate::script::pfui_host::pet_book(lua).slots
                    } else {
                        crate::script::pfui_host::player_book(lua).slots
                    };
                    let slots = &book;
                    if let Some(index) = lua.coerce_number(input.clone())? {
                        let index = (index - 1.0).trunc();
                        if !(0.0..1024.0).contains(&index) {
                            return Err(mlua::Error::runtime(
                                "Invalid spell slot in CastSpellNoQueue",
                            ));
                        }
                        slots
                            .get(index as usize)
                            .filter(|s| !s.passive)
                            .map(|s| s.spell_id)
                    } else if let Value::String(text) = input {
                        let text = text.to_str()?;
                        if text
                            .get(..8)
                            .is_some_and(|p| p.eq_ignore_ascii_case("spellId:"))
                        {
                            let id = text[8..]
                                .parse::<u32>()
                                .map_err(|_| mlua::Error::runtime("Invalid spellId: identifier"))?;
                            slots
                                .iter()
                                .find(|s| s.spell_id == id && !s.passive)
                                .map(|s| s.spell_id)
                        } else {
                            resolve_spell_in_books(
                                &SpellBookState {
                                    tabs: Vec::new(),
                                    slots: slots.clone(),
                                },
                                &PetBookState::default(),
                                &text,
                            )
                            .map(|s| s.spell_id)
                        }
                    } else {
                        return Err(mlua::Error::runtime(
                            "Invalid spell slot in CastSpellNoQueue",
                        ));
                    }
                };
                if let Some(id) = resolved {
                    queue(lua, id, pet, unit, &policy)?;
                }
                Ok(())
            },
        )?,
    )
}
