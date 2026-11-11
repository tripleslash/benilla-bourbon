//! The engine half of ClassicAPI's macro slash commands (`Util\SlashCommandsRegistry.lua`) and
//! macro conditions (`Util\MacroOptions.lua`): unit tokens told from names, a unit found by its
//! name, a spell or item at a unit or at the cursor, the cast that never toggles off, and the
//! attack's start and stop legs.

use super::attributes::ClickAction;
use crate::script::protected::{self, Tag};
use crate::script::{parse_unit_token, ScriptCall, UnitTokenParse};
use mlua::{Lua, Table, Value};

/// The world point under the cursor in wire coordinates, as the client's pick last left it: where
/// `C_Spell.CastAtCursor` and `C_Item.UseAtCursor` place a ground-target effect.
struct CursorPoint(Option<[f32; 3]>);

/// Publish the world point under the cursor (wire coordinates), `None` over no ground.
pub fn set_cursor_point(lua: &Lua, point: Option<[f32; 3]>) {
    lua.set_app_data(CursorPoint(point));
}

fn cursor_point(lua: &Lua) -> Option<[f32; 3]> {
    lua.app_data_ref::<CursorPoint>().and_then(|point| point.0)
}

/// The extension's own token families ([`super::units::custom_guid`]), each with a run of
/// `target` hops: a guid literal, `focus`, `nameplateN` and `markN` (1 to 8).
fn custom_token(token: &str) -> bool {
    let token = token.to_ascii_lowercase();
    if let Some(hex) = token.strip_prefix("0x") {
        return matches!(hex.len(), 8 | 16) && hex.bytes().all(|b| b.is_ascii_hexdigit());
    }
    let hops = if let Some(rest) = token.strip_prefix("focus") {
        rest
    } else if let Some(rest) = token.strip_prefix("nameplate") {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return false;
        }
        &rest[digits..]
    } else if let Some(rest) = token.strip_prefix("mark") {
        match rest.as_bytes().first() {
            Some(b'1'..=b'8') => &rest[1..],
            _ => return false,
        }
    } else {
        return false;
    };
    hops.len().is_multiple_of(6) && hops.as_bytes().chunks(6).all(|hop| hop == b"target")
}

/// `IsUnitToken(value)`: whether a resolver takes `value` as a token, never an error. ClassicAPI
/// asks the engine's resolver under a protected call; a name (`"Feral"`) is no token.
fn is_unit_token(value: &Value) -> bool {
    let Value::String(text) = value else {
        return false;
    };
    let text = text.to_string_lossy();
    !text.is_empty()
        && (custom_token(&text) || !matches!(parse_unit_token(&text), UnitTokenParse::Unknown))
}

fn truthy(value: Option<&Value>) -> bool {
    !matches!(value, None | Some(Value::Nil) | Some(Value::Boolean(false)))
}

/// A ground-target spell's aim argument: omitted or nil places it, only `false` holds the reticle.
fn places_ground(value: &Value) -> bool {
    !matches!(value, Value::Boolean(false))
}

fn submit(lua: &Lua, action: ClickAction) -> mlua::Result<()> {
    crate::script::calls::submit(lua, ScriptCall::PfuiClick(action))
}

/// The unit a cast or use names: its guid, `Ok(None)` for a token naming nobody; an unknown token
/// raises `Unknown unit name`, as the engine's resolver does.
fn unit_guid(lua: &Lua, unit: &str) -> mlua::Result<Option<u64>> {
    super::units::guid(lua, unit)
}

/// `C_Spell.CastAtUnit` and `CastSpellNoToggle`'s shared body: whether the cast went to the host.
fn cast_at(
    lua: &Lua,
    spell: Value,
    unit: Option<String>,
    place_ground: bool,
    no_toggle: bool,
) -> mlua::Result<bool> {
    // `/cast !Name` lands here with its prefix; the cast it asks for is the name behind it.
    let spell = match &spell {
        Value::String(text) if no_toggle && text.as_bytes().first() == Some(&b'!') => {
            Value::String(lua.create_string(&text.as_bytes()[1..])?)
        }
        _ => spell,
    };
    let Some(spell_id) = super::units::spell_identifier(lua, spell)? else {
        return Ok(false);
    };
    let target = match unit {
        Some(unit) => match unit_guid(lua, &unit)? {
            Some(guid) => Some(guid),
            None => return Ok(false),
        },
        None => None,
    };
    submit(
        lua,
        ClickAction::CastAt {
            spell_id,
            target,
            place_ground,
            no_toggle,
        },
    )?;
    Ok(true)
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let globals = lua.globals();
    globals.set(
        "IsUnitToken",
        lua.create_function(|_, value: Value| Ok(is_unit_token(&value)))?,
    )?;
    // `UnitTokenFromName(name [, exactMatch])`: the unit `TargetByName` would find, as a token,
    // else as its guid literal, which the resolvers take as well.
    globals.set(
        "UnitTokenFromName",
        lua.create_function(|lua, (name, exact): (Value, Option<Value>)| {
            let name = match &name {
                Value::String(_) | Value::Integer(_) | Value::Number(_) => {
                    lua.coerce_string(name)?.map(|name| name.to_string_lossy())
                }
                _ => None,
            }
            .ok_or_else(|| {
                mlua::Error::runtime("Usage: UnitTokenFromName(\"name\" [, exactMatch])")
            })?;
            if name.is_empty() {
                return Ok(None);
            }
            let Some(guid) = super::units::guid_by_name(lua, &name, truthy(exact.as_ref())) else {
                return Ok(None);
            };
            Ok(Some(
                super::units::token_from_guid(lua, guid)
                    .unwrap_or_else(|| format!("0x{guid:016X}")),
            ))
        })?,
    )?;
    globals.set(
        "CastSpellNoToggle",
        lua.create_function(
            |lua, (spell, unit, place): (Value, Option<String>, Option<Value>)| {
                cast_at(
                    lua,
                    spell,
                    unit,
                    places_ground(&place.unwrap_or(Value::Nil)),
                    true,
                )
            },
        )?,
    )?;
    // `StartAttack([unit])`: a name or a token naming nobody falls back to the selection.
    globals.set(
        "StartAttack",
        lua.create_function(|lua, unit: Option<Value>| {
            let target = match unit {
                Some(Value::String(token)) => {
                    unit_guid(lua, &token.to_string_lossy()).ok().flatten()
                }
                _ => None,
            };
            // The validator's pick of the nearest enemy asks the gate's tag 4, as `AttackTarget`.
            let may_pick = protected::gate(lua, Tag::TARGET);
            submit(lua, ClickAction::StartAttack { target, may_pick })
        })?,
    )?;
    globals.set(
        "StopAttack",
        lua.create_function(|lua, ()| submit(lua, ClickAction::StopAttack))?,
    )?;

    let items: Table = globals.get("C_Item")?;
    // `C_Item.UseAtUnit(item, unit [, placeGroundSpell])`: the item used at the unit. A
    // ground-target item keeps its reticle here, whatever the third argument asks.
    items.set(
        "UseAtUnit",
        lua.create_function(|lua, (item, unit, _place): (Value, Value, Option<Value>)| {
            let Value::String(unit) = unit else {
                return Ok(false);
            };
            let Some(target) = unit_guid(lua, &unit.to_string_lossy())? else {
                return Ok(false);
            };
            let Some(guid) = super::items::instance(lua, item)? else {
                return Ok(false);
            };
            submit(
                lua,
                ClickAction::Item {
                    guid,
                    target: Some(target),
                },
            )?;
            Ok(true)
        })?,
    )?;
    // `C_Item.UseAtCursor(item)`: the item used, a ground-target use placed at the cursor's ground.
    items.set(
        "UseAtCursor",
        lua.create_function(|lua, item: Value| {
            let Some(guid) = super::items::instance(lua, item)? else {
                return Ok(false);
            };
            let point = cursor_point(lua);
            submit(
                lua,
                match point {
                    Some(point) => ClickAction::ItemAtPoint { guid, point },
                    None => ClickAction::Item { guid, target: None },
                },
            )?;
            Ok(point.is_some())
        })?,
    )?;
    Ok(())
}

/// `C_Spell.CastAtUnit` and `C_Spell.CastAtCursor`, once the spell catalog has made `C_Spell`.
pub fn install_spell_verbs(lua: &Lua) -> mlua::Result<()> {
    let spells: Table = lua.globals().get("C_Spell")?;
    spells.set(
        "CastAtUnit",
        lua.create_function(|lua, (spell, unit, place): (Value, Value, Option<Value>)| {
            let Value::String(unit) = unit else {
                return Ok(false);
            };
            cast_at(
                lua,
                spell,
                Some(unit.to_string_lossy().to_string()),
                places_ground(&place.unwrap_or(Value::Nil)),
                false,
            )
        })?,
    )?;
    // `C_Spell.CastAtCursor(spell)`: a ground-target spell placed at the cursor's ground; any
    // other spell is cast as it would be. Whether the cursor had ground to place it on.
    spells.set(
        "CastAtCursor",
        lua.create_function(|lua, spell: Value| {
            let Some(spell_id) = super::units::spell_identifier(lua, spell)? else {
                return Ok(false);
            };
            let point = cursor_point(lua);
            submit(
                lua,
                match point {
                    Some(point) => ClickAction::GroundSpell { spell_id, point },
                    None => ClickAction::CastAt {
                        spell_id,
                        target: None,
                        place_ground: false,
                        no_toggle: false,
                    },
                },
            )?;
            Ok(point.is_some())
        })?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extension_tokens_are_told_from_names() {
        for token in [
            "focus",
            "focustargettarget",
            "nameplate12target",
            "mark8",
            "0x00000000000004D2",
        ] {
            assert!(custom_token(token), "{token}");
        }
        for name in [
            "Markus",
            "mark9",
            "Focusfire",
            "nameplate",
            "0x123",
            "Feral",
        ] {
            assert!(!custom_token(name), "{name}");
        }
    }
}
