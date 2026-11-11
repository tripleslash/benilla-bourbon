//! Attribute-script delivery through the same legacy/modern argument bridge as stock scripts.

use crate::script::{Model, ScriptCall, ScriptValue};
use mlua::{Function, Lua, MultiValue, Table};

#[cfg(feature = "nampower")]
mod nampower;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueuePolicy {
    #[default]
    Automatic,
    Force,
    Never,
    /// Engine-owned release: bypass queuing while retaining the original retry budget.
    Release {
        retry: Option<(u8, u64)>,
        generation: (u64, u64),
    },
}

/// Extension-owned actions share the engine's queue with target, cast and macro calls.
#[derive(Clone, Debug, PartialEq)]
pub enum ClickAction {
    /// The queue policy belongs to this request, including calls made by the same macro.
    Cast {
        call: Box<ScriptCall>,
        policy: QueuePolicy,
    },
    /// Capture the selection to restore when the released call actually executes.
    RestoreSelectionCast {
        call: Box<ScriptCall>,
        selection: Option<u64>,
    },
    /// Resolve selection-rooted tokens after preceding macro selection calls have committed.
    TargetSpell {
        spell_id: u32,
        pet: bool,
        hops: usize,
    },
    QueueScript {
        body: String,
        priority: u8,
    },
    RunScript {
        body: String,
    },
    /// Cancel extension queues even when the stock API has no active cast/cursor to stop.
    CancelQueue {
        ground_only: bool,
    },
    PetSpell {
        spell_id: u32,
        target: u64,
    },
    Spell {
        spell_id: u32,
        target: Option<u64>,
    },
    /// A quickcast's terrain hit, in wire coordinates, retained while queued.
    GroundSpell {
        spell_id: u32,
        point: [f32; 3],
    },
    Item {
        guid: u64,
        target: Option<u64>,
    },
    TargetItem {
        guid: u64,
        hops: usize,
    },
    Macro {
        input: String,
        named: bool,
        target: Option<u64>,
    },
    /// The macro body runs after its preceding selection has committed.
    RunMacro {
        input: String,
        named: bool,
    },
    /// `C_Spell.CastAtUnit` and `CastSpellNoToggle`: the spell at `target`, else as the stock cast
    /// aims. A ground-target spell lands at the unit's feet when `place_ground`, else its reticle
    /// stays up; `no_toggle` leaves an active aura, form or auto-repeat of the spell running.
    CastAt {
        spell_id: u32,
        target: Option<u64>,
        place_ground: bool,
        no_toggle: bool,
    },
    /// `StartAttack`: the attack's start leg alone, at `target`, else the selection; an attack
    /// already running is left alone.
    StartAttack {
        target: Option<u64>,
        may_pick: bool,
    },
    /// `StopAttack`: the attack's stop leg alone.
    StopAttack,
    /// `C_Item.UseAtCursor`: the item used, a ground-target use placed at `point` (wire
    /// coordinates).
    ItemAtPoint {
        guid: u64,
        point: [f32; 3],
    },
    /// A profession link clicked in chat: `opener`'s window shows the linker `guid`'s `spells`
    /// at its `rank` and `max_rank` in place of the player's book.
    OpenTradeLink {
        opener: u32,
        guid: u64,
        spells: Vec<u32>,
        rank: u32,
        max_rank: u32,
    },
}

fn target(lua: &Lua, unit: Option<String>) -> mlua::Result<Option<Option<u64>>> {
    match unit {
        Some(unit) => Ok(super::units::guid(lua, &unit)?.map(Some)),
        None => Ok(Some(None)),
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let actions = lua.create_table()?;
    #[cfg(feature = "nampower")]
    nampower::install(lua, &actions)?;
    actions.set(
        "spell",
        lua.create_function(|lua, (spell, unit): (mlua::Value, Option<String>)| {
            let Some(spell_id) = super::units::spell_identifier(lua, spell)? else {
                return Ok(false);
            };
            let Some(target) = target(lua, unit)? else {
                return Ok(false);
            };
            crate::script::calls::submit(
                lua,
                ScriptCall::PfuiClick(ClickAction::Spell { spell_id, target }),
            )?;
            Ok(true)
        })?,
    )?;
    actions.set(
        "item",
        lua.create_function(|lua, (item, unit): (mlua::Value, Option<String>)| {
            super::items::request_use(lua, item, unit)
        })?,
    )?;
    actions.set(
        "macro",
        lua.create_function(
            |lua, (input, named, unit): (Option<String>, bool, Option<String>)| {
                let Some(input) = input.filter(|input| !input.is_empty()) else {
                    return Ok(false);
                };
                let Some(target) = target(lua, unit)? else {
                    return Ok(false);
                };
                crate::script::calls::submit(
                    lua,
                    ScriptCall::PfuiClick(ClickAction::Macro {
                        input,
                        named,
                        target,
                    }),
                )?;
                Ok(true)
            },
        )?,
    )?;
    lua.set_named_registry_value("benilla.pfui.attribute_actions", actions)?;
    let invoke = lua.create_function(
        |lua, (frame, callback, name, value): (Table, Function, mlua::String, mlua::Value)| {
            super::super::event::invoke_with_globals(
                lua,
                frame,
                &callback,
                vec![mlua::Value::String(name), value],
            )
        },
    )?;
    lua.set_named_registry_value("benilla.pfui.attribute_invoke", invoke)
}

/// A macro sees the committed clicked target even though the ordinary frame feed ran earlier.
/// Addon macro runners retain their own syntax; the fallback uses the stock chat-line event.
pub fn run_macro(lua: &Lua, input: &str, named: bool, target: Option<u64>) -> mlua::Result<()> {
    super::units::with_macro_target(lua, target, || {
        if let Some(runner) = lua.globals().get::<Option<Function>>("RunMacro")? {
            return runner.call(input);
        }
        let body = if named {
            let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
            let index = input
                .parse::<usize>()
                .ok()
                .unwrap_or_else(|| model.macros.index_by_name(input));
            let Some(macro_) = model.macros.get(index) else {
                return Ok(());
            };
            macro_.body.clone()
        } else {
            input.to_string()
        };
        for line in body
            .split(['\r', '\n'])
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            super::fire_pfui_event(
                lua,
                "EXECUTE_CHAT_LINE",
                vec![ScriptValue::Str(line.to_string())],
            );
        }
        Ok(())
    })
}

/// Queued scripts read the selection committed before their turn in the ordered call stream.
pub fn run_script(lua: &Lua, body: &str, target: Option<u64>) -> mlua::Result<()> {
    super::units::with_macro_target(lua, target, || {
        lua.load(body).set_name("@nampower/queued-script").exec()
    })
}

/// The app resolves this token with its live unit resolver before publishing a world mouseover.
pub fn hovered_unit(lua: &Lua) -> mlua::Result<Option<String>> {
    let reader: Option<Function> = lua.named_registry_value("benilla.pfui.attribute_hover")?;
    match reader {
        Some(reader) => reader.call(MultiValue::new()),
        None => Ok(None),
    }
}
