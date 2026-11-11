//! Equipment sets use a local cursor tag and character bindings, never a server action type.

use crate::script::{
    cursor::{self, CursorAction, CursorPayload},
    Model, ScriptValue,
};
use mlua::{Lua, Table};

// Reserved only inside the shared cursor. The stock PlaceAction packer explicitly refuses it.
const SET_CURSOR: u8 = 0xfe;

pub(crate) fn held_set(model: &Model) -> Option<u32> {
    match &model.cursor {
        Some(CursorPayload::Action(action)) if action.kind == SET_CURSOR => Some(action.action),
        _ => None,
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CursorState {
    kind: i64,
    id: u32,
}

fn cursor_state(model: &Model) -> CursorState {
    let (kind, id) = match &model.cursor {
        None => (0, 0),
        Some(CursorPayload::Item(item)) => (1, item.item_id),
        Some(CursorPayload::Ammo(ammo)) => (1, ammo.item_id),
        Some(CursorPayload::Money(money)) => (2, money.copper),
        Some(CursorPayload::Spell(spell)) => (3, spell.spell_id),
        Some(CursorPayload::PetAction(pet)) => (4, pet.packed & 0x00ff_ffff),
        Some(CursorPayload::Merchant(item)) => (5, item.row + 1),
        Some(CursorPayload::Macro(macro_)) => (7, macro_.id),
        Some(CursorPayload::StablePet(pet)) => (9, u32::from(pet.slot)),
        Some(CursorPayload::Action(action)) => (
            match action.kind & 0xf0 {
                0x00 => 3,
                0x40 => 7,
                0x80 => 1,
                _ if action.kind == SET_CURSOR => 12,
                _ => 6,
            },
            action.action,
        ),
    };
    CursorState { kind, id }
}

thread_local! {
    /// Set while a setter's inner clear runs ([`hold`]).
    static HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Releases [`hold`] when dropped.
pub(crate) struct Hold;

impl Drop for Hold {
    fn drop(&mut self) {
        HELD.with(|held| held.set(false));
    }
}

/// Keep [`cursor_changed`] quiet through a setter's inner clear, so the setter reports its one
/// transition, old payload to new, once the new payload is on.
pub(crate) fn hold() -> Hold {
    HELD.with(|held| held.set(true));
    Hold
}

/// ClassicAPI's CURSOR_CHANGED describes each actual transition, before the stock update event.
pub(crate) fn cursor_changed(model: &mut Model) {
    if HELD.with(std::cell::Cell::get) {
        return;
    }
    let Some(previous) = model.pfui_cursor else {
        return;
    };
    let next = cursor_state(model);
    if next == previous {
        return;
    }
    model.pfui_cursor = Some(next);
    model.raise(
        "CURSOR_CHANGED",
        vec![
            if next.kind == 0 {
                ScriptValue::Int(1)
            } else {
                ScriptValue::Nil
            },
            ScriptValue::Int(next.kind),
            ScriptValue::Int(previous.kind),
            ScriptValue::Int(i64::from(previous.id)),
        ],
    );
}

pub(super) fn install(lua: &Lua, bridge: &Table) -> mlua::Result<()> {
    {
        let mut model = lua.app_data_mut::<crate::script::ModelCell>().expect("model");
        model.pfui_cursor = Some(cursor_state(&model));
    }
    bridge.set(
        "heldSet",
        lua.create_function(|lua, ()| Ok(held_set(&lua.app_data_ref::<crate::script::ModelCell>().expect("model"))))?,
    )?;
    bridge.set(
        "hasCursor",
        lua.create_function(|lua, ()| {
            Ok(lua.app_data_ref::<crate::script::ModelCell>().expect("model").cursor.is_some())
        })?,
    )?;
    bridge.set(
        "hasServerAction",
        lua.create_function(|lua, slot: u32| {
            Ok(lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model")
                .actions
                .get(&slot)
                .is_some_and(|action| action.action != 0))
        })?,
    )?;
    bridge.set(
        "setCursor",
        lua.create_function(|lua, (id, texture): (Option<u32>, Option<String>)| {
            // Through the engine's own setter, so the clear and the set each fire their events
            // inside this call, as any cursor verb's do.
            match id {
                Some(id) => {
                    cursor::set_cursor(lua, true, cursor::Setter::Free, move |_| {
                        Some(CursorPayload::Action(CursorAction {
                            src_slot: 0,
                            kind: SET_CURSOR,
                            action: id,
                            texture,
                            count: 0,
                            consumable: false,
                            state: None,
                        }))
                    });
                }
                None => crate::script::tick::verb(lua, cursor::clear_cursor),
            }
            Ok(())
        })?,
    )?;
    bridge.set(
        "displaceServerAction",
        lua.create_function(|lua, slot: u32| {
            if !(1..=120).contains(&slot) {
                return Ok(false);
            }
            let displaced = {
                let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
                model
                    .actions
                    .get(&slot)
                    .cloned()
                    .map(|action| (action, model.action_states.get(&slot).cloned()))
            };
            match displaced {
                Some((action, state)) => {
                    // As `PickupAction` lifts a slot: the action onto the cursor, then the slot
                    // cleared and sent as 0, signalled in the call.
                    cursor::set_cursor(lua, true, cursor::Setter::Free, move |_| {
                        Some(CursorPayload::Action(CursorAction {
                            src_slot: slot,
                            kind: action.kind,
                            action: action.action,
                            texture: action.texture,
                            count: action.count,
                            consumable: action.consumable,
                            state,
                        }))
                    });
                    crate::script::tick::verb(lua, |model| {
                        model.actions.remove(&slot);
                        model.action_states.remove(&slot);
                        model.call_in_verb(crate::script::ScriptCall::ActionBar(
                            crate::script::ActionBarCall::Set {
                                id: slot,
                                packed: 0,
                            },
                        ));
                        model.raise(
                            "ACTIONBAR_SLOT_CHANGED",
                            vec![ScriptValue::Int(i64::from(slot))],
                        );
                    });
                }
                None => crate::script::tick::verb(lua, cursor::clear_cursor),
            }
            Ok(true)
        })?,
    )?;
    Ok(())
}
