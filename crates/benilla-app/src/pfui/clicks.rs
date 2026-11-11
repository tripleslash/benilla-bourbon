//! Frame spell/macro actions stay ordered with ordinary selection and cast calls.

use crate::spell::{CastCommit, ScriptCast, TargetedBind, TargetingWants};
use benilla_ui::script::{
    pfui_host::attributes::{self, ClickAction},
    ScriptCall, SelectionRequest, UiScript,
};

/// A frame action's `target` chain, resolved off the selection as the call applies: `None` when it
/// names nobody, and the call is dropped.
pub(crate) fn normalize_targets(call: ScriptCall, cast: &ScriptCast) -> Option<ScriptCall> {
    Some(match call {
        ScriptCall::PfuiClick(ClickAction::TargetItem { guid, hops }) => {
            let target = cast.targeting.pfui_selected_target(hops)?;
            ScriptCall::PfuiClick(ClickAction::Item {
                guid,
                target: Some(target),
            })
        }
        ScriptCall::PfuiClick(ClickAction::TargetSpell {
            spell_id,
            pet,
            hops,
        }) => {
            let target = cast.targeting.pfui_selected_target(hops)?;
            ScriptCall::PfuiClick(if pet {
                ClickAction::PetSpell { spell_id, target }
            } else {
                ClickAction::Spell {
                    spell_id,
                    target: Some(target),
                }
            })
        }
        call => call,
    })
}

pub(crate) fn apply(script: &mut UiScript, action: ClickAction, cast: &mut ScriptCast) {
    match action {
        ClickAction::RestoreSelectionCast { call, selection } => {
            if selection.is_some_and(|guid| cast.targeting.pfui_context(guid).is_none()) {
                return;
            }
            let select = |guid: Option<u64>| {
                guid.map_or(ScriptCall::ClearTarget, |guid| ScriptCall::Select {
                    request: SelectionRequest::PfuiGuid(guid),
                    may_cast: true,
                })
            };
            crate::pfui::requeue(
                script,
                vec![
                    select(selection),
                    *call,
                    select(cast.targeting.selection.guid),
                ],
            );
        }
        ClickAction::Cast { .. }
        | ClickAction::Item { .. }
        | ClickAction::ItemAtPoint { .. }
        | ClickAction::OpenTradeLink { .. }
        | ClickAction::StartAttack { .. }
        | ClickAction::StopAttack
        | ClickAction::TargetItem { .. }
        | ClickAction::TargetSpell { .. }
        | ClickAction::CancelQueue { .. }
        | ClickAction::QueueScript { .. } => {
            // The queue's own actions, a target chain `normalize_targets` resolved first, and the
            // item, attack and profession-link steps the call host applies itself.
            unreachable!("frame actions are normalized before they apply")
        }
        ClickAction::RunScript { body } => {
            if let Err(error) =
                attributes::run_script(script.lua(), &body, cast.targeting.selection.guid)
            {
                benilla_ui::script::pfui_host::report_error(
                    script.lua(),
                    (format!("Nampower queued script: {error}")).to_string(),
                );
            }
        }
        ClickAction::PetSpell { spell_id, target } => {
            if cast.targeting.pfui_context(target).is_none() {
                return;
            }
            let restore = match cast.targeting.selection.guid {
                Some(guid) => ScriptCall::Select {
                    request: SelectionRequest::PfuiGuid(guid),
                    may_cast: true,
                },
                None => ScriptCall::ClearTarget,
            };
            crate::pfui::requeue(
                script,
                vec![
                    ScriptCall::Select {
                        request: SelectionRequest::PfuiGuid(target),
                        may_cast: true,
                    },
                    ScriptCall::CastPetSpell {
                        spell_id,
                        on_self: false,
                    },
                    restore,
                ],
            );
        }
        ClickAction::Spell {
            spell_id,
            target: None,
        } => crate::ui_spellbook::cast_spell(cast, spell_id, false),
        ClickAction::Spell {
            spell_id,
            target: Some(guid),
        } => {
            if let Some(point) = cast_at_unit(cast, spell_id, guid, true) {
                commit_or_follow(script, cast, spell_id, point);
            }
        }
        ClickAction::CastAt {
            spell_id,
            target,
            place_ground,
            no_toggle,
        } => {
            if no_toggle && toggled_on(cast, spell_id) {
                return;
            }
            match target {
                Some(guid) => {
                    if let Some(point) = cast_at_unit(cast, spell_id, guid, place_ground) {
                        commit_or_follow(script, cast, spell_id, point);
                    }
                }
                None => crate::ui_spellbook::cast_spell(cast, spell_id, false),
            }
        }
        ClickAction::GroundSpell { spell_id, point } => {
            // A second press of the spell whose cursor is up only commits: the stock ladder would
            // take the cursor down, and the point would find nothing to bind.
            if cast.ladder.ground.spell() != Some(spell_id) {
                crate::ui_spellbook::cast_spell(cast, spell_id, false);
            }
            commit_or_follow(script, cast, spell_id, point);
        }
        ClickAction::Macro {
            input,
            named,
            target,
        } => {
            let mut calls = Vec::new();
            if let Some(guid) = target {
                // A disappeared unit never runs the macro against an unrelated selection.
                if cast.targeting.pfui_context(guid).is_none() {
                    return;
                }
                calls.push(ScriptCall::Select {
                    request: SelectionRequest::PfuiGuid(guid),
                    may_cast: true,
                });
            }
            calls.push(ScriptCall::PfuiClick(ClickAction::RunMacro {
                input,
                named,
            }));
            if target.is_some() {
                calls.push(match cast.targeting.selection.guid {
                    Some(guid) => ScriptCall::Select {
                        request: SelectionRequest::PfuiGuid(guid),
                        may_cast: true,
                    },
                    None => ScriptCall::ClearTarget,
                });
            }
            crate::pfui::requeue(script, calls);
        }
        ClickAction::RunMacro { input, named } => {
            if let Err(error) =
                attributes::run_macro(script.lua(), &input, named, cast.targeting.selection.guid)
            {
                benilla_ui::script::pfui_host::report_error(
                    script.lua(),
                    (format!("pfUI click macro: {error}")).to_string(),
                );
            }
        }
    }
}

/// Whether casting `spell_id` would turn it off: its auto-repeat runs, its own cancelable aura is
/// up, or it is the player's form. `CastSpellNoToggle` leaves those alone.
fn toggled_on(cast: &ScriptCast, spell_id: u32) -> bool {
    let ScriptCast { targeting, ladder } = cast;
    if ladder.auto_repeat.0 == Some(spell_id) {
        return true;
    }
    let ctx = targeting.context();
    let (Some(spells), Some(store)) = (ladder.spells.as_ref(), ctx.rel.self_store) else {
        return false;
    };
    let Some(def) = spells.catalog.get(spell_id) else {
        return false;
    };
    let form = store.0.unit_shapeshift_form();
    crate::ui_action::toggle::active_action_toggle(spell_id, def, store)
        || crate::ui_action::toggle::form_recast_disposition(
            def,
            form,
            spells.forms.get(&u32::from(form)),
        )
        .is_some()
}

/// The spell at the unit `guid`; a ground-target spell lands at its feet when `place_ground` (the
/// point returned, for the caller to commit), else its reticle stays up for the player.
fn cast_at_unit(
    cast: &mut ScriptCast,
    spell_id: u32,
    guid: u64,
    place_ground: bool,
) -> Option<[f32; 3]> {
    let ScriptCast { targeting, ladder } = cast;
    let ctx = targeting.pfui_context(guid)?;
    // The same aura/form toggle predicates used by the stock spellbook's cast entry.
    if let (Some(spells), Some(store)) = (ladder.spells.as_ref(), ctx.rel.self_store) {
        if let Some(def) = spells.catalog.get(spell_id) {
            let form = store.0.unit_shapeshift_form();
            let toggle = crate::ui_action::toggle::active_action_toggle(spell_id, def, store);
            let form_toggle = crate::ui_action::toggle::form_recast_disposition(
                def,
                form,
                spells.forms.get(&u32::from(form)),
            );
            if toggle || form_toggle == Some(true) {
                let _ = ladder
                    .commands
                    .0
                    .send(crate::net::ClientCommand::CancelAura { spell_id });
                return None;
            }
            if form_toggle == Some(false) {
                return None;
            }
        }
    }
    let ground = ladder
        .spells
        .as_ref()
        .and_then(|spells| spells.catalog.get(spell_id))
        .is_some_and(|def| crate::spell::cast_target::cast_target_mask(def) & 0x60 != 0);
    let point = (place_ground && ground)
        .then_some(ctx.range.target_pos)
        .flatten()
        .map(benilla_assets::coords::bevy_to_wow);
    // No attack auto-pick: it could move selection and override the clicked GUID.
    ladder.send(spell_id, &ctx, CastCommit::Spell);
    point
}

/// The point a ground action owes its spell's cursor: a call parks its cast at the push
/// (`0x6e54f0`), and the cursor comes up only as the call host resumes the cast past the action.
struct OwedGround {
    spell_id: u32,
    point: [f32; 3],
}

/// Commit `point` now when the spell's cursor is up, else once the call's casts have run
/// ([`commit_owed_ground`]).
fn commit_or_follow(script: &UiScript, cast: &mut ScriptCast, spell_id: u32, point: [f32; 3]) {
    if cast.ladder.ground.spell() == Some(spell_id)
        && cast.ladder.ground.wants(TargetingWants::Location)
    {
        commit_ground(cast, spell_id, point);
    } else {
        script.lua().set_app_data(OwedGround { spell_id, point });
    }
}

/// The owed point committed once its spell's cursor is up, as the call that cast it returns; it
/// is kept while its cast is still on the way there, and dropped once the cast went elsewhere.
pub(crate) fn commit_owed_ground(script: &UiScript, cast: &mut ScriptCast) {
    let lua = script.lua();
    let Some(owed) = lua.remove_app_data::<OwedGround>() else {
        return;
    };
    if cast.ladder.ground.spell() == Some(owed.spell_id) {
        commit_ground(cast, owed.spell_id, owed.point);
    } else if cast.ladder.pending.current() == Some(owed.spell_id) {
        lua.set_app_data(owed);
    }
}

fn commit_ground(cast: &mut ScriptCast, spell_id: u32, point: [f32; 3]) {
    commit_ground_ladder(&mut cast.ladder, spell_id, point);
}

pub(crate) fn commit_ground_ladder(
    ladder: &mut crate::spell::CastLadder,
    spell_id: u32,
    point: [f32; 3],
) {
    if ladder.ground.spell() != Some(spell_id) || !ladder.ground.wants(TargetingWants::Location) {
        return;
    }
    // Another cast in flight refuses the commit, as a fresh press is refused (`6e4d97`): only the
    // cursor's own spell may hold the slot when its click commits.
    if ladder.pending.current().is_some_and(|id| id != spell_id) {
        return;
    }
    // BindLocation takes SOURCE before DEST, including patched rows carrying both bits.
    let source = ladder
        .spells
        .as_ref()
        .and_then(|spells| spells.catalog.get(spell_id))
        .is_some_and(|def| crate::spell::cast_target::cast_target_mask(def) & 0x20 != 0);
    let bound = if source {
        TargetedBind::Source(point)
    } else {
        TargetedBind::Dest(point)
    };
    ladder.commit_targeted(spell_id, CastCommit::Spell, bound);
}
