//! Bounded retries of our own sends, driven only by transient server verdicts.

use super::*;
use crate::spell::cast_target::{resolve_cast_target, CastCandidates, CastWireTarget};

const MAX_RETRIES: u8 = 3;
const LIFETIME_MS: u64 = 2000;
const MAX_SENT: usize = 16;

pub(super) struct Sent {
    pub(super) queued: Queued,
}

fn now_ms(script: &UiScript) -> u64 {
    (api::now(script.lua()) * 1000.0) as u64
}

/// Snapshot the bound wire target after the normal attack-pick continuation has finished.
fn bound_call(a: &Attempt, cast: &crate::spell::ScriptCast) -> Option<ScriptCall> {
    if matches!(
        a.call,
        ScriptCall::PfuiClick(ClickAction::GroundSpell { .. })
    ) {
        return Some(a.call.clone());
    }
    let ctx = match &a.call {
        ScriptCall::PfuiClick(ClickAction::Spell {
            target: Some(guid), ..
        }) => cast.targeting.pfui_context(*guid)?,
        _ => cast.targeting.context().self_bound(a.on_self),
    };
    let def = cast.ladder.spells.as_ref()?.catalog.get(a.spell)?;
    if crate::spell::cast_target::cast_target_mask(def) & 0x60 != 0 {
        let point = ctx.range.target_pos.filter(|p| p.is_finite())?;
        return Some(ScriptCall::PfuiClick(ClickAction::GroundSpell {
            spell_id: a.spell,
            point: benilla_assets::coords::bevy_to_wow(point),
        }));
    }
    // As the attempt's own candidates (`crate::spell::cast_send`), the caster the active player.
    let candidates = CastCandidates {
        selection: ctx.selection_guid,
        selected: ctx.selected,
        selection_held: ctx.selection_held,
        caster: ctx.self_guid,
        auto_self: ctx.self_guid,
        auto_self_store: ctx.rel.self_store,
        main_hand_item: ctx.main_hand_item,
        long_range: {
            let row = cast
                .ladder
                .spells
                .as_ref()
                .and_then(|s| s.ranges.get(def.range_index));
            let targets = benilla_formats::RangeTargets {
                target: None,
                attack_target: None,
            };
            cast.ladder
                .spell_mods
                .min_max_range(def, row, ctx.range.caster, targets)
                .is_some_and(|(_, max)| max > crate::spell::cast_target::LONG_RANGE)
        },
    };
    match resolve_cast_target(Some(def), &candidates, ctx.auto_self_cast, &ctx.rel) {
        CastWireTarget::SelfImplicit => Some(ScriptCall::CastSpell {
            spell_id: a.spell,
            on_self: true,
            may_cast: true,
        }),
        CastWireTarget::Unit(guid) => Some(ScriptCall::PfuiClick(ClickAction::Spell {
            spell_id: a.spell,
            target: Some(guid),
        })),
        _ => None, // Item/GO/manual-cursor sends do not belong to this spell-only subset.
    }
}

pub(super) fn remember(script: &UiScript, a: &Attempt, cast: &crate::spell::ScriptCast) {
    let now = now_ms(script);
    if let Some(mut state) = script.lua().app_data_mut::<State>() {
        state.sent.remove(&a.spell);
        state
            .sent
            .retain(|_, sent| sent.queued.expires > now as f64 / 1000.0);
    }
    if api::cvar(script.lua(), "NP_RetryServerRejectedSpells") == 0.0 {
        return;
    }
    let budget = match a.policy {
        QueuePolicy::Never => return,
        QueuePolicy::Release {
            retry: Some(budget),
            ..
        } => budget,
        _ => (MAX_RETRIES, now.saturating_add(LIFETIME_MS)),
    };
    if budget.0 == 0 || now >= budget.1 {
        return;
    }
    let Some(def) = cast
        .ladder
        .spells
        .as_ref()
        .and_then(|s| s.catalog.get(a.spell))
    else {
        return;
    };
    if !retryable(def) {
        return;
    }
    let Some(call) = bound_call(a, cast) else {
        return;
    };
    let mut state = script.lua().app_data_mut::<State>().expect("queue");
    if state.sent.len() >= MAX_SENT {
        return; // Never grow an unbounded history of unanswered packets.
    }
    state.sent.insert(
        a.spell,
        Sent {
            queued: Queued {
                call,
                spell: a.spell,
                selection: None,
                ready_after: None,
                expires: budget.1 as f64 / 1000.0,
                lane: lane(a.spell, def),
                category: def.start_recovery_category,
                ground: crate::spell::cast_target::cast_target_mask(def) & 0x60 != 0,
                retry: Some(budget),
            },
        },
    );
}

/// Acceptance can race a scheduled retry. Remove only retries, preserving newer user requests.
pub(crate) fn accept(script: &UiScript, spell: u32) {
    let old = {
        let Some(mut state) = script.lua().app_data_mut::<State>() else {
            return;
        };
        state.sent.remove(&spell);
        let retry = |q: &Queued| q.spell == spell && q.retry.is_some();
        let mut old = Vec::new();
        if state.queued.as_ref().is_some_and(retry) {
            old.extend(state.queued.take().map(|q| (q.lane, q.spell)));
        }
        if state.on_swing.as_ref().is_some_and(retry) {
            old.extend(state.on_swing.take().map(|q| (q.lane, q.spell)));
        }
        state.non_gcd.retain(|q| {
            if retry(q) {
                old.push((q.lane, q.spell));
                false
            } else {
                true
            }
        });
        old
    };
    for (lane, spell) in old {
        event(script, lane.popped(), spell);
    }
}

pub(crate) fn reject(script: &UiScript, spell: u32, reason: u8) {
    let Some(mut q) = script
        .lua()
        .app_data_mut::<State>()
        .and_then(|mut state| state.sent.remove(&spell))
        .map(|sent| sent.queued)
    else {
        return;
    };
    if !matches!(reason, 0x28 | 0x3c | 0x61)
        || api::cvar(script.lua(), "NP_RetryServerRejectedSpells") == 0.0
        || api::now(script.lua()) >= q.expires
    {
        return;
    }
    let Some((remaining, deadline)) = q.retry else {
        return;
    };
    if remaining == 0 {
        return;
    }
    q.retry = Some((remaining - 1, deadline));
    let lane = q.lane;
    {
        let mut state = script.lua().app_data_mut::<State>().expect("queue");
        // A later player request wins. Rejections must not replace it or reorder its lane.
        match lane {
            Lane::Normal if state.queued.is_none() => state.queued = Some(q),
            Lane::OnSwing if state.on_swing.is_none() => state.on_swing = Some(q),
            Lane::NonGcd
                if state.non_gcd.len() < 6
                    && !state.non_gcd.iter().any(|old| old.spell == spell) =>
            {
                state.non_gcd.push_back(q)
            }
            _ => return,
        }
    }
    event(script, lane.pushed(), spell);
}
