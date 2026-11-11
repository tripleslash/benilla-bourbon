//! One replaceable queued Lua script, ordered around the spell lanes.
//! Nampower documents priority 1 before spells, 2 between the lanes and 3 after spells.

use super::{api, ClickAction, ScriptCall, UiScript};
use crate::spell::ScriptCast;

struct Queued {
    body: String,
    priority: u8,
}

#[derive(Default)]
struct State(Option<Queued>);

fn initialize(script: &UiScript) {
    if script.lua().app_data_ref::<State>().is_none() {
        script.lua().set_app_data(State::default());
    }
}

pub(super) fn clear(script: &UiScript) {
    if let Some(mut state) = script.lua().app_data_mut::<State>() {
        state.0 = None;
    }
}

fn remaining_ms(script: &UiScript, cast: &ScriptCast) -> f64 {
    let remaining_cast = benilla_ui::script::pfui_host::casts::player_cast(script.lua())
        .map_or(0.0, |cast| cast.end_ms - api::now(script.lua()) * 1000.0);
    remaining_cast
        .max(cast.ladder.cooldowns.pfui_gcd_remaining_ms(133))
        .max(0.0)
}

fn run(body: String) -> ScriptCall {
    ScriptCall::PfuiClick(ClickAction::RunScript { body })
}

pub(super) fn intercept(
    script: &UiScript,
    cast: &ScriptCast,
    body: String,
    priority: u8,
) -> Option<ScriptCall> {
    let remaining = remaining_ms(script, cast);
    // Outside the ordinary window QueueScript behaves like RunScript immediately.
    if (remaining == 0.0 && cast.ladder.pending.current().is_none())
        || remaining > api::cvar(script.lua(), "NP_SpellQueueWindowMs")
    {
        return Some(run(body));
    }
    initialize(script);
    script
        .lua()
        .app_data_mut::<State>()
        .expect("script queue")
        .0 = Some(Queued { body, priority });
    None
}

/// The queued script's run, once its window and the spell lanes it waits on are clear.
pub(super) fn resume(script: &mut UiScript, cast: &ScriptCast) -> Option<ScriptCall> {
    let priority = script
        .lua()
        .app_data_ref::<State>()
        .and_then(|state| state.0.as_ref().map(|queued| queued.priority))?;
    if remaining_ms(script, cast) > 0.0 || cast.ladder.pending.current().is_some() {
        return None;
    }
    let waiting_for_spells = script
        .lua()
        .app_data_ref::<super::State>()
        .is_some_and(|state| {
            (priority >= 2 && !state.non_gcd.is_empty())
                || (priority == 3 && (state.queued.is_some() || state.on_swing.is_some()))
        });
    if waiting_for_spells {
        return None;
    }
    // Remove before invoking Lua: a reentrant QueueScript owns the next request.
    let queued = script
        .lua()
        .app_data_mut::<State>()
        .expect("script queue")
        .0
        .take();
    queued.map(|queued| run(queued.body))
}
