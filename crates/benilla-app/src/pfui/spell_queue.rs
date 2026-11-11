//! Opt-in spell queues that resume through the ordinary cast validator.

use benilla_pfui::nampower as api;
use benilla_ui::script::{
    pfui_host::attributes::{ClickAction, QueuePolicy},
    ScriptCall, ScriptValue as V, UiScript,
};
use bevy::platform::time::Instant;
use std::collections::VecDeque;

#[path = "script_queue.rs"]
mod scripts;

#[path = "spell_retry.rs"]
mod retries;
pub(crate) use retries::{accept, reject};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    Normal,
    NonGcd,
    OnSwing,
}

fn lane(spell: u32, def: &benilla_formats::SpellDisplay) -> Lane {
    if def.on_next_swing() {
        return Lane::OnSwing;
    }
    // Nampower's GCD lane is category 133. Turtle removes Power Overwhelming's GCD.
    if def.start_recovery_category == 133 && !(cfg!(feature = "pfui-turtle") && spell == 51714) {
        Lane::Normal
    } else {
        Lane::NonGcd
    }
}

impl Lane {
    fn pushed(self) -> i64 {
        match self {
            Self::OnSwing => 0,
            Self::Normal => 2,
            Self::NonGcd => 4,
        }
    }
    fn popped(self) -> i64 {
        self.pushed() + 1
    }
}

struct Queued {
    call: ScriptCall,
    spell: u32,
    selection: Option<u64>,
    ready_after: Option<f64>,
    expires: f64,
    lane: Lane,
    category: u32,
    ground: bool,
    retry: Option<(u8, u64)>,
}

#[derive(Default)]
struct State {
    queued: Option<Queued>,
    non_gcd: VecDeque<Queued>,
    non_gcd_after: f64,
    on_swing: Option<Queued>,
    on_swing_after: f64,
    sent: std::collections::HashMap<u32, retries::Sent>,
    policy: QueuePolicy,
    generation: u64,
    ground_generation: u64,
}

fn initialize(script: &UiScript) {
    if script.lua().app_data_ref::<State>().is_none() {
        script.lua().set_app_data(State::default());
    }
}

fn event(script: &UiScript, code: i64, spell: u32) {
    api::fire(
        script.lua(),
        None,
        "SPELL_QUEUE_EVENT",
        vec![V::Int(code), V::Int(i64::from(spell))],
    );
}

fn clear_ground(script: &UiScript) {
    let old: Vec<_> = script
        .lua()
        .app_data_mut::<State>()
        .map(|mut state| {
            state.ground_generation = state.ground_generation.wrapping_add(1);
            state.sent.retain(|_, sent| !sent.queued.ground);
            let is_ground = |q: &Queued| q.ground;
            let mut old = Vec::new();
            if state.queued.as_ref().is_some_and(is_ground) {
                old.extend(state.queued.take());
            }
            let mut keep = VecDeque::new();
            for q in state.non_gcd.drain(..) {
                if is_ground(&q) {
                    old.push(q);
                } else {
                    keep.push_back(q);
                }
            }
            state.non_gcd = keep;
            old
        })
        .unwrap_or_default();
    for q in old {
        event(script, q.lane.popped(), q.spell);
    }
}

pub(super) fn register(app: &mut bevy::prelude::App) {
    use bevy::prelude::*;
    app.add_systems(
        Update,
        (track_pick, release)
            .chain()
            .in_set(crate::ui_script::UiFeed),
    );
}

/// The world point under the cursor as the frame's pick left it, where a quickcast ground spell
/// lands; kept on the VM, so a call applying inside the host reads it.
struct PickPoint(Option<bevy::prelude::Vec3>);

pub(super) fn track_pick(
    pick: Option<bevy::prelude::Res<crate::target::PickOcclusion>>,
    script: Option<bevy::prelude::NonSend<UiScript>>,
) {
    if let Some(script) = script {
        script
            .lua()
            .set_app_data(PickPoint(pick.and_then(|pick| pick.point)));
    }
}

/// The queue's release, each frame: the casts whose lane is ready go through the ordinary
/// validator as calls, applied through the host as the turn ends.
pub(super) fn release(
    mut vm: crate::script_calls::VmWith<crate::spell::ScriptCast<'static, 'static>>,
) {
    vm.turn(|script, cast| {
        let Some(script) = script else { return };
        let calls = resume(script, &cast);
        if !calls.is_empty() {
            crate::pfui::requeue(script, calls);
        }
    });
}

pub(crate) fn clear(script: &UiScript) {
    let old: Vec<_> = script
        .lua()
        .app_data_mut::<State>()
        .map(|mut state| {
            let mut old: Vec<_> = state.non_gcd.drain(..).collect();
            old.extend(state.queued.take());
            old.extend(state.on_swing.take());
            state.non_gcd_after = 0.0;
            state.on_swing_after = 0.0;
            state.sent.clear();
            state.policy = QueuePolicy::Automatic;
            state.generation = state.generation.wrapping_add(1);
            old
        })
        .unwrap_or_default();
    for old in old {
        event(script, old.lane.popped(), old.spell);
    }
}

pub(crate) fn reset(script: &UiScript) {
    clear(script);
    scripts::clear(script);
}

/// Resolve an action at press time. A later bar edit never changes the queued spell.
fn spell(script: &UiScript, call: &ScriptCall) -> Option<(u32, bool, Option<u64>)> {
    match call {
        ScriptCall::CastSpell {
            spell_id, on_self, ..
        } => Some((*spell_id, *on_self, None)),
        ScriptCall::PfuiClick(ClickAction::Spell { spell_id, target }) => {
            Some((*spell_id, false, *target))
        }
        ScriptCall::PfuiClick(ClickAction::GroundSpell { spell_id, .. }) => {
            Some((*spell_id, false, None))
        }
        ScriptCall::UseAction(press) => script
            .lua()
            .globals()
            .get::<benilla_pfui::LuaFunction>("GetActionInfo")
            .and_then(|f| f.call::<(Option<String>, Option<u32>)>(press.action))
            .ok()
            .and_then(|(kind, id)| {
                (kind.as_deref() == Some("spell")).then_some((id?, press.on_self, None))
            }),
        _ => None,
    }
}

pub(crate) fn intercept(
    script: &UiScript,
    call: ScriptCall,
    cast: &crate::spell::ScriptCast,
) -> Option<ScriptCall> {
    if !api::installed(script.lua()) {
        return Some(call);
    }
    // Nampower's right click with nothing held drops a queued ground spell: the TurnOrAction
    // press arms the world click.
    if matches!(
        call,
        ScriptCall::WorldClick(benilla_ui::script::ClickCall::Arm(
            benilla_ui::script::ClickMode::Action
        ))
    ) && script.cursor_payload().is_none()
    {
        clear_ground(script);
        return Some(call);
    }
    let point = script
        .lua()
        .app_data_ref::<PickPoint>()
        .and_then(|pick| pick.0);
    let (call, policy) = match call {
        ScriptCall::PfuiClick(ClickAction::Cast { call, policy }) => (*call, policy),
        call => (call, QueuePolicy::Automatic),
    };
    let call = super::clicks::normalize_targets(call, cast)?;
    if !api::installed(script.lua()) {
        return Some(call);
    }
    if let ScriptCall::PfuiClick(ClickAction::QueueScript { body, priority }) = call {
        return scripts::intercept(script, cast, body, priority);
    }
    initialize(script);
    if let QueuePolicy::Release { generation, .. } = policy {
        if script
            .lua()
            .app_data_ref::<State>()
            .expect("queue")
            .generation
            != generation.0
        {
            return None;
        }
    }
    script.lua().app_data_mut::<State>().expect("queue").policy = policy;
    if let ScriptCall::PfuiClick(ClickAction::CancelQueue { ground_only }) = call {
        if ground_only {
            clear_ground(script);
        } else {
            clear(script);
        }
        return None;
    }
    if matches!(call, ScriptCall::SpellStopCasting) {
        clear(script);
        return Some(call);
    }
    if matches!(call, ScriptCall::SpellStopTargeting) {
        clear_ground(script);
        return Some(call);
    }
    let Some((id, on_self, target)) = spell(script, &call) else {
        return Some(call);
    };
    let Some(def) = cast.ladder.spells.as_ref().and_then(|s| s.catalog.get(id)) else {
        return Some(call);
    };
    // Attack exits TryCast before the ordinary caster. Trade activities also retain the stock
    // path, as documented by Nampower's FAQ.
    if !retryable(def) {
        return Some(call);
    }
    let lane = lane(id, def);
    let ground = crate::spell::cast_target::cast_target_mask(def) & 0x60 != 0;
    if let QueuePolicy::Release { generation, .. } = policy {
        if ground
            && script
                .lua()
                .app_data_ref::<State>()
                .expect("queue")
                .ground_generation
                != generation.1
        {
            return None;
        }
    }
    let point = point
        .filter(|point| point.is_finite())
        .map(benilla_assets::coords::bevy_to_wow);
    let quickcast = ground
        && target.is_none()
        && !matches!(call, ScriptCall::PfuiClick(ClickAction::GroundSpell { .. }))
        && (api::cvar(script.lua(), "NP_QuickcastTargetingSpells") != 0.0
            || (cast.ladder.ground.spell() == Some(id)
                && api::cvar(script.lua(), "NP_QuickcastOnDoubleCast") != 0.0));
    let call = if let Some(point) = point.filter(|_| quickcast) {
        ScriptCall::PfuiClick(ClickAction::GroundSpell {
            spell_id: id,
            point,
        })
    } else {
        call
    };
    if matches!(policy, QueuePolicy::Never | QueuePolicy::Release { .. }) {
        return Some(call);
    }
    let now = api::now(script.lua());
    if policy != QueuePolicy::Force
        && ((ground && api::cvar(script.lua(), "NP_QueueTargetingSpells") == 0.0)
            || (lane == Lane::OnSwing
                && (api::cvar(script.lua(), "NP_QueueOnSwingSpells") == 0.0
                    || script
                        .lua()
                        .app_data_ref::<State>()
                        .is_some_and(|state| now < state.on_swing_after))))
    {
        return Some(call);
    }
    let is_cast = cast.ladder.spells.as_ref().is_some_and(|s| {
        s.cast_times
            .get(def.casting_time_index)
            .is_some_and(|t| t.base_ms > 0)
    });
    if policy != QueuePolicy::Force
        && lane == Lane::Normal
        && !ground
        && api::cvar(
            script.lua(),
            if is_cast {
                "NP_QueueCastTimeSpells"
            } else {
                "NP_QueueInstantSpells"
            },
        ) == 0.0
    {
        return Some(call);
    }
    let active = benilla_ui::script::pfui_host::casts::player_cast(script.lua());
    let remaining = active
        .as_ref()
        .map_or(0.0, |c| c.end_ms - now * 1000.0)
        .max(0.0);
    let channel = active.as_ref().is_some_and(|c| c.channel);
    if policy != QueuePolicy::Force
        && (channel || def.attributes_ex & 0x44 != 0)
        && api::cvar(script.lua(), "NP_QueueChannelingSpells") == 0.0
    {
        return Some(call);
    }
    let cooldown_info = cast.ladder.cooldowns.info(id, 0, Some(def), Instant::now());
    let cooldown = cooldown_info.remaining_ms;
    let window = api::cvar(
        script.lua(),
        if channel {
            "NP_ChannelQueueWindowMs"
        } else if ground {
            "NP_TargetingQueueWindowMs"
        } else {
            "NP_SpellQueueWindowMs"
        },
    );
    let buffered = script.lua().app_data_ref::<State>().is_some_and(|s| {
        (lane != Lane::OnSwing && now < s.non_gcd_after)
            || (lane == Lane::NonGcd && !s.non_gcd.is_empty())
            || (lane == Lane::OnSwing && cast.ladder.queued_melee.current().is_some())
    });
    let blocked =
        cast.ladder.pending.current().is_some() || remaining > 0.0 || cooldown > 0 || buffered;
    if !blocked {
        let old = script.lua().app_data_mut::<State>().and_then(|mut s| {
            if lane == Lane::Normal {
                s.queued.take()
            } else {
                None
            }
        });
        if let Some(old) = old {
            event(script, old.lane.popped(), old.spell);
        }
        return Some(call);
    }
    let cooldown_window = if cooldown_info.duration_ms > def.start_recovery_ms {
        api::cvar(script.lua(), "NP_CooldownQueueWindowMs")
    } else {
        window
    };
    if policy != QueuePolicy::Force
        && ((lane != Lane::OnSwing && remaining > window)
            || f64::from(cooldown) > cooldown_window
            || (cooldown > 0
                && cooldown_info.duration_ms > def.start_recovery_ms
                && api::cvar(script.lua(), "NP_QueueSpellsOnCooldown") == 0.0))
    {
        return Some(call);
    }
    // A queued ground cast quickcasts even when ordinary quickcasting is disabled. Sky has no
    // location to retain, so it stays on the ordinary targeting path rather than inventing one.
    if ground && target.is_none() && point.is_none() {
        return Some(call);
    }
    let selection = if on_self || ground {
        None
    } else {
        cast.targeting.selection.guid
    };
    let call = if ground && target.is_none() {
        ScriptCall::PfuiClick(ClickAction::GroundSpell {
            spell_id: id,
            point: point.expect("ground hit checked"),
        })
    } else if let Some(target) = target {
        ScriptCall::PfuiClick(ClickAction::Spell {
            spell_id: id,
            target: Some(target),
        })
    } else {
        // The queue releases it itself, as Nampower's own code does, outside the verb's gate.
        ScriptCall::CastSpell {
            spell_id: id,
            on_self,
            may_cast: true,
        }
    };
    let replace_category = api::cvar(script.lua(), "NP_ReplaceMatchingNonGcdCategory") != 0.0;
    let queued = Queued {
        call,
        spell: id,
        selection,
        ready_after: None,
        expires: if policy == QueuePolicy::Force || lane == Lane::OnSwing {
            f64::INFINITY
        } else {
            now + remaining.max(f64::from(cooldown)).max(window) / 1000.0 + 3.0
        },
        lane,
        category: def.start_recovery_category,
        ground,
        retry: None,
    };
    let old = {
        let mut state = script.lua().app_data_mut::<State>().expect("queue");
        if lane == Lane::Normal {
            state.queued.replace(queued)
        } else if lane == Lane::OnSwing {
            state.on_swing.replace(queued)
        } else if let Some(index) = state.non_gcd.iter().position(|old| {
            old.spell == id
                || (replace_category && queued.category != 0 && old.category == queued.category)
        }) {
            Some(std::mem::replace(&mut state.non_gcd[index], queued))
        } else if state.non_gcd.len() < 6 {
            state.non_gcd.push_back(queued);
            None
        } else {
            return Some(queued.call);
        }
    };
    if let Some(old) = old {
        event(script, old.lane.popped(), old.spell)
    }
    event(script, lane.pushed(), id);
    None
}

/// The calls the queue releases this frame: a queued script, or the first lane whose spell is
/// ready. The cancels reach the queue as their calls apply ([`intercept`]).
fn resume(script: &mut UiScript, cast: &crate::spell::ScriptCast) -> Vec<ScriptCall> {
    if !api::installed(script.lua()) {
        return Vec::new();
    }
    initialize(script);
    if let Some(call) = scripts::resume(script, cast) {
        return vec![call];
    }
    let now = api::now(script.lua());
    let buffer = api::cvar(script.lua(), "NP_MinBufferTimeMs") / 1000.0;
    let queued = (|| {
        let mut state = script.lua().app_data_mut::<State>().expect("queue");
        let buffered = now < state.non_gcd_after;
        let non_gcd = !state.non_gcd.is_empty();
        // The on-swing lane is independent: a normal spell on cooldown cannot hold up a strike,
        // and a strike awaiting its swing cannot hold up the other lanes.
        let first = if non_gcd { Lane::NonGcd } else { Lane::Normal };
        for chosen in [first, Lane::OnSwing] {
            let Some(q) = (if chosen == Lane::NonGcd {
                state.non_gcd.front_mut()
            } else if chosen == Lane::Normal {
                state.queued.as_mut()
            } else {
                state.on_swing.as_mut()
            }) else {
                continue;
            };
            let def = cast
                .ladder
                .spells
                .as_ref()
                .and_then(|s| s.catalog.get(q.spell));
            let ready = (chosen != Lane::OnSwing || cast.ladder.queued_melee.current().is_none())
                && !cast.ladder.pending.current().is_some()
                && cast
                    .ladder
                    .cooldowns
                    .info(q.spell, 0, def, Instant::now())
                    .remaining_ms
                    == 0
                && benilla_ui::script::pfui_host::casts::player_cast(script.lua()).is_none();
            if !ready {
                q.ready_after = None;
            }
            if ready && q.ready_after.is_none() {
                q.ready_after = Some(now + buffer);
            }
            if now >= q.expires
                || ((chosen == Lane::OnSwing || !buffered)
                    && q.ready_after.is_some_and(|at| now >= at))
            {
                return if chosen == Lane::NonGcd {
                    state.non_gcd.pop_front()
                } else if chosen == Lane::Normal {
                    state.queued.take()
                } else {
                    state.on_swing.take()
                };
            }
        }
        None
    })();
    let Some(queued) = queued else {
        return Vec::new();
    };
    let generation = script
        .lua()
        .app_data_ref::<State>()
        .map(|state| (state.generation, state.ground_generation))
        .expect("queue");
    event(script, queued.lane.popped(), queued.spell);
    if now >= queued.expires
        || (queued.retry.is_some()
            && api::cvar(script.lua(), "NP_RetryServerRejectedSpells") == 0.0)
    {
        return Vec::new();
    }
    // Release once through the validator; only an observed transient server rejection may retry.
    let call = ScriptCall::PfuiClick(ClickAction::Cast {
        call: Box::new(queued.call.clone()),
        policy: QueuePolicy::Release {
            retry: queued.retry,
            generation,
        },
    });
    if matches!(
        queued.call,
        ScriptCall::PfuiClick(ClickAction::Spell {
            target: Some(_),
            ..
        }) | ScriptCall::PfuiClick(ClickAction::GroundSpell { .. })
            | ScriptCall::CastSpell { on_self: true, .. }
    ) {
        return vec![call];
    }
    if queued
        .selection
        .is_some_and(|guid| cast.targeting.pfui_context(guid).is_none())
    {
        return Vec::new();
    }
    vec![ScriptCall::PfuiClick(ClickAction::RestoreSelectionCast {
        call: Box::new(call),
        selection: queued.selection,
    })]
}

pub(crate) struct Attempt {
    spell: u32,
    target: u64,
    kind: i64,
    failures: usize,
    pending: Option<u32>,
    strike: Option<u32>,
    call: ScriptCall,
    on_self: bool,
    policy: QueuePolicy,
}

pub(crate) fn before(
    script: &UiScript,
    call: &ScriptCall,
    cast: &crate::spell::ScriptCast,
) -> Option<Attempt> {
    if !api::installed(script.lua()) {
        return None;
    }
    let (spell, on_self, target) = spell(script, call)?;
    let def = cast
        .ladder
        .spells
        .as_ref()
        .and_then(|s| s.catalog.get(spell));
    let kind = def.map_or(0, |d| {
        if d.on_next_swing() {
            2
        } else if crate::spell::cast_target::cast_target_mask(d) & 0x60 != 0 {
            if lane(spell, d) == Lane::NonGcd {
                5
            } else {
                4
            }
        } else if d.attributes_ex & 0x44 != 0 {
            3
        } else if lane(spell, d) == Lane::NonGcd {
            1
        } else {
            0
        }
    });
    Some(Attempt {
        spell,
        target: target.unwrap_or_else(|| {
            if on_self {
                api::player(script.lua())
            } else {
                cast.targeting.selection.guid.unwrap_or(0)
            }
        }),
        kind,
        failures: cast.ladder.reports.pfui_refusals(spell).0,
        pending: cast.ladder.pending.current(),
        strike: cast.ladder.queued_melee.current(),
        call: call.clone(),
        on_self,
        policy: script
            .lua()
            .app_data_ref::<State>()
            .map_or(QueuePolicy::Automatic, |state| state.policy),
    })
}

pub(crate) fn after(script: &UiScript, attempt: Option<Attempt>, cast: &crate::spell::ScriptCast) {
    let Some(a) = attempt else { return };
    let (refusals, reason) = cast.ladder.reports.pfui_refusals(a.spell);
    if let Some(reason) = reason.filter(|_| refusals > a.failures) {
        api::fire(
            script.lua(),
            None,
            "SPELL_FAILED_SELF",
            vec![
                V::Int(i64::from(a.spell)),
                V::Int(i64::from(reason)),
                V::Int(0),
            ],
        );
        api::fire(
            script.lua(),
            None,
            "SPELL_CAST_EVENT",
            vec![
                V::Int(0),
                V::Int(i64::from(a.spell)),
                V::Int(a.kind),
                V::Str(format!("0x{:016X}", a.target)),
                V::Int(0),
            ],
        );
        return;
    }
    if (cast.ladder.pending.current() == Some(a.spell) && a.pending != Some(a.spell))
        || (a.kind == 2
            && cast.ladder.queued_melee.current() == Some(a.spell)
            && a.strike != Some(a.spell))
    {
        retries::remember(script, &a, cast);
        if a.kind == 1 || a.kind == 5 {
            let after =
                api::now(script.lua()) + api::cvar(script.lua(), "NP_NonGcdBufferTimeMs") / 1000.0;
            if let Some(mut state) = script.lua().app_data_mut::<State>() {
                state.non_gcd_after = after;
            }
        }
        if a.kind == 2 {
            let after = api::now(script.lua())
                + api::cvar(script.lua(), "NP_OnSwingBufferCooldownMs") / 1000.0;
            if let Some(mut state) = script.lua().app_data_mut::<State>() {
                state.on_swing_after = after;
            }
        }
        api::fire(
            script.lua(),
            None,
            "SPELL_CAST_EVENT",
            vec![
                V::Int(1),
                V::Int(i64::from(a.spell)),
                V::Int(a.kind),
                V::Str(format!("0x{:016X}", a.target)),
                V::Int(0),
            ],
        );
    }
}

fn retryable(def: &benilla_formats::SpellDisplay) -> bool {
    !def.ranged_attack()
        && !def.effects.contains(&78)
        && def.attributes & 0x20 == 0
        && !def
            .effects
            .iter()
            .any(|effect| matches!(effect, 24 | 33 | 47 | 53 | 54 | 59))
}

#[cfg(test)]
#[path = "spell_queue_tests.rs"]
pub(crate) mod tests;
