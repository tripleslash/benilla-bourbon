//! Feed `UnitHasIncomingResurrection` ([`benilla_pfui::resurrect`]) from the casts the session
//! shows: a spell with a resurrection effect cast at a unit or a released player's corpse, from
//! its start to its go, its failure, or the end of its cast time.

use crate::net::NetHandlerApp;
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::UiScript;
use bevy::prelude::*;
use std::collections::HashMap;

/// `SPELL_EFFECT_RESURRECT` and `SPELL_EFFECT_RESURRECT_NEW`.
const RESURRECT: [u32; 2] = [18, 113];
/// A cast whose go or failure never came counts this long past its cast time.
const GRACE: f64 = 0.5;

#[derive(Resource, Default)]
struct Events(Vec<SessionEvent>);

pub(super) fn register(app: &mut App) {
    app.init_resource::<Events>();
    app.net_handler(SessionEventKind::SpellStart, receive);
    app.net_handler(SessionEventKind::SpellGo, receive);
    app.net_handler(SessionEventKind::SpellFailedOther, receive);
    app.net_handler(SessionEventKind::Disconnected, receive);
    app.net_handler(SessionEventKind::LoggedOut, receive);
    app.add_systems(
        Update,
        feed.in_set(crate::ui_script::UiFeed)
            .after(crate::ui_unit::UnitFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
}

fn receive(In(event): In<SessionEvent>, mut events: ResMut<Events>) {
    match event {
        SessionEvent::SpellStart { .. }
        | SessionEvent::SpellGo { .. }
        | SessionEvent::SpellFailedOther { .. }
        | SessionEvent::Disconnected { .. }
        | SessionEvent::LoggedOut => events.0.push(event),
        _ => (),
    }
}

/// The unit tokens that name `guid`: the player, the target, the focus, the pet, and the party and
/// raid slots.
fn tokens(
    guid: u64,
    me: u64,
    pet: u64,
    selection: &crate::target::Selection,
    focus: Option<u64>,
    group: &crate::ui_party::GroupState,
) -> Vec<String> {
    let mut out = Vec::new();
    if guid == me {
        out.push("player".to_string());
    }
    if pet != 0 && guid == pet {
        out.push("pet".to_string());
    }
    if selection.guid == Some(guid) {
        out.push("target".to_string());
    }
    if focus == Some(guid) {
        out.push("focus".to_string());
    }
    for (i, member) in group.party_slots().enumerate() {
        if member.guid == guid {
            out.push(format!("party{}", i + 1));
        }
    }
    for (i, row) in group.raid_rows().enumerate() {
        if row == guid {
            out.push(format!("raid{}", i + 1));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn feed(
    script: Option<NonSend<UiScript>>,
    mut events: ResMut<Events>,
    time: Res<Time>,
    objects: crate::net::Objects,
    spells: Option<Res<crate::ui_action::Spells>>,
    selection: Res<crate::target::Selection>,
    group: Res<crate::ui_party::GroupState>,
    me: Res<crate::net::SelfGuid>,
    mut casts: Local<crate::ui_script::VmMemo<HashMap<u64, (u64, f64)>>>,
) {
    let Some(script) = script else {
        events.0.clear();
        return;
    };
    let casts = casts.get(&script);
    let lua = script.lua();
    let now = time.elapsed_secs_f64();
    // A corpse's cast names its owner.
    let unit_of = |guid: u64| {
        objects
            .object(guid)
            .and_then(|f| f.corpse_owner())
            .filter(|owner| *owner != 0)
            .unwrap_or(guid)
    };
    let before: Vec<u64> = casts.values().map(|(target, _)| *target).collect();
    for event in events.0.drain(..) {
        match event {
            SessionEvent::SpellStart {
                caster,
                spell_id,
                cast_time_ms,
                target: Some(target),
                ..
            } => {
                let resurrects = spells
                    .as_ref()
                    .and_then(|s| s.catalog.get(spell_id))
                    .is_some_and(|d| d.effects.iter().any(|e| RESURRECT.contains(e)));
                if resurrects && cast_time_ms > 0 {
                    let ends = now + f64::from(cast_time_ms) / 1000.0 + GRACE;
                    casts.insert(caster, (unit_of(target), ends));
                }
            }
            SessionEvent::SpellGo { caster, .. }
            | SessionEvent::SpellFailedOther { caster, .. } => {
                casts.remove(&caster);
            }
            SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => {
                casts.clear();
                benilla_pfui::resurrect::clear(lua);
            }
            _ => {}
        }
    }
    casts.retain(|_, (_, ends)| *ends > now);
    let after: Vec<u64> = casts.values().map(|(target, _)| *target).collect();
    let me = me.0.unwrap_or(0);
    let pet = objects
        .player_fields()
        .and_then(|f| f.unit_summon())
        .unwrap_or(0);
    let focus = benilla_ui::script::pfui_host::units::custom_guid(lua, "focus").flatten();
    for guid in before.iter().chain(&after) {
        let names = tokens(*guid, me, pet, &selection, focus, &group);
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        benilla_pfui::resurrect::set(lua, *guid, after.contains(guid), &names);
    }
}
