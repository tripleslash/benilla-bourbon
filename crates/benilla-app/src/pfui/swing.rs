//! Feed the extension's swing clock from the attack and spell packets, and announce its resets.

use crate::net::NetHandlerApp;
use benilla_pfui::swing;
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::UiScript;
use bevy::prelude::*;

/// `HITINFO_LEFTSWING`: the off hand swung.
const LEFT_SWING: u32 = 0x4;
/// `VICTIMSTATE_PARRY`.
const PARRY: u32 = 3;

#[derive(Resource, Default)]
struct Events(Vec<SessionEvent>);

pub(super) fn register(app: &mut App) {
    app.init_resource::<Events>();
    app.net_handler(SessionEventKind::AttackerState, receive);
    app.net_handler(SessionEventKind::AttackStart, receive);
    app.net_handler(SessionEventKind::AttackStop, receive);
    app.net_handler(SessionEventKind::SpellStart, receive);
    app.net_handler(SessionEventKind::SpellGo, receive);
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
        SessionEvent::AttackerState(_)
        | SessionEvent::AttackStart { .. }
        | SessionEvent::AttackStop { .. }
        | SessionEvent::SpellStart { .. }
        | SessionEvent::SpellGo { .. }
        | SessionEvent::Disconnected { .. }
        | SessionEvent::LoggedOut => events.0.push(event),
        _ => (),
    }
}

/// Runs after the unit feed, so the weapon speeds a reset reads are this frame's.
fn feed(
    script: Option<NonSend<UiScript>>,
    mut events: ResMut<Events>,
    me: Res<crate::net::SelfGuid>,
) {
    let Some(script) = script else {
        events.0.clear();
        return;
    };
    let lua = script.lua();
    // No player yet: no packet below can name one.
    let me = me.0.filter(|guid| *guid != 0);
    for event in events.0.drain(..) {
        match event {
            SessionEvent::AttackerState(a) if Some(a.attacker) == me => {
                swing::swung(lua, a.hit_info & LEFT_SWING != 0);
            }
            SessionEvent::AttackerState(a) if Some(a.victim) == me && a.victim_state == PARRY => {
                swing::parried(lua);
            }
            SessionEvent::AttackStart { attacker, .. } if Some(attacker) == me => {
                swing::attacking(lua, true);
            }
            SessionEvent::AttackStop { attacker, .. } if Some(attacker) == me => {
                swing::attacking(lua, false);
            }
            SessionEvent::SpellStart {
                caster, spell_id, ..
            } if Some(caster) == me => swing::cast_started(lua, spell_id),
            SessionEvent::SpellGo {
                caster, spell_id, ..
            } if Some(caster) == me => swing::cast_went(lua, spell_id),
            SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => swing::clear(lua),
            _ => (),
        }
    }
    swing::flush(lua);
}
