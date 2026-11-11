//! A school lockout for `C_LossOfControl`: the server interrupts a cast by prohibiting its school
//! (`Player::ProhibitSpellSchool`), which sends one `SMSG_SPELL_COOLDOWN` for the player listing
//! every known spell of that school at one duration. A normal cooldown lists one spell.

use crate::net::NetHandlerApp;
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::UiScript;
use bevy::prelude::*;

#[derive(Resource, Default)]
struct Events(Vec<SessionEvent>);

pub(super) fn register(app: &mut App) {
    app.init_resource::<Events>();
    app.net_handler(SessionEventKind::SpellCooldowns, receive);
    app.add_systems(
        Update,
        feed.in_set(crate::ui_script::UiFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
}

fn receive(In(event): In<SessionEvent>, mut events: ResMut<Events>) {
    if let SessionEvent::SpellCooldowns { .. } = event {
        events.0.push(event);
    }
}

fn feed(
    script: Option<NonSend<UiScript>>,
    mut events: ResMut<Events>,
    me: Res<crate::net::SelfGuid>,
    spells: Option<Res<crate::ui_action::Spells>>,
) {
    let Some(script) = script else {
        events.0.clear();
        return;
    };
    let me = me.0.unwrap_or(0);
    for event in events.0.drain(..) {
        let SessionEvent::SpellCooldowns { caster, cooldowns } = event else {
            continue;
        };
        if caster != me || cooldowns.len() < 2 {
            continue;
        }
        let Some(&(_, ms)) = cooldowns.first() else {
            continue;
        };
        let school = |id: u32| {
            spells
                .as_ref()
                .and_then(|s| s.catalog.get(id))
                .map(|d| d.school)
        };
        let first = school(cooldowns[0].0);
        let uniform = ms > 0
            && cooldowns
                .iter()
                .all(|&(id, d)| d == ms && school(id) == first && first.is_some());
        if let (true, Some(school)) = (uniform, first) {
            benilla_pfui::school_lockout(script.lua(), 1 << school.min(6), f64::from(ms) / 1000.0);
        }
    }
}
