//! Additional packet observers preserve the stock roll UI and feed the optional history VM.

use crate::net::NetHandlerApp;
use benilla_pfui::loot::{self, Event, Key, Player};
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::UiScript;
use bevy::prelude::*;

pub(super) fn register(app: &mut App) {
    app.net_handler(SessionEventKind::LootStartRoll, receive);
    app.net_handler(SessionEventKind::LootRoll, receive);
    app.net_handler(SessionEventKind::LootRollWon, receive);
    app.net_handler(SessionEventKind::LootAllPassed, receive);
}

fn receive(
    In(event): In<SessionEvent>,
    script: Option<NonSend<UiScript>>,
    names: Res<crate::names::NameCache>,
    objects: crate::net::Objects,
    commands: Res<crate::net::NetCommands>,
    me: Res<crate::net::SelfGuid>,
    clock: Res<crate::ui_script::UiClock>,
    self_q: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
) {
    let Some(script) = script else { return };
    let player = |guid, kind, roll| {
        let class = names
            .player_traits(guid)
            .map(|(_, class, _)| class)
            .or_else(|| {
                (me.0 == Some(guid))
                    .then(|| self_q.single().ok()?.0.unit_class())
                    .flatten()
            });
        Player {
            guid,
            kind,
            roll,
            is_me: me.0 == Some(guid),
            name: names
                .resolve(&objects, guid, crate::query_cache::Ask::New(&commands))
                .map(str::to_owned),
            class: class
                .and_then(crate::pfui::class_names)
                .map(|(_, token)| token.to_owned()),
        }
    };
    let (key, id, suffix, event) = match event {
        SessionEvent::LootStartRoll(p) => (
            Key {
                target: p.looted_target,
                slot: p.item_slot,
            },
            p.item_id,
            p.random_property_id,
            Event::Start,
        ),
        SessionEvent::LootRoll(p) => {
            let Some(kind) = p.vote().or_else(|| p.is_dice().then_some(p.roll_type)) else {
                return;
            };
            (
                Key {
                    target: p.looted_target,
                    slot: p.item_slot,
                },
                p.item_id,
                p.random_property_id,
                Event::Roll(player(
                    p.roller,
                    kind,
                    if p.is_dice() { p.roll_number } else { 0 },
                )),
            )
        }
        SessionEvent::LootRollWon(p) => (
            Key {
                target: p.looted_target,
                slot: p.item_slot,
            },
            p.item_id,
            p.random_property_id,
            Event::Won(player(p.winner, p.roll_type, p.roll_number)),
        ),
        SessionEvent::LootAllPassed(p) => (
            Key {
                target: p.looted_target,
                slot: p.item_slot,
            },
            p.item_id,
            p.random_property_id,
            Event::Passed,
        ),
        _ => return,
    };
    let timestamp = clock.ui_now
        + bevy::platform::time::Instant::now()
            .duration_since(clock.anchor)
            .as_secs_f64();
    loot::observe(script.lua(), key, id, suffix as i32, timestamp, event);
}
