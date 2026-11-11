//! Keep independently observed quest-template packets across UI reloads, never across realms.

use crate::net::NetHandlerApp;
use benilla_protocol::{messages::QuestTemplate, SessionEvent, SessionEventKind};
use benilla_ui::script::{
    pfui_host::quests::{self, Detail, Requirement},
    UiScript,
};
use bevy::prelude::*;
use std::collections::HashMap;

#[derive(Resource, Default)]
struct State {
    templates: HashMap<u32, QuestTemplate>,
    generation: u64,
    completed: Vec<(u32, u32, u32)>,
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<State>();
    app.net_handler(SessionEventKind::QuestTemplate, receive);
    app.net_handler(SessionEventKind::QuestComplete, receive);
    app.net_handler(SessionEventKind::Disconnected, receive);
    app.net_handler(SessionEventKind::LoggedOut, receive);
    app.add_systems(Update, feed.in_set(super::InputSnapshots));
    app.add_systems(
        Update,
        feed_lifecycle
            .after(crate::ui_unit::UnitFeed)
            .in_set(crate::ui_script::UiFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
}

fn receive(In(event): In<SessionEvent>, mut state: ResMut<State>) {
    match event {
        SessionEvent::QuestTemplate(template) => {
            state.templates.insert(template.quest_id, *template);
        }
        SessionEvent::QuestComplete(complete) => {
            state
                .completed
                .push((complete.quest_id, complete.xp, complete.money));
            return;
        }
        SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => {
            state.templates.clear();
            state.completed.clear();
        }
        _ => return,
    }
    state.generation = state.generation.wrapping_add(1);
}

fn feed_lifecycle(
    script: Option<NonSend<UiScript>>,
    mut state: ResMut<State>,
    player: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
) {
    let Some(script) = script else { return };
    let Ok(player) = player.single() else { return };
    // A confirmation precedes the removal event, even when both land in the same frame.
    for (id, xp, copper) in std::mem::take(&mut state.completed) {
        quests::turned_in(script.lua(), id, xp, copper);
    }
    let slots = (0..benilla_protocol::messages::PLAYER_QUEST_LOG_SLOTS)
        .map(|slot| {
            player
                .0
                .player_quest_log(slot)
                .map(|quest| (quest.quest_id, quest.state))
                .unwrap_or_default()
        })
        .collect();
    quests::observe_log_slots(script.lua(), slots);
    quests::dispatch_log_events(script.lua());
}

fn project(template: &QuestTemplate, tag: Option<&str>) -> Detail {
    let mut requirements = Vec::new();
    for objective in &template.objectives {
        let raw = objective.creature_or_go;
        if raw != 0 {
            let game_object = raw & 0x8000_0000 != 0;
            requirements.push(Requirement {
                kind: if game_object { "object" } else { "monster" }.into(),
                id: if game_object {
                    (raw as i32).unsigned_abs() & 0x7fff_ffff
                } else {
                    raw
                },
                count: objective.required_count,
                text: objective.text.clone(),
            });
        }
    }
    for objective in &template.objectives {
        if objective.item_id != 0 {
            requirements.push(Requirement {
                kind: "item".into(),
                id: objective.item_id,
                count: objective.item_count,
                text: String::new(),
            });
        }
    }
    Detail {
        quest_id: template.quest_id,
        title: template.title.clone(),
        level: template.level,
        tag: tag.map(str::to_string),
        money: template.money,
        money_max_level: template.money_max_level,
        reward_spell: template.reward_spell,
        src_item: template.src_item_id,
        flags: template.flags,
        description: template.details.clone(),
        objectives: template.objectives_text.clone(),
        completion: template.end_text.clone(),
        poi: (template.point_map_id != 0).then_some((
            template.point_map_id,
            template.point_x,
            template.point_y,
            template.point_opt,
        )),
        rewards: template.rewards.to_vec(),
        choices: template.choices.to_vec(),
        requirements,
    }
}

fn feed(
    script: Option<NonSend<UiScript>>,
    state: Res<State>,
    tags: Option<Res<crate::ui_quest_log::QuestTagNamesRes>>,
    mut memo: Local<crate::ui_script::VmMemo<Option<u64>>>,
) {
    let Some(script) = script else { return };
    let last = memo.get(&script);
    if *last == Some(state.generation) {
        return;
    }
    quests::set_details(
        script.lua(),
        state.templates.values().map(|template| {
            project(
                template,
                tags.as_ref()
                    .and_then(|tags| tags.0.resolve(template.quest_type)),
            )
        }),
    );
    *last = Some(state.generation);
}
