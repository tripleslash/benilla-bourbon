//! Summons follow successful casts, DBC effects and the actual owner/entry object lifetime.

use crate::net::NetHandlerApp;
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::{
    pfui_host::totems::{self, Totem},
    UiScript,
};
use bevy::{platform::time::Instant, prelude::*};

struct Active {
    spell: u32,
    entry: u32,
    start: Instant,
    duration: f64,
    guid: Option<u64>,
}
#[derive(Resource, Default)]
struct State {
    owner: Option<u64>,
    active: [Option<Active>; 4],
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<State>();
    app.net_handler(SessionEventKind::SpellGo, receive);
    app.net_handler(SessionEventKind::Disconnected, receive);
    app.net_handler(SessionEventKind::LoggedOut, receive);
    app.add_systems(Update, feed.in_set(super::InputSnapshots));
}

fn summon(display: &benilla_formats::SpellDisplay) -> Option<(usize, u32)> {
    display
        .effects
        .iter()
        .enumerate()
        .find_map(|(index, &effect)| {
            (87..=90).contains(&effect).then(|| {
                (
                    (effect - 87) as usize,
                    display.effect_misc_value[index] as u32,
                )
            })
        })
}

fn receive(
    In(event): In<SessionEvent>,
    mut state: ResMut<State>,
    me: Res<crate::net::SelfGuid>,
    spells: Option<Res<crate::ui_action::Spells>>,
    skill_lines: Option<Res<crate::ui_spellbook::SkillLines>>,
    self_q: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
    mods: Res<crate::spell::SpellModifiers>,
) {
    match event {
        SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => *state = State::default(),
        SessionEvent::SpellGo {
            caster, spell_id, ..
        } if me.0 == Some(caster) => {
            let Some(spells) = spells else { return };
            let Some(display) = spells.catalog.get(spell_id) else {
                return;
            };
            let Some((slot, entry)) = summon(display) else {
                return;
            };
            let skill = crate::spell::spell_skill_value(
                self_q.single().ok(),
                skill_lines.as_ref().map(|lines| &lines.catalog),
                spell_id,
            );
            let duration = spells
                .durations
                .get(display.duration_index)
                .map_or(0, |row| {
                    let levels = display.skill_level(skill) as i32 - display.base_level as i32;
                    mods.apply(
                        display,
                        1,
                        row.base_ms
                            .saturating_add(levels.saturating_mul(row.per_level_ms))
                            .min(row.max_ms),
                    )
                })
                .max(0) as f64
                / 1000.;
            state.owner = Some(caster);
            state.active[slot] = Some(Active {
                spell: spell_id,
                entry,
                start: Instant::now(),
                duration,
                guid: None,
            });
        }
        _ => {}
    }
}

fn feed(
    script: Option<NonSend<UiScript>>,
    mut state: ResMut<State>,
    me: Res<crate::net::SelfGuid>,
    spells: Option<Res<crate::ui_action::Spells>>,
    actions: Res<crate::ui_action::PlayerActions>,
    objects: Query<(&crate::net::Guid, &crate::net::ObjectStore)>,
    clock: Res<crate::ui_script::UiClock>,
) {
    let Some(script) = script else { return };
    if state.owner != me.0 {
        state.owner = me.0;
        state.active = std::array::from_fn(|_| None);
    }
    let now = Instant::now();
    for active in &mut state.active {
        let Some(totem) = active else { continue };
        let elapsed = now.duration_since(totem.start).as_secs_f64();
        let guid = objects
            .iter()
            .find(|(guid, store)| {
                totem.guid.is_none_or(|seen| guid.0 == seen)
                    && store.0.object_entry() == Some(totem.entry)
                    && me.0.is_some_and(|owner| {
                        store.0.unit_summoned_by() == Some(owner)
                            || store.0.unit_created_by() == Some(owner)
                    })
            })
            .map(|(guid, _)| guid.0);
        if (totem.duration > 0. && elapsed >= totem.duration)
            || (totem.guid.is_some() && guid.is_none())
        {
            *active = None;
        } else {
            totem.guid = guid.or(totem.guid);
        }
    }
    let mut tools = [0; 4];
    if let Some(spells) = &spells {
        for spell in &actions.spells {
            if let Some(display) = spells.catalog.get(*spell) {
                if let Some((slot, _)) = summon(display) {
                    tools[slot] = display.totems[0];
                }
            }
        }
    }
    let active = std::array::from_fn(|slot| {
        state.active[slot].as_ref().map(|active| {
            let display = spells
                .as_ref()
                .and_then(|spells| spells.catalog.get(active.spell));
            let offset = if active.start >= clock.anchor {
                active.start.duration_since(clock.anchor).as_secs_f64()
            } else {
                -clock.anchor.duration_since(active.start).as_secs_f64()
            };
            Totem {
                spell_id: active.spell,
                guid: active.guid,
                name: display
                    .map(|display| display.name.clone())
                    .unwrap_or_default(),
                icon: display.and_then(|display| display.icon.clone()),
                start: clock.ui_now + offset,
                duration: active.duration,
            }
        })
    });
    totems::set(script.lua(), active, tools);
}
