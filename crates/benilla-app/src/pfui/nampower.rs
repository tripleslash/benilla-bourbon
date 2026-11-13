//! Project observed server packets into Nampower's documented pfUI event ABI.

use super::casts::duration;
use crate::net::NetHandlerApp;
use benilla_pfui::nampower as api;
use benilla_protocol::{messages::PeriodicTick, SessionEvent, SessionEventKind};
use benilla_ui::script::{ScriptValue as V, UiScript};
use bevy::prelude::*;
use std::collections::HashSet;

#[derive(Resource, Default)]
struct Events(Vec<SessionEvent>);

#[cfg(test)]
#[path = "focus_session_tests.rs"]
mod focus_session_tests;

#[cfg(test)]
#[path = "spell_retry_tests.rs"]
mod retry_tests;

pub(super) fn register(app: &mut App) {
    app.init_resource::<Events>();
    app.net_handler(SessionEventKind::SpellStart, receive);
    app.net_handler(SessionEventKind::SpellGo, receive);
    app.net_handler(SessionEventKind::SpellFailedOther, receive);
    app.net_handler(SessionEventKind::CastResult, receive);
    app.net_handler(SessionEventKind::SpellDelayed, receive);
    app.net_handler(SessionEventKind::ChannelStart, receive);
    app.net_handler(SessionEventKind::ChannelUpdate, receive);
    app.net_handler(SessionEventKind::AuraDuration, receive);
    app.net_handler(SessionEventKind::SpellHealLog, receive);
    app.net_handler(SessionEventKind::PeriodicAuraLog, receive);
    app.net_handler(SessionEventKind::AttackerState, receive);
    app.net_handler(SessionEventKind::Disconnected, receive);
    app.net_handler(SessionEventKind::LoggedOut, receive);
    app.add_systems(
        Update,
        feed.in_set(crate::ui_script::UiFeed)
            .after(super::spells::SpellSnapshots)
            .after(super::casts::CastFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
    app.add_systems(
        Update,
        aura_edges
            .in_set(crate::ui_script::UiFeed)
            .after(crate::ui_unit::UnitFeed)
            .after(crate::ui_aura::pfui::VisibleFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
}

fn receive(
    In(event): In<SessionEvent>,
    mut events: ResMut<Events>,
    script: Option<NonSend<UiScript>>,
    focus: Res<super::SessionFocus>,
) {
    match event {
        SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => {
            events.0.clear();
            focus.0.clear();
            if let Some(script) = script {
                benilla_ui::script::pfui_host::units::clear_focus(script.lua());
                super::spell_queue::reset(&script);
                api::reset(script.lua());
            }
        }
        SessionEvent::SpellStart { .. }
        | SessionEvent::SpellGo { .. }
        | SessionEvent::SpellFailedOther { .. }
        | SessionEvent::CastResult { .. }
        | SessionEvent::SpellDelayed { .. }
        | SessionEvent::ChannelStart { .. }
        | SessionEvent::ChannelUpdate { .. }
        | SessionEvent::AuraDuration { .. }
        | SessionEvent::SpellHealLog { .. }
        | SessionEvent::PeriodicAuraLog { .. }
        | SessionEvent::AttackerState(_) => events.0.push(event),
        _ => (),
    }
}
fn guid(g: u64) -> V {
    V::Str(format!("0x{g:016X}"))
}
fn int(n: impl Into<i64>) -> V {
    V::Int(n.into())
}

#[allow(clippy::too_many_arguments)]
fn feed(
    script: Option<NonSend<UiScript>>,
    mut events: ResMut<Events>,
    me: Res<crate::net::SelfGuid>,
    spells: Option<Res<crate::ui_action::Spells>>,
    objects: crate::net::Objects,
    mods: Res<crate::spell::SpellModifiers>,
    mut channel: Local<crate::ui_script::VmMemo<Option<(u32, u64)>>>,
) {
    let Some(script) = script else {
        events.0.clear();
        return;
    };
    let channel = channel.get(&script);
    let lua = script.lua();
    let me = me.0.unwrap_or(0);
    for event in events.0.drain(..) {
        match event {
            SessionEvent::SpellStart {
                caster,
                spell_id,
                target,
                cast_flags,
                cast_time_ms,
                ..
            } => {
                if caster == me {
                    super::spell_queue::accept(&script, spell_id);
                }
                api::target(lua, caster, target.unwrap_or(0));
                let def = spells.as_ref().and_then(|s| s.catalog.get(spell_id));
                let channel = def.is_some_and(|s| s.attributes_ex & 0x44 != 0);
                let duration = spells.as_ref().zip(def).map_or(0, |(s, d)| {
                    duration(
                        s,
                        d,
                        objects.object(caster),
                        (caster == me).then_some(&*mods),
                    )
                });
                api::fire(
                    lua,
                    Some("NP_EnableSpellStartEvents"),
                    if caster == me {
                        "SPELL_START_SELF"
                    } else {
                        "SPELL_START_OTHER"
                    },
                    vec![
                        int(0),
                        int(spell_id),
                        guid(caster),
                        guid(target.unwrap_or(0)),
                        int(cast_flags),
                        int(cast_time_ms),
                        int(if channel { duration.max(0) } else { 0 }),
                        int(if channel {
                            1
                        } else if def.is_some_and(|s| s.auto_repeat()) {
                            2
                        } else {
                            0
                        }),
                    ],
                );
            }
            SessionEvent::SpellGo {
                caster,
                spell_id,
                target,
                cast_flags,
                hits,
                misses,
                item_caster,
                ..
            } => {
                if caster == me {
                    super::spell_queue::accept(&script, spell_id);
                }
                let item = item_caster
                    .and_then(|g| objects.object(g))
                    .and_then(|o| o.object_entry())
                    .unwrap_or(0);
                api::fire(
                    lua,
                    Some("NP_EnableSpellGoEvents"),
                    if caster == me {
                        "SPELL_GO_SELF"
                    } else {
                        "SPELL_GO_OTHER"
                    },
                    vec![
                        int(item),
                        int(spell_id),
                        guid(caster),
                        guid(target.unwrap_or(0)),
                        int(cast_flags),
                        int(hits.len() as i64),
                        int(misses.len() as i64),
                    ],
                );
                if let Some((spells, def)) = spells
                    .as_ref()
                    .and_then(|s| s.catalog.get(spell_id).map(|d| (s, d)))
                {
                    let duration = duration(
                        spells,
                        def,
                        objects.object(caster),
                        (caster == me).then_some(&*mods),
                    );
                    for i in 0..3 {
                        if !matches!(def.effects[i], 6 | 35 | 65 | 119 | 128 | 129) {
                            continue;
                        }
                        let targets = if def.effect_implicit_target_a[i] == 1 {
                            vec![caster]
                        } else {
                            hits.clone()
                        };
                        for target in targets.into_iter().collect::<HashSet<_>>() {
                            api::application(lua, caster, target, spell_id, duration);
                            let cap = objects.object(target).map_or(0, |store| {
                                let buffs = store.unit_auras().filter(|a| a.slot < 32).count();
                                let debuffs = store.unit_auras().filter(|a| a.slot >= 32).count();
                                i64::from(buffs >= 32) | (i64::from(debuffs >= 16) << 1)
                            });
                            api::fire(
                                lua,
                                Some("NP_EnableAuraCastEvents"),
                                if target == me {
                                    "AURA_CAST_ON_SELF"
                                } else {
                                    "AURA_CAST_ON_OTHER"
                                },
                                vec![
                                    int(spell_id),
                                    guid(caster),
                                    guid(target),
                                    int(def.effects[i]),
                                    int(def.effect_apply_aura[i]),
                                    int(def.effect_amplitude[i]),
                                    int(def.effect_misc_value[i]),
                                    int(duration),
                                    int(cap),
                                ],
                            );
                        }
                    }
                }
            }
            SessionEvent::CastResult {
                spell_id,
                success,
                reason,
                ..
            } => {
                if success {
                    super::spell_queue::accept(&script, spell_id);
                } else {
                    super::spell_queue::reject(&script, spell_id, reason.unwrap_or(0));
                    api::fire(
                        lua,
                        None,
                        "SPELL_FAILED_SELF",
                        vec![int(spell_id), int(reason.unwrap_or(0)), int(1)],
                    );
                }
            }
            SessionEvent::SpellFailedOther { caster, spell_id } => api::fire(
                lua,
                None,
                "SPELL_FAILED_OTHER",
                vec![guid(caster), int(spell_id)],
            ),
            SessionEvent::SpellDelayed { caster, delay_ms } => api::fire(
                lua,
                None,
                if caster == me {
                    "SPELL_DELAYED_SELF"
                } else {
                    "SPELL_DELAYED_OTHER"
                },
                vec![guid(caster), int(delay_ms)],
            ),
            SessionEvent::ChannelStart {
                spell_id,
                duration_ms,
            } => {
                let target = objects
                    .object(me)
                    .and_then(|s| s.unit_channel_object())
                    .unwrap_or(0);
                api::target(lua, me, target);
                *channel = (duration_ms > 0).then_some((spell_id, target));
                api::fire(
                    lua,
                    None,
                    "SPELL_CHANNEL_START",
                    vec![int(spell_id), guid(target), int(duration_ms)],
                );
            }
            SessionEvent::ChannelUpdate { remaining_ms } => {
                // Stock has already torn down a completed channel by the time this feed runs.
                let (spell, target) = channel.unwrap_or((0, 0));
                api::fire(
                    lua,
                    None,
                    "SPELL_CHANNEL_UPDATE",
                    vec![int(spell), guid(target), int(remaining_ms)],
                );
                if remaining_ms == 0 {
                    *channel = None;
                }
            }
            SessionEvent::AuraDuration { slot, remaining_ms } => {
                let spell = objects
                    .object(me)
                    .and_then(|s| s.unit_auras().find(|a| a.slot == slot))
                    .map_or(0, |a| a.spell_id);
                // A negative span is no duration (`ui_aura::AuraDurations::set`), 0 to the extension.
                api::update_duration(lua, slot, spell, u32::try_from(remaining_ms).unwrap_or(0));
                let expiration = if remaining_ms == 0 {
                    0.0
                } else {
                    api::now(lua) * 1000.0 + f64::from(remaining_ms)
                };
                api::fire(
                    lua,
                    None,
                    if slot < 32 {
                        "BUFF_UPDATE_DURATION_SELF"
                    } else {
                        "DEBUFF_UPDATE_DURATION_SELF"
                    },
                    vec![
                        int(slot),
                        int(remaining_ms),
                        V::Number(expiration),
                        int(spell),
                    ],
                );
            }
            SessionEvent::SpellHealLog(h) => heal(
                lua, me, h.target, h.healer, h.spell_id, h.amount, h.critical, false,
            ),
            SessionEvent::PeriodicAuraLog(h) => {
                for tick in h.ticks {
                    if let PeriodicTick::Heal { amount } = tick {
                        heal(lua, me, h.target, h.caster, h.spell_id, amount, false, true)
                    }
                }
            }
            SessionEvent::AttackerState(a) => {
                // The stock decoder aggregates damage components; their original count is unknown.
                api::fire(
                    lua,
                    Some("NP_EnableAutoAttackEvents"),
                    if a.attacker == me {
                        "AUTO_ATTACK_SELF"
                    } else {
                        "AUTO_ATTACK_OTHER"
                    },
                    vec![
                        guid(a.attacker),
                        guid(a.victim),
                        int(a.damage),
                        int(a.hit_info),
                        int(a.victim_state),
                        V::Nil,
                        int(a.blocked),
                        int(a.absorb),
                        int(a.resist),
                    ],
                );
            }
            SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => {
                super::spell_queue::clear(&script);
                api::reset(lua);
                *channel = None;
            }
            _ => (),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn heal(
    lua: &benilla_pfui::Lua,
    me: u64,
    target: u64,
    caster: u64,
    spell: u32,
    amount: u32,
    crit: bool,
    periodic: bool,
) {
    let args = vec![
        guid(target),
        guid(caster),
        int(spell),
        int(amount),
        int(i64::from(crit)),
        int(i64::from(periodic)),
    ];
    api::fire(
        lua,
        Some("NP_EnableSpellHealEvents"),
        if caster == me {
            "SPELL_HEAL_BY_SELF"
        } else {
            "SPELL_HEAL_BY_OTHER"
        },
        args.clone(),
    );
    if target == me {
        api::fire(
            lua,
            Some("NP_EnableSpellHealEvents"),
            "SPELL_HEAL_ON_SELF",
            args,
        );
    }
}

fn aura_edges(
    script: Option<NonSend<UiScript>>,
    objects: Query<(&crate::net::Guid, &crate::net::ObjectStore)>,
    spells: Option<Res<crate::ui_action::Spells>>,
    cooldowns: Res<crate::spell::Cooldowns>,
    clock: Res<crate::ui_script::UiClock>,
) {
    let Some(script) = script else { return };
    // A VM without Nampower's state keeps none of what follows.
    if !api::installed(script.lua()) {
        return;
    }
    let mut present = HashSet::new();
    let (now, me) = (api::now(script.lua()), api::player(script.lua()));
    let gcd_end = benilla_ui::script::pfui_host::casts::player_cast(script.lua())
        .and_then(|cast| {
            spells
                .as_ref()
                .and_then(|spells| spells.catalog.get(cast.spell_id))
        })
        .map_or(0.0, |spell| {
            cooldowns.pfui_gcd_end(spell.start_recovery_category, &clock)
        });
    api::set_gcd_end(script.lua(), gcd_end);
    for (guid, store) in &objects {
        if !store.is_unit() {
            continue;
        }
        present.insert(guid.0);
        if let Some(health) = store.0.unit_health() {
            api::health(script.lua(), guid.0, health);
        }
        api::auras_at(
            script.lua(),
            guid.0,
            store
                .0
                .unit_auras()
                .map(|a| api::Aura {
                    slot: a.slot,
                    spell: a.spell_id,
                    stacks: a.stacks,
                    level: a.level,
                    visible: spells
                        .as_ref()
                        .and_then(|s| s.catalog.get(a.spell_id))
                        .is_none_or(|d| !d.hidden_from_aura_bar() && !d.tracking_aura()),
                })
                .collect(),
            now,
            me,
        );
    }
    api::retain_units(script.lua(), &present);
}
