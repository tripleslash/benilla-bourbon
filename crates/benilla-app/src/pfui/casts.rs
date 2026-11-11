//! Cast timing comes from real spell packets; this module owns no stock casting state.

use crate::net::NetHandlerApp;
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::{
    pfui_host::casts::{self, Cast},
    UiScript,
};
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

#[derive(Clone)]
struct Observed {
    info: Cast,
    start: Instant,
    end: Instant,
}

struct RemoteChannel {
    cast: Observed,
    awaiting_go: bool,
}

#[derive(Resource, Default)]
struct State {
    active: HashMap<u64, Observed>,
    /// Each caster's cast's unit target, as its start named it (`UnitSpellTargetName`).
    targets: HashMap<u64, u64>,
    remote_channels: HashMap<u64, RemoteChannel>,
    recent: HashMap<(u64, u32), (Observed, &'static str, Instant)>,
    edges: Vec<(u64, &'static str, Observed)>,
    serial: u64,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct CastFeed;

impl State {
    fn record(
        &mut self,
        guid: u64,
        spell: u32,
        duration: u32,
        channel: bool,
        now: Instant,
        spells: Option<&crate::ui_action::Spells>,
    ) -> Observed {
        self.serial = self.serial.wrapping_add(1);
        let display = spells.and_then(|spells| spells.catalog.get(spell));
        Observed {
            start: now,
            end: now + Duration::from_millis(u64::from(duration)),
            info: Cast {
                spell_id: spell,
                name: display.map(|s| s.name.clone()).unwrap_or_default(),
                rank: display.and_then(|s| s.rank.clone()).unwrap_or_default(),
                icon: display.and_then(|s| s.icon.clone()),
                start_ms: 0.0,
                end_ms: 0.0,
                tradeskill: display
                    .is_some_and(|s| s.attributes & benilla_formats::SPELL_ATTR_IS_TRADESKILL != 0),
                cast_guid: format!("Cast-{guid:016X}-{}", self.serial),
                delay_ms: 0,
                channel,
            },
        }
    }

    fn stop(&mut self, guid: u64, spell: u32, kind: &'static str) -> Option<Observed> {
        if !self
            .active
            .get(&guid)
            .is_some_and(|s| s.info.spell_id == spell)
        {
            return None;
        }
        let cast = self.active.remove(&guid)?;
        let kind = if cast.info.channel {
            "UNIT_SPELLCAST_CHANNEL_STOP"
        } else {
            kind
        };
        self.edges.push((guid, kind, cast.clone()));
        self.recent
            .insert((guid, spell), (cast.clone(), kind, Instant::now()));
        Some(cast)
    }
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<State>();
    app.net_handler(SessionEventKind::SpellStart, receive);
    app.net_handler(SessionEventKind::SpellGo, receive);
    app.net_handler(SessionEventKind::SpellFailedOther, receive);
    app.net_handler(SessionEventKind::CastResult, receive);
    app.net_handler(SessionEventKind::SpellDelayed, receive);
    app.net_handler(SessionEventKind::ChannelStart, receive);
    app.net_handler(SessionEventKind::ChannelUpdate, receive);
    app.net_handler(SessionEventKind::Disconnected, receive);
    app.net_handler(SessionEventKind::LoggedOut, receive);
    app.add_systems(
        Update,
        feed.in_set(super::InputSnapshots)
            .in_set(CastFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
}

fn receive(
    In(event): In<SessionEvent>,
    mut state: ResMut<State>,
    me: Res<crate::net::SelfGuid>,
    spells: Option<Res<crate::ui_action::Spells>>,
    objects: crate::net::Objects,
) {
    let now = Instant::now();
    match event {
        SessionEvent::SpellStart {
            caster,
            spell_id,
            cast_time_ms,
            target,
            ..
        } if caster != 0 => {
            state.remote_channels.remove(&caster);
            match target.filter(|target| *target != 0) {
                Some(target) => state.targets.insert(caster, target),
                None => state.targets.remove(&caster),
            };
            if cast_time_ms > 0 {
                if let Some(old) = state.active.get(&caster).cloned() {
                    state.stop(caster, old.info.spell_id, "UNIT_SPELLCAST_STOP");
                }
            }
            state.recent.remove(&(caster, spell_id));
            let regular = (cast_time_ms > 0).then(|| {
                state.record(
                    caster,
                    spell_id,
                    cast_time_ms,
                    false,
                    now,
                    spells.as_deref(),
                )
            });
            if me.0 != Some(caster) {
                if let Some((spells, def)) = spells
                    .as_deref()
                    .and_then(|spells| spells.catalog.get(spell_id).map(|def| (spells, def)))
                    .filter(|(_, def)| def.tooltip_channeled())
                {
                    // Observers receive no MSG_CHANNEL_START/UPDATE. A seen cast supplies the
                    // time anchor; UNIT_CHANNEL_SPELL must confirm that the channel is live.
                    let duration = duration(spells, def, objects.object(caster), None);
                    if duration > 0 {
                        let mut cast = state.record(
                            caster,
                            spell_id,
                            duration as u32,
                            true,
                            now,
                            Some(spells),
                        );
                        if let Some(regular) = &regular {
                            cast.info.cast_guid.clone_from(&regular.info.cast_guid);
                            cast.start += Duration::from_millis(u64::from(cast_time_ms));
                            cast.end += Duration::from_millis(u64::from(cast_time_ms));
                        }
                        state.remote_channels.insert(
                            caster,
                            RemoteChannel {
                                cast,
                                awaiting_go: cast_time_ms > 0,
                            },
                        );
                    }
                }
            }
            if let Some(record) = regular {
                state.active.insert(caster, record.clone());
                state.edges.push((caster, "UNIT_SPELLCAST_START", record));
            }
        }
        SessionEvent::SpellGo {
            caster, spell_id, ..
        } => {
            let current = state
                .active
                .get(&caster)
                .filter(|s| s.info.spell_id == spell_id)
                .cloned();
            let was_regular = current.as_ref().is_some_and(|s| !s.info.channel);
            if let Some(pending) = state
                .remote_channels
                .get_mut(&caster)
                .filter(|pending| pending.cast.info.spell_id == spell_id)
            {
                if pending.awaiting_go {
                    let span = pending.cast.end.duration_since(pending.cast.start);
                    pending.cast.start = now;
                    pending.cast.end = now + span;
                    pending.awaiting_go = false;
                }
            }
            let recent = state
                .recent
                .remove(&(caster, spell_id))
                .filter(|(_, kind, ended)| {
                    *kind == "UNIT_SPELLCAST_STOP"
                        && now.duration_since(*ended) < Duration::from_secs(2)
                })
                .map(|(cast, _, _)| cast);
            let pending = state
                .remote_channels
                .get(&caster)
                .filter(|pending| pending.cast.info.spell_id == spell_id)
                .map(|pending| pending.cast.clone());
            let record = current.or(recent).or(pending).unwrap_or_else(|| {
                state.record(caster, spell_id, 0, false, now, spells.as_deref())
            });
            if was_regular {
                state.active.remove(&caster);
            }
            state
                .edges
                .push((caster, "UNIT_SPELLCAST_SUCCEEDED", record.clone()));
            if was_regular {
                state.edges.push((caster, "UNIT_SPELLCAST_STOP", record));
            }
        }
        SessionEvent::SpellFailedOther { caster, spell_id } => {
            if state
                .remote_channels
                .get(&caster)
                .is_some_and(|pending| pending.cast.info.spell_id == spell_id)
            {
                state.remote_channels.remove(&caster);
            }
            state.stop(caster, spell_id, "UNIT_SPELLCAST_INTERRUPTED");
        }
        SessionEvent::CastResult {
            spell_id,
            success: false,
            ..
        } => {
            if let Some(guid) = me.0 {
                if state
                    .stop(guid, spell_id, "UNIT_SPELLCAST_INTERRUPTED")
                    .is_none()
                {
                    let recent = state.recent.get(&(guid, spell_id)).cloned();
                    if !recent
                        .as_ref()
                        .is_some_and(|(_, kind, _)| *kind == "UNIT_SPELLCAST_INTERRUPTED")
                    {
                        let record = recent.map(|(cast, _, _)| cast).unwrap_or_else(|| {
                            state.record(guid, spell_id, 0, false, now, spells.as_deref())
                        });
                        state.edges.push((guid, "UNIT_SPELLCAST_FAILED", record));
                    }
                }
            }
        }
        SessionEvent::SpellDelayed { caster, delay_ms } => {
            if let Some(record) = state.active.get_mut(&caster) {
                record.end += Duration::from_millis(u64::from(delay_ms));
                record.info.delay_ms = record.info.delay_ms.saturating_add(delay_ms);
                let record = record.clone();
                state.edges.push((caster, "UNIT_SPELLCAST_DELAYED", record));
            }
        }
        SessionEvent::ChannelStart {
            spell_id,
            duration_ms,
        } => {
            if let Some(guid) = me.0.filter(|_| duration_ms > 0) {
                let record =
                    state.record(guid, spell_id, duration_ms, true, now, spells.as_deref());
                state.active.insert(guid, record.clone());
                state
                    .edges
                    .push((guid, "UNIT_SPELLCAST_CHANNEL_START", record));
            }
        }
        SessionEvent::ChannelUpdate { remaining_ms } => {
            if let Some(guid) = me.0 {
                if let Some(record) = state.active.get_mut(&guid).filter(|s| s.info.channel) {
                    if remaining_ms == 0 {
                        let spell = record.info.spell_id;
                        state.stop(guid, spell, "UNIT_SPELLCAST_CHANNEL_STOP");
                    } else {
                        record.end = now + Duration::from_millis(u64::from(remaining_ms));
                        let record = record.clone();
                        state
                            .edges
                            .push((guid, "UNIT_SPELLCAST_CHANNEL_UPDATE", record));
                    }
                }
            }
        }
        SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => {
            state.active.clear();
            state.remote_channels.clear();
            state.edges.clear();
            state.recent.clear();
        }
        _ => {}
    }
}

fn project(record: &Observed, clock: &crate::ui_script::UiClock) -> Cast {
    let offset = |time: Instant| {
        let seconds = if time >= clock.anchor {
            time.duration_since(clock.anchor).as_secs_f64()
        } else {
            -clock.anchor.duration_since(time).as_secs_f64()
        };
        // ClassicAPI's millisecond timestamps share GetTime's epoch, not this frame's anchor.
        (clock.ui_now + seconds) * 1000.0
    };
    Cast {
        start_ms: offset(record.start),
        end_ms: offset(record.end),
        ..record.info.clone()
    }
}

fn feed(
    script: Option<NonSend<UiScript>>,
    mut state: ResMut<State>,
    clock: Res<crate::ui_script::UiClock>,
    me: Res<crate::net::SelfGuid>,
    channel: Res<crate::spell::ActiveChannel>,
    casting: Query<(&crate::net::Guid, &crate::creature_anim::Casting)>,
    objects: crate::net::Objects,
) {
    let Some(script) = script else {
        state.edges.clear();
        return;
    };
    let now = Instant::now();
    state
        .recent
        .retain(|_, (_, _, ended)| now.duration_since(*ended) < Duration::from_secs(2));
    let live: HashMap<_, _> = casting
        .iter()
        .map(|(guid, cast)| (guid.0, cast.spell_id))
        .collect();
    let remote_spell = |guid| objects.object(guid).map(|store| store.unit_channel_spell());
    let mut stopped = Vec::new();
    state.active.retain(|guid, record| {
        let alive = now < record.end
            && if record.info.channel {
                if me.0 == Some(*guid) {
                    channel.current(now) == Some(record.info.spell_id)
                } else {
                    remote_spell(*guid) == Some(record.info.spell_id)
                }
            } else {
                live.get(guid) == Some(&record.info.spell_id)
            };
        if !alive {
            stopped.push((
                *guid,
                if record.info.channel {
                    "UNIT_SPELLCAST_CHANNEL_STOP"
                } else if me.0 == Some(*guid) && now < record.end {
                    // Local cancellation clears Casting before its failure packet is processed.
                    "UNIT_SPELLCAST_INTERRUPTED"
                } else {
                    "UNIT_SPELLCAST_STOP"
                },
                record.clone(),
            ));
        }
        alive
    });
    for (guid, kind, record) in stopped {
        state
            .recent
            .insert((guid, record.info.spell_id), (record.clone(), kind, now));
        state.edges.push((guid, kind, record));
    }
    let mut started = Vec::new();
    state.remote_channels.retain(|guid, pending| {
        if now >= pending.cast.end {
            return false;
        }
        if !pending.awaiting_go && remote_spell(*guid) == Some(pending.cast.info.spell_id) {
            started.push((*guid, pending.cast.clone()));
            return false;
        }
        true
    });
    for (guid, record) in started {
        if let Some(old) = state.active.get(&guid).cloned() {
            state.stop(guid, old.info.spell_id, "UNIT_SPELLCAST_STOP");
        }
        state.active.insert(guid, record.clone());
        state
            .edges
            .push((guid, "UNIT_SPELLCAST_CHANNEL_START", record));
    }
    casts::set_casts(
        script.lua(),
        state
            .active
            .iter()
            .map(|(guid, record)| (*guid, project(record, &clock))),
    );
    let State {
        active, targets, ..
    } = &mut *state;
    targets.retain(|caster, _| active.contains_key(caster));
    casts::set_cast_targets(script.lua(), targets.iter().map(|(c, t)| (*c, *t)));
    for (guid, kind, record) in state.edges.drain(..) {
        casts::event(script.lua(), guid, kind, &project(&record, &clock));
    }
}

/// A spell's duration as the caster's level scales it (`SpellDuration.dbc`), with the caster's
/// talent modifiers when known: what an observer's channel bar and Nampower's queue time.
pub(super) fn duration(
    spells: &crate::ui_action::Spells,
    def: &benilla_formats::SpellDisplay,
    caster: Option<&benilla_protocol::ObjectFields>,
    mods: Option<&crate::spell::SpellModifiers>,
) -> i32 {
    let Some(row) = spells.durations.get(def.duration_index) else {
        return 0;
    };
    // Remote players expose their level, but never their talent modifiers.
    let level = def.skill_level(
        caster
            .and_then(|s| s.unit_level())
            .unwrap_or(0)
            .saturating_mul(5),
    );
    let span = row
        .base_ms
        .wrapping_add(
            (level as i32)
                .wrapping_sub(def.base_level as i32)
                .wrapping_mul(row.per_level_ms),
        )
        .min(row.max_ms);
    mods.map_or(span, |m| m.apply(def, 1, span))
}

#[cfg(all(test, feature = "nampower"))]
#[path = "cast_channel_tests.rs"]
mod channel_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_times_share_the_ui_epoch_and_late_failures_preserve_replacement_casts() {
        let anchor = Instant::now();
        let clock = crate::ui_script::UiClock {
            anchor,
            ui_now: 120.0,
        };
        let mut state = State::default();
        let first = state.record(3, 10, 2000, false, anchor + Duration::from_secs(1), None);
        state.active.insert(3, first);
        let replacement = state.record(3, 11, 3000, false, anchor + Duration::from_secs(2), None);
        state.active.insert(3, replacement.clone());
        assert!(state.stop(3, 10, "UNIT_SPELLCAST_INTERRUPTED").is_none());
        let shown = project(state.active.get(&3).unwrap(), &clock);
        assert_eq!((shown.start_ms, shown.end_ms), (122000.0, 125000.0));
        assert_eq!(shown.cast_guid, replacement.info.cast_guid);
        let next_clock = crate::ui_script::UiClock {
            anchor: anchor + Duration::from_secs(3),
            ui_now: 123.0,
        };
        let shown = project(&replacement, &next_clock);
        assert_eq!((shown.start_ms, shown.end_ms), (122000.0, 125000.0));
        assert!(shown.start_ms <= next_clock.ui_now * 1000.0);
        assert!(shown.end_ms > next_clock.ui_now * 1000.0);
        state.stop(3, 11, "UNIT_SPELLCAST_STOP");
        assert!(state.active.is_empty() && state.edges.len() == 1);
    }
}
