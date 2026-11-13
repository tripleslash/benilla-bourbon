//! Project the session's combat packets into the ClassicAPI combat log
//! ([`benilla_pfui::combat_log`]): melee swings, spell and periodic damage, heals, power gains and
//! drains, damage shields, the environment, misses, casts, interrupts, dispels, kills and
//! summons; and, from the units' own fields, deaths and their auras coming, going and stacking.
//! Each end is named with the `COMBATLOG_OBJECT_*` flags a ClassicAPI client gives it.

use crate::net::NetHandlerApp;
use benilla_pfui::combat_log::{self as log, Event, Unit};
use benilla_protocol::messages::{ExecuteLog, PeriodicTick};
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::{ScriptValue as V, UiScript};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Resource, Default)]
struct Events(Vec<SessionEvent>);

/// What the units looked like at the last frame: alive or not, and their auras by slot as
/// `(spell, stacks)`, so the next frame's differences are events.
#[derive(Default)]
struct Seen {
    alive: HashMap<u64, bool>,
    /// Each unit's auras as `(slot, spell, stacks)`, in the descriptor's slot order.
    auras: HashMap<u64, Vec<(u8, u32, u8)>>,
    /// The last cast each caster began with a cast time: its go then raises no `SPELL_CAST_SUCCESS`.
    timed_cast: HashMap<u64, u32>,
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<Events>();
    app.net_handler(SessionEventKind::AttackerState, receive);
    app.net_handler(SessionEventKind::SpellDamageLog, receive);
    app.net_handler(SessionEventKind::PeriodicAuraLog, receive);
    app.net_handler(SessionEventKind::SpellHealLog, receive);
    app.net_handler(SessionEventKind::SpellEnergizeLog, receive);
    app.net_handler(SessionEventKind::DamageShield, receive);
    app.net_handler(SessionEventKind::EnvironmentalDamageLog, receive);
    app.net_handler(SessionEventKind::SpellLogMiss, receive);
    app.net_handler(SessionEventKind::PartyKillLog, receive);
    app.net_handler(SessionEventKind::SpellInstaKillLog, receive);
    app.net_handler(SessionEventKind::ProcResist, receive);
    app.net_handler(SessionEventKind::SpellOrDamageImmune, receive);
    app.net_handler(SessionEventKind::SpellDispelLog, receive);
    app.net_handler(SessionEventKind::DispelFailed, receive);
    app.net_handler(SessionEventKind::SpellLogExecute, receive);
    app.net_handler(SessionEventKind::SpellStart, receive);
    app.net_handler(SessionEventKind::SpellGo, receive);
    app.net_handler(SessionEventKind::CastResult, receive);
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
        | SessionEvent::SpellDamageLog(_)
        | SessionEvent::PeriodicAuraLog(_)
        | SessionEvent::SpellHealLog(_)
        | SessionEvent::SpellEnergizeLog(_)
        | SessionEvent::DamageShield(_)
        | SessionEvent::EnvironmentalDamageLog(_)
        | SessionEvent::SpellLogMiss(_)
        | SessionEvent::PartyKillLog(_)
        | SessionEvent::SpellInstaKillLog(_)
        | SessionEvent::ProcResist(_)
        | SessionEvent::SpellOrDamageImmune(_)
        | SessionEvent::SpellDispelLog(_)
        | SessionEvent::DispelFailed(_)
        | SessionEvent::SpellLogExecute(_)
        | SessionEvent::SpellStart { .. }
        | SessionEvent::SpellGo { .. }
        | SessionEvent::CastResult { .. }
        | SessionEvent::Disconnected { .. }
        | SessionEvent::LoggedOut => events.0.push(event),
        _ => (),
    }
}

/// Everything an event's ends and spells are described with.
#[derive(SystemParam)]
struct Describe<'w, 's> {
    objects: crate::net::Objects<'w, 's>,
    names: Res<'w, crate::names::NameCache>,
    group: Res<'w, crate::ui_party::GroupState>,
    relations: crate::target::RelationWorld<'w, 's>,
    selection: Res<'w, crate::target::Selection>,
    me: Res<'w, crate::net::SelfGuid>,
    spells: Option<Res<'w, crate::ui_action::Spells>>,
}

/// `HIGHGUID_GAMEOBJECT` and `HIGHGUID_PET`, the guids' top halves.
const HIGH_GAMEOBJECT: u64 = 0xF110;
const HIGH_PET: u64 = 0xF140;

impl Describe<'_, '_> {
    fn me(&self) -> u64 {
        self.me.0.unwrap_or(0)
    }

    fn is_player(guid: u64) -> bool {
        guid >> 48 == 0
    }

    fn owner(&self, guid: u64) -> Option<u64> {
        let fields = self.objects.object(guid)?;
        [
            fields.unit_charmed_by(),
            fields.unit_summoned_by(),
            fields.unit_created_by(),
        ]
        .into_iter()
        .flatten()
        .find(|g| *g != 0)
    }

    fn in_party(&self, guid: u64) -> bool {
        self.group.party_slots().any(|m| m.guid == guid)
    }

    fn in_raid(&self, guid: u64) -> bool {
        self.group.raid_rows().any(|g| g == guid)
    }

    fn flags(&self, guid: u64, focus: Option<u64>) -> u32 {
        let me = self.me();
        let owner = self.owner(guid);
        let high = guid >> 48;
        let player_owned = owner.is_some_and(Self::is_player);
        let kind = if Self::is_player(guid) {
            log::TYPE_PLAYER
        } else if high == HIGH_GAMEOBJECT {
            log::TYPE_OBJECT
        } else if high == HIGH_PET
            || (player_owned
                && self
                    .objects
                    .object(guid)
                    .and_then(|f| f.unit_summoned_by())
                    .is_some_and(|g| g != 0))
        {
            log::TYPE_PET
        } else if owner.is_some() {
            log::TYPE_GUARDIAN
        } else {
            log::TYPE_NPC
        };
        let control = if Self::is_player(guid) || player_owned {
            log::CONTROL_PLAYER
        } else {
            log::CONTROL_NPC
        };
        let anchor = if player_owned {
            owner.unwrap_or(guid)
        } else {
            guid
        };
        let affiliation = if anchor == me {
            log::AFFILIATION_MINE
        } else if self.in_party(anchor) {
            log::AFFILIATION_PARTY
        } else if self.in_raid(anchor) {
            log::AFFILIATION_RAID
        } else {
            log::AFFILIATION_OUTSIDER
        };
        let reaction = match self.reaction(guid) {
            Some(rank) if rank <= 2 => log::REACTION_HOSTILE,
            Some(3) => log::REACTION_NEUTRAL,
            _ => log::REACTION_FRIENDLY,
        };
        let mut flags = kind | control | affiliation | reaction;
        if self.selection.guid == Some(guid) {
            flags |= log::TARGET;
        }
        if focus == Some(guid) {
            flags |= log::FOCUS;
        }
        flags
    }

    /// The unit's reaction toward the player, as `UnitReaction(unit, "player")` ranks it.
    fn reaction(&self, guid: u64) -> Option<u8> {
        let store = self.objects.store(guid)?;
        let unit = self.relations.unit_of(guid, store);
        let me = self.relations.me()?;
        Some(self.relations.get().reaction(&unit, &me))
    }

    fn unit(&self, guid: u64, focus: Option<u64>) -> Option<Unit> {
        (guid != 0).then(|| {
            let store = self.objects.store(guid);
            let name = crate::ui_chat::combat::unit_name(
                guid,
                store,
                &self.names,
                crate::query_cache::Ask::Read,
            );
            let mark = self.group.raid_target_index(guid);
            Unit {
                guid,
                name,
                flags: self.flags(guid, focus),
                raid_flags: if (1..=8).contains(&mark) {
                    1 << (mark - 1)
                } else {
                    0
                },
            }
        })
    }

    /// A spell's `spellId, spellName, spellSchool`, the school as a mask.
    fn spell(&self, id: u32) -> [V; 3] {
        let def = self.spells.as_ref().and_then(|s| s.catalog.get(id));
        [
            V::Int(i64::from(id)),
            def.map_or(V::Nil, |d| V::Str(d.name.clone())),
            V::Int(def.map_or(1, |d| 1 << d.school.min(6))),
        ]
    }

    fn spell_name(&self, id: u32) -> Option<String> {
        self.spells
            .as_ref()
            .and_then(|s| s.catalog.get(id))
            .map(|d| d.name.clone())
    }

    /// Auto Shot, Shoot and Throw log as `RANGE_*`.
    fn ranged(&self, id: u32) -> bool {
        matches!(id, 75 | 5019 | 2764 | 3018)
    }

    fn health(&self, guid: u64) -> Option<(u32, u32)> {
        let f = self.objects.object(guid)?;
        Some((f.unit_health()?, f.unit_max_health()?))
    }
}

fn int(n: impl Into<i64>) -> V {
    V::Int(n.into())
}

/// `1` or nil, as a flag reads in the log.
fn flag(on: bool) -> V {
    if on {
        V::Int(1)
    } else {
        V::Nil
    }
}

/// `nil` for 0, as a damage suffix's resisted, blocked and absorbed read.
fn nonzero(n: i64) -> V {
    if n == 0 {
        V::Nil
    } else {
        V::Int(n)
    }
}

const MISS_TYPES: [&str; 12] = [
    "", "MISS", "RESIST", "DODGE", "PARRY", "BLOCK", "EVADE", "IMMUNE", "IMMUNE", "DEFLECT",
    "ABSORB", "REFLECT",
];

struct Emit<'a, 'w, 's> {
    lua: &'a benilla_ui::mlua::Lua,
    d: &'a Describe<'w, 's>,
    focus: Option<u64>,
}

impl Emit<'_, '_, '_> {
    fn push(
        &self,
        subevent: &'static str,
        source: u64,
        dest: u64,
        spell: Option<u32>,
        values: Vec<V>,
    ) {
        log::record(
            self.lua,
            Event {
                subevent,
                source: self.d.unit(source, self.focus),
                dest: self.d.unit(dest, self.focus),
                spell: spell.map(|id| (id, self.d.spell_name(id))),
                values,
            },
        );
    }

    fn damage(
        &self,
        prefix: Vec<V>,
        dest: u64,
        amount: u32,
        school: u32,
        details: [i64; 3],
        crit: bool,
        glancing: bool,
        crushing: bool,
        off_hand: bool,
    ) -> Vec<V> {
        let overkill = self
            .d
            .health(dest)
            .map_or(0, |(hp, _)| i64::from(amount.saturating_sub(hp)));
        let mut values = prefix;
        values.extend([
            int(amount),
            V::Int(overkill),
            int(school),
            nonzero(details[0]),
            nonzero(details[1]),
            nonzero(details[2]),
            flag(crit),
            flag(glancing),
            flag(crushing),
            flag(off_hand),
        ]);
        values
    }
}

fn feed(
    script: Option<NonSend<UiScript>>,
    mut events: ResMut<Events>,
    d: Describe,
    mut seen: Local<crate::ui_script::VmMemo<Seen>>,
) {
    let Some(script) = script else {
        events.0.clear();
        return;
    };
    let seen = seen.get(&script);
    let lua = script.lua();
    let focus = benilla_ui::script::pfui_host::units::custom_guid(lua, "focus").flatten();
    let e = Emit { lua, d: &d, focus };
    let me = d.me();
    for event in events.0.drain(..) {
        match event {
            SessionEvent::AttackerState(a) => {
                const MISS: u32 = 0x10;
                const CRIT: u32 = 0x80;
                const LEFT: u32 = 0x4;
                const GLANCING: u32 = 0x4000;
                const CRUSHING: u32 = 0x8000;
                let off_hand = a.hit_info & LEFT != 0;
                let miss = if a.hit_info & MISS != 0 {
                    Some("MISS")
                } else {
                    match a.victim_state {
                        2 => Some("DODGE"),
                        3 => Some("PARRY"),
                        6 => Some("EVADE"),
                        7 => Some("IMMUNE"),
                        8 => Some("DEFLECT"),
                        5 if a.damage == 0 => Some("BLOCK"),
                        _ if a.damage == 0 && a.absorb > 0 => Some("ABSORB"),
                        _ if a.damage == 0 && a.resist > 0 => Some("RESIST"),
                        _ => None,
                    }
                };
                match miss {
                    Some(kind) => {
                        let mut values = vec![V::Str(kind.into()), flag(off_hand)];
                        match kind {
                            "ABSORB" => values.push(int(a.absorb)),
                            "RESIST" => values.push(int(a.resist.max(0))),
                            "BLOCK" => values.push(int(a.blocked)),
                            _ => {}
                        }
                        e.push("SWING_MISSED", a.attacker, a.victim, None, values);
                    }
                    None => {
                        let values = e.damage(
                            Vec::new(),
                            a.victim,
                            a.damage,
                            1 << a.school.min(6),
                            [
                                i64::from(a.resist.max(0)),
                                i64::from(a.blocked),
                                i64::from(a.absorb),
                            ],
                            a.hit_info & CRIT != 0,
                            a.hit_info & GLANCING != 0,
                            a.hit_info & CRUSHING != 0,
                            off_hand,
                        );
                        e.push("SWING_DAMAGE", a.attacker, a.victim, None, values);
                    }
                }
            }
            SessionEvent::SpellDamageLog(s) => {
                let subevent = if d.ranged(s.spell_id) {
                    "RANGE_DAMAGE"
                } else if s.periodic {
                    "SPELL_PERIODIC_DAMAGE"
                } else {
                    "SPELL_DAMAGE"
                };
                let values = e.damage(
                    d.spell(s.spell_id).to_vec(),
                    s.target,
                    s.damage,
                    1 << u32::from(s.school).min(6),
                    [
                        i64::from(s.resist.max(0)),
                        i64::from(s.blocked),
                        i64::from(s.absorb),
                    ],
                    s.hit_info & 0x2 != 0,
                    false,
                    false,
                    false,
                );
                e.push(subevent, s.attacker, s.target, Some(s.spell_id), values);
            }
            SessionEvent::PeriodicAuraLog(p) => {
                for tick in p.ticks {
                    let prefix = d.spell(p.spell_id).to_vec();
                    match tick {
                        PeriodicTick::Damage {
                            amount,
                            school,
                            absorb,
                            resist,
                        } => {
                            let values = e.damage(
                                prefix,
                                p.target,
                                amount,
                                1 << school.min(6),
                                [i64::from(resist.max(0)), 0, i64::from(absorb)],
                                false,
                                false,
                                false,
                                false,
                            );
                            e.push(
                                "SPELL_PERIODIC_DAMAGE",
                                p.caster,
                                p.target,
                                Some(p.spell_id),
                                values,
                            );
                        }
                        PeriodicTick::Heal { amount } => {
                            let over = overheal(&d, p.target, amount);
                            let mut values = prefix;
                            values.extend([int(amount), V::Int(over), int(0), flag(false)]);
                            e.push(
                                "SPELL_PERIODIC_HEAL",
                                p.caster,
                                p.target,
                                Some(p.spell_id),
                                values,
                            );
                        }
                        PeriodicTick::Energize { power, amount } => {
                            let mut values = prefix;
                            values.extend([int(amount), int(power), V::Nil]);
                            e.push(
                                "SPELL_PERIODIC_ENERGIZE",
                                p.caster,
                                p.target,
                                Some(p.spell_id),
                                values,
                            );
                        }
                        PeriodicTick::ManaLeech {
                            power,
                            amount,
                            multiplier,
                        } => {
                            let mut values = prefix;
                            let extra = (amount as f32 * multiplier) as i64;
                            values.extend([int(amount), int(power), V::Int(extra), V::Nil]);
                            let subevent = if multiplier > 0.0 {
                                "SPELL_PERIODIC_LEECH"
                            } else {
                                "SPELL_PERIODIC_DRAIN"
                            };
                            e.push(subevent, p.caster, p.target, Some(p.spell_id), values);
                        }
                    }
                }
            }
            SessionEvent::SpellHealLog(h) => {
                let over = overheal(&d, h.target, h.amount);
                let mut values = d.spell(h.spell_id).to_vec();
                values.extend([int(h.amount), V::Int(over), int(0), flag(h.critical)]);
                e.push("SPELL_HEAL", h.healer, h.target, Some(h.spell_id), values);
            }
            SessionEvent::SpellEnergizeLog(s) => {
                let mut values = d.spell(s.spell_id).to_vec();
                values.extend([int(s.amount), int(s.power), V::Nil]);
                e.push(
                    "SPELL_ENERGIZE",
                    s.caster,
                    s.target,
                    Some(s.spell_id),
                    values,
                );
            }
            SessionEvent::DamageShield(s) => {
                let school = 1 << s.school.min(6);
                let values = e.damage(
                    vec![int(0), V::Nil, int(school)],
                    s.attacker,
                    s.damage,
                    school,
                    [0, 0, 0],
                    false,
                    false,
                    false,
                    false,
                );
                // The shield's owner is the one struck, and its damage goes back to the striker.
                e.push("DAMAGE_SHIELD", s.victim, s.attacker, None, values);
            }
            SessionEvent::EnvironmentalDamageLog(s) => {
                const TYPES: [&str; 6] =
                    ["FATIGUE", "DROWNING", "FALLING", "LAVA", "SLIME", "FIRE"];
                let kind = TYPES
                    .get(usize::from(s.damage_type))
                    .copied()
                    .unwrap_or("FIRE");
                let values = e.damage(
                    vec![V::Str(kind.into())],
                    s.victim,
                    s.damage,
                    1,
                    [i64::from(s.resist.max(0)), 0, i64::from(s.absorb)],
                    false,
                    false,
                    false,
                    false,
                );
                e.push("ENVIRONMENTAL_DAMAGE", 0, s.victim, None, values);
            }
            SessionEvent::SpellLogMiss(m) => {
                for (target, reason) in m.misses {
                    let kind = MISS_TYPES
                        .get(usize::from(reason))
                        .copied()
                        .unwrap_or("MISS");
                    if kind.is_empty() {
                        continue;
                    }
                    let subevent = if d.ranged(m.spell_id) {
                        "RANGE_MISSED"
                    } else {
                        "SPELL_MISSED"
                    };
                    let mut values = d.spell(m.spell_id).to_vec();
                    values.extend([V::Str(kind.into()), flag(false)]);
                    e.push(subevent, m.caster, target, Some(m.spell_id), values);
                }
            }
            SessionEvent::ProcResist(o) | SessionEvent::SpellOrDamageImmune(o) => {
                let kind = if o.log_format == 0 {
                    "RESIST"
                } else {
                    "IMMUNE"
                };
                let mut values = d.spell(o.spell_id).to_vec();
                values.extend([V::Str(kind.into()), flag(false)]);
                e.push("SPELL_MISSED", o.caster, o.target, Some(o.spell_id), values);
            }
            SessionEvent::PartyKillLog(k) => {
                e.push("PARTY_KILL", k.killer, k.victim, None, Vec::new());
            }
            SessionEvent::SpellInstaKillLog(k) => {
                e.push(
                    "SPELL_INSTAKILL",
                    0,
                    k.victim,
                    Some(k.spell_id),
                    d.spell(k.spell_id).to_vec(),
                );
            }
            SessionEvent::SpellDispelLog(s) => {
                for removed in s.spell_ids {
                    let mut values = vec![int(0), V::Nil, int(1)];
                    values.extend(d.spell(removed));
                    let aura = if d.reaction(s.caster).is_some_and(|r| r <= 2)
                        == d.reaction(s.victim).is_some_and(|r| r <= 2)
                    {
                        "DEBUFF"
                    } else {
                        "BUFF"
                    };
                    values.push(V::Str(aura.into()));
                    e.push("SPELL_DISPEL", s.caster, s.victim, Some(removed), values);
                }
            }
            SessionEvent::DispelFailed(s) => {
                for kept in s.spell_ids {
                    let mut values = vec![int(0), V::Nil, int(1)];
                    values.extend(d.spell(kept));
                    e.push(
                        "SPELL_DISPEL_FAILED",
                        s.caster,
                        s.victim,
                        Some(kept),
                        values,
                    );
                }
            }
            SessionEvent::SpellLogExecute(x) => {
                for (effect, logs) in x.effects {
                    for entry in logs {
                        execute(&e, &d, x.caster, x.spell_id, effect, entry);
                    }
                }
            }
            SessionEvent::SpellStart {
                caster,
                spell_id,
                cast_time_ms,
                ..
            } => {
                if cast_time_ms > 0 {
                    seen.timed_cast.insert(caster, spell_id);
                    e.push(
                        "SPELL_CAST_START",
                        caster,
                        0,
                        Some(spell_id),
                        d.spell(spell_id).to_vec(),
                    );
                }
            }
            SessionEvent::SpellGo {
                caster,
                spell_id,
                target,
                ..
            } => {
                if seen.timed_cast.get(&caster) == Some(&spell_id) {
                    seen.timed_cast.remove(&caster);
                } else {
                    let dest = target.unwrap_or(0);
                    e.push(
                        "SPELL_CAST_SUCCESS",
                        caster,
                        dest,
                        Some(spell_id),
                        d.spell(spell_id).to_vec(),
                    );
                }
            }
            SessionEvent::CastResult {
                spell_id,
                success: false,
                reason,
                ..
            } => {
                let mut values = d.spell(spell_id).to_vec();
                values.push(V::Str(cast_failed_text(&script, reason)));
                e.push("SPELL_CAST_FAILED", me, 0, Some(spell_id), values);
            }
            SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => {
                log::clear(lua);
                *seen = Seen::default();
            }
            _ => {}
        }
    }
    deaths_and_auras(&e, &d, seen);
}

/// The part of a heal above the health the target was missing.
fn overheal(d: &Describe, target: u64, amount: u32) -> i64 {
    d.health(target).map_or(0, |(hp, max)| {
        i64::from(amount.saturating_sub(max.saturating_sub(hp)))
    })
}

/// The error text a failed cast of ours showed: the `SPELL_FAILED_*` string of the reason code.
fn cast_failed_text(script: &UiScript, reason: Option<u8>) -> String {
    reason
        .and_then(|r| crate::ui_action::CAST_FAIL_KEYS.get(usize::from(r)))
        .and_then(|key| script.lua().globals().get::<String>(*key).ok())
        .unwrap_or_default()
}

/// `SPELL_EFFECT_*` ids whose execute log names the event.
const RESURRECT: [u32; 2] = [18, 113];
const SUMMON: [u32; 12] = [28, 41, 42, 56, 73, 74, 87, 88, 89, 90, 97, 112];
const CREATE: [u32; 6] = [50, 76, 104, 105, 106, 107];

fn execute(e: &Emit, d: &Describe, caster: u64, spell: u32, effect: u32, entry: ExecuteLog) {
    let prefix = d.spell(spell).to_vec();
    match entry {
        ExecuteLog::PowerDrain {
            target,
            amount,
            power,
            multiplier,
        } => {
            let mut values = prefix;
            values.extend([
                int(amount),
                int(power),
                V::Int((amount as f32 * multiplier) as i64),
                V::Nil,
            ]);
            let subevent = if multiplier > 0.0 {
                "SPELL_LEECH"
            } else {
                "SPELL_DRAIN"
            };
            e.push(subevent, caster, target, Some(spell), values);
        }
        ExecuteLog::ExtraAttacks { target, count } => {
            let mut values = prefix;
            values.push(int(count));
            e.push("SPELL_EXTRA_ATTACKS", caster, target, Some(spell), values);
        }
        ExecuteLog::InterruptCast { target, spell_id } => {
            let mut values = prefix;
            values.extend(d.spell(spell_id));
            e.push("SPELL_INTERRUPT", caster, target, Some(spell), values);
        }
        ExecuteLog::DurabilityDamage {
            target,
            item_entry,
            slot,
        } => {
            if slot < 0 {
                e.push(
                    "SPELL_DURABILITY_DAMAGE_ALL",
                    caster,
                    target,
                    Some(spell),
                    prefix,
                );
            } else {
                let mut values = prefix;
                values.extend([V::Int(i64::from(item_entry)), V::Nil]);
                e.push(
                    "SPELL_DURABILITY_DAMAGE",
                    caster,
                    target,
                    Some(spell),
                    values,
                );
            }
        }
        ExecuteLog::Target { target } => {
            let subevent = if RESURRECT.contains(&effect) {
                "SPELL_RESURRECT"
            } else if SUMMON.contains(&effect) {
                "SPELL_SUMMON"
            } else if CREATE.contains(&effect) {
                "SPELL_CREATE"
            } else {
                return;
            };
            e.push(subevent, caster, target, Some(spell), prefix);
        }
        // Heals and power gains have their own logs; the rest names no combat event.
        ExecuteLog::Heal { .. }
        | ExecuteLog::Energize { .. }
        | ExecuteLog::CreateItem { .. }
        | ExecuteLog::FeedPet { .. } => {}
    }
}

/// One unit's auras, `(slot, spell, stacks)` in slot order.
type Auras = [(u8, u32, u8)];

/// What changed between two reads of a unit's auras, as `(subevent, slot, spell, stacks)`: a slot
/// holding a new spell applied it, the same spell with another stack count changed its dose (the
/// only two that carry the stacks), and a slot that lost its spell removed it.
fn aura_changes(before: &Auras, now: &Auras) -> Vec<(&'static str, u8, u32, Option<u8>)> {
    let at = |list: &Auras, slot: u8| {
        list.iter()
            .find(|(s, ..)| *s == slot)
            .map(|&(_, spell, stacks)| (spell, stacks))
    };
    let mut out = Vec::new();
    for &(slot, spell, stacks) in now {
        match at(before, slot) {
            Some((old, old_stacks)) if old == spell => {
                if stacks > old_stacks {
                    out.push(("SPELL_AURA_APPLIED_DOSE", slot, spell, Some(stacks)));
                } else if stacks < old_stacks {
                    out.push(("SPELL_AURA_REMOVED_DOSE", slot, spell, Some(stacks)));
                }
            }
            _ => out.push(("SPELL_AURA_APPLIED", slot, spell, None)),
        }
    }
    for &(slot, spell, _) in before {
        if at(now, slot).is_none_or(|(s, _)| s != spell) {
            out.push(("SPELL_AURA_REMOVED", slot, spell, None));
        }
    }
    out
}

/// The frame's deaths (a unit whose health fell to 0: `UNIT_DIED`, or `UNIT_DESTROYED` for one a
/// player summoned) and aura changes by slot (`SPELL_AURA_APPLIED` / `_REMOVED` /
/// `_APPLIED_DOSE` / `_REMOVED_DOSE`, the caster unknown: no packet names it).
fn deaths_and_auras(e: &Emit, d: &Describe, seen: &mut Seen) {
    let mut present = HashSet::new();
    for (guid, _, fields) in d.objects.iter() {
        if guid >> 48 == HIGH_GAMEOBJECT || fields.unit_max_health().is_none_or(|m| m == 0) {
            continue;
        }
        present.insert(guid);
        let alive = fields.unit_health().is_some_and(|h| h > 0);
        if let Some(was) = seen.alive.insert(guid, alive) {
            if was && !alive {
                let summoned = d.owner(guid).is_some_and(Describe::is_player);
                let subevent = if summoned && !Describe::is_player(guid) {
                    "UNIT_DESTROYED"
                } else {
                    "UNIT_DIED"
                };
                e.push(subevent, 0, guid, None, Vec::new());
            }
        }
        let now: Vec<(u8, u32, u8)> = fields
            .unit_auras()
            .filter(|a| a.spell_id != 0)
            .map(|a| (a.slot, a.spell_id, a.stacks))
            .collect();
        let before = seen.auras.get(&guid);
        // Nearly every unit on nearly every frame: nothing moved, nothing to say.
        if before == Some(&now) {
            continue;
        }
        for (subevent, slot, spell, dose) in before.map_or(Vec::new(), |b| aura_changes(b, &now)) {
            let mut values = d.spell(spell).to_vec();
            values.push(V::Str(if slot < 32 { "BUFF" } else { "DEBUFF" }.into()));
            values.extend(dose.map(int));
            e.push(subevent, 0, guid, Some(spell), values);
        }
        seen.auras.insert(guid, now);
    }
    seen.alive.retain(|g, _| present.contains(g));
    seen.auras.retain(|g, _| present.contains(g));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slot's new spell applies, its lost one is removed, the same spell with more or fewer
    /// stacks changes its dose, and a spell replaced in its slot is both; nothing else speaks.
    #[test]
    fn aura_changes_name_what_moved_by_slot() {
        let before = [(0, 1126, 1), (1, 21562, 1), (32, 589, 2), (33, 172, 1)];
        assert!(aura_changes(&before, &before).is_empty());
        let now = [(0, 1126, 1), (2, 467, 1), (32, 589, 3), (33, 980, 1)];
        assert_eq!(
            aura_changes(&before, &now),
            [
                ("SPELL_AURA_APPLIED", 2, 467, None),
                ("SPELL_AURA_APPLIED_DOSE", 32, 589, Some(3)),
                ("SPELL_AURA_APPLIED", 33, 980, None),
                ("SPELL_AURA_REMOVED", 1, 21562, None),
                ("SPELL_AURA_REMOVED", 33, 172, None),
            ]
        );
        assert_eq!(
            aura_changes(&now, &before)[1],
            ("SPELL_AURA_REMOVED_DOSE", 32, 589, Some(2))
        );
        assert_eq!(
            aura_changes(&[], &[(0, 1126, 1)]),
            [("SPELL_AURA_APPLIED", 0, 1126, None)]
        );
    }
}
