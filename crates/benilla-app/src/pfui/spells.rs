//! Reuse the stock algorithms for arbitrary DBC spells, independent of action-bar placement.

use benilla_formats::{RangeTargets, SpellDisplay, SpellRange};
use benilla_pfui::spell::runtime::{self, Cost};
use benilla_ui::script::UiScript;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::{collections::HashMap, sync::Arc};

struct RangeSpell {
    definition: SpellDisplay,
    row: Option<SpellRange>,
}

#[derive(Default)]
struct Memo {
    costs: Option<(u32, u32, u32)>,
    /// The cooldown list's flush counters when the rows were last built.
    cooldowns: Option<(u64, u64, u64)>,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct SpellSnapshots;

pub(super) fn register(app: &mut App) {
    app.configure_sets(
        Update,
        SpellSnapshots
            .in_set(crate::ui_script::UiFeed)
            .after(crate::ui_unit::UnitFeed),
    );
    app.add_systems(Update, feed.in_set(SpellSnapshots));
}

#[allow(clippy::type_complexity)] // Bevy's per-frame spell inputs
fn feed(
    script: Option<NonSend<UiScript>>,
    spells: Option<Res<crate::ui_action::Spells>>,
    actions: Res<crate::ui_action::PlayerActions>,
    cooldowns: Res<crate::spell::Cooldowns>,
    mods: Res<crate::spell::SpellModifiers>,
    clock: Res<crate::ui_script::UiClock>,
    self_q: Query<(&crate::net::ObjectStore, &Transform), With<crate::net::SelfPlayer>>,
    selection: Res<crate::target::Selection>,
    objects: crate::net::Objects,
    positions: Query<(
        &crate::net::Guid,
        Entity,
        &Transform,
        &crate::net::ObjectStore,
    )>,
    range_units: crate::spell::RangeUnits,
    relations: crate::target::RelationWorld,
    // The item cache; `[0xb4b3e4]`, which the usable walk reads; the skill lines the cost finds
    // its level through; `ItemSubClass.dbc`, whose flags the ammo leg reads.
    supplies: (
        Res<crate::items::Items>,
        Option<Res<crate::player::OwnControlLost>>,
        Option<Res<crate::ui_spellbook::SkillLines>>,
        Option<Res<crate::ui_items::ItemSubClasses>>,
    ),
    mut memo: Local<crate::ui_script::VmMemo<Memo>>,
    mut ranges: Local<crate::ui_script::VmMemo<Option<Arc<HashMap<u32, RangeSpell>>>>>,
) {
    let (Some(script), Some(spells)) = (script, spells) else {
        return;
    };
    // A VM without the extension (a stock interface, pfUI off) has no reader for any of this.
    if !runtime::installed(script.lua()) {
        return;
    }
    let memo = memo.get(&script);
    let ranges = ranges.get(&script);
    let now = Instant::now();
    let me = self_q.single().ok();
    let basis = me.map(|(store, _)| {
        (
            store.0.unit_level().unwrap_or(0),
            store.0.unit_base_mana().unwrap_or(0),
            store.0.unit_base_health().unwrap_or(0),
        )
    });
    if mods.is_changed() || spells.is_changed() || basis != memo.costs {
        let costs = me
            .map(|(store, _)| {
                spells
                    .catalog
                    .iter()
                    .map(|(id, d)| {
                        (
                            id,
                            Cost {
                                kind: d.power_type as i32,
                                amount: u32::try_from(crate::spell::usable::power_cost(
                                    d,
                                    store,
                                    crate::spell::PlayerSkills {
                                        lines: supplies.2.as_deref().map(|s| &s.catalog),
                                        spells: Some(&spells.catalog),
                                        items: Some(&supplies.0),
                                    },
                                    &mods,
                                ))
                                .unwrap_or(0),
                                percent: d.mana_cost_pct,
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        runtime::set_costs(script.lua(), costs);
        memo.costs = basis;
    }
    // The getter reads the list as of its call wherever a host is lent, so these rows are its
    // list of catalog spells and its answer without a host. They are rebuilt when the list
    // flushes, as the stock interface hears of a change (`0x4b31b0`, `0x4b31c0`, `0x4f93d0`), and
    // never for time alone: the table holds a row per catalog spell. The resource's change tick
    // is no test for a flush: in a live session the list reads as changed on every frame.
    let flushed = (
        cooldowns.generation,
        cooldowns.usable_refreshes,
        cooldowns.bag_flushes,
    );
    if memo.cooldowns != Some(flushed) {
        runtime::set_cooldowns(
            script.lua(),
            spells
                .catalog
                .iter()
                .map(|(id, d)| {
                    (
                        id,
                        cooldowns
                            .info(id, 0, Some(d), now)
                            .ui_triple(clock.anchor, clock.ui_now)
                            .unwrap_or((0, 0, true)),
                    )
                })
                .collect(),
        );
        memo.cooldowns = Some(flushed);
    }
    let carried = me
        .map(|(store, _)| crate::ui_items::stack_counts(&store.0, &objects))
        .unwrap_or_default();
    let (items, own_control, skill_lines, sub_classes) = &supplies;
    let usable = me
        .map(|(store, _)| {
            let context = crate::spell::usable::UsableCtx {
                store,
                target: selection.guid.zip(
                    selection
                        .guid
                        .and_then(|guid| objects.entity(guid))
                        .and_then(|entity| positions.get(entity).ok())
                        .map(|(_, _, _, store)| store),
                ),
                rel: relations.get(),
                cooldowns: &cooldowns,
                spell_mods: &mods,
                control_lost: own_control.as_ref().is_some_and(|c| c.0),
                stacks: &carried,
                skill_lines: skill_lines.as_deref().map(|s| &s.catalog),
                sub_classes: sub_classes.as_deref().map(|s| &s.0),
            };
            actions
                .spells
                .iter()
                .filter_map(|id| {
                    let d = spells.catalog.get(*id)?;
                    Some((
                        *id,
                        crate::spell::usable::spell_usable(
                            *id, d, &spells, &context, &objects, items,
                        ),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    runtime::set_usable(script.lua(), usable);
    if ranges.is_none() || spells.is_changed() {
        // These are exactly the fields min_max_range and the range SpellMod read. Keeping
        // this immutable projection avoids cloning the complete catalog every frame.
        *ranges = Some(Arc::new(
            spells
                .catalog
                .iter()
                .map(|(id, d)| {
                    (
                        id,
                        RangeSpell {
                            row: spells.ranges.get(d.range_index).copied(),
                            definition: SpellDisplay {
                                attributes: d.attributes,
                                attributes_ex3: d.attributes_ex3,
                                spell_family: d.spell_family,
                                spell_family_flags: d.spell_family_flags,
                                ..Default::default()
                            },
                        },
                    )
                })
                .collect(),
        ));
    }
    let query = me.map(|(_, transform)| {
        let caster = range_units.caster();
        let origin = transform.translation;
        let units: HashMap<_, _> = positions
            .iter()
            .filter_map(|(guid, entity, transform, _)| {
                Some((guid.0, (transform.translation, range_units.unit(entity)?)))
            })
            .collect();
        let ranges = ranges.as_ref().expect("range projection").clone();
        let mods = (*mods).clone();
        Box::new(move |id, guid| {
            let spell = ranges.get(&id)?;
            let (position, target) = units.get(&guid)?;
            let targets = RangeTargets {
                target: Some(*target),
                attack_target: None,
            };
            let (min, max) =
                mods.min_max_range(&spell.definition, spell.row.as_ref(), caster, targets)?;
            let distance = origin.distance_squared(*position);
            Some(distance >= min * min && distance <= max * max)
        }) as runtime::RangeQuery
    });
    runtime::set_range_query(script.lua(), query);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pfui::test_support::{fixture, HEAL};

    fn feeds(world: &World) -> u64 {
        runtime::cooldown_feeds(world.non_send_resource::<UiScript>().lua())
    }

    /// The fixture's spell, as its catalog holds it, with a cooldown of its own.
    fn heal() -> SpellDisplay {
        SpellDisplay {
            start_recovery_category: 133,
            start_recovery_ms: 1500,
            recovery_ms: 30_000,
            ..Default::default()
        }
    }

    /// The rows are built once and again when the list flushes, not on every frame: neither for
    /// a record the two-minute sweep has yet to free (a GCD-only one is its from the cast on),
    /// nor because the resource was taken mutably.
    #[test]
    fn the_cooldown_rows_are_rebuilt_at_a_flush_alone() {
        let (mut app, _commands) = fixture();
        let world = app.world_mut();
        let feed = world.register_system(feed);
        let now = Instant::now();
        world.resource_mut::<crate::spell::Cooldowns>().start_gcd(
            HEAL,
            &heal(),
            now,
            &crate::spell::SpellModifiers::default(),
        );
        assert!(
            world
                .resource::<crate::spell::Cooldowns>()
                .sweep_pending(now),
            "a GCD-only record is the sweep's from the start"
        );

        world.run_system(feed).unwrap();
        let built = feeds(world);
        assert_eq!(built, 1, "the first frame builds the rows");
        for _ in 0..3 {
            // A mutable borrow that writes nothing, as the list's lane takes every frame.
            let _ = &mut *world.resource_mut::<crate::spell::Cooldowns>();
            world.run_system(feed).unwrap();
        }
        assert_eq!(feeds(world), built, "frames with no flush rebuild nothing");

        world.resource_mut::<crate::spell::Cooldowns>().flush();
        world.run_system(feed).unwrap();
        assert_eq!(feeds(world), built + 1, "a flush rebuilds them");
    }

    /// `C_Spell.GetSpellCooldown` reads the list as of the call, so a cooldown started after the
    /// rows were built shows with no rebuild.
    #[test]
    fn the_cooldown_getter_reads_the_list_at_the_call() {
        let (mut app, _commands) = fixture();
        let feed = app.world_mut().register_system(feed);
        app.world_mut().run_system(feed).unwrap();
        let built = feeds(app.world());
        crate::pfui::test_support::run(
            &mut app,
            &format!("cold = C_Spell.GetSpellCooldown({HEAL}).duration"),
        );
        app.world_mut()
            .resource_mut::<crate::spell::Cooldowns>()
            .start_spell(HEAL, &heal(), 0, Instant::now(), None);
        crate::pfui::test_support::run(
            &mut app,
            &format!("running = C_Spell.GetSpellCooldown({HEAL}).duration"),
        );
        let script = app.world().non_send_resource::<UiScript>();
        assert_eq!(
            script.eval::<(f64, f64)>("return cold, running").unwrap(),
            (0.0, 30.0)
        );
        assert_eq!(feeds(app.world()), built, "and no rebuild between");
    }
}
