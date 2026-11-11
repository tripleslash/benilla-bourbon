//! Exercise extension queues through the real cast ladder and outgoing command stream.

use super::*;
use crate::net::ClientCommand;

fn special_spells(app: &mut App, ground: bool) {
    app.world_mut()
        .resource_mut::<crate::ui_action::Spells>()
        .catalog = benilla_formats::SpellCatalog::from_displays(
        [HEAL, OTHER]
            .map(|id| {
                (
                    id,
                    benilla_formats::SpellDisplay {
                        name: format!("Heal{id}"),
                        targets: if ground {
                            if id == HEAL {
                                0x40
                            } else {
                                0x20
                            }
                        } else {
                            0
                        },
                        attributes: if ground { 0 } else { 0x400 },
                        implicit_target_a1: if ground { 0 } else { 1 },
                        start_recovery_category: if id == HEAL && ground { 133 } else { 0 },
                        ..Default::default()
                    },
                )
            })
            .into(),
    );
}

fn aim(app: &mut App, point: Option<Vec3>) {
    app.world_mut()
        .resource_mut::<crate::target::PickOcclusion>()
        .point = point;
}

fn terrain_packets(rx: &crossbeam_channel::Receiver<ClientCommand>) -> Vec<(u32, bool, [f32; 3])> {
    rx.try_iter()
        .filter_map(|c| match c {
            ClientCommand::CastSpellAtDest { spell_id, dest } => Some((spell_id, false, dest)),
            ClientCommand::CastSpellAtSource { spell_id, src } => Some((spell_id, true, src)),
            _ => None,
        })
        .collect()
}

#[test]
fn queued_ground_spells_retain_the_hit_and_use_the_targeting_window() {
    for (name, id, source, kind, pushed, quickcast) in [
        ("Heal2050", HEAL, false, 4, 2, 0),
        ("Heal2052", OTHER, true, 5, 4, 0),
        ("Heal2050", HEAL, false, 4, 2, 1),
        ("Heal2052", OTHER, true, 5, 4, 1),
    ] {
        let (mut app, rx) = fixture();
        special_spells(&mut app, true);
        aim(&mut app, Some(Vec3::new(2.0, 3.0, -4.0)));
        casting(&app, Some(700.0));
        run(
            &mut app,
            &format!(
                "SetCVar('NP_SpellQueueWindowMs',50);SetCVar('NP_TargetingQueueWindowMs',800);\
             SetCVar('NP_QuickcastTargetingSpells',{quickcast});\
             listener:RegisterEvent('SPELL_CAST_EVENT');CastSpellByName('{name}')"
            ),
        );
        assert!(terrain_packets(&rx).is_empty());
        // Movement, a sky aim and a vanished selection cannot redirect a queued location.
        aim(&mut app, (quickcast != 0).then_some(Vec3::splat(100.0)));
        app.world_mut()
            .resource_mut::<crate::target::Selection>()
            .guid = Some(0xdead);
        casting(&app, None);
        run(&mut app, "");
        assert_eq!(terrain_packets(&rx), vec![(id, source, [4.0, -2.0, 3.0])]);
        assert!(!app
            .world()
            .resource::<crate::spell::SpellTargeting>()
            .active());
        app.world()
            .non_send::<UiScript>()
            .run(&format!(
                "assert(#rows==3 and rows[1][1]=={pushed} and rows[2][1]=={pushed}+1);\
             assert(rows[3][1]==1 and rows[3][2]=={id} and rows[3][3]=={kind})"
            ))
            .unwrap();
        run(&mut app, "");
        assert!(
            terrain_packets(&rx).is_empty(),
            "one pop produces one packet"
        );
    }
}

#[test]
fn ground_queue_switch_and_window_do_not_disable_ordinary_targeting() {
    for settings in [
        "SetCVar('NP_QueueTargetingSpells',0)",
        "SetCVar('NP_TargetingQueueWindowMs',200)",
    ] {
        let (mut app, rx) = fixture();
        special_spells(&mut app, true);
        aim(&mut app, Some(Vec3::ZERO));
        casting(&app, Some(300.0));
        run(&mut app, &format!("{settings};CastSpellByName('Heal2050')"));
        assert!(terrain_packets(&rx).is_empty());
        assert!(app
            .world()
            .resource::<crate::spell::SpellTargeting>()
            .active());
        app.world()
            .non_send::<UiScript>()
            .run("assert(#rows==0)")
            .unwrap();
    }
}

#[test]
fn quickcast_settings_and_noqueue_use_the_same_validated_terrain_commit() {
    for (first, second) in [
        (
            "SetCVar('NP_QuickcastTargetingSpells',1)",
            "CastSpellByName('Heal2050')",
        ),
        (
            "SetCVar('NP_QuickcastTargetingSpells',1)",
            "CastSpellByNameNoQueue('Heal2050')",
        ),
        (
            "SetCVar('NP_QuickcastOnDoubleCast',1);CastSpellByName('Heal2050')",
            "CastSpellByName('Heal2050')",
        ),
    ] {
        let (mut app, rx) = fixture();
        special_spells(&mut app, true);
        aim(&mut app, Some(Vec3::new(-2.0, 5.0, 8.0)));
        run(&mut app, first);
        assert!(terrain_packets(&rx).is_empty());
        run(&mut app, second);
        assert_eq!(terrain_packets(&rx), vec![(HEAL, false, [-8.0, 2.0, 5.0])]);
        assert!(!app
            .world()
            .resource::<crate::spell::SpellTargeting>()
            .active());
    }
    let (mut app, rx) = fixture();
    special_spells(&mut app, true);
    run(
        &mut app,
        "SetCVar('NP_QuickcastTargetingSpells',1);CastSpellByName('Heal2050')",
    );
    assert!(
        terrain_packets(&rx).is_empty(),
        "sky is never a fabricated coordinate"
    );
    assert!(app
        .world()
        .resource::<crate::spell::SpellTargeting>()
        .active());
    aim(&mut app, Some(Vec3::ZERO));
    app.world_mut()
        .resource_mut::<crate::spell::PendingCast>()
        .push(OTHER, None, Instant::now());
    run(&mut app, "CastSpellByNameNoQueue('Heal2050')");
    assert!(
        terrain_packets(&rx).is_empty(),
        "quickcast retains the already-casting gate"
    );
}

/// A right press in the world, TurnOrAction's world click armed, with a payload `held` on the
/// cursor or none; the payload is dropped again after.
fn press_right(app: &mut App, held: bool) {
    crate::script_calls::with_hosted_vm(app.world_mut(), |script| {
        let script = script.expect("a VM");
        if held {
            script.run("PickupSpell(1, 'spell')").unwrap();
            assert!(script.cursor_payload().is_some(), "a payload is held");
        }
        script.submit_click(benilla_ui::script::ClickCall::Arm(
            benilla_ui::script::ClickMode::Action,
        ));
        if held {
            script.run("ClearCursor()").unwrap();
        }
    });
}

#[test]
fn ground_cancellation_wins_over_a_ready_release() {
    for right_press in [false, true] {
        let (mut app, rx) = fixture();
        special_spells(&mut app, true);
        aim(&mut app, Some(Vec3::ZERO));
        casting(&app, Some(300.0));
        run(&mut app, "CastSpellByName('Heal2050')");
        casting(&app, None);
        if right_press {
            press_right(&mut app, false);
        } else {
            run(&mut app, "SpellStopTargeting()");
        }
        run(&mut app, "");
        assert!(terrain_packets(&rx).is_empty());
        app.world()
            .non_send::<UiScript>()
            .run("assert(#rows==2 and rows[2][1]==3)")
            .unwrap();
    }
}

#[test]
fn a_world_cancel_preserves_queued_unit_spells_and_a_held_payload_preempts_it() {
    for held in [false, true] {
        let (mut app, rx) = fixture();
        casting(&app, Some(300.0));
        run(&mut app, "CastSpellByName('Heal2050',1)");
        press_right(&mut app, held);
        casting(&app, None);
        run(&mut app, "");
        assert_eq!(cast_packets(&rx), vec![HEAL]);
    }
    let (mut app, rx) = fixture();
    special_spells(&mut app, true);
    aim(&mut app, Some(Vec3::ZERO));
    casting(&app, Some(300.0));
    run(&mut app, "CastSpellByName('Heal2050')");
    press_right(&mut app, true);
    casting(&app, None);
    run(&mut app, "");
    assert_eq!(terrain_packets(&rx), vec![(HEAL, false, [0.0, 0.0, 0.0])]);
}

#[test]
fn six_non_gcd_entries_retain_fifo_order() {
    let (mut app, rx) = fixture();
    app.world_mut()
        .resource_mut::<crate::ui_action::Spells>()
        .catalog = benilla_formats::SpellCatalog::from_displays(
        (5001..=5006)
            .map(|id| {
                (
                    id,
                    benilla_formats::SpellDisplay {
                        name: format!("Spell{id}"),
                        implicit_target_a1: 1,
                        ..Default::default()
                    },
                )
            })
            .collect(),
    );
    let mut script = app.world_mut().non_send_mut::<UiScript>();
    let mut book = script.spellbook();
    book.slots.extend((5001..=5006).map(|id| SpellSlotView {
        spell_id: id,
        name: format!("Spell{id}"),
        ..Default::default()
    }));
    script.set_spellbook(book);
    casting(&app, Some(3000.0));
    run(
        &mut app,
        "for i=5001,5006 do QueueSpellByName('Spell'..i) end",
    );
    assert!(cast_packets(&rx).is_empty());
    casting(&app, None);
    for id in 5001..=5006 {
        app.insert_resource(crate::spell::PendingCast::default());
        app.world_mut()
            .non_send_mut::<UiScript>()
            .tick(0.11);
        run(&mut app, "");
        assert_eq!(cast_packets(&rx), vec![id]);
    }
}

#[test]
fn on_swing_is_opt_in_and_cannot_immediately_double_queue() {
    let (mut app, rx) = fixture();
    with_unit_targets(&mut app);
    special_spells(&mut app, false);
    run(&mut app, "listener:RegisterEvent('SPELL_CAST_EVENT');CastSpellByName('Heal2050','player');CastSpellByName('Heal2050','player')");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    app.world()
        .non_send::<UiScript>()
        .run("assert(#rows==1 and rows[1][3]==2)")
        .unwrap();
    run(
        &mut app,
        "SetCVar('NP_QueueOnSwingSpells',1);CastSpellByName('Heal2050','player')",
    );
    assert!(cast_packets(&rx).is_empty());
    assert!(app
        .world()
        .non_send::<UiScript>()
        .lua()
        .app_data_ref::<State>()
        .unwrap()
        .on_swing
        .is_none());
    app.world_mut()
        .non_send_mut::<UiScript>()
        .tick(0.51);
    run(&mut app, "CastSpellByName('Heal2050','player')");
    assert!(cast_packets(&rx).is_empty());
    app.world()
        .non_send::<UiScript>()
        .run("assert(#rows==2 and rows[2][1]==0)")
        .unwrap();
    // Even a very slow swing may retain the next strike until the server clears the melee slot.
    app.world_mut()
        .non_send_mut::<UiScript>()
        .tick(8.0);
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
    // The server's answer empties the melee slot.
    app.insert_resource(crate::spell::QueuedMeleeSpell::default());
    app.insert_resource(crate::spell::PendingCast::default());
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    app.world()
        .non_send::<UiScript>()
        .run("assert(#rows==4 and rows[3][1]==1 and rows[4][3]==2)")
        .unwrap();
}

#[test]
fn on_swing_replacement_and_cancellation_do_not_replace_the_normal_lane() {
    let (mut app, rx) = fixture();
    with_unit_targets(&mut app);
    special_spells(&mut app, false);
    // A synthetic normal spell alongside the two strike rows.
    let strike = |id: u32| benilla_formats::SpellDisplay {
        name: format!("Heal{id}"),
        attributes: 0x400,
        implicit_target_a1: 1,
        ..Default::default()
    };
    app.world_mut()
        .resource_mut::<crate::ui_action::Spells>()
        .catalog = benilla_formats::SpellCatalog::from_displays(
        [
            (HEAL, strike(HEAL)),
            (OTHER, strike(OTHER)),
            (
                5001,
                benilla_formats::SpellDisplay {
                    name: "Normal".into(),
                    start_recovery_category: 133,
                    implicit_target_a1: 1,
                    ..Default::default()
                },
            ),
        ]
        .into(),
    );
    let mut script = app.world_mut().non_send_mut::<UiScript>();
    let mut book = script.spellbook();
    book.slots.push(SpellSlotView {
        spell_id: 5001,
        name: "Normal".into(),
        ..Default::default()
    });
    script.set_spellbook(book);
    casting(&app, Some(300.0));
    run(&mut app, "SetCVar('NP_QueueOnSwingSpells',1);CastSpellByName('Normal');CastSpellByName('Heal2050','player');CastSpellByName('Heal2052','player')");
    assert!(cast_packets(&rx).is_empty());
    casting(&app, None);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![5001]);
    app.insert_resource(crate::spell::PendingCast::default());
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![OTHER]);
    app.world().non_send::<UiScript>().run("assert(rows[1][1]==2 and rows[2][1]==0 and rows[3][1]==1 and rows[4][1]==0 and rows[6][1]==1)").unwrap();
    casting(&app, Some(300.0));
    run(&mut app, "QueueSpellByName('Heal2050')");
    casting(&app, None);
    run(&mut app, "SpellStopCasting()");
    assert!(cast_packets(&rx).is_empty());
    assert!(app
        .world()
        .non_send::<UiScript>()
        .lua()
        .app_data_ref::<State>()
        .unwrap()
        .on_swing
        .is_none());
}
