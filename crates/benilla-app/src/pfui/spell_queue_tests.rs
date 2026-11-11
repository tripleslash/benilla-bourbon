use super::{State, *};
use benilla_ui::script::SpellSlotView;
use bevy::prelude::*;

#[path = "spell_queue_special_tests.rs"]
mod special;

pub(crate) use crate::pfui::test_support::*;

fn with_non_gcd(app: &mut App) {
    let mut spells = crate::ui_action::Spells::empty_for_tests();
    spells.catalog = benilla_formats::SpellCatalog::from_displays(
        [HEAL, OTHER, 5001, 5002, 5003]
            .map(|id| {
                (
                    id,
                    benilla_formats::SpellDisplay {
                        name: if id < 5000 {
                            format!("Heal{id}")
                        } else {
                            format!("Spell{id}")
                        },
                        start_recovery_category: if id < 5000 {
                            133
                        } else if id == 5001 {
                            0
                        } else {
                            7
                        },
                        implicit_target_a1: 1,
                        ..Default::default()
                    },
                )
            })
            .into(),
    );
    app.insert_resource(spells);
    let mut script = app.world_mut().non_send_mut::<UiScript>();
    let mut book = script.spellbook();
    book.slots
        .extend([5001, 5002, 5003].map(|id| SpellSlotView {
            spell_id: id,
            name: format!("Spell{id}"),
            ..Default::default()
        }));
    script.set_spellbook(book);
}

#[test]
fn explicit_target_tokens_follow_macro_call_order_and_live_target_hops() {
    for (lua, expected) in [
        (
            "CastSpellByNameNoQueue('Heal2050','target');TargetUnit('player')",
            vec![Some(4)],
        ),
        (
            "TargetUnit('player');CastSpellByNameNoQueue('Heal2050','target')",
            vec![Some(ME)],
        ),
        (
            "ClearTarget();CastSpellByNameNoQueue('Heal2050','target')",
            vec![],
        ),
        (
            "ClearTarget();TargetUnit('player');CastSpell('Heal2050','spell','target')",
            vec![Some(ME)],
        ),
        (
            "TargetUnit('player');CastSpellNoQueue(1,0,'TARGETTARGET')",
            vec![Some(4)],
        ),
    ] {
        let (mut app, rx) = fixture();
        with_unit_targets(&mut app);
        run(&mut app, lua);
        assert_eq!(targets(&rx), expected, "{lua}");
    }
}

#[test]
fn queued_explicit_target_is_bound_after_selection_and_keeps_that_guid() {
    let (mut app, rx) = fixture();
    with_unit_targets(&mut app);
    casting(&app, Some(300.0));
    run(
        &mut app,
        "TargetUnit('player');CastSpellByName('Heal2050','target');ClearTarget()",
    );
    assert!(targets(&rx).is_empty());
    casting(&app, None);
    run(&mut app, "");
    assert_eq!(targets(&rx), vec![Some(ME)]);
    assert_eq!(
        app.world().resource::<crate::target::Selection>().guid,
        None
    );
}

#[test]
fn queued_scripts_run_at_their_documented_priority_against_real_cast_commands() {
    for priority in 1..=3 {
        let (mut app, rx) = fixture();
        with_non_gcd(&mut app);
        casting(&app, Some(300.0));
        run(
            &mut app,
            &format!(
                r#"
          order={{}};listener:RegisterEvent('SPELL_CAST_EVENT')
          listener:HookScript('OnEvent',function()
            if event=='SPELL_CAST_EVENT' and arg1==1 then table.insert(order,arg2) end
          end)
          QueueScript('table.insert(order,"script")',{priority})
          QueueSpellByName('Heal2050');QueueSpellByName('Spell5001')
        "#
            ),
        );
        assert!(cast_packets(&rx).is_empty());
        casting(&app, None);
        for _ in 0..5 {
            app.insert_resource(crate::spell::PendingCast::default());
            app.world_mut().non_send_mut::<UiScript>().tick(0.11);
            run(&mut app, "");
        }
        assert_eq!(cast_packets(&rx), vec![5001, HEAL]);
        let expected = match priority {
            1 => "assert(order[1]=='script' and order[2]==5001 and order[3]==2050)",
            2 => "assert(order[1]==5001 and order[2]=='script' and order[3]==2050)",
            _ => "assert(order[1]==5001 and order[2]==2050 and order[3]=='script')",
        };
        app.world().non_send::<UiScript>().run(expected).unwrap();
    }
}

#[test]
fn queued_script_replacement_window_and_session_reset_are_per_vm() {
    let (mut app, _) = fixture();
    casting(&app, Some(5000.0));
    run(&mut app, "count=0;QueueScript('count=count+1')");
    app.world()
        .non_send::<UiScript>()
        .run("assert(count==1)")
        .unwrap();
    casting(&app, Some(300.0));
    run(
        &mut app,
        "QueueScript('count=count+2');QueueScript('count=count+4',2)",
    );
    casting(&app, None);
    run(&mut app, "");
    app.world()
        .non_send::<UiScript>()
        .run("assert(count==5)")
        .unwrap();
    run(&mut app, r#"QueueScript("QueueScript('count=count+8')")"#);
    app.world()
        .non_send::<UiScript>()
        .run("assert(count==13)")
        .unwrap();
    casting(&app, Some(300.0));
    run(&mut app, "QueueScript('count=count+16')");
    reset(app.world().non_send::<UiScript>());
    casting(&app, None);
    run(&mut app, "");
    app.world()
        .non_send::<UiScript>()
        .run("assert(count==13)")
        .unwrap();
    casting(&app, Some(300.0));
    run(&mut app, "QueueScript('count=999')");
    let new_vm = UiScript::new().unwrap();
    benilla_pfui::install(&new_vm).unwrap();
    new_vm.run("count=100").unwrap();
    app.world_mut().insert_non_send(new_vm);
    casting(&app, None);
    run(&mut app, "");
    app.world()
        .non_send::<UiScript>()
        .run("assert(count==100)")
        .unwrap();
}

#[test]
fn queued_scripts_wait_for_the_real_gcd_without_a_cast_bar() {
    let (mut app, _) = fixture();
    let def = benilla_formats::SpellDisplay {
        start_recovery_category: 133,
        start_recovery_ms: 1500,
        ..Default::default()
    };
    app.world_mut()
        .resource_mut::<crate::spell::Cooldowns>()
        .start_gcd(
            HEAL,
            &def,
            Instant::now(),
            &crate::spell::SpellModifiers::default(),
        );
    run(
        &mut app,
        "count=0;SetCVar('NP_SpellQueueWindowMs',2000);QueueScript('count=1')",
    );
    run(&mut app, "assert(count==0)");
    app.insert_resource(crate::spell::Cooldowns::default());
    run(&mut app, "");
    app.world()
        .non_send::<UiScript>()
        .run("assert(count==1)")
        .unwrap();
}

#[test]
fn non_gcd_queue_has_priority_and_spaces_sends_without_replacing_the_normal_queue() {
    let (mut app, rx) = fixture();
    with_non_gcd(&mut app);
    casting(&app, Some(3000.0));
    run(&mut app, "QueueSpellByName('Heal2050');QueueSpellByName('Spell5001');QueueSpellByName('Spell5002');QueueSpellByName('Spell5001')");
    let script = app.world().non_send::<UiScript>();
    assert_eq!(
        script.lua().app_data_ref::<State>().unwrap().non_gcd.len(),
        2
    );
    script.run("assert(#rows==5 and rows[1][1]==2 and rows[2][1]==4 and rows[4][1]==5 and rows[5][1]==4)").unwrap();
    casting(&app, None);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![5001]);
    app.insert_resource(crate::spell::PendingCast::default());
    app.world_mut().non_send_mut::<UiScript>().tick(0.05);
    run(&mut app, "");
    assert!(
        cast_packets(&rx).is_empty(),
        "non-GCD casts retain their 100 ms buffer after an ACK"
    );
    app.world_mut().non_send_mut::<UiScript>().tick(0.06);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![5002]);
    app.insert_resource(crate::spell::PendingCast::default());
    app.world_mut().non_send_mut::<UiScript>().tick(0.11);
    run(&mut app, "");
    assert_eq!(
        cast_packets(&rx),
        vec![HEAL],
        "the normal entry survives both non-GCD releases"
    );
}

#[test]
fn non_gcd_category_replacement_is_opt_in_and_cancellation_clears_both_lanes() {
    let (mut app, rx) = fixture();
    with_non_gcd(&mut app);
    casting(&app, Some(3000.0));
    run(
        &mut app,
        "QueueSpellByName('Spell5002');QueueSpellByName('Spell5003');QueueSpellByName('Heal2050')",
    );
    assert_eq!(
        app.world()
            .non_send::<UiScript>()
            .lua()
            .app_data_ref::<State>()
            .unwrap()
            .non_gcd
            .len(),
        2
    );
    clear(app.world().non_send::<UiScript>());
    run(&mut app, "SetCVar('NP_ReplaceMatchingNonGcdCategory',1);QueueSpellByName('Spell5002');QueueSpellByName('Spell5003');QueueSpellByName('Heal2050')");
    let script = app.world().non_send::<UiScript>();
    let state = script.lua().app_data_ref::<State>().unwrap();
    assert_eq!(state.non_gcd.len(), 1);
    assert_eq!(state.non_gcd[0].spell, 5003);
    drop(state);
    casting(&app, None);
    // A cast in flight, which `SpellStopCasting` reads through the host and stops.
    app.world_mut()
        .resource_mut::<crate::spell::PendingCast>()
        .push(OTHER, None, Instant::now());
    run(&mut app, "SpellStopCasting()");
    assert_eq!(cast_packets(&rx), Vec::<u32>::new());
    let script = app.world().non_send::<UiScript>();
    let state = script.lua().app_data_ref::<State>().unwrap();
    assert!(state.non_gcd.is_empty() && state.queued.is_none());
}

#[test]
fn forced_queue_waits_outside_window_and_a_noqueue_attempt_does_not_replace_it() {
    let (mut app, rx) = fixture();
    casting(&app, Some(10000.0));
    app.world_mut()
        .resource_mut::<crate::spell::PendingCast>()
        .push(OTHER, None, Instant::now());
    run(
        &mut app,
        "QueueSpellByName('Heal2050');CastSpellByNameNoQueue('Heal2052',1)",
    );
    assert!(
        cast_packets(&rx).is_empty(),
        "no-queue cast is still validated against the active cast"
    );
    let script = app.world().non_send::<UiScript>();
    script
        .run("assert(#rows==1 and rows[1][1]==2 and rows[1][2]==2050)")
        .unwrap();
    assert_eq!(
        script
            .lua()
            .app_data_ref::<State>()
            .unwrap()
            .queued
            .as_ref()
            .unwrap()
            .spell,
        HEAL
    );
    // A forced request survives a long cast, even if its server start arrived after the press.
    app.world_mut().non_send_mut::<UiScript>().tick(8.0);
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
    casting(&app, None);
    app.insert_resource(crate::spell::PendingCast::default());
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    app.world()
        .non_send::<UiScript>()
        .run("assert(#rows==2 and rows[2][1]==3)")
        .unwrap();
    run(&mut app, "");
    assert!(
        cast_packets(&rx).is_empty(),
        "a popped request is sent only once"
    );
}

#[test]
fn two_forced_calls_in_one_macro_send_one_and_queue_one() {
    let (mut app, rx) = fixture();
    run(
        &mut app,
        "QueueSpellByName('Heal2050');QueueSpellByName('Heal2050')",
    );
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    app.world()
        .non_send::<UiScript>()
        .run("assert(#rows==1 and rows[1][1]==2)")
        .unwrap();
    run(&mut app, "QueueSpellByName('Heal2052')");
    app.world().non_send::<UiScript>().run("assert(#rows==3 and rows[2][1]==3 and rows[2][2]==2050 and rows[3][1]==2 and rows[3][2]==2052)").unwrap();
}

#[test]
fn a_queued_self_cast_survives_an_unrelated_selection_leaving_the_stream() {
    let (mut app, rx) = fixture();
    app.world_mut()
        .resource_mut::<crate::target::Selection>()
        .guid = Some(0xdead);
    casting(&app, Some(300.0));
    run(&mut app, "CastSpellByName('Heal2050',1)");
    assert!(cast_packets(&rx).is_empty());
    casting(&app, None);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    assert_eq!(
        app.world().resource::<crate::target::Selection>().guid,
        Some(0xdead)
    );
}

#[test]
fn cancellation_precedes_release_and_the_queue_is_per_vm() {
    let (mut app, rx) = fixture();
    casting(&app, Some(3000.0));
    run(&mut app, "QueueSpellByName('Heal2050')");
    assert_eq!(
        cast_packets(&rx),
        Vec::<u32>::new(),
        "the cast is queued before cancellation"
    );
    casting(&app, None);
    // A cast in flight, which `SpellStopCasting` reads through the host and stops.
    app.world_mut()
        .resource_mut::<crate::spell::PendingCast>()
        .push(OTHER, None, Instant::now());
    run(&mut app, "assert(SpellStopCasting()==1)");
    assert_eq!(
        cast_packets(&rx),
        Vec::<u32>::new(),
        "cancellation wins over release"
    );
    app.world()
        .non_send::<UiScript>()
        .run("assert(#rows==2 and rows[2][1]==3)")
        .unwrap();
    casting(&app, Some(3000.0));
    run(&mut app, "QueueSpellByName('Heal2050')");
    let new_vm = UiScript::new().unwrap();
    benilla_pfui::install(&new_vm).unwrap();
    app.world_mut().insert_non_send(new_vm);
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
}
