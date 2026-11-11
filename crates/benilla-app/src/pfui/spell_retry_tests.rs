//! Drive server verdicts through the registered stock handlers and the extension event feed.

use super::*;
use crate::pfui::spell_queue::tests::{
    cast_packets, casting, fixture, run, targets, with_unit_targets,
};
use bevy::ecs::system::RunSystemOnce;

const HEAL: u32 = 2050;
const OTHER: u32 = 2052;

fn packet(app: &mut App, event: SessionEvent) {
    crate::net::handlers::dispatch(app.world_mut(), vec![event]);
    app.world_mut().run_system_once(super::feed).unwrap();
}

fn verdict(app: &mut App, success: bool, reason: u8) {
    packet(
        app,
        SessionEvent::CastResult {
            spell_id: HEAL,
            success,
            reason: (!success).then_some(reason),
            arg: None,
        },
    );
}

fn tick(app: &mut App, seconds: f32) {
    app.world_mut()
        .non_send_mut::<UiScript>()
        .tick(seconds);
}

#[test]
fn only_owned_transient_server_failures_retry_and_report_each_verdict_once() {
    for reason in [0x28, 0x3c, 0x61, 0x59, 0x0a, 0x00] {
        let (mut app, rx) = fixture();
        run(&mut app, "failures=0;listener:RegisterEvent('SPELL_FAILED_SELF');listener:HookScript('OnEvent',function() if event=='SPELL_FAILED_SELF' and arg3==1 then failures=failures+1 end end);CastSpellByName('Heal2050',1)");
        assert_eq!(cast_packets(&rx), vec![HEAL]);
        verdict(&mut app, false, reason);
        run(&mut app, "assert(failures==1)");
        assert_eq!(
            cast_packets(&rx),
            if matches!(reason, 0x28 | 0x3c | 0x61) {
                vec![HEAL]
            } else {
                vec![]
            },
            "reason {reason:#x}"
        );
        run(&mut app, "");
        assert!(cast_packets(&rx).is_empty());
    }
    let (mut app, rx) = fixture();
    verdict(&mut app, false, 0x3c);
    run(&mut app, "");
    assert!(
        cast_packets(&rx).is_empty(),
        "unowned server failures cannot invent a cast"
    );
}

#[test]
fn retry_settings_noqueue_and_cancel_keep_requests_single_attempt() {
    for (before, cast, after) in [
        (
            "SetCVar('NP_RetryServerRejectedSpells',0)",
            "CastSpellByName",
            "",
        ),
        ("", "CastSpellByNameNoQueue", ""),
        (
            "",
            "CastSpellByName",
            "SetCVar('NP_RetryServerRejectedSpells',0)",
        ),
        ("", "CastSpellByName", "SpellStopCasting()"),
    ] {
        let (mut app, rx) = fixture();
        run(&mut app, &format!("{before}\n{cast}('Heal2050',1)"));
        assert_eq!(cast_packets(&rx), vec![HEAL]);
        run(&mut app, after);
        verdict(&mut app, false, 0x3c);
        run(&mut app, "");
        assert!(cast_packets(&rx).is_empty(), "{before};{cast};{after}");
    }
}

#[test]
fn retries_keep_the_bound_guid_and_never_fall_back_from_a_missing_unit() {
    for missing in [false, true] {
        let (mut app, rx) = fixture();
        with_unit_targets(&mut app);
        run(&mut app, "CastSpellByName('Heal2050')");
        assert_eq!(targets(&rx), vec![Some(4)]);
        verdict(&mut app, false, 0x61);
        if missing {
            app.world_mut()
                .resource_mut::<crate::net::GuidIndex>()
                .0
                .remove(&4);
        }
        run(&mut app, "TargetUnit('player');SetCVar('autoSelfCast',1)");
        assert_eq!(targets(&rx), if missing { vec![] } else { vec![Some(4)] });
        run(&mut app, "");
        assert!(targets(&rx).is_empty());
    }
}

#[test]
fn retries_stop_after_three_resends_or_two_seconds() {
    let (mut app, rx) = fixture();
    run(&mut app, "CastSpellByName('Heal2050',1)");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    for attempt in 1..=5 {
        verdict(&mut app, false, 0x28);
        run(&mut app, "");
        assert_eq!(
            cast_packets(&rx),
            if attempt <= 3 { vec![HEAL] } else { vec![] },
            "attempt {attempt}"
        );
    }
    for delay in [true, false] {
        let (mut app, rx) = fixture();
        run(&mut app, "CastSpellByName('Heal2050',1)");
        assert_eq!(cast_packets(&rx), vec![HEAL]);
        if delay {
            tick(&mut app, 2.01)
        }
        verdict(&mut app, false, 0x3c);
        if !delay {
            tick(&mut app, 2.01)
        }
        run(&mut app, "");
        assert!(cast_packets(&rx).is_empty());
    }
}

#[test]
fn a_newer_player_queue_wins_and_acceptance_only_removes_retries() {
    let (mut app, rx) = fixture();
    run(
        &mut app,
        "CastSpellByName('Heal2050',1);QueueSpellByName('Heal2052')",
    );
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    verdict(&mut app, false, 0x3c);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![OTHER]);
    for retry in [true, false] {
        let (mut app, rx) = fixture();
        run(&mut app, "CastSpellByName('Heal2050',1)");
        assert_eq!(cast_packets(&rx), vec![HEAL]);
        if retry {
            verdict(&mut app, false, 0x61)
        } else {
            run(&mut app, "QueueSpellByName('Heal2050')")
        }
        verdict(&mut app, true, 0);
        // Acceptance preserves the ordinary request, which still waits for the stock GCD.
        app.world_mut()
            .resource_mut::<crate::spell::Cooldowns>()
            .clear_gcd(HEAL, bevy::platform::time::Instant::now());
        run(&mut app, "");
        assert_eq!(cast_packets(&rx), if retry { vec![] } else { vec![HEAL] });
    }
}

#[test]
fn logout_disconnect_and_replacement_vm_discard_retries() {
    for event in [
        SessionEvent::LoggedOut,
        SessionEvent::Disconnected {
            reason: "retry fixture".into(),
            end: benilla_protocol::SessionEnd::Lost,
        },
    ] {
        let (mut app, rx) = fixture();
        run(&mut app, "CastSpellByName('Heal2050',1)");
        assert_eq!(cast_packets(&rx), vec![HEAL]);
        verdict(&mut app, false, 0x3c);
        crate::net::handlers::dispatch(app.world_mut(), vec![event]);
        run(&mut app, "");
        assert!(cast_packets(&rx).is_empty());
    }
    let (mut app, rx) = fixture();
    run(&mut app, "CastSpellByName('Heal2050',1)");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    verdict(&mut app, false, 0x3c);
    let (mut fresh, _) = fixture();
    let script = fresh
        .world_mut()
        .remove_non_send::<UiScript>()
        .unwrap();
    app.world_mut().insert_non_send(script);
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
}

#[test]
fn ordinary_queued_releases_can_retry_without_resetting_the_budget() {
    let (mut app, rx) = fixture();
    casting(&app, Some(300.0));
    run(&mut app, "CastSpellByName('Heal2050',1)");
    assert!(cast_packets(&rx).is_empty());
    casting(&app, None);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    verdict(&mut app, false, 0x3c);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
}

#[test]
fn disabling_retries_after_rejection_discards_the_waiting_attempt() {
    let (mut app, rx) = fixture();
    run(&mut app, "CastSpellByName('Heal2050',1)");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    verdict(&mut app, false, 0x3c);
    run(&mut app, "SetCVar('NP_RetryServerRejectedSpells',0)");
    assert!(cast_packets(&rx).is_empty());
}

#[test]
fn automatic_self_fallback_is_retained_after_selection_changes() {
    let (mut app, rx) = fixture();
    with_unit_targets(&mut app);
    app.world_mut()
        .resource_mut::<crate::spell::AutoSelfCast>()
        .0 = true;
    run(&mut app, "ClearTarget();CastSpellByName('Heal2050')");
    assert_eq!(targets(&rx), vec![Some(3)]);
    verdict(&mut app, false, 0x3c);
    app.world_mut()
        .resource_mut::<crate::spell::AutoSelfCast>()
        .0 = false;
    run(&mut app, "TargetUnit('0x0000000000000004')");
    assert_eq!(targets(&rx), vec![Some(3)]);
}

#[test]
fn terrain_retries_keep_wire_coordinates_and_stop_targeting_cancels_them() {
    use crate::net::ClientCommand;
    for (source, cancel, nested) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (false, true, true),
    ] {
        let (mut app, rx) = fixture();
        app.world_mut()
            .resource_mut::<crate::ui_action::Spells>()
            .catalog = benilla_formats::SpellCatalog::from_displays(
            [(
                HEAL,
                benilla_formats::SpellDisplay {
                    name: "Heal2050".into(),
                    targets: if source { 0x20 } else { 0x40 },
                    start_recovery_category: 133,
                    ..Default::default()
                },
            )]
            .into(),
        );
        app.world_mut()
            .resource_mut::<crate::target::PickOcclusion>()
            .point = Some(Vec3::new(2.0, 3.0, -4.0));
        run(
            &mut app,
            "SetCVar('NP_QuickcastTargetingSpells',1);CastSpellByName('Heal2050')",
        );
        let points = |rx: &crossbeam_channel::Receiver<ClientCommand>| {
            rx.try_iter()
                .filter_map(|c| match c {
                    ClientCommand::CastSpellAtDest { dest, .. } => Some((false, dest)),
                    ClientCommand::CastSpellAtSource { src, .. } => Some((true, src)),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let first = points(&rx);
        assert_eq!(
            first,
            vec![(
                source,
                benilla_assets::coords::bevy_to_wow(Vec3::new(2.0, 3.0, -4.0))
            )]
        );
        verdict(&mut app, false, 0x28);
        app.world_mut()
            .resource_mut::<crate::target::PickOcclusion>()
            .point = Some(Vec3::splat(99.0));
        run(
            &mut app,
            if nested {
                "QueueScript('SpellStopTargeting()')"
            } else if cancel {
                "SpellStopTargeting()"
            } else {
                "ClearTarget()"
            },
        );
        assert_eq!(points(&rx), if cancel { vec![] } else { first });
    }
}

#[test]
fn late_self_start_clears_a_retry_even_with_optional_start_events_disabled() {
    let (mut app, rx) = fixture();
    run(&mut app, "CastSpellByName('Heal2050',1)");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    verdict(&mut app, false, 0x3c);
    packet(
        &mut app,
        SessionEvent::SpellStart {
            caster: 3,
            spell_id: HEAL,
            target: Some(3),
            cast_flags: 0,
            cast_time_ms: 1500,
            ammo_display_id: None,
        },
    );
    // Clear the stock in-flight slot independently: the extension must have removed the retry.
    app.insert_resource(crate::spell::PendingCast::default());
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
}

#[test]
fn a_local_refusal_is_never_remembered_as_a_server_retry() {
    let (mut app, rx) = fixture();
    app.world_mut()
        .resource_mut::<crate::spell::PendingCast>()
        .push(OTHER, None, bevy::platform::time::Instant::now());
    run(
        &mut app,
        "SetCVar('NP_QueueInstantSpells',0);CastSpellByName('Heal2050',1)",
    );
    assert!(cast_packets(&rx).is_empty());
    verdict(&mut app, false, 0x3c);
    app.insert_resource(crate::spell::PendingCast::default());
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
}

#[test]
fn retries_observe_minimum_buffer_and_keep_the_first_sends_deadline() {
    let (mut app, rx) = fixture();
    run(
        &mut app,
        "SetCVar('NP_MinBufferTimeMs',55);CastSpellByName('Heal2050',1)",
    );
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    verdict(&mut app, false, 0x61);
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
    tick(&mut app, 0.04);
    run(&mut app, "");
    assert!(cast_packets(&rx).is_empty());
    tick(&mut app, 0.02);
    run(&mut app, "");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    tick(&mut app, 1.95);
    verdict(&mut app, false, 0x3c);
    run(&mut app, "");
    assert!(
        cast_packets(&rx).is_empty(),
        "a resend cannot restart the two-second lifetime"
    );
}

#[test]
fn a_queue_pop_callback_or_preceding_script_can_cancel_a_released_retry() {
    for callback in [true, false] {
        let (mut app, rx) = fixture();
        run(&mut app, "CastSpellByName('Heal2050',1)");
        assert_eq!(cast_packets(&rx), vec![HEAL]);
        verdict(&mut app, false, 0x61);
        run(
            &mut app,
            if callback {
                "listener:HookScript('OnEvent',function() if event=='SPELL_QUEUE_EVENT' and arg1==3 then SpellStopCasting() end end)"
            } else {
                "QueueScript('SpellStopCasting()')"
            },
        );
        assert!(cast_packets(&rx).is_empty(), "callback={callback}");
    }
    // Terrain cancellation must not invalidate a released unit spell.
    let (mut app, rx) = fixture();
    run(&mut app, "CastSpellByName('Heal2050',1)");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
    verdict(&mut app, false, 0x61);
    run(&mut app, "QueueScript('SpellStopTargeting()')");
    assert_eq!(cast_packets(&rx), vec![HEAL]);
}

#[test]
fn released_player_queues_respect_nested_cancellation_and_restore_the_new_selection() {
    for cancel in [true, false] {
        let (mut app, rx) = fixture();
        with_unit_targets(&mut app);
        casting(&app, Some(300.0));
        run(&mut app, "CastSpellByName('Heal2050')");
        assert!(targets(&rx).is_empty());
        casting(&app, None);
        run(
            &mut app,
            if cancel {
                "TargetUnit('player');QueueScript('SpellStopCasting()')"
            } else {
                "TargetUnit('player')"
            },
        );
        assert_eq!(targets(&rx), if cancel { vec![] } else { vec![Some(4)] });
        assert_eq!(
            app.world().resource::<crate::target::Selection>().guid,
            Some(3)
        );
    }
}
