use super::*;
use bevy::ecs::system::RunSystemOnce;

#[test]
fn pfui_focus_survives_reload_but_logout_and_disconnect_clear_session_memory() {
    for event in [
        SessionEvent::LoggedOut,
        SessionEvent::Disconnected {
            reason: "test disconnect".into(),
            end: benilla_protocol::SessionEnd::Lost,
        },
    ] {
        let mut app = crate::game_plugins::schedule_tests::headless_client();
        let mut script = UiScript::new().unwrap();
        crate::pfui::enable_vm(app.world(), &script);
        benilla_pfui::install(&script).unwrap();
        script.set_unit_guids(&benilla_ui::script::UnitGuids {
            player: 3,
            ..Default::default()
        });
        script.set_unit(
            "player",
            Some(benilla_ui::script::UnitState {
                guid: 3,
                exists: true,
                ..Default::default()
            }),
        );
        script.run("FocusUnit('player')").unwrap();
        app.world_mut().insert_non_send(script);
        // The same lifecycle hook that attaches memory to a replacement UI VM.
        let replacement = UiScript::new().unwrap();
        crate::pfui::enable_vm(app.world(), &replacement);
        benilla_pfui::install(&replacement).unwrap();
        replacement
            .run("assert(UnitGUID('focus')=='0x0000000000000003')")
            .unwrap();
        app.world_mut().insert_non_send(replacement);
        app.world_mut()
            .run_system_once_with(super::receive, event)
            .unwrap();
        app.world()
            .non_send::<UiScript>()
            .run("assert(UnitGUID('focus')==nil)")
            .unwrap();
        let next_character = UiScript::new().unwrap();
        crate::pfui::enable_vm(app.world(), &next_character);
        benilla_pfui::install(&next_character).unwrap();
        next_character
            .run("assert(UnitGUID('focus')==nil)")
            .unwrap();
    }
}
