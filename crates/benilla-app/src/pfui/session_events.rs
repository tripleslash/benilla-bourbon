//! Events for things the session reports that the stock client keeps silent: a confirmed
//! hearthstone bind and raw mouse buttons.

use crate::net::NetHandlerApp;
use benilla_protocol::{SessionEvent, SessionEventKind};
use benilla_ui::script::{pfui_host, ScriptValue, UiScript};
use bevy::prelude::*;

#[derive(Resource, Default)]
struct Bound(bool);

pub(super) fn register(app: &mut App) {
    app.init_resource::<Bound>();
    // SMSG_PLAYERBOUND is the innkeeper's confirmation; the bind-point resync on a map change
    // arrives without it.
    app.net_handler(SessionEventKind::PlayerBound, bound);
    app.net_handler(SessionEventKind::Disconnected, bound);
    app.net_handler(SessionEventKind::LoggedOut, bound);
    app.add_systems(
        Update,
        feed.in_set(crate::ui_script::UiFeed)
            .after(crate::ui_unit::UnitFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
}

fn bound(In(event): In<SessionEvent>, mut pending: ResMut<Bound>) {
    match event {
        SessionEvent::PlayerBound { .. } => pending.0 = true,
        SessionEvent::Disconnected { .. } | SessionEvent::LoggedOut => pending.0 = false,
        _ => (),
    }
}

fn name(button: MouseButton) -> Option<&'static str> {
    Some(match button {
        MouseButton::Left => "LeftButton",
        MouseButton::Right => "RightButton",
        MouseButton::Middle => "MiddleButton",
        MouseButton::Back => "Button4",
        MouseButton::Forward => "Button5",
        MouseButton::Other(_) => return None,
    })
}

fn feed(
    script: Option<NonSend<UiScript>>,
    mut pending: ResMut<Bound>,
    buttons: Option<Res<ButtonInput<MouseButton>>>,
) {
    let Some(script) = script else { return };
    if std::mem::take(&mut pending.0) {
        pfui_host::fire_pfui_event(script.lua(), "HEARTHSTONE_BOUND", Vec::new());
    }
    let Some(buttons) = buttons else { return };
    let pressed: Vec<_> = buttons.get_just_pressed().copied().collect();
    let released: Vec<_> = buttons.get_just_released().copied().collect();
    for (event, changed) in [
        ("GLOBAL_MOUSE_DOWN", pressed),
        ("GLOBAL_MOUSE_UP", released),
    ] {
        let mut names: Vec<_> = changed.into_iter().filter_map(name).collect();
        names.sort_unstable();
        for button in names {
            pfui_host::fire_pfui_event(script.lua(), event, vec![ScriptValue::Str(button.into())]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    #[test]
    fn a_confirmed_bind_and_raw_buttons_reach_lua_once_and_a_session_end_drops_the_bind() {
        let mut app = crate::game_plugins::schedule_tests::headless_client();
        let script = UiScript::new().unwrap();
        benilla_pfui::install(&script).unwrap();
        script
            .run(
                r#"
            seen = {}
            local frame = CreateFrame("Frame")
            frame:RegisterEvent("HEARTHSTONE_BOUND")
            frame:RegisterEvent("GLOBAL_MOUSE_DOWN")
            frame:RegisterEvent("GLOBAL_MOUSE_UP")
            frame:SetScript("OnEvent", function(_, name, button)
              seen[#seen + 1] = name .. ":" .. tostring(button)
            end)
        "#,
            )
            .unwrap();
        app.world_mut().insert_non_send(script);
        app.world_mut().init_resource::<ButtonInput<MouseButton>>();
        let drain = |app: &mut App| {
            app.world_mut().run_system_once(feed).unwrap();
            app.world()
                .non_send::<UiScript>()
                .eval::<String>("local text = table.concat(seen, '|') seen = {} return text")
                .unwrap()
        };
        let bind = SessionEvent::PlayerBound {
            binder: 9,
            area: 12,
        };
        app.world_mut()
            .run_system_once_with(bound, bind.clone())
            .unwrap();
        assert_eq!(drain(&mut app), "HEARTHSTONE_BOUND:nil");
        assert_eq!(drain(&mut app), "");
        app.world_mut().run_system_once_with(bound, bind).unwrap();
        app.world_mut()
            .run_system_once_with(bound, SessionEvent::LoggedOut)
            .unwrap();
        assert_eq!(drain(&mut app), "");
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Right);
        assert_eq!(drain(&mut app), "GLOBAL_MOUSE_DOWN:RightButton");
        let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
        buttons.clear();
        buttons.release(MouseButton::Right);
        assert_eq!(drain(&mut app), "GLOBAL_MOUSE_UP:RightButton");
    }
}
