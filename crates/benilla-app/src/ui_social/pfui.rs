//! pfUI's friend guids (`C_FriendList` by name): the friend slots' names and guids, pushed to the
//! extension's table whenever the social state changes.

use benilla_ui::script::UiScript;
use bevy::prelude::*;

use super::SocialState;

pub(crate) fn feed_friend_guids(
    script: Option<NonSend<UiScript>>,
    social: Option<Res<SocialState>>,
) {
    let (Some(script), Some(social)) = (script, social) else {
        return;
    };
    if !social.is_changed() {
        return;
    }
    let rows = social
        .slots
        .iter()
        .filter_map(|slot| slot.name.as_deref().map(|name| (name, slot.guid)));
    if let Err(error) = benilla_ui::script::pfui_host::social::set_friend_guids(script.lua(), rows)
    {
        benilla_ui::script::pfui_host::report_error(
            script.lua(),
            (format!("pfUI friend GUIDs: {error}")).to_string(),
        );
    }
}
