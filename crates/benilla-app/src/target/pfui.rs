//! Publish attribute-bound frame hovers through the ordinary world hover path.

use benilla_ui::script::UiScript;
use bevy::prelude::*;

pub(crate) fn register(app: &mut App) {
    app.add_systems(
        Update,
        frame_hover
            .in_set(super::TargetUpdate)
            .after(super::hover::update_hover)
            .before(super::hover::update_hovered_object),
    );
    app.add_systems(
        Update,
        swing_ranges
            .in_set(crate::ui_script::UiFeed)
            .run_if(crate::ui_script::ingame_ui_up),
    );
}

/// The ranged weapons' span, centre to centre: Auto Shot's dead zone below 8 yd, the 35 yd of a
/// bow, gun or crossbow above.
const RANGED_SPAN: std::ops::RangeInclusive<f32> = 8.0..=35.0;

/// `C_SwingTimer`'s answers for the current target, one per `Enum.PlayerSwingType`: `None` with no
/// attackable target (alive, and `CanAttack` from the player) or no weapon for the type (an
/// off-hand or ranged attack time of 0); else the distance against the melee reach
/// ([`super::cursor_mode::melee_reach`]) or the ranged span.
fn swing_ranges(
    script: Option<NonSend<UiScript>>,
    selection: Res<super::Selection>,
    relations: super::RelationWorld,
    me: Query<(&crate::net::ObjectStore, &Transform), With<crate::net::SelfPlayer>>,
    units: Query<(&crate::net::Guid, &crate::net::ObjectStore, &Transform)>,
) {
    let Some(script) = script else { return };
    let answers = (|| {
        let (my_store, my_place) = me.single().ok()?;
        let (guid, store, place) = units.get(selection.target?).ok()?;
        let unit = relations.unit_of(guid.0, store);
        let rel = relations.get();
        if !store.is_unit()
            || !super::scan::attack_target_valid(&rel, relations.me().as_ref(), &unit, store)
        {
            return None;
        }
        let d = my_place.translation.distance(place.translation);
        let melee = d <= super::cursor_mode::melee_reach(store, my_store);
        let armed = |time: Option<u32>| time.is_some_and(|t| t > 0);
        Some([
            Some(melee),
            armed(my_store.0.unit_base_attack_time(1)).then_some(melee),
            armed(my_store.0.unit_ranged_attack_time()).then(|| RANGED_SPAN.contains(&d)),
        ])
    })();
    benilla_pfui::swing_range::set(script.lua(), answers.unwrap_or([None; 3]));
}

fn frame_hover(
    script: Option<NonSend<UiScript>>,
    rig: Res<crate::player::CameraControl>,
    selection: Res<super::Selection>,
    mut hover: ParamSet<(crate::ui_unit::UnitTokens, ResMut<super::Hovered>)>,
) {
    if rig.is_looking() {
        return;
    }
    let Some(script) = script else { return };
    let Ok(Some(token)) = benilla_ui::script::pfui_host::attributes::hovered_unit(script.lua())
    else {
        return;
    };
    let unit = if let Some(guid) =
        benilla_ui::script::pfui_host::units::custom_guid(script.lua(), &token)
    {
        guid.and_then(|guid| hover.p0().held(guid))
    } else {
        hover.p0().resolve(&token, &selection)
    };
    if let Some((entity, guid)) = unit {
        *hover.p1() = super::Hovered {
            target: Some(entity),
            guid: Some(guid),
            distance: 0.0,
            ..Default::default()
        };
    }
}
