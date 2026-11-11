//! Feed visible template identities and the complete known-spell set, including hidden talents.

use benilla_ui::script::{pfui_host::units, UiScript};
use bevy::prelude::*;

pub(super) fn register(app: &mut App) {
    app.add_systems(Update, (feed, speeds).in_set(super::InputSnapshots));
}

/// Our own mover's speed follows its move flags through the reference's selector; a remote
/// unit's is the speed its motion is being advanced with.
fn speeds(
    script: Option<NonSend<UiScript>>,
    player: Option<Res<crate::player::Player>>,
    units: Query<(
        &crate::net::Guid,
        &crate::net::UnitSpeeds,
        Option<&crate::net::RemoteMotion>,
        Has<crate::net::SelfPlayer>,
    )>,
) {
    let Some(script) = script else { return };
    benilla_ui::script::pfui_host::player_stats::set_speeds(
        script.lua(),
        units.iter().map(|(guid, speeds, motion, own)| {
            let current = if own {
                player.as_ref().map_or(0.0, |player| {
                    crate::net::current_speed(&speeds.0, player.move_flags())
                })
            } else {
                motion.map_or(0.0, |motion| motion.speed)
            };
            (guid.0, [current, speeds.0.run, speeds.0.swim])
        }),
    );
}

fn feed(
    script: Option<NonSend<UiScript>>,
    objects: Query<(&crate::net::Guid, Ref<crate::net::ObjectStore>)>,
    self_q: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
    mut removed: RemovedComponents<crate::net::ObjectStore>,
    actions: Res<crate::ui_action::PlayerActions>,
    mut memo: Local<crate::ui_script::VmMemo<bool>>,
) {
    let Some(script) = script else { return };
    let had_removed = removed.read().count() != 0;
    let fresh = !*memo.get(&script);
    let player = self_q.single().ok();
    benilla_ui::script::pfui_host::player_stats::set_damage(
        script.lua(),
        std::array::from_fn(|school| {
            player.map_or(0, |store| {
                i64::from(
                    store
                        .0
                        .player_mod_damage_done_pos(school as u8)
                        .unwrap_or(0),
                ) + i64::from(
                    store
                        .0
                        .player_mod_damage_done_neg(school as u8)
                        .unwrap_or(0),
                )
            })
        }),
    );
    if fresh || actions.is_changed() {
        units::set_known_spells(script.lua(), actions.spells.iter().copied());
    }
    if fresh || had_removed || objects.iter().any(|(_, object)| object.is_changed()) {
        // UpdateFields_1_12_1.h: UNIT_MOD_CAST_SPEED follows UNIT_FIELD_BYTES_1 (138).
        benilla_ui::script::pfui_host::player_stats::set_cast_speeds(
            script.lua(),
            objects.iter().filter_map(|(guid, object)| {
                object
                    .is_unit()
                    .then(|| {
                        object
                            .0
                            .raw_fields()
                            .take_while(|(index, _)| *index <= 139)
                            .find_map(|(index, bits)| {
                                (index == 139).then_some((guid.0, f32::from_bits(bits)))
                            })
                    })
                    .flatten()
            }),
        );
        units::set_raw_units(
            script.lua(),
            objects
                .iter()
                .filter(|(_, object)| object.is_unit())
                .map(|(guid, object)| {
                    let base = benilla_protocol::messages::field::FIELD_UNIT_AURAAPPLICATIONS;
                    let mut words = [0u32; 12];
                    for (index, word) in object.0.raw_fields() {
                        if (base..base + 12).contains(&index) {
                            words[(index - base) as usize] = word;
                        }
                    }
                    (
                        guid.0,
                        units::RawUnit {
                            health: object.0.unit_health(),
                            form: object.0.unit_shapeshift_form(),
                            mount_display: object.0.unit_mount_display_id(),
                            auras: object.0.unit_aura_ids().collect(),
                            applications: (0u16..48)
                                .map(|slot| {
                                    let word = words[(slot / 4) as usize];
                                    ((word >> ((slot % 4) * 8)) & 255) as u8
                                })
                                .collect(),
                        },
                    )
                }),
        );
        units::set_creature_ids(
            script.lua(),
            objects.iter().filter_map(|(guid, object)| {
                let high = guid.0 >> 48;
                (high == 0xf130 || high == 0xf140)
                    .then(|| object.0.object_entry().map(|id| (guid.0, id)))
                    .flatten()
            }),
        );
        units::set_powers(
            script.lua(),
            objects.iter().filter_map(|(guid, object)| {
                let high = guid.0 >> 48;
                if high != 0 && high != 0xf130 && high != 0xf140 {
                    return None;
                }
                object.0.unit_bytes_0().map(|_| {
                    (
                        guid.0,
                        units::Power {
                            kind: object.0.unit_power_type(),
                            current: std::array::from_fn(|index| object.0.unit_power(index as u8)),
                            maximum: std::array::from_fn(|index| {
                                object.0.unit_max_power(index as u8)
                            }),
                        },
                    )
                })
            }),
        );
    }
    *memo.get(&script) = true;
}
