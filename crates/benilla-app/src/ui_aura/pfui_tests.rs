use super::*;
use benilla_formats::SpellDisplay;

#[test]
fn a_streamed_out_group_member_keeps_the_roster_feed_while_other_units_are_removed() {
    use benilla_ui::script::UnitGuids;
    for grouped in [false, true] {
        let mut app = App::new();
        let mut script = UiScript::new().unwrap();
        benilla_pfui::install(&script).unwrap();
        script.set_unit_guids(&UnitGuids {
            player: 3,
            party: [if grouped { 5 } else { 0 }, 0, 0, 0],
            ..Default::default()
        });
        app.insert_non_send(script);
        app.add_systems(Update, super::feed_visible);
        let entity = app
            .world_mut()
            .spawn((
                crate::net::Guid(5),
                crate::net::ObjectStore(
                    benilla_protocol::ObjectFields::from_pairs(&[(2, 9), (47, 118), (95, 8)])
                        .into_created(benilla_protocol::messages::ObjectType::Unit),
                ),
            ))
            .id();
        app.update();
        let script = app.world().non_send::<UiScript>();
        script
            .run("assert(C_UnitAuras.GetAuraDataBySlot('0x0000000000000005',0).spellId==118)")
            .unwrap();
        // In the real schedule, the stock roster fallback precedes the visible-unit cleanup.
        super::unit(
            script,
            5,
            &[crate::ui_aura::AuraInput {
                slot: 0,
                spell_id: 139,
                count: 1,
                flags: 0,
                level: 0,
                raid_listed: false,
                vision_hidden: false,
            }],
            &[AuraState {
                spell_id: 139,
                helpful: true,
                ..Default::default()
            }],
            None,
        );
        app.world_mut().despawn(entity);
        app.update();
        let script = app.world().non_send::<UiScript>();
        if grouped {
            script
                .run("assert(C_UnitAuras.GetAuraDataBySlot('party1',0).spellId==139)")
                .unwrap();
        } else {
            script
                .run("assert(C_UnitAuras.GetAuraDataBySlot('0x0000000000000005',0)==nil)")
                .unwrap();
        }
    }
}

#[test]
fn overflow_projection_classifies_real_control_and_damage_auras_without_rewriting_stock_slots() {
    let data = benilla_formats::wow_data_or_skip!();
    let mut chain = benilla_formats::open_chain(&data).unwrap();
    let catalog = benilla_formats::load_spell_catalog(&mut chain).unwrap();
    for (spell_id, harmful, control) in [
        (118, true, true),
        (116, true, true),
        (589, true, false),
        (139, false, false),
        (168, false, false),
    ] {
        assert!(catalog.get(spell_id).is_some());
        let stock = AuraState {
            spell_id,
            helpful: true,
            ..Default::default()
        };
        let projected = project(5, 14, &stock, Some(&catalog));
        assert_eq!(
            (projected.state.helpful, projected.crowd_control),
            (!harmful, control),
            "spell {spell_id}"
        );
        assert_eq!(projected.slot, 5);
        assert!(
            stock.helpful,
            "the stock snapshot keeps its original slot-based classification"
        );
    }
}

#[test]
fn only_active_effects_classify_overflow_and_cancelable_buffs_keep_server_polarity() {
    let catalog = SpellCatalog::from_displays(
        vec![
            (
                1,
                SpellDisplay {
                    effect_apply_aura: [8, 12, 0], // healing and a separate stun effect
                    effect_mechanic: [0, 12, 0],
                    ..Default::default()
                },
            ),
            (
                2,
                SpellDisplay {
                    attributes: 0x0400_0000,
                    ..Default::default()
                },
            ),
        ]
        .into_iter()
        .collect(),
    );
    let state = AuraState {
        spell_id: 1,
        helpful: true,
        ..Default::default()
    };
    let healing = project(5, 8, &state, Some(&catalog));
    assert!(healing.state.helpful && !healing.crowd_control);
    let stun = project(5, 4, &state, Some(&catalog));
    assert!(!stun.state.helpful && stun.crowd_control);
    let roster = project(5, 0, &state, Some(&catalog));
    assert!(!roster.state.helpful && roster.crowd_control);
    let cancelable = AuraState {
        spell_id: 2,
        cancelable: true,
        ..state.clone()
    };
    assert!(project(5, 9, &cancelable, Some(&catalog)).state.helpful);
    let unknown = AuraState {
        spell_id: 999,
        ..state
    };
    assert!(project(5, 8, &unknown, Some(&catalog)).state.helpful);
    let negative_slot = AuraState {
        helpful: false,
        ..unknown
    };
    assert!(!project(32, 8, &negative_slot, Some(&catalog)).state.helpful);
}
