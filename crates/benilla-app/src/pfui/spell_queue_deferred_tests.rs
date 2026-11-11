#[test]
fn pfui_cast_events_follow_a_deferred_attack_pick_and_its_actual_send() {
    let mut f = frame(false);
    spawn_mob(&mut f, &[]);
    with_strike(&mut f);
    let mut script = f.app.world_mut().non_send_mut::<UiScript>();
    benilla_pfui::install(&script).unwrap();
    script.set_action(
        1,
        Some(benilla_ui::script::ActionSlot {
            kind: 0,
            action: SINISTER_STRIKE,
            ..Default::default()
        }),
    );
    script.run("castEvents={};castListener=CreateFrame('Frame');castListener:RegisterEvent('SPELL_CAST_EVENT');castListener:SetScript('OnEvent',function() table.insert(castEvents,{arg1,arg2}) end)").unwrap();
    run(&mut f, "UseAction(1)");
    assert_eq!(
        strike_wire(&f),
        vec![("select", MOB), ("cast", MOB), ("swing", MOB)]
    );
    f.app
        .world()
        .non_send::<UiScript>()
        .run("assert(#castEvents==1 and castEvents[1][1]==1 and castEvents[1][2]==1752)")
        .unwrap();
}

#[test]
fn pfui_auto_attack_buttons_keep_the_stock_toggle_while_a_cast_is_pending() {
    let mut f = frame(true);
    let world = f.app.world_mut();
    world.insert_resource(crate::ui_action::Spells {
        catalog: benilla_formats::SpellCatalog::from_displays(
            [(
                6603,
                benilla_formats::SpellDisplay {
                    effects: [78, 0, 0],
                    ..Default::default()
                },
            )]
            .into(),
        ),
        ..crate::ui_action::Spells::empty_for_tests()
    });
    world
        .resource_mut::<crate::ui_action::ActionSlots>()
        .buttons
        .insert(
            0,
            ActionButton {
                slot: 0,
                action: 6603,
                kind: ACTION_KIND_SPELL,
            },
        );
    world.resource_mut::<crate::spell::PendingCast>().push(HEAL, None, bevy::platform::time::Instant::now());
    let mut script = world.non_send_mut::<UiScript>();
    benilla_pfui::install(&script).unwrap();
    script.set_action(
        1,
        Some(benilla_ui::script::ActionSlot {
            kind: ACTION_KIND_SPELL,
            action: 6603,
            ..Default::default()
        }),
    );
    script.run("queueEvents=0;attackListener=CreateFrame('Frame');attackListener:RegisterEvent('SPELL_QUEUE_EVENT');attackListener:SetScript('OnEvent',function() queueEvents=queueEvents+1 end)").unwrap();
    run(&mut f, "UseAction(1)");
    assert_eq!(swings(&f), (vec![OLD], false));
    f.app
        .world()
        .non_send::<UiScript>()
        .run("assert(queueEvents==0)")
        .unwrap();
}
