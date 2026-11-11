//! Real registered packet handlers followed by the addon-visible channel query/event stream.

use super::*;
use benilla_ui::script::{pfui_host::units, UnitGuids, UnitState};
use bevy::ecs::system::RunSystemOnce;

const REMOTE: u64 = 99;
const CHANNEL: u32 = 740;
const REPLACEMENT: u32 = 741;

fn fixture(data: &std::path::Path) -> (App, Entity) {
    let (mut app, _) = crate::pfui::test_support::fixture();
    let mut chain = benilla_formats::open_chain(data).unwrap();
    let mut spells = app.world_mut().resource_mut::<crate::ui_action::Spells>();
    spells.durations = benilla_formats::load_spell_durations(&mut chain).unwrap();
    spells.catalog = benilla_formats::SpellCatalog::from_displays(
        [CHANNEL, REPLACEMENT]
            .map(|id| {
                (
                    id,
                    benilla_formats::SpellDisplay {
                        name: format!("Channel{id}"),
                        attributes_ex: 4,
                        duration_index: 1,
                        ..Default::default()
                    },
                )
            })
            .into(),
    );
    let entity = app
        .world_mut()
        .spawn((crate::net::Guid(REMOTE), store(0)))
        .id();
    app.world_mut()
        .resource_mut::<crate::net::GuidIndex>()
        .0
        .insert(REMOTE, entity);
    let mut script = app.world_mut().non_send_mut::<UiScript>();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        target: REMOTE,
        ..Default::default()
    });
    let unit = UnitState {
        guid: REMOTE,
        exists: true,
        has_object: true,
        ..Default::default()
    };
    script.set_unit("target", Some(unit.clone()));
    units::set_unit_states(script.lua(), [(REMOTE, unit, 0)]);
    script.set_screen_size(1024., 768.);
    script
        .run("WorldFrame=CreateFrame('WorldFrame','WorldFrame'); WorldFrame:SetAllPoints()")
        .unwrap();
    script.sync_nameplates(
        benilla_ui::script::PlateGeometry {
            width: 128.,
            height: 32.,
            bar_off_x: 4.,
            bar_off_y: 4.,
            bar_width: 103.,
            bar_height: 9.,
            level_off_x: 12.,
            level_off_y: 9.,
            skull_size: 13.,
            raid_size: 26.,
            name_height: 13.,
            level_height: 11.,
            shadow_offset: 1.,
        },
        &[benilla_ui::script::PlateState {
            key: REMOTE,
            top_centre: (500., 400.),
            health: 40.,
            max_health: 50.,
            bar_colour: [1., 0., 0.],
            name: "Remote".into(),
            level: Some(20),
            skull: false,
            level_colour: [1., 1., 0.],
            raid_icon: None,
            alpha: 255,
            lit: false,
            hovered: false,
        }],
    );
    script
        .run(
            r#"
        FocusUnit('target')
        channelEvents={}
        plateStarts=0; plateStops=0
        for _, e in ipairs({'UNIT_SPELLCAST_START','UNIT_SPELLCAST_STOP',
            'UNIT_SPELLCAST_SUCCEEDED','UNIT_SPELLCAST_INTERRUPTED',
            'UNIT_SPELLCAST_CHANNEL_START','UNIT_SPELLCAST_CHANNEL_STOP'}) do
            listener:RegisterEvent(e)
        end
        listener:HookScript('OnEvent', function()
            if arg1=='nameplate1' and event=='UNIT_SPELLCAST_CHANNEL_START' then
                plateStarts=plateStarts+1
                assert(C_Spell.UnitChannelInfo(arg1))
            elseif arg1=='nameplate1' and event=='UNIT_SPELLCAST_CHANNEL_STOP' then
                plateStops=plateStops+1
                assert(not C_Spell.UnitChannelInfo(arg1))
            end
            if arg1=='target' or arg1=='focus' then
                table.insert(channelEvents,{event,arg1,arg2,arg3,C_Spell.UnitChannelInfo(arg1)})
            end
        end)
    "#,
        )
        .unwrap();
    (app, entity)
}

fn store(spell: u32) -> crate::net::ObjectStore {
    crate::net::ObjectStore(benilla_protocol::ObjectFields::from_pairs(&[
        (144, spell),
        (34, 20),
    ]))
}

fn run(app: &App, lua: &str) {
    app.world()
        .non_send::<UiScript>()
        .run(lua)
        .unwrap();
}

fn feed(app: &mut App) {
    app.world_mut().run_system_once(super::feed).unwrap();
}

fn packet(app: &mut App, event: SessionEvent) {
    crate::net::handlers::dispatch(app.world_mut(), vec![event]);
    feed(app);
}

fn start(app: &mut App, spell: u32, cast_time_ms: u32) {
    packet(
        app,
        SessionEvent::SpellStart {
            caster: REMOTE,
            spell_id: spell,
            cast_time_ms,
            target: Some(3),
            cast_flags: 0,
            ammo_display_id: None,
        },
    );
}

fn go(app: &mut App, spell: u32) {
    packet(
        app,
        SessionEvent::SpellGo {
            caster: REMOTE,
            spell_id: spell,
            cast_flags: 0,
            hits: vec![3],
            misses: vec![],
            target: Some(3),
            go_target: Some(3),
            target_mask: 0,
            source: None,
            dest: None,
            ammo_display_id: None,
            item_caster: None,
            unit_caster: REMOTE,
        },
    );
}

#[test]
fn remote_channel_waits_for_its_descriptor_and_fans_out_readable_start_and_stop_events() {
    let data = benilla_formats::wow_data_or_skip!();
    let (mut app, entity) = fixture(&data);
    start(&mut app, CHANNEL, 0);
    run(
        &app,
        "assert(C_Spell.UnitChannelInfo('target')==nil and table.getn(channelEvents)==0)",
    );
    go(&mut app, CHANNEL);
    run(
        &app,
        "assert(table.getn(channelEvents)==2); observedID=channelEvents[1][3]",
    );
    app.world_mut().entity_mut(entity).insert(store(CHANNEL));
    feed(&mut app);
    run(
        &app,
        r#"
        assert(C_Spell.UnitCastingInfo('target')==nil)
        assert(C_Spell.UnitChannelInfo('target')=='Channel740')
        assert(C_Spell.UnitChannelInfo('focus')=='Channel740')
        assert(C_Spell.UnitChannelInfo('nameplate1')=='Channel740' and plateStarts==1)
        assert(C_Spell.UnitChannelInfo('0x0000000000000063')=='Channel740')
        assert(table.getn(channelEvents)==4)
        for i=3,4 do
            assert(channelEvents[i][1]=='UNIT_SPELLCAST_CHANNEL_START')
            assert(channelEvents[i][3]==observedID and channelEvents[i][4]==740)
            assert(channelEvents[i][5]=='Channel740')
        end
        startMS=select(4,C_Spell.UnitChannelInfo('target'))
        endMS=select(5,C_Spell.UnitChannelInfo('target'))
        assert(endMS>startMS)
    "#,
    );
    feed(&mut app);
    run(&app, "assert(table.getn(channelEvents)==4)");
    // An unrelated instant START carries no evidence that the running channel ended.
    start(&mut app, 999, 0);
    run(
        &app,
        "assert(C_Spell.UnitChannelInfo('target')=='Channel740' and table.getn(channelEvents)==4)",
    );
    app.world_mut().entity_mut(entity).insert(store(0));
    feed(&mut app);
    run(
        &app,
        r#"
        assert(C_Spell.UnitChannelInfo('target')==nil and table.getn(channelEvents)==6)
        assert(plateStops==1)
        for i=5,6 do
            assert(channelEvents[i][1]=='UNIT_SPELLCAST_CHANNEL_STOP')
            assert(channelEvents[i][3]==observedID and channelEvents[i][5]==nil)
        end
    "#,
    );
}

#[test]
fn cast_prelude_transitions_to_a_channel_and_late_failures_cannot_end_a_replacement() {
    let data = benilla_formats::wow_data_or_skip!();
    let (mut app, entity) = fixture(&data);
    start(&mut app, CHANNEL, 500);
    run(
        &app,
        "assert(C_Spell.UnitCastingInfo('target')=='Channel740'); observedID=channelEvents[1][3]",
    );
    app.world_mut().entity_mut(entity).insert(store(CHANNEL));
    feed(&mut app);
    run(&app, "assert(C_Spell.UnitChannelInfo('target')==nil)");
    go(&mut app, CHANNEL);
    run(
        &app,
        r#"
        assert(C_Spell.UnitCastingInfo('target')==nil and C_Spell.UnitChannelInfo('target')=='Channel740')
        assert(table.getn(channelEvents)==8)
        for i=1,8 do assert(channelEvents[i][3]==observedID) end
    "#,
    );
    start(&mut app, REPLACEMENT, 0);
    app.world_mut()
        .entity_mut(entity)
        .insert(store(REPLACEMENT));
    feed(&mut app);
    packet(
        &mut app,
        SessionEvent::SpellFailedOther {
            caster: REMOTE,
            spell_id: CHANNEL,
        },
    );
    run(
        &app,
        "assert(C_Spell.UnitChannelInfo('target')=='Channel741')",
    );
    app.world_mut().despawn(entity);
    feed(&mut app);
    run(
        &app,
        r#"
        assert(C_Spell.UnitChannelInfo('target')==nil)
        for _,row in ipairs(channelEvents) do assert(row[1]~='UNIT_SPELLCAST_INTERRUPTED') end
    "#,
    );
}

#[test]
fn unseen_channels_have_no_invented_timing_and_pending_observations_end_with_the_session() {
    let data = benilla_formats::wow_data_or_skip!();
    let (mut app, entity) = fixture(&data);
    app.world_mut().entity_mut(entity).insert(store(CHANNEL));
    feed(&mut app);
    go(&mut app, CHANNEL);
    run(&app, "assert(C_Spell.UnitChannelInfo('target')==nil)");
    app.world_mut().entity_mut(entity).insert(store(0));
    start(&mut app, CHANNEL, 0);
    packet(
        &mut app,
        SessionEvent::SpellFailedOther {
            caster: REMOTE,
            spell_id: CHANNEL,
        },
    );
    app.world_mut().entity_mut(entity).insert(store(CHANNEL));
    feed(&mut app);
    run(&app, "assert(C_Spell.UnitChannelInfo('target')==nil)");
    for event in [
        SessionEvent::LoggedOut,
        SessionEvent::Disconnected {
            reason: "test disconnect".into(),
            end: benilla_protocol::SessionEnd::Lost,
        },
    ] {
        let (mut app, _) = fixture(&data);
        start(&mut app, CHANNEL, 0);
        packet(&mut app, event);
        let replacement = app
            .world_mut()
            .spawn((crate::net::Guid(REMOTE), store(CHANNEL)));
        let replacement = replacement.id();
        app.world_mut()
            .resource_mut::<crate::net::GuidIndex>()
            .0
            .insert(REMOTE, replacement);
        feed(&mut app);
        run(&app, "assert(C_Spell.UnitChannelInfo('target')==nil)");
        assert!(app.world().resource::<State>().remote_channels.is_empty());
    }
}
