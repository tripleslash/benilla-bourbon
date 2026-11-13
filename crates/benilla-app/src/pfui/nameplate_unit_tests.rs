//! pfUI's plates read their unit through `nameplateN` (`modules/nameplates.lua`, `ReadName`,
//! `UpdateLevel`, the health bar), served to the stock getters as an alias; with the host lent, as
//! in game, each reads the plate's own unit.

use benilla_ui::script::UiScript;

const WOLF: u64 = 0xF130_0000_0044_0001;

fn fixture() -> bevy::prelude::App {
    let (mut app, _) = crate::pfui::test_support::fixture();
    let world = app.world_mut();
    let entity = world
        .spawn((
            crate::net::Guid(WOLF),
            crate::net::ObjectStore(benilla_protocol::ObjectFields::from_pairs(&[
                (2, 9),
                (22, 40),
                (28, 50),
                (34, 12),
            ])),
        ))
        .id();
    world
        .resource_mut::<crate::net::GuidIndex>()
        .0
        .insert(WOLF, entity);
    let mut script = world.non_send_mut::<UiScript>();
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
            key: WOLF,
            top_centre: (500., 400.),
            health: 40.,
            max_health: 50.,
            bar_colour: [1., 0., 0.],
            name: "Wolf".into(),
            level: Some(12),
            skull: false,
            level_colour: [1., 1., 0.],
            raid_icon: None,
            alpha: 255,
            lit: false,
            hovered: false,
        }],
    );
    app
}

/// An untargeted plate's unit reads its own level and health, alone and beside a second alias.
#[test]
fn a_plate_token_reads_its_own_unit_through_the_host() {
    let mut app = fixture();
    crate::script_calls::with_hosted_vm(app.world_mut(), |script| {
        script
            .expect("a VM")
            .run(
                r#"
            assert(UnitGUID('nameplate1') == '0xF130000000440001', tostring(UnitGUID('nameplate1')))
            assert(UnitExists('nameplate1'))
            assert(UnitLevel('nameplate1') == 12, tostring(UnitLevel('nameplate1')))
            assert(UnitHealth('nameplate1') == 40 and UnitHealthMax('nameplate1') == 50)
            assert(UnitIsUnit('nameplate1', '0xF130000000440001'))
            assert(not UnitIsUnit('nameplate1', 'player'))
        "#,
            )
            .unwrap();
    });
}
