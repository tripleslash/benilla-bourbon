//! The engine half of ClassicAPI's macro slash commands: what each verb hands the host.

use benilla_ui::script::{
    pfui_host::{
        attributes::ClickAction,
        items::{self, Instance},
        macro_verbs, units,
    },
    ScriptCall, UiScript, UnitGuids, UnitState,
};

fn fixture() -> UiScript {
    let mut script = UiScript::new().unwrap();
    script.keep_calls();
    super::install(&script).unwrap();
    crate::spell::Catalog(Default::default())
        .install(script.lua())
        .unwrap();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        target: 4,
        ..Default::default()
    });
    let unit = |guid: u64, name: &str| {
        (
            guid,
            UnitState {
                guid,
                exists: true,
                has_object: true,
                name: Some(name.into()),
                ..Default::default()
            },
            0,
        )
    };
    units::set_unit_states(script.lua(), [unit(4, "Feralia"), unit(9, "Feral")]);
    items::set_instances(
        script.lua(),
        [(
            (0, 1),
            Instance {
                guid: 0x4000000000000001,
                item_id: 12345,
                name: Some("An Item of Testing".into()),
                count: 1,
                uses: 0,
            },
        )],
    );
    script
}

#[test]
fn unit_tokens_are_told_from_names_and_a_name_finds_its_unit() {
    let script = fixture();
    script
        .run(
            r#"
        for _, token in ipairs({"player", "party1target", "focus", "mark3", "0x0000000000000009"}) do
          assert(IsUnitToken(token) == true, token)
        end
        for _, name in ipairs({"Feral", "Markus", ""}) do
          assert(IsUnitToken(name) == false, name)
        end
        assert(IsUnitToken(5) == false and IsUnitToken() == false)
        -- A whole name wins over the start of another; a unit no token names reads as its guid.
        assert(UnitTokenFromName("feral") == "0x0000000000000009", UnitTokenFromName("feral"))
        assert(UnitTokenFromName("Ferali") == "target")
        assert(UnitTokenFromName("Fer", true) == nil and UnitTokenFromName("Nobody") == nil)
        assert(not pcall(UnitTokenFromName, nil))
    "#,
        )
        .unwrap();
}

#[test]
fn the_macro_verbs_hand_the_host_their_casts_attacks_and_uses() {
    let mut script = fixture();
    script
        .run(
            r#"
        assert(C_Spell.CastAtUnit(133, "target") == true)
        assert(C_Spell.CastAtUnit(133, "player", false) == true)
        assert(CastSpellNoToggle("!133") == true)
        assert(C_Spell.CastAtUnit(133, "party1") == false)
        assert(not pcall(C_Spell.CastAtUnit, 133, "nobody"))
        assert(C_Spell.CastAtCursor(133) == false)
        StartAttack("target")
        StartAttack("Feral")
        StopAttack()
        assert(C_Item.UseAtUnit("An Item of Testing", "target") == true)
        assert(C_Item.UseAtUnit(99999, "target") == false)
    "#,
        )
        .unwrap();
    let cast = |target, place_ground, no_toggle| {
        ScriptCall::PfuiClick(ClickAction::CastAt {
            spell_id: 133,
            target,
            place_ground,
            no_toggle,
        })
    };
    let calls = script.take_script_calls();
    assert_eq!(
        calls[..4],
        [
            cast(Some(4), true, false),
            cast(Some(3), false, false),
            cast(None, true, true),
            cast(None, false, false),
        ]
    );
    assert!(matches!(
        calls[4..6],
        [
            ScriptCall::PfuiClick(ClickAction::StartAttack {
                target: Some(4),
                ..
            }),
            ScriptCall::PfuiClick(ClickAction::StartAttack { target: None, .. }),
        ]
    ));
    assert_eq!(
        calls[6..],
        [
            ScriptCall::PfuiClick(ClickAction::StopAttack),
            ScriptCall::PfuiClick(ClickAction::Item {
                guid: 0x4000000000000001,
                target: Some(4),
            }),
        ]
    );

    // With ground under the cursor, the `AtCursor` verbs place there.
    macro_verbs::set_cursor_point(script.lua(), Some([1.0, 2.0, 3.0]));
    script
        .run("assert(C_Spell.CastAtCursor(133) and C_Item.UseAtCursor(12345))")
        .unwrap();
    assert_eq!(
        script.take_script_calls(),
        [
            ScriptCall::PfuiClick(ClickAction::GroundSpell {
                spell_id: 133,
                point: [1.0, 2.0, 3.0],
            }),
            ScriptCall::PfuiClick(ClickAction::ItemAtPoint {
                guid: 0x4000000000000001,
                point: [1.0, 2.0, 3.0],
            }),
        ]
    );
}

#[test]
fn an_items_use_spell_and_a_click_frame_resolve() {
    let script = fixture();
    items::set_use_spell(script.lua(), 12345, 8690);
    script
        .run(
            r#"
        assert(select(2, C_Item.GetItemSpell(12345)) == 8690)
        assert(select(2, C_Item.GetItemSpell("an item of testing")) == 8690)
        assert(C_Item.GetItemSpell(54321) == nil and GetItemSpell == C_Item.GetItemSpell)
        assert(C_Item.IsEquippableItem(54321) == false)
        local button = CreateFrame("Button", "ClickMe")
        assert(GetClickFrame("ClickMe") == button)
        ClickAlias = button
        assert(GetClickFrame("ClickAlias") == nil and GetClickFrame("Nothing") == nil)
    "#,
        )
        .unwrap();
}
