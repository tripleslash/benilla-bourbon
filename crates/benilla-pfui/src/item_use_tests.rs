use benilla_ui::script::{
    pfui_host::{
        attributes::ClickAction,
        items::{self, Instance},
    },
    ContainerSlot, ContainerState, ScriptCall, UiScript, UnitGuids, EQUIPMENT_BAG,
};

fn fixture() -> UiScript {
    let mut script = UiScript::new().unwrap();
    script.keep_calls();
    super::install(&script).unwrap();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        target: 4,
        ..Default::default()
    });
    script.set_container(
        0,
        Some(ContainerState {
            num_slots: 2,
            slots: [(
                1,
                ContainerSlot {
                    item_id: 12345,
                    count: 1,
                    ..Default::default()
                },
            )]
            .into(),
            ..Default::default()
        }),
    );
    items::set_instances(
        script.lua(),
        [
            (
                (0, 1),
                Instance {
                    guid: 0x4000000000000001,
                    item_id: 12345,
                    name: Some("An Item of Testing".into()),
                    count: 1,
                    uses: 0,
                },
            ),
            (
                (1, 1),
                Instance {
                    guid: 0x4000000000000002,
                    item_id: 12345,
                    name: Some("An Item of Testing".into()),
                    count: 1,
                    uses: 0,
                },
            ),
            (
                (EQUIPMENT_BAG, 13),
                Instance {
                    guid: 0x4000000000000003,
                    item_id: 56789,
                    name: Some("A Trinket".into()),
                    count: 1,
                    uses: 0,
                },
            ),
            (
                (-1, 1),
                Instance {
                    guid: 0x4000000000000004,
                    item_id: 98765,
                    name: Some("Bank Only".into()),
                    count: 1,
                    uses: 0,
                },
            ),
        ],
    );
    script
}

#[test]
fn item_references_bind_the_first_carried_instance_or_the_exact_location() {
    for (input, guid) in [
        ("12345", 0x4000000000000001),
        ("'item:12345'", 0x4000000000000001),
        (
            "'|cffffffff|Hitem:12345:0:0:0|h[An Item of Testing]|h|r'",
            0x4000000000000001,
        ),
        ("'AN ITEM OF TESTING'", 0x4000000000000001),
        ("{bagID=1,slotIndex=1}", 0x4000000000000002),
        ("'0x4000000000000002'", 0x4000000000000002),
        ("{equipmentSlotIndex=13}", 0x4000000000000003),
    ] {
        let mut script = fixture();
        script
            .run(&format!(
                "assert(C_Item.UseItemByName({input},'player')==nil)"
            ))
            .unwrap();
        assert_eq!(
            script.take_script_calls(),
            vec![ScriptCall::PfuiClick(ClickAction::Item {
                guid,
                target: Some(3)
            })],
            "{input}"
        );
    }
}

#[test]
fn invalid_missing_bank_and_locked_items_cannot_invent_an_item_use() {
    for input in [
        "nil",
        "''",
        "{}",
        "{bagID='invalid',slotIndex=1}",
        "{bagID=0,slotIndex=0}",
        "{bagID=-1,slotIndex=1}",
        "98765",
        "'0x4000000000000004'",
        "true",
    ] {
        let mut script = fixture();
        script
            .run(&format!("C_Item.UseItemByName({input},'player')"))
            .unwrap();
        assert!(script.take_script_calls().is_empty(), "{input}");
    }
    let mut script = fixture();
    script.run("C_Item.UseItemByName(12345,'party3')").unwrap();
    assert!(script.take_script_calls().is_empty());
    script.set_container(
        0,
        Some(ContainerState {
            num_slots: 2,
            slots: [(
                1,
                ContainerSlot {
                    item_id: 12345,
                    count: 1,
                    guid: 0x4000_0000_0000_0001,
                    ..Default::default()
                },
            )]
            .into(),
            ..Default::default()
        }),
    );
    script.lock_item(0x4000_0000_0000_0001);
    script.run("C_Item.UseItemByName(12345)").unwrap();
    assert!(
        script.take_script_calls().is_empty(),
        "a locked first match does not silently use the second stack"
    );
}

#[test]
fn item_clicks_capture_the_frame_unit_and_keep_the_documented_bag_click_path() {
    let mut script = fixture();
    script.run("button=CreateFrame('Button');button:SetAttribute('type','item');button:SetAttribute('unit','player');button:SetAttribute('item',12345);button:Click('LeftButton')").unwrap();
    assert_eq!(
        script.take_script_calls(),
        vec![ScriptCall::PfuiClick(ClickAction::Item {
            guid: 0x4000000000000001,
            target: Some(3)
        })]
    );
    for attributes in ["button:SetAttribute('item','0 1')", "button:SetAttribute('item',nil);button:SetAttribute('bag',0);button:SetAttribute('slot',1)"] {
        script.run(&format!("{attributes};button:Click('LeftButton')")).unwrap();
        let calls = script.take_script_calls();
        assert!(
            matches!(calls.as_slice(), [ScriptCall::UseContainerItem(u)] if u.bag == 0 && u.slot == 1),
            "{calls:?}"
        );
    }
}

#[test]
fn item_selection_tokens_defer_to_the_ordered_engine_but_stock_vms_stay_unchanged() {
    let mut script = fixture();
    script
        .run("C_Item.UseItemByName(12345,'TARGETTARGET')")
        .unwrap();
    assert_eq!(
        script.take_script_calls(),
        vec![ScriptCall::PfuiClick(ClickAction::TargetItem {
            guid: 0x4000000000000001,
            hops: 1
        })]
    );
    UiScript::new().unwrap().run("assert(C_Item==nil)").unwrap();
}
