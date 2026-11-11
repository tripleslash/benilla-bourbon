use benilla_ui::script::{
    pfui_host::{items, units},
    FactionEntry, InvSlotView, InventorySlots, ReputationState, UiScript, UnitGuids,
};

fn watching(player: u64) -> UiScript {
    let mut script = UiScript::new().unwrap();
    script.keep_calls();
    super::install(&script).unwrap();
    script.set_unit_guids(&UnitGuids {
        player,
        ..Default::default()
    });
    script
        .run(
            r#"
        seen = {}
        local frame = CreateFrame("Frame")
        for _, name in ipairs({"UPDATE_SHAPESHIFT_FORM", "UPDATE_INVENTORY_DURABILITY", "FACTION_STANDING_CHANGED"}) do
          frame:RegisterEvent(name)
        end
        frame:SetScript("OnEvent", function(_, name, a, b, c)
          local form = name == "UPDATE_SHAPESHIFT_FORM" and GetShapeshiftFormID() or ""
          local worn = name == "UPDATE_INVENTORY_DURABILITY" and (GetInventoryItemDurability(5) or "none") or ""
          seen[#seen + 1] = name .. ":" .. tostring(a) .. ":" .. tostring(b) .. ":" .. tostring(c) .. ":" .. form .. worn
        end)
        function drain() local text = table.concat(seen, "|") seen = {} return text end
    "#,
        )
        .unwrap();
    script
}

fn form(script: &UiScript, guid: u64, form: u8) {
    units::set_raw_units(
        script.lua(),
        [(
            guid,
            units::RawUnit {
                health: Some(1),
                form,
                mount_display: 0,
                auras: Vec::new(),
                applications: Vec::new(),
            },
        )],
    );
}

fn seen(script: &UiScript) -> String {
    items::dispatch_inventory_events(script.lua());
    script.eval::<String>("return drain()").unwrap()
}

#[test]
fn form_changes_announce_once_with_the_new_form_readable() {
    let mut script = watching(3);
    assert_eq!(seen(&script), "");
    form(&script, 3, 0);
    assert_eq!(seen(&script), "", "the first reading is a baseline");
    form(&script, 3, 17);
    assert_eq!(seen(&script), "UPDATE_SHAPESHIFT_FORM:nil:nil:nil:17");
    assert_eq!(seen(&script), "");
    form(&script, 3, 0);
    assert_eq!(seen(&script), "UPDATE_SHAPESHIFT_FORM:nil:nil:nil:0");
    // Another character's form is its own baseline.
    script.set_unit_guids(&UnitGuids {
        player: 4,
        ..Default::default()
    });
    form(&script, 4, 1);
    assert_eq!(seen(&script), "");
}

#[test]
fn worn_durability_and_item_changes_coalesce_to_one_event() {
    let mut script = watching(3);
    let worn = |chest: Option<(u32, u32)>, legs: u32| {
        let mut slots: InventorySlots = Default::default();
        slots[5] = Some(InvSlotView {
            item_id: 10,
            durability: chest,
            ..Default::default()
        });
        slots[7] = Some(InvSlotView {
            item_id: legs,
            durability: Some((30, 30)),
            ..Default::default()
        });
        slots
    };
    script.set_inventory_slots(worn(Some((50, 50)), 20));
    assert_eq!(seen(&script), "", "the first snapshot is a baseline");
    script.set_inventory_slots(worn(Some((50, 50)), 20));
    assert_eq!(seen(&script), "");
    script.set_inventory_slots(worn(Some((41, 50)), 21));
    assert_eq!(seen(&script), "UPDATE_INVENTORY_DURABILITY:nil:nil:nil:41");
    script.set_inventory_slots(Default::default());
    assert_eq!(
        seen(&script),
        "UPDATE_INVENTORY_DURABILITY:nil:nil:nil:none"
    );
}

#[test]
fn standing_changes_report_the_delta_and_skip_the_login_sync() {
    let entry = |faction_id, standing, is_header| FactionEntry {
        faction_id,
        rep_list_id: faction_id,
        parent_id: 0,
        name: "Faction".into(),
        description: String::new(),
        standing,
        standing_id: 4,
        bar_min: 0,
        bar_max: 3000,
        visible: true,
        is_header,
        at_war: false,
        can_toggle_at_war: true,
        inactive: false,
    };
    let mut script = watching(3);
    let list = |script: &mut UiScript, values: &[(u32, i32)]| {
        let mut entries: Vec<_> = values.iter().map(|(id, v)| entry(*id, *v, false)).collect();
        entries.push(entry(900, 0, true));
        script.set_reputation(ReputationState {
            entries,
            watched: None,
        });
    };
    list(&mut script, &[(69, 100), (72, 500)]);
    assert_eq!(seen(&script), "", "the login sync is a baseline");
    list(&mut script, &[(69, 100), (72, 500)]);
    assert_eq!(seen(&script), "");
    list(&mut script, &[(69, 175), (72, 475)]);
    assert_eq!(
        seen(&script),
        "FACTION_STANDING_CHANGED:69:175:75:|FACTION_STANDING_CHANGED:72:475:-25:"
    );
    // A relisted table (another slot appears) is a new baseline, never a gain.
    list(&mut script, &[(69, 300), (72, 475), (81, 10)]);
    assert_eq!(seen(&script), "");
    list(&mut script, &[(69, 300), (72, 475), (81, 60)]);
    assert_eq!(seen(&script), "FACTION_STANDING_CHANGED:81:60:50:");
}
