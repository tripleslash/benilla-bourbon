use benilla_ui::script::UiScript;
use std::collections::BTreeMap;

fn extended() -> UiScript {
    let mut script = UiScript::new().unwrap();
    // A VM with no host keeps its calls, which the tests read back.
    script.keep_calls();
    super::install(&script).unwrap();
    crate::tests::load_tooltip_template(&script);
    script
}

fn spell(name: &str, rank: &str, aura: f64) -> crate::spell::Spell {
    let mut effects = vec![BTreeMap::new(); 3];
    effects[0].insert("effect", if aura == 0.0 { 2.0 } else { 6.0 });
    effects[0].insert("auraName", aura);
    crate::spell::Spell {
        name: name.into(),
        rank: rank.into(),
        icon: Some(format!("Interface\\Icons\\{name}")),
        dispel: 0,
        attributes: 0,
        reagents: Vec::new(),
        effects,
        cast_time: 0,
        min_range: 0.0,
        max_range: 0.0,
        cost: 0,
        power_type: 0,
        funnel: false,
        swing: None,
    }
}

fn book(script: &mut UiScript) {
    use benilla_ui::script::{PetBookState, SpellBookState, SpellSlotView};
    let slot = |id, name: &str, rank: &str| SpellSlotView {
        spell_id: id,
        name: name.into(),
        rank: (!rank.is_empty()).then(|| rank.into()),
        texture: Some(format!("Interface\\Icons\\{name}")),
        ..Default::default()
    };
    script.set_spellbook(SpellBookState {
        slots: vec![
            slot(133, "Fireball", "Rank 1"),
            slot(2580, "Find Minerals", ""),
            slot(143, "Fireball", "Rank 2"),
            slot(2383, "Find Herbs", ""),
        ],
        ..Default::default()
    });
    script.set_pet_book(PetBookState {
        slots: vec![slot(2649, "Growl", "Rank 1")],
        ..Default::default()
    });
    crate::spell::Catalog(BTreeMap::from([
        (133, spell("Fireball", "Rank 1", 0.0)),
        (143, spell("Fireball", "Rank 2", 0.0)),
        (2580, spell("Find Minerals", "", 45.0)),
        (2383, spell("Find Herbs", "", 45.0)),
        (2649, spell("Growl", "Rank 1", 0.0)),
    ]))
    .install(script.lua())
    .unwrap();
}

#[test]
fn modifier_rollup_and_literal_trim_follow_the_documented_contract() {
    let mut script = extended();
    script
        .run(
            r#"
        assert(IsModifierKeyDown() == nil)
        assert(strtrim("  hello \t\n") == "hello" and strtrim("xxhelloxx", "x") == "hello")
        assert(strtrim("[a]", "[]") == "a" and strtrim("%a%", "%") == "a")
        assert(strtrim("   ") == "" and strtrim("") == "" and strtrim(12) == "12")
        assert(not pcall(strtrim, nil))
    "#,
        )
        .unwrap();
    script.set_modifiers(false, false, true);
    script.run("assert(IsModifierKeyDown() == 1)").unwrap();
}

#[test]
fn item_id_companions_read_the_slot_the_stock_link_names() {
    use benilla_ui::script::{MerchantItem, MerchantState};
    let mut script = extended();
    script.set_merchant(Some(MerchantState {
        items: vec![
            MerchantItem {
                item_id: 4540,
                link: Some("|cffffffff|Hitem:4540:0:0:0|h[Tough Hunk of Bread]|h|r".into()),
                ..Default::default()
            },
            MerchantItem::default(),
        ],
        ..Default::default()
    }));
    script
        .run(
            r#"
        assert(GetMerchantItemID(1) == 4540)
        assert(GetMerchantItemID(2) == nil and GetMerchantItemID(9) == nil)
        assert(GetLootSlotItemID(1) == nil and GetLootRollItemID(1) == nil)
        assert(GetQuestItemID("reward", 1) == nil and GetQuestLogItemID("choice", 1) == nil)
        assert(GetTradeSkillItemID(1) == nil and GetTradeSkillReagentItemID(1, 1) == nil)
        assert(GetCraftReagentItemID(1, 1) == nil)
    "#,
        )
        .unwrap();
}

#[test]
fn spellbook_slot_lookup_prefers_the_player_book_and_rejects_unknown_ids() {
    let mut script = extended();
    book(&mut script);
    script
        .run(
            r#"
        local slot, kind = FindSpellBookSlotByID(143)
        assert(slot == 3 and kind == "spell")
        slot, kind = FindSpellBookSlotByID(2649)
        assert(slot == 1 and kind == "pet")
        assert(FindSpellBookSlotByID(999) == nil and FindSpellBookSlotByID(0) == nil)
        assert(FindSpellBookSlotByID("143") == 3 and FindSpellBookSlotByID(nil) == nil)
    "#,
        )
        .unwrap();
}

#[test]
fn macro_spell_reads_directives_then_the_first_cast_and_only_known_spells() {
    use benilla_ui::script::{MacroState, MacroView};
    let mut script = extended();
    book(&mut script);
    // The table is kept sorted by name, as 1.12.1's is: the names keep the bodies in this order.
    let view = |n: u8, body: &str| MacroView {
        name: format!("m{n}"),
        body: body.into(),
        ..Default::default()
    };
    script.set_macros(MacroState::from_scopes(
        vec![
            view(1, "/cast Fireball(Rank 1)"),
            view(2, "/say hi\n/cast Fireball"),
            view(3, "#showtooltip Find Herbs\n/cast Fireball"),
            view(4, "/script CastSpellByName(\"Fireball\")"),
            view(5, "/castsequence reset=5 Find Minerals, Fireball"),
            view(6, "/say nothing"),
            view(7, "#showtooltip Linen Bandage\n/cast Fireball"),
            view(8, "#showtooltip\n/cast Unknown Spell"),
            view(9, "/cast [mod:alt] Fireball"),
        ],
        Vec::new(),
    ));
    script
        .run(
            r#"
        local function same(slot, name, rank, id)
          local n, r, i = GetMacroSpell(slot)
          assert(n == name and r == rank and i == id, slot .. ": " .. tostring(n) .. tostring(r) .. tostring(i))
        end
        same(1, "Fireball", "Rank 1", 133)
        same(2, "Fireball", "Rank 2", 143)
        same(3, "Find Herbs", "", 2383)
        same(4, "Fireball", "Rank 2", 143)
        same(5, "Find Minerals", "", 2580)
        for slot = 6, 10 do assert(GetMacroSpell(slot) == nil, slot) end
    "#,
        )
        .unwrap();
}

#[test]
fn tracking_lists_book_spells_with_a_tracking_aura_and_casts_the_chosen_slot() {
    use benilla_ui::script::ScriptCall;
    let mut script = extended();
    book(&mut script);
    script
        .run(
            r#"
        assert(GetNumTrackingTypes() == 2)
        local name, texture, active, category, id = GetTrackingInfo(1)
        assert(name == "Find Minerals" and texture == "Interface\\Icons\\Find Minerals")
        assert(active == false and category == "spell" and id == 2580)
        name, _, _, _, id = GetTrackingInfo(2)
        assert(name == "Find Herbs" and id == 2383)
        assert(GetTrackingInfo(3) == nil and GetTrackingInfo(0) == nil)
        SetTracking(9)
    "#,
        )
        .unwrap();
    assert!(script.take_script_calls().is_empty());
    script.run("SetTracking(2)").unwrap();
    assert!(matches!(
        script.take_script_calls().as_slice(),
        [ScriptCall::CastSpell {
            spell_id: 2383,
            on_self: false,
            may_cast: true
        }]
    ));
}

#[test]
fn faction_lookup_by_id_separates_met_unmet_and_absent_factions() {
    use benilla_ui::script::{FactionEntry, ReputationState};
    let entry = |faction_id, rep_list_id, visible, is_header| FactionEntry {
        faction_id,
        rep_list_id,
        parent_id: 0,
        name: format!("Faction {faction_id}"),
        description: "About".into(),
        standing: 4200,
        standing_id: 5,
        bar_min: 3000,
        bar_max: 9000,
        visible,
        is_header,
        at_war: false,
        can_toggle_at_war: true,
        inactive: false,
    };
    let mut script = extended();
    script.set_reputation(ReputationState {
        entries: vec![
            entry(69, 1, true, false),
            entry(574, 2, false, false),
            entry(469, 3, true, true),
        ],
        watched: Some(1),
    });
    script
        .run(
            r#"
        local name, text, rank, low, high, value, war, toggle, header, collapsed, watched = GetFactionInfoByID(69)
        assert(name == "Faction 69" and text == "About" and rank == 5 and low == 3000 and high == 9000)
        assert(value == 4200 and war == nil and toggle == 1 and header == nil and collapsed == nil and watched == 1)
        name, _, rank, low, high, value, war, toggle = GetFactionInfoByID(574)
        assert(name == "Faction 574" and rank == 4 and low == 0 and high == 3000 and value == 0)
        assert(war == nil and toggle == nil)
        assert(GetFactionInfoByID(469) == nil and GetFactionInfoByID(1) == nil and GetFactionInfoByID(nil) == nil)
    "#,
        )
        .unwrap();
}

#[test]
fn guid_name_lookups_use_visible_units_then_real_name_replies() {
    use benilla_ui::script::{pfui_host::units, UnitState};
    let script = extended();
    units::set_unit_states(
        script.lua(),
        [(
            5,
            UnitState {
                guid: 5,
                exists: true,
                has_object: true,
                name: Some("Seen".into()),
                ..Default::default()
            },
            0,
        )],
    );
    crate::player_cache::observe_name(script.lua(), 0x777, "Heard", Some("Mage"), 0, 1).unwrap();
    script
        .run(
            r#"
        local name, realm = UnitNameFromGUID("0x0000000000000005")
        assert(name == "Seen" and realm == "")
        name, realm = UnitNameFromGUID("0x00000777")
        assert(name == "Heard" and realm == "")
        local _, class, _, _, sex, cached = GetPlayerInfoByGUID("0x0000000000000777")
        assert(class == "MAGE" and sex == 3 and cached == "Heard")
        assert(UnitNameFromGUID("0x0000000000000999") == nil and GetPlayerInfoByGUID("0x0000000000000999") == nil)
        assert(UnitNameFromGUID("junk") == nil and GetPlayerInfoByGUID(nil) == nil)
        assert(not C_PlayerCache.IsEnabled() and C_PlayerCache.GetPlayerInfoByName("Heard") == nil)
    "#,
        )
        .unwrap();
}

#[test]
fn packed_locations_unpack_to_the_slots_the_snapshot_publishes() {
    let script = extended();
    script
        .run(
            r#"
        assert(ITEM_INVENTORY_LOCATION_PLAYER == 1048576)
        local worn = EquipmentManager_GetLocationData(1048583) -- 0x100007
        assert(worn.isPlayer and not worn.isBags and not worn.isBank and worn.slot == 7 and worn.bag == nil)
        local bag = EquipmentManager_GetLocationData(3146245)
        assert(bag.isPlayer and bag.isBags and not bag.isBank and bag.bag == 2 and bag.slot == 5)
        local bank = EquipmentManager_GetLocationData(4194307)
        assert(bank.isBank and not bank.isBags and not bank.isPlayer and bank.slot == 3)
        local bankbag = EquipmentManager_GetLocationData(6291713)
        assert(bankbag.isBank and bankbag.isBags and bankbag.bag == 5 and bankbag.slot == 1)
        local none = EquipmentManager_GetLocationData(-1)
        assert(not none.isPlayer and not none.isBank and not none.isBags)
    "#,
        )
        .unwrap();
}

#[test]
fn stock_font_colours_gain_colour_methods_before_an_addon_loads() {
    let script = extended();
    script
        .run("NORMAL_FONT_COLOR = { r = 1, g = .82, b = 0 }; OTHER_TABLE = { r = 1 }")
        .unwrap();
    drop(benilla_ui::script::pfui_host::addon_scope(script.lua(), "pfUI").unwrap());
    script
        .run(
            r#"
        local r, g, b, a = NORMAL_FONT_COLOR:GetRGBA()
        assert(r == 1 and g == .82 and b == 0 and a == 1)
        assert(WHITE_FONT_COLOR:GetRGB() == 1 and OTHER_TABLE.GetRGB == nil)
        assert(INTERFACE_VERSION == 11200 and EQUIPMENT_MANAGER == "Equipment Manager")
    "#,
        )
        .unwrap();
}

#[test]
fn slot_candidates_filter_carried_items_by_inventory_type_and_hand() {
    use benilla_ui::script::{
        pfui_host::items::{self, Instance},
        ContainerSlot, ContainerState, ItemTemplateView,
    };
    let mut script = extended();
    // 13 one-hand, 17 two-hand, 22 off-hand only, 14 shield, 11 ring; 99 is not cached.
    let kinds = [(1, 13), (2, 17), (3, 22), (4, 14), (5, 11)];
    for (id, inventory_type) in kinds {
        script.set_item_template(
            id,
            ItemTemplateView {
                inventory_type,
                ..Default::default()
            },
        );
    }
    let ids = [1, 2, 3, 4, 5, 99];
    script.set_container(
        1,
        Some(ContainerState {
            num_slots: 8,
            slots: ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    (
                        index as u32 + 1,
                        ContainerSlot {
                            item_id: *id,
                            count: 1,
                            link: Some(format!("|Hitem:{id}:0:0:0|h[Item]|h")),
                            ..Default::default()
                        },
                    )
                })
                .collect(),
            ..Default::default()
        }),
    );
    items::set_instances(
        script.lua(),
        ids.iter().enumerate().map(|(index, id)| {
            (
                (1, index as u32 + 1),
                Instance {
                    guid: 100 + u64::from(*id),
                    item_id: *id,
                    count: 1,
                    uses: 1,
                    name: None,
                },
            )
        }),
    );
    script
        .run(
            r#"
        local function ids(slot)
          local found, count = {}, 0
          local result = {}
          assert(GetInventoryItemsForSlot(slot, result) == result)
          for location, link in pairs(result) do
            local data = EquipmentManager_GetLocationData(location)
            assert(data.isBags and data.bag == 1 and GetContainerItemLink(1, data.slot) == link)
            found[tonumber(link:match("item:(%d+)"))] = true
            count = count + 1
          end
          return found, count
        end
        local main, mainCount = ids(16)
        assert(mainCount == 2 and main[1] and main[2])
        local off, offCount = ids(17)
        assert(offCount == 3 and off[1] and off[3] and off[4])
        local ring, ringCount = ids(12)
        assert(ringCount == 1 and ring[5])
        assert(select(2, ids(1)) == 0 and select(2, ids(0)) == 0 and select(2, ids(20)) == 0)
        assert(not pcall(GetInventoryItemsForSlot, 16))
    "#,
        )
        .unwrap();
}

#[test]
fn repair_cost_and_mail_links_read_the_stock_feeds() {
    use benilla_ui::script::{ContainerSlot, ContainerState, MailInboxRow, MailState, RepairCosts};
    let mut script = extended();
    script.set_repair_costs(RepairCosts {
        equipped: [(5, 1234)].into(),
        ..Default::default()
    });
    script.set_mail(Some(MailState {
        inbox: vec![
            MailInboxRow {
                item_id: 2589,
                item_name: Some("Linen Cloth".into()),
                item_quality: Some(1),
                ..Default::default()
            },
            MailInboxRow::default(),
            MailInboxRow {
                item_id: 4306,
                ..Default::default()
            },
        ],
        open: true,
    }));
    let link = "|cffffffff|Hitem:2589:0:0:0|h[Linen Cloth]|h|r";
    script.set_container(
        0,
        Some(ContainerState {
            num_slots: 16,
            slots: [(
                1,
                ContainerSlot {
                    item_id: 2589,
                    count: 3,
                    link: Some(link.into()),
                    ..Default::default()
                },
            )]
            .into(),
            ..Default::default()
        }),
    );
    script
        .run(
            r#"
        assert(GetInventoryItemRepairCost(5) == 1234 and GetInventoryItemRepairCost(6) == 0)
        assert(GetInventoryItemRepairCost(0) == 0 and GetInventoryItemRepairCost(20) == 0)
        local link, id = GetInboxItemLink(1)
        assert(link == "|cffffffff|Hitem:2589:0:0:0|h[Linen Cloth]|h|r" and id == 2589)
        assert(GetInboxItemLink(1, 1) == link and GetInboxItemLink(1, 2) == nil)
        assert(GetInboxItemLink(2) == nil and GetInboxItemLink(9) == nil and GetInboxItemLink(0) == nil)
        assert(GetInboxItemLink(3) == nil, "a name still in flight builds no link")
        assert(GetInboxItemID(1) == 2589 and GetInboxItemID(2) == nil and GetInboxItemID(3) == 4306)
        assert(GetSendMailItemLink() == nil)
        PickupContainerItem(0, 1)
        ClickSendMailItemButton()
        link, id = GetSendMailItemLink()
        assert(link == GetContainerItemLink(0, 1) and id == 2589, tostring(link))
        assert(GetSendMailItemLink(2) == nil)
        assert(CreateSimpleTextureMarkup("Interface\\Icons\\X", 24) == "|TInterface\\Icons\\X:24:24|t")
    "#,
        )
        .unwrap();
}

#[test]
fn unit_speed_reads_the_fed_mover_and_zeroes_an_unseen_unit() {
    use benilla_ui::script::{pfui_host::player_stats, UnitGuids};
    let mut script = extended();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        target: 5,
        ..Default::default()
    });
    player_stats::set_speeds(script.lua(), [(3, [7.0, 7.0, 4.722_222])]);
    script
        .run(
            r#"
        local current, run, flight, swim = GetUnitSpeed("player")
        assert(current == 7 and run == 7 and flight == 0 and math.abs(swim - 4.7222) < .001)
        current, run, flight, swim = GetUnitSpeed("target")
        assert(current == 0 and run == 0 and flight == 0 and swim == 0)
        assert(GetUnitSpeed("nonsense") == 0 and GetUnitSpeed(nil) == 0)
    "#,
        )
        .unwrap();
    player_stats::set_speeds(script.lua(), [(3, [0.0, 14.0, 4.722_222])]);
    script
        .run("local current, run = GetUnitSpeed('player') assert(current == 0 and run == 14)")
        .unwrap();
}

#[test]
fn hyperlink_compare_draws_the_worn_counterpart_and_reanchoring_keeps_the_text() {
    use benilla_ui::script::{InvSlotView, InventorySlots, ItemTemplateView};
    let mut script = extended();
    // 11 is a ring: two candidate slots. 99 is equippable with nothing worn; 50 is not cached.
    for (id, name, inventory_type) in [
        (1, "Offered Ring", 11),
        (2, "Worn Ring", 11),
        (3, "Other Ring", 11),
        (99, "Helm", 1),
    ] {
        script.set_item_template(
            id,
            ItemTemplateView {
                name: name.into(),
                class: 4,
                inventory_type,
                ..Default::default()
            },
        );
    }
    let mut slots: InventorySlots = Default::default();
    for (slot, id, name) in [(11, 2, "Worn Ring"), (12, 3, "Other Ring")] {
        slots[slot] = Some(InvSlotView {
            item_id: id,
            name: Some(name.into()),
            quality: 1,
            ..Default::default()
        });
    }
    script.set_inventory_slots(slots);
    script
        .run(
            r#"
        CURRENTLY_EQUIPPED = "Currently Equipped"
        UIParent = CreateFrame("Frame", "UIParent")
        local owner = CreateFrame("Frame", "CompareOwner", UIParent)
        local tip = CreateFrame("GameTooltip", "CompareTip", UIParent, "TestTooltipTemplate")
        local function lines()
          return CompareTipTextLeft1:GetText(), CompareTipTextLeft2:GetText()
        end
        tip:SetOwner(owner, "ANCHOR_RIGHT")
        assert(tip:SetHyperlinkCompareItem("|cffffffff|Hitem:1:0:0:0|h[Offered Ring]|h|r", 1, true) == 1)
        local header, name = lines()
        assert(header == "Currently Equipped" and name == "Worn Ring", tostring(header) .. tostring(name))
        tip:SetAnchorType("ANCHOR_RIGHT", 30, 4)
        local point, relative, relativePoint, x, y = tip:GetPoint(1)
        -- Offsets are kept in layout units, as the reference keeps them: equal to a float's step.
        local function near(a, b) return math.abs(a - b) < .001 end
        assert(point == "BOTTOMLEFT" and relative == owner and relativePoint == "TOPRIGHT" and near(x, 30) and near(y, 4))
        assert(select(2, lines()) == "Worn Ring" and tip:GetAnchorType() == "ANCHOR_RIGHT")
        tip:SetAnchorType("ANCHOR_CURSOR", 9, 9)
        assert(near(select(4, tip:GetPoint(1)), 30))
        tip:SetOwner(owner, "ANCHOR_RIGHT")
        assert(tip:SetHyperlinkCompareItem(1, 2) == 1 and select(2, lines()) == "Other Ring")
        tip:SetOwner(owner, "ANCHOR_RIGHT")
        assert(tip:SetHyperlinkCompareItem("item:1", 3) == nil and tip:SetHyperlinkCompareItem("item:1", 0) == nil)
        assert(tip:SetHyperlinkCompareItem("item:99") == nil, "nothing is worn in that slot")
        assert(tip:SetHyperlinkCompareItem("item:50") == nil and tip:SetHyperlinkCompareItem("junk") == nil)
        assert(tip:SetHyperlinkCompareItem(nil) == nil)
        local unowned = CreateFrame("GameTooltip", "UnownedTip", UIParent, "TestTooltipTemplate")
        unowned:SetAnchorType("ANCHOR_RIGHT", 1, 1)
        assert(unowned:GetNumPoints() == 0)
        assert(unowned:GetOwner() == nil)
        assert(tip:GetOwner() == owner)
        tip:Show()
        tip:Hide()
        assert(tip:GetOwner() == nil, "a hide drops the owner")
    "#,
        )
        .unwrap();
}

#[test]
fn a_widgets_index_answers_a_call_as_the_references_function_does() {
    let script = extended();
    script
        .run(
            r#"
        UIParent = CreateFrame("Frame", "UIParent")
        local function check(object, method)
          local found = getmetatable(object).__index(object, method)
          assert(found ~= nil and found == object[method], method)
          assert(getmetatable(object).__index(object, "NoSuchMethod") == nil)
        end
        local plain = CreateFrame("Frame", nil, UIParent)
        check(plain, "Show")
        local button = CreateFrame("Button", nil, UIParent)
        check(button, "Enable")
        check(button, "Show")
        check(CreateFrame("CheckButton", nil, UIParent), "Show")
        check(plain:CreateTexture(), "SetTexCoord")
        check(plain:CreateFontString(), "SetText")
    "#,
        )
        .unwrap();
}

#[test]
fn a_pool_reset_returns_a_frame_to_defaults_and_keeps_its_scripts() {
    let script = extended();
    script
        .run(
            r#"
        UIParent = CreateFrame("Frame", "UIParent")
        local f = CreateFrame("Button", "PooledButton", UIParent)
        local update = function() end
        f:SetScript("OnUpdate", update)
        f:SetPoint("TOPLEFT", UIParent, "TOPLEFT", 5, 7)
        f:SetWidth(20); f:SetHeight(30); f:SetAlpha(.5); f:SetScale(2)
        f:EnableMouse(true); f:Disable(); f:LockHighlight(); f:SetButtonState("PUSHED", 1)
        f:SetPropagateMouseMotion(true)

        f:ClearPointsOffset()
        local _, _, _, x, y = f:GetPoint(1)
        assert(f:GetNumPoints() == 1 and x == 0 and y == 0)

        f:SetToDefaults()
        assert(f:GetNumPoints() == 0 and f:GetWidth() == 0 and f:GetHeight() == 0)
        assert(f:GetAlpha() == 1 and f:GetScale() == 1)
        assert(not f:IsMouseEnabled() and f:IsEnabled() and f:GetButtonState() == "NORMAL")
        assert(not f:GetPropagateMouseMotion())
        assert(f:GetScript("OnUpdate") == update)

        local t = f:CreateTexture(nil, "ARTWORK")
        t:SetPoint("CENTER", f); t:SetWidth(4); t:SetAlpha(.25)
        t:SetToDefaults()
        assert(t:GetNumPoints() == 0 and t:GetAlpha() == 1)
        local fs = f:CreateFontString(nil, "ARTWORK")
        fs:SetPoint("CENTER", f)
        fs:SetMaxLines(1)
        assert(fs:GetMaxLines() == 1)
        fs:SetToDefaults()
        assert(fs:GetNumPoints() == 0 and fs:GetMaxLines() == 0)

        local pool = { Release = function(self, a, b) return self, a, b end }
        local this, a, b = securecallmethod(pool, "Release", 1, nil)
        assert(this == pool and a == 1 and b == nil)
    "#,
        )
        .unwrap();
}

#[test]
fn coin_strings_show_the_non_zero_denominations_with_money_sheet_crops() {
    let script = extended();
    let coins: (String, String, String, String) = script
        .eval(
            r#"return GetCoinTextureString(123456), GetCoinTextureString(0),
                GetCoinTextureString(500, 24), GetCoinTextureString(-3.2)"#,
        )
        .unwrap();
    let sheet = "Interface\\MoneyFrame\\UI-MoneyIcons";
    assert_eq!(
        coins.0,
        format!(
            "12|T{sheet}:0:0:2:0:64:16:0:16:0:16|t 34|T{sheet}:0:0:2:0:64:16:16:32:0:16|t \
             56|T{sheet}:0:0:2:0:64:16:32:48:0:16|t"
        )
    );
    assert_eq!(coins.1, format!("0|T{sheet}:0:0:2:0:64:16:32:48:0:16|t"));
    assert_eq!(coins.2, format!("5|T{sheet}:24:24:2:0:64:16:16:32:0:16|t"));
    assert_eq!(coins.3, coins.1);
}

/// `gxResolution` answers the window's real pixels once the host has fed them, and the stock
/// string before; every other CVar is untouched.
#[test]
fn the_resolution_query_answers_the_windows_real_pixels() {
    let script = extended();
    // The host registers the reference's CVars; an unregistered name raises, as 1.12.1's does.
    script.register_cvars([("gxResolution", "1024x768"), ("uiScale", "1.0")]);
    let stock: Option<String> = script.eval(r#"return GetCVar("gxResolution")"#).unwrap();
    crate::display::set_window_pixels(script.lua(), 2880, 1800).unwrap();
    let (fed, cased, other): (String, String, Option<String>) = script
        .eval(r#"return GetCVar("gxResolution"), GetCVar("GXRESOLUTION"), GetCVar("uiScale")"#)
        .unwrap();
    assert_eq!(fed, "2880x1800");
    assert_eq!(cased, "2880x1800");
    assert_eq!(
        other,
        script
            .eval::<Option<String>>(r#"return GetCVar("uiScale")"#)
            .unwrap()
    );
    crate::display::set_window_pixels(script.lua(), 0, 0).unwrap();
    assert_eq!(
        script
            .eval::<Option<String>>(r#"return GetCVar("gxResolution")"#)
            .unwrap(),
        stock
    );
}

/// A token that names a unit by guid always has a name, `UNKNOWNOBJECT` when nothing is held for
/// it; only a token that names nobody answers nil. pfUI's tooltip resolves a token from a guid
/// and passes the name straight to `gsub`.
#[test]
fn a_named_but_unheld_unit_reads_as_unknown_and_nobody_as_nil() {
    use benilla_ui::script::UnitGuids;

    let mut script = extended();
    script.run(r#"UNKNOWNOBJECT = "Unknown""#).unwrap();
    script.set_unit_guids(&UnitGuids {
        player: 1,
        party: [5, 0, 0, 0],
        ..Default::default()
    });
    let (held_by_guid_only, nobody): (Option<String>, Option<String>) = script
        .eval(r#"return (UnitName("party1")), (UnitName("party2"))"#)
        .unwrap();
    assert_eq!(held_by_guid_only.as_deref(), Some("Unknown"));
    assert_eq!(nobody, None);
    // The way pfUI uses it must not raise.
    script
        .run(r#"local name = UnitName(UnitTokenFromGUID("0x0000000000000005")); string.gsub(name, name, "", 1)"#)
        .unwrap();
}
