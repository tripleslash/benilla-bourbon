//! The layer's rows on the Turtle WoW dungeon finder's eye menu (`FrameXMLFixes.xml`): the way
//! out of the finder's dungeon and back in. Turtle's `LFT.lua` is not on a 1.12.1 chain, so the
//! harness stands in the four names of it the layer touches and drives the stock dropdown.

use benilla_ui::script::{ScriptValue, UiScript};

use super::test_ui::load_ui as load_xml;

/// The stock dropdown, Turtle's finder names as `LFT.lua` declares them (l.1-4, l.1773-1816: the
/// eye's own rows end in a Cancel row, and only when there is a row), then the layer's fixes.
fn harness() -> UiScript {
    let s = UiScript::new().unwrap();
    for file in [
        "Interface\\FrameXML\\GlobalStrings.lua",
        "Interface\\FrameXML\\Fonts.xml",
        r"Interface\FrameXML\UIParent.xml",
        r"Interface\FrameXML\MoneyFrame.lua",
        r"Interface\FrameXML\MoneyFrame.xml",
        "Interface\\FrameXML\\GameTooltip.xml",
        "Interface\\FrameXML\\UIDropDownMenu.xml",
    ] {
        load_xml(&s, file);
    }
    s.run(
        r#"
        LFT_ADDON_PREFIX = "TW_LFG"
        LFT_ADDON_CHANNEL = "GUILD"
        LFT_ADDON_FIELD_DELIMITER = ";"
        CreateFrame("Button", "LFTMinimapButton", UIParent)
        CreateFrame("Frame", "LFTActionsDropDown", UIParent, "UIDropDownMenuTemplate")
        LFTActionsDropDown.owner = LFTMinimapButton
        function LFTActionsDropDown_Init()
            if Queued and LFTActionsDropDown.owner == LFTMinimapButton then
                local info = { notCheckable = 1, text = "Leave Queue", func = function() end }
                UIDropDownMenu_AddButton(info)
                info = { notCheckable = 1, text = CANCEL, func = HideDropDownMenu, arg1 = 1 }
                UIDropDownMenu_AddButton(info)
            end
        end

        members, inside, sent = 4, nil, {}
        function GetNumPartyMembers() return members end
        function GetNumRaidMembers() return 0 end
        function IsInInstance() return inside end
        function SendAddonMessage(prefix, text, channel)
            table.insert(sent, prefix .. "|" .. text .. "|" .. channel)
        end
        "#,
    )
    .unwrap();
    load_xml(&s, "FrameXMLFixes.xml");
    s
}

/// What the server says of the finder run, as the client hears it.
fn server_says(s: &mut UiScript, text: &str) {
    s.fire_event(
        "CHAT_MSG_ADDON",
        vec![
            ScriptValue::Str("TW_LFG".into()),
            ScriptValue::Str(text.into()),
            ScriptValue::Str("GUILD".into()),
            ScriptValue::Str("Server".into()),
        ],
    );
}

/// The rows the eye's menu builds, the way `ToggleDropDownMenu` builds them.
fn rows(s: &UiScript) -> Vec<String> {
    let joined: String = s
        .eval(
            r#"
            UIDropDownMenu_Initialize(LFTActionsDropDown, LFTActionsDropDown_Init, "MENU")
            local out = {}
            for i = 1, DropDownList1.numButtons do
                table.insert(out, getglobal("DropDownList1Button" .. i):GetText())
            end
            return table.concat(out, "|")
            "#,
        )
        .unwrap();
    joined
        .split('|')
        .filter(|row| !row.is_empty())
        .map(str::to_owned)
        .collect()
}

fn sent(s: &UiScript) -> Vec<String> {
    let joined: String = s.eval(r#"return table.concat(sent, "\n")"#).unwrap();
    joined.lines().map(str::to_owned).collect()
}

#[test]
fn the_eye_offers_the_way_in_and_out_only_in_a_finder_run() {
    let _data = benilla_formats::wow_data_or_skip!();
    let mut s = harness();

    assert!(
        rows(&s).is_empty(),
        "no run announced: the menu is Turtle's own, which is empty here"
    );

    server_says(&mut s, "S2C_DUNGEON_RUN;1");
    assert_eq!(rows(&s), ["Teleport to dungeon", "Cancel"]);

    s.run("inside = 1").unwrap();
    assert_eq!(rows(&s), ["Teleport out of dungeon", "Cancel"]);

    s.run("members = 0").unwrap();
    assert!(rows(&s).is_empty(), "out of the group there is no run");

    s.run("members = 4 LFTActionsDropDown.owner = UIParent")
        .unwrap();
    assert!(
        rows(&s).is_empty(),
        "the menu of a row in the group browser is not the eye's"
    );

    s.run("LFTActionsDropDown.owner = LFTMinimapButton")
        .unwrap();
    server_says(&mut s, "S2C_DUNGEON_RUN;0");
    assert!(rows(&s).is_empty(), "the run is over");
}

#[test]
fn the_row_leads_the_stock_rows_and_adds_no_second_cancel() {
    let _data = benilla_formats::wow_data_or_skip!();
    let mut s = harness();
    server_says(&mut s, "S2C_DUNGEON_RUN;1");
    s.run("Queued = 1").unwrap();

    assert_eq!(rows(&s), ["Teleport to dungeon", "Leave Queue", "Cancel"]);
}

#[test]
fn the_row_asks_the_server_for_the_journey() {
    let _data = benilla_formats::wow_data_or_skip!();
    let mut s = harness();
    server_says(&mut s, "S2C_DUNGEON_RUN;1");

    assert_eq!(rows(&s)[0], "Teleport to dungeon");
    s.run("DropDownList1Button1:Click()").unwrap();

    s.run("inside = 1").unwrap();
    assert_eq!(rows(&s)[0], "Teleport out of dungeon");
    s.run("DropDownList1Button1:Click()").unwrap();

    assert_eq!(
        sent(&s),
        [
            "TW_LFG|C2S_DUNGEON_TELEPORT;in|GUILD",
            "TW_LFG|C2S_DUNGEON_TELEPORT;out|GUILD"
        ]
    );
}

/// Turtle's handler matches its messages by substring (`LFT.lua:167-239`), so the new one must
/// contain none of them, or a finder run would read as a queue event.
#[test]
fn the_run_message_is_none_of_the_stock_handlers() {
    for stock in [
        "S2C_QUEUE_JOINED",
        "S2C_QUEUE_LEFT",
        "S2C_OFFER_NEW",
        "S2C_OFFER_UPDATE_COUNT",
        "S2C_OFFER_COMPLETE",
        "S2C_ROLECHECK_START",
        "S2C_ROLECHECK_INFO",
        "S2C_QUEUE_ERROR",
        "S2C_UPDATE_QUEUE_STATUS",
        "S2C_GROUPS_STATUS",
        "S2C_GROUPS_LIST_UPDATE",
        "S2C_UPDATE_GROUP",
    ] {
        assert!(!"S2C_DUNGEON_RUN;1".contains(stock), "{stock}");
    }
}
