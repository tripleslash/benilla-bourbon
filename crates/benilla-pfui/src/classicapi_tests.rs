//! The engine half the ClassicAPI DLL's own Lua addon (`!!!ClassicAPI`) builds on, and the
//! FrameXML it reads: each behaviour as that addon or pfUI uses it.

use benilla_ui::script::{ScriptValue, UiScript};

fn extended() -> UiScript {
    let mut script = UiScript::new().unwrap();
    script.keep_calls();
    super::install(&script).unwrap();
    script
}

fn load(script: &UiScript, xml: &str) {
    let doc = benilla_ui::framexml::parse(xml).expect("parses");
    let report = benilla_ui::loader::load(script, &doc, &|_| None);
    assert!(report.errors.is_empty(), "load: {:?}", report.errors);
}

#[test]
fn a_regions_relative_key_finds_a_frame_nested_in_the_same_load() {
    let script = extended();
    load(
        &script,
        r#"<Ui>
          <Frame name="KeyedOuter">
            <Size x="100" y="20"/>
            <Frames>
              <Frame parentKey="Arrow">
                <Size x="16" y="16"/>
                <Anchors><Anchor point="RIGHT" x="-2"/></Anchors>
              </Frame>
            </Frames>
            <Layers>
              <Layer level="OVERLAY">
                <FontString name="KeyedText" parentKey="Text" inherits="GameFontWhite">
                  <Anchors>
                    <Anchor point="LEFT" x="2"/>
                    <Anchor point="RIGHT" relativeKey="$parent.Arrow" relativePoint="LEFT" x="-4"/>
                  </Anchors>
                </FontString>
              </Layer>
            </Layers>
          </Frame>
        </Ui>"#,
    );
    script
        .run(
            r#"
        local text = KeyedOuter.Text
        assert(text:GetNumPoints() == 2)
        for i = 1, 2 do
          local point, relative, relativePoint = text:GetPoint(i)
          if point == "RIGHT" then
            assert(relative == KeyedOuter.Arrow and relativePoint == "LEFT")
          else
            assert(point == "LEFT" and relative == KeyedOuter and relativePoint == "LEFT")
          end
        end
    "#,
        )
        .unwrap();
}

#[test]
fn an_anchored_unscaled_size_reads_in_the_units_of_its_edges_at_any_scale() {
    let script = extended();
    script
        .run(
            r#"
        local near = function(a, b) return math.abs(a - b) < 0.01 end
        local f = CreateFrame("Frame")
        f:SetScale(0.5)
        f:SetPoint("TOPLEFT", UIParent, "TOPLEFT", 10, -10)
        f:SetPoint("BOTTOMRIGHT", UIParent, "TOPLEFT", 110, -60)
        assert(near(f:GetUnscaledWidth(), f:GetRight() - f:GetLeft()), f:GetUnscaledWidth())
        assert(near(f:GetUnscaledHeight(), f:GetTop() - f:GetBottom()), f:GetUnscaledHeight())
        local w, h = f:GetUnscaledSize()
        assert(near(w, f:GetUnscaledWidth()) and near(h, f:GetUnscaledHeight()))
        -- ClassicAPI 1.20: GetWidth/GetHeight keep 1.12's scaled read.
        assert(near(f:GetHeight(), (f:GetTop() - f:GetBottom()) * f:GetEffectiveScale()), f:GetHeight())
        assert(f:GetWidth(true) == 0 and f:GetHeight(true) == 0 and f:GetUnscaledHeight(true) == 0)
        local t = f:CreateTexture()
        t:SetAllPoints()
        assert(near(t:GetUnscaledHeight(), t:GetTop() - t:GetBottom()), t:GetUnscaledHeight())
        assert(t:GetHeight(true) == 0 and t:GetUnscaledHeight(true) == 0)
        f:SetHeight(40)
        assert(near(f:GetHeight(true), 40) and near(f:GetHeight(), 40) and near(f:GetUnscaledHeight(), 40))
    "#,
        )
        .unwrap();
}

#[test]
fn strsplit_splits_on_any_separator_character_and_caps_the_pieces() {
    let script = extended();
    script
        .run(
            r#"
        local a, b, c = strsplit(",", "a,b,c")
        assert(a == "a" and b == "b" and c == "c")
        local x, y = strsplit(",", "a,b,c,d", 2)
        assert(x == "a" and y == "b,c,d")
        local p, q, r = strsplit(" -", "a b-c")
        assert(p == "a" and q == "b" and r == "c")
        local e1, e2, e3 = strsplit(",", "a,,b")
        assert(e1 == "a" and e2 == "" and e3 == "b")
        assert(strsplit(",", "whole", 1) == "whole")
        local t = strsplittable(":", "hello:world:banana")
        assert(table.getn(t) == 3 and t[3] == "banana")
        assert(string.split == strsplit)
    "#,
        )
        .unwrap();
}

#[test]
fn securecall_reports_an_error_to_the_handler_and_goes_on() {
    let script = extended();
    script
        .run(
            r#"
        local reported = {}
        seterrorhandler(function(message) table.insert(reported, message) end)
        assert(securecall(math.max, 1, 9) == 9)
        assert(securecall(function() error("boom") end) == nil)
        assert(table.getn(reported) == 1)
        local seen = {}
        secureexecuterange({10, 20}, function(key, value, extra)
          if value == 10 then error("first fails") end
          seen[key] = value .. extra
        end, "x")
        assert(seen[2] == "20x" and table.getn(reported) == 2)
        assert(not pcall(securecallfunction, "notAFunction"))
    "#,
        )
        .unwrap();
}

#[test]
fn ui_colors_are_built_through_the_callers_create_color() {
    let script = extended();
    script
        .run(
            r#"
        local made = 0
        local create = CreateColor
        CreateColor = function(r, g, b, a) made = made + 1; return create(r, g, b, a) end
        local rows = C_UIColor.GetColors()
        assert(table.getn(rows) > 300 and made == table.getn(rows))
        local normal
        for _, row in ipairs(rows) do
          if row.baseTag == "NORMAL_FONT_COLOR" then normal = row.color end
        end
        assert(normal.r == 1 and math.abs(normal.g - 0.8235) < 0.001 and normal.b == 0)
        CreateColor = create
    "#,
        )
        .unwrap();
}

#[test]
fn the_color_code_reads_the_color_itself_after_the_addon_replaces_the_mixin() {
    let script = extended();
    // !!!ClassicAPI's ColorMixin asks the engine for its code: the two must not call each other.
    script
        .run(
            r#"
        ColorMixin = {}
        function ColorMixin:GenerateHexColor() return C_ColorUtil.GenerateTextColorCode(self) end
        local color = {r = 1, g = 0.2, b = 0, a = 1}
        assert(ColorMixin.GenerateHexColor(color) == "ffff3300")
    "#,
        )
        .unwrap();
}

#[test]
fn the_classicapi_table_keeps_the_engine_functions_an_addon_replaces() {
    let script = extended();
    script
        .run(
            r#"
        assert(ClassicAPI.strsplit == strsplit and ClassicAPI.C_CVar.GetCVarBool == C_CVar.GetCVarBool)
        local engine = IsModifiedClick
        IsModifiedClick = function() return "addon" end
        assert(ClassicAPI.IsModifiedClick == engine)
    "#,
        )
        .unwrap();
}

#[test]
fn an_unknown_cvar_reads_nil_and_a_known_one_reads_as_a_boolean() {
    let script = extended();
    script.register_cvars([("benillaFlag", "1"), ("benillaOff", "0")]);
    script
        .run(
            r#"
        assert(GetCVar("noSuchCVar") == nil)
        assert(C_CVar.GetCVarBool("noSuchCVar") == nil)
        assert(C_CVar.GetCVarBool("benillaFlag") == true and C_CVar.GetCVarBool("benillaOff") == false)
        assert(C_CVar.DoesCVarExist("benillaFlag") and not C_CVar.DoesCVarExist("noSuchCVar"))
    "#,
        )
        .unwrap();
}

#[test]
fn modified_clicks_have_the_documented_actions_and_defaults() {
    let script = extended();
    script
        .run(
            r#"
        assert(GetModifiedClick("SELFCAST") == "ALT" and GetModifiedClick("CHATLINK") == "SHIFT-BUTTON1")
        assert(GetNumModifiedClickActions() == 18)
        SetModifiedClick("SELFCAST", "ctrl")
        assert(GetModifiedClick("SELFCAST") == "CTRL")
        assert(not pcall(SetModifiedClick, "NOSUCHACTION", "ALT"))
        assert(IsModifiedClick("SELFCAST") == nil)
    "#,
        )
        .unwrap();
}

#[test]
fn class_names_fill_a_table_the_addon_rebuilds_the_global_with() {
    let script = extended();
    super::install_class_names(&script, [("WARRIOR", "Warrior"), ("MAGE", "Mage")]).unwrap();
    script
        .run(
            r#"
        LOCALIZED_CLASS_NAMES_MALE = {}
        FillLocalizedClassList(LOCALIZED_CLASS_NAMES_MALE, false)
        assert(LOCALIZED_CLASS_NAMES_MALE.MAGE == "Mage")
        local keep = {OTHER = 1}
        assert(FillLocalizedClassList(keep, true) == keep and keep.WARRIOR == "Warrior" and keep.OTHER == 1)
    "#,
        )
        .unwrap();
}

#[test]
fn parent_keys_mixins_and_key_values_reach_the_object_before_its_onload() {
    let script = extended();
    script
        .run("UIParent = CreateFrame('Frame', 'UIParent')")
        .unwrap();
    script
        .run(
            r#"
        TestMixin = {}
        function TestMixin:OnLoad()
          loaded = {self.Child ~= nil, self.Child.Art ~= nil, self.label, self.size, self.flag}
        end
    "#,
        )
        .unwrap();
    load(
        &script,
        r#"<Ui>
          <Frame name="TestBase" mixin="TestMixin" virtual="true">
            <KeyValues>
              <KeyValue key="label" value="text"/>
              <KeyValue key="size" value="12" type="number"/>
              <KeyValue key="flag" value="true" type="boolean"/>
            </KeyValues>
            <Frames>
              <Frame parentKey="Child">
                <Layers><Layer><Texture parentKey="Art"/></Layer></Layers>
              </Frame>
            </Frames>
            <Scripts><OnLoad method="OnLoad"/></Scripts>
          </Frame>
          <Frame name="TestFrame" inherits="TestBase" parent="UIParent"/>
        </Ui>"#,
    );
    script
        .run(
            r#"
        assert(loaded[1] and loaded[2] and loaded[3] == "text" and loaded[4] == 12 and loaded[5] == true)
        assert(TestFrame.Child:GetParent() == TestFrame)
    "#,
        )
        .unwrap();
}

#[test]
fn an_intrinsic_tag_runs_its_handlers_around_the_frames_own() {
    let script = extended();
    script
        .run("UIParent = CreateFrame('Frame', 'UIParent'); order = {}")
        .unwrap();
    script
        .run(
            r#"
        TestIntrinsicMixin = {}
        function TestIntrinsicMixin:OnLoad_Intrinsic() table.insert(order, "intrinsic load") end
        function TestIntrinsicMixin:OnShow_Intrinsic() table.insert(order, "intrinsic show") end
    "#,
        )
        .unwrap();
    load(
        &script,
        r#"<Ui>
          <Frame name="TestEventFrame" mixin="TestIntrinsicMixin" intrinsic="true">
            <Scripts>
              <OnLoad method="OnLoad_Intrinsic"/>
              <OnShow method="OnShow_Intrinsic" intrinsicOrder="postcall"/>
            </Scripts>
          </Frame>
          <TestEventFrame name="TestInstance" parent="UIParent" hidden="true">
            <Scripts>
              <OnLoad>table.insert(order, "own load")</OnLoad>
              <OnShow>table.insert(order, "own show")</OnShow>
            </Scripts>
          </TestEventFrame>
        </Ui>"#,
    );
    script
        .run(
            r#"
        assert(TestInstance:GetObjectType() == "Frame" and _G.TestEventFrame == nil)
        TestInstance:Show()
        assert(table.concat(order, ",") == "intrinsic load,own load,own show,intrinsic show",
          table.concat(order, ","))
        local made = CreateFrame("TestEventFrame", nil, UIParent)
        assert(made.OnLoad_Intrinsic ~= nil)
    "#,
        )
        .unwrap();
}

#[test]
fn template_lists_apply_each_template_and_inherit_prepends_the_handler() {
    let script = extended();
    script
        .run("UIParent = CreateFrame('Frame', 'UIParent'); calls = {}")
        .unwrap();
    load(
        &script,
        r#"<Ui>
          <Frame name="TestA" virtual="true">
            <KeyValues><KeyValue key="a" value="1" type="number"/></KeyValues>
            <Scripts><OnShow>table.insert(calls, "A")</OnShow></Scripts>
          </Frame>
          <Frame name="TestB" virtual="true">
            <KeyValues><KeyValue key="b" value="2" type="number"/></KeyValues>
            <Scripts><OnShow inherit="prepend">table.insert(calls, "B")</OnShow></Scripts>
          </Frame>
        </Ui>"#,
    );
    script
        .run(
            r#"
        local f = CreateFrame("Frame", nil, UIParent, "TestA, TestB")
        assert(f.a == 1 and f.b == 2)
        calls = {}
        f:Hide(); f:Show()
        assert(table.concat(calls, ",") == "A,B", table.concat(calls, ","))
    "#,
        )
        .unwrap();
}

#[test]
fn a_relative_key_anchors_to_a_sibling_and_an_anchor_gives_its_own_offset() {
    let script = extended();
    script
        .run("UIParent = CreateFrame('Frame', 'UIParent')")
        .unwrap();
    load(
        &script,
        r#"<Ui>
          <Frame name="TestOuter" parent="UIParent">
            <Size x="100" y="100"/>
            <Anchors><Anchor point="CENTER"/></Anchors>
            <Frames>
              <Frame parentKey="First"><Size x="10"/></Frame>
              <Frame parentKey="Second">
                <Size x="10" y="10"/>
                <Anchors><Anchor point="TOP" relativeKey="$parent.First" relativePoint="BOTTOM" y="-5"/></Anchors>
              </Frame>
            </Frames>
          </Frame>
        </Ui>"#,
    );
    script
        .run(
            r#"
        local point, relative, relativePoint, x, y = TestOuter.Second:GetPoint(1)
        assert(relative == TestOuter.First and point == "TOP" and relativePoint == "BOTTOM")
        -- Offsets are kept in layout units.
        assert(x == 0 and math.abs(y + 5) < 0.001, tostring(x) .. "," .. tostring(y))
        assert(math.abs(TestOuter.First:GetWidth() - 10) < 0.001 and TestOuter.First:GetHeight() == 0,
          TestOuter.First:GetWidth() .. "x" .. TestOuter.First:GetHeight())
    "#,
        )
        .unwrap();
}

#[test]
fn an_atlas_points_a_texture_at_its_rectangle() {
    let script = extended();
    script
        .run(
            r#"
        C_Texture.RegisterAtlas("test-gem", "Interface\\Test\\sheet", 32, 16, 0, 0.25, 0.5, 1)
        local info = C_Texture.GetAtlasInfo("TEST-GEM")
        assert(info.filename == "Interface\\Test\\sheet" and info.width == 32 and info.rightTexCoord == 0.25)
        assert(C_Texture.GetAtlasExists("test-gem") and not C_Texture.GetAtlasExists("nope"))
        local t = CreateFrame("Frame"):CreateTexture()
        t:SetAtlas("test-gem", true)
        assert(t:GetAtlas() == "test-gem" and t:GetWidth() == 32 and t:GetHeight() == 16)
        t:SetAtlas("unknown")
        assert(t:GetAtlas() == "test-gem")
        t:SetTexture("Interface\\Other")
        assert(t:GetAtlas() == nil)
    "#,
        )
        .unwrap();
}

#[test]
fn a_cooldown_frame_is_the_indicator_model_with_the_modern_methods() {
    let script = extended();
    script
        .run("UIParent = CreateFrame('Frame', 'UIParent')")
        .unwrap();
    load(
        &script,
        r#"<Ui><Model name="CooldownFrameTemplate" setAllPoints="true" hidden="true" virtual="true"/></Ui>"#,
    );
    script
        .run(
            r#"
        local cd = CreateFrame("Cooldown", nil, UIParent, "CooldownFrameTemplate")
        assert(cd:IsObjectType("Model"))
        cd:SetReverse(true); cd:SetDrawBling(false)
        assert(cd:GetReverse() and not cd:GetDrawBling() and cd:GetDrawSwipe())
        cd:SetCooldown(100, 10)
        assert(cd:IsShown() and select(2, cd:GetCooldownTimes()) == 10000)
        cd:Clear()
        assert(not cd:IsShown() and cd:GetCooldownTimes() == 0)
    "#,
        )
        .unwrap();
}

#[test]
fn a_cooldown_counts_down_in_retails_units_while_the_cvar_is_on() {
    let script = extended();
    script
        .run("UIParent = CreateFrame('Frame', 'UIParent')")
        .unwrap();
    load(
        &script,
        r#"<Ui><Model name="CooldownFrameTemplate" setAllPoints="true" hidden="true" virtual="true"/></Ui>"#,
    );
    script
        .run(
            r#"
        local cd = CreateFrame("Cooldown", nil, UIParent, "CooldownFrameTemplate")
        local function step() this = cd; cd:GetScript("OnUpdateModel")() end
        local function text() return cd.benillaCountdownText or "" end
        GetTime = function() return 102 end
        assert(GetCVar("countdownForCooldowns") == "0")
        cd:SetCooldown(100, 10)
        step()
        assert(text() == "" and cd.benillaCountdown == nil, text())
        SetCVar("countdownForCooldowns", "1")
        step()
        assert(text() == "8" and cd:GetCountdownFontString():GetText() == "8", text())
        cd:SetCooldown(100, 225)
        step()
        assert(text() == "4m", text())
        cd:SetCountdownAbbrevThreshold(300)
        step()
        assert(text() == "3:43", text())
        cd:SetCooldown(98.76, 7)
        cd:SetCountdownMillisecondsThreshold(5)
        step()
        assert(text() == "3.8", text())
        cd:SetHideCountdownNumbers(true)
        step()
        assert(text() == "", text())
        cd:SetHideCountdownNumbers(false)
        cd:SetCooldown(101, 1.5)
        step()
        assert(text() == "", "below the minimum: " .. text())
    "#,
        )
        .unwrap();
}

#[test]
fn a_cooldowns_flags_steer_its_sweep() {
    let script = extended();
    script
        .run("UIParent = CreateFrame('Frame', 'UIParent')")
        .unwrap();
    load(
        &script,
        r#"<Ui><Model name="CooldownFrameTemplate" setAllPoints="true" hidden="true" virtual="true"/></Ui>"#,
    );
    script
        .run(
            r#"
        local cd = CreateFrame("Cooldown", nil, UIParent, "CooldownFrameTemplate")
        local at
        cd.SetSequenceTime = function(self, sequence, ms) at = ms end
        local function step() this = cd; cd:GetScript("OnUpdateModel")() end
        GetTime = function() return 102 end
        cd:SetCooldown(100, 10)
        step()
        assert(at and math.abs(at - 200) < 50, tostring(at))
        cd:SetReverse(true)
        step()
        assert(math.abs(at - 800) < 50, tostring(at))
        cd:SetPaused(true); at = nil
        step()
        assert(at == nil)
        cd:SetPaused(false)
        cd:SetDrawSwipe(false)
        assert(cd:GetAlpha() == 0 and not cd:GetDrawSwipe())
        cd:SetDrawBling(false)
        cd:SetCooldown(80, 10)
        step()
        assert(not cd:IsShown())
    "#,
        )
        .unwrap();
}

#[test]
fn an_alpha_animation_fades_its_parent_and_reports_the_finish() {
    let mut script = extended();
    script
        .run(
            r#"
        frame = CreateFrame("Frame")
        frame:SetAlpha(1)
        group = frame:CreateAnimationGroup()
        local fade = group:CreateAnimation("Alpha")
        fade:SetChange(-1)
        fade:SetDuration(1)
        finished = 0
        group:SetScript("OnFinished", function() finished = finished + 1 end)
        group:Play()
        assert(group:IsPlaying())
    "#,
        )
        .unwrap();
    script.tick(0.5);
    script
        .run("assert(math.abs(frame:GetAlpha() - 0.5) < 0.01, frame:GetAlpha())")
        .unwrap();
    script.tick(0.6);
    // The target's own alpha comes back at the end, as it does without SetToFinalAlpha.
    script
        .run("assert(finished == 1 and not group:IsPlaying() and frame:GetAlpha() == 1)")
        .unwrap();
}

#[test]
fn no_classicapi_file_is_skipped() {
    assert!(!super::skipped_file(
        "!!!classicapi",
        "Util/TradeSkillLink.lua"
    ));
    assert!(!super::skipped_file(
        "!!!ClassicAPI",
        "Util\\SlashCommandsRegistry.lua"
    ));
    assert!(!super::skipped_file("!!!ClassicAPI", "Util\\Pools.lua"));
    assert!(!super::skipped_file(
        "pfUI",
        "Util\\SlashCommandsRegistry.lua"
    ));
}

#[test]
fn the_version_is_the_classicapi_addons_once_it_has_loaded() {
    let mut script = extended();
    script.run("assert(CLASSIC_API_VERSION == 11510)").unwrap();
    script.fire_event(
        "ADDON_LOADED",
        vec![benilla_ui::script::ScriptValue::Str("SomeAddon".into())],
    );
    script.run("assert(CLASSIC_API_VERSION == 11510)").unwrap();
    script.fire_event(
        "ADDON_LOADED",
        vec![benilla_ui::script::ScriptValue::Str("!!!ClassicAPI".into())],
    );
    script
        .run(
            "assert(CLASSIC_API_VERSION == 12006 and BenillaClassicAPICompatibility.apiVersion == '1.20.6'
               and BenillaClassicAPICompatibility.fullClassicAPI == false)",
        )
        .unwrap();
}

#[test]
fn scale_and_path_animations_move_their_target_and_give_it_back() {
    let mut script = extended();
    script
        .run(
            r#"
        parent = CreateFrame("Frame")
        frame = CreateFrame("Frame", nil, parent)
        frame:SetWidth(10); frame:SetHeight(20)
        frame:SetPoint("CENTER", parent, "CENTER", 0, 0)
        group = frame:CreateAnimationGroup()
        local zoom = group:CreateAnimation("Scale")
        zoom:SetScale(2, 2); zoom:SetDuration(1)
        local path = group:CreateAnimation("Path")
        local p1 = path:CreateControlPoint(); p1:SetOffset(10, 0)
        local p2 = path:CreateControlPoint(); p2:SetOffset(10, 10)
        path:SetDuration(1)
        group:Play()
    "#,
        )
        .unwrap();
    script.tick(0.5);
    script
        .run(
            r#"
        assert(math.abs(frame:GetWidth() - 15) < 0.01 and math.abs(frame:GetHeight() - 30) < 0.01,
          frame:GetWidth() .. "x" .. frame:GetHeight())
        local _, _, _, x, y = frame:GetPoint(1)
        assert(math.abs(x - 10) < 0.01 and math.abs(y) < 0.01, x .. "," .. y)
    "#,
        )
        .unwrap();
    script.tick(0.6);
    script
        .run(
            r#"
        assert(not group:IsPlaying())
        assert(math.abs(frame:GetWidth() - 10) < 0.01 and math.abs(frame:GetHeight() - 20) < 0.01)
        local _, _, _, x, y = frame:GetPoint(1)
        assert(math.abs(x) < 0.01 and math.abs(y) < 0.01)
    "#,
        )
        .unwrap();
}

#[test]
fn swing_ranges_answer_and_raise_for_the_enabled_types() {
    let script = extended();
    script
        .run(
            r#"
        changes = {}
        local f = CreateFrame("Frame")
        f:RegisterEvent("PLAYER_SWING_RANGE_UPDATE")
        f:SetScript("OnEvent", function() table.insert(changes, arg1 .. ":" .. tostring(arg2) .. ":" .. tostring(arg3)) end)
        C_SwingTimer.EnableRangeCheck(Enum.PlayerSwingType.MainHand, true)
        assert(C_SwingTimer.IsTargetWithinSwingRange(0) == nil)
    "#,
        )
        .unwrap();
    super::swing_range::set(script.lua(), [Some(false), None, Some(true)]);
    super::swing_range::set(script.lua(), [Some(true), None, Some(false)]);
    script
        .run(
            r#"
        assert(C_SwingTimer.IsTargetWithinSwingRange(0) == true)
        assert(C_SwingTimer.IsTargetWithinSwingRange(1) == nil)
        assert(C_SwingTimer.IsTargetWithinSwingRange(2) == false)
        -- Only the enabled main hand raises: unknown to out of range, then in range.
        assert(table.concat(changes, ",") == "0:nil:1,0:1:1", table.concat(changes, ","))
    "#,
        )
        .unwrap();
}

#[test]
fn paper_doll_slots_and_inventory_types_answer_by_number() {
    let script = extended();
    script
        .run(
            r#"
        GetInventorySlotInfo = function(name) if name == "RangedSlot" then return 18, "ranged.blp" end end
        local id, tex, relic, name = C_PaperDollInfo.GetInventorySlotInfoForInvSlot(18)
        assert(id == 18 and tex == "ranged.blp" and relic == true and name == "RangedSlot")
        assert(C_PaperDollInfo.GetInventorySlotInfoForInvSlot(40) == nil)
        assert(not pcall(C_PaperDollInfo.GetInventorySlotInfoForInvSlot, "x"))
        assert(type(C_Item.GetItemInventoryType) == "function")
    "#,
        )
        .unwrap();
}

#[test]
fn the_addons_scroll_and_dropdown_atlases_are_built_in() {
    let script = extended();
    script
        .run(
            r#"
        for _, name in ipairs({"UI-ClassicScrollBar-EndCap-Top", "UI-ClassicScrollBar-Center",
            "UI-ClassicScrollBar-EndCap-Bottom", "_UI-ScrollBar-Center",
            "UI-ScrollBar-ClassicScrollUpButton-Highlight", "common-dropdown-c-button-open",
            "common-dropdown-c-bg"}) do
          assert(C_Texture.GetAtlasExists(name), name)
        end
        local top = C_Texture.GetAtlasInfo("UI-ClassicScrollBar-EndCap-Top")
        assert(top.filename == "Interface\\PaperDollInfoFrame\\UI-Character-ScrollBar" and top.width == 31)
        assert(C_Texture.GetAtlasInfo("UI-ClassicScrollBar-Center").tilesVertically)
    "#,
        )
        .unwrap();
}

fn combat_event(
    subevent: &'static str,
    source: u64,
    flags: u32,
    amount: i64,
) -> super::combat_log::Event {
    use benilla_ui::script::ScriptValue;
    super::combat_log::Event {
        subevent,
        source: Some(super::combat_log::Unit {
            guid: source,
            name: Some("Me".into()),
            flags,
            raid_flags: 0,
        }),
        dest: None,
        spell: Some((133, Some("Fireball".into()))),
        values: vec![
            ScriptValue::Int(133),
            ScriptValue::Str("Fireball".into()),
            ScriptValue::Int(4),
            ScriptValue::Int(amount),
        ],
    }
}

#[test]
fn the_combat_log_raises_its_events_with_the_values_and_filters_the_second() {
    let script = extended();
    script
        .run(
            r#"
        seen = {}
        local f = CreateFrame("Frame")
        f:RegisterEvent("COMBAT_LOG_EVENT_UNFILTERED")
        f:RegisterEvent("COMBAT_LOG_EVENT")
        f:SetScript("OnEvent", function()
          local ts, sub, hide, sguid, sname, sflags, sraid, dguid, dname, dflags, draid, id, name, school, amount =
            CombatLogGetCurrentEventInfo()
          table.insert(seen, event .. ":" .. sub .. ":" .. tostring(sname) .. ":" .. tostring(dname) .. ":" ..
            tostring(dflags == COMBATLOG_OBJECT_NONE) .. ":" .. name .. ":" .. amount .. ":" .. tostring(arg2 == sub))
        end)
        CombatLogAddFilter("SPELL_DAMAGE", COMBATLOG_FILTER_MINE)
        assert(not pcall(CombatLogAddFilter, nil, COMBATLOG_OBJECT_AFFILIATION_MINE))
    "#,
        )
        .unwrap();
    let mine = 0x1 | 0x10 | 0x100 | 0x400;
    let theirs = 0x8 | 0x40 | 0x200 | 0x800;
    super::combat_log::record(script.lua(), combat_event("SPELL_DAMAGE", 1, mine, 100));
    super::combat_log::record(script.lua(), combat_event("SPELL_DAMAGE", 2, theirs, 50));
    script
        .run(
            r#"
        assert(table.getn(seen) == 3, table.concat(seen, " | "))
        assert(seen[1] == "COMBAT_LOG_EVENT_UNFILTERED:SPELL_DAMAGE:Me:nil:true:Fireball:100:true", seen[1])
        assert(seen[2] == "COMBAT_LOG_EVENT:SPELL_DAMAGE:Me:nil:true:Fireball:100:true", seen[2])
        assert(string.find(seen[3], "^COMBAT_LOG_EVENT_UNFILTERED:.*:50:"))
        assert(CombatLogGetCurrentEventInfo() == nil)
        -- The kept entries: the filter counts one, both without it.
        assert(CombatLogGetNumEntries() == 1 and CombatLogGetNumEntries(true) == 2)
        assert(CombatLogSetCurrentEntry(1) == 1)
        local _, sub, _, _, _, _, _, _, _, _, _, _, _, _, amount = CombatLogGetCurrentEntry()
        assert(sub == "SPELL_DAMAGE" and amount == 100)
        assert(CombatLogSetCurrentEntry(0, true) == 1)
        assert(select(15, CombatLogGetCurrentEntry()) == 50)
        assert(CombatLogAdvanceEntry(-1, true) == 1 and select(15, CombatLogGetCurrentEntry()) == 100)
        assert(CombatLog_Object_IsA(COMBATLOG_FILTER_ME, COMBATLOG_FILTER_MINE) == 1)
        assert(CombatLog_Object_IsA(COMBATLOG_OBJECT_NONE, COMBATLOG_FILTER_MINE) == nil)
        CombatLogClearEntries()
        assert(CombatLogGetNumEntries(true) == 0)
    "#,
        )
        .unwrap();
}

#[test]
fn an_incoming_resurrection_raises_once_per_token_and_only_on_a_change() {
    let script = extended();
    script
        .run(
            r#"
        seen = {}
        local f = CreateFrame("Frame")
        f:RegisterEvent("INCOMING_RESURRECT_CHANGED")
        f:SetScript("OnEvent", function() table.insert(seen, arg1) end)
        assert(UnitHasIncomingResurrection("nosuchunit") == false)
    "#,
        )
        .unwrap();
    let lua = script.lua();
    super::resurrect::set(lua, 7, true, &["target", "party1"]);
    super::resurrect::set(lua, 7, true, &["target", "party1"]);
    super::resurrect::set(lua, 7, false, &["target"]);
    script
        .run(
            r#"assert(table.concat(seen, ",") == "target,party1,target", table.concat(seen, ","))"#,
        )
        .unwrap();
}

#[test]
fn a_school_lockout_reads_as_a_school_interrupt() {
    let script = extended();
    super::school_lockout(script.lua(), 4, 4.0);
    script
        .run(
            r#"
        local mask, remaining = C_LossOfControl.GetSchoolLockout()
        assert(mask == 4 and remaining > 3.9, tostring(mask) .. " " .. tostring(remaining))
        assert(C_LossOfControl.GetSchoolLockout(2) == 0)
        assert(C_LossOfControl.GetSchoolLockout(4) == 4)
    "#,
        )
        .unwrap();
}

#[test]
fn an_addons_memory_is_what_the_heap_grew_while_it_loaded() {
    let script = extended();
    {
        let _scope = benilla_ui::script::pfui_host::addon_scope(script.lua(), "Big").unwrap();
        script
            .run("BigData = {} for i = 1, 20000 do BigData[i] = 'entry ' .. i end")
            .unwrap();
    }
    script
        .run(
            r#"
        UpdateAddOnMemoryUsage()
        assert(GetAddOnMemoryUsage("Big") > 100, GetAddOnMemoryUsage("Big"))
        assert(GetAddOnMemoryUsage("NeverLoaded") == 0)
    "#,
        )
        .unwrap();
}

#[test]
fn a_registered_atlas_reaches_the_text_escapes() {
    let script = extended();
    let (_, before) =
        benilla_ui::script::pfui_host::inline_texture::atlases_since(script.lua(), 0).unwrap();
    assert!(before.contains_key("ui-classicscrollbar-center"));
    script
        .run(r#"C_Texture.RegisterAtlas("Test-Gem", "Interface\\Test\\sheet", 32, 16, 0, 0.25, 0.5, 1)"#)
        .unwrap();
    let (_, atlases) =
        benilla_ui::script::pfui_host::inline_texture::atlases_since(script.lua(), 0).unwrap();
    let gem = &atlases["test-gem"];
    assert_eq!(gem.file, "Interface\\Test\\sheet");
    assert_eq!((gem.width, gem.coords), (32.0, [0.0, 0.25, 0.5, 1.0]));
}

#[test]
fn a_units_selection_type_follows_classicapis_ladder() {
    let script = extended();
    // The unit functions stand in for each case; the ladder is what is tested.
    script
        .run(
            r#"
        local unit
        UnitExists = function() return unit ~= nil end
        UnitIsVisible = function() return unit ~= nil end
        UnitPlayerControlled = function() return unit.controlled end
        UnitIsDead = function() return unit.dead end
        UnitReaction = function() return unit.reaction end
        UnitCanAttack = function(a, b)
          if a == "player" then return unit.weAttack end
          return unit.theyAttack
        end
        UnitIsPVP = function() return unit.pvp end
        UnitIsUnit = function() return false end
        UnitInParty = function() return unit.party end
        UnitInRaid = function() return nil end
        local function check(case, expected, extended)
          unit = case
          local got = UnitSelectionType("target", extended)
          assert(got == expected, tostring(got) .. " for " .. tostring(expected))
        end
        check(nil, 4) check(nil, 5, true)
        check({reaction = 1}, 0) check({reaction = 3}, 1) check({reaction = 4}, 2) check({reaction = 6}, 3)
        check({reaction = 6, dead = 1}, 9, true)
        check({controlled = 1, weAttack = 1, theyAttack = 1}, 0)
        check({controlled = 1, theyAttack = 1}, 4)
        check({controlled = 1, weAttack = 1}, 2)
        check({controlled = 1, pvp = 1}, 3) check({controlled = 1}, 4) check({controlled = 1}, 5, true)
        check({controlled = 1, party = 1, pvp = 1}, 7, true)
    "#,
        )
        .unwrap();
}

#[test]
fn the_reputation_headers_expand_and_collapse_at_once() {
    let script = extended();
    script
        .run(
            r#"
        local headers = {[1] = true, [3] = true}
        local collapsed = {[1] = true}
        GetNumFactions = function() return 4 end
        GetFactionInfo = function(i) return "F", "", 0, 0, 0, 0, nil, nil, headers[i], collapsed[i] end
        ExpandFactionHeader = function(i) collapsed[i] = nil end
        CollapseFactionHeader = function(i) collapsed[i] = true end
        ExpandAllFactionHeaders()
        assert(not collapsed[1] and not collapsed[3])
        CollapseAllFactionHeaders()
        assert(collapsed[1] and collapsed[3] and not collapsed[2])
        local name, kind, difficulty = GetInstanceInfo()
        assert(kind == "none" and difficulty == 1)
        GetSpellInfo = function(spell)
          if spell == 133 then return "Fireball", "Rank 1", nil, 0, false, 0, 1500, 0, 35, 133 end
        end
        local link, id = GetSpellLink(133)
        assert(link == "|cff71d5ff|Hspell:133:0|h[Fireball]|h|r" and id == 133, link)
        assert(GetSpellLink(0) == nil)
        local group = CreateFrame("Frame"):CreateAnimationGroup()
        group:SetIgnoreFramerateThrottle(true)
        assert(group:IsIgnoringFramerateThrottle())
    "#,
        )
        .unwrap();
}

#[test]
fn a_casts_unit_target_is_named_while_the_cast_runs() {
    use benilla_ui::script::pfui_host::{casts, units};
    use benilla_ui::script::{UnitGuids, UnitState};
    let mut script = extended();
    crate::spell::Catalog(Default::default())
        .install(script.lua())
        .unwrap();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        target: 7,
        ..Default::default()
    });
    let healer = UnitState {
        guid: 9,
        exists: true,
        has_object: true,
        name: Some("Healbot".into()),
        ..Default::default()
    };
    units::set_unit_states(script.lua(), [(9, healer, 0)]);
    let cast = casts::Cast {
        spell_id: 2054,
        name: "Heal".into(),
        rank: String::new(),
        icon: None,
        start_ms: 0.0,
        end_ms: 3000.0,
        tradeskill: false,
        cast_guid: "Cast-7-1".into(),
        delay_ms: 0,
        channel: false,
        silenceable: false,
        interruptible: false,
    };
    casts::set_casts(script.lua(), [(7, cast)]);
    casts::set_cast_targets(script.lua(), [(7, 9)]);
    script
        .run(r#"assert(UnitSpellTargetName("target") == "Healbot") assert(UnitSpellTargetName("player") == nil)"#)
        .unwrap();
    casts::set_casts(script.lua(), []);
    script
        .run(r#"assert(UnitSpellTargetName("target") == nil)"#)
        .unwrap();
}

/// ClassicAPI 1.20's name completion over this session's whispers: the most recent first, the
/// text before the cursor starting the name and the text after it later in the name.
#[test]
fn autocomplete_matches_whispered_names_most_recent_first() {
    let mut script = extended();
    for (event, who) in [
        ("CHAT_MSG_WHISPER", "Johnson"),
        ("CHAT_MSG_WHISPER_INFORM", "Jonas"),
        ("CHAT_MSG_WHISPER", "Mira"),
    ] {
        script.fire_event(
            event,
            vec![ScriptValue::Str("hi".into()), ScriptValue::Str(who.into())],
        );
    }
    script.tick(0.0);
    script
        .run(
            r#"
      local r = C_AutoComplete.GetAutoCompleteResults("jo", 0, 0, false, 0, 0)
      assert(table.getn(r) == 2 and r[1].name == "Jonas" and r[2].name == "Johnson", table.getn(r))
      assert(r[1].priority == Enum.AutoCompletePriority.Interacted and r[1].bnetID == 0)
      local mid = C_AutoComplete.GetAutoCompleteResults("Jhn", 0, 1, false, 0, 0)
      assert(table.getn(mid) == 1 and mid[1].name == "Johnson")
      assert(table.getn(C_AutoComplete.GetAutoCompleteResults("mira", 0, 0, false, 0, 0)) == 0)
      assert(table.getn(C_AutoComplete.GetAutoCompleteResults("mira", 0, 0, true, 0, 0)) == 1)
      local online = C_AutoComplete.GetAutoCompleteResults("", 1, 0, true, Enum.AutoCompleteEntryFlag.Online, 0)
      assert(table.getn(online) == 1 and online[1].name == "Mira")
      assert(table.getn(C_AutoComplete.GetAutoCompleteResults("", 0, 0, true, Enum.AutoCompleteEntryFlag.InGuild, 0)) == 0)
      assert(C_AutoComplete.IsRecognizedName("JOHNSON") and not C_AutoComplete.IsRecognizedName("Nobody"))
      assert(C_AutoComplete.GetAutoCompletePresenceID("Mira") == nil)
    "#,
        )
        .unwrap();
}

/// ClassicAPI 1.20's base-library additions: the stack heights an error handler reads, and the
/// reference's examples for the rest.
#[test]
fn baselib_additions_follow_the_reference() {
    let mut script = extended();
    script.set_screen_size(1024.0, 768.0);
    script
        .run(
            r#"
      local function depth() return GetCallstackHeight() end
      local outer = GetCallstackHeight()
      assert(depth() == outer + 1, depth() .. " " .. outer)
      local function where() local text = debugstack(1, 1, 0); return text end
      assert(string.find(where(), "in function `where'", 1, true), where())
      assert(string.find(debugstack(), "in main chunk", 1, true))
      handler = function(message)
        seen = message
        local height = GetErrorCallstackHeight()
        stackFromError = debugstack(GetCallstackHeight() - height + 1, 1, 0)
        -- The error level of an `error` call is that native, which has no locals; its caller's
        -- are one further out.
        atError = debuglocals(GetCallstackHeight() - height + 1)
        locals = debuglocals(GetCallstackHeight() - height + 2)
        skipped = debuglocals(GetCallstackHeight() - height + 2, true)
      end
      seterrorhandler(handler)
      assert(geterrorhandler() == handler)
      local f = CreateFrame("Frame")
      f:RegisterEvent("PLAYER_ALIVE")
      f:SetScript("OnEvent", function()
        local function failing() local hp, who, row = 42, "Kel", { size = 3 } local cb = function() end error("boom") end
        failing()
      end)
    "#,
        )
        .unwrap();
    script.fire_event("PLAYER_ALIVE", vec![]);
    let stack = script.eval::<String>("return stackFromError").unwrap();
    assert!(
        script
            .eval::<String>("return seen")
            .unwrap()
            .contains("boom"),
        "the addon's handler heard the error"
    );
    assert!(
        stack.starts_with("[C]: in function `error'"),
        "the error height names the failing code: {stack}"
    );
    script
        .run(
            r#"
      assert(atError == "", atError)
      assert(string.find(locals, 'hp = 42\nwho = "Kel"\nrow = <table> {\n size = 3\n}', 1, true), locals)
      assert(string.find(locals, 'cb = <function> defined [string "(chunk)"]:', 1, true), locals)
      assert(not string.find(skipped, "cb =", 1, true) and string.find(skipped, "hp = 42", 1, true), skipped)
      assert(debuglocals() == nil and GetErrorCallstackHeight() == nil)
      local proxy = newproxy(true)
      assert(type(proxy) == "userdata" and type(getmetatable(proxy)) == "table")
      getmetatable(proxy).__index = function(_, k) return k .. "!" end
      local twin = newproxy(proxy)
      assert(getmetatable(twin) == getmetatable(proxy) and twin.x == "x!")
      assert(getmetatable(newproxy()) == nil and getmetatable(newproxy(false)) == nil)
      assert(not pcall(newproxy, {}) and not pcall(newproxy, newproxy()))
      SetErrorCallstackHeight(4); assert(GetErrorCallstackHeight() == 4); SetErrorCallstackHeight(nil)
      local a, b, c, d, e = scrub(1, "a", true, {}, print)
      assert(a == 1 and b == "a" and c == true and d == nil and e == nil)
      assert(difftime(10, 4) == 6 and difftime(5) == 5)
      local t = { "a", "b", "c", "d", "e" }
      local x, y, z = table.removemulti(t, 2, 3)
      assert(x == "b" and y == "c" and z == "d" and table.getn(t) == 2 and t[2] == "e")
      assert(table.removemulti({}) == nil and not pcall(table.removemulti, { 1 }, 2, 3))
      local p, q, r = string.byte("ABC", 1, -1)
      assert(p == 65 and q == 66 and r == 67 and string.byte("ABC") == 65)
      local m, n2 = strbyte("ABC", -2, -1)
      assert(m == 66 and n2 == 67 and string.byte("ABC", 4) == nil)
      assert(string.gsub("$name the $class", "%$(%w+)", { name = "Bob", class = "Druid" }) == "Bob the Druid")
      assert(gsub("$name and $unknown", "%$(%w+)", { name = "Bob" }) == "Bob and $unknown")
      assert(string.gsub("aXb", "X", { X = "-" }) == "a-b")
      assert(not pcall(string.gsub, "a", "a", { a = {} }))
    "#,
        )
        .unwrap();
}

/// ClassicAPI 1.20's derived events: an enabled slot's range fires on enable and on change, a
/// usable change carries its list of slots, and a reactive spell glows each rank in the book.
#[test]
fn derived_events_fire_on_change_with_their_payloads() {
    let mut script = extended();
    script
        .run(
            r#"
      UnitGUID = function(u) if u == "player" then return "0x1" end end
      ranges, usables, glows = {}, {}, {}
      rangeAnswer = 1
      IsActionInRange = function(slot) return rangeAnswer end
      HasAction = function(slot) return slot == 5 end
      usableAnswer = 1
      IsUsableAction = function(slot) return usableAnswer, nil end
      GetSpellName = function(i) if i <= 2 then return "Overpower" end end
      GetSpellInfo = function(i, book)
        if book == "spell" then return "Overpower", nil, nil, nil, nil, nil, nil, nil, nil, 7383 + i end
      end
      C_Spell = C_Spell or {}
      C_Spell.IsSpellUsable = function() return spellUsable end
      local f = CreateFrame("Frame")
      for _, e in ipairs({ "ACTION_RANGE_CHECK_UPDATE", "ACTION_USABLE_CHANGED",
          "SPELL_ACTIVATION_OVERLAY_GLOW_SHOW", "SPELL_ACTIVATION_OVERLAY_GLOW_HIDE" }) do f:RegisterEvent(e) end
      f:SetScript("OnEvent", function()
        if event == "ACTION_RANGE_CHECK_UPDATE" then table.insert(ranges, { arg1, arg2, arg3 })
        elseif event == "ACTION_USABLE_CHANGED" then table.insert(usables, arg1)
        else table.insert(glows, { event, arg1, IsSpellOverlayed(arg1) }) end
      end)
      C_ActionBar.EnableActionRangeCheck(5, true)
    "#,
        )
        .unwrap();
    script.tick(0.01);
    script.tick(0.01);
    script.run("rangeAnswer = 0").unwrap();
    script.tick(0.01);
    script.run("rangeAnswer = nil").unwrap();
    script.tick(0.01);
    script.fire_event("PLAYER_ENTERING_WORLD", vec![]);
    script.run("usableAnswer = nil").unwrap();
    script.fire_event("ACTIONBAR_UPDATE_USABLE", vec![]);
    script.fire_event(
        "COMBAT_TEXT_UPDATE",
        vec![
            ScriptValue::Str("SPELL_ACTIVE".into()),
            ScriptValue::Str("Overpower".into()),
        ],
    );
    script.run("spellUsable = false").unwrap();
    script.fire_event("SPELL_UPDATE_USABLE", vec![]);
    script
        .run(
            r#"
      assert(table.getn(ranges) == 3, table.getn(ranges))
      assert(ranges[1][1] == 5 and ranges[1][2] == true and ranges[1][3] == true, "on enable")
      assert(ranges[2][2] == false and ranges[2][3] == true, "out of range")
      assert(ranges[3][3] == false, "no range check")
      assert(table.getn(usables) == 1 and usables[1][1].slot == 5 and usables[1][1].usable == false)
      assert(table.getn(glows) == 4, table.getn(glows))
      assert(glows[1][1] == "SPELL_ACTIVATION_OVERLAY_GLOW_SHOW" and glows[1][2] == 7384 and glows[1][3])
      assert(glows[2][2] == 7385 and glows[4][1] == "SPELL_ACTIVATION_OVERLAY_GLOW_HIDE" and not glows[4][3])
      assert(not IsSpellOverlayed(7384) and C_SpellActivationOverlay.IsSpellOverlayed == IsSpellOverlayed)
    "#,
        )
        .unwrap();
}

/// ClassicAPI's notInterruptible, by the server's rules: a known interrupt stops an interruptible
/// cast until the caster wears an immunity that blocks it, a hostile immunity blocks only with the
/// both-polarities attribute, a cast nothing can stop reads uninterruptible, and the target's
/// changes during a cast fire UNIT_SPELLCAST_(NOT_)INTERRUPTIBLE.
#[test]
fn not_interruptible_follows_known_interrupts_and_the_casters_immunities() {
    use benilla_ui::script::pfui_host::{
        auras::{self, Aura},
        casts,
        interrupts::{self, Immunity, Stopper},
        units,
    };
    use benilla_ui::script::{AuraState, UnitGuids, UnitState};
    let mut script = extended();
    crate::spell::Catalog(Default::default())
        .install(script.lua())
        .unwrap();
    script.set_unit_guids(&UnitGuids {
        player: 3,
        target: 7,
        ..Default::default()
    });
    let caster = UnitState {
        guid: 7,
        exists: true,
        has_object: true,
        name: Some("Caster".into()),
        ..Default::default()
    };
    units::set_unit_states(script.lua(), [(7, caster, 0)]);
    let cast = |silenceable, interruptible| casts::Cast {
        spell_id: 2054,
        name: "Heal".into(),
        rank: String::new(),
        icon: None,
        start_ms: 0.0,
        end_ms: 3000.0,
        tradeskill: false,
        cast_guid: "Cast-7-1".into(),
        delay_ms: 0,
        channel: false,
        silenceable,
        interruptible,
    };
    let immune = |aura, misc, helpful, both_polarities| Aura {
        slot: 0,
        crowd_control: false,
        source_guid: None,
        self_only: false,
        applies_aura: true,
        immunities: vec![Immunity {
            aura,
            misc,
            both_polarities,
        }],
        state: AuraState {
            spell_id: 642,
            name: Some("Shield".into()),
            helpful,
            ..Default::default()
        },
    };
    let read = |script: &mut UiScript| {
        script
            .eval::<Option<bool>>(r#"return (select(8, C_Spell.UnitCastingInfo("target")))"#)
            .unwrap()
    };
    script
        .run(
            r#"
      flips = {}
      local f = CreateFrame("Frame")
      f:RegisterEvent("UNIT_SPELLCAST_INTERRUPTIBLE")
      f:RegisterEvent("UNIT_SPELLCAST_NOT_INTERRUPTIBLE")
      f:SetScript("OnEvent", function() table.insert(flips, event .. ":" .. arg1) end)
    "#,
        )
        .unwrap();
    casts::set_casts(script.lua(), [(7, cast(true, true))]);
    // Without a known interrupt or silence the value is always false.
    assert_eq!(read(&mut script), Some(false));
    let kick = Stopper {
        silence: false,
        school_mask: 1,
        effect: 68,
        effect_mechanic: 26,
        ..Default::default()
    };
    interrupts::set_stoppers(script.lua(), vec![kick.clone()]);
    assert_eq!(read(&mut script), Some(false));
    script.tick(0.01);
    // A helpful physical immunity blocks a physical interrupt.
    auras::set_auras(script.lua(), 7, Some(vec![immune(39, 1, true, false)]));
    assert_eq!(read(&mut script), Some(true));
    script.tick(0.01);
    // A hostile one does not, unless it blocks both polarities (Banish).
    auras::set_auras(script.lua(), 7, Some(vec![immune(39, 127, false, false)]));
    assert_eq!(read(&mut script), Some(false));
    script.tick(0.01);
    auras::set_auras(script.lua(), 7, Some(vec![immune(39, 127, false, true)]));
    assert_eq!(read(&mut script), Some(true));
    // An arcane silence passes a physical immunity.
    let silence = Stopper {
        silence: true,
        school_mask: 64,
        effect: 6,
        effect_aura: 27,
        ..kick
    };
    interrupts::set_stoppers(script.lua(), vec![kick, silence]);
    auras::set_auras(script.lua(), 7, Some(vec![immune(39, 1, true, false)]));
    assert_eq!(read(&mut script), Some(false));
    // A cast no interrupt takes still falls to the silence; one neither takes cannot be stopped.
    casts::set_casts(script.lua(), [(7, cast(true, false))]);
    assert_eq!(read(&mut script), Some(false));
    casts::set_casts(script.lua(), [(7, cast(false, false))]);
    assert_eq!(read(&mut script), Some(true));
    let flips = script
        .eval::<String>("return table.concat(flips, ',')")
        .unwrap();
    assert_eq!(
        flips,
        "UNIT_SPELLCAST_NOT_INTERRUPTIBLE:target,UNIT_SPELLCAST_INTERRUPTIBLE:target"
    );
}

/// ClassicAPI's tiling: a tiled axis repeats the image at its own size, so its coordinates are
/// the texture's size over the image's, following the size; the file loads repeating; turning
/// the axis off gives back 0 and 1.
#[test]
fn a_tiled_texture_repeats_its_image_at_its_own_size() {
    let mut script = extended();
    script.set_texture_size_probe(Box::new(|_| Some((32, 16))));
    script
        .run(
            r#"
      local f = CreateFrame("Frame")
      f:SetWidth(100) f:SetHeight(20)
      f:SetPoint("TOPLEFT", UIParent or WorldFrame, "TOPLEFT", 0, 0)
      t = f:CreateTexture(nil, "OVERLAY")
      t:SetTexture("Interface\\AddOns\\pfUI\\img\\stripes")
      t:SetAllPoints(f)
      t:SetHorizTile(true)
      assert(t:GetHorizTile() and not t:GetVertTile())
      frame = f
    "#,
        )
        .unwrap();
    script.tick(0.01);
    let coords = |script: &mut UiScript| {
        script
            .eval::<String>(
                r#"local a = { t:GetTexCoord() } return string.format("%.4g %.4g %.4g %.4g", a[1], a[5], a[2], a[4])"#,
            )
            .unwrap()
    };
    assert_eq!(coords(&mut script), "0 3.125 0 1");
    script
        .run("frame:SetWidth(64) t:SetVertTile(true)")
        .unwrap();
    script.tick(0.01);
    assert_eq!(coords(&mut script), "0 2 0 1.25");
    script.run("t:SetHorizTile(false)").unwrap();
    assert_eq!(coords(&mut script), "0 1 0 1.25");
}

/// ClassicAPI's status bars: the bar texture lays out as the filled part, so a region anchored
/// to `GetStatusBarTexture()` moves with the fill's edge, and an empty bar's texture spans 0 at
/// the start edge; a vertical bar fills from the bottom.
#[test]
fn a_region_anchored_to_the_bar_texture_follows_the_fill() {
    let mut script = extended();
    script
        .run(
            r#"
      bar = CreateFrame("StatusBar")
      bar:SetWidth(200) bar:SetHeight(20)
      bar:SetPoint("BOTTOMLEFT", UIParent or WorldFrame, "BOTTOMLEFT", 100, 50)
      bar:SetStatusBarTexture("Interface\\TargetingFrame\\UI-StatusBar")
      bar:SetMinMaxValues(0, 100)
      bar:SetValue(25)
      over = bar:CreateTexture(nil, "OVERLAY")
      over:SetAllPoints(bar:GetStatusBarTexture())
    "#,
        )
        .unwrap();
    let span = |script: &mut UiScript| {
        script.tick(0.01);
        script
            .eval::<String>(
                r#"return string.format("%.4g %.4g %.4g %.4g", over:GetLeft(), over:GetRight(), over:GetBottom(), over:GetTop())"#,
            )
            .unwrap()
    };
    assert_eq!(span(&mut script), "100 150 50 70");
    script.run("bar:SetValue(100)").unwrap();
    assert_eq!(span(&mut script), "100 300 50 70");
    script.run("bar:SetValue(0)").unwrap();
    assert_eq!(span(&mut script), "100 100 50 70");
    script
        .run(r#"bar:SetOrientation("VERTICAL") bar:SetValue(50)"#)
        .unwrap();
    assert_eq!(span(&mut script), "100 300 50 60");
}
