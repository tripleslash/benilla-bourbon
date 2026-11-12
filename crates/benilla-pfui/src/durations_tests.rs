//! ClassicAPI 1.20's duration objects, text bindings, numeric formatter and timer bars, against
//! the examples of its API reference.

use benilla_ui::script::{QuadContent, UiScript};

fn extended() -> UiScript {
    let mut script = UiScript::new().unwrap();
    script.keep_calls();
    super::install(&script).unwrap();
    script
}

/// The fill's right edge (`u1`) of the one textured quad: the bar's shown fraction.
fn fill(script: &UiScript) -> f32 {
    script
        .extract()
        .iter()
        .find_map(|q| match &q.content {
            QuadContent::Texture {
                path: Some(_),
                tex_coords,
                ..
            } => tex_coords.map(|tc| tc.edges()[1]),
            _ => None,
        })
        .expect("the bar quad")
}

#[test]
fn numeric_formatter_follows_the_reference_examples() {
    let script = extended();
    script
        .run(
            r#"
      local Up, Down = Enum.NumericRuleFormatRounding.Up, Enum.NumericRuleFormatRounding.Down
      local f = C_StringUtil.CreateNumericRuleFormatter()
      f:SetBreakpoints({
        { threshold = 0, step = 0.1, rounding = Up, format = "%.1f" },
        { threshold = 5, step = 1, rounding = Up, format = "%d" },
        { threshold = 60, format = "%dm", components = { { div = 60, step = 1, rounding = Up } } },
      })
      assert(f:FormatNumber(3.24) == "3.3", f:FormatNumber(3.24))
      assert(f:FormatNumber(42.5) == "43")
      assert(f:FormatNumber(125) == "3m")
      assert(f:FormatNumber(-1) == "-1")
      -- Nearest rounds halfway up; the breakpoint is chosen before the rounding.
      local n = C_StringUtil.CreateNumericRuleFormatter()
      n:AddBreakpoint({ threshold = -100, step = 1, format = "%d" })
      assert(n:FormatNumber(2.5) == "3" and n:FormatNumber(-2.5) == "-2")
      local c = C_StringUtil.CreateNumericRuleFormatter()
      c:AddBreakpoint({ threshold = 0, format = "%d:%02d",
        components = { { div = 60, step = 1, rounding = Down }, { mod = 60, step = 1, rounding = Up } } })
      assert(c:FormatNumber(116) == "1:56" and c:FormatNumber(119.5) == "1:60")
      local m = C_StringUtil.CreateNumericRuleFormatter()
      m:AddBreakpoint({ threshold = 0, format = "%d", components = { { mod = 60 } } })
      assert(m:FormatNumber(65) == "5")
      -- Equal thresholds: the first added applies; the list reads largest first.
      local t = C_StringUtil.CreateNumericRuleFormatter()
      t:AddBreakpoint({ threshold = 0, format = "a" })
      t:AddBreakpoint({ threshold = 0, format = "b" })
      t:AddBreakpoint({ threshold = 10, format = "c", min = 1 })
      assert(t:FormatNumber(1) == "a")
      local list = t:GetBreakpoints()
      assert(list[1].threshold == 10 and list[1].min == 1 and list[1].step == 0 and list[1].rounding == 0)
      assert(list[2].format == "a" and list[2].min == nil and type(list[2].components) == "table")
      -- A wrong table leaves the breakpoints as they were; a copy is independent.
      assert(not pcall(t.SetBreakpoints, t, { { threshold = 0 } }))
      assert(t:FormatNumber(1) == "a")
      local copy = t:Copy(); copy:ClearBreakpoints()
      assert(copy:FormatNumber(9.99) == "9.99" and t:FormatNumber(1) == "a")
      local ok, err = pcall(function() local x = C_StringUtil.CreateNumericRuleFormatter()
        x:AddBreakpoint({ threshold = 0, format = "%d %d" }); return x:FormatNumber(1) end)
      assert(not ok and string.find(err, "NumericFormatter:FormatNumber():", 1, true))
      assert(not pcall(function() f.extra = 1 end))
      -- pfUI's abbreviation formatter (api.lua).
      local a = C_StringUtil.CreateNumericRuleFormatter()
      a:SetBreakpoints({
        { threshold = 0, format = "%d", components = { { step = 1, rounding = Down } } },
        { threshold = 1000, format = "%gk", components = { { div = 1000, step = 0.1, rounding = Down } } },
      })
      assert(a:FormatNumber(999.9) == "999" and a:FormatNumber(1270) == "1.2k" and a:FormatNumber(1700) == "1.7k")
    "#,
        )
        .unwrap();
}

#[test]
fn duration_objects_read_their_span_with_both_modifiers() {
    let mut script = extended();
    script.tick(10.0);
    script
        .run(
            r#"
      local d = C_DurationUtil.CreateDuration()
      assert(d:IsZero() and d:HasStarted() and d:HasExpired() and not d:IsActive())
      assert(d:GetElapsedPercent() == 1 and d:GetRemainingPercent() == 0 and d:GetModRate() == 1)
      local now = GetTime()
      d:SetTimeFromStart(now - 1, 4, 2)
      assert(math.abs(d:GetRemainingDuration() - 3) < 1e-6 and d:IsActive())
      local Base = Enum.DurationTimeModifier.BaseTime
      assert(math.abs(d:GetTotalDuration(Base) - 2) < 1e-6 and math.abs(d:GetRemainingDuration(Base) - 1) < 1e-6)
      d:SetTimeSpan(now + 5, now + 2)
      assert(d:IsZero() and d:GetStartTime() == now + 5 and not d:HasStarted())
      d:SetTimeFromEnd(now + 2, 4)
      assert(d:GetStartTime() == now - 2 and d:GetEndTime() == now + 2)
      local copy = d:Copy(); copy:Reset()
      assert(not d:IsZero() and copy:IsZero())
      copy:Assign(d)
      assert(copy:GetEndTime() == d:GetEndTime() and d:GetClock() == nil and not d:HasSecretValues())
      assert(not pcall(d.SetTimeFromStart, d, "x", 1) and not pcall(d.SetTimeFromStart, d, 1, 1, 0))
      assert(not pcall(d.GetTotalDuration, d, 7))
      local f = C_StringUtil.CreateNumericRuleFormatter()
      f:AddBreakpoint({ threshold = 0, format = "%.1f" })
      assert(d:FormatRemainingDuration(f) == "2.0" and d:FormatTotalDuration(f) == "4.0")
    "#,
        )
        .unwrap();
}

/// A timer bar fills by itself, reads 0 and 0, 0, keeps its duration object, fires no
/// OnValueChanged, and stays full after its end; SetMinMaxValues removes the timer.
#[test]
fn timer_bars_fill_from_their_duration_and_report_no_value() {
    let mut script = extended();
    script.set_screen_size(1024.0, 768.0);
    script.tick(10.0);
    script
        .run(
            r#"
      changed = 0
      bar = CreateFrame("StatusBar")
      bar:SetPoint("BOTTOMLEFT", nil, "BOTTOMLEFT", 0, 0)
      bar:SetWidth(200); bar:SetHeight(20)
      bar:SetStatusBarTexture("Interface\\TargetingFrame\\UI-StatusBar")
      bar:SetMinMaxValues(0, 100); bar:SetValue(10)
      bar:SetScript("OnValueChanged", function() changed = changed + 1 end)
      span = C_DurationUtil.CreateDuration()
      span:SetTimeFromStart(GetTime(), 4)
      bar:SetTimerDuration(span, Enum.StatusBarInterpolation.Immediate, Enum.StatusBarTimerDirection.ElapsedTime)
      local a, b = bar:GetMinMaxValues()
      assert(bar:GetValue() == 0 and a == 0 and b == 0)
      local d = bar:GetTimerDuration()
      assert(d ~= span and d:GetEndTime() == span:GetEndTime())
      assert(not pcall(bar.SetTimerDuration, bar, {}) and not pcall(bar.SetTimerDuration, bar, span, 5))
    "#,
        )
        .unwrap();
    script.tick(1.0);
    assert!((fill(&script) - 0.25).abs() < 1e-3, "{}", fill(&script));
    script.tick(1.0);
    assert!((fill(&script) - 0.5).abs() < 1e-3);
    // The bar keeps the object, not a copy: a span changed later shows on the next frame.
    script
        .run("span:SetTimeFromStart(GetTime(), 8, 1)")
        .unwrap();
    script.tick(2.0);
    assert!((fill(&script) - 0.25).abs() < 1e-3);
    script.tick(20.0);
    assert!((fill(&script) - 1.0).abs() < 1e-6, "full after its end");
    script
        .run(
            r#"
      assert(changed == 0, "the timer runs no OnValueChanged")
      assert(bar:GetTimerDuration())
      bar:SetValue(50)
      assert(bar:GetTimerDuration(), "SetValue keeps the timer")
      bar:SetMinMaxValues(0, 100)
      assert(bar:GetTimerDuration() == nil and bar:GetValue() == 50)
    "#,
        )
        .unwrap();
    script.tick(0.0);
    assert!((fill(&script) - 0.5).abs() < 1e-6, "the value's fill again");
    // RemainingTime empties.
    script
        .run(
            r#"
      local d = C_DurationUtil.CreateDuration(); d:SetTimeFromStart(GetTime(), 2)
      bar:SetTimerDuration(d, nil, Enum.StatusBarTimerDirection.RemainingTime)
    "#,
        )
        .unwrap();
    script.tick(0.5);
    assert!((fill(&script) - 0.75).abs() < 1e-3);
}

#[test]
fn eased_values_report_the_target_and_land_on_it() {
    let mut script = extended();
    script.set_screen_size(1024.0, 768.0);
    script
        .run(
            r#"
      changed = {}
      bar = CreateFrame("StatusBar")
      bar:SetPoint("BOTTOMLEFT", nil, "BOTTOMLEFT", 0, 0)
      bar:SetWidth(200); bar:SetHeight(20)
      bar:SetStatusBarTexture("Interface\\TargetingFrame\\UI-StatusBar")
      bar:SetMinMaxValues(0, 100); bar:SetValue(0)
      bar:SetScript("OnValueChanged", function() table.insert(changed, arg1) end)
      bar:SetValue(100, Enum.StatusBarInterpolation.ExponentialEaseOut)
      assert(bar:GetValue() == 100 and bar:IsInterpolating() and table.getn(changed) == 1)
    "#,
        )
        .unwrap();
    script.tick(0.0);
    script.tick(0.05);
    let mid = fill(&script);
    assert!(mid > 0.5 && mid < 0.7, "1 - exp(-0.9): {mid}");
    script
        .run("local v = bar:GetInterpolatedValue(); assert(v > 50 and v < 70, v)")
        .unwrap();
    for _ in 0..30 {
        script.tick(0.05);
    }
    assert!((fill(&script) - 1.0).abs() < 1e-6);
    script
        .run("assert(not bar:IsInterpolating() and bar:GetInterpolatedValue() == 100)")
        .unwrap();
    // An Immediate value stops the ease.
    script
        .run(
            r#"
      bar:SetValue(0, Enum.StatusBarInterpolation.ExponentialEaseOut)
      bar:SetValue(30)
      assert(not bar:IsInterpolating())
    "#,
        )
        .unwrap();
    script.tick(0.0);
    assert!((fill(&script) - 0.3).abs() < 1e-6);
}

/// The castbar's binding (pfUI `castbar.lua`): "{} / {}" of the elapsed and total seconds,
/// written on enable and on each frame while the text changes; zero and expired texts.
#[test]
fn text_bindings_write_their_duration_text_each_frame() {
    let mut script = extended();
    script.tick(10.0);
    script
        .run(
            r#"
      local holder = CreateFrame("Frame")
      text = holder:CreateFontString(nil, "OVERLAY")
      text:SetFont("Fonts\\FRIZQT__.TTF", 12)
      local f = C_StringUtil.CreateNumericRuleFormatter()
      f:AddBreakpoint({ threshold = 0, step = .1, format = "%.1f" })
      span = C_DurationUtil.CreateDuration()
      span:SetTimeFromStart(GetTime(), 2.5)
      binding = C_DurationUtil.CreateDurationTextBinding()
      binding:SetFontString(text)
      binding:SetDuration(span)
      binding:SetTextFormat("{} / {}", {
        { property = Enum.DurationTextBindingProperty.ElapsedDuration, formatter = f },
        { property = Enum.DurationTextBindingProperty.TotalDuration, formatter = f },
      })
      assert(not binding:IsEnabled() and binding:CanUpdateFontString())
      binding:Enable()
      assert(text:GetText() == "0.0 / 2.5", text:GetText())
      binding = nil
    "#,
        )
        .unwrap();
    script.tick(1.0);
    script
        .run(
            r#"
      assert(text:GetText() == "1.0 / 2.5", text:GetText())
      local b = C_DurationUtil.CreateDurationTextBinding()
      local f = C_StringUtil.CreateNumericRuleFormatter()
      f:AddBreakpoint({ threshold = 0, format = "%d" })
      b:SetFormatter(f)
      assert(b:GetFormattedText() == "0")
      b:SetZeroDurationText("none"); b:SetExpiredText("done")
      assert(b:GetFormattedText() == "none")
      local d = C_DurationUtil.CreateDuration(); d:SetTimeFromStart(GetTime() - 5, 1)
      b:SetDuration(d)
      assert(b:GetFormattedText() == "done")
      local copy = b:Copy()
      assert(not copy:IsEnabled() and copy:GetExpiredText() == "done")
      b:SetToDefaults()
      assert(b:GetFormattedText() == "" and not b:CanFormatText())
    "#,
        )
        .unwrap();
}

#[test]
fn unit_cast_durations_carry_the_cast_times_in_seconds() {
    let script = extended();
    script
        .run(
            r#"
      C_Spell = C_Spell or {}
      C_Spell.UnitCastingInfo = function(unit) if unit == "player" then return "Cast", nil, nil, 1000, 3500 end end
      C_Spell.UnitChannelInfo = function() end
      local d = UnitCastingDuration("player")
      assert(d:GetStartTime() == 1 and d:GetEndTime() == 3.5)
      assert(UnitCastingDuration("target") == nil and UnitChannelDuration("player") == nil)
      assert(UnitCastingDuration("player") ~= d)
    "#,
        )
        .unwrap();
}
