-- ClassicAPI's duration objects (`C_DurationUtil`), their text bindings, the numeric formatter
-- (`C_StringUtil.CreateNumericRuleFormatter`), timer status bars and eased fills, and the
-- `Unit*Duration`/`GetTotemDuration` getters, as ClassicAPI 1.20's API reference specifies them.
-- Times are GetTime seconds. The chunk returns its per-frame pass, which the host runs after the
-- frame's OnUpdate walk, beside C_Timer's: enabled bindings write their text, timer and easing
-- bars move their fill. The fill a bar shows apart from its value is the engine's `set_display`.
local tables, engine = ...
local statusbar = tables.statusbar

local function finite(n) return type(n) == "number" and n == n and n ~= math.huge and n ~= -math.huge end
local function check(ok, message) if not ok then error(message, 3) end end
-- The remainder with the sign of the dividend, as C's fmod gives it.
local function fmod(a, b)
  local q = a / b
  if q >= 0 then q = math.floor(q) else q = math.ceil(q) end
  return a - b * q
end
local function strict(kind)
  return function() error("Attempt to modify a " .. kind, 2) end
end

Enum = Enum or {}
Enum.DurationTimeModifier = { RealTime = 0, BaseTime = 1 }
Enum.DurationTextBindingProperty = { RemainingDuration = 0, RemainingPercent = 1, ElapsedDuration = 2,
  ElapsedPercent = 3, TotalDuration = 4, StartTime = 5, EndTime = 6 }
Enum.StatusBarInterpolation = { Immediate = 0, ExponentialEaseOut = 1 }
Enum.StatusBarTimerDirection = { ElapsedTime = 0, RemainingTime = 1 }
Enum.NumericRuleFormatRounding = { Nearest = 0, Up = 1, Down = 2 }

---------------------------------------------------------------------------------------------------
-- Numeric formatter
---------------------------------------------------------------------------------------------------
local Rounding = Enum.NumericRuleFormatRounding
local breakpointsOf = setmetatable({}, { __mode = "k" })

-- A multiple of `step`; a quotient within 1e-9 of a whole number counts as whole, so 0.7 / 0.1
-- rounds down to 7, not 6.
local function round(value, step, rounding)
  if not step or step <= 0 then return value end
  local q = value / step
  local whole = math.floor(q + 0.5)
  if math.abs(q - whole) < 1e-9 then q = whole end
  if rounding == Rounding.Up then q = math.ceil(q)
  elseif rounding == Rounding.Down then q = math.floor(q)
  else q = math.floor(q + 0.5) end
  return q * step
end

local function optionalNumber(t, key) return t[key] == nil or finite(t[key]) end
local function validRounding(r) return r == nil or r == 0 or r == 1 or r == 2 end
local function validBreakpoint(bp)
  if type(bp) ~= "table" or not finite(bp.threshold) or type(bp.format) ~= "string" then return false end
  if not (optionalNumber(bp, "step") and optionalNumber(bp, "min") and optionalNumber(bp, "max")) then return false end
  if not validRounding(bp.rounding) then return false end
  if bp.components ~= nil then
    if type(bp.components) ~= "table" then return false end
    for _, c in ipairs(bp.components) do
      if type(c) ~= "table" or not (optionalNumber(c, "div") and optionalNumber(c, "mod")
          and optionalNumber(c, "step")) or not validRounding(c.rounding) then
        return false
      end
    end
  end
  return true
end
local function copyBreakpoint(bp)
  local copy = { threshold = bp.threshold, step = bp.step or 0, rounding = bp.rounding or 0,
    min = bp.min, max = bp.max, format = bp.format, components = {} }
  for i, c in ipairs(bp.components or {}) do
    copy.components[i] = { div = c.div, mod = c.mod, step = c.step, rounding = c.rounding }
  end
  return copy
end

local Formatter = {}
local FormatterMeta = { __index = Formatter, __newindex = strict("NumericRuleFormatter"), __metatable = false }
local function newFormatter(list)
  local formatter = setmetatable({}, FormatterMeta)
  breakpointsOf[formatter] = list or {}
  return formatter
end
local function own(self)
  local list = breakpointsOf[self]
  if not list then error("Usage: formatter:Method(...) on a NumericRuleFormatter", 3) end
  return list
end
function Formatter:AddBreakpoint(bp)
  local list = own(self)
  check(validBreakpoint(bp), "Usage: formatter:AddBreakpoint(breakpoint)")
  table.insert(list, copyBreakpoint(bp))
end
function Formatter:SetBreakpoints(bps)
  local list = own(self)
  check(type(bps) == "table", "Usage: formatter:SetBreakpoints(breakpoints)")
  for _, bp in ipairs(bps) do check(validBreakpoint(bp), "Usage: formatter:SetBreakpoints(breakpoints)") end
  for i = table.getn(list), 1, -1 do table.remove(list, i) end
  for _, bp in ipairs(bps) do table.insert(list, copyBreakpoint(bp)) end
end
function Formatter:ClearBreakpoints()
  local list = own(self)
  for i = table.getn(list), 1, -1 do table.remove(list, i) end
end
-- The largest threshold first; equal thresholds keep the order they were added in.
function Formatter:GetBreakpoints()
  local order = {}
  for i, bp in ipairs(own(self)) do order[i] = { bp = bp, index = i } end
  table.sort(order, function(a, b)
    if a.bp.threshold ~= b.bp.threshold then return a.bp.threshold > b.bp.threshold end
    return a.index < b.index
  end)
  local out = {}
  for i, entry in ipairs(order) do
    local copy = copyBreakpoint(entry.bp)
    if entry.bp.min == nil then copy.min = nil end
    if entry.bp.max == nil then copy.max = nil end
    out[i] = copy
  end
  return out
end
function Formatter:Copy()
  local list = {}
  for i, bp in ipairs(own(self)) do list[i] = copyBreakpoint(bp) end
  return newFormatter(list)
end
function Formatter:FormatNumber(number)
  local list = own(self)
  check(finite(number), "Usage: formatter:FormatNumber(number)")
  local chosen
  for _, bp in ipairs(list) do
    if bp.threshold <= number and (not chosen or bp.threshold > chosen.threshold) then chosen = bp end
  end
  if not chosen then return tostring(number) end
  local value = round(number, chosen.step, chosen.rounding)
  if chosen.min and value < chosen.min then value = chosen.min end
  if chosen.max and value > chosen.max then value = chosen.max end
  local ok, text
  if table.getn(chosen.components) > 0 then
    local values = {}
    for i, c in ipairs(chosen.components) do
      local v = value
      if c.div and c.div ~= 0 then v = v / c.div end
      if c.mod and c.mod ~= 0 then v = fmod(v, c.mod) end
      values[i] = round(v, c.step, c.rounding)
    end
    ok, text = pcall(string.format, chosen.format, unpack(values))
  else
    ok, text = pcall(string.format, chosen.format, value)
  end
  if not ok then error("NumericFormatter:FormatNumber(): " .. tostring(text), 2) end
  return text
end

C_StringUtil = C_StringUtil or {}
function C_StringUtil.CreateNumericRuleFormatter() return newFormatter() end
local function isFormatter(value) return type(value) == "table" and breakpointsOf[value] ~= nil end

---------------------------------------------------------------------------------------------------
-- Duration objects: a Lua object over a shared span, so two objects can hold one time span.
---------------------------------------------------------------------------------------------------
local spanOf = setmetatable({}, { __mode = "k" })
local Duration = {}
local DurationMeta = { __index = Duration, __newindex = strict("duration object"), __metatable = false }
local function wrap(span)
  local object = setmetatable({}, DurationMeta)
  spanOf[object] = span
  return object
end
local function newSpan() return { start = 0, total = 0, rate = 1 } end
local function span(self)
  local s = spanOf[self]
  if not s then error("Usage: duration:Method(...) on a duration object", 3) end
  return s
end
local function isDuration(value) return type(value) == "table" and spanOf[value] ~= nil end
local function modRate(rate)
  if rate == nil then return 1 end
  check(finite(rate) and rate > 0, "Usage: modRate must be more than 0")
  return rate
end
-- The total a modifier reads: BaseTime divides it by the mod rate, from the same start.
local function total(s, modifier)
  check(modifier == nil or modifier == 0 or modifier == 1, "Usage: modifier must be an Enum.DurationTimeModifier")
  if modifier == 1 then return s.total / s.rate end
  return s.total
end
local function elapsed(s, modifier)
  local t = total(s, modifier)
  local e = GetTime() - s.start
  if e < 0 then return 0 elseif e > t then return t end
  return e
end

function Duration:SetTimeFromStart(startTime, duration, rate)
  local s = span(self)
  check(finite(startTime) and finite(duration), "Usage: duration:SetTimeFromStart(startTime, duration [, modRate])")
  s.start, s.total, s.rate = startTime, math.max(duration, 0), modRate(rate)
end
function Duration:SetTimeFromEnd(endTime, duration, rate)
  local s = span(self)
  check(finite(endTime) and finite(duration), "Usage: duration:SetTimeFromEnd(endTime, duration [, modRate])")
  duration = math.max(duration, 0)
  s.start, s.total, s.rate = endTime - duration, duration, modRate(rate)
end
function Duration:SetTimeSpan(startTime, endTime)
  local s = span(self)
  check(finite(startTime) and finite(endTime), "Usage: duration:SetTimeSpan(startTime, endTime)")
  s.start, s.total = startTime, math.max(endTime - startTime, 0)
end
function Duration:Reset()
  local s = span(self)
  s.start, s.total, s.rate = 0, 0, 1
end
Duration.SetToDefaults = Duration.Reset
function Duration:Assign(other)
  local s = span(self)
  check(isDuration(other), "Usage: duration:Assign(other)")
  local o = spanOf[other]
  s.start, s.total, s.rate = o.start, o.total, o.rate
end
function Duration:GetStartTime(modifier) local s = span(self); total(s, modifier); return s.start end
function Duration:GetEndTime(modifier) local s = span(self); return s.start + total(s, modifier) end
function Duration:GetTotalDuration(modifier) return total(span(self), modifier) end
function Duration:GetElapsedDuration(modifier) return elapsed(span(self), modifier) end
function Duration:GetRemainingDuration(modifier)
  local s = span(self)
  return total(s, modifier) - elapsed(s, modifier)
end
function Duration:GetElapsedPercent(modifier)
  local s = span(self)
  local t = total(s, modifier)
  if t <= 0 then return 1 end
  return elapsed(s, modifier) / t
end
function Duration:GetRemainingPercent(modifier)
  local s = span(self)
  local t = total(s, modifier)
  if t <= 0 then return 0 end
  return (t - elapsed(s, modifier)) / t
end
function Duration:HasStarted(modifier) local s = span(self); total(s, modifier); return GetTime() >= s.start end
function Duration:HasExpired(modifier) local s = span(self); return GetTime() >= s.start + total(s, modifier) end
function Duration:IsActive(modifier)
  local s = span(self)
  local now = GetTime()
  return now >= s.start and now < s.start + total(s, modifier)
end
function Duration:IsZero() return span(self).total == 0 end
function Duration:GetModRate() return span(self).rate end
function Duration:Copy()
  local s = span(self)
  return wrap({ start = s.start, total = s.total, rate = s.rate })
end
function Duration:GetClockTime() span(self); return GetTime() end
function Duration:GetClock() span(self); return nil end
function Duration:HasSecretValues() span(self); return false end
local function formatWith(self, formatter, read, modifier)
  check(isFormatter(formatter), "Usage: duration:Format...(formatter [, modifier])")
  return formatter:FormatNumber(read(self, modifier))
end
function Duration:FormatElapsedDuration(f, m) return formatWith(self, f, Duration.GetElapsedDuration, m) end
function Duration:FormatRemainingDuration(f, m) return formatWith(self, f, Duration.GetRemainingDuration, m) end
function Duration:FormatTotalDuration(f, m) return formatWith(self, f, Duration.GetTotalDuration, m) end

C_DurationUtil = C_DurationUtil or {}
function C_DurationUtil.CreateDuration() return wrap(newSpan()) end

-- A cast's or channel's span, from the millisecond times C_Spell gives; nothing without one. Each
-- call is a new object holding the span of its call.
local function castDuration(info, unit)
  local name, _, _, startMs, endMs = info(unit)
  if not name or not startMs or not endMs then return end
  local d = wrap(newSpan())
  d:SetTimeSpan(startMs / 1000, endMs / 1000)
  return d
end
function UnitCastingDuration(unit) return castDuration(C_Spell.UnitCastingInfo, unit) end
function UnitChannelDuration(unit) return castDuration(C_Spell.UnitChannelInfo, unit) end
-- The totem's span from its appearance for its full duration; zero for an empty slot.
function GetTotemDuration(slot)
  local haveTotem, _, startTime, duration = GetTotemInfo(slot)
  local d = wrap(newSpan())
  if haveTotem and startTime and duration and duration > 0 then d:SetTimeFromStart(startTime, duration) end
  return d
end

---------------------------------------------------------------------------------------------------
-- Text bindings
---------------------------------------------------------------------------------------------------
local Property = Enum.DurationTextBindingProperty
local readProperty = {
  [Property.RemainingDuration] = Duration.GetRemainingDuration,
  [Property.RemainingPercent] = Duration.GetRemainingPercent,
  [Property.ElapsedDuration] = Duration.GetElapsedDuration,
  [Property.ElapsedPercent] = Duration.GetElapsedPercent,
  [Property.TotalDuration] = Duration.GetTotalDuration,
  [Property.StartTime] = Duration.GetStartTime,
  [Property.EndTime] = Duration.GetEndTime,
}
local bindingOf = setmetatable({}, { __mode = "k" })
-- Enabled bindings, held strongly: one keeps updating with no reference left to it.
local enabledBindings = {}
local Binding = {}
local BindingMeta = { __index = Binding, __newindex = strict("duration text binding"), __metatable = false }
local function defaults(b)
  b.fontString, b.duration, b.formatter, b.textFormat, b.components = nil, nil, nil, nil, nil
  b.expiredText, b.zeroText, b.interval, b.modifier = nil, nil, 0, 0
  b.lastText, b.lastUpdate = nil, nil
end
local function newBinding()
  local object = setmetatable({}, BindingMeta)
  local b = { enabled = false }
  defaults(b)
  bindingOf[object] = b
  return object
end
local function binding(self)
  local b = bindingOf[self]
  if not b then error("Usage: binding:Method(...) on a duration text binding", 3) end
  return b
end
local function optionalText(text, usage)
  check(text == nil or type(text) == "string" or type(text) == "number", usage)
  return text ~= nil and tostring(text) or nil
end
local function canFormat(b) return b.formatter ~= nil or b.textFormat ~= nil end
local function formatted(b)
  if not canFormat(b) then return "" end
  local d = b.duration
  if (not d or d:IsZero()) and b.zeroText then return b.zeroText end
  if not d then d = wrap(newSpan()) end
  if b.expiredText and d:HasExpired(b.modifier) then return b.expiredText end
  if not b.textFormat then return b.formatter:FormatNumber(d:GetRemainingDuration(b.modifier)) end
  local index = 0
  local text = string.gsub(b.textFormat, "{}", function()
    index = index + 1
    local c = b.components[index]
    if not c then return "" end
    return c.formatter:FormatNumber(readProperty[c.property](d, b.modifier))
  end)
  return text
end
local function write(b)
  if not (b.fontString and canFormat(b)) then return end
  local text = formatted(b)
  if text ~= b.lastText then
    b.lastText = text
    pcall(b.fontString.SetText, b.fontString, text)
  end
end

function Binding:SetFontString(fontString)
  local b = binding(self)
  check(fontString == nil or type(fontString) == "table" and type(fontString.SetText) == "function",
    "Usage: binding:SetFontString(fontString)")
  b.fontString, b.lastText = fontString, nil
end
function Binding:GetFontString() return binding(self).fontString end
function Binding:SetDuration(duration)
  local b = binding(self)
  check(duration == nil or isDuration(duration), "Usage: binding:SetDuration(duration)")
  b.duration = duration
end
function Binding:GetDuration() return binding(self).duration end
function Binding:SetFormatter(formatter)
  local b = binding(self)
  check(isFormatter(formatter), "Usage: binding:SetFormatter(formatter)")
  b.formatter, b.textFormat, b.components = formatter, nil, nil
end
function Binding:SetTextFormat(formatString, components)
  local b = binding(self)
  check(type(formatString) == "string" and type(components) == "table",
    "Usage: binding:SetTextFormat(formatString, components)")
  local list = {}
  for i, c in ipairs(components) do
    check(type(c) == "table" and readProperty[c.property] and isFormatter(c.formatter),
      "Usage: binding:SetTextFormat(formatString, components)")
    list[i] = { property = c.property, formatter = c.formatter }
  end
  b.textFormat, b.components, b.formatter = formatString, list, nil
end
function Binding:SetExpiredText(text) binding(self).expiredText = optionalText(text, "Usage: binding:SetExpiredText([text])") end
function Binding:GetExpiredText() return binding(self).expiredText end
function Binding:SetZeroDurationText(text) binding(self).zeroText = optionalText(text, "Usage: binding:SetZeroDurationText([text])") end
function Binding:GetZeroDurationText() return binding(self).zeroText end
function Binding:SetUpdateInterval(seconds)
  check(finite(seconds) and seconds >= 0, "Usage: binding:SetUpdateInterval(seconds)")
  binding(self).interval = seconds
end
function Binding:GetUpdateInterval() return binding(self).interval end
function Binding:SetTimeModifier(modifier)
  check(modifier == 0 or modifier == 1, "Usage: binding:SetTimeModifier(modifier)")
  binding(self).modifier = modifier
end
function Binding:GetTimeModifier() return binding(self).modifier end
function Binding:SetEnabled(enabled)
  local b = binding(self)
  if enabled then
    b.enabled = true
    enabledBindings[self] = b
    b.lastUpdate = GetTime()
    write(b)
  else
    b.enabled = false
    enabledBindings[self] = nil
  end
end
function Binding:Enable() self:SetEnabled(true) end
function Binding:Disable() self:SetEnabled(false) end
function Binding:IsEnabled() return binding(self).enabled end
function Binding:GetFormattedText() return formatted(binding(self)) end
function Binding:UpdateFontString() write(binding(self)) end
function Binding:CanFormatText() return canFormat(binding(self)) end
function Binding:CanUpdateFontString() local b = binding(self); return canFormat(b) and b.fontString ~= nil end
local function assign(to, from)
  to.fontString, to.duration, to.formatter = from.fontString, from.duration, from.formatter
  to.textFormat, to.components = from.textFormat, from.components
  to.expiredText, to.zeroText, to.interval, to.modifier = from.expiredText, from.zeroText, from.interval, from.modifier
  to.lastText = nil
end
function Binding:Copy()
  local copy = newBinding()
  assign(bindingOf[copy], binding(self))
  return copy
end
function Binding:Assign(other)
  local b = binding(self)
  check(type(other) == "table" and bindingOf[other] ~= nil, "Usage: binding:Assign(other)")
  assign(b, bindingOf[other])
end
function Binding:SetToDefaults()
  local b = binding(self)
  defaults(b)
  self:SetEnabled(false)
end
function Binding:HasSecretValues() binding(self); return false end

function C_DurationUtil.CreateDurationTextBinding() return newBinding() end

---------------------------------------------------------------------------------------------------
-- Status bars: timers and eased fills
---------------------------------------------------------------------------------------------------
local native = {
  SetValue = statusbar.SetValue, GetValue = statusbar.GetValue,
  SetMinMaxValues = statusbar.SetMinMaxValues, GetMinMaxValues = statusbar.GetMinMaxValues,
}
local barState = {}  -- bar -> { timer = span, direction, ease = { shown, target }, last }
local Interpolation, Direction = Enum.StatusBarInterpolation, Enum.StatusBarTimerDirection

local function valueFraction(bar)
  local min, max = native.GetMinMaxValues(bar)
  local value = native.GetValue(bar)
  if not (min and max and value) or max - min <= 0 then return 0 end
  local f = (value - min) / (max - min)
  if f < 0 then return 0 elseif f > 1 then return 1 end
  return f
end
local function timerFraction(state)
  local d = wrap(state.timer)
  if state.direction == Direction.RemainingTime then return d:GetRemainingPercent() end
  return d:GetElapsedPercent()
end
-- The fill the bar shows now: an ease's, a timer's, or the value's.
local function shownFraction(bar)
  local state = barState[bar]
  if state and state.ease then return state.ease.shown end
  if state and state.timer then return timerFraction(state) end
  return valueFraction(bar)
end
local function settle(bar)
  local state = barState[bar]
  if state and not state.timer and not state.ease then
    barState[bar] = nil
    engine.set_display(bar, nil)
  end
end
local function startEase(bar, from)
  local state = barState[bar] or {}
  barState[bar] = state
  state.ease = { shown = from }
  state.last = nil
end

function statusbar:SetTimerDuration(duration, interpolation, direction)
  check(isDuration(duration), "Usage: statusBar:SetTimerDuration(duration [, interpolation [, direction]])")
  check(interpolation == nil or interpolation == 0 or interpolation == 1,
    "Usage: interpolation must be an Enum.StatusBarInterpolation")
  check(direction == nil or direction == 0 or direction == 1,
    "Usage: direction must be an Enum.StatusBarTimerDirection")
  local from = shownFraction(self)
  local state = barState[self] or {}
  barState[self] = state
  state.timer, state.direction = spanOf[duration], direction or Direction.ElapsedTime
  if interpolation == Interpolation.ExponentialEaseOut then
    startEase(self, from)
  else
    state.ease = nil
    engine.set_display(self, timerFraction(state))
  end
end
function statusbar:GetTimerDuration()
  local state = barState[self]
  if state and state.timer then return wrap(state.timer) end
  return nil
end
function statusbar:GetValue()
  local state = barState[self]
  if state and state.timer then return 0 end
  return native.GetValue(self)
end
function statusbar:GetMinMaxValues()
  local state = barState[self]
  if state and state.timer then return 0, 0 end
  return native.GetMinMaxValues(self)
end
function statusbar:SetValue(value, interpolation)
  local from = shownFraction(self)
  native.SetValue(self, value)
  local state = barState[self]
  if interpolation == Interpolation.ExponentialEaseOut then
    startEase(self, from)
  elseif state then
    state.ease = nil
    if state.timer then engine.set_display(self, timerFraction(state)) end
    settle(self)
  end
end
-- Removes the timer. An eased range keeps the fill's part of the bar and eases to the value's.
function statusbar:SetMinMaxValues(min, max, interpolation)
  local from = shownFraction(self)
  native.SetMinMaxValues(self, min, max)
  local state = barState[self]
  if state then state.timer = nil end
  if interpolation == Interpolation.ExponentialEaseOut then
    startEase(self, from)
  elseif state then
    state.ease = nil
    settle(self)
  end
end
function statusbar:GetInterpolatedValue()
  local state = barState[self]
  if state and state.ease and not state.timer then
    local min, max = native.GetMinMaxValues(self)
    return min + state.ease.shown * (max - min)
  end
  return self:GetValue()
end
function statusbar:IsInterpolating()
  local state = barState[self]
  return state ~= nil and state.ease ~= nil
end
function statusbar:SetToTargetValue()
  local state = barState[self]
  if not state then return end
  state.ease = nil
  if state.timer then engine.set_display(self, timerFraction(state)) end
  settle(self)
end

-- The ease multiplies the gap by exp(-18 * elapsed) a frame and lands below 0.1% of the bar. A
-- hidden bar neither moves nor eases; it resumes from its first shown frame.
local function moveBars(now)
  for bar, state in pairs(barState) do
    if bar:IsVisible() then
      local target = state.timer and timerFraction(state) or valueFraction(bar)
      local ease = state.ease
      if ease then
        local dt = state.last and now - state.last or 0
        ease.shown = target + (ease.shown - target) * math.exp(-18 * dt)
        if math.abs(ease.shown - target) < 0.001 then
          state.ease = nil
          engine.set_display(bar, state.timer and target or nil)
          settle(bar)
        else
          engine.set_display(bar, ease.shown)
        end
      else
        engine.set_display(bar, target)
      end
      state.last = now
    else
      state.last = nil
    end
  end
end

local function writeBindings(now)
  for _, b in pairs(enabledBindings) do
    if b.interval <= 0 or not b.lastUpdate or now - b.lastUpdate >= b.interval then
      b.lastUpdate = now
      write(b)
    end
  end
end

return function()
  local now = GetTime()
  moveBars(now)
  writeBindings(now)
end
