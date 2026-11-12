-- A ClassicAPI `Cooldown` frame here is 1.12's indicator model (CooldownFrameTemplate), its sweep
-- driven by the model's sequence time as 1.12's CooldownFrame_SetTimer drives it. Reverse runs
-- the sweep backwards (it starts clear and darkens), a cleared draw-bling flag hides the frame at
-- the end instead of the flash, a cleared draw-swipe flag keeps the timing and draws nothing, and
-- a pause holds the sweep. Colors and textures are kept for their getters.
BenillaCooldownMixin = {}
local M = BenillaCooldownMixin

-- The countdown numbers a ClassicAPI client writes over the sweep while `countdownForCooldowns` is
-- on (registered at 0, as the client registers it), the frame does not hide them and the cooldown
-- outlasts its minimum: retail's text for the time left, whole units rounded up, tenths below the
-- milliseconds threshold and m:ss below the abbreviation threshold, which counts from one minute
-- to one hour.
local known, value = pcall(GetCVar, "countdownForCooldowns")
if not known or value == nil then RegisterCVar("countdownForCooldowns", "0") end
local mod = math.mod or math.fmod
local function ceilDiv(n, d) return math.floor(n / d) + (mod(n, d) ~= 0 and 1 or 0) end
local function countdownText(ms, abbrevThreshold, millisecondsThreshold)
  if ms / 1000 < millisecondsThreshold then
    local tenths = ceilDiv(ms, 100)
    return math.floor(tenths / 10) .. "." .. mod(tenths, 10)
  end
  local s = ceilDiv(ms, 1000)
  if s >= 86400 then return ceilDiv(s, 86400) .. "d" end
  if s >= 3600 then return ceilDiv(s, 3600) .. "h" end
  if s >= 60 and s >= math.min(abbrevThreshold, 3600) then return ceilDiv(s, 60) .. "m" end
  if s >= 60 then return string.format("%d:%02d", math.floor(s / 60), mod(s, 60)) end
  return tostring(s)
end
-- The numbers for `ms` left, none for nil; the font string is made the first time it shows any.
local function countdown(self, ms)
  local text = ""
  if ms and ms > 0 and not self:GetHideCountdownNumbers()
    and self.duration * 1000 > self:GetMinimumCountdownDuration()
    and (tonumber(GetCVar("countdownForCooldowns") or 0) or 0) ~= 0 then
    local formatter = self.benillaCountdownFormatter
    if formatter then
      -- A formatter replaces the rules: it gets the seconds left, unrounded; an error shows none.
      local ok, formatted = pcall(formatter.FormatNumber, formatter, ms / 1000)
      text = ok and formatted or ""
    else
      text = countdownText(ms, self:GetCountdownAbbrevThreshold(),
        self:GetCountdownMillisecondsThreshold())
    end
  end
  if text == (self.benillaCountdownText or "") then return end
  self.benillaCountdownText = text
  self:GetCountdownFontString():SetText(text)
end

-- The sweep's step, in place of the template's OnUpdateModel.
local function step()
  local self = this
  if self.benillaPaused then return end
  if self.stopping == 0 then
    local elapsed = GetTime() - self.start
    local finished = elapsed / self.duration
    if finished < 1 then
      countdown(self, math.floor((self.duration - elapsed) * 1000))
      if self:GetReverse() then finished = 1 - finished end
      self:SetSequenceTime(0, finished * 1000)
      return
    end
    countdown(self, nil)
    if not self:GetDrawBling() then
      self:Hide()
      return
    end
    self.stopping = 1
    self:SetSequence(1)
    self:SetSequenceTime(1, 0)
  else
    self:AdvanceTime()
  end
end

local function flag(name, default)
  M["Set" .. name] = function(self, value) self["benilla" .. name] = value end
  M["Get" .. name] = function(self)
    local value = self["benilla" .. name]
    if value == nil then return default end
    return value
  end
end
flag("Reverse", false)
flag("DrawEdge", false)
flag("DrawSwipe", true)
local setDrawSwipe = M.SetDrawSwipe
function M:SetDrawSwipe(draw)
  setDrawSwipe(self, draw)
  self:SetAlpha(draw == false and 0 or 1)
end
flag("DrawBling", true)
flag("HideCountdownNumbers", false)
flag("CountdownAbbrevThreshold", 120)
flag("CountdownMillisecondsThreshold", 0)
flag("MinimumCountdownDuration", 2000)
-- The frame keeps the formatter itself, so a later change to its breakpoints changes the text.
function M:SetCountdownFormatter(formatter)
  if formatter ~= nil and (type(formatter) ~= "table" or type(formatter.FormatNumber) ~= "function") then
    error("Usage: " .. (self:GetName() or "<unnamed>") .. ":SetCountdownFormatter([formatter])", 2)
  end
  self.benillaCountdownFormatter = formatter
end
function M:GetCountdownFormatter() return self.benillaCountdownFormatter end
function M:SetSwipeColor() end
function M:SetEdgeColor() end
function M:SetSwipeTexture() end
function M:SetEdgeTexture() end
function M:SetBlingTexture() end

function M:SetCooldown(start, duration, modRate)
  if start and duration and start > 0 and duration > 0 then
    self.benillaTimes = {start, duration, modRate or 1}
    self.start, self.duration, self.stopping = start, duration, 0
    if not self.benillaStepping then
      self.benillaStepping = true
      self:SetScript("OnUpdateModel", step)
    end
    self:SetSequence(0)
    self:Show()
  else
    self:Clear()
  end
end
function M:SetCooldownDuration(duration, modRate) self:SetCooldown(GetTime(), duration, modRate) end
function M:SetCooldownFromExpirationTime(expiration, duration, modRate)
  self:SetCooldown(expiration - duration, duration, modRate)
end
function M:Clear()
  self.benillaTimes = nil
  countdown(self, nil)
  self:Hide()
end
function M:GetCooldownTimes()
  local times = self.benillaTimes
  if not times then return 0, 0 end
  return times[1] * 1000, times[2] * 1000
end
function M:GetCooldownDuration()
  local _, duration = self:GetCooldownTimes()
  return duration
end
function M:GetCooldownDisplayDuration()
  local times = self.benillaTimes
  return times and times[2] * 1000 / times[3] or 0
end
function M:SetPaused(paused) self.benillaPaused = paused and true or false end
function M:IsPaused() return self.benillaPaused == true end
function M:Pause() self:SetPaused(true) end
function M:Resume() self:SetPaused(false) end
function M:GetCountdownFontString()
  if not self.benillaCountdown then
    self.benillaCountdown = self:CreateFontString(nil, "OVERLAY")
    self.benillaCountdown:SetFont("Fonts\\FRIZQT__.TTF", 16, "OUTLINE")
    self.benillaCountdown:SetPoint("CENTER", self, "CENTER", 0, -1)
  end
  return self.benillaCountdown
end
