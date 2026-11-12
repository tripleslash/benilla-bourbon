-- Independently implemented pfUI utility subset. Keep legacy FrameXML's globals usable.
-- The VM's legacy vararg table retains its argument count, including trailing nils.
function SafePack(...) return arg end

-- Friend-note buttons use this ClassicAPI utility as their OnLeave callback.
function GameTooltip_Hide()
  GameTooltip:Hide()
end

function Mixin(object, ...)
  for i = 1, arg.n do
    for key, value in pairs(arg[i]) do object[key] = value end
  end
  return object
end

function CreateFromMixins(...)
  return Mixin({}, unpack(arg, 1, arg.n))
end

function CreateAndInitFromMixin(mixin, ...)
  local object = Mixin({}, mixin)
  object:Init(unpack(arg, 1, arg.n))
  return object
end

function tInvert(values)
  local inverted = {}
  for key, value in pairs(values) do inverted[value] = key end
  return inverted
end

ColorMixin = {}
function ColorMixin:GetRGB() return self.r, self.g, self.b end
function ColorMixin:GetRGBA() return self.r, self.g, self.b, self.a end
function ColorMixin:SetRGB(r, g, b) self.r, self.g, self.b = r, g, b end
function ColorMixin:SetRGBA(r, g, b, a) self.r, self.g, self.b, self.a = r, g, b, a end
local function byte(value)
  return math.floor(math.max(0, math.min(1, value)) * 255)
end
function ColorMixin:GenerateHexColor()
  return string.format("%02x%02x%02x%02x", byte(self.a or 1), byte(self.r), byte(self.g), byte(self.b))
end
function ColorMixin:GenerateHexColorMarkup() return "|c" .. self:GenerateHexColor() end
function ColorMixin:WrapTextInColorCode(text) return self:GenerateHexColorMarkup() .. text .. "|r" end
function CreateColor(r, g, b, a)
  return Mixin({r = r, g = g, b = b, a = a or 1}, ColorMixin)
end
C_ColorUtil = {}
-- An engine function: it reads the color's fields itself, as ClassicAPI's ColorMixin calls it.
function C_ColorUtil.GenerateTextColorCode(color)
  return string.format("%02x%02x%02x%02x", byte(color.a or 1), byte(color.r), byte(color.g), byte(color.b))
end

CallbackRegistryMixin = {}
function CallbackRegistryMixin:OnLoad()
  self.callbacks, self.definedEvents = {}, {}
end
function CallbackRegistryMixin:SetUndefinedEventsAllowed(allowed)
  self.undefinedEventsAllowed = allowed
end
function CallbackRegistryMixin:GenerateCallbackEvents(events)
  for _, name in ipairs(events) do self.definedEvents[name] = true end
end
function CallbackRegistryMixin:RegisterCallback(name, callback, owner, ...)
  assert(self.undefinedEventsAllowed or self.definedEvents[name], "undefined callback event: " .. name)
  assert(type(callback) == "function", "callback must be a function")
  local entries = self.callbacks[name] or {}
  self.callbacks[name] = entries
  local entry = {callback = callback, owner = owner, extra = arg}
  table.insert(entries, entry)
  return entry
end
function CallbackRegistryMixin:UnregisterCallback(name, owner)
  local entries = self.callbacks[name] or {}
  for i = table.getn(entries), 1, -1 do
    if entries[i].owner == owner then
      entries[i].removed = true
      table.remove(entries, i)
    end
  end
end
function CallbackRegistryMixin:UnregisterAllCallbacks(owner)
  for name in pairs(self.callbacks) do self:UnregisterCallback(name, owner) end
end
function CallbackRegistryMixin:TriggerEvent(name, ...)
  local snapshot = {}
  for _, entry in ipairs(self.callbacks[name] or {}) do table.insert(snapshot, entry) end
  for _, entry in ipairs(snapshot) do
    if not entry.removed then
      local values = {entry.owner}
      local n = 1
      for i = 1, entry.extra.n do n = n + 1; values[n] = entry.extra[i] end
      for i = 1, arg.n do n = n + 1; values[n] = arg[i] end
      entry.callback(unpack(values, 1, n))
    end
  end
end

-- Player-login readiness belongs to this VM; it is reset when the UI is rebuilt.
local loggedIn = false
local loginFrame = CreateFrame("Frame")
loginFrame:RegisterEvent("PLAYER_LOGIN")
loginFrame:SetScript("OnEvent", function()
  loggedIn = true
  loginFrame:UnregisterEvent("PLAYER_LOGIN")
end)
function IsLoggedIn() return loggedIn end
EventUtil = {}
local function continueWhen(eventName, ready, callback, matches)
  if ready then callback(); return end
  local frame = CreateFrame("Frame")
  frame:RegisterEvent(eventName)
  frame:SetScript("OnEvent", function()
    if not matches or matches(arg1) then
      frame:UnregisterEvent(eventName)
      frame:SetScript("OnEvent", nil)
      callback()
    end
  end)
end
function EventUtil.ContinueOnPlayerLogin(callback)
  continueWhen("PLAYER_LOGIN", loggedIn, callback)
end
function EventUtil.ContinueOnAddOnLoaded(name, callback)
  continueWhen("ADDON_LOADED", IsAddOnLoaded(name), callback,
    function(loaded) return string.lower(loaded) == string.lower(name) end)
end
-- Preserve the table identity while removing every keyed and sequence entry.
function table.wipe(value)
  for key in pairs(value) do value[key] = nil end
  return value
end
-- ClassicAPI registers it as a global too, which its addon reads back as `ClassicAPI.wipe`.
wipe = table.wipe
