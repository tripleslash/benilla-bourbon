-- ClassicAPI 1.20's state events that the client derives from what 1.12 already answers, each a
-- per-frame (or per-event) diff raised through the engine's dispatcher, as its API reference
-- specifies them: action range and usability, spell glows and alerts, unit range and sight, spell
-- haste, the mount display and the interface scale.
local bridge = ...
local fire = bridge.fire

-- One frame drives every diff after the frame's OnUpdate walk has run its own.
local driver = CreateFrame("Frame")
local passes = {}
local function onFrame(pass) table.insert(passes, pass) end

---------------------------------------------------------------------------------------------------
-- ACTION_RANGE_CHECK_UPDATE: slot, isInRange, checksRange for the slots with an enable count.
---------------------------------------------------------------------------------------------------
local rangeCount, rangeState, rangeFresh = {}, {}, {}
C_ActionBar = C_ActionBar or {}
function C_ActionBar.EnableActionRangeCheck(slot, enable)
  slot = tonumber(slot)
  if not slot or slot < 1 or slot > 120 or slot ~= math.floor(slot) then return end
  local count = rangeCount[slot] or 0
  if enable then
    rangeCount[slot], rangeFresh[slot] = count + 1, true
  elseif count > 0 then
    rangeCount[slot] = count - 1
    if count == 1 then rangeCount[slot], rangeState[slot], rangeFresh[slot] = nil, nil, nil end
  end
end
onFrame(function()
  if not next(rangeCount) then return end
  for slot in pairs(rangeCount) do
    local answer = IsActionInRange(slot)
    local state = answer == nil and "none" or (answer == 1 or answer == true) and "in" or "out"
    if rangeFresh[slot] or state ~= rangeState[slot] then
      rangeFresh[slot], rangeState[slot] = nil, state
      fire("ACTION_RANGE_CHECK_UPDATE", slot, state == "in", state ~= "none")
    end
  end
end)

---------------------------------------------------------------------------------------------------
-- ACTION_USABLE_CHANGED: the slots whose IsUsableAction answer changed, right after
-- ACTIONBAR_UPDATE_USABLE; a slot's own change is ACTIONBAR_SLOT_CHANGED's, which only re-reads it.
---------------------------------------------------------------------------------------------------
local usable = {}
local function usableOf(slot)
  local u, noMana = IsUsableAction(slot)
  return u and true or false, noMana and true or false
end
local usableFrame = CreateFrame("Frame")
usableFrame:RegisterEvent("ACTIONBAR_UPDATE_USABLE")
usableFrame:RegisterEvent("ACTIONBAR_SLOT_CHANGED")
usableFrame:RegisterEvent("PLAYER_ENTERING_WORLD")
usableFrame:SetScript("OnEvent", function()
  if event == "ACTIONBAR_SLOT_CHANGED" then
    local slot = tonumber(arg1)
    if slot and slot >= 1 and slot <= 120 then
      local u, n = usableOf(slot)
      usable[slot] = { u, n }
    end
    return
  end
  local changes = {}
  for slot = 1, 120 do
    if HasAction(slot) then
      local u, n = usableOf(slot)
      local was = usable[slot]
      if event == "ACTIONBAR_UPDATE_USABLE" and was and (was[1] ~= u or was[2] ~= n) then
        table.insert(changes, { slot = slot, usable = u, noMana = n })
      end
      usable[slot] = { u, n }
    else
      usable[slot] = nil
    end
  end
  if table.getn(changes) > 0 then fire("ACTION_USABLE_CHANGED", changes) end
end)

---------------------------------------------------------------------------------------------------
-- Spell glows and alerts: a reactive spell (COMBAT_TEXT_UPDATE SPELL_ACTIVE) glows every
-- spellbook spell of its name while it stays usable; Shadow Trance glows Shadow Bolt and, with
-- Clearcasting, shows an alert.
---------------------------------------------------------------------------------------------------
local glowing, glowByName = {}, {}
local function bookIDs(name)
  local ids, i = {}, 1
  while true do
    local spell = GetSpellName(i, "spell")
    if not spell then break end
    if spell == name then
      local id = select(10, GetSpellInfo(i, "spell"))
      if id then table.insert(ids, id) end
    end
    i = i + 1
  end
  return ids
end
local function glow(name, on, source)
  local key = source .. ":" .. name
  if on == (glowByName[key] ~= nil) then return end
  if on then
    local ids = bookIDs(name)
    glowByName[key] = ids
    for _, id in ipairs(ids) do
      glowing[id] = (glowing[id] or 0) + 1
      if glowing[id] == 1 then fire("SPELL_ACTIVATION_OVERLAY_GLOW_SHOW", id) end
    end
  else
    local ids = glowByName[key]
    glowByName[key] = nil
    for _, id in ipairs(ids) do
      glowing[id] = (glowing[id] or 1) - 1
      if glowing[id] <= 0 then
        glowing[id] = nil
        fire("SPELL_ACTIVATION_OVERLAY_GLOW_HIDE", id)
      end
    end
  end
end
C_SpellActivationOverlay = C_SpellActivationOverlay or {}
function C_SpellActivationOverlay.IsSpellOverlayed(spellID) return glowing[tonumber(spellID)] ~= nil end
IsSpellOverlayed = C_SpellActivationOverlay.IsSpellOverlayed
Enum = Enum or {}
Enum.ScreenLocationType = { Center = 0, Left = 1, Right = 2, Top = 3, Bottom = 4, TopLeft = 5,
  TopRight = 6, LeftOutside = 7, RightOutside = 8, LeftRight = 9, TopBottom = 10, LeftRightOutside = 11 }
-- The two proc auras 1.12 keeps unchanged: aura, art, location, scale, and the spell it glows.
local procs = {
  { aura = 17941, texture = "Textures\\SpellActivationOverlays\\Nightfall", glows = 686 },
  { aura = 16870, texture = "Textures\\SpellActivationOverlays\\Natures_Grace" },
}
local alerts = {}
local function hasAura(spellID)
  for i = 1, 32 do
    local name, _, _, _, _, _, _, _, _, id = C_UnitAuras.UnitAura("player", i, "HELPFUL")
    if not name then return false end
    if id == spellID then return true end
  end
  return false
end
local function scanProcs(redraw)
  for _, proc in ipairs(procs) do
    local on = hasAura(proc.aura)
    if on ~= (alerts[proc.aura] == true) or (redraw and on) then
      alerts[proc.aura] = on or nil
      if on then
        fire("SPELL_ACTIVATION_OVERLAY_SHOW", proc.aura, proc.texture, Enum.ScreenLocationType.LeftRight, 1, 255, 255, 255)
      else
        fire("SPELL_ACTIVATION_OVERLAY_HIDE", proc.aura)
      end
    end
    if proc.glows then
      local name = GetSpellInfo(proc.glows)
      if name then glow(name, on, "proc" .. proc.aura) end
    end
  end
end
local glowFrame = CreateFrame("Frame")
glowFrame:RegisterEvent("COMBAT_TEXT_UPDATE")
glowFrame:RegisterEvent("SPELL_UPDATE_USABLE")
glowFrame:RegisterEvent("PLAYER_AURAS_CHANGED")
glowFrame:RegisterEvent("PLAYER_ENTERING_WORLD")
glowFrame:SetScript("OnEvent", function()
  if event == "COMBAT_TEXT_UPDATE" and arg1 == "SPELL_ACTIVE" and arg2 then
    glow(arg2, true, "reactive")
  elseif event == "SPELL_UPDATE_USABLE" then
    for key, ids in pairs(glowByName) do
      local name = string.sub(key, 1, 9) == "reactive:" and string.sub(key, 10)
      if name and not (C_Spell.IsSpellUsable(ids[1] or name)) then glow(name, false, "reactive") end
    end
  elseif event == "PLAYER_AURAS_CHANGED" then
    scanProcs(false)
  elseif event == "PLAYER_ENTERING_WORLD" then
    -- Every active alert shows again after a HIDE with no spell, so a /reload redraws them.
    fire("SPELL_ACTIVATION_OVERLAY_HIDE")
    scanProcs(true)
  end
end)

---------------------------------------------------------------------------------------------------
-- UNIT_IN_RANGE_UPDATE and UNIT_DISTANCE_CHECK_UPDATE over the group, their pets, pet, target
-- and focus: a change of UnitInRange's answer (the first check fires nothing) and of sight.
---------------------------------------------------------------------------------------------------
local inRange, inSight = {}, {}
local function groupUnits()
  local list = { "pet", "target", "focus" }
  local raid = GetNumRaidMembers()
  if raid > 0 then
    for i = 1, raid do table.insert(list, "raid" .. i); table.insert(list, "raidpet" .. i) end
  end
  for i = 1, GetNumPartyMembers() do table.insert(list, "party" .. i); table.insert(list, "partypet" .. i) end
  return list
end
local function isMember(unit) return string.find(unit, "^party%d") or string.find(unit, "^raid%d") end
local rangeTick = 0
onFrame(function(elapsed)
  if not UnitGUID("player") then return end
  rangeTick = rangeTick + elapsed
  if rangeTick < 0.1 then return end
  rangeTick = 0
  local seen = {}
  for _, unit in ipairs(groupUnits()) do
    seen[unit] = true
    local guid = UnitGUID(unit)
    if guid then
      local r = UnitInRange(unit) and true or false
      local key = unit .. guid
      if inRange[unit] and inRange[unit].key == key and inRange[unit].value ~= r then
        fire("UNIT_IN_RANGE_UPDATE", unit, r)
      end
      inRange[unit] = { key = key, value = r }
      if isMember(unit) and not string.find(unit, "pet") then
        local v = UnitIsVisible(unit) and true or false
        if inSight[unit] ~= nil and inSight[unit] ~= v then fire("UNIT_DISTANCE_CHECK_UPDATE", unit, v) end
        inSight[unit] = v
      end
    else
      inRange[unit], inSight[unit] = nil, nil
    end
  end
  for unit in pairs(inRange) do if not seen[unit] then inRange[unit] = nil end end
  for unit in pairs(inSight) do if not seen[unit] then inSight[unit] = nil end end
end)

---------------------------------------------------------------------------------------------------
-- UNIT_SPELL_HASTE, PLAYER_MOUNT_DISPLAY_CHANGED and UI_SCALE_CHANGED: a change of
-- UnitSpellHaste for the player, target and focus, of the player's mounted state, and of
-- UIParent's scale or the screen's size (also a uiScale/useUiScale set that changed neither).
---------------------------------------------------------------------------------------------------
local haste, mounted = {}, nil
local scale, width, height, scaleSet
local nativeSetCVar = SetCVar
function SetCVar(name, value, ...)
  local result = nativeSetCVar(name, value, unpack(arg))
  local lower = type(name) == "string" and string.lower(name)
  if lower == "uiscale" or lower == "useuiscale" then scaleSet = true end
  return result
end
onFrame(function()
  for _, unit in ipairs({ "player", "target", "focus" }) do
    local h = UnitGUID(unit) and UnitSpellHaste and UnitSpellHaste(unit) or nil
    if h and haste[unit] and haste[unit] ~= h then fire("UNIT_SPELL_HASTE", unit) end
    haste[unit] = h
  end
  if not UnitGUID("player") then return end
  local m = IsMounted() and true or false
  if mounted ~= nil and m ~= mounted then fire("PLAYER_MOUNT_DISPLAY_CHANGED") end
  mounted = m
  if not UIParent then return end
  local s, w, h = UIParent:GetScale(), GetScreenWidth(), GetScreenHeight()
  if scale and (scaleSet or s ~= scale or w ~= width or h ~= height) then fire("UI_SCALE_CHANGED") end
  scale, width, height, scaleSet = s, w, h, nil
end)

---------------------------------------------------------------------------------------------------
-- UNIT_SPELLCAST_INTERRUPTIBLE / UNIT_SPELLCAST_NOT_INTERRUPTIBLE: a change of the
-- notInterruptible value during the target's or focus's cast; a cast's first value fires nothing.
---------------------------------------------------------------------------------------------------
local shield = {}
local function castShield(unit)
  local name, _, _, startMs, _, _, castID, noInterrupt = C_Spell.UnitCastingInfo(unit)
  if name then return "cast" .. tostring(castID) .. tostring(startMs), noInterrupt and true or false end
  local spellID
  name, _, _, startMs, _, _, noInterrupt, spellID = C_Spell.UnitChannelInfo(unit)
  if name then return "channel" .. tostring(spellID) .. tostring(startMs), noInterrupt and true or false end
end
onFrame(function()
  if not UnitGUID("player") then return end
  for _, unit in ipairs({ "target", "focus" }) do
    local guid = UnitGUID(unit)
    local key, value
    if guid then key, value = castShield(unit) end
    if key then key = guid .. key end
    local was = shield[unit]
    if key and was and was.key == key and was.value ~= value then
      fire(value and "UNIT_SPELLCAST_NOT_INTERRUPTIBLE" or "UNIT_SPELLCAST_INTERRUPTIBLE", unit)
    end
    shield[unit] = key and { key = key, value = value } or nil
  end
end)

driver:SetScript("OnUpdate", function()
  local elapsed = arg1
  for _, pass in ipairs(passes) do pass(elapsed) end
end)
