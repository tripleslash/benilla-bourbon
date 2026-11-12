-- The engine functions that read namespaces the rest of the install builds (C_Item, C_Spell,
-- C_Container, C_EquipmentSet): loaded last, before the ClassicAPI table is taken.
local bridge = ...
local strfind = string.find
function IsChatAFK() return UnitIsAFK("player") and true or false end
function IsChatDND() return UnitIsDND("player") and true or false end

function C_AddOns.GetAddOnName(addon) return (GetAddOnInfo(addon)) end

function C_Item.GetItemIconByID(item)
  local _, _, _, _, icon = C_Item.GetItemInfoInstant(item)
  return icon
end

-- The bag search: an item matches when the text is part of its name, its subclass, its equip
-- slot's name or the name of an equipment set holding it.
local search = ""
local function setsHolding(bag, slot)
  local names = {}
  if not (C_EquipmentSet and C_EquipmentSet.GetEquipmentSetIDs) then return names end
  for _, id in ipairs(C_EquipmentSet.GetEquipmentSetIDs() or {}) do
    for _, location in pairs(C_EquipmentSet.GetItemLocations(id) or {}) do
      local _, _, inBags, setSlot, setBag = EquipmentManager_UnpackLocation(location)
      if inBags and setSlot == slot and setBag == bag then
        tinsert(names, (C_EquipmentSet.GetEquipmentSetInfo(id)))
        break
      end
    end
  end
  return names
end
function C_Container.GetContainerItemEquipmentSetInfo(bag, slot)
  local names = setsHolding(bag, slot)
  if table.getn(names) == 0 then return false, nil end
  return true, table.concat(names, PLAYER_LIST_DELIMITER or ", ")
end
local function matches(bag, slot, link)
  if search == "" then return true end
  local _, _, name = strfind(link or "", "%[(.-)%]")
  local itemID = C_Container.GetContainerItemID(bag, slot)
  local _, _, subType, equipLoc = C_Item.GetItemInfoInstant(itemID or 0)
  for _, text in ipairs({name or "", subType or "", equipLoc and getglobal(equipLoc) or ""}) do
    if strfind(string.lower(text), search, 1, true) then return true end
  end
  for _, set in ipairs(setsHolding(bag, slot)) do
    if strfind(string.lower(set), search, 1, true) then return true end
  end
  return false
end
function C_Container.SetItemSearch(text)
  text = string.lower(text or "")
  if text == search then return end
  search = text
  bridge.fire("INVENTORY_SEARCH_UPDATE")
end
function C_Container.GetContainerItemInfo(bag, slot)
  local texture, count, locked, quality, readable = GetContainerItemInfo(bag, slot)
  if not texture then return nil end
  local link = GetContainerItemLink(bag, slot)
  return {
    iconFileID = texture, stackCount = count, isLocked = locked and true or false,
    quality = quality, isReadable = readable and true or false, hasLoot = false,
    hyperlink = link, isFiltered = not matches(bag, slot, link), hasNoValue = false,
    itemID = C_Container.GetContainerItemID(bag, slot), isBound = false,
  }
end

-- The ClassicAPI addon this layer's engine half is written for, v1.20.6: once it has loaded, the
-- version an addon's load gate reads is that one. Without it the layer stands alone at 1.15.10.
local versionFrame = CreateFrame("Frame")
versionFrame:RegisterEvent("ADDON_LOADED")
versionFrame:SetScript("OnEvent", function()
  if arg1 == "!!!ClassicAPI" then
    CLASSIC_API_VERSION = 12006
    BenillaClassicAPICompatibility.apiVersion = "1.20.6"
    versionFrame:UnregisterEvent("ADDON_LOADED")
  end
end)

-- The paper doll slot for an inventory slot number, the reverse of GetInventorySlotInfo.
local slotNames = {
  "HeadSlot", "NeckSlot", "ShoulderSlot", "ShirtSlot", "ChestSlot", "WaistSlot", "LegsSlot",
  "FeetSlot", "WristSlot", "HandsSlot", "Finger0Slot", "Finger1Slot", "Trinket0Slot",
  "Trinket1Slot", "BackSlot", "MainHandSlot", "SecondaryHandSlot", "RangedSlot", "TabardSlot",
  "Bag0Slot", "Bag1Slot", "Bag2Slot", "Bag3Slot",
}
slotNames[0] = "AmmoSlot"
C_PaperDollInfo = C_PaperDollInfo or {}
function C_PaperDollInfo.GetInventorySlotInfoForInvSlot(invSlot)
  if type(invSlot) ~= "number" then
    error("Usage: C_PaperDollInfo.GetInventorySlotInfoForInvSlot(invSlot)", 2)
  end
  local name = slotNames[invSlot]
  if not name then return end
  local id, texture = GetInventorySlotInfo(name)
  return id, texture, invSlot == 18, name
end

-- Loss of control: the player's harmful auras whose spell applies a control aura, by its aura
-- type, and a school lockout the server sent (SCHOOL_INTERRUPT). A per-frame diff raises
-- LOSS_OF_CONTROL_ADDED for each new effect and one LOSS_OF_CONTROL_UPDATE when the set changed.
local controlAuras = {
  [2] = "POSSESS", [5] = "CONFUSE", [6] = "CHARM", [7] = "FEAR", [12] = "STUN", [25] = "PACIFY",
  [26] = "ROOT", [27] = "SILENCE", [60] = "PACIFYSILENCE", [67] = "DISARM",
}
local controlOf = {}
local function controlType(spellID)
  local known = controlOf[spellID]
  if known ~= nil then return known or nil end
  local found = false
  local effects = C_Spell and C_Spell.GetSpellEffectInfo and C_Spell.GetSpellEffectInfo(spellID)
  for _, effect in ipairs(effects or {}) do
    if controlAuras[effect.auraName] then found = controlAuras[effect.auraName] break end
  end
  controlOf[spellID] = found
  return found or nil
end
local active = {}
local function lockout(now)
  local l = bridge.school_lockout()
  if l and l.ends > now then return l.mask, l.ends - now, l end
  return 0, 0
end
local function scan()
  local now, list = GetTime(), {}
  local mask, remaining, l = lockout(now)
  if mask ~= 0 then
    tinsert(list, {
      locType = "SCHOOL_INTERRUPT", spellID = 0, displayText = INTERRUPTED or "Interrupted",
      iconTexture = "", startTime = l.starts, timeRemaining = remaining,
      duration = l.ends - l.starts, lockoutSchool = mask, priority = 0, displayType = 2,
    })
  end
  for i = 1, 16 do
    local name, icon, _, _, duration, expiration, _, _, _, spellID =
      C_UnitAuras.UnitAura("player", i, "HARMFUL")
    if not name then break end
    local locType = spellID and controlType(spellID)
    if locType then
      local timed = duration and duration > 0 and expiration and expiration > 0
      tinsert(list, {
        locType = locType, spellID = spellID, displayText = name, iconTexture = icon or "",
        startTime = timed and expiration - duration or nil,
        timeRemaining = timed and math.max(0, expiration - now) or nil,
        duration = timed and duration or nil, lockoutSchool = 0, priority = 0, displayType = 2,
      })
    end
  end
  return list
end
local function key(entry) return entry.locType .. ":" .. entry.spellID .. ":" .. entry.lockoutSchool end
local lossFrame = CreateFrame("Frame")
lossFrame:SetScript("OnUpdate", function()
  if not UnitExists("player") then return end
  local list = scan()
  local before = {}
  for _, entry in ipairs(active) do before[key(entry)] = true end
  local changed = table.getn(list) ~= table.getn(active)
  active = list
  for index, entry in ipairs(list) do
    if not before[key(entry)] then
      changed = true
      bridge.fire("LOSS_OF_CONTROL_ADDED", index)
    end
  end
  if changed then bridge.fire("LOSS_OF_CONTROL_UPDATE", "player") end
end)
C_LossOfControl = C_LossOfControl or {}
function C_LossOfControl.GetActiveLossOfControlDataCount() return table.getn(active) end
-- The locked schools (within `filterMask` when given) and the seconds left on them.
function C_LossOfControl.GetSchoolLockout(filterMask)
  local mask, remaining = lockout(GetTime())
  if filterMask then mask = bit.band(mask, filterMask) end
  if mask == 0 then return 0, 0 end
  return mask, remaining
end
function C_LossOfControl.GetActiveLossOfControlData(index)
  local entry = active[index]
  if not entry then return nil end
  local copy = {}
  for field, value in pairs(entry) do copy[field] = value end
  if copy.startTime then copy.timeRemaining = math.max(0, copy.startTime + copy.duration - GetTime()) end
  return copy
end

-- CopyToClipboard: the client writes the text once this frame's scripts are done.
function CopyToClipboard(text, removeMarkup)
  text = tostring(text or "")
  if removeMarkup then
    text = string.gsub(text, "|c%x%x%x%x%x%x%x%x", "")
    text = string.gsub(text, "|r", "")
    text = string.gsub(text, "|H.-|h(.-)|h", "%1")
    text = string.gsub(text, "|T.-|t", "")
    text = string.gsub(text, "|A.-|a", "")
    text = string.gsub(text, "|n", "\n")
    text = string.gsub(text, "||", "|")
  end
  bridge.clipboard(text)
  return string.len(text)
end

-- The macro slash commands' frame, item and aura helpers (the ClassicAPI addon's
-- Util\SlashCommandsRegistry.lua). `/click`'s frame: a global frame whose own name is the one asked
-- for, so a reassigned global never stands in for it.
function GetClickFrame(name)
  if type(name) ~= "string" then return nil end
  local frame = getglobal(name)
  if type(frame) ~= "table" or type(frame.GetName) ~= "function" then return nil end
  local ok, own = pcall(frame.GetName, frame)
  if ok and own == name then return frame end
  return nil
end

-- What a link matches by `item`: an item id, an `item:` string or link, else a name, any case.
local function itemMatcher(item)
  local id = tonumber(item)
  if not id and type(item) == "string" then
    local _, _, linked = strfind(item, "item:(%d+)")
    id = tonumber(linked)
  end
  if id then
    return function(link)
      local _, _, have = strfind(link, "item:(%d+)")
      return tonumber(have) == id
    end
  end
  if type(item) ~= "string" or item == "" then return nil end
  local wanted = strlower(item)
  return function(link)
    local _, _, name = strfind(link, "%[(.-)%]")
    return name ~= nil and strlower(name) == wanted
  end
end

-- Where `item` is: a bag and slot, else an equipment slot. A location names its own slot; a
-- reference takes the first bag match, then the first equipped one.
local function findItem(item)
  if type(item) == "table" then
    if item.equipmentSlotIndex then return nil, nil, item.equipmentSlotIndex end
    if item.bagID and item.slotIndex then return item.bagID, item.slotIndex end
    return nil
  end
  local matches = itemMatcher(item)
  if not matches then return nil end
  for bag = 0, 4 do
    for slot = 1, GetContainerNumSlots(bag) do
      local link = GetContainerItemLink(bag, slot)
      if link and matches(link) then return bag, slot end
    end
  end
  for slot = 1, 19 do
    local link = GetInventoryItemLink("player", slot)
    if link and matches(link) then return nil, nil, slot end
  end
  return nil
end

-- `C_Item.EquipItemByName(item [, dstSlot])`: into `dstSlot` (1-19), from a bag or another
-- equipment slot; without one, the slot the item's type picks, from a bag only. Whatever the
-- cursor held goes back first.
function C_Item.EquipItemByName(item, dstSlot)
  ClearCursor()
  local bag, slot, equipped = findItem(item)
  dstSlot = tonumber(dstSlot)
  if dstSlot then
    if dstSlot < 1 or dstSlot > 19 or equipped == dstSlot then return end
    if equipped then
      PickupInventoryItem(equipped)
    elseif bag then
      PickupContainerItem(bag, slot)
    else
      return
    end
    EquipCursorItem(dstSlot)
  elseif bag then
    PickupContainerItem(bag, slot)
    AutoEquipCursorItem()
  end
end

-- An item id for `item`, a carried item's name included.
local function itemID(item)
  if type(item) == "table" then return C_Item.GetItemID(item) end
  local bag, slot, equipped = findItem(item)
  local link = equipped and GetInventoryItemLink("player", equipped)
    or bag and GetContainerItemLink(bag, slot)
  local _, _, id = strfind(link or tostring(item or ""), "item:(%d+)")
  return tonumber(id) or tonumber(item)
end

-- Whether the item goes in an equipment slot: any inventory type, bags and ammo included. An
-- uncached item is not.
function C_Item.IsEquippableItem(item)
  local id = itemID(item)
  local inventoryType = id and C_Item.GetItemInventoryTypeByID(id)
  return inventoryType ~= nil and inventoryType ~= 0
end
if not IsEquippableItem then
  function IsEquippableItem(item) return C_Item.IsEquippableItem(item) and 1 or nil end
end

-- The item's on-use spell, `(name, spellID)`, or nothing.
function C_Item.GetItemSpell(item)
  local spellID = bridge.item_use_spell(item)
  if not spellID then return nil end
  return (GetSpellInfo(spellID)), spellID
end
if not GetItemSpell then GetItemSpell = C_Item.GetItemSpell end

-- `/cancelaura`: the player's helpful aura of that name, a rank suffix ignored, cancelled through
-- its spell.
function CancelSpellByName(name)
  if type(name) ~= "string" then return end
  local wanted = strlower((string.gsub(name, "%s*%(.-%)%s*$", "")))
  if wanted == "" then return end
  for i = 1, 32 do
    local auraName, _, _, _, _, _, _, _, _, spellID = C_UnitAuras.UnitAura("player", i, "HELPFUL")
    if not auraName then return end
    if strlower(auraName) == wanted and spellID then
      C_Spell.CancelSpellByID(spellID)
      return
    end
  end
end

-- `UnitSelectionType(unit [, useExtendedColors])`: the rung of the selection colour's ladder, as
-- ClassicAPI reads it. A unit no player controls takes its reaction to the player (0 hostile,
-- 1 unfriendly, 2 neutral, 3 friendly), 9 when dead with the extended ladder. A player-controlled
-- one: 0 when each can attack the other, 4 (5 extended) when only it can attack the player, 2 when
-- only the player can attack it, else 3 when PvP-flagged and 4 (5) when not; the extended ladder
-- puts a party or raid member (not the player) on 6, or 7 when PvP-flagged. A unit the client
-- holds no object for reads as an unflagged player, as its no-object branch does.
local reactionTypes = { [1] = 0, [2] = 0, [3] = 1, [4] = 2, [5] = 3, [6] = 3, [7] = 3, [8] = 3 }
function UnitSelectionType(unit, extended)
  local player = extended and 5 or 4
  if not UnitExists(unit) or not UnitIsVisible(unit) then return player end
  if not UnitPlayerControlled(unit) then
    if extended and UnitIsDead(unit) then return 9 end
    return reactionTypes[UnitReaction(unit, "player") or 4] or 2
  end
  if UnitCanAttack(unit, "player") then
    return UnitCanAttack("player", unit) and 0 or player
  end
  if UnitCanAttack("player", unit) then return 2 end
  local pvp = UnitIsPVP(unit) and true or false
  if extended and not UnitIsUnit(unit, "player") and (UnitInParty(unit) or UnitInRaid(unit)) then
    return pvp and 7 or 6
  end
  return pvp and 3 or player
end

-- `GetSpellLink(spell [, bookType])`: the spell's link, with the trailing sub-data field modern
-- links carry, and its id.
function GetSpellLink(spell, book)
  local name, _, _, _, _, _, _, _, _, id = GetSpellInfo(spell, book)
  if not name or not id then return nil end
  return "|cff71d5ff|Hspell:" .. id .. ":0|h[" .. name .. "]|h|r", id
end

-- The reputation list's headers, every one at once.
local function factionHeaders(expand)
  for index = GetNumFactions(), 1, -1 do
    local _, _, _, _, _, _, _, _, isHeader, isCollapsed = GetFactionInfo(index)
    if isHeader and (isCollapsed and expand or not isCollapsed and not expand) then
      if expand then ExpandFactionHeader(index) else CollapseFactionHeader(index) end
    end
  end
end
function ExpandAllFactionHeaders() factionHeaders(true) end
function CollapseAllFactionHeaders() factionHeaders(false) end

-- `GetInstanceInfo()`: the zone and its instance type; no difficulties before TBC.
local instanceCaps = { party = 5, raid = 40, pvp = 40, arena = 5 }
function GetInstanceInfo()
  local inInstance, kind = IsInInstance()
  kind = inInstance and kind or "none"
  local cap = instanceCaps[kind] or 0
  return GetRealZoneText(), kind, 1, "Normal", cap, 0, false, 0, cap
end
