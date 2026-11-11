-- GUID strings retain all 64 bits in Lua's double-number VM. Storage follows the local writer.
local bridge = ...
local api, ignored, active = {}, {}, nil
C_EquipmentSet = api
local function storage()
  if not UnitExists('player') then return end
  local realm, name = GetRealmName(), UnitName('player')
  if not realm or not name then return end
  BenillaPfuiDB = BenillaPfuiDB or {}
  BenillaPfuiDB.equipment = BenillaPfuiDB.equipment or {}
  local key = realm .. ':' .. name
  local data = BenillaPfuiDB.equipment[key]
  if not data then
    data = { nextID = 1, sets = {} }
    BenillaPfuiDB.equipment[key] = data
  end
  return data
end
local function get(id)
  local data = storage()
  if not data then return end
  for index, set in ipairs(data.sets) do if set.id == id then return set, index, data end end
end
local function changed() bridge.event('EQUIPMENT_SETS_CHANGED') end
local function capture(set)
  local worn = {}
  for _, item in ipairs(bridge.snapshot()) do
    if item.bag == -100 then worn[item.slot] = item end
  end
  set.items, set.details = {}, {}
  for slot = 1, 19 do
    if ignored[slot] then set.items[slot] = false
    elseif worn[slot] then
      set.items[slot] = worn[slot].guid
      set.details[slot] = { id = worn[slot].itemID, name = worn[slot].name }
    end
  end
end
local function locations()
  local result = {}
  for _, item in ipairs(bridge.snapshot()) do result[item.guid] = item end
  return result
end
function api.CanUseEquipmentSets() return true end
function api.GetNumEquipmentSets()
  local data = storage()
  return data and #data.sets or 0
end
function api.GetEquipmentSetIDs()
  local result, data = {}, storage()
  if data then for _, set in ipairs(data.sets) do result[#result+1] = set.id end end
  return result
end
function api.GetEquipmentSetID(name)
  local data = storage()
  if data then for _, set in ipairs(data.sets) do if set.name == name then return set.id end end end
end
function api.GetEquipmentSetInfo(id)
  local set = get(id)
  if not set then return end
  local found = locations()
  local total, worn, carried, missing, skips = 0, 0, 0, 0, 0
  for slot = 1, 19 do
    local guid = set.items[slot]
    if guid == false then skips = skips + 1
    elseif guid then
      total = total + 1
      local item = found[guid]
      if not item then missing = missing + 1
      elseif item.bag == -100 and item.slot == slot then worn = worn + 1
      else carried = carried + 1 end
    end
  end
  return set.name, set.icon, set.id, total > 0 and carried == 0, total, worn, carried, missing, skips
end
function api.GetIgnoredSlots(id)
  local set = get(id)
  if not set then return end
  local result = {}
  for slot = 1, 19 do if set.items[slot] == false then result[#result+1] = slot end end
  return result
end
function api.GetItemIDs(id)
  local set = get(id)
  if not set then return end
  local result, found = {}, locations()
  for slot = 1, 19 do
    local item = found[set.items[slot]]
    if item then result[slot] = item.itemID end
  end
  return result
end
function api.GetItemLocations(id)
  local set = get(id)
  if not set then return end
  local result, found = {}, locations()
  for slot = 1, 19 do
    local guid = set.items[slot]
    if guid == false then result[slot] = 1
    elseif guid then result[slot] = found[guid] and found[guid].location or -1 end
  end
  return result
end
function api.CreateEquipmentSet(name, icon)
  if type(name) ~= 'string' or name == '' or api.GetEquipmentSetID(name) then return end
  local data = storage()
  if not data then return end
  local set = { id = data.nextID, name = name, icon = icon or 'INV_Misc_QuestionMark' }
  data.nextID = data.nextID + 1
  capture(set)
  data.sets[#data.sets+1] = set
  changed()
  return set.id
end
function api.SaveEquipmentSet(id, icon)
  local set = get(id)
  if not set then return end
  capture(set)
  if icon ~= nil then set.icon = icon end
  changed()
end
function api.ModifyEquipmentSet(id, name)
  local set = get(id)
  if not set or type(name) ~= 'string' or name == '' then return end
  local existing = api.GetEquipmentSetID(name)
  if existing and existing ~= id then return end
  set.name = name
  changed()
end
function api.DeleteEquipmentSet(id)
  local set, index, data = get(id)
  if not set then return end
  table.remove(data.sets, index)
  changed()
end
local function valid(slot) return type(slot) == 'number' and slot >= 1 and slot <= 19 and slot == math.floor(slot) end
function api.IgnoreSlotForSave(slot)
  if not valid(slot) then return end
  ignored[slot] = true
  changed()
end
function api.UnignoreSlotForSave(slot)
  if not valid(slot) then return end
  ignored[slot] = nil
  changed()
end
function api.IsSlotIgnoredForSave(slot) return valid(slot) and ignored[slot] == true end
function api.ClearIgnoredSlotsForSave() ignored = {}; changed() end
function api.EquipmentSetContainsLockedItems(id)
  local set = get(id)
  if not set then return false end
  local found = locations()
  for slot = 1, 19 do
    local item = found[set.items[slot]]
    if item and item.locked then return true end
  end
  return false
end
local function finish(job, success)
  if active ~= job then return end
  active = nil
  bridge.event('EQUIPMENT_SWAP_FINISHED', success and 1 or 0, job.id)
end
local function step(job)
  if active ~= job then return end
  if not UnitExists('player') then finish(job, false); return end
  local found = locations()
  if job.pending then
    local item = found[job.pending.guid]
    if item and item.bag == -100 and item.slot == job.pending.slot then job.pending = nil
    elseif GetTime() >= job.deadline then finish(job, false); return
    else C_Timer.After(.05, function() step(job) end); return end
  end
  for slot = 1, 19 do
    local guid = job.items[slot]
    local item = found[guid]
    if item and (item.bag == -100 or item.bag >= 0 and item.bag <= 4) and
      (item.bag ~= -100 or item.slot ~= slot) then
      if item.locked then finish(job, false); return end
      job.pending = { guid = guid, slot = slot }
      job.deadline = GetTime() + 10
      if not bridge.swap(guid, slot) then finish(job, false); return end
      C_Timer.After(.05, function() step(job) end)
      return
    end
  end
  finish(job, true)
end
function api.UseEquipmentSet(id)
  local set = get(id)
  if not set or active then bridge.event('EQUIPMENT_SWAP_FINISHED', 0, id); return false end
  local job = { id = id, items = CopyTable(set.items) }
  active = job
  bridge.event('EQUIPMENT_SWAP_PENDING', id)
  -- Resolve each GUID again after the prior server transaction; swapped pairs need no second send.
  step(job)
  return true
end

local function localized(key, fallback, value)
  local pattern = _G[key]
  if type(pattern) == 'string' then
    local ok, text = pcall(string.format, pattern, value)
    if ok then return text end
  end
  return string.format(fallback, value)
end
function bridge.tooltips:SetEquipmentSet(name)
  local id = api.GetEquipmentSetID(name)
  local set = id and get(id)
  if not set then return end
  local _, _, _, _, total, worn, carried, missing, skips = api.GetEquipmentSetInfo(id)
  self:ClearLines()
  self:AddLine(set.name, 1, 1, 1)
  self:AddLine(localized('ITEMS_VARIABLE_QUANTITY', '%d items', total), 1, 1, 1)
  if worn > 0 then self:AddLine(localized('ITEMS_EQUIPPED', '%d equipped', worn), 0, 1, 0) end
  if carried > 0 then self:AddLine(localized('ITEMS_IN_INVENTORY', '%d in inventory', carried), 1, 1, 1) end
  if skips > 0 then self:AddLine(localized('ITEM_SLOTS_IGNORED', '%d slots ignored', skips), .5, .5, .5) end
  if missing > 0 then
    local found = locations()
    for slot = 1, 19 do
      local guid = set.items[slot]
      if guid and not found[guid] then
        local detail = set.details and set.details[slot]
        local label = detail and (C_Item.GetItemNameByID(detail.id) or detail.name)
        -- Old saves can contain only a GUID; retain that real identity rather than inventing a name.
        self:AddLine(localized('ITEM_MISSING', 'Missing: %s', label or guid), 1, 0, 0)
      end
    end
  end
end

-- Inventory types each paperdoll slot accepts. Two-handers stay in the main hand, and the
-- literal main/off-hand types stay in their own hand. 1.12 relics equip in the ranged slot.
local accepts = {
  [1] = { INVTYPE_HEAD = true }, [2] = { INVTYPE_NECK = true }, [3] = { INVTYPE_SHOULDER = true },
  [4] = { INVTYPE_BODY = true }, [5] = { INVTYPE_CHEST = true, INVTYPE_ROBE = true },
  [6] = { INVTYPE_WAIST = true }, [7] = { INVTYPE_LEGS = true }, [8] = { INVTYPE_FEET = true },
  [9] = { INVTYPE_WRIST = true }, [10] = { INVTYPE_HAND = true },
  [11] = { INVTYPE_FINGER = true }, [12] = { INVTYPE_FINGER = true },
  [13] = { INVTYPE_TRINKET = true }, [14] = { INVTYPE_TRINKET = true },
  [15] = { INVTYPE_CLOAK = true },
  [16] = { INVTYPE_WEAPON = true, INVTYPE_2HWEAPON = true, INVTYPE_WEAPONMAINHAND = true },
  [17] = { INVTYPE_WEAPON = true, INVTYPE_SHIELD = true, INVTYPE_WEAPONOFFHAND = true,
    INVTYPE_HOLDABLE = true },
  [18] = { INVTYPE_RANGED = true, INVTYPE_THROWN = true, INVTYPE_RANGEDRIGHT = true,
    INVTYPE_RELIC = true },
  [19] = { INVTYPE_TABARD = true },
}
function GetInventoryItemsForSlot(slot, result)
  assert(type(result) == 'table', 'GetInventoryItemsForSlot requires a result table')
  local types = accepts[slot]
  if not types then return result end
  for _, item in ipairs(bridge.snapshot()) do
    if item.bag == -100 or item.bag >= 0 and item.bag <= 4 then
      local _, _, _, token = C_Item.GetItemInfoInstant(item.itemID)
      if token and types[token] then
        local link
        if item.bag == -100 then link = GetInventoryItemLink('player', item.slot)
        else link = GetContainerItemLink(item.bag, item.slot) end
        if link then result[item.location] = link end
      end
    end
  end
  return result
end
