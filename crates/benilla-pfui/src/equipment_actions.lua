-- Equipment-set buttons persist beside their character's sets. Server slots remain stock.
local bridge = ...
local api = C_EquipmentSet
local original = {}
for _,name in ipairs({'GetActionInfo','GetActionText','GetActionTexture','HasAction',
  'IsUsableAction','GetActionCooldown','GetActionCount','ActionHasRange','IsActionInRange',
  'IsCurrentAction','IsAutoRepeatAction','IsAttackAction','IsConsumableAction','IsEquippedAction',
  'GetCursorInfo','PickupAction','PlaceAction','UseAction'}) do original[name]=_G[name] end

local function buttons()
  if not UnitExists('player') then return end
  local realm,name=GetRealmName(),UnitName('player')
  if not realm or not name then return end
  local equipment = BenillaPfuiDB and BenillaPfuiDB.equipment
  local data = equipment and equipment[realm..':'..name]
  if not data then return end
  data.buttons = data.buttons or {}
  return data.buttons
end
local function valid(slot)
  return type(slot)=='number' and slot>=1 and slot<=120 and slot==math.floor(slot)
end
local function button(slot)
  slot=tonumber(slot)
  if not valid(slot) then return end
  local bindings=buttons()
  local id=bindings and bindings[slot]
  if not id then return end
  local name=api.GetEquipmentSetInfo(id)
  -- A later server login can replace a slot from another client. Its real action takes priority.
  if not name or bridge.hasServerAction(slot) then bindings[slot]=nil; return end
  return id,name
end
local function icon(id)
  local _,path=api.GetEquipmentSetInfo(id)
  if not path then return end
  if not string.find(path,'[\\/]') then path='Interface\\Icons\\'..path end
  return path
end
local function changed(slot) bridge.event('ACTIONBAR_SLOT_CHANGED',slot) end
function api.PickupEquipmentSet(id)
  if not api.GetEquipmentSetInfo(id) then return end
  bridge.setCursor(id,icon(id))
end
function GetCursorInfo()
  local id=bridge.heldSet()
  if id then
    local name=api.GetEquipmentSetInfo(id)
    return 'equipmentset',name
  end
  return original.GetCursorInfo()
end
function GetActionInfo(slot)
  local id,name=button(slot)
  if id then return 'equipmentset',name end
  return original.GetActionInfo(slot)
end
function GetActionText(slot)
  local id,name=button(slot)
  if id then return name end
  return original.GetActionText(slot)
end
function GetActionTexture(slot)
  local id=button(slot)
  if id then return icon(id) end
  return original.GetActionTexture(slot)
end
function HasAction(slot)
  if button(slot) then return 1 end
  return original.HasAction(slot)
end
function IsUsableAction(slot)
  local id=button(slot)
  if id then
    if api.EquipmentSetContainsLockedItems(id) then return nil,nil end
    return 1,nil
  end
  return original.IsUsableAction(slot)
end
function GetActionCooldown(slot)
  if button(slot) then return 0,0,1 end
  return original.GetActionCooldown(slot)
end
function GetActionCount(slot)
  if button(slot) then return 0 end
  return original.GetActionCount(slot)
end
for _,name in ipairs({'ActionHasRange','IsActionInRange','IsCurrentAction','IsAutoRepeatAction',
  'IsAttackAction','IsConsumableAction','IsEquippedAction'}) do
  local query,stock=name,original[name]
  _G[query]=function(slot,...)
    if button(slot) then return nil end
    return stock(slot,unpack(arg,1,arg.n))
  end
end
function PlaceAction(slot)
  slot=tonumber(slot)
  if not valid(slot) then return false end
  local held=bridge.heldSet()
  local displaced=button(slot)
  if held then
    local bindings=buttons()
    if not bindings or not api.GetEquipmentSetInfo(held) then return false end
    -- Remove a real server action through its existing clear queue, retaining it on the cursor.
    bridge.displaceServerAction(slot)
    bindings[slot]=held
    if displaced then api.PickupEquipmentSet(displaced) end
    changed(slot)
    return true
  elseif displaced and bridge.hasCursor() then
    -- Let the stock filters reject passive spells and non-placeable items without losing the set.
    -- The stock verb answers nothing, as in 1.12; the slot says whether it took the payload.
    original.PlaceAction(slot)
    local placed=bridge.hasServerAction(slot)
    if placed then
      buttons()[slot]=nil
      api.PickupEquipmentSet(displaced)
      changed(slot)
    end
    return placed
  end
  return original.PlaceAction(slot)
end
function PickupAction(slot)
  slot=tonumber(slot)
  if not valid(slot) then return false end
  if bridge.hasCursor() then return PlaceAction(slot) end
  local id=button(slot)
  if id then
    buttons()[slot]=nil
    api.PickupEquipmentSet(id)
    changed(slot)
    return true
  end
  return original.PickupAction(slot)
end
function UseAction(slot,checkCursor,onSelf)
  if bridge.hasCursor() and checkCursor and checkCursor~=0 then return PlaceAction(slot) end
  local id=button(slot)
  if id then bridge.event('WEAR_EQUIPMENT_SET',id); return end
  return original.UseAction(slot,checkCursor,onSelf)
end
local stockTooltip=bridge.tooltips.SetAction
function bridge.tooltips:SetAction(slot)
  local id,name=button(slot)
  if id then self:SetEquipmentSet(name); return end
  return stockTooltip(self,slot)
end
local events=CreateFrame('Frame')
events:RegisterEvent('EQUIPMENT_SETS_CHANGED')
events:RegisterEvent('WEAR_EQUIPMENT_SET')
events:SetScript('OnEvent',function(self,event,id)
  if event=='WEAR_EQUIPMENT_SET' then
    if api.EquipmentSetContainsLockedItems(id) or C_Spell and
      (C_Spell.UnitCastingInfo('player') or C_Spell.ChannelInfo()) then
      bridge.event('UI_ERROR_MESSAGE',ERR_CLIENT_LOCKED_OUT or "You can't do that right now.")
    else api.UseEquipmentSet(id) end
    return
  end
  local bindings=buttons()
  if bindings then
    for slot,setID in pairs(bindings) do
      if not api.GetEquipmentSetInfo(setID) then bindings[slot]=nil end
      changed(slot)
    end
  end
  local held=bridge.heldSet()
  if held then
    if api.GetEquipmentSetInfo(held) then api.PickupEquipmentSet(held)
    else ClearCursor() end
  end
end)
