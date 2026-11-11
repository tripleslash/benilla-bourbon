-- Only the item helpers used by pfUI. Cache misses finish through the host's real item event.
ItemLocation = {}
function ItemLocation:CreateFromBagAndSlot(bag, slot)
  return {bagID=bag, slotIndex=slot}
end
function ItemLocation:CreateFromEquipmentSlot(slot)
  return {equipmentSlotIndex=slot}
end

local Methods = {}
function Methods:GetItemID()
  if self.id then return self.id end
  return C_Item.GetItemID(self.location)
end
function Methods:IsItemEmpty()
  local id = self:GetItemID()
  return id == nil or id == 0
end
function Methods:IsItemDataCached()
  local id = self:GetItemID()
  return id ~= nil and id ~= 0 and C_Item.IsItemDataCachedByID(id) and true or false
end
function Methods:GetItemName()
  return C_Item.GetItemNameByID(self:GetItemID())
end
function Methods:GetItemQuality()
  return C_Item.GetItemQualityByID(self:GetItemID())
end
function Methods:GetItemQualityColorRGB()
  local quality = self:GetItemQuality()
  if quality == nil then return nil end
  return GetItemQualityColor(quality)
end
function Methods:ContinueOnItemLoad(callback)
  assert(type(callback) == 'function', 'item callback must be a function')
  local id = self:GetItemID()
  if not id then return end
  if C_Item.IsItemDataCachedByID(id) then callback(); return end
  local frame = CreateFrame('Frame')
  frame:RegisterEvent('GET_ITEM_INFO_RECEIVED')
  frame:SetScript('OnEvent', function(_, _, received, success)
    if received ~= id then return end
    frame:UnregisterAllEvents()
    frame:SetScript('OnEvent', nil)
    if success then callback() end
  end)
  C_Item.GetItemNameByID(id)
end

Item = {}
function Item:CreateFromItemID(id)
  return Mixin({id=id}, Methods)
end
function Item:CreateFromBagAndSlot(bag, slot)
  return Mixin({location=ItemLocation:CreateFromBagAndSlot(bag, slot)}, Methods)
end
function Item:CreateFromEquipmentSlot(slot)
  return Mixin({location=ItemLocation:CreateFromEquipmentSlot(slot)}, Methods)
end
