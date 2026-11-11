-- Independent icon picker implementation. Engine enumeration retains archive/loose sources.
IconDataProviderIconType = { Spell = 1, Item = 2 }
IconDataProviderExtraType = { Spellbook = 1, Equipment = 2, None = 3 }
function IconDataProvider_GetAllIconTypes() return { 1, 2 } end

IconDataProviderMixin = {}
function IconDataProviderMixin:Init(extraType, extrasOnly, requested)
  self.extraIconType, self.extraIconsOnly = extraType, extrasOnly
  self.extraIcons, self.baseIcons = {}, { {}, {} }
  local unique = {}
  local function add(icon)
    if type(icon) ~= "string" or icon == "" then return end
    local key = string.upper(icon)
    if not unique[key] then unique[key] = true; table.insert(self.extraIcons, icon) end
  end
  if extraType == IconDataProviderExtraType.Spellbook then
    for tab = 1, GetNumSpellTabs() do
      local _, _, offset, count = GetSpellTabInfo(tab)
      for slot = offset + 1, offset + count do add(GetSpellTexture(slot, "spell")) end
    end
    for tab = 1, GetNumTalentTabs() do
      for index = 1, GetNumTalents(tab) do local _, icon = GetTalentInfo(tab, index); add(icon) end
    end
  elseif extraType == IconDataProviderExtraType.Equipment then
    for slot = 1, 19 do add(GetInventoryItemTexture("player", slot)) end
    for bag = 0, 4 do
      for slot = 1, GetContainerNumSlots(bag) do
        local id = C_Container.GetContainerItemID(bag, slot)
        if id then
          local _, _, _, equip, icon = C_Item.GetItemInfoInstant(id)
          if equip and equip ~= "" then add(icon) end
        end
      end
    end
  end
  table.sort(self.extraIcons, function(a, b) return string.upper(a) < string.upper(b) end)
  if not extrasOnly then
    GetLooseMacroIcons(self.baseIcons[1]); GetMacroIcons(self.baseIcons[1])
    GetLooseMacroItemIcons(self.baseIcons[2]); GetMacroItemIcons(self.baseIcons[2])
  end
  self:SetIconTypes(requested)
end
function IconDataProviderMixin:SetIconTypes(types)
  self.requestedIconTypes = types or IconDataProvider_GetAllIconTypes()
end
function IconDataProviderMixin:ShouldShowExtraIcons()
  local kind = self.extraIconType == IconDataProviderExtraType.Spellbook and 1 or
    (self.extraIconType == IconDataProviderExtraType.Equipment and 2 or nil)
  for _, requested in ipairs(self.requestedIconTypes) do if requested == kind then return true end end
  return false
end
function IconDataProviderMixin:GetNumIcons()
  local count = 1 + (self:ShouldShowExtraIcons() and table.getn(self.extraIcons) or 0)
  for _, kind in ipairs(self.requestedIconTypes) do count = count + table.getn(self.baseIcons[kind] or {}) end
  return count
end
function IconDataProviderMixin:GetIconByIndex(index)
  if type(index) ~= "number" or index < 1 or index ~= math.floor(index) then return nil end
  if index == 1 then return "INTERFACE\\ICONS\\INV_MISC_QUESTIONMARK" end
  index = index - 1
  if self:ShouldShowExtraIcons() then
    if index <= table.getn(self.extraIcons) then return self.extraIcons[index] end
    index = index - table.getn(self.extraIcons)
  end
  for _, kind in ipairs(self.requestedIconTypes) do
    local icons = self.baseIcons[kind] or {}
    if index <= table.getn(icons) then return "INTERFACE\\ICONS\\" .. icons[index] end
    index = index - table.getn(icons)
  end
end
function IconDataProviderMixin:GetIconForSaving(index)
  local icon = self:GetIconByIndex(index)
  if type(icon) ~= "string" then return icon end
  return string.gsub(icon, "^.*[\\/]", "")
end
function IconDataProviderMixin:GetIndexOfIcon(icon)
  for index = 1, self:GetNumIcons() do if self:GetIconByIndex(index) == icon then return index end end
end
function IconDataProviderMixin:Release()
  self.extraIcons, self.baseIcons = {}, { {}, {} }
end
