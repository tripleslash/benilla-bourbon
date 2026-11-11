-- Independently implemented frame attributes and actions in the engine's ordered call stream.
local native, invoke, actions = ...
local error = error
local attributes = setmetatable({}, { __mode = 'k' })
local handlers = setmetatable({}, { __mode = 'k' })
local firing = setmetatable({}, { __mode = 'k' })
local getScript, setScript, hasScript = native.GetScript, native.SetScript, native.HasScript
local methods = {}
local function store(frame)
  attributes[frame] = attributes[frame] or {}
  return attributes[frame]
end
function methods:SetAttributeNoHandler(name, value)
  if type(name) ~= 'string' then error('attribute name must be a string', 2) end
  store(self)[name:lower()] = value
  if name:lower() == 'unit' and type(value) == 'string' then self:EnableMouse(true) end
  if name:lower():find('type', 1, true) then methods.wireClick(self) end
  return true
end
function methods:SetAttribute(name, value)
  self:SetAttributeNoHandler(name, value)
  local callback = handlers[self]
  if callback and not firing[self] then
    firing[self] = true
    local ok, problem = pcall(invoke, self, callback, name:lower(), value)
    firing[self] = nil
    if not ok then error(problem, 2) end
  end
end
function methods.wireClick(frame)
  -- The method wrapper preserves this dispatch when an addon replaces its OnClick.
  local own = store(frame)
  if not own.clickInstalled then
    local previous = getScript(frame, 'OnClick')
    own.previousClick = previous
    own.clickInstalled = true
    setScript(frame, 'OnClick', function(frame, button, ...)
      if not methods.click(frame, button) and own.previousClick then
        own.previousClick(frame, button, unpack(arg, 1, arg.n))
      end
    end)
  end
end
function methods:ClearAttribute(name)
  local own = store(self)
  name = name:lower()
  local exists = own[name] ~= nil
  own[name] = nil
  return exists
end
function methods:GetAttribute(prefix, name, suffix)
  local own = store(self)
  if name == nil then return own[prefix:lower()] end
  prefix, name, suffix = (prefix or ''):lower(), name:lower(), (suffix or ''):lower()
  for _, key in ipairs({ prefix .. name .. suffix, '*' .. name .. suffix,
    prefix .. name .. '*', '*' .. name .. '*', name }) do
    if own[key] ~= nil then return own[key] end
  end
end
function methods:SetScript(kind, callback)
  if kind:lower() == 'onattributechanged' then
    if callback ~= nil and type(callback) ~= 'function' then error('script must be a function or nil', 2) end
    handlers[self] = callback
  elseif kind:lower() == 'onclick' and attributes[self] and attributes[self].clickInstalled then
    attributes[self].previousClick = callback
  else
    return setScript(self, kind, callback)
  end
end
function methods:GetScript(kind)
  if kind:lower() == 'onattributechanged' then return handlers[self] end
  if kind:lower() == 'onclick' and attributes[self] and attributes[self].clickInstalled then
    return attributes[self].previousClick
  end
  return getScript(self, kind)
end
function methods:HasScript(kind)
  if kind:lower() == 'onattributechanged' then return 1 end
  return hasScript(self, kind)
end
local dropdown
function ClassicAPI_ToggleUnitMenu(unit)
  if not UnitExists(unit) then return end
  if not dropdown then
    dropdown = CreateFrame('Frame', 'BenillaPfuiUnitDropDown', UIParent, 'UIDropDownMenuTemplate')
  end
  local which = UnitIsUnit(unit, 'player') and 'SELF' or UnitIsUnit(unit, 'pet') and 'PET'
    or UnitIsPlayer(unit) and (UnitInParty(unit) and 'PARTY' or 'PLAYER') or 'RAID_TARGET_ICON'
  local name = which == 'RAID_TARGET_ICON' and RAID_TARGET_ICON or UnitName(unit)
  dropdown.unit = unit
  dropdown.name = name
  UIDropDownMenu_Initialize(dropdown, function()
    UnitPopup_ShowMenu(dropdown, which, unit, name)
  end, 'MENU')
  ToggleDropDownMenu(1, nil, dropdown, 'cursor', 0, 0)
end
local buttons = { LeftButton = '1', RightButton = '2', MiddleButton = '3', Button4 = '4', Button5 = '5' }
function methods.click(frame, button)
  local suffix = buttons[button]
  if not suffix then return false end
  local prefix = (IsAltKeyDown() and 'alt-' or '') .. (IsControlKeyDown() and 'ctrl-' or '')
    .. (IsShiftKeyDown() and 'shift-' or '')
  local own = store(frame)
  local function value(name)
    for _, key in ipairs({ prefix .. name .. suffix, name .. suffix, name }) do
      if own[key] ~= nil then return own[key] end
    end
  end
  local verb, unit = value('type'), value('unit')
  if verb == nil then return false end
  if verb == 'target' then
    if unit == 'none' then ClearTarget()
    elseif unit then
      if SpellIsTargeting() then SpellTargetUnit(unit)
      elseif CursorHasItem() then DropItemOnUnit(unit)
      else TargetUnit(unit) end
    end
  elseif verb == 'assist' then
    if unit then AssistUnit(unit) end
  elseif verb == 'focus' then
    if unit == 'none' then ClearFocus()
    elseif unit then FocusUnit(unit) end
  elseif verb == 'stopcasting' then SpellStopCasting()
  elseif verb == 'menu' or verb == 'togglemenu' then
    if unit then ClassicAPI_ToggleUnitMenu(unit) end
  elseif verb == 'spell' then
    actions.spell(value('spell'), unit)
  elseif verb == 'macro' then
    local text = value('macrotext')
    if text ~= nil then actions.macro(text, false, unit)
    else actions.macro(value('macro'), true, unit) end
  elseif verb == 'item' then
    local item = value('item')
    local bag, slot
    if type(item) == 'string' then
      bag, slot = item:match('^%s*(-?%d+)%s+(%d+)%s*$')
    elseif item == nil then bag, slot = value('bag'), value('slot') end
    if bag and slot then UseContainerItem(tonumber(bag), tonumber(slot))
    else actions.item(item, unit) end
  else return false end
  return true
end
function methods.hoveredUnit()
  local frame = GetMouseFocus()
  local unit = frame and attributes[frame] and attributes[frame].unit
  if type(unit) == 'string' then return unit end
end
return methods
