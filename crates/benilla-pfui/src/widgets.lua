local native = ...
-- pfUI replaces the global error function with a chat printer after this installs.
local error = error
local methods = {}
local filters = setmetatable({}, { __mode = 'k' })
methods.filters = filters
local register = native.RegisterEvent
local unregister = native.UnregisterEvent
local unregisterAll = native.UnregisterAllEvents

function methods:RegisterUnitEvent(kind, ...)
  local wasRegistered = self:IsEventRegistered(kind)
  local events = filters[self]
  if wasRegistered and (not events or not events[kind]) then return true end
  local units = {}
  for i = 1, select('#', ...) do
    local unit = select(i, ...)
    if type(unit) == 'string' then table.insert(units, unit) end
  end
  register(self, kind)
  events = events or {}; filters[self] = events
  events[kind] = #units > 0 and units or nil
  return true
end

function methods:UnregisterEvent(kind)
  unregister(self, kind)
  if filters[self] then filters[self][kind] = nil end
end

function methods:UnregisterAllEvents()
  unregisterAll(self)
  filters[self] = nil
end

function methods:SetSize(width, height)
  self:SetWidth(width)
  self:SetHeight(height)
end

function methods:GetSize()
  return self:GetWidth(), self:GetHeight()
end

function methods:SetShown(shown)
  if shown then self:Show() else self:Hide() end
end

function methods:SetFormattedText(formatString, ...)
  self:SetText(string.format(formatString, ...))
end

function methods:HookScript(kind, callback)
  if type(callback) ~= 'function' then error('HookScript requires a function', 2) end
  local previous = self:GetScript(kind)
  self:SetScript(kind, function(...)
    if previous then previous(...) end
    callback(...)
  end)
end

local forbidden = {}
for _, name in ipairs({ 'getfenv', 'getmetatable', 'hooksecurefunc', 'ipairs',
  'issecurevalue', 'issecurevariable', 'next', 'pairs', 'pcall', 'pcallwithenv',
  'rawget', 'rawset', 'scrub', 'securecall', 'securecallfunction', 'securecallmethod',
  'secureexecuterange',
  'select', 'setfenv', 'setmetatable', 'type', 'unpack', 'wipe', 'xpcall' }) do
  forbidden[name] = true
end

function hooksecurefunc(target, name, callback)
  if type(target) == 'string' then
    callback, name, target = name, target, _G
    if forbidden[name] then error('hooksecurefunc: function is unhookable', 2) end
  end
  if type(target) ~= 'table' or type(name) ~= 'string' or type(callback) ~= 'function' then
    error('hooksecurefunc requires a function name and callback', 2)
  end
  local original = target[name]
  if type(original) ~= 'function' then error('hooksecurefunc: target is not a function', 2) end
  target[name] = function(...)
    -- The packed table preserves nils in both the argument and return lists.
    local arguments = { n = select('#', ...), ... }
    local function capture(...)
      return { n = select('#', ...), ... }
    end
    local results = capture(original(...))
    callback(unpack(arguments, 1, arguments.n))
    return unpack(results, 1, results.n)
  end
end

return methods
