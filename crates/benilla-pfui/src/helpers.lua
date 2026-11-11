-- General helpers referenced directly by pfUI, independently implemented here.
function Round(value)
  local magnitude = math.floor(math.abs(value) + 0.5)
  return value < 0 and -magnitude or magnitude
end
function Clamp(value, minimum, maximum)
  return math.min(maximum, math.max(minimum, value))
end
function CopyTable(source, shallow)
  local seen = {}
  local function copy(value)
    if type(value) ~= 'table' then return value end
    if seen[value] then return seen[value] end
    local result = {}; seen[value] = result
    for key, child in pairs(value) do
      result[key] = shallow and child or copy(child)
    end
    return result
  end
  return copy(source)
end

local Pool = {}
function Pool:Acquire()
  local object = table.remove(self.free)
  local created = object == nil
  if created then
    object = self.create(self)
    assert(object ~= nil, 'pool constructor returned nil')
    if self.reset then self.reset(self, object, true) end
  end
  self.active[object] = true
  self.count = self.count + 1
  return object, created
end
function Pool:Release(object)
  if not self.active[object] then return false end
  self.active[object] = nil
  self.count = self.count - 1
  if self.reset then self.reset(self, object) end
  table.insert(self.free, object)
  return true
end
function Pool:ReleaseAll()
  local snapshot = {}
  for object in pairs(self.active) do table.insert(snapshot, object) end
  for _, object in ipairs(snapshot) do self:Release(object) end
end
function Pool:EnumerateActive() return pairs(self.active) end
function Pool:GetNumActive() return self.count end

function CreateObjectPool(create, reset)
  assert(type(create) == 'function', 'pool constructor must be a function')
  assert(reset == nil or type(reset) == 'function', 'pool reset must be a function')
  return Mixin({create=create, reset=reset, free={}, active={}, count=0}, Pool)
end
function CreateTexturePool(parent, layer, template, subLevel, reset)
  return CreateObjectPool(function()
    return parent:CreateTexture(nil, layer, template, subLevel)
  end, reset or function(_, texture)
    texture:Hide()
    texture:ClearAllPoints()
  end)
end
