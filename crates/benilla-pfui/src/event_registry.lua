-- A VM-owned bridge from real frame events to the callback registry.
EventRegistry = CreateFromMixins(CallbackRegistryMixin)
EventRegistry:OnLoad()
EventRegistry:SetUndefinedEventsAllowed(true)
local dispatcher = CreateFrame('Frame')
dispatcher:SetScript('OnEvent', function(_, name, ...)
  EventRegistry:TriggerEvent(name, unpack(arg, 1, arg.n))
end)
function EventRegistry:RegisterFrameEventAndCallback(name, callback, owner, ...)
  local entry = self:RegisterCallback(name, callback, owner, unpack(arg, 1, arg.n))
  dispatcher:RegisterEvent(name)
  return entry
end
function EventRegistry:UnregisterFrameEventAndCallback(name, owner)
  self:UnregisterCallback(name, owner)
  if not self.callbacks[name] or #self.callbacks[name] == 0 then dispatcher:UnregisterEvent(name) end
end
