-- Re-anchor an owned tooltip without clearing it. The owner is the last SetOwner's frame; the
-- six owner-relative modes take the offset, and the mode SetOwner recorded is left as it was.
local methods = ...
local owners = setmetatable({}, { __mode = 'k' })
local setOwner = methods.SetOwner
function methods:SetOwner(owner, ...)
  owners[self] = owner
  return setOwner(self, owner, ...)
end
-- The owner while it still owns the tooltip: a hide drops it in the engine.
function methods:GetOwner()
  local owner = owners[self]
  if owner and self:IsOwned(owner) then return owner end
end
local points = {
  ANCHOR_RIGHT = { 'BOTTOMLEFT', 'TOPRIGHT' }, ANCHOR_LEFT = { 'BOTTOMRIGHT', 'TOPLEFT' },
  ANCHOR_TOPRIGHT = { 'BOTTOMRIGHT', 'TOPRIGHT' }, ANCHOR_TOPLEFT = { 'BOTTOMLEFT', 'TOPLEFT' },
  ANCHOR_BOTTOMRIGHT = { 'TOPLEFT', 'BOTTOMRIGHT' }, ANCHOR_BOTTOMLEFT = { 'TOPRIGHT', 'BOTTOMLEFT' },
}
function methods:SetAnchorType(anchor, x, y)
  local owner = owners[self]
  local pair = type(anchor) == 'string' and points[anchor:upper()]
  if not owner or not pair then return end
  self:ClearAllPoints()
  self:SetPoint(pair[1], owner, pair[2], x or 0, y or 0)
end
