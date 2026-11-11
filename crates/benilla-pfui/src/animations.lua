-- Animation groups, as a ClassicAPI client gives regions: `region:CreateAnimationGroup()` and its
-- animations, played from one frame's OnUpdate. An `Alpha` fades its target; a `Scale` resizes it
-- about its points; a `Translation` or `Path` moves its points by the offset. 1.12 has no draw-time
-- transform, so the target's own size and points carry the motion, and they come back when the
-- animation ends. A `Rotation` keeps its values and turns nothing.
local tables = ...
local getn, tinsert = table.getn, table.insert

local Group, Animation = {}, {}
local GroupMeta = {__index = Group}
local playing = {}
local driver = CreateFrame("Frame")

local smoothing = {
  NONE = function(p) return p end,
  IN = function(p) return p * p end,
  OUT = function(p) return 1 - (1 - p) * (1 - p) end,
  IN_OUT = function(p)
    if p < 0.5 then return 2 * p * p end
    return 1 - 2 * (1 - p) * (1 - p)
  end,
}

local function fire(object, script, a1, a2)
  local handler = object.scripts[script]
  if handler then handler(object, a1, a2) end
end

-- Scripts on groups and animations, as on frames.
local function scripted(class)
  function class:SetScript(script, handler) self.scripts[script] = handler end
  function class:GetScript(script) return self.scripts[script] end
  function class:HasScript(script)
    return script == "OnPlay" or script == "OnFinished" or script == "OnStop"
      or script == "OnLoop" or script == "OnUpdate" or script == "OnPause"
  end
  function class:HookScript(script, handler)
    local before = self.scripts[script]
    if not before then self.scripts[script] = handler; return end
    self.scripts[script] = function(a, b, c) before(a, b, c); handler(a, b, c) end
  end
end
scripted(Group)
scripted(Animation)

-- Each animation's own setters by type.
local kinds = {Alpha = {}, Translation = {}, Scale = {}, Rotation = {}, LineScale = {},
  LineTranslation = {}, Path = {}, Animation = {}}
for kind, methods in pairs(kinds) do
  kinds[kind] = setmetatable(methods, {__index = Animation})
end
local Alpha = kinds.Alpha
function Alpha:SetChange(change) self.change = change; self.fromAlpha, self.toAlpha = nil, nil end
function Alpha:GetChange() return self.change or 0 end
function Alpha:SetFromAlpha(alpha) self.fromAlpha = alpha end
function Alpha:GetFromAlpha() return self.fromAlpha or 0 end
function Alpha:SetToAlpha(alpha) self.toAlpha = alpha end
function Alpha:GetToAlpha() return self.toAlpha or 0 end
function kinds.Translation:SetOffset(x, y) self.offsetX, self.offsetY = x, y end
function kinds.Translation:GetOffset() return self.offsetX or 0, self.offsetY or 0 end
function kinds.Scale:SetScale(x, y) self.scaleX, self.scaleY = x, y end
function kinds.Scale:GetScale() return self.scaleX or 1, self.scaleY or 1 end
function kinds.Scale:SetScaleFrom(x, y) self.fromX, self.fromY = x, y end
function kinds.Scale:SetScaleTo(x, y) self.scaleX, self.scaleY = x, y end
function kinds.Scale:SetOrigin(point, x, y) self.origin = {point, x, y} end
function kinds.Path:SetCurve(curve) self.curve = curve end
function kinds.Path:GetCurve() return self.curve or "NONE" end
function kinds.Path:CreateControlPoint(name, template, order)
  local point = {offsetX = 0, offsetY = 0, order = order or (table.getn(self.points or {}) + 1)}
  function point:SetOffset(x, y) self.offsetX, self.offsetY = x or 0, y or 0 end
  function point:GetOffset() return self.offsetX, self.offsetY end
  function point:SetOrder(order) self.order = order end
  function point:GetOrder() return self.order end
  self.points = self.points or {}
  tinsert(self.points, point)
  return point
end
function kinds.Path:GetControlPoints() return unpack(self.points or {}) end
function kinds.Rotation:SetDegrees(degrees) self.degrees = degrees end
function kinds.Rotation:GetDegrees() return self.degrees or 0 end
function kinds.Rotation:SetRadians(radians) self.degrees = radians * 180 / math.pi end
function kinds.Rotation:GetRadians() return (self.degrees or 0) * math.pi / 180 end
function kinds.Rotation:SetOrigin(point, x, y) self.origin = {point, x, y} end

function Animation:SetDuration(seconds) self.duration = seconds or 0 end
function Animation:GetDuration() return self.duration end
function Animation:SetStartDelay(seconds) self.startDelay = seconds or 0 end
function Animation:GetStartDelay() return self.startDelay end
function Animation:SetEndDelay(seconds) self.endDelay = seconds or 0 end
function Animation:GetEndDelay() return self.endDelay end
function Animation:SetOrder(order) self.order = order or 1 end
function Animation:GetOrder() return self.order end
function Animation:SetSmoothing(kind)
  kind = string.upper(kind or "NONE")
  self.smoothing = smoothing[kind] and kind or "NONE"
end
function Animation:GetSmoothing() return self.smoothing end
function Animation:SetTarget(target) self.target = target end
function Animation:GetTarget() return self.target or self.group.parent end
function Animation:SetChildKey(key) self.target = self.group.parent[key] end
Animation.SetTargetKey = Animation.SetChildKey
function Animation:GetRegionParent() return self.group.parent end
function Animation:GetParent() return self.group end
function Animation:GetName() return self.name end
function Animation:GetObjectType() return self.kind end
function Animation:IsObjectType(kind) return kind == self.kind or kind == "Animation" end
function Animation:IsPlaying() return self.group.playing and self.active == true end
function Animation:GetProgress() return self.progress or 0 end
function Animation:GetElapsed() return self.elapsed or 0 end
function Animation:Play() self.group:Play() end
function Animation:Stop() self.group:Stop() end
function Animation:SetPlaying(on) if on then self:Play() else self:Stop() end end
function Animation:GetSmoothProgress()
  return smoothing[self.smoothing](self.progress or 0)
end

-- An order's animations run together; the next order starts when the longest has ended.
local function orders(group)
  local list, seen = {}, {}
  for _, animation in ipairs(group.animations) do
    if not seen[animation.order] then
      seen[animation.order] = true
      tinsert(list, animation.order)
    end
  end
  table.sort(list)
  return list
end
local function span(animation)
  return animation.startDelay + animation.duration + animation.endDelay
end
local function phaseLength(group, order)
  local longest = 0
  for _, animation in ipairs(group.animations) do
    if animation.order == order then longest = math.max(longest, span(animation)) end
  end
  return longest
end

-- What a moving animation changes on its target, kept to hand back: its size and its points.
local held = setmetatable({}, {__mode = "k"})
local function hold(target)
  if held[target] or not target.GetNumPoints then return held[target] end
  local points = {}
  for i = 1, target:GetNumPoints() do points[i] = {target:GetPoint(i)} end
  held[target] = {width = target:GetWidth(), height = target:GetHeight(), points = points}
  return held[target]
end
local function release(target)
  local base = held[target]
  if not base then return end
  held[target] = nil
  target:SetWidth(base.width)
  target:SetHeight(base.height)
  target:ClearAllPoints()
  for _, p in ipairs(base.points) do target:SetPoint(p[1], p[2], p[3], p[4], p[5]) end
end
local function move(target, dx, dy)
  local base = held[target]
  target:ClearAllPoints()
  for _, p in ipairs(base.points) do
    target:SetPoint(p[1], p[2], p[3], (p[4] or 0) + dx, (p[5] or 0) + dy)
  end
end

-- The point along a path's control points, in their order, from the start at 0, 0.
local function along(points, eased)
  local list = {{offsetX = 0, offsetY = 0}}
  local sorted = {}
  for _, point in ipairs(points or {}) do tinsert(sorted, point) end
  table.sort(sorted, function(a, b) return a.order < b.order end)
  for _, point in ipairs(sorted) do tinsert(list, point) end
  local legs = table.getn(list) - 1
  if legs < 1 then return 0, 0 end
  local at = math.min(eased * legs, legs - 1e-9)
  local leg = math.floor(at) + 1
  local f = at - (leg - 1)
  local a, b = list[leg], list[leg + 1]
  return a.offsetX + (b.offsetX - a.offsetX) * f, a.offsetY + (b.offsetY - a.offsetY) * f
end

-- One animation at its progress; `reverse` plays it backwards (a bouncing loop).
local function apply(animation, progress)
  animation.progress = progress
  local target = animation:GetTarget()
  if not target then return end
  local eased = smoothing[animation.smoothing](progress)
  if animation.kind == "Alpha" and target.SetAlpha then
    local start = animation.startAlpha or 1
    local from = animation.fromAlpha or start
    local to = animation.toAlpha or (start + (animation.change or 0))
    target:SetAlpha(math.max(0, math.min(1, from + (to - from) * eased)))
  elseif animation.kind == "Scale" then
    local base = hold(target)
    if not base then return end
    local fx = (animation.fromX or 1) + ((animation.scaleX or 1) - (animation.fromX or 1)) * eased
    local fy = (animation.fromY or 1) + ((animation.scaleY or 1) - (animation.fromY or 1)) * eased
    target:SetWidth(base.width * fx)
    target:SetHeight(base.height * fy)
  elseif animation.kind == "Translation" then
    if hold(target) then move(target, (animation.offsetX or 0) * eased, (animation.offsetY or 0) * eased) end
  elseif animation.kind == "Path" then
    if hold(target) then move(target, along(animation.points, eased)) end
  end
end

local function startOrder(group)
  group.phaseElapsed = 0
  for _, animation in ipairs(group.animations) do
    animation.active = animation.order == group.orders[group.phase]
    if animation.active then
      animation.elapsed = 0
      local target = animation:GetTarget()
      if animation.startAlpha == nil and target and target.GetAlpha then
        animation.startAlpha = target:GetAlpha()
      end
      fire(animation, "OnPlay")
    end
  end
end

local function finish(group, stopped)
  group.playing, group.paused = false, false
  playing[group] = nil
  for _, animation in ipairs(group.animations) do
    if animation.active then
      animation.active = false
      if stopped then fire(animation, "OnStop") else fire(animation, "OnFinished") end
    end
    -- An Alpha that does not keep its final value hands the target back its alpha; a moving
    -- animation its size and points.
    local target = animation:GetTarget()
    if animation.startAlpha and not group.toFinalAlpha and target and target.SetAlpha then
      target:SetAlpha(animation.startAlpha)
    end
    animation.startAlpha = nil
    if target then release(target) end
  end
  if stopped then fire(group, "OnStop") else fire(group, "OnFinished", false) end
end

local function step(group, elapsed)
  if group.paused then return end
  group.phaseElapsed = group.phaseElapsed + elapsed
  for _, animation in ipairs(group.animations) do
    if animation.active then
      animation.elapsed = group.phaseElapsed
      local run = group.phaseElapsed - animation.startDelay
      local progress = 0
      if animation.duration > 0 then progress = math.max(0, math.min(1, run / animation.duration)) end
      if run >= 0 then
        if group.reverse then apply(animation, 1 - progress) else apply(animation, progress) end
        fire(animation, "OnUpdate", elapsed)
      end
    end
  end
  fire(group, "OnUpdate", elapsed)
  if group.phaseElapsed < phaseLength(group, group.orders[group.phase]) then return end
  for _, animation in ipairs(group.animations) do
    if animation.active then
      animation.active = false
      fire(animation, "OnFinished")
    end
  end
  group.phase = group.phase + 1
  if group.orders[group.phase] then startOrder(group); return end
  if group.looping == "REPEAT" then
    group.phase = 1
    fire(group, "OnLoop", "FORWARD")
    startOrder(group)
  elseif group.looping == "BOUNCE" then
    group.phase = 1
    group.reverse = not group.reverse
    fire(group, "OnLoop", group.reverse and "REVERSE" or "FORWARD")
    startOrder(group)
  else
    finish(group, false)
  end
end

driver:SetScript("OnUpdate", function()
  local elapsed = arg1
  local now = {}
  for group in pairs(playing) do tinsert(now, group) end
  for _, group in ipairs(now) do
    if playing[group] then step(group, elapsed) end
  end
end)

function Group:CreateAnimation(kind, name, template)
  kind = kind or "Animation"
  local methods = kinds[kind] or kinds.Animation
  local animation = setmetatable({
    group = self, kind = kind, name = name, scripts = {}, duration = 0, startDelay = 0,
    endDelay = 0, order = 1, smoothing = "NONE",
  }, {__index = methods})
  tinsert(self.animations, animation)
  return animation
end
function Group:GetAnimations() return unpack(self.animations) end
function Group:Play(reverse)
  if self.playing and not self.paused then return end
  if self.paused then self.paused = false; return end
  self.orders = orders(self)
  if getn(self.orders) == 0 then return end
  self.playing, self.reverse, self.phase = true, reverse and true or false, 1
  playing[self] = true
  fire(self, "OnPlay")
  startOrder(self)
end
function Group:Restart(reverse) self:Stop(); self:Play(reverse) end
function Group:Stop() if self.playing then finish(self, true) end end
function Group:Finish() if self.playing then finish(self, false) end end
function Group:Pause()
  if self.playing then self.paused = true; fire(self, "OnPause") end
end
function Group:IsPlaying() return self.playing and not self.paused end
function Group:IsPaused() return self.paused end
function Group:IsDone() return not self.playing end
function Group:IsPendingFinish() return false end
-- Every group here steps each frame the frame draws; the flag is kept for its getter.
function Group:SetIgnoreFramerateThrottle(ignore) self.ignoreThrottle = ignore and true or false end
function Group:IsIgnoringFramerateThrottle() return self.ignoreThrottle == true end
function Group:IsReverse() return self.reverse end
function Group:SetPlaying(on) if on then self:Play() else self:Stop() end end
function Group:SetLooping(kind) self.looping = string.upper(kind or "NONE") end
function Group:GetLooping() return self.looping end
function Group:GetLoopState() return self.reverse and "REVERSE" or "FORWARD" end
function Group:SetToFinalAlpha(on) self.toFinalAlpha = on and true or false end
function Group:IsSetToFinalAlpha() return self.toFinalAlpha end
function Group:GetDuration()
  local total = 0
  for _, order in ipairs(orders(self)) do total = total + phaseLength(self, order) end
  return total
end
function Group:GetProgress()
  local duration = self:GetDuration()
  if duration <= 0 or not self.playing then return 0 end
  return math.min(1, (self.phaseElapsed or 0) / duration)
end
function Group:GetParent() return self.parent end
function Group:GetName() return self.name end
function Group:GetObjectType() return "AnimationGroup" end
function Group:IsObjectType(kind) return kind == "AnimationGroup" end

-- The regions' side.
local groupsOf = setmetatable({}, {__mode = "k"})
local function create(self, name, template)
  local group = setmetatable({
    parent = self, name = name, scripts = {}, animations = {}, looping = "NONE",
    playing = false, paused = false,
  }, GroupMeta)
  groupsOf[self] = groupsOf[self] or {}
  tinsert(groupsOf[self], group)
  return group
end
local function groups(self) return unpack(groupsOf[self] or {}) end
local function stopAnimating(self)
  for _, group in ipairs(groupsOf[self] or {}) do group:Stop() end
end
for _, methods in ipairs({tables.frame, tables.texture, tables.fontstring}) do
  if methods.CreateAnimationGroup == nil then
    methods.CreateAnimationGroup = create
    methods.GetAnimationGroups = groups
    methods.StopAnimating = stopAnimating
  end
end
