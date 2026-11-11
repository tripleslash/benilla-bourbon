-- The functions the ClassicAPI DLL registers from its engine half, which its own Lua addon
-- (!!!ClassicAPI, loaded from AddOns) and pfUI build on. Each one stands alone: it never calls a
-- Lua global that addon redefines, or the two would call each other.
local bridge = ...
local error, type, xpcall, unpack, getglobal = error, type, xpcall, unpack, getglobal
local strfind, strsub, strlen, gsub = string.find, string.sub, string.len, string.gsub

local function pack(...) return arg end

-- An error goes to the error handler and the caller goes on (`securecall`).
local function report(message)
  local handler = geterrorhandler()
  if handler then handler(message) end
  return message
end

local function protected(func, args)
  local results = pack(xpcall(function() return func(unpack(args, 1, args.n)) end, report))
  if results[1] then return unpack(results, 2, results.n) end
end

function securecall(func, ...)
  if type(func) == "string" or type(func) == "number" then func = getglobal(func) end
  if type(func) ~= "function" then error("Usage: securecall(func, ...)", 2) end
  return protected(func, arg)
end

function securecallfunction(func, ...)
  if type(func) ~= "function" then error("Usage: securecallfunction(func, ...)", 2) end
  return protected(func, arg)
end

function securecallmethod(tbl, name, ...)
  if type(tbl) ~= "table" or type(name) ~= "string" then
    error("Usage: securecallmethod(table, method, ...)", 2)
  end
  local args = { n = arg.n + 1, tbl }
  for i = 1, arg.n do args[i + 1] = arg[i] end
  return protected(tbl[name], args)
end

function secureexecuterange(tbl, func, ...)
  if type(tbl) ~= "table" or type(func) ~= "function" then
    error("Usage: secureexecuterange(table, func, ...)", 2)
  end
  for key, value in pairs(tbl) do
    local args = {key, value, n = arg.n + 2}
    for i = 1, arg.n do args[i + 2] = arg[i] end
    protected(func, args)
  end
end

-- Addon code is never secure here, and this is a release client.
function issecure() return nil end
function IsPublicBuild() return true end

-- strsplit splits on any character of `sep`; `pieces` caps the count, the last piece keeping the
-- rest of the string.
local function split(name, sep, str, pieces)
  if type(str) == "number" then str = tostring(str) end
  if type(sep) ~= "string" or type(str) ~= "string" then
    error("Usage: " .. name .. "(separator, string [, pieces])", 3)
  end
  pieces = tonumber(pieces) or 0
  local class = "[" .. gsub(sep, "(%W)", "%%%1") .. "]"
  local out, n, start = {}, 0, 1
  while pieces <= 0 or n < pieces - 1 do
    local at = sep ~= "" and strfind(str, class, start)
    if not at then break end
    n = n + 1
    out[n] = strsub(str, start, at - 1)
    start = at + 1
  end
  n = n + 1
  out[n] = strsub(str, start, strlen(str))
  out.n = n
  return out
end

function strsplit(sep, str, pieces)
  local out = split("strsplit", sep, str, pieces)
  return unpack(out, 1, out.n)
end
string.split = strsplit

function strsplittable(sep, str, pieces)
  local out = split("strsplittable", sep, str, pieces)
  out.n = nil
  return out
end

-- The time the current frame advanced: what this frame's OnUpdate scripts receive as elapsed.
local tick = 0
local clock = CreateFrame("Frame")
clock:SetScript("OnUpdate", function() tick = arg1 end)
function GetTickTime() return tick end

function GetMouseFoci()
  local focus = GetMouseFocus()
  return focus and {focus} or {}
end

-- The class names of ChrClasses.dbc, by token; `isFemale` is accepted and ignored, as the 1.12
-- table has one name per class.
function FillLocalizedClassList(tbl, isFemale)
  for token, name in pairs(bridge.class_names()) do tbl[token] = name end
  return tbl
end

-- Frames have no source location here, so there is nothing to exclude.
function AddSourceLocationExclude(fileName) end

C_AddOns = C_AddOns or {}
function C_AddOns.IsAddOnLoaded(addon)
  local loaded = IsAddOnLoaded(addon) and true or false
  return loaded, loaded
end

-- Retail's UI colors as ColorMixin objects, built through the caller's CreateColor.
C_UIColor = {}
function C_UIColor.GetColors()
  local rows = {}
  for i, row in ipairs(bridge.ui_colors) do
    local hex = row[2]
    local function channel(at) return tonumber(strsub(hex, at, at + 1), 16) / 255 end
    rows[i] = {baseTag = row[1], color = CreateColor(channel(3), channel(5), channel(7), channel(1))}
  end
  return rows
end

-- Item qualities as globals, Enum.ItemQuality's values.
LE_ITEM_QUALITY_POOR, LE_ITEM_QUALITY_COMMON, LE_ITEM_QUALITY_UNCOMMON = 0, 1, 2
LE_ITEM_QUALITY_RARE, LE_ITEM_QUALITY_EPIC, LE_ITEM_QUALITY_LEGENDARY = 3, 4, 5
LE_ITEM_QUALITY_ARTIFACT, LE_ITEM_QUALITY_HEIRLOOM, LE_ITEM_QUALITY_WOWTOKEN = 6, 7, 8

-- The game window's size in device pixels.
function GetPhysicalScreenSize()
  local pixels = bridge.display.windowPixels and bridge.display.windowPixels()
  local _, _, width, height = strfind(pixels or GetCVar("gxResolution") or "", "^(%d+)x(%d+)")
  if width then return tonumber(width), tonumber(height) end
  return 1024, 768
end

C_CVar = {}
function C_CVar.GetCVar(name) return GetCVar(name) end
function C_CVar.SetCVar(name, value) return SetCVar(name, value) end
function C_CVar.GetCVarDefault(name) return GetCVarDefault(name) end
function C_CVar.DoesCVarExist(name) return GetCVar(name) ~= nil end
-- `nil` for an unknown name; true for "true" or a non-zero number.
function C_CVar.GetCVarBool(name)
  local value = GetCVar(name)
  if value == nil then return nil end
  if string.lower(value) == "true" then return true end
  local number = tonumber(value)
  return number ~= nil and number ~= 0
end

-- Modified clicks: an action active while its modifier keys are held. A mouse button in a
-- binding is not checked, as no click records it here.
local modifiedClicks = {
  SELFCAST = "ALT", FOCUSCAST = "NONE", AUTOLOOTTOGGLE = "SHIFT", MAILAUTOLOOTTOGGLE = "SHIFT",
  STICKYCAMERA = "CTRL", CHATLINK = "SHIFT-BUTTON1", DRESSUP = "CTRL-BUTTON1",
  SOCKETITEM = "SHIFT-BUTTON2", SPLITSTACK = "SHIFT", PICKUPACTION = "SHIFT", PICKUPITEM = "SHIFT",
  COMPAREITEMS = "SHIFT", OPENALLBAGS = "SHIFT", QUESTWATCHTOGGLE = "SHIFT",
  TOKENWATCHTOGGLE = "SHIFT", SHOWITEMFLYOUT = "ALT", SHOWMULTICASTFLYOUT = "ALT",
  TRADESEARCHADD = "SHIFT",
}
local modifiedClickOrder = {}
for action in pairs(modifiedClicks) do tinsert(modifiedClickOrder, action) end
table.sort(modifiedClickOrder)

local function modifierDown(name)
  if name == "SHIFT" or name == "LSHIFT" or name == "RSHIFT" then return IsShiftKeyDown() end
  if name == "CTRL" or name == "LCTRL" or name == "RCTRL" then return IsControlKeyDown() end
  if name == "ALT" or name == "LALT" or name == "RALT" then return IsAltKeyDown() end
end

local function bindingDown(binding)
  if binding == "NONE" then return false end
  local any = false
  for part in string.gfind(binding, "[^-]+") do
    if not strfind(part, "^BUTTON%d$") then
      if not modifierDown(part) then return false end
      any = true
    end
  end
  return any
end

function IsModifiedClick(action)
  if action == nil then
    return (IsShiftKeyDown() or IsControlKeyDown() or IsAltKeyDown()) and 1 or nil
  end
  action = string.upper(action)
  local binding = modifiedClicks[action] or (modifierDown(action) ~= nil and action)
  return binding and bindingDown(binding) and 1 or nil
end
function GetModifiedClick(action) return modifiedClicks[string.upper(action or "")] end
function SetModifiedClick(action, binding)
  action = string.upper(action or "")
  if not modifiedClicks[action] then error("SetModifiedClick: unknown action " .. tostring(action), 2) end
  modifiedClicks[action] = string.upper(binding)
end
function GetNumModifiedClickActions() return table.getn(modifiedClickOrder) end
function GetModifiedClickAction(index) return modifiedClickOrder[index] end

-- Atlases: a name for a rectangle of a texture file. `SetAtlas` points a texture at one; a name
-- that is not registered leaves the texture as it was.
local atlases, atlasOf = {}, setmetatable({}, {__mode = "k"})
C_Texture = {}
function C_Texture.RegisterAtlas(name, file, width, height, left, right, top, bottom, tilesH, tilesV)
  atlases[string.lower(name)] = {
    elementName = name, filename = file, width = width, height = height,
    leftTexCoord = left, rightTexCoord = right, topTexCoord = top, bottomTexCoord = bottom,
    tilesHorizontally = tilesH and true or false, tilesVertically = tilesV and true or false,
  }
  -- `|A…|a` text escapes draw from the host's copy, which takes only a complete atlas.
  pcall(bridge.atlas_registered, name, file, width, height, left, right, top, bottom)
  return true
end
local function atlas(name) return type(name) == "string" and atlases[string.lower(name)] end
function C_Texture.GetAtlasInfo(name)
  local info = atlas(name)
  if not info then return nil end
  local copy = {}
  for key, value in pairs(info) do copy[key] = value end
  return copy
end
function C_Texture.GetAtlasExists(name) return atlas(name) and true or false end

local textures = bridge.textures
local setTexture = textures.SetTexture
-- Pointing a texture at a file or a color drops its atlas.
textures.SetTexture = function(self, a1, a2, a3, a4)
  atlasOf[self] = nil
  return setTexture(self, a1, a2, a3, a4)
end
textures.SetAtlas = function(self, name, useAtlasSize)
  local info = atlas(name)
  if not info then return end
  setTexture(self, info.filename)
  self:SetTexCoord(info.leftTexCoord, info.rightTexCoord, info.topTexCoord, info.bottomTexCoord)
  if useAtlasSize then
    self:SetWidth(info.width)
    self:SetHeight(info.height)
  end
  atlasOf[self] = info.elementName
end
textures.GetAtlas = function(self) return atlasOf[self] end
textures.SetColorTexture = function(self, r, g, b, a)
  atlasOf[self] = nil
  return setTexture(self, r, g, b, a)
end

-- Whether the player knows a spell, its passives included: the spellbook's answer.
local isSpellKnown = IsSpellKnown
function IsPlayerSpell(spellID)
  return isSpellKnown ~= nil and isSpellKnown(spellID) and true or false
end

