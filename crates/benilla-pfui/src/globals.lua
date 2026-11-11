-- ClassicAPI globals pfUI calls directly. Each reads the engine's own state or stock queries.
local bridge = ...

-- pfUI's load gate requires ClassicAPI 1.15.10 (X*10000 + Y*100 + Z) and otherwise raises a
-- dialog at every login that says pfUI is disabled and takes the keyboard focus. This is the
-- version whose pfUI-used surface the extension implements, not the DLL's complete API: the
-- second table says so for an addon that needs to tell the two apart.
CLASSIC_API_VERSION = 11510
BenillaClassicAPICompatibility = {
  implementation = 'Benilla', apiVersion = '1.15.10', emulated = true, fullClassicAPI = false,
}

function IsModifierKeyDown()
  if IsShiftKeyDown() or IsControlKeyDown() or IsAltKeyDown() then return 1 end
end

-- `chars` is a literal character set, never a pattern.
function strtrim(text, chars)
  assert(type(text) == 'string' or type(text) == 'number', 'strtrim requires a string')
  text = tostring(text)
  chars = chars or ' \t\n\v\f\r'
  local first, last = 1, #text
  while first <= last and chars:find(text:sub(first, first), 1, true) do first = first + 1 end
  while last >= first and chars:find(text:sub(last, last), 1, true) do last = last - 1 end
  return text:sub(first, last)
end

-- The ID companions return the item the stock link builder already names for the same slot.
local function linked(link)
  if type(link) ~= 'string' then return end
  local id = link:match('|Hitem:(%d+)')
  return id and tonumber(id)
end
for name, source in pairs({
  GetLootRollItemID = 'GetLootRollItemLink',
  GetLootSlotItemID = 'GetLootSlotLink',
  GetMerchantItemID = 'GetMerchantItemLink',
  GetQuestItemID = 'GetQuestItemLink',
  GetQuestLogItemID = 'GetQuestLogItemLink',
  GetTradeSkillItemID = 'GetTradeSkillItemLink',
  GetTradeSkillReagentItemID = 'GetTradeSkillReagentItemLink',
  GetCraftReagentItemID = 'GetCraftReagentItemLink',
}) do
  _G[name] = function(...) return linked(_G[source](...)) end
end

-- The spell a macro shows: its #showtooltip/#show argument, else its first cast. Only a name
-- the player's books resolve is returned; conditionals and items resolve to nothing.
local function shown(body)
  local first
  for line in body:gmatch('[^\r\n]+') do
    line = strtrim(line)
    local directive, argument = line:match('^#(%a+)%s*(.*)$')
    if directive then
      directive = directive:lower()
      if (directive == 'showtooltip' or directive == 'show') and argument ~= '' then
        return argument
      end
    elseif not first then
      local command, rest = line:match('^/(%a+)%s+(.+)$')
      command = command and command:lower()
      if command == 'cast' then
        first = rest
      elseif command == 'castsequence' then
        rest = rest:gsub('^reset=%S+%s*', '')
        first = rest:match('^([^,]+)')
      else
        first = line:match('CastSpellByName%(%s*"([^"]+)"') or
          line:match("CastSpellByName%(%s*'([^']+)'")
      end
    end
  end
  return first
end
function GetMacroSpell(slot)
  local _, _, body = GetMacroInfo(slot)
  if type(body) ~= 'string' or not GetSpellInfo then return end
  local wanted = shown(body)
  if not wanted then return end
  local name, rank, _, _, _, _, _, _, _, id = GetSpellInfo(strtrim(wanted))
  if name then return name, rank, id end
end

-- Tracking spells are the player's book entries whose DBC effect applies a tracking aura:
-- SPELL_AURA_TRACK_CREATURES (44), TRACK_RESOURCES (45) or TRACK_STEALTHED (151).
local TRACKING = { [44] = true, [45] = true, [151] = true }
local function trackers()
  local result = {}
  if not GetSpellInfo then return result end
  local slot = 1
  while GetSpellName(slot, 'spell') do
    local name, _, _, _, _, _, _, _, _, id = GetSpellInfo(slot, 'spell')
    local effects = id and C_Spell.GetSpellEffectInfo(id)
    if effects then
      for _, effect in ipairs(effects) do
        if effect.effect ~= 0 and TRACKING[effect.auraName] then
          result[#result + 1] = { slot = slot, name = name, id = id }
          break
        end
      end
    end
    slot = slot + 1
  end
  return result
end
function GetNumTrackingTypes() return #trackers() end
function GetTrackingInfo(index)
  local tracker = trackers()[index]
  if not tracker then return end
  local texture = GetSpellTexture(tracker.slot, 'spell')
  -- One tracker is in effect at a time, and the stock query names it by its icon.
  return tracker.name, texture, texture ~= nil and texture == GetTrackingTexture(), 'spell', tracker.id
end
function SetTracking(index)
  local tracker = trackers()[index]
  if tracker then CastSpell(tracker.slot, 'spell') end
end

-- Packed item locations, as C_EquipmentSet.GetItemLocations publishes them.
ITEM_INVENTORY_LOCATION_PLAYER = 1048576 -- 0x00100000
ITEM_INVENTORY_LOCATION_BAGS = 2097152 -- 0x00200000
ITEM_INVENTORY_LOCATION_BANK = 4194304 -- 0x00400000
function EquipmentManager_UnpackLocation(location)
  if type(location) ~= 'number' or location < 0 then return false, false, false, 0 end
  local flags = math.floor(location / 1048576)
  local player, bags, bank = flags % 2 == 1, math.floor(flags / 2) % 2 == 1, math.floor(flags / 4) % 2 == 1
  local rest = location % 1048576
  if not bags then return player, bank, false, rest % 256 end
  local bag = math.floor(rest / 256) % 256
  if bank then bag = bag + 4 end
  return player, bank, true, rest % 256, bag
end
function EquipmentManager_GetLocationData(location)
  local player, bank, bags, slot, bag = EquipmentManager_UnpackLocation(location)
  return { isPlayer = player, isBank = bank, isBags = bags, slot = slot, bag = bag }
end

-- `gxResolution` as the reference means it, the size the game draws at: the host stores only the
-- windowed size there, and pfUI computes one device pixel from this string.
local stockGetCVar = GetCVar
function GetCVar(name)
  if type(name) == 'string' and string.lower(name) == 'gxresolution' then
    local pixels = bridge.windowPixels()
    if pixels then return pixels end
  end
  -- An unknown name answers nothing, as under the ClassicAPI DLL, where 1.12 raises.
  local results = {pcall(stockGetCVar, name)}
  if results[1] then return results[2] end
  if type(name) == 'string' and string.find(tostring(results[2]), "Couldn't find CVar named", 1, true) then
    return
  end
  error(results[2], 2)
end

-- Addon memory: what the heap grew by while each addon loaded (gcinfo's kilobytes), an addon
-- loaded inside another's load counted with that one. A ClassicAPI client counts every
-- allocation an addon's code makes; most of what an addon keeps, it builds as it loads.
local loadingAt, loadDepth, addonMemory = 0, 0, {}
function bridge.memoryBefore()
  if loadDepth == 0 then loadingAt = gcinfo() end
  loadDepth = loadDepth + 1
end
function bridge.afterAddon(name)
  loadDepth = math.max(0, loadDepth - 1)
  if loadDepth == 0 and name then
    addonMemory[name] = (addonMemory[name] or 0) + math.max(0, gcinfo() - loadingAt)
  end
end
function UpdateAddOnMemoryUsage() end
function GetAddOnMemoryUsage(addon)
  if type(addon) == 'number' then addon = GetAddOnInfo(addon) end
  return addonMemory[addon] or 0
end

-- Run before each addon loads: the stock colour tables exist by then, and FrameXML assigns them
-- as plain tables after this extension installs.
function bridge.beforeAddon()
  bridge.memoryBefore()
  for name, value in pairs(_G) do
    if type(name) == 'string' and type(value) == 'table' and name:find('_FONT_COLOR$') and
      type(value.r) == 'number' and value.GetRGB == nil then
      if value.a == nil then value.a = 1 end
      Mixin(value, ColorMixin)
    end
  end
  if WHITE_FONT_COLOR == nil then WHITE_FONT_COLOR = CreateColor(1, 1, 1) end
end

-- Inline texture markup as later clients spell it. The client draws it once the extension is
-- enabled (benilla-app's pfui/inline_texture.rs); a stock 1.12 client draws the bytes.
function CreateSimpleTextureMarkup(file, width, height)
  return string.format('|T%s:%d:%d|t', file, height or width, width)
end

-- A money amount with coin icons, cropped from the stock money sheet (64x16, three 16 px
-- coins). A locale may define the three formats first; `fontHeight` 0 sizes a coin to its font.
local SHEET = 'Interface\\MoneyFrame\\UI-MoneyIcons'
local function coin(left) return '%d|T' .. SHEET .. ':%d:%d:2:0:64:16:' .. left .. ':' .. (left + 16) .. ':0:16|t' end
GOLD_AMOUNT_TEXTURE = GOLD_AMOUNT_TEXTURE or coin(0)
SILVER_AMOUNT_TEXTURE = SILVER_AMOUNT_TEXTURE or coin(16)
COPPER_AMOUNT_TEXTURE = COPPER_AMOUNT_TEXTURE or coin(32)
function GetCoinTextureString(amount, fontHeight)
  amount = math.max(0, math.floor((tonumber(amount) or 0) + 0.5))
  local size = math.max(0, math.floor(tonumber(fontHeight) or 0))
  local gold, silver, copper = math.floor(amount / 10000), math.floor(amount / 100) % 100, amount % 100
  local parts = {}
  if gold > 0 then parts[#parts + 1] = string.format(GOLD_AMOUNT_TEXTURE, gold, size, size) end
  if silver > 0 then parts[#parts + 1] = string.format(SILVER_AMOUNT_TEXTURE, silver, size, size) end
  if copper > 0 or #parts == 0 then parts[#parts + 1] = string.format(COPPER_AMOUNT_TEXTURE, copper, size, size) end
  return table.concat(parts, ' ')
end

-- The client's game menu carries two rungs of its own that pfUI has never heard of. When pfUI's
-- game-menu skin has run (it creates `GameMenuButtonPFUI`), they take the same skin, once.
local menu = CreateFrame('Frame')
menu:RegisterEvent('PLAYER_ENTERING_WORLD')
menu:SetScript('OnEvent', function()
  menu:UnregisterEvent('PLAYER_ENTERING_WORLD')
  if not (pfUI and pfUI.api and pfUI.api.SkinButton and GameMenuButtonPFUI) then return end
  for _, name in ipairs({ 'BenillaGameMenuButtonEditMode', 'BenillaGameMenuButtonSupport' }) do
    if _G[name] then pfUI.api.SkinButton(_G[name]) end
  end
  -- The client stacks its rungs flush, where the stock menu leaves one unit between two. pfUI's
  -- backdrops reach one unit past each button, so flush neighbours cover each other's border
  -- and a highlighted rung loses an edge. Put the stock gap back.
  local opened = 0
  for _, rung in ipairs({ GameMenuFrame:GetChildren() }) do
    if rung:IsShown() and rung:GetObjectType() == 'Button' and rung:GetNumPoints() == 1 then
      local point, to, toPoint, x, y = rung:GetPoint(1)
      if point == 'TOP' and toPoint == 'BOTTOM' and y == 0 and to and to ~= GameMenuFrame then
        rung:SetPoint(point, to, toPoint, x, -1)
        opened = opened + 1
      end
    end
  end
  GameMenuFrame:SetHeight(GameMenuFrame:GetHeight() + opened)
  -- The client shrinks its own menu at every show. A skinned menu belongs to pfUI's layout,
  -- which draws at UIParent's scale as on the original client, where its one-pixel borders
  -- land on whole pixels.
  local shown = GameMenuFrame:GetScript('OnShow')
  GameMenuFrame:SetScript('OnShow', function()
    if shown then shown() end
    this:SetScale(1)
  end)
end)
