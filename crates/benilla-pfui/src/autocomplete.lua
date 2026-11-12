-- ClassicAPI's name completion (`C_AutoComplete`): the players this client knows, from the group,
-- the guild roster, the friends list and this session's whispers, matched against an edit box's
-- text as ClassicAPI 1.20's API reference specifies. Nothing is drawn; an addon lists the names.
C_AutoComplete = C_AutoComplete or {}
Enum = Enum or {}
Enum.AutoCompleteEntryFlag = { InGroup = 1, InGuild = 2, Friend = 4, Bnet = 8, InteractedWith = 16,
  Online = 32, InAOI = 64, AccountCharacter = 128, RecentPlayer = 256 }
Enum.AutoCompletePriority = { Other = 0, Interacted = 1, InGroup = 2, Guild = 3, Friend = 4,
  AccountCharacter = 5, AccountCharacterSameRealm = 6 }
local Flag, Priority = Enum.AutoCompleteEntryFlag, Enum.AutoCompletePriority

for _, name in ipairs({ "autoCompleteResortNamesOnRecency", "autoCompleteWhenEditingFromCenter",
    "autoCompleteUseContext" }) do
  local known, value = pcall(GetCVar, name)
  if not known or value == nil then RegisterCVar(name, "1") end
end
local function setting(name) return (tonumber(GetCVar(name) or 1) or 1) ~= 0 end

local function hasBit(flags, bit)
  return math.floor(flags / bit) - math.floor(flags / bit / 2) * 2 == 1
end
local function anyBit(flags, mask)
  local bit = 1
  while bit <= mask do
    if hasBit(mask, bit) and hasBit(flags, bit) then return true end
    bit = bit * 2
  end
  return false
end

-- Whispered players, most recent first, for this session.
local whispered = {}
local whisperFrame = CreateFrame("Frame")
whisperFrame:RegisterEvent("CHAT_MSG_WHISPER")
whisperFrame:RegisterEvent("CHAT_MSG_WHISPER_INFORM")
whisperFrame:SetScript("OnEvent", function()
  local who = arg2
  if type(who) ~= "string" or who == "" then return end
  local key = string.lower(who)
  for i = table.getn(whispered), 1, -1 do
    if string.lower(whispered[i]) == key then table.remove(whispered, i) end
  end
  table.insert(whispered, 1, who)
end)

-- Every known name once, with the bits of each list it is on; the player is never one.
local function known()
  local entries, order = {}, {}
  local me = UnitName("player")
  local function add(name, flag, online)
    if type(name) ~= "string" or name == "" or name == me then return end
    local key = string.lower(name)
    local entry = entries[key]
    if not entry then
      entry = { name = name, flags = 0 }
      entries[key] = entry
      table.insert(order, entry)
    end
    if not hasBit(entry.flags, flag) then entry.flags = entry.flags + flag end
    if online and not hasBit(entry.flags, Flag.Online) then entry.flags = entry.flags + Flag.Online end
  end
  local raid = GetNumRaidMembers and GetNumRaidMembers() or 0
  if raid > 0 then
    for i = 1, raid do
      local name, _, _, _, _, _, _, online = GetRaidRosterInfo(i)
      add(name, Flag.InGroup, online)
    end
  else
    for i = 1, GetNumPartyMembers() do
      local unit = "party" .. i
      add(UnitName(unit), Flag.InGroup, UnitIsConnected(unit))
    end
  end
  if IsInGuild() then
    for i = 1, GetNumGuildMembers(true) do
      local name, _, _, _, _, _, _, _, online = GetGuildRosterInfo(i)
      add(name, Flag.InGuild, online)
    end
  end
  for i = 1, GetNumFriends() do
    local name, _, _, _, connected = GetFriendInfo(i)
    add(name, Flag.Friend, connected)
  end
  for _, name in ipairs(whispered) do
    local entry = entries[string.lower(name)]
    add(name, Flag.InteractedWith, not entry)
  end
  return entries, order
end

local function priority(flags)
  if hasBit(flags, Flag.Friend) then return Priority.Friend end
  if hasBit(flags, Flag.InGuild) then return Priority.Guild end
  if hasBit(flags, Flag.InGroup) then return Priority.InGroup end
  if hasBit(flags, Flag.InteractedWith) then return Priority.Interacted end
  return Priority.Other
end

local function selected(entry, include, exclude)
  if not setting("autoCompleteUseContext") then return true end
  if include and include ~= 0 and not anyBit(entry.flags, include) then return false end
  if exclude and exclude ~= 0 and anyBit(entry.flags, exclude) then return false end
  return true
end

-- The text's characters, so a cursor position counts UTF-8 characters, not bytes.
local function characters(text)
  local list = {}
  for c in (string.gmatch or string.gfind)(text, "[%z\1-\127\194-\244][\128-\191]*") do table.insert(list, c) end
  return list
end

function C_AutoComplete.GetAutoCompleteResults(name, numResults, cursorPosition, allowFullMatch, includeFlags, excludeFlags)
  if type(name) ~= "string" then error("Usage: C_AutoComplete.GetAutoCompleteResults(name, numResults, cursorPosition, allowFullMatch, includeFlags, excludeFlags)", 2) end
  local chars = characters(name)
  local cursor = tonumber(cursorPosition) or 0
  if cursor < 1 or cursor > table.getn(chars) then cursor = table.getn(chars) end
  local before, after = table.concat(chars, "", 1, cursor), table.concat(chars, "", cursor + 1)
  if not setting("autoCompleteWhenEditingFromCenter") then before, after = name, "" end
  before, after = string.lower(before), string.lower(after)
  local whole = string.lower(name)
  local entries, order = known()
  local matches = {}
  for _, entry in ipairs(order) do
    local lower = string.lower(entry.name)
    local ok = string.sub(lower, 1, string.len(before)) == before
    if ok and after ~= "" then ok = string.find(lower, after, string.len(before) + 1, true) ~= nil end
    if ok and not allowFullMatch and lower == whole then ok = false end
    if ok and selected(entry, includeFlags, excludeFlags) then table.insert(matches, entry) end
  end
  local rank = {}
  if setting("autoCompleteResortNamesOnRecency") then
    for i, who in ipairs(whispered) do rank[string.lower(who)] = i end
  end
  table.sort(matches, function(a, b)
    local ra, rb = rank[string.lower(a.name)], rank[string.lower(b.name)]
    if ra and rb then return ra < rb end
    if ra or rb then return ra ~= nil end
    return string.lower(a.name) < string.lower(b.name)
  end)
  local limit = tonumber(numResults) or 0
  local results = {}
  for i, entry in ipairs(matches) do
    if limit > 0 and i > limit then break end
    results[i] = { name = entry.name, priority = priority(entry.flags), bnetID = 0 }
  end
  return results
end

function C_AutoComplete.IsRecognizedName(name, includeFlags, excludeFlags)
  if type(name) ~= "string" then return false end
  local entries = known()
  local entry = entries[string.lower(name)]
  return entry ~= nil and selected(entry, includeFlags, excludeFlags)
end

function C_AutoComplete.GetAutoCompletePresenceID() return nil end
function C_AutoComplete.GetAutoCompleteRealms() return {} end
