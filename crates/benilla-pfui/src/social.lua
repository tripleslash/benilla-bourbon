-- Extension data uses the host's saved-variable writer and follows VM reloads.
local bridge = ...
local error = error
RegisterForSave('BenillaPfuiDB')
local function database()
  BenillaPfuiDB = BenillaPfuiDB or {}
  return BenillaPfuiDB
end
local function notes()
  local db = database()
  db.friendNotes = db.friendNotes or {}
  local realm, player = GetRealmName(), UnitName('player')
  if not realm or not player then return end
  local key = realm .. ':' .. player
  db.friendNotes[key] = db.friendNotes[key] or {}
  return db.friendNotes[key]
end
local function classToken(localized)
  for token, name in pairs(LOCALIZED_CLASS_NAMES_MALE or {}) do
    if name == localized then return token end
  end
end
local function optional(text) return text ~= '' and text or nil end
local function friend(index)
  if index < 1 or index > GetNumFriends() then return end
  local name, level, class, area, online, status = GetFriendInfo(index)
  if not name then return end
  local saved = notes()
  return { name = name, level = level, className = optional(class),
    classFilename = classToken(class), area = optional(area), connected = online ~= nil and online ~= 0,
    afk = status == (CHAT_FLAG_AFK or '<AFK>'), dnd = status == (CHAT_FLAG_DND or '<DND>'),
    guid = bridge.guid(name), notes = saved and saved[name:lower()],
    mobile = false, referAFriend = false, rafLinkType = 0 }
end
local function friendIndex(name)
  if type(name) ~= 'string' then return end
  for index = 1, GetNumFriends() do
    local found = GetFriendInfo(index)
    if found and found:lower() == name:lower() then return index end
  end
end
C_FriendList = {}
function C_FriendList.GetNumFriends() return GetNumFriends() end
function C_FriendList.GetNumOnlineFriends()
  local count = 0
  for index = 1, GetNumFriends() do
    local _, _, _, _, online = GetFriendInfo(index)
    if online and online ~= 0 then count = count + 1 end
  end
  return count
end
function C_FriendList.GetFriendInfoByIndex(index) return friend(index) end
function C_FriendList.GetFriendInfo(name)
  local index = friendIndex(name)
  if index then return friend(index) end
end
function C_FriendList.SetFriendNotesByIndex(index, text)
  local info, saved = friend(index), notes()
  if not info or not saved then return end
  if text ~= nil and type(text) ~= 'string' then error('notes must be a string or nil', 2) end
  saved[info.name:lower()] = text ~= '' and text or nil
  bridge.updated()
end
function C_FriendList.SetFriendNotes(name, text)
  local index = friendIndex(name)
  if index then C_FriendList.SetFriendNotesByIndex(index, text) end
end
function C_FriendList.GetNumWhoResults() return GetNumWhoResults() end
function C_FriendList.GetWhoInfo(index)
  local name, guild, level, race, class, area = GetWhoInfo(index)
  if not name then return end
  return { fullName = name, fullGuildName = guild, level = level,
    raceStr = optional(race), classStr = optional(class), filename = classToken(class), area = area }
end
local sentAt
function C_FriendList.IsWhoQueryPending()
  return sentAt ~= nil and GetTime() - sentAt < 5
end
function C_FriendList.SendWhoQueryByName(name)
  if type(name) ~= 'string' or not UnitExists('player') or C_FriendList.IsWhoQueryPending() then
    return false
  end
  name = name:match('^([^:]+)')
  if not name or #name > 12 or not name:match('^%a+$') then return false end
  SetWhoToUI(1)
  SendWho('n-"' .. name .. '"')
  sentAt = GetTime()
  return true
end

-- Cache records are scoped to the realm. The default remains disabled, as documented.
local function cache()
  local db = database()
  db.playerCache = db.playerCache or {}
  local realm = GetRealmName()
  if not realm then return end
  db.playerCache[realm] = db.playerCache[realm] or { entries = {}, names = {} }
  return db.playerCache[realm]
end
C_PlayerCache = {}
function C_PlayerCache.IsEnabled() return database().playerCacheEnabled == true end
function C_PlayerCache.SetEnabled(value)
  database().playerCacheEnabled = value ~= nil and value ~= false and value ~= 0
end
local function raceInfo(token)
  if not C_CreatureInfo then return end
  for id = 1, 255 do
    local info = C_CreatureInfo.GetRaceInfo(id)
    if info and (info.clientFileString:lower() == token:lower() or info.raceName == token) then
      return info
    end
  end
end
local function remember(guid, name, class, race, sex, authoritative)
  if not C_PlayerCache.IsEnabled() or type(guid) ~= 'string' or type(name) ~= 'string' then return false end
  local digits = guid:match('^0[xX](%x+)$')
  if not digits or (#digits ~= 8 and #digits ~= 16) or digits:match('^0+$') then return false end
  guid = '0x' .. string.rep('0', 16 - #digits) .. digits:upper()
  name = name:gsub('[^%a]', ''):sub(1, 12)
  if name == '' then return false end
  local saved = cache()
  if not saved then return false end
  local previous = saved.names[name]
  if previous and previous ~= guid then saved.entries[previous] = nil end
  local record = saved.entries[guid] or {}
  if record.name and record.name ~= name then saved.names[record.name] = nil end
  record.name = name
  if type(class) == 'string' then
    local token = class:upper()
    if (LOCALIZED_CLASS_NAMES_MALE or {})[token] then record.class = token end
  end
  if type(race) == 'string' then
    local info = raceInfo(race)
    if info then record.race = info.clientFileString end
  end
  -- Explicit sightings preserve a previous sex on zero; a wire answer assigns male too.
  if sex == 1 or (authoritative and sex == 0) then record.sex = sex end
  saved.entries[guid], saved.names[name] = record, guid
  return true
end
function C_PlayerCache.RememberPlayer(guid, name, class, race, sex)
  return remember(guid, name, class, race, sex, false)
end
-- Every name-query reply of this VM's session, independent of the opt-in saved cache.
local seen = {}
local function canonical(guid)
  if type(guid) ~= 'string' then return end
  local digits = guid:match('^0[xX](%x+)$')
  if not digits or (#digits ~= 8 and #digits ~= 16) or digits:match('^0+$') then return end
  return '0x' .. string.rep('0', 16 - #digits) .. digits:upper()
end
function bridge.nameReply(guid, name, class, raceID, sex)
  local race = C_CreatureInfo and C_CreatureInfo.GetRaceInfo(raceID)
  local key = canonical(guid)
  if key and type(name) == 'string' and name ~= '' then
    seen[key] = { name = name, class = type(class) == 'string' and class:upper() or nil,
      race = race and race.clientFileString, sex = sex }
  end
  return remember(guid, name, class, race and race.clientFileString, sex, true)
end
local function record(guid)
  local key = canonical(guid)
  if not key then return end
  if seen[key] then return seen[key] end
  if C_PlayerCache.IsEnabled() then
    local saved = cache()
    return saved and saved.entries[key]
  end
end
-- A visible unit answers first, then the friends list, then name-query replies.
function UnitNameFromGUID(guid)
  local key = canonical(guid)
  if not key then return end
  local name = UnitExists(key) and UnitName(key)
  if name then return name, '' end
  for index = 1, GetNumFriends() do
    local info = friend(index)
    if info and info.guid == key then return info.name, '' end
  end
  local known = record(key)
  if known then return known.name, '' end
end
function GetPlayerInfoByGUID(guid)
  local key = canonical(guid)
  if not key then return end
  if UnitExists(key) and UnitIsPlayer(key) then
    local class, classToken = UnitClass(key)
    local race, raceToken = UnitRace(key)
    return class, classToken, race, raceToken, UnitSex(key), UnitName(key), ''
  end
  local known = record(key)
  if not known then return end
  local race = known.race and raceInfo(known.race)
  return (LOCALIZED_CLASS_NAMES_MALE or {})[known.class], known.class,
    race and race.raceName, known.race, (known.sex or 0) + 2, known.name, ''
end
function C_PlayerCache.GetPlayerInfoByName(name)
  if not C_PlayerCache.IsEnabled() then return end
  local saved = cache()
  local guid = saved and saved.names[name]
  local record = guid and saved.entries[guid]
  if not record then return end
  local race = record.race and raceInfo(record.race)
  return (LOCALIZED_CLASS_NAMES_MALE or {})[record.class], record.class,
    race and race.raceName, record.race, (record.sex or 0) + 2, record.name, '', guid
end
