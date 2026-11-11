-- Nampower's documented addon ABI, implemented by Benilla rather than a loaded DLL.
local bridge = ...
local error = error
local originalGet, originalSet, originalDefault = GetCVar, SetCVar, GetCVarDefault
local originalStopCasting, originalStopTargeting = SpellStopCasting, SpellStopTargeting
local defaults = {
  np_enablespellstartevents = "0",
  np_enablespellgoevents = "0",
  np_enableauracastevents = "0",
  np_enableautoattackevents = "0",
  np_enablespellhealevents = "0",
  np_queuecasttimespells = "1",
  np_queueinstantspells = "1",
  np_queuechannelingspells = "1",
  np_queuetargetingspells = "1",
  np_queueonswingspells = "0",
  np_queuespellsoncooldown = "1",
  np_spellqueuewindowms = "500",
  np_channelqueuewindowms = "1500",
  np_targetingqueuewindowms = "500",
  np_onswingbuffercooldownms = "500",
  np_cooldownqueuewindowms = "250",
  np_minbuffertimems = "55",
  np_nongcdbuffertimems = "100",
  np_replacematchingnongcdcategory = "0",
  np_quickcasttargetingspells = "0",
  np_quickcastondoublecast = "0",
  np_retryserverrejectedspells = "1",
  -- The rest of Nampower 4.6.1's settings, which NampowerSettings' menu reads and writes: kept as
  -- the player sets them, with no behaviour behind them here.
  np_interruptchannelsoutsidequeuewindow = "0",
  np_optimizebufferusingpackettimings = "0",
  np_preventrightclicktargetchange = "0",
  np_preventrightclickpvpattack = "0",
  np_doublecasttoendchannelearly = "0",
  np_spamprotectionenabled = "1",
  np_enableuniteventspet = "1",
  np_enableuniteventsparty = "1",
  np_enableuniteventsraid = "1",
  np_enableuniteventsmouseover = "1",
  np_enableuniteventsguid = "1",
  np_enableuniteventsguidfiltering = "0",
  np_enablespellenergizeevents = "0",
  np_preventmountingwhenbuffcapped = "1",
  np_maxbufferincreasems = "30",
  np_channellatencyreductionpercentage = "75",
  np_nameplatedistance = "20",
}
local function settings()
  BenillaPfuiDB = BenillaPfuiDB or {}
  BenillaPfuiDB.nampower = BenillaPfuiDB.nampower or {}
  return BenillaPfuiDB.nampower
end
function GetCVar(name)
  local key = type(name) == "string" and string.lower(name)
  if defaults[key] then return settings()[key] or defaults[key] end
  return originalGet(name)
end
function GetCVarDefault(name)
  local key = type(name) == "string" and string.lower(name)
  if defaults[key] then return defaults[key] end
  return originalDefault(name)
end
function SetCVar(name, value, scriptCvar)
  local key = type(name) == "string" and string.lower(name)
  if not defaults[key] then return originalSet(name, value, scriptCvar) end
  local n = tonumber(value)
  if not n or n ~= n or n < 0 or n == math.huge then error("Invalid Nampower CVar value") end
  if string.sub(defaults[key], 1) == "0" or defaults[key] == "1" then
    n = n ~= 0 and 1 or 0
  else
    n = math.min(math.floor(n), 10000)
  end
  local text = tostring(n)
  settings()[key] = text ~= defaults[key] and text or nil
end
function GetNampowerVersion() return 3, 0, 0 end
function CastSpellByName(name, target)
  bridge.namedCast(name, target, "automatic")
end
function CastSpell(spell, book, unit)
  bridge.bookCast(spell, book, unit, "automatic")
end
function QueueSpellByName(name)
  bridge.namedCast(name, nil, "force")
end
function CastSpellByNameNoQueue(name, target)
  bridge.namedCast(name, target, "never")
end
function CastSpellNoQueue(spell, book, unit)
  bridge.bookCast(spell, book, unit, "never")
end
function QueueScript(body, priority)
  bridge.queueScript(body, priority)
end
function SpellStopCasting()
  bridge.cancelQueue(false)
  return originalStopCasting()
end
function SpellStopTargeting()
  bridge.cancelQueue(true)
  return originalStopTargeting()
end
BenillaNampowerCompatibility = {
  implementation = "Benilla", apiVersion = "3.0.0", emulated = true,
  spellEvents = true, auraEvents = true, healEvents = true, spellQueue = true,
  fullNampower = false,
}
function GetCastInfo() return bridge.castInfo() end
function GetWowTimeMs() return math.floor(GetTime() * 1000) end
function CancelPlayerAuraSpellId(spellId) C_Spell.CancelSpellByID(spellId) end
function GetPlayerAuraDuration(slot)
  local result = bridge.auraDuration(slot)
  if result then return unpack(result) end
end
