-- ClassicAPI's bag sort (`C_Container.SortBags`, `SortBankBags` and their three settings), as its
-- API reference specifies it: partial stacks combine first, then the items take their order from
-- the first slot (Hearthstone; weapons and armor by the slot they go into; consumables; reagents;
-- trade goods; quest items; everything else, best quality first) and junk fills from the last slot
-- backward. A specialty bag keeps only what it accepts. An item still loading or locked stays put.
--
-- The host sends each queued move once the server confirmed the one before, and a queued move
-- needs its source filled now, so the sort works in rounds: it plans from the bags as they are,
-- queues the moves that touch distinct slots, and plans again once the host is idle.
local engine = ...

local HEARTHSTONE = 6948
local ROUND_LIMIT, STUCK_LIMIT, TIME_LIMIT = 400, 4, 120

for _, name in ipairs({ "sortBagsRightToLeft", "backpackAutosortDisabled", "bankAutosortDisabled" }) do
  local known, value = pcall(GetCVar, name)
  if not known or value == nil then RegisterCVar(name, "0") end
end
local function flag(name) return GetCVar(name) == "1" end
local function setFlag(name, on) SetCVar(name, on and "1" or "0") end
function C_Container.GetSortBagsRightToLeft() return flag("sortBagsRightToLeft") end
function C_Container.SetSortBagsRightToLeft(on) setFlag("sortBagsRightToLeft", on) end
function C_Container.GetBackpackAutosortDisabled() return flag("backpackAutosortDisabled") end
function C_Container.SetBackpackAutosortDisabled(on) setFlag("backpackAutosortDisabled", on) end
function C_Container.GetBankAutosortDisabled() return flag("bankAutosortDisabled") end
function C_Container.SetBankAutosortDisabled(on) setFlag("bankAutosortDisabled", on) end

-- Weapons first (one-hand, two-hand, ranged), then shields and off-hands, head to feet, then neck,
-- rings and trinkets.
local SLOT_ORDER = {
  INVTYPE_WEAPON = 1, INVTYPE_WEAPONMAINHAND = 1, INVTYPE_WEAPONOFFHAND = 1, INVTYPE_2HWEAPON = 2,
  INVTYPE_RANGED = 3, INVTYPE_RANGEDRIGHT = 3, INVTYPE_THROWN = 3, INVTYPE_RELIC = 4,
  INVTYPE_SHIELD = 10, INVTYPE_HOLDABLE = 11, INVTYPE_HEAD = 12, INVTYPE_SHOULDER = 13,
  INVTYPE_CLOAK = 14, INVTYPE_CHEST = 15, INVTYPE_ROBE = 15, INVTYPE_BODY = 16, INVTYPE_TABARD = 17,
  INVTYPE_WRIST = 18, INVTYPE_HAND = 19, INVTYPE_WAIST = 20, INVTYPE_LEGS = 21, INVTYPE_FEET = 22,
  INVTYPE_NECK = 23, INVTYPE_FINGER = 24, INVTYPE_TRINKET = 25,
}
-- Item class ids: consumable 0, weapon 2, armor 4, reagent 5, trade goods 7, quest 12.
local CLASS_GROUP = { [2] = 2, [4] = 2, [0] = 3, [5] = 4, [7] = 5, [12] = 6 }

local function itemID(link)
  local _, _, id = string.find(link or "", "item:(%d+)")
  return tonumber(id)
end

-- The bag's family bits: 0 for the backpack, the bank and a plain bag.
local function bagFamily(bag)
  if bag <= 0 then return 0 end
  local id = itemID(GetInventoryItemLink("player", ContainerIDToInventoryID(bag)))
  return id and C_Item.GetItemFamily(id) or 0
end

local function accepts(family, item)
  if family == 0 then return true end
  local bits, mask = item.family or 0, family
  local bit = 1
  while bit <= mask do
    if math.mod(math.floor(mask / bit), 2) == 1 and math.mod(math.floor(bits / bit), 2) == 1 then
      return true
    end
    bit = bit * 2
  end
  return false
end

local function read(bags)
  local slots, items = {}, {}
  for _, bag in ipairs(bags) do
    local family = bagFamily(bag)
    for slot = 1, GetContainerNumSlots(bag) do
      local entry = { bag = bag, slot = slot, family = family }
      local id = C_Container.GetContainerItemID(bag, slot)
      if id then
        local _, count, locked = GetContainerItemInfo(bag, slot)
        -- The link tells apart two items of one id (a random suffix); the id stands in without.
        local link = GetContainerItemLink(bag, slot) or ("item:" .. id)
        local name, _, quality, _, _, _, _, maxStack, equipLoc, _, _, class, subclass =
          C_Item.GetItemInfo(id)
        local item = { link = link, id = id, count = count or 1 }
        if name and not locked then
          item.name, item.quality, item.max = name, quality or 1, math.max(maxStack or 1, 1)
          item.equipLoc, item.class, item.subclass = equipLoc or "", class or 15, subclass or 0
          item.family = C_Item.GetItemFamily(id) or 0
        else
          item.fixed = true
        end
        entry.item = item
        table.insert(items, entry)
      end
      table.insert(slots, entry)
    end
  end
  return slots, items
end

local function group(item)
  if item.id == HEARTHSTONE then return 1 end
  if item.quality == 0 then return 8 end
  return CLASS_GROUP[item.class] or 7
end

local function before(a, b)
  local ga, gb = group(a), group(b)
  if ga ~= gb then return ga < gb end
  local function by(x, y, descending)
    if x == y then return nil end
    if descending then return x > y end
    return x < y
  end
  local keys
  if ga == 2 then
    keys = {
      { SLOT_ORDER[a.equipLoc] or 30, SLOT_ORDER[b.equipLoc] or 30 },
      { a.subclass, b.subclass }, { a.quality, b.quality, true }, { a.name, b.name },
    }
  elseif ga == 7 then
    keys = {
      { a.quality, b.quality, true }, { a.class, b.class }, { a.subclass, b.subclass },
      { a.name, b.name },
    }
  else
    keys = { { a.class, b.class }, { a.subclass, b.subclass }, { a.name, b.name } }
  end
  table.insert(keys, { a.count, b.count, true })
  table.insert(keys, { a.link, b.link })
  for _, k in ipairs(keys) do
    local r = by(k[1], k[2], k[3])
    if r ~= nil then return r end
  end
  return false
end

local function same(a, b)
  if a == nil or b == nil then return a == b end
  return a.link == b.link and a.count == b.count
end

-- One round's moves: stack merges if any, else the swaps that bring items to their places.
local function plan(bags, rightToLeft)
  local slots, items = read(bags)
  local moves, used = {}, {}
  local function key(e) return e.bag .. ":" .. e.slot end
  local function take(a, b)
    if used[key(a)] or used[key(b)] then return false end
    used[key(a)], used[key(b)] = true, true
    return true
  end

  -- Combine partial stacks: the last partial stack of an item into the first. A swap of two
  -- stacks of one item merges them, or fills the first and keeps the rest (Player::SwapItem).
  local partials, order = {}, {}
  for _, e in ipairs(items) do
    local item = e.item
    if not item.fixed and item.max > 1 and item.count < item.max then
      if not partials[item.link] then partials[item.link] = {}; table.insert(order, item.link) end
      table.insert(partials[item.link], e)
    end
  end
  for _, link in ipairs(order) do
    local list = partials[link]
    local n = table.getn(list)
    if n > 1 and take(list[n], list[1]) then
      table.insert(moves, { from = list[n], to = list[1] })
    end
  end
  if table.getn(moves) > 0 then return moves end

  -- Places: the slots no fixed item holds, specialty bags apart, in fill order.
  local general, special = {}, {}
  for i = 1, table.getn(slots) do
    local e = slots[rightToLeft and (table.getn(slots) + 1 - i) or i]
    if not (e.item and e.item.fixed) then
      table.insert(e.family ~= 0 and special or general, e)
    end
  end
  local sorted = {}
  for _, e in ipairs(items) do
    if not e.item.fixed then table.insert(sorted, e.item) end
  end
  table.sort(sorted, before)
  local want, placed = {}, {}
  for _, e in ipairs(special) do
    for i, item in ipairs(sorted) do
      if not placed[i] and accepts(e.family, item) then
        want[key(e)], placed[i] = item, true
        break
      end
    end
  end
  local first, last = 1, table.getn(general)
  for i, item in ipairs(sorted) do
    if not placed[i] then
      if group(item) == 8 then
        want[key(general[last])], last = item, last - 1
      else
        want[key(general[first])], first = item, first + 1
      end
    end
  end

  -- Swaps: each place whose item differs takes a matching item from a slot not yet in place.
  local places = {}
  for _, e in ipairs(special) do table.insert(places, e) end
  for _, e in ipairs(general) do table.insert(places, e) end
  for _, target in ipairs(places) do
    local wanted = want[key(target)]
    if wanted and not same(wanted, target.item) and not used[key(target)] then
      for _, source in ipairs(places) do
        if source ~= target and not used[key(source)] and same(source.item, wanted)
            and not same(source.item, want[key(source)])
            and (target.item == nil or accepts(source.family, target.item)) then
          if take(source, target) then table.insert(moves, { from = source, to = target }) end
          break
        end
      end
    end
  end
  return moves
end

local function signature(bags)
  local parts = {}
  for _, bag in ipairs(bags) do
    for slot = 1, GetContainerNumSlots(bag) do
      local _, count = GetContainerItemInfo(bag, slot)
      table.insert(parts, (C_Container.GetContainerItemID(bag, slot) or "-") .. ":" .. (count or ""))
    end
  end
  return table.concat(parts, ";")
end

local driver = CreateFrame("Frame")
local job

local function finish()
  job = nil
  driver:Hide()
end

driver:Hide()
driver:SetScript("OnUpdate", function()
  if not job then return finish() end
  if engine.busy() then return end
  if job.bank and not engine.bank_open() then return finish() end
  if GetTime() - job.started > TIME_LIMIT or job.rounds >= ROUND_LIMIT then return finish() end
  local now = signature(job.bags)
  if now == job.last then
    job.stuck = job.stuck + 1
    if job.stuck >= STUCK_LIMIT then return finish() end
  else
    job.stuck = 0
  end
  job.last = now
  local moves = plan(job.bags, flag("sortBagsRightToLeft"))
  if table.getn(moves) == 0 then return finish() end
  job.rounds = job.rounds + 1
  for _, m in ipairs(moves) do
    C_Container.SwapItems(m.from.bag, m.from.slot, m.to.bag, m.to.slot)
  end
end)

local function start(bags, bank)
  if job then return end
  job = { bags = bags, bank = bank, started = GetTime(), rounds = 0, stuck = 0 }
  driver:Show()
end

function C_Container.SortBags()
  local bags = {}
  if not flag("backpackAutosortDisabled") then table.insert(bags, 0) end
  for bag = 1, 4 do table.insert(bags, bag) end
  start(bags, false)
end

function C_Container.SortBankBags()
  if not engine.bank_open() then return end
  local bags = {}
  if not flag("bankAutosortDisabled") then table.insert(bags, -1) end
  for bag = 5, 10 do table.insert(bags, bag) end
  start(bags, true)
end
