-- Combat Log settings: the Combat Log page of Classic Era's ChatConfigFrame
-- (Blizzard_ChatFrame/Classic/ChatConfigFrame), built with pfUI's widgets.
-- The combatlog module opens it from the Combat Log tab's menu and hands it
-- the filter state as `log`.
setfenv(1, pfUI:GetEnvironment())

pfUI.combatlog = {}

-- ChatConfigFrame's eventList states for the subtypes of an unchecked group
local GRAY_CHECKED = 1
local UNCHECKED_ENABLED = 2
local UNCHECKED_DISABLED = 3

local SCHOOL_MASK_FIRE = 4
local SCHOOL_MASK_FROST = 16

-- ChatConfigFrame's cap on saved filters
local MAX_COMBATLOG_FILTERS = 20
local FILTER_ROWS = 4
local FILTER_ROW_HEIGHT = 16

local function Resolve(value)
  if type(value) == "function" then return value() end
  return value
end

local function SetCheck(check, checked)
  check:SetChecked(checked and 1 or nil)
end

local function SetCheckEnabled(check, enabled)
  if enabled then
    check:Enable()
    check.text:SetTextColor(1, 1, 1)
  else
    check:Disable()
    check.text:SetTextColor(.5, .5, .5)
  end
end

-- ChatConfigCheckButtonTemplate and ChatConfigSmallCheckButtonTemplate
local function CreateCheck(parent, text, tooltip, small)
  local check = CreateFrame("CheckButton", nil, parent, "UICheckButtonTemplate")
  SkinCheckbox(check, small and 12 or 14)
  check.text = check:CreateFontString(nil, "OVERLAY", small and "GameFontHighlightSmall" or "GameFontHighlight")
  check.text:SetPoint("LEFT", check, "RIGHT", 2, 0)
  check.text:SetText(text)
  check:SetHitRectInsets(0, small and -70 or -145, 0, 0)
  check.tooltip = tooltip
  check:SetScript("OnClick", function(self)
    local checked = self:GetChecked()
    if self.func then self.func(self, checked) end
    PlaySound(checked and SOUNDKIT.IG_MAINMENU_OPTION_CHECKBOX_ON or SOUNDKIT.IG_MAINMENU_OPTION_CHECKBOX_OFF)
  end)
  check:SetScript("OnEnter", function(self)
    if self.tooltip then
      GameTooltip:SetOwner(self, "ANCHOR_RIGHT")
      GameTooltip:SetText(self.tooltip, 1, 1, 1, 1, 1)
      GameTooltip:Show()
    end
  end)
  check:SetScript("OnLeave", function()
    GameTooltip:Hide()
  end)
  return check
end

-- ChatConfigBoxWithHeaderTemplate
local function CreateBox(parent, title)
  local box = CreateFrame("Frame", nil, parent)
  CreateBackdrop(box, nil, nil, .75)
  if title then
    box.title = box:CreateFontString(nil, "OVERLAY", "GameFontNormal")
    box.title:SetPoint("BOTTOMLEFT", box, "TOPLEFT", 0, 4)
    box.title:SetText(title)
  end
  return box
end

local function CreateSwatch(parent)
  local swatch = CreateFrame("Button", nil, parent)
  swatch:SetSize(24, 12)
  CreateBackdrop(swatch)
  swatch.prev = swatch.backdrop:CreateTexture(nil, "OVERLAY")
  swatch.prev:SetAllPoints(swatch)
  if swatch.backdrop_shadow then swatch.backdrop_shadow:Hide() end
  return swatch
end

local function CreateButton(parent, text, width)
  local button = CreateFrame("Button", nil, parent, "UIPanelButtonTemplate")
  SkinButton(button)
  button:SetSize(width, 22)
  button:SetText(text)
  return button
end

function pfUI.combatlog.CreateConfig(log)
  local selected      -- CHATCONFIG_SELECTED_FILTER
  local selectedIndex -- ChatConfigCombatSettingsFilters.selectedFilter
  local changed       -- the log's active filter was edited; it reapplies on close
  local Update        -- refreshes every page from the selected filter

  -- the log reapplies its filter on close only after an edit to that filter,
  -- so closing the window doesn't clear what the log holds
  local function Edited()
    local filters = log.GetFilters()
    if selected == filters.filters[filters.currentFilter] then changed = true end
    Update()
  end

  --
  -- The filter model (ChatConfigFrame's helpers). A filter's first part holds
  -- the event list and the source flags. Its second part holds the dest flags,
  -- unless it names units by GUID; a one-part filter keeps both in its first.
  --
  local function UsesGUID(direction)
    local flags = selected.filters[1][direction == "SOURCE" and "sourceFlags" or "destFlags"]
    for k in pairs(flags or {}) do
      if type(k) == "string" then return true end
    end
    return false
  end

  local function DestFilter()
    if UsesGUID("SOURCE") or UsesGUID("DEST") then return selected.filters[1] end
    return selected.filters[2] or selected.filters[1]
  end

  -- no flag table means every unit; unchecking one starts from all of them
  local function AllFlags()
    return {
      [COMBATLOG_FILTER_MINE] = true, [COMBATLOG_FILTER_MY_PET] = true,
      [COMBATLOG_FILTER_FRIENDLY_UNITS] = true, [COMBATLOG_FILTER_HOSTILE_PLAYERS] = true,
      [COMBATLOG_FILTER_HOSTILE_UNITS] = true, [COMBATLOG_FILTER_NEUTRAL_UNITS] = true,
      [COMBATLOG_FILTER_UNKNOWN_UNITS] = true,
    }
  end

  local function IsMessageDoneBy(flag)
    local flags = selected.filters[1].sourceFlags
    return not flags or flags[flag]
  end

  local function IsMessageDoneTo(flag)
    local flags = DestFilter().destFlags
    return not flags or flags[flag]
  end

  local function ToggleMessageSource(checked, flag)
    local filter = selected.filters[1]
    filter.sourceFlags = filter.sourceFlags or AllFlags()
    filter.sourceFlags[flag] = checked and true or false
  end

  local function ToggleMessageDest(checked, flag)
    local filter = DestFilter()
    filter.destFlags = filter.destFlags or AllFlags()
    filter.destFlags[flag] = checked and true or false
  end

  local function GetMessageTypeState(messageType)
    if type(messageType) == "table" then messageType = messageType[1] end
    return selected.filters[1].eventList[messageType]
  end

  -- only the first type counts; a subtype's types go on and off together
  local function HasMessageType(messageType)
    local state = GetMessageTypeState(messageType)
    return state and state ~= UNCHECKED_ENABLED and state ~= UNCHECKED_DISABLED
  end

  local function ToggleMessageType(checked, ...)
    local eventList = selected.filters[1].eventList
    for _, messageType in pairs({ ... }) do
      eventList[messageType] = checked and true or false
    end
  end

  -- A group shows checked unless its first subtype with a state is parked.
  local function HasMessageTypeGroup(group)
    for _, sub in ipairs(group.subTypes) do
      local state = GetMessageTypeState(sub.type)
      if state == GRAY_CHECKED or state == UNCHECKED_DISABLED then
        return false
      elseif state then
        return true
      end
    end
    return false
  end

  -- Unchecking a group parks its subtypes' states, so checking it again
  -- restores the ones that were on.
  local function ToggleMessageTypeGroup(checked, group)
    local eventList = selected.filters[1].eventList
    for _, sub in ipairs(group.subTypes) do
      local state = GetMessageTypeState(sub.type)
      local on = state == true or state == GRAY_CHECKED
      local value
      if checked then
        value = on and true or UNCHECKED_ENABLED
      else
        value = on and GRAY_CHECKED or UNCHECKED_DISABLED
      end
      local types = type(sub.type) == "table" and sub.type or { sub.type }
      for _, messageType in pairs(types) do
        eventList[messageType] = value
      end
    end
  end

  --
  -- The checkbox tables (COMBAT_CONFIG_MESSAGESOURCES_*, _MESSAGETYPES_*).
  -- Enchants and pet loyalty are left out: ClassicAPI never logs them.
  --
  local function Source(text, flag, tooltip)
    return {
      text = text,
      checked = function() return IsMessageDoneBy(flag) end,
      hidden = function() return UsesGUID("SOURCE") end,
      func = function(self, checked) ToggleMessageSource(checked, flag) end,
      tooltip = tooltip,
    }
  end

  local function Dest(text, flag, tooltip)
    return {
      text = text,
      checked = function() return IsMessageDoneTo(flag) end,
      hidden = function() return UsesGUID("DEST") end,
      func = function(self, checked) ToggleMessageDest(checked, flag) end,
      tooltip = tooltip,
    }
  end

  local MESSAGESOURCES_BY = {
    {
      text = function() return UsesGUID("SOURCE") and COMBATLOG_FILTER_STRING_CUSTOM_UNIT or COMBATLOG_FILTER_STRING_ME end,
      checked = function() return UsesGUID("SOURCE") or IsMessageDoneBy(COMBATLOG_FILTER_MINE) end,
      disabled = function() return UsesGUID("SOURCE") end,
      func = function(self, checked) ToggleMessageSource(checked, COMBATLOG_FILTER_MINE) end,
      tooltip = FILTER_BY_ME_COMBATLOG_TOOLTIP,
    },
    Source(COMBATLOG_FILTER_STRING_MY_PET, COMBATLOG_FILTER_MY_PET, FILTER_BY_PET_COMBATLOG_TOOLTIP),
    Source(COMBATLOG_FILTER_STRING_FRIENDLY_UNITS, COMBATLOG_FILTER_FRIENDLY_UNITS, FILTER_BY_FRIENDS_COMBATLOG_TOOLTIP),
    Source(COMBATLOG_FILTER_STRING_HOSTILE_PLAYERS, COMBATLOG_FILTER_HOSTILE_PLAYERS, FILTER_BY_HOSTILE_PLAYERS_COMBATLOG_TOOLTIP),
    Source(COMBATLOG_FILTER_STRING_HOSTILE_UNITS, COMBATLOG_FILTER_HOSTILE_UNITS, FILTER_BY_ENEMIES_COMBATLOG_TOOLTIP),
    Source(COMBATLOG_FILTER_STRING_NEUTRAL_UNITS, COMBATLOG_FILTER_NEUTRAL_UNITS, FILTER_BY_NEUTRAL_COMBATLOG_TOOLTIP),
    Source(COMBATLOG_FILTER_STRING_UNKNOWN_UNITS, COMBATLOG_FILTER_UNKNOWN_UNITS, FILTER_BY_UNKNOWN_COMBATLOG_TOOLTIP),
  }

  local MESSAGESOURCES_TO = {
    {
      text = function() return UsesGUID("DEST") and COMBATLOG_FILTER_STRING_CUSTOM_UNIT or COMBATLOG_FILTER_STRING_ME end,
      checked = function() return UsesGUID("DEST") or IsMessageDoneTo(COMBATLOG_FILTER_MINE) end,
      disabled = function() return UsesGUID("DEST") end,
      func = function(self, checked) ToggleMessageDest(checked, COMBATLOG_FILTER_MINE) end,
      tooltip = FILTER_TO_ME_COMBATLOG_TOOLTIP,
    },
    Dest(COMBATLOG_FILTER_STRING_MY_PET, COMBATLOG_FILTER_MY_PET, FILTER_TO_PET_COMBATLOG_TOOLTIP),
    Dest(COMBATLOG_FILTER_STRING_FRIENDLY_UNITS, COMBATLOG_FILTER_FRIENDLY_UNITS, FILTER_TO_FRIENDS_COMBATLOG_TOOLTIP),
    Dest(COMBATLOG_FILTER_STRING_HOSTILE_PLAYERS, COMBATLOG_FILTER_HOSTILE_PLAYERS, FILTER_TO_HOSTILE_PLAYERS_COMBATLOG_TOOLTIP),
    Dest(COMBATLOG_FILTER_STRING_HOSTILE_UNITS, COMBATLOG_FILTER_HOSTILE_UNITS, FILTER_TO_HOSTILE_COMBATLOG_TOOLTIP),
    Dest(COMBATLOG_FILTER_STRING_NEUTRAL_UNITS, COMBATLOG_FILTER_NEUTRAL_UNITS, FILTER_TO_NEUTRAL_COMBATLOG_TOOLTIP),
    Dest(COMBATLOG_FILTER_STRING_UNKNOWN_UNITS, COMBATLOG_FILTER_UNKNOWN_UNITS, FILTER_TO_UNKNOWN_COMBATLOG_TOOLTIP),
  }

  -- a subtype that turns its event types on and off together
  local function Type(text, tooltip, ...)
    local types = { ... }
    return {
      text = text,
      type = types,
      checked = function() return HasMessageType(types) end,
      func = function(self, checked) ToggleMessageType(checked, unpack(types)) end,
      tooltip = tooltip,
    }
  end

  local AURA_TYPES = { "SPELL_AURA_APPLIED", "SPELL_AURA_APPLIED_DOSE", "SPELL_AURA_REMOVED", "SPELL_AURA_REMOVED_DOSE", "SPELL_AURA_REFRESH" }

  -- buffs and debuffs share the aura events; those stop only when both are off
  local function Aura(text, tooltip, setting, other)
    return {
      text = text,
      type = AURA_TYPES,
      checked = function() return not selected.settings[setting] end,
      func = function(self, checked)
        selected.settings[setting] = not checked
        if checked or selected.settings[other] then
          ToggleMessageType(checked, unpack(AURA_TYPES))
        end
      end,
      tooltip = tooltip,
    }
  end

  local MESSAGETYPES_LEFT = {
    {
      text = MELEE, tooltip = MELEE_COMBATLOG_TOOLTIP,
      subTypes = {
        Type(DAMAGE, SWING_DAMAGE_COMBATLOG_TOOLTIP, "SWING_DAMAGE"),
        Type(MISSES, SWING_MISSED_COMBATLOG_TOOLTIP, "SWING_MISSED"),
      },
    },
    {
      text = RANGED, tooltip = RANGED_COMBATLOG_TOOLTIP,
      subTypes = {
        Type(DAMAGE, RANGE_DAMAGE_COMBATLOG_TOOLTIP, "RANGE_DAMAGE"),
        Type(MISSES, RANGE_MISSED_COMBATLOG_TOOLTIP, "RANGE_MISSED"),
      },
    },
    {
      text = AURAS, tooltip = AURAS_COMBATLOG_TOOLTIP,
      subTypes = {
        Aura(BENEFICIAL, BENEFICIAL_AURA_COMBATLOG_TOOLTIP, "hideBuffs", "hideDebuffs"),
        Aura(HOSTILE, HARMFUL_AURA_COMBATLOG_TOOLTIP, "hideDebuffs", "hideBuffs"),
        Type(DISPELS, DISPEL_AURA_COMBATLOG_TOOLTIP, "SPELL_DISPEL", "SPELL_DISPEL_FAILED", "SPELL_STOLEN"),
      },
    },
    {
      text = PERIODIC, tooltip = SPELL_PERIODIC_COMBATLOG_TOOLTIP,
      subTypes = {
        Type(DAMAGE, SPELL_PERIODIC_DAMAGE_COMBATLOG_TOOLTIP, "SPELL_PERIODIC_DAMAGE"),
        Type(MISSES, SPELL_PERIODIC_MISSED_COMBATLOG_TOOLTIP, "SPELL_PERIODIC_MISSED"),
        Type(HEALS, SPELL_PERIODIC_HEAL_COMBATLOG_TOOLTIP, "SPELL_PERIODIC_HEAL"),
        Type(OTHER, SPELL_PERIODIC_OTHER_COMBATLOG_TOOLTIP, "SPELL_PERIODIC_DRAIN", "SPELL_PERIODIC_LEECH"),
      },
    },
  }

  local MESSAGETYPES_RIGHT = {
    {
      text = SPELLS, tooltip = SPELLS_COMBATLOG_TOOLTIP,
      subTypes = {
        Type(DAMAGE, SPELL_DAMAGE_COMBATLOG_TOOLTIP, "SPELL_DAMAGE"),
        Type(MISSES, SPELL_MISSED_COMBATLOG_TOOLTIP, "SPELL_MISSED"),
        Type(HEALS, SPELL_HEAL_COMBATLOG_TOOLTIP, "SPELL_HEAL"),
        Type(POWER_GAINS, POWER_GAINS_COMBATLOG_TOOLTIP, "SPELL_ENERGIZE", "SPELL_PERIODIC_ENERGIZE"),
        Type(DRAINS, SPELL_DRAIN_COMBATLOG_TOOLTIP, "SPELL_DRAIN", "SPELL_LEECH"),
        Type(INTERRUPTS, SPELL_INTERRUPT_COMBATLOG_TOOLTIP, "SPELL_INTERRUPT"),
        Type(SPECIAL, SPELL_INSTAKILL_COMBATLOG_TOOLTIP, "SPELL_INSTAKILL", "SPELL_DURABILITY_DAMAGE", "SPELL_DURABILITY_DAMAGE_ALL"),
        Type(EXTRA_ATTACKS, SPELL_EXTRA_ATTACKS_COMBATLOG_TOOLTIP, "SPELL_EXTRA_ATTACKS"),
        Type(SUMMONS, SPELL_SUMMON_COMBATLOG_TOOLTIP, "SPELL_SUMMON"),
        Type(RESURRECT, SPELL_RESURRECT_COMBATLOG_TOOLTIP, "SPELL_RESURRECT"),
      },
    },
    {
      text = SPELL_CASTING, tooltip = SPELL_CASTING_COMBATLOG_TOOLTIP,
      subTypes = {
        Type(START, SPELL_CAST_START_COMBATLOG_TOOLTIP, "SPELL_CAST_START"),
        Type(SUCCESS, SPELL_CAST_SUCCESS_COMBATLOG_TOOLTIP, "SPELL_CAST_SUCCESS"),
        Type(FAILURES, SPELL_CAST_FAILED_COMBATLOG_TOOLTIP, "SPELL_CAST_FAILED"),
      },
    },
  }

  local MESSAGETYPES_MISC = {
    Type(DAMAGE_SHIELD, DAMAGE_SHIELD_COMBATLOG_TOOLTIP, "DAMAGE_SHIELD", "DAMAGE_SHIELD_MISSED"),
    Type(ENVIRONMENTAL_DAMAGE, ENVIRONMENTAL_DAMAGE_COMBATLOG_TOOLTIP, "ENVIRONMENTAL_DAMAGE"),
    Type(KILLS, KILLS_COMBATLOG_TOOLTIP, "PARTY_KILL"),
    Type(DEATHS, DEATHS_COMBATLOG_TOOLTIP, "UNIT_DIED", "UNIT_DESTROYED", "UNIT_DISSIPATES"),
  }

  local UNIT_COLORS = {
    { text = COMBATLOG_FILTER_STRING_ME, flag = COMBATLOG_FILTER_MINE },
    { text = COMBATLOG_FILTER_STRING_MY_PET, flag = COMBATLOG_FILTER_MY_PET },
    { text = COMBATLOG_FILTER_STRING_FRIENDLY_UNITS, flag = COMBATLOG_FILTER_FRIENDLY_UNITS },
    { text = COMBATLOG_FILTER_STRING_HOSTILE_UNITS, flag = COMBATLOG_FILTER_HOSTILE_UNITS },
    { text = COMBATLOG_FILTER_STRING_HOSTILE_PLAYERS, flag = COMBATLOG_FILTER_HOSTILE_PLAYERS },
    { text = COMBATLOG_FILTER_STRING_NEUTRAL_UNITS, flag = COMBATLOG_FILTER_NEUTRAL_UNITS },
    { text = COMBATLOG_FILTER_STRING_UNKNOWN_UNITS, flag = COMBATLOG_FILTER_UNKNOWN_UNITS },
  }

  --
  -- Checkbox builders (ChatConfig_CreateCheckboxes, _CreateTieredCheckboxes)
  --
  local function CreateCheckboxList(parent, title, entries)
    local box = CreateBox(parent, title)
    box:SetWidth(200)
    box.entries = entries
    box.checks = {}
    for i, entry in ipairs(entries) do
      local check = CreateCheck(box, nil, entry.tooltip)
      local func = entry.func
      check.func = function(self, checked) func(self, checked) Edited() end
      box.checks[i] = check
    end
    return box
  end

  -- hidden entries close up, and the box fits the rest
  local function UpdateCheckboxList(box)
    local shown = 0
    for i, entry in ipairs(box.entries) do
      local check = box.checks[i]
      if Resolve(entry.hidden) then
        check:Hide()
      else
        check:ClearAllPoints()
        check:SetPoint("TOPLEFT", box, "TOPLEFT", 8, -6 - shown * 22)
        check.text:SetText(Resolve(entry.text))
        SetCheck(check, Resolve(entry.checked))
        SetCheckEnabled(check, not Resolve(entry.disabled))
        check:Show()
        shown = shown + 1
      end
    end
    box:SetHeight(shown * 22 + 8)
  end

  -- a checkbox per group with its subtypes in two columns under it; an entry
  -- without subtypes is a plain checkbox
  local function CreateTieredList(parent, entries, small)
    local list = CreateFrame("Frame", nil, parent)
    list:SetWidth(220)
    list.entries = entries
    list.checks = {}
    local y = 0
    for i, entry in ipairs(entries) do
      local check = CreateCheck(list, entry.text, entry.tooltip, small)
      check:SetPoint("TOPLEFT", list, "TOPLEFT", 0, -y)
      y = y + 22
      check.subs = {}
      if entry.subTypes then
        check.func = function(self, checked) ToggleMessageTypeGroup(checked, entry) Edited() end
        for k, sub in ipairs(entry.subTypes) do
          local subcheck = CreateCheck(list, sub.text, sub.tooltip, true)
          subcheck:SetPoint("TOPLEFT", check, "BOTTOMLEFT", 15 + (k - 1) % 2 * 95, -2 - ceil(k / 2 - 1) * 20)
          local func = sub.func
          subcheck.func = function(self, checked) func(self, checked) Edited() end
          check.subs[k] = subcheck
        end
        y = y + ceil(#entry.subTypes / 2) * 20 + 8
      else
        local func = entry.func
        check.func = function(self, checked) func(self, checked) Edited() end
      end
      list.checks[i] = check
    end
    list:SetHeight(y)
    return list
  end

  -- subtypes take clicks only under a checked group
  local function UpdateTieredList(list)
    for i, entry in ipairs(list.entries) do
      local check = list.checks[i]
      if entry.subTypes then
        local groupChecked = HasMessageTypeGroup(entry)
        SetCheck(check, groupChecked)
        for k, sub in ipairs(entry.subTypes) do
          SetCheck(check.subs[k], Resolve(sub.checked))
          SetCheckEnabled(check.subs[k], groupChecked)
        end
      else
        SetCheck(check, Resolve(entry.checked))
      end
    end
  end

  local function PickColor(color)
    local r, g, b = color.r, color.g, color.b
    -- SetColorRGB runs the installed func; drop the last picker user's first
    ColorPickerFrame.func = nil
    ColorPickerFrame.cancelFunc = nil
    ColorPickerFrame.hasOpacity = nil
    ColorPickerFrame.opacityFunc = nil
    ColorPickerFrame:SetColorRGB(r, g, b)
    ColorPickerFrame.func = function()
      color.r, color.g, color.b = ColorPickerFrame:GetColorRGB()
      Edited()
    end
    ColorPickerFrame.cancelFunc = function()
      color.r, color.g, color.b = r, g, b
      Edited()
    end
    ColorPickerFrame:SetFrameStrata("DIALOG")
    ShowUIPanel(ColorPickerFrame)
  end

  -- CombatConfig_Colorize_Update's example lines: you hit a target, it hits you
  local function UpdateExamples(page)
    local player, guid = UnitName("player"), UnitGUID("player")
    local text, r, g, b = log.FormatEvent(selected, 0, "SPELL_DAMAGE", false,
      guid, player, 0x511, 0, "0xF13000012B000820", EXAMPLE_TARGET_MONSTER, 0x10a28, 0,
      116, EXAMPLE_SPELL_FROSTBOLT, SCHOOL_MASK_FROST, 27, 0, SCHOOL_MASK_FROST, nil, nil, nil, 1)
    page.example1:SetText(text)
    page.example1:SetTextColor(r, g, b)

    text, r, g, b = log.FormatEvent(selected, 0, "SPELL_DAMAGE", false,
      "0xF13000024D002914", EXAMPLE_TARGET_MONSTER, 0x10a48, 0, guid, player, 0x511, 0,
      20793, EXAMPLE_SPELL_FIREBALL, SCHOOL_MASK_FIRE, 68, 0, SCHOOL_MASK_FIRE)
    page.example2:SetText(text)
    page.example2:SetTextColor(r, g, b)
  end

  local function CreateExamples(page)
    page.example1 = page:CreateFontString(nil, "OVERLAY", "GameFontNormal")
    page.example1:SetPoint("TOPLEFT", page, "TOPLEFT", 25, -27)
    page.example1:SetWidth(540)
    page.example1:SetJustifyH("LEFT")
    page.example2 = page:CreateFontString(nil, "OVERLAY", "GameFontNormal")
    page.example2:SetPoint("TOPLEFT", page.example1, "BOTTOMLEFT", 0, 0)
    page.example2:SetWidth(540)
    page.example2:SetJustifyH("LEFT")
    local title = page:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
    title:SetPoint("BOTTOMLEFT", page.example1, "TOPLEFT", 0, 3)
    title:SetText(EXAMPLE_TEXT)
  end

  -- a checkbox bound to one of the selected filter's settings
  local function SettingCheck(parent, text, tooltip, setting, small)
    local check = CreateCheck(parent, text, tooltip, small)
    check.setting = setting
    check.func = function(self, checked)
      selected.settings[setting] = checked and true or false
      Edited()
    end
    return check
  end

  --
  -- The window (ChatConfigFrame with only its Combat Log category)
  --
  local frame = CreateFrame("Frame", "pfCombatLogConfig", UIParent)
  frame:SetSize(620, 600)
  frame:SetPoint("CENTER", 0, 0)
  frame:SetFrameStrata("HIGH")
  frame:SetMovable(true)
  frame:EnableMouse(true)
  frame:SetClampedToScreen(true)
  frame:RegisterForDrag("LeftButton")
  frame:SetScript("OnDragStart", function() this:StartMoving() end)
  frame:SetScript("OnDragStop", function() this:StopMovingOrSizing() end)
  CreateBackdrop(frame, nil, nil, .9)
  CreateBackdropShadow(frame)
  tinsert(UISpecialFrames, "pfCombatLogConfig")
  frame:Hide()

  frame.title = frame:CreateFontString(nil, "OVERLAY", "GameFontNormal")
  frame.title:SetPoint("TOP", frame, "TOP", 0, -10)
  frame.title:SetText(COMBAT_LOG)

  local closeButton = CreateFrame("Button", nil, frame, "UIPanelCloseButton")
  closeButton:SetPoint("TOPRIGHT", frame, "TOPRIGHT", -4, -4)
  SkinCloseButton(closeButton, frame.backdrop or frame, -6, -6)

  --
  -- The saved filters: a list to pick one from, and buttons to reorder,
  -- copy, add and delete them
  --
  local filterList = CreateFrame("Frame", nil, frame, "pfUIScrollBoxListTemplate")
  filterList:SetPoint("TOPLEFT", frame, "TOPLEFT", 12, -32)
  filterList:SetPoint("TOPRIGHT", frame, "TOPRIGHT", -12, -32)
  -- the template keeps 5px around the rows, which sit 2px apart
  filterList:SetHeight(FILTER_ROWS * (FILTER_ROW_HEIGHT + 2) - 2 + 10)
  CreateBackdrop(filterList, nil, nil, .75)

  local SelectFilter

  local function InitFilterRow(row, entry)
    if not row.text then
      row.highlight = row:CreateTexture(nil, "BACKGROUND")
      row.highlight:SetAllPoints(row)
      row.highlight:SetColorTexture(.3, .3, .3, .4)
      row.hover = row:CreateTexture(nil, "BACKGROUND")
      row.hover:SetAllPoints(row)
      row.hover:SetColorTexture(1, 1, 1, .1)
      row.hover:Hide()
      row.text = row:CreateFontString(nil, "OVERLAY", "GameFontNormal")
      row.text:SetPoint("LEFT", row, "LEFT", 4, 0)
      row:SetScript("OnClick", function() SelectFilter(row.index) end)
      row:SetScript("OnEnter", function() row.hover:Show() end)
      row:SetScript("OnLeave", function() row.hover:Hide() end)
    end
    row.index = entry.index
    row.text:SetText(entry.name)
    row.highlight:SetShown(entry.index == selectedIndex)
  end

  local filterView = CreateScrollBoxListLinearView(0, 0, 0, 2, 2)
  filterView:SetElementExtent(FILTER_ROW_HEIGHT)
  filterView:SetElementInitializer("Button", InitFilterRow)
  ScrollUtil.InitScrollBoxListWithScrollBar(filterList.ScrollBox, filterList.ScrollBar, filterView)

  -- ChatConfig_MoveFilterUp / _MoveFilterDown; the active filter keeps its
  -- place in the log
  local function MoveFilter(offset)
    local filters = log.GetFilters()
    local to = selectedIndex + offset
    if to < 1 or to > #filters.filters then return end
    filters.filters[selectedIndex], filters.filters[to] = filters.filters[to], filters.filters[selectedIndex]
    if filters.currentFilter == selectedIndex then
      filters.currentFilter = to
    elseif filters.currentFilter == to then
      filters.currentFilter = selectedIndex
    end
    log.UpdateQuickButtons()
    SelectFilter(to)
  end

  local function CreateMoveButton(dir, tooltip, offset)
    local button = CreateFrame("Button", nil, frame)
    SkinArrowButton(button, dir, 16)
    button:SetScript("OnClick", function()
      MoveFilter(offset)
      PlaySound(SOUNDKIT.U_CHAT_SCROLL_BUTTON)
    end)
    button:SetScript("OnEnter", function()
      GameTooltip:SetOwner(this, "ANCHOR_RIGHT")
      GameTooltip:SetText(tooltip, 1, 1, 1)
      GameTooltip:Show()
    end)
    button:SetScript("OnLeave", function() GameTooltip:Hide() end)
    return button
  end

  local moveUp = CreateMoveButton("up", MOVE_FILTER_UP, -1)
  moveUp:SetPoint("TOPLEFT", filterList, "BOTTOMLEFT", 4, -8)
  local moveDown = CreateMoveButton("down", MOVE_FILTER_DOWN, 1)
  moveDown:SetPoint("LEFT", moveUp, "RIGHT", 8, 0)

  -- CombatConfig_DeleteCurrentCombatFilter; the log switches to the first
  -- filter only when the one it shows is deleted
  local function DeleteFilter()
    local filters = log.GetFilters()
    if #filters.filters <= 1 then return end
    tremove(filters.filters, selectedIndex)
    if selectedIndex == filters.currentFilter then
      log.SelectFilter(1)
      changed = nil
    else
      if selectedIndex < filters.currentFilter then
        filters.currentFilter = filters.currentFilter - 1
      end
      log.UpdateQuickButtons()
    end
    SelectFilter(1)
  end

  local deleteButton = CreateButton(frame, DELETE, 70)
  deleteButton:SetPoint("TOPRIGHT", filterList, "BOTTOMRIGHT", 0, -6)
  deleteButton:SetScript("OnClick", function()
    CreateQuestionDialog(CONFIRM_COMBAT_FILTER_DELETE, DeleteFilter)
  end)

  local addButton = CreateButton(frame, ADD_FILTER, 90)
  addButton:SetPoint("RIGHT", deleteButton, "LEFT", -6, 0)
  addButton:SetScript("OnClick", function()
    log.NewFilter(nil, SelectFilter)
  end)

  local copyButton = CreateButton(frame, COPY_FILTER, 90)
  copyButton:SetPoint("RIGHT", addButton, "LEFT", -6, 0)
  copyButton:SetScript("OnClick", function()
    log.NewFilter(selected, SelectFilter)
  end)

  --
  -- The selected filter's settings, one tab at a time
  --
  local content = CreateFrame("Frame", nil, frame)
  content:SetPoint("TOPLEFT", frame, "TOPLEFT", 12, -172)
  content:SetPoint("BOTTOMRIGHT", frame, "BOTTOMRIGHT", -12, 42)
  CreateBackdrop(content, nil, nil, .75)

  local tabs = {}
  local function SelectTab(id)
    for i, tab in ipairs(tabs) do
      tab.page:SetShown(i == id)
      if i == id then tab:LockHighlight() else tab:UnlockHighlight() end
    end
  end

  local function CreateTab(text)
    local tab = CreateButton(frame, text, 10)
    tab:SetHeight(20)
    tab:SetWidth(tab:GetTextWidth() + 24)
    local id = #tabs + 1
    if id == 1 then
      tab:SetPoint("BOTTOMLEFT", content, "TOPLEFT", 0, 6)
    else
      tab:SetPoint("LEFT", tabs[id - 1], "RIGHT", 4, 0)
    end
    tab:SetScript("OnClick", function()
      PlaySound(SOUNDKIT.U_CHAT_SCROLL_BUTTON)
      SelectTab(id)
    end)
    tab.page = CreateFrame("Frame", nil, content)
    tab.page:SetAllPoints(content)
    tabs[id] = tab
    return tab.page
  end

  -- Message Sources
  local page = CreateTab(MESSAGE_SOURCES)
  local doneBy = CreateCheckboxList(page, DONE_BY, MESSAGESOURCES_BY)
  doneBy:SetPoint("TOPLEFT", page, "TOPLEFT", 13, -25)
  local doneTo = CreateCheckboxList(page, DONE_TO, MESSAGESOURCES_TO)
  doneTo:SetPoint("TOPLEFT", doneBy, "TOPRIGHT", 50, 0)

  -- Message Types
  page = CreateTab(MESSAGE_TYPES)
  local typesLeft = CreateTieredList(page, MESSAGETYPES_LEFT)
  typesLeft:SetPoint("TOPLEFT", page, "TOPLEFT", 13, -10)
  local typesRight = CreateTieredList(page, MESSAGETYPES_RIGHT)
  typesRight:SetPoint("TOPLEFT", typesLeft, "TOPRIGHT", 60, 0)
  local typesMisc = CreateTieredList(page, MESSAGETYPES_MISC, true)
  typesMisc:SetPoint("TOPLEFT", typesRight, "BOTTOMLEFT", 0, -22)
  local miscTitle = page:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
  miscTitle:SetPoint("BOTTOMLEFT", typesMisc, "TOPLEFT", 0, 4)
  miscTitle:SetText(MISCELLANEOUS)

  -- Colors
  local colors = CreateTab(COLORS)
  CreateExamples(colors)

  local unitColors = CreateBox(colors, UNIT_COLORS)
  unitColors:SetPoint("TOPLEFT", colors, "TOPLEFT", 13, -100)
  unitColors:SetSize(200, #UNIT_COLORS * 22 + 8)
  unitColors.swatches = {}
  for i, entry in ipairs(UNIT_COLORS) do
    local label = unitColors:CreateFontString(nil, "OVERLAY", "GameFontHighlight")
    label:SetPoint("TOPLEFT", unitColors, "TOPLEFT", 8, -10 - (i - 1) * 22)
    label:SetText(entry.text)
    local swatch = CreateSwatch(unitColors)
    swatch:SetPoint("RIGHT", unitColors, "TOPRIGHT", -8, -15 - (i - 1) * 22)
    swatch.flag = entry.flag
    swatch:SetScript("OnClick", function() PickColor(selected.colors.unitColoring[this.flag]) end)
    unitColors.swatches[i] = swatch
  end

  local highlighting = CreateBox(colors, HIGHLIGHTING)
  highlighting:SetPoint("TOPLEFT", unitColors, "BOTTOMLEFT", 0, -25)
  highlighting:SetSize(200, 50)
  local highlightChecks = {
    SettingCheck(highlighting, COMBATLOG_HIGHLIGHT_KILL, HIGHLIGHT_KILL_COMBATLOG_TOOLTIP, "lineHighlighting", true),
    SettingCheck(highlighting, COMBATLOG_HIGHLIGHT_ABILITY, HIGHLIGHT_ABILITY_COMBATLOG_TOOLTIP, "abilityHighlighting", true),
    SettingCheck(highlighting, COMBATLOG_HIGHLIGHT_DAMAGE, HIGHLIGHT_DAMAGE_COMBATLOG_TOOLTIP, "amountHighlighting", true),
    SettingCheck(highlighting, COMBATLOG_HIGHLIGHT_SCHOOL, HIGHLIGHT_SCHOOL_COMBATLOG_TOOLTIP, "schoolNameHighlighting", true),
  }
  for i, check in ipairs(highlightChecks) do
    check:SetPoint("TOPLEFT", highlighting, "TOPLEFT", 8 + (i - 1) % 2 * 95, -6 - ceil(i / 2 - 1) * 20)
  end

  local colorize = CreateBox(colors, COLORIZE)
  colorize:SetPoint("TOPLEFT", unitColors, "TOPRIGHT", 50, 0)
  colorize:SetSize(240, 192)

  local unitNames = CreateCheck(colorize, UNIT_NAMES, UNIT_NAMES_COMBATLOG_TOOLTIP)
  unitNames:SetPoint("TOPLEFT", colorize, "TOPLEFT", 8, -6)
  unitNames.func = function(self, checked)
    local on = checked and true or false
    selected.settings.unitColoring, selected.settings.sourceColoring, selected.settings.destColoring = on, on, on
    Edited()
  end

  local spellNames = SettingCheck(colorize, SPELL_NAMES, SPELL_NAMES_COMBATLOG_TOOLTIP, "abilityColoring")
  spellNames:SetPoint("TOPLEFT", unitNames, "BOTTOMLEFT", 0, -8)
  local spellSchool = SettingCheck(colorize, COLOR_BY_SCHOOL, SPELL_COLOR_BY_SCHOOL_COMBATLOG_TOOLTIP, "abilitySchoolColoring", true)
  spellSchool:SetPoint("TOPLEFT", spellNames, "BOTTOMLEFT", 15, -2)
  local spellSwatch = CreateSwatch(colorize)
  spellSwatch:SetPoint("RIGHT", colorize, "RIGHT", -8, 0)
  spellSwatch:SetPoint("TOP", spellNames, "TOP", 0, -4)
  spellSwatch:SetScript("OnClick", function() PickColor(selected.colors.defaults.spell) end)

  local damageNumber = SettingCheck(colorize, DAMAGE_NUMBER, SPELL_DAMAGE_NUMBER_COMBATLOG_TOOLTIP, "amountColoring")
  damageNumber:SetPoint("TOPLEFT", spellSchool, "BOTTOMLEFT", -15, -8)
  local damageSchool = SettingCheck(colorize, COLOR_BY_SCHOOL, SPELL_COLOR_BY_SCHOOL_COMBATLOG_TOOLTIP, "amountSchoolColoring", true)
  damageSchool:SetPoint("TOPLEFT", damageNumber, "BOTTOMLEFT", 15, -2)
  local damageSwatch = CreateSwatch(colorize)
  damageSwatch:SetPoint("RIGHT", colorize, "RIGHT", -8, 0)
  damageSwatch:SetPoint("TOP", damageNumber, "TOP", 0, -4)
  damageSwatch:SetScript("OnClick", function() PickColor(selected.colors.defaults.damage) end)

  local schoolNames = SettingCheck(colorize, DAMAGE_SCHOOL_TEXT, SPELL_DAMAGE_SCHOOL_COMBATLOG_TOOLTIP, "schoolNameColoring")
  schoolNames:SetPoint("TOPLEFT", damageSchool, "BOTTOMLEFT", -15, -8)

  local entireLine = SettingCheck(colorize, ENTIRE_LINE, ENTIRE_LINE_COMBATLOG_TOOLTIP, "lineColoring")
  entireLine:SetPoint("TOPLEFT", schoolNames, "BOTTOMLEFT", 0, -8)

  -- the two line color priorities, as radio buttons
  local function PriorityCheck(text, tooltip, priority)
    local check = CreateCheck(colorize, text, tooltip, true)
    check.func = function()
      selected.settings.lineColorPriority = priority
      Edited()
    end
    return check
  end
  local bySource = PriorityCheck(BY_SOURCE, BY_SOURCE_COMBATLOG_TOOLTIP, 1)
  bySource:SetPoint("TOPLEFT", entireLine, "BOTTOMLEFT", 15, -2)
  local byTarget = PriorityCheck(BY_TARGET, BY_TARGET_COMBATLOG_TOOLTIP, 2)
  byTarget:SetPoint("LEFT", bySource, "RIGHT", 80, 0)

  -- Formatting
  local formatting = CreateTab(FORMATTING)
  CreateExamples(formatting)

  local timestamp = SettingCheck(formatting, SHOW_TIMESTAMP, TIMESTAMP_COMBATLOG_TOOLTIP, "timestamp")
  timestamp:SetPoint("TOPLEFT", formatting, "TOPLEFT", 25, -100)
  local braces = SettingCheck(formatting, SHOW_BRACES, SHOW_BRACES_COMBATLOG_TOOLTIP, "braces")
  braces:SetPoint("TOPLEFT", timestamp, "BOTTOMLEFT", 0, -15)

  local unitBraces = CreateCheck(formatting, UNIT_NAMES, UNIT_NAMES_SHOW_BRACES_COMBATLOG_TOOLTIP, true)
  unitBraces:SetPoint("TOPLEFT", braces, "BOTTOMLEFT", 15, -2)
  unitBraces.func = function(self, checked)
    local on = checked and true or false
    selected.settings.unitBraces, selected.settings.sourceBraces, selected.settings.destBraces = on, on, on
    Edited()
  end
  local spellBraces = SettingCheck(formatting, SPELL_NAMES, SPELL_NAMES_SHOW_BRACES_COMBATLOG_TOOLTIP, "spellBraces", true)
  spellBraces:SetPoint("TOPLEFT", unitBraces, "BOTTOMLEFT", 0, -2)
  local itemBraces = SettingCheck(formatting, ITEM_NAMES, ITEM_NAMES_SHOW_BRACES_COMBATLOG_TOOLTIP, "itemBraces", true)
  itemBraces:SetPoint("TOPLEFT", spellBraces, "BOTTOMLEFT", 0, -2)

  local fullText = SettingCheck(formatting, USE_FULL_TEXT_MODE, FULL_TEXT_COMBATLOG_TOOLTIP, "fullText")
  fullText:SetPoint("TOPLEFT", itemBraces, "BOTTOMLEFT", -15, -13)

  -- Settings: the filter's name and its quick button
  local settings = CreateTab(SETTINGS)

  local SetFilterName
  local nameBox = CreateTextBox(nil, settings)
  nameBox:SetSize(150, 20)
  nameBox:SetMaxLetters(32)
  nameBox:SetPoint("TOPLEFT", settings, "TOPLEFT", 37, -36)
  nameBox:SetScript("OnEnterPressed", function()
    SetFilterName(this:GetText())
    this:ClearFocus()
  end)
  local nameLabel = settings:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
  nameLabel:SetPoint("BOTTOMLEFT", nameBox, "TOPLEFT", 0, 4)
  nameLabel:SetText(FILTER_NAME)

  local saveButton = CreateButton(settings, SAVE, 78)
  saveButton:SetPoint("LEFT", nameBox, "RIGHT", 8, 0)
  saveButton:SetScript("OnClick", function()
    SetFilterName(nameBox:GetText())
    nameBox:ClearFocus()
  end)

  -- CombatConfig_SetFilterName
  function SetFilterName(name)
    if name == "" then
      name = format(DEFAULT_COMBATLOG_FILTER_NAME, #log.GetFilters().filters)
    end
    selected.name = name
    log.UpdateQuickButtons()
    Update()
  end

  local quickButton = CreateCheck(settings, SHOW_QUICK_BUTTON, QUICK_BUTTON_COMBATLOG_TOOLTIP)
  quickButton:SetPoint("TOPLEFT", nameBox, "BOTTOMLEFT", -6, -19)
  quickButton.func = function(self, checked)
    selected.hasQuickButton = checked and true or false
    log.UpdateQuickButtons()
    Update()
  end

  local quickDisplay = {}
  for i, entry in ipairs({ { SOLO, "solo" }, { PARTY, "party" }, { RAID, "raid" } }) do
    local key = entry[2]
    local check = CreateCheck(settings, entry[1], nil, true)
    check:SetPoint("TOPLEFT", quickButton, "BOTTOMLEFT", 20 + (i - 1) * 95, -2)
    check.func = function(self, checked)
      selected.quickButtonDisplay[key] = checked and true or false
      log.UpdateQuickButtons()
    end
    check.key = key
    quickDisplay[i] = check
  end

  --
  -- Refresh (ChatConfig_UpdateCombatSettings)
  --
  local function UpdateColors()
    for _, swatch in ipairs(unitColors.swatches) do
      local color = selected.colors.unitColoring[swatch.flag]
      swatch.prev:SetColorTexture(color.r, color.g, color.b)
    end
    for _, check in ipairs(highlightChecks) do
      SetCheck(check, selected.settings[check.setting])
    end

    local s = selected.settings
    SetCheck(unitNames, s.unitColoring)
    SetCheck(spellNames, s.abilityColoring)
    SetCheck(spellSchool, s.abilityColoring and s.abilitySchoolColoring)
    SetCheckEnabled(spellSchool, s.abilityColoring)
    spellSwatch.prev:SetColorTexture(selected.colors.defaults.spell.r, selected.colors.defaults.spell.g, selected.colors.defaults.spell.b)
    SetCheck(damageNumber, s.amountColoring)
    SetCheck(damageSchool, s.amountSchoolColoring)
    SetCheckEnabled(damageSchool, s.amountColoring)
    damageSwatch.prev:SetColorTexture(selected.colors.defaults.damage.r, selected.colors.defaults.damage.g, selected.colors.defaults.damage.b)
    SetCheck(schoolNames, s.schoolNameColoring)
    SetCheck(entireLine, s.lineColoring)
    SetCheck(bySource, s.lineColorPriority == 1)
    SetCheck(byTarget, s.lineColorPriority ~= 1)
    SetCheckEnabled(bySource, s.lineColoring)
    SetCheckEnabled(byTarget, s.lineColoring)

    UpdateExamples(colors)
  end

  local function UpdateFormatting()
    local s = selected.settings
    SetCheck(timestamp, s.timestamp)
    SetCheck(braces, s.braces)
    SetCheck(unitBraces, s.unitBraces)
    SetCheck(spellBraces, s.spellBraces)
    SetCheck(itemBraces, s.itemBraces)
    SetCheckEnabled(unitBraces, s.braces)
    SetCheckEnabled(spellBraces, s.braces)
    SetCheckEnabled(itemBraces, s.braces)
    SetCheck(fullText, s.fullText)

    UpdateExamples(formatting)
  end

  local function UpdateSettings()
    nameBox:SetText(selected.name or "")
    SetCheck(quickButton, selected.hasQuickButton)
    for _, check in ipairs(quickDisplay) do
      SetCheck(check, selected.quickButtonDisplay[check.key])
      SetCheckEnabled(check, selected.hasQuickButton)
    end
  end

  function Update()
    local filters = log.GetFilters().filters

    local entries = {}
    for i, filter in ipairs(filters) do
      entries[i] = { index = i, name = filter.name }
    end
    filterList.ScrollBox:SetDataProvider(CreateDataProvider(entries), ScrollBoxConstants.RetainScrollPosition)

    -- the last filter can't be deleted
    if #filters > 1 then deleteButton:Enable() else deleteButton:Disable() end
    if #filters < MAX_COMBATLOG_FILTERS then
      addButton:Enable()
      copyButton:Enable()
    else
      addButton:Disable()
      copyButton:Disable()
    end
    if selectedIndex > 1 then moveUp:Enable() else moveUp:Disable() end
    if selectedIndex < #filters then moveDown:Enable() else moveDown:Disable() end

    UpdateCheckboxList(doneBy)
    UpdateCheckboxList(doneTo)
    UpdateTieredList(typesLeft)
    UpdateTieredList(typesRight)
    UpdateTieredList(typesMisc)
    UpdateColors()
    UpdateFormatting()
    UpdateSettings()
  end

  -- ChatConfigFilter_OnClick
  function SelectFilter(index)
    selectedIndex = index
    selected = log.GetFilters().filters[index]
    Update()
  end

  --
  -- Defaults and Okay
  --
  local defaultsButton = CreateButton(frame, COMBATLOG_DEFAULTS, 170)
  defaultsButton:SetPoint("BOTTOMLEFT", frame, "BOTTOMLEFT", 12, 12)
  defaultsButton:SetScript("OnClick", function()
    CreateQuestionDialog(CONFIRM_COMBAT_FILTER_DEFAULTS, function()
      log.Reset()
      changed = nil
      SelectFilter(1)
    end)
  end)

  local okayButton = CreateButton(frame, OKAY, 78)
  okayButton:SetPoint("BOTTOMRIGHT", frame, "BOTTOMRIGHT", -12, 12)
  okayButton:SetScript("OnClick", function()
    PlaySound(SOUNDKIT.GS_TITLE_OPTION_OK)
    frame:Hide()
  end)

  -- opens on the filter the log shows
  frame:SetScript("OnShow", function()
    changed = nil
    SelectFilter(log.GetFilters().currentFilter)
  end)

  frame:SetScript("OnHide", function()
    if changed then
      changed = nil
      log.SelectFilter(log.GetFilters().currentFilter)
    end
  end)

  SelectTab(1)
  return frame
end
