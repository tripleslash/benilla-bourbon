Enum = {
  UICursorType = { Default=0, Item=1, Money=2, Spell=3, PetAction=4, Merchant=5,
    ActionBar=6, Macro=7, Ammo=8, Pet=9, EquipmentSet=12 },
  PowerType = { HealthCost = -2, None = -1, Mana = 0, Rage = 1, Focus = 2, Energy = 3, Happiness = 4 },
  PlayerSwingType = { MainHand = 0, OffHand = 1, Ranged = 2 },
  ItemQuality = { Poor = 0, Common = 1, Uncommon = 2, Rare = 3, Epic = 4, Legendary = 5, Artifact = 6 },
  ItemClass = { Consumable = 0, Container = 1, Weapon = 2, Gem = 3, Armor = 4,
    Reagent = 5, Projectile = 6, Tradegoods = 7, ItemEnhancement = 8, Recipe = 9,
    CurrencyTokenObsolete = 10, Quiver = 11, Questitem = 12, Key = 13,
    PermanentObsolete = 14, Miscellaneous = 15 },
}

-- Slot IDs match the host's 1.12 inventory snapshots (ammo is slot zero).
local slots = { 'AMMO', 'HEAD', 'NECK', 'SHOULDER', 'BODY', 'CHEST', 'WAIST',
  'LEGS', 'FEET', 'WRIST', 'HAND', 'FINGER1', 'FINGER2', 'TRINKET1', 'TRINKET2',
  'BACK', 'MAINHAND', 'OFFHAND', 'RANGED', 'TABARD' }
for index, name in ipairs(slots) do _G['INVSLOT_' .. name] = index - 1 end
INVSLOT_FIRST_EQUIPPED, INVSLOT_LAST_EQUIPPED = 1, 19
INVSLOTS_EQUIPABLE_IN_COMBAT = { [16] = true, [17] = true, [18] = true }
HEARTHSTONE_ITEM_ID = 6948
FIRE_TOTEM_SLOT, EARTH_TOTEM_SLOT, WATER_TOTEM_SLOT, AIR_TOTEM_SLOT, MAX_TOTEMS = 1, 2, 3, 4, 4
