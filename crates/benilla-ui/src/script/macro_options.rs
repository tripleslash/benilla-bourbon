//! Retail macro options (this fork; 1.12.1 has none): `[@unit,cond,…][…] value; [cond] value2;
//! value3`, as 4.1's `SecureCmdOptionParse` (`0x7c65f0` over `EvaluateMacroOptions`) answers them.
//! Clauses are tried in order, a clause's bracket groups are or-ed and a group's conditions and-ed;
//! the first clause that holds gives its value and the group's `@unit` (or `target=unit`), nil
//! without one. An empty group `[]` always holds, and a clause without brackets too. A condition
//! takes `:a/b` alternatives and a `no` prefix negates it; one 1.12 cannot answer (swimming,
//! indoors, …) never holds, negated or not. `flying` and `flyable` are known false, since nothing
//! flies in 1.12, so `noflying` holds.
//!
//! The evaluator is Lua over the stock unit, party, form and modifier verbs, so its relations are
//! the ones `UnitCanAssist` and the rest already answer. `CastSpellByName` reads options through
//! the copy in the registry, which an addon cannot replace; `SecureCmdOptionParse` is the same
//! function, for `/script` lines and macro addons written against the retail name.

use mlua::{Function, Lua, MultiValue, Value};

use super::UiScript;

/// The registry key of the evaluator `CastSpellByName` calls.
const REGISTRY_KEY: &str = "benilla.macro_options";

/// The evaluator: `options -> value, unit` or nothing when no clause holds.
const EVALUATOR: &str = r#"
-- Built over the engine helpers the stock verbs lack: whether the player is mounted, and
-- whether a spell id or name is in the player's or the pet's book.
return function(player_mounted, spell_known)
local find, sub, lower, gsub = string.find, string.sub, string.lower, string.gsub

local function trim(s)
    return (gsub(s, "^%s*(.-)%s*$", "%1"))
end

-- `s` cut at every plain `sep`, empty pieces kept.
local function split(s, sep)
    local out, start = {}, 1
    while true do
        local at = find(s, sep, start, true)
        if not at then
            table.insert(out, sub(s, start))
            return out
        end
        table.insert(out, sub(s, start, at - 1))
        start = at + 1
    end
end

-- Retail's side-specific names read as either side: 1.12's verbs do not tell left from right.
local function modifier(key)
    key = lower(key)
    if key == "shift" or key == "lshift" or key == "rshift" then return IsShiftKeyDown() end
    if key == "ctrl" or key == "lctrl" or key == "rctrl" then return IsControlKeyDown() end
    if key == "alt" or key == "lalt" or key == "ralt" then return IsAltKeyDown() end
end

-- The active shapeshift form's index, 0 in none.
local function form()
    for i = 1, GetNumShapeshiftForms() do
        local _, _, active = GetShapeshiftFormInfo(i)
        if active then return i end
    end
    return 0
end

local function any(args, test)
    for _, a in ipairs(args) do
        if test(trim(a)) then return true end
    end
end

-- The player buffs whose icons are Stealth, Vanish, Prowl/Shadowmeld and the invisibility potions.
local STEALTH_ICONS = {
    "Ability_Stealth", "Ability_Vanish", "Ability_Ambush", "Spell_Magic_LesserInvisibilty",
}
local function stealthed()
    for i = 1, 32 do
        local icon = UnitBuff("player", i)
        if not icon then return false end
        for _, name in ipairs(STEALTH_ICONS) do
            if find(icon, name, 1, true) then return true end
        end
    end
end

-- Whether a worn item is `kind`: its `GetItemInfo` type, subtype or equip slot (`Armor`,
-- `Shields`, `INVTYPE_SHIELD`), without case. An item the cache has not answered yet does not count.
local function equipped(kind)
    kind = lower(kind)
    for slot = 0, 19 do
        local link = GetInventoryItemLink("player", slot)
        local _, _, id = find(link or "", "item:(%d+)")
        id = tonumber(id)
        if id then
            local _, _, _, _, itype, subtype, _, loc = GetItemInfo(id)
            for _, v in ipairs({ itype, subtype, loc }) do
                if type(v) == "string" and lower(v) == kind then return true end
            end
        end
    end
end

local TESTS = {
    help = function(unit) return UnitCanAssist("player", unit) end,
    harm = function(unit) return UnitCanAttack("player", unit) end,
    dead = function(unit) return UnitIsDeadOrGhost(unit) end,
    exists = function(unit) return UnitExists(unit) end,
    party = function(unit) return UnitInParty(unit) end,
    raid = function(unit) return UnitInRaid(unit) or UnitInParty(unit) end,
    combat = function() return UnitAffectingCombat("player") end,
    stealth = function() return stealthed() end,
    mounted = function() return player_mounted() end,
    channeling = function() return CastingBarFrame and CastingBarFrame.channeling end,
    mod = function(_, args)
        if not args then return IsShiftKeyDown() or IsControlKeyDown() or IsAltKeyDown() end
        return any(args, modifier)
    end,
    group = function(_, args)
        local party, raid = GetNumPartyMembers() > 0, GetNumRaidMembers() > 0
        if not args then return party or raid end
        return any(args, function(a)
            a = lower(a)
            return (a == "party" and party) or (a == "raid" and raid)
        end)
    end,
    pet = function(_, args)
        if not UnitExists("pet") then return false end
        if not args then return true end
        local name, family = UnitName("pet"), UnitCreatureFamily("pet")
        return any(args, function(a) return a == name or a == family end)
    end,
    form = function(_, args)
        local f = form()
        if not args then return f > 0 end
        return any(args, function(a) return tonumber(a) == f end)
    end,
    actionbar = function(_, args)
        return args and any(args, function(a) return tonumber(a) == CURRENT_ACTIONBAR_PAGE end)
    end,
    button = function(_, args)
        return not args or any(args, function(a) return a == "1" or a == "LeftButton" end)
    end,
    -- `known:123456` or `known:Spell Name`: in the player's or the pet's book.
    known = function(_, args)
        return args and any(args, function(a) return spell_known(a) end)
    end,
    equipped = function(_, args)
        return args and any(args, equipped)
    end,
    cursor = function() return CursorHasItem() or CursorHasSpell() end,
    bonusbar = function(_, args)
        local offset = GetBonusBarOffset()
        if not args then return offset > 0 end
        return any(args, function(a) return tonumber(a) == offset end)
    end,
    flying = function() return false end,
    flyable = function() return false end,
}
TESTS.modifier, TESTS.stance, TESTS.bar = TESTS.mod, TESTS.form, TESTS.actionbar
TESTS.worn = TESTS.equipped

-- One condition against the group's unit; an unknown one, or one whose verb raises (a token this
-- client has no unit for, `focus`), does not hold.
local function holds(cond, unit)
    local name, args = cond, nil
    local colon = find(cond, ":", 1, true)
    if colon then
        name, args = sub(cond, 1, colon - 1), split(sub(cond, colon + 1), "/")
    end
    name = lower(trim(name))
    local negate = false
    if not TESTS[name] and sub(name, 1, 2) == "no" then
        negate, name = true, sub(name, 3)
    end
    local test = TESTS[name]
    if not test then return false end
    local ok, answer = pcall(test, unit, args)
    if not ok then return false end
    if negate then return not answer end
    return answer and true or false
end

-- The unit of the unit frame under the cursor, read off its `unit` field (`UnitFrame_Initialize`
-- sets it, and on its bars), climbing to the parent; nil over the world or a frame without one.
local function frame_unit()
    local f = GetMouseFocus and GetMouseFocus()
    while type(f) == "table" do
        local unit = rawget(f, "unit")
        if type(unit) == "string" and UnitExists(unit) then return unit end
        f = f.GetParent and f:GetParent()
    end
end

-- One bracket group's verdict and its unit, nil without an `@`.
local function group_holds(body)
    local unit, conds = nil, {}
    for _, raw in ipairs(split(body, ",")) do
        local c = trim(raw)
        if sub(c, 1, 1) == "@" then
            unit = trim(sub(c, 2))
        elseif lower(sub(c, 1, 7)) == "target=" then
            unit = trim(sub(c, 8))
        elseif c ~= "" then
            table.insert(conds, c)
        end
    end
    -- Retail's `mouseover` names a hovered unit frame's unit too; 1.12's names world units only.
    if unit and lower(unit) == "mouseover" then
        local ok, hovered = pcall(frame_unit)
        if ok and hovered then unit = hovered end
    end
    for _, c in ipairs(conds) do
        if not holds(c, unit or "target") then return false end
    end
    return true, unit
end

return function(options)
    if type(options) ~= "string" then
        error('Usage: SecureCmdOptionParse("options")', 2)
    end
    for _, raw in ipairs(split(options, ";")) do
        local clause = trim(raw)
        local groups, matched, unit = 0, false, nil
        while sub(clause, 1, 1) == "[" do
            local close = find(clause, "]", 2, true)
            if not close then break end
            groups = groups + 1
            if not matched then
                matched, unit = group_holds(sub(clause, 2, close - 1))
            end
            clause = trim(sub(clause, close + 1))
        end
        if groups == 0 or matched then
            return clause, unit
        end
    end
end
end
"#;

/// A macro's `#show` or `#showtooltip` line (4.1's macro node, `memcmp(line, "#show", 5)`): the
/// button shows `arg`'s spell, or the first cast line's without one, and `#showtooltip` its
/// tooltip too; `#show none` shows nothing. The macro runner never sends such a line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShowDirective<'a> {
    pub tooltip: bool,
    pub arg: Option<&'a str>,
}

impl ShowDirective<'_> {
    /// `#show none`: the button keeps the macro's own icon and name.
    pub fn is_none(&self) -> bool {
        self.arg.is_some_and(|a| a.eq_ignore_ascii_case("none"))
    }
}

/// The body's first `#show` line, if any.
pub fn show_directive(body: &str) -> Option<ShowDirective<'_>> {
    body.split(['\r', '\n']).find_map(|line| {
        let rest = line.trim().strip_prefix("#show")?;
        let (tooltip, rest) = match rest.strip_prefix("tooltip") {
            Some(r) => (true, r),
            None => (false, rest),
        };
        if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
            return None;
        }
        let arg = rest.trim();
        Some(ShowDirective {
            tooltip,
            arg: (!arg.is_empty()).then_some(arg),
        })
    })
}

/// The spell a `#show` macro's button shows, with whether `#showtooltip` asks for its tooltip:
/// the macro's bound spell, which the app derives from the `#show` argument or the cast line.
pub(super) fn shown_spell(model: &super::Model, macro_id: u32) -> Option<(u32, bool)> {
    let body = &model.macros.by_id(macro_id)?.body;
    let show = show_directive(body).filter(|s| !s.is_none())?;
    let spell = model.macro_bindings.get(&macro_id)?.spell;
    u32::try_from(spell)
        .ok()
        .filter(|&s| s != 0)
        .map(|s| (s, show.tooltip))
}

/// Whether `text` uses option syntax: a leading bracket group or a second clause.
pub fn has_options(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with('[') || t.contains(';')
}

/// Build the evaluator, keep it in the registry and publish it as `SecureCmdOptionParse`.
pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    // `[mounted]`: the player unit's mount display, which no 1.12 verb answers.
    let mounted = lua.create_function(|lua, ()| {
        Ok(lua
            .app_data_ref::<crate::script::ModelCell>()
            .is_some_and(|m| m.units_by_lower.get("player").is_some_and(|u| u.mounted)))
    })?;
    // `[known:…]`: 1.12 names no spell id to Lua, so the books answer.
    let known = lua.create_function(|lua, arg: String| {
        Ok(lua
            .app_data_ref::<crate::script::ModelCell>()
            .is_some_and(|m| spell_known(&m, &arg)))
    })?;
    let evaluator: Function = lua
        .load(EVALUATOR)
        .set_name("=macro_options")
        .eval::<Function>()?
        .call((mounted, known))?;
    lua.set_named_registry_value(REGISTRY_KEY, evaluator.clone())?;
    lua.globals().set("SecureCmdOptionParse", evaluator)?;
    Ok(())
}

/// Whether `arg`, a spell id or a name (a rank in parentheses pins it, as `CastSpellByName`
/// reads one), is in the player's book or the pet's.
fn spell_known(model: &super::Model, arg: &str) -> bool {
    let arg = arg.trim();
    let pet = &model.pet_book.slots;
    match arg.parse::<u32>() {
        Ok(id) => model
            .spellbook
            .slots
            .iter()
            .chain(pet)
            .any(|s| s.spell_id == id),
        // Both books by `CastSpellByName`'s own search (`0x4b3a10`).
        Err(_) => super::resolve_spell_in_books(&model.spellbook, &model.pet_book, arg).is_some(),
    }
}

/// The value and unit `options` selects, `None` when no clause holds; text without option syntax
/// is its own value.
pub(super) fn evaluate(lua: &Lua, options: &str) -> mlua::Result<Option<(String, Option<String>)>> {
    if !has_options(options) {
        return Ok(Some((options.trim().to_string(), None)));
    }
    let evaluator: Function = lua.named_registry_value(REGISTRY_KEY)?;
    let out: MultiValue = evaluator.call(options)?;
    let mut out = out.into_iter();
    let value = match out.next() {
        Some(Value::String(s)) => s.to_str()?.to_string(),
        _ => return Ok(None),
    };
    let unit = match out.next() {
        Some(Value::String(s)) => Some(s.to_str()?.to_string()).filter(|u| !u.is_empty()),
        _ => None,
    };
    Ok(Some((value, unit)))
}

impl UiScript {
    /// [`evaluate`] for the app: the spell a conditional macro line casts as the world stands now.
    /// A raising evaluator answers `None`.
    pub fn macro_option_value(&self, options: &str) -> Option<(String, Option<String>)> {
        evaluate(self.lua(), options).ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A VM with the evaluator over stub verbs: `UNITS` names who exists and how they relate.
    fn vm() -> Lua {
        let lua = Lua::new();
        lua.load(
            r#"
            UNITS = { target = "foe", player = "self" }
            MODS = {}
            function UnitExists(u) if u == "focus" then error("Unknown unit name: focus") end return UNITS[u] ~= nil end
            function UnitCanAssist(_, u) return UNITS[u] == "friend" or UNITS[u] == "self" or UNITS[u] == "deadfriend" end
            function UnitCanAttack(_, u) return UNITS[u] == "foe" end
            function UnitIsDeadOrGhost(u) return UNITS[u] == "deadfriend" end
            function UnitInParty(u) return UNITS[u] == "friend" end
            function UnitInRaid(u) return nil end
            function UnitAffectingCombat() return IN_COMBAT end
            function UnitBuff() return nil end
            function UnitName() return nil end
            function UnitCreatureFamily() return nil end
            function IsShiftKeyDown() return MODS.shift end
            function IsControlKeyDown() return MODS.ctrl end
            function IsAltKeyDown() return MODS.alt end
            function GetNumPartyMembers() return 0 end
            function GetNumRaidMembers() return 0 end
            function GetNumShapeshiftForms() return 0 end
            CURRENT_ACTIONBAR_PAGE = 1
            "#,
        )
        .exec()
        .unwrap();
        install(&lua).unwrap();
        lua
    }

    fn eval(lua: &Lua, s: &str) -> Option<(String, Option<String>)> {
        evaluate(lua, s).unwrap()
    }

    fn some(value: &str, unit: Option<&str>) -> Option<(String, Option<String>)> {
        Some((value.to_string(), unit.map(str::to_string)))
    }

    #[test]
    fn the_mouseover_heal_takes_a_live_friend_and_falls_back_to_the_empty_group() {
        let lua = vm();
        let hop = "[@mouseover,help,nodead][] Hand of Protection";
        // Nobody under the cursor: the empty group holds, with no unit.
        assert_eq!(eval(&lua, hop), some("Hand of Protection", None));
        lua.load(r#"UNITS.mouseover = "friend""#).exec().unwrap();
        assert_eq!(
            eval(&lua, hop),
            some("Hand of Protection", Some("mouseover"))
        );
        lua.load(r#"UNITS.mouseover = "deadfriend""#)
            .exec()
            .unwrap();
        assert_eq!(eval(&lua, hop), some("Hand of Protection", None));
        lua.load(r#"UNITS.mouseover = "foe""#).exec().unwrap();
        assert_eq!(eval(&lua, hop), some("Hand of Protection", None));
    }

    /// Over a party frame the world names no mouseover, so `@mouseover` takes the hovered frame's
    /// `unit`, from the frame or the nearest parent that carries one.
    #[test]
    fn mouseover_names_the_hovered_unit_frames_unit() {
        let lua = vm();
        let hop = "[@mouseover,help,nodead][] Hand of Protection";
        lua.load(
            r#"
            UNITS.party1 = "friend"
            PartyFrame = { unit = "party1" }
            HealthBar = { GetParent = function() return PartyFrame end }
            function GetMouseFocus() return FOCUS end
            "#,
        )
        .exec()
        .unwrap();
        lua.load("FOCUS = PartyFrame").exec().unwrap();
        assert_eq!(eval(&lua, hop), some("Hand of Protection", Some("party1")));
        lua.load("FOCUS = HealthBar").exec().unwrap();
        assert_eq!(eval(&lua, hop), some("Hand of Protection", Some("party1")));
        lua.load(r#"UNITS.party1 = "deadfriend""#).exec().unwrap();
        assert_eq!(eval(&lua, hop), some("Hand of Protection", None));
        // A frame naming nobody, or none at all, leaves the world's mouseover.
        lua.load(r#"UNITS.party1 = nil UNITS.mouseover = "friend""#)
            .exec()
            .unwrap();
        assert_eq!(
            eval(&lua, hop),
            some("Hand of Protection", Some("mouseover"))
        );
        lua.load("FOCUS = nil").exec().unwrap();
        assert_eq!(
            eval(&lua, hop),
            some("Hand of Protection", Some("mouseover"))
        );
    }

    #[test]
    fn clauses_are_tried_in_order_and_a_bare_clause_always_holds() {
        let lua = vm();
        let s = "[help] Flash Heal; [harm] Smite; Renew";
        assert_eq!(eval(&lua, s), some("Smite", None));
        lua.load(r#"UNITS.target = "friend""#).exec().unwrap();
        assert_eq!(eval(&lua, s), some("Flash Heal", None));
        lua.load(r#"UNITS.target = nil"#).exec().unwrap();
        assert_eq!(eval(&lua, s), some("Renew", None));
        // No clause holds: nothing.
        assert_eq!(eval(&lua, "[help] Flash Heal; [harm] Smite"), None);
    }

    #[test]
    fn modifiers_negation_and_target_equals() {
        let lua = vm();
        let s = "[mod:shift/alt,target=player] Heal; [nomod] Smite";
        assert_eq!(eval(&lua, s), some("Smite", None));
        lua.load("MODS.alt = 1").exec().unwrap();
        assert_eq!(eval(&lua, s), some("Heal", Some("player")));
        lua.load("MODS.alt = nil MODS.ctrl = 1").exec().unwrap();
        assert_eq!(eval(&lua, s), None);
        lua.load("IN_COMBAT = 1").exec().unwrap();
        assert_eq!(
            eval(&lua, "[nocombat] Drink; [combat] Fade"),
            some("Fade", None)
        );
    }

    #[test]
    fn an_unknown_condition_or_a_raising_unit_never_holds() {
        let lua = vm();
        assert_eq!(eval(&lua, "[swimming] A; B"), some("B", None));
        assert_eq!(eval(&lua, "[@focus,exists] A; B"), some("B", None));
        // Without option syntax the text is its own value, untouched by the evaluator.
        assert_eq!(
            eval(&lua, "  Fireball(Rank 1) "),
            some("Fireball(Rank 1)", None)
        );
        assert!(!has_options("Fireball"));
        assert!(has_options(" [] Fireball"));
        assert!(has_options("Fireball; Frostbolt"));
    }

    #[test]
    fn mouseover_then_target_heal_and_the_state_conditions() {
        let lua = vm();
        let heal = "[@mouseover,help] Flash Heal; [@target,help] Flash Heal";
        assert_eq!(eval(&lua, heal), None, "a hostile target and no mouseover");
        lua.load(r#"UNITS.target = "friend""#).exec().unwrap();
        assert_eq!(eval(&lua, heal), some("Flash Heal", Some("target")));
        lua.load(r#"UNITS.mouseover = "friend""#).exec().unwrap();
        assert_eq!(eval(&lua, heal), some("Flash Heal", Some("mouseover")));

        lua.load(r#"UNITS.target = "deadfriend""#).exec().unwrap();
        assert_eq!(
            eval(&lua, "[dead] Resurrection; Heal"),
            some("Resurrection", None)
        );
        assert_eq!(eval(&lua, "[harm] Smite; [help] Heal"), some("Heal", None));

        assert_eq!(
            eval(&lua, "[stealth] Ambush; Backstab"),
            some("Backstab", None)
        );
        lua.load(
            r#"function UnitBuff(_, i) if i == 1 then return "Interface\\Icons\\Ability_Stealth" end end"#,
        )
        .exec()
        .unwrap();
        assert_eq!(
            eval(&lua, "[stealth] Ambush; Backstab"),
            some("Ambush", None)
        );

        assert_eq!(eval(&lua, "[mod:shift] A; B"), some("B", None));
        lua.load("MODS.shift = 1").exec().unwrap();
        assert_eq!(eval(&lua, "[mod:shift] A; B"), some("A", None));
        // No VM model here: never mounted.
        assert_eq!(
            eval(&lua, "[mounted] Dismount; [nomounted] Mount"),
            some("Mount", None)
        );
    }

    /// `[mounted]` reads the player unit's mount display through the engine model.
    #[test]
    fn mounted_reads_the_player_unit() {
        let mut s = UiScript::new().unwrap();
        let options = "[mounted] Dismount; Mount";
        assert_eq!(s.macro_option_value(options), some("Mount", None));
        s.set_unit(
            "player",
            Some(super::super::UnitState {
                exists: true,
                mounted: true,
                ..Default::default()
            }),
        );
        assert_eq!(s.macro_option_value(options), some("Dismount", None));
    }

    #[test]
    fn the_global_is_the_retail_function() {
        let lua = vm();
        let (value, unit): (String, Option<String>) = lua
            .load(r#"return SecureCmdOptionParse("[@player] Renew")"#)
            .eval()
            .unwrap();
        assert_eq!((value.as_str(), unit.as_deref()), ("Renew", Some("player")));
    }

    /// [`vm`] with an engine model whose book holds `known` (spell 123456 "Premium") and the
    /// verbs the state conditions read: three forms, one worn shield, the cursor, the bonus bar.
    fn vm_with_state(known: bool) -> Lua {
        let lua = vm();
        let mut model = super::super::Model::new();
        if known {
            model.spellbook.slots.push(super::super::SpellSlotView {
                spell_id: 123456,
                name: "Premium".into(),
                ..Default::default()
            });
        }
        lua.set_app_data(crate::script::ModelCell::new(model));
        lua.load(
            r#"
            FORM = 0
            function GetNumShapeshiftForms() return 3 end
            function GetShapeshiftFormInfo(i) return "icon", "form" .. i, FORM == i, 1 end
            function GetInventoryItemLink(_, slot)
                if slot == 17 then return "|cffffffff|Hitem:2129:0:0:0|h[Large Round Shield]|h|r" end
            end
            function GetItemInfo(id)
                if id == 2129 then
                    return "Large Round Shield", "item:2129", 1, 1, "Armor", "Shields", 1, "INVTYPE_SHIELD"
                end
            end
            function CursorHasItem() return CURSOR_ITEM end
            function CursorHasSpell() return nil end
            function GetBonusBarOffset() return BONUS or 0 end
            "#,
        )
        .exec()
        .unwrap();
        lua
    }

    /// The heal/damage macro, clause by clause: shift heals the player, a live friendly mouseover
    /// is healed, a known spell takes a hostile mouseover or target before the standard one, and
    /// the target's target, or nobody, is healed last.
    #[test]
    fn the_heal_or_damage_macro_takes_each_clause_in_turn() {
        let m = "[mod:shift,@player][@mouseover,help,nodead] Heilung; \
                 [known:123456,@mouseover,harm,nodead][known:123456,harm] PremiumSchaden; \
                 [@mouseover,harm,nodead][harm] StandardSchaden; \
                 [@targettarget,help,nodead][] Heilung";
        for known in [true, false] {
            let lua = vm_with_state(known);
            let damage = if known {
                "PremiumSchaden"
            } else {
                "StandardSchaden"
            };
            let run = |setup: &str| {
                lua.load(setup).exec().unwrap();
                eval(&lua, m)
            };
            // A hostile target: the damage clause, at the selection.
            assert_eq!(run(""), some(damage, None), "known {known}");
            // Shift wins over everything, on the player.
            assert_eq!(run("MODS.shift = 1"), some("Heilung", Some("player")));
            lua.load("MODS.shift = nil").exec().unwrap();
            // A live friendly mouseover is healed; a dead one falls through to the damage.
            assert_eq!(
                run(r#"UNITS.mouseover = "friend""#),
                some("Heilung", Some("mouseover"))
            );
            assert_eq!(run(r#"UNITS.mouseover = "deadfriend""#), some(damage, None));
            // A hostile mouseover takes the damage at the mouseover.
            assert_eq!(
                run(r#"UNITS.mouseover = "foe""#),
                some(damage, Some("mouseover"))
            );
            // A friendly target with a living friend as its target: the last clause's unit.
            assert_eq!(
                run(
                    r#"UNITS.mouseover = nil UNITS.target = "friend" UNITS.targettarget = "friend""#
                ),
                some("Heilung", Some("targettarget"))
            );
            // Nobody at all: the empty group heals with no unit.
            assert_eq!(
                run("UNITS.target = nil UNITS.targettarget = nil"),
                some("Heilung", None)
            );
        }
    }

    /// `stance`/`form` read the active shapeshift form, `0` and `nostance` none; `known` takes an
    /// id or a name; `equipped` a worn item's type, subtype or slot; `noflying` holds.
    #[test]
    fn stance_known_equipped_cursor_and_the_never_flying_conditions() {
        let lua = vm_with_state(true);
        let stance = "[stance:1] Battle; [stance:2/3] Other; [nostance] None";
        assert_eq!(eval(&lua, stance), some("None", None));
        assert_eq!(eval(&lua, "[stance:0] Caster; X"), some("Caster", None));
        lua.load("FORM = 1").exec().unwrap();
        assert_eq!(eval(&lua, stance), some("Battle", None));
        lua.load("FORM = 3").exec().unwrap();
        assert_eq!(eval(&lua, stance), some("Other", None));
        assert_eq!(eval(&lua, "[form] In; Out"), some("In", None));
        assert_eq!(eval(&lua, "[form:1/2] In; Out"), some("Out", None));

        assert_eq!(eval(&lua, "[known:Premium] A; B"), some("A", None));
        assert_eq!(eval(&lua, "[known:999] A; B"), some("B", None));
        assert_eq!(eval(&lua, "[noknown:123456] A; B"), some("B", None));

        assert_eq!(eval(&lua, "[equipped:Shields] Bash; X"), some("Bash", None));
        assert_eq!(
            eval(&lua, "[worn:INVTYPE_SHIELD] Bash; X"),
            some("Bash", None)
        );
        assert_eq!(eval(&lua, "[equipped:Daggers] Stab; X"), some("X", None));

        assert_eq!(eval(&lua, "[cursor] Drop; X"), some("X", None));
        lua.load("CURSOR_ITEM = 1 BONUS = 2").exec().unwrap();
        assert_eq!(eval(&lua, "[cursor] Drop; X"), some("Drop", None));
        assert_eq!(eval(&lua, "[bonusbar:2] Prowl; X"), some("Prowl", None));

        assert_eq!(eval(&lua, "[flying] A; [noflying] B"), some("B", None));
        assert_eq!(eval(&lua, "[mod:lshift] A; B"), some("B", None));
        lua.load("MODS.shift = 1").exec().unwrap();
        assert_eq!(eval(&lua, "[mod:rshift] A; B"), some("A", None));
    }
}
