//! Opt-in building blocks for the ClassicAPI edition of pfUI.
//!
//! This is an extension, not a claim of ClassicAPI parity. The caller explicitly installs it
//! on each in-game VM, before loading third-party addons. It publishes the ClassicAPI version
//! pfUI's load gate requires, beside a table that marks the surface as emulated and partial.
//! The headless audit is separate from the client and does not install or modify addons.

mod attributes;
pub mod bags;
mod baselib;
pub mod clock;
pub mod combat_log;
pub mod creature;
pub mod display;
mod encoding;
pub mod icons;
pub mod loot;
mod lua51;
pub mod maps;
#[cfg(feature = "nampower")]
pub mod nampower;
pub mod player_cache;
pub mod resurrect;
pub mod spell;
pub mod swing;
pub mod swing_range;
pub mod trade_link;
mod widgets;

#[cfg(feature = "audit")]
pub mod audit;

use benilla_ui::script::UiScript;
use mlua::Value;
pub use mlua::{Function as LuaFunction, Lua};

const INSTALLED: &str = "benilla.pfui.installed";

/// Install the independently implemented compatibility subset, once per VM. Default clients
/// never call this. No dummy engine API is published; `BenillaClassicAPICompatibility` states
/// that the surface behind `CLASSIC_API_VERSION` is partial.
pub fn install(script: &UiScript) -> mlua::Result<()> {
    let lua = script.lua();
    if lua.named_registry_value::<Option<bool>>(INSTALLED)? == Some(true) {
        return Ok(());
    }
    for name in [
        "C_Timer",
        "C_EncodingUtil",
        "C_ColorUtil",
        "Mixin",
        "SafePack",
        "GameTooltip_Hide",
        "tInvert",
        "CreateFromMixins",
        "CreateAndInitFromMixin",
        "IconDataProviderMixin",
        "IconDataProviderExtraType",
        "IconDataProviderIconType",
        "IconDataProvider_GetAllIconTypes",
        "GetMacroIcons",
        "GetMacroItemIcons",
        "GetLooseMacroIcons",
        "GetLooseMacroItemIcons",
        "GetCursorInfo",
        "INVSLOTS_EQUIPABLE_IN_COMBAT",
        "HEARTHSTONE_ITEM_ID",
        "C_LootHistory",
        "C_ChatBubbles",
        "C_Map",
        "UnitCreatureTypeID",
        "UnitCreatedBySpell",
        "ColorMixin",
        "CreateColor",
        "CallbackRegistryMixin",
        "EventUtil",
        "EventRegistry",
        "RunNextFrame",
        "hooksecurefunc",
        "UnitClassBase",
        "Enum",
        "C_DateAndTime",
        "UnitGUID",
        "UnitTokenFromGUID",
        "C_Item",
        "C_Container",
        "C_PlayerInfo",
        "C_AddOns",
        "C_FriendList",
        "C_UnitAuras",
        "C_PlayerCache",
        "C_Macro",
        "C_Reputation",
        "C_QuestLog",
        "C_EquipmentSet",
        "IsPlayerGuid",
        "C_SpellBook",
        "UnitPower",
        "UnitPowerMax",
        "FocusUnit",
        "ClearFocus",
        "IsInGroup",
        "IsInRaid",
        "UnitSubName",
        "C_NamePlate",
        "C_NewItems",
        "C_MerchantFrame",
        "IsSpellKnown",
        "UnitIsFeignDeath",
        "UnitIsMinion",
        "GetUnitField",
        "GetPlayerGuid",
        "GetNampowerVersion",
        "QueueSpellByName",
        "QueueScript",
        "CastSpellByNameNoQueue",
        "CastSpellNoQueue",
        "GetCastInfo",
        "GetWowTimeMs",
        "GetPlayerAuraDuration",
        "CancelPlayerAuraSpellId",
        "BenillaNampowerCompatibility",
        "GetTotemInfo",
        "GetShapeshiftFormID",
        "GetSpellInfo",
        "GetSpellBonusDamage",
        "UnitSpellHaste",
        "IsMounted",
        "IsStealthed",
        "Dismount",
        "CancelShapeshiftForm",
        "TargetTotem",
        "GetActionInfo",
        "GetInventoryItemID",
        "GetInventoryItemDurability",
        "ClassicAPI_ToggleUnitMenu",
        "Item",
        "ItemLocation",
        "Round",
        "Clamp",
        "CopyTable",
        "CreateObjectPool",
        "CreateTexturePool",
        "IsModifierKeyDown",
        "strtrim",
        "GetLootRollItemID",
        "GetLootSlotItemID",
        "GetMerchantItemID",
        "GetQuestItemID",
        "GetQuestLogItemID",
        "GetTradeSkillItemID",
        "GetTradeSkillReagentItemID",
        "GetCraftReagentItemID",
        "GetMacroSpell",
        "GetNumTrackingTypes",
        "GetTrackingInfo",
        "SetTracking",
        "FindSpellBookSlotByID",
        "GetFactionInfoByID",
        "UnitNameFromGUID",
        "GetPlayerInfoByGUID",
        "GetInventoryItemsForSlot",
        "EquipmentManager_UnpackLocation",
        "EquipmentManager_GetLocationData",
        "INTERFACE_VERSION",
        "CLASSIC_API_VERSION",
        "BenillaClassicAPICompatibility",
        "GetUnitSpeed",
        "GetInventoryItemRepairCost",
        "GetInboxItemLink",
        "GetInboxItemID",
        "GetSendMailItemLink",
        "CreateSimpleTextureMarkup",
        "GetCoinTextureString",
        "CLASS_SORT_ORDER",
        "securecall",
        "securecallfunction",
        "secureexecuterange",
        "issecure",
        "IsPublicBuild",
        "strsplit",
        "strsplittable",
        "GetTickTime",
        "GetMouseFoci",
        "FillLocalizedClassList",
        "AddSourceLocationExclude",
        "IsLoggedIn",
        "C_EventUtils",
        "C_UIColor",
        "C_CVar",
        "C_Texture",
        "BenillaCooldownMixin",
        "CombatLogGetCurrentEventInfo",
        "IsPlayerSpell",
        "IsChatAFK",
        "C_SwingTimer",
        "C_LossOfControl",
        "CopyToClipboard",
        "UnitHasIncomingResurrection",
        "IsUnitToken",
        "UnitTokenFromName",
        "CastSpellNoToggle",
        "StartAttack",
        "StopAttack",
        "GetClickFrame",
        "CancelSpellByName",
        "GetTradeSkillListLink",
        "GetCraftListLink",
        "IsTradeSkillLinked",
        "IsCraftLinked",
        "ClassicAPI",
        "GetPhysicalScreenSize",
        "IsModifiedClick",
        "GetModifiedClick",
        "SetModifiedClick",
        "LE_ITEM_QUALITY_POOR",
    ] {
        if !lua.globals().get::<Value>(name)?.is_nil() {
            return Err(mlua::Error::runtime(format!(
                "pfUI compatibility cannot replace existing {name}"
            )));
        }
    }
    lua51::install(lua)?;
    encoding::install(lua)?;
    lua.load(include_str!("utilities.lua"))
        .set_name("@benilla-pfui/utilities.lua")
        .exec()?;
    let globals = lua.create_table()?;
    display::install(lua, &globals)?;
    lua.load(include_str!("globals.lua"))
        .set_name("@benilla-pfui/globals.lua")
        .call::<()>(globals.clone())?;
    lua.set_named_registry_value("benilla.pfui.globals_bridge", globals)?;
    lua.load(include_str!("strings.lua"))
        .set_name("@benilla-pfui/strings.lua")
        .exec()?;
    lua.load(include_str!("event_registry.lua"))
        .set_name("@benilla-pfui/event_registry.lua")
        .exec()?;
    lua.load(include_str!("helpers.lua"))
        .set_name("@benilla-pfui/helpers.lua")
        .exec()?;
    let timer_pass = lua
        .load(include_str!("timers.lua"))
        .set_name("@benilla-pfui/timers.lua")
        .call::<mlua::Function>(lua.create_function(|lua, message: String| {
            benilla_ui::script::pfui_host::report_error(lua, message);
            Ok(())
        })?)?;
    lua.set_named_registry_value(benilla_ui::script::pfui_host::TIMER_PASS, timer_pass)?;
    widgets::install(lua)?;
    baselib::install(lua)?;
    benilla_ui::script::pfui_host::install(lua)?;
    let engine = lua.create_table()?;
    engine.set(
        "class_names",
        lua.create_function(|lua, ()| {
            lua.named_registry_value::<Option<mlua::Table>>(CLASS_NAMES)?
                .map_or_else(|| lua.create_table(), Ok)
        })?,
    )?;
    // `fire(event, ...)`: plain values ride the held-raise path; a table payload is dispatched
    // with its Lua values.
    engine.set(
        "fire",
        lua.create_function(|lua, (event, args): (String, mlua::Variadic<Value>)| {
            use benilla_ui::script::{pfui_host, ScriptValue};
            if args.iter().any(|v| matches!(v, Value::Table(_))) {
                pfui_host::fire_pfui_values(lua, &event, args.into_iter().collect());
                return Ok(());
            }
            let args = args
                .into_iter()
                .map(|v| match v {
                    Value::Boolean(b) => ScriptValue::Bool(b),
                    Value::Integer(i) => ScriptValue::Int(i),
                    Value::Number(n) => ScriptValue::Number(n),
                    Value::String(s) => ScriptValue::Str(s.to_string_lossy()),
                    _ => ScriptValue::Nil,
                })
                .collect();
            pfui_host::fire_pfui_event(lua, &event, args);
            Ok(())
        })?,
    )?;
    engine.set(
        "clipboard",
        lua.create_function(|lua, text: String| {
            lua.named_registry_value::<mlua::Table>(CLIPBOARD)?
                .set(1, text)
        })?,
    )?;
    lua.set_named_registry_value(CLIPBOARD, lua.create_table()?)?;
    engine.set(
        "school_lockout",
        lua.create_function(|lua, ()| {
            lua.named_registry_value::<Option<mlua::Table>>(SCHOOL_LOCKOUT)
        })?,
    )?;
    engine.set(
        "atlas_registered",
        lua.create_function(
            |lua,
             (name, file, width, height, left, right, top, bottom): (
                String,
                String,
                f32,
                f32,
                f32,
                f32,
                f32,
                f32,
            )| {
                benilla_ui::script::pfui_host::inline_texture::register_atlas(
                    lua,
                    &name,
                    benilla_ui::script::pfui_host::inline_texture::AtlasPiece {
                        file,
                        width,
                        height,
                        coords: [left, right, top, bottom],
                    },
                );
                Ok(())
            },
        )?,
    )?;
    engine.set(
        "item_use_spell",
        lua.create_function(|lua, item: mlua::Value| {
            benilla_ui::script::pfui_host::items::use_spell(lua, item)
        })?,
    )?;
    engine.set(
        "ui_colors",
        lua.load(include_str!("ui_colors.lua"))
            .set_name("@benilla-pfui/ui_colors.lua")
            .eval::<mlua::Table>()?,
    )?;
    engine.set(
        "display",
        lua.named_registry_value::<mlua::Table>("benilla.pfui.globals_bridge")?,
    )?;
    engine.set(
        "textures",
        lua.named_registry_value::<mlua::Table>("__benilla_texture_methods")?,
    )?;
    let tables = lua.create_table()?;
    for (key, registry) in [
        ("frame", "__benilla_frame_methods"),
        ("button", "__benilla_button_methods"),
        ("editbox", "__benilla_editbox_methods"),
        ("slider", "__benilla_slider_methods"),
        ("statusbar", "__benilla_statusbar_methods"),
        ("texture", "__benilla_texture_methods"),
        ("fontstring", "__benilla_fontstring_methods"),
        ("model", "__benilla_model_methods"),
        ("title", "__benilla_title_methods"),
        ("region", "__benilla_region_methods"),
        (
            "scrollingmessageframe",
            "__benilla_scrollingmessageframe_methods",
        ),
    ] {
        tables.set(key, lua.named_registry_value::<mlua::Table>(registry)?)?;
    }
    let tile_pass = lua
        .load(include_str!("modern_widgets.lua"))
        .set_name("@benilla-pfui/modern_widgets.lua")
        .call::<mlua::Function>((
            tables.clone(),
            lua.named_registry_value::<mlua::Table>(
                benilla_ui::script::pfui_host::tiling::TILE_ENGINE,
            )?,
        ))?;
    lua.load(include_str!("animations.lua"))
        .set_name("@benilla-pfui/animations.lua")
        .call::<()>(tables.clone())?;
    // Duration objects, text bindings and timer bars; their per-frame pass runs after C_Timer's.
    let duration_pass = lua
        .load(include_str!("durations.lua"))
        .set_name("@benilla-pfui/durations.lua")
        .call::<mlua::Function>((
            tables,
            lua.named_registry_value::<mlua::Table>(
                benilla_ui::script::pfui_host::durations::DURATION_ENGINE,
            )?,
        ))?;
    let timer_pass: mlua::Function =
        lua.named_registry_value(benilla_ui::script::pfui_host::TIMER_PASS)?;
    lua.set_named_registry_value(
        benilla_ui::script::pfui_host::TIMER_PASS,
        lua.create_function(move |_, ()| {
            let timers = timer_pass.call::<()>(());
            let durations = duration_pass.call::<()>(());
            // Tiled textures follow the sizes the fills above just set.
            tile_pass.call::<()>(()).and(durations).and(timers)
        })?,
    )?;
    swing_range::install(lua)?;
    combat_log::install(lua)?;
    resurrect::install(lua)?;
    lua.load(include_str!("cooldown.lua"))
        .set_name("@benilla-pfui/cooldown.lua")
        .exec()?;
    let intrinsics = benilla_ui::framexml::parse(INTRINSICS)
        .map_err(|e| mlua::Error::runtime(format!("benilla-pfui intrinsics: {e:?}")))?;
    let report = benilla_ui::loader::load_into(lua, &intrinsics, "", &|_| None);
    if let Some(error) = report.errors.first() {
        return Err(mlua::Error::runtime(format!(
            "benilla-pfui intrinsics: {error}"
        )));
    }
    lua.load(include_str!("engine.lua"))
        .set_name("@benilla-pfui/engine.lua")
        .call::<()>(engine.clone())?;
    lua.load(include_str!("atlases.lua"))
        .set_name("@benilla-pfui/atlases.lua")
        .exec()?;
    icons::install(lua)?;
    loot::install(lua)?;
    maps::Catalog::default().install(lua)?;
    lua.load(include_str!("merchant.lua"))
        .set_name("@benilla-pfui/merchant.lua")
        .call::<()>(lua.named_registry_value::<mlua::Table>("benilla.pfui.merchant_bridge")?)?;
    lua.load(include_str!("social.lua"))
        .set_name("@benilla-pfui/social.lua")
        .call::<()>(lua.named_registry_value::<mlua::Table>("benilla.pfui.social_bridge")?)?;
    lua.load(include_str!("autocomplete.lua"))
        .set_name("@benilla-pfui/autocomplete.lua")
        .exec()?;
    lua.load(include_str!("bagsort.lua"))
        .set_name("@benilla-pfui/bagsort.lua")
        .call::<()>(lua.named_registry_value::<mlua::Table>(
            benilla_ui::script::pfui_host::items::BAG_ENGINE,
        )?)?;
    attributes::install(lua)?;
    #[cfg(feature = "nampower")]
    nampower::install(lua)?;
    lua.set_named_registry_value("benilla.pfui.addon_privates", lua.create_table()?)?;
    lua.load("function UnitClassBase(unit) local _, token = UnitClass(unit); return token end")
        .set_name("@benilla-pfui/units.lua")
        .exec()?;
    lua.load(include_str!("enums.lua"))
        .set_name("@benilla-pfui/enums.lua")
        .exec()?;
    clock::install(lua)?;
    lua.load(include_str!("items.lua"))
        .set_name("@benilla-pfui/items.lua")
        .exec()?;
    lua.load(include_str!("equipment.lua"))
        .set_name("@benilla-pfui/equipment.lua")
        .call::<()>(lua.named_registry_value::<mlua::Table>("benilla.pfui.equipment_bridge")?)?;
    lua.load(include_str!("tooltips.lua"))
        .set_name("@benilla-pfui/tooltips.lua")
        .call::<()>(
            lua.named_registry_value::<mlua::Table>("benilla.pfui.equipment_bridge")?
                .get::<mlua::Table>("tooltips")?,
        )?;
    lua.load(include_str!("equipment_actions.lua"))
        .set_name("@benilla-pfui/equipment_actions.lua")
        .call::<()>(lua.named_registry_value::<mlua::Table>("benilla.pfui.equipment_bridge")?)?;
    lua.load(include_str!("engine_tail.lua"))
        .set_name("@benilla-pfui/engine_tail.lua")
        .call::<()>(engine.clone())?;
    lua.load(include_str!("events.lua"))
        .set_name("@benilla-pfui/events.lua")
        .call::<()>(engine)?;
    mirror(lua)?;
    lua.set_named_registry_value(INSTALLED, true)
}

/// The `ClassicAPI` table: every engine function under its own name, a namespace as a subtable,
/// as the DLL binds each one twice. FrameXML and addons load later, so a global they replace
/// still reads here as the engine's.
fn mirror(lua: &Lua) -> mlua::Result<()> {
    let mirror = lua.create_table()?;
    for pair in lua.globals().pairs::<Value, Value>() {
        let (name, value) = pair?;
        let Value::String(key) = &name else {
            continue;
        };
        match value {
            Value::Function(_) => mirror.raw_set(name, value)?,
            Value::Table(namespace) if key.to_string_lossy().starts_with("C_") => {
                let copy = lua.create_table()?;
                for entry in namespace.pairs::<Value, Value>() {
                    let (field, value) = entry?;
                    copy.raw_set(field, value)?;
                }
                mirror.raw_set(name, copy)?;
            }
            _ => {}
        }
    }
    for library in ["string", "table", "math", "coroutine"] {
        if let Value::Table(namespace) = lua.globals().get::<Value>(library)? {
            let copy = lua.create_table()?;
            for entry in namespace.pairs::<Value, Value>() {
                let (field, value) = entry?;
                copy.raw_set(field, value)?;
            }
            mirror.raw_set(library, copy)?;
        }
    }
    lua.globals().set("ClassicAPI", mirror)
}

/// Execute an addon file with its modern `(addonName, privateTable)` arguments. The caller owns
/// the private table's lifetime and reuses it for every file in that addon, including XML files.
/// Kept here so the stock loader's 1.12 calling convention needs no change.
pub fn run_addon_chunk(
    lua: &Lua,
    source: &[u8],
    chunk_name: &str,
    addon_name: &str,
    private_table: mlua::Table,
) -> mlua::Result<()> {
    lua.load(benilla_ui::source::chunk(source))
        .set_name(chunk_name)
        .set_mode(mlua::ChunkMode::Text)
        .call((addon_name, private_table))
}

/// Provide the class-name table pfUI's environment requires, using the host's localized DBC
/// names. No hardcoded locale/class names are shipped. Install after the utility subset and
/// before loading pfUI. The host calls this again when constructing a replacement VM.
pub fn install_class_names<'a>(
    script: &UiScript,
    names: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> mlua::Result<()> {
    let lua = script.lua();
    if !lua
        .globals()
        .get::<Value>("LOCALIZED_CLASS_NAMES_MALE")?
        .is_nil()
    {
        return Err(mlua::Error::runtime("class-name table already installed"));
    }
    let table = lua.create_table()?;
    let copy = lua.create_table()?;
    for (token, name) in names {
        table.set(token, name)?;
        copy.set(token, name)?;
    }
    // FillLocalizedClassList's own copy: the ClassicAPI addon rebuilds the global through it.
    lua.set_named_registry_value(CLASS_NAMES, copy)?;
    lua.globals().set("LOCALIZED_CLASS_NAMES_MALE", table)
}

/// The frame types a ClassicAPI client adds that this layer builds from 1.12 ones: `Cooldown` is
/// the indicator model with the modern methods.
const INTRINSICS: &str = r#"<Ui>
  <Model name="Cooldown" inherits="CooldownFrameTemplate" mixin="BenillaCooldownMixin" intrinsic="true"/>
</Ui>"#;

/// The ClassicAPI addon's files this client does not load, as their engine half is missing: none
/// at present. A file a later ClassicAPI adds before its engine half is here goes in this list.
const SKIPPED: &[(&str, &str)] = &[];

/// Whether `file` of `addon`'s manifest stays unloaded ([`SKIPPED`]).
pub fn skipped_file(addon: &str, file: &str) -> bool {
    let file = file.replace('/', "\\");
    SKIPPED
        .iter()
        .any(|(a, f)| a.eq_ignore_ascii_case(addon) && f.eq_ignore_ascii_case(&file))
}

/// The school lockout `C_LossOfControl` reports: `{mask, starts, ends}` on the `GetTime` clock.
const SCHOOL_LOCKOUT: &str = "benilla.pfui.school_lockout";

/// The server locked the player out of the schools in `mask` for `seconds` (an interrupt).
pub fn school_lockout(lua: &Lua, mask: u32, seconds: f64) {
    let now = lua
        .globals()
        .get::<mlua::Function>("GetTime")
        .and_then(|f| f.call::<f64>(()))
        .unwrap_or(0.0);
    if let Ok(lockout) = lua.create_table() {
        let _ = lockout.set("mask", mask);
        let _ = lockout.set("starts", now);
        let _ = lockout.set("ends", now + seconds);
        let _ = lua.set_named_registry_value(SCHOOL_LOCKOUT, lockout);
    }
}

/// The text `CopyToClipboard` asked for this frame, which the client writes ([`take_clipboard`]).
const CLIPBOARD: &str = "benilla.pfui.clipboard";

/// The text a script asked `CopyToClipboard` to copy since the last call, the last one winning.
pub fn take_clipboard(script: &UiScript) -> Option<String> {
    let lua = script.lua();
    let pending = lua.named_registry_value::<mlua::Table>(CLIPBOARD).ok()?;
    let text: Option<String> = pending.get(1).ok().flatten();
    if text.is_some() {
        let _ = pending.set(1, Value::Nil);
    }
    text
}

/// The class names by token, as `install_class_names` received them.
const CLASS_NAMES: &str = "benilla.pfui.class_names";

#[cfg(test)]
mod tests;

#[cfg(test)]
mod classicapi_tests;

#[cfg(test)]
mod durations_tests;

#[cfg(test)]
mod aura_tests;

#[cfg(test)]
mod globals_tests;

#[cfg(test)]
mod state_events_tests;

#[cfg(test)]
mod focus_retention_tests;

#[cfg(test)]
mod item_use_tests;

#[cfg(test)]
mod macro_verb_tests;
