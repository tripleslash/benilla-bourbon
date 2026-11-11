//! The open profession windows, as the extension's profession links (`|Htrade:…|h`) read them.

use crate::script::Model;
use mlua::Lua;

/// The open TradeSkill window: its `SkillLine.dbc` line and `GetTradeSkillLine`'s rank and
/// maximum, each with the permanent bonus.
pub fn open_trade_skill(lua: &Lua) -> Option<(u32, u32, u32)> {
    let model = lua.app_data_ref::<crate::script::ModelCell>()?;
    let state = model
        .trade_skill
        .as_ref()
        .filter(|_| model.trade_skill_open)?;
    Some((state.line, state.rank, state.max_rank))
}

/// The open Craft window: its craft type (the opener's `EffectMiscValue[0]`) and its rank and
/// maximum, each with the permanent bonus.
pub fn open_craft(lua: &Lua) -> Option<(u32, u32, u32)> {
    let model = lua.app_data_ref::<crate::script::ModelCell>()?;
    let state = model.craft.as_ref().filter(|_| model.craft_open)?;
    Some((state.craft_type, state.rank, state.max_rank))
}

/// The players whose links the TradeSkill and Craft windows show, as the app opened them.
#[derive(Default)]
struct Linked {
    trade_skill: Option<u64>,
    craft: Option<u64>,
}

/// The app opened (`Some`) or dropped (`None`) a link in the TradeSkill window, or the Craft
/// window with `craft`.
pub fn set_linked(lua: &Lua, craft: bool, guid: Option<u64>) {
    if lua.app_data_ref::<Linked>().is_none() {
        lua.set_app_data(Linked::default());
    }
    let mut linked = lua.app_data_mut::<Linked>().expect("linked");
    if craft {
        linked.craft = guid;
    } else {
        linked.trade_skill = guid;
    }
}

/// The linker whose book the window shows, `None` for the player's own.
pub fn linked(lua: &Lua, craft: bool) -> Option<u64> {
    let linked = lua.app_data_ref::<Linked>()?;
    if craft {
        linked.craft
    } else {
        linked.trade_skill
    }
}

/// The extension's reader of a `trade:` link, which `SetHyperlink` hands one to.
const OPEN_TRADE_LINK: &str = "benilla.pfui.open_trade_link";

/// Install the reader a `trade:` link clicked in chat goes to.
pub fn set_link_reader(lua: &Lua, reader: mlua::Function) -> mlua::Result<()> {
    lua.set_named_registry_value(OPEN_TRADE_LINK, reader)
}

/// A `trade:` link the tooltip was handed, read by the extension's reader when it has one.
pub(super) fn open_link(lua: &Lua, link: &str) -> mlua::Result<()> {
    if let Some(reader) = lua.named_registry_value::<Option<mlua::Function>>(OPEN_TRADE_LINK)? {
        reader.call::<()>(link)?;
    }
    Ok(())
}

/// Ask the host to show another player's book: `opener`'s window lists `spells` at the linker
/// `guid`'s `rank` and `max_rank`.
pub fn request_link(
    lua: &Lua,
    opener: u32,
    guid: u64,
    spells: Vec<u32>,
    rank: u32,
    max_rank: u32,
) -> mlua::Result<()> {
    crate::script::calls::submit(
        lua,
        crate::script::ScriptCall::PfuiClick(super::attributes::ClickAction::OpenTradeLink {
            opener,
            guid,
            spells,
            rank,
            max_rank,
        }),
    )
}
