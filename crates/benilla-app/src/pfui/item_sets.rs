//! ClassicAPI's item-set getters (`C_Item.GetItemSetID`, `GetItemSetIDByID`, `GetItemSetInfo`):
//! a template's `itemset` id, as `GetItemInfo`'s 16th return carries it, and its `ItemSet.dbc` row.

use benilla_ui::mlua::{self, Lua, Value};
use std::sync::OnceLock;

/// `ItemSet.dbc`, read once; `None` without the install, so the getters answer nil.
fn catalog() -> Option<&'static benilla_formats::ItemSetCatalog> {
    static CATALOG: OnceLock<Option<benilla_formats::ItemSetCatalog>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            let data = benilla_formats::wow_data()?;
            let mut chain = benilla_formats::open_chain(&data).ok()?;
            benilla_formats::load_item_sets(&mut chain).ok()
        })
        .as_ref()
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let api: mlua::Table = lua.globals().get("C_Item")?;
    api.set(
        "GetItemSetInfo",
        lua.create_function(|lua, id: Value| {
            let Some(id) = lua
                .coerce_number(id)?
                .filter(|n| n.is_finite() && *n >= 0.0 && *n <= f64::from(u32::MAX))
            else {
                return Ok(Value::Nil);
            };
            let Some(set) = catalog().and_then(|c| c.set(id as u32)) else {
                return Ok(Value::Nil);
            };
            let info = lua.create_table()?;
            info.set("setID", id as u32)?;
            info.set("name", set.name.as_str())?;
            info.set("requiredSkill", set.required_skill)?;
            info.set("requiredSkillRank", set.required_skill_rank)?;
            let items = lua.create_sequence_from(set.items.iter().copied().filter(|&i| i != 0))?;
            info.set("items", items)?;
            let bonuses = lua.create_table()?;
            for &(threshold, spell) in set.bonuses.iter().filter(|(_, spell)| *spell != 0) {
                let bonus = lua.create_table()?;
                bonus.set("spellID", spell)?;
                bonus.set("threshold", threshold)?;
                bonuses.raw_push(bonus)?;
            }
            info.set("bonuses", bonuses)?;
            Ok(Value::Table(info))
        })?,
    )?;
    lua.load(
        r#"
        -- The 16th return of GetItemInfo is the template's set; nil while uncached or setless.
        function C_Item.GetItemSetIDByID(item)
          return (select(16, C_Item.GetItemInfo(item)))
        end
        function C_Item.GetItemSetID(itemLocation)
          local id = C_Item.GetItemID(itemLocation)
          if not id then return nil end
          return C_Item.GetItemSetIDByID(id)
        end
    "#,
    )
    .set_name("=[benilla-pfui item sets]")
    .exec()
}
