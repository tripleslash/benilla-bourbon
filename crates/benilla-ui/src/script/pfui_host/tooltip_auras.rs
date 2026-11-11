//! `GameTooltip:SetSpellByID` through the stock spell builder, and `SetUnitAura` through the stock
//! aura tooltips (`SetUnitBuff`, `SetUnitDebuff`), by the filter's polarity.

use mlua::{Function, Lua, MultiValue, Table, Value};

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let methods: Table = lua.named_registry_value(crate::script::tooltip::REG_TOOLTIP_METHODS)?;
    methods.set(
        "SetSpellByID",
        lua.create_function(|lua, (this, id): (Table, u32)| {
            super::set_spell_by_id(lua, &this, id, None, super::SpellRenderOpts::default())
        })?,
    )?;
    methods.set(
        "SetUnitAura",
        lua.create_function(
            |_, (this, unit, index, filter): (Table, Value, Value, Option<String>)| {
                let harmful = filter.as_deref().is_some_and(|filter| {
                    filter
                        .split(['|', ' '])
                        .any(|token| token.eq_ignore_ascii_case("HARMFUL"))
                });
                let method = if harmful {
                    "SetUnitDebuff"
                } else {
                    "SetUnitBuff"
                };
                this.get::<Function>(method)?
                    .call::<MultiValue>((this.clone(), unit, index))
            },
        )?,
    )
}
