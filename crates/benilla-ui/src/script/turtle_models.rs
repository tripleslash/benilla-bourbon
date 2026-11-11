//! Opt-in Turtle model preview extensions. Stock unit-token resolution remains unchanged.

use mlua::{Function, Lua, Table, Value};

use super::modelframe::{with_model, REG_PLAYERMODEL_METHODS};
use super::object::frame_handle_of;
use super::UiScript;

const ENABLED: &str = "__benilla_turtle_model_previews";

fn display_id(value: &Value) -> Option<u32> {
    match value {
        Value::Integer(n) => u32::try_from(*n).ok().filter(|id| *id != 0),
        Value::Number(n)
            if n.is_finite() && n.fract() == 0.0 && *n >= 1.0 && *n <= u32::MAX as f64 =>
        {
            Some(*n as u32)
        }
        _ => None,
    }
}

fn set_display(lua: &Lua, this: &Table, display: u32) -> mlua::Result<()> {
    with_model(lua, this, |_| ())?;
    let handle = frame_handle_of(lua, this)?;
    // Use the same subject and renderer as the stable's display-only paperdoll. This retains
    // the display's skins and animations, instead of treating its id as a unit or bare M2 path.
    lua.app_data_mut::<crate::script::ModelCell>()
        .expect("model app_data")
        .set_pane_display(handle, display);
    Ok(())
}

impl UiScript {
    /// Turtle's `PlayerModel:SetUnit(displayId)` overload, inherited by DressUpModel. Named
    /// units keep the stock binding; RefreshUnit keeps a numeric preview's display subject.
    pub fn enable_turtle_model_previews(&self) -> mlua::Result<()> {
        let lua = self.lua();
        if lua.named_registry_value::<bool>(ENABLED).unwrap_or(false) {
            return Ok(());
        }
        let Some(methods) = lua.named_registry_value::<Option<Table>>(REG_PLAYERMODEL_METHODS)?
        else {
            return Ok(()); // Glue has no PlayerModel method table.
        };
        let stock_set: Function = methods.get("SetUnit")?;
        methods.set(
            "SetUnit",
            lua.create_function(move |lua, (this, unit): (Table, Value)| {
                if let Some(display) = display_id(&unit) {
                    return set_display(lua, &this, display);
                }
                stock_set.call::<()>((this, unit))
            })?,
        )?;
        let stock_refresh: Function = methods.get("RefreshUnit")?;
        methods.set(
            "RefreshUnit",
            lua.create_function(move |lua, this: Table| {
                let display = with_model(lua, &this, |m| {
                    m.character.as_ref().map_or(0, |c| c.display)
                })?;
                if display != 0 {
                    return set_display(lua, &this, display);
                }
                stock_refresh.call::<()>(this)
            })?,
        )?;
        lua.set_named_registry_value(ENABLED, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preview(turtle: bool) -> UiScript {
        let mut s = UiScript::new().unwrap();
        if turtle {
            s.enable_turtle_model_previews().unwrap();
            s.enable_turtle_model_previews().unwrap();
        }
        s.run(r#"p = CreateFrame("DressUpModel", "ShopPreview") p:SetWidth(100) p:SetHeight(100)"#)
            .unwrap();
        s.resolve();
        s
    }

    #[test]
    fn stock_models_still_reject_a_numeric_unit() {
        let s = preview(false);
        let error = s.run("p:SetUnit(20418)").unwrap_err();
        assert!(error.to_string().contains("Unknown unit name: 20418"));
    }

    #[test]
    fn shop_display_ids_request_real_character_copies_and_survive_refresh() {
        let s = preview(true);
        for id in [20418, 20506, 18778, 11444] {
            s.run(&format!("p:SetUnit({id})")).unwrap();
            let pane = s.model_pane("ShopPreview").unwrap().character.unwrap();
            assert_eq!((pane.guid, pane.display), (0, id));
            assert!(pane.pending());
            s.run("p:RefreshUnit()").unwrap();
            let refreshed = s.model_pane("ShopPreview").unwrap().character.unwrap();
            assert_eq!((refreshed.guid, refreshed.display), (0, id));
            assert!(refreshed.builds > pane.builds);
        }
    }

    /// On a plain PlayerModel: a DressUpModel's `RefreshUnit` redresses from the unit's live
    /// model, which reads through a lent call host this bare script has not got.
    #[test]
    fn named_units_replace_the_display_and_other_unit_apis_stay_strict() {
        let mut s = preview(true);
        s.run(r#"q = CreateFrame("PlayerModel", "ShopPlayer") q:SetWidth(100) q:SetHeight(100)"#)
            .unwrap();
        s.resolve();
        s.set_unit_guids(&super::super::UnitGuids {
            player: 1,
            held: [(1, 0)].into_iter().collect(),
            ..Default::default()
        });
        s.run(r#"q:SetUnit(20418) q:SetUnit("player") q:RefreshUnit()"#)
            .unwrap();
        let pane = s.model_pane("ShopPlayer").unwrap().character.unwrap();
        assert_eq!((pane.guid, pane.display), (1, 0));
        for chunk in ["q:SetUnit('bogus')", "q:SetUnit(0.5)", "UnitName(20418)"] {
            assert!(s
                .run(chunk)
                .unwrap_err()
                .to_string()
                .contains("Unknown unit name"));
        }
    }

    #[test]
    fn shop_portrait_accepts_its_texture_object() {
        let s = preview(true);
        s.run(
            r#"
            icon = p:CreateTexture("ShopItemTexture")
            SetPortraitToTexture(icon, "Interface\\Icons\\INV_Misc_QuestionMark")
        "#,
        )
        .unwrap();
        assert_eq!(
            s.eval::<String>("return icon:GetTexture()").unwrap(),
            r"Interface\Icons\INV_Misc_QuestionMark"
        );
    }

    #[test]
    fn glue_does_not_require_a_player_model_table() {
        UiScript::new_glue()
            .unwrap()
            .enable_turtle_model_previews()
            .unwrap();
    }
}
