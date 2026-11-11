use crate::script::Model;
use mlua::{Lua, MultiValue, Value};

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let api = lua.create_table()?;
    api.set(
        "DoesAddOnExist",
        lua.create_function(|lua, name: String| {
            Ok(lua
                .app_data_ref::<crate::script::ModelCell>()
                .expect("model")
                .addons
                .iter()
                .any(|addon| addon.name.eq_ignore_ascii_case(&name)))
        })?,
    )?;
    api.set(
        "GetAddOnOptionalDependencies",
        lua.create_function(|lua, key: Value| {
            let dependencies = {
                let model = lua.app_data_ref::<crate::script::ModelCell>().expect("model");
                let row = match key {
                    Value::String(name) => model
                        .addons
                        .iter()
                        .find(|a| a.name.eq_ignore_ascii_case(&name.to_string_lossy())),
                    Value::Integer(i) if i > 0 => model
                        .addon_index
                        .get((i - 1) as usize)
                        .and_then(|i| model.addons.get(*i)),
                    Value::Number(i) if i.is_finite() && i >= 1.0 => model
                        .addon_index
                        .get((i as usize) - 1)
                        .and_then(|i| model.addons.get(*i)),
                    _ => None,
                };
                row.map(|addon| {
                    addon
                        .directives
                        .iter()
                        .filter(|(name, _)| {
                            name.eq_ignore_ascii_case("OptionalDeps")
                                || name.eq_ignore_ascii_case("OptionalDependencies")
                        })
                        .flat_map(|(_, names)| {
                            names
                                .split([',', ' ', '\t'])
                                .filter(|n| !n.is_empty())
                                .map(str::to_owned)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
            };
            Ok(MultiValue::from_vec(
                dependencies
                    .iter()
                    .map(|name| lua.create_string(name).map(Value::String))
                    .collect::<mlua::Result<Vec<_>>>()?,
            ))
        })?,
    )?;
    lua.globals().set("C_AddOns", api)
}
