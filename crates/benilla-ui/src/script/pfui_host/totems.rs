//! The optional bar reads observed summons and carried tools, never a synthetic timer.

use crate::script::{ScriptCall, ScriptValue, SelectionRequest};
use mlua::{Lua, MultiValue, Value};

#[derive(Clone, PartialEq)]
pub struct Totem {
    pub spell_id: u32,
    pub guid: Option<u64>,
    pub name: String,
    pub icon: Option<String>,
    pub start: f64,
    pub duration: f64,
}
#[derive(Default)]
struct State {
    active: [Option<Totem>; 4],
    tools: [bool; 4],
}

pub fn set(lua: &Lua, active: [Option<Totem>; 4], tools: [u32; 4]) {
    let tools = std::array::from_fn(|slot| {
        lua.app_data_ref::<super::items::Extras>()
            .is_some_and(|extras| {
                tools[slot] != 0
                    && extras.instances.iter().any(|((bag, _), item)| {
                        (*bag == -100 || (0..=4).contains(bag))
                            && item.item_id == tools[slot]
                            && item.count > 0
                    })
            })
    });
    let changed = {
        let Some(mut state) = lua.app_data_mut::<State>() else {
            return;
        };
        let changed: Vec<_> = (0..4)
            .filter(|&slot| state.active[slot] != active[slot] || state.tools[slot] != tools[slot])
            .collect();
        state.active = active;
        state.tools = tools;
        changed
    };
    for slot in changed {
        crate::script::pfui_host::fire_pfui_event(
            lua,
            "PLAYER_TOTEM_UPDATE",
            vec![ScriptValue::Int(slot as i64 + 1)],
        );
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(State::default());
    lua.globals().set(
        "GetTotemInfo",
        lua.create_function(|lua, index: usize| {
            let Some(slot) = index.checked_sub(1).filter(|slot| *slot < 4) else {
                return Ok(MultiValue::new());
            };
            let state = lua.app_data_ref::<State>().expect("totem state");
            let active = state.active[slot].as_ref();
            Ok(MultiValue::from_vec(vec![
                Value::Boolean(state.tools[slot]),
                Value::String(lua.create_string(active.map_or("", |totem| totem.name.as_str()))?),
                Value::Number(active.map_or(0., |totem| totem.start)),
                Value::Number(active.map_or(0., |totem| totem.duration)),
                active
                    .and_then(|totem| totem.icon.as_deref())
                    .map(|icon| lua.create_string(icon))
                    .transpose()?
                    .map_or(Value::Nil, Value::String),
                Value::Integer(1),
                Value::Integer(active.map_or(0, |totem| i64::from(totem.spell_id))),
            ]))
        })?,
    )?;
    lua.globals().set(
        "TargetTotem",
        lua.create_function(|lua, index: usize| {
            let guid = index.checked_sub(1).and_then(|slot| {
                lua.app_data_ref::<State>()?
                    .active
                    .get(slot)?
                    .as_ref()?
                    .guid
            });
            if let Some(guid) = guid {
                // The gate's tag 2, which the selection's bind asks while the cursor is up.
                let may_cast =
                    crate::script::protected::gate(lua, crate::script::protected::Tag::CAST);
                crate::script::calls::submit(
                    lua,
                    ScriptCall::Select {
                        request: SelectionRequest::PfuiGuid(guid),
                        may_cast,
                    },
                )?;
            }
            Ok(())
        })?,
    )
}
