//! A ClassicAPI ScrollingMessageFrame's history: `GetMessageInfo(index)`, the oldest first, and
//! `BackFillMessage`, which adds a message as the oldest (pfUI restores its chat history with it).

use crate::script::object::frame_handle_of;
use crate::script::UiScript;
use crate::widget::{KindState, MessageLine};
use mlua::{Lua, MultiValue, Table, Value};

fn lines(lua: &Lua, this: &Table) -> mlua::Result<Vec<MessageLine>> {
    let h = frame_handle_of(lua, this)?;
    let model = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model");
    match model.arena.frame(h).map(|f| &f.kind_state) {
        Some(KindState::ScrollingMessage(smf)) => Ok(smf.lines.iter().cloned().collect()),
        _ => Err(mlua::Error::runtime(crate::script::object::ERR_WRONG_TYPE)),
    }
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let methods: Table =
        lua.named_registry_value(crate::script::messageframe::REG_SCROLLINGMESSAGEFRAME_METHODS)?;
    methods.set(
        "GetMessageInfo",
        lua.create_function(|lua, (this, index): (Table, i64)| {
            let lines = lines(lua, &this)?;
            let Some(line) = usize::try_from(index - 1).ok().and_then(|i| lines.get(i)) else {
                return Ok(MultiValue::new());
            };
            let [r, g, b] = line.color.map(|c| Value::Number(f64::from(c) / 255.0));
            Ok(MultiValue::from_vec(vec![
                Value::String(lua.create_string(&line.text)?),
                r,
                g,
                b,
                Value::Integer(i64::from(line.id)),
            ]))
        })?,
    )?;
    // The ring keeps its slots by the order lines came in, so a message added as the oldest
    // rebuilds it: the new line first, then every line it held, in order. Their fades start over.
    methods.set(
        "BackFillMessage",
        lua.create_function(
            |lua, (this, text, r, g, b, id): (Table, Value, Value, Value, Value, Value)| {
                let Some(text) = crate::script::messageframe::message_text(lua, &text) else {
                    return Ok(());
                };
                let h = frame_handle_of(lua, &this)?;
                let channel = |v: &Value| crate::script::messageframe::channel(lua, v);
                let gated = |v: &Value| crate::script::messageframe::is_number(lua, v).is_some();
                let color = if [&r, &g, &b].into_iter().all(gated) {
                    crate::color::Rgba8::rgba(
                        quantize(channel(&r)),
                        quantize(channel(&g)),
                        quantize(channel(&b)),
                        255,
                    )
                } else {
                    let model = lua
                        .app_data_ref::<crate::script::ModelCell>()
                        .expect("model");
                    UiScript::message_frame_color(&model, h)
                };
                let id = crate::script::messageframe::is_number(lua, &id)
                    .map_or(0, |n| crate::script::binding_abi::ftol(n) as u32);
                {
                    let mut model = lua
                        .app_data_mut::<crate::script::ModelCell>()
                        .expect("model");
                    let viewport = UiScript::message_viewport(&model, h);
                    let Some(KindState::ScrollingMessage(smf)) =
                        model.arena.frame_mut(h).map(|f| &mut f.kind_state)
                    else {
                        return Err(mlua::Error::runtime(crate::script::object::ERR_WRONG_TYPE));
                    };
                    if smf.lines.len() >= smf.max_lines {
                        return Ok(());
                    }
                    let held: Vec<MessageLine> = smf.lines.drain(..).collect();
                    smf.clear(viewport);
                    smf.add_packed(text, color, id, viewport);
                    for line in held {
                        let [r, g, b] = line.color;
                        let alpha = (line.alpha * 255.0).round() as u8;
                        let packed = crate::color::Rgba8::rgba(r, g, b, alpha.max(1));
                        smf.add_packed(line.text, packed, line.id, viewport);
                    }
                }
                crate::script::messageframe::fire_scroll_changes(lua, h);
                Ok(())
            },
        )?,
    )
}

fn quantize(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
