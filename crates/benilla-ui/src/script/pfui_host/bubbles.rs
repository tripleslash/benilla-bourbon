//! `C_ChatBubbles`, ClassicAPI's list of the chat bubbles, over the frames the engine builds.

use crate::script::object::frame_wrapper;
use mlua::Lua;

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let api = lua.create_table()?;
    api.set(
        "GetAllChatBubbles",
        lua.create_function(|lua, ()| {
            // By the pool's own record, not by the bubble's art: an addon that skins a bubble
            // blanks its textures, and the bubble is a bubble still. Hidden ones are listed too,
            // as a bubble out of sight is hidden and shown again.
            let ids: Vec<u32> = {
                let mut model = lua
                    .app_data_mut::<crate::script::ModelCell>()
                    .expect("model");
                crate::script::chat_bubble::frames(&model)
                    .into_iter()
                    .map(|frame| model.frame_id(frame))
                    .collect()
            };
            let list = lua.create_table()?;
            for (index, id) in ids.into_iter().enumerate() {
                list.set(index + 1, frame_wrapper(lua, id)?)?;
            }
            Ok(list)
        })?,
    )?;
    lua.globals().set("C_ChatBubbles", api)
}
