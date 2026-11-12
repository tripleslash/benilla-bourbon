//! The engine half of ClassicAPI's `texture:SetHorizTile`/`SetVertTile`: a tiled texture's file
//! loads repeating, and the per-frame pass (`benilla-pfui`'s `modern_widgets.lua`) learns the
//! image's pixel size to set the repeat count as texture coordinates.

use crate::script::region::region_handle_of;
use crate::script::texel::TexSet;
use crate::script::Model;
use mlua::{Lua, MultiValue, Table, Value};

/// Registry key of the table `modern_widgets.lua` takes: `prepare(texture)`.
pub const TILE_ENGINE: &str = "benilla.pfui.tile_engine";

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let engine = lua.create_table()?;
    // `prepare(texture)`: loads the texture's file repeating when it does not yet, and answers
    // the image's width and height in pixels; nothing for no file or an unknown size.
    engine.set(
        "prepare",
        lua.create_function(|lua, texture: Table| {
            let rh = region_handle_of(lua, &texture)?;
            let mut model = lua.app_data_mut::<Model>().expect("model app_data");
            let Some((path, wrap)) = model
                .region_data
                .get(&rh)
                .and_then(|data| Some((data.tex.path().clone()?, data.tex.wrap())))
            else {
                return Ok(MultiValue::new());
            };
            if !wrap {
                // A load of the file a texture already shows keeps the flag it loaded with
                // (`0x770225`), so the texture lets go of it first; its other state stays.
                let Model {
                    region_data,
                    texels,
                    ..
                } = &mut *model;
                if let Some(data) = region_data.get_mut(&rh) {
                    texels.set(&mut data.tex, TexSet::None, |_| false);
                    texels.set(&mut data.tex, TexSet::WrapFile(path.clone()), |_| false);
                }
                model.note_art(rh);
            }
            let size = model
                .texture_size_probe
                .as_ref()
                .and_then(|probe| probe(&path))
                .filter(|(w, h)| *w > 0 && *h > 0);
            Ok(match size {
                Some((w, h)) => {
                    MultiValue::from_vec(vec![Value::Integer(w.into()), Value::Integer(h.into())])
                }
                None => MultiValue::new(),
            })
        })?,
    )?;
    lua.set_named_registry_value(TILE_ENGINE, engine)
}
