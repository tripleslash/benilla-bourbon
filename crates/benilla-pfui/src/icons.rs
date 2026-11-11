//! Icon enumeration from the owner's archive listing, separate from stock macro indices.

use mlua::{Lua, Table};
use std::collections::BTreeSet;

#[derive(Clone, Default)]
pub struct Catalog {
    pub spells: Vec<String>,
    pub items: Vec<String>,
    pub loose_spells: Vec<String>,
    pub loose_items: Vec<String>,
}

fn names(paths: impl IntoIterator<Item = String>) -> (Vec<String>, Vec<String>) {
    let mut spells = BTreeSet::new();
    let mut items = BTreeSet::new();
    for path in paths {
        let path = path.replace('/', "\\").to_ascii_uppercase();
        let Some(file) = path.strip_prefix("INTERFACE\\ICONS\\") else {
            continue;
        };
        if file.contains('\\') {
            continue;
        }
        let Some((stem, extension)) = file.rsplit_once('.') else {
            continue;
        };
        if extension != "BLP" && extension != "TGA" {
            continue;
        }
        if stem.starts_with("SPELL_") || stem.starts_with("ABILITY_") {
            spells.insert(stem.to_owned());
        } else if stem.starts_with("INV_") {
            items.insert(stem.to_owned());
        }
    }
    (spells.into_iter().collect(), items.into_iter().collect())
}

impl Catalog {
    pub fn from_paths(paths: impl IntoIterator<Item = String>) -> Self {
        let (spells, items) = names(paths);
        Self {
            spells,
            items,
            ..Self::default()
        }
    }

    pub fn with_loose_paths(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        (self.loose_spells, self.loose_items) = names(paths);
        self
    }

    pub fn feed(&self, lua: &Lua) {
        lua.set_app_data(self.clone());
    }
}

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_app_data(Catalog::default());
    for (name, item, loose) in [
        ("GetMacroIcons", false, false),
        ("GetMacroItemIcons", true, false),
        ("GetLooseMacroIcons", false, true),
        ("GetLooseMacroItemIcons", true, true),
    ] {
        lua.globals().set(
            name,
            lua.create_function(move |lua, buffer: Table| {
                let catalog = lua.app_data_ref::<Catalog>().expect("icon catalog");
                let names = match (item, loose) {
                    (false, false) => &catalog.spells,
                    (true, false) => &catalog.items,
                    (false, true) => &catalog.loose_spells,
                    (true, true) => &catalog.loose_items,
                };
                let offset = buffer.raw_len();
                for (index, name) in names.iter().enumerate() {
                    buffer.raw_set(offset + index + 1, name.as_str())?;
                }
                Ok(())
            })?,
        )?;
    }
    lua.load(include_str!("icons.lua"))
        .set_name("@benilla-pfui/icons.lua")
        .exec()
}
