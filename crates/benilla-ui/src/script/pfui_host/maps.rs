//! Read the engine's viewed map; the player's current zone is a separate selection.

pub fn viewed_area(lua: &mlua::Lua) -> Option<u32> {
    let model = lua.app_data_ref::<crate::script::ModelCell>()?;
    let map = &model.worldmap;
    if let Some(id) = map.direct_area {
        return map
            .direct_areas
            .iter()
            .find(|(row, _)| *row == id)
            .map(|(_, zone)| zone.area_id);
    }
    let (continent, zone) = map.selection;
    map.continents
        .get(continent.checked_sub(1)? as usize)?
        .zones
        .get(zone.checked_sub(1)? as usize)
        .map(|zone| zone.area_id)
}

/// The player's explored zones, `PLAYER_EXPLORED_ZONES_1`'s 64 words: what the lent host answers
/// now, else what the VM last read.
pub fn explored_zones(lua: &mlua::Lua) -> Vec<u32> {
    if let Some(explored) = crate::script::live::ask(lua, &crate::script::worldmap::MapSetterQuery)
        .and_then(|reads| reads.explored)
    {
        return explored;
    }
    lua.app_data_ref::<crate::script::ModelCell>()
        .map(|model| model.worldmap.explored.clone())
        .unwrap_or_default()
}

/// `MAP_EXPLORATION_UPDATED`, with no payload, once the player's explored zones changed.
pub(crate) fn exploration_changed(lua: &mlua::Lua) {
    if super::installed(lua).unwrap_or(false) {
        super::fire_pfui_event(lua, "MAP_EXPLORATION_UPDATED", Vec::new());
    }
}
