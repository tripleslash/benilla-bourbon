//! Locality grouping: deciding which blobs share a pack.
//!
//! This is the packer's one genuinely game-specific judgement, and the reason the container format
//! stays ignorant of what it carries. On a host without HTTP `Range` — which is what was measured
//! against a static host — a client that wants one 4 KB icon downloads the whole pack it sits in. That
//! is only acceptable if the rest of the pack is what it was about to ask for anyway. So a group is
//! not "some blobs that fit"; it is a set the client reads *together*:
//!
//! - the DBC set, which ~80 catalog loaders read in one Startup burst;
//! - an `Interface\` subtree, whose sprites a frame pulls as a unit;
//! - a 4x4 block of ADT tiles, because `terrain_stream` moves a window across the map and the
//!   neighbours are the next thing it wants;
//! - a model or WMO directory, because a doodad set spawns together.
//!
//! Get this right and a whole-pack fetch is a *prefetch*. Get it wrong and it is waste — which is
//! why the label is derived from the path and not from anything cleverer: the archive's own
//! directory layout already encodes how the game groups its content.

/// The locality label a blob sorts under. Groups are emitted in sorted label order, so packs end up
/// holding adjacent paths, which is the whole point.
pub fn label_of(
    name: Option<&str>,
    raw_len: u32,
    singleton_bytes: u32,
    archive: usize,
    block: u32,
) -> String {
    // Big blobs go alone regardless of where they live: bundling a 15 MiB sound with its neighbours
    // would make a Range-less client pay 15 MiB for a 4 KB sibling.
    if raw_len >= singleton_bytes {
        return format!("@big/{archive:02}/{block:08}");
    }
    let Some(name) = name else {
        // Measured: 2 of 79,735 entries land here. Archive block order is the only locality signal
        // left, and it is a decent one — it is the order Blizzard's own packer wrote them in.
        return format!("~raw/{archive:02}/{:06}", block / 512);
    };
    let lower = name.replace('/', "\\").to_ascii_lowercase();
    let parts: Vec<&str> = lower.split('\\').collect();

    if lower.starts_with("dbfilesclient\\") {
        // One group: the client reads essentially all of them before it draws a frame.
        return "dbc".to_string();
    }
    if lower.starts_with("interface\\") {
        return format!("ui/{}", head(&parts, 2));
    }
    // World\Maps\<map>\<map>_<x>_<y>.adt — block into 4x4 tile neighbourhoods.
    if lower.starts_with("world\\maps\\") && parts.len() >= 4 {
        if let Some((x, y)) = tile_coords(parts[3]) {
            return format!("map/{}/{:02}_{:02}", parts[2], x / 4, y / 4);
        }
        return format!("map/{}/misc", parts[2]);
    }
    if lower.starts_with("world\\wmo\\") {
        return format!("wmo/{}", head(&parts, 3));
    }
    if lower.starts_with("sound\\") || lower.starts_with("speech\\") {
        return format!("snd/{}", head(&parts, 2));
    }
    if let Some(
        "character" | "creature" | "item" | "spells" | "tileset" | "textures" | "world"
        | "environments" | "particles",
    ) = parts.first().copied()
    {
        return format!("art/{}", head(&parts, 2));
    }
    format!(
        "misc/{}",
        lower
            .rsplit('.')
            .next()
            .filter(|e| e.len() <= 5)
            .unwrap_or("none")
    )
}

/// The first `n` path components, joined — the group key for a directory-shaped label.
fn head(parts: &[&str], n: usize) -> String {
    let take = n.min(parts.len().saturating_sub(1)).max(1);
    parts[..take].join("/")
}

/// `<map>_<x>_<y>.<ext>` -> `(x, y)`. Returns `None` for anything that is not a tile file, so
/// `<map>.wdt` and friends fall to the map's `misc` group instead of being mis-blocked.
fn tile_coords(file: &str) -> Option<(u32, u32)> {
    let stem = file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file);
    let mut it = stem.rsplitn(3, '_');
    let y = it.next()?.parse().ok()?;
    let x = it.next()?.parse().ok()?;
    it.next()?; // the map name must actually be there
    Some((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BIG: u32 = 4 << 20;

    fn label(name: &str) -> String {
        label_of(Some(name), 1000, BIG, 0, 0)
    }

    #[test]
    fn the_dbc_set_is_one_group_because_startup_reads_all_of_it() {
        assert_eq!(label("DBFilesClient\\Spell.dbc"), "dbc");
        assert_eq!(label("DBFilesClient\\Map.dbc"), "dbc");
    }

    /// Adjacent tiles must share a group: the terrain streamer moves a window, so a pack fetch is
    /// the prefetch of the tiles it is about to want.
    #[test]
    fn map_tiles_block_into_4x4_neighbourhoods() {
        let a = label("World\\Maps\\Azeroth\\Azeroth_32_48.adt");
        assert_eq!(a, "map/azeroth/08_12");
        assert_eq!(label("World\\Maps\\Azeroth\\Azeroth_33_49.adt"), a);
        assert_eq!(label("World\\Maps\\Azeroth\\Azeroth_35_51.adt"), a);
        // The next block over is a different group.
        assert_ne!(label("World\\Maps\\Azeroth\\Azeroth_36_48.adt"), a);
        // And a different map never shares one.
        assert_ne!(label("World\\Maps\\Kalimdor\\Kalimdor_32_48.adt"), a);
    }

    /// A `.wdt` is not a tile; mis-parsing it into a block would scatter it away from its map.
    #[test]
    fn non_tile_map_files_fall_to_the_maps_misc_group() {
        assert_eq!(
            label("World\\Maps\\Azeroth\\Azeroth.wdt"),
            "map/azeroth/misc"
        );
    }

    #[test]
    fn interface_and_art_group_by_subtree() {
        assert_eq!(
            label("Interface\\Icons\\Spell_Holy_Heal.blp"),
            label("Interface\\Icons\\Spell_Fire_Fireball.blp")
        );
        assert_ne!(
            label("Interface\\Icons\\Spell_Holy_Heal.blp"),
            label("Interface\\Glues\\Common\\Glue-Panel-Button-Up.blp")
        );
        assert_eq!(label("Interface\\Icons\\A.blp"), "ui/interface/icons");
    }

    /// Grouping must inherit the archive's name equivalence, or the same file reached by two
    /// spellings would land in two packs.
    #[test]
    fn labels_are_case_and_separator_insensitive() {
        assert_eq!(
            label("Interface/Icons/Spell_Holy_Heal.blp"),
            label("INTERFACE\\ICONS\\SPELL_HOLY_HEAL.BLP")
        );
    }

    /// A big blob is a singleton wherever it lives — otherwise a Range-less client pays its whole
    /// size to reach a small neighbour.
    #[test]
    fn big_blobs_leave_their_directory_group() {
        let small = label_of(Some("Sound\\Music\\GlueScreenMusic.mp3"), 1000, BIG, 3, 12);
        let big = label_of(Some("Sound\\Music\\GlueScreenMusic.mp3"), BIG, BIG, 3, 12);
        assert!(small.starts_with("snd/"));
        assert!(big.starts_with("@big/"));
    }

    /// The 2 unnamed entries in the real corpus still need a home, and archive block order is a
    /// better one than a single bucket.
    #[test]
    fn unnamed_entries_group_by_archive_block_order() {
        assert_eq!(label_of(None, 100, BIG, 8, 1000), "~raw/08/000001");
        assert_eq!(label_of(None, 100, BIG, 8, 1100), "~raw/08/000002");
    }
}
