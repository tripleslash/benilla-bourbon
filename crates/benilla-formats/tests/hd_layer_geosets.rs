//! **The updated-character-model layer authors geosets above 1700, which 1.12.1 never switches
//! off** (`0x477520` disables only 0..=1700) — and until this test they were all hidden, which took the feet off ten of its bodies and the waistline off
//! eight (reported 2026-09-08 against the hosted page: HD orcs standing on stumps, an HD human
//! with a gap where the belt goes).
//!
//! A real-data difftest over both corpora, because the fix is a claim about DATA and not about
//! code: the shipped 1.12.1 bodies author nothing above the sixteen managed groups, the layer's
//! bodies do, and `VisibleGeosets::contains` is what tells the two apart. Skips when the
//! client is not present, and again when the optional layer is not in its `Data` folder.

use std::collections::BTreeSet;
use std::path::Path;

use benilla_formats::{Chain, CharacterGeosets, EquipGeosets};

/// The last geoset id the compositor switches off (`0x477520`, through `0x7110d0`).
const LAST_MANAGED_GEOSET: u16 = 1700;

/// Every `.m2` under `Character\` in the chain as it stands, with the geoset ids it authors.
fn bodies(chain: &mut Chain) -> Vec<(String, BTreeSet<u16>)> {
    let names: Vec<String> = chain
        .list()
        .expect("chain listfile")
        .into_iter()
        .map(|e| e.name)
        .filter(|n| {
            let l = n.to_ascii_lowercase();
            l.starts_with("character") && l.ends_with(".m2")
        })
        .collect();
    let mut out = Vec::new();
    for name in names {
        let Ok(bytes) = chain.read_file(&name) else {
            continue;
        };
        let dir = name.rsplit_once('\\').map(|(d, _)| d).unwrap_or("");
        let Ok(subs) = benilla_formats::parse_m2_render_submeshes(&bytes, dir, &[]) else {
            continue;
        };
        out.push((name, subs.iter().map(|s| s.geoset_id).collect()));
    }
    out
}

#[test]
fn the_layer_s_unmanaged_geosets_are_drawn_and_the_shipped_art_has_none() {
    let data = benilla_formats::wow_data_or_skip!();
    let path = Path::new(&data);
    let mut vanilla = Chain::open_layered(path, false).expect("open the vanilla chain");
    let shipped = bodies(&mut vanilla);
    assert!(!shipped.is_empty(), "the install has character models");

    // Half one, and the reason the selection needs no `hd_character_models()` check: on the art
    // 1.12 shipped, the above-1700 rule can never fire. Nothing above the managed groups is authored,
    // so every goldens-bearing body renders exactly as it did.
    for (name, ids) in &shipped {
        let high: Vec<u16> = ids
            .iter()
            .copied()
            .filter(|i| *i > LAST_MANAGED_GEOSET)
            .collect();
        assert!(
            high.is_empty(),
            "{name}: the shipped art authors no geoset the 1.12 law cannot name, got {high:?}"
        );
    }

    let mut layered = Chain::open_layered(path, true).expect("open the layered chain");
    let hd = bodies(&mut layered);
    if hd.len() == shipped.len() && hd.iter().zip(&shipped).all(|(a, b)| a.1 == b.1) {
        eprintln!("no patch-A.MPQ in this Data folder - skipping the layer half");
        return;
    }

    // Half two: what the layer actually authors up there, and that the selection draws all of it.
    // The naked set is the strictest case - nothing worn can be blamed for a hidden body part.
    let geosets = CharacterGeosets::load(&mut layered).expect("the customization DBCs");
    let naked = EquipGeosets::default();
    let mut seen: BTreeSet<u16> = BTreeSet::new();
    for (name, ids) in &hd {
        // Race/sex do not reach the groups in question at all - they choose hair and facial hair -
        // so one look stands for every body here.
        let visible = geosets.visible_geosets(1, 0, 0, 0, &naked);
        for id in ids.iter().copied().filter(|i| *i > LAST_MANAGED_GEOSET) {
            seen.insert(id);
            assert!(
                visible.contains(&id),
                "{name}: geoset {id} is body this model always draws, and it is hidden"
            );
        }
    }
    // The four the layer ships, pinned: the waist band, both feet variants, and the night elves'
    // eye glow. A layer that grows a fifth should say so here rather than silently rendering it.
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        vec![1702, 1801, 2001, 2002],
        "the layer's unmanaged geosets"
    );
}
