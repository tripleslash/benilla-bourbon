//! Sanity for `web/boot-manifest.json` — the boot-prefetch read-set (captured from a traced boot
//! via `?boottrace=1`; regeneration procedure in boot.js's header). The manifest is data,
//! hand-carried between builds, so this is the guard against the hand: shape, non-emptiness, no
//! duplicates, and chain-style names.
//!
//! **These manifests now do two jobs, and the second raised the stakes.** They were a list of URLs
//! to warm an HTTP cache with, and drift cost one uncached read. They are now translated by
//! `benilla-pack` into `data/boot-packs.json` — the page's entire prefetch — *and* fed back into
//! the packer's grouping, so the boot read set is laid out contiguously in the container
//! (`benilla_packer::sidecars::ReadSets`). A name that no longer resolves is therefore a name that
//! silently drops out of both, and enough of them turn the prefetch into a stub that says nothing
//! is needed. Hence the last test here, which checks every name against a built `index.bpi`.
//!
//! Staleness in the other direction — a name the boot no longer reads — is still not tested, and
//! still degrades gracefully: it costs a pack in the prefetch, nothing more.

use std::path::Path;

fn manifest(file: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../web")
        .join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "reading {}: {e} — capture one with ?boottrace=1",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{file} parses as JSON: {e}"))
}

/// The shape both manifests share: version 1, a non-trivial list of unique raw chain names.
fn checked_names(file: &str, at_least: usize) -> Vec<String> {
    let m = manifest(file);
    assert_eq!(
        m["version"], 1,
        "{file}: unknown manifest version: {}",
        m["version"]
    );
    let names: Vec<String> = m["names"]
        .as_array()
        .expect("names is an array")
        .iter()
        .map(|v| v.as_str().expect("every name is a string").to_string())
        .collect();
    assert!(
        names.len() >= at_least,
        "{file}: {} names smells like a truncated capture (expected ≥ {at_least})",
        names.len()
    );
    let mut seen = std::collections::HashSet::new();
    for name in &names {
        assert!(
            seen.insert(name.as_str()),
            "{file}: duplicate entry: {name}"
        );
        // Chain names are internally backslash-separated and never URL-encoded here — boot.js
        // runs encodeURIComponent at fetch time (the exact twin of web.rs::encode_name).
        assert!(
            !name.contains('/'),
            "{file}: forward slash in {name} — store raw chain names"
        );
        assert!(
            !name.contains('%'),
            "{file}: pre-encoded name {name} — store raw chain names"
        );
        assert!(
            !name.starts_with("benilla:"),
            "{file}: {name} is a virtual name, not a chain file"
        );
        assert!(!name.is_empty());
    }
    names
}

#[test]
fn the_boot_manifest_is_version_1_with_chain_names() {
    let names = checked_names("boot-manifest.json", 50);
    // The one file every boot certainly reads: if this is missing, the capture never reached
    // the catalog pile at all.
    assert!(
        names
            .iter()
            .any(|n| n.eq_ignore_ascii_case("DBFilesClient\\Map.dbc")),
        "Map.dbc missing — was the trace captured from a real boot?"
    );
}

/// The world-entry set (`web/boot.js` warms it after `ready`): the UI sprites the first in-game
/// draw resolves through blocking reads. It is the boot capture's complement — nothing in it
/// may already be in the boot manifest, or the boot capture was stale when it was cut.
#[test]
fn the_world_manifest_is_the_ui_sprite_set_disjoint_from_boot() {
    let boot: std::collections::HashSet<String> = checked_names("boot-manifest.json", 50)
        .into_iter()
        .collect();
    let world = checked_names("world-manifest.json", 50);
    let sprites = world
        .iter()
        .filter(|n| n.to_ascii_lowercase().starts_with("interface\\"))
        .count();
    assert!(
        sprites * 2 > world.len(),
        "the world-entry set is mostly Interface\\ sprites; got {sprites} of {}",
        world.len()
    );
    for name in &world {
        assert!(
            !boot.contains(name),
            "{name} is in both manifests — recapture both together"
        );
    }
}

/// **The guard against the manifests rotting into an empty pack list.** Every name in both sets
/// must resolve in a built container's `index.bpi`, by the same `(name_a, name_b)` hash the archive
/// itself resolves on.
///
/// A handful of misses is the honest tolerance, not a hole: some captured names are probes for
/// files that genuinely are not in a stock 5875 install (the sprite walk asks for `Foo.blp` and
/// then `Foo.tga` by design, and the icon list carries two entries with no file behind them), and
/// failing on those would make a correct manifest untestable. What the threshold catches is the
/// failure that matters — a re-pack, a rename, or a hand-edit that quietly detaches the read set
/// from the container.
///
/// Skips with a message when there is no container, like every test that needs the real install.
#[test]
fn every_manifest_name_resolves_in_a_built_container() {
    let index = benilla_pack::pack_index_or_skip!();
    for file in ["boot-manifest.json", "world-manifest.json"] {
        let names = checked_names(file, 50);
        let missing: Vec<&String> = names
            .iter()
            .filter(|n| index.locate(benilla_mpq::name_key(n)).is_none())
            .collect();
        assert!(
            missing.len() * 20 < names.len(),
            "{file}: {} of {} names are absent from the container — the prefetch has detached \
             from what the client reads. Re-run `benilla-pack sidecars` and recapture if needed. \
             First few: {:?}",
            missing.len(),
            names.len(),
            &missing[..missing.len().min(8)]
        );
    }
}
