//! The one side file that is not game data: `boot-packs.json`.
//!
//! ## `boot-packs.json` — the read-set manifests, translated
//!
//! `web/boot-manifest.json` (403 names) and `web/world-manifest.json` (200 names) are captured from
//! a traced boot and stay checked in as the source of truth for *what the client actually reads*.
//! They stay in names, not pack ids, deliberately: a name survives a re-pack, and
//! `benilla-app/tests/boot_manifest.rs` can keep checking them for shape and staleness. But the
//! page can no longer prefetch a name — there is no per-name route — so the packer translates
//! them here: name -> key -> index entry -> pack id, deduplicated, with each pack's served size so
//! the boot overlay can report megabytes instead of a file count.
//!
//! The collapse is large and it is the container's design showing up as a number: both sets are
//! dominated by `DBFilesClient\*` and `Interface\*`, which group together by construction, so a few
//! hundred names become a few dozen objects.
//!
//! A name that does not resolve is **counted and reported, not fatal**. Drift between a captured
//! manifest and a re-packed install is expected and degrades to exactly one slower read; a build
//! that failed over it would be a build that fails because someone renamed a texture.
//!
//! ## The glue set — prefixes, not a recording
//!
//! `web/glue-manifest.json` is the third set, and the one that is not captured: it names **path
//! prefixes** (`Interface\Glues\`, `Character\`, the `Char*` DBCs, the glue music), resolved
//! here against the container's own `listfile.txt`, with its `exclude` prefixes taken back out. A
//! traced boot stops at the first glue screen, so a recording holds the login screen and none of
//! what the character screens read later: every race's model, skin, hair and face, and every race's
//! creation backdrop. A trace would have to visit every race, gender and customization to record
//! them; "all of `Character\`" is the set those screens can reach, so the manifest says that.
//!
//! One manifest serves every container: the corpus and each layer (`--packs web/packs-hd`,
//! `web/packs-turtle`, …), whose listfile picks out its own glue art and character models. A layer
//! with none of them gets an empty set.
//!
//! The page warms the glue set behind the boot overlay, after the boot set, into the service
//! worker's disk cache rather than the resident map (`web/packs.js`, `persistPacks`): it is too big
//! to hold decoded, and from disk a read is a local answer instead of a round trip.
//!
//! ## Where `config.json` went
//!
//! It used to be written here as a stub, and that was the wrong home for it. The gateway address
//! and the realmlist are **operator configuration, not a function of the game files**: writing them
//! into `web/packs/` put them in a 3.3 GiB gitignored directory that is regenerated from a WoW
//! install, where they could not be checked in, reviewed, or found. They now live at
//! `web/config.json`, ship with the *build* (Trunk copies it to the dist root), and are served at
//! `/config.json`. The packer has nothing to say about them.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use benilla_mpq::name_key;
use benilla_pack::Index;

/// The two captured read sets as key sets, for the grouping pass.
///
/// **The read sets are a locality signal, and the strongest one in the tree.** Every other label in
/// `group.rs` is a guess about what the client reads together, inferred from the archive's own
/// directory layout; these two are a *recording* of what it actually read. Measured on the first
/// container built without them: the 403 boot names landed in 141 packs — 91 MB fetched to reach
/// 14.7 MB, a **6.2x over-fetch** before the first frame — because a boot read set is a thin
/// cross-section of the whole tree (one icon here, one font there) and path-shaped groups scatter
/// exactly that shape. Grouping them together instead takes the ratio to ~1.
#[derive(Default)]
pub struct ReadSets {
    pub boot: HashSet<u64>,
    pub world: HashSet<u64>,
}

impl ReadSets {
    pub fn load(boot_manifest: &Path, world_manifest: &Path) -> Self {
        Self {
            boot: keys_of(boot_manifest),
            world: keys_of(world_manifest),
        }
    }

    /// The label prefix a blob's read-set membership forces, if any. Boot wins over world; the two
    /// manifests are asserted disjoint by `benilla-app/tests/boot_manifest.rs`, so that only ever
    /// settles a manifest that has drifted.
    ///
    /// `!` sorts below every other label's first character, so the read sets take the low pack ids —
    /// which is incidental but pleasant: the objects a cold visitor needs first are `p/00/*`.
    pub fn prefix(&self, key: u64) -> Option<&'static str> {
        if self.boot.contains(&key) {
            Some("!boot")
        } else if self.world.contains(&key) {
            Some("!world")
        } else {
            None
        }
    }
}

fn keys_of(manifest: &Path) -> HashSet<u64> {
    std::fs::read_to_string(manifest)
        .ok()
        .and_then(|text| manifest_names(&text).ok())
        .map(|names| names.iter().map(|n| name_key(n)).collect())
        .unwrap_or_default()
}

/// One translated read-set: the packs it touches, in ascending id order, and what they weigh.
struct PackSet {
    /// `[pack_id, size_wire]` pairs — the shape `web/packs.js` iterates.
    packs: Vec<(u32, u32)>,
    /// What the prefetch actually downloads: every touched pack, whole.
    bytes: u64,
    /// What the read set alone weighs, as **served** bytes: each touched pack's wire size scaled by
    /// the fraction of its body the wanted frames occupy.
    ///
    /// The scaling is not a nicety. A frame's `frame_len` is its length inside the pack's *decoded*
    /// body, and in a SOLID pack the frames are stored while the body around them is one zstd
    /// stream — so summing `frame_len` against `size_wire` compares uncompressed bytes to
    /// compressed ones and can report an over-fetch below 1, which is nonsense. Attributing each
    /// pack's wire bytes in proportion to the body it holds compares like with like, and a pack
    /// every one of whose frames is wanted contributes exactly its wire size.
    ///
    /// `bytes / needed` is then the **over-fetch ratio**, the one number that says whether the
    /// grouping is working for this read set. Near 1 means the set's names really do live together;
    /// a large one means the prefetch is paying for neighbours it will not read.
    needed: u64,
    names: usize,
    unresolved: usize,
}

/// Translate the checked-in read-set manifests into `boot-packs.json` beside the container.
pub fn write_boot_packs(
    out: &Path,
    index: &Index,
    boot_manifest: &Path,
    world_manifest: &Path,
    glue_manifest: &Path,
) -> Result<()> {
    let boot = translate(index, boot_manifest)?;
    let world = translate(index, world_manifest)?;
    let listfile = std::fs::read_to_string(out.join("listfile.txt")).unwrap_or_default();
    let mut glue = match std::fs::read_to_string(glue_manifest) {
        Ok(text) => {
            let names = glue_names(&text, &listfile).with_context(|| {
                format!(
                    "parsing {} — expected {{\"prefixes\": [...]}}",
                    glue_manifest.display()
                )
            })?;
            resolve(index, &names)
        }
        Err(_) => {
            eprintln!(
                "warning: no {} — the page will prefetch no glue set",
                glue_manifest.display()
            );
            resolve(index, &[])
        }
    };
    // The boot set is warmed, and resident, before the glue set starts.
    let in_boot: HashSet<u32> = boot.packs.iter().map(|(id, _)| *id).collect();
    glue.packs.retain(|(id, _)| !in_boot.contains(id));
    glue.bytes = glue.packs.iter().map(|(_, wire)| u64::from(*wire)).sum();
    for (label, set) in [("boot", &boot), ("world", &world), ("glue", &glue)] {
        eprintln!(
            "{label}-packs: {} names -> {} packs, {:.1} MB fetched for {:.1} MB needed ({:.1}x){}",
            set.names,
            set.packs.len(),
            mb(set.bytes),
            mb(set.needed),
            set.bytes as f64 / set.needed.max(1) as f64,
            if set.unresolved > 0 {
                format!(", {} names not in the container", set.unresolved)
            } else {
                String::new()
            }
        );
    }
    let path = out.join("boot-packs.json");
    std::fs::write(
        &path,
        format!(
            "{{\"version\":1,\"build_id\":\"{:016x}\",\"boot\":{},\"world\":{},\"glue\":{}}}\n",
            index.build_id(),
            set_json(&boot),
            set_json(&world),
            set_json(&glue),
        ),
    )
    .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn set_json(set: &PackSet) -> String {
    let packs: Vec<String> = set
        .packs
        .iter()
        .map(|(id, wire)| format!("[{id},{wire}]"))
        .collect();
    format!(
        "{{\"packs\":[{}],\"bytes\":{},\"names\":{}}}",
        packs.join(","),
        set.bytes,
        set.names
    )
}

/// Read one manifest's `names` array and resolve it to a deduplicated pack set.
fn translate(index: &Index, manifest: &Path) -> Result<PackSet> {
    let Ok(text) = std::fs::read_to_string(manifest) else {
        eprintln!(
            "warning: no {} — the page will prefetch nothing for that stage",
            manifest.display()
        );
        return Ok(PackSet {
            packs: Vec::new(),
            needed: 0,
            bytes: 0,
            names: 0,
            unresolved: 0,
        });
    };
    let names = manifest_names(&text).with_context(|| {
        format!(
            "parsing {} — expected {{\"names\": [...]}}",
            manifest.display()
        )
    })?;
    Ok(resolve(index, &names))
}

/// Resolve names to a deduplicated pack set.
fn resolve(index: &Index, names: &[String]) -> PackSet {
    // Frame bytes wanted per pack, so the wire size can be attributed proportionally below.
    let mut wanted: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut unresolved = 0usize;
    for name in names {
        match index.locate(name_key(name)) {
            Some(at) => *wanted.entry(at.pack_id).or_default() += at.frame_len as u64,
            None => unresolved += 1,
        }
    }
    let ids: BTreeSet<u32> = wanted.keys().copied().collect();
    let mut packs = Vec::with_capacity(ids.len());
    let (mut bytes, mut needed) = (0u64, 0u64);
    for id in ids {
        let Some(meta) = index.pack(id) else { continue };
        bytes += meta.size_wire as u64;
        let hit = wanted[&id].min(meta.size_raw as u64);
        needed += meta.size_wire as u64 * hit / (meta.size_raw as u64).max(1);
        packs.push((id, meta.size_wire));
    }
    PackSet {
        packs,
        bytes,
        needed,
        names: names.len(),
        unresolved,
    }
}

/// The glue manifest's names: every listfile name under one of its `prefixes` and under none of
/// its `exclude`s, compared case-insensitively, the chain's own rule for a name.
fn glue_names(manifest: &str, listfile: &str) -> Result<Vec<String>> {
    let lower =
        |v: Vec<String>| -> Vec<String> { v.iter().map(|p| p.to_ascii_lowercase()).collect() };
    let prefixes = lower(manifest_array(manifest, "prefixes")?);
    let exclude = lower(manifest_array(manifest, "exclude").unwrap_or_default());
    Ok(listfile
        .lines()
        .map(str::trim)
        .filter(|name| !name.is_empty() && !name.starts_with('('))
        .filter(|name| {
            let n = name.to_ascii_lowercase();
            prefixes.iter().any(|p| n.starts_with(p)) && !exclude.iter().any(|p| n.starts_with(p))
        })
        .map(str::to_string)
        .collect())
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// The `names` array out of a manifest, without taking a JSON dependency for one array of strings.
///
/// The manifests are machine-written (captured from `?boottrace=1` and pasted in) and already
/// guarded for shape by `benilla-app/tests/boot_manifest.rs` — no forward slashes, no percent
/// encoding, no duplicates — so what is needed here is the string extraction, not a parser. The
/// one real escape a chain name carries is `\\`, since names are backslash-separated.
fn manifest_names(text: &str) -> Result<Vec<String>> {
    manifest_array(text, "names")
}

/// One string array out of a manifest, by key.
fn manifest_array(text: &str, key: &str) -> Result<Vec<String>> {
    let quoted = format!("\"{key}\"");
    let start = text
        .find(&quoted)
        .and_then(|i| text[i..].find('[').map(|j| i + j + 1))
        .with_context(|| format!("no {quoted} array"))?;
    let end = text[start..]
        .find(']')
        .map(|j| start + j)
        .with_context(|| format!("unterminated {quoted} array"))?;
    let mut out = Vec::new();
    let mut chars = text[start..end].chars();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut name = String::new();
        loop {
            match chars.next() {
                Some('\\') => name.push(chars.next().context("manifest ends inside an escape")?),
                Some('"') | None => break,
                Some(c) => name.push(c),
            }
        }
        out.push(name);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_names_reads_backslash_separated_chain_names() {
        let text =
            r#"{"version":1,"names":["DBFilesClient\\Map.dbc","Interface\\Glues\\a b.blp"]}"#;
        assert_eq!(
            manifest_names(text).unwrap(),
            vec![
                "DBFilesClient\\Map.dbc".to_string(),
                "Interface\\Glues\\a b.blp".to_string()
            ]
        );
    }

    #[test]
    fn the_glue_set_is_every_listfile_name_under_a_prefix_less_the_exclusions() {
        let manifest = r#"{"version":1,"prefixes":["Interface\\Glues\\","character\\"],
            "exclude":["Interface\\Glues\\Credits\\"]}"#;
        let listfile = "(listfile)\nCharacter\\Human\\Male\\HumanMale.m2\n\
                        Interface\\GLUES\\Login\\a.blp\nInterface\\Glues\\Credits\\b.blp\n\
                        World\\c.adt\n";
        assert_eq!(
            glue_names(manifest, listfile).unwrap(),
            vec![
                "Character\\Human\\Male\\HumanMale.m2".to_string(),
                "Interface\\GLUES\\Login\\a.blp".to_string()
            ]
        );
    }

    #[test]
    fn a_manifest_without_a_names_array_is_an_error_not_an_empty_set() {
        assert!(manifest_names(r#"{"version":1}"#).is_err());
    }
}
