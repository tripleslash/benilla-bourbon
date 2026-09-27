//! `benilla-pack merge` — **the updated-character-model layer**: an HD character archive
//! (`patch-A.mpq`) with a later archive's files under one prefix laid over it wherever they improve
//! on it (Project Reforged's `patch-T.mpq`, its `Character\` textures), as the key list and the
//! archive pins `build --only … --pins …` packs.
//!
//! An improvement is measured, not assumed. A texture both archives ship is taken from the later
//! one only at exactly twice the earlier one's size: the same art at the compositor's 2×
//! (`patch-T` states the factor itself, `VanillaHelpers\ResizeCharacterSkin.txt` = `2`), where the
//! earlier copy would otherwise be resampled up. At the same size or smaller the later copy is no
//! improvement and the earlier one stays; a name only the later archive has is taken, unless this
//! client's readers refuse it. Every composite input of the result that is not at 2× goes to
//! `compose/` with its 2× size in `compose.tsv`, as `overlay` writes them; model-bound art keeps
//! its own size.
//!
//! ```text
//! benilla-pack merge --base "C:\TurtleWoW-1.18.1\Data\patch-A.mpq" \
//!     --over "C:\TurtleWoW-1.18.1\Data\patch-T.mpq" --out target/hd-merge
//! python scripts/upscale-hd-textures.py target/hd-merge/compose --targets target/hd-merge/compose.tsv
//! benilla-pack build --data "C:\TurtleWoW-1.18.1\Data\patch-A.mpq" \
//!     --data "C:\TurtleWoW-1.18.1\Data\patch-T.mpq" --only target/hd-merge/merge.tsv \
//!     --pins target/hd-merge/merge.tsv --replace-from target/hd-merge/compose \
//!     --require-size target/hd-merge/compose.tsv --out web/packs-hd
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use benilla_formats::Chain;
use benilla_mpq::{name_key, Archive};
use clap::Parser;
use rayon::prelude::*;

use crate::overlay::{composite_refs, composite_target, write_compose_targets, COMPOSE_DIR};

/// The key list with its archive pins that `merge` writes and `build --only --pins` reads.
pub const MERGE_LIST: &str = "merge.tsv";

#[derive(Parser)]
pub struct MergeArgs {
    /// The archive every key of the layer starts from (`patch-A.mpq`).
    #[arg(long)]
    pub base: PathBuf,
    /// The archive laid over it where it improves on it (`patch-T.mpq`).
    #[arg(long)]
    pub over: PathBuf,
    /// Only the over archive's names under this prefix are considered, any case.
    #[arg(long, default_value = "Character\\")]
    pub prefix: String,
    /// Where `merge.tsv`, `compose/` and `compose.tsv` go. Created if absent.
    #[arg(long)]
    pub out: PathBuf,
}

/// Why a key of the over archive is or is not in the layer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Pick {
    /// The base lacks it.
    Added,
    /// Exactly twice the base copy's size.
    Improved,
    /// The same bytes as the base copy.
    Identical,
    /// Both ship it, and the over copy is not twice the base's.
    NotImproved,
    /// The base lacks it and this client's readers refuse it.
    Unreadable,
}

/// A composite input to resample: its name, its size, and the 2× size it takes.
type Rescale = (String, (u32, u32), (u32, u32));

/// One archive entry's bytes: by block, or by name for an entry encrypted under its name.
fn read(archive: &Archive, block: u32, name: Option<&String>) -> Result<Vec<u8>> {
    if archive.is_encrypted(block) {
        let name = name.context("encrypted and named in no listfile")?;
        return archive.read_file(name).map_err(|e| anyhow::anyhow!("{e}"));
    }
    archive
        .read_block(block)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub fn merge(a: MergeArgs) -> Result<()> {
    let base = Archive::open(&a.base).with_context(|| format!("opening {}", a.base.display()))?;
    let over = Archive::open(&a.over).with_context(|| format!("opening {}", a.over.display()))?;
    let chain = Chain::open_archives(&[a.base.clone(), a.over.clone()])?;
    let names = crate::name_hints(&chain);
    let (base_name, over_name) = (file_name(&a.base), file_name(&a.over));
    let metadata: HashSet<u64> = crate::overlay::METADATA
        .iter()
        .map(|n| name_key(n))
        .collect();
    let prefix = a.prefix.replace('/', "\\").to_ascii_lowercase();

    let slots = |ar: &Archive| -> HashMap<u64, u32> {
        ar.entries()
            .filter(|e| !e.is_delete_marker && !metadata.contains(&e.key))
            .map(|e| (e.key, e.block_index))
            .collect()
    };
    let base_slots = slots(&base);
    let over_slots: HashMap<u64, u32> = slots(&over)
        .into_iter()
        .filter(|(k, _)| {
            names.get(k).is_some_and(|n| {
                n.replace('/', "\\")
                    .to_ascii_lowercase()
                    .starts_with(&prefix)
            })
        })
        .collect();
    eprintln!(
        "{base_name}: {} keys; {over_name}: {} under {}",
        base_slots.len(),
        over_slots.len(),
        a.prefix
    );

    // The over archive's keys, judged in parallel.
    let mut over_keys: Vec<u64> = over_slots.keys().copied().collect();
    over_keys.sort_unstable();
    let judged: Vec<Result<(u64, Pick, String)>> = over_keys
        .par_iter()
        .map(|&key| {
            let name = names.get(&key);
            let label = name.cloned().unwrap_or_else(|| format!("{key:#018x}"));
            let ob = read(&over, over_slots[&key], name).with_context(|| label.clone())?;
            let Some(&bb) = base_slots.get(&key) else {
                let pick = match crate::overlay::readable(name.map(String::as_str), &ob) {
                    Ok(()) => Pick::Added,
                    Err(_) => Pick::Unreadable,
                };
                return Ok((key, pick, String::new()));
            };
            let bb = read(&base, bb, name).with_context(|| label.clone())?;
            if bb == ob {
                return Ok((key, Pick::Identical, String::new()));
            }
            let dims = |b: &[u8]| benilla_blp::dimensions(b).ok();
            Ok(match (dims(&bb), dims(&ob)) {
                (Some((bw, bh)), Some((ow, oh))) if (ow, oh) == (bw * 2, bh * 2) => {
                    (key, Pick::Improved, String::new())
                }
                (Some((bw, bh)), Some((ow, oh))) => {
                    (key, Pick::NotImproved, format!("{bw}x{bh} -> {ow}x{oh}"))
                }
                _ => (key, Pick::NotImproved, "not a BLP pair".into()),
            })
        })
        .collect();

    let mut pinned: HashMap<u64, String> =
        base_slots.keys().map(|&k| (k, base_name.clone())).collect();
    let mut counts: BTreeMap<Pick, usize> = BTreeMap::new();
    let mut kept_base: BTreeMap<String, usize> = BTreeMap::new();
    for r in judged {
        let (key, pick, why) = r?;
        *counts.entry(pick).or_default() += 1;
        match pick {
            Pick::Added | Pick::Improved => {
                pinned.insert(key, over_name.clone());
            }
            Pick::NotImproved => *kept_base.entry(why).or_default() += 1,
            Pick::Identical | Pick::Unreadable => {}
        }
    }

    // The composite inputs of the result, sized for the 2× atlas: the layer's own CharSections.
    let refs = composite_refs(&[&chain]);
    std::fs::create_dir_all(a.out.join(COMPOSE_DIR))?;
    let mut keys: Vec<u64> = pinned.keys().copied().collect();
    keys.sort_unstable();
    let targets: Vec<Result<Option<Rescale>>> = keys
        .par_iter()
        .map(|&key| {
            let Some(name) = names.get(&key) else {
                return Ok(None);
            };
            let lower = name.to_ascii_lowercase();
            let candidate = lower.ends_with(".blp")
                && (refs.contains_key(&key)
                    || lower.starts_with("item\\texturecomponents\\")
                    || lower.starts_with("textures\\guildemblems\\"));
            if !candidate {
                return Ok(None);
            }
            let (ar, slot) = if pinned[&key] == over_name {
                (&over, over_slots[&key])
            } else {
                (&base, base_slots[&key])
            };
            let bytes = read(ar, slot, Some(name)).with_context(|| name.clone())?;
            let Ok(dims) = benilla_blp::dimensions(&bytes) else {
                return Ok(None);
            };
            let Some(target) = composite_target(name, dims, &refs) else {
                return Ok(None);
            };
            let path = crate::loose_path(&a.out.join(COMPOSE_DIR), name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
            Ok(Some((name.clone(), dims, target)))
        })
        .collect();
    let mut compose = Vec::new();
    let mut sizes: BTreeMap<String, usize> = BTreeMap::new();
    for t in targets {
        if let Some((name, (w, h), (tw, th))) = t? {
            *sizes.entry(format!("{w}x{h} -> {tw}x{th}")).or_default() += 1;
            compose.push((name, (tw, th)));
        }
    }
    compose.sort();
    write_compose_targets(&a.out.join(crate::overlay::COMPOSE_LIST), &compose)?;
    write_pins(&a.out.join(MERGE_LIST), &pinned, &names)?;

    eprintln!("\n{over_name} under {}:", a.prefix);
    for (pick, n) in &counts {
        eprintln!("  {pick:?}: {n}");
    }
    eprintln!("kept from {base_name} (the later copy is no improvement):");
    for (why, n) in &kept_base {
        eprintln!("  {why}: {n}");
    }
    let from_over = pinned.values().filter(|a| **a == over_name).count();
    eprintln!(
        "layer: {} keys, {from_over} from {over_name}; {} composite inputs to rescale:",
        pinned.len(),
        compose.len()
    );
    for (resize, n) in &sizes {
        eprintln!("  {resize}: {n}");
    }
    if counts.get(&Pick::Unreadable).is_some_and(|&n| n > 0) {
        eprintln!("(unreadable names from {over_name} are left out)");
    }
    Ok(())
}

/// `merge.tsv`: every key of the layer and the archive it is read from, `key archive name`.
fn write_pins(
    path: &Path,
    pins: &HashMap<u64, String>,
    names: &HashMap<u64, String>,
) -> Result<()> {
    let mut rows: Vec<(&u64, &String)> = pins.iter().collect();
    rows.sort_unstable();
    let mut text = String::from("# key\tarchive\tname\n");
    for (key, archive) in rows {
        let name = names.get(key).map_or("", String::as_str);
        text.push_str(&format!("{key:#018x}\t{archive}\t{name}\n"));
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// `merge.tsv` read back: the archive each key is read from — `build --pins`.
pub fn read_pins(path: &Path) -> Result<HashMap<u64, String>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            let (Some(key), Some(archive)) = (f.next(), f.next()) else {
                bail!("bad line in {}: {l}", path.display());
            };
            let key = u64::from_str_radix(key.trim().trim_start_matches("0x"), 16)
                .with_context(|| format!("bad key in {}: {l}", path.display()))?;
            Ok((key, archive.trim().to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pin_list_round_trips() {
        let dir = std::env::temp_dir().join(format!("benilla_merge_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(MERGE_LIST);
        let pins = HashMap::from([(7u64, "patch-A.mpq".to_string()), (9, "patch-T.mpq".into())]);
        let names = HashMap::from([(9u64, r"Character\Human\Male\HumanMaleSkin00_00.blp".into())]);
        write_pins(&path, &pins, &names).unwrap();
        assert_eq!(read_pins(&path).unwrap(), pins);
        // The key list `build --only` reads is the same file's first column.
        assert_eq!(
            crate::overlay::read_list(&path).unwrap(),
            HashSet::from([7, 9])
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
