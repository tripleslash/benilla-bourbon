//! Turning labelled blobs into packs, with the object budget as a guarantee.
//!
//! Two constraints and one preference, in that order of authority:
//!
//! 1. **No object over the cap.** Enforced structurally: singletons are routed out by size before
//!    anything is bundled, and a bundle is closed the moment it would exceed the target, which is
//!    orders of magnitude below the cap. The encoder still refuses an over-cap pack as a last line,
//!    so this can never be merely believed.
//! 2. **No more objects than the budget.** If the first pass overshoots, the target is scaled and
//!    the pass re-run. The corpus measures 4,740 objects at the default against a budget of
//!    18,000, so this is a guard rather than a routine step — but it is the guard that makes the
//!    output servable by construction rather than by luck.
//! 3. **Locality, preserved.** Groups are visited in sorted label order and a bundle only ever
//!    holds consecutive blobs, so a pack is a run of adjacent paths. Without the merge step you get
//!    a long tail of 4 KB packs (`ui/interface/buttons` and its dozens of siblings); with it you
//!    get uniform packs that are still runs.

use std::collections::HashMap;

use anyhow::{bail, Result};
use benilla_pack::PackClass;

use crate::group::label_of;
use crate::sidecars::ReadSets;
use crate::Blob;

/// Plan every pack. Returns groups of blobs, in pack-id order.
pub fn plan<'a>(
    blobs: &'a [Blob],
    names: &HashMap<u64, String>,
    read_sets: &ReadSets,
    singleton_bytes: u32,
    target_pack_bytes: u64,
    max_objects: u32,
) -> Result<Vec<Vec<&'a Blob>>> {
    // Label once — it is string work over 79,735 entries and the retry loop must not repeat it.
    let mut labelled: Vec<(String, &Blob)> = blobs
        .iter()
        .map(|b| {
            let natural = label_of(
                names.get(&b.key).map(|s| s.as_str()),
                b.len,
                singleton_bytes,
                b.archive,
                b.block,
            );
            // A recorded read set outranks the path-shaped guess — see `ReadSets`. The natural label
            // is kept as the suffix so ORDER WITHIN the read set is still path order, which is what
            // keeps the boot packs individually coherent instead of hash-shuffled.
            //
            // `@big/` is not overridden: a singleton is about the object-size cap, not locality, and
            // a 15 MiB blob bundled into the boot set would be exactly the over-fetch this is for.
            let label = match read_sets.prefix(b.key) {
                Some(prefix) if !natural.starts_with("@big/") => format!("{prefix}/{natural}"),
                _ => natural,
            };
            (label, b)
        })
        .collect();
    // Sorted label order IS path order for the directory-shaped labels, which is what makes a pack
    // a run of adjacent files. The key tiebreak keeps it deterministic.
    labelled.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.key.cmp(&b.1.key)));

    let mut target = target_pack_bytes.max(1);
    for attempt in 0..8 {
        let packs = bundle(&labelled, target);
        if packs.len() as u32 <= max_objects {
            if attempt > 0 {
                eprintln!("  (target grown to {target} bytes to fit the object budget)");
            }
            return Ok(packs);
        }
        // Overshoot: grow the target proportionally, with headroom so this converges in a couple of
        // rounds rather than creeping.
        target = (target as f64 * (packs.len() as f64 / max_objects as f64) * 1.15) as u64;
    }
    bail!("could not fit the corpus into {max_objects} objects; raise --max-objects or --target-pack-bytes")
}

/// One bundling pass. Blobs are consumed in label order; a `@big/` label always stands alone.
fn bundle<'a>(labelled: &[(String, &'a Blob)], target: u64) -> Vec<Vec<&'a Blob>> {
    let mut packs: Vec<Vec<&Blob>> = Vec::new();
    let mut cur: Vec<&Blob> = Vec::new();
    let mut cur_bytes = 0u64;

    for (label, blob) in labelled {
        if label.starts_with("@big/") {
            if !cur.is_empty() {
                packs.push(std::mem::take(&mut cur));
                cur_bytes = 0;
            }
            packs.push(vec![*blob]);
            continue;
        }
        cur.push(*blob);
        cur_bytes += blob.len as u64;
        if cur_bytes >= target {
            packs.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
    }
    if !cur.is_empty() {
        packs.push(cur);
    }
    packs
}

/// Which layout to ask the encoder for.
///
/// Solid compression only pays where the members are many, small and alike — a directory of icons,
/// the DBC set, an `Interface\` subtree. It costs range-extractability, so it is not asked for
/// otherwise. The encoder measures and falls back on its own if the guess was wrong, so this is a
/// hint, and a wrong hint costs ratio rather than correctness.
pub fn prefer_for(group: &[&Blob], names: &HashMap<u64, String>) -> PackClass {
    if group.len() < 8 {
        return PackClass::Rangeable;
    }
    let mean = group.iter().map(|b| b.len as u64).sum::<u64>() / group.len() as u64;
    if mean > 128 * 1024 {
        return PackClass::Rangeable;
    }
    // Audio is already compressed; a solid stream over MP3 wins nothing and forfeits ranges.
    let audioish = group
        .iter()
        .filter(|b| {
            names
                .get(&b.key)
                .map(|n| {
                    let l = n.to_ascii_lowercase();
                    l.ends_with(".mp3") || l.ends_with(".wav") || l.ends_with(".ogg")
                })
                .unwrap_or(false)
        })
        .count();
    if audioish * 2 > group.len() {
        return PackClass::Rangeable;
    }
    PackClass::Solid
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs(sizes: &[u32]) -> Vec<Blob> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &len)| Blob {
                key: i as u64,
                archive: 0,
                block: i as u32,
                len,
            })
            .collect()
    }

    /// Names drive the labels; with none, everything lands in archive-order buckets, which is still
    /// enough to exercise bundling.
    fn no_names() -> HashMap<u64, String> {
        HashMap::new()
    }

    #[test]
    fn bundles_close_at_the_target_so_packs_stay_uniform() {
        let bs = blobs(&[400_000; 10]);
        let packs = plan(
            &bs,
            &no_names(),
            &ReadSets::default(),
            4 << 20,
            1_000_000,
            18_000,
        )
        .unwrap();
        // 400 KB blobs against a 1 MB target: three per pack.
        assert!(
            packs.iter().all(|p| p.len() <= 3),
            "{:?}",
            packs.iter().map(|p| p.len()).collect::<Vec<_>>()
        );
        assert_eq!(packs.iter().map(|p| p.len()).sum::<usize>(), 10);
    }

    /// A blob over the singleton threshold must get its own object wherever it sits in the order —
    /// this is what stops a Range-less client paying 15 MiB to reach a 4 KB neighbour.
    #[test]
    fn big_blobs_are_alone_in_their_pack() {
        let bs = blobs(&[1000, 5 << 20, 1000]);
        let packs = plan(
            &bs,
            &no_names(),
            &ReadSets::default(),
            4 << 20,
            1 << 20,
            18_000,
        )
        .unwrap();
        let big = packs
            .iter()
            .find(|p| p.iter().any(|b| b.len == 5 << 20))
            .expect("the big blob is somewhere");
        assert_eq!(big.len(), 1, "a singleton must not share its object");
    }

    /// The budget is honoured by growing the target, not by failing — and not by exceeding it.
    #[test]
    fn an_overshooting_plan_grows_its_target_until_it_fits_the_budget() {
        let bs = blobs(&[10_000; 1000]);
        // A tiny target would want ~1000 packs; the budget allows 20.
        let packs = plan(&bs, &no_names(), &ReadSets::default(), 4 << 20, 10_000, 20).unwrap();
        assert!(
            packs.len() <= 20,
            "{} packs against a budget of 20",
            packs.len()
        );
        assert_eq!(packs.iter().map(|p| p.len()).sum::<usize>(), 1000);
    }

    /// Every blob must appear exactly once. A planner that silently dropped or duplicated one would
    /// produce a container that looks fine and is missing files.
    #[test]
    fn planning_is_a_partition_of_the_input() {
        let bs = blobs(&[1, 2, 3, 5 << 20, 100_000, 7, 4 << 20, 9]);
        let packs = plan(
            &bs,
            &no_names(),
            &ReadSets::default(),
            4 << 20,
            50_000,
            18_000,
        )
        .unwrap();
        let mut keys: Vec<u64> = packs.iter().flat_map(|p| p.iter().map(|b| b.key)).collect();
        keys.sort_unstable();
        assert_eq!(keys, (0..bs.len() as u64).collect::<Vec<_>>());
    }

    #[test]
    fn planning_is_deterministic() {
        let bs = blobs(&[1000; 50]);
        let a = plan(
            &bs,
            &no_names(),
            &ReadSets::default(),
            4 << 20,
            8000,
            18_000,
        )
        .unwrap();
        let b = plan(
            &bs,
            &no_names(),
            &ReadSets::default(),
            4 << 20,
            8000,
            18_000,
        )
        .unwrap();
        let shape = |p: &Vec<Vec<&Blob>>| -> Vec<Vec<u64>> {
            p.iter()
                .map(|g| g.iter().map(|b| b.key).collect())
                .collect()
        };
        assert_eq!(shape(&a), shape(&b));
    }

    /// Solid is asked for only where it can pay: many small alike files, and not audio.
    #[test]
    fn solid_is_only_preferred_for_many_small_non_audio_blobs() {
        let small = blobs(&[4000; 40]);
        let refs: Vec<&Blob> = small.iter().collect();
        assert_eq!(prefer_for(&refs, &no_names()), PackClass::Solid);

        let few = blobs(&[4000; 4]);
        let refs: Vec<&Blob> = few.iter().collect();
        assert_eq!(prefer_for(&refs, &no_names()), PackClass::Rangeable);

        let large = blobs(&[900_000; 40]);
        let refs: Vec<&Blob> = large.iter().collect();
        assert_eq!(prefer_for(&refs, &no_names()), PackClass::Rangeable);

        let mut names = HashMap::new();
        for b in &small {
            names.insert(b.key, format!("Sound\\a{}.mp3", b.key));
        }
        let refs: Vec<&Blob> = small.iter().collect();
        assert_eq!(
            prefer_for(&refs, &names),
            PackClass::Rangeable,
            "already-compressed audio must keep its ranges"
        );
    }
}
