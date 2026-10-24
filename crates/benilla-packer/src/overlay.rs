//! `benilla-pack overlay` — **a layer over the 1.12.1 corpus cut from a Turtle WoW install's
//! patches**, as a key list the `build` step packs into its own container. Two layers are cut this
//! way: the Turtle WoW 1.18.1 layer (`patch-3` to `patch-9`) and the reforged art layer (the
//! Project Reforged letter patches, `patch-B` to `patch-Y`).
//!
//! The Turtle layer is served at `/data-turtle` and mounted over the 1.12.1 corpus only while the
//! client is on a realm that advertises 1.18.1, so it carries nothing that corpus already answers
//! the same way. A key is kept iff the bytes the layer's patches resolve it to differ **both** from
//! what the install's own lower archives hold for it **and** from what the 1.12.1 chain holds. The
//! first comparison drops a stock file a patch merely re-ships; the second keeps the base
//! container's upscaled character art from being displaced by a byte-identical original.
//!
//! The chain is read **as it stood at the layer's top patch**: the reference mounts `patch-?.MPQ`
//! by name, so the letter patches sit above the numbered ones, and a Turtle layer cut from an
//! install that also holds them must neither count them as lower archives nor read their bytes.
//!
//! The reforged layer mounts above the Turtle layer (and above the HD layer, `patch-A`, when that is
//! on), as its letters do in the reference's order. `--over` names those containers: a key one of
//! them holds that the layer ships at the 1.12.1 bytes is kept, since dropping it would let that
//! container answer where the reforged patch wins. A model this client's M2 reader cannot parse
//! (a newer format than 1.12's) is left out, so the art below it answers instead of nothing.
//!
//! The one transformation a layer needs is the one the HD layer needs: the body compositor tiles
//! at 2× (`benilla_formats::characters::sections`), and copies texel for texel, so every composite
//! input must be exactly twice its reference width. Which files are composite inputs is the
//! install's own `CharSections.dbc` (base skins at 256 wide, head strips and underwear at 128) plus
//! the `Item\TextureComponents\` regions and guild emblem halves (128). Every kept one at any other
//! size is written to `<out>/compose/` with its 2× size in `<out>/compose.tsv`, for
//! `scripts/upscale-hd-textures.py --targets` to resample in place, and `build --replace-from
//! --require-size` takes them from there. Model-bound art is left at its own size.
//!
//! The HD layer mounts above the Turtle layer, and its copy of a table answers for every row. With
//! `--hd`, a table whose HD copy lacks Turtle rows gets a row-merged copy in `<out>/add/`, under
//! [`benilla_formats::layers::TURTLE_HD_ALIAS_PREFIX`], for the client to serve while both layers
//! are mounted; `build --add-from` packs it.
//!
//! ```text
//! benilla-pack overlay --turtle-data "C:\TurtleWoW-1.18.1\Data" \
//!     --base-data "C:\WoW-1.12.1\Data" \
//!     --hd "C:\WoW-1.12.1\Data\patch-A.mpq" --out target/turtle-overlay [--audit]
//! python scripts/upscale-hd-textures.py target/turtle-overlay/compose --targets target/turtle-overlay/compose.tsv
//! benilla-pack build --data "C:\TurtleWoW-1.18.1\Data" --top patch-9.mpq \
//!     --only target/turtle-overlay/overlay.tsv --replace-from target/turtle-overlay/compose \
//!     --require-size target/turtle-overlay/compose.tsv --add-from target/turtle-overlay/add \
//!     --out web/packs-turtle
//!
//! benilla-pack overlay --turtle-data "C:\TurtleWoW-1.18.1\Data" --patches B,C,D,E,G,M,P,S,T,Y \
//!     --base-data "C:\WoW-1.12.1\Data" --over web/packs-turtle --over web/packs-hd \
//!     --out target/reforged-overlay
//! python scripts/upscale-hd-textures.py target/reforged-overlay/compose --targets target/reforged-overlay/compose.tsv
//! benilla-pack build --data "C:\TurtleWoW-1.18.1\Data" --only target/reforged-overlay/overlay.tsv \
//!     --replace-from target/reforged-overlay/compose --require-size target/reforged-overlay/compose.tsv \
//!     --target-pack-bytes 4194304 --out web/packs-reforged
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use benilla_formats::Chain;
use benilla_mpq::{name_key, Archive};
use clap::Parser;
use rayon::prelude::*;

/// The key list `overlay` writes and `build --only` reads.
pub const OVERLAY_LIST: &str = "overlay.tsv";
/// The loose tree of composite inputs not at 2×.
pub const COMPOSE_DIR: &str = "compose";
/// The size each file in [`COMPOSE_DIR`] is resampled to: `name width height` per line.
pub const COMPOSE_LIST: &str = "compose.tsv";
/// The loose tree of tables merged with the HD layer's, under their alias names.
pub const ADD_DIR: &str = "add";

/// Archive bookkeeping entries: never game content, never packed.
pub(crate) const METADATA: [&str; 3] = ["(listfile)", "(attributes)", "(signature)"];

#[derive(Parser)]
pub struct OverlayArgs {
    /// The Turtle WoW 1.18.1 install's `Data` directory.
    #[arg(long)]
    pub turtle_data: PathBuf,
    /// The 1.12.1 `Data` directory the base container was built from.
    #[arg(long)]
    pub base_data: PathBuf,
    /// The numbered patches that form the layer: `patch-<n>.mpq` for each.
    #[arg(long, value_delimiter = ',', default_value = "3,4,5,6,7,8,9")]
    pub patches: Vec<String>,
    /// Where `overlay.tsv` and `compose/` go. Created if absent.
    #[arg(long)]
    pub out: PathBuf,
    /// Also report what the layer leaves out and what it collides with: Turtle content in its lower
    /// archives, DBC layouts against 1.12.1's, and (with `--hd`) the HD layer's DBC row coverage.
    #[arg(long)]
    pub audit: bool,
    /// The HD layer's archive (`patch-A.mpq`): its tables that lack Turtle rows get a row-merged
    /// copy in `add/`, and with `--audit` the row coverage of every table it shares is reported.
    #[arg(long)]
    pub hd: Option<PathBuf>,
    /// A container this layer mounts above besides the 1.12.1 corpus (a directory holding its
    /// `index.bpi`): a key it holds that the layer ships at the 1.12.1 bytes is kept.
    #[arg(long, value_name = "DIR")]
    pub over: Vec<PathBuf>,
    /// Leave out every key whose name starts with this, any case — a tree of the layer's patches
    /// another layer carries (`patch-T`'s `Character\` textures, in the HD layer).
    #[arg(long, value_name = "PREFIX")]
    pub leave_out_prefix: Vec<String>,
    /// Leave out every key this archive holds: a patch the reference mounts above the layer's that
    /// is served from another container (`patch-Y`, laid into the corpus), whose copy must answer.
    #[arg(long, value_name = "MPQ")]
    pub under: Vec<PathBuf>,
}

/// One archive entry a composite resolved a key to.
#[derive(Clone, Copy)]
struct Slot {
    archive: usize,
    block: u32,
    size: u32,
    tomb: bool,
}

/// The composite of `archives` (ascending priority, later wins), tombstones kept so a caller can
/// tell "deleted" from "absent". The same resolution `crate::resolve` performs for a build.
fn composite<'a>(archives: impl Iterator<Item = (usize, &'a Archive)>) -> HashMap<u64, Slot> {
    let mut out = HashMap::new();
    for (i, archive) in archives {
        for e in archive.entries() {
            out.insert(
                e.key,
                Slot {
                    archive: i,
                    block: e.block_index,
                    size: e.file_size,
                    tomb: e.is_delete_marker,
                },
            );
        }
    }
    out
}

/// A slot's bytes: by block index, or by name for an entry encrypted under its name.
fn read_slot(archives: &[Archive], slot: Slot, name: Option<&String>) -> Result<Vec<u8>> {
    let archive = &archives[slot.archive];
    if archive.is_encrypted(slot.block) {
        let name = name.context("encrypted and named in no listfile")?;
        return archive.read_file(name).map_err(|e| anyhow::anyhow!("{e}"));
    }
    archive
        .read_block(slot.block)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Whether an archive file name is one of the layer's numbered patches.
fn is_layer_patch(file: &str, patches: &[String]) -> bool {
    let lower = file.to_ascii_lowercase();
    patches
        .iter()
        .any(|n| lower == format!("patch-{}.mpq", n.to_ascii_lowercase()))
}

/// Why a key of the layer's patches is in the layer, or why not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    /// Neither Turtle's lower archives nor the 1.12.1 chain hold it.
    New,
    /// Held below, with different bytes.
    Changed,
    /// Byte-identical to Turtle's own lower archives: a re-shipped stock file.
    SameAsLower,
    /// Byte-identical to the 1.12.1 chain.
    SameAsBase,
    /// New or changed, but not a file this client's readers take ([`readable`]).
    Unreadable,
}

/// What a source below the layer holds for a key.
enum Below {
    Absent,
    /// Present at another size: different without a read.
    OtherSize,
    Bytes(Vec<u8>),
}

impl Below {
    fn holds(&self, bytes: &[u8]) -> bool {
        matches!(self, Below::Bytes(b) if b == bytes)
    }
}

/// The layer decision for one key, from what the three sources resolve it to.
fn verdict(layer: &[u8], lower: &Below, base: &Below) -> Verdict {
    if lower.holds(layer) {
        return Verdict::SameAsLower;
    }
    if base.holds(layer) {
        return Verdict::SameAsBase;
    }
    if matches!((lower, base), (Below::Absent, Below::Absent)) {
        Verdict::New
    } else {
        Verdict::Changed
    }
}

/// One kept key, as `overlay.tsv` records it.
struct Row {
    key: u64,
    verdict: Verdict,
    /// `0` not a composite input, `1` one already at 2×, `2` one at another size (in `compose/`,
    /// its 2× size in `compose.tsv`).
    compose: u8,
    dims: Option<(u32, u32)>,
    /// The 2× size a `compose == 2` file is resampled to.
    target: Option<(u32, u32)>,
    name: Option<String>,
}

/// Whether a key's bytes are a file this client's readers take, by its extension: a model the M2
/// reader or a WMO the group walk would refuse, or a texture the BLP decoder would, is not layered
/// over art that works. Other kinds pass unread.
pub(crate) fn readable(name: Option<&str>, bytes: &[u8]) -> std::result::Result<(), String> {
    let Some(name) = name else { return Ok(()) };
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".m2") {
        return benilla_m2::parse_m2(&mut std::io::Cursor::new(bytes))
            .map(|_| ())
            .map_err(|e| e.to_string());
    }
    if lower.ends_with(".wmo") {
        // Root and group alike open with `MVER` (stored reversed) at 1.12's version, 17.
        let version = bytes
            .get(8..12)
            .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]));
        return match (bytes.get(..4), version) {
            (Some(b"REVM"), Some(17)) => Ok(()),
            (_, v) => Err(format!("not a version-17 WMO ({v:?})")),
        };
    }
    if lower.ends_with(".blp") {
        return benilla_blp::decode_native(bytes)
            .map(|_| ())
            .map_err(|e| e.to_string());
    }
    Ok(())
}

/// The width each body-composite input has at the reference's 1×, by name key: `CharSections.dbc`'s
/// base skins at 256 (the whole partition, `0x475c50`) and its head strips (face, facial hair, the
/// hair row's scalp columns) and underwear at 128, the width of their tiles. Every table `chains`
/// answers counts, so a layer under another's `CharSections` still sizes that one's names.
pub(crate) fn composite_refs(chains: &[&Chain]) -> HashMap<u64, u32> {
    // `(sectionType, column, 1× width)`: 0 skin, 1 face, 2 facial hair, 3 hair, 4 underwear. The
    // skin's second column and the hair row's first are bound to the model uncomposited.
    const COLUMNS: [(u32, usize, u32); 9] = [
        (0, 0, 256),
        (1, 0, 128),
        (1, 1, 128),
        (2, 0, 128),
        (2, 1, 128),
        (3, 1, 128),
        (3, 2, 128),
        (4, 0, 128),
        (4, 1, 128),
    ];
    let mut out = HashMap::new();
    for chain in chains {
        let Ok(bytes) = chain.read(r"DBFilesClient\CharSections.dbc") else {
            continue;
        };
        let Some((records, fields, size)) = dbc_header(&bytes) else {
            continue;
        };
        if fields != 10 || size != 40 {
            continue;
        }
        let strings = 20 + (records * size) as usize;
        let word = |r: usize, f: usize| {
            let at = 20 + r * size as usize + f * 4;
            bytes
                .get(at..at + 4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        };
        for r in 0..records as usize {
            let Some(section) = word(r, 3) else { continue };
            for &(ty, col, width) in &COLUMNS {
                if ty != section {
                    continue;
                }
                let Some(off) = word(r, 6 + col) else {
                    continue;
                };
                let start = strings + off as usize;
                let Some(tail) = bytes.get(start..) else {
                    continue;
                };
                let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
                let name = String::from_utf8_lossy(&tail[..end]);
                if !name.is_empty() {
                    out.insert(name_key(&name), width);
                }
            }
        }
    }
    out
}

/// The size a composite input has to be for the 2× atlas, when it is not that already: twice its
/// 1× width ([`composite_refs`], and 128 for a region under one of the eight
/// `Item\TextureComponents\` directories the compositor reads or a guild emblem half), its height
/// scaled to match. `None` for a file that is not a composite input or is at 2×.
pub(crate) fn composite_target(
    name: &str,
    (w, h): (u32, u32),
    refs: &HashMap<u64, u32>,
) -> Option<(u32, u32)> {
    let lower = name.replace('/', "\\").to_ascii_lowercase();
    let region = lower
        .strip_prefix("item\\texturecomponents\\")
        .and_then(|rest| rest.split_once('\\'))
        .is_some_and(|(dir, _)| {
            (0..8)
                .filter_map(benilla_formats::equip_tex_dir)
                .any(|d| d.eq_ignore_ascii_case(dir))
        });
    let reference = if region || lower.starts_with("textures\\guildemblems\\") {
        128
    } else {
        *refs.get(&name_key(name))?
    };
    let want = reference * 2;
    if w == want || w == 0 {
        return None;
    }
    Some((want, (h as u64 * want as u64 / w as u64).max(1) as u32))
}

/// The keys every `--over` container's index holds.
fn over_keys(dirs: &[PathBuf]) -> Result<HashSet<u64>> {
    let mut out = HashSet::new();
    for dir in dirs {
        let path = dir.join("index.bpi");
        let index = benilla_pack::Index::parse(
            std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let before = out.len();
        out.extend(index.iter().map(|(k, _)| k));
        eprintln!(
            "--over {}: {} keys ({} new to the set)",
            dir.display(),
            index.len(),
            out.len() - before
        );
    }
    Ok(out)
}

pub fn overlay(a: OverlayArgs) -> Result<()> {
    let mut turtle = Chain::open_layered(&a.turtle_data, false)
        .with_context(|| format!("opening {}", a.turtle_data.display()))?;
    let base = Chain::open_layered(&a.base_data, false)
        .with_context(|| format!("opening {}", a.base_data.display()))?;
    let file_name = |ar: &Archive| {
        ar.path()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    // The install as it stood at the layer's top patch: whatever mounts above it is not this
    // layer's to read, nor below it.
    let Some(top) = turtle
        .archives()
        .iter()
        .rposition(|ar| is_layer_patch(&file_name(ar), &a.patches))
    else {
        bail!(
            "no patch-{{{}}}.mpq in {}",
            a.patches.join(","),
            a.turtle_data.display()
        );
    };
    let top_name = file_name(&turtle.archives()[top]);
    let above = turtle.archives().len() - top - 1;
    turtle.truncate_after(&top_name);
    if above > 0 {
        eprintln!("read through {top_name}: {above} archive(s) mounted above it left out");
    }
    let archives = turtle.archives();
    let (layer_ix, lower_ix): (Vec<usize>, Vec<usize>) =
        (0..archives.len()).partition(|&i| is_layer_patch(&file_name(&archives[i]), &a.patches));
    eprintln!(
        "layer archives: {}",
        layer_ix
            .iter()
            .map(|&i| file_name(&archives[i]))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let layer = composite(layer_ix.iter().map(|&i| (i, &archives[i])));
    let lower = composite(lower_ix.iter().map(|&i| (i, &archives[i])));
    let base_slots = composite(base.archives().iter().enumerate());
    let mut names = crate::name_hints(&turtle);
    for (k, n) in crate::name_hints(&base) {
        names.entry(k).or_insert(n);
    }
    let metadata: HashSet<u64> = METADATA.iter().map(|n| name_key(n)).collect();
    let hd = match &a.hd {
        Some(path) => Some(
            Chain::open_layered(path, true)
                .with_context(|| format!("opening {}", path.display()))?,
        ),
        None => None,
    };
    let refs = composite_refs(
        &[Some(&turtle), hd.as_ref()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    );
    eprintln!("{} composite inputs named by CharSections.dbc", refs.len());
    let over = over_keys(&a.over)?;

    // Keys another container answers for this layer: a later patch's, or a tree carried elsewhere.
    let mut above: HashSet<u64> = HashSet::new();
    for path in &a.under {
        let ar = Archive::open(path).with_context(|| format!("opening {}", path.display()))?;
        above.extend(ar.entries().filter(|e| !e.is_delete_marker).map(|e| e.key));
    }
    let prefixes: Vec<String> = a
        .leave_out_prefix
        .iter()
        .map(|p| p.replace('/', "\\").to_ascii_lowercase())
        .collect();
    let left_out = |k: &u64| {
        above.contains(k)
            || names.get(k).is_some_and(|n| {
                let n = n.replace('/', "\\").to_ascii_lowercase();
                prefixes.iter().any(|p| n.starts_with(p.as_str()))
            })
    };
    let mut keys: Vec<u64> = layer
        .iter()
        .filter(|(k, s)| !s.tomb && !metadata.contains(k))
        .map(|(k, _)| *k)
        .collect();
    let before = keys.len();
    keys.retain(|k| !left_out(k));
    if keys.len() != before {
        eprintln!(
            "{} layer keys left out (--under {} archive(s), --leave-out-prefix {:?})",
            before - keys.len(),
            a.under.len(),
            a.leave_out_prefix
        );
    }
    keys.sort_unstable();
    let tombs: Vec<u64> = layer
        .iter()
        .filter(|(_, s)| s.tomb)
        .map(|(k, _)| *k)
        .collect();
    eprintln!(
        "{} layer keys ({} named), {} tombstones",
        keys.len(),
        keys.iter().filter(|k| names.contains_key(k)).count(),
        tombs.len()
    );

    std::fs::create_dir_all(a.out.join(COMPOSE_DIR))?;
    let base_archives = base.archives();
    let judged: Vec<Result<(Verdict, Row, Option<String>)>> = keys
        .par_iter()
        .map(|&key| {
            let name = names.get(&key);
            let slot = layer[&key];
            let bytes = read_slot(archives, slot, name).with_context(|| {
                format!("{key:#018x} {}", name.map_or("(unnamed)", |n| n.as_str()))
            })?;
            // Size first: most keys differ there, and an equal-size pair is the only one worth a
            // second read.
            let below = |slots: &HashMap<u64, Slot>, ars: &[Archive]| -> Result<Below> {
                Ok(match slots.get(&key) {
                    Some(s) if !s.tomb && s.size == slot.size => {
                        Below::Bytes(read_slot(ars, *s, name)?)
                    }
                    Some(s) if !s.tomb => Below::OtherSize,
                    _ => Below::Absent,
                })
            };
            let mut v = verdict(
                &bytes,
                &below(&lower, archives)?,
                &below(&base_slots, base_archives)?,
            );
            // A container between the corpus and this layer would answer instead of the 1.12.1
            // bytes this layer re-ships.
            if v == Verdict::SameAsBase && over.contains(&key) {
                v = Verdict::Changed;
            }
            let mut why = None;
            if matches!(v, Verdict::New | Verdict::Changed) {
                if let Err(e) = readable(name.map(String::as_str), &bytes) {
                    v = Verdict::Unreadable;
                    why = Some(e);
                }
            }
            let dims = name
                .filter(|n| n.to_ascii_lowercase().ends_with(".blp"))
                .and_then(|_| benilla_blp::dimensions(&bytes).ok());
            let target = name
                .zip(dims)
                .and_then(|(n, d)| composite_target(n, d, &refs));
            let compose = match (name, target) {
                (Some(_), Some(_)) => 2,
                (Some(n), None) if dims.is_some() && is_composite_dir(n) => 1,
                _ => 0,
            };
            if compose == 2 && matches!(v, Verdict::New | Verdict::Changed) {
                let path = crate::loose_path(&a.out.join(COMPOSE_DIR), name.expect("named"));
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, &bytes)
                    .with_context(|| format!("writing {}", path.display()))?;
            }
            Ok((
                v,
                Row {
                    key,
                    verdict: v,
                    compose,
                    dims,
                    target,
                    name: name.cloned(),
                },
                why,
            ))
        })
        .collect();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut unreadable = Vec::new();
    for r in judged {
        match r {
            Ok((v, row, why)) => {
                *counts.entry(verdict_label(v)).or_default() += 1;
                if let Some(why) = why {
                    unreadable.push(format!(
                        "{}: {why}",
                        row.name.as_deref().unwrap_or("(unnamed)")
                    ));
                }
                if matches!(v, Verdict::New | Verdict::Changed) {
                    rows.push(row);
                }
            }
            Err(e) => failures.push(format!("{e:#}")),
        }
    }
    write_list(&a.out.join(OVERLAY_LIST), &rows)?;
    write_compose_list(&a.out.join(COMPOSE_LIST), &rows)?;
    if !unreadable.is_empty() {
        unreadable.sort();
        eprintln!(
            "\n{} layer files this client cannot read, left out so the art below answers:",
            unreadable.len()
        );
        for u in &unreadable {
            eprintln!("  {u}");
        }
    }

    let kept_bytes: u64 = rows.iter().map(|r| layer[&r.key].size as u64).sum();
    eprintln!("\nverdicts:");
    for (label, n) in &counts {
        eprintln!("  {label:<14} {n}");
    }
    eprintln!(
        "kept {} keys, {:.2} GiB uncompressed -> {}",
        rows.len(),
        crate::gib(kept_bytes),
        a.out.join(OVERLAY_LIST).display()
    );
    report_kept(&rows);
    if !tombs.is_empty() {
        eprintln!(
            "\n{} tombstones in the layer's patches — the container cannot delete a base name:",
            tombs.len()
        );
        for k in tombs.iter().take(12) {
            eprintln!("  {}", names.get(k).map_or("(unnamed)", |n| n.as_str()));
        }
    }
    if !failures.is_empty() {
        eprintln!("\n{} keys unreadable:", failures.len());
        for f in failures.iter().take(12) {
            eprintln!("  {f}");
        }
        bail!("{} layer keys could not be read", failures.len());
    }

    if let Some(hd) = &hd {
        merge_with_hd(&a.out.join(ADD_DIR), &turtle, hd, &rows)?;
    }
    if a.audit {
        audit(
            &a,
            &turtle,
            &base,
            &lower,
            &base_slots,
            &names,
            &rows,
            &metadata,
        )?;
    }
    Ok(())
}

/// For every kept table the HD layer also carries and lacks Turtle rows of, write the HD rows plus
/// the Turtle-only rows under [`benilla_formats::layers::TURTLE_HD_ALIAS_PREFIX`].
///
/// HD wins a shared id: its copy is the one tuned to its models (its `HelmetGeosetVisData` hides
/// different geosets on 17 of the 20 ids both carry). Only a table without strings is merged — a
/// string block would have to be rebuilt and every offset into it rewritten — and one with strings
/// that needed it is reported, not merged.
fn merge_with_hd(dir: &Path, turtle: &Chain, hd: &Chain, rows: &[Row]) -> Result<()> {
    let kept: HashSet<String> = rows
        .iter()
        .filter_map(|r| r.name.as_deref())
        .filter(|n| n.to_ascii_lowercase().ends_with(".dbc"))
        .map(|n| n.to_ascii_lowercase())
        .collect();
    let mut merged = 0;
    for entry in hd.list()? {
        if !kept.contains(&entry.name.to_ascii_lowercase()) {
            continue;
        }
        let (Ok(h), Ok(t)) = (hd.read(&entry.name), turtle.read(&entry.name)) else {
            continue;
        };
        let Some(bytes) = merge_tables(&h, &t) else {
            continue;
        };
        let bytes = match bytes {
            Ok(b) => b,
            Err(why) => {
                eprintln!("  NOT MERGED {}: {why}", entry.name);
                continue;
            }
        };
        let alias = format!(
            "{}{}",
            benilla_formats::layers::TURTLE_HD_ALIAS_PREFIX,
            entry.name
        );
        let path = crate::loose_path(dir, &alias);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("merged with the HD layer: {alias}");
        merged += 1;
    }
    eprintln!(
        "{merged} tables merged with the HD layer -> {}",
        dir.display()
    );
    Ok(())
}

/// The HD table plus the Turtle-only rows, sorted by id: `None` when HD already has every Turtle
/// id, `Some(Err)` when the merge is needed and cannot be done.
fn merge_tables(hd: &[u8], turtle: &[u8]) -> Option<std::result::Result<Vec<u8>, String>> {
    let (hi, ti) = (dbc_rows(hd)?, dbc_rows(turtle)?);
    let mut extra: Vec<(u32, &[u8])> = ti
        .iter()
        .filter(|(k, _)| !hi.contains_key(k))
        .map(|(k, r)| (*k, *r))
        .collect();
    if extra.is_empty() {
        return None;
    }
    let (Some((_, hf, hs)), Some((_, tf, ts))) = (dbc_header(hd), dbc_header(turtle)) else {
        return Some(Err("not a WDBC file".into()));
    };
    if (hf, hs) != (tf, ts) {
        return Some(Err(format!(
            "layouts differ: HD {hf}/{hs}, Turtle {tf}/{ts}"
        )));
    }
    let strings = |b: &[u8]| u32::from_le_bytes([b[16], b[17], b[18], b[19]]);
    if strings(hd) > 1 || strings(turtle) > 1 {
        return Some(Err(format!(
            "{} Turtle-only rows in a table with strings",
            extra.len()
        )));
    }
    let mut all: Vec<(u32, &[u8])> = hi.into_iter().collect();
    all.append(&mut extra);
    all.sort_unstable_by_key(|(k, _)| *k);
    let mut out = Vec::with_capacity(21 + all.len() * hs as usize);
    out.extend_from_slice(b"WDBC");
    for w in [all.len() as u32, hf, hs, 1] {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for (_, rec) in &all {
        out.extend_from_slice(rec);
    }
    out.push(0); // the one-byte empty string block every string-less table ships
    Some(Ok(out))
}

/// Whether `name` lives where composite inputs do, whatever its scale.
fn is_composite_dir(name: &str) -> bool {
    let lower = name.replace('/', "\\").to_ascii_lowercase();
    lower.starts_with("character\\")
        || lower.starts_with("item\\texturecomponents\\")
        || lower.starts_with("textures\\guildemblems\\")
}

fn verdict_label(v: Verdict) -> &'static str {
    match v {
        Verdict::New => "new",
        Verdict::Changed => "changed",
        Verdict::SameAsLower => "same-as-lower",
        Verdict::SameAsBase => "same-as-base",
        Verdict::Unreadable => "unreadable",
    }
}

/// `overlay.tsv`: one kept key per line, `key class compose width height name`, sorted by key.
fn write_list(path: &Path, rows: &[Row]) -> Result<()> {
    let mut text = String::from("# key\tclass\tcompose\twidth\theight\tname\n");
    for r in rows {
        let (w, h) = r.dims.unwrap_or((0, 0));
        text.push_str(&format!(
            "{:#018x}\t{}\t{}\t{w}\t{h}\t{}\n",
            r.key,
            verdict_label(r.verdict),
            r.compose,
            r.name.as_deref().unwrap_or("")
        ));
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// `compose.tsv`: each kept file in `compose/` and the size it is resampled to, `name width
/// height`, for `scripts/upscale-hd-textures.py --targets` and `build --require-size`.
fn write_compose_list(path: &Path, rows: &[Row]) -> Result<()> {
    let targets: Vec<(String, (u32, u32))> = rows
        .iter()
        .filter_map(|r| Some((r.name.clone()?, r.target?)))
        .collect();
    write_compose_targets(path, &targets)
}

/// `compose.tsv` from `(name, 2× size)` pairs — the one writer `overlay` and `merge` share.
pub(crate) fn write_compose_targets(path: &Path, targets: &[(String, (u32, u32))]) -> Result<()> {
    let mut text = String::from("# name\twidth\theight\n");
    for (name, (w, h)) in targets {
        text.push_str(&format!("{name}\t{w}\t{h}\n"));
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// `compose.tsv` read back: the size each listed name must have, by name key.
pub fn read_compose_list(path: &Path) -> Result<HashMap<u64, (u32, u32)>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            let (Some(name), Some(w), Some(h)) = (f.next(), f.next(), f.next()) else {
                bail!("bad line in {}: {l}", path.display());
            };
            let dim = |s: &str| {
                s.trim()
                    .parse::<u32>()
                    .with_context(|| format!("bad size in {}: {l}", path.display()))
            };
            Ok((name_key(name), (dim(w)?, dim(h)?)))
        })
        .collect()
}

/// The keys `overlay.tsv` lists — `build --only`'s filter.
pub fn read_list(path: &Path) -> Result<HashSet<u64>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let hex = l.split('\t').next().unwrap_or_default();
            u64::from_str_radix(hex.trim_start_matches("0x"), 16)
                .with_context(|| format!("bad key in {}: {l}", path.display()))
        })
        .collect()
}

/// The kept keys by top-level directory, and the composite inputs by scale.
fn report_kept(rows: &[Row]) {
    let mut dirs: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in rows {
        let top = r
            .name
            .as_deref()
            .map(|n| {
                n.split(['\\', '/'])
                    .next()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            })
            .unwrap_or_else(|| "(unnamed)".into());
        let e = dirs.entry(top).or_default();
        match r.verdict {
            Verdict::New => e.0 += 1,
            _ => e.1 += 1,
        }
    }
    eprintln!("\nkept by directory (new / changed):");
    for (d, (n, c)) in &dirs {
        eprintln!("  {d:<24} {n:>6} {c:>6}");
    }
    let rescaled = rows.iter().filter(|r| r.compose == 2).count();
    let at_2x = rows.iter().filter(|r| r.compose == 1).count();
    eprintln!(
        "composite inputs kept: {rescaled} to rescale (written to {COMPOSE_DIR}/, sizes in \
         {COMPOSE_LIST}), {at_2x} composite-directory textures left at their own size"
    );
    // `from WxH -> to WxH` per rescale, counted.
    let mut sizes: BTreeMap<String, usize> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.compose == 2) {
        if let (Some((w, h)), Some((tw, th))) = (r.dims, r.target) {
            *sizes.entry(format!("{w}x{h} -> {tw}x{th}")).or_default() += 1;
        }
    }
    for (resize, n) in &sizes {
        eprintln!("  {resize}: {n}");
    }
}

/// The DBC header words the readers' fixed offset walks depend on: records, fields, record bytes.
fn dbc_header(bytes: &[u8]) -> Option<(u32, u32, u32)> {
    if bytes.len() < 20 || &bytes[0..4] != b"WDBC" {
        return None;
    }
    let w = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    Some((w(4), w(8), w(12)))
}

/// Every record by its first field, the id column of most tables the client reads (a table keyed
/// otherwise, like `CharacterFacialHairStyles` by race, collapses to its distinct first fields).
fn dbc_rows(bytes: &[u8]) -> Option<HashMap<u32, &[u8]>> {
    let (records, _, record_size) = dbc_header(bytes)?;
    let size = record_size as usize;
    let body = bytes.get(20..)?;
    (0..records as usize)
        .map(|r| {
            let rec = body.get(r * size..(r + 1) * size)?;
            let id = u32::from_le_bytes(rec.get(..4)?.try_into().ok()?);
            Some((id, rec))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn audit(
    a: &OverlayArgs,
    turtle: &Chain,
    base: &Chain,
    lower: &HashMap<u64, Slot>,
    base_slots: &HashMap<u64, Slot>,
    names: &HashMap<u64, String>,
    rows: &[Row],
    metadata: &HashSet<u64>,
) -> Result<()> {
    // 1. What Turtle's lower archives hold that the 1.12.1 chain does not answer the same way —
    // content the patch-only rule leaves out, judged by hash, not by listfile (vanilla's
    // texture.MPQ names almost nothing).
    eprintln!("\naudit 1: Turtle's lower archives against 1.12.1 (by hash)");
    let lower_keys: Vec<u64> = lower
        .iter()
        .filter(|(k, s)| !s.tomb && !metadata.contains(k))
        .map(|(k, _)| *k)
        .collect();
    let outcomes: Vec<(u64, Option<Verdict>)> = lower_keys
        .par_iter()
        .map(|&k| {
            let s = lower[&k];
            let v = match base_slots.get(&k) {
                Some(b) if !b.tomb && b.size == s.size => {
                    let name = names.get(&k);
                    match (
                        read_slot(turtle.archives(), s, name),
                        read_slot(base.archives(), *b, name),
                    ) {
                        (Ok(x), Ok(y)) if x == y => None,
                        _ => Some(Verdict::Changed),
                    }
                }
                Some(b) if !b.tomb => Some(Verdict::Changed),
                _ => Some(Verdict::New),
            };
            (k, v)
        })
        .collect();
    let mut by_dir: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (k, v) in &outcomes {
        let Some(v) = v else { continue };
        let top = names
            .get(k)
            .map(|n| {
                n.split(['\\', '/'])
                    .next()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            })
            .unwrap_or_else(|| "(unnamed)".into());
        let e = by_dir.entry(top).or_default();
        if *v == Verdict::New {
            e.0 += 1
        } else {
            e.1 += 1
        }
    }
    eprintln!(
        "  {} lower keys; outside the layer and not answered alike by 1.12.1 (new / changed):",
        lower_keys.len()
    );
    for (d, (n, c)) in &by_dir {
        eprintln!("  {d:<24} {n:>6} {c:>6}");
    }

    // 2. DBC layouts: a kept DBC whose field count or record size moved is read wrong by a fixed
    // offset walk.
    eprintln!("\naudit 2: kept DBCs against 1.12.1 (fields and record bytes)");
    let mut tables = 0;
    for r in rows {
        let Some(name) = r.name.as_deref() else {
            continue;
        };
        if !name.to_ascii_lowercase().ends_with(".dbc") {
            continue;
        }
        tables += 1;
        let t = turtle.read(name).ok().and_then(|b| dbc_header(&b));
        let v = base.read(name).ok().and_then(|b| dbc_header(&b));
        match (t, v) {
            (Some((_, tf, ts)), Some((_, vf, vs))) if tf != vf || ts != vs => eprintln!(
                "  {name:<52} LAYOUT CHANGED: {tf} fields / {ts} bytes, 1.12.1 {vf} / {vs}"
            ),
            (Some((n, f, sz)), None) => {
                eprintln!("  {name:<52} new table: {n} rows, {f} fields / {sz} bytes")
            }
            (None, _) => eprintln!("  {name:<52} not a WDBC file"),
            _ => {}
        }
    }
    eprintln!("  {tables} kept tables; every one not listed keeps 1.12.1's layout");

    // 2b. A kept table answers for every row, so an id the install's lower copy has and the
    // layer's lacks disappears with the layer mounted.
    eprintln!("\naudit 2b: kept DBCs against the install's lower archives (ids the layer drops)");
    for r in rows {
        let Some(name) = r.name.as_deref() else {
            continue;
        };
        if !name.to_ascii_lowercase().ends_with(".dbc") {
            continue;
        }
        let Some(below) = lower.get(&r.key).filter(|s| !s.tomb) else {
            eprintln!("  {name:<52} new table");
            continue;
        };
        let (Ok(l), Ok(t)) = (
            turtle.read(name),
            read_slot(turtle.archives(), *below, Some(&name.to_string())),
        ) else {
            eprintln!("  {name:<52} unreadable");
            continue;
        };
        let (Some(li), Some(ti)) = (dbc_rows(&l), dbc_rows(&t)) else {
            eprintln!("  {name:<52} not a WDBC file");
            continue;
        };
        let mut dropped: Vec<u32> = ti.keys().filter(|k| !li.contains_key(k)).copied().collect();
        dropped.sort_unstable();
        eprintln!(
            "  {name:<52} layer {:>6} ids, lower {:>6}, dropped {:>5} {:?}",
            li.len(),
            ti.len(),
            dropped.len(),
            &dropped[..dropped.len().min(8)]
        );
    }

    // 3. The HD layer's DBCs over Turtle's: mounted above the Turtle layer, HD's copy of a table
    // answers for every row, so a Turtle row it lacks disappears.
    if let Some(hd_path) = &a.hd {
        eprintln!("\naudit 3: HD layer DBCs over the Turtle layer (Turtle rows HD lacks)");
        let hd = Chain::open_layered(hd_path, true)
            .with_context(|| format!("opening {}", hd_path.display()))?;
        for entry in hd.list()? {
            let lower_name = entry.name.to_ascii_lowercase();
            if !lower_name.ends_with(".dbc") {
                continue;
            }
            let (Ok(h), Ok(t)) = (hd.read(&entry.name), turtle.read(&entry.name)) else {
                continue;
            };
            let (Some(hi), Some(ti)) = (dbc_rows(&h), dbc_rows(&t)) else {
                eprintln!("  {:<52} unreadable", entry.name);
                continue;
            };
            let mut missing: Vec<u32> =
                ti.keys().filter(|k| !hi.contains_key(k)).copied().collect();
            missing.sort_unstable();
            // Raw record bytes: exact for a table without strings, an upper bound for one with them,
            // whose string offsets move with its string block.
            let differ = ti
                .iter()
                .filter(|(k, rec)| hi.get(k).is_some_and(|h| h != *rec))
                .count();
            eprintln!(
                "  {:<52} turtle {:>6} ids, hd {:>6}, turtle-only {:>5}, shared-but-different {:>5} {:?}{}",
                entry.name,
                ti.len(),
                hi.len(),
                missing.len(),
                differ,
                &missing[..missing.len().min(8)],
                if dbc_header(&h).map(|x| (x.1, x.2)) != dbc_header(&t).map(|x| (x.1, x.2)) {
                    "  LAYOUT DIFFERS"
                } else {
                    ""
                }
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_key_is_kept_only_when_it_differs_from_both_sources() {
        use Below::*;
        let x: &[u8] = b"turtle";
        let b = |v: &[u8]| Bytes(v.to_vec());
        assert_eq!(verdict(x, &Absent, &Absent), Verdict::New);
        assert_eq!(verdict(x, &b(b"stock"), &Absent), Verdict::Changed);
        assert_eq!(verdict(x, &Absent, &b(b"esrgan")), Verdict::Changed);
        assert_eq!(verdict(x, &b(x), &b(b"esrgan")), Verdict::SameAsLower);
        assert_eq!(verdict(x, &b(b"stock"), &b(x)), Verdict::SameAsBase);
        assert_eq!(verdict(x, &OtherSize, &Absent), Verdict::Changed);
        // An empty layer file is not "the same" as a differently sized one below it.
        assert_eq!(verdict(&[], &OtherSize, &OtherSize), Verdict::Changed);
    }

    /// A composite input is rescaled to twice its 1× width from whatever size it ships at — up
    /// from 1×, down from 4× — and one at 2× or outside the composite is left alone.
    #[test]
    fn composite_inputs_are_sized_to_twice_their_reference_width() {
        let skin = r"Character\Human\Male\HumanMaleSkin00_00.blp";
        let face = r"Character\Human\Male\HumanMaleFaceLower00_00.blp";
        let refs = HashMap::from([(name_key(skin), 256), (name_key(face), 128)]);
        assert_eq!(composite_target(skin, (256, 256), &refs), Some((512, 512)));
        assert_eq!(composite_target(skin, (512, 512), &refs), None);
        assert_eq!(
            composite_target(skin, (1024, 1024), &refs),
            Some((512, 512))
        );
        // A 2× head strip is 256 wide: the partition's own width is not the test for an overlay.
        assert_eq!(composite_target(face, (256, 64), &refs), None);
        assert_eq!(composite_target(face, (128, 32), &refs), Some((256, 64)));
        let arm = r"Item\TextureComponents\ArmUpperTexture\Plate_A_01Silver_Sleeve_AU_U.blp";
        assert_eq!(composite_target(arm, (128, 64), &refs), Some((256, 128)));
        assert_eq!(composite_target(arm, (512, 256), &refs), Some((256, 128)));
        assert_eq!(composite_target(arm, (256, 128), &refs), None);
        // A file beside the region directories is a model's, not a region.
        let loose = r"Item\TextureComponents\helm_dk_a_bluegold.blp";
        assert_eq!(composite_target(loose, (512, 512), &refs), None);
        // Model-bound art under `Character\` (a hair texture) is not a composite input.
        let hair = r"Character\Human\Hair04_07.blp";
        assert_eq!(composite_target(hair, (256, 256), &refs), None);
    }

    /// A model or WMO in a format this client does not read is refused by name; other kinds pass.
    #[test]
    fn unreadable_models_are_refused() {
        assert!(readable(Some(r"World\a.m2"), b"MD20\x08\x01\0\0").is_err());
        let mut wmo = b"REVM\x04\0\0\0".to_vec();
        wmo.extend_from_slice(&17u32.to_le_bytes());
        assert!(readable(Some(r"World\wmo\a.wmo"), &wmo).is_ok());
        wmo[8] = 18;
        assert!(readable(Some(r"World\wmo\a.wmo"), &wmo).is_err());
        assert!(readable(Some(r"Sound\a.wav"), b"RIFF").is_ok());
        assert!(readable(None, b"").is_ok());
    }

    #[test]
    fn only_the_named_numbered_patches_form_the_layer() {
        let p: Vec<String> = ["3", "9"].iter().map(|s| s.to_string()).collect();
        assert!(is_layer_patch("patch-3.mpq", &p));
        assert!(is_layer_patch("PATCH-9.MPQ", &p));
        assert!(!is_layer_patch("patch-2.MPQ", &p));
        assert!(!is_layer_patch("patch.MPQ", &p));
        assert!(!is_layer_patch("patch-30.mpq", &p));
    }

    #[test]
    fn the_list_round_trips_its_keys() {
        let dir = std::env::temp_dir().join(format!("benilla_overlay_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(OVERLAY_LIST);
        let rows = vec![
            Row {
                key: 0x0123_4567_89ab_cdef,
                verdict: Verdict::New,
                compose: 2,
                dims: Some((256, 256)),
                target: Some((512, 512)),
                name: Some("Character\\Goblin\\Male\\GoblinMaleSkin00_00.blp".into()),
            },
            Row {
                key: 7,
                verdict: Verdict::Changed,
                compose: 0,
                dims: None,
                target: None,
                name: None,
            },
        ];
        write_list(&path, &rows).unwrap();
        let keys = read_list(&path).unwrap();
        assert_eq!(keys, HashSet::from([0x0123_4567_89ab_cdef, 7]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A string-less table of `(id, flag)` records.
    fn table(rows: &[(u32, u32)]) -> Vec<u8> {
        let mut dbc = Vec::new();
        dbc.extend_from_slice(b"WDBC");
        for w in [rows.len() as u32, 2, 8, 1] {
            dbc.extend_from_slice(&w.to_le_bytes());
        }
        for (id, flag) in rows {
            dbc.extend_from_slice(&id.to_le_bytes());
            dbc.extend_from_slice(&flag.to_le_bytes());
        }
        dbc.push(0);
        dbc
    }

    /// HD wins a shared id, Turtle's other rows are appended, the result is sorted by id — and a
    /// table HD already covers is left alone.
    #[test]
    fn the_hd_merge_keeps_hd_rows_and_adds_turtle_only_ones() {
        let hd = table(&[(1, 100), (2, 200)]);
        let turtle = table(&[(1, 1), (3, 3), (2, 2)]);
        let merged = merge_tables(&hd, &turtle).unwrap().unwrap();
        assert_eq!(merged, table(&[(1, 100), (2, 200), (3, 3)]));
        assert!(
            merge_tables(&hd, &table(&[(2, 2)])).is_none(),
            "nothing to add"
        );
    }

    #[test]
    fn a_table_with_strings_is_not_merged() {
        let mut turtle = table(&[(1, 1), (3, 3)]);
        turtle[16] = 5; // a 5-byte string block
        turtle.extend_from_slice(b"abcd");
        assert!(merge_tables(&table(&[(1, 1)]), &turtle).unwrap().is_err());
    }

    #[test]
    fn dbc_rows_are_keyed_by_the_first_field_of_each_record() {
        let mut dbc = Vec::new();
        dbc.extend_from_slice(b"WDBC");
        for w in [2u32, 2, 8, 1] {
            dbc.extend_from_slice(&w.to_le_bytes());
        }
        for w in [9u32, 0, 10, 0] {
            dbc.extend_from_slice(&w.to_le_bytes());
        }
        dbc.push(0);
        assert_eq!(dbc_header(&dbc), Some((2, 2, 8)));
        let rows = dbc_rows(&dbc).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[&9], &[9, 0, 0, 0, 0, 0, 0, 0][..]);
        assert_eq!(rows[&10], &[10, 0, 0, 0, 0, 0, 0, 0][..]);
    }
}
