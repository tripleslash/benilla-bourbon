//! `benilla-pack overlay` — **the Turtle WoW 1.18.1 layer**: what a Turtle install adds to or
//! changes over the 1.12.1 corpus, as a key list the `build` step packs into its own container.
//!
//! The layer is served at `/data-turtle` and mounted over the 1.12.1 corpus only while the client is
//! on a realm that advertises 1.18.1, so it carries nothing that corpus already answers the same
//! way. Its source is the numbered patches Turtle ships on top of the stock archives (`patch-3` to
//! `patch-9`), and a key is kept iff the bytes those patches resolve it to differ **both** from what
//! Turtle's own lower archives hold for it **and** from what the 1.12.1 chain holds. The first
//! comparison drops a stock file a patch merely re-ships; the second keeps the base container's
//! upscaled character art from being displaced by a byte-identical original.
//!
//! The one transformation the layer needs is the one the HD layer needs: the body compositor tiles
//! at 2× ([`benilla_formats::composite_input_at_1x`]), and Turtle's new skins and armour regions are
//! authored at 1×. Every kept composite input still at 1× is written to `<out>/compose/` for
//! `scripts/upscale-hd-textures.py` (or any other upscaler) to rewrite in place, and `build
//! --replace-from` takes them from there.
//!
//! The HD layer mounts above this one, and its copy of a table answers for every row. With `--hd`,
//! a table whose HD copy lacks Turtle rows gets a row-merged copy in `<out>/add/`, under
//! [`benilla_formats::layers::TURTLE_HD_ALIAS_PREFIX`], for the client to serve while both layers
//! are mounted; `build --add-from` packs it.
//!
//! ```text
//! benilla-pack overlay --turtle-data "C:\TurtleWoW-1.18.1\Data" \
//!     --base-data "C:\WoW-1.12.1\Data" \
//!     --hd "C:\WoW-1.12.1\Data\patch-A.mpq" --out target/turtle-overlay [--audit]
//! python scripts/upscale-hd-textures.py target/turtle-overlay/compose
//! benilla-pack build --data "C:\TurtleWoW-1.18.1\Data" --only target/turtle-overlay/overlay.tsv \
//!     --replace-from target/turtle-overlay/compose --require-doubled \
//!     --add-from target/turtle-overlay/add --out web/packs-turtle
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
/// The loose tree of composite inputs still at 1×.
pub const COMPOSE_DIR: &str = "compose";
/// The loose tree of tables merged with the HD layer's, under their alias names.
pub const ADD_DIR: &str = "add";

/// Archive bookkeeping entries: never game content, never packed.
const METADATA: [&str; 3] = ["(listfile)", "(attributes)", "(signature)"];

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
    /// `0` not a composite input, `1` one already at 2×, `2` one still at 1× (in `compose/`).
    compose: u8,
    dims: Option<(u32, u32)>,
    name: Option<String>,
}

pub fn overlay(a: OverlayArgs) -> Result<()> {
    let turtle = Chain::open_layered(&a.turtle_data, false)
        .with_context(|| format!("opening {}", a.turtle_data.display()))?;
    let base = Chain::open_layered(&a.base_data, false)
        .with_context(|| format!("opening {}", a.base_data.display()))?;
    let archives = turtle.archives();
    let file_name = |ar: &Archive| {
        ar.path()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let (layer_ix, lower_ix): (Vec<usize>, Vec<usize>) =
        (0..archives.len()).partition(|&i| is_layer_patch(&file_name(&archives[i]), &a.patches));
    if layer_ix.is_empty() {
        bail!(
            "no patch-{{{}}}.mpq in {}",
            a.patches.join(","),
            a.turtle_data.display()
        );
    }
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

    let mut keys: Vec<u64> = layer
        .iter()
        .filter(|(k, s)| !s.tomb && !metadata.contains(k))
        .map(|(k, _)| *k)
        .collect();
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
    let judged: Vec<Result<(Verdict, Row)>> = keys
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
            let v = verdict(
                &bytes,
                &below(&lower, archives)?,
                &below(&base_slots, base_archives)?,
            );
            let dims = name
                .filter(|n| n.to_ascii_lowercase().ends_with(".blp"))
                .and_then(|_| benilla_blp::dimensions(&bytes).ok());
            let compose = match (name, dims) {
                (Some(n), Some((w, _))) if benilla_formats::composite_input_at_1x(n, w) => 2,
                (Some(n), Some(_)) if is_composite_dir(n) => 1,
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
                    name: name.cloned(),
                },
            ))
        })
        .collect();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for r in judged {
        match r {
            Ok((v, row)) => {
                *counts.entry(verdict_label(v)).or_default() += 1;
                if matches!(v, Verdict::New | Verdict::Changed) {
                    rows.push(row);
                }
            }
            Err(e) => failures.push(format!("{e:#}")),
        }
    }
    write_list(&a.out.join(OVERLAY_LIST), &rows)?;

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

    if let Some(hd_path) = &a.hd {
        let hd = Chain::open_layered(hd_path, true)
            .with_context(|| format!("opening {}", hd_path.display()))?;
        merge_with_hd(&a.out.join(ADD_DIR), &turtle, &hd, &rows)?;
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
    let at_1x = rows.iter().filter(|r| r.compose == 2).count();
    let at_2x = rows.iter().filter(|r| r.compose == 1).count();
    eprintln!(
        "composite inputs kept: {at_1x} at 1x (written to {COMPOSE_DIR}/ for the upscale), {at_2x} already at 2x"
    );
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
                name: Some("Character\\Goblin\\Male\\GoblinMaleSkin00_00.blp".into()),
            },
            Row {
                key: 7,
                verdict: Verdict::Changed,
                compose: 0,
                dims: None,
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
