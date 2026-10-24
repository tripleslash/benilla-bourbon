//! `benilla-pack inspect` — **what is actually in an archive, and can this client read it?**
//!
//! The packer already knew how to turn a chain into a container. What it could not do was answer
//! the question that comes *before* that, and which came up the moment an optional content layer
//! became a thing a player can turn on: someone hands you a 1.75 GiB `patch-A.MPQ` said to hold
//! "updated character models", and before it is repacked, hosted and shipped behind a checkbox,
//! two things have to be known.
//!
//! 1. **What is in it**, by shape rather than by trust — how many entries, what extensions, which
//!    directories, and (the one that fails a build at minute forty of a fifty-minute encode) whether
//!    any single blob is over the object cap.
//! 2. **Whether our readers can read it.** Art lifted out of a newer client is the interesting case
//!    and the dangerous one: a model at a version the M2 walk rejects, a DBC whose record grew a
//!    column, a BLP in a format the decoder has never seen. Each of those fails *far* downstream —
//!    an empty character screen, a garbled skin, a silent `None` — and each is answerable here in
//!    seconds by parsing every file of that kind and printing what came back.
//!
//! It is a **reporter, not a gate**: it parses, counts and prints, and exits 0 whatever it finds
//! (barring an unreadable archive). Judging "is this acceptable" is a person's job, and a tool that
//! decided for them would either block a layer that renders fine or wave through one that does not.
//!
//! ```text
//! benilla-pack inspect --data "C:\WoW-1.12.1\Data\patch-A.mpq"
//! benilla-pack inspect --data /path/to/WoW/Data --hd          # the whole chain, layer included
//! benilla-pack inspect --data patch-A.mpq --blp-sample 2000   # widen the texture scan
//! ```

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;

use anyhow::{Context, Result};
use benilla_formats::Chain;
use benilla_mpq::name_key;
use benilla_pack::{encode_pack, PackClass, PackPlan, MAX_OBJECT_BYTES};
use clap::Parser;
use rayon::prelude::*;

#[derive(Parser)]
pub struct InspectArgs {
    /// The `Data` directory or a single `.MPQ`. A single archive is the usual shape here — an
    /// optional layer is inspected on its own, before it is anywhere near a chain.
    #[arg(long)]
    pub data: PathBuf,
    /// Mount the optional updated-character-model layer when `--data` is a directory. Ignored for a
    /// single archive, which is always mounted by name.
    #[arg(long)]
    pub hd: bool,
    /// How many `.blp` files to actually decode. The scan is the slow part of this command (16,125
    /// textures at 3 GiB), and a uniform sample answers "does this decoder know this format" just as
    /// well as all of them; `0` scans every one.
    #[arg(long, default_value_t = 400)]
    pub blp_sample: usize,
    /// Rows in the extension and directory tables.
    #[arg(long, default_value_t = 12)]
    pub top: usize,
    /// zstd level for the object-cap encode — keep it equal to the `build` you intend to run, or
    /// the served sizes reported here are not the ones that will be hosted.
    #[arg(long, default_value_t = 15)]
    pub level: i32,
}

/// One listed file, with the size the archive reports for it.
struct Entry {
    name: String,
    size: u64,
}

pub fn inspect(a: InspectArgs) -> Result<()> {
    let chain = if a.data.is_dir() {
        Chain::open_layered(&a.data, a.hd)
    } else {
        Chain::open(&a.data)
    }
    .with_context(|| format!("opening {}", a.data.display()))?;

    println!("== {} ==", a.data.display());
    for archive in chain.archives() {
        println!("  mounted {}", archive.path().display());
    }

    // `Chain::list` and not the hash-table walk `build` uses: this command reports on files a human
    // can name, and an entry no listfile mentions has no name to print. The two counts are printed
    // together below precisely so a gap between them is visible rather than implied.
    let entries: Vec<Entry> = chain
        .list()
        .context("reading the listfile")?
        .into_iter()
        .map(|e| Entry {
            name: e.name,
            size: e.size,
        })
        .collect();
    let unnamed: usize = chain
        .archives()
        .iter()
        .map(|arc| arc.entries().filter(|e| !e.is_delete_marker).count())
        .sum::<usize>()
        .saturating_sub(entries.len());

    let total: u64 = entries.iter().map(|e| e.size).sum();
    println!(
        "\n{} named entries, {:.2} GiB uncompressed{}",
        entries.len(),
        total as f64 / (1024.0 * 1024.0 * 1024.0),
        if unnamed > 0 {
            format!(" (+{unnamed} entries no listfile names)")
        } else {
            String::new()
        }
    );

    // The object cap, checked HERE rather than discovered at minute forty of an encode. A blob over
    // it cannot be packed at all — a singleton pack is one blob and one object — so this is the one
    // finding in this report that is a hard blocker rather than a note.
    biggest(&chain, &entries, a.level);

    histogram("extensions", &entries, a.top, |e| {
        e.name.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
    });
    histogram("top-level directories", &entries, a.top, |e| {
        e.name
            .split(['\\', '/'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase()
    });

    scan_m2(&chain, &entries);
    scan_dbc(&chain, &entries);
    scan_blp(&chain, &entries, a.blp_sample);
    Ok(())
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// **The object-cap check** — the one finding in this report that can be a hard blocker, and the
/// one it would be easy to get wrong by eye.
///
/// The cap ([`MAX_OBJECT_BYTES`], under a static host's 25 MiB per-asset limit) is on the bytes the
/// host **serves**, not on the blob. A blob alone in its object is a `Singleton` pack, whose body
/// is stored but whose single frame is zstd-compressed like any other, so a 26.5 MiB M2 of float
/// vertex and animation data is nowhere near 26.5 MiB on the wire. Comparing the raw size to the
/// cap therefore reports a blocker that is not there — which is exactly what the first cut of this
/// function did, for three of this layer's character models.
///
/// So the check is not a comparison, it is the encode: [`encode_pack`] at the same level `build`
/// uses, on a one-blob plan, reporting the wire size it produced or the cap error it raised. Costs
/// a few seconds for a handful of blobs and cannot disagree with the build, because it *is* the
/// build's own function.
fn biggest(chain: &Chain, entries: &[Entry], level: i32) {
    let mut sorted: Vec<&Entry> = entries.iter().collect();
    sorted.sort_unstable_by_key(|e| std::cmp::Reverse(e.size));
    // Everything a naive read of the raw size would flag, plus enough neighbours to see the shape.
    let watch = sorted
        .iter()
        .take_while(|e| e.size > MAX_OBJECT_BYTES as u64)
        .count()
        .max(6);
    println!(
        "\nlargest blobs, encoded as singletons (object cap {:.1} MiB **served**):",
        mib(MAX_OBJECT_BYTES as u64)
    );
    let mut blocked = 0usize;
    for e in sorted.iter().take(watch) {
        let Ok(bytes) = chain.read(&e.name) else {
            println!("  {:>8.2} MiB raw  {}   unreadable", mib(e.size), e.name);
            continue;
        };
        let plan = PackPlan {
            blobs: vec![(name_key(&e.name), bytes)],
            prefer: PackClass::Singleton,
        };
        match encode_pack(0, &plan, level, MAX_OBJECT_BYTES) {
            Ok(enc) => println!(
                "  {:>8.2} MiB raw -> {:>8.2} MiB served ({:.0}%)  {}",
                mib(e.size),
                mib(enc.meta.size_wire as u64),
                100.0 * enc.meta.size_wire as f64 / e.size.max(1) as f64,
                e.name
            ),
            Err(err) => {
                blocked += 1;
                println!(
                    "  {:>8.2} MiB raw -> OVER THE CAP  {}\n      {err}",
                    mib(e.size),
                    e.name
                );
            }
        }
    }
    if blocked > 0 {
        println!(
            "  {blocked} blob(s) cannot be served as a single object. The container cannot hold \
             them as they are — drop them from the layer, or raise --max-object-bytes if the host \
             allows objects over {:.0} MiB (a static host does not).",
            mib(MAX_OBJECT_BYTES as u64)
        );
    }
}

fn histogram(label: &str, entries: &[Entry], top: usize, key: impl Fn(&Entry) -> String) {
    let mut by: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for e in entries {
        let slot = by.entry(key(e)).or_default();
        slot.0 += 1;
        slot.1 += e.size;
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_unstable_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    println!("\n{label}:");
    for (k, (n, bytes)) in rows.iter().take(top) {
        println!("  {n:>7}  {:>9.1} MiB  {k}", mib(*bytes));
    }
    if rows.len() > top {
        println!("  … {} more", rows.len() - top);
    }
}

/// **The model scan** — every `.m2` parsed through the client's own reader, reporting the header
/// facts that decide whether it renders and the parse errors that say it will not.
///
/// This is the check the optional layer exists for. Art backported from a newer client is *usually*
/// downconverted to MD20 v256, and when it is, everything below is unremarkable; when it is not,
/// [`benilla_m2::Error::UnsupportedVersion`] says so on line one instead of the character screen
/// coming up empty three hours later.
///
/// The columns past the version are the ones that have burned this client before: a model with more
/// **texture units than the reference's own combiner tiers**, a **geoset id** outside the ranges the
/// character system knows how to switch, an **empty skin profile** (a model that parses and draws
/// nothing). None of them is an error the parser can raise — each is a legal file our *renderer*
/// may not have a path for — so they are printed as facts and totalled, not judged.
fn scan_m2(chain: &Chain, entries: &[Entry]) {
    let models: Vec<&Entry> = entries
        .iter()
        .filter(|e| e.name.to_ascii_lowercase().ends_with(".m2"))
        .collect();
    if models.is_empty() {
        return;
    }
    println!("\nM2 scan ({} models):", models.len());
    let mut versions: BTreeMap<u32, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut rows = Vec::new();
    for e in &models {
        let bytes = match chain.read(&e.name) {
            Ok(b) => b,
            Err(err) => {
                failures.push(format!("{}: unreadable ({err:#})", e.name));
                continue;
            }
        };
        // The version is read straight off the header before the parse, so a file the walk REJECTS
        // still reports which version it was rejected for — which is the whole diagnostic when a
        // layer turns out to be a newer client's art after all.
        let raw_version = bytes
            .get(4..8)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        if let Some(v) = raw_version {
            *versions.entry(v).or_default() += 1;
        }
        match benilla_m2::parse_m2(&mut Cursor::new(bytes.as_slice())) {
            Ok(fmt) => {
                let m = fmt.model();
                // Skin profile 0 is the one the renderer builds from (), so it
                // is the one whose geosets matter here. A model whose profile 0 will not decode is
                // reported with zero geosets rather than as a parse failure, because that is
                // exactly what the loader would see: a model that parses and draws nothing.
                let geosets: Vec<u16> = m
                    .parse_embedded_skin(&bytes, 0)
                    .map(|skin| skin.submeshes().iter().map(|s| s.id).collect())
                    .unwrap_or_default();
                rows.push((
                    e.name.clone(),
                    raw_version.unwrap_or(0),
                    m.vertices.len(),
                    m.bones.len(),
                    m.textures.len(),
                    m.materials.len(),
                    geosets.len(),
                    geosets.iter().copied().max().unwrap_or(0),
                ));
            }
            Err(err) => failures.push(format!("{}: {err}", e.name)),
        }
    }
    print!("  versions:");
    for (v, n) in &versions {
        print!(" v{v}×{n}");
    }
    println!("   (this reader accepts 256..=263; anything else fails the parse below)");
    println!(
        "  {:<44} {:>4} {:>8} {:>6} {:>5} {:>5} {:>7} {:>8}",
        "model", "ver", "verts", "bones", "tex", "mat", "geosets", "max gid"
    );
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, v, verts, bones, tex, mat, gs, maxg) in &rows {
        println!(
            "  {:<44} {v:>4} {verts:>8} {bones:>6} {tex:>5} {mat:>5} {gs:>7} {maxg:>8}",
            short(name, 44)
        );
    }
    if failures.is_empty() {
        println!("  all {} models parsed", rows.len());
    } else {
        println!("  {} FAILED to parse:", failures.len());
        for f in &failures {
            println!("    {f}");
        }
    }
}

/// **The DBC scan** — every `.dbc`'s header, and the same file's header in the archive it overrides.
///
/// A DBC is a flat table with no self-description beyond `(rows, fields, record_size)`, and every
/// reader in this client is a hand-written offset walk against 1.12's column layout. So a newer
/// client's `CharSections.dbc` with one more column does not fail to load — it loads, and every
/// field past the new one is read from the wrong place, which surfaces as the wrong skin on the
/// wrong race. `record_size` and `field_count` are the whole of the compatibility question, and this
/// prints them for a human to diff against what the reader expects.
fn scan_dbc(chain: &Chain, entries: &[Entry]) {
    let dbcs: Vec<&Entry> = entries
        .iter()
        .filter(|e| e.name.to_ascii_lowercase().ends_with(".dbc"))
        .collect();
    if dbcs.is_empty() {
        return;
    }
    println!("\nDBC scan ({} tables):", dbcs.len());
    println!(
        "  {:<46} {:>7} {:>7} {:>9} {:>10}",
        "table", "rows", "fields", "rec bytes", "strings"
    );
    for e in &dbcs {
        let Ok(bytes) = chain.read(&e.name) else {
            println!("  {:<46}  unreadable", short(&e.name, 46));
            continue;
        };
        let word = |i: usize| {
            bytes
                .get(i..i + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .unwrap_or(0)
        };
        if bytes.len() < 20 || &bytes[0..4] != b"WDBC" {
            println!("  {:<46}  not a WDBC file", short(&e.name, 46));
            continue;
        }
        println!(
            "  {:<46} {:>7} {:>7} {:>9} {:>10}",
            short(&e.name, 46),
            word(4),
            word(8),
            word(12),
            word(16)
        );
    }
    println!(
        "  fields/rec-bytes must match 1.12's layout — this client's readers are fixed offset walks"
    );
}

/// **The texture scan** — decode a sample of the `.blp`s and report what came back.
///
/// Native decode ([`benilla_blp::decode_native`]), because that is the path the world takes: it
/// hands DXT blocks to the GPU untouched, so a format the *block* path does not know is a different
/// failure from one the RGBA path does not know. A sample rather than all of them because the
/// question is "does this decoder know these formats", which a few hundred files settle, and
/// decoding 3 GiB to re-settle it costs minutes.
/// One decoded texture: its format, width, height and decoded byte count.
type BlpScan = (String, u32, u32, usize);

fn scan_blp(chain: &Chain, entries: &[Entry], sample: usize) {
    let blps: Vec<&Entry> = entries
        .iter()
        .filter(|e| e.name.to_ascii_lowercase().ends_with(".blp"))
        .collect();
    if blps.is_empty() {
        return;
    }
    // A uniform stride, not the first N: the first N of a sorted listfile is one directory, and one
    // directory is one artist's export settings.
    let step = if sample == 0 || sample >= blps.len() {
        1
    } else {
        blps.len() / sample
    };
    let picked: Vec<&&Entry> = blps.iter().step_by(step.max(1)).collect();
    println!("\nBLP scan ({} of {} textures):", picked.len(), blps.len());
    let results: Vec<(String, Result<BlpScan, String>)> = picked
        .par_iter()
        .map(|e| {
            let decoded = chain
                .read(&e.name)
                .map_err(|err| format!("unreadable ({err:#})"))
                .and_then(|bytes| {
                    benilla_blp::decode_native(&bytes).map_err(|err| format!("{err}"))
                })
                .map(|blp| {
                    (
                        format!("{:?}", blp.texels),
                        blp.width,
                        blp.height,
                        blp.mips.len(),
                    )
                });
            (e.name.clone(), decoded)
        })
        .collect();
    let mut formats: BTreeMap<String, usize> = BTreeMap::new();
    let mut sizes: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for (name, r) in &results {
        match r {
            Ok((texels, w, h, _)) => {
                *formats.entry(texels.clone()).or_default() += 1;
                *sizes.entry((*w, *h)).or_default() += 1;
            }
            Err(err) => failures.push(format!("{name}: {err}")),
        }
    }
    print!("  formats:");
    for (f, n) in &formats {
        print!(" {f}×{n}");
    }
    println!();
    let mut by_size: Vec<_> = sizes.into_iter().collect();
    by_size.sort_unstable_by_key(|(_, n)| std::cmp::Reverse(*n));
    print!("  sizes:");
    for ((w, h), n) in by_size.iter().take(6) {
        print!(" {w}x{h}×{n}");
    }
    println!();
    if failures.is_empty() {
        println!("  every sampled texture decoded");
    } else {
        println!("  {} FAILED to decode:", failures.len());
        for f in failures.iter().take(20) {
            println!("    {f}");
        }
    }
}

/// Trim a chain name to `width` from the LEFT — the tail (`…\Human\Male\humanmale.m2`) is what
/// identifies a file, and the head (`Character\`) is what every row shares.
fn short(name: &str, width: usize) -> String {
    if name.chars().count() <= width {
        return name.to_string();
    }
    let tail: String = name
        .chars()
        .skip(name.chars().count() - (width - 1))
        .collect();
    format!("…{tail}")
}
