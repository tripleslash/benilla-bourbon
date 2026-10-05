//! `benilla-pack` — the one-time repackaging of a WoW 1.12.1 install into a BENPACK container.
//!
//! Reads the MPQ patch chain natively and emits a tree a dumb static object store can serve:
//! `index.bpi` plus a few thousand `.bpk` packs. The browser client (`benilla-formats`'s wasm
//! `Chain`) reads that tree instead of asking a companion host for one file at a time.
//!
//! Run once per install; re-runs are deterministic and incremental at the object store's end,
//! because identical input produces byte-identical output and a static host uploads by content hash.
//!
//! ```text
//! benilla-pack build  --data "C:\WoW-1.12.1\Data" --out web/packs
//! benilla-pack verify --data "C:\WoW-1.12.1\Data" --packs web/packs
//! ```

mod append;
mod group;
mod inspect;
mod merge;
mod overlay;
mod plan;
mod sidecars;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{bail, Context, Result};
use benilla_formats::Chain;
use benilla_mpq::name_key;
use benilla_pack::{
    decode_pack_body, encode_pack, frame_from_body, pack_body, pack_path, Index, PackSetBuilder,
    MAX_OBJECT_BYTES,
};
use clap::{Parser, Subcommand};
use rayon::prelude::*;

#[derive(Parser)]
#[command(
    name = "benilla-pack",
    about = "Repackage a WoW 1.12.1 install for static hosting"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Read the patch chain and write the container.
    Build(BuildArgs),
    /// Read the container back and compare it, blob for blob, with the chain it came from.
    Verify(VerifyArgs),
    /// Regenerate `boot-packs.json` against an existing container.
    ///
    /// Its own subcommand because the read-set manifests are recaptured far more often than the
    /// corpus is repacked — a `?boottrace=1` capture is minutes, a build is half an hour over
    /// 6.3 GiB — and translating names to pack ids needs only the index.
    Sidecars(SidecarArgs),
    /// Report what an archive holds and whether this client's readers can read it — run this
    /// BEFORE building a container out of unfamiliar art. See `inspect.rs`.
    Inspect(inspect::InspectArgs),
    /// Write an archive's blobs out as loose files, for a tool that is not this one to rewrite.
    ///
    /// Exists for exactly one pipeline: the HD layer's art is authored 1x and the character
    /// atlas of a 2x corpus is 512², so `scripts/upscale-hd-textures.py` has to see the BLPs to
    /// keep a 1x overlay from being sampled up blocky. Extract, upscale in place, then
    /// `build --replace-from` the same directory.
    Extract(ExtractArgs),
    /// Work out the updated-character-model layer: an HD archive with a later archive's files laid
    /// over it where they improve on it, as the key list and archive pins `build` packs. See
    /// `merge.rs`.
    Merge(merge::MergeArgs),
    /// Lay an archive over a built container without re-packing it. See `append.rs`.
    Append(append::AppendArgs),
    /// Work out the Turtle WoW 1.18.1 layer: what a Turtle install's numbered patches add to or
    /// change over the 1.12.1 corpus, as the key list `build --only` packs. See `overlay.rs`.
    Overlay(overlay::OverlayArgs),
}

#[derive(Parser)]
struct ExtractArgs {
    /// The `Data` directory or a single `.MPQ`.
    #[arg(long)]
    data: PathBuf,
    /// Where the loose tree goes, mirroring chain paths with `/` for `\`.
    #[arg(long)]
    out: PathBuf,
    /// Only names starting with this, matched case-insensitively — e.g. `Character\`.
    ///
    /// Not a nicety: the HD layer's `Textures\` tree is 2.9 GiB already authored at 512-1024, and
    /// doubling it would take the container somewhere nobody should download. The composite reads
    /// its base skin and every overlay from `Character\`, so that prefix IS the set whose
    /// resolution the atlas size depends on.
    #[arg(long, default_value = "")]
    prefix: String,
}

#[derive(Parser)]
struct BuildArgs {
    /// The install's `Data` directory, or one or more single `.MPQ`s mounted in the order given (a
    /// later one wins) — a layer stacked from more than one install.
    #[arg(long, required = true)]
    data: Vec<PathBuf>,
    /// Where the container goes. Created if absent.
    #[arg(long)]
    out: PathBuf,
    /// zstd level. 15 is the knee on this corpus; 19 costs several times the build for ~1%.
    #[arg(long, default_value_t = 15)]
    level: i32,
    /// Target **uncompressed** bytes per bundled pack. Served size lands near half of this (the
    /// corpus measures at 52.8% overall), and it is the served size that bounds what a Range-less
    /// client over-fetches for a single small blob.
    #[arg(long, default_value_t = 1024 * 1024)]
    target_pack_bytes: u64,
    /// Blobs at or above this get an object to themselves, so a Range-less client never downloads
    /// a big neighbour to reach a small one.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    singleton_bytes: u32,
    /// Fail rather than emit more objects than this. A typical static host allows 20,000 per
    /// Worker version; the default leaves room for the app's own files and for growth.
    #[arg(long, default_value_t = 18_000)]
    max_objects: u32,
    /// Per-object byte cap. Defaults to 24 MiB — under a static host's 25 MiB, deliberately.
    #[arg(long, default_value_t = MAX_OBJECT_BYTES)]
    max_object_bytes: u32,
    /// The boot read-set manifest to translate into `boot-packs.json` — captured from a traced
    /// boot (`?boottrace=1`) and checked in, so it stays meaningful across re-packs.
    #[arg(long, default_value = "web/boot-manifest.json")]
    boot_manifest: PathBuf,
    /// The world-entry read-set manifest, same treatment.
    #[arg(long, default_value = "web/world-manifest.json")]
    world_manifest: PathBuf,
    /// The glue-screen set: path prefixes, resolved against the container's own listfile
    /// (`sidecars::glue_names`).
    #[arg(long, default_value = "web/glue-manifest.json")]
    glue_manifest: PathBuf,
    /// Prefer a blob's bytes from this directory when it holds one, over the archive's own.
    ///
    /// The other half of the `extract` pipeline: the layer is packed from `patch-A.MPQ` as usual,
    /// except that every name `scripts/upscale-hd-textures.py` rewrote comes from the loose tree
    /// instead. Doing it here rather than by rebuilding the MPQ means no MPQ writer has to exist,
    /// and the archive on disk stays exactly what the player installed.
    ///
    /// Lookup is by the same `\` to `/` path mapping `extract` wrote, case-insensitively, so a name
    /// with no loose file simply comes from the archive.
    #[arg(long, value_name = "DIR")]
    replace_from: Option<PathBuf>,
    /// Pack only the keys this list names — `overlay`'s `overlay.tsv`, which makes the container a
    /// layer over another rather than a corpus of its own. Its listfile names only those keys.
    #[arg(long, value_name = "LIST")]
    only: Option<PathBuf>,
    /// Add every loose file under this directory as a blob of its own, named by its path from the
    /// directory (`\` for the separator) — `overlay`'s `add/`, whose merged tables no archive holds.
    #[arg(long, value_name = "DIR")]
    add_from: Option<PathBuf>,
    /// Refuse a `--replace-from` texture that is not exactly twice its archive original in both
    /// dimensions: the whole point of such a tree is the compositor's 2x, and a 1x file slipping
    /// through renders a character with quarter-size armour and nothing to point at.
    #[arg(long, requires = "replace_from")]
    require_doubled: bool,
    /// Refuse a `--replace-from` texture whose size is not the one this list (`overlay`'s
    /// `compose.tsv`) names for it — `--require-doubled` for a tree whose files are rescaled to the
    /// compositor's 2× from any size, not only doubled from 1×.
    #[arg(long, value_name = "LIST", requires = "replace_from")]
    require_size: Option<PathBuf>,
    /// Read the chain only through this archive (its file name, e.g. `patch-9.mpq`): the
    /// composite as it stood there, for a layer cut from an install that has more patches mounted
    /// above it (`overlay` reads it the same way).
    #[arg(long, value_name = "ARCHIVE")]
    top: Option<String>,
    /// Read each key this list (`merge`'s `merge.tsv`) names from the archive it names, rather than
    /// from whichever mounted archive is highest: a layer that takes a later archive's file only
    /// where it improves on the earlier one's.
    #[arg(long, value_name = "LIST")]
    pins: Option<PathBuf>,
    /// Plan and report, but write nothing.
    #[arg(long)]
    dry_run: bool,
    /// Stop after this many entries. For a fast smoke run over a real install.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Parser)]
struct VerifyArgs {
    /// As `build --data`.
    #[arg(long, required = true)]
    data: Vec<PathBuf>,
    #[arg(long)]
    packs: PathBuf,
    /// Check every Nth named entry. 1 checks all of them (slow, and the right thing before a
    /// site).
    #[arg(long, default_value_t = 20)]
    stride: usize,
    /// A layer container: check only the names this key list (`overlay.tsv`) holds.
    #[arg(long, value_name = "LIST")]
    only: Option<PathBuf>,
    /// The tree the build took replacements from: a replaced texture is checked to be exactly 2x
    /// its chain original rather than byte-equal to it.
    #[arg(long, value_name = "DIR")]
    replace_from: Option<PathBuf>,
    /// The sizes the replacements were rescaled to (`overlay`'s `compose.tsv`), checked instead of
    /// exactly 2x.
    #[arg(long, value_name = "LIST", requires = "replace_from")]
    require_size: Option<PathBuf>,
    /// Read the chain only through this archive — see `build --top`.
    #[arg(long, value_name = "ARCHIVE")]
    top: Option<String>,
    /// As `build --pins`.
    #[arg(long, value_name = "LIST")]
    pins: Option<PathBuf>,
}

#[derive(Parser)]
struct SidecarArgs {
    /// The container to read `index.bpi` from and write the side files into.
    #[arg(long)]
    packs: PathBuf,
    #[arg(long, default_value = "web/boot-manifest.json")]
    boot_manifest: PathBuf,
    #[arg(long, default_value = "web/world-manifest.json")]
    world_manifest: PathBuf,
    #[arg(long, default_value = "web/glue-manifest.json")]
    glue_manifest: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Build(a) => build(a),
        Cmd::Verify(a) => verify(a),
        Cmd::Inspect(a) => inspect::inspect(a),
        Cmd::Extract(a) => extract(a),
        Cmd::Overlay(a) => overlay::overlay(a),
        Cmd::Merge(a) => merge::merge(a),
        Cmd::Append(a) => append::append(a),
        Cmd::Sidecars(a) => {
            let index =
                Index::parse(std::fs::read(a.packs.join("index.bpi")).context("reading index")?)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            sidecars::write_boot_packs(
                &a.packs,
                &index,
                &a.boot_manifest,
                &a.world_manifest,
                &a.glue_manifest,
            )
        }
    }
}

/// One resolved blob, before its bytes are read.
struct Blob {
    key: u64,
    archive: usize,
    block: u32,
    len: u32,
}

/// Walk the chain's archives and resolve them into the composite the client would see.
///
/// Deliberately **not** built on `Chain::list`: that unions the archives' listfiles, and a file no
/// listfile names is reachable by name but not enumerable. Walking the hash tables is ground truth.
/// Priority and tombstone semantics are reproduced here rather than borrowed because `Chain::read`
/// resolves one name at a time and there is no bulk equivalent — but the *order* is the chain's own
/// (`Chain::archives` hands them back in `mount_order`'s ascending priority), so the mount law is
/// not restated.
fn resolve(chain: &Chain) -> Vec<Blob> {
    let mut resolved: HashMap<u64, (usize, u32, u32, bool)> = HashMap::new();
    for (i, archive) in chain.archives().iter().enumerate() {
        for e in archive.entries() {
            // Later archives win, tombstones included: a tombstone must shadow the lower-priority
            // copy, not be skipped past to it (decision 0246).
            resolved.insert(e.key, (i, e.block_index, e.file_size, e.is_delete_marker));
        }
    }
    let mut out: Vec<Blob> = resolved
        .into_iter()
        .filter(|(_, (_, _, _, tomb))| !tomb)
        .map(|(key, (archive, block, len, _))| Blob {
            key,
            archive,
            block,
            len,
        })
        .collect();
    // A HashMap drain is arbitrary; sort so the whole build is deterministic from here on.
    out.sort_unstable_by_key(|b| b.key);
    out
}

/// A resolved blob's bytes: by block index, or by name for an entry encrypted under its name
/// (Turtle's patch archives encrypt a few), which the nameless read cannot decrypt.
fn read_blob(
    archives: &[benilla_mpq::Archive],
    b: &Blob,
    names: &HashMap<u64, String>,
) -> Result<Vec<u8>> {
    let archive = &archives[b.archive];
    if archive.is_encrypted(b.block) {
        let name = names
            .get(&b.key)
            .with_context(|| format!("block {} is encrypted and named in no listfile", b.block))?;
        return archive
            .read_file(name)
            .map_err(|e| anyhow::anyhow!("reading {name}: {e}"));
    }
    archive
        .read_block(b.block)
        .map_err(|e| anyhow::anyhow!("reading block {} : {e}", b.block))
}

/// `--require-doubled`: a replacement texture is exactly twice its original in both dimensions.
fn check_doubled(original: &[u8], replacement: &[u8]) -> Result<()> {
    let dims = |b: &[u8]| benilla_blp::dimensions(b).map_err(|e| anyhow::anyhow!("{e}"));
    let (ow, oh) = dims(original).context("the archive original is not a BLP2")?;
    let (rw, rh) = dims(replacement).context("the replacement is not a BLP2")?;
    if (rw, rh) != (ow * 2, oh * 2) {
        bail!("{rw}x{rh} is not twice the original {ow}x{oh} (still 1x? run the upscale first)");
    }
    Ok(())
}

/// `--require-size`: a replacement texture is exactly the size its list names.
fn check_size(replacement: &[u8], want: (u32, u32)) -> Result<()> {
    let got = benilla_blp::dimensions(replacement)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("the replacement is not a BLP2")?;
    if got != want {
        bail!(
            "{}x{} is not the listed {}x{} (run the rescale first)",
            got.0,
            got.1,
            want.0,
            want.1
        );
    }
    Ok(())
}

/// Open the chain a `build`/`verify` reads — one `Data` directory or `.MPQ`, or several `.MPQ`s
/// stacked in order — through `--top` when one is named.
fn open_chain(data: &[PathBuf], top: Option<&str>) -> Result<Chain> {
    let mut chain = match data {
        [one] => Chain::open(one).with_context(|| format!("opening {}", one.display()))?,
        many => Chain::open_archives(many)?,
    };
    if let Some(top) = top {
        if !chain.truncate_after(top) {
            bail!("--top {top}: no such archive mounted");
        }
    }
    Ok(chain)
}

/// The index of the mounted archive whose file name is `name`, any case.
fn archive_named(chain: &Chain, name: &str) -> Result<usize> {
    chain
        .archives()
        .iter()
        .position(|a| {
            a.path()
                .file_name()
                .is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(name))
        })
        .with_context(|| format!("--pins names {name}, which is not mounted"))
}

/// `--pins`: re-resolve each pinned key against the archive its pin names.
fn apply_pins(chain: &Chain, blobs: &mut [Blob], pins: &HashMap<u64, String>) -> Result<usize> {
    let mut by_archive: HashMap<usize, HashMap<u64, (u32, u32, bool)>> = HashMap::new();
    for name in pins.values().collect::<std::collections::BTreeSet<_>>() {
        let i = archive_named(chain, name)?;
        by_archive.insert(
            i,
            chain.archives()[i]
                .entries()
                .map(|e| (e.key, (e.block_index, e.file_size, e.is_delete_marker)))
                .collect(),
        );
    }
    let mut moved = 0;
    for b in blobs.iter_mut() {
        let Some(pin) = pins.get(&b.key) else {
            continue;
        };
        let i = archive_named(chain, pin)?;
        let Some(&(block, len, tomb)) = by_archive[&i].get(&b.key) else {
            bail!("{:#018x} is pinned to {pin}, which does not hold it", b.key);
        };
        if tomb {
            bail!("{:#018x} is pinned to {pin}, which deletes it", b.key);
        }
        if b.archive != i {
            moved += 1;
        }
        (b.archive, b.block, b.len) = (i, block, len);
    }
    Ok(moved)
}

/// The loose path a chain name takes on disk: backslashes become separators, everything else is
/// kept verbatim. The twin of [`replace_lookup`]'s key, and the reason both live next to each other.
fn loose_path(out: &Path, name: &str) -> PathBuf {
    let mut p = out.to_path_buf();
    for seg in name.split(['\\', '/']).filter(|s| !s.is_empty()) {
        p.push(seg);
    }
    p
}

/// `benilla-pack extract` — the archive's blobs as loose files, for a tool that is not this one.
fn extract(a: ExtractArgs) -> Result<()> {
    let chain = Chain::open(&a.data).with_context(|| format!("opening {}", a.data.display()))?;
    let prefix = a.prefix.to_ascii_lowercase().replace('/', "\\");
    let entries: Vec<String> = chain
        .list()
        .context("reading the listfile")?
        .into_iter()
        .map(|e| e.name)
        .filter(|n| {
            prefix.is_empty()
                || n.to_ascii_lowercase()
                    .replace('/', "\\")
                    .starts_with(&prefix)
        })
        .collect();
    eprintln!(
        "extracting {} entries{} to {}",
        entries.len(),
        if a.prefix.is_empty() {
            String::new()
        } else {
            format!(" under {}", a.prefix)
        },
        a.out.display()
    );

    // Sequential, and deliberately: the read is a decompress off one file handle and the write is
    // one create per blob, so this is disk-bound either way — and a failure here should name the
    // one file it happened on rather than arrive interleaved from eight threads.
    let mut written = 0usize;
    let mut bytes = 0u64;
    for name in &entries {
        let data = match chain.read(name) {
            Ok(d) => d,
            // A listfile can name a file the composite chain tombstoned. That is not an error here
            // any more than it is at read time — it is simply not part of the composite.
            Err(_) => continue,
        };
        let path = loose_path(&a.out, name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(&path, &data).with_context(|| format!("writing {}", path.display()))?;
        written += 1;
        bytes += data.len() as u64;
        if written.is_multiple_of(1000) {
            eprintln!("  {written}/{}", entries.len());
        }
    }
    eprintln!(
        "wrote {written} files, {:.1} MiB\nnow: python scripts/upscale-hd-textures.py {}",
        bytes as f64 / (1024.0 * 1024.0),
        a.out.display()
    );
    Ok(())
}

/// The loose-file overrides `--replace-from` supplies, keyed the way the container is.
///
/// Walked once into a `key -> path` map rather than probed per blob, because the probe would be a
/// `stat` per name over ~16,000 names and the walk is one directory traversal. Keyed by
/// [`benilla_mpq::name_key`] — MPQ's own hash — so the case-insensitive, `/`-equals-`\` matching
/// comes from the same place every other name comparison in this tree gets it, rather than being
/// restated as a string rule that could drift.
fn replace_lookup(dir: &Path) -> Result<HashMap<u64, PathBuf>> {
    fn walk(dir: &Path, rel: &mut Vec<String>, out: &mut HashMap<u64, PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            if path.is_dir() {
                rel.push(name);
                walk(&path, rel, out)?;
                rel.pop();
            } else {
                rel.push(name);
                out.insert(name_key(&rel.join("\\")), path.clone());
                rel.pop();
            }
        }
        Ok(())
    }
    let mut out = HashMap::new();
    walk(dir, &mut Vec::new(), &mut out)?;
    Ok(out)
}

/// Every file under `dir` as `(chain name, path)`: the name is its path from `dir` with `\` for the
/// separator, the inverse of [`loose_path`]. Sorted, so the build stays deterministic.
fn loose_names(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    fn walk(dir: &Path, rel: &mut Vec<String>, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            rel.push(entry.file_name().to_string_lossy().into_owned());
            if path.is_dir() {
                walk(&path, rel, out)?;
            } else {
                out.push((rel.join("\\"), path));
            }
            rel.pop();
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, &mut Vec::new(), &mut out)?;
    out.sort();
    Ok(out)
}

/// Every name any archive's `(listfile)` mentions, keyed the way the container is. Names are a
/// grouping *hint* and a debugging aid; none of them ship in `index.bpi`.
fn name_hints(chain: &Chain) -> HashMap<u64, String> {
    match chain.list() {
        Ok(entries) => entries
            .into_iter()
            .map(|e| (name_key(&e.name), e.name))
            .collect(),
        Err(e) => {
            eprintln!("warning: no listfile ({e:#}); grouping falls back to archive order");
            HashMap::new()
        }
    }
}

fn build(a: BuildArgs) -> Result<()> {
    let chain = open_chain(&a.data, a.top.as_deref())?;
    eprintln!("mounted {} archives", chain.archives().len());
    let sizes = match &a.require_size {
        Some(list) => overlay::read_compose_list(list)?,
        None => HashMap::new(),
    };

    let mut blobs = resolve(&chain);
    if let Some(list) = &a.pins {
        let moved = apply_pins(&chain, &mut blobs, &merge::read_pins(list)?)?;
        eprintln!("--pins: {moved} keys read from a lower archive than the one mounted highest");
    }
    let mut names = name_hints(&chain);
    if let Some(list) = &a.only {
        let keep = overlay::read_list(list)?;
        blobs.retain(|b| keep.contains(&b.key));
        names.retain(|k, _| keep.contains(k));
        if blobs.len() != keep.len() {
            bail!(
                "{} lists {} keys but the chain resolves {} of them",
                list.display(),
                keep.len(),
                blobs.len()
            );
        }
    }
    // `--add-from`: blobs no archive holds, named by their path under the directory.
    let mut added: HashMap<u64, PathBuf> = HashMap::new();
    if let Some(dir) = &a.add_from {
        for (name, path) in loose_names(dir)? {
            let key = name_key(&name);
            if blobs.iter().any(|b| b.key == key) {
                bail!("--add-from {name} is already a name in the chain; add only new names");
            }
            let len = std::fs::metadata(&path)?.len();
            blobs.push(Blob {
                key,
                archive: usize::MAX,
                block: 0,
                len: u32::try_from(len).context("an added file over 4 GiB")?,
            });
            eprintln!("--add-from: {name}");
            names.insert(key, name);
            added.insert(key, path);
        }
        blobs.sort_unstable_by_key(|b| b.key);
    }
    if let Some(n) = a.limit {
        blobs.truncate(n);
    }
    let corpus: u64 = blobs.iter().map(|b| b.len as u64).sum();
    eprintln!(
        "resolved {} entries, {:.2} GiB uncompressed, {} named ({:.3}%)",
        blobs.len(),
        gib(corpus),
        blobs.iter().filter(|b| names.contains_key(&b.key)).count(),
        100.0 * blobs.iter().filter(|b| names.contains_key(&b.key)).count() as f64
            / blobs.len().max(1) as f64
    );

    // The captured read sets are a locality signal, not just a prefetch list — see `ReadSets`.
    let read_sets = sidecars::ReadSets::load(&a.boot_manifest, &a.world_manifest);
    let groups = plan::plan(
        &blobs,
        &names,
        &read_sets,
        a.singleton_bytes,
        a.target_pack_bytes,
        a.max_objects,
    )?;
    eprintln!(
        "planned {} packs (budget {}), {:.0} KiB uncompressed mean",
        groups.len(),
        a.max_objects,
        groups
            .iter()
            .map(|g| g.iter().map(|b| b.len as u64).sum::<u64>())
            .sum::<u64>() as f64
            / groups.len().max(1) as f64
            / 1024.0
    );
    report_read_set("boot", &groups, &read_sets.boot);
    report_read_set("world", &groups, &read_sets.world);
    if a.dry_run {
        eprintln!("--dry-run: nothing written");
        return Ok(());
    }

    // Loaded before the encode, so a missing or unreadable directory fails in a second rather than
    // forty minutes in. Empty when the flag is absent, which costs one hash lookup per blob.
    let replacements = match &a.replace_from {
        Some(dir) => {
            let map = replace_lookup(dir).with_context(|| format!("walking {}", dir.display()))?;
            let covered = blobs.iter().filter(|b| map.contains_key(&b.key)).count();
            eprintln!(
                "--replace-from: {} loose files, {covered} of {} blobs come from {}",
                map.len(),
                blobs.len(),
                dir.display()
            );
            map
        }
        None => HashMap::new(),
    };

    std::fs::create_dir_all(&a.out)?;
    let archives = chain.archives();
    let done = AtomicUsize::new(0);
    let total = groups.len();

    // Encode AND WRITE in parallel — this is the build's whole cost — carrying only metadata back
    // out. Collecting the encoded packs and writing them afterwards is the obvious shape and it
    // holds the entire 3 GiB output in memory (measured, before this: 2 GB and climbing).
    // Committing is still sequential, because a pack names its own id and its path derives from it;
    // ids come from the deterministic plan order, so parallelism cannot change the output.
    let mut records: Vec<_> = groups
        .par_iter()
        .enumerate()
        .map(|(i, g)| -> Result<_> {
            let plan_blobs = g
                .iter()
                .map(|b| {
                    if let Some(path) = added.get(&b.key) {
                        let bytes = std::fs::read(path)
                            .with_context(|| format!("reading added {}", path.display()))?;
                        return Ok((b.key, bytes));
                    }
                    // The loose override first — see `--replace-from`. A read failure here is fatal
                    // rather than a silent fall back to the archive: the whole point of the flag is
                    // that these bytes ship, and a container that quietly used the un-upscaled
                    // original would be a character that renders wrong with nothing to point at.
                    if let Some(path) = replacements.get(&b.key) {
                        let bytes = std::fs::read(path)
                            .with_context(|| format!("reading override {}", path.display()))?;
                        if a.require_doubled {
                            check_doubled(&read_blob(archives, b, &names)?, &bytes)
                                .with_context(|| format!("override {}", path.display()))?;
                        }
                        if a.require_size.is_some() {
                            let want = sizes.get(&b.key).with_context(|| {
                                format!("override {} is not in --require-size", path.display())
                            })?;
                            check_size(&bytes, *want)
                                .with_context(|| format!("override {}", path.display()))?;
                        }
                        return Ok((b.key, bytes));
                    }
                    Ok((b.key, read_blob(archives, b, &names)?))
                })
                .collect::<Result<Vec<_>>>()?;
            let prefer = plan::prefer_for(g, &names);
            let e = encode_pack(
                i as u32,
                &benilla_pack::PackPlan {
                    blobs: plan_blobs,
                    prefer,
                },
                a.level,
                a.max_object_bytes,
            )
            .map_err(|e| anyhow::anyhow!("encoding pack {i}: {e}"))?;
            // Write here, inside the parallel task, and carry only the record out — see
            // `PackSetBuilder::record` for why holding the encoded packs is not an option.
            let rec = e
                .write_to(&a.out)
                .map_err(|e| anyhow::anyhow!("writing pack {i}: {e}"))?;
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(250) {
                eprintln!("  {n}/{total} packs");
            }
            Ok(rec)
        })
        .collect::<Result<Vec<_>>>()?;
    eprintln!("  {total}/{total} packs");

    records.sort_by_key(|r| r.pack_id);
    let mut builder =
        PackSetBuilder::new(&a.out, a.level).with_max_object_bytes(a.max_object_bytes);
    let mut wire_total = 0u64;
    let mut wire_sizes = Vec::with_capacity(records.len());
    for r in records {
        wire_total += r.meta.size_wire as u64;
        wire_sizes.push(r.meta.size_wire);
        builder
            .record(r)
            .map_err(|e| anyhow::anyhow!("committing pack: {e}"))?;
    }
    let index_path = builder
        .finish()
        .map_err(|e| anyhow::anyhow!("writing index: {e}"))?;

    // The listfile still ships: `Chain::list` is reached on the web target (macro_icons enumerates
    // Interface\Icons at boot), so the names have to be available even though the index has none.
    let listfile = a.out.join("listfile.txt");
    let mut sorted: Vec<&String> = names.values().collect();
    sorted.sort();
    std::fs::write(
        &listfile,
        sorted.iter().fold(String::new(), |mut s, n| {
            s.push_str(n);
            s.push('\n');
            s
        }),
    )?;

    // Read the index back rather than carrying a key->pack map through the build: it is 1.6 MB and
    // a parse away, and going through the same `Index` the client uses means the prefetch lists are
    // resolved by exactly the code path that will later serve them.
    let index = Index::parse(std::fs::read(&index_path)?).map_err(|e| anyhow::anyhow!("{e}"))?;
    sidecars::write_boot_packs(
        &a.out,
        &index,
        &a.boot_manifest,
        &a.world_manifest,
        &a.glue_manifest,
    )?;

    wire_sizes.sort_unstable();
    let pct = |p: f64| wire_sizes[((wire_sizes.len() - 1) as f64 * p) as usize];
    eprintln!(
        "\nwrote {} objects, {:.2} GiB served ({:.1}% of uncompressed)\n\
         object sizes: p50 {:.0} KiB  p90 {:.0} KiB  max {:.2} MiB (cap {:.2} MiB)\n\
         {} + {}",
        wire_sizes.len(),
        gib(wire_total),
        100.0 * wire_total as f64 / corpus as f64,
        pct(0.5) as f64 / 1024.0,
        pct(0.9) as f64 / 1024.0,
        *wire_sizes.last().unwrap() as f64 / 1048576.0,
        a.max_object_bytes as f64 / 1048576.0,
        index_path.display(),
        listfile.display(),
    );
    Ok(())
}

/// Read the container back and compare against the chain, which is the only check that matters:
/// the client must see exactly the bytes the native build sees.
fn verify(a: VerifyArgs) -> Result<()> {
    let chain = open_chain(&a.data, a.top.as_deref())?;
    let sizes = match &a.require_size {
        Some(list) => overlay::read_compose_list(list)?,
        None => HashMap::new(),
    };
    let pins = match &a.pins {
        Some(list) => merge::read_pins(list)?,
        None => HashMap::new(),
    };
    let index = Index::parse(std::fs::read(a.packs.join("index.bpi")).context("reading index")?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "index: {} entries, {} packs, build {:#018x}",
        index.len(),
        index.pack_count(),
        index.build_id()
    );

    let mut names: Vec<String> = chain.list()?.into_iter().map(|e| e.name).collect();
    if let Some(list) = &a.only {
        let keep = overlay::read_list(list)?;
        names.retain(|n| keep.contains(&name_key(n)));
        // One name per key: a listfile can spell one path several ways.
        let mut seen = std::collections::HashSet::new();
        names.retain(|n| seen.insert(name_key(n)));
    }
    let replaced = match &a.replace_from {
        Some(dir) => replace_lookup(dir)?,
        None => HashMap::new(),
    };
    let checked = AtomicUsize::new(0);
    let doubled = AtomicUsize::new(0);
    let missing = AtomicUsize::new(0);

    names
        .par_iter()
        .step_by(a.stride.max(1))
        .try_for_each(|name| -> Result<()> {
            let key = name_key(name);
            // A pinned key is the pinned archive's bytes, not the highest mounted copy's.
            let read = match pins.get(&key) {
                Some(pin) => chain.archives()[archive_named(&chain, pin)?]
                    .read_file(name)
                    .map_err(|e| anyhow::anyhow!("{e}")),
                None => chain.read(name),
            };
            let want = match read {
                Ok(b) => b,
                // A name the listfile has but the chain cannot read is a tombstone; the container
                // correctly omits those, so it is not a mismatch.
                Err(_) => return Ok(()),
            };
            let Some(at) = index.locate(key) else {
                missing.fetch_add(1, Ordering::Relaxed);
                bail!("{name} ({key:#018x}) is in the chain but not in the container");
            };
            let meta = index
                .pack(at.pack_id)
                .ok_or_else(|| anyhow::anyhow!("{name}: pack {} not in index", at.pack_id))?;
            let wire = std::fs::read(a.packs.join(pack_path(at.pack_id)))
                .with_context(|| format!("{name}: reading pack {}", at.pack_id))?;
            let body = pack_body(&wire, at.pack_id).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let body = decode_pack_body(body, &meta).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let got = frame_from_body(&body, &at).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            if replaced.contains_key(&key) {
                match sizes.get(&key) {
                    Some(size) => check_size(&got, *size),
                    None => check_doubled(&want, &got),
                }
                .with_context(|| format!("{name}: replacement"))?;
                doubled.fetch_add(1, Ordering::Relaxed);
            } else if got != want {
                bail!(
                    "{name}: container has {} bytes, chain has {}",
                    got.len(),
                    want.len()
                );
            }
            let n = checked.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(1000) {
                eprint!("\r  verified {n}");
            }
            Ok(())
        })?;

    eprintln!(
        "\rverified {} blobs against the chain: {} byte-for-byte, {} replacements at their 2x size ({} missing)",
        checked.load(Ordering::Relaxed),
        checked.load(Ordering::Relaxed) - doubled.load(Ordering::Relaxed),
        doubled.load(Ordering::Relaxed),
        missing.load(Ordering::Relaxed)
    );
    Ok(())
}

fn gib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// What a captured read set will cost the page to prefetch, in **uncompressed** bytes, straight off
/// the plan — so a grouping change can be judged with `--dry-run` in minutes rather than after a
/// half-hour build.
///
/// The number that matters is the ratio: how much of what the prefetch downloads the client will
/// actually read. `sidecars` reports the same thing again from the finished index in served bytes;
/// this is the one that closes the loop while the decision is still cheap to change.
fn report_read_set(label: &str, groups: &[Vec<&Blob>], want: &std::collections::HashSet<u64>) {
    if want.is_empty() {
        return;
    }
    let (mut packs, mut fetched, mut needed) = (0usize, 0u64, 0u64);
    for g in groups {
        let hit: u64 = g
            .iter()
            .filter(|b| want.contains(&b.key))
            .map(|b| b.len as u64)
            .sum();
        if hit == 0 {
            continue;
        }
        packs += 1;
        needed += hit;
        fetched += g.iter().map(|b| b.len as u64).sum::<u64>();
    }
    eprintln!(
        "  {label} read set: {} names -> {packs} packs, {:.1} / {:.1} MiB uncompressed ({:.1}x over-fetch)",
        want.len(),
        fetched as f64 / 1048576.0,
        needed as f64 / 1048576.0,
        fetched as f64 / needed.max(1) as f64,
    );
}
