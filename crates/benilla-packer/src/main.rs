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

mod group;
mod plan;
mod sidecars;

use std::collections::HashMap;
use std::path::PathBuf;
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
    /// Regenerate `boot-packs.json` (and the `config.json` stub) against an existing container.
    ///
    /// Its own subcommand because the read-set manifests are recaptured far more often than the
    /// corpus is repacked — a `?boottrace=1` capture is minutes, a build is half an hour over
    /// 6.3 GiB — and translating names to pack ids needs only the index.
    Sidecars(SidecarArgs),
}

#[derive(Parser)]
struct BuildArgs {
    /// The install's `Data` directory (or a single `.MPQ`).
    #[arg(long)]
    data: PathBuf,
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
    /// Plan and report, but write nothing.
    #[arg(long)]
    dry_run: bool,
    /// Stop after this many entries. For a fast smoke run over a real install.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Parser)]
struct VerifyArgs {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    packs: PathBuf,
    /// Check every Nth named entry. 1 checks all of them (slow, and the right thing before a
    /// site).
    #[arg(long, default_value_t = 20)]
    stride: usize,
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
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Build(a) => build(a),
        Cmd::Verify(a) => verify(a),
        Cmd::Sidecars(a) => {
            let index = Index::parse(std::fs::read(a.packs.join("index.bpi")).context("reading index")?)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            sidecars::write_boot_packs(&a.packs, &index, &a.boot_manifest, &a.world_manifest)?;
            sidecars::write_config_stub(&a.packs)
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
    let chain = Chain::open(&a.data).with_context(|| format!("opening {}", a.data.display()))?;
    eprintln!("mounted {} archives", chain.archives().len());

    let mut blobs = resolve(&chain);
    if let Some(n) = a.limit {
        blobs.truncate(n);
    }
    let names = name_hints(&chain);
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
                    let bytes = archives[b.archive]
                        .read_block(b.block)
                        .map_err(|e| anyhow::anyhow!("reading block {} : {e}", b.block))?;
                    Ok((b.key, bytes))
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
            if n % 250 == 0 {
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
    sidecars::write_boot_packs(&a.out, &index, &a.boot_manifest, &a.world_manifest)?;
    sidecars::write_config_stub(&a.out)?;

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
    let chain = Chain::open(&a.data)?;
    let index = Index::parse(std::fs::read(a.packs.join("index.bpi")).context("reading index")?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "index: {} entries, {} packs, build {:#018x}",
        index.len(),
        index.pack_count(),
        index.build_id()
    );

    let names: Vec<String> = chain.list()?.into_iter().map(|e| e.name).collect();
    let checked = AtomicUsize::new(0);
    let missing = AtomicUsize::new(0);

    names
        .par_iter()
        .step_by(a.stride.max(1))
        .try_for_each(|name| -> Result<()> {
            let want = match chain.read(name) {
                Ok(b) => b,
                // A name the listfile has but the chain cannot read is a tombstone; the container
                // correctly omits those, so it is not a mismatch.
                Err(_) => return Ok(()),
            };
            let key = name_key(name);
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
            if got != want {
                bail!(
                    "{name}: container has {} bytes, chain has {}",
                    got.len(),
                    want.len()
                );
            }
            let n = checked.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 1000 == 0 {
                eprint!("\r  verified {n}");
            }
            Ok(())
        })?;

    eprintln!(
        "\rverified {} blobs byte-for-byte against the chain ({} missing)",
        checked.load(Ordering::Relaxed),
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
