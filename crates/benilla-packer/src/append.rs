//! `benilla-pack append` — **lay an archive over a built container** without re-packing it: the
//! archive's files go into new packs after the container's own, its keys answer from them (a file
//! the container already holds is superseded), and every existing object stays byte-identical.
//!
//! For a patch the reference mounts above everything the corpus was built from and that applies on
//! every realm — Project Reforged's `patch-Y` night sky, laid into `web/packs` — where a rebuild of
//! the corpus would reassign its pack ids and re-upload all of it. The index, `listfile.txt` and
//! `boot-packs.json` are rewritten; the new build id is what drops a client's cached copy of the
//! old index.
//!
//! ```text
//! benilla-pack append --packs web/packs --data "C:\TurtleWoW-1.18.1\Data\patch-Y.MPQ"
//! ```

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use benilla_mpq::name_key;
use benilla_pack::{encode_pack, Index, PackSetBuilder, MAX_OBJECT_BYTES};
use clap::Parser;

#[derive(Parser)]
pub struct AppendArgs {
    /// The built container to lay the archives over, rewritten in place.
    #[arg(long)]
    pub packs: PathBuf,
    /// The archives laid over it, in order (a later one wins).
    #[arg(long, required = true)]
    pub data: Vec<PathBuf>,
    /// zstd level, as `build --level`.
    #[arg(long, default_value_t = 15)]
    pub level: i32,
    /// As `build --target-pack-bytes`.
    #[arg(long, default_value_t = 1024 * 1024)]
    pub target_pack_bytes: u64,
    /// As `build --singleton-bytes`.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    pub singleton_bytes: u32,
    #[arg(long, default_value = "web/boot-manifest.json")]
    pub boot_manifest: PathBuf,
    #[arg(long, default_value = "web/world-manifest.json")]
    pub world_manifest: PathBuf,
    #[arg(long, default_value = "web/glue-manifest.json")]
    pub glue_manifest: PathBuf,
}

pub fn append(a: AppendArgs) -> Result<()> {
    let index_path = a.packs.join("index.bpi");
    let index = Index::parse(
        std::fs::read(&index_path).with_context(|| format!("reading {}", index_path.display()))?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (packs_before, entries_before) = (index.pack_count(), index.len());
    let chain = crate::open_chain(&a.data, None)?;
    let names = crate::name_hints(&chain);
    let metadata: HashSet<u64> = crate::overlay::METADATA
        .iter()
        .map(|n| name_key(n))
        .collect();
    let mut blobs = crate::resolve(&chain);
    blobs.retain(|b| !metadata.contains(&b.key));
    let superseded = blobs
        .iter()
        .filter(|b| index.locate(b.key).is_some())
        .count();
    eprintln!(
        "{} keys to lay over {} ({} of them superseding the container's own)",
        blobs.len(),
        a.packs.display(),
        superseded
    );

    let groups = crate::plan::plan(
        &blobs,
        &names,
        &crate::sidecars::ReadSets::default(),
        a.singleton_bytes,
        a.target_pack_bytes,
        u32::MAX,
    )?;
    let mut builder = PackSetBuilder::from_index(&a.packs, a.level, &index)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let archives = chain.archives();
    for g in &groups {
        let plan_blobs = g
            .iter()
            .map(|b| Ok((b.key, crate::read_blob(archives, b, &names)?)))
            .collect::<Result<Vec<_>>>()?;
        let id = builder.pack_count();
        let encoded = encode_pack(
            id,
            &benilla_pack::PackPlan {
                blobs: plan_blobs,
                prefer: crate::plan::prefer_for(g, &names),
            },
            a.level,
            MAX_OBJECT_BYTES,
        )
        .map_err(|e| anyhow::anyhow!("encoding pack {id}: {e}"))?;
        builder
            .add_encoded(encoded)
            .map_err(|e| anyhow::anyhow!("writing pack {id}: {e}"))?;
    }
    builder.finish().map_err(|e| anyhow::anyhow!("{e}"))?;

    // The listfile: the container's names plus the archives', one spelling per key.
    let listfile = a.packs.join("listfile.txt");
    let text = std::fs::read_to_string(&listfile).unwrap_or_default();
    let mut all: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    let mut have: HashSet<u64> = all.iter().map(|n| name_key(n)).collect();
    let mut added: HashMap<u64, &String> = HashMap::new();
    for b in &blobs {
        if let Some(n) = names.get(&b.key) {
            if have.insert(b.key) {
                added.insert(b.key, n);
            }
        }
    }
    all.extend(added.values().map(|n| n.to_string()));
    all.sort();
    std::fs::write(&listfile, all.join("\n") + "\n")
        .with_context(|| format!("writing {}", listfile.display()))?;

    let index = Index::parse(std::fs::read(&index_path)?).map_err(|e| anyhow::anyhow!("{e}"))?;
    crate::sidecars::write_boot_packs(
        &a.packs,
        &index,
        &a.boot_manifest,
        &a.world_manifest,
        &a.glue_manifest,
    )?;
    eprintln!(
        "{}: {} -> {} packs, {} -> {} entries, build {:#018x}; {} names added to listfile.txt",
        a.packs.display(),
        packs_before,
        index.pack_count(),
        entries_before,
        index.len(),
        index.build_id(),
        added.len()
    );
    Ok(())
}
