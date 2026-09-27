//! The writer half of BENPACK (`--features write`, native only).
//!
//! Kept behind a feature because it pulls the C zstd *encoder*, which the browser has no use for —
//! the client only ever decodes, and it does that through pure-Rust `ruzstd`. Splitting here is
//! what lets the reader compile to `wasm32-unknown-unknown` with no cfg gymnastics.
//!
//! The builder owns encoding and index bookkeeping; **grouping is the caller's job**. That split is
//! deliberate: which blobs belong together is a question about the game (a map's tile block, an
//! `Interface\` subtree, a doodad directory) and belongs in the packer that knows about WoW, not in
//! the container format, which should stay ignorant of what it is carrying.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{
    Codec, Error, PackClass, PackMeta, Result, FRAME_HEADER_LEN, INDEX_HEADER_LEN, MAGIC_INDEX,
    MAGIC_PACK, MAX_OBJECT_BYTES, PACK_HEADER_LEN, PACK_REC_LEN,
};

/// Compression is not free and, on this corpus, frequently not worth it: BLPs hold DXT blocks and
/// the sound archives are MP3, which zstd measures at 87% — paying CPU on both ends to save an
/// eighth. A frame is stored verbatim unless compressing it beats this fraction of the original.
const WORTH_COMPRESSING: f64 = 0.98;

/// One pack the caller wants built: a locality group, already ordered.
pub struct PackPlan {
    /// `(key, bytes)` in the order they should sit in the pack. Order is the caller's locality
    /// decision and is preserved exactly, because adjacency in the file *is* the prefetch.
    pub blobs: Vec<(u64, Vec<u8>)>,
    /// [`PackClass::Solid`] asks for one stream over the whole body; the builder still measures and
    /// falls back to [`PackClass::Rangeable`] if solid does not actually win, so a bad grouping
    /// guess costs ratio, never range-extractability.
    pub prefer: PackClass,
}

/// Accumulates packs, then writes `index.bpi`.
pub struct PackSetBuilder {
    root: PathBuf,
    level: i32,
    max_object_bytes: u32,
    packs: Vec<PackMeta>,
    /// `(key, pack_id, frame_off, frame_len)`, sorted at [`PackSetBuilder::finish`].
    entries: Vec<(u64, u32, u32, u32)>,
    /// How many of `entries` came from [`PackSetBuilder::from_index`]: a key a later entry also
    /// holds is dropped from these at [`PackSetBuilder::finish`].
    seeded: usize,
    corpus_bytes: u64,
}

impl PackSetBuilder {
    pub fn new(root: impl AsRef<Path>, level: i32) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            level,
            max_object_bytes: MAX_OBJECT_BYTES,
            packs: Vec::new(),
            entries: Vec::new(),
            seeded: 0,
            corpus_bytes: 0,
        }
    }

    /// Start from a container already on disk at `root`: its packs keep their ids, its objects are
    /// not touched, and packs committed after this take the next ids. A key a later pack also holds
    /// is answered from the later pack; the earlier copy stays in its object, unreferenced. This is
    /// how an archive is laid over a built corpus without re-packing (and re-uploading) the corpus.
    ///
    /// The index's corpus total is carried as it was and grows by each new blob; a superseded blob
    /// is not subtracted (the index does not record its raw length), so the total may overstate by
    /// what was superseded.
    pub fn from_index(root: impl AsRef<Path>, level: i32, index: &crate::Index) -> Result<Self> {
        let packs = (0..index.pack_count())
            .map(|id| {
                index
                    .pack(id)
                    .ok_or_else(|| Error::BadIndex(format!("pack {id} has no readable row")))
            })
            .collect::<Result<Vec<_>>>()?;
        let entries: Vec<_> = index
            .iter()
            .map(|(k, at)| (k, at.pack_id, at.frame_off, at.frame_len))
            .collect();
        Ok(Self {
            root: root.as_ref().to_path_buf(),
            level,
            max_object_bytes: MAX_OBJECT_BYTES,
            seeded: entries.len(),
            packs,
            entries,
            corpus_bytes: index.corpus_bytes(),
        })
    }

    /// Lower the object cap (tests use this to exercise the overflow path without building 24 MiB
    /// of anything). Never raise it past [`MAX_OBJECT_BYTES`] — the headroom under a static host's
    /// 25 MiB is the point.
    pub fn with_max_object_bytes(mut self, cap: u32) -> Self {
        self.max_object_bytes = cap.min(MAX_OBJECT_BYTES);
        self
    }

    pub fn max_object_bytes(&self) -> u32 {
        self.max_object_bytes
    }

    pub fn pack_count(&self) -> u32 {
        self.packs.len() as u32
    }

    /// Encode one plan, write its `.bpk`, and record its entries. The sequential convenience path.
    ///
    /// A build over the real corpus should use [`encode_pack`] + [`PackSetBuilder::add_encoded`]
    /// instead: encoding 6.3 GiB at a useful zstd level is the packer's whole cost, and it
    /// parallelises perfectly per pack, which this path cannot.
    pub fn add(&mut self, plan: &PackPlan) -> Result<u32> {
        let pack_id = self.packs.len() as u32;
        let encoded = encode_pack(pack_id, plan, self.level, self.max_object_bytes)?;
        self.add_encoded(encoded)
    }

    /// Write a pack encoded elsewhere and commit it.
    pub fn add_encoded(&mut self, e: EncodedPack) -> Result<u32> {
        let rec = e.write_to(&self.root)?;
        self.record(rec)
    }

    /// Commit a pack that has **already been written** — metadata only.
    ///
    /// This is the seam that keeps a 3 GiB build inside a small working set. The obvious shape,
    /// encoding every pack in parallel and committing the results afterwards, holds the entire
    /// output in memory first (measured: 2 GB and climbing, heading for 3.3). Writing inside the
    /// parallel task and carrying only [`PackRecord`] out of it costs ~36 bytes per pack plus 20
    /// per blob — under 2 MB for the whole corpus.
    ///
    /// Ids must still arrive in order: a pack names its own id in its header and its path derives
    /// from it, so accepting them out of order would silently mislabel objects. A concurrent caller
    /// assigns ids up front from its (deterministic) plan order and sorts the records back.
    pub fn record(&mut self, rec: PackRecord) -> Result<u32> {
        let pack_id = self.packs.len() as u32;
        if rec.pack_id != pack_id {
            return Err(Error::BadIndex(format!(
                "pack {} committed out of order (next id is {pack_id})",
                rec.pack_id
            )));
        }
        self.packs.push(rec.meta);
        for (key, off, len, raw_len) in rec.locs {
            self.entries.push((key, pack_id, off, len));
            self.corpus_bytes += raw_len as u64;
        }
        Ok(pack_id)
    }

    /// Total uncompressed bytes committed so far.
    pub fn corpus_bytes(&self) -> u64 {
        self.corpus_bytes
    }

    /// Write `index.bpi` and consume the builder.
    ///
    /// Fails on a duplicate key rather than silently keeping one of the two: a collision in the
    /// `(name_a, name_b)` pair means two files the archive itself cannot tell apart, which is
    /// something to be told about, not to paper over.
    pub fn finish(mut self) -> Result<PathBuf> {
        // A seeded entry a new pack also holds is superseded by it.
        if self.seeded > 0 {
            let fresh: std::collections::HashSet<u64> =
                self.entries[self.seeded..].iter().map(|e| e.0).collect();
            let seeded = self.seeded;
            let mut i = 0;
            self.entries.retain(|e| {
                let keep = i >= seeded || !fresh.contains(&e.0);
                i += 1;
                keep
            });
        }
        self.entries.sort_unstable_by_key(|e| e.0);
        if let Some(w) = self.entries.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(Error::BadIndex(format!(
                "duplicate key {:#018x}: packs {} and {} both claim it",
                w[0].0, w[0].1, w[1].1
            )));
        }

        let n = self.entries.len();
        let packs_off = INDEX_HEADER_LEN;
        let keys_off = packs_off + self.packs.len() * PACK_REC_LEN;
        let locs_off = keys_off + n * 8;
        let lens_off = locs_off + n * 8;
        let total = lens_off + n * 4;

        // The build id is a digest of the ordered entry stream: two runs over the same corpus give
        // the same value, and any change to what is where gives a different one. The client uses it
        // to namespace its Cache Storage, so a re-pack invalidates rather than mixes.
        let build_id = fnv1a64_entries(&self.entries);

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC_INDEX);
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&1u32.to_le_bytes()); // flags: bit0 = keys sorted ascending
        out.extend_from_slice(&(self.packs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&(packs_off as u32).to_le_bytes());
        out.extend_from_slice(&(keys_off as u32).to_le_bytes());
        out.extend_from_slice(&(locs_off as u32).to_le_bytes());
        out.extend_from_slice(&(lens_off as u32).to_le_bytes());
        out.extend_from_slice(&self.corpus_bytes.to_le_bytes());
        out.extend_from_slice(&build_id.to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // reserved
        debug_assert_eq!(out.len(), INDEX_HEADER_LEN);

        for p in &self.packs {
            out.extend_from_slice(&p.size_wire.to_le_bytes());
            out.extend_from_slice(&p.size_raw.to_le_bytes());
            out.push(p.codec_pack as u8);
            out.push(0); // reserved (the per-frame codec lives in the frame header)
            out.push(p.class as u8);
            out.push(0); // reserved
            out.extend_from_slice(&p.crc32.to_le_bytes());
        }
        for e in &self.entries {
            out.extend_from_slice(&e.0.to_le_bytes());
        }
        for e in &self.entries {
            out.extend_from_slice(&e.1.to_le_bytes());
            out.extend_from_slice(&e.2.to_le_bytes());
        }
        for e in &self.entries {
            out.extend_from_slice(&e.3.to_le_bytes());
        }
        debug_assert_eq!(out.len(), total);

        std::fs::create_dir_all(&self.root).map_err(io_err)?;
        let path = self.root.join("index.bpi");
        let mut f = std::fs::File::create(&path).map_err(io_err)?;
        f.write_all(&out).map_err(io_err)?;
        Ok(path)
    }
}

/// One pack, encoded but not yet committed. Produced by [`encode_pack`], which is pure — no I/O, no
/// shared state — so a build can run one of these per core and hand the results to
/// [`PackSetBuilder::add_encoded`] in id order.
pub struct EncodedPack {
    pub pack_id: u32,
    /// The `.bpk` file's exact bytes, header included.
    pub wire: Vec<u8>,
    pub meta: PackMeta,
    /// `(key, frame_off, frame_len, raw_len)` per blob, in pack order.
    pub locs: Vec<(u64, u32, u32, u32)>,
}

/// What is left of an [`EncodedPack`] once its bytes are on disk: everything the index needs and
/// nothing that scales with the pack's size.
pub struct PackRecord {
    pub pack_id: u32,
    pub meta: PackMeta,
    pub locs: Vec<(u64, u32, u32, u32)>,
}

impl EncodedPack {
    /// Write the `.bpk` and drop its bytes, keeping only what [`PackSetBuilder::record`] needs.
    ///
    /// Consuming `self` is the point: it makes it impossible to write a pack and then keep holding
    /// the megabytes it was made of, which is exactly how the first version of the build reached a
    /// multi-gigabyte working set.
    pub fn write_to(self, root: &Path) -> Result<PackRecord> {
        let path = root.join(crate::pack_path(self.pack_id));
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(io_err)?;
        }
        std::fs::write(&path, &self.wire).map_err(io_err)?;
        Ok(PackRecord {
            pack_id: self.pack_id,
            meta: self.meta,
            locs: self.locs,
        })
    }
}

/// Encode one locality group into a pack.
///
/// Errors with [`Error::Truncated`] if the result would exceed `max_object_bytes` — the caller
/// answers by splitting the group and retrying, which is what makes the cap a property the output
/// *has* rather than one it gets audited for after a multi-gigabyte build.
pub fn encode_pack(
    pack_id: u32,
    plan: &PackPlan,
    level: i32,
    max_object_bytes: u32,
) -> Result<EncodedPack> {
    // Frame the body per-blob-compressed first; that is the Rangeable shape and the baseline a
    // solid attempt has to beat.
    let (framed_body, framed_locs) = frame_blobs(&plan.blobs, true, level)?;
    let solid_wanted = plan.prefer == PackClass::Solid && plan.blobs.len() > 1;

    let (body, locs, codec_pack, size_raw, class) = if solid_wanted {
        let (stored_body, stored_locs) = frame_blobs(&plan.blobs, false, level)?;
        let squeezed = zstd::encode_all(&stored_body[..], level)
            .map_err(|e| Error::Decompress(format!("zstd encode: {e}")))?;
        // Solid only earns its inflexibility if it actually beats per-frame compression. A grouping
        // guess that was wrong therefore costs ratio, never range-extractability.
        if squeezed.len() < framed_body.len() {
            let raw = stored_body.len() as u32;
            (squeezed, stored_locs, Codec::Zstd, raw, PackClass::Solid)
        } else {
            let raw = framed_body.len() as u32;
            (
                framed_body,
                framed_locs,
                Codec::Store,
                raw,
                PackClass::Rangeable,
            )
        }
    } else {
        let raw = framed_body.len() as u32;
        let class = if plan.blobs.len() == 1 {
            PackClass::Singleton
        } else {
            PackClass::Rangeable
        };
        (framed_body, framed_locs, Codec::Store, raw, class)
    };

    let wire_len = PACK_HEADER_LEN + body.len();
    if wire_len > max_object_bytes as usize {
        return Err(Error::Truncated(format!(
            "pack of {} blobs encodes to {wire_len} bytes, over the {max_object_bytes} cap",
            plan.blobs.len()
        )));
    }

    let mut wire = Vec::with_capacity(wire_len);
    wire.extend_from_slice(&MAGIC_PACK);
    wire.extend_from_slice(&pack_id.to_le_bytes());
    wire.extend_from_slice(&body);

    Ok(EncodedPack {
        pack_id,
        meta: PackMeta {
            size_wire: wire.len() as u32,
            size_raw,
            codec_pack,
            class,
            crc32: crc32fast::hash(&wire),
        },
        wire,
        locs,
    })
}

/// Lay blobs out as frames. `compress` picks whether each frame's payload is squeezed individually
/// (the Rangeable shape) or stored (so the whole body can be squeezed as one).
#[allow(clippy::type_complexity)]
fn frame_blobs(
    blobs: &[(u64, Vec<u8>)],
    compress: bool,
    level: i32,
) -> Result<(Vec<u8>, Vec<(u64, u32, u32, u32)>)> {
    let mut body = Vec::new();
    let mut locs = Vec::with_capacity(blobs.len());
    for (key, raw) in blobs {
        let off = body.len() as u32;
        // Store when compression doesn't pay — see WORTH_COMPRESSING. The choice is per frame, so
        // the frame records which way it went; nothing above it has to agree.
        let (codec, payload) = if compress {
            let squeezed = zstd::encode_all(&raw[..], level)
                .map_err(|e| Error::Decompress(format!("zstd encode: {e}")))?;
            if (squeezed.len() as f64) < raw.len() as f64 * WORTH_COMPRESSING {
                (Codec::Zstd, squeezed)
            } else {
                (Codec::Store, raw.clone())
            }
        } else {
            (Codec::Store, raw.clone())
        };
        body.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        body.push(codec as u8);
        body.extend_from_slice(&payload);
        locs.push((
            *key,
            off,
            (FRAME_HEADER_LEN + payload.len()) as u32,
            raw.len() as u32,
        ));
    }
    Ok((body, locs))
}

/// FNV-1a over the entry stream. Not a security hash and not trying to be — it identifies a build
/// for cache namespacing, where the only requirement is that a different layout gives a different
/// number.
fn fnv1a64_entries(entries: &[(u64, u32, u32, u32)]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    for e in entries {
        eat(&e.0.to_le_bytes());
        eat(&e.1.to_le_bytes());
        eat(&e.2.to_le_bytes());
        eat(&e.3.to_le_bytes());
    }
    h
}

fn io_err(e: std::io::Error) -> Error {
    Error::Truncated(format!("io: {e}"))
}
