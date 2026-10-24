//! **BENPACK** — the container the browser client streams its game data from.
//!
//! The web build cannot mount MPQ archives, and the host it used to fetch single files from
//! (`wenilla-host`'s `/data/<name>`) is going away: the corpus moves to a static object store
//! (a static asset host) with no server-side logic at all. That store imposes the shape of
//! this format:
//!
//! - **20,000 objects, 25 MiB each** (free tier). 79,735 files cannot each be an object, so blobs
//!   are grouped into packs. We cap a pack at [`MAX_OBJECT_BYTES`] (24 MiB) rather than the platform's
//!   25, because being one byte over is a failed upload and a megabyte of headroom is free.
//! - **No HTTP `Range` support** — measured, not assumed: a CDN edge only slices ranges for
//!   objects it has *stored*, and the default `max-age=0, must-revalidate` means it never stores
//!   one. So the client must be able to work by fetching whole objects. But that may change (an
//!   `immutable` header, or a different host later), so the format also has to *permit* ranged
//!   extraction and let the client pick at runtime.
//!
//! Those two pull in opposite directions, and no single layout satisfies both: a bundle of a
//! thousand 4 KB icons compresses far better as one stream, which destroys range-extractability;
//! a 15 MiB sound file wants to be reachable without downloading its neighbours. So **the layout is
//! flagged per pack** ([`PackClass`]), the packer chooses per group, and the client honours whatever
//! it finds. One hosted tree is correct — and near-optimal — on a host with ranges and on a host
//! without.
//!
//! ## Keys are MPQ's, not ours
//!
//! An entry is identified by [`benilla_mpq::name_key`]: the `(name_a, name_b)` hash pair the archive
//! itself resolves on. That is deliberate and it is the reason no name strings ship in the index.
//! MPQ's name equivalence — case-insensitive, `/` ≡ `\` — comes along for free instead of being
//! restated here where the two copies could drift, and two names that collide in this container are
//! already indistinguishable inside the archive it came from. 79,735 entries × 20 bytes is a 1.5 MB
//! index with *complete* coverage, against the 4.9 MB JSON name list it replaces, and the client
//! binary-searches it in place without parsing it into anything.
//!
//! ## Layout
//!
//! ```text
//! index.bpi   64-byte header, then four parallel arrays:
//!             packs[pack_count]  16 B each — see PackMeta
//!             keys[N]             8 B each — u64, ASCENDING (the binary search touches only this)
//!             locs[N]             8 B each — u32 pack_id, u32 frame_off
//!             lens[N]             4 B each — u32 frame_len
//!
//! p/<hi>/<id>.bpk
//!             "BPK1", u32 pack_id, then the body.
//!             The body is a run of frames: u32 len_r, u8 codec, then the payload.
//!             A frame carries its OWN codec because this corpus is mixed: within one pack a
//!             FrameXML file compresses to a sixth and the MP3 beside it to seven eighths, so the
//!             packer stores some frames and squeezes others. A per-pack codec could not describe
//!             that, and a ranged fetch — which delivers a frame and nothing else — would have
//!             nowhere else to learn it from.
//!             `frame_off` is relative to the start of the body (i.e. past the 8-byte header),
//!             ALWAYS in the body's *decoded* bytes — which for a STORE-coded pack are the wire
//!             bytes, so a frame is directly range-addressable there.
//! ```
//!
//! Struct-of-arrays, not array-of-structs, on purpose: the lookup path touches only `keys`, so the
//! 8N bytes of incompressible hash live apart from `locs`/`lens`, which are near-monotone and
//! compress well on the wire.

use benilla_bytes::ByteExt;

#[cfg(feature = "write")]
mod write;
#[cfg(feature = "write")]
pub use write::{encode_pack, EncodedPack, PackPlan, PackRecord, PackSetBuilder};

/// Finding a built container on disk, for tests and tools ([`pack_index_or_skip`]). Not compiled
/// for the browser, which is served a container rather than searching for one.
#[cfg(not(target_arch = "wasm32"))]
pub mod find;

/// `index.bpi`'s magic. The trailing byte is the format generation: a client that fetches an index
/// it cannot read must say so, not misparse it.
pub const MAGIC_INDEX: [u8; 8] = *b"BENPACK\x01";
/// Every `.bpk` opens with this. Cheap, and load-bearing: a static host misconfigured to answer a
/// missing object with `index.html` (a static host's `not_found_handling: single-page-application`
/// does exactly that, with a 200) would otherwise hand HTML to a zstd decoder and corrupt silently.
pub const MAGIC_PACK: [u8; 4] = *b"BPK1";

/// Bytes before the first array in `index.bpi`.
pub const INDEX_HEADER_LEN: usize = 64;
/// Bytes before the first frame in a `.bpk` — `MAGIC_PACK` plus the pack's own id.
pub const PACK_HEADER_LEN: usize = 8;
/// One [`PackMeta`] on disk.
pub const PACK_REC_LEN: usize = 16;
/// One frame's own header: the uncompressed length of what follows, then that frame's codec.
pub const FRAME_HEADER_LEN: usize = 5;

/// The object-size cap the packer enforces: **24 MiB**, under a static host's 25 MiB per-asset limit.
/// The MiB of headroom is deliberate — an object one byte over the platform limit fails the whole
/// site, and nothing in the 1.12.1 corpus is anywhere near this (the largest blob is 15.5 MiB).
pub const MAX_OBJECT_BYTES: u32 = 24 * 1024 * 1024;

/// How a run of bytes was compressed. Applied at two levels independently: to a whole pack body
/// ([`PackMeta::codec_pack`]) and to each frame within it (the frame's own header byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    /// Verbatim. Not a fallback but the *right* answer for most of this corpus: BLPs carry DXT
    /// blocks and the sound archives are MP3, so re-compressing them costs CPU to save nothing
    /// (measured: audio compresses to 87%). The packer stores whenever compression doesn't pay.
    Store = 0,
    Zstd = 1,
    /// Raw DEFLATE. Kept because the browser can inflate it natively via `DecompressionStream`,
    /// which is a real option for the whole-pack path if zstd-in-wasm ever shows up in a profile.
    Deflate = 2,
}

impl Codec {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Codec::Store),
            1 => Some(Codec::Zstd),
            2 => Some(Codec::Deflate),
            _ => None,
        }
    }
}

/// Which of the two irreconcilable objectives a pack was built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PackClass {
    /// Body stored verbatim, each frame compressed on its own. A frame is a contiguous byte range
    /// of the served object, so a Range-capable host can deliver exactly one blob.
    Rangeable = 0,
    /// The whole body is one compressed stream. Materially better on a bundle of many small similar
    /// files, and unusable in parts — the client must fetch it whole. That is not the loss it
    /// sounds like: these are locality groups, so the rest of the pack is what it will ask for next.
    Solid = 1,
    /// One blob, alone in its object, for blobs too big to bundle. Always [`Codec::Store`] at the
    /// pack level so the single frame stays range-addressable.
    Singleton = 2,
}

impl PackClass {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(PackClass::Rangeable),
            1 => Some(PackClass::Solid),
            2 => Some(PackClass::Singleton),
            _ => None,
        }
    }
}

/// One pack's row in the index — everything the client needs to decide *how* to fetch it before it
/// has fetched anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackMeta {
    /// Length of the `.bpk` file as served, header included. Lets the client size a buffer, and
    /// lets it recognise a `Range` request that came back as a full 200 (which a static host does on a
    /// cache miss) rather than mistaking the body for a partial one.
    pub size_wire: u32,
    /// Length of the body after `codec_pack` is undone. Equals `size_wire - PACK_HEADER_LEN` when
    /// the pack is [`Codec::Store`]d.
    pub size_raw: u32,
    pub codec_pack: Codec,
    pub class: PackClass,
    /// CRC32 of the whole `.bpk`. Integrity, and a fast way to tell a truncated download from a
    /// format bug when something goes wrong in a browser you cannot attach a debugger to.
    pub crc32: u32,
}

impl PackMeta {
    /// Whether one frame can be pulled out of this pack with a byte range. False for [`Solid`], where
    /// the frame's bytes do not exist until the whole body is decompressed.
    ///
    /// [`Solid`]: PackClass::Solid
    pub fn is_range_extractable(&self) -> bool {
        self.codec_pack == Codec::Store
    }
}

/// Where one blob lives: which pack, and the byte range of its frame within that pack's decoded body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Located {
    pub pack_id: u32,
    /// Offset of the frame from the start of the pack **body** (past [`PACK_HEADER_LEN`]).
    pub frame_off: u32,
    /// Length of the frame, its own 4-byte header included.
    pub frame_len: u32,
}

impl Located {
    /// The byte range to ask a Range-capable host for — offsets into the served object, so the
    /// 8-byte pack header is added back here. Inclusive at both ends, as HTTP wants it.
    pub fn wire_range(&self) -> (u64, u64) {
        let start = PACK_HEADER_LEN as u64 + self.frame_off as u64;
        (start, start + self.frame_len as u64 - 1)
    }
}

/// What can go wrong reading a container.
#[derive(Debug)]
pub enum Error {
    /// The bytes are not a BENPACK index, or are a generation this build cannot read.
    BadIndex(String),
    /// A `.bpk` did not start with [`MAGIC_PACK`], or carried the wrong pack id. Usually not a
    /// corrupt file at all — usually a static host answering a 404 with an HTML page.
    BadPack(String),
    /// A header offset or length points outside the buffer it describes.
    Truncated(String),
    Decompress(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadIndex(m) => write!(f, "bad pack index: {m}"),
            Error::BadPack(m) => write!(f, "bad pack: {m}"),
            Error::Truncated(m) => write!(f, "truncated: {m}"),
            Error::Decompress(m) => write!(f, "decompress: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The container's index: owns the raw `index.bpi` bytes and reads them **in place**.
///
/// Nothing is parsed into a `HashMap` or a `Vec<Entry>` on load. At 79,735 entries the alternative
/// costs a second copy plus per-entry allocation on a client whose whole boot budget is the thing
/// this format exists to protect; a binary search over the sorted key array is ~17 probes into
/// bytes that are already there.
pub struct Index {
    bytes: Vec<u8>,
    pack_count: u32,
    entry_count: u32,
    packs_off: usize,
    keys_off: usize,
    locs_off: usize,
    lens_off: usize,
    corpus_bytes: u64,
    build_id: u64,
}

impl Index {
    /// Parse the header of an `index.bpi` and validate that every array it declares is actually
    /// inside the buffer, so no later lookup has to re-check.
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let bad = |m: String| Error::BadIndex(m);
        if bytes.len() < INDEX_HEADER_LEN {
            return Err(bad(format!(
                "{} bytes is shorter than the {INDEX_HEADER_LEN}-byte header",
                bytes.len()
            )));
        }
        if bytes[..8] != MAGIC_INDEX {
            return Err(bad(format!(
                "magic is {:02X?}, expected {:02X?}",
                &bytes[..8],
                MAGIC_INDEX
            )));
        }
        let g = |o: usize| bytes.u32_at(o).ok_or_else(|| bad("short header".into()));
        let version = g(8)?;
        if version != 1 {
            return Err(bad(format!("format version {version}, this build reads 1")));
        }
        let pack_count = g(16)?;
        let entry_count = g(20)?;
        let packs_off = g(24)? as usize;
        let keys_off = g(28)? as usize;
        let locs_off = g(32)? as usize;
        let lens_off = g(36)? as usize;
        let corpus_bytes = bytes.u64_at(40).ok_or_else(|| bad("short header".into()))?;
        let build_id = bytes.u64_at(48).ok_or_else(|| bad("short header".into()))?;

        // Validate every array's extent once, here, so `locate` can index without re-checking.
        let n = entry_count as usize;
        for (name, off, len) in [
            ("packs", packs_off, pack_count as usize * PACK_REC_LEN),
            ("keys", keys_off, n * 8),
            ("locs", locs_off, n * 8),
            ("lens", lens_off, n * 4),
        ] {
            let end = off.checked_add(len).ok_or_else(|| {
                bad(format!(
                    "{name} array extent overflows (off {off}, len {len})"
                ))
            })?;
            if end > bytes.len() {
                return Err(bad(format!(
                    "{name} array runs to {end}, past the {}-byte index",
                    bytes.len()
                )));
            }
        }

        Ok(Index {
            bytes,
            pack_count,
            entry_count,
            packs_off,
            keys_off,
            locs_off,
            lens_off,
            corpus_bytes,
            build_id,
        })
    }

    /// Number of blobs the container holds.
    pub fn len(&self) -> usize {
        self.entry_count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    pub fn pack_count(&self) -> u32 {
        self.pack_count
    }

    /// Total uncompressed size of the corpus — for a boot progress bar that can say MB rather than
    /// "N of M files".
    pub fn corpus_bytes(&self) -> u64 {
        self.corpus_bytes
    }

    /// Identifies this build of the container. The client namespaces its Cache Storage on it, so a
    /// re-pack invalidates cleanly instead of mixing packs from two builds.
    pub fn build_id(&self) -> u64 {
        self.build_id
    }

    /// The `i`-th key. Only called from the binary search; extents were validated in [`Index::parse`].
    #[inline]
    fn key_at(&self, i: usize) -> u64 {
        self.bytes.u64_at(self.keys_off + i * 8).unwrap_or(u64::MAX)
    }

    /// Where `key`'s blob lives, or `None` if the container has no such blob. Pure computation — no
    /// I/O, no allocation, no network. This is what makes `Chain::contains` free on the web target,
    /// where it used to be a blocking `HEAD` request (2,145 of them in one world entry).
    pub fn locate(&self, key: u64) -> Option<Located> {
        let n = self.entry_count as usize;
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.key_at(mid).cmp(&key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return self.located_at(mid),
            }
        }
        None
    }

    fn located_at(&self, i: usize) -> Option<Located> {
        Some(Located {
            pack_id: self.bytes.u32_at(self.locs_off + i * 8)?,
            frame_off: self.bytes.u32_at(self.locs_off + i * 8 + 4)?,
            frame_len: self.bytes.u32_at(self.lens_off + i * 4)?,
        })
    }

    /// Metadata for one pack.
    pub fn pack(&self, pack_id: u32) -> Option<PackMeta> {
        if pack_id >= self.pack_count {
            return None;
        }
        let o = self.packs_off + pack_id as usize * PACK_REC_LEN;
        Some(PackMeta {
            size_wire: self.bytes.u32_at(o)?,
            size_raw: self.bytes.u32_at(o + 4)?,
            codec_pack: Codec::from_u8(self.bytes.u8_at(o + 8)?)?,
            // byte 9 reserved
            class: PackClass::from_u8(self.bytes.u8_at(o + 10)?)?,
            crc32: self.bytes.u32_at(o + 12)?,
        })
    }

    /// Every `(key, Located)` in key order. For tooling and the round-trip test; the client never
    /// enumerates.
    pub fn iter(&self) -> impl Iterator<Item = (u64, Located)> + '_ {
        (0..self.entry_count as usize).filter_map(|i| Some((self.key_at(i), self.located_at(i)?)))
    }
}

/// The path a pack is served at, relative to the container root: `p/<high byte>/<id>.bpk`.
///
/// Derived from the id rather than stored, so the index carries no path table at all. The two-level
/// fan-out keeps any one directory under ~256 entries, which matters to the tools that walk this
/// tree far more than to the object store, and costs nothing.
pub fn pack_path(pack_id: u32) -> String {
    format!("p/{:02x}/{:04x}.bpk", (pack_id >> 8) & 0xff, pack_id)
}

/// Strip and check a `.bpk`'s header, returning the body. This is where a static host's 404-as-HTML
/// is caught: the magic check turns silent corruption into a legible error.
pub fn pack_body(wire: &[u8], pack_id: u32) -> Result<&[u8]> {
    if wire.len() < PACK_HEADER_LEN {
        return Err(Error::BadPack(format!(
            "pack {pack_id}: {} bytes, shorter than its header",
            wire.len()
        )));
    }
    if wire[..4] != MAGIC_PACK {
        return Err(Error::BadPack(format!(
            "pack {pack_id}: magic is {:02X?}, not BPK1 — a 404 page or a truncated download, not a pack",
            &wire[..4]
        )));
    }
    let got = wire.u32_at(4).unwrap_or(u32::MAX);
    if got != pack_id {
        return Err(Error::BadPack(format!(
            "pack {pack_id}: object identifies itself as pack {got}"
        )));
    }
    Ok(&wire[PACK_HEADER_LEN..])
}

/// Undo a pack's body-level codec, giving the bytes `Located::frame_off` indexes into.
///
/// For a [`Codec::Store`] pack this borrows — the wire bytes already *are* the body, which is the
/// whole reason a range fetch can address a frame directly.
pub fn decode_pack_body<'a>(body: &'a [u8], meta: &PackMeta) -> Result<std::borrow::Cow<'a, [u8]>> {
    match meta.codec_pack {
        Codec::Store => Ok(std::borrow::Cow::Borrowed(body)),
        c => Ok(std::borrow::Cow::Owned(decompress(
            c,
            body,
            meta.size_raw as usize,
            "pack body",
        )?)),
    }
}

/// Pull one blob out of a pack's decoded body.
pub fn frame_from_body(body: &[u8], at: &Located) -> Result<Vec<u8>> {
    let start = at.frame_off as usize;
    let end = start
        .checked_add(at.frame_len as usize)
        .ok_or_else(|| Error::Truncated("frame extent overflows".into()))?;
    let frame = body.get(start..end).ok_or_else(|| {
        Error::Truncated(format!(
            "frame {start}..{end} past a {}-byte body",
            body.len()
        ))
    })?;
    frame_bytes(frame)
}

/// Decode a frame that has already been isolated — the shape a **ranged** fetch delivers, where the
/// response body is exactly one frame and no surrounding pack was ever downloaded.
pub fn frame_bytes(frame: &[u8]) -> Result<Vec<u8>> {
    let short = || Error::Truncated("frame shorter than its header".into());
    let len_r = frame.u32_at(0).ok_or_else(short)? as usize;
    let codec = Codec::from_u8(frame.u8_at(4).ok_or_else(short)?)
        .ok_or_else(|| Error::Decompress(format!("frame: unknown codec {}", frame[4])))?;
    let payload = frame.get(FRAME_HEADER_LEN..).ok_or_else(short)?;
    match codec {
        Codec::Store => {
            if payload.len() != len_r {
                return Err(Error::Truncated(format!(
                    "stored frame declares {len_r} bytes but carries {}",
                    payload.len()
                )));
            }
            Ok(payload.to_vec())
        }
        c => decompress(c, payload, len_r, "frame"),
    }
}

fn decompress(codec: Codec, input: &[u8], expect: usize, what: &str) -> Result<Vec<u8>> {
    let out = match codec {
        Codec::Store => input.to_vec(),
        Codec::Zstd => {
            // `decode_all_to_vec` fills up to the vector's CAPACITY and truncates to what it wrote,
            // so the exact-size reservation is what bounds it — and `expect` is exact, from the index.
            let mut out = Vec::with_capacity(expect);
            ruzstd::decoding::FrameDecoder::new()
                .decode_all_to_vec(input, &mut out)
                .map_err(|e| Error::Decompress(format!("{what}: zstd: {e}")))?;
            out
        }
        Codec::Deflate => miniz_oxide::inflate::decompress_to_vec(input)
            .map_err(|e| Error::Decompress(format!("{what}: deflate: {e:?}")))?,
    };
    if out.len() != expect {
        return Err(Error::Decompress(format!(
            "{what}: decoded {} bytes, index says {expect}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_path_fans_out_by_high_byte() {
        assert_eq!(pack_path(0), "p/00/0000.bpk");
        assert_eq!(pack_path(0x1234), "p/12/1234.bpk");
        assert_eq!(pack_path(0xff), "p/00/00ff.bpk");
    }

    /// A missing object answered with an HTML page — the exact failure a misconfigured
    /// `not_found_handling` produces — must be a legible error, not bytes fed to a decompressor.
    #[test]
    fn an_html_404_page_is_rejected_as_a_pack() {
        let html = b"<!DOCTYPE html><html><head><title>404";
        match pack_body(html, 7) {
            Err(Error::BadPack(m)) => assert!(m.contains("404 page"), "{m}"),
            other => panic!("expected BadPack, got {other:?}"),
        }
    }

    #[test]
    fn a_pack_that_names_the_wrong_id_is_rejected() {
        let mut wire = MAGIC_PACK.to_vec();
        wire.extend_from_slice(&9u32.to_le_bytes());
        match pack_body(&wire, 7) {
            Err(Error::BadPack(m)) => assert!(m.contains("identifies itself as pack 9"), "{m}"),
            other => panic!("expected BadPack, got {other:?}"),
        }
    }

    #[test]
    fn wire_range_covers_the_frame_and_skips_the_pack_header() {
        let at = Located {
            pack_id: 1,
            frame_off: 100,
            frame_len: 50,
        };
        assert_eq!(at.wire_range(), (108, 157));
        // Inclusive at both ends: the range spans exactly frame_len bytes.
        let (a, b) = at.wire_range();
        assert_eq!(b - a + 1, 50);
    }

    #[test]
    fn a_short_or_unmagicked_index_is_rejected_rather_than_misread() {
        assert!(matches!(Index::parse(vec![0; 8]), Err(Error::BadIndex(_))));
        assert!(matches!(
            Index::parse(vec![0; INDEX_HEADER_LEN]),
            Err(Error::BadIndex(_))
        ));
    }
}
