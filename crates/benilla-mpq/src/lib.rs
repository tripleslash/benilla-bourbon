//! A read-only MPQ reader for the 1.12.1 `Data/` chain: format V1/V2, every file stored whole or
//! `COMPRESS`-flagged and sectored, no single-unit files, no PTCH patches. Anything else is a hard
//! error. An encrypted file reads by name only, as its key derives from the name: 1.12.1 ships
//! none, and a Turtle WoW 1.18.1 patch archive encrypts its `(listfile)`.
//!
//! The patch archives carry delete markers (flag `0x02000000`, size 0): the path is deleted from
//! the composite chain. [`Archive::contains`] reports one present and [`Archive::read_file`]
//! refuses it with [`Error::NotFound`], so a chain walker asks [`Archive::is_delete_marker`] and
//! stops there.
//!
//! [`Archive`] is a cheap `Arc` handle; each read opens its own file handle, so reads need no lock.
//! [`Archive::open_file`] reads one file in place, a sector at a time, for a stream.
//! Every allocation sized from a header count or a sector length is capped by the file's size, so
//! a lying header fails with [`Error::Corrupt`].

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use benilla_bytes::{capped, ByteExt};

mod crypto;
use crypto::{decrypt_block, decrypt_bytes, hash_string, hash_type};

/// `MPQ\x1A` as a little-endian `u32`: the archive header signature.
const SIGNATURE: u32 = 0x1A51_504D;
/// `MPQ\x1B`: a user-data header, which points past itself to the real one.
const USERDATA_SIGNATURE: u32 = 0x1B51_504D;

const FLAG_IMPLODE: u32 = 0x0000_0100;
const FLAG_COMPRESS: u32 = 0x0000_0200;
const FLAG_ENCRYPTED: u32 = 0x0001_0000;
/// The file key is offset by the block position and XOR'd with the size (Storm's `MPQ_FILE_FIX_KEY`).
const FLAG_FIX_KEY: u32 = 0x0002_0000;
const FLAG_SINGLE_UNIT: u32 = 0x0100_0000;
/// A patch archive's delete marker (size 0): the path is deleted from the composite chain.
const FLAG_DELETE_MARKER: u32 = 0x0200_0000;
const FLAG_EXISTS: u32 = 0x8000_0000;

/// Hash-table sentinels.
const HASH_EMPTY: u32 = 0xFFFF_FFFF; // never used: ends the probe
const HASH_DELETED: u32 = 0xFFFF_FFFE; // deleted: skip, keep probing

/// One decrypted hash-table entry (16 bytes on disk).
#[derive(Clone, Copy)]
struct HashEntry {
    name_a: u32,
    name_b: u32,
    block_index: u32,
}

/// One decrypted block-table entry (16 bytes on disk).
#[derive(Clone, Copy)]
struct BlockEntry {
    file_pos: u32,
    file_size: u32,
    flags: u32,
}

/// An archive's index, parsed once and immutable, shared behind an `Arc`.
struct Index {
    path: PathBuf,
    /// Byte offset of the MPQ header in the file: 0 in 1.12 data, but found, not assumed.
    archive_offset: u64,
    /// The uncompressed sector size, `512 << block_size`.
    sector_size: usize,
    hash_table: Vec<HashEntry>,
    block_table: Vec<BlockEntry>,
}

/// An open MPQ archive, a cheap `Clone + Send + Sync` handle.
#[derive(Clone)]
pub struct Archive {
    index: Arc<Index>,
}

/// One live hash-table slot, as [`Archive::entries`] reports it: an entry identified by
/// [`name_key`] instead of by name, because most of the corpus has no name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// [`name_key`] of whatever name hashes here: the slot's `(name_a, name_b)` pair as a u64.
    pub key: u64,
    /// Row in the block table, the argument [`Archive::read_block`] takes.
    pub block_index: u32,
    /// Uncompressed size, straight from the block entry. No I/O.
    pub file_size: u32,
    /// This slot is a patch tombstone, not a readable file: a chain walker lets it shadow the
    /// lower-priority copy rather than skip it.
    pub is_delete_marker: bool,
}

/// The archive's own identity for a file name: `(name_a << 32) | name_b`, the pair
/// [`Archive::entries`] reports and the hash-table probe matches on. The hashes fold `/` to `\` and
/// uppercase ASCII, so it inherits MPQ's name equivalence rather than reimplementing it.
pub fn name_key(name: &str) -> u64 {
    (hash_string(name, hash_type::NAME_A) as u64) << 32
        | hash_string(name, hash_type::NAME_B) as u64
}

/// A read error; anything outside the 1.12.1 envelope surfaces here.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    NotMpq,
    NotFound(String),
    /// A file or codec shape 1.12.1 data does not contain.
    Unsupported(String),
    Decompress(String),
    /// A count or length claims more than the archive file holds: a corrupt or truncated archive.
    Corrupt(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::NotMpq => write!(f, "not an MPQ archive (no header signature found)"),
            Error::NotFound(n) => write!(f, "file not in archive: {n}"),
            Error::Unsupported(m) => write!(f, "unsupported (outside 1.12.1 envelope): {m}"),
            Error::Decompress(m) => write!(f, "decompress: {m}"),
            Error::Corrupt(m) => write!(f, "corrupt archive: {m}"),
        }
    }
}

impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

type Result<T> = std::result::Result<T, Error>;

fn to_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Bytes from `pos` to EOF, saturating: the cap for every reservation sized from a header value.
fn avail_from(file_len: u64, pos: u64) -> usize {
    usize::try_from(file_len.saturating_sub(pos)).unwrap_or(usize::MAX)
}

impl Archive {
    /// Open an archive: find the header, then read and decrypt the hash and block tables.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();

        let archive_offset = find_header(&mut file, file_len)?;

        // The 32-byte V1 header: later versions only extend it, and every 1.12 table offset fits
        // a `u32`, so the V2 hi-block table is never needed.
        file.seek(SeekFrom::Start(archive_offset))?;
        let mut hdr = [0u8; 32];
        file.read_exact(&mut hdr)?;
        let signature = hdr.u32_at(0).ok_or(Error::NotMpq)?;
        debug_assert_eq!(signature, SIGNATURE);
        let block_size_shift = hdr.u16_at(14).ok_or(Error::NotMpq)?;
        // A raw header `u16`, capped before `512 << shift` can overflow; 1.12 data uses 3 (4 KiB
        // sectors), and 23 is already 4 GiB.
        if block_size_shift > 23 {
            return Err(Error::Corrupt(format!(
                "block size shift {block_size_shift} exceeds the sanity cap"
            )));
        }
        let sector_size = 512usize << block_size_shift;
        let hash_table_pos = hdr.u32_at(16).ok_or(Error::NotMpq)? as u64;
        let block_table_pos = hdr.u32_at(20).ok_or(Error::NotMpq)? as u64;
        let hash_table_size = hdr.u32_at(24).ok_or(Error::NotMpq)? as usize;
        let block_table_size = hdr.u32_at(28).ok_or(Error::NotMpq)? as usize;

        let hash_table = read_hash_table(
            &mut file,
            archive_offset + hash_table_pos,
            hash_table_size,
            file_len,
        )?;
        let block_table = read_block_table(
            &mut file,
            archive_offset + block_table_pos,
            block_table_size,
            file_len,
        )?;

        Ok(Archive {
            index: Arc::new(Index {
                path,
                archive_offset,
                sector_size,
                hash_table,
                block_table,
            }),
        })
    }

    /// Whether the archive holds `name`, case-insensitive with `/` and `\` alike. No I/O.
    pub fn contains(&self, name: &str) -> bool {
        self.index.find(name).is_some()
    }

    /// The uncompressed size of `name`, from the block table.
    pub fn file_size(&self, name: &str) -> Option<u32> {
        self.index.find(name).map(|b| b.file_size)
    }

    /// Whether `name` is a delete marker: a chain walker stops there instead of falling through.
    pub fn is_delete_marker(&self, name: &str) -> bool {
        self.index
            .find(name)
            .is_some_and(|b| b.flags & FLAG_DELETE_MARKER != 0)
    }

    /// The archive's path.
    pub fn path(&self) -> &Path {
        &self.index.path
    }

    /// Every live entry in the archive by hash slot, not by name: the complete contents, including
    /// the files no `(listfile)` names (the base archives ship none, so a listfile walk would drop
    /// most of the corpus). Hash-table order, stable per archive, so a pack build is deterministic.
    /// Empty, deleted and out-of-range slots are skipped; delete-markers are yielded, flagged, so a
    /// chain walker can shadow the lower-priority copy.
    pub fn entries(&self) -> impl Iterator<Item = Entry> + '_ {
        let idx = &self.index;
        idx.hash_table.iter().filter_map(move |e| {
            if e.block_index == HASH_EMPTY || e.block_index == HASH_DELETED {
                return None;
            }
            let block = idx.block_table.get(e.block_index as usize)?;
            if block.flags & FLAG_EXISTS == 0 {
                return None;
            }
            Some(Entry {
                key: (e.name_a as u64) << 32 | e.name_b as u64,
                block_index: e.block_index,
                file_size: block.file_size,
                is_delete_marker: block.flags & FLAG_DELETE_MARKER != 0,
            })
        })
    }

    /// Read and decompress a file by its internal path (`/` or `\`, case-insensitive), decrypting an
    /// encrypted one with the key its name derives.
    pub fn read_file(&self, name: &str) -> Result<Vec<u8>> {
        self.open_file(name)?.read_all()
    }

    /// Open a file for reading in place: its own OS handle, its sectors decompressed one at a
    /// time as they are read, so a long sound streams without the whole file read first, as the
    /// reference's FMOD streams read the archive through the file callbacks it registers
    /// (`FSOUND_File_SetCallbacks` at `0x7a45e4`). An encrypted file decrypts with the key its
    /// name derives.
    pub fn open_file(&self, name: &str) -> Result<ArchiveFile> {
        let block = self
            .index
            .find(name)
            .ok_or_else(|| Error::NotFound(name.into()))?;
        let key = (block.flags & FLAG_ENCRYPTED != 0).then(|| file_key(name, &block));
        self.open_entry(&block, name, key)
    }

    /// Read and decompress the file at `block_index` — the block-table row [`Archive::entries`]
    /// reported for a hash slot. The **nameless** read: the base content archives carry no
    /// `(listfile)`, so most of `texture.MPQ` is reachable by name but not enumerable, and a tool
    /// that must see the whole archive (the pack builder) has only the hash table to walk.
    ///
    /// An encrypted entry is refused here: its key derives from the name, the one thing this read
    /// does not have ([`Archive::is_encrypted`] says which). Everything else (sector table, codec
    /// mask, sizes) comes from the block entry and the archive itself, so every other entry reads.
    pub fn read_block(&self, block_index: u32) -> Result<Vec<u8>> {
        let block = self
            .index
            .block_table
            .get(block_index as usize)
            .copied()
            .ok_or_else(|| Error::NotFound(format!("block {block_index}")))?;
        self.open_entry(&block, &format!("block {block_index}"), None)?
            .read_all()
    }

    /// Whether the entry at `block_index` is encrypted, so that only [`Archive::read_file`] can read
    /// it. No I/O.
    pub fn is_encrypted(&self, block_index: u32) -> bool {
        self.index
            .block_table
            .get(block_index as usize)
            .is_some_and(|b| b.flags & FLAG_ENCRYPTED != 0)
    }

    /// The shared body of [`Archive::open_file`] and [`Archive::read_block`]: everything after the
    /// block entry is in hand. `name` is used only for error messages, so the nameless caller passes
    /// its block index and gets errors that still say which entry failed. `key` is an encrypted
    /// entry's file key; the nameless caller has none and so refuses one.
    fn open_entry(&self, block: &BlockEntry, name: &str, key: Option<u32>) -> Result<ArchiveFile> {
        let idx = &self.index;
        if block.flags & FLAG_EXISTS == 0 {
            return Err(Error::NotFound(name.into()));
        }
        // A delete marker is not an empty file: the composite deleted this path.
        if block.flags & FLAG_DELETE_MARKER != 0 {
            return Err(Error::NotFound(name.into()));
        }
        // Encrypted with no key to hand: refuse rather than guess.
        let key = match (block.flags & FLAG_ENCRYPTED != 0, key) {
            (false, _) => None,
            (true, Some(key)) => Some(key),
            (true, None) => return Err(Error::Unsupported(format!("encrypted file {name}"))),
        };
        if block.flags & FLAG_SINGLE_UNIT != 0 {
            return Err(Error::Unsupported(format!("single-unit file {name}")));
        }

        let file = File::open(&idx.path)?;
        let file_pos = idx.archive_offset + block.file_pos as u64;
        let file_size = block.file_size as usize;
        let compressed = block.flags & (FLAG_COMPRESS | FLAG_IMPLODE) != 0;
        let implode_only = block.flags & FLAG_IMPLODE != 0 && block.flags & FLAG_COMPRESS == 0;

        let file_len = file.metadata()?.len();
        let avail = avail_from(file_len, file_pos);

        let mut open = ArchiveFile {
            file,
            name: name.to_owned(),
            base: file_pos,
            size: file_size,
            sector_size: idx.sector_size,
            offsets: None,
            implode_only,
            key,
            avail,
            pos: 0,
            sector: None,
        };
        if !compressed {
            // Stored: exactly `file_size` raw bytes, refused up front if the file cannot hold them.
            if capped(file_size, 1, avail) < file_size {
                return Err(Error::Corrupt(format!(
                    "{name}: stored size ({file_size}) larger than the archive"
                )));
            }
            return Ok(open);
        }

        // Sectored: `sector_count + 1` offsets at `file_pos`, measured from it, so the patch
        // archives' SECTOR_CRC table between them and the first sector is skipped for free.
        let sector_count = file_size.div_ceil(idx.sector_size);
        let otab_len = sector_count.checked_add(1).ok_or_else(|| {
            Error::Corrupt(format!("{name}: sector count overflow ({sector_count})"))
        })?;
        let otab_cap = capped(otab_len, 4, avail);
        if otab_cap < otab_len {
            return Err(Error::Corrupt(format!(
                "{name}: sector offset table ({otab_len} entries) larger than the archive"
            )));
        }
        open.file.seek(SeekFrom::Start(file_pos))?;
        let mut otab = vec![0u8; otab_cap * 4]; // == otab_len * 4, proven <= avail above
        open.file.read_exact(&mut otab)?;
        // An encrypted file's offset table is encrypted with `key - 1`.
        if let Some(key) = key {
            decrypt_bytes(&mut otab, key.wrapping_sub(1));
        }
        let offsets = otab
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| {
                c.u32_at(0)
                    .ok_or_else(|| Error::Corrupt(format!("{name}: corrupt sector offset table")))
            })
            .collect::<Result<Vec<u32>>>()?;
        open.offsets = Some(offsets);
        Ok(open)
    }
}

/// A file opened in an archive ([`Archive::open_file`]): `Read + Seek` over its uncompressed
/// bytes, holding the one decompressed sector under the read position.
pub struct ArchiveFile {
    file: File,
    name: String,
    /// The file's first byte in the archive file.
    base: u64,
    /// The uncompressed size, from the block table.
    size: usize,
    sector_size: usize,
    /// The sector offset table, relative to `base`; `None` for a stored file.
    offsets: Option<Vec<u32>>,
    implode_only: bool,
    /// An encrypted file's key: sector i decrypts with `key + i`, stored or compressed.
    key: Option<u32>,
    /// Bytes from `base` to the archive's end: the cap on every length read from a header.
    avail: usize,
    /// The read position in the uncompressed file.
    pos: u64,
    /// The decompressed sector under `pos`, by index.
    sector: Option<(usize, Vec<u8>)>,
}

impl ArchiveFile {
    /// The uncompressed size.
    pub fn size(&self) -> u64 {
        self.size as u64
    }

    /// The whole file from its start, sector by sector.
    fn read_all(mut self) -> Result<Vec<u8>> {
        if self.offsets.is_none() {
            self.file.seek(SeekFrom::Start(self.base))?;
            let mut out = vec![0u8; self.size]; // proven <= avail by `open_file`
            self.file.read_exact(&mut out)?;
            // A stored file is still encrypted sector by sector, sector i with `key + i`.
            if let Some(key) = self.key {
                for (i, sector) in out.chunks_mut(self.sector_size).enumerate() {
                    decrypt_bytes(sector, key.wrapping_add(i as u32));
                }
            }
            return Ok(out);
        }
        let sector_count = self.size.div_ceil(self.sector_size);
        let mut out = Vec::with_capacity(capped(self.size, 1, self.avail));
        for i in 0..sector_count {
            out.extend_from_slice(&self.decode_sector(i)?);
        }
        Ok(out)
    }

    /// Sector `i`, read, decrypted and decompressed to exactly its share of the file: the sector
    /// size, or the remainder for the last. One that inflates short is refused, since every later
    /// byte would shift into the gap. A stored file's sector is its raw span (reached only for an
    /// encrypted one, whose key steps per sector).
    fn decode_sector(&mut self, i: usize) -> Result<Vec<u8>> {
        let want = (self.size - i * self.sector_size).min(self.sector_size);
        if self.offsets.is_none() {
            self.file
                .seek(SeekFrom::Start(self.base + (i * self.sector_size) as u64))?;
            let mut raw = vec![0u8; want]; // within `size`, proven <= avail by `open_file`
            self.file.read_exact(&mut raw)?;
            if let Some(key) = self.key {
                decrypt_bytes(&mut raw, key.wrapping_add(i as u32));
            }
            return Ok(raw);
        }
        let name = &self.name;
        let offsets = self.offsets.as_deref().unwrap_or_default();
        let (Some(&start), Some(&end)) = (offsets.get(i), offsets.get(i + 1)) else {
            return Err(Error::Corrupt(format!(
                "{name}: sector {i} past the offset table"
            )));
        };
        let comp_len = u64::from(end)
            .checked_sub(u64::from(start))
            .ok_or_else(|| Error::Decompress(format!("{name}: sector {i} offsets reversed")))?
            as usize;
        let comp_cap = capped(comp_len, 1, self.avail);
        if comp_cap < comp_len {
            return Err(Error::Corrupt(format!(
                "{name}: sector {i} length ({comp_len}) larger than the archive"
            )));
        }
        if comp_len == 0 {
            return Err(Error::Corrupt(format!("{name}: sector {i} is empty")));
        }

        self.file
            .seek(SeekFrom::Start(self.base + u64::from(start)))?;
        let mut raw = vec![0u8; comp_len]; // proven <= avail above
        self.file.read_exact(&mut raw)?;
        // Sector i is encrypted with `key + i`, over its stored (compressed) bytes.
        if let Some(key) = self.key {
            decrypt_bytes(&mut raw, key.wrapping_add(i as u32));
        }

        let out = if comp_len >= want {
            // A sector that would not shrink is stored verbatim.
            raw.truncate(want);
            raw
        } else if self.implode_only {
            // IMPLODE sectors carry no method byte; 1.12.1 data has none.
            decompress(0x08, &raw, want, name)?
        } else {
            // COMPRESS: the leading byte is the codec mask, the rest the payload.
            let method = raw[0]; // non-empty: refused above
            decompress(method, &raw[1..], want, name)?
        };
        // `decompress` stops at `want`, so only a short sector is left to refuse.
        if out.len() != want {
            return Err(Error::Decompress(format!(
                "{name}: sector {i} inflates to {} bytes, not {want}",
                out.len()
            )));
        }
        Ok(out)
    }

    /// Stored bytes at `pos` into `buf`, straight from the archive file.
    fn read_stored(&mut self, buf: &mut [u8]) -> Result<usize> {
        let n = buf.len().min(self.size - self.pos as usize);
        self.file.seek(SeekFrom::Start(self.base + self.pos))?;
        self.file.read_exact(&mut buf[..n])?;
        Ok(n)
    }

    /// Sectored bytes at `pos` into `buf`, from the sector under it, decompressed on first touch.
    fn read_sectored(&mut self, buf: &mut [u8]) -> Result<usize> {
        let i = self.pos as usize / self.sector_size;
        if self.sector.as_ref().is_none_or(|(at, _)| *at != i) {
            let bytes = self.decode_sector(i)?;
            self.sector = Some((i, bytes));
        }
        let sector = self
            .sector
            .as_ref()
            .map(|(_, b)| b.as_slice())
            .unwrap_or_default();
        let off = self.pos as usize - i * self.sector_size;
        let n = buf.len().min(sector.len() - off);
        buf[..n].copy_from_slice(&sector[off..off + n]);
        Ok(n)
    }
}

impl Read for ArchiveFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.pos >= self.size as u64 {
            return Ok(0);
        }
        let n = if self.offsets.is_some() || self.key.is_some() {
            self.read_sectored(buf)
        } else {
            self.read_stored(buf)
        }
        .map_err(|e| match e {
            Error::Io(e) => e,
            other => io::Error::other(other),
        })?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for ArchiveFile {
    /// Moves the read position only; the sector under it is read by the next `read`. A position
    /// past the end reads as EOF.
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let pos = match to {
            SeekFrom::Start(p) => Some(p),
            SeekFrom::End(d) => (self.size as u64).checked_add_signed(d),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        }
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{}: seek before the start", self.name),
            )
        })?;
        self.pos = pos;
        Ok(pos)
    }
}

impl Index {
    /// Locate a file's block entry via the hash table (Blizzard's open-addressed probe).
    fn find(&self, name: &str) -> Option<BlockEntry> {
        let len = self.hash_table.len();
        if len == 0 {
            return None;
        }
        let mask = (len - 1) as u32;
        let start = (hash_string(name, hash_type::TABLE_OFFSET) & mask) as usize;
        let want_a = hash_string(name, hash_type::NAME_A);
        let want_b = hash_string(name, hash_type::NAME_B);
        for i in 0..len {
            let e = &self.hash_table[(start + i) % len];
            if e.block_index == HASH_EMPTY {
                return None; // never-used slot ends the chain
            }
            if e.block_index == HASH_DELETED {
                continue;
            }
            if e.name_a == want_a && e.name_b == want_b {
                return self.block_table.get(e.block_index as usize).copied();
            }
        }
        None
    }
}

/// The key Storm decrypts an encrypted file with: the name's last path component under the
/// `FILE_KEY` hash, offset by the block position and XOR'd with the size for a `FIX_KEY` entry.
fn file_key(name: &str, block: &BlockEntry) -> u32 {
    let base = name.rsplit(['\\', '/']).next().unwrap_or(name);
    let key = hash_string(base, hash_type::FILE_KEY);
    if block.flags & FLAG_FIX_KEY != 0 {
        key.wrapping_add(block.file_pos) ^ block.file_size
    } else {
        key
    }
}

/// The `MPQ\x1A` header's offset: the first 512-aligned match, or where a `MPQ\x1B` user-data
/// header points.
fn find_header(file: &mut File, file_len: u64) -> Result<u64> {
    let mut off = 0u64;
    while off + 4 <= file_len {
        file.seek(SeekFrom::Start(off))?;
        let mut sig = [0u8; 4];
        if file.read_exact(&mut sig).is_err() {
            break;
        }
        match u32::from_le_bytes(sig) {
            SIGNATURE => return Ok(off),
            USERDATA_SIGNATURE => {
                // User-data header: u32 sig, u32 user_data_size, u32 header_offset from here.
                let mut rest = [0u8; 8];
                file.read_exact(&mut rest)?;
                let header_offset = u32::from_le_bytes([rest[4], rest[5], rest[6], rest[7]]) as u64;
                return Ok(off + header_offset);
            }
            _ => {}
        }
        off += 512;
    }
    Err(Error::NotMpq)
}

fn read_hash_table(
    file: &mut File,
    pos: u64,
    count: usize,
    file_len: u64,
) -> Result<Vec<HashEntry>> {
    // A `count` the file cannot hold is refused before the loop indexes past a short `words`.
    let avail = avail_from(file_len, pos);
    let cap = capped(count, 16, avail);
    if cap < count {
        return Err(Error::Corrupt(format!(
            "hash table: header claims {count} entries, only room for {cap} in the archive"
        )));
    }
    let mut bytes = vec![0u8; cap * 16]; // == count * 16, proven <= avail above
    file.seek(SeekFrom::Start(pos))?;
    file.read_exact(&mut bytes)?;
    let mut words = to_u32s(&bytes);
    decrypt_block(&mut words, hash_string("(hash table)", hash_type::FILE_KEY));
    Ok((0..count)
        .map(|i| {
            let o = i * 4;
            HashEntry {
                name_a: words[o],
                name_b: words[o + 1],
                // words[o + 2] is locale and platform, unread: 1.12.1 data is locale-neutral.
                block_index: words[o + 3],
            }
        })
        .collect())
}

fn read_block_table(
    file: &mut File,
    pos: u64,
    count: usize,
    file_len: u64,
) -> Result<Vec<BlockEntry>> {
    // A `count` the file cannot hold is refused, as in `read_hash_table`.
    let avail = avail_from(file_len, pos);
    let cap = capped(count, 16, avail);
    if cap < count {
        return Err(Error::Corrupt(format!(
            "block table: header claims {count} entries, only room for {cap} in the archive"
        )));
    }
    let mut bytes = vec![0u8; cap * 16]; // == count * 16, proven <= avail above
    file.seek(SeekFrom::Start(pos))?;
    file.read_exact(&mut bytes)?;
    let mut words = to_u32s(&bytes);
    decrypt_block(
        &mut words,
        hash_string("(block table)", hash_type::FILE_KEY),
    );
    Ok((0..count)
        .map(|i| {
            let o = i * 4;
            BlockEntry {
                file_pos: words[o],
                // words[o + 1] is compressed_size, unread: the offset table sizes the sectors.
                file_size: words[o + 2],
                flags: words[o + 3],
            }
        })
        .collect())
}

/// Decompress one sector by its codec mask; 1.12.1 data uses zlib alone.
fn decompress(method: u8, data: &[u8], expected: usize, name: &str) -> Result<Vec<u8>> {
    match method {
        0x02 => {
            use flate2::read::ZlibDecoder;
            let mut out = Vec::with_capacity(expected);
            // Bounded: a sector inflating past its slot would wrap the caller's next `want`.
            ZlibDecoder::new(data)
                .take(expected as u64)
                .read_to_end(&mut out)
                .map_err(|e| Error::Decompress(format!("{name}: zlib: {e}")))?;
            Ok(out)
        }
        other => Err(Error::Unsupported(format!(
            "{name}: sector codec 0x{other:02X} not yet implemented"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique temp path (pid and a counter), so parallel tests never collide.
    fn temp_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "benilla_mpq_test_{}_{}_{}.mpq",
            std::process::id(),
            n,
            tag
        ))
    }

    /// A minimal 32-byte V1 MPQ header, unused fields zeroed.
    fn header(
        block_size_shift: u16,
        hash_table_pos: u32,
        block_table_pos: u32,
        hash_table_size: u32,
        block_table_size: u32,
    ) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0..4].copy_from_slice(&SIGNATURE.to_le_bytes());
        h[14..16].copy_from_slice(&block_size_shift.to_le_bytes());
        h[16..20].copy_from_slice(&hash_table_pos.to_le_bytes());
        h[20..24].copy_from_slice(&block_table_pos.to_le_bytes());
        h[24..28].copy_from_slice(&hash_table_size.to_le_bytes());
        h[28..32].copy_from_slice(&block_table_size.to_le_bytes());
        h
    }

    /// `Archive::open` over `bytes` in a temp file, removed whatever the outcome.
    fn open_temp(tag: &str, bytes: &[u8]) -> Result<Archive> {
        let path = temp_path(tag);
        std::fs::write(&path, bytes).expect("write temp archive");
        let result = Archive::open(&path);
        let _ = std::fs::remove_file(&path);
        result
    }

    #[test]
    fn open_rejects_hash_table_count_bigger_than_the_file() {
        // u32::MAX hash-table entries in a 40-byte file, ~68 GiB if the reservation were uncapped.
        let hdr = header(0, 32, 32, u32::MAX, 0);
        let mut bytes = hdr.to_vec();
        bytes.extend_from_slice(&[0u8; 8]); // pad past the header so `pos` (32) is in-bounds
        match open_temp("hash_overflow", &bytes) {
            Err(Error::Corrupt(_)) => {}
            Ok(_) => panic!("expected Error::Corrupt, got Ok(_)"),
            Err(other) => panic!("expected Error::Corrupt, got {other:?}"),
        }
    }

    #[test]
    fn open_rejects_block_table_count_bigger_than_the_file() {
        let hdr = header(0, 32, 32, 0, u32::MAX);
        let mut bytes = hdr.to_vec();
        bytes.extend_from_slice(&[0u8; 8]);
        match open_temp("block_overflow", &bytes) {
            Err(Error::Corrupt(_)) => {}
            Ok(_) => panic!("expected Error::Corrupt, got Ok(_)"),
            Err(other) => panic!("expected Error::Corrupt, got {other:?}"),
        }
    }

    #[test]
    fn open_accepts_a_valid_archive_with_empty_tables() {
        // Zero-entry tables pointing exactly at EOF: the boundary the cap must not reject.
        let hdr = header(0, 32, 32, 0, 0);
        let archive =
            open_temp("empty_tables", &hdr).expect("a well-formed empty-table archive must open");
        assert!(!archive.contains("anything"));
        assert_eq!(archive.file_size("anything"), None);
    }

    /// Like [`open_temp`], but the file stays for `read_file`, which reopens it by path.
    fn open_temp_kept(tag: &str, bytes: &[u8]) -> (Archive, PathBuf) {
        let path = temp_path(tag);
        std::fs::write(&path, bytes).expect("write temp archive");
        let archive = Archive::open(&path).expect("open temp archive");
        (archive, path)
    }

    /// A minimal V1 archive holding one stored entry, its tables encrypted with the real keys.
    fn archive_with_one_entry(name: &str, flags: u32, data: &[u8]) -> Vec<u8> {
        archive_with_one_block(name, flags, data, data.len() as u32)
    }

    /// [`archive_with_one_entry`] with `file_size` set apart from the data: a lying block table.
    fn archive_with_one_block(name: &str, flags: u32, data: &[u8], file_size: u32) -> Vec<u8> {
        use crypto::{encrypt_block, hash_type};
        const HASH_SLOTS: u32 = 4; // a power of two: the reader masks with len - 1
        let hash_pos = 32u32;
        let block_pos = hash_pos + HASH_SLOTS * 16;
        let data_pos = block_pos + 16; // one 16-byte block entry

        // Hash table (plaintext): every slot empty (0xFFFFFFFF) except our name's home slot.
        let mut hash = vec![0xFFFF_FFFFu32; HASH_SLOTS as usize * 4];
        let slot = (hash_string(name, hash_type::TABLE_OFFSET) & (HASH_SLOTS - 1)) as usize;
        hash[slot * 4] = hash_string(name, hash_type::NAME_A);
        hash[slot * 4 + 1] = hash_string(name, hash_type::NAME_B);
        hash[slot * 4 + 2] = 0; // locale | platform
        hash[slot * 4 + 3] = 0; // block index 0
        encrypt_block(&mut hash, hash_string("(hash table)", hash_type::FILE_KEY));

        // Block table (plaintext): one entry pointing at the trailing data.
        let mut block = vec![data_pos, data.len() as u32, file_size, flags];
        encrypt_block(
            &mut block,
            hash_string("(block table)", hash_type::FILE_KEY),
        );

        let mut bytes = header(0, hash_pos, block_pos, HASH_SLOTS, 1).to_vec();
        for w in hash.into_iter().chain(block) {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes.extend_from_slice(data);
        bytes
    }

    /// The control for the delete-marker test: the same builder's plain entry reads back.
    #[test]
    fn stored_entry_reads_its_bytes() {
        let (arc, path) = open_temp_kept(
            "stored",
            &archive_with_one_entry("Interface\\a.txt", FLAG_EXISTS, b"hello"),
        );
        assert!(arc.contains("Interface\\a.txt"));
        assert!(!arc.is_delete_marker("Interface\\a.txt"));
        assert_eq!(arc.read_file("Interface\\a.txt").unwrap(), b"hello");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_marker_is_not_a_readable_empty_file() {
        let name = "Interface\\FrameXML\\ClassTrainerFrame.xml";
        let (arc, path) = open_temp_kept(
            "delete_marker",
            &archive_with_one_entry(name, FLAG_EXISTS | FLAG_DELETE_MARKER, &[]),
        );
        assert!(arc.contains(name), "the hash entry exists");
        assert!(arc.is_delete_marker(name), "flagged as a tombstone");
        // The backing file exists, so NotFound here is the delete-marker guard.
        match arc.read_file(name) {
            Err(Error::NotFound(_)) => {}
            other => panic!("expected NotFound for a tombstone, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The nameless path agrees with the named one: what `entries` reports for a slot is the key
    /// `name_key` derives from that slot's name, and reading it by block index yields the same bytes
    /// `read_file` yields by name. This is the whole contract the pack builder rests on.
    #[test]
    fn entries_key_and_read_block_agree_with_the_named_path() {
        let name = "Interface\\a.txt";
        let (arc, path) = open_temp_kept(
            "entries",
            &archive_with_one_entry(name, FLAG_EXISTS, b"hello"),
        );

        let found: Vec<Entry> = arc.entries().collect();
        assert_eq!(found.len(), 1, "one live slot, the rest are HASH_EMPTY");
        assert_eq!(found[0].key, name_key(name));
        assert_eq!(found[0].file_size, 5);
        assert!(!found[0].is_delete_marker);
        assert_eq!(
            arc.read_block(found[0].block_index).unwrap(),
            arc.read_file(name).unwrap()
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A tombstone is *yielded, flagged* — not filtered out. A chain walker needs to see it to shadow
    /// the lower-priority copy; silently skipping it would fall through to the stale base version,
    /// which is exactly the mislead decision 0246 exists to prevent.
    #[test]
    fn entries_yields_tombstones_flagged_rather_than_hiding_them() {
        let name = "Interface\\FrameXML\\ClassTrainerFrame.xml";
        let (arc, path) = open_temp_kept(
            "entries_tomb",
            &archive_with_one_entry(name, FLAG_EXISTS | FLAG_DELETE_MARKER, &[]),
        );
        let found: Vec<Entry> = arc.entries().collect();
        assert_eq!(found.len(), 1);
        assert!(found[0].is_delete_marker);
        assert_eq!(found[0].key, name_key(name));
        let _ = std::fs::remove_file(&path);
    }

    /// `name_key` inherits MPQ's name equivalence rather than restating it: separator-agnostic and
    /// case-insensitive, because the two hashes it packs already fold `/`→`\` and uppercase.
    #[test]
    fn name_key_inherits_mpq_name_equivalence() {
        let canonical = name_key("Interface\\Icons\\Spell_Holy_Heal.blp");
        assert_eq!(canonical, name_key("interface/icons/spell_holy_heal.BLP"));
        assert_eq!(canonical, name_key("INTERFACE\\ICONS\\SPELL_HOLY_HEAL.BLP"));
        assert_ne!(canonical, name_key("Interface\\Icons\\Spell_Holy_Heal.tga"));
    }

    /// Over a **real** archive: for every name its `(listfile)` does name, reading by block index is
    /// byte-identical to reading by name. The synthetic tests above prove the wiring on one entry;
    /// this proves it against the sector layouts, codecs and table sizes an actual 1.12.1 archive
    /// presents. Skipped without an install (`$WOW_DATA`), like every other corpus-backed test here.
    #[test]
    fn read_block_matches_read_file_across_a_real_archive() {
        let Some(data) = std::env::var_os("WOW_DATA").filter(|v| !v.is_empty()) else {
            eprintln!("skipping: $WOW_DATA not set");
            return;
        };
        let mpq = PathBuf::from(data).join("patch.MPQ");
        if !mpq.is_file() {
            eprintln!("skipping: {} not found", mpq.display());
            return;
        }
        let arc = Archive::open(&mpq).expect("open patch.MPQ");

        // The block index a name resolves to isn't exposed by `find`, so go the other way: build the
        // key → block_index map from `entries`, then look each listfile name up in it. A name whose
        // key is absent would mean `entries` missed a live slot, which is the failure this guards.
        let by_key: std::collections::HashMap<u64, Entry> =
            arc.entries().map(|e| (e.key, e)).collect();

        let listfile = arc
            .read_file("(listfile)")
            .expect("patch.MPQ has a listfile");
        let mut checked = 0usize;
        for raw in String::from_utf8_lossy(&listfile).split([';', '\r', '\n']) {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            let entry = by_key
                .get(&name_key(name))
                .unwrap_or_else(|| panic!("{name} is in the listfile but not in entries()"));
            if entry.is_delete_marker {
                continue; // read_file refuses these by design (0246); nothing to compare
            }
            assert_eq!(
                arc.read_block(entry.block_index).ok(),
                arc.read_file(name).ok(),
                "{name} differs between the nameless and named read"
            );
            checked += 1;
        }
        // The archive's listfile is 26,350 entries; a near-empty run would mean the loop silently did
        // nothing and the assertions above never fired.
        assert!(checked > 20_000, "only checked {checked} names");

        // And the point of the whole seam: the hash table holds far more than the listfile names.
        assert!(
            by_key.len() > checked,
            "entries() ({}) should exceed the listfile ({checked})",
            by_key.len()
        );
    }

    #[test]
    fn avail_from_never_underflows_when_pos_exceeds_file_len() {
        assert_eq!(avail_from(10, 100), 0);
        assert_eq!(avail_from(100, 10), 90);
        assert_eq!(avail_from(0, 0), 0);
    }

    /// Two equal consecutive offsets name a zero-length sector, with no codec byte to read.
    #[test]
    fn an_empty_sector_is_refused_not_indexed() {
        // One 8-byte file, one sector: offsets [8, 8].
        let mut data = Vec::new();
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        let (arc, path) = open_temp_kept(
            "empty_sector",
            &archive_with_one_entry("a.bin", FLAG_EXISTS | FLAG_COMPRESS, &data),
        );
        match arc.read_file("a.bin") {
            Err(Error::Corrupt(_)) => {}
            other => panic!("expected Error::Corrupt for an empty sector, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_over_inflating_sector_cannot_run_the_output_past_file_size() {
        use flate2::{write::ZlibEncoder, Compression};
        use std::io::Write;
        // A COMPRESS sector as stored: the 0x02 (zlib) codec byte, then the stream.
        let zlib = |raw: &[u8]| {
            let mut e = ZlibEncoder::new(vec![0x02u8], Compression::default());
            e.write_all(raw).unwrap();
            e.finish().unwrap()
        };
        // 612 bytes: a full 512-byte sector (shift 0) and a 100-byte tail; sector 0 inflates to
        // 1000 bytes, twice its slot.
        let s0 = zlib(&[0u8; 1000]);
        let s1 = zlib(&[7u8; 100]);
        let otab_len = 12u32;
        let mut data = Vec::new();
        data.extend_from_slice(&otab_len.to_le_bytes());
        data.extend_from_slice(&(otab_len + s0.len() as u32).to_le_bytes());
        data.extend_from_slice(&(otab_len + (s0.len() + s1.len()) as u32).to_le_bytes());
        data.extend_from_slice(&s0);
        data.extend_from_slice(&s1);
        let (arc, path) = open_temp_kept(
            "over_inflate",
            &archive_with_one_block("a.bin", FLAG_EXISTS | FLAG_COMPRESS, &data, 612),
        );
        let out = arc
            .read_file("a.bin")
            .expect("a bounded inflate reads cleanly");
        assert_eq!(out.len(), 612);
        assert!(out[..512].iter().all(|&b| b == 0));
        assert!(out[512..].iter().all(|&b| b == 7));
        let _ = std::fs::remove_file(&path);
    }

    /// An encrypted, stored entry reads back by name and is refused by block index: the key derives
    /// from the name's last component (`hello.txt`), which the nameless read does not have.
    #[test]
    fn an_encrypted_stored_entry_reads_by_name_only() {
        use crypto::encrypt_bytes;
        let name = "Interface\\hello.txt";
        let plain = b"hello, encrypted world";
        let mut data = plain.to_vec();
        encrypt_bytes(&mut data, hash_string("hello.txt", hash_type::FILE_KEY));
        let (arc, path) = open_temp_kept(
            "encrypted_stored",
            &archive_with_one_entry(name, FLAG_EXISTS | FLAG_ENCRYPTED, &data),
        );
        assert_eq!(arc.read_file(name).unwrap(), plain);
        let entry = arc.entries().next().expect("one entry");
        assert!(arc.is_encrypted(entry.block_index));
        match arc.read_block(entry.block_index) {
            Err(Error::Unsupported(_)) => {}
            other => panic!("expected Unsupported for a nameless encrypted read, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The shape a Turtle patch archive's `(listfile)` takes: `COMPRESS | ENCRYPTED | FIX_KEY`, the
    /// offset table under `key - 1` and sector 0 under `key + 0`, with the key offset by the block
    /// position and XOR'd with the size.
    #[test]
    fn an_encrypted_fix_key_compressed_entry_reads_by_name() {
        use crypto::encrypt_bytes;
        use flate2::{write::ZlibEncoder, Compression};
        use std::io::Write;
        let name = "(listfile)";
        let plain =
            b"Interface\\FrameXML\\LFT\\LFT.lua\r\nDBFilesClient\\ChrRaces.dbc\r\n".repeat(4);
        let mut sector = vec![0x02u8];
        let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
        e.write_all(&plain).unwrap();
        sector.extend_from_slice(&e.finish().unwrap());

        // Laid out as `archive_with_one_block` lays it: 4 hash slots and one block entry after the
        // 32-byte header, so the data starts at 32 + 64 + 16.
        let data_pos = 32u32 + 4 * 16 + 16;
        let key =
            hash_string(name, hash_type::FILE_KEY).wrapping_add(data_pos) ^ plain.len() as u32;
        let mut otab = Vec::new();
        otab.extend_from_slice(&8u32.to_le_bytes());
        otab.extend_from_slice(&(8 + sector.len() as u32).to_le_bytes());
        encrypt_bytes(&mut otab, key.wrapping_sub(1));
        encrypt_bytes(&mut sector, key);
        let mut data = otab;
        data.extend_from_slice(&sector);

        let (arc, path) = open_temp_kept(
            "encrypted_fix_key",
            &archive_with_one_block(
                name,
                FLAG_EXISTS | FLAG_COMPRESS | FLAG_ENCRYPTED | FLAG_FIX_KEY,
                &data,
                plain.len() as u32,
            ),
        );
        assert_eq!(arc.read_file(name).unwrap(), plain);
        let _ = std::fs::remove_file(&path);
    }

    /// `Corrupt`, not the read's `Io`, pins that the guard fires before the buffer is sized.
    #[test]
    fn a_stored_size_larger_than_the_archive_is_refused_before_allocating() {
        let (arc, path) = open_temp_kept(
            "stored_overflow",
            &archive_with_one_block("a.txt", FLAG_EXISTS, b"hello", u32::MAX),
        );
        match arc.read_file("a.txt") {
            Err(Error::Corrupt(_)) => {}
            other => panic!("expected Error::Corrupt for a lying stored size, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A sectored COMPRESS file as stored, 512-byte sectors (shift 0): a zlib sector where it
    /// shrinks, the raw bytes where it would not, as the archive writer leaves them.
    fn sectored(plain: &[u8]) -> Vec<u8> {
        use flate2::{write::ZlibEncoder, Compression};
        use std::io::Write;
        let sectors: Vec<Vec<u8>> = plain
            .chunks(512)
            .map(|raw| {
                let mut e = ZlibEncoder::new(vec![0x02u8], Compression::default());
                e.write_all(raw).unwrap();
                let z = e.finish().unwrap();
                if z.len() < raw.len() {
                    z
                } else {
                    raw.to_vec()
                }
            })
            .collect();
        let mut at = (sectors.len() as u32 + 1) * 4;
        let mut out = at.to_le_bytes().to_vec();
        for s in &sectors {
            at += s.len() as u32;
            out.extend_from_slice(&at.to_le_bytes());
        }
        for s in &sectors {
            out.extend_from_slice(s);
        }
        out
    }

    /// Three sectors: a compressible run, an incompressible one stored verbatim, and a short tail.
    fn three_sector_plain() -> Vec<u8> {
        let mut rng = 0x1234_5678u32;
        (0..1300u32)
            .map(|i| match i {
                0..512 => (i / 7) as u8,
                512..1024 => {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    rng as u8
                }
                _ => (i % 3) as u8,
            })
            .collect()
    }

    #[test]
    fn an_open_file_reads_what_read_file_reads_from_any_position() {
        let plain = three_sector_plain();
        let stored = sectored(&plain);
        let (arc, path) = open_temp_kept(
            "open_file",
            &archive_with_one_block(
                "Sound\\bed.wav",
                FLAG_EXISTS | FLAG_COMPRESS,
                &stored,
                plain.len() as u32,
            ),
        );
        assert_eq!(arc.read_file("Sound\\bed.wav").unwrap(), plain);

        let mut f = arc.open_file("Sound\\bed.wav").unwrap();
        assert_eq!(f.size(), plain.len() as u64);
        let mut all = Vec::new();
        f.read_to_end(&mut all).unwrap();
        assert_eq!(all, plain, "a sequential read is the whole file");

        // Reads straddling each sector boundary, backwards and forwards.
        for (at, len) in [
            (1000, 100),
            (500, 30),
            (0, 1300),
            (1290, 64),
            (511, 2),
            (1023, 1),
        ] {
            f.seek(SeekFrom::Start(at)).unwrap();
            let mut got = vec![0u8; len];
            let n = {
                let mut n = 0;
                while n < len {
                    match f.read(&mut got[n..]).unwrap() {
                        0 => break,
                        k => n += k,
                    }
                }
                n
            };
            let end = (at as usize + len).min(plain.len());
            assert_eq!(&got[..n], &plain[at as usize..end], "read {len} at {at}");
        }

        assert_eq!(f.seek(SeekFrom::End(-4)).unwrap(), 1296);
        assert_eq!(f.seek(SeekFrom::Current(10)).unwrap(), 1306);
        assert_eq!(
            f.read(&mut [0u8; 8]).unwrap(),
            0,
            "past the end reads as EOF"
        );
        assert!(
            f.seek(SeekFrom::Current(-2000)).is_err(),
            "before the start"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_open_stored_file_reads_in_place() {
        let (arc, path) = open_temp_kept(
            "open_stored",
            &archive_with_one_entry("a.txt", FLAG_EXISTS, b"hello world"),
        );
        let mut f = arc.open_file("a.txt").unwrap();
        f.seek(SeekFrom::Start(6)).unwrap();
        let mut got = String::new();
        f.read_to_string(&mut got).unwrap();
        assert_eq!(got, "world");
        let _ = std::fs::remove_file(&path);
    }

    /// A sector that inflates short would shift every later byte into the gap: a whole read and
    /// a stream both refuse it.
    #[test]
    fn a_sector_that_inflates_short_is_refused() {
        let plain = three_sector_plain();
        // Sector 0 claims 512 bytes but its stream holds 500.
        let mut short = plain.clone();
        short.drain(500..512);
        let mut stored = sectored(&short[..500]);
        let tail = sectored(&plain[512..]);
        // Splice: sector 0 from the short build, sectors 1 and 2 from the full one.
        let (s0_start, s0_end) = (u32_at(&stored, 0), u32_at(&stored, 4));
        let s0 = stored[s0_start as usize..s0_end as usize].to_vec();
        let (t0, t1, t2) = (u32_at(&tail, 0), u32_at(&tail, 4), u32_at(&tail, 8));
        stored = Vec::new();
        let base = 16u32;
        for off in [
            base,
            base + s0.len() as u32,
            base + s0.len() as u32 + (t1 - t0),
            base + s0.len() as u32 + (t2 - t0),
        ] {
            stored.extend_from_slice(&off.to_le_bytes());
        }
        stored.extend_from_slice(&s0);
        stored.extend_from_slice(&tail[t0 as usize..]);
        let (arc, path) = open_temp_kept(
            "short_sector",
            &archive_with_one_block(
                "a.bin",
                FLAG_EXISTS | FLAG_COMPRESS,
                &stored,
                plain.len() as u32,
            ),
        );
        match arc.read_file("a.bin") {
            Err(Error::Decompress(e)) => {
                assert!(e.contains("sector 0 inflates to 500 bytes, not 512"), "{e}")
            }
            other => panic!("expected Error::Decompress for a short sector, got {other:?}"),
        }
        let mut f = arc.open_file("a.bin").unwrap();
        let err = f.read(&mut [0u8; 16]).unwrap_err().to_string();
        assert!(
            err.contains("sector 0 inflates to 500 bytes, not 512"),
            "{err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }
}
