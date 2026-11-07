//! fork: desktop — the native client's store when it has no install: the web build's **BENPACK
//! container** ([`benilla_pack`]), fetched from the origin it is hosted to and kept on disk.
//!
//! The browser keeps fetched packs in Cache Storage and invalidates them by the container's build
//! id (`web/sw.js`); this is the same corpus and the same law with a folder for a cache:
//!
//! ```text
//! <root>/index.bpi            the index the store answers from, replaced when the origin's changes
//! <root>/listfile.txt         beside it, when the container ships one
//! <root>/p/<hi>/<id>-<crc>.bpk  one fetched pack, named by its index row's CRC32
//! ```
//!
//! **Invalidation is by content, per pack.** At open the store asks the origin's version pointer
//! (`boot-packs.json`, served `no-cache`) for the build id; an index of another build is fetched
//! anew, and every cached pack whose `(id, crc)` the new index no longer names is deleted. A re-pack
//! that leaves a pack's bytes alone leaves its CRC alone, so that pack is kept; one that changes it
//! changes its file name, so a stale body can never be read for a fresh row. A download is kept only
//! once its length and CRC32 match the row, so a truncated transfer is never a cached pack.
//!
//! With the origin unreachable the store opens on the index it holds, and reads what it has cached;
//! with neither it fails at open, which is the one place that can say why.
//!
//! A store with no origin reads a container built on this machine (`benilla-pack build`'s `--out`)
//! in place, its packs at their served paths.
//!
//! **The background fill** ([`NetPackStore::next_unfilled`], [`NetPackStore::fill`], driven by
//! `crate::remote`'s fill thread) fetches the packs nothing has asked for yet, one at a time, and
//! only while no read or warm is downloading anything ([`foreground_busy`]) and no warm waits
//! ([`NetPackStore::warm_pending`]): a client idling at the login screen keeps filling its cache,
//! and the moment the game asks for something the fill stands aside.

mod fetch;

pub use fetch::{Fetch, Folder, Http};

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

use anyhow::{anyhow, bail, Context, Result};
use benilla_pack::{decode_pack_body, frame_bytes, pack_body, Index, Located, PackMeta};

use crate::{ChainEntry, TryRead};

/// The index file, at the container root.
const INDEX: &str = "index.bpi";
/// The container's name list, beside the index.
const LISTFILE: &str = "listfile.txt";
/// The origin's version pointer: `{"version":1,"build_id":"<16 hex>",…}`.
const POINTER: &str = "boot-packs.json";
/// Downloads in flight at once from the warm queue: the browser's six per host.
const WARM_THREADS: usize = 6;
/// Attempts at one object before its read fails.
const ATTEMPTS: usize = 3;
/// Decoded bodies of solid packs kept in memory. A stored pack is read a frame at a time off its
/// file, so only a solid one (one zstd stream, unreadable in parts) is held whole.
const SOLID_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Downloads in flight for a read or a warm, every store's together: the background fill takes the
/// line only while this is 0.
static FOREGROUND: AtomicUsize = AtomicUsize::new(0);

/// Whether a read or a warm is downloading a pack now, in any store.
pub fn foreground_busy() -> bool {
    FOREGROUND.load(Ordering::Acquire) > 0
}

/// Holds [`FOREGROUND`] up for one download.
struct Foreground;

impl Foreground {
    fn enter() -> Self {
        FOREGROUND.fetch_add(1, Ordering::AcqRel);
        Foreground
    }
}

impl Drop for Foreground {
    fn drop(&mut self) {
        FOREGROUND.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A container, open: its index in memory, its packs on disk or a fetch away.
pub struct NetPackStore {
    root: PathBuf,
    origin: Option<Box<dyn Fetch>>,
    index: Index,
    listfile: OnceLock<Option<Vec<String>>>,
    solid: Mutex<SolidCache>,
    /// Packs a thread is downloading now; a second asker waits on [`Self::landed`].
    loading: Mutex<HashSet<u32>>,
    landed: Condvar,
    /// Packs whose download failed this session: a `try_read` of them is a miss, not a wait.
    failed: Mutex<HashSet<u32>>,
    warm: WarmQueue,
    /// The background fill's place: every pack below it is cached or skipped.
    fill_cursor: AtomicU32,
    /// Packs whose background download failed: the fill passes them, a read still asks.
    fill_skipped: Mutex<HashSet<u32>>,
}

impl NetPackStore {
    /// Open the container cached at `root`, synced against `origin` when there is one.
    pub fn open(root: &Path, origin: Option<Box<dyn Fetch>>) -> Result<Self> {
        std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let index = match &origin {
            Some(origin) => sync_index(root, origin.as_ref())?,
            None => read_index(root)?.ok_or_else(|| anyhow!("no {INDEX} in {}", root.display()))?,
        };
        Ok(Self {
            root: root.to_path_buf(),
            origin,
            index,
            listfile: OnceLock::new(),
            solid: Mutex::new(SolidCache::default()),
            loading: Mutex::new(HashSet::new()),
            landed: Condvar::new(),
            failed: Mutex::new(HashSet::new()),
            warm: WarmQueue::default(),
            fill_cursor: AtomicU32::new(0),
            fill_skipped: Mutex::new(HashSet::new()),
        })
    }

    /// The index, for diagnostics and the byte total.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// The folder this container is cached in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether the container holds `name` (`/` or `\`, any case): a binary search, no I/O.
    pub fn contains(&self, name: &str) -> bool {
        self.index.locate(benilla_mpq::name_key(name)).is_some()
    }

    fn locate(&self, name: &str) -> Option<(Located, PackMeta)> {
        let at = self.index.locate(benilla_mpq::name_key(name))?;
        Some((at, self.index.pack(at.pack_id)?))
    }

    /// Read `name`, blocking on a download when its pack is not cached.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let Some((at, meta)) = self.locate(name) else {
            // The native chain's wording: callers match on it.
            bail!("file not in patch chain: {name}");
        };
        let path = self
            .ensure(at.pack_id, &meta)
            .map_err(|e| pack_err(name, at.pack_id, e))?;
        self.frame(&path, &at, &meta)
            .and_then(|frame| frame_bytes(&frame).map_err(|e| anyhow!("{e}")))
            .with_context(|| format!("reading {name} from pack {}", at.pack_id))
    }

    /// Read `name` without waiting on the network: a cached pack answers now, an uncached one is
    /// queued and answers `Pending` until it lands.
    pub fn try_read(self: &Arc<Self>, name: &str) -> TryRead {
        let Some((at, meta)) = self.locate(name) else {
            return TryRead::Missing;
        };
        if self.cached(at.pack_id, &meta).is_none() {
            if self.has_failed(at.pack_id) {
                return TryRead::Missing;
            }
            self.queue(at.pack_id, crate::web::warm_priority(name));
            return TryRead::Pending;
        }
        match self.read(name) {
            Ok(bytes) => TryRead::Ready(bytes),
            Err(_) => TryRead::Missing,
        }
    }

    /// Would [`Self::try_read`] answer now? Queues the pack when not. A name the container does
    /// not hold is ready: waiting changes nothing.
    pub fn is_ready(self: &Arc<Self>, name: &str) -> bool {
        let Some((at, meta)) = self.locate(name) else {
            return true;
        };
        if self.cached(at.pack_id, &meta).is_some() || self.has_failed(at.pack_id) {
            return true;
        }
        self.queue(at.pack_id, crate::web::warm_priority(name));
        false
    }

    /// Start fetching the pack `name` lives in, and return at once.
    pub fn prefetch(self: &Arc<Self>, name: &str) {
        if let Some((at, meta)) = self.locate(name) {
            if self.cached(at.pack_id, &meta).is_none() {
                self.queue(at.pack_id, crate::web::warm_priority(name));
            }
        }
    }

    /// Queue the packs the origin's version pointer lists for boot, world entry and the glue
    /// screens, so a first run's opening reads find them landing rather than start them.
    pub fn warm_boot_set(self: &Arc<Self>) {
        let Some(origin) = &self.origin else {
            return;
        };
        let Ok(pointer) = origin.get(POINTER) else {
            return;
        };
        for id in boot_pack_ids(&String::from_utf8_lossy(&pointer)) {
            if let Some(meta) = self.index.pack(id) {
                if self.cached(id, &meta).is_none() {
                    self.queue(id, crate::web::PRIORITY_NORMAL);
                }
            }
        }
    }

    /// The container's listfile, for `Chain::list`: names only (the index is keyed by hash).
    pub fn listfile(&self) -> Result<Vec<ChainEntry>> {
        let names = self
            .listfile
            .get_or_init(|| {
                let bytes = std::fs::read(self.root.join(LISTFILE)).ok()?;
                Some(
                    String::from_utf8_lossy(&bytes)
                        .lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty())
                        .map(String::from)
                        .collect(),
                )
            })
            .as_ref()
            .ok_or_else(|| anyhow!("this container ships no {LISTFILE}"))?;
        Ok(names
            .iter()
            .map(|name| ChainEntry {
                name: name.clone(),
                size: 0,
            })
            .collect())
    }

    /// The file a pack is kept in: named by its CRC when fetched, at its served path otherwise.
    fn pack_file(&self, pack_id: u32, meta: &PackMeta) -> PathBuf {
        if self.origin.is_some() {
            self.root.join(cached_name(pack_id, meta.crc32))
        } else {
            self.root.join(benilla_pack::pack_path(pack_id))
        }
    }

    /// The pack's file when it is on disk and the size its row says.
    fn cached(&self, pack_id: u32, meta: &PackMeta) -> Option<PathBuf> {
        let path = self.pack_file(pack_id, meta);
        let len = std::fs::metadata(&path).ok()?.len();
        (len == u64::from(meta.size_wire)).then_some(path)
    }

    fn has_failed(&self, pack_id: u32) -> bool {
        self.failed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(&pack_id)
    }

    /// Whether a warm waits in this store's queue.
    pub fn warm_pending(&self) -> bool {
        !self
            .warm
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .heap
            .is_empty()
    }

    /// The next pack the background fill should fetch, or `None` when every pack is cached (or
    /// failed): the first past the fill's place that is neither. A store with no origin has
    /// nothing to fill.
    pub fn next_unfilled(&self) -> Option<u32> {
        self.origin.as_ref()?;
        let skipped = |id: u32| {
            self.has_failed(id)
                || self
                    .fill_skipped
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .contains(&id)
        };
        let mut at = self.fill_cursor.load(Ordering::Relaxed);
        while at < self.index.pack_count() {
            if let Some(meta) = self.index.pack(at) {
                if self.cached(at, &meta).is_none() && !skipped(at) {
                    self.fill_cursor.store(at, Ordering::Relaxed);
                    return Some(at);
                }
            }
            at += 1;
        }
        self.fill_cursor.store(at, Ordering::Relaxed);
        None
    }

    /// Fetch one pack for the background fill. It does not count as foreground, and a failure
    /// skips the pack for the fill alone: a read of it later still downloads it.
    pub fn fill(&self, pack_id: u32) -> io::Result<()> {
        let meta = self
            .index
            .pack(pack_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no pack {pack_id}")))?;
        self.ensure_as(pack_id, &meta, false).map(|_| ())
    }

    /// The pack on disk, downloading it first when it is not; one download per pack however many
    /// threads ask.
    fn ensure(&self, pack_id: u32, meta: &PackMeta) -> io::Result<PathBuf> {
        self.ensure_as(pack_id, meta, true)
    }

    /// [`Self::ensure`], `foreground` saying whether the download is a read's or a warm's (which
    /// hold the background fill off) or the fill's own.
    fn ensure_as(&self, pack_id: u32, meta: &PackMeta, foreground: bool) -> io::Result<PathBuf> {
        if let Some(path) = self.cached(pack_id, meta) {
            return Ok(path);
        }
        let Some(origin) = &self.origin else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("pack {pack_id} is not in {}", self.root.display()),
            ));
        };
        {
            let mut loading = self.loading.lock().unwrap_or_else(PoisonError::into_inner);
            while loading.contains(&pack_id) {
                loading = self
                    .landed
                    .wait(loading)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if let Some(path) = self.cached(pack_id, meta) {
                return Ok(path);
            }
            loading.insert(pack_id);
        }
        let held = foreground.then(Foreground::enter);
        let got = download(
            origin.as_ref(),
            &self.pack_file(pack_id, meta),
            pack_id,
            meta,
        );
        drop(held);
        if got.is_err() {
            let skip = if foreground {
                &self.failed
            } else {
                &self.fill_skipped
            };
            skip.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(pack_id);
        }
        self.loading
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&pack_id);
        self.landed.notify_all();
        got.map(|()| self.pack_file(pack_id, meta))
    }

    /// One frame, its header included, out of a pack file: a byte range of the file for a stored
    /// pack, a slice of the decoded body for a solid one.
    fn frame(&self, path: &Path, at: &Located, meta: &PackMeta) -> Result<Vec<u8>> {
        if meta.is_range_extractable() {
            let mut file =
                std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            let (start, _) = at.wire_range();
            file.seek(SeekFrom::Start(start))?;
            let mut frame = vec![0; at.frame_len as usize];
            file.read_exact(&mut frame)
                .with_context(|| format!("reading {}", path.display()))?;
            return Ok(frame);
        }
        let body = self.solid_body(path, at.pack_id, meta)?;
        let start = at.frame_off as usize;
        start
            .checked_add(at.frame_len as usize)
            .and_then(|end| body.get(start..end))
            .map(<[u8]>::to_vec)
            .ok_or_else(|| anyhow!("frame {start}+{} past the body", at.frame_len))
    }

    fn solid_body(&self, path: &Path, pack_id: u32, meta: &PackMeta) -> Result<Arc<[u8]>> {
        if let Some(body) = self
            .solid
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(pack_id)
        {
            return Ok(body);
        }
        let wire = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let body = pack_body(&wire, pack_id).map_err(|e| anyhow!("{e}"))?;
        let body: Arc<[u8]> = decode_pack_body(body, meta)
            .map_err(|e| anyhow!("{e}"))?
            .into_owned()
            .into();
        self.solid
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .put(pack_id, Arc::clone(&body));
        Ok(body)
    }

    /// Hand `pack_id` to the warm threads, starting them on first use.
    fn queue(self: &Arc<Self>, pack_id: u32, priority: u8) {
        if self.origin.is_none() {
            return;
        }
        let mut q = self
            .warm
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !q.queued.insert(pack_id) {
            return;
        }
        q.seq += 1;
        let seq = q.seq;
        q.heap.push(std::cmp::Reverse((priority, seq, pack_id)));
        if !q.started {
            q.started = true;
            for n in 0..WARM_THREADS {
                let store = Arc::downgrade(self);
                let _ = std::thread::Builder::new()
                    .name(format!("pack-warm-{n}"))
                    .spawn(move || warm_loop(&store));
            }
        }
        drop(q);
        self.warm.ready.notify_one();
    }
}

/// One warm thread: pop the best-priority pack, download it, repeat; ends with its store.
fn warm_loop(store: &std::sync::Weak<NetPackStore>) {
    loop {
        let Some(this) = store.upgrade() else {
            return;
        };
        let pack_id = {
            let mut q = this
                .warm
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            loop {
                if let Some(std::cmp::Reverse((_, _, id))) = q.heap.pop() {
                    break id;
                }
                let (next, _) = this
                    .warm
                    .ready
                    .wait_timeout(q, std::time::Duration::from_secs(5))
                    .unwrap_or_else(PoisonError::into_inner);
                q = next;
                if q.heap.is_empty() && Arc::strong_count(&this) == 1 {
                    return;
                }
            }
        };
        if let Some(meta) = this.index.pack(pack_id) {
            if let Err(e) = this.ensure(pack_id, &meta) {
                eprintln!("benilla: pack {pack_id} could not be fetched: {e}");
            }
        }
        this.warm
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .queued
            .remove(&pack_id);
    }
}

#[derive(Default)]
struct WarmQueue {
    state: Mutex<WarmState>,
    ready: Condvar,
}

#[derive(Default)]
struct WarmState {
    /// `(priority, arrival, pack)`, least first: a lower priority number is wanted sooner.
    heap: BinaryHeap<std::cmp::Reverse<(u8, u64, u32)>>,
    queued: HashSet<u32>,
    seq: u64,
    started: bool,
}

/// Decoded solid-pack bodies, least recently used evicted past [`SOLID_CACHE_BYTES`].
#[derive(Default)]
struct SolidCache {
    bodies: HashMap<u32, Arc<[u8]>>,
    order: VecDeque<u32>,
    bytes: usize,
}

impl SolidCache {
    fn get(&mut self, pack_id: u32) -> Option<Arc<[u8]>> {
        let body = self.bodies.get(&pack_id).cloned()?;
        self.order.retain(|&id| id != pack_id);
        self.order.push_back(pack_id);
        Some(body)
    }

    fn put(&mut self, pack_id: u32, body: Arc<[u8]>) {
        if let Some(old) = self.bodies.insert(pack_id, Arc::clone(&body)) {
            self.bytes -= old.len();
            self.order.retain(|&id| id != pack_id);
        }
        self.bytes += body.len();
        self.order.push_back(pack_id);
        while self.bytes > SOLID_CACHE_BYTES && self.order.len() > 1 {
            if let Some(id) = self.order.pop_front() {
                if let Some(gone) = self.bodies.remove(&id) {
                    self.bytes -= gone.len();
                }
            }
        }
    }
}

/// `p/<hi>/<id>-<crc>.bpk`: the served path, the row's CRC32 in the name.
fn cached_name(pack_id: u32, crc32: u32) -> String {
    let served = benilla_pack::pack_path(pack_id);
    let stem = served.strip_suffix(".bpk").unwrap_or(&served);
    format!("{stem}-{crc32:08x}.bpk")
}

/// Fetch one pack and keep it at `dest` once its length, CRC32 and header check out; written
/// beside and renamed, so a crash mid-write leaves no file of the right name.
fn download(origin: &dyn Fetch, dest: &Path, pack_id: u32, meta: &PackMeta) -> io::Result<()> {
    let rel = benilla_pack::pack_path(pack_id);
    let mut last = None;
    for _ in 0..ATTEMPTS {
        match origin.get(&rel) {
            Ok(wire) => match check_pack(&wire, pack_id, meta) {
                Ok(()) => return write_atomic(dest, &wire),
                Err(e) => last = Some(e),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(e),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| io::Error::other("no attempt made")))
}

fn check_pack(wire: &[u8], pack_id: u32, meta: &PackMeta) -> io::Result<()> {
    if wire.len() != meta.size_wire as usize {
        return Err(io::Error::other(format!(
            "pack {pack_id}: {} bytes, the index says {}",
            wire.len(),
            meta.size_wire
        )));
    }
    let crc = crc32fast::hash(wire);
    if crc != meta.crc32 {
        return Err(io::Error::other(format!(
            "pack {pack_id}: CRC32 {crc:08x}, the index says {:08x}",
            meta.crc32
        )));
    }
    pack_body(wire, pack_id).map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = dest.as_os_str().to_owned();
    tmp.push(format!(".part{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, dest).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

fn read_index(root: &Path) -> Result<Option<Index>> {
    let Ok(bytes) = std::fs::read(root.join(INDEX)) else {
        return Ok(None);
    };
    Index::parse(bytes)
        .map(Some)
        .map_err(|e| anyhow!("{}: {e}", root.join(INDEX).display()))
}

/// The index to open on: the cached one when it is the origin's build, else the origin's, fetched
/// and kept with its listfile, the packs it no longer names deleted. Offline, the cached one.
fn sync_index(root: &Path, origin: &dyn Fetch) -> Result<Index> {
    // A cached index that will not parse is no index: the origin's replaces it.
    let cached = read_index(root).unwrap_or(None);
    let wanted = origin
        .get(POINTER)
        .ok()
        .and_then(|p| pointer_build_id(&String::from_utf8_lossy(&p)));
    let Some(wanted) = wanted else {
        // No pointer: offline, or an origin that serves none. Hold what is cached.
        return match cached {
            Some(index) => Ok(index),
            None => fetch_index(root, origin),
        };
    };
    if let Some(index) = cached {
        if index.build_id() == wanted {
            return Ok(index);
        }
    }
    fetch_index(root, origin)
}

/// The origin's index (and listfile), kept at `root`, and the cache swept of packs it does not
/// name.
fn fetch_index(root: &Path, origin: &dyn Fetch) -> Result<Index> {
    let bytes = origin
        .get(INDEX)
        .with_context(|| format!("fetching {INDEX} from {}", origin.describe()))?;
    let index = Index::parse(bytes.clone())
        .map_err(|e| anyhow!("{INDEX} from {}: {e}", origin.describe()))?;
    match origin.get(LISTFILE) {
        Ok(list) => write_atomic(&root.join(LISTFILE), &list)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let _ = std::fs::remove_file(root.join(LISTFILE));
        }
        Err(e) => {
            return Err(anyhow!(
                "fetching {LISTFILE} from {}: {e}",
                origin.describe()
            ))
        }
    }
    write_atomic(&root.join(INDEX), &bytes)?;
    // The loose files the site serves beside the index (`crate::remote::loose`) turn over with it.
    let _ = std::fs::remove_dir_all(root.join("loose"));
    sweep(root, &index);
    Ok(index)
}

/// Delete every cached pack file the index does not name by `(id, crc)`, and any partial write.
fn sweep(root: &Path, index: &Index) {
    let keep: HashSet<String> = (0..index.pack_count())
        .filter_map(|id| Some(cached_name(id, index.pack(id)?.crc32)))
        .collect();
    let Ok(fans) = std::fs::read_dir(root.join("p")) else {
        return;
    };
    for fan in fans.flatten() {
        let Ok(files) = std::fs::read_dir(fan.path()) else {
            continue;
        };
        for file in files.flatten() {
            let rel = format!(
                "p/{}/{}",
                fan.file_name().to_string_lossy(),
                file.file_name().to_string_lossy()
            );
            if !keep.contains(&rel) {
                let _ = std::fs::remove_file(file.path());
            }
        }
    }
}

/// `build_id` out of the version pointer: `"build_id":"<16 hex>"`.
fn pointer_build_id(json: &str) -> Option<u64> {
    let at = json.find("\"build_id\"")?;
    let rest = &json[at + "\"build_id\"".len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let hex = &rest[..rest.find('"')?];
    u64::from_str_radix(hex, 16).ok()
}

/// Every pack id the pointer's `packs` lists name (`[[id, bytes], …]`), first mention first.
fn boot_pack_ids(json: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut rest = json;
    while let Some(at) = rest.find("\"packs\"") {
        rest = &rest[at + "\"packs\"".len()..];
        let Some(open) = rest.find('[') else { break };
        let body = &rest[open + 1..];
        let mut depth = 1usize;
        let mut end = body.len();
        for (i, c) in body.char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        for pair in body[..end].split('[').skip(1) {
            let id = pair.split(',').next().and_then(|s| s.trim().parse().ok());
            if let Some(id) = id {
                if seen.insert(id) {
                    out.push(id);
                }
            }
        }
        rest = &body[end..];
    }
    out
}

fn pack_err(name: &str, pack_id: u32, e: io::Error) -> anyhow::Error {
    if e.kind() == io::ErrorKind::NotFound {
        anyhow!("file not in patch chain: {name}")
    } else {
        anyhow!("fetching {name} from pack {pack_id}: {e}")
    }
}

#[cfg(test)]
mod tests;
