//! The web target's game-data store: a **BENPACK container** read out of a dumb object store.
//!
//! Before this, the browser build fetched one game file per HTTP request from a companion process
//! (`wenilla-host --data`) holding the real 5.4 GB MPQ chain on disk. That coupled the page to a
//! machine with a WoW install and to an origin that had to stay up for the whole session, and it
//! cost a round trip per file — 2,145 blocking `HEAD`s in one world entry just to answer
//! `contains`. The container replaces it with ~4,700 static objects and one 1.6 MB index, which
//! A static asset host serves for free and which `wenilla-host --packs` serves off a laptop.
//!
//! ## What this module is responsible for
//!
//! [`crate::Chain`]'s wasm body is a thin delegation to [`PackStore`]; everything interesting is
//! here. Three problems, one per submodule:
//!
//! - [`bootstrap`] — getting `index.bpi` (and, lazily, `listfile.txt`) into wasm memory. The page
//!   has already fetched them by the time the app starts, so the normal path is a copy, not I/O.
//! - [`resident`] — the pack map, which lives on the **JS heap** (`web/packs.js`), not in wasm
//!   linear memory. 3.3 GiB of corpus must never touch a 32-bit heap, and the JS GC can evict.
//!   It is also where the fetch/decode Web Worker is reached ([`resident::warm`]): a pack it warms
//!   arrives decoded, so the read that wanted it is a memcpy rather than a download and a zstd
//!   stream on the thread that draws.
//! - [`probe`] — whether this host honours HTTP `Range`, which is a *measurement*, not a constant.
//!
//! ## Why both fetch shapes exist
//!
//! A CDN edge slices ranges only for objects it has stored, and its default
//! `max-age=0, must-revalidate` means it never stores one — measured: `Range: bytes=100-199`
//! against a live site answers **200 with the full body**. So the whole-object path is not a
//! fallback, it is the path that runs today, and the ranged path is the optimisation that switches
//! on if `immutable` (or a different host later) turns 206s on. Neither is chosen at build time:
//! the policy is probed, demoted the instant a range request is answered with a 200, and promoted
//! again when a 206 shows up. One hosted tree is correct either way.
//!
//! A whole-object miss is not the loss it sounds like. Packs are **locality groups**, so fetching
//! one warms the few hundred reads that come next, and at the container's measured p50 of 655 KiB
//! a miss is smaller than the largest single file the old per-name path could ask for.

use anyhow::{anyhow, Result};
use benilla_pack::{
    decode_pack_body, frame_bytes, pack_body, Codec, Index, Located, PackMeta, FRAME_HEADER_LEN,
};

use crate::{StoredBlob, TryRead};

pub mod bootstrap;
pub mod probe;
pub mod resident;

pub use probe::Policy;

// The warm priority a name is asked for at lives in `crate::web` — string math over a name and a
// tile grid, with no browser dependency, and this module compiles only on `wasm32`, so keeping it
// here would put its law where no native test can reach it (the same reasoning that keeps
// `encode_name` there). The terrain streamer's `set_terrain_focus` is over there for the same
// reason: the writer is a Bevy system that runs on every target.
pub use crate::web::{
    warm_priority, PRIORITY_GEOMETRY, PRIORITY_NORMAL, PRIORITY_TERRAIN, PRIORITY_TERRAIN_FOCUS,
    PRIORITY_TERRAIN_NEAR,
};

/// The container, open and ready to read: the parsed index plus the base URL its objects hang off.
///
/// Deliberately holds **no bytes of game data**. Blob bytes live either in the JS-side resident map
/// or nowhere; what is in wasm memory is the 1.6 MB index and, once `Chain::list` is first asked,
/// the listfile.
pub struct PackStore {
    /// `{origin}/data`, no trailing slash — [`crate::web::data_base`].
    base: String,
    index: Index,
    /// What this container's pack ids are offset by **as resident-map keys** — 0 for the base
    /// corpus, [`crate::layers::HD_PACK_KEY_BASE`] for the HD layer.
    ///
    /// Two containers are open at once now, both numbering their packs from 0, and the resident map
    /// is one `Map` on the JS heap keyed by a plain number. Without an offset the layer's pack 7 and
    /// the base corpus's pack 7 are the same entry, and a read of one is served the other's body —
    /// which surfaces as a zstd error if you are lucky and as a frame decoded out of the wrong bytes
    /// if you are not. Keys are offset; **URLs are not** ([`Self::pack_url`] uses the container's own
    /// id, because `p/<hi>/<id>.bpk` is the path the packer wrote).
    key_base: u32,
    /// Which page properties hold this container's prefetched index and listfile — `index` /
    /// `listfile` for the base corpus, `hdIndex` / `hdListfile` for the layer.
    ///
    /// Two containers hand themselves over through the one `window.__wenilla_packs` object, so the
    /// property names have to differ or the second overwrites the first. See [`bootstrap`].
    index_property: &'static str,
    listfile_property: &'static str,
    /// `Chain::list`'s answer, built on first ask. `None` inside = the container shipped no
    /// listfile, which is not fatal: only `load_macro_icons` enumerates, and it degrades to an
    /// empty icon picker rather than a failed boot.
    listfile: std::sync::OnceLock<Option<Vec<String>>>,
}

impl PackStore {
    /// Read `index.bpi` and open the container at `base`.
    ///
    /// Fails when there is no index or it cannot be parsed, and that is the right severity: without
    /// it *nothing* is readable — there is no per-name route left to fall back to — so a clear
    /// error at open beats thousands of identical failures at every read site.
    pub fn open(base: String) -> Result<Self> {
        Self::open_at(base, 0, "index", "listfile")
    }

    /// The **updated-character-model layer**'s container, at `/data-hd`.
    ///
    /// Same format, same code, two differences that both come from it being the *second* container
    /// open in one page: its resident keys are offset ([`Self::key_base`]) and it reads its
    /// prefetched index and listfile from their own page properties, so `packs.js` can hand over two
    /// containers without either overwriting the other.
    ///
    /// Unlike [`Self::open`], a failure here is recoverable by the caller — see
    /// [`crate::Chain::open_layered`], which reports it and runs on the base corpus.
    pub fn open_hd(base: String) -> Result<Self> {
        Self::open_at(
            base,
            crate::layers::HD_PACK_KEY_BASE,
            "hdIndex",
            "hdListfile",
        )
    }

    /// The **Turtle WoW 1.18.1 layer**'s container, at `/data-turtle`: the HD layer's shape, in its
    /// own key range ([`crate::layers::TURTLE_PACK_KEY_BASE`]) and page slots. Opened when a 1.18.1
    /// realm is entered, off the index the page fetched for it
    /// ([`crate::web::mount_turtle_layer`]).
    pub fn open_turtle(base: String) -> Result<Self> {
        Self::open_at(
            base,
            crate::layers::TURTLE_PACK_KEY_BASE,
            "turtleIndex",
            "turtleListfile",
        )
    }

    /// The **reforged art layer**'s container, at `/data-reforged`: the Turtle layer's shape, in its
    /// own key range ([`crate::layers::REFORGED_PACK_KEY_BASE`]) and page slots. Opened with the
    /// Turtle layer when the setting has it on ([`crate::web::mount_reforged_layer`]).
    pub fn open_reforged(base: String) -> Result<Self> {
        Self::open_at(
            base,
            crate::layers::REFORGED_PACK_KEY_BASE,
            "reforgedIndex",
            "reforgedListfile",
        )
    }

    fn open_at(
        base: String,
        key_base: u32,
        index_property: &'static str,
        listfile_property: &'static str,
    ) -> Result<Self> {
        let bytes = bootstrap::index_bytes(&base, index_property)?;
        let index = Index::parse(bytes).map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            base,
            index,
            key_base,
            index_property,
            listfile_property,
            listfile: std::sync::OnceLock::new(),
        })
    }

    /// This container's resident-map key for one of its pack ids — see [`Self::key_base`].
    fn key(&self, pack_id: u32) -> u32 {
        self.key_base + pack_id
    }

    /// The index — for the boot progress bar's byte total and for diagnostics.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Where this container came from and which page slot handed it over — the diagnostics line's
    /// half of "which corpus is this session actually running on".
    pub fn origin(&self) -> (&str, &'static str) {
        (&self.base, self.index_property)
    }

    /// Where a name's blob lives, plus its pack's metadata. `None` when the container has no such
    /// blob, which is the same answer the native chain gives for a file no archive holds.
    ///
    /// Pure computation: a binary search over the index's sorted key array, ~17 probes into bytes
    /// that are already in memory. This is what makes `Chain::contains` free on the web target.
    fn locate(&self, name: &str) -> Option<(Located, PackMeta)> {
        let at = self.index.locate(benilla_mpq::name_key(name))?;
        let meta = self.index.pack(at.pack_id)?;
        Some((at, meta))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.locate(benilla_mpq::name_key(name)).is_some()
    }

    /// The URL one pack is served at.
    fn pack_url(&self, pack_id: u32) -> String {
        format!("{}/{}", self.base, benilla_pack::pack_path(pack_id))
    }

    /// Read a blob, blocking. The ~60 synchronous `Chain::read` call sites land here.
    ///
    /// Three outcomes in cost order: the pack is resident (a memcpy plus a frame decode), the host
    /// honours ranges (one round trip for exactly this blob's frame), or the whole pack comes down
    /// and is kept — which costs one round trip now and makes its whole locality group free.
    pub fn read_sync(&self, name: &str) -> Result<Vec<u8>> {
        self.read_sync_stored(name)?
            .into_bytes()
            .map_err(|e| anyhow!("{name}: {e}"))
    }

    /// [`Self::read_sync`] without the frame decode (`stored`).
    pub fn read_sync_stored(&self, name: &str) -> Result<StoredBlob> {
        let Some((at, meta)) = self.locate(name) else {
            // Byte-identical wording to the native path: callers match on this text.
            return Err(anyhow!("file not in patch chain: {name}"));
        };
        if let Some(frame) = resident::slice(self.key(at.pack_id), at.frame_off, at.frame_len) {
            return stored(name, frame);
        }
        // Metered: from here on this read is a synchronous XHR on the thread that draws.
        let t0 = js_sys::Date::now();
        let out = self.fetch_sync(name, at, &meta);
        sync_meter::record(name, (js_sys::Date::now() - t0) as f32);
        out
    }

    /// [`Self::read_sync`]'s blocking half: the pack is not resident.
    fn fetch_sync(&self, name: &str, at: Located, meta: &PackMeta) -> Result<StoredBlob> {
        let url = self.pack_url(at.pack_id);
        if probe::policy() == Policy::Range && meta.is_range_extractable() {
            let (start, end) = at.wire_range();
            let (status, body) = crate::web::fetch_sync_range(&url, Some((start, end)))
                .map_err(|e| fetch_err(name, at.pack_id, e))?;
            if status == 206 && body.len() as u64 == end - start + 1 {
                return stored(name, body);
            }
            // A range answered with the whole object is a static host on a cache miss, not an error.
            // The body is already paid for, so it is absorbed rather than discarded, and the policy
            // steps down; the async lane promotes it back if a 206 ever shows up.
            probe::demote();
            return self.absorb(name, at, meta, &body);
        }
        let (_, wire) =
            crate::web::fetch_sync_range(&url, None).map_err(|e| fetch_err(name, at.pack_id, e))?;
        self.absorb(name, at, meta, &wire)
    }

    /// Read a blob **only if it costs nothing** — the frame thread's lane.
    ///
    /// [`Self::read_sync`] is correct and sometimes catastrophic: a resident miss there is a
    /// synchronous XHR, and a synchronous XHR on the thread that draws is the whole tab frozen for
    /// a round trip. That is the right trade at boot, where there is no frame to protect and the
    /// alternative is a client that cannot start. It is the wrong trade for a UI icon or a footstep
    /// — work whose deadline is "some frame soon", not "this instruction".
    ///
    /// So this is the same read with the blocking branch replaced by an answer: `Pending`, plus a
    /// fire-and-forget warm of the pack ([`resident::prefetch`]) so the retry a frame or two later
    /// finds it resident and returns [`TryRead::Ready`]. The caller's contract is to ask again and
    /// **not** to cache `Pending` as a miss.
    ///
    /// `Missing` is final and identical to `read_sync`'s missing-file error: the index answered, and
    /// no amount of waiting puts a blob in the container that was never packed into it.
    pub fn try_read_sync(&self, name: &str) -> TryRead {
        let Some((at, _)) = self.locate(name) else {
            return TryRead::Missing;
        };
        match resident::slice(self.key(at.pack_id), at.frame_off, at.frame_len) {
            Some(frame) => match frame_bytes(&frame) {
                Ok(bytes) => TryRead::Ready(bytes),
                // A frame that will not decode is not going to decode next frame either, and
                // reporting it as `Pending` would spin the caller forever on a warm that already
                // succeeded. Corrupt reads as absent, which is what every caller here handles.
                Err(_) => TryRead::Missing,
            },
            None if resident::can_warm(self.key(at.pack_id)) => {
                resident::prefetch(
                    self.key(at.pack_id),
                    &self.pack_url(at.pack_id),
                    warm_priority(name),
                );
                TryRead::Pending
            }
            // Nothing here can warm this pack — no decode worker on this page, or this one has
            // already failed. Answering `Pending` would ask the caller to retry forever, so the
            // read degrades to the blocking path instead: a stall on a page that was already
            // going to stall, rather than art that never arrives.
            None => match self.read_sync(name) {
                Ok(bytes) => TryRead::Ready(bytes),
                Err(_) => TryRead::Missing,
            },
        }
    }

    /// Would [`Self::try_read_sync`] answer without deferring? Starts the warm if not.
    ///
    /// [`Self::prefetch`] without the blind spot and [`Self::try_read_sync`] without the bytes — for
    /// a caller that must decide **before** it starts, because what it is about to do reads several
    /// blobs and is only correct if it can finish. The character-skin composite is that caller: it
    /// blends a base skin with up to fifteen overlays into one atlas that is then cached for the
    /// session, so an overlay arriving late is not "a late overlay", it is a tabard that is missing
    /// for as long as the player wears it.
    ///
    /// **A blob the container does not hold counts as ready**, because "not there" is a final answer
    /// and every caller of this treats a missing overlay as one — waiting for it would be waiting
    /// forever. So is a blob whose pack cannot be warmed: that read will block, which is the old
    /// behaviour and not something a probe can improve.
    pub fn is_ready(&self, name: &str) -> bool {
        let Some((at, _)) = self.locate(name) else {
            return true;
        };
        if resident::is_resident(self.key(at.pack_id)) || !resident::can_warm(self.key(at.pack_id))
        {
            return true;
        }
        resident::prefetch(
            self.key(at.pack_id),
            &self.pack_url(at.pack_id),
            warm_priority(name),
        );
        false
    }

    /// Say that `name` will be wanted soon, and return immediately.
    ///
    /// The point of a locality group is that fetching one blob's pack warms the few hundred reads
    /// that come next — but only if something asks before the frame that needs them. This is that
    /// ask: no bytes come back, nothing is awaited, and a pack already resident or already in
    /// flight costs a lookup.
    pub fn prefetch(&self, name: &str) {
        let Some((at, _)) = self.locate(name) else {
            return;
        };
        if resident::is_resident(self.key(at.pack_id)) {
            return;
        }
        resident::prefetch(
            self.key(at.pack_id),
            &self.pack_url(at.pack_id),
            warm_priority(name),
        );
    }

    /// Read a blob without blocking — the Bevy `AssetReader` lane (`benilla-assets`), which is
    /// `async fn` all the way down and therefore never needs a synchronous XHR.
    ///
    /// This is also the lane that should pull SINGLETONs: a multi-megabyte object through
    /// [`crate::web::fetch_sync_range`]'s per-character decode is exactly what that decode is bad
    /// at, and here there is a real `ArrayBuffer` instead. It is also the only lane that can read
    /// Cache Storage, which is async-only — see [`resident::fetch_wire`] — and the only one that
    /// can wait on the decode worker ([`resident::warm`]), for the same structural reason.
    ///
    /// The frame comes back undecoded (`stored`): the caller decompresses it, off the thread
    /// that draws where it can.
    pub async fn read_async_stored(&self, name: &str) -> Result<StoredBlob> {
        let Some((at, meta)) = self.locate(name) else {
            return Err(anyhow!("file not in patch chain: {name}"));
        };
        if let Some(frame) = resident::slice(self.key(at.pack_id), at.frame_off, at.frame_len) {
            return stored(name, frame);
        }
        // The **Web Worker** first (`web/packworker.js`): it fetches the object and undoes the
        // pack's zstd stream on its own thread, and hands the decoded body straight to the resident
        // map. That is strictly better than either branch below — both of which would do the same
        // work here, on the thread that draws — so it is asked first and the rest is what happens
        // when there is no worker to ask.
        if resident::warm(
            self.key(at.pack_id),
            &self.pack_url(at.pack_id),
            warm_priority(name),
        )
        .await
        {
            if let Some(frame) = resident::slice(self.key(at.pack_id), at.frame_off, at.frame_len) {
                return stored(name, frame);
            }
        }
        if probe::policy() == Policy::Range && meta.is_range_extractable() {
            let (start, end) = at.wire_range();
            let (ranged, body) =
                resident::fetch_frame(self.key(at.pack_id), &self.pack_url(at.pack_id), start, end)
                    .await
                    .map_err(|e| anyhow!("{name}: fetching pack {}: {e}", at.pack_id))?;
            if ranged && body.len() as u64 == end - start + 1 {
                return stored(name, body);
            }
            probe::demote();
            return self.absorb(name, at, &meta, &body);
        }
        let wire = resident::fetch_wire(self.key(at.pack_id), &self.pack_url(at.pack_id))
            .await
            .map_err(|e| anyhow!("{name}: fetching pack {}: {e}", at.pack_id))?;
        self.absorb(name, at, &meta, &wire)
    }

    /// Take a whole `.bpk` as served, hand its decoded body to the resident map, and return the one
    /// frame the caller actually asked for.
    ///
    /// The pack-id check inside [`pack_body`] is load-bearing here rather than paranoid: a static
    /// host misconfigured to answer a missing object with `index.html` and a 200 would otherwise
    /// feed an HTML page to a zstd decoder and corrupt silently.
    fn absorb(&self, name: &str, at: Located, meta: &PackMeta, wire: &[u8]) -> Result<StoredBlob> {
        let body = pack_body(wire, at.pack_id).map_err(|e| anyhow!("{name}: {e}"))?;
        let body = decode_pack_body(body, meta).map_err(|e| anyhow!("{name}: {e}"))?;
        resident::put(self.key(at.pack_id), &body);
        let start = at.frame_off as usize;
        let frame = start
            .checked_add(at.frame_len as usize)
            .and_then(|end| body.get(start..end))
            .ok_or_else(|| {
                anyhow!(
                    "{name}: frame {start}+{} past a {}-byte body",
                    at.frame_len,
                    body.len()
                )
            })?;
        stored(name, frame.to_vec())
    }

    /// The container's name list, for [`crate::Chain::list`].
    ///
    /// Sizes are not part of it — the index is keyed by hash and carries no name-addressable length
    /// — and the one web caller, `load_macro_icons` enumerating `Interface\Icons\` for the macro
    /// picker, reads only names.
    pub fn listfile(&self) -> Result<Vec<crate::ChainEntry>> {
        let names = self
            .listfile
            .get_or_init(|| bootstrap::listfile_names(&self.base, self.listfile_property))
            .as_ref()
            .ok_or_else(|| anyhow!("this container ships no listfile.txt"))?;
        Ok(names
            .iter()
            .map(|name| crate::ChainEntry {
                name: name.clone(),
                size: 0,
            })
            .collect())
    }
}

/// One frame as a read hands it over: a stored frame is its payload, a copy away, and a compressed
/// one goes back undecoded, since its decompression is CPU the thread that draws need not pay
/// (`Chain::read_async_stored`).
fn stored(name: &str, frame: Vec<u8>) -> Result<StoredBlob> {
    // The header's last byte is the frame's codec.
    if frame.get(FRAME_HEADER_LEN - 1) == Some(&(Codec::Store as u8)) {
        frame_bytes(&frame)
            .map(StoredBlob::Plain)
            .map_err(|e| anyhow!("{name}: {e}"))
    } else {
        Ok(StoredBlob::Frame(frame))
    }
}

/// **What blocking reads cost the frame** — the meter `benilla_app::webdiag` reports.
///
/// A [`PackStore::read_sync`] whose pack is not resident is a synchronous XHR plus a pack decode on
/// the thread that draws, and no frame meter can say that is why a frame went long. This names the
/// file as well as the cost, so a freeze points at its caller.
pub mod sync_meter {
    use std::cell::RefCell;

    thread_local! {
        /// `name -> (reads, total ms)` since the last [`take`].
        static SEEN: RefCell<Vec<(String, u32, f32)>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn record(name: &str, ms: f32) {
        SEEN.with(|s| {
            let mut s = s.borrow_mut();
            match s.iter_mut().find(|(n, ..)| n == name) {
                Some((_, calls, total)) => {
                    *calls += 1;
                    *total += ms;
                }
                None => s.push((name.to_string(), 1, ms)),
            }
        });
    }

    /// Drain the window's blocking reads, costliest first: `(name, reads, ms)`.
    pub fn take() -> Vec<(String, u32, f32)> {
        SEEN.with(|s| {
            let mut v = std::mem::take(&mut *s.borrow_mut());
            v.sort_by(|a, b| b.2.total_cmp(&a.2));
            v
        })
    }
}

/// A fetch failure, worded so a 404 reads the same as the native chain's missing-file error —
/// several callers match on that text, and a missing *pack* is a missing *file* from where they
/// stand.
fn fetch_err(name: &str, pack_id: u32, e: std::io::Error) -> anyhow::Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        anyhow!("file not in patch chain: {name}")
    } else {
        anyhow!("fetching {name} from pack {pack_id}: {e}")
    }
}
