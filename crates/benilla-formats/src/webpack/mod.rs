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
//! - [`probe`] — whether this host honours HTTP `Range`, which is a *measurement*, not a constant.
//!
//! ## Why both fetch shapes exist
//!
//! the host's edge slices ranges only for objects it has stored, and its default
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
    decode_pack_body, frame_bytes, frame_from_body, pack_body, Index, Located, PackMeta,
};

pub mod bootstrap;
pub mod probe;
pub mod resident;

pub use probe::Policy;

/// The container, open and ready to read: the parsed index plus the base URL its objects hang off.
///
/// Deliberately holds **no bytes of game data**. Blob bytes live either in the JS-side resident map
/// or nowhere; what is in wasm memory is the 1.6 MB index and, once `Chain::list` is first asked,
/// the listfile.
pub struct PackStore {
    /// `{origin}/data`, no trailing slash — [`crate::web::data_base`].
    base: String,
    index: Index,
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
        let bytes = bootstrap::index_bytes(&base)?;
        let index = Index::parse(bytes).map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            base,
            index,
            listfile: std::sync::OnceLock::new(),
        })
    }

    /// The index — for the boot progress bar's byte total and for diagnostics.
    pub fn index(&self) -> &Index {
        &self.index
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
        let Some((at, meta)) = self.locate(name) else {
            // Byte-identical wording to the native path: callers match on this text.
            return Err(anyhow!("file not in patch chain: {name}"));
        };
        if let Some(frame) = resident::slice(at.pack_id, at.frame_off, at.frame_len) {
            return frame_bytes(&frame).map_err(|e| anyhow!("{name}: {e}"));
        }
        let url = self.pack_url(at.pack_id);
        if probe::policy() == Policy::Range && meta.is_range_extractable() {
            let (start, end) = at.wire_range();
            let (status, body) = crate::web::fetch_sync_range(&url, Some((start, end)))
                .map_err(|e| fetch_err(name, at.pack_id, e))?;
            if status == 206 && body.len() as u64 == end - start + 1 {
                return frame_bytes(&body).map_err(|e| anyhow!("{name}: {e}"));
            }
            // A range answered with the whole object is a static host on a cache miss, not an error.
            // The body is already paid for, so it is absorbed rather than discarded, and the policy
            // steps down; the async lane promotes it back if a 206 ever shows up.
            probe::demote();
            return self.absorb(name, at, &meta, &body);
        }
        let (_, wire) =
            crate::web::fetch_sync_range(&url, None).map_err(|e| fetch_err(name, at.pack_id, e))?;
        self.absorb(name, at, &meta, &wire)
    }

    /// Read a blob without blocking — the Bevy `AssetReader` lane (`benilla-assets`), which is
    /// `async fn` all the way down and therefore never needs a synchronous XHR.
    ///
    /// This is also the lane that should pull SINGLETONs: a multi-megabyte object through
    /// [`crate::web::fetch_sync_range`]'s per-character decode is exactly what that decode is bad
    /// at, and here there is a real `ArrayBuffer` instead. It is also the only lane that can read
    /// Cache Storage, which is async-only — see [`resident::fetch_wire`].
    pub async fn read_async(&self, name: &str) -> Result<Vec<u8>> {
        let Some((at, meta)) = self.locate(name) else {
            return Err(anyhow!("file not in patch chain: {name}"));
        };
        if let Some(frame) = resident::slice(at.pack_id, at.frame_off, at.frame_len) {
            return frame_bytes(&frame).map_err(|e| anyhow!("{name}: {e}"));
        }
        if probe::policy() == Policy::Range && meta.is_range_extractable() {
            let (start, end) = at.wire_range();
            let (ranged, body) = resident::fetch_frame(at.pack_id, &self.pack_url(at.pack_id), start, end)
                .await
                .map_err(|e| anyhow!("{name}: fetching pack {}: {e}", at.pack_id))?;
            if ranged && body.len() as u64 == end - start + 1 {
                return frame_bytes(&body).map_err(|e| anyhow!("{name}: {e}"));
            }
            probe::demote();
            return self.absorb(name, at, &meta, &body);
        }
        let wire = resident::fetch_wire(at.pack_id, &self.pack_url(at.pack_id))
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
    fn absorb(&self, name: &str, at: Located, meta: &PackMeta, wire: &[u8]) -> Result<Vec<u8>> {
        let body = pack_body(wire, at.pack_id).map_err(|e| anyhow!("{name}: {e}"))?;
        let body = decode_pack_body(body, meta).map_err(|e| anyhow!("{name}: {e}"))?;
        resident::put(at.pack_id, &body);
        frame_from_body(&body, &at).map_err(|e| anyhow!("{name}: {e}"))
    }

    /// The container's name list, for [`crate::Chain::list`].
    ///
    /// Sizes are not part of it — the index is keyed by hash and carries no name-addressable length
    /// — and the one web caller, `load_macro_icons` enumerating `Interface\Icons\` for the macro
    /// picker, reads only names.
    pub fn listfile(&self) -> Result<Vec<crate::ChainEntry>> {
        let names = self
            .listfile
            .get_or_init(|| bootstrap::listfile_names(&self.base))
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
