//! The vanilla patch chain: a priority-ordered set of MPQ archives, read through `benilla-mpq`. A
//! read resolves a name to the highest-priority archive holding it, so a patch wins; base archives
//! carry no `(listfile)`, so resolution is by name hash.
//!
//! **On `wasm32`** there is no filesystem to mount archives from at all, so `Chain` reads a
//! **BENPACK container** — the same corpus repackaged into ~4,700 static objects plus one index
//! ([`crate::webpack`]) — off whatever object store serves the page. The public API
//! (`open`/`contains`/`find_file_archive`/`read`/`read_file`/`list`) is unchanged on both targets,
//! which is the whole point: the ~60 synchronous call sites and the async Bevy `AssetReader` above
//! both stayed exactly where they were while the thing underneath them was replaced. Only the two
//! `impl Chain` blocks below differ, and the web one adds one method the native target has no use
//! for ([`Chain::read_async`]).

use std::path::Path;

use anyhow::Result;

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashSet;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::{anyhow, bail, Context};
#[cfg(not(target_arch = "wasm32"))]
use benilla_mpq::Archive;

#[cfg(not(target_arch = "wasm32"))]
use crate::VANILLA_BASE_ORDER;

/// One entry from a chain listing: an internal path and its uncompressed size.
pub struct ChainEntry {
    pub name: String,
    pub size: u64,
}

/// What [`Chain::try_read`] found — a read that is allowed to answer "not yet".
///
/// The third arm is the whole point, and it exists only because the web target's store is a
/// network. On native every blob is one `pread` away and [`TryRead::Pending`] is unreachable; in a
/// browser a blob whose pack is not resident costs a round trip, and a round trip taken
/// synchronously on the thread that draws is a frozen tab.
///
/// A caller that takes this lane owes it two things: **ask again next frame**, and **do not cache
/// `Pending` as a miss**. In exchange the frame it asked on never blocks, and the pack is already
/// being warmed by the time it asks again.
pub enum TryRead {
    /// The bytes, at memcpy cost. On native this is every read that resolves.
    Ready(Vec<u8>),
    /// The blob exists but is not here yet; a warm is in flight. Ask again.
    Pending,
    /// Not readable, and no waiting changes that — the chain does not hold this name, or the read
    /// of it failed. The two are one arm on purpose: no caller of this lane does anything different
    /// with them (they draw nothing, or play nothing, and warn), and splitting them would put a
    /// `cfg` in every match, since the web target can only ever answer the first.
    Missing,
}

/// A priority-ordered patch chain of MPQ archives (`Send + Sync`; reads are `&self` and lock-free).
#[cfg(not(target_arch = "wasm32"))]
pub struct Chain {
    /// Ascending priority: later archives win.
    archives: Vec<Archive>,
}

/// The web build's `Chain`: a **BENPACK container** (`crate::webpack`) read out of a static object
/// store, behind the native API byte for byte.
///
/// There is nothing here but the store because there is nothing else to hold: the container's
/// index answers resolution from memory, and blob bytes live on the JS heap where 3.3 GiB of corpus
/// cannot crowd a 32-bit wasm address space. Every design note that used to live on this struct —
/// the `/data/__index` name list, the `HEAD` fallback — is gone with the per-name HTTP scheme it
/// belonged to.
///
/// Worth keeping the measurement that motivated it, because the container answers it outright: on
/// world entry (2026-08-31) the UI's texture probes asked `contains` **2,145 times** in one entry,
/// each a synchronous `HEAD` the browser would not serve from a `GET`-warmed cache — ~125 s of
/// frozen tab at 100 ms RTT. `contains` is now a binary search over bytes already in memory, so
/// that entire class of cost is not optimised but *absent*.
#[cfg(target_arch = "wasm32")]
pub struct Chain {
    store: crate::webpack::PackStore,
    /// The updated-character-model layer, when the player has it on: a **second container**, at
    /// `/data-hd`, consulted ahead of `store` for every question.
    ///
    /// `None` is the default and the cheap state — not "an empty overlay" but *no overlay*, which
    /// is the whole point of the option: nothing above is fetched, not even the 400 KB index that
    /// would say what the layer holds. Every method below reads as `hd.and_then(…).unwrap_or_else(base)`
    /// for that reason: overlay-first resolution is exactly MPQ's own priority rule
    /// (`patch-A.MPQ` mounts above every numbered patch), restated here because the wasm target has
    /// no archive stack to inherit it from.
    hd: Option<crate::webpack::PackStore>,
}

/// Whether `name` is a DBC table, by its suffix and without allocating: the one kind of name the
/// Turtle layer can carry a merge of (see [`crate::layers::TURTLE_HD_ALIAS_PREFIX`]).
#[cfg(target_arch = "wasm32")]
fn is_table(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() >= 4 && b[b.len() - 4..].eq_ignore_ascii_case(b".dbc")
}

/// `patch-?.MPQ` as the reference's `FindFirstFileW` glob matches it (template `0x82edbc`, wrapper
/// `0x42ad10`): `?` is exactly one character, any case, so `patch-10.MPQ` never mounts.
#[cfg(not(target_arch = "wasm32"))]
fn is_patch_glob_match(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let Some(mid) = lower
        .strip_prefix("patch-")
        .and_then(|rest| rest.strip_suffix(".mpq"))
    else {
        return false;
    };
    mid.chars().count() == 1
}

/// The reference's mount order over a `Data` listing, ascending priority (`0x403740`): the ten
/// [`VANILLA_BASE_ORDER`] archives, `patch.MPQ`, every `patch-?.MPQ` by case-folded name (the
/// reference sorts descending with `strnicmp` and walks backwards), then `speech2.MPQ`. Names
/// match case-insensitively and come back as found on disk.
#[cfg(not(target_arch = "wasm32"))]
fn mount_order(dir_names: &[String]) -> Vec<String> {
    let find = |want: &str| {
        dir_names
            .iter()
            .find(|n| n.eq_ignore_ascii_case(want))
            .cloned()
    };
    let mut order: Vec<String> = VANILLA_BASE_ORDER.iter().filter_map(|b| find(b)).collect();
    order.extend(find("patch.MPQ"));
    let mut patches: Vec<String> = dir_names
        .iter()
        .filter(|n| is_patch_glob_match(n))
        .cloned()
        .collect();
    patches.sort_by_key(|n| n.to_ascii_lowercase());
    order.extend(patches);
    order.extend(find("speech2.MPQ"));
    order
}

#[cfg(not(target_arch = "wasm32"))]
impl Chain {
    /// Open a `Data` directory's archives in [`mount_order`], or a single `.MPQ` file.
    ///
    /// Deviation: an archive that fails to open is an error, where the reference logs
    /// `"Failed to open archive"` and goes on, because a skipped corrupt archive surfaces only as
    /// missing files far downstream.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_layered(path, crate::layers::hd_character_models())
    }

    /// [`Chain::open`] with the optional layers named outright instead of read off the process-wide
    /// switch — for a tool, a test, or the packer, each of which wants *this* composite rather than
    /// whatever the client was last configured for.
    ///
    /// `hd_models` mounts (or leaves out) [`crate::HD_MODELS_ARCHIVE`]. Leaving it out is a filter
    /// over [`mount_order`]'s result and never a change to the law itself: the reference has no
    /// notion of an optional archive, so there is nothing here to be faithful to, and the one thing
    /// worth protecting is that the order the remaining archives mount in is still the order the
    /// reference would have mounted them in.
    ///
    /// Single-`.MPQ` opens ignore the flag: naming an archive is an instruction to mount it, and a
    /// caller that passes `patch-A.MPQ` by path means that archive (the packer's HD build does).
    pub fn open_layered(path: &Path, hd_models: bool) -> Result<Self> {
        let mut archives = Vec::new();
        if path.is_dir() {
            let mut names: Vec<String> = std::fs::read_dir(path)
                .with_context(|| format!("listing {}", path.display()))?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    // `path().is_file()` follows symlinks (`read_dir`'s file_type doesn't).
                    entry.path().is_file().then(|| entry.file_name())
                })
                .filter_map(|name| name.into_string().ok())
                .collect();
            // read_dir order is arbitrary; sort so case-variant ties resolve deterministically.
            names.sort();
            for name in mount_order(&names) {
                // The one filter over the law: an optional layer the player did not ask for is not
                // in the composite at all. Applied here rather than inside `mount_order` so that
                // function stays the reference's rule and nothing else (decision 1300).
                if !hd_models && name.eq_ignore_ascii_case(crate::HD_MODELS_ARCHIVE) {
                    continue;
                }
                let mpq = path.join(&name);
                archives.push(
                    Archive::open(&mpq).with_context(|| format!("opening {}", mpq.display()))?,
                );
            }
            if archives.is_empty() {
                bail!("no known vanilla MPQs found in {}", path.display());
            }
        } else {
            archives.push(
                Archive::open(path).with_context(|| format!("opening MPQ {}", path.display()))?,
            );
        }
        Ok(Self { archives })
    }

    /// The highest-priority archive with an entry for `name`, a delete marker included, as a
    /// tombstone shadows every lower copy: check [`Archive::is_delete_marker`] for a readable file.
    fn resolve(&self, name: &str) -> Option<&Archive> {
        self.archives.iter().rev().find(|a| a.contains(name))
    }

    /// Whether `name` (`/` or `\`, any case) is a readable file, not a delete marker.
    pub fn contains(&self, name: &str) -> bool {
        self.resolve(name)
            .is_some_and(|a| !a.is_delete_marker(name))
    }

    /// The path of the archive `name` resolves to, for debugging and extraction.
    pub fn find_file_archive(&self, name: &str) -> Option<&Path> {
        self.resolve(name).map(|a| a.path())
    }

    /// Read a file by internal path (`/` or `\`) from its winning archive.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let archive = self
            .resolve(name)
            .ok_or_else(|| anyhow!("file not in patch chain: {name}"))?;
        // A tombstone deletes the path from the composite: not found, never a stale lower copy.
        if archive.is_delete_marker(name) {
            bail!(
                "file deleted from patch chain: {name} (tombstoned by {})",
                archive.path().display()
            );
        }
        archive
            .read_file(name)
            .with_context(|| format!("reading {name} from {}", archive.path().display()))
    }

    /// `&mut` alias of [`Chain::read`] for call sites that thread a `&mut Chain`.
    pub fn read_file(&mut self, name: &str) -> Result<Vec<u8>> {
        self.read(name)
    }

    /// The mounted archives in **ascending priority** — the order [`mount_order`] put them in, so a
    /// caller walking them back-to-front reproduces `resolve`'s "later wins" without restating the
    /// law.
    ///
    /// For the pack builder (`benilla-packer`), which cannot use [`Chain::list`]: it needs every
    /// entry, including the ones no listfile names, and it needs the block index behind each so it
    /// can read them by [`Archive::read_block`]. Both come from [`Archive::entries`], per archive,
    /// which is what this exposes. Nothing on the read path uses it.
    pub fn archives(&self) -> &[Archive] {
        &self.archives
    }

    /// [`Self::read`] in the shape the web target needs — see [`TryRead`]. Here it is a plain read:
    /// a local archive has no "not yet", so this only ever answers `Ready` or `Missing` (a failed
    /// read folding into the latter), and the call sites that use it behave on native exactly as
    /// they did before the lane existed.
    pub fn try_read(&self, name: &str) -> TryRead {
        match self.read(name) {
            Ok(bytes) => TryRead::Ready(bytes),
            Err(_) => TryRead::Missing,
        }
    }

    /// Would [`Self::try_read`] answer without deferring? **Always true on native**, where every
    /// read resolves on the spot — see the web twin for what it is for.
    #[inline]
    pub fn ready(&self, _name: &str) -> bool {
        true
    }

    /// Say that `name` will be wanted soon. **Nothing to do on native** — a file on disk is not
    /// warmed by being mentioned — so this is an empty call the optimiser deletes, and the shared
    /// call sites can say what they mean without a `cfg`.
    #[inline]
    pub fn prefetch(&self, _name: &str) {}

    /// The chain's named files with sizes, for development and extraction; a file in no listfile
    /// (most of `texture.MPQ`) is readable by name but not listed. Unions every archive's
    /// `(listfile)`, as each names only its own files; sizes come from the winning archive.
    pub fn list(&self) -> Result<Vec<ChainEntry>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for archive in &self.archives {
            let Ok(listfile) = archive.read_file("(listfile)") else {
                continue;
            };
            for raw in String::from_utf8_lossy(&listfile).split([';', '\r', '\n']) {
                let name = raw.trim();
                // Dedupe the way MPQ hashing compares names: any case, `/` and `\` alike.
                if name.is_empty() || !seen.insert(name.replace('/', "\\").to_ascii_lowercase()) {
                    continue;
                }
                if let Some(a) = self.resolve(name) {
                    // A tombstoned path is not a file in the composite.
                    if a.is_delete_marker(name) {
                        continue;
                    }
                    out.push(ChainEntry {
                        name: name.to_string(),
                        size: a.file_size(name).unwrap_or(0) as u64,
                    });
                }
            }
        }
        Ok(out)
    }
}

#[cfg(target_arch = "wasm32")]
impl Chain {
    /// Open the container the page is served alongside — `crate::web::data_base()`, i.e. `/data` on
    /// this origin. `path` is accepted only to keep the signature identical to the native target's
    /// (call sites pass `wow_data()`, which on wasm is always `/data`); it names nothing real here,
    /// where the corpus is a few thousand HTTP objects rather than a directory of archives.
    ///
    /// **This can now fail, and that is the correction.** The old web `Chain` never failed at open
    /// because every question was deferred to a per-name fetch. The container has no such
    /// deferral: without `index.bpi` nothing resolves at all, so a missing or malformed index is
    /// reported once, here, instead of as thousands of identical read failures with no cause.
    pub fn open(_path: &Path) -> Result<Self> {
        Self::open_layered(_path, crate::layers::hd_character_models())
    }

    /// [`Chain::open`] with the optional layers named outright — the web twin of the native
    /// signature, so a caller that decides the layer itself writes one line for both targets.
    ///
    /// A layer that **fails to open** is reported and dropped, not propagated, and that is the one
    /// place the two targets deliberately differ. Natively an unopenable archive is fatal (decision
    /// 1300: a corrupt `dbc.MPQ` must not become mystery missing-file errors); here the only way to
    /// fail is that `/data-hd/index.bpi` is not hosted, and the right answer to "this origin
    /// ships no HD layer" is the game in its original art, not a refusal to boot.
    pub fn open_layered(_path: &Path, hd_models: bool) -> Result<Self> {
        let store = crate::webpack::PackStore::open(crate::web::data_base())?;
        let hd = hd_models
            .then(|| crate::webpack::PackStore::open_hd(crate::web::hd_data_base()))
            .and_then(|opened| match opened {
                Ok(store) => Some(store),
                Err(e) => {
                    crate::web::console_warn(&format!(
                        "updated character models are on but this origin serves no {} container \
                         ({e:#}) — running on the original art",
                        crate::layers::HD_DATA_SEGMENT
                    ));
                    None
                }
            });
        Ok(Self { store, hd })
    }

    /// The container that should answer for `name` — [`crate::layers::pick_layer`] over the layers
    /// mounted now: the HD overlay when it holds the name, else the Turtle layer when it is mounted
    /// and holds it, else the base container; a table the Turtle layer merged with the HD layer's
    /// answers under its alias while both are mounted. **The whole of the priority law on this
    /// target**, in one place so no method below restates it.
    ///
    /// `contains` on a `PackStore` is a binary search over an index already in memory, so asking the
    /// overlays first costs ~17 probes each and no I/O. On a 1.12.1 realm the Turtle layer costs one
    /// relaxed atomic load here ([`crate::layers::turtle`]), and only a `.dbc` read with both layers
    /// mounted builds an alias name.
    ///
    /// `turtle` is the caller's handle on the mounted layer, so the answer can borrow it.
    fn pick<'a>(
        &'a self,
        name: &str,
        turtle: Option<&'a crate::webpack::PackStore>,
    ) -> (&'a crate::webpack::PackStore, Option<String>) {
        let alias = match (&self.hd, turtle) {
            (Some(_), Some(t)) if is_table(name) => {
                let alias = crate::layers::turtle_hd_alias(name);
                t.contains(&alias).then_some(alias)
            }
            _ => None,
        };
        let layer = crate::layers::pick_layer(
            self.hd.as_ref().map(|hd| hd.contains(name)),
            turtle.map(|t| t.contains(name)),
            alias.is_some(),
        );
        // The layer halves of the boot trace, so a `?boottrace=1` capture can say whether a layer is
        // being read at all and which names reach it (`web::trace_hd`); no-ops unless the page armed
        // the arrays.
        match (layer, turtle, &self.hd) {
            (crate::layers::Layer::TurtleHdAlias, Some(t), _) => {
                crate::web::trace_turtle(name);
                (t, alias)
            }
            (crate::layers::Layer::Hd, _, Some(hd)) => {
                crate::web::trace_hd(name);
                (hd, None)
            }
            (crate::layers::Layer::Turtle, Some(t), _) => {
                crate::web::trace_turtle(name);
                (t, None)
            }
            _ => (&self.store, None),
        }
    }

    /// Whether the Turtle WoW 1.18.1 layer is mounted now — see [`crate::layers::turtle_mounted`].
    pub fn turtle_mounted(&self) -> bool {
        crate::layers::turtle_mounted()
    }

    /// Whether the updated-character-model layer is mounted on this chain — for the boot report and
    /// the diagnostics line, which should say which corpus a session is actually running on.
    pub fn hd_character_models(&self) -> bool {
        self.hd.is_some()
    }

    /// Whether the container holds `name` (accepts `/` or `\`; case-insensitive — the equivalence is
    /// MPQ's own, inherited through `benilla_mpq::name_key` rather than restated).
    ///
    /// No I/O, no allocation, no network: a binary search over the index's sorted key array.
    pub fn contains(&self, name: &str) -> bool {
        self.hd.as_ref().is_some_and(|hd| hd.contains(name))
            || crate::layers::turtle().is_some_and(|t| t.contains(name))
            || self.store.contains(name)
    }

    /// No archive *files* exist on the web target — a blob's home is a pack id, not a path — so
    /// there is nothing more specific than `contains` to report.
    pub fn find_file_archive(&self, _name: &str) -> Option<&Path> {
        None
    }

    /// Read a file by internal path (accepts `/` or `\`), blocking — see
    /// `crate::web::fetch_sync_range` for why this is synchronous and
    /// [`crate::webpack::PackStore::read_sync`] for what it costs. The error wording on a missing
    /// file matches the native path's (`"file not in patch chain: {name}"`) so a caller that
    /// matches on that text — there are some — behaves the same on both targets.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        crate::web::trace(name); // boot-manifest capture; no-op unless the page armed it
        let turtle = crate::layers::turtle();
        let (store, alias) = self.pick(name, turtle.as_deref());
        store.read_sync(alias.as_deref().unwrap_or(name))
    }

    /// Read `name` **without ever blocking the frame** — the lane the UI and the sound engine take.
    ///
    /// See [`TryRead`] for the contract and [`crate::webpack::PackStore::try_read_sync`] for what
    /// each arm costs. A `Pending` has already started warming the pack, so a caller that simply
    /// asks again next frame converges without a scheduler of its own.
    pub fn try_read(&self, name: &str) -> TryRead {
        // Boot-manifest capture; no-op unless the page armed one. Traced on every ask, retries
        // included — the array is deduplicated by hand and first push is still first need, so a
        // name that took three frames to arrive appears exactly where it was first wanted.
        crate::web::trace(name);
        let turtle = crate::layers::turtle();
        let (store, alias) = self.pick(name, turtle.as_deref());
        store.try_read_sync(alias.as_deref().unwrap_or(name))
    }

    /// Would [`Self::try_read`] answer without deferring? Starts the warm if not — see
    /// [`crate::webpack::PackStore::is_ready`], including why a name the container does not hold is
    /// reported ready.
    ///
    /// For a caller whose work reads several blobs and is only correct once it can read all of them:
    /// it asks first, does nothing while the answer is no, and the packs warm in the meantime.
    pub fn ready(&self, name: &str) -> bool {
        let turtle = crate::layers::turtle();
        let (store, alias) = self.pick(name, turtle.as_deref());
        store.is_ready(alias.as_deref().unwrap_or(name))
    }

    /// Warm the pack `name` lives in, and return immediately — see
    /// [`crate::webpack::PackStore::prefetch`]. Idempotent and cheap enough to call every frame.
    pub fn prefetch(&self, name: &str) {
        let turtle = crate::layers::turtle();
        let (store, alias) = self.pick(name, turtle.as_deref());
        store.prefetch(alias.as_deref().unwrap_or(name));
    }

    /// [`Chain::read`] for a caller that can await — the Bevy `AssetReader`
    /// (`benilla-assets`), which is `async fn` on every target and so has no reason to block a
    /// browser's main thread. **Web-only**: there is no native twin, because on native the read is
    /// a file read and there is nothing to await.
    ///
    /// Prefer it wherever it is reachable. It is the lane that can read Cache Storage (async-only),
    /// and the lane that pulls multi-megabyte SINGLETON packs through a real `ArrayBuffer` instead
    /// of the sync path's per-character body decode.
    pub async fn read_async(&self, name: &str) -> Result<Vec<u8>> {
        crate::web::trace(name);
        let turtle = crate::layers::turtle();
        let (store, alias) = self.pick(name, turtle.as_deref());
        store.read_async(alias.as_deref().unwrap_or(name)).await
    }

    /// `&mut` alias of [`Chain::read`] — see the native impl for why this exists.
    pub fn read_file(&mut self, name: &str) -> Result<Vec<u8>> {
        self.read(name)
    }

    /// List the container's named files, from the `listfile.txt` it ships beside the index.
    ///
    /// Sizes are not available by name (the index is keyed by hash and a listing would need a
    /// lookup per entry to report one), so every entry reports `size: 0` — unchanged from what the
    /// web target reported before, and the one web caller, `load_macro_icons`, reads only names.
    pub fn list(&self) -> Result<Vec<ChainEntry>> {
        let mut out = self.store.listfile()?;
        // The overlay's names are APPENDED, not merged: `Chain::list` is a name enumeration, its one
        // web caller (`load_macro_icons`) walks it for a prefix, and a name held by both layers is
        // the same name either way — which layer would *answer* a read of it is `layer_for`'s
        // business, not this list's. An HD container shipping no listfile is simply no extra names,
        // on the same terms as the base container shipping without one.
        let turtle = crate::layers::turtle();
        let overlays = self.hd.iter().chain(turtle.as_deref());
        let mut seen: std::collections::HashSet<String> = out
            .iter()
            .map(|e| e.name.replace('/', "\\").to_ascii_lowercase())
            .collect();
        for layer in overlays {
            let Ok(extra) = layer.listfile() else {
                continue;
            };
            out.extend(extra.into_iter().filter(|e| {
                let key = e.name.replace('/', "\\").to_ascii_lowercase();
                // A merged table's alias is how the pair answers `<name>`, not a name of its own.
                !key.starts_with(&crate::layers::TURTLE_HD_ALIAS_PREFIX.to_ascii_lowercase())
                    && seen.insert(key)
            }));
        }
        Ok(out)
    }

    /// The store behind this chain — for the boot progress UI, which wants the container's byte
    /// total and build id, and for diagnostics.
    pub fn store(&self) -> &crate::webpack::PackStore {
        &self.store
    }
}

// Native only: exercises `is_patch_glob_match`/`mount_order`, which don't exist on the web target.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn owned(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn patch_glob_matches_exactly_one_character_case_insensitively() {
        assert!(is_patch_glob_match("patch-2.MPQ"));
        assert!(is_patch_glob_match("patch-3.MPQ"));
        assert!(is_patch_glob_match("PATCH-A.mpq"));
        assert!(!is_patch_glob_match("patch-.MPQ"));
        assert!(!is_patch_glob_match("patch-10.MPQ"));
        assert!(!is_patch_glob_match("patch-33.MPQ"));
        assert!(!is_patch_glob_match("patch.MPQ"));
        assert!(!is_patch_glob_match("patch-2.MPQ.bak"));
        assert!(!is_patch_glob_match("mypatch-2.MPQ"));
    }

    #[test]
    fn mount_order_is_the_carved_law() {
        // base.MPQ is telemetry-only in the reference and never mounts.
        let dir = owned(&[
            "patch-2.MPQ",
            "backup.MPQ",
            "model.MPQ",
            "base.MPQ",
            "dbc.MPQ",
            "patch.MPQ",
            "eula.html",
            "patch-3.MPQ",
            "speech2.MPQ",
            "texture.MPQ",
        ]);
        assert_eq!(
            mount_order(&dir),
            owned(&[
                "dbc.MPQ",
                "texture.MPQ",
                "model.MPQ",
                "patch.MPQ",
                "patch-2.MPQ",
                "patch-3.MPQ",
                "speech2.MPQ",
            ])
        );
    }

    #[test]
    fn patch_sort_is_ascending_and_case_folded() {
        let dir = owned(&["patch-B.MPQ", "patch-3.MPQ", "patch-a.MPQ", "patch-2.MPQ"]);
        assert_eq!(
            mount_order(&dir),
            owned(&["patch-2.MPQ", "patch-3.MPQ", "patch-a.MPQ", "patch-B.MPQ"])
        );
    }

    #[test]
    fn base_archives_are_found_case_insensitively() {
        let dir = owned(&["DBC.mpq", "Model.MPQ", "PATCH.mpq"]);
        assert_eq!(
            mount_order(&dir),
            owned(&["DBC.mpq", "Model.MPQ", "PATCH.mpq"])
        );
    }
}
