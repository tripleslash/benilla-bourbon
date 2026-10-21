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

use crate::Locale;
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

/// A blob as a read hands it over: its bytes, or (web) its pack frame still compressed, so the
/// decompression runs wherever the caller takes [`Self::into_bytes`], a worker included: it is
/// plain CPU over owned bytes.
pub enum StoredBlob {
    /// The bytes themselves.
    Plain(Vec<u8>),
    /// fork: web — one BENPACK frame, its header included, whose payload is compressed.
    #[cfg(target_arch = "wasm32")]
    Frame(Vec<u8>),
}

impl StoredBlob {
    /// Whether [`Self::into_bytes`] decompresses, rather than hand the bytes back.
    pub fn is_encoded(&self) -> bool {
        match self {
            Self::Plain(_) => false,
            #[cfg(target_arch = "wasm32")]
            Self::Frame(_) => true,
        }
    }

    /// The blob's bytes, decompressing a frame.
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        match self {
            Self::Plain(bytes) => Ok(bytes),
            #[cfg(target_arch = "wasm32")]
            Self::Frame(frame) => {
                benilla_pack::frame_bytes(&frame).map_err(|e| anyhow::anyhow!("{e}"))
            }
        }
    }
}

/// A priority-ordered patch chain of MPQ archives (`Send + Sync`; reads are `&self` and lock-free).
#[cfg(not(target_arch = "wasm32"))]
pub struct Chain {
    /// Ascending priority: later archives win.
    archives: Vec<Archive>,
    /// The locale the chain's `Wow.ini` names, read once at open as start-up reads it once.
    locale: Locale,
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
    /// The locale the container's `Wow.ini` names, read once at open as start-up reads it once.
    locale: Locale,
}

/// The live layers one read resolves against — the updated character models, the Turtle WoW
/// 1.18.1 layer and the reforged art over it ([`crate::layers`]), each mounted or not for the whole
/// page — taken once per read, so the read keeps its handles past the locks. With a layer off its
/// share is one relaxed load.
#[cfg(target_arch = "wasm32")]
struct Mounts {
    hd: Option<std::sync::Arc<crate::webpack::PackStore>>,
    turtle: Option<std::sync::Arc<crate::webpack::PackStore>>,
    reforged: Option<std::sync::Arc<crate::webpack::PackStore>>,
}

#[cfg(target_arch = "wasm32")]
impl Mounts {
    fn now() -> Self {
        Self {
            hd: crate::layers::hd(),
            turtle: crate::layers::turtle(),
            reforged: crate::layers::reforged(),
        }
    }
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
        let mut chain = Self {
            archives,
            locale: Locale::default(),
        };
        chain.locale = crate::wow_ini::client_locale(&chain);
        Ok(chain)
    }

    /// The client locale, from the chain's `Wow.ini` (`0x402350`): its index picks the column of
    /// every localized DBC string.
    pub fn locale(&self) -> Locale {
        self.locale
    }

    /// The same archives read as another locale, for a test of the column a loader reads.
    #[cfg(test)]
    pub(crate) fn with_locale(mut self, locale: Locale) -> Self {
        self.locale = locale;
        self
    }

    /// A chain of exactly these archives, in ascending priority (a later one wins), whatever
    /// directories they sit in — for the pack builder, whose layers stack archives from more than
    /// one install (`patch-A.mpq` under `patch-T.mpq`). The mount law has no say here: naming the
    /// archives is the order.
    pub fn open_archives(paths: &[std::path::PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            bail!("no archives to mount");
        }
        let archives = paths
            .iter()
            .map(|p| Archive::open(p).with_context(|| format!("opening MPQ {}", p.display())))
            .collect::<Result<Vec<_>>>()?;
        let mut chain = Self {
            archives,
            locale: Locale::default(),
        };
        chain.locale = crate::wow_ini::client_locale(&chain);
        Ok(chain)
    }

    /// The highest-priority archive with an entry for `name`, a delete marker included, as a
    /// tombstone shadows every lower copy: check [`Archive::is_delete_marker`] for a readable file.
    fn resolve(&self, name: &str) -> Option<&Archive> {
        self.archives.iter().rev().find(|a| a.contains(name))
    }

    /// Whether `name` (`/` or `\`, any case) is a readable file, not a delete marker.
    pub fn contains(&self, name: &str) -> bool {
        match self.resolve(name) {
            Some(a) => !a.is_delete_marker(name),
            None => crate::bundled_addons::contains(name),
        }
    }

    /// The path of the archive `name` resolves to, for debugging and extraction.
    pub fn find_file_archive(&self, name: &str) -> Option<&Path> {
        self.resolve(name).map(|a| a.path())
    }

    /// Read a file by internal path (`/` or `\`) from its winning archive.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        if self.resolve(name).is_none() {
            if let Some(bytes) = crate::bundled_addons::read(name) {
                return Ok(bytes);
            }
        }
        let archive = self.archive_for(name)?;
        archive
            .read_file(name)
            .with_context(|| format!("reading {name} from {}", archive.path().display()))
    }

    /// The archive `name` reads from, as a cheap handle: a caller holding the chain behind a lock
    /// looks the file up under it and reads (`Archive::open_file`) after releasing it. No I/O.
    pub fn archive_for(&self, name: &str) -> Result<Archive> {
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
        Ok(archive.clone())
    }

    /// [`Self::read`] as a [`StoredBlob`]: always plain on native, where nothing is left to decode.
    pub fn read_stored(&self, name: &str) -> Result<StoredBlob> {
        self.read(name).map(StoredBlob::Plain)
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

    /// Unmount every archive above the one whose file name is `name` (any case), so the chain
    /// answers as the install did at that archive: a layer packed from some of an install's
    /// patches must not read the patches mounted over them. `false`, with nothing unmounted, when
    /// no mounted archive has that name. For the pack builder; nothing on the read path uses it.
    pub fn truncate_after(&mut self, name: &str) -> bool {
        let Some(at) = self.archives.iter().position(|a| {
            a.path()
                .file_name()
                .is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(name))
        }) else {
            return false;
        };
        self.archives.truncate(at + 1);
        true
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

    /// `EnumerateArchiveFiles` (`0x401470`): every `(listfile)` line that begins with `prefix`,
    /// without ASCII case and with `\` only as `\` (`SStrCmpI` at `0x4014fe`), as written and
    /// unresolved. The archives are the mount's slots below the base ones, the patches and
    /// `speech2.MPQ` in their opening order (`0x4014a0`-`0x4014bf`, up to `[0x882740]`), then
    /// `interface.MPQ`, base slot 6 (`0x4014c1`-`0x4014e1`); a slot with no archive is passed over.
    ///
    /// Not walked, as benilla mounts neither: the `speech2` child record each patch gets beside a
    /// real `speech2.MPQ` (`0x648e80` at `0x403886`), which the reference's slots also hold and
    /// which names its parent's `speech2\`-prefixed files, and an SZip archive, `0x648fb0`'s
    /// type-1 arm (`0x668b30`; `0x648e38` types any name not ending `.mpq`). A child's type word is
    /// never written (`0x648e85`-`0x648ea0`, `SMemAlloc` without the zero flag), so which arm
    /// reads it is not fixed by the bytes.
    pub fn enumerate_archive_files(&self, prefix: &str) -> Vec<String> {
        let named = |a: &&Archive, name: &str| {
            a.path()
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.eq_ignore_ascii_case(name))
        };
        let base = |a: &&Archive| VANILLA_BASE_ORDER.iter().any(|b| named(a, b));
        let interface = self.archives.iter().find(|a| named(a, "interface.MPQ"));
        let mut out = Vec::new();
        for archive in self.archives.iter().filter(|a| !base(a)).chain(interface) {
            if let Ok(listfile) = archive.read_file("(listfile)") {
                listfile_lines(&listfile, prefix, &mut out);
            }
        }
        out
    }
}

/// The longest line `0x648fb0` keeps: its buffer takes bytes while below `[ebp-5]` of `[ebp-0x108]`
/// (`0x64902a`-`0x649031`), and drops the rest of a longer one.
const LISTFILE_LINE_MAX: usize = 0x103;

#[cfg(not(target_arch = "wasm32"))]
/// `EnumerateArchiveListfile` (`0x648fb0`) over one listfile, with `0x4014f0`'s prefix test: a line
/// ends at a CR or LF (`0x649022`, `0x649026`); an empty one, and one whose text from its last dot
/// is `.md5` without case (`0x865bd8`, `0x64905f`), is passed over. Deviation: a last line with no
/// CR or LF after it is passed too, where `0x648fb0` ends at the buffer's end without a flush
/// (`0x64907f`) and drops it, a bug that loses a listfile's last name when it lacks a newline.
fn listfile_lines(listfile: &[u8], prefix: &str, out: &mut Vec<String>) {
    let mut line: Vec<u8> = Vec::with_capacity(LISTFILE_LINE_MAX);
    for &b in listfile {
        if b != b'\r' && b != b'\n' {
            if line.len() < LISTFILE_LINE_MAX {
                line.push(b);
            }
            continue;
        }
        pass_line(&line, prefix, out);
        line.clear();
    }
    pass_line(&line, prefix, out);
}

/// One finished listfile line, passed unless empty, a `.md5` or outside the prefix.
fn pass_line(line: &[u8], prefix: &str, out: &mut Vec<String>) {
    let md5 = line
        .iter()
        .rposition(|&c| c == b'.')
        .is_some_and(|dot| line[dot..].eq_ignore_ascii_case(b".md5"));
    let under =
        line.len() >= prefix.len() && line[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes());
    if !line.is_empty() && !md5 && under {
        out.push(String::from_utf8_lossy(line).into_owned());
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
    ///
    /// The HD layer is the page's one live mount ([`crate::layers::hd`]), shared by every chain:
    /// the first open with `hd_models` mounts it, and a later open finds it mounted. `false` leaves
    /// a mount alone; turning the layer off is [`crate::web::close_hd_layer`]'s.
    pub fn open_layered(_path: &Path, hd_models: bool) -> Result<Self> {
        let store = crate::webpack::PackStore::open(crate::web::data_base())?;
        if hd_models && !crate::layers::hd_mounted() {
            match crate::webpack::PackStore::open_hd(crate::web::hd_data_base()) {
                Ok(hd) => crate::layers::mount_hd(hd),
                Err(e) => crate::web::console_warn(&format!(
                    "updated character models are on but this origin serves no {} container \
                     ({e:#}) — running on the original art",
                    crate::layers::HD_DATA_SEGMENT
                )),
            }
        }
        let mut chain = Self {
            store,
            locale: Locale::default(),
        };
        chain.locale = crate::wow_ini::client_locale(&chain);
        Ok(chain)
    }

    /// The client locale, from the container's `Wow.ini` (`0x402350`): its index picks the column
    /// of every localized DBC string.
    pub fn locale(&self) -> Locale {
        self.locale
    }

    /// The container that should answer for `name` — [`crate::layers::pick_layer`] over the layers
    /// mounted now: the reforged art when it holds the name, else the HD overlay when it does, else
    /// the Turtle layer when it is mounted and holds it, else the base container; a table the
    /// Turtle layer merged with the HD layer's answers under its alias while both are mounted.
    /// **The whole of the priority law on this target**, in one place so no method below restates
    /// it.
    ///
    /// `contains` on a `PackStore` is a binary search over an index already in memory, so asking the
    /// overlays first costs ~17 probes each and no I/O. On a 1.12.1 realm the Turtle layer costs one
    /// relaxed atomic load here ([`crate::layers::turtle`]), and only a `.dbc` read with both layers
    /// mounted builds an alias name.
    ///
    /// `mounts` is the caller's handle on the live layers, so the answer can borrow them.
    fn pick<'a>(
        &'a self,
        name: &str,
        mounts: &'a Mounts,
    ) -> (&'a crate::webpack::PackStore, Option<String>) {
        let (hd, turtle) = (mounts.hd.as_deref(), mounts.turtle.as_deref());
        let alias = match (hd, turtle) {
            (Some(_), Some(t)) if is_table(name) => {
                let alias = crate::layers::turtle_hd_alias(name);
                t.contains(&alias).then_some(alias)
            }
            _ => None,
        };
        let reforged = mounts.reforged.as_deref();
        let layer = crate::layers::pick_layer(
            reforged.map(|r| r.contains(name)),
            hd.map(|hd| hd.contains(name)),
            turtle.map(|t| t.contains(name)),
            alias.is_some(),
        );
        // The layer halves of the boot trace, so a `?boottrace=1` capture can say whether a layer is
        // being read at all and which names reach it (`web::trace_hd`); no-ops unless the page armed
        // the arrays.
        match (layer, turtle, hd, reforged) {
            (crate::layers::Layer::Reforged, _, _, Some(r)) => {
                crate::web::trace_reforged(name);
                (r, None)
            }
            (crate::layers::Layer::TurtleHdAlias, Some(t), _, _) => {
                crate::web::trace_turtle(name);
                (t, alias)
            }
            (crate::layers::Layer::Hd, _, Some(hd), _) => {
                crate::web::trace_hd(name);
                (hd, None)
            }
            (crate::layers::Layer::Turtle, Some(t), _, _) => {
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

    /// Whether the updated-character-model layer is mounted now — for the boot report and the
    /// diagnostics line, which should say which corpus a session is actually running on.
    pub fn hd_character_models(&self) -> bool {
        crate::layers::hd_mounted()
    }

    /// Whether the container holds `name` (accepts `/` or `\`; case-insensitive — the equivalence is
    /// MPQ's own, inherited through `benilla_mpq::name_key` rather than restated).
    ///
    /// No I/O, no allocation, no network: a binary search over the index's sorted key array.
    pub fn contains(&self, name: &str) -> bool {
        crate::layers::reforged().is_some_and(|r| r.contains(name))
            || crate::layers::hd().is_some_and(|hd| hd.contains(name))
            || crate::layers::turtle().is_some_and(|t| t.contains(name))
            || self.store.contains(name)
            || crate::bundled_addons::contains(name)
    }

    /// A bundled addon's file ([`crate::bundled_addons`]), for a name no container holds.
    fn bundled(&self, name: &str) -> Option<Vec<u8>> {
        if !crate::bundled_addons::contains(name) {
            return None; // every other read leaves at the prefix check
        }
        let held = crate::layers::reforged().is_some_and(|r| r.contains(name))
            || crate::layers::hd().is_some_and(|hd| hd.contains(name))
            || crate::layers::turtle().is_some_and(|t| t.contains(name))
            || self.store.contains(name);
        if held {
            return None;
        }
        crate::bundled_addons::read(name)
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
        self.read_stored(name)?
            .into_bytes()
            .map_err(|e| anyhow::anyhow!("{name}: {e}"))
    }

    /// [`Self::read`] without the frame decode: a compressed frame comes back as
    /// [`StoredBlob::Frame`], for a caller that decompresses it off the thread that draws.
    pub fn read_stored(&self, name: &str) -> Result<StoredBlob> {
        crate::web::trace(name); // boot-manifest capture; no-op unless the page armed it
        if let Some(bytes) = self.bundled(name) {
            return Ok(StoredBlob::Plain(bytes));
        }
        let mounts = Mounts::now();
        let (store, alias) = self.pick(name, &mounts);
        store.read_sync_stored(alias.as_deref().unwrap_or(name))
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
        // A bundled addon file is a small blocking fetch, once per name: the sprite and font caches
        // keep what it answers.
        if let Some(bytes) = self.bundled(name) {
            return TryRead::Ready(bytes);
        }
        let mounts = Mounts::now();
        let (store, alias) = self.pick(name, &mounts);
        store.try_read_sync(alias.as_deref().unwrap_or(name))
    }

    /// Would [`Self::try_read`] answer without deferring? Starts the warm if not — see
    /// [`crate::webpack::PackStore::is_ready`], including why a name the container does not hold is
    /// reported ready.
    ///
    /// For a caller whose work reads several blobs and is only correct once it can read all of them:
    /// it asks first, does nothing while the answer is no, and the packs warm in the meantime.
    pub fn ready(&self, name: &str) -> bool {
        let mounts = Mounts::now();
        let (store, alias) = self.pick(name, &mounts);
        store.is_ready(alias.as_deref().unwrap_or(name))
    }

    /// Warm the pack `name` lives in, and return immediately — see
    /// [`crate::webpack::PackStore::prefetch`]. Idempotent and cheap enough to call every frame.
    pub fn prefetch(&self, name: &str) {
        let mounts = Mounts::now();
        let (store, alias) = self.pick(name, &mounts);
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
        self.read_async_stored(name)
            .await?
            .into_bytes()
            .map_err(|e| anyhow::anyhow!("{name}: {e}"))
    }

    /// [`Self::read_async`] without the frame decode, as [`Self::read_stored`]: the asset reader's
    /// lane, which decompresses on a worker (`benilla_assets::frame_budget::offload`).
    pub async fn read_async_stored(&self, name: &str) -> Result<StoredBlob> {
        crate::web::trace(name);
        if let Some(bytes) = self.bundled(name) {
            return Ok(StoredBlob::Plain(bytes));
        }
        let mounts = Mounts::now();
        let (store, alias) = self.pick(name, &mounts);
        store
            .read_async_stored(alias.as_deref().unwrap_or(name))
            .await
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
        let mounts = Mounts::now();
        let overlays = mounts
            .reforged
            .as_deref()
            .into_iter()
            .chain(mounts.hd.as_deref())
            .chain(mounts.turtle.as_deref());
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

    /// The web twin of the native `EnumerateArchiveFiles` (`0x401470`): the [`Chain::list`] names
    /// under `prefix`, each cut and tested as one listfile line is ([`pass_line`]). Deviation: the
    /// container ships one listfile for the whole chain, its layers' appended, so the base
    /// archives' names are walked too where the reference reads only the patches, `speech2.MPQ`
    /// and `interface.MPQ`; a name is spelled with `\`, whichever separator the listing wrote.
    pub fn enumerate_archive_files(&self, prefix: &str) -> Vec<String> {
        let mut out = Vec::new();
        for entry in self.list().unwrap_or_default() {
            let name = entry.name.replace('/', "\\");
            let line = &name.as_bytes()[..name.len().min(LISTFILE_LINE_MAX)];
            pass_line(line, prefix, &mut out);
        }
        out
    }
}

// Native only: exercises `is_patch_glob_match`/`mount_order`, which don't exist on the web target.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Lines end at CR or LF; the prefix compares without case and reads `/` as itself; `.md5`
    /// lines and empty lines are passed over, an unterminated last line is kept; a long line is cut.
    #[test]
    fn a_listfile_walk_passes_the_lines_under_the_prefix() {
        let long = format!("Interface\\Icons\\{}", "x".repeat(400));
        let text = format!(
            "Interface\\Icons\\Spell_A.blp\r\ninterface\\ICONS\\b.blp\n\nInterface/Icons/c.blp\n\
             Interface\\Icons\\d.blp.MD5\rTextures\\e.blp\n{long}\nInterface\\Icons\\last.blp"
        );
        let mut out = Vec::new();
        listfile_lines(text.as_bytes(), "Interface\\Icons\\", &mut out);
        assert_eq!(out.len(), 4, "{out:?}");
        assert_eq!(out[0], "Interface\\Icons\\Spell_A.blp");
        assert_eq!(out[1], "interface\\ICONS\\b.blp");
        assert_eq!(out[2].len(), LISTFILE_LINE_MAX);
        assert_eq!(
            out[3], "Interface\\Icons\\last.blp",
            "the unterminated last line"
        );
    }

    /// A minimal V1 archive holding stored files, its tables encrypted as the reader expects: the
    /// MPQ string hash and table cipher, written out here since `benilla-mpq` keeps its own private.
    fn fixture_mpq(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut table = [0u32; 0x500];
        let mut seed: u32 = 0x0010_0001;
        for i in 0..0x100 {
            for j in 0..5 {
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                let hi = (seed & 0xFFFF) << 16;
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                table[i + j * 0x100] = hi | (seed & 0xFFFF);
            }
        }
        let hash = |name: &str, kind: usize| {
            let (mut s1, mut s2): (u32, u32) = (0x7FED_7FED, 0xEEEE_EEEE);
            for b in name.bytes() {
                let ch = b.to_ascii_uppercase() as u32;
                s1 = table[kind + ch as usize] ^ s1.wrapping_add(s2);
                s2 = ch
                    .wrapping_add(s1)
                    .wrapping_add(s2)
                    .wrapping_add(s2 << 5)
                    .wrapping_add(3);
            }
            s1
        };
        let encrypt = |data: &mut [u32], mut key: u32| {
            let mut seed: u32 = 0xEEEE_EEEE;
            for v in data.iter_mut() {
                seed = seed.wrapping_add(table[0x400 + (key & 0xFF) as usize]);
                let plain = *v;
                *v = plain ^ key.wrapping_add(seed);
                key = (!key << 0x15).wrapping_add(0x1111_1111) | (key >> 0x0B);
                seed = plain
                    .wrapping_add(seed)
                    .wrapping_add(seed << 5)
                    .wrapping_add(3);
            }
        };
        const SLOTS: usize = 16;
        let hash_pos = 32u32;
        let block_pos = hash_pos + SLOTS as u32 * 16;
        let mut data_pos = block_pos + files.len() as u32 * 16;
        let mut hashes = vec![0xFFFF_FFFFu32; SLOTS * 4];
        let mut blocks = Vec::new();
        let mut body = Vec::new();
        for (i, (name, bytes)) in files.iter().enumerate() {
            let mut slot = hash(name, 0) as usize & (SLOTS - 1);
            while hashes[slot * 4 + 3] != 0xFFFF_FFFF {
                slot = (slot + 1) & (SLOTS - 1);
            }
            hashes[slot * 4..slot * 4 + 4].copy_from_slice(&[
                hash(name, 0x100),
                hash(name, 0x200),
                0,
                i as u32,
            ]);
            blocks.extend([
                data_pos,
                bytes.len() as u32,
                bytes.len() as u32,
                0x8000_0000,
            ]);
            body.extend_from_slice(bytes);
            data_pos += bytes.len() as u32;
        }
        encrypt(&mut hashes, hash("(hash table)", 0x300));
        encrypt(&mut blocks, hash("(block table)", 0x300));
        let mut out = Vec::new();
        for w in [
            0x1A51_504D,
            32,
            data_pos,
            0,
            hash_pos,
            block_pos,
            SLOTS as u32,
        ] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&(files.len() as u32).to_le_bytes());
        // The 16-bit format version and block size shift (bytes 12..16) stay 0.
        for w in hashes.into_iter().chain(blocks) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&body);
        out
    }

    /// The walk reads the patches and `speech2.MPQ` in mount order, then `interface.MPQ`, and no
    /// other base archive, each listfile's lines as written.
    #[test]
    fn the_archive_walk_reads_the_patches_then_interface_mpq() {
        let dir = std::env::temp_dir().join(format!(
            "benilla-chain-walk-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for archive in [
            "interface.MPQ",
            "model.MPQ",
            "dbc.MPQ",
            "patch.MPQ",
            "patch-2.MPQ",
            "speech2.MPQ",
        ] {
            let line = format!("X\\{archive}\r\nY\\{archive}\r\n");
            std::fs::write(
                dir.join(archive),
                fixture_mpq(&[("(listfile)", line.as_bytes())]),
            )
            .unwrap();
        }
        let chain = Chain::open(&dir).expect("open the fixture chain");
        let walked = chain.enumerate_archive_files("x\\");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            walked,
            [
                "X\\patch.MPQ",
                "X\\patch-2.MPQ",
                "X\\speech2.MPQ",
                "X\\interface.MPQ"
            ]
        );
    }

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
