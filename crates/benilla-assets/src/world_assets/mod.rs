//! The shared asset store: the one open patch chain, the dedup caches every model and texture load
//! goes through, and the helpers that turn BLP pixels into textures and per-submesh materials. It
//! reads no game state.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Buffer, Face};

use crate::materials::{WowModelExt, WowModelMaterial, VANILLA_ALPHA_KEY_REF};
use crate::SpatialCache;
/// Native only: the web target's [`WorldAssets::texture`] hands the same read to the `AssetServer`
/// instead of decoding it here.
#[cfg(not(target_arch = "wasm32"))]
use benilla_formats::read_texture_native_chain;
use benilla_formats::{
    blp_to_rgba, tga_texture_size, tga_to_rgba, Chain, ModelBlend, TryRead, TGA_GATE_PREFIX,
};
pub use benilla_formats::{LooseFile, LooseIndex, LooseIndexCell, LooseOpen};

/// BLP and RGBA to `Image` helpers, re-exported.
mod images;
pub use images::*;

/// How many times a sprite resolution has answered [`SpriteLoad::Pending`]: the signal that a
/// cached UI conversion is stale. A caller that draws once and keeps the result while its inputs
/// stay equal (the FrameXML extract gate) would otherwise hold the placeholder for as long as a
/// window is open. Snapshotted around a conversion, it tells that conversion whether anything it
/// resolved has yet to arrive. Bumped inside [`decode_sprite`], so it counts every sprite lane.
static SPRITE_DEFERRALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Read [`SPRITE_DEFERRALS`]. Snapshot it before a conversion and compare after: unequal means
/// something that conversion drew is still coming, and the result must not be cached as final.
pub fn sprite_deferrals() -> u64 {
    SPRITE_DEFERRALS.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`WorldAssets::pending_sprite`]'s slot in the `generated` cache. A reserved name, not a path: no
/// BLP is ever asked for under it.
const PENDING_SPRITE_KEY: &str = "@pending";

/// The shared asset store: the one open patch chain and the dedup caches, so each BLP decodes once
/// and submeshes sharing a texture and blend share a material that Bevy batches.
#[derive(Resource)]
pub struct WorldAssets {
    /// The [`AssetServer`], which [`Self::texture`] hands its loads to on the web target. Held on
    /// every target so `open`'s signature does not fork.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    server: AssetServer,
    /// The one open patch chain. The `Mutex` is `Send + Sync` glue for the streaming loaders'
    /// `&mut Chain`, not a read serializer: `Chain` reads are `&self` on a fresh OS handle.
    pub chain: Arc<Mutex<Chain>>,
    /// Decoded GPU textures by `(normalized path, wrap_u, wrap_v)`: one BLP can be sampled both
    /// ways (tiled on a wall, clamped on a cutout card), and the address mode lives on the sampler.
    pub textures: SpatialCache<(String, bool, bool), Handle<Image>>,
    /// The UI sprite caches.
    cache: SpriteCaches,
    /// UI sprites whose pixels the client computes ([`Self::generated_sprite`]).
    generated: HashMap<&'static str, Handle<Image>>,
    /// The loose files a flagged open reads before the chain: the install's index, and the AddOns
    /// folder once the app installs it (never in capture runs).
    loose: LooseFiles,
    /// Materials deduped by [`MaterialKey`].
    pub model_materials: SpatialCache<MaterialKey, Handle<WowModelMaterial>>,
    /// The shared global-light buffer, cloned into every deduped model material's `light_buf`.
    pub shared_light: Buffer,
}

/// Identity of a deduped model material; everything textureless shares one fallback per WMO flag.
/// The cutoff and fade distance are in it so ground clutter never shares a tree's material.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum MaterialKey {
    /// `(texture, blend, two_sided, alpha_key_u8, fade_far_yd_u16, is_wmo, is_fade_variant,
    /// wrap_u, wrap_v)`; the address mode selects a different `Image` upload.
    Textured(String, ModelBlend, bool, u8, u16, bool, bool, bool, bool),
    Fallback(bool),
}

/// Fold case and slashes so variants of one file share a cache entry, as MPQ lookup does.
pub fn normalize_path(path: &str) -> String {
    path.replace('/', "\\").to_ascii_lowercase()
}

/// The cache key a UI sprite reference folds to: [`normalize_path`], then any three-character
/// extension stripped, as the reference strips it (`0x449590`) and keys its texture cache on the
/// stem. `Foo`, `Foo.blp` and `Foo.tga` are one entry; `Foo.jpeg` keys as itself.
pub fn sprite_key(path: &str) -> String {
    let key = normalize_path(path);
    match key.len().checked_sub(4) {
        Some(dot) if key.as_bytes()[dot] == b'.' => key[..dot].to_string(),
        _ => key,
    }
}

/// The two files a UI sprite reference may live in, in the order of `TextureCreate`'s two-candidate
/// chain (`0x449d90`, extension table `0x835248 = {".tga", ".blp"}`): `.tga` first for a `.tga`
/// reference, `.blp` first otherwise, and a miss on the first is silent. The sweeps that assert our
/// own paths resolve call this too, so they cannot disagree with what draws.
pub fn sprite_candidates(path: &str) -> [String; 2] {
    let stem = sprite_key(path);
    let normalized = normalize_path(path);
    // `normalize_path` has already folded case, so a plain compare is the case-insensitive one.
    if normalized.len() >= 4 && normalized[normalized.len() - 4..] == *".tga" {
        [format!("{stem}.tga"), format!("{stem}.blp")]
    } else {
        [format!("{stem}.blp"), format!("{stem}.tga")]
    }
}

/// The texel size of a UI sprite, which `CSimpleTexture`'s size getters read (`[tex+0x144]`,
/// `[tex+0x148]`; `0x770720`, `0x770790`) for a region with no authored size, one texel per
/// FrameXML unit. Decoded, so layout gets the size the screen shows, and read by an
/// interface texture's open ([`LooseOpen::Index`]) as `TextureCreate` makes it ([`decode_texture`]);
/// three-valued, because the art can still be coming ([`SpriteDims`]).
pub fn sprite_dimensions(chain: &Mutex<Chain>, loose: &LooseFiles, path: &str) -> SpriteDims {
    match decode_texture(chain, loose, path, LooseOpen::Index, ReadMode::Defer) {
        SpriteLoad::Ready((w, h, _)) => SpriteDims::Ready(w, h),
        SpriteLoad::Pending => SpriteDims::Pending,
        SpriteLoad::Missing => SpriteDims::Missing,
    }
}

/// [`sprite_dimensions`]'s answer, three-valued for the same reason the sprite caches' is: the
/// caller **memoises**, and a measurement taken while the art was still in flight must not be the
/// number a region keeps for the rest of the session.
///
/// A `Missing` is worth remembering — a path that resolves to nothing measures nothing however often
/// it is asked. A `Pending` is worth nothing at all. On native only the first and last arms occur.
pub enum SpriteDims {
    Ready(u32, u32),
    /// The art exists but has not arrived. Ask again; do not remember this.
    Pending,
    Missing,
}

/// Decode the first of [`sprite_candidates`] that reads and decodes, each read by
/// [`read_loose_or_chain`]'s rule for `open`, as the reference's two loaders fail silently. A miss
/// warns, once per key since every cache stores its misses: a missing sprite draws nothing, and
/// nothing else says why.
fn decode_sprite(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
    open: LooseOpen,
    mode: ReadMode,
) -> SpriteLoad {
    decode_sprite_as(chain, loose, path, open, mode, OnRefusal::Miss)
}

/// [`decode_sprite`] for a load that `TextureCreate` (`0x449d90`) makes: the first candidate that
/// makes a texture is the texture, whether or not its pixels decode
/// ([`OnRefusal::WhiteDefault`]).
fn decode_texture(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
    open: LooseOpen,
    mode: ReadMode,
) -> SpriteLoad {
    decode_sprite_as(chain, loose, path, open, mode, OnRefusal::WhiteDefault)
}

/// What a candidate file that opens but will not decode becomes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OnRefusal {
    /// Nothing: the next candidate is tried, then the sprite is a miss.
    Miss,
    /// `TextureCreate`'s TGA loader `0x44a140` returns the texture once the header opens
    /// (`0x44a169`) and decodes later, in the callback `0x44a260`; a refused decode fills opaque
    /// white at the header's size (`0x44a2eb` into `0x448920`) and logs the reference's line
    /// (`0x835538`), with no next candidate. A header that makes no texture (`0x448450`, see
    /// [`tga_texture_size`]) fails the candidate as a missing file does, whether the pixels would
    /// decode or not. The BLP loader `0x44a310` returns the texture once the file opens
    /// (`0x44a399`), and a parse or decode its callback refuses (`0x44a560`) leaves the solid
    /// green texture (`0x44a50e`), with no next candidate either.
    WhiteDefault,
}

fn decode_sprite_as(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
    open: LooseOpen,
    mode: ReadMode,
    on_refusal: OnRefusal,
) -> SpriteLoad {
    let open = loose_open_for(path, open);
    let candidates = sprite_candidates(path);
    // Asked before the chain's lock, so an index build never stalls an archive reader.
    let hits: Vec<Option<LooseFile>> = candidates.iter().map(|c| loose.find(c, open)).collect();
    let chain = chain.lock_recover();
    // Set by any candidate the store has but has not got here yet. It does not short-circuit the
    // walk — a later candidate may be resident and answer outright — but it does decide the verdict
    // if none is, because "not yet" and "not there" must not produce the same cached answer.
    let mut waiting = false;
    // Candidates that a store DID answer and the decoder then refused — the difference between
    // "no such file" and "that file is broken", which the miss warning below reports apart.
    let mut undecodable: Vec<&str> = Vec::new();
    for (candidate, hit) in candidates.iter().zip(&hits) {
        // A loose file answers the candidate whether or not it decodes, and one that will not
        // read is a miss with no turn to the chain ([`read_loose_or_chain`]).
        let bytes = if let Some(file) = hit {
            match file.read() {
                Some(bytes) => bytes,
                None => continue,
            }
        } else {
            match mode.read(&chain, candidate) {
                TryRead::Ready(bytes) => bytes,
                TryRead::Pending => {
                    waiting = true;
                    continue;
                }
                TryRead::Missing => continue,
            }
        };
        match decode_candidate(candidate, &bytes, on_refusal) {
            Candidate::Pixels(width, height, rgba) => {
                return SpriteLoad::Ready((width, height, rgba));
            }
            Candidate::Next => undecodable.push(candidate),
        }
    }
    if waiting {
        SPRITE_DEFERRALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return SpriteLoad::Pending;
    }
    // "Not there" and "there but would not decode" are different faults; say which. The walk
    // above already read every candidate and collected the undecodable ones, so nothing is
    // re-read here (a browser re-read would be a second trip for bytes already in hand).
    if undecodable.is_empty() {
        warn!(
            "texture miss: '{path}' does not resolve in the patch chain{} (tried {})",
            if open == LooseOpen::Archives {
                ""
            } else {
                ", the install's loose files or the AddOns folder"
            },
            candidates.join(", ")
        );
    } else {
        warn!(
            "texture miss: '{path}' RESOLVES but will not decode ({}) — the file is there and the \
             decoder refused it",
            undecodable.join(", ")
        );
    }
    SpriteLoad::Missing
}

/// Which read lane a sprite resolution takes — the answer to "**can this caller come back?**".
///
/// [`ReadMode::Defer`] is right for anything asked per frame: the FrameXML extract re-resolves every
/// texture on every conversion, so a `Pending` costs a frame or two of transparent placeholder and
/// nothing else.
///
/// [`ReadMode::Block`] is for the **one-shot warmers** — the glue screens' art, the chat-bubble and
/// V-plate boot sets, the loading-screen backdrop — which resolve a fixed set once and then keep the
/// handles (and, in two cases, bake pixel sizes off them) for the rest of the session. A placeholder
/// reaching one of those is permanent: the art never appears, and the size baked from a 1x1 image is
/// wrong forever. They pay a blocking read instead, which costs what it has always cost and lands on
/// a boot or loading frame where there is no smooth animation to protect.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadMode {
    /// May answer [`SpriteLoad::Pending`]. The caller must ask again and cache nothing.
    Defer,
    /// Never answers `Pending`; blocks until the bytes are in hand (on native both are the same read).
    Block,
}

impl ReadMode {
    fn read(self, chain: &Chain, name: &str) -> TryRead {
        match self {
            ReadMode::Defer => chain.try_read(name),
            ReadMode::Block => match chain.read(name) {
                Ok(bytes) => TryRead::Ready(bytes),
                Err(_) => TryRead::Missing,
            },
        }
    }
}

/// What [`decode_sprite`] found. The third arm exists for the browser, where a texture whose pack
/// is not resident is neither a hit nor a miss: it is a fetch in flight, and the caller's job is to
/// draw without it for a frame or two and ask again.
///
/// It matters because every sprite cache below stores **misses as well as hits** — the UI extractor
/// asks per quad per frame, so an uncached miss would re-walk the chain forever. Caching a `Pending`
/// under that rule would be permanent: the icon would arrive in the resident map a moment later and
/// nothing would ever look again.
enum SpriteLoad {
    Ready((u32, u32, Vec<u8>)),
    /// In flight. Do not cache; ask again next frame.
    Pending,
    /// Nothing in the chain (or the AddOns folder) resolves this path, or everything that did
    /// refused to decode. Cached, and warned about once.
    Missing,
}

/// The open a name makes: `open`, unless the name is written with `/`, which no loose file
/// answers ([`LooseFiles::find`]), so it reads as an open with flag 0 would. The candidates the
/// sprite loaders try are normalized, so the rule is taken from the name as the caller wrote it.
fn loose_open_for(path: &str, open: LooseOpen) -> LooseOpen {
    if path.contains('/') {
        LooseOpen::Archives
    } else {
        open
    }
}

/// Whether an interface texture's open of `path` makes a texture, in the loose files or the chain,
/// for the probes that answer `SetTexture`'s 1|nil before anything decodes: a candidate that opens
/// and, if it is a TGA, whose header passes the size gate ([`makes_texture`]). A fresh
/// [`SpriteProbe`] answers it, reading a TGA each time; a host asked per frame keeps one.
pub fn sprite_found(chain: &Mutex<Chain>, loose: &LooseFiles, path: &str) -> bool {
    SpriteProbe::default().found(chain, loose, path)
}

/// `SetTexture`'s 1|nil probe ([`sprite_found`]), which keeps the size gate's answer for each
/// resolved file it has read: an addon sets an icon from its `OnUpdate`, every frame. The host
/// makes one probe per UI load, so an answer lasts that load and a `/reload` asks again; a file
/// edited meanwhile is not read again. A file that will not read is asked again, and whether a
/// name opens at all is asked each time, since the loose index can change.
#[derive(Default)]
pub struct SpriteProbe {
    /// The gate's answer by resolved file ([`source_key`]).
    gate: std::cell::RefCell<HashMap<String, bool>>,
    /// How many TGA headers [`Self::found`] has read.
    header_reads: std::cell::Cell<usize>,
}

impl SpriteProbe {
    /// See [`sprite_found`].
    pub fn found(&self, chain: &Mutex<Chain>, loose: &LooseFiles, path: &str) -> bool {
        let open = loose_open_for(path, LooseOpen::Index);
        let candidates = sprite_candidates(path);
        // The loose files first, outside the chain's lock.
        let hits: Vec<Option<LooseFile>> = candidates.iter().map(|c| loose.find(c, open)).collect();
        candidates.iter().zip(hits).any(|(candidate, hit)| {
            let opens = hit.is_some() || chain.lock_recover().contains(candidate);
            if !opens || !is_tga_name(candidate) {
                return opens;
            }
            let key = match &hit {
                Some(file) => source_key(file),
                None => format!("chain:{candidate}"),
            };
            // Per frame, so the chain is asked on the lane that never blocks ([`ReadMode::Defer`]).
            self.gate(&key, candidate, || match hit {
                Some(file) => match file.read_prefix(TGA_GATE_PREFIX) {
                    Some(bytes) => TryRead::Ready(bytes),
                    None => TryRead::Missing,
                },
                None => chain.lock_recover().try_read(candidate),
            })
        })
    }

    /// Whether a `.tga` candidate makes a texture, from the bytes `read` gives, for the resolved
    /// file `key` names.
    fn gate(&self, key: &str, candidate: &str, read: impl FnOnce() -> TryRead) -> bool {
        if let Some(&answer) = self.gate.borrow().get(key) {
            return answer;
        }
        self.header_reads.set(self.header_reads.get() + 1);
        let bytes = match read() {
            TryRead::Ready(bytes) => bytes,
            // Deviation (browser): a header still in flight answers by its open alone and is not
            // remembered, where the reference reads it at once; the gate decides once it lands.
            TryRead::Pending => return true,
            TryRead::Missing => return false,
        };
        let answer = makes_texture(candidate, &bytes);
        self.gate.borrow_mut().insert(key.to_string(), answer);
        answer
    }
}

/// The file an entry opens, as a key: the path on disk, and an archive's member.
fn source_key(file: &LooseFile) -> String {
    match file {
        LooseFile::Plain(path) => path.display().to_string(),
        LooseFile::Archive { archive, member } => format!("{}!{member}", archive.display()),
    }
}

/// Whether `TextureCreate` (`0x449d90`) makes a texture of a candidate's bytes: a TGA the loader
/// `0x44a140` reads needs a header that `0x448450` makes a device texture at ([`tga_texture_size`]),
/// decoded or not, whatever the bytes are. A `.blp` candidate makes one once it opens.
fn makes_texture(candidate: &str, bytes: &[u8]) -> bool {
    !is_tga_name(candidate) || tga_texture_size(bytes).is_some()
}

/// Whether `candidate` is read by the TGA loader: `TextureCreate` picks the loader by the
/// candidate's extension (`0x449e16`-`0x449e3d`), never by its bytes.
fn is_tga_name(candidate: &str) -> bool {
    candidate.ends_with(".tga")
}

/// What one candidate file yields for a load.
#[derive(Debug, PartialEq)]
enum Candidate {
    /// Pixels to draw.
    Pixels(u32, u32, Vec<u8>),
    /// No texture: the next candidate is tried.
    Next,
}

/// `TextureCreate`'s solid texture, the colour `0xff00ff00` (`0x447c00`) in an 8 by 8 texture
/// (`0x44a900`), as RGBA.
fn solid_green() -> (u32, u32, Vec<u8>) {
    (8, 8, [0, 0xff, 0, 0xff].repeat(64))
}

/// One candidate's bytes through the decoder its name picks ([`decode_by_name`]). By
/// `on_refusal`, a TGA load gates its header's size first ([`makes_texture`]), decoded or not, and
/// a refused decode of one that passes leaves that size of opaque white; a refused BLP is the
/// solid green.
fn decode_candidate(candidate: &str, bytes: &[u8], on_refusal: OnRefusal) -> Candidate {
    let texture = on_refusal == OnRefusal::WhiteDefault;
    let size = if texture && is_tga_name(candidate) {
        match tga_texture_size(bytes) {
            Some(size) => Some(size),
            None => return Candidate::Next,
        }
    } else {
        None
    };
    if let Ok((width, height, rgba)) = decode_by_name(candidate, bytes) {
        return Candidate::Pixels(width, height, rgba);
    }
    match size {
        Some((width, height)) => {
            warn!("Texture {candidate} not loaded -- replaced with default.");
            Candidate::Pixels(
                width,
                height,
                vec![0xff; width as usize * height as usize * 4],
            )
        }
        None if texture => {
            warn!("Texture failure: \"{candidate}\" -- replaced with the solid texture.");
            let (width, height, rgba) = solid_green();
            Candidate::Pixels(width, height, rgba)
        }
        None => Candidate::Next,
    }
}

/// Decode a candidate by its name: a `.tga` through the TGA decoder, anything else through the BLP
/// one, as `TextureCreate` picks its loader (`0x449e16`-`0x449e3d`). A file whose bytes are the
/// other format is refused by the decoder its name picks.
fn decode_by_name(candidate: &str, bytes: &[u8]) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    if is_tga_name(candidate) {
        tga_to_rgba(bytes)
    } else {
        blp_to_rgba(bytes)
    }
}

/// The pixels of a minimap mask, read by `open`: a `TextureCreate` load (`Minimap:SetMaskTexture`
/// at `0x4ee5e4`, the world-entry load at `0x4ed862`), so a candidate that opens and will not
/// decode is the white or green default ([`decode_texture`]), and `None` only a name that does
/// not resolve.
fn mask_rgba(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
    open: LooseOpen,
    mode: ReadMode,
) -> SpriteLoad {
    decode_texture(chain, loose, path, open, mode)
}

/// One BLP file decoded, read with the loose flag ([`LooseOpen::Index`], or the archives for a
/// `path` written with `/`): the loads that open a `.blp` by name and nothing else.
fn decode_blp_file(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
    name: &str,
    mode: ReadMode,
) -> SpriteLoad {
    let bytes = match loose.find(name, loose_open_for(path, LooseOpen::Index)) {
        Some(file) => file.read(),
        None => match mode.read(&chain.lock_recover(), name) {
            TryRead::Ready(bytes) => Some(bytes),
            TryRead::Pending => {
                SPRITE_DEFERRALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return SpriteLoad::Pending;
            }
            TryRead::Missing => None,
        },
    };
    match bytes.map(|b| blp_to_rgba(&b)) {
        Some(Ok(decoded)) => SpriteLoad::Ready(decoded),
        Some(Err(e)) => {
            warn!("texture miss: '{name}' RESOLVES but will not decode as a BLP: {e:#}");
            SpriteLoad::Missing
        }
        None => {
            warn!("texture miss: '{name}' does not resolve");
            SpriteLoad::Missing
        }
    }
}

/// The files a tabard emblem cell reads: its name with `.BLP` appended (`0x50338a`, `0x83574c`),
/// whatever it ends in, and nothing else.
fn emblem_names(path: &str) -> Vec<String> {
    vec![format!("{}.blp", normalize_path(path))]
}

/// The pixels of a tabard emblem cell, read from the archives alone (`0x5a7b40` with flag 0) and
/// decoded as a BLP ([`emblem_names`]).
fn emblem_rgba(chain: &Mutex<Chain>, path: &str, mode: ReadMode) -> SpriteLoad {
    let chain = chain.lock_recover();
    let mut waiting = false;
    for name in emblem_names(path) {
        match mode.read(&chain, &name) {
            TryRead::Ready(bytes) => {
                if let Ok(decoded) = decode_by_name(&name, &bytes) {
                    return SpriteLoad::Ready(decoded);
                }
            }
            TryRead::Pending => waiting = true,
            TryRead::Missing => {}
        }
    }
    if waiting {
        SPRITE_DEFERRALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return SpriteLoad::Pending;
    }
    SpriteLoad::Missing
}

/// The pixels of a portrait: the sink `0x525d30` formats `%s.blp` onto the name as written
/// (`0x525e0f`, whatever it ends in) and reads it with the BLP reader `0x5a7b40` and flag 1
/// (`0x525e37`), so no `.tga` is read and `Foo.tga` opens `Foo.tga.blp`.
fn portrait_rgba(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
    mode: ReadMode,
) -> SpriteLoad {
    let name = format!("{}.blp", normalize_path(path));
    decode_blp_file(chain, loose, path, &name, mode)
}

/// The pixels of a picture the cursor code opens through `0x449610`: `SetCursor`'s custom cursor
/// (`0x523c86`) and the dragged icon (`0x5239f7`, `0x523a3e`, `0x523af7`, `0x523bb1`), all with the
/// loose flag. It strips a three-letter extension (`0x449590` at `0x44963c`) and appends `.blp`
/// (`0x449641`), and the bytes decode as a BLP (`0x449840`): `Foo.tga` opens `Foo.blp`, and no
/// `.tga` is read. Blocking: `0x523c98` waits on the load, and the dragged composite is a hardware
/// cursor for the whole drag, asked once.
pub fn file_texture_rgba(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
) -> Option<(u32, u32, Vec<u8>)> {
    let name = format!("{}.blp", sprite_key(path));
    match decode_blp_file(chain, loose, path, &name, ReadMode::Block) {
        SpriteLoad::Ready(decoded) => Some(decoded),
        // `Block` never defers; the arm exists because the enum has three of them.
        SpriteLoad::Pending | SpriteLoad::Missing => None,
    }
}

/// The loose files an open that sets the loose flag reads before the archive chain, as the
/// reference's resolver (`0x647e60`) answers from its index of the install (`0x647f89`) before the
/// chain (`0x648116`): the install's index for any path, and benilla's AddOns folder for an
/// `Interface\AddOns\` path ([`loose_addon_file`]), which stands in for the install's by a
/// written deviation (`benilla-app`'s `ui_script::addons::root`).
#[derive(Clone, Default)]
pub struct LooseFiles {
    /// benilla's AddOns folder, or `None` (always under `$WOW_CAPTURE`).
    pub addons: Option<PathBuf>,
    /// The install's index ([`benilla_formats::install_index`]), or `None` for a test's chain.
    pub install: Option<Arc<LooseIndexCell>>,
}

impl LooseFiles {
    /// The player's install, with no AddOns folder until the app installs one.
    pub fn install() -> Self {
        LooseFiles {
            addons: None,
            install: Some(benilla_formats::install_index()),
        }
    }

    /// The loose file that answers an open of `path` with `open`, or `None` for the archives:
    /// the font open's bare name first ([`LooseOpen::BareNameFirst`]), then the whole name, an
    /// `Interface\AddOns\` one in the AddOns folder, which the install's index leaves out. A name
    /// written with `/` finds nothing, as the index's compare does not read it as `\` (`0x647f37`);
    /// a caller whose path benilla joined with `/` turns it back first.
    pub fn find(&self, path: &str, open: LooseOpen) -> Option<LooseFile> {
        if open == LooseOpen::Archives || path.contains('/') {
            return None;
        }
        let install = self.install.as_ref().map(|cell| cell.get());
        if open == LooseOpen::BareNameFirst {
            let bare = path.rsplit('\\').next().unwrap_or(path);
            if let Some(file) = install.as_ref().and_then(|ix| ix.get(bare)) {
                return Some(file);
            }
        }
        let norm = normalize_path(path);
        if norm.starts_with(ADDONS_PREFIX) {
            return self
                .addons
                .as_deref()
                .and_then(|root| loose_addon_file(root, &norm))
                .map(LooseFile::Plain);
        }
        install.as_ref().and_then(|ix| ix.get(&norm))
    }
}

/// Read a file by path for an open of kind `open`: the loose file that answers it
/// ([`LooseFiles::find`]), else the patch chain. A file the loose files answer and the read then
/// fails is a miss, with no turn to the chain (`0x647865`).
pub fn read_loose_or_chain(
    chain: &Mutex<Chain>,
    loose: &LooseFiles,
    path: &str,
    open: LooseOpen,
) -> Option<Vec<u8>> {
    match loose.find(path, open) {
        Some(file) => file.read(),
        None => chain.lock_recover().read(path).ok(),
    }
}

/// The normalized prefix of an addon path.
const ADDONS_PREFIX: &str = "interface\\addons\\";

/// Map a normalized `interface\addons\…` path onto a file under the addon root, matching each
/// component case-insensitively as the Windows reference does; a dot-component, or one the host
/// reads as more than a plain name (a Windows `C:`), is refused, so a path cannot escape the root.
pub fn loose_addon_file(root: &Path, candidate: &str) -> Option<PathBuf> {
    let rel = candidate.strip_prefix(ADDONS_PREFIX)?;
    let mut at = root.to_path_buf();
    for (depth, comp) in rel.split('\\').enumerate() {
        let plain = matches!(
            Path::new(comp).components().collect::<Vec<_>>()[..],
            [std::path::Component::Normal(_)]
        );
        if !plain || comp.starts_with('.') {
            return None;
        }
        at = child_ignoring_case(&at, comp)?;
        if depth == 0 && moved_aside(&at, comp) {
            return None;
        }
    }
    at.is_file().then_some(at)
}

/// `dir`'s entry named `name` without regard to case, as on the reference's Windows filesystem.
fn child_ignoring_case(dir: &Path, name: &str) -> Option<PathBuf> {
    let direct = dir.join(name);
    if direct.exists() {
        return Some(direct);
    }
    let found = std::fs::read_dir(dir).ok()?.flatten().find(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| n.eq_ignore_ascii_case(name))
    })?;
    Some(found.path())
}

/// Whether the AddOns folder `dir`, named `name`, is one the glue build moves aside before the
/// add-on scan: a `Blizzard*` folder holding `<name>\<name>.toc` (`0x51fbf0`, walked by `0x51fb50`
/// at `0x46a8eb`), renamed `%s.old` so the stock add-on loads from the archive. benilla moves
/// nothing a player put there; it registers no add-on from such a folder and reads no file in it.
pub fn moved_aside(dir: &Path, name: &str) -> bool {
    name.get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case("blizzard"))
        && child_ignoring_case(dir, &format!("{name}.toc")).is_some_and(|toc| toc.is_file())
}

/// Sprite handles by key, a miss as `None`, each with the open that made it. The key holds no
/// open: the reference's texture cache keys on the stripped name and the flags word `0x449ae0`
/// builds from the filter and force arguments (`0x449dd2` to `0x449dec`, compared at `0x449f39`
/// and `0x449f47`), and the file flag is read only by the loaders (`0x449e20`, `0x449e32`), so a
/// name's first open wins while it is cached (`0x449df3`).
struct Opened<K>(HashMap<K, (LooseOpen, Option<Handle<Image>>)>);

impl<K> Default for Opened<K> {
    fn default() -> Self {
        Opened(HashMap::new())
    }
}

impl<K: Eq + std::hash::Hash> Opened<K> {
    /// The cached answer, `Some(None)` for a cached miss.
    fn get(&self, key: &K) -> Option<Option<Handle<Image>>> {
        self.0.get(key).map(|(_, handle)| handle.clone())
    }

    fn insert(&mut self, key: K, open: LooseOpen, handle: Option<Handle<Image>>) {
        self.0.insert(key, (open, handle));
    }

    /// Keep the entries `keep(key, open, hit)` says to.
    fn retain(&mut self, mut keep: impl FnMut(&K, LooseOpen, bool) -> bool) {
        self.0
            .retain(|key, (open, handle)| keep(key, *open, handle.is_some()));
    }
}

/// The UI sprite caches, and the install index's generation they were filled under.
#[derive(Default)]
struct SpriteCaches {
    /// UI sprites by [`sprite_key`].
    sprites: Opened<String>,
    /// UI sprites loaded with the wrap flag, by key: the sampler bakes into the `Image`.
    tiled: Opened<String>,
    /// Sprites resampled to an exact physical size, by `(path, w, h)`, always opened loose.
    resampled: HashMap<(String, u32, u32), Option<Handle<Image>>>,
    /// The decoded pixels behind [`Self::resampled`], so a resize does not re-decode.
    resample_sources: HashMap<String, Option<(u32, u32, Vec<u8>)>>,
    /// Portrait sprites, the circular mask baked in ([`portrait_image`]).
    portraits: Opened<String>,
    /// Coverage masks ([`mask_image`]): the minimap's circular clip.
    masks: Opened<String>,
    /// The [`LooseIndexCell::generation`] the entries opened loose were filled under.
    loose_generation: u64,
}

impl SpriteCaches {
    /// Drop every entry, hit or miss, keeping the index generation they were synced to.
    fn clear(&mut self) {
        *self = SpriteCaches {
            loose_generation: self.loose_generation,
            ..SpriteCaches::default()
        };
    }

    /// Drop every entry opened with the loose flag when the install's index was emptied since
    /// they filled, so the next request opens again, as the reference opens a texture again once
    /// it was released (every UI texture at a `ReloadUI` or a logout, the moments that empty the
    /// index). Entries opened with flag 0 stay: nothing loose answered them.
    fn sync(&mut self, generation: Option<u64>) {
        let Some(generation) = generation.filter(|g| *g != self.loose_generation) else {
            return;
        };
        self.loose_generation = generation;
        let archives = |_: &_, open: LooseOpen, _| open == LooseOpen::Archives;
        self.sprites.retain(archives);
        self.tiled.retain(|_, open, _| open == LooseOpen::Archives);
        self.portraits.retain(archives);
        self.masks.retain(archives);
        self.resampled.clear();
        self.resample_sources.clear();
    }

    /// Drop every miss and every `Interface\AddOns\` entry, when the AddOns folder changes: a path
    /// asked before it existed gets a second look, and a loose file shadows the chain's
    /// ([`read_loose_or_chain`]); any other hit came from the chain and stays.
    fn drop_addons_and_misses(&mut self) {
        // `emblem_mask_texture` prefixes its `sprites` key, so the test looks past a tag.
        let stays =
            |key: &str, hit: bool| hit && !normalize_path(key).contains("interface\\addons\\");
        self.sprites.retain(|k, _, hit| stays(k, hit));
        self.tiled.retain(|k, _, hit| stays(k, hit));
        self.resampled.retain(|k, v| stays(&k.0, v.is_some()));
        self.resample_sources.retain(|k, v| stays(k, v.is_some()));
        self.portraits.retain(|k, _, hit| stays(k, hit));
        self.masks.retain(|k, _, hit| stays(k, hit));
    }
}

/// Mutate an asset only when `differs` says it does not already match: `Assets::get_mut` alone
/// marks it Modified, a uniform re-upload and, on Metal's non-bindless path, a bind-group rebuild.
/// Pair it with values quantized to display precision ([`quantize`], [`quant255`]).
pub fn write_gated<M: bevy::asset::Asset>(
    assets: &mut Assets<M>,
    handle: &Handle<M>,
    differs: impl Fn(&M) -> bool,
    apply: impl FnOnce(&mut M),
) {
    if assets.get(handle).is_none_or(differs) {
        if let Some(mut m) = assets.get_mut(handle) {
            apply(&mut m);
        }
    }
}

/// Quantize to `n`ths, the write gate's floor for drifting inputs: 255 for colour ([`quant255`]),
/// a sub-pixel step for geometry.
pub fn quantize(x: f32, n: f32) -> f32 {
    (x * n).round() / n
}

/// [`quantize`] each channel to 1/255: the reference packs its sky and celestial colours to bytes.
pub fn quant255(c: [f32; 3]) -> [f32; 3] {
    [
        quantize(c[0], 255.0),
        quantize(c[1], 255.0),
        quantize(c[2], 255.0),
    ]
}

/// Lock a `Mutex`, recovering the guard if a holder panicked, so one failed read of the shared
/// chain does not fail every later lock; a half-read texture is harmless.
pub trait LockRecover<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> LockRecover<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl WorldAssets {
    /// Drop every cache filled from the chain — decoded textures, UI sprites and their misses, the
    /// derived images and the materials over them — for a change of the data the chain answers
    /// from (the Turtle layer mounted or unmounted). The next request re-reads the chain; a miss is
    /// forgotten too, since the new data may hold what the old did not. Procedural sprites stay.
    pub fn forget_chain_art(&mut self) {
        self.textures.clear();
        self.model_materials.clear();
        self.cache.clear();
    }

    /// Open the store over an opened patch chain. `shared_light` is a raw `Buffer`, not the
    /// client's `SharedLightBuffer`, which keeps this crate below the renderer.
    pub fn open(chain: Chain, shared_light: Buffer, server: AssetServer) -> Self {
        Self {
            server,
            chain: Arc::new(Mutex::new(chain)),
            textures: SpatialCache::default(),
            cache: SpriteCaches::default(),
            generated: HashMap::new(),
            loose: LooseFiles::install(),
            model_materials: SpatialCache::default(),
            shared_light,
        }
    }

    /// [`read_loose_or_chain`]'s rule with the loose flag set ([`LooseOpen::Index`]) and the chain
    /// half non-blocking ([`TryRead`]), for an open with no decoder here: every sound file
    /// (`0x7a8860`), as FMOD's file callbacks open with flag 1, and the glue's own interface files.
    /// Such a read must not fetch a pack synchronously on the page. A loose file that will not
    /// read is `Missing`, with no turn to the chain; the page has no loose files.
    pub fn try_read_loose_or_chain(&self, path: &str) -> TryRead {
        match self.loose.find(path, LooseOpen::Index) {
            Some(file) => file.read().map_or(TryRead::Missing, TryRead::Ready),
            None => self.chain.lock_recover().try_read(path),
        }
    }

    /// [`Self::try_read_loose_or_chain`]'s blocking twin, for a one-shot read at boot that has no
    /// frame to come back on: the glue's own interface files (`SFile::Load`, flag 1).
    pub fn read_file_or_loose(&self, path: &str) -> Option<Vec<u8>> {
        read_loose_or_chain(&self.chain, &self.loose, path, LooseOpen::Index)
    }

    /// The loose files this store reads, for a probe that must answer what draws.
    pub fn loose(&self) -> &LooseFiles {
        &self.loose
    }

    /// Install or clear the AddOns folder. Cached misses are evicted so a path asked before it
    /// existed gets a second look, and so is every `Interface\AddOns\` entry, since a loose file
    /// shadows the chain's ([`read_loose_or_chain`]); any other hit came from the chain and stays.
    pub fn set_loose_addon_root(&mut self, root: Option<PathBuf>) {
        if self.loose.addons == root {
            return;
        }
        self.loose.addons = root;
        self.cache.drop_addons_and_misses();
    }

    /// [`SpriteCaches::sync`] to the install index's generation.
    fn sync_loose(&mut self) {
        let generation = self.loose.install.as_ref().map(|c| c.generation());
        self.cache.sync(generation);
    }

    /// Decode and upload a BLP once per path and wrap; later requests share the handle.
    pub fn texture(
        &mut self,
        path: &str,
        wrap: (bool, bool),
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        let key = (normalize_path(path), wrap.0, wrap.1);
        if let Some(handle) = self.textures.fetch(&key) {
            return Some(handle);
        }
        let handle = self.upload_world_texture(&key.0, wrap, images)?;
        self.textures.insert(key, handle.clone());
        Some(handle)
    }

    /// [`Self::texture`]'s decode where loaders run on their own threads: read the BLP out of the
    /// chain and upload it now. World art uploads as the BLP stores it; nothing reads it main-side.
    #[cfg(not(target_arch = "wasm32"))]
    fn upload_world_texture(
        &self,
        key: &str,
        wrap: (bool, bool),
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        let chain = read_texture_native_chain(&mut self.chain.lock_recover(), key).ok()?;
        Some(images.add(repeat_texture_authored(crate::for_upload(chain), wrap)))
    }

    /// [`Self::texture`]'s decode in a browser, where a chain read on the frame thread is a
    /// blocking fetch of a pack nothing warmed. The load goes to the [`AssetServer`] at the same
    /// `mpq://` URL every other texture uses, which resolves to [`crate::BlpImageLoader`]'s
    /// `WorldArt` variant, this function's twin (the address mode rides the URL), so the `Image` is
    /// identical and only arrives later: the batch draws a few frames late instead of stalling.
    /// `contains` first: a path no archive holds must still return `None`, so `model_material`
    /// falls back to an untextured material instead of a handle that never loads.
    #[cfg(target_arch = "wasm32")]
    fn upload_world_texture(
        &self,
        key: &str,
        wrap: (bool, bool),
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        let _ = images; // the AssetServer owns the upload on this target
        if !self.chain.lock_recover().contains(key) {
            return None;
        }
        Some(self.server.load(crate::texture_url(key, wrap)))
    }

    /// A clamp-sampled sRGB sprite, mip 0 only, opened as an interface texture
    /// ([`LooseOpen::Index`], as `CSimpleTexture::SetTexture` loads at `0x770281`): every UI quad.
    pub fn sprite_texture(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.sprite_texture_opened(path, LooseOpen::Index, images)
    }

    /// [`Self::sprite_texture`] with the open the caller's reference load makes: the celestial
    /// discs, the loading screen and the minimap's default art open with flag 0
    /// ([`LooseOpen::Archives`]).
    pub fn sprite_texture_opened(
        &mut self,
        path: &str,
        open: LooseOpen,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.sync_loose();
        let key = sprite_key(path);
        // Hits and misses are cached: the UI extractor asks per quad per frame, and a fresh handle
        // would defeat its dirty check.
        if let Some(cached) = self.cache.sprites.get(&key) {
            return cached;
        }
        let loaded = match decode_texture(&self.chain, &self.loose, path, open, ReadMode::Defer) {
            SpriteLoad::Ready((w, h, rgba)) => Some(images.add(sprite_image(w, h, rgba))),
            // In flight: hand back the transparent placeholder and cache NOTHING, so the next
            // frame asks again and gets the real art the moment it lands.
            SpriteLoad::Pending => return Some(self.pending_sprite(images)),
            SpriteLoad::Missing => None,
        };
        let open = loose_open_for(path, open);
        self.cache.sprites.insert(key, open, loaded.clone());
        loaded
    }

    /// Decode a sprite to raw RGBA8 without uploading or caching, for a caller that resamples,
    /// read by `open`.
    pub fn decode_rgba(&mut self, path: &str, open: LooseOpen) -> Option<(u32, u32, Vec<u8>)> {
        match decode_sprite(&self.chain, &self.loose, path, open, ReadMode::Defer) {
            SpriteLoad::Ready(decoded) => Some(decoded),
            // Uncached and one-shot, so `Pending` folds into `None` here: the caller asks at load
            // and asks again if it needs to, and there is no cache entry to poison either way.
            SpriteLoad::Pending | SpriteLoad::Missing => None,
        }
    }

    /// A UI sprite resampled to an exact pixel size by the caller's kernel, cached by size: the
    /// nameplate border, whose 1 px bevel bilinear magnification would smear.
    pub fn resampled_sprite(
        &mut self,
        path: &str,
        (w, h): (u32, u32),
        images: &mut Assets<Image>,
        resample: impl FnOnce(&[u8], u32, u32, u32, u32) -> Vec<u8>,
    ) -> Option<Handle<Image>> {
        self.sync_loose();
        let key = (path.to_string(), w, h);
        if let Some(cached) = self.cache.resampled.get(&key) {
            return cached.clone();
        }
        if !self.cache.resample_sources.contains_key(path) {
            // `Defer`, and **a `Pending` caches nothing** — not the source and not the resampled
            // handle. A plate re-asks for its border every frame it is up, so a fetch in flight
            // costs a frame or two of unbordered plate and then resolves itself; writing either
            // cache under `Pending` would make that permanent, because the pack lands in the
            // resident map a moment later and nothing would ever look again. See [`ReadMode`].
            let decoded = match decode_texture(
                &self.chain,
                &self.loose,
                path,
                LooseOpen::Index,
                ReadMode::Defer,
            ) {
                SpriteLoad::Ready(decoded) => Some(decoded),
                SpriteLoad::Pending => return None,
                SpriteLoad::Missing => None,
            };
            self.cache
                .resample_sources
                .insert(path.to_string(), decoded);
        }
        let made = self.cache.resample_sources[path]
            .as_ref()
            .map(|(sw, sh, src)| images.add(sprite_image(w, h, resample(src, *sw, *sh, w, h))));
        self.cache.resampled.insert(key, made.clone());
        made
    }

    /// A UI sprite loaded with the wrap flag, repeating on both axes ([`sprite_image_tiled`]).
    pub fn sprite_texture_wrapped(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.wrapped_sprite(path, images, ReadMode::Defer)
    }

    /// [`Self::sprite_texture_wrapped`] for a one-shot warmer ([`Self::sprite_texture_blocking`]):
    /// the glue screens' tiled backdrops.
    pub fn sprite_texture_wrapped_blocking(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.wrapped_sprite(path, images, ReadMode::Block)
    }

    fn wrapped_sprite(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
        mode: ReadMode,
    ) -> Option<Handle<Image>> {
        self.sync_loose();
        let key = sprite_key(path);
        if let Some(cached) = self.cache.tiled.get(&key) {
            return cached;
        }
        let loaded = match decode_texture(&self.chain, &self.loose, path, LooseOpen::Index, mode) {
            SpriteLoad::Ready((w, h, rgba)) => Some(images.add(sprite_image_tiled(w, h, rgba))),
            // In flight: hand back the transparent placeholder and cache NOTHING, so the next
            // frame asks again and gets the real art the moment it lands.
            SpriteLoad::Pending => return Some(self.pending_sprite(images)),
            SpriteLoad::Missing => None,
        };
        let open = loose_open_for(path, LooseOpen::Index);
        self.cache.tiled.insert(key, open, loaded.clone());
        loaded
    }

    /// The stand-in a sprite cache hands back while its real art is still coming: 1x1, fully
    /// transparent. Not `None`: a UI quad with no texture draws the shared white image, an opaque
    /// white rectangle at an icon's rect. One image for the process, in the `generated` cache, so
    /// the handle is stable and the quad-list dirty check does not see a new texture every frame.
    fn pending_sprite(&mut self, images: &mut Assets<Image>) -> Handle<Image> {
        self.generated_sprite(PENDING_SPRITE_KEY, images, || (1, 1, vec![0, 0, 0, 0]))
    }

    /// [`Self::sprite_texture`] for a one-shot warmer, a caller that resolves its art once and
    /// keeps the handle, so it cannot take a placeholder ([`ReadMode::Block`]): the glue screens'
    /// art table, the V-plate boot set, and the loading-screen backdrop, all on a
    /// boot or loading frame.
    pub fn sprite_texture_blocking(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.sprite_texture_opened_blocking(path, LooseOpen::Index, images)
    }

    /// [`Self::sprite_texture_blocking`] with the open the caller's reference load makes
    /// ([`Self::sprite_texture_opened`]): the loading screen and the minimap's default art.
    pub fn sprite_texture_opened_blocking(
        &mut self,
        path: &str,
        open: LooseOpen,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.sync_loose();
        let key = sprite_key(path);
        if let Some(cached) = self.cache.sprites.get(&key) {
            return cached;
        }
        let loaded = match decode_texture(&self.chain, &self.loose, path, open, ReadMode::Block) {
            SpriteLoad::Ready((w, h, rgba)) => Some(images.add(sprite_image(w, h, rgba))),
            // `Block` never defers; the arm exists because the enum has three of them.
            SpriteLoad::Pending | SpriteLoad::Missing => None,
        };
        let open = loose_open_for(path, open);
        self.cache.sprites.insert(key, open, loaded.clone());
        loaded
    }

    /// [`Self::mask_texture`]'s blocking twin, for the minimap's boot art table and
    /// `SetMaskTexture`, which commits the requested path before it resolves.
    pub fn mask_texture_blocking(
        &mut self,
        path: &str,
        open: LooseOpen,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.sync_loose();
        let key = sprite_key(path);
        if let Some(cached) = self.cache.masks.get(&key) {
            return cached;
        }
        let loaded = match mask_rgba(&self.chain, &self.loose, path, open, ReadMode::Block) {
            SpriteLoad::Ready((w, h, rgba)) => Some(images.add(mask_image(w, h, rgba))),
            SpriteLoad::Pending | SpriteLoad::Missing => None,
        };
        let open = loose_open_for(path, open);
        self.cache.masks.insert(key, open, loaded.clone());
        loaded
    }

    /// A sprite whose pixels the client computes, built once by `make`: the colour picker's
    /// `<ColorWheelTexture>` and `<ColorValueTexture>`, which name no `file=`.
    pub fn generated_sprite(
        &mut self,
        key: &'static str,
        images: &mut Assets<Image>,
        make: impl FnOnce() -> (u32, u32, Vec<u8>),
    ) -> Handle<Image> {
        if let Some(cached) = self.generated.get(key) {
            return cached.clone();
        }
        let (w, h, rgba) = make();
        let handle = images.add(sprite_image(w, h, rgba));
        self.generated.insert(key, handle.clone());
        handle
    }

    /// A portrait, the circular mask baked in ([`portrait_image`]): `<path>.blp` as the sink
    /// `0x525d30` opens it ([`portrait_rgba`]). Keyed by the whole name, since `Foo` and `Foo.blp`
    /// open different files.
    pub fn portrait_texture(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.sync_loose();
        let key = normalize_path(path);
        if let Some(cached) = self.cache.portraits.get(&key) {
            return cached;
        }
        let loaded = match portrait_rgba(&self.chain, &self.loose, path, ReadMode::Defer) {
            SpriteLoad::Ready((w, h, rgba)) => Some(images.add(portrait_image(w, h, rgba))),
            // In flight: hand back the transparent placeholder and cache NOTHING, so the next
            // frame asks again and gets the real art the moment it lands.
            SpriteLoad::Pending => return Some(self.pending_sprite(images)),
            SpriteLoad::Missing => None,
        };
        let open = loose_open_for(path, LooseOpen::Index);
        self.cache.portraits.insert(key, open, loaded.clone());
        loaded
    }

    /// The tabard designer's emblem cell for the caller to tint: white carrying its own alpha,
    /// `(a << 24) | 0x00FFFFFF` per texel (the reference's `0x503431` loop). A file that fails to
    /// open gives `None`, where the reference keeps its static array's stale contents. Read from
    /// the archives alone, as the emblems' `0x5a7b40` opens with flag 0.
    pub fn emblem_mask_texture(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        let key = format!("emblem-mask:{}", sprite_key(path));
        if let Some(cached) = self.cache.sprites.get(&key) {
            return cached;
        }
        let open = LooseOpen::Archives;
        let loaded = match emblem_rgba(&self.chain, path, ReadMode::Defer) {
            SpriteLoad::Ready((w, h, mut rgba)) => {
                for px in rgba.as_chunks_mut::<4>().0 {
                    px[0] = 0xFF;
                    px[1] = 0xFF;
                    px[2] = 0xFF;
                }
                Some(images.add(sprite_image(w, h, rgba)))
            }
            // In flight: hand back the transparent placeholder and cache NOTHING, so the next
            // frame asks again and gets the real art the moment it lands. The caller is the
            // FrameXML extract, which re-resolves every texture per conversion.
            SpriteLoad::Pending => return Some(self.pending_sprite(images)),
            SpriteLoad::Missing => None,
        };
        self.cache.sprites.insert(key, open, loaded.clone());
        loaded
    }

    /// A coverage mask, its bytes handed to the shader 1:1 ([`mask_image`]), read by `open`: the
    /// minimap's circle.
    pub fn mask_texture(
        &mut self,
        path: &str,
        open: LooseOpen,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        self.sync_loose();
        let key = sprite_key(path);
        if let Some(cached) = self.cache.masks.get(&key) {
            return cached;
        }
        let loaded = match mask_rgba(&self.chain, &self.loose, path, open, ReadMode::Defer) {
            SpriteLoad::Ready((w, h, rgba)) => Some(images.add(mask_image(w, h, rgba))),
            // In flight: hand back the transparent placeholder and cache NOTHING, so the next
            // frame asks again and gets the real art the moment it lands.
            SpriteLoad::Pending => return Some(self.pending_sprite(images)),
            SpriteLoad::Missing => None,
        };
        let open = loose_open_for(path, open);
        self.cache.masks.insert(key, open, loaded.clone());
        loaded
    }

    /// A cursor picture's RGBA, a mode's (`Interface\Cursor\Point.blp`), the custom cursor's or a
    /// dragged icon's, opened as the cursor code's `0x449610` opens it ([`file_texture_rgba`]).
    pub fn cursor_rgba(&self, path: &str) -> Option<(u32, u32, Vec<u8>)> {
        file_texture_rgba(&self.chain, &self.loose, path)
    }

    /// Decode a cursor BLP ([`Self::cursor_rgba`]) for winit's custom OS cursor; macOS drives
    /// `NSCursor` from the raw RGBA instead.
    #[cfg(not(target_os = "macos"))]
    pub fn decode_cursor(
        &mut self,
        path: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        let (w, h, rgba) = self.cursor_rgba(path)?;
        Some(images.add(cursor_texture(w, h, rgba)))
    }

    /// A model submesh material, deduped by [`MaterialKey`] so submeshes sharing a texture share a
    /// handle and batch; a missing texture falls back to one shared untextured material.
    pub fn model_material(
        &mut self,
        texture: Option<&str>,
        blend: ModelBlend,
        two_sided: bool,
        alpha_cutoff: Option<f32>,
        fade_far: Option<f32>,
        is_wmo: bool,
        is_fade_variant: bool,
        // The batch's authored sampler address mode (`RenderSubmesh::wrap_x/wrap_y`).
        wrap: (bool, bool),
        images: &mut Assets<Image>,
        materials: &mut Assets<WowModelMaterial>,
    ) -> Handle<WowModelMaterial> {
        // Ground clutter passes its own cutoff (128/255); everything else takes the M2 alpha key.
        let cutoff = alpha_cutoff.unwrap_or(VANILLA_ALPHA_KEY_REF);
        // Ground-clutter fade: near is 0.75 of far, the client's ratio.
        let clutter_fade = match fade_far {
            Some(far) => Vec4::new(far * 0.75, far, 0.0, 1.0),
            None => Vec4::ZERO,
        };
        let resolved = texture.and_then(|t| {
            self.texture(t, wrap, images)
                .map(|h| (normalize_path(t), h))
        });
        // Only AlphaTest reads the cutoff, so opaque and blend materials do not split on it.
        let key_alpha = if blend == ModelBlend::AlphaTest {
            (cutoff * 255.0).round() as u8
        } else {
            0
        };
        let key_fade = fade_far.map(|f| f.round() as u16).unwrap_or(0);
        let key = match &resolved {
            Some((path, _)) => MaterialKey::Textured(
                path.clone(),
                blend,
                two_sided,
                key_alpha,
                key_fade,
                is_wmo,
                is_fade_variant,
                wrap.0,
                wrap.1,
            ),
            None => MaterialKey::Fallback(is_wmo),
        };
        if let Some(handle) = self.model_materials.fetch(&key) {
            return handle;
        }
        // This builder serves ground clutter, which never authors Mod/Mod2x; every Mod-capable
        // consumer uses `model_render::model_material`.
        let alpha_mode = match blend {
            ModelBlend::Opaque => AlphaMode::Opaque,
            ModelBlend::AlphaTest => AlphaMode::Mask(cutoff),
            ModelBlend::Blend | ModelBlend::Mod | ModelBlend::Mod2x => AlphaMode::Blend,
        };
        // Single-sided unless the M2's 0x04 flag is set, as in the reference.
        let cull_mode = if two_sided { None } else { Some(Face::Back) };
        let base = match resolved {
            Some((_, image)) => StandardMaterial {
                base_color_texture: Some(image),
                alpha_mode,
                double_sided: two_sided,
                cull_mode,
                ..default()
            },
            // No texture: an untextured white draw, the reference's disabled stage (`0x59d020`).
            None => StandardMaterial {
                base_color: Color::WHITE,
                ..default()
            },
        };
        let handle = materials.add(WowModelMaterial {
            base,
            extension: WowModelExt {
                modern_shadow: crate::materials::MODERN_SHADOW_MAP, // fork: modern lighting (FORK.md, P2)
                modern_cascades: crate::materials::MODERN_SHADOW_CASCADES,
                modern_fog: crate::materials::MODERN_FOG_VOLUME, // fork: modern lighting (FORK.md, P4)
                clutter_fade,
                // x = WMO (FFP N·L × MOCV), y = the fade blend twin (no caller passes it here).
                model_flags: Vec4::new(
                    if is_wmo { 1.0 } else { 0.0 },
                    if is_fade_variant { 1.0 } else { 0.0 },
                    0.0,
                    0.0,
                ),
                // Clutter lights on the FFP N·L path with its own MCSH tint; no selector.
                sun_scale: Vec4::new(1.0, 0.0, 0.0, 0.0),
                tint: Vec4::ONE, // clutter has no animated M2Color tint and is not a WMO batch
                sidn: Vec4::ZERO, // clutter is never SIDN/WINDOW glass (WMO-only)
                anim_slots: Vec4::ZERO,
                stage1: Vec4::ZERO, // a tuft is one texture
                stage1_texture: None,
                light_buf: self.shared_light.clone(),
            },
        });
        self.model_materials.insert(key, handle.clone());
        handle
    }
}

/// Render configuration read once from the environment at startup, inserted beside
/// [`WorldAssets`]; its presence is the "there is a client install" gate world setups key on.
/// Terrain residency is not here: it derives from the live `farclip`, as in the reference.
#[derive(Resource, Clone, Copy)]
pub struct RenderConfig {
    /// Stale tiles released per frame on a within-map window shift (`$WOW_TILE_UNLOAD`, default 1;
    /// `0` releases the whole trailing row in one frame, the unbudgeted A/B leg).
    pub unload_budget: usize,
}

/// Startup ordering: the patch chain opens in [`AssetSet::Open`], so other startup systems run
/// after it and borrow [`WorldAssets`]; the catalogs are built from it in [`AssetSet::Catalogs`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetSet {
    Open,
    /// The one run of [`DataLoad`] at startup, right after [`AssetSet::Open`].
    Catalogs,
}

/// **The schedule every catalog loader runs in**: once at startup, in [`AssetSet::Catalogs`], and
/// again whenever the data the chain answers from changes — the Turtle WoW 1.18.1 layer mounted or
/// unmounted (`benilla_formats::layers::data_generation`) — so every catalog is rebuilt from the
/// chain it will be read against. A loader here replaces what it inserts and spawns nothing, which
/// is what makes a second run safe.
#[derive(bevy::ecs::schedule::ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataLoad;

/// **The data the chain answers from changed** — the Turtle WoW 1.18.1 layer mounted or unmounted —
/// and [`DataLoad`] has already rerun. Triggered once per change, never per frame: a cache of
/// chain-derived data that outlives the change registers an observer that forgets it, and the
/// next use rebuilds it from the new data.
#[derive(bevy::ecs::event::Event, Debug, Clone, Copy)]
pub struct ChainDataChanged;

/// Marks an `App` whose [`DataLoad`] schedule and startup run are installed.
#[derive(Resource)]
struct DataLoadInstalled;

/// Run [`DataLoad`]: at startup and on every data change. Its commands apply when the run ends,
/// so a caller sees every catalog the run built.
pub fn run_data_load(world: &mut World) {
    world.try_run_schedule(DataLoad).ok();
}

/// Registering a catalog loader.
pub trait DataLoadAppExt {
    /// Add `systems` to [`DataLoad`], installing the schedule and its startup run the first time
    /// any plugin does, so every `App` that has a loader also runs it — a test `App` with one
    /// plugin included.
    fn add_data_loaders<M>(
        &mut self,
        systems: impl bevy::ecs::schedule::IntoScheduleConfigs<bevy::ecs::system::ScheduleSystem, M>,
    ) -> &mut Self;
}

impl DataLoadAppExt for App {
    fn add_data_loaders<M>(
        &mut self,
        systems: impl bevy::ecs::schedule::IntoScheduleConfigs<bevy::ecs::system::ScheduleSystem, M>,
    ) -> &mut Self {
        if !self.world().contains_resource::<DataLoadInstalled>() {
            self.insert_resource(DataLoadInstalled)
                .init_schedule(DataLoad)
                .configure_sets(Startup, AssetSet::Catalogs.after(AssetSet::Open))
                .add_systems(Startup, run_data_load.in_set(AssetSet::Catalogs));
        }
        self.add_systems(DataLoad, systems)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loose_addon_file_maps_addon_paths_case_insensitively() {
        let root =
            std::env::temp_dir().join(format!("benilla-loose-sprite-test-{}", std::process::id()));
        let dir = root.join("Atlas").join("Images").join("Maps");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("BlackrockDepths.blp"), b"x").unwrap();

        // The returned spelling depends on the filesystem's case sensitivity, so check the file.
        let hit = loose_addon_file(
            &root,
            "interface\\addons\\atlas\\images\\maps\\blackrockdepths.blp",
        )
        .expect("mixed-case tree resolves a lowercase candidate");
        assert_eq!(std::fs::read(&hit).unwrap(), b"x");

        assert_eq!(loose_addon_file(&root, "interface\\icons\\foo.blp"), None);
        // A directory is not a file, and a missing file is a miss, not an error.
        assert_eq!(loose_addon_file(&root, "interface\\addons\\atlas"), None);
        assert_eq!(
            loose_addon_file(&root, "interface\\addons\\atlas\\images\\maps\\nope.blp"),
            None
        );
        // Dot-components never reach the filesystem; `normalize_path` already leaves no `/`.
        assert_eq!(
            loose_addon_file(&root, "interface\\addons\\..\\..\\etc\\passwd"),
            None
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// Against the real chain: a client file comes from the archive, an addon's own file from
    /// disk, and a loose copy of a path the archive also holds shadows it, as `0x647e60` asks the
    /// install tree (`0x647f89`) before the chain (`0x648116`), even when it will not read.
    #[test]
    fn read_loose_or_chain_asks_an_addons_own_file_on_disk_then_the_chain() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let root =
            std::env::temp_dir().join(format!("benilla-loose-audio-test-{}", std::process::id()));
        let addons = root.join("AddOns");
        std::fs::create_dir_all(addons.join("Jukebox")).unwrap();
        std::fs::write(addons.join("Jukebox").join("Track.mp3"), b"ID3!").unwrap();
        let folder = LooseFiles {
            addons: Some(addons.clone()),
            install: None,
        };
        let none = LooseFiles::default();
        let open = LooseOpen::Index;

        // The archive leg: a client track the chain does carry.
        let theme = read_loose_or_chain(
            &chain,
            &folder,
            "Sound\\Music\\GlueScreenMusic\\wow_main_theme.mp3",
            open,
        )
        .expect("the glue theme lives in sound.MPQ");
        assert!(theme.len() > 1_000_000, "the theme is a ~3 MB stored MP3");

        // The loose leg: an addon's own track, named by the virtual path, spelt in its own case.
        assert_eq!(
            read_loose_or_chain(
                &chain,
                &folder,
                "Interface\\AddOns\\Jukebox\\Track.mp3",
                open
            )
            .as_deref(),
            Some(&b"ID3!"[..])
        );

        // The shadow: a stock addon's file the archive carries, with a loose copy on disk.
        let stock = "Interface\\AddOns\\Blizzard_AuctionUI\\Blizzard_AuctionUI.lua";
        let archived = chain
            .lock_recover()
            .read(stock)
            .expect("interface.MPQ carries it");
        assert_eq!(
            read_loose_or_chain(&chain, &none, stock, open).as_deref(),
            Some(&archived[..]),
            "with no folder the archive answers"
        );
        std::fs::create_dir_all(addons.join("Blizzard_AuctionUI")).unwrap();
        std::fs::write(
            addons
                .join("Blizzard_AuctionUI")
                .join("Blizzard_AuctionUI.lua"),
            b"-- loose",
        )
        .unwrap();
        assert_eq!(
            read_loose_or_chain(&chain, &folder, stock, open).as_deref(),
            Some(&b"-- loose"[..]),
            "the loose copy shadows the archive's"
        );
        // An open with flag 0 never sees it.
        assert_eq!(
            read_loose_or_chain(&chain, &folder, stock, LooseOpen::Archives).as_deref(),
            Some(&archived[..]),
            "a flag-0 open reads the archive"
        );
        // A loose file that will not read is a miss: the index answered, so the archive is not
        // asked (`0x647865`).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let loose = addons
                .join("Blizzard_AuctionUI")
                .join("Blizzard_AuctionUI.lua");
            std::fs::set_permissions(&loose, std::fs::Permissions::from_mode(0o000)).unwrap();
            let unreadable = std::fs::read(&loose).is_err();
            let got = read_loose_or_chain(&chain, &folder, stock, open);
            std::fs::set_permissions(&loose, std::fs::Permissions::from_mode(0o644)).unwrap();
            if unreadable {
                assert_eq!(got, None, "no fallback to the archive copy");
            }
        }

        // The folder holding its own `.toc` is moved aside: the archive's copy answers again.
        std::fs::write(
            addons
                .join("Blizzard_AuctionUI")
                .join("Blizzard_AuctionUI.toc"),
            b"## Interface: 11200",
        )
        .unwrap();
        assert_eq!(
            read_loose_or_chain(&chain, &folder, stock, open).as_deref(),
            Some(&archived[..]),
            "a moved-aside folder answers nothing"
        );

        // In neither store: a plain miss, which every caller renders as silence.
        assert!(read_loose_or_chain(&chain, &folder, "Sound\\Music\\nope.mp3", open).is_none());
        // No folder configured (a capture run, where `local_state` is hermetic): chain only.
        assert!(
            read_loose_or_chain(&chain, &none, "Interface\\AddOns\\Jukebox\\Track.mp3", open)
                .is_none()
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// A fresh folder under the temp dir, removed when dropped.
    struct Tree(PathBuf);

    impl Tree {
        fn new(tag: &str) -> Tree {
            let dir = std::env::temp_dir().join(format!(
                "benilla-loose-files-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Tree(dir)
        }

        fn file(&self, rel: &str, body: &str) {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, body).unwrap();
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn body(file: Option<LooseFile>) -> Option<String> {
        file.map(|f| String::from_utf8(f.read().unwrap()).unwrap())
    }

    /// A flagged open finds an install's loose file by its path below the install or `Data\`, a
    /// font's by its bare name first; an `Interface\AddOns\` path reads benilla's AddOns folder
    /// alone; an open with flag 0, or a name written with `/`, finds nothing loose.
    #[test]
    fn loose_files_answer_the_flagged_opens_from_the_install_and_the_addons_folder() {
        let install = Tree::new("install");
        install.file("Fonts/FRIZQT__.TTF", "fonts dir");
        install.file("Data/FRIZQT__.TTF", "data root");
        install.file("Data/Interface/Glues/Common/Glues-WoW-Logo.blp", "logo");
        install.file("Interface/AddOns/Atlas/Atlas.toc", "install addon");
        let addons = Tree::new("addons");
        addons.file("Atlas/Atlas.toc", "benilla addon");
        let root = install.0.clone();
        let cell = LooseIndexCell::new(move || LooseIndex::build(Some(&root), &root.join("Data")));
        let loose = LooseFiles {
            addons: Some(addons.0.clone()),
            install: Some(Arc::new(cell)),
        };
        let logo = "Interface\\Glues\\Common\\Glues-WoW-Logo.blp";
        assert_eq!(
            body(loose.find(logo, LooseOpen::Index)).as_deref(),
            Some("logo")
        );
        assert_eq!(loose.find(logo, LooseOpen::Archives), None);
        let font = "Fonts\\FRIZQT__.TTF";
        assert_eq!(
            body(loose.find(font, LooseOpen::Index)).as_deref(),
            Some("fonts dir")
        );
        assert_eq!(
            body(loose.find(font, LooseOpen::BareNameFirst)).as_deref(),
            Some("data root")
        );
        assert_eq!(
            body(loose.find("Interface\\AddOns\\atlas\\ATLAS.toc", LooseOpen::Index)).as_deref(),
            Some("benilla addon")
        );
        assert_eq!(
            loose.find(
                "Interface/Glues/Common/Glues-WoW-Logo.blp",
                LooseOpen::Index
            ),
            None
        );
        assert_eq!(
            loose.find("Fonts/FRIZQT__.TTF", LooseOpen::BareNameFirst),
            None
        );
        assert_eq!(
            loose.find("Interface/AddOns/Atlas/Atlas.toc", LooseOpen::Index),
            None
        );
        let no_folder = LooseFiles {
            addons: None,
            install: loose.install.clone(),
        };
        assert_eq!(
            no_folder.find("Interface\\AddOns\\Atlas\\Atlas.toc", LooseOpen::Index),
            None,
            "the install's AddOns folder is never read"
        );
    }

    /// `SetTexture`'s probe and decode find a texture loose in the install by its name as written
    /// with `\`, and not when it is written with `/`, which the index's compare does not fold.
    #[test]
    fn a_loose_texture_answers_its_backslash_name_and_not_its_slash_name() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let bytes = chain
            .lock_recover()
            .read("Interface\\Icons\\INV_Misc_QuestionMark.blp")
            .expect("the question-mark icon is in the chain");
        let install = Tree::new("slash");
        let path = install.0.join("Interface/Glues/Loose/Custom.blp");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let root = install.0.clone();
        let cell = LooseIndexCell::new(move || LooseIndex::build(Some(&root), &root.join("Data")));
        let loose = LooseFiles {
            addons: None,
            install: Some(Arc::new(cell)),
        };
        let back = "Interface\\Glues\\Loose\\Custom";
        let slash = "Interface/Glues/Loose/Custom";
        assert!(sprite_found(&chain, &loose, back));
        assert!(matches!(
            sprite_dimensions(&chain, &loose, back),
            SpriteDims::Ready(..)
        ));
        assert!(!sprite_found(&chain, &loose, slash));
        assert!(matches!(
            sprite_dimensions(&chain, &loose, slash),
            SpriteDims::Missing
        ));
    }

    /// An emptied index drops every sprite opened with the loose flag, hit or miss, and keeps those
    /// opened with flag 0; the same generation again drops nothing.
    #[test]
    fn a_new_index_generation_drops_the_loose_opened_sprites() {
        let mut c = SpriteCaches::default();
        let hit = Some(Handle::<Image>::default());
        c.sprites.insert("ui".into(), LooseOpen::Index, hit.clone());
        c.sprites.insert("miss".into(), LooseOpen::Index, None);
        c.sprites
            .insert("sun".into(), LooseOpen::Archives, hit.clone());
        c.tiled.insert("ui".into(), LooseOpen::Index, hit.clone());
        c.portraits
            .insert("p".into(), LooseOpen::Index, hit.clone());
        c.masks
            .insert("mask".into(), LooseOpen::Archives, hit.clone());
        c.resampled.insert(("ui".into(), 4, 4), hit.clone());
        c.sync(Some(0));
        assert!(
            c.sprites.get(&"ui".to_string()).is_some(),
            "an unchanged generation keeps all"
        );
        c.sync(Some(1));
        assert!(c.sprites.get(&"ui".to_string()).is_none());
        assert!(c.sprites.get(&"miss".to_string()).is_none());
        assert!(c.sprites.get(&"sun".to_string()).is_some());
        assert!(c.tiled.get(&"ui".to_string()).is_none());
        assert!(c.portraits.get(&"p".to_string()).is_none());
        assert!(c.masks.get(&"mask".to_string()).is_some());
        assert!(c.resampled.is_empty());
    }

    /// A `Blizzard*` folder holding its own `.toc`, any case, is moved aside and answers no path;
    /// one without it, or a folder of another name, answers as before.
    #[test]
    fn a_blizzard_folder_with_its_toc_is_moved_aside() {
        let addons = Tree::new("aside");
        addons.file("Blizzard_Foo/Blizzard_Foo.toc", "toc");
        addons.file("Blizzard_Foo/Foo.lua", "foo");
        addons.file("BLIZZARD_Bar/blizzard_bar.TOC", "toc");
        addons.file("BLIZZARD_Bar/Bar.lua", "bar");
        addons.file("Blizzard_AuctionUI/Blizzard_AuctionUI.pub", "pub");
        addons.file("Blizzard_AuctionUI/Templates.xml", "templates");
        addons.file("MyBlizzard/MyBlizzard.toc", "toc");
        addons.file("MyBlizzard/Mine.lua", "mine");
        let root = &addons.0;
        assert!(moved_aside(&root.join("Blizzard_Foo"), "Blizzard_Foo"));
        assert!(moved_aside(&root.join("BLIZZARD_Bar"), "BLIZZARD_Bar"));
        assert!(!moved_aside(
            &root.join("Blizzard_AuctionUI"),
            "Blizzard_AuctionUI"
        ));
        assert!(!moved_aside(&root.join("MyBlizzard"), "MyBlizzard"));
        let get = |p: &str| body(loose_addon_file(root, &normalize_path(p)).map(LooseFile::Plain));
        assert_eq!(get("Interface\\AddOns\\Blizzard_Foo\\Foo.lua"), None);
        assert_eq!(get("Interface\\AddOns\\blizzard_bar\\Bar.lua"), None);
        assert_eq!(
            get("Interface\\AddOns\\Blizzard_AuctionUI\\Templates.xml").as_deref(),
            Some("templates")
        );
        assert_eq!(
            get("Interface\\AddOns\\MyBlizzard\\Mine.lua").as_deref(),
            Some("mine")
        );
    }
    /// A TGA header of `w` by `h` at 32 bits with no alpha bits and `data` after it. `0x5a37e0`
    /// refuses the pixels, and `0x44a140` still makes the texture.
    fn refused_tga(w: u16, h: u16, data: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; 18];
        out[2] = 2;
        out[12..14].copy_from_slice(&w.to_le_bytes());
        out[14..16].copy_from_slice(&h.to_le_bytes());
        out[16] = 32;
        out[17] = 0x20;
        out.extend_from_slice(data);
        out
    }

    /// `0x44a140` returns the texture once the header opens, and the refused decode in `0x44a260`
    /// fills it opaque white (`0x44a2eb` into `0x448920`) at the size `0x448450` made it. A header
    /// that makes no texture fails the candidate as a missing file does, whether the pixels would
    /// decode or not, and so does any refused load that is not `TextureCreate`'s.
    #[test]
    fn a_refused_tga_texture_is_opaque_white_at_its_headers_size() {
        let tga = "interface\\addons\\foo\\icon.tga";
        let white =
            |candidate, bytes: &[u8]| decode_candidate(candidate, bytes, OnRefusal::WhiteDefault);
        let bytes = refused_tga(16, 8, &[0; 512]);
        assert!(tga_to_rgba(&bytes).is_err(), "the decoder refuses it");
        assert_eq!(
            white(tga, &bytes),
            Candidate::Pixels(16, 8, vec![0xff; 16 * 8 * 4])
        );
        // The pixels need not be there at all.
        assert_eq!(
            white(tga, &refused_tga(8, 512, &[])),
            Candidate::Pixels(8, 512, vec![0xff; 8 * 512 * 4])
        );

        // Next candidate: not a size `0x448450` makes (decoded or not), a side below the 8
        // `GxTexCreate` takes (`0x58ab40`), a header that does not open.
        let mut wide = refused_tga(24, 8, &[9; 24 * 8 * 4]);
        wide[17] = 0x28;
        assert!(
            tga_to_rgba(&wide).is_ok(),
            "it decodes, and still fails the gate"
        );
        for bytes in [
            refused_tga(24, 8, &[0; 24]),
            refused_tga(2048, 8, &[0; 24]),
            refused_tga(4, 8, &[0; 128]),
            refused_tga(0, 8, &[]),
            wide,
            vec![2, 0, 2],
        ] {
            assert_eq!(white(tga, &bytes), Candidate::Next);
        }
        // A load that is no `TextureCreate`'s leaves them all to the next candidate.
        for bytes in [&bytes, &refused_tga(24, 8, &[0; 24])] {
            assert_eq!(
                decode_candidate(tga, bytes, OnRefusal::Miss),
                Candidate::Next
            );
        }

        // A TGA that decodes is its pixels, white or not.
        let mut ok = refused_tga(8, 8, &[1, 2, 3, 77].repeat(64));
        ok[17] = 0x28;
        assert_eq!(
            white(tga, &ok),
            Candidate::Pixels(8, 8, [3, 2, 1, 77].repeat(64))
        );
    }

    /// The addon corpus' three `BigWigs` ready-check icons are 32-bit TGAs with no alpha bits:
    /// `0x44a140` returns each as a texture of its header's size once the header opens, and the
    /// refused decode leaves it white (`0x44a2eb`).
    /// [`sprite_dimensions`] as a size: a native chain never answers `Pending`.
    fn dims(chain: &Mutex<Chain>, loose: &LooseFiles, path: &str) -> Option<(u32, u32)> {
        match sprite_dimensions(chain, loose, path) {
            SpriteDims::Ready(w, h) => Some((w, h)),
            SpriteDims::Missing => None,
            SpriteDims::Pending => panic!("a native chain never defers"),
        }
    }

    /// A load's pixels, `None` for a miss: a native chain never answers `Pending`.
    fn loaded(load: SpriteLoad) -> Option<(u32, u32, Vec<u8>)> {
        match load {
            SpriteLoad::Ready(decoded) => Some(decoded),
            SpriteLoad::Missing => None,
            SpriteLoad::Pending => panic!("a native chain never defers"),
        }
    }

    /// [`decode_texture`]'s pixels, by the interface open.
    fn texture(
        chain: &Mutex<Chain>,
        loose: &LooseFiles,
        path: &str,
    ) -> Option<(u32, u32, Vec<u8>)> {
        loaded(decode_texture(
            chain,
            loose,
            path,
            LooseOpen::Index,
            ReadMode::Block,
        ))
    }

    /// [`decode_sprite`]'s pixels, by the interface open.
    fn sprite(chain: &Mutex<Chain>, loose: &LooseFiles, path: &str) -> Option<(u32, u32, Vec<u8>)> {
        loaded(decode_sprite(
            chain,
            loose,
            path,
            LooseOpen::Index,
            ReadMode::Block,
        ))
    }

    #[test]
    fn the_corpus_ready_check_icons_are_textures_of_their_size() {
        let data = benilla_formats::wow_data_or_skip!();
        let corpus = benilla_formats::addon_corpus_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let loose = LooseFiles {
            addons: Some(corpus),
            install: None,
        };
        for name in ["Ready", "Waiting", "NotReady"] {
            let path = format!("Interface\\AddOns\\BigWigs\\Icons\\ReadyCheck-{name}");
            assert!(sprite_found(&chain, &loose, &path), "{name}");
            assert_eq!(dims(&chain, &loose, &path), Some((64, 64)), "{name}");
            let (_, _, rgba) = texture(&chain, &loose, &path).unwrap();
            assert!(rgba.iter().all(|&b| b == 0xff), "{name}: opaque white");
        }
    }

    /// Through the store: a refused `.tga` in the AddOns folder is a texture of its header's size
    /// for `SetTexture`'s loads. A header that makes no texture (`0x448450` takes powers of two
    /// below 1024, `GxTexCreate` no side below 8) is a miss, its pixels decoded or not.
    #[test]
    fn a_refused_tga_in_the_addons_folder_is_a_texture_of_its_size() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let addons = Tree::new("white");
        let dir = addons.0.join("Foo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Icon.tga"), refused_tga(16, 8, &[0; 512])).unwrap();
        std::fs::write(dir.join("Odd.tga"), refused_tga(6, 8, &[0; 192])).unwrap();
        let loose = LooseFiles {
            addons: Some(addons.0.clone()),
            install: None,
        };
        let icon = "Interface\\AddOns\\Foo\\Icon";
        assert!(sprite_found(&chain, &loose, icon));
        assert_eq!(dims(&chain, &loose, icon), Some((16, 8)));
        let (w, h, rgba) = texture(&chain, &loose, icon).unwrap();
        assert_eq!((w, h, rgba.len()), (16, 8, 16 * 8 * 4));
        assert!(rgba.iter().all(|&b| b == 0xff));
        // A load that is not `TextureCreate`'s still gets no pixels.
        assert!(sprite(&chain, &loose, icon).is_none());
        // Not a size `0x448450` makes (`0x4484ae`-`0x4484d0`): `TextureCreate` fails the candidate
        // whether its pixels decode or not, and `SetTexture` answers nil.
        std::fs::write(dir.join("Wide.tga"), decodable_tga(24, 8)).unwrap();
        std::fs::write(dir.join("Small.tga"), decodable_tga(4, 8)).unwrap();
        std::fs::write(dir.join("Zero.tga"), refused_tga(0, 8, &[])).unwrap();
        std::fs::write(dir.join("Fine.tga"), decodable_tga(8, 16)).unwrap();
        for (name, found, size) in [
            ("Odd", false, None),
            ("Wide", false, None),
            ("Small", false, None),
            ("Zero", false, None),
            ("Fine", true, Some((8, 16))),
        ] {
            let path = format!("Interface\\AddOns\\Foo\\{name}");
            assert_eq!(sprite_found(&chain, &loose, &path), found, "{name}");
            assert_eq!(dims(&chain, &loose, &path), size, "{name}");
        }
    }

    /// A palette BLP2 (compression 1, no alpha bits, no mips) of `w` by `h`, every texel palette
    /// entry 0, red `r`.
    fn tiny_blp(w: u32, h: u32, r: u8) -> Vec<u8> {
        let mut b = vec![0u8; 148 + 1024];
        b[0..4].copy_from_slice(b"BLP2");
        b[4..8].copy_from_slice(&1u32.to_le_bytes());
        b[8] = 1;
        b[12..16].copy_from_slice(&w.to_le_bytes());
        b[16..20].copy_from_slice(&h.to_le_bytes());
        b[20..24].copy_from_slice(&(148u32 + 1024).to_le_bytes());
        b[84..88].copy_from_slice(&(w * h).to_le_bytes());
        b[148 + 2] = r;
        b.extend(vec![0u8; (w * h) as usize]);
        b
    }

    /// `TextureCreate`'s solid texture, `0xff00ff00` (`0x447c00`), 8 by 8 (`0x44a900`).
    fn green() -> Candidate {
        Candidate::Pixels(8, 8, [0, 0xff, 0, 0xff].repeat(64))
    }

    /// `TextureCreate` takes the loader by the candidate's extension (`0x449e16`-`0x449e3d`), never
    /// by the bytes. BLP bytes under a `.tga` name reach the TGA loader `0x44a140`, whose header
    /// is the BLP's: its height word is the BLP width's high half, 0, which `GxTexCreate` refuses
    /// (`0x58ab4c`), so the next candidate is tried. Any bytes under a `.blp` name open as the BLP
    /// loader's texture (`0x44a399`), and a refused parse (the magic, `0x5a83dc`) or decode leaves
    /// the solid green (`0x44a50e`), with no next candidate.
    #[test]
    fn the_loader_is_the_candidates_extension_not_its_content() {
        let tga = "interface\\addons\\foo\\icon.tga";
        let blp = "interface\\addons\\foo\\icon.blp";
        let white =
            |candidate, bytes: &[u8]| decode_candidate(candidate, bytes, OnRefusal::WhiteDefault);
        let good_blp = tiny_blp(16, 8, 7);
        let good_tga = decodable_tga(8, 8);
        assert!(matches!(white(blp, &good_blp), Candidate::Pixels(16, 8, _)));
        assert!(matches!(white(tga, &good_tga), Candidate::Pixels(8, 8, _)));

        assert_eq!(
            white(tga, &good_blp),
            Candidate::Next,
            "a BLP under a .tga name"
        );
        assert_eq!(white(blp, &good_tga), green(), "a TGA under a .blp name");
        assert_eq!(white(blp, b"BLP2"), green(), "a BLP that will not parse");
        assert_eq!(white(blp, &[]), green(), "an empty .blp");

        // A load that is not `TextureCreate`'s takes the same decoder, and a refusal is a miss.
        assert_eq!(
            decode_candidate(tga, &good_blp, OnRefusal::Miss),
            Candidate::Next
        );
        assert_eq!(
            decode_candidate(blp, &good_tga, OnRefusal::Miss),
            Candidate::Next
        );
    }

    /// Through the store: a `.tga` name holding BLP bytes is passed over for its `.blp`
    /// candidate (`0x449e4b`), and a `.blp` that will not decode is green, whatever `.tga` lies
    /// beside it.
    #[test]
    fn a_mislabelled_addon_file_does_not_draw() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let addons = Tree::new("mislabel");
        let dir = addons.0.join("Foo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Art.tga"), tiny_blp(16, 16, 1)).unwrap();
        std::fs::write(dir.join("Art.blp"), tiny_blp(8, 8, 2)).unwrap();
        std::fs::write(dir.join("Pic.blp"), decodable_tga(8, 8)).unwrap();
        std::fs::write(dir.join("Pic.tga"), decodable_tga(16, 16)).unwrap();
        std::fs::write(dir.join("Only.tga"), tiny_blp(16, 16, 1)).unwrap();
        let loose = LooseFiles {
            addons: Some(addons.0.clone()),
            install: None,
        };
        let load = |name: &str| texture(&chain, &loose, &format!("Interface\\AddOns\\Foo\\{name}"));
        let (w, h, rgba) = load("Art.tga").unwrap();
        assert_eq!((w, h, rgba[0]), (8, 8, 2), "the .blp beside it draws");
        let (w, h, rgba) = load("Pic").unwrap();
        assert_eq!((w, h, &rgba[..4]), (8, 8, &[0, 0xff, 0, 0xff][..]), "green");
        assert!(load("Only.tga").is_none(), "no candidate makes a texture");
        assert!(!sprite_found(
            &chain,
            &loose,
            "Interface\\AddOns\\Foo\\Only.tga"
        ));
    }

    /// The minimap's mask loads through `TextureCreate` (`Minimap:SetMaskTexture` at `0x4ee5e4`,
    /// the world-entry load at `0x4ed862`): a `.tga` whose header opens and whose pixels the
    /// decoder refuses is opaque white at its header's size (`0x44a2eb` into `0x448920`), not a
    /// miss the minimap would paint green.
    #[test]
    fn a_refused_tga_mask_is_opaque_white() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let addons = Tree::new("mask");
        let dir = addons.0.join("Foo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Mask.tga"), refused_tga(16, 8, &[0; 512])).unwrap();
        let loose = LooseFiles {
            addons: Some(addons.0.clone()),
            install: None,
        };
        let mask = |name: &str| {
            loaded(mask_rgba(
                &chain,
                &loose,
                &format!("Interface\\AddOns\\Foo\\{name}"),
                LooseOpen::Index,
                ReadMode::Block,
            ))
        };
        let (w, h, rgba) = mask("Mask").unwrap();
        assert_eq!((w, h), (16, 8));
        assert!(rgba.iter().all(|&b| b == 0xff));
        assert!(
            mask("None").is_none(),
            "a name that resolves nowhere stays the minimap's green"
        );
    }

    /// The portrait sink formats `%s.blp` onto the name as written (`0x525e0f`, format
    /// `0x8540b4`) and reads that file with the BLP reader (`0x5a7b40` at `0x525e37`, flag 1): a
    /// `.tga` is never read, and a name ending `.tga` opens `<name>.tga.blp`.
    #[test]
    fn a_portrait_opens_its_name_with_blp_appended() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let addons = Tree::new("portrait");
        let dir = addons.0.join("Foo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Icon.tga"), decodable_tga(64, 64)).unwrap();
        std::fs::write(dir.join("Bag.blp"), tiny_blp(64, 64, 3)).unwrap();
        std::fs::write(dir.join("Odd.tga.blp"), tiny_blp(64, 64, 4)).unwrap();
        std::fs::write(dir.join("Odd.blp"), tiny_blp(64, 64, 5)).unwrap();
        let loose = LooseFiles {
            addons: Some(addons.0.clone()),
            install: None,
        };
        let red = |name: &str| {
            let path = format!("Interface\\AddOns\\Foo\\{name}");
            loaded(portrait_rgba(&chain, &loose, &path, ReadMode::Block))
                .map(|(_, _, rgba)| rgba[0])
        };
        assert_eq!(red("Icon"), None, "no Icon.blp, and the .tga is not read");
        assert_eq!(red("Icon.tga"), None, "Icon.tga.blp");
        assert_eq!(red("Bag"), Some(3));
        assert_eq!(red("Bag.blp"), None, "Bag.blp.blp");
        assert_eq!(red("Odd.tga"), Some(4), "Odd.tga.blp, not Odd.blp");
    }

    /// The cursor code opens every picture through `0x449610`, `SetCursor`'s (`0x523c86`) and
    /// the dragged icon's (`0x5239f7`, `0x523a3e`, `0x523af7`, `0x523bb1`) alike: it strips a
    /// three-letter extension (`0x449590` at `0x44963c`), appends `.blp` (`0x449641`) and decodes
    /// the bytes as a BLP (`0x449840`), so `Foo.tga` opens `Foo.blp` and no `.tga` is read.
    #[test]
    fn a_cursor_picture_opens_only_its_blp() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let addons = Tree::new("cursorpic");
        let dir = addons.0.join("Foo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Arrow.tga"), decodable_tga(32, 32)).unwrap();
        std::fs::write(dir.join("Both.tga"), decodable_tga(16, 16)).unwrap();
        std::fs::write(dir.join("Both.blp"), tiny_blp(32, 32, 6)).unwrap();
        std::fs::write(dir.join("Fake.blp"), decodable_tga(32, 32)).unwrap();
        let loose = LooseFiles {
            addons: Some(addons.0.clone()),
            install: None,
        };
        let pic = |name: &str| {
            file_texture_rgba(&chain, &loose, &format!("Interface\\AddOns\\Foo\\{name}"))
                .map(|(w, h, rgba)| (w, h, rgba[0]))
        };
        assert_eq!(pic("Arrow"), None);
        assert_eq!(pic("Arrow.tga"), None);
        assert_eq!(pic("Both.tga"), Some((32, 32, 6)), "Both.blp, not Both.tga");
        assert_eq!(pic("Both"), Some((32, 32, 6)));
        assert_eq!(pic("Fake"), None, "TGA bytes under a .blp name");
    }

    /// The emblem cell appends `.BLP` to its name (`0x50338a`, `0x83574c`) and reads that one file
    /// with the BLP reader `0x5a7b40` and flag 0: no `.tga` of the same stem is read.
    #[test]
    fn an_emblem_cell_reads_only_its_blp() {
        let stem = "Textures\\GuildEmblems\\Emblem_01_02_TU_U";
        assert_eq!(
            emblem_names(stem),
            vec![format!("{}.blp", normalize_path(stem))]
        );
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let (w, h, _) =
            loaded(emblem_rgba(&chain, stem, ReadMode::Block)).expect("the emblem BLP decodes");
        assert!(w > 0 && h > 0);
    }

    /// A 32-bit TGA with 8 alpha bits, `w` by `h`, that the decoder reads.
    fn decodable_tga(w: u16, h: u16) -> Vec<u8> {
        let mut out = refused_tga(w, h, &vec![9; usize::from(w) * usize::from(h) * 4]);
        out[17] = 0x28;
        out
    }

    /// `SetTexture` runs per frame in the addons that set an icon every `OnUpdate`: a TGA's header
    /// is read once for a resolved file, and a read that fails is asked again.
    #[test]
    fn the_size_gate_reads_a_resolved_file_once() {
        let probe = SpriteProbe::default();
        let tga = "interface\\addons\\foo\\icon.tga";
        let good = refused_tga(16, 8, &[]);
        let reads = std::cell::Cell::new(0);
        let read = |bytes: Option<&Vec<u8>>| {
            reads.set(reads.get() + 1);
            bytes.cloned().map_or(TryRead::Missing, TryRead::Ready)
        };
        for _ in 0..3 {
            assert!(probe.gate("loose", tga, || read(Some(&good))));
        }
        assert_eq!(reads.get(), 1, "the same file answered from the memo");
        assert_eq!(probe.header_reads.get(), 1);

        // Another resolved file under the same name is its own question: the chain's copy, once a
        // loose one shadows it, or the other way round.
        assert!(!probe.gate("chain", tga, || read(Some(&refused_tga(3, 8, &[])))));
        assert!(!probe.gate("chain", tga, || read(Some(&good))));
        assert_eq!(reads.get(), 2);

        // A file that will not read says no and is asked again.
        assert!(!probe.gate("gone", tga, || read(None)));
        assert!(probe.gate("gone", tga, || read(Some(&good))));
        assert_eq!(reads.get(), 4);
    }

    /// The same, through the store: a second `SetTexture` of the path reads no file.
    #[test]
    fn a_second_probe_of_a_tga_reads_no_file() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = Mutex::new(Chain::open(&data).expect("open the chain"));
        let addons = Tree::new("probe");
        let dir = addons.0.join("Foo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Icon.tga"), refused_tga(16, 8, &[0; 512])).unwrap();
        std::fs::write(dir.join("Odd.tga"), refused_tga(6, 8, &[0; 192])).unwrap();
        let loose = LooseFiles {
            addons: Some(addons.0.clone()),
            install: None,
        };
        let probe = SpriteProbe::default();
        for _ in 0..3 {
            assert!(probe.found(&chain, &loose, "Interface\\AddOns\\Foo\\Icon"));
            assert!(!probe.found(&chain, &loose, "Interface\\AddOns\\Foo\\Odd"));
            assert!(!probe.found(&chain, &loose, "Interface\\AddOns\\Foo\\Missing"));
        }
        // One read each of the two files, and none of a name that is not there.
        assert_eq!(probe.header_reads.get(), 2);
    }
}
