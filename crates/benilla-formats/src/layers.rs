//! **The optional content layers a chain may or may not carry**: the updated character models
//! (`patch-A.MPQ` / the `data-hd` container), the Turtle WoW 1.18.1 layer (the `data-turtle`
//! container) and the reforged art over it (Project Reforged's letter patches / the
//! `data-reforged` container). On the web build all three are live mounts shared by every chain of
//! the page: the HD layer mounts at the first chain open when the player has it on and again
//! whenever they turn it on ([`crate::web::mount_hd_layer`]), the Turtle layer as the client enters
//! a realm that advertises 1.18.1, and the reforged layer with it when the player's setting, read
//! as the page loads ([`set_reforged_models`]), has it on. Natively the HD layer is an archive each
//! chain decides on as it opens, and there is no Turtle or reforged layer.
//!
//! ## What a layer is
//!
//! Vanilla's mount law ([`crate::Chain::open`]) is a *census*: whatever archives the `Data`
//! directory holds, mounted at the priorities the reference gives them. A layer is the one thing
//! that law has no room for — an archive the player **chose** to have, whose absence is as correct
//! a state as its presence. `patch-A.MPQ` is a 1.75 GiB drop-in of HD character art; it mounts
//! above every numbered patch by the reference's own glob rule, so with it in the folder the
//! reference client would simply use it and never ask. benilla asks, because on the web build the
//! question is not "which art" but "download another 1.7 GiB or not", and the honest default for a
//! player on a phone tether is *no*.
//!
//! ## Why this is a global and not a parameter
//!
//! Two chains open in one run — the Bevy `AssetReader`'s (`benilla_assets::MpqAssetReader`) and
//! the world store's (`benilla_world::assets`) — from two crates that do not know each other, both
//! before any resource the answer could live in exists. Threading a `bool` from `benilla-app`
//! through both would put the same argument on `Chain::open`, `open_chain`, `MpqAssetReader::open`,
//! `register_mpq_source` and the world foundation system, so that five signatures could carry one
//! boot-time constant. `benilla_protocol::transport::web::set_ws_base` is the established shape for
//! exactly this — one value, decided before the `App` is built, read wherever it is needed — and
//! this follows it.
//!
//! **This static's default is OFF**, and that is load-bearing beyond the feature: this repo's
//! golden tests read the real install, and a machine with `patch-A.MPQ` in its `Data` folder would
//! otherwise silently start asserting HD art against 1.12 expectations. Off by default means
//! dropping the archive in changes nothing until something turns it on —
//! [`crate::Chain::open_layered`] for a tool or a test that wants it by name, and the client's own
//! `hdCharacterModels` CVar for a player.
//!
//! **The CLIENT's default is ON** (that CVar is registered `"1"` since 2026-09-09, so its options
//! row is an opt-out), and the two are deliberately not the same statement. A player gets the
//! updated art; a test binary, a packer run and `benilla-worldview` get 1.12's, because none of
//! them reads the CVar and nothing here reads back into them. The one lever that crosses both is
//! [`hd_models_env`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The archive that carries the updated character models, matched case-insensitively against the
/// `Data` directory listing.
///
/// A **benilla convention**, not a reference one: the letter-suffixed patch slots (`patch-A`…) are
/// real to the reference's `patch-?.MPQ` glob but nameless to it — it mounts whatever it finds. We
/// reserve this one name for the HD character layer so that "is this the optional layer?" has an
/// answer that does not depend on reading 1.75 GiB to find out what is inside.
pub const HD_MODELS_ARCHIVE: &str = "patch-A.MPQ";

/// The URL path segment the HD layer's BENPACK container is served from, beside `/data`.
///
/// A **separate container**, not a section of the main one, and that is the whole feature: the base
/// `index.bpi` never mentions HD art, so a player with the option off downloads no index entry, no
/// pack and no byte of it. Merging the two would have put 16,158 more keys in the index every
/// visitor fetches before the first frame.
pub const HD_DATA_SEGMENT: &str = "data-hd";

/// What the HD layer's pack ids are offset by in the **resident map**, whose keys are plain numbers
/// shared by every open container (`web/packs.js`'s `Map`).
///
/// Both containers number their packs from 0, so without this the HD layer's pack 7 and the base
/// corpus's pack 7 are the same map entry — and a read of one would be served the other's bytes,
/// which fails as a zstd error or, worse, as a frame decoded out of the wrong body. The offset is
/// the resident **key** only: [`crate::webpack::PackStore`] still builds URLs from the container's
/// own id, because `p/<hi>/<id>.bpk` is the id the packer wrote.
///
/// `1 << 30` against a corpus capped at 18,000 objects — no real id comes within four orders of
/// magnitude of it, and `key_base + pack_id` cannot overflow a `u32`.
///
/// **Twinned in `web/packs.js`** (`HD_PACK_KEY_BASE`), which offsets the HD entries of
/// `boot-packs.json` the same way when it warms them. The two must agree; they are checked against
/// each other by `tests/web_names.rs`.
pub const HD_PACK_KEY_BASE: u32 = 1 << 30;

/// The URL path segment the Turtle WoW 1.18.1 layer's BENPACK container is served from, beside
/// `/data` and `/data-hd`. Fetched only when a 1.18.1 realm is entered.
///
/// **Twinned in `web/packs.js` and `web/sw.js`**; `tests/web_names.rs` checks them.
pub const TURTLE_DATA_SEGMENT: &str = "data-turtle";

/// What the Turtle layer's pack ids are offset by in the resident map — see [`HD_PACK_KEY_BASE`]
/// for why a layer needs its own key range. `1 << 31` sits above the HD layer's `1 << 30` with the
/// same four orders of magnitude to spare, and `key_base + pack_id` still fits a `u32`.
///
/// **Twinned in `web/packs.js`** (`TURTLE_PACK_KEY_BASE`); `tests/web_names.rs` checks them.
pub const TURTLE_PACK_KEY_BASE: u32 = 1 << 31;

/// The URL path segment the reforged art layer's BENPACK container is served from, beside the
/// others. Fetched only when a 1.18.1 realm is entered with the setting on.
///
/// **Twinned in `web/packs.js` and `web/sw.js`**; `tests/web_names.rs` checks them.
pub const REFORGED_DATA_SEGMENT: &str = "data-reforged";

/// What the reforged layer's pack ids are offset by in the resident map — see
/// [`HD_PACK_KEY_BASE`]. `3 << 30` sits above the Turtle layer's `1 << 31` with the same four
/// orders of magnitude to spare, and `key_base + pack_id` still fits a `u32`.
///
/// **Twinned in `web/packs.js`** (`REFORGED_PACK_KEY_BASE`); `tests/web_names.rs` checks them.
pub const REFORGED_PACK_KEY_BASE: u32 = 3 << 30;

/// The name prefix under which the Turtle layer carries a table **merged with the HD layer's**.
///
/// Mounted above the Turtle layer, the HD layer's copy of a table answers for every row, so a row
/// only Turtle has disappears. Where the HD copy lacks Turtle rows (`HelmetGeosetVisData`: 20 of
/// Turtle's 40 ids), `benilla-pack overlay --hd` writes the HD rows plus the Turtle-only rows under
/// `<prefix><name>`, and the web chain serves that alias for `<name>` only while both layers are
/// mounted. Neither layer's own copy changes, so each still answers alone.
pub const TURTLE_HD_ALIAS_PREFIX: &str = "benilla-turtle-hd\\";

/// The name a table merged with the HD layer's is carried under — see [`TURTLE_HD_ALIAS_PREFIX`].
pub fn turtle_hd_alias(name: &str) -> String {
    format!("{TURTLE_HD_ALIAS_PREFIX}{name}")
}

/// Which mounted container answers a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// The 1.12.1 corpus.
    Base,
    /// The updated character models.
    Hd,
    /// The Turtle WoW 1.18.1 layer.
    Turtle,
    /// The Turtle layer's merge of a table with the HD layer's, under [`turtle_hd_alias`].
    TurtleHdAlias,
    /// The reforged art over the Turtle layer.
    Reforged,
}

/// **The layer priority law**, in one place: reforged above HD above Turtle above the base corpus
/// — MPQ's own "later wins", as the reference mounts `patch-?.MPQ` by name, so `patch-B` to
/// `patch-Y` above `patch-A` above every numbered patch — except that a table merged for the HD and
/// Turtle pair answers for the pair. Each layer is `None` when it is not mounted and whether it
/// holds the name otherwise; `alias` is whether the Turtle layer holds a merge of it.
pub fn pick_layer(
    reforged: Option<bool>,
    hd: Option<bool>,
    turtle: Option<bool>,
    alias: bool,
) -> Layer {
    if reforged == Some(true) {
        return Layer::Reforged;
    }
    if hd.is_some() && turtle.is_some() && alias {
        return Layer::TurtleHdAlias;
    }
    if hd == Some(true) {
        return Layer::Hd;
    }
    if turtle == Some(true) {
        return Layer::Turtle;
    }
    Layer::Base
}

/// Bumped on every live mount change, so a cache of chain-derived data can tell it is stale.
static DATA_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Which mount set the chain answers from: it changes whenever a live layer is mounted or
/// unmounted, and never otherwise.
pub fn data_generation() -> u64 {
    DATA_GENERATION.load(Ordering::Acquire)
}

/// Whether this target can mount the Turtle layer at all: the browser build, which is served the
/// `data-turtle` container. The native build reads MPQs and has no Turtle container to read.
pub const fn turtle_available() -> bool {
    cfg!(target_arch = "wasm32")
}

#[cfg(target_arch = "wasm32")]
mod live {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, RwLock};

    use crate::webpack::PackStore;

    /// One live layer's mount, shared by every chain of the process (the `AssetReader`'s, the world
    /// store's, the UI's): one live mount, not a parameter three crates would have to thread.
    pub(super) struct Slot {
        store: RwLock<Option<Arc<PackStore>>>,
        /// Whether `store` holds a layer: the read path's fast path. Every chain read asks, so with
        /// the layer off its whole cost is this one relaxed load — the lock is never taken.
        mounted: AtomicBool,
    }

    impl Slot {
        const fn new() -> Self {
            Self {
                store: RwLock::new(None),
                mounted: AtomicBool::new(false),
            }
        }

        fn mount(&self, store: PackStore) {
            *self.store.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(store));
            self.mounted.store(true, Ordering::Release);
            super::DATA_GENERATION.fetch_add(1, Ordering::AcqRel);
        }

        fn unmount(&self) {
            self.mounted.store(false, Ordering::Release);
            let was = self.store.write().unwrap_or_else(|e| e.into_inner()).take();
            if was.is_some() {
                super::DATA_GENERATION.fetch_add(1, Ordering::AcqRel);
            }
        }

        /// The mounted layer, as a handle a read can keep past the lock — the lock is never held
        /// while a read waits on the network, which on the page's own thread would be a wait it
        /// cannot make.
        fn get(&self) -> Option<Arc<PackStore>> {
            if !self.mounted.load(Ordering::Relaxed) {
                return None;
            }
            self.store.read().unwrap_or_else(|e| e.into_inner()).clone()
        }

        fn mounted(&self) -> bool {
            self.mounted.load(Ordering::Relaxed)
        }
    }

    static TURTLE: Slot = Slot::new();
    static HD: Slot = Slot::new();
    static REFORGED: Slot = Slot::new();

    pub fn mount_reforged(store: PackStore) {
        REFORGED.mount(store);
    }

    pub fn unmount_reforged() {
        REFORGED.unmount();
    }

    /// The mounted reforged art layer.
    pub fn reforged() -> Option<Arc<PackStore>> {
        REFORGED.get()
    }

    pub(super) fn reforged_mounted() -> bool {
        REFORGED.mounted()
    }

    pub fn mount_turtle(store: PackStore) {
        TURTLE.mount(store);
    }

    pub fn unmount_turtle() {
        TURTLE.unmount();
    }

    /// The mounted Turtle layer.
    pub fn turtle() -> Option<Arc<PackStore>> {
        TURTLE.get()
    }

    pub(super) fn turtle_mounted() -> bool {
        TURTLE.mounted()
    }

    pub fn mount_hd(store: PackStore) {
        HD.mount(store);
    }

    pub fn unmount_hd() {
        HD.unmount();
    }

    /// The mounted updated-character-model layer.
    pub fn hd() -> Option<Arc<PackStore>> {
        HD.get()
    }

    pub(super) fn hd_mounted() -> bool {
        HD.mounted()
    }
}

#[cfg(target_arch = "wasm32")]
pub use live::{
    hd, mount_hd, mount_reforged, mount_turtle, reforged, turtle, unmount_hd, unmount_reforged,
    unmount_turtle,
};

/// Whether the reforged art layer is mounted now. Always `false` natively.
pub fn reforged_mounted() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        live::reforged_mounted()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// Whether the player's setting has the reforged art on, as read when the page loaded. **Latched
/// for the session**: the layer mounts with the Turtle layer on every entry into a 1.18.1 realm
/// while this is on, and a change of the setting takes effect at the next load.
static REFORGED_MODELS: AtomicBool = AtomicBool::new(false);

/// Latch the reforged setting for the session — called once, before the `App` is built, by
/// whoever read the player's setting (`benilla_app`'s `reforgedModels` CVar). Off until then, so a
/// test or a tool never mounts it.
pub fn set_reforged_models(on: bool) {
    REFORGED_MODELS.store(on, Ordering::Relaxed);
}

/// Whether the reforged art mounts with the Turtle layer this session ([`set_reforged_models`]).
pub fn reforged_models() -> bool {
    REFORGED_MODELS.load(Ordering::Relaxed)
}

/// Whether the Turtle layer is mounted now. Always `false` natively.
pub fn turtle_mounted() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        live::turtle_mounted()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// Whether the updated character models are mounted now: the live mount on the web build, and
/// natively the switch every chain opens by ([`hd_character_models`]), since a native chain
/// decides at its open and the client opens its chains at boot.
pub fn hd_mounted() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        live::hd_mounted()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        hd_character_models()
    }
}

/// Whether this target can mount or unmount the updated character models under a running client:
/// the browser build, whose layer is a live mount. A native chain decides at its open.
pub const fn hd_live() -> bool {
    cfg!(target_arch = "wasm32")
}

/// The live answer. Off until something says otherwise — see the module header.
static HD_MODELS: AtomicBool = AtomicBool::new(false);

/// Turn the updated-character-model layer on or off for every chain opened **after** this call.
///
/// Called before the `App` is built, from whoever knows the player's setting (`benilla_app`'s
/// `hdCharacterModels` CVar, `benilla-worldview`'s `$WOW_HD_MODELS`). It does not reach back into
/// a chain already open: natively the layer is a mount decision a chain makes as it opens, so the
/// native options row takes effect on restart. On the web build the layer is a live mount, and
/// [`crate::web::mount_hd_layer`] / [`crate::web::close_hd_layer`] change it under a running client
/// (and set this switch to match).
pub fn set_hd_character_models(on: bool) {
    HD_MODELS.store(on, Ordering::Relaxed);
}

/// Whether [`crate::Chain::open`] will mount the updated character models.
pub fn hd_character_models() -> bool {
    HD_MODELS.load(Ordering::Relaxed)
}

/// How a layer switch reads a value, wherever it comes from — `$WOW_HD_MODELS`, `?hd_models=1`,
/// or the `hdCharacterModels` CVar out of `config.toml`.
///
/// One function because there are three sources and they must not disagree about what "on" means:
/// a page that read `"0"` as truthy would fetch 1.5 GiB the client then refused to mount. Empty is
/// **not** a value — `WOW_HD_MODELS=` is how a shell unsets an inherited one — which is why this
/// answers `Option` rather than `bool`.
pub fn layer_switch(raw: &str) -> Option<bool> {
    match raw.trim() {
        "" => None,
        "0" | "false" | "off" => Some(false),
        _ => Some(true),
    }
}

/// The session-only environment override, in the shape every other `$WOW_*` switch in this client
/// takes: `WOW_HD_MODELS=1` forces the layer on, `0` forces it off, unset leaves the caller's own
/// answer alone.
///
/// It exists for the callers a CVar cannot reach: a test binary running the model suite against
/// the HD art (`WOW_HD_MODELS=1 cargo test -p benilla-formats models::`), `wenilla-host`, and the
/// detached `benilla-worldview` probe — none of which has a `config.toml`.
///
/// **Compiled on every target and always `None` on `wasm32`**, where `std::env` is an empty map: a
/// browser tab has no process environment. It is not `cfg`-gated, and that is the point —
/// `benilla-world`'s `worldview` and `wenilla-host` both compile for the web build, and a
/// native-only helper turns "this switch does nothing there" into a build break in a crate that has
/// no business knowing about it. The client's own web arm is `benilla_app::webenv::var`, which
/// reads the page's env object and `?hd_models=1` instead — the same ladder every other `$WOW_*`
/// switch takes on that target.
pub fn hd_models_env() -> Option<bool> {
    layer_switch(&std::env::var("WOW_HD_MODELS").ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HD over Turtle over the base corpus, each only when mounted and holding the name, and the
    /// pair's merged table only when both are mounted.
    #[test]
    fn the_priority_is_hd_then_turtle_then_base() {
        assert_eq!(pick_layer(None, None, None, false), Layer::Base);
        assert_eq!(
            pick_layer(None, Some(false), Some(false), false),
            Layer::Base
        );
        assert_eq!(pick_layer(None, Some(true), None, false), Layer::Hd);
        assert_eq!(pick_layer(None, None, Some(true), false), Layer::Turtle);
        assert_eq!(pick_layer(None, Some(true), Some(true), false), Layer::Hd);
        assert_eq!(
            pick_layer(None, Some(false), Some(true), false),
            Layer::Turtle
        );
        assert_eq!(
            pick_layer(None, Some(true), Some(true), true),
            Layer::TurtleHdAlias
        );
        // A merge answers only for the pair: with either layer down, the alias is not consulted.
        assert_eq!(pick_layer(None, None, Some(true), true), Layer::Turtle);
        assert_eq!(pick_layer(None, Some(true), None, true), Layer::Hd);
    }

    /// The reforged art answers every name it holds, over every other layer and merge — its
    /// letters mount above `patch-A` and the numbered patches — and nothing it does not.
    #[test]
    fn the_reforged_layer_is_on_top() {
        assert_eq!(
            pick_layer(Some(true), Some(true), Some(true), true),
            Layer::Reforged
        );
        assert_eq!(
            pick_layer(Some(true), None, Some(true), false),
            Layer::Reforged
        );
        assert_eq!(
            pick_layer(Some(false), Some(true), Some(true), true),
            Layer::TurtleHdAlias
        );
        assert_eq!(
            pick_layer(Some(false), None, Some(true), false),
            Layer::Turtle
        );
        assert_eq!(
            pick_layer(Some(false), None, Some(false), false),
            Layer::Base
        );
    }

    #[test]
    fn the_layer_key_ranges_do_not_overlap_and_fit_a_u32() {
        const { assert!(HD_PACK_KEY_BASE < TURTLE_PACK_KEY_BASE) };
        const { assert!(TURTLE_PACK_KEY_BASE < REFORGED_PACK_KEY_BASE) };
        // The corpus cap is 18,000 objects per container.
        assert!(HD_PACK_KEY_BASE.checked_add(18_000).unwrap() < TURTLE_PACK_KEY_BASE);
        assert!(TURTLE_PACK_KEY_BASE.checked_add(18_000).unwrap() < REFORGED_PACK_KEY_BASE);
        assert!(REFORGED_PACK_KEY_BASE.checked_add(18_000).is_some());
    }

    /// The reforged setting is off until the client latches it, so no test or tool mounts it, and
    /// the layer is never mounted natively.
    #[test]
    fn the_reforged_layer_is_off_by_default_and_browser_only() {
        assert!(!reforged_models());
        if !turtle_available() {
            assert!(!reforged_mounted());
        }
    }

    #[test]
    fn the_turtle_layer_is_browser_only_and_never_mounted_natively() {
        assert_eq!(turtle_available(), cfg!(target_arch = "wasm32"));
        if !turtle_available() {
            assert!(!turtle_mounted());
        }
    }

    /// The HD layer is a live mount on the web build only; natively a chain decides at its open.
    #[test]
    fn the_hd_layer_is_live_only_in_the_browser() {
        assert_eq!(hd_live(), cfg!(target_arch = "wasm32"));
    }
}
