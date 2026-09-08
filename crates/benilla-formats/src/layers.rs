//! **The optional content layers a chain may or may not carry** — today exactly one, the updated
//! character models (`patch-A.MPQ` / the `data-hd` container).
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
//! **The default is OFF**, and that is load-bearing beyond the feature: this repo's golden tests
//! read the real install, and a machine with `patch-A.MPQ` in its `Data` folder would otherwise
//! silently start asserting HD art against 1.12 expectations. Off by default means dropping the
//! archive in changes nothing until something turns it on — [`crate::Chain::open_layered`] for a
//! tool or a test that wants it by name, and the client's own `hdCharacterModels` CVar for a
//! player.

use std::sync::atomic::{AtomicBool, Ordering};

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

/// The live answer. Off until something says otherwise — see the module header.
static HD_MODELS: AtomicBool = AtomicBool::new(false);

/// Turn the updated-character-model layer on or off for every chain opened **after** this call.
///
/// Called once, before the `App` is built, from whoever knows the player's setting
/// (`benilla_app`'s `hdCharacterModels` CVar, `benilla-worldview`'s `$WOW_HD_MODELS`). It does not
/// reach back into a chain already open: the layer is a mount decision, and a mounted chain has
/// made it. That is why the client's options row says the change takes effect on restart rather
/// than pretending to swap 3.3 GiB of art under a running world.
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
