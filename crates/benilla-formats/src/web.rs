//! Web-target chain plumbing: `wasm32-unknown-unknown` has no filesystem, so [`crate::Chain`]
//! answers every read with an HTTP call. This module is that HTTP shim — the bytes-over-the-wire
//! half; *what* is fetched is [`crate::webpack`]'s question.
//!
//! The client reads a **BENPACK container** (`data/index.bpi` + `data/p/**.bpk`) served by a dumb
//! object store, so the only routes left here are "get these bytes" and "get this byte range".
//! [`encode_name`] survives one phase longer than the scheme it belongs to: `wenilla-host --data`
//! still answers `GET /data/{encoded name}` out of a live MPQ chain, which is how a session runs
//! without a 3.3 GiB container build, and that route's contract is this function.
//!
//! [`encode_name`] is plain string math with no browser dependency, so it is exercised natively
//! (`tests/web_names.rs`); [`data_base`], [`fetch_sync`] and [`fetch_sync_range`] need `web-sys`
//! and only compile their bodies on `wasm32`. [`warm_priority`] lives here for the first reason
//! rather than the second: it decides the ORDER packs are fetched in, which is
//! [`crate::webpack`]'s business — but that module compiles on `wasm32` only, and a rule this
//! load-bearing should not be somewhere no native test can reach it.

/// **Which reads the page should fetch first** — the warm priority a name is asked for at
/// (`web/packs.js`'s `warmPack`/`prefetch`; a lower number starts sooner).
///
/// The container is a network, and a network has an order. At a zone boundary the client asks for
/// one `.adt`, a handful of `.wmo` roots and groups, a few dozen `.m2`s — and, in the same frame,
/// several hundred `.blp`s for all of them. Every one of those is a pack warm, the browser will run
/// six at a time, and *nothing about the reads themselves says which six matter*. Before this
/// existed the answer was arrival order, which is very nearly the worst possible one: the four
/// objects that decide whether there is a floor under the player queue behind the art of the
/// building they hold up, land seconds late, and the body walks onto ground that has no collider.
///
/// ## The ladder
///
/// Five tiers, best first. The first three are **terrain**, and they exist because "collision
/// first" turned out to be one lane too coarse: the tile a body is about to walk onto shared
/// [`PRIORITY_GEOMETRY`] with every doodad and building of the zone it is standing in, of which
/// there are hundreds. Six slots drained in arrival order then land the neighbouring tile *after*
/// the art — which is the reported symptom: an ADT that only turns up once the player has stepped
/// onto it and a loading cover has stopped everything else from being asked for.
///
/// - [`PRIORITY_TERRAIN_FOCUS`] — the map's WDT (nothing streams before it), the tile under the
///   view focus, and the one neighbouring tile the focus is nearest to stepping onto.
/// - [`PRIORITY_TERRAIN_NEAR`] — the rest of the focus tile's 8 neighbours.
/// - [`PRIORITY_TERRAIN`] — terrain further out, the WDL horizon, and `Tileset\` layer art. A
///   layer texture is here rather than with the other `.blp`s because the `AdtTile` asset does not
///   *finish* until its layers have been read (`benilla_assets`' `AdtLoader`), and the tile's
///   entity — mesh, water, clutter and collider alike — is spawned off that asset landing. A
///   tileset read is terrain wearing a texture's extension.
/// - [`PRIORITY_GEOMETRY`] — the collision-bearing model art: `.wmo` roots *and* their group files
///   (the MOPY/MOVI faces both per-WMO collider bakes come from), `.m2`/`.mdx` (the header's
///   bounding box and the collision hull behind it).
/// - [`PRIORITY_NORMAL`] — everything else: textures, sounds, DBCs, UI art.
///
/// Nothing is *starved* by this: the queue is work-conserving, so with no terrain outstanding the
/// geometry lane has the whole pool exactly as before, and `packs.js` still reserves slots for the
/// lanes below the collision one (`WARM_GEOMETRY_SLOTS`). That reservation is why the tiers are
/// ordered as they are: the collision lane stays the single test `priority <= PRIORITY_GEOMETRY`,
/// with every terrain tier inside it.
///
/// ## Why the near tiers need a side channel
///
/// The first two tiers are the only thing here that is not a pure function of the name: "near" is a
/// distance, and a name does not carry one. So the terrain streamer publishes the one fact that
/// resolves it — [`set_terrain_focus`] — and this reads it back. With nothing published (a native
/// build, boot before the first stream frame, a released world) every tile is [`PRIORITY_TERRAIN`],
/// which is what the whole terrain lane was worth before the tiers existed.
pub fn warm_priority(name: &str) -> u8 {
    // `Tileset\…` — the terrain layer art, whatever extension it wears. Tested before the
    // extension law because it is a *tree*, not a family: the census resolved all 1 048 of these
    // names under this one prefix (`benilla_assets`' `terrain`), and `AdtLoader` is the only thing
    // that reads them.
    if is_tileset_name(name) {
        return PRIORITY_TERRAIN;
    }
    // The extension, or nothing: the last dot, refused when a path separator follows it (a
    // directory with a dot in its name, holding a file that has no extension of its own).
    let ext = match name.rsplit_once('.') {
        Some((_, ext)) if !ext.contains(['\\', '/']) => ext,
        _ => return PRIORITY_NORMAL,
    };
    let is = |want: &str| ext.eq_ignore_ascii_case(want);
    if is("adt") {
        return tile_priority(name);
    }
    // The tile-existence index the streamer gates on: no tile of this map can even be *asked* for
    // until it lands, so it can never sit below the tiles it unblocks.
    if is("wdt") {
        return PRIORITY_TERRAIN_FOCUS;
    }
    // The coarse whole-map heightmap the stand-in ground floor is built from (`benilla_world`'s
    // `terrain_stream::provisional`) — one object for the map, and what a body stands on while a
    // real tile is still coming.
    if is("wdl") {
        return PRIORITY_TERRAIN;
    }
    const GEOMETRY: [&str; 3] = ["wmo", "m2", "mdx"];
    if GEOMETRY.iter().any(|g| is(g)) {
        // A model a server entity wears is within the server's visibility range of the player,
        // where a scenery placement can be a tile away: it outranks the scenery queue.
        if is_urgent_model(name) {
            PRIORITY_TERRAIN_NEAR
        } else {
            PRIORITY_GEOMETRY
        }
    } else {
        PRIORITY_NORMAL
    }
}

/// Is this name under the `Tileset\` tree? Case- and separator-insensitive, because a name reaches
/// the container from two directions: the chain's own authored `Tileset\Elwynn\…`, and the
/// lowercased forward-slash form `AdtLoader` builds for the `mpq://` asset source.
fn is_tileset_name(name: &str) -> bool {
    const TREE: &str = "tileset";
    name.len() > TREE.len()
        && name.is_char_boundary(TREE.len())
        && name[..TREE.len()].eq_ignore_ascii_case(TREE)
        && matches!(name.as_bytes()[TREE.len()], b'\\' | b'/')
}

/// Where one `.adt` name sits in the terrain tiers — see [`warm_priority`]'s ladder.
///
/// The published focus decides it, and a name that does not resolve against it settles at
/// [`PRIORITY_TERRAIN`]: no focus published, a name from a different map (a worldport's first
/// frames, where both maps' tiles are briefly in flight), or a tile name this cannot parse.
/// Falling back *down* the ladder rather than up is deliberate — an unrecognised tile must never
/// outrank the one the body is standing on.
fn tile_priority(name: &str) -> u8 {
    let Some(focus) = terrain_focus() else {
        return PRIORITY_TERRAIN;
    };
    let Some(tile) = tile_of(name, &focus.map_dir) else {
        return PRIORITY_TERRAIN;
    };
    if tile == focus.tile || tile == focus.lean {
        PRIORITY_TERRAIN_FOCUS
    } else if (tile.0 - focus.tile.0).abs() <= 1 && (tile.1 - focus.tile.1).abs() <= 1 {
        PRIORITY_TERRAIN_NEAR
    } else {
        PRIORITY_TERRAIN
    }
}

/// The tile `name` addresses, if it is `<map_dir>_<x>_<y>.adt` for this map.
///
/// The map check is what keeps a worldport honest: `Azeroth_32_48.adt` and `Kalimdor_32_48.adt`
/// name the same square of a 64×64 grid and utterly different ground, and for the handful of frames
/// where the old map's tiles are still draining they are both on the wire.
fn tile_of(name: &str, map_dir: &str) -> Option<(i32, i32)> {
    let file = name.rsplit(['\\', '/']).next()?;
    let stem = file.get(..file.len().checked_sub(".adt".len())?)?;
    // `<dir>_<x>_<y>`, split from the right: a map directory is free to hold underscores of its
    // own, and the two coordinates are always the last two fields.
    let (head, y) = stem.rsplit_once('_')?;
    let (dir, x) = head.rsplit_once('_')?;
    if !dir.eq_ignore_ascii_case(map_dir) {
        return None;
    }
    Some((x.parse().ok()?, y.parse().ok()?))
}

/// The terrain streamer's view focus, as the warm queue needs it — see [`set_terrain_focus`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainFocus {
    /// The current map's directory (`"Azeroth"`), matched against a tile name's own prefix.
    pub map_dir: String,
    /// The tile the view focus stands on.
    pub tile: (i32, i32),
    /// The one neighbouring tile the focus is nearest to — the focus tile itself at a map edge,
    /// where there is no neighbour on that side to name.
    pub lean: (i32, i32),
}

/// **Where terrain is being streamed from**, published by `benilla_world`'s terrain streamer once a
/// frame and read by [`warm_priority`] to place a `.adt` name in the tiers.
///
/// A side channel rather than an argument, because the read that needs it is four layers away and
/// carries nothing but a name: `AssetServer` → `benilla_assets`' `MpqAssetReader` → [`crate::Chain`]
/// → `PackStore::read_async_stored`, none of which has any business knowing where the player is.
/// One global the one writer sets and the one reader reads is the narrower seam.
///
/// Cheap to call every frame: a read, and a write only when the answer changed.
pub fn set_terrain_focus(map_dir: &str, tile: (i32, i32), lean: (i32, i32)) {
    let unchanged = TERRAIN_FOCUS.read().is_ok_and(|f| {
        f.as_ref()
            .is_some_and(|f| f.tile == tile && f.lean == lean && f.map_dir == map_dir)
    });
    if unchanged {
        return;
    }
    if let Ok(mut slot) = TERRAIN_FOCUS.write() {
        *slot = Some(TerrainFocus {
            map_dir: map_dir.to_string(),
            tile,
            lean,
        });
    }
}

/// Forget the published focus — the world was released and there is no map to be near to. Every
/// tile then warms at [`PRIORITY_TERRAIN`], the tier the whole lane had before the tiers.
pub fn clear_terrain_focus() {
    if let Ok(mut slot) = TERRAIN_FOCUS.write() {
        *slot = None;
    }
}

/// The published focus, or `None` where nothing has published one.
pub fn terrain_focus() -> Option<TerrainFocus> {
    TERRAIN_FOCUS.read().ok().and_then(|f| f.clone())
}

/// A model name's identity for [`mark_urgent_model`]: lowercased, backslash-separated, without its
/// extension, and a WMO group (`<root>_NNN.wmo`) folded onto its root, so marking a root covers
/// every group file its loader reads after it.
fn model_stem(name: &str) -> String {
    let mut s = name.to_ascii_lowercase().replace('/', "\\");
    if let Some(rest) = s.strip_prefix("mpq:\\\\") {
        s = rest.to_string();
    }
    let Some((stem, ext)) = s.rsplit_once('.') else {
        return s;
    };
    if ext == "wmo" {
        if let Some((root, group)) = stem.rsplit_once('_') {
            if group.len() == 3 && group.bytes().all(|b| b.is_ascii_digit()) {
                return root.to_string();
            }
        }
    }
    if matches!(ext, "wmo" | "m2" | "mdx") {
        stem.to_string()
    } else {
        s
    }
}

/// Mark `path`'s model urgent: its root and group files warm at [`PRIORITY_TERRAIN_NEAR`], ahead
/// of the scenery queue. For the model a server entity (a GameObject, a creature) wears, which is
/// within visibility range; Turtle WoW spawns whole buildings as GameObjects, and at
/// [`PRIORITY_GEOMETRY`] one waits behind every doodad of the zones around, its collider with it.
pub fn mark_urgent_model(path: &str) {
    if let Ok(mut set) = URGENT_MODELS.write() {
        set.insert(model_stem(path));
    }
}

fn is_urgent_model(name: &str) -> bool {
    URGENT_MODELS
        .read()
        .is_ok_and(|set| !set.is_empty() && set.contains(&model_stem(name)))
}

/// The stems [`mark_urgent_model`] marked. Never cleared: a stem stays small, the set is bounded
/// by the entity displays a session meets, and a model once near is the one likeliest near again.
static URGENT_MODELS: std::sync::LazyLock<std::sync::RwLock<std::collections::HashSet<String>>> =
    std::sync::LazyLock::new(Default::default);

/// The one published focus. A `RwLock` rather than the thread-local this would be on wasm alone:
/// the writer is a Bevy system, and on native that is whichever worker thread the scheduler
/// happened to run it on.
static TERRAIN_FOCUS: std::sync::RwLock<Option<TerrainFocus>> = std::sync::RwLock::new(None);

/// The tile under the view focus, the neighbour it is nearest to, and the map's WDT. See
/// [`warm_priority`].
pub const PRIORITY_TERRAIN_FOCUS: u8 = 0;
/// The rest of the focus tile's 8 neighbours. See [`warm_priority`].
pub const PRIORITY_TERRAIN_NEAR: u8 = 1;
/// Terrain further out, the WDL horizon, and `Tileset\` layer art. See [`warm_priority`].
pub const PRIORITY_TERRAIN: u8 = 2;
/// Collision-bearing model art — `.wmo` roots and groups, `.m2`/`.mdx` bounds and hulls. **The
/// bottom of the collision lane**: `packs.js` reserves slots against `priority <= PRIORITY_GEOMETRY`,
/// so every tier above this one is inside that reservation. See [`warm_priority`].
pub const PRIORITY_GEOMETRY: u8 = 3;
/// Everything else. See [`warm_priority`].
pub const PRIORITY_NORMAL: u8 = 4;

/// Percent-encode `name` exactly like JavaScript's `encodeURIComponent`: the unreserved set is
/// `A-Za-z0-9-_.!~*'()`; every other byte, including `\` (chain names are internally
/// backslash-separated), becomes `%XX` uppercase-hex. This is the client half of the Data URL
/// scheme's "the full name percent-encoded as one component" rule — the web host's decode must be
/// this function's exact inverse, so it is kept pure and unit-tested rather than left to whatever
/// a URL-building crate happens to escape.
pub fn encode_name(name: &str) -> String {
    const UNRESERVED: &[u8] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'()";
    let mut out = String::with_capacity(name.len());
    for &byte in name.as_bytes() {
        if UNRESERVED.contains(&byte) {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::JsValue;
    use web_sys::XmlHttpRequest;

    /// The root the container is served from: `/data` under this page's own origin — `index.bpi`,
    /// `listfile.txt` and `p/<hi>/<id>.bpk` all hang off it. Same origin whether that is a static
    /// asset host serving `web/site/data` or `wenilla-host --packs` serving the same tree off
    /// a laptop, so there is never a cross-origin question to configure.
    pub fn data_base() -> String {
        let origin = web_sys::window()
            .expect("benilla_formats::web only runs inside a browser tab")
            .location()
            .origin()
            .expect("window.location.origin");
        format!("{origin}/data")
    }

    /// The root the **updated-character-model layer** is served from: `/data-hd`, beside `/data` on
    /// the same origin, laid out identically (`index.bpi`, `listfile.txt`, `p/<hi>/<id>.bpk`).
    ///
    /// A sibling directory rather than a subdirectory of `/data`, because `web/sw.js` is scoped to
    /// `/data/` and the two containers have different cache lifetimes to a player who turns the
    /// layer off: a nested tree would be swept by rules written for the corpus that is always there.
    pub fn hd_data_base() -> String {
        let origin = web_sys::window()
            .expect("benilla_formats::web only runs inside a browser tab")
            .location()
            .origin()
            .expect("window.location.origin");
        format!("{origin}/{}", crate::layers::HD_DATA_SEGMENT)
    }

    /// The root the **Turtle WoW 1.18.1 layer** is served from: `/data-turtle`, beside `/data` and
    /// `/data-hd`, laid out like them.
    pub fn turtle_data_base() -> String {
        let origin = web_sys::window()
            .expect("benilla_formats::web only runs inside a browser tab")
            .location()
            .origin()
            .expect("window.location.origin");
        format!("{origin}/{}", crate::layers::TURTLE_DATA_SEGMENT)
    }

    /// The root the **reforged art layer** is served from: `/data-reforged`, beside the others,
    /// laid out like them.
    pub fn reforged_data_base() -> String {
        let origin = web_sys::window()
            .expect("benilla_formats::web only runs inside a browser tab")
            .location()
            .origin()
            .expect("window.location.origin");
        format!("{origin}/{}", crate::layers::REFORGED_DATA_SEGMENT)
    }

    /// Tell the page to open or close its half of a live layer (`web/packs.js`'s `openLayer` /
    /// `closeLayer`: the container in the decode worker, the boot packs, the resident packs), and
    /// return at once — an open's promise is the page's to settle; nothing here waits on it.
    fn page_layer(method: &str, layer: &str) {
        use wasm_bindgen::JsCast;
        let Some(window) = web_sys::window() else {
            return;
        };
        let Ok(packs) = js_sys::Reflect::get(&window, &"__wenilla_packs".into()) else {
            return;
        };
        if let Ok(f) = js_sys::Reflect::get(&packs, &method.into()) {
            if let Some(f) = f.dyn_ref::<js_sys::Function>() {
                let _ = f.call1(&packs, &layer.into());
            }
        }
    }

    /// Mount the Turtle WoW 1.18.1 layer now, for a 1.18.1 realm being entered: off the index the
    /// page already holds from an earlier visit, else one blocking fetch of it — this runs behind
    /// the connect dialog, and a realm switch is a load, not a frame. An origin that serves no
    /// `/data-turtle` is an error the caller turns into a refused realm.
    pub fn mount_turtle_layer() -> anyhow::Result<()> {
        if crate::layers::turtle_mounted() {
            return Ok(());
        }
        let store = crate::webpack::PackStore::open_turtle(turtle_data_base())?;
        crate::layers::mount_turtle(store);
        // The page's half: the worker's container and the boot packs. Idempotent there. Never
        // earlier: a realm list that merely advertises a 1.18.1 realm fetches nothing.
        page_layer("openLayer", "turtle");
        Ok(())
    }

    /// Unmount the Turtle layer and let the page drop what it holds of it (its resident packs and
    /// its worker container), for a realm that does not advertise 1.18.1.
    pub fn close_turtle_layer() {
        crate::layers::unmount_turtle();
        page_layer("closeLayer", "turtle");
    }

    /// Mount the reforged art layer now, with the Turtle layer for a 1.18.1 realm being entered:
    /// the Turtle layer's terms, off the index the page holds or one blocking fetch of it. An
    /// origin that serves no `/data-reforged` is an error the caller reports and plays on without.
    pub fn mount_reforged_layer() -> anyhow::Result<()> {
        if crate::layers::reforged_mounted() {
            return Ok(());
        }
        let store = crate::webpack::PackStore::open_reforged(reforged_data_base())?;
        crate::layers::mount_reforged(store);
        page_layer("openLayer", "reforged");
        Ok(())
    }

    /// Unmount the reforged art layer and let the page drop what it holds of it, for a realm that
    /// does not advertise 1.18.1.
    pub fn close_reforged_layer() {
        crate::layers::unmount_reforged();
        page_layer("closeLayer", "reforged");
    }

    /// Mount the updated character models now, for the player turning them on: off the index the
    /// page holds (it fetched one at boot with the option on, or on an earlier turn), else one
    /// blocking fetch of it — this runs on the Apply click, an edge, never in a frame. Every later
    /// read of a name the layer holds answers from it, and the switch chains open by
    /// ([`crate::layers::hd_character_models`]) follows. An origin that serves no `/data-hd` is an
    /// error and leaves nothing mounted.
    pub fn mount_hd_layer() -> anyhow::Result<()> {
        if crate::layers::hd_mounted() {
            return Ok(());
        }
        let store = crate::webpack::PackStore::open_hd(hd_data_base())?;
        crate::layers::mount_hd(store);
        crate::layers::set_hd_character_models(true);
        // The page's half: the worker's container and the boot packs. Idempotent there.
        page_layer("openLayer", "hd");
        Ok(())
    }

    /// Unmount the updated character models, for the player turning them off, and let the page
    /// drop what it holds of the layer (its resident packs and its worker container). Cache
    /// Storage keeps its objects, so a later turn back on refetches the index and not the packs.
    pub fn close_hd_layer() {
        crate::layers::unmount_hd();
        crate::layers::set_hd_character_models(false);
        page_layer("closeLayer", "hd");
    }

    /// One `console.warn`, for the handful of places that must say something to a player's console
    /// on a target where `tracing` has been taken off the build (`quiet_console_on_web`).
    pub fn console_warn(msg: &str) {
        web_sys::console::warn_1(&JsValue::from_str(msg));
    }

    /// Append `name` to the page's boot-read trace, if the page armed one — the input to
    /// `web/boot-manifest.json` (see `web/boot.js`). A page that wants the trace defines
    /// `window.__wenilla_boottrace = []` before `init()` (the `?boottrace=1` switch does);
    /// every chain read then pushes its raw backslash name in true first-need order, and a
    /// person copies the deduplicated array out of the console into the manifest. Without the
    /// array this is one snapshotted `Reflect::get` for the whole session and per-call nothing —
    /// tracing must cost the boot it measures as close to zero as possible.
    pub fn trace(name: &str) {
        use wasm_bindgen::JsCast;
        thread_local! {
            static TRACE: std::cell::OnceCell<Option<js_sys::Array>> =
                const { std::cell::OnceCell::new() };
        }
        TRACE.with(|cell| {
            let arr = cell.get_or_init(|| {
                let window = web_sys::window()?;
                js_sys::Reflect::get(&window, &"__wenilla_boottrace".into())
                    .ok()?
                    .dyn_into::<js_sys::Array>()
                    .ok()
            });
            if let Some(arr) = arr {
                arr.push(&name.into());
            }
        });
    }

    /// The **layer half** of the boot trace: `name` was answered by the updated-character-model
    /// layer rather than by the base corpus.
    ///
    /// A second array (`window.__wenilla_boottrace_hd`), not a second column in the first one, and
    /// that is deliberate. `boot-manifest.json` is a flat list of names that
    /// `benilla-app/tests/boot_manifest.rs` guards the shape of and `benilla-pack sidecars`
    /// translates against *each* container's index — one name list, two trees, no format change.
    /// Adding a layer tag to it would have made the manifest a different file for the sake of a
    /// fact the manifest does not need.
    ///
    /// What the second array is for is the question a re-capture actually has to answer: **is the
    /// layer being read at boot at all, and where.** With the option on and this array empty, the
    /// layer is mounted and nothing wants it — a wiring bug that is otherwise invisible, because a
    /// container that is never read looks exactly like a container that is working.
    pub fn trace_hd(name: &str) {
        use wasm_bindgen::JsCast;
        thread_local! {
            static TRACE_HD: std::cell::OnceCell<Option<js_sys::Array>> =
                const { std::cell::OnceCell::new() };
        }
        TRACE_HD.with(|cell| {
            let arr = cell.get_or_init(|| {
                let window = web_sys::window()?;
                js_sys::Reflect::get(&window, &"__wenilla_boottrace_hd".into())
                    .ok()?
                    .dyn_into::<js_sys::Array>()
                    .ok()
            });
            if let Some(arr) = arr {
                arr.push(&name.into());
            }
        });
    }

    /// The Turtle layer's half of the boot trace (`window.__wenilla_boottrace_turtle`): `name` was
    /// answered by the Turtle WoW 1.18.1 layer. The instrument for "no Turtle read on a 1.12.1
    /// realm": after switching back, this array stops growing.
    pub fn trace_turtle(name: &str) {
        use wasm_bindgen::JsCast;
        thread_local! {
            static TRACE_TURTLE: std::cell::OnceCell<Option<js_sys::Array>> =
                const { std::cell::OnceCell::new() };
        }
        TRACE_TURTLE.with(|cell| {
            let arr = cell.get_or_init(|| {
                let window = web_sys::window()?;
                js_sys::Reflect::get(&window, &"__wenilla_boottrace_turtle".into())
                    .ok()?
                    .dyn_into::<js_sys::Array>()
                    .ok()
            });
            if let Some(arr) = arr {
                arr.push(&name.into());
            }
        });
    }

    /// The reforged layer's half of the boot trace (`window.__wenilla_boottrace_reforged`): `name`
    /// was answered by the reforged art layer.
    pub fn trace_reforged(name: &str) {
        use wasm_bindgen::JsCast;
        thread_local! {
            static TRACE_REFORGED: std::cell::OnceCell<Option<js_sys::Array>> =
                const { std::cell::OnceCell::new() };
        }
        TRACE_REFORGED.with(|cell| {
            let arr = cell.get_or_init(|| {
                let window = web_sys::window()?;
                js_sys::Reflect::get(&window, &"__wenilla_boottrace_reforged".into())
                    .ok()?
                    .dyn_into::<js_sys::Array>()
                    .ok()
            });
            if let Some(arr) = arr {
                arr.push(&name.into());
            }
        });
    }

    /// A blocking `XMLHttpRequest` GET of `url`, returning `(status, body)`.
    ///
    /// Synchronous on purpose: [`crate::Chain::read`] is called from Bevy systems on the main
    /// thread today — `WorldAssets` and ~60 other call sites read the chain synchronously — and
    /// making the whole call chain async to reach it would ripple through every one of them. A
    /// *synchronous* `XMLHttpRequest` is the one browser primitive that can return bytes from a
    /// call that must return before the function does; the Bevy `AssetReader` path
    /// (`benilla-assets`) is already `async` end to end, so it uses `fetch` instead (no such
    /// constraint there).
    ///
    /// `range` asks for `bytes=start-end` (inclusive, as HTTP wants it). **The status comes back to
    /// the caller** because asking for a range and being given the whole object is not an error —
    /// it is a static host's measured behaviour on a cache miss, and the pack reader is built to use
    /// the full body when that happens (`crate::webpack`). A caller that cannot tell 206 from 200
    /// would either mis-slice the body or throw away a response it already paid for.
    ///
    /// A sync XHR's `responseType` is stuck at the default `""` (text), so an `arraybuffer`
    /// response type — the normal way to get binary out of an XHR — isn't available here. The
    /// standard workaround, used below, is `override_mime_type("text/plain; charset=x-user-defined")`:
    /// it forces the browser to decode the response body as one code unit (0x00-0xFF) per byte
    /// instead of guessing UTF-8 and mangling anything non-ASCII into U+FFFD, so the response
    /// round-trips arbitrary binary losslessly through `& 0xFF`, except a body that opens with a
    /// UTF-8 BOM: the decode's BOM sniff reads it as UTF-8 over the charset, and
    /// [`binary_string_bytes`] re-encodes it.
    ///
    /// The unpacking of that binary string happens in JS ([`binary_string_bytes`]), not here. What
    /// still reaches this function is boot and the character-skin composite — the reads that did not
    /// move to `Chain::try_read` — so it is worth what it costs past the round trip being small.
    pub fn fetch_sync_range(
        url: &str,
        range: Option<(u64, u64)>,
    ) -> std::io::Result<(u16, Vec<u8>)> {
        let xhr = XmlHttpRequest::new().map_err(js_err)?;
        xhr.open_with_async("GET", url, false).map_err(js_err)?;
        if let Some((start, end)) = range {
            xhr.set_request_header("Range", &format!("bytes={start}-{end}"))
                .map_err(js_err)?;
        }
        xhr.override_mime_type("text/plain; charset=x-user-defined")
            .map_err(js_err)?;
        xhr.send().map_err(js_err)?;
        let status = xhr.status().map_err(js_err)?;
        match status {
            200 | 206 => {
                // `response()`, not `response_text()`: the bytes come back as a JS *binary string*
                // and [`binary_string_bytes`] unpacks it inside JS in one pass. `response_text()`
                // would first transcode all of it to UTF-8 across the wasm boundary — every byte
                // above 0x7F becoming two — only for us to decode it back, which is three passes
                // and two allocations of a body we already had.
                let value = xhr.response().map_err(js_err)?;
                Ok((status, binary_string_bytes(&value)))
            }
            404 => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                url.to_string(),
            )),
            status => Err(std::io::Error::other(format!("{url}: HTTP {status}"))),
        }
    }

    /// Unpack an XHR **binary string** (one code unit per byte, the `x-user-defined` charset's
    /// contract) into the bytes it stands for, with the per-character loop running *in JS*.
    ///
    /// Doing it in Rust — `response_text()` then `chars().map()` — costs three passes over the body
    /// and two allocations of it: wasm-bindgen transcodes the JS string to UTF-8 to hand it across
    /// the boundary (a body of near-random bytes grows ~1.5x once every unit above 0x7F becomes
    /// two), Rust copies that into a `String`, and `chars()` decodes it back. The page's
    /// `unpackBinary` does one `charCodeAt` loop into a `Uint8Array` inside the engine, and the
    /// boundary then sees a single typed-array copy of exactly `len` bytes.
    ///
    /// It is reached the way [`trace`] reaches `__wenilla_boottrace`: a plain global, looked up
    /// once. **A page that ships no `packs.js` still works** — the Rust loop below is the fallback,
    /// and it is what this function was before the page grew a helper.
    fn binary_string_bytes(value: &JsValue) -> Vec<u8> {
        use wasm_bindgen::JsCast;
        thread_local! {
            static UNPACK: std::cell::OnceCell<Option<js_sys::Function>> =
                const { std::cell::OnceCell::new() };
        }
        let unpacked = UNPACK.with(|cell| {
            let f = cell.get_or_init(|| {
                let window = web_sys::window()?;
                let packs = js_sys::Reflect::get(&window, &"__wenilla_packs".into()).ok()?;
                js_sys::Reflect::get(&packs, &"unpackBinary".into())
                    .ok()?
                    .dyn_into::<js_sys::Function>()
                    .ok()
            });
            f.as_ref()?
                .call1(&JsValue::NULL, value)
                .ok()?
                .dyn_into::<js_sys::Uint8Array>()
                .ok()
                .map(|a| a.to_vec())
        });
        unpacked.unwrap_or_else(|| {
            let s = value.as_string().unwrap_or_default();
            // A BOM-sniffed UTF-8 body, as `unpackBinary` reads it: the BOM and the text.
            if s.chars()
                .any(|c| c > '\u{7f}' && !('\u{f780}'..='\u{f7ff}').contains(&c))
            {
                return ["\u{feff}", &s].concat().into_bytes();
            }
            s.chars().map(|c| (c as u32 & 0xff) as u8).collect()
        })
    }

    /// [`fetch_sync_range`] for a whole object — the shape every pre-container caller wanted.
    pub fn fetch_sync(url: &str) -> std::io::Result<Vec<u8>> {
        fetch_sync_range(url, None).map(|(_, body)| body)
    }

    fn js_err(e: JsValue) -> std::io::Error {
        std::io::Error::other(format!("{e:?}"))
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::{
    close_hd_layer, close_reforged_layer, close_turtle_layer, console_warn, data_base, fetch_sync,
    fetch_sync_range, hd_data_base, mount_hd_layer, mount_reforged_layer, mount_turtle_layer,
    reforged_data_base, trace, trace_hd, trace_reforged, trace_turtle, turtle_data_base,
};

#[cfg(test)]
mod priority_tests {
    use super::*;

    /// The published focus is one global, and these tests set it — so they run under one lock
    /// rather than against each other. (`cargo test` threads a module's tests in parallel by
    /// default; two of these racing would be a flake that only ever shows up in CI.)
    static FOCUS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `body` with a focus published, and leave nothing behind for the next test.
    fn with_focus<T>(map: &str, tile: (i32, i32), lean: (i32, i32), body: impl FnOnce() -> T) -> T {
        let _guard = FOCUS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_terrain_focus(map, tile, lean);
        let out = body();
        clear_terrain_focus();
        out
    }

    /// The lane order itself: every tier is distinct, terrain outranks model art, and the whole
    /// terrain ladder sits *inside* the collision lane — the range `packs.js` reserves slots
    /// against with one `priority <= PRIORITY_GEOMETRY` test.
    #[test]
    fn the_ladder_is_ordered_and_terrain_is_inside_the_collision_lane() {
        const {
            assert!(PRIORITY_TERRAIN_FOCUS < PRIORITY_TERRAIN_NEAR);
            assert!(PRIORITY_TERRAIN_NEAR < PRIORITY_TERRAIN);
            assert!(PRIORITY_TERRAIN < PRIORITY_GEOMETRY);
            assert!(PRIORITY_GEOMETRY < PRIORITY_NORMAL);
        }
    }

    /// The families that carry collision geometry or model bounds are the collision lane, and
    /// nothing else is. The one worth pinning is the WMO **group** file — it is where the collider
    /// faces actually live, and it is the easy one to miss, since it is `..._014.wmo` rather than
    /// an extension of its own.
    #[test]
    fn geometry_names_outrank_everything_else() {
        for name in [
            r"World\wmo\Azeroth\Buildings\Stormwind\Stormwind.WMO",
            r"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_014.wmo",
            r"Creature\Wolf\Wolf.m2",
        ] {
            assert_eq!(warm_priority(name), PRIORITY_GEOMETRY, "{name}");
        }
        for name in [
            r"Sound\Ambience\Forest.wav",
            r"DBFilesClient\Spell.dbc",
            r"Interface\FrameXML\UIParent.lua",
            // No extension at all, and a dotted directory holding one — neither is a geometry name,
            // and the second is the one a naive "text after the last dot" would get wrong.
            "listfile",
            r"Interface\v1.2\readme",
            // A directory whose name merely *starts* with the tileset tree's is not that tree.
            r"TilesetExtras\Elwynn\Grass.blp",
        ] {
            assert_eq!(warm_priority(name), PRIORITY_NORMAL, "{name}");
        }
    }

    /// Terrain outranks the model art, and the map's index outranks the tiles it unblocks.
    #[test]
    fn terrain_outranks_the_art_standing_on_it() {
        assert_eq!(
            warm_priority(r"World\Maps\Azeroth\Azeroth.wdt"),
            PRIORITY_TERRAIN_FOCUS
        );
        assert_eq!(
            warm_priority(r"World\Maps\Azeroth\Azeroth.wdl"),
            PRIORITY_TERRAIN
        );
        // A layer texture gates the tile asset that owns it, so it is terrain and not art — in
        // both the authored spelling and the lowercased forward-slash one `AdtLoader` asks with.
        for name in [
            r"Tileset\Elwynn\ElwynnGrass01.blp",
            "tileset/elwynn/elwynngrass01.blp",
        ] {
            assert_eq!(warm_priority(name), PRIORITY_TERRAIN, "{name}");
        }
        // With no focus published, every tile is the plain terrain tier — better than the art it
        // holds up, and no claim about a distance nobody has stated.
        assert_eq!(
            warm_priority(r"World\Maps\Azeroth\Azeroth_32_48.adt"),
            PRIORITY_TERRAIN
        );
    }

    /// **The tiers this whole change exists for**: the tile under the focus and the one the body
    /// is nearest to stepping onto go first, the other seven neighbours next, and the rest of the
    /// residency window behind them.
    #[test]
    fn the_focus_tile_and_the_one_being_approached_go_first() {
        with_focus("Azeroth", (32, 48), (33, 48), || {
            let p = |t: &str| warm_priority(&format!(r"World\Maps\Azeroth\{t}.adt"));
            assert_eq!(p("Azeroth_32_48"), PRIORITY_TERRAIN_FOCUS);
            assert_eq!(p("Azeroth_33_48"), PRIORITY_TERRAIN_FOCUS);
            // The other seven of the eight neighbours, corners included.
            assert_eq!(p("Azeroth_31_48"), PRIORITY_TERRAIN_NEAR);
            assert_eq!(p("Azeroth_33_49"), PRIORITY_TERRAIN_NEAR);
            assert_eq!(p("Azeroth_31_47"), PRIORITY_TERRAIN_NEAR);
            // Two tiles out — inside the residency window at 777 farclip, still terrain, and
            // behind the neighbourhood.
            assert_eq!(p("Azeroth_34_48"), PRIORITY_TERRAIN);
            assert_eq!(p("Azeroth_32_50"), PRIORITY_TERRAIN);
            // Still ahead of the buildings standing on any of them.
            assert!(PRIORITY_TERRAIN < warm_priority(r"World\wmo\a\b.wmo"));
        });
    }

    /// A tile of a *different* map never borrows this map's neighbourhood — the case a worldport
    /// puts on the wire for a few frames, where the two maps' 64×64 grids name the same squares.
    #[test]
    fn another_maps_tiles_do_not_ride_this_maps_focus() {
        with_focus("Azeroth", (32, 48), (33, 48), || {
            assert_eq!(
                warm_priority(r"World\Maps\Kalimdor\Kalimdor_32_48.adt"),
                PRIORITY_TERRAIN
            );
            // Case is the container's business, not the caller's: the streamer's URL is authored
            // however the DBC spelled the directory.
            assert_eq!(
                warm_priority("world/maps/azeroth/AZEROTH_32_48.adt"),
                PRIORITY_TERRAIN_FOCUS
            );
        });
    }

    /// A tile name that will not parse falls back **down** the ladder, never up: an unrecognised
    /// name must not outrank the tile the body is standing on.
    #[test]
    fn an_unparseable_tile_name_falls_back_to_the_plain_terrain_tier() {
        with_focus("Azeroth", (32, 48), (33, 48), || {
            for name in [
                r"World\Maps\Azeroth\Azeroth_32.adt",
                r"World\Maps\Azeroth\Azeroth_x_48.adt",
                r"World\Maps\Azeroth\.adt",
            ] {
                assert!(
                    warm_priority(name) >= PRIORITY_TERRAIN,
                    "{name} must not outrank the focus tile"
                );
            }
            // A loud extension is still this map's tile, though — the container's case rules are
            // not the caller's problem.
            assert_eq!(warm_priority("Azeroth_32_48.ADT"), PRIORITY_TERRAIN_FOCUS);
        });
    }

    /// A released world publishes nothing, and the tiers go quiet with it.
    #[test]
    fn clearing_the_focus_returns_every_tile_to_the_plain_tier() {
        with_focus("Azeroth", (32, 48), (33, 48), || {});
        assert_eq!(terrain_focus(), None);
        assert_eq!(
            warm_priority(r"World\Maps\Azeroth\Azeroth_32_48.adt"),
            PRIORITY_TERRAIN
        );
    }

    /// A WMO group folds onto its root and case and separators do not split a stem, so marking the
    /// root a GameObject names covers the group files its loader reads next.
    #[test]
    fn a_model_stem_folds_groups_onto_their_root() {
        let root = model_stem(r"World\wmo\Azeroth\Buildings\HumanTwoStory\HumanTwoStory.wmo");
        assert_eq!(
            root,
            r"world\wmo\azeroth\buildings\humantwostory\humantwostory"
        );
        assert_eq!(
            model_stem("world/wmo/azeroth/buildings/humantwostory/humantwostory_002.wmo"),
            root
        );
        assert_eq!(
            model_stem(r"Creature\Wolf\Wolf.mdx"),
            model_stem(r"creature\wolf\wolf.m2")
        );
        // Not a group number: the name is its own stem.
        assert_ne!(
            model_stem(r"World\wmo\X\Tower_A.wmo"),
            model_stem(r"World\wmo\X\Tower.wmo")
        );
    }

    /// An entity's model rises above the scenery queue; an unmarked model and a texture do not.
    #[test]
    fn a_marked_entity_model_outranks_the_scenery() {
        let root = r"World\wmo\Test\UrgentOnly\UrgentOnly.wmo";
        assert_eq!(warm_priority(root), PRIORITY_GEOMETRY);
        mark_urgent_model(root);
        assert_eq!(warm_priority(root), PRIORITY_TERRAIN_NEAR);
        assert_eq!(
            warm_priority(r"world\wmo\test\urgentonly\urgentonly_004.wmo"),
            PRIORITY_TERRAIN_NEAR,
            "its group files follow the root"
        );
        assert_eq!(
            warm_priority(r"World\wmo\Test\UrgentOnly\Other.wmo"),
            PRIORITY_GEOMETRY
        );
        assert_eq!(
            warm_priority(r"World\wmo\Test\UrgentOnly\UrgentOnly.blp"),
            PRIORITY_NORMAL,
            "only model files are lifted"
        );
    }
}
