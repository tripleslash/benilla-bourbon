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
/// So the geometry lane goes first. What is in it is exactly the set of files that carry
/// **collision geometry or model bounds**:
///
/// - `.adt` — the terrain tile: its MCNK heightfield IS the ground collider.
/// - `.wdt` / `.wdl` — the tile-existence index the streamer gates on, and the coarse whole-map
///   heightmap the provisional ground floor is built from (`benilla_world`'s
///   `terrain_stream::provisional`).
/// - `.wmo` — building roots *and* their group files, which hold the MOPY/MOVI faces both per-WMO
///   collider bakes come from.
/// - `.m2` / `.mdx` — the model header (its bounding box) and the collision hull that follows it.
///
/// Everything else — textures, sounds, DBCs, UI art — is [`PRIORITY_NORMAL`]. Nothing is *starved*
/// by this: the queue is work-conserving, so with no geometry outstanding the art has the whole
/// six-wide pool exactly as before. The extension law lives here, in the one place that already
/// knows what a name means; `packs.js` only orders what it is told.
pub fn warm_priority(name: &str) -> u8 {
    // The extension, or nothing: the last dot, refused when a path separator follows it (a
    // directory with a dot in its name, holding a file that has no extension of its own).
    let ext = match name.rsplit_once('.') {
        Some((_, ext)) if !ext.contains(['\\', '/']) => ext,
        _ => return PRIORITY_NORMAL,
    };
    const GEOMETRY: [&str; 6] = ["adt", "wmo", "m2", "mdx", "wdt", "wdl"];
    if GEOMETRY.iter().any(|g| ext.eq_ignore_ascii_case(g)) {
        PRIORITY_GEOMETRY
    } else {
        PRIORITY_NORMAL
    }
}

/// Collision geometry and model bounds — the reads a body's floor depends on. See
/// [`warm_priority`].
pub const PRIORITY_GEOMETRY: u8 = 0;
/// Everything else. See [`warm_priority`].
pub const PRIORITY_NORMAL: u8 = 1;

#[cfg(test)]
mod priority_tests {
    use super::*;

    /// The families that carry collision geometry or model bounds are the geometry lane, and
    /// nothing else is. The one worth pinning is the WMO **group** file — it is where the collider
    /// faces actually live, and it is the easy one to miss, since it is `..._014.wmo` rather than
    /// an extension of its own.
    #[test]
    fn geometry_names_outrank_everything_else() {
        for name in [
            r"World\Maps\Azeroth\Azeroth_32_48.adt",
            r"World\wmo\Azeroth\Buildings\Stormwind\Stormwind.WMO",
            r"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_014.wmo",
            r"World\Maps\Azeroth\Azeroth.wdt",
            r"World\Maps\Azeroth\Azeroth.wdl",
            r"Creature\Wolf\Wolf.m2",
        ] {
            assert_eq!(warm_priority(name), PRIORITY_GEOMETRY, "{name}");
        }
        for name in [
            r"Tileset\Elwynn\ElwynnGrass01.blp",
            r"Sound\Ambience\Forest.wav",
            r"DBFilesClient\Spell.dbc",
            r"Interface\FrameXML\UIParent.lua",
            // No extension at all, and a dotted directory holding one — neither is a geometry name,
            // and the second is the one a naive "text after the last dot" would get wrong.
            "listfile",
            r"Interface\v1.2\readme",
        ] {
            assert_eq!(warm_priority(name), PRIORITY_NORMAL, "{name}");
        }
    }
}

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
    /// round-trips arbitrary binary losslessly through `& 0xFF`.
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
            value
                .as_string()
                .unwrap_or_default()
                .chars()
                .map(|c| (c as u32 & 0xff) as u8)
                .collect()
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
pub use wasm::{data_base, fetch_sync, fetch_sync_range, trace};
