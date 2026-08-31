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
//! and only compile their bodies on `wasm32`.

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
    /// instead of guessing UTF-8 and mangling anything non-ASCII into U+FFFD, so `response_text()`
    /// round-trips arbitrary binary losslessly through `& 0xFF`.
    ///
    /// That decode is `chars().map()` over the whole body, which is why the pack layout keeps
    /// bundled objects near 512 KiB and leaves multi-megabyte SINGLETONs to the async lane.
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
                let text = xhr.response_text().map_err(js_err)?.unwrap_or_default();
                Ok((
                    status,
                    text.chars().map(|c| (c as u32 & 0xff) as u8).collect(),
                ))
            }
            404 => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                url.to_string(),
            )),
            status => Err(std::io::Error::other(format!("{url}: HTTP {status}"))),
        }
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
