//! Getting the container's two side files — `index.bpi` and `listfile.txt` — into wasm memory.
//!
//! Both have the same shape of answer: **the page usually already has them**, because `web/boot.js`
//! fetches them while the wasm module is still downloading, and a synchronous XHR is only the
//! fallback for a page that did not. That ordering is not an optimisation detail, it is what keeps
//! the index off the sync path: `crate::web::fetch_sync_range` is a *blocking* fetch with no async
//! seam at `Chain::open` to move it to, and the index is 1.6 MB of it.
//!
//! The page hands them over as ordinary properties on `window.__wenilla_packs` (a `Uint8Array` for
//! the index, a string for the listfile), so copying them in costs one `Uint8Array::to_vec` and no
//! parse at all — the index is binary-searched in place from there on.
//!
//! **Which properties** is a parameter now, not a constant, because two containers can be open at
//! once: the base corpus arrives as `index`/`listfile` and the updated-character-model layer as
//! `hdIndex`/`hdListfile`. One object, two slots — see [`crate::webpack::PackStore::open_hd`].

use anyhow::{anyhow, Context, Result};
use wasm_bindgen::JsCast;

/// `index.bpi`'s bytes: from the page if it prefetched them, else one blocking fetch.
///
/// A failure here is fatal to the client and should be: the index is the *only* way to resolve a
/// name to bytes now that the per-name HTTP route is gone, so there is nothing to degrade to.
pub fn index_bytes(base: &str, property: &str) -> Result<Vec<u8>> {
    if let Some(bytes) = super::resident::property(property)
        .and_then(|v| v.dyn_into::<js_sys::Uint8Array>().ok())
        .map(|a| a.to_vec())
    {
        return Ok(bytes);
    }
    let url = format!("{base}/index.bpi");
    crate::web::fetch_sync(&url)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| {
            format!("fetching {url} — is this origin serving a `benilla-pack build` container?")
        })
}

/// The listfile's names, or `None` if the container ships none.
///
/// `None` is a real state, not an error path: the listfile exists only to answer `Chain::list`, the
/// one web caller of which is `load_macro_icons` (the macro icon picker enumerates
/// `Interface\Icons\`). A container without one still plays; the picker is just empty. So this is
/// deliberately lossy about *why* it failed — there is nothing a caller could do differently.
pub fn listfile_names(base: &str, property: &str) -> Option<Vec<String>> {
    let text = match super::resident::property(property).and_then(|v| v.as_string()) {
        Some(text) => text,
        None => {
            let bytes = crate::web::fetch_sync(&format!("{base}/listfile.txt")).ok()?;
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };
    Some(
        text.split(['\r', '\n'])
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect(),
    )
}
