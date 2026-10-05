//! Static hosting for the browser build's output directory — everything `trunk build --release`
//! writes to `web/dist/`: the page and its scripts, the wasm bundle, `sw.js`, `packworker.js` and
//! its `wenilla-packdecode` pair, and `config.json`.
//!
//! Everything under `--www` gets `no-cache` rather than the immutable long cache the `/data/*`
//! route uses, and every file there wants it: the bundle changes on every rebuild but keeps the
//! same filename (no content-hash), so a long cache would pin a stale client against a server that
//! has already moved on; `sw.js` held would be a service worker that cannot be replaced; and
//! `config.json` held would be a client that cannot be repointed. Dev-cycle friction the `/data/*`
//! files don't have (those are content the game itself never changes).
//!
//! It also reproduces the site's **cross-origin isolation** headers. That is not a nicety: the
//! threaded browser build's linear memory is a `SharedArrayBuffer`, and a page that is not
//! cross-origin isolated cannot construct one — so without these two headers the wasm module does
//! not instantiate at all and the failure is a bare `WebAssembly.Memory` TypeError with nothing in
//! it naming the cause. `web/_headers` sets exactly this pair on `/*` for the static-host setup;
//! this is the same rule for `--www` and for `trunk serve` (`Trunk.toml`'s `[serve] headers`), so
//! a bug reproduces locally instead of only in production.
//!
//! No `Cross-Origin-Resource-Policy` here, and none on the `/data/*` routers either:
//! `require-corp` gates *cross-origin* subresources, and every byte this client loads — the packs,
//! the index, the listfile, the worker scripts — is same-origin, which it allows unconditionally.

use std::path::Path;

use axum::http::{header::HeaderName, HeaderValue};
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;

/// `Cross-Origin-Opener-Policy` — severs the page from any opener, half of what the browser wants
/// before it will hand out `SharedArrayBuffer`. Must match `web/_headers`.
const COOP: (&str, &str) = ("cross-origin-opener-policy", "same-origin");
/// `Cross-Origin-Embedder-Policy` — the other half: every cross-origin subresource must opt in.
/// This page loads none, so the requirement is free. Must match `web/_headers`.
const COEP: (&str, &str) = ("cross-origin-embedder-policy", "require-corp");
/// `Document-Policy: js-profiling` — lets the page start the JS Self-Profiling API
/// (`new Profiler(...)`), whose samples carry wasm frames: a frame-thread profile of the browser
/// build from a plain console, on the local host only. The site does not set it.
const DOC_POLICY: (&str, &str) = ("document-policy", "js-profiling");

pub fn router(www: &Path) -> axum::Router {
    let no_cache = HeaderValue::from_static("no-cache");
    // Precompressed, never on the fly: the wasm is ~90 MB, and brotli-ing that per request is
    // seconds of CPU for every page load.
    //
    // **Trunk does not write the `.br`/`.gz` siblings** the way `scripts/web-build.sh` did, and
    // that is deliberate on the site path — A static host compresses on the fly at the edge, so
    // shipping precompressed copies would just double the object count. These two lines stay
    // because they cost nothing when the siblings are absent (a browser then gets the plain file)
    // and still pay off for anyone serving `web/dist` to real players over a real link, who can
    // `brotli -q 5` the bundle once by hand.
    let serve = ServeDir::new(www)
        .append_index_html_on_directories(true)
        .precompressed_br()
        .precompressed_gzip();
    axum::Router::new().fallback_service(
        tower::ServiceBuilder::new()
            .layer(SetResponseHeaderLayer::overriding(
                axum::http::header::CACHE_CONTROL,
                no_cache,
            ))
            .layer(SetResponseHeaderLayer::overriding(
                HeaderName::from_static(COOP.0),
                HeaderValue::from_static(COOP.1),
            ))
            .layer(SetResponseHeaderLayer::overriding(
                HeaderName::from_static(COEP.0),
                HeaderValue::from_static(COEP.1),
            ))
            .layer(SetResponseHeaderLayer::overriding(
                HeaderName::from_static(DOC_POLICY.0),
                HeaderValue::from_static(DOC_POLICY.1),
            ))
            .service(serve),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two headers this module exists to add are the ones `web/_headers` sets on the site.
    /// Read them out of that file rather than restating them, so the dev host and the site
    /// cannot drift apart silently — the failure mode if they do is "threads work in production
    /// and nowhere else", which costs an upload to notice.
    #[test]
    fn the_isolation_headers_match_the_sites_headers_file() {
        let headers = include_str!("../../../web/_headers");
        for (name, value) in [COOP, COEP] {
            // `_headers` writes them in canonical case; compare case-insensitively on the name and
            // exactly on the value.
            let found = headers.lines().any(|line| {
                let line = line.trim();
                // That file discusses these header names at length in its comments, and a comment
                // is not a rule — matching one would let the real rule be deleted without this
                // test noticing.
                if line.starts_with('#') {
                    return false;
                }
                match line.split_once(':') {
                    Some((k, v)) => k.trim().eq_ignore_ascii_case(name) && v.trim() == value,
                    None => false,
                }
            });
            assert!(found, "web/_headers is missing `{name}: {value}`");
        }
    }
}
