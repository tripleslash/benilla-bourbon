//! Serving a BENPACK container (`benilla-pack build`) as plain static files at `/data/*` — the
//! local stand-in for the object store the hosted client reads from.
//!
//! Unlike [`crate::data`], which resolves names out of a live MPQ chain, this serves a directory
//! exactly as a dumb host would: no logic, no name decoding, just files. That is the point. The
//! hosted client talks to a static asset host; being able to point it at this instead,
//! locally, is what makes the pack path testable before a site exists.
//!
//! ## `--no-ranges`
//!
//! `ServeDir` honours HTTP `Range` and answers 206. A static host, measured, does not: its edge only
//! slices ranges for objects it has stored, and the default `max-age=0, must-revalidate` means it
//! never stores one, so a range request comes back as a full 200. The client is built to work
//! either way and to detect which it is talking to — but a fallback that is never exercised is a
//! fallback that does not work. [`router`]'s `allow_ranges: false` reproduces a Range-less host's
//! behaviour exactly (strip the request's `Range`, answer 200 with the whole object) so the
//! Range-less path can be run, and demonstrated, on a laptop.

use std::path::Path;

use axum::http::{HeaderValue, Request};
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;

/// Mount a container directory at `/data`.
///
/// `Cache-Control` matches what the hosted `_headers` sets for `/data/p/*`: packs are content the
/// game never changes, and their names derive from a content-addressed id, so `immutable` is
/// honest. It is also the header that makes an edge *store* the object — the precondition for
/// range slicing — so serving it here keeps local and hosted behaviour aligned.
pub fn router(packs: &Path, allow_ranges: bool) -> axum::Router {
    mount(packs, allow_ranges, "/data")
}

/// The optional updated-character-model layer's container, at `/data-hd` — the same tree shape,
/// the same headers, the same everything. It is a second mount rather than a subdirectory of the
/// first because that is how it is hosted (`scripts/stage-packs.sh`, `web/_headers`), and a local
/// host whose URL layout differs from the site's is a local host that cannot reproduce its bugs.
pub fn hd_router(packs: &Path, allow_ranges: bool) -> axum::Router {
    mount(packs, allow_ranges, "/data-hd")
}

/// The Turtle WoW 1.18.1 layer's container at `/data-turtle`, on the HD layer's terms: the client
/// asks for it only while it is on a realm that advertises 1.18.1.
pub fn turtle_router(packs: &Path, allow_ranges: bool) -> axum::Router {
    mount(packs, allow_ranges, "/data-turtle")
}

/// The reforged art layer's container at `/data-reforged`, on the Turtle layer's terms: the client
/// asks for it only with the Turtle layer, while the player's setting has it on.
pub fn reforged_router(packs: &Path, allow_ranges: bool) -> axum::Router {
    mount(packs, allow_ranges, "/data-reforged")
}

fn mount(packs: &Path, allow_ranges: bool, at: &str) -> axum::Router {
    let serve = ServeDir::new(packs).append_index_html_on_directories(false);
    let inner = axum::Router::new().fallback_service(
        tower::ServiceBuilder::new()
            .layer(SetResponseHeaderLayer::overriding(
                axum::http::header::CACHE_CONTROL,
                HeaderValue::from_static("public, max-age=31536000, immutable"),
            ))
            .service(serve),
    );
    let inner = if allow_ranges {
        inner
    } else {
        inner.layer(axum::middleware::from_fn(strip_range))
    };
    axum::Router::new().nest_service(at, inner)
}

/// Drop the request's `Range` header, so the response is always the whole object with a 200 — what
/// A static host does today. Emulating the *absence* of a feature is the only way to keep the client's
/// fallback honest.
async fn strip_range(
    mut req: Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    req.headers_mut().remove(axum::http::header::RANGE);
    next.run(req).await
}
