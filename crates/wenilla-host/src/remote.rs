//! Forwarding `/data/*`, `/data-hd/*`, `/data-turtle/*` and `/data-reforged/*` to a hosted static tree (`--packs-remote`), for working
//! on the client without a local container.
//!
//! The page cannot simply fetch the site cross-origin: it reads its containers from its own
//! origin, and the site answers `Cross-Origin-Resource-Policy: same-site`, which a page on
//! `127.0.0.1` is not. So the host relays the requests instead, with `Range` and the validators
//! passed through, which keeps the client's own range detection talking to the real edge.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response};

/// Request headers worth passing upstream; anything else (cookies, `Host`) stays here.
const FORWARD_REQ: [HeaderName; 4] = [
    header::RANGE,
    header::IF_NONE_MATCH,
    header::IF_MODIFIED_SINCE,
    header::ACCEPT_ENCODING,
];

/// Response headers copied back. The edge's own COEP/COOP/CORP are dropped on purpose: this origin
/// sets its own, and a forwarded `same-site` would name the wrong site.
const FORWARD_RESP: [HeaderName; 7] = [
    header::CONTENT_TYPE,
    header::CONTENT_LENGTH,
    header::CONTENT_RANGE,
    header::CONTENT_ENCODING,
    header::ACCEPT_RANGES,
    header::ETAG,
    header::CACHE_CONTROL,
];

#[derive(Clone)]
struct Upstream {
    client: reqwest::Client,
    base: String,
}

/// Mount `/data`, `/data-hd`, `/data-turtle` and `/data-reforged`, each forwarded to the same path under `base`
/// (e.g. `https://wow.example.com`).
pub fn router(base: &str) -> axum::Router {
    let state = Upstream {
        // No automatic decompression: bodies go back byte-for-byte with their `Content-Encoding`.
        client: reqwest::Client::builder()
            .no_gzip()
            .no_brotli()
            .build()
            .expect("building the HTTP client"),
        base: base.trim_end_matches('/').to_string(),
    };
    axum::Router::new()
        .route("/data/{*rest}", axum::routing::any(forward))
        .route("/data-hd/{*rest}", axum::routing::any(forward))
        .route("/data-turtle/{*rest}", axum::routing::any(forward))
        .route("/data-reforged/{*rest}", axum::routing::any(forward))
        .with_state(state)
}

async fn forward(State(up): State<Upstream>, req: Request) -> Response {
    let path = req
        .uri()
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let url = format!("{}{}", up.base, path);
    let mut out = up.client.request(req.method().clone(), &url);
    for name in &FORWARD_REQ {
        if let Some(v) = req.headers().get(name) {
            out = out.header(name, v);
        }
    }
    let resp = match out.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(%url, error = %e, "remote data request failed");
            return (StatusCode::BAD_GATEWAY, e.to_string()).into_response();
        }
    };
    let mut builder = Response::builder().status(resp.status());
    for name in &FORWARD_RESP {
        if let Some(v) = resp.headers().get(name) {
            builder = builder.header(name, v);
        }
    }
    builder
        .body(Body::from_stream(resp.bytes_stream()))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}
