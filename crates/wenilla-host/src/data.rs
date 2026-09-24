//! `GET`/`HEAD /data/{*name}` and `GET /data/__index` — the Data URL scheme Lane A's wasm
//! `Chain` fetches against verbatim (see the plan's "Shared interfaces"): the browser build has
//! no filesystem, so every asset load the client makes becomes one of these requests, answered
//! straight from the same [`Chain`] the native client reads off disk.
//!
//! The name in the URL is percent-decoded here, not left to axum's own path-segment decoding —
//! `Chain` names use `\` as their separator (`Interface\Glues\...`), and the wildcard capture
//! would otherwise hand us a single decoded segment with no way to tell an encoded `/` (a literal
//! path separator in some other scheme) apart from an encoded `\`. Decoding the raw tail
//! ourselves and mapping `/` -> `\` afterward matches exactly what the client's `encode_name`
//! produces on the way out.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{header, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use benilla_formats::Chain;

/// Router state: the opened patch chain. `Chain::read`/`list` are `&self` and lock-free (see the
/// chain module doc), so cloning the `Arc` per request is the whole synchronization story.
#[derive(Clone)]
pub struct DataState {
    pub chain: Arc<Chain>,
    /// The read log, when `--log-reads` armed one: every name the client asks for, in **first-need
    /// order**, deduplicated. See [`ReadLog`].
    pub reads: Option<Arc<ReadLog>>,
}

/// **The boot read set, recorded server-side** — the input to `web/boot-manifest.json`, captured
/// without a browser console and without a person copying an array out of it.
///
/// The manifests are the container's strongest locality signal (`benilla_packer::sidecars`), and
/// until now the only way to refresh one was `?boottrace=1`: arm a JS array, boot, then read the
/// deduplicated result out of devtools by hand. That works and it is fiddly, and it only works
/// against a container — the very thing the manifest is an input to.
///
/// Against `--data` the host sees something better: every read the client makes arrives here as an
/// HTTP request carrying the chain name, in the order the client wanted it. That IS the manifest.
/// So `--log-reads <file>` writes one, in the exact JSON shape `benilla-pack` parses, and a capture
/// becomes: run the host, boot the client, log in, stop the host.
///
/// **First-need order, deduplicated**, which is the manifest's own contract: the grouping pass uses
/// the set, but a human reading the file wants it in the order a boot actually walks. A `Vec` plus
/// a `HashSet` rather than an ordered map because the file is written once, at shutdown, and 400
/// names do not need a data structure.
pub struct ReadLog {
    path: PathBuf,
    seen: Mutex<(std::collections::HashSet<String>, Vec<String>)>,
}

impl ReadLog {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            seen: Mutex::new((std::collections::HashSet::new(), Vec::new())),
        }
    }

    /// Record one read. Keyed the way MPQ compares names — case-insensitive, `/` equivalent to
    /// `\\` — so a file the client asks for twice under two spellings appears once, at its
    /// first need.
    ///
    /// **Stored backslash-separated, whatever spelling arrived.** The chain accepts either and
    /// `benilla_mpq::name_key` hashes them identically, so a call site that happens to spell one
    /// read `Interface/FrameXML/Fonts.xml` is not a bug — but a *manifest* carrying that spelling
    /// is, because `benilla-app/tests/boot_manifest.rs` holds the file to raw chain names and would
    /// reject it. Normalising here rather than asking 60 call sites to agree puts the rule where the
    /// file is written. (Observed: exactly one such name in a real capture.)
    pub(crate) fn note(&self, name: &str) {
        let name = name.replace('/', "\\");
        let mut guard = match self.seen.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.0.insert(name.to_ascii_lowercase()) {
            guard.1.push(name);
        }
    }

    /// Where the manifest lands — for the log line that tells a capturing session which file to
    /// look at without making them re-read their own command.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.seen.lock().map(|g| g.1.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write the manifest. The shape is `benilla_packer::sidecars::manifest_names`'s: a `names`
    /// array of backslash-separated chain names, with `\\` escaped — the one escape those names
    /// carry. Hand-rolled rather than through `serde_json` for the same reason the reader is:
    /// this is one array of strings, and the escape rule for it is one line.
    pub fn write(&self) -> std::io::Result<usize> {
        let guard = match self.seen.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut out = String::from("{\n\"version\": 1,\n\"names\": [\n");
        for (i, name) in guard.1.iter().enumerate() {
            out.push('"');
            out.push_str(&name.replace('\\', "\\\\"));
            out.push('"');
            if i + 1 < guard.1.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("]\n}\n");
        std::fs::write(&self.path, out)?;
        Ok(guard.1.len())
    }
}

/// Build the `/data/*` router. Kept separate from `static_site`'s and `ws`'s so `main.rs` can
/// merge them with `Router::merge` and each test file can stand its half up alone.
pub fn router(chain: Arc<Chain>) -> Router {
    router_logged(chain, None)
}

/// [`router`] with a read log armed — `--log-reads`. See [`ReadLog`].
pub fn router_logged(chain: Arc<Chain>, reads: Option<Arc<ReadLog>>) -> Router {
    Router::new()
        .route("/data/__index", get(index))
        // axum's matchit picks the more specific literal route above over this wildcard on its
        // own — registration order here doesn't matter.
        .route("/data/{*name}", get(file).head(file))
        .with_state(DataState { chain, reads })
}

/// The **trace sink**: `POST /__boottrace`, a JSON array of chain names the page collected.
///
/// The read log's other half, and the half that actually works against a container. `--data`'s
/// per-name route carries the name in the URL, so the host can record the read set by watching
/// requests — but the current client does not use that route at all: it reads `index.bpi` and
/// `p/<hi>/<id>.bpk`, and a pack id names a few hundred files at once. Against the mode everyone
/// actually runs, the host can see *what was fetched* and never *what was wanted*.
///
/// The client can. `?boottrace=1` already fills `window.__wenilla_boottrace` with every name
/// `Chain::read`/`try_read` asked for, in first-need order (`benilla_formats::web::trace`), and
/// until now a person copied that array out of devtools by hand. This route is the page handing it
/// over instead: `web/boot.js` posts it once the client reports ready, and the host writes the same
/// `boot-manifest.json` it would have written from the per-name route.
///
/// Armed only by `--log-reads`, so a host that was not asked to capture one has no such endpoint
/// and a stray POST is a 404. It merges rather than replaces ([`ReadLog::note`] is
/// first-need-order, deduplicated), so a session can post the boot set, walk into the world, and
/// post again.
pub fn trace_router(log: Arc<ReadLog>) -> Router {
    Router::new()
        .route("/__boottrace", axum::routing::post(take_trace))
        .with_state(log)
}

async fn take_trace(State(log): State<Arc<ReadLog>>, body: String) -> Response {
    // Hand-parsed for the same reason `ReadLog::write` is hand-written: it is one array of strings,
    // and taking a JSON dependency into this crate for it would be the tail wagging the dog. The
    // shape is whatever `JSON.stringify(array)` produces, and anything that is not a string in it
    // is skipped rather than refused — a truncated capture is worth more than no capture.
    let mut count = 0usize;
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut name = String::new();
        while let Some(c) = chars.next() {
            match c {
                '"' => break,
                '\\' => match chars.next() {
                    Some('\\') => name.push('\\'),
                    Some('n') => name.push('\n'),
                    Some('t') => name.push('\t'),
                    Some('"') => name.push('"'),
                    Some(other) => name.push(other),
                    None => break,
                },
                other => name.push(other),
            }
        }
        if !name.is_empty() {
            log.note(&name);
            count += 1;
        }
    }
    // Written HERE, not only at shutdown. The shutdown write is what `--data`'s passive recording
    // needs (nothing else marks the end of a capture there), but a POST is an explicit "this is the
    // set" and there is no reason to make someone kill the host to see it — or to risk losing a
    // capture to a hard kill that never runs the graceful path. Cheap: a few hundred names.
    match log.write() {
        Ok(total) => {
            tracing::info!(posted = count, total, path = %log.path().display(), "boot trace written")
        }
        Err(e) => tracing::error!(error = %e, "boot trace received but could not be written"),
    }
    StatusCode::NO_CONTENT.into_response()
}

/// A chain-read failure that means "this path doesn't exist in the composite" — the two shapes
/// `Chain::read` produces for a missing file (never mounted at all, or tombstoned by a patch) —
/// as opposed to a real I/O fault reading a corrupt archive, which is a 500, not a 404.
fn is_missing(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("not in patch chain") || msg.contains("deleted from patch chain")
}

async fn file(method: Method, uri: Uri, State(state): State<DataState>) -> Response {
    let raw = uri.path().strip_prefix("/data/").unwrap_or("");
    let decoded = percent_encoding::percent_decode_str(raw).decode_utf8_lossy();
    let name = decoded.replace('/', "\\");

    // Logged BEFORE the read, so a name the chain does not hold still lands in the manifest at the
    // point the client wanted it. That is the right behaviour for a read-set capture: the client
    // asked, the request was made, and a manifest that quietly dropped every miss would understate
    // what a boot costs. `benilla-pack` already reports unresolved names rather than failing on
    // them, which is the other half of the same decision.
    if let Some(log) = &state.reads {
        log.note(&name);
    }

    let chain = Arc::clone(&state.chain);
    // `Chain::read` does synchronous file I/O (through benilla-mpq's blocking reads); running it
    // on the async runtime thread would stall every other in-flight request behind one disk seek.
    let read = tokio::task::spawn_blocking(move || chain.read(&name)).await;
    match read {
        Ok(Ok(bytes)) => {
            let body = if method == Method::HEAD {
                Vec::new()
            } else {
                bytes
            };
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, "application/octet-stream"),
                    // `private`: the bytes are the operator's game files — cacheable by the player's
                    // browser (they never change under one name), never by a shared cache.
                    (
                        header::CACHE_CONTROL,
                        "private, max-age=31536000, immutable",
                    ),
                ],
                body,
            )
                .into_response()
        }
        Ok(Err(e)) if is_missing(&e) => StatusCode::NOT_FOUND.into_response(),
        Ok(Err(e)) => {
            tracing::error!(error = %e, "chain read failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(join_err) => {
            tracing::error!(error = %join_err, "chain read task panicked");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// `GET /data/__index` — every name [`Chain::list`] can enumerate, for the wasm `Chain::list`
/// (Lane A) to mirror the native directory walk it can't do in a browser.
async fn index(State(state): State<DataState>) -> Response {
    let chain = Arc::clone(&state.chain);
    let listed = tokio::task::spawn_blocking(move || chain.list()).await;
    match listed {
        Ok(Ok(entries)) => {
            let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
            // Cacheable, unlike before: the wasm `Chain` now reads this once per session to
            // answer `contains` from memory (its struct doc has the numbers), and `web/boot.js`
            // prefetches it at character select so that read is a cache hit. A day, not the
            // files' year: the list changes when an operator adds a patch archive, and a stale
            // index would make the client believe those files absent until it expires.
            (
                [(header::CACHE_CONTROL, "private, max-age=86400")],
                Json(names),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            tracing::error!(error = %e, "chain list failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(join_err) => {
            tracing::error!(error = %join_err, "chain list task panicked");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}
