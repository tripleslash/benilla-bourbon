//! `GET /ws/{port}` — the other half of the shared WebSocket scheme (Lane T ↔ Lane H): a browser
//! tab cannot open a raw TCP socket, so `benilla-protocol::transport::web::Conn` opens this
//! instead, and this proxy relays its binary frames to and from the real login (3724) / world
//! (8085) TCP ports. `port` is checked against an explicit allowlist, not just "any u16" — a
//! gateway binds a publicly reachable address, so an unchecked proxy would be an open relay onto
//! whatever else is listening on the box.
//!
//! ## Which box gets dialed, and who decides
//!
//! Two answers, and [`RemoteRealms`] is which one is in force.
//!
//! By default the *operator* decides and nobody else: the proxy dials the host configured for that
//! port, and the client's `?host=` is a log label. That is what keeps a relay out of this — the
//! realmlist a page carries cannot aim the socket, so the furthest a hostile page gets is talking
//! to the server the operator already pointed at.
//!
//! With [`RemoteRealms::Allow`] the client's `?host=` becomes the address actually dialed, which is
//! how one gateway fronts realms it does not itself run. It is a real widening, and that is why it
//! is a flag rather than the behaviour: an allowed gateway will connect wherever a page asks, on
//! the two allowlisted ports. `wenilla-host --allow-remote-realms` opts in — a gateway in front of
//! strangers leaves it off, and can then reach only the realm on its own box.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Whether a client may name the box this proxy dials — the policy half of
/// `wenilla-host --allow-remote-realms`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteRealms {
    /// The client's `?host=` is a log label and nothing else; the configured upstream is dialed
    /// whatever it says. **The default**, and the only setting under which this proxy is safe in
    /// front of people you do not know — it is what makes a public gateway reach one box and two
    /// ports.
    Refuse,
    /// The client's `?host=` is the address dialed, falling back to the configured upstream when it
    /// names none.
    Allow,
}

/// The longest host this will dial — DNS's own limit. Anything longer was never going to resolve,
/// so it is refused here instead of handed to the resolver.
const MAX_HOST: usize = 253;

#[derive(Clone)]
struct WsState {
    /// Port → host the proxy dials for it — plain hostname/IP, not a URL. The key set *is* the
    /// allowlist: a port with no entry is refused, which is the property worth the map. Today
    /// every port carries the same host (`main.rs` defaults to loopback); the per-port shape is
    /// what a realmd and a mangosd on different boxes would need.
    upstreams: Arc<HashMap<u16, Arc<str>>>,
    /// May a client's `?host=` replace the value above for its own session?
    remote: RemoteRealms,
}

/// Build the `/ws/{port}` router with one upstream host for every allowed port. `allowed` is the
/// exact port set the proxy will dial — production passes `{3724, 8085}` (the plan's shared
/// scheme: "only 3724 and 8085 are allowed"); tests pass their own echo-server port so the
/// allowlist check doesn't have to be bypassed to exercise it.
pub fn router(
    upstream: impl Into<Arc<str>>,
    allowed: impl IntoIterator<Item = u16>,
    remote: RemoteRealms,
) -> Router {
    let upstream: Arc<str> = upstream.into();
    router_map(
        allowed.into_iter().map(|p| (p, Arc::clone(&upstream))),
        remote,
    )
}

/// Build the `/ws/{port}` router from an explicit port → host map; the keys are the allowlist.
/// Private: [`router`] is the only caller, and the only shipped hosting gives every port the
/// same host.
fn router_map(
    upstreams: impl IntoIterator<Item = (u16, Arc<str>)>,
    remote: RemoteRealms,
) -> Router {
    Router::new()
        .route("/ws/{port}", get(upgrade))
        .with_state(WsState {
            upstreams: Arc::new(upstreams.into_iter().collect()),
            remote,
        })
}

/// `host` arrives as `?host=` — the realmlist the page is dialing. Under [`RemoteRealms::Refuse`]
/// it is logging only (the shared scheme's own words) and the configured upstream is dialed, so a
/// client cannot redirect the socket somewhere else on the network. Under [`RemoteRealms::Allow`]
/// it *is* the address dialed: the whole of what that setting buys, and the whole of what it costs.
async fn upgrade(
    State(state): State<WsState>,
    Path(port): Path<u16>,
    Query(params): Query<HashMap<String, String>>,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(upstream) = state.upstreams.get(&port).cloned() else {
        return StatusCode::FORBIDDEN.into_response();
    };
    let asked = params.get("host").cloned().unwrap_or_default();
    let dial: Arc<str> = match state.remote {
        RemoteRealms::Refuse => upstream,
        RemoteRealms::Allow => dialable_host(&asked).map_or(upstream, Arc::from),
    };
    // Debug, not info: under the default policy this fires on every connection (the page always
    // sends its realmlist), so an info line would be one per login for a decision that never
    // changes. It exists at all because an address that is silently not the one dialed is exactly
    // the kind of thing that costs an hour — `RUST_LOG=wenilla_host=debug` answers "why is it not
    // talking to X".
    tracing::debug!(asked = %asked, dial = %dial, port, policy = ?state.remote, "ws proxy dialing");
    ws.on_upgrade(move |socket| async move {
        if let Err(e) = relay(socket, &dial, port).await {
            tracing::warn!(error = %e, asked = %asked, dial = %dial, port, "ws proxy session ended");
        }
    })
}

/// The host half of a client-supplied `host[:port]`, or `None` if there is nothing dialable in it.
///
/// **The port is not taken from here** — it is the path's, and the path's is checked against the
/// allowlist. A client that could smuggle its own port past that check would undo the one part of
/// this proxy that is a security boundary rather than a default.
///
/// The split mirrors `benilla_protocol::host_port`, the law the client builds this URL through: a
/// single colon means a port, two or more is an IPv6 literal and is kept whole. Mirrored rather
/// than imported — a byte-pump proxy that linked the game protocol crate to split a string would be
/// the tail wagging the dog — and pinned to that behaviour by the tests below.
fn dialable_host(asked: &str) -> Option<&str> {
    let asked = asked.trim();
    let host = match asked.rsplit_once(':') {
        // A single colon whose suffix parses as a port: drop it. A suffix that does not parse is
        // `host_port`'s own tolerance — the whole string stays the host — so nothing that connects
        // through the client fails here.
        Some((h, p)) if !h.contains(':') && p.parse::<u16>().is_ok() => h,
        _ => asked,
    };
    if host.is_empty() || host.len() > MAX_HOST {
        return None;
    }
    // Whitespace and control characters are not in any host, and are how a log line gets forged.
    if host.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return None;
    }
    Some(host)
}

/// Relay one session: dial the upstream TCP port, then pump bytes both directions until both
/// sides have closed. "One TCP read chunk = one binary frame, any frame = one `write_all`"
/// (shared scheme) — no re-framing or buffering beyond the OS's own read/write granularity.
///
/// Each direction propagates its own end-of-stream to the *other* transport, rather than the two
/// futures racing in a `select!`: a `select!` here would drop whichever direction lost the race
/// mid-flight, which for the ws->tcp direction means the client's Close frame never gets echoed
/// back (tungstenite then reports a `ResetWithoutClosingHandshake` protocol error instead of a
/// clean close — caught by `tests/ws_proxy.rs`'s close-handshake test hanging, then failing,
/// before this shape). `join!` runs both to their own natural end instead:
/// receiving a WS Close (or a read error) shuts down our TCP write half, which the upstream reads
/// as EOF and — for a well-behaved peer — closes its own side, which our TCP read then sees as
/// EOF and answers with our own Close frame back to the client.
async fn relay(ws: WebSocket, upstream: &str, port: u16) -> anyhow::Result<()> {
    let tcp = TcpStream::connect((upstream, port)).await?;
    tcp.set_nodelay(true)?;
    let (mut tcp_r, mut tcp_w) = tcp.into_split();
    let (mut ws_tx, mut ws_rx) = ws.split();

    let tcp_to_ws = async {
        let mut buf = [0u8; 65536];
        loop {
            match tcp_r.read(&mut buf).await {
                Ok(0) | Err(_) => {
                    // If the client already sent its own Close, tungstenite auto-queues a reply
                    // the moment `ws_rx` reads it (the behaviour `Message::Close`'s docs mention:
                    // "axum will automatically respond with a close frame if necessary") — but
                    // only *queues* it; nothing has driven a write-side poll since, so it's still
                    // sitting unflushed. `send` here fails with a "send after closing" protocol
                    // error in that case (there's already a Close in flight); flushing instead is
                    // what actually puts those bytes on the wire. Without this the socket would
                    // just drop with the reply never sent — tungstenite calls that a
                    // `ResetWithoutClosingHandshake`, and it's exactly the shape
                    // `tests/ws_proxy.rs`'s close test hit before this fallback existed.
                    if ws_tx.send(Message::Close(None)).await.is_err() {
                        let _ = ws_tx.flush().await;
                    }
                    break;
                }
                Ok(n) => {
                    if ws_tx
                        .send(Message::binary(buf[..n].to_vec()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    };

    let ws_to_tcp = async {
        loop {
            match ws_rx.next().await {
                Some(Ok(Message::Binary(data))) => {
                    if tcp_w.write_all(&data).await.is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Text(text))) => {
                    if tcp_w.write_all(text.as_bytes()).await.is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
            }
        }
        // Half-close our write side so the upstream sees EOF, not just a dangling socket.
        let _ = tcp_w.shutdown().await;
    };

    tokio::join!(tcp_to_ws, ws_to_tcp);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::dialable_host;

    /// What a client may hand this proxy as its realm, and what comes back out of it. The port half
    /// is dropped on purpose: the socket's port is the path's, which is allowlisted.
    #[test]
    fn a_client_host_keeps_its_name_and_loses_its_port() {
        assert_eq!(
            dialable_host("logon.example.org"),
            Some("logon.example.org")
        );
        assert_eq!(
            dialable_host("  logon.example.org  "),
            Some("logon.example.org")
        );
        assert_eq!(
            dialable_host("logon.example.org:3724"),
            Some("logon.example.org")
        );
        assert_eq!(dialable_host("127.0.0.1:8085"), Some("127.0.0.1"));
        // `host_port`'s tolerance: a suffix that is not a port was never a port.
        assert_eq!(
            dialable_host("logon.example.org:nope"),
            Some("logon.example.org:nope")
        );
        // Two or more colons is an IPv6 literal, which keeps its own colons.
        assert_eq!(dialable_host("::1"), Some("::1"));
        assert_eq!(dialable_host("fe80::1"), Some("fe80::1"));
    }

    /// Nothing dialable = fall back to the configured upstream, which is what `None` means to the
    /// caller. The control-character case is the one worth having: it is a forged log line.
    #[test]
    fn nothing_dialable_is_refused() {
        assert_eq!(dialable_host(""), None);
        assert_eq!(dialable_host("   "), None);
        assert_eq!(dialable_host(":3724"), None);
        assert_eq!(dialable_host("logon.example.org\nGET /"), None);
        assert_eq!(dialable_host("a b"), None);
        assert_eq!(dialable_host(&"a".repeat(super::MAX_HOST + 1)), None);
    }
}
