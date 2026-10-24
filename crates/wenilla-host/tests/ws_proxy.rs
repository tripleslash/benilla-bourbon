//! Exercises the `/ws/{port}` proxy against a plain TCP echo server — no real mangosd needed, so
//! this runs anywhere (the box's actual 3724/8085 are used for the manual smoke test only, per
//! the plan).

use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use wenilla_host::ws::RemoteRealms;

/// A host that exists in no routing table anywhere: TEST-NET-1 (RFC 5737), reserved for
/// documentation and guaranteed not to be reachable. Used as the address a test must NOT dial — a
/// connection attempt to it cannot succeed by accident, so a test that passes proves the dial went
/// somewhere else.
const UNREACHABLE: &str = "192.0.2.1";

async fn spawn_echo_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind echo");
    let port = listener.local_addr().expect("local addr").port();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                loop {
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            if sock.write_all(&buf[..n]).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            });
        }
    });
    port
}

#[tokio::test]
async fn relays_binary_frames_both_ways_and_closes_with_the_upstream() {
    let echo_port = spawn_echo_server().await;
    // The allowlist is by port, so the test builds its own router with the echo port allowed —
    // production's fixed {3724, 8085} set lives in main.rs, not in ws::router itself.
    let app = wenilla_host::ws::router("127.0.0.1", [echo_port], RemoteRealms::Refuse);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("ws://{addr}/ws/{echo_port}?host=test");
    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");
    let (mut tx, mut rx) = ws.split();

    for i in 0u8..3 {
        let payload = vec![i; 16];
        tx.send(Message::Binary(payload.clone().into()))
            .await
            .expect("send frame");
        let echoed = rx.next().await.expect("frame").expect("ws frame ok");
        assert_eq!(echoed.into_data().as_ref(), payload.as_slice());
    }

    // A forbidden port is rejected before the upgrade even starts.
    let forbidden = format!("ws://{addr}/ws/{}", echo_port.wrapping_add(1));
    let err = tokio_tungstenite::connect_async(&forbidden)
        .await
        .expect_err("non-allowlisted port must be refused");
    assert!(
        err.to_string().contains("403") || err.to_string().to_lowercase().contains("forbidden"),
        "unexpected error: {err}"
    );

    // `SplitSink`/`SplitStream` share the one underlying connection through a lock (that's how
    // `.split()` works) — dropping just the sink half releases the write lock but sends nothing
    // and closes nothing, so the proxy's read loop would never see EOF and this would hang
    // forever (caught the hard way: the first version of this test did exactly that). Sending an
    // actual Close frame is what tears the connection down: the proxy's `ws_rx.next()` sees it,
    // ends the ws->tcp task, which drops the TCP write half, which the echo server reads as EOF —
    // closing its side too, which axum then answers with its own Close frame back to us.
    tx.close().await.expect("close ws");
    let end = rx.next().await;
    assert!(
        end.is_none() || matches!(end, Some(Ok(Message::Close(_)))),
        "expected the proxy to close, got {end:?}"
    );
}

/// Serve `app` on loopback and return its address.
async fn spawn_proxy(app: axum::Router) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind proxy");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// One round trip through the proxy, or the error that stopped it. `Ok` means the byte reached
/// something that echoed it — i.e. the echo server, i.e. the dial went to loopback.
async fn round_trip(addr: std::net::SocketAddr, port: u16, host: &str) -> Result<(), String> {
    let url = format!("ws://{addr}/ws/{port}?host={host}");
    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .map_err(|e| e.to_string())?;
    let (mut tx, mut rx) = ws.split();
    tx.send(Message::Binary(vec![7u8; 4].into()))
        .await
        .map_err(|e| e.to_string())?;
    match rx.next().await {
        Some(Ok(Message::Binary(b))) if b.as_ref() == [7u8; 4] => Ok(()),
        other => Err(format!("no echo back: {other:?}")),
    }
}

/// **The default, and the whole point of it.** With `RemoteRealms::Refuse` the client's `?host=`
/// changes nothing: a page naming an unreachable box still lands on the configured upstream. The
/// echo answering is the proof — TEST-NET-1 could not have.
#[tokio::test]
async fn a_refused_proxy_ignores_the_host_the_client_names() {
    let echo_port = spawn_echo_server().await;
    let addr = spawn_proxy(wenilla_host::ws::router(
        "127.0.0.1",
        [echo_port],
        RemoteRealms::Refuse,
    ))
    .await;

    round_trip(addr, echo_port, UNREACHABLE)
        .await
        .expect("the client-named host must be ignored, not dialed");
    // ...and a host-shaped attempt at something else is equally inert.
    round_trip(addr, echo_port, "evil.example.org")
        .await
        .expect("still the configured upstream");
}

/// With `RemoteRealms::Allow` the client's `?host=` *is* the address. Read the setup: the
/// configured upstream is the unreachable one, so an echo coming back can only mean the dial
/// followed the client's host instead.
#[tokio::test]
async fn an_allowed_proxy_dials_the_host_the_client_names() {
    let echo_port = spawn_echo_server().await;
    let addr = spawn_proxy(wenilla_host::ws::router(
        UNREACHABLE,
        [echo_port],
        RemoteRealms::Allow,
    ))
    .await;

    // The port comes off the PATH, not off the host — so a realmlist carrying one is accepted and
    // its port is dropped rather than smuggled past the allowlist.
    round_trip(addr, echo_port, "127.0.0.1:3724")
        .await
        .expect("the client-named host must be dialed");
}

/// **The port allowlist is not a function of the remote-realm policy.** `--allow-remote-realms`
/// widens *which box* may be dialed and nothing else — the port set stays the compiled-in
/// `{3724, 8085, 8086}` (realmd and the 1.12.1 and 1.18.1 world servers) under either setting, so a gateway sharing the host's network namespace cannot be
/// aimed at sshd, the database, or its own listener whatever a page asks for.
///
/// Written against `wenilla_host::ALLOWED_PORTS` itself — the exact value `main.rs` hands the
/// router — so widening that constant, or making it conditional, fails here rather than in front of
/// strangers. The 403 is the assertion that matters: it lands *before* the upgrade, so a port that
/// slipped through would show up as a successful handshake, not as a refused one.
#[tokio::test]
async fn the_port_allowlist_holds_whatever_the_remote_realm_policy_is() {
    assert_eq!(
        wenilla_host::ALLOWED_PORTS,
        [3724, 8085, 8086],
        "the hosted allowlist is realmd + the two world servers and nothing else"
    );

    for remote in [RemoteRealms::Refuse, RemoteRealms::Allow] {
        let addr = spawn_proxy(wenilla_host::ws::router(
            UNREACHABLE,
            wenilla_host::ALLOWED_PORTS,
            remote,
        ))
        .await;

        // What is actually within reach of a `network_mode: host` gateway on the gateway box: sshd,
        // the mysql the vmangos compose runs, the gateway's own listener, and the near-misses a
        // mistyped realmlist produces.
        for port in [
            22u16, 80, 443, 3306, 8080, 8090, 3723, 3725, 8084, 8087, 65535,
        ] {
            let url = format!("ws://{addr}/ws/{port}?host=127.0.0.1");
            let err = tokio_tungstenite::connect_async(&url)
                .await
                .map(|_| ())
                .expect_err(&format!("port {port} must never upgrade under {remote:?}"));
            let err = err.to_string();
            assert!(
                err.contains("403") || err.to_lowercase().contains("forbidden"),
                "port {port} under {remote:?} was refused for the wrong reason: {err}"
            );
        }
    }
}
