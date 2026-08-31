//! `wenilla-host` — the piece of the browser build that can't be static hosting: it serves the
//! wasm bundle, answers game-data reads out of the real MPQ chain, and proxies the two TCP ports
//! the client needs (login 3724, world 8085) over WebSocket, since a browser tab cannot open a
//! raw socket. See `web/README.md` for
//! the exact URL/encoding rules the client lanes code against.
//!
//! This host is a **local development tool**. `/data` hands out the operator's game files to
//! anyone who can reach the socket, with no login, so it binds to loopback by default and must
//! never be exposed on the open internet. Multi-user hosting is `wenilla-realm`, which mounts the
//! same routers behind a session cookie.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use benilla_formats::Chain;
use clap::Parser;

#[derive(Parser)]
#[command(
    about = "Serve the benilla browser build, its game data, and its net proxy (local testing only)",
    after_help = "wenilla-host is for local testing. /data serves the game files under --data to \
                  anyone who can reach the socket, without any login. Keep it on loopback or a \
                  private network; never expose it on the open internet. For hosting players, \
                  use wenilla-realm."
)]
struct Cli {
    /// Address to listen on. Loopback by default; anything else exposes `/data` to that
    /// network unauthenticated (see `--help`).
    #[arg(long, default_value = "127.0.0.1:8090")]
    bind: String,
    /// Directory holding the wasm-bindgen output (`index.html`, `wenilla.js`, `*.wasm`) —
    /// `scripts/web-build.sh`'s `web/dist/`.
    #[arg(long)]
    www: PathBuf,
    /// The vanilla `Data` directory (or a single `.MPQ`) the chain opens — the same one `benilla`
    /// itself reads on the desktop. The per-name `/data/<name>` route; the hosted client no
    /// longer uses it, but it is how a session runs without a container build.
    #[arg(long, conflicts_with = "packs")]
    data: Option<PathBuf>,
    /// A BENPACK container (`benilla-pack build --out …`), served at `/data/*` as plain files —
    /// the local stand-in for the object store the hosted client reads. Replaces `--data`.
    #[arg(long, conflicts_with = "data")]
    packs: Option<PathBuf>,
    /// Answer every request as if the host had no HTTP `Range` support — which is what a static host
    /// measurably does. Only meaningful with `--packs`, and the only way to actually exercise the
    /// client's whole-object fallback locally.
    #[arg(long)]
    no_ranges: bool,
    /// Host the `/ws/{port}` proxy dials for the allowed ports — the mangos boxes this host
    /// itself runs against, so it defaults to loopback.
    #[arg(long, default_value = "127.0.0.1")]
    upstream: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();

    let game = match (&cli.data, &cli.packs) {
        (Some(data), None) => {
            let chain = Arc::new(
                Chain::open(data).with_context(|| format!("opening {}", data.display()))?,
            );
            tracing::info!(data = %data.display(), "patch chain open");
            wenilla_host::data::router(chain)
        }
        (None, Some(packs)) => {
            if !packs.join("index.bpi").is_file() {
                anyhow::bail!("{} has no index.bpi — run `benilla-pack build` first", packs.display());
            }
            tracing::info!(
                packs = %packs.display(),
                ranges = !cli.no_ranges,
                "serving a BENPACK container"
            );
            wenilla_host::packs::router(packs, !cli.no_ranges)
        }
        _ => anyhow::bail!("pass exactly one of --data (live MPQ chain) or --packs (a built container)"),
    };

    let app = game
        .merge(wenilla_host::ws::router(
            cli.upstream.clone(),
            wenilla_host::ALLOWED_PORTS,
        ))
        .merge(wenilla_host::static_site::router(&cli.www));

    let listener = tokio::net::TcpListener::bind(&cli.bind)
        .await
        .with_context(|| format!("binding {}", cli.bind))?;
    if !binds_loopback(&listener) {
        tracing::warn!(
            bind = %cli.bind,
            "wenilla-host is a local testing tool: /data serves your game files to anyone who \
             can reach this address, with no login. Do not expose it on the open internet — use \
             wenilla-realm to host players."
        );
    }
    tracing::info!(bind = %cli.bind, www = %cli.www.display(), upstream = %cli.upstream, "wenilla-host listening");
    axum::serve(listener, app).await.context("serving")
}

/// Is the listener bound to a loopback address (the only place `/data` is safe unauthenticated)?
fn binds_loopback(listener: &tokio::net::TcpListener) -> bool {
    listener
        .local_addr()
        .map(|a| a.ip().is_loopback())
        .unwrap_or(false)
}
