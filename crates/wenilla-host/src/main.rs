//! `wenilla-host` — the piece of the browser build that can't be static hosting: it proxies the
//! two TCP ports the client needs (login 3724, world 8085) over WebSocket, since a browser tab
//! cannot open a raw socket, and — for local work — also serves the wasm bundle and the game data
//! beside it. See `web/README.md` for the exact URL rules the client lanes code against.
//!
//! Three shapes, decided by which flags are present:
//!
//! - **Gateway** (`--bind` and nothing else): only `/ws/{port}`. This is the hosted shape — the
//!   static tree lives on a CDN, the client is pointed at this address by `config.json`'s `ws`,
//!   and this process sits on the realm box relaying to realmd/mangosd on **loopback**. It is the
//!   only shape meant to face the internet, and the only one that hands out nothing: without
//!   `--allow-remote-realms` it dials `127.0.0.1` and nowhere else, on two ports.
//! - **Full local host** (`--www` with `--packs`/`--data`): the page, the game data and the proxy
//!   from one process, which is how a session runs without a site.
//! - **Anything between** — `--www` alone serves a build against a remote realm, for instance.
//!
//! `--www` and `/data` hand out the operator's files to anyone who can reach the socket, with no
//! login, which is why the bind defaults to loopback and why serving them off it warns.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use benilla_formats::Chain;
use clap::Parser;

#[derive(Parser)]
#[command(
    about = "The browser build's WebSocket gateway, and (locally) the page and game data beside it",
    after_help = "With no --www/--packs/--data this is a gateway and nothing else: /ws/3724 and \
                  /ws/8085 relayed to 127.0.0.1, which is the shape a public site runs — the \
                  page and the container are static files elsewhere, pointed here by config.json's \
                  `ws`. Adding --www and --packs makes it the whole local host instead; those two \
                  routes have no login, so keep such a run on loopback or a private network."
)]
struct Cli {
    /// Address to listen on. Loopback by default, which is right for a run that serves files
    /// (`--www`/`--packs` have no login) and for a gateway behind a TLS proxy on the same box;
    /// a gateway that terminates its own connections binds the public address instead.
    #[arg(long, default_value = "127.0.0.1:8090")]
    bind: String,
    /// Directory holding the Trunk output (`index.html`, `wenilla.js`, `*.wasm`) — `web/dist/`.
    /// Omit it to run as a gateway only, which is what a site does: the page is static files on
    /// A CDN, and the only thing this process owns is `/ws/{port}`.
    #[arg(long)]
    www: Option<PathBuf>,
    /// The vanilla `Data` directory (or a single `.MPQ`) the chain opens — the same one `benilla`
    /// itself reads on the desktop. The per-name `/data/<name>` route; the hosted client no
    /// longer uses it, but it is how a session runs without a container build.
    #[arg(long, conflicts_with = "packs")]
    data: Option<PathBuf>,
    /// A BENPACK container (`benilla-pack build --out …`), served at `/data/*` as plain files —
    /// the local stand-in for the object store the hosted client reads. Replaces `--data`.
    #[arg(long, conflicts_with = "data")]
    packs: Option<PathBuf>,
    /// The optional updated-character-model layer's container (`benilla-pack build --data
    /// .../patch-A.mpq --out web/packs-hd`), served at `/data-hd/*` beside `--packs`.
    ///
    /// Only reached when the player has the `hdCharacterModels` CVar on: with it off the client
    /// never requests this tree, which is exactly what serving it here lets you verify.
    #[arg(long, requires = "packs")]
    packs_hd: Option<PathBuf>,
    /// The Turtle WoW 1.18.1 layer's container (`benilla-pack overlay` then `build --only …
    /// --out web/packs-turtle`), served at `/data-turtle/*` beside `--packs`. Fetched only while the
    /// client is on a realm that advertises 1.18.1.
    #[arg(long, requires = "packs")]
    packs_turtle: Option<PathBuf>,
    /// The reforged art layer's container (`benilla-pack overlay --patches B,…,Y` then `build
    /// --only … --out web/packs-reforged`), served at `/data-reforged/*` beside `--packs`. Fetched
    /// only with the Turtle layer, while the player's setting has it on.
    #[arg(long, requires = "packs")]
    packs_reforged: Option<PathBuf>,
    /// Forward `/data/*` and `/data-hd/*` to a hosted static tree (e.g. `https://wow.example.com`)
    /// instead of serving a local container — for working on the client with no game data here.
    #[arg(long, conflicts_with_all = ["data", "packs"])]
    packs_remote: Option<String>,
    /// Record every chain name the client reads into `<file>`, in `web/boot-manifest.json`'s own
    /// shape — the server-side capture of a read set, and the input to `benilla-pack sidecars`.
    ///
    /// Fed from two places, because the two serving modes see different things:
    ///
    /// - **`--packs`** (what everyone runs): the host sees pack objects, and a pack id names a few
    ///   hundred files at once — so the *client* reports instead. Open the page with
    ///   `?boottrace=1` and `web/boot.js` posts its collected read set to `POST /__boottrace`,
    ///   which this flag arms.
    /// - **`--data`**: the per-name route carries the name in the URL, so every read is recorded
    ///   as it arrives, with no page cooperation at all.
    ///
    /// Either way the file is written when the host shuts down cleanly (Ctrl-C), so a capture is
    /// "start the host, boot the client, get where you want the manifest to cover, stop the host".
    /// Translate it into a container's prefetch with `benilla-pack sidecars`.
    #[arg(long)]
    log_reads: Option<PathBuf>,
    /// Answer every request as if the host had no HTTP `Range` support — which is what a static host
    /// measurably does. Only meaningful with `--packs`, and the only way to actually exercise the
    /// client's whole-object fallback locally.
    #[arg(long)]
    no_ranges: bool,
    /// Host the `/ws/{port}` proxy dials for the allowed ports — the mangos boxes this host
    /// itself runs against, so it defaults to loopback. **Ignored without
    /// `--allow-remote-realms`**, which is what keeps the proxy on this machine by default.
    #[arg(long, default_value = LOCAL_REALM)]
    upstream: String,
    /// Let the proxy reach a realm that is not on this machine: `--upstream` starts being honoured,
    /// and the client's own realmlist (`?host=`) becomes the address dialed for its session.
    ///
    /// Off by default, and the default is the safe half of the pair: without it every session goes
    /// to loopback whatever anyone asks for, so a page cannot aim this proxy and neither can a
    /// mistyped flag. With it, this host will connect wherever a page asks — on 3724, 8085 and 8086
    /// only, but wherever. Turn it on for a gateway that fronts a realm somewhere else; leave it
    /// off for the single-box case, which is what this tool is for.
    #[arg(long)]
    allow_remote_realms: bool,
}

/// Where the proxy dials when it may not leave this machine. Not `localhost`: that resolves
/// through the host's own name lookup and can answer `::1` on a box where mangosd is listening on
/// IPv4 only — an address literal cannot be misresolved.
const LOCAL_REALM: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();

    let serves_data = cli.data.is_some() || cli.packs.is_some();
    // Held past the router so the shutdown path can write it — see `--log-reads`.
    let mut read_log: Option<Arc<wenilla_host::data::ReadLog>> = None;
    let game = match (&cli.data, &cli.packs) {
        (Some(data), None) => {
            // The layer is a mount decision and this host makes it the way every other tool does:
            // `$WOW_HD_MODELS`. A read-set capture against the updated character models is
            // `WOW_HD_MODELS=1 wenilla-host --data ... --log-reads ...`, and the manifest that comes
            // out names the HD art because the chain that answered it was the HD chain.
            let hd = benilla_formats::layers::hd_models_env().unwrap_or(false);
            let chain = Arc::new(
                Chain::open_layered(data, hd)
                    .with_context(|| format!("opening {}", data.display()))?,
            );
            tracing::info!(data = %data.display(), hd_character_models = hd, "patch chain open");
            let log = cli.log_reads.as_ref().map(|p| {
                tracing::info!(file = %p.display(), "recording the read set");
                Arc::new(wenilla_host::data::ReadLog::new(p.clone()))
            });
            read_log = log.clone();
            wenilla_host::data::router_logged(chain, log)
        }
        (None, Some(packs)) => {
            if !packs.join("index.bpi").is_file() {
                anyhow::bail!(
                    "{} has no index.bpi — run `benilla-pack build` first",
                    packs.display()
                );
            }
            tracing::info!(
                packs = %packs.display(),
                ranges = !cli.no_ranges,
                "serving a BENPACK container"
            );
            let mut router = wenilla_host::packs::router(packs, !cli.no_ranges);
            // The trace sink, in the mode that needs it: against a container the host cannot see
            // names, so the client posts them. See `--log-reads` and `data::trace_router`.
            if let Some(p) = &cli.log_reads {
                tracing::info!(
                    file = %p.display(),
                    "recording the read set — open the page with ?boottrace=1, then Ctrl-C here"
                );
                let log = Arc::new(wenilla_host::data::ReadLog::new(p.clone()));
                read_log = Some(Arc::clone(&log));
                router = router.merge(wenilla_host::data::trace_router(log));
            }
            if let Some(hd) = &cli.packs_hd {
                if !hd.join("index.bpi").is_file() {
                    anyhow::bail!(
                        "{} has no index.bpi — run `benilla-pack build --data <Data>/patch-A.mpq                          --out {}` first",
                        hd.display(),
                        hd.display()
                    );
                }
                tracing::info!(packs_hd = %hd.display(), "serving the updated-character-model layer");
                router = router.merge(wenilla_host::packs::hd_router(hd, !cli.no_ranges));
            }
            if let Some(turtle) = &cli.packs_turtle {
                if !turtle.join("index.bpi").is_file() {
                    anyhow::bail!(
                        "{} has no index.bpi — run `benilla-pack overlay` and `build --only` first",
                        turtle.display()
                    );
                }
                tracing::info!(packs_turtle = %turtle.display(), "serving the Turtle WoW 1.18.1 layer");
                router = router.merge(wenilla_host::packs::turtle_router(turtle, !cli.no_ranges));
            }
            if let Some(reforged) = &cli.packs_reforged {
                if !reforged.join("index.bpi").is_file() {
                    anyhow::bail!(
                        "{} has no index.bpi — run `benilla-pack overlay` and `build --only` first",
                        reforged.display()
                    );
                }
                tracing::info!(packs_reforged = %reforged.display(), "serving the reforged art layer");
                router = router.merge(wenilla_host::packs::reforged_router(
                    reforged,
                    !cli.no_ranges,
                ));
            }
            router
        }
        // Neither: a gateway. Not an error — it is the *hosted* shape, where the container is
        // static objects on a CDN and this process exists only for the two sockets.
        (None, None) => match &cli.packs_remote {
            Some(base) => {
                tracing::info!(remote = %base, "forwarding /data, /data-hd, /data-turtle and /data-reforged to a remote site");
                wenilla_host::remote::router(base)
            }
            None => {
                tracing::info!("no --data/--packs: serving no game data");
                axum::Router::new()
            }
        },
        // Unreachable: clap's `conflicts_with` refuses both before this runs.
        (Some(_), Some(_)) => anyhow::bail!("--data and --packs are mutually exclusive"),
    };

    // **The localhost pin.** Without `--allow-remote-realms` the configured upstream is not merely
    // defaulted to loopback, it is *replaced* by it — an operator who passed `--upstream` and no
    // flag gets loopback and a line saying so, rather than a proxy quietly reaching a box they had
    // only half asked for. The client's own `?host=` is refused by the same switch, one layer down
    // (`ws::RemoteRealms`), so both ways of naming a remote realm are off together or on together.
    let (upstream, remote) = if cli.allow_remote_realms {
        (cli.upstream.clone(), wenilla_host::ws::RemoteRealms::Allow)
    } else {
        if !is_local_realm(&cli.upstream) {
            tracing::warn!(
                upstream = %cli.upstream,
                "--upstream names another machine but --allow-remote-realms is not set: dialing \
                 127.0.0.1 instead. Pass --allow-remote-realms to reach it."
            );
        }
        (
            LOCAL_REALM.to_string(),
            wenilla_host::ws::RemoteRealms::Refuse,
        )
    };

    let app = game.merge(wenilla_host::ws::router(
        upstream.clone(),
        wenilla_host::ALLOWED_PORTS,
        remote,
    ));
    let app = match &cli.www {
        Some(www) => app.merge(wenilla_host::static_site::router(www)),
        None => app,
    };

    let listener = tokio::net::TcpListener::bind(&cli.bind)
        .await
        .with_context(|| format!("binding {}", cli.bind))?;
    // Only a run that actually serves files is worth the warning. A gateway on `0.0.0.0` is the
    // *intended* public hosting — no files at all, loopback on two ports — and warning about it
    // would teach the operator to ignore the line that still matters.
    if serves_data && !binds_loopback(&listener) {
        tracing::warn!(
            bind = %cli.bind,
            "/data serves your game files to anyone who can reach this address, with no login. \
             Keep a run that serves data on loopback or a private network."
        );
    }
    if cli.allow_remote_realms {
        tracing::warn!(
            "--allow-remote-realms is set: this proxy will connect to whatever host a page names, \
             on ports 3724, 8085 and 8086. Only do this on a gateway you mean to be one."
        );
    }
    tracing::info!(
        bind = %cli.bind,
        www = cli.www.as_ref().map(|w| w.display().to_string()),
        upstream = %upstream,
        remote_realms = cli.allow_remote_realms,
        "wenilla-host listening"
    );
    // Ctrl-C is a NORMAL end to a capture run, not a fault: `--log-reads` records the read set in
    // memory and this is where it lands on disk. Without a graceful shutdown the process would die
    // mid-boot-capture with nothing written, which is the one way to lose a capture that took a
    // login and a walk around Elwynn to produce.
    //
    // Armed only when there is something to write. A gateway run keeps the old behaviour exactly:
    // `axum::serve` with no shutdown signal, and Ctrl-C ends the process the way it always did.
    let result = match &read_log {
        None => axum::serve(listener, app).await.context("serving"),
        Some(_) => axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                tracing::info!("shutting down — writing the read set");
            })
            .await
            .context("serving"),
    };
    if let Some(log) = read_log {
        match log.write() {
            Ok(n) => tracing::info!(
                names = n,
                "read set written — translate it with `benilla-pack sidecars --packs <container>                  --boot-manifest <this file>`"
            ),
            Err(e) => tracing::error!(error = %e, "could not write the read set"),
        }
    }
    result
}

/// Is this upstream on the machine the host is running on? Only used to decide whether the
/// localhost pin is worth *saying* — the pin applies either way, so a name this cannot recognise
/// (a `/etc/hosts` alias for loopback) costs one warn line and nothing else.
fn is_local_realm(upstream: &str) -> bool {
    let host = upstream
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host == "::1"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Is the listener bound to a loopback address (the only place `--www` and `/data` are safe
/// unauthenticated)? A gateway-only run serves neither and never consults this.
fn binds_loopback(listener: &tokio::net::TcpListener) -> bool {
    listener
        .local_addr()
        .map(|a| a.ip().is_loopback())
        .unwrap_or(false)
}
