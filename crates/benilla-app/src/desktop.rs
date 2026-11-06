//! fork: desktop — the boot of a native client that reads its game data from an origin (the web
//! build's site) instead of an install, set up the way the page sets itself up: the containers
//! cached in the resident folder, the bundled addons fetched beside them, and the site's
//! `config.json` read for the WebSocket gateway and the realmlist ([`crate::webenv::config`]).
//!
//! Nothing here runs without an origin ([`benilla_formats::remote::origin`]): an install-reading
//! client boots exactly as before.

use std::sync::OnceLock;

/// The site's `config.json`, as fetched (or kept from the last start) at [`prepare`].
static CONFIG: OnceLock<Option<String>> = OnceLock::new();

/// Before any chain opens: name the cache, read the site config, bring the bundled addons up to
/// the origin's, and dial through the gateway the config names.
pub(crate) fn prepare() {
    let Some(origin) = benilla_formats::remote::origin() else {
        return;
    };
    let cache = crate::local_state::pack_cache();
    benilla_formats::remote::set_cache_root(cache.clone());
    println!(
        "data: game data from {origin}, cached in {}",
        cache.display()
    );
    let _ = CONFIG.set(benilla_formats::remote::site_config());
    match benilla_formats::remote::sync_addons() {
        Ok(0) => {}
        Ok(n) => println!("data: {n} bundled addon file(s) fetched"),
        Err(e) => eprintln!("data: the bundled addons could not be fetched: {e:#}"),
    }
    if let Some(ws) = gateway() {
        println!("net: dialing the realm through the gateway {ws}");
        benilla_protocol::transport::set_ws_base(ws);
    }
}

/// The gateway to dial through: `$WOW_WS` (set and empty: none), else none when `$WOW_HOST` pins a
/// server (a probe's local realm is a plain socket), else the site config's `ws`.
fn gateway() -> Option<String> {
    if let Ok(ws) = std::env::var("WOW_WS") {
        return (!ws.trim().is_empty()).then(|| ws.trim().to_string());
    }
    if std::env::var_os("WOW_HOST").is_some() {
        return None;
    }
    config("ws")
}

/// A string out of the site's `config.json`; an empty one is "not configured", as on the page.
pub(crate) fn config(key: &str) -> Option<String> {
    let json = CONFIG.get()?.as_deref()?;
    benilla_formats::remote::json_string(json, key).filter(|v| !v.trim().is_empty())
}
