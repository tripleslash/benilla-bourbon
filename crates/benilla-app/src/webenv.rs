//! Config that would be an environment variable on native and can't be one in a browser tab —
//! wasm32 has no process environment (`std::env::var` always answers `NotPresent` there), so the
//! web build reads the page's own query string instead. Every `WOW_*` read this crate ships to
//! wasm goes through [`var`] (`net/io.rs`'s `NetConfig::from_env`, `char_select`'s
//! create-if-empty pick, `realm_select`'s realm fast path, `login`'s env fast path) so a native
//! `WOW_HOST=… cargo run` and a web `?host=…` URL are the exact same fast path from the caller's
//! point of view — see the plan's shared interface, `benilla_app::webenv::var`.
//!
//! Key mapping is mechanical: `WOW_<NAME>` reads back as the query key `<name>` lowercased with
//! the prefix dropped — `WOW_HOST` → `?host=`, `WOW_USER` → `?user=`, `WOW_PASS` → `?pass=`,
//! `WOW_CHAR` → `?char=`, `WOW_REALM` → `?realm=`. `WOW_WIN` (the windowed-vs-fullscreen dev
//! switch) is native-only and never routed through here — there is no window chrome to resize in
//! a browser tab.

/// Native: exactly `std::env::var(name).ok()` — every existing caller already treats a missing
/// var this way, so routing through here changes nothing on the platform that already worked.
#[cfg(not(target_arch = "wasm32"))]
pub fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// wasm32: a page-provided env object first, then the page's `?key=value&…` query string —
/// both keyed by `name` lowercased with its `WOW_` prefix dropped.
///
/// The env object is `window.__wenilla_env` (`{user, pass, host, …}`), which a hosting page sets
/// before `init()` after fetching the values over its own authenticated session; it is
/// snapshotted on the first read so the page can `delete` the global right after boot. The
/// query string is the dev fast path (`?user=&pass=`), but for the three credential keys —
/// `user`, `pass`, `char` — it is consulted **only if** the env object carries
/// `dev_query_creds: "1"`: a production host never wants a password in a shareable URL, and a
/// page that sets no env object at all (upstream's plain `index.html`) opts in explicitly. Every
/// other key (`host`, and each CVar via `cvars.rs`) keeps the unconditional query fallback.
///
/// Values are percent-decoded through `js_sys::decode_uri_component` (the browser's own decoder,
/// so whatever encoded the URL round-trips) — a key or value that fails to decode (not valid
/// percent-encoding, or not valid UTF-8 once decoded) is treated as absent rather than panicking
/// on a malformed link.
#[cfg(target_arch = "wasm32")]
pub fn var(name: &str) -> Option<String> {
    let key = name.strip_prefix("WOW_").unwrap_or(name).to_lowercase();
    // fork: the render worker has no `window`; it reads what the page published for it.
    if web_sys::window().is_none() {
        let (env, search) = FOR_WORKERS.get()?;
        return lookup(env, search, &key);
    }
    let env = page_env();
    let search = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default();
    lookup(&env, &search, &key)
}

/// A value from **`/config.json`** — the hosted tree's own configuration, fetched by
/// `web/boot.js` before `init()` and left on `window.__wenilla_config`. It ships at the origin
/// root with the build (`web/config.json` → Trunk → `dist/`), not under `/data` with the 3.3 GiB
/// container: it is the one file in the hosted tree an operator edits. Its two keys — `ws`, the
/// WebSocket gateway the browser dials, and `realmlist`, the logon address the handshake carries —
/// are documented in `web/README.md`.
///
/// A separate seam from [`var`], not a fourth fallback inside it, and the separation is the design:
///
/// - `__wenilla_env` outranks the query string in [`lookup`], so folding a hosted default into it
///   would let `config.json` beat a hand-written `?ws=…`. Backwards. Callers consult `var` first and
///   this second, which puts the precedence where it can be read.
/// - It is read *after* `var`, so the harness levers still win: `$WOW_HOST`/`?host=` mark the
///   session `pinned_by_env` and cannot be moved off, which is right for a probe that must dial
///   exactly one server. Below that, `realmlist` here is the page's only source (`cvars`'
///   `apply_hosted_realmlist`) — there is no login-screen control under it any more.
///
/// It exists at all because the WebSocket gateway's address is supplied *after* the client is built:
/// a build-time constant would bake an address into a 90 MB wasm module, while this is a ~100-byte
/// `no-cache` asset that a re-upload of one file repoints.
///
/// Native: always `None`. There is no page, and the native client has real environment variables.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn config(_key: &str) -> Option<String> {
    None
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn config(key: &str) -> Option<String> {
    thread_local! {
        static CONFIG: std::cell::OnceCell<std::collections::HashMap<String, String>> =
            const { std::cell::OnceCell::new() };
    }
    CONFIG
        .with(|cell| {
            cell.get_or_init(|| js_string_map("__wenilla_config"))
                .get(key)
                .cloned()
        })
        // An empty string is how the checked-in `web/config.json` ships; it means "not
        // configured", not "configure this to the empty address".
        .filter(|v| !v.trim().is_empty())
}

/// fork: the page's env object and query string, as the page read them, for code running on a
/// worker ([`var`] there). Published by [`publish_for_workers`] before the render worker runs
/// anything; a later `history.replaceState` does not reach it.
#[cfg(target_arch = "wasm32")]
static FOR_WORKERS: std::sync::OnceLock<(std::collections::HashMap<String, String>, String)> =
    std::sync::OnceLock::new();

/// fork: snapshot the page's env object and query string for [`var`] on a worker. Page only.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics", feature = "webgpu"))]
pub(crate) fn publish_for_workers() {
    let search = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default();
    let _ = FOR_WORKERS.set((page_env(), search));
}

/// The keys the query string may only answer when the page opted in with `dev_query_creds`.
const CREDENTIAL_KEYS: [&str; 3] = ["user", "pass", "char"];
const DEV_QUERY_CREDS: &str = "dev_query_creds";

/// The pure lookup behind [`var`] — env object first, query string second, credentials gated —
/// kept free of `web_sys` so it is unit-tested natively.
#[allow(dead_code)]
fn lookup(
    env: &std::collections::HashMap<String, String>,
    search: &str,
    key: &str,
) -> Option<String> {
    if let Some(v) = env.get(key) {
        return Some(v.clone());
    }
    if CREDENTIAL_KEYS.contains(&key) && env.get(DEV_QUERY_CREDS).map(String::as_str) != Some("1") {
        return None;
    }
    let query = search.strip_prefix('?').unwrap_or(search);
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (decode(k)? == key).then(|| decode(v)).flatten()
    })
}

/// `window.__wenilla_env` as a string map, read once. Non-string values are ignored; a missing
/// or non-object global is an empty map (so upstream's page — which sets nothing — behaves as
/// "no env, credentials not allowed from the query").
#[cfg(target_arch = "wasm32")]
fn page_env() -> std::collections::HashMap<String, String> {
    thread_local! {
        static ENV: std::cell::OnceCell<std::collections::HashMap<String, String>> = const { std::cell::OnceCell::new() };
    }
    ENV.with(|cell| cell.get_or_init(|| js_string_map("__wenilla_env")).clone())
}

/// One page global read as a flat string map. Shared by [`page_env`] and [`config`] because the two
/// seams differ in *precedence and meaning*, not in how a JS object is read.
#[cfg(target_arch = "wasm32")]
fn js_string_map(global: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let Some(window) = web_sys::window() else {
        return map;
    };
    let Ok(obj) = js_sys::Reflect::get(&window, &global.into()) else {
        return map;
    };
    if !obj.is_object() {
        return map;
    }
    for entry in js_sys::Object::entries(&js_sys::Object::from(obj)).iter() {
        let pair = js_sys::Array::from(&entry);
        if let (Some(k), Some(v)) = (pair.get(0).as_string(), pair.get(1).as_string()) {
            map.insert(k, v);
        }
    }
    map
}

/// `decodeURIComponent`, tolerant of failure — a query string a person hand-edited can contain a
/// bare `%` or an invalid UTF-8 sequence, and this is a login convenience, not a protocol parser.
#[cfg(target_arch = "wasm32")]
fn decode(s: &str) -> Option<String> {
    js_sys::decode_uri_component(s).ok().map(String::from)
}

/// Native stand-in for the browser decoder so [`lookup`] can be tested here: percent-decoding
/// only, which is what the tests exercise.
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `WOW_HOST`'s fallback when unset — `net/io.rs`'s `NetConfig::from_env` calls this instead of
/// hard-coding a default, so it stays a one-line swap there (`var("WOW_HOST").unwrap_or_else(…)`)
/// no matter what the right default is per platform.
///
/// Native: `localhost`, decision 0539's original default, unchanged.
///
/// Web: **the gateway's hostname when one is configured, and the page's own only otherwise.** This
/// used to be unconditionally the page's hostname, on the premise that "the web host proxies to the
/// realm beside it" — true of `wenilla-host`, which serves the bundle and the `/ws/{port}` proxy
/// from one process, and false the moment the static tree moves to a static host and the gateway is a
/// box somewhere else. On such a site the page's hostname is a CDN edge with no realmd behind it,
/// so defaulting to it would put every first-time visitor's login on an address that cannot answer.
///
/// The `ws` base is the honest source: it is the endpoint the client will actually dial, so the
/// realm it reaches is on that host by construction. This is only the **fallback**, for a page
/// whose `config.json` names no `realmlist` at all — which is how the checked-in one ships, so a
/// local `wenilla-host --www web/dist` still dials the origin that served it. `?host=` and
/// `config.json`'s own `realmlist` both win over it (`Realmlist`, `cvars::load_config`).
#[cfg(target_arch = "wasm32")]
pub(crate) fn default_wow_host() -> String {
    if let Some(host) = ws_base().as_deref().and_then(host_of_ws_base) {
        return host.to_string();
    }
    web_sys::window()
        .and_then(|w| w.location().hostname().ok())
        .unwrap_or_else(|| "localhost".into())
}

/// The WebSocket gateway base (`wss://host[:port]`), in precedence order: `?ws=` /
/// `window.__wenilla_env.ws`, then `/config.json`'s `ws`. `None` = the page's own origin, which
/// is `benilla_protocol::transport::web`'s own default and today's `wenilla-host` behaviour.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn ws_base() -> Option<String> {
    var("WOW_WS").or_else(|| config("ws"))
}

/// `wss://gw.example.org:8443` -> `gw.example.org`. Pure string math, so it is unit-tested here
/// rather than only exercised in a browser; a base that does not parse yields `None` and the caller
/// falls back, because a malformed config value must not be worse than an absent one.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn host_of_ws_base(base: &str) -> Option<&str> {
    let rest = base
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base)
        .trim_end_matches('/');
    let host = rest.split(['/', '?']).next()?;
    // Strip the port, but leave a bracketed IPv6 literal alone — its colons are part of the address.
    let host = match host.strip_prefix('[') {
        Some(_) => host.split(']').next().map(|h| &h[1..])?,
        None => host.split(':').next()?,
    };
    (!host.is_empty()).then_some(host)
}

#[cfg(test)]
mod tests {
    use super::{host_of_ws_base, lookup};
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn env_object_wins_over_query() {
        let e = env(&[("user", "fromenv")]);
        assert_eq!(
            lookup(&e, "?user=fromquery", "user").as_deref(),
            Some("fromenv")
        );
    }

    #[test]
    fn credentials_from_query_need_the_dev_flag() {
        let e = env(&[]);
        assert_eq!(lookup(&e, "?user=alice&pass=pw&char=Al", "user"), None);
        assert_eq!(lookup(&e, "?user=alice&pass=pw&char=Al", "pass"), None);
        assert_eq!(lookup(&e, "?user=alice&pass=pw&char=Al", "char"), None);
        let e = env(&[("dev_query_creds", "1")]);
        assert_eq!(
            lookup(&e, "?user=alice&pass=pw", "user").as_deref(),
            Some("alice")
        );
        assert_eq!(
            lookup(&e, "?user=alice&pass=pw", "pass").as_deref(),
            Some("pw")
        );
    }

    #[test]
    fn non_credential_keys_always_read_the_query() {
        let e = env(&[]);
        assert_eq!(
            lookup(&e, "?host=realm.example&renderscale=0.5", "host").as_deref(),
            Some("realm.example")
        );
        assert_eq!(
            lookup(&e, "?renderscale=0.5", "renderscale").as_deref(),
            Some("0.5")
        );
        assert_eq!(lookup(&e, "", "host"), None);
    }

    #[test]
    fn query_values_are_percent_decoded_and_malformed_pairs_skipped() {
        let e = env(&[]);
        assert_eq!(
            lookup(&e, "?host=a%20b&junk&host2=%zz", "host").as_deref(),
            Some("a b")
        );
        assert_eq!(lookup(&e, "?host=%zz", "host"), None);
    }

    /// The realmlist default follows the GATEWAY, not the page — the whole correction behind
    /// `default_wow_host` on a static-host setup, where the page's own hostname is a CDN edge with
    /// no realmd behind it.
    #[test]
    fn the_ws_base_yields_the_gateway_hostname() {
        assert_eq!(
            host_of_ws_base("wss://gw.example.org:8443"),
            Some("gw.example.org")
        );
        assert_eq!(
            host_of_ws_base("ws://gw.example.org"),
            Some("gw.example.org")
        );
        assert_eq!(
            host_of_ws_base("wss://gw.example.org/"),
            Some("gw.example.org")
        );
        // No scheme, and an IPv6 literal whose colons are part of the address, not a port.
        assert_eq!(
            host_of_ws_base("gw.example.org:443"),
            Some("gw.example.org")
        );
        assert_eq!(
            host_of_ws_base("wss://[2001:db8::1]:8443"),
            Some("2001:db8::1")
        );
        // A value that does not parse must be no worse than an absent one: the caller falls back.
        assert_eq!(host_of_ws_base(""), None);
        assert_eq!(host_of_ws_base("wss://"), None);
    }
}
