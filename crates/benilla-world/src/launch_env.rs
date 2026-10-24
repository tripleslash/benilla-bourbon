//! **Launch-time settings a shim can pre-seed** (fork-owned; `modern/HOOKS.md`).
//!
//! The browser has no process environment — `std::env::var` always answers `NotPresent` on
//! wasm32 — and this crate has no query-string reader of its own (that is `benilla_app::webenv`,
//! one layer up). So the browser's world viewer (`?worldview=1`) has its launcher copy the few
//! `WOW_*` keys the engine reads out of the page's query string into here before the app is
//! built. Native launchers seed nothing, and [`var`] falls through to the real environment, so a
//! native run behaves exactly as before.

use std::collections::HashMap;
use std::sync::OnceLock;

static SEEDED: OnceLock<HashMap<String, String>> = OnceLock::new();

/// Record values for this process. The first call wins; later calls are ignored.
pub fn seed(pairs: impl IntoIterator<Item = (String, String)>) {
    let _ = SEEDED.set(pairs.into_iter().collect());
}

/// A seeded value first, then the process environment.
pub fn var(name: &str) -> Option<String> {
    SEEDED
        .get()
        .and_then(|m| m.get(name).cloned())
        .or_else(|| std::env::var(name).ok())
}

/// fork: modern lighting — a dev reader the shim may install for values that can change while the
/// app runs (the browser's live query string, rewritten with `history.replaceState`). Tuning knobs
/// read through [`live_var`] then follow the page without a reload.
static LIVE: OnceLock<fn(&str) -> Option<String>> = OnceLock::new();

/// Install the live reader. The first call wins.
pub fn set_live_reader(reader: fn(&str) -> Option<String>) {
    let _ = LIVE.set(reader);
}

/// How long a live answer is reused. Systems ask every frame, and on the browser one read is the
/// page env object, `location.search` and a percent-decode of every pair across the JS bridge; a
/// knob edited with `history.replaceState` is a person typing, for whom half a second is instant.
const LIVE_TTL: std::time::Duration = std::time::Duration::from_millis(500);

type LiveCache = HashMap<String, (Option<String>, bevy::platform::time::Instant)>;

/// The live value if a reader is installed, else [`var`].
pub fn live_var(name: &str) -> Option<String> {
    let Some(read) = LIVE.get() else {
        return var(name);
    };
    // `try_lock`, never `lock`: a contended wait is `Atomics.wait`, which traps on the browser's
    // main thread. A miss just reads through.
    static CACHE: std::sync::Mutex<Option<LiveCache>> = std::sync::Mutex::new(None);
    let now = bevy::platform::time::Instant::now();
    if let Ok(cache) = CACHE.try_lock() {
        if let Some((value, at)) = cache.as_ref().and_then(|c| c.get(name)) {
            if now.duration_since(*at) < LIVE_TTL {
                return value.clone();
            }
        }
    }
    let value = read(name);
    if let Ok(mut cache) = CACHE.try_lock() {
        cache
            .get_or_insert_with(HashMap::new)
            .insert(name.to_string(), (value.clone(), now));
    }
    value
}
