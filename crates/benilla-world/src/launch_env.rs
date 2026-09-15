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
