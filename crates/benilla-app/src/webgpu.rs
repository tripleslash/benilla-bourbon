//! fork: web build — the client's side of `web/gpu.js`'s recovery ladder.
//!
//! After a lost GPU device the page reloads one rung down the ladder and remembers the rung in
//! `localStorage` (`wenilla.gpu`). That guess about the machine belongs to the page only until the
//! player makes their own: when they apply Graphics settings ([`crate::cvars::GRAPHICS_CVARS`]),
//! the client calls `window.__wenilla_gpu.reset()`, which persists tier 0 and lifts the pixel
//! budget on the spot. The render-scale half of the tier is the CVar host's to release
//! (`Cvars::release_fallbacks`).
//!
//! The same contract as [`crate::webprogress`]: a page without the hook costs one failed
//! `Reflect::get`, and every error is swallowed. Native builds: a no-op; there is no page.

/// Ask the page to retire its recovery tier. Returns whether it had one to clear.
#[cfg(target_arch = "wasm32")]
pub(crate) fn settings_applied() -> bool {
    use wasm_bindgen::JsCast;
    let Some(window) = web_sys::window() else {
        return false;
    };
    let Ok(gpu) = js_sys::Reflect::get(&window, &"__wenilla_gpu".into()) else {
        return false;
    };
    if !gpu.is_object() {
        return false; // a page without the guard
    }
    let Ok(reset) = js_sys::Reflect::get(&gpu, &"reset".into()) else {
        return false;
    };
    let Ok(reset) = reset.dyn_into::<js_sys::Function>() else {
        return false; // an older gpu.js, from before the client could clear its tier
    };
    reset
        .call0(&gpu)
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn settings_applied() -> bool {
    false
}
