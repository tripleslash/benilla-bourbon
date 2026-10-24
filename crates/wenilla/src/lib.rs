//! The browser entry point — benilla's third launcher shim, after `benilla` and
//! `benilla-worldview` (decision 1160), and like them ~20 lines whose only job is to carry the
//! build-id stamp so a commit dirties this crate and not `benilla-app` (decision 0993).
//!
//! `web/index.html` loads the `wasm-bindgen --target web` glue this crate compiles to
//! (`trunk build --release`), which calls `init()` then this module's exported [`start`] —
//! `#[wasm_bindgen(start)]` also runs it automatically the moment the module finishes
//! instantiating, so the two are equivalent; the page calls it explicitly only to control
//! sequencing with `wasi_stubs.js`'s `bind(memory)` (see that file's own header for why the wasm
//! Lua VM needs it).

use wasm_bindgen::prelude::*;

use benilla_app::BuildId;

/// Set the panic hook (so a Rust panic reaches `console.error` with a real message instead of
/// the browser's bare "unreachable executed"), then hand off to [`benilla_app::run`] exactly like
/// the native `benilla` shim does — same `BuildId`, same entry point, so behaviour that differs
/// between the two targets differs inside `benilla_app`, never at this seam.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    // **Worker threads instantiate this same module, and `start` runs on every instantiation.**
    // That is what `#[wasm_bindgen(start)]` means: the glue calls `__wbindgen_start` from
    // `initSync`, and a pool thread (`web/threadworker.js`) calls `initSync` to attach itself to
    // the page's shared memory. Without this line every worker would boot a second copy of the
    // whole client — `benilla_app::run` down to `WindowPlugin` — and die on the first thing that
    // wants a document. Measured before it existed: 15 workers, 15 opaque `Worker` error events,
    // and no pool with anything in it.
    //
    // The test is `document`, not `window`: a `DedicatedWorkerGlobalScope` has neither, but
    // `document` is the one that means "there is a page here" rather than "there is a global here".
    // A worker returns immediately and its real entry point takes over — `__wenilla_thread_entry`,
    // which the same message that carried the module hands it.
    if !js_sys::Reflect::has(&js_sys::global(), &JsValue::from_str("document")).unwrap_or(false) {
        return;
    }
    // `AppExit` is discarded deliberately: on wasm32, Bevy's winit runner hands the event loop to
    // the browser (`requestAnimationFrame`) and returns here immediately rather than blocking
    // until the app actually closes, so the value this call produces answers a question — "what
    // did the run exit with?" — that hasn't been decided yet. The native shims' `fn main() ->
    // AppExit` (where the return value IS the process exit code) has no wasm32 equivalent: a
    // browser tab has no exit code for a page to report.
    let build = BuildId {
        version: env!("CARGO_PKG_VERSION"),
        describe: env!("BENILLA_GIT_DESCRIBE"),
        sha: env!("BENILLA_GIT_SHA"),
        short: env!("BENILLA_GIT_SHORT"),
        date: env!("BENILLA_GIT_DATE"),
        profile: env!("BENILLA_PROFILE"),
        ..Default::default()
    };
    // `?worldview=1` — the engine with no game attached, over the same packs (fork-owned; the
    // browser twin of `benilla-worldview`, used to look at the lighting work without a login).
    if benilla_app::wants_worldview() {
        benilla_app::run_worldview(build);
        return;
    }
    benilla_app::run(build);
}
