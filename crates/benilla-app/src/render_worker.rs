//! The render worker's platform half (fork: the browser build's render thread): start the Web
//! Worker the render sub-app moves to, and answer `bevy::render::render_worker`'s two calls into
//! the page. The mechanism is in the bevy_render fork (`third_party/bevy_render/BENILLA.md`, "The
//! render worker"); `web/threads.js` (`startRender`, `kickRender`) and `web/renderworker.js` are
//! the page's side of the same contract.
//!
//! Started when the page shipped the bridge (a threaded, cross-origin isolated build without
//! `?nothreads=1`) and the build renders with WebGPU. `?renderworker=0` keeps the render app on
//! the page's thread, the single-threaded path the build had before, so the two can be A/B'd live.

/// Start the render worker, if this build and page can. Call before the `App` is built:
/// `RenderPlugin::build` reads the mode.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics", feature = "webgpu")))]
pub fn install() {}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics", feature = "webgpu"))]
pub use imp::install;

#[cfg(all(target_arch = "wasm32", target_feature = "atomics", feature = "webgpu"))]
mod imp {
    use bevy::app::App;
    use bevy::render::render_worker;
    use js_sys::Function;
    use wasm_bindgen::prelude::*;

    /// The canvas the client draws into, as `benilla_world::boot` names it to winit.
    const CANVAS: &str = "#benilla";

    /// The worker's stack, which must be a multiple of 64 KiB: the render schedule and every
    /// plugin `finish` used to run on the page's 1 MiB, and wasm has no guard page to say when a
    /// stack runs over, so it gets twice that.
    const STACK: u32 = 2 << 20;

    /// See [`super::install`].
    pub fn install() {
        if crate::webenv::var("WOW_RENDERWORKER").as_deref() == Some("0") {
            web_sys::console::log_1(
                &"wenilla: ?renderworker=0 — rendering on the page thread".into(),
            );
            return;
        }
        let Some((bridge, start)) = crate::web_threads::bridge("startRender") else {
            return;
        };
        let args = js_sys::Array::of4(
            &wasm_bindgen::module(),
            &wasm_bindgen::memory(),
            &CANVAS.into(),
            &JsValue::from_f64(f64::from(STACK)),
        );
        if !start.apply(&bridge, &args).is_ok_and(|v| v.is_truthy()) {
            return;
        }
        // Before the worker can read anything through `webenv`: it has no `window`.
        crate::webenv::publish_for_workers();
        render_worker::install(kick);
        // `?rwtrace=1`: the handshake's timeline (`__wenilla_render_trace`).
        if crate::webenv::var("WOW_RWTRACE").as_deref() == Some("1") {
            render_worker::set_trace(true);
        }
        // The device is made on the worker; its arrival must wake the page's event loop.
        render_worker::set_wake_page(bevy::winit::wake_event_loop);
        bevy::winit::set_finish_plugins(finish_on_worker);
        // An update whose extract the worker could not take yet is held, not waited on.
        bevy::winit::set_may_update(render_worker::before_update);
        // And while the page waits in the browser, the worker may take that extract itself.
        bevy::winit::set_lend_world(render_worker::lend, render_worker::reclaim);
        web_sys::console::log_1(
            &"wenilla: rendering on a worker (?renderworker=0 to compare)".into(),
        );
    }

    thread_local! {
        /// `__wenilla_threads.kickRender`, looked up once: it is called every extracted frame.
        static KICK: Option<(JsValue, Function)> = crate::web_threads::bridge("kickRender");
    }

    /// Wake the worker's event loop.
    fn kick() {
        KICK.with(|kick| {
            if let Some((bridge, f)) = kick {
                let _ = f.call0(bridge);
            }
        });
    }

    /// The runner's `App::finish` and `App::cleanup`, run on the worker: every plugin `finish`
    /// that reads the device or builds a GPU resource then runs where the device lives, and the
    /// render worker plugin's `cleanup` parks the render app there.
    fn finish_on_worker(app: &mut App) {
        let app = app as *mut App as usize;
        let finish = move || {
            // SAFETY: the pointer is the runner's `&mut App`, alive for the whole blocking call
            // below, during which this thread does not touch the `App`.
            let app = unsafe { &mut *(app as *mut App) };
            app.finish();
            app.cleanup();
        };
        // SAFETY: `run_blocking` spins until the job returns, and this thread touches nothing of
        // the `App` meanwhile. Everything the plugins reach in it is `Send` but `NonSend`
        // resources, which bevy checks by thread.
        unsafe { render_worker::run_blocking(finish) };
    }

    /// The worker's first call, with the canvas the page transferred to it
    /// (`web/renderworker.js`). **Not for anything else to call.**
    #[wasm_bindgen]
    pub fn __wenilla_render_entry(canvas: JsValue) {
        render_worker::enter(canvas);
    }

    /// The handshake's state as one line (`bevy::render::render_worker::describe`), for a console
    /// on the page: `__wenilla_render_state()` when the client looks stuck.
    #[wasm_bindgen]
    pub fn __wenilla_render_state() -> String {
        render_worker::describe()
    }

    /// The handshake's counters since the last call (`render_worker::take_stats`), as one line, for
    /// `?gpucensus=1` (`web/renderworker.js`), which asks every 5 s; `webdiag` leaves the
    /// counters to it then.
    #[wasm_bindgen]
    pub fn __wenilla_render_stats() -> String {
        let s = render_worker::take_stats();
        let ms = |us: u64| us as f32 / 1000.0;
        let avg = |sum: u64, n: u64| if n > 0 { ms(sum) / n as f32 } else { 0.0 };
        format!(
            "renders={} render_avg={:.2}ms render_max={:.2}ms extracts={} wait_avg={:.2}ms \
             wait_max={:.2}ms owed={} lent={} lent_paid={} skipped={} withdrawn={}",
            s.renders,
            avg(s.render_us, s.renders),
            ms(s.render_max_us),
            s.extracts,
            avg(s.wait_us, s.extracts),
            ms(s.wait_max_us),
            s.owed,
            s.lent,
            s.lent_paid,
            s.skipped,
            s.withdrawn,
        )
    }

    /// The handshake's timeline since the last call (`render_worker::trace`, `?rwtrace=1`), as
    /// `kind arg µs` lines, for a console on the page. Nothing else calls it.
    #[wasm_bindgen]
    pub fn __wenilla_render_trace() -> String {
        render_worker::trace()
    }

    /// Each wake of the worker's event loop (`web/renderworker.js`); `true` when it rendered, and
    /// the worker then calls [`__wenilla_render_settle`] from its next task. **Not for anything
    /// else to call.**
    #[wasm_bindgen]
    pub fn __wenilla_render_poll() -> bool {
        render_worker::poll()
    }

    /// The first task after a poll or a settle that rendered (`web/renderworker.js`): the worker
    /// is free again, and renders a world the page lent it meanwhile; `true` when it did, and the
    /// worker then settles again from its next task. **Not for anything else to call.**
    #[wasm_bindgen]
    pub fn __wenilla_render_settle() -> bool {
        render_worker::settle()
    }
}
