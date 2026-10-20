//! Web Worker threads for the browser build's task pools — the platform half of
//! `bevy_tasks::thread_spawner`.
//!
//! `std::thread::spawn` does not exist on `wasm32-unknown-unknown`, with or without `+atomics`:
//! std's `sys/thread/mod.rs` gives that target `sleep` and takes everything else from
//! `unsupported.rs`. A thread there is a Web Worker instantiating the *same* module against the
//! *same* `SharedArrayBuffer` memory, which only the page can arrange. So the forked `bevy_tasks`
//! asks for a thread through a hook, and this is what answers.
//!
//! ## Why the Worker is built in JS
//!
//! The same reason `benilla_formats::webpack::resident` reaches the pack store through
//! `window.__wenilla_packs`: a plain page global, no wasm-bindgen module import, so the page can
//! ship it as an ordinary script. Here it also keeps `Worker`/`WorkerOptions` out of this crate's
//! `web-sys` feature list, and it puts the module/memory handoff beside the `init()` call it has to
//! match — `web/threads.js` and `web/threadworker.js` are two halves of one contract, and both are
//! JS.
//!
//! **A page that ships neither still runs.** [`install`] then leaves no spawner installed, every
//! pool comes up with zero workers, and `TaskPool::scope` does the work on the calling thread — the
//! single-threaded behaviour the web build has always had. That is the designed fallback and it is
//! worth testing deliberately (`?nothreads=1`), because it is also what a browser that refuses a
//! Worker for its own reasons will produce.
//!
//! ## What this does NOT do
//!
//! It does not make asset loading concurrent. `TaskPool::spawn` on this target is still the
//! browser's microtask queue, because the asset path awaits `JsFuture`s and reaches `window` —
//! `JsValue`s belong to one agent and cannot cross to a worker. Only `spawn_pooled` and `scope`
//! reach these threads: the collider builds, the `par_iter_mut` sweeps, and (if it ever grades out)
//! a multi-threaded ECS schedule.

/// Install the browser's thread spawner and tell `bevy_tasks` how wide this machine is.
///
/// Must run **before** the first task pool is built — `TaskPoolPlugin` builds all three inside
/// `Plugin::build`, so this belongs at the very top of [`crate::run`], ahead of `add_plugins`.
///
/// Compiles to nothing except on a browser build with shared memory. Everywhere else `bevy_tasks`
/// already has `std::thread` and a working `available_parallelism`, and installing anything would
/// only be a way to get those wrong.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
pub fn install() {}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub use imp::{install, spawn_detached};
// The render worker (`crate::render_worker`) is started through the same page global.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub(crate) use imp::bridge;

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod imp {
    use bevy::tasks::{
        set_available_parallelism, set_thread_spawner, SpawnedThread, ThreadRequest,
    };
    use js_sys::{Function, Reflect};
    use wasm_bindgen::{JsCast, JsValue};

    /// The page global this module reaches, in the shape every other page seam here uses
    /// (`__wenilla_env`, `__wenilla_packs`, `__wenilla_progress`, `__wenilla_boottrace`).
    const GLOBAL: &str = "__wenilla_threads";

    /// How many worker threads to ask for, given what the browser says the machine has.
    ///
    /// `hardwareConcurrency - 1`, floored at 1. The subtraction is the point: the page's own thread
    /// is not free — it draws, it runs every `NonSend` system, and it is the thread that *waits* in
    /// `TaskPool::scope` while the workers run. Asking for one worker per core would oversubscribe
    /// by exactly one thread on every machine, and the thread it oversubscribes is the frame.
    ///
    /// Browsers also lie here, deliberately and downward (privacy budgets, efficiency cores). That
    /// is fine: the number only sizes the pools, and being one worker short costs a little
    /// throughput while being one long costs frame time.
    fn worker_threads() -> usize {
        let cores = web_sys::window()
            .map(|w| w.navigator().hardware_concurrency())
            .filter(|n| n.is_finite() && *n >= 1.0)
            .map(|n| n as usize)
            .unwrap_or(1);
        cores.saturating_sub(1).max(1)
    }

    /// Resolve one function off the page global, or `None` if the page ships no bridge.
    pub(crate) fn bridge(method: &str) -> Option<(JsValue, Function)> {
        let window = web_sys::window()?;
        let obj = Reflect::get(&window, &GLOBAL.into()).ok()?;
        if !obj.is_object() {
            return None;
        }
        let f = Reflect::get(&obj, &method.into())
            .ok()?
            .dyn_into::<Function>()
            .ok()?;
        Some((obj, f))
    }

    /// The body of a pool thread, boxed twice so it crosses the JS boundary as one integer.
    ///
    /// `Box<dyn FnOnce()>` is a *fat* pointer — data plus vtable — and `postMessage` carries
    /// numbers. Boxing the box gives a thin pointer to it; the worker hands that same integer back
    /// to [`wenilla_thread_entry`], which is the only thing that ever reads it, exactly once.
    type Body = Box<dyn FnOnce() + Send + 'static>;

    /// Start one Web Worker running `body`, or hand `body` back if the page would not.
    ///
    /// The whole mechanism, factored out of [`spawn`] so it has a second caller: the audio render
    /// thread (`crate::sound::web_output`). That thread is not a pool thread — it runs one loop for
    /// the session and must not occupy a pool worker — but it needs exactly this, a Web Worker
    /// running the same module against the same memory, and there is no second way to make one.
    fn start_worker(name: &str, stack_size: Option<usize>, body: Body) -> Result<(), Body> {
        let Some((obj, spawn_fn)) = bridge("spawn") else {
            // No bridge on the page. Not an error — see the module header.
            return Err(body);
        };

        let ptr = Box::into_raw(Box::new(body)) as u32;
        let args = js_sys::Array::of5(
            &name.into(),
            &JsValue::from_f64(f64::from(ptr)),
            &stack_size.map_or(JsValue::NULL, |n| JsValue::from_f64(n as f64)),
            &wasm_bindgen::module(),
            &wasm_bindgen::memory(),
        );

        match spawn_fn.apply(&obj, &args) {
            Ok(v) if v.is_truthy() => Ok(()),
            _ => {
                // The worker never started, so nothing will ever claim that pointer. Take it back
                // rather than leaking a closure (and, with it, everything it captured).
                //
                // SAFETY: `ptr` came from `Box::into_raw` immediately above and has not been handed
                // to a worker — `spawn` either started one and returned truthy, or it did not.
                Err(unsafe { *Box::from_raw(ptr as *mut Body) })
            }
        }
    }

    /// Start one pool thread as a Web Worker.
    fn spawn(request: ThreadRequest) -> Result<SpawnedThread, ThreadRequest> {
        let ThreadRequest {
            name,
            stack_size,
            body,
        } = request;
        match start_worker(&name, stack_size, body) {
            Ok(()) => Ok(SpawnedThread::detached()),
            Err(body) => Err(ThreadRequest {
                name,
                stack_size,
                body,
            }),
        }
    }

    /// Start a **non-pool** worker running `body` once. `true` if the page started one.
    ///
    /// For work that owns a thread for the session rather than taking one from a pool — see
    /// [`start_worker`]. Dropping the returned unit does not stop it; the body is expected to
    /// watch its own stop flag.
    pub fn spawn_detached(name: &str, body: impl FnOnce() + Send + 'static) -> bool {
        start_worker(name, None, Box::new(body)).is_ok()
    }

    /// The entry point a spawned worker calls once it has instantiated the module against the
    /// page's shared memory. **Not for anything else to call.**
    ///
    /// Exported into the main bundle's glue (`wenilla.js`) because that is the module a worker
    /// imports; `web/threadworker.js` is the only caller, and it passes back the integer it was
    /// given. Returns when the pool that owns this thread is dropped, at which point the worker
    /// closes itself.
    #[wasm_bindgen::prelude::wasm_bindgen]
    pub fn __wenilla_thread_entry(ptr: u32) {
        // SAFETY: `ptr` is the value `spawn` produced with `Box::into_raw`, delivered to exactly
        // one worker by exactly one `postMessage`, and this consumes it. A second call with the
        // same pointer would be a double free — which is why nothing but the worker script has the
        // number, and why the worker sends it nowhere else.
        let body = unsafe { *Box::from_raw(ptr as *mut Body) };
        body();
    }

    /// See [`super::install`].
    pub fn install() {
        // This is the page's thread, which traps instead of parking on a contended lock; bevy's
        // locks spin here and park on the workers (`third_party/bevy_platform/BENILLA.md`).
        bevy::platform::sync::forbid_parking_on_this_thread();
        set_available_parallelism(worker_threads());
        // An `Err` here means a spawner was already installed, which can only happen if this ran
        // twice. The first one is the one in use either way, so there is nothing to report.
        let _ = set_thread_spawner(spawn);
    }
}
