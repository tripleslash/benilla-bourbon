// threads.js — the page's half of `bevy_tasks`'s thread spawner.
//
// The forked `bevy_tasks` (`third_party/bevy_tasks/BENILLA.md`) asks the host for a thread instead
// of calling `std::thread::spawn`, because that function does not exist on
// `wasm32-unknown-unknown` — std gives that target `sleep` and takes everything else from its
// `unsupported` backend, atomics or no atomics. A thread here is a Web Worker running the same
// module against the same `SharedArrayBuffer`, and only the page can make one.
//
// **Page contract: `window.__wenilla_threads`** — a plain global with one method, the same shape as
// every other seam this client uses (`__wenilla_env`, `__wenilla_packs`, `__wenilla_progress`,
// `__wenilla_boottrace`). No wasm-bindgen import, so this stays an ordinary script that the Rust
// side reaches through `Reflect`.
//
// **A page that never calls `installThreads()` still runs.** Rust then finds no bridge, every pool
// comes up with zero workers, and `TaskPool::scope` does the work on the calling thread — the
// single-threaded behaviour the web build has always had. `?nothreads=1` is that path on purpose,
// and it is worth exercising: it is also what a browser that refuses a Worker will produce.
//
// ## What actually crosses the boundary
//
// `module` and `memory` are the page's own `WebAssembly.Module` and `WebAssembly.Memory`, read out
// of wasm-bindgen by Rust (`wasm_bindgen::module()` / `::memory()`) and passed straight back down.
// Both are structured-cloneable, so `postMessage` shares them rather than copying — the worker gets
// the same heap and the same compiled code, and the 77 MB module is not downloaded or compiled a
// second time.
//
// `ptr` is a boxed Rust closure — the pool's worker loop — as one integer. It is handed to exactly
// one worker, which consumes it exactly once. **It must not be sent anywhere else**: a second
// consumer would be a double free. That is why this function posts it and does not keep it.

/// Where the worker script lives, relative to the page, exactly as `packs.js` names `packworker.js`.
const WORKER_URL = './threadworker.js';

/// Install the bridge. Call before the wasm module starts, because the pools are built during
/// `init()` and a spawner that arrives afterwards has nothing left to spawn for.
export function installThreads() {
  if (new URLSearchParams(location.search).get('nothreads') === '1') {
    console.warn('wenilla: ?nothreads=1 — task pools will come up with no workers (serial).');
    return;
  }
  if (!self.crossOriginIsolated) {
    // Not fatal here, and not this function's failure to report: a shared-memory module cannot
    // instantiate at all without isolation, so `boot.js` will say so with a better message in a
    // moment. Saying nothing would be worse — if this is a PLAIN build the page runs fine and the
    // absence of workers is expected, and this line is what tells the two cases apart.
    console.warn('wenilla: page is not cross-origin isolated — no worker threads.');
    return;
  }

  let started = 0;
  window.__wenilla_threads = {
    /// Start one pool thread. Returns true only if the Worker was constructed and given its work;
    /// anything else is false, and Rust takes its closure back rather than leaking it.
    spawn(name, ptr, stackSize, module, memory) {
      try {
        const worker = new Worker(WORKER_URL, { type: 'module', name });
        // An error here means the worker script itself failed to load or threw at top level —
        // a stale `web/dist`, a bad MIME type. The pool cannot be told after the fact (it has
        // already counted this thread), so it comes up one worker short and says so here.
        worker.onerror = (e) => {
          console.error(`wenilla: thread worker "${name}" failed —`, e.message || e);
        };
        worker.postMessage({ module, memory, ptr, stackSize });
        started += 1;
        return true;
      } catch (e) {
        console.error('wenilla: could not start a thread worker —', e);
        return false;
      }
    },

    /// How many workers were handed out. Read by nothing; here because "did threads actually
    /// happen" is the first question of every performance report about this build, and the answer
    /// should not require a debugger. `__wenilla_threads.started()` in the console.
    started: () => started,
  };
}
