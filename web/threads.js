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
/// The render worker's script (`benilla_app::render_worker`), beside it.
const RENDER_WORKER_URL = './renderworker.js';

/// `?gpucensus=1` (`gpucensus.js`): the render worker posts its counts here, and with
/// `?renderworker=0` the page counts its own. Without the flag the module is never loaded.
const CENSUS = new URLSearchParams(location.search).get('gpucensus') === '1';
const censusLast = {};
function printCensus(where, report) {
  import('./gpucensus.js').then(({ censusLine }) => {
    console.log(censusLine(where, report, censusLast[where]));
    censusLast[where] = report.made;
  });
}

/// The page's own census, for the leg that renders here: linear memory from the first pool spawn
/// (the page's externref table is not reachable from here, so `refs` reads 0).
let censusMemory = null;
function startPageCensus() {
  import('./gpucensus.js').then(({ installGpuCensus }) => {
    const snapshot = installGpuCensus();
    setInterval(() => {
      const mib = Math.round((censusMemory?.buffer?.byteLength ?? 0) / 1048576);
      printCensus('page', { ...snapshot(), refs: 0, mib });
    }, 5000);
  });
}

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
  let render = null;
  // `?rwtrace=1`: the page's animation frames for the render worker's timeline
  // (`__wenilla_render_trace`): `[callback entry, frame time]`, both `timeOrigin + now()` in ms.
  if (new URLSearchParams(location.search).get('rwtrace') === '1') {
    const frames = (window.__wenilla_rwframes = []);
    const raf = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = (cb) =>
      raf((t) => {
        if (frames.length < 100000) {
          frames.push([performance.timeOrigin + performance.now(), performance.timeOrigin + t]);
        }
        cb(t);
      });
  }
  // With `?renderworker=0` the page renders, so the census counts here.
  if (CENSUS && new URLSearchParams(location.search).get('renderworker') === '0') {
    startPageCensus();
  }
  window.__wenilla_threads = {
    /// Start one pool thread. Returns true only if the Worker was constructed and given its work;
    /// anything else is false, and Rust takes its closure back rather than leaking it.
    spawn(name, ptr, stackSize, module, memory) {
      if (CENSUS) censusMemory ??= memory;
      try {
        const worker = new Worker(WORKER_URL, { type: 'module', name });
        // An error here means the worker script itself failed to load or threw at top level —
        // a stale `web/dist`, a bad MIME type. The pool cannot be told after the fact (it has
        // already counted this thread), so it comes up one worker short and says so here.
        //
        // It is also where a worker's out-of-memory abort surfaces: `rust_oom` traps with a bare
        // `unreachable` and no panic message (the default alloc-error hook writes to a stderr this
        // target does not have), so the size of linear memory at that moment is the only evidence
        // of which it was. At the `--max-memory` ceiling (4096 MiB, `scripts/web-build-threads.sh`)
        // it is exhaustion; well under it, one absurd allocation.
        worker.onerror = (e) => {
          const mib = Math.round(memory.buffer.byteLength / 1048576);
          console.error(
            `wenilla: thread worker "${name}" failed (linear memory ${mib} MiB) —`,
            e.message || e,
          );
        };
        worker.postMessage({ module, memory, ptr, stackSize });
        started += 1;
        return true;
      } catch (e) {
        console.error('wenilla: could not start a thread worker —', e);
        return false;
      }
    },

    /// Start the render worker: the render sub-app moves there (`bevy::render::render_worker`),
    /// and so does the canvas, as an `OffscreenCanvas`. True only if the worker exists and the
    /// canvas went to it; anything else is false and Rust renders on this thread as before.
    ///
    /// The worker is made first and the canvas transferred last: a transferred canvas can never
    /// again have a context on this page, so it moves only once nothing can fail after it.
    startRender(module, memory, selector, stackSize) {
      if (render) return false;
      const canvas = document.querySelector(selector);
      if (!canvas || typeof canvas.transferControlToOffscreen !== 'function') return false;
      let worker = null;
      try {
        worker = new Worker(RENDER_WORKER_URL, { type: 'module', name: 'render' });
        worker.onerror = (e) => {
          const mib = Math.round(memory.buffer.byteLength / 1048576);
          console.error(
            `wenilla: the render worker failed (linear memory ${mib} MiB) —`,
            e.message || e,
          );
        };
        // The worker's device is not one `gpu.js` wrapped, so the worker reports its own losses
        // and errors here, and the guard acts on them as on its own.
        worker.onmessage = (e) => {
          const msg = e.data || {};
          const gpu = window.__wenilla_gpu;
          if (msg.gpu === 'lost') gpu?.deviceLost?.(msg);
          else if (msg.gpu === 'error') gpu?.deviceError?.(msg.kind || '');
          else if (msg.census) printCensus('worker', msg.census);
        };
        const offscreen = canvas.transferControlToOffscreen();
        worker.postMessage({ module, memory, canvas: offscreen, stackSize, census: CENSUS }, [
          offscreen,
        ]);
      } catch (e) {
        console.error('wenilla: could not start the render worker —', e);
        worker?.terminate();
        return false;
      }
      render = worker;
      started += 1;
      return true;
    },

    /// Wake the render worker's event loop: an extract is waiting (`render_worker::poll`).
    /// Called once per extracted frame, so it carries nothing.
    kickRender() {
      render?.postMessage(0);
    },

    /// How many workers were handed out. Read by nothing; here because "did threads actually
    /// happen" is the first question of every performance report about this build, and the answer
    /// should not require a debugger. `__wenilla_threads.started()` in the console.
    started: () => started,
  };
}
