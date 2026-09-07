// threadworker.js — one pool thread. A Web Worker that becomes a `bevy_tasks` worker thread.
//
// This is the smaller half of the contract in `web/threads.js`; read that one first for why the
// page owns worker creation at all. What happens here is three lines of real work:
//
//   1. instantiate the SAME module against the SAME memory the page is already running on, so this
//      worker shares the heap rather than getting one of its own;
//   2. call back into Rust with the pointer the page handed over, which is a boxed closure — the
//      pool's worker loop;
//   3. that loop runs until the pool is dropped, and then this worker closes itself.
//
// **`initSync`, not `init`.** The page passes an already-compiled `WebAssembly.Module`, so there is
// nothing to fetch and nothing to compile — instantiation is synchronous and the 77 MB module is
// never downloaded twice. (`new WebAssembly.Instance` on a *compiled* module is allowed on any
// thread; the size limit that makes people reach for the async form applies to compiling bytes.)
//
// **The worker blocks, and that is the point.** `bevy_tasks`'s worker loop parks on
// `memory.atomic.wait32` when it has nothing to do. That instruction traps on the page's main
// thread and is perfectly legal here, which is the whole reason there is a worker.

import { initSync, __wenilla_thread_entry } from './wenilla.js';

self.onmessage = (event) => {
  const { module, memory, ptr, stackSize } = event.data || {};

  // One message, one thread. A second would try to instantiate over a module that is already
  // running this worker's loop, and the loop does not return until the pool shuts down.
  self.onmessage = null;

  try {
    // `thread_stack_size` must be a multiple of 64 KiB or the glue throws; `undefined` takes
    // wasm-bindgen's own default, which is what the pools ask for unless someone sets
    // `TaskPoolBuilder::stack_size`. If a deeply recursive job ever faults a worker — parry's QBVH
    // build is the candidate — this is the knob, and it is passed from Rust so it stays one number
    // in one place.
    initSync({ module, memory, thread_stack_size: stackSize ?? undefined });
  } catch (e) {
    // The page cannot see this failure any other way: it handed the closure over and moved on, and
    // a worker that dies before entering the loop simply never runs any tasks. The pool survives it
    // (it comes up smaller), so this is a diagnostic, not a fatal.
    console.error('wenilla: thread worker failed to initialise —', e);
    self.close();
    return;
  }

  // Returns when the pool that owns this thread is dropped. Until then this call IS the thread.
  __wenilla_thread_entry(ptr);
  self.close();
};
