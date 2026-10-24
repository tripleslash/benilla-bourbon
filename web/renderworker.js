// renderworker.js — the render worker. A Web Worker that holds the client's render sub-app, its
// WebGPU device and the canvas, for the whole session.
//
// The design is `third_party/bevy_render/BENILLA.md`, "The render worker"; `web/threads.js`
// (`startRender`) is the page's half. Like `threadworker.js` it instantiates the SAME module
// against the SAME memory the page runs on. Unlike a pool thread it never blocks: the browser
// presents an `OffscreenCanvas` and settles the renderer's promises (async pipeline creation, the
// device itself) only between this worker's tasks, so each wake runs one step and returns.
//
//   1. the first message carries the module, the memory and the transferred `OffscreenCanvas`;
//   2. every later message is a wake: run `render_worker::poll` (boot jobs, then an extract and a
//      render if the page parked its world for one), and return to the event loop, which presents;
//   3. after a poll that rendered, the next task of this worker settles it (`render_worker::settle`):
//      only then may the page publish another extract. Between the two tasks the browser can hold
//      this worker until the page's thread is free, so the page must not be spinning on it then.
//      A settle also renders a world the page lent while it waits in its own event loop, and is
//      then followed by another settle, as a poll that rendered is. The settle after a render
//      waits until the GPU has at most `IN_FLIGHT` frames of this worker's in hand.

import {
  initSync,
  __wenilla_render_entry,
  __wenilla_render_poll,
  __wenilla_render_settle,
  __wenilla_render_stats,
} from './wenilla.js';

/// `web/gpu.js` wraps `requestDevice` on the page to watch for a lost or failing device, and a
/// device made here is not one it sees. So this worker wraps its own and forwards what it observes;
/// `threads.js` hands it to the guard, which acts on it exactly as on a page device.
function watchDevices() {
  if (!self.GPUAdapter) return;
  const orig = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc) {
    const device = await orig.call(this, desc);
    try {
      device.lost.then(
        (info) => self.postMessage({ gpu: 'lost', reason: info?.reason, message: info?.message }),
        () => {},
      );
      device.addEventListener('uncapturederror', (e) =>
        self.postMessage({ gpu: 'error', kind: e?.error?.constructor?.name || '' }),
      );
    } catch (_) {
      /* an implementation without `lost` or events: nothing to watch */
    }
    return device;
  };
}

/// `?gpucensus=1` (the page passes it in the first message): `gpucensus.js` counts this worker's
/// WebGPU objects, posted to the page every 5 s with the linear memory and this agent's externref
/// table (`threads.js` prints them). Loaded only with the flag, after the module is up, so a
/// session without it fetches and wraps nothing; objects made before it lands go uncounted.
function startCensus(exports) {
  import('./gpucensus.js').then(({ installGpuCensus }) => {
    const snapshot = installGpuCensus();
    setInterval(() => {
      const refs = exports?.__wbindgen_externrefs?.length ?? 0;
      const mib = Math.round((exports?.memory?.buffer?.byteLength ?? 0) / 1048576);
      // The handshake's counters for the same window (`render_worker::take_stats`).
      const handshake = __wenilla_render_stats();
      self.postMessage({ census: { ...snapshot(), refs, mib, handshake } });
    }, 5000);
  });
}

/// A Rust panic here is a wasm trap: the instance on this agent is left mid-call, so nothing more
/// may run in it. The page notices when its next extract goes unanswered and says so.
let dead = false;

function guarded(f) {
  if (dead) return false;
  try {
    return f();
  } catch (e) {
    dead = true;
    console.error('wenilla: the render worker stopped —', e);
    return false;
  }
}

/// A task of its own after each render: a message to ourselves runs once the browser has had the
/// event loop back from the task that rendered.
const settler = new MessageChannel();
settler.port1.onmessage = () => {
  if (guarded(__wenilla_render_settle)) rendered();
};

function poll() {
  if (guarded(__wenilla_render_poll)) rendered();
}

/// How many frames the GPU may still be working on when the worker takes the next extract. The
/// browser lets a canvas run ~5 frames ahead of a GPU that cannot keep up with the display
/// (measured at Wailing Caverns, 2x: 44-48 ms from a frame's last submit to its completion, one
/// frame done every 8.7 ms); held to 2 the GPU is as busy, the latency halves and the frames it
/// completes come evenly (gap p99 16 ms against 26-32). While the GPU is behind the worker stays
/// settling, so the page holds its next update instead of queueing it (`render_worker`).
const IN_FLIGHT = 2;
/// The device's queue, seen at its first submit, and the frames submitted to it not yet done.
let queue = null;
const inFlight = [];
function watchQueue() {
  const submit = self.GPUQueue?.prototype?.submit;
  if (typeof submit !== 'function') return;
  self.GPUQueue.prototype.submit = function (...args) {
    queue = this;
    return submit.apply(this, args);
  };
}

/// After a render: settle once at most `IN_FLIGHT` frames, this one included, are on the GPU. A
/// frame whose completion fails (a lost device, which the page's guard handles) counts as done.
function rendered() {
  const settle = () => settler.port2.postMessage(0);
  if (queue) {
    const done = queue.onSubmittedWorkDone();
    const drop = () => inFlight.splice(inFlight.indexOf(done), 1);
    inFlight.push(done);
    done.then(drop, drop);
  }
  const ahead = inFlight.length > IN_FLIGHT ? inFlight[inFlight.length - 1 - IN_FLIGHT] : null;
  if (ahead) ahead.then(settle, settle);
  else settle();
}

self.onmessage = (event) => {
  const { module, memory, canvas, stackSize, census } = event.data || {};
  self.onmessage = poll;
  watchDevices();
  watchQueue();
  try {
    const exports = initSync({ module, memory, thread_stack_size: stackSize ?? undefined });
    if (census) startCensus(exports);
  } catch (e) {
    dead = true;
    console.error('wenilla: the render worker failed to initialise —', e);
    return;
  }
  try {
    __wenilla_render_entry(canvas);
  } catch (e) {
    dead = true;
    console.error('wenilla: the render worker stopped —', e);
  }
};
