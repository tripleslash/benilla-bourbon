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

import {
  initSync,
  __wenilla_render_entry,
  __wenilla_render_poll,
  __wenilla_render_settle,
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
settler.port1.onmessage = () => guarded(__wenilla_render_settle);

function poll() {
  if (guarded(__wenilla_render_poll)) settler.port2.postMessage(0);
}

self.onmessage = (event) => {
  const { module, memory, canvas, stackSize } = event.data || {};
  self.onmessage = poll;
  watchDevices();
  try {
    initSync({ module, memory, thread_stack_size: stackSize ?? undefined });
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
