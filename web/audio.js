// audio.js — the page half of the browser build's audio output.
//
// The shape every other page seam in this client uses (`threads.js`, `packs.js`, `platform.js`): an
// ES module with one installer, imported from `index.html` and hanging a plain global off `window`,
// so Rust reaches it through `Reflect::get` and needs no wasm-bindgen import here and no `web-sys`
// feature for `AudioWorkletNode` there.
//
// **A page that ships neither this nor `audioworklet.js` still has sound.** `start` then answers
// false and `sound::web_output` falls back to kira's own cpal backend — the `ScriptProcessorNode`
// on the main thread this client shipped with until now. That is the designed degradation, and it
// is what `?audioworklet=0` selects deliberately.
//
// ## Why the AudioContext is made here and not in Rust
//
// Because the page already owns the autoplay problem. `index.html` patches the `AudioContext`
// constructor to remember every context and resume them all on the first user gesture — a browser
// suspends any context created before one, which is why the tab used to start silent. Constructing
// ours through that same patched global is how it joins the set for free.

/// The one AudioContext this page will ever make, created lazily by `rate()`. One per page: the
/// mixer opens exactly one device, a second context would be a second set of speakers nobody asked
/// for, and browsers cap how many a document may have.
let ctx = null;

/// The connected worklet node, or null until `start` succeeds.
let node = null;

/// Did the processor's constructor actually run? Distinguishes "the worklet never started" (a bad
/// module URL, a constructor that threw) from "the worklet is starved" (the ring is empty). Those
/// are opposite bugs and they sound identical.
let started = false;

/// Resolve a sibling page file relative to THIS module, not to the document. The site serves `/`
/// from a Worker and the dev host from a directory; both keep these files beside each other, and
/// neither guarantees the document's base URL is that directory.
function sibling(name) {
  return new URL(name, import.meta.url).href;
}

/// The context, made on first ask.
///
/// No `sampleRate` is requested: asking for one the device does not run at makes the browser
/// resample every sample forever, and the mixer is perfectly happy to render at whatever the device
/// is — it is told the rate below. `latencyHint: 'playback'` asks for the largest device buffer the
/// browser will give, which is the right end of that dial here: nothing in this client is
/// rhythm-critical (the shortest UI click is an order of magnitude longer than any buffer on
/// offer), and a longer buffer is more slack for the worklet.
function context() {
  if (ctx) return ctx;
  try {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) return null;
    ctx = new Ctor({ latencyHint: 'playback' });
  } catch (e) {
    console.warn('wenilla: no AudioContext —', (e && e.message) || e);
    ctx = null;
  }
  return ctx;
}

/// Install the page bridge. Called from `index.html` **before** `boot(init)`, for the same reason
/// `installThreads` is: the mixer is built inside the plugin graph `init()` constructs, so a bridge
/// installed afterwards would find the device already opened the other way.
export function installAudio() {
  window.__wenilla_audio = {
    /// The device's sample rate, creating the context if this is the first ask.
    ///
    /// **Synchronous, and that is why it is separate from `start`.** kira's `Backend::setup` has to
    /// answer the rate before `start` is ever called, and `addModule` is a promise — so the context
    /// (which IS constructed synchronously) is made here and the worklet is added to it later. `0`
    /// means no Web Audio at all, and the caller falls back.
    rate() {
      const c = context();
      return c ? c.sampleRate : 0;
    },

    /// Connect the worklet to the ring. `memory` is the module's `WebAssembly.Memory` — under
    /// `+atomics` its `.buffer` is a `SharedArrayBuffer`, so the processor gets a direct view of the
    /// very bytes the render worker writes — and `ringPtr` is a byte offset into it.
    ///
    /// Every failure answers `false` rather than throwing: the caller's fallback is a whole working
    /// audio path, so a refusal here is a routing decision and not an error.
    async start(memory, ringPtr) {
      if (node) return true;
      try {
        const c = context();
        if (!c) return false;
        await c.audioWorklet.addModule(sibling('audioworklet.js'));
        node = new AudioWorkletNode(c, 'wenilla-mix-out', {
          numberOfInputs: 0,
          numberOfOutputs: 1,
          outputChannelCount: [2],
          processorOptions: { memory: memory.buffer, ringPtr },
        });
        node.port.onmessage = (e) => {
          if (e.data && e.data.type === 'started') started = true;
        };
        // Nothing between the node and the speakers: the mix is already limited and volume-
        // controlled inside kira (`sound::limiter`, `sound::mixer`), and a gain node here would be
        // a second place for gain to live and a second place to look when it is wrong.
        node.connect(c.destination);
        // A context made before a user gesture starts suspended. `index.html`'s hook resumes every
        // remembered context on the first gesture; this covers the commoner case here, where a
        // gesture has ALREADY happened by the time sound starts — the client does not open its
        // device until the player has clicked through the glue screen.
        if (c.state === 'suspended') c.resume().then(() => {}, () => {});
        return true;
      } catch (e) {
        console.warn('wenilla: AudioWorklet output unavailable —', (e && e.message) || e);
        node = null;
        return false;
      }
    },

    /// Has the processor run at least once? See `started`.
    running() {
      return started;
    },

    /// `AudioContext.state` — `'running'`, `'suspended'` or `'closed'`, `''` before any ask. The one
    /// thing that tells a tab waiting on a gesture apart from a broken output.
    state() {
      return ctx ? ctx.state : '';
    },

    /// Best-effort resume, for the page's gesture hook and the client's own focus handling.
    resume() {
      if (ctx && ctx.state === 'suspended') ctx.resume().then(() => {}, () => {});
    },
  };
}
