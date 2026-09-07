// audioworklet.js — the browser build's IOProc, and the whole reason sound survives a long frame.
//
// ## Why this file exists
//
// `sound/output` (decision 1857) is built on one idea: a render thread runs kira's mixer AHEAD of
// the device into a lock-free ring, and the device's callback only COPIES from that ring. A stall
// then has the ring's depth to hide in instead of one cycle's budget.
//
// The browser had none of it. `mixer::PlatformBackend` ran kira's own cpal backend on wasm, and
// cpal's Web Audio backend is a `ScriptProcessorNode` — whose callback runs on the MAIN THREAD,
// with the mix inside it. So the mix competed with the frame for the same thread, and any hitch
// long enough to see was also long enough to hear: a 290 ms frame is ~27 missed device cycles.
// `sound/mod.rs` said why it was that way — "a render thread the page cannot give it" — and that
// stopped being true when the module got shared memory and Web Workers.
//
// An `AudioWorkletProcessor` runs on the browser's own audio rendering thread. It is not the main
// thread, it is not a worker, and nothing the page does can block it. That is the property this
// whole change is for: the drain is independent of the frame. The fill is made independent
// separately, by running the mixer on a Web Worker (`sound::web_output`).
//
// ## The ring
//
// Deliberately NOT rtrb's. The producer is Rust and the consumer is this file, so the layout is a
// contract across a language boundary and has to be written down rather than inherited from a
// crate's internals — a silent layout change on a `cargo update` would be a corrupted mix, which
// is the worst way to find out. It is the same discipline `web/threads.js` and `web/packs.js`
// already follow with their globals.
//
// The ring lives in WASM LINEAR MEMORY, not a separate SharedArrayBuffer. Under `+atomics` the
// module's memory IS a `SharedArrayBuffer`, so handing this processor `memory.buffer` plus a byte
// offset gives it a direct view of the very bytes the render thread writes — no second buffer, no
// copy between them, and no allocation this file has to own.
//
// Layout at `ringPtr` (all little-endian, the wasm ABI):
//
//   +0   i32  read   — frames consumed, monotonic. WRITTEN HERE, read by the producer.
//   +4   i32  write  — frames produced, monotonic. Written by the producer, read here.
//   +8   i32  capacity_frames — a power of two, so the mask is `& (capacity - 1)`.
//   +12  i32  channels
//   +16  i32  underruns — frames this processor had to invent. WRITTEN HERE.
//   +20  i32  _pad (keeps the f32 data 8-byte aligned)
//   +24  f32  data[capacity_frames * channels], interleaved
//
// `read` and `write` are monotonic counters rather than wrapped indices on purpose: the difference
// is the fill level with no ambiguity between "empty" and "full", which is the classic SPSC ring
// bug. They are `i32` because `Atomics` operates on `Int32Array`; the wrap at 2^31 frames is
// ~12 hours of continuous audio at 48 kHz, and it is handled below rather than ignored.
//
// ## What runs here must never block
//
// `Atomics.wait` is forbidden on the audio thread, and would be wrong here anyway: a processor that
// waits is a processor that misses its deadline. Only `load`, `store` and `notify` are used — none
// of which blocks — which is exactly the SPSC discipline: one writer per index, and the `notify`
// is what wakes the producer that IS allowed to wait (it runs on a worker).

/// Byte offsets into the ring header. Must match `sound::web_output::Ring`.
const H_READ = 0;
const H_WRITE = 1;
const H_CAPACITY = 2;
const H_CHANNELS = 3;
const H_UNDERRUNS = 4;
/// Where the interleaved f32 payload starts, in bytes from `ringPtr`.
const DATA_OFFSET = 24;

class WenillaMixOut extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { memory, ringPtr } = options.processorOptions;
    // Two views over the SAME bytes: the header as i32 (for `Atomics`), the payload as f32.
    this.head = new Int32Array(memory, ringPtr, DATA_OFFSET / 4);
    this.capacity = this.head[H_CAPACITY];
    this.channels = this.head[H_CHANNELS];
    this.data = new Float32Array(memory, ringPtr + DATA_OFFSET, this.capacity * this.channels);
    this.mask = this.capacity - 1;
    // Set once the first `process` runs, so the page can tell "worklet never started" (a bad
    // module URL, a construction that threw) from "worklet is starved" (the ring is empty). Those
    // are different bugs and they sound identical.
    this.port.postMessage({ type: 'started' });
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    if (!out || out.length === 0) return true;
    const frames = out[0].length; // the quantum, 128 today and not promised to stay so
    const channels = this.channels;

    // One `Atomics.load` for the whole quantum: the producer only ever grows `write`, so a value
    // read here is a floor on what is available and using it for the whole block is safe. Loading
    // per frame would be correct too and would cost 128 atomics for nothing.
    const write = Atomics.load(this.head, H_WRITE);
    let read = Atomics.load(this.head, H_READ);
    // `| 0` keeps this in int32 the way the counters are, so the subtraction stays correct across
    // the 2^31 wrap instead of going through a double and losing the wrap.
    const available = (write - read) | 0;
    const take = available < frames ? (available > 0 ? available : 0) : frames;

    for (let i = 0; i < take; i++) {
      const base = ((read + i) & this.mask) * channels;
      for (let c = 0; c < out.length; c++) {
        // A mono output from a stereo ring (or the reverse) is not a case this client produces,
        // but reading past the ring's channel count would read the NEXT frame's sample and
        // sound like a tone rather than silence — so clamp instead.
        out[c][i] = this.data[base + (c < channels ? c : channels - 1)];
      }
    }
    // The starved tail: silence, not stale samples. A repeated buffer is the loud, buzzing
    // failure; silence is the quiet one, and it is also what every underrun meter assumes.
    for (let i = take; i < frames; i++) {
      for (let c = 0; c < out.length; c++) out[c][i] = 0;
    }
    if (take < frames) {
      // Counted, never logged: this runs on the audio thread and a `console.log` here would be
      // the very stall it is reporting. `sound::web_output` reads it and reports on the main one.
      Atomics.store(this.head, H_UNDERRUNS, (Atomics.load(this.head, H_UNDERRUNS) + (frames - take)) | 0);
    }
    read = (read + take) | 0;
    Atomics.store(this.head, H_READ, read);
    // Wake the producer. It waits on this index with `memory_atomic_wait32`, so without this it
    // would only ever come back on its own timeout and the ring would refill in coarse steps.
    // `notify` does not block and is legal on the audio thread — `wait` is the forbidden one.
    Atomics.notify(this.head, H_READ);
    return true; // keep the node alive for the session; the page disconnects it to stop
  }
}

registerProcessor('wenilla-mix-out', WenillaMixOut);
