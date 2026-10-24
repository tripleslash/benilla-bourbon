//! The browser build's audio output — an `AudioWorklet` drain and a Web Worker fill, so neither
//! half of the mix is on the thread that draws.
//!
//! ## What this replaces, and why it had to
//!
//! `sound::output` (decision 1857) is built on one idea, and it is the right one: a render thread
//! runs kira's [`Renderer`] **ahead** of the device into a lock-free ring, and the device's callback
//! only *copies* from that ring. A stall then has the ring's depth to hide in instead of one cycle's
//! budget.
//!
//! The browser had none of it. `mixer::PlatformBackend` ran kira's own cpal backend on wasm, and
//! cpal's Web Audio backend is a `ScriptProcessorNode` — **whose callback runs on the main thread,
//! with the mix inside it**. So the mix competed with the frame for one thread, and any hitch long
//! enough to see was also long enough to hear: the 292 ms frame in the 2026-09-07 stutter report is
//! ~27 missed device cycles. `sound/mod.rs` said exactly why it was that way — *"this layer's shape
//! is a render thread the page cannot give it"* — and that stopped being true when the module got
//! shared memory and Web Workers.
//!
//! Both halves are moved, and both are needed:
//!
//! - **The drain** is `web/audioworklet.js`, an `AudioWorkletProcessor`. It runs on the browser's
//!   own audio rendering thread — not the main thread, not a worker — and nothing the page does can
//!   block it.
//! - **The fill** is [`render_loop`] on a Web Worker. If the mixer still ran on the main thread, a
//!   290 ms frame would drain the ring dry however independent the worklet was.
//!
//! With both off the main thread, the ring no longer has to cover *frame* stalls at all — only the
//! worker's own scheduling jitter. That is why [`MIX_AHEAD_MS`] is 100 ms rather than the 300+
//! it would need to ride out a hitch: latency is bounded by what the producer runs ahead by, and
//! buying slack we no longer need would cost every UI click that much delay.
//!
//! ## Why the ring is ours and not `rtrb`'s
//!
//! The producer is Rust and the consumer is JavaScript, so the layout is a contract across a
//! language boundary. It has to be written down and owned rather than inherited from a crate's
//! internals, where a silent change on a `cargo update` would be a corrupted mix — the worst way to
//! find that out. `web/audioworklet.js` carries the same table; the two must be edited together.
//!
//! The ring lives in **wasm linear memory**, not a separate `SharedArrayBuffer`. Under `+atomics`
//! the module's memory *is* a `SharedArrayBuffer`, so handing the processor `memory.buffer` plus a
//! byte offset gives it a direct view of the very bytes this file writes: no second buffer, no copy
//! between them, and nothing to allocate.
//!
//! ## Degradation
//!
//! A page shipping neither `audio.js` nor `audioworklet.js`, a browser with no `AudioContext`, or
//! `?audioworklet=0` all make [`WebOutput::setup`] fail, and `mixer` falls back to kira's cpal
//! backend — the main-thread `ScriptProcessorNode` this client shipped with until now. Sound is then
//! exactly as good, and exactly as stutter-prone, as it was before this file existed.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use anyhow::{bail, Result};
use js_sys::{Function, Reflect};
use kira::backend::cpal::{CpalBackend, CpalBackendSettings};
use kira::backend::{Backend, Renderer};
use wasm_bindgen::{JsCast, JsValue};

/// The page global this module reaches, in the shape every other page seam uses
/// (`__wenilla_threads`, `__wenilla_packs`, `__wenilla_env`).
const GLOBAL: &str = "__wenilla_audio";

/// Interleaved stereo, the only shape the mixer produces and the only one the worklet reads.
const CHANNELS: usize = 2;

/// Ring capacity in frames. **A power of two**, because the worklet indexes with `& (capacity - 1)`
/// and a modulo in an audio callback is a modulo per frame.
///
/// 8192 frames is ~170 ms at 48 kHz — comfortably more than [`MIX_AHEAD_MS`], which is what
/// actually bounds latency. The headroom is there so the producer never has to stop mid-chunk.
const CAPACITY_FRAMES: usize = 8192;

/// How far ahead of the worklet the mixer runs — the reference's `SoundBufferSize`, and the same
/// 100 ms `sound::output::MIX_AHEAD_MS` uses natively.
///
/// This is the latency dial, not the safety dial: the producer stops here rather than filling the
/// ring, so a click is heard 100 ms after the mixer renders it and not 170. It can be this small
/// only because the worker doing the filling is not the thread that stalls.
const MIX_AHEAD_MS: usize = 100;

/// Frames rendered per `Renderer::process` call. Matches `sound::output::RENDER_CHUNK_FRAMES`.
const CHUNK_FRAMES: usize = 256;

/// How long the producer waits for the worklet to make room before looking again. A backstop, not
/// the mechanism: the worklet calls `Atomics.notify` when it drains, so the normal wake is prompt
/// and this only matters if the worklet never starts — where it turns a spin into a poll.
const WAIT_TIMEOUT_NS: i64 = 20_000_000; // 20 ms

/// The shared ring: header then interleaved payload, laid out to match `web/audioworklet.js`.
///
/// `#[repr(C)]` is load-bearing — the JS side indexes by byte offset, so field order and padding
/// are the contract. The header is six `i32`s (24 bytes), which is where `DATA_OFFSET` comes from.
#[repr(C)]
struct Ring {
    /// Frames consumed, monotonic. Written by the **worklet**, read here.
    read: AtomicI32,
    /// Frames produced, monotonic. Written **here**, read by the worklet.
    write: AtomicI32,
    capacity: i32,
    channels: i32,
    /// Frames the worklet had to invent because this ring was empty. Written by the worklet.
    underruns: AtomicI32,
    _pad: i32,
    /// Interleaved samples. `UnsafeCell` because a `&'static Ring` is shared while this is written:
    /// the SPSC discipline is what makes it sound, not the borrow checker, and `UnsafeCell` is how
    /// that is said in a type.
    data: UnsafeCell<[f32; CAPACITY_FRAMES * CHANNELS]>,
}

// SAFETY: every field is either an atomic or is touched under the single-producer/single-consumer
// discipline this module and `web/audioworklet.js` jointly implement — exactly one writer for each
// index, and a frame's samples are written before the `write` release that publishes them.
unsafe impl Sync for Ring {}

/// The one ring, in `.bss`. A `static` rather than a heap allocation on purpose: it needs a stable
/// address for the worklet's whole life, it must never move or be freed, and 64 KiB of zeroes costs
/// nothing in the module (`.bss` is not stored, only sized).
static RING: Ring = Ring {
    read: AtomicI32::new(0),
    write: AtomicI32::new(0),
    capacity: CAPACITY_FRAMES as i32,
    channels: CHANNELS as i32,
    underruns: AtomicI32::new(0),
    _pad: 0,
    data: UnsafeCell::new([0.0; CAPACITY_FRAMES * CHANNELS]),
};

/// Set when the app is going down, so the render worker's loop can end and let its Worker close.
static STOP: AtomicBool = AtomicBool::new(false);

/// Stop the render worker. Idempotent; safe to call when none was ever started.
pub(super) fn stop() {
    STOP.store(true, Ordering::Release);
}

/// Resolve one function off the page global, or `None` if the page ships no bridge.
fn bridge(method: &str) -> Option<(JsValue, Function)> {
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

/// The device's sample rate, or 0 if this page has no Web Audio at all.
fn device_rate() -> u32 {
    bridge("rate")
        .and_then(|(obj, f)| f.call0(&obj).ok())
        .and_then(|v| v.as_f64())
        .filter(|r| r.is_finite() && *r > 0.0)
        .map_or(0, |r| r as u32)
}

/// Frames the worklet has had to invent since boot — the one audible failure this design leaves,
/// counted rather than guessed (the same meter `sound::output` calls `underruns`).
pub(crate) fn underruns() -> i32 {
    RING.underruns.load(Ordering::Relaxed)
}

/// `AudioContext.state`, for the report: `"suspended"` is a tab waiting on a user gesture, which is
/// silence with a completely different cause from an underrun.
pub(crate) fn context_state() -> String {
    bridge("state")
        .and_then(|(obj, f)| f.call0(&obj).ok())
        .and_then(|v| v.as_string())
        .unwrap_or_default()
}

/// Is the worklet path wanted? `?audioworklet=0` says no and takes the old main-thread path.
///
/// The lever exists for the same reason `?offload=0` does: this is a new mechanism in a subsystem
/// where a mistake is instantly audible, and a page that disagrees with it needs a way back that is
/// not a re-upload.
fn worklet_wanted() -> bool {
    crate::webenv::var("WOW_AUDIOWORKLET").as_deref() != Some("0")
}

/// Which output this session got. The fallback lives **inside** the backend rather than beside it
/// because `Mixer` holds one `AudioManager<PlatformBackend>` and kira's backend is a type
/// parameter — two paths at that level would mean two manager types and an enum through every
/// method that touches one. One type, one runtime choice, made once at `setup`.
enum Inner {
    /// The worklet drains a ring a Web Worker fills. Neither is the thread that draws.
    Worklet,
    /// kira's own cpal backend — Web Audio through a main-thread `ScriptProcessorNode`. What this
    /// client shipped with, kept as the degradation for a page that cannot do better.
    Cpal(CpalBackend),
}

/// kira's backend: the ring, the worklet that drains it and the worker that fills it — or cpal.
pub(super) struct WebOutput {
    inner: Inner,
    sample_rate: u32,
}

/// Which arm [`WebOutput::setup`] chose. A `static` rather than a method on the backend because
/// the backend lives inside kira's `AudioManager`, and the one reader — `webdiag` — would otherwise
/// need a path through `Mixer` and `backend_mut` to ask a question with no state behind it.
///
/// Worth reporting at all because "sound is fine" and "sound is fine *because it fell back*" are
/// different states, and only one of them is this change working.
static ON_WORKLET: AtomicBool = AtomicBool::new(false);

/// Did this session get the worklet, or did it fall back to the main-thread cpal path?
pub(crate) fn on_worklet() -> bool {
    ON_WORKLET.load(Ordering::Relaxed)
}

impl Backend for WebOutput {
    /// cpal's, because the fallback arm needs them and the worklet arm needs none of its own — the
    /// device is the browser's and its rate is not ours to pick.
    type Settings = CpalBackendSettings;
    type Error = anyhow::Error;

    fn setup(settings: CpalBackendSettings, internal_buffer_size: usize) -> Result<(Self, u32)> {
        // Everything that can refuse, refuses HERE — before kira hands over the `Renderer`. Once
        // `start` has it there is no way back to cpal, so the choice has to be made while the
        // fallback is still reachable. `addModule` failing later is the one hole this leaves, and
        // `?audioworklet=0` is what covers it.
        if worklet_wanted() {
            let sample_rate = device_rate();
            if sample_rate > 0 {
                ON_WORKLET.store(true, Ordering::Relaxed);
                return Ok((
                    Self {
                        inner: Inner::Worklet,
                        sample_rate,
                    },
                    sample_rate,
                ));
            }
        }
        let (backend, sample_rate) = CpalBackend::setup(settings, internal_buffer_size)
            .map_err(|e| anyhow::anyhow!("cpal backend: {e}"))?;
        Ok((
            Self {
                inner: Inner::Cpal(backend),
                sample_rate,
            },
            sample_rate,
        ))
    }

    fn start(&mut self, renderer: Renderer) -> Result<()> {
        let rate = self.sample_rate;
        match &mut self.inner {
            Inner::Cpal(backend) => backend
                .start(renderer)
                .map_err(|e| anyhow::anyhow!("cpal backend start: {e}")),
            Inner::Worklet => {
                let (obj, start) =
                    bridge("start").ok_or_else(|| anyhow::anyhow!("no audio bridge"))?;
                // Fire-and-forget: `addModule` is a promise and kira's `start` is not async. The
                // ring simply fills while the worklet is fetched — the producer stops at the
                // mix-ahead mark and waits, which is the same state it sits in whenever the
                // worklet is keeping up.
                let ptr = &RING as *const Ring as u32;
                start
                    .call2(
                        &obj,
                        &wasm_bindgen::memory(),
                        &JsValue::from_f64(f64::from(ptr)),
                    )
                    .map_err(|e| anyhow::anyhow!("audio bridge start failed: {e:?}"))?;

                // **Not a pool thread.** This loop owns its thread for the session; taking one of
                // the async-compute workers would park a pool worker forever and leave every
                // `spawn_pooled` caller one short for the life of the tab.
                if !crate::web_threads::spawn_detached("audio-render", move || {
                    render_loop(renderer, rate)
                }) {
                    bail!("no Web Worker for the audio render thread");
                }
                Ok(())
            }
        }
    }
}

impl Drop for WebOutput {
    fn drop(&mut self) {
        // Ends the render worker's loop so its Worker can close. Inert on the cpal arm, where no
        // worker was ever started.
        stop();
    }
}

/// Fill the ring, ahead of the worklet, until told to stop. Runs on a Web Worker.
fn render_loop(mut renderer: Renderer, sample_rate: u32) {
    let ahead_frames =
        (sample_rate as usize * MIX_AHEAD_MS / 1000).min(CAPACITY_FRAMES - CHUNK_FRAMES);
    let mut chunk = vec![0f32; CHUNK_FRAMES * CHANNELS];
    while !STOP.load(Ordering::Acquire) {
        loop {
            // `Acquire` on the consumer's index pairs with its store: everything the worklet read
            // before publishing this value is done with by the time we reuse those slots.
            let read = RING.read.load(Ordering::Acquire);
            let write = RING.write.load(Ordering::Relaxed);
            // `wrapping_sub` on monotonic counters, so the 2^31-frame wrap (~12 hours at 48 kHz)
            // is arithmetic rather than a bug. The difference is the fill level, which is why the
            // counters are monotonic instead of wrapped indices — a wrapped pair cannot tell
            // "empty" from "full".
            let filled = write.wrapping_sub(read) as usize;
            if filled >= ahead_frames {
                break;
            }
            renderer.on_start_processing();
            renderer.process(&mut chunk, CHANNELS as u16);
            write_chunk(write, &chunk);
            // `Release`: the samples above are visible to the worklet before the index that
            // advertises them. This is the whole publication, and it is the only store to `write`.
            RING.write
                .store(write.wrapping_add(CHUNK_FRAMES as i32), Ordering::Release);
        }
        // Wait for the worklet to drain. It calls `Atomics.notify` after each quantum, so this
        // normally wakes promptly; the timeout is what keeps a worklet that never started from
        // turning this into a spin. `memory_atomic_wait32` is legal here — this is a worker, and
        // only the page's main thread is forbidden to execute it.
        let read_now = RING.read.load(Ordering::Relaxed);
        // SAFETY: the address is a live `AtomicI32` in this module's `static`, and the value is the
        // one just read — a mismatch simply returns "not equal" and we loop.
        unsafe {
            core::arch::wasm32::memory_atomic_wait32(RING.read.as_ptr(), read_now, WAIT_TIMEOUT_NS);
        }
    }
}

/// Copy one rendered chunk into the ring at `write`, wrapping once at most.
///
/// Split rather than looped per frame: a chunk either fits before the wrap or straddles it exactly
/// once, so this is two `copy_from_slice`s and never a bounds check per sample.
fn write_chunk(write: i32, chunk: &[f32]) {
    let mask = CAPACITY_FRAMES - 1;
    let start_frame = (write as usize) & mask;
    let first_frames = (CAPACITY_FRAMES - start_frame).min(CHUNK_FRAMES);
    // SAFETY: this is the only writer of `data` (the worklet only reads), and both slices below are
    // inside the array — `start_frame` is masked into it and `first_frames` is clamped to the
    // distance from there to the end.
    let data = unsafe { &mut *RING.data.get() };
    let head = first_frames * CHANNELS;
    data[start_frame * CHANNELS..start_frame * CHANNELS + head].copy_from_slice(&chunk[..head]);
    if head < chunk.len() {
        data[..chunk.len() - head].copy_from_slice(&chunk[head..]);
    }
}
