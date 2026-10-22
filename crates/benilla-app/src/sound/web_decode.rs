//! wasm32: the **audio decode**, taken off the frame.
//!
//! ## The stall this closes
//!
//! The browser build already reads its audio without blocking: `Chain::try_read` answers
//! `Pending` while the pack holding a file is on its way, and the zone soundscape, the kit cache
//! and the glue theme all yield the frame and ask again (`benilla_formats::TryRead`). That closed
//! the *fetch* half of a stall this client had at every zone line and every first play of a
//! sound. It left the other half untouched, and the other half is the bigger one:
//!
//! kira drops its whole streaming module on wasm32 — there are no decode threads there, so
//! `sound.rs`'s `pub mod streaming` is `#[cfg(not(target_arch = "wasm32"))]` — and
//! [`super::mixer::stream_from_bytes`]'s wasm arm therefore decodes the **whole file up front**,
//! on the thread that draws. Measured at a tavern doorway over loopback, where the fetch is
//! nearly free: a 1.1 MB zone MP3 costs ~60 ms to fetch and **~91 ms to decode**, inside a 220 ms
//! frame. Taking only the fetch off the frame leaves most of that behind.
//!
//! ## The fix: hand the decode to the browser
//!
//! `decodeAudioData` decodes off the main thread and hands back an `AudioBuffer`. Copying its
//! planar channels into kira's interleaved [`Frame`]s is a memcpy where a full MP3 decode used to
//! be. The bytes come from wherever the caller got them — this module never fetches, because on
//! this branch the fetch is the pack chain's job and has already happened by the time anyone
//! calls [`begin`].
//!
//! It runs on a [`web_sys::OfflineAudioContext`] rather than the playing `AudioContext`: this
//! context only ever decodes, it is never started, and an offline one is outside the autoplay
//! policy entirely — so nothing here can perturb the output device the mixer owns.
//!
//! **The fallback is load-bearing.** WoW ships IMA/MS-ADPCM WAVs, which `decodeAudioData` rejects
//! and kira's own decoder (its `adpcm` feature) accepts. A refusal falls back to decoding the same
//! bytes in wasm — which does cost a frame, but only for the formats that need it, and never for
//! music, which is the expensive one.
//!
//! ## What defers, and why that is safe
//!
//! A decode now completes some frames after the read that asked for it. Every caller here already
//! absorbs that by construction, because every caller already absorbs `TryRead::Pending`: this
//! adds frames to a wait that exists, it does not create a new one.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use kira::Frame;
use wasm_bindgen::JsCast;

use super::mixer::StaticSoundData;

/// One decode in flight.
///
/// `Arc<Mutex>` rather than `Rc<RefCell>` so a `Pending` is `Send`: the SFX cache that holds these
/// is a Bevy `Resource` and the narration's is a `Local`, both of which demand it. wasm has one
/// thread, so the lock is never contended — it is a type-level formality.
///
/// A `Pending` dropped while still running (the player crossed a second border before the first
/// decode landed) leaves its task writing into an `Arc` nobody reads — the last edge wins, which
/// is the same answer the synchronous path gave.
pub(super) struct Pending {
    slot: Arc<Mutex<Option<Result<StaticSoundData, String>>>>,
}

impl Pending {
    /// The decoded sound, once the browser has it. `None` = still decoding. Answers **once**: the
    /// slot is taken, so the frame this returns `Some` is the frame the caller plays.
    pub(super) fn take(&mut self) -> Option<Result<StaticSoundData, String>> {
        self.slot.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

/// Begin decoding bytes the caller already holds. Returns immediately — nothing after this point
/// runs on the frame until [`Pending::take`] answers.
///
/// `looping` marks the whole-file loop a bed needs (what `mixer::loop_from_bytes` does on the
/// synchronous path): music never loops, an ambience bed always does, a one-shot never does.
pub(super) fn begin(bytes: Vec<u8>, looping: bool) -> Pending {
    let slot: Arc<Mutex<Option<Result<StaticSoundData, String>>>> = Arc::new(Mutex::new(None));
    let write = slot.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let out = decode(bytes, looping).await;
        *write.lock().unwrap_or_else(|p| p.into_inner()) = Some(out);
    });
    Pending { slot }
}

/// In the browser where possible, in wasm where not.
async fn decode(bytes: Vec<u8>, looping: bool) -> Result<StaticSoundData, String> {
    let data = match decode_in_browser(&bytes).await {
        Ok(data) => data,
        Err(e) => {
            debug!("web_decode: decodeAudioData refused ({e}) — decoding in wasm instead");
            super::mixer::sfx_from_bytes(bytes).map_err(|e| format!("{e:#}"))?
        }
    };
    Ok(if looping { data.loop_region(..) } else { data })
}

/// `decodeAudioData` on the decode-only offline context, into kira's PCM.
///
/// The JS copy of the bytes is unavoidable and is the compressed file, not the decoded PCM — one
/// memcpy of ~1 MB against a decode of tens of milliseconds. `decodeAudioData` *detaches* the
/// buffer it is given, which is exactly why the caller's `Vec` is kept: it is what the ADPCM
/// fallback above decodes from.
async fn decode_in_browser(bytes: &[u8]) -> Result<StaticSoundData, String> {
    let ctx = decode_context()?;
    let buf = js_sys::Uint8Array::from(bytes).buffer();
    let promise = ctx.decode_audio_data(&buf).map_err(js_err)?;
    let audio: web_sys::AudioBuffer = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(js_err)?
        .dyn_into()
        .map_err(|_| "decodeAudioData did not resolve to an AudioBuffer".to_string())?;
    // The one part of this that IS on the main thread: copying the browser's planar channels into
    // kira's interleaved frames, sliced across tasks so a long track does not take one frame.
    frames_from(&audio).await
}

/// The one context this module decodes on, built on first use and kept.
///
/// Offline, and 1 frame long: its only job is to own `decodeAudioData`. The length and rate are
/// the constructor's required arguments, not a choice about the audio — `decodeAudioData` sizes
/// its own output from the file, and we read the rate back off the buffer it returns.
fn decode_context() -> Result<web_sys::OfflineAudioContext, String> {
    thread_local! {
        static CTX: std::cell::RefCell<Option<web_sys::OfflineAudioContext>> =
            const { std::cell::RefCell::new(None) };
    }
    CTX.with(|cell| {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(
                web_sys::OfflineAudioContext::new_with_number_of_channels_and_length_and_sample_rate(
                    2, 1, 44_100.0,
                )
                .map_err(js_err)?,
            );
        }
        Ok(cell.as_ref().expect("just filled").clone())
    })
}

/// Frames copied per slice of [`frames_from`]: ~6 s of audio, about a millisecond of copying.
const COPY_SLICE: usize = 1 << 18;

/// An `AudioBuffer`'s planar channels into kira's interleaved [`Frame`]s, **a slice at a time**.
///
/// A zone track is minutes long, so the whole copy is tens of MB of `f32`s. That landed in one
/// frame and froze the tab at the zone line. So the copy reads bounded ranges and yields to the
/// browser between them, and the result is filled in place (one allocation, no final copy).
///
/// Mono is duplicated to both sides rather than left half-silent (`Frame::from_mono`, the same
/// answer kira's own decoder gives a mono file); anything above two channels takes the first two,
/// which is every file in the corpus and all the mixer can place anyway.
///
/// `copyFromChannel` refuses a shared `Float32Array`, and with threads every view of wasm memory
/// is one, so the channels land in a JS scratch array first and `copy_to` moves them into memory.
/// Copying straight into the slices failed every decode there and fell back to the wasm decoder.
async fn frames_from(audio: &web_sys::AudioBuffer) -> Result<StaticSoundData, String> {
    let len = audio.length() as usize;
    let stereo = audio.number_of_channels() > 1;
    let mut frames: Arc<[std::mem::MaybeUninit<Frame>]> = Arc::new_uninit_slice(len);
    let slice = COPY_SLICE.min(len.max(1));
    let (mut left, mut right) = (vec![0.0f32; slice], vec![0.0f32; slice]);
    let scratch = js_sys::Float32Array::new_with_length(slice as u32);
    let channel = |dst: &mut [f32], ch: i32, at: usize| -> Result<(), String> {
        let view = scratch.subarray(0, dst.len() as u32);
        audio
            .copy_from_channel_with_f32_array_and_start_in_channel(&view, ch, at as u32)
            .map_err(js_err)?;
        view.copy_to(dst);
        Ok(())
    };
    let mut at = 0;
    while at < len {
        let n = slice.min(len - at);
        channel(&mut left[..n], 0, at)?;
        if stereo {
            channel(&mut right[..n], 1, at)?;
        }
        let dst = Arc::get_mut(&mut frames).expect("the only owner until it is returned");
        for i in 0..n {
            dst[at + i].write(if stereo {
                Frame {
                    left: left[i],
                    right: right[i],
                }
            } else {
                Frame::from_mono(left[i])
            });
        }
        at += n;
        if at < len {
            next_task().await;
        }
    }
    // SAFETY: the loop above wrote every index in `0..len`.
    let frames = unsafe { frames.assume_init() };
    Ok(StaticSoundData {
        sample_rate: audio.sample_rate() as u32,
        frames,
        settings: Default::default(),
        slice: None,
    })
}

/// Resolve on the browser's next task (`setTimeout(0)`), so a paint can happen in between; a
/// resolved promise would only yield to the microtask queue, which drains before the frame.
async fn next_task() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let queued =
            web_sys::window().is_some_and(|w| w.set_timeout_with_callback(&resolve).is_ok());
        if !queued {
            let _ = resolve.call0(&wasm_bindgen::JsValue::NULL);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

fn js_err(e: wasm_bindgen::JsValue) -> String {
    format!("{e:?}")
}
