//! The cinematic narration: the race intro's voice-over, held by a handle so a skip cuts it off.
//!
//! Each race fly-by's `CinematicCamera` row names one streamed `<Race>Narration.mp3` kit (type
//! 31); the two non-intro rows name id 0. A dropped kira handle keeps playing, so the handle is
//! held here. The reference keeps the narration in `[0xb4e274]` (written at `0x48ef3e`) and
//! releases it on shot advance (`0x48efef`), stop/ESC (`0x48f055`) and local abort (`0x490b8d`);
//! this module follows [`Cinematic`]'s published shot the same way.

use bevy::prelude::*;

use benilla_assets::WorldAssets;
use benilla_world::schedule::WorldStage;

use super::mixer::{self, StreamingSoundHandle};
use super::{kit::SoundKits, SoundOutput};
use crate::cinematic::Cinematic;

/// A stopped narration is cut with no fade: every release of `[0xb4e274]` is plain. The 0.25 s
/// fade the reference schedules at both edges (`0x4c0d10`, `[0x804550]`) is a screen fade.
const CUT_FADE_MS: u64 = 0;

/// The narration channel: which shot it belongs to, and its live handle.
#[derive(Default)]
pub(super) struct CinematicVoice {
    /// `(run, shot index)` of the playing narration. Keyed on the run, not the sequence id:
    /// re-triggering the sequence on screen restarts the picture and must restart the voice.
    shot: Option<(u64, usize)>,
    handle: Option<StreamingSoundHandle<kira::sound::FromFileError>>,
    /// **wasm32: the line's load in flight** — the shot it belongs to, its file and kit volume,
    /// and the browser's decode.
    ///
    /// The race intro was the last per-shot blocking read in the sound engine: a whole narration
    /// MP3 fetched synchronously and decoded in wasm, both on the frame, at the exact moment the
    /// picture is supposed to start moving. It now takes the same two-step wait everything else
    /// on this target takes — `Chain::try_read` for the bytes, [`super::web_decode`] for the PCM.
    ///
    /// The shot identity rides along because a load that lands after its shot has gone must be
    /// **dropped, not played over the next one**: the narration follows the picture, and a voice
    /// belonging to a shot that is over is not late, it is wrong.
    #[cfg(target_arch = "wasm32")]
    pending: Option<PendingLine>,
}

/// wasm32: the narration line a shot asked for, and how far along it is.
///
/// The **pick is latched with it**, for the reason [`super::zone`]'s two slots latch theirs:
/// `pick_stream` depletes, so re-rolling the kit on every frame of the wait would scatter the
/// warm across variations and never finish the one that was chosen.
#[cfg(target_arch = "wasm32")]
struct PendingLine {
    /// The shot that asked. A load that lands after its shot has gone is **dropped, not played
    /// over the next one**: the narration follows the picture, and a voice belonging to a shot
    /// that is over is not late, it is wrong.
    shot: (u64, usize),
    path: String,
    kit_vol: f32,
    phase: LinePhase,
}

/// wasm32: the two waits a narration goes through — the pack, then the decode.
#[cfg(target_arch = "wasm32")]
enum LinePhase {
    /// Waiting on the pack. `Chain::try_read` started the warm; ask again next frame.
    Reading,
    /// The bytes are in hand and the browser is decoding them ([`super::web_decode`]).
    Decoding(super::web_decode::Pending),
}

impl CinematicVoice {
    /// Whether the device is still mixing this voice; a narration shorter than its shot ends
    /// `Stopped` on its own.
    fn is_live(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|h| !matches!(h.state(), kira::sound::PlaybackState::Stopped))
    }

    fn stop(&mut self) {
        self.shot = None;
        if let Some(mut h) = self.handle.take() {
            h.stop(mixer::fade(CUT_FADE_MS));
        }
        // A load still running for the shot being cut is abandoned with it — see [`Self::pending`].
        #[cfg(target_arch = "wasm32")]
        {
            self.pending = None;
        }
    }
}

pub(super) fn plugin(app: &mut App) {
    // In `Present`, after the cinematic driver in `Input` has settled this frame's shot. Gated on
    // `SoundKits`, which exists only once the install's kits load: without it the bare `ResMut`
    // panics.
    app.add_systems(
        Update,
        drive_narration
            .in_set(WorldStage::Present)
            .run_if(resource_exists::<SoundKits>),
    );
}

/// The narration channel, plus its line in the voice budget: each stream owner rewrites its own
/// count in `SoundOutput` every frame from its live handle, after the body has settled it.
fn drive_narration(
    cine: Option<Res<Cinematic>>,
    voice: Local<CinematicVoice>,
    out: NonSendMut<SoundOutput>,
    kits: ResMut<SoundKits>,
    assets: Option<Res<WorldAssets>>,
) {
    let (mut voice, mut out, mut kits) = (voice, out, kits);
    narrate(&cine, &mut voice, &mut out, &mut kits, &assets);
    out.cinematic_streams = usize::from(voice.is_live());
}

fn narrate(
    cine: &Option<Res<Cinematic>>,
    voice: &mut CinematicVoice,
    out: &mut SoundOutput,
    kits: &mut SoundKits,
    assets: &Option<Res<WorldAssets>>,
) {
    let playing = cine.as_deref().and_then(Cinematic::playing_shot);
    let Some((run, index, sound_id)) = playing else {
        // No cinematic, or it just ended or was skipped: cut whatever was narrating.
        voice.stop();
        return;
    };
    if voice.shot == Some((run, index)) {
        // The shot has not changed — but on the page its line may still be arriving. This is the
        // only frame shape that says "the shot that asked is still the shot on screen", which is
        // exactly the condition a deferred narration is allowed to start under.
        #[cfg(target_arch = "wasm32")]
        if let Some(assets) = assets.as_deref() {
            drive_pending_line(voice, out, assets, (run, index));
        }
        return;
    }
    voice.stop();
    voice.shot = Some((run, index));
    if sound_id == 0 {
        return;
    }
    let Some(assets) = assets else { return };
    // Checked, not borrowed: on the page the play is frames away and the borrow would collide
    // with the one the wait's driver needs. There is no point reading a line for no device.
    if out.mixer.is_none() {
        return;
    }
    let Some((path, kit_vol)) = kits.pick_stream(sound_id) else {
        warn!("cinematic voice: sound {sound_id} has no file");
        return;
    };
    #[cfg(target_arch = "wasm32")]
    {
        // Latch the pick and take the wait's first step now — a pack already resident and a file
        // the browser decodes quickly can still start on this frame or the next.
        voice.pending = Some(PendingLine {
            shot: (run, index),
            path,
            kit_vol,
            phase: LinePhase::Reading,
        });
        drive_pending_line(voice, out, assets, (run, index));
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let data = {
        // Every file FMOD opens is a loose-flagged open (`0x7a8860`).
        let bytes = match assets.try_read_loose_or_chain(&path) {
            benilla_formats::TryRead::Ready(bytes) => Ok(bytes),
            _ => Err(anyhow::anyhow!("not in the loose files or the patch chain")),
        };
        match bytes.and_then(mixer::stream_from_bytes) {
            Ok(d) => d,
            Err(e) => {
                warn!("cinematic voice: {path} — {e:#}");
                return;
            }
        }
    };
    // No category slider applies: the narration opens (`0x48ef29` → `0x458a40` → `0x45ce60`)
    // with flags `0x15`, and bit `0x10` skips the category multiply at `0x7a5dc0`, so its gain is
    // the kit volume flat. Only bit `0x2` bypasses the `MasterSoundEffects` gate (`0x7a529c`),
    // so "Enable Sound Effects" still silences it; that gate rides the master track, which
    // re-reads `SoundConfig::enabled` every frame, never this starting amp.
    #[cfg(not(target_arch = "wasm32"))]
    play_line(voice, out, &path, kit_vol, data);
}

/// Open the narration channel on a line that is **already decoded** — the tail of [`narrate`],
/// split out because on the web target the read and the decode happen between the halves.
fn play_line(
    voice: &mut CinematicVoice,
    out: &mut SoundOutput,
    path: &str,
    kit_vol: f32,
    data: mixer::StreamingSoundData<kira::sound::FromFileError>,
) {
    let Some(mixer_ref) = out.mixer.as_mut() else {
        return;
    };
    let amp = kit_vol;
    match mixer_ref.play_stream(data.volume(mixer::amp_to_db(amp))) {
        Ok(h) => {
            info!("cinematic voice: {path}");
            voice.handle = Some(h);
        }
        Err(e) => warn!("cinematic voice: {path} — {e:#}"),
    }
}

/// wasm32: advance the line the current shot is waiting on by one step, and play it if it lands.
///
/// Called on every frame the shot is unchanged, which is the only condition under which a
/// deferred line may still start — the identity check is [`narrate`]'s, and [`CinematicVoice::stop`]
/// drops the latch on every shot change, so a line whose shot has gone is never reached here.
#[cfg(target_arch = "wasm32")]
fn drive_pending_line(
    voice: &mut CinematicVoice,
    out: &mut SoundOutput,
    assets: &WorldAssets,
    shot: (u64, usize),
) {
    use benilla_formats::TryRead;

    let Some(line) = voice.pending.as_mut() else {
        return;
    };
    // Belt and braces on [`PendingLine::shot`]: `CinematicVoice::stop` already drops the latch on
    // every shot change, so this cannot fire — and if it ever does, a line is being driven for a
    // shot that is no longer on screen, which must be dropped rather than played.
    if line.shot != shot {
        voice.pending = None;
        return;
    }
    match &mut line.phase {
        // Every file FMOD opens is a loose-flagged open (`0x7a8860`).
        LinePhase::Reading => match assets.try_read_loose_or_chain(&line.path) {
            TryRead::Ready(bytes) => {
                line.phase = LinePhase::Decoding(super::web_decode::begin(bytes, false));
            }
            // The ask started the pack warming. Come back next frame.
            TryRead::Pending => {}
            TryRead::Missing => {
                warn!(
                    "cinematic voice: {} — not in the loose files or the patch chain",
                    line.path
                );
                voice.pending = None;
            }
        },
        LinePhase::Decoding(pending) => match pending.take() {
            None => {}
            Some(Ok(data)) => {
                let line = voice.pending.take().expect("just matched");
                play_line(
                    voice,
                    out,
                    &line.path,
                    line.kit_vol,
                    mixer::stream_from_decoded(data),
                );
            }
            Some(Err(e)) => {
                warn!("cinematic voice: {} — {e}", line.path);
                voice.pending = None;
            }
        },
    }
}
