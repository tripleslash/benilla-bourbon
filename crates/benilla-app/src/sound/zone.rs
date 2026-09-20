//! Zone music + ambience — the area-driven schedulers over the kit player (decision 0070 slice 1).
//!
//! The data chain: the player's [`CurrentArea`] (MCNK areaId) → `AreaTable` →
//! `ZoneMusic`/`SoundAmbience`/`ZoneIntroMusicTable` (parent-inherited, `benilla_formats::
//! AreaSoundCatalog::resolve`) → SoundEntries kits → streamed MP3/WAV.
//!
//! **Music transport** (wow-re `0x460040`–`0x460ca0`, `benilla-pins.md` **B15/B16**). Within one
//! zone, tracks are separated by a uniform random **silence interval** per day/night phase. On a
//! zone-music-id CHANGE the outgoing track fades to silence over **4.0 s** then stops (byte-exact:
//! `0x4602e0` → `0x7a5a10(0x40800000 = 4.0f)`) and the incoming track starts **immediately** — the
//! change forces the next-start time to −1 (`0x4602e0`: `DAT_00836400 = 0xffffffff`), so `0x460040`
//! opens the stream that tick (`0x460240`). The randomized silence interval ([`next_track_time`],
//! `0x4601f0`) is ONLY the *same-zone* track-to-track spacing — reusing it on a change (as decision
//! 0100 did) left a newly-entered zone silent for 3–5 min. **There is no cold start**: world entry
//! itself arms the same −1 (`0x401570` → `0x457770` → the zone-music init `0x45ffc0`, which sets
//! `DAT_00836400 = -1` at `0x45ffdb`), so the session's FIRST track is immediate too. `0x4601f0`'s
//! `== 0 → now + 6000 ms` branch is reachable only from the natural-end reap `0x4600b6` and only
//! ever *replaces* the −1, so it cannot fire on an entry — the 6 s wait benilla used to serve there
//! was that branch misapplied (wow-re §5 `glue-music-world-entry.md` §9, decision 1553). The incoming track starts at **full** volume, **no fade-in** — faithful (`0x460240` →
//! `0x7a5dc0`; §5 B16 refuted every music-slot fade-in primitive, and the director confirmed by ear
//! there is none). The reference's *perceived* slightly-soft onset is the delegated audio engine
//! priming the stream to full amplitude over ~a moment (FMOD there, kira's own stream-buffering
//! here), which leaves the 4.0 s fade audible — NOT owned gain math, flagged in wow-re for a live
//! capture if benilla's onset reads too abrupt. (The transition is **not** `0x457960` — that is the
//! SFX-bus auto-duck, decision 0100.) Intro music (`ZoneIntroMusicTable`) plays at full on zone
//! entry, throttled by its per-entry minute delay;
//! higher `Priority` wins when nested areas compete (we resolve one row).
//!
//! **The music slot holds exactly one stream**, and every start goes through
//! [`start_music_stream`], which fades the outgoing one out as it opens the incoming. Dropping a
//! kira handle does *not* stop its sound (that is precisely what the fade-stop rides on), so a
//! start that merely overwrote the handle would leave the old track playing forever, unreapable
//! and deaf to the volume slider.
//!
//! **Server-pushed music** (`SMSG_PLAY_MUSIC`) is a SoundEntries kit for that same slot, and the
//! server **re-pushes the same id on a timer** to keep an event track looping: the Darkmoon Faire's
//! emitter sends it every **5 s** (vmangos `go_darkmoon_faire_music`, commented as a sniffed retail
//! value). So a push for the kit already playing is a **no-op** ([`slot_holds`]) — the client's own
//! already-playing early-out (`0x4602e0` @`0x460342`: `cmp [0xb06cbc],eax; je` → desired == playing
//! → return); when the track *does* end, the next push restarts it, which is what makes the loop.
//! Any other push takes the slot the way a zone-music change does (outgoing 4.0 s fade, incoming at
//! full) — inferred from the slot's one transition idiom, as the `SMSG_PLAY_MUSIC` handler itself
//! is not pinned in wow-re.
//!
//! **Ambience transport** (`0x460b00`, the separate ambience updater; §5 B16). An area / interior /
//! day↔night / ghost ambience-id change is a **5.0 s crossfade**: the old bed fades out over 5.0 s
//! then stops (`0x7a5a10(0x40a00000 = 5.0f)`) while the new bed starts at volume 0 and fades in over
//! 5.0 s (`0x7a5dc0(0)` → `0x7a57b0(5.0f, kit vol)`). The **submerge/emerge** swap is the exception:
//! it takes `0x460b00`'s **instant** no-fade branch (`0x458650` → `0x460af0(param = 1)`) — correcting
//! both the old "250 ms area swap" INTERIM and B6's "5 s underwater fade".
//!
//! The day/night boundary is the server-synced game clock's hard step — **day iff
//! 05:30 ≤ t < 21:00** (minutes 330..1259), re-evaluated per tick, no fade (`0x4578c0` via the
//! `0x642710` range check; wow-re `benilla-pins.md` B4, VERIFIED).

use bevy::prelude::*;

use benilla_formats::{AreaSoundCatalog, TryRead};

use crate::net::{ServerSoundKind, ServerSoundMessage};
use benilla_assets::{AssetSet, LockRecover, WorldAssets};
use benilla_world::lighting::GameClock;
use benilla_world::schedule::WorldStage;

use super::kit::{play_kit_ext, KitRef, Latch, PlayExtras, SoundCategory, SoundKits};
use super::mixer::{self, StreamingSoundHandle};
use super::{AudioListener, SoundConfig, SoundOutput};

/// The zone→audio catalog (AreaTable + ZoneMusic + SoundAmbience + ZoneIntroMusicTable).
/// Absent when the client data didn't load.
#[derive(Resource)]
pub(crate) struct AreaSounds(pub(crate) AreaSoundCatalog);

/// The race → exploration-jingle catalog (`ChrRaces.dbc` column 3) — the discovery sound
/// `net::apply`'s `SMSG_EXPLORATION_EXPERIENCE` arm plays (decision 0829). Absent when the
/// client data didn't load.
#[derive(Resource)]
pub(crate) struct ExplorationSounds(pub(crate) benilla_formats::ExplorationSoundCatalog);

/// Day iff 05:30 ≤ clock < 21:00 (module docs — the client's hard step, B4-verified).
/// Index into the `[day, night]` DBC pairs.
/// How long after a cinematic ends before the zone track comes back — the reference's
/// `[0x836400] = tick + 0xbb8` at `0x4603b0(0)`, i.e. **3.000 s**. Its own number: not the `-1`
/// immediate sentinel a zone change arms, and not the randomized `SilenceInterval`.
const CINEMATIC_MUSIC_RESUME_SECS: f64 = 3.0;

fn phase(clock: &GameClock) -> usize {
    if (330..1260).contains(&clock.minute) {
        0
    } else {
        1
    }
}

/// SoundEntries 4123 `UnderWaterLoop` — the submerged ambience bed the client swaps the
/// ambience slot to (the swap is instant on submerge/emerge — §5 B16).
const UNDERWATER_LOOP_KIT: u32 = 4123;

/// The zone-music transition fade: on a zone-music-id change the outgoing track fades linearly to
/// silence over 4.0 s then stops, byte-exact (`0x4602e0` → `0x7a5a10(0x40800000 = 4.0f)`; wow-re §5
/// **B16**, decision 0100). Both directions. The incoming track starts immediately at FULL volume —
/// the client has no music fade-in (§5 B16, director-confirmed by ear), so this is the only fade on
/// the music slot. It rides the backend fade-stop — drop the handle, the ramp finishes on kira's
/// thread (verified device-free by `mixer::tests::stop_fade_ramps_after_handle_drop`).
const MUSIC_FADE_OUT_MS: u64 = 4000;

/// The Lua `PlayMusic` slot's own base volume: **1.0**, set instantly. `0x460450` puts it through
/// `0x7a5dc0` with no fade and at 1.0f — where the glue theme uses 0.8f (wow-re §5c
/// `glue-music-world-entry.md`) — as a plain scalar under the MusicVolume slider, exactly like a
/// kit's volume, so the slider still rescales it live.
const LUA_MUSIC_VOLUME: f32 = 1.0;

/// What **either** Lua music verb leaves on the zone pump's clock: **now + 6.000 s**
/// (`0x46049b`, `[0x836400] = 0x42c010() + 0x1770`).
///
/// A fourth number beside the three the zone slot already has, and the only one written by a verb
/// rather than by the soundscape: not the `-1` "start now" a zone change and a world entry arm,
/// not the cinematic's 3.000 s ([`CINEMATIC_MUSIC_RESUME_SECS`]), and not `0x4601f0`'s randomized
/// `SilenceInterval` (whose only caller is the natural-end reap, unreachable from here).
///
/// **Unconditional, on both verbs, even from cold** — the write sits between `0x460499`'s
/// `test esi,esi` and its `je`, ahead of the NULL branch — so `StopMusic()` with nothing playing
/// still moves the pump's next start, and can push out a deadline that was already due. That is
/// the side effect a re-implementation drops, and it is why it has a name here.
const LUA_MUSIC_SCHEDULE_SECS: f64 = 6.0;

/// The zone-ambience crossfade: **5.0 s** for an area / interior / day↔night / ghost swap —
/// `0x460b00` fades the old bed out over 5.0 s (`0x7a5a10(0x40a00000 = 5.0f)`) and fades the new
/// bed in from silence over the same 5.0 s (`0x7a5dc0(0)` → `0x7a57b0(5.0f, kit vol)`). Byte-exact
/// (wow-re §5 B16). The submerge/emerge swap is NOT this — it is instant (see the reconcile).
const AMBIENCE_TRANSITION_FADE_MS: u64 = 5000;

/// A linear fade-in envelope for a freshly-started ambience bed: gain ramps 0→1 over `dur_secs`
/// from `start_secs` (`Time::elapsed_secs_f64`). benilla drives the incoming leg of the ambience
/// crossfade itself, per frame — the per-frame volume sync would otherwise snap the new bed
/// straight to full. (Music has no fade-in — §5 B16 — so only ambience uses this.)
#[derive(Clone, Copy)]
struct FadeIn {
    start: f64,
    dur: f64,
}

impl FadeIn {
    fn new(start: f64, dur_ms: u64) -> Self {
        Self {
            start,
            dur: dur_ms as f64 / 1000.0,
        }
    }

    /// Gain in `[0, 1]` at `now`; ramps linearly and holds at 1 once complete.
    fn gain(&self, now: f64) -> f32 {
        if self.dur <= 0.0 {
            return 1.0;
        }
        (((now - self.start) / self.dur) as f32).clamp(0.0, 1.0)
    }
}

/// The active fade-in gain for a slot, clearing the envelope once it reaches full so the
/// per-frame sync returns to a plain snap.
fn fade_in_gain(fade: &mut Option<FadeIn>, now: f64) -> f32 {
    match fade {
        Some(f) => {
            let g = f.gain(now);
            if g >= 1.0 {
                *fade = None;
            }
            g
        }
        None => 1.0,
    }
}

/// Scheduler state + the held stream handles. Non-Send (the handles aren't `Sync`).
pub(super) struct ZoneAudio {
    /// The `ZoneMusic` row currently scheduled (0 = none).
    zone_music: u32,
    /// The playing music stream (a zone track, an intro, or a server-pushed event track).
    music: Option<StreamingSoundHandle<kira::sound::FromFileError>>,
    /// Starvation watch over the music slot (decision 1109) — a streamed track whose decode
    /// thread is outrun zero-fills the mix, which no other meter sees.
    music_watch: mixer::StreamWatch,
    /// The SoundEntries kit on the music slot (0 = never started one) — the "what is playing" a
    /// repeat `SMSG_PLAY_MUSIC` is compared against ([`slot_holds`]).
    music_kit: u32,
    /// The playing track's kit base volume (multiplied under the category slider + fade).
    music_kit_vol: f32,
    /// **The Lua `PlayMusic` slot** — the reference's `[0xb06ccc]`, a *second* music stream beside
    /// the zone track's `[0xb06cc4]` (`0x460450` opens it at `0x4604a7`, stores the handle at
    /// `0x4604b8`), driven by the `PlayMusic`/`StopMusic` pair and nothing else. Held rather than
    /// dropped: a dropped kira handle keeps playing, and the loop below has to be able to see it.
    lua_music: Option<StreamingSoundHandle<kira::sound::FromFileError>>,
    /// The file on that slot, kept because the stream **loops forever**: the reference opens it
    /// through `0x7a5620`'s `a1 != 1` arm, whose `SetLoopCount(-1)` (`0x7a5592`, one call site
    /// image-wide) makes this and the glue theme the client's only two infinite streams. kira
    /// cannot loop a decode-stream (`mixer::loop_from_bytes` docs: an mp3's *estimated* duration
    /// puts `loop_region` past EOF and the stream dies mid-file), so the loop is a restart at the
    /// natural end — the same shape [`super::glue`] gives the theme.
    lua_music_path: Option<String>,
    /// Starvation watch over that slot (decision 1109), like the zone track's — it is a streamed
    /// track of exactly the same class, and an addon's music is the one nothing else reports on.
    lua_music_watch: mixer::StreamWatch,
    /// [`Self::music_pending`] for the **override slot**: the path a `PlayMusic` asked for while
    /// its pack is still warming. Same argument as the zone track's latch — a multi-megabyte
    /// track read from a system would block the frame on a fetch — and the same retry, except
    /// that it is [`lua_music`] that drives it rather than [`retry_deferred`], because that
    /// system is unconditional and this slot is the caller's rather than the world's.
    ///
    /// [`Self::lua_music_path`] is set alongside it, so [`Self::lua_slot_live`] answers `true`
    /// for the whole wait: the slot is *taken* by a track a few frames out, and the zone pump's
    /// five gates must see that or they would start a track underneath it.
    ///
    /// Always `None` on native, where a chain read never defers.
    lua_music_pending: Option<String>,
    /// [`Self::music_decode`] for the override slot — the browser half of the same wait.
    #[cfg(target_arch = "wasm32")]
    lua_music_decode: Option<(String, super::web_decode::Pending)>,
    /// When the next zone track starts (Time::elapsed_secs_f64; None = a track is playing or
    /// no music in this zone).
    next_track_at: Option<f64>,
    /// Last frame's [`SoundConfig::music_suppressed`] — the edge this module acts on.
    music_suppressed: bool,
    /// The looping ambience kit currently up (0 = none) + its handle and base volume. A static
    /// loop, not a stream ([`mixer::loop_from_bytes`] — streaming loop beds die at EOF).
    ambience_kit: u32,
    ambience: Option<mixer::StaticSoundHandle>,
    ambience_kit_vol: f32,
    /// The incoming leg of an ambience crossfade — `Some` while the new bed ramps up.
    ambience_fade_in: Option<FadeIn>,
    /// **The web target's "not here yet" latch for the music slot** (kit id), retried every frame
    /// by [`retry_deferred`] until the bytes are in memory.
    ///
    /// It exists because a track is a multi-megabyte object and the slot is opened from a *system*:
    /// reading one that is not resident would block the frame on a fetch — seconds of frozen tab in
    /// the middle of walking across a zone border. So the open is refused, the pack starts warming,
    /// and the intent is remembered here rather than lost. Every scheduling decision around it
    /// (`zone_music`, `intro_last`, `next_track_at`) is unchanged and already committed; the only
    /// difference is that the track begins a few frames later than it asked to.
    ///
    /// Always `None` on native, where a chain read is a file read and nothing ever defers.
    ///
    /// It holds the whole **pick** — kit, file and kit volume — not just the kit id, because
    /// `pick_stream` depletes: re-rolling the kit on every frame of the wait would scatter the warm
    /// across variations and never finish one. See [`open_music_stream`].
    music_pending: Option<(u32, String, f32)>,
    /// [`Self::music_pending`] for the ambience bed: the pick (kit, file, kit volume) and the fade
    /// it was asked to arrive with. `ambience_kit` is already committed when this is set, so the
    /// retry re-enters the bed player directly rather than going back through the swap's edge check.
    ambience_pending: Option<(u32, String, f32, u64)>,
    /// **The browser's decode in flight for the music slot** — the second half of the same wait
    /// [`Self::music_pending`] is the first half of.
    ///
    /// Deferring the *read* took the fetch off the frame and left the decode on it, and on this
    /// target the decode is the bigger number: kira has no streaming module on wasm32, so
    /// `stream_from_bytes` decodes a whole multi-megabyte track up front, measured at ~91 ms for
    /// a 1.1 MB tavern MP3 against ~60 ms for its fetch. The bytes now go to the browser instead
    /// ([`super::web_decode`]) and the slot opens on the frame the PCM comes back.
    ///
    /// Carries the same pick the read latch does, for the same reason: nothing is re-decided
    /// while a wait is in progress.
    #[cfg(target_arch = "wasm32")]
    music_decode: Option<(u32, String, f32, super::web_decode::Pending)>,
    /// [`Self::music_decode`] for the ambience bed, carrying the fade it was asked to arrive
    /// under as well. The bed absorbs the extra frames by construction — its incoming leg is a
    /// per-frame envelope that simply starts later.
    #[cfg(target_arch = "wasm32")]
    ambience_decode: Option<(u32, String, f32, u64, super::web_decode::Pending)>,
    /// Intro throttle: intro row id → last-played time (secs), against `MinDelayMinutes`.
    intro_last: std::collections::HashMap<u32, f64>,
    /// The last area id acted on (change detection).
    area: Option<u32>,
    /// The last WMO interior acted on (change detection — the override layer, decision 0076).
    interior: Option<super::interior::InteriorAudio>,
    /// Last submersion state — a flip makes the ambience swap INSTANT (the underwater no-fade
    /// branch, wow-re §5 B16), vs the 5.0 s crossfade every other swap gets.
    was_underwater: bool,
    rng: u32,
}

impl Default for ZoneAudio {
    fn default() -> Self {
        Self {
            zone_music: 0,
            music: None,
            music_watch: mixer::StreamWatch::new("zone music"),
            music_kit: 0,
            music_kit_vol: 1.0,
            lua_music: None,
            lua_music_path: None,
            lua_music_watch: mixer::StreamWatch::new("Lua music"),
            lua_music_pending: None,
            #[cfg(target_arch = "wasm32")]
            lua_music_decode: None,
            next_track_at: None,
            music_suppressed: false,
            ambience_kit: 0,
            ambience: None,
            ambience_kit_vol: 1.0,
            ambience_fade_in: None,
            music_pending: None,
            ambience_pending: None,
            #[cfg(target_arch = "wasm32")]
            music_decode: None,
            #[cfg(target_arch = "wasm32")]
            ambience_decode: None,
            intro_last: std::collections::HashMap::new(),
            area: None,
            interior: None,
            was_underwater: false,
            rng: 0x1234_5677,
        }
    }
}

impl ZoneAudio {
    fn rand(&mut self) -> u32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        x
    }

    /// Whether a **Lua `PlayMusic` track is on the override slot** — the reference's
    /// `mov ecx,[0xb06ccc]; test` (`0x4603d6`), asked of the slot's *record* rather than of the
    /// live handle so it is also true across the restart gap between a stream's natural end and
    /// its reopen: that gap is still the caller's track.
    fn lua_slot_live(&self) -> bool {
        self.lua_music_path.is_some()
    }

    /// Uniform silence interval (ms) in `[min, max]` for the phase.
    fn silence_ms(&mut self, min: u32, max: u32) -> u32 {
        if max > min {
            min + self.rand() % (max - min + 1)
        } else {
            min
        }
    }
}

/// Startup: load the zone→audio catalog.
fn load_area_sounds(mut commands: Commands, assets: Option<Res<WorldAssets>>) {
    let Some(assets) = assets else { return };
    let loaded = {
        let mut chain = assets.chain.lock_recover();
        benilla_formats::load_area_sound_catalog(&mut chain)
    };
    match loaded {
        Ok(cat) => {
            info!("sound: {} areas in the zone-audio catalog", cat.len());
            commands.insert_resource(AreaSounds(cat));
        }
        Err(e) => warn!("sound: zone-audio catalog failed to load: {e:#}"),
    }
    let loaded = {
        let mut chain = assets.chain.lock_recover();
        benilla_formats::load_exploration_sound_catalog(&mut chain)
    };
    match loaded {
        Ok(cat) => {
            info!(
                "sound: {} races in the exploration-sound catalog",
                cat.len()
            );
            commands.insert_resource(ExplorationSounds(cat));
        }
        Err(e) => warn!("sound: exploration-sound catalog failed to load: {e:#}"),
    }
}

/// The per-frame scheduler: react to area/phase changes, arm the transition fade-stop, run the
/// silence timer, start the next track, and keep the stream volumes on the sliders.
fn zone_audio(
    mut zone: NonSendMut<ZoneAudio>,
    mut out: NonSendMut<SoundOutput>,
    areas: Option<Res<AreaSounds>>,
    kits: Option<ResMut<SoundKits>>,
    assets: Option<Res<WorldAssets>>,
    config: Res<SoundConfig>,
    clock: Res<GameClock>,
    time: Res<Time>,
    // The stream watch's own clock — WALL, not the paced virtual one `time` carries (see
    // `watch_glue_music`). Kept as a second param because `time` above schedules zone audio and
    // wants the clock it already has.
    real: Res<Time<bevy::time::Real>>,
    world: benilla_world::world_point::WorldPoint,
    interior: Res<super::interior::CurrentInterior>,
    weather: Res<super::weather::WeatherAmbience>,
    self_store: Query<&crate::net::ObjectStore, With<crate::net::SelfPlayer>>,
) {
    let (Some(areas), Some(mut kits), Some(assets)) = (areas, kits, assets) else {
        return;
    };
    let now = time.elapsed_secs_f64();
    let phase = phase(&clock);
    let zone = &mut *zone;

    // ---- the cinematic's music stop (wow-re `sound/scratch/cinematic-audio-law.md`, VERIFIED) ----
    // The flag itself is [`SoundConfig::music_suppressed`], whose doc carries the mechanism. Here
    // are its two edges, and both are byte-shaped rather than chosen:
    //
    // * **Down: a CUT, not a fade.** The disable edge runs `0x7a5700`, stop-and-destroy, which
    //   takes no duration argument at all — there is nothing to fade with. Not the 4.0 s
    //   zone-change fade, not the 250 ms teardown declick.
    // * **Up: +3.000 s, then full volume.** Re-enabling at the stop (`0x4603b0(0)`) sets
    //   `[0x836400] = tick + 0xbb8` — **neither** the `-1` immediate sentinel a zone change or a
    //   world entry arms, **nor** [`next_track_time`]'s randomized `SilenceInterval`. A third
    //   number, and the only place it appears.
    //
    // Ambience is untouched on both edges, deliberately — see the flag's doc for why that is a
    // finding and not an omission.
    apply_music_suppression(zone, config.music_suppressed, now);

    // **Ahead of the cover's hold arm below, which returns.** The flag has to track the cinematic
    // whether or not a loading screen happens to be up, or the edge is deferred until the cover
    // clears — measured at 1.8 s late on a shot far enough to unload the body's ground. Under a
    // cover the work itself is a no-op (the teardown already took the slot), so this is about the
    // flag staying honest, not about the stream.
    // The cover's audio hold ([`SoundConfig::world_hold`]): the raise kills the world
    // soundscape — the reference's loading screen sits over a torn-down world, so a departure
    // zone's bed playing under the destination's cover is a sound it cannot make. Resetting
    // `area`/`interior` with the beds is what makes the reveal a fresh zone entry: the first
    // uncovered frame re-resolves the destination and starts its music/ambience through the
    // ordinary change block below, exactly like walking into the zone.
    if config.world_hold {
        stop_world_soundscape(zone, "loading cover");
        return;
    }

    // ---- react to an area or interior change (the WMO row overrides the terrain chain where
    // its FKs are nonzero — sound::interior, decision 0076) ----
    let inside = interior.0;
    let area = world.area();
    if area != zone.area || inside != zone.interior {
        zone.area = area;
        zone.interior = inside;
        let resolved = area.and_then(|id| areas.0.resolve(id));
        let (mut music_row, mut intro) = match &resolved {
            Some(r) => (r.music.map(|m| m.id).unwrap_or(0), r.intro),
            None => (0, None),
        };
        if let Some(i) = inside {
            if i.zone_music != 0 {
                music_row = i.zone_music;
            }
            if i.intro_sound != 0 {
                intro = areas.0.intro(i.intro_sound);
            }
        }

        if music_row != zone.zone_music {
            zone.zone_music = music_row;
            // Fade the outgoing track linearly to silence over 4.0 s and let the backend stop it
            // (`0x4602e0` → `0x7a5a10(4.0f)`): hand the fade to kira and drop the handle so the
            // per-frame sync stops fighting it — the ramp finishes on the audio thread.
            if let Some(mut h) = zone.music.take() {
                h.stop(mixer::fade(MUSIC_FADE_OUT_MS));
            }
            zone.next_track_at = None;
            // Intro fanfare preempts the zone track: play immediately at full if off cooldown.
            let mut slot_taken = false;
            if let Some(intro) = intro {
                let ok_at = zone
                    .intro_last
                    .get(&intro.id)
                    .map(|t| t + intro.min_delay_minutes as f64 * 60.0);
                if ok_at.is_none_or(|t| now >= t)
                    && intro.sound_id != 0
                    && start_music_stream(
                        zone,
                        &mut out,
                        &mut kits,
                        &assets,
                        &config,
                        intro.sound_id,
                    )
                {
                    zone.intro_last.insert(intro.id, now);
                    slot_taken = true;
                }
            }
            // Otherwise the incoming zone track starts NOW, at full volume — with no exception.
            // The client starts it immediately on a zone-music change (`0x4602e0` forces the
            // next-start time to −1, so `0x460040` opens the stream that tick at full) AND on a
            // fresh world entry (`0x45ffc0` arms the same −1). The randomized silence gap
            // ([`next_track_time`]) is only the *same-zone* track-to-track spacing: decision 0100
            // wrongly reused it on a change, leaving a newly-entered zone silent for 3–5 min, and
            // the 6 s "cold start" that used to sit here was `0x4601f0`'s unreachable arm applied
            // to an entry that never calls it (1553).
            if !slot_taken {
                if let Some(kit) = zone_music_row(&areas.0, music_row).map(|m| m.sounds[phase]) {
                    if kit != 0 {
                        start_music_stream(zone, &mut out, &mut kits, &assets, &config, kit);
                    }
                }
            }
        }
    }

    // ---- ambience reconciliation. The desired-bed priority is the client's own selector
    // (`FUN_00460bd0`, byte-verified): GHOST > submerged > weather-gated zone day/night. The
    // ghost bed is the named "Ghost" SoundEntries kit, resolved through the name registry like
    // the client's literal; its enter/leave swap rides the standard 5.0 s crossfade (the ghost
    // row of the trigger table — zone-music-ambience-transition.md). ----
    let ghost = self_store
        .single()
        .is_ok_and(|store| store.0.player_is_ghost());
    let desired = if ghost {
        kits.id_by_name("Ghost").unwrap_or(0)
    } else if world.submersion().is_water() {
        UNDERWATER_LOOP_KIT
    } else if weather.0 != 0 && world.area_interior().is_none() {
        // The selector's weather branch (`0x460bd0`, 0x460bf7–0x460c17; wow-re
        // `rf-weather-emission-timeline` ROUND 5 Q-C): while the **zonetext indoor bit** is clear
        // (the keep-flag `[0xb06d44]`, dl=0 from the outdoor area feeder) the raw weather
        // SoundEntries IS the ambience bed; the indoor feeder (dl=1) makes the selector ignore
        // the weather id entirely — the area/interior row below plays instead. Both directions
        // ride the one 5.0 s crossfade: stepping into the inn fades the storm out, never ducks it.
        weather.0
    } else if let Some(kit) = zone
        .interior
        .filter(|i| i.ambience != 0)
        .and_then(|i| areas.0.ambience_row(i.ambience))
        .map(|a| a.kits[phase])
        .filter(|k| *k != 0)
    {
        kit
    } else {
        zone.area
            .and_then(|id| areas.0.resolve(id))
            .and_then(|r| r.ambience.map(|a| a.kits[phase]))
            .unwrap_or(0)
    };
    if desired != zone.ambience_kit {
        // The submerge/emerge swap is INSTANT — the underwater caller takes `0x460b00`'s no-fade
        // branch (`0x458650` → `0x460af0(param=1)`, wow-re §5 B16, correcting the old "5 s
        // underwater" of B6). Every other ambience swap — interior, area, day↔night — is the one
        // 5.0 s crossfade (`0x460b00`'s fade branch).
        let fade_ms = if world.submersion().is_water() != zone.was_underwater {
            0
        } else {
            AMBIENCE_TRANSITION_FADE_MS
        };
        swap_ambience(
            zone, &mut out, &mut kits, &assets, &config, desired, fade_ms, now,
        );
    }
    zone.was_underwater = world.submersion().is_water();

    // ---- deferred opens (web target) ----
    // A slot that refused to block on a cold pack left its intent latched; the packs it asked for
    // have been warming since. Retried before the transport below so a track that lands this frame
    // is playing before anything looks at whether one is.
    retry_deferred(zone, &mut out, &assets, &config, now);

    // ---- music transport ----
    // Reap a finished track → schedule the next after a fresh silence interval.
    if let Some(h) = &zone.music {
        if h.state() == kira::sound::PlaybackState::Stopped {
            zone.music = None;
            let zm = zone.zone_music;
            zone.next_track_at =
                next_track_time(zone, &areas.0, zm, phase, now, config.zone_music_no_delay);
        }
    }
    // Silence elapsed → start the next track (same-zone cycle, the cold-start first track, the
    // 3.0 s resume after a cinematic, or the 6.0 s one a Lua music verb left). The pump is dead
    // while suppressed, which is the reference's own shape — `0x460040` bails at its first
    // instruction on the flag, so no track is selected, opened or scheduled for the duration.
    //
    // It bails at its SECOND test on a live Lua override slot (`0x460057`), i.e. without reading
    // the deadline at all, where this block reads it, clears it and is then refused at the slot
    // by [`start_music_stream`]. The audible outcome is the same and cannot come apart, because
    // the only thing that ends an override is the verb itself and the verb always rewrites the
    // deadline on its way through ([`LUA_MUSIC_SCHEDULE_SECS`]): whatever this block consumed
    // meanwhile, zone music comes back 6.0 s after the `StopMusic`.
    if zone.next_track_at.is_some_and(|t| now >= t) && zone.music.is_none() {
        zone.next_track_at = None;
        if let Some(m) = zone_music_row(&areas.0, zone.zone_music) {
            let kit = m.sounds[phase];
            if kit != 0 {
                start_music_stream(zone, &mut out, &mut kits, &assets, &config, kit);
            }
        }
    }

    // ---- per-frame volumes (sliders are live) ----
    // Both feeds `glide` rather than snap (decision 1026): these are the two loudest, longest-lived
    // channels in the mix, so a stepped gain on either is the most audible click there is.
    // Music holds full kit × category volume every frame — its only transition fade is the
    // outgoing 4.0 s fade-stop (on the backend), and the incoming starts at full (faithful).
    if let Some(h) = &mut zone.music {
        h.set_volume(
            mixer::amp_to_db(config.category_amp(SoundCategory::Music) * zone.music_kit_vol),
            mixer::glide(),
        );
    }
    // The starvation watch (1109) over the playing track. A track fading out on a dropped handle
    // isn't watched — under the load bursts that starve a decoder, the *playing* slot's watch is
    // the indicator either way (every stream decoder shares the same scheduling class).
    if let Some(h) = &zone.music {
        zone.music_watch.feed(h, f64::from(real.delta_secs()));
    }
    // Ambience runs the incoming leg of its 5.0 s crossfade as a per-frame fade-in envelope (the
    // per-frame feed would otherwise stomp a handle-level ramp to full); it clears itself at full.
    let ambience_gain = fade_in_gain(&mut zone.ambience_fade_in, now);
    if let Some(h) = &mut zone.ambience {
        h.set_volume(
            mixer::amp_to_db(
                config.category_amp(SoundCategory::Ambience)
                    * zone.ambience_kit_vol
                    * ambience_gain,
            ),
            mixer::glide(),
        );
    }
}

/// The two edges of [`SoundConfig::music_suppressed`] on the music slot — split out so the two
/// byte-shaped numbers in it (a cut with no fade; a resume at exactly +3.000 s) can be asserted
/// without standing up the whole soundscape.
fn apply_music_suppression(zone: &mut ZoneAudio, suppressed: bool, now: f64) {
    if suppressed == zone.music_suppressed {
        return;
    }
    zone.music_suppressed = suppressed;
    // **A live Lua `PlayMusic` track is PAUSED and RESUMED, and nothing else happens on either
    // edge.** `0x4603b0` writes the disabled flag first (`0x4603ba`, so the pump dies here too),
    // then reads the Lua slot at `0x4603d6` and — finding it live — hands the stream `0x7a5ac0`
    // with that same flag and **returns** (`0x4603e0`): the zone stream is not stopped, the intro
    // layer is not killed, and `[0x836400]` is never rearmed, on the way in or out (wow-re
    // `sound/scratch/cinematic-audio-law.md` §1; `sound.md` §"the two handle slots"). A cinematic
    // borrows the caller's track; it does not take it.
    if zone.lua_slot_live() {
        if let Some(h) = zone.lua_music.as_mut() {
            if suppressed {
                h.pause(mixer::declick());
                info!("Lua music: paused for a cinematic");
            } else {
                h.resume(mixer::declick());
                info!("Lua music: resumed");
            }
        }
        return;
    }
    if suppressed {
        if let Some(mut h) = zone.music.take() {
            h.stop(mixer::fade(0));
            info!("zone music: cut for a cinematic");
        }
        // The next login's first track starts at position 0, far behind this one's baseline — the
        // same reason `stop_world_soundscape` resets it (1109).
        zone.music_watch.reset();
        zone.next_track_at = None;
    } else {
        zone.next_track_at = Some(now + CINEMATIC_MUSIC_RESUME_SECS);
        info!("zone music: resumes in {CINEMATIC_MUSIC_RESUME_SECS:.1}s");
    }
}

fn zone_music_row(cat: &AreaSoundCatalog, _id: u32) -> Option<&benilla_formats::ZoneMusicEntry> {
    // The catalog keys zone-music rows by id; expose via resolve()'s cached row instead of a
    // second map walk. (Small helper so the call sites read; see AreaSoundCatalog::zone_music.)
    cat.zone_music(_id)
}

/// When the NEXT track should start after this one ends — the client's `0x4601f0`, whose sole
/// caller is the natural-end reap `0x4600b6`: `None` if the zone has no music; otherwise `now +`
/// the row's randomized per-phase silence interval — **unless `SoundZoneMusicNoDelay` is set, in
/// which case the next track starts now** (`0x42c010`, the CVar's own branch, and the first of the
/// function's three).
///
/// That branch is the whole of what the CVar does, and it is narrower than its options-panel
/// label (*Loop Music*) suggests: it deletes the randomised `ZoneMusic.dbc` SilenceIntervalMin/Max
/// wait between successive plays of the SAME zone's track. A zone CHANGE is immediate either way —
/// the incoming track starts on the next tick while the outgoing fades over 4 s, an overlap rather
/// than a gap (wow-re `zone-music-ambience-transition.md` Q1/Q2, which corrects an earlier framing
/// of exactly this).
///
/// **Not a cold start** — `0x4601f0`'s `== 0 → now + 6000 ms` arm needs a *cleared* currently-playing
/// row, which end-of-track flow never presents, and an entry never reaches this function at all
/// (it takes the −1 "start now" path; module docs, 1553).
fn next_track_time(
    zone: &mut ZoneAudio,
    cat: &AreaSoundCatalog,
    music_row: u32,
    phase: usize,
    now: f64,
    no_delay: bool,
) -> Option<f64> {
    if music_row == 0 {
        return None;
    }
    if no_delay {
        return Some(now);
    }
    // Read the min/max out from under the catalog borrow before the rng draw (which needs `zone`).
    let interval =
        zone_music_row(cat, music_row).map(|m| (m.silence_min[phase], m.silence_max[phase]));
    Some(match interval {
        Some((min, max)) => now + zone.silence_ms(min, max) as f64 / 1000.0,
        None => now + 6.0,
    })
}

/// Whether the music slot is *already* running `kit_id` — the client's already-playing early-out
/// (`0x4602e0` @`0x460342`: `cmp [0xb06cbc],eax; je` → desired == playing → return), which is what
/// makes the server's every-5-s re-push of an event track a no-op instead of a restart. A handle
/// the end-of-track reap hasn't collected yet (`Stopped`) does **not** hold the slot: the track is
/// over, and a push is exactly what should start it again.
fn slot_holds(music_kit: u32, kit_id: u32, slot: Option<kira::sound::PlaybackState>) -> bool {
    music_kit == kit_id && slot.is_some_and(|s| s != kira::sound::PlaybackState::Stopped)
}

/// Open a music kit as a stream on the music slot (a zone track, an intro, or a server-pushed
/// track), replacing whatever is on it. The stream starts at **full** kit × category volume for
/// every start — the client has no music fade-in (§5 B16); the slot's only fade is the outgoing
/// 4.0 s fade-stop. Returns whether a stream actually started.
fn start_music_stream(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    kits: &mut SoundKits,
    assets: &WorldAssets,
    config: &SoundConfig,
    kit_id: u32,
) -> bool {
    // **Nothing opens the music slot while a cinematic is running** — and the guard belongs here,
    // at the slot, not at each caller. The reference's music pump dies at its own first
    // instruction (`0x460040 mov al,[0xb06cc8]; test al,al; jne 0x4600f8`), so suppression is a
    // property of the slot whatever asked for it. benilla gated two of this function's four
    // callers, which left the zone **intro fanfare** and `SMSG_PLAY_MUSIC` free to start a track
    // over a cinematic's narration and play it to the end — a cut only ever happens on the
    // suppression *edge*, so anything that starts after that edge is never cut. The fanfare was
    // the worse half: it stamped itself as played on success, so it was consumed for
    // `MinDelayMinutes` by a shot the player could not hear it under. Refusing here means the
    // stamp never happens, because it is gated on this returning `true`.
    if config.music_suppressed {
        return false;
    }
    // **And nothing opens it while a Lua `PlayMusic` track holds the override slot** — the
    // reference gates on the slot ITSELF, not on a schedule, in five separate places: the pump's
    // own second test (`0x460050 mov eax,[0xb06ccc]; test eax,eax; jne 0x4600f8`), the
    // zone-music change `0x4602e0`, the music-enable toggle `0x460420`, the play-by-id
    // `0x460520` (which is where a server push lands there) and `0x4603b0`'s pause leg. Five
    // gates on one condition is the same argument the cinematic gate above makes: the refusal
    // belongs at the slot, whatever asked for it — and here that covers all four of this
    // function's callers, the zone change, the intro fanfare, the silence-timer pump and
    // `SMSG_PLAY_MUSIC`. With loop count `-1` the override is indefinite: it ends when the
    // caller ends it.
    if zone.lua_slot_live() {
        return false;
    }
    if out.mixer.is_none() {
        return false;
    }
    let Some((path, kit_vol)) = kits.pick_stream(kit_id) else {
        return false;
    };
    open_music_stream(zone, out, assets, config, kit_id, path, kit_vol)
}

/// Open a music track that has **already been picked** — the half of [`start_music_stream`] past the
/// variation roll, split out so the web target's retry ([`retry_deferred`]) can re-enter it with the
/// same file rather than rolling the kit again.
///
/// That distinction is the whole reason for the split. `pick_stream` *depletes*: rolling a kit once
/// per frame while its pack downloads would churn the pool and scatter warms across every variation
/// instead of finishing the one that was chosen. The slot's own guards are re-checked here rather
/// than assumed from the first call — a latch can outlive the state that made it, and a cinematic
/// that started during the wait must still silence the track.
fn open_music_stream(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    assets: &WorldAssets,
    config: &SoundConfig,
    kit_id: u32,
    path: String,
    kit_vol: f32,
) -> bool {
    if config.music_suppressed {
        return false;
    }
    // …and the override slot, re-checked for the same reason suppression is: a latch made frames
    // ago can outlive the state that made it, and a `PlayMusic` that landed during the wait owns
    // the music slot now. Without this the zone track would come back up underneath a Lua track on
    // the web target only — the one arrival the reference's five gates have no frame gap to miss.
    if zone.lua_slot_live() {
        zone.music_pending = None;
        return false;
    }
    // Checked, not borrowed: the mixer is taken by the tail ([`play_music_data`]), which on this
    // target runs frames later. No point reading a track for a device that is not there.
    if out.mixer.is_none() {
        return false;
    }
    let bytes = match assets.chain.lock_recover().try_read(&path) {
        TryRead::Ready(bytes) => bytes,
        // Not in memory yet (web only): the read has started the warm. Remember the pick — file and
        // all — so the retry opens this track rather than re-rolling the kit, and leave the frame.
        //
        // **`true`, not `false`** — the slot IS taken, by a track that has not started yet. The
        // caller reads this as "did this track get the slot?", and answering `false` would let the
        // zone track start on the same edge and overwrite the latch the fanfare just made, so the
        // fanfare could never play on this target. The **disclosed divergence** is the other side
        // of that coin: `MinDelayMinutes` is stamped when the load starts rather than when the
        // track sounds, so a load that fails consumes the cooldown.
        TryRead::Pending => {
            zone.music_pending = Some((kit_id, path, kit_vol));
            return true;
        }
        TryRead::Missing => {
            warn!("zone music: {path} — not readable from the patch chain");
            zone.music_pending = None;
            return false;
        }
    };
    zone.music_pending = None;
    // The browser decodes it (`super::web_decode`) and [`retry_deferred`] opens the slot on the
    // frame the PCM lands. This is the decode half of the same wait the read half above just
    // finished: a whole track decoded here would cost more frame than the fetch did.
    #[cfg(target_arch = "wasm32")]
    {
        zone.music_decode = Some((
            kit_id,
            path,
            kit_vol,
            super::web_decode::begin(bytes, false),
        ));
        // `true` for the same reason the `Pending` arm above answers `true`: the slot is spoken
        // for, by a track a few frames away.
        return true;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let data = match mixer::stream_from_bytes(bytes) {
            Ok(d) => d,
            Err(e) => {
                warn!("zone music: {path} — {e:#}");
                return false;
            }
        };
        play_music_data(zone, out, config, kit_id, &path, kit_vol, data)
    }
}

/// The tail of [`open_music_stream`]: a **decoded** track into the slot.
///
/// Split out because on the web target the decode happens between the two halves, off the frame —
/// so the tail has to be reachable from [`retry_deferred`] as well. The slot's guards were checked
/// by the head and are not re-checked here: this is one open, not two.
fn play_music_data(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    config: &SoundConfig,
    kit_id: u32,
    path: &str,
    kit_vol: f32,
    data: mixer::StreamingSoundData<kira::sound::FromFileError>,
) -> bool {
    let Some(mixer_ref) = out.mixer.as_mut() else {
        return false;
    };
    let start_amp = config.category_amp(SoundCategory::Music) * kit_vol;
    match mixer_ref.play_stream(data.volume(mixer::amp_to_db(start_amp))) {
        Ok(h) => {
            info!("zone music: {path}");
            // One stream per slot: hand the outgoing track the 4.0 s music fade as the incoming
            // takes over (the same overlap `0x4602e0` arms on a zone-music change). It cannot just
            // be dropped — a dropped kira handle keeps playing, so an overwritten stream would
            // stay in the mix at full volume until its file ran out.
            if let Some(mut outgoing) = zone.music.replace(h) {
                outgoing.stop(mixer::fade(MUSIC_FADE_OUT_MS));
            }
            zone.music_kit = kit_id;
            zone.music_kit_vol = kit_vol;
            true
        }
        Err(e) => {
            warn!("zone music: {path} — {e:#}");
            false
        }
    }
}

/// **The Lua music verb — `PlayMusic(name)` and `StopMusic()` are ONE function with one branch.**
///
/// `StopMusic 0x458770` is four instructions, three of which are `0x460450(NULL)` — the very
/// function `PlayMusic 0x458720` calls. So *stopping is playing a NULL name*, and this is that
/// function (wow-re `sound/scratch/lua-music-bindings.md`, the §5 round dispatched for B391;
/// `0x460450` has two callers image-wide and no address-takes, so nothing else in the client can
/// start or stop this slot). The order below is its order:
///
/// 1. **[`take_lua_music_slot`]** — the shared head, which is also the whole of the NULL arm.
/// 2. `0x4604a0` — the branch. A stop is done; everything after this is the name arm.
/// 3. **[`open_lua_music`]** — the stream, looping, at full volume.
/// 4. `0x4604d4`–`0x4604e8` — **if the stream opened**, the zone track fades out over 4.0 s and
///    its slot is cleared. The incoming stream is already at full (the client has no music
///    fade-in), so this is an overlap, not a gap — the same 4.0 s overlap a zone-music change
///    makes.
///
/// **A name that does not resolve stops at step 3**, and that is a confirmed branch rather than a
/// reading: `0x4604bd je 0x460517` returns the moment `0x7a5620` answers NULL, above the volume
/// set and everything after it. What makes it look linear from a distance is that the handle
/// *store* (`0x4604b8`) IS unconditional — it is wedged between `test eax,eax` and that `jcc`. So
/// a mistyped `PlayMusic` still ends whatever the caller had playing and still pushes the pump out
/// 6 s, because both of those are in the head; it leaves a live zone track exactly where it was.
///
/// `assets`/`path` are both `None` for the NULL arm. A *name* arm the caller cannot serve — no
/// store, or the loading cover up — is dropped before it gets here ([`lua_music`]), because a
/// start held behind a cover would arrive in the zone after the one that asked for it.
fn set_lua_music(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    assets: Option<&WorldAssets>,
    config: &SoundConfig,
    path: Option<&str>,
    now: f64,
) {
    take_lua_music_slot(zone, now);
    let (Some(path), Some(assets)) = (path, assets) else {
        return;
    };
    // Step 4 — *only a stream that actually opened takes the zone track down* (`0x4604bd`'s
    // return) — lives in [`play_lua_music_data`], the tail, and not here. On native that is the
    // same instant: the tail runs inside this call. On the web target the open can defer for a
    // few frames while the pack fetches, and taking the zone track down *now* would be the one
    // reading of the reference this seam can get wrong in a way the player hears — a gap of
    // silence where the reference has a 4 s overlap, and a dead zone track if the file turns out
    // not to resolve at all. Deferring the fade with the open keeps both branches faithful.
    open_lua_music(zone, out, assets, config, path);
}

/// `0x460450`'s **shared head**, which is also the whole of its NULL arm: fade this slot's own
/// stream out over **4.0 s** (`0x460480` — a fade, never the `0x7a5700` hard cut, so a second
/// `PlayMusic` crossfades), clear the slot (`0x460485`), and rearm the zone pump
/// ([`LUA_MUSIC_SCHEDULE_SECS`]).
///
/// Split out for the reason [`apply_music_suppression`] is: its two numbers are byte-shaped and
/// this way they can be asserted without standing up a device.
///
/// **It does not touch the zone track.** `[0xb06cc4]`'s only mentions in the reference's function
/// sit *past* the NULL branch, which is exactly what the reporter said from the outside: *"not
/// used to stop normal music played by being in a zone"*.
fn take_lua_music_slot(zone: &mut ZoneAudio, now: f64) {
    if let Some(mut h) = zone.lua_music.take() {
        h.stop(mixer::fade(MUSIC_FADE_OUT_MS));
        info!("Lua music: stopped");
    }
    zone.lua_music_path = None;
    zone.lua_music_watch.reset();
    // A load still in flight for the slot this call is clearing is abandoned here — a `StopMusic`
    // during the wait must not be overtaken by the track it stopped. The pack keeps warming; only
    // the intent is dropped.
    zone.lua_music_pending = None;
    #[cfg(target_arch = "wasm32")]
    {
        zone.lua_music_decode = None;
    }
    zone.next_track_at = Some(now + LUA_MUSIC_SCHEDULE_SECS);
}

/// Open a Lua music stream on the override slot — by path, not by kit, and looping.
///
/// `0x4604a7`–`0x4604bf`: the stream through `0x7a5620`'s looping arm, the handle stored
/// unconditionally, volume **1.0f** recorded instantly (an immediate in `.text`, not a `.data`
/// cell) on the MusicVolume category. Also the **loop**: the reference's `SetLoopCount(-1)` makes
/// the natural end unreachable, so [`lua_music`] reopens here instead.
///
/// **Opened paused under a cinematic.** With music disabled the reference still opens the file,
/// stores the handle and records the volume — only the deferred `_FSOUND_Stream_PlayEx` is gated
/// (`0x4604f2`/`0x4604f9`), and the stream starts the moment music comes back, mid-nothing rather
/// than from the top. A paused stream is that, and the falling edge in
/// [`apply_music_suppression`] is what resumes it. (The `EnableMusic` CVar's own half of that byte
/// needs nothing here: benilla zeroes the category instead of pausing the stream, the divergence
/// [`SoundConfig::music_enabled`] already discloses — and for *this* slot that divergence is the
/// faithful behaviour, because the reference resumes this very stream too.)
// `config` only ever reaches the tail, and on the web target the tail runs from the retry instead
// — so on that target this really is a parameter this function does not read.
#[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
fn open_lua_music(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    assets: &WorldAssets,
    config: &SoundConfig,
    path: &str,
) {
    // Checked, not borrowed: on the web target the mixer is taken by the tail
    // ([`play_lua_music_data`]), which runs frames later.
    if out.mixer.is_none() {
        return;
    }
    // The chain, then the addon folder — an addon that ships its own track names it by a virtual
    // `Interface\AddOns\…` path no MPQ carries ([`benilla_assets::read_chain_or_loose`]).
    //
    // The chain half is the **non-blocking** read, for the reason the zone track's is
    // ([`open_music_stream`]): a track is megabytes, this runs from a system, and on the page a
    // non-resident pack makes that read a synchronous fetch on the thread that draws. The loose
    // half stays a plain read — it is a local file, and the page never has an AddOns folder.
    let attempt = assets.chain.lock_recover().try_read(path);
    let bytes = match attempt {
        TryRead::Ready(bytes) => bytes,
        // Not in memory yet (web only). The read has started the warm; remember the path and
        // **take the slot now** — `lua_music_path` is what [`ZoneAudio::lua_slot_live`] answers
        // from, and the zone pump's gates have to see this frame that the override is spoken for.
        TryRead::Pending => {
            zone.lua_music_pending = Some(path.to_owned());
            zone.lua_music_path = Some(path.to_owned());
            zone.lua_music_watch.reset();
            return;
        }
        TryRead::Missing => match assets.read_loose_only(path) {
            Some(bytes) => bytes,
            None => {
                // A file that does not resolve is the client's own silent nothing: the binding
                // has already returned, so there is nobody left to raise at, and the name went to
                // the file layer verbatim (no extension appended, no normalisation —
                // `0x458720` does neither). The zone track is left exactly where it was, which is
                // `0x4604bd`'s early return.
                debug!("Lua music: {path} — not in the patch chain or the AddOns folder");
                zone.lua_music_pending = None;
                // Hand the slot back. On the first call it is already clear
                // ([`take_lua_music_slot`] ran a line earlier); this matters on the two calls
                // that reach here holding it — a deferred open whose pack turned out not to
                // carry the file, and the loop's reopen of a file that has since gone — where
                // leaving it set would gate the zone pump on a track that can never sound.
                zone.lua_music_path = None;
                return;
            }
        },
    };
    zone.lua_music_pending = None;
    // The decode half of the same wait: kira has no streaming module on wasm32, so
    // `stream_from_bytes` would decode a whole track on the frame. The browser does it instead
    // ([`super::web_decode`]) and the slot opens when the PCM lands.
    #[cfg(target_arch = "wasm32")]
    {
        zone.lua_music_decode = Some((path.to_owned(), super::web_decode::begin(bytes, false)));
        zone.lua_music_path = Some(path.to_owned());
        zone.lua_music_watch.reset();
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let data = match mixer::stream_from_bytes(bytes) {
            Ok(d) => d,
            Err(e) => {
                debug!("Lua music: {path} — {e:#}");
                return;
            }
        };
        play_lua_music_data(zone, out, config, path, data);
    }
}

/// The tail of [`open_lua_music`]: a **decoded** track onto the override slot, and — because this
/// is the only place that knows a stream really started — `0x4604d4`'s fade-out of the zone track
/// underneath it.
///
/// Split out for the reason [`play_music_data`] is: on the web target the decode happens between
/// the two halves, off the frame, so the tail has to be reachable from the retry as well. The
/// slot's guards were checked by the head and are not re-checked here: this is one open, not two.
fn play_lua_music_data(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    config: &SoundConfig,
    path: &str,
    data: mixer::StreamingSoundData<kira::sound::FromFileError>,
) {
    let Some(mixer_ref) = out.mixer.as_mut() else {
        return;
    };
    let start_amp = config.category_amp(SoundCategory::Music) * LUA_MUSIC_VOLUME;
    match mixer_ref.play_stream(data.volume(mixer::amp_to_db(start_amp))) {
        Ok(mut h) => {
            info!("Lua music: {path}");
            if config.music_suppressed {
                h.pause(mixer::fade(0));
                info!("Lua music: held — a cinematic has music disabled");
            }
            // One stream per slot. The verb has already faded the outgoing one, so this only
            // catches the loop's own reopen — and a dropped kira handle keeps playing, so it can
            // never just be overwritten.
            if let Some(mut outgoing) = zone.lua_music.replace(h) {
                outgoing.stop(mixer::declick());
            }
            zone.lua_music_path = Some(path.to_owned());
            zone.lua_music_watch.reset();
            // The zone track's own 4.0 s fade-stop — handed to the backend, handle dropped,
            // exactly as a zone-music change arms it (`0x7a5a10(4.0f)` there and here are the
            // same call). The incoming stream is already at full, so this is an overlap, not a
            // gap. It happens here and not in [`set_lua_music`] because only this point knows the
            // stream opened; see that function's step 4.
            if let Some(mut h) = zone.music.take() {
                h.stop(mixer::fade(MUSIC_FADE_OUT_MS));
                info!("zone music: faded out under a Lua PlayMusic");
            }
        }
        Err(e) => warn!("Lua music: {path} — {e:#}"),
    }
}

/// Drop the Lua stream on a **teardown** — the world's audio dying, not the verb. A short declick
/// rather than the verb's 4.0 s musical fade (the same distinction
/// [`stop_world_soundscape`] draws for the zone slot), and no schedule rearm: nothing is coming
/// back. Idempotent.
fn drop_lua_music(zone: &mut ZoneAudio) {
    if let Some(mut h) = zone.lua_music.take() {
        h.stop(mixer::fade(250));
    }
    zone.lua_music_path = None;
    zone.lua_music_watch.reset();
    // …and a load still in flight for it: nothing is coming back, so nothing should arrive.
    zone.lua_music_pending = None;
    #[cfg(target_arch = "wasm32")]
    {
        zone.lua_music_decode = None;
    }
}

/// Crossfade the looping ambience bed to a new kit (0 = stop), over `fade_ms`:/// Crossfade the looping ambience bed to a new kit (0 = stop), over `fade_ms`: the old bed fades
/// out on the backend (drop the handle — the ramp finishes on kira's thread) while the new bed
/// starts silent and fades in under the per-frame envelope. This mirrors the client's `0x460b00`
/// (a 5.0 s crossfade — out `0x7a5a10(5.0f)`, in `0x7a5dc0(0)` → `0x7a57b0(5.0f)`).
fn swap_ambience(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    kits: &mut SoundKits,
    assets: &WorldAssets,
    config: &SoundConfig,
    kit_id: u32,
    fade_ms: u64,
    now: f64,
) {
    if kit_id == zone.ambience_kit {
        return;
    }
    if let Some(mut h) = zone.ambience.take() {
        h.stop(mixer::fade(fade_ms));
    }
    zone.ambience_kit = kit_id;
    zone.ambience_fade_in = None;
    zone.ambience_pending = None;
    // A decode still running for the bed we just swapped away from is abandoned here rather than
    // left to be refused when it lands: the same drop the read latch above gets, one wait later.
    #[cfg(target_arch = "wasm32")]
    {
        zone.ambience_decode = None;
    }
    if kit_id == 0 {
        return;
    }
    open_ambience_bed(zone, out, kits, assets, config, kit_id, fade_ms, now);
}

/// Open the bed [`swap_ambience`] decided on — the half that touches bytes, split out so the web
/// target's retry ([`retry_deferred`]) can re-enter it without going back through the edge check
/// above, which has already committed `ambience_kit`.
#[allow(clippy::too_many_arguments)]
fn open_ambience_bed(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    kits: &mut SoundKits,
    assets: &WorldAssets,
    config: &SoundConfig,
    kit_id: u32,
    fade_ms: u64,
    now: f64,
) {
    if out.mixer.is_none() {
        return;
    }
    let Some((path, kit_vol)) = kits.pick_stream(kit_id) else {
        return;
    };
    play_ambience_bed(
        zone, out, assets, config, kit_id, path, kit_vol, fade_ms, now,
    );
}

/// Play an ambience bed that has **already been picked** — [`open_ambience_bed`] past the variation
/// roll, split for the same reason [`open_music_stream`] is: `pick_stream` depletes, and the web
/// target's retry must re-open the file it chose rather than roll the kit again on every frame the
/// pack is still coming down.
#[allow(clippy::too_many_arguments)]
fn play_ambience_bed(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    assets: &WorldAssets,
    config: &SoundConfig,
    kit_id: u32,
    path: String,
    kit_vol: f32,
    fade_ms: u64,
    now: f64,
) {
    // Checked, not borrowed — the mixer is the tail's ([`play_ambience_data`]), and so are the
    // config and the clock: on the page they are read frames later, when the bed actually opens.
    if out.mixer.is_none() {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    let (_, _) = (config, now);
    let bytes = match assets.chain.lock_recover().try_read(&path) {
        TryRead::Ready(bytes) => bytes,
        // Not in memory yet (web only) — latch the pick and the fade it asked for, and let the
        // per-frame retry open the bed once the pack lands. Blocking here would freeze the frame on
        // a fetch for a sound the player has not noticed is missing yet.
        TryRead::Pending => {
            zone.ambience_pending = Some((kit_id, path, kit_vol, fade_ms));
            return;
        }
        TryRead::Missing => {
            warn!("ambience: {path} — not readable from the patch chain");
            zone.ambience_pending = None;
            return;
        }
    };
    zone.ambience_pending = None;
    // The decode goes to the browser and the bed opens on the frame the PCM lands — `true` is the
    // whole-file loop `loop_from_bytes` applies on the synchronous path. The crossfade absorbs the
    // extra frames: its incoming leg is a per-frame envelope that simply starts later.
    #[cfg(target_arch = "wasm32")]
    {
        zone.ambience_decode = Some((
            kit_id,
            path,
            kit_vol,
            fade_ms,
            super::web_decode::begin(bytes, true),
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // A static loop, NOT a stream: streaming loop beds die at EOF (mixer::loop_from_bytes docs).
        let data = match mixer::loop_from_bytes(bytes) {
            Ok(d) => d,
            Err(e) => {
                warn!("ambience: {path} — {e:#}");
                return;
            }
        };
        play_ambience_data(zone, out, config, kit_id, &path, kit_vol, fade_ms, now, data);
    }
}

/// The tail of [`play_ambience_bed`]: a **decoded** bed into the slot. Split for the same reason
/// [`play_music_data`] is — on the web target the decode happens between the halves, off the frame.
#[allow(clippy::too_many_arguments)]
fn play_ambience_data(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    config: &SoundConfig,
    kit_id: u32,
    path: &str,
    kit_vol: f32,
    fade_ms: u64,
    now: f64,
    data: mixer::StaticSoundData,
) {
    let Some(mixer_ref) = out.mixer.as_mut() else {
        return;
    };
    let _ = kit_id;
    // Start silent and let the per-frame envelope ramp it up (the fade-in leg of the crossfade).
    let start_amp = if fade_ms > 0 {
        0.0
    } else {
        config.category_amp(SoundCategory::Ambience) * kit_vol
    };
    match mixer_ref.play_2d(data.volume(mixer::amp_to_db(start_amp))) {
        Ok(h) => {
            info!("ambience: {path}");
            zone.ambience = Some(h);
            zone.ambience_kit_vol = kit_vol;
            if fade_ms > 0 {
                zone.ambience_fade_in = Some(FadeIn::new(now, fade_ms));
            }
        }
        Err(e) => warn!("ambience: {path} — {e:#}"),
    }
}

/// Re-attempt the opens the **web target** deferred: a music track or an ambience bed whose bytes
/// were not in memory when the scheduler asked for them.
///
/// It is a per-frame poll and not a callback because that is all it needs to be — the read that
/// deferred also started the pack warming, so this converges in the handful of frames a fetch takes,
/// and the poll itself is a `Map` lookup through the resident bridge. **On native both latches are
/// permanently `None`** (`Chain::try_read` never answers `Pending` there), so this is two `Option`
/// tests and a return.
///
/// Nothing here re-decides anything: the scheduling was settled when the intent was latched, and
/// this only carries it out. A latch is dropped the moment its open succeeds or the file turns out
/// to be missing, both inside the openers.
fn retry_deferred(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    assets: &WorldAssets,
    config: &SoundConfig,
    now: f64,
) {
    // `take`, not `clone`: the openers re-latch on a second `Pending`, so taking it here avoids
    // cloning the path on every frame of the wait and leaves exactly one owner of the intent.
    if let Some((kit, path, kit_vol)) = zone.music_pending.take() {
        // Suppression (a cinematic) is re-checked inside `open_music_stream`, which is the one place
        // that owns the slot's gate — a latch that outlives the reason it was made must not route
        // around it.
        open_music_stream(zone, out, assets, config, kit, path, kit_vol);
    }
    if let Some((kit, path, kit_vol, fade_ms)) = zone.ambience_pending.take() {
        // Only while the bed is still the one wanted — a second swap during the wait re-latched,
        // and a swap to silence cleared it. Either way `take` above is the drop.
        if kit == zone.ambience_kit {
            play_ambience_bed(zone, out, assets, config, kit, path, kit_vol, fade_ms, now);
        }
    }
    // The decode half of the same wait (`super::web_decode`). Polled here rather than woken by a
    // callback for the reason the read half is: this system already runs every frame, and a poll
    // of an `Option` is cheaper than the machinery a callback would need to reach a `NonSend`.
    #[cfg(target_arch = "wasm32")]
    {
        if let Some((kit, path, kit_vol, mut pending)) = zone.music_decode.take() {
            match pending.take() {
                None => zone.music_decode = Some((kit, path, kit_vol, pending)),
                Some(Ok(data)) => {
                    // Re-checked here and not assumed: a cinematic that started during the decode
                    // must still silence the track, which is the same rule the read half follows.
                    if !config.music_suppressed {
                        play_music_data(
                            zone,
                            out,
                            config,
                            kit,
                            &path,
                            kit_vol,
                            mixer::stream_from_decoded(data),
                        );
                    }
                }
                Some(Err(e)) => warn!("zone music: {path} — {e}"),
            }
        }
        if let Some((kit, path, kit_vol, fade_ms, mut pending)) = zone.ambience_decode.take() {
            match pending.take() {
                None => zone.ambience_decode = Some((kit, path, kit_vol, fade_ms, pending)),
                Some(Ok(data)) => {
                    if kit == zone.ambience_kit {
                        play_ambience_data(
                            zone, out, config, kit, &path, kit_vol, fade_ms, now, data,
                        );
                    }
                }
                Some(Err(e)) => warn!("ambience: {path} — {e}"),
            }
        }
    }
}

/// **The Lua music slot, start to finish** — the `PlayMusic`/`StopMusic` intents the VM queued
/// this frame ([`MusicRequest`]), then the slot's own upkeep: the infinite loop, the live slider,
/// the starvation watch.
///
/// Unconditional, unlike the zone scheduler beside it, and that is the slot's nature rather than a
/// convenience: this stream belongs to the *caller*, not to the world's soundscape. It is a second
/// handle in the reference too (`[0xb06ccc]`, not `[0xb06cc4]`), it is not what a zone change,
/// an area's silence interval or the day/night phase acts on, and the one thing the reference is
/// recorded doing to it — a cinematic pausing and resuming it (`0x4603b0`'s `0x7a5ac0` early
/// return, wow-re `cinematic-audio-law.md` §1) — is precisely *not* what that edge does to the
/// zone track, which it stops and reschedules. So it rides its own system, on no run condition
/// but the VM existing.
fn lua_music(
    script: Option<NonSendMut<benilla_ui::script::UiScript>>,
    mut zone: NonSendMut<ZoneAudio>,
    mut out: NonSendMut<SoundOutput>,
    assets: Option<Res<WorldAssets>>,
    config: Res<SoundConfig>,
    // The zone scheduler's own clock — the verb writes `next_track_at`, so it must be the same
    // clock [`zone_audio`] reads it against.
    time: Res<Time>,
    // WALL time, not the paced virtual clock — the watch compares elapsed time against the
    // stream's own position (the reason spelled out on [`zone_audio`]'s own `real` parameter).
    real: Res<Time<bevy::time::Real>>,
) {
    let zone = &mut *zone;
    let now = time.elapsed_secs_f64();
    let requests = script.map(|mut s| s.take_music()).unwrap_or_default();
    for req in requests {
        match req {
            // The NULL arm. It always lands: it needs no file, no device and no catalog — and it
            // is *not* a no-op even from cold, because the pump's clock moves either way.
            benilla_ui::script::MusicRequest::Stop => {
                set_lua_music(zone, &mut out, None, &config, None, now)
            }
            benilla_ui::script::MusicRequest::Play(path) => {
                // Under the loading cover a start is DROPPED, not deferred — the same posture as
                // a server push ([`server_sounds`]) and as the kit player's own gate: the
                // reference's blocking load plays nothing, and a track queued behind the cover
                // would arrive in the destination zone instead of the one that asked for it.
                let Some(assets) = assets.as_deref().filter(|_| !config.world_hold) else {
                    debug!("Lua music: {path} dropped — no assets, or the loading cover is up");
                    continue;
                };
                set_lua_music(zone, &mut out, Some(assets), &config, Some(&path), now);
            }
        }
    }

    // ---- the deferred open (web target) ----
    // A start that refused to block on a cold pack left its intent latched; the pack it asked for
    // has been warming since. Driven from HERE rather than from [`retry_deferred`] because this
    // system is the slot's owner and is unconditional, where the zone scheduler runs only while
    // the world's soundscape is live — and an addon's track is the caller's, not the world's.
    // After the drain above, so a `StopMusic` this frame has already dropped the latch it stops.
    if let Some(assets) = assets.as_deref() {
        retry_lua_music(zone, &mut out, assets, &config);
    }

    // **The loop.** The reference opens this slot with `SetLoopCount(-1)`, so the natural end is a
    // restart from the top rather than a stop; kira cannot loop a decode-stream, so the restart is
    // explicit (the field's own docs, and the same shape [`super::glue`] gives the theme).
    if zone
        .lua_music
        .as_ref()
        .is_some_and(|h| h.state() == kira::sound::PlaybackState::Stopped)
    {
        zone.lua_music = None;
        if let (Some(path), Some(assets)) = (zone.lua_music_path.clone(), assets.as_deref()) {
            open_lua_music(zone, &mut out, assets, &config, &path);
        }
    }

    // The Music slider is live on this stream, as it is on every other music stream in the client
    // (`0x7a6660(ecx=2)`, the MusicVolume re-apply walker) — and `glide`, not a snap, for the same
    // reason the zone track's feed glides (decision 1026).
    if let Some(h) = zone.lua_music.as_mut() {
        h.set_volume(
            mixer::amp_to_db(config.category_amp(SoundCategory::Music) * LUA_MUSIC_VOLUME),
            mixer::glide(),
        );
    }
    if let Some(h) = zone.lua_music.as_ref() {
        zone.lua_music_watch.feed(h, f64::from(real.delta_secs()));
    }
}

/// [`retry_deferred`] for the override slot — the read that answered `Pending` and, on the page,
/// the browser decode behind it.
///
/// Re-enters [`open_lua_music`] with the **same path** rather than re-deciding anything: the verb
/// committed to this file when it ran, the slot has been held for it since
/// ([`ZoneAudio::lua_music_pending`]), and a `StopMusic` in the meantime dropped the latch instead
/// of racing it. **On native both latches are permanently `None`** — `Chain::try_read` never
/// answers `Pending` there — so this is two `Option` tests and a return.
fn retry_lua_music(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    assets: &WorldAssets,
    config: &SoundConfig,
) {
    // `take`, not `clone`: the opener re-latches on a second `Pending`, so this leaves exactly one
    // owner of the intent.
    if let Some(path) = zone.lua_music_pending.take() {
        open_lua_music(zone, out, assets, config, &path);
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let Some((path, mut pending)) = zone.lua_music_decode.take() {
            match pending.take() {
                None => zone.lua_music_decode = Some((path, pending)),
                Some(Ok(data)) => play_lua_music_data(
                    zone,
                    out,
                    config,
                    &path,
                    mixer::stream_from_decoded(data),
                ),
                Some(Err(e)) => {
                    warn!("Lua music: {path} — {e}");
                    // The slot was held for a track that will never sound; hand it back so the
                    // zone pump is not gated on it forever.
                    zone.lua_music_path = None;
                }
            }
        }
    }
}

/// Server-pushed sounds (`SMSG_PLAY_SOUND`/`PLAY_MUSIC`/`PLAY_OBJECT_SOUND`, bridged by
/// `net::apply_net_updates`): 2D kits and object-positioned 3D kits go through the kit player;
/// music takes the zone music slot (interrupting the current track — the zone scheduler resumes
/// its own rotation after the pushed track ends, via the normal silence interval).
fn server_sounds(
    mut msgs: MessageReader<ServerSoundMessage>,
    mut zone: NonSendMut<ZoneAudio>,
    mut out: NonSendMut<SoundOutput>,
    kits: Option<ResMut<SoundKits>>,
    assets: Option<Res<WorldAssets>>,
    config: Res<SoundConfig>,
    listener: Res<AudioListener>,
    transforms: Query<&Transform, Without<Camera3d>>,
) {
    if msgs.is_empty() {
        return;
    }
    // Under the cover a push is dropped, not deferred (the hold, [`SoundConfig::world_hold`]):
    // the reference's blocking load hears none of these, and the recurring pushes (the Darkmoon
    // Faire emitter re-pushes every 5 s) re-arrive on their own after the reveal.
    if config.world_hold {
        msgs.clear();
        return;
    }
    let (Some(mut kits), Some(assets)) = (kits, assets) else {
        return;
    };
    let listener = listener.pos;
    for m in msgs.read() {
        match m.kind {
            ServerSoundKind::Music => {
                // A push for the track already on the slot is a NO-OP (module docs): the server
                // re-pushes an event track on a timer to loop it — the Darkmoon Faire emitter
                // every 5 s — so restarting on each push stacked a fresh full-volume copy of the
                // faire theme onto the mix every 5 s, a dozen at a time.
                if slot_holds(
                    zone.music_kit,
                    m.sound_id,
                    zone.music.as_ref().map(|h| h.state()),
                ) {
                    continue;
                }
                start_music_stream(&mut zone, &mut out, &mut kits, &assets, &config, m.sound_id);
            }
            ServerSoundKind::Sound2d | ServerSoundKind::ObjectSound => {
                // Object sounds position at the source entity when it's streamed to us;
                // otherwise (or for plain 2D) they play flat.
                let pos = m
                    .source
                    .and_then(|e| transforms.get(e).ok())
                    .map(|t| t.translation);
                // A resolved object sound also REGISTERS on its unit — the reference's
                // `AISOUNDDESC` pool (`0x278` → `0x458fb0` → `0x459120`), which its own vocal
                // gates then query through `0x4591f0` so a scripted voice line is not talked over
                // by the creature's grunts ([`Latch::ObjectSound`]). A plain 2D push, or one
                // whose source is not streamed to us, has no unit to register on.
                let source = match m.kind {
                    ServerSoundKind::ObjectSound => m.source,
                    _ => None,
                };
                let latch = if source.is_some() {
                    Latch::ObjectSound
                } else {
                    Latch::None
                };
                if let Err(e) = play_kit_ext(
                    &mut kits,
                    &assets,
                    &mut out,
                    &config,
                    listener,
                    KitRef::Id(m.sound_id),
                    pos,
                    SoundCategory::Sfx,
                    PlayExtras {
                        source,
                        latch,
                        ..default()
                    },
                ) {
                    warn!("server sound {}: {e:#}", m.sound_id);
                }
            }
        }
    }
}

/// `OnExit(InWorld)` — the world soundscape dies with the world. A logout is a session teardown,
/// not a zone change: the beds hard-stop on a short declick ramp (NOT the 4.0 s / 5.0 s musical
/// transitions — the real client destroys the world's sound state outright and the glue theme
/// takes over), and the scheduler resets to cold so the next login resolves fresh from its own
/// area. Deliberately kept: `intro_last` and the RNG — their client counterparts are process-global
/// statics, not per-session state (the intro throttle survives a relog).
fn leave_world(mut zone: NonSendMut<ZoneAudio>) {
    stop_world_soundscape(&mut zone, "left world");
    // The Lua slot too — and only here, not under the loading cover: an addon's track is the
    // caller's, not the world's (see [`lua_music`]), so a teleport's cover leaves it playing,
    // while a logout takes the VM that owns it with it.
    drop_lua_music(&mut zone);
}

/// Stop the beds and reset the scheduler to cold — shared by the two edges where the world's
/// audio dies: leaving the world ([`leave_world`]) and the loading cover rising over it
/// ([`zone_audio`]'s hold arm, decision 1345 — the cover kills the world soundscape exactly
/// as the reference's blocking load does). Idempotent: with the handles already taken, every
/// line is a no-op and nothing logs.
fn stop_world_soundscape(zone: &mut ZoneAudio, reason: &str) {
    // The teardown declick — fast enough to read as a cut, long enough not to pop.
    const WORLD_TEARDOWN_FADE_MS: u64 = 250;
    let had = zone.music.is_some() || zone.ambience.is_some();
    if let Some(mut h) = zone.music.take() {
        h.stop(mixer::fade(WORLD_TEARDOWN_FADE_MS));
    }
    // The next login's first track starts at position 0, far behind this one's baseline — a
    // stale watch would read that as a freeze (1109).
    zone.music_watch.reset();
    if let Some(mut h) = zone.ambience.take() {
        h.stop(mixer::fade(WORLD_TEARDOWN_FADE_MS));
    }
    zone.zone_music = 0;
    zone.music_kit = 0;
    zone.music_kit_vol = 1.0;
    zone.next_track_at = None;
    zone.ambience_kit = 0;
    zone.ambience_kit_vol = 1.0;
    zone.ambience_fade_in = None;
    zone.area = None;
    zone.interior = None;
    zone.was_underwater = false;
    // The cinematic latch is scheduler state like the rest of this list, and leaving it set was
    // the one field that survived a world transition. It only ever self-corrected by luck of
    // ordering: leave the world mid-cinematic and the next entry saw a falling edge that belonged
    // to the *previous* session, logging a resume for a world that had had no cinematic in it.
    zone.music_suppressed = false;
    if had {
        info!("sound: world soundscape stopped ({reason})");
    }
}

/// Registration hook for [`super::SoundPlugin`].
/// Report this module's live stream voices into the global budget (decision 1557).
///
/// Rewritten from the live handles every frame rather than incremented and decremented, so a
/// fade that gets interrupted — or a slot replaced mid-crossfade — cannot leave the budget
/// believing in a voice that stopped. The outgoing leg of an ambience crossfade is deliberately
/// not counted: its handle is released at the swap, so we do not hold it and cannot honestly say
/// whether it is still ringing.
fn report_stream_voices(zone: NonSend<ZoneAudio>, mut out: NonSendMut<super::SoundOutput>) {
    let live = |s: kira::sound::PlaybackState| s != kira::sound::PlaybackState::Stopped;
    out.zone_streams = usize::from(zone.music.as_ref().is_some_and(|h| live(h.state())))
        + usize::from(zone.lua_music.as_ref().is_some_and(|h| live(h.state())))
        + usize::from(zone.ambience.as_ref().is_some_and(|h| live(h.state())));
}

pub(super) fn plugin(app: &mut App) {
    app.insert_non_send_resource(ZoneAudio::default())
        .add_systems(Startup, load_area_sounds.after(AssetSet::Open))
        .add_systems(
            Update,
            (server_sounds, zone_audio)
                .chain()
                .run_if(super::world_audio_live)
                .in_set(WorldStage::Present),
        )
        // Before them, and unconditional — the reasons are on [`lua_music`]. Ordered first so the
        // slot this frame's `PlayMusic` takes is already taken when the zone scheduler runs, and
        // **after the tick** because that is the side of it this system is on: it drains what the
        // VM produced this frame, exactly as [`super::ui`]'s `PlaySound` drain does (2304's rule
        // for a VM holder — `.after(UiInput)` for a drain, `UiFeed` for a push).
        .add_systems(
            Update,
            lua_music
                .before(server_sounds)
                .after(crate::ui_script::UiInput)
                // And after the dev mute chord, for the reason `apply_master_volume` already
                // declares that same order: this system's per-frame gain feed reads
                // `SoundConfig`, so reading it after the flip is what makes the chord audible on
                // the frame it is pressed instead of the next one.
                .after(super::toggle_mute)
                .in_set(WorldStage::Present),
        )
        // Unconditional: the budget must fall back to zero when the soundscape stops, and a
        // system gated on `world_audio_live` would freeze the last count instead.
        .add_systems(Update, report_stream_voices)
        .add_systems(
            OnExit(crate::char_select::ClientState::InWorld),
            leave_world,
        );
}

#[cfg(test)]
mod tests {
    use super::{
        apply_music_suppression, slot_holds, take_lua_music_slot, ZoneAudio,
        CINEMATIC_MUSIC_RESUME_SECS, LUA_MUSIC_SCHEDULE_SECS,
    };
    use kira::sound::PlaybackState;

    /// **The cinematic's music stop, both edges** — wow-re `cinematic-audio-law.md`, VERIFIED.
    /// Down is a CUT: `0x7a5700` is stop-and-destroy and takes no duration argument, so there is
    /// nothing to fade with — not the 4.0 s zone-change fade, not the 250 ms teardown declick.
    /// Up is `[0x836400] = tick + 0xbb8` = **3.000 s**, which is a third number, neither the `-1`
    /// immediate sentinel a zone change arms nor the randomized `SilenceInterval`.
    #[test]
    fn a_cinematic_cuts_the_music_and_brings_it_back_three_seconds_later() {
        let mut zone = ZoneAudio {
            zone_music: 42,
            next_track_at: Some(100.0),
            ..ZoneAudio::default()
        };

        // Down: the pending track is dropped and nothing is scheduled — the pump is dead, not
        // waiting.
        apply_music_suppression(&mut zone, true, 10.0);
        assert!(zone.music_suppressed);
        assert_eq!(
            zone.next_track_at, None,
            "a suppressed pump must schedule nothing"
        );

        // Held: no edge, no rescheduling, however long the intro runs.
        apply_music_suppression(&mut zone, true, 90.0);
        assert_eq!(zone.next_track_at, None);

        // Up: exactly +3.000 s from the release, and the zone row is still remembered so the
        // resume plays THIS zone's track rather than re-resolving from nothing.
        apply_music_suppression(&mut zone, false, 112.5);
        assert!(!zone.music_suppressed);
        assert_eq!(zone.next_track_at, Some(115.5));
        assert_eq!(zone.zone_music, 42);
    }

    /// **A cinematic BORROWS a Lua `PlayMusic` track** — wow-re `cinematic-audio-law.md` §1,
    /// VERIFIED, and the one thing the reference is recorded doing to this slot. `0x4603b0` writes
    /// the disabled flag, sees the slot live at `0x4603d6`, pauses the stream and returns: the
    /// pump still dies (that flag is written *before* the early return) and nothing else moves —
    /// no stop, no clear, and `[0x836400]` never rearmed on the way back out. The zone track's own
    /// edges (a cut, then +3.000 s) are the test above; these are those same two edges with a
    /// caller's track on the slot, and they must do neither of those things.
    #[test]
    fn a_cinematic_borrows_a_lua_music_track_and_leaves_the_schedule_alone() {
        let mut zone = ZoneAudio {
            zone_music: 42,
            next_track_at: Some(100.0),
            lua_music_path: Some("Sound\\Music\\x.mp3".into()),
            ..ZoneAudio::default()
        };

        apply_music_suppression(&mut zone, true, 10.0);
        assert!(
            zone.music_suppressed,
            "the flag is written before the early return, so the pump dies either way"
        );
        assert_eq!(
            zone.next_track_at,
            Some(100.0),
            "the zone schedule is neither cleared nor rearmed"
        );

        apply_music_suppression(&mut zone, false, 112.5);
        assert!(!zone.music_suppressed);
        assert_eq!(
            zone.next_track_at,
            Some(100.0),
            "the +3.000 s resume belongs to the other edge — `[0x836400]` is never rearmed here"
        );
        assert!(
            zone.lua_slot_live(),
            "and the track is still the caller's, across both edges"
        );
    }

    /// The resume is its own number and must not be confused with the two neighbours it sits
    /// between — a zone change's immediate start, and the natural-end silence interval.
    #[test]
    fn the_resume_delay_is_the_references_own_third_number() {
        assert!((CINEMATIC_MUSIC_RESUME_SECS - 3.0).abs() < f64::EPSILON);
    }

    /// **`StopMusic()` is `PlayMusic(NULL)`, and this is the whole of it** — wow-re
    /// `sound/scratch/lua-music-bindings.md`, the round dispatched for B391. `0x458770` is four
    /// instructions, three of which are `0x460450(NULL)`, and that shared head does exactly
    /// three things before the branch: fade this slot over 4.0 s (`0x460480`), clear it
    /// (`0x460485`), and write the zone pump's next start (`0x46049b`).
    ///
    /// Two claims worth a test each, because both are the kind that gets dropped:
    ///
    /// * **The pump write is unconditional** — it sits between `0x460499`'s `test` and its `je`,
    ///   ahead of the NULL branch — so a stop from **cold** still moves the clock, and pushes out
    ///   a start that was already due.
    /// * **The zone track is not what it stops.** `[0xb06cc4]` is mentioned only past that
    ///   branch. This is the reporter's own sentence from the outside — *"not used to stop normal
    ///   music played by being in a zone"* — and it is the half a "stop the music" verb would
    ///   most naturally get wrong.
    #[test]
    fn the_lua_music_verb_rearms_the_zone_pump_at_six_seconds_even_from_cold() {
        // Its own number, and neither of the two it sits between: not the cinematic's 3.000 s,
        // and not the randomized `SilenceInterval` the natural-end reap draws. The second is a
        // `const` assertion because it compares two constants — it holds at compile time, which
        // is the right time for "these are two numbers, not one".
        assert!((LUA_MUSIC_SCHEDULE_SECS - 6.0).abs() < f64::EPSILON);
        const { assert!(LUA_MUSIC_SCHEDULE_SECS != CINEMATIC_MUSIC_RESUME_SECS) };

        // From cold: nothing on the slot, a zone track due this very tick.
        let mut zone = ZoneAudio {
            zone_music: 42,
            next_track_at: Some(10.0),
            ..ZoneAudio::default()
        };
        take_lua_music_slot(&mut zone, 10.0);
        assert_eq!(
            zone.next_track_at,
            Some(16.0),
            "a stop from cold still moves the pump — and clobbers a start that was due now"
        );
        assert!(!zone.lua_slot_live());

        // With a track on the slot: the slot is cleared, the zone's own state is not.
        let mut zone = ZoneAudio {
            zone_music: 42,
            music_kit: 8440,
            lua_music_path: Some("Sound\\Music\\x.mp3".into()),
            ..ZoneAudio::default()
        };
        take_lua_music_slot(&mut zone, 100.0);
        assert!(!zone.lua_slot_live(), "the caller's track is over");
        assert_eq!(zone.next_track_at, Some(106.0));
        assert_eq!(
            (zone.music_kit, zone.zone_music),
            (8440, 42),
            "zone music is not what this verb stops"
        );
    }

    /// The repeat-push guard. The server keeps an event track looping by re-pushing its id every
    /// few seconds (the Darkmoon Faire emitter: `SMSG_PLAY_MUSIC` 8440 every 5 s), so a push for
    /// the track *currently playing* must do nothing — without this the faire stacked a new
    /// full-volume copy of its theme into the mix every 5 s. Once the track ends (or never
    /// started), the same id must start it again: that repeat is what loops the music.
    #[test]
    fn repeat_music_push_only_no_ops_while_the_track_plays() {
        assert!(slot_holds(8440, 8440, Some(PlaybackState::Playing)));
        assert!(!slot_holds(8440, 8440, Some(PlaybackState::Stopped)));
        assert!(!slot_holds(8440, 8440, None));
        // A different track always takes the slot (an intro, a scripted track, a zone change).
        assert!(!slot_holds(8440, 4123, Some(PlaybackState::Playing)));
    }
}
