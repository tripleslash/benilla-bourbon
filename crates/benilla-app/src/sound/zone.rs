//! Zone music and ambience: the area-driven schedulers over the kit player.
//!
//! `CurrentArea` (MCNK areaId) → `AreaTable` → `ZoneMusic`/`SoundAmbience`/`ZoneIntroMusicTable`
//! (parent-inherited, `AreaSoundCatalog::resolve`) → SoundEntries kits → streamed MP3/WAV.
//!
//! Music (`0x460040`–`0x460ca0`): same-zone tracks are spaced by a random per-phase silence
//! interval (`0x4601f0`). A zone-music change (`0x4602e0`) or world entry (`0x45ffc0`, at
//! `0x45ffdb`) arms the next start to −1 at `0x836400`, so the incoming track opens that tick
//! (`0x460240`) at full volume: no fade-in primitive (`0x7a57b0`, `0x7a5730`) is called on a
//! music slot.
//!
//! Server-pushed music (`SMSG_PLAY_MUSIC`) takes the same slot. The server re-pushes an event
//! track to loop it (the Darkmoon Faire every 5 s, vmangos `go_scripts.cpp:318`), so a push for
//! the kit already playing is a no-op, the reference's early-out at `0x460342`. Any other push
//! takes the slot as a zone-music change does; the reference's handler is untraced.
//!
//! Ambience (`0x460b00`): an area, interior, day/night or ghost change crossfades over 5.0 s;
//! submerge/emerge is instant (`0x458650` → `0x460af0`). Day is 05:30 ≤ t < 21:00 on the
//! server-synced clock, a hard step with no fade (`0x4578c0`, via the `0x642710` range check).

use bevy::prelude::*;

use benilla_formats::{AreaSoundCatalog, TryRead};

use crate::net::{ServerSoundKind, ServerSoundMessage};
use benilla_assets::DataLoadAppExt;
use benilla_assets::{LockRecover, WorldAssets};
use benilla_world::lighting::GameClock;
use benilla_world::schedule::WorldStage;

use super::kit::{play_kit_ext, KitRef, Latch, PlayExtras, SoundCategory, SoundKits};
use super::mixer::{self, StreamingSoundHandle};
use super::{AudioListener, SoundConfig, SoundOutput};

/// The zone-to-audio catalog; absent when the client data did not load.
#[derive(Resource)]
pub(crate) struct AreaSounds(pub(crate) AreaSoundCatalog);

/// The race to exploration-jingle catalog (`ChrRaces.dbc` column 3), played on
/// `SMSG_EXPLORATION_EXPERIENCE`; absent when the client data did not load.
#[derive(Resource)]
pub(crate) struct ExplorationSounds(pub(crate) benilla_formats::ExplorationSoundCatalog);

/// How long after a cinematic ends the zone track comes back: `[0x836400] = tick + 0xbb8` at
/// `0x4603b0(0)`.
const CINEMATIC_MUSIC_RESUME_SECS: f64 = 3.0;

fn phase(clock: &GameClock) -> usize {
    if (330..1260).contains(&clock.minute) {
        0
    } else {
        1
    }
}

/// SoundEntries 4123 `UnderWaterLoop`, the submerged ambience bed.
const UNDERWATER_LOOP_KIT: u32 = 4123;

/// The outgoing music fade, the only fade on the music slot: `0x4602e0` →
/// `0x7a5a10(0x40800000 = 4.0f)`.
const MUSIC_FADE_OUT_MS: u64 = 4000;

/// The Lua `PlayMusic` slot's base volume, set with no fade under the MusicVolume slider
/// (`0x460450` → `0x7a5dc0`; the glue theme uses 0.8, `0x45aeb0`).
const LUA_MUSIC_VOLUME: f32 = 1.0;

/// What either Lua music verb leaves on the zone pump's clock: now + 6.000 s (`0x46049b`,
/// `[0x836400] = 0x42c010() + 0x1770`). Unconditional, even from cold: the write sits between
/// `0x460499`'s `test` and its `je`, so `StopMusic()` with nothing playing still pushes out a
/// start that was due.
const LUA_MUSIC_SCHEDULE_SECS: f64 = 6.0;

/// The ambience crossfade, both legs (`0x460b00`: out `0x7a5a10(0x40a00000 = 5.0f)`, in
/// `0x7a5dc0(0)` → `0x7a57b0(5.0f, kit vol)`); submerge/emerge is instant instead.
const AMBIENCE_TRANSITION_FADE_MS: u64 = 5000;

/// The linear fade-in of a new ambience bed, driven per frame because the per-frame volume feed
/// would override a handle-level ramp.
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

    /// Gain in `[0, 1]` at `now`, holding at 1 once complete.
    fn gain(&self, now: f64) -> f32 {
        if self.dur <= 0.0 {
            return 1.0;
        }
        (((now - self.start) / self.dur) as f32).clamp(0.0, 1.0)
    }
}

/// The active fade-in gain for a slot, clearing the envelope once it reaches full.
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

/// Scheduler state and the held stream handles; non-Send, as the handles are not `Sync`.
pub(super) struct ZoneAudio {
    /// The `ZoneMusic` row currently scheduled (0 = none).
    zone_music: u32,
    /// The music slot: a zone track, an intro, or a server-pushed event track.
    music: Option<StreamingSoundHandle<kira::sound::FromFileError>>,
    music_watch: mixer::StreamWatch,
    /// The SoundEntries kit on the music slot (0 = none), what a repeat `SMSG_PLAY_MUSIC` is
    /// compared against.
    music_kit: u32,
    music_kit_vol: f32,
    /// The Lua `PlayMusic` slot, the reference's `[0xb06ccc]`: a second music stream beside the
    /// zone track's `[0xb06cc4]`, driven only by `PlayMusic`/`StopMusic`.
    lua_music: Option<StreamingSoundHandle<kira::sound::FromFileError>>,
    /// The file on that slot, kept to restart it: `0x7a5620` opens it with `SetLoopCount(-1)`
    /// (`0x7a5592`), and kira cannot loop a decode-stream.
    lua_music_path: Option<String>,
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
    /// When the next zone track starts (`Time::elapsed_secs_f64`); `None` while one plays or the
    /// zone has no music.
    next_track_at: Option<f64>,
    /// Last frame's [`SoundConfig::music_suppressed`], for its edges.
    music_suppressed: bool,
    /// The looping ambience kit currently up (0 = none), its handle and base volume.
    ambience_kit: u32,
    ambience: Option<mixer::StaticSoundHandle>,
    ambience_kit_vol: f32,
    /// The incoming leg of an ambience crossfade, while the new bed ramps up.
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
    /// Intro row id to last-played time (secs), against `MinDelayMinutes`.
    intro_last: std::collections::HashMap<u32, f64>,
    area: Option<u32>,
    /// The last WMO interior acted on, the override layer.
    interior: Option<super::interior::InteriorAudio>,
    /// Last submersion state; a flip makes the ambience swap instant.
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

    /// Whether a Lua `PlayMusic` track holds the override slot (`0x4603d6`); asked of the path,
    /// not the handle, so it stays true across the loop's restart gap.
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

/// Startup: load the zone-audio and exploration-sound catalogs.
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

/// The per-frame scheduler: area and phase changes, the silence timer, the next track, and the
/// stream volumes.
fn zone_audio(
    mut zone: NonSendMut<ZoneAudio>,
    mut out: NonSendMut<SoundOutput>,
    areas: Option<Res<AreaSounds>>,
    kits: Option<ResMut<SoundKits>>,
    assets: Option<Res<WorldAssets>>,
    config: Res<SoundConfig>,
    clock: Res<GameClock>,
    time: Res<Time>,
    // The stream watch compares against wall time, not the paced virtual clock.
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

    // The cinematic's music stop: a cut, then a resume at +3.000 s; ambience is untouched.
    // Ahead of the cover's hold arm, which returns, so the edge is not deferred to the reveal.
    apply_music_suppression(zone, config.music_suppressed, now);

    // The loading cover kills the world soundscape, as the reference's loading screen sits over
    // a torn-down world; resetting `area`/`interior` makes the reveal a fresh zone entry.
    if config.world_hold {
        stop_world_soundscape(zone, "loading cover");
        return;
    }

    // ---- area or interior change (the WMO row overrides the terrain chain where nonzero) ----
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
            // The fade finishes on the audio thread after the handle drops.
            if let Some(mut h) = zone.music.take() {
                h.stop(mixer::fade(MUSIC_FADE_OUT_MS));
            }
            zone.next_track_at = None;
            // The intro preempts the zone track, at full, if off cooldown.
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
            // Otherwise the zone track starts now, on a change and on world entry alike (the −1
            // that `0x4602e0` and `0x45ffc0` arm); the silence interval is same-zone spacing only.
            if !slot_taken {
                if let Some(kit) = zone_music_row(&areas.0, music_row).map(|m| m.sounds[phase]) {
                    if kit != 0 {
                        start_music_stream(zone, &mut out, &mut kits, &assets, &config, kit);
                    }
                }
            }
        }
    }

    // ---- ambience: the reference's selector (`0x460bd0`) ranks ghost > submerged > weather >
    // interior/zone day/night; the ghost bed is the kit named "Ghost", its swap the 5.0 s
    // crossfade (`0x458680` → `0x460c20`) ----
    let ghost = self_store
        .single()
        .is_ok_and(|store| store.0.player_is_ghost());
    let desired = if ghost {
        kits.id_by_name("Ghost").unwrap_or(0)
    } else if world.submersion().is_water() {
        UNDERWATER_LOOP_KIT
    } else if weather.0 != 0 && world.area_interior().is_none() {
        // `0x460bf7`–`0x460c17`: with the zonetext indoor bit `[0xb06d44]` clear, the weather's
        // SoundEntries is the bed; indoors the selector ignores it, and the swap crossfades.
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
        // Submerge/emerge takes `0x460b00`'s no-fade branch (`0x458650` → `0x460af0(1)`).
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
    // Reap a finished track and schedule the next after a silence interval (`0x4600b6`).
    if let Some(h) = &zone.music {
        if h.state() == kira::sound::PlaybackState::Stopped {
            zone.music = None;
            let zm = zone.zone_music;
            zone.next_track_at =
                next_track_time(zone, &areas.0, zm, phase, now, config.zone_music_no_delay);
        }
    }
    // The deadline passed: start the next track. Under a live Lua slot the reference's pump bails
    // before reading the deadline (`0x460057`) while this consumes it and is refused at the slot;
    // the same outcome, since only the verb ends the override and it always rewrites the deadline.
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
    if let Some(h) = &mut zone.music {
        h.set_volume(
            mixer::amp_to_db(config.category_amp(SoundCategory::Music) * zone.music_kit_vol),
            mixer::glide(),
        );
    }
    // A track fading out on a dropped handle is not watched.
    if let Some(h) = &zone.music {
        zone.music_watch.feed(h, f64::from(real.delta_secs()));
    }
    // The incoming leg of the ambience crossfade.
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

/// The two edges of [`SoundConfig::music_suppressed`] on the music slot: down is a cut, since
/// `0x7a5700` stop-and-destroy takes no duration; up resumes at +3.000 s.
fn apply_music_suppression(zone: &mut ZoneAudio, suppressed: bool, now: f64) {
    if suppressed == zone.music_suppressed {
        return;
    }
    zone.music_suppressed = suppressed;
    // A live Lua track is paused and resumed, and nothing else: `0x4603b0` writes the flag
    // (`0x4603ba`), finds the slot live at `0x4603d6`, pauses it (`0x7a5ac0`) and returns
    // (`0x4603e0`), so the zone stream is not stopped and `[0x836400]` is never rearmed.
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
        // The next track starts at position 0, behind this one's baseline.
        zone.music_watch.reset();
        zone.next_track_at = None;
    } else {
        zone.next_track_at = Some(now + CINEMATIC_MUSIC_RESUME_SECS);
        info!("zone music: resumes in {CINEMATIC_MUSIC_RESUME_SECS:.1}s");
    }
}

fn zone_music_row(cat: &AreaSoundCatalog, _id: u32) -> Option<&benilla_formats::ZoneMusicEntry> {
    cat.zone_music(_id)
}

/// When the next track starts after this one ends, the reference's `0x4601f0` (called only from
/// the natural-end reap `0x4600b6`): `now` plus the row's random per-phase silence interval, or
/// `now` under `SoundZoneMusicNoDelay` (`0x42c010`). That CVar removes only the same-zone gap; a
/// zone change is immediate either way (`0x460346`). Its `== 0 → now + 6000 ms` arm is not a cold
/// start: world entry never reaches this function.
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
    let interval =
        zone_music_row(cat, music_row).map(|m| (m.silence_min[phase], m.silence_max[phase]));
    Some(match interval {
        Some((min, max)) => now + zone.silence_ms(min, max) as f64 / 1000.0,
        None => now + 6.0,
    })
}

/// Whether the music slot is already running `kit_id`, the reference's early-out at `0x460342`
/// (`cmp [0xb06cbc],eax; je`). A `Stopped` handle not yet reaped does not hold the slot.
fn slot_holds(music_kit: u32, kit_id: u32, slot: Option<kira::sound::PlaybackState>) -> bool {
    music_kit == kit_id && slot.is_some_and(|s| s != kira::sound::PlaybackState::Stopped)
}

/// Open a music kit on the music slot at full volume (`0x460240` → `0x7a5dc0`), replacing what
/// is there; whether a stream started.
fn start_music_stream(
    zone: &mut ZoneAudio,
    out: &mut SoundOutput,
    kits: &mut SoundKits,
    assets: &WorldAssets,
    config: &SoundConfig,
    kit_id: u32,
) -> bool {
    // Nothing opens the slot during a cinematic, whoever asks: the reference's pump dies at its
    // first instruction (`0x460040`, `[0xb06cc8]`). Refusing here also keeps an intro from being
    // stamped as played.
    if config.music_suppressed {
        return false;
    }
    // Nor while a Lua track holds the override slot: the reference gates on the slot itself in
    // the pump (`0x460050`), the zone-music change `0x4602e0`, the enable toggle `0x460420`, the
    // play-by-id `0x460520` and `0x4603b0`'s pause leg.
    if zone.lua_slot_live() {
        return false;
    }
    if out.mixer.is_none() {
        return false;
    }
    let Some((path, kit_vol)) = kits.pick_stream(kit_id) else {
        return false;
    };
    // Deviation: Turtle WoW's radio kits (60401, 60423, Everlook's `.radio` stations) name an
    // http stream, which its client plays off the network; this one reads the chain only, so the
    // kit is dropped, reported once, however often the server re-pushes it. No chain path holds
    // "://", and the flavor is read only for one that does.
    if path.contains("://") && crate::flavor::turtle() {
        if kits.first_unplayable(kit_id) {
            warn!("zone music: kit {kit_id} streams {path} — no network music, dropped");
        }
        return false;
    }
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
            // One stream per slot: a dropped kira handle keeps playing, so the old one fades.
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

/// The Lua music verb, the reference's `0x460450`: `PlayMusic` (`0x458720`) calls it with a name
/// and `StopMusic` (`0x458770`) with NULL, in this order:
///
/// 1. [`take_lua_music_slot`], the shared head and the whole NULL arm;
/// 2. the branch at `0x4604a0`;
/// 3. [`open_lua_music`], looping at full volume; a name that does not resolve returns here
///    (`0x4604bd`), leaving the zone track playing;
/// 4. only if the stream opened, the zone track fades out over 4.0 s (`0x4604d4`–`0x4604e8`).
///
/// `assets` and `path` are `None` for the NULL arm.
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

/// `0x460450`'s shared head: fade this slot out over 4.0 s (`0x460480`, so a second `PlayMusic`
/// crossfades), clear it (`0x460485`) and rearm the zone pump. It never touches the zone track:
/// `[0xb06cc4]` appears only past the NULL branch.
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

/// Open a Lua music stream on the override slot, by path and looping (`0x4604a7`–`0x4604bf`,
/// through `0x7a5620`'s looping arm); [`lua_music`] reopens here at the natural end.
///
/// Opened paused under a cinematic: the reference still opens the file and gates only the
/// deferred `_FSOUND_Stream_PlayEx` (`0x4604f2`/`0x4604f9`), starting it when music comes back.
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
    // The chain, then the addon folder: an addon's own track has an `Interface\AddOns\…` path.
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
            // Only the loop's reopen finds a stream here; the verb already faded the old one.
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

/// Drop the Lua stream on a teardown: a short declick and no schedule rearm. Idempotent.
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

/// Crossfade the looping ambience bed to a new kit (0 = stop) over `fade_ms`: the old bed fades
/// out on the backend while the new one starts silent under the per-frame envelope.
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
        play_ambience_data(
            zone, out, config, kit_id, &path, kit_vol, fade_ms, now, data,
        );
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

/// The Lua music slot: this frame's queued verbs, the loop, the slider and the watch.
/// Unconditional: the stream is the caller's, not the world soundscape's.
fn lua_music(
    script: Option<NonSendMut<benilla_ui::script::UiScript>>,
    mut zone: NonSendMut<ZoneAudio>,
    mut out: NonSendMut<SoundOutput>,
    assets: Option<Res<WorldAssets>>,
    config: Res<SoundConfig>,
    // The clock `zone_audio` reads `next_track_at` against.
    time: Res<Time>,
    // Wall time for the stream watch.
    real: Res<Time<bevy::time::Real>>,
) {
    let zone = &mut *zone;
    let now = time.elapsed_secs_f64();
    let requests = script.map(|mut s| s.take_music()).unwrap_or_default();
    for req in requests {
        match req {
            // The NULL arm always lands, and moves the pump's clock even from cold.
            benilla_ui::script::MusicRequest::Stop => {
                set_lua_music(zone, &mut out, None, &config, None, now)
            }
            benilla_ui::script::MusicRequest::Play(path) => {
                // Under the loading cover a start is dropped, not deferred: the reference's
                // blocking load plays nothing.
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

    // The loop: `SetLoopCount(-1)` in the reference, an explicit restart here.
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

    // The Music slider is live here too (`0x7a6660(ecx=2)`, the MusicVolume re-apply walker).
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
                Some(Ok(data)) => {
                    play_lua_music_data(zone, out, config, &path, mixer::stream_from_decoded(data))
                }
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

/// Server-pushed sounds (`SMSG_PLAY_SOUND`/`PLAY_MUSIC`/`PLAY_OBJECT_SOUND`): kits go through
/// the kit player, music takes the zone music slot until it ends.
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
    // Under the cover a push is dropped, not deferred; recurring pushes re-arrive after it.
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
                // A push for the track already on the slot is a no-op.
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
                // At the source entity when it is streamed to us; otherwise flat.
                let pos = m
                    .source
                    .and_then(|e| transforms.get(e).ok())
                    .map(|t| t.translation);
                // An object sound registers on its unit (the reference's `AISOUNDDESC` pool,
                // `0x458fb0` → `0x459120`), so the vocal gates (`0x4591f0`) keep the creature's
                // grunts off a scripted voice line.
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

/// `OnExit(InWorld)`: the world soundscape dies with the world. The reference destroys its world
/// sound state outright; here the beds stop on a 250 ms declick, not the musical fades.
/// `intro_last` and the RNG survive a relog, as process-global statics do in the reference.
fn leave_world(mut zone: NonSendMut<ZoneAudio>) {
    stop_world_soundscape(&mut zone, "left world");
    // The Lua slot only here: a loading cover leaves the caller's track playing.
    drop_lua_music(&mut zone);
}

/// Stop the beds and reset the scheduler to cold, on leaving the world or under the loading
/// cover. Idempotent.
fn stop_world_soundscape(zone: &mut ZoneAudio, reason: &str) {
    // Fast enough to read as a cut, long enough not to pop.
    const WORLD_TEARDOWN_FADE_MS: u64 = 250;
    let had = zone.music.is_some() || zone.ambience.is_some();
    if let Some(mut h) = zone.music.take() {
        h.stop(mixer::fade(WORLD_TEARDOWN_FADE_MS));
    }
    // The next track starts at position 0, behind this one's baseline.
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
    // Else the next entry sees a falling edge left from this session's cinematic.
    zone.music_suppressed = false;
    if had {
        info!("sound: world soundscape stopped ({reason})");
    }
}

/// Report this module's live stream voices into the global budget, recounted from the held
/// handles every frame; a released outgoing ambience bed is not counted.
fn report_stream_voices(zone: NonSend<ZoneAudio>, mut out: NonSendMut<super::SoundOutput>) {
    let live = |s: kira::sound::PlaybackState| s != kira::sound::PlaybackState::Stopped;
    out.zone_streams = usize::from(zone.music.as_ref().is_some_and(|h| live(h.state())))
        + usize::from(zone.lua_music.as_ref().is_some_and(|h| live(h.state())))
        + usize::from(zone.ambience.as_ref().is_some_and(|h| live(h.state())));
}

pub(super) fn plugin(app: &mut App) {
    app.insert_non_send(ZoneAudio::default())
        .add_data_loaders(load_area_sounds)
        .add_systems(
            Update,
            (server_sounds, zone_audio)
                .chain()
                .run_if(super::world_audio_live)
                .in_set(WorldStage::Present),
        )
        // Before the zone scheduler, so this frame's `PlayMusic` already holds the slot, and
        // after the VM tick, whose output it drains.
        .add_systems(
            Update,
            lua_music
                .before(server_sounds)
                .after(crate::ui_script::UiInput)
                // After the dev mute chord, so the gain feed hears it the same frame.
                .after(super::toggle_mute)
                .in_set(WorldStage::Present),
        )
        // Unconditional, so the count falls to zero when the soundscape stops.
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

    #[test]
    fn a_cinematic_cuts_the_music_and_brings_it_back_three_seconds_later() {
        let mut zone = ZoneAudio {
            zone_music: 42,
            next_track_at: Some(100.0),
            ..ZoneAudio::default()
        };

        apply_music_suppression(&mut zone, true, 10.0);
        assert!(zone.music_suppressed);
        assert_eq!(
            zone.next_track_at, None,
            "a suppressed pump must schedule nothing"
        );

        apply_music_suppression(&mut zone, true, 90.0);
        assert_eq!(zone.next_track_at, None);

        // Up: +3.000 s, and the zone row is remembered.
        apply_music_suppression(&mut zone, false, 112.5);
        assert!(!zone.music_suppressed);
        assert_eq!(zone.next_track_at, Some(115.5));
        assert_eq!(zone.zone_music, 42);
    }

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

    #[test]
    fn the_resume_delay_is_the_references_own_third_number() {
        assert!((CINEMATIC_MUSIC_RESUME_SECS - 3.0).abs() < f64::EPSILON);
    }

    /// `StopMusic()` is `PlayMusic(NULL)` (`0x458770` → `0x460450`): the pump write is
    /// unconditional, and the zone track is not what it stops.
    #[test]
    fn the_lua_music_verb_rearms_the_zone_pump_at_six_seconds_even_from_cold() {
        assert!((LUA_MUSIC_SCHEDULE_SECS - 6.0).abs() < f64::EPSILON);
        const { assert!(LUA_MUSIC_SCHEDULE_SECS != CINEMATIC_MUSIC_RESUME_SECS) };

        // From cold: nothing on the slot, a zone track due this tick.
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

    /// 8440 is the Darkmoon Faire music the server re-pushes every 5 s.
    #[test]
    fn repeat_music_push_only_no_ops_while_the_track_plays() {
        assert!(slot_holds(8440, 8440, Some(PlaybackState::Playing)));
        assert!(!slot_holds(8440, 8440, Some(PlaybackState::Stopped)));
        assert!(!slot_holds(8440, 8440, None));
        assert!(!slot_holds(8440, 4123, Some(PlaybackState::Playing)));
    }
}
