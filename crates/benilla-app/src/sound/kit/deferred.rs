//! wasm32: the one-shot SFX cache's misses, and the shots that asked for them.
//!
//! ## What was left, and what this closes
//!
//! [`super::SoundKits::sfx`] is the decode cache every kit play goes through — weapon swings and
//! impacts, spell casts, creature barks, footsteps, UI clicks. On this target a miss already did
//! not block: `Chain::try_read` answers `Pending` while the pack is on its way, and the play was
//! simply **dropped**. That kept the frame, and it cost the sound: the first hit on a creature
//! type you had not met, the first cast of a spell, the first step on a new material — each was
//! silent, and the cache is dropped on every map change, so it went silent again after every
//! teleport.
//!
//! Two things were missing to make the drop unnecessary:
//!
//! - **The decode was still on the frame.** Once the bytes arrived, `mixer::sfx_from_bytes`
//!   decoded them inline. [`super::super::web_decode`] hands that to the browser.
//! - **Nothing remembered the shot.** The play that missed was gone by the time the file landed.
//!
//! So a miss now records the play against the file it is waiting on — the kit, the variation
//! already drawn, the position, the category, the extras — and [`land_sfx_loads`] replays it
//! through the ordinary play path on the frame the sound is ready. The replay carries the
//! variation as an **explicit index**, so it is the file that was asked for and the depleting pool
//! is not drawn twice. Every gate runs again on the replay, which is what you want: a bark whose
//! unit died in the meantime is refused by the same rules that refuse any other play.
//!
//! **A shot that lands late is dropped, not played late.** [`LATE_WINDOW`] bounds how far behind
//! its event a one-shot may sound; past it the file goes into the cache silently and the *next*
//! play of it is instant. A hit sound a whole second after the hit is worse than no hit sound.
//!
//! ## Pre-warming
//!
//! [`prewarm_creature_voices`] warms the packs behind a creature's vocal kits when the creature
//! appears, so its aggro roar and its wound and death cries are resident before the first swing.
//! Nothing plays, and nothing is decoded — the pack warm is the right granularity here, because a
//! kit's files share one locality group and most of a creature's voice set is never rolled.

use std::collections::hash_map::Entry;
use std::time::Duration;

use bevy::platform::time::Instant;
use bevy::prelude::*;

use benilla_assets::WorldAssets;
use benilla_formats::TryRead;
use benilla_protocol::events::EntityKind;

use super::{
    play_file, play_kit_ext, stop_older_emote_voices, KitRef, Latch, PlayExtras, SoundCategory,
    SoundKits,
};
use crate::sound::creature::CreatureVoices;
use crate::sound::{web_decode, AudioListener, SoundConfig, SoundOutput};

/// How long after its event a one-shot may still sound. A resident pack decodes in a frame or
/// two; a pack still coming down the wire takes a few more. Past this the shot is dropped and only
/// the cache fills.
pub(super) const LATE_WINDOW: Duration = Duration::from_millis(200);

/// One SFX file on its way into the cache, and every play that asked for it while it was.
pub(super) struct PendingSfx {
    /// The chain name as written — what the retry re-reads and what an error line names.
    path: String,
    phase: SfxPhase,
    shots: Vec<(Instant, Deferred)>,
}

/// The two waits a sound goes through on this target: the pack, then the decode.
enum SfxPhase {
    /// Waiting on the pack. The `try_read` that produced this started the warm; ask again next
    /// frame.
    Reading,
    /// The bytes are in hand and the browser is decoding them ([`web_decode`]).
    Decoding(web_decode::Pending),
}

/// A play recorded against a load — everything [`play_kit_ext`] or [`play_file`] needs to be
/// called again exactly as it was, minus the listener position, which the replay reads live.
pub(super) enum Deferred {
    Kit {
        kit: u32,
        /// The variation already drawn from the depleting pool — replayed as an explicit index,
        /// so the pool is not drawn a second time for one event.
        variant: usize,
        pos: Option<Vec3>,
        category: SoundCategory,
        extras: PlayExtras,
    },
    File {
        path: String,
        category: SoundCategory,
    },
}

impl SoundKits {
    /// Note that `key` is on its way, so [`Self::defer`] has something to record a shot against.
    /// Called from [`Self::sfx`]'s two "not here yet" arms; idempotent.
    pub(super) fn begin_sfx(&mut self, key: &str, path: &str, phase_bytes: Option<Vec<u8>>) {
        match self.pending.entry(key.to_string()) {
            // Already in flight. A second ask while a read is outstanding is the ordinary case —
            // several shots of one sound in one burst — and must not restart it.
            Entry::Occupied(mut e) => {
                if let (SfxPhase::Reading, Some(bytes)) = (&e.get().phase, phase_bytes) {
                    e.get_mut().phase = SfxPhase::Decoding(web_decode::begin(bytes, false));
                }
            }
            Entry::Vacant(e) => {
                e.insert(PendingSfx {
                    path: path.to_string(),
                    phase: match phase_bytes {
                        Some(bytes) => SfxPhase::Decoding(web_decode::begin(bytes, false)),
                        None => SfxPhase::Reading,
                    },
                    shots: Vec::new(),
                });
            }
        }
    }

    /// Record a play against the load in flight for `key`, to be replayed when it lands. A key
    /// with nothing in flight (the file turned out to be missing) drops the shot, which is the
    /// answer it would have got anyway.
    pub(super) fn defer(&mut self, key: &str, shot: Deferred) {
        if let Some(p) = self.pending.get_mut(key) {
            p.shots.push((Instant::now(), shot));
        }
    }
}

/// Once a frame: advance every load a step, and replay the shots recorded against the ones that
/// finished and are still inside [`LATE_WINDOW`].
///
/// Ordered before the channel pump so a shot replayed here is pumped on the same frame it starts.
pub(super) fn land_sfx_loads(
    kits: Option<ResMut<SoundKits>>,
    assets: Option<Res<WorldAssets>>,
    mut out: NonSendMut<SoundOutput>,
    config: Res<SoundConfig>,
    listener: Res<AudioListener>,
) {
    let (Some(mut kits), Some(assets)) = (kits, assets) else {
        return;
    };
    if kits.pending.is_empty() {
        return;
    }
    // Step the reads first: a pack that landed this frame starts decoding this frame.
    let mut gone: Vec<String> = Vec::new();
    for (key, p) in kits.pending.iter_mut() {
        if matches!(p.phase, SfxPhase::Reading) {
            // The same loose-flagged read the first attempt made (`0x7a8860`).
            match assets.try_read_loose_or_chain(&p.path) {
                TryRead::Ready(bytes) => {
                    p.phase = SfxPhase::Decoding(web_decode::begin(bytes, false));
                }
                TryRead::Pending => {}
                TryRead::Missing => {
                    warn!("sfx: {} — not readable from the patch chain", p.path);
                    gone.push(key.clone());
                }
            }
        }
    }
    for key in gone {
        kits.pending.remove(&key);
    }
    // Then collect what the browser has finished. Taken out of the map before anything is played,
    // because the replay below re-enters the play path and needs `kits` whole.
    let mut landed = Vec::new();
    kits.pending.retain(|key, p| match &mut p.phase {
        SfxPhase::Reading => true,
        SfxPhase::Decoding(pending) => match pending.take() {
            None => true,
            Some(result) => {
                landed.push((key.clone(), result, std::mem::take(&mut p.shots)));
                false
            }
        },
    });
    if landed.is_empty() {
        return;
    }
    let now = Instant::now();
    let listener = listener.pos;
    for (key, result, shots) in landed {
        let data = match result {
            Ok(d) => d,
            Err(e) => {
                warn!("sfx: {key} — {e}");
                continue;
            }
        };
        kits.cache.insert(key.clone(), data);
        super::DECODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for (asked, shot) in shots {
            let late = now.duration_since(asked);
            if late > LATE_WINDOW {
                debug!(
                    "sfx: {key} landed {} ms after the play that asked — cached, not played",
                    late.as_millis()
                );
                continue;
            }
            let replayed = match shot {
                Deferred::Kit {
                    kit,
                    variant,
                    pos,
                    category,
                    extras,
                } => play_kit_ext(
                    &mut kits,
                    &assets,
                    &mut out,
                    &config,
                    listener,
                    KitRef::Id(kit),
                    pos,
                    category,
                    PlayExtras {
                        variant: Some(variant),
                        ..extras
                    },
                )
                .map(|opened| {
                    // An emote voice that missed opened no channel, so its caller stopped
                    // nothing (`0x623c55`): the replay is the play that takes the unit's slot.
                    if let (true, Latch::EmoteVoice, Some(unit)) =
                        (opened, extras.latch, extras.source)
                    {
                        stop_older_emote_voices(&mut out, unit);
                    }
                }),
                Deferred::File { path, category } => {
                    play_file(&mut kits, &assets, &mut out, &config, &path, category)
                }
            };
            if let Err(e) = replayed {
                debug!("sfx: replay of {key} — {e:#}");
            }
        }
    }
}

/// A creature that has just appeared (or changed its display) warms the packs behind its vocal
/// kits — the barks a fight will ask for first. `Changed` covers the spawn and a later display
/// write alike, and fires rarely enough that the per-kit lookups cost nothing.
///
/// A **pack** warm, not a load: `prefetch_kit` asks the chain for every file the kit could roll,
/// which on this target is one warm for the whole locality group. Decoding a creature's ten vocal
/// files up front would be work for sounds most of them never make.
pub(super) fn prewarm_creature_voices(
    units: Query<&crate::net::NetEntity, Changed<crate::net::NetEntity>>,
    voices: Option<Res<CreatureVoices>>,
    kits: Option<Res<SoundKits>>,
    assets: Option<Res<WorldAssets>>,
) {
    let (Some(voices), Some(kits), Some(assets)) = (voices, kits, assets) else {
        return;
    };
    for net in &units {
        if !matches!(net.kind, EntityKind::Unit) {
            continue;
        }
        let Some(v) = net.display_id.and_then(|d| voices.0.for_display(d)) else {
            continue;
        };
        let warm = [v.aggro, v.alert, v.death, v.stun]
            .into_iter()
            .chain(v.exertion)
            .chain(v.injury)
            .chain(v.custom_attack);
        for kit in warm {
            if kit != 0 {
                kits.prefetch_kit(&assets, kit);
            }
        }
    }
}
