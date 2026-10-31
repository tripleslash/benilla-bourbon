//! What a shot waiting on its file must honour when it lands ([`super::deferred`]): the stops that
//! came after it, and the retries of the same voice. Native decodes inline, so a play is a channel
//! before anything can stop it and a retry finds it live; the page has only a shot on record.

use std::sync::atomic::{AtomicU64, Ordering};

use bevy::prelude::Entity;

/// One clock for asks and stops, since a ring and its stop can share a frame and a timer tick.
static CLOCK: AtomicU64 = AtomicU64::new(0);

/// The next tick of the clock.
pub(super) fn tick() -> u64 {
    CLOCK.fetch_add(1, Ordering::Relaxed)
}

/// Whether two asks are one voice: a source's kit, whose loop or body sound a driver asks for again
/// every frame until its channel is live. A shot with no source is an event of its own.
pub(super) fn same_voice(a: (u32, Option<Entity>), b: (u32, Option<Entity>)) -> bool {
    a.1.is_some() && a == b
}

/// The stops since the oldest waiting shot; empty once nothing waits.
#[derive(Default)]
pub(crate) struct StopLedger(Vec<Stop>);

struct Stop {
    /// `None` stops everything, the world's exit.
    source: Option<Entity>,
    /// `None` stops all of the source's kits, its despawn.
    kit: Option<u32>,
    at: u64,
}

impl StopLedger {
    /// A stop just reached the channels; a shot waiting under it must not open one afterwards.
    pub(crate) fn note(&mut self, source: Option<Entity>, kit: Option<u32>) {
        self.0.push(Stop {
            source,
            kit,
            at: tick(),
        });
    }

    /// Whether a stop since the ask at `asked` covers `kit` on `source`.
    pub(super) fn stopped_since(&self, source: Option<Entity>, kit: u32, asked: u64) -> bool {
        self.0.iter().any(|s| {
            s.at > asked
                && (s.source.is_none() || (s.source == source && s.kit.is_none_or(|k| k == kit)))
        })
    }

    /// Nothing waits any more.
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIT: u32 = 702;

    fn unit(i: u32) -> Option<Entity> {
        Some(Entity::from_raw_u32(i).unwrap())
    }

    /// An instant cast: the kit rings on a file still loading and its go stops it first. The loop
    /// must not open when the file lands, or nothing is left to end it.
    #[test]
    fn a_stop_after_the_ask_cancels_the_shot() {
        let mut stops = StopLedger::default();
        let asked = tick();
        stops.note(unit(1), Some(KIT));
        assert!(stops.stopped_since(unit(1), KIT, asked));
    }

    /// A rebuild's replay stops the kit and rings it afresh in one frame: the ring stands.
    #[test]
    fn a_stop_before_the_ask_leaves_the_shot() {
        let mut stops = StopLedger::default();
        stops.note(unit(1), Some(KIT));
        let asked = tick();
        assert!(!stops.stopped_since(unit(1), KIT, asked));
    }

    /// One unit's stop spares another unit's shot and the unit's other kits; its despawn takes all
    /// its kits, and the world's exit everything, a shot with no source too.
    #[test]
    fn a_stop_covers_what_it_names() {
        let asked = tick();
        let mut stops = StopLedger::default();
        stops.note(unit(1), Some(KIT));
        assert!(!stops.stopped_since(unit(2), KIT, asked), "another unit");
        assert!(!stops.stopped_since(unit(1), 703, asked), "another kit");
        assert!(!stops.stopped_since(None, KIT, asked), "no source");

        stops.note(unit(2), None);
        assert!(stops.stopped_since(unit(2), 703, asked), "a despawn");

        stops.note(None, None);
        assert!(stops.stopped_since(None, KIT, asked), "the world's exit");
        assert!(stops.stopped_since(unit(3), 703, asked), "the world's exit");

        stops.clear();
        assert!(!stops.stopped_since(unit(3), 703, asked));
    }

    /// A torch's emitter asks for its loop every frame its file is on the way: however many frames
    /// that is, one voice, and one loop when it lands. Events with no source each stand.
    #[test]
    fn a_retry_of_a_sources_kit_is_the_same_voice() {
        assert!(same_voice((KIT, unit(1)), (KIT, unit(1))));
        assert!(!same_voice((KIT, unit(1)), (KIT, unit(2))));
        assert!(!same_voice((KIT, unit(1)), (703, unit(1))));
        assert!(!same_voice((KIT, None), (KIT, None)));
    }
}
