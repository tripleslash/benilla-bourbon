//! The creature-template half of [`crate::names::NameCache`], kept across sessions: benilla's
//! `creaturecache.wdb`.
//!
//! The reference persists only `'WNPC'`; `'WNAM'` and `'WPNM'` (player and pet names) are built
//! with persistence off (`0x554cd0`) and cleared at world-session start
//! ([`NameCache::clear_world_session`]). A template entry keeps its meaning, while a player guid
//! or pet number can come to name another character or spawn.
//!
//! From `DBCache.cpp` (`0x554b00`-`0x5738c0`): the 20-byte header
//! `[FourCC | build 0x16f3 | locale | recordSize | version 1]` is compared by equality, with no
//! checksum or TTL, and a mismatch discards the file ([`NameCache::to_tsv`] writes ours).
//! Eviction is explicit only: a high-bit key in a response, or `SMSG_INVALIDATE_PLAYER` (`0x31C`).
//!
//! Deviation: the file is `benilla-config/cache/<realm>.tsv` through [`crate::local_state`], not
//! `WDB/` in the install, because the install is read-only; and it is per realm, because another
//! server may hold a different template under the same entry.

use std::path::PathBuf;

use bevy::prelude::*;

use crate::glue_session::roster::Roster;
use crate::names::NameCache;

/// The shortest gap between writes: answers arrive in bursts on zone-in, and a crash costs only
/// re-asking.
const SAVE_DEBOUNCE: f32 = 10.0;

/// …and the reason there is a second, longer gap on top of it: **this debounce bounds how often we
/// write, and nothing bounded how much each write costs.**
///
/// Every write serialises the *whole* cache ([`NameCache::to_tsv`]) and hands it to
/// [`crate::local_state::write_atomic`], and the cache only ever grows — the header's own law, from
/// the carve: eviction is explicit, nothing ages out. So the cost of a write climbs all session
/// while the interval stayed at a flat ten seconds, and the share of wall time spent persisting a
/// cache climbed with it. On the web that is felt directly and immediately: `write_atomic` there is
/// `localStorage.setItem`, which is **synchronous on the thread drawing the frame**, so each write
/// is a dropped frame that gets longer the longer the session runs — smooth for the first couple of
/// minutes, then a micro-stutter every ten seconds, worsening. (Native pays it too, into a
/// `sync_all()`, just off the frame.)
///
/// The fix is to make the interval answer to the cost instead of ignoring it: after a write that
/// took `T`, wait at least `T * SAVE_COST_RATIO` before the next one. That turns "how often" into a
/// consequence of "how expensive" and gives the whole mechanism an invariant it did not have —
/// **cache persistence can never take more than 1/SAVE_COST_RATIO of wall time, at any cache
/// size.** A cache that grows does not cost more per second; it just writes less often, which is
/// the right trade for a file whose staleness costs only a re-query.
///
/// 1000 = 0.1% of wall time. A 10 ms write keeps the 10 s floor; a 40 ms one backs off to 40 s.
const SAVE_COST_RATIO: f32 = 1000.0;

pub(crate) struct NamePersistPlugin;

impl Plugin for NamePersistPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NameCacheFile>()
            .add_systems(Update, (load_name_cache, save_name_cache).chain());
        // The exit write catches the last burst, which the debounce would not.
        crate::shutdown::on_app_exit(app, save_on_exit.into_configs());
    }
}

/// Where this realm's cache lives and what we last wrote there.
#[derive(Resource)]
pub(crate) struct NameCacheFile {
    /// The realm the loaded file belongs to; a change triggers a load.
    realm: Option<String>,
    path: Option<PathBuf>,
    /// [`NameCache::generation`] at the last write: the dirty bit, which ticks on an answer.
    saved_generation: u64,
    since_save: f32,
    /// The gap [`since_save`] is currently racing, seconds — [`SAVE_DEBOUNCE`] until a write has
    /// been timed, then whatever [`SAVE_COST_RATIO`] makes of that measurement. Held rather than
    /// recomputed so the backoff survives the frames between writes.
    save_gap: f32,
}

impl Default for NameCacheFile {
    fn default() -> Self {
        Self {
            realm: None,
            path: None,
            saved_generation: 0,
            since_save: 0.0,
            // Before the first write there is nothing to have measured, so the floor stands.
            save_gap: SAVE_DEBOUNCE,
        }
    }
}

/// Loads the realm's cache the first frame its identity is known, and on a realm change.
fn load_name_cache(
    roster: Res<Roster>,
    mut file: ResMut<NameCacheFile>,
    mut names: ResMut<NameCache>,
) {
    let Some((realm, _)) = crate::ui_macro::identity(&roster) else {
        return;
    };
    if file.realm.as_deref() == Some(realm.as_str()) {
        return;
    }
    let path = crate::local_state::name_cache_path(&realm);
    // The last realm's templates go first, so a realm with no file, or a discarded one, starts
    // empty and its first save does not write the other server's records under its name.
    names.install_persisted(NameCache::default());
    file.realm = Some(realm.clone());
    file.path = path.clone();
    file.saved_generation = names.generation();
    let Some(path) = path else {
        return; // a hermetic capture, or no state folder: session-only
    };
    // Through `local_state`, not `std::fs`: on the web the state folder is `localStorage`.
    let Ok(text) = crate::local_state::read_to_string(&path) else {
        return; // no cache yet
    };
    match NameCache::from_tsv(&text, &realm) {
        Some(loaded) => {
            let n = loaded.len();
            // Install only the file's records: this runs as the pick is in flight, beside the
            // login's clear and seed of our own name, and must not replace them.
            names.install_persisted(loaded);
            file.saved_generation = names.generation();
            debug!(
                "names: loaded {n} cached records for {realm} from {}",
                path.display()
            );
        }
        // A header mismatch discards the file, as the reference does; the first save overwrites it.
        None => debug!(
            "names: discarding {} — header is not this build/locale/format",
            path.display()
        ),
    }
}

/// Writes the cache when answers have landed and the current gap has elapsed, then sets the next gap
/// from what that write cost ([`SAVE_COST_RATIO`]).
fn save_name_cache(time: Res<Time>, mut file: ResMut<NameCacheFile>, names: Res<NameCache>) {
    file.since_save += time.delta_secs();
    if file.since_save < file.save_gap {
        return;
    }
    file.since_save = 0.0;
    // Only a write that HAPPENED re-prices the gap. A tick that found nothing to save measures the
    // two comparisons in `write_now` and nothing else, and letting that ~0 ms reading through would
    // reset the backoff to the floor on every quiet tick — so the next real write, however
    // expensive, would always be preceded by a 10 s gap and the invariant would hold only on paper.
    if let Some(ms) = write_now(&mut file, &names) {
        file.save_gap = next_gap(ms);
    }
}

/// The gap a write costing `ms` earns: never under the floor, and never so short that persistence
/// exceeds 1/[`SAVE_COST_RATIO`] of wall time. Split out so the invariant is testable without a
/// filesystem, a `localStorage` or a clock.
fn next_gap(ms: f32) -> f32 {
    SAVE_DEBOUNCE.max(ms / 1000.0 * SAVE_COST_RATIO)
}

/// The exit write, regardless of the debounce and the backoff: there is no next frame to protect.
fn save_on_exit(mut file: ResMut<NameCacheFile>, names: Res<NameCache>) {
    write_now(&mut file, &names);
}

/// Write if anything landed. `Some(ms)` = a write happened and took that long; `None` = there was
/// nothing to write, or nowhere to write it.
fn write_now(file: &mut NameCacheFile, names: &NameCache) -> Option<f32> {
    if names.generation() == file.saved_generation {
        return None; // nothing landed since the last write
    }
    let (Some(path), Some(realm)) = (file.path.clone(), file.realm.clone()) else {
        return None;
    };
    // Both halves are timed, not just the store: serialising the whole cache is itself work that
    // grows with it, and on the web the two are spent back to back on the frame's own thread.
    let t0 = bevy::platform::time::Instant::now();
    // `write_atomic` creates the folder itself (and has none to create on the web).
    let wrote = crate::local_state::write_atomic(&path, &names.to_tsv(&realm));
    let ms = t0.elapsed().as_secs_f32() * 1000.0;
    match wrote {
        Ok(()) => {
            file.saved_generation = names.generation();
            debug!(
                "names: wrote {} records to {} in {ms:.1} ms",
                names.len(),
                path.display()
            );
        }
        Err(e) => warn!("names: cannot write {}: {e}", path.display()),
    }
    // Priced either way: a write that FAILED still spent the serialisation and the store attempt,
    // and a quota-full `localStorage` fails slowly. Backing off on it is right for the same reason
    // backing off on a slow success is.
    Some(ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The invariant the backoff exists for: however big the cache gets, persisting it never takes
    /// more than 1/[`SAVE_COST_RATIO`] of wall time. A flat ten-second gap had no such ceiling —
    /// a 40 ms write every 10 s is 0.4% of the frame budget spent on a cache, four times over.
    #[test]
    fn a_write_can_never_cost_more_than_its_share_of_wall_time() {
        for ms in [0.5, 5.0, 12.0, 40.0, 120.0, 400.0] {
            let gap = next_gap(ms);
            let share = ms / 1000.0 / gap;
            assert!(
                share <= 1.0 / SAVE_COST_RATIO + f32::EPSILON,
                "a {ms} ms write every {gap} s is {share} of wall time"
            );
        }
    }

    /// …and the floor still holds underneath it, so a cheap write does not turn into a busy loop
    /// of rewrites the moment the cache is small.
    #[test]
    fn a_cheap_write_keeps_the_floor() {
        assert_eq!(next_gap(0.0), SAVE_DEBOUNCE);
        assert_eq!(next_gap(1.0), SAVE_DEBOUNCE, "1 ms earns no backoff");
        // 10 ms is exactly the floor's worth; anything above it starts backing off.
        assert_eq!(next_gap(10.0), SAVE_DEBOUNCE);
        assert!(next_gap(40.0) > SAVE_DEBOUNCE, "40 ms must back off");
        assert_eq!(next_gap(40.0), 40.0);
    }

    /// A fresh file races the floor, not zero — before any write there is nothing measured, and
    /// starting at zero would write on the first frame after login.
    #[test]
    fn the_first_gap_is_the_floor() {
        let f = NameCacheFile::default();
        assert_eq!(f.save_gap, SAVE_DEBOUNCE);
        assert_eq!(f.since_save, 0.0);
    }

    use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
    use crate::query_cache::Ask;

    /// A roster with a pick in flight and no realm: [`crate::ui_macro::identity`] answers
    /// `("Realm", <name>)`.
    fn roster_with_pick(name: &str, guid: u64) -> Roster {
        Roster::with_pending_pick(
            vec![benilla_protocol::Character {
                guid,
                name: name.into(),
                race: 1,
                class: 1,
                gender: 0,
                skin: 0,
                face: 0,
                hair_style: 0,
                hair_color: 0,
                facial_hair: 0,
                level: 1,
                zone: 0,
                map: 0,
                position: benilla_protocol::wire::Vector3d {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                flags: 0,
                equipment: [benilla_protocol::CharEnumItem::default(); 19],
                pet_display_id: 0,
                pet_level: 0,
                pet_family: 0,
            }],
            guid,
        )
    }

    /// The load runs after the login's seed of our own name, the order that would lose it, which
    /// left `UnitName("player")` nil on the first login.
    #[test]
    fn the_disk_load_installs_templates_without_touching_the_live_session() {
        use benilla_protocol::guid;
        /// `counter | (entry << 24) | (high << 48)`, the server's composition.
        fn compose(high: u16, entry: u32, counter: u32) -> u64 {
            u64::from(counter) | (u64::from(entry) << 24) | (u64::from(high) << 48)
        }
        const CREATURE_ENTRY: u32 = 1234;
        const PET_NUMBER: u32 = 7;
        let me = compose(guid::HIGH_PLAYER, 0, 0x2A);
        let pet = compose(guid::HIGH_PET, PET_NUMBER, 9);
        let guard = compose(guid::HIGH_UNIT, CREATURE_ENTRY, 1);

        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-namecache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _capture = EnvGuard::unset("WOW_CAPTURE");
        let _home = EnvGuard::set("BENILLA_HOME", tmp.to_str().expect("utf-8 temp path"));

        // A realm file with one creature template, written by the saver so the header matches.
        let mut on_disk = NameCache::default();
        on_disk.insert_creature(
            CREATURE_ENTRY,
            Some(crate::names::CreatureRecord {
                name: "Stormwind Guard".into(),
                subname: None,
                creature_type: 7,
                pet_family: 0,
                rank: 0,
                type_flags: 0,
                civilian: false,
                racial_leader: false,
                display_id: 0,
            }),
        );
        let path = crate::local_state::name_cache_path("Realm").expect("a state folder");
        std::fs::create_dir_all(path.parent().expect("cache dir")).expect("cache dir");
        std::fs::write(&path, on_disk.to_tsv("Realm")).expect("write the realm cache");

        // The login has already seeded our own name and a pet's, as `net::session::connected` does.
        let mut names = NameCache::default();
        names.insert_player(me, "Nelprifour".into(), None);
        names.insert_pet(PET_NUMBER, "Fluffy".into());
        let before = names.generation();

        let mut app = App::new();
        app.insert_resource(roster_with_pick("Nelprifour", me))
            .insert_resource(names)
            .init_resource::<NameCacheFile>()
            .add_systems(Update, load_name_cache);
        app.update();

        let names = app.world().resource::<NameCache>();
        assert_eq!(
            names.resolve_unit(me, None, Ask::Read),
            Some("Nelprifour"),
            "the login's seed of our own name must survive the disk load — losing it is the bug"
        );
        assert_eq!(
            names.resolve_unit(pet, None, Ask::Read),
            Some("Fluffy"),
            "…and so must a pet name: the file carries neither, so neither is its to replace"
        );
        assert_eq!(
            names.resolve_unit(guard, None, Ask::Read),
            Some("Stormwind Guard"),
            "the file's own records are what the load is for"
        );
        assert!(
            names.generation() > before,
            "a landed record moves the counter the gated feeds watch"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
