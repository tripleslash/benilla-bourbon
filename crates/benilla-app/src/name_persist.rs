//! **The name cache, kept across sessions** (decision 1689) — the load/save half of
//! [`crate::names::NameCache`], and benilla's answer to the reference's `WDB/*.wdb` files.
//!
//! ## Why this exists at all
//!
//! Three of benilla's caches are filled by a server round trip: player names, creature templates
//! and pet names. The reference persists all three (`namecache.wdb`, `creaturecache.wdb`,
//! `petnamecache.wdb`), so on a machine that has played before, an answer is usually *already
//! there* when the packet that needs it arrives. benilla asked fresh every login, which made a
//! race the reference has practically closed wide open for us — the director hit it as a Lua error
//! when unstabling a pet, where `PetStable_Update` hands a nil `UnitName("pet")` to a
//! `GameTooltip:SetText` whose signature requires a string (decision 1688 closed that one path by
//! seeding from the stable list; this closes the class).
//!
//! ## The law, from the carve
//!
//! wow-re carved the whole `DBCache.cpp` machine (`system/dbcache/dbcache.md`, T3 — all 12 record
//! decoders diffed bit-exact). Two of its contracts are the ones a re-implementation must honour,
//! and both are about what the cache does *not* do:
//!
//! - **The header is compared by equality, and carries no checksum, no timestamp and no TTL.** Its
//!   20 bytes are `[FourCC | build 0x16f3 | locale | recordSize | version 1]`. A mismatch discards
//!   the file wholesale. Ours is a header line with the same job ([`NameCache::to_tsv`]).
//! - **Eviction is explicit only** — a high-bit key in a response, or `SMSG_INVALIDATE_PLAYER`
//!   (`0x31C` → remove-by-key `0x556ff0`). Nothing ages out. So a persisted name lives until the
//!   server says otherwise, which is why that opcode had to be wired before anything was written
//!   to disk: in memory a stale name costs a session, on disk it costs forever.
//!
//! ## Where it lives, and the one place we deviate
//!
//! The reference writes `WDB/` **inside the install**, which is precisely where benilla may not
//! write (the contract's read-only rule). Ours goes to `benilla-config/cache/<realm>.tsv` through
//! [`crate::local_state`], like every other thing we persist — and it is **realm-scoped**, which
//! the reference's is not. That is a fix rather than a preference: every key is realm-local (a
//! player guid, a creature entry, a pet number), so one shared file would serve another realm's
//! names to this one.

use std::path::PathBuf;

use bevy::prelude::*;

use crate::char_select::Roster;
use crate::names::NameCache;

/// The **shortest** gap between writes. The cache is worth keeping but never worth a stall: names
/// arrive in bursts (a city's worth of creature queries on zone-in), and writing the whole file per
/// answer would turn a burst into hundreds of rewrites. A crash inside the window costs re-asking,
/// which is exactly what the cache is for.
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
        // The exit edge (decision 1528): the last burst of answers is worth the one write, and it
        // is the only chance to take it — the debounce above will not fire again.
        crate::shutdown::on_app_exit(app, save_on_exit.into_configs());
    }
}

/// Where this realm's cache lives and what we last wrote there.
#[derive(Resource)]
pub(crate) struct NameCacheFile {
    /// The realm the loaded file belongs to; `None` before the first login. A *change* is what
    /// triggers a load — logging into a second realm must not keep the first's names.
    realm: Option<String>,
    path: Option<PathBuf>,
    /// [`NameCache::generation`] as of the last successful write. The cache's own landed-answer
    /// counter is the dirty bit — it ticks on an arrival and never on an ask, which is exactly the
    /// edge worth writing on.
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

/// Load the realm's cache the first frame its identity is known, and on any realm change.
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
    file.realm = Some(realm.clone());
    file.path = path.clone();
    file.saved_generation = names.generation();
    let Some(path) = path else {
        return; // a hermetic capture, or no state folder — session-only, exactly as before
    };
    // Through `local_state`, not `std::fs`: on the web the state folder is `localStorage`.
    let Ok(text) = crate::local_state::read_to_string(&path) else {
        return; // no cache yet: the ordinary first-run path, not an error
    };
    match NameCache::from_tsv(&text, &realm) {
        Some(loaded) => {
            let n = loaded.len();
            // Replace rather than merge: this is the *start* of a realm's session, so there is
            // nothing of this realm's in memory to preserve, and a merge would silently keep the
            // previous realm's records alive under their own keys.
            *names = loaded;
            file.saved_generation = names.generation();
            debug!(
                "names: loaded {n} cached records for {realm} from {}",
                path.display()
            );
        }
        // The header did not match — a different build, locale or format. Discarding the whole
        // file is the reference's own rule, and the file is left on disk to be overwritten by the
        // first save rather than deleted out from under a player who may be switching back.
        None => debug!(
            "names: discarding {} — header is not this build/locale/format",
            path.display()
        ),
    }
}

/// Write the cache when answers have landed since the last write and the current gap has elapsed —
/// then set the next gap from what that write actually cost ([`SAVE_COST_RATIO`]).
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

/// The exit-edge write (decision 1528) — unconditional on the debounce, because there is no next
/// frame to defer to. The backoff does not apply here either: the gap exists to keep persistence
/// off a frame the player is watching, and on the way out there is no next frame to protect.
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
        // A cache we could not write is a cache we re-fill next session — worth a line, never
        // worth failing a session over.
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
}
