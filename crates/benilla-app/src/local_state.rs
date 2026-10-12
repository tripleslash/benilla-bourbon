//! Where benilla keeps local state: one folder, `benilla-config/`, never inside the install, and
//! this module is the only place that computes a path into it; each path fn's doc says what it
//! holds. The 1.12 client is portable the same way, its `WTF/` and `Cache/` in its own folder.
//!
//! Resolution, in order, the same shape as [`benilla_formats::wow_data`]:
//! 1. `$BENILLA_HOME`.
//! 2. An unattended run ([`crate::run_mode::unattended`]): a fresh folder of its own for the run,
//!    under the system temp folder, so a probe neither reads the player's settings nor writes back
//!    over them.
//! 3. `<project folder>/benilla-config/`, dev builds only ([`benilla_formats::project_folder`], the
//!    launcher's): a shipped binary must not carry the build machine's source tree.
//! 4. `<exe dir>/benilla-config/`.
//!
//! With `$WOW_CAPTURE` set every path resolves to `None`: a capture neither reads nor writes
//! player state.

use std::io::Write;
use std::path::{Path, PathBuf};

/// The folder's name: not `benilla`, which beside the binary is the executable itself, and not
/// `WTF`, which a real install already uses.
const STATE_DIR: &str = "benilla-config";

/// The state folder, or `None` when persistence is off (a capture run, or no executable path).
/// It may not exist yet: [`write_atomic`] creates it, and a reader treats a missing file as
/// defaults.
// The wasm32 early return below makes the rest of the function genuinely unreachable there
// (by design — see its own comment); `cfg_attr` keeps that lint quiet only on the target it
// actually applies to, so a real unreachable branch on native would still warn.
#[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
pub(crate) fn home() -> Option<PathBuf> {
    // On the page the folder is virtual: every resident is a small text file, and they live in
    // the browser's `localStorage` under keys named by these same paths (see [`web_store`]). The
    // callers' contract is unchanged — a path in, `read_to_string`/`write_atomic` on it — which
    // is what makes this the one place the web needs to know about. An unattended page keeps the
    // same store: a tab has no temp folder to give a run of its own.
    #[cfg(target_arch = "wasm32")]
    return Some(PathBuf::from("/benilla-config"));
    home_in(cfg!(test))
}

/// [`home`], with the unit-test stop as an argument so a test can walk the steps behind it.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn home_in(unit_test: bool) -> Option<PathBuf> {
    if std::env::var_os("WOW_CAPTURE").is_some() {
        return None; // hermetic: captures neither read nor write player state
    }
    if let Some(over) = std::env::var_os("BENILLA_HOME") {
        return Some(PathBuf::from(over));
    }
    if unit_test {
        return None; // a unit test reaches no real state folder unless it pins `$BENILLA_HOME`
    }
    if crate::run_mode::unattended() {
        return Some(run_home().to_path_buf());
    }
    resident_home()
}

/// Step 2 of [`home`]: the unattended run's own folder, one per process, created on first write.
/// Left behind for whoever reads what the run wrote; the system cleans its temp folder.
pub(crate) fn run_home() -> &'static Path {
    static RUN_HOME: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    RUN_HOME.get_or_init(|| run_home_in(&std::env::temp_dir(), std::process::id()))
}

/// [`run_home`]'s path under `temp` for process `pid`.
fn run_home_in(temp: &Path, pid: u32) -> PathBuf {
    temp.join(format!("benilla-run-{pid}")).join(STATE_DIR)
}

/// Steps 3 and 4 of [`home`]: the folder the build itself lives in.
fn resident_home() -> Option<PathBuf> {
    // 3 · the project folder, dev builds only.
    if let Some(root) = dev_project_root() {
        return Some(root.join(STATE_DIR));
    }
    // 4 · beside the binary; a dev build never reaches here.
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
        .map(|dir| dir.join(STATE_DIR))
}

/// The project folder a dev build keeps its state in ([`benilla_formats::project_folder`]), through
/// [`shared_root`].
fn dev_project_root() -> Option<PathBuf> {
    Some(shared_root(benilla_formats::project_folder()?))
}

/// `here`, or for a linked worktree its primary checkout, whichever worktree built the binary, so
/// every worktree shares one settings folder. Anything unexpected keeps `here`.
fn shared_root(here: &Path) -> PathBuf {
    let dot_git = here.join(".git");
    // A linked worktree: `.git` is a file pointing at `<primary>/.git/worktrees/<name>`.
    if dot_git.is_file() {
        if let Some(gitdir) = std::fs::read_to_string(&dot_git).ok().and_then(|s| {
            s.trim()
                .strip_prefix("gitdir:")
                .map(|p| p.trim().to_owned())
        }) {
            // …/.git/worktrees/<name> → …/.git → the primary checkout.
            if let Some(primary) = Path::new(&gitdir)
                .ancestors()
                .nth(2)
                .and_then(|d| d.parent())
            {
                if primary.is_dir() {
                    return primary.to_path_buf();
                }
            }
        }
    }
    here.to_path_buf()
}

/// `benilla-config/config.toml`: the CVar overrides, the `Config.wtf` analog.
pub(crate) fn config_path() -> Option<PathBuf> {
    home().map(|h| h.join("config.toml"))
}

/// `benilla-config/macros/account.txt`: the account-wide macro tab (indices 1..=18).
pub(crate) fn macros_account_path() -> Option<PathBuf> {
    home().map(|h| h.join("macros/account.txt"))
}

/// `benilla-config/macros/<realm>-<character>.txt`: the per-character macro tab (19..=36), the
/// reference's `WTF/Account/<ACC>/<REALM>/<CHAR>/macros-cache.txt` flattened into one folder.
pub(crate) fn macros_character_path(realm: &str, character: &str) -> Option<PathBuf> {
    let key = format!("{}-{}", file_token(realm), file_token(character));
    home().map(|h| h.join("macros").join(format!("{key}.txt")))
}

/// `benilla-config/bindings/account.txt`: the account-wide key bindings (the `bindings-cache.wtf`
/// analog), a diff against the defaults.
pub(crate) fn bindings_account_path() -> Option<PathBuf> {
    home().map(|h| h.join("bindings/account.txt"))
}

/// `benilla-config/bindings/<realm>-<character>.txt`: the character-specific binding set. Its
/// existence is the "character specific key bindings" state; going back to general deletes it.
pub(crate) fn bindings_character_path(realm: &str, character: &str) -> Option<PathBuf> {
    let key = format!("{}-{}", file_token(realm), file_token(character));
    home().map(|h| h.join("bindings").join(format!("{key}.txt")))
}

/// `benilla-config/saved-variables.lua`: the flat channel our FrameXML saves through
/// `RegisterForSave` (the `SavedVariables.lua` analog), written whole at logout, run at UI load.
pub(crate) fn saved_variables_path() -> Option<PathBuf> {
    home().map(|h| h.join("saved-variables.lua"))
}

/// `benilla-config/addons/<Realm>-<Character>.txt`: the AddOn enable state, per character in the
/// reference's `AddOns.txt` format (`<AddOnName>: enabled|disabled`), which it writes at the tail
/// of its UI shutdown (`0x490bd0`). An addon absent from the file is enabled.
pub(crate) fn addons_state_path(realm: &str, character: &str) -> Option<PathBuf> {
    let key = format!("{}-{}", file_token(realm), file_token(character));
    home().map(|h| h.join("addons").join(format!("{key}.txt")))
}

/// `benilla-config/saved/`: per-addon saved variables, account scope, one `<Addon>.lua` per addon
/// declaring `## SavedVariables` (the reference's `WTF/Account/<ACC>/SavedVariables/<Addon>.lua`).
pub(crate) fn addon_saved_account_dir() -> Option<PathBuf> {
    home().map(|h| h.join("saved"))
}

/// `benilla-config/saved/<Realm>-<Character>/`: per-addon saved variables, character scope
/// (`## SavedVariablesPerCharacter`), loaded after the account file so it wins.
pub(crate) fn addon_saved_character_dir(realm: &str, character: &str) -> Option<PathBuf> {
    let key = format!("{}-{}", file_token(realm), file_token(character));
    home().map(|h| h.join("saved").join(key))
}

/// `benilla-config/camera/<realm>-<character>.txt`: the third-person camera pose (the
/// `<Char>/camera-settings.txt` analog), two lines in the reference's keys and order.
pub(crate) fn camera_character_path(realm: &str, character: &str) -> Option<PathBuf> {
    let key = format!("{}-{}", file_token(realm), file_token(character));
    home().map(|h| h.join("camera").join(format!("{key}.txt")))
}

/// `benilla-config/account`: the account name the login screen remembers
/// (`GetSavedAccountName`/`SetSavedAccountName`; the reference's is `Config.wtf`'s `accountName`).
pub(crate) fn saved_account_path() -> Option<PathBuf> {
    home().map(|h| h.join("account"))
}

/// `benilla-config/chat/<realm>-<character>.txt`: the chat windows' tint, alpha, font size and
/// lock, the four the reference keeps in its per-character `chat-cache.txt`.
pub(crate) fn chat_character_path(realm: &str, character: &str) -> Option<PathBuf> {
    Some(home()?.join("chat").join(format!(
        "{}-{}.txt",
        file_token(realm),
        file_token(character)
    )))
}

/// `benilla-config/layout/<realm>-<character>.txt`: the layout cache, the geometry of every
/// user-placed frame, as the reference's per-character `layout-cache.txt`; [`crate::ui_layout`]
/// owns the shape.
pub(crate) fn layout_character_path(realm: &str, character: &str) -> Option<PathBuf> {
    Some(home()?.join("layout").join(format!(
        "{}-{}.txt",
        file_token(realm),
        file_token(character)
    )))
}

/// `benilla-config/cache/<realm>.tsv`: the player, creature and pet names the server has answered,
/// one file for the reference's `WDB/namecache.wdb`, `creaturecache.wdb` and `petnamecache.wdb`.
///
/// Realm-scoped, where the reference's is not: guids, creature entries and pet numbers are
/// realm-local, so one shared file would serve a realm another realm's names.
pub(crate) fn name_cache_path(realm: &str) -> Option<PathBuf> {
    Some(
        home()?
            .join("cache")
            .join(format!("{}.tsv", file_token(realm))),
    )
}

/// `benilla-config/Logs/`: `WoWChatLog.txt` and `WoWCombatLog.txt`, the reference's names.
pub(crate) fn logs_dir() -> Option<PathBuf> {
    home().map(|h| h.join("Logs"))
}

/// `benilla-config/shots.txt`: the framing instrument's camera poses (`/shot`, dev builds).
pub(crate) fn shots_path() -> Option<PathBuf> {
    home().map(|h| h.join("shots.txt"))
}

/// `benilla-config/Screenshots/`: where the print-screen key writes, each image once, not through
/// [`write_atomic`].
///
/// Deviation: the reference writes `Screenshots\\` inside the install, which benilla never writes
/// to; the folder keeps the reference's name.
pub(crate) fn screenshots_dir() -> Option<PathBuf> {
    home().map(|h| h.join("Screenshots"))
}

/// `benilla-config/Diagnostics/`: the stuck-thread sampler's profiles ([`crate::perf::stall`]) and
/// the FPS journal; `None` on a capture run like every path here, so a capture has no sampler.
pub(crate) fn diagnostics_dir() -> Option<PathBuf> {
    home().map(|h| h.join("Diagnostics"))
}

/// `benilla-config/Diagnostics/fps-journal.csv`: the FPS journal's rows while the `fpsJournal`
/// CVar is on; a capture names its own path through `WOW_FPS_JOURNAL`.
///
/// **And `None` on the page**, which is the one resident of this folder that does not follow
/// [`home`] into `localStorage`. Every other resident is a small file rewritten when something
/// changes; the journal is an APPEND once a second whose contents grow for as long as the tab is
/// open, and a `localStorage` write is synchronous on the thread drawing the frame (`write_meter`
/// below is the meter that exists because of it) — so the instrument built to find stutter would be the
/// tab's biggest source of it, worsening the longer a reporter ran it. The browser's own
/// instrument is `crate::webdiag`, which reports to the console and holds nothing.
pub(crate) fn fps_journal_path() -> Option<PathBuf> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        diagnostics_dir().map(|d| d.join("fps-journal.csv"))
    }
}

/// A realm or character name as one path component: anything not a letter or digit becomes `_`.
/// Letters of any script stay, as in the reference's raw-name folders: vmangos accepts non-Latin
/// names (`StrictPlayerNames = 0`) and a character name is letters only, so no two share a file.
fn file_token(s: &str) -> String {
    let t: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    if t.is_empty() {
        "unknown".into()
    } else {
        t
    }
}

/// Write a state file atomically, through `<path>.tmp` and a rename, so a crash leaves the old
/// file intact.
pub(crate) fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    write_atomic_bytes(path, contents.as_bytes())
}

/// [`write_atomic`] for bytes: the saved-variables files carry Lua byte strings as they are.
///
/// fork: on the page the store holds text, so bytes that are not UTF-8 are kept losslessly by
/// [`web_store::set_bytes`] and handed back as the same bytes by [`read`].
pub(crate) fn write_atomic_bytes(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    #[cfg(target_arch = "wasm32")]
    {
        // Metered (see [`write_meter`]): on the page this call is a **synchronous**
        // `localStorage.setItem` on the frame's own thread, and every byte of it is spent there.
        let t0 = bevy::platform::time::Instant::now();
        let r = web_store::set_bytes(path, contents);
        write_meter::record(path, contents.len(), t0.elapsed().as_secs_f32() * 1000.0);
        return r;
    }
    #[cfg(not(target_arch = "wasm32"))]
    write_atomic_fs(path, contents)
}

/// **What the page's state writes cost the frame** — the meter `perf::webdiag` reports.
///
/// `write_atomic` is one line on native (a file write on a thread nobody is waiting for) and
/// something else entirely on the web: `localStorage.setItem` is synchronous, it blocks the thread
/// that calls it, and that thread is the one drawing the frame. So a resident of this folder that
/// is rewritten on a timer, and whose contents *grow* with the session, is a stutter that gets
/// worse the longer the tab is open — and it is invisible to every frame-cost meter we have, which
/// can say a frame went long but not that a storage write is why.
///
/// The meter names the file as well as the cost, because "state writes cost 41 ms this window" and
/// "`<realm>.tsv` cost 41 ms this window" are one round trip apart.
#[cfg(target_arch = "wasm32")]
pub(crate) mod write_meter {
    use std::cell::RefCell;
    use std::path::Path;

    thread_local! {
        /// `file name -> (writes, bytes on the last one, total ms)` since the last [`take`].
        static SEEN: RefCell<Vec<(String, u32, usize, f32)>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn record(path: &Path, bytes: usize, ms: f32) {
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        SEEN.with(|s| {
            let mut s = s.borrow_mut();
            match s.iter_mut().find(|(n, ..)| *n == name) {
                Some((_, calls, last_bytes, total)) => {
                    *calls += 1;
                    *last_bytes = bytes;
                    *total += ms;
                }
                None => s.push((name, 1, bytes, ms)),
            }
        });
    }

    /// Drain the window's writes, costliest first.
    pub(crate) fn take() -> Vec<(String, u32, usize, f32)> {
        SEEN.with(|s| {
            let mut v = std::mem::take(&mut *s.borrow_mut());
            v.sort_by(|a, b| b.3.total_cmp(&a.3));
            v
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_atomic_fs(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// Read a state file — the read twin of [`write_atomic`], and the only way a resident of the
/// folder is read, so that the page's store answers too. Absent is `ErrorKind::NotFound` on both
/// targets (every caller treats that as the first-run default).
pub(crate) fn read_to_string(path: &Path) -> std::io::Result<String> {
    #[cfg(target_arch = "wasm32")]
    return web_store::get(path);
    #[cfg(not(target_arch = "wasm32"))]
    std::fs::read_to_string(path)
}

/// [`read_to_string`] for the residents read as bytes (the addon enable state decodes its own
/// encoding, and the saved-variables files carry a Lua byte string's bytes as they are); on the
/// page the store holds text, and [`web_store::get_bytes`] undoes [`web_store::set_bytes`].
pub(crate) fn read(path: &Path) -> std::io::Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    return web_store::get_bytes(path);
    #[cfg(not(target_arch = "wasm32"))]
    std::fs::read(path)
}

/// Remove a state file (a cleared saved account name). Absent is not an error.
pub(crate) fn remove_file(path: &Path) -> std::io::Result<()> {
    #[cfg(target_arch = "wasm32")]
    return web_store::remove(path);
    #[cfg(not(target_arch = "wasm32"))]
    match std::fs::remove_file(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        r => r,
    }
}

/// The page's state folder: `window.localStorage`, one entry per file, keyed by the file's own
/// path (`/benilla-config/config.toml`, `/benilla-config/macros/account.txt`, …). Everything
/// [`write_atomic`] writes is text, and the whole folder is a few tens of KB, far under the
/// ~5 MB an origin gets — so the simplest store there is also the right one. Per browser and
/// per origin, like a `benilla-config/` beside each install; nothing crosses to the server.
#[cfg(target_arch = "wasm32")]
mod web_store {
    use std::io::{Error, ErrorKind, Result};
    use std::path::Path;

    fn key(path: &Path) -> String {
        format!("benilla:{}", path.display())
    }

    fn storage() -> Result<web_sys::Storage> {
        web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .ok_or_else(|| Error::new(ErrorKind::Unsupported, "localStorage unavailable"))
    }

    pub(super) fn get(path: &Path) -> Result<String> {
        storage()?
            .get_item(&key(path))
            .map_err(|_| Error::new(ErrorKind::Other, "localStorage read failed"))?
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::NotFound,
                    format!("{} not stored", path.display()),
                )
            })
    }

    pub(super) fn set(path: &Path, contents: &str) -> Result<()> {
        storage()?
            .set_item(&key(path), contents)
            .map_err(|_| Error::new(ErrorKind::Other, "localStorage write failed (quota?)"))
    }

    pub(super) fn remove(path: &Path) -> Result<()> {
        storage()?
            .remove_item(&key(path))
            .map_err(|_| Error::new(ErrorKind::Other, "localStorage remove failed"))
    }

    /// Store bytes ([`super::bytes_as_text`]).
    pub(super) fn set_bytes(path: &Path, contents: &[u8]) -> Result<()> {
        set(path, &super::bytes_as_text(contents))
    }

    /// The bytes [`set_bytes`] stored, or the UTF-8 of a text entry ([`super::text_as_bytes`]).
    pub(super) fn get_bytes(path: &Path) -> Result<Vec<u8>> {
        get(path).map(super::text_as_bytes)
    }
}

/// fork: marks a page-store entry whose bytes are not UTF-8: every byte after it is one `char` of
/// the same value (Latin-1), so the store's string round-trips them exactly. A NUL and a word no
/// text resident begins with, so it can never be mistaken for one.
#[cfg(any(target_arch = "wasm32", test))]
const BYTES_MARK: &str = "\u{0}benilla-bytes\u{0}";

/// fork: bytes as the page store's text. UTF-8 is stored as the text it is — exactly what a text
/// write would have stored, so every entry from before the saved-variables files became bytes
/// still reads back; anything else is [`BYTES_MARK`] and a Latin-1 spelling.
#[cfg(any(target_arch = "wasm32", test))]
fn bytes_as_text(contents: &[u8]) -> std::borrow::Cow<'_, str> {
    match std::str::from_utf8(contents) {
        Ok(text) => std::borrow::Cow::Borrowed(text),
        Err(_) => {
            let mut text = String::with_capacity(BYTES_MARK.len() + contents.len() * 2);
            text.push_str(BYTES_MARK);
            text.extend(contents.iter().map(|&b| char::from(b)));
            std::borrow::Cow::Owned(text)
        }
    }
}

/// fork: the inverse of [`bytes_as_text`].
#[cfg(any(target_arch = "wasm32", test))]
fn text_as_bytes(text: String) -> Vec<u8> {
    match text.strip_prefix(BYTES_MARK) {
        Some(latin1) => latin1.chars().map(|c| c as u32 as u8).collect(),
        None => text.into_bytes(),
    }
}

/// Test plumbing for the persistence env vars: `set_var` is process-global, so every such test
/// takes [`ENV_LOCK`] and scopes its overrides in [`EnvGuard`]s.
#[cfg(test)]
pub(crate) mod test_env {
    pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) struct EnvGuard(&'static str, Option<std::ffi::OsString>);
    impl EnvGuard {
        pub(crate) fn set(key: &'static str, value: &str) -> Self {
            let old = std::env::var_os(key);
            std::env::set_var(key, value);
            Self(key, old)
        }
        pub(crate) fn unset(key: &'static str) -> Self {
            let old = std::env::var_os(key);
            std::env::remove_var(key);
            Self(key, old)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// fork: the page store keeps a saved-variables file's bytes EXACTLY — a Lua byte string need
    /// not be UTF-8, and the store holds text — while a text file stays the text it always was.
    #[test]
    fn the_page_store_round_trips_bytes_and_leaves_text_alone() {
        let text = "SV = { name = \"Zoë\" }\n";
        assert_eq!(
            super::bytes_as_text(text.as_bytes()),
            text,
            "UTF-8 is stored as itself"
        );
        assert_eq!(super::text_as_bytes(text.to_string()), text.as_bytes());
        let raw: Vec<u8> = (0u8..=255).chain([0xff, 0xfe, 0x00, 0x80]).collect();
        let stored = super::bytes_as_text(&raw).into_owned();
        assert!(stored.starts_with(super::BYTES_MARK));
        assert_eq!(
            super::text_as_bytes(stored),
            raw,
            "every byte value comes back"
        );
    }

    use super::test_env::{EnvGuard, ENV_LOCK};
    use super::*;

    /// Every per-character file keys on this token; ASCII names map unchanged.
    #[test]
    fn distinct_names_never_share_a_token() {
        assert_ne!(file_token("Вася"), file_token("Петя"));
        assert_ne!(file_token("Zoë"), file_token("Zoé"));
        assert_eq!(file_token("Zoë"), "Zoë");
        assert_eq!(file_token("Onehunter"), "Onehunter");
        assert_eq!(file_token("Hydraxian Waterlords"), "Hydraxian_Waterlords");
        assert_eq!(
            file_token("a/b\\c:d"),
            "a_b_c_d",
            "separators never survive"
        );
        assert_eq!(
            file_token("realm-name"),
            "realm_name",
            "the key's own `-` stays unforgeable"
        );
    }

    /// The layout under the override, the one step of [`home`] whose answer a test can state.
    #[test]
    fn the_home_law_override_then_the_residents_and_hermetic_captures() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-ls-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        // 1 · the explicit override.
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.join(STATE_DIR).to_str().unwrap());
        assert_eq!(home(), Some(tmp.join(STATE_DIR)));
        assert_eq!(config_path(), Some(tmp.join("benilla-config/config.toml")));
        assert_eq!(
            macros_account_path(),
            Some(tmp.join("benilla-config/macros/account.txt"))
        );
        assert_eq!(
            saved_variables_path(),
            Some(tmp.join("benilla-config/saved-variables.lua"))
        );
        assert_eq!(
            macros_character_path("Hydraxian Waterlords", "Probeone"),
            Some(tmp.join("benilla-config/macros/Hydraxian_Waterlords-Probeone.txt"))
        );
        assert_eq!(
            macros_character_path("../evil", "a/b"),
            Some(tmp.join("benilla-config/macros/___evil-a_b.txt")),
            "no name can escape the folder"
        );
        assert_eq!(
            camera_character_path("Hydraxian Waterlords", "Probeone"),
            Some(tmp.join("benilla-config/camera/Hydraxian_Waterlords-Probeone.txt"))
        );
        assert_eq!(
            chat_character_path("Hydraxian Waterlords", "Probeone"),
            Some(tmp.join("benilla-config/chat/Hydraxian_Waterlords-Probeone.txt"))
        );
        assert_eq!(
            layout_character_path("Hydraxian Waterlords", "Probeone"),
            Some(tmp.join("benilla-config/layout/Hydraxian_Waterlords-Probeone.txt"))
        );

        assert_eq!(
            saved_account_path(),
            Some(tmp.join("benilla-config/account"))
        );
        assert_eq!(shots_path(), Some(tmp.join("benilla-config/shots.txt")));
        assert_eq!(
            screenshots_dir(),
            Some(tmp.join("benilla-config/Screenshots"))
        );

        // 0 · a capture run resolves nothing, even with an override set.
        let _c2 = EnvGuard::set("WOW_CAPTURE", "ui-options");
        assert_eq!(home(), None);
        assert_eq!(
            saved_account_path(),
            None,
            "a capture reads no saved account"
        );
        assert_eq!(shots_path(), None);
        assert_eq!(
            screenshots_dir(),
            None,
            "a capture writes no player screenshots"
        );
        assert_eq!(
            chat_character_path("Hydraxian Waterlords", "Probeone"),
            None,
            "a capture reads no player's chat look"
        );
        assert_eq!(
            layout_character_path("Hydraxian Waterlords", "Probeone"),
            None,
            "a capture reads no player's window layout"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// An unattended run keeps its state in a folder of its own, never the player's, unless
    /// `$BENILLA_HOME` pins one; a capture still resolves none.
    #[test]
    fn an_unattended_run_keeps_its_own_state_folder() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _r = EnvGuard::unset("WOW_RIG");
        let _h = EnvGuard::unset("BENILLA_HOME");
        let _u = EnvGuard::set("WOW_UNATTENDED", "1");
        let run = home_in(false).expect("an unattended run has a state folder");
        assert_eq!(run, run_home());
        assert_ne!(
            Some(&run),
            resident_home().as_ref(),
            "never the player's folder"
        );
        assert_eq!(
            home_in(false),
            Some(run.clone()),
            "one folder for the whole run"
        );
        assert_eq!(
            run_home_in(Path::new("/tmp"), 42),
            Path::new("/tmp/benilla-run-42/benilla-config")
        );

        // A rig is unattended too.
        let _u2 = EnvGuard::unset("WOW_UNATTENDED");
        let _r2 = EnvGuard::set("WOW_RIG", "tauren druid 60");
        assert_eq!(home_in(false), Some(run));

        // The override still wins, and a capture still resolves nothing.
        let _h2 = EnvGuard::set("BENILLA_HOME", "/tmp/benilla-pinned");
        assert_eq!(home_in(false), Some(PathBuf::from("/tmp/benilla-pinned")));
        let _c2 = EnvGuard::set("WOW_CAPTURE", "ui-options");
        assert_eq!(home_in(false), None);
    }

    /// A unit test that does not pin `$BENILLA_HOME` resolves no state folder, so no test can write
    /// the real one.
    #[test]
    fn a_test_without_an_override_has_no_state_folder() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _h = EnvGuard::unset("BENILLA_HOME");
        assert_eq!(home(), None);
        assert_eq!(config_path(), None);
    }

    /// A missing or wrong `$WOW_DATA` does not move or lose the state folder.
    #[test]
    fn the_state_folder_no_longer_depends_on_finding_the_install() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _h = EnvGuard::unset("BENILLA_HOME");
        let _d = EnvGuard::set("WOW_DATA", "/nonexistent/benilla-test/Data");
        let h =
            resident_home().expect("a broken install path must not cost the player their config");
        assert!(
            !h.starts_with("/nonexistent"),
            "home() still reads $WOW_DATA: {}",
            h.display()
        );
        assert!(h.ends_with(STATE_DIR), "{}", h.display());
    }

    /// A player build resolves beside the binary; a dev build to the primary checkout, even from a
    /// linked worktree (whose `.git` is a file).
    #[test]
    fn the_state_folder_lands_where_the_build_says() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _h = EnvGuard::unset("BENILLA_HOME");
        let h = resident_home().expect("the resident folder always resolves");
        assert!(h.ends_with(STATE_DIR), "{}", h.display());

        let Some(here) = benilla_formats::project_folder() else {
            // Player build: beside the binary.
            let exe_dir = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();
            assert_eq!(h, exe_dir.join(STATE_DIR));
            return;
        };

        if here.join(".git").is_file() {
            assert_ne!(
                h,
                here.join(STATE_DIR),
                "a linked worktree must not get its own settings folder — the worktrees share one"
            );
            assert!(
                h.parent().unwrap().join(".git").is_dir(),
                "resolved to {}, which is not a primary checkout",
                h.display()
            );
        } else {
            assert_eq!(h, here.join(STATE_DIR));
        }
    }

    /// A project folder that is a linked worktree keeps its state in the primary checkout, so every
    /// worktree shares one; any other folder, a crate on top of benilla's included, keeps its own.
    #[test]
    fn the_state_root_is_the_folder_or_its_primary_checkout() {
        let tmp = std::env::temp_dir().join(format!("benilla-root-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let primary = tmp.join("benilla");
        std::fs::create_dir_all(primary.join(".git/worktrees/pool-3")).unwrap();
        let slot = tmp.join("pool-3");
        std::fs::create_dir_all(&slot).unwrap();
        let gitdir = primary.join(".git/worktrees/pool-3");
        std::fs::write(slot.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        assert_eq!(shared_root(&slot), primary);
        assert_eq!(shared_root(&primary), primary);
        let hello_mod = tmp.join("hello-mod");
        std::fs::create_dir_all(hello_mod.join(".git")).unwrap();
        assert_eq!(shared_root(&hello_mod), hello_mod);
        let plain = tmp.join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        assert_eq!(shared_root(&plain), plain);
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// A launcher's recorded folder is where a dev build keeps its state and reads its probe
    /// identity; a player build resolves neither from it. Recording is process-wide, so the parent
    /// runs this same test in a child process of the test binary, which records and resolves.
    #[test]
    fn a_recorded_launcher_folder_holds_the_state_and_the_identity() {
        const CHILD: &str = "BENILLA_TEST_LAUNCHER_FOLDER";
        if let Some(dir) = std::env::var_os(CHILD) {
            let dir = PathBuf::from(dir);
            benilla_formats::set_project_folder(dir.to_str().unwrap());
            if crate::run_mode::dev_affordances() {
                assert_eq!(resident_home(), Some(dir.join(STATE_DIR)));
                assert_eq!(
                    crate::run_mode::declared_identity(),
                    Some(crate::run_mode::DeclaredIdentity {
                        user: "probe9".into(),
                        character: "Modchar".into()
                    })
                );
            } else {
                let exe_dir = std::env::current_exe().unwrap();
                assert_eq!(
                    resident_home(),
                    Some(exe_dir.parent().unwrap().join(STATE_DIR))
                );
                assert_eq!(crate::run_mode::declared_identity(), None);
            }
            return;
        }
        let tmp = std::env::temp_dir().join(format!("benilla-launcher-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join(".probe-identity"),
            "WOW_USER=probe9\nWOW_PASS=unused\nWOW_CHAR=Modchar\n",
        )
        .unwrap();
        let name =
            "local_state::tests::a_recorded_launcher_folder_holds_the_state_and_the_identity";
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env(CHILD, &tmp)
            .env_remove("BENILLA_HOME")
            .env_remove("WOW_CAPTURE")
            .output()
            .unwrap();
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.status.success() && report.contains("1 passed"),
            "the child run:\n{report}"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn write_atomic_creates_dirs_and_replaces_whole_files() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-wa-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let path = tmp.join("nested/config.toml");
        write_atomic(&path, "a = \"1\"\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a = \"1\"\n");
        write_atomic(&path, "a = \"2\"\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a = \"2\"\n");
        assert!(!path.with_extension("tmp").exists(), "tmp renamed away");
        std::fs::remove_dir_all(&tmp).ok();
    }
}
