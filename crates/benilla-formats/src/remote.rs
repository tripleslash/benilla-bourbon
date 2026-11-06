//! fork: desktop — where a client with no install reads its game data from: the origin the web
//! build is hosted to (`https://wow.example.com`), its containers cached under one local folder.
//!
//! The switch is [`origin`]: `$BENILLA_DATA_URL` when set (set and empty: off), else what the
//! launcher named ([`set_origin`]), else off, which leaves every install-reading path as it was.
//! With an origin, [`crate::wow_data`] answers [`data_dir`] and [`crate::Chain::open`] reads the
//! container there ([`crate::netpack`]) in place of an MPQ chain.
//!
//! The cache ([`cache_root`]) holds one folder per container segment the page serves beside its
//! own files — `data`, `data-hd` — plus `addons/` (the bundled addons, [`sync_addons`]) and
//! `config.json` (the site's gateway and realmlist, [`site_config`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use anyhow::Result;

use crate::netpack::{Http, NetPackStore};

/// The base corpus's segment: `{origin}/data`.
pub const DATA_SEGMENT: &str = "data";

static ORIGIN: OnceLock<String> = OnceLock::new();
static CACHE_ROOT: OnceLock<PathBuf> = OnceLock::new();
/// Every container opened this process, by its cache folder: a chain opened twice shares one index
/// and one download queue, and the origin is asked for the version pointer once.
static STORES: Mutex<Option<HashMap<PathBuf, Arc<NetPackStore>>>> = Mutex::new(None);

/// Name the origin this process reads from when `$BENILLA_DATA_URL` does not (the desktop
/// launcher's). The first call wins.
pub fn set_origin(url: &str) {
    let _ = ORIGIN.set(url.trim_end_matches('/').to_string());
}

/// Name the folder fetched containers are cached in (the app's, under its state folder). The first
/// call wins; unset, the system temp folder's `benilla-packs`.
pub fn set_cache_root(dir: PathBuf) {
    let _ = CACHE_ROOT.set(dir);
}

/// The origin, or `None` when this process reads an install.
pub fn origin() -> Option<String> {
    match std::env::var("BENILLA_DATA_URL") {
        Ok(url) if url.trim().is_empty() => None,
        Ok(url) => Some(url.trim().trim_end_matches('/').to_string()),
        Err(_) => ORIGIN.get().cloned(),
    }
}

/// The folder every fetched container, the bundled addons and the site config are kept in.
pub fn cache_root() -> PathBuf {
    CACHE_ROOT
        .get()
        .cloned()
        .unwrap_or_else(|| std::env::temp_dir().join("benilla-packs"))
}

/// The base container's cache folder, when there is an origin: what [`crate::wow_data`] answers.
pub fn data_dir() -> Option<PathBuf> {
    origin().map(|_| cache_root().join(DATA_SEGMENT))
}

/// The container a chain opened at `path` reads, or `None` for a path that names none (an MPQ
/// install): the origin's segment when `path` is its cache folder, else a container built on this
/// machine when `path` holds an `index.bpi`.
pub fn store_at(path: &Path) -> Option<Result<Arc<NetPackStore>>> {
    let segment = origin().and_then(|o| {
        let root = cache_root();
        let seg = path.strip_prefix(&root).ok()?.to_str()?.to_string();
        (!seg.is_empty() && !seg.contains(['/', '\\'])).then_some((o, seg))
    });
    if segment.is_none() && !path.join("index.bpi").is_file() {
        return None;
    }
    let mut stores = STORES.lock().unwrap_or_else(PoisonError::into_inner);
    let stores = stores.get_or_insert_with(HashMap::new);
    if let Some(store) = stores.get(path) {
        return Some(Ok(Arc::clone(store)));
    }
    let opened = match &segment {
        Some((origin, seg)) => {
            NetPackStore::open(path, Some(Box::new(Http::new(&format!("{origin}/{seg}")))))
        }
        None => NetPackStore::open(path, None),
    };
    Some(opened.map(|store| {
        let store = Arc::new(store);
        if segment.is_some() {
            store.warm_boot_set();
        }
        stores.insert(path.to_path_buf(), Arc::clone(&store));
        store
    }))
}

/// The origin's container `segment` (`data-hd`), opened through [`store_at`].
pub fn segment(segment: &str) -> Option<Result<Arc<NetPackStore>>> {
    origin()?;
    store_at(&cache_root().join(segment))
}

/// How long nothing may have been fetched for a read or a warm before the background fill takes
/// the line, so a burst's next request does not queue behind a fill download.
const FILL_QUIET: std::time::Duration = std::time::Duration::from_secs(1);
/// How often the fill looks again once every container it fills is complete (or unreachable).
const FILL_DONE_POLL: std::time::Duration = std::time::Duration::from_secs(60);

/// Start the background fill for this process, once: one thread that fetches the packs nothing
/// has asked for yet while nothing else is being fetched ([`crate::netpack`]'s header), so a
/// client idling at the login screen keeps filling its cache. Off with
/// `$BENILLA_BACKGROUND_FETCH=0`; nothing to do without an origin.
pub fn start_background_fill() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    if origin().is_none() || !background_fill_enabled() {
        return;
    }
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("pack-fill".into())
            .spawn(fill_loop);
    });
}

/// `$BENILLA_BACKGROUND_FETCH`: on unless it reads `0`, `off` or `false`.
fn background_fill_enabled() -> bool {
    match std::env::var("BENILLA_BACKGROUND_FETCH") {
        Ok(v) => crate::layers::layer_switch(&v) != Some(false),
        Err(_) => true,
    }
}

/// The containers the fill completes, in order: the base corpus, the updated character models
/// while that setting is on, the Turtle WoW layer (a realm list may offer a 1.18.1 realm), and its
/// reforged art while that setting is on. Asked afresh on every pass, so a setting changed after
/// boot is followed.
fn fill_segments() -> Vec<&'static str> {
    let mut out = vec![DATA_SEGMENT];
    if crate::layers::hd_character_models() {
        out.push(crate::layers::HD_DATA_SEGMENT);
    }
    out.push(crate::layers::TURTLE_DATA_SEGMENT);
    if crate::layers::reforged_models() {
        out.push(crate::layers::REFORGED_DATA_SEGMENT);
    }
    out
}

/// Whether a read or a warm is fetching, or a warm waits, in any open container.
fn foreground_active() -> bool {
    if crate::netpack::foreground_busy() {
        return true;
    }
    let stores: Vec<Arc<NetPackStore>> = STORES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .map(|m| m.values().cloned().collect())
        .unwrap_or_default();
    stores.iter().any(|s| s.warm_pending())
}

/// The fill thread: wait for [`FILL_QUIET`], fetch the next pack the first incomplete container
/// lacks, and look again; with every container complete, look again after [`FILL_DONE_POLL`].
fn fill_loop() {
    let mut quiet_since: Option<std::time::Instant> = None;
    loop {
        if foreground_active() {
            quiet_since = None;
            std::thread::sleep(std::time::Duration::from_millis(250));
            continue;
        }
        let since = *quiet_since.get_or_insert_with(std::time::Instant::now);
        if since.elapsed() < FILL_QUIET {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }
        let mut fetched = false;
        for name in fill_segments() {
            // A container that will not open (offline, or not hosted) is passed this round.
            let Some(Ok(store)) = segment(name) else {
                continue;
            };
            if let Some(pack_id) = store.next_unfilled() {
                if let Err(e) = store.fill(pack_id) {
                    eprintln!("benilla: background fill of {name} pack {pack_id}: {e}");
                }
                fetched = true;
                break;
            }
        }
        if !fetched {
            std::thread::sleep(FILL_DONE_POLL);
        }
    }
}

/// One of the install's loose top-level files the site serves beside a container, at
/// `{origin}/{segment}/loose/<name lowercased>` (`eula.html`, `tos.html`, `Credits.html`): the
/// page's `read_loose_web`. Fetched once into `<segment>/loose/` and read from there after; a miss
/// is remembered for the process. Only a bare name is asked.
pub fn loose(segment: &str, name: &str) -> Option<Vec<u8>> {
    let origin = origin()?;
    if name.is_empty() || name.contains(['\\', '/', ':']) || name.contains("..") {
        return None;
    }
    let name = name.to_ascii_lowercase();
    let kept = cache_root().join(segment).join("loose").join(&name);
    if let Ok(bytes) = std::fs::read(&kept) {
        return Some(bytes);
    }
    static MISSES: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);
    let key = format!("{segment}/{name}");
    if MISSES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(Default::default)
        .contains(&key)
    {
        return None;
    }
    match addons::get(&format!("{origin}/{segment}/loose/{name}"), None) {
        Ok(addons::Got::Body(bytes, _)) => {
            if let Some(dir) = kept.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&kept, &bytes);
            Some(bytes)
        }
        _ => {
            MISSES
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert_with(Default::default)
                .insert(key);
            None
        }
    }
}

/// A live layer the whole process reads through: mounted when a 1.18.1 realm is entered, as the
/// page mounts its own (`crate::layers`' `live`).
struct Slot(std::sync::RwLock<Option<Arc<NetPackStore>>>);

impl Slot {
    fn get(&self) -> Option<Arc<NetPackStore>> {
        self.0.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn mount(&self, segment_name: &str) -> Result<()> {
        if self.get().is_some() {
            return Ok(());
        }
        let store = segment(segment_name)
            .ok_or_else(|| anyhow::anyhow!("no data origin to read {segment_name} from"))??;
        *self.0.write().unwrap_or_else(PoisonError::into_inner) = Some(store);
        crate::layers::bump_generation();
        Ok(())
    }

    fn close(&self) {
        if self
            .0
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .is_some()
        {
            crate::layers::bump_generation();
        }
    }
}

static TURTLE: Slot = Slot(std::sync::RwLock::new(None));
static REFORGED: Slot = Slot(std::sync::RwLock::new(None));

/// Mount the origin's Turtle WoW 1.18.1 layer (`data-turtle`), for a 1.18.1 realm being entered.
pub fn mount_turtle() -> Result<()> {
    TURTLE.mount(crate::layers::TURTLE_DATA_SEGMENT)
}

/// Unmount the Turtle layer, for a 1.12.1 realm.
pub fn close_turtle() {
    TURTLE.close();
}

/// The mounted Turtle layer.
pub fn turtle() -> Option<Arc<NetPackStore>> {
    TURTLE.get()
}

/// Mount the origin's reforged art (`data-reforged`) over the Turtle layer.
pub fn mount_reforged() -> Result<()> {
    REFORGED.mount(crate::layers::REFORGED_DATA_SEGMENT)
}

/// Unmount the reforged art.
pub fn close_reforged() {
    REFORGED.close();
}

/// The mounted reforged art layer.
pub fn reforged() -> Option<Arc<NetPackStore>> {
    REFORGED.get()
}

/// The bundled addons' folder in the cache: `addons/<Folder>/<file>`.
pub fn addons_dir() -> Option<PathBuf> {
    origin().map(|_| cache_root().join(crate::bundled_addons::WEB_SEGMENT))
}

/// Bring the cached bundled addons up to the origin's `/addons/`: its `index.txt` names every file;
/// each is fetched when it is missing or its `ETag` changed, and a cached file the index no longer
/// names is deleted. Offline, the cache stays as it is. Returns the files fetched.
pub fn sync_addons() -> Result<usize> {
    let (Some(origin), Some(dir)) = (origin(), addons_dir()) else {
        return Ok(0);
    };
    addons::sync(
        &format!("{origin}/{}", crate::bundled_addons::WEB_SEGMENT),
        &dir,
    )
}

/// The site's `config.json` (`{"ws": …, "realmlist": …}`), fetched fresh and kept for an offline
/// start, or the kept copy when the origin cannot be reached.
pub fn site_config() -> Option<String> {
    let origin = origin()?;
    let kept = cache_root().join("config.json");
    match addons::get(&format!("{origin}/config.json"), None) {
        Ok(addons::Got::Body(body, _)) => {
            let _ = std::fs::create_dir_all(cache_root());
            let _ = std::fs::write(&kept, &body);
            String::from_utf8(body).ok()
        }
        _ => std::fs::read_to_string(kept).ok(),
    }
}

/// `"key":"value"` out of a flat JSON object — `config.json`'s two strings, without a JSON
/// dependency for them.
pub fn json_string(json: &str, key: &str) -> Option<String> {
    let quoted = format!("\"{key}\"");
    let at = json.find(&quoted)?;
    let rest = json[at + quoted.len()..]
        .trim_start()
        .strip_prefix(':')?
        .trim_start();
    let rest = rest.strip_prefix('"')?;
    Some(rest[..rest.find('"')?].to_string())
}

mod addons {
    use std::collections::{HashMap, HashSet};
    use std::io::Read;
    use std::path::Path;
    use std::sync::Mutex;
    use std::time::Duration;

    use anyhow::{anyhow, Context, Result};

    /// The kept `ETag` of every fetched file, one `etag\tpath` per line.
    const TAGS: &str = ".etags";
    const THREADS: usize = 8;

    pub(super) enum Got {
        Body(Vec<u8>, Option<String>),
        NotModified,
    }

    fn agent() -> ureq::Agent {
        ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_recv_body(Some(Duration::from_secs(120)))
            .http_status_as_error(false)
            .build()
            .new_agent()
    }

    /// `GET url`, conditional on `etag` when there is one.
    pub(super) fn get(url: &str, etag: Option<&str>) -> Result<Got> {
        get_with(&agent(), url, etag)
    }

    fn get_with(agent: &ureq::Agent, url: &str, etag: Option<&str>) -> Result<Got> {
        let mut req = agent.get(url);
        if let Some(tag) = etag {
            req = req.header("If-None-Match", tag);
        }
        let mut resp = req.call().map_err(|e| anyhow!("GET {url}: {e}"))?;
        match resp.status().as_u16() {
            304 => Ok(Got::NotModified),
            200 => {
                let tag = resp
                    .headers()
                    .get("etag")
                    .and_then(|v| v.to_str().ok())
                    .map(String::from);
                let mut body = Vec::new();
                resp.body_mut()
                    .with_config()
                    .limit(64 * 1024 * 1024)
                    .reader()
                    .read_to_end(&mut body)
                    .with_context(|| format!("GET {url}"))?;
                Ok(Got::Body(body, tag))
            }
            s => Err(anyhow!("GET {url}: HTTP {s}")),
        }
    }

    /// Encode one path segment for a URL.
    fn encode(seg: &str) -> String {
        let mut out = String::with_capacity(seg.len());
        for b in seg.bytes() {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
                out.push(b as char);
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    }

    /// Whether `rel` is a plain relative path that stays under the folder it is joined to.
    fn safe(rel: &str) -> bool {
        !rel.is_empty()
            && !rel.starts_with(['/', '\\'])
            && rel
                .split(['/', '\\'])
                .all(|c| !c.is_empty() && c != "." && c != ".." && !c.contains(':'))
    }

    pub(super) fn sync(base: &str, dir: &Path) -> Result<usize> {
        let agent = agent();
        let listing = match get_with(&agent, &format!("{base}/index.txt"), None) {
            Ok(Got::Body(body, _)) => String::from_utf8_lossy(&body).into_owned(),
            Ok(Got::NotModified) => return Ok(0),
            // Offline: the cache stands.
            Err(_) if dir.is_dir() => return Ok(0),
            Err(e) => return Err(e),
        };
        let files: Vec<String> = listing
            .lines()
            .map(str::trim)
            .filter(|l| safe(l))
            .map(String::from)
            .collect();
        std::fs::create_dir_all(dir)?;
        let kept: HashMap<String, String> = std::fs::read_to_string(dir.join(TAGS))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.split_once('\t'))
            .map(|(tag, path)| (path.to_string(), tag.to_string()))
            .collect();

        let next = Mutex::new(files.iter());
        let tags = Mutex::new(HashMap::new());
        let fetched = Mutex::new(0usize);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| loop {
                    let Some(rel) = next.lock().unwrap_or_else(|e| e.into_inner()).next() else {
                        return;
                    };
                    let dest = dir.join(rel);
                    let tag = kept.get(rel).filter(|_| dest.is_file()).map(String::as_str);
                    let url = format!(
                        "{base}/{}",
                        rel.split('/').map(encode).collect::<Vec<_>>().join("/")
                    );
                    match get_with(&agent, &url, tag) {
                        Ok(Got::NotModified) => {
                            if let Some(t) = tag {
                                tags.lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .insert(rel.clone(), t.to_string());
                            }
                        }
                        Ok(Got::Body(body, new_tag)) => {
                            if let Some(parent) = dest.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            if std::fs::write(&dest, body).is_ok() {
                                *fetched.lock().unwrap_or_else(|e| e.into_inner()) += 1;
                                if let Some(t) = new_tag {
                                    tags.lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .insert(rel.clone(), t);
                                }
                            }
                        }
                        Err(e) => eprintln!("benilla: bundled addon file {rel}: {e:#}"),
                    }
                });
            }
        });

        // Files the listing no longer names go, with folders they leave empty.
        let wanted: HashSet<String> = files.iter().map(|f| f.to_ascii_lowercase()).collect();
        sweep(dir, dir, &wanted);

        let tags = tags.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut lines: Vec<String> = tags.iter().map(|(p, t)| format!("{t}\t{p}")).collect();
        lines.sort();
        std::fs::write(dir.join(TAGS), lines.join("\n"))?;
        Ok(fetched.into_inner().unwrap_or_else(|e| e.into_inner()))
    }

    fn sweep(root: &Path, at: &Path, wanted: &HashSet<String>) {
        let Ok(entries) = std::fs::read_dir(at) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                sweep(root, &path, wanted);
                let _ = std::fs::remove_dir(&path); // only when empty
                continue;
            }
            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            let rel = rel
                .to_string_lossy()
                .replace('\\', "/")
                .to_ascii_lowercase();
            if rel != TAGS && !wanted.contains(&rel) {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_listing_line_never_leaves_the_folder() {
            assert!(safe("pfQuest/db/enUS/items.lua"));
            assert!(!safe("../evil.lua"));
            assert!(!safe("pfQuest/../../x"));
            assert!(!safe("/etc/passwd"));
            assert!(!safe("C:/Windows/x"));
            assert!(!safe(""));
        }

        #[test]
        fn a_segment_is_url_encoded() {
            assert_eq!(encode("pfQuest-turtle"), "pfQuest-turtle");
            assert_eq!(encode("a b+c"), "a%20b%2Bc");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_strings_read_out_of_the_deploy_json() {
        let json = r#"{ "ws": "wss://realm.example.com", "realmlist": "realm.example.com" }"#;
        assert_eq!(
            json_string(json, "ws").as_deref(),
            Some("wss://realm.example.com")
        );
        assert_eq!(
            json_string(json, "realmlist").as_deref(),
            Some("realm.example.com")
        );
        assert_eq!(json_string(json, "missing"), None);
    }
}
