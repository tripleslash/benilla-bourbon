//! fork: desktop — the self-update. Before the client starts, the player build reads the site's
//! [`MANIFEST`] (`scripts/split-parts.sh` writes it) and, when it names a build newer than this
//! one, downloads the parts beside the executable under a progress window, checks the byte count
//! and SHA-1, renames the running executable aside, moves the new one into its place and launches
//! it with the same arguments. Windows lets a running executable be renamed, never deleted, so the
//! renamed copy is removed by the new process once the old one has exited.
//!
//! Anything short of a whole, verified copy (no manifest, an unstamped or not newer build, a
//! missing part, a short or corrupt download, a folder it cannot write, the window closed, a new
//! file that will not start) leaves the executable as it was, and the current client starts.
//!
//! "Newer" is the build's commit time: `BENILLA_DESKTOP_VERSION`, which `build.rs` stamps, against
//! the manifest's `version`, which `scripts/build-desktop.sh` records beside the exe. A build of a
//! later commit than the hosted one is never replaced by it.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sha1::{Digest, Sha1};

#[cfg(windows)]
mod window;

/// The site's manifest of the executable's parts:
/// `{"bytes":N,"version":T,"sha1":"…","parts":[…]}`.
pub const MANIFEST: &str = "wow-reforged.exe.parts.json";

/// Set on the relaunched process, which then checks for no update: a manifest that names a version
/// its build does not carry cannot make it update in a loop.
const RELAUNCHED: &str = "BENILLA_UPDATED";

/// No client is this large; a manifest that says otherwise is not followed.
const MAX_BYTES: u64 = 1 << 30;

/// The splitter's parts are 24 MiB; a body larger than this is not a part.
const PART_MAX: u64 = 32 << 20;

/// What [`run`] leaves the caller to do.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Start the client in this process.
    Current,
    /// A newer executable is running in its place: exit.
    Relaunched,
}

/// The parsed [`MANIFEST`].
#[derive(Debug, PartialEq, Eq)]
pub struct Manifest {
    /// The whole executable's length.
    pub bytes: u64,
    /// The part names, in order, relative to the manifest.
    pub parts: Vec<String>,
    /// The build's commit time; absent for a build staged without its stamp.
    pub version: Option<u64>,
    /// The whole executable's SHA-1.
    pub sha1: Option<[u8; 20]>,
}

impl Manifest {
    /// Parse the splitter's flat JSON object: numbers, strings and string arrays, unknown keys
    /// skipped. Anything else is `None`, which updates nothing.
    pub fn parse(text: &str) -> Option<Manifest> {
        let mut c = Cursor {
            s: text.as_bytes(),
            i: 0,
        };
        let (mut bytes, mut parts, mut version, mut sha1) = (None, None, None, None);
        c.eat(b'{')?;
        if !c.next_is(b'}') {
            loop {
                let key = c.string()?;
                c.eat(b':')?;
                match key {
                    "bytes" => bytes = Some(c.number()?),
                    "version" => version = Some(c.number()?),
                    "sha1" => sha1 = Some(hex20(c.string()?)?),
                    "parts" => parts = Some(c.strings()?),
                    _ => c.skip_value()?,
                }
                if !c.next_is(b',') {
                    break;
                }
                c.i += 1;
            }
        }
        c.eat(b'}')?;
        c.ws();
        (c.i == c.s.len()).then_some(())?;
        Some(Manifest {
            bytes: bytes?,
            parts: parts?,
            version,
            sha1,
        })
    }

    /// Whether this manifest is an update for a build stamped `current`: a stamped build strictly
    /// newer, with a hash to check it against, a sane length and plain part names.
    pub fn offers(&self, current: Option<u64>) -> bool {
        matches!((self.version, current), (Some(v), Some(c)) if v > c)
            && self.sha1.is_some()
            && (1..=MAX_BYTES).contains(&self.bytes)
            && !self.parts.is_empty()
            && self.parts.iter().all(|p| plain(p))
    }
}

/// Update this executable from `origin` when it serves a build newer than `current` (this build's
/// commit time, `None` when unstamped). Every failure is [`Outcome::Current`].
pub fn run(origin: &str, current: Option<u64>) -> Outcome {
    let Ok(exe) = std::env::current_exe() else {
        return Outcome::Current;
    };
    sweep(&exe);
    if std::env::var_os(RELAUNCHED).is_some() {
        return Outcome::Current;
    }
    match update(origin.trim_end_matches('/'), current, &exe) {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("update: {e}; starting this build");
            Outcome::Current
        }
    }
}

/// Where the worker is: [`RUNNING`] until it ends in [`DONE`] or [`FAILED`], or the window is
/// closed first ([`CANCELLED`]). Whichever side moves it off [`RUNNING`] decides who removes the
/// downloaded file.
pub(crate) const RUNNING: u8 = 0;
pub(crate) const DONE: u8 = 1;
pub(crate) const FAILED: u8 = 2;
pub(crate) const CANCELLED: u8 = 3;

/// What the download thread and the progress window share.
#[derive(Default)]
pub(crate) struct Shared {
    /// Bytes written so far.
    pub(crate) done: AtomicU64,
    pub(crate) state: AtomicU8,
    /// fork: dlss (FORK.md) — the window's heading; empty is the client update's. Only the
    /// Win32 progress window reads it.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) heading: &'static str,
}

fn update(origin: &str, current: Option<u64>, exe: &Path) -> io::Result<Outcome> {
    let manifest = fetch_manifest(origin)?;
    if !manifest.offers(current) {
        return Ok(Outcome::Current);
    }
    let (Some(version), Some(sha1)) = (manifest.version, manifest.sha1) else {
        return Ok(Outcome::Current);
    };
    let staged = sibling(exe, "download");
    // A folder that cannot be written ends it here, before a window or a byte of download.
    let file = File::create(&staged)?;
    let shared = Arc::new(Shared::default());
    let total = manifest.bytes;
    let worker = {
        let (shared, staged, origin) = (shared.clone(), staged.clone(), origin.to_string());
        std::thread::spawn(move || {
            let result = download(file, &origin, &manifest, &sha1, &shared, 0);
            if let Err(e) = &result {
                eprintln!("update: {e}");
            }
            let end = if result.is_ok() { DONE } else { FAILED };
            let moved =
                shared
                    .state
                    .compare_exchange(RUNNING, end, Ordering::AcqRel, Ordering::Acquire);
            if moved.is_err() || result.is_err() {
                let _ = fs::remove_file(&staged);
            }
        })
    };

    if wait(&shared, total) {
        // Closed: the worker removes its file when it next looks, unless it already finished.
        if shared
            .state
            .compare_exchange(RUNNING, CANCELLED, Ordering::AcqRel, Ordering::Acquire)
            == Err(DONE)
        {
            let _ = fs::remove_file(&staged);
        }
        return Ok(Outcome::Current);
    }
    let _ = worker.join();
    if shared.state.load(Ordering::Acquire) != DONE {
        return Ok(Outcome::Current);
    }
    swap(exe, &staged, version)
}

/// The progress window until the worker ends; `true` when the player closed it first. With no
/// window to show, the wait is silent.
pub(crate) fn wait(shared: &Arc<Shared>, total: u64) -> bool {
    #[cfg(windows)]
    if let Some(closed) = window::run(shared.clone(), total) {
        return closed;
    }
    let _ = total;
    while shared.state.load(Ordering::Acquire) == RUNNING {
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// A blocking HTTPS agent whose every call, body included, ends within `limit`.
pub(crate) fn agent(limit: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(limit))
        .timeout_connect(Some(Duration::from_secs(5)))
        .http_status_as_error(false)
        .user_agent(concat!("benilla-desktop/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent()
}

/// The site's manifest; asked with a short limit, since the client waits on it to start.
fn fetch_manifest(origin: &str) -> io::Result<Manifest> {
    let url = format!("{origin}/{MANIFEST}");
    let mut resp = agent(Duration::from_secs(6))
        .get(&url)
        .call()
        .map_err(|e| io::Error::other(format!("GET {url}: {e}")))?;
    let status = resp.status().as_u16();
    if status != 200 {
        return Err(io::Error::other(format!("GET {url}: HTTP {status}")));
    }
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| io::Error::other(format!("GET {url}: {e}")))?;
    Manifest::parse(&text).ok_or_else(|| io::Error::other(format!("{url}: not a parts manifest")))
}

/// Fetch the parts in order into `file`, counting and hashing as they arrive, and stop as soon as
/// the window is closed. Ok only for exactly the manifest's length with the manifest's hash.
/// `base`: bytes already counted in `shared` by earlier files of the same window (fork: dlss).
pub(crate) fn download(
    mut file: File,
    origin: &str,
    manifest: &Manifest,
    sha1: &[u8; 20],
    shared: &Shared,
    base: u64,
) -> io::Result<()> {
    let agent = agent(Duration::from_secs(600));
    let mut hash = Sha1::new();
    let mut done = 0u64;
    let mut buf = vec![0u8; 64 << 10];
    for part in &manifest.parts {
        let url = format!("{origin}/{part}");
        let mut resp = agent
            .get(&url)
            .call()
            .map_err(|e| io::Error::other(format!("GET {url}: {e}")))?;
        let status = resp.status().as_u16();
        if status != 200 {
            return Err(io::Error::other(format!("GET {url}: HTTP {status}")));
        }
        let mut body = resp.body_mut().with_config().limit(PART_MAX).reader();
        loop {
            if shared.state.load(Ordering::Acquire) != RUNNING {
                return Err(io::Error::other("cancelled"));
            }
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            done += n as u64;
            if done > manifest.bytes {
                return Err(io::Error::other(
                    "the parts are longer than the manifest says",
                ));
            }
            hash.update(&buf[..n]);
            file.write_all(&buf[..n])?;
            shared.done.store(base + done, Ordering::Relaxed);
        }
    }
    if done != manifest.bytes {
        return Err(io::Error::other(format!(
            "{done} of {} bytes: a part is missing or short",
            manifest.bytes
        )));
    }
    if hash.finalize().as_slice() != sha1.as_slice() {
        return Err(io::Error::other(
            "the download does not match the manifest's sha1",
        ));
    }
    file.sync_all()
}

/// Put the verified `staged` file in the running executable's place and start it. Each step that
/// fails undoes the ones before it, so the executable is always one whole build.
fn swap(exe: &Path, staged: &Path, version: u64) -> io::Result<Outcome> {
    let old = sibling(exe, "old");
    let _ = fs::remove_file(&old);
    if let Err(e) = fs::rename(exe, &old) {
        let _ = fs::remove_file(staged);
        return Err(e);
    }
    if let Err(e) = fs::rename(staged, exe) {
        let _ = fs::rename(&old, exe);
        let _ = fs::remove_file(staged);
        return Err(e);
    }
    match Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env(RELAUNCHED, version.to_string())
        .spawn()
    {
        Ok(_) => Ok(Outcome::Relaunched),
        Err(e) => {
            let _ = fs::rename(exe, staged);
            let _ = fs::rename(&old, exe);
            let _ = fs::remove_file(staged);
            Err(e)
        }
    }
}

/// Remove the executable a previous update renamed aside. The process that ran it may still be
/// exiting, which holds the file, so a thread retries for a few seconds.
fn sweep(exe: &Path) {
    let old = sibling(exe, "old");
    if !old.exists() {
        return;
    }
    std::thread::spawn(move || {
        for _ in 0..40 {
            if fs::remove_file(&old).is_ok() || !old.exists() {
                return;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    });
}

/// `<exe>.<ext>`, beside it: the same volume, so a rename never copies.
fn sibling(exe: &Path, ext: &str) -> PathBuf {
    let mut name = OsString::from(exe.as_os_str());
    name.push(".");
    name.push(ext);
    PathBuf::from(name)
}

/// A bare file name: no separator, no `..`, nothing a URL or path would read as structure.
pub(crate) fn plain(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

fn hex20(s: &str) -> Option<[u8; 20]> {
    if s.len() != 40 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 20];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// A reader of the manifest's JSON subset.
struct Cursor<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn ws(&mut self) {
        while self.s.get(self.i).is_some_and(u8::is_ascii_whitespace) {
            self.i += 1;
        }
    }

    fn next_is(&mut self, c: u8) -> bool {
        self.ws();
        self.s.get(self.i) == Some(&c)
    }

    fn eat(&mut self, c: u8) -> Option<()> {
        self.next_is(c).then(|| self.i += 1)
    }

    /// A string without escapes, which the splitter never writes.
    fn string(&mut self) -> Option<&'a str> {
        self.eat(b'"')?;
        let s: &'a [u8] = self.s;
        let start = self.i;
        let len = s[start..].iter().position(|&b| b == b'"')?;
        let raw = &s[start..start + len];
        if raw.contains(&b'\\') {
            return None;
        }
        self.i = start + len + 1;
        std::str::from_utf8(raw).ok()
    }

    fn number(&mut self) -> Option<u64> {
        self.ws();
        let start = self.i;
        while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
            self.i += 1;
        }
        std::str::from_utf8(&self.s[start..self.i])
            .ok()?
            .parse()
            .ok()
    }

    fn strings(&mut self) -> Option<Vec<String>> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        if self.eat(b']').is_some() {
            return Some(out);
        }
        loop {
            out.push(self.string()?.to_string());
            if self.eat(b',').is_none() {
                break;
            }
        }
        self.eat(b']')?;
        Some(out)
    }

    fn skip_value(&mut self) -> Option<()> {
        if self.next_is(b'"') {
            self.string().map(drop)
        } else if self.next_is(b'[') {
            self.strings().map(drop)
        } else {
            self.number().map(drop)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "00112233445566778899aabbccddeeff00112233";

    fn written() -> String {
        format!(
            "{{\"bytes\":50331649,\"version\":1762500000,\"sha1\":\"{SHA}\",\"parts\":\
             [\"wow-reforged.exe.part00\",\"wow-reforged.exe.part01\",\"wow-reforged.exe.part02\"]}}\n"
        )
    }

    /// What `scripts/split-parts.sh` writes for the exe reads back whole.
    #[test]
    fn the_splitters_manifest_reads_back() {
        let m = Manifest::parse(&written()).unwrap();
        assert_eq!(m.bytes, 50_331_649);
        assert_eq!(m.version, Some(1_762_500_000));
        assert_eq!(m.sha1.unwrap()[..3], [0x00, 0x11, 0x22]);
        assert_eq!(m.sha1.unwrap()[19], 0x33);
        assert_eq!(m.parts.len(), 3);
        assert_eq!(m.parts[2], "wow-reforged.exe.part02");
    }

    /// The wasm module's manifest, with neither stamp nor hash, parses and offers nothing; so
    /// does an exe staged before builds were stamped.
    #[test]
    fn a_manifest_without_version_or_hash_offers_nothing() {
        let m = Manifest::parse(r#"{"bytes":10,"parts":["wow.part00.wasm"]}"#).unwrap();
        assert_eq!((m.version, m.sha1), (None, None));
        assert!(!m.offers(Some(1)));
        let unhashed = Manifest::parse(r#"{"bytes":10,"version":9,"parts":["a"]}"#).unwrap();
        assert!(!unhashed.offers(Some(1)));
    }

    /// Only a strictly newer stamped build is an update: never the same commit, an older one, or
    /// for a build that carries no stamp.
    #[test]
    fn only_a_newer_stamped_build_is_offered() {
        let m = Manifest::parse(&written()).unwrap();
        assert!(m.offers(Some(1_762_499_999)));
        assert!(!m.offers(Some(1_762_500_000)));
        assert!(!m.offers(Some(1_762_500_001)));
        assert!(!m.offers(None));
    }

    /// A part name that would reach outside the site's root, or a length no client has, is not
    /// followed.
    #[test]
    fn unsafe_names_and_lengths_offer_nothing() {
        let manifest = |bytes: u64, name: &str| {
            format!(r#"{{"bytes":{bytes},"version":9,"sha1":"{SHA}","parts":["{name}"]}}"#)
        };
        for name in ["../x", "a/b", "a\\b", ".hidden", "", "x?y=1"] {
            let m = Manifest::parse(&manifest(10, name));
            assert!(m.is_none_or(|m| !m.offers(Some(1))), "{name}");
        }
        for bytes in [0, MAX_BYTES + 1] {
            let m = Manifest::parse(&manifest(bytes, "a")).unwrap();
            assert!(!m.offers(Some(1)), "{bytes}");
        }
    }

    /// A truncated, garbled or trailing-junk manifest, or a bad hash, is no manifest.
    #[test]
    fn a_damaged_manifest_is_none() {
        let text = written();
        for cut in [1, text.len() / 2, text.len() - 3] {
            assert!(Manifest::parse(&text[..cut]).is_none(), "{cut}");
        }
        assert!(Manifest::parse(&format!("{text}x")).is_none());
        assert!(Manifest::parse(r#"{"bytes":1,"sha1":"00zz","parts":["a"]}"#).is_none());
        assert!(Manifest::parse(r#"{"parts":["a"]}"#).is_none());
        // Unknown keys of the known shapes are skipped.
        let m = Manifest::parse(r#"{ "bytes" : 3 , "x" : "y", "z":[], "parts" : [ "a" ] }"#);
        assert_eq!(m.unwrap().parts, ["a"]);
    }

    #[test]
    fn the_staged_and_old_files_sit_beside_the_executable() {
        let exe = Path::new("C:/wow/wow-reforged.exe");
        assert_eq!(
            sibling(exe, "download"),
            Path::new("C:/wow/wow-reforged.exe.download")
        );
        assert_eq!(
            sibling(exe, "old"),
            Path::new("C:/wow/wow-reforged.exe.old")
        );
    }
}
