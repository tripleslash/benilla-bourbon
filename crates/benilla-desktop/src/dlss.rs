//! fork: dlss (FORK.md) — NVIDIA's DLSS runtime for the standalone `wow-reforged.exe`, which
//! ships as one file. Before the client starts, each runtime file the client loads is checked
//! against the site's `dlss/<file>.parts.json` (`scripts/stage-desktop.sh` writes it, in
//! `scripts/split-parts.sh`'s format with the file's SHA-1) and fetched into `wow-data/dlss/` when
//! it is missing or differs, under the self-update's progress window. The client reads the folder
//! at boot (`benilla_world::dlss::DlssRuntime`), so DLSS runs from the first launch.
//!
//! A file is current when its length matches and its `<file>.sha1` record, written beside it after
//! a verified download, holds the manifest's hash: no 60 MB re-hash per launch. Anything short of
//! a whole, verified copy (offline, no manifest, a short or corrupt part, the window closed, a
//! folder it cannot write) leaves the folder as it was, and the client runs DLSS with what it has,
//! or FSR without it. A file the site does not serve is skipped.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::update::{self, Manifest, Shared, CANCELLED, DONE, FAILED, RUNNING};

/// No DLSS runtime file is this large; a manifest that says otherwise is not followed.
const MAX_BYTES: u64 = 512 << 20;

fn hex(sha1: &[u8; 20]) -> String {
    sha1.iter().map(|b| format!("{b:02x}")).collect()
}

fn record(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.sha1"))
}

/// Whether `dir/name` is the manifest's file: its length, and the hash recorded at its download.
fn current(dir: &Path, name: &str, manifest: &Manifest, sha1: &[u8; 20]) -> bool {
    fs::metadata(dir.join(name)).is_ok_and(|m| m.len() == manifest.bytes)
        && fs::read_to_string(record(dir, name)).is_ok_and(|r| r.trim() == hex(sha1))
}

/// The site's manifest for one file: `Ok(None)` when it serves none (or not a usable one),
/// `Err` when the site cannot be reached, which ends the whole check.
fn fetch(agent: &ureq::Agent, url: &str) -> io::Result<Option<(Manifest, [u8; 20])>> {
    let mut resp = agent
        .get(url)
        .call()
        .map_err(|e| io::Error::other(format!("GET {url}: {e}")))?;
    if resp.status().as_u16() != 200 {
        return Ok(None);
    }
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| io::Error::other(format!("GET {url}: {e}")))?;
    Ok(Manifest::parse(&text).and_then(|m| {
        let usable = m.bytes > 0
            && m.bytes <= MAX_BYTES
            && !m.parts.is_empty()
            && m.parts.iter().all(|p| update::plain(p));
        let sha1 = m.sha1?;
        usable.then_some((m, sha1))
    }))
}

/// Fetch into `dir` every file of `files` the site at `origin` serves and `dir` lacks.
pub fn ensure(origin: &str, dir: &Path, files: &[&str]) {
    let base = format!("{}/dlss", origin.trim_end_matches('/'));
    let agent = update::agent(Duration::from_secs(5));
    let mut wanted = Vec::new();
    for &name in files {
        if !update::plain(name) {
            continue;
        }
        match fetch(&agent, &format!("{base}/{name}.parts.json")) {
            Ok(Some((manifest, sha1))) => {
                if !current(dir, name, &manifest, &sha1) {
                    wanted.push((name.to_string(), manifest, sha1));
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("dlss: {e}; using what {} holds", dir.display());
                return;
            }
        }
    }
    if wanted.is_empty() {
        return;
    }
    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("dlss: {}: {e}", dir.display());
        return;
    }
    let total: u64 = wanted.iter().map(|(_, m, _)| m.bytes).sum();
    let shared = Arc::new(Shared {
        heading: "Downloading NVIDIA DLSS\u{2026}",
        ..Default::default()
    });
    let worker = {
        let (shared, dir) = (shared.clone(), dir.to_path_buf());
        std::thread::spawn(move || {
            let result = install_all(&dir, &base, &wanted, &shared);
            if let Err(e) = &result {
                eprintln!("dlss: {e}");
            }
            let end = if result.is_ok() { DONE } else { FAILED };
            let _ =
                shared
                    .state
                    .compare_exchange(RUNNING, end, Ordering::AcqRel, Ordering::Acquire);
        })
    };
    if update::wait(&shared, total) {
        // Closed: the worker stops at its next chunk and removes its partial file.
        let _ =
            shared
                .state
                .compare_exchange(RUNNING, CANCELLED, Ordering::AcqRel, Ordering::Acquire);
        return;
    }
    let _ = worker.join();
}

/// Each file in turn: download beside it, verified, then moved into place and its hash recorded.
fn install_all(
    dir: &Path,
    base: &str,
    wanted: &[(String, Manifest, [u8; 20])],
    shared: &Shared,
) -> io::Result<()> {
    let mut counted = 0;
    for (name, manifest, sha1) in wanted {
        let staged = dir.join(format!("{name}.download"));
        let result = File::create(&staged).and_then(|file| {
            update::download(file, base, manifest, sha1, shared, counted)?;
            // The old copy first: Windows renames onto no existing file. Its record goes with it,
            // so an interrupted swap is a missing file, fetched again next launch.
            let _ = fs::remove_file(record(dir, name));
            match fs::remove_file(dir.join(name)) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
                _ => {}
            }
            fs::rename(&staged, dir.join(name))?;
            fs::write(record(dir, name), hex(sha1))
        });
        if result.is_err() {
            let _ = fs::remove_file(&staged);
        }
        result.map_err(|e| io::Error::other(format!("{name}: {e}")))?;
        counted += manifest.bytes;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file is current only with its length and its recorded hash both matching.
    #[test]
    fn a_file_is_current_by_length_and_recorded_hash() {
        let dir = std::env::temp_dir().join(format!("benilla-dlss-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let sha1 = [0xab; 20];
        let manifest = Manifest {
            bytes: 3,
            parts: vec!["x.part00".into()],
            version: None,
            sha1: Some(sha1),
        };
        assert!(!current(&dir, "x.dll", &manifest, &sha1), "absent");
        fs::write(dir.join("x.dll"), b"abc").unwrap();
        assert!(!current(&dir, "x.dll", &manifest, &sha1), "no record");
        fs::write(record(&dir, "x.dll"), hex(&sha1)).unwrap();
        assert!(current(&dir, "x.dll", &manifest, &sha1));
        fs::write(dir.join("x.dll"), b"abcd").unwrap();
        assert!(!current(&dir, "x.dll", &manifest, &sha1), "length moved");
        let _ = fs::remove_dir_all(&dir);
    }
}
