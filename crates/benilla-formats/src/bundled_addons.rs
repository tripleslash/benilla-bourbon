//! The third-party addons benilla ships, as files under `Interface\AddOns\<Folder>\` that the
//! [`crate::Chain`] answers after its archives miss: a `.toc`, its Lua and XML, and the art, fonts
//! and sounds the addon names by path all resolve through the one read every caller already makes.
//!
//! Deviation: the reference client ships no addon. benilla bundles a few (`third_party/addons`),
//! which the app registers as off until the player ticks them (`ui_script::addons`). Which folders
//! exist is this module's; which of them a flavor offers is the app's.
//!
//! Where the files are:
//! * native: `$BENILLA_BUNDLED_ADDONS` (set and empty: none), else `<project folder>/third_party/
//!   addons` in a dev build, else `<exe dir>/bundled-addons`;
//! * web: `/addons/<Folder>/<file>` on the page's origin, listed by `/addons/index.txt` (one
//!   relative path per line), since a static origin has no directory listing.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::LazyLock;

/// The path prefix every bundled file lives under, in the index's key form.
const PREFIX: &str = "interface\\addons\\";

/// The URL segment the web build serves the bundled addons from, beside `/data`.
pub const WEB_SEGMENT: &str = "addons";

/// Where one indexed file is read from.
#[cfg(not(target_arch = "wasm32"))]
type Location = PathBuf;
/// The file's path relative to [`WEB_SEGMENT`], in its own case.
#[cfg(target_arch = "wasm32")]
type Location = String;

/// The index: `interface\addons\<folder>\<file>` (lowercase, backslashes) to where it is read from,
/// plus each top-level folder in its own case.
struct Index {
    files: HashMap<String, Location>,
    folders: Vec<String>,
}

static INDEX: LazyLock<Index> = LazyLock::new(build_index);

/// The key a chain name maps to, or `None` for a name outside `Interface\AddOns\`. The prefix is
/// checked before anything is allocated, as every chain miss asks.
fn key(name: &str) -> Option<String> {
    let head = name.as_bytes().get(..PREFIX.len())?;
    let same = |(&a, &b): (&u8, &u8)| a.to_ascii_lowercase() == b || (a == b'/' && b == b'\\');
    if !head.iter().zip(PREFIX.as_bytes()).all(same) {
        return None;
    }
    Some(name.replace('/', "\\").to_ascii_lowercase())
}

/// Whether a bundled file answers `name` (`/` or `\`, any case).
pub fn contains(name: &str) -> bool {
    key(name).is_some_and(|k| INDEX.files.contains_key(&k))
}

/// The bundled file `name`, or `None` when there is none.
pub fn read(name: &str) -> Option<Vec<u8>> {
    let location = INDEX.files.get(&key(name)?)?;
    read_location(location)
}

/// The bundled addon folders, in their own case.
pub fn folders() -> &'static [String] {
    &INDEX.folders
}

#[cfg(not(target_arch = "wasm32"))]
fn read_location(path: &Location) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

#[cfg(target_arch = "wasm32")]
fn read_location(rel: &Location) -> Option<Vec<u8>> {
    // Each segment encoded on its own: the `/` between them is the URL's.
    let path: Vec<String> = rel.split('/').map(crate::web::encode_name).collect();
    crate::web::fetch_sync(&format!("{}/{}", web_base(), path.join("/"))).ok()
}

#[cfg(target_arch = "wasm32")]
fn web_base() -> String {
    let origin = web_sys::window()
        .and_then(|w| w.location().origin().ok())
        .unwrap_or_default();
    format!("{origin}/{WEB_SEGMENT}")
}

/// The native root, per the module docs.
#[cfg(not(target_arch = "wasm32"))]
fn root() -> Option<PathBuf> {
    if let Some(over) = std::env::var_os("BENILLA_BUNDLED_ADDONS") {
        return (!over.is_empty()).then(|| PathBuf::from(over));
    }
    if let Some(project) = crate::project_folder() {
        return Some(project.join("third_party").join("addons"));
    }
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join("bundled-addons")))
}

#[cfg(not(target_arch = "wasm32"))]
fn build_index() -> Index {
    let mut index = Index {
        files: HashMap::new(),
        folders: Vec::new(),
    };
    let Some(root) = root() else {
        return index;
    };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return index; // no bundle is a valid build
    };
    let mut folders: Vec<(String, PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| Some((e.file_name().to_str()?.to_owned(), e.path())))
        .collect();
    folders.sort();
    for (name, dir) in folders {
        walk(
            &dir,
            &format!("{PREFIX}{}", name.to_ascii_lowercase()),
            &mut index.files,
        );
        index.folders.push(name);
    }
    index
}

/// Index every file under `dir` as `<key>\<relative path>`.
#[cfg(not(target_arch = "wasm32"))]
fn walk(dir: &std::path::Path, key: &str, files: &mut HashMap<String, PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let Some(name) = e.file_name().to_str().map(str::to_ascii_lowercase) else {
            continue;
        };
        let path = e.path();
        let k = format!("{key}\\{name}");
        if path.is_dir() {
            walk(&path, &k, files);
        } else {
            files.insert(k, path);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn build_index() -> Index {
    let text = crate::web::fetch_sync(&format!("{}/index.txt", web_base()))
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    index_from_listing(&text)
}

/// The index for a listing of relative paths (`<Folder>/<file>`, one per line), the web's form.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn index_from_listing(text: &str) -> Index {
    let mut index = Index {
        files: HashMap::new(),
        folders: Vec::new(),
    };
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let rel = line.replace('\\', "/");
        let Some((folder, _)) = rel.split_once('/') else {
            continue;
        };
        if !index.folders.iter().any(|f| f == folder) {
            index.folders.push(folder.to_owned());
        }
        let k = format!("{PREFIX}{}", rel.replace('/', "\\").to_ascii_lowercase());
        #[cfg(target_arch = "wasm32")]
        index.files.insert(k, rel);
        #[cfg(not(target_arch = "wasm32"))]
        index.files.insert(k, PathBuf::from(rel));
    }
    index.folders.sort();
    index
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn only_names_under_the_addons_folder_have_a_key() {
        assert_eq!(
            key("Interface/AddOns/pfQuest/pfQuest.toc").as_deref(),
            Some("interface\\addons\\pfquest\\pfquest.toc")
        );
        assert_eq!(key("Interface\\FrameXML\\UIParent.lua"), None);
        assert_eq!(key("x"), None);
    }

    #[test]
    fn a_listing_indexes_files_by_key_and_names_its_folders() {
        let index = index_from_listing("pfQuest/pfQuest.toc\r\npfQuest-turtle/db/x.lua\n\n");
        assert_eq!(index.folders, ["pfQuest", "pfQuest-turtle"]);
        assert!(index
            .files
            .contains_key("interface\\addons\\pfquest-turtle\\db\\x.lua"));
    }

    /// The checkout's bundle is what the dev build reads: both pfQuest folders, each with its toc.
    #[cfg(feature = "dev")]
    #[test]
    fn the_checkout_bundles_pfquest_and_its_turtle_database() {
        assert!(folders().iter().any(|f| f == "pfQuest"));
        assert!(folders().iter().any(|f| f == "pfQuest-turtle"));
        assert!(contains("Interface\\AddOns\\pfQuest\\pfQuest.toc"));
        assert!(read("interface/addons/PFQUEST-TURTLE/pfQuest-turtle.toc")
            .is_some_and(|b| b.starts_with(b"## Interface: 11200")));
    }
}
