//! Locating a built container from a test or a tool. Native only — the browser is *served* one.
//!
//! The container is gitignored: 3.3 GiB derived from a Blizzard install, and the project's standing
//! contract is that no Blizzard asset is ever committed. So anything that reads it takes the same
//! shape as the tests that read the install itself ([`benilla_formats::wow_data_or_skip`]): find it,
//! or **skip without asserting**, saying where it looked.

use std::path::PathBuf;

/// Where a built container might be, in order. `$BENILLA_PACKS` first, so one built anywhere can be
/// pointed at without moving it.
pub fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = std::env::var_os("BENILLA_PACKS") {
        out.push(PathBuf::from(dir));
    }
    // `benilla-pack build`'s default `--out`, relative to the workspace root — two levels up from
    // any crate's own manifest directory.
    out.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/packs"));
    out
}

/// The first candidate that actually holds an `index.bpi`.
pub fn find() -> Option<PathBuf> {
    candidates()
        .into_iter()
        .find(|dir| dir.join("index.bpi").is_file())
}

/// The container's parsed index, or **skip this test**.
///
/// ```ignore
/// let index = pack_index_or_skip!();
/// ```
#[macro_export]
macro_rules! pack_index_or_skip {
    () => {
        $crate::pack_index_or_skip!(())
    };
    ($ret:expr) => {
        match $crate::find::find() {
            Some(dir) => {
                let path = dir.join("index.bpi");
                let bytes = match std::fs::read(&path) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        eprintln!("skipping: reading {}: {e}", path.display());
                        return $ret;
                    }
                };
                match $crate::Index::parse(bytes) {
                    Ok(index) => index,
                    Err(e) => {
                        eprintln!("skipping: {}: {e}", path.display());
                        return $ret;
                    }
                }
            }
            None => {
                eprintln!(
                    "skipping: no BENPACK container — looked in {:?}; build one with \
                     `benilla-pack build --data <WoW>/Data --out web/packs`",
                    $crate::find::candidates()
                );
                return $ret;
            }
        }
    };
}
