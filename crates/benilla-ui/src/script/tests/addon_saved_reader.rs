//! An addon's saved-variables files read through the host's seated reader, so a host whose state
//! folder is not on disk (the page's `localStorage`) still restores them at `LoadAddOn`.

use std::path::{Path, PathBuf};

use crate::script::{AddOnInfo, UiScript};

/// The declared global comes back from a file only the reader knows, and the per-character file
/// the reader answers `NotFound` for is the silent first-run case, not an error.
#[test]
fn saved_variables_read_through_the_seated_reader() {
    let mut s = UiScript::new().unwrap();
    s.register_addons(
        vec![AddOnInfo {
            name: "Probe".into(),
            interface: 11200,
            enabled: true,
            saved_variables: vec!["ProbeDB".into()],
            saved_variables_per_character: vec!["ProbeCharDB".into()],
            ..Default::default()
        }],
        None,
        Some(PathBuf::from("/virtual/saved")),
        Some(PathBuf::from("/virtual/saved/Realm-Char")),
    );
    s.set_addon_saved_reader(Box::new(|path: &Path| {
        if path == Path::new("/virtual/saved/Probe.lua") {
            Ok(b"ProbeDB = { x = 7 }\n".to_vec())
        } else {
            Err(std::io::ErrorKind::NotFound.into())
        }
    }));
    s.load_addon_saved_variables("Probe");
    assert_eq!(s.eval::<i64>("return ProbeDB.x").unwrap(), 7);
    assert!(
        s.eval::<bool>("return ProbeCharDB == nil").unwrap(),
        "an absent file leaves the default"
    );
    assert_eq!(s.errors(), Vec::<String>::new(), "a missing file is silent");
}
