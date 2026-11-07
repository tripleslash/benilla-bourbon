//! fork: desktop — the self-contained client. The `benilla` launcher's shim, plus the two facts
//! that make it stand alone: its game data comes from the web build's site (no install, none
//! shipped), and everything it keeps — settings, addons, saved variables and the fetched data — is
//! in one `wow-data` folder created beside the executable.
//!
//! `$BENILLA_DATA_URL` points it at another site; `$WOW_HOST` at a server reached directly.
//!
//! The player build first replaces itself with a newer build when that site serves one
//! (`benilla_desktop::update`); a dev build runs as built.

// A player's build opens no console window; a dev build keeps its log on one.
#![cfg_attr(not(feature = "dev"), windows_subsystem = "windows")]

use benilla_app::BuildId;

/// The site the browser build is served from: its `/data` containers, `/addons` and
/// `/config.json` (the realm gateway and realmlist).
const ORIGIN: &str = "https://wow.example.com";

/// The one folder beside the executable that holds everything the client writes.
const HOME: &str = "wow-data";

fn main() -> benilla_app::AppExit {
    benilla_app::set_portable_home(HOME);
    benilla_app::set_data_origin(ORIGIN);
    if relaunched_newer() {
        return benilla_app::AppExit::Success;
    }
    benilla_app::run(BuildId {
        version: env!("CARGO_PKG_VERSION"),
        describe: env!("BENILLA_GIT_DESCRIBE"),
        sha: env!("BENILLA_GIT_SHA"),
        short: env!("BENILLA_GIT_SHORT"),
        date: env!("BENILLA_GIT_DATE"),
        profile: env!("BENILLA_PROFILE"),
        project_dir: env!("BENILLA_PROJECT_DIR"),
        ..Default::default()
    })
}

/// Whether a newer build from the data origin now runs in this one's place, so this one exits.
#[cfg(not(feature = "dev"))]
fn relaunched_newer() -> bool {
    use benilla_desktop::update::{self, Outcome};
    // This build's commit time (`build.rs`); empty when unstamped, and such a build never updates.
    let built = env!("BENILLA_DESKTOP_VERSION").parse().ok();
    benilla_app::data_origin()
        .is_some_and(|origin| update::run(&origin, built) == Outcome::Relaunched)
}

#[cfg(feature = "dev")]
fn relaunched_newer() -> bool {
    false
}
