//! fork: desktop — the self-contained client. The `benilla` launcher's shim, plus the two facts
//! that make it stand alone: its game data comes from the web build's site (no install, none
//! shipped), and everything it keeps — settings, addons, saved variables and the fetched data — is
//! in one `wow-data` folder created beside the executable.
//!
//! `$BENILLA_DATA_URL` points it at another site; `$WOW_HOST` at a server reached directly.

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
