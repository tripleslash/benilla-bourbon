//! Stamps the commit this binary was built from, gives the main thread the stack the app needs on
//! Windows (the `benilla` launcher's build script), and on a Windows target embeds the client's
//! icon: `web/favicon.ico`, WoW.exe's own icon group, which the web build serves as its favicon.
//! It also stamps the commit time the self-update compares against the site's manifest.

use std::process::Command;

fn main() {
    benilla_buildstamp::emit();
    // `BENILLA_DESKTOP_VERSION`: the commit's time in seconds since the epoch, the version
    // `src/update.rs` compares with the manifest's (`scripts/build-desktop.sh` records the same
    // `git log -1 --format=%ct`). Empty when git cannot answer. The stamp's own rerun triggers
    // (HEAD and its ref) rerun this too.
    let time = Command::new("git")
        .args([
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "log",
            "-1",
            "--format=%ct",
        ])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    println!("cargo::rustc-env=BENILLA_DESKTOP_VERSION={time}");
    let icon = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/favicon.ico");
    println!("cargo:rerun-if-changed={icon}");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon(icon);
        if let Err(e) = res.compile() {
            panic!("embedding the icon {icon}: {e}");
        }
    }
}
