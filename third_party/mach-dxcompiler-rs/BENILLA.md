# benilla's `mach-dxcompiler-rs` fork — what differs from upstream

Upstream: `mach-dxcompiler-rs 0.1.7+2026.10.26-141dd7f.1` (crates.io), MIT. Wired in through
`[patch.crates-io]` in the workspace root. Unpacked from the registry's `.crate` (sha256
`ae7b01febf4806c6a95a217784adcd0f0d19413eef5afb1fbea60656ea28c081`, the `Cargo.lock` checksum),
with cargo's packaging metadata (`.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`) dropped
as the other `third_party` forks drop it.

It is the statically linked DXC that bevy's `statically-linked-dxc` feature pulls in through
wgpu-hal's `static-dxc` (x64 Windows MSVC only), so DX12 compiles shaders with DXC rather than FXC.

## The one delta: a local release archive — `build.rs`, `main` and `local_archive`

Upstream's build script always asks the GitHub API whether release `2026.10.26+141dd7f.1` is
immutable, then downloads the prebuilt archive from the release and checks it against the SHA-256
pinned in `targets.rs`. A GitHub outage (a 503 page for the archive, a "Unicorn!" page for the API)
fails every clean build.

`local_archive` names a copy to use instead: `$MACH_DXC_ARCHIVE` when set, else the archive's own
file name in this folder (gitignored), for this build
`x86_64-windows-msvc_ReleaseFast_Dynamic_lib.tar.gz` from
<https://github.com/DouglasDwyer/mach-dxcompiler/releases/download/2026.10.26+141dd7f.1/x86_64-windows-msvc_ReleaseFast_Dynamic_lib.tar.gz>.
A local copy skips the API query and the download, and goes through the same `verify_checksum`
(`abe009a4abb88dab9a1659ca12515e35baa34478a2ed42103595ebf65f8f6b25`): a copy that is not those exact
bytes fails the build. Without one, the script does what upstream does.

## How to check

`diff -r` against the registry copy (`~/.cargo/registry/src/*/mach-dxcompiler-rs-0.1.7*`) shows
the `build.rs` hunks marked `BENILLA:`, the `.gitignore` line, this file, and the dropped metadata.
