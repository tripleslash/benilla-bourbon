#!/usr/bin/env bash
#
# build-desktop.sh — fork: desktop — the self-contained Windows x64 client, one executable.
#
#   scripts/build-desktop.sh            -> target/desktop/wow-reforged.exe
#
# The player build of `crates/benilla-desktop` (no `dev` feature, so no source-tree path is baked
# in), with the MSVC C runtime linked statically (`+crt-static`), so the .exe needs no Visual C++
# redistributable: only what Windows itself ships. It carries no game data: on first launch it
# creates `wow-data/` beside itself and fetches what it reads from https://wow.example.com.
#
# `--target` is named explicitly so the static-CRT flags apply to the target alone and land in
# their own folder (`target/x86_64-pc-windows-msvc/`): the host build cache is left as it is.
#
#   BENILLA_DESKTOP_PROFILE=ship   fat LTO (slower to build, smaller and faster binary)

set -euo pipefail
cd "$(dirname "$0")/.."

triple=x86_64-pc-windows-msvc
profile="${BENILLA_DESKTOP_PROFILE:-release}"

export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="-C target-feature=+crt-static"
cargo build -p benilla-desktop --no-default-features --profile "$profile" --target "$triple"

out="target/desktop"
mkdir -p "$out"
cp "target/$triple/$profile/wow-reforged.exe" "$out/wow-reforged.exe"
echo "built $out/wow-reforged.exe ($(wc -c < "$out/wow-reforged.exe") bytes)"
