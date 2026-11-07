#!/usr/bin/env bash
#
# stage-desktop.sh — fork: desktop — stage wow-reforged.exe into web/site/ for an upload to the static host.
#
#   scripts/stage-desktop.sh [exe]      (default: target/desktop/wow-reforged.exe)
#
# Run by `scripts/build-desktop.sh` once the exe is built, and by `scripts/stage-packs.sh` (which
# clears web/site/ on every web build) so a later web build keeps it. Its twins are the desktop
# steps in `stage-packs.py` and `stage-packs.ps1`, which change with it.
#
# The exe is larger than a static host's 25 MiB per-asset limit, so `scripts/split-parts.sh` splits it
# into `wow-reforged.exe.partNN` files plus `wow-reforged.exe.parts.json`, the wasm module's layout.
# The manifest also carries the exe's SHA-1 and, from the `<exe>.version` file `build-desktop.sh`
# writes beside it, the build's commit time as `version`: the client's self-update
# (`crates/benilla-desktop/src/update.rs`) replaces an older build with a newer one only, and only
# with a copy that matches the hash. The page's "Desktop client" link (`web/platform.js`) reads the
# same manifest. Absent exe is an ordinary state: the site has no manifest, the page hides the
# link and no client updates.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=split-parts.sh
. scripts/split-parts.sh

EXE="${1:-target/desktop/wow-reforged.exe}"
SITE=web/site
NAME=wow-reforged.exe

mkdir -p "${SITE}"
clear_parts "${SITE}" "${NAME}"

if [ ! -f "${EXE}" ]; then
  echo "note: no ${EXE} — staging without the desktop client (scripts/build-desktop.sh)" >&2
  exit 0
fi

if command -v sha1sum >/dev/null 2>&1; then
  sha1=$(sha1sum "${EXE}" | cut -d' ' -f1)
else
  sha1=$(shasum -a 1 "${EXE}" | cut -d' ' -f1)
fi
extra="\"sha1\":\"${sha1}\""
# An exe with no recorded commit time ships without `version`, which no client updates to.
version=""
if [ -f "${EXE}.version" ]; then
  version=$(tr -dc '0-9' < "${EXE}.version")
fi
if [ -n "${version}" ]; then
  extra="\"version\":${version},${extra}"
else
  echo "note: no ${EXE}.version — clients will not update to this exe" >&2
fi
split_parts "${EXE}" "${SITE}" "${NAME}" "" "${extra}"
