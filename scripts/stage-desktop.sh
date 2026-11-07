#!/usr/bin/env bash
#
# stage-desktop.sh — fork: desktop — stage wow-reforged.exe into web/site/ for an upload to the static host.
#
#   scripts/stage-desktop.sh [exe]      (default: target/desktop/wow-reforged.exe)
#
# Run by `scripts/build-desktop.sh` once the exe is built, and by `scripts/stage-packs.sh` (which
# clears web/site/ on every web build) so a later web build keeps it. Its twins in
# `stage-packs.py` and `stage-packs.ps1` change with it.
#
# The exe is larger than a static host's 25 MiB per-asset limit, so it is split into 24 MiB
# `wow-reforged.exe.partNN` files plus `wow-reforged.exe.parts.json` ({"bytes":N,"parts":[...]}),
# the same layout as the wasm module's. The page's "Desktop client" link (`web/platform.js`,
# `installDesktopDownload`) reads the manifest, fetches the parts and saves them back as one
# `wow-reforged.exe`. Absent exe is an ordinary state: the site has no manifest and the page hides
# the link.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXE="${1:-target/desktop/wow-reforged.exe}"
SITE=web/site
PART_MAX=$((24 * 1024 * 1024))
NAME=wow-reforged.exe

mkdir -p "${SITE}"
rm -f "${SITE}/${NAME}".part* "${SITE}/${NAME}.parts.json" "${SITE}/${NAME}.split."*

if [ ! -f "${EXE}" ]; then
  echo "note: no ${EXE} — staging without the desktop client (scripts/build-desktop.sh)" >&2
  exit 0
fi

bytes=$(wc -c < "${EXE}")
# BSD split (macOS) has no `-d`; the default `aa`, `ab`, ... suffixes sort in order and are renamed.
split -b "${PART_MAX}" "${EXE}" "${SITE}/${NAME}.split."
i=0
for chunk in "${SITE}/${NAME}".split.*; do
  mv "${chunk}" "$(printf '%s/%s.part%02d' "${SITE}" "${NAME}" "${i}")"
  i=$((i + 1))
done
{
  printf '{"bytes":%s,"parts":[' "${bytes}"
  sep=""
  for part in "${SITE}/${NAME}".part*; do
    printf '%s"%s"' "${sep}" "$(basename "${part}")"
    sep=","
  done
  printf ']}\n'
} > "${SITE}/${NAME}.parts.json"
echo "split ${NAME} (${bytes} B) into ${i} parts in ${SITE}"
