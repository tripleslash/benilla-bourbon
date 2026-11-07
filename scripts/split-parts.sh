#!/usr/bin/env bash
#
# split-parts.sh — split one file into 24 MiB parts plus a manifest, for a static host's 25 MiB
# per-asset cap. The one splitter the site has: the wasm module (`scripts/stage-packs.sh`) and the
# desktop client (`scripts/stage-desktop.sh`) both go through it. Its twins are `split_parts` in
# `stage-packs.py` and `Split-Parts` in `stage-packs.ps1`, which change with it.
#
#   scripts/split-parts.sh <src> <dest-dir> <stem> <suffix> [extra]
#   . scripts/split-parts.sh            (defines PART_MAX, clear_parts and split_parts)
#
# Writes `<stem>.part00<suffix>`, `<stem>.part01<suffix>`, ... and `<stem>.parts.json`,
# `{"bytes":N,<extra>,"parts":["<stem>.part00<suffix>",...]}`, after removing the previous ones.
# `extra` is a JSON fragment of further keys (the desktop client's `"version":T,"sha1":"…"`, which
# its self-update reads); readers of the manifest skip keys they do not know. `src` is only read.
#
# Its readers: `readParts` in `web/parts.js` (the page's wasm boot and the desktop download) and
# `crates/benilla-desktop/src/update.rs` (the client's self-update).

# 24 MiB, the same cap `benilla-pack build` holds a pack object to, a MiB under a static host's limit.
PART_MAX=$((24 * 1024 * 1024))

# clear_parts <dest-dir> <stem>: remove a previous split of <stem>.
clear_parts() {
  rm -f "$1/$2".part* "$1/$2.parts.json" "$1/$2.split."*
}

# split_parts <src> <dest-dir> <stem> <suffix> [extra]
split_parts() {
  local src="$1" dest="$2" stem="$3" suffix="$4" extra="${5:-}"
  local bytes chunk sep="" i=0 n
  clear_parts "${dest}" "${stem}"
  bytes=$(wc -c < "${src}" | tr -d ' ')
  # Not `split -d --additional-suffix`: those are GNU-only and BSD split (macOS) refuses them. The
  # default `aa`, `ab`, ... suffixes both share sort in order and are renamed.
  split -b "${PART_MAX}" "${src}" "${dest}/${stem}.split."
  for chunk in "${dest}/${stem}".split.*; do
    mv "${chunk}" "$(printf '%s/%s.part%02d%s' "${dest}" "${stem}" "${i}" "${suffix}")"
    i=$((i + 1))
  done
  {
    printf '{"bytes":%s,' "${bytes}"
    if [ -n "${extra}" ]; then
      printf '%s,' "${extra}"
    fi
    printf '"parts":['
    for ((n = 0; n < i; n++)); do
      printf '%s"%s.part%02d%s"' "${sep}" "${stem}" "${n}" "${suffix}"
      sep=","
    done
    printf ']}\n'
  } > "${dest}/${stem}.parts.json"
  echo "split $(basename "${src}") (${bytes} B) into ${i} parts in ${dest}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  [ $# -ge 4 ] || { echo "usage: $0 <src> <dest-dir> <stem> <suffix> [extra]" >&2; exit 2; }
  split_parts "$@"
fi
