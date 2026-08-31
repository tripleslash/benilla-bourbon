#!/usr/bin/env bash
#
# stage-packs.sh — assemble web/site/, the directory a static host serves.
#
# Runs as Trunk's post_build hook (see Trunk.toml); also runnable on its own after a container
# rebuild. Three directories, and the split is forced:
#
#   web/dist/    Trunk's output. CLEARED ON EVERY BUILD — which is why 3.3 GiB of container
#                cannot live here, and why the site root is not simply `dist`.
#   web/packs/   `benilla-pack build` output: ~4,700 .bpk objects plus index.bpi, listfile.txt
#                and boot-packs.json. Rebuilt only when the game files change. (`config.json` is
#                NOT here: it configures the client, ships with the build, and lands at the site
#                root — see web/index.html's copy-file link.)
#   web/site/  what a static host serves: hardlinks to the build, plus `data` -> ../packs. The one
#                file that is NOT a link is the wasm module: it is 77 MB against a static host's
#                25 MiB per-asset limit, so it is split here into 24 MiB `wenilla_bg.part*.wasm`
#                and `web/boot.js` streams them back together. See the split step below.
#
# Both links are cheap on purpose. Hardlinking the build is O(1) per file against a ~90 MB wasm
# module that changes on every build, and `data` is a link rather than a copy because copying
# 3.3 GiB per build is not a thing anyone would run twice.
#
# **The build is not in `web/dist` when this runs as a hook.** Measured: a `post_build` hook fires
# BEFORE Trunk moves its staging directory into place, so `dist/` is still empty and the first
# version of this script died on `cp: cannot stat 'web/dist/*'`. Trunk 0.21.14 also does not export
# `$TRUNK_STAGING_DIR` (checked — the second run failed identically), so the source is resolved by
# looking: staging first, then `dist`. Hardlinks survive the move that follows, because a link names
# the inode and not the path.
#
#   STAGE_PACKS_LINK=hardlink   link every pack individually instead of linking the directory.
#                               The fallback if the upload tool's asset walker will not traverse a
#                               junction/symlink — still seconds, still no extra disk.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PACKS=web/packs
SITE=web/site

# First candidate that exists and is non-empty. Staging outranks `dist` so a hook run stages THIS
# build rather than the last one — silently staging a stale wasm is the failure worth ruling out.
DIST=""
for candidate in "${TRUNK_STAGING_DIR:-}" web/dist/.stage web/dist; do
  if [ -n "${candidate}" ] && [ -n "$(ls -A "${candidate}" 2>/dev/null)" ]; then
    DIST="${candidate}"
    break
  fi
done
[ -n "${DIST}" ] || { echo "no build found in web/dist or web/dist/.stage — run \`trunk build --release\` first" >&2; exit 1; }
echo "staging from ${DIST}"
[ -d "${PACKS}" ] || echo "warning: no ${PACKS} — staging the client with no game data (benilla-pack build)" >&2

mkdir -p "${SITE}"
# Everything EXCEPT data/. Deleting a junction or symlink never touches its target, but leaving it
# in place also skips re-creating it, and on Windows that is one fewer `cmd //c` per build.
find "${SITE}" -mindepth 1 -maxdepth 1 ! -name data -exec rm -rf {} +

# Git Bash maps `cp -l` to CreateHardLink, so this is metadata-only even for the 90 MB wasm. The
# fallback copy is for a filesystem that refuses links (a FAT stick, a mounted share).
cp -l "${DIST}"/* "${SITE}"/ 2>/dev/null || cp -r "${DIST}"/* "${SITE}"/

# Split the wasm module. A static host refuses ANY static asset over 25 MiB and the module is
# 77 MB — already `-O3 --strip-debug --strip-producers`, one 184-byte custom section left, so there
# is nothing in it to remove and no flag that closes a 3x gap. `web/boot.js` concatenates the parts
# back into ONE stream, which it can do for free because it was already re-wrapping the download in
# a synthetic Response to count its bytes; the browser still compiles while it downloads.
#
# `web/dist` is NOT split — `trunk serve` and `wenilla-host --www` serve that directory and have no
# such limit. So both layouts exist at once, and `wenilla_bg.parts.json` is how the page tells them
# apart: written here, absent there.
#
# 24 MiB, the same cap `benilla-pack build` holds a pack object to. One number for both is one
# number to remember, and the MiB of headroom is there for the same reason it is there.
WASM_PART_MAX=$((24 * 1024 * 1024))
if [ -f "${SITE}/wenilla_bg.wasm" ]; then
  bytes=$(wc -c < "${SITE}/wenilla_bg.wasm")
  # `--additional-suffix=.wasm` is load-bearing, not tidiness: A static host types an asset by its
  # extension and compresses on the fly BY TYPE, so a part named `.part00` would be
  # `application/octet-stream` and could ship uncompressed — 77 MB on the wire against the ~15 MB
  # the module costs today. Named `*.wasm` it is the same kind of object the whole file is.
  split -b "${WASM_PART_MAX}" -d --additional-suffix=.wasm \
    "${SITE}/wenilla_bg.wasm" "${SITE}/wenilla_bg.part"
  # The hardlink, not the build: removing a link never touches the inode `web/dist` still names.
  rm "${SITE}/wenilla_bg.wasm"
  # Unconditional, even if the module ever fits: one layout per directory is one code path in
  # `fetchWasm`, and a split that only sometimes happens is a path that only sometimes runs.
  {
    printf '{"bytes":%s,"parts":[' "${bytes}"
    sep=""
    for part in "${SITE}"/wenilla_bg.part*.wasm; do
      printf '%s"%s"' "${sep}" "$(basename "${part}")"
      sep=","
    done
    printf ']}\n'
  } > "${SITE}/wenilla_bg.parts.json"
  echo "split wenilla_bg.wasm (${bytes} B) into $(ls "${SITE}"/wenilla_bg.part*.wasm | wc -l) parts"
fi

# The class of bug, not the instance: anything else that grows past the cap fails the same way, and
# it fails inside an upload to the static host minutes into an upload rather than here. `-maxdepth 1` on
# purpose — the packs under `data/` are capped by the packer (measured max 9.49 MiB) and walking a
# junction over 4,700 files to re-prove that is not worth the seconds.
oversize=$(find "${SITE}" -maxdepth 1 -type f -size +"${WASM_PART_MAX}"c)
if [ -n "${oversize}" ]; then
  echo "error: asset over a static host's 25 MiB limit (cap here is 24 MiB):" >&2
  echo "${oversize}" >&2
  exit 1
fi

link_data() {
  case "${STAGE_PACKS_LINK:-dir}" in
    hardlink)
      rm -rf "${SITE}/data"
      mkdir -p "${SITE}/data"
      # -r follows the tree, -l hardlinks the leaves: no bytes move.
      cp -lr "${PACKS}"/. "${SITE}/data"/
      ;;
    *)
      [ -e "${SITE}/data" ] && return 0
      case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
          # A DIRECTORY JUNCTION, not a symlink: junctions need no administrator rights and no
          # Developer Mode on Windows, which a symlink does.
          #
          # `MSYS2_ARG_CONV_EXCL='*'` is required and its two effects have to be taken together:
          # without it MSYS mangles mklink's `/J` switch into a path and cmd answers "Invalid
          # switch"; with it, `cmd //c` is no longer rewritten to `/c` either, so cmd must be given
          # `/c` literally — otherwise it ignores the arguments and opens an interactive shell that
          # exits successfully having done nothing. Both were observed, in that order.
          MSYS2_ARG_CONV_EXCL='*' cmd /c mklink /J \
            "$(cygpath -w "${SITE}/data")" "$(cygpath -w "${PACKS}")" >/dev/null
          ;;
        *)
          ln -s ../packs "${SITE}/data"
          ;;
      esac
      ;;
  esac
}
[ -d "${PACKS}" ] && link_data

echo "staged ${SITE}:"
ls -la "${SITE}"
