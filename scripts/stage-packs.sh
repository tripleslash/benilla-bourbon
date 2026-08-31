#!/usr/bin/env bash
#
# stage-packs.sh — assemble web/site/, the directory a static host serves.
#
# Runs as Trunk's post_build hook (see Trunk.toml); also runnable on its own after a container
# rebuild. Three directories, and the split is forced:
#
#   web/dist/    Trunk's output. CLEARED ON EVERY BUILD — which is why 3.3 GiB of container
#                cannot live here, and why the site root is not simply `dist`.
#   web/packs/   `benilla-pack build` output: ~4,700 .bpk objects plus index.bpi, listfile.txt,
#                boot-packs.json and config.json. Rebuilt only when the game files change.
#   web/site/  what a static host serves: hardlinks to the build, plus `data` -> ../packs.
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
