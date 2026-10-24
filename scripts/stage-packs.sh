#!/usr/bin/env bash
#
# stage-packs.sh — assemble web/site/, the directory a static host serves.
#
# Run by the build scripts once `trunk build` returns (see Trunk.toml); also runnable on its own after
# a container rebuild. Three directories, and the split is forced:
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
#                               The fallback if a host's upload tool will not traverse a
#                               junction/symlink — still seconds, still no extra disk.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PACKS=web/packs
# The optional updated-character-model layer's container, built from `patch-A.MPQ` alone:
#
#   benilla-pack build --data ".../Data/patch-A.mpq" --out web/packs-hd
#
# A SEPARATE tree, hosted at `/data-hd`, and that is the feature rather than an implementation
# detail: a player with the option off never fetches its index, so the ~1.5 GiB of objects here cost
# them nothing at all. Absent is an ordinary state -- the site simply has no layer, the client
# says so once in the console and runs on the original art.
PACKS_HD=web/packs-hd
# The Turtle WoW 1.18.1 layer's container, hosted at `/data-turtle` and fetched only while the
# client is on a realm that advertises 1.18.1:
#
#   benilla-pack overlay ... --out <dir>, then build --only <dir>/overlay.tsv --out web/packs-turtle
#
# Absent is an ordinary state: the site serves 1.12.1 realms only, and the client refuses a 1.18.1
# realm with a message saying why.
PACKS_TURTLE=web/packs-turtle
# The reforged art layer's container (Turtle WoW's Project Reforged letter patches), hosted at
# `/data-reforged` and fetched only with the Turtle layer while the player's setting has it on:
#
#   benilla-pack overlay --patches B,C,D,E,G,M,P,S,T,Y ... --out <dir>, then build --only
#   <dir>/overlay.tsv --target-pack-bytes 3145728 --out web/packs-reforged (see overlay.rs)
#
# Absent is an ordinary state: a Turtle realm then plays on Turtle's own art.
PACKS_REFORGED=web/packs-reforged
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

# ---------------------------------------------------------------------------------------------
# Rewrite the glue's ONE bare module specifier, before anything links or copies this build.
#
# `wasm-bindgen --target web` emits `import * as importN from "wasi_snapshot_preview1"` — one line
# per WASI function the C Lua VM imports — because that is the wasm import module's NAME, and
# wasm-bindgen has no way to know it is meant to be a file. `web/index.html` resolves it with an
# import map, which works for the page and **does not work for a Web Worker**: import maps are a
# document feature, a worker does not inherit the page's, and there is no way to give one its own.
#
# So a worker that imports `./wenilla.js` — which is every task-pool thread (`web/threadworker.js`)
# — fails at module resolution before it runs a line. The failure surfaces as an `error` event on
# the Worker with no message, no filename and no line number, which is as close to undebuggable as
# the platform gets. Measured 2026-09-06: 15 workers, 15 opaque error events, empty pools.
#
# Rewriting the specifier to the relative path the import map pointed at makes the page and the
# worker resolve it the same way, through the ordinary module resolver. `index.html` keeps its
# import map as a fallback for anyone driving cargo + wasm-bindgen by hand without this script.
#
# **Here, and not in `scripts/web-build-threads.sh`.** This must happen before the `cp -l` below:
# the site is HARDLINKED to these files, `sed -i` replaces an inode rather than editing one, and
# a rewrite afterwards would fix `dist` while leaving `site` pointing at the old bytes.
for js in "${DIST}"/*.js; do
  [ -f "${js}" ] || continue
  if grep -q 'wasi_snapshot_preview1' "${js}" 2>/dev/null; then
    # Not `sed -i`: BSD sed (macOS) reads `-i -e` as a backup suffix, and `cat >` keeps the inode.
    sed -e 's|from "wasi_snapshot_preview1"|from "./wasi_stubs.js"|g' \
        -e "s|from 'wasi_snapshot_preview1'|from './wasi_stubs.js'|g" "${js}" > "${js}.tmp"
    cat "${js}.tmp" > "${js}"
    rm -f "${js}.tmp"
    echo "  rewrote the wasi_snapshot_preview1 specifier in $(basename "${js}") (workers have no import map)"
  fi
done
# ---------------------------------------------------------------------------------------------
[ -d "${PACKS}" ] || echo "warning: no ${PACKS} — staging the client with no game data (benilla-pack build)" >&2
[ -d "${PACKS_HD}" ] || echo "note: no ${PACKS_HD} — staging without the updated-character-model layer" >&2
[ -d "${PACKS_TURTLE}" ] || echo "note: no ${PACKS_TURTLE} — staging without the Turtle WoW 1.18.1 layer" >&2
[ -d "${PACKS_REFORGED}" ] || echo "note: no ${PACKS_REFORGED} — staging without the reforged art layer" >&2

# The install's loose files the interface reads by bare name (`Credits.html`, `eula.html`,
# `tos.html`, off `Data\`; `reference_ui::read_loose`), lowercased into `<container>/loose/`:
# the archives do not hold them, so the packs do not either. Read from the installs named by
# LOOSE_DATA (the 1.12.1 `Data`) and LOOSE_DATA_TURTLE (Turtle's), never written to.
stage_loose() {
  [ -n "$1" ] && [ -d "$1" ] && [ -d "$2" ] || return 0
  mkdir -p "$2/loose"
  for f in "$1"/*.html; do
    [ -f "$f" ] || continue
    cp "$f" "$2/loose/$(basename "$f" | tr '[:upper:]' '[:lower:]')"
  done
  echo "  staged $(ls "$2/loose" | wc -l) loose file(s) from $1 into $2/loose"
}
stage_loose "${LOOSE_DATA:-}" "${PACKS}"
stage_loose "${LOOSE_DATA_TURTLE:-}" "${PACKS_TURTLE}"

mkdir -p "${SITE}"
# Everything EXCEPT data/. Deleting a junction or symlink never touches its target, but leaving it
# in place also skips re-creating it, and on Windows that is one fewer `cmd //c` per build.
find "${SITE}" -mindepth 1 -maxdepth 1 ! -name data ! -name data-hd ! -name data-turtle ! -name data-reforged -exec rm -rf {} +

# Git Bash maps `cp -l` to CreateHardLink, so this is metadata-only even for the 90 MB wasm. The
# fallback copy is for a filesystem that refuses links (a FAT stick, a mounted share). A directory
# (wasm-bindgen's `snippets/`, from an `inline_js`) is copied, as the .py and .ps1 twins do.
for src in "${DIST}"/*; do
  if [ -d "${src}" ]; then
    cp -r "${src}" "${SITE}"/
  else
    cp -l "${src}" "${SITE}"/ 2>/dev/null || cp "${src}" "${SITE}"/
  fi
done

# The bundled third-party addons (`benilla_formats::bundled_addons`), served as loose files under
# `/addons/`, with `index.txt` naming each one: a static origin lists no directory, and the client
# resolves a path in any case against that list, as the reference's Windows filesystem does.
ADDONS=third_party/addons
if [ -d "${ADDONS}" ]; then
  mkdir -p "${SITE}/addons"
  cp -r "${ADDONS}"/. "${SITE}/addons"/
  (cd "${SITE}/addons" && find . -type f ! -name index.txt | sed 's|^\./||' | LC_ALL=C sort > index.txt)
  echo "staged $(wc -l < "${SITE}/addons/index.txt") bundled addon files"
fi

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
  # Not `split -d --additional-suffix`: those are GNU-only and BSD split (macOS) refuses them. The
  # default `aa`, `ab`, ... suffixes both share are renamed in order: wenilla_bg.part00.wasm, ...
  split -b "${WASM_PART_MAX}" "${SITE}/wenilla_bg.wasm" "${SITE}/wenilla_bg.split."
  i=0
  for chunk in "${SITE}"/wenilla_bg.split.*; do
    mv "${chunk}" "$(printf '%s/wenilla_bg.part%02d.wasm' "${SITE}" "${i}")"
    i=$((i + 1))
  done
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
# it fails minutes into an upload rather than here. `-maxdepth 1` on
# purpose — the packs under `data/` are capped by the packer (measured max 9.49 MiB) and walking a
# junction over 4,700 files to re-prove that is not worth the seconds.
oversize=$(find "${SITE}" -maxdepth 1 -type f -size +"${WASM_PART_MAX}"c)
if [ -n "${oversize}" ]; then
  echo "error: asset over a static host's 25 MiB limit (cap here is 24 MiB):" >&2
  echo "${oversize}" >&2
  exit 1
fi

# $1 = the container directory, $2 = the name it takes under ${SITE} (`data`, `data-hd`,
# `data-turtle`, `data-reforged`).
link_data() {
  src="$1"
  dst="${SITE}/$2"
  case "${STAGE_PACKS_LINK:-dir}" in
    hardlink)
      rm -rf "${dst}"
      mkdir -p "${dst}"
      # -r follows the tree, -l hardlinks the leaves: no bytes move.
      cp -lr "${src}"/. "${dst}"/
      ;;
    *)
      [ -e "${dst}" ] && return 0
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
            "$(cygpath -w "${dst}")" "$(cygpath -w "${src}")" >/dev/null
          ;;
        *)
          ln -s "../$(basename "${src}")" "${dst}"
          ;;
      esac
      ;;
  esac
}
[ -d "${PACKS}" ] && link_data "${PACKS}" data
[ -d "${PACKS_HD}" ] && link_data "${PACKS_HD}" data-hd
[ -d "${PACKS_TURTLE}" ] && link_data "${PACKS_TURTLE}" data-turtle
[ -d "${PACKS_REFORGED}" ] && link_data "${PACKS_REFORGED}" data-reforged

# A typical static host serves at most 20,000 assets per version, and four containers plus
# the page are counted together. Counted off the container trees themselves (not through the
# links), so a staged junction is walked by nothing but this line.
objects=$(find "${SITE}" -maxdepth 1 -type f | wc -l)
for tree in "${PACKS}" "${PACKS_HD}" "${PACKS_TURTLE}" "${PACKS_REFORGED}"; do
  [ -d "${tree}" ] && objects=$((objects + $(find "${tree}" -type f | wc -l)))
done
echo "assets to upload: ${objects} (a typical static host's limit: 20000)"
if [ "${objects}" -gt 20000 ]; then
  echo "error: over a static host's 20,000-asset limit — drop a layer or rebuild with fewer objects" >&2
  exit 1
fi

echo "staged ${SITE}:"
ls -la "${SITE}"
