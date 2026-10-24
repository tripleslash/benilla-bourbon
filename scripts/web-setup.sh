#!/usr/bin/env bash
#
# web-setup.sh - one-time toolchain for the browser build (no sudo, nothing system-wide):
#   - rustup target wasm32-unknown-unknown
#   - trunk (the build; it downloads its OWN wasm-bindgen and wasm-opt, pinned in Trunk.toml)
#   - wasi-sdk (a C sysroot for wasm32: the Lua 5.1 VM is C) unpacked into tools/wasi-sdk/
#   - binaryen, only as the local wasm-opt for anyone still invoking it by hand
#
# Re-runnable; each step is skipped when already done. WASI_SDK_VERSION overrides the release.
#
# The wasm-bindgen-cli install this used to do is GONE. Trunk owns that version now (Trunk.toml's
# `[tools] wasm_bindgen`), which removes the failure this script existed to prevent: a CLI a version
# off from the `wasm-bindgen = "=0.2.126"` crate emits glue that does not match, and the symptom is
# a page that loads and then does nothing. Keep the two numbers equal if either moves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WASI_SDK_VERSION="${WASI_SDK_VERSION:-27}"
WASI_DIR="tools/wasi-sdk"

# The host triple in the naming schemes wasi-sdk and binaryen both use. Windows is here because the
# project's own dev machine is one (`uname -s` = MINGW64_NT-...) and this script used to `exit 1` on
# it, which blocked the entire browser build on that machine.
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   plat=x86_64-linux;  bplat=x86_64-linux ;;
  Linux-aarch64)  plat=arm64-linux;   bplat=aarch64-linux ;;
  Darwin-arm64)   plat=arm64-macos;   bplat=arm64-macos ;;
  Darwin-x86_64)  plat=x86_64-macos;  bplat=x86_64-macos ;;
  MINGW*|MSYS*|CYGWIN*)
                  plat=x86_64-windows; bplat=x86_64-windows ;;
  *) plat=""; bplat="" ;;
esac

echo "== rustup target"
rustup target add wasm32-unknown-unknown

echo "== trunk"
if command -v trunk >/dev/null; then
  trunk_version="$(trunk --version)"
  echo "trunk ${trunk_version#* } already installed"
else
  cargo install trunk --locked
fi

echo "== wasi-sdk ${WASI_SDK_VERSION}"
if [ -f "${WASI_DIR}/share/wasi-sysroot/lib/wasm32-wasip1/libsetjmp.a" ]; then
  echo "already at ${WASI_DIR}"
elif [ -z "${plat}" ]; then
  echo "unsupported host $(uname -s)-$(uname -m); set WASI_SDK to an unpacked SDK instead" >&2
  exit 1
else
  url="https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_SDK_VERSION}/wasi-sdk-${WASI_SDK_VERSION}.0-${plat}.tar.gz"
  mkdir -p "${WASI_DIR}"
  echo "downloading ${url}"
  curl -sL "${url}" | tar xz --strip-components=1 -C "${WASI_DIR}"
fi

echo "== binaryen (Trunk fetches its own wasm-opt; this is for invoking it by hand)"
BINARYEN_VERSION="${BINARYEN_VERSION:-130}"
if [ -x tools/binaryen/bin/wasm-opt ] || [ -x tools/binaryen/bin/wasm-opt.exe ]; then
  echo "already at tools/binaryen"
elif [ -n "${bplat}" ]; then
  mkdir -p tools/binaryen
  curl -sL "https://github.com/WebAssembly/binaryen/releases/download/version_${BINARYEN_VERSION}/binaryen-version_${BINARYEN_VERSION}-${bplat}.tar.gz" | tar xz --strip-components=1 -C tools/binaryen
else
  echo "  no binaryen release for this host — Trunk's own wasm-opt still runs"
fi

cat <<'EOF'

ready. The build is now:

  WASI_SDK=$PWD/tools/wasi-sdk trunk build --release      # -> web/dist
  scripts/stage-packs.sh                                  # -> web/site

or, for the build with worker threads (nightly + -Zbuild-std + shared memory; same output dirs,
and it verifies the emitted module really came out threaded):

  scripts/web-build-threads.sh

and the game data is a separate, far slower, one-time job:

  cargo run --release -p benilla-packer --bin benilla-pack -- \
      build --data "/path/to/WoW 1.12.1/Data" --out web/packs

Serve both locally with:

  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs
  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs --no-ranges  # as a static host behaves
EOF
