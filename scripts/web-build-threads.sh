#!/usr/bin/env bash
#
# web-build-threads.sh — the browser build WITH THREADS.
#
# Same Trunk build as `trunk build --release`, same `web/index.html`, same output directories. The
# only difference is the environment, and this script exists so that environment has one home
# instead of living in a README paragraph that drifts.
#
# `scripts/web-build-threads.ps1` (Windows PowerShell 5.1) and `scripts/web-build-threads.py` are the
# same build; the three change together.
#
# What each variable is for:
#
#   RUSTUP_TOOLCHAIN        nightly, and not by preference: `-Zbuild-std` is nightly-only, and it is
#                           not optional — the `rust-std-wasm32-unknown-unknown` component rustup
#                           ships is compiled WITHOUT atomics, so std itself has to be rebuilt from
#                           source with the same target features as everything else. `rust-toolchain.toml`
#                           stays on its stable pin: native builds and the whole test suite are unaffected
#                           by any of this, and pinning the tree to nightly to serve one target would
#                           be the tail wagging the dog.
#   CARGO_UNSTABLE_BUILD_STD the config-file key `[unstable] build-std` as an environment override,
#                           which is how it reaches the `cargo` that TRUNK spawns. A `.cargo/config.toml`
#                           would do the same thing but would also be read by every stable
#                           invocation in the tree, where `[unstable]` has no business being.
#   RUSTFLAGS               `+atomics,+bulk-memory,+mutable-globals` is the shared-memory feature
#                           triple; `--shared-memory` and `--max-memory` are passed through to
#                           `wasm-ld` (see the note on max-memory below). These flags change the
#                           fingerprint of every crate in the graph, so this build does not share a
#                           `target/` directory with the plain one — that is why CARGO_TARGET_DIR is
#                           set, and it is a feature: switching back and forth does not rebuild
#                           760 dependencies each way.
#   WASI_SDK                the C sysroot for the Lua VM. `third_party/lua-src` reads the same
#                           `+atomics` out of `CARGO_CFG_TARGET_FEATURE` and switches itself to the
#                           SDK's `wasm32-wasip1-threads` archives; see that crate's BENILLA.md.
#
# ## --max-memory is a real decision, not a formality
#
# Shared memory must declare a maximum at link time, the engine reserves that much address space up
# front, and it can never grow past it. The corpus does NOT live in linear memory:
# `benilla_formats::webpack::resident` keeps all 3.3 GiB of decoded pack body on the JS heap
# precisely so that wasm's 32-bit address space stays the renderer's.
#
# **4 GiB — 65536 pages, the whole wasm32 address space — and it was 2 GiB until a session hit it.**
# World entry and a fast run across the map aborted every async-compute worker at once with a bare
# `RuntimeError: unreachable`; the frames under the trap were `rust_oom` ← `RawVec::handle_error` ←
# `grow_amortized`, i.e. a `Vec` that could not grow because the arena was at its ceiling. The
# out-of-memory abort never reaches a panic hook, which is why it arrives with no message
# (`web/threads.js` now prints linear memory beside a worker failure for exactly this). The ceiling
# was not the whole story — `benilla_assets::frame_budget::offload` had no bound on decodes in
# flight, and now does — so a session near this limit again is a leak or a new burst, not a reason
# to reach for a bigger number: there is none. Reserving it costs address space only, not memory.
#
# Past 2 GiB a pointer's top bit is set, so any JS that takes one as a signed `i32` sees it negative.
# The wasm-bindgen glue already writes `ptr >>> 0` everywhere; the hand-written imports must too
# (`web/wasi_stubs.js`), and Rust hands pointers to the page as `f64::from(u32)`, never as `i32`.
#
# ## The verification at the end is not optional
#
# Every failure mode this build has is silent. A module that lost `--shared-memory` still loads and
# still runs — single-threaded, with the worker spawn failing in a way the client is designed to
# survive — so "it works" is not evidence that threads are on. The check below reads the emitted
# module and says what is actually in it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Overridable, and worth pinning to a date the moment a nightly regression bites — the value that
# built a shipped bundle belongs in the commit that shipped it:
#   rustup toolchain install nightly-YYYY-MM-DD
#   BENILLA_NIGHTLY=nightly-YYYY-MM-DD scripts/web-build-threads.sh
export RUSTUP_TOOLCHAIN="${BENILLA_NIGHTLY:-nightly}"
export CARGO_UNSTABLE_BUILD_STD="std,panic_abort"
export WASI_SDK="${WASI_SDK:-$PWD/tools/wasi-sdk}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target/threads}"

MAX_MEMORY="${BENILLA_MAX_MEMORY:-4294967296}"
# `--shared-memory` and `--import-memory` are BOTH required and rustc supplies NEITHER. Measured on
# the 2026-09-06 spike, which passed only `--max-memory`: the module came out `flags=0x01
# shared=no` with a DEFINED memory, and both halves of that are fatal. An unshared memory is an
# ordinary ArrayBuffer no worker can attach to; a *defined* memory is created fresh by every
# instantiation, so each worker would get its own heap and share nothing. `--import-memory` is what
# makes the page hand the same `WebAssembly.Memory` to every instance, which is the whole
# mechanism. `--export-memory` keeps it readable from the glue as `wasm.memory`, which
# `web/index.html` passes to `wasi_stubs.js`. Verify with the check at the bottom of this script,
# never by reading these flags back.
#
# The five `--export=`s are for **wasm-bindgen**, not for the page. Its threading transform has to
# place a stack and a TLS block for every new thread inside the module's own memory, and it does
# that by reading linker-defined globals out of the export section — `__heap_base` to find the top
# of static data, `__wasm_init_tls` / `__tls_base` / `__tls_size` / `__tls_align` to set a thread up
# once it starts. rustc does not export any of them by default, and the failure is at
# `wasm-bindgen` rather than at the link: "failed to prepare module for threading ... failed to find
# `__heap_base`", then the same for each of the others in turn. Measured 2026-09-06: this is the
# minimal set — dropping any one of the five reintroduces exactly that error naming it.
export RUSTFLAGS="${RUSTFLAGS:-} -Ctarget-feature=+atomics,+bulk-memory,+mutable-globals -Clink-arg=--shared-memory -Clink-arg=--import-memory -Clink-arg=--export-memory -Clink-arg=--max-memory=${MAX_MEMORY} -Clink-arg=--export=__heap_base -Clink-arg=--export=__wasm_init_tls -Clink-arg=--export=__tls_base -Clink-arg=--export=__tls_size -Clink-arg=--export=__tls_align"

echo "== toolchain"
rustc -vV | sed 's/^/   /'
echo "   RUSTFLAGS=${RUSTFLAGS}"
echo "   CARGO_TARGET_DIR=${CARGO_TARGET_DIR}"

# `rust-src` is what `-Zbuild-std` compiles std out of; without it the failure is a wall of
# "can't find crate for `core`" that names nothing.
if ! rustup component list --toolchain "${RUSTUP_TOOLCHAIN}" --installed | grep -q '^rust-src$'; then
  echo "installing rust-src for ${RUSTUP_TOOLCHAIN} (-Zbuild-std needs std's sources)"
  rustup component add rust-src --toolchain "${RUSTUP_TOOLCHAIN}"
fi

echo "== trunk build --release"
trunk build --release

echo "== stage-packs"
scripts/stage-packs.sh

# ---------------------------------------------------------------------------------------------
# Verify the module actually came out threaded. Reads the build's own output, never the flags that
# were meant to produce it.
# ---------------------------------------------------------------------------------------------
WASM=$(ls web/dist/wenilla_bg*.wasm 2>/dev/null | head -1 || true)
if [ -z "${WASM}" ]; then
  echo "no wenilla_bg*.wasm in web/dist — the build did not get that far" >&2
  exit 1
fi

echo "== verify: $(basename "${WASM}") ($(du -h "${WASM}" | cut -f1))"
REPORT=$(scripts/wasm-memory.sh "${WASM}")
echo "${REPORT}" | sed 's/^/   /'

# All three have to hold. Each is a different way to end up single-threaded without an error.
fail=0
case "${REPORT}" in *shared=YES*) ;; *)
  echo "   FAIL: linear memory is not shared — no worker can attach to it." >&2; fail=1 ;;
esac
case "${REPORT}" in *"IMPORTED memory"*) ;; *)
  echo "   FAIL: linear memory is DEFINED, not imported — every instance would make its own heap" >&2
  echo "         and the workers would share nothing with the page." >&2; fail=1 ;;
esac
case "${REPORT}" in *+atomics*) ;; *)
  echo "   FAIL: no +atomics in target_features — RUSTFLAGS did not reach the cargo Trunk spawns." >&2; fail=1 ;;
esac
[ "${fail}" -eq 0 ] || exit 1
echo "   OK: shared, imported, atomics — this module can take worker threads."

# ---------------------------------------------------------------------------------------------
# BENILLA_KEEP_NAMES=1 — rebuild the glue from the cargo artifact WITHOUT wasm-opt, so the module
# keeps its `name` section and a stack trace names Rust functions instead of `$func55320`.
#
# This exists because of how the first main-thread `Atomics.wait` trap was found: the shipped module
# is `-O3 --strip-debug`, so the browser could only report wasm function INDICES, and working out
# which lock had blocked took a source audit rather than a stack read. A threading bug that only
# reproduces on a real GPU is exactly the bug you cannot afford to debug that way.
#
# The result is much larger and much slower — it is for localhost and a console, not for a site.
# `stage-packs.sh` has already hardlinked `web/site` to the optimized build; this only replaces
# what `--www web/dist` serves.
# ---------------------------------------------------------------------------------------------
if [ "${BENILLA_KEEP_NAMES:-}" = "1" ]; then
  echo "== BENILLA_KEEP_NAMES=1: re-emitting unoptimized, named glue"
  ARTIFACT="${CARGO_TARGET_DIR}/wasm32-unknown-unknown/release/wenilla.wasm"
  if [ ! -f "${ARTIFACT}" ]; then
    echo "   no cargo artifact at ${ARTIFACT}" >&2
    exit 1
  fi
  # The wasm-bindgen Trunk downloaded, whose version MUST match the crate pin (Trunk.toml).
  WBG=""
  for cand in     "${LOCALAPPDATA:-$HOME/AppData/Local}/trunkrs/trunk/cache/wasm-bindgen-0.2.126/wasm-bindgen.exe"     "$HOME/.cache/trunk/wasm-bindgen-0.2.126/wasm-bindgen"     "$(command -v wasm-bindgen 2>/dev/null || true)"; do
    [ -n "${cand}" ] && [ -x "${cand}" ] && WBG="${cand}" && break
  done
  if [ -z "${WBG}" ]; then
    echo "   no wasm-bindgen 0.2.126 found (Trunk caches one; or cargo install wasm-bindgen-cli)" >&2
    exit 1
  fi
  "${WBG}" --target web --out-dir web/dist --out-name wenilla --no-demangle "${ARTIFACT}"
  # Same rewrite `stage-packs.sh` does, for the same reason — workers have no import map.
  sed 's|from "wasi_snapshot_preview1"|from "./wasi_stubs.js"|g' web/dist/wenilla.js > web/dist/wenilla.js.tmp
  cat web/dist/wenilla.js.tmp > web/dist/wenilla.js
  rm -f web/dist/wenilla.js.tmp
  echo "   web/dist now carries the NAMED module ($(du -h web/dist/wenilla_bg.wasm | cut -f1)); web/site still has the optimized one."
fi

cat <<'EOF'

built. Serve it somewhere cross-origin isolated — the page cannot construct a SharedArrayBuffer
without COOP/COEP, and without one this module does not instantiate at all:

  cargo run --release -p wenilla-host -- --www web/dist --packs web/packs

(`wenilla-host::static_site` and `Trunk.toml`'s `[serve] headers` both set the same pair
`web/_headers` sets on the site, so all three agree.)
EOF
