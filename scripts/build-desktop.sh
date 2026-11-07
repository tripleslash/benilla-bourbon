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
# Built for speed, where the web build cannot follow:
# - the `ship` profile: fat LTO over the whole program, one codegen unit;
# - `-C target-cpu=x86-64-v3`: AVX2, FMA, BMI2 (Haswell, 2013, and later). The binary will not
#   start on an older CPU; `BENILLA_DESKTOP_CPU=x86-64` builds the portable baseline instead;
# - optionally profile-guided (below).
#
# `--target` is named explicitly so the flags apply to the target alone and land in their own
# folder (`target/x86_64-pc-windows-msvc/`): the host build cache is left as it is.
#
#   BENILLA_DESKTOP_PROFILE=release   thin LTO, quicker to build (default: ship)
#   BENILLA_DESKTOP_CPU=<cpu>         the target CPU (default: x86-64-v3)
#
# Profile-guided optimisation, in three steps (needs `rustup component add llvm-tools`):
#   BENILLA_DESKTOP_PGO=generate scripts/build-desktop.sh   # an instrumented exe
#   target/desktop/wow-reforged.exe                          # play a representative session:
#                                                            # login, a city, a flight; then quit
#   BENILLA_DESKTOP_PGO=use scripts/build-desktop.sh         # merge the profiles, build with them
# The profiles land in target/desktop-pgo/; `generate` empties it first.

set -euo pipefail
cd "$(dirname "$0")/.."

triple=x86_64-pc-windows-msvc
profile="${BENILLA_DESKTOP_PROFILE:-ship}"
cpu="${BENILLA_DESKTOP_CPU:-x86-64-v3}"
pgo="${BENILLA_DESKTOP_PGO:-}"
pgo_dir="$(pwd)/target/desktop-pgo"

flags="-C target-feature=+crt-static -C target-cpu=${cpu}"
case "${pgo}" in
  "") ;;
  generate)
    rm -rf "${pgo_dir}"
    mkdir -p "${pgo_dir}"
    flags="${flags} -C profile-generate=${pgo_dir}"
    ;;
  use)
    sysroot="$(rustc --print sysroot)"
    profdata="$(find "${sysroot}" -name 'llvm-profdata*' -type f | head -1)"
    [ -n "${profdata}" ] || { echo "no llvm-profdata: rustup component add llvm-tools" >&2; exit 1; }
    ls "${pgo_dir}"/*.profraw >/dev/null 2>&1 \
      || { echo "no profiles in ${pgo_dir}: run a BENILLA_DESKTOP_PGO=generate build first" >&2; exit 1; }
    "${profdata}" merge -o "${pgo_dir}/merged.profdata" "${pgo_dir}"/*.profraw
    flags="${flags} -C profile-use=${pgo_dir}/merged.profdata -C llvm-args=-pgo-warn-missing-function"
    ;;
  *) echo "BENILLA_DESKTOP_PGO is generate or use, not '${pgo}'" >&2; exit 1 ;;
esac

export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="${flags}"
echo "== ${profile}, ${cpu}${pgo:+, pgo ${pgo}}: ${flags}"
cargo build -p benilla-desktop --no-default-features --profile "$profile" --target "$triple"

out="target/desktop"
mkdir -p "$out"
cp "target/$triple/$profile/wow-reforged.exe" "$out/wow-reforged.exe"
echo "built $out/wow-reforged.exe ($(wc -c < "$out/wow-reforged.exe") bytes)"
