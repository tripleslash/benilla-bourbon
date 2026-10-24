#!/usr/bin/env bash
# The in-loop verify: fmt everywhere, clippy and test on the changed crates and their dependents.
# Each scoped suite runs once; the second pass without the install (`test-no-install`) is the
# land's alone, in scripts/gates.sh. A change reaching `benilla-app` also runs the player build's
# schedule-shape tests, the part of the land's `player-tests` a dev-build round cannot see.
#
# The change set is the fork point against main plus staged, unstaged and untracked files. A path
# under crates/<dir>/ scopes that package, docs/ and *.md scope nothing, and any other path runs
# scripts/gates.sh, the workspace-wide chain run at land.
# CHECK_FULL=1 goes straight to gates.sh.
#
#   scripts/check.sh
set -uo pipefail

# Check the checkout you stand in when it is one of this repo, else this file's (as gates.sh does).
here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$root" ] && [ -f "$root/scripts/check.sh" ] || root="$here"
cd "$root" || exit 1

if [ "${CHECK_FULL:-}" = "1" ]; then
    exec "$root/scripts/gates.sh"
fi

# ── The round's change set ───────────────────────────────────────────────────────────────────────
base="$(git merge-base HEAD main 2>/dev/null || git rev-parse HEAD)"
changed="$( { git diff --name-only "$base" HEAD 2>/dev/null
              git diff --name-only HEAD 2>/dev/null
              git ls-files --others --exclude-standard 2>/dev/null; } | sort -u )"

if [ -z "$changed" ]; then
    echo "check: no changes against main and a clean tree — nothing to verify"
    exit 0
fi

# ── Map files → crate dirs; anything unmappable escalates to the full chain ──────────────────────
dirs=""
full_reason=""
while IFS= read -r f; do
    case "$f" in
    docs/* | *.md) ;;                                 # provably gate-inert (gates.sh's docs-only rule)
    crates/*/*) d="${f#crates/}"; dirs="$dirs DIR:crates/${d%%/*}" ;;
    *) full_reason="$f" ;;
    esac
done <<EOF
$changed
EOF

if [ -n "$full_reason" ]; then
    echo "check: '$full_reason' is outside the crate map — the whole workspace could be affected."
    echo "check: escalating to the full chain: scripts/gates.sh"
    exec "$root/scripts/gates.sh"
fi

dirlist="$(printf '%s\n' $dirs | sed 's/.*DIR://' | sort -u)"
if [ -z "$dirlist" ]; then
    echo "check: docs-only round ($(printf '%s\n' "$changed" | wc -l | tr -d ' ') file(s)) — no gate can change; fmt only"
    cargo fmt --all -- --check || exit 1
    echo "check: green (docs only)"
    exit 0
fi

# ── Changed crates → reverse-dependency closure, off cargo tree ──────────────────────────────────
# A changed dir with no package manifest is not our map's to scope.
names=""
missing=""
for d in $dirlist; do
    name="$(sed -n 's/^name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$d/Cargo.toml" 2>/dev/null | head -1)"
    if [ -n "$name" ]; then names="$names $name"; else missing="$missing $d"; fi
done
if [ -n "$missing" ]; then
    echo "check: changed path(s) not in the workspace map (${missing# }) — full chain"
    exec "$root/scripts/gates.sh"
fi
# `--invert` lists every workspace package that reaches the changed one, over every dependency
# kind and target, the package included.
pkgs=""
for name in $names; do
    dependents="$(cargo tree -q --workspace --invert "$name" --edges normal,build,dev --target all \
        --prefix none 2>/dev/null)" ||
        { echo "check: cargo tree failed — falling back to the full chain"; exec "$root/scripts/gates.sh"; }
    pkgs="$pkgs $(printf '%s\n' "$dependents" | sed -n 's/^\([A-Za-z0-9_-]*\) v[0-9].*/\1/p')"
done
pkgs="$(printf '%s\n' $pkgs | LC_ALL=C sort -u | tr '\n' ' ' | sed 's/ $//')"

pflags=""
for p in $pkgs; do pflags="$pflags -p $p"; done
echo "check: scope = $pkgs"
echo "check:   (changed crates + everything that depends on them; the full chain runs once, at land)"

log="$(mktemp "${TMPDIR:-/tmp}/benilla-check.XXXXXX")"
skips="$(mktemp "${TMPDIR:-/tmp}/benilla-check-skips.XXXXXX")"
trap 'rm -f "$log" "$skips"' EXIT
# Each gate logs to a file tailed afterwards: piping through `tail` would mask its exit code.
run() {
    local name="$1"; shift
    if ! "$@" >"$log" 2>&1; then
        echo "CHECK FAILED: $name"
        # The failing test or error first: a test gate's tail is the doc-test noise after it.
        grep -E 'panicked at|^---- .* ----$|^test .* FAILED$|^ +(FAIL|SIGSEGV|SIGABRT|SIGKILL|TIMEOUT) \[|^error(\[E[0-9]+\])?:' "$log" | head -40
        tail -30 "$log"
        exit 1
    fi
    echo "check ok: $name"
}

run fmt cargo fmt --all -- --check
run clippy cargo clippy $pflags --all-targets -- -D warnings
# The player build (no `dev` feature) is linted on its own: the run above sees only the dev set.
case " $pkgs " in *" benilla-app "* | *" benilla "*)
    run player-clippy cargo clippy -p benilla -p benilla-app --no-default-features -- -D warnings ;;
esac
# Every test binary runs whatever another fails (`scripts/unit-tests.sh`), and the doctests after.
run test env BENILLA_SKIP_LOG="$skips" "$root/scripts/unit-tests.sh" $pflags
# The player build's schedule, whose systems differ from the dev set's, checked by the tests that
# read its shape alone; the land's `player-tests` runs the rest of that build's suite.
case " $pkgs " in *" benilla-app "*)
    run player-schedule env BENILLA_SKIP_LOG="$skips" "$root/scripts/unit-tests.sh" \
        -p benilla-app --no-default-features --lib -- \
        vm_pass:: game_plugins::schedule_tests:: script_calls::tests::every_ lanes::guard_tests:: ;;
esac
# shellcheck disable=SC2086 # one word per package
run doctest "$root/scripts/doc-tests.sh" $pkgs
# Data-gated tests skip without the install or the addon corpus and libtest swallows their line,
# so each skip is also logged to $BENILLA_SKIP_LOG and counted here.
if [ -s "$skips" ]; then
    echo "  test: $(wc -l <"$skips" | tr -d ' ') data-gated tests SKIPPED on this machine —"
    sort "$skips" | uniq -c | sort -rn | sed 's/^ *\([0-9]*\) \(.*\)/    \1 × \2/'
    echo "    (they run where the data is: docs/CONTRIBUTING.md, \"Setting up\")"
fi

echo "CHECK GREEN (scoped: $pkgs)"
echo "  (the full chain, scripts/gates.sh, runs once on the tree that lands; a run before it repeats it)"
