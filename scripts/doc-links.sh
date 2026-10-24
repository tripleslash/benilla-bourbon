#!/usr/bin/env bash
#
# doc-links.sh — fail on a doc link whose target is declared nowhere in the workspace.
# The bash twin of scripts/doc-links.py; that header holds the why, and the twins change together.
#
#   scripts/doc-links.sh            # the gate: exit 1 on any dead-target link
#   scripts/doc-links.sh --report   # + the benign counts
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

report=0
for a in "$@"; do
  case "$a" in
    -h | --help)
      sed -n '3,7s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    --report) report=1 ;;
  esac
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/benilla-doc-links.XXXXXX")" || exit 1
trap 'rm -rf "$tmp"' EXIT

cargo doc --workspace --no-deps --document-private-items >"$tmp/doc.log" 2>&1

# Every name the workspace declares, in one scan. `fn` has its own pattern: in the alternation,
# `const fn foo` captures `fn` and never indexes `foo`.
{
  grep -rhoE --include='*.rs' '\bfn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' crates
  grep -rhoE --include='*.rs' '\b(struct|enum|trait|type|const|static|union|mod)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' crates
  grep -rhoE --include='*.rs' '\bmacro_rules![[:space:]]+[A-Za-z_][A-Za-z0-9_]*' crates
} | sed 's/.*[[:space:]]//' | grep -vx 'fn' | LC_ALL=C sort -u >"$tmp/names"

# Each unresolved-link warning paired with the `-->` line after it. A local path whose leaf is an
# identifier is a candidate (`leaf<TAB>target<TAB>path<TAB>line`); everything else is benign.
warn_re='warning: unresolved link to `([^`]+)`'
at_re='^[[:space:]]*--> ([^[:space:]:]+):([0-9]+)'
local_re='^(crate|super|self|Self)::'
ident_re='^[A-Za-z_][A-Za-z0-9_]*$'
found=0
pending=""
: >"$tmp/candidates"
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'\r'}"
  if [[ $line =~ $warn_re ]]; then
    pending="${BASH_REMATCH[1]}"
    continue
  fi
  [ -n "$pending" ] || continue
  if [[ $line =~ $at_re ]]; then
    path="${BASH_REMATCH[1]}"
    lineno="${BASH_REMATCH[2]}"
    target="$pending"
    pending=""
    found=$((found + 1))
    [[ $target =~ $local_re ]] || continue
    leaf="${target##*::}"
    leaf="${leaf%%(*}"
    leaf="${leaf#"${leaf%%[![:space:]]*}"}"
    leaf="${leaf%"${leaf##*[![:space:]]}"}"
    [[ $leaf =~ $ident_re ]] || continue
    printf '%s\t%s\t%s\t%s\n' "$leaf" "$target" "$path" "$lineno" >>"$tmp/candidates"
  elif [[ ! $line =~ ^[[:space:]]*$ ]]; then
    # Only blank lines may stand between a warning and its `-->`.
    pending=""
  fi
done <"$tmp/doc.log"

# The leaves declared nowhere, in one pass over the name list.
cut -f1 "$tmp/candidates" | LC_ALL=C sort -u | grep -vxFf "$tmp/names" >"$tmp/missing" || true
dead=0
dead_lines=""
[ -s "$tmp/missing" ] && while IFS=$'\t' read -r leaf target path lineno; do
  if grep -qxF -- "$leaf" "$tmp/missing"; then
    dead=$((dead + 1))
    dead_lines+="  ${path}:${lineno}"$'\n'"      [\`${target}\`]"$'\n'
  fi
done <"$tmp/candidates"
benign=$((found - dead))

if [ "$report" -eq 1 ]; then
  echo "doc-links: ${found} unresolved-link warnings"
  echo "doc-links: ${benign} name something that still exists (see this file's header)"
fi

if [ "$dead" -eq 0 ]; then
  echo "doc-links ok: no doc link points at a deleted item (${found} benign)"
  exit 0
fi

echo "doc-links FAILED: ${dead} doc link(s) point at something that no longer exists:"
echo
printf '%s' "$dead_lines"
echo
echo "Each names an item declared nowhere in the workspace. Repoint it at whatever replaced"
echo "the thing, or drop the brackets — a dangling link promises navigation that is not there."
exit 1
