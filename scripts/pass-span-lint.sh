#!/usr/bin/env bash
#
# pass-span-lint.sh — refuse a second open `pass_span` on the same render pass (a wgpu validation
# error on Vulkan). The bash twin of scripts/pass-span-lint.py; that header holds the why, and the
# twins change together.
#
#   scripts/pass-span-lint.sh [path …]      (default: every crates/**/*.rs)
#   Exit: 0 clean, 1 findings.
set -uo pipefail

open_re='(^|[^A-Za-z0-9_])let[[:space:]]+(mut[[:space:]]+)?([A-Za-z0-9_]+)[[:space:]]*=[[:space:]]*[A-Za-z0-9_.]+\.pass_span\([[:space:]]*&mut[[:space:]]+([A-Za-z0-9_]+)[[:space:]]*,'
any_open_re='\.pass_span\([[:space:]]*&mut[[:space:]]+([A-Za-z0-9_]+)[[:space:]]*,'
end_re='([A-Za-z0-9_]+)\.end\([[:space:]]*&mut[[:space:]]+([A-Za-z0-9_]+)[[:space:]]*\)'
fn_re='^[[:space:]]*(pub(\([^)]*\))?[[:space:]]+)?(async[[:space:]]+)?(unsafe[[:space:]]+)?fn[[:space:]]+[A-Za-z0-9_]+'

findings=0
lint() { # $1 = file
  local guards=() passes=() lines=() lineno=0 raw line guard pass_name i end_guard
  while IFS= read -r raw || [ -n "$raw" ]; do
    lineno=$((lineno + 1))
    line="${raw%%//*}"
    if [[ $line =~ $fn_re ]]; then
      guards=() passes=() lines=()
    fi
    guard=""
    pass_name=""
    if [[ $line =~ $open_re ]]; then
      guard="${BASH_REMATCH[3]}"
      pass_name="${BASH_REMATCH[4]}"
    elif [[ $line =~ $any_open_re ]]; then
      pass_name="${BASH_REMATCH[1]}"
    fi
    if [ -n "$pass_name" ]; then
      for ((i = 0; i < ${#passes[@]}; i++)); do
        if [ "${passes[$i]}" = "$pass_name" ]; then
          echo "$1:$lineno: a second pass_span on \`$pass_name\` while \`${guards[$i]}\` (line ${lines[$i]}) is still open — one pipeline-statistics query at a time"
          findings=$((findings + 1))
        fi
      done
      if [ -n "$guard" ]; then
        guards+=("$guard")
        passes+=("$pass_name")
        lines+=("$lineno")
      fi
    fi
    if [[ $line =~ $end_re ]]; then
      end_guard="${BASH_REMATCH[1]}"
      local kg=() kp=() kl=()
      for ((i = 0; i < ${#guards[@]}; i++)); do
        if [ "${guards[$i]}" != "$end_guard" ]; then
          kg+=("${guards[$i]}")
          kp+=("${passes[$i]}")
          kl+=("${lines[$i]}")
        fi
      done
      guards=(${kg[@]+"${kg[@]}"})
      passes=(${kp[@]+"${kp[@]}"})
      lines=(${kl[@]+"${kl[@]}"})
    fi
  done <"$1"
}

[ "$#" -gt 0 ] || set -- crates
files=()
for a in "$@"; do
  if [ -d "$a" ]; then
    while IFS= read -r f; do
      files+=("$f")
    done < <(find "$a" -type f -name '*.rs' | LC_ALL=C sort)
  else
    files+=("$a")
  fi
done

# A file that never names `pass_span` cannot open one, so only those are walked line by line.
for f in ${files[@]+"${files[@]}"}; do
  if grep -q 'pass_span' "$f" 2>/dev/null; then
    lint "$f"
  fi
done

if [ "$findings" -eq 0 ]; then
  echo "pass-span-lint: clean — ${#files[@]} file(s)"
  exit 0
fi
echo "pass-span-lint: ${findings} finding(s) — ${#files[@]} file(s)"
exit 1
