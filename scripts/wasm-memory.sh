#!/usr/bin/env bash
#
# wasm-memory.sh — say what a wasm module's linear memory is, and which target features it was
# built with. The bash twin of scripts/wasm-memory.py; that header holds the why, and the twins
# change together. Only the section headers and the import, memory and target_features sections are
# read (through `od`), so a 90 MB module is never loaded whole.
#
#   scripts/wasm-memory.sh web/dist/wenilla_bg.wasm
set -euo pipefail

f="${1:-}"
if [ -z "${f}" ] || [ ! -f "${f}" ]; then
  echo "usage: wasm-memory.sh <module.wasm>" >&2
  exit 1
fi
total=$(wc -c < "${f}" | tr -d ' ')

# The window of bytes being parsed, one decimal per element, and the cursor into it. The readers
# return through V rather than stdout, so no byte costs a subshell.
B=()
P=0
V=0
load() { # $1 = offset, $2 = length
  B=($(od -An -v -tu1 -j "$1" -N "$2" "${f}"))
  P=0
}
u8() {
  if [ "${P}" -ge "${#B[@]}" ]; then
    echo "unexpected end of module" >&2
    exit 1
  fi
  V=${B[${P}]}
  P=$((P + 1))
}
uleb() {
  local r=0 s=0 b
  while :; do
    u8
    b=${V}
    r=$((r | ((b & 0x7f) << s)))
    if [ $((b & 0x80)) -eq 0 ]; then
      break
    fi
    s=$((s + 7))
  done
  V=${r}
}
name() {
  uleb
  local n=${V} i o c s=""
  for ((i = 0; i < n; i++)); do
    u8
    printf -v o '%03o' "${V}"
    printf -v c "\\${o}"
    s+="${c}"
  done
  V=${s}
}

LF=0
LMIN=0
LMAX=None
limits() {
  u8
  LF=${V}
  uleb
  LMIN=${V}
  LMAX=None
  if [ $((LF & 1)) -ne 0 ]; then
    uleb
    LMAX=${V}
  fi
}
describe() {
  local shared=no mx=None
  if [ $((LF & 2)) -ne 0 ]; then
    shared=YES
  fi
  if [ "${LMAX}" != None ]; then
    mx="${LMAX} pages ($((LMAX * 64 / 1024)) MiB)"
  fi
  printf 'flags=0x%02x shared=%s min=%s pages (%s MiB) max=%s' \
    "${LF}" "${shared}" "${LMIN}" "$((LMIN * 64 / 1024))" "${mx}"
}

found=0
imports() {
  uleb
  local count=${V} i m n kind
  for ((i = 0; i < count; i++)); do
    name
    m=${V}
    name
    n=${V}
    u8
    kind=${V}
    case ${kind} in
      0) uleb ;;
      1) u8; limits ;;
      2)
        limits
        found=1
        echo "IMPORTED memory  ${m}.${n}  $(describe)"
        ;;
      3) u8; u8 ;;
      # A tag (the exception-handling proposal): unread, the walk would lose its place.
      4) u8; uleb ;;
    esac
  done
}
memories() {
  uleb
  local count=${V} i
  for ((i = 0; i < count; i++)); do
    limits
    found=1
    echo "DEFINED memory   $(describe)"
  done
}
features() {
  uleb
  local cnt=${V} i pfx list=""
  for ((i = 0; i < cnt; i++)); do
    u8
    pfx=${V}
    name
    if [ "${pfx}" -eq 43 ]; then
      list+="+${V}"$'\n'
    else
      list+="-${V}"$'\n'
    fi
  done
  echo "target_features: $(printf '%s' "${list}" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')"
}

load 0 8
if [ "${B[*]:0:4}" != "0 97 115 109" ]; then
  echo "not a wasm module" >&2
  exit 1
fi
off=8
while [ "${off}" -lt "${total}" ]; do
  load "${off}" 11
  u8
  sid=${V}
  uleb
  size=${V}
  start=$((off + P))
  case ${sid} in
    2) load "${start}" "${size}"; imports ;;
    5) load "${start}" "${size}"; memories ;;
    0)
      # A custom section: only the name is read, unless it is target_features.
      load "${start}" $((size < 64 ? size : 64))
      uleb
      if [ "${V}" -eq 15 ] && [ $((P + 15)) -le "${#B[@]}" ]; then
        load "${start}" "${size}"
        name
        if [ "${V}" = target_features ]; then
          features
        fi
      fi
      ;;
  esac
  off=$((start + size))
done
if [ "${found}" -eq 0 ]; then
  echo "no memory declaration found"
fi
