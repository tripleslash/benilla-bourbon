#!/usr/bin/env bash
#
# perf-baseline.sh — one perf run of the native world viewer at a pinned scene, the same way every
# time, so a run before a dependency bump and a run after it differ only in the binary.
#
#   scripts/perf-baseline.sh <label> <scene> <vulkan|dx12> [KEY=VAL ...]
#   scripts/perf-baseline.sh scenes        # list the scenes
#
# It starts `benilla-worldview` with the pins below plus any KEY=VAL, waits for the probe's
# `PERF_PROBE cfg=` line (`WOW_PERF` samples and keeps running), then kills it. The full log is
# `<out>/<label>.log`; `<out>/<label>.txt` keeps only the instrument lines:
#
#   PERF_PROBE     wall frame time over WOW_PERF frames once the streamer is quiet (perf_probe.rs)
#   PERF_PASSES    per-pass GPU ms, costliest first (WOW_GPU_PASSES=1)
#   PERF_OVERLAP   async-queue passes beside the main queue
#   PERF_*, GPU_STATS   pipeline statistics, only with WOW_GPU_STATS=1 (async compute off)
#   GX_PERF, GX_CULL, GX_MESHLET   the static pass's CPU cost and its cull lanes (WOW_GX_PERF=1)
#   SYSTEM_PROFILE ms/s per system, only from a `--features trace_systems` build
#
# A row counts only with `settled=yes railed=no` and the backend asked for in its `gpu=… [..]`;
# the verdict line says which, and any error, validation message or panic is kept.
#
# Env: EXE (the binary, default target/release/benilla-worldview.exe), PERF_OUT (default
# target/perf-baseline), FRAMES (600), TIMEOUT (420 s), WOW_DATA (the TurtleWoW 1.18.1 install the
# earlier tables used).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# scene: map | x,y,z ("" = the viewer's own Northshire spawn) | face | pitch (degrees)
scene_row() {
  case "$1" in
    northshire) echo "0|-8980,-160,95|30|-8" ;;
    goldshire)  echo "0|-9400,60,72|180|-8" ;;
    stormwind)  echo "0|-8798.70,478.64,95.79|0|" ;;
    bootybay)   echo "0|-14412,446,13|55|-6" ;;
    darnassus)  echo "1|9935,2610,1352|-90|-14" ;;
    *) return 1 ;;
  esac
}

if [ "${1:-}" = scenes ]; then
  for s in northshire goldshire stormwind bootybay darnassus; do echo "$s $(scene_row $s)"; done
  exit 0
fi
[ $# -ge 3 ] || { sed -n '3,8p' "$0"; exit 2; }
label="$1"; scene="$2"; backend="$3"; shift 3
row="$(scene_row "$scene")" || { echo "unknown scene: $scene (scripts/perf-baseline.sh scenes)"; exit 2; }
IFS='|' read -r map at face pitch <<<"$row"
case "$backend" in vulkan|dx12) ;; *) echo "backend: vulkan or dx12"; exit 2 ;; esac

out="${PERF_OUT:-target/perf-baseline}"
mkdir -p "$out"
log="$out/$label.log"
exe="${EXE:-target/release/benilla-worldview.exe}"
[ -x "$exe" ] || { echo "no binary at $exe"; exit 2; }

# The pins: Cinematic modern lighting at noon, 1080p, uncapped, foreground (a background run is
# throttled to 16.67 ms on Windows, bgwin.rs), silent, no Streamline hook on the swap chain.
envs=(WOW_DATA="${WOW_DATA:-C:/TurtleWoW-1.18.1/Data}" WOW_NOSOUND=1 WOW_UNATTENDED=1
      WOW_BG=0 WOW_NOVSYNC=1 WOW_WIN=1920x1080 WOW_LIGHTING=modern WOW_LIGHTING_QUALITY=4
      WOW_CLOCK=720 WOW_STREAMLINE=0 WOW_PERF="${FRAMES:-600}" WOW_PERF_LABEL="$label"
      WOW_GPU_PASSES=1 WOW_GX_PERF=1 WGPU_BACKEND="$backend" WOW_MAP="$map"
      RUST_LOG="${RUST_LOG:-info,wgpu_core=warn,wgpu_hal=warn,naga=warn}")
[ -n "$at" ] && envs+=(WOW_WORLDVIEW_AT="$at")
[ -n "$face" ] && envs+=(WOW_WORLDVIEW_FACE="$face")
[ -n "$pitch" ] && envs+=(WOW_WORLDVIEW_PITCH="$pitch")

start=$(date +%s)
env "${envs[@]}" "$@" "$exe" >"$log" 2>&1 &
pid=$!
deadline=$((start + ${TIMEOUT:-420}))
got=""
while kill -0 "$pid" 2>/dev/null; do
  if grep -aq "PERF_PROBE cfg=" "$log"; then got=1; sleep 3; break; fi
  [ "$(date +%s)" -ge "$deadline" ] && break
  sleep 2
done
# Git Bash: the Windows pid is what taskkill needs; elsewhere the plain kill does it.
if [ -r "/proc/$pid/winpid" ]; then taskkill //F //PID "$(cat "/proc/$pid/winpid")" >/dev/null 2>&1; fi
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
secs=$(($(date +%s) - start))

clean() { sed -E 's/\x1b\[[0-9;]*m//g; s/^[0-9TZ:.-]+ +//'; }
probe="$(grep -a "PERF_PROBE cfg=" "$log" | tail -1 | clean)"
want="Vulkan"; [ "$backend" = dx12 ] && want="Dx12"
verdict="NO RESULT"
if [ -n "$got" ]; then
  verdict="valid"
  case "$probe" in *"settled=yes"*) ;; *) verdict="INVALID (not settled)" ;; esac
  case "$probe" in *"railed=no"*) ;; *) verdict="INVALID (railed)" ;; esac
  case "$probe" in *"[$want]"*) ;; *) verdict="INVALID (backend is not $want)" ;; esac
fi
{
  echo "### $label scene=$scene backend=$backend ${*:-} (${secs}s) $verdict"
  grep -aE "caps:|PERF_|GPU_STATS|perf probe:|panic|ERROR|Validation|validation|failed to process shader" "$log" \
    | clean | awk '!seen[$0]++'
  for t in GX_PERF "GX_CULL world" "GX_CULL mirror" "GX_MESHLET"; do grep -a "$t" "$log" | tail -1 | clean; done
  # The profiler reports every 3 s; the last report before the kill is the one sampled at rest.
  grep -aE "SYSTEM_PROFILE|SYSTEM_SPIKE" "$log" | clean \
    | awk '/SYSTEM_PROFILE wall=/{block=""} {block=block $0 "\n"} END{printf "%s", block}'
} >"$out/$label.txt"
# Written first, printed after: a reader that closes the pipe early must not cut the file short.
cat "$out/$label.txt"
