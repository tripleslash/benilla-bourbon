#!/usr/bin/env bash
#
# perf-matrix.sh — the rendering perf rig's runbook, as a script rather than a wiki page.
#
# It does two things and nothing else: start the host the browser worldview reads from, and print
# the URL matrix that drives `benilla_world::perf_probe`. It deliberately does NOT drive a browser:
# there is no node/puppeteer on this machine, and a rig whose driver is a dependency nobody has is
# a rig nobody runs. You paste a URL, the tab prints one `PERF_PROBE` line, you move on.
#
# ## The one thing to get right before reading any number
#
# Decision 0717: while synced, wall frame time measures the DISPLAY'S PRESENT GRANT, not our cost.
# A browser tab cannot uncap vsync — WebGPU presents on requestAnimationFrame — so a frame cheaper
# than the refresh interval reads as exactly the refresh interval, and every ablation delta comes
# out as zero. The probe labels this itself (`railed=yes`) and warns.
#
# The lever is `&superscale=N`: render the same frame at N× linear resolution until `railed=no`,
# then ablate at that resolution. This is the tree's own instrument for the problem — see
# `benilla_app::world_backdrop::RENDER_SCALE_RANGE`'s note that supersampling is "the only way to
# price a pixel on a machine whose present is railed at the display's grant".
#
# Start at 1.0, and step 1.5 → 2.0 → 2.5 until the probe stops saying railed. Keep the value FIXED
# across a zone's whole ablation sweep; it is not comparable across zones.
#
# ## Usage
#
#   scripts/perf-matrix.sh serve              # start the host (remote packs), Ctrl-C to stop
#   scripts/perf-matrix.sh urls [superscale]  # print the matrix (default superscale 1.0)
#   scripts/perf-matrix.sh urls 2.0 stormwind # just one zone
#   scripts/perf-matrix.sh urls 2.0 all 3 0   # every zone at Ultra, at midnight
#
# The third and fourth arguments are the lighting tier (0 Low … 4 Cinematic, default 1 Medium) and
# the clock (minutes past midnight, default 720 noon). `0` lights the windows and every lamp, which
# is where the emissive-source tiers spend what they spend. `all` (or `-`) as the zone keeps every
# zone. The tier and clock ride the label (`stormwind-q3-t0-base`), so rows from two sweeps never
# read as the same configuration.
#
# Env: BENILLA_PERF_HOST, BENILLA_PACKS_REMOTE, BENILLA_PERF_FRAMES, BENILLA_PERF_PORT.
#
# ## The probe waits for the world, you do not
#
# Open the URL and leave the tab alone. The probe discards a flat warm-up, then waits for the
# terrain streamer to do NOTHING for 60 consecutive frames, and only then samples `perf=<n>`
# frames. Against remote packs that wait is long — the first Elwynn run was still streaming a
# minute in — which is exactly why the gate is a state and not a frame count.
#
# Two columns say whether the row is worth reading, and BOTH must be right:
#
#   settled=yes   the streamer was quiet before sampling started. `no` means the backstop fired
#                 and the number includes streaming work — discard it, or raise `&perf_warmup=`.
#   railed=no     the frame was slower than the present interval, so wall time is cost.
#                 `yes` means every ablation delta on that row is noise (see above).
#
# A row with `settled=no` or `railed=yes` is not a measurement. It is a note saying to take it
# again.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 8099, not 8090: a session running this should not have to stop the host the director already has
# on the default port, and a measurement taken against the wrong tree is worse than no measurement.
PORT="${BENILLA_PERF_PORT:-8099}"
HOST="${BENILLA_PERF_HOST:-http://127.0.0.1:$PORT}"
PACKS_REMOTE="${BENILLA_PACKS_REMOTE:-https://wow.example.com}"
FRAMES="${BENILLA_PERF_FRAMES:-600}"

# The zones. Every pin here is one the tree already verified and recorded — a director's `.go xyz`
# report or a capture scenario — rather than a coordinate typed from memory, because a wrong pin
# does not fail loudly: it puts the camera under the terrain and returns a plausible, meaningless
# number.
#
#   label | map | x,y,z | face | why this zone is in the matrix
ZONES=(
  # `frame_pace.rs:16`'s own measurement pin. The baseline every other reading is read against.
  "elwynn|0|-9481.53,76.10,56.57|-|open country, light art — the floor"
  # `wmo_portal/mod.rs:1352`, read off a live cull trace.
  "stormwind|0|-8798.70,478.64,95.79|-|city: static_gx runs, light bins, portal flood, residency — and INLAND, which is what prices the water mirror's idle cost"
  # `wmo_portal/audit/mod.rs:131`'s pin.
  "darnassus|1|10188.36,2348.35,1328.96|-|canopy: probe GI, canopy taps, sky taps — and where the black-tree-card bug lived"
  # `FORK.md:451`, the Darkshore reference framing.
  "darkshore|1|6330,640,8|95|coast: the planar mirror, height fog, low-sun shafts"
  # `wmo_portal/audit/mod.rs:100` — the Great Forge's lava grates.
  "ironforge|0|-4845,-995,421|-|interior: interior probe cascade, local-light grid, emissive"
  # `capture/scenarios.rs:436`.
  "felwood|1|4060.9,-944.3,256.8|-|forest at distance: cascades, WDL, farclip"
)

# The ablations. Each names ONE pass or lane, so a delta against `base` prices that thing and
# nothing else. `classic` is not an ablation — it is the bracket: every other delta must fit
# inside base→classic, and if the individual ablations sum to more than that gap, a lever is lying.
# NOTE the `lighting=` key rides the ablation, not the base URL. A duplicate query key would make
# the winner depend on `webenv::lookup`'s scan order, which is not a thing a measurement should
# depend on.
ABLATIONS=(
  "base|&lighting=modern"
  "classic|&lighting=classic"
  "ssao0|&lighting=modern&ssao=0"
  "gi0|&lighting=modern&modern_gi=0"
  "lights0|&lighting=modern&modern_lights=0"
  # Lights derived from things that only glow (lamps with no M2 light, lit windows, gear): off.
  # On the tiers that light from them, base minus this is what that lighting costs.
  "emissive0|&lighting=modern&modern_emissive=0"
  "mirrortier|&lighting=modern&modern_mirrortier=1"
  "water1|&lighting=modern&water_detail=1"
  "water0|&lighting=modern&water_detail=0"
  "nostaticgx|&lighting=modern&static_gx=0"
)

case "${1:-}" in
  serve)
    echo "# host: $HOST  packs: $PACKS_REMOTE"
    echo "# (Ctrl-C to stop)"
    exec ./target/release/wenilla-host --bind "127.0.0.1:$PORT" --www web/dist --packs-remote "$PACKS_REMOTE"
    ;;
  urls)
    scale="${2:-1.0}"
    only="${3:-}"
    case "$only" in all | -) only="" ;; esac
    quality="${4:-1}"
    clock="${5:-720}"
    echo "# superscale=$scale frames=$FRAMES tier=$quality clock=$clock — keep superscale FIXED across a zone's sweep."
    echo "# Read the console for: PERF_PROBE cfg=<label> … p50_ms= … railed="
    echo "# If railed=yes, raise superscale and redo the whole zone."
    for z in "${ZONES[@]}"; do
      IFS='|' read -r zlabel zmap zat zface zwhy <<<"$z"
      [ -n "$only" ] && [ "$only" != "$zlabel" ] && continue
      echo
      echo "## $zlabel — $zwhy"
      face=""
      [ "$zface" != "-" ] && face="&worldview_face=$zface"
      for a in "${ABLATIONS[@]}"; do
        IFS='|' read -r alabel aq <<<"$a"
        # `lighting_quality`, not `lightingquality`: the page's keys are the launch env's names
        # lowercased without `WOW_` (`benilla-app` `webenv::var`), so this is `WOW_LIGHTING_QUALITY`.
        # The old spelling reached nothing, and every row ran at the platform default (Medium).
        echo "$HOST/?worldview=1&lighting_quality=$quality&map=$zmap&worldview_at=$zat$face&clock=$clock&perf=$FRAMES&superscale=$scale&perf_label=$zlabel-q$quality-t$clock-$alabel$aq"
      done
    done
    ;;
  *)
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
