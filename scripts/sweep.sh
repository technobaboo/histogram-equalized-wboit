#!/usr/bin/env bash
# Sweep the OIT quality knobs headless, collecting a timing, a PNG and an image-loss
# score per configuration.
#
#   scripts/sweep.sh splats/rem_v3_clear.ply out/
#
# Everything runs without a window, so it is safe to use the machine while this runs and
# results do not depend on compositor or vsync behaviour.
#
# Cost and loss come from two separate runs of two separate harnesses: --headless times a
# GPU fence from a pinned camera, --quality scores N random views against an exactly
# sorted reference. They measure different things and neither substitutes for the other.
set -euo pipefail

PLY="${1:?usage: sweep.sh <scene.ply> [outdir]}"
OUT="${2:-bench-out}"
BIN="./target/release/wboit-demo"
FRAMES="${FRAMES:-200}"
SIZE="${SIZE:-1280x720}"
VIEWS="${VIEWS:-16}"
QSIZE="${QSIZE:-640x360}"

[ -x "$BIN" ] || { echo "build first: cargo build --release" >&2; exit 1; }
mkdir -p "$OUT"

# median ms for one configuration, plus its screenshot
run() { # mode tile bins label
  local mode=$1 tile=$2 bins=$3 label=$4
  "$BIN" "$PLY" --headless --mode "$mode" --tile "$tile" --bins "$bins" \
      --frames "$FRAMES" --size "$SIZE" \
      --screenshot "$OUT/$label.png" 2>/dev/null \
    | awk '/Alpha Blend|Naive WBOIT|Histogram-Eq|Quantile-Sliced/ {print $(NF-2)}'
}

# foreground MSE against the sorted reference for one configuration
loss() { # mode tile bins
  local mode=$1 tile=$2 bins=$3
  "$BIN" "$PLY" --quality "$VIEWS" --mode "$mode" --tile "$tile" --bins "$bins" \
      --size "$QSIZE" 2>/dev/null \
    | awk '/Naive WBOIT|Histogram-Eq|Quantile-Sliced/ && NF > 4 {print $(NF-4); exit}'
}

echo "=== baseline: every mode (tile 8, 64 bins) ==="
printf '%-28s %10s %12s\n' config "median ms" "fg MSE"
for m in 1 2 3 4; do
  # Mode 1 is the quality harness's reference, so it has no loss of its own to report.
  if [ "$m" = 1 ]; then
    printf '%-28s %10s %12s\n' "mode$m" "$(run 1 8 64 mode1)" "reference"
  else
    printf '%-28s %10s %12s\n' "mode$m" "$(run "$m" 8 64 "mode$m")" "$(loss "$m" 8 64)"
  fi
done

echo
echo "=== tile size sweep, modes 3 and 4 (64 bins) ==="
printf '%-28s %10s %12s\n' config "median ms" "fg MSE"
for m in 3 4; do
  for t in 32 16 8 4; do
    printf '%-28s %10s %12s\n' "mode$m tile$t" \
      "$(run "$m" "$t" 64 "mode$m.tile$t")" "$(loss "$m" "$t" 64)"
  done
done

echo
echo "=== bin count sweep, modes 3 and 4 (tile 8) ==="
printf '%-28s %10s %12s\n' config "median ms" "fg MSE"
for m in 3 4; do
  for b in 32 64 128 256; do
    printf '%-28s %10s %12s\n' "mode$m bins$b" \
      "$(run "$m" 8 "$b" "mode$m.bins$b")" "$(loss "$m" 8 "$b")"
  done
done

echo
echo "PNGs in $OUT/ -- compare mode1.png (sorted reference) against the rest."
