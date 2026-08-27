#!/usr/bin/env bash
# Sweep the mode 3 quality knobs headless, collecting timings and a PNG per configuration.
#
#   scripts/sweep.sh splats/rem_v3_clear.ply out/
#
# Everything runs without a window, so it is safe to use the machine while this runs and
# results do not depend on compositor or vsync behaviour.
set -euo pipefail

PLY="${1:?usage: sweep.sh <scene.ply> [outdir]}"
OUT="${2:-bench-out}"
BIN="./target/release/wboit-demo"
FRAMES="${FRAMES:-200}"
SIZE="${SIZE:-1280x720}"

[ -x "$BIN" ] || { echo "build first: cargo build --release" >&2; exit 1; }
mkdir -p "$OUT"

# median ms for one configuration, plus its screenshot
run() { # mode tile bins label
  local mode=$1 tile=$2 bins=$3 label=$4
  "$BIN" "$PLY" --headless --mode "$mode" --tile "$tile" --bins "$bins" \
      --frames "$FRAMES" --size "$SIZE" \
      --screenshot "$OUT/$label.png" 2>/dev/null \
    | awk '/Alpha Blend|Naive WBOIT|Histogram-Eq/ {print $(NF-2)}'
}

echo "=== baseline: all three modes (tile 8, 64 bins) ==="
printf '%-28s %10s\n' config "median ms"
for m in 1 2 3; do
  printf '%-28s %10s\n' "mode$m" "$(run "$m" 8 64 "mode$m")"
done

echo
echo "=== mode 3: tile size sweep (64 bins) ==="
printf '%-28s %10s\n' config "median ms"
for t in 32 16 8 4; do
  printf '%-28s %10s\n' "tile$t" "$(run 3 "$t" 64 "tile$t")"
done

echo
echo "=== mode 3: bin count sweep (tile 8) ==="
printf '%-28s %10s\n' config "median ms"
for b in 32 64 128 256; do
  printf '%-28s %10s\n' "bins$b" "$(run 3 8 "$b" "bins$b")"
done

echo
echo "PNGs in $OUT/ -- compare mode1.png (sorted reference) against the rest."
