#!/bin/bash
# Coverage eval: fly through the world in VOXEL_EVAL_HOLES mode (magenta
# background, monotone geometry, water off) and fail if ANY captured frame
# contains a single background pixel — i.e. a hole in the world.
# Usage: scripts/eval_holes.sh [level.json] [seconds]
set -e
LEVEL=${1:-levels/planet.json}
SECS=${2:-60}
OUT=$(mktemp -d)
cargo build -p voxel2 2>&1 | tail -1
VOXEL_EVAL_HOLES=1 VOXEL_AUTOPILOT_LEVEL=1 VOXEL_START="-29840,2400,-36767" VOXEL_LOOK="0.05,-1.0,0.05" \
  VOXEL_AUTOPILOT=140 VOXEL_SCREENSHOT="$OUT/frame_%.png,4" \
  caffeinate -dis ./target/debug/voxel2 "$LEVEL" > "$OUT/run.log" 2>&1 &
PID=$!
# Warmup: let the initial world stream in before judging coverage
# (cold-start loading is a loading-screen concern, not a hole; the
# running-world invariant is what must never break).
sleep 30
rm -f "$OUT/frame_%.png"
# Rotate capture files so every interval is kept, not overwritten.
for i in $(seq 1 $((SECS / 4))); do
  sleep 4
  [ -f "$OUT/frame_%.png" ] && mv "$OUT/frame_%.png" "$OUT/frame_$i.png"
done
kill $PID 2>/dev/null || true
sleep 1
ERRS=$(grep -cE "Validation Error|panicked" "$OUT/run.log" || true)
# `set -e` must not kill the script before the diagnostics print: a
# hole failure still reports validation errors and the kept-frames path.
HOLES=0
python3 scripts/count_holes.py "$OUT"/frame_*.png || HOLES=$?
echo "validation-errors: $ERRS"
[ "$ERRS" = "0" ] && [ "$HOLES" = "0" ] && echo "EVAL PASS" || { echo "EVAL FAIL (frames kept in $OUT)"; exit 1; }
