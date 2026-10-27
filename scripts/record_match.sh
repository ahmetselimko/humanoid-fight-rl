#!/bin/bash
# Record a robot-vs-robot match (both sides driven by the tactic net) to an MP4.
# Usage: bash scripts/record_match.sh [output.mp4] [seed] [seconds]
set -e
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-media/my_match.mp4}")
source scripts/best_system.sh
cd sim
PY=${PYTHON:-python}
PYTHONWARNINGS=ignore $PY dovus_kayit.py --mp4 "$OUT" --tohum "${2:-3}" --sure "${3:-40}"
echo "saved: $OUT"
