#!/bin/bash
# Play N matches (default 24): tactic net vs the rule-based opponent, same seeds every run.
# Usage: bash scripts/evaluate.sh [N] [WORKERS]
set -e
cd "$(dirname "$0")/.."
source scripts/best_system.sh
cd sim
PY=${PYTHON:-python}
HAVUZ=${2:-2} PYTHONWARNINGS=ignore $PY degerlendirme/taktik_olc.py "${1:-24}" "$TAKTIK"
