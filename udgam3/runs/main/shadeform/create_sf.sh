#!/usr/bin/env bash
# Create one Shadeform instance for a Udgam run. Usage: bash create_sf.sh CONFIG_FILE [--dry-run]
# CONFIG_FILE sources common.sh (shape/recipe/data) and sets RUN_NAME TRAIN_ARGS ... plus SF_* and MAX_MIN.
set -u
set -a; source "$1"; set +a
export OUT=${OUT:-"$(cd "$(dirname "$0")" && pwd)/state"}; mkdir -p "$OUT"
exec python3 "$(dirname "$0")/create_sf_instance.py" ${2:-}
