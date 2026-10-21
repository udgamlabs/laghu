#!/bin/zsh
# Run full-run slices one by one, each with its own budget on top of what full_v1 has already spent.
# usage: run_slices.sh <global_cap> <slice>:<budget> [<slice>:<budget> ...]
cd "$(dirname "$0")"
GCAP=$1; shift
for spec in "$@"; do
  sl=${spec%%:*}; b=${spec##*:}
  spent=$(python3 -c "import json; print(json.load(open('../full_v1/spend.json'))['usd'])")
  cap=$(python3 -c "print(min($GCAP, $spent + $b))")
  echo "$(date '+%H:%M:%S') start $sl budget=$b spent=$spent cap=$cap"
  uv run --project "${LAGHU_REPO:-.}/evals/harness" python -I run_full.py --plan full --slice $sl --out ../full_v1 --cap $cap 2>&1 | tail -1
done
echo "$(date '+%H:%M:%S') done"
