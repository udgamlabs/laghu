#!/bin/zsh
# $0 validation of our own benchmarks on models already cached on the Mac, one at a time (fixed small memory).
cd "${LAGHU_REPO:-.}"
O=udgam3/evals/own/validation
for m in HuggingFaceTB/SmolLM2-135M tiiuae/Falcon-H1-Tiny-90M-Base udgamlabs/Gargi-Base UdgamLabs/Udgam-384M-Base-4K HuggingFaceTB/SmolLM2-360M; do
  n=$(basename $m)
  [ -f $O/bpb_$n.json ] || uv run --project evals/harness python udgam3/evals/own/bpb_hf.py $m $O/bpb_$n.json --device mps 2>&1 | grep -v -i -E "warn|Loading weights" | tail -1
  [ -f $O/pea_$n.json ] || uv run --project evals/harness python udgam3/evals/own/pea.py score $m udgam3/evals/own/pea_v1.jsonl $O/pea_$n.json --device mps 2>&1 | grep -v -i -E "warn|Loading weights" | tail -1
done
echo VALIDATION_DONE
