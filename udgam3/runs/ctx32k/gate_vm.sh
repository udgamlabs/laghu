#!/bin/bash
# Udgam-3 32K stretch, benchmark gate (pre-registration 6.3): the base suite on VALIDATION splits for the stretch model
# and the main-run EMA, same env, same settings. Runs ON the ctx32k Colab VM after CTX_DONE (started by base_gate.py).
# Needs /root/.cache/huggingface/token only for the EMA download; it is deleted before any evaluation runs.
set -uo pipefail
G=/content/gate; L=/content/gate.log
exec > $L 2>&1
fail() { echo "GATE_FAILED: $1"; rm -f /root/.cache/huggingface/token; exit 1; }
t0=$(date +%s)
pip -q install uv || fail uv
mkdir -p /content/harness && cp $G/pyproject.toml $G/uv.lock /content/harness/
(cd /content/harness && uv sync --frozen -q) || fail harness
PY=/content/harness/.venv/bin/python
$PY - <<'EOF' || fail download
from huggingface_hub import snapshot_download
snapshot_download("YOUR_HF_NAMESPACE/laghu-base", revision="1f2e414c7ae7c9fa9ac313a06b0694a72f7d8e23",
                  allow_patterns=["udgam3-base-v1/ema/*"], local_dir="/content/gate/ema_dl")
print("ema downloaded")
EOF
rm -f /root/.cache/huggingface/token
echo "token removed: $([ -e /root/.cache/huggingface/token ] && echo NO || echo yes)"
[ -f /content/ctx32k_hf/model.safetensors ] || fail "stretch export missing"
TASKS=arc_easy_val,arc_challenge_val,hellaswag,piqa,winogrande
for m in ema:/content/gate/ema_dl/udgam3-base-v1/ema ctx32k:/content/ctx32k_hf; do
  n=${m%%:*}; p=${m#*:}
  echo "GATE eval $n $(date -u +%T)"
  HF_HUB_OFFLINE=0 $PY -W ignore -m lm_eval --model hf --model_args pretrained=$p,dtype=bfloat16,add_bos_token=True \
    --device cuda --tasks $TASKS --include_path $G/tasks --batch_size 64 --seed 1234 --log_samples \
    --output_path $G/out/$n > $G/lmeval_$n.log 2>&1 || { tail -20 $G/lmeval_$n.log; fail "lm-eval $n"; }
  tail -12 $G/lmeval_$n.log
done
$PY -c "import torch, transformers, lm_eval; print('env:', torch.__version__, transformers.__version__, lm_eval.__version__)"
cd /content/gate && tar czf /content/gate_out.tgz out lmeval_*.log
echo "GATE_DONE after $(( $(date +%s) - t0 )) s"
