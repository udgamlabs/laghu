#!/bin/bash
# Laghu SFT dev pick (DEV_RULE.md): generate both candidates' replies on the frozen dev sets. Runs ON the Colab VM,
# started by devpick_run.py. The HF token is used only to download the two private exports, then deleted.
set -uo pipefail
G=/content/devpick; L=/content/devpick.log
exec > $L 2>&1
fail() { echo "DEVPICK_FAILED: $1"; rm -f /root/.cache/huggingface/token; exit 1; }
t0=$(date +%s)
cd $G && sha256sum -c DEV_SETS.sha256 2>/dev/null | grep -q "dev_gsm8k.jsonl: OK" && sha256sum -c DEV_SETS.sha256 2>/dev/null | grep -q "dev_chat.jsonl: OK" || fail "dev set hashes"
pip -q install uv || fail uv
mkdir -p /content/harness && cp $G/pyproject.toml $G/uv.lock /content/harness/
(cd /content/harness && uv sync --frozen -q) || fail harness
PY=/content/harness/.venv/bin/python
$PY - <<'EOF' || fail download
from huggingface_hub import snapshot_download
snapshot_download("YOUR_HF_NAMESPACE/laghu-sft", revision="c8c36d1c852a0953f94fc7abcfdf58356c9c5a50",
                  allow_patterns=["laghu-sft-v1-ep1/hf/*", "laghu-sft-v1-ep2/hf/*"], local_dir="/content/devpick/models")
print("models downloaded")
EOF
rm -f /root/.cache/huggingface/token
echo "token removed: $([ -e /root/.cache/huggingface/token ] && echo NO || echo yes)"
$PY -c "import torch, transformers; print('env:', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
HF_HUB_OFFLINE=1 $PY $G/devpick_gen.py --model ep1=$G/models/laghu-sft-v1-ep1/hf --model ep2=$G/models/laghu-sft-v1-ep2/hf \
  --chat $G/dev_chat.jsonl --gsm $G/dev_gsm8k.jsonl --out $G/gen || fail "generation"
cd $G && tar czf /content/devpick_gen.tgz gen
echo "DEVPICK_DONE after $(( $(date +%s) - t0 )) s"
