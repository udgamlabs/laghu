#!/bin/bash
# Laghu (Udgam-3) fine-tuning (SFT) on a Colab G4. Plan: udgam3/sft/SFT_RUN_PLAN.md. Structure = the 32K stretch's
# pipeline (udgam3/runs/ctx32k/colab_ctx32k_pipeline.sh): frozen code bundle by revision, private repos, preflight upload,
# background process, "SFT ..." stage lines, "SFT_FAILED: why" on failure, token removed at the end.
#   bash colab_sft_pipeline.sh EPOCHS [DRY]      EPOCHS = passes over the mix (identity rows get 2x); DRY=1 -> 10 steps
# Needs /root/.cache/huggingface/token, /content/chat_smoke.py.
set -uo pipefail
EP=${1:-2}; DRY=${2:-0}
[[ "$EP" =~ ^[0-9]+$ ]] || { echo "SFT_FAILED: EPOCHS must be an integer, got '$EP'"; exit 1; }
RUN=laghu-sft-v1-ep$EP; [ "$DRY" = 1 ] && RUN=$RUN-dry
CODE_REV=${CODE_REV:?set CODE_REV (udgam-build-code revision with the SFT changes)}
INIT_REPO=YOUR_HF_NAMESPACE/laghu-ctx32k; INIT_PT=udgam3-ctx32k/model_001900.pt; INIT_META=udgam3-ctx32k/meta_001900.json
DATA_REPO=YOUR_HF_NAMESPACE/laghu-sft-data; DATA_REV=11dd41c59558c0e57a1d81492ab1696803624f2c; DATA_SUB=v1-laghu
REPO=YOUR_HF_NAMESPACE/laghu-sft                                # private: RUN/{hf,ckpt,logs,smoke}
export INIT_REPO INIT_PT INIT_META DATA_REPO DATA_REV DATA_SUB CODE_REV
L=/content/sft.log; exec > >(tee -a $L) 2>&1
ts() { date -u +%T; }
cleanup() { rm -f /root/.cache/huggingface/token; }
fail() { echo "SFT_FAILED: $1 $(ts)"; cleanup; exit 1; }
cd /content
rm -rf /content/sft_out /content/sft_hf   # never mix in a previous run's checkpoints
echo "SFT setup run=$RUN epochs=$EP dry=$DRY code=$CODE_REV $(ts)"
python3 - <<'PY' || fail "code download"
import os
from huggingface_hub import snapshot_download
snapshot_download("YOUR_HF_NAMESPACE/laghu-build-code", revision=os.environ["CODE_REV"], allow_patterns=["train-udgam3/*"], local_dir="/content/code")
print("code ok")
PY
C=/content/code/train-udgam3
cd $C/nanochat && pip -q install uv > /dev/null 2>&1 && uv sync --extra gpu -q > /content/uv_sync.log 2>&1 \
  && uv pip install --python .venv/bin/python -q "wandb>=0.30" "transformers==5.18.0" setuptools huggingface_hub >> /content/uv_sync.log 2>&1 \
  && source .venv/bin/activate && python -c "import transformers, wandb, huggingface_hub" \
  || { tail -20 /content/uv_sync.log; fail "env"; }
grep -q "new-token-ids" scripts/udgam_sft.py || fail "code revision lacks the SFT changes"
python -c "import torch; print('SFT torch', torch.__version__, torch.cuda.get_device_name(0))"
python - <<PY || fail "preflight upload"
import os
from huggingface_hub import HfApi
a = HfApi(); a.create_repo("$REPO", private=True, exist_ok=True)
assert a.model_info("$REPO").private, "repo is not private"
p = "/content/preflight.bin"; open(p, "wb").write(os.urandom(64 << 20))
a.upload_file(path_or_fileobj=p, path_in_repo="$RUN/preflight.bin", repo_id="$REPO", commit_message="preflight upload test")
a.delete_file("$RUN/preflight.bin", repo_id="$REPO", commit_message="remove preflight test file"); os.remove(p)
print("preflight upload OK (64 MB, private)")
PY
echo "SFT downloads $(ts)"
python - <<'PY' || fail downloads
import os
from huggingface_hub import hf_hub_download, snapshot_download
E = os.environ
for f in (E["INIT_PT"], E["INIT_META"]):
    hf_hub_download(E["INIT_REPO"], f, local_dir="/content/init")
snapshot_download(E["DATA_REPO"], repo_type="dataset", revision=E["DATA_REV"], local_dir="/content/sftdata",
                  allow_patterns=[E["DATA_SUB"] + "/train.parquet", E["DATA_SUB"] + "/val.parquet", E["DATA_SUB"] + "/stats.json"])
print("downloads ok")
PY
export NANOCHAT_BASE_DIR=/content/cache NANOCHAT_TOKENIZER_DIR=$C/tokenizer PYTHONUNBUFFERED=1 WANDB_MODE=offline
D=/content/sftdata/$DATA_SUB
MIX="{\"identity\":$((2 * EP)),\"*\":$EP}"
SFT="python -m scripts.udgam_sft --train $D/train.parquet --val $D/val.parquet --init-from /content/init/$INIT_PT
  --init-meta /content/init/$INIT_META --mix $MIX --seq 16384 --device-batch ${SFT_DEVICE_BATCH:-4} --total-batch 262144
  --new-token-ids 5-16 --seed 1234 --out /content/sft_out --run dummy"
if [ "$DRY" = 1 ]; then SFT="$SFT --max-steps 10 --eval-every 5"; else SFT="$SFT --eval-every 20 --save-at 0.5,1.0"; fi
cd $C/nanochat
echo "SFT train $(ts)"
set -f; $SFT > /content/sft_train.log 2>&1; RC=$?; set +f   # no globbing of the "*" in the mix JSON
echo "SFT train rc=$RC $(ts)"; grep -E "^mix:|^packed|re-initialised|val all|^done" /content/sft_train.log | tail -8
F=$(ls /content/sft_out/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort | tail -1)
python - <<PY || echo "SFT (log upload failed)"
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/content/sft_train.log", path_in_repo="$RUN/logs/sft_train.log", repo_id="$REPO")
PY
[ $RC -eq 0 ] && [ -n "$F" ] || { tail -30 /content/sft_train.log; fail "training rc=$RC"; }
echo "SFT export $(ts)"
python -m scripts.export_hf --checkpoint-dir /content/sft_out --step $((10#$F)) --tokenizer-dir $C/tokenizer \
  --chat-template $C/tokenizer/chat_template.jinja --out /content/sft_hf --dtype bfloat16 --verify --weights final \
  --max-positions 16384 > /content/export_sft.log 2>&1
grep -E "exported|verify" /content/export_sft.log
grep -q "verify:.*PASS" /content/export_sft.log && [ -f /content/sft_hf/model.safetensors ] || fail "export / parity check"
echo "SFT smoke $(ts)"
python /content/chat_smoke.py /content/sft_hf /content/chat_smoke.json > /content/chat_smoke.log 2>&1 || { tail -20 /content/chat_smoke.log; fail "chat smoke"; }
head -c 2500 /content/chat_smoke.log
echo "SFT upload $(ts)"
python - <<PY || fail "upload"
from huggingface_hub import HfApi
a = HfApi()
a.upload_folder(repo_id="$REPO", folder_path="/content/sft_hf", path_in_repo="$RUN/hf", commit_message="$RUN: HF export")
a.upload_folder(repo_id="$REPO", folder_path="/content/sft_out", path_in_repo="$RUN/ckpt", commit_message="$RUN: nanochat checkpoint")
for f in ("chat_smoke.json", "chat_smoke.log", "export_sft.log", "sft.log"):
    a.upload_file(path_or_fileobj=f"/content/{f}", path_in_repo=f"$RUN/logs/{f}", repo_id="$REPO")
print("upload ok")
PY
cleanup
echo "SFT_DONE run=$RUN step=$((10#$F)) $(ts)"
