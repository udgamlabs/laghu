#!/bin/bash
# Udgam-3 long-context stretch (train at 32K, claim 16K) on a Colab G4. Plan: udgam3/runs/ctx32k/PLAN.md.
# Adapted from udgam2/ctx32k/colab_ctx32k_pipeline.sh (Gargi's three Colab stages).
# Runs ON the VM as a background process (started by colab_run.py), so the kernel stays free for the dashboard's reads.
#   bash colab_ctx32k_pipeline.sh STEPS [DEVICE_BATCH]     (STEPS < 100 = dry run: RUN gets "-dry", 4 needle trials)
# Needs /root/.cache/huggingface/token (removed again at the end). Every stage prints "CTX ..."; failure = "CTX_FAILED: why".
set -uo pipefail
STEPS=${1:-1900}; DB=${2:-2}
[[ "$STEPS" =~ ^[0-9]+$ ]] || { echo "CTX_FAILED: STEPS must be an integer, got '$STEPS'"; exit 1; }
RUN=${RUN:-udgam3-ctx32k}; [ "$STEPS" -lt 100 ] && RUN=$RUN-dry
CODE_REV=ef821d67177f60d64dc8d7cd85b8d7543b31d4b1      # the main run's frozen code (udgam-build-code)
INIT_REPO=YOUR_HF_NAMESPACE/laghu-base-ckpt; INIT_PT=udgam3-base-v1/model_ema_076294.pt; INIT_META=udgam3-base-v1/meta_076294.json
REPO=YOUR_HF_NAMESPACE/laghu-ctx32k                       # checkpoints (RUN/), eval/, logs/, hf/ (private)
DATA_REPO=YOUR_HF_NAMESPACE/laghu-ctx32k-data
export INIT_REPO INIT_PT INIT_META DATA_REPO CODE_REV
L=/content/ctx32k.log; exec > >(tee -a $L) 2>&1
ts() { date -u +%T; }
cleanup() { rm -f /root/.cache/huggingface/token; }
fail() { echo "CTX_FAILED: $1 $(ts)"; cleanup; exit 1; }
cd /content
echo "CTX setup run=$RUN steps=$STEPS device_batch=$DB $(ts)"
python3 - <<'PY' || fail "code download"
import os
from huggingface_hub import snapshot_download
snapshot_download("YOUR_HF_NAMESPACE/laghu-build-code", revision=os.environ["CODE_REV"], allow_patterns=["train-udgam3/*"], local_dir="/content/code")
print("code ok")
PY
C=/content/code/train-udgam3
# uv pip gets an explicit --python: without it the dry run (2026-10-19) installed transformers somewhere else and the
# export failed. transformers 5.18.0 = the version the main run's export used.
cd $C/nanochat && pip -q install uv > /dev/null 2>&1 && uv sync --extra gpu -q > /content/uv_sync.log 2>&1 \
  && uv pip install --python .venv/bin/python -q "wandb>=0.30" "transformers==5.18.0" setuptools huggingface_hub >> /content/uv_sync.log 2>&1 \
  && source .venv/bin/activate && python -c "import transformers, wandb, huggingface_hub" \
  || { tail -20 /content/uv_sync.log; fail "env"; }
python -c "import torch; print('CTX torch', torch.__version__, torch.cuda.get_device_name(0))"
H="python $C/ops/pod_hf.py"
python - <<PY || fail "preflight upload"
import os
from huggingface_hub import HfApi
a = HfApi(); a.create_repo("$REPO", private=True, exist_ok=True)
assert a.model_info("$REPO").private, "repo is not private"
p = "/content/preflight.bin"; open(p, "wb").write(os.urandom(256 << 20))
a.upload_file(path_or_fileobj=p, path_in_repo="$RUN/preflight.bin", repo_id="$REPO", commit_message="preflight upload test")
a.delete_file("$RUN/preflight.bin", repo_id="$REPO", commit_message="remove preflight test file"); os.remove(p)
print("preflight upload OK (256 MB, private)")
PY
echo "CTX downloads $(ts)"
python - <<'PY' || fail downloads
import os, shutil
from huggingface_hub import hf_hub_download, snapshot_download
E = os.environ
for f in (E["INIT_PT"], E["INIT_META"]):
    hf_hub_download(E["INIT_REPO"], f, local_dir="/content/init")
shutil.rmtree("/content/d32/train", ignore_errors=True)
snapshot_download(E["DATA_REPO"], repo_type="dataset", local_dir="/content/d32", allow_patterns=["train/*", "val_long/*", "report.json"])
for repo, v in (("YOUR_HF_NAMESPACE/udgam-val-sets", "val_web"), ("YOUR_HF_NAMESPACE/udgam-val-sets", "val_task"),
                ("YOUR_HF_NAMESPACE/udgam-val-sets", "val_reason"), ("YOUR_HF_NAMESPACE/laghu-val-sets", "val_fresh")):
    p = hf_hub_download(repo, f"val_sets/{v}/val.parquet", repo_type="dataset", local_dir="/content/dl")
    os.makedirs(f"/content/d32/{v}", exist_ok=True); shutil.copy(p, f"/content/d32/{v}/val.parquet")
print("downloads ok:", len(os.listdir("/content/d32/train")), "train shards")
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export NANOCHAT_BASE_DIR=/content/cache NANOCHAT_DATA_DIR=/content/d32/train NANOCHAT_TOKENIZER_DIR=$C/tokenizer
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
CKDIR=/content/cache/base_checkpoints/$RUN; mkdir -p $CKDIR
STEP=$($H latest $REPO $RUN)
if [ -n "$STEP" ]; then
  echo "CTX resume: found step $STEP on HF $(ts)"; $H fetch $REPO $RUN $STEP $CKDIR | tr '\n' ' '; echo
  touch $CKDIR/saved_$(printf %06d $STEP).ok
fi
( last="$STEP"; while sleep 30; do
    s=$(ls $CKDIR/saved_*.ok 2>/dev/null | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | tail -1)
    [ -z "$s" ] && continue; s=$((10#$s))
    if [ "$s" != "$last" ] && [ ! -f /content/ctx_final_push ]; then
      $H push $REPO $RUN $s $CKDIR 2 && echo "CTX uploaded checkpoint step $s $(ts)" && last=$s
      for old in $(ls $CKDIR/saved_*.ok | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | head -n -2); do rm -f $CKDIR/*_${old}*; done
    fi
  done ) &
UPLOADER=$!
# 10% of the main-run peak LRs (same 524,288 batch, so nanochat's batch scaling is the same), warmup 30 (1 for dry runs),
# linear decay to 0, no weight decay: Gargi's stretch recipe
WU=$(( STEPS < 300 ? 1 : 30 ))
ARGS="--arch gemma4 --run dummy --model-tag $RUN --depth 28 --n-embd 640 --n-head 10 --n-kv-head 2 --g4-head-dim 64
  --g4-global-head-dim 128 --intermediate-size 1792 --ple-dim 0 --window-pattern SSSL --sliding-window 512 --tie-embeddings
  --max-seq-len 32768 --device-batch-size $DB --total-batch-size 524288 --num-iterations $STEPS
  --warmup-steps $WU --warmdown-ratio $(python3 -c "print(1 - $WU / $STEPS)") --final-lr-frac 0.0 --weight-decay 0.0
  --matrix-lr 0.002 --embedding-lr 0.03 --unembedding-lr 0.0008 --scalar-lr 0.05
  --init-from /content/init/$INIT_PT --seed 42 --save-every $(( STEPS < 100 ? 10 : 250 ))
  --eval-every $(( STEPS < 100 ? 10 : 250 )) --eval-tokens 4194304 --core-metric-every -1 --sample-every -1
  --extra-val web=/content/d32/val_web --extra-val task=/content/d32/val_task --extra-val reason=/content/d32/val_reason
  --extra-val fresh=/content/d32/val_fresh"
cd $C/nanochat
RC=1
for attempt in 1 2 3; do
  Rs=$(ls $CKDIR/saved_*.ok 2>/dev/null | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | tail -1)
  RES=""; [ -n "$Rs" ] && RES="--resume-from-step $((10#$Rs))"
  echo "CTX train attempt $attempt $RES $(ts)"
  python -m scripts.base_train $ARGS $RES >> /content/ctx32k_train.log 2>&1; RC=$?
  echo "CTX train rc=$RC $(ts)"; grep -E "^step|bpb|Peak|Error|out of memory" /content/ctx32k_train.log | tail -6
  [ $RC -eq 0 ] && break; sleep 10
done
touch /content/ctx_final_push; kill $UPLOADER 2>/dev/null
F=$(ls $CKDIR/saved_*.ok 2>/dev/null | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | tail -1)
[ -n "$F" ] || fail "no checkpoint"
F=$((10#$F))
$H push $REPO $RUN $F $CKDIR 2 || $H push $REPO $RUN $F $CKDIR 2 || fail "final push (weights only on the VM: $CKDIR)"
$H put $REPO /content/ctx32k_train.log logs/$RUN/train.log
[ $RC -eq 0 ] || fail "training rc=$RC (last good checkpoint step $F is on HF)"
echo "CTX eval $(ts)"
FS=$(printf %06d $F)
EV="--val web=/content/d32/val_web --val task=/content/d32/val_task --val reason=/content/d32/val_reason --val fresh=/content/d32/val_fresh
  --long-val /content/d32/val_long/val_long.parquet --long-lengths 8192,16384,32768 --long-batch 8
  --needle-lengths 1024,2048,4096,8192,12288,16384,24576,32768 --copy-lengths 2048,8192,16384,32768"
NT=$(( STEPS < 100 ? 4 : 100 ))
python -m scripts.eval_long --ckpt $CKDIR/model_$FS.pt --meta $CKDIR/meta_$FS.json --out /content/eval_ctx32k.json $EV --needle-trials $NT > /content/eval_ctx32k.log 2>&1 \
  || { tail -20 /content/eval_ctx32k.log; fail "eval 32k model"; }
python -m scripts.eval_long --ckpt /content/init/$INIT_PT --meta /content/init/$INIT_META \
  --out /content/eval_base2k.json $EV --needle-trials $NT > /content/eval_base2k.log 2>&1 || { tail -20 /content/eval_base2k.log; fail "eval base"; }
grep -E "bpb@2048|needle|copy|long docs" /content/eval_ctx32k.log /content/eval_base2k.log | cut -c1-260
for f in eval_ctx32k eval_base2k; do $H put $REPO /content/$f.json eval/$RUN/$f.json; $H put $REPO /content/$f.log eval/$RUN/$f.log; done
echo "CTX export $(ts)"
python -m scripts.export_hf --checkpoint-dir $CKDIR --step $F --tokenizer-dir $C/tokenizer --chat-template $C/tokenizer/chat_template.jinja \
  --out /content/ctx32k_hf --dtype bfloat16 --verify --weights final --max-positions 32768 > /content/export_ctx.log 2>&1
grep -E "exported|verify" /content/export_ctx.log
grep -q "verify:.*PASS" /content/export_ctx.log && [ -f /content/ctx32k_hf/model.safetensors ] || fail "export / parity check"
PYTHONPATH=$C/nanochat python /content/long_parity.py $CKDIR/model_$FS.pt $CKDIR/meta_$FS.json /content/ctx32k_hf /content/d32/val_long/val_long.parquet --len 16384 \
  > /content/long_parity.log 2>&1; tail -1 /content/long_parity.log
grep -q "long_parity:.*PASS" /content/long_parity.log || fail "long parity (HF export differs from our model at 16K)"
$H put $REPO /content/ctx32k_hf hf/$RUN
cleanup
echo "CTX_DONE step=$F $(ts)"
