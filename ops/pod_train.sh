#!/usr/bin/env bash
# =============================================================================
# UdgamLabs training-pod bootstrap. Runs ON the Runpod GPU pod (created by ops/pod_build/create_train_pod.sh, which
# already started the hard-deadline watchdog in the container start command before this script was downloaded).
# Self-sufficient: does not need the Mac after launch.
#
#   1. second deadline watchdog (absolute DEADLINE_EPOCH, so container restarts can't extend it)
#   2. code (train/ in CODE_REPO) + data (shards/phaseA, shards/phaseB, val_sets/) from private HF repos, in parallel
#      with the Python env (uv sync --extra gpu, kernels==0.11.7 for FA3, wandb>=0.30, transformers for export)
#   3. resumes from the newest checkpoint in CKPT_REPO/RUN_NAME (LATEST) if one exists
#   4. torchrun on all GPUs; a background uploader pushes every completed checkpoint (saved_<step>.ok marker) to HF,
#      keeping the newest 2 steps; if torchrun crashes it is retried (up to 2x) from the newest local checkpoint
#   5. final checkpoint (incl. EMA) pushed synchronously, then HF export of final + EMA weights (--verify), quick
#      samples, upload to EXPORT_REPO/RUN_NAME/{final,ema}, DONE marker, log upload, self-delete
#
# Env (set at pod creation): RUN_NAME CODE_REPO DATA_REPO CKPT_REPO EXPORT_REPO DEADLINE_EPOCH TRAIN_ARGS
#   optional: SKIP_EXPORT=1 (export later, off-pod), DATA_PATTERNS (default 'shards/* val_sets/*'), SAVE_EVERY (default 1000) WANDB_PROJECT WANDB_ENTITY WANDB_RUN_ID NPROC
# Secrets (Runpod): HF_TOKEN, WANDB_API_KEY
# =============================================================================
set -uo pipefail
[ -f /etc/rp_environment ] && source /etc/rp_environment
: "${RUN_NAME:?}" "${CODE_REPO:?}" "${DATA_REPO:?}" "${CKPT_REPO:?}" "${EXPORT_REPO:?}" "${DEADLINE_EPOCH:?}" "${TRAIN_ARGS:?}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
WORK=/workspace; C=$WORK/code/train; mkdir -p $WORK/{data,cache,logs,export}
LOG=$WORK/logs/$RUN_NAME.log; exec > >(tee -a "$LOG") 2>&1
ts() { date -u +%FT%TZ; }
echo "[$(ts)] pod ${RUNPOD_POD_ID:-?} run $RUN_NAME | deadline $(date -u -d @$DEADLINE_EPOCH +%FT%TZ) | gpus $(nvidia-smi -L | wc -l)"
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1

selfdel() { echo "[$(ts)] self-delete: $1"; python3 $C/ops/pod_delete.py; sleep 30; runpodctl remove pod "${RUNPOD_POD_ID:-x}" 2>/dev/null; sleep infinity; }
upload_log() { python3 $C/ops/pod_hf.py put "$CKPT_REPO" "$LOG" "$RUN_NAME/logs/$(date -u +%Y%m%dT%H%M%S)_${RUNPOD_POD_ID:-pod}.log" >/dev/null 2>&1; }

# 1) watchdog #2 (the start command runs #1): kill training, push the log, delete the pod at the deadline
( left=$((DEADLINE_EPOCH - $(date +%s))); [ $left -gt 0 ] && sleep $left
  echo "[$(ts)] DEADLINE HIT - stopping training and deleting the pod"; pkill -f torchrun; sleep 5; upload_log; selfdel "deadline" ) &

# 2) code is already in $WORK/code (start command); data + env in parallel
cd $C/nanochat
( python3 -c "import os; from huggingface_hub import snapshot_download as d; d(os.environ['DATA_REPO'], repo_type='dataset', local_dir='$WORK/data', allow_patterns=os.environ.get('DATA_PATTERNS', 'shards/* val_sets/*').split(), max_workers=16)" >/dev/null \
  && echo "[$(ts)] data ready: A $(ls $WORK/data/shards/phaseA | wc -l) files, B $(ls $WORK/data/shards/phaseB | wc -l) files, $(du -sh $WORK/data | cut -f1)" \
  || echo "DATA_FAILED" > $WORK/data_failed ) &
DATA_PID=$!
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; source $HOME/.local/bin/env; }
# 2026-10-11: use uv's managed Python (ships Python.h). On images without python3.10-dev headers, torch.compile's Triton
# helper build failed ("gcc ... -I/usr/include/python3.10 ... exit status 1").
export UV_PYTHON_PREFERENCE=only-managed
uv sync --extra gpu -q && source .venv/bin/activate && uv pip install -q "kernels==0.11.7" "wandb>=0.30" "transformers==5.18.0" setuptools \
  || { echo "ENV SETUP FAILED"; upload_log; selfdel "env setup failed"; }
echo "[$(ts)] python env ready: $(python -c 'import torch,transformers; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)')"
NG=$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null)
if [ "${NG:-0}" -lt 1 ]; then  # the image's CUDA forward-compat libs break CUDA init on some hosts (error 804): retry without them
  export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v compat | paste -sd: -)
  NG=$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null)
  echo "[$(ts)] CUDA unavailable with compat libs; without them: $NG GPUs"
fi
if [ "${NG:-0}" -lt 1 ]; then echo "CUDA NOT AVAILABLE on this host (LD_LIBRARY_PATH=$LD_LIBRARY_PATH)"; nvidia-smi | head -5; upload_log; selfdel "no CUDA"; fi
wait $DATA_PID
[ -f $WORK/data_failed ] && { echo "DATA DOWNLOAD FAILED"; upload_log; selfdel "data download failed"; }
# 2026-10-11: optional extra data repos merged into the shard dirs by symlink (EXTRA_DATA_JSON, a JSON list of
# {"repo", "patterns": [...], "links": [{"src": glob, "dst": dir under $WORK/data, "prefix": str, "drop_last": bool}],
#  "copy": [{"src": file under $WORK/data, "dst": path under $WORK/data}]}). nanochat reads a dir's parquet files in
# sorted order and uses the LAST one as validation, so prefixes must keep the validation shard last.
if [ -n "${EXTRA_DATA_JSON:-}" ]; then
  python3 - <<'PY' || { echo "EXTRA DATA FAILED"; upload_log; selfdel "extra data failed"; }
import glob, json, os
from huggingface_hub import snapshot_download
W = "/workspace"
for i, e in enumerate(json.loads(os.environ["EXTRA_DATA_JSON"])):
    root = f"{W}/data_extra{i}"
    snapshot_download(e["repo"], repo_type="dataset", local_dir=root, allow_patterns=e["patterns"], max_workers=16)
    for l in e.get("links", []):
        files = sorted(glob.glob(f"{root}/{l['src']}"))
        if l.get("drop_last"):
            files = files[:-1]
        dst = f"{W}/data/{l['dst']}"; os.makedirs(dst, exist_ok=True)
        for f in files:
            os.symlink(f, f"{dst}/{l['prefix']}{os.path.basename(f)}")
        print(f"linked {len(files)} files from {e['repo']}:{l['src']} -> {l['dst']} (prefix {l['prefix']})")
    for c in e.get("copy", []):
        os.makedirs(os.path.dirname(f"{W}/data/{c['dst']}"), exist_ok=True)
        os.symlink(f"{W}/data/{c['src']}", f"{W}/data/{c['dst']}")
        print(f"linked {c['src']} -> {c['dst']}")
for d in sorted(glob.glob(f"{W}/data/shards/*")):
    fs = sorted(os.listdir(d)); print(f"{d}: {len(fs)} files, first {fs[0]}, last (validation) {fs[-1]}")
PY
fi

export NANOCHAT_BASE_DIR=$WORK/cache NANOCHAT_DATA_DIR=$WORK/data/shards/phaseA NANOCHAT_TOKENIZER_DIR=$C/tokenizer
export WANDB_PROJECT="${WANDB_PROJECT:-udgam-main}" WANDB_ENTITY="${WANDB_ENTITY:-udgamlabs}" WANDB_RESUME=allow
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}" PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
CKDIR=$WORK/cache/base_checkpoints/$RUN_NAME; mkdir -p $CKDIR

# 3) resume from HF if this run already has a checkpoint there
STEP=$(python3 $C/ops/pod_hf.py latest "$CKPT_REPO" "$RUN_NAME"); SRC_REPO=$CKPT_REPO
if [ -z "$STEP" ] && [ -n "${RESUME_FROM_REPO:-}" ]; then   # 2026-10-11: start from another repo's checkpoint (e.g. a lost run's last save)
  STEP=$(python3 $C/ops/pod_hf.py latest "$RESUME_FROM_REPO" "$RUN_NAME"); SRC_REPO=$RESUME_FROM_REPO
fi
if [ -n "$STEP" ]; then
  echo "[$(ts)] found checkpoint step $STEP in $SRC_REPO/$RUN_NAME, downloading"
  python3 $C/ops/pod_hf.py fetch "$SRC_REPO" "$RUN_NAME" "$STEP" "$CKDIR" | tr '\n' ' '; echo
  touch $CKDIR/saved_$(printf %06d $STEP).ok
fi

# 4) background uploader: push each completed checkpoint, keep the newest 2 locally and on HF
( last="$STEP"; while sleep 20; do
    s=$(ls $CKDIR/saved_*.ok 2>/dev/null | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | tail -1)
    [ -z "$s" ] && continue; s=$((10#$s))
    if [ "$s" != "$last" ] && [ ! -f $WORK/final_push_started ]; then
      python3 $C/ops/pod_hf.py push "$CKPT_REPO" "$RUN_NAME" "$s" "$CKDIR" 2 && echo "[$(ts)] uploaded checkpoint step $s" && last=$s
      for old in $(ls $CKDIR/saved_*.ok | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | head -n -2); do rm -f $CKDIR/*_${old}*; done
    fi
  done ) &
UPLOADER_PID=$!

# 2026-10-11 preflight: prove we can write a sizable file to CKPT_REPO BEFORE paying for training (the first Udgam-2
# main run lost its weights: "storage limit reached for org UdgamLabs" -> every checkpoint push after step 1000 failed).
python3 - <<'PY' || { echo "[$(ts)] PREFLIGHT UPLOAD FAILED - not training"; upload_log; selfdel "preflight upload failed"; }
import os
from huggingface_hub import HfApi
api, repo, run = HfApi(), os.environ["CKPT_REPO"], os.environ["RUN_NAME"]
api.create_repo(repo, private=True, exist_ok=True)
p = "/workspace/preflight.bin"; open(p, "wb").write(os.urandom(256 << 20))
api.upload_file(path_or_fileobj=p, path_in_repo=f"{run}/preflight.bin", repo_id=repo, commit_message="preflight upload test")
api.delete_file(f"{run}/preflight.bin", repo_id=repo, commit_message="remove preflight test file")
print("preflight upload OK (256 MB to", repo, ")")
PY

NGPU=${NPROC:-$(nvidia-smi -L | wc -l)}
RC=1
for attempt in 1 2 3; do
  R=$(ls $CKDIR/saved_*.ok 2>/dev/null | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | tail -1)
  RESUME_ARGS=""; [ -n "$R" ] && RESUME_ARGS="--resume-from-step $((10#$R))"
  echo "[$(ts)] torchrun attempt $attempt on $NGPU GPUs $RESUME_ARGS"
  # shellcheck disable=SC2086
  torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- --arch gemma4 --run "$RUN_NAME" \
    --model-tag "$RUN_NAME" --save-every "$SAVE_EVERY" $RESUME_ARGS $TRAIN_ARGS
  RC=$?
  echo "[$(ts)] training exited rc=$RC"
  [ $RC -eq 0 ] && break
  sleep 10
done

# 5) final push (synchronous), export, samples, upload, done
touch $WORK/final_push_started; kill $UPLOADER_PID 2>/dev/null
F=$(ls $CKDIR/saved_*.ok 2>/dev/null | sed -E 's/.*saved_([0-9]+)\.ok/\1/' | sort | tail -1)
PUSHED=0
if [ -n "$F" ]; then
  python3 $C/ops/pod_hf.py push "$CKPT_REPO" "$RUN_NAME" "$((10#$F))" "$CKDIR" 2 && PUSHED=1 && echo "[$(ts)] final push: step $((10#$F))"
  if [ $PUSHED = 0 ] && [ -n "${FALLBACK_REPO:-}" ]; then
    python3 $C/ops/pod_hf.py push "$FALLBACK_REPO" "$RUN_NAME" "$((10#$F))" "$CKDIR" 2 && PUSHED=1 && echo "[$(ts)] final push to FALLBACK $FALLBACK_REPO: step $((10#$F))"
  fi
  if [ $PUSHED = 0 ]; then   # never delete the only copy of the weights: hold for a manual SSH rescue until the deadline
    echo "[$(ts)] FINAL PUSH FAILED - HOLDING POD (weights in $CKDIR; rescue over SSH before the deadline)"
    upload_log; sleep infinity
  fi
fi
if [ $RC -eq 0 ] && [ "${SKIP_EXPORT:-0}" != "1" ]; then
  for W in final ema; do
    [ $W = ema ] && [ ! -f $CKDIR/model_ema_${F}.pt ] && continue
    python -m scripts.export_hf --checkpoint-dir $CKDIR --tokenizer-dir $C/tokenizer --chat-template $C/tokenizer/chat_template.jinja \
      --out $WORK/export/$W --dtype bfloat16 --verify --weights $W 2>&1 | grep -v -i warn
    python $C/ops/pod_samples.py $WORK/export/$W $WORK/export/$W/samples.json 2>&1 | grep -v -i warn | tail -60
    python3 $C/ops/pod_hf.py put "$EXPORT_REPO" $WORK/export/$W "$RUN_NAME/$W"
  done
  grep -E "Validation bpb|web bpb|task bpb|EMA weights|Total training time|Peak memory|Minimum validation" "$LOG" | tail -12 > $WORK/export/summary.txt
  python3 $C/ops/pod_hf.py put "$EXPORT_REPO" $WORK/export/summary.txt "$RUN_NAME/summary.txt"
fi
if [ $RC -eq 0 ]; then
  grep -E "Validation bpb|web bpb|task bpb|EMA weights|Total training time|Peak memory|Minimum validation" "$LOG" | tail -12 > $WORK/summary.txt
  python3 $C/ops/pod_hf.py put "$CKPT_REPO" $WORK/summary.txt "$RUN_NAME/summary.txt"
  echo "DONE rc=0 step=$((10#$F)) $(ts)" > $WORK/DONE && python3 $C/ops/pod_hf.py put "$CKPT_REPO" $WORK/DONE "$RUN_NAME/DONE"
  echo "[$(ts)] ALL DONE"
fi
upload_log
selfdel "finished (rc=$RC)"
