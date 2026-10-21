#!/bin/bash
# Laghu RL on a Colab G4 (plan: udgam3/rl/RL_PLAN.md). Runs ON the VM in the background (started by colab_rl_run.py).
#   bash colab_rl_pipeline.sh STAGE
# STAGE = dry  : baseline dev evals of the SFT model (all sets) and of Gargi (maths sets), 2 full-shape steps of the maths
#                stage (GRPO + distillation) and 2 of the tool stage, a short eval of the result, upload, end
#         r1   : maths learnability gate, 40 steps from the SFT model (GRPO + distillation from Gargi), then dev evals
#         r2   : maths main stage, 200 more steps from r1's weights (pool continues after r1's prompts), then dev evals
#         r3   : tool stage, 100 steps from the kept maths model (Python tool, executed rewards, no teacher), then dev evals
# Every stage prints "RLP ..." lines; failure = "RL_FAILED: why"; end = "RL_DONE stage=...". The HF token is only on the
# VM while downloading and uploading; it is deleted at the end (and on failure).
set -uo pipefail
STAGE=${1:?stage}
B=/content/udgam3/rl; L=/content/rl_pipeline.log; OUT=/content/rl_out; REPO=YOUR_HF_NAMESPACE/laghu-rl
SFT_REPO=YOUR_HF_NAMESPACE/laghu-sft; SFT_REV=c8c36d1c852a0953f94fc7abcfdf58356c9c5a50; SFT_SUB=laghu-sft-v1-ep2/hf
TEACHER=udgamlabs/Gargi; TEACHER_REV=7e0e6dd39d28ca1a6f5f47ffe4f893fa0fb542bf
exec > >(tee -a $L) 2>&1
ts() { date -u +%T; }
cleanup() { rm -f /root/.cache/huggingface/token; }
fail() { echo "RL_FAILED: $1 $(ts)"; cleanup; exit 1; }
echo "RLP setup stage=$STAGE $(ts)"
cd $B && sha256sum -c SETS.sha256 2>/dev/null | grep -c ": OK" | grep -q 3 || fail "set hashes"
if [ ! -x /content/harness/.venv/bin/python ]; then
  pip -q install uv || fail uv
  mkdir -p /content/harness && cp $B/pyproject.toml $B/uv.lock /content/harness/
  (cd /content/harness && uv sync --frozen -q) || fail harness
fi
PY=/content/harness/.venv/bin/python
$PY - <<PY || fail "preflight upload"
import os
from huggingface_hub import HfApi
a = HfApi(); a.create_repo("$REPO", private=True, exist_ok=True)
assert a.model_info("$REPO").private, "repo is not private"
p = "/content/preflight.bin"; open(p, "wb").write(os.urandom(32 << 20))
a.upload_file(path_or_fileobj=p, path_in_repo="preflight/$STAGE.bin", repo_id="$REPO", commit_message="preflight")
a.delete_file("preflight/$STAGE.bin", repo_id="$REPO", commit_message="remove preflight"); os.remove(p)
print("preflight upload OK (private)")
PY
$PY - <<PY || fail downloads
from huggingface_hub import snapshot_download
snapshot_download("$SFT_REPO", revision="$SFT_REV", allow_patterns=["$SFT_SUB/*"], local_dir="/content/models/sft")
snapshot_download("$TEACHER", revision="$TEACHER_REV", allow_patterns=["*.json", "*.jinja", "model.safetensors"], local_dir="/content/models/gargi")
print("downloads ok")
PY
SFT=/content/models/sft/$SFT_SUB; T=/content/models/gargi; mkdir -p $OUT
$PY -c "import torch, transformers; print('RLP env', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
RL="$PY $B/laghu_rl.py --pool $B/pool_maths.jsonl --out-root $OUT --pps 32 --k 8 --lr 5e-6 --beta 0.02 --max-new 768"
EV="$PY $B/rl_eval.py"
up() {   # upload a folder or file under the stage's prefix
  $PY - "$1" "$2" <<'PY' || echo "RLP upload failed: $1"
import os, sys
from huggingface_hub import HfApi
src, dst = sys.argv[1], sys.argv[2]; a = HfApi()
(a.upload_folder(folder_path=src, path_in_repo=dst, repo_id="YOUR_HF_NAMESPACE/laghu-rl") if os.path.isdir(src)
 else a.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id="YOUR_HF_NAMESPACE/laghu-rl"))
print("RLP uploaded", dst)
PY
}
case $STAGE in
  dry)
    echo "RLP baseline evals $(ts)"
    $EV $SFT $OUT/eval_sft > $OUT/eval_sft.log 2>&1 || fail "eval sft"; tail -6 $OUT/eval_sft.log
    $EV $T $OUT/eval_gargi --sets gsm8k_think,math_think > $OUT/eval_gargi.log 2>&1 || fail "eval gargi"; tail -3 $OUT/eval_gargi.log
    echo "RLP train maths (2 full-shape steps) $(ts)"
    $RL --init $SFT --name dry-maths --steps 2 --modes think:0.85,plain:0.15 --teacher $T --opd 1.0 --save-every 2 > $OUT/dry-maths.log 2>&1 || { tail -20 $OUT/dry-maths.log; fail "train maths"; }
    grep "^RL " $OUT/dry-maths.log | cut -c1-600
    echo "RLP train tool (2 full-shape steps) $(ts)"
    $RL --init $OUT/dry-maths/hf --name dry-tool --steps 2 --modes tool:1.0 --start-offset 64 --save-every 2 > $OUT/dry-tool.log 2>&1 || { tail -20 $OUT/dry-tool.log; fail "train tool"; }
    grep "^RL " $OUT/dry-tool.log | cut -c1-600
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    echo "RLP eval result (short) $(ts)"
    $EV $OUT/dry-tool/hf $OUT/eval_dry --limit 20 > $OUT/eval_dry.log 2>&1 || fail "eval result"; tail -6 $OUT/eval_dry.log
    for x in eval_sft eval_gargi eval_dry; do up $OUT/$x dry/$x; done
    for x in dry-maths.log dry-tool.log dry-maths.log.jsonl dry-tool.log.jsonl eval_sft.log eval_gargi.log; do [ -f $OUT/$x ] && up $OUT/$x dry/logs/$x; done
    ;;
  r1|r2|r3)
    case $STAGE in   # weights to start from, steps, modes, teacher, first pool index (R1 + R2 use 40 x 32 + 200 x 32 prompts)
      r1) INIT=$SFT; N=40;  MODES=think:0.85,plain:0.15; OPD="--teacher $T --opd 1.0"; OFF=0;;
      r2) INIT=/content/models/prev/r1/hf; N=200; MODES=think:0.85,plain:0.15; OPD="--teacher $T --opd 1.0"; OFF=1280;;
      r3) INIT=/content/models/prev/${KEEP:-r2}/hf; N=100; MODES=tool:1.0; OPD=""; OFF=0;;
    esac
    if [ "$STAGE" != r1 ]; then
      P=$([ "$STAGE" = r2 ] && echo r1 || echo ${KEEP:-r2})
      $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', allow_patterns=['$P/hf/*'], local_dir='/content/models/prev')" || fail "download $P"
    fi
    echo "RLP train $STAGE from $INIT: $N steps, modes $MODES, opd [$OPD], pool offset $OFF $(ts)"
    $RL --init $INIT --ref $SFT --name $STAGE --steps $N --modes $MODES $OPD --start-offset $OFF --save-every 50 \
      > $OUT/$STAGE.log 2>&1 || { tail -20 $OUT/$STAGE.log; up $OUT/$STAGE.log $STAGE/logs/$STAGE.log; fail "train $STAGE"; }
    grep "^RL " $OUT/$STAGE.log | tail -3 | cut -c1-600
    echo "RLP eval $STAGE $(ts)"
    $EV $OUT/$STAGE/hf $OUT/eval_$STAGE > $OUT/eval_$STAGE.log 2>&1 || fail "eval $STAGE"; grep -E '^(gsm8k|math|chat)' $OUT/eval_$STAGE.log
    up $OUT/$STAGE/hf $STAGE/hf; up $OUT/eval_$STAGE $STAGE/eval
    for x in $STAGE.log $STAGE.log.jsonl eval_$STAGE.log; do up $OUT/$x $STAGE/logs/$x; done
    for d in $OUT/$STAGE-step*/hf; do [ -d "$d" ] && up $d $STAGE/snapshots/$(basename $(dirname $d))/hf; done
    ;;
  *) fail "unknown stage $STAGE";;
esac
up $L $STAGE/logs/rl_pipeline.log
cleanup
echo "RL_DONE stage=$STAGE $(ts)"
