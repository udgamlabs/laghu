#!/usr/bin/env bash
# Create a Runpod GPU training pod that runs ops/pod_train.sh from the private code repo (train/ prefix).
# Cost safety: the container start command FIRST starts a watchdog that deletes the pod at DEADLINE_EPOCH
# (= creation time + MAX_MIN minutes, absolute, so container restarts cannot extend it), before anything else runs.
# Udgam-3 frozen copy (code from udgam-build-code train-udgam3/). Usage: bash udgam3/runs/main/create_train_pod.sh CONFIG_FILE [--dry-run]
#   CONFIG_FILE (bash) sets: RUN_NAME GPU_TYPE GPU_COUNT CLOUD (SECURE|COMMUNITY) MAX_MIN CKPT_REPO EXPORT_REPO
#                            TRAIN_ARGS SAVE_EVERY WANDB_PROJECT [WANDB_RUN_ID] [TRIES] [DISK_GB] [RETRY_SLEEP]
# Prints "CREATED <pod id> ..." and writes $OUT/pod_created.json.
set -u
set -a; source "$1"; set +a
DRY=${2:-}
export TRIES=${TRIES:-10} OUT=${OUT:-/tmp} DISK_GB=${DISK_GB:-300} WANDB_RUN_ID=${WANDB_RUN_ID:-}
KEY=$(grep apikey ~/.runpod/config.toml | sed -E "s/.*= *['\"]?([^'\"]*)['\"]?/\1/")
export PUB=$(runpodctl template get ua2ipsh8lb | python3 -c "import json,sys; print(json.load(sys.stdin)['env']['PUBLIC_KEY'])")
export IMG=${IMAGE:-$(runpodctl template get ua2ipsh8lb | python3 -c "import json,sys; print(json.load(sys.stdin)['imageName'])")}
for i in $(seq 1 $TRIES); do
  export DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
  python3 > $OUT/train_req.json <<'PY'
import json, os
E = os.environ
wd = ('( L=$((DEADLINE_EPOCH - $(date +%s))); [ $L -gt 0 ] && sleep $L; echo WATCHDOG1 deadline >> /tmp/watchdog.log; '
      'curl -s -m 30 -X DELETE https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID -H "Authorization: Bearer $RUNPOD_API_KEY" >> /tmp/watchdog.log; '
      'sleep 60; curl -s -m 30 https://api.runpod.io/graphql -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" '
      '-d "{\\"query\\":\\"mutation { podTerminate(input: {podId: \\\\\\"$RUNPOD_POD_ID\\\\\\"}) }\\"}" >> /tmp/watchdog.log ) & ')
start = (wd + "(nohup /start.sh > /tmp/start.log 2>&1 &); sleep 2; "
         "pip install -q --break-system-packages --root-user-action=ignore 'huggingface_hub[hf_xet]==1.33.0' "
         "&& hf download $CODE_REPO --revision ${CODE_REV:-main} --include 'train-udgam3/*' --local-dir /workspace/code_dl --quiet "
         "&& mkdir -p /workspace/code && mv /workspace/code_dl/train-udgam3 /workspace/code/train "
         "&& exec bash /workspace/code/train/ops/pod_train.sh; echo START_FAILED; sleep infinity")
env = {"HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}", "WANDB_API_KEY": "{{ RUNPOD_SECRET_WANDB_API_KEY }}",
       "WANDB_ENTITY": "udgamlabs", "PUBLIC_KEY": E["PUB"], "DEADLINE_EPOCH": E["DEADLINE"],
       "RUN_NAME": E["RUN_NAME"], "CODE_REPO": "YOUR_HF_NAMESPACE/laghu-build-code", "DATA_REPO": E.get("DATA_REPO", "UdgamLabs/udgam-pretrain-v1"),
       "CKPT_REPO": E["CKPT_REPO"], "EXPORT_REPO": E["EXPORT_REPO"], "TRAIN_ARGS": E["TRAIN_ARGS"],
       "SAVE_EVERY": E["SAVE_EVERY"], "WANDB_PROJECT": E["WANDB_PROJECT"], "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
for k in ("DATA_PATTERNS", "SKIP_EXPORT", "FALLBACK_REPO", "EXTRA_DATA_JSON", "RESUME_FROM_REPO", "CODE_REV"):
    if E.get(k):
        env[k] = E[k]
if E.get("WANDB_RUN_ID"):
    env["WANDB_RUN_ID"] = E["WANDB_RUN_ID"]
print(json.dumps({
    "name": "udgam-" + E["RUN_NAME"], "computeType": "GPU", "cloudType": E["CLOUD"],
    "gpuTypeIds": [E["GPU_TYPE"]], "gpuCount": int(E["GPU_COUNT"]),
    # 2026-10-11: a host with an older driver failed with "OCI runtime create failed" for the cu129 image;
    # only accept hosts whose driver supports the image's CUDA
    "allowedCudaVersions": E.get("CUDA_VERSIONS", "12.9,13.0").split(","),
    "imageName": E["IMG"], "containerDiskInGb": int(E["DISK_GB"]), "volumeInGb": 0, "ports": ["22/tcp"],
    "env": env, "dockerStartCmd": ["bash", "-c", start]}))
PY
  if [ "$DRY" = "--dry-run" ]; then
    python3 -c "import json; d=json.load(open('$OUT/train_req.json')); d['env']['PUBLIC_KEY']='...'; print(json.dumps(d, indent=1)); print(d['dockerStartCmd'][2])"
    exit 0
  fi
  curl -s -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
       -d @$OUT/train_req.json > $OUT/train_create_resp.json
  if grep -q '"id"' $OUT/train_create_resp.json; then
    cp $OUT/train_create_resp.json $OUT/pod_created.json
    python3 -c "import json; d=json.load(open('$OUT/pod_created.json')); print('CREATED', d['id'], d.get('machine',{}).get('gpuTypeId') or d.get('gpu'), 'cost/h', d.get('costPerHr') or d.get('adjustedCostPerHr'), 'deadline_epoch', $DEADLINE)"
    exit 0
  fi
  echo "$(date -u +%T) attempt $i: $(head -c 300 $OUT/train_create_resp.json)"; sleep ${RETRY_SLEEP:-30}
done
exit 1
