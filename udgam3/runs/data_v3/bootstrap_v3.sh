#!/usr/bin/env bash
# Udgam-3 data build (frozen per-run copy; do not edit while a pod runs it). Adapted from ops/pod_build/bootstrap_v2.sh.
# On a Runpod CPU pod: deps -> smoke build (v3) + checks -> small G1 pools (v3, v3s) -> full v3 pool -> verified uploads.
#   1. smoke: build_mix.py --mix v3 --scale 0.0002 ; test_mix.py ; smoke_check.py   (fail fast, cents)
#   2. G1 pools: --mix v3 and --mix v3s at SMALL_SCALE -> $G1_REPO_V3 / $G1_REPO_V3S (gate arms D1, D2; D0 uses v2b + v3 phase B)
#   3. full: --mix v3 at SCALE -> $DATA_REPO (main run)
# Every repo is verified after upload; the pod self-deletes only after the last verified upload (or on failure, after logs).
set -uo pipefail
[ -f /etc/rp_environment ] && source /etc/rp_environment
: "${HF_TOKEN:?}" "${CODE_REPO:?}" "${DATA_REPO:?}" "${G1_REPO_V3:?}" "${G1_REPO_V3S:?}"
MAX_HOURS="${MAX_HOURS:-6}"; SCALE="${SCALE:-5.0}"; SMALL_SCALE="${SMALL_SCALE:-0.3}"
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
C=/code; W=/work; LOGS=$W/logs; mkdir -p $LOGS
IDX=$W/index_13gram_v3.npz          # rebuilt on the pod: adds MATH-500 to the eval sets
exec > >(tee -a $LOGS/bootstrap.log) 2>&1
ts() { date -u +%FT%TZ; }
echo "[$(ts)] BOOTSTRAP v3 pod=${RUNPOD_POD_ID:-?} scale=$SCALE small=$SMALL_SCALE max_hours=$MAX_HOURS"
selfdel() { echo "[$(ts)] self-delete: $1"; python3 $C/ops/pod_build/pod_delete.py; sleep infinity; }
if [ -f $W/.started ]; then echo "[$(ts)] BUILD_FAILED: container restarted (bootstrap already ran once)"; selfdel "restart guard"; fi
touch $W/.started
( sleep "$(python3 -c "print(int(float('$MAX_HOURS')*3600))")"; echo "[$(ts)] BUILD_FAILED: MAX RUNTIME ${MAX_HOURS}h HIT"; \
  python3 $C/ops/pod_build/pod_delete.py; pkill -f build_mix.py ) &
up_logs() { hf upload "$DATA_REPO" $LOGS logs --repo-type dataset --quiet >/dev/null 2>&1 || echo "(log upload failed)"; }
fail() { echo "[$(ts)] BUILD_FAILED: $1"; up_logs; selfdel "failed"; exit 1; }

# ---------------------------------------------------------------- 1) deps
NCPU=$(python3 $C/ops/pod_build/cpu_info.py); WORKERS="${BUILD_WORKERS:-$NCPU}"; NCPU=$(( NCPU < WORKERS ? NCPU : WORKERS ))
echo "vCPUs usable: $NCPU; build workers: $WORKERS"; grep -E "MemTotal|MemAvailable" /proc/meminfo; df -h $W | tail -1
pip install -q --no-cache-dir --root-user-action=ignore "pyarrow==25.0.1" "tokenizers==0.23.2" "transformers==5.18.0" \
    "huggingface_hub==1.33.0" "numpy==2.5.3" "jinja2==3.1.6" 2>&1 | tail -3 || fail "pip install"
python3 -c "import pyarrow,tokenizers,transformers,huggingface_hub,numpy;print('deps ok')" || fail "imports"
hf auth whoami | head -1
for R in "$DATA_REPO" "$G1_REPO_V3" "$G1_REPO_V3S"; do hf repos create "$R" --type dataset --private --exist-ok 2>&1 | tail -1 || fail "create $R"; done
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 RAYON_NUM_THREADS=$NCPU
EXTRA_BASE="--decontam-index $IDX --seed 1234"

# ---------------------------------------------------------------- helpers
stage_and_upload() {   # $1 build dir, $2 repo, $3 description
  local B=$1 REPO=$2 DESC=$3 ST=$1/_stage
  python3 - "$B" "$ST" <<'PY' || return 1
import json, os, sys
B, st = sys.argv[1], sys.argv[2]; m = json.load(open(f"{B}/manifest.json"))
ph = {s["file"]: s["phase"] for s in m["shards"]}
val = [f for f, p in ph.items() if p == "val"]; assert len(val) == 1, val
layout = {"phaseA": sorted(f for f, p in ph.items() if p == "A") + val, "phaseB": sorted(f for f, p in ph.items() if p == "B") + val}
for d, files in layout.items():
    os.makedirs(f"{st}/shards/{d}", exist_ok=True)
    for f in files:
        os.link(f"{B}/{f}", f"{st}/shards/{d}/{f}")
    assert sorted(files)[-1] == val[0]          # nanochat uses the LAST shard (sorted) as validation
os.link(f"{B}/manifest.json", f"{st}/manifest.json")
os.makedirs(f"{st}/_meta", exist_ok=True)
for f in os.listdir(f"{B}/_meta"):
    os.link(f"{B}/_meta/{f}", f"{st}/_meta/{f}")
json.dump(dict(layout=layout, val_shard=val[0], tokens=m["totals"]["phase_tokens"],
               usage="NANOCHAT_DATA_DIR=shards/phaseA, --phase-b-data-dir shards/phaseB (val shard is last in both)"),
          open(f"{st}/phase_layout.json", "w"), indent=1)
print("staged", {k: len(v) for k, v in layout.items()})
PY
  mkdir -p $ST/tokenizer_v1 && cp $C/build_out/tokenizer_v1/* $ST/tokenizer_v1/
  printf "# %s (PRIVATE)\n%s\nBuilt by udgam/build_mix.py on %s. Licences and provenance per source: manifest.json; data/LICENSES_AND_PROVENANCE.md in the code repo.\n" \
    "$(basename $REPO)" "$DESC" "$(date -u +%F)" > $ST/README.md
  local U_RC=1
  for i in 1 2 3; do
    hf upload-large-folder "$REPO" $ST --repo-type dataset --private --num-workers 16 > $LOGS/upload_$(basename $REPO).log 2>&1 && U_RC=0 && break
    echo "upload attempt $i to $REPO failed"; tail -5 $LOGS/upload_$(basename $REPO).log; sleep 30
  done
  [ $U_RC -eq 0 ] || return 1
  python3 - "$REPO" "$ST" <<'PY' || return 1
import json, os, sys
from huggingface_hub import HfApi
repo, st = sys.argv[1], sys.argv[2]
want = sorted(os.path.relpath(os.path.join(d, f), st) for d, _, fs in os.walk(st) for f in fs
              if not os.path.relpath(d, st).startswith(".cache"))   # hf upload-large-folder writes .cache/ locks
have = set(HfApi().list_repo_files(repo, repo_type="dataset"))
missing = [f for f in want if f not in have]
print(f"verify {repo}: {len(want)} local files, {len(missing)} missing")
assert not missing, missing[:10]
PY
}

# ---------------------------------------------------------------- 2) smoke build
echo "[$(ts)] SMOKE build (v3, scale 0.0002)"
python3 $C/udgam/build_mix.py --mix v3 --out $W/smoke --scale 0.0002 --workers $WORKERS --shard-chars 2e6 $EXTRA_BASE > $LOGS/smoke_build.log 2>&1 \
    || { tail -40 $LOGS/smoke_build.log; fail "smoke build"; }
tail -25 $LOGS/smoke_build.log
python3 $C/udgam/test_mix.py $W/smoke --samples 1 --chars 300 > $LOGS/smoke_test_mix.txt 2>&1; SM_RC=$?
grep -E "FAIL|FAILURES|plain" $LOGS/smoke_test_mix.txt | head -20
[ $SM_RC -eq 0 ] || fail "smoke test_mix rc=$SM_RC"
cp $W/smoke/manifest.json $LOGS/smoke_manifest.json; rm -rf $W/smoke
echo "[$(ts)] SMOKE OK"

# ---------------------------------------------------------------- 3) small G1 pools (D1 = v3, D2 = v3s)
for M in ${G1_MIXES:-v3 v3s}; do      # G1_MIXES=v3s after the first attempt uploaded v3 (verified separately)
  REPO=$G1_REPO_V3; [ $M = v3s ] && REPO=$G1_REPO_V3S
  echo "[$(ts)] G1 pool --mix $M scale=$SMALL_SCALE -> $REPO"
  python3 $C/udgam/build_mix.py --mix $M --out $W/g1_$M --scale $SMALL_SCALE --workers $WORKERS --assemble-workers ${ASSEMBLE_WORKERS:-8} $EXTRA_BASE \
      > $LOGS/build_g1_$M.log 2>&1 || { tail -40 $LOGS/build_g1_$M.log; fail "G1 build $M"; }
  python3 $C/udgam/test_mix.py $W/g1_$M --rg-per-shard 2 --samples 1 --chars 300 > $LOGS/test_mix_g1_$M.txt 2>&1 || fail "G1 test_mix $M"
  grep -E "target|phase [AB]:" $LOGS/test_mix_g1_$M.txt | head -24
  stage_and_upload $W/g1_$M $REPO "Udgam-3 G1 gate pool, --mix $M at scale $SMALL_SCALE (65M proxy arms)." || fail "upload G1 $M"
  echo "[$(ts)] G1_POOL_DONE $M"; up_logs
  rm -rf $W/g1_$M
done

# ---------------------------------------------------------------- 4) full v3 pool
echo "[$(ts)] FULL build --mix v3 scale=$SCALE"
( while sleep 300; do echo "[$(ts)] mem: $(awk '/MemAvailable/{printf "%.1f GB avail", $2/1048576}' /proc/meminfo); disk: $(df -h $W | awk 'NR==2{print $3" used"}')"; done ) &
MEMPID=$!
python3 $C/udgam/build_mix.py --mix v3 --out $W/mix --scale $SCALE --workers $WORKERS --assemble-workers ${ASSEMBLE_WORKERS:-8} $EXTRA_BASE \
    2>&1 | tee $LOGS/build_mix.log | grep --line-buffered -E "^ *\[ *[0-9]+s\]|plan|wrote|ERROR|WARNING|streamed|Traceback" \
    | awk '{ if (NR%8==1 || $0 !~ /^ *\[/) { print; fflush() } }' ; B_RC=${PIPESTATUS[0]}
kill $MEMPID 2>/dev/null
[ $B_RC -eq 0 ] && [ -f $W/mix/manifest.json ] || { tail -60 $LOGS/build_mix.log; fail "full build rc=$B_RC"; }
python3 $C/udgam/test_mix.py $W/mix --rg-per-shard 2 --samples 2 --chars 500 > $LOGS/test_mix.txt 2>&1; T_RC=$?
grep -E "FAIL|FAILURES|target|phase [AB]:" $LOGS/test_mix.txt | head -40
[ $T_RC -eq 0 ] || fail "full test_mix rc=$T_RC"
stage_and_upload $W/mix $DATA_REPO "Udgam-3 pretraining pool (--mix v3, scale $SCALE): phase A = stable mix, phase B = anneal." || fail "upload full"
up_logs
echo "[$(ts)] BUILD_DONE"
selfdel "done"
