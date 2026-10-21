#!/usr/bin/env bash
# Udgam-3 main run (user approval 2026-10-18, APPROVED_main.md): wait for an 8x H100 SXM Community node and create the
# approved pod ONCE. Every attempt first checks:
#   - the approved files are unchanged (APPROVED_main.sha256);
#   - no main pod already exists (one run at a time; never two main pods);
#   - the Runpod balance covers the hard cap ($158) plus a margin.
# A failed create costs nothing. On success it prints CREATED, writes the ssh helper's pod record and exits 0.
# Exit 1 = no node within MAXH hours, 2 = config changed, 3 = main pod already exists, 4 = balance too low.
# Usage: bash udgam3/runs/main/launch_when_free.sh   [MAXH=12 EVERY=180]
set -u
cd "${LAGHU_REPO:-.}"
D=udgam3/runs/main; OUTD=$D/state/main; mkdir -p $OUTD
MAXH=${MAXH:-12}; EVERY=${EVERY:-180}; MIN_BAL=${MIN_BAL:-165}; APPROVAL=${APPROVAL:-$D/APPROVED_main.sha256}
end=$(( $(date +%s) + MAXH * 3600 )); n=0
KEY=$(grep apikey ~/.runpod/config.toml | sed -E "s/.*= *['\"]?([^'\"]*)['\"]?/\1/")
stock() {
  local q
  q=$(python3 -c "import json; print(json.dumps({'query':'query { gpuTypes(input:{id:\"NVIDIA H100 80GB HBM3\"}) { lowestPrice(input:{gpuCount:8, secureCloud:false}) { stockStatus } } }'}))")
  curl -s -m 30 https://api.runpod.io/graphql -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" -d "$q" \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["data"]["gpuTypes"][0]["lowestPrice"]["stockStatus"])
except Exception: print("?")'
}
while [ "$(date +%s)" -lt "$end" ]; do
  n=$((n + 1)); ts=$(date +%H:%M)
  shasum -a 256 -s -c "$APPROVAL" || { echo "$ts CONFIG CHANGED since approval - not launching"; exit 2; }
  names=$(curl -s -m 30 -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods | python3 -c 'import json,sys
try: print(" ".join(p.get("name","") for p in json.load(sys.stdin)))
except Exception: print("LISTFAIL")')
  [ "$names" = "LISTFAIL" ] && { echo "$ts could not list pods - skipping this attempt"; sleep "$EVERY"; continue; }
  if echo " $names " | grep -q " udgam-udgam3-base-v1 "; then echo "$ts a main pod already exists - not creating another"; exit 3; fi
  bal=$(runpodctl user 2>/dev/null | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["clientBalance"]))' 2>/dev/null || echo 0)
  if [ "$bal" -lt "$MIN_BAL" ]; then echo "$ts balance \$$bal < \$$MIN_BAL - not launching"; exit 4; fi
  s=$(stock)
  r=$(OUT=$OUTD TRIES=1 RETRY_SLEEP=1 bash $D/create_train_pod.sh $D/main.conf 2>&1 | tail -2)
  if echo "$r" | grep -q "^CREATED"; then
    id=$(echo "$r" | grep "^CREATED" | awk '{print $2}')
    echo "{\"id\": \"$id\"}" > udgam3/runs/g1/pods/state/main_pod.json
    echo "$ts attempt $n (stock $s): $(echo "$r" | grep '^CREATED')"
    exit 0
  fi
  [ $((n % 10)) -eq 1 ] && echo "$ts attempt $n: no node (stock $s, balance \$$bal): $(echo "$r" | tr '\n' ' ' | cut -c1-160)"
  sleep "$EVERY"
done
echo "$(date +%H:%M) no 8x H100 Community node within ${MAXH} h ($n attempts)"; exit 1
