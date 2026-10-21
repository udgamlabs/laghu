#!/usr/bin/env bash
# Create the Udgam-3 v3 data-build CPU pod from pod_req_v3.json (approved 2026-10-17: 32 vCPU, 320 GB disk (was 400, no stock), 6 h cap).
# Tries 32-vCPU flavours only (cpu3g, cpu5g, cpu3c, cpu5c); never changes vCPU count, disk or the time cap.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; OUT=${OUT:-$D/state}; mkdir -p "$OUT"
KEY=$(grep apikey ~/.runpod/config.toml | sed -E "s/.*= *['\"]?([^'\"]*)['\"]?/\1/")
for flavor in cpu3g cpu5g cpu3c cpu5c; do
  python3 -c "import json,sys; r=json.load(open('$D/pod_req_v3.json')); r['cpuFlavorIds']=['$flavor']; print(json.dumps(r))" > "$OUT/req.json"
  curl -s -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d @"$OUT/req.json" > "$OUT/create_resp.json"
  if grep -q '"id"' "$OUT/create_resp.json"; then
    python3 -c "import json; d=json.load(open('$OUT/create_resp.json')); print('CREATED', d.get('id'), d.get('name'), 'flavor=$flavor', 'costPerHr=', d.get('costPerHr'))"
    exit 0
  fi
  echo "no stock for $flavor: $(head -c 200 "$OUT/create_resp.json")"
done
exit 1
