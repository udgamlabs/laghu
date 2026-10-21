#!/usr/bin/env bash
# Watch a dry/main training pod (read-only: never deletes). Prints new key log lines every 60 s; exits on DONE, a
# failure marker, the pod disappearing, or after MAX_MIN. Usage: bash watch_dry.sh <state-name> <run-name> [max_min]
cd "${LAGHU_REPO:-.}"
N=$1; RUN=$2; MAXM=${3:-50}; P=udgam3/runs/g1/pods; seen=0
for i in $(seq 1 $MAXM); do
  st=$(bash $P/ssh.sh $N --cost 2>/dev/null | awk '{print $1}')
  if [ "$st" != "RUNNING" ]; then echo "$(date +%H:%M) pod status: ${st:-gone}"; exit 0; fi
  out=$(bash $P/ssh.sh $N "grep -E 'python env ready|data ready|preflight|torchrun attempt|tok/sec|Validation bpb|web bpb|fresh bpb|uploaded checkpoint|final push|Peak memory|EMA|DONE|FAILED|Traceback|Error|DEADLINE|exported|self-delete' /workspace/logs/$RUN.log 2>/dev/null | grep -v '^\s*$'" </dev/null 2>/dev/null)
  n=$(echo "$out" | grep -c .)
  if [ "$n" -gt "$seen" ]; then echo "$out" | tail -n $((n - seen)) | sed "s/^/$(date +%H:%M) /" | grep -v -E 'step 0*[0-9]*[1-9]/|tok/sec' ; echo "$out" | grep 'tok/sec' | tail -1 | sed "s/^/$(date +%H:%M) LAST: /" | cut -c1-220; seen=$n; fi
  echo "$out" | grep -q -E "ALL DONE|DONE rc=" && { echo "$(date +%H:%M) DONE"; exit 0; }
  echo "$out" | grep -q -E "FAILED|Traceback" && { echo "$(date +%H:%M) FAILURE seen"; exit 1; }
  sleep 60
done
echo "watch timeout"
