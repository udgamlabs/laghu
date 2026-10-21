#!/usr/bin/env bash
# Udgam-3 main run, one background job: wait for a node and create the approved pod (launch_when_free.sh), then watch
# it (watch_main.py, read-only) until DONE, the pod is gone, or the first stop-rule alert. The job's exit code is the
# launcher's (1-4) if no pod was created, else the watcher's (0 = done/gone, 3-8 = alert).
set -u
cd "${LAGHU_REPO:-.}"
D=udgam3/runs/main
LOG=$D/state/main/launch.log; mkdir -p $D/state/main
bash $D/launch_when_free.sh | tee -a $LOG; rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && exit $rc
line=$(grep "CREATED" $LOG | tail -1)
id=$(echo "$line" | sed -E 's/.*CREATED ([a-z0-9]+) .*/\1/'); dl=$(echo "$line" | sed -E 's/.*deadline_epoch ([0-9]+).*/\1/')
case "$dl" in ''|*[!0-9]*) echo "could not parse the deadline from: $line"; exit 9;; esac
echo "$(date +%H:%M) watching pod $id, deadline $(date -r "$dl" +%H:%M)"
exec python3 $D/watch_main.py main udgam3-base-v1 --gpus 8 --deadline "$dl" --save-every 2000 --every 120
