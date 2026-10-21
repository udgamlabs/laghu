#!/usr/bin/env bash
# Create ONE Shadeform instance from CONFIG, start the Mac-side self-delete mirror (log: state/<RUN>_mirror.log), then
# watch the run with watch_main.py (alerts only). Usage: bash run_sf.sh CONFIG GPUS [EXPECT_SCALE] [NORMAL_OVER_TST]
set -u
cd "${LAGHU_REPO:-.}"
D=udgam3/runs/main/shadeform
out=$(OUT=$D/state bash $D/create_sf.sh "$1"); rc=$?; echo "$out"
[ $rc -ne 0 ] && exit $rc
RUN=$(set -a; source "$1"; echo "$RUN_NAME")
dl=$(python3 -c "import json; print(json.load(open('$D/state/${RUN}_sf.json'))['deadline_epoch'])")
SE=$(set -a; source "$1"; echo "$SAVE_EVERY")
nohup python3 $D/selfdel_mirror.py "$RUN" "$RUN" > $D/state/${RUN}_mirror.log 2>&1 &
echo "$(date +%H:%M) mirror started (pid $!, log $D/state/${RUN}_mirror.log); watching until $(date -r "$dl" +%H:%M)"
exec python3 udgam3/runs/main/watch_main.py "$RUN" "$RUN" --ssh $D/sf_ssh.sh --gpus "$2" --expect-scale "${3:-1.0}" --normal-over-tst "${4:-0.769}" \
  --deadline "$dl" --save-every "$SE" --every 60 --log-dir /udgam_ws/logs
