#!/bin/bash
# Stand-in for sf_ssh.sh so the dashboard pusher (ops/train-dash/push/dash_push.py --ssh) and watch_main.py can read a
# run on a Colab session:  bash colab_ssh.sh STATE "remote shell command"   |   bash colab_ssh.sh STATE --cost
# STATE is ignored; the session is $COLAB_SESSION (default "ctx32k"; the Laghu SFT series uses "sft"). --cost prints "<STATUS> <price/h>": Colab bills compute units, not dollars,
# so the price is 0.00 and the dashboard shows the units in its notes.
S=${COLAB_SESSION:-ctx32k}; C="$HOME/.local/bin/colab"
# One exec at a time on the session (two concurrent execs hung a call on 2026-10-19): a mkdir lock, stale after 10 min.
LOCK=/tmp/colab_cli.lock   # shared with udgam3/evals/colab_lock.py: one Colab CLI call at a time, all sessions
for i in $(seq 1 600); do mkdir "$LOCK" 2>/dev/null && break
  [ -n "$(find "$LOCK" -maxdepth 0 -mmin +15 2>/dev/null)" ] && rmdir "$LOCK" 2>/dev/null; sleep 1; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
if [ "${2:-}" = "--cost" ]; then
  if "$C" --auth=adc status -s "$S" 2>/dev/null | grep -q "Status:"; then echo "RUNNING 0.00"; else echo "gone 0.00"; fi
  exit 0
fi
CMD="$2"
printf 'import subprocess, sys\nr = subprocess.run(%s, shell=True, capture_output=True, text=True)\nsys.stdout.write(r.stdout)\n' \
  "$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$CMD")" | "$C" --auth=adc exec -s "$S" 2>/dev/null
