"""Line-based math-verify helper, run with the data factory's venv (which has math-verify):
  datafactory/.venv/bin/python -I udgam3/sft/gen/mv_worker.py
stdin: one JSON {"pred", "gold"} per line -> stdout: {"ok": bool} per line.
"""
import json
import sys

from math_verify import parse, verify


def wrap(s):
    s = str(s).strip()
    return s if s.startswith("$") else "$" + s + "$"


for line in sys.stdin:
    try:
        q = json.loads(line)
        ok = bool(verify(parse(wrap(q["gold"])), parse(wrap(q["pred"]))))
    except Exception:  # noqa: BLE001
        ok = False
    sys.stdout.write(json.dumps({"ok": ok}) + "\n")
    sys.stdout.flush()
