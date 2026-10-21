"""Backfill a W&B run from the training log saved on Hugging Face, for steps the live sync missed (2026-10-18: the live
mirror hung at 20:07 IST on a dead SSH connection, ~step 43k; the instance was deleted after the run, so its offline
W&B file is gone). Appends only steps after the run's last logged step, with the training code's own metric names.

  nanochat/.venv/bin/python udgam3/runs/main/shadeform/wandb_backfill.py            # dry run: counts only
  nanochat/.venv/bin/python udgam3/runs/main/shadeform/wandb_backfill.py --yes      # append to the W&B run
"""
import argparse
import re

import wandb
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--entity", default="udgamlabs"); ap.add_argument("--project", default="udgam3-main")
ap.add_argument("--run-id", default="udgam3-base-v1"); ap.add_argument("--repo", default="YOUR_HF_NAMESPACE/laghu-base-ckpt")
ap.add_argument("--log", default="udgam3-base-v1/logs/20261018T200830_pod.log"); ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

STEP = re.compile(r"^step (\d+)/\d+ \([\d.]+%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| "
                  r"bf16_mfu: ([\d.]+) \| epoch: (.+?) \| total time: ([\d.]+)m")
EVAL = re.compile(r"^Step (\d+) \| (\w[\w-]*) bpb: ([\d.]+)")
EMA = re.compile(r"^EMA weights: (.+)$")
rows, last_step = [], 0
for line in open(hf_hub_download(a.repo, a.log), errors="replace"):
    line = line.rstrip("\n")
    if m := STEP.match(line):
        s = int(m.group(1)); last_step = s
        if s % 100 == 0:   # the training code logs train/* every 100 steps
            rows.append((s, 0, {"step": s, "total_training_time": float(m.group(8)) * 60, "train/loss": float(m.group(2)),
                                "train/lrm": float(m.group(3)), "train/dt": float(m.group(4)) / 1000,
                                "train/tok_per_sec": int(m.group(5).replace(",", "")), "train/mfu": float(m.group(6)),
                                "train/epoch": m.group(7)}))
    elif m := EVAL.match(line):
        s, k, v = int(m.group(1)), m.group(2), float(m.group(3))
        rows.append((s, 1, {"step": s, ("val/bpb" if k == "Validation" else f"val_{k}/bpb"): v}))
    elif m := EMA.match(line):
        d = {"step": last_step}
        for part in m.group(1).split(" | "):
            k, _, v = part.rpartition(" ")
            try:
                d[k] = float(v)
            except ValueError:
                pass
        rows.append((last_step, 2, d))
# merge eval lines of the same step into one log call, as the training code does
merged = {}
for s, kind, d in rows:
    merged.setdefault((s, kind), {}).update(d)
ordered = [merged[k] for k in sorted(merged)]

api = wandb.Api()
run = api.run(f"{a.entity}/{a.project}/{a.run_id}")
have = max((r.get("step") or 0 for r in run.scan_history(keys=["step"])), default=0)
todo = [d for d in ordered if d["step"] > have]
print(f"log rows {len(ordered)} (last step {last_step}); W&B has up to step {have}; to append: {len(todo)} rows, "
      f"steps {todo[0]['step'] if todo else '-'}..{todo[-1]['step'] if todo else '-'}")
if not a.yes or not todo:
    raise SystemExit("dry run: add --yes to append" if todo else "nothing to append")
r = wandb.init(entity=a.entity, project=a.project, id=a.run_id, resume="must")
for d in todo:
    r.log(d)
r.summary["backfilled_from_log"] = f"steps {todo[0]['step']}..{todo[-1]['step']} from {a.repo}/{a.log}"
r.finish()
print("appended", len(todo), "rows")
