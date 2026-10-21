"""Read-only watcher for the Udgam-3 main run: the MAIN_RUN_PLAN stop rules as code.

Never deletes or changes anything. It exits with a non-zero code on the first alert, so the session is notified and
a person decides what to do. Every ~10 minutes it prints one progress line.

  python3 udgam3/runs/main/watch_main.py STATE_NAME RUN_NAME --gpus 8 --deadline EPOCH --save-every 2000 \
      [--ckpt-repo YOUR_HF_NAMESPACE/laghu-base-ckpt] [--every 120]

Alert exit codes: 3 slow host, 4 cannot finish before the deadline, 5 NaN in held-out loss, 6 held-out loss rose >5%
on two or more sets, 7 checkpoints not landing, 8 traceback / failure marker in the log, 9 no training step within
20 minutes of the watcher start (stuck pod start, data download or compile). 0 = DONE or pod gone.
"""
import os
import argparse, json, math, re, statistics, subprocess, sys, time
from datetime import datetime

ROOT = os.environ.get("LAGHU_REPO", ".")
SSH = f"{ROOT}/udgam3/runs/g1/pods/ssh.sh"
TST_TOKS, NORM_TOKS = 224_000, 291_000          # dry run A, 1x H100 SXM [MEASURED]
EFF, SLOW = 0.95, 0.80                          # assumed multi-GPU efficiency; alert below 80% of the expectation

ap = argparse.ArgumentParser()
ap.add_argument("state"); ap.add_argument("run")
ap.add_argument("--gpus", type=int, required=True); ap.add_argument("--deadline", type=int, required=True)
ap.add_argument("--save-every", type=int, default=2000); ap.add_argument("--ckpt-repo", default="YOUR_HF_NAMESPACE/laghu-base-ckpt")
ap.add_argument("--every", type=int, default=120)
ap.add_argument("--log-file", default=None, help="TEST ONLY: read this local log instead of the pod")
ap.add_argument("--ssh", default=SSH, help="ssh helper with the ssh.sh interface (Runpod default; Shadeform: shadeform/sf_ssh.sh)")
ap.add_argument("--log-dir", default="/workspace/logs", help="run-log folder on the host (Shadeform VMs: /udgam_ws/logs)")
ap.add_argument("--normal-over-tst", type=float, default=1.80 / 2.34, help="normal-phase step time / TST step time until measured (H100 SXM 0.77; A100 ~1.0)")
ap.add_argument("--expect-scale", type=float, default=1.0, help="scale the per-GPU speed expectation (e.g. 0.5 for a PCIe smoke test)")
a = ap.parse_args()


def ssh(cmd):
    if a.log_file:
        return open(a.log_file).read()
    try:
        r = subprocess.run(["bash", a.ssh, a.state, cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=90)
    except subprocess.TimeoutExpired:  # a slow SSH read is a retry, not a failure (crashed the watcher on 2026-10-18)
        return None
    return r.stdout


def status():
    if a.log_file:
        return "RUNNING"
    try:
        r = subprocess.run(["bash", a.ssh, a.state, "--cost"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    return (r.stdout.split() or ["gone"])[0]


def hf_latest():
    """Newest checkpoint step on HF (LATEST), 0 if the run has none yet, or None if the check itself failed.
    Runs in the project env (the Mac's system python has no huggingface_hub: 2026-10-18 false alert)."""
    code = ("import sys\nfrom huggingface_hub import hf_hub_download\nfrom huggingface_hub.errors import EntryNotFoundError\n"
            f"try:\n    print(open(hf_hub_download('{a.ckpt_repo}', '{a.run}/LATEST', force_download=True)).read().strip())\n"
            "except EntryNotFoundError:\n    print(0)\n")
    try:
        r = subprocess.run(["uv", "run", "--project", f"{ROOT}/evals/harness", "python", "-c", code],
                           capture_output=True, text=True, timeout=180)
        return int(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def say(msg):
    print(f"{datetime.now():%H:%M} {msg}", flush=True)


def alert(code, msg):
    say(f"ALERT {msg}")
    sys.exit(code)


STEP = re.compile(r"^step (\d+)/(\d+) .*?dt: ([\d.]+)ms \| tok/sec: ([\d,]+).*?(?:eta: ([\d.]+)m)?$")
EVAL = re.compile(r"^Step (\d+) \| (\w+) bpb: ([\w.+-]+)")
UPL = re.compile(r"uploaded checkpoint step (\d+)")
last_print, last_hf_check, prev_eval, t_start = 0, time.time(), {}, time.time()
tst_end = None
last_step, last_progress, total_steps = -1, time.time(), None


def check_stall():
    """No new step (or no answer from the machine) for 20 min mid-run -> alert. Not after the last step (final export)."""
    if last_step >= 0 and not (total_steps and last_step >= total_steps - 1) and time.time() - last_progress > 1200:
        alert(9, f"no new training step for {(time.time() - last_progress) / 60:.0f} min (last seen {last_step}), "
                 "or the machine did not answer")


while True:
    st = status()
    log = ssh(f"tail -n 3000 {a.log_dir}/{a.run}.log 2>/dev/null") if st == "RUNNING" else None
    if st is None or (st == "RUNNING" and log is None):
        say("slow SSH/API answer; retrying"); check_stall(); time.sleep(a.every); continue
    if st != "RUNNING":
        say(f"pod status: {st}"); sys.exit(0)
    lines = log.splitlines()
    if any(("Traceback" in l) or ("FAILED" in l) or ("START_FAILED" in l) for l in lines):
        bad = [l for l in lines if ("Traceback" in l) or ("FAILED" in l)][-1]
        alert(8, f"failure marker in log: {bad[:200]}")
    if any(("ALL DONE" in l) or ("DONE rc=" in l) for l in lines):
        say("DONE"); sys.exit(0)
    for l in lines:
        m = re.search(r"TST: bag size \d+ for steps \[0, (\d+)\)", l)
        if m:
            tst_end = int(m.group(1))
    steps = [(int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4).replace(",", "")), m.group(5))
             for m in (STEP.match(l) for l in lines) if m]
    if not steps and time.time() - t_start > 1200:
        tailtxt = ' | '.join(l[:120] for l in lines[-3:])
        alert(9, f"no training step 20 min after the watcher started; log tail: {tailtxt or '(no log / no ssh)'}")
    if steps:
        step, total, _, _, eta = steps[-1]
        total_steps = total
        if step > last_step:
            last_step, last_progress = step, time.time()
        check_stall()
        # speed: median of the last 30 steady steps (skip compile/recompile stalls), per phase
        in_tst = tst_end is not None and step < tst_end
        recent = [s for s in steps[-40:] if s[2] < 10_000 and ((s[0] < tst_end) == in_tst if tst_end else True) and s[0] >= 20]
        if len(recent) >= 20:
            med = statistics.median(s[3] for s in recent)
            exp = (TST_TOKS if in_tst else NORM_TOKS) * a.gpus * EFF * a.expect_scale
            if med < SLOW * exp:
                alert(3, f"slow host: median {med:,.0f} tok/s vs expected {exp:,.0f} ({'TST' if in_tst else 'normal'} phase), step {step}")
        # ETA vs deadline, phase-aware (nanochat's own eta averages all steps so far, so during TST it extrapolates the
        # slower TST rate to the whole run). Normal-phase dt is measured once available, else TST dt x 1.80/2.34
        # (dry run A). 15 min for the final save, export and uploads.
        steady = [s for s in steps if s[2] < 10_000 and s[0] >= 20]
        tst_dt = [s[2] for s in steady if tst_end and s[0] < tst_end][-30:]
        norm_dt = [s[2] for s in steady if not tst_end or s[0] >= tst_end][-30:]
        if step > 200 and (tst_dt or norm_dt):
            d_norm = statistics.median(norm_dt) if len(norm_dt) >= 10 else statistics.median(tst_dt) * a.normal_over_tst
            d_tst = statistics.median(tst_dt) if tst_dt else d_norm
            te = tst_end or 0
            rem_s = (max(0, te - step) * d_tst + (total - max(step, te)) * d_norm) / 1000
            finish = time.time() + rem_s + 900
            if finish > a.deadline:
                alert(4, f"phase-aware ETA {rem_s/60:.0f} min + 15 min export ends after the deadline "
                         f"({datetime.fromtimestamp(a.deadline):%H:%M}), step {step}/{total}")
            eta = f"{rem_s/60:.0f}"
        # checkpoints: two missed save boundaries -> alert
        ups = [int(m.group(1)) for m in (UPL.search(l) for l in lines) if m]
        last_up = max(ups) if ups else 0
        due = (step // a.save_every) * a.save_every
        if due - last_up >= 2 * a.save_every:
            alert(7, f"checkpoints not landing: last uploaded {last_up}, step {step}")
        if time.time() - last_hf_check > 1800 and last_up:
            h = hf_latest(); last_hf_check = time.time()
            if h is None:
                say("note: could not read HF LATEST this time (will retry in 30 min)")
            elif h < last_up - a.save_every:
                alert(7, f"HF LATEST is {h}, the log says uploaded {last_up}")
    # held-out evals
    evals = {}
    for m in (EVAL.match(l) for l in lines):
        if m:
            evals.setdefault(int(m.group(1)), {})[m.group(2)] = m.group(3)
    if evals:
        s = max(evals)
        cur = evals[s]
        if any(v.lower() in ("nan", "inf") or not math.isfinite(float(v)) for v in cur.values() if v.replace(".", "").replace("-", "").isalnum()):
            alert(5, f"NaN/inf in held-out loss at step {s}: {cur}")
        earlier = [k for k in evals if k < s]
        if earlier:
            p = evals[max(earlier)]
            rises = [k for k in cur if k in p and float(cur[k]) > 1.05 * float(p[k])]
            if len(rises) >= 2:
                alert(6, f"held-out loss up >5% on {rises} at step {s}: {p} -> {cur}")
        prev_eval = cur
    if time.time() - last_print > 600 and steps:
        step, total, dt, toks, _ = steps[-1]
        ev = ", ".join(f"{k} {v}" for k, v in (evals[max(evals)].items() if evals else []))
        say(f"step {step}/{total} ({100*step/total:.1f}%) {toks:,} tok/s eta {eta} min | eval@{max(evals) if evals else '-'}: {ev}")
        last_print = time.time()
    time.sleep(a.every)
