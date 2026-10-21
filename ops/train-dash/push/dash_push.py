"""Push a training run's live state to the private Udgam dashboard (Cloudflare Worker `udgam-train-dash`).

Reads only the NEW part of the run log each poll (over SSH, via an ssh helper with the ssh.sh interface), parses steps,
held-out evals, checkpoints and events, adds cost from Shadeform (optional), and POSTs one snapshot JSON.
Read-only on the training machine. Secrets from files, never printed: ~/.config/udgam/dash_push_secret.

  python3 ops/train-dash/push/dash_push.py --config ops/train-dash/push/configs/udgam3-base-v1.json \
      --ssh udgam3/runs/main/shadeform/sf_ssh.sh --state udgam3-base-v1 --remote-log /udgam_ws/logs/udgam3-base-v1.log \
      [--sf-id INSTANCE] [--every 60] [--dry] [--log-file LOCAL_LOG_FOR_TESTS] [--once]
"""
import argparse, json, os, re, statistics, subprocess, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.environ.get("LAGHU_REPO", ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_public import leaks
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--ssh"); ap.add_argument("--state")
ap.add_argument("--remote-log"); ap.add_argument("--sf-id"); ap.add_argument("--every", type=int, default=60)
ap.add_argument("--dry", action="store_true"); ap.add_argument("--once", action="store_true")
ap.add_argument("--log-file", help="TEST: parse this local log instead of the remote one")
ap.add_argument("--public-out", default=None, help="dry mode: also write the public snapshot to this file (for check_public.py)")
ap.add_argument("--url", default=None, help="dashboard base URL (default: ~/.config/udgam/dash_url)")
a = ap.parse_args()

CFG = json.load(open(a.config))
RUN = CFG["run"]; C = CFG["config"]
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", f"{RUN}{'_test' if a.log_file else ''}.json")
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
ISO = lambda dt: dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
now = lambda: datetime.now(timezone.utc)

STEP = re.compile(r"^step (\d+)/(\d+) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+) .*?\| total time: ([\d.]+)m")
EVAL = re.compile(r"^Step (\d+) \| (\w+) bpb: ([\d.]+|nan|inf)")
POS = re.compile(r"epoch: (\d+) pq: (\d+) rg: (\d+)")
TSL = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] (.*)$")
# scripts/udgam_sft.py (fine-tuning) lines: held-out values are mean loss per trained token (nats), not bpb
SFT_STEP = re.compile(r"^step (\d+)/(\d+) \| loss ([\d.]+) \| lrm ([\d.]+) \| trained tok [\d,]+ \| dt (\d+)ms \| tok/s ([\d,]+) \| mfu ([\d.]+)")
SFT_VAL = re.compile(r"^step (\d+) val (.+)$")


def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"offset": 0, "rem": "", "train": [], "evals": {}, "ckpts": [], "events": [], "seen": [], "start_iso": None,
                "last_steps": [], "ema": {}, "last_ts": None, "done": False, "failed": False, "tst_ended": False, "pb": False, "ema_on": False}


def add_event(S, t_iso, kind, text, key):
    if key in S["seen"]:
        return
    S["seen"].append(key); S["events"].append({"t": t_iso, "kind": kind, "text": text})


def last_t(S):
    return S.get("last_ts") or ISO(now())


def est_time(S, total_min):
    if S["start_iso"] and total_min is not None:
        return ISO(datetime.fromisoformat(S["start_iso"].replace("Z", "+00:00")) + timedelta(minutes=total_min + 2.5))
    return ISO(now())


def parse(S, text):
    lines = (S["rem"] + text).split("\n")
    S["rem"] = lines.pop()  # partial last line
    for raw in lines:
        l = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        if l.startswith("RL {"):   # udgam3/rl/laghu_rl.py: one JSON record per step; the "loss" slot carries the mean reward
            try:
                rec = json.loads(l[3:])
            except ValueError:
                continue
            st = int(rec["step"]); ss = float(rec.get("step_s") or 0)
            S["rl_min"] = S.get("rl_min", 0.0) + ss / 60
            toks = int(rec.get("ntok", 0) / ss) if ss else 0
            S["last_steps"] = (S["last_steps"] + [[st, ss * 1000, toks, 0.0, round(S["rl_min"], 2), rec["reward"], 0.0]])[-60:]
            S["train"].append([st, rec["reward"], toks, 0.0, 0.0])
            for k_ in ("acc", "cap_rate", "loop_rate", "informative_groups", "opd_rkl", "kl"):
                if k_ in rec:
                    arr = S["evals"].setdefault("rl_" + k_, [])
                    if not arr or arr[-1][0] != st:
                        arr.append([st, rec[k_]])
            continue
        m = SFT_STEP.match(l)
        if m:   # fine-tuning: same snapshot fields; elapsed minutes estimated from the step time
            st, tot, loss, lrm = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
            dt, toks, mfu = float(m.group(5)), int(m.group(6).replace(",", "")), float(m.group(7))
            S["sft_min"] = S.get("sft_min", 0.0) + dt * max(1, st - S.get("sft_last", 0)) / 60000; S["sft_last"] = st
            S["last_steps"] = (S["last_steps"] + [[st, dt, toks, mfu, round(S["sft_min"], 2), round(loss, 4), lrm]])[-60:]
            S["train"].append([st, round(loss, 4), toks, round(mfu, 1), lrm])
            for f in (0.25, 0.5, 0.75):
                if st >= f * tot and f"ms-{f}" not in S["seen"]:
                    S["seen"].append(f"ms-{f}")
                    S.setdefault("pub_events", []).append({"t": est_time(S, S["sft_min"]), "kind": "info", "text": f"{int(f * 100)}% of fine-tuning done"})
            continue
        m = SFT_VAL.match(l)
        if m:
            st = int(m.group(1))
            for part in m.group(2).split(" | "):
                name, _, v = part.rpartition(" ")
                try:
                    val = float(v)
                except ValueError:
                    continue
                arr = S["evals"].setdefault(name.replace(":", "_") or "all", [])
                if not arr or arr[-1][0] != st:
                    arr.append([st, val])
            continue
        m = STEP.match(l)
        if m:
            st, tot = int(m.group(1)), int(m.group(2))
            loss, lrm, dt, toks, mfu, tmin = float(m.group(4)), float(m.group(5)), float(m.group(6)), int(m.group(7).replace(",", "")), float(m.group(8)), float(m.group(9))
            S["last_steps"] = (S["last_steps"] + [[st, dt, toks, mfu, tmin, round(loss, 4), lrm]])[-60:]
            if st % 25 == 0:
                S["train"].append([st, round(loss, 4), toks, round(mfu, 1), lrm])
            for f in (0.1, 0.25, 0.5, 0.75, 0.9):
                if st >= f * tot and f"ms-{f}" not in S["seen"]:
                    S["seen"].append(f"ms-{f}")
                    S.setdefault("pub_events", []).append({"t": est_time(S, tmin), "kind": "info", "text": f"{int(f * 100)}% of training done"})
            mp = POS.search(l)
            if mp:
                S["data_pos"] = {"epoch": int(mp.group(1)), "pq": int(mp.group(2)), "rg": int(mp.group(3)), "phase": "B" if S["pb"] else "A"}
            if C.get("ema_start") and st >= C["ema_start"] and not S["ema_on"]:
                S["ema_on"] = True; add_event(S, est_time(S, tmin), "info", f"Weight averaging (EMA) switched on at step {st:,}", "ema")
            continue
        m = EVAL.match(l)
        if m:
            st, name, v = int(m.group(1)), m.group(2), m.group(3)
            val = float(v) if v not in ("nan", "inf") else None
            arr = S["evals"].setdefault(name, [])
            if not arr or arr[-1][0] != st:
                arr.append([st, val])
            if val is None:
                add_event(S, last_t(S), "bad", f"{name} held-out loss is {v} at step {st:,}", f"nan-{name}-{st}")
            continue
        if "TST: data rows now 2048 tokens (bag size 1)" in l and not S["tst_ended"]:
            S["tst_ended"] = True
            st = int(re.match(r"^Step (\d+)", l).group(1)) if l.startswith("Step") else None
            tm = S["last_steps"][-1][4] if S["last_steps"] else None
            add_event(S, est_time(S, tm), "info", f"Token bags (TST) ended at step {st:,}: next-token training from here" if st else "Token bags (TST) ended", "tst")
            continue
        if "switching training data to Phase B" in l and not S["pb"]:
            S["pb"] = True; tm = S["last_steps"][-1][4] if S["last_steps"] else None
            add_event(S, est_time(S, tm), "info", "Cool-down data mix switched on", "phaseb")
            continue
        if l.startswith("EMA weights:"):
            for k, v in re.findall(r"val_ema(?:_(\w+))?/bpb ([\d.]+)", l):
                S["ema"][k or "Validation"] = float(v)
            continue
        if "Traceback" in l or l.endswith("FAILED") or "BOOTSTRAP FAILED" in l or "START_FAILED" in l:
            S["failed"] = True; add_event(S, last_t(S), "bad", "Failure in the run log: " + l[:140], "fail-" + l[:60])
            continue
        if l.startswith("verify:") and "PASS" in l:
            add_event(S, last_t(S), "ok", "Export check passed: " + l.split("verify: ")[1][:90], "verify-" + str(len(S["seen"])))
            continue
        m = TSL.match(l)
        if m:
            t, msg = m.group(1), m.group(2)
            S["last_ts"] = t
            if msg.startswith("torchrun attempt"):
                if S["start_iso"] is None:
                    S["start_iso"] = t
                add_event(S, t, "info", "Training process started (" + msg + ")", "torchrun-" + t)
            elif msg.startswith("uploaded checkpoint step"):
                s = int(msg.split()[-1])
                if not any(c[0] == s for c in S["ckpts"]):
                    S["ckpts"].append([s, t])
                add_event(S, t, "ok", f"Checkpoint {s:,} saved to Hugging Face", f"ck-{s}")
            elif msg.startswith("final push"):
                add_event(S, t, "ok", "Final checkpoint saved to Hugging Face (" + msg + ")", "final-push")
            elif msg.startswith("training exited"):
                add_event(S, t, "ok" if "rc=0" in msg else "bad", "Training " + msg.replace("training ", ""), "exit-" + t)
            elif msg.startswith("ALL DONE"):
                S["done"] = True; add_event(S, t, "ok", "Run complete: weights exported, logs uploaded", "done")
            elif msg.startswith("data ready") or msg.startswith("python env ready") or msg.startswith("host:") or msg.startswith("container started"):
                add_event(S, t, "info", msg[:150], "boot-" + msg[:20])
            elif "DEADLINE" in msg or "HOLDING" in msg or "FAILED" in msg:
                S["failed"] = True; add_event(S, t, "bad", msg[:150], "x-" + t)


def read_new(S):
    if a.log_file:
        with open(a.log_file, "rb") as f:
            f.seek(S["offset"]); b = f.read(5_000_000)
        S["offset"] += len(b); return b.decode("utf-8", "replace")
    cmd = f"tail -c +{S['offset'] + 1} {a.remote_log} 2>/dev/null | head -c 5000000"
    r = subprocess.run(["bash", a.ssh, a.state, cmd], capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    if r.returncode != 0:
        return None
    S["offset"] += len(r.stdout)
    return r.stdout.decode("utf-8", "replace")


def sf_cost():
    if not a.sf_id:
        return None
    try:
        sys.path.insert(0, os.path.join(ROOT, "udgam3/runs/main/shadeform"))
        from sf_api import info
        d = info(a.sf_id)
        return {"so_far": float(d.get("cost_estimate") or 0), "price_h": (d.get("hourly_price") or 0) / 100, "status": d.get("status")}
    except Exception as e:
        return {"error": str(e)[:80]}


def snapshot(S, cost):
    ls = [s for s in S["last_steps"] if s[1] < C.get("max_step_ms", 10_000)]   # RL steps take minutes: set max_step_ms
    step = S["last_steps"][-1][0] if S["last_steps"] else 0
    total = C["steps_total"]; tst = C.get("tst_end", 0)
    p = {"step": step, "tokens_seen": step * C["batch"]}
    if ls:
        recent = ls[-20:]
        p["tok_s"] = int(statistics.median(s[2] for s in recent)); p["mfu"] = round(statistics.median(s[3] for s in recent), 1)
        p["step_ms"] = round(statistics.median(s[1] for s in recent))
        d_tst = statistics.median(s[1] for s in ls[-30:] if s[0] < tst) if any(s[0] < tst for s in ls) else None
        d_norm = statistics.median(s[1] for s in ls[-30:] if s[0] >= tst) if sum(1 for s in ls if s[0] >= tst) >= 10 else None
        if d_norm is None:
            d_norm = (d_tst or p["step_ms"]) * C.get("normal_over_tst", 1.0)
        d_tst = d_tst or d_norm
        rem_ms = max(0, tst - step) * d_tst + (total - max(step, tst)) * d_norm
        p["eta_min"] = round(rem_ms / 60000, 1)
        p["finish_iso"] = ISO(now() + timedelta(milliseconds=rem_ms) + timedelta(minutes=12))
    status = "done" if S["done"] else ("failed" if S["failed"] and not ls else ("training" if ls else "starting"))
    if cost and cost.get("status") in ("deleted", "deleting") and not S["done"]:
        status = "stopped"
    c = {}
    if cost and "so_far" in cost:
        c["so_far"] = round(cost["so_far"], 2)
        if p.get("eta_min") is not None and status == "training":
            c["projected"] = round(cost["so_far"] + (p["eta_min"] + 12) / 60 * (cost["price_h"] or C.get("price_h", 0)), 2)
        if p["tokens_seen"] > 1e8:
            c["per_b"] = round(cost["so_far"] / (p["tokens_seen"] / 1e9), 2)
    if status == "done":
        p["finished_iso"] = next((e["t"] for e in S["events"] if e["text"].startswith("Run complete")), None)
    ev = CFG.get("events", []) + S["events"]
    extra = {"data_pos": S.get("data_pos"), "gpus": gpu_health(), "monitors": monitor_health(),
             "compare": CFG.get("compare"), "compare_runs": CFG.get("compare_runs"), "targets": CFG.get("targets"), "next_steps": CFG.get("next_steps"), "budget": CFG.get("budget")}
    return {"run": RUN, "title": CFG.get("title", RUN), "status": status, "updated_iso": ISO(now()), "config": C, "refs": CFG.get("refs", []),
            "progress": p, "cost": c, "train": S["train"] + latest_point(S),
            "evals": S["evals"], "ema": S["ema"], "checkpoints": S["ckpts"], "events": ev, **extra}


def gpu_health():
    """Per-GPU use, memory, power and temperature from nvidia-smi on the training machine (read-only)."""
    if a.log_file:
        return None
    q = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits"
    try:
        r = subprocess.run(["bash", a.ssh, a.state, q], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
        rows = [[x.strip() for x in l.split(",")] for l in r.stdout.strip().splitlines() if l.count(",") == 5]
        return [{"i": int(x[0]), "util": float(x[1]), "mem": round(float(x[2]) / 1024, 1), "mem_total": round(float(x[3]) / 1024), "power": round(float(x[4])), "temp": int(float(x[5]))} for x in rows] or None
    except Exception:
        return None


def monitor_health():
    """Are the Mac-side safety processes alive, and what did each say last? (config "monitors": name, pgrep pattern, log)."""
    out = []
    for m in CFG.get("monitors", []):
        alive = subprocess.run(["pgrep", "-f", m["pgrep"]], capture_output=True).returncode == 0
        last = ""
        try:
            with open(m["log"], "rb") as f:
                f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4000))
                lines = [x for x in f.read().decode("utf-8", "replace").splitlines() if x.strip() and "Malloc" not in x]
                last = re.sub(r"\x1b\[[0-9;]*m", "", lines[-1])[:160] if lines else ""
        except Exception:
            pass
        out.append({"name": m["name"], "role": m.get("role", ""), "alive": alive, "last": last})
    return out


def latest_point(S):
    """The newest step as the last chart point (the series keeps every 25th step)."""
    if not S["last_steps"]:
        return []
    st, dt, toks, mfu, tmin, loss, lrm = S["last_steps"][-1]
    return [] if (S["train"] and S["train"][-1][0] == st) else [[st, loss, toks, round(mfu, 1), lrm]]


def public_snapshot(S, snap):
    """ALLOWLIST ONLY: the sanitized public view (user rule 2026-10-18). Nothing is copied wholesale from the private
    snapshot; every field below is chosen. No cost, provider, recipe, steps, batch, data, targets or monitoring."""
    P = CFG.get("public") or {}
    total = C["steps_total"]; frm = P.get("from_frac", 0.25) * total
    pr = snap["progress"]
    pct = lambda st: round(100 * st / total, 2)
    held = [[pct(st), v] for st, v in snap["evals"].get("fresh", []) if st >= frm and v is not None]
    train = [[pct(r[0]), r[1]] for i, r in enumerate(S["train"]) if r[0] >= frm and r[1] is not None and r[0] % 100 == 0]
    ev = list(P.get("events", [])) + list(S.get("pub_events", []))
    st0 = next((e["t"] for e in S["events"] if e["text"].startswith("Training process started")), None)
    if st0:
        ev.append({"t": st0, "kind": "info", "text": "Training started"})
    if S["done"]:
        ev.append({"t": next((e["t"] for e in S["events"] if e["text"].startswith("Run complete")), ISO(now())), "kind": "ok", "text": "Training finished; weights saved"})
    gp = [{"util": g["util"], "temp": g["temp"], "power": g["power"], "mem": g["mem"], "mem_total": g["mem_total"]} for g in (snap.get("gpus") or [])]
    return {"run": RUN + ".public", "kind": "public", "status": snap["status"] if snap["status"] in ("training", "done", "starting") else "paused",
            "updated_iso": snap["updated_iso"], "title": P.get("title"), "headline": P.get("headline"), "summary": P.get("summary"),
            "facts": P.get("facts", []), "started_iso": st0,
            "progress": {"pct": round(100 * pr.get("step", 0) / total, 2), "tokens_seen": pr.get("tokens_seen"), "tokens_goal": C["tokens_target"],
                         "tok_s": pr.get("tok_s"), "eta_min": pr.get("eta_min"), "finish_iso": pr.get("finish_iso")},
            "from_pct": round(100 * P.get("from_frac", 0.25)), "heldout": held, "train": train, "gpus": gp,
            "checkpoints": {"count": len(snap["checkpoints"]), "last_iso": snap["checkpoints"][-1][1] if snap["checkpoints"] else None},
            "events": sorted(ev, key=lambda e: e["t"]), "next": P.get("next", []), "note": P.get("note")}


def push(snap):
    url = a.url or open(os.path.expanduser("~/.config/udgam/dash_url")).read().strip()
    secret = open(os.path.expanduser("~/.config/udgam/dash_push_secret")).read().strip()
    body = json.dumps(snap, separators=(",", ":")).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/push", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json", "User-Agent": "udgam-dash-push/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, len(body)


while True:
    S = load_state()
    txt = read_new(S)
    if txt is not None:
        parse(S, txt)
        json.dump(S, open(STATE_FILE, "w"))
    snap = snapshot(S, sf_cost())
    if a.dry:
        s = dict(snap); s["train"] = f"{len(snap['train'])} points"; s["events"] = snap["events"][-6:]
        print(json.dumps({k: s[k] for k in ("status", "progress", "cost", "train", "ema", "checkpoints", "events")}, indent=1)[:4000])
        print("evals:", {k: v[-2:] for k, v in snap["evals"].items()}, "| snapshot bytes:", len(json.dumps(snap)))
        if CFG.get("public"):
            pub = public_snapshot(S, snap); print("PUBLIC keys:", sorted(pub))
            if a.public_out:
                open(a.public_out, "w").write(json.dumps(pub))
    else:
        try:
            code, n = push(snap)
            code2, n2 = (None, 0)
            if CFG.get("public"):
                pub = public_snapshot(S, snap)
                bad = leaks(json.dumps(pub))
                if bad:
                    code2 = "BLOCKED (leak check: " + ", ".join(bad) + ")"
                else:
                    code2, n2 = push(pub)
            print(f"{datetime.now():%H:%M:%S} pushed step {snap['progress'].get('step')} ({n/1024:.0f} KB) -> {code}; public ({n2/1024:.0f} KB) -> {code2}", flush=True)
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} push failed: {str(e)[:120]}", flush=True)
    if a.once or (snap["status"] in ("done", "stopped") and not a.log_file):
        break
    if a.log_file and txt == "":
        break
    time.sleep(a.every)
