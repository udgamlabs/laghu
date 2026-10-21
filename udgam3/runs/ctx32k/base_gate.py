"""Benchmark gate for the 32K stretch (pre-registration 6.3): base suite on validation splits, stretch vs main-run EMA,
on the ctx32k Colab session after CTX_DONE. Pass = the 5-task average drops by <= 1 point.

  python3 udgam3/runs/ctx32k/base_gate.py start     # upload token + harness lock + task files + gate_vm.sh, start it
  python3 udgam3/runs/ctx32k/base_gate.py poll      # wait for GATE_DONE / GATE_FAILED, download, score
"""
import json, os, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam3", "evals"))
from colab_lock import colab as locked  # noqa: E402  (one Colab CLI call at a time, all sessions)

S = "ctx32k"; OUT = os.path.join(HERE, "gate")
# the Phase 0 table's metric per task (acc_norm where the CUDA table uses it)
METRIC = {"arc_easy_val": "acc", "arc_challenge_val": "acc_norm", "hellaswag": "acc_norm", "piqa": "acc_norm", "winogrande": "acc"}


def colab(*args, stdin=None, timeout=600):
    for _ in range(3):
        try:
            return locked(*args, stdin=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  (colab {args[0]} hung; retrying)", flush=True)
    raise SystemExit("colab call hung 3 times")


def ex(code, timeout=300):
    return colab("exec", "-s", S, stdin=code, timeout=timeout).stdout


def score():
    res = {}
    for n in ("ema", "ctx32k"):
        f = [os.path.join(dp, x) for dp, _, fs in os.walk(os.path.join(OUT, "out", n)) for x in fs if x.startswith("results_")]
        r = json.load(open(sorted(f)[-1]))["results"]
        res[n] = {t: round(100 * r[t][f"{m},none"], 2) for t, m in METRIC.items()}
        res[n]["avg5"] = round(sum(res[n][t] for t in METRIC) / 5, 2)
    res["delta"] = {k: round(res["ctx32k"][k] - res["ema"][k], 2) for k in res["ema"]}
    res["pass"] = res["delta"]["avg5"] >= -1.0
    json.dump(res, open(os.path.join(OUT, "gate.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
if cmd == "start":
    assert "CTX_DONE" in ex("print(open('/content/ctx32k.log').read()[-400:])"), "pipeline not finished (CTX_DONE missing)"
    ex("import os; [os.makedirs(d, exist_ok=True) for d in ('/root/.cache/huggingface', '/content/gate/tasks')]")
    ups = [(os.path.expanduser("~/.cache/huggingface/token"), "/root/.cache/huggingface/token"),
           (os.path.join(ROOT, "evals", "harness", "pyproject.toml"), "/content/gate/pyproject.toml"),
           (os.path.join(ROOT, "evals", "harness", "uv.lock"), "/content/gate/uv.lock"),
           (os.path.join(HERE, "gate_vm.sh"), "/content/gate/gate_vm.sh")]
    ups += [(os.path.join(ROOT, "udgam3", "runs", "g1", "tasks", t), f"/content/gate/tasks/{t}")
            for t in ("arc_easy_val.yaml", "arc_challenge_val.yaml")]
    for local, remote in ups:
        r = colab("upload", "-s", S, local, remote)
        print(("token" if "token" in remote else remote), "rc", r.returncode)
    print(ex("import os, subprocess\nos.chmod('/root/.cache/huggingface/token', 0o600)\n"
             "subprocess.Popen(['bash', '/content/gate/gate_vm.sh'], start_new_session=True)\nprint('started')"))
elif cmd == "poll":
    while True:
        t = ex("import os\nprint(open('/content/gate.log').read()[-1500:] if os.path.exists('/content/gate.log') else 'no log')", 180)
        if "GATE_DONE" in t or "GATE_FAILED" in t:
            print(t[-1200:]); break
        time.sleep(60)
    if "GATE_DONE" in t:
        os.makedirs(OUT, exist_ok=True)
        tgz = os.path.join(OUT, "gate_out.tgz")
        colab("download", "-s", S, "/content/gate_out.tgz", tgz, timeout=1800)
        with tarfile.open(tgz) as tf:
            tf.extractall(OUT, filter="data")
        os.remove(tgz)
        score()
    print(ex("import os; os.system('rm -f /root/.cache/huggingface/token'); "
             "print('token removed:', not os.path.exists('/root/.cache/huggingface/token'))"))
