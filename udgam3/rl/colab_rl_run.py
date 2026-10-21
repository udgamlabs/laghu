"""Run one Laghu RL pipeline stage on a new Colab G4 session "rl", with a live dashboard; stop the session at the end.

  python3 udgam3/rl/colab_rl_run.py STAGE        # dry | r1 | r2 | r3 (each real stage needs its own checklist and go)

Uploads the HF token, the pinned harness lock, the RL code and the frozen sets; starts colab_rl_pipeline.sh in the
background; polls; runs the dashboard pusher on the training logs; removes the token and stops the session in a finally.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam3", "evals"))
from colab_lock import colab  # noqa: E402

S = "rl"; STAGE = sys.argv[1]; LIMIT_MIN = {"dry": 60, "r1": 60, "r2": 240, "r3": 110}[STAGE]   # hard stops: 8.9 + 35.6 + 16.3 units max


def ex(code, timeout=300):
    for _ in range(3):
        try:
            return colab("exec", "-s", S, stdin=code, timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            print("  (colab exec hung; retrying)", flush=True)
    return ""


r = colab("new", "-s", S, "--gpu", "G4", timeout=900); print(r.stdout[-200:], r.stderr[-200:], flush=True)
pusher = None
try:
    ex("import os; [os.makedirs(d, exist_ok=True) for d in ('/root/.cache/huggingface', '/content/rl')]")
    ups = [(os.path.expanduser("~/.cache/huggingface/token"), "/root/.cache/huggingface/token"),
           (os.path.join(ROOT, "evals", "harness", "pyproject.toml"), "/content/rl/pyproject.toml"),
           (os.path.join(ROOT, "evals", "harness", "uv.lock"), "/content/rl/uv.lock"),
           (os.path.join(ROOT, "udgam3", "evals", "chat_suite", "common.py"), "/content/udgam3/evals/chat_suite/common.py"),
           (os.path.join(ROOT, "release", "lilavati", "code", "lilavati", "safe_exec.py"), "/content/release/lilavati/code/lilavati/safe_exec.py"),
           (os.path.join(ROOT, "udgam3", "sft", "devpick", "dev_chat.jsonl"), "/content/udgam3/sft/devpick/dev_chat.jsonl")]
    ups += [(os.path.join(HERE, f), f"/content/rl/{f}") for f in
            ("colab_rl_pipeline.sh", "laghu_rl.py", "rollout.py", "sandbox.py", "rl_eval.py",
             "pool_maths.jsonl", "dev_gsm8k.jsonl", "dev_math.jsonl", "SETS.sha256")]
    ex("import os; [os.makedirs(d, exist_ok=True) for d in ('/content/udgam3/evals/chat_suite', '/content/release/lilavati/code/lilavati', "
       "'/content/udgam3/sft/devpick')]")
    for local, remote in ups:
        rr = colab("upload", "-s", S, local, remote); print("token" if "token" in remote else remote, "rc", rr.returncode, flush=True)
    # the code expects ROOT = two levels above udgam3/rl: link /content/udgam3/rl -> /content/rl
    print(ex("import os, subprocess\nos.chmod('/root/.cache/huggingface/token', 0o600)\n"
             "os.path.exists('/content/udgam3/rl') or os.symlink('/content/rl', '/content/udgam3/rl')\n"
             f"subprocess.Popen('KEEP={os.environ.get('KEEP', 'r2')} nohup bash /content/udgam3/rl/colab_rl_pipeline.sh {STAGE} > /content/rl_pipeline.out 2>&1 &', shell=True)\n"
             "print('started')"), flush=True)
    cfg = os.path.join(ROOT, "ops", "train-dash", "push", "configs", f"laghu-rl-{STAGE}.json")
    if os.path.exists(cfg):
        env = dict(os.environ, COLAB_SESSION=S)
        pusher = subprocess.Popen([sys.executable, os.path.join(ROOT, "ops", "train-dash", "push", "dash_push.py"), "--config", cfg,
                                   "--ssh", os.path.join(ROOT, "udgam3", "runs", "ctx32k", "colab_ssh.sh"), "--state", f"laghu-rl-{STAGE}",
                                   "--remote-log", "/content/rl_pipeline.log", "--every", "60"],
                                  env=env, stdout=open(f"/tmp/dash_laghu-rl-{STAGE}.log", "a"), stderr=subprocess.STDOUT)
    t0 = time.time(); t = ""
    while time.time() - t0 < LIMIT_MIN * 60:
        time.sleep(60)
        t = ex("import os\nprint(open('/content/rl_pipeline.log').read()[-2500:] if os.path.exists('/content/rl_pipeline.log') else 'no log')", 180)
        last = [l for l in t.strip().splitlines() if l.startswith(("RLP", "RL ", "RL_"))][-1:] or [""]
        print(time.strftime("%H:%M:%S"), last[0][:220], flush=True)
        if "RL_DONE" in t or "RL_FAILED" in t:
            print(t[-2500:], flush=True); break
    else:
        print(f"no end after {LIMIT_MIN} min: stopping", flush=True)
finally:
    if pusher:
        time.sleep(70); pusher.terminate()
    print(ex("import os; os.system('rm -f /root/.cache/huggingface/token'); print('token removed:', not os.path.exists('/root/.cache/huggingface/token'))"))
    print(colab("stop", "-s", S).stdout[-200:], flush=True)
