"""Run the dev pick generation on a new Colab G4 session "devpick", then download it and stop the session.

  python3 udgam3/sft/devpick/devpick_run.py      # new session, upload, start, poll, download to devpick/gen/, stop
"""
import os, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam3", "evals"))
from colab_lock import colab  # noqa: E402

S = "devpick"


def ex(code, timeout=300):
    for _ in range(3):
        try:
            return colab("exec", "-s", S, stdin=code, timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            print("  (colab exec hung; retrying)", flush=True)
    return ""


r = colab("new", "-s", S, "--gpu", "G4", timeout=900); print(r.stdout[-200:], r.stderr[-200:], flush=True)
try:
    ex("import os; [os.makedirs(d, exist_ok=True) for d in ('/root/.cache/huggingface', '/content/devpick')]")
    ups = [(os.path.expanduser("~/.cache/huggingface/token"), "/root/.cache/huggingface/token"),
           (os.path.join(ROOT, "evals", "harness", "pyproject.toml"), "/content/devpick/pyproject.toml"),
           (os.path.join(ROOT, "evals", "harness", "uv.lock"), "/content/devpick/uv.lock")]
    ups += [(os.path.join(HERE, f), f"/content/devpick/{f}") for f in
            ("devpick_vm.sh", "devpick_gen.py", "dev_chat.jsonl", "dev_gsm8k.jsonl", "DEV_SETS.sha256")]
    for local, remote in ups:
        rr = colab("upload", "-s", S, local, remote); print("token" if "token" in remote else remote, "rc", rr.returncode, flush=True)
    print(ex("import os, subprocess\nos.chmod('/root/.cache/huggingface/token', 0o600)\n"
             "subprocess.Popen(['bash', '/content/devpick/devpick_vm.sh'], start_new_session=True)\nprint('started')"), flush=True)
    t0 = time.time()
    while time.time() - t0 < 45 * 60:   # hard stop: 45 min of generation (~6.7 units)
        time.sleep(60)
        t = ex("import os\nprint(open('/content/devpick.log').read()[-1500:] if os.path.exists('/content/devpick.log') else 'no log')", 180)
        print(time.strftime("%H:%M:%S"), (t.strip().splitlines() or [""])[-1][:160], flush=True)
        if "DEVPICK_DONE" in t or "DEVPICK_FAILED" in t:
            print(t[-1200:], flush=True); break
    if "DEVPICK_DONE" in t:
        tgz = os.path.join(HERE, "devpick_gen.tgz")
        colab("download", "-s", S, "/content/devpick_gen.tgz", tgz, timeout=1800)
        with tarfile.open(tgz) as tf:
            tf.extractall(HERE, filter="data")
        os.remove(tgz); print("downloaded to", os.path.join(HERE, "gen"), flush=True)
finally:
    print(ex("import os; os.system('rm -f /root/.cache/huggingface/token'); print('token removed:', not os.path.exists('/root/.cache/huggingface/token'))"))
    print(colab("stop", "-s", S).stdout[-200:], flush=True)
