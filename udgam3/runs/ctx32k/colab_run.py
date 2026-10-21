"""Start / watch / stop the Udgam-3 32K stretch on its own Colab G4 session (never the eval session).

  python3 udgam3/runs/ctx32k/colab_run.py start STEPS [--db 2]   # new session "ctx32k", upload token + scripts, start in background
  python3 udgam3/runs/ctx32k/colab_run.py tail [N]               # last N lines of the pipeline log (one short exec)
  python3 udgam3/runs/ctx32k/colab_run.py stop                   # remove the token from the VM, stop the session

Every exec is short; the pipeline runs as a nohup background process, so only ONE process (the dashboard pusher via
colab_ssh.sh, or this script) should talk to the session at a time (concurrent execs hung a call on 2026-10-19).
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
COLAB = [os.path.expanduser("~/.local/bin/colab"), "--auth=adc"]
S = "ctx32k"


sys.path.insert(0, os.path.join(ROOT, "udgam3", "evals"))
from colab_lock import colab as locked_colab  # noqa: E402  (one Colab CLI call at a time, all sessions)


def colab(*args, stdin=None, timeout=600):
    return locked_colab(*args, stdin=stdin, timeout=timeout)


def ex(code, timeout=600):
    return colab("exec", "-s", S, stdin=code, timeout=timeout).stdout


cmd = sys.argv[1] if len(sys.argv) > 1 else "tail"
if cmd == "start":
    steps = int(sys.argv[2]); db = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "2"
    r = colab("new", "-s", S, "--gpu", "G4", timeout=900); print(r.stdout[-300:], r.stderr[-300:])
    tok = os.path.expanduser("~/.cache/huggingface/token")
    ex("import os; os.makedirs('/root/.cache/huggingface', exist_ok=True)")
    for local, remote in ((tok, "/root/.cache/huggingface/token"),
                          (os.path.join(HERE, "colab_ctx32k_pipeline.sh"), "/content/colab_ctx32k_pipeline.sh"),
                          (os.path.join(ROOT, "udgam2", "ctx32k", "long_parity.py"), "/content/long_parity.py")):
        r = colab("upload", "-s", S, local, remote); print(r.stdout.strip()[-160:].replace(tok, "<token>"))
    print(ex("import os, subprocess\nos.chmod('/root/.cache/huggingface/token', 0o600)\n"
             f"subprocess.Popen('nohup bash /content/colab_ctx32k_pipeline.sh {steps} {db} > /content/pipeline.out 2>&1 &', shell=True)\n"
             "print('started')"))
elif cmd == "tail":   # through colab_ssh.sh, so it shares the lock with the dashboard pusher
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(subprocess.run(["bash", os.path.join(HERE, "colab_ssh.sh"), "x",
                          f"tail -n {n} /content/ctx32k.log; grep -E '^step|bpb' /content/ctx32k_train.log 2>/dev/null | tail -n 3; "
                          "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"],
                         capture_output=True, text=True, timeout=900).stdout)
elif cmd == "stop":
    print(ex("import os, subprocess; subprocess.run('rm -f /root/.cache/huggingface/token', shell=True); print('token removed')"))
    print(colab("stop", "-s", S).stdout[-200:])
