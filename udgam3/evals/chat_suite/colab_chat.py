"""Mac-side driver for the chat suite on the Colab session `chatsuite` (G4). Every Colab call goes through
udgam3/evals/colab_lock.py (one CLI call at a time across all sessions) and is kept short: long work runs in the
background on the VM and is polled.

  python3 udgam3/evals/chat_suite/colab_chat.py up            # create the session, upload bundle + HF token, start setup
  python3 udgam3/evals/chat_suite/colab_chat.py status        # setup log tail, lane status, GPU memory
  python3 udgam3/evals/chat_suite/colab_chat.py lane NAME JOBS.json
  python3 udgam3/evals/chat_suite/colab_chat.py fetch         # download finished result tarballs into raw/
  python3 udgam3/evals/chat_suite/colab_chat.py down          # remove the token (if still there) and stop the session

The HF token is uploaded to /root/.cache/huggingface/token, used only by setup to download the models (two are
gated), and deleted by setup before any job runs; `down` deletes it again in case setup failed early. It is never printed.
"""
import json, os, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))
from colab_lock import colab as locked_colab  # noqa: E402

S = "chatsuite"
SCRATCH = os.environ.get("CHAT_SCRATCH", "/tmp")


def colab(*args, stdin=None, timeout=300):
    try:
        r = locked_colab(*args, stdin=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"(colab {args[0]} hung for {timeout} s)", flush=True)
        return subprocess.CompletedProcess(args, -9, "", "")
    return r


def vm(code, timeout=180):
    r = colab("exec", "-s", S, "--timeout", "120", stdin=code, timeout=timeout)
    return (r.stdout or "") + (("\nSTDERR: " + r.stderr[-800:]) if r.returncode else "")


def bundle():
    path = os.path.join(SCRATCH, "chatsuite_bundle.tgz")
    with tarfile.open(path, "w:gz") as t:
        for f in ("chat_gen.py", "common.py", "vm/setup_vm.sh", "vm/vm_queue.py", "env/vllm.lock"):
            t.add(os.path.join(HERE, f), arcname=f"chatsuite/{os.path.basename(f) if f.startswith('vm/') else f}")
        # CHAT_MANIFEST=manifest_laghu.json ships a one-model manifest, so setup downloads only that model
        t.add(os.path.join(HERE, os.environ.get("CHAT_MANIFEST", "manifest.json")), arcname="chatsuite/manifest.json")
        for f in sorted(os.listdir(os.path.join(HERE, "items"))):
            t.add(os.path.join(HERE, "items", f), arcname=f"chatsuite/items/{f}")
        for f in ("pyproject.toml", "uv.lock"):
            t.add(os.path.join(ROOT, "evals", "harness", f), arcname=f"chatsuite/harness/{f}")
    return path


def up():
    sessions = colab("sessions", timeout=120).stdout or ""
    if f"[{S}]" not in sessions:
        r = colab("new", "-s", S, "--gpu", "G4", timeout=600)
        print("new:", r.returncode, (r.stdout or "")[-400:], (r.stderr or "")[-400:])
        if r.returncode:
            return
    b = bundle()
    print("upload bundle:", colab("upload", "-s", S, b, "/content/chatsuite_bundle.tgz", timeout=600).returncode)
    print(vm("import os, tarfile\nos.makedirs('/root/.cache/huggingface', exist_ok=True)\n"
             "tarfile.open('/content/chatsuite_bundle.tgz').extractall('/content', filter='data')\nprint(sorted(os.listdir('/content/chatsuite')))\n"))
    tok = os.path.expanduser("~/.cache/huggingface/token")
    r = colab("upload", "-s", S, tok, "/root/.cache/huggingface/token", timeout=300)
    print("upload token: rc", r.returncode)   # contents never printed
    print(vm("import subprocess\nsubprocess.Popen(['bash', '/content/chatsuite/setup_vm.sh'], start_new_session=True)\nprint('setup started')\n"))


def status():
    print(vm("import glob, json, os, subprocess\n"
             "print(open('/content/setup.log').read()[-1500:] if os.path.exists('/content/setup.log') else 'no setup log')\n"
             "for p in sorted(glob.glob('/content/status_*.json')):\n"
             "    s = json.load(open(p)); print(os.path.basename(p), {k: v for k, v in s.items() if k != 'jobs'})\n"
             "    for j in s['jobs']: print('  ', j)\n"
             "print(subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout)\n"
             "print('token present:', os.path.exists('/root/.cache/huggingface/token'))\n"))


def lane(name, jobs_path):
    jobs = json.load(open(jobs_path))
    print(vm("import json, subprocess, os\n"
             f"json.dump({jobs!r}, open('/content/jobs_{name}.json', 'w'))\n"
             f"running = subprocess.run(['pgrep', '-f', 'vm_queue.py {name} '], capture_output=True).returncode == 0\n"
             "if running:\n    print('lane already running')\nelse:\n"
             f"    subprocess.Popen(['python3', '/content/chatsuite/vm_queue.py', '{name}', '/content/jobs_{name}.json'], start_new_session=True,\n"
             f"                     stdout=open('/content/lane_{name}.out', 'w'), stderr=subprocess.STDOUT)\n"
             f"    print('lane {name} started with', len({jobs!r}), 'jobs')\n"))


def fetch():
    raw = os.path.join(HERE, "raw")
    os.makedirs(raw, exist_ok=True)
    listing = vm("import glob, os\nfor p in sorted(glob.glob('/content/ready/*.tgz')): print(p, os.path.getsize(p))\n")
    have = set(json.load(open(os.path.join(raw, ".fetched.json")))) if os.path.exists(os.path.join(raw, ".fetched.json")) else set()
    for line in listing.splitlines():
        if not line.startswith("/content/ready/"):
            continue
        p, size = line.split()
        key = f"{os.path.basename(p)}:{size}"
        if key in have:
            continue
        local = os.path.join(SCRATCH, os.path.basename(p))
        r = colab("download", "-s", S, p, local, timeout=900)
        if r.returncode == 0 and os.path.exists(local):
            with tarfile.open(local) as t:
                t.extractall(raw, filter="data")
            os.remove(local)
            have.add(key)
            print("fetched", os.path.basename(p), size)
        else:
            print("download failed", p, r.returncode)
    json.dump(sorted(have), open(os.path.join(raw, ".fetched.json"), "w"), indent=0)


def down():
    print(vm("import os\nif os.path.exists('/root/.cache/huggingface/token'): os.remove('/root/.cache/huggingface/token')\n"
             "print('token present:', os.path.exists('/root/.cache/huggingface/token'))\n"))
    r = colab("stop", "-s", S, timeout=300)
    print("stop:", r.returncode, (r.stdout or "")[-300:], (r.stderr or "")[-300:])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "up":
        up()
    elif cmd == "status":
        status()
    elif cmd == "lane":
        lane(sys.argv[2], sys.argv[3])
    elif cmd == "fetch":
        fetch()
    elif cmd == "down":
        down()
    elif cmd == "exec":
        print(vm(sys.stdin.read()))
    else:
        raise SystemExit(__doc__)
