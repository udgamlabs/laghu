"""Start / watch / stop the Laghu SFT pipeline on a Colab G4 session (all Colab calls through the shared lock).

  python3 udgam3/sft/colab_sft_run.py start EPOCHS [--dry] [--session S] [--new]   # upload token + scripts, start in background
  python3 udgam3/sft/colab_sft_run.py tail [N] [--session S]
  python3 udgam3/sft/colab_sft_run.py stop [--session S]                           # remove the token, stop the session

--new creates the session first (otherwise an existing one is reused, e.g. the 32K stretch's "ctx32k" after its gate).
CODE_REV is pinned here: the udgam-build-code revision = the main run's code + the SFT changes (re-tie, --new-token-ids).
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam3", "evals"))
from colab_lock import colab  # noqa: E402

CODE_REV = "571e4330a13ec1689af3da167b87abd869b29873"
argv = sys.argv[1:]
S = argv[argv.index("--session") + 1] if "--session" in argv else "ctx32k"
cmd = argv[0] if argv else "tail"


def ex(code, timeout=300):
    for _ in range(3):   # the CLI's exec sometimes hangs (2026-10-19): retry instead of crashing a running series
        try:
            return colab("exec", "-s", S, stdin=code, timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            print(f"  (colab exec hung for {timeout}s; retrying)", flush=True)
    return ""


if cmd == "start":
    ep = argv[1]; dry = "1" if "--dry" in argv else "0"
    if "--new" in argv:
        r = colab("new", "-s", S, "--gpu", "G4", timeout=900); print(r.stdout[-300:], r.stderr[-300:])
    busy = ex("import subprocess\nprint(subprocess.run(['pgrep', '-f', 'colab_sft_pipeline|gate_vm|lm_eval|base_train'], "
              "capture_output=True, text=True).stdout.strip() or 'idle')")
    assert busy.strip() == "idle", f"session busy: {busy}"
    ex("import os; os.makedirs('/root/.cache/huggingface', exist_ok=True)")
    tok = os.path.expanduser("~/.cache/huggingface/token")
    for local, remote in ((tok, "/root/.cache/huggingface/token"),
                          (os.path.join(HERE, "colab_sft_pipeline.sh"), "/content/colab_sft_pipeline.sh"),
                          (os.path.join(HERE, "chat_smoke.py"), "/content/chat_smoke.py")):
        r = colab("upload", "-s", S, local, remote); print("token" if "token" in remote else remote, "rc", r.returncode)
    print(ex("import os, subprocess\nos.chmod('/root/.cache/huggingface/token', 0o600)\n"
             "for f in ('/content/sft.log', '/content/sft_train.log'):\n    os.path.exists(f) and os.remove(f)\n"
             f"subprocess.Popen('CODE_REV={CODE_REV} nohup bash /content/colab_sft_pipeline.sh {ep} {dry} > /content/sft_pipeline.out 2>&1 &', shell=True)\n"
             "print('started')"))
elif cmd == "tail":
    n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 20
    print(ex(f"import subprocess\nprint(subprocess.run('tail -n {n} /content/sft.log; grep -E \"^step\" /content/sft_train.log 2>/dev/null | tail -n 3; "
             "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader', shell=True, capture_output=True, text=True).stdout)"))
elif cmd == "series":   # series 1,2,3 [--session sft] [--new]: the runs one at a time, a dashboard per run, stop at the end
    import json, re, time
    eps = argv[1].split(","); here_py = os.path.abspath(__file__)
    env = dict(os.environ, COLAB_SESSION=S)
    ssh = os.path.join(ROOT, "udgam3", "runs", "ctx32k", "colab_ssh.sh")
    base_cfg = json.load(open(os.path.join(ROOT, "ops", "train-dash", "push", "configs", "laghu-sft-v1.json")))
    try:
        for k, ep in enumerate(eps):
            extra = ["--session", S] + (["--new"] if "--new" in argv and k == 0 else []) + (["--dry"] if "--dry" in argv else [])
            r = subprocess.run([sys.executable, here_py, "start", ep] + extra, capture_output=True, text=True)
            print(r.stdout[-300:], r.stderr[-300:], flush=True)
            if "started" not in r.stdout:
                print(f"ep{ep}: start failed; stopping the series", flush=True); break
            run = f"laghu-sft-v1-ep{ep}" + ("-dry" if "--dry" in argv else ""); steps = None; pusher = None; t0 = time.time()
            while True:
                time.sleep(60)
                t = ex("import subprocess\nprint(subprocess.run('tail -n 5 /content/sft.log; grep -m1 ^packed /content/sft_train.log', "
                       "shell=True, capture_output=True, text=True).stdout)", 180)
                m = re.search(r"; (\d+) steps x", t)
                if m and pusher is None:
                    steps = int(m.group(1))
                    cfg = json.loads(json.dumps(base_cfg)); cfg["run"] = run; cfg["title"] = f"Laghu fine-tuning (SFT), {ep} pass{'es' if ep != '1' else ''}"
                    cfg["config"].update(steps_total=steps, tokens_target=steps * 262144, warmup_steps=max(1, int(0.05 * steps)),
                                         started_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                    cp = os.path.join(ROOT, "ops", "train-dash", "push", "configs", f"{run}.json"); json.dump(cfg, open(cp, "w"), indent=1, ensure_ascii=False)
                    pusher = subprocess.Popen([sys.executable, os.path.join(ROOT, "ops", "train-dash", "push", "dash_push.py"), "--config", cp,
                                               "--ssh", ssh, "--state", run, "--remote-log", "/content/sft_train.log", "--every", "60"],
                                              env=env, stdout=open(f"/tmp/dash_{run}.log", "a"), stderr=subprocess.STDOUT)
                    print(f"{run}: {steps} steps; dashboard pusher started", flush=True)
                if "SFT_DONE" in t or "SFT_FAILED" in t:
                    break
                if time.time() - t0 > 3 * 3600:
                    print(f"{run}: no end after 3 h; stopping the series", flush=True); t = "SFT_FAILED: series timeout"; break
            ok = "SFT_DONE" in t
            line = (f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] " + ("ALL DONE: export verify PASS, chat smoke run, uploaded" if ok else "FAILED: " + t.strip()[-120:]))
            ex(f"open('/content/sft_train.log', 'a').write({line!r} + '\\n')")
            if pusher:
                time.sleep(90); pusher.terminate()
                sf = os.path.join(ROOT, "ops", "train-dash", "push", "state", f"{run}.json")
                for _ in range(3):   # a stalled read leaves the run "training" on the dashboard (dry series ep1): retry the final push
                    subprocess.run([sys.executable, os.path.join(ROOT, "ops", "train-dash", "push", "dash_push.py"), "--config", cp, "--ssh", ssh,
                                    "--state", run, "--remote-log", "/content/sft_train.log", "--once"], env=env, capture_output=True)
                    if json.load(open(sf)).get("done") or not ok:
                        break
            print(f"{run}: {'SFT_DONE' if ok else 'FAILED'}\n{t.strip()[-400:]}", flush=True)
            if not ok:
                break
    finally:   # never leave a session running (and burning units) after the series, even on a crash
        print(ex("import os; os.system('rm -f /root/.cache/huggingface/token'); print('token removed:', not os.path.exists('/root/.cache/huggingface/token'))"))
        print(colab("stop", "-s", S).stdout[-200:], flush=True)
elif cmd == "stop":
    print(ex("import os; os.system('rm -f /root/.cache/huggingface/token'); print('token removed:', not os.path.exists('/root/.cache/huggingface/token'))"))
    print(colab("stop", "-s", S).stdout[-200:])
