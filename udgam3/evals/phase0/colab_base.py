"""Phase 0 base suite on a Colab G4 (CUDA) session, one model per call (PREREGISTRATION.md v1.6: every row of a table
on the same device class). Runs from the Mac; the Colab session must exist (`colab --auth=adc new -s evals --gpu G4`)
with the pinned harness installed at /content/harness (uv sync --frozen of evals/harness).

  python3 udgam3/evals/phase0/colab_base.py --smoke                 # arc_easy --limit 20 on the first model
  python3 udgam3/evals/phase0/colab_base.py [--only a,b] [--tasks base|gsm8k|all]

Per model: lm-eval 0.4.13, bf16, batch 64, seed 1234, --log_samples; zero-shot ARC-E, ARC-C, HellaSwag, PIQA,
WinoGrande, MMLU (letter) and MMLU cloze; GSM8K 5-shot as a separate run. Udgam models get add_bos_token=True.
Results land in udgam3/evals/phase0/cuda/<model>/ (resumable: models with results are skipped).
"""
import argparse, json, os, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_base import MODELS, pin  # same model list and pinned revisions as the Phase 0 plan
sys.path.insert(0, os.path.dirname(HERE))
from colab_lock import colab as locked_colab  # one Colab CLI call at a time, all sessions (2026-10-19 hangs)

REMOTE_PINS = json.load(open(os.path.join(HERE, 'remote_code_pins.json'))) if os.path.exists(os.path.join(HERE, 'remote_code_pins.json')) else {}
BASE_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,mmlu,mmlu_continuation"
COLAB = [os.path.expanduser("~/.local/bin/colab"), "--auth=adc"]
EXTRA = [  # added rivals (pre-registration section 3 list and the 2026-10-18 small-scratch scan): (name, repo, udgam, trust)
    ("nltower-100m-base", "sfanm/nltower-100m-base", False, False),
]
OURS_NEW = [  # Udgam-3: run only after part 2a is written (PREREGISTRATION.md, 2026-10-19)
    ("udgam3-base-ema", "YOUR_HF_NAMESPACE/laghu-base::udgam3-base-v1/ema", True, False),
]
REMOTE = [  # need the repo's own modeling code: read in REMOTE_CODE_REVIEW.md; pinned in remote_code_pins.json
    ("surjo-100m", "SurjoLabs/Surjo-100m", False, True),
    ("gpt-x3-150m", "AxiomicLabs/GPT-X3-150M", False, True),
    ("cagliostro-v3.5", "bench-labs/cagliostro-v3.5", False, True),
    ("boris-2-preview-1", "opencerebral/Boris-2-Preview-1", False, True),
    ("jugnulm-110m-r2plus", "altslate/JugnuLM-110M-R2plus", False, True),
    ("lilm1-230m-base", "glouriousgautam/LiLM1-230M-base", False, True),
    ("bananamind-2-pro", "SLM-Archive/BananaMind-2-Pro", False, True),
    ("100m-exp", "User01110/100M-exp", False, True),
    ("tinctura-v1", "bench-labs/tinctura-v1", False, True),
    ("q-164m", "q-project/Q-164M", False, True),
]


def colab(*args, stdin=None, timeout=600):
    """A hung Colab call (the CLI's websocket sometimes never answers, 2026-10-19) returns rc=-9 instead of raising."""
    try:
        return locked_colab(*args, stdin=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  (colab {args[0]} hung for {timeout}s; retrying later)", flush=True)
        return subprocess.CompletedProcess(args, -9, "", "")


def remote_job(name, repo, sha, udgam, tasks, fewshot, limit, trust=False):
    pre = ""
    if "::" in repo:   # private export in a repo subfolder (e.g. Udgam-3): fetch it on the VM, evaluate the local copy
        rid, sub = repo.split("::")
        local = f"/content/models/{name}"
        pre = (f"cd /content/harness && .venv/bin/python -c \"from huggingface_hub import snapshot_download; "
               f"snapshot_download('{rid}', revision='{sha}', allow_patterns=['{sub}/*'], local_dir='/content/models/_{name}')\" && "
               f"rm -rf {local} && cp -r /content/models/_{name}/{sub} {local} && ")
        margs = f"pretrained={local},dtype=bfloat16" + (",add_bos_token=True" if udgam else "")
    else:
        margs = (f"pretrained={repo},revision={sha},dtype=bfloat16" + (",add_bos_token=True" if udgam else "")
                 + (",trust_remote_code=True" if trust else ""))
    out = f"/content/out/{name}/{'gsm8k' if fewshot else 'base'}"
    cmd = (pre + f"cd /content/harness && HF_HUB_ENABLE_HF_TRANSFER=0 .venv/bin/python -W ignore -m lm_eval --model hf "
           f"--model_args {margs} --device cuda --tasks {tasks} --batch_size 64 --seed 1234 --output_path {out} "
           + (f"--num_fewshot {fewshot} " if fewshot else "") + (f"--limit {limit} " if limit else "--log_samples ")
           + f"> {out}.log 2>&1; echo rc=$? >> {out}.log; cd /content/out && tar czf /content/{name}_{'gsm8k' if fewshot else 'base'}.tgz {name}"
           + f"; touch {out}.done")
    # started in the background: the exec returns at once and the job is polled with short execs
    # start-or-attach: if the job already finished or is still running on the VM (a restart after a hung call), don't
    # launch a duplicate
    return ("import subprocess, os\n"
            f"os.makedirs('{os.path.dirname(out)}', exist_ok=True)\n"
            f"running = subprocess.run(['pgrep', '-f', '--', '--output_path {out}'], capture_output=True).returncode == 0\n"
            f"if os.path.exists('{out}.done') or running:\n"
            "    print('started (attached)')\n"
            "else:\n"
            f"    subprocess.Popen(['bash', '-c', {cmd!r}], start_new_session=True)\n"
            "    print('started')\n")


def poll_job(session, name, kind, limit_min=90):
    out = f"/content/out/{name}/{kind}"
    t0 = time.time()
    while time.time() - t0 < limit_min * 60:
        time.sleep(30)
        r = colab("exec", "-s", session, stdin=(f"import os\nd = os.path.exists('{out}.done')\n"
                                                 f"print('DONE' if d else 'RUNNING', open('{out}.log').read()[-1200:] if d and os.path.exists('{out}.log') else '')\n"),
                  timeout=120)
        if (r.stdout or "").startswith("DONE"):
            return r.stdout
    return "TIMEOUT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--only", default="")
    ap.add_argument("--tasks", default="all", choices=["base", "gsm8k", "all"]); ap.add_argument("--session", default="evals")
    ap.add_argument("--remote", action="store_true", help="also run the reviewed remote-code models")
    ap.add_argument("--ours", action="store_true", help="also run Udgam-3 (only after part 2a is written)")
    a = ap.parse_args()
    root = os.path.join(HERE, "cuda_smoke" if a.smoke else "cuda")
    os.makedirs(root, exist_ok=True)
    mpath = os.path.join(HERE, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    only = set(x for x in a.only.split(",") if x)
    models = [m + (False,) for m in MODELS] + EXTRA + (REMOTE if a.remote else []) + (OURS_NEW if a.ours else [])
    if a.smoke:
        models = models[:1]
    status = {}
    for name, repo, udgam, trust in models:
        if only and name not in only:
            continue
        sha = REMOTE_PINS.get(repo) or pin(manifest, repo)
        json.dump(manifest, open(mpath, "w"), indent=1, sort_keys=True)
        runs = [("base", BASE_TASKS, 0)] if a.tasks in ("base", "all") else []
        runs += [("gsm8k", "gsm8k", 5)] if a.tasks in ("gsm8k", "all") and not a.smoke else []
        for kind, tasks, fewshot in runs:
            dest = os.path.join(root, name, kind)
            if os.path.isdir(dest) and any(f.startswith("results_") for _, _, fs in os.walk(dest) for f in fs):
                status[f"{name}/{kind}"] = "done (skipped)"; continue
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {name} {kind} ({repo}@{sha[:8]}) ...", flush=True)
            r = colab("exec", "-s", a.session, stdin=remote_job(name, repo, sha, udgam, "arc_easy" if a.smoke else tasks,
                                                                fewshot, 20 if a.smoke else 0, trust), timeout=120)
            if r.returncode == -9:   # the start call hung: try once more (start-or-attach makes this safe)
                r = colab("exec", "-s", a.session, stdin=remote_job(name, repo, sha, udgam, "arc_easy" if a.smoke else tasks,
                                                                    fewshot, 20 if a.smoke else 0, trust), timeout=120)
            if "started" not in (r.stdout or ""):
                status[f"{name}/{kind}"] = "FAILED to start"; print(f"{name} {kind}: start failed {r.stdout[-300:]} {r.stderr[-300:]}", flush=True); continue
            tail = poll_job(a.session, name, kind)[-600:]
            tgz = os.path.join(root, f"{name}_{kind}.tgz")
            d = colab("download", "-s", a.session, f"/content/{name}_{kind}.tgz", tgz, timeout=1800)
            if d.returncode == 0 and os.path.exists(tgz):
                with tarfile.open(tgz) as t:
                    t.extractall(root, filter="data")
                os.remove(tgz)
            ok = "rc=0" in tail
            status[f"{name}/{kind}"] = f"{'ok' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min"
            print(f"[{time.strftime('%H:%M:%S')}] {name} {kind}: {status[f'{name}/{kind}']}" + ("" if ok else f"\n{tail}"), flush=True)
    json.dump(status, open(os.path.join(root, "status.json"), "w"), indent=1)
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    sys.exit(main())
