"""Phase 0: base-suite measurements of every size-class rival (and our own models) in one harness.

lm-evaluation-harness 0.4.13 from evals/harness (its own uv env). Zero-shot ARC-E, ARC-C, HellaSwag, PIQA,
WinoGrande, MMLU (letter format) and MMLU cloze (mmlu_continuation), with per-item logs (--log_samples) so later
comparisons can use paired tests (PREREGISTRATION.md section 4).

- Models that need trust_remote_code are NOT run here (they would execute the repo's own Python on this Mac); they run
  on a disposable pod after their code has been read and pinned.
- Udgam models get add_bos_token=True (scores were badly understated without it).
- Every model is pinned to the commit sha resolved at first run (manifest.json), so reruns use the same weights.
- Resumable: a model whose results file exists is skipped. One model at a time.

  cd "~/Work/Model Training" && uv run --project evals/harness python udgam3/evals/phase0/run_base.py [--smoke] [--only name,...]
"""
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(ROOT, "evals", "harness")
TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,mmlu,mmlu_continuation"

# (short name, repo, udgam model?) in priority order: size-class rivals first, then our own bars, then references.
MODELS = [
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", False),
    ("falcon-h1-tiny-90m-base", "tiiuae/Falcon-H1-Tiny-90M-Base", False),
    ("mobilellm-r1-140m-base", "facebook/MobileLLM-R1-140M-base", False),
    ("monad-56m", "PleIAs/Monad", False),
    ("gargi-base", "udgamlabs/Gargi-Base", True),
    ("gargi", "udgamlabs/Gargi", True),
    ("udgam-384m-base-4k", "UdgamLabs/Udgam-384M-Base-4K", True),
    ("udgam-384m-chat", "UdgamLabs/Udgam-384M-Chat", True),
    ("supra2-100m-instruct", "SupraLabs/Supra2-100M-Instruct", False),
    ("smollm2-135m-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", False),
    ("falcon-h1-tiny-90m-instruct", "tiiuae/Falcon-H1-Tiny-90M-Instruct", False),
    ("gemma-3-270m", "google/gemma-3-270m", False),
    ("lfm2.5-230m", "LiquidAI/LFM2.5-230M", False),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", False),
    ("pythia-160m", "EleutherAI/pythia-160m", False),
]


def pin(manifest, repo):
    if repo not in manifest:
        from huggingface_hub import HfApi
        manifest[repo] = HfApi().model_info(repo).sha
    return manifest[repo]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="arc_easy with --limit 20 only, to check every model loads")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout-min", type=int, default=240)
    a = ap.parse_args()
    out_root = os.path.join(HERE, "smoke" if a.smoke else "base")
    os.makedirs(out_root, exist_ok=True)
    mpath = os.path.join(HERE, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    only = set(x for x in a.only.split(",") if x)
    status = {}
    for name, repo, udgam in MODELS:
        if only and name not in only:
            continue
        out = os.path.join(out_root, name)
        if glob.glob(os.path.join(out, "**", "results_*.json"), recursive=True):
            status[name] = "done (skipped)"; continue
        sha = pin(manifest, repo)
        json.dump(manifest, open(mpath, "w"), indent=1, sort_keys=True)
        margs = f"pretrained={repo},revision={sha},dtype=bfloat16" + (",add_bos_token=True" if udgam else "")
        cmd = ["uv", "run", "--project", HARNESS, "python", "-W", "ignore", "-m", "lm_eval",
               "--model", "hf", "--model_args", margs, "--device", "mps",
               "--tasks", "arc_easy" if a.smoke else TASKS, "--batch_size", "32",   # NOT auto: on MPS unified memory it grew to 23.5 GB (2026-10-17)
               "--output_path", out, "--seed", "1234"]
        cmd += ["--limit", "20"] if a.smoke else ["--log_samples"]
        log = os.path.join(out_root, f"{name}.log")
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} ({repo}@{sha[:8]}) ...", flush=True)
        with open(log, "w") as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout_min * 60,
                                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        status[name] = f"rc={rc} in {(time.time() - t0) / 60:.1f} min"
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {status[name]}", flush=True)
    json.dump(status, open(os.path.join(out_root, "status.json"), "w"), indent=1)
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    sys.exit(main())
