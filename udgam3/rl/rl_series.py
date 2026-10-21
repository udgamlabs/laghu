"""Run the approved Laghu RL stages one at a time and apply the gates of RL_PLAN.md after each (written before any run).
Stops at the first failed gate (no automatic retries, no changed settings) and writes GATES.md.

  python3 udgam3/rl/rl_series.py r1[,r2,r3]          # exactly the stages the user approved
  python3 udgam3/rl/rl_series.py --check r1          # only re-check a finished stage's gate
Baseline = the SFT model's dev evals from the dry run (YOUR_HF_NAMESPACE/laghu-rl dry/eval_sft, same env and settings).
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = "YOUR_HF_NAMESPACE/laghu-rl"


def summary(prefix):
    out = subprocess.run(["uv", "run", "--project", os.path.join(ROOT, "evals", "harness"), "python", "-c",
                          f"from huggingface_hub import hf_hub_download; print(open(hf_hub_download('{R}', '{prefix}/summary.json')).read())"],
                         capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.stdout[out.stdout.index("{"):])


def gate(stage):
    sft = summary("dry/eval_sft"); cur = summary(f"{stage}/eval")
    acc = lambda s, k: s[k]["acc"]
    if stage == "r1":
        checks = [("dev GSM8K thinking on >= SFT + 3.0", acc(cur, "gsm8k_think"), acc(sft, "gsm8k_think") + 3.0),
                  ("dev GSM8K cap rate <= SFT's", -cur["gsm8k_think"]["cap_rate"], -sft["gsm8k_think"]["cap_rate"])]
    elif stage == "r2":
        r1 = summary("r1/eval")
        checks = [("dev GSM8K thinking on >= R1", acc(cur, "gsm8k_think"), acc(r1, "gsm8k_think")),
                  ("dev MATH thinking on >= SFT - 2", acc(cur, "math_think"), acc(sft, "math_think") - 2.0)]
    else:
        start = summary(f"{os.environ.get('KEEP', 'r2')}/eval")
        checks = [("dev GSM8K tool mode >= start + 5", acc(cur, "gsm8k_tool"), acc(start, "gsm8k_tool") + 5.0),
                  ("dev GSM8K thinking on >= start - 1", acc(cur, "gsm8k_think"), acc(start, "gsm8k_think") - 1.0)]
    ok = all(v >= bar for _, v, bar in checks)
    lines = [f"## {stage}: {'PASS' if ok else 'FAIL'} ({time.strftime('%Y-%m-%d %H:%M')})"]
    lines += [f"- {name}: {abs(v):.2f} vs bar {abs(bar):.2f} -> {'ok' if v >= bar else 'MISS'}" for name, v, bar in checks]
    lines += [f"- all dev numbers: " + json.dumps({k: v for k, v in cur.items() if isinstance(v, dict)})]
    with open(os.path.join(HERE, "GATES.md"), "a") as f:
        f.write("\n".join(lines) + "\n\n")
    print("\n".join(lines), flush=True)
    return ok


if sys.argv[1] == "--check":
    sys.exit(0 if gate(sys.argv[2]) else 1)
for stage in sys.argv[1].split(","):
    print(f"=== {stage} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "colab_rl_run.py"), stage], cwd=ROOT)
    if r.returncode != 0:
        print(f"{stage}: driver exited {r.returncode}; stopping", flush=True); sys.exit(1)
    try:
        ok = gate(stage)
    except Exception as e:   # no eval summary = the stage failed before its evals
        print(f"{stage}: no eval summary ({str(e)[:120]}); stopping", flush=True); sys.exit(1)
    if not ok:
        print(f"{stage}: gate failed; stopping and bringing options to the user", flush=True); sys.exit(1)
print("SERIES_DONE", flush=True)
