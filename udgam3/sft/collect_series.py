"""Collect the Laghu SFT series (1, 2, 3 passes) from private HF YOUR_HF_NAMESPACE/laghu-sft and apply the pick rule written
in CHECKLIST_sft.md: lowest final held-out loss `all`; within 0.01 of the best -> fewer passes. The smoke answers are
printed for reading (the chosen run must say it is Laghu by Udgam Labs, keep the turn format, and switch thinking).

  uv run --project evals/harness python udgam3/sft/collect_series.py [--eps 1,2,3]
"""
import argparse, json, os
from huggingface_hub import HfApi, hf_hub_download

ap = argparse.ArgumentParser(); ap.add_argument("--eps", default="1,2,3"); a = ap.parse_args()
R = "YOUR_HF_NAMESPACE/laghu-sft"; HERE = os.path.dirname(os.path.abspath(__file__))
api = HfApi(); files = set(api.list_repo_files(R))
rows, smoke = {}, {}
for ep in a.eps.split(","):
    run = f"laghu-sft-v1-ep{ep}"
    if f"{run}/ckpt/val_history.json" not in files:
        print(run, "not finished"); continue
    hist = json.load(open(hf_hub_download(R, f"{run}/ckpt/val_history.json")))
    rows[ep] = {"steps": hist[-1]["step"], "start": hist[0]["all"], **{k: v for k, v in hist[-1].items() if k != "step"}}
    smoke[ep] = json.load(open(hf_hub_download(R, f"{run}/logs/chat_smoke.json")))
best = min(r["all"] for r in rows.values())
pick = min((int(ep) for ep, r in rows.items() if r["all"] <= best + 0.01), default=None)
keys = sorted({k for r in rows.values() for k in r} - {"steps", "start", "all"})
lines = ["# Laghu SFT series: results (held-out loss per conversation source, nats per trained token; lower is better)", "",
         "| Passes | Steps | all (start → end) | " + " | ".join(keys) + " |", "|---" * (3 + len(keys)) + "|"]
for ep, r in sorted(rows.items()):
    lines.append(f"| {ep} | {r['steps']} | {r['start']:.4f} → **{r['all']:.4f}** | " + " | ".join(f"{r.get(k, float('nan')):.4f}" for k in keys) + " |")
lines += ["", f"**Pick rule** (CHECKLIST_sft.md): lowest `all`, fewer passes if within 0.01 → **{pick} pass(es)** "
          f"(best `all` {best:.4f}); the smoke test below must also pass.", ""]
for ep, s in sorted(smoke.items()):
    lines.append(f"## Smoke test, {ep} pass(es) (greedy)")
    for k, v in s.items():
        lines.append(f"- **{k}{' (thinking on)' if v['thinking'] else ''}:** `{v['reply'][:300].replace(chr(10), ' / ').replace('`', chr(39))}`")
    lines.append("")
out = os.path.join(HERE, "SERIES_RESULTS.md"); open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:8 + len(rows)])); print("wrote", out)
