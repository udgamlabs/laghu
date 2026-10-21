"""Summarise Phase 0 base-suite results (lm-eval JSON per model) into a Markdown table.
  uv run --project evals/harness python udgam3/evals/phase0/summarize_base.py > udgam3/evals/phase0/BASE_RESULTS.md
"""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = [("arc_easy", "acc,none", "ARC-E"), ("arc_challenge", "acc_norm,none", "ARC-C (n)"), ("hellaswag", "acc_norm,none", "HS (n)"),
        ("piqa", "acc_norm,none", "PIQA (n)"), ("winogrande", "acc,none", "WG"), ("mmlu", "acc,none", "MMLU-MCF"),
        ("mmlu_continuation", "acc_norm,none", "MMLU-CF (n)")]
man = json.load(open(os.path.join(HERE, "manifest.json")))
rows = []
for d in sorted(glob.glob(os.path.join(HERE, "base", "*"))):
    fs = sorted(glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True))
    if not fs:
        continue
    r = json.load(open(fs[-1]))["results"]
    vals = [r.get(t, {}).get(m) for t, m, _ in COLS]
    core = [v for v in vals[:5] if v is not None]
    rows.append((os.path.basename(d), vals, sum(core) / len(core) if len(core) == 5 else None))
print("# Phase 0 base suite (lm-eval 0.4.13, zero-shot, Mac mini MPS bf16, 2026-10-17) [MEASURED]\n")
print("Udgam models with add_bos_token=True; others with their own tokenizer defaults. (n) = acc_norm. Avg = mean of the "
      "first five columns. Pinned revisions: manifest.json. Per-item logs: base/<model>/.\n")
print("| Model | " + " | ".join(c[2] for c in COLS) + " | Avg (5) |")
print("|---|" + "---|" * (len(COLS) + 1))
for name, vals, avg in sorted(rows, key=lambda x: -(x[2] or 0)):
    print(f"| {name} | " + " | ".join("—" if v is None else f"{100*v:.1f}" for v in vals) + f" | {'—' if avg is None else f'{100*avg:.1f}'} |")
