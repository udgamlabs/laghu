"""Project the full-run DeepSeek cost from a pilot's spend and keep rates.

  python3 udgam3/sft/gen/project.py udgam3/sft/pilot_v1
Cost per kept row = everything the slice spent in the pilot (failed attempts, retries and judge calls included) / kept
rows. The real-prompt screen is costed per usable prompt. Targets follow SFT_DATA_PLAN.md (identity is templated: $0).
"""
import json
import sys
from pathlib import Path

TARGETS = {"chat": 14000, "instruction_following": 4000, "system_prompts": 1500, "maths_logic": 6000,
           "python_tool": 3000, "json_function_calls": 1500, "long_documents": 1500, "safety": 1200, "identity": 400}
REAL_PROMPT_SLICES = ("chat", "instruction_following", "system_prompts")

d = Path(sys.argv[1])
stats = json.loads((d / "stats.json").read_text())
spend = json.loads((d / "spend.json").read_text())
# cost of ONE clean pass = the requests the final run actually used (pilot iterations and prompt fixes excluded)
by = {}
rr = d / "run_requests.jsonl"
if rr.exists():
    for line in rr.open():
        x = json.loads(line)
        k = x["slice"].split(":")[0]
        by[k] = by.get(k, 0.0) + x["cost_when_made"]
else:
    for k, v in spend["by_slice"].items():
        by[k.split(":")[0]] = by.get(k.split(":")[0], 0.0) + v["usd"]
by["screen"] = sum(v["usd"] for k, v in spend["by_slice"].items() if k == "screen")
out, total = {}, 0.0
for sl, tgt in TARGETS.items():
    s = stats["slices"].get(sl, {})
    kept = s.get("kept", 0)
    usd = by.get(sl, 0.0)
    per = usd / kept if kept else 0.0
    proj = per * tgt
    total += proj
    out[sl] = {"pilot_kept": kept, "pilot_usd": round(usd, 5), "usd_per_kept_row": round(per, 6), "target_rows": tgt,
               "projected_usd": round(proj, 2)}
scr = by.get("screen", 0.0)
# screened prompts in the pilot cache
n_scr, n_ok = 0, 0
p = Path.home() / ".cache" / "udgam3_sft" / "screen_v1.jsonl"
if p.exists():
    for line in p.open():
        n_scr += 1
        n_ok += json.loads(line)["ok"]
need = sum(TARGETS[s] for s in REAL_PROMPT_SLICES)
scr_proj = (scr / max(1, n_scr)) * need / max(1e-9, n_ok / max(1, n_scr)) if n_scr else 0.0
total += scr_proj
print(json.dumps({"per_slice": out, "prompt_screen": {"pilot_usd": round(scr, 5), "screened": n_scr, "ok": n_ok,
                                                      "projected_usd": round(scr_proj, 2)},
                  "projected_total_usd": round(total, 2), "pilot_total_usd_all_iterations": round(spend["usd"], 5),
                  "one_clean_pass_usd": round(sum(v for k, v in by.items() if k != "screen"), 5)}, indent=1))
