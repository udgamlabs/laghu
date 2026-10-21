"""Render the chat-suite tables (markdown) from scored/<tag>/summary.json and validation.json.

  python3 udgam3/evals/chat_suite/make_tables.py [--tag full] > udgam3/evals/chat_suite/TABLES.md

RESULTS.md embeds these tables; every number links back to per-item files in scored/<tag>/<model>/ and raw outputs in
raw/<tag>/<model>/<engine>/.
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = [("candidate", "Laghu (Udgam-3), measured after part 2b was written"), ("rival", "Size-class rivals (<= 200M)"), ("rival_scan", "Size-class chat models added by the 2026-10-19 scan"),
          ("ours", "Our own bars"), ("reference", "Near-size references (230-362M, reported separately)")]


def ci(t):
    return f"{t[0]:.1f} [{max(t[1], 0.0):.1f}, {min(t[2], 100.0):.1f}]" if t and t[0] == t[0] else "n/a"


def capcol(r, ctx_note):
    return f"{r['cap']}" if r["cap_eff_max"] >= r["cap"] and r["cap_eff_min"] >= r["cap"] else f"{r['cap']} (≤ {r['cap_eff_max']}, context {ctx_note})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    S = json.load(open(os.path.join(HERE, "scored", a.tag, "summary.json")))
    man = {m["name"]: m for m in json.load(open(os.path.join(HERE, "manifest.json")))["models"]}
    V = json.load(open(os.path.join(HERE, "scored", a.tag, "validation.json"))) if os.path.exists(os.path.join(HERE, "scored", a.tag, "validation.json")) else {}
    out = []
    P = out.append

    def head(name):
        m, s = man[name], S[name]
        return f"| {name} | {m['params'] / 1e6:.0f}M | `{m['sha'][:10]}` | {s['engine']} |"

    def rows(task_prefix, fn, var="greedy"):
        for g, title in GROUPS:
            names = [n for n in S if man[n]["group"] == g]
            if not names:
                continue
            P(f"| **{title}** |" + " |" * (ncols - 1))
            for n in names:
                for mode in man[n]["modes"]:
                    key = f"{task_prefix}__{mode}" + (f"__{var}" if var else "")
                    r = S[n]["tasks"].get(key)
                    if r is None:
                        P(head(n) + f" {mode} | not run |" + " |" * (ncols - 6))
                    else:
                        P(head(n) + f" {mode} | " + fn(n, r) + " |")

    # best size-class rival per task (point estimate; the CI shows how sure)
    P("### Best size-class rival per task (≤ 200M; groups 'rival' and 'rival_scan'; best mode per model)\n")
    P("| Task (metric) | Direction | Best rival | Mode | Score [95% CI] | Runner-up | Our bars: Gargi / Udgam-384M-Chat (best mode) |")
    P("|---|---|---|---|---|---|---|")
    rivals = [n for n in S if man[n]["group"] in ("rival", "rival_scan")]
    specs = [("GSM8K accuracy", "gsm8k", "acc", "__greedy", max), ("GSM8K strict (truncated = wrong; extra)", "gsm8k", "acc_strict", "__greedy", max),
             ("MATH-500 L1-3 accuracy", "math", "acc", "__greedy", max), ("MATH-500 L1-3 strict (extra)", "math", "acc_strict", "__greedy", max),
             ("IFEval prompt-level strict", "ifeval", "prompt_strict", "__greedy", max), ("IFEval instruction-level strict", "ifeval", "inst_strict", "__greedy", max),
             ("IFEval prompt strict, answered only (extra)", "ifeval", "prompt_strict_answered", "__greedy", max),
             ("Multi-turn E1 right %", "mt", "right", "", max), ("Multi-turn E1 near-copy % (lower better)", "mt", "copy", "", min),
             ("E2: GSM8K as 2nd turn", "e2", "second", "", max), ("Loop rate, greedy (lower better)", "loop", "loop", "__greedy", min),
             ("Loop rate, shipped sampling (lower better)", "loop", "loop", "__sampled", min)]
    for label, task, metric, suffix, pick in specs:
        cand = []
        for n in list(S):
            for k, r in S[n]["tasks"].items():
                if k.startswith(task + "__") and (k.endswith(suffix) if suffix else True) and r.get(metric) and (task != "e2" or True):
                    if task in ("gsm8k", "math") and not k.endswith("__greedy"):
                        continue
                    cand.append((r[metric][0], n, k.split("__")[1], r[metric]))
        best = {}
        for v, n, mode, t in cand:   # best mode per model
            if n not in best or pick(v, best[n][0]) != best[n][0]:
                best[n] = (v, mode, t)
        rv = sorted(((best[n][0], n) for n in best if n in rivals), reverse=(pick is max))
        if not rv:
            continue
        top, second = rv[0][1], (rv[1][1] if len(rv) > 1 else None)
        ours = " / ".join(f"{ci(best[n][2])} ({best[n][1]})" if n in best else "—" for n in ("gargi", "udgam-384m-chat"))
        P(f"| {label} | {'higher' if pick is max else 'lower'} | **{top}** | {best[top][1]} | **{ci(best[top][2])}** | "
          f"{second} {ci(best[second][2]) if second else ''} | {ours} |")
    P("")
    # GSM8K
    P("### GSM8K test (1,319), greedy, own chat template, answer = last number (math_eval.py rule)\n")
    cols = ("| Model | Params | sha | Engine | Thinking | Cap | Accuracy % [95% CI] | Truncated % [95% CI] | "
            "Strict: truncated = wrong (extra) | Mean tokens | 'Final answer' % |")
    ncols = cols.count("|") - 1
    P(cols); P("|" + "---|" * ncols)
    rows("gsm8k", lambda n, r: f"{capcol(r, man[n]['ctx'])} | **{ci(r['acc'])}** | {ci(r['trunc'])} | {ci(r.get('acc_strict'))} | "
                               f"{r['mean_tokens']:.0f} | {r['final_answer_rate']:.0f}")
    sec = [(n, k, r) for n in S for k, r in S[n]["tasks"].items() if k.startswith("gsm8k__") and not k.endswith("__greedy")]
    if sec:
        P("\nSecondary rows (reasoning models at their recommended cap):\n")
        P("| Model | Thinking | Cap (source) | Items | Accuracy % [95% CI] | Same items at cap 4,096 | Truncated % | Strict (extra) | Mean tokens |"); P("|---|---|---|---|---|---|---|---|---|")
        for n, k, r in sec:
            P(f"| {n} | {k.split('__')[1]} | {r['cap']} ({man[n].get('rec_cap_src', '')}) | {r['n']} | **{ci(r['acc'])}** | "
              f"{ci(r.get('main_cap_same_items'))} | {ci(r['trunc'])} | {ci(r.get('acc_strict'))} | {r['mean_tokens']:.0f} |")
    # MATH
    P("\n### MATH-500, levels 1-3 (238 items: 43 / 90 / 105), greedy, cap 4,096, Math-Verify on the final answer\n")
    cols = "| Model | Params | sha | Engine | Thinking | Cap | Accuracy % [95% CI] | L1 | L2 | L3 | Truncated % | Strict: truncated = wrong (extra) |"
    ncols = cols.count("|") - 1
    P(cols); P("|" + "---|" * ncols)
    rows("math", lambda n, r: f"{capcol(r, man[n]['ctx'])} | **{ci(r['acc'])}** | {r['L1'][0]:.1f} | {r['L2'][0]:.1f} | {r['L3'][0]:.1f} | "
                              f"{r['trunc'][0]:.1f} | {ci(r.get('acc_strict'))}")
    # IFEval
    P("\n### IFEval (541 prompts), lm-eval 0.4.13 scorer, own chat template, greedy, cap 1,280\n")
    cols = ("| Model | Params | sha | Engine | Thinking | Prompt strict % [95% CI] | Prompt loose % | Instruction strict % | Instruction loose % | "
            "Truncated % | Unfinished thinking | Prompt strict, answered only (extra) |")
    ncols = cols.count("|") - 1
    P(cols); P("|" + "---|" * ncols)
    rows("ifeval", lambda n, r: f"**{ci(r['prompt_strict'])}** | {ci(r['prompt_loose'])} | {ci(r['inst_strict'])} | {ci(r['inst_loose'])} | "
                                f"{r['trunc'][0]:.1f} | {r.get('unclosed', 0)} | {ci(r.get('prompt_strict_answered'))}")
    # Multi-turn
    P("\n### Multi-turn (mt_eval.py rules), greedy, same engine for every model\n")
    P("E1: 214 two-turn follow-ups (cap 1,024 per turn). Near-copy = reply 2 nearly repeats reply 1 (lower is better); "
      "right = reply 2 contains the expected answer (144 checkable items).\n")
    cols = ("| Model | Params | sha | Engine | Thinking | Near-copy % [95% CI] (lower better) | Right % [95% CI] | "
            "Near-copy and wrong % (checkable; extra) | Right, answered only (extra) | Reply 2 truncated % | Reply 2 unfinished thinking |")
    ncols = cols.count("|") - 1
    P(cols); P("|" + "---|" * ncols)
    rows("mt", lambda n, r: f"{ci(r['copy'])} | **{ci(r['right'])}** | {ci(r.get('copy_wrong'))} | {ci(r.get('right_answered'))} | "
                            f"{r['trunc2'][0]:.1f} | {r.get('unclosed2', 0)}", var=None)
    P("\nE2: GSM8K items 200-399 asked as the second turn (turn 1 = items 0-199 with their reference solutions) vs the same items asked alone "
      "(the GSM8K rows above); thinking on where the model has a switch; cap 4,096.\n")
    P("| Model | Thinking | Alone % [95% CI] | As 2nd turn % [95% CI] | Difference, points [95% CI] | McNemar p | 2nd-turn truncated % |"); P("|---|---|---|---|---|---|---|")
    for g, title in GROUPS:
        for n in [n for n in S if man[n]["group"] == g]:
            for k, r in S[n]["tasks"].items():
                if k.startswith("e2__"):
                    P(f"| {n} | {k.split('__')[1]} | {ci(r['single'])} | {ci(r['second'])} | {r['diff'][0]:+.1f} [{r['diff'][1]:+.1f}, {r['diff'][2]:+.1f}] | {r['mcnemar_p']:.3f} | {r['trunc'][0]:.1f} |")
    # Loop
    P("\n### Loop rate (324 held-out single-turn chat prompts, cap 1,024): reply hits the cap AND its last 256 tokens repeat\n")
    cols = "| Model | Params | sha | Engine | Thinking | Greedy loop % [95% CI] | Sampled loop % [95% CI] | Shipped sampling defaults (source) | Cap hit %, greedy / sampled |"
    ncols = cols.count("|") - 1
    P(cols); P("|" + "---|" * ncols)

    def loopfn(n, r):
        s = S[n]["tasks"].get(r["_k"].replace("__greedy", "__sampled"))
        return (f"**{ci(r['loop'])}** | {ci(s['loop']) if s else 'n/a'} | {json.dumps(man[n]['sampling'])} ({man[n]['sampling_src']}) | "
                f"{r['cap_hit'][0]:.1f} / {s['cap_hit'][0]:.1f}" if s else "")
    for n in S:
        for k in S[n]["tasks"]:
            S[n]["tasks"][k]["_k"] = k
    rows("loop", loopfn)
    # validation
    if V:
        P("\n### Scorer and generation checks (PREREGISTRATION.md 2.4)\n")
        P("First 40 GSM8K items, first 512 tokens: identical outputs / items compared (median first differing token).\n")
        P("| Model | bf16: batched vs one at a time (same engine) | Production vs HF fp32 one at a time | fp32: batched vs one at a time | Same prompt tokens (vs HF) | bf16 batch-noise rerun (full GSM8K): flips, accuracy change, McNemar p |"); P("|---|---|---|---|---|---|")
        f = lambda c: f"{c['identical']}/{c['n']} ({c['median_first_divergence']})" if c else "—"
        for n in S:
            v = V.get(n, {})
            fp = v.get("vllm_float32_batched_vs_one_by_one") or v.get("hf_float32_batched_vs_one_by_one")
            noise = [x for k, x in v.items() if k.startswith("noise_")]
            nz = "; ".join(f"{x['flips']}/{x['n']} items, {x['acc_rerun'] - x['acc_production']:+.1f} pts, p={x['mcnemar_p']:.2f}" for x in noise) or "—"
            pv = v.get("production_vs_hf_fp32_one_by_one")
            P(f"| {n} | {f(v.get('bf16_batched_vs_one_by_one'))} | {f(pv)} | {f(fp)} | {pv['same_prompt']}/{pv['n'] if pv else ''} | {nz} |" if pv else
              f"| {n} | {f(v.get('bf16_batched_vs_one_by_one'))} | — | {f(fp)} | — | {nz} |")
        ife = {n: v["ifeval_lmeval_check"] for n, v in V.items() if "ifeval_lmeval_check" in v}
        if ife:
            P("\nIFEval scorer check: lm-eval 0.4.13's own run (HF backend, bf16, batch 64, --apply_chat_template) vs ours.\n")
            P("| Model | Items | Our scorer = lm-eval's on lm-eval's responses | lm-eval prompt strict % | Our run prompt strict % | Same reply text (different engine) | Same prompt-strict verdict |"); P("|---|---|---|---|---|---|---|")
            for n, x in ife.items():
                ours = S[n]["tasks"].get(f"ifeval__{man[n]['modes'][0]}__greedy", {}).get("prompt_strict", [float('nan')])[0]
                P(f"| {n} | {x['n']} | {x['our_scorer_agrees_on_lmeval_responses']}/{x['n']} | {100 * x['lmeval_metrics']['prompt_level_strict_acc,none']:.1f} | {ours:.1f} | "
                  f"{x.get('our_engine_same_text', '—')} | {x.get('our_engine_same_prompt_strict', '—')} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
