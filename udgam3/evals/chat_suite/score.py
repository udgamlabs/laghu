"""Score the chat suite from the saved raw outputs (Mac, CPU only) and write per-item scores + summary tables.

  uv run --project evals/harness --with langdetect==1.0.9 --with immutabledict==4.2.1 --with math-verify==0.9.0 \
      python udgam3/evals/chat_suite/score.py [--tag full]

Rules (fixed 2026-10-19, before any Udgam-3 chat model exists; identical for every model):
- Visible reply = the raw output with thinking removed (common.split_thinking) and special tokens dropped.
- GSM8K: math_eval.py's rule, the number after the last "Final answer", else the last number (common.gsm8k_ok).
- MATH-500 L1-3: Math-Verify 0.9.0 on the text after the last "Final answer" (also tried wrapped in $...$), else on the
  whole reply (its last \\boxed{} / expression); when the gold is a plain number, the GSM8K last-number rule also counts.
- IFEval: lm-eval 0.4.13's own ifeval scorer (lm_eval.tasks.ifeval.utils.process_results) on the visible reply.
- Multi-turn E1: mt_eval.py's near-copy (difflib ratio > 0.6 on the first 400 characters) and right% (word-boundary
  match of a check string in reply 2); E2: GSM8K items 200-399 asked as the second turn vs asked alone (the same items
  of the GSM8K task, same mode and cap).
- Loop: the reply hits its cap without stopping AND its last 256 tokens are periodic (common.periodic_tail).
- Truncation: the reply hit its output cap (finish == "length").
- 95% CIs: Wilson for proportions; paired bootstrap (10,000 resamples over prompts / items) for instruction-level
  IFEval and the E2 difference; exact McNemar p for E2 (PREREGISTRATION.md 4).
"""
import argparse, glob, gzip, json, math, os, random, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import E2_N, gsm8k_ok, hit, num, sim, split_thinking, visible  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def boot_ci(groups, stat, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(groups)
    vals = sorted(stat([groups[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return 100 * vals[int(0.025 * B)], 100 * vals[int(0.975 * B) - 1]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_gz(p):
    return [json.loads(l) for l in gzip.open(p, "rt")]


_MV = {}


def math_ok(ans, gold):
    if not _MV:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
        _MV.update(parse=parse, verify=verify, cfg=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
                   gcfg=[LatexExtractionConfig(), ExprExtractionConfig()])
    parse, verify = _MV["parse"], _MV["verify"]
    g = parse("$" + gold + "$", extraction_config=_MV["gcfg"])
    if "Final answer" in ans:
        t = ans.split("Final answer")[-1].lstrip(":*_ \t").strip()
        cands = [t] + ([] if ("$" in t or "\\boxed" in t or "\\(" in t) else ["$" + t.split("\n")[0].strip().rstrip(".") + "$"])
    else:
        cands = [ans]
    for c in cands:
        try:
            p = parse(c, extraction_config=_MV["cfg"])
            if p and verify(g, p):
                return True
        except Exception:
            pass
    if re.fullmatch(r"-?\d+(\.\d+)?", gold.strip()):
        return gsm8k_ok(ans, gold.strip())
    return False


_IF = {}


def ifeval_scores(doc, response):
    if not _IF:
        import langdetect
        from lm_eval.tasks.ifeval import utils
        langdetect.DetectorFactory.seed = 0   # IFEval's language checks call langdetect, which is random unless seeded
        _IF["f"] = utils.process_results
    # an instruction whose argument is invalid gets a random one inside lm-eval (e.g. IFEval key 1122's "letter" is
    # "#", so a random letter is drawn on every call): seed per item so rescoring is deterministic
    random.seed(int(doc["key"]))
    return _IF["f"](doc, [response])


def fmt(p, lo, hi):
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    raw_root = os.path.join(HERE, "raw", a.tag)
    out_root = os.path.join(HERE, "scored", a.tag)
    os.makedirs(out_root, exist_ok=True)
    manifest = {m["name"]: m for m in json.load(open(os.path.join(HERE, "manifest.json")))["models"]}
    items = {k: {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "items", f)))}
             for k, f in [("gsm8k", "gsm8k.jsonl"), ("math", "math_l123.jsonl"), ("ifeval", "ifeval.jsonl"), ("mt", "mt.jsonl")]}
    summary = {}
    for name, spec in manifest.items():
        if a.only and name not in a.only.split(","):
            continue
        d = os.path.join(raw_root, name, spec["engine"])
        if not os.path.exists(os.path.join(d, "meta.json")):
            print(f"{name}: no outputs in {d}")
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        specials, stops_s = meta["specials"], meta["stop_strings"]
        vis = lambda raw: visible(raw, specials, stops_s)
        S = summary[name] = dict(group=spec["group"], params=spec["params"], sha=spec["sha"], engine=spec["engine"],
                                 versions=meta["versions"], dtype=meta.get("dtype", "bfloat16"), max_len=meta["max_len"],
                                 skipped=len(meta["skipped"]), limit=meta["limit"], tasks={})
        files = {os.path.basename(p)[:-9]: load_gz(p) for p in glob.glob(os.path.join(d, "*.jsonl.gz"))}
        per_item = {}

        def caps(recs):
            m = [r["max_new"] for r in recs]
            return dict(cap=recs[0]["cap"], cap_eff_min=min(m), cap_eff_max=max(m))

        order = dict(gsm8k=0, math=1, ifeval=2, mt1=3, e2=4, loop=5)   # e2 needs the GSM8K items scored first
        for key, recs in sorted(((k, v) for k, v in files.items() if k.split("__")[0] in order),
                                key=lambda kv: (order[kv[0].split("__")[0]], not kv[0].endswith("__greedy"), kv[0])):
            task, mode, var = key.split("__")
            if task in ("gsm8k", "math"):
                rows = []
                for r in recs:
                    ans = vis(r["raw"])
                    gold = items[task][r["id"]]["gold"]
                    ok = gsm8k_ok(ans, gold) if task == "gsm8k" else math_ok(ans, gold)
                    rows.append(dict(id=r["id"], ok=bool(ok), trunc=r["finish"] == "length", n=r["n"],
                                     pred=num(ans.split("Final answer")[-1]) if task == "gsm8k" else None,
                                     fa="Final answer" in ans, unclosed=split_thinking(r["raw"])[2]))
                k, n = sum(x["ok"] for x in rows), len(rows)
                t = sum(x["trunc"] for x in rows)
                res = dict(n=n, acc=wilson(k, n), trunc=wilson(t, n), mean_tokens=round(sum(x["n"] for x in rows) / n, 1),
                           # extra diagnostic (not pre-registered): a reply cut off by the cap counts as wrong
                           acc_strict=wilson(sum(x["ok"] and not x["trunc"] for x in rows), n),
                           final_answer_rate=round(100 * sum(x["fa"] for x in rows) / n, 1), **caps(recs))
                if task == "math":
                    for lv in (1, 2, 3):
                        sub = [x for x in rows if items["math"][x["id"]]["level"] == lv]
                        res[f"L{lv}"] = wilson(sum(x["ok"] for x in sub), len(sub)) + (len(sub),)
                if var != "greedy":   # a long-cap row: the main-cap accuracy on the same items, for a paired reading
                    main = {x["id"]: x for x in per_item.get(f"{task}__{mode}__greedy", [])}
                    same = [main[x["id"]]["ok"] for x in rows if x["id"] in main]
                    res["main_cap_same_items"] = wilson(sum(same), len(same))
                S["tasks"][key] = res
                per_item[key] = rows
            elif task == "ifeval":
                rows = []
                for r in recs:
                    doc = items["ifeval"][r["id"]]
                    resp = vis(r["raw"])
                    sc = ifeval_scores(doc, resp)
                    rows.append(dict(id=r["id"], trunc=r["finish"] == "length", unclosed=split_thinking(r["raw"])[2],
                                     **{k: (v if isinstance(v, list) else bool(v)) for k, v in sc.items()}))
                n = len(rows)
                res = dict(n=n, trunc=wilson(sum(x["trunc"] for x in rows), n), unclosed=sum(x["unclosed"] for x in rows),
                           # extra diagnostic (not pre-registered): no credit when the reply is an unfinished thinking block
                           prompt_strict_answered=wilson(sum(x["prompt_level_strict_acc"] and not x["unclosed"] for x in rows), n),
                           **caps(recs))
                for lvl in ("strict", "loose"):
                    res[f"prompt_{lvl}"] = wilson(sum(x[f"prompt_level_{lvl}_acc"] for x in rows), n)
                    flat = [v for x in rows for v in x[f"inst_level_{lvl}_acc"]]
                    stat = lambda g, lvl=lvl: sum(sum(x[f"inst_level_{lvl}_acc"]) for x in g) / sum(len(x[f"inst_level_{lvl}_acc"]) for x in g)
                    res[f"inst_{lvl}"] = (100 * sum(flat) / len(flat),) + boot_ci(rows, stat, B=2000 if a.tag != "full" else 10000)
                S["tasks"][key] = res
                per_item[key] = rows
            elif task == "mt1":
                r2 = {r["id"]: r for r in files.get(f"mt2__{mode}__{var}", [])}
                rows = []
                for r in recs:
                    it = items["mt"][r["id"]]
                    if r["id"] not in r2:
                        continue
                    v1, v2 = vis(r["raw"]).strip(), vis(r2[r["id"]]["raw"]).strip()
                    rows.append(dict(id=r["id"], copy=sim(v1, v2) > 0.6, right=hit(v2, it["check"]) if it["check"] else None,
                                     trunc1=r["finish"] == "length", trunc2=r2[r["id"]]["finish"] == "length", empty2=not v2,
                                     unclosed2=split_thinking(r2[r["id"]]["raw"])[2]))
                n = len(rows)
                chk = [x for x in rows if x["right"] is not None]
                S["tasks"][f"mt__{mode}"] = dict(n=n, copy=wilson(sum(x["copy"] for x in rows), n), n_check=len(chk),
                                                 right=wilson(sum(x["right"] for x in chk), len(chk)),
                                                 # extra diagnostic (not pre-registered): a near-copy that is also wrong
                                                 copy_wrong=wilson(sum(x["copy"] and not x["right"] for x in chk), len(chk)),
                                                 right_answered=wilson(sum(x["right"] and not x["unclosed2"] for x in chk), len(chk)),
                                                 unclosed2=sum(x["unclosed2"] for x in rows),
                                                 trunc2=wilson(sum(x["trunc2"] for x in rows), n), empty2=sum(x["empty2"] for x in rows),
                                                 **caps(recs))
                per_item[f"mt__{mode}"] = rows
            elif task == "e2":
                single = {x["id"]: x for x in per_item.get(f"gsm8k__{mode}__greedy", [])}
                rows = []
                for r in recs:
                    gold = items["gsm8k"][r["id"]]["gold"]
                    s = single.get(r["id"])
                    if s is None:
                        continue
                    rows.append(dict(id=r["id"], two=bool(gsm8k_ok(vis(r["raw"]), gold)), one=s["ok"], trunc=r["finish"] == "length"))
                n = len(rows)
                if n == 0:   # (smoke runs: the limited GSM8K items don't include items 200-399)
                    continue
                b = sum(1 for x in rows if x["one"] and not x["two"]); c = sum(1 for x in rows if x["two"] and not x["one"])
                diff = 100 * (sum(x["two"] for x in rows) - sum(x["one"] for x in rows)) / max(n, 1)
                S["tasks"][f"e2__{mode}"] = dict(n=n, single=wilson(sum(x["one"] for x in rows), n), second=wilson(sum(x["two"] for x in rows), n),
                                                 diff=(diff,) + boot_ci(rows, lambda g: sum(x["two"] - x["one"] for x in g) / len(g), B=10000),
                                                 mcnemar_p=mcnemar_p(b, c), trunc=wilson(sum(x["trunc"] for x in rows), n), **caps(recs))
                per_item[f"e2__{mode}"] = rows
            elif task == "loop":
                rows = [dict(id=r["id"], loop=r["finish"] == "length" and r["period"] > 0, cap_hit=r["finish"] == "length", period=r["period"])
                        for r in recs]
                n = len(rows)
                S["tasks"][key] = dict(n=n, loop=wilson(sum(x["loop"] for x in rows), n), cap_hit=wilson(sum(x["cap_hit"] for x in rows), n),
                                       sampling=spec["sampling"] if var == "sampled" else "greedy", **caps(recs))
                per_item[key] = rows
        od = os.path.join(out_root, name)
        os.makedirs(od, exist_ok=True)
        for key, rows in per_item.items():
            with open(os.path.join(od, f"{key}.jsonl"), "w") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
        print(f"scored {name}: {', '.join(sorted(S['tasks']))}", flush=True)
    json.dump(summary, open(os.path.join(out_root, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
