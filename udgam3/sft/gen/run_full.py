"""Full-run driver (and the post-pilot re-check): one slice per invocation, hard spend cap shared by everything in
--out (spend.json is cumulative), per-slice outputs so finished slices are never rewritten.

  uv run --project evals/harness python -I udgam3/sft/gen/run_full.py --plan full --slice maths_logic \
      --out udgam3/sft/full_v1 --cap 9.30
  ... --merge-only     # rebuild <out>/stats.json from the per-slice stats

Writes <out>/<slice>.jsonl (kept rows, trimmed to the slice target), <out>/rejected_<slice>.jsonl,
<out>/stats_<slice>.json, <out>/stats.json (merged), spend.json and raw_cache.jsonl (client).
Resumable: re-running a slice replays cached requests for free (all choices are seeded per task).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slices as S  # noqa: E402
import slices_full as F  # noqa: E402
from client import DeepSeek  # noqa: E402
from common import CACHE, DupIndex, nwords, pct  # noqa: E402
from run import check_family, post_checks, run_task  # noqa: E402
from verify import MathVerify  # noqa: E402

ORDER = ["maths_logic", "python_tool", "json_function_calls", "safety", "identity", "long_documents", "system_prompts",
         "chat_thinking", "instruction_following", "chat"]
MATHS_FULL = [("prob_gen", lambda r: True, 1090), ("gsm_xcheck", lambda r: True, 530),
              ("math", lambda r: "gsm8k" in r.get("source", "").lower(), 530),
              ("math", lambda r: "MATH" in r.get("source", "") and re.search(r"Level [1-4]", r.get("level", "")), 715),
              ("logic", lambda r: True, 485), ("code_trace", lambda r: True, 245)]   # +10% top-up after first pass
TOOL_FULL = [("prob_gen", lambda r: bool(r.get("code")), 1800), ("gsm_xcheck", lambda r: True, 520),
             ("math", lambda r: "gsm8k" in r.get("source", "").lower(), 560), ("code_trace", lambda r: True, 520)]
PLANS = {
    # targets = kept rows wanted; the builders get enough tasks for the pilot keep rates
    "full": {"targets": {"chat": 10000, "chat_thinking": 1500, "instruction_following": 4000, "system_prompts": 1500,
                         "maths_logic": 6000, "python_tool": 3000, "json_function_calls": 1500,
                         "long_documents": 1500, "safety": 1200, "identity": 400},
             "chat": (8800, 0.30, 0.10), "chat_thinking": 2100, "if": (4700, 0.25),
             "system": (820, 63, 5, 0.15, 40), "maths": MATHS_FULL, "tool": (TOOL_FULL, 120, 3), "fc": 2100,
             "longdoc": 513, "safety": (880, 880, 4), "identity": 400,
             "chat_extra_passages": 220},
    "recheck": {"targets": {}, "chat": (30, 0.30, 0.10), "chat_thinking": 30, "if": (30, 0.25),
                "system": (12, 1, 5, 0.15, 3), "maths": [(j, f, max(1, k // 200)) for j, f, k in MATHS_FULL],
                "tool": ([(j, f, max(1, k // 150)) for j, f, k in TOOL_FULL], 2, 3), "fc": 30, "longdoc": 10,
                "safety": (16, 16, 4), "identity": 0, "longdoc_path": CACHE / "longdocs_pilot.jsonl"},
}


def build(plan: dict, slice_: str, seed: int):
    if slice_ == "chat":
        n, ms, ts = plan["chat"]
        return F.build_chat_full(n, ms, ts, seed, set(), extra_passages=plan.get("chat_extra_passages", 0))
    if slice_ == "chat_thinking":
        return F.build_chat_thinking(plan["chat_thinking"], seed)
    if slice_ == "instruction_following":
        return F.build_if_full(plan["if"][0], plan["if"][1], seed)
    maths = S.maths_items(seed, plan["maths"])
    msids = {m["sid"] for m in maths}
    if slice_ == "maths_logic":
        return S.build_maths(maths)
    if slice_ == "python_tool":
        items, ngen, per = plan["tool"]
        tool_items = [m for m in S.maths_items(seed + 1, items) if m["sid"] not in msids]
        return S.build_tool(tool_items, ngen, per)
    if slice_ == "system_prompts":
        npool, ngen, per, ms, nsoc = plan["system"]
        extra = [m for m in S.maths_items(seed + 2, [("prob_gen", lambda r: "tier 1" in r.get("level", "") or
                                                      "tier 2" in r.get("level", ""), nsoc + 50)])
                 if m["sid"] not in msids][:nsoc]
        return F.build_system_full(npool, ngen, per, ms, seed, extra)
    if slice_ == "json_function_calls":
        return F.build_fc_full(plan["fc"], seed)
    if slice_ == "long_documents":
        return F.build_longdoc_full(plan["longdoc"], plan.get("longdoc_path"))
    if slice_ == "safety":
        nd, na, per = plan["safety"]
        return F.build_safety_full(nd, na, per, seed)
    raise SystemExit(f"unknown slice {slice_}")


def slice_stats(sl, os_, spend_slice):
    kept = [o for o in os_ if o["ok"]]
    fam = collections.defaultdict(lambda: [0, 0])
    for o in os_:
        for k, v in o["checks"].items():
            f = fam[check_family(k)]
            f[0] += 1
            f[1] += bool(v)
    reasons = collections.Counter((o["reason"] or "")[:90] for o in os_ if not o["ok"])
    ans_w, think_w, turns = [], [], []
    for o in kept:
        ms = o["row"]["messages"]
        for m in ms:
            if m["role"] == "assistant" and (m.get("content") or "").strip():
                ans_w.append(nwords(m["content"]))
            if m["role"] == "assistant" and m.get("reasoning_content"):
                think_w.append(nwords(m["reasoning_content"]))
        turns.append(sum(m["role"] == "user" for m in ms))
    first_len = [x for o in os_ for x in o["debug"].get("first_try_length_ok", [])]
    return {"outcomes": len(os_), "kept": len(kept), "keep_rate": round(len(kept) / max(1, len(os_)), 3),
            "checks_pass_rate": {k: f"{v[1]}/{v[0]} ({v[1] / max(1, v[0]):.0%})" for k, v in sorted(fam.items())},
            "reject_reasons": dict(reasons.most_common(25)),
            "answer_words": {"median": pct(ans_w, 0.5), "p95": pct(ans_w, 0.95), "n": len(ans_w)},
            "thinking_words": {"median": pct(think_w, 0.5), "p95": pct(think_w, 0.95), "n": len(think_w)},
            "multi_turn_share": round(sum(t > 1 for t in turns) / max(1, len(turns)), 3),
            "thinking_on_rows": sum(o["row"]["enable_thinking"] for o in kept),
            "sensitive_topic_rows": sum(bool(o["row"]["provenance"].get("sensitive_topic")) for o in kept),
            "categories": dict(collections.Counter(o["row"]["category"] for o in kept).most_common()),
            "chat_first_try_length_ok": f"{sum(first_len)}/{len(first_len)}" if first_len else None,
            "spend_usd": round(spend_slice, 4)}


def merge(out: Path, plan_name: str):
    st = {"plan": plan_name, "slices": {}}
    for f in sorted(out.glob("stats_*.json")):
        st["slices"][f.stem[6:]] = json.loads(f.read_text())
    st["total_kept"] = sum(v["kept"] for v in st["slices"].values())
    sp = json.loads((out / "spend.json").read_text()) if (out / "spend.json").exists() else {}
    st["spend_usd_total_in_dir"] = round(sp.get("usd", 0.0), 4)
    st["spend_by_call_type"] = {k: round(v["usd"], 4) for k, v in sp.get("by_slice", {}).items()}
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    return st


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="full")
    ap.add_argument("--slice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = PLANS[a.plan]
    if a.merge_only:
        print(json.dumps({k: v["kept"] for k, v in merge(out, a.plan)["slices"].items()}))
        return
    S.POOL_FILE = CACHE / "real_pool_v2.jsonl"
    t0 = time.time()
    sl = a.slice
    if sl == "identity":
        rows = S.build_identity(plan["identity"], a.seed)
        outcomes = [{"ok": True, "id": r["id"], "slice": "identity", "category": "identity", "row": r, "reason": None,
                     "checks": {}, "debug": {}} for r in rows]
        before = 0.0
        spend_by = {}
    else:
        tasks = build(plan, sl, a.seed)
        print(f"{sl}: {len(tasks)} tasks ({dict(collections.Counter(t.category for t in tasks).most_common(12))})",
              file=sys.stderr)
        if a.dry:
            for t in tasks[:40]:
                d = t.data
                txt = (d.get("prompt") or {}).get("t1") or (d.get("item") or {}).get("problem") or d.get("user") \
                    or (d.get("doc") or {}).get("title") or d.get("domain") or d.get("theme") or d.get("cat")
                print(f"{t.category:24s} {t.id:36s} | {str(txt)[:120]!r}")
            return
        mv = MathVerify()
        async with DeepSeek(out, cap_usd=a.cap, concurrency=min(a.concurrency, 32)) as ds:
            before = sum(v["usd"] for k, v in ds.spend["by_slice"].items() if k.split(":")[0] == sl)
            ctx = S.Ctx(ds=ds, mv=mv, seed=a.seed, judge=True)
            results = await asyncio.gather(*[run_task(ctx, t) for t in tasks])
            spend_by = dict(ds.spend["by_slice"])
            served = list(ds.served)
        await mv.close()
        outcomes = [o for rs in results for o in rs]
        with (out / f"requests_{sl}.jsonl").open("w") as f:          # accounting: cost of the requests used
            for x in served:
                f.write(json.dumps(x) + "\n")
    # global filters + dedup of the first user turn (paired thinking rows share a prompt)
    dups, seen, seen_pids, seen_ids = DupIndex(), set(), set(), set()
    for o in outcomes:
        if not o["ok"]:
            continue
        r = o["row"]
        pc = post_checks(r)
        o["checks"].update(pc)
        key = r.get("pair_id") or r["id"]
        if r["id"] in seen_ids:                 # the same task twice (e.g. a problem listed by two factory jobs)
            o["checks"]["not_duplicate"] = False
        elif sl == "instruction_following":
            # IF prompts share constraint sentences, so 8-gram containment would flag different prompts; their base
            # prompts were already near-deduplicated in the pool, so dedup by base prompt id instead
            pid = r["provenance"].get("prompt_id")
            o["checks"]["not_duplicate"] = pid not in seen_pids
            seen_pids.add(pid)
        elif sl not in ("identity", "long_documents") and key not in seen:
            first = next(m["content"] for m in r["messages"] if m["role"] == "user")
            o["checks"]["not_duplicate"] = not dups.is_dup(first)
            if o["checks"]["not_duplicate"]:
                dups.add(first)
        seen.add(key)
        seen_ids.add(r["id"])
        failed = [k for k, v in o["checks"].items() if not v]
        if failed:
            o["ok"], o["reason"] = False, "post_filter:" + ",".join(failed)
        else:
            r["verified"]["passed"] = True
            r["verified"].setdefault("details", {})["global_filters"] = sorted(pc)
    target = plan["targets"].get(sl)
    kept = [o for o in outcomes if o["ok"]]
    if target and len(kept) > target:                   # trim to the target (task order is seeded)
        if sl == "maths_logic":                         # keep think/no-think pairs together
            pairs = list(dict.fromkeys(o["row"]["pair_id"] for o in kept))[:target // 2]
            keepset = set(pairs)
            for o in kept:
                if o["row"]["pair_id"] not in keepset:
                    o["ok"], o["reason"] = False, "trimmed_over_target"
        else:                                           # trim from the largest category (keeps rare ones whole)
            excess = len(kept) - target
            by_cat = collections.defaultdict(list)
            for o in kept:
                by_cat[o["row"]["category"]].append(o)
            while excess > 0:
                big = max(by_cat, key=lambda c: len(by_cat[c]))
                o = by_cat[big].pop()
                o["ok"], o["reason"] = False, "trimmed_over_target"
                excess -= 1
    with open(out / f"{sl}.jsonl", "w", encoding="utf-8") as f:
        for o in outcomes:
            if o["ok"]:
                f.write(json.dumps(o["row"], ensure_ascii=False) + "\n")
    with open(out / f"rejected_{sl}.jsonl", "w", encoding="utf-8") as f:
        for o in outcomes:
            if not o["ok"] and o["reason"] != "trimmed_over_target":
                f.write(json.dumps({k: o[k] for k in ("id", "slice", "category", "reason", "checks", "debug", "row")},
                                   ensure_ascii=False) + "\n")
    used = sum(v["usd"] for k, v in spend_by.items() if k.split(":")[0] == sl)
    st = slice_stats(sl, [o for o in outcomes if o["reason"] != "trimmed_over_target"], used)
    st["trimmed_over_target"] = sum(o["reason"] == "trimmed_over_target" for o in outcomes)
    st["seconds"] = round(time.time() - t0, 1)
    st["not_run_spend_cap"] = sum((o["reason"] or "") == "not_run_spend_cap" for o in outcomes)
    (out / f"stats_{sl}.json").write_text(json.dumps(st, indent=1))
    m = merge(out, a.plan)
    print(json.dumps({"slice": sl, "kept": sum(o["ok"] for o in outcomes), "outcomes": st["outcomes"],
                      "not_run_spend_cap": st["not_run_spend_cap"], "slice_spend": round(used, 4),
                      "dir_spend_total": m["spend_usd_total_in_dir"], "seconds": st["seconds"]}))


if __name__ == "__main__":
    asyncio.run(main())
