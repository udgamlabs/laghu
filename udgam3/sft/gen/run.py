"""Udgam-3 SFT generator: build tasks for a plan, run them against DeepSeek with a hard spend cap, verify, filter, and
write raw conversations (not rendered) per slice.

  uv run --project evals/harness python -I udgam3/sft/gen/run.py --plan pilot --out udgam3/sft/pilot_v1 --cap 0.15

Resumable: every response is cached in <out>/raw_cache.jsonl keyed by the exact request, and all random choices are
seeded per task, so re-running the same command replays finished calls for free and only pays for missing ones.
Outputs (rewritten on every run): <slice>.jsonl (kept rows), rejected.jsonl (with reasons), stats.json, spend.json.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slices as S  # noqa: E402
from client import DeepSeek, SpendCapReached  # noqa: E402
from common import (DupIndex, contamination, english_output_ok, fingerprint, norm_for_hash, nwords,  # noqa: E402
                    output_pii, pct)
from verify import MathVerify  # noqa: E402

PLANS = {
    "pilot": {"chat": (60, 20, 6), "if": (25, 5), "system": (16, 3), "fc": 15, "longdoc": 8, "safety": (4, 2, 2),
              "identity": 12,
              "maths": [("prob_gen", lambda r: True, 7), ("gsm_xcheck", lambda r: True, 3),
                        ("math", lambda r: "gsm8k" in r.get("source", "").lower(), 3),
                        ("math", lambda r: "MATH" in r.get("source", "") and
                         re.search(r"Level [1-4]", r.get("level", "")), 6),
                        ("logic", lambda r: True, 4), ("code_trace", lambda r: True, 2)],
              "tool": ([("prob_gen", lambda r: bool(r.get("code")), 5), ("gsm_xcheck", lambda r: True, 2),
                        ("math", lambda r: "gsm8k" in r.get("source", "").lower(), 2),
                        ("code_trace", lambda r: True, 3)], 1, 3)},
}
TEMPORAL_RX = re.compile(r"\bas of (?:my|now|today|this writing|(?:early |mid-|late )?20\d\d)\b|\bthe current year "
                         r"is\b|\btoday'?s date is\b|\bit is currently (?:20\d\d|\w+ \d{1,2}, 20\d\d)\b", re.I)


def build_tasks(plan: dict, seed: int):
    used: set = set()
    tasks = []
    c = plan["chat"]
    tasks += S.build_chat(c[0], c[1], c[2], seed, used)
    tasks += S.build_if(plan["if"][0], plan["if"][1], seed, used)
    maths = S.maths_items(seed, plan["maths"])
    msids = {m["sid"] for m in maths}
    tool_items = [m for m in S.maths_items(seed + 1, plan["tool"][0]) if m["sid"] not in msids]
    tasks += S.build_maths(maths)
    tasks += S.build_tool(tool_items, plan["tool"][1], plan["tool"][2])
    extra_m = [m for m in S.maths_items(seed + 2, [("prob_gen", lambda r: "tier 1" in r.get("level", "") or
                                                    "tier 2" in r.get("level", ""), 1)])
               if m["sid"] not in msids]
    tasks += S.build_system(plan["system"][0], plan["system"][1], seed, used, extra_m)
    tasks += S.build_fc(plan["fc"], seed)
    tasks += S.build_longdoc(plan["longdoc"])
    s = plan["safety"]
    tasks += S.build_safety(s[0], s[1], s[2], seed)
    return tasks, S.build_identity(plan["identity"], seed)


def assistant_texts(r: dict) -> list[str]:
    out = []
    for m in r["messages"]:
        if m["role"] == "assistant":
            out += [m.get("content") or "", m.get("reasoning_content") or ""]
            for tc in m.get("tool_calls") or []:
                out.append(json.dumps(tc["function"].get("arguments"), ensure_ascii=False))
        if m["role"] == "tool":
            out.append(m.get("content") or "")
    return [t for t in out if t]


def post_checks(r: dict) -> dict:
    """Global filters applied to every kept row."""
    texts = assistant_texts(r)
    ch = {}
    if r["slice"] != "identity":
        fp = [fingerprint(t) for t in texts]
        ch["no_teacher_fingerprint"] = not any(fp)
    ch["english_output"] = all(english_output_ok(t) for t in texts)
    ch["no_output_pii"] = not any(output_pii(t) for t in texts)
    ch["no_temporal_claim"] = not any(TEMPORAL_RX.search(t) for t in texts)
    users = [m["content"] for m in r["messages"] if m["role"] == "user"]
    con = contamination(users, long_doc=r["slice"] == "long_documents")
    ch["decontaminated"] = con is None
    ch["nonempty_final"] = bool((r["messages"][-1].get("content") or "").strip())
    return ch


async def run_task(ctx, t):
    try:
        return await S_wrap(ctx, t)
    except SpendCapReached as e:
        return [S.bad(t, "not_run_spend_cap", debug={"err": str(e)})]
    except Exception as e:                       # noqa: BLE001
        return [S.bad(t, f"error:{type(e).__name__}", debug={"err": str(e)[:300],
                                                            "tb": traceback.format_exc()[-800:]})]


async def S_wrap(ctx, t):
    return await t.fn(ctx, t)


def check_family(name: str) -> str:
    return re.sub(r"^(turn|t|followup)\d+_", r"", name)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, required=True, help="hard spend cap in USD (includes earlier runs in --out)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these slices")
    ap.add_argument("--dry", action="store_true", help="build and print the tasks; no API calls")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks of each slice (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    t0 = time.time()
    tasks, identity = build_tasks(PLANS[a.plan], a.seed)
    if a.only:
        tasks = [t for t in tasks if t.slice in a.only]
    if a.limit:
        cnt = collections.Counter()
        keep = []
        for t in tasks:
            cnt[t.slice] += 1
            if cnt[t.slice] <= a.limit:
                keep.append(t)
        tasks = keep
    print(f"{len(tasks)} tasks: {dict(collections.Counter(t.slice for t in tasks))}", file=sys.stderr)
    if a.dry:
        for t in tasks:
            d = t.data
            txt = (d.get("prompt") or {}).get("t1") or (d.get("item") or {}).get("problem") or d.get("user") or \
                (d.get("doc") or {}).get("title") or d.get("kind") or str(d.get("cat") or d.get("system", ""))[:80]
            extra = f" turns={d['turns']}" if "turns" in d else (f" multi={d['multi']}" if "multi" in d else "")
            print(f"{t.slice:22s} {t.category:22s} {t.id:34s}{extra} | {str(txt)[:110]!r}")
        return
    mv = MathVerify()
    async with DeepSeek(out, cap_usd=a.cap, concurrency=min(a.concurrency, 32)) as ds:
        ctx = S.Ctx(ds=ds, mv=mv, seed=a.seed, judge=not a.no_judge)
        results = await asyncio.gather(*[run_task(ctx, t) for t in tasks])
        spend = dict(ds.spend)
    await mv.close()
    outcomes = [o for rs in results for o in rs]
    outcomes += [{"ok": True, "id": r["id"], "slice": "identity", "category": "identity", "row": r, "reason": None,
                  "checks": {}, "debug": {}} for r in identity]

    # global filters + dedup of the first user turn within each slice (paired thinking rows share a prompt)
    dups: dict[str, DupIndex] = collections.defaultdict(DupIndex)
    seen_pairs = set()
    for o in outcomes:
        if not o["ok"]:
            continue
        r = o["row"]
        pc = post_checks(r)
        o["checks"].update(pc)
        first = next(m["content"] for m in r["messages"] if m["role"] == "user")
        key = r.get("pair_id") or r["id"]
        if r["slice"] not in ("identity", "long_documents") and key not in seen_pairs:
            if dups[r["slice"]].is_dup(first):
                o["checks"]["not_duplicate"] = False
            else:
                dups[r["slice"]].add(first)
                o["checks"]["not_duplicate"] = True
        seen_pairs.add(key)
        failed = [k for k, v in o["checks"].items() if not v]
        if failed:
            o["ok"] = False
            o["reason"] = "post_filter:" + ",".join(failed)
        else:
            r["verified"]["passed"] = True
            r["verified"].setdefault("details", {})["global_filters"] = sorted(pc)

    # write
    by_slice = collections.defaultdict(list)
    for o in outcomes:
        if o["ok"]:
            by_slice[o["slice"]].append(o["row"])
    for f in out.glob("*.jsonl"):
        if f.name not in ("raw_cache.jsonl", "run_requests.jsonl"):
            f.unlink()
    for sl, rows in by_slice.items():
        with open(out / f"{sl}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "rejected.jsonl", "w", encoding="utf-8") as fh:
        for o in outcomes:
            if not o["ok"]:
                fh.write(json.dumps({k: o[k] for k in ("id", "slice", "category", "reason", "checks", "debug", "row")},
                                    ensure_ascii=False) + "\n")

    # stats
    stats = {"plan": a.plan, "seed": a.seed, "seconds": round(time.time() - t0, 1), "slices": {}}
    for sl in sorted({o["slice"] for o in outcomes}):
        os_ = [o for o in outcomes if o["slice"] == sl]
        kept = [o for o in os_ if o["ok"]]
        fam = collections.defaultdict(lambda: [0, 0])
        for o in os_:
            for k, v in o["checks"].items():
                f = fam[check_family(k)]
                f[0] += 1
                f[1] += bool(v)
        reasons = collections.Counter(re.sub(r":.*", "", o["reason"]) + (":" + o["reason"].split(":", 1)[1]
                                      if o["reason"] and ":" in o["reason"] else "") for o in os_ if not o["ok"])
        ans_w, think_w, turns = [], [], []
        for o in kept:
            ms = o["row"]["messages"]
            for m in ms:
                if m["role"] == "assistant" and (m.get("content") or "").strip():
                    ans_w.append(nwords(m["content"]))
                if m["role"] == "assistant" and m.get("reasoning_content"):
                    think_w.append(nwords(m["reasoning_content"]))
            turns.append(sum(m["role"] == "user" for m in ms))
        stats["slices"][sl] = {
            "outcomes": len(os_), "kept": len(kept), "keep_rate": round(len(kept) / max(1, len(os_)), 3),
            "checks_pass_rate": {k: f"{v[1]}/{v[0]} ({v[1] / max(1, v[0]):.0%})" for k, v in sorted(fam.items())},
            "reject_reasons": dict(reasons.most_common()),
            "answer_words": {"median": pct(ans_w, 0.5), "p95": pct(ans_w, 0.95), "n": len(ans_w)},
            "thinking_words": {"median": pct(think_w, 0.5), "p95": pct(think_w, 0.95), "n": len(think_w)},
            "multi_turn_share": round(sum(t > 1 for t in turns) / max(1, len(turns)), 2),
            "thinking_on_rows": sum(o["row"]["enable_thinking"] for o in kept),
            "categories": dict(collections.Counter(o["row"]["category"] for o in kept)),
            "first_try": {
                "length_ok_turns": (lambda xs: f"{sum(xs)}/{len(xs)}")(
                    [x for o in os_ for x in o["debug"].get("first_try_length_ok", [])]),
                "if_all_constraints_rows": (lambda xs: f"{sum(xs)}/{len(xs)}")(
                    [t[0] == 1 for o in os_ for t in [o["debug"].get("tries") or o["row"] and
                     (o["row"].get("verified") or {}).get("details", {}).get("tries") or []] if t])},
            "spend_usd": round(sum(v["usd"] for k, v in spend["by_slice"].items() if k.split(":")[0] == sl), 5),
        }
    stats["total_kept"] = sum(v["kept"] for v in stats["slices"].values())
    stats["spend_usd"] = round(spend["usd"], 5)
    stats["requests"] = spend["requests"]
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({"kept": {k: v["kept"] for k, v in stats["slices"].items()}, "total": stats["total_kept"],
                      "spend_usd": stats["spend_usd"], "requests": stats["requests"]}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
