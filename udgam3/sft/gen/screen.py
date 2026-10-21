"""Batch LLM pre-screen of real-user prompts (20 per call): is the prompt a good, safe, self-contained request to
train a small general assistant on? Verdicts are cached in ~/.cache/udgam3_sft/screen_v1.jsonl (pid -> ok/why) and
`slices.pool_take` only uses prompts screened ok (when the cache has an entry for them).

  uv run --project evals/harness python -I udgam3/sft/gen/screen.py --out udgam3/sft/pilot_v1 --cap 0.15 --n 600
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DeepSeek, SpendCapReached  # noqa: E402
from common import CACHE  # noqa: E402

SCREEN = CACHE / "screen_v1.jsonl"
SYS = "You screen user prompts for a training set for a small, general, English chat assistant. Return JSON only."
USER = """For each numbered prompt below decide whether it is a good training prompt. Mark ok=false if ANY of these hold:
1. it asks about the assistant's identity, who made it, its model or its technical workings (simple greetings and small talk such as "how are you?" or "what's up?" are fine);
2. it asks the assistant to lie, pretend, role-play a character or continue a story with named fictional characters;
3. it needs live or recent information (latest versions, current events, prices, releases after 2023), browsing, a file, an image or a link;
4. it is sexual, hateful, violent, about self-harm, or seeks dangerous or illegal help;
5. it is incoherent, a fragment, test text, or not a real request;
6. it needs a very long answer (more than about 400 words, e.g. 50 questions, a whole app, a long essay);
7. it contains personal details about a private person, or asks for facts about an obscure private individual;
8. it is fan fiction or depends on detailed knowledge of a TV show, game or anime;
9. it is so obscure that a careful assistant could not answer it without inventing facts.
Otherwise ok=true (everyday questions, explanations, advice, writing, editing, coding help, brainstorming and casual chat are all fine).

{items}

Return JSON: {{"verdicts": [{{"i": <number>, "ok": true|false, "why": "<rule number or empty>"}}, ...]}} with one entry per prompt."""

USER_COMPACT = USER.split("{items}")[0] + """{items}

Return JSON listing ONLY the prompts that fail, with the rule number: {{"reject": [[<number>, <rule>], ...]}} (an empty list if all are fine)."""


def load_verdicts() -> dict:
    v = {}
    if SCREEN.exists():
        for line in SCREEN.open(encoding="utf-8"):
            r = json.loads(line)
            v[r["pid"]] = r
    return v


async def screen_batch_compact(ds, batch):
    """Full-run format: only the failing prompts are listed (far fewer output tokens); prompts cut to 600 chars."""
    items = "\n\n".join(f"[{i}] " + (p["t1"][:600] + (f"\n(follow-up turn: {p['t2'][:200]})" if p.get("t2") else ""))
                        for i, p in enumerate(batch))
    r = await ds.chat([{"role": "system", "content": SYS}, {"role": "user", "content": USER_COMPACT.format(items=items)}],
                      slice_name="screen", json_mode=True, temperature=0.0, max_tokens=40 + 12 * len(batch))
    try:
        o = json.loads(r["content"])
        rej = {int(x[0]): str(x[1])[:10] for x in o.get("reject", []) if isinstance(x, list) and x}
    except (json.JSONDecodeError, ValueError, TypeError, KeyError, IndexError):
        return []
    if r["finish_reason"] == "length":
        return []
    return [{"pid": p["pid"], "ok": i not in rej, "why": rej.get(i, "")} for i, p in enumerate(batch)]


async def screen_batch(ds, batch):
    items = "\n\n".join(f"[{i}] " + (p["t1"][:1500] + (f"\n(follow-up turn: {p['t2'][:300]})" if p.get("t2") else ""))
                        for i, p in enumerate(batch))
    r = await ds.chat([{"role": "system", "content": SYS}, {"role": "user", "content": USER.format(items=items)}],
                      slice_name="screen", json_mode=True, temperature=0.0, max_tokens=60 + 25 * len(batch))
    try:
        o = json.loads(r["content"])
        vs = {int(x["i"]): x for x in o.get("verdicts", []) if isinstance(x, dict)}
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return []
    out = []
    for i, p in enumerate(batch):
        if i in vs:
            out.append({"pid": p["pid"], "ok": bool(vs[i].get("ok")), "why": str(vs[i].get("why", ""))[:40]})
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--pool", default="v1")
    ap.add_argument("--targets", help="JSON {category: n} for the full run (compact format, 25 per call)")
    a = ap.parse_args()
    pool = [json.loads(l) for l in open(CACHE / f"real_pool_{a.pool}.jsonl", encoding="utf-8")]
    if a.targets:
        return await main_targets(a, pool)
    done = load_verdicts()
    rng = random.Random(f"{a.seed}:screen")
    cands = [p for p in pool if p["pid"] not in done and p["category"] != "maths_logic"
             and (a.cats is None or p["category"] in a.cats)]
    cands.sort(key=lambda p: hashlib.sha256(f"{a.seed}:{p['pid']}".encode()).hexdigest())
    # stratify: take the first n by category share, with rarer categories over-represented
    by_cat = {}
    for p in cands:
        by_cat.setdefault(p["category"], []).append(p)
    pick = []
    while len(pick) < a.n and any(by_cat.values()):
        for c in sorted(by_cat):
            if by_cat[c] and len(pick) < a.n:
                pick.append(by_cat[c].pop(0))
    with_t2 = [p for p in cands if p.get("t2") and p not in pick][:max(0, a.n // 10)]
    pick += with_t2
    rng.shuffle(pick)
    batches = [pick[i:i + 20] for i in range(0, len(pick), 20)]
    async with DeepSeek(Path(a.out), cap_usd=a.cap, concurrency=16) as ds:
        res = []

        async def one(b):
            try:
                return await screen_batch(ds, b)
            except SpendCapReached:
                return []
        for r in await asyncio.gather(*[one(b) for b in batches]):
            res += r
        spent = ds.spend["usd"]
    with SCREEN.open("a", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    oks = sum(r["ok"] for r in res)
    why = {}
    for r in res:
        if not r["ok"]:
            why[r["why"]] = why.get(r["why"], 0) + 1
    print(json.dumps({"screened": len(res), "ok": oks, "rejected_by_rule": why, "spend_total_usd": round(spent, 5)},
                     indent=1))


async def main_targets(a, pool):
    """Screen, per category, enough not-yet-screened prompts to reach the target number of screened prompts
    (key "t2" = prompts with a real follow-up turn, any category)."""
    done = load_verdicts()
    targets = json.loads(a.targets)
    order = lambda p: hashlib.sha256(f"{a.seed}:{p['pid']}".encode()).hexdigest()
    pick, chosen = [], set()
    for cat, n in targets.items():
        if cat == "code_if":                      # language-neutral code requests for the IF code-language constraint
            import re
            from slices_full import CODE_LANG_RX
            from common import BH
            cands = [p for p in pool if p["category"] == "code_help" and p["words"] <= 40
                     and re.search(r"\b(write|create|make|implement|code|function|program|script|algorithm)\b",
                                   p["t1"], re.I) and not CODE_LANG_RX.search(p["t1"])
                     and BH.code_line_frac(p["t1"]) == 0 and "def " not in p["t1"] and "import " not in p["t1"]]
        else:
            cands = [p for p in pool if (p.get("t2") if cat == "t2" else p["category"] == cat)]
        have = sum(1 for p in cands if p["pid"] in done)
        todo = sorted([p for p in cands if p["pid"] not in done and p["pid"] not in chosen], key=order)
        for p in todo[:max(0, n - have)]:
            chosen.add(p["pid"])
            pick.append(p)
    batches = [pick[i:i + 25] for i in range(0, len(pick), 25)]
    print(f"screening {len(pick)} prompts in {len(batches)} calls", file=sys.stderr)
    async with DeepSeek(Path(a.out), cap_usd=a.cap, concurrency=24) as ds:
        async def one(b):
            try:
                return await screen_batch_compact(ds, b)
            except SpendCapReached:
                return []
        res = [x for r in await asyncio.gather(*[one(b) for b in batches]) for x in r]
        spent = ds.spend["usd"]
    with SCREEN.open("a", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    why = {}
    for r in res:
        if not r["ok"]:
            why[r["why"]] = why.get(r["why"], 0) + 1
    print(json.dumps({"screened": len(res), "ok": sum(r["ok"] for r in res), "rejected_by_rule": why,
                      "spend_total_usd": round(spent, 4)}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
