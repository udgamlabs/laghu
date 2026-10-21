"""Apply PREREGISTRATION.md part 2b to Laghu's chat-suite results and write CLAIMS_2B.md: the 9 rival comparisons
(maths, multi-turn, chat quality x SmolLM2-135M-Instruct, Falcon-H1-Tiny-90M-Instruct, MobileLLM-R1.5-140M) with
exact McNemar or bootstrap p-values, Holm across the 9, which categories count, the own-model bar, the loop threshold
and the release-gate result.

  python3 udgam3/evals/chat_suite/claims_2b.py --judge gpt-oss [--laghu-name laghu]
Reads scored/full/<model>/*.jsonl (score.py) and judge/<judge>/results.json (chat_quality_judge.py).
"""
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True); ap.add_argument("--laghu-name", default="laghu")
ap.add_argument("--out", default=os.path.join(HERE, "CLAIMS_2B.md"))
a = ap.parse_args()
L = a.laghu_name
RIV = [("smollm2-135m-instruct", "native"), ("falcon-h1-tiny-90m-instruct", "native"), ("mobilellm-r1.5-140m", "native")]
load = lambda n, f: {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "scored", "full", n, f)))}
checkable = {x["id"] for x in map(json.loads, open(os.path.join(HERE, "items", "mt.jsonl"))) if x.get("check")}


def mcnemar(b, c):
    """Exact two-sided McNemar p from discordant counts b (only ours right) and c (only theirs right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(ours, theirs, key):
    ids = sorted(set(ours) & set(theirs))
    o = [key(ours[i]) for i in ids]; t = [key(theirs[i]) for i in ids]
    b = sum(x and not y for x, y in zip(o, t)); c = sum(y and not x for x, y in zip(o, t))
    return 100 * sum(o) / len(ids), 100 * sum(t) / len(ids), mcnemar(b, c), len(ids)


strict = lambda r: r["ok"] and not r["trunc"]
rows = []
g_l = load(L, "gsm8k__on__greedy.jsonl")
mt_l = {k: v for k, v in load(L, "mt__off.jsonl").items() if k in checkable}
for n, m in RIV:
    ours, theirs, p, k = paired(g_l, load(n, f"gsm8k__{m}__greedy.jsonl"), strict)
    rows.append(dict(cat="Maths with thinking (GSM8K strict)", rival=n, ours=ours, theirs=theirs, p=p, n=k, higher=ours > theirs))
for n, m in RIV:
    mt_r = {k: v for k, v in load(n, f"mt__{m}.jsonl").items() if k in checkable}
    ours, theirs, p, k = paired(mt_l, mt_r, lambda r: r["right"])
    rows.append(dict(cat="Multi-turn (E1 right %)", rival=n, ours=ours, theirs=theirs, p=p, n=k, higher=ours > theirs))
J = json.load(open(os.path.join(HERE, "judge", a.judge, "results.json")))
for n, _ in RIV:
    r = J["opponents"][n]
    rows.append(dict(cat="General chat quality (LC win rate)", rival=n, ours=r["lc_win"], theirs=50.0, p=r["p_two_sided"], n=r["n_judgments"],
                     higher=r["lc_win"] > 50.0, ci=r["lc_ci"], raw=r["raw_win"]))
# Holm across the 9
order = sorted(range(len(rows)), key=lambda i: rows[i]["p"]); m = len(rows); run = 0.0
for rank, i in enumerate(order):   # Holm step-down; adjusted p-values made monotone
    run = max(run, min(1.0, (m - rank) * rows[i]["p"]))
    rows[i]["p_holm"] = run
    rows[i]["beats"] = run < 0.05 and rows[i]["higher"]
cats = {}
for r in rows:
    cats.setdefault(r["cat"], []).append(r)
won = [c for c, rs in cats.items() if all(r["beats"] for r in rs)]

# own-model bar
gargi_mt = {k: v for k, v in load("gargi", "mt__off.jsonl").items() if k in checkable}
u384 = {k: v for k, v in load("udgam-384m-chat", "mt__on.jsonl").items() if k in checkable}
own = []
for name, d in (("Gargi (off)", gargi_mt), ("Udgam-384M-Chat (on)", u384)):
    ours, theirs, p, k = paired(mt_l, d, lambda r: r["right"])
    own.append(dict(what=f"Multi-turn right % vs {name}", ours=ours, theirs=theirs, p=p, ok=not (theirs > ours and p < 0.05)))
gq = J["opponents"]["gargi"]
own.append(dict(what="Chat quality LC win rate vs Gargi", ours=gq["lc_win"], theirs=50.0, p=gq["p_two_sided"], ok=not (gq["lc_win"] < 50 and gq["p_two_sided"] < 0.05)))
gg = load("gargi", "gsm8k__on__greedy.jsonl")
ours, theirs, p, k = paired(g_l, gg, strict)
own.append(dict(what="Maths (GSM8K strict) vs Gargi (on) [reported]", ours=ours, theirs=theirs, p=p, ok=None))
loop = load(L, "loop__off__greedy.jsonl"); loop_rate = 100 * sum(r["loop"] for r in loop.values()) / len(loop)

gate2 = len(won) >= 3; gate3 = all(o["ok"] for o in own if o["ok"] is not None); loop_ok = loop_rate <= 10.0
out = ["# Part 2b claims: results (PREREGISTRATION.md part 2b)", "",
       f"Model: `{L}`; judge: {J['judge']} {J.get('provider', '')}; Holm over {m} comparisons.", "",
       "| Category | Rival | Laghu | Rival | Paired n | p | p (Holm) | Verdict |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    verdict = "**beats**" if r["beats"] else ("trails" if (not r["higher"] and r["p_holm"] < 0.05) else "matches (not significant)")
    lag = f"{r['ours']:.1f}" + (f" [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] (raw {r['raw']:.1f})" if "ci" in r else "")
    out.append(f"| {r['cat']} | {r['rival']} | {lag} | {r['theirs']:.1f} | {r['n']} | {r['p']:.3g} | {r['p_holm']:.3g} | {verdict} |")
out += ["", f"**Categories won (beats all three rivals):** {', '.join(won) if won else 'none'}.", "",
        "## Own-model bar", "| Check | Laghu | Ours | p | Holds? |", "|---|---|---|---|---|"]
out += [f"| {o['what']} | {o['ours']:.1f} | {o['theirs']:.1f} | {o['p']:.3g} | {'reported' if o['ok'] is None else ('yes' if o['ok'] else 'NO')} |" for o in own]
out += ["", f"**Loop rate** (greedy, thinking off, 324 prompts): {loop_rate:.1f}% (threshold ≤ 10%): {'OK' if loop_ok else 'ABOVE'}.", "",
        "## Release gate (section 5)",
        f"- Item 2 (beat every rival on ≥ 3 contested categories): **{'PASS' if gate2 else 'FAIL'}** ({len(won)} of 3).",
        f"- Item 3 (own-model bar): **{'PASS' if gate3 else 'FAIL'}**.",
        f"- Loop threshold: **{'PASS' if loop_ok else 'FAIL'}**.",
        f"- Items 1, 4, 5 (fresh scan, exact scope, template matrix): not run yet.",
        "", f"**Gate: {'can proceed to items 1/4/5' if (gate2 and gate3 and loop_ok) else 'FAILS: no release under the pre-registered rule; write down what would create the edge and ask the user'}.**"]
open(a.out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
