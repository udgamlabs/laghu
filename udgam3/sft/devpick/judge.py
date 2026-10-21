"""Score the dev pick (DEV_RULE.md) on the Mac: GSM8K-train accuracy per model, the batched-vs-unbatched scorer check,
and the pairwise judge (DeepSeek-V4.1-Flash, thinking off, temperature 0, both orders, spend cap $0.50). Writes
DEV_RESULTS.md and judgments.jsonl.

  uv run --project evals/harness python udgam3/sft/devpick/judge.py --gen GEN_DIR [--dry]
"""
import argparse, asyncio, collections, json, os, random, re, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import importlib.util
_spec = importlib.util.spec_from_file_location("cs_common", os.path.join(ROOT, "udgam3", "evals", "chat_suite", "common.py"))
cs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cs)   # the chat suite's rules (a different common.py sits in sft/gen)
gsm8k_ok, split_thinking = cs.gsm8k_ok, cs.split_thinking
sys.path.insert(0, os.path.join(ROOT, "udgam3", "sft", "gen"))
from client import DeepSeek  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--dry", action="store_true")
ap.add_argument("--cap", type=float, default=0.50)
a = ap.parse_args()
A, B = "ep1", "ep2"   # 1 pass, 2 passes
SPECIALS = ["<|turn>", "<turn|>", "<eos>", "<bos>", "<pad>", "<|channel>", "<channel|>", "<|think|>", "<|tool_call>", "<tool_call|>",
            "<|tool_response>", "<tool_response|>", "<|tool>", "<tool|>", '<|"|>']
load = lambda p: [json.loads(l) for l in open(p)]


def visible(raw):
    ans = split_thinking(raw)[1]
    for s in SPECIALS:
        ans = ans.replace(s, "")
    return ans.strip()


chat = {x["id"]: x for x in load(os.path.join(HERE, "dev_chat.jsonl"))}
gsm = {x["id"]: x for x in load(os.path.join(HERE, "dev_gsm8k.jsonl"))}
G = {m: {t: {r["id"]: r for r in load(os.path.join(a.gen, f"{m}_{t}.jsonl"))} for t in ("chat", "gsm", "gsm_ref")} for m in (A, B)}

# --- maths and the scorer check
acc, ref_agree, stats = {}, {}, {}
for m in (A, B):
    ok = {i: gsm8k_ok(visible(r["raw"]), gsm[i]["gold"]) for i, r in G[m]["gsm"].items()}
    acc[m] = 100 * sum(ok.values()) / len(ok)
    ref_agree[m] = sum(ok[i] == gsm8k_ok(visible(r["raw"]), gsm[i]["gold"]) for i, r in G[m]["gsm_ref"].items())
    stats[m] = {t: {"hit_cap": sum(r["hit_cap"] for r in G[m][t].values()), "mean_new": round(sum(r["n_new"] for r in G[m][t].values()) / len(G[m][t]))}
                for t in ("chat", "gsm")}
    stats[m]["chat_visible_chars"] = round(sum(len(visible(r["raw"])) for r in G[m]["chat"].values()) / len(G[m]["chat"]))

# --- pairwise judge
SYS = ("You are an impartial judge comparing two AI assistant replies to the same conversation. Judge helpfulness, "
       "correctness, instruction following and clarity. Do not prefer a reply because it is longer; prefer the one that "
       "better serves the user. If both are equally good or equally bad, say tie. Answer only with JSON: "
       '{"reason": "<one sentence>", "verdict": "A" | "B" | "tie"}')


def convo_text(it):
    parts = []
    if it.get("tools"):
        parts.append("[Tools available to the assistant]\n" + json.dumps(it["tools"], ensure_ascii=False)[:4000])
    for msg in it["messages"]:
        c = msg.get("content") if isinstance(msg.get("content"), str) else json.dumps(msg.get("content"), ensure_ascii=False)
        if msg.get("tool_calls"):
            c = (c or "") + "\n[tool call] " + json.dumps(msg["tool_calls"], ensure_ascii=False)
        parts.append(f"[{msg['role']}]\n{c}")
    return "\n\n".join(parts)


def parse(content):
    m = re.search(r'"verdict"\s*:\s*"(A|B|tie)"', content or "", re.I)
    return {"a": "A", "b": "B"}.get(m.group(1).lower(), "tie") if m else "tie"


async def judge_all():
    out = []
    async with DeepSeek(Path(HERE) / "judge_cache", cap_usd=a.cap, concurrency=16) as ds:
        async def one(i, order):
            it = chat[i]
            ra, rb = visible(G[A]["chat"][i]["raw"]), visible(G[B]["chat"][i]["raw"])
            first, second = (ra, rb) if order == 0 else (rb, ra)
            user = (f"{convo_text(it)}\n\n=== Reply A ===\n{first or '(empty)'}\n\n=== Reply B ===\n{second or '(empty)'}\n\n"
                    "Which reply is better for the user's last message?")
            r = await ds.chat([{"role": "system", "content": SYS}, {"role": "user", "content": user}], slice_name="devpick",
                              json_mode=True, max_tokens=200, temperature=0.0)
            v = parse(r["content"])
            winner = {"A": (A if order == 0 else B), "B": (B if order == 0 else A), "tie": "tie"}[v]
            out.append({"id": i, "src": it["src"], "order": order, "verdict": v, "winner": winner, "raw": r["content"]})
        await asyncio.gather(*(one(i, o) for i in chat for o in (0, 1)))
        spend = ds.spend["usd"]
    return out, spend


if a.dry:
    print("acc", acc, "ref_agree", ref_agree, stats); sys.exit()
J, spend = asyncio.run(judge_all())
with open(os.path.join(HERE, "judgments.jsonl"), "w") as f:
    for j in sorted(J, key=lambda j: (j["id"], j["order"])):
        f.write(json.dumps(j, ensure_ascii=False) + "\n")
score = collections.defaultdict(list)
for j in J:
    score[j["id"]].append(1.0 if j["winner"] == B else 0.5 if j["winner"] == "tie" else 0.0)
per = {i: sum(v) / len(v) for i, v in score.items()}
wr = 100 * sum(per.values()) / len(per)
rng = random.Random(0); ids = list(per)
boots = sorted(100 * sum(per[rng.choice(ids)] for _ in ids) / len(ids) for _ in range(10000))
lo, hi = boots[250], boots[9749]
by_src = collections.defaultdict(list)
for i, s in per.items():
    by_src[chat[i]["src"]].append(s)
c1, c2 = wr >= 55.0, acc[B] >= acc[A] - 2.0
pick = "2 passes (laghu-sft-v1-ep2)" if (c1 and c2) else "1 pass (laghu-sft-v1-ep1)"
order_bias = 100 * sum(j["verdict"] == "A" for j in J) / len(J)
L = ["# Laghu SFT dev pick: results", "", f"Rule: `DEV_RULE.md` (written before measuring). Judge spend ${spend:.3f}.", "",
     "| | 1 pass | 2 passes | Rule |", "|---|---|---|---|",
     f"| Pairwise win rate of 2 passes vs 1 pass (200 dev conversations, both orders) | | **{wr:.1f}%** (95% CI {lo:.1f}-{hi:.1f}) | ≥ 55%: {'PASS' if c1 else 'FAIL'} |",
     f"| GSM8K-train accuracy (200, thinking on) | {acc[A]:.1f}% | {acc[B]:.1f}% | ≥ 1-pass − 2.0: {'PASS' if c2 else 'FAIL'} |",
     f"| Scorer check: batched = unbatched verdicts (of 20) | {ref_agree[A]}/20 | {ref_agree[B]}/20 | ≥ 18 each: {'PASS' if min(ref_agree.values()) >= 18 else 'FAIL'} |",
     "", f"**Pick: {pick}.**", "",
     "Not deciding (reported): per-source win rate of 2 passes: " + ", ".join(f"{s} {100 * sum(v) / len(v):.0f}% (n={len(v)})" for s, v in sorted(by_src.items())) + ".",
     f"Judge position bias: first reply preferred in {order_bias:.0f}% of verdicts. Lengths and caps: {json.dumps(stats)}.", ""]
open(os.path.join(HERE, "DEV_RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
