"""Final regression check, part 1 (RL_PLAN.md): pairwise judge of the kept RL model vs the SFT model on the 200 dev
chats, from rl_eval replies made with identical settings (greedy, 1,024 new tokens): DeepSeek-V4.1-Flash, thinking off,
temperature 0, both orders, spend cap $0.50 (the dev-pick judge's prompt). Pass: RL win rate >= 45%; dev chat loop rate
<= SFT + 1 point.

  uv run --project evals/harness python udgam3/rl/final_judge.py STAGE        # e.g. r3 (reads YOUR_HF_NAMESPACE/laghu-rl)
"""
import asyncio, collections, json, os, random, re, sys
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "udgam3", "sft", "gen"))
import rollout as RO  # noqa: E402
from client import DeepSeek  # noqa: E402

STAGE = sys.argv[1]; REPO = "YOUR_HF_NAMESPACE/laghu-rl"
load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
sft = load(hf_hub_download(REPO, "dry/eval_sft/chat.jsonl")); rl = load(hf_hub_download(REPO, f"{STAGE}/eval/chat.jsonl"))
chat = {x["id"]: x for x in map(json.loads, open(os.path.join(ROOT, "udgam3", "sft", "devpick", "dev_chat.jsonl")))}
SYS = ("You are an impartial judge comparing two AI assistant replies to the same conversation. Judge helpfulness, "
       "correctness, instruction following and clarity. Do not prefer a reply because it is longer; prefer the one that "
       "better serves the user. If both are equally good or equally bad, say tie. Answer only with JSON: "
       '{"reason": "<one sentence>", "verdict": "A" | "B" | "tie"}')


def convo(it):
    parts = (["[Tools available to the assistant]\n" + json.dumps(it["tools"], ensure_ascii=False)[:4000]] if it.get("tools") else [])
    for m in it["messages"]:
        c = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"), ensure_ascii=False)
        if m.get("tool_calls"):
            c = (c or "") + "\n[tool call] " + json.dumps(m["tool_calls"], ensure_ascii=False)
        parts.append(f"[{m['role']}]\n{c}")
    return "\n\n".join(parts)


async def main():
    J = []
    async with DeepSeek(Path(HERE) / "judge_cache", cap_usd=0.50, concurrency=16) as ds:
        async def one(i, order):
            a, b = RO.visible(sft[i]["text"]), RO.visible(rl[i]["text"])
            first, second = (a, b) if order == 0 else (b, a)
            user = f"{convo(chat[i])}\n\n=== Reply A ===\n{first or '(empty)'}\n\n=== Reply B ===\n{second or '(empty)'}\n\nWhich reply is better for the user's last message?"
            r = await ds.chat([{"role": "system", "content": SYS}, {"role": "user", "content": user}], slice_name=f"final-{STAGE}",
                              json_mode=True, max_tokens=200, temperature=0.0)
            m = re.search(r'"verdict"\s*:\s*"(A|B|tie)"', r["content"] or "", re.I)
            v = {"a": "A", "b": "B"}.get(m.group(1).lower(), "tie") if m else "tie"
            J.append({"id": i, "order": order, "verdict": v, "winner": {"A": "sft" if order == 0 else "rl", "B": "rl" if order == 0 else "sft", "tie": "tie"}[v],
                      "raw": r["content"]})
        await asyncio.gather(*(one(i, o) for i in chat for o in (0, 1)))
        return J, ds.spend["usd"]


J, usd = asyncio.run(main())
per = collections.defaultdict(list)
for j in J:
    per[j["id"]].append(1.0 if j["winner"] == "rl" else 0.5 if j["winner"] == "tie" else 0.0)
sc = {i: sum(v) / len(v) for i, v in per.items()}; wr = 100 * sum(sc.values()) / len(sc)
rng = random.Random(0); ids = list(sc)
boots = sorted(100 * sum(sc[rng.choice(ids)] for _ in ids) / len(ids) for _ in range(10000))
loop = lambda d: 100 * sum(r["loop"] for r in d.values()) / len(d)
res = {"stage": STAGE, "win_rate_rl_vs_sft": round(wr, 1), "ci95": [round(boots[250], 1), round(boots[9749], 1)],
       "loop_rate_sft": round(loop(sft), 1), "loop_rate_rl": round(loop(rl), 1), "judge_usd": round(usd, 4),
       "pass_win": wr >= 45.0, "pass_loop": loop(rl) <= loop(sft) + 1.0}
json.dump(res, open(os.path.join(HERE, f"FINAL_JUDGE_{STAGE}.json"), "w"), indent=1)
with open(os.path.join(HERE, f"final_judgments_{STAGE}.jsonl"), "w") as f:
    for j in sorted(J, key=lambda j: (j["id"], j["order"])):
        f.write(json.dumps(j, ensure_ascii=False) + "\n")
print(json.dumps(res, indent=1))
