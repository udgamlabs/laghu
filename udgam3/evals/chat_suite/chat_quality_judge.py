"""General chat quality (PREREGISTRATION.md part 2b): pairwise judging of Laghu vs each measured rival and Gargi on the
324 held-out single-turn prompts (items/loop.jsonl), from the greedy loop-task replies (cap 1,024) of every model,
visible reply only (common.visible with each model's own special tokens). Each pair is judged in both orders.

  uv run --project evals/harness python udgam3/evals/chat_suite/chat_quality_judge.py --judge gpt-oss --provider P [--limit N]
  uv run --project evals/harness python udgam3/evals/chat_suite/chat_quality_judge.py --judge deepseek     (only after an amendment)

Judge: gpt-oss-120b (pre-registered) via Hugging Face Inference Providers (temperature 0, reasoning effort low), or
DeepSeek-V4.1-Flash if an amendment says so. Writes judge/<judge>/<opponent>.jsonl (every verdict, cached: a re-run
costs nothing) and judge/<judge>/results.json: raw and length-controlled win rate (AlpacaEval-2 style logistic fit on
tanh(len diff / sd); a tie = half a win), paired bootstrap 95% CI and two-sided p over prompts (10,000), position bias.
"""
import argparse, asyncio, gzip, json, os, random, re, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from common import visible  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--judge", choices=["gpt-oss", "deepseek"], required=True); ap.add_argument("--provider", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--cap-usd", type=float, default=3.0)
ap.add_argument("--laghu-mode", default="off")
ap.add_argument("--laghu-name", default="laghu", help="tests only: another model in Laghu's slot")
ap.add_argument("--out-tag", default="", help="tests only: a separate output folder")
a = ap.parse_args()
OPP = [("smollm2-135m-instruct", "native"), ("falcon-h1-tiny-90m-instruct", "native"), ("mobilellm-r1.5-140m", "native"), ("gargi", "off")]
OUT = os.path.join(HERE, "judge", a.judge + (f"_{a.out_tag}" if a.out_tag else "")); os.makedirs(OUT, exist_ok=True)
SYS = ("You are an impartial judge comparing two AI assistant replies to the same user message. Judge helpfulness, "
       "correctness, instruction following and clarity. Do not prefer a reply because it is longer; prefer the one that "
       "better serves the user. If both are equally good or equally bad, say tie. Answer only with JSON: "
       '{"reason": "<one sentence>", "verdict": "A" | "B" | "tie"}')
items = [json.loads(l) for l in open(os.path.join(HERE, "items", "loop.jsonl"))][: a.limit or None]


def replies(name, mode):
    meta = json.load(open(os.path.join(HERE, "raw", "full", name, "vllm", "meta.json")))
    rows = {}
    for l in gzip.open(os.path.join(HERE, "raw", "full", name, "vllm", f"loop__{mode}__greedy.jsonl.gz"), "rt"):
        r = json.loads(l); rows[r["id"]] = visible(r["raw"], meta.get("specials", []), meta.get("stop_strings", [])).strip()
    return rows


def prompt_text(it):
    return it.get("prompt") or it.get("text") or (it.get("turns") or [""])[0] or json.dumps(it.get("messages", ""))


def parse(c):
    m = re.search(r'"verdict"\s*:\s*"(A|B|tie)"', c or "", re.I)
    return {"a": "A", "b": "B", "tie": "tie"}[m.group(1).lower()] if m else None


class GptOss:
    def __init__(self):
        from huggingface_hub import AsyncInferenceClient
        assert a.provider, "--provider is required for gpt-oss (recorded with the run)"
        self.c = AsyncInferenceClient(provider=a.provider); self.sem = asyncio.Semaphore(16); self.usage = [0, 0]

    async def __call__(self, user):
        async with self.sem:
            for att in range(4):
                try:
                    r = await self.c.chat_completion(model="openai/gpt-oss-120b", temperature=0.0, max_tokens=1024,
                                                     messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                                     extra_body={"reasoning_effort": "low"})
                    self.usage[0] += r.usage.prompt_tokens; self.usage[1] += r.usage.completion_tokens
                    return r.choices[0].message.content or ""
                except Exception as e:
                    if att == 3:
                        return f"ERROR {str(e)[:200]}"
                    await asyncio.sleep(2 * (att + 1))


class DS:
    def __init__(self):
        from pathlib import Path
        sys.path.insert(0, os.path.join(ROOT, "udgam3", "sft", "gen"))
        from client import DeepSeek
        self.ds = DeepSeek(Path(OUT) / "cache", cap_usd=a.cap_usd, concurrency=16)

    async def __call__(self, user):
        r = await self.ds.chat([{"role": "system", "content": SYS}, {"role": "user", "content": user}], slice_name="chat-quality",
                               json_mode=True, max_tokens=200, temperature=0.0)
        return r["content"]


def lc_fit(y, d):
    """Logistic fit P = sigmoid(a + b * d) (IRLS, weights for ties already in y in [0, 1]); returns a, b."""
    X = np.stack([np.ones_like(d), d], 1); w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p) + 1e-9
        g = X.T @ (y - p); H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


async def main():
    L = replies(a.laghu_name, a.laghu_mode)
    judge = GptOss() if a.judge == "gpt-oss" else DS()
    if a.judge == "deepseek":
        await judge.ds.__aenter__()
    results = {"judge": a.judge, "provider": a.provider, "laghu_mode": a.laghu_mode, "n_prompts": len(items), "opponents": {}}
    for opp, mode in OPP:
        R = replies(opp, mode); path = os.path.join(OUT, f"{opp}.jsonl")
        done = {(j["id"], j["order"]): j for j in map(json.loads, open(path))} if os.path.exists(path) else {}
        todo = [(it, o) for it in items for o in (0, 1) if (it["id"], o) not in done or done[(it["id"], o)]["verdict"] is None]

        async def one(it, order):
            la, op = L.get(it["id"], ""), R.get(it["id"], "")
            first, second = (la, op) if order == 0 else (op, la)
            user = (f"[user]\n{prompt_text(it)}\n\n=== Reply A ===\n{first or '(empty)'}\n\n=== Reply B ===\n{second or '(empty)'}\n\n"
                    "Which reply is better for the user's message?")
            c = await judge(user); v = parse(c)
            if v is None:
                c = await judge(user); v = parse(c)          # one retry, then a tie (2b)
            win = {"A": "laghu" if order == 0 else opp, "B": opp if order == 0 else "laghu", "tie": "tie", None: "tie"}[v]
            done[(it["id"], order)] = {"id": it["id"], "order": order, "verdict": v, "winner": win, "len_laghu": len(la), "len_opp": len(op), "raw": c[:600]}
        t0 = time.time()
        await asyncio.gather(*(one(it, o) for it, o in todo))
        with open(path, "w") as f:
            for k in sorted(done):
                f.write(json.dumps(done[k], ensure_ascii=False) + "\n")
        J = [done[k] for k in sorted(done) if k[0] in {it["id"] for it in items}]
        ids = sorted({j["id"] for j in J})
        y = np.array([1.0 if j["winner"] == "laghu" else 0.5 if j["winner"] == "tie" else 0.0 for j in J])
        dl = np.array([j["len_laghu"] - j["len_opp"] for j in J], dtype=float); sd = dl.std() or 1.0
        d = np.tanh(dl / sd)
        lc = lambda yy, dd: float(1 / (1 + np.exp(-lc_fit(yy, dd)[0])))
        byid = {}
        for k, j in enumerate(J):
            byid.setdefault(j["id"], []).append(k)
        rng = random.Random(0); bl, br = [], []
        for _ in range(10000):
            ix = [k for i in (rng.choice(ids) for _ in ids) for k in byid[i]]
            bl.append(lc(y[ix], d[ix])); br.append(float(y[ix].mean()))
        bl, br = np.sort(bl), np.sort(br)
        res = {"mode": mode, "n_judgments": len(J), "raw_win": round(100 * float(y.mean()), 2), "raw_ci": [round(100 * br[250], 2), round(100 * br[9749], 2)],
               "lc_win": round(100 * lc(y, d), 2), "lc_ci": [round(100 * bl[250], 2), round(100 * bl[9749], 2)],
               "p_two_sided": float(min(1.0, 2 * min((bl <= 0.5).mean(), (bl >= 0.5).mean()))),
               "ties": int(sum(j["winner"] == "tie" for j in J)), "unparsed": int(sum(j["verdict"] is None for j in J)),
               "first_preferred_pct": round(100 * sum(j["verdict"] == "A" for j in J) / len(J), 1),
               "mean_len_laghu": round(float(np.mean([j["len_laghu"] for j in J]))), "mean_len_opp": round(float(np.mean([j["len_opp"] for j in J]))),
               "secs": round(time.time() - t0)}
        results["opponents"][opp] = res
        print(opp, json.dumps(res), flush=True)
    if a.judge == "gpt-oss":
        results["usage_tokens"] = {"prompt": judge.usage[0], "completion": judge.usage[1]}
    else:
        await judge.ds.__aexit__(None, None, None); results["usd"] = judge.ds.spend["usd"]
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)


asyncio.run(main())
