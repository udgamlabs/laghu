"""Laghu RL dev evals (greedy; no test benchmark): GSM8K dev (500, thinking on), MATH dev (200, thinking on), tool mode
(the first 200 GSM8K dev problems with the Python tool, executed), thinking off (the same 200, no tool), and replies to
the 200 SFT dev-pick chats (for the pairwise regression judge and the loop rate).

  python rl_eval.py MODEL_DIR OUT_DIR [--limit N] [--max-new 1024]
Writes OUT_DIR/summary.json and OUT_DIR/<set>.jsonl (raw replies).
"""
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import rollout as RO  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--sets", default="gsm8k_think,math_think,gsm8k_tool,gsm8k_plain,chat")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
load = lambda p: [json.loads(l) for l in open(p)][: a.limit or None]
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
G = load(os.path.join(HERE, "dev_gsm8k.jsonl")); M = load(os.path.join(HERE, "dev_math.jsonl"))
C = load(os.path.join(ROOT, "udgam3", "sft", "devpick", "dev_chat.jsonl"))
sets = {"gsm8k_think": [dict(x, mode="think") for x in G], "math_think": [dict(x, mode="think") for x in M],
        "gsm8k_tool": [dict(x, mode="tool", prompt=x["question"]) for x in G[:200]],
        "gsm8k_plain": [dict(x, mode="plain") for x in G[:200]],
        "chat": [dict(x, mode="think" if x["think"] else "plain") for x in C]}
sets = {k: v for k, v in sets.items() if k in a.sets.split(",")}
summ = {"model": a.model, "max_new": a.max_new}
for name, items in sets.items():
    t0 = time.time()
    R = RO.rollouts(m, tok, items, k=1, max_new=a.max_new, sample=False)
    with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
        for r in R:
            f.write(json.dumps({"id": r["item"]["id"], "text": r["text"], "n_model": r["n_model"], "stopped": r["stopped"],
                                "loop": r["loop"], "tool_calls": r["tool_calls"], "tool_errors": r["tool_errors"]}, ensure_ascii=False) + "\n")
    s = {"n": len(R), "cap_rate": round(100 * sum(not r["stopped"] for r in R) / len(R), 2),
         "loop_rate": round(100 * sum(r["loop"] for r in R) / len(R), 2), "mean_len": round(sum(r["n_model"] for r in R) / len(R), 1),
         "secs": round(time.time() - t0)}
    if name != "chat":
        s["acc"] = round(100 * sum(RO.correct(r["item"], r["text"]) for r in R) / len(R), 2)
    if name == "gsm8k_tool":
        s["tool_use"] = round(100 * sum(r["tool_calls"] > 0 for r in R) / len(R), 2)
        s["tool_errors"] = sum(r["tool_errors"] for r in R)
    summ[name] = s
    print(name, json.dumps(s), flush=True)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print("RL_EVAL_DONE", flush=True)
