"""Laghu RL: training pools and frozen dev sets ($0, Mac). Plan: udgam3/rl/RL_PLAN.md.

Pools (gold answers from the datasets, never from a model):
  GSM8K train (openai/gsm8k main) and MATH train levels 1-3 with a plain-number answer (EleutherAI/hendrycks_math),
  minus every problem whose text appears in the SFT mix (any shared 10-word window) and minus the dev problems.
Dev sets (frozen, never trained on, used for the RL gates):
  dev_gsm8k.jsonl  500 GSM8K-train problems (the 200 of the SFT dev pick + 300 more), none in the SFT mix
  dev_math.jsonl   200 MATH-train L1-3 numeric problems, none in the SFT mix
Writes pool_maths.jsonl, dev_gsm8k.jsonl, dev_math.jsonl, SETS.sha256 next to this file.

  uv run --project evals/harness python udgam3/rl/build_rl_sets.py
"""
import glob, hashlib, json, os, random, re, sys

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUFFIX = "\n\nSolve it step by step, then give the final answer on the last line as: Final answer: <number>"   # chat suite's
SEED, W = 1234, 10
words = lambda t: re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()

# every user message in the SFT mix (all slices + multi-turn follow-ups), as 10-word windows
sft = set()
files = [f for f in glob.glob(os.path.join(ROOT, "udgam3", "sft", "full_v1", "*.jsonl"))
         if not os.path.basename(f).startswith(("rejected_", "requests_", "raw_", "run_requests"))]
files.append(os.path.join(ROOT, "data", "udgam2_sft_v4", "dl", "mt_v1", "gen_mt.jsonl"))
for f in files:
    for line in open(f):
        for m in json.loads(line)["messages"]:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                w = words(m["content"]); sft.update(hash(" ".join(w[i:i + W])) for i in range(max(0, len(w) - W + 1)))
in_sft = lambda q: any(hash(" ".join(w[i:i + W])) in sft for w in [words(q)] for i in range(max(0, len(w) - W + 1)))


def number(s):
    s = s.strip().replace(",", "").replace("$", "").strip()
    return s if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i + 7:j - 1] if d == 0 else None


api = HfApi()
grev = api.dataset_info("openai/gsm8k").sha; mrev = api.dataset_info("EleutherAI/hendrycks_math").sha
gsm = load_dataset("openai/gsm8k", "main", split="train", revision=grev)
G = [dict(id=f"gsm8k-train-{i}", src="gsm8k", question=r["question"], gold=r["answer"].split("####")[-1].strip().replace(",", ""))
     for i, r in enumerate(gsm) if not in_sft(r["question"])]
subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
math = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="train", revision=mrev) for s in subjects])
M = []
for i, r in enumerate(math):
    lvl = r["level"].split()[-1]
    b = boxed(r["solution"]); n = number(b) if b else None
    if lvl.isdigit() and int(lvl) <= 3 and n is not None and not in_sft(r["problem"]):
        M.append(dict(id=f"math-train-{i}", src="math", question=r["problem"], gold=n, level=int(lvl)))

old = [json.loads(l)["id"] for l in open(os.path.join(ROOT, "udgam3", "sft", "devpick", "dev_gsm8k.jsonl"))]
by_id = {x["id"]: x for x in G}
assert all(i in by_id for i in old), "SFT dev-pick GSM8K items must still be outside the SFT mix"
rng = random.Random(SEED)
rest = [x for x in G if x["id"] not in set(old)]; rng.shuffle(rest)
dev_g = [by_id[i] for i in old] + rest[:300]
rng.shuffle(M); dev_m = M[:200]
dev_ids = {x["id"] for x in dev_g + dev_m}
pool = [x for x in G + M if x["id"] not in dev_ids]; rng.shuffle(pool)


def dump(name, rows):
    with open(os.path.join(HERE, name), "w") as f:
        for x in rows:
            f.write(json.dumps(dict(x, prompt=x["question"] + SUFFIX), ensure_ascii=False) + "\n")


dump("pool_maths.jsonl", pool); dump("dev_gsm8k.jsonl", dev_g); dump("dev_math.jsonl", dev_m)
with open(os.path.join(HERE, "SETS.sha256"), "w") as f:
    for n in ("pool_maths.jsonl", "dev_gsm8k.jsonl", "dev_math.jsonl"):
        f.write(hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() + "  " + n + "\n")
    f.write(f"# openai/gsm8k {grev}; EleutherAI/hendrycks_math {mrev}\n")
print(f"GSM8K train {len(gsm)} -> {len(G)} outside SFT; MATH L1-3 numeric outside SFT {len(M)}")
print(f"dev: gsm8k {len(dev_g)}, math {len(dev_m)}; pool {len(pool)} (gsm8k {sum(x['src'] == 'gsm8k' for x in pool)}, math {sum(x['src'] == 'math' for x in pool)})")
