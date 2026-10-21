"""Build the frozen dev sets of DEV_RULE.md ($0, Mac): dev_chat.jsonl (200 SFT-val conversations) and dev_gsm8k.jsonl
(200 GSM8K train problems), plus DEV_SETS.sha256.

  uv run --project evals/harness python udgam3/sft/devpick/build_dev.py
"""
import collections, glob, hashlib, json, os, random, re, sys

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam3", "evals", "chat_suite"))
from common import SUFFIX  # noqa: E402  (the chat suite's maths prompt suffix)

N, SEED, SKIP = 200, 1234, {"identity", "python_tool"}
val = pq.read_table(os.path.join(ROOT, "data", "udgam3_sft_v1", "val.parquet"), columns=["id", "src", "think"]).to_pylist()
val = [r for r in val if r["src"] not in SKIP]

# source rows by id: the verified slices + the multi-turn follow-ups (as render_v1.py reads them)
src_rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, "udgam3", "sft", "full_v1", "*.jsonl"))):
    b = os.path.basename(f)
    if not b.startswith(("rejected_", "requests_", "raw_", "run_requests")):
        for line in open(f):
            r = json.loads(line); src_rows[r["id"]] = r
for line in open(os.path.join(ROOT, "data", "udgam2_sft_v4", "dl", "mt_v1", "gen_mt.jsonl")):
    r = json.loads(line); src_rows[r["id"]] = r

W = 10
words = lambda t: re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()
sft_windows = set()
for row in src_rows.values():   # every SFT row, train and val (for GSM8K)
    for m in row["messages"]:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            w = words(m["content"])
            sft_windows.update(hash(" ".join(w[i:i + W])) for i in range(max(0, len(w) - W + 1)))
def in_sft(q):
    w = words(q)
    return any(hash(" ".join(w[i:i + W])) in sft_windows for i in range(max(0, len(w) - W + 1)))

# val conversations whose last user message mostly appears in a TRAIN row (e.g. a maths row in val whose thinking twin is
# in train: render_v1 splits "-think" and "-nothink" source rows separately) are left out of the chat dev set
tr_ids = pq.read_table(os.path.join(ROOT, "data", "udgam3_sft_v1", "train.parquet"), columns=["id"]).column("id").to_pylist()
tr_windows = set()
for i in tr_ids:
    row = src_rows.get(i) or src_rows.get(i[:-len("-nothink")])
    for m in (row or {}).get("messages", []):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            w = words(m["content"])
            tr_windows.update(hash(" ".join(w[k:k + W])) for k in range(max(0, len(w) - W + 1)))
def in_train(r):
    base = r["id"] if r["id"] in src_rows else r["id"][:-len("-nothink")]
    last = [m for m in src_rows[base]["messages"] if m["role"] == "user"][-1]["content"]
    w = words(last); win = [hash(" ".join(w[k:k + W])) for k in range(max(0, len(w) - W + 1))]
    # >= 50% of its 10-word windows in train = the same question or document (twins, long-doc rows); shared boilerplate
    # (e.g. instruction-following format phrases) stays below that and is kept
    return bool(win) and sum(h in tr_windows for h in win) >= 0.5 * len(win)
n_val = len(val); val = [r for r in val if not in_train(r)]
print(f"val conversations: {n_val} -> {len(val)} after dropping prompts also in train")

# stratified sample, proportional to the val split
rng = random.Random(SEED)
by = collections.defaultdict(list)
for r in sorted(val, key=lambda r: r["id"]):
    by[r["src"]].append(r)
quota = {s: round(N * len(v) / len(val)) for s, v in by.items()}
while sum(quota.values()) != N:   # fix rounding on the largest source
    big = max(quota, key=lambda s: len(by[s])); quota[big] += 1 if sum(quota.values()) < N else -1
picked = [x for s in sorted(by) for x in rng.sample(by[s], quota[s])]


def prompt_of(r):
    base = r["id"] if r["id"] in src_rows else r["id"][:-len("-nothink")]   # render_v1's thinking-off twins add "-nothink"
    row = src_rows[base]
    msgs = [dict(m) for m in row["messages"]]
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    msgs = msgs[:last_user + 1]
    for m in msgs:
        m.pop("reasoning_content", None)
        if isinstance(m.get("content"), str):
            m["content"] = m["content"].replace("{MODEL_NAME}", "Laghu")
    return {"id": r["id"], "src": r["src"], "think": bool(r["think"]), "messages": msgs, "tools": row.get("tools")}


chat = [prompt_of(r) for r in picked]
rev = HfApi().dataset_info("openai/gsm8k").sha
ds = load_dataset("openai/gsm8k", "main", split="train", revision=rev)
# leave out every GSM8K train problem that appears in the SFT mix (both runs trained on those, the 2-pass run twice):
# any 10-word window of a GSM8K question found in an SFT user message
order = list(range(len(ds))); random.Random(SEED).shuffle(order)
idx, n_skip = [], 0
for i in order:
    if len(idx) == N:
        break
    if in_sft(ds[i]["question"]):
        n_skip += 1; continue
    idx.append(i)
idx.sort()
gsm = [{"id": f"gsm8k-train-{i}", "question": ds[i]["question"], "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", ""),
        "messages": [{"role": "user", "content": ds[i]["question"] + SUFFIX}], "think": True} for i in idx]

with open(os.path.join(HERE, "dev_chat.jsonl"), "w") as f:
    for x in chat: f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open(os.path.join(HERE, "dev_gsm8k.jsonl"), "w") as f:
    for x in gsm: f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open(os.path.join(HERE, "DEV_SETS.sha256"), "w") as f:
    for n in ("dev_chat.jsonl", "dev_gsm8k.jsonl"):
        f.write(hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() + "  " + n + "\n")
    f.write(f"# openai/gsm8k revision {rev}; {n_skip} sampled problems skipped because they appear in the SFT mix\n")
print("chat", len(chat), dict(collections.Counter(x["src"] for x in chat)), "think", sum(x["think"] for x in chat))
print("gsm8k", len(gsm), "rev", rev[:10], "skipped (in the SFT mix):", n_skip)
