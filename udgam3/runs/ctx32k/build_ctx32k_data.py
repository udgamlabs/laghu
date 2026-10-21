"""Udgam-3 32K-stretch training data (plan: udgam3/runs/ctx32k/PLAN.md). $0, runs on the Mac (CPU).

Mix (~1.0B tokens):
  - 42% long pool: rows whose src starts with "long_" in YOUR_HF_NAMESPACE/udgam2-ctx32k-data (batch 1) and -data2 (batch 2):
    books, papers, gov text, repo-level code. Already licence-checked, cleaned and split into pieces <= 32,000 tokens.
  - ~8% modern long docs: long_web_hq, long_code, long_wiki from -data3. long_math (Nemotron-CC-Math, NVIDIA
    agreement) and long_b2_* (re-used batch-2 rows) are excluded.
  - 50% replay: rows of the v3 cool-down pool the main run never read (YOUR_HF_NAMESPACE/laghu-pretrain-v3
    shards/phaseB/shard_00387..00398; the loader stopped in shard_00386, pq_idx 63; shard_00399 is the val shard).
  Gargi-era replay rows (v2b-derived) are not used.
Filters on every row: 13-gram decontamination with the v3 index (data/decontam/index_13gram_v3.npz, rebuilt with
udgam/decontam.py, same rule as the v3 build) and a credential filter (live-looking API keys or private keys).
Output (data/udgam3_ctx32k): train/shard_00000.parquet ... + train/shard_zzzzz_val.parquet (= v3 val shard, so short
val bpb stays comparable), val_long/val_long.parquet (held-out docs >= 33K tokens from the three builds, checked for no
overlap with train), report.json.

  uv run --project evals/harness python udgam3/runs/ctx32k/build_ctx32k_data.py [--total 1.0e9] [--dry]
"""
import argparse, collections, hashlib, json, os, random, re, sys, time

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "udgam"))
import decontam  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--total", type=float, default=1.0e9); ap.add_argument("--long-share", type=float, default=0.42)
ap.add_argument("--replay-share", type=float, default=0.50); ap.add_argument("--out", default=os.path.join(ROOT, "data", "udgam3_ctx32k"))
ap.add_argument("--index", default=os.path.join(ROOT, "data", "decontam", "index_13gram_v3.npz"))
ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--dry", action="store_true", help="1% of the budget, for tests")
a = ap.parse_args()
scale = 0.01 if a.dry else 1.0
T_LONG, T_REPLAY = a.total * a.long_share * scale, a.total * a.replay_share * scale
GARGI = ["YOUR_HF_NAMESPACE/udgam2-ctx32k-data", "YOUR_HF_NAMESPACE/udgam2-ctx32k-data2", "YOUR_HF_NAMESPACE/udgam2-ctx32k-data3"]
MODERN = {"long_web_hq", "long_code", "long_wiki"}
EXCLUDE = re.compile(r"^long_(math|b2_)")
REPLAY_SHARDS = [f"shards/phaseB/shard_{i:05d}.parquet" for i in range(387, 399)]
VAL_SHARD = "shards/phaseB/shard_00399.parquet"
SECRET = re.compile(r"infura\.io/v3/[0-9a-f]{32}|\bAKIA[0-9A-Z]{16}\b|\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{32,}|\bghp_[A-Za-z0-9]{36}\b|"
                    r"\bgithub_pat_[A-Za-z0-9_]{60,}|\bhf_[A-Za-z0-9]{34}\b|\bxox[abprs]-[A-Za-z0-9-]{20,}|\bsk_live_[A-Za-z0-9]{24,}|"
                    r"\bAIza[0-9A-Za-z_-]{35}\b|-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----")
rng = random.Random(a.seed)
ix = decontam.Index(a.index)
report = {"args": vars(a), "drops": collections.Counter(), "tokens": collections.Counter(), "rows": collections.Counter()}
t0 = time.time()


def ok(text, kind):
    if SECRET.search(text):
        report["drops"][f"secret_{kind}"] += 1; return False
    if ix.is_contaminated(text)[0]:  # returns (bad, hits, eval names)
        report["drops"][f"decontam_{kind}"] += 1; return False
    return True


# 1. long rows from Gargi's builds
long_rows, modern_rows, seen = [], [], set()
for rid in GARGI:
    idx = pq.read_table(hf_hub_download(rid, "train_index.parquet", repo_type="dataset")).to_pylist()
    want = collections.defaultdict(dict)
    for r in idx:
        s = r["src"]
        if s.startswith("long_") and not EXCLUDE.match(s):
            want[r["shard"]][r["row"]] = (s, r["ntok_v1"])
    for shard, rows in sorted(want.items())[: 2 if a.dry else None]:
        name = shard if str(shard).endswith(".parquet") else f"train/shard_{int(shard):05d}.parquet"
        t = pq.read_table(hf_hub_download(rid, name if "/" in name else f"train/{name}", repo_type="dataset"), columns=["text"])
        texts = t.column("text").to_pylist()
        for row, (s, n) in rows.items():
            text = texts[row]
            h = hashlib.sha1(text[:4096].encode()).hexdigest()
            if h in seen:
                report["drops"]["dup_long"] += 1; continue
            seen.add(h)
            if ok(text, "long"):
                (modern_rows if s in MODERN else long_rows).append((text, s, n))
    print(f"[{time.time() - t0:.0f}s] {rid}: long {len(long_rows)} modern {len(modern_rows)}", flush=True)

# sample the long pool to its budget, proportionally across sources (seeded)
rng.shuffle(long_rows)
long_pick, tok = [], 0
for r in long_rows:
    if tok >= T_LONG: break
    long_pick.append(r); tok += r[2]
modern_pick = modern_rows if not a.dry else modern_rows[: max(1, int(len(modern_rows) * scale))]
for text, s, n in long_pick + modern_pick:
    report["tokens"][s] += n; report["rows"][s] += 1

# 2. replay from the unread v3 cool-down shards; tokens counted with the v3 tokenizer
from tokenizers import Tokenizer  # noqa: E402
tk = Tokenizer.from_file(hf_hub_download("YOUR_HF_NAMESPACE/laghu-pretrain-v3", "tokenizer_v1/tokenizer.json", repo_type="dataset"))
replay, rtok = [], 0
for shard in REPLAY_SHARDS:
    if rtok >= T_REPLAY: break
    texts = pq.read_table(hf_hub_download("YOUR_HF_NAMESPACE/laghu-pretrain-v3", shard, repo_type="dataset"), columns=["text"]).column("text").to_pylist()
    for i in range(0, len(texts), 2048):
        batch = texts[i:i + 2048]
        for text, enc in zip(batch, tk.encode_batch(batch, add_special_tokens=False)):
            if rtok >= T_REPLAY: break
            if SECRET.search(text):
                report["drops"]["secret_replay"] += 1; continue
            replay.append((text, "replay_v3B", len(enc.ids))); rtok += len(enc.ids)
    print(f"[{time.time() - t0:.0f}s] {shard}: replay tokens {rtok / 1e6:.1f}M", flush=True)
report["tokens"]["replay_v3B"] = rtok; report["rows"]["replay_v3B"] = len(replay)

# 3. shuffle and write
allrows = long_pick + modern_pick + replay
rng.shuffle(allrows)
os.makedirs(os.path.join(a.out, "train"), exist_ok=True); os.makedirs(os.path.join(a.out, "val_long"), exist_ok=True)
per_shard = 15_000_000 * (1 if not a.dry else 1)
cur, cur_tok, k, index_rows = [], 0, 0, []
def flush():
    global cur, cur_tok, k
    if not cur: return
    pq.write_table(pa.table({"text": [r[0] for r in cur]}), os.path.join(a.out, "train", f"shard_{k:05d}.parquet"), row_group_size=1024, compression="zstd")
    index_rows.extend({"shard": k, "row": j, "src": r[1], "ntok": r[2]} for j, r in enumerate(cur))
    k += 1; cur, cur_tok = [], 0
for r in allrows:
    cur.append(r); cur_tok += r[2]
    if cur_tok >= per_shard: flush()
flush()
pq.write_table(pa.Table.from_pylist(index_rows), os.path.join(a.out, "train_index.parquet"))
# short val shard (same as the main run's) copied last, so val bpb stays comparable
vt = pq.read_table(hf_hub_download("YOUR_HF_NAMESPACE/laghu-pretrain-v3", VAL_SHARD, repo_type="dataset"))
pq.write_table(vt, os.path.join(a.out, "train", "shard_zzzzz_val.parquet"), row_group_size=1024, compression="zstd")
# held-out long docs: union of the three builds' val_long, minus anything that also appears in train
train_h = {hashlib.sha1(r[0][:4096].encode()).hexdigest() for r in allrows}
vl, vl_drop = [], 0
for rid in GARGI:
    for r in pq.read_table(hf_hub_download(rid, "val_long/val_long.parquet", repo_type="dataset")).to_pylist():
        if hashlib.sha1(r["text"][:4096].encode()).hexdigest() in train_h or not ok(r["text"], "val_long"):
            vl_drop += 1; continue
        vl.append(r)
pq.write_table(pa.Table.from_pylist(vl), os.path.join(a.out, "val_long", "val_long.parquet"))
total = sum(r[2] for r in allrows)
report.update({"train_shards": k, "total_tokens": total, "long_tokens": sum(r[2] for r in long_pick), "modern_tokens": sum(r[2] for r in modern_pick),
               "replay_tokens": rtok, "val_long_docs": len(vl), "val_long_dropped": vl_drop, "wall_seconds": round(time.time() - t0)})
report["shares"] = {"long": round(report["long_tokens"] / total, 3), "modern": round(report["modern_tokens"] / total, 3), "replay": round(rtok / total, 3)}
json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=lambda o: dict(o))
print(json.dumps({k_: report[k_] for k_ in ("total_tokens", "shares", "train_shards", "val_long_docs", "drops", "wall_seconds")}, indent=1, default=dict))
