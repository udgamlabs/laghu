"""Pack text documents into nanochat-style parquet shards ('text' column, several row groups each).
The LAST shard is the validation split (nanochat convention).
Usage: python make_shards.py OUT_DIR --sources a.jsonl:weight b.jsonl:weight ... [--docs-per-shard N] [--rows-per-group N] [--val-docs N]"""
import json, random, argparse, os
import pyarrow as pa, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--sources", nargs="+", required=True)
ap.add_argument("--docs-per-shard", type=int, default=20000); ap.add_argument("--rows-per-group", type=int, default=1024)
ap.add_argument("--val-docs", type=int, default=2000); ap.add_argument("--max-docs", type=int, default=10**9); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = random.Random(a.seed)
docs = []
for spec in a.sources:
    path, w = (spec.rsplit(":", 1) + ["1"])[:2]
    src = [json.loads(l)["text"] for l in open(path)]
    k = int(len(src) * float(w)); docs += [rng.choice(src) for _ in range(k)] if float(w) > 1 else src[:k]
rng.shuffle(docs); docs = docs[:a.max_docs]
val, train = docs[:a.val_docs], docs[a.val_docs:]
shards = [train[i:i + a.docs_per_shard] for i in range(0, len(train), a.docs_per_shard)] + [val]
for i, sh in enumerate(shards):
    pq.write_table(pa.table({"text": sh}), f"{a.out}/shard_{i:05d}.parquet", row_group_size=a.rows_per_group)
print(f"{len(train)} train docs in {len(shards)-1} shards, {len(val)} val docs -> {a.out}")
