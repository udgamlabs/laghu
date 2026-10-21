#!/usr/bin/env python
"""Validate the output of build_long32k.py (layout + invariants). Exit code 1 on any failed check.

    $PY udgam2/ctx32k/test_long32k.py data/udgam2_ctx32k [--dry] [--show 2]
    $PY udgam2/ctx32k/test_long32k.py data/udgam2_ctx32k_t2 --compare-to data/udgam2_ctx32k   # top-up batch

Checks
  layout      train/shard_NNNNN.parquet..., last sorted parquet = shard_zzzzz_val.parquet (nanochat: last = val),
              val_long/val_long.parquet, train_index.parquet, report.json
  format      every train shard: exactly one column "text" (string), ZSTD, row groups of 1024 rows (last may be
              smaller), same as the v2b shards; shard sizes 100-250 MB (full build only)
  tokens      every train row is re-tokenised with tokenizer_v1: counts == train_index, no row > 32,000 tokens,
              no long piece < 4,096 tokens, long share within +-5 points of args.long_share (default 50%),
              long pool within +-15% of args.long_target (full build), totals == report.json
  content     no special-token literals in long rows, no U+FFFD runs, decoded samples per source (--show)
  val_long    columns text/src/ntok_v1, every doc >= 33,000 tokens (re-tokenised), ~300 docs (full build);
              no held-out doc overlaps the training long rows (sampled word 10-grams, see below);
              a copied val_long (--copy-val-long) must match its source's sha256; val shard sha256 == report
  --compare-to  earlier batch (repeatable): no train file of this batch was a train or held-out file there, held-out
              files are the same, replay row ranges are disjoint, val shard and val_long sha256 are identical, and no
              train row text (exact hash) occurs in its train shards. Repair batch (report args.mix == "repair3", built
              by build_repair_r3.py): the batch-2 long-pool reuse (src long_b2_*) is the one allowed overlap -- against
              the batch named in report reuse.from_dir its files must be a subset of that batch's train files and every
              reuse row must be found there verbatim; every other row / file must still be disjoint.
  repair3     pool token shares (modern / reuse / replay) within +-3 / +-2 / +-3 points of the targets, modern source
              shares within +-5 points of 55/15/15/15, full word 10-gram check of every modern row vs val_long
              (< 2% of its distinct 10-grams), >= 381 x 524,288 tokens (full build)
"""
import argparse, collections, glob, hashlib, json, os, random, sys, zlib
import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam"))
import decontam  # noqa: E402

SPECIAL = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<|turn>", "<turn|>", "<|tool>", "<tool|>", "<|tool_call>",
           "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>", "<|channel>", "<channel|>"]
FAILS = []


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def row_hash(t):
    return int.from_bytes(hashlib.blake2b(t.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def compare_to(out, rep, prev_dir, my_hash=None, my_src=None):
    """Zero overlap with an earlier batch: train files vs its train + held-out files, replay rows, same eval files.
    Repair batch: the reuse pool (b2_* sources, rows long_b2_*) must overlap exactly the batch it was drawn from."""
    prev = json.load(open(f"{prev_dir}/report.json"))
    reuse_here = rep.get("args", {}).get("mix") == "repair3" and \
        os.path.abspath(rep.get("reuse", {}).get("from_dir", "")) == os.path.abspath(prev_dir)

    def files(r, key):
        if key == "train":
            return {(n, f["file"]) for n, fs in r["files_used"].items() for f in fs}
        return {(n, f) for n, fs in r["heldout_files"].items() for f in fs}
    # compare per (source family = repo) by file name: the same file under two source names is still the same file
    fam = lambda s: {f for _, f in s}
    tr, ptr, pev, ev = files(rep, "train"), files(prev, "train"), files(prev, "eval"), files(rep, "eval")
    reuse_files = {(n, f) for n, f in tr if n.startswith("b2_")}
    tr = tr - reuse_files
    o1, o2 = fam(tr) & fam(ptr), fam(tr) & fam(pev)
    print(f"compare {prev_dir}: this batch reads {len(fam(tr))} train files (+{len(fam(reuse_files))} reuse-pool files); "
          f"earlier batch: {len(fam(ptr))} train, {len(fam(pev))} held-out")
    check(not o1, f"train files shared with the earlier batch's train files: {sorted(o1)}")
    if reuse_here:
        check(fam(reuse_files) <= fam(ptr), f"reuse-pool files ({len(fam(reuse_files))}) are all train files of the reuse batch")
    elif reuse_files:
        o3 = fam(reuse_files) & (fam(ptr) | fam(pev))
        check(not o3, f"reuse-pool files shared with this (non-reuse) earlier batch: {sorted(o3)[:5]}")
    check(not o2, f"train files that were the earlier batch's held-out files: {sorted(o2)}")
    check(fam(ev) == fam(pev), f"held-out files identical to the earlier batch ({len(fam(ev))} files)")
    rows = collections.defaultdict(list)
    for r in (prev, rep):
        for sh in r["replay"]["shards"]:
            rows[sh["shard"]].append((sh.get("row_start", 0), sh.get("row_start", 0) + sh["rows_taken"]))
    clash = [(k, v) for k, v in rows.items() if len(v) > 1 and max(a for a, _ in v) < min(b for _, b in v)]
    check(not clash, f"replay row ranges disjoint from the earlier batch ({dict(rows)})")
    a, b = sha256_file(f"{out}/train/shard_zzzzz_val.parquet"), sha256_file(f"{prev_dir}/train/shard_zzzzz_val.parquet")
    check(a == b, f"val shard sha256 == earlier batch ({a[:16]}... vs {b[:16]}...)")
    a, b = sha256_file(f"{out}/val_long/val_long.parquet"), sha256_file(f"{prev_dir}/val_long/val_long.parquet")
    check(a == b, f"val_long sha256 == earlier batch ({a[:16]}... vs {b[:16]}...)")
    if my_hash is not None:                              # exact row-text overlap with the earlier batch's train rows
        ph = []
        for f in sorted(glob.glob(f"{prev_dir}/train/shard_[0-9]*.parquet")):
            ph += [row_hash(t) for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()]
        ph = np.unique(np.array(ph, dtype=np.uint64))
        found = np.isin(my_hash, ph)
        is_reuse = np.array([s.startswith("long_b2_") for s in my_src])
        n_other = int((found & ~is_reuse).sum())
        by_src = collections.Counter(s for s, f_, r in zip(my_src, found, is_reuse) if f_ and not r)
        check(n_other == 0, f"non-reuse train rows found verbatim in the earlier batch: {n_other} {dict(by_src.most_common(5))}")
        if reuse_here:
            check(bool(found[is_reuse].all()), f"all {int(is_reuse.sum())} reuse rows found verbatim in the reuse batch "
                  f"({int(found[is_reuse].sum())} found)")
        else:
            check(not found[is_reuse].any(), f"reuse rows found in this (non-reuse) earlier batch: {int(found[is_reuse].sum())}")


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dry", action="store_true", help="dry-run output: skip size / count targets")
    ap.add_argument("--show", type=int, default=1, help="decoded samples to print per source")
    ap.add_argument("--compare-to", action="append", default=[], help="earlier batch dir (repeatable): check zero file / row overlap")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/artifacts/tokenizer_v1/tokenizer.json")
    out = os.path.abspath(a.out)

    # ---- layout
    files = sorted(f for f in os.listdir(f"{out}/train") if f.endswith(".parquet"))
    check(len(files) >= 2, f"train/ has {len(files)} parquet files")
    check(files[-1] == "shard_zzzzz_val.parquet", f"last train parquet is the val shard ({files[-1]})")
    shards = files[:-1]
    check(all(f == f"shard_{i:05d}.parquet" for i, f in enumerate(shards)), "train shards are shard_00000.. contiguous")
    for p in ("val_long/val_long.parquet", "train_index.parquet", "report.json"):
        check(os.path.exists(f"{out}/{p}"), f"{p} exists")
    rep = json.load(open(f"{out}/report.json"))

    # ---- format
    for f in files:
        pf = pq.ParquetFile(f"{out}/train/{f}")
        md = pf.metadata
        names = pf.schema_arrow.names
        ok_cols = names == ["text"] and str(pf.schema_arrow.field("text").type) == "string"
        rgs = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        comp = {md.row_group(i).column(0).compression for i in range(md.num_row_groups)}
        mb = os.path.getsize(f"{out}/train/{f}") / 1e6
        check(ok_cols and comp == {"ZSTD"} and all(r == 1024 for r in rgs[:-1]) and rgs[-1] <= 1024,
              f"{f}: cols={names} comp={comp} rows={md.num_rows} row_groups={len(rgs)} ({mb:.1f} MB)")
        if not a.dry and f != files[-1]:
            check(100 <= mb <= 250, f"{f}: size {mb:.1f} MB in 100-250 MB")
        check(len(rgs) >= 8 or a.dry, f"{f}: >= 8 row groups (8-GPU DDP strides row groups by rank)")

    # ---- tokens
    it = pq.read_table(f"{out}/train_index.parquet")
    i_sh, i_row = it.column("shard").to_numpy(), it.column("row").to_numpy()
    i_src, i_n = np.array(it.column("src").to_pylist(), dtype=object), it.column("ntok_v1").to_numpy()
    tot_rows = 0
    worst, long_min, bad_count, n_special, n_fffd = 0, 10**9, 0, 0, 0
    src_tok = collections.Counter()
    samples = collections.defaultdict(list)
    rng = random.Random(0)
    long_texts_for_overlap = []
    my_hash, my_src = [], []
    repair = rep["args"].get("mix") == "repair3"
    modern_texts = []
    for i, f in enumerate(shards):
        texts = pq.read_table(f"{out}/train/{f}").column("text").to_pylist()
        sel = np.nonzero(i_sh == i)[0]
        sel = sel[np.argsort(i_row[sel])]
        check(len(sel) == len(texts) and (i_row[sel] == np.arange(len(sel))).all(),
              f"{f}: index rows {len(sel)} == shard rows {len(texts)}")
        tot_rows += len(texts)
        ns = []
        for j in range(0, len(texts), 256):
            ns += [len(e.ids) for e in tok.encode_batch(texts[j:j + 256], add_special_tokens=False)]
        ns = np.array(ns)
        bad_count += int((ns != i_n[sel]).sum())
        worst = max(worst, int(ns.max()))
        for t, s, n in zip(texts, i_src[sel].tolist(), ns.tolist()):
            src_tok[s] += n
            if a.compare_to:
                my_hash.append(row_hash(t)); my_src.append(s)
            if repair and s.startswith("long_") and not s.startswith("long_b2_"):
                modern_texts.append(t)
            if s.startswith("long_"):
                long_min = min(long_min, n)
                if any(x in t for x in SPECIAL):
                    n_special += 1
                if "���" in t:
                    n_fffd += 1
                long_texts_for_overlap.append(t)
            if len(samples[s]) < a.show and rng.random() < 0.05:
                samples[s].append((n, t))
        del texts
    check(bad_count == 0, f"re-tokenised counts match train_index ({bad_count} mismatches)")
    check(worst <= 32000, f"max train row = {worst} tokens (<= 32,000)")
    check(long_min >= 4096, f"min long piece = {long_min} tokens (>= 4,096)")
    check(n_special == 0, f"long rows with special-token literals: {n_special}")
    check(n_fffd == 0, f"long rows with U+FFFD runs: {n_fffd}")
    long_tok = sum(v for k, v in src_tok.items() if k.startswith("long_"))
    rep_tok = sum(v for k, v in src_tok.items() if k.startswith("replay_"))
    share = long_tok / max(1, long_tok + rep_tok)
    want = rep["args"].get("long_share", 0.5)
    check(abs(share - want) <= 0.05, f"long {long_tok:,} vs replay {rep_tok:,} tokens (long share {share:.3f}, target {want})")
    check(long_tok == rep["long"]["tokens"] and rep_tok == rep["replay"]["tokens"], "token totals == report.json")
    if not a.dry:
        lt = rep["args"]["long_target"]
        check(0.85 * lt <= long_tok <= 1.15 * lt, f"long pool {long_tok / 1e6:.1f}M tokens ~ target {lt / 1e6:.0f}M")
    bucket = collections.Counter()
    for k, v in src_tok.items():
        if k.startswith("long_"):
            bucket[rep["sources"][k[5:]]["bucket"]] += v
    print("bucket shares:", {b: round(v / long_tok, 3) for b, v in bucket.items()})
    if repair:                                           # ---- 3-pool repair mix
        tot = long_tok + rep_tok
        mod = sum(v for k, v in src_tok.items() if k.startswith("long_") and not k.startswith("long_b2_"))
        reu = long_tok - mod
        ra = rep["args"]
        want_mod = ra["modern_target"] / (ra["modern_target"] + ra["reuse_target"]) * (1 - ra["replay_share"])
        want_reu = (1 - ra["replay_share"]) - want_mod
        check(abs(mod / tot - want_mod) <= 0.03, f"modern pool {mod:,} tokens = {mod / tot:.3f} of total (target {want_mod:.3f})")
        check(abs(reu / tot - want_reu) <= 0.02, f"reuse pool {reu:,} tokens = {reu / tot:.3f} of total (target {want_reu:.3f})")
        check(abs(rep_tok / tot - ra["replay_share"]) <= 0.03, f"replay pool {rep_tok:,} tokens = {rep_tok / tot:.3f} (target {ra['replay_share']})")
        check(mod == rep["modern"]["tokens"] and reu == rep["reuse"]["tokens"], "modern / reuse totals == report.json")
        tol = 0.05 if not a.dry else 0.15
        for n, cfg in ra["modern"].items():
            sh = src_tok.get(f"long_{n}", 0) / max(1, mod)
            check(abs(sh - cfg["share"]) <= tol, f"modern {n}: {src_tok.get('long_' + n, 0):,} tokens, share {sh:.3f} (target {cfg['share']})")
        if not a.dry:
            check(tot >= 381 * 524288, f"total {tot:,} tokens >= 381 x 524,288 = {381 * 524288:,}")
        vg = []
        for t in pq.read_table(f"{out}/val_long/val_long.parquet", columns=["text"]).column("text").to_pylist():
            vg.append(np.unique(decontam.ngram_hashes(t, n=10)[0]))
        vg = np.unique(np.concatenate(vg))
        fr, hits_tot = [], 0
        for t in modern_texts:
            g = np.unique(decontam.ngram_hashes(t, n=10)[0])
            if g.size:
                pos = np.searchsorted(vg, g)
                ok = pos < vg.size
                h = int((vg[pos[ok]] == g[ok]).sum())
            else:
                h = 0
            hits_tot += h
            fr.append(h / max(1, g.size))
        fr = np.array(fr or [0.0])
        check(bool((fr < 0.02).all()), f"modern rows vs val_long word 10-grams (all, not sampled): {len(modern_texts)} rows, "
              f"max shared fraction {fr.max():.5f}, rows with any shared 10-gram {int((fr > 0).sum())}, "
              f"shared 10-grams total {hits_tot}")
    del modern_texts

    # ---- val shard
    vmd = pq.ParquetFile(f"{out}/train/{files[-1]}").metadata
    check(vmd.num_rows == rep["train"]["val_shard"]["rows"], f"val shard rows {vmd.num_rows}")
    if rep["train"]["val_shard"].get("sha256"):
        sha = sha256_file(f"{out}/train/{files[-1]}")
        check(sha == rep["train"]["val_shard"]["sha256"], f"val shard sha256 == report ({sha[:16]}...)")
    if rep["val_long"].get("copied_from"):
        sha = sha256_file(f"{out}/val_long/val_long.parquet")
        check(sha == rep["val_long"]["sha256"] == rep["val_long"]["source_sha256"],
              f"copied val_long sha256 == source {rep['val_long']['copied_from']} ({sha[:16]}...)")

    # ---- val_long
    vt = pq.read_table(f"{out}/val_long/val_long.parquet")
    check(vt.schema.names == ["text", "src", "ntok_v1"], f"val_long columns {vt.schema.names}")
    vtexts = vt.column("text").to_pylist()
    vn = [len(tok.encode(t, add_special_tokens=False).ids) for t in vtexts]
    check(vn == vt.column("ntok_v1").to_pylist(), "val_long ntok_v1 matches re-tokenisation")
    check(bool(vn) and min(vn) >= 33000, f"val_long: {len(vn)} docs, min {min(vn) if vn else 0} tokens (>= 33,000)")
    if not a.dry:
        check(250 <= len(vn) <= 350, f"val_long has ~300 docs ({len(vn)})")
    print("val_long per source:", dict(collections.Counter(vt.column("src").to_pylist())))
    # overlap: 20 random word 10-grams per held-out doc; flag a doc if >= 5 of them occur in the training long rows
    probes, owner = [], []
    for k, t in enumerate(vtexts):
        g, _ = decontam.ngram_hashes(t, n=10)
        if g.size:
            pick = np.random.default_rng(k).choice(g.size, size=min(20, g.size), replace=False)
            probes += g[pick].tolist()
            owner += [k] * len(pick)
    probes = np.array(probes, dtype=np.uint64)
    order = np.argsort(probes)
    ps, po = probes[order], np.array(owner)[order]
    found = np.zeros(ps.size, bool)
    for t in long_texts_for_overlap:
        g, _ = decontam.ngram_hashes(t, n=10)
        if g.size and ps.size:
            pos = np.searchsorted(ps, g)
            pos[pos >= ps.size] = 0
            hit = ps[pos] == g
            found[pos[hit]] = True
            # duplicates of a probe value: mark all equal entries
    per_doc = collections.Counter(po[found].tolist())
    overl = [k for k, c in per_doc.items() if c >= 5]
    check(len(overl) == 0, f"held-out docs overlapping training long rows (>= 5/20 sampled 10-grams): {len(overl)}"
          + (f" e.g. {[vt.column('src')[k].as_py() for k in overl[:5]]}" if overl else ""))

    if a.compare_to:
        mh_ = np.array(my_hash, dtype=np.uint64)
        for d in a.compare_to:
            compare_to(out, rep, os.path.abspath(d), mh_, my_src)

    # ---- decoded samples
    for s in sorted(samples):
        for n, t in samples[s]:
            mid = len(t) // 2
            print(f"\n----- {s} ({n} tokens) head:\n{t[:400]}\n----- mid:\n{t[mid:mid + 300]}\n----- tail:\n{t[-250:]}")
    print("\nFAILED:" if FAILS else "\nALL CHECKS PASSED", *FAILS, sep="\n  ")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
