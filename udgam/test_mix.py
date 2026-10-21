#!/usr/bin/env python
"""Verify a build_mix.py output dir: shards load, layout/ordering, sample docs per source, Gemma 4 rendering of
task docs, token estimates vs an exact re-tokenisation, achieved mix vs targets, throughput projection.
Usage: python test_mix.py OUT_DIR [--samples 3] [--chars 700] [--seed 0] [--rg-per-shard K]
--rg-per-shard K (full-scale builds): read only K random row groups per shard instead of every row; the exact
manifest-token check is then done per sampled row group against nothing (skipped) and the achieved mix comes from
the manifest's per-shard per-source token counts (exact) instead of re-tokenisation."""
import argparse, collections, glob, json, os, random, re, sys
import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_mix import TOKENIZER, SOURCES, SPECIAL_STRS, BUCKETS, SUBWEIGHTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--samples", type=int, default=3); ap.add_argument("--chars", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rg-per-shard", type=int, default=0, help="sample K row groups per shard (0 = read everything)")
    a = ap.parse_args()
    sampled = a.rg_per_shard > 0
    rg_rng = random.Random(a.seed + 7)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)
    man = json.load(open(f"{a.out}/manifest.json"))
    ids = man["source_ids"]
    shards = sorted(glob.glob(f"{a.out}/shard_*.parquet"))
    by_name = {s["file"]: s for s in man["shards"]}
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    print(f"== layout: {len(shards)} shards")
    ph = [by_name[os.path.basename(s)]["phase"] for s in shards]
    check(ph[-1] == "val", "last shard is validation")
    check(ph[:-1] == sorted(ph[:-1]), "phase A shards sort before phase B shards: " + "".join(p[0] for p in ph))
    docs = []          # (shard, row, src, text)
    for s in shards:
        pf = pq.ParquetFile(s)
        check(pf.schema_arrow.names == ["text"], f"{os.path.basename(s)}: single 'text' column; rows={pf.metadata.num_rows} "
              f"row_groups={pf.metadata.num_row_groups} size={os.path.getsize(s)/1e6:.1f}MB")
        rg = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
        check(all(r == 1024 for r in rg[:-1]) and rg[-1] <= 1024, f"   row groups of 1024 rows (last {rg[-1]})")
        src = np.load(f"{a.out}/_meta/{os.path.basename(s).replace('.parquet', '.sources.npy')}")
        check(len(src) == pf.metadata.num_rows, "   sidecar source ids aligned")
        starts = np.cumsum([0] + rg)
        pick = sorted(rg_rng.sample(range(len(rg)), min(a.rg_per_shard, len(rg)))) if sampled else range(len(rg))
        for g in pick:
            texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            docs += [(os.path.basename(s), starts[g] + i, ids[src[starts[g] + i]], t) for i, t in enumerate(texts)]
        # nanochat dataloader reads `rg.column('text')` per row group -> make sure that works
        pf.read_row_group(0).column("text")

    print("\n== token estimates (exact re-tokenisation vs manifest, and chars/4.4)")
    by_shard = collections.defaultdict(int)
    by_src_tok = collections.defaultdict(int); by_src_chr = collections.defaultdict(int)
    enc = tok.encode_batch([d[3] for d in docs], add_special_tokens=False)
    for d, e in zip(docs, enc):
        by_shard[d[0]] += len(e.ids); by_src_tok[d[2]] += len(e.ids); by_src_chr[d[2]] += len(d[3])
    for s in man["shards"]:
        if sampled:
            continue
        check(by_shard[s["file"]] == s["tokens"], f"{s['file']}: manifest tokens {s['tokens']} == exact {by_shard[s['file']]}")
    if sampled:
        print(f"  (sampled {len(docs)} docs from {a.rg_per_shard} row groups/shard: exact per-shard check skipped)")
    tot_t = sum(by_src_tok.values()); tot_c = sum(by_src_chr.values())
    print(f"  overall chars/token = {tot_c/tot_t:.3f}; chars/4.4 estimate error = {(tot_c/4.4 - tot_t)/tot_t:+.1%}")
    for k in sorted(by_src_tok):
        print(f"    {k:20s} chars/token {by_src_chr[k]/by_src_tok[k]:5.2f}   chars/4.4 err {(by_src_chr[k]/4.4-by_src_tok[k])/by_src_tok[k]:+6.1%}")

    print("\n== achieved mix (exact tokens) vs target")
    phase_of = {s["file"]: s["phase"] for s in man["shards"]}
    tokp = collections.defaultdict(lambda: collections.Counter())
    if sampled:                                    # exact counts from the manifest (written by build_mix at assembly)
        for s in man["shards"]:
            tokp[s["phase"]].update(s["per_source_tokens"])
    else:
        for d, e in zip(docs, enc):
            tokp[phase_of[d[0]]][d[2]] += len(e.ids)
    # Udgam-2: a build made with --mix v2 is checked against the v2 targets (its task share is deliberately small)
    mix = (man.get("args") or {}).get("mix")
    v2 = mix in ("v2", "v2b")
    v3 = mix in ("v3", "v3s")
    if v2:
        from build_mix import BUCKETS_V2, BUCKETS_V2B, TASK_SUBWEIGHTS_V2
        BUCKETS.clear(); BUCKETS.update(BUCKETS_V2 if man["args"]["mix"] == "v2" else BUCKETS_V2B)
        SUBWEIGHTS["task"] = dict(TASK_SUBWEIGHTS_V2)
    if v3:   # Udgam-3: v3 targets; chat rows are plain text and tool rows are not in pretraining
        from build_mix import BUCKETS_V3, BUCKETS_V3S, SUBWEIGHTS_V3, SOURCES_V3
        BUCKETS.clear(); BUCKETS.update(BUCKETS_V3 if mix == "v3" else BUCKETS_V3S)
        SUBWEIGHTS.update({k: dict(v) for k, v in SUBWEIGHTS_V3.items()})
        SOURCES.update(SOURCES_V3)
    sub_bucket = {sub: b for b, w in SUBWEIGHTS.items() for sub in w}
    sub_bucket["extra_task"] = "task"
    for p in ("A", "B"):
        T = sum(tokp[p].values())
        bk = collections.Counter()
        for k, v in tokp[p].items():
            bk[sub_bucket.get(k, k)] += v
        print(f"  phase {p}: {T/1e6:.2f}M tokens")
        for b, share in BUCKETS[p].items():
            print(f"    {b:12s} target {share:5.1%}  got {bk[b]/max(T,1):5.1%}")
    task_tool = sum(v for p in ("A", "B") for k, v in tokp[p].items() if SOURCES.get(k, {}).get("group") == "tool" or k == "extra_task")
    task_all = sum(v for p in ("A", "B") for k, v in tokp[p].items() if sub_bucket.get(k) == "task")
    if v3:
        check(task_tool == 0, f"v3: no tool/JSON rows in pretraining (got {task_tool} tokens)")
    else:
        min_tool = 0.15 if v2 else 0.35       # v2 keeps tool calling alive (nemo_tool .15 + xlam + table_gpt + gold), v1 >= 35%
        check(task_tool / max(task_all, 1) >= min_tool, f"tool/JSON share of task tokens = {task_tool/max(task_all,1):.1%} (>= {min_tool:.0%})")

    print("\n== licence policy (data/LICENSES_AND_PROVENANCE.md)")
    from build_mix import generator_ok, EXCLUDED_SOURCES
    for n, s in man["sources"].items():
        prov = s.get("provenance")
        check(bool(prov) and all(generator_ok(g["model"]) for g in prov["generators"]),
              f"{n}: provenance recorded, declared generators allowed "
              f"({', '.join(g['model'] for g in (prov or {}).get('generators', [])) or 'organic / no generator'})")
        kept = {g: c for g, c in s.get("generator_rows_read", {}).items() if not g.startswith("excluded: ")}
        if s.get("generator_rows_read"):
            check(all(generator_ok(g) for g in kept), f"   {n}: rows kept by generator {kept}; "
                  f"excluded {({g[10:]: c for g, c in s['generator_rows_read'].items() if g.startswith('excluded: ')})}")
    gone = [k for k in EXCLUDED_SOURCES if ":" not in k and k in man["source_ids"]]
    check(not gone, f"excluded sources absent from the build ({', '.join(k for k in EXCLUDED_SOURCES if ':' not in k)})")

    task_srcs = sorted({d[2] for d in docs if SOURCES.get(d[2], {}).get("task") or d[2] == "extra_task"})
    if v3:
        print("\n== v3 plain-text task docs (no chat special tokens)")
        for k in task_srcs:
            ds = [d[3] for d in docs if d[2] == k]
            n = max(len(ds), 1)
            spec = sum(1 for t in ds if any(x in t for x in SPECIAL_STRS))
            print(f"  {k:14s} n={len(ds):5d} starts 'User:' {sum(t.startswith('User:') or t.startswith('System:') for t in ds)/n:4.0%}"
                  f"  has 'Assistant:' {sum('Assistant:' in t for t in ds)/n:4.0%}  special-token literals {spec}")
            check(spec == 0 and all(t.startswith(("User:", "System:")) for t in ds), f"   {k}: plain text, no special tokens")
        task_srcs = []
    print("\n== Gemma 4 rendering of task docs")
    tid = {t: tok.token_to_id(t) for t in SPECIAL_STRS}
    for k in task_srcs:
        ds = [d[3] for d in docs if d[2] == k]
        n = len(ds)
        f = lambda pat: sum(1 for t in ds if pat in t) / max(n, 1)
        bal = sum(1 for t in ds if t.count("<|turn>") == t.count("<turn|>") + (1 if t.rstrip().endswith("<|tool_response>") else 0)) / n
        print(f"  {k:14s} n={n:5d} starts<|turn> {sum(t.startswith('<|turn>') for t in ds)/n:4.0%}  no<bos> {sum('<bos>' not in t for t in ds)/n:4.0%}"
              f"  tool_decl {f('<|tool>declaration:'):4.0%}  tool_call {f('<|tool_call>call:'):4.0%}  tool_resp {f('<|tool_response>response:'):4.0%}"
              f"  think {f('<|channel>thought'):4.0%}  turns_balanced {bal:4.0%}")
        check(all(t.startswith("<|turn>") for t in ds) and all("<bos>" not in t for t in ds), f"   {k}: starts with <|turn>, no <bos>")
    # specials must encode as single ids
    sample = next((d[3] for d in docs if "<|tool_call>call:" in d[3]), None)
    if sample:
        e = tok.encode(sample, add_special_tokens=False)
        check(tid["<|tool_call>"] in e.ids and tid["<|turn>"] in e.ids, "special tokens encode to single ids (<|tool_call>, <|turn>)")
    rx_call = re.compile(r"<\|tool_call>call:[A-Za-z0-9_.\-]+\{.*?\}<tool_call\|>", re.S)
    calls = [d for d in docs if "<|tool_call>" in d[3]]
    ok_calls = sum(1 for d in calls if len(rx_call.findall(d[3])) == d[3].count("<|tool_call>"))
    check(ok_calls == len(calls), f"tool_call blocks well-formed: {ok_calls}/{len(calls)}")
    nontask_special = sum(1 for d in docs if d[2] not in task_srcs and any(s in d[3] for s in SPECIAL_STRS))
    check(nontask_special == 0, f"no special-token literals in non-task docs ({nontask_special})")

    print(f"\n== {a.samples} random docs per source")
    rng = random.Random(a.seed)
    for k in sorted({d[2] for d in docs}):
        ds = [d for d in docs if d[2] == k]
        for d in rng.sample(ds, min(a.samples, len(ds))):
            t = d[3]
            body = t if len(t) <= a.chars else t[: a.chars * 2 // 3] + "\n   [...]\n" + t[-a.chars // 3:]
            print(f"\n----- {k} | {d[0]} row {d[1]} | {len(t)} chars")
            print(body)

    print("\n== nanochat BOS best-fit packing simulation (T=2048, buffer 1000) on the train shards")
    # mirrors nanochat/dataloader.py: rows of T+1 tokens; take the largest doc that fits, else crop the shortest
    lens = [len(e.ids) + 1 for d, e in zip(docs, enc) if phase_of[d[0]] != "val"]
    random.Random(1).shuffle(lens)
    cap, buf, used, i, rows = 2049, [], 0, 0, 0
    total = sum(lens)
    while True:
        pos = 0
        while pos < cap:
            while len(buf) < 1000 and i < len(lens):
                buf.append(lens[i]); i += 1
            if not buf:
                break
            rem = cap - pos
            fit = [k for k, L in enumerate(buf) if L <= rem]
            if fit:
                k = max(fit, key=lambda k: buf[k]); pos += buf.pop(k)
            else:
                k = min(range(len(buf)), key=lambda k: buf[k]); buf.pop(k); pos += rem
        if pos < cap:
            break
        rows += 1; used += cap
    consumed = total - sum(buf)
    print(f"  {rows} rows; trained tokens / consumed shard tokens = {used/consumed:.1%} "
          f"(nanochat's docstring quotes ~65% for unchunked web data)")
    over = sum(1 for L in lens if L > 2049)
    print(f"  docs longer than one row: {over} ({over/len(lens):.2%})")

    print("\n== decontamination (sample) per source")
    for k, s in man["sources"].items():
        print(f"  {k:20s} drop rate {s['decontam_drop_rate']:.3%}  {s['decontam_drops']}")
    print("\n== throughput (per worker) and projection")
    for k, s in man["sources"].items():
        print(f"  {k:20s} {s['docs_per_worker_second']:8.1f} docs/s  {s['tokens_per_worker_second']/1e3:8.1f}K tok/s  "
              f"{s['bytes_read_per_token']:6.2f} B read/token")
    print("  " + json.dumps(man["projection_full_build_32vcpu"]))
    print(f"\n{len(fails)} FAILURES" + ("".join("\n  - " + f for f in fails) if fails else ""))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
