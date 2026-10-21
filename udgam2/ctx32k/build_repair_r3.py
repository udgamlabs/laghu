#!/usr/bin/env python
"""Udgam-2 32K "repair" stage data (batch 3, spec udgam2/specs/ctx32k.md "Repair stage"): 3-pool mix -> nanochat shards.

README
------
Why: the 32K stages' long pool (old books/OCR, papers, gov, code) had no modern web text and the commonsense benchmarks
fell 4-7 points. The repair batch puts modern long web-style documents back, like the Udgam-384M 4K stage
(ops/modal_4k_data.py), whose approach this mirrors, run locally on the Mac (CPU only, $0, streamed, no raw downloads kept).

    PY=<venv with tokenizers==0.23.2, pyarrow, numpy, huggingface_hub>
    $PY udgam2/ctx32k/build_repair_r3.py --dry-run            # small: 1 unit per source, ~6M tokens -> data/udgam2_ctx32k_r3_dry
    $PY udgam2/ctx32k/build_repair_r3.py                      # full: ~0.21B tokens -> data/udgam2_ctx32k_r3
    $PY udgam2/ctx32k/test_long32k.py data/udgam2_ctx32k_r3 --compare-to data/udgam2_ctx32k --compare-to data/udgam2_ctx32k_t2

Pools (tokenizer_v1 tokens, no BOS):
  modern  (~45%) FRESH long docs >= 4,096 tokens from the licence-audited pretraining sources, shares like the 4K stage:
          Ultra-FineWeb en 55%, Wikipedia 20231101.en 15%, Nemotron-CC-Math-v1 4plus 15%, github-code-clean 15%. Read
          through udgam/build_mix.py's own run_unit (its makers, licence / generator filters, mojibake / special-token
          drops, 13-gram decontamination with the same index and rule), with a cheap vectorised char-length prefilter
          first. Docs > 32,000 tokens are split at paragraph breaks into <= 32,000-token pieces (build_long32k.make_pieces:
          tail piece < 4,096 dropped, at most MAX_PIECES pieces per doc). Longer docs are preferred: within each source
          docs >= 8,192 tokens are taken first, then 4,096-8,191 (each group seeded-shuffled).
  reuse   (~10%) long-pool rows of batch 2 (data/udgam2_ctx32k_t2/train, src long_*), seeded random sample stratified by
          batch 2's bucket token shares (keeps long recall alive). Labelled long_b2_<source>.
  replay  (~45%) fresh YOUR_HF_NAMESPACE/udgam-pretrain-v2b phase-B rows not used by batch 1 or 2 (both report.json files):
          continues at the first unused row (shard_00098 row 70,832), then shard_00097, ...
Which source units are unused (recorded in report.json "unit_audit"):
  build_mix.list_units(name, spec, seed 1234) reproduces every build's shuffled unit list (checked: the 4K stage's
  recorded units in ops/cpt4k/data_units.json must match the recomputed list at the same indices). Pretraining builds
  consume a HEAD prefix of that list (the scheduler pops units in order): v1, then v2 (--skip-units-from v1), then v2b
  (--skip-units-from v1,v2); each manifest's sources[name].units counts unit results incl. resumes, so the summed head
  is an upper bound of what they touched. The 4K stage took the TAIL (data_units.json; v2/v2b --skip-tail agrees).
  Units are picked from just before that tail, and must lie in [head_used, n - tail_used).
  github-code-clean is the exception: v1 + v2 + v2b + 4K touched all 880 units (v2b ran out: code_shell kept 125M of a
  275M quota with epochs_used 1). Every build admitted only files < 20,000 chars (build_mix code_prefilter and mk_code
  "code_too_long"), so files >= 20,000 chars were never used anywhere; the code pool takes ONLY those (same licence /
  language / auto-generated filters, plus minified / data-file filters), from units just before the 4K tail.
Also: modern docs sharing >= 2% (and >= 50) of their word 10-grams with val_long are dropped, and the test's sampled
val_long probe check is enforced greedily over all long rows (reuse + modern); Wikipedia docs are dropped if their title
or MinHash matches a common-pile wikipedia doc collected for batch 1 or 2 (same article, other snapshot).
Output (--out): same layout as batch 1/2 -- train/shard_NNNNN.parquet (column text, zstd, 1024-row groups) + batch 1's
train/shard_zzzzz_val.parquet copied LAST (sha256 checked), val_long/val_long.parquet copied from batch 1, train_index.parquet
(shard, row, src, ntok_v1), report.json; _work/ holds parts + logs (safe to delete).
"""
import argparse, collections, glob, hashlib, json, os, random, re, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam"))
sys.path.insert(0, HERE)
import build_mix as bm            # noqa: E402
import build_long32k as bl        # noqa: E402  (make_pieces, val_stats, prepare_val, mh, dedupe helpers, constants)

TOK_V1 = f"{ROOT}/artifacts/tokenizer_v1/tokenizer.json"
DECONTAM_INDEX = f"{ROOT}/data/decontam/index_13gram.npz"
B1, B2 = f"{ROOT}/data/udgam2_ctx32k", f"{ROOT}/data/udgam2_ctx32k_t2"
SEED_UNITS = 1234                  # build_mix --seed of v1 / v2 / v2b (the unit shuffle)
MIN_TOK = 4096                     # modern docs: >= this many v1 tokens
LONG_PREF = 8192                   # docs >= this are taken first
CODE_MIN_CHARS, CODE_MAX_CHARS = 20000, 300000
# per source: share of the modern pool, char prefilter (cheap, before decontam / tokenising), units counted back from the
# 4K stage's tail (offset 1 = the unit right before it), pieces cap per doc
MODERN = {
    "web_hq": dict(share=0.55, min_chars=16000, n_units=5, max_pieces=6),
    "wiki":   dict(share=0.15, min_chars=14000, n_units=3, max_pieces=6),
    "math":   dict(share=0.15, min_chars=12000, n_units=3, max_pieces=6),
    "code":   dict(share=0.15, min_chars=CODE_MIN_CHARS, n_units=4, max_pieces=2),
}
OVERSHOOT = 1.6                    # collect target x this (dedupe / val drops / longest-first selection)
MANIFESTS = [("v1", f"{ROOT}/data/build_v1/manifest.json"), ("v2", f"{ROOT}/data/build_v2/manifest.json"),
             ("v2b", "hf:YOUR_HF_NAMESPACE/udgam-pretrain-v2b/manifest.json")]
FOURK_UNITS = f"{ROOT}/ops/cpt4k/data_units.json"


# ------------------------------------------------------------------------------------------- build_mix plug-ins
def _len_prefilter(col, min_chars, max_chars=None):
    def f(batch, want_subs):
        n = pc.utf8_length(batch.column(col))
        m = pc.greater_equal(n, min_chars)
        if max_chars:
            m = pc.and_(m, pc.less_equal(n, max_chars))
        return batch.filter(pc.fill_null(m, False))
    return f


def code_long_prefilter(batch, want_subs):
    """build_mix.code_prefilter (licence + language) but keeps files >= 20,000 chars: the ones pretraining never used."""
    langs = bm.SHELL_LANGS | bm.GENERAL_LANGS
    n = pc.utf8_length(batch.column("code"))
    m = pc.and_(pc.is_in(batch.column("license"), pa.array(sorted(bm.PERMISSIVE_CODE))),
                pc.is_in(batch.column("language"), pa.array(sorted(langs))))
    m = pc.and_(m, pc.and_(pc.greater_equal(n, CODE_MIN_CHARS), pc.less_equal(n, CODE_MAX_CHARS)))
    return batch.filter(pc.fill_null(m, False))


_JSON_LINE = re.compile(r'^\s*("[^"]*"\s*:|[\[\]{}(),]+;?\s*$|-?[\d.eE+-]+,?\s*$|"[^"]*",?\s*$)')


def mk_code_long(rec, ctx, rng, src):
    """build_mix.mk_code with the < 20,000-char ceiling inverted, plus minified / data-file filters for long files."""
    if rec["license"] not in bm.PERMISSIVE_CODE:
        return [(bm.DROP, "code_license")]
    code = rec["code"] or ""
    if len(code) < CODE_MIN_CHARS:
        return [(bm.DROP, "code_lt_20000_chars_seen_in_pretraining")]
    if len(code) > CODE_MAX_CHARS:
        return [(bm.DROP, "code_gt_300000_chars")]
    if code.startswith("version https://git-lfs"):
        return [(bm.DROP, "code_git_lfs_pointer")]
    head = code[:600].lower()
    if any(k in head for k in ("auto-generated", "autogenerated", "generated by", "do not edit", "code generated", "@generated")):
        return [(bm.DROP, "code_autogenerated")]
    lines = code.split("\n")
    lens = [len(x) for x in lines]
    if max(lens) > 1000 or len(code) / max(1, len(lines)) > 100:
        return [(bm.DROP, "code_minified_or_long_lines")]
    digits = sum(c.isdigit() for c in code)
    if digits / len(code) > 0.25:
        return [(bm.DROP, "code_numeric_data")]
    if sum(x.lstrip().upper().startswith("INSERT INTO") for x in lines) > 50:
        return [(bm.DROP, "code_sql_dump")]
    if code.lstrip("\ufeff \t\r\n").startswith("<?xml") or "<!DOCTYPE TS>" in code[:2000]:
        return [(bm.DROP, "code_xml_or_qt_translation")]     # e.g. Qt Linguist .ts files labelled TypeScript
    nonempty = [x for x in lines if x.strip()]
    if sum(bool(_JSON_LINE.match(x)) for x in nonempty) > 0.5 * max(1, len(nonempty)):
        return [(bm.DROP, "code_json_like_data")]
    lang = rec["language"]
    if lang in bm.SHELL_LANGS:
        return [("code_shell", code, None)]
    if lang in bm.GENERAL_LANGS:
        return [("code_general", code, None)]
    return [(bm.DROP, "code_language_not_selected")]


bm.MAKERS["code_long"] = mk_code_long
bm.PREFILTERS["code_long"] = code_long_prefilter
bm.PREFILTERS["long_web_hq"] = _len_prefilter("content", MODERN["web_hq"]["min_chars"])
bm.PREFILTERS["long_wiki"] = _len_prefilter("text", MODERN["wiki"]["min_chars"])
bm.PREFILTERS["long_math"] = _len_prefilter("text", MODERN["math"]["min_chars"])


def source_spec(name):
    spec = dict(bm.SOURCES[name])
    spec["min_chars"] = MODERN[name]["min_chars"]
    spec["prefilter"] = "code_long" if name == "code" else f"long_{name}"
    if name == "code":
        spec["make"] = "code_long"
    return spec


# ------------------------------------------------------------------------------------------- unit audit
def _load_manifest(ref, work):
    if ref.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        repo, fn = ref[3:].rsplit("/", 1)
        return json.load(open(hf_hub_download(repo, fn, repo_type="dataset", local_dir=f"{work}/manifests/{repo.replace('/', '__')}")))
    return json.load(open(ref))


def unit_audit(work, n_units_override=None):
    """Recompute each source's shuffled unit list, the head used by v1/v2/v2b, the tail used by the 4K stage; pick units."""
    mans = [(tag, _load_manifest(ref, work), ref) for tag, ref in MANIFESTS]
    fourk = json.load(open(FOURK_UNITS))
    audit, picks = {}, {}
    for tag, m, ref in mans:
        assert m["seed"] == SEED_UNITS, f"{tag}: seed {m['seed']}"
    for name in MODERN:
        units = bm.list_units(name, bm.SOURCES[name], SEED_UNITS)
        n = len(units)
        head = {tag: int(m["sources"][name]["units"]) for tag, m, _ in mans}
        epochs = {tag: int(m["sources"][name].get("epochs_used", 1)) for tag, m, _ in mans}
        assert all(e == 1 for e in epochs.values()), f"{name}: a build cycled the source ({epochs})"
        rows = [r for r in fourk if isinstance(r, dict) and r.get("name") == name]
        match = all((units[r["which"]]["path"], units[r["which"]]["k"], units[r["which"]]["K"]) == (r["unit"], r["k"], r["K"])
                    for r in rows)
        assert match, f"{name}: recomputed unit list does not reproduce the 4K stage's units"
        tail = max(-r["which"] for r in rows)
        skip_tail = {tag: m["args"].get("skip_tail", "") for tag, m, _ in mans}
        head_used = sum(head.values())
        free = (head_used, n - tail)
        k = (n_units_override or {}).get(name, MODERN[name]["n_units"])
        idx = [n - tail - 1 - j for j in range(k)]
        if name == "code":
            note = (f"all {n} units were read: head {head_used} (v1+v2+v2b, incl. resumes) + tail {tail} >= {n}; v2b's "
                    f"code_shell kept {m_code_shell(mans)} of its quota with epochs_used 1 (source exhausted). Only files "
                    f">= {CODE_MIN_CHARS} chars are used: build_mix's code_prefilter and mk_code drop every file >= 20,000 "
                    "chars in every build (v1, v2, v2b) and the 4K stage used mk_code too, so these files were never trained on.")
        else:
            assert all(free[0] <= i < free[1] for i in idx), f"{name}: picked units {idx} outside the free range {free}"
            note = f"picked units lie in the never-used range [{free[0]}, {free[1]})"
        picks[name] = [dict(units[i], idx=i, pos=(0, 0), epoch=1, resumes=0) for i in idx]
        audit[name] = dict(n_units=n, head_units_by_build=head, head_used_upper_bound=head_used, tail_used_by_4k=tail,
                           tail_skip_args=skip_tail, fourk_units_reproduced=match, free_range=list(free),
                           picked=[dict(idx=i, offset_from_end=i - n, path=units[i]["path"], k=units[i]["k"], K=units[i]["K"])
                                   for i in idx], note=note)
    audit["_method"] = ("build_mix.list_units(name, SOURCES[name], seed 1234) recomputed; it reproduces the 4K stage's "
                        "recorded units (ops/cpt4k/data_units.json) at the same indices, so the order is the one every "
                        "build used. Builds consume a head prefix (units popped in order); manifest sources[name].units "
                        "counts unit results incl. resumes (upper bound). Manifests: " +
                        ", ".join(f"{t}={r}" for t, _, r in mans) + f"; 4K tail: {FOURK_UNITS}")
    return audit, picks


def m_code_shell(mans):
    m = dict((t, mm) for t, mm, _ in mans)["v2b"]["sources"]["code"]
    kept = sum(v for v in m["kept_tokens"]["code_shell"].values())
    q = sum(m["quota_tokens"]["code_shell"].values())
    return f"{kept / 1e6:.0f}M/{q / 1e6:.0f}M tokens"


# ------------------------------------------------------------------------------------------- collect (workers)
def _winit():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    bm.TOKENIZER = TOK_V1          # run_unit's exact token counts in tokenizer_v1 (the trainer's tokenizer)
    bm._init_worker(DECONTAM_INDEX)


def run_modern_unit(task):
    t0 = time.time()
    name, u = task["name"], task["unit"]
    spec = source_spec(name)
    raw = task["out"] + ".raw.parquet"
    budgets = {s: int(task["budget"]) for s in spec["subs"]}
    st = bm.run_unit(dict(src=spec, unit=u, budgets=budgets, uid=f"r3-{name}-{u['idx']}", max_doc_tokens=0, minhash=False,
                          out=raw, time_limit=task["time_limit"], phase="B", source=name))
    tok = bm._CTX.tok
    t = pq.read_table(raw, columns=["text", "ntok", "sub"])
    drops, info = collections.Counter(), collections.Counter()
    w = bl.PartWriter(task["out"])
    kept, kept_tok = 0, 0
    for order, (text, n, sub) in enumerate(zip(t.column("text").to_pylist(), t.column("ntok").to_pylist(), t.column("sub").to_pylist())):
        if n < MIN_TOK:
            drops["length_lt_4096_tokens"] += 1
            continue
        pieces, pinfo = bl.make_pieces(text, n, tok, MODERN[name]["max_pieces"])
        info.update(pinfo)
        pieces = [(p, m) for p, m in pieces if m >= MIN_TOK]
        if not pieces:
            drops["no_piece_ge_4096"] += 1
            continue
        title = text.split("\n", 1)[0] if name == "wiki" else ""
        w.add(doc_id=f"{u['path']}#{u['k']}/{u['K']}:{order}", src=name, unit=f"{u['path']}#{u['k']}/{u['K']}", order=order,
              title_key=bl.norm_title(title), head_hash=bl.head_hash(text), mh=bl.mh_signed(text), ntok_doc=n,
              pieces=[p for p, _ in pieces], piece_ntok=[m for _, m in pieces], **bl.val_stats([p for p, _ in pieces], task["val_dir"]))
        kept += 1
        kept_tok += sum(m for _, m in pieces)
    w.close()
    os.remove(raw)
    return dict(name=name, idx=u["idx"], unit=f"{u['path']}#{u['k']}/{u['K']}", budget=task["budget"], out=task["out"],
                records=st["records"], rows_prefiltered=st["prefiltered"], bytes_read=st["bytes_read"],
                exhausted=st["exhausted"], produced_ge_min_chars=st["produced"], docs_ge_min_chars=st["docs"],
                drops_build_mix=st["drops"], decontam=st["decontam"], drops_long=dict(drops), info=dict(info),
                kept_docs=kept, kept_tokens=kept_tok, gen_rows=st.get("gen_rows", {}),
                seconds=round(time.time() - t0), stream_seconds=round(st["seconds"]))


def collect(args, work, picks):
    os.makedirs(f"{work}/parts", exist_ok=True)
    os.makedirs(f"{work}/results", exist_ok=True)
    tasks = []
    for name, us in picks.items():
        tgt = args.modern_target * MODERN[name]["share"] * OVERSHOOT
        per_unit = tgt / max(1, len(us) - 1) if len(us) > 1 else tgt     # one unit of slack
        for u in us:
            rp = f"{work}/results/{name}_{u['idx']:05d}.json"
            if args.resume and os.path.exists(rp):
                continue
            tasks.append(dict(name=name, unit=u, budget=per_unit, time_limit=args.unit_time_limit,
                              out=f"{work}/parts/{name}_{u['idx']:05d}.parquet", val_dir=work))
    print(f"collect: {len(tasks)} unit tasks, {args.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit) as ex:
        futs = {ex.submit(run_modern_unit, t): t for t in tasks}
        for f in as_completed(futs):
            t = futs[f]
            try:
                r = f.result()
            except Exception as e:
                import traceback
                r = dict(name=t["name"], idx=t["unit"]["idx"], error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
            json.dump(r, open(f"{work}/results/{t['name']}_{t['unit']['idx']:05d}.json", "w"), indent=1, default=str)
            print(f"  [{time.time() - t0:6.0f}s] {r['name']} unit {r['idx']}: " +
                  (f"ERROR {r['error']}" if "error" in r else
                   f"{r['kept_docs']} docs {r['kept_tokens'] / 1e6:.2f}M tok, {r['records']} recs "
                   f"(+{r['rows_prefiltered']} prefiltered), {r['bytes_read'] / 1e6:.0f} MB, {r['seconds']}s, exhausted={r['exhausted']}"),
                  flush=True)


def load_results(work):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{work}/results/*.json"))]


# ------------------------------------------------------------------------------------------- assemble
def _rh(t):
    return int.from_bytes(hashlib.blake2b(t.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def assemble(args, work, results, audit, t_start):
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tok = Tokenizer.from_file(TOK_V1)
    rng = random.Random(args.seed)
    out = args.out
    errors = [r for r in results if "error" in r]
    rep = dict(args=dict(mix="repair3", modern_target=args.modern_target, reuse_target=args.reuse_target,
                         replay_share=args.replay_share, long_share=round(1 - args.replay_share, 4),
                         long_target=args.modern_target + args.reuse_target, seed=args.seed, dry_run=args.dry_run,
                         ctx_max=bl.CTX_MAX, min_tok=MIN_TOK, long_pref=LONG_PREF, overshoot=OVERSHOOT,
                         modern=MODERN, code_chars=[CODE_MIN_CHARS, CODE_MAX_CHARS], reuse_from=B2,
                         copy_val_long=f"{B1}/val_long/val_long.parquet", val_shard_from=f"{B1}/train/{bl.VAL_NAME}"),
               unit_audit=audit, sources={}, drops={}, files_used={}, heldout_files={}, errors=errors)

    # ---- modern docs: load, drop damaged / val-overlapping, dedupe
    docs = {n: [] for n in MODERN}
    for r in sorted([r for r in results if "error" not in r], key=lambda r: (r["name"], -r["idx"])):
        docs[r["name"]] += pq.read_table(r["out"]).to_pylist()
    ddrops = collections.Counter()
    vfh = collections.Counter()
    for n in MODERN:
        keep = []
        for d in docs[n]:
            if any("���" in p for p in d["pieces"]):
                ddrops[f"{n}:replacement_char"] += 1
                continue
            fr = (d["vh"] or 0) / max(1, d["vng"] or 0)
            vfh["0" if fr == 0 else "<0.1%" if fr < 1e-3 else "<1%" if fr < 1e-2 else "<2%" if fr < bl.VAL_FRAC_MAX else ">=2%"] += 1
            if fr >= bl.VAL_FRAC_MAX and (d["vh"] or 0) >= bl.VAL_HITS_MIN:
                ddrops[f"{n}:val_long_10gram_overlap"] += 1
                continue
            keep.append(d)
        docs[n] = keep
    # Wikipedia: same article from common-pile wikipedia (batch 1 / 2 collected docs) -> drop (title or MinHash)
    pt, pm = set(), set()
    for pdir in (f"{B1}/_work/parts", f"{B2}/_work/parts"):
        p = f"{pdir}/train_wikipedia.parquet"
        if os.path.exists(p):
            for x in pq.read_table(p, columns=["title_key", "mh"]).to_pylist():
                if x["title_key"]:
                    pt.add(x["title_key"])
                pm.update(enumerate(x["mh"] or []))
    rep["wiki_prior"] = dict(batch12_wikipedia_titles=len(pt))
    for n in MODERN:
        sh, sm, keep = set(), set(), []
        for d in docs[n]:
            if n == "wiki" and d["title_key"] and d["title_key"] in pt:
                ddrops[f"{n}:batch12_same_title"] += 1
                continue
            if n == "wiki" and d["mh"] and bl._hit_bands(d["mh"], pm):
                ddrops[f"{n}:batch12_near_dup"] += 1
                continue
            if d["head_hash"] and d["head_hash"] in sh:
                ddrops[f"{n}:head_hash"] += 1
                continue
            if d["mh"] and bl._hit_bands(d["mh"], sm):
                ddrops[f"{n}:minhash"] += 1
                continue
            sh.add(d["head_hash"])
            sm.update(enumerate(d["mh"] or []))
            keep.append(d)
        docs[n] = keep

    # ---- reuse pool: batch-2 long rows, stratified by batch 2's bucket token shares
    r2 = json.load(open(f"{B2}/report.json"))
    it = pq.read_table(f"{B2}/train_index.parquet").to_pydict()
    cand = collections.defaultdict(list)
    for s, rw, src, nt in zip(it["shard"], it["row"], it["src"], it["ntok_v1"]):
        if src.startswith("long_"):
            cand[r2["sources"][src[5:]]["bucket"]].append((s, rw, src, nt))
    btok = {b: sum(x[3] for x in v) for b, v in cand.items()}
    ball = sum(btok.values())
    pick = []
    for b in sorted(cand):
        v = sorted(cand[b])
        rng.shuffle(v)
        want, got = args.reuse_target * btok[b] / ball, 0
        for x in v:
            if got >= want:
                break
            pick.append(x)
            got += x[3]
    by_shard = collections.defaultdict(list)
    for x in pick:
        by_shard[x[0]].append(x)
    reuse_rows = []
    for s in sorted(by_shard):
        texts = pq.read_table(f"{B2}/train/shard_{s:05d}.parquet", columns=["text"]).column("text")
        for _, rw, src, nt in sorted(by_shard[s], key=lambda x: x[1]):
            reuse_rows.append((texts[rw].as_py(), "long_b2_" + src[5:], nt, (s, rw)))
        del texts
    reuse_src = collections.Counter()
    for _, src, nt, _ in reuse_rows:
        reuse_src[src[8:]] += nt
    reuse_tok = sum(reuse_src.values())
    print(f"reuse: {len(reuse_rows)} rows, {reuse_tok / 1e6:.1f}M tokens", flush=True)

    # ---- greedy val_long probe enforcement (as in batch 2): reuse rows first, then modern docs
    probe_owner = np.load(f"{work}/val_probes_owner_sorted.npy")
    probe_found, owner_found = set(), collections.Counter()

    def probe_ok(vprobe, commit=True, force=False):
        new = [k for k in (vprobe or []) if k not in probe_found]
        add = collections.Counter(int(probe_owner[k]) for k in new)
        if not force and any(owner_found[o] + c >= bl.EVAL_OVERLAP_MAX for o, c in add.items()):
            return False
        if commit:
            probe_found.update(new)
            owner_found.update(add)
        return True
    reuse_probe_viol = 0
    for t, _, _, _ in reuse_rows:                      # kept regardless (batch 2 passed the check); count violations
        vp = bl.val_stats([t], work)["vprobe"]
        if not probe_ok(vp, commit=False):
            reuse_probe_viol += 1
        probe_ok(vp, force=True)
    probe_drops = collections.Counter()

    # ---- modern selection: per-source targets (shortfall moved to sources with spare), >= 8K docs first
    tg = {n: args.modern_target * MODERN[n]["share"] for n in MODERN}
    avail = {n: sum(sum(d["piece_ntok"]) for d in docs[n]) for n in MODERN}
    final_tg = {n: min(tg[n], avail[n]) for n in MODERN}
    short = sum(tg[n] - final_tg[n] for n in MODERN)
    spare = {n: avail[n] - final_tg[n] for n in MODERN if avail[n] > final_tg[n]}
    if short > 0 and spare:
        ts = sum(spare.values())
        for n, s in spare.items():
            final_tg[n] += min(s, short * s / ts)
    modern_rows, used_units = [], collections.defaultdict(set)
    for n in MODERN:
        ds = sorted(docs[n], key=lambda d: d["doc_id"])
        rng.shuffle(ds)
        ds.sort(key=lambda d: d["ntok_doc"] < LONG_PREF)          # stable: >= 8K first, each group shuffled
        got, nd, hist_doc = 0, 0, collections.Counter()
        for d in ds:
            if got >= final_tg[n]:
                break
            if not probe_ok(d["vprobe"]):
                probe_drops[n] += 1
                continue
            nd += 1
            used_units[n].add(d["unit"])
            m0 = d["ntok_doc"]
            hist_doc["4K-8K" if m0 < 8192 else "8K-16K" if m0 < 16384 else "16K-32K" if m0 <= bl.CTX_MAX else ">32K(split)"] += 1
            for p, m in zip(d["pieces"], d["piece_ntok"]):
                modern_rows.append((p, f"long_{n}", m))
                got += m
        ns = [m for _, s, m in modern_rows if s == f"long_{n}"]
        res = [r for r in results if r.get("name") == n and "error" not in r]
        rep["sources"][n] = dict(
            bucket=f"modern_{n}", pool="modern", target_tokens=round(tg[n]), final_target_tokens=round(final_tg[n]),
            tokens=got, docs=nd, pieces=len(ns), available_tokens_after_filters=avail[n],
            share_of_modern=None, doc_length_hist=dict(hist_doc),
            piece_length_hist=bl._piece_hist(ns), piece_length_hist_tokens=bl._piece_hist(ns, weight=True),
            token_share_pieces_ge_8k=round(sum(m for m in ns if m >= 8192) / max(1, got), 4),
            token_share_pieces_ge_16k=round(sum(m for m in ns if m >= 16384) / max(1, got), 4),
            stream=dict(units=len(res), records=sum(r["records"] for r in res), rows_prefiltered=sum(r["rows_prefiltered"] for r in res),
                        bytes_read=sum(r["bytes_read"] for r in res), kept_docs=sum(r["kept_docs"] for r in res),
                        kept_tokens=sum(r["kept_tokens"] for r in res),
                        kept_tokens_per_gb=round(sum(r["kept_tokens"] for r in res) / max(1e-9, sum(r["bytes_read"] for r in res) / 1e9)),
                        unit_seconds=[r["seconds"] for r in res], exhausted=[r["exhausted"] for r in res]))
        dr = collections.Counter()
        for r in res:
            dr.update(r["drops_build_mix"]); dr.update(r["drops_long"])
        dr["dedupe_and_val"] = sum(v for k, v in ddrops.items() if k.startswith(n + ":"))
        dr["val_probe_greedy"] = probe_drops[n]
        rep["drops"][n] = dict(dr)
        rep["files_used"][n] = [dict(file=x) for x in sorted(used_units[n])]
    modern_tok = sum(m for _, _, m in modern_rows)
    for n in MODERN:
        rep["sources"][n]["share_of_modern"] = round(rep["sources"][n]["tokens"] / max(1, modern_tok), 4)
    # reuse sources in the report (bucket lookup by the test; files = batch 2's files for that source)
    for s, nt in reuse_src.items():
        rep["sources"]["b2_" + s] = dict(bucket="reuse_" + r2["sources"][s]["bucket"], pool="reuse", tokens=nt,
                                         rows=sum(1 for _, x, _, _ in reuse_rows if x == "long_b2_" + s))
        rep["files_used"]["b2_" + s] = r2["files_used"].get(s, [])
    rep["dedupe_detail"] = dict(ddrops)
    rep["val_overlap"] = dict(frac_max=bl.VAL_FRAC_MAX, hits_min=bl.VAL_HITS_MIN, modern_doc_frac_hist=dict(vfh),
                              probe_dropped_docs=dict(probe_drops), reuse_rows_over_probe_limit=reuse_probe_viol,
                              probes_found=len(probe_found), max_probes_found_per_val_doc=max(owner_found.values() or [0]))
    long_rows = modern_rows + [(t, s, n) for t, s, n, _ in reuse_rows]
    long_tok = modern_tok + reuse_tok
    rep["modern"] = dict(tokens=modern_tok, pieces=len(modern_rows),
                         piece_length_hist=bl._piece_hist([m for _, _, m in modern_rows]),
                         piece_length_hist_tokens=bl._piece_hist([m for _, _, m in modern_rows], weight=True),
                         max_piece_tokens=max([m for _, _, m in modern_rows] or [0]),
                         min_piece_tokens=min([m for _, _, m in modern_rows] or [0]))
    rep["reuse"] = dict(from_dir=B2, target_tokens=args.reuse_target, tokens=reuse_tok, rows=len(reuse_rows),
                        tokens_per_source=dict(reuse_src.most_common()),
                        bucket_target_shares={b: round(v / ball, 4) for b, v in btok.items()},
                        rows_by_shard={f"shard_{s:05d}.parquet": len(v) for s, v in sorted(by_shard.items())},
                        sample=f"stratified by batch-2 bucket token shares, seed {args.seed}")
    rep["long"] = dict(tokens=long_tok, pieces=len(long_rows), modern_tokens=modern_tok, reuse_tokens=reuse_tok,
                       piece_length_hist=bl._piece_hist([m for _, _, m in long_rows]),
                       max_piece_tokens=max([m for _, _, m in long_rows] or [0]))
    print(json.dumps({k: rep[k] for k in ("modern", "reuse")}, indent=1)[:3000], flush=True)
    del docs

    # ---- replay: v2b phase-B rows after the ones batches 1 and 2 used
    _, used = bl.load_exclusions([f"{B1}/report.json", f"{B2}/report.json"])
    prior_h = set()                                       # exact texts of batch 1 / 2 train rows: never repeat a row
    for bdir in (B1, B2):
        for f in sorted(glob.glob(f"{bdir}/train/shard_[0-9]*.parquet")):
            prior_h.update(_rh(t) for t in pq.read_table(f, columns=["text"]).column("text").to_pylist())
    seen_h = {_rh(t) for t, _, _ in long_rows}
    replay_dup = collections.Counter()
    rep["replay_rows_used_before"] = used
    dl = f"{work}/dl"
    os.makedirs(dl, exist_ok=True)
    replay_target = round(long_tok * args.replay_share / (1 - args.replay_share))
    man = json.load(open(hf_hub_download(bl.REPLAY_REPO, "manifest.json", repo_type="dataset", local_dir=dl)))
    sids = man.get("source_ids", [])
    replay_rows, rgot, rshards, per_sub = [], 0, [], collections.Counter()
    for s in bl.REPLAY_SHARDS:
        if rgot >= replay_target:
            break
        key = f"shards/phaseB/shard_{s:05d}.parquet"
        u = used.get(key, {})
        if u.get("whole"):
            continue
        row0 = u.get("rows", 0)
        f = hf_hub_download(bl.REPLAY_REPO, key, repo_type="dataset", local_dir=dl)
        mf = hf_hub_download(bl.REPLAY_REPO, f"_meta/shard_{s:05d}.sources.npy", repo_type="dataset", local_dir=dl)
        src_ids = np.load(mf).tolist()
        texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        os.remove(f)
        assert len(src_ids) == len(texts)
        taken, ttok = 0, 0
        i = row0
        while i < len(texts) and rgot < replay_target:
            chunk = texts[i:i + 4000]
            ns = [len(e.ids) for e in tok.encode_batch(chunk, add_special_tokens=False)]
            for j, (t, n) in enumerate(zip(chunk, ns)):
                if rgot >= replay_target:
                    break
                sub = sids[src_ids[i + j]]
                h = _rh(t)
                if h in prior_h or h in seen_h:
                    replay_dup["in_batch1_or_2" if h in prior_h else "within_batch3"] += 1
                    taken += 1                          # row consumed (skipped), so the next batch resumes after it
                    continue
                seen_h.add(h)
                replay_rows.append((t, f"replay_{sub}", n))
                per_sub[sub] += n
                rgot += n
                ttok += n
                taken += 1
            i += len(chunk)
        rshards.append(dict(shard=key, rows=len(texts), row_start=row0, rows_taken=taken,
                            whole=row0 + taken == len(texts), tokens_taken=ttok))
        del texts
    rep["replay"] = dict(repo=bl.REPLAY_REPO, target_tokens=replay_target, tokens=rgot, docs=len(replay_rows),
                         rows_skipped_exact_dup=dict(replay_dup), prior_batch_row_hashes=len(prior_h),
                         shards=rshards, tokens_per_source=dict(per_sub.most_common()),
                         max_doc_tokens=max([n for _, _, n in replay_rows] or [0]))
    print("replay:", json.dumps(rep["replay"])[:800], flush=True)

    # ---- write train shards + batch 1's val shard last
    rows = long_rows + replay_rows
    del long_rows, replay_rows, modern_rows, reuse_rows
    rng.shuffle(rows)
    tdir = f"{out}/train"
    shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(tdir)
    raw = sum(len(r[0].encode("utf-8", "surrogatepass")) for r in rows)
    n_sh = max(1, round(raw / bl.SHARD_RAW_BYTES))
    bounds = [round(i * len(rows) / n_sh) for i in range(n_sh + 1)]
    idx_shard, idx_row, idx_src, idx_n, shard_info = [], [], [], [], []
    for i in range(n_sh):
        part = rows[bounds[i]:bounds[i + 1]]
        fn = f"{tdir}/shard_{i:05d}.parquet"
        pq.write_table(pa.table({"text": [r[0] for r in part]}), fn, row_group_size=1024, compression="zstd")
        shard_info.append(dict(file=os.path.basename(fn), rows=len(part), tokens=sum(r[2] for r in part),
                               mb=round(os.path.getsize(fn) / 1e6, 1)))
        for j, r in enumerate(part):
            idx_shard.append(i); idx_row.append(j); idx_src.append(r[1]); idx_n.append(r[2])
    shutil.copy(f"{B1}/train/{bl.VAL_NAME}", f"{tdir}/{bl.VAL_NAME}")
    vsha, b1sha = bl.sha256_file(f"{tdir}/{bl.VAL_NAME}"), bl.sha256_file(f"{B1}/train/{bl.VAL_NAME}")
    assert vsha == b1sha
    vrows = pq.ParquetFile(f"{tdir}/{bl.VAL_NAME}").metadata.num_rows
    pq.write_table(pa.table({"shard": idx_shard, "row": idx_row, "src": idx_src, "ntok_v1": idx_n}),
                   f"{out}/train_index.parquet", compression="zstd")
    tot = sum(r[2] for r in rows)
    rep["train"] = dict(docs=len(rows), tokens=tot, long_tokens=long_tok, modern_tokens=modern_tok, reuse_tokens=reuse_tok,
                        replay_tokens=rgot, share_long=round(long_tok / max(1, tot), 4),
                        pool_shares=dict(modern=round(modern_tok / max(1, tot), 4), reuse=round(reuse_tok / max(1, tot), 4),
                                         replay=round(rgot / max(1, tot), 4)),
                        shards=shard_info,
                        val_shard=dict(file=bl.VAL_NAME, source=f"{B1}/train/{bl.VAL_NAME} (= {bl.VAL_REPO}/{bl.VAL_FILE})",
                                       rows=vrows, mb=round(os.path.getsize(f"{tdir}/{bl.VAL_NAME}") / 1e6, 1),
                                       sha256=vsha, batch1_sha256=b1sha),
                        est_steps_at_524288=round(tot / 524288, 1))
    del rows

    # ---- val_long copied from batch 1
    os.makedirs(f"{out}/val_long", exist_ok=True)
    src_vl = f"{B1}/val_long/val_long.parquet"
    shutil.copy(src_vl, f"{out}/val_long/val_long.parquet")
    vt = pq.read_table(f"{out}/val_long/val_long.parquet")
    vn = vt.column("ntok_v1").to_pylist()
    for rp in (f"{B1}/report.json", f"{B2}/report.json"):
        for name, fs in json.load(open(rp)).get("heldout_files", {}).items():
            rep["heldout_files"][name] = sorted(set(rep["heldout_files"].get(name, [])) | set(fs))
    rep["val_long"] = dict(copied_from=src_vl, sha256=bl.sha256_file(f"{out}/val_long/val_long.parquet"),
                           source_sha256=bl.sha256_file(src_vl), docs=len(vn), tokens=sum(vn), min_ntok=min(vn or [0]),
                           per_source=dict(collections.Counter(vt.column("src").to_pylist())))
    rep["wall_seconds"] = round(time.time() - t_start)
    rep["collect_seconds_max_unit"] = max([r.get("seconds", 0) for r in results] or [0])
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1, default=str)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--modern-target", type=float, default=9.45e7)
    ap.add_argument("--reuse-target", type=float, default=2.1e7)
    ap.add_argument("--replay-share", type=float, default=0.45, help="replay / total tokens")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3232)
    ap.add_argument("--unit-time-limit", type=int, default=3000, help="seconds per unit (build_mix run_unit time_limit)")
    ap.add_argument("--stage", default="all", choices=["all", "collect", "assemble"])
    ap.add_argument("--resume", action="store_true", help="skip unit tasks that already have a result")
    ap.add_argument("--dry-run", action="store_true", help="1 unit per source, ~6M tokens total")
    args = ap.parse_args()
    nu = None
    if args.dry_run:
        args.modern_target = min(args.modern_target, 3e6)
        args.reuse_target = min(args.reuse_target, 7e5)
        args.unit_time_limit = min(args.unit_time_limit, 300)
        nu = {n: 1 for n in MODERN}
    args.out = os.path.abspath(args.out or f"{ROOT}/data/udgam2_ctx32k_r3" + ("_dry" if args.dry_run else ""))
    for d in (B1, B2):
        assert os.path.abspath(args.out) != d
    work = f"{args.out}/_work"
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    if not os.path.exists(f"{work}/val_probes_owner_sorted.npy"):
        print(f"val_long: {bl.prepare_val(f'{B1}/val_long/val_long.parquet', work)} docs -> 10-gram arrays", flush=True)
    audit, picks = unit_audit(work, nu)
    json.dump(audit, open(f"{work}/unit_audit.json", "w"), indent=1)
    for n in MODERN:
        print(f"units {n}: n={audit[n]['n_units']} head<= {audit[n]['head_used_upper_bound']} tail {audit[n]['tail_used_by_4k']} "
              f"picked {[p['idx'] for p in audit[n]['picked']]}", flush=True)
    if args.stage in ("all", "collect"):
        collect(args, work, picks)
    if args.stage in ("all", "assemble"):
        rep = assemble(args, work, load_results(work), audit, t0)
        print(json.dumps({k: rep[k] for k in ("train", "errors")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
