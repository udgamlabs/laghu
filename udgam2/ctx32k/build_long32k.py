#!/usr/bin/env python
"""Udgam-2 32K context-extension (P4) training data: long pool + replay pool -> nanochat parquet shards.

README
------
Spec: udgam2/specs/ctx32k.md (user-approved decisions) and udgam2/specs/long_context_data.md (sources, licences).
Runs locally on the Mac (CPU only, $0). Mirrors ops/modal_4k_data.py (long pool + replay pool, chunking, 13-gram
decontamination, held-out long eval docs, report.json), adapted to 32K rows.

    PY=<venv with tokenizers==0.23.2, pyarrow, numpy, huggingface_hub>
    $PY udgam2/ctx32k/build_long32k.py --dry-run                 # ~2M long tokens per bucket -> data/udgam2_ctx32k_dry
    $PY udgam2/ctx32k/build_long32k.py                           # full build (~0.28B long + ~0.28B replay tokens)
    $PY udgam2/ctx32k/test_long32k.py data/udgam2_ctx32k         # validate layout + invariants

Stages (--stage all = collect, then assemble):
  collect   one process per (source, role, file list): stream each common-pile .jsonl.gz over HTTP (one GET, resumed
            with Range on errors, gunzipped on the fly; nothing raw is written to disk), and per record:
            licence (metadata.license) -> English -> source cleaning -> mojibake -> length prefilter (chars) ->
            13-gram decontamination (data/decontam, same index + rule as pretraining) -> exact tokenizer_v1 count ->
            length filter (>= 8,192 tokens; arXiv >= 12K, DOAB >= 16K) -> split docs > 32,000 tokens at paragraph
            breaks (build_mix.chunk_doc) into pieces <= 32,000 tokens, drop tail pieces < 4,096, keep at most
            MAX_PIECES consecutive pieces per document (a seeded random window, so one 1M-token book cannot dominate).
            Each task stops at its token budget (target x OVERSHOOT, the slack covers the cross-file dedupe).
            Code (Stack v2): the whole file is streamed, permissive non-vendor/non-generated/non-minified source files
            of programming languages are spilled to hash(repo) partitions on disk, then each repo's files are
            concatenated (README first, then path order, "==> path <==" headers) into one document.
            Eval tasks read DIFFERENT files and keep full-length docs >= 33,000 tokens (not split).
  assemble  cross-source book dedupe (normalised title + exact hash of the first 4K normalised chars + MinHash bands of
            word 5-grams, priority Gutenberg > pre-1929 > LoC; USGPO / Hansard / wiki near-dup MinHash too), per-source
            token targets (bucket shortfalls are moved to sibling sources), replay pool = whole phase-B shards of
            YOUR_HF_NAMESPACE/udgam-pretrain-v2b from the newest (shard_00101) down (the last one is cut to a row prefix
            so long == replay tokens), global seeded shuffle, shards of ~450 MB raw text (~150-200 MB zstd), and the
            main run's validation shard (UdgamLabs/udgam-pretrain-v2 shards/phaseA/shard_00135.parquet, used as
            shard_zzzzz_val.parquet in the main run's phase B) copied LAST so short-context val bpb stays comparable.
Output (--out, default data/udgam2_ctx32k):
  train/shard_00000.parquet ... + train/shard_zzzzz_val.parquet   (column "text", row groups of 1024, zstd: same as v2b)
  val_long/val_long.parquet       (text, src, ntok_v1): held-out full-length docs >= 33K tokens
  train_index.parquet             (shard, row, src, ntok_v1) for audits
  report.json                     tokens per source/bucket, length histograms, drops by reason, files used, wall time
  _work/                          per-task parts + logs (safe to delete after the build)
Token counts are tokenizer_v1 ids without BOS (the loader prepends BOS; a 32,000-token piece + BOS fits a 32,769 row).

Top-up batches (all options off by default; without them the build is exactly the batch-1 build above):
    $PY udgam2/ctx32k/build_long32k.py --out data/udgam2_ctx32k_t2 --long-target 3.85e8 --long-share 0.7 \
        --sources-json udgam2/ctx32k/sources_t2.json --exclude-report data/udgam2_ctx32k/report.json \
        --copy-val-long data/udgam2_ctx32k/val_long/val_long.parquet --dedupe-against-parts data/udgam2_ctx32k/_work/parts
  --sources-json         per-source overrides {name: {"files": [...], "rec_sources": [...]}} (train files; rec_sources
                         keeps only records whose top-level "source" is listed, e.g. Hansard Commons vs Lords debates)
  --exclude-report       earlier report.json (repeatable): every train / held-out file it lists (and every configured
                         eval file) is forbidden (a listed override file -> error, a default file -> skipped); replay
                         continues after the phase-B rows it used (whole shards skipped, a cut shard resumes at its row)
  --copy-val-long        reuse an earlier val_long.parquet (no eval collection) so eval is identical across stages; every
                         training doc is checked against it with word 10-grams: a doc sharing >= VAL_FRAC_MAX of its
                         distinct 10-grams with val_long is dropped (other editions of a held-out book), and the test's
                         sampled check (20 10-grams per held-out doc) is enforced greedily (< EVAL_OVERLAP_MAX found)
  --dedupe-against-parts earlier _work/parts dir (repeatable): its title / head-hash / MinHash keys seed the dedupe, so
                         no near-duplicate of an earlier batch's collected doc is used again
  --long-share           long / (long + replay) token share (default 0.5); replay target = long tokens x (1-s)/s
"""
import argparse, collections, glob, hashlib, html, json, os, random, re, shutil, sys, time, traceback, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam"))
import build_mix as bm          # noqa: E402  (chunk_doc, mojibake, minhash_bands, PERMISSIVE_CODE)
import decontam                 # noqa: E402

TOKENIZER = f"{ROOT}/artifacts/tokenizer_v1/tokenizer.json"
DECONTAM_INDEX = f"{ROOT}/data/decontam/index_13gram.npz"
CTX_MAX = 32000            # max tokens per training piece
TAIL_MIN = 4096            # tail pieces of a split doc shorter than this are dropped
MIN_DOC = 8192             # default min doc length (tokens)
EVAL_MIN = 33000           # held-out eval docs: full length, >= this many tokens
MAX_PIECES = 6             # at most this many consecutive 32K pieces per document (4 for code repos)
OVERSHOOT = 1.25           # collect budget = target x this (dedupe slack)
REPLAY_REPO = "YOUR_HF_NAMESPACE/udgam-pretrain-v2b"
REPLAY_SHARDS = list(range(101, 81, -1))           # phase-B anneal shards, newest first (00102 is the val shard)
VAL_REPO, VAL_FILE = "UdgamLabs/udgam-pretrain-v2", "shards/phaseA/shard_00135.parquet"
VAL_NAME = "shard_zzzzz_val.parquet"
SHARD_RAW_BYTES = 450e6
CODE_MAX_BYTES = 700 << 20  # per Stack v2 file (compressed bytes)

BUCKET_SHARE = {"books": 0.38, "papers": 0.27, "govwiki": 0.15, "code": 0.20}
# name: bucket, repo, share within bucket, train files (in order, later ones are fallbacks), kind, min tokens
SOURCES = {
    "gutenberg":   dict(bucket="books", repo="common-pile/project_gutenberg", frac=0.45, kind="gutenberg",
                        files=["v0/documents/00003_pg.jsonl.gz", "v0/documents/00004_pg.jsonl.gz"],
                        eval_files=["v0/documents/00029_pg.jsonl.gz"], eval_cap=80, min_tok=MIN_DOC),
    "pre1929":     dict(bucket="books", repo="common-pile/pre_1929_books_filtered", frac=0.30, kind="ocr",
                        files=["public_library_1929_dolma-0001.json.gz", "public_library_1929_dolma-0002.json.gz"],
                        eval_files=["public_library_1929_dolma-0025.json.gz"], eval_cap=50, min_tok=MIN_DOC),
    "loc":         dict(bucket="books", repo="common-pile/library_of_congress", frac=0.25, kind="ocr",
                        files=["data/00001_loc_books.jsonl.gz", "data/00002_loc_books.jsonl.gz"],
                        eval_files=["data/00049_loc_books.jsonl.gz"], eval_cap=40, min_tok=MIN_DOC),
    "arxiv":       dict(bucket="papers", repo="common-pile/arxiv_papers", frac=0.45, kind="paper",
                        files=["00001_arxiv-papers.jsonl.gz", "00002_arxiv-papers.jsonl.gz"],
                        eval_files=["00021_arxiv-papers.jsonl.gz"], eval_cap=30, min_tok=12000),
    "doab":        dict(bucket="papers", repo="common-pile/doab", frac=0.30, kind="doab",
                        files=["00001_doab.json.gz", "00002_doab.json.gz"],
                        eval_files=["00012_doab.json.gz"], eval_cap=40, min_tok=16000),
    "pubmed":      dict(bucket="papers", repo="common-pile/pubmed", frac=0.25, kind="paper",
                        files=["data/00001_pubmedcentral.jsonl.gz", "data/00002_pubmedcentral.jsonl.gz"], min_tok=MIN_DOC),
    "hansard_commons": dict(bucket="govwiki", repo="common-pile/uk_hansard", frac=0.25, kind="hansard",
                        files=["uk_hansard/00003_ukhansard.jsonl.gz", "uk_hansard/00004_ukhansard.jsonl.gz"],
                        eval_files=["uk_hansard/00008_ukhansard.jsonl.gz"], eval_cap=60, min_tok=MIN_DOC),
    "hansard_lords": dict(bucket="govwiki", repo="common-pile/uk_hansard", frac=0.10, kind="hansard",
                        files=["uk_hansard/00001_ukhansard.jsonl.gz"], min_tok=MIN_DOC),
    # USGPO is boilerplate-heavy: only hearings / committee reports / GAO reports / Congressional Record, each capped
    "usgpo_chrg":  dict(bucket="govwiki", repo="common-pile/usgpo", frac=0.12, kind="usgpo", collections={"CHRG"},
                        files=["data/00013_usgpo.jsonl.gz", "data/00035_usgpo.jsonl.gz"], min_tok=MIN_DOC),
    "usgpo_crpt":  dict(bucket="govwiki", repo="common-pile/usgpo", frac=0.08, kind="usgpo", collections={"CRPT"},
                        files=["data/00072_usgpo.jsonl.gz", "data/00010_usgpo.jsonl.gz"], min_tok=MIN_DOC),
    "usgpo_gao":   dict(bucket="govwiki", repo="common-pile/usgpo", frac=0.06, kind="usgpo", collections={"GAOREPORTS"},
                        files=["data/00071_usgpo.jsonl.gz", "data/00070_usgpo.jsonl.gz", "data/00008_usgpo.jsonl.gz"], min_tok=MIN_DOC),
    "usgpo_crec":  dict(bucket="govwiki", repo="common-pile/usgpo", frac=0.04, kind="usgpo", collections={"CREC"},
                        files=["data/00011_usgpo.jsonl.gz", "data/00063_usgpo.jsonl.gz"], min_tok=MIN_DOC),
    "wikipedia":   dict(bucket="govwiki", repo="common-pile/wikimedia", frac=0.21, kind="wiki",
                        files=["00001_wikipedia.com.jsonl.gz", "00002_wikipedia.com.jsonl.gz", "00003_wikipedia.com.jsonl.gz"], min_tok=MIN_DOC),
    "wikisource":  dict(bucket="govwiki", repo="common-pile/wikimedia", frac=0.14, kind="wiki",
                        files=["00001_wikisource.com.jsonl.gz", "00002_wikisource.com.jsonl.gz"], min_tok=MIN_DOC),
    # Stack v2 files are concatenations of ~50-80 MB single-language blocks (the first ~125 files are dominated by
    # huge notebooks / CSV / JSON). Each code task streams the first CODE_MAX_BYTES of files whose first block is a
    # mainstream language (probed 2026-10-12) and moves to the next file until its budget is met.
    "stackv2_a":   dict(bucket="code", repo="common-pile/stackv2", frac=1 / 3, kind="code", min_tok=MIN_DOC,
                        files=["documents/00303_stackv2.jsonl.gz", "documents/00500_stackv2.jsonl.gz",
                               "documents/00553_stackv2.jsonl.gz", "documents/00217_stackv2.jsonl.gz"]),
    "stackv2_b":   dict(bucket="code", repo="common-pile/stackv2", frac=1 / 3, kind="code", min_tok=MIN_DOC,
                        files=["documents/00246_stackv2.jsonl.gz", "documents/00522_stackv2.jsonl.gz",
                               "documents/00330_stackv2.jsonl.gz", "documents/00268_stackv2.jsonl.gz"]),
    "stackv2_c":   dict(bucket="code", repo="common-pile/stackv2", frac=1 / 3, kind="code", min_tok=MIN_DOC,
                        files=["documents/00126_stackv2.jsonl.gz", "documents/00360_stackv2.jsonl.gz",
                               "documents/00402_stackv2.jsonl.gz", "documents/00367_stackv2.jsonl.gz"]),
}
BOOK_PRIORITY = ["gutenberg", "pre1929", "loc"]
DEDUPE_GROUPS = {"books": BOOK_PRIORITY, "hansard": ["hansard_commons", "hansard_lords"],
                 "usgpo": ["usgpo_chrg", "usgpo_crpt", "usgpo_gao", "usgpo_crec"], "wiki": ["wikipedia", "wikisource"],
                 "arxiv": ["arxiv"], "doab": ["doab"], "pubmed": ["pubmed"]}

# ----------------------------------------------------------------------------------------------- streaming
def iter_jsonl_gz(repo, path, stats, chunk=1 << 20, retries=10, max_bytes=0):
    """Stream a .jsonl.gz from a HF dataset repo with one GET (resumed with a Range request on network errors),
    gunzip + split lines on the fly. Nothing is written to disk. stats["bytes"] = compressed bytes read."""
    import httpx
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    url = hf_hub_url(repo, path, repo_type="dataset")
    headers = build_hf_headers()
    pos, d, buf, attempt = 0, zlib.decompressobj(31), b"", 0
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0)) as cl:
        while True:
            h = dict(headers)
            if pos:
                h["Range"] = f"bytes={pos}-"
            try:
                with cl.stream("GET", url, headers=h) as r:
                    if r.status_code in (403, 404):
                        raise FileNotFoundError(f"{repo}/{path}: HTTP {r.status_code}")
                    r.raise_for_status()
                    if pos and r.status_code != 206:
                        raise RuntimeError("server ignored Range header")
                    for raw in r.iter_raw(chunk):
                        pos += len(raw)
                        stats["bytes"] = pos
                        data = d.decompress(raw)
                        while d.unused_data:                    # multi-member gzip
                            rest = d.unused_data
                            d = zlib.decompressobj(31)
                            data += d.decompress(rest)
                        buf += data
                        lines = buf.split(b"\n")
                        buf = lines.pop()
                        for ln in lines:
                            if ln.strip():
                                yield json.loads(ln)
                        if max_bytes and pos >= max_bytes:
                            stats["truncated"] = True
                            return
                    break
            except (httpx.HTTPError, OSError) as e:
                if isinstance(e, FileNotFoundError):
                    raise
                attempt += 1
                stats["retries"] = attempt
                if attempt > retries:
                    raise
                time.sleep(min(60, 2 ** attempt))
    if buf.strip():
        yield json.loads(buf)
    stats["complete"] = True


# ----------------------------------------------------------------------------------------------- cleaning
_WORD = re.compile(r"[A-Za-z]+")
EN_SW = set("the of and to in a is that for it as was with be by on not he i this are or his from at which but have "
            "an they you were her she all their there has been one we its would will had more can if who so what them "
            "other than these may when into also any only some no".split())
FOREIGN_SW = set("der die und das nicht ist ein eine zu den von mit sich auf für le la les et est une dans pour qui pas "
                 "sur el los las que por una con para del het een van ik niet il che di della per non sono och att".split())


def en_ratio(t):
    s = t[:30000] + " " + t[len(t) // 2: len(t) // 2 + 30000]
    w = _WORD.findall(s.lower())
    if len(w) < 200:
        return 0.0, 0
    en = sum(x in EN_SW for x in w)
    fo = sum(x in FOREIGN_SW for x in w)
    return en / len(w), fo / max(1, en)


EN_MIN = {"gutenberg": 0.22, "ocr": 0.20, "doab": 0.22, "paper": 0.12, "hansard": 0.22, "usgpo": 0.12, "wiki": 0.20}
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RULE = re.compile(r"([_\-=.*~])\1{9,}")
_MULTI_NL = re.compile(r"\n{3,}")
_TRAIL_WS = re.compile(r"[ \t]+\n")


def clean_common(t):
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    t = _CTRL.sub("", t)
    t = _RULE.sub(lambda m: m.group(1) * 3, t)        # long rules / dot leaders -> 3 chars
    t = _TRAIL_WS.sub("\n", t)
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()


_DEHYPH = re.compile(r"([A-Za-z])[-\u00ac\u00ad][ \t]*\n[ \t]*([a-z])")
_NOT_SIGN = re.compile(r"([A-Za-z])[\u00ac\u00ad][ \t]+([a-z])")     # inline OCR hyphenation mark, e.g. "gar\u00ac ment"
_SPACES = re.compile(r"[ \t]{2,}")


def clean_ocr(t):
    """OCR books (pre-1929 / LoC): soft-hyphen + line-break dehyphenation, collapse space runs, drop noise lines
    (< 3 letters: stray marks, page numbers), then the common normalisation."""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _DEHYPH.sub(r"\1\2", t)
    t = _NOT_SIGN.sub(r"\1\2", t)
    t = _SPACES.sub(" ", t)
    out = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
        elif sum(c.isalpha() for c in s) >= 3:
            out.append(s)
    return clean_common("\n".join(out))


_PG_START = re.compile(r"^\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)
_PG_END = re.compile(r"^\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)
_PG_SMALLPRINT = re.compile(r"^.*\*END\*THE SMALL PRINT.*$", re.M)
_PG_ENDLINE = re.compile(r"^.*End of (the |this )?Project Gutenberg.*$", re.I | re.M)
_PG_PRODUCED = re.compile(r"^\s*(Produced by|E-?text prepared by|Transcribed (from|by)|Prepared by|Scanned by|"
                          r"This (e-?text|ebook|e-book) (was|is)|Note: Project Gutenberg)", re.I)
_PG_LINE = re.compile(r"Project Gutenberg|gutenberg\.org|gutenberg\.net|pgdp\.net|Distributed Proofread|\.htm\b|\.zip\b|"
                      r"archive\.org|Internet Archive|HTML (version|file)|Images of the original|original illustrations", re.I)


def strip_gutenberg(t):
    """Remove Project Gutenberg licence header/footer + producer credits (common-pile already stripped most)."""
    m = _PG_START.search(t, 0, max(20000, len(t) // 4))
    if m:
        t = t[m.end():]
    m = _PG_SMALLPRINT.search(t, 0, 40000)
    if m:
        t = t[m.end():]
    m = _PG_END.search(t, len(t) * 3 // 4)
    if m:
        t = t[:m.start()]
    m = None
    for m in _PG_ENDLINE.finditer(t, max(0, len(t) - 20000)):
        break
    if m:
        t = t[:m.start()]
    paras = t.lstrip().split("\n\n")
    k = 0
    while k < min(4, len(paras)) and (_PG_PRODUCED.match(paras[k]) or _PG_LINE.search(paras[k])):
        k += 1
    t = "\n\n".join(paras[k:])
    head, tail = t[:3000], t[-3000:] if len(t) > 6000 else ""
    body = t[3000:len(t) - 3000] if len(t) > 6000 else t[3000:]
    head = "\n".join(l for l in head.split("\n") if not _PG_LINE.search(l))
    tail = "\n".join(l for l in tail.split("\n") if not _PG_LINE.search(l))
    return head + body + tail


_USGPO_DROP = re.compile(r"^\s*(<DOC>|</DOC>|\[From the U\.S\. Government (Publishing|Printing) Office.*\]|"
                         r"\[Congressional .*\]|\[[A-Z]\. ?R\..*\])\s*$", re.M)


def norm_title(title):
    s = (title or "").lower()
    s = re.split(r"\s/\s|\s+by\s+|;", s)[0]
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"^(the|a|an) ", "", s)
    return s[:80] if len(s) >= 10 else ""


def head_hash(text):
    s = re.sub(r"[^a-z0-9]+", "", text[:6000].lower())[:4000]
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True) if len(s) >= 500 else 0


# ----------------------------------------------------------------------------------------------- record makers
LICENCE_DATASET_LEVEL = {"gutenberg", "pre1929", "loc", "hansard_commons", "hansard_lords",
                         "usgpo_chrg", "usgpo_crpt", "usgpo_gao", "usgpo_crec"}


def licence_ok(lic, name):
    """True / False from metadata.license; None = missing (dataset-level licence decides)."""
    if not lic:
        return None
    s = str(lic).lower()
    if re.search(r"non-?commercial|no-?deriv|\bnc\b|\bnd\b|-nc|-nd", s):
        return False
    if "public domain" in s or "cc0" in s or "creative commons zero" in s:
        return True
    if "creative commons - attribution" in s or "cc-by" in s or "cc by" in s:
        return True                                 # CC-BY and CC-BY-SA
    if "open parliament licence" in s or "open government licence" in s:
        return True
    return False


def iter_docs(name, cfg, records, drops):
    """Yield (doc_id, title, text) of cleaned, licence- and language-checked docs for a prose source."""
    kind = cfg["kind"]
    if kind == "doab":                                  # sections of one book are consecutive: concatenate them
        records = _group_doab(records)
    for r in records:
        md = r.get("metadata") or {}
        ok = licence_ok(md.get("license"), name)
        if ok is False or (ok is None and name not in LICENCE_DATASET_LEVEL):
            drops["licence"] += 1
            continue
        if ok is None:
            drops["licence_missing_dataset_level_kept"] += 1
        lang = str(md.get("language") or "").lower()
        if lang and lang not in ("en", "eng", "english"):
            drops["not_english_meta"] += 1
            continue
        if kind == "wiki" and str(md.get("namespace", "0")) != "0":
            drops["wiki_not_ns0"] += 1
            continue
        if kind == "usgpo" and r.get("collection") not in cfg["collections"]:
            drops["usgpo_collection"] += 1
            continue
        if cfg.get("rec_sources") and r.get("source") not in cfg["rec_sources"]:
            drops["record_source"] += 1
            continue
        text = r.get("text") or ""
        if len(text) < cfg["min_tok"] * 2.4:              # cheap prefilter (English ~4.2 chars / token)
            drops["length_chars"] += 1
            continue
        if kind == "gutenberg":
            text = clean_common(strip_gutenberg(text))
        elif kind == "ocr":
            text = clean_ocr(text)
        elif kind == "usgpo":
            text = clean_common(_USGPO_DROP.sub("", text))
        elif kind == "wiki":
            text = clean_common(html.unescape(text))
            if "may refer to:" in text[:400]:
                drops["wiki_disambiguation"] += 1
                continue
        else:
            text = clean_common(text)
        if bm.mojibake(text) or "���" in text:
            drops["mojibake"] += 1
            continue
        if bm.has_special(text):                         # literal special-token strings would tokenise as specials
            drops["special_token_literal"] += 1
            continue
        er, fo = en_ratio(text)
        if er < EN_MIN[kind] or fo > 0.33:
            drops["not_english"] += 1
            continue
        if kind == "ocr":
            ns = re.sub(r"\s+", "", text[:200000])
            if ns and sum(c.isalpha() for c in ns) / len(ns) < 0.72:
                drops["ocr_noise"] += 1
                continue
        title = md.get("title") or r.get("title") or ""
        yield str(r.get("id")), title, text


def _group_doab(records):
    cur, secs = None, []
    for r in records:
        md = r.get("metadata") or {}
        b = md.get("book_id")
        if b != cur and secs:
            yield _merge_doab(secs)
            secs = []
        cur = b
        secs.append(r)
    if secs:
        yield _merge_doab(secs)


def _merge_doab(secs):
    secs.sort(key=lambda r: (r.get("metadata") or {}).get("section_idx", 0))
    r = dict(secs[0])
    r["text"] = "\n\n".join((s.get("text") or "").strip() for s in secs)
    r["metadata"] = dict(secs[0].get("metadata") or {})
    lics = {str((s.get("metadata") or {}).get("license")) for s in secs}
    if len(lics) > 1 and not all(licence_ok(l, "doab") for l in lics):
        r["metadata"]["license"] = "mixed-not-ok"
    return r


# ----------------------------------------------------------------------------------------------- code (Stack v2)
CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Shell",
              "Kotlin", "Swift", "Scala", "Lua", "Haskell", "R", "Julia", "Perl", "Dart", "Objective-C", "Markdown",
              "reStructuredText", "Makefile", "CMake", "Dockerfile", "SQL", "Vue", "Elixir", "Clojure", "OCaml",
              "F#", "Erlang", "Zig", "Nim", "TeX", "Fortran", "MATLAB", "Groovy", "PowerShell", "Assembly", "Cuda",
              "Verilog", "VHDL", "Racket", "Scheme", "Common Lisp", "Emacs Lisp", "Elm", "Solidity", "Svelte"}
_CODE_BAD_PATH = re.compile(r"(^|/)(vendor|vendors|node_modules|third[_-]?party|external|dist|build|\.git|"
                            r"data|datasets|fixtures|testdata|test_data)/|\.min\.(js|css)$|\.map$|"
                            r"(^|/)(package-lock|yarn\.lock|composer\.lock|Cargo\.lock)", re.I)
_GEN_MARK = ("auto-generated", "autogenerated", "generated by", "do not edit", "code generated", "@generated")


def code_file_reason(md, text):
    if str(md.get("license") or "").lower() not in bm.PERMISSIVE_CODE:
        return "code_licence"
    det = md.get("detected_licenses") or []
    if any(str(x).lower() not in bm.PERMISSIVE_CODE for x in det):
        return "code_licence_detected"
    if md.get("language") not in CODE_LANGS:
        return "code_language"
    if md.get("is_vendor") or md.get("is_generated"):
        return "code_vendor_generated"
    if _CODE_BAD_PATH.search(md.get("path") or ""):
        return "code_path"
    if len(text) > 100_000 or len(text) < 50:
        return "code_size"
    if text.startswith("version https://git-lfs"):
        return "code_lfs"
    if any(k in text[:800].lower() for k in _GEN_MARK):
        return "code_autogenerated"
    lines = text.split("\n")
    if max(len(l) for l in lines) > 1000 or len(text) / len(lines) > 120:
        return "code_minified"
    if sum(c.isalnum() for c in text) / len(text) < 0.25:
        return "code_data_like"
    if bm.mojibake(text):
        return "mojibake"
    if bm.has_special(text):
        return "special_token_literal"
    if "\ufffd\ufffd\ufffd" in text:
        return "mojibake"
    return None


def repo_doc(repo, files):
    def key(f):
        base = f[0].rsplit("/", 1)[-1].lower()
        return (0 if base.startswith("readme") else 1, f[0])
    parts = [f"Repository: {repo}"]
    for path, text in sorted(files, key=key):
        parts.append(f"==> {path.lstrip('/')} <==\n{text.rstrip()}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------------------------- worker
_W = {}


def _ctx():
    if not _W:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from tokenizers import Tokenizer
        _W["tok"] = Tokenizer.from_file(TOKENIZER)
        _W["ix"] = decontam.Index(DECONTAM_INDEX)
    return _W["tok"], _W["ix"]


def doc_seed(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def make_pieces(text, n, tok, max_pieces):
    """Split a doc > CTX_MAX tokens into <= CTX_MAX pieces at paragraph breaks; keep a seeded random window of at
    most max_pieces consecutive pieces; drop a split doc's tail piece < TAIL_MIN. Returns ([(piece, ntok)], info)."""
    info = collections.Counter()
    if n <= CTX_MAX:
        return [(text, n)], info
    span = max_pieces * CTX_MAX
    if n > span + CTX_MAX:                               # pre-trim very long docs to a window (saves chunking work)
        enc = tok.encode(text, add_special_tokens=False)
        offs = enc.offsets
        rng = random.Random(doc_seed(text[:2000]))
        s_tok = rng.randrange(0, n - span)
        a = offs[s_tok][0]
        j = text.find("\n\n", a, a + 20000)
        a = j + 2 if j >= 0 else a
        e_tok = min(n - 1, s_tok + span + CTX_MAX // 2)
        b = offs[e_tok][1]
        info["tokens_outside_window"] += n - (e_tok - s_tok)
        text = text[a:b]
    pieces = bm.chunk_doc(text, tok, CTX_MAX)
    if len(pieces) > 1 and pieces[-1][1] < TAIL_MIN:
        info["tail_pieces_dropped"] += 1
        info["tail_tokens_dropped"] += pieces[-1][1]
        pieces = pieces[:-1]
    if len(pieces) > max_pieces:
        info["tokens_outside_window"] += sum(m for _, m in pieces[max_pieces:])
        pieces = pieces[:max_pieces]
    return pieces, info


PART_SCHEMA = pa.schema([("doc_id", pa.string()), ("src", pa.string()), ("unit", pa.string()), ("order", pa.int64()),
                         ("title_key", pa.string()), ("head_hash", pa.int64()), ("mh", pa.list_(pa.int64())),
                         ("ntok_doc", pa.int64()), ("pieces", pa.list_(pa.string())), ("piece_ntok", pa.list_(pa.int32())),
                         # --copy-val-long only: distinct word 10-grams of the pieces, how many are in val_long, and
                         # which sampled val_long probes (indices into the sorted probe array) the pieces contain
                         ("vng", pa.int64()), ("vh", pa.int64()), ("vprobe", pa.list_(pa.int32()))])


def _val_arrays(vdir):
    """Memory-mapped val_long 10-gram arrays written by prepare_val() (shared page cache across workers)."""
    if _W.get("vdir") != vdir:
        _W["vdir"] = vdir
        _W["vgrams"] = np.load(f"{vdir}/val_ngrams.npy", mmap_mode="r")
        _W["vprobes"] = np.load(f"{vdir}/val_probes_sorted.npy")
    return _W["vgrams"], _W["vprobes"]


def val_stats(pieces, vdir):
    """(distinct 10-grams of the pieces, how many occur in val_long, sorted probe indices hit). Pieces are hashed
    separately, like the test does for training rows."""
    if not vdir:
        return {}
    V, P = _val_arrays(vdir)
    gs = [decontam.ngram_hashes(p, n=10)[0] for p in pieces]
    g = np.unique(np.concatenate(gs)) if gs else np.empty(0, np.uint64)
    if g.size == 0:
        return dict(vng=0, vh=0, vprobe=[])
    pos = np.searchsorted(V, g)
    ok = pos < V.size
    vh = int((V[pos[ok]] == g[ok]).sum())
    lo, hi = np.searchsorted(P, g, "left"), np.searchsorted(P, g, "right")
    m = hi > lo
    vp = sorted({k for a, b in zip(lo[m].tolist(), hi[m].tolist()) for k in range(a, b)})
    return dict(vng=int(g.size), vh=vh, vprobe=vp)


class PartWriter:
    def __init__(self, path):
        self.path, self.rows, self.w = path, [], None

    def add(self, **row):
        self.rows.append(row)
        if len(self.rows) >= 64:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        t = pa.Table.from_pylist(self.rows, schema=PART_SCHEMA)
        if self.w is None:
            self.w = pq.ParquetWriter(self.path, PART_SCHEMA, compression="zstd")
        self.w.write_table(t)
        self.rows = []

    def close(self):
        self.flush()
        if self.w is None:
            pq.write_table(pa.Table.from_pylist([], schema=PART_SCHEMA), self.path, compression="zstd")
        else:
            self.w.close()


def mh_signed(text):
    b = bm.minhash_bands(text)
    return np.array(b, dtype=np.uint64).view(np.int64).tolist() if b else []


def run_task(task):
    """Collect one (source, role) over its file list until the budget is met. Writes a part parquet."""
    t0 = time.time()
    name, role, budget, out, max_bytes = task["name"], task["role"], task["budget"], task["out"], task.get("max_bytes", 0)
    cfg = dict(SOURCES[name], **task.get("cfg_over", {}))
    tok, ix = _ctx()
    drops, info = collections.Counter(), collections.Counter()
    files_used, got, kept, order = [], 0, 0, 0
    w = PartWriter(out)
    min_tok = EVAL_MIN if role == "eval" else cfg["min_tok"]
    cfg_eff = dict(cfg, min_tok=min_tok)
    try:
        if cfg["kind"] == "code":
            res = _run_code(task, cfg, tok, ix, w, drops, info, files_used)
            got, kept = res
        else:
            for f in task["files"]:
                if got >= budget:
                    break
                st = {"bytes": 0}
                fu = dict(file=f, records=0, kept_docs=0, kept_tokens=0)
                files_used.append(fu)
                def recs():
                    for r in iter_jsonl_gz(cfg["repo"], f, st, max_bytes=max_bytes):
                        fu["records"] += 1
                        yield r
                for doc_id, title, text in iter_docs(name, cfg_eff, recs(), drops):
                    hit, h, srcs = ix.is_contaminated(text)
                    if hit:
                        drops["decontam"] += 1
                        info["decontam_" + "+".join(srcs)] += 1
                        continue
                    n = len(tok.encode(text, add_special_tokens=False).ids)
                    if n < min_tok:
                        drops["length_tokens"] += 1
                        continue
                    if role == "eval":
                        pieces, pinfo = [(text, n)], {}
                    else:
                        pieces, pinfo = make_pieces(text, n, tok, MAX_PIECES)
                        info.update(pinfo)
                    tk = sum(m for _, m in pieces)
                    w.add(doc_id=doc_id, src=name, unit=f, order=order, title_key=norm_title(title),
                          head_hash=head_hash(text), mh=mh_signed(text), ntok_doc=n,
                          pieces=[p for p, _ in pieces], piece_ntok=[m for _, m in pieces],
                          **(val_stats([p for p, _ in pieces], task.get("val_dir")) if role == "train" else {}))
                    order += 1
                    got += tk if role == "train" else 1
                    kept += 1
                    fu["kept_docs"] += 1
                    fu["kept_tokens"] += tk
                    if got >= budget:
                        break
                fu["bytes_read"] = st["bytes"]
                fu["complete"] = bool(st.get("complete"))
                fu["retries"] = st.get("retries", 0)
    finally:
        w.close()
    return dict(name=name, role=role, budget=budget, got=got, kept_docs=kept, files=files_used,
                drops=dict(drops), info=dict(info), seconds=round(time.time() - t0), part=out)


def _run_code(task, cfg, tok, ix, w, drops, info, files_used):
    """Stack v2, per file: spill accepted source files to hash(repo) partitions on disk, then build one
    repo-concatenated document per repo (repos live in one file). Next file only if the budget is not met."""
    NP = 16
    sch = pa.schema([("repo", pa.string()), ("path", pa.string()), ("lang", pa.string()), ("text", pa.string())])
    got, kept, order = 0, 0, 0
    repo_lang = collections.Counter()
    rng = random.Random(1234)
    min_tok = task.get("min_tok", MIN_DOC)
    lang_cap = 0.30 * task["budget"]                      # no single main language above 30% of a code task's tokens
    for f in task["files"]:
        if got >= task["budget"]:
            break
        spill = task["out"] + ".spill"
        shutil.rmtree(spill, ignore_errors=True)
        os.makedirs(spill)
        writers, bufs, langs = [None] * NP, [[] for _ in range(NP)], collections.Counter()

        def flush(k):
            if bufs[k]:
                if writers[k] is None:
                    writers[k] = pq.ParquetWriter(f"{spill}/p{k:02d}.parquet", sch, compression="zstd")
                writers[k].write_table(pa.Table.from_pylist(bufs[k], schema=sch))
                bufs[k] = []
        st = {"bytes": 0}
        fu = dict(file=f, records=0)
        files_used.append(fu)
        all_langs = collections.Counter()
        for r in iter_jsonl_gz(cfg["repo"], f, st, max_bytes=task.get("max_bytes", 0)):
            fu["records"] += 1
            md = r.get("metadata") or {}
            text = r.get("text") or ""
            all_langs[md.get("language")] += len(text)
            why = code_file_reason(md, text)
            if why:
                drops[why] += 1
                continue
            repo = md.get("repo_name") or ""
            k = zlib.crc32(repo.encode()) % NP
            bufs[k].append(dict(repo=repo, path=md.get("path") or md.get("filename") or "", lang=md["language"], text=text))
            langs[md["language"]] += 1
            if len(bufs[k]) >= 2000:
                flush(k)
        for k in range(NP):
            flush(k)
            if writers[k] is not None:
                writers[k].close()
        fu["bytes_read"] = st["bytes"]
        fu["complete"] = bool(st.get("complete"))
        fu["retries"] = st.get("retries", 0)
        fu["accepted_files_by_lang"] = dict(langs.most_common(25))
        fu["raw_chars_by_lang_top"] = dict(all_langs.most_common(15))
        parts = sorted(glob.glob(f"{spill}/p*.parquet"))
        rng.shuffle(parts)
        fu["kept_docs"], fu["kept_tokens"] = 0, 0
        for p in parts:
            if got >= task["budget"]:
                break
            t = pq.read_table(p)
            by = collections.defaultdict(list)
            for repo, path, lang, text in zip(*(t.column(c).to_pylist() for c in ("repo", "path", "lang", "text"))):
                by[repo].append((path, lang, text))
            del t
            repos = [r for r, fs in by.items() if sum(len(x[2]) for x in fs) >= min_tok * 2.2]
            drops["code_repo_too_short"] += len(by) - len(repos)
            repos.sort()
            rng.shuffle(repos)
            for repo in repos:
                fs = by[repo]
                main_lang = collections.Counter(l for _, l, _ in fs).most_common(1)[0][0]
                if repo_lang[main_lang] >= lang_cap:
                    drops["code_language_cap"] += 1
                    continue
                text = repo_doc(repo, [(pth, tx) for pth, _, tx in fs])
                hit, h, srcs = ix.is_contaminated(text)
                if hit:
                    drops["decontam"] += 1
                    continue
                n = len(tok.encode(text, add_special_tokens=False).ids)
                if n < min_tok:
                    drops["length_tokens"] += 1
                    continue
                pieces, pinfo = make_pieces(text, n, tok, 4)
                info.update(pinfo)
                tk = sum(m for _, m in pieces)
                repo_lang[main_lang] += tk
                w.add(doc_id=repo, src=task["name"], unit=f, order=order, title_key="", head_hash=head_hash(text),
                      mh=[], ntok_doc=n, pieces=[x for x, _ in pieces], piece_ntok=[m for _, m in pieces],
                      **val_stats([x for x, _ in pieces], task.get("val_dir")))
                order += 1
                got += tk
                kept += 1
                fu["kept_docs"] += 1
                fu["kept_tokens"] += tk
                if got >= task["budget"]:
                    break
        shutil.rmtree(spill, ignore_errors=True)
    info["tokens_by_main_language"] = dict(repo_lang.most_common())
    return got, kept


# ----------------------------------------------------------------------------------------------- collect stage
def source_targets(long_target):
    return {n: long_target * BUCKET_SHARE[c["bucket"]] * c["frac"] for n, c in SOURCES.items()}


def collect(args, work):
    os.makedirs(f"{work}/parts", exist_ok=True)
    tg = source_targets(args.long_target)
    tasks = []
    for name, cfg in SOURCES.items():
        t = dict(name=name, role="train", budget=tg[name] * OVERSHOOT, files=args.train_files.get(name, cfg["files"]),
                 out=f"{work}/parts/train_{name}.parquet", min_tok=cfg["min_tok"])
        if args.cfg_over.get(name):
            t["cfg_over"] = args.cfg_over[name]
        if args.copy_val_long:
            t["val_dir"] = work
        if cfg["kind"] == "code":
            t["max_bytes"] = args.code_max_bytes or CODE_MAX_BYTES
        if not t["files"]:
            print(f"collect: {name} has no train files left after exclusions", flush=True)
            continue
        tasks.append(t)
        if cfg.get("eval_files") and not args.copy_val_long:
            cap = cfg["eval_cap"] if not args.dry_run else 3
            tasks.append(dict(name=name, role="eval", budget=int(cap * 1.6) + 1, files=cfg["eval_files"],
                              out=f"{work}/parts/eval_{name}.parquet", max_bytes=args.eval_max_bytes))
    if args.only:
        tasks = [t for t in tasks if t["name"] in args.only.split(",")]
    prev = [r for r in load_results(work) if not r.get("error")]
    run_keys = {(t["name"], t["role"]) for t in tasks}
    if args.resume:                                     # reuse finished tasks (same budgets) from an earlier collect
        done = {(r["name"], r["role"]) for r in prev if os.path.exists(r.get("part", ""))}
        tasks = [t for t in tasks if (t["name"], t["role"]) not in done]
        run_keys = {(t["name"], t["role"]) for t in tasks}
    results = [r for r in prev if (r["name"], r["role"]) not in run_keys]
    print(f"collect: {len(tasks)} tasks to run, {len(results)} earlier results kept", flush=True)
    tasks.sort(key=lambda t: (SOURCES[t["name"]]["kind"] != "code", t["role"] != "train"))
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                r = dict(name=t["name"], role=t["role"], error=repr(e), tb=traceback.format_exc()[-2000:])
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("tb",)}, default=str)[:600], flush=True)
            save_result(work, r)
    return load_results(work)


def save_result(work, r):
    """One JSON per task (safe with several collect processes) + a merged collect_results.json."""
    os.makedirs(f"{work}/results", exist_ok=True)
    json.dump(r, open(f"{work}/results/{r['name']}__{r['role']}.json", "w"), indent=1, default=str)
    tmp = f"{work}/collect_results.json.tmp{os.getpid()}"
    json.dump(load_results(work), open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, f"{work}/collect_results.json")


def load_results(work):
    res = {}
    if os.path.exists(f"{work}/collect_results.json"):
        for r in json.load(open(f"{work}/collect_results.json")):
            res[(r["name"], r["role"])] = r
    for f in sorted(glob.glob(f"{work}/results/*.json")):
        r = json.load(open(f))
        res[(r["name"], r["role"])] = r
    return list(res.values())


# ----------------------------------------------------------------------------------------------- assemble stage
def _hit_bands(mh, seen):
    return any((i, b) in seen for i, b in enumerate(mh))


def assemble(args, work, results, t_start):
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tok = Tokenizer.from_file(TOKENIZER)
    rng = random.Random(args.seed)
    out = args.out
    res_by = {(r["name"], r["role"]): r for r in results}
    rep = dict(args=dict(long_target=args.long_target, seed=args.seed, dry_run=args.dry_run, ctx_max=CTX_MAX,
                         tail_min=TAIL_MIN, min_doc=MIN_DOC, eval_min=EVAL_MIN, max_pieces=MAX_PIECES,
                         overshoot=OVERSHOOT, bucket_share=BUCKET_SHARE, long_share=args.long_share,
                         replay_target=args.replay_target, sources_json=args.sources_json,
                         exclude_report=args.exclude_report, copy_val_long=args.copy_val_long,
                         dedupe_against_parts=args.dedupe_against_parts),
               sources={}, drops={}, files_used={}, heldout_files={}, errors=[])
    if args.exclude_report:
        rep["excluded"] = dict(files=sorted(f"{r}/{f}" for r, f in args.excluded_files),
                               skipped_default_files=args.skipped_default_files, replay_rows_used=args.replay_used)
    for r in results:
        if r.get("error"):
            rep["errors"].append(dict(name=r["name"], role=r["role"], error=r["error"]))

    # ---- load train parts and dedupe
    docs = {}
    for name in SOURCES:
        p = f"{work}/parts/train_{name}.parquet"
        docs[name] = pq.read_table(p).to_pylist() if os.path.exists(p) else []
        docs[name].sort(key=lambda d: d["order"])
    dedupe_drops = collections.Counter()
    for name in SOURCES:                                 # U+FFFD runs (decoding damage) -> drop the doc
        n0 = len(docs[name])
        docs[name] = [d for d in docs[name] if not any("\ufffd\ufffd\ufffd" in p for p in d["pieces"])]
        if len(docs[name]) < n0:
            dedupe_drops[f"{name}:replacement_char"] += n0 - len(docs[name])
    val_on = bool(args.copy_val_long)
    vrep = dict(frac_max=VAL_FRAC_MAX, hits_min=VAL_HITS_MIN, dropped_docs={}, dropped_tokens={}, frac_hist={})
    if val_on:                                           # other editions / copies of a held-out val_long doc
        fh = collections.Counter()
        for name in SOURCES:
            keep = []
            for d in docs[name]:
                fr = (d["vh"] or 0) / max(1, d["vng"] or 0)
                fh["0" if fr == 0 else "<0.1%" if fr < 1e-3 else "<1%" if fr < 1e-2 else "<10%" if fr < 0.1 else ">=10%"] += 1
                if fr >= VAL_FRAC_MAX and (d["vh"] or 0) >= VAL_HITS_MIN:
                    vrep["dropped_docs"][name] = vrep["dropped_docs"].get(name, 0) + 1
                    vrep["dropped_tokens"][name] = vrep["dropped_tokens"].get(name, 0) + sum(d["piece_ntok"])
                    continue
                keep.append(d)
            docs[name] = keep
        vrep["frac_hist"] = dict(fh)
    prior = {g: (set(), set(), set()) for g in DEDUPE_GROUPS}
    prior_code = set()
    group_of = {n: g for g, m in DEDUPE_GROUPS.items() for n in m}
    for pdir in args.dedupe_against_parts:               # keys of an earlier batch's collected docs
        for p in sorted(glob.glob(f"{pdir}/train_*.parquet")):
            for x in pq.read_table(p, columns=["src", "title_key", "head_hash", "mh"]).to_pylist():
                if x["src"] in group_of:
                    pst, psh, psm = prior[group_of[x["src"]]]
                    if group_of[x["src"]] == "books" and x["title_key"]:
                        pst.add(x["title_key"])
                    if x["head_hash"]:
                        psh.add(x["head_hash"])
                    psm.update(enumerate(x["mh"] or []))
                elif x["head_hash"]:
                    prior_code.add(x["head_hash"])
    seen_title, seen_head, seen_mh = {}, {}, {}
    for group, members in DEDUPE_GROUPS.items():
        st, sh, sm = set(), set(), set()
        pst, psh, psm = prior[group]
        for name in members:
            keep = []
            for d in docs.get(name, []):
                tk, hh, mh = d["title_key"], d["head_hash"], d["mh"] or []
                if group == "books" and tk and tk in pst:
                    dedupe_drops[f"{name}:prior_title"] += 1
                    continue
                if (hh and hh in psh) or (mh and _hit_bands(mh, psm)):
                    dedupe_drops[f"{name}:prior_near_dup"] += 1
                    continue
                if group == "books" and tk and tk in st:
                    dedupe_drops[f"{name}:title"] += 1
                    continue
                if hh and hh in sh:
                    dedupe_drops[f"{name}:head_hash"] += 1
                    continue
                if mh and _hit_bands(mh, sm):
                    dedupe_drops[f"{name}:minhash"] += 1
                    continue
                if group == "books" and tk:
                    st.add(tk)
                if hh:
                    sh.add(hh)
                sm.update(enumerate(mh))
                keep.append(d)
            docs[name] = keep
        seen_title[group], seen_head[group], seen_mh[group] = st, sh, sm
    code_heads = set()                                   # exact repo-doc duplicates in code
    for name in [n for n in SOURCES if SOURCES[n]["kind"] == "code"]:
        keep = []
        for d in docs.get(name, []):
            if d["head_hash"] and d["head_hash"] in prior_code:
                dedupe_drops[f"{name}:prior_head_hash"] += 1
                continue
            if d["head_hash"] and d["head_hash"] in code_heads:
                dedupe_drops[f"{name}:head_hash"] += 1
                continue
            code_heads.add(d["head_hash"])
            keep.append(d)
        docs[name] = keep

    # ---- per-source targets, bucket shortfall moved to siblings with spare tokens
    tg = source_targets(args.long_target)
    avail = {n: sum(sum(d["piece_ntok"]) for d in docs[n]) for n in SOURCES}
    final_tg = dict(tg)
    for b in BUCKET_SHARE:
        mem = [n for n in SOURCES if SOURCES[n]["bucket"] == b]
        short = sum(max(0, tg[n] - avail[n]) for n in mem)
        for n in mem:
            final_tg[n] = min(tg[n], avail[n])
        spare = {n: avail[n] - final_tg[n] for n in mem if avail[n] > final_tg[n]}
        tot_spare = sum(spare.values())
        if short > 0 and tot_spare > 0:
            for n, s in spare.items():
                final_tg[n] += min(s, short * s / tot_spare)
    long_rows, docs_used_ids = [], {}
    hist_doc = collections.Counter()
    if val_on:                                           # the test's check: < EVAL_OVERLAP_MAX probes found per val doc
        probe_owner = np.load(f"{work}/val_probes_owner_sorted.npy")
        probe_found, owner_found = set(), collections.Counter()
        vrep["probe_dropped_docs"] = {}
    for name in SOURCES:
        got, nd, npieces = 0, 0, 0
        ids = []
        for d in docs[name]:
            if got >= final_tg[name]:
                break
            if val_on:
                new = [k for k in (d["vprobe"] or []) if k not in probe_found]
                add = collections.Counter(int(probe_owner[k]) for k in new)
                if any(owner_found[o] + c >= EVAL_OVERLAP_MAX for o, c in add.items()):
                    vrep["probe_dropped_docs"][name] = vrep["probe_dropped_docs"].get(name, 0) + 1
                    continue
                probe_found.update(new)
                owner_found.update(add)
            nd += 1
            ids.append(d["doc_id"])
            n = d["ntok_doc"]
            hist_doc["8K-16K" if n < 16384 else "16K-32K" if n <= CTX_MAX else "32K-64K" if n < 65536 else
                     "64K-128K" if n < 131072 else ">=128K"] += 1
            for p, m in zip(d["pieces"], d["piece_ntok"]):
                long_rows.append((p, f"long_{name}", m))
                got += m
                npieces += 1
        docs_used_ids[name] = ids
        r = res_by.get((name, "train"), {})
        rep["sources"][name] = dict(bucket=SOURCES[name]["bucket"], target_tokens=round(tg[name]),
                                    final_target_tokens=round(final_tg[name]), tokens=got, docs=nd, pieces=npieces,
                                    available_tokens_after_dedupe=avail[name], collected_docs=r.get("kept_docs"),
                                    collect_seconds=r.get("seconds"), info=r.get("info", {}))
        rep["drops"][name] = dict(r.get("drops", {}))
        rep["drops"][name]["dedupe"] = sum(v for k, v in dedupe_drops.items() if k.startswith(name + ":"))
        rep["files_used"][name] = r.get("files", [])
        del docs[name]
    rep["dedupe_detail"] = dict(dedupe_drops)
    long_tok = sum(m for _, _, m in long_rows)
    bucket_tok = collections.Counter()
    for _, s, m in long_rows:
        bucket_tok[SOURCES[s[5:]]["bucket"]] += m
    rep["long"] = dict(tokens=long_tok, pieces=len(long_rows),
                       buckets={b: dict(tokens=bucket_tok[b], share=round(bucket_tok[b] / max(1, long_tok), 4),
                                        target_share=BUCKET_SHARE[b]) for b in BUCKET_SHARE},
                       piece_length_hist=_piece_hist([m for _, _, m in long_rows]),
                       piece_length_hist_tokens=_piece_hist([m for _, _, m in long_rows], weight=True),
                       doc_length_hist_before_split=dict(hist_doc),
                       max_piece_tokens=max([m for _, _, m in long_rows] or [0]))
    print(json.dumps(rep["long"], indent=1), flush=True)
    if val_on:
        vrep["probes_found"] = len(probe_found)
        vrep["max_probes_found_per_val_doc"] = max(owner_found.values() or [0])
        rep["val_overlap"] = vrep
        print("val_overlap:", json.dumps(vrep), flush=True)
    overlap = eval_overlap(work, long_rows) if not val_on else {}

    # ---- replay pool: whole phase-B shards (newest first), last one cut to a row prefix so replay == long tokens
    dl = f"{work}/dl"
    os.makedirs(dl, exist_ok=True)
    replay_target = round(long_tok * (1 - args.long_share) / args.long_share) if args.replay_target <= 0 else args.replay_target
    man = json.load(open(hf_hub_download(REPLAY_REPO, "manifest.json", repo_type="dataset", local_dir=dl)))
    sids = man.get("source_ids", [])
    replay_rows, rgot, rshards, per_sub = [], 0, [], collections.Counter()
    for s in REPLAY_SHARDS:
        if rgot >= replay_target:
            break
        key = f"shards/phaseB/shard_{s:05d}.parquet"
        used = args.replay_used.get(key, {})
        if used.get("whole"):
            continue
        row0 = used.get("rows", 0)
        f = hf_hub_download(REPLAY_REPO, f"shards/phaseB/shard_{s:05d}.parquet", repo_type="dataset", local_dir=dl)
        try:
            mf = hf_hub_download(REPLAY_REPO, f"_meta/shard_{s:05d}.sources.npy", repo_type="dataset", local_dir=dl)
            src_ids = np.load(mf).tolist()
        except Exception:
            src_ids = None
        texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        os.remove(f)
        ntoks = []
        for i in range(0, len(texts), 4000):
            ntoks += [len(e.ids) for e in tok.encode_batch(texts[i:i + 4000], add_special_tokens=False)]
        taken = 0
        for i, (t, n) in enumerate(zip(texts, ntoks)):
            if i < row0:
                continue
            if rgot >= replay_target:
                break
            sub = sids[src_ids[i]] if (src_ids is not None and sids) else "?"
            replay_rows.append((t, f"replay_{sub}", n))
            per_sub[sub] += n
            rgot += n
            taken += 1
        rshards.append(dict(shard=key, rows=len(texts), row_start=row0, rows_taken=taken,
                            whole=taken == len(texts), tokens_taken=sum(ntoks[row0:row0 + taken]),
                            tokens_in_shard=sum(ntoks)))
        del texts
    rep["replay"] = dict(repo=REPLAY_REPO, target_tokens=replay_target, tokens=rgot, docs=len(replay_rows),
                         shards=rshards, tokens_per_source=dict(per_sub.most_common()),
                         max_doc_tokens=max([n for _, _, n in replay_rows] or [0]),
                         docs_over_ctx_max=sum(n > CTX_MAX for _, _, n in replay_rows))

    # ---- write train shards + main val shard last
    rows = long_rows + replay_rows
    del long_rows, replay_rows
    rng.shuffle(rows)
    tdir = f"{out}/train"
    shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(tdir)
    raw = sum(len(r[0].encode("utf-8", "surrogatepass")) for r in rows)
    n_sh = max(1, round(raw / SHARD_RAW_BYTES))
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
    vf = hf_hub_download(VAL_REPO, VAL_FILE, repo_type="dataset", local_dir=dl)
    shutil.copy(vf, f"{tdir}/{VAL_NAME}")
    os.remove(vf)
    vrows = pq.ParquetFile(f"{tdir}/{VAL_NAME}").metadata.num_rows
    pq.write_table(pa.table({"shard": idx_shard, "row": idx_row, "src": idx_src, "ntok_v1": idx_n}),
                   f"{out}/train_index.parquet", compression="zstd")
    tot = sum(r[2] for r in rows)
    rep["train"] = dict(docs=len(rows), tokens=tot, long_tokens=long_tok, replay_tokens=rgot,
                        share_long=round(long_tok / max(1, tot), 4), shards=shard_info,
                        val_shard=dict(file=VAL_NAME, source=f"{VAL_REPO}/{VAL_FILE}", rows=vrows,
                                       mb=round(os.path.getsize(f"{tdir}/{VAL_NAME}") / 1e6, 1),
                                       sha256=sha256_file(f"{tdir}/{VAL_NAME}")),
                        est_steps_at_524288=round(tot / 524288, 1))
    del rows

    if val_on:                                           # ---- val_long copied from an earlier batch
        os.makedirs(f"{out}/val_long", exist_ok=True)
        shutil.copy(args.copy_val_long, f"{out}/val_long/val_long.parquet")
        vt = pq.read_table(f"{out}/val_long/val_long.parquet")
        vn = vt.column("ntok_v1").to_pylist()
        for rp in args.exclude_report:
            for name, fs in json.load(open(rp)).get("heldout_files", {}).items():
                rep["heldout_files"][name] = sorted(set(rep["heldout_files"].get(name, [])) | set(fs))
        rep["val_long"] = dict(copied_from=args.copy_val_long, sha256=sha256_file(f"{out}/val_long/val_long.parquet"),
                               source_sha256=sha256_file(args.copy_val_long), docs=len(vn), tokens=sum(vn),
                               min_ntok=min(vn or [0]),
                               per_source=dict(collections.Counter(vt.column("src").to_pylist())))
        rep["wall_seconds"] = round(time.time() - t_start)
        rep["collect_seconds_max"] = max([r.get("seconds", 0) for r in results] or [0])
        json.dump(rep, open(f"{out}/report.json", "w"), indent=1, default=str)
        return rep

    # ---- held-out long eval docs (other files; not near-dups of any training doc)
    val, vrep = [], {}
    for name, cfg in SOURCES.items():
        if not cfg.get("eval_files"):
            continue
        p = f"{work}/parts/eval_{name}.parquet"
        if not os.path.exists(p):
            continue
        ev = sorted(pq.read_table(p).to_pylist(), key=lambda d: d["order"])
        cap = cfg["eval_cap"] if not args.dry_run else 3
        group = next(g for g, m in DEDUPE_GROUPS.items() if name in m)
        st, sh, sm = seen_title[group], seen_head[group], seen_mh[group]
        sel, dup, ovl = [], 0, []
        for d in ev:
            if (group == "books" and d["title_key"] and d["title_key"] in st) or (d["head_hash"] and d["head_hash"] in sh) \
                    or (d["mh"] and _hit_bands(d["mh"], sm)):
                dup += 1
                continue
            if overlap.get((name, d["order"]), 0) >= EVAL_OVERLAP_MAX:
                ovl.append(dict(hits=overlap[(name, d["order"])], head=d["pieces"][0][:160]))
                continue
            st.add(d["title_key"]) if d["title_key"] else None
            sm.update(enumerate(d["mh"] or []))
            sel.append((d["pieces"][0], name, d["ntok_doc"]))
            if len(sel) >= cap:
                break
        val += sel
        r = res_by.get((name, "eval"), {})
        vrep[name] = dict(docs=len(sel), cap=cap, dropped_as_dup_of_train=dup, dropped_as_overlap_with_train=ovl,
                          files=r.get("files", []),
                          median_ntok=int(np.median([n for _, _, n in sel])) if sel else 0, drops=r.get("drops", {}))
        rep["heldout_files"][name] = [f["file"] for f in r.get("files", [])]
    rng.shuffle(val)
    os.makedirs(f"{out}/val_long", exist_ok=True)
    pq.write_table(pa.table({"text": [v[0] for v in val], "src": [v[1] for v in val],
                             "ntok_v1": pa.array([v[2] for v in val], pa.int64())}),
                   f"{out}/val_long/val_long.parquet", compression="zstd")
    rep["val_long"] = dict(docs=len(val), per_source=vrep, min_ntok=min([v[2] for v in val] or [0]),
                           tokens=sum(v[2] for v in val))
    rep["wall_seconds"] = round(time.time() - t_start)
    rep["collect_seconds_max"] = max([r.get("seconds", 0) for r in results] or [0])
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1, default=str)
    return rep


EVAL_PROBES, EVAL_OVERLAP_MAX = 20, 3      # held-out doc rejected if >= 3 of its 20 sampled word 10-grams are in train
VAL_FRAC_MAX, VAL_HITS_MIN = 0.02, 50      # --copy-val-long: train doc dropped if >= 2% (and >= 50) of its 10-grams are in val_long


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def prepare_val(path, work):
    """--copy-val-long: all distinct word 10-grams of the val_long docs (sorted) + the test's sampled probes
    (20 per doc, np.random.default_rng(doc index), as in test_long32k.py), sorted, with their owner doc index."""
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    allg, probes, owner = [], [], []
    for k, t in enumerate(texts):
        g, _ = decontam.ngram_hashes(t, n=10)
        allg.append(np.unique(g))
        if g.size:
            pick = np.random.default_rng(k).choice(g.size, size=min(EVAL_PROBES, g.size), replace=False)
            probes += g[pick].tolist()
            owner += [k] * len(pick)
    np.save(f"{work}/val_ngrams.npy", np.unique(np.concatenate(allg)))
    pr = np.array(probes, dtype=np.uint64)
    o = np.argsort(pr, kind="stable")
    np.save(f"{work}/val_probes_sorted.npy", pr[o])
    np.save(f"{work}/val_probes_owner_sorted.npy", np.array(owner, dtype=np.int32)[o])
    return len(texts)


def load_exclusions(paths):
    """Files (repo, path) listed as train / held-out in earlier reports + every configured eval file, and the replay
    rows they used ({shard: {"rows": prefix rows used, "whole": bool}})."""
    files, replay = set(), {}
    for name, cfg in SOURCES.items():
        for f in cfg.get("eval_files", []):
            files.add((cfg["repo"], f))
    for p in paths:
        r = json.load(open(p))
        for name, fs in r.get("files_used", {}).items():
            files.update((SOURCES[name]["repo"], f["file"]) for f in fs)
        for name, fs in r.get("heldout_files", {}).items():
            files.update((SOURCES[name]["repo"], f) for f in fs)
        for name, v in (r.get("val_long", {}).get("per_source") or {}).items():
            if isinstance(v, dict):
                files.update((SOURCES[name]["repo"], f["file"]) for f in v.get("files", []))
        for sh in r.get("replay", {}).get("shards", []):
            u = replay.setdefault(sh["shard"], dict(rows=0, whole=False))
            u["rows"] = max(u["rows"], sh.get("row_start", 0) + sh["rows_taken"])
            u["whole"] = u["whole"] or u["rows"] >= sh["rows"]
    return files, replay


def eval_overlap(work, long_rows):
    """Sample EVAL_PROBES word 10-grams per held-out candidate and count how many occur anywhere in the training
    long rows (catches other editions / reprints that title + MinHash dedupe missed). Returns {(src, order): hits}."""
    probes, owner = [], []
    for p in sorted(glob.glob(f"{work}/parts/eval_*.parquet")):
        for d in pq.read_table(p, columns=["src", "order", "pieces"]).to_pylist():
            g, _ = decontam.ngram_hashes(d["pieces"][0], n=10)
            if g.size:
                pick = np.random.default_rng(d["order"] + 7919 * len(probes)).choice(g.size, min(EVAL_PROBES, g.size), replace=False)
                probes.append(g[pick])
                owner += [(d["src"], d["order"])] * len(pick)
    if not probes:
        return {}
    pr = np.concatenate(probes)
    order = np.argsort(pr, kind="stable")
    ps = pr[order]
    found = np.zeros(ps.size, bool)
    for text, _, _ in long_rows:
        g, _ = decontam.ngram_hashes(text, n=10)
        if g.size:
            lo = np.searchsorted(ps, g, side="left")
            hi = np.searchsorted(ps, g, side="right")
            for a, b in zip(lo[hi > lo], hi[hi > lo]):
                found[a:b] = True
    hits = collections.Counter()
    for k in np.nonzero(found)[0]:
        hits[owner[order[k]]] += 1
    return dict(hits)


def _piece_hist(ns, weight=False):
    edges = [("<4K", 0, 4096), ("4K-8K", 4096, 8192), ("8K-16K", 8192, 16384), ("16K-31K", 16384, 31000),
             ("31K-32K(at cap)", 31000, CTX_MAX + 1), (">32K", CTX_MAX + 1, 10**12)]
    h = collections.Counter()
    for n in ns:
        for k, a, b in edges:
            if a <= n < b:
                h[k] += n if weight else 1
    return {k: h[k] for k, _, _ in edges}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--long-target", type=float, default=2.8e8)
    ap.add_argument("--replay-target", type=float, default=0, help="0 = same as the long pool's tokens")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=32768)
    ap.add_argument("--stage", default="all", choices=["all", "collect", "assemble"])
    ap.add_argument("--dry-run", action="store_true", help="~2M long tokens per bucket, small eval, partial code file")
    ap.add_argument("--code-max-bytes", type=float, default=0)
    ap.add_argument("--eval-max-bytes", type=float, default=900e6)
    ap.add_argument("--only", default="", help="comma list of sources to collect (others keep earlier results)")
    ap.add_argument("--resume", action="store_true", help="skip tasks that already finished in an earlier collect")
    ap.add_argument("--long-share", type=float, default=0.5, help="long / (long + replay) tokens (replay-target 0)")
    ap.add_argument("--sources-json", default="", help="per-source overrides: files, rec_sources (top-up batches)")
    ap.add_argument("--exclude-report", action="append", default=[], help="earlier report.json: never reuse its files/rows")
    ap.add_argument("--copy-val-long", default="", help="reuse this val_long.parquet; drop train docs overlapping it")
    ap.add_argument("--dedupe-against-parts", action="append", default=[], help="earlier _work/parts dir to dedupe against")
    args = ap.parse_args()
    args.exclude_report = [os.path.abspath(p) for p in args.exclude_report]
    args.dedupe_against_parts = [os.path.abspath(p) for p in args.dedupe_against_parts]
    args.copy_val_long = os.path.abspath(args.copy_val_long) if args.copy_val_long else ""
    args.train_files, args.cfg_over, args.skipped_default_files = {}, {}, {}
    if args.sources_json:
        for name, o in json.load(open(args.sources_json)).items():
            if name.startswith("_"):
                continue
            if name not in SOURCES or set(o) - {"files", "rec_sources"}:
                sys.exit(f"--sources-json: bad entry {name}: {sorted(o)}")
            if "files" in o:
                args.train_files[name] = list(o["files"])
            if "rec_sources" in o:
                args.cfg_over[name] = dict(rec_sources=list(o["rec_sources"]))
    args.excluded_files, args.replay_used = load_exclusions(args.exclude_report) if args.exclude_report else (set(), {})
    if args.exclude_report:
        bad = [f"{n}: {f}" for n, fs in args.train_files.items() for f in fs if (SOURCES[n]["repo"], f) in args.excluded_files]
        if bad:
            sys.exit("--sources-json lists files used by an earlier batch (or eval files):\n  " + "\n  ".join(bad))
        for n, cfg in SOURCES.items():
            if n not in args.train_files:
                keep = [f for f in cfg["files"] if (cfg["repo"], f) not in args.excluded_files]
                args.skipped_default_files[n] = [f for f in cfg["files"] if f not in keep]
                args.train_files[n] = keep
        dup = collections.Counter((SOURCES[n]["repo"], f, tuple(args.cfg_over.get(n, {}).get("rec_sources", [])),
                                   tuple(sorted(SOURCES[n].get("collections", [])))) for n, fs in args.train_files.items() for f in fs)
        if any(c > 1 for c in dup.values()):
            sys.exit(f"same file + record filter used by two sources: {[k for k, c in dup.items() if c > 1]}")
    if args.dry_run:
        args.long_target = min(args.long_target, 8e6)
        args.code_max_bytes = args.code_max_bytes or 150e6
        args.eval_max_bytes = min(args.eval_max_bytes, 150e6)
    args.code_max_bytes = int(args.code_max_bytes)
    args.eval_max_bytes = int(args.eval_max_bytes)
    args.out = os.path.abspath(args.out or f"{ROOT}/data/udgam2_ctx32k" + ("_dry" if args.dry_run else ""))
    work = f"{args.out}/_work"
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    if args.copy_val_long and not os.path.exists(f"{work}/val_probes_owner_sorted.npy"):
        print(f"val_long: {prepare_val(args.copy_val_long, work)} docs -> 10-gram arrays in {work}", flush=True)
    if args.stage in ("all", "collect"):
        results = collect(args, work)
    else:
        results = load_results(work)
    if args.stage in ("all", "assemble"):
        rep = assemble(args, work, results, t0)
        print(json.dumps({k: rep[k] for k in ("long", "train", "val_long", "errors")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
