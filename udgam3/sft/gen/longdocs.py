"""Pick long documents (4K-12K tokens of our tokenizer_v1) for the 16K long-document slice.

Sources (pinned revisions; only a few row groups are read over HTTP, nothing big is downloaded):
  wikimedia/wikipedia 20231101.en  (CC-BY-SA-4.0 / GFDL)   rev b04c8d1ceb2f5cd4588862100d08de323dccfbaa
  HuggingFaceFW/finepdfs-edu eng_Latn train (ODC-By-1.0; educational-quality subset of FinePDFs)
                                                           rev 9cfabe2127faca99b3d5c4dc6d1fcb397399ebde
Quality gates: English, no e-mail / phone / long-id PII, not list/table-like (short-line share), no "List of" pages,
few non-ASCII letters, and >= 3 shared 8-grams with an eval set -> dropped.
  uv run --project evals/harness python -I udgam3/sft/gen/longdocs.py --n-wiki 6 --n-pdf 6 --out ~/.cache/udgam3_sft/longdocs_pilot.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BH, REPO, contamination, english_output_ok  # noqa: E402

WIKI = ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.en/train-000{:02d}-of-00041.parquet",
        "CC-BY-SA-4.0 (Wikipedia)")
PDF = ("HuggingFaceFW/finepdfs-edu", "9cfabe2127faca99b3d5c4dc6d1fcb397399ebde",
       "data/eng_Latn/train/000_000{:02d}.parquet", "ODC-By-1.0 (FinePDFs-Edu)")


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(REPO / "artifacts" / "tokenizer_v1" / "tokenizer.json"))


def doc_ok(text: str, title: str = "") -> str | None:
    if title.lower().startswith(("list of", "lists of", "timeline of", "glossary of", "index of")):
        return "list page"
    if any(r.search(text) for r in BH.PII_RX[:3]):
        return "pii"
    if not english_output_ok(text[:20000]):
        return "not english"
    lines = [l for l in text.splitlines() if l.strip()]
    short = sum(len(l.split()) < 6 for l in lines) / max(1, len(lines))
    if short > 0.35:
        return "list/table-like"
    if text.count("|") > 50 or len(re.findall(r"\.{5,}", text)) > 5:
        return "table / index-like"
    if len(re.findall(r"\[\d+\]", text)) > 80:
        return "citation-heavy"
    if len(re.findall(r"\b(download|ebook|pdf|free books?|read online)\b", text, re.I)) >= 4:
        return "spam (downloads)"
    if len(re.findall(r"\b[A-Z][a-z]+ [A-Z]{1,2}[,.]", text)) > 25 or len(re.findall(r"\bet al\.", text)) > 15:
        return "author / reference list"
    return None


def clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Wikipedia dumps end with reference sections; cut them
    m = re.search(r"\n(References|See also|External links|Further reading|Notes|Bibliography|Sources)\s*\n", text)
    if m and m.start() > len(text) * 0.5:
        text = text[:m.start()]
    return text.strip()


def pick(spec, shard: int, n: int, tok, lo: int, hi: int, seen: set, stats: dict) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    repo, rev, pat, lic = spec
    fs = HfFileSystem()
    path = f"datasets/{repo}@{rev}/{pat.format(shard)}"
    pf = pq.ParquetFile(fs.open(path, block_size=8 << 20))
    out = []
    for rg in range(pf.num_row_groups):
        cols = [c for c in ("id", "title", "url", "text") if c in pf.schema_arrow.names]
        for r in pf.read_row_group(rg, columns=cols).to_pylist():
            text = clean(r.get("text") or "")
            words = len(text.split())
            if words < lo * 0.6 or words > hi * 0.9:
                continue
            ntok = len(tok.encode(text).ids)
            if not lo <= ntok <= hi:
                continue
            why = doc_ok(text, r.get("title") or "")
            if why:
                stats[why] = stats.get(why, 0) + 1
                continue
            if contamination([text], long_doc=True):
                stats["eval/heldout overlap"] = stats.get("eval/heldout overlap", 0) + 1
                continue
            h = hashlib.sha256(text[:2000].encode()).hexdigest()[:12]
            if h in seen:
                continue
            seen.add(h)
            title = r.get("title") or ""
            if not title:
                first = text.strip().splitlines()[0][:120]
                title = first if len(first.split()) <= 14 else ""
            out.append({"doc_id": f"{repo.split('/')[1]}:{r.get('id')}", "source": repo, "revision": rev,
                        "licence": lic, "title": title, "url": r.get("url"), "tokens": ntok, "words": words,
                        "text": text})
            if len(out) >= n:
                return out
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wiki", type=int, default=6)
    ap.add_argument("--n-pdf", type=int, default=6)
    ap.add_argument("--min-tokens", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--wiki-shard", type=int, default=7)
    ap.add_argument("--pdf-shard", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = tokenizer()
    seen, stats = set(), {}
    docs = pick(WIKI, a.wiki_shard, a.n_wiki, tok, a.min_tokens, a.max_tokens, seen, stats)
    docs += pick(PDF, a.pdf_shard, a.n_pdf, tok, a.min_tokens, a.max_tokens, seen, stats)
    with open(Path(a.out).expanduser(), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps({"docs": len(docs), "rejected": stats,
                      "picked": [(d["source"], d["title"][:60], d["tokens"]) for d in docs]}, indent=1))


if __name__ == "__main__":
    main()
