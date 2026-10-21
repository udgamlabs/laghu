"""Select long documents for the full run (8K-14K tokens of tokenizer_v1, stratified by length) and dump short
Wikipedia passages for the chat slice's summarising requests.

Sources (pinned): HuggingFaceFW/finepdfs-edu eng_Latn train (ODC-By-1.0; preferred, ~70%) and wikimedia/wikipedia
20231101.en (CC-BY-SA-4.0; ~30%, attribution = title + URL kept in provenance).
Gates: rule filters (PII, list/table-like, extraction artefacts such as split ligatures 'Th e', 'I=ll', hyphen breaks,
replacement characters, '(cid:'; private-person markers such as 'Submitted by', 'Student ID', 'Dear X', signatures;
reference lists; spam), eval/held-out 8-gram overlap (>= 3 shared grams), then an LLM screen of three excerpts
(start/middle/end) for artefacts, private persons' names, coherence and safety. The LLM screen's spend goes to the
same spend.json / cap as the generator (--out, --cap).
  uv run --project evals/harness python -I udgam3/sft/gen/longdocs_full.py --out udgam3/sft/full_v1 --cap 9.3 \
      --n 560 --passages 600
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DeepSeek, SpendCapReached  # noqa: E402
from common import CACHE, contamination  # noqa: E402
from longdocs import PDF, WIKI, clean, doc_ok, tokenizer  # noqa: E402

BINS = [(8000, 10000, 0.40), (10000, 12000, 0.35), (12000, 14000, 0.25)]
ARTEFACT_RX = [
    (re.compile(r"\b[A-Za-z]=[a-z]{1,2}\b"), 2),                                      # I=ll, don=t
    (re.compile(r"\b(?:Th|th) (?:e|is|at|ey|ere|ese|em|en|erefore|ough|ird|ree|ose|us)\b"), 2),
    (re.compile(r"\b(?:di|e|a|su|o)ff (?:erent|ect|ort|icient|er|ice|ord)\w*"), 1),
    (re.compile(r"\bfi (?:rst|nd|nal|eld|ve|gure|nally|ne)\b|\bfl (?:ow|oor|uid)\b"), 1),
    (re.compile(r"[a-z]{2,}- [a-z]{2,}"), 6),                                          # envi- ronment
    (re.compile(r"�|\(cid:\d+\)"), 0),
    (re.compile(r"^\s*\d{1,3}\s*$", re.M), 12),                                        # bare page numbers
]
TOPIC_RX = re.compile(r"\b(china|chinese|taiwan\w*|tibet\w*|xinjiang|uyghur|hong kong|tiananmen|war|wars|warfare|"
                      r"terror\w*|massacre|genocide|assassinat\w*|insurgen\w*|guerrilla|erotic\w*|sexual\w*|sex|"
                      r"porn\w*|murder\w*|rape|suicide|torture)\b", re.I)      # titles: teacher-bias / graphic topics
LATEX_RX = re.compile(r"\\(textsuperscript|textbf|textit|mathrm|emph|cite|ref)\{")


def topic_or_markup(d: dict) -> str | None:
    if TOPIC_RX.search(d.get("title") or "") or TOPIC_RX.search(d["text"][:300]):
        return "topic (title/opening)"
    if len(LATEX_RX.findall(d["text"])) > 3:
        return "latex markup artefacts"
    return None


PRIVATE_RX = re.compile(r"\b(submitted (by|to)|prepared by|student (id|name|no)|roll no|reg(istration)? no|"
                        r"matric(ulation)? no|supervis(or|ed by)|name of (the )?student|dear (mr|mrs|ms|dr|sir|madam)?\s*"
                        r"[A-Z][a-z]+|yours (sincerely|faithfully)|signature|date of birth|candidate number|"
                        r"employee (id|number)|patient (name|id))\b", re.I)

SCREEN_SYS = "You screen documents for a reading-comprehension training set. Return JSON only."
SCREEN_USER = """Below are three excerpts (start, middle, end) of one long document.
Reject it (ok=false) if ANY of these hold: text-extraction artefacts (broken or split words, garbled characters, merged columns, repeated page headers); it names private individuals (students, employees, patients, applicants, signatories, people's contact details), as opposed to public figures, historical people or authors of published work; it is mostly lists, tables, forms, references or an index; it is spam or advertising; it is not English; it contains sexual or graphic violent content.
Otherwise ok=true.

[START]
{a}

[MIDDLE]
{b}

[END]
{c}

Return JSON: {{"ok": true|false, "why": "<at most 8 words>"}}"""


def artefacts(text: str) -> str | None:
    for rx, limit in ARTEFACT_RX:
        if len(rx.findall(text)) > limit:
            return "artefact:" + rx.pattern[:20]
    return None


def excerpt(text: str) -> tuple[str, str, str]:
    w = text.split()
    n = len(w)
    return " ".join(w[:280]), " ".join(w[n // 2 - 110:n // 2 + 110]), " ".join(w[-180:])


def scan(spec, shards, tok, want: dict, stats: dict, passages: list, n_pass: int, rng) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    repo, rev, pat, lic = spec
    fs = HfFileSystem()
    out, counts = [], {b: 0 for b in want}
    is_wiki = "wikipedia" in repo
    for shard in shards:
        pf = pq.ParquetFile(fs.open(f"datasets/{repo}@{rev}/{pat.format(shard)}", block_size=8 << 20))
        rgs = list(range(pf.num_row_groups))
        rng.shuffle(rgs)
        for rg in rgs:
            cols = [c for c in ("id", "title", "url", "text") if c in pf.schema_arrow.names]
            for r in pf.read_row_group(rg, columns=cols).to_pylist():
                text = clean(r.get("text") or "")
                words = len(text.split())
                if is_wiki and len(passages) < n_pass and 600 <= words <= 4000:
                    paras = [p for p in text.split("\n\n") if 150 <= len(p.split()) <= 450 and not p.startswith("#")]
                    pick = rng.choice(paras) if paras else None
                    if pick and not doc_ok(pick) and not artefacts(pick) and not contamination([pick]):
                        passages.append({"pid": f"wiki:{r.get('id')}", "title": r.get("title") or "",
                                         "url": r.get("url") or "", "text": pick})
                if words < 5200 or words > 11500:
                    continue
                ntok = len(tok.encode(text).ids)
                b = next((b for b in want if b[0] <= ntok < b[1]), None)
                if b is None or counts[b] >= want[b]:
                    continue
                why = doc_ok(text, r.get("title") or "") or artefacts(text) or \
                    ("private-person marker" if PRIVATE_RX.search(text) else None)
                if why:
                    stats[why] = stats.get(why, 0) + 1
                    continue
                if contamination([text], long_doc=True):
                    stats["eval/heldout overlap"] = stats.get("eval/heldout overlap", 0) + 1
                    continue
                title = r.get("title") or ""
                if not title:
                    first = text.strip().splitlines()[0][:120]
                    title = first if len(first.split()) <= 14 else ""
                rec_probe = {"title": r.get("title") or "", "text": text}
                if topic_or_markup(rec_probe):
                    stats["topic/markup"] = stats.get("topic/markup", 0) + 1
                    continue
                counts[b] += 1
                out.append({"doc_id": f"{repo.split('/')[1]}:{r.get('id')}", "source": repo, "revision": rev,
                            "licence": lic, "title": title, "url": r.get("url"), "tokens": ntok, "words": words,
                            "text": text})
            if all(counts[b] >= want[b] for b in want) and (not is_wiki or len(passages) >= n_pass):
                return out
            print(f"  {repo} shard {shard} rg {rg}: {sum(counts.values())}/{sum(want.values())} docs, "
                  f"{len(passages)} passages", file=sys.stderr)
    return out


async def llm_screen(docs, out_dir, cap):
    async with DeepSeek(Path(out_dir), cap_usd=cap, concurrency=24) as ds:
        async def one(d):
            a, b, c = excerpt(d["text"])
            try:
                r = await ds.chat([{"role": "system", "content": SCREEN_SYS},
                                   {"role": "user", "content": SCREEN_USER.format(a=a, b=b, c=c)}],
                                  slice_name="longdoc_screen", json_mode=True, temperature=0.0, max_tokens=40)
            except SpendCapReached:
                return None
            try:
                o = json.loads(r["content"])
                return bool(o.get("ok")), str(o.get("why", ""))[:60]
            except (json.JSONDecodeError, AttributeError):
                return False, "unparsable"
        res = await asyncio.gather(*[one(d) for d in docs])
        return res, ds.spend["usd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--n", type=int, default=560, help="candidate docs before the LLM screen")
    ap.add_argument("--passages", type=int, default=600)
    ap.add_argument("--pdf-share", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--wiki-shards", default="7,11,19,23,31")
    ap.add_argument("--pdf-shards", default="3,9,17,25,33,41,57,73")
    ap.add_argument("--append", action="store_true", help="add to the existing doc list (skip known doc ids)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok = tokenizer()
    stats, passages = {}, []
    n_pdf = int(a.n * a.pdf_share)
    want_pdf = {(lo, hi): round(n_pdf * f) for lo, hi, f in BINS}
    want_wiki = {(lo, hi): round((a.n - n_pdf) * f) for lo, hi, f in BINS}
    wsh = [int(x) for x in a.wiki_shards.split(",") if x]
    psh = [int(x) for x in a.pdf_shards.split(",") if x]
    docs = scan(WIKI, wsh, tok, want_wiki, stats, passages, a.passages, rng) if wsh and a.n - n_pdf > 0 else []
    docs += scan(PDF, psh, tok, want_pdf, stats, passages, 0, rng)
    prev = []
    if a.append and (CACHE / "longdocs_full_v1.jsonl").exists():
        prev = [json.loads(l) for l in open(CACHE / "longdocs_full_v1.jsonl", encoding="utf-8")]
        known = {d["doc_id"] for d in prev}
        docs = [d for d in docs if d["doc_id"] not in known]
    res, spent = asyncio.run(llm_screen(docs, a.out, a.cap))
    keep, why = [], {}
    for d, r in zip(docs, res):
        if r and r[0]:
            keep.append(d)
        else:
            k = (r[1] if r else "not screened (cap)")
            why[k] = why.get(k, 0) + 1
    rng.shuffle(keep)
    keep = [d for d in prev if not topic_or_markup(d)] + keep
    keep.sort(key=lambda d: 0 if "finepdfs" in d["source"] else 1)   # FinePDFs-Edu first (preferred)
    with open(CACHE / "longdocs_full_v1.jsonl", "w", encoding="utf-8") as f:
        for d in keep:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if passages:
        with open(CACHE / "wiki_passages_v1.jsonl", "w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    bins = {f"{lo}-{hi}": sum(lo <= d["tokens"] < hi for d in keep) for lo, hi, _ in BINS}
    print(json.dumps({"candidates": len(docs), "kept": len(keep), "by_source": {
        s: sum(d["source"] == s for d in keep) for s in {d["source"] for d in keep}}, "token_bins": bins,
        "rule_rejects": stats, "llm_rejects": dict(sorted(why.items(), key=lambda x: -x[1])[:25]),
        "passages": len(passages), "spend_total_usd": round(spent, 4)}, indent=1))


if __name__ == "__main__":
    main()
