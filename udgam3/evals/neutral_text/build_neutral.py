#!/usr/bin/env python3
"""Build the Udgam-3 "neutral held-out text" set (neutral_v1.jsonl) for bits-per-byte comparisons.

Goal: text that none of the compared models could have trained on (published after ~2026-09-01) and that is
licence-clean, so that models with different tokenizers can be compared by bits per byte.

Sources (both fetched live from public APIs, standard library only):
  1. English Wikipedia articles whose FIRST revision was created on/after --wiki-start (default 2026-09-01).
     Licence: CC BY-SA 4.0 (attribution = the `url` field, a permalink to the exact revision fetched).
     Pipeline: list=logevents&letype=create (mainspace) -> drop redirects / recreated pages / copied-content
     edit summaries -> prop=info (length, redirect, disambiguation) -> first-revision check (parentid == 0 and
     timestamp >= start) -> prop=extracts&explaintext (one page per request) -> prose-only cleaning
     (drop reference-like sections, headings, list-like lines) -> keep >= 300 words of prose.
  2. arXiv abstracts submitted on/after --arxiv-start (default 2026-10-01), via export.arxiv.org/api/query
     (arXiv metadata, including abstracts, is CC0). Many categories x 3 date windows, >= 3.2 s between calls.
Each document's text is  "<title>\\n\\n<body>"  (Wikipedia: cleaned prose paragraphs; arXiv: abstract).

Reproducibility: all randomness uses --seed; all candidate pages / entries (with page id, creation revid,
fetched revid, timestamps, accept/reject reason) are written to raw/*.jsonl; every HTTP response can be cached
(--cache-dir) so a re-run is byte-identical. Without the cache, results depend on live Wikipedia (pages can be
deleted/edited later) and arXiv, so a re-run may differ slightly.

Known properties of the v1 build: arXiv dates cluster at the last days of the three windows (Oct 5, 11, 15)
because each window keeps the 10 newest hits per category; Wikipedia is balanced by creation day (7-20 per day).
Long Wikipedia articles are truncated at a paragraph boundary at --wiki-max-words (1500) words.

Usage:
  python3 build_neutral.py [--out-dir DIR] [--cache-dir DIR] [--wiki-bytes 2400000] [--arxiv-bytes 1600000]
"""
import argparse
import collections
import datetime as dt
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = ("UdgamLabsEvalBuilder/1.0 (https://huggingface.co/udgamlabs; non-commercial research evaluation set) "
      "python-urllib")
WIKI_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"

# ----------------------------------------------------------------------------------------------------------
# HTTP with retry, pacing and an optional on-disk cache of raw responses
# ----------------------------------------------------------------------------------------------------------
_last_call = collections.defaultdict(float)
CACHE_DIR = None
STATS = collections.Counter()


def http_get(url, min_interval, host_key, tries=8):
    """GET url (bytes). Paces calls per host (min_interval seconds); retries on 429/5xx/network errors."""
    key = hashlib.sha1(url.encode()).hexdigest()
    cpath = os.path.join(CACHE_DIR, key[:2], key) if CACHE_DIR else None
    if cpath and os.path.exists(cpath):
        STATS["cache_hit"] += 1
        with open(cpath, "rb") as f:
            return f.read()
    delay = 2.0
    for attempt in range(tries):
        wait = _last_call[host_key] + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[host_key] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
            STATS["http_" + host_key] += 1
            if cpath:
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                with open(cpath, "wb") as f:
                    f.write(body)
            return body
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
            code = getattr(e, "code", None)
            if code not in (None, 429, 500, 502, 503, 504) and attempt >= 1:
                raise
            retry_after = None
            if code == 429 and getattr(e, "headers", None) is not None:
                try:
                    retry_after = float(e.headers.get("Retry-After"))
                except (TypeError, ValueError):
                    retry_after = None
            sleep_s = max(delay, (retry_after or 0) + 2)
            print(f"  [retry {attempt + 1}/{tries}] {host_key} {code or type(e).__name__}; sleeping {sleep_s:.0f}s",
                  file=sys.stderr)
            time.sleep(sleep_s)
            delay = min(delay * 2, 60)
    raise RuntimeError("giving up on " + url)


def wiki_api(**params):
    params.update(format="json", formatversion=2, maxlag=5)
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    for _ in range(5):
        d = json.loads(http_get(url, 1.2, "wiki"))
        if "error" in d and d["error"].get("code") == "maxlag":
            time.sleep(5)
            continue
        if "error" in d:
            raise RuntimeError(f"wiki api error {d['error']} for {params}")
        return d
    raise RuntimeError("maxlag persisted")


# ----------------------------------------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------------------------------------
def norm_text(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ").replace("​", "").replace("﻿", "")
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def words(s):
    return len(s.split())


def ascii_letter_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return (sum(c.isascii() for c in letters) / len(letters)) if letters else 0.0


def dedup_key(s):
    return hashlib.sha1(re.sub(r"\W+", " ", s.lower()).strip().encode()).hexdigest()


def utf8_len(s):
    return len(s.encode("utf-8"))


# ----------------------------------------------------------------------------------------------------------
# Wikipedia
# ----------------------------------------------------------------------------------------------------------
STOP_SECTIONS = {
    "references", "external links", "see also", "further reading", "notes", "sources", "bibliography", "footnotes",
    "citations", "works cited", "explanatory notes", "general references", "notes and references",
    "references and notes", "selected bibliography", "gallery", "external links and further reading",
    "notes and sources", "cited works", "footnotes and references",
}
BAD_TITLE = re.compile(r"^(List|Lists|Index|Outline|Timeline|Glossary|Bibliography|Discography|Filmography) of |"
                       r"\((disambiguation|surname|given name|name)\)$|^\d{4}.*(season|results|elections?)$", re.I)
COPIED_COMMENT = re.compile(r"\b(split|spun|spin(?:ning)?[ -](?:off|out)|copied|copying|copy from|merged?|forked|"
                            r"translat\w+|import(?:ed)?|restor\w+|undelet\w+|unredirect\w+|draft|moved|rv|revert)\b",
                            re.I)
MARKUP_LEFTOVER = re.compile(r"\{\{|\}\}|\[\[|\]\]|</?[a-zA-Z][^>]*>|\\displaystyle|&[a-z]{2,8};|\|\s*\w+\s*=|"
                             r"\bwikitable\b|^\s*[*#:;]", re.M)
HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")
SENT_END = re.compile(r"[.!?…][\"”’')\]]*$")


def clean_wiki_extract(extract, max_words):
    """Return (paragraphs, prose_words, other_words) from a plain-text extract with == headings ==."""
    paras, prose_w, other_w = [], 0, 0
    skip_level = None
    for raw in extract.split("\n"):
        line = norm_text(raw)
        if not line:
            continue
        m = HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip().lower()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and title in STOP_SECTIONS:
                skip_level = level
            continue
        if skip_level is not None:
            continue
        w = words(line)
        if w >= 12 and SENT_END.search(line) and not MARKUP_LEFTOVER.search(line):
            paras.append(line)
            prose_w += w
        else:
            other_w += w
    return paras, prose_w, other_w


def truncate_paras(paras, max_words):
    out, n = [], 0
    for p in paras:
        w = words(p)
        if out and n + w > max_words:
            break
        out.append(p)
        n += w
    return out


def wiki_enumerate_creates(day_start, day_end):
    """All mainspace 'create' log events with day_start <= timestamp < day_end (UTC dates), one day at a time."""
    events = []
    d = day_start
    while d < day_end:
        nxt = d + dt.timedelta(days=1)
        cont = {}
        n_day = 0
        while True:
            res = wiki_api(action="query", list="logevents", letype="create", lenamespace=0, lelimit=500,
                           lestart=nxt.strftime("%Y-%m-%dT00:00:00Z"), leend=d.strftime("%Y-%m-%dT00:00:00Z"),
                           ledir="older", leprop="ids|title|timestamp|comment|tags|type", **cont)
            for e in res["query"]["logevents"]:
                events.append({"logid": e["logid"], "pageid": e["pageid"], "revid": e["revid"], "title": e["title"],
                               "timestamp": e["timestamp"], "comment": e.get("comment", ""), "tags": e.get("tags", [])})
                n_day += 1
            if "continue" in res:
                cont = res["continue"]
            else:
                break
        print(f"  {d} creates: {n_day}", file=sys.stderr)
        d = nxt
    return events


def wiki_build(args, rng, rejects):
    start = dt.date.fromisoformat(args.wiki_start)
    end = dt.date.fromisoformat(args.wiki_end) + dt.timedelta(days=1)  # inclusive end date
    start_ts = start.strftime("%Y-%m-%dT00:00:00Z")
    print(f"[wiki] enumerating mainspace page creations {start} .. {end - dt.timedelta(days=1)}", file=sys.stderr)
    events = wiki_enumerate_creates(start, end)
    counts = collections.Counter()
    pool = collections.defaultdict(list)
    for e in events:
        counts["events"] += 1
        tags = set(e["tags"])
        # the auto-generated summary "Created page with '<first 200 chars of wikitext>'" must not be keyword-matched
        cmt = re.sub(r"Created page with .*", "", e["comment"], flags=re.S)
        if "mw-new-redirect" in tags or re.search(r"redirect", cmt, re.I):
            counts["skip_redirect_event"] += 1
            continue
        if "mw-recreated" in tags:
            counts["skip_recreated"] += 1
            continue
        if COPIED_COMMENT.search(cmt):
            counts["skip_copied_comment"] += 1
            continue
        if BAD_TITLE.search(e["title"]):
            counts["skip_title_pattern"] += 1
            continue
        pool[e["timestamp"][:10]].append(e)
    for day in pool.values():
        rng.shuffle(day)
    days = sorted(pool)
    print(f"[wiki] {counts['events']} create events; candidates after event-level filters: "
          f"{sum(len(v) for v in pool.values())} over {len(days)} days", file=sys.stderr)
    # round-robin across days so the sample is balanced in time
    order = []
    i = 0
    while any(i < len(pool[d]) for d in days):
        for d in days:
            if i < len(pool[d]):
                order.append(pool[d][i])
        i += 1

    docs, seen, total_bytes = [], set(), 0
    chunk = 50
    for c0 in range(0, len(order), chunk):
        if total_bytes >= args.wiki_bytes:
            break
        batch = order[c0:c0 + chunk]
        # (a) page info for the batch
        info = wiki_api(action="query", pageids="|".join(str(e["pageid"]) for e in batch), prop="info|pageprops",
                        ppprop="disambiguation")
        pages = {p["pageid"]: p for p in info["query"]["pages"] if not p.get("missing") and not p.get("invalid")}
        # (b) first-revision check using the log's creation revid
        rev = wiki_api(action="query", revids="|".join(str(e["revid"]) for e in batch), prop="revisions",
                       rvprop="ids|timestamp|size")
        revinfo = {}
        for p in rev["query"]["pages"]:
            for r in p.get("revisions", []):
                revinfo[r["revid"]] = (p.get("pageid"), r)
        for e in batch:
            if total_bytes >= args.wiki_bytes:
                break
            rec = dict(e, source="wikipedia")
            pg = pages.get(e["pageid"])
            if pg is None:
                rec["status"] = "rejected:missing_or_deleted"
            elif pg.get("redirect"):
                rec["status"] = "rejected:redirect"
            elif pg.get("ns") != 0:
                rec["status"] = "rejected:not_mainspace"
            elif "disambiguation" in pg.get("pageprops", {}):
                rec["status"] = "rejected:disambiguation"
            elif pg.get("length", 0) < args.wiki_min_wikitext_bytes:
                rec["status"] = "rejected:too_short_wikitext"
            elif e["revid"] not in revinfo or revinfo[e["revid"]][1].get("parentid") != 0 \
                    or revinfo[e["revid"]][1]["timestamp"] < start_ts:
                rec["status"] = "rejected:not_first_revision_after_start"
            else:
                rec["first_rev_timestamp"] = revinfo[e["revid"]][1]["timestamp"]
                # (c) full plain-text extract, one page per request
                ex = wiki_api(action="query", pageids=str(e["pageid"]), prop="extracts|info", explaintext=1,
                              exsectionformat="wiki", exlimit=1)
                p = ex["query"]["pages"][0]
                extract = p.get("extract", "")
                paras, prose_w, other_w = clean_wiki_extract(extract, args.wiki_max_words)
                rec.update(current_title=p.get("title"), fetched_revid=p.get("lastrevid"),
                           prose_words=prose_w, other_words=other_w)
                if prose_w < args.wiki_min_words:
                    rec["status"] = "rejected:under_min_words"
                elif prose_w / max(1, prose_w + other_w) < 0.7:
                    rec["status"] = "rejected:list_like"
                else:
                    paras = truncate_paras(paras, args.wiki_max_words)
                    body = "\n\n".join(paras)
                    title = norm_text(p["title"])
                    text = f"{title}\n\n{body}"
                    if ascii_letter_ratio(text) < 0.93:
                        rec["status"] = "rejected:non_english_like"
                    elif dedup_key(body[:400]) in seen or dedup_key(text) in seen:
                        rec["status"] = "rejected:duplicate"
                    else:
                        seen.add(dedup_key(text))
                        seen.add(dedup_key(body[:400]))
                        nb = utf8_len(text)
                        total_bytes += nb
                        rec["status"] = "accepted"
                        rec["bytes"] = nb
                        docs.append({
                            "id": f"wiki:{e['pageid']}",
                            "source": "wikipedia",
                            "title": title,
                            "created": rev_ts(revinfo[e["revid"]][1]["timestamp"]),
                            "url": f"https://en.wikipedia.org/w/index.php?title="
                                   f"{urllib.parse.quote(p['title'].replace(' ', '_'))}&oldid={p.get('lastrevid')}",
                            "licence": "CC-BY-SA-4.0 (English Wikipedia text; attribute via url / page history)",
                            "text": text,
                            "meta": {"page_id": e["pageid"], "creation_revid": e["revid"],
                                     "fetched_revid": p.get("lastrevid"), "log_title": e["title"],
                                     "prose_words": prose_w, "words_in_text": words(body)},
                        })
            rejects.append({k: v for k, v in rec.items() if k != "comment"})
        print(f"[wiki] processed {min(c0 + chunk, len(order))}/{len(order)} candidates; "
              f"{len(docs)} accepted, {total_bytes / 1e6:.2f} MB", file=sys.stderr)
    return docs, counts


def rev_ts(ts):
    return ts  # ISO-8601 UTC, e.g. 2026-09-09T23:32:00Z


# ----------------------------------------------------------------------------------------------------------
# arXiv
# ----------------------------------------------------------------------------------------------------------
ARXIV_CATS = [
    # computer science
    "cs.CL", "cs.CV", "cs.LG", "cs.AI", "cs.CR", "cs.DS", "cs.SE", "cs.RO", "cs.IR", "cs.HC", "cs.DC", "cs.GT",
    "cs.LO", "cs.NE", "cs.SI", "cs.CY", "cs.IT", "cs.CC",
    # mathematics
    "math.AP", "math.PR", "math.CO", "math.NT", "math.OC", "math.AG", "math.DG", "math.NA", "math.DS", "math.FA",
    "math.GT", "math.RT",
    # statistics
    "stat.ML", "stat.ME", "stat.AP", "stat.TH",
    # physics (general, high-energy, astro, condensed matter, quantum)
    "physics.optics", "physics.flu-dyn", "physics.chem-ph", "physics.bio-ph", "physics.comp-ph", "physics.soc-ph",
    "physics.atom-ph", "physics.ao-ph", "hep-th", "hep-ph", "hep-ex", "gr-qc", "nucl-th", "quant-ph",
    "astro-ph.CO", "astro-ph.GA", "astro-ph.SR", "astro-ph.HE", "astro-ph.EP",
    "cond-mat.mes-hall", "cond-mat.str-el", "cond-mat.mtrl-sci", "cond-mat.stat-mech", "cond-mat.soft",
    # quantitative biology
    "q-bio.NC", "q-bio.PE", "q-bio.QM", "q-bio.GN", "q-bio.BM", "q-bio.MN", "q-bio.CB",
    # economics / finance / engineering
    "econ.EM", "econ.TH", "econ.GN", "q-fin.ST", "q-fin.PM", "q-fin.RM", "q-fin.MF", "eess.SP", "eess.IV", "eess.AS",
]
NS = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/",
      "x": "http://arxiv.org/schemas/atom"}


def arxiv_windows(start, end):
    """Three roughly equal date windows [YYYYMMDD0000, YYYYMMDD2359] covering start..end inclusive."""
    n = (end - start).days + 1
    cuts = [0, n // 3, 2 * n // 3, n]
    out = []
    for i in range(3):
        a = start + dt.timedelta(days=cuts[i])
        b = start + dt.timedelta(days=cuts[i + 1] - 1)
        out.append((a.strftime("%Y%m%d0000"), b.strftime("%Y%m%d2359")))
    return out


def arxiv_fetch(cat, win, n):
    q = {"search_query": f"cat:{cat} AND submittedDate:[{win[0]} TO {win[1]}]", "start": 0, "max_results": n,
         "sortBy": "submittedDate", "sortOrder": "descending"}
    url = ARXIV_API + "?" + urllib.parse.urlencode(q)
    root = ET.fromstring(http_get(url, 3.3, "arxiv"))
    out = []
    for e in root.findall("a:entry", NS):
        eid = e.find("a:id", NS).text.strip()
        prim = e.find("x:primary_category", NS)
        out.append({
            "arxiv_url": eid,
            "arxiv_id": re.sub(r"v\d+$", "", eid.rsplit("/abs/", 1)[-1]),
            "published": e.find("a:published", NS).text.strip(),
            "updated": e.find("a:updated", NS).text.strip(),
            "primary_category": prim.get("term") if prim is not None else None,
            "queried_category": cat,
            "title": e.find("a:title", NS).text or "",
            "summary": e.find("a:summary", NS).text or "",
        })
    return out


def arxiv_build(args, rng, rejects):
    start = dt.date.fromisoformat(args.arxiv_start)
    end = dt.date.fromisoformat(args.arxiv_end)
    wins = arxiv_windows(start, end)
    cats_to_query = args.arxiv_cats.split(",") if args.arxiv_cats else ARXIV_CATS
    print(f"[arxiv] windows {wins}; {len(cats_to_query)} categories", file=sys.stderr)
    entries = {}
    for ci, cat in enumerate(cats_to_query):
        for win in wins:
            for ent in arxiv_fetch(cat, win, args.arxiv_per_window):
                entries.setdefault(ent["arxiv_id"], ent)  # first category that returns it
        print(f"[arxiv] {ci + 1}/{len(cats_to_query)} {cat}: {len(entries)} unique entries so far", file=sys.stderr)
    by_cat = collections.defaultdict(list)
    for ent in entries.values():
        by_cat[ent["queried_category"]].append(ent)
    for v in by_cat.values():
        rng.shuffle(v)
    order, i = [], 0
    cats = sorted(by_cat)
    while any(i < len(by_cat[c]) for c in cats):
        for c in cats:
            if i < len(by_cat[c]):
                order.append(by_cat[c][i])
        i += 1
    start_iso = start.strftime("%Y-%m-%dT00:00:00Z")
    docs, seen, total_bytes = [], set(), 0
    for ent in order:
        rec = {k: v for k, v in ent.items() if k not in ("summary", "title")}
        rec["source"] = "arxiv"
        title = norm_text(re.sub(r"\s+", " ", ent["title"]))
        abstract = norm_text(re.sub(r"\s+", " ", ent["summary"]))
        if total_bytes >= args.arxiv_bytes:
            rec["status"] = "unused:target_reached"
        elif ent["published"] < start_iso:
            rec["status"] = "rejected:published_before_start"
        elif words(abstract) < 60 or words(abstract) > 600:
            rec["status"] = "rejected:abstract_length"
        elif re.search(r"\bwithdrawn\b", abstract[:200], re.I):
            rec["status"] = "rejected:withdrawn"
        elif ascii_letter_ratio(title + " " + abstract) < 0.93:
            rec["status"] = "rejected:non_english_like"
        else:
            text = f"{title}\n\n{abstract}"
            k = dedup_key(text)
            if k in seen:
                rec["status"] = "rejected:duplicate"
            else:
                seen.add(k)
                nb = utf8_len(text)
                total_bytes += nb
                rec["status"] = "accepted"
                rec["bytes"] = nb
                docs.append({
                    "id": f"arxiv:{ent['arxiv_id']}",
                    "source": "arxiv",
                    "title": title,
                    "created": ent["published"],
                    "url": ent["arxiv_url"],
                    "licence": "CC0-1.0 (arXiv metadata)",
                    "text": text,
                    "meta": {"primary_category": ent["primary_category"], "queried_category": ent["queried_category"],
                             "updated": ent["updated"]},
                })
        rejects.append(rec)
    return docs


# ----------------------------------------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=here)
    ap.add_argument("--cache-dir", default=None, help="cache raw HTTP responses here (makes re-runs identical)")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--wiki-start", default="2026-09-01")
    ap.add_argument("--wiki-end", default="2026-10-15", help="last creation date (inclusive)")
    ap.add_argument("--wiki-bytes", type=int, default=2_400_000)
    ap.add_argument("--wiki-min-words", type=int, default=300)
    ap.add_argument("--wiki-max-words", type=int, default=1500)
    ap.add_argument("--wiki-min-wikitext-bytes", type=int, default=5000)
    ap.add_argument("--arxiv-start", default="2026-10-01")
    ap.add_argument("--arxiv-end", default="2026-10-17")
    ap.add_argument("--arxiv-bytes", type=int, default=1_600_000)
    ap.add_argument("--arxiv-per-window", type=int, default=10)
    ap.add_argument("--arxiv-cats", default=None, help="comma-separated override of the category list")
    ap.add_argument("--skip-wiki", action="store_true")
    ap.add_argument("--skip-arxiv", action="store_true")
    ap.add_argument("--out-name", default="neutral_v1.jsonl")
    args = ap.parse_args()

    global CACHE_DIR
    CACHE_DIR = args.cache_dir
    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    rng = random.Random(args.seed)
    docs = []
    t0 = time.time()
    meta = {"script": os.path.basename(__file__),
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "run_started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "args": vars(args), "user_agent": UA}
    from concurrent.futures import ThreadPoolExecutor

    def run_wiki():
        rej = []
        wdocs, wcounts = wiki_build(args, random.Random(args.seed + 1), rej)
        return wdocs, wcounts, rej

    def run_arxiv():
        rej = []
        adocs = arxiv_build(args, random.Random(args.seed + 2), rej)
        return adocs, rej

    with ThreadPoolExecutor(max_workers=2) as ex:
        fw = None if args.skip_wiki else ex.submit(run_wiki)
        fa = None if args.skip_arxiv else ex.submit(run_arxiv)
        if fw is not None:
            wdocs, wcounts, rej = fw.result()
            docs += wdocs
            meta["wiki_event_counts"] = dict(wcounts)
            with open(os.path.join(raw_dir, "wiki_candidates.jsonl"), "w") as f:
                for r in rej:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            meta["wiki_status_counts"] = dict(collections.Counter(r["status"] for r in rej))
        if fa is not None:
            adocs, rej = fa.result()
            docs += adocs
            with open(os.path.join(raw_dir, "arxiv_entries.jsonl"), "w") as f:
                for r in rej:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            meta["arxiv_status_counts"] = dict(collections.Counter(r["status"] for r in rej))

    # global dedup (exact text) and stable order
    final, seen = [], set()
    for d in sorted(docs, key=lambda d: (d["source"], d["created"], d["id"])):
        k = dedup_key(d["text"])
        if k in seen:
            continue
        seen.add(k)
        final.append(d)
    out_path = os.path.join(args.out_dir, args.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in final:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    summ = {}
    for src in ("wikipedia", "arxiv"):
        ds = [d for d in final if d["source"] == src]
        if ds:
            summ[src] = {"docs": len(ds), "bytes": sum(utf8_len(d["text"]) for d in ds),
                         "words": sum(words(d["text"]) for d in ds),
                         "created_min": min(d["created"] for d in ds), "created_max": max(d["created"] for d in ds)}
    summ["total_bytes"] = sum(v["bytes"] for v in summ.values())
    meta.update(summary=summ, http_stats=dict(STATS), elapsed_s=round(time.time() - t0, 1),
                output_sha256=hashlib.sha256(open(out_path, "rb").read()).hexdigest())
    with open(os.path.join(args.out_dir, "neutral_v1.build_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
