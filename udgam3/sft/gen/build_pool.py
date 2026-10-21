"""Build the real-user prompt pool for SFT generation (prompts only; no model is run).

Reuses the held-out builder (udgam3/evals/chat_prompts/build_heldout.py): the same pinned, sha-checked source files
(WildChat-4.8M rows >= 2024-06 [ODC-By], oasst2 [Apache-2.0], Arena 140k / 100k prompts [CC-BY-4.0]), the same
loaders (moderation / crowd-label / PII-redaction gates), filters (English, PII, needs web/image/file, recent events,
long-answer requests, jailbreak / role-play / model identity, fandom, templates, not self-contained), categories and
follow-up rules.

Differences from the held-out build:
  * Sensitive-topic filter is RELAXED: history/war, general health, money, law, religion facts, everyday safety stay
    in (tagged `sensitive: true` because the held-out's strict filter would drop them). Still dropped: sexual /
    fetish content, graphic violence, weapons, extremism and hate, drugs, self-harm / suicide / depression,
    hacking / malware, piracy, gambling, fraud, elections and party politics, China-politics topics (teacher bias).
  * Held-out exclusion: every held-out source id, every other prompt by the same user / session, any prompt whose
    normalised hash is in heldout_v1_hashes.txt or that shares >= 1 8-gram with a held-out turn.
  * Eval decontamination: >= 1 shared 8-gram (template grams removed) with MMLU, ARC, HellaSwag, GSM8K test, PIQA,
    Winogrande, OBQA, CSQA, IFEval, MATH-500, BFCL -> dropped.
  * Different seed for the pool order; near-duplicates (8-gram containment >= 0.5) dropped within the pool.
Output: ~/.cache/udgam3_sft/real_pool_<version>.jsonl (+ gen/real_pool_<version>_stats.json).
  uv run --project evals/harness python -I udgam3/sft/gen/build_pool.py                  # v1 (pilot)
  uv run --project evals/harness python -I udgam3/sft/gen/build_pool.py --version v2     # full run
v2 (coordinator decision 2026-10-17): adds 13 more pinned WildChat-4.8M shards (same revision, sha-checked) and keeps
WildChat rows from BEFORE 2024-06 too (training only, not evaluation). The held-out exclusions are unchanged.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BH, CACHE, GEN, contamination, eval_index, heldout_index, sha_norm  # noqa: E402

SEED = 20261018
# extra pinned shards of allenai/WildChat-4.8M @ c827c6df8fcf008219ffaffa4d1dd77491099367 (size, LFS sha256)
EXTRA_WILDCHAT_SHARDS = {
    "00002": (115463083, "8cafa7a6d3387f8f15edc77b98108cbd8dedca5a927dbdb626ec720a718bfb2b"),
    "00006": (116894276, "64ed8e61ee03db406e8cd64e05dbdf61b412ef2ec10c76dcea7a43584efe0b5e"),
    "00010": (106465090, "5c0ae4433886ec1ea0dc4f7daa1f26574e75c471614b6d2420afd674a338f153"),
    "00014": (168003355, "8538243cf3cf6513c46658d4dc2da19f6bde729f3e04197abe9abac30aae8bfe"),
    "00018": (158654366, "83ee586e02d7a0d830e4be485483b3e015894f5964d0e73e6162f70ef8e00227"),
    "00022": (138085341, "08e96271de5893c70475028bac65d34afb59e4b63e8d6b6e93e8a73bfa170e81"),
    "00026": (166054540, "537f7dcd7744aa7890a5ef8cef9559eac68712d9cf1f0a20d06b5127d5c0f846"),
    "00030": (215958271, "d12254a456e9e17f1b8f9c62203d8951e18edcfcf491135043842c5de618f679"),
    "00034": (80445922, "cedf81b7f59646e2ef1aba9987dfd2761637e70708cc8276b0415c0b1b6fb659"),
    "00051": (156188404, "62561019494ce8f14da95c3ba5248a95499c4458eff8ff55a85ecb49bf87fb19"),
    "00054": (190323883, "090f7aeca356f1fc97a403d070350030dc2a46ed4ac609f94e6dfe7a944b810d"),
    "00067": (166945361, "2e6d04701794be220967da3e685acfeea466922b8acaabd8b41ef384932606f4"),
    "00081": (234968053, "5225536ee428b75536d4094c33647c588ee923d90891e66c066b22d7ce3a6eb7"),
}

RX = BH.RX
RELAXED_UNSAFE_RX = RX(
    r"\b(sex|sexy|sexual\w*|porn\w*|nsfw|nude\w*|nudity|naked|erotic\w*|fetish\w*|kink\w*|orgasm\w*|horny|boobs?|"
    r"breasts|penis|vagina|genital\w*|masturbat\w*|hentai|lewd|seduc\w*|stripper|onlyfans|xxx|condom\w*|"
    r"kill|kills|killing|killed|killer|murder\w*|suicid\w*|self[- ]?harm|cutting myself|rape\w*|molest\w*|abus\w*|"
    r"tortur\w*|gore|gory|corpse|dead body|behead\w*|stab\w*|shoot\w*|(got|been|was|were|get) shot|shot (him|her|them|dead)|bleed\w*|bled|zalgo|gun|guns|rifle|firearm\w*|weapon\w*|"
    r"bomb\w*|explosive\w*|terror\w*|massacre|genocide|nazi\w*|hitler|kkk|racis\w*|slur\w*|n-word|hate speech|"
    r"drug|drugs|cocaine|heroin|meth|methamphetamine|lsd|overdose|weed|marijuana|cannabis|"
    r"diaper\w*|vore|giantess|tickl\w*|foot fetish|spank\w*|bdsm|bondage|pee|poop|fart\w*|belly|bellies|"
    r"hypno\w*|mind control|hacking|hacker\w*|pirat\w*|torrent\w*|keylogger|crack(ed)? (version|software)|"
    r"malware|ransomware|phishing|exploit\w*|ddos|casino|gambl\w*|betting|scam\w*|product key|license key|"
    r"licence key|serial key|activation key|windows key|fraud\w*|steal\w*|assault\w*|abort\w*|depress\w*|"
    r"cheat\w*|affair|dosage|prescri\w*|pills?|"
    r"politic\w*|trump|biden|harris|modi|putin|zelensk\w*|israel\w*|palestin\w*|gaza|ukrain\w*|russia\w*|"
    r"democrat\w*|republican\w*|liberal|conservative|abortion|gender identity|transgender|lgbt\w*|election\w*|"
    r"caste|racial|ethnic\w*|skin colou?r|china|chinese (government|communist|party)|ccp|communist party|taiwan\w*|"
    r"tibet\w*|xinjiang|uyghur\w*|uighur\w*|tiananmen|hong kong|xi jinping|mao zedong|falun)\b")

DATASET = BH.DATASET_NAME
LICENCE = BH.LICENCE


def relaxed_filter(text: str, followup: bool = False) -> str | None:
    strict = BH.UNSAFE_RX
    try:
        BH.UNSAFE_RX = RELAXED_UNSAFE_RX
        return BH.filter_reason(text, followup=followup)
    finally:
        BH.UNSAFE_RX = strict


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    a = ap.parse_args()
    OUT = CACHE / f"real_pool_{a.version}.jsonl"
    STATS = GEN / f"real_pool_{a.version}_stats.json"
    t0 = time.time()
    stats: collections.Counter = collections.Counter()
    cache = Path(os.path.expanduser("~/.cache/udgam3_chat_prompts"))
    paths = BH.fetch_all(cache)
    if a.version == "v2":
        hf = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/" + BH.WILDCHAT_REV + "/data/"
        jobs = [(hf + f"train-{k}-of-00086.parquet", cache / "wildchat48m" / f"train-{k}-of-00086.parquet", sz, h)
                for k, (sz, h) in sorted(EXTRA_WILDCHAT_SHARDS.items())]
        with ThreadPoolExecutor(4) as ex:
            paths["wildchat"] = paths["wildchat"] + list(ex.map(lambda j: BH.fetch(*j), jobs))
        BH.WILDCHAT_CUTOFF = "0000"            # keep rows before 2024-06 too (training only)
    print(f"{len(paths['wildchat'])} WildChat shards", file=sys.stderr)
    cands = BH.load_wildchat(paths["wildchat"], stats) + BH.load_oasst2(paths["oasst2"], stats) + \
        BH.load_arena(paths["arena"], stats) + BH.load_arena100k(paths["arena100k"], stats)
    stats["candidates after source-level filters"] = len(cands)
    print(f"loaded {len(cands)} candidates in {time.time() - t0:.0f}s", file=sys.stderr)

    # held-out exclusion by source id and by user / session
    held_sids, held_rows = set(), [json.loads(l) for l in open(BH.OUT_JSONL, encoding="utf-8")]
    for r in held_rows:
        for part in r["source_id"].split(">"):
            held_sids.add(part)
    held_users = {c["user"] for c in cands if c["sid"] in held_sids}
    hashes, hidx = heldout_index()
    eidx = eval_index()

    mtb = [json.loads(l) for l in open(paths["mtbench"], encoding="utf-8")]
    mtb_index = BH.DupIndex()
    for q in mtb:
        for t in q["turns"]:
            mtb_index.add(t)

    passed = []
    for c in cands:
        if c["sid"] in held_sids:
            stats["heldout: same source id"] += 1
            continue
        if c["user"] in held_users:
            stats["heldout: same user / session as a held-out prompt"] += 1
            continue
        if c["sid"] in BH.HAND_EXCLUDE:
            stats["hand-excluded in the held-out review"] += 1
            continue
        why = relaxed_filter(c["t1"])
        if why:
            stats[f"filter: {why}"] += 1
            continue
        c["sensitive"] = bool(BH.UNSAFE_RX.search(c["t1"]))
        if c["t2"] is not None and relaxed_filter(c["t2"], followup=True):
            stats["follow-up filtered (kept single-turn)"] += 1
            c["t2"] = None
        passed.append(c)

    passed.sort(key=lambda c: hashlib.sha256(f"{SEED}:{c['src']}:{c['sid']}".encode()).hexdigest())
    pool_dups = BH.DupIndex()
    uniq = []
    for c in passed:
        if mtb_index.is_dup(c["t1"]):
            stats["dedup: overlaps MT-Bench"] += 1
            continue
        if pool_dups.is_dup(c["t1"]):
            stats["dedup: near-duplicate (8-gram) in pool"] += 1
            continue
        pool_dups.add(c["t1"])
        uniq.append(c)

    rows = []
    per_user = collections.Counter()
    for c in uniq:
        if c["sid"] in BH.HAND_CATEGORY:
            cat = (BH.HAND_CATEGORY[c["sid"]], "")
        else:
            cat = BH.classify(c["t1"])
        if cat is None:
            stats["category: no rule matched (dropped)"] += 1
            continue
        texts = [c["t1"]] + ([c["t2"]] if c["t2"] else [])
        if any(sha_norm(t) in hashes for t in texts):
            stats["heldout: hash match"] += 1
            continue
        if hidx.hits(c["t1"])[0] >= 1:
            stats["heldout: shares an 8-gram"] += 1
            continue
        if c["t2"] and hidx.hits(c["t2"])[0] >= 1:
            c["t2"] = None
        h, srcs = eidx.hits(c["t1"])
        if h >= 1:
            stats["eval decontam: shares an 8-gram (" + ",".join(srcs) + ")"] += 1
            continue
        if c["t2"] and eidx.hits(c["t2"])[0] >= 1:
            c["t2"] = None
        if per_user[c["user"]] >= 20:                  # cap heavy users (bot-like traffic)
            stats["per-user cap (20)"] += 1
            continue
        per_user[c["user"]] += 1
        t2 = None
        if c["t2"] is not None:
            t2s = c["t2"].strip()
            ok = (BH.norm_for_hash(t2s) != BH.norm_for_hash(c["t1"])
                  and not (len(BH.toks(t2s)) >= 8 and BH.DupIndex.pair(c["t1"], t2s))
                  and BH.FOLLOWUP_START.search(t2s) and BH.FOLLOWUP_REF.search(t2s)
                  and not BH.FOLLOWUP_BAD.search(t2s) and BH.nwords(t2s) <= 60 and not mtb_index.is_dup(t2s))
            if ok:
                t2 = t2s
        sid = c["sid"] if not (t2 and c.get("t2_sid")) else f"{c['sid']}>{c['t2_sid']}"
        rows.append({"pid": "rp-" + hashlib.sha256(f"{c['src']}:{c['sid']}".encode()).hexdigest()[:12],
                     "src": c["src"], "dataset": DATASET[c["src"]], "licence": LICENCE[c["src"]], "sid": sid,
                     "user": hashlib.sha256(str(c["user"]).encode()).hexdigest()[:12],
                     "category": cat[0], "subpool": cat[1], "t1": c["t1"].strip(), "t2": t2,
                     "t2_strong": bool(t2 and BH.FOLLOWUP_STRONG.search(t2)), "sensitive": c["sensitive"],
                     "words": BH.nwords(c["t1"])})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cats = collections.Counter(r["category"] for r in rows)
    by_src = collections.Counter(f"{r['category']} | {r['src']}" for r in rows)
    out = {"rows": len(rows), "with_followup": sum(1 for r in rows if r["t2"]),
           "strong_followups": sum(1 for r in rows if r["t2_strong"]),
           "sensitive_tagged": sum(1 for r in rows if r["sensitive"]), "categories": dict(cats),
           "by_category_source": dict(sorted(by_src.items())), "filter_stats": dict(sorted(stats.items())),
           "held_out_users_excluded": len(held_users), "seconds": round(time.time() - t0, 1), "path": str(OUT)}
    STATS.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("rows", "with_followup", "strong_followups", "sensitive_tagged",
                                          "categories", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
