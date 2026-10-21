"""13-gram decontamination against the evaluation sets UdgamLabs will report.

Build the index (small, ~minutes, streams only the eval files):
    python decontam.py build [--out data/decontam/index_13gram.npz]
Check a text file / string:
    python decontam.py check "some text ..."

Normalisation (identical for eval items and training docs): lowercase -> `\\w+` word tokens
(punctuation and whitespace runs are dropped). Each word -> crc32 (32 bit); a 13-gram is a
position-weighted sum of its 13 word hashes mod 2^64 (random odd 64-bit multipliers), so a doc's
grams are computed with ~13 vectorised numpy ops. The index is a sorted uint64 array (+ a parallel
uint8 array saying which eval set the gram came from); membership is a searchsorted.

Drop rule (used by build_mix.py, see `is_contaminated`):
  * docs with <= LONG_DOC_WORDS words: drop on >= SHORT_DOC_MIN_HITS (2) distinct shared 13-grams. A doc
    that contains an eval item of >= 14 words shares >= 2 grams with it, so this still catches every copied
    item except the few exactly-13-word ones; single-gram hits on the test sample were all instruction
    template wording (e.g. IFEval's "At the end of your response, please explicitly add a postscript starting
    with P.S.") inside ordinary chat data. Single-hit docs are counted (`single_hit_kept`) for audit.
    (Use --strict / SHORT_DOC_MIN_HITS=1 for the literal ">= 1 gram" rule.)
  * longer docs: drop on >= LONG_DOC_MIN_HITS distinct shared 13-grams. One or two hits in a long web page
    / Wikipedia article / code file are almost always a famous quotation, legal boilerplate or a WikiHow
    sentence that an eval item happens to quote, while a doc that really contains an eval item contains
    the whole item -> (item_words - 12) >= 3 grams for any item >= 15 words. Matches are logged so the
    threshold can be audited in manifest.json.
Template n-grams: a gram that occurs in >= COMMON_ITEMS (3) distinct eval items (e.g. IFEval's "at the end of your
response, please explicitly add a postscript starting with ..." constraint wording, MMLU/ARC stock phrasing) is
removed from the index. Those are instruction/question templates shared by many items, not an item's content;
keeping them flagged ~4% of ordinary instruction-following chat data as "IFEval-contaminated".
Eval items shorter than 13 words cannot be indexed; their count is reported in the index stats.
"""
import collections, json, os, re, sys, zlib, argparse, time
import numpy as np

N = 13
LONG_DOC_WORDS = 1500
LONG_DOC_MIN_HITS = 3
SHORT_DOC_MIN_HITS = 2
COMMON_ITEMS = 3
_WORD = re.compile(r"\w+")
_rng = np.random.default_rng(0x5EED13)
MULT = (_rng.integers(1, 2**63 - 1, size=N, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1))

EVAL_NAMES = ["mmlu", "arc", "hellaswag", "gsm8k", "piqa", "winogrande", "openbookqa", "commonsenseqa", "ifeval", "bfcl", "math500"]
DEFAULT_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/decontam/index_13gram.npz")


def words(text):
    return _WORD.findall(text.lower())


def ngram_hashes(text, n=N):
    """uint64 hashes of all word n-grams of `text` (empty array if fewer than n words). Also returns #words."""
    w = words(text)
    L = len(w)
    if L < n:
        return np.empty(0, dtype=np.uint64), L
    wh = np.fromiter((zlib.crc32(x.encode()) for x in w), dtype=np.uint64, count=L)
    m = L - n + 1
    g = wh[0:m] * MULT[0]
    for j in range(1, n):
        g = g + wh[j:j + m] * MULT[j]          # uint64 arithmetic wraps mod 2^64
    return g, L


class Index:
    def __init__(self, path=DEFAULT_INDEX):
        z = np.load(path)
        self.keys, self.src = z["keys"], z["src"]
        self.names = list(z["names"])

    def hits(self, text):
        """(#distinct matched grams, #words, list of eval-set names matched)."""
        g, L = ngram_hashes(text)
        if g.size == 0 or self.keys.size == 0:
            return 0, L, []
        pos = np.searchsorted(self.keys, g)
        pos[pos >= self.keys.size] = 0
        m = self.keys[pos] == g
        if not m.any():
            return 0, L, []
        mg = np.unique(g[m])
        srcs = sorted({self.names[s] for s in self.src[np.searchsorted(self.keys, mg)]})
        return int(mg.size), L, srcs

    def is_contaminated(self, text):
        h, L, srcs = self.hits(text)
        if h == 0:
            return False, h, srcs
        need = SHORT_DOC_MIN_HITS if L <= LONG_DOC_WORDS else LONG_DOC_MIN_HITS
        return h >= need, h, srcs


# ----------------------------------------------------------------------------- eval-set readers
def _pq(path, cols=None):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem()
    return pq.read_table(fs.open("datasets/" + path, block_size=8 << 20), columns=cols).to_pylist()


def _jsonl(path):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open("datasets/" + path) as f:
        return [json.loads(l) for l in f.read().decode().splitlines() if l.strip()]


def eval_items():
    """Yield (eval_name, item_id, text) for every eval item we will report on (several texts may share an item)."""
    import itertools
    _ID = itertools.count()
    for r in _pq("cais/mmlu/all/test-00000-of-00001.parquet"):
        yield "mmlu", next(_ID), r["question"] + "\n" + "\n".join(map(str, r["choices"]))
    for sub in ["ARC-Challenge", "ARC-Easy"]:
        for r in _pq(f"allenai/ai2_arc/{sub}/test-00000-of-00001.parquet"):
            yield "arc", next(_ID), r["question"] + "\n" + "\n".join(r["choices"]["text"])
    for r in _pq("Rowan/hellaswag/data/validation-00000-of-00001.parquet"):
        i = next(_ID)
        yield "hellaswag", i, r["ctx"] + " " + " ".join(r["endings"])
        for e in r["endings"]:                       # each candidate completion as the eval scores it
            yield "hellaswag", i, r["ctx"] + " " + e
    for r in _pq("openai/gsm8k/main/test-00000-of-00001.parquet"):
        yield "gsm8k", next(_ID), r["question"] + "\n" + r["answer"]
    for split in ["validation", "test"]:
        for r in _pq(f"ybisk/piqa@refs%2Fconvert%2Fparquet/plain_text/{split}/0000.parquet"):
            i = next(_ID)
            yield "piqa", i, r["goal"] + " " + r["sol1"]
            yield "piqa", i, r["goal"] + " " + r["sol2"]
    for split in ["validation", "test"]:
        for r in _pq(f"allenai/winogrande/winogrande_xl/{split}-00000-of-00001.parquet"):
            i = next(_ID)
            for o in (r["option1"], r["option2"]):
                yield "winogrande", i, r["sentence"].replace("_", o)
    for split in ["test", "validation"]:
        for r in _pq(f"allenai/openbookqa/main/{split}-00000-of-00001.parquet"):
            yield "openbookqa", next(_ID), r["question_stem"] + "\n" + "\n".join(r["choices"]["text"])
    for split in ["validation", "test"]:
        for r in _pq(f"tau/commonsense_qa/data/{split}-00000-of-00001.parquet"):
            yield "commonsenseqa", next(_ID), r["question"] + "\n" + "\n".join(r["choices"]["text"])
    for r in _jsonl("google/IFEval/ifeval_input_data.jsonl"):
        yield "ifeval", next(_ID), r["prompt"]
    for r in _jsonl("HuggingFaceH4/MATH-500/test.jsonl"):          # Udgam-3 (2026-10-17): MATH is a likely claim
        yield "math500", next(_ID), r["problem"] + "\n" + r["solution"]
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    for f in sorted(fs.ls("datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard", detail=False)):
        if not (f.endswith(".json") and "BFCL_v3_" in f):
            continue
        for r in _jsonl(f.split("datasets/", 1)[1]):
            i = next(_ID)
            # Questions only: function docs are generic API descriptions that legitimately appear in code
            # and docs; what must not leak is the user request (+ its answer, which is not text-like).
            for turn in r.get("question", []):
                for msg in (turn if isinstance(turn, list) else [turn]):
                    if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                        # BFCL variants (base / miss_func / long_context ...) repeat the same question: key by text
                        yield "bfcl", 10**12 + zlib.crc32(msg["content"].encode()), msg["content"]


def build(out):
    t0 = time.time()
    keys, srcs, items = [], [], []
    stats = {n: {"texts": 0, "too_short": 0, "grams": 0} for n in EVAL_NAMES}
    for name, item, text in eval_items():
        s = stats[name]
        s["texts"] += 1
        g, L = ngram_hashes(text)
        if g.size == 0:
            s["too_short"] += 1
            continue
        g = np.unique(g)
        s["grams"] += int(g.size)
        keys.append(g)
        srcs.append(np.full(g.size, EVAL_NAMES.index(name), dtype=np.uint8))
        items.append(np.full(g.size, item, dtype=np.int64))
    k = np.concatenate(keys)
    sr = np.concatenate(srcs)
    it = np.concatenate(items)
    # distinct items per gram -> drop template grams shared by >= COMMON_ITEMS items
    order = np.lexsort((it, k))
    k, sr, it = k[order], sr[order], it[order]
    new_pair = np.ones(k.size, bool)
    new_pair[1:] = (k[1:] != k[:-1]) | (it[1:] != it[:-1])
    kk, ss = k[new_pair], sr[new_pair]
    uk, first, cnt = np.unique(kk, return_index=True, return_counts=True)
    common = cnt >= COMMON_ITEMS
    removed = collections.Counter(EVAL_NAMES[x] for x in ss[first][common])
    for n in EVAL_NAMES:
        stats[n]["template_grams_removed"] = removed.get(n, 0)
    k, sr = uk[~common], ss[first][~common]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, keys=k, src=sr, names=np.array(EVAL_NAMES))
    meta = {"n": N, "unique_grams": int(k.size), "per_eval": stats, "seconds": round(time.time() - t0, 1),
            "long_doc_words": LONG_DOC_WORDS, "long_doc_min_hits": LONG_DOC_MIN_HITS, "short_doc_min_hits": SHORT_DOC_MIN_HITS, "common_items": COMMON_ITEMS}
    json.dump(meta, open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("text", nargs="?")
    ap.add_argument("--out", default=DEFAULT_INDEX)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out)
    else:
        ix = Index(a.out)
        t = open(a.text).read() if a.text and os.path.exists(a.text) else (a.text or sys.stdin.read())
        print(ix.is_contaminated(t))
