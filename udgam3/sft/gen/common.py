"""Shared helpers for the Udgam-3 SFT generator: paths, text measures, stable 8-gram indexes (held-out chat prompts and
eval sets), teacher-fingerprint / language / PII checks, and a near-duplicate index.

8-grams use the held-out builder's normalisation (`norm_for_hash`: NFKC, lower-case, non-alphanumeric runs -> one
space) so that the held-out exclusion matches exactly how `heldout_v1_hashes.txt` was made. Gram ids are stable
(blake2b-64), so indexes can be saved and reloaded.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

GEN = Path(__file__).resolve().parent
SFT = GEN.parent
U3 = SFT.parent
REPO = U3.parent
CACHE = Path.home() / ".cache" / "udgam3_sft"
CACHE.mkdir(parents=True, exist_ok=True)
HELDOUT_JSONL = U3 / "evals" / "chat_prompts" / "heldout_v1.jsonl"
HELDOUT_HASHES = U3 / "evals" / "chat_prompts" / "heldout_v1_hashes.txt"
EVAL8 = CACHE / "eval8_v1.npz"


def load_builder():
    """Import udgam3/evals/chat_prompts/build_heldout.py as a module (filters, loaders, classify, norm_for_hash)."""
    p = U3 / "evals" / "chat_prompts" / "build_heldout.py"
    spec = importlib.util.spec_from_file_location("build_heldout", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BH = load_builder()
norm_for_hash = BH.norm_for_hash


def sha_norm(text: str) -> str:
    return hashlib.sha256(norm_for_hash(text).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ text measures
CODE_FENCE = re.compile(r"```.*?(```|$)", re.S)


def nwords(text: str) -> int:
    return len((text or "").split())


def strip_code(text: str) -> str:
    return CODE_FENCE.sub(" ", text or "")


def prose_words(text: str) -> int:
    return nwords(strip_code(text))


def code_blocks(text: str) -> list[tuple[str, str]]:
    """[(language tag, code)] for every fenced block."""
    return [(m.group(1).strip().lower(), m.group(2)) for m in
            re.finditer(r"```([\w+#.-]*)[^\n]*\n(.*?)```", text or "", re.S)]


def code_lines(text: str) -> int:
    return sum(c.count("\n") + 1 for _, c in code_blocks(text))


# ------------------------------------------------------------------------------------------------ stable 8-grams
def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def grams8(text: str, n: int = 8) -> set[int]:
    t = norm_for_hash(text).split()
    return {_h64(" ".join(t[i:i + n])) for i in range(len(t) - n + 1)}


class GramIndex:
    """8-gram index over reference items; grams shared by >= `common` distinct items (templates) are removed."""

    def __init__(self, keys: np.ndarray, src: np.ndarray, names: list[str]):
        self.keys, self.src, self.names = keys, src, names

    @classmethod
    def build(cls, items, names: list[str], common: int = 3):
        """items: iterable of (name, item_id, text)."""
        ids: dict = {}
        K, S, I = [], [], []
        for name, item, text in items:
            g = grams8(text)
            if not g:
                continue
            iid = ids.setdefault(item, len(ids))
            K.append(np.fromiter(g, dtype=np.uint64, count=len(g)))
            S.append(np.full(len(g), names.index(name), dtype=np.uint8))
            I.append(np.full(len(g), iid, dtype=np.int64))
        if not K:
            return cls(np.empty(0, np.uint64), np.empty(0, np.uint8), names), 0
        k, s, it = np.concatenate(K), np.concatenate(S), np.concatenate(I)
        order = np.lexsort((it, k))
        k, s, it = k[order], s[order], it[order]
        new_pair = np.ones(k.size, bool)
        new_pair[1:] = (k[1:] != k[:-1]) | (it[1:] != it[:-1])
        kk, ss = k[new_pair], s[new_pair]
        uk, first, cnt = np.unique(kk, return_index=True, return_counts=True)
        keep = cnt < common
        return cls(uk[keep], ss[first][keep], names), int((~keep).sum())

    def save(self, path: Path):
        np.savez(path, keys=self.keys, src=self.src, names=np.array(self.names))

    @classmethod
    def load(cls, path: Path):
        z = np.load(path)
        return cls(z["keys"], z["src"], [str(x) for x in z["names"]])

    def hits(self, text: str) -> tuple[int, list[str]]:
        g = grams8(text)
        if not g or self.keys.size == 0:
            return 0, []
        a = np.fromiter(g, dtype=np.uint64, count=len(g))
        pos = np.searchsorted(self.keys, a)
        pos[pos >= self.keys.size] = 0
        m = self.keys[pos] == a
        if not m.any():
            return 0, []
        srcs = sorted({self.names[int(s)] for s in self.src[pos[m]]})
        return int(m.sum()), srcs


_HELDOUT = None
_EVAL = None


def heldout_index():
    """(set of sha256 hashes, GramIndex over held-out turns)."""
    global _HELDOUT
    if _HELDOUT is None:
        hashes = {l.strip() for l in HELDOUT_HASHES.open() if l.strip()}
        items = []
        for line in HELDOUT_JSONL.open(encoding="utf-8"):
            r = json.loads(line)
            for t in r["turns"]:
                items.append(("heldout", r["id"], t))
        idx, _ = GramIndex.build(items, ["heldout"], common=3)
        _HELDOUT = (hashes, idx)
    return _HELDOUT


def eval_index():
    global _EVAL
    if _EVAL is None:
        if not EVAL8.exists():
            raise SystemExit(f"missing {EVAL8}: run `python -I decontam8.py build` first")
        _EVAL = GramIndex.load(EVAL8)
    return _EVAL


def contamination(texts: list[str], *, long_doc: bool = False) -> dict | None:
    """None if clean, else {'why', ...}. Rule: any user text whose normalised hash is a held-out hash, or that shares
    >= 1 8-gram with a held-out prompt, or shares >= 1 (>= 3 for long documents) 8-gram with an eval item."""
    hashes, hidx = heldout_index()
    eidx = eval_index()
    for t in texts:
        if not t:
            continue
        if sha_norm(t) in hashes:
            return {"why": "heldout_hash"}
        h, _ = hidx.hits(t)
        if h >= 1:
            return {"why": "heldout_8gram", "hits": h}
        h, srcs = eidx.hits(t)
        if h >= (3 if long_doc else 1):
            return {"why": "eval_8gram", "hits": h, "evals": srcs}
    return None


# ------------------------------------------------------------------------------------------------ output checks
FINGERPRINT_RX = re.compile(
    r"\bdeep ?seek\b|深度求索|\bopen ?ai\b|\bchat ?gpt\b|\bgpt-?[2345](\.\d)?o?\b|\banthropic\b|"
    r"\bclaude (?:\d|ai|opus|sonnet|haiku)\b|\bgoogle (?:gemini|bard)\b|\bgemini (?:pro|ultra|flash|nano|\d)\b|"
    r"\bqwen\b|\bllama[- ]?\d|\bmeta'?s llama\b|\bmistral (?:ai|large|\d)|"
    r"\bas an ai\b|\bas an artificial intelligence\b|\bas a (large )?language model\b|\bi(?:'m| am) an ai\b|"
    r"\bi(?:'m| am) (?:just )?an? (?:ai|artificial intelligence|language model|chatbot|virtual assistant)\b|"
    r"\b(?:knowledge|training)[- ](?:cut-?off|cutoff)\b|\bas of my (?:last|latest|most recent) "
    r"(?:update|training|knowledge)\b|\bmy training data\b|\bmy (?:knowledge|information) (?:only )?"
    r"(?:goes|extends|is current) (?:up )?(?:to|until|through)\b|\bi was (?:trained|created|developed|made|built) by\b|"
    r"\bmy (?:creators|developers)\b|\b(?:trained|developed|created) by (?:a team|engineers)\b", re.I)


def fingerprint(text: str) -> str | None:
    m = FINGERPRINT_RX.search(text or "")
    return m.group(0) if m else None


NON_LATIN_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "DEVANAGARI", "THAI", "HEBREW",
                     "BENGALI", "TAMIL", "TELUGU", "GUJARATI", "GURMUKHI", "KANNADA", "MALAYALAM")


def english_output_ok(text: str) -> bool:
    """Assistant text: no run of another script (2+ letters) and >= 97% ASCII letters."""
    letters = [c for c in strip_code(text) if c.isalpha()]
    if not letters:
        return True
    bad = sum(unicodedata.name(c, "").startswith(NON_LATIN_SCRIPTS) for c in letters)
    if bad >= 2:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= 0.97


OUT_PII_RX = [BH.PII_RX[0], BH.PII_RX[1], BH.PII_RX[2]]      # e-mail, phone, long digit ids


def output_pii(text: str) -> bool:
    t = strip_code(text)
    return any(r.search(t) for r in OUT_PII_RX)


# ------------------------------------------------------------------------------------------------ near-duplicates
class DupIndex:
    """Same normalised text, or 8-gram containment >= thr of the shorter text."""

    def __init__(self, thr: float = 0.5):
        self.thr = thr
        self.exact: set[str] = set()
        self.grams: dict[int, set[int]] = collections.defaultdict(set)
        self.sizes: list[int] = []

    def is_dup(self, text: str) -> bool:
        n = norm_for_hash(text)
        if n in self.exact:
            return True
        g = grams8(text)
        if not g:
            return False
        hits = collections.Counter()
        for x in g:
            for j in self.grams.get(x, ()):
                hits[j] += 1
        return any(c / min(len(g), self.sizes[j]) >= self.thr for j, c in hits.items())

    def add(self, text: str):
        self.exact.add(norm_for_hash(text))
        g = grams8(text)
        j = len(self.sizes)
        self.sizes.append(max(1, len(g)))
        for x in g:
            self.grams[x].add(j)


def containment(a: str, b: str) -> float:
    """Share of the shorter text's 8-grams that also occur in the other text."""
    ga, gb = grams8(a), grams8(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)
