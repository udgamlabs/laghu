"""Udgam-Loop (bytecore Stage 1) data-side features.

- Little-endian numbers: every run of digit tokens is reversed (ones digit first) in the token stream, at load time.
  tokenizer_v1 never merges digits, so this needs no re-tokenisation. A run starts at a space+digit token
  ('▁4') or at a digit after a non-digit; the leading space stays on the first token of the run.
- Program-generated arithmetic worksheets mixed into training at a fixed token fraction (answers computed by Python,
  so exact and licence-free).
- An arithmetic probe (exact-match greedy generation) scored natively during training, split into practice-length
  (<= the worksheets' max digits) and longer-than-practice numbers.
"""
import random

import numpy as np
import torch

# ---------------------------------------------------------------- digit tables
_TABLES = {}


def digit_tables(tokenizer):
    """(plain_ids[10], space_ids[10]) for '0'..'9' and '▁0'..'▁9'."""
    key = id(tokenizer)
    if key not in _TABLES:
        tok = tokenizer.tok
        plain = [tok.token_to_id(str(d)) for d in range(10)]
        space = [tok.token_to_id("▁" + str(d)) for d in range(10)]
        assert None not in plain and None not in space, "tokenizer lacks single-digit tokens"
        _TABLES[key] = (plain, space)
    return _TABLES[key]


class DigitReverser:
    """Vectorised little-endian transform on token-id arrays. Applying it twice is the identity."""

    def __init__(self, plain, space, vocab_size):
        n = max(vocab_size, max(plain + space) + 1)
        self.val = np.full(n, -1, dtype=np.int64)
        self.spaced = np.zeros(n, dtype=bool)
        for d in range(10):
            self.val[plain[d]] = d
            self.val[space[d]] = d
            self.spaced[space[d]] = True
        self.plain = np.array(plain, dtype=np.int64)
        self.space = np.array(space, dtype=np.int64)

    def __call__(self, ids):
        a = np.asarray(ids, dtype=np.int64)
        if a.size == 0:
            return a
        v = self.val[a]
        dig = v >= 0
        if not dig.any():
            return a
        sp = self.spaced[a]
        prev = np.concatenate([[False], dig[:-1]])
        start = dig & (sp | ~prev)
        ar = np.arange(a.size)
        start_pos = np.maximum.accumulate(np.where(start, ar, -1))
        # run end = next start or next non-digit, computed from the right
        nxt_break = np.where(np.concatenate([start[1:] | ~dig[1:], [True]]), ar, a.size)
        end_pos = np.minimum.accumulate(nxt_break[::-1])[::-1]
        src = np.where(dig, start_pos + end_pos - ar, ar)       # mirror inside the run
        out = a.copy()
        nv = v[src]
        first = dig & (ar == start_pos)
        lead_space = sp[np.where(dig, start_pos, ar)]
        out[dig] = np.where(first[dig] & lead_space[dig], self.space[nv[dig]], self.plain[nv[dig]])
        return out


# ---------------------------------------------------------------- synthetic arithmetic worksheets
SYNTH_MAX_DIGITS = 12   # practice range for addition / subtraction operands


def _num(rng, n):
    return rng.randrange(10) if n == 1 else rng.randrange(10 ** (n - 1), 10 ** n)


TEMPLATES = [
    "{a} {op} {b} = {c}",
    "Calculate {a} {op} {b}.\nAnswer: {c}",
    "What is {a} {opw} {b}? It is {c}.",
    "{a} {opw} {b} equals {c}.",
    "Q: {a} {op} {b}\nA: {c}",
]
OPS = {"+": "plus", "-": "minus", "*": "times"}


def worksheet(rng, n_problems=12):
    lines = []
    for _ in range(n_problems):
        r = rng.random()
        if r < 0.45:
            op = "+"
            a, b = _num(rng, rng.randint(1, SYNTH_MAX_DIGITS)), _num(rng, rng.randint(1, SYNTH_MAX_DIGITS))
            c = a + b
        elif r < 0.8:
            op = "-"
            a, b = _num(rng, rng.randint(1, SYNTH_MAX_DIGITS)), _num(rng, rng.randint(1, SYNTH_MAX_DIGITS))
            if b > a:
                a, b = b, a
            c = a - b
        else:
            op = "*"
            a, b = _num(rng, rng.randint(1, 6)), _num(rng, rng.randint(1, 6))
            c = a * b
        t = rng.choice(TEMPLATES)
        lines.append(t.format(a=a, b=b, c=c, op=op, opw=OPS[op]))
    return "\n".join(lines)


class SynthMixer:
    """Adds worksheet documents so they make up `frac` of the tokens of each refill (train split only)."""

    def __init__(self, frac, seed):
        self.frac, self.rng, self.debt = frac, random.Random(seed), 0.0

    def extra_docs(self, real_tokens, encode):
        self.debt += real_tokens * self.frac / (1.0 - self.frac)
        docs = []
        while self.debt > 0:
            d = encode(worksheet(self.rng))
            docs.append(d)
            self.debt -= len(d)
        return docs


# ---------------------------------------------------------------- loader configuration (set by base_train)
CONFIG = {"reverser": None, "synth_frac": 0.0, "seed": 0}


def configure(tokenizer, digits_reversed=False, synth_frac=0.0, seed=0):
    plain, space = digit_tables(tokenizer)
    CONFIG["reverser"] = DigitReverser(plain, space, tokenizer.get_vocab_size()) if digits_reversed else None
    CONFIG["synth_frac"] = synth_frac
    CONFIG["seed"] = seed


# ---------------------------------------------------------------- arithmetic probe
def probe_set(n_per_len=20, seed=777):
    """Addition/subtraction in the 'Calculate' template. Lengths 1..20 digits (practice: <= SYNTH_MAX_DIGITS)."""
    rng = random.Random(seed)
    items = []
    for n in range(1, 21):
        for _ in range(n_per_len):
            op = rng.choice("+-")
            a, b = _num(rng, n), _num(rng, rng.randint(1, n))
            if op == "-" and b > a:
                a, b = b, a
            c = a + b if op == "+" else a - b
            items.append((n, f"Calculate {a} {op} {b}.\nAnswer:", f" {c}"))
    return items


@torch.no_grad()
def eval_probe(model, tokenizer, device, items=None, batch=64):
    """Greedy exact match. Returns dict with practice / longer accuracy and per-length accuracy."""
    items = items or probe_set()
    rev = CONFIG["reverser"]
    bos = tokenizer.get_bos_token_id()
    plain, space = digit_tables(tokenizer)
    digit_set = set(plain) | set(space)
    prompts, golds = [], []
    for n, q, a in items:
        ids = tokenizer.encode(q, prepend=bos)
        full = tokenizer.encode(q + a, prepend=bos)
        if rev is not None:
            ids, full = list(rev(ids)), list(rev(full))
        assert full[:len(ids)] == ids, "prompt tokenisation must be a prefix of prompt+answer"
        prompts.append(ids)
        golds.append(full[len(ids):])
    was_training = model.training
    model.eval()
    ok = [False] * len(items)
    buckets = {}
    for i, p in enumerate(prompts):
        buckets.setdefault(len(p), []).append(i)   # same-length prompts batch without padding
    for _, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), batch):
            chunk = idxs[s:s + batch]
            seq = torch.tensor([prompts[i] for i in chunk], device=device)
            outs = []
            for _ in range(max(len(golds[i]) for i in chunk) + 1):
                nxt = model(seq)[:, -1].argmax(-1)
                outs.append(nxt)
                seq = torch.cat([seq, nxt[:, None]], 1)
            gen = torch.stack(outs, 1).tolist()
            for j, i in enumerate(chunk):
                g = golds[i]
                ok[i] = gen[j][:len(g)] == g and (gen[j][len(g)] not in digit_set)
    if was_training:
        model.train()
    by_len = {}
    for (n, _, _), r in zip(items, ok):
        c = by_len.setdefault(n, [0, 0])
        c[0] += r
        c[1] += 1
    acc = lambda lo, hi: sum(by_len[n][0] for n in by_len if lo <= n <= hi) / max(1, sum(by_len[n][1] for n in by_len if lo <= n <= hi))
    return {"practice": acc(1, SYNTH_MAX_DIGITS), "longer": acc(SYNTH_MAX_DIGITS + 1, 20),
            "by_len": {n: round(c[0] / c[1], 3) for n, c in sorted(by_len.items())}}
