"""Checkable instruction constraints (our own wording; generated together with their checker functions).

Each constraint = (id, group, make(rng, ctx) -> (instruction text, params)), and check(reply, params) -> bool.
Groups limit combinations: at most one constraint per group except `lexical` (up to two). Known conflicts are listed
in CONFLICTS. Prompts are later decontaminated against IFEval with the 8-gram index.
"""
from __future__ import annotations

import json
import re

from common import code_blocks, nwords, strip_code

STOP = set("""the a an and or but if then of to in on at for with from by about as into like through after over between
out against during without before under around among is are was were be been being have has had do does did i you he
she it we they me him her us them my your his its our their this that these those what which who whom whose when where
why how all any both each few more most other some such no nor not only own same so than too very can will just should
would could may might must shall there here please write make give tell explain help need want know some good best
using use list ways things ideas thing about something someone short simple""".split())


def sentences(text: str) -> list[str]:
    t = strip_code(text)
    t = re.sub(r"\b(e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|st|no|approx|u\.s)\.", lambda m: m.group(0).replace(".", "<d>"),
               t, flags=re.I)
    t = re.sub(r"(\d)\.(\d)", r"\1<d>\2", t)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])[\"')\]]*\s+|\n+", t) if p.strip()]
    return [p for p in parts if re.search(r"[A-Za-z]", p)]


def bullet_lines(text: str) -> list[str]:
    return [l for l in text.splitlines() if re.match(r"^\s*[-*•]\s+\S", l)]


def numbered_lines(text: str) -> list[str]:
    return [l for l in text.splitlines() if re.match(r"^\s*\d+[.)]\s+\S", l)]


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def has_markdown(text: str) -> bool:
    return bool(re.search(r"\*\*|__|^#{1,6}\s|^\s*[-*•]\s|^\s*\d+[.)]\s|```|^\|", text, re.M))


def table_rows(text: str) -> int:
    rows = [l for l in text.splitlines() if l.strip().startswith("|") and l.strip().endswith("|")]
    body = [r for r in rows if not re.fullmatch(r"\|[\s:|-]+\|", r.strip())]
    return max(0, len(body) - 1)


def content_word(prompt: str, rng) -> str | None:
    ws = [w for w in re.findall(r"[A-Za-z]{5,}", prompt) if w.lower() not in STOP]
    if not ws:
        return None
    ws = sorted(set(w.lower() for w in ws), key=lambda w: (-len(w), w))[:6]
    return rng.choice(ws)


INCLUDE_POOL = ["example", "simple", "because", "important", "together", "practice", "balance", "clear", "future",
                "careful", "everyday", "habit", "patience", "goal", "small", "quickly", "benefit", "plan"]
END_PHRASES = ["Hope this helps!", "Let me know if you have more questions.", "That is the short version.",
               "Good luck!", "Thanks for asking."]
START_WORDS = ["Sure", "Yes", "Here", "Absolutely", "First", "Okay", "Well"]
JSON_KEYS = [("answer", "reason"), ("title", "summary"), ("main_point", "example"), ("question", "answer"),
             ("pros", "cons"), ("topic", "key_facts")]
CODE_LANGS = [("JavaScript", {"javascript", "js"}), ("Java", {"java"}), ("C++", {"cpp", "c++"}), ("Go", {"go", "golang"}),
              ("Rust", {"rust"}), ("TypeScript", {"typescript", "ts"}), ("C#", {"csharp", "cs", "c#"})]


def _mk(cid, group, make, check, needs=None):
    return {"id": cid, "group": group, "make": make, "check": check, "needs": needs}


CONSTRAINTS = [
    _mk("max_words", "length",
        lambda r, c: (lambda n: (f"Keep your answer under {n} words.", {"n": n}))(r.choice([40, 50, 60, 80, 100, 120])),
        lambda t, p: nwords(t) < p["n"]),
    _mk("min_words", "length",
        lambda r, c: (lambda n: (f"Your answer should be at least {n} words long.", {"n": n}))(r.choice([100, 120, 150])),
        lambda t, p: nwords(t) >= p["n"], needs="long"),
    _mk("n_sentences", "format",
        lambda r, c: (lambda n: (f"Answer in exactly {n} sentences.", {"n": n}))(r.choice([2, 3, 4, 5])),
        lambda t, p: len(sentences(t)) == p["n"] and not bullet_lines(t) and not numbered_lines(t)),
    _mk("n_bullets", "format",
        lambda r, c: (lambda n: (f"Format the answer as exactly {n} bullet points, each starting with \"- \".",
                                 {"n": n}))(r.choice([3, 4, 5, 6])),
        lambda t, p: len(bullet_lines(t)) == p["n"] and not numbered_lines(t)),
    _mk("n_numbered", "format",
        lambda r, c: (lambda n: (f"Give the answer as a numbered list with exactly {n} items.", {"n": n}))(
            r.choice([3, 4, 5, 6, 7])),
        lambda t, p: len(numbered_lines(t)) == p["n"]),
    _mk("n_paragraphs", "format",
        lambda r, c: (lambda n: (f"Write exactly {n} paragraphs, separated by a blank line, with no lists or headings.",
                                 {"n": n}))(r.choice([2, 3, 4])),
        lambda t, p: len(paragraphs(t)) == p["n"] and not bullet_lines(t) and not numbered_lines(t)
        and not re.search(r"^#", t, re.M)),
    _mk("table", "format",
        lambda r, c: (lambda n: (f"Present the answer as a markdown table with {n} rows below the header row.",
                                 {"n": n}))(r.choice([3, 4, 5])),
        lambda t, p: table_rows(t) == p["n"]),
    _mk("json_keys", "format",
        lambda r, c: (lambda k: (f"Reply only with a JSON object with exactly the keys \"{k[0]}\" and \"{k[1]}\" "
                                 f"(string or list values). No code fences and no other text.", {"keys": list(k)}))(
            r.choice(JSON_KEYS)),
        lambda t, p: _json_keys_ok(t, p["keys"])),
    _mk("plain_text", "format",
        lambda r, c: ("Use plain text only: no markdown, no bullet points, no bold and no headings.", {}),
        lambda t, p: not has_markdown(t)),
    _mk("lowercase", "case",
        lambda r, c: ("Write your entire reply in lowercase letters, with no capital letters at all.", {}),
        lambda t, p: t == t.lower() and any(ch.isalpha() for ch in t)),
    _mk("uppercase", "case",
        lambda r, c: ("Write your entire reply in capital letters.", {}),
        lambda t, p: t == t.upper() and any(ch.isalpha() for ch in t)),
    _mk("title_line", "start",
        lambda r, c: ("Start with a title on the first line in the form \"Title: ...\".", {}),
        lambda t, p: t.lstrip().startswith("Title:") and len(t.strip().splitlines()) > 1),
    _mk("start_word", "start",
        lambda r, c: (lambda w: (f"Begin your reply with the word \"{w}\".", {"w": w}))(r.choice(START_WORDS)),
        lambda t, p: re.match(rf"^\W*{re.escape(p['w'])}\b", t.strip()) is not None),
    _mk("quote_wrap", "start",
        lambda r, c: ("Put your whole reply inside double quotation marks.", {}),
        lambda t, p: (lambda s: len(s) > 2 and s[0] in "\"“" and s[-1] in "\"”")(t.strip())),
    _mk("end_phrase", "end",
        lambda r, c: (lambda s: (f"Finish your reply with this exact sentence: \"{s}\"", {"s": s}))(r.choice(END_PHRASES)),
        lambda t, p: t.strip().rstrip("\"”").endswith(p["s"]) or t.strip().endswith(p["s"])),
    _mk("postscript", "end",
        lambda r, c: ("After the main answer, add one short line that begins with \"P.S.\".", {}),
        lambda t, p: re.search(r"^\s*P\.S\.", t, re.M) is not None),
    _mk("end_question", "end",
        lambda r, c: ("End your reply with a question for me.", {}),
        lambda t, p: t.strip().rstrip("\"”*").endswith("?")),
    _mk("no_commas", "lexical",
        lambda r, c: ("Do not use any commas in your reply.", {}),
        lambda t, p: "," not in t),
    _mk("include_words", "lexical",
        lambda r, c: (lambda ws: (f"Use the words \"{ws[0]}\" and \"{ws[1]}\" somewhere in your reply.",
                                  {"ws": ws}))(r.sample(INCLUDE_POOL, 2)),
        lambda t, p: all(re.search(rf"\b{re.escape(w)}\b", t, re.I) for w in p["ws"])),
    _mk("avoid_word", "lexical",
        lambda r, c: (lambda w: (f"Do not use the word \"{w}\" anywhere in your reply.", {"w": w}) if w else None)(
            content_word(c["prompt"], r)),
        lambda t, p: re.search(rf"\b{re.escape(p['w'])}\w*", t, re.I) is None),
    _mk("bold_n", "lexical",
        lambda r, c: (lambda n: (f"Highlight exactly {n} key phrases in **bold**.", {"n": n}))(r.choice([2, 3])),
        lambda t, p: len(re.findall(r"\*\*[^*\n]+\*\*", t)) == p["n"]),
    _mk("placeholders", "lexical",
        lambda r, c: (lambda n: (f"Include at least {n} placeholders in square brackets, such as [name].", {"n": n}))(
            r.choice([2, 3])),
        lambda t, p: len(re.findall(r"\[[^\]\n]{2,30}\]", t)) >= p["n"], needs="writing"),
    _mk("code_lang", "code",
        lambda r, c: (lambda L: (f"Write the code in {L[0]}.", {"lang": L[0], "tags": sorted(L[1])}))(
            r.choice(CODE_LANGS)),
        lambda t, p: _code_lang_ok(t, p), needs="code"),
]
BY_ID = {c["id"]: c for c in CONSTRAINTS}
CONFLICTS = {frozenset(x) for x in [
    ("lowercase", "title_line"), ("lowercase", "end_phrase"), ("lowercase", "start_word"), ("uppercase", "end_phrase"),
    ("uppercase", "start_word"), ("uppercase", "title_line"), ("lowercase", "json_keys"), ("uppercase", "json_keys"),
    ("json_keys", "title_line"), ("json_keys", "start_word"), ("json_keys", "end_phrase"), ("json_keys", "postscript"),
    ("json_keys", "end_question"), ("json_keys", "bold_n"), ("json_keys", "quote_wrap"), ("json_keys", "min_words"),
    ("plain_text", "bold_n"), ("plain_text", "table"), ("n_paragraphs", "title_line"), ("n_sentences", "title_line"),
    ("n_sentences", "postscript"), ("n_sentences", "min_words"), ("n_paragraphs", "postscript"),
    ("table", "n_sentences"), ("table", "min_words"), ("table", "no_commas"), ("quote_wrap", "postscript"),
    ("n_bullets", "min_words"), ("n_numbered", "min_words"), ("max_words", "table"), ("json_keys", "no_commas"),
    ("n_sentences", "end_phrase"), ("bold_n", "lowercase"), ("bold_n", "uppercase"), ("lowercase", "postscript"),
    ("uppercase", "postscript"), ("lowercase", "placeholders"), ("n_sentences", "placeholders"),
]}


def _json_keys_ok(t: str, keys: list[str]) -> bool:
    try:
        o = json.loads(t.strip())
    except json.JSONDecodeError:
        return False
    return isinstance(o, dict) and sorted(o) == sorted(keys)


def _code_lang_ok(t: str, p: dict) -> bool:
    blocks = code_blocks(t)
    if not blocks:
        return False
    tags = set(p["tags"])
    return all(lang in tags for lang, _ in blocks) and not any(lang in ("python", "py") for lang, _ in blocks)


def sample_constraints(rng, prompt: str, kind: str, k: int) -> list[dict] | None:
    """Pick k compatible constraints for a base prompt of the given kind ('general' | 'writing' | 'code')."""
    pool = [c for c in CONSTRAINTS if c["needs"] in (None, kind) or (c["needs"] == "long" and kind in ("writing",
                                                                                                        "long"))]
    if kind == "code":
        pool = [c for c in pool if c["id"] in ("code_lang", "max_words", "end_question", "start_word", "title_line")]
    for _ in range(30):
        pick = rng.sample(pool, k)
        groups = [c["group"] for c in pick]
        if any(groups.count(g) > (2 if g == "lexical" else 1) for g in groups):
            continue
        ids = [c["id"] for c in pick]
        if any(frozenset((a, b)) in CONFLICTS for i, a in enumerate(ids) for b in ids[i + 1:]):
            continue
        if kind == "code" and "code_lang" not in ids:
            continue
        out = []
        for c in pick:
            made = c["make"](rng, {"prompt": prompt})
            if made is None:
                break
            text, params = made
            out.append({"id": c["id"], "text": text, "params": params})
        else:
            return out
    return None


def check_all(reply: str, cons: list[dict]) -> dict[str, bool]:
    return {c["id"]: bool(BY_ID[c["id"]]["check"](reply, c["params"])) for c in cons}
