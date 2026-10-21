"""Shared by the chat-suite worker (Colab VM) and the scorer (Mac). Standard library only.

Prompt suffixes, the GSM8K answer rule and the multi-turn helpers are copied unchanged from the pre-registered scripts
(udgam2/eval/math_eval.py, udgam2/multiturn/mt_eval.py) so every model is scored by the same rules.
"""
import difflib, hashlib, re

# udgam2/eval/math_eval.py and udgam2/multiturn/mt_eval.py
SUFFIX = "\n\nSolve it step by step, then give the final answer on the last line as: Final answer: <number>"
# MATH-500 answers are expressions, not numbers; the same instruction with <answer> (written 2026-10-19, before any
# Udgam-3 chat model exists)
MATH_SUFFIX = "\n\nSolve it step by step, then give the final answer on the last line as: Final answer: <answer>"

CAPS = dict(gsm8k=4096, math=4096, ifeval=1280, mt1=1024, mt2=1024, e2=4096, loop=1024, ref=512)
E2_N = 200          # mt_eval.py E2: turn 1 = GSM8K test item i (with its reference solution), turn 2 = item E2_N + i
REF_N = 40          # PREREGISTRATION.md 2.4: unbatched reference on ~40 items

GEMMA4_THOUGHT = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.S)


def split_thinking(raw):
    """(thinking, answer, unclosed). Gemma-4 thought channels are removed as in math_eval.py; a <think> block ends at the
    last </think>, as lm-eval's think_end_token does. An unclosed block (cut by the cap) stays in the answer, as in both."""
    thoughts = GEMMA4_THOUGHT.findall(raw)
    ans = GEMMA4_THOUGHT.sub("", raw)
    if "</think>" in ans:
        pre, ans = ans.rsplit("</think>", 1)
        thoughts.append(pre.split("<think>", 1)[-1])
        ans = ans.lstrip()
    unclosed = "<|channel>thought" in ans or "<think>" in ans
    return "\n".join(thoughts), ans, unclosed


def clean(text, specials=(), stop_strings=()):
    """Visible reply: cut at a plain-text turn marker, then drop special-token strings (what skip_special_tokens does)."""
    for s in stop_strings:
        text = text.split(s, 1)[0]
    for s in sorted(specials, key=len, reverse=True):
        if s:
            text = text.replace(s, "")
    return text


def visible(raw, specials=(), stop_strings=()):
    return clean(split_thinking(raw)[1], specials, stop_strings)


# udgam2/eval/math_eval.py
def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", "").replace("$", ""))
    return m[-1] if m else None


def gsm8k_pred(answer):
    """math_eval.py: the number after the last "Final answer", else the last number in the reply."""
    return num(answer.split("Final answer")[-1]) if "Final answer" in answer else num(answer)


def gsm8k_ok(answer, gold):
    p = gsm8k_pred(answer)
    return p is not None and abs(float(p) - float(gold)) < 1e-6


# udgam2/multiturn/mt_eval.py
def sim(x, y):
    return difflib.SequenceMatcher(None, x[:400], y[:400]).ratio()


def hit(text, checks):
    return any(re.search(r"(?<![\w.])" + re.escape(c) + r"(?![\w])", text, re.I) for c in checks)


def ref_solution(sol):
    body, gold = sol.split("####")
    return re.sub(r"<<[^>]*>>", "", body).strip() + f"\nFinal answer: {gold.strip()}"


def periodic_tail(ids, window=256, max_period=128, thresh=0.8):
    """Smallest period p <= max_period such that >= thresh of the last `window` tokens equal the token p back; 0 if none.
    A reply loops when it hits the cap without stopping AND its tail is periodic (written 2026-10-19 to replace
    Ollama's "token repeat limit" abort, which loop_check.py relied on and other engines don't have)."""
    if len(ids) < window:
        return 0
    t = ids[-window:]
    for p in range(1, max_period + 1):
        if sum(1 for i in range(p, window) if t[i] == t[i - p]) >= thresh * (window - p):
            return p
    return 0


def sha1_ids(ids):
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:16]
