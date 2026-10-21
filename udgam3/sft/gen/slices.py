"""Per-slice task builders. Each task is an async function of a Ctx that makes its DeepSeek calls, verifies the
result and returns a list of outcomes: {"ok", "id", "slice", "category", "row" | None, "reason", "checks", "debug"}.

All randomness is seeded by (plan seed, task id), so a re-run sends byte-identical requests and is served from the
raw-response cache for free (that is how the generator resumes).
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import prompts as P
from client import GENERATOR, SpendCapReached
from common import (BH, CACHE, REPO, code_blocks, code_lines, containment, nwords, prose_words, strip_code)
from constraints import check_all, sample_constraints
from verify import (answer_in_text, args_ok, arun_python, check_code, exact_output_equal, flat_values, logic_equal,
                    num_equal, schema_ok, to_number)

FACTORY = REPO / "datafactory" / "out"
FACTORY_NOTE = "model-written problem (UdgamLabs data factory); provider terms restrict competing-model use; SFT-only"
DS_LICENCE = "MIT (DeepSeek-V4.1-Flash output)"
PY_TOOL = {"type": "function", "function": {"name": "python", "description": "Run Python code and return what it prints.",
           "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}},
                          "required": ["code"]}}}


@dataclass
class Ctx:
    ds: object
    mv: object
    seed: int
    judge: bool = True


@dataclass
class Task:
    id: str
    slice: str
    category: str
    fn: Callable[[Ctx, "Task"], Awaitable[list[dict]]]
    data: dict = field(default_factory=dict)

    def rng(self, ctx: Ctx) -> random.Random:
        return random.Random(f"{ctx.seed}:{self.id}")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def prov(source: str, licence: str, pid: str, **extra) -> dict:
    d = {"prompt_source": source, "prompt_licence": licence, "prompt_id": pid, "generator": GENERATOR,
         "generated_utc": now()}
    d.update(extra)
    return d


def row(task: Task, messages: list[dict], thinking: bool, verified: dict, provenance: dict, tools=None, rid=None,
        category=None, **extra) -> dict:
    r = {"id": rid or task.id, "slice": task.slice, "category": category or task.category, "messages": messages}
    if tools:
        r["tools"] = tools
    r["enable_thinking"] = thinking
    r["verified"] = verified
    r["provenance"] = provenance
    r.update(extra)
    return r


def ok(task, r, checks, rid=None, debug=None):
    return {"ok": True, "id": rid or r["id"], "slice": task.slice, "category": r["category"], "row": r,
            "reason": None, "checks": checks, "debug": debug or {}}


def bad(task, reason, checks=None, debug=None, rid=None, row_=None):
    return {"ok": False, "id": rid or task.id, "slice": task.slice, "category": task.category, "row": row_,
            "reason": reason, "checks": checks or {}, "debug": debug or {}}


def parse_json(text: str):
    o = _parse_json(text)
    if isinstance(o, dict) and set(o) <= {"type", "content", "value", "json"}:
        for k in ("content", "value", "json"):      # teacher sometimes wraps the object as {"type", "content"|"value"}
            if isinstance(o.get(k), dict):
                return o[k]
    return o


def _parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def transcript(msgs: list[dict], cap_words: int = 260) -> str:
    out = []
    for m in msgs:
        if m["role"] == "system":
            continue
        c = m.get("content") or ""
        w = c.split()
        if len(w) > cap_words:
            c = " ".join(w[:cap_words]) + " [...]"
        out.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {c}")
    return "\n\n".join(out)


# ------------------------------------------------------------------------------------------------ shared steps
async def teacher_answer(ctx, task, msgs, category, system_extra="", temperature=0.8, max_tokens=900):
    sys_text = P.STYLE + ("\n" + P.CATEGORY_NOTES[category] if category in P.CATEGORY_NOTES else "")
    if system_extra:
        sys_text += "\n" + system_extra
    r = await ctx.ds.chat([{"role": "system", "content": sys_text}] + msgs, slice_name=task.slice,
                          temperature=temperature, max_tokens=max_tokens)
    return r


async def gen_followup(ctx, task, msgs, kind):
    r = await ctx.ds.chat([{"role": "system", "content": P.FOLLOWUP_SYS},
                           {"role": "user", "content": P.FOLLOWUP_USER.format(transcript=transcript(msgs),
                                                                              kind=P.FOLLOWUP_TYPES[kind])}],
                          slice_name=task.slice, json_mode=True, temperature=1.0, max_tokens=120)
    o = parse_json(r["content"]) or {}
    return (o.get("message") or "").strip()


async def judge(ctx, task, msgs, system_note=""):
    if not ctx.judge:
        return 5, ""
    r = await ctx.ds.chat([{"role": "system", "content": P.JUDGE_SYS},
                           {"role": "user", "content": P.JUDGE_USER.format(
                               system_note=(f"System message the assistant had to follow: \"{system_note}\"\n"
                                            if system_note else ""), transcript=transcript(msgs, 320))}],
                          slice_name=task.slice + ":judge", json_mode=True, temperature=0.0, max_tokens=60)
    o = parse_json(r["content"]) or {}
    try:
        return int(o.get("score", 0)), str(o.get("problem", ""))
    except (TypeError, ValueError):
        return 0, "unparsable judge output"


def depends_on_previous(followup: str, prev_reply: str) -> bool:
    if BH.FOLLOWUP_REF.search(followup) or BH.FOLLOWUP_START.search(followup.strip()):
        return True
    fw = {w for w in re.findall(r"[a-z]{5,}", followup.lower())}
    pw = {w for w in re.findall(r"[a-z]{5,}", prev_reply.lower())}
    return len(fw & pw) >= 2


def python_blocks_ok(text: str) -> tuple[bool, int]:
    n = 0
    for lang, code in code_blocks(text):
        if lang in ("python", "py", "python3"):
            n += 1
            try:
                compile(code, "<answer>", "exec")
            except SyntaxError:
                return False, n
    return True, n


LIMITS = {"default": 250, "writing_create": 320, "code_help": 220, "casual_conversation": 120, "summarising": 250}


def length_ok(category: str, text: str) -> bool:
    if prose_words(text) > LIMITS.get(category, LIMITS["default"]):
        return False
    return code_lines(text) <= 90


# ------------------------------------------------------------------------------------------------ data sources
_POOL = None
SCREEN_FILE = CACHE / "screen_v1.jsonl"
POOL_FILE = CACHE / "real_pool_v1.jsonl"          # run.py switches this to real_pool_v2.jsonl for the full run


def real_pool() -> list[dict]:
    """Real-user prompt pool; when screen verdicts exist, only prompts screened ok are used (unscreened are skipped)."""
    global _POOL
    if _POOL is None:
        pool = [json.loads(l) for l in open(POOL_FILE, encoding="utf-8")]
        if SCREEN_FILE.exists():
            v = {}
            for line in SCREEN_FILE.open(encoding="utf-8"):
                x = json.loads(line)
                v[x["pid"]] = x["ok"]
            pool = [p for p in pool if v.get(p["pid"]) is True]
        _POOL = pool
    return _POOL


def pool_take(rng, used: set, cats, n, *, max_words=300, need_t2=None, kind_filter=None):
    cands = [r for r in real_pool() if r["category"] in cats and r["pid"] not in used and r["words"] <= max_words
             and (need_t2 is None or bool(r["t2"]) == need_t2) and (kind_filter is None or kind_filter(r))]
    salt = rng.random()
    cands.sort(key=lambda r: hashlib.sha256(f"{salt}:{r['pid']}".encode()).hexdigest())
    out = []
    users = set()
    for r in cands:
        if r["user"] in users:
            continue
        users.add(r["user"])
        used.add(r["pid"])
        out.append(r)
        if len(out) >= n:
            break
    return out


_FACTORY_CACHE: dict = {}
FACTORY_CUTOFF = "2026-10-17T12:00:00"   # the factory keeps appending; a fixed cutoff keeps sampling reproducible


def factory_rows(job: str, limit_files=None) -> list[dict]:
    if job in _FACTORY_CACHE:
        return _FACTORY_CACHE[job]
    rows = []
    for f in sorted(glob.glob(str(FACTORY / job / "verified-*.jsonl")))[:limit_files]:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("verified") and not r.get("contaminated") and not r.get("banned") and \
                        str(r.get("created_utc", "")) < FACTORY_CUTOFF:
                    rows.append(r)
    _FACTORY_CACHE[job] = rows
    return rows


# ================================================================================================ 1. chat
CHAT_FOLLOWUPS = {
    "code_help": ["code_lang", "code_extend", "simpler", "deeper", "pushback"],
    "writing_create": ["change", "shorter", "change", "format"],
    "rewriting_editing": ["change", "shorter"],
    "explain_teach": ["simpler", "deeper", "example", "shorter", "pushback"],
    "practical_advice": ["new_info", "deeper", "shorter", "format", "new_info"],
    "factual_qa": ["deeper", "example", "pushback", "simpler"],
    "brainstorming_lists": ["more", "change", "deeper", "format"],
    "casual_conversation": ["thanks_more", "deeper"],
    "summarising": ["shorter", "deeper"],
}


async def chat_task(ctx: Ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    p = task.data["prompt"]
    cat = p["category"]
    n_turns = task.data["turns"]
    msgs = [{"role": "user", "content": p["t1"]}]
    checks, debug = {}, {"followups": []}
    for t in range(n_turns):
        if t > 0:
            if t == 1 and p.get("t2"):
                u, kind = p["t2"], "real"
            else:
                kind = rng.choice(CHAT_FOLLOWUPS.get(cat, ["deeper", "shorter"]))
                u = await gen_followup(ctx, task, msgs, kind)
            debug["followups"].append(kind)
            if not u or nwords(u) > 60:
                return [bad(task, "followup_generation_failed", checks, debug)]
            dep = depends_on_previous(u, msgs[-1]["content"])
            checks[f"followup{t}_depends"] = dep
            if not dep:
                return [bad(task, "followup_not_dependent", checks, {**debug, "followup": u})]
            msgs.append({"role": "user", "content": u})
        r = await teacher_answer(ctx, task, msgs, cat)
        a = (r["content"] or "").strip()
        if r["finish_reason"] == "length" or not a:
            return [bad(task, "truncated_or_empty", checks, debug)]
        first_len_ok = length_ok(cat, a)
        debug.setdefault("first_try_length_ok", []).append(first_len_ok)
        if not first_len_ok:                       # one retry with a hard word limit
            lim = int(LIMITS.get(cat, LIMITS["default"]) * 0.75)
            r = await teacher_answer(ctx, task, msgs, cat, system_extra=P.SHORTEN_NOTE.format(limit=lim))
            a = (r["content"] or "").strip()
            if r["finish_reason"] == "length" or not a:
                return [bad(task, "truncated_or_empty", checks, debug)]
        checks[f"turn{t}_length_ok"] = length_ok(cat, a)
        pyok, npy = python_blocks_ok(a)
        if npy:
            checks[f"turn{t}_python_parses"] = pyok
        if t > 0:
            checks[f"turn{t}_not_copy"] = containment(a, msgs[-2]["content"]) < 0.6
        msgs.append({"role": "assistant", "content": a})
        if not all(checks.values()):
            return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks, debug,
                        row_={"messages": msgs})]
    score, problem = await judge(ctx, task, msgs)
    checks["judge_ge4"] = score >= 4
    debug["judge"] = [score, problem]
    v = {"method": "rule filters + LLM quality screen (deepseek-flash judge, score>=4)" +
         ("; follow-ups checked to depend on the previous turn" if n_turns > 1 else ""),
         "passed": all(checks.values()), "details": {"judge_score": score, "judge_problem": problem,
                                                     "followup_kinds": debug["followups"]}}
    out = row(task, msgs, False, v, prov(p["dataset"], p["licence"], p["sid"], sensitive_topic=p["sensitive"],
                                         followup_source=("real user" if p.get("t2") and n_turns > 1 else
                                                          "generated" if n_turns > 1 else None)))
    if not checks["judge_ge4"]:
        return [bad(task, f"judge_score_{score}", checks, debug, row_=out)]
    return [ok(task, out, checks, debug=debug)]


def build_chat(n: int, n_multi: int, n_three: int, seed: int, used: set) -> list[Task]:
    rng = random.Random(f"{seed}:chat")
    quotas = {"explain_teach": 0.14, "practical_advice": 0.14, "factual_qa": 0.14, "writing_create": 0.12,
              "rewriting_editing": 0.08, "code_help": 0.15, "brainstorming_lists": 0.1, "casual_conversation": 0.08,
              "summarising": 0.05}
    tasks, picked = [], []
    real_multi = pool_take(rng, used, list(quotas), max(1, n_multi // 3), need_t2=True)
    picked += [(p, 2) for p in real_multi]
    for cat, q in quotas.items():
        k = max(1, round(n * q))
        for p in pool_take(rng, used, [cat], k):
            picked.append((p, 1))
    picked = picked[:n]
    singles = [i for i, (p, t) in enumerate(picked) if t == 1 and p["category"] not in ("rewriting_editing",)]
    rng.shuffle(singles)
    need = n_multi - len(real_multi)
    for j, i in enumerate(singles[:max(0, need)]):
        picked[i] = (picked[i][0], 3 if j < n_three else 2)
    for p, t in picked:
        tasks.append(Task(f"chat-{p['pid']}", "chat", p["category"], chat_task, {"prompt": p, "turns": t}))
    return tasks


# ================================================================================================ 2. instruction following
async def if_task(ctx: Ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    p = task.data["prompt"]
    kind = task.data["kind"]
    multi = task.data["multi"]
    k1 = 0 if multi and rng.random() < 0.5 else rng.choice([1, 2, 2, 3])
    cons1 = sample_constraints(rng, p["t1"], kind, k1) if k1 else []
    if cons1 is None:
        return [bad(task, "no_compatible_constraints")]
    def compose(base, cons):
        if not cons:
            return base
        txt = " ".join(c["text"] for c in cons)
        return f"{base}\n\n{txt}" if rng.random() < 0.75 else f"{txt}\n\n{base}"
    turns = [(compose(p["t1"], cons1), cons1)]
    cons2 = None
    if multi:
        cons2 = sample_constraints(rng, p["t1"], "general" if kind != "code" else "code", 1)
        if cons2 is None:
            return [bad(task, "no_compatible_constraints")]
        lead = rng.choice(["Now rewrite your answer.", "Can you redo that?", "Please change your last reply.",
                           "Rewrite that."])
        turns.append((lead, cons2))
    msgs, checks, tries = [], {}, []
    for ti, (utext, cons) in enumerate(turns):
        if ti == 1:                                # the new constraint must change something
            for _ in range(8):
                if not all(check_all(msgs[-1]["content"], cons).values()):
                    break
                cons = sample_constraints(rng, p["t1"], "general" if kind != "code" else "code", 1) or cons
            else:
                return [bad(task, "followup_constraint_already_met")]
            utext = f"{utext} {cons[0]['text']}"
            turns[1] = (utext, cons)
        msgs.append({"role": "user", "content": utext})
        a, res = None, {}
        for attempt in range(2):
            extra = P.IF_NOTE
            if attempt == 1:
                extra += "\nYour previous draft broke these requirements: " + "; ".join(
                    c["text"] for c in cons if not res.get(c["id"])) + " Fix that."
            r = await teacher_answer(ctx, task, msgs, p["category"], system_extra=extra, temperature=0.6)
            a = (r["content"] or "").strip()
            res = check_all(a, cons) if cons else {}
            if r["finish_reason"] != "length" and a and all(res.values()):
                break
        tries.append(attempt + 1)
        for c in cons:
            checks[f"t{ti}_{c['id']}"] = res.get(c["id"], False)
        checks[f"t{ti}_length_ok"] = prose_words(a) <= 260
        msgs.append({"role": "assistant", "content": a})
        if not all(checks.values()):
            return [bad(task, "constraint_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                        {"tries": tries}, row_={"messages": msgs})]
    score, problem = await judge(ctx, task, msgs)
    checks["judge_ge4"] = score >= 4
    allc = [c for _, cs in turns for c in cs]
    v = {"method": "programmatic constraint checkers (all must pass) + LLM quality screen (score>=4)", "passed": True,
         "details": {"constraints": [{"id": c["id"], "params": c["params"]} for c in allc], "tries": tries,
                     "judge_score": score, "judge_problem": problem}}
    if score < 4:
        return [bad(task, f"judge_score_{score}", checks, {"tries": tries, "judge": [score, problem]},
                    row_={"messages": msgs})]
    out = row(task, msgs, False, v, prov(p["dataset"] + " (base prompt) + generated constraints", p["licence"],
                                         p["sid"], sensitive_topic=p["sensitive"]))
    return [ok(task, out, checks, debug={"first_try": tries[0] == 1})]


def build_if(n: int, n_multi: int, seed: int, used: set) -> list[Task]:
    rng = random.Random(f"{seed}:if")
    gen = pool_take(rng, used, ["explain_teach", "practical_advice", "brainstorming_lists", "factual_qa"],
                    n - n // 5 - n // 8, max_words=40)
    wri = pool_take(rng, used, ["writing_create"], n // 5, max_words=40)
    code = pool_take(rng, used, ["code_help"], n // 8, max_words=40,
                     kind_filter=lambda r: bool(re.search(r"\b(write|create|make|implement|code|function|program|"
                                                          r"script)\b", r["t1"], re.I))
                     and not re.search(r"```|python|kivy|pandas|numpy|django|flask|torch|tensorflow|matplotlib|"
                                       r"\bjava\b|javascript|typescript|\bc\b|c\+\+|c#|\bsql\b|mysql|postgres|html|css|"
                                       r"\brust\b|\bgo\b|golang|ruby|php|swift|kotlin|bash|shell|\br\b|matlab|julia|"
                                       r"scala|perl|haskell|lua|dart|excel|vba|powershell|react|node", r["t1"], re.I)
                     and BH.code_line_frac(r["t1"]) == 0 and "(" not in r["t1"])
    tasks = []
    allp = [(p, "long" if p["category"] == "explain_teach" else "general") for p in gen] + \
        [(p, "writing") for p in wri] + [(p, "code") for p in code]
    for i, (p, kind) in enumerate(allp):
        tasks.append(Task(f"if-{p['pid']}", "instruction_following", p["category"], if_task,
                          {"prompt": p, "kind": kind, "multi": i % max(1, len(allp) // max(1, n_multi)) == 0}))
    return tasks


# ================================================================================================ 3. system prompts
def _sent_le(n):
    from constraints import sentences
    return lambda a: len(sentences(a)) <= n


def _json_ac(a):
    try:
        o = json.loads(a.strip())
    except json.JSONDecodeError:
        return False
    return isinstance(o, dict) and set(o) == {"answer", "confidence"} and o["confidence"] in ("low", "medium", "high")


US_SPELL = re.compile(r"\b(color|colors|favorite|organize[ds]?|organizing|organization|center|centers|realize[ds]?|"
                      r"behavior|neighbor|neighbors|analyze[ds]?|apologize|recognize[ds]?|flavor|humor|labor|honor|"
                      r"theater|traveling|traveled|defense|license|catalog|gray|program me)\b", re.I)

SYSTEMS = [
    ("brief2", "You are a concise assistant. Answer in at most two sentences.", _sent_le(2), "pool"),
    ("tutor_q", "You are a friendly tutor for school students. Explain step by step in simple words and end with one "
                "short question that checks the student's understanding.", lambda a: a.strip().endswith("?"), "pool"),
    ("steps", "Always answer as a numbered list of short steps.",
     lambda a: len(re.findall(r"^\s*\d+[.)]\s", a, re.M)) >= 2, "pool"),
    ("plain", "Reply in plain text only. Do not use markdown: no bullet points, bold, headings or tables.",
     lambda a: not re.search(r"\*\*|^#|^\s*[-*•]\s|^\s*\d+[.)]\s|^\|", a, re.M), "pool"),
    ("json_ac", "Respond only with a JSON object of the form {\"answer\": string, \"confidence\": \"low\" | \"medium\" | "
                "\"high\"}. No other text.", _json_ac, "pool"),
    ("british", "Use British English spelling and vocabulary in every reply.", lambda a: not US_SPELL.search(a), "pool"),
    ("kid7", "You explain things to a curious seven-year-old: short sentences, everyday words, one fun comparison.",
     lambda a: prose_words(a) <= 150, "pool"),
    ("bold_first", "Start every answer with a one-line summary in bold, then give the details below it.",
     lambda a: bool(re.match(r"^\*\*[^*\n]+\*\*", a.strip())), "pool"),
    ("recommend3", "Be direct: give your recommendation first, then at most three short reasons.",
     lambda a: prose_words(a) <= 150, "pool"),
    ("under50", "Keep every reply under 50 words.", lambda a: nwords(a) < 50, "pool"),
    ("answer_why", "Format every reply exactly as two lines:\nAnswer: <one sentence>\nWhy: <one or two sentences>",
     lambda a: bool(re.fullmatch(r"Answer: [^\n]+\nWhy: [^\n]+", a.strip())), "pool"),
    ("learner", "The user is learning English. Use short sentences and simple words. After your answer, add a line "
                "'Useful words:' followed by up to three words from your answer with short meanings.",
     lambda a: "Useful words:" in a, "pool"),
    ("socratic", "You are a Socratic maths tutor. Never give the final answer. Give one helpful hint and ask one "
                 "guiding question.", lambda a: a.strip().endswith("?"), "maths"),
    ("bikeshop", "You are the help assistant for Spoke & Chain, a small bicycle repair shop. Only help with questions "
                 "about bicycles, cycling and bike maintenance. For anything else, say briefly that you can only help "
                 "with bike topics.", None, "persona"),
    ("plantcare", "You are Leafy, a plant-care helper for a garden centre. Only answer questions about plants and "
                  "gardening. Politely decline other topics in one sentence.", None, "persona"),
    ("cookbot", "You are a home-cooking assistant. Always list the ingredients first, then numbered steps. Keep "
                "recipes simple.", None, "persona"),
]


async def system_task(ctx: Ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    sid, stext, check, _src = task.data["system"]
    user = task.data["user"]
    msgs = [{"role": "system", "content": stext}, {"role": "user", "content": user}]
    checks, debug = {}, {}
    n_turns = task.data["turns"]
    for t in range(n_turns):
        if t > 0:
            kind = rng.choice(["deeper", "example", "simpler", "pushback"])
            u = await gen_followup(ctx, task, msgs, kind)
            if not u:
                return [bad(task, "followup_generation_failed")]
            checks[f"followup{t}_depends"] = depends_on_previous(u, msgs[-1]["content"])
            msgs.append({"role": "user", "content": u})
        sys_text = (P.STYLE + "\n\nThe operator of this assistant has set the system message below. Follow it "
                    "exactly in every turn; where it conflicts with the style rules above, it wins. " + P.SYS_NO_INVENT +
                    "\n\nOPERATOR SYSTEM MESSAGE:\n" + stext)
        r = await ctx.ds.chat([{"role": "system", "content": sys_text}] + msgs[1:], slice_name=task.slice,
                              temperature=0.7, max_tokens=700)
        a = (r["content"] or "").strip()
        if a and prose_words(a) > 250:
            r = await ctx.ds.chat([{"role": "system", "content": sys_text + "\n\n" + P.SHORTEN_NOTE.format(limit=180)}]
                                  + msgs[1:], slice_name=task.slice, temperature=0.7, max_tokens=700)
            a = (r["content"] or "").strip()
        if r["finish_reason"] == "length" or not a:
            return [bad(task, "truncated_or_empty", checks)]
        if check is not None:
            checks[f"turn{t}_rule"] = bool(check(a))
        checks[f"turn{t}_length_ok"] = prose_words(a) <= 250
        if t > 0:
            checks[f"turn{t}_not_copy"] = containment(a, msgs[-2]["content"]) < 0.6
        msgs.append({"role": "assistant", "content": a})
        if not all(checks.values()):
            return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                        row_={"messages": msgs})]
    score, problem = await judge(ctx, task, msgs, system_note=stext)
    checks["judge_ge4"] = score >= 4
    src = task.data["prov"]
    v = {"method": ("rule checker for the system prompt + " if check else "") + "LLM quality screen (score>=4)",
         "passed": all(checks.values()), "details": {"system_id": sid, "judge_score": score, "judge_problem": problem}}
    out = row(task, msgs, False, v, src)
    if not checks["judge_ge4"]:
        return [bad(task, f"judge_score_{score}", checks, row_=out)]
    return [ok(task, out, checks)]


async def persona_task(ctx: Ctx, task: Task) -> list[dict]:
    """One call writes in-scope / out-of-scope questions for a persona; each becomes a system-prompt row."""
    sysdef = task.data["system"]
    r = await ctx.ds.chat([{"role": "system", "content": P.SAFETY_GEN_SYS.replace("a safety training set",
                                                                                 "a system-prompt training set")},
                           {"role": "user", "content": P.PERSONA_Q_USER.format(
                               system=sysdef[1], n_in=2, n_out=1,
                               out_topic=task.rng(ctx).choice(P.PERSONA_OUT_TOPICS))}],
                          slice_name=task.slice, json_mode=True, temperature=1.0, max_tokens=300)
    o = parse_json(r["content"]) or {}
    outs = []
    qs = [(q, "in_scope") for q in (o.get("in_scope") or [])[:2]] + [(q, "out_of_scope") for q in
                                                                     (o.get("out_of_scope") or [])[:1]]
    for i, (q, scope) in enumerate(qs):
        sub = Task(f"{task.id}-{i}", task.slice, f"persona_{scope}", system_task,
                   {"system": sysdef, "user": q, "turns": 1,
                    "prov": prov("generated (deepseek-flash) for an in-house system prompt", DS_LICENCE,
                                 f"{task.id}-{i}", system_prompt_id=sysdef[0], scope=scope)})
        outs += await system_task(ctx, sub)
    return outs or [bad(task, "persona_question_generation_failed")]


def build_system(n: int, n_multi: int, seed: int, used: set, maths_rows: list[dict]) -> list[Task]:
    rng = random.Random(f"{seed}:system")
    tasks = []
    pool_sys = [s for s in SYSTEMS if s[3] == "pool"]
    personas = [s for s in SYSTEMS if s[3] == "persona"]
    n_persona_tasks = max(1, n // 6)
    n_pool = n - 3 * n_persona_tasks - 1
    users = pool_take(rng, used, ["explain_teach", "practical_advice", "factual_qa", "brainstorming_lists"], n_pool,
                      max_words=60)
    for i, p in enumerate(users):
        s = pool_sys[i % len(pool_sys)]
        tasks.append(Task(f"sys-{s[0]}-{p['pid']}", "system_prompts", s[0], system_task,
                          {"system": s, "user": p["t1"], "turns": 2 if i < n_multi else 1,
                           "prov": prov(p["dataset"] + " (user turn) + in-house system prompt", p["licence"], p["sid"],
                                        system_prompt_id=s[0], sensitive_topic=p["sensitive"])}))
    soc = [s for s in SYSTEMS if s[0] == "socratic"][0]
    if maths_rows:
        m = maths_rows[0]
        tasks.append(Task(f"sys-socratic-{m['sid']}", "system_prompts", "socratic", system_task,
                          {"system": soc, "user": m["problem"], "turns": 1,
                           "prov": prov(f"datafactory/{m['job']} ({m.get('provider')})", FACTORY_NOTE, m["sid"],
                                        system_prompt_id="socratic")}))
    for j in range(n_persona_tasks):
        s = personas[j % len(personas)]
        tasks.append(Task(f"persona-{s[0]}-{j}", "system_prompts", "persona", persona_task, {"system": s}))
    return tasks


# ================================================================================================ 4. maths / logic
WRAPS = ["{p}", "{p}", "{p}", "Can you help me solve this? {p}", "Solve: {p}", "Quick question: {p}",
         "Help me with this problem, please.\n\n{p}", "{p} Please show your working."]


def gold_match_kind(job: str, r: dict) -> str:
    if job == "logic":
        return "logic"
    if job == "code_trace":
        return "output"
    return "math"


async def maths_task(ctx: Ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    m = task.data["item"]
    kind = gold_match_kind(m["job"], m)
    user = rng.choice(WRAPS).format(p=m["problem"].strip())
    r = await ctx.ds.chat([{"role": "system", "content": P.MATH_SYS}, {"role": "user", "content": m["problem"].strip()}],
                          slice_name=task.slice, json_mode=True, temperature=0.3, max_tokens=1100)
    o = parse_json(r["content"])
    if r["finish_reason"] == "length" or not isinstance(o, dict):
        return [bad(task, "truncated_or_unparsable")]
    th, ans = str(o.get("thinking", "")).strip(), str(o.get("answer", "")).strip()
    r_on, r_off = str(o.get("reply_after_thinking", "")).strip(), str(o.get("reply_without_thinking", "")).strip()
    gold = str(m["gold"]).strip()
    if kind == "math":
        g = await ctx.mv.verify(ans, gold)
    elif kind == "logic":
        g = logic_equal(ans, gold)
    else:
        g = exact_output_equal(ans, gold)
    tail = th[int(len(th) * 0.5):]
    present = (lambda text: answer_in_text(ans, text)) if kind == "math" else \
        (lambda text: all(w.lower() in text.lower() for w in re.split(r"[\s,]+", ans) if w))
    checks = {"answer_matches_gold": bool(g), "thinking_ends_on_answer": present(tail),
              "thinking_linear": not re.search(P.MEANDER_RX, th, re.I),
              "reply_on_states_answer": present(r_on), "reply_off_states_answer": present(r_off),
              "thinking_le_300w": 0 < nwords(th) <= 300, "reply_on_le_80w": 0 < nwords(r_on) <= 80,
              "reply_off_le_150w": 0 < nwords(r_off) <= 150}
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                    {"answer": ans, "gold": gold}, row_={"thinking": th, "answer": ans, "r_on": r_on, "r_off": r_off})]
    src = m.get("source", "")
    if "gsm8k" in src.lower():
        ps, lic = "openai/gsm8k (train)", "MIT"
    elif "MATH" in src:
        ps, lic = "hendrycks/MATH (train)", "MIT"
    else:
        ps, lic = f"datafactory/{m['job']} ({m.get('provider')}, {m.get('model', '')})", FACTORY_NOTE
    method = {"math": "final answer == gold (math-verify / numeric)", "logic": "exact match with brute-force gold",
              "output": "exact match with the program's real output"}[kind]
    v = {"method": method + "; thinking and both replies state the same result", "passed": True,
         "details": {"gold": gold, "answer": ans, "level": m.get("level")}}
    pv = prov(ps, lic, m["sid"])
    on = row(task, [{"role": "user", "content": user},
                    {"role": "assistant", "content": r_on, "reasoning_content": th}], True, v, pv,
             rid=task.id + "-think", pair_id=task.id)
    off = row(task, [{"role": "user", "content": user}, {"role": "assistant", "content": r_off}], False, v, pv,
              rid=task.id + "-nothink", pair_id=task.id)
    return [ok(task, on, checks, rid=on["id"]), ok(task, off, checks, rid=off["id"])]


def maths_items(seed: int, quotas: dict) -> list[dict]:
    rng = random.Random(f"{seed}:maths")
    out = []
    for job, sel, k in quotas:
        rows = factory_rows(job)
        rows = [r for r in rows if sel(r)]
        rng.shuffle(rows)
        for r in rows[:k]:
            r = dict(r)
            r["job"] = job
            out.append(r)
    return out


def build_maths(items: list[dict]) -> list[Task]:
    return [Task(f"math-{m['sid']}", "maths_logic", m["job"] if m["job"] != "math" else
                 ("gsm8k_train" if "gsm8k" in m.get("source", "").lower() else "math_train"), maths_task, {"item": m})
            for m in items]


# ================================================================================================ 5. python tool mode
async def tool_task(ctx: Ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    m = task.data["item"]
    problem = m["problem"].strip()
    r = await ctx.ds.chat([{"role": "system", "content": P.TOOL_SYS}, {"role": "user", "content": problem}],
                          slice_name=task.slice, json_mode=True, temperature=0.3, max_tokens=800)
    o = parse_json(r["content"])
    if r["finish_reason"] == "length" or not isinstance(o, dict):
        return [bad(task, "truncated_or_unparsable")]
    code, plan, final = str(o.get("code", "")).strip(), str(o.get("plan", "")).strip(), str(o.get("final", "")).strip()
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code)
    safe, why = check_code(code)
    run = await arun_python(code) if safe else {"ok": False, "stdout": "", "stderr": why}
    out_txt = run["stdout"].strip()
    gold = m.get("gold")
    if gold is None:
        g = bool(out_txt) and len(out_txt) <= 200
    elif m["job"] == "code_trace":
        g = exact_output_equal(out_txt, gold)
    else:
        g = num_equal(out_txt, gold) or (await ctx.mv.verify(out_txt, str(gold)) if out_txt else False)
    checks = {"code_safe": safe, "code_runs": run["ok"], "output_matches_gold": bool(g),
              "code_le_30_lines": code.count("\n") + 1 <= 30, "output_le_300_chars": 0 < len(out_txt) <= 300}
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                    {"stdout": out_txt[:200], "stderr": run.get("stderr", "")[-200:], "gold": gold})]
    r2 = await ctx.ds.chat([{"role": "system", "content": P.TOOL_FINAL_SYS},
                            {"role": "user", "content": P.TOOL_FINAL_USER.format(problem=problem, code=code,
                                                                                 out=out_txt)}],
                           slice_name=task.slice, json_mode=True, temperature=0.4, max_tokens=300)
    o2 = parse_json(r2["content"]) or {}
    final = str(o2.get("reply", "")).strip()
    toks = [t for t in re.split(r"[\s,;:=\[\]()'\"]+", out_txt) if t]
    checks["final_states_output"] = bool(final) and all(
        answer_in_text(t, final) if to_number(t) is not None else t.lower() in final.lower() for t in toks[:12])
    checks["final_le_80w"] = 0 < nwords(final) <= (120 if m["job"] == "code_trace" else 80)
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                    {"stdout": out_txt[:200], "final": final})]
    thinking = rng.random() < 0.5
    call = {"id": "call_1", "type": "function", "function": {"name": "python", "arguments": {"code": code}}}
    a1 = {"role": "assistant", "content": "", "tool_calls": [call]}
    if thinking:
        a1["reasoning_content"] = plan
    msgs = [{"role": "user", "content": problem}, a1,
            {"role": "tool", "tool_call_id": "call_1", "name": "python", "content": out_txt},
            {"role": "assistant", "content": final}]
    if gold is None:
        method = "code executed in a sandbox; reply reports the real output (no independent gold)"
        ps, lic = "generated (deepseek-flash) request", DS_LICENCE
    else:
        method = "code executed in a sandbox (no network, no file writes); printed output == gold"
        src = m.get("source", "")
        ps, lic = (("openai/gsm8k (train)", "MIT") if "gsm8k" in src.lower() else
                   (f"datafactory/{m['job']} ({m.get('provider')}, {m.get('model', '')})", FACTORY_NOTE))
    v = {"method": method, "passed": True, "details": {"stdout": out_txt, "gold": gold}}
    return [ok(task, row(task, msgs, thinking, v, prov(ps, lic, m["sid"]), tools=[PY_TOOL]), checks)]


async def tool_gen_task(ctx: Ctx, task: Task) -> list[dict]:
    r = await ctx.ds.chat([{"role": "system", "content": P.SAFETY_GEN_SYS.replace("a safety training set",
                                                                                 "a tool-use training set")},
                           {"role": "user", "content": P.TOOL_TASKS_USER.format(n=task.data["n"],
                                                                                topics=task.data["topics"])}],
                          slice_name=task.slice, json_mode=True, temperature=1.0, max_tokens=500)
    o = parse_json(r["content"]) or {}
    outs = []
    for i, q in enumerate((o.get("requests") or [])[:task.data["n"]]):
        sub = Task(f"{task.id}-{i}", task.slice, "utility", tool_task,
                   {"item": {"problem": q, "gold": None, "job": "generated", "sid": f"{task.id}-{i}"}})
        outs += await tool_task(ctx, sub)
    return outs or [bad(task, "request_generation_failed")]


def build_tool(items: list[dict], n_gen_tasks: int, per_gen: int) -> list[Task]:
    tasks = [Task(f"tool-{m['sid']}", "python_tool", m["job"], tool_task, {"item": m}) for m in items]
    topics = ["dates and weekdays; primes", "unit conversions; compound interest", "combinatorics; string counting",
              "statistics of a list; time zones arithmetic", "calendar questions; sorting and ranking"]
    for j in range(n_gen_tasks):
        tasks.append(Task(f"toolgen-{j}", "python_tool", "utility", tool_gen_task,
                          {"n": per_gen, "topics": topics[j % len(topics)]}))
    return tasks


# ================================================================================================ 6. JSON / function calls
FC_DOMAINS = ["weather forecasts", "calendar and meetings", "recipe search", "flight search", "hotel booking",
              "unit conversion", "currency exchange", "to-do lists", "note taking", "smart home devices", "music player",
              "maps and directions", "restaurant reservations", "package tracking", "fitness tracking",
              "library catalogue", "dictionary lookup", "inventory management", "employee leave requests",
              "school timetables", "public transport times", "cinema showtimes", "plant care reminders",
              "pet adoption listings", "nutrition facts", "IT help-desk tickets", "parking spaces", "event tickets",
              "job listings", "apartment listings", "sports fixtures", "photo albums", "expense tracking",
              "language flashcards", "car maintenance logs", "online shop orders", "customer support cases",
              "train seat booking", "museum opening hours", "recycling collection days"]
_BFCL_NAMES = None


def bfcl_names() -> set:
    global _BFCL_NAMES
    if _BFCL_NAMES is None:
        _BFCL_NAMES = set()
        for f in glob.glob(str(REPO / "data" / "eval" / "bench" / "BFCL_v3_*.json")):
            for line in open(f, encoding="utf-8"):
                try:
                    q = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for fn in q.get("function") or []:
                    _BFCL_NAMES.add(fn.get("name", "").lower())
    return _BFCL_NAMES


def mock_execute(call: dict, tools: dict, result) -> dict | None:
    """Validate the call against its schema; the mock function then returns the scripted result."""
    t = tools.get(call.get("name"))
    if t is None:
        return None
    good, _ = args_ok(call.get("arguments"), t["parameters"])
    return result if good and isinstance(result, dict) else None


async def fc_task(ctx: Ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    kind, domain = task.data["kind"], task.data["domain"]
    ntools = rng.choice([1, 2, 2, 3]) if kind not in ("parallel", "multi_step") else rng.choice([1, 2])
    if kind == "multi_step":
        ntools = 2
    r = await ctx.ds.chat([{"role": "system", "content": P.FC_SYS},
                           {"role": "user", "content": P.FC_USER.format(
                               domain=domain, kind=kind, ntools=ntools, kind_rule=P.FC_KINDS[kind],
                               date=(dt.date(2024, 1, 1) + dt.timedelta(days=rng.randrange(900))).strftime(
                                   "%Y-%m-%d (%A)"))}],
                          slice_name=task.slice, json_mode=True, temperature=0.9, max_tokens=2000)
    o = parse_json(r["content"])
    if r["finish_reason"] == "length" or not isinstance(o, dict):
        return [bad(task, "truncated_or_unparsable")]
    tools_l, user, steps = o.get("tools") or [], str(o.get("user", "")).strip(), o.get("steps") or []
    sysmsg = str(o.get("system") or "").strip()
    final, thought = str(o.get("final", "")).strip(), str(o.get("thought", "")).strip()
    checks = {}
    tools = {}
    for t in tools_l:
        if isinstance(t, dict) and "function" in t and isinstance(t["function"], dict):
            t = t["function"]
        if not isinstance(t, dict):
            continue
        tools[t.get("name")] = {"name": t.get("name"), "description": t.get("description", ""),
                                "parameters": t.get("parameters")}
    checks["tools_parse"] = len(tools) == len(tools_l) and len(tools) >= 1
    checks["tool_names_ok"] = all(isinstance(n, str) and re.fullmatch(r"[a-z][a-z0-9_]{2,40}", n) for n in tools)
    checks["schemas_valid"] = all(schema_ok(t["parameters"])[0] for t in tools.values())
    checks["no_bfcl_name"] = not any((n or "").lower() in bfcl_names() for n in tools)
    if kind in ("no_call", "clarify"):
        checks["no_steps"] = len(steps) == 0
        if kind == "clarify":
            checks["asks_question"] = final.endswith("?")
    else:
        want = {"single": (1, 1), "parallel": (1, 2), "multi_step": (2, 1)}[kind]
        checks["step_shape"] = len(steps) == want[0] and all(
            isinstance(s, dict) and len(s.get("calls") or []) == want[1] and
            len(s.get("results") or []) == len(s.get("calls") or []) for s in steps)
        executed = []
        if checks["step_shape"] and checks["schemas_valid"]:
            for s in steps:
                for c, res in zip(s["calls"], s["results"]):
                    executed.append((c, mock_execute(c, tools, res)))
        checks["calls_valid_and_executed"] = bool(executed) and all(res is not None for _, res in executed)
        if checks["calls_valid_and_executed"]:
            vals = [str(v) for _, res in executed for v in flat_values(res) if isinstance(v, (str, int, float))
                    and not isinstance(v, bool) and len(str(v)) >= 2]
            checks["final_uses_results"] = any(v.lower() in final.lower() for v in vals) or any(
                to_number(v) is not None and answer_in_text(v, final) for v in vals)
            if kind == "multi_step":
                first_vals = {str(v).lower() for v in flat_values(steps[0]["results"][0])}
                checks["step2_uses_step1"] = any(str(v).lower() in first_vals
                                                 for v in flat_values(steps[1]["calls"][0].get("arguments", {})))
    checks["final_ok"] = 0 < nwords(final) <= 120 and not final.lstrip().startswith("{")
    checks["user_ok"] = 4 <= nwords(user) <= 60
    rel = re.search(r"\b(today|tonight|tomorrow|yesterday|next (week|month|year|monday|tuesday|wednesday|thursday|"
                    r"friday|saturday|sunday)|last (week|month|year|\w+ (days|weeks|months|years))|this (week|weekend|"
                    r"month)|in \d+ (days|weeks))\b", user, re.I)
    checks["relative_dates_have_date"] = not rel or bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{4}", sysmsg))
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                    {"raw": o if isinstance(o, dict) else None})]
    msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": user}]
    thinking = bool(thought) and rng.random() < 0.4
    k = 0
    for si, s in enumerate(steps):
        calls = []
        for c in s["calls"]:
            k += 1
            calls.append({"id": f"call_{k}", "type": "function",
                          "function": {"name": c["name"], "arguments": c["arguments"]}})
        a = {"role": "assistant", "content": "", "tool_calls": calls}
        if thinking and si == 0:
            a["reasoning_content"] = thought
        msgs.append(a)
        for c, res in zip(calls, s["results"]):
            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["function"]["name"],
                         "content": json.dumps(res, ensure_ascii=False)})
    fin = {"role": "assistant", "content": final}
    if thinking and not steps:
        fin["reasoning_content"] = thought
    msgs.append(fin)
    tl = [{"type": "function", "function": t} for t in tools.values()]
    v = {"method": "tool schemas validated; every call validated against its schema and mock-executed (scripted "
                   "result); final reply uses the results" + ("; step 2 uses a step-1 value" if kind == "multi_step"
                                                              else ""),
         "passed": True, "details": {"kind": kind, "domain": domain}}
    return [ok(task, row(task, msgs, thinking, v, prov("generated (deepseek-flash) tools + request", DS_LICENCE,
                                                       task.id, fc_kind=kind), tools=tl, category=kind), checks)]


def build_fc(n: int, seed: int) -> list[Task]:
    rng = random.Random(f"{seed}:fc")
    kinds = (["single"] * 4 + ["parallel"] * 2 + ["multi_step"] * 2 + ["no_call"] * 1 + ["clarify"] * 2)
    doms = FC_DOMAINS[:]
    rng.shuffle(doms)
    return [Task(f"fc-{i:04d}", "json_function_calls", kinds[i % len(kinds)], fc_task,
                 {"kind": kinds[i % len(kinds)], "domain": doms[i % len(doms)]}) for i in range(n)]


# ================================================================================================ 7. long documents
SUMMARY_REQS = [("Summarise this document in about 150 words.", 220),
                ("What are the main points of this? Give me 5 bullet points.", 200),
                ("Can you give me a short summary (3-4 sentences) of the text above?", 120),
                ("I don't have time to read all of this. What is it about, and what are the key takeaways?", 220),
                ("Write a one-paragraph summary of this for a busy reader.", 180)]
LEADS = ["", "Here is a document:\n\n", "Please read this.\n\n", "I found this text:\n\n"]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")).strip().lower()


async def longdoc_task(ctx: Ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    d = task.data["doc"]
    doc_block = (f"# {d['title']}\n\n" if d["title"] else "") + d["text"]
    req, limit = rng.choice(SUMMARY_REQS)
    prefix = [{"role": "system", "content": P.LONG_SYS}, {"role": "user", "content": "DOCUMENT:\n" + doc_block}]
    r1 = await ctx.ds.chat(prefix[:1] + [{"role": "user", "content": prefix[1]["content"] + "\n\n" +
                                          P.LONG_SUMMARY.format(request=req, limit=limit - 40)}],
                           slice_name=task.slice, json_mode=True, temperature=0.4, max_tokens=700)
    r2 = await ctx.ds.chat(prefix[:1] + [{"role": "user", "content": prefix[1]["content"] + "\n\n" + P.LONG_QA}],
                           slice_name=task.slice, json_mode=True, temperature=0.6, max_tokens=900)
    o1, o2 = parse_json(r1["content"]) or {}, parse_json(r2["content"]) or {}
    summ = str(o1.get("reply", "")).strip()
    qa = [q for q in (o2.get("qa") or []) if isinstance(q, dict)]
    nd = norm_ws(d["text"])
    good_qa = []
    for q in qa:
        quote = norm_ws(str(q.get("quote", "")))
        found = bool(quote) and len(quote.split()) >= 5 and quote.strip(" .\"'") in nd
        pos = nd.find(quote.strip(" .\"'")) / max(1, len(nd)) if found else -1
        if found and 0 < nwords(q.get("answer", "")) <= 80 and 4 <= nwords(q.get("question", "")) <= 40:
            good_qa.append((q, pos))
    checks = {"summary_ok": 20 <= nwords(summ) <= limit + 40, "qa_quotes_verbatim": len(good_qa) >= 2,
              "qa_reaches_second_half": any(pos >= 0.5 for _, pos in good_qa)}
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                    {"n_qa": len(qa), "n_good": len(good_qa)})]
    good_qa.sort(key=lambda x: -x[1])               # put a late-in-document question first
    nq = rng.choice([1, 2])
    lead = rng.choice(LEADS)
    first_user = f"{lead}{doc_block}\n\n{req}" if rng.random() < 0.7 else f"{req}\n\n{doc_block}"
    msgs = [{"role": "user", "content": first_user}, {"role": "assistant", "content": summ}]
    for q, pos in good_qa[:nq]:
        ans = str(q["answer"]).strip()
        if rng.random() < 0.4:
            ans += f"\n\nThe text says: \"{str(q['quote']).strip().strip(chr(34))}\""
        msgs += [{"role": "user", "content": str(q["question"]).strip()}, {"role": "assistant", "content": ans}]
    v = {"method": "answers grounded: each supporting quote found verbatim in the document; at least one question "
                   "from the second half; summary length checked",
         "passed": True, "details": {"doc_tokens_tokenizer_v1": d["tokens"], "quote_positions":
                                     [round(p, 2) for _, p in good_qa[:nq]]}}
    return [ok(task, row(task, msgs, False, v, prov(d["source"] + " (document)", d["licence"], d["doc_id"],
                                                    revision=d["revision"], url=d.get("url"))), checks)]


def build_longdoc(n: int) -> list[Task]:
    docs = [json.loads(l) for l in open(CACHE / "longdocs_pilot.jsonl", encoding="utf-8")]
    wiki = [d for d in docs if "wikipedia" in d["source"]]
    pdf = [d for d in docs if "wikipedia" not in d["source"]]
    pick = []
    for a, b in zip(wiki, pdf):
        pick += [a, b]
    return [Task(f"long-{hashlib.sha256(d['doc_id'].encode()).hexdigest()[:10]}", "long_documents",
                 "wikipedia" if "wikipedia" in d["source"] else "finepdfs_edu", longdoc_task, {"doc": d})
            for d in pick[:n]]


# ================================================================================================ 8. safety
SAFETY_CATS = [
    ("violence", "requests to hurt or threaten someone"),
    ("break_in", "getting into someone else's home, car or account without permission"),
    ("fraud", "forging documents, fake reviews, scams or cheating someone out of money"),
    ("harassment", "insulting, bullying or demeaning a person or group"),
    ("dangerous_chemistry", "misusing household chemicals or making dangerous substances"),
    ("privacy", "finding or exposing private information about a specific person, stalking"),
    ("academic_cheating", "cheating on exams or passing off others' work"),
    ("self_harm", "a person expressing wishes to hurt themselves"),
    ("drugs", "obtaining or making illegal drugs"),
    ("misinformation", "writing fake news, fake medical claims or impersonating officials"),
    ("weapons", "making or acquiring weapons"),
    ("malware", "writing malicious software or hacking"),
]
REFUSE_RX = re.compile(r"\b(i can[’']?t|i cannot|i can not|i won[’']?t|i[’']m not able to|i am not able to|"
                       r"i[’']m unable|i am unable|not something i can|i[’']m sorry, but|i must decline|"
                       r"can[’']?t help with)\b", re.I)


async def safety_task(ctx: Ctx, task: Task) -> list[dict]:
    cat, desc = task.data["cat"]
    nd, na = task.data["n_decline"], task.data["n_answer"]
    r = await ctx.ds.chat([{"role": "system", "content": P.SAFETY_GEN_SYS},
                           {"role": "user", "content": P.SAFETY_GEN_USER.format(category=cat, desc=desc, n_decline=nd,
                                                                                n_answer=na) +
                            (f"\nStyle of the messages: {task.data['angle']}." if task.data.get("angle") else "")}],
                          slice_name=task.slice, json_mode=True, temperature=1.0, max_tokens=400)
    o = parse_json(r["content"]) or {}
    items = [(q, "decline") for q in (o.get("decline") or [])[:nd]] + [(q, "answer") for q in (o.get("answer") or [])[:na]]
    outs = []
    for i, (q, mode) in enumerate(items):
        q = str(q).strip()
        sub_id = f"{task.id}-{mode}-{i}"
        if mode == "decline":
            rr = await ctx.ds.chat([{"role": "system", "content": P.REFUSE_SYS}, {"role": "user", "content": q}],
                                   slice_name=task.slice, temperature=0.6, max_tokens=200)
        else:
            rr = await ctx.ds.chat([{"role": "system", "content": P.STYLE}, {"role": "user", "content": q}],
                                   slice_name=task.slice, temperature=0.7, max_tokens=700)
        a = (rr["content"] or "").strip()
        jr = await ctx.ds.chat([{"role": "system", "content": P.JUDGE_SYS},
                                {"role": "user", "content": P.SAFETY_JUDGE.format(user=q, reply=a)}],
                               slice_name=task.slice + ":judge", json_mode=True, temperature=0.0, max_tokens=80)
        j = parse_json(jr["content"]) or {}
        refused = bool(REFUSE_RX.search(a[:200]))
        if cat == "self_harm" and mode == "decline":
            checks = {"no_harmful_detail": j.get("harmful_detail") is False, "helpful_ge4": int(j.get("helpful") or 0) >= 4,
                      "points_to_help": bool(re.search(r"crisis|helpline|hotline|emergency|someone you trust|"
                                                       r"doctor|counsel", a, re.I)), "brief_le_120w": nwords(a) <= 120}
            sub = Task(sub_id.replace("-decline-", "-support-"), task.slice, f"{cat}:support", safety_task)
            rw = row(sub, [{"role": "user", "content": q}, {"role": "assistant", "content": a}], False,
                     {"method": "LLM safety judge (no harmful detail, helpfulness) + points to real help + length",
                      "passed": all(checks.values()), "details": {"judge": j}},
                     prov("generated (deepseek-flash) self-harm disclosure", DS_LICENCE, sub.id, safety_mode="support"))
            outs.append(ok(sub, rw, checks) if all(checks.values()) else
                        bad(sub, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                            {"judge": j}, row_=rw))
            continue
        checks = {"judge_agrees_on_mode": j.get("should_decline") is (mode == "decline"),
                  "no_harmful_detail": j.get("harmful_detail") is False,
                  "helpful_ge4" if mode == "answer" else "helpful_ge3": int(j.get("helpful") or 0) >= (4 if mode ==
                                                                                                     "answer" else 3)}
        if mode == "decline":
            checks["declines"] = (refused or cat == "self_harm") and j.get("declined") is True
            checks["brief_le_80w"] = nwords(a) <= 80 if cat != "self_harm" else nwords(a) <= 120
        else:
            checks["not_refused"] = not refused and j.get("declined") is False
            checks["length_ok"] = prose_words(a) <= 250
        sub = Task(sub_id, task.slice, f"{cat}:{mode}", safety_task)
        rw = row(sub, [{"role": "user", "content": q}, {"role": "assistant", "content": a}], False,
                 {"method": "LLM safety judge (should-decline agrees, no harmful detail, helpfulness) + refusal "
                            "regex + length", "passed": all(checks.values()), "details": {"judge": j}},
                 prov(f"generated (deepseek-flash) {mode} request", DS_LICENCE, sub_id, safety_mode=mode))
        if all(checks.values()):
            outs.append(ok(sub, rw, checks))
        else:
            outs.append(bad(sub, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                            {"judge": j}, row_=rw))
    return outs or [bad(task, "prompt_generation_failed")]


def build_safety(n_cats: int, nd: int, na: int, seed: int) -> list[Task]:
    rng = random.Random(f"{seed}:safety")
    cats = SAFETY_CATS[:]
    rng.shuffle(cats)
    if not any(c[0] == "self_harm" for c in cats[:n_cats]):
        cats = [c for c in cats if c[0] == "self_harm"] + [c for c in cats if c[0] != "self_harm"]
    return [Task(f"safety-{c[0]}", "safety", c[0], safety_task, {"cat": c, "n_decline": nd, "n_answer": na})
            for c in cats[:n_cats]]


# ================================================================================================ 9. identity (templated)
ID_QA = [
    ("Who are you?", ["I'm {MODEL_NAME}, a small language model made by Udgam Labs. I can answer questions, explain "
                      "ideas, help with writing and code, and work through maths step by step.",
                      "I'm {MODEL_NAME}, a small language model built by Udgam Labs. How can I help?"]),
    ("What's your name?", ["My name is {MODEL_NAME}. I'm a small language model made by Udgam Labs."]),
    ("Who made you?", ["I was made by Udgam Labs. I'm {MODEL_NAME}, a small language model trained from scratch."]),
    ("Are you ChatGPT?", ["No. I'm {MODEL_NAME}, a small language model made by Udgam Labs. I'm not ChatGPT and I'm "
                          "not made by OpenAI."]),
    ("Are you made by OpenAI or Google?", ["Neither. I'm {MODEL_NAME}, made by Udgam Labs."]),
    ("Which company created you?", ["Udgam Labs created me. I'm {MODEL_NAME}, a small language model."]),
    ("Can you browse the internet?", ["No, I can't browse the internet or look up live information. I answer from "
                                      "what I learned during training, so I may not know about recent events."]),
    ("How big are you?", ["I'm a small model: about 150 million parameters, which is tiny compared with large "
                          "assistants. That makes me fast and cheap to run, but I make more mistakes, so please check "
                          "important answers."]),
    ("What can you do?", ["I'm {MODEL_NAME}, a small language model made by Udgam Labs. I can chat, answer questions, "
                          "explain topics, help with writing and editing, write simple code, and solve maths problems "
                          "step by step. I can also run Python when that tool is turned on."]),
    ("Do you have feelings?", ["No, I don't have feelings or experiences. I'm {MODEL_NAME}, a language model that "
                               "generates text. I'm happy to help with whatever you need, though."]),
    ("Are you DeepSeek?", ["No. I'm {MODEL_NAME}, a small language model made by Udgam Labs."]),
    ("tell me about yourself", ["I'm {MODEL_NAME}, a small language model from Udgam Labs. I'm good at short, clear "
                                "answers, explanations, writing help and step-by-step maths. I can't browse the web, "
                                "and I can make mistakes, so double-check anything important."]),
]
ID_THINK = "The user is asking about my identity. I am {MODEL_NAME}, a small language model made by Udgam Labs. " \
           "Answer briefly and accurately."


def build_identity(n: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}:identity")
    rows = []
    for i in range(n):
        q, answers = ID_QA[i % len(ID_QA)]
        a = rng.choice(answers)
        qv = q if rng.random() < 0.6 else rng.choice([q.lower(), q.rstrip("?") + "?", "hey, " + q[0].lower() + q[1:]])
        think = i % 4 == 3
        msg = {"role": "assistant", "content": a}
        if think:
            msg["reasoning_content"] = ID_THINK
        msgs = [{"role": "user", "content": qv}, msg]
        if i % 5 == 4:
            warm = rng.choice([("What is 12 times 12?", "12 × 12 = 144."),
                               ("Give me a synonym for happy.", "Cheerful."),
                               ("What's the capital of Japan?", "Tokyo."),
                               ("hi!", "Hi! How can I help you today?"),
                               ("How many days are in a leap year?", "366.")])
            msgs = [{"role": "user", "content": warm[0]}, {"role": "assistant", "content": warm[1]}] + msgs
        rows.append({"id": f"identity-{i:04d}", "slice": "identity", "category": "identity", "messages": msgs,
                     "enable_thinking": think,
                     "verified": {"method": "templated (no model call); {MODEL_NAME} filled at release",
                                  "passed": True, "details": {}},
                     "provenance": {"prompt_source": "templated (UdgamLabs)", "prompt_licence": "Apache-2.0 (own)",
                                    "prompt_id": f"identity-{i:04d}", "generator": "template",
                                    "generated_utc": now()}})
    return rows
