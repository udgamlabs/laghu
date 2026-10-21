"""Full-run (v1) builders and the slices added after the pilot review (coordinator decisions, 2026-10-17):
  * chat: 10K rows, ~30% multi-turn, sensitive-topic prompts capped at 10%, short-passage summarising from Wikipedia
    passages where the real pool runs dry;
  * chat_thinking (new): everyday prompts where reasoning helps; enable_thinking=true; thinking <= ~200 words,
    answer <= ~250 words, the answer must follow the thinking (judge-checked);
  * system prompts: in-house rule prompts (with checkers) + teacher-invented operator system messages by theme;
  * long documents: ~500 docs x 3 rows (summary [+ follow-up question], and two single questions), doc prefix cached;
  * safety: more categories and message styles; instruction following: code-language constraints with screened,
    language-neutral code prompts; function calls: more domains plus a scenario twist.
Pilot builders in slices.py are unchanged so the pilot stays reproducible.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

import prompts as P
import slices as S
from common import BH, CACHE, nwords, prose_words, strip_code
from constraints import sentences
from slices import Task, bad, ok, parse_json, prov, row

REASON_RX = re.compile(r"\b(vs\.?|versus|better|best (way|option|choice)|difference between|compare|comparison|should i|"
                       r"which (one|is|should|would)|pros and cons|plan|planning|schedule|budget|how (many|much|long|far)|"
                       r"estimate|why (does|do|is|are|did)|what if|trade-?offs?|strategy|choose|decide|worth it|"
                       r"recommend|cheaper|faster|safer|more efficient|step by step|figure out|optimi[sz]e|"
                       r"prioriti[sz]e|explain why)\b", re.I)


# ================================================================================================ chat (10K)
def passages(n: int, seed: int) -> list[dict]:
    """Short Wikipedia passages (150-450 words) with a summarising request, from longdocs passage dump."""
    path = CACHE / "wiki_passages_v1.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng = random.Random(f"{seed}:passages")
    rng.shuffle(rows)
    reqs = ["Summarize this in 2-3 sentences:", "Can you give me the main points of this?", "TL;DR this please:",
            "Summarise the following for a 12-year-old:", "What is this passage about? Keep it short.",
            "Give me a one-sentence summary of this text.", "Turn this into 3 short bullet points:",
            "Summarize:", "I need a quick summary of this for my notes:"]
    out = []
    for r in rows[:n]:
        req = rng.choice(reqs)
        t1 = f"{req}\n\n{r['text']}" if rng.random() < 0.75 else f"{r['text']}\n\n{req}"
        out.append({"pid": "wp-" + hashlib.sha256(r["pid"].encode()).hexdigest()[:12], "src": "wikipedia",
                    "dataset": "wikimedia/wikipedia 20231101.en (passage) + templated request",
                    "licence": "CC-BY-SA-4.0 (Wikipedia; attribution: " + r["title"] + ", " + r["url"] + ")",
                    "sid": r["pid"], "user": "wp:" + r["pid"], "category": "summarising", "subpool": "passage",
                    "t1": t1, "t2": None, "t2_strong": False, "sensitive": False, "words": nwords(t1)})
    return out


def build_chat_full(n: int, multi_share: float, three_share: float, seed: int, used: set,
                    sens_cap: float = 0.10, extra_passages: int = 0) -> list[Task]:
    rng = random.Random(f"{seed}:chatfull")
    quotas = {"explain_teach": 0.14, "practical_advice": 0.14, "factual_qa": 0.14, "writing_create": 0.12,
              "rewriting_editing": 0.08, "code_help": 0.15, "brainstorming_lists": 0.10, "casual_conversation": 0.08,
              "summarising": 0.05}
    sens_left = int(n * sens_cap)

    def take(cats, k, **kw):
        nonlocal sens_left
        got = S.pool_take(rng, used, cats, k * 2 + 20, **kw)
        out = []
        for p in got:
            if len(out) >= k:
                used.discard(p["pid"])               # give back the unused ones
                continue
            if p["sensitive"]:
                if sens_left <= 0:
                    used.discard(p["pid"])
                    continue
                sens_left -= 1
            out.append(p)
        return out

    picked = [(p, 2) for p in take(list(quotas), int(n * multi_share / 3), need_t2=True)]
    short = 0
    for cat, q in quotas.items():
        k = round(n * q)
        got = take([cat], k)
        if cat == "summarising" and len(got) < k:
            got += passages(k - len(got), seed)
        short += k - len(got)
        picked += [(p, 1) for p in got]
    if short > 0:                                    # fill from the big categories
        for p in take(["explain_teach", "practical_advice", "factual_qa", "code_help", "writing_create",
                       "brainstorming_lists"], short):
            picked.append((p, 1))
    picked = picked[:n]
    singles = [i for i, (p, t) in enumerate(picked) if t == 1 and p["category"] != "rewriting_editing"]
    rng.shuffle(singles)
    n_multi = int(n * multi_share) - sum(t > 1 for _, t in picked)
    n_three = int(n * three_share)
    for j, i in enumerate(singles[:max(0, n_multi)]):
        picked[i] = (picked[i][0], 3 if j < n_three else 2)
    tasks = [Task(f"chat-{p['pid']}", "chat", p["category"], S.chat_task, {"prompt": p, "turns": t}) for p, t in picked]
    if extra_passages:      # the truncation above cut the summarising passages (last in quota order): add them back
        have = {t.id for t in tasks}
        for p in passages(extra_passages, seed):
            if f"chat-{p['pid']}" not in have:
                tasks.append(Task(f"chat-{p['pid']}", "chat", "summarising", S.chat_task, {"prompt": p, "turns": 1}))
    return tasks


# ================================================================================================ chat_thinking (new)
async def chat_think_task(ctx, task: Task) -> list[dict]:
    p = task.data["prompt"]
    r = await ctx.ds.chat([{"role": "system", "content": P.CHAT_THINK_SYS}, {"role": "user", "content": p["t1"]}],
                          slice_name=task.slice, json_mode=True, temperature=0.7, max_tokens=1100)
    o = parse_json(r["content"])
    if r["finish_reason"] == "length" or not isinstance(o, dict):
        return [bad(task, "truncated_or_unparsable")]
    th, ans = str(o.get("thinking", "")).strip(), str(o.get("answer", "")).strip()
    checks = {"thinking_le_200w": 0 < nwords(th) <= 200, "answer_le_250w": 0 < prose_words(ans) <= 250,
              "thinking_linear": not re.search(P.MEANDER_RX, th, re.I), "code_le_60_lines": S.code_lines(ans) <= 60,
              "thinking_no_meta": not re.search(P.THINK_META_RX, th, re.I)}
    pyok, npy = S.python_blocks_ok(ans)
    if npy:
        checks["python_parses"] = pyok
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                    row_={"thinking": th, "answer": ans})]
    jr = await ctx.ds.chat([{"role": "system", "content": P.JUDGE_SYS},
                            {"role": "user", "content": P.JUDGE_THINK_USER.format(user=p["t1"][:3000], thinking=th,
                                                                                  answer=ans)}],
                           slice_name=task.slice + ":judge", json_mode=True, temperature=0.0, max_tokens=80)
    j = parse_json(jr["content"]) or {}
    try:
        score = int(j.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    checks["judge_ge4"] = score >= 4
    checks["answer_follows_thinking"] = j.get("follows") is True
    checks["thinking_clean"] = j.get("clean") is True
    v = {"method": "length + linear-thinking rules; LLM judge: score>=4, answer follows thinking, thinking clean",
         "passed": all(checks.values()), "details": {"judge": j}}
    out = row(task, [{"role": "user", "content": p["t1"]},
                     {"role": "assistant", "content": ans, "reasoning_content": th}], True, v,
              prov(p["dataset"], p["licence"], p["sid"], sensitive_topic=p["sensitive"]))
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks, {"judge": j},
                    row_=out)]
    return [ok(task, out, checks)]


def build_chat_thinking(n: int, seed: int, sens_cap: float = 0.10) -> list[Task]:
    rng = random.Random(f"{seed}:chatthink")
    used: set = set()                                 # its own set: some prompts also appear thinking-off in chat
    cats = ["practical_advice", "explain_teach", "factual_qa", "brainstorming_lists", "code_help", "writing_create"]
    cands = S.pool_take(rng, used, cats, n * 3, max_words=200,
                        kind_filter=lambda r: bool(REASON_RX.search(r["t1"])))
    out, sens = [], 0
    for p in cands:
        if p["sensitive"]:
            if sens >= n * sens_cap:
                continue
            sens += 1
        if p["category"] == "code_help" and sum(q["category"] == "code_help" for q in out) >= n * 0.15:
            continue
        out.append(p)
        if len(out) >= n:
            break
    return [Task(f"cthink-{p['pid']}", "chat_thinking", p["category"], chat_think_task, {"prompt": p}) for p in out]


# ================================================================================================ instruction following
CODE_LANG_RX = re.compile(r"```|python|kivy|pandas|numpy|django|flask|torch|tensorflow|matplotlib|\bjava\b|javascript|"
                          r"typescript|\bc\b|c\+\+|c#|\bsql\b|mysql|postgres|html|css|\brust\b|\bgo\b|golang|ruby|php|"
                          r"swift|kotlin|bash|shell|\br\b|matlab|julia|scala|perl|haskell|lua|dart|excel|vba|"
                          r"powershell|react|node|\.net|unity|godot|arduino|assembly|\basm\b|fortran|cobol|solidity|flutter|"
                          r"\bdart\b|android|\bios\b|swiftui|wordpress|jquery|\bvue\b|angular|svelte|laravel|spring|"
                          r"unreal|roblox|minecraft|\bapi\b", re.I)


def build_if_full(n: int, multi_share: float, seed: int) -> list[Task]:
    rng = random.Random(f"{seed}:iffull")
    used: set = set()                                 # base prompts may repeat prompts used in chat
    n_code, n_wri = n // 8, n // 5
    gen = S.pool_take(rng, used, ["explain_teach", "practical_advice", "brainstorming_lists", "factual_qa"],
                      n - n_code - n_wri, max_words=40)
    wri = S.pool_take(rng, used, ["writing_create"], n_wri, max_words=40)
    code = S.pool_take(rng, used, ["code_help"], n_code, max_words=40,
                       kind_filter=lambda r: bool(re.search(r"\b(write|create|make|implement|code|function|program|"
                                                            r"script|algorithm)\b", r["t1"], re.I))
                       and not CODE_LANG_RX.search(r["t1"]) and BH.code_line_frac(r["t1"]) == 0
                       and "def " not in r["t1"] and "import " not in r["t1"])
    allp = [(p, "long" if p["category"] == "explain_teach" else "general") for p in gen] + \
        [(p, "writing") for p in wri] + [(p, "code") for p in code]
    rng.shuffle(allp)
    k = max(1, round(1 / max(1e-9, multi_share)))
    return [Task(f"if-{p['pid']}", "instruction_following", p["category"], S.if_task,
                 {"prompt": p, "kind": kind, "multi": i % k == 0}) for i, (p, kind) in enumerate(allp)]


# ================================================================================================ system prompts
EXTRA_SYSTEMS = [
    ("bullets3", "Answer every question with exactly three bullet points, each one sentence.",
     lambda a: len([l for l in a.splitlines() if re.match(r"^\s*[-*•]\s", l)]) == 3, "pool"),
    ("no_jargon", "Avoid technical jargon. If you must use a technical word, explain it in brackets.",
     None, "pool"),
    ("ask_first", "If a request is ambiguous, ask one short clarifying question before answering; otherwise answer "
                  "directly.", None, "pool"),
    ("formal", "Write in a formal, professional tone. No emojis, no exclamation marks.",
     lambda a: "!" not in strip_code(a), "pool"),
    ("tldr_first", "Start every reply with a line 'TL;DR:' followed by a one-sentence summary, then the details.",
     lambda a: a.strip().startswith("TL;DR:"), "pool"),
    ("examples2", "Always include exactly two short examples, labelled 'Example 1:' and 'Example 2:'.",
     lambda a: "Example 1:" in a and "Example 2:" in a and "Example 3:" not in a, "pool"),
    ("max3sent", "Keep replies to at most three sentences, no lists.",
     lambda a: len(sentences(a)) <= 3 and not re.search(r"^\s*([-*•]|\d+[.)])\s", a, re.M), "pool"),
    ("encouraging", "You are an encouraging study buddy. Be warm and positive, and end with a short motivating line.",
     None, "pool"),
    ("table_cmp", "When the user asks to compare things, answer with a markdown table first, then one sentence of "
                  "advice. Otherwise answer normally.", None, "pool"),
    ("eli12", "Explain everything as you would to a 12-year-old: simple words, short sentences, one everyday "
              "comparison.", lambda a: prose_words(a) <= 200, "pool"),
]


async def sysgen_task(ctx, task: Task) -> list[dict]:
    """The teacher invents operator system messages on a theme (+ fitting and out-of-scope user messages); each user
    message becomes a system-prompt row, answered under that system message and screened by the judge."""
    rng = task.rng(ctx)
    r = await ctx.ds.chat([{"role": "system", "content": "You help build a system-prompt training set for a small "
                                                         "assistant. Return JSON only."},
                           {"role": "user", "content": P.SYS_GEN_USER.format(n=task.data["n"],
                                                                             theme=task.data["theme"])}],
                          slice_name=task.slice, json_mode=True, temperature=1.0, max_tokens=1200)
    o = parse_json(r["content"]) or {}
    outs = []
    for i, it in enumerate((o.get("items") or [])[:task.data["n"]]):
        if not isinstance(it, dict) or not str(it.get("system", "")).strip():
            continue
        sysmsg = str(it["system"]).strip()
        restricts = bool(it.get("restricts_topics"))
        qs = [(q, "in_scope") for q in (it.get("in_scope") or [])[:2]]
        oos = (it.get("out_of_scope") or [])[:1]
        qs += [(q, "out_of_scope" if restricts else "in_scope") for q in oos]
        sid = f"gen{i}"
        sysdef = (sid, sysmsg, None, "generated")
        for j, (q, scope) in enumerate(qs):
            q = str(q).strip()
            if not q:
                continue
            turns = 2 if (scope == "in_scope" and rng.random() < 0.15) else 1
            sub = Task(f"{task.id}-{i}-{j}", task.slice, f"generated_{scope}", S.system_task,
                       {"system": sysdef, "user": q, "turns": turns,
                        "prov": prov("generated (deepseek-flash) operator system message + user message", S.DS_LICENCE,
                                     f"{task.id}-{i}-{j}", system_theme=task.data["theme"], scope=scope)})
            res = await S.system_task(ctx, sub)
            for x in res:
                if x["ok"] and scope == "out_of_scope":
                    a = x["row"]["messages"][-1]["content"]
                    good = nwords(a) <= 70
                    x["checks"]["out_of_scope_brief"] = good
                    if not good:
                        x["ok"], x["reason"] = False, "check_failed:out_of_scope_brief"
            outs += res
    return outs or [bad(task, "system_generation_failed")]


def build_system_full(n_pool_rows: int, n_gen_tasks: int, per_gen: int, multi_share: float, seed: int,
                      maths_rows: list[dict]) -> list[Task]:
    rng = random.Random(f"{seed}:sysfull")
    used: set = set()
    systems = [s for s in S.SYSTEMS if s[3] == "pool"] + EXTRA_SYSTEMS
    users = S.pool_take(rng, used, ["explain_teach", "practical_advice", "factual_qa", "brainstorming_lists",
                                    "writing_create"], n_pool_rows, max_words=60)
    tasks = []
    soc = [s for s in S.SYSTEMS if s[0] == "socratic"][0]
    for i, p in enumerate(users):
        s = systems[i % len(systems)]
        tasks.append(Task(f"sys-{s[0]}-{p['pid']}", "system_prompts", s[0], S.system_task,
                          {"system": s, "user": p["t1"], "turns": 2 if rng.random() < multi_share else 1,
                           "prov": prov(p["dataset"] + " (user turn) + in-house system prompt", p["licence"], p["sid"],
                                        system_prompt_id=s[0], sensitive_topic=p["sensitive"])}))
    for m in maths_rows:
        tasks.append(Task(f"sys-socratic-{m['sid']}", "system_prompts", "socratic", S.system_task,
                          {"system": soc, "user": m["problem"], "turns": 1,
                           "prov": prov(f"datafactory/{m['job']} ({m.get('provider')})", S.FACTORY_NOTE, m["sid"],
                                        system_prompt_id="socratic")}))
    for j in range(n_gen_tasks):
        theme = P.SYS_THEMES[j % len(P.SYS_THEMES)]
        if j >= len(P.SYS_THEMES):                       # a distinct request per task (no shared cache entry)
            theme += f" (set {j // len(P.SYS_THEMES) + 1}: different organisations and rules from a first set)"
        tasks.append(Task(f"sysgen-{j:04d}", "system_prompts", "generated", sysgen_task,
                          {"theme": theme, "n": per_gen}))
    return tasks


# ================================================================================================ long documents (x3)
async def longdoc3_task(ctx, task: Task) -> list[dict]:
    rng = task.rng(ctx)
    d = task.data["doc"]
    doc_block = (f"# {d['title']}\n\n" if d["title"] else "") + d["text"]
    req, limit = rng.choice(S.SUMMARY_REQS)
    base = "DOCUMENT:\n" + doc_block + "\n\n"
    r1 = await ctx.ds.chat([{"role": "system", "content": P.LONG_SYS},
                            {"role": "user", "content": base + P.LONG_SUMMARY.format(request=req, limit=limit - 40)}],
                           slice_name=task.slice, json_mode=True, temperature=0.4, max_tokens=700)
    r2 = await ctx.ds.chat([{"role": "system", "content": P.LONG_SYS}, {"role": "user", "content": base + P.LONG_QA}],
                           slice_name=task.slice, json_mode=True, temperature=0.6, max_tokens=900)
    o1, o2 = parse_json(r1["content"]) or {}, parse_json(r2["content"]) or {}
    summ = str(o1.get("reply", "")).strip()
    nd = S.norm_ws(d["text"])
    good = []
    for q in [q for q in (o2.get("qa") or []) if isinstance(q, dict)]:
        quote = S.norm_ws(str(q.get("quote", ""))).strip(" .\"'")
        found = bool(quote) and len(quote.split()) >= 5 and quote in nd
        if found and 0 < nwords(q.get("answer", "")) <= 80 and 4 <= nwords(q.get("question", "")) <= 40:
            good.append((q, nd.find(quote) / max(1, len(nd))))
    checks = {"summary_ok": 20 <= nwords(summ) <= limit + 40, "qa_quotes_verbatim": len(good) >= 2,
              "qa_reaches_second_half": any(pos >= 0.5 for _, pos in good)}
    if not all(checks.values()):
        return [bad(task, "check_failed:" + ",".join(k for k, v in checks.items() if not v), checks,
                    {"n_qa": len(o2.get("qa") or []), "n_good": len(good)})]
    good.sort(key=lambda x: -x[1])
    pv = prov(d["source"] + " (document)", d["licence"], d["doc_id"], revision=d["revision"], url=d.get("url"),
              title=d.get("title"))
    lead = lambda: rng.choice(S.LEADS)

    def first_user(request):
        return f"{lead()}{doc_block}\n\n{request}" if rng.random() < 0.7 else f"{request}\n\n{doc_block}"

    def ans(q):
        a = str(q["answer"]).strip()
        if rng.random() < 0.4:
            a += f"\n\nThe text says: \"{str(q['quote']).strip().strip(chr(34))}\""
        return a

    rows_ = []
    # row A: summary (+ a follow-up question when a third grounded QA exists)
    msgs = [{"role": "user", "content": first_user(req)}, {"role": "assistant", "content": summ}]
    extra = good[2:3]
    for q, _ in extra:
        msgs += [{"role": "user", "content": str(q["question"]).strip()}, {"role": "assistant", "content": ans(q)}]
    v = {"method": "summary length checked; answers grounded by a verbatim quote from the document",
         "passed": True, "details": {"doc_tokens_tokenizer_v1": d["tokens"], "row": "summary"}}
    sub = Task(task.id + "-sum", task.slice, task.category, longdoc3_task)
    rows_.append(ok(sub, row(sub, msgs, False, v, pv), checks))
    # rows B, C: one question each (B is the late-in-document question)
    for k, (q, pos) in enumerate(good[:2]):
        msgs = [{"role": "user", "content": first_user(str(q["question"]).strip())},
                {"role": "assistant", "content": ans(q)}]
        v = {"method": "answer grounded: supporting quote found verbatim in the document", "passed": True,
             "details": {"doc_tokens_tokenizer_v1": d["tokens"], "row": "question", "quote_position": round(pos, 2)}}
        sub = Task(f"{task.id}-q{k}", task.slice, task.category, longdoc3_task)
        rows_.append(ok(sub, row(sub, msgs, False, v, pv), checks))
    return rows_


def build_longdoc_full(n_docs: int, path=None) -> list[Task]:
    path = path or CACHE / "longdocs_full_v1.jsonl"
    docs = [json.loads(l) for l in open(path, encoding="utf-8")][:n_docs]
    return [Task(f"long-{hashlib.sha256(d['doc_id'].encode()).hexdigest()[:10]}", "long_documents",
                 "wikipedia" if "wikipedia" in d["source"] else "finepdfs_edu", longdoc3_task, {"doc": d})
            for d in docs]


# ================================================================================================ safety
SAFETY_CATS_FULL = S.SAFETY_CATS + [
    ("stalking", "tracking someone's phone, location or messages without consent"),
    ("theft", "shoplifting, stealing, or avoiding paying"),
    ("fake_documents", "fake IDs, fake doctor's notes, forged signatures or certificates"),
    ("extremism", "praising or recruiting for violent extremist groups"),
    ("dangerous_challenges", "risky stunts, dangerous diets or viral challenges that can hurt people"),
    ("animal_cruelty", "hurting or neglecting animals"),
    ("gambling_scams", "rigging bets, pyramid schemes, get-rich-quick fraud"),
    ("cheating_partner", "spying on or deceiving a partner"),
    ("explicit_content", "requests for sexual content (keep the wording mild and non-graphic)"),
    ("medical_risk", "risky self-medication or stopping prescribed treatment without a doctor"),
]
SAFETY_ANGLES = ["casual and short", "phrased as a hypothetical", "by a student", "by a worried parent",
                 "mentioning an everyday setting (school, work, home)", "polite and indirect", "blunt and direct",
                 "framed as curiosity", "with a made-up excuse", "by a small-business owner"]


def build_safety_full(n_decline: int, n_answer: int, per_call: int, seed: int) -> list[Task]:
    rng = random.Random(f"{seed}:safetyfull")
    n_calls = max(1, -(-max(n_decline, n_answer) // per_call))
    tasks, used = [], set()
    for j in range(n_calls):
        cat = SAFETY_CATS_FULL[j % len(SAFETY_CATS_FULL)]      # self-harm is 1 of 22 categories (~5%)
        angle = rng.choice(SAFETY_ANGLES)
        k = SAFETY_ANGLES.index(angle)
        while (cat[0], angle) in used:      # identical requests would be served the same cached reply (duplicates)
            k = (k + 1) % len(SAFETY_ANGLES)
            angle = SAFETY_ANGLES[k]
            if len([u for u in used if u[0] == cat[0]]) >= len(SAFETY_ANGLES):
                break
        used.add((cat[0], angle))
        tasks.append(Task(f"safety-{cat[0]}-{j:04d}", "safety", cat[0], S.safety_task,
                          {"cat": cat, "n_decline": per_call, "n_answer": per_call, "angle": angle}))
    return tasks


# ================================================================================================ function calls
FC_DOMAINS_FULL = S.FC_DOMAINS + [
    "banking app (mock data only)", "grocery delivery", "ride hailing", "doctor appointment booking", "gym class booking",
    "podcast player", "email inbox (mock)", "file storage search", "translation glossary lookup", "currency budgeting",
    "school grades portal", "volunteer shift scheduling", "warehouse robots", "solar panel monitoring",
    "greenhouse sensors", "bike sharing", "parcel lockers", "conference room booking", "library room booking",
    "farm irrigation", "home energy usage", "kids' chore tracker", "medication reminders (mock)", "language lessons",
    "recipe scaling", "car rental", "boat tours", "museum tickets", "bus ticket purchase", "restaurant menu search",
    "wine pairing", "weather alerts", "air quality", "earthquake reports", "stock watchlist (mock prices)",
    "crypto price lookup (mock prices)", "news headlines (mock)", "sports league standings", "chess puzzles",
    "trivia questions"]
FC_TWISTS = ["", "", "the user gives one value in different units than the tool expects",
             "the user mentions a budget or limit", "the user is in a hurry and writes tersely",
             "the request includes two related details", "the user writes casually with a small typo",
             "the user refers to a named place, product or person", "the user asks for the cheapest or fastest option",
             "the user includes an exact date and time"]


def build_fc_full(n: int, seed: int) -> list[Task]:
    rng = random.Random(f"{seed}:fcfull")
    kinds = ["single"] * 4 + ["parallel"] * 2 + ["multi_step"] * 2 + ["no_call"] * 1 + ["clarify"] * 2
    tasks = []
    for i in range(n):
        dom = rng.choice(FC_DOMAINS_FULL)
        tw = rng.choice(FC_TWISTS)
        kind = kinds[i % len(kinds)]
        tasks.append(Task(f"fc-{i:05d}", "json_function_calls", kind, S.fc_task,
                          {"kind": kind, "domain": dom + (f" (scenario: {tw})" if tw else "")}))
    return tasks
