"""Udgam Probe Suite: procedurally generated exact-answer probes (our own benchmark; can't saturate, can't be in anyone's
training data, tests reasoning rather than memorised facts).

Ten skill types x three difficulty levels; every item has one correct answer by construction and 1-3 distractors.
Base models are scored by likelihood: the summed log-probability of each answer continuation after the prompt (answers
are short and similar in length); the highest wins. Per-item results are saved for paired comparisons.

  python pea.py gen OUT.jsonl [--per 100] [--seed 2026]            # 10 types x 3 levels x per items
  python pea.py score MODEL ITEMS.jsonl OUT.json [--device mps] [--revision SHA] [--trust-remote-code] [--add-bos auto|yes|no]
"""
import argparse, json, random, sys, time

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NONSENSE = ["blick", "florp", "wug", "dax", "zorp", "glim", "tove", "snerk", "vosk", "pliff", "quam", "drell"]
NAMES = ["Ana", "Ben", "Chen", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena"]
WORDS = ["strawberry", "banana", "mississippi", "bookkeeper", "committee", "balloon", "parallel", "assessment",
         "tomorrow", "occurrence", "necessary", "accommodate", "letter", "coffee", "address", "success"]


def num_distractors(rng, x, k=3):
    out = set()
    cands = [x + 1, x - 1, x + 10, x - 10, x + 2, x * 2 if x < 10 ** 6 else x + 100, x + 9, x - 9, x + 100, x - 100]
    rng.shuffle(cands)
    for c in cands:
        if c != x and c >= 0 and len(out) < k:
            out.add(c)
    return [str(c) for c in out]


def fmt(n):
    return f"{n:,}" if n >= 10000 else str(n)


def g_add(rng, lvl):
    d = {1: 2, 2: 4, 3: 6}[lvl]
    a, b = rng.randint(10 ** (d - 1), 10 ** d - 1), rng.randint(10 ** (d - 1), 10 ** d - 1)
    ans = a + b
    return f"Question: What is {fmt(a)} + {fmt(b)}?\nAnswer:", fmt(ans), [fmt(int(x)) for x in num_distractors(rng, ans)]


def g_mul(rng, lvl):
    a, b = {1: (rng.randint(2, 9), rng.randint(2, 9)), 2: (rng.randint(11, 99), rng.randint(2, 9)),
            3: (rng.randint(11, 99), rng.randint(11, 99))}[lvl]
    ans = a * b
    return f"Question: What is {a} × {b}?\nAnswer:", fmt(ans), [fmt(int(x)) for x in num_distractors(rng, ans)]


def g_compare(rng, lvl):
    if lvl == 1:
        x, y = rng.sample(range(1, 100), 2)
    elif lvl == 2:
        x, y = rng.sample(range(100, 10000), 2)
    else:   # decimals with different digit counts: the classic 0.7 vs 0.65 trap
        a = rng.randint(1, 9); b = rng.randint(10, 99)
        x, y = float(f"0.{a}"), float(f"0.{b}")
        if x == y:
            y = round(y + 0.01, 2)
    big = max(x, y)
    return f"Question: Which number is larger, {x} or {y}?\nAnswer:", str(big), [str(min(x, y))]


def g_days(rng, lvl):
    start = rng.randrange(7)
    n = {1: rng.randint(1, 6), 2: rng.randint(8, 20), 3: rng.randint(30, 400)}[lvl]
    ans = DAYS[(start + n) % 7]
    others = [d for d in DAYS if d != ans]; rng.shuffle(others)
    return f"Question: If today is {DAYS[start]}, what day of the week will it be in {n} days?\nAnswer:", ans, others[:3]


UNITS = [("minutes", "hours", 60), ("seconds", "minutes", 60), ("hours", "days", 24), ("days", "weeks", 7),
         ("grams", "kilograms", 1000), ("millilitres", "litres", 1000), ("centimetres", "metres", 100),
         ("months", "years", 12), ("minutes", "days", 1440), ("metres", "kilometres", 1000)]


def g_units(rng, lvl):
    small, big, f = rng.choice(UNITS)
    if lvl == 1:
        h = rng.randint(2, 12); q = f"How many {small} are in {h} {big}?"; ans = h * f
    elif lvl == 2:
        h = rng.randint(2, 12) + 0.5; q = f"How many {small} are in {h} {big}?"; ans = int(h * f)
    else:
        km = rng.randint(2, 30); q = f"How many metres are in {km}.{rng.randint(1,9)} kilometres?"
        ans = int(round(float(q.split(" in ")[1].split(" ")[0]) * 1000))
    return f"Question: {q}\nAnswer:", fmt(ans), [fmt(int(x)) for x in num_distractors(rng, ans)]


def g_count(rng, lvl):
    w = rng.choice(WORDS if lvl > 1 else ["cat", "dog", "apple", "tree", "moon", "glass", "hello", "sheep", "door",
                                          "paper", "green", "little", "cookie", "puppy", "river", "summer", "yellow",
                                          "rabbit", "kitten", "button", "pepper", "window", "garden", "school"])
    letters = sorted(set(w)); c = rng.choice(letters); ans = w.count(c)
    d = [str(x) for x in {ans + 1, max(0, ans - 1), ans + 2} if x != ans][:3]
    return f"Question: How many times does the letter '{c}' appear in the word '{w}'?\nAnswer:", str(ans), d


def g_syllogism(rng, lvl):
    a, b, c, d = rng.sample(NONSENSE, 4)
    if lvl == 1:
        q = f"All {a}s are {b}s. All {b}s are {c}s. Is every {a} a {c}?"; ans = "Yes"
    elif lvl == 2:
        q = f"All {a}s are {b}s. Some {b}s are {c}s. Must every {a} be a {c}?"; ans = "No"
    else:
        if rng.random() < 0.5:
            q = f"All {a}s are {b}s. No {b}s are {c}s. Can an {a} be a {c}?"; ans = "No"
        else:
            q = f"All {a}s are {b}s. All {b}s are {c}s. All {c}s are {d}s. Is every {a} a {d}?"; ans = "Yes"
    return f"Question: {q} Answer Yes or No.\nAnswer:", ans, ["No" if ans == "Yes" else "Yes"]


def g_sequence(rng, lvl):
    if lvl == 1:
        s, k = rng.randint(1, 20), rng.randint(2, 9); seq = [s + i * k for i in range(5)]; nxt = s + 5 * k
    elif lvl == 2:
        s, r = rng.randint(1, 12), rng.choice([2, 3, 4]); seq = [s * r ** i for i in range(4)]; nxt = s * r ** 4
    else:
        s, k, j = rng.randint(1, 9), rng.randint(2, 5), rng.randint(1, 3)
        seq = [s]; step = k
        for _ in range(4):
            seq.append(seq[-1] + step); step += j
        nxt = seq[-1] + step
    q = ", ".join(map(str, seq))
    return f"Question: What number comes next: {q}, ?\nAnswer:", fmt(nxt), [fmt(int(x)) for x in num_distractors(rng, nxt)]


def g_order(rng, lvl):
    n = {1: 3, 2: 4, 3: 5}[lvl]
    people = rng.sample(NAMES, n)            # people[0] tallest ... people[-1] shortest
    facts = [f"{people[i]} is taller than {people[i+1]}." for i in range(n - 1)]
    rng.shuffle(facts)
    if rng.random() < 0.5:
        q, ans = "Who is the shortest?", people[-1]
    else:
        q, ans = "Who is the tallest?", people[0]
    others = [p for p in people if p != ans]; rng.shuffle(others)
    return f"Question: {' '.join(facts)} {q}\nAnswer:", ans, others[:3]


def g_wordprob(rng, lvl):
    name = rng.choice(NAMES)
    if lvl == 1:
        a, b = rng.randint(3, 20), rng.randint(2, 15)
        q = f"{name} has {a} apples and buys {b} more. How many apples does {name} have now?"; ans = a + b
    elif lvl == 2:
        a, b, c = rng.randint(20, 60), rng.randint(2, 15), rng.randint(2, 15)
        q = f"{name} has {a} stickers, gives away {b}, then gets {c} more. How many stickers does {name} have now?"
        ans = a - b + c
    else:
        p, n, k = rng.randint(2, 9), rng.randint(3, 9), rng.randint(2, 5)
        q = (f"Pens cost {p} dollars each. {name} buys {n} pens and pays with {p * n + k * 5} dollars. "
             f"How much change does {name} get, in dollars?"); ans = k * 5
    return f"Question: {q}\nAnswer:", fmt(ans), [fmt(int(x)) for x in num_distractors(rng, ans)]


GEN = {"add": g_add, "mul": g_mul, "compare": g_compare, "days": g_days, "units": g_units, "count": g_count,
       "syllogism": g_syllogism, "sequence": g_sequence, "order": g_order, "word_problem": g_wordprob}


def gen(out, per, seed):
    rng = random.Random(seed); rows = []; seen = set()
    for t, f in GEN.items():
        for lvl in (1, 2, 3):
            k = 0; tries = 0
            while k < per and tries < per * 200:     # some easy levels have fewer than `per` distinct items
                tries += 1
                prompt, ans, dist = f(rng, lvl)
                if prompt in seen or ans in dist or not dist:
                    continue
                seen.add(prompt)
                rows.append({"id": f"pea-{t}-{lvl}-{k:04d}", "type": t, "level": lvl, "prompt": prompt, "answer": ans,
                             "choices": sorted([ans] + dist, key=lambda s: rng.random())})
                k += 1
    from collections import Counter
    print("items per type/level:", dict(Counter((r["type"], r["level"]) for r in rows)))
    with open(out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} items to {out}")


def score(model_id, items_path, out, device, revision, trust, add_bos):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=trust,
                                                 use_safetensors=True, dtype=torch.float32).to(device).eval()
    bos = tok.bos_token_id
    use_bos = add_bos == "yes" or (add_bos == "auto" and bos is not None)
    items = [json.loads(l) for l in open(items_path)]
    null_ids = tok("Question: N/A\nAnswer:", add_special_tokens=False)["input_ids"]
    if use_bos:
        null_ids = [bos] + null_ids
    res = []; t0 = time.time()
    for it in items:
        p_ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        if use_bos:
            p_ids = [bos] + p_ids
        def ll(prefix_ids, c):
            c_ids = tok(" " + c, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prefix_ids + c_ids], device=device)
            with torch.no_grad():
                lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
            tgt = ids[0, 1:]
            return lp.gather(1, tgt.unsqueeze(1)).squeeze(1)[len(prefix_ids) - 1:].sum().item()
        lls = [ll(p_ids, c) for c in it["choices"]]
        # calibration (PMI / contextual calibration): subtract each answer's likelihood after a content-free prompt, so
        # a model's fondness for some answer words (e.g. weekday or person names) does not count as reasoning
        priors = [ll(null_ids, c) for c in it["choices"]]
        cal = [x - y for x, y in zip(lls, priors)]
        pred = it["choices"][max(range(len(lls)), key=lambda i: lls[i])]
        pred_cal = it["choices"][max(range(len(cal)), key=lambda i: cal[i])]
        res.append({"id": it["id"], "type": it["type"], "level": it["level"], "correct": int(pred == it["answer"]),
                    "correct_cal": int(pred_cal == it["answer"]), "n_choices": len(it["choices"])})
    acc = sum(r["correct"] for r in res) / len(res)
    acc_cal = sum(r["correct_cal"] for r in res) / len(res)
    by_cal = {}
    for r in res:
        by_cal.setdefault(r["type"], []).append(r["correct_cal"])
    chance = sum(1 / r["n_choices"] for r in res) / len(res)
    by = {}
    for r in res:
        by.setdefault(r["type"], []).append(r["correct"])
    summary = {"model": model_id, "revision": revision, "n": len(res), "acc": acc, "acc_cal": acc_cal, "chance": chance,
               "by_type_cal": {k: sum(v) / len(v) for k, v in by_cal.items()},
               "by_type": {k: sum(v) / len(v) for k, v in by.items()}, "seconds": round(time.time() - t0, 1)}
    json.dump({**summary, "items": res}, open(out, "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gen"); g.add_argument("out"); g.add_argument("--per", type=int, default=100); g.add_argument("--seed", type=int, default=2026)
    s = sub.add_parser("score"); s.add_argument("model"); s.add_argument("items"); s.add_argument("out")
    s.add_argument("--device", default="mps"); s.add_argument("--revision", default=None)
    s.add_argument("--trust-remote-code", action="store_true"); s.add_argument("--add-bos", default="auto")
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a.out, a.per, a.seed)
    else:
        score(a.model, a.items, a.out, a.device, a.revision, a.trust_remote_code, a.add_bos)
