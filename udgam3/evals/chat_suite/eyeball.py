"""Ten raw outputs per model per task for reading by eye (PREREGISTRATION.md 2.4.3), with what the scorer decided.

  python3 udgam3/evals/chat_suite/eyeball.py [--tag full]   ->  eyeball/<tag>/<model>.md

The ten items are fixed (evenly spaced through each task's item list), so every model is read on the same items.
"""
import argparse, gzip, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import split_thinking, visible  # noqa: E402


def clip(s, head=700, tail=300):
    s = s.strip()
    return s if len(s) <= head + tail + 20 else s[:head] + f"\n[... {len(s) - head - tail} chars ...]\n" + s[-tail:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(HERE, "manifest.json")))["models"]
    items = {}
    for k, f, key in [("gsm8k", "gsm8k.jsonl", "question"), ("math", "math_l123.jsonl", "problem"), ("ifeval", "ifeval.jsonl", "prompt"),
                      ("mt", "mt.jsonl", "q1")]:
        items[k] = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "items", f)))}
    loop_items = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "items", "loop.jsonl")))}
    od = os.path.join(HERE, "eyeball", a.tag)
    os.makedirs(od, exist_ok=True)
    for spec in manifest:
        d = os.path.join(HERE, "raw", a.tag, spec["name"], spec["engine"])
        sd = os.path.join(HERE, "scored", a.tag, spec["name"])
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        sp, ss = meta["specials"], meta["stop_strings"]
        lines = [f"# {spec['name']} ({spec['repo']}@{spec['sha'][:10]}, {spec['engine']}): ten outputs per task\n"]
        js = []
        for f in sorted(os.listdir(d)):
            if not f.endswith(".jsonl.gz") or f.startswith(("ref__", "mt2__")):
                continue
            key = f[:-9]
            task = key.split("__")[0]
            recs = [json.loads(l) for l in gzip.open(os.path.join(d, f), "rt")]
            scored = {}
            sk = {"mt1": "mt__", "e2": "e2__"}.get(task, "") + key.split("__")[1] if task in ("mt1", "e2") else key
            if os.path.exists(os.path.join(sd, sk + ".jsonl")):
                scored = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(sd, sk + ".jsonl"))}
            mt2 = {}
            if task == "mt1" and os.path.exists(os.path.join(d, f.replace("mt1__", "mt2__"))):
                mt2 = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(os.path.join(d, f.replace("mt1__", "mt2__")), "rt")}
            step = max(1, len(recs) // 10)
            lines.append(f"\n## {key} ({len(recs)} items; showing every {step}th)\n")
            for r in recs[::step][:10]:
                th, ans, unclosed = split_thinking(r["raw"])
                s = scored.get(r["id"], {})
                s = {k: v for k, v in s.items() if k not in ("id", "inst_level_strict_acc", "inst_level_loose_acc")}
                q = (items.get("mt" if task == "mt1" else task, {}).get(r["id"]) or {})
                prompt = q.get("question") or q.get("problem") or q.get("prompt") or q.get("q1") or loop_items.get(r["id"], {}).get("prompt", "")
                if task == "e2":
                    prompt = items["gsm8k"][r["id"]]["question"] + "  (second turn)"
                gold = q.get("gold") if task in ("gsm8k", "math") else (q.get("check") if task == "mt1" else None)
                lines.append(f"### {r['id']}  finish={r['finish']} tokens={r['n']} thinking_chars={len(th)}{' UNCLOSED' if unclosed else ''}  gold={gold!r}  scored={s}")
                lines.append(f"**Prompt:** {clip(prompt, 300, 0)}\n\n```text\n{clip(visible(r['raw'], sp, ss))}\n```")
                js.append(dict(key=key, id=r["id"], finish=r["finish"], n=r["n"], thinking_chars=len(th), unclosed=unclosed, gold=gold,
                               scored=s, prompt=prompt[:300], reply=visible(r["raw"], sp, ss),
                               followup=q.get("q2") if task == "mt1" else None,
                               reply2=visible(mt2[r["id"]]["raw"], sp, ss) if task == "mt1" and r["id"] in mt2 else None))
                if task == "mt1" and r["id"] in mt2:
                    lines.append(f"**Follow-up:** {q.get('q2')}\n\n```text\n{clip(visible(mt2[r['id']]['raw'], sp, ss), 500, 200)}\n```")
        open(os.path.join(od, spec["name"] + ".md"), "w").write("\n".join(lines) + "\n")
        with open(os.path.join(od, spec["name"] + ".jsonl"), "w") as f:
            for x in js:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print("wrote", os.path.join(od, spec["name"] + ".md"))


if __name__ == "__main__":
    main()
