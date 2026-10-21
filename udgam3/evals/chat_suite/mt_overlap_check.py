"""Overlap of the multi-turn held-out set v2 with the training data that exists on 2026-10-19 (Udgam-2 multi-turn
seeds, Udgam-3 SFT full_v1 and pilot_v1). Pair-level: a conversation whose user turns contain q1 followed by q2.
Also counts q1 alone appearing as a user turn (short generic questions are expected to recur; only pairs leak the test).

  python3 udgam3/evals/chat_suite/mt_overlap_check.py > udgam3/evals/chat_suite/mt_overlap.txt
"""
import glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

items = [json.loads(l) for l in open(os.path.join(HERE, "mt_heldout_v2.jsonl"))]
q1s = {norm(r["q1"]): r["id"] for r in items}
pairs = {(norm(r["q1"]), norm(r["q2"])): r["id"] for r in items}
files = [os.path.join(ROOT, "udgam2", "multiturn", "seeds.jsonl")]
files += sorted(f for f in glob.glob(os.path.join(ROOT, "udgam3", "sft", "*", "*.jsonl"))
                if not os.path.basename(f).startswith(("rejected_", "requests_", "raw_cache", "run_requests")))
q1_hits, pair_hits, n_conv = {}, {}, 0
for f in files:
    if not os.path.exists(f):
        continue
    for line in open(f):
        try:
            msgs = json.loads(line).get("messages") or []
        except Exception:
            continue
        n_conv += 1
        users = [norm(m.get("content") if isinstance(m.get("content"), str) else "") for m in msgs if m.get("role") == "user"]
        for i, u in enumerate(users):
            if u in q1s:
                q1_hits.setdefault(q1s[u], set()).add(os.path.relpath(f, ROOT))
                if i + 1 < len(users) and (u, users[i + 1]) in pairs:
                    pair_hits.setdefault(pairs[(u, users[i + 1])], set()).add(os.path.relpath(f, ROOT))
print(f"scanned {n_conv} conversations in {len(files)} files")
print(f"pair (q1 then q2) overlaps: {len(pair_hits)}")
for k, v in sorted(pair_hits.items()):
    print("  PAIR", k, sorted(v))
print(f"q1 alone appears as a user turn: {len(q1_hits)} of {len(items)} items")
for k, v in sorted(q1_hits.items()):
    print("  q1", k, sorted(v))
