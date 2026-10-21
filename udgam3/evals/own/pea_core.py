"""Udgam Probe Suite v1.1 scoring: CORE = the 7 skill types that discriminate at 50-400M (validated 2026-10-17/18);
FRONTIER (days, order) reported separately (at or below chance for every <=400M model tested; copy bias); COUNT dropped
(raw accuracy reflected the prior of the answer '1'). Raw likelihood scoring (calibration did not help numeric answers).
Paired bootstrap over items for the difference between two models."""
import json, random, sys
CORE = ["add", "mul", "units", "sequence", "word_problem", "compare", "syllogism"]
FRONTIER = ["days", "order"]
def load(p):
    d = json.load(open(p)); return {r["id"]: r for r in d["items"]}
def core_acc(items, types=CORE):
    xs = [r["correct"] for r in items.values() if r["type"] in types]; return 100 * sum(xs) / len(xs), len(xs)
def paired(a, b, n=4000, seed=0):
    A, B = load(a), load(b); ids = [i for i in A if A[i]["type"] in CORE and i in B]
    d = [B[i]["correct"] - A[i]["correct"] for i in ids]; rng = random.Random(seed); bs = []
    for _ in range(n):
        bs.append(100 * sum(d[rng.randrange(len(d))] for _ in d) / len(d))
    bs.sort(); return 100 * sum(d) / len(d), bs[int(0.025 * n)], bs[int(0.975 * n)]
if __name__ == "__main__":
    D = sys.argv[1]; names = sys.argv[2:]
    for n in names:
        it = load(f"{D}/pea_{n}.json"); c, k = core_acc(it); f, _ = core_acc(it, FRONTIER)
        print(f"{n:26s} CORE {c:5.1f} (n={k})   frontier {f:5.1f}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd, lo, hi = paired(f"{D}/pea_{names[i]}.json", f"{D}/pea_{names[j]}.json")
            print(f"  {names[j]:24s} - {names[i]:24s}: {dd:+5.1f} [95% CI {lo:+.1f}, {hi:+.1f}]")
