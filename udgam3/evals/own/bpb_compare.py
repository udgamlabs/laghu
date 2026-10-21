"""Paired bootstrap over documents for the difference in fresh-text bits per byte between two models."""
import json, random, sys
def load(p):
    d = json.load(open(p)); return {r["i"]: (r["bits"], r["bytes"]) for r in d["per_doc"]}, d["bpb"]
def diff(a, b, n=4000, seed=0):
    A, ba = load(a); B, bb = load(b); ids = sorted(set(A) & set(B)); rng = random.Random(seed); ds = []
    for _ in range(n):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        ds.append(sum(B[i][0] for i in s) / sum(B[i][1] for i in s) - sum(A[i][0] for i in s) / sum(A[i][1] for i in s))
    ds.sort(); return bb - ba, ds[int(0.025 * n)], ds[int(0.975 * n)]
if __name__ == "__main__":
    D = sys.argv[1]; names = sys.argv[2:]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d, lo, hi = diff(f"{D}/bpb_{names[i]}.json", f"{D}/bpb_{names[j]}.json")
            print(f"{names[j]:22s} - {names[i]:22s}: {d:+.4f} bpb  [95% CI {lo:+.4f}, {hi:+.4f}]")
