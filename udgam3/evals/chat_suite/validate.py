"""Scorer and generation checks for the chat suite (PREREGISTRATION.md 2.4), from saved raw outputs (Mac).

  uv run --project evals/harness --with langdetect==1.0.9 --with immutabledict==4.2.1 --with math-verify==0.9.0 \
      python udgam3/evals/chat_suite/validate.py [--tag full]

Writes scored/<tag>/validation.json and prints a summary:
1. IFEval scorer: lm-eval 0.4.13's own run (its HF backend, --apply_chat_template) per prompt vs our scorer on the same
   response text (must agree on every item), plus prompt identity (our rendered prompt == lm-eval's).
2. Unbatched references on the first 40 GSM8K items (cap 512, one request at a time):
   a. production engine, bf16, batched vs one-by-one (same engine): identical token outputs (first 512 tokens);
   b. production vs HF transformers fp32 one-by-one (engine cross-check; also checks the prompt tokens are identical);
   c. fp32 runs, batched (padded for HF) vs one-by-one: identical outputs expected when batching is the only change.
3. bf16 batch-noise floor: the full GSM8K rerun with a shuffled request order (different batch composition):
   items whose correctness flips, accuracy difference, exact McNemar p.
"""
import argparse, glob, gzip, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import gsm8k_ok, visible  # noqa: E402
from score import ifeval_scores, mcnemar_p, wilson  # noqa: E402


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in gzip.open(p, "rt")} if os.path.exists(p) else {}


def first_div(a, b, n=512):
    a, b = a[:n], b[:n]
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(A, B, n=512):
    """Token identity of A vs B over shared ids (first n tokens)."""
    ids = [k for k in A if k in B and "ids" in A[k] and "ids" in B[k]]
    divs = [first_div(A[k]["ids"], B[k]["ids"], n) for k in ids]
    same_prompt = sum(A[k]["psha"] == B[k]["psha"] for k in ids)
    d = sorted(x for x in divs if x is not None)
    return dict(n=len(ids), identical=sum(x is None for x in divs), same_prompt=same_prompt,
                median_first_divergence=d[len(d) // 2] if d else None, min_first_divergence=d[0] if d else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    raw = os.path.join(HERE, "raw", a.tag)
    manifest = {m["name"]: m for m in json.load(open(os.path.join(HERE, "manifest.json")))["models"]}
    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in open(os.path.join(HERE, "items", "gsm8k.jsonl"))}
    ifdocs = {json.loads(l)["key"]: json.loads(l) for l in open(os.path.join(HERE, "items", "ifeval.jsonl"))}
    out = {}
    for name, spec in manifest.items():
        d = os.path.join(raw, name)
        if not os.path.isdir(d):
            continue
        V = out[name] = {}
        eng = spec["engine"]
        m0 = "off" if "off" in spec["modes"] else spec["modes"][0]
        prod = load(os.path.join(d, eng, f"gsm8k__{m0}__greedy.jsonl.gz"))
        one = load(os.path.join(d, eng, f"ref__{m0}__one_by_one.jsonl.gz"))
        if prod and one:
            V["bf16_batched_vs_one_by_one"] = compare(prod, one)
        ref32 = load(os.path.join(d, "hf_float32", f"ref__{m0}__one_by_one.jsonl.gz"))
        if prod and ref32:
            V["production_vs_hf_fp32_one_by_one"] = compare(prod, ref32)
        for e in ("vllm_float32", "hf_float32"):
            b32 = load(os.path.join(d, e, f"gsm8k__{m0}__greedy.jsonl.gz"))
            o32 = load(os.path.join(d, e, f"ref__{m0}__one_by_one.jsonl.gz"))
            if b32 and o32:
                V[f"{e}_batched_vs_one_by_one"] = compare(b32, o32)
        for sub in sorted(glob.glob(os.path.join(d, f"{eng}_order*"))):
            rr = load(os.path.join(sub, f"gsm8k__{m0}__greedy.jsonl.gz"))
            if not (prod and rr):
                continue
            meta = json.load(open(os.path.join(sub, "meta.json")))
            sp, ss = meta["specials"], meta["stop_strings"]
            ids = [k for k in prod if k in rr]
            ok1 = {k: gsm8k_ok(visible(prod[k]["raw"], sp, ss), gold[k]) for k in ids}
            ok2 = {k: gsm8k_ok(visible(rr[k]["raw"], sp, ss), gold[k]) for k in ids}
            b = sum(ok1[k] and not ok2[k] for k in ids); c = sum(ok2[k] and not ok1[k] for k in ids)
            V[f"noise_{os.path.basename(sub)}"] = dict(n=len(ids), same_output=sum(prod[k]["osha"] == rr[k]["osha"] for k in ids),
                                                     acc_production=wilson(sum(ok1.values()), len(ids))[0], acc_rerun=wilson(sum(ok2.values()), len(ids))[0],
                                                     flips=b + c, mcnemar_p=mcnemar_p(b, c))
        for p in glob.glob(os.path.join(d, "lmeval_ifeval", "*", "samples_ifeval_*.jsonl")):
            rows = [json.loads(l) for l in open(p)]
            agree = n_inst = 0
            for r in rows:
                ours = ifeval_scores(ifdocs[r["doc"]["key"]], r["filtered_resps"][0])
                agree += all(ours[k] == r[k] for k in ("prompt_level_strict_acc", "inst_level_strict_acc", "prompt_level_loose_acc", "inst_level_loose_acc"))
            res = json.load(open(glob.glob(os.path.join(os.path.dirname(p), "results_*.json"))[0]))["results"]["ifeval"]
            ours_run = load(os.path.join(d, eng, "ifeval__" + m0 + "__greedy.jsonl.gz"))
            V["ifeval_lmeval_check"] = dict(n=len(rows), our_scorer_agrees_on_lmeval_responses=agree,
                                            lmeval_metrics={k: v for k, v in res.items() if "stderr" not in k and k != "alias"})
            if ours_run:
                meta = json.load(open(os.path.join(d, eng, "meta.json")))
                sp, ss = meta["specials"], meta["stop_strings"]
                same_text = same_strict = 0
                for r in rows:
                    o = ours_run.get(f"ifeval-{r['doc']['key']}")
                    if o is None:
                        continue
                    v = visible(o["raw"], sp, ss)
                    same_text += v == r["filtered_resps"][0]
                    same_strict += ifeval_scores(ifdocs[r["doc"]["key"]], v)["prompt_level_strict_acc"] == r["prompt_level_strict_acc"]
                V["ifeval_lmeval_check"].update(our_engine_same_text=same_text, our_engine_same_prompt_strict=same_strict)
        print(name, json.dumps(V))
    os.makedirs(os.path.join(HERE, "scored", a.tag), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "scored", a.tag, "validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
