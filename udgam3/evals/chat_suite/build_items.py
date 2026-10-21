"""Fixed item files for the chat suite, built on the Mac from pinned dataset revisions (no model is run).

  uv run --project evals/harness python udgam3/evals/chat_suite/build_items.py

Writes items/*.jsonl (gitignored: public test sets stay out of the repo) and items/ITEMS.json (revisions, counts,
sha256 of each file; committed), so the same items can be rebuilt and checked.
- gsm8k.jsonl: openai/gsm8k main test, 1,319 items, gold = text after "####".
- math_l123.jsonl: HuggingFaceH4/MATH-500 test, levels 1-3.
- ifeval.jsonl: google/IFEval train (541 prompts; the split lm-eval's `ifeval` task uses), all fields kept for scoring.
- mt.jsonl: mt_heldout_v2.jsonl (214 two-turn items, built by build_mt_heldout.py).
- loop.jsonl: single-turn prompts of the held-out chat set udgam3/evals/chat_prompts/heldout_v1.jsonl, filtered as in
  udgam2/multiturn/loop_check.py (>= 97% ASCII, < 600 characters). That set is never used in training (the SFT data is
  decontaminated against heldout_v1_hashes.txt), unlike the Udgam-2 SFT validation pool loop_check.py used for Gargi.
"""
import hashlib, json, os

from datasets import load_dataset
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "items")
DATASETS = {"gsm8k": ("openai/gsm8k", "main", "test"), "math": ("HuggingFaceH4/MATH-500", None, "test"),
            "ifeval": ("google/IFEval", None, "train")}


def write(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(file=name, n=len(rows), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())


def main():
    os.makedirs(OUT, exist_ok=True)
    api, meta = HfApi(), {}
    for key, (repo, cfg, split) in DATASETS.items():
        sha = api.dataset_info(repo).sha
        ds = load_dataset(repo, cfg, split=split, revision=sha)
        if key == "gsm8k":
            rows = [dict(id=f"gsm8k-{i:04d}", question=r["question"], answer=r["answer"],
                         gold=r["answer"].split("####")[-1].strip().replace(",", "")) for i, r in enumerate(ds)]
            info = write("gsm8k.jsonl", rows)
        elif key == "math":
            rows = [dict(id=r["unique_id"], problem=r["problem"], gold=r["answer"], level=int(r["level"]), subject=r["subject"])
                    for r in ds if int(r["level"]) <= 3]
            info = write("math_l123.jsonl", rows)
            info["by_level"] = {lv: sum(1 for r in rows if r["level"] == lv) for lv in (1, 2, 3)}
        else:
            rows = [dict(id=f"ifeval-{r['key']}", **r) for r in ds]
            info = write("ifeval.jsonl", rows)
        meta[key] = dict(repo=repo, config=cfg, split=split, revision=sha, **info)
    mt = [json.loads(l) for l in open(os.path.join(HERE, "mt_heldout_v2.jsonl"))]
    meta["mt"] = dict(source="udgam3/evals/chat_suite/mt_heldout_v2.jsonl", **write("mt.jsonl", mt))
    chat = [json.loads(l) for l in open(os.path.join(HERE, "..", "chat_prompts", "heldout_v1.jsonl"))]
    loop = [dict(id=r["id"], prompt=r["turns"][0], category=r["category"]) for r in chat
            if len(r["turns"]) == 1 and r["turns"][0].strip() and len(r["turns"][0]) < 600
            and sum(c.isascii() for c in r["turns"][0]) / len(r["turns"][0]) >= 0.97]
    meta["loop"] = dict(source="udgam3/evals/chat_prompts/heldout_v1.jsonl (single-turn, >=97% ASCII, <600 chars)", **write("loop.jsonl", loop))
    json.dump(meta, open(os.path.join(OUT, "ITEMS.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
