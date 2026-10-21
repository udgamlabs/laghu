"""Build / query the 8-gram eval-set index used to decontaminate SFT prompts.

Eval items come from udgam/decontam.py's `eval_items()` (MMLU test, ARC-C/E test, HellaSwag val, GSM8K test, PIQA,
Winogrande, OpenBookQA, CommonsenseQA, IFEval prompts, MATH-500, BFCL v3 questions), plus the GSM8K test questions
alone (so a copied question without its answer still matches). Grams shared by >= 3 distinct eval items (stock
question / instruction templates) are removed, as in decontam.py.

  python -I decontam8.py build              # writes ~/.cache/udgam3_sft/eval8_v1.npz + gen/eval8_v1.json
  python -I decontam8.py check "some text"
"""
from __future__ import annotations

import collections
import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import EVAL8, GEN, REPO, GramIndex, eval_index  # noqa: E402

NAMES = ["mmlu", "arc", "hellaswag", "gsm8k", "piqa", "winogrande", "openbookqa", "commonsenseqa", "ifeval", "bfcl",
         "math500"]


def items():
    sys.path.insert(0, str(REPO / "udgam"))
    import decontam as DC  # noqa: E402
    for name, item, text in DC.eval_items():
        yield name, item, text
        if name == "gsm8k":
            yield name, item, text.split("\n", 1)[0]


def build():
    t0 = time.time()
    counts = collections.Counter()

    def counted():
        for name, item, text in items():
            counts[name] += 1
            yield name, item, text

    idx, removed = GramIndex.build(counted(), NAMES, common=3)
    idx.save(EVAL8)
    meta = {"n": 8, "texts_per_eval": dict(counts), "unique_grams": int(idx.keys.size), "template_grams_removed": removed,
            "seconds": round(time.time() - t0, 1), "path": str(EVAL8),
            "normalisation": "build_heldout.norm_for_hash tokens, blake2b-64 per 8-gram"}
    (GEN / "eval8_v1.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        print(eval_index().hits(" ".join(sys.argv[2:])))
