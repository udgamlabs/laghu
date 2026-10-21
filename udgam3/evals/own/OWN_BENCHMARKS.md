# Udgam Labs' own small-model benchmarks (v1.1, validated 2026-10-17/18)

**Why (user, 2026-10-17):** experiments must be able to decide. Public benchmarks sit near chance at 50-200M
(HellaSwag, WinoGrande, ARC-C), and they may be in rivals' training data, so we built and validated our own: aligned
with our goal (an efficient, general small model), contamination-free, and sensitive at this scale.
**Evidence:** [MEASURED] on the Mac (MPS, fp32), five public models, $0.

## 1. Fresh-text loss (bits per byte)

- **Data:** `fresh_v1/val.parquet`, 1,695 documents, 4.0 MB: English Wikipedia pages *created* 2026-09-01..10-15
  (CC-BY-SA) and arXiv abstracts submitted 2026-10-01..15 (CC0); built by `../neutral_text/build_neutral.py`. Text
  that no model released before then can have trained on.
- **Metric:** total bits / total UTF-8 bytes, per document, BOS if the model has one, 2,048-token sliding window
  (`bpb_hf.py`). Tokenizer-independent; every token scored exactly once (asserted).
- **Validation:**

| Model | bpb |
|---|---|
| SmolLM2-360M | 0.850 |
| Gargi-Base (ours) | 0.914 |
| Udgam-384M-Base-4K (ours) | 0.937 |
| SmolLM2-135M | 0.941 |
| Falcon-H1-Tiny-90M-Base | 0.979 |

  Paired bootstrap over documents (`bpb_compare.py`): every pairwise gap is significant; 95% CIs are ±0.001-0.003, and
  even the smallest gap (Udgam-384M vs SmolLM2-135M, −0.0043 [−0.0067, −0.0020]) is clear.
- **Also a live training monitor:** the same file is an extra validation set in the Udgam-3 runs (`--extra-val
  fresh=`; private HF `YOUR_HF_NAMESPACE/laghu-val-sets`).

## 2. Udgam Probe Suite (procedural exact-answer probes)

- **Items:** `pea_v1.jsonl` (`pea.py gen`, seed 2026): 2,717 generated questions with one correct answer by
  construction, scored by answer likelihood (`pea.py score`). Generated fresh: can't be in anyone's training data,
  can't saturate (add levels), and tests reasoning rather than memorised facts.
- **CORE score** (`pea_core.py`): the 7 types that discriminate at 50-400M, 2,000 items: addition, multiplication, unit
  conversion, sequences, word problems, number comparison, nonsense-word syllogisms.

| Model | CORE | frontier (days, order) |
|---|---|---|
| SmolLM2-135M | 44.4 | 18.0 |
| Udgam-384M-Base-4K | 61.9 | 14.6 |
| Gargi-Base | 69.2 | 17.6 |
| SmolLM2-360M | 71.2 | 24.4 |
| Falcon-H1-Tiny-90M-Base | 74.4 | 15.8 |

  Paired bootstrap over items: 95% CIs ±2-3 points; 9 of 10 pairs differ significantly (SmolLM2-360M vs Gargi:
  +1.9 [−0.3, +4.2]).
- **What validation removed:**
  - **Weekday arithmetic and ordering** are at or below chance for every ≤ 400M model, even with PMI calibration.
    Likely copy bias toward the day or names in the prompt. Kept as a *frontier* set, reported separately, not in
    CORE.
  - **Letter counting is dropped.** Its raw accuracy (~67% for every model) was the prior of the answer "1";
    calibrated, it fell to 0-5%.
  - **Calibration** (subtracting each answer's likelihood after a content-free prompt) is recorded per item but not
    used for CORE: it did not help numeric answers.

## 3. Planned (built when needed)

- **Assistant-answer bpb:** likelihood of high-quality answers to the 484 held-out real prompts
  (`../chat_prompts/heldout_v1.jsonl`).
- **Judged chat quality:** pairwise against rival chat models on the same prompts.
- **Our own instruction-constraint suite:** checker functions, as in `udgam3/sft/gen/constraints.py`, decontaminated
  from training.

## Targets for the Udgam-3 base model (≤ 200M size class)

- Fresh-text bpb below SmolLM2-135M's **0.941**.
- CORE probes above SmolLM2-135M's **44.4**, measured against the strongest ≤ 200M model, Falcon-H1-Tiny-90M at
  **74.4**.
