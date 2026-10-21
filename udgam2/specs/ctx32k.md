# Udgam-2 32K context stage (P4): spec

Status: decisions APPROVED by the user on 2026-10-11. Verification in progress, nothing trained yet.

## Decisions (user, 2026-10-11)
- **Start point:** extend the **base** (YOUR_HF_NAMESPACE/udgam2-base-ckpt, `udgam2-base-v1/model_ema_008500.pt`), then rerun thinking SFT and maths RL with the existing Colab pipelines (`udgam2/sft/`).
- **Tokens:** ~0.5B training tokens. Top up later only if the needle tests fall short.
- **Schedule:** one stage straight at seq 32,768. No 8K step.

## Fixed choices (from the 4K-stage lessons, not user decisions)
- **Architecture unchanged:**
  - Sliding layers: window 512, RoPE θ 10K.
  - Full layers (1 in 4, head_dim 128): RoPE θ 1M.
  - The 4K stage showed that usable length is set by the training length, not by θ.
- **Optimiser:** fresh, starting from EMA weights.
  - LR: 30-step warmup to 10% of the main-run peak, then linear decay to 0.
  - Weight decay: 0.
- **Batch:** 524,288 tokens = 16 rows × 32,768, giving ~950 steps.
- **Data:** 50% long documents + 50% replay of normal pretraining data.
  - Long documents longer than 32K tokens are split at paragraph breaks into pieces of ≤ 32,000 tokens, because the best-fit packer crops anything longer than a row.
- **Compute:** Colab G4 ($0 cash).
  - Fallback: Runpod 1×H100, about $5, only after the pre-launch checklist and an explicit go.

## Data (sources and licences in long_context_data.md)
- **Long pool:** ~0.28B tokenizer_v1 tokens.
  - Books ~38%: Gutenberg, pre-1929, LoC.
  - Papers ~27%: arXiv ≥12K, DOAB ≥16K, PubMed ≥8K.
  - Gov/wiki ~15%: UK Hansard, USGPO, Wikimedia ns0.
  - Code ~20%: Stack v2 permissive, repo-concatenated where possible.
- **Replay pool:** ~0.28B tokens from YOUR_HF_NAMESPACE/udgam-pretrain-v2b phase-B shards (the anneal mix).
- **Decontamination:** the same 13-gram index as pretraining (`data/decontam`).
- **Held-out long eval docs:** ≥ 33K tokens, from units not used for training.

## Checks
- **Needle (5-digit code) and copy test** at 2K / 4K / 8K / 16K / 32K.
  - Targets: ≥ 80% at 16K and ≥ 60% at 32K.
- **bpb by position** on held-out long docs.
- **bpb at 2,048** on web / task / reason val: must not rise more than 1%.
- **After the SFT + RL rerun:** GSM8K (math_eval.py) and the lm-eval suite, compared with the 2K think model.
- **Export:** HF `max_position_embeddings` 32,768, with a parity check at 32K.

## Stage 1 result (2026-10-12, [MEASURED]; details in evals/udgam2_ctx32k/results.md)
- **Needle:** 4K 100%, 8K 100%, 16K 63%, 32K 0%. Both targets missed.
- **Short-text bpb:** about +1%.

## Top-up (user-approved 2026-10-12)
- **Start:** stage 1's final weights (`udgam2-ctx32k/model_000954.pt`), fresh optimiser, same LR recipe, 954 steps (~0.5B tokens).
- **Data:** a fresh batch with no overlap with stage 1. Mix 70% long / 30% replay (YOUR_HF_NAMESPACE/udgam2-ctx32k-data2).
- **Eval:** the same val_long docs and needle seeds as stage 1.
- **Decision rule:** if needle@32K is still < 60%, declare the context honestly as the longest length that reaches ≥ 80%, then run SFT + RL.
- **Run:** `RUN=udgam2-ctx32k-t2 INIT_REPO=YOUR_HF_NAMESPACE/udgam2-ctx32k INIT_PT=udgam2-ctx32k/model_000954.pt INIT_META=udgam2-ctx32k/meta_000954.json DATA_REPO=YOUR_HF_NAMESPACE/udgam2-ctx32k-data2 bash colab_ctx32k_pipeline.sh 954`

## Top-up result and the benchmark regression (2026-10-12, [MEASURED])
- **Top-up needle:** 16K 99%, 20K 87%, 24K 33%, 32K 0%.
- **Commonsense benchmarks** (same-VM control) fell 4–7 points in stage 1 and did not recover in the top-up:
  - ARC-E 49.3 → 41.8
  - HellaSwag 36.7 → 32.4
  - PIQA 62.4 → 58.3
- **Weight blends** trade linearly; none keeps both (evals/udgam2_ctx32k/results.md).

## Repair stage (user-approved 2026-10-12)
- **Hypothesis [ESTIMATE]:** the long pool (books/OCR, papers, gov, code) has no modern web text, which shifts the model away from the web/commonsense style. The Udgam-384M 4K stage used web-heavy long docs and had no drop.
- **Start:** top-up final weights (`udgam2-ctx32k-t2/model_000954.pt`), fresh optimiser, same LR recipe (10% of peak, 30-step warmup, linear to 0).
- **Length:** 381 steps × 524,288 tokens ≈ 0.2B tokens at seq 32,768. About 1 h, ~10 Colab units.
- **Data (~0.21B tokens, YOUR_HF_NAMESPACE/udgam2-ctx32k-data3):**
  - 45% long modern docs (≥ 4,096 tokens) from the user-approved pretraining sources, same shares as the 4K stage: Ultra-FineWeb 55%, Wikipedia 15%, Nemotron-CC-Math 15%, github-code-clean 15%. Use units not used in pretraining or the 4K stage.
  - 10% long-pool rows reused from batch 2 (books/papers/gov/code), to keep long recall.
  - 45% fresh v2b phase-B replay (rows not used by batches 1 and 2).
  - Same val_long and val shard as before.
- **Gate (new, from the lesson):** the pipeline runs a quick lm-eval (ARC-E, HellaSwag, PIQA, WinoGrande) right after export. SFT + RL only if:
  - benchmarks improve over the top-up, and
  - needle@16K stays ≥ 80%.
- **Fallback:** the current think-32K models stay as they are.
