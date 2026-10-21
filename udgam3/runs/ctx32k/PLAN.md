# Udgam-3 long-context stretch (train at 32K, claim 16K): plan v1, 2026-10-19

**Status:** [EXPLORATION]. Nothing has run. The data build is $0 on the Mac. The training run needs the pre-launch checklist
for this exact config and the user's go.

## What it is
Pretraining used 2,048-token rows. This stage teaches the model to use up to 32,768 tokens of context. We train at 32K
and claim only 16K (Gargi's lesson: recall drops past ~20K). Pre-registration 6.3 sets the pass bar:
- needle retrieval ≥ 95% at 4K, 8K, 12K and 16K;
- the base suite (validation splits) drops by ≤ 1 point.

## Lessons from Gargi's stretch (udgam2/specs/ctx32k.md, evals/udgam2_ctx32k/results.md)
| Gargi stage | Tokens | Data | Result |
|---|---|---|---|
| 1 | 0.5B | 50% long pool (books/papers/gov/code), 50% replay | needle 16K 63%: missed |
| top-up | 0.5B | 70% long, 30% replay | 16K 99%, but the short benchmarks regressed |
| repair r3 | 0.2B | 45% modern long docs, 10% long pool, 45% replay | benchmarks back; 16K kept |

So: about 1B tokens in total, with modern long text and plenty of replay from the start. We do it as **one stage**.

## Proposed design
| Item | Choice | Why |
|---|---|---|
| Start | main-run **EMA** weights, step 76,294 (`model_ema_076294.pt`) | the pre-registered base |
| Optimiser | fresh, same optimiser as the main run | Gargi's recipe |
| Learning rate | 30-step warmup to 10% of the main-run peak, then linear to 0 | Gargi's recipe (all 3 stages) |
| Length | seq 32,768; 16 rows × 32,768 = 524,288 tokens per step; **~1,900 steps ≈ 1.0B tokens** | Gargi needed ~1.2B in 3 stages |
| RoPE | unchanged: sliding layers θ 10K (window 512), full layers θ 1M | Gargi kept θ 1M; we claim 16K only |
| Data (~1.0B) | **40% long pool**: books, papers, government text and repo-level code from Gargi's builds 1 and 2. **20% modern long docs**: web and Wikipedia ≥ 4K tokens from build 3, Nemotron-CC-Math rows removed so no NVIDIA-agreement data enters. **40% replay**: v3 cool-down rows the main run never read (files after phase B `pq_idx` 63; ~2.4B available) | recall needs long docs; benchmarks need modern text and replay |
| Decontamination | every reused row re-checked against the **v3** 13-gram index (it adds IFEval, MATH-500 and BFCL to the earlier one) | the earlier builds used the v2 index |
| Held-out | Gargi's `val_long` docs (≥ 33K tokens, never trained on) + the main run's short validation shard | comparable before and after |
| Monitoring | live dashboard page from step 0 (rule), checkpoints to HF every 500 steps | the every-run rule |

## Evaluation (all $0 after the run)
1. Needle (5-digit code) at 4K / 8K / 12K / 16K / 24K / 32K, 100 trials each (`udgam2/ctx32k/hf_needle.py`).
2. Long-document bpb on `val_long`.
3. Base suite on the **validation** splits, against the main-run EMA: drop ≤ 1 point (pre-registration 6.3).
4. Fresh, web and reasoning bpb with the same after-training script.

## Where to run it (decision for the user)
| Option | Cost | Notes |
|---|---|---|
| **Colab G4** (recommended) | $0 extra (~25-30 of 205 compute units; ~3 h at an estimated 0.35-0.45B tokens/h, to be measured in the dry run) | Gargi's three stretch stages ran on Colab (`udgam2/ctx32k/colab_ctx32k_pipeline.sh`). Needs our HF token on the VM to save checkpoints |
| Shadeform 1× H100 or A100 | ~$7-9 | reuses the main-run launcher (self-delete, watcher, dashboard) |

## Steps
1. **Build data ($0, Mac):** pull the long rows from `YOUR_HF_NAMESPACE/udgam2-ctx32k-data{,2,3}` and the unread v3 phase-B
   shards, re-decontaminate, mix, shard. Run tests in the style of `test_long32k.py`.
2. **Dry run (~10 min):** 50 steps at 32K. Measure tokens/s, check for NaN, a checkpoint and resume.
3. **Checklist + user go** for the full run, with its exact config.
4. **Full run (~3 h)** with the dashboard, then evals 1-4. If needle@16K < 95% or the benchmarks drop > 1 point, stop and
   bring options to the user (no automatic top-up).
