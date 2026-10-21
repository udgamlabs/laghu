# Udgam-3 main-run dry runs (2026-10-18)

**Evidence:** [MEASURED] from the pod logs (copies in `state/dry-*/`), HF `YOUR_HF_NAMESPACE/laghu-ckpt-test` and W&B
`udgamlabs/udgam3-main`. Configs: `dryrun_a.conf`, `dryrun_a_fp8.conf`, `dryrun_b.conf` (+ `common.sh`); code bundle
`train-udgam3/` (nanochat `udgam3`).

## Dry run A: bf16, real shape, real batch (1× H100 SXM Secure, EU-FR-1, pod `qjtxxgx578asth`)

300 steps × 524,288 positions (157M), TST bags of 4 for steps 0-74, cool-down data from step 240, checkpoints every
100 steps. **PASSED every check.**

| Check | Result |
|---|---|
| Data download + layout | 4 + 2 shards (1.1 GB) in 12 s; val shard 00399 found in both folders |
| Env | torch 2.9.1+cu128, transformers 5.18.0, CUDA OK, in 16 s |
| Preflight upload | 256 MB to the checkpoint repo OK |
| Speed, TST phase (steps 1-74) | **~224K tok/s** (2.34 s/step), MFU ~27.5% |
| Speed, normal phase | **~291K tok/s** (1.80 s/step), MFU ~35.8% |
| One-time stalls | step 0 compile 82 s; TST → normal recompile at step 75: 51 s |
| Peak memory | **57.8 GB** of 80 GB (device batch 32 × 2,048) |
| Held-out loss (bpb) | mixed val 4.83 → 1.695 (step 100) → 1.222 (200) → **1.123** (300); web 1.200; reason 1.047; **fresh text 4.64 → 1.297**; EMA: val 1.137, fresh 1.312 |
| Checkpoints | steps 100/200/300 pushed (newest 2 kept); final push verified |
| Export | raw + EMA to HF format, 151.8M params, bf16; logits parity vs nanochat max rel diff 2.5e-6, top-1 agreement 1.000 → PASS |
| Samples | prompts render; text is fluent-looking but repetitive, as expected after 157M tokens |
| End | DONE marker, logs uploaded, pod self-deleted |
| Cost | ~$1.13 (17 min × $3.99/h) [ESTIMATE from pod time] |

Lesson: per-GPU speed is **~30% higher than the plan's estimate** (200-220K), so a dollar buys more tokens than
`MAIN_RUN_PLAN.md` assumed.

## Dry run A-fp8: identical to A plus `--fp8` (pod `74ra9ieucs6nhr`)

Adoption rule (set before the run): ≥ 1.2× tok/s and held-out loss within 0.3% of A.
- TST phase: **164K tok/s vs 224K in bf16 (0.73×)**: FP8 is slower at this width (640): casting overhead outweighs the
  faster matmuls, as Karpathy reported for GPT-2-size models.
- Normal phase (steps 76-82): **218K tok/s vs 291K (0.75×)**.
- Training loss matched bf16 to within 0.1-0.4% at steps 30/60/74/82 (no held-out eval reached; not needed once speed
  failed).
- **Verdict: not adopted; the main run stays bf16.** Stopped at step 82 (~01:29 IST) instead of paying for the
  remaining 15 minutes; log saved in `state/dry-fp8/`. Cost ~$0.80 (12 min) [ESTIMATE from pod time].

## Dry run B: resume in a fresh pod on another GPU type (1× L40S Secure, pod `k4g21f5fcshkdz`)

Resumes A's step-300 checkpoint from HF and continues to step 400, with device batch 16 (16 gradient-accumulation micro-steps on one GPU).
- Resumed weights, optimiser and weight EMA: the step-300 evaluation reproduces A's exactly
  (val 1.122668 vs 1.122666, web 1.199864 vs 1.199861, fresh 1.297439 vs 1.297443).
- The cool-down data stream resumed too: A's checkpoint was saved in phase B, and the meta file's `data_phase` made B
  continue the phase-B shards at the saved row group (0/45 → 0/48), not restart them.
- Steps 300 → 400 trained normally: val 1.084, web 1.169, fresh 1.260 at step 400; peak memory 26.5 GB at device
  batch 16.
- Checkpoint 400 pushed, rotation kept the newest 2 (300, 400), raw + EMA exports with parity PASS (max rel diff
  2.9e-6 / 2.5e-6, top-1 agreement 1.000), DONE marker, logs uploaded, pod self-deleted. **PASSED.**
- Speed on L40S: ~88K tok/s (as expected; not a candidate for the main run). Cost ~$0.31 (17 min) [ESTIMATE].

## Totals and what changes in the main-run plan

- Dry runs ≈ **$2.24** (approved max $6.5).
- Main run: bf16 (no FP8); per-GPU speed 224K tok/s in TST and 291K after; memory 57.8 GB at device batch 32;
  the checkpoint, resume, export and self-delete paths are all proven.
