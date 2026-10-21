# Udgam-3 main pretraining run: plan (draft for the user's approval; nothing launched)

**Status:** [EXPLORATION] draft v2, 2026-10-18 01:50 IST, after the dry runs. The user approves the final, exact config
(token count + GPU option); `main.conf` holds the recommended one.

## 1. Objective and success test

**Objective:** train the Udgam-3 base model from scratch, with the recipe and data the gates chose, to the most tokens
the budget allows, and save it safely for the 16K/32K stretch and post-training:
- 150M parameters, d28 × 640, tied embeddings, Gemma-4 layout;
- data: v3 pool, 42.2B stable + 10.3B cool-down tokens;
- recipe: Muon, TST s=4 for the first 25%, cool-down data from 80%, WSD warmdown 0.65, EMA from 90%.

**Success test (checked during and after the run):**
1. Held-out loss falls throughout, with no NaN and no spike above +5% between consecutive evals:
   - common sets: web, reason, task;
   - **our fresh-text set** (Wikipedia Sep-Oct 2026 + arXiv Oct 2026), which no model can have trained on.
2. Final fresh-text bpb of the 2K base (EMA or raw, whichever is lower; same scorer as `OWN_BENCHMARKS.md`):
   - **must** be below SmolLM2-135M's 0.941 (the ≤ 200M bar);
   - **target**: Gargi-Base's 0.914 after our 32K stretch [ESTIMATE for 40B: 0.900-0.912].
3. Every scheduled checkpoint lands on HF (`YOUR_HF_NAMESPACE/laghu-base-ckpt`) and is verified. The final raw and EMA
   weights export to HF format with logits parity.
4. Cost ≤ the approved cap; finishes before the pod deadline.

## 2. Dry runs: done, all passed (2026-10-18, ≈ $2.24; `DRY_RUNS.md`)

- A (bf16): data, env, preflight, checkpoints, export parity, samples, DONE + self-delete all work.
- A-fp8: FP8 0.73-0.75× slower → **bf16**.
- B: exact resume on another GPU type, including the cool-down data stream.

## 3. Tokens, cost and expected quality [MEASURED speed, ESTIMATE projection]

- Speed per H100 SXM: **224K tok/s in TST (first 25% of steps), 291K after** → 8 GPUs ≈ 1.70M / 2.21M tok/s at 95%
  scaling (checked in the first minutes by the watcher).
- The data pool has 42.2B stable + 10.3B cool-down tokens. TST reads 4 tokens per position, so a run of D positions
  reads 1.55 D stable tokens: above ~27B positions some stable text is read twice (harmless up to ~4 epochs,
  Muennighoff et al. 2023).
- Projection (`token_projection.py`): two published scaling-law fits, used only as ratios, anchored on Gargi-Base
  (same lineage; 384M, 8.9B positions + 32K stretch, fresh bpb 0.914). The size term reproduces our measured
  SmolLM2-360M vs -135M gap (−9.7%). Read the bpb as "after our own 32K stretch", like Gargi-Base.

| Positions | Stable epochs | Projected fresh bpb (two fits) | 8×H100 Community | 8×H100 Secure | 4×H100 Secure |
|---|---|---|---|---|---|
| 26B | 0.96 | 0.918-0.927 | 3.8 h, ~$81 | ~$120 | 7.3 h, ~$116 |
| 30B | 1.10 | 0.912-0.922 | 4.3 h, ~$92 | ~$137 | 8.3 h, ~$133 |
| 35B | 1.29 | 0.905-0.916 | 5.0 h, ~$107 | ~$159 | 9.7 h, ~$155 |
| **40B (recommended)** | 1.47 | **0.900-0.912** | **5.6 h, ~$121** | ~$180 | 11.0 h, ~$176 |
| 50B | 1.84 | 0.892-0.905 | 7.0 h, ~$151 | ~$223 | 13.7 h, ~$219 |

Bars on the same fresh-text set: SmolLM2-135M 0.941, Udgam-384M 0.937, **Gargi 0.914**, SmolLM2-360M 0.850.
40B is the smallest option whose whole projected range reaches Gargi's level at 40% of its size (goal S8). Beyond
it, each extra 10B buys only ~0.006.

- Availability (01:30 IST): no 8×H100 node on either cloud; 4× Secure "Low". Stock changes by the hour.
- Hard maximum = the pod deadline (MAX_MIN = training ETA × 1.25 + 35 min): 40B on 8× Community → 440 min →
  ≤ $158.

## 4. Budget (Udgam-3; spent ≈ $51.2 at 01:35 IST 2026-10-18)

| Item | Plan |
|---|---|
| Main run | user's choice above (40B Community ≈ $121) |
| 16K stretch (trained to 32K) | ~$6-8 |
| SFT + RL GPU time | ~$6-8 (data already paid: $9.49) |
| Rival evals + final evals (GPU pod) | ~$5-6 |
| **Total with 40B on Community** | **≈ $190** (above the original $150; user, 2026-10-18: stretch for tokens if needed) |

Runpod balance $204.75, shared with the voice project (its RTX 5090 pod is running) and ~$50 reserved for other
projects. Community 40B fits; any Secure option needs a top-up first.

## 5. Stop rules (in code: `watch_main.py`, 8 offline tests passed on the dry-run log; it alerts, I stop pods by recorded id only)

- **Start:** no SSH or pod start in 5 min; env/CUDA gate fails; data download fails; preflight upload fails → the pod
  stops itself (built into `ops/pod_train.sh`).
- **First 10 minutes of training:** node tok/s below 80% of the expectation for its phase (TST 224K, normal 291K per
  GPU × 95%), or the ETA + 15 min of export runs past the deadline → stop and report. Never wait and hope.
- **Every eval (every 2,000 steps):** held-out loss NaN, or up > 5% vs the previous eval on two sets → stop and report.
- **Every checkpoint:** the log must report each upload, and HF `LATEST` is checked every 30 min. Two missed save
  points → stop and investigate.
- **Crash or failure marker** in the log → report; resume from the newest HF checkpoint (path proven in dry run B).
- **Deadline:** an absolute watchdog deletes the pod at creation + MAX_MIN (training ETA × 1.25 + 35 min). That cap
  is the maximum cost; at worst it loses the last ≤ 8.5 min since a checkpoint. Otherwise the pod self-deletes
  only after the final push is verified on HF.

## 6. Exact config (recommended option; frozen in `main.conf`)

`udgam3/runs/main/common.sh` + `main.conf`:
- shape `--depth 28 --n-embd 640 --n-head 10 --n-kv-head 2 --g4-head-dim 64 --g4-global-head-dim 128
  --intermediate-size 1792 --ple-dim 0 --window-pattern SSSL --sliding-window 512 --max-seq-len 2048
  --tie-embeddings`;
- `--device-batch-size 32 --total-batch-size 524288` (no gradient accumulation on 8 GPUs);
- `--num-iterations 76294` (40.0B positions; another option changes only this and MAX_MIN);
- `--warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 --ema-start-frac 0.9 --ema-beta 0.998 --seed 42
  --tst-s 4 --tst-frac 0.25 --phase-b-data-dir .../phaseB --phase-b-start-frac 0.8`;
- `--eval-every 2000`, extra-val web/task/reason + fresh;
- `SAVE_EVERY=2000` (newest 2 kept on HF);
- bf16 (FP8 failed its adoption rule);
- 8× H100 SXM Community, MAX_MIN 440, disk 400 GB, all v3 shards.

Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (cached widely; Gargi's main run used it). Code:
nanochat `udgam3` @ `557c4c9` via `train-udgam3/`, pinned to HF revision `ef821d67` (`CODE_REV`).
