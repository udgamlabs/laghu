# Pre-launch checklist: Udgam-3 32K stretch, full run (2026-10-19 03:20 IST)

**Exact config for approval:** Colab G4 (1× RTX PRO 6000 Blackwell 96 GB), session `ctx32k`.
- Pipeline `udgam3/runs/ctx32k/colab_ctx32k_pipeline.sh` @ git 7a1b199.
- Training code `YOUR_HF_NAMESPACE/laghu-build-code` @ ef821d6 (the main run's).
- Data `YOUR_HF_NAMESPACE/laghu-ctx32k-data` @ bc9ee02.
- Start: `udgam3-base-ckpt/udgam3-base-v1/model_ema_076294.pt`.
- **1,900 steps × 524,288 tokens = 996M tokens**, seq 32,768, device batch 2 (16 rows per step).
- LR: matrix 0.002 / embedding 0.03 / unembedding 0.0008 / scalar 0.05; warmup 30 steps, then linear to 0. No weight decay.
- Checkpoints every 250 steps to private `YOUR_HF_NAMESPACE/laghu-ctx32k`.
- **Cost: $0 extra; ~36 Colab compute units** (~4 h × 8.9/h) of the 192 left.

| # | Check | Status |
|---|---|---|
| 1 | Storage | HF personal private 816 / 1,000 GB; this run adds ~12 GB of checkpoints plus a 0.3 GB export. ✅ |
| 2 | Memory at the real shape and batch | **56 GB peak** of 96 GB, measured in the dry run at 32K, device batch 2. ✅ |
| 3 | Dry run of the exact pipeline | 20 steps, 2026-10-19 02:50-03:09 IST: env, code, preflight upload, data, training, checkpoint upload, needle and long-doc evals, final push. ✅ The export failed on an env bug (transformers missing). Fixed (explicit `--python`, import check) and re-tested on the dry checkpoint: export verify PASS (1.6e-5), **16K long parity PASS** (4.0e-5, top-1 100%). |
| 4 | End-to-end save | final push + logs + evals + HF export uploaded before `CTX_DONE`. The token is deleted at the end. ✅ |
| 5 | Price and capacity | Colab G4 is included in the plan; a second session ran fine alongside the eval session. ✅ |
| 6 | Time | **75.8K tok/s → 3.65 h** of training, plus ~15 min of evals and export. Colab sessions last up to 24 h. ✅ |
| 7 | Budget | $0 extra; ~36 units. ✅ |
| 8 | Learnability | continued training from a trained base. In the dry run the loss fell 2.74 → 2.65 in 20 steps; needle at 1-2K is 100% before and after. ✅ |
| 9 | Frozen copies | pipeline committed (7a1b199) and uploaded at start; the code is pinned by revision. ✅ |
| 10 | End-of-run steps in the exact env | export + 16K parity run in that env (item 3). ✅ |
| 11 | Scorer validated | needle/copy/long-doc scorer = Gargi's (`eval_long.py`), which separated Gargi's stages (63% → 99% at 16K). ✅ |
| 12 | Approval = exact config | this file. |
| 13 | Progress from held-out numbers | the dashboard shows 2K held-out bpb every 250 steps and needle results at the end. ✅ |
| 14 | One run at a time | the only training run. The rival benchmarks run separately on CUDA (evaluation only). ✅ |
| 16 | Results repo accepts uploads | the preflight 256 MB upload passed in the dry run. ✅ |
| 17 | Speed guard | the dashboard and watch flag a step rate below 60K tok/s or no new step for 20 min. |
| 18 | Decision metric validated | needle: base and dry run give 100% at ≤ 2K and 0% beyond (the metric moves with training in Gargi's runs). ✅ |
| 19 | Objective and success test | **needle ≥ 95% at 4K / 8K / 12K / 16K**; base benchmarks (validation splits) drop ≤ 1 point vs the main-run EMA; 2K held-out bpb rises ≤ 1%. If any fails: **stop and bring options** (no automatic top-up). |
| 21 | Multi-GPU data check | single GPU; the val shard has 144 row groups anyway. ✅ |
| 22 | Monitors survive stalls | one shared Colab CLI lock (no concurrent calls), short polls, stall alert at 20 min. ✅ |

**Risk:** if the Colab session is reclaimed, the pipeline resumes from the last checkpoint on HF (at most 250 steps ≈ 29 min lost) when it is restarted with the same command.
