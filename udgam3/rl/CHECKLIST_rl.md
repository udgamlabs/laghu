# Pre-launch checklist: Laghu RL, stages R1 → R2 → R3 (2026-10-19)

**Exact config for approval** (option C of `RL_PLAN.md`):
- Colab G4, one new session `rl` per stage; `rl_series.py r1,r2,r3` runs the stages one at a time and **stops at the
  first failed gate** (gates in `RL_PLAN.md`, checked by `rl_series.py` against the dry run's SFT baseline).
- Code: `udgam3/rl/` at the commit of this file (laghu_rl.py, rollout.py, sandbox.py, rl_eval.py,
  colab_rl_pipeline.sh, colab_rl_run.py); pinned harness env (transformers 5.19.0, torch 2.14.1).
- Start: SFT `YOUR_HF_NAMESPACE/laghu-sft` @ c8c36d1 `laghu-sft-v1-ep2`; teacher `udgamlabs/Gargi` @ 7e0e6dd (public, same
  tokenizer and template); KL reference = the SFT model in every stage.
- Data: `pool_maths.jsonl` (7,709 problems, none in SFT), dev sets `dev_gsm8k.jsonl` / `dev_math.jsonl` + the 200 dev
  chats; hashes in `SETS.sha256`, checked on the VM.
- **R1:** 40 steps from SFT, maths (thinking on 85% / off 15%), GRPO + distillation weight 1.0. **R2:** 200 more steps
  from R1 (next prompts in the pool). **R3:** 100 steps from R2, Python tool, GRPO only.
- All stages: 32 problems x 8 samples, T 1.0, 768 new tokens, AdamW 5e-6 (fp32 weights), grad clip 1.0, KL 0.02,
  rewards as in `RL_PLAN.md`; snapshots every 50 steps + final; every stage ends with the full dev evals.
- Output: private `YOUR_HF_NAMESPACE/laghu-rl/{r1,r2,r3}/{hf,eval,logs,snapshots}`.
- **Cost: $0 extra. ~37 Colab units [ESTIMATE from the dry run's 39 s/step]** (R1 ~5, R2 ~21, R3 ~10) of 81.7 left;
  hard stops per stage: 60 / 240 / 110 min (max 61 units).

| # | Check | Status |
|---|---|---|
| 1 | Storage | udgamlabs (Team) private; per stage 0.6 GB export + 2-4 snapshots ≈ 3 GB total. ✅ |
| 2 | Memory at the real shape | dry run: 2 full-shape steps of each stage type with policy + sampler + SFT reference + teacher on the G4, no OOM. ✅ |
| 3 | Dry run of the exact pipeline | `DRY_RUN.md`: env, hashes, preflight upload, downloads, baseline evals, maths and tool training at full shape, eval, uploads, token removed, session stopped. The R1-R3 branches add only downloads of the previous stage and the same commands. ✅ |
| 4 | End-to-end save | each stage uploads hf + evals + logs + snapshots before `RL_DONE`; the driver removes the token and stops the session in a `finally`. ✅ |
| 5 | Price and capacity | Colab subscription; the session is created fresh per stage. ✅ |
| 6 | Time | 39 s/step measured: R1 ~36 min, R2 ~2.4 h, R3 ~1.2 h incl. setup and evals [ESTIMATE; longer replies would slow R2]. ✅ |
| 7 | Budget | $0 extra; units as above. ✅ |
| 8 | Learnability | R1 IS the learnability gate: 40 steps must raise dev GSM8K by ≥ 3 points (~2.3 standard errors at n = 500) before R2 spends 21 units. Dry run: the trainer moves the model (distillation gap 1.56 → 1.49 in 2 steps). ✅ |
| 9 | Frozen copies | code, sets and this file committed; models and teacher pinned by revision. ✅ |
| 10 | End-of-run steps in the exact env | rl_eval ran on the VM in the dry run (SFT, Gargi, the trained result). ✅ |
| 11 | Scorer validated | the chat suite's GSM8K rule (validated for the chat suite); greedy single-sequence-per-row batches with left padding (the dev-pick check found batched = unbatched on 20/20). ✅ |
| 12 | Approval = exact config | this file + `RL_PLAN.md`. Any change (option, LR, steps, data) means asking again. |
| 13 | Progress from held-out numbers | stage decisions only from the frozen dev sets; the dashboard shows reward, sampled accuracy, cap/loop rates, distillation gap, KL. ✅ |
| 14 | One run at a time | stages are sequential; nothing else runs. ✅ |
| 16 | Results repo accepts uploads | preflight upload to the private repo at the start of every stage. ✅ |
| 17 | Speed guard | per-stage hard stop; the dashboard shows seconds per step and the ETA. ✅ |
| 18 | Decision metric validated | the dev GSM8K score separates models (SFT 8.6% vs Gargi 74.2%) and is measured identically before and after. ✅ |
| 19 | Objective and success test | better maths with thinking and a working tool mode on dev sets, without chat regression; the final pairwise / loop / base-suite check runs after the kept stage (its own small checklist, ~2 units + ~$0.07). |
| 22 | Monitors survive stalls | shared Colab CLI lock, exec retries, hard stops. ✅ |

**APPROVED by the user 2026-10-19 18:13 IST:** "option C (GRPO + Gargi distillation), run R1 -> R2 -> R3 exactly as in udgam3/rl/CHECKLIST_rl.md (~37 Colab units, hard stops 60/240/110 min), stopping at the first failed gate."
