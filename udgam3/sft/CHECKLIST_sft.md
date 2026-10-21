# Pre-launch checklist: Laghu (Udgam-3) fine-tuning (SFT), 2026-10-19

**Exact config for approval:** Colab G4 (1× RTX PRO 6000 Blackwell 96 GB), a new session `sft`.
- Pipeline `udgam3/sft/colab_sft_pipeline.sh`, run by `udgam3/sft/colab_sft_run.py series 1,2,3 --new --session sft` (one at a time, a dashboard per run, session stopped at the end).
- Training code `YOUR_HF_NAMESPACE/laghu-build-code` @ **571e433** = the main run's code (ef821d6) plus one file:
  `scripts/udgam_sft.py` re-ties the shared embedding after loading and re-draws the 12 never-trained chat-token rows
  (ids 5-16) from N(mean, cov) of the trained rows.
- Start: the 32K-stretch final weights `YOUR_HF_NAMESPACE/laghu-ctx32k/udgam3-ctx32k/model_001900.pt` (stretch accepted
  by the user with its reasoning-bpb miss recorded).
- Data `YOUR_HF_NAMESPACE/laghu-sft-data` @ 11dd41c, `v1-laghu/` (41,467 train / 853 val conversations, model name **Laghu**).
- **Three runs, one at a time: 1, 2 and 3 passes over the mix** (identity rows 2×, i.e. 2 / 4 / 6), seq 16,384,
  best-fit packing, 262,144 tokens per step, device batch 4; LRs matrix 0.0071 / embedding 0.106 / unembedding 0.0028 /
  scalar 0.177, 5% warmup, 50% warmdown to 0, no weight decay; checkpoints at 50% and 100%.
- **Pick rule (written before the runs):** lowest held-out loss `all` on the 853-conversation val set; if two are within
  0.01, take fewer passes. The chosen run must also pass the chat smoke test (says it is Laghu by Udgam Labs; correct turn
  format; thinking on/off works).
- Output: private `YOUR_HF_NAMESPACE/laghu-sft/<run>/{hf,ckpt,logs}`; HF export with `max_position_embeddings` 16,384 (the
  claim), export verify must PASS.
- **Cost: $0 extra; ~7 Colab units for all three runs** (~45 min × 8.9/h) of the 91.5 left; the dollar total stays ≈ $198.9.

| # | Check | Status |
|---|---|---|
| 1 | Storage | udgamlabs (Team) private; 3 runs × (0.3 GB export + 1.8 GB checkpoint) ≈ 6.5 GB. ✅ |
| 2 | Memory at the real shape and batch | **58.6 GiB peak** of 96 at seq 16,384, device batch 4 (dry run). ✅ |
| 3 | Dry run of the exact pipeline | 10 steps, 2026-10-19 10:29-10:32 UTC: env, code 571e433, preflight upload, data, re-tie + chat-token re-draw (12 rows from 32,639 trained rows), training, held-out loss, export verify PASS (1.8e-5), chat smoke, upload, token removed. ✅ The series driver + per-run dashboard: dry series 2 × 10 steps on a new session `sft`, 10:34-10:45 UTC: new session, start, a dashboard per run (pushes OK), SFT_DONE twice, token removed, session stopped (1.5 units). Run 1's dashboard missed its final "done" (a stalled read); fixed with a retried final push. ✅ |
| 4 | End-to-end save | export + checkpoint + logs + smoke uploaded before `SFT_DONE`; token removed at the end. ✅ |
| 5 | Price and capacity | Colab G4, subscription; 91.5 units left. ✅ |
| 6 | Time | **99K tok/s** (2.65 s per 262K-token step): 1 pass ≈ 107 steps ≈ 5 min, 2 ≈ 215 ≈ 10 min, 3 ≈ 323 ≈ 15 min; with setup, export and smoke ≈ 45 min for all three. ✅ |
| 7 | Budget | $0 extra. ✅ |
| 8 | Learnability | continued training from a trained base. In 10 steps held-out loss fell 2.160 → 1.878 (identity 2.08 → 1.09); the 10-step model already uses the turn format and answers 7 × 12 = 84 inside a thought block, but still says "Yes, I am ChatGPT" (identity not learned yet at 10 steps, as expected). ✅ |
| 9 | Frozen copies | pipeline committed; code pinned by revision (571e433); data pinned (11dd41c). ✅ |
| 10 | End-of-run steps in the exact env | export + chat smoke ran in the dry run's env. ✅ |
| 11 | Scorer validated | val loss = token-level mean over assistant tokens (the script Gargi's SFT used); the smoke test is greedy decoding, read by hand. ✅ |
| 12 | Approval = exact config | this file. |
| 13 | Progress from held-out numbers | held-out loss per source every 20 steps on the dashboard. ✅ |
| 14 | One run at a time | the three runs are sequential; nothing else trains. ✅ |
| 16 | Results repo accepts uploads | preflight 64 MB upload to the private repo passed in the dry run. ✅ |
| 17 | Speed guard | the dashboard shows tok/s and ETA per run; the series driver stops a run with no end after 3 h. ✅ |
| 18 | Decision metric validated | held-out loss moves with training (step 0 → 10 in the dry run: 2.160 → 1.878). ✅ |
| 19 | Objective and success test | a chat model that (a) has the lowest held-out loss of the three runs, (b) passes the smoke test. Its chat-suite scores are measured AFTER part 2b claims are written (pre-registration). |
| 22 | Monitors survive stalls | one shared Colab CLI lock; short polls. ✅ |

**APPROVED by the user 2026-10-19 16:18 IST:** "start SFT exactly as in udgam3/sft/CHECKLIST_sft.md: 3 runs (1, 2, 3 passes) on Colab G4, ~7 units, pick by held-out loss plus the smoke test."
