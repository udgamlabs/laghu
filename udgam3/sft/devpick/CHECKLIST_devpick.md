# Pre-launch checklist: Laghu SFT dev pick (1 pass vs 2 passes), 2026-10-19

**Exact config:** the rule in `DEV_RULE.md` (frozen in git before any reply is generated).
- Colab G4, new session `devpick`, run by `devpick_run.py` → `devpick_vm.sh` → `devpick_gen.py`; pinned harness env
  (`evals/harness/uv.lock`); models `YOUR_HF_NAMESPACE/laghu-sft` @ c8c36d1 (`laghu-sft-v1-ep1/hf`, `laghu-sft-v1-ep2/hf`).
- Dev sets `dev_chat.jsonl` (200) + `dev_gsm8k.jsonl` (200), sha256 in `DEV_SETS.sha256` (checked on the VM).
- Judge: `judge.py` on the Mac, DeepSeek-V4.1-Flash, thinking off, temperature 0, both orders, **spend cap $0.50**.
- **Cost:** ~2-3 Colab units (~15-20 min; hard stop at 45 min ≈ 6.7 units) of 84.85; judge ~$0.10-0.15 (off-peak).
  Udgam-3 dollar total ≈ $198.9 → ≈ $199.1.

| # | Check | Status |
|---|---|---|
| 1 | Storage | outputs are text (~5 MB), kept in git; no checkpoints written. ✅ |
| 2 | Memory | 152M model, batches capped at 262K prompt+output tokens and 64 rows; the SFT run trained at 58.6 GiB with 4 × 16K rows. ✅ |
| 3 | Dry run | generation script run on the Mac (CPU, 3 items, tiny caps): templates, left padding, stops, outputs. Scorer + judge path tested (`--dry`) and one live judge call (valid JSON, $0.00003). ✅ |
| 4 | End-to-end save | outputs tarred, downloaded, session stopped and token removed in a `finally`. ✅ |
| 5 | Price and capacity | Colab subscription; DeepSeek off-peak prices; cap $0.50 enforced by the client. ✅ |
| 6 | Time | ~15-20 min [ESTIMATE]; hard stop 45 min. ✅ |
| 7 | Budget | as above. ✅ |
| 8 | Learnability | n/a (no training). |
| 9 | Frozen copies | rule, dev sets, scripts committed before the run. ✅ |
| 10 | End-of-run steps in the exact env | generation in the pinned harness env; scoring on the Mac with the chat suite's `common.py`. ✅ |
| 11 | Scorer validated | the chat suite's GSM8K rule; batched vs unbatched check on 20 items per model (≥ 18 agree). ✅ |
| 12 | Approval = exact config | this file + `DEV_RULE.md`. |
| 14 | One run at a time | nothing else is running. ✅ |
| 18 | Decision metric validated | pairwise judge: tested on a trivial pair; position bias and 20 judgments read by hand are reported. GSM8K: the chat suite's validated rule. ✅ |
| 19 | Objective and success test | pick between 1 and 2 passes by `DEV_RULE.md`; no test benchmark touched. ✅ |
| 22 | Monitors survive stalls | shared Colab CLI lock; exec retries; 45-min hard stop. ✅ |

**Leakage found while building it:** 293 of the 787 eligible SFT val conversations share most of their last user
message with a training row (maths thinking twins, long documents asked several questions, multi-turn rows on one
seed). They are excluded here; the SFT held-out loss is partly leaky and should not be quoted as held-out.
