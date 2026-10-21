# Laghu SFT dev pick: results

Rule: `DEV_RULE.md` (written before measuring). Judge spend $0.066.

| | 1 pass | 2 passes | Rule |
|---|---|---|---|
| Pairwise win rate of 2 passes vs 1 pass (200 dev conversations, both orders) | | **60.0%** (95% CI 54.9-65.0) | ≥ 55%: PASS |
| GSM8K-train accuracy (200, thinking on) | 4.0% | 7.0% | ≥ 1-pass − 2.0: PASS |
| Scorer check: batched = unbatched verdicts (of 20) | 20/20 | 20/20 | ≥ 18 each: PASS |

**Pick: 2 passes (laghu-sft-v1-ep2).**

Not deciding (reported): per-source win rate of 2 passes: chat 63% (n=51), chat_thinking 86% (n=7), gen_mt 54% (n=88), instruction_following 63% (n=13), json_function_calls 57% (n=11), long_documents 44% (n=4), maths_logic 81% (n=4), safety 75% (n=11), system_prompts 61% (n=11).
Judge position bias: first reply preferred in 42% of verdicts. Lengths and caps: {"ep1": {"chat": {"hit_cap": 27, "mean_new": 249}, "gsm": {"hit_cap": 15, "mean_new": 281}, "chat_visible_chars": 801}, "ep2": {"chat": {"hit_cap": 11, "mean_new": 177}, "gsm": {"hit_cap": 6, "mean_new": 205}, "chat_visible_chars": 549}}.

**Hand check (20 random conversations, both orders, read 2026-10-19):** the judge's reasons match the replies; it
penalises loops, wrong facts and ignored constraints, and gives ties when both are nonsense (e.g. a Spanish question
both models answered wrongly). The 2-pass replies are shorter (549 vs 801 visible characters) and loop less (11 vs 27
chat replies hit the cap), so the win is not a length effect. Both models are often wrong or incoherent on open
questions, as expected at 150M.

**Concern for the claims (not part of this rule):** GSM8K-train accuracy with thinking is only 4.0% / 7.0%. The replies
use the thought format but the arithmetic reasoning is wrong (e.g. "5 × 7 = 35 hours" for 10 h × 5 days), and they
ignore the "Final answer:" line (the last-number rule still scores them). Size-class rivals score 19-25% on GSM8K test
and Gargi 35.6%, so "maths with thinking" is not a realistic part-2b claim for the SFT model; RL on maths is the lever.

**Cost:** 1.5 Colab units (84.85 → 83.36), judge $0.066 (400 calls, 0 errors).

**Decision applied: Laghu SFT = `laghu-sft-v1-ep2` (2 passes)** (`YOUR_HF_NAMESPACE/laghu-sft` @ c8c36d1, `laghu-sft-v1-ep2/hf`).
