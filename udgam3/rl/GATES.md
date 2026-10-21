## r1: FAIL (2026-10-19 18:44)
- dev GSM8K thinking on >= SFT + 3.0: 8.20 vs bar 11.60 -> MISS
- dev GSM8K cap rate <= SFT's: 5.80 vs bar 3.00 -> MISS
- all dev numbers: {"gsm8k_think": {"n": 500, "cap_rate": 5.8, "loop_rate": 5.2, "mean_len": 213.9, "secs": 77, "acc": 8.2}, "math_think": {"n": 200, "cap_rate": 22.5, "loop_rate": 19.5, "mean_len": 387.7, "secs": 43, "acc": 5.0}, "gsm8k_tool": {"n": 200, "cap_rate": 2.0, "loop_rate": 1.5, "mean_len": 87.5, "secs": 34, "acc": 8.0, "tool_use": 98.5, "tool_errors": 33}, "gsm8k_plain": {"n": 200, "cap_rate": 1.0, "loop_rate": 1.0, "mean_len": 116.8, "secs": 31, "acc": 6.0}, "chat": {"n": 200, "cap_rate": 10.0, "loop_rate": 8.5, "mean_len": 222.8, "secs": 47}}

**Diagnosis (2026-10-19, after the R1 failure):**
- Sampled training accuracy was flat (4.4% / 4.8% / 3.4% / 4.5% over steps 1-10 / 11-20 / 21-30 / 31-40).
- The distillation gap to Gargi fell 1.52 → 1.34 nats per token, and the KL to SFT jumped 0.001 → 0.15 in the last
  third: the model moved, but toward Gargi's surface form, not toward right answers.
- Dev GSM8K 8.6 → 8.2 is churn, not change: 15 problems right in both, 28 only by SFT, 26 only by R1.
- Loops rose: GSM8K cap 3.0 → 5.8%, MATH loop 10.5 → 19.5%, dev chats cap 7 → 10%. R1 replies repeat lines.
- Third failed RL attempt at this scale (Gargi v1 +0.5, v2 +0.1, now −0.4).
- Per the plan: series stopped; R2/R3 not run; the SFT model remains the kept model.
