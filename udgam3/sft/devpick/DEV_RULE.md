# Laghu SFT: dev rule for 1 pass vs 2 passes (written 2026-10-19, before any of it is measured)

**Why:** held-out loss picks 1 pass (1.724 vs 1.784), but 2 passes reads better in the smoke test
(`../SERIES_RESULTS.md`). The user approved a generation-based tie-breaker on dev data (2026-10-19). Pre-registration
4 lets a dev rule choose the checkpoint if it is written before measuring; no test benchmark (GSM8K test, MATH-500,
IFEval, the chat suite's held-out prompts) is used.

**Candidates** (private `YOUR_HF_NAMESPACE/laghu-sft`, HF exports): `laghu-sft-v1-ep1/hf` (1 pass) and `laghu-sft-v1-ep2/hf`
(2 passes). The 3-pass run is out (held-out loss 2.036 and worse answers).

## Dev sets (frozen by `build_dev.py`, hashes in `DEV_SETS.sha256`)
1. **Chat (200 conversations)** from the SFT **val** split (853 conversations never trained on), stratified by source
   in proportion to what remains, seed 1234. Left out: identity rows (trivial), Python-tool rows (they need a live
   tool loop), and **val conversations whose last user message mostly appears in a training row** (≥ 50% of its
   10-word windows): 293 of 787. Finding: render_v1 split by id, so a maths question's thinking-on and thinking-off
   rows, a long document's several questions, and multi-turn rows on the same seed can sit on both sides of the
   split. The SFT held-out loss (853 conversations) is therefore partly leaky; this dev set is not. Each item = the conversation up to its last user message (earlier assistant turns as in the
   data, their reasoning removed), with the row's own thinking switch and tool schemas.
2. **Maths (200 problems)** from GSM8K's **train** split (`openai/gsm8k`, config `main`, revision 740312a), seed 1234,
   **excluding every problem that appears in the SFT mix** (38 skipped while sampling; any shared 10-word window),
   with the chat suite's prompt suffix ("Solve it step by step, then give the final answer on the last line as:
   Final answer: <number>"), thinking on.

## Generation (Colab G4, one session, both models, identical settings)
- Pinned harness env (`evals/harness/uv.lock`: transformers 5.19.0, torch 2.14.1), bf16, each model's own chat
  template rendered as text and tokenized without extra special tokens, greedy.
- Stop at `<turn|>` or `<eos>`. Caps: chat 1,024 new tokens with thinking off, 2,048 with thinking on; maths 2,048.
- Left padding with an attention mask, prompts batched by similar length. **Scorer check:** the first 20 maths items
  are also generated one at a time; the batched and unbatched GSM8K verdicts must agree on ≥ 18 of 20 per model
  (bf16 greedy is not batch-invariant, so small text differences are allowed).

## Scoring
- **Visible reply** = output with the thought channel removed and special tokens dropped (`chat_suite/common.py`,
  `visible`); a thought block cut off by the cap stays in the reply.
- **Maths:** `common.gsm8k_ok` (the number after the last "Final answer", else the last number) vs the gold after `####`.
- **Pairwise judge:** DeepSeek-V4.1-Flash (`deepseek-flash`, thinking off, temperature 0), each pair judged in both
  orders (A/B swapped). The judge sees the conversation and the two visible replies and returns JSON
  `{"verdict": "A" | "B" | "tie"}` with a one-sentence reason; it is told to judge helpfulness, correctness, instruction
  following and clarity, and not to prefer a reply for being longer. Per conversation, the 2-pass score = mean over the
  two orders of (win 1, tie 0.5, loss 0); unparseable verdicts count as ties. Spend cap $0.50.

## Decision (fixed now)
**Pick 2 passes if BOTH:**
1. its pairwise win rate vs 1 pass is **≥ 55%** (mean over the 200 conversations), and
2. its GSM8K-train accuracy is **no more than 2.0 points below** 1 pass's.

**Otherwise pick 1 pass.** Reported but not deciding: 95% bootstrap CI of the win rate, per-source win rates, mean
reply lengths, truncation and loop counts, and 20 judgments read by hand.
