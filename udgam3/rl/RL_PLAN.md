# Laghu RL plan v1 (2026-10-19)

**Status:** [EXPLORATION] draft for the user. Built and tested at $0: prompt pools, frozen dev sets, sandboxed Python
tool, the trainer (GRPO + distillation + tool rollouts), dev evals, the Colab pipeline and a dry run
(`DRY_RUN.md`). **No RL stage runs before the user approves its checklist** (`CHECKLIST_rl.md`).

**Starting point:** the SFT model `YOUR_HF_NAMESPACE/laghu-sft` @ c8c36d1 `laghu-sft-v1-ep2` (2 passes, picked by the dev rule).
It chats, follows the thinking switch and calls the Python tool, but it is weak at maths: 7% on GSM8K-train problems with
thinking on (`../sft/devpick/DEV_RESULTS.md`).

## What RL does here (first principles)
1. Give the model a maths problem. It writes 8 different answers (sampling at temperature 1).
2. A checker scores each answer: +1 if the final number is right; minus points for looping, running past the length
   cap, or breaking the thinking format.
3. Within the 8, answers better than their siblings' average are pushed up (every token in them becomes a little more
   likely); worse ones are pushed down. This is GRPO. If all 8 score the same, that problem teaches nothing.
4. **Teacher (distillation):** Gargi (our 384M model, same tokenizer and chat template, 35.6% on GSM8K test) reads
   each of Laghu's answers and says, token by token, how likely *it* would have written that token. Laghu is nudged
   toward Gargi's choices on every token, even when all 8 answers are wrong, so there is signal on every problem.
5. **Leash:** a small penalty (KL) keeps Laghu close to its SFT self so chat, identity and format don't drift.
6. Repeat for a few hundred steps; check the frozen dev sets at the end of each stage.

## What we learned before (Gargi and Udgam-384M; learnings log)
- GRPO moved Gargi's GSM8K by only +0.5 and +0.1 points in two runs (format improved, accuracy didn't). Causes we
  identified: a learning rate so small that bf16 weights barely changed, prompts that were mostly all-right or
  all-wrong (no signal), and short runs.
- At Laghu's 7%, about half the problem groups would be all-wrong, so plain GRPO starves. That is why the teacher term
  is the core of this plan.
- More verified teacher solutions (rejection sampling) gave Gargi +5.8 GSM8K: the fallback if RL stalls.

## Options for the maths stage (decision for the user)
| Option | What | For | Against |
|---|---|---|---|
| A. GRPO only | correctness reward + penalties, as Gargi but with fp32 weights, LR 5e-6 and skipped no-signal groups | simplest; the claim stays "RL from verifiable rewards" | failed twice at this size; little signal at 7% |
| B. Distillation only | Gargi's per-token judgement, no correctness reward | dense signal on every problem; [EXTERNAL] on-policy distillation matched RL with ~10× less compute (Thinking Machines, 2025) | copies Gargi's mistakes too; capped by Gargi (35.6%) |
| **C. Both (recommended)** | correctness reward (GRPO) + Gargi's per-token term + KL leash + penalties | correctness keeps it honest, Gargi gives signal when all 8 fail | two knobs to set (we fix them now, see below) |

The tool stage (R3) uses GRPO only: Gargi never learned the Python tool.

## Stages, gates and decision rules (written before any measurement)
All gates use the frozen dev sets (`dev_gsm8k.jsonl` 500 GSM8K-train problems, `dev_math.jsonl` 200 MATH-train L1-3,
none in SFT or the RL pool; the 200 SFT dev-pick chats). No test benchmark is touched; the chat-suite measurement of
the final model comes only after part 2b claims are written.

| Stage | What | Steps | Pass rule (else stop and bring options) |
|---|---|---|---|
| R1 learnability gate | maths, option C, thinking on 85% / off 15% | 40 | dev GSM8K (thinking on, greedy) **≥ SFT + 3.0 points** and dev cap rate not higher than SFT's |
| R2 maths main | continue from R1, next prompts in the pool | +200 | dev GSM8K ≥ R1's; dev MATH not lower than SFT − 2 |
| R3 tool mode | Python tool, executed answers, GRPO only, from the kept maths model | 100 | dev tool-mode GSM8K **≥ the R3 start + 5 points**, and dev GSM8K thinking-on not lower than the R3 start − 1 |
| Final regression check | the kept model vs SFT | — | pairwise judge on the 200 dev chats: **win rate ≥ 45%** (not clearly worse); dev chat loop rate ≤ SFT + 1 point; base suite (validation splits) average drop ≤ 1 point |

Checkpoint rule (pre-registration 4): the final checkpoint of each stage; a stage that fails its gate is dropped and the
previous model is kept.

## Rewards (`rollout.reward`)
| Part | Value |
|---|---|
| Correct final number (chat suite's GSM8K rule on the visible answer) | +1 |
| No proper stop (cap reached, loop, stray tool hand-over) | −0.5 |
| Periodic loop tail (chat suite's loop test) | −0.25 more |
| Overlong: linear over the last 192 tokens of the 768-token cap | up to −0.5 |
| Thinking format (on without a closed thought, or a thought with thinking off) | −0.2 |
| Malformed tool call | −0.2 |

## Settings
32 problems × 8 samples per step, temperature 1.0, 768 new tokens; AdamW (fp32 weights) LR 5e-6, gradient clip 1.0;
KL 0.02 to the SFT model; distillation weight 1.0 (maths stages only). Pool: 7,709 problems (5,776 GSM8K + 1,933 MATH
L1-3 numeric), none in SFT; R1 + R2 use 7,680 of them without repeats; R3 reuses the pool in tool mode.

## Cost and time
Measured in the dry run: 39 s per step (256 answers). R1 ~36 min (~5 units), R2 ~2.4 h (~21), R3 ~1.2 h (~10): ~37 of
the 81.7 units left, $0 extra. Hard stops per stage (60 / 240 / 110 min) cap it at 61 units.

## Baselines from the dry run (`DRY_RUN.md`)
Laghu SFT: dev GSM8K 8.6% (thinking on), MATH 4.0%, tool mode 10.5%, thinking off 5.0%. Gargi: 74.2% / 42.0% on the same
dev problems (inflated: Gargi trained on GSM8K/MATH train problems; its GSM8K test is 35.6%). Gargi is a far stronger
teacher on exactly this kind of problem, which supports option C.

## Risks
- Distillation can copy Gargi's style (longer thoughts) or errors; the correctness reward and the KL leash limit that.
- Reward hacking: "Final answer" text is not rewarded by itself; only the right number is.
- Chat drift: RL sees only maths/tool prompts; the KL leash and the final pairwise check guard it.
- Disclosure: the card must say Laghu's maths RL used Gargi (our own model) as a teacher.
