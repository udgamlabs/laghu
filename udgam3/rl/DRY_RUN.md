# Laghu RL dry run (2026-10-19, Colab G4, $0, 1.7 units incl. a first attempt that failed on a path bug)

Exact pipeline `colab_rl_pipeline.sh dry` @ git 0f8c636 + the stage code added after it (not run). Everything uploaded to
private `YOUR_HF_NAMESPACE/laghu-rl` (`dry/`); token removed; session stopped.

## Baseline dev evals (greedy, 1,024 new tokens; same env as every later stage)
| Dev set | Laghu SFT (`laghu-sft-v1-ep2`) | Gargi (teacher) |
|---|---|---|
| GSM8K-train 500, thinking on | **8.6%** (cap 3.0%, loop 3.0%, 174 tokens) | 74.2%* (287 tokens) |
| MATH-train L1-3 200, thinking on | **4.0%** (cap 12.0%, loop 10.5%) | 42.0%* (490 tokens) |
| GSM8K 200, Python tool | **10.5%** (tool used 100%; 42 tool errors) | — |
| GSM8K 200, thinking off | 5.0% | — |
| 200 dev chats | cap 7.0%, loop 7.0% | — |

*Gargi's SFT and RL used GSM8K-train and MATH-train problems, so these are inflated by memorisation (its GSM8K
**test** score is 35.6%). For a teacher that is fine: it knows these problems. Laghu never trains on the dev problems.

## Two full-shape steps per stage type (32 problems x 8 samples, 768 new tokens)
| | Maths (GRPO + Gargi distillation) | Tool (GRPO) |
|---|---|---|
| Seconds per step | 39.7, 38.9 (generation 34) | 38.4, 34.0 |
| Problem groups with a correctness signal | 47% | 41%, 16% |
| Sampled accuracy (T = 1.0) | 3.5%, 3.9% | 7.0%, 2.3% |
| Distillation gap (student - teacher log-prob per token) | 1.56 → 1.49 | — |
| KL to SFT | 0.0007 | 0.0006 |
| Cap rate | 0.4%, 0% | 0.8% |
| Tool errors per 256 answers | — | 136, 155 (sampled code often breaks the sandbox rules) |
GPU memory stayed well under 96 GB (no OOM; the 3 models + teacher fit).

Short eval of the 4-step result (20 items each, not a measurement): GSM8K think 10%, MATH 0%, tool 15%, chats cap 15%.
