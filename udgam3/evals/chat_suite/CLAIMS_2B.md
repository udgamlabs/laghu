# Part 2b claims: results (PREREGISTRATION.md part 2b)

Model: `laghu`; judge: deepseek ; Holm over 9 comparisons.

| Category | Rival | Laghu | Rival | Paired n | p | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|
| Maths with thinking (GSM8K strict) | smollm2-135m-instruct | 7.9 | 1.8 | 1319 | 3.06e-14 | 1.53e-13 | **beats** |
| Maths with thinking (GSM8K strict) | falcon-h1-tiny-90m-instruct | 7.9 | 23.3 | 1319 | 1.13e-31 | 7.89e-31 | trails |
| Maths with thinking (GSM8K strict) | mobilellm-r1.5-140m | 7.9 | 0.0 | 1319 | 9.86e-32 | 7.89e-31 | **beats** |
| Multi-turn (E1 right %) | smollm2-135m-instruct | 41.7 | 37.5 | 144 | 0.461 | 1 | matches (not significant) |
| Multi-turn (E1 right %) | falcon-h1-tiny-90m-instruct | 41.7 | 54.9 | 144 | 0.0094 | 0.0376 | trails |
| Multi-turn (E1 right %) | mobilellm-r1.5-140m | 41.7 | 3.5 | 144 | 6.14e-15 | 3.69e-14 | **beats** |
| General chat quality (LC win rate) | smollm2-135m-instruct | 51.0 [46.5, 55.5] (raw 53.2) | 50.0 | 648 | 0.68 | 1 | matches (not significant) |
| General chat quality (LC win rate) | falcon-h1-tiny-90m-instruct | 28.7 [24.0, 33.6] (raw 34.6) | 50.0 | 648 | 0 | 0 | trails |
| General chat quality (LC win rate) | mobilellm-r1.5-140m | 57.3 [27.0, 76.2] (raw 91.0) | 50.0 | 648 | 0.632 | 1 | matches (not significant) |

**Categories won (beats all three rivals):** none.

## Own-model bar
| Check | Laghu | Ours | p | Holds? |
|---|---|---|---|---|
| Multi-turn right % vs Gargi (off) | 41.7 | 49.3 | 0.117 | yes |
| Multi-turn right % vs Udgam-384M-Chat (on) | 41.7 | 40.3 | 0.89 | yes |
| Chat quality LC win rate vs Gargi | 45.9 | 50.0 | 0.06 | yes |
| Maths (GSM8K strict) vs Gargi (on) [reported] | 7.9 | 35.6 | 4.31e-79 | reported |

**Loop rate** (greedy, thinking off, 324 prompts): 8.3% (threshold ≤ 10%): OK.

## Release gate (section 5)
- Item 2 (beat every rival on ≥ 3 contested categories): **FAIL** (0 of 3).
- Item 3 (own-model bar): **PASS**.
- Loop threshold: **PASS**.
- Items 1, 4, 5 (fresh scan, exact scope, template matrix): not run yet.

**Gate: FAILS: no release under the pre-registered rule; write down what would create the edge and ask the user.**
