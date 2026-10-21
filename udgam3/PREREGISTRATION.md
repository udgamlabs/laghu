# Udgam-3 pre-registration

**Version:** part 1, v1.7, 2026-10-19 (training session). **Status:** [EXPLORATION] decided method; the claims (part 2)
are written after the Phase 0 rival measurements and **before any training of our model starts** (before Phase 1's
gate runs). After that, nothing here changes except through the amendment log (section 10), dated, with the reason,
and never after seeing the result it would affect.

**Why this exists.** Small-model launches lose trust through moving targets: picking the benchmark, the checkpoint or
the comparison after seeing results. This file fixes how we measure, whom we compare against, what "beats" means and
when we are allowed to release. Sources: `udgam3/HANDOFF.md` (mission, decided product), `udgam3/PLAN_REVIEW.md`
(changes adopted 2026-10-17).

---

## 1. The model (decided by the user)

- ~150M parameters total (tied embeddings counted once); planned shape d28 × 640, Gemma-4 layout, `tokenizer_v1`
  (32,768); **151.8M** by the model's own count [MEASURED].
- From scratch: random initialisation, our tokenizer, our data recipe. Open data and open-model-generated data are
  allowed within the licence policy (HANDOFF §14).
- English chat model; thinking on/off (native Gemma-4 `enable_thinking` switch); 16K context; Python tool mode
  (a feature, with no "best" claim against our own Lilavati).
- Pretraining at 2K context; the 16K stretch follows. Budget ≤ $150 all-in.
- **General and scalable recipe (user, 2026-10-17).** This is the recipe we will scale to bigger models. It aims at
  the all-round assistant experience people get from ChatGPT or Claude, scaled down: less knowledge, shorter
  context, same breadth. Data and post-training stay balanced across general-assistant skills and are never weighted
  toward a benchmark or leaderboard term. Only methods that work the same at larger scale are used.

---

## 2. Evaluation protocol

### 2.1 Software and hardware

- lm-evaluation-harness **0.4.13**, transformers **5.19.0**, torch **2.14.1**, accelerate 1.15.0, datasets 5.1.0
  (`evals/harness/uv.lock`).
- Generation evals: our scripts (2.3), each validated as in 2.4.
- **One device class per claim table.** The harness gave arc_easy 0.4979 on MPS vs 0.4958 on CUDA for the same model
  [MEASURED 2026-10-10/11]; every row of a comparison table is measured on the same device class, recorded per row.
- Udgam models: `add_bos_token=True` and a check that the first token id is `<bos>` (scores were badly understated
  without it). Rivals: their own tokenizer defaults and their own chat templates.
- Precision bf16 for every model unless a model's card requires fp32; recorded per row.

### 2.2 Base-model suite (lm-eval, zero-shot unless stated)

| Task | lm-eval task | Shots | Reported metric (both acc and acc_norm kept in raw files) |
|---|---|---|---|
| ARC-Easy | `arc_easy` | 0 | acc (Gargi's convention) and acc_norm |
| ARC-Challenge | `arc_challenge` | 0 | acc_norm and acc |
| HellaSwag | `hellaswag` | 0 | acc_norm and acc (they differ by 6-12 points; always say which) |
| PIQA | `piqa` | 0 | acc_norm and acc |
| WinoGrande | `winogrande` | 0 | acc |
| MMLU, letter format (MCF) | `mmlu` | 0 | acc |
| MMLU, cloze format (CF) | `mmlu_continuation` (or the 0.4.13 equivalent, confirmed in Phase 0) | 0 | acc_norm |
| MMLU-Redux | confirmed in Phase 0 (lm-eval task, or our scorer over `edinburgh-dawg/mmlu-redux-2.0`) | 0 | MCF and CF |
| GSM8K (base) | `gsm8k` | 5 | exact match, flexible-extract |
| Open SLM Index | HellaSwag, ARC-E, ARC-C, PIQA zero-shot + ArithMark-3 with the board's own script | 0 | the board's formula; both acc and acc_norm variants until the board's choice is confirmed |

### 2.3 Chat-model suite (greedy decoding unless stated; same output cap for every model)

| Skill | Script | Settings | Reported |
|---|---|---|---|
| Maths with thinking | `udgam2/eval/math_eval.py` (GSM8K test, 1,319) | own chat template; thinking on and off where a model has a switch; **cap 4,096 new tokens** for all models; answer = last number | accuracy; truncation rate; for reasoning models also at their recommended cap (secondary row) |
| MATH levels 1-3 | same script, MATH-500 subset | as above | accuracy |
| Arithmetic | ArithMark-3 if its script is public, else our own fixed set (part 2) | as above | accuracy by digit length |
| **General chat quality** | pairwise judging by an open judge (gpt-oss-120b, Apache-2.0) | a fixed, held-out set of ~400 real-user-style prompts across everyday categories (explain, advise, write, rewrite/summarise, code help, maths, brainstorm, factual QA, multi-turn), built in Phase 0 from a licence-clean prompt source and never used in training; each pair judged in both orders; length-controlled | win rate vs each rival chat model, with 95% CI; per-category win rates; 20 judgments checked by hand |
| Instruction following | lm-eval `ifeval` with `--apply_chat_template` | greedy, cap 1,280 | prompt-level strict and loose, instruction-level strict and loose |
| Multi-turn | `udgam2/multiturn/mt_eval.py` | same engine for every model; greedy | near-copy rate of reply 2 vs reply 1; follow-up correctness; two-turn GSM8K vs single-turn. The held-out set grows from 50 to **≥ 200 items** in Phase 0 (power) |
| Long context | `udgam2/ctx32k/hf_needle.py` + a multi-key needle + long-document QA | 4K / 8K / 12K / 16K; 100 trials per length | retrieval accuracy |
| Looping | `udgam2/multiturn/loop_check.py` | held-out prompts, thinking on and off, greedy **and** shipped sampling defaults | loop rate |
| Tool mode | Lilavati's scorer (code executed in the safe runner) on GSM8K / SVAMP / GSM-Plus | single try; voting reported separately | accuracy |
| Templates | release matrix | llama.cpp with and without `--jinja`, Ollama, MLX, transformers | pass/fail: thinking-off emits no reasoning text; tool calls parse; JSON-only outputs contain no thinking |

### 2.4 Scorer validation (before any score is trusted)

1. Every generation scorer is checked against an unbatched reference on ~40 items (identical outputs required).
2. Batched generation masks padding or batches identical lengths only.
3. Ten raw outputs per model per task are read by eye before the number is used.
4. Output caps, item counts and truncation rates are recorded for every model and every task.
5. Raw outputs for every number in a public table are saved under `udgam3/evals/`.

---

## 3. Rivals

**Size class:** ≤ 200M total parameters (embeddings included, tied counted once). **Open-weight:** weights
downloadable under any licence; the licence is shown in a column. Closed-weight models are mentioned, not compared.

| Model | Params | Licence | Notes |
|---|---|---|---|
| SmolLM2-135M / -Instruct | 135M | Apache-2.0 | the bar everyone cites |
| Falcon-H1-Tiny-90M-Base / -Instruct / R-90M / Tool-Calling-90M / Coder-90M | 91M | Falcon-LLM License | IFEval 66.08 claimed (Instruct) |
| MobileLLM-R1-140M-base / R1.5-140M | 140M | FAIR non-commercial | compare only, never a teacher |
| Surjo-100m / -50m | 98M / ~50M | Apache-2.0 | |
| Pythia-160M | 162M | Apache-2.0 | context |
| PleIAs Monad | 56M | Apache-2.0 | |
| GPT-X3-150M (AxiomicLabs) | 149M | Apache-tagged | Open SLM #1 |
| cagliostro-v3.5 / -chat (Bench Labs) | 146M | as published | |
| Boris-2-Preview-1 (OpenCerebral) | 144M | as published | |
| GPT-X3, Supra2-100M-Instruct, JugnuLM-110M-R2plus, lille-130m | 100-150M | as published | indie models from Lilavati's comparison |
| LilM1-230M, Needle 3 (121M) | | as published | tool-use rivals (Needle 3 does not chat) |
| **Gargi v2** (ours, 384M), **Udgam-384M** (ours), **Lilavati** (ours, 95M, tool mode) | | Apache-2.0 | our own bars |
| Near-size references (reported separately): Gemma-3-270m / -it, LFM2.5-230M, SmolLM2-360M, MobileLLM-R1-360M | 230-362M | various | not counted as size-class rivals |
| Mentioned, not compared: Madhuram-v1 (MaruthLabs, 151M, closed weights) | | | |

The list is final for part 2 once Phase 0 has found every model that loads in our harness; any model that can't be
run is listed with the reason. A fresh scan in release week can add rivals but never remove one.

---

## 4. Statistics

- **Per-item pairing.** Every comparison uses the same items for both models.
  - Binary per-item scores (accuracy): exact McNemar test.
  - Averages, composites and rates (e.g. the Index, near-copy rate): paired bootstrap over items, 10,000 resamples.
- **Words we may use:**
  - **"beats"**: our score is higher and p < 0.05 after Holm correction across the pre-registered claim set
    (part 2);
  - **"matches"**: the difference is not significant;
  - **"trails"**: theirs is significantly higher.
- 95% confidence intervals next to every headline number.
- **One main run.** The budget allows one main pretraining run; its seed-to-seed spread is unknown and is disclosed.
  Gates use seeds as defined in section 6.
- **Fixed checkpoint rule.** Base model: the final EMA weights of the main run (EMA is pre-declared). Chat model: the
  final checkpoint of the last post-training stage, unless a dev-set selection rule is written into part 2 first.
  Dev sets are disjoint from every test set.

---

## 5. Categories and the release gate

- **Contested category:** at least 3 size-class rivals measured on it. Only contested categories count toward
  release. Uncontested capabilities (Python tool mode, 16K retrieval) are reported, not counted as wins.
- **Release gate (all must hold):**
  1. A fresh scan in release week (HF, web, X via Monid) finds no newer ≤ 200M open model that beats us on the
     claimed categories.
  2. We **beat** every measured size-class rival on at least **3 contested categories** from part 2.
  3. The bar against our own models holds (part 2 names the categories and the Gargi v2 / Udgam-384M rows).
  4. Every public number has exact scope (size class, benchmark, metric, thinking on/off, tool or not, greedy or
     voting, harness, date) and a raw-output file in `udgam3/evals/`.
  5. The template matrix (2.3) passes.
- If the gate fails we don't release. We write down what would create the edge and ask the user.

---

## 6. Gate decision rules (fixed before the gates run)

### 6.1 G1: stable data mix

- **Proxy:** d16 × 512, tied, 2K context (64.5M parameters [MEASURED]); equal FLOPs per arm (~1.5B tokens).
- **Arms:** D0 (Gargi's v2 mix) with seeds 1, 2, 3; D1 (v3, ~22% structured rephrase); D2 (v3, ~45%). D4 (100%
  DCLM-baseline, the open-recipe anchor) only if the research report is in scope.
- **Same ending for every arm:** the last 20% of steps decays the LR on one identical anneal mix, so arms differ only
  in the stable mix.
- **Order:** D0 seed 1 first, as the learnability check (it must learn held-out data and decode sensibly); then the
  rest, one at a time until D0 has learned.
- **Primary metric:** mean correct-answer probability (per-character normalised, × 100) over the **validation**
  splits of ARC-E, ARC-C, HellaSwag, PIQA, WinoGrande and MMLU. Test splits are never used for gate decisions.
- **Rule:** let *s* be the standard deviation of the three D0 seeds on the primary metric. An arm wins if its primary
  metric exceeds the D0 mean by more than **max(2·√2·*s*, 0.5)** and its HellaSwag + PIQA average is not below the D0
  mean by more than 2·√2 times their own seed SD. Among winners, the highest primary metric is chosen. If none wins,
  keep D1.
- Every arm, winners and losers, is logged.

### 6.1b G2: technique gate (added v1.2; user: "use every technology that works at this size")

- **Proxy and data:** the G1 proxy shape on the G1-winning mix, equal training FLOPs per arm, same ending anneal.
- **Baseline:** the G1 winner (its D0 seed spread *s* sets the noise band).
- **Arms (one variable each):**
  - **T1 document masking:** attention restricted to the current document inside packed rows (standard at large
    scale, e.g. Llama 3; the speedrun uses it).
  - **T2 partial loop:** the middle third of the layers runs twice, same stored parameters; at equal FLOPs it sees
    fewer tokens. This is the right comparison for "best at this size on a fixed budget". Our earlier tests found
    loops worse at small scale (Gargi Gate A: full loop +1.81%, partial +0.88% bpb; Udgam-Loop: no gain over dense);
    this re-tests it on our new data and shape because the user wants it used if it works.
- **Rule:** adopt an arm only if it beats the baseline's primary metric by more than max(2·√2·*s*, 0.5) with no
  HellaSwag + PIQA regression beyond noise. For T2 also: shipping a looped model through GGUF/MLX/browser runtimes
  means exporting it unrolled (about 2× file size), so a T2 win goes back to the user before adoption.
- **Measured, not gated:** FP8 training speed on the main GPU type (dry run). Adopt only if ≥ 1.2× tokens/s and a
  short A/B shows no held-out loss penalty.

### 6.2 G3: multiple-choice dose (micro-anneals from the main run)

- **Branch point:** the main run's checkpoint at the start of its anneal phase.
- **Arms:** 0%, 8% and 15% lettered MCQ in the anneal mix, ~0.5B tokens each, identical otherwise.
- **Metrics (validation splits):** MMLU-MCF and ARC-MCF; regression checks on cloze HellaSwag, PIQA and ARC.
- **Rule:** choose the smallest dose that lifts MMLU-MCF by ≥ 3 points over the 0% arm, with no regression > 1 point
  on any regression check. If none qualifies, the anneal uses 0% MCQ and MMLU is reported in cloze format only (not
  claimed).

### 6.3 16K stretch

- Training length: the stretch trains at up to 32K (as Gargi, whose 32K training gave 100% recall at 16K); we claim
  16K only.
- Pass: single-needle retrieval ≥ 95% at 4K, 8K, 12K and 16K, and the base suite average (validation splits) drops by
  ≤ 1 point. Otherwise stop and fix before post-training.

### 6.4 Post-training data shares

- Shares are set on dev sets disjoint from the tests: 500 held-out GSM8K *train* problems; generated IFEval-style
  constraints not overlapping IFEval prompts; multi-turn seeds not in the held-out set.

---

## 7. Efficiency multiplier *k* (research claim)

- **Primary measure:** bits per byte on neutral held-out text (tokenizer-independent), measured on our stable-phase
  checkpoint **and** on micro-annealed branches.
- **References:** DataDecide **final** checkpoints across sizes (DCLM-baseline and the best DataDecide recipe, "DCLM-Baseline
  (QC 7%, FW3)"), pulled to `udgam3/evals/datadecide/` [EXTERNAL]. Finals reach ~100 tokens per non-embedding parameter
  at 60-150M and ~89-92 at 300-530M; **the 750M "final" stops at ~45 and is not used as a curve point**. DataDecide's
  `compute` is 6 × non-embedding params × tokens, so our FLOPs are converted to the same convention before
  interpolating; Pythia and OLMo-2 early checkpoints as context only (not annealed).
- **Statement form:** "our model at C training FLOPs matches the reference curve's value at k·C", interpolated in
  log-compute across the reference sizes; FLOPs from each model's own count, never from model names (DataDecide's
  "150M" counts non-embedding parameters).
- Benchmark-based *k* is reported separately and labelled: our anneal contains lettered MCQs and instruction-shaped
  text, and the compute used to generate synthetic data is not counted.
- The neutral held-out text is fixed in Phase 0, before any of our checkpoints exist.

---

## 8. Contamination

- 13-gram overlap of every pretraining source with every eval set (including IFEval prompts, MMLU-Redux, GSM-Plus and
  the long-context templates); 8-gram filter on SFT and anneal data.
- The generated MCQ slice also gets a semantic near-duplicate check against MMLU, MMLU-Redux and ARC test items
  (embedding similarity, plus a manual sample).
- Public tables report overlap rates and scores on the clean subsets.

---

## 9. Part 2: claims and thresholds (to be written after Phase 0, before Phase 1)

### Part 2a: base-suite categories (written 2026-10-19 ~07:00 IST, BEFORE Udgam-3's base suite was run)

Measured by us: 26 models, CUDA (Colab G4), `udgam3/evals/phase0/BASE_RESULTS_CUDA.md` and `bars_2a.json`.
Known at writing time about Udgam-3: only its held-out bpb (same-script table, 2026-10-19), no benchmark accuracy.

| Category | Metric and setting | Bar (best ≤ 200M rival, measured by us) | Our threshold | Counts toward the gate? |
|---|---|---|---|---|
| ARC-Easy | `arc_easy` acc, 0-shot | 64.5 (SmolLM2-135M) | none | **No: reported only** |
| ARC-Challenge | `arc_challenge` acc_norm, 0-shot | 31.4 (GPT-X3-150M) | none | **No: reported only** |
| HellaSwag | `hellaswag` acc_norm, 0-shot | 45.9 (GPT-X3-150M) | none | **No: reported only** |
| PIQA | `piqa` acc_norm, 0-shot | 71.3 (GPT-X3-150M) | none | **No: reported only** |
| WinoGrande | `winogrande` acc, 0-shot | 53.6 (GPT-X3-150M) | none | **No: reported only** |
| MMLU (letter) | `mmlu` acc, 0-shot | 33.1 (Falcon-H1-Tiny-90M-Instruct) | none | **No: reported only** |
| MMLU (cloze) | `mmlu_continuation` acc, 0-shot | 29.9 (SmolLM2-135M) | none | **No: reported only** |
| GSM8K (base) | `gsm8k` 5-shot, flexible-extract | 19.7 (MobileLLM-R1-140M, FAIR non-commercial) | none | **No: reported only** |

**Why no base claims.** Section 9 allows base commonsense claims "only if Phase 0 shows a reachable bar". Our own
Gargi-Base (384M, 2.5× Udgam-3's size) averages 51.7 on the first five tasks against GPT-X3-150M's 53.0. It is below
the best size-class rival on HellaSwag, PIQA and ARC-Easy, and within noise on the rest. A 152M model on 40B tokens is
not expected to clear these bars, so none is claimed. All base results are still published in full, with these bars
next to them, including where we trail.

### Part 2b: chat categories (written 2026-10-19 18:59 IST, BEFORE any Laghu chat model is measured on the chat suite)

**Known at writing time about Laghu (Udgam-3) chat models:** only dev numbers, never a chat-suite item.
- SFT pick `laghu-sft-v1-ep2` by the dev rule (`udgam3/sft/devpick/`): 60.0% pairwise vs 1 pass; 7.0% on 200 GSM8K-*train* problems.
- RL dry run and R1 (`udgam3/rl/`): dev GSM8K-train 8.6% (SFT) vs 8.2% (R1, gate failed).
- So the maths claim below is expected to fail. It is kept because v1.7 fixed the three candidates before this was known.

**Model measured:** `laghu-sft-v1-ep2` (`YOUR_HF_NAMESPACE/laghu-sft` @ c8c36d1). It is the only chat model, because RL R1 failed its gate (checkpoint rule, section 4: the dev rule in `udgam3/sft/devpick/DEV_RULE.md` was written before measuring; its dev sets are disjoint from every test set here).
- Same scripts, engine (vLLM 0.31.0, hash-locked env), dtype (bf16), caps and batch settings as the rivals (`udgam3/evals/chat_suite/`, RESULTS_METHODS.md). The run includes the 40-item one-at-a-time check (2.4).

**Laghu's mode per category is fixed now** (no best-of picking):
- maths: thinking **on**;
- multi-turn, chat quality, loop rate: thinking **off** (its default for chat).
- The other mode is reported, never claimed. Rivals use their best mode as in TABLES_full.md (all are "native" except Gargi).

**Measured rivals (v1.7):** SmolLM2-135M-Instruct, Falcon-H1-Tiny-90M-Instruct, MobileLLM-R1.5-140M. Own bar: Gargi (and Udgam-384M-Chat where measured).

| Category | Metric and setting | Rivals measured by us (best mode) | Our threshold | Counts toward the gate? |
|---|---|---|---|---|
| **Maths with thinking** | GSM8K test (1,319), greedy, cap 4,096, **strict** (a reply cut off by the cap is wrong) | Falcon-90M 23.3 [21.1, 25.6]; SmolLM2 1.8; MobileLLM-R1.5 0.0 (100% truncated) | **beats** each of the three | yes |
| **Multi-turn** | E1 **right %** on the 144 checkable two-turn items (mt_eval rules, cap 1,024 per turn) | Falcon-90M 54.9 [46.7, 62.8]; SmolLM2 37.5; MobileLLM-R1.5 3.5 | **beats** each of the three | yes |
| **General chat quality** | pairwise, **length-controlled win rate** vs each rival on the 324 held-out single-turn prompts (`items/loop.jsonl`), greedy replies at cap 1,024 (the loop-task outputs), visible reply only, both orders | win rate 50% = parity | **beats** each of the three: LC win rate > 50% | yes |

**Reported only (never a claim or a bar):**
- MATH-500 L1-3 strict (rivals: Falcon-90M 23.9, SmolLM2 0.4, MobileLLM-R1.5 0.0);
- GSM8K non-strict;
- near-copy % and E2 (GSM8K as the second turn);
- IFEval;
- the other thinking mode;
- loop rate with shipped sampling.

**Statistics (section 4):**
- **"Beats"** = higher, with p < 0.05 after Holm correction across the 9 rival comparisons (3 categories × 3 rivals).
- **GSM8K and multi-turn:** exact McNemar on paired items (per-item files in `scored/full/`).
- **Chat quality:** paired bootstrap over prompts (10,000) of the LC win rate; two-sided p for "≠ 50%".
- **A category counts** only if Laghu beats all three rivals in it.

**Length control (fixed now):** AlpacaEval-2 style.
- Each judgment (Laghu = A after re-ordering) is fitted with a logistic model `P(Laghu preferred) = sigmoid(a + b · tanh(Δlen / σ))`.
  - Δlen = Laghu's visible reply length minus the rival's, in characters;
  - σ = the standard deviation of Δlen over the set;
  - a tie counts as half a win and half a loss.
- LC win rate = sigmoid(a), i.e. the win rate at equal length. The raw win rate is reported beside it.

**Judge (amended 2026-10-19 19:04 IST, before measuring; user-approved):** DeepSeek-V4.1-Flash (`deepseek-flash`, MIT open weights).
- Why: our HF token cannot call Inference Providers for the pre-registered gpt-oss-120b (403, 2026-10-19).
- Settings: thinking off, temperature 0, JSON mode, spend cap $3, the dev-pick judge's instructions (judge helpfulness, correctness, instruction following and clarity; ignore length; JSON verdict A / B / tie).
- The same judge model ran the SFT dev pick, on different prompts (SFT validation chats), never on these 324.
- Both orders per pair. An unparseable verdict is retried once, then counted as a tie.
- 20 judgments are read by hand. Judge position bias is reported.

**Own-model bar (gate item 3):**
- Laghu does not **trail** Gargi or Udgam-384M-Chat on multi-turn right %.
- Laghu does not **trail** Gargi on chat quality (LC win rate vs Gargi not significantly below 50%).
- Maths vs Gargi (35.6) is reported.

**Loop rate (fixed now):** greedy loop rate on the 324 prompts, thinking off, must be **≤ 10%**. Above that, no release until fixed.

**Contamination (checked 2026-10-19):**
- **Held-out chat prompts** (484, incl. the 324): 0 overlap with SFT training user turns (exact text and 10-word windows).
- **Multi-turn items:** no (q1, q2) pair inside one SFT training conversation. 3 generic turns appear separately ("What is the capital of France?" / "and of Germany?", Brazil / Argentina, "Tell me a riddle." / "what's the answer?").
- **GSM8K test and MATH-500:** in the v3 13-gram decontamination index used for all SFT data.

**Release gate consequence:** v1.7 has exactly three candidates, so gate item 2 ("≥ 3 contested categories") requires all three. Per section 5, if it fails we don't release. We write down what would create the edge and ask the user.

---|---|---|---|---|
| *pending* | | | | |

Candidates (PLAN_REVIEW §5.1): **general chat quality (judged, pairwise)**, maths with thinking (GSM8K, MATH L1-3),
multi-turn chat; instruction following and base commonsense only if Phase 0 shows a reachable bar. Candidates are
picked from where a balanced general model can win; the data is not tilted toward any of them. Part 2 also fixes: the Gargi v2 / Udgam-384M
categories for the own-model bar, the loop-rate threshold, and any dev-set checkpoint rule.

---

## 10. Amendment log

| Date | Change | Reason | Before or after the affected result? |
|---|---|---|---|
| 2026-10-21 | **Founder override of the release gate (section 5):** the gate failed (0 of 3, CLAIMS_2B.md), and the founder chose a **fully open research release** with no "beats"/best-in-class claim. All part-2b results are published as measured, including where Laghu trails | founder decision (HQ `registry/decisions.md` 2026-10-21) | after the part-2b results (no result or claim is changed) |
| 2026-10-19 | **v1.8 judge amendment (user-approved 19:04 IST)**: the part-2b chat-quality judge is DeepSeek-V4.1-Flash (thinking off, temperature 0) instead of gpt-oss-120b | our HF token cannot call Inference Providers (403); the user chose to amend rather than add a token | before any Laghu chat-suite measurement |
| 2026-10-19 | **Part 2b written** (chat claims; before any Laghu chat-suite result): maths with thinking = GSM8K strict, multi-turn = E1 right %, chat quality = length-controlled pairwise win rate on the 324 held-out prompts; Laghu's mode per category fixed; Holm over 9 comparisons; loop threshold ≤ 10%; own-model bar; judge gpt-oss-120b as pre-registered | rival chat suite measured; Laghu SFT chosen, RL R1 failed | before any Laghu chat-suite measurement (only dev numbers known, listed in 2b) |
| 2026-10-19 | v1.7 (user-approved): **chat suite narrowed.** Part 2b claim candidates are exactly three categories: **multi-turn**, **general chat quality (pairwise judged)** and **maths with thinking**. Each is measured by us (same harness, same items, CUDA) on **SmolLM2-135M-Instruct, Falcon-H1-Tiny-90M-Instruct and MobileLLM-R1.5-140M** (the size-class rivals; 3 per category, so each stays contested under section 5) plus Gargi (own bar). All other rivals and categories (e.g. IFEval) use the authors' **reported** numbers, labelled "as reported", for context only; they never count as a bar or a win | time and cost: the full chat suite on every rival was too slow | before any Udgam-3 chat model exists |
| 2026-10-19 | Part 2a written (base suite: all reported, none claimed; reason above). Rival table: 26 models on CUDA | the rival base suite finished | before Udgam-3's base suite was run |
| 2026-10-19 | v1.6 (user-approved 02:10 IST): (1) every comparison table is measured on **CUDA, Colab G4 (RTX PRO 6000 Blackwell)**, bf16, instead of the Mac (MPS needed 1-2 days for 16 models); one device class per table still holds, and no MPS row is mixed into a CUDA table. (2) **Part 2 is written in two stages:** 2a (base-suite claims) after the rivals' base suite and **before Udgam-3's base suite is run**; 2b (chat claims) after the rivals' chat suite and **before any Udgam-3 chat model is measured**. Phase 0 was not finished before the main run (only a partial MPS SmolLM2-135M run existed), so part 2 could not be written before training as planned; this is disclosed. Known before writing 2a: Udgam-3's held-out bpb on fresh/web/reasoning text (same-script table, 2026-10-19), not any benchmark accuracy | speed (Mac too slow) and honesty about the order of work | before any Udgam-3 benchmark or chat result |
| 2026-10-17 | Part 1 v1 written | — | before any training |
| 2026-10-17 | v1.5: G1/G2 outcome recorded (`udgam3/evals/g1/G1_RESULTS.md`). The per-character correct-probability metric compressed all arms to 35.68-35.76 (seeds identical), so the fixed 1.0 floor was unreachable: no arm won, and the pre-registered default (D1 = v3) applies; held-out web bpb supports it. Future gates: summed-log-likelihood softmax or accuracy with paired tests, plus held-out bpb, with noise floors in the metric's own units | metric design error found when the results arrived | after the G1 results (the decision follows the pre-registered default; nothing was re-picked) |
| 2026-10-17 | v1.4: §7 corrected with the pulled DataDecide data (tokens/param per size, 750M not fully trained, FLOPs convention); neutral held-out text fixed as `udgam3/evals/neutral_text/neutral_v1.jsonl` (4.0 MB, Wikipedia pages created 2026-09-01..10-15 + arXiv abstracts 2026-10-01..15) | data pulled | before any of our checkpoints exist |
| 2026-10-17 | v1.3 (one-week release): G1 = D0 × **2** seeds + D1 + D2 (+ T2 loop), run in parallel after D0 seed 1 shows learning; G1 primary metric drops MMLU (noise at 65M; per-char-normalised softmax probability of the gold choice on the validation splits of ARC-E, ARC-C, HellaSwag, PIQA, WinoGrande; `udgam3/runs/g1/primary_metric.py`); with 2 seeds, *s* = |seed1 − seed2| / √2 and the fixed floor rises to **1.0** point; **G3 dropped** (anneal uses existing ARC/OBQA-style task data, no generated MCQ slice; MMLU reported, not claimed); **T1 document masking deferred** to the next model (needs new attention code) | user: release within one week | before any gate run |
| 2026-10-17 | v1.2: G2 technique gate (document masking, partial loop) and FP8 speed check | user: use every technology that works at this size, including looping if it can | before any training |
| 2026-10-17 | v1.1: general-and-scalable rule (section 1); general chat quality added to the chat suite (2.3) and to the part 2 candidates; "arithmetic" removed as a candidate | user steer: the recipe must stay general and scalable | before any training |
