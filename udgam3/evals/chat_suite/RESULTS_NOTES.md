## Things the part 2b writer should know

1. **bf16 greedy decoding is not batch-invariant at this size.** For the same prompt tokens, a batched run and a
   one-at-a-time run of the same engine give identical first 512 tokens on only a minority of the 40 check items, and
   the first differing token is often within the first ~50 (validation table). In fp32 the HF engine's padded batches
   are identical to one-at-a-time runs (40/40), so this is rounding, not a padding or masking bug. The rerun with a
   shuffled request order measures what it does to scores: the share of GSM8K items whose correctness flips, and the
   accuracy change. Treat differences of that size between two models as noise, on top of the sampling CIs. Udgam-3
   must be measured with the same scripts, engine, dtype and batch settings.
2. **PREREGISTRATION.md 2.4.1 says "identical outputs required".** That holds in fp32 for the padded HF path, but not
   in bf16 for any engine, as above. The pre-registered precision is bf16 (2.1), so production rows stay bf16 and
   the check is reported as measured. Moving production to fp32 would need an amendment (section 10).
3. **Near-copy (mt_eval.py) also fires on short templated replies whose content changed** ("The first month is
   January" → "The last month is July" has a difflib ratio > 0.6). In the first models read, about half of the
   near-copies were correct second answers. "Right %" (on the 144 checkable items) is the more meaningful
   multi-turn measure. The extra column "near-copy and wrong" is a diagnostic, not pre-registered.
4. **Cut-off thinking still earns credit under the pre-registered rules.** math_eval.py's rule (and lm-eval's
   `think_end_token`) leaves an unfinished thinking block in the reply. So a reasoning model that finds the answer and
   then loops ("Thus answer: 9. Thus answer: 9. ...") until the cap is scored correct on GSM8K/MATH. On IFEval, its
   raw thinking text is checked against the instructions. Read by eye, the GSM8K/MATH cases are mostly real answers
   (the model got there, then failed to stop), but the IFEval cases are not answers at all. The tables therefore add
   "extra" columns: GSM8K/MATH **strict** (a reply cut off by the cap counts as wrong), IFEval **answered only** (no
   credit for an unfinished thinking block), multi-turn **right, answered only**. These matter mostly for
   Falcon-H1-Tiny-R and MobileLLM-R1.5. Part 2b should say which variant a claim uses; strict is the fairer
   comparison for a model meant to stop and answer.
5. **Reasoning models are cap-limited at 4,096.** MobileLLM-R1.5-140M and Falcon-H1-Tiny-R-90M think inside
   `<think>` and often run past the cap. Their secondary rows use 8,192 (MobileLLM's card) and 32,000 on the first
   300 items (Falcon-R; no cap is published for the 90M, and 32,000 is its authors' GRPO generation length).
6. **Short contexts cap some models below 4,096** (Supra2-100M, cagliostro, Surjo, Blaze: 2,048; Boris, Supra2-Medium:
   1,024; Udgam-384M-Chat: 4,096 including the prompt). Their effective caps are in the tables; their truncation rates
   come from the context, not the task cap.
7. **Contamination guard for Udgam-3's SFT:** drop any conversation containing one of the (q1, q2) pairs in
   `mt_heldout_v2_hashes.txt` (214 pairs; none found in the SFT data on disk on 2026-10-19, `mt_overlap.txt`), as
   well as the `heldout_v1_hashes.txt` prompts already listed in SFT_DATA_PLAN.md (the loop check uses those).
8. **Paired tests:** per-item scores for every model are in `scored/full/<model>/<task>__<mode>[__<var>].jsonl`
   (GSM8K/MATH `ok`, IFEval per-prompt and per-instruction booleans, E1 `copy`/`right`, E2 `one`/`two`, loop `loop`),
   keyed by item id, ready for McNemar / paired bootstrap against Udgam-3's run on the same items.
9. **IFEval scoring is made deterministic.** lm-eval's IFEval checks call `langdetect` (random unless seeded) and,
   for an instruction with an invalid argument, draw a random one (key 1122's "letter" is `#`, so a random letter is
   checked on every call). score.py seeds both (langdetect seed 0; `random.seed(key)` per item). Rescoring gives
   identical numbers; lm-eval's own runs can differ by about 0.1 point on these items.
