# Udgam-3 post-training data plan (v1, 2026-10-17)

**Goal (user):** the all-round ChatGPT/Claude experience, scaled down to 150M, built with a recipe that scales. The
user gave the training session full control over the post-training data ("make data that is very useful for
post-training a small model"). **Budget:** DeepSeek generation ≤ ~$8 (of the $150 cap; post-training total ~$18).
Teacher: `deepseek-flash` = DeepSeek-V4.1-Flash-1017 (MIT open weights) via the API; off-peak (all weekend) $0.15 /
$0.60 per 1M in/out tokens.

## What a 150M model learns best from (design rules)

1. **Short, clear, well-structured answers.** Small models imitate length and verbosity; long answers turn into
   rambling and loops. Default answer ≤ ~250 words, with markdown only when it helps (lists, a small table, code).
   Longer only when the task needs it (code, summaries of long texts).
2. **Short thinking.** Small students learn poorly from long expert traces (they need short, step-by-step reasoning).
   Thinking ≤ ~300 words for maths and logic; the final answer always restates the result plainly.
3. **The answer must follow the thinking.** Gargi's answers sometimes contradicted its own correct reasoning, so every
   thinking example is checked: the final answer must equal the reasoning's result and, where a gold answer exists,
   the gold answer.
4. **Real-user breadth.** Prompts come from real requests (WildChat-4.8M, oasst2, Chatbot Arena; licence-checked) plus
   generated prompts for gaps, stratified across everyday categories, not benchmark-shaped.
5. **Multi-turn by default.** ≥ 30% of conversations have 2-4 turns, where later turns depend on earlier ones
   (shorter, simpler, a correction, another language for code, a follow-up question). Gargi v1 had 0% and copied its
   previous reply.
6. **The thinking switch is taught explicitly.** Thinking-on examples for maths/logic/tricky questions; thinking-off
   for chat. Some prompts appear in both modes (the same answer, with and without visible thinking).
7. **Verifiable where possible.** Maths answers checked against gold or by executing code; instruction constraints
   checked by code; JSON parsed; code run. Unverifiable chat answers get an LLM quality screen and rule filters.
8. **No teacher fingerprints.** Drop or rewrite any mention of DeepSeek, other assistants, knowledge cutoffs of the
   teacher, or "As an AI language model". Identity data says the model is Udgam Labs' small model (name filled at
   release).
9. **Clean.** English only; dedup (8-gram); decontaminated against every eval set and the held-out chat prompts
   (`udgam3/evals/chat_prompts/heldout_v1_hashes.txt` + 8-gram overlap); no PII.

## Mix (targets; rows)

| Slice | Source | Rows | Notes |
|---|---|---|---|
| Everyday chat, thinking off | real-user prompts → DeepSeek answers | ~14,000 | explain, advice, factual QA (say "I'm not sure" when it's obscure), writing, rewriting, summarising, code help, brainstorming, casual; ~30% multi-turn |
| Instruction following | generated prompts with checkable constraints | ~4,000 | word/sentence limits, format (bullets, JSON, title), include/avoid keywords, case, language of code; verified by code |
| System prompts and personas | generated | ~1,500 | "Answer like a tutor", "Be brief", output format rules; teaches following a system prompt |
| Maths and logic, thinking on | factory verified problems + GSM8K/MATH train → DeepSeek short reasoning + answer | ~6,000 new + factory traces (~60K available) | answer checked; thinking ≤ ~300 words; same prompts also thinking-off |
| Python tool mode | factory programs (prob_gen, code_trace) + DeepSeek tool conversations | ~3,000 | real execution results in the conversation; no code inside JSON strings |
| JSON / function calls | generated tool schemas + calls | ~1,500 | parsed and validated against the schema; mock execution |
| Long documents (to 16K) | FinePDFs / Wikipedia long docs → summary + Q&A | ~1,500 | for the 16K claim; answers grounded in the text |
| Safety | generated harmful requests | ~600 | brief refusal + safe alternative; plus ~600 borderline-but-fine requests answered normally (avoid over-refusal) |
| Identity | templated | ~400 | name filled at release |
| Existing free data | Gargi SFT v4 (licence-checked), gen-v1 multi-turn (8,625), factory think/logic/code traces | subsample | balanced against the new slices on a dev score |

New DeepSeek rows ≈ 30K → ~10-12M output tokens + ~12M input ≈ **$7-9** off-peak [ESTIMATE].

## Process

1. Pilot: ~200 rows across all slices (≤ $0.10). Read 40 by eye; check lengths, cleanliness, verification pass rates.
2. Fix prompts and filters; then the full run with a hard spend stop (tracked from the API's token usage).
3. Render with the native Gemma-4 chat template (thinking via `enable_thinking`; tools via the template's tool
   format); record per-row provenance (source, licence, generator, verification status).
4. Choose final shares on a dev set (held-out from training; disjoint from the release evals).
