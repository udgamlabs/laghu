## How these numbers were made

**What was measured:** every size-class rival chat model we could find, plus our two existing chat models as bars and three
near-size references, on the pre-registered Udgam-3 chat suite (PREREGISTRATION.md 2.3). No Udgam-3 model was run.

**Hardware and software (one device class for every row, 2.1):** Colab G4 (NVIDIA RTX PRO 6000 Blackwell Server
Edition, 96 GB), bf16 for every model.
- **vLLM engine** (13 models): vLLM 0.31.0, torch 2.13.0+cu130, transformers 5.17.0, hash-locked in `env/vllm.lock`.
  vLLM's PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), since FlashInfer's needs a compiler the VM lacks; greedy is argmax
  either way.
- **HF transformers engine** (cagliostro-v3.5-chat, whose architecture vLLM doesn't have, plus every fp32 reference):
  the pinned harness, `evals/harness/uv.lock` (transformers 5.19.0, torch 2.14.1, lm-eval 0.4.13), installed with
  `uv sync --frozen`. Left padding with an attention mask, prompts batched by similar length.
- Each model is pinned to a commit sha (`manifest.json`). Items are fixed files built from pinned dataset revisions
  (`items/ITEMS.json`: GSM8K test 1,319; MATH-500 L1-3 238; IFEval 541; multi-turn 214; loop prompts 324).

**Prompting, identical for every model:** each model's own chat template (rendered as text, then tokenized with
`add_special_tokens=False`, as lm-eval does). No system prompt is added; templates that insert their own default
(SmolLM2, MobileLLM-R1.5, Falcon-H1-Tiny-R) keep it. Thinking: "off"/"on" = the template's `enable_thinking` switch
(Gargi, Udgam-384M-Chat); every other model has no switch ("native"). MobileLLM-R1.5 and Falcon-H1-Tiny-R always think
(`<think> ... </think>`). Stop tokens: the generation_config eos ids, the tokenizer eos, and the template's own end-of-turn
token (found by rendering an assistant turn). Boris-125M-Instruct's Alpaca template has no end-of-turn token, so it
also stops at the text `### Instruction:`. Output cap = min(task cap, context length − prompt length), recorded per
item; Supra2-100M (2,048), cagliostro (2,048), Boris (1,024), Supra2-Medium (1,024) and Udgam-384M-Chat (4,096) have
short contexts, so their effective caps are lower (shown in the tables).

**Scoring (score.py, fixed before any Udgam-3 chat model exists):**
- Visible reply = output with the thinking block removed (Gemma-4 thought channel as in `udgam2/eval/math_eval.py`;
  `</think>` as lm-eval's `think_end_token`) and special tokens dropped. A thinking block cut off by the cap stays in the
  reply, as in both of those.
- GSM8K: `math_eval.py`'s prompt suffix and rule (the number after the last "Final answer", else the last number).
- MATH-500 L1-3: the same suffix with `<answer>`. Math-Verify 0.9.0 checks the text after the last "Final answer" (also
  tried wrapped in `$...$`), else the whole reply (its last `\boxed{}` / expression). When the gold is a plain number,
  the GSM8K last-number rule also counts.
- IFEval: lm-eval 0.4.13's own scorer (`lm_eval.tasks.ifeval.utils.process_results`) on the visible reply, prompt
  and instruction level, strict and loose.
- Multi-turn (`mt_eval.py`'s rules, run through the same engine as everything else instead of Ollama):
  - E1: 214 two-turn items (`mt_heldout_v2.jsonl`: the 50 original + 164 new, hand-written on 2026-10-19).
    Near-copy = difflib ratio > 0.6 on the first 400 characters of reply 2 vs reply 1. Right = reply 2 contains an
    expected string (144 checkable items). Reply 1 goes back into the history as the visible reply. Cap 1,024 per turn.
  - E2: GSM8K items 200-399 asked as the second turn (turn 1 = items 0-199 with their reference solutions), vs the same
    items in the single-turn GSM8K run (same mode, same cap 4,096).
- Loop rate: a reply loops when it hits its cap (1,024) without stopping AND its last 256 tokens are periodic (≥ 80% of
  tokens equal the token p back, for some p ≤ 128). This replaces loop_check.py's reliance on Ollama's "token repeat
  limit" abort, which other engines don't have. Prompts: the 324 single-turn prompts of the held-out chat set
  `udgam3/evals/chat_prompts/heldout_v1.jsonl` (never used in training). Greedy, and the model's shipped sampling
  defaults (generation_config.json, else the card's own example, else vLLM's default temperature 1.0), seed 1000 + item.
- Truncation = the reply hit its cap. 95% CIs: Wilson for proportions; paired bootstrap over prompts (10,000) for
  instruction-level IFEval and the E2 difference; exact McNemar for E2. Per-item scores are in `scored/full/<model>/`
  for the paired tests of part 2b.
