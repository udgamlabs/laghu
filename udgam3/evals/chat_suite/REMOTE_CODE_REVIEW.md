# Remote-code review for the chat suite (2026-10-19)

One chat model in the suite ships its own modeling code (`trust_remote_code`): **bench-labs/cagliostro-v3.5-chat**,
pinned at `e10b21717356d87652f87345677ae160b2d4809d` (manifest.json). Its `.py` files were downloaded at that commit
and read **without executing them**, as in `udgam3/evals/phase0/REMOTE_CODE_REVIEW.md`.

- **Searched for:** network access, shell or subprocess calls, `exec`/`eval`/`compile`/`__import__`, base64, pickle
  loads, file reads or writes, downloads, environment-variable reads, and unusual imports.

| File | Lines | Imports | Findings | Verdict |
|---|---|---|---|---|
| `configuration_cagliostro.py` | 43 | `transformers.PretrainedConfig` | a plain config class | OK |
| `modeling_cagliostro.py` | 475 | `math`, `torch`, `transformers` (PreTrainedModel, GenerationMixin, CausalLMOutputWithPast, DynamicCache) | standard decoder (RMSNorm, RoPE, GQA attention with an optional XSA projection, SwiGLU/ReLU² MLP, logit cap); `generate_fast()` uses `torch.compile` of its own decode step and CUDA graphs; no I/O, no network, no subprocess, no exec/eval | OK |

The repo's `evals/code/*.py` (the authors' own evaluation scripts) are never imported by `AutoModelForCausalLM` and were
not run. Its tokenizer needs no remote code (`tokenizer.json` + `PreTrainedTokenizerFast`), so tokenizers load with
`trust_remote_code=False` everywhere, including on the Mac for the dry runs.

**How it ran:** on the disposable Colab G4 VM, via transformers (the HF engine of `chat_gen.py`); our code calls
`generate()`, never `generate_fast()`. All model files were downloaded during setup while the HF token was present.
The token was then deleted (`setup.log`: "token removed: yes") before any job ran, and every job ran with
`HF_HUB_OFFLINE=1`. So the remote code never ran with a secret on the machine.

**Batching check:** with left padding, the model builds its own attention mask from `attention_mask` (padded keys
masked) and uses relative RoPE, so padding is masked (PREREGISTRATION.md 2.4.2). The fp32 batched-vs-one-at-a-time
check in RESULTS.md confirms this.

No other model in the suite needs remote code. SmolLM2, Falcon-H1-Tiny (all three), Supra2 (both), MobileLLM-R1.5,
Boris, Gargi, Udgam-384M-Chat, Gemma-3, LFM2.5 and SmolLM2-360M load with their built-in transformers and vLLM
architectures.
