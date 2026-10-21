# Phase 0 base suite on CUDA (Colab G4), all rivals [MEASURED 2026-10-19]

lm-eval 0.4.13, transformers 5.19.0, torch 2.14.1+cu130, bf16, batch 64, seed 1234, zero-shot except GSM8K (5-shot,
flexible-extract). (n) = acc_norm; MMLU-CF = `mmlu_continuation` group acc. Avg = mean of the first five columns.
Udgam models with add_bos_token=True; others with their own tokenizer defaults. Per-item logs: `cuda/<model>/`.
Pinned revisions: `manifest.json` (+ `remote_code_pins.json` for the 10 remote-code models, reviewed in
`REMOTE_CODE_REVIEW.md`; Q-164M has a built-in calculator, so its maths is tool-assisted).
Not runnable in the harness: lille-130m (no modeling code), GoLLeM-v5-128M (no model_type), Needle 3 (tool-calling only).

## size class (≤ 200M)

| Model | Params (M) | ARC-E | ARC-C (n) | HS (n) | PIQA (n) | WG | MMLU-MCF | MMLU-CF | GSM8K 5-shot | Avg (5) | Revision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AxiomicLabs/GPT-X3-150M | 149 | 63.0 | 31.4 | 45.9 | 71.3 | 53.6 | 24.6 | 28.8 | 4.2 | 53.0 | 46407499 |
| opencerebral/Boris-2-Preview-1 | 144 | 63.3 | 29.8 | 45.2 | 69.2 | 52.6 | 23.8 | 28.0 | 4.8 | 52.0 | 431ba2b8 |
| HuggingFaceTB/SmolLM2-135M | 135 | 64.5 | 30.0 | 43.0 | 68.3 | 53.0 | 24.5 | 29.9 | 2.0 | 51.8 | 93efa2f0 |
| bench-labs/cagliostro-v3.5 | 146 | 58.8 | 29.2 | 43.4 | 67.7 | 51.5 | 24.7 | 28.3 | 6.9 | 50.1 | 2e3a2cdb |
| SLM-Archive/BananaMind-2-Pro | 160 | 58.7 | 27.8 | 42.8 | 67.6 | 51.9 | 23.5 | 28.4 | 1.4 | 49.8 | d1c77cff |
| HuggingFaceTB/SmolLM2-135M-Instruct | 135 | 54.3 | 27.6 | 43.0 | 66.6 | 51.4 | 22.9 | 27.6 | 1.8 | 48.6 | 12fd25f7 |
| User01110/100M-exp | 98 | 55.3 | 26.6 | 37.8 | 66.8 | 51.2 | 24.8 | 26.7 | 1.7 | 47.5 | 6341d606 |
| tiiuae/Falcon-H1-Tiny-90M-Base | 91 | 54.3 | 28.5 | 37.7 | 64.8 | 50.8 | 32.4 | 28.6 | 2.7 | 47.2 | 7994372e |
| altslate/JugnuLM-110M-R2plus | 110 | 55.1 | 26.5 | 36.8 | 65.3 | 51.1 | 24.5 | 27.2 | 1.7 | 47.0 | d3dd9ab3 |
| bench-labs/tinctura-v1 | 96 | 53.9 | 26.4 | 38.0 | 65.9 | 50.4 | 25.2 | 27.0 | 4.4 | 46.9 | f475b8a2 |
| tiiuae/Falcon-H1-Tiny-90M-Instruct | 91 | 50.6 | 28.8 | 38.4 | 63.4 | 51.6 | 33.1 | 28.9 | 7.7 | 46.6 | e6389502 |
| sfanm/nltower-100m-base | 124 | 54.3 | 28.3 | 33.7 | 65.0 | 51.0 | 28.4 | 26.1 | 2.1 | 46.5 | 43d734a0 |
| SurjoLabs/Surjo-100m | 98 | 54.4 | 25.9 | 34.9 | 63.9 | 52.9 | 24.9 | 27.0 | 1.8 | 46.4 | bd2dd851 |
| SupraLabs/Supra2-100M-Instruct | 100 | 51.2 | 24.7 | 35.9 | 64.5 | 49.7 | 26.0 | 26.6 | 1.3 | 45.2 | 2c90ade6 |
| facebook/MobileLLM-R1-140M-base | 140 | 52.7 | 24.3 | 34.1 | 63.1 | 50.4 | 24.3 | 27.1 | 19.7 | 44.9 | 27cb0e3b |
| PleIAs/Monad | 56 | 44.9 | 24.8 | 30.7 | 54.0 | 52.3 | 24.2 | 27.4 | 1.8 | 41.3 | d8fce6b0 |
| q-project/Q-164M | 181 | 42.3 | 23.0 | 31.6 | 57.6 | 51.5 | 23.1 | 25.6 | 1.7 | 41.2 | 5ed694f0 |
| EleutherAI/pythia-160m | 162 | 37.0 | 23.6 | 30.5 | 60.4 | 51.6 | 22.9 | 25.1 | 0.6 | 40.6 | 50f5173d |

## near-size reference

| Model | Params (M) | ARC-E | ARC-C (n) | HS (n) | PIQA (n) | WG | MMLU-MCF | MMLU-CF | GSM8K 5-shot | Avg (5) | Revision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M | 362 | 70.2 | 38.1 | 56.2 | 71.6 | 58.9 | 25.5 | 33.8 | 3.9 | 59.0 | f8027fd0 |
| LiquidAI/LFM2.5-230M | 230 | 60.9 | 32.0 | 40.9 | 64.7 | 54.6 | 38.8 | 30.5 | 25.5 | 50.6 | 40cb2ad3 |
| google/gemma-3-270m | 268 | 58.8 | 28.3 | 41.3 | 68.2 | 53.6 | 26.2 | 28.5 | 2.0 | 50.0 | 9b0cfec8 |
| glouriousgautam/LiLM1-230M-base | 234 | 57.4 | 26.3 | 40.5 | 67.1 | 51.0 | 23.6 | 27.5 | 2.5 | 48.5 | 5391c31c |

## ours

| Model | Params (M) | ARC-E | ARC-C (n) | HS (n) | PIQA (n) | WG | MMLU-MCF | MMLU-CF | GSM8K 5-shot | Avg (5) | Revision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **YOUR_HF_NAMESPACE/laghu-base (EMA, step 76,294), measured after part 2a** | 152 | 60.6 | 30.1 | 40.9 | 65.7 | 52.2 | 25.3 | 28.8 | 4.2 | 49.9 | 1f2e414c |
| udgamlabs/Gargi-Base | 384 | 63.0 | 32.5 | 41.5 | 66.9 | 54.5 | 25.3 | 30.1 | 5.7 | 51.7 | 8a11ccf5 |
| udgamlabs/Gargi | 384 | 62.5 | 30.5 | 41.9 | 66.6 | 56.1 | 24.9 | 29.6 | 19.4 | 51.5 | 4398abf9 |
| UdgamLabs/Udgam-384M-Chat | 384 | 59.0 | 30.2 | 40.9 | 65.3 | 51.9 | 24.9 | 29.0 | 1.7 | 49.5 | af005e4c |
| UdgamLabs/Udgam-384M-Base-4K | 384 | 59.0 | 30.0 | 39.8 | 66.1 | 52.1 | 25.2 | 29.0 | 2.4 | 49.4 | bfe97283 |

## Best size-class rival per task (the bars for part 2a)

| Task | Best ≤ 200M rival | Score |
|---|---|---|
| ARC-E | HuggingFaceTB/SmolLM2-135M | 64.5 |
| ARC-C (n) | AxiomicLabs/GPT-X3-150M | 31.4 |
| HS (n) | AxiomicLabs/GPT-X3-150M | 45.9 |
| PIQA (n) | AxiomicLabs/GPT-X3-150M | 71.3 |
| WG | AxiomicLabs/GPT-X3-150M | 53.6 |
| MMLU-MCF | tiiuae/Falcon-H1-Tiny-90M-Instruct | 33.1 |
| MMLU-CF | HuggingFaceTB/SmolLM2-135M | 29.9 |
| GSM8K 5-shot | facebook/MobileLLM-R1-140M-base | 19.7 |
