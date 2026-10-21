# Udgam-3 fine-tuning (SFT) run plan v1, 2026-10-19

**Status (2026-10-19): DONE.** Three runs (1/2/3 passes) on Colab G4; **chosen: 2 passes (`YOUR_HF_NAMESPACE/laghu-sft` @ c8c36d1,
`laghu-sft-v1-ep2/hf`)** by the user-approved dev rule (`devpick/DEV_RESULTS.md`). Next: the RL plan.

## Data (rendered, `data/udgam3_sft_v1`, built by `udgam3/sft/render_v1.py --model-name Laghu`; private HF `YOUR_HF_NAMESPACE/laghu-sft-data/v1-laghu` @ 11dd41c)
| | |
|---|---|
| Conversations | **41,467 train + 853 val** (2% of ids per source, stable hash) |
| Tokens | 28.19M per epoch, of which **9.57M trained** (assistant turns only; tool-result payloads masked) |
| Multi-turn | **34.5%** (plan target ≥ 30%) |
| Sources | 10 verified DeepSeek-V4.1-Flash slices (chat 8,389; maths/logic 5,973 + their thinking-off twins; instruction following 3,823; Python tool 3,000; JSON/function calls 1,500; chat with thinking 1,500; system prompts 1,500; long documents 1,434; safety 1,200; identity 400 + 12 name-meaning rows (`identity_name.jsonl`: "Laghu is Sanskrit for small and light"), × 2 epochs) + Gargi-era multi-turn follow-ups by gpt-oss-120b (Apache-2.0), Gargi identity rows removed |
| Filters | credentials (caught the 2026-10-18 Infura key once), teacher fingerprints (26 rows), "not us" (797 Gargi-identity rows), > 16,384 tokens |
| Template | Udgam-3's exported chat template = the Gemma 4 canonical template (asserted byte for byte) |

## Training (proposed)
| Item | Choice | Why |
|---|---|---|
| Start | the 32K-stretch final weights (after its gate passes) | the chat model needs 16K for long documents |
| Script | `nanochat/scripts/udgam_sft.py` via Gargi's Colab pipeline (`udgam2/sft/colab_sft_pipeline.sh`), adapted | proven on Gargi |
| Sequence | 16,384 (best-fit packing, padding, no cropping) | long-document rows |
| Batch | 262,144 tokens per step | Gargi's SFT batch |
| LR | the script's defaults first (matrix 0.0071, embedding 0.106, unembedding 0.0028), 5% warmup, 50% warmdown to 0 | Gargi's SFT recipe |
| Epochs | 2 over the whole mix (identity 4), **chosen on the val set between 1, 2 and 3** in short dev runs | plan step 4: shares set on a dev set |
| Chat tokens | the base model never saw the chat-marker tokens (plain-text pretraining). Before SFT, set their input embeddings to the mean of the trained embeddings | their embeddings are untrained random vectors; base samples showed junk after them |
| Hardware | Colab G4 (~30-60 min, ~5-10 compute units) | $0 extra |

## Then RL (outline; its own checklist later)
Gargi's RL (`scripts/chat_rl.py`, Colab) used verifiable maths rewards. For Udgam-3:
- maths with thinking, GSM8K/MATH-style **train** problems and factory problems with checked answers, never test items;
- tool-mode problems rewarded on the executed answer;
- a loop and length penalty.

Checkpoint rule (pre-registration 4): the final checkpoint of the last stage, unless a dev rule is written into part 2b first.

## Decisions for the user
1. ~~The model's name~~ **Decided 2026-10-19: Laghu.** Identity rows fill `{MODEL_NAME}` with it.
2. Go for the SFT run once its checklist is shown (after the stretch).
