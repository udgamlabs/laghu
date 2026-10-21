# Udgam-3 (Laghu) 32K stretch: results (2026-10-19)

Run: exactly the approved config (`CHECKLIST_full.md`): Colab G4, 1,900 steps × 524,288 tokens = 996M tokens at seq
32,768, from `model_ema_076294.pt`. Training 06:08 → 10:02 UTC (rc=0, 75.9K tok/s, 56 GB peak). Final weights:
private HF `YOUR_HF_NAMESPACE/laghu-ctx32k` (`udgam3-ctx32k/model_001900.pt`, HF export `hf/udgam3-ctx32k`).

## Pass bars

| Check (source) | Bar | Result | |
|---|---|---|---|
| Needle at 4K / 8K / 12K / 16K (pre-registration 6.3) | ≥ 95% each | **100 / 100 / 99 / 96%** (100 trials each) | PASS |
| Base suite, validation splits, 5-task average vs main-run EMA (6.3) | drop ≤ 1 point | **50.81 → 50.05 (−0.76)** | PASS |
| 2K held-out bpb vs the base (checklist item 19) | rise ≤ 1% | web +0.40%, fresh −0.04%, task +0.82%, **reasoning +1.10%** | **narrow miss (reasoning)** |
| HF export verify (300 tokens) | PASS | max rel diff 2.6e-5, top-1 100% | PASS |
| Long parity at 16K (HF export vs our model) | PASS | max rel diff 3.1e-5, top-1 100% | PASS |

## Details

**Needle (single 5-digit code; exact match; 100 trials per length)**

| Context | 1K | 2K | 4K | 8K | 12K | 16K | 24K | 32K |
|---|---|---|---|---|---|---|---|---|
| Stretch | 100 | 100 | 100 | 100 | 99 | 96 | 0 | 0 |
| Base (2K) | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |

Copy (next-token accuracy): stretch 0.947 / 0.890 / 0.749 / 0.192 at 2K / 8K / 16K / 32K; base 0.948 / 0.043 / 0.023 / 0.023.
As planned, we claim 16K only (recall fails beyond ~16K, as with Gargi).

**2K held-out bpb (`eval_long.py`, same documents)**

| Set | Base EMA | Stretch | Change |
|---|---|---|---|
| web | 0.8501 | 0.8535 | +0.40% |
| task (chat-formatted) | 0.6341 | 0.6393 | +0.82% |
| reasoning | 0.6845 | 0.6920 | **+1.10%** |
| fresh | 0.9209 | 0.9206 | −0.04% |

**Base suite on validation splits** (lm-eval 0.4.13, transformers 5.19.0, bf16, batch 64, `<bos>`; same env for both;
per-item logs in `gate/out/`; ARC-E acc, ARC-C / HellaSwag / PIQA acc_norm, WinoGrande acc)

| Model | ARC-E | ARC-C | HellaSwag | PIQA | WinoGrande | Average |
|---|---|---|---|---|---|---|
| Main-run EMA | 62.81 | 32.44 | 40.93 | 65.72 | 52.17 | 50.81 |
| Stretch | 61.40 | 31.77 | 40.03 | 64.96 | 52.09 | 50.05 |
| Change | −1.41 | −0.67 | −0.90 | −0.76 | −0.08 | **−0.76** |

Long-document bpb by position (val_long, 400 docs): stretch stays flat to 32K (0.92-0.94 from 2K to 32K); the base
rises to 1.39 by 16K.

## Decision needed
The pre-registered bar (6.3) passes. The checklist's extra bpb bar misses on the reasoning set by 0.1 percentage
points. Per the checklist: stop and bring options to the user (no automatic top-up).

**User decision 2026-10-19 16:18 IST:** accept the stretch as is; the reasoning-set 2K bpb rise (+1.10% vs the checklist's 1%) is recorded as a miss here and in the technical report. SFT starts from these weights.
