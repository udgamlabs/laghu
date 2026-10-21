"""Udgam-3 main run: how many training tokens, at what cost, and what fresh-text loss to expect.

[ESTIMATE] Projection only. Two published scaling-law fits (Hoffmann et al. 2022; Besiroglu et al. 2024 replication),
used only for RATIOS and anchored on our own Gargi-Base (same lineage: tokenizer_v1, Gemma-4 layout, TST, FinePhrase/
SYNTH data; 384M params, 8.9B training positions, fresh-text bpb 0.914 [MEASURED]). Check of the size term: both fits
predict SmolLM2-360M vs SmolLM2-135M at -9.7..-10.0% loss; we measured -9.7% on our fresh-text set.
Throughput is dry run A [MEASURED] on 1x H100 SXM: TST steps 2.34 s, normal steps 1.80 s (524,288 positions/step).

  python3 udgam3/runs/main/token_projection.py
"""
import math

FITS = {"Hoffmann": (1.69, 406.4, 410.7, 0.34, 0.28), "Besiroglu": (1.8172, 482.01, 2085.43, 0.3478, 0.3658)}
EMB = 32768
GARGI = dict(n=384e6 - EMB * 1024, d=8.9e9, bpb=0.914)
OURS_N = 151.8e6 - EMB * 640               # tied embeddings: 130.8M non-embedding
V3_GAIN = 0.995                            # G1: v3 vs Gargi's v2 mix, -0.5% held-out web bpb at 65M [MEASURED]
STABLE_POOL, ANNEAL_POOL = 42.17e9, 10.33e9
BATCH = 524288
S_TST, S_NORM = 2.34, 1.80                 # 1-GPU seconds per step
OVERHEAD_H = 0.25                          # start, compiles, evals, final export [ESTIMATE]


def loss(fit, n, d):
    e, a, b, al, be = FITS[fit]
    return e + a / n ** al + b / d ** be


def hours(d, gpus, eff=0.95):
    steps = d / BATCH
    return (steps * (0.25 * S_TST + 0.75 * S_NORM) / (gpus * eff)) / 3600 + OVERHEAD_H


print(f"{'tokens':>7} {'stable epochs':>13} {'anneal ep':>9} | " + " | ".join(f"{f} bpb" for f in FITS) +
      " | 8xH100 Comm $21.52/h | 8xH100 Sec $31.92/h | 4xH100 Sec $15.96/h")
for d in [22e9, 26e9, 30e9, 35e9, 40e9, 45e9, 50e9]:
    stable = d * (0.25 * 4 + 0.55) / STABLE_POOL       # TST reads 4 tokens per position for the first 25% of steps
    anneal = d * 0.20 / ANNEAL_POOL
    proj = []
    for f in FITS:
        r = loss(f, OURS_N, d) / loss(f, GARGI["n"], GARGI["d"])
        proj.append(f"{GARGI['bpb'] * r * V3_GAIN:.3f} ({GARGI['bpb'] * r:.3f})")
    h8, h4 = hours(d, 8), hours(d, 4)
    print(f"{d/1e9:5.0f}B {stable:13.2f} {anneal:9.2f} | " + " | ".join(proj) +
          f" | {h8:4.1f} h ${h8*21.52:5.0f} | {h8:4.1f} h ${h8*31.92:5.0f} | {h4:4.1f} h ${h4*15.96:5.0f}")
print("bpb: with the v3 gain (without it). Bars: SmolLM2-135M 0.941, Udgam-384M 0.937, Gargi 0.914, SmolLM2-360M 0.850")
