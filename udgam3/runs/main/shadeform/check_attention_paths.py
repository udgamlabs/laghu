"""$0 check (CPU): the non-FA3 attention path (A100, L40S, ...) computes exactly FA3's sliding-window attention at the
Udgam-3 training shape (T 2048, 10 q / 2 kv heads, head dim 64, window (511, 0) -> block-local fast path), with a
negative control. Result 2026-10-18: max diff 8.9e-7 (control 0.45); global layers 6.6e-7.
Note: export_hf.py --verify uses 300 tokens (< the 512 window), so it does not test window semantics; this script does.

  uv run --project evals/harness python udgam3/runs/main/shadeform/check_attention_paths.py
"""
import math, sys
import torch
sys.path.insert(0, "nanochat")
import nanochat.flash_attention as fa

torch.manual_seed(0)
B, T, Hq, Hk, D, W = 2, 2048, 10, 2, 64, 511
q = torch.randn(B, T, Hq, D); k = torch.randn(B, T, Hk, D); v = torch.randn(B, T, Hk, D)
fa.USE_FA3 = False
got = fa.flash_attn_func(q, k, v, causal=True, window_size=(W, 0))
qh = q.transpose(1, 2)
kh = k.repeat_interleave(Hq // Hk, dim=2).transpose(1, 2); vh = v.repeat_interleave(Hq // Hk, dim=2).transpose(1, 2)
i = torch.arange(T).unsqueeze(1); j = torch.arange(T).unsqueeze(0)
s = (qh @ kh.transpose(-1, -2)) / math.sqrt(D)
ref = (torch.softmax(s.masked_fill(~((j <= i) & (j >= i - W)), float("-inf")), -1) @ vh).transpose(1, 2)
full = fa.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
ref_full = (torch.softmax(s.masked_fill(~(j <= i), float("-inf")), -1) @ vh).transpose(1, 2)
d_win, d_ctrl, d_glob = (got - ref).abs().max().item(), (full - ref).abs().max().item(), (full - ref_full).abs().max().item()
print(f"sliding layers vs FA3 definition: {d_win:.1e} | control (full attention) {d_ctrl:.2f} | global layers {d_glob:.1e}")
ok = d_win < 1e-5 and d_glob < 1e-5 and d_ctrl > 1e-2
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
