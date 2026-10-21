"""Parity: our nanochat Gemma4LM vs HF transformers Gemma4ForCausalLM on identical random weights.
Run (CPU, fp32): NANOCHAT_DTYPE=float32 python tests/test_gemma4_parity.py"""
import os, json, tempfile, torch
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
import sys
import nanochat.gemma4 as g4
F64 = "--f64" in sys.argv
if F64:
    g4.COMPUTE_DTYPE = torch.float64; torch.set_default_dtype(torch.float64)
from nanochat.gemma4 import Gemma4Config, Gemma4LM
from safetensors.torch import save_file
from transformers import Gemma4ForCausalLM

def run(ple, pattern, T=300):
    torch.manual_seed(0)
    cfg = Gemma4Config(sequence_len=512, vocab_size=1000, n_layer=6, n_head=4, n_kv_head=2, n_embd=128,
                       head_dim=32, global_head_dim=64, intermediate_size=384, ple_dim=ple,
                       window_pattern=pattern, sliding_window=64, final_logit_softcap=15.0)
    m = Gemma4LM(cfg)
    m.to_empty(device="cpu"); m.init_weights()
    with torch.no_grad():                      # make every tensor non-trivial
        for n, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.ndim == 1 else 0.05) + (1.0 if p.ndim == 1 else 0.0))
    m.eval()
    if F64: m.double()
    d = tempfile.mkdtemp()
    sd = m.hf_state_dict()
    save_file(sd, f"{d}/model.safetensors")
    json.dump(m.hf_config(), open(f"{d}/config.json", "w"))
    hf = Gemma4ForCausalLM.from_pretrained(d, dtype=torch.float64 if F64 else torch.float32, attn_implementation="eager").eval()
    missing = set(hf.state_dict()) - set(sd) - {k for k in hf.state_dict() if k.endswith("layer_scalar")}
    idx = torch.randint(0, cfg.vocab_size, (2, T))
    with torch.no_grad():
        a = m(idx); b = hf(idx).logits
    diff = (a - b).abs().max().item(); rel = diff / b.abs().max().item()
    # HF computes norms + RoPE in fp32 internally, so ~1e-5..1e-4 relative is the rounding floor;
    # a structural bug (e.g. window off-by-one, wrong norm) shows up as >1e-2.
    ok = rel < 5e-4 and not missing
    print(f"ple={ple:3d} pattern={pattern:5s} T={T}: max|diff| {diff:.2e} (rel {rel:.1e}) unmatched HF keys {sorted(missing)[:3]} -> {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    results = [run(ple, pat) for ple in (0, 16) for pat in ("SSSL", "L", "SL")]
    print("ALL PASS" if all(results) else "SOME FAILED")
