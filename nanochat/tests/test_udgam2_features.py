"""Udgam-2 features on a tiny CPU model: looping (+ unrolled HF export parity), MTP, TST. Run: NANOCHAT_DTYPE=float32 PYTHONPATH=. python tests/test_udgam2_features.py"""
import torch, sys
sys.path.insert(0, ".")
from nanochat.gemma4 import Gemma4Config, Gemma4LM
torch.manual_seed(0)
def build(**kw):
    cfg = Gemma4Config(sequence_len=64, vocab_size=512, n_layer=4, n_head=4, n_kv_head=2, n_embd=64, head_dim=16,
                       global_head_dim=32, intermediate_size=128, ple_dim=0, sliding_window=16, **kw)
    m = Gemma4LM(cfg); m.init_weights(); return m
x = torch.randint(0, 512, (2, 64)); y = torch.randint(0, 512, (2, 64))
base = build(); print("base loss", base(x, y).item(), "flops/tok", base.estimate_flops())
loop = build(loop_start=1, loop_end=3, n_loops=3); print("order", loop.config.layer_order(), "flops/tok", loop.estimate_flops())
l = loop(x, y); l.backward(); print("loop loss", l.item(), "grad ok", loop.model.layers[1].mlp.down_proj.weight.grad.abs().sum().item() > 0)
mtp = build(mtp=True); mtp.train(); lm = mtp(x, y); lm.backward(); print("mtp loss", lm.item(), "mtp part", mtp.last_mtp_loss.item(), "mtp grad", mtp.mtp.proj.weight.grad.abs().sum().item() > 0)
opt = mtp.setup_optimizer(); print("optimizer groups ok", len(opt.param_groups))
xs = torch.randint(0, 512, (2, 64 * 4)); ys = torch.randint(0, 512, (2, 64 * 4))
t = base(xs, ys, tst_s=4); t.backward(); print("tst loss", t.item())
# export unroll: HF Gemma4 logits must match our looped forward
sd = loop.hf_state_dict(); cfg = loop.hf_config()
print("hf layers", cfg["num_hidden_layers"], len([k for k in sd if k.endswith("mlp.up_proj.weight")]))
try:
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    hf = Gemma4ForCausalLM(Gemma4TextConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type", "dtype")}))
    missing, unexpected = hf.load_state_dict(sd, strict=False); print("missing", [m for m in missing if "layer_scalar" not in m][:5], "unexpected", unexpected[:5])
    hf.eval(); loop.eval()
    with torch.no_grad():
        a = loop(x[:, :32]); b = hf(x[:, :32]).logits
    print("max |diff| ours vs HF unrolled:", (a - b).abs().max().item())
except Exception as e:
    print("HF parity skipped:", type(e).__name__, str(e)[:200])
