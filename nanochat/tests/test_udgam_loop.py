"""Udgam-Loop features on a tiny CPU model: per-loop loss, input re-injection, Abacus digit positions, TST combos,
export guards. Run: NANOCHAT_DTYPE=float32 PYTHONPATH=. python tests/test_udgam_loop.py"""
import sys
import torch
sys.path.insert(0, ".")
from nanochat.gemma4 import Gemma4Config, Gemma4LM

torch.manual_seed(0)
PLAIN, SPACE = tuple(range(10, 20)), tuple(range(20, 30))


def build(**kw):
    cfg = Gemma4Config(sequence_len=64, vocab_size=512, n_layer=4, n_head=4, n_kv_head=2, n_embd=64, head_dim=16,
                       global_head_dim=32, intermediate_size=128, ple_dim=0, sliding_window=16,
                       digit_ids=PLAIN, digit_space_ids=SPACE, **kw)
    m = Gemma4LM(cfg)
    m.init_weights()
    return m


x = torch.randint(0, 512, (2, 64))
y = torch.randint(0, 512, (2, 64))
LOOP = dict(loop_start=1, loop_end=3, n_loops=3)
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# 1. per-loop loss: train-mode loss includes aux terms, eval/bpb path unchanged, grads flow, flops counted
m = build(loop_loss=True, **LOOP)
m.train()
lt = m(x, y)
lt.backward()
check("loop_loss trains", torch.isfinite(lt).item() and m.model.layers[1].mlp.down_proj.weight.grad.abs().sum() > 0)
check("loop_loss averages final+aux", abs(lt.item() - m.last_final_loss.item()) > 1e-6)
m.eval()
with torch.no_grad():
    le = m(x, y, loss_reduction="none")
check("eval path is final-loop only", le.shape == (2 * 64,))
check("flops include aux heads", build(loop_loss=True, **LOOP).estimate_flops() > build(**LOOP).estimate_flops())
with torch.no_grad():   # zero-init output projections make every layer an identity at init; use real weights
    for prm in m.parameters():
        prm.add_(0.02 * torch.randn_like(prm))
same = build(**LOOP)
same.load_state_dict(m.state_dict())
with torch.no_grad():
    check("eval logits identical with/without loop_loss", torch.allclose(m(x), same.eval()(x)))

# 2. re-injection changes the function and blocks stock export
r = build(loop_reinject=True, **LOOP)
r.load_state_dict(same.state_dict(), strict=False)
r.eval()
with torch.no_grad():
    check("reinject changes outputs", not torch.allclose(r(x), same(x)))
try:
    r.hf_config()
    check("reinject export refused", False)
except NotImplementedError:
    check("reinject export refused", True)

# 3. Abacus positions: ' 4','8' -> 1,2 ; '7' after non-digit -> 1 ; ' 1',' 2' are two numbers
seq = torch.tensor([[5, SPACE[4], PLAIN[8], 6, PLAIN[7], PLAIN[1], PLAIN[2], SPACE[1], SPACE[2], 7]])
a = build(abacus=32, abacus_offset=8)
pos, dig = a.digit_positions(seq)
check("abacus positions", pos.tolist() == [[0, 1, 2, 0, 1, 2, 3, 1, 1, 0]])
a.eval()
with torch.no_grad():
    e1 = a.abacus_embeddings(seq)
check("abacus zero on non-digits", e1[0, 0].abs().sum().item() == 0 and e1[0, 1].abs().sum().item() > 0)
a.train()
la = a(x, y)
la.backward()
check("abacus trains", a.abacus.weight.grad is not None and a.abacus.weight.grad.abs().sum() > 0)
check("abacus in optimizer", len(a.setup_optimizer().param_groups) > 0)
try:
    a.hf_state_dict()
    check("abacus export refused", False)
except NotImplementedError:
    check("abacus export refused", True)

# 4. everything together, incl. TST phase
allf = build(loop_loss=True, loop_reinject=True, abacus=32, abacus_offset=8, **LOOP)
allf.train()
xs, ys = torch.randint(0, 512, (2, 64 * 4)), torch.randint(0, 512, (2, 64 * 4))
lt = allf(xs, ys, tst_s=4)
lt.backward()
check("TST + loop_loss + reinject + abacus", torch.isfinite(lt).item())
l2 = allf(x, y)
l2.backward()
check("next-token + all features", torch.isfinite(l2).item())

# 5. the plain looped model still exports and matches HF unrolled (regression)
try:
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    p = build(loop_loss=True, **LOOP)   # loop_loss is training-only, so this stays exportable
    cfg = p.hf_config()
    hf = Gemma4ForCausalLM(Gemma4TextConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type", "dtype")}))
    hf.load_state_dict(p.hf_state_dict(), strict=False)
    hf.eval(); p.eval()
    with torch.no_grad():
        d = (p(x[:, :32]) - hf(x[:, :32]).logits).abs().max().item()
    check(f"loop_loss model exports unrolled (max diff {d:.2e})", d < 1e-4)
except ImportError:
    print("SKIP HF parity (transformers not installed)")

print("ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
