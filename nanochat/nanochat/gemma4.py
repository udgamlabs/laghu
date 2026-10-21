"""
UdgamLabs training model: the Gemma 4 text architecture, written for fast training.

Every parameter name and every operation mirrors Hugging Face `Gemma4ForCausalLM`
(transformers >= 5.18), so a trained checkpoint exports to HF / GGUF (llama.cpp, Ollama) /
MLX by plain renaming-free saving. tests/test_gemma4_parity.py checks logits against HF.

Gemma 4 features used (all exportable):
- sandwich RMSNorms (input / post_attention / pre_feedforward / post_feedforward), learnable scales
- QK RMSNorm with learnable scales, V RMSNorm without scale, attention scale 1.0
- sliding-window ("S") and full ("L") attention layers; full layers use a larger head_dim
- standard RoPE on sliding layers; "proportional" RoPE (factor 1.0 == standard) on full layers
- GeGLU MLP (tanh-approx GELU)
- Per-Layer Embeddings (PLE): a token-indexed table + a projection of the input embedding,
  gated into every layer (the exportable stand-in for nanochat's value embeddings / x0 lambdas)
- per-layer output scalar (`layer_scalar`; trained here, stored as a buffer in HF)
- final logit softcap, untied input/output embeddings
Training-only differences: weights may be fp8/bf16 during training; nothing that changes the graph.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW
from nanochat.flash_attention import flash_attn


@dataclass
class Gemma4Config:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 20
    n_head: int = 16
    n_kv_head: int = 4
    n_embd: int = 1024
    head_dim: int = 64            # sliding-window layers
    global_head_dim: int = 128    # full-attention layers (llama.cpp's converter requires != head_dim)
    intermediate_size: int = 4096
    ple_dim: int = 64             # hidden_size_per_layer_input; 0 disables PLE
    window_pattern: str = "SSSL"  # tiled across layers; last layer is always L (HF forces this too)
    sliding_window: int = 512     # HF semantics: a token sees itself + (sliding_window - 1) previous tokens
    rope_theta_sliding: float = 10_000.0
    rope_theta_full: float = 1_000_000.0
    final_logit_softcap: float = 15.0
    rms_norm_eps: float = 1e-6
    # Looping (Udgam-2): stored layers [loop_start, loop_end) run n_loops times in a row. The
    # passes share weights exactly, so export "unrolls" them into a plain Gemma 4 model.
    loop_start: int = -1
    loop_end: int = -1
    n_loops: int = 1
    # Multi-token prediction (DeepSeek-V3 style, training only; dropped at export).
    mtp: bool = False
    # Udgam-Loop (bytecore Stage 1). None of these change the plain-Gemma path when left at defaults.
    loop_loss: bool = False        # also train every intermediate loop pass (deep supervision; bytecore found it essential)
    loop_loss_stride: int = 4      # aux losses use every k-th position (keeps the extra (N, V) logits small)
    loop_loss_weight: float = -1.0 # <0: (final + sum aux) / n_loops (Gate A); >=0: final + weight * mean(aux)
    loop_reinject: bool = False    # add the loop input back at the start of every loop pass (NOT exportable to stock Gemma 4)
    abacus: int = 0                # >0: digit-position embedding table size (NOT exportable to stock Gemma 4)
    abacus_offset: int = 0         # training-time random offset added to digit positions (length generalisation)
    digit_ids: tuple = ()          # token ids of plain digits '0'..'9'
    digit_space_ids: tuple = ()    # token ids of space+digit '▁0'..'▁9'
    tie_embeddings: bool = False   # Pocket Mathematician: share input embedding and output head (HF tie_word_embeddings)

    def layer_types(self):
        pat = self.window_pattern.upper()
        assert all(c in "SL" for c in pat)
        t = ["sliding_attention" if pat[i % len(pat)] == "S" else "full_attention" for i in range(self.n_layer)]
        t[-1] = "full_attention"
        return t

    def layer_order(self):
        """Stored-layer index for every executed (unrolled) layer."""
        if self.n_loops <= 1 or self.loop_start < 0:
            return list(range(self.n_layer))
        assert 0 <= self.loop_start < self.loop_end <= self.n_layer
        block = list(range(self.loop_start, self.loop_end))
        return list(range(self.loop_start)) + block * self.n_loops + list(range(self.loop_end, self.n_layer))


class Linear(nn.Linear):
    """nn.Linear that casts fp32 master weights to the activation dtype (nanochat convention)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class RMSNorm(nn.Module):
    """Gemma4RMSNorm: x * (mean(x^2) + eps)^-0.5 * weight (plain weight, not 1+weight)."""
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        return F.rms_norm(x, (x.size(-1),), w, self.eps)


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_cos_sin(seq_len, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)          # HF convention: (T, head_dim)
    return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


class Attention(nn.Module):
    def __init__(self, cfg: Gemma4Config, layer_idx: int, is_sliding: bool):
        super().__init__()
        self.is_sliding = is_sliding
        self.head_dim = cfg.head_dim if is_sliding else cfg.global_head_dim
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.window = (cfg.sliding_window - 1, 0) if is_sliding else (-1, 0)
        self.q_proj = Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, with_scale=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_head, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim))
        v = self.v_norm(self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim))
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # Gemma uses attention scale 1.0; flash attention applies 1/sqrt(head_dim), so pre-scale q.
        q = q * math.sqrt(self.head_dim)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=self.window)
        return self.o_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.gate_proj = Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up_proj = Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down_proj = Linear(cfg.intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4Config, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        eps = cfg.rms_norm_eps
        self.self_attn = Attention(cfg, layer_idx, layer_type == "sliding_attention")
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.n_embd, eps)
        self.post_attention_layernorm = RMSNorm(cfg.n_embd, eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.n_embd, eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.n_embd, eps)
        self.ple_dim = cfg.ple_dim
        if cfg.ple_dim:
            self.per_layer_input_gate = Linear(cfg.n_embd, cfg.ple_dim, bias=False)
            self.per_layer_projection = Linear(cfg.ple_dim, cfg.n_embd, bias=False)
            self.post_per_layer_input_norm = RMSNorm(cfg.n_embd, eps)
        self.layer_scalar = nn.Parameter(torch.ones(1))  # trained here; exported as HF buffer

    def forward(self, x, per_layer_input, cos, sin):
        h = self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin))
        x = x + h
        h = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        x = x + h
        if self.ple_dim:
            g = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * per_layer_input
            x = x + self.post_per_layer_input_norm(self.per_layer_projection(g))
        return x * self.layer_scalar.to(x.dtype)


class Gemma4Body(nn.Module):
    """Matches HF `Gemma4TextModel` parameter names (state dict prefix `model.`)."""
    def __init__(self, cfg: Gemma4Config, padded_vocab: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(padded_vocab, cfg.n_embd)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i, t) for i, t in enumerate(cfg.layer_types())])
        self.norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        if cfg.ple_dim:
            self.embed_tokens_per_layer = nn.Embedding(padded_vocab, cfg.n_layer * cfg.ple_dim)
            self.per_layer_model_projection = Linear(cfg.n_embd, cfg.n_layer * cfg.ple_dim, bias=False)
            self.per_layer_projection_norm = RMSNorm(cfg.ple_dim, cfg.rms_norm_eps)


class MTPModule(nn.Module):
    """One DeepSeek-V3 MTP depth: combine the main model's last hidden state at position t with the
    embedding of token t+1, run one decoder layer, and predict token t+2 with the shared lm_head."""
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        d = cfg.n_embd
        self.norm_h = RMSNorm(d, cfg.rms_norm_eps)
        self.norm_e = RMSNorm(d, cfg.rms_norm_eps)
        self.proj = Linear(2 * d, d, bias=False)
        self.block = DecoderLayer(cfg, cfg.n_layer, "sliding_attention")
        self.norm_out = RMSNorm(d, cfg.rms_norm_eps)


class Gemma4LM(nn.Module):
    def __init__(self, config: Gemma4Config, pad_vocab_size_to=64):
        super().__init__()
        self.config = cfg = config
        assert not (cfg.n_loops > 1 and cfg.ple_dim), "looping with PLE is not exportable; use --ple-dim 0"
        assert not (cfg.mtp and cfg.ple_dim), "the MTP module assumes --ple-dim 0"
        padded = ((cfg.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab = padded
        self.model = Gemma4Body(cfg, padded)
        self.lm_head = Linear(cfg.n_embd, padded, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight   # one (padded_vocab, d) table for input and output
        self.mtp = MTPModule(cfg) if cfg.mtp else None
        self.abacus = nn.Embedding(cfg.abacus, cfg.n_embd) if cfg.abacus else None
        if cfg.abacus:
            assert len(cfg.digit_ids) == 10 and len(cfg.digit_space_ids) == 10, "abacus needs the digit token ids"
        self._digit_cache = {}
        self.mtp_weight = 0.3   # set by the training script; used only in training-mode mean loss
        self.rotary_seq_len = cfg.sequence_len * 4
        for name, hd, th in [("s", cfg.head_dim, cfg.rope_theta_sliding), ("f", cfg.global_head_dim, cfg.rope_theta_full)]:
            cos, sin = rope_cos_sin(self.rotary_seq_len, hd, th, device="meta")
            self.register_buffer(f"cos_{name}", cos, persistent=False)
            self.register_buffer(f"sin_{name}", sin, persistent=False)

    # ---------------------------------------------------------------- init
    @torch.no_grad()
    def init_weights(self):
        cfg = self.config
        if cfg.tie_embeddings:   # meta -> to_empty() gives the head its own storage; re-share it before initialising
            self.lm_head.weight = self.model.embed_tokens.weight
        d = cfg.n_embd
        # Gemma multiplies embeddings by sqrt(d); std 1/sqrt(d) gives unit-RMS residual input.
        nn.init.normal_(self.model.embed_tokens.weight, std=d ** -0.5)
        if not cfg.tie_embeddings:   # tied: the head IS the embedding table (init above, unit-scale logits)
            nn.init.normal_(self.lm_head.weight, std=0.001)
        s = 3 ** 0.5 * d ** -0.5
        layers = list(self.model.layers)
        if self.mtp is not None:
            layers.append(self.mtp.block)
            nn.init.uniform_(self.mtp.proj.weight, -3 ** 0.5 * (2 * d) ** -0.5, 3 ** 0.5 * (2 * d) ** -0.5)
            for n in (self.mtp.norm_h, self.mtp.norm_e, self.mtp.norm_out):
                n.weight.fill_(1.0)
        for layer in layers:
            a = layer.self_attn
            for w in (a.q_proj.weight, a.k_proj.weight, a.v_proj.weight):
                nn.init.uniform_(w, -s, s)
            nn.init.zeros_(a.o_proj.weight)
            nn.init.uniform_(layer.mlp.gate_proj.weight, -s, s)
            nn.init.uniform_(layer.mlp.up_proj.weight, -s, s)
            nn.init.zeros_(layer.mlp.down_proj.weight)
            # q/k norm scales start at head_dim^-0.25 each => effective 1/sqrt(head_dim) attention temperature
            qk = a.head_dim ** -0.25
            a.q_norm.weight.fill_(qk); a.k_norm.weight.fill_(qk)
            for n in (layer.input_layernorm, layer.post_attention_layernorm, layer.pre_feedforward_layernorm, layer.post_feedforward_layernorm):
                n.weight.fill_(1.0)
            layer.layer_scalar.fill_(1.0)
            if cfg.ple_dim and layer.ple_dim:
                nn.init.uniform_(layer.per_layer_input_gate.weight, -s, s)
                nn.init.zeros_(layer.per_layer_projection.weight)
                layer.post_per_layer_input_norm.weight.fill_(1.0)
        self.model.norm.weight.fill_(1.0)
        if self.abacus is not None:   # starts as a small perturbation of the token embedding
            nn.init.normal_(self.abacus.weight, std=0.1 * d ** -0.5)
        if cfg.ple_dim:
            nn.init.normal_(self.model.embed_tokens_per_layer.weight, std=cfg.ple_dim ** -0.5)
            nn.init.uniform_(self.model.per_layer_model_projection.weight, -s, s)
            self.model.per_layer_projection_norm.weight.fill_(1.0)
        dev = self.model.embed_tokens.weight.device
        for name, hd, th in [("s", cfg.head_dim, cfg.rope_theta_sliding), ("f", cfg.global_head_dim, cfg.rope_theta_full)]:
            cos, sin = rope_cos_sin(self.rotary_seq_len, hd, th, device=dev)
            setattr(self, f"cos_{name}", cos); setattr(self, f"sin_{name}", sin)
        if COMPUTE_DTYPE != torch.float16 and not cfg.tie_embeddings:   # a tied table stays fp32 (it is also the output head)
            self.model.embed_tokens.to(dtype=COMPUTE_DTYPE)
            if self.abacus is not None:
                self.abacus.to(dtype=COMPUTE_DTYPE)
            if cfg.ple_dim:
                self.model.embed_tokens_per_layer.to(dtype=COMPUTE_DTYPE)

    @torch.no_grad()
    def init_nonzero_outputs(self):
        """MuonH keeps every matrix on a sphere of its initial norm, so zero-initialised output projections
        (o_proj, down_proj) would be stuck at radius 0. Give them a small random init instead (std ~0.5/sqrt(fan_in))."""
        layers = list(self.model.layers) + ([self.mtp.block] if self.mtp is not None else [])
        for layer in layers:
            for w in (layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight):
                b = 0.5 * 3 ** 0.5 * w.shape[1] ** -0.5
                nn.init.uniform_(w, -b, b)

    def get_device(self):
        return self.model.embed_tokens.weight.device

    # ---------------------------------------------------------------- accounting
    def num_matmul_params(self):
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def estimate_flops(self):
        """Training FLOPs per token, counting every executed (looped) layer and the MTP module."""
        cfg, t = self.config, self.config.sequence_len
        def layer_cost(layer):
            a = layer.self_attn
            eff = t if a.window[0] < 0 else min(a.window[0] + 1, t)
            mm = sum(m.weight.numel() for m in layer.modules() if isinstance(m, Linear))
            return 6 * mm + 12 * cfg.n_head * a.head_dim * eff
        flops = sum(layer_cost(self.model.layers[i]) for i in cfg.layer_order())
        flops += 6 * self.lm_head.weight.numel()
        if cfg.ple_dim:
            flops += 6 * self.model.per_layer_model_projection.weight.numel()
        if self.mtp is not None:
            flops += layer_cost(self.mtp.block) + 6 * (self.mtp.proj.weight.numel() + self.lm_head.weight.numel())
        if cfg.loop_loss and cfg.n_loops > 1:   # each intermediate pass also runs the coda + a strided head
            coda = sum(layer_cost(self.model.layers[i]) for i in range(cfg.loop_end, cfg.n_layer))
            flops += (cfg.n_loops - 1) * (coda + 6 * self.lm_head.weight.numel() / cfg.loop_loss_stride)
        return flops

    def num_scaling_params(self):
        emb = self.model.embed_tokens.weight.numel()
        ple = self.model.embed_tokens_per_layer.weight.numel() if self.config.ple_dim else 0
        lm_head = 0 if self.config.tie_embeddings else self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return {"wte": emb, "value_embeds": ple, "lm_head": lm_head,
                "transformer_matrices": total - emb - ple - lm_head, "scalars": 0, "total": total}

    # ---------------------------------------------------------------- optimizer
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5, hyperball=False):
        d = self.config.n_embd
        dscale = (d / 768) ** -0.5
        matrix, norms, scalars = [], [], []
        named = list(self.model.named_parameters())
        if self.mtp is not None:
            named += [("mtp." + n, p) for n, p in self.mtp.named_parameters()]
        for n, p in named:
            if n.startswith(("embed_tokens", "embed_tokens_per_layer")):
                continue
            if p.ndim == 2:
                matrix.append(p)
            elif n.endswith("layer_scalar"):
                scalars.append(p)
            else:
                norms.append(p)
        emb = [self.model.embed_tokens.weight] + ([self.abacus.weight] if self.abacus is not None else [])
        ple = [self.model.embed_tokens_per_layer.weight] if self.config.ple_dim else []
        if self.config.tie_embeddings:   # one shared table: train it at the (gentler) output-head learning rate
            emb_rest = [p for p in emb if p is not self.model.embed_tokens.weight]
            groups = [dict(kind="adamw", params=[self.lm_head.weight], lr=unembedding_lr * dscale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01)]
            if emb_rest:
                groups.append(dict(kind="adamw", params=emb_rest, lr=embedding_lr * dscale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001))
        else:
            groups = [
                dict(kind="adamw", params=[self.lm_head.weight], lr=unembedding_lr * dscale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
                dict(kind="adamw", params=emb, lr=embedding_lr * dscale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            ]
        groups += [
            dict(kind="adamw", params=norms, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=scalars, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if ple:
            groups.append(dict(kind="adamw", params=ple, lr=embedding_lr * dscale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01))
        for shape in sorted({p.shape for p in matrix}):
            groups.append(dict(kind="muon", params=[p for p in matrix if p.shape == shape], lr=matrix_lr,
                               momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0 if hyperball else weight_decay,
                               hyperball=hyperball))
        assert sum(len(g["params"]) for g in groups) == len(list(self.parameters()))
        opt = MuonAdamW(groups)
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
        return opt

    # ---------------------------------------------------------------- digits (Abacus)
    def _digit_tensors(self, device):
        key = str(device)
        if key not in self._digit_cache:
            cfg = self.config
            self._digit_cache[key] = (torch.tensor(cfg.digit_ids, device=device), torch.tensor(cfg.digit_space_ids, device=device))
        return self._digit_cache[key]

    def digit_positions(self, idx):
        """1-based position of each digit token inside its number (a number starts at a space+digit token or at a
        digit after a non-digit), 0 for non-digits. With little-endian data the ones digit is position 1."""
        plain_ids, space_ids = self._digit_tensors(idx.device)
        spaced = torch.isin(idx, space_ids)
        dig = torch.isin(idx, plain_ids) | spaced
        prev = F.pad(dig[:, :-1], (1, 0), value=False)
        start = dig & (spaced | ~prev)
        ar = torch.arange(idx.size(1), device=idx.device).expand_as(idx)
        start_pos = torch.where(start, ar, torch.full_like(ar, -1)).cummax(dim=1).values
        return torch.where(dig, ar - start_pos + 1, torch.zeros_like(ar)), dig

    def abacus_embeddings(self, idx):
        cfg = self.config
        pos, dig = self.digit_positions(idx)
        if self.training and cfg.abacus_offset > 0:
            off = torch.randint(0, cfg.abacus_offset + 1, (idx.size(0), 1), device=idx.device)
            pos = torch.where(dig, pos + off, pos)
        pos = pos.clamp(max=cfg.abacus - 1)
        return self.abacus(pos) * dig.unsqueeze(-1).to(self.abacus.weight.dtype)

    # ---------------------------------------------------------------- forward
    def _logits(self, x):
        cfg = self.config
        logits = self.lm_head(x)[..., :cfg.vocab_size].float()
        if cfg.final_logit_softcap:
            c = cfg.final_logit_softcap
            logits = c * torch.tanh(logits / c)
        return logits

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", tst_s=1):
        """tst_s > 1: Token Superposition Training phase-1 step (udgam2/specs/tst.md). idx/targets are
        (B, s*T) standard next-token pairs; each position sees the mean embedding of a bag of s tokens and
        is trained to predict every token of the next bag (mean CE)."""
        assert kv_cache is None, "use the exported HF/llama.cpp model for KV-cached inference"
        cfg = self.config
        dt = COMPUTE_DTYPE
        m = self.model
        ab = self.abacus_embeddings(idx) if self.abacus is not None else None   # (B, L, d) or None
        if tst_s > 1:
            assert cfg.ple_dim == 0, "TST needs --ple-dim 0"
            B, L = idx.shape
            T = L // tst_s
            e = m.embed_tokens(idx.view(B, T, tst_s)).float()
            if ab is not None:
                e = e + ab.float().view(B, T, tst_s, -1)
            x = e.mean(dim=2).to(dt) * torch.tensor(cfg.n_embd ** 0.5, dtype=dt)
        else:
            B, T = idx.shape
            e = m.embed_tokens(idx).to(dt)
            if ab is not None:
                e = e + ab.to(dt)
            x = e * torch.tensor(cfg.n_embd ** 0.5, dtype=dt)
        assert T <= self.rotary_seq_len
        ple = None
        if cfg.ple_dim:
            tok = m.embed_tokens_per_layer(idx).to(dt) * torch.tensor(cfg.ple_dim ** 0.5, dtype=dt)
            tok = tok.view(B, T, cfg.n_layer, cfg.ple_dim)
            proj = m.per_layer_model_projection(x) * (cfg.n_embd ** -0.5)
            proj = m.per_layer_projection_norm(proj.view(B, T, cfg.n_layer, cfg.ple_dim))
            ple = (proj + tok) * (2.0 ** -0.5)
        cs = {"sliding_attention": (self.cos_s[:, :T].to(dt), self.sin_s[:, :T].to(dt)),
              "full_attention": (self.cos_f[:, :T].to(dt), self.sin_f[:, :T].to(dt))}

        def run(i, x):
            layer = m.layers[i]
            cos, sin = cs[layer.layer_type]
            return layer(x, ple[:, :, i] if ple is not None else None, cos, sin)

        looped = cfg.n_loops > 1 and cfg.loop_start >= 0
        want_aux = looped and cfg.loop_loss and targets is not None and self.training and loss_reduction == "mean"
        aux_h = []
        if not looped:
            for i in cfg.layer_order():
                x = run(i, x)
        else:
            for i in range(cfg.loop_start):
                x = run(i, x)
            x_in = x
            for r in range(cfg.n_loops):
                if r > 0 and cfg.loop_reinject:
                    x = x + x_in
                for i in range(cfg.loop_start, cfg.loop_end):
                    x = run(i, x)
                if want_aux and r < cfg.n_loops - 1:
                    aux_h.append(x)
            for i in range(cfg.loop_end, cfg.n_layer):
                x = run(i, x)
        h = x
        logits = self._logits(m.norm(x))
        if targets is None:
            return logits
        V = logits.size(-1)
        aux_loss = None
        if aux_h:   # deep supervision: each intermediate pass goes through the coda and a strided head
            k = cfg.loop_loss_stride
            if tst_s > 1:
                lab_all = F.pad(targets, (0, tst_s - 1), value=-1)[:, tst_s - 1:].reshape(-1, tst_s)[::k]
            else:
                lab_all = targets.reshape(-1, 1)[::k]
            terms = []
            for z in aux_h:
                for i in range(cfg.loop_end, cfg.n_layer):
                    z = run(i, z)
                zf = m.norm(z).reshape(-1, z.size(-1))[::k]
                lg = self._logits(zf)
                lse = torch.logsumexp(lg, dim=-1, keepdim=True)
                valid = lab_all >= 0
                tgt = lg.gather(1, lab_all.clamp_min(0))
                terms.append(((lse - tgt) * valid).sum() / valid.sum().clamp_min(1))
            aux_loss = sum(terms)
            if cfg.loop_loss_weight >= 0:
                aux_loss = cfg.loop_loss_weight * aux_loss / len(terms)
        if tst_s > 1:
            # labels of bag t = tokens of bag t+1: shift standard labels left by s-1, pad, view (B, T, s)
            lab = F.pad(targets, (0, tst_s - 1), value=-1)[:, tst_s - 1:].reshape(-1, tst_s)
            flat = logits.view(-1, V)
            # Memory: one logsumexp over the (B*T, V) logits + a gather of the s targets, instead of s full
            # cross-entropy passes (each materialised another fp32 (B*T, V) buffer -> OOM at 65K x 32K on H100).
            # Mean NLL over all valid (position, bag-slot) pairs == the mean of the s CE terms up to the few
            # padded slots of the last bag.
            lse = torch.logsumexp(flat, dim=-1, keepdim=True)
            valid = lab >= 0
            tgt = flat.gather(1, lab.clamp_min(0))
            loss = ((lse - tgt) * valid).sum() / valid.sum().clamp_min(1)
            if aux_loss is not None:
                loss = loss + aux_loss if cfg.loop_loss_weight >= 0 else (loss + aux_loss) / cfg.n_loops
            if self.mtp is not None:   # MTP is idle during TST; give its params zero grads so the optimizer can stack them
                loss = loss + 0.0 * sum(p.sum() for p in self.mtp.parameters())
            return loss
        loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        if aux_loss is not None:
            self.last_final_loss = loss.detach()
            loss = loss + aux_loss if cfg.loop_loss_weight >= 0 else (loss + aux_loss) / cfg.n_loops
        if self.mtp is not None and self.training and loss_reduction == "mean" and T > 1:
            mt = self.mtp
            e = m.embed_tokens(idx[:, 1:]).to(dt) * torch.tensor(cfg.n_embd ** 0.5, dtype=dt)
            z = mt.proj(torch.cat([mt.norm_h(h[:, :-1]), mt.norm_e(e)], dim=-1))
            cos, sin = cs["sliding_attention"]
            z = mt.block(z, None, cos[:, :T - 1], sin[:, :T - 1])
            mlog = self._logits(mt.norm_out(z))
            mtp_loss = F.cross_entropy(mlog.reshape(-1, V), targets[:, 1:].reshape(-1), ignore_index=-1)
            self.last_mtp_loss = mtp_loss.detach()
            loss = loss + self.mtp_weight * mtp_loss
        return loss

    # ---------------------------------------------------------------- export
    def _assert_exportable(self):
        cfg = self.config
        if cfg.loop_reinject or cfg.abacus:
            raise NotImplementedError("loop_reinject / abacus are not part of stock Gemma 4: this model needs the custom "
                                      "Udgam-Loop runtime (trust_remote_code), not the plain HF/GGUF export")

    def hf_config(self, bos_id=2, eos_id=1, pad_id=0):
        self._assert_exportable()
        cfg = self.config
        stored = cfg.layer_types()
        lt = [stored[i] for i in cfg.layer_order()]   # looped models export unrolled
        return {
            "architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text",
            "vocab_size": cfg.vocab_size, "hidden_size": cfg.n_embd, "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": len(lt), "num_attention_heads": cfg.n_head, "num_key_value_heads": cfg.n_kv_head,
            "head_dim": cfg.head_dim, "global_head_dim": cfg.global_head_dim,
            "per_layer_config": {str(i): {"head_dim": cfg.global_head_dim} for i, t in enumerate(lt) if t == "full_attention"},
            "hidden_activation": "gelu_pytorch_tanh", "max_position_embeddings": cfg.sequence_len * 4,
            "rms_norm_eps": cfg.rms_norm_eps, "sliding_window": cfg.sliding_window, "layer_types": lt,
            "final_logit_softcapping": cfg.final_logit_softcap or None, "tie_word_embeddings": bool(cfg.tie_embeddings),
            "hidden_size_per_layer_input": cfg.ple_dim, "vocab_size_per_layer_input": cfg.vocab_size,
            "num_kv_shared_layers": 0, "use_double_wide_mlp": False, "attention_k_eq_v": False, "enable_moe_block": False,
            "attention_bias": False, "attention_dropout": 0.0, "use_cache": True,
            "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": cfg.rope_theta_sliding},
                                "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 1.0, "rope_theta": cfg.rope_theta_full}},
            "pad_token_id": pad_id, "eos_token_id": eos_id, "bos_token_id": bos_id, "dtype": "bfloat16",
        }

    @torch.no_grad()
    def hf_state_dict(self):
        """State dict with HF names/shapes: vocab padding cropped, layer_scalar kept as a (1,) tensor."""
        self._assert_exportable()
        V = self.config.vocab_size
        order = self.config.layer_order()
        out = {}
        for k, v in self.state_dict().items():
            if k.startswith(("cos_", "sin_", "mtp.", "abacus.")):
                continue
            if k == "lm_head.weight" and self.config.tie_embeddings:   # HF ties it to model.embed_tokens
                continue
            v = v.detach().float().cpu()
            if k in ("model.embed_tokens.weight", "model.embed_tokens_per_layer.weight", "lm_head.weight"):
                v = v[:V]
            if k.startswith("model.layers."):
                stored_i, rest = k[len("model.layers."):].split(".", 1)
                for pos, si in enumerate(order):   # unroll: write the shared weights at every position that runs them
                    if si == int(stored_i):
                        out[f"model.layers.{pos}.{rest}"] = v.clone().contiguous()
                continue
            out[k] = v.contiguous()
        return out
