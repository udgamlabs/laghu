"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# UdgamLabs: Gemma 4 architecture (HF/GGUF/MLX-exportable). --depth sets the number of layers.
parser.add_argument("--arch", type=str, default="gpt", choices=["gpt", "gemma4"], help="model architecture")
parser.add_argument("--n-embd", type=int, default=1024, help="[gemma4] hidden size")
parser.add_argument("--n-head", type=int, default=16, help="[gemma4] query heads")
parser.add_argument("--n-kv-head", type=int, default=4, help="[gemma4] key/value heads")
parser.add_argument("--g4-head-dim", type=int, default=64, help="[gemma4] head dim of sliding-window layers")
parser.add_argument("--g4-global-head-dim", type=int, default=128, help="[gemma4] head dim of full-attention layers (must differ)")
parser.add_argument("--intermediate-size", type=int, default=4096, help="[gemma4] GeGLU hidden size")
parser.add_argument("--ple-dim", type=int, default=64, help="[gemma4] per-layer-embedding dim (0 = off)")
parser.add_argument("--sliding-window", type=int, default=512, help="[gemma4] sliding window size (HF semantics)")
parser.add_argument("--softcap", type=float, default=15.0, help="[gemma4] final logit softcap")
parser.add_argument("--loop-start", type=int, default=-1, help="[gemma4] first stored layer of the looped block (-1 = no looping)")
parser.add_argument("--loop-end", type=int, default=-1, help="[gemma4] end (exclusive) of the looped block")
parser.add_argument("--n-loops", type=int, default=1, help="[gemma4] times the looped block runs")
parser.add_argument("--mtp", action="store_true", help="[gemma4] add a DeepSeek-V3 style multi-token-prediction module (training only)")
parser.add_argument("--mtp-weight", type=float, default=0.3, help="[gemma4] weight of the MTP loss")
# Udgam-Loop (bytecore Stage 1)
parser.add_argument("--tie-embeddings", action="store_true", help="[gemma4] share input embedding and output head (Pocket Mathematician)")
parser.add_argument("--loop-loss", action="store_true", help="[gemma4] deep supervision: also train every intermediate loop pass")
parser.add_argument("--loop-loss-weight", type=float, default=-1.0, help="[gemma4] >=0: loss = final + w*mean(aux); <0: equal-weight average")
parser.add_argument("--loop-loss-stride", type=int, default=4, help="[gemma4] aux loop losses use every k-th position")
parser.add_argument("--loop-reinject", action="store_true", help="[gemma4] add the loop input back at every loop pass (custom runtime only)")
parser.add_argument("--abacus", type=int, default=0, help="[gemma4] digit-position embedding table size (0 = off; custom runtime only)")
parser.add_argument("--abacus-offset", type=int, default=0, help="[gemma4] max random training offset for digit positions")
parser.add_argument("--digits-reversed", action="store_true", help="little-endian numbers: reverse every digit run at load time")
parser.add_argument("--synth-arith-frac", type=float, default=0.0, help="token fraction of program-generated arithmetic worksheets (train only)")
parser.add_argument("--arith-probe", action="store_true", help="score the arithmetic probe at every eval and at the end")
parser.add_argument("--gsm-eval", type=str, default="none", choices=["none", "text", "code"], help="score GSM8K-test at the end (falsifier models)")
parser.add_argument("--gsm-dev-file", type=str, default="", help="jsonl {question, gold}: a held-out dev set scored during training (checkpoint choice)")
parser.add_argument("--gsm-dev-every", type=int, default=0, help="score the dev set (and save a checkpoint) every N steps")
parser.add_argument("--gsm-eval-n", type=int, default=0, help="limit GSM8K problems (0 = all 1,319)")
parser.add_argument("--results-json", type=str, default="", help="write the final metrics (val bpb, extra vals, EMA, arith probe) to this file")
parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "muonh"], help="matrix optimizer: muon (cautious WD) or muonh (Hyperball, no WD, matrix LR decays linearly over the whole run)")
parser.add_argument("--tst-s", type=int, default=1, help="Token Superposition Training bag size s (1 = off)")
parser.add_argument("--tst-frac", type=float, default=0.0, help="fraction of steps run in the TST superposition phase")
# UdgamLabs: extra held-out sets (e.g. task data) and end-of-training weight averaging
parser.add_argument("--extra-val", type=str, action="append", default=[], help="NAME=DIR: also report val bpb on the last parquet in DIR (repeatable)")
parser.add_argument("--ema-start-frac", type=float, default=-1.0, help="keep an EMA of the weights from this fraction of training onward (-1 = off)")
parser.add_argument("--ema-beta", type=float, default=0.998, help="EMA decay per step")
parser.add_argument("--seed", type=int, default=42, help="torch seed for init")
parser.add_argument("--phase-b-data-dir", type=str, default="", help="switch training data to these shards at --phase-b-start-frac (default: when LR warmdown starts)")
parser.add_argument("--phase-b-start-frac", type=float, default=-1.0, help="fraction of num_iterations at which to switch to --phase-b-data-dir (-1 = at LR warmdown start)")
parser.add_argument("--init-from", type=str, default="", help="continued pretraining: initialise the model weights from this state-dict file (e.g. model_ema_007500.pt) with a fresh optimizer and schedule; ignored when resuming this run's own checkpoint")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=os.environ.get("WANDB_PROJECT", "nanochat"), name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

from nanochat import udgam_loop
_digit_tabs = udgam_loop.digit_tables(tokenizer) if hasattr(tokenizer, "tok") else ([], [])
if args.digits_reversed or args.synth_arith_frac > 0 or args.arith_probe:
    udgam_loop.configure(tokenizer, digits_reversed=args.digits_reversed, synth_frac=args.synth_arith_frac, seed=args.seed)
    print0(f"Udgam-Loop data: digits_reversed={args.digits_reversed} synth_arith_frac={args.synth_arith_frac}")

def build_model_meta(depth, arch=None):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    if (arch or args.arch) == "gemma4":
        from nanochat.gemma4 import Gemma4Config, Gemma4LM
        config = Gemma4Config(
            sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=depth, n_head=args.n_head,
            n_kv_head=args.n_kv_head, n_embd=args.n_embd, head_dim=args.g4_head_dim,
            global_head_dim=args.g4_global_head_dim, intermediate_size=args.intermediate_size,
            ple_dim=args.ple_dim, window_pattern=args.window_pattern, sliding_window=args.sliding_window,
            final_logit_softcap=args.softcap,
            loop_start=args.loop_start, loop_end=args.loop_end, n_loops=args.n_loops, mtp=args.mtp,
            loop_loss=args.loop_loss, loop_loss_stride=args.loop_loss_stride, loop_loss_weight=args.loop_loss_weight, loop_reinject=args.loop_reinject,
            abacus=args.abacus, abacus_offset=args.abacus_offset, tie_embeddings=args.tie_embeddings,
            digit_ids=tuple(_digit_tabs[0]), digit_space_ids=tuple(_digit_tabs[1]),
        )
        with torch.device("meta"):
            return Gemma4LM(config)
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
torch.manual_seed(args.seed)
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized
if getattr(model, "mtp", None) is not None:
    model.mtp_weight = args.mtp_weight
if args.optimizer == "muonh":
    model.init_nonzero_outputs()

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy
elif args.init_from:
    print0(f"Initialising weights from {args.init_from} (fresh optimizer, step 0)")
    init_sd = torch.load(args.init_from, map_location=device)
    init_sd = {k.removeprefix("_orig_mod."): v for k, v in init_sd.items()}
    model.load_state_dict(init_sd, strict=True, assign=True)
    del init_sd
if (resuming or args.init_from) and getattr(model.config, "tie_embeddings", False):
    model.lm_head.weight = model.model.embed_tokens.weight   # assign=True loads two separate Parameters: re-tie them

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12, arch="gpt") # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    **({"hyperball": True} if args.optimizer == "muonh" else {}),
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
data_phase = meta_data.get("data_phase", "A") if resuming else "A"
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict,
                                                                          data_dir=(args.phase_b_data_dir if data_phase == "B" else None))
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
# Token Superposition Training: the first tst_frac of steps read s*T tokens per row (udgam2/specs/tst.md)
tst_end_step = int(args.tst_frac * num_iterations) if args.tst_s > 1 else 0
loader_s = 1
if tst_end_step:
    print0(f"TST: bag size {args.tst_s} for steps [0, {tst_end_step}), then next-token training")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Step at which training data switches to Phase B (if --phase-b-data-dir is given)
phase_b_start_step = round(args.phase_b_start_frac * num_iterations) if args.phase_b_start_frac >= 0 else num_iterations - round(args.warmdown_ratio * num_iterations)
if args.phase_b_data_dir:
    print0(f"Phase B data switch at step {phase_b_start_step:,} of {num_iterations:,} ({args.phase_b_data_dir})")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
ema_params = None
if resuming and args.ema_start_frac >= 0:
    ema_path = os.path.join(checkpoint_dir, f"ema_{args.resume_from_step:06d}.pt")
    if os.path.exists(ema_path):
        ema_params = [t.to(device) for t in torch.load(ema_path, map_location=device)]
        print0(f"Resumed weight EMA from {ema_path}")
final_metrics = {}
gsm_dev_items = None
if args.gsm_dev_file:
    import json as _json
    gsm_dev_items = [(r["question"], str(r["gold"])) for r in map(_json.loads, open(args.gsm_dev_file))]
    print0(f"GSM dev set: {len(gsm_dev_items)} problems from {args.gsm_dev_file}")
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        extra_logs = {}
        for spec in args.extra_val:
            name, vdir = spec.split("=", 1)
            vl = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device, data_dir=vdir)
            with disable_fp8(model):
                extra_logs[f"val_{name}/bpb"] = evaluate_bpb(model, vl, eval_steps, token_bytes)
            print0(f"Step {step:05d} | {name} bpb: {extra_logs[f'val_{name}/bpb']:.6f}")
        final_metrics.update({"step": step, "val_bpb": val_bpb, **{k: v for k, v in extra_logs.items()}})
        if args.arith_probe:
            pr = udgam_loop.eval_probe(orig_model, tokenizer, device)
            final_metrics["arith_by_len"] = pr["by_len"]
            extra_logs["arith/practice"], extra_logs["arith/longer"] = pr["practice"], pr["longer"]
            print0(f"Step {step:05d} | arith probe practice {pr['practice']:.3f} longer {pr['longer']:.3f} | by_len {pr['by_len']}")
        if extra_logs:
            final_metrics.update(extra_logs)
            wandb_run.log({"step": step, **extra_logs})
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # Pocket Mathematician: held-out dev accuracy during training (master only) + a checkpoint at the same step
    if gsm_dev_items and args.gsm_dev_every > 0 and step > 0 and (step % args.gsm_dev_every == 0 or last_step) and master_process:
        from nanochat.gsm_eval import eval_gsm
        mode = args.gsm_eval if args.gsm_eval != "none" else "code"
        dv = eval_gsm(orig_model, tokenizer, device, mode=mode, items=gsm_dev_items)
        final_metrics.setdefault("gsm_dev_curve", []).append([step, dv["acc"]])
        print0(f"Step {step:05d} | GSM dev ({len(gsm_dev_items)}, {mode}): {dv['acc']}%")
        wandb_run.log({"step": step, "gsm_dev/acc": dv["acc"]})

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "data_phase": data_phase,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        if ema_params is not None and master_process:  # persist the weight EMA so a resumed run keeps averaging
            torch.save([e.cpu() for e in ema_params], os.path.join(checkpoint_dir, f"ema_{step:06d}.pt"))
        if is_ddp_initialized():
            dist.barrier()  # every rank has written its optimizer shard
        if master_process:  # marker for external uploaders: this step's checkpoint is complete on disk
            open(os.path.join(checkpoint_dir, f"saved_{step:06d}.ok"), "w").close()

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step and ema_params is not None:
        with torch.no_grad():
            backup = [p.detach().clone() for p in orig_model.parameters()]
            for p, e in zip(orig_model.parameters(), ema_params):
                p.copy_(e.to(p.dtype))
            model.eval()
            with disable_fp8(model):
                ema_bpb = evaluate_bpb(model, build_val_loader(), args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size), token_bytes)
                ema_logs = {"val_ema/bpb": ema_bpb}
                for spec in args.extra_val:
                    name, vdir = spec.split("=", 1)
                    vl = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device, data_dir=vdir)
                    ema_logs[f"val_ema_{name}/bpb"] = evaluate_bpb(model, vl, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size), token_bytes)
            if args.gsm_eval != "none" and master_process:
                from nanochat.gsm_eval import eval_gsm
                g = eval_gsm(orig_model, tokenizer, device, mode=args.gsm_eval, n=args.gsm_eval_n)
                ema_logs["gsm8k_ema"] = g["acc"]
                final_metrics["gsm8k_ema_samples"] = g["samples"]
                print0(f"GSM8K (EMA weights, {args.gsm_eval}): {g['acc']}%")
            if args.arith_probe:
                pr = udgam_loop.eval_probe(orig_model, tokenizer, device)
                ema_logs["arith_ema/practice"], ema_logs["arith_ema/longer"] = pr["practice"], pr["longer"]
                final_metrics["arith_ema_by_len"] = pr["by_len"]
            final_metrics.update(ema_logs)
            print0("EMA weights: " + " | ".join(f"{k} {v:.6f}" for k, v in ema_logs.items()))
            wandb_run.log({"step": step, **ema_logs})
            if master_process:
                torch.save(orig_model.state_dict(), os.path.join(checkpoint_dir, f"model_ema_{step:06d}.pt"))
            for p, b in zip(orig_model.parameters(), backup):
                p.copy_(b)
    if last_step:
        if args.gsm_eval != "none" and master_process:
            from nanochat.gsm_eval import eval_gsm
            g = eval_gsm(orig_model, tokenizer, device, mode=args.gsm_eval, n=args.gsm_eval_n)
            final_metrics["gsm8k_final"], final_metrics["gsm8k_final_samples"] = g["acc"], g["samples"]
            print0(f"GSM8K (final weights, {args.gsm_eval}): {g['acc']}%")
        if args.results_json and master_process:
            import json as _json
            final_metrics.update({"num_iterations": num_iterations, "flops_per_token": num_flops_per_token,
                                  "total_batch_size": total_batch_size, "args": user_config,
                                  "params": orig_model.num_scaling_params() if hasattr(orig_model, "num_scaling_params") else None,
                                  "total_training_time": total_training_time})
            with open(args.results_json, "w") as _f:
                _json.dump(final_metrics, _f, indent=1, default=str)
            print0(f"wrote {args.results_json}")
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    if args.phase_b_data_dir and data_phase == "A" and step >= phase_b_start_step:
        data_phase = "B"
        print0(f"Step {step:05d} | switching training data to Phase B shards in {args.phase_b_data_dir}")
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, data_dir=args.phase_b_data_dir)
        x, y, dataloader_state_dict = next(train_loader)
        wandb_run.log({"step": step, "data_phase_b_start": 1})
    cur_s = args.tst_s if step < tst_end_step else 1
    if cur_s != loader_s:
        print0(f"Step {step:05d} | TST: data rows now {args.max_seq_len * cur_s} tokens (bag size {cur_s})")
        loader_s = cur_s
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len * cur_s, split="train", device=device,
                                                                                  resume_state_dict=dataloader_state_dict, data_dir=(args.phase_b_data_dir if data_phase == "B" else None))
        x, y, dataloader_state_dict = next(train_loader)
        wandb_run.log({"step": step, "tst_bag_size": cur_s})
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y, tst_s=cur_s) if cur_s > 1 else model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get("hyperball"):   # MuonH: linear decay to 0 over the whole run, no warmup (udgam2/specs/muonh.md)
            group["lr"] = group["initial_lr"] * max(0.0, 1.0 - step / num_iterations)
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    if args.ema_start_frac >= 0 and step >= int(args.ema_start_frac * num_iterations):
        with torch.no_grad():
            if ema_params is None:
                ema_params = [p.detach().float().clone() for p in orig_model.parameters()]
            else:
                for e, p in zip(ema_params, orig_model.parameters()):
                    e.lerp_(p.detach().float(), 1 - args.ema_beta)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
