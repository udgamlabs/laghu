"""
Export a nanochat Gemma4LM checkpoint to a Hugging Face `Gemma4ForCausalLM` folder
(safetensors + config + Gemma-4-style tokenizer + chat template). That folder is what we
convert to GGUF (llama.cpp / Ollama / LM Studio) and MLX, and what we publish.

Usage:
  python -m scripts.export_hf --checkpoint-dir DIR [--step N] --tokenizer-dir DIR --out DIR
      [--chat-template PATH] [--dtype bfloat16|float32] [--verify]
--verify reloads the folder with transformers and checks logits against our model (fp32, CPU).
"""
import os, json, glob, re, shutil, argparse
# verify compares fp32 math on CPU: without this, nanochat picks bf16 activations on any CUDA box (COMPUTE_DTYPE) and the
# check "fails" (~7e-2) on GPU containers even though the export is correct (the base-v1 L4 export showed exactly that)
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
import torch
from safetensors.torch import save_file

from nanochat.gemma4 import Gemma4Config, Gemma4LM
import nanochat.flash_attention as _fa
_fa.USE_FA3 = False  # export/verify runs on CPU in fp32: use the SDPA path even on a GPU box

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint-dir", required=True)
ap.add_argument("--step", type=int, default=-1, help="-1 = latest")
ap.add_argument("--tokenizer-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chat-template", default=os.path.join(os.path.dirname(__file__), "../../tools/ref/gemma-4-E2B-it/chat_template.jinja"))
ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
ap.add_argument("--verify", action="store_true")
ap.add_argument("--max-positions", type=int, default=0, help="max_position_embeddings / model_max_length written to the config (llama.cpp context_length); 0 = 4x the training sequence length (old default)")
ap.add_argument("--weights", default="final", choices=["final", "ema"], help="ema = the end-of-training weight average (model_ema_<step>.pt)")
a = ap.parse_args()

step = a.step
if step < 0:
    step = max(int(m.group(1)) for p in glob.glob(f"{a.checkpoint_dir}/model_*.pt") if (m := re.search(r"model_(\d+)\.pt$", p)))
meta = json.load(open(f"{a.checkpoint_dir}/meta_{step:06d}.json"))
cfg = Gemma4Config(**meta["model_config"])
sd = torch.load(f"{a.checkpoint_dir}/model_{'ema_' if a.weights == 'ema' else ''}{step:06d}.pt", map_location="cpu")
sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
model = Gemma4LM(cfg)
model.to_empty(device="cpu"); model.init_weights()   # buffers (rope tables)
model.load_state_dict(sd, strict=True)
model.float().eval()

tokj = json.load(open(f"{a.tokenizer_dir}/tokenizer.json"))
vocab = tokj["model"]["vocab"]; vocab.update({t["content"]: t["id"] for t in tokj["added_tokens"]})
ids = {k: vocab[k] for k in ["<pad>", "<eos>", "<bos>", "<turn|>", "<|tool_response>"]}

os.makedirs(a.out, exist_ok=True)
dt = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
hf_sd = {k: v.to(dt) for k, v in model.hf_state_dict().items()}
save_file(hf_sd, f"{a.out}/model.safetensors", metadata={"format": "pt"})
conf = model.hf_config(bos_id=ids["<bos>"], eos_id=ids["<eos>"], pad_id=ids["<pad>"])
conf["dtype"] = a.dtype
if a.max_positions:
    conf["max_position_embeddings"] = a.max_positions
json.dump(conf, open(f"{a.out}/config.json", "w"), indent=2)
json.dump({"bos_token_id": ids["<bos>"], "pad_token_id": ids["<pad>"],
           "eos_token_id": [ids["<eos>"], ids["<turn|>"], ids["<|tool_response>"]],
           "do_sample": True, "temperature": 0.7, "top_p": 0.95, "top_k": 64},
          open(f"{a.out}/generation_config.json", "w"), indent=2)
# 2026-10-13: tokenizer_v1's tokenizer.json has no post-processor, and PreTrainedTokenizerFast ignores add_bos_token without
# one, so plain-text callers (lm-eval, raw completions) never got <bos>. The models are trained with <bos> on every
# document and lose 26-59% bpb without it. Add a TemplateProcessing rule: "<bos> $A". Chat templates already start
# with <bos>, and apply_chat_template / vLLM / llama.cpp chat paths tokenize with add_special_tokens=False (no double).
tok_out = json.load(open(f"{a.tokenizer_dir}/tokenizer.json"))
tok_out["post_processor"] = {"type": "TemplateProcessing",
    "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
    "special_tokens": {"<bos>": {"id": "<bos>", "ids": [ids["<bos>"]], "tokens": ["<bos>"]}}}
json.dump(tok_out, open(f"{a.out}/tokenizer.json", "w"), ensure_ascii=False)
template = open(a.chat_template).read() if os.path.exists(a.chat_template) else None
tcfg = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
        "unk_token": "<unk>", "mask_token": "<mask>", "add_bos_token": True, "add_eos_token": False,
        "model_max_length": conf["max_position_embeddings"], "padding_side": "left", "clean_up_tokenization_spaces": False,
        # Gemma 4 chat/tool-call special tokens (llama.cpp & Ollama Gemma 4 parsers rely on these strings)
        "sot_token": "<|turn>", "eot_token": "<turn|>", "std_token": "<|tool>", "etd_token": "<tool|>",
        "stc_token": "<|tool_call>", "etc_token": "<tool_call|>", "str_token": "<|tool_response>",
        "etr_token": "<tool_response|>", "escape_token": '<|"|>', "think_token": "<|think|>",
        "soc_token": "<|channel>", "eoc_token": "<channel|>"}
if template:
    tcfg["chat_template"] = template
    open(f"{a.out}/chat_template.jinja", "w").write(template)
json.dump(tcfg, open(f"{a.out}/tokenizer_config.json", "w"), indent=2)
print(f"exported step {step} [{a.weights} weights] ({sum(v.numel() for v in hf_sd.values())/1e6:.1f}M params, {a.dtype}) -> {a.out}")

if a.verify:
    from transformers import Gemma4ForCausalLM
    hf = Gemma4ForCausalLM.from_pretrained(a.out, dtype=torch.float32, attn_implementation="eager").eval()
    if a.dtype == "bfloat16":  # compare like with like: our model with the same bf16-rounded weights (still fp32 math)
        with torch.no_grad():
            for p_ in model.parameters():
                p_.copy_(p_.to(torch.bfloat16).float())
    x = torch.randint(0, cfg.vocab_size, (2, min(300, cfg.sequence_len)))
    with torch.no_grad():
        ours, theirs = model(x), hf(x).logits
    rel = ((ours - theirs).abs().max() / theirs.abs().max()).item()
    tol = 5e-4
    print(f"verify: ours vs HF max rel diff {rel:.1e} (tol {tol}) top1 agreement {(ours.argmax(-1) == theirs.argmax(-1)).float().mean():.4f} -> {'PASS' if rel < tol else 'FAIL'}")
