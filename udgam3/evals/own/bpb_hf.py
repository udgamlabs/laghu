"""Bits per byte of any Hugging Face causal LM on a text set (Udgam-3 own benchmark: fresh-text loss).

Tokenizer-independent: total negative log-likelihood (in bits) of each document's tokens, divided by the document's
UTF-8 bytes. Each document is scored on its own (starting from the model's BOS token if it has one), with a sliding
window for documents longer than the context (stride = half the window; only the new tokens of each window count).
Per-document results are written so two models can be compared with a paired bootstrap.

  uv run --project evals/harness python udgam3/evals/own/bpb_hf.py MODEL OUT.json [--data udgam3/evals/own/fresh_v1/val.parquet]
      [--revision SHA] [--ctx 2048] [--device mps|cuda|cpu] [--limit N] [--trust-remote-code] [--add-bos auto|yes|no]
"""
import argparse, json, math, time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--data", default="udgam3/evals/own/fresh_v1/val.parquet")
ap.add_argument("--revision", default=None); ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--trust-remote-code", action="store_true")
ap.add_argument("--add-bos", default="auto", choices=["auto", "yes", "no"])
ap.add_argument("--min-chars", type=int, default=0, help="only documents at least this long (for testing the window)")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=a.trust_remote_code)
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, trust_remote_code=a.trust_remote_code,
                                             use_safetensors=True, dtype=torch.float32).to(a.device).eval()
bos = tok.bos_token_id
use_bos = (a.add_bos == "yes") or (a.add_bos == "auto" and bos is not None)
texts = pq.read_table(a.data, columns=["text"]).column("text").to_pylist()
if a.min_chars:
    texts = [t for t in texts if len(t) >= a.min_chars]
if a.limit:
    texts = texts[:a.limit]

t0 = time.time(); per_doc = []; tot_bits = 0.0; tot_bytes = 0
for i, text in enumerate(texts):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if use_bos:
        ids = [bos] + ids
    ids = torch.tensor(ids, device=a.device)
    nll = 0.0; scored = 0
    stride = a.ctx // 2
    start = 0
    with torch.no_grad():
        while True:
            end = min(start + a.ctx, len(ids))
            window = ids[start:end].unsqueeze(0)
            logits = model(window).logits[0, :-1].float()
            targets = window[0, 1:]
            lp = torch.log_softmax(logits, -1).gather(1, targets.unsqueeze(1)).squeeze(1)
            first_new = 0 if start == 0 else (a.ctx - stride) - 1   # only tokens not scored by the previous window
            nll += -lp[first_new:].sum().item(); scored += lp[first_new:].numel()
            if end == len(ids):
                break
            start += stride
    assert scored == len(ids) - 1, (i, scored, len(ids))   # every token after the first is scored exactly once
    bits = nll / math.log(2); nbytes = len(text.encode("utf-8"))
    per_doc.append({"i": i, "bits": bits, "bytes": nbytes, "tokens": scored})
    tot_bits += bits; tot_bytes += nbytes
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(texts)} docs, bpb so far {tot_bits/tot_bytes:.4f}, {time.time()-t0:.0f}s", flush=True)

res = {"model": a.model, "revision": a.revision, "data": a.data, "n_docs": len(texts), "bpb": tot_bits / tot_bytes,
       "bos": bool(use_bos), "ctx": a.ctx, "seconds": round(time.time() - t0, 1), "per_doc": per_doc}
json.dump(res, open(a.out, "w"))
print(json.dumps({k: v for k, v in res.items() if k != "per_doc"}))
