"""GSM8K-test scoring for small from-scratch maths models (Pocket Mathematician falsifier, 2026-10-14).

Prompt formats match the falsifier training data (udgam-loop/falsifier/prep_data.py):
  text: "Question: {q}\\nAnswer:"                   -> reasoning ending in \\boxed{N} (or a last number)
  code: "Question: {q}\\nAnswer (Python):\\n```python\\n" -> a program defining solution(); we run it
Greedy decoding without a KV cache (fine for <=100M models), batched by IDENTICAL prompt length (no padding). With little-endian
training data the prompt goes through the same digit reverser and the output is reversed back before scoring.
"""
import multiprocessing as mp
import re

import torch

from nanochat import udgam_loop


def _boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    d, k = 0, j
    while 0 <= j < len(s) and k < len(s):
        d += {"{": 1, "}": -1}.get(s[k], 0)
        if d == 0:
            return s[j + 1:k]
        k += 1
    return None


def _last_num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", "").replace("$", ""))
    return m[-1] if m else None


def _exec_solution(code, q):
    try:
        env = {}
        exec(code, {"__builtins__": __builtins__, "math": __import__("math")}, env)
        fn = env.get("solution") or next((v for k, v in env.items() if callable(v)), None)
        q.put(fn() if fn else None)
    except Exception:
        q.put(None)


def run_code(code, timeout=3.0):
    """Run model-written code in a separate process with a timeout (pod only; model trained on TinyGSM programs)."""
    q = mp.get_context("fork").Queue()
    p = mp.get_context("fork").Process(target=_exec_solution, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


def _cut(text):
    """The answer ends where the model starts a new document/question (it often continues with '<bos>Question: ...')."""
    for marker in ("<bos>", "\nQuestion:", "Question:"):
        i = text.find(marker)
        if i > 0:
            text = text[:i]
    return text


def _stop(text, mode):
    return ("```" in text) if mode == "code" else ("<bos>" in text or "\nQuestion:" in text)


@torch.no_grad()
def eval_gsm(model, tokenizer, device, mode="text", n=0, max_new=320, batch=128, items=None):
    """items: optional list of (question, gold); default = GSM8K-test."""
    if items is None:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]
    if n:
        items = items[:n]
    rev = udgam_loop.CONFIG["reverser"]
    bos = tokenizer.get_bos_token_id()
    suffix = "\nAnswer (Python):\n```python\n" if mode == "code" else "\nAnswer:"
    prompts = []
    for q, _ in items:
        ids = tokenizer.encode(f"Question: {q}{suffix}", prepend=bos)
        prompts.append(list(rev(ids)) if rev is not None else ids)
    was = model.training
    model.eval()
    outs = [None] * len(items)
    # Batch ONLY prompts of exactly the same length, so there is no padding. (Until 2026-10-15 prompts were left-padded
    # with <bos> and no attention mask; that understated S2 badly: dev 17.5% padded vs ~60% unpadded on the same items.)
    by_len = {}
    for i in sorted(range(len(items)), key=lambda i: len(prompts[i])):
        by_len.setdefault(len(prompts[i]), []).append(i)
    chunks = [idx[s:s + batch] for idx in by_len.values() for s in range(0, len(idx), batch)]
    patched = hasattr(type(model), "_logits")
    if patched:   # vocabulary scores for the last position only: exact, and far less memory
        full = type(model)._logits.__get__(model)
        model._logits = lambda x: full(x[:, -1:])
    try:
        for chunk in chunks:
            L = len(prompts[chunk[0]])
            seq = torch.tensor([prompts[i] for i in chunk], device=device)
            for step in range(max_new):
                nxt = model(seq)[:, -1].argmax(-1)
                seq = torch.cat([seq, nxt[:, None]], 1)
                if step % 32 == 31:
                    texts = [tokenizer.decode(list(rev(r)) if rev is not None else r) for r in seq[:, L:].tolist()]
                    if all(_stop(t, mode) for t in texts):
                        break
            for j, i in enumerate(chunk):
                gen = seq[j, L:].tolist()
                outs[i] = tokenizer.decode(list(rev(gen)) if rev is not None else gen)
    finally:
        if patched:
            del model._logits
    if was:
        model.train()
    correct, samples = 0, []
    for (q, gold), out in zip(items, outs):
        if mode == "code":
            code = out.split("```")[0]
            val = run_code(code)
            pred = None if val is None else str(val)
        else:
            text = _cut(out)
            b = _boxed(text)
            pred = _last_num(b) if b is not None else _last_num(text)
        try:
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-4
        except (TypeError, ValueError):
            ok = False
        correct += ok
        if len(samples) < 8:
            samples.append(dict(q=q[:200], gold=gold, pred=pred, out=out[:600]))
    return dict(acc=round(100 * correct / len(items), 2), n=len(items), samples=samples)
