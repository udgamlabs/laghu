#!/usr/bin/env python3
"""Quick base-model sanity samples from an exported HF folder (greedy, short). Writes JSON next to the model.
Usage: python pod_samples.py HF_DIR [OUT_JSON]"""
import sys, json, time, torch
from transformers import AutoTokenizer, Gemma4ForCausalLM

md = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"{md}/samples.json"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(md)
model = Gemma4ForCausalLM.from_pretrained(md, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()

plain = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "To list all files in the current directory, including hidden ones, run the shell command",
    '{"name": "Ada Lovelace", "born": 1815, "known_for": [',
]
tools = [{"type": "function", "function": {
    "name": "set_timer", "description": "Start a countdown timer on the phone.",
    "parameters": {"type": "object", "properties": {
        "minutes": {"type": "integer", "description": "Length of the timer in minutes"},
        "label": {"type": "string", "description": "Optional name for the timer"}}, "required": ["minutes"]}}}]
chats = [
    dict(messages=[{"role": "user", "content": "Set a timer for 15 minutes for the pasta."}], tools=tools),
    dict(messages=[{"role": "user", "content": "Extract the date and amount from: 'Invoice #442, due 3 March 2026, total $1,250.00'. Answer as JSON."}], tools=None),
    dict(messages=[{"role": "user", "content": "Find all .log files larger than 10 MB under /var/log."}], tools=None),
]


def gen(ids, n=80):
    ids = torch.tensor([ids], device=dev)
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)


res = []
t0 = time.time()
for p in plain:
    ids = [tok.bos_token_id] + tok(p, add_special_tokens=False).input_ids
    res.append({"prompt": p, "completion": gen(ids, 60)})
for c in chats:
    text = tok.apply_chat_template(c["messages"], tools=c["tools"], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False).input_ids
    if ids[0] != tok.bos_token_id:
        ids = [tok.bos_token_id] + ids
    res.append({"prompt": text, "completion": gen(ids, 100)})
for r in res:
    print("=" * 80, "\nPROMPT:", r["prompt"][-300:], "\n--> ", r["completion"], flush=True)
json.dump({"model": md, "seconds": round(time.time() - t0, 1), "samples": res}, open(out, "w"), indent=1)
