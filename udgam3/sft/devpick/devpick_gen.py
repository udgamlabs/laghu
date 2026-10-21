"""Generate both candidates' replies on the dev sets (DEV_RULE.md), on the Colab VM, in the pinned harness env.

  python devpick_gen.py --model NAME=DIR [--model ...] --chat dev_chat.jsonl --gsm dev_gsm8k.jsonl --out OUT_DIR

Greedy, bf16, each model's own chat template rendered as text then tokenized without extra special tokens; stop at
<turn|> / <eos>; caps: chat 1,024 (thinking off) / 2,048 (thinking on), maths 2,048. Left padding with an attention mask,
prompts batched by similar length under a token budget. The first 20 maths items are also generated one at a time
(the scorer check). Writes OUT_DIR/<name>_{chat,gsm,gsm_ref}.jsonl with the raw text (special tokens kept).
"""
import argparse, json, os, time
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True); ap.add_argument("--chat", required=True)
ap.add_argument("--gsm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, default=262144, help="max (prompt + cap) tokens per batch")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cap-override", type=int, default=0, help="tests only")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
load = lambda p: [json.loads(l) for l in open(p)][: a.limit or None]
CHAT, GSM = load(a.chat), load(a.gsm)


def run(tok, m, items, caps, batched=True):
    v = tok.get_vocab(); stops = [v["<turn|>"], v["<eos>"]]
    enc = []
    for it, cap in zip(items, caps):
        text = tok.apply_chat_template(it["messages"], tools=it.get("tools") or None, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=it["think"])
        enc.append((tok(text, add_special_tokens=False)["input_ids"], a.cap_override or cap))
    order = sorted(range(len(items)), key=lambda i: (enc[i][1], len(enc[i][0])))
    out = [None] * len(items); i = 0
    while i < len(order):
        L, cap = len(enc[order[i]][0]), enc[order[i]][1]
        j = i
        while j < len(order) and enc[order[j]][1] == cap and (j - i + 1) * (len(enc[order[j]][0]) + cap) <= a.budget and (j - i) < 64:
            j += 1
        j = max(j, i + 1) if batched else i + 1
        grp = order[i:j]; mx = max(len(enc[k][0]) for k in grp)
        ids = torch.full((len(grp), mx), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(grp), mx), dtype=torch.long)
        for r, k in enumerate(grp):
            e = enc[k][0]; ids[r, mx - len(e):] = torch.tensor(e); att[r, mx - len(e):] = 1
        with torch.no_grad():
            g = m.generate(input_ids=ids.to(DEV), attention_mask=att.to(DEV), max_new_tokens=cap, do_sample=False,
                           eos_token_id=stops, pad_token_id=tok.pad_token_id)
        for r, k in enumerate(grp):
            new = g[r, mx:].tolist()
            while new and new[-1] == tok.pad_token_id:
                new.pop()
            stopped = bool(new) and new[-1] in stops
            out[k] = {"id": items[k]["id"], "raw": tok.decode(new, skip_special_tokens=False), "n_new": len(new),
                      "hit_cap": (not stopped) and len(new) >= cap, "cap": cap, "prompt_tokens": len(enc[k][0])}
        i = j
    return out


for spec in a.model:
    name, path = spec.split("=", 1)
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
    for tag, items, caps, batched in (("chat", CHAT, [2048 if x["think"] else 1024 for x in CHAT], True),
                                      ("gsm", GSM, [2048] * len(GSM), True),
                                      ("gsm_ref", GSM[:20], [2048] * min(20, len(GSM)), False)):
        t0 = time.time()
        res = run(tok, m, items, caps, batched)
        with open(os.path.join(a.out, f"{name}_{tag}.jsonl"), "w") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name} {tag}: {len(res)} items in {time.time() - t0:.0f}s, hit cap {sum(r['hit_cap'] for r in res)}", flush=True)
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("GEN_OK")
