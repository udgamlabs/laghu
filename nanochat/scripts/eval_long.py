"""
Long-context evaluation of a Gemma4LM checkpoint (UdgamLabs 4K-context stage). Single GPU, bf16, FA3 if present.

  python -m scripts.eval_long --ckpt model_ema_007500.pt --meta meta_007500.json --out res.json \
      --val main=/data/mix_task11 --val web=/data/val_sets_rg64/val_web --val task=/data/val_sets_rg64/val_task \
      --long-val /data/cpt4k/val_long/val_long.parquet --gold /data/cpt4k/gold_long.jsonl

Reports
  1. bpb at seq 2048 on the held-out sets (same BOS-bestfit loader as training -> identical tokens for every model)
  2. per-position bpb on long held-out documents (no packing: one document per row, truncated to L = 4096 / 8192)
  3. needle test: "the access code for locker <name> is <5 digits>" early in a long context, asked at the end;
     teacher-forced exact match of the digits + mean log-prob; and a copy test (a 48-token random-word span from
     the start repeated at the end: next-token accuracy on its last 40 tokens)
  4. task-token loss on long gold examples (model-turn tokens, tool responses excluded), split by position < / >= 2048
"""
import os, re, json, math, random, argparse, time
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

from nanochat.common import COMPUTE_DTYPE
from nanochat.gemma4 import Gemma4Config, Gemma4LM
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val", action="append", default=[], help="NAME=DIR (last parquet in DIR is the eval set)")
ap.add_argument("--eval-tokens", type=int, default=2**23)
ap.add_argument("--long-val", default="")
ap.add_argument("--max-long-docs", type=int, default=400)
ap.add_argument("--gold", default="")
ap.add_argument("--needle-trials", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch-2048", type=int, default=32)
ap.add_argument("--long-batch", type=int, default=8, help="docs per forward at L=4096 (scaled down for longer L, min 1)")
ap.add_argument("--long-lengths", default="4096,8192", help="per-position bpb window lengths L")
ap.add_argument("--needle-lengths", default="1024,2048,3584,7168")
ap.add_argument("--copy-lengths", default="1024,3584,7168")
ap.add_argument("--skip-2048", action="store_true", help="skip the seq-2048 val bpb (section 1)")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = get_tokenizer()
token_bytes = get_token_bytes(device=dev)
BOS = tok.get_bos_token_id()
meta = json.load(open(a.meta))
LONG_L = [int(x) for x in a.long_lengths.split(",") if x]; NEEDLE_L = [int(x) for x in a.needle_lengths.split(",") if x]
COPY_L = [int(x) for x in a.copy_lengths.split(",") if x]
mc = dict(meta["model_config"])
# rope tables cover 4x sequence_len: make them cover the longest test (an eval-only change; weights are unaffected)
mc["sequence_len"] = max(mc["sequence_len"], 4096, -(-max(LONG_L + NEEDLE_L + COPY_L + [0]) // 4) + 1)
model = Gemma4LM(Gemma4Config(**mc))
model.to_empty(device=dev); model.init_weights()
sd = torch.load(a.ckpt, map_location=dev)
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()}, strict=True)
del sd
model.eval()
res = {"ckpt": a.ckpt, "config": mc}
t0 = time.time()


@torch.no_grad()
def token_losses(ids_batch):
    """ids_batch: LongTensor (B, L+1) -> per-target nats (B, L) in fp32."""
    x, y = ids_batch[:, :-1].contiguous(), ids_batch[:, 1:].contiguous()
    return model(x, y, loss_reduction="none").view(y.shape).float(), y


# 1) bpb at seq 2048 ------------------------------------------------------------------------------------------
B2 = a.batch_2048
for spec in ([] if a.skip_2048 else a.val):
    name, d = spec.split("=", 1)
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, B2, 2048, split="val", device=dev, data_dir=d)
    with torch.no_grad():
        res[f"bpb2048_{name}"] = evaluate_bpb(model, loader, a.eval_tokens // (B2 * 2048), token_bytes)
    print(f"bpb@2048 {name}: {res[f'bpb2048_{name}']:.6f}", flush=True)

# 2) per-position bpb on long held-out documents --------------------------------------------------------------
docs = []
if a.long_val:
    t = pq.read_table(a.long_val)
    texts, srcs = t.column("text").to_pylist(), t.column("src").to_pylist()
    enc = tok.encode(texts, prepend=BOS)
    docs = [(s, e) for s, e in zip(srcs, enc)]
    edges = [0, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    for L, bsz in [(L, max(1, a.long_batch * 4096 // L)) for L in LONG_L]:
        sel = [(s, e[:L + 1]) for s, e in docs if len(e) >= L + 1][:a.max_long_docs]
        nats = torch.zeros(len(edges) - 1, dtype=torch.float64); byts = torch.zeros_like(nats)
        per_src = {}
        for i in range(0, len(sel), bsz):
            chunk = sel[i:i + bsz]
            ids = torch.tensor([e for _, e in chunk], dtype=torch.long, device=dev)
            loss, y = token_losses(ids)
            nb = token_bytes[y].float().cpu().double()
            ln = loss.cpu().double() * (nb > 0)
            for j in range(len(edges) - 1):
                lo, hi = edges[j], min(edges[j + 1], L)
                if lo >= L:
                    continue
                nats[j] += ln[:, lo:hi].sum().cpu(); byts[j] += nb[:, lo:hi].sum().cpu()
            for r, (s, _) in enumerate(chunk):   # per-source split: first half vs second half of the window
                d_ = per_src.setdefault(s, [0.0, 0.0, 0.0, 0.0, 0])
                d_[0] += ln[r, :L // 2].sum().item(); d_[1] += nb[r, :L // 2].sum().item()
                d_[2] += ln[r, L // 2:].sum().item(); d_[3] += nb[r, L // 2:].sum().item(); d_[4] += 1
        out = {f"{edges[j]}-{min(edges[j+1], L)}": round(nats[j].item() / math.log(2) / byts[j].item(), 6)
               for j in range(len(edges) - 1) if edges[j] < L and byts[j] > 0}
        out["n_docs"] = len(sel)
        out["per_src_first_half_second_half"] = {s: [round(v[0] / math.log(2) / max(v[1], 1), 5),
                                                     round(v[2] / math.log(2) / max(v[3], 1), 5), v[4]]
                                                 for s, v in per_src.items()}
        res[f"long_bpb_L{L}"] = out
        print(f"long docs L={L}: {out}", flush=True)

# 3) needle + copy tests ---------------------------------------------------------------------------------------
if docs:
    rng = random.Random(a.seed)
    filler_pool = [e[1:] for _, e in docs]   # strip BOS; plain document tokens
    colors = ["red", "blue", "green", "amber", "violet", "silver", "orange", "teal", "black", "white"]
    animals = ["otter", "falcon", "badger", "heron", "lynx", "walrus", "gecko", "bison", "raven", "salmon"]
    words = "apple river stone cloud pencil garden window ladder candle mirror basket rocket velvet copper forest \
harbor meadow lantern marble saddle thunder violin wagon zebra anchor bridge canyon desert engine feather glacier \
island jungle kettle lemon magnet needle orchard pepper quarry ribbon summit tunnel umbrella valley whistle yogurt".split()

    def filler(n):
        out = []
        while len(out) < n:
            out += rng.choice(filler_pool)[:n - len(out)] + tok.encode("\n\n")
        return out[:n]

    needle = {}
    for total in NEEDLE_L:
        hits, lps = 0, []
        for _ in range(a.needle_trials):
            name = f"{rng.choice(colors)} {rng.choice(animals)}"
            key = f"{rng.randint(10000, 99999)}"
            pre = filler(64)
            ans = tok.encode(f" {key}")   # the same key tokens in the needle and in the answer
            ndl = (tok.encode(f"\n\nImportant: the access code for the {name} locker is") + ans
                   + tok.encode(". Remember it.\n\n"))
            q = tok.encode(f"\n\nQuestion: What is the access code for the {name} locker?\nAnswer: The access code for the {name} locker is")
            body = filler(max(0, total - 1 - len(pre) - len(ndl) - len(q) - len(ans)))
            ids = [BOS] + pre + ndl + body + q + ans
            x = torch.tensor([ids], device=dev)
            with torch.no_grad():
                logits = model(x[:, :-1])[0].float()
            k = len(ans)
            tgt = x[0, -k:]
            lg = logits[-k:]
            hits += int((lg.argmax(-1) == tgt).all().item())
            lps.append(F.log_softmax(lg, -1).gather(1, tgt[:, None]).mean().item())
        needle[str(total)] = {"exact": hits / a.needle_trials, "mean_logprob_key_token": round(sum(lps) / len(lps), 4),
                              "distance_tokens_approx": total - 64 - 40}
        print(f"needle ctx {total}: {needle[str(total)]}", flush=True)
    res["needle"] = needle

    copy = {}
    for total in COPY_L:
        accs = []
        for _ in range(a.needle_trials // 2):
            span = tok.encode(" " + " ".join(rng.choice(words) for _ in range(60)))[:48]
            pre = filler(32)
            body = filler(max(0, total - 1 - len(pre) - 2 * len(span)))
            ids = [BOS] + pre + span + body + span
            x = torch.tensor([ids], device=dev)
            with torch.no_grad():
                logits = model(x[:, :-1])[0].float()
            p = logits[-40:].argmax(-1)          # predictions for the last 40 tokens of the repeated span
            accs.append((p == x[0, -40:]).float().mean().item())
        copy[str(total)] = round(sum(accs) / len(accs), 4)
        print(f"copy ctx {total}: next-token acc {copy[str(total)]}", flush=True)
    res["copy"] = copy

# 4) long gold examples: task-token loss ------------------------------------------------------------------------
if a.gold:
    from tokenizers import Tokenizer
    htok = Tokenizer.from_file(os.path.join(os.environ["NANOCHAT_TOKENIZER_DIR"], "tokenizer.json"))
    sums = {"lt2048": [0.0, 0], "ge2048": [0.0, 0], "all": [0.0, 0]}
    per_ex, n_ex = [], 0
    for line in open(a.gold):
        r = json.loads(line)
        text = r["text"]
        spans = []
        for m in re.finditer(r"<\|turn>model\n(.*?)(?:<turn\|>|\Z)", text, re.S):   # a turn may end at the <|tool_response> stop token
            s, e = m.start(1), m.end(0)
            inner = [(s, e)]
            # tool output is not the model's: exclude it, but keep the <|tool_response> token the model emits to stop
            for tr in re.finditer(r"<\|tool_response>(.*?)(?:<tool_response\|>|\Z)", text[s:e], re.S):
                new = []
                for a_, b_ in inner:
                    ts, te = s + tr.start(1), s + tr.end()
                    if te <= a_ or ts >= b_:
                        new.append((a_, b_))
                    else:
                        if a_ < ts: new.append((a_, ts))
                        if te < b_: new.append((te, b_))
                inner = new
            spans += inner
        enc = htok.encode(text, add_special_tokens=False)
        ids = [BOS] + enc.ids
        if len(ids) - 1 > 4096 or len(ids) < 2049:
            continue
        mask = [False] + [any(s <= o[0] < e for s, e in spans) for o in enc.offsets]
        x = torch.tensor([ids], device=dev)
        loss, _ = token_losses(x)
        loss = loss[0].cpu()
        m = torch.tensor(mask[1:])                      # target j+1 mask for prediction at j
        pos = torch.arange(len(ids) - 1) + 1            # position of the target token
        for k_, sel in (("all", m), ("lt2048", m & (pos < 2048)), ("ge2048", m & (pos >= 2048))):
            sums[k_][0] += loss[sel].sum().item(); sums[k_][1] += int(sel.sum())
        per_ex.append(round(loss[m].mean().item(), 4) if m.any() else None)
        n_ex += 1
    res["gold_long"] = {k: {"nats_per_task_token": round(v[0] / max(v[1], 1), 5), "task_tokens": v[1]} for k, v in sums.items()}
    res["gold_long"]["n_examples"] = n_ex
    print(f"gold long ({n_ex} ex): {res['gold_long']}", flush=True)

res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
