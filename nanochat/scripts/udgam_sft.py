"""
UdgamLabs SFT on pre-rendered conversations (udgam/sft/render_sft.py output: parquet with ids, mask, src, job, epochs).

Single GPU (the 384M model trains at ~170K tok/s on one H100). Differences from scripts/chat_sft.py:
  * data = our parquet (already rendered with the exported Gemma 4 chat template; mask = assistant tokens only),
  * the mix is given per source / source:job as (possibly fractional) epochs, plus optional general "replay" sized to a
    share of the loss tokens; the full packed schedule is built up front (best-fit packing, BOS-aligned rows, padding
    instead of cropping) so the step count and the LR schedule are exact,
  * the loss is the token-level mean over all trained tokens of the optimizer step (not a mean of micro-batch means),
  * val loss is reported per source (and per job for the specialist sources),
  * weights come from a state-dict file (--init-from) + its meta json (model_config); optional optimizer warm start.
Checkpoints: <out>/model_<step>.pt, meta_<step>.json (model_config for scripts/export_hf), optim_<step>_rank0.pt.

  python -m scripts.udgam_sft --train T.parquet --val V.parquet --init-from model.pt --init-meta meta.json \
      --mix '{"nemo_chat":1,"synth":1,"gen_*":2,"spec_gold":1}' --out DIR --run NAME
"""
import argparse, fnmatch, gc, json, math, os, random, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pyarrow.parquet as pq
import torch

from nanochat.common import COMPUTE_DTYPE, print0, get_peak_flops
from nanochat.gemma4 import Gemma4Config, Gemma4LM
from nanochat.flash_attention import HAS_FA3

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="parquet file(s), comma-separated (rows are concatenated)")
ap.add_argument("--val", required=True)
ap.add_argument("--init-from", required=True, help="state dict (.pt)")
ap.add_argument("--init-meta", required=True, help="meta json with model_config")
ap.add_argument("--init-optim", default="", help="optimizer state to warm-start momentum buffers (LRs are reset)")
ap.add_argument("--mix", required=True, help='JSON {pattern: epochs}; pattern = src or src:job, fnmatch globs ok; float epochs = repeat int part + random fraction')
ap.add_argument("--replay-frac", type=float, default=0.0, help="add general replay = this share of the final loss tokens")
ap.add_argument("--replay-srcs", default="nemo_chat,synth")
ap.add_argument("--out", required=True)
ap.add_argument("--run", default="dummy")
ap.add_argument("--seq", type=int, default=4096)
ap.add_argument("--device-batch", type=int, default=16)
ap.add_argument("--total-batch", type=int, default=262144)
ap.add_argument("--matrix-lr", type=float, default=0.0071)
ap.add_argument("--embedding-lr", type=float, default=0.106)
ap.add_argument("--unembedding-lr", type=float, default=0.0028)
ap.add_argument("--scalar-lr", type=float, default=0.177)
ap.add_argument("--warmup-frac", type=float, default=0.05)
ap.add_argument("--warmdown-frac", type=float, default=0.5)
ap.add_argument("--final-lr-frac", type=float, default=0.0)
ap.add_argument("--eval-every", type=int, default=40)
ap.add_argument("--save-at", default="1.0", help="comma list of progress fractions at which to save checkpoints")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N steps")
ap.add_argument("--new-token-ids", default="", help="e.g. 5-16: rows the base never trained (plain-text pretraining left the "
                "chat tokens identical); re-drawn from N(mean, cov) of the trained rows, so each starts as a distinct typical token")
a = ap.parse_args()
cfg_dump = vars(a).copy()

torch.manual_seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = random.Random(a.seed)
BOS = 2


# ----------------------------------------------------------------------------- data
def load(path):
    t = pq.read_table(path, columns=["id", "src", "job", "ids", "mask", "n_loss"])
    d = t.to_pydict()
    return [dict(id=d["id"][i], src=d["src"][i], job=d["job"][i], ids=np.asarray(d["ids"][i], dtype=np.int32),
                 mask=np.asarray(d["mask"][i], dtype=np.uint8), n_loss=d["n_loss"][i]) for i in range(len(d["id"]))]


def key_of(r):
    return f"{r['src']}:{r['job']}"


def epochs_for(r, mix):
    for pat, ep in mix.items():
        if fnmatch.fnmatch(key_of(r), pat if ":" in pat else pat + ":*"):
            return ep
    return 0.0


train_rows = [r for p in a.train.split(",") for r in load(p)]
val_rows = load(a.val)
mix = json.loads(a.mix)
order = []                       # indices into train_rows, with repeats
comp = {}
for i, r in enumerate(train_rows):
    ep = epochs_for(r, mix)
    n = int(ep) + (1 if rng.random() < ep - int(ep) else 0)
    order += [i] * n
    if n:
        c = comp.setdefault(key_of(r), [0, 0, 0]); c[0] += n; c[1] += n * len(r["ids"]); c[2] += n * r["n_loss"]
spec_loss = sum(train_rows[i]["n_loss"] for i in order)
if a.replay_frac > 0:
    srcs = set(a.replay_srcs.split(","))
    pool = [i for i, r in enumerate(train_rows) if r["src"] in srcs]
    rng.shuffle(pool)
    want = spec_loss * a.replay_frac / (1 - a.replay_frac)
    got = 0
    for i in pool:
        if got >= want:
            break
        order.append(i); got += train_rows[i]["n_loss"]
        c = comp.setdefault("replay:" + train_rows[i]["src"], [0, 0, 0]); c[0] += 1; c[1] += len(train_rows[i]["ids"]); c[2] += train_rows[i]["n_loss"]
rng.shuffle(order)
tot_tok = sum(len(train_rows[i]["ids"]) for i in order); tot_loss = sum(train_rows[i]["n_loss"] for i in order)
print0(f"mix: {len(order):,} examples, {tot_tok/1e6:.2f}M tokens, {tot_loss/1e6:.2f}M loss tokens")
for k in sorted(comp):
    print0(f"   {k:24s} n {comp[k][0]:7,d}  tok {comp[k][1]/1e6:7.2f}M  loss {comp[k][2]/1e6:6.2f}M ({100*comp[k][2]/tot_loss:5.1f}%)")

CAP = a.seq + 1


def pack(indices, rows, buffer_size=100):
    """Best-fit packing (nanochat sft_data_generator_bos_bestfit, deterministic): list of rows, each = list of indices."""
    out, buf, it = [], [], iter(indices)
    exhausted = False
    while True:
        while not exhausted and len(buf) < buffer_size:
            try:
                buf.append(next(it))
            except StopIteration:
                exhausted = True
        if not buf:
            break
        row, used = [], 0
        while True:
            rem = CAP - used
            best, bl = -1, 0
            for j, i in enumerate(buf):
                L = min(len(rows[i]["ids"]), CAP)
                if L <= rem and L > bl:
                    best, bl = j, L
            if best < 0:
                break
            row.append(buf.pop(best)); used += bl
            while not exhausted and len(buf) < buffer_size:
                try:
                    buf.append(next(it))
                except StopIteration:
                    exhausted = True
            if not buf:
                break
        out.append(row)
    return out


def tensors(packed_rows, rows):
    B = len(packed_rows)
    x = np.full((B, CAP), BOS, dtype=np.int64)
    m = np.zeros((B, CAP), dtype=np.int8)
    for b, pr in enumerate(packed_rows):
        p = 0
        for i in pr:
            ids, mk = rows[i]["ids"][:CAP - p], rows[i]["mask"][:CAP - p]
            x[b, p:p + len(ids)] = ids; m[b, p:p + len(ids)] = mk; p += len(ids)
    xt = torch.from_numpy(x)
    inp = xt[:, :-1].to(dev, dtype=torch.int32, non_blocking=True)
    tgt = xt[:, 1:].clone()
    tgt[torch.from_numpy(m[:, 1:]) == 0] = -1
    return inp, tgt.to(dev, non_blocking=True)


t0 = time.time()
packed = pack(order, train_rows)
rows_per_step = a.total_batch // a.seq
assert a.total_batch % (a.seq * a.device_batch) == 0
accum = rows_per_step // a.device_batch
steps = len(packed) // rows_per_step
fill = sum(min(len(train_rows[i]["ids"]), CAP) for r in packed for i in r) / (len(packed) * CAP)
print0(f"packed {len(packed):,} rows of {CAP} in {time.time()-t0:.0f}s (fill {fill:.3f}); {steps} steps x {rows_per_step} rows "
       f"({a.total_batch:,} tok), grad accum {accum}; dropping {len(packed) - steps*rows_per_step} tail rows")
if a.max_steps > 0:
    steps = min(steps, a.max_steps)

val_groups = {}
for i, r in enumerate(val_rows):
    k = r["src"] + (":" + r["job"] if r["src"].startswith("spec") else "")
    val_groups.setdefault(k, []).append(i)
val_groups = {k: v for k, v in val_groups.items() if len(v) >= 3}
val_packed = {k: pack(v, val_rows) for k, v in val_groups.items()}

# ----------------------------------------------------------------------------- model
meta0 = json.load(open(a.init_meta))
mc = dict(meta0["model_config"]); mc["sequence_len"] = a.seq
cfg = Gemma4Config(**mc)
with torch.device("meta"):
    model = Gemma4LM(cfg)
model.to_empty(device=dev); model.init_weights()
sd = torch.load(a.init_from, map_location=dev)
sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True, assign=True); del sd
if cfg.tie_embeddings:
    model.lm_head.weight = model.model.embed_tokens.weight   # assign=True loads two separate Parameters: re-tie them
if a.new_token_ids:
    new_ids = sorted({i for part in a.new_token_ids.split(",") for i in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)})
    with torch.no_grad():
        E = model.model.embed_tokens.weight
        ref = torch.ones(E.shape[0], dtype=torch.bool, device=E.device); ref[new_ids] = False
        # also leave out the other never-trained rows (unused specials and bytes: ~130 rows at cos > 0.999 to them)
        ref &= torch.nn.functional.cosine_similarity(E.float(), E[new_ids[0]].float()[None], dim=1) < 0.999
        X = E[ref].float(); mu = X.mean(0)
        C = torch.cov(X.T) + 1e-6 * torch.eye(X.shape[1], device=X.device)
        z = torch.randn(len(new_ids), X.shape[1], generator=torch.Generator().manual_seed(a.seed)).to(X.device)
        E[new_ids] = (mu + z @ torch.linalg.cholesky(C).T).to(E.dtype)
    print0(f"re-initialised {len(new_ids)} new-token rows {new_ids[0]}..{new_ids[-1]} from N(mean, cov) of {int(ref.sum()):,} "
           f"trained rows (tied head: {cfg.tie_embeddings})")
orig = model
model = torch.compile(model, dynamic=False)
opt = orig.setup_optimizer(unembedding_lr=a.unembedding_lr, embedding_lr=a.embedding_lr, matrix_lr=a.matrix_lr,
                           weight_decay=0.0, scalar_lr=a.scalar_lr)
if a.init_optim:
    base = [g["lr"] for g in opt.param_groups]
    opt.load_state_dict(torch.load(a.init_optim, map_location=dev))
    for g, lr in zip(opt.param_groups, base):
        g["lr"] = lr; g["initial_lr"] = lr
        if g["kind"] == "muon":
            g["momentum"] = 0.95
    print0(f"warm-started optimizer state from {a.init_optim} (LRs reset)")
print0(f"FA3: {HAS_FA3} | compute dtype {COMPUTE_DTYPE} | params {sum(p.numel() for p in orig.parameters())/1e6:.1f}M")

import wandb
wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "udgam-post"), entity=os.environ.get("WANDB_ENTITY", "udgamlabs"),
                name=a.run, id=os.environ.get("WANDB_RUN_ID"), resume="allow", config=cfg_dump | {"steps": steps}) if a.run != "dummy" else None


def lr_mult(step):
    p = step / max(steps, 1)
    if p < a.warmup_frac:
        return (step + 1) / max(1, int(a.warmup_frac * steps))
    if p <= 1 - a.warmdown_frac:
        return 1.0
    d = (p - (1 - a.warmdown_frac)) / a.warmdown_frac
    return (1 - d) + d * a.final_lr_frac


@torch.no_grad()
def evaluate():
    model.eval()
    res, all_s, all_n = {}, 0.0, 0
    for k, pr in val_packed.items():
        s, n = 0.0, 0
        for j in range(0, len(pr), a.device_batch):
            chunk = pr[j:j + a.device_batch]
            while len(chunk) < a.device_batch:      # fixed shapes for the compiled model
                chunk = chunk + [[]]
            x, y = tensors(chunk, val_rows)
            s += model(x, y, loss_reduction="sum").item(); n += int((y >= 0).sum())
        res[k] = s / max(n, 1); all_s += s; all_n += n
    res["all"] = all_s / max(all_n, 1)
    model.train()
    return res


def save(step):
    os.makedirs(a.out, exist_ok=True)
    torch.save(orig.state_dict(), f"{a.out}/model_{step:06d}.pt")
    torch.save(opt.state_dict(), f"{a.out}/optim_{step:06d}_rank0.pt")
    json.dump({"step": step, "model_config": mc, "user_config": cfg_dump, "init_from": a.init_from, "max_seq_len": a.seq,
               "composition": comp, "val": hist[-1] if hist else None}, open(f"{a.out}/meta_{step:06d}.json", "w"), indent=1)
    print0(f"saved step {step} -> {a.out}")


save_steps = {max(1, round(float(f) * steps)) for f in a.save_at.split(",")}
hist = []
peak = get_peak_flops(torch.cuda.get_device_name(0)) if dev.type == "cuda" else float("inf")
flops_tok = orig.estimate_flops()
v = evaluate(); hist.append({"step": 0, **v})
print0("step 0 val " + " | ".join(f"{k} {x:.4f}" for k, x in v.items()))
if wb:
    wb.log({"step": 0, **{f"val/{k}": x for k, x in v.items()}})
model.train()
t_train, smooth = 0.0, None
for step in range(steps):
    batch_rows = packed[step * rows_per_step:(step + 1) * rows_per_step]
    mbs = [tensors(batch_rows[k * a.device_batch:(k + 1) * a.device_batch], train_rows) for k in range(accum)]
    ntok = sum(int((y >= 0).sum()) for _, y in mbs)
    lm = lr_mult(step)
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"] * lm
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    tot = 0.0
    for x, y in mbs:
        loss = model(x, y, loss_reduction="sum") / ntok
        loss.backward(); tot += loss.item()
    opt.step(); model.zero_grad(set_to_none=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 3:
        t_train += dt
    smooth = tot if smooth is None else 0.9 * smooth + 0.1 * tot
    tps = a.total_batch / dt
    if step % 5 == 0 or step == steps - 1:
        print0(f"step {step+1:05d}/{steps} | loss {smooth:.4f} | lrm {lm:.3f} | trained tok {ntok:,} | dt {dt*1000:.0f}ms | "
               f"tok/s {tps:,.0f} | mfu {100*flops_tok*tps/peak:.1f}")
        if wb:
            wb.log({"step": step + 1, "train/loss": smooth, "train/lrm": lm, "train/tok_per_sec": tps, "train/loss_tokens": ntok})
    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()
    if (step + 1) % a.eval_every == 0 or step + 1 == steps or (step + 1) in save_steps:
        v = evaluate(); hist.append({"step": step + 1, **v})
        print0(f"step {step+1} val " + " | ".join(f"{k} {x:.4f}" for k, x in v.items()))
        if wb:
            wb.log({"step": step + 1, **{f"val/{k}": x for k, x in v.items()}})
    if (step + 1) in save_steps:
        save(step + 1)
print0(f"done: {steps} steps, train time {t_train/60:.2f} min, peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB" if dev.type == "cuda" else "done")
json.dump(hist, open(f"{a.out}/val_history.json", "w"), indent=1)
if wb:
    wb.finish()
