"""Laghu RL (plan: udgam3/rl/RL_PLAN.md). On-policy GRPO with verifiable rewards, optional on-policy distillation from a
teacher with the same tokenizer (Gargi), a small KL to the SFT model, and loop/length penalties. Adapted from
udgam/eval/grpo_train.py (Gargi's RL).

  python laghu_rl.py --init SFT_HF --name RUN --pool pool.jsonl --out-root OUT --steps N [--teacher GARGI_HF --opd 1.0]
                     [--modes think:0.85,plain:0.15 | tool:1.0] [--pps 32 --k 8 --lr 5e-6 --beta 0.02 --max-new 768]

Each step: `pps` prompts x `k` samples (T = 1.0). Per sequence: reward (rollout.reward) -> group-normalised advantage
A = (r - mean) / (std + 1e-4); groups with identical rewards carry no GRPO signal. With --opd > 0 every model token also
gets a distillation term a_t = opd * (log p_teacher - log p_student) (per-token reverse KL, detached), so groups without
GRPO signal still teach. Loss per model token (tool outputs masked): -(A + a_t) * log p + beta * KL_k3(ref || policy),
summed and divided by the number of model tokens. Policy fp32 (AdamW, grad clip 1.0), bf16 autocast; the sampler is a bf16
copy refreshed every step (on-policy); ref = frozen SFT model; teacher = frozen bf16.
Logs one JSON line per step to OUT/RUN.log.jsonl (also printed); HF snapshots every --save-every steps and at the end.
"""
import argparse, json, os, random, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rollout as RO  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--name", required=True); ap.add_argument("--pool", required=True)
ap.add_argument("--out-root", required=True); ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--modes", default="think:0.85,plain:0.15"); ap.add_argument("--pps", type=int, default=32)
ap.add_argument("--k", type=int, default=8); ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--beta", type=float, default=0.02); ap.add_argument("--max-new", type=int, default=768)
ap.add_argument("--teacher", default=""); ap.add_argument("--opd", type=float, default=0.0)
ap.add_argument("--ref", default="", help="KL reference (default: --init); the plan keeps the SFT model as the leash in every stage")
ap.add_argument("--save-every", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mb", type=int, default=8, help="sequences per forward/backward micro-batch")
ap.add_argument("--start-offset", type=int, default=0, help="first pool index (later stages continue after earlier ones)")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
torch.manual_seed(a.seed); rng = random.Random(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ac = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
tok = AutoTokenizer.from_pretrained(a.init)
load = lambda p, dt: AutoModelForCausalLM.from_pretrained(p, dtype=dt, attn_implementation="sdpa").to(dev)
pol = load(a.init, torch.float32); pol.gradient_checkpointing_enable()
ref = load(a.ref or a.init, torch.bfloat16).eval() if a.beta > 0 else None
gen = load(a.init, torch.bfloat16).eval()
tea = load(a.teacher, torch.bfloat16).eval() if (a.teacher and a.opd > 0) else None
for m in (ref, gen, tea):
    if m is not None:
        for q in m.parameters():
            q.requires_grad_(False)
opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
pad = tok.pad_token_id

modes = [(m.split(":")[0], float(m.split(":")[1])) for m in a.modes.split(",")]
pool = [json.loads(l) for l in open(a.pool)][a.start_offset:]
need = a.steps * a.pps
assert len(pool) >= need, f"pool has {len(pool)} prompts, the run needs {need} (steps x pps)"
os.makedirs(a.out_root, exist_ok=True)
logf = open(f"{a.out_root}/{a.name}.log.jsonl", "a")


def pick_mode():
    x, acc = rng.random(), 0.0
    for m, w in modes:
        acc += w
        if x < acc:
            return m
    return modes[-1][0]


def token_logp(model, ids, att, keep):
    lg = model(input_ids=ids, attention_mask=att, logits_to_keep=keep).logits[:, :-1].float()
    tgt = ids[:, ids.shape[1] - keep + 1:]
    return lg.gather(-1, tgt[..., None])[..., 0] - torch.logsumexp(lg, -1)


def save(d):
    pol.save_pretrained(d, safe_serialization=True); tok.save_pretrained(d)
    for f in ("chat_template.jinja", "generation_config.json", "tokenizer_config.json"):
        if os.path.exists(f"{a.init}/{f}"):
            os.system(f"cp '{a.init}/{f}' '{d}/{f}'")


print(json.dumps({"config": vars(a), "pool": len(pool), "device": dev}), flush=True)
for step in range(a.steps):
    t0 = time.time()
    items = [dict(x, mode=pick_mode()) for x in pool[step * a.pps:(step + 1) * a.pps]]
    with torch.no_grad():                                   # on-policy: sampler = bf16 copy of the current policy
        for gp, pp in zip(gen.parameters(), pol.parameters()):
            gp.copy_(pp.detach().to(gp.dtype))
    R = RO.rollouts(gen, tok, items, k=a.k, max_new=a.max_new, sample=True, temperature=1.0)
    gen_s = time.time() - t0
    rew = [RO.reward(r, a.max_new) for r in R]
    adv = [0.0] * len(R); informative = 0
    for g0 in range(0, len(R), a.k):
        rs = rew[g0:g0 + a.k]; mu = sum(rs) / a.k; sd = (sum((x - mu) ** 2 for x in rs) / a.k) ** 0.5
        if sd > 1e-6:
            informative += 1
            for i in range(g0, g0 + a.k):
                adv[i] = (rew[i] - mu) / (sd + 1e-4)
    use_opd = tea is not None
    used = [i for i in range(len(R)) if (adv[i] != 0.0 or use_opd) and sum(R[i]["gen_mask"]) > 0]
    ntok = sum(sum(R[i]["gen_mask"]) for i in used)
    st = {"kl": 0.0, "opd_rkl": 0.0, "pg": 0.0}
    pol.train()
    if ntok:
        opt.zero_grad(set_to_none=True)
        for m0 in range(0, len(used), a.mb):
            mb = used[m0:m0 + a.mb]
            L = max(len(R[i]["prompt_ids"]) + len(R[i]["gen_ids"]) for i in mb)
            keep = max(len(R[i]["gen_ids"]) for i in mb) + 1
            ids = torch.full((len(mb), L), pad, dtype=torch.long); att = torch.zeros_like(ids)
            msk = torch.zeros((len(mb), keep - 1)); A = torch.zeros(len(mb))
            for r_, i in enumerate(mb):
                s = R[i]["prompt_ids"] + R[i]["gen_ids"]; o = L - len(s)
                ids[r_, o:] = torch.tensor(s); att[r_, o:] = 1; A[r_] = adv[i]
                g = R[i]["gen_mask"]; msk[r_, keep - 1 - len(g):] = torch.tensor(g, dtype=torch.float)
            ids, att, msk, A = ids.to(dev), att.to(dev), msk.to(dev), A.to(dev)
            with torch.autocast(**ac):
                lp = token_logp(pol, ids, att, keep)
                with torch.no_grad():
                    lr_ = token_logp(ref, ids, att, keep) if ref is not None else None
                    lt = token_logp(tea, ids, att, keep) if use_opd else None
            per = -A[:, None] * lp
            if lr_ is not None:
                kl = torch.exp(lr_ - lp) - (lr_ - lp) - 1
                per = per + a.beta * kl; st["kl"] += (kl * msk).sum().item() / ntok
            if lt is not None:
                at = (a.opd * (lt - lp)).detach()
                per = per - at * lp; st["opd_rkl"] += ((lp - lt).detach() * msk).sum().item() / ntok
            loss = (per * msk).sum() / ntok
            loss.backward(); st["pg"] += ((-A[:, None] * lp).detach() * msk).sum().item() / ntok
            del lp, lr_, lt, per, loss
        gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0).item()
        opt.step()
    else:
        gn = 0.0
    if dev == "cuda":
        torch.cuda.empty_cache()
    by = {}
    for r, x in zip(R, rew):
        by.setdefault(r["item"]["mode"], []).append((x, RO.correct(r["item"], r["text"])))
    rec = {"step": step + 1, "steps": a.steps, "reward": round(sum(rew) / len(rew), 4),
           "acc": round(sum(RO.correct(r["item"], r["text"]) for r in R) / len(R), 4),
           "informative_groups": round(informative / (len(R) // a.k), 3), "used_frac": round(len(used) / len(R), 3), "ntok": ntok,
           "cap_rate": round(sum(not r["stopped"] for r in R) / len(R), 4), "loop_rate": round(sum(r["loop"] for r in R) / len(R), 4),
           "mean_len": round(sum(r["n_model"] for r in R) / len(R), 1),
           "tool_calls": round(sum(r["tool_calls"] for r in R) / len(R), 3), "tool_errors": sum(r["tool_errors"] for r in R),
           **{f"acc_{m}": round(sum(c for _, c in v) / len(v), 4) for m, v in by.items()},
           "grad_norm": round(gn, 4), **{k_: round(v_, 5) for k_, v_ in st.items()}, "gen_s": round(gen_s, 1), "step_s": round(time.time() - t0, 1)}
    print("RL " + json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
    if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
        save(f"{a.out_root}/{a.name}/hf" if step + 1 == a.steps else f"{a.out_root}/{a.name}-step{step + 1}/hf")
print("RL_TRAIN_DONE", flush=True)
