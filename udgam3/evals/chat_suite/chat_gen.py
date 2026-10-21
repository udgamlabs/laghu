"""Chat-suite generation for one model on the Colab G4 VM (PREREGISTRATION.md 2.3). Writes raw outputs only; scoring
runs on the Mac (score.py), so every number can be re-scored from the saved text.

  python chat_gen.py --name NAME --engine vllm|hf --manifest manifest.json --items items --out out
                     [--tasks gsm8k,math,ifeval,mt,loop,rec,ref] [--limit N] [--batch 192] [--gpu-util 0.5]

Tasks, each over the model's thinking modes ("off"/"on" = enable_thinking; "native" = no switch):
  gsm8k   GSM8K test (1,319), cap 4,096            math   MATH-500 levels 1-3 (238), cap 4,096
  ifeval  IFEval (541), cap 1,280                  mt     E1: 214 two-turn follow-ups, cap 1,024 per turn;
  loop    held-out chat prompts (324), cap 1,024,         E2: 200 two-turn GSM8K, cap 4,096, thinking mode only
          greedy and shipped sampling defaults     rec    GSM8K at a reasoning model's recommended cap
  ref     validation (2.4): the first 40 GSM8K items, one request at a time, cap 512
Every request uses the model's own chat template (rendered as text, then tokenized with add_special_tokens=False, as
lm-eval does), greedy unless stated, stop tokens = generation_config eos + tokenizer eos + the template's end-of-turn
token, and an output cap of min(cap, context - prompt length), recorded per item. The HF engine left-pads with an
attention mask (2.4.2) and batches prompts of similar length.
"""
import argparse, gzip, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CAPS, E2_N, MATH_SUFFIX, REF_N, SUFFIX, periodic_tail, ref_solution, sha1_ids, visible  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path)]


def special_ids(tok):
    s = set(tok.all_special_ids or [])
    for i, t in getattr(tok, "added_tokens_decoder", {}).items():
        if getattr(t, "special", False):
            s.add(int(i))
    return s


def end_of_turn_id(tok, specials):
    """The first special token after an assistant message in the model's own template (where its turn ends)."""
    try:
        probe = tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "XQZ"}], tokenize=False)
    except Exception:
        return None
    if "XQZ" not in probe:
        return None
    for t in tok(probe.split("XQZ", 1)[1], add_special_tokens=False)["input_ids"]:
        if t in specials:
            return t
    return None


class VllmEngine:
    def __init__(self, spec, max_len, gpu_util, stops, dtype="bfloat16"):
        import subprocess
        from vllm import LLM
        # other jobs share the GPU: never ask vLLM for more than is free at launch (nvidia-smi, so CUDA stays
        # uninitialised in this process before vLLM starts its engine)
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.split(",")
        free_frac = float(q[0]) / float(q[1])
        self.gpu_util = round(min(gpu_util, free_frac - 0.05), 3)
        print(f"vLLM gpu_memory_utilization {self.gpu_util} (asked {gpu_util}, free {free_frac:.2f})", flush=True)
        self.llm = LLM(model=spec["repo"], revision=spec["sha"], tokenizer=spec["repo"], tokenizer_revision=spec["sha"],
                       dtype=dtype, max_model_len=max_len, gpu_memory_utilization=self.gpu_util, seed=0,
                       trust_remote_code=spec["trust"], generation_config="vllm", max_num_seqs=512)
        self.stops, self.stop_strings = stops, spec.get("stop_strings") or []

    def _sp(self, r):
        from vllm import SamplingParams
        kw = dict(max_tokens=r["max_new"], stop_token_ids=self.stops, skip_special_tokens=False,
                  spaces_between_special_tokens=False, detokenize=bool(self.stop_strings))
        if self.stop_strings:
            kw["stop"] = self.stop_strings
        if r.get("sampling"):
            return SamplingParams(seed=r["seed"], **r["sampling"], **kw)
        return SamplingParams(temperature=0.0, **kw)

    def generate(self, reqs, one_by_one=False):
        prompts = [dict(prompt_token_ids=r["prompt_ids"]) for r in reqs]
        sps = [self._sp(r) for r in reqs]
        if one_by_one:
            outs = [self.llm.generate([p], [s], use_tqdm=False)[0] for p, s in zip(prompts, sps)]
        else:
            outs = self.llm.generate(prompts, sps, use_tqdm=False)
        return [dict(ids=list(o.outputs[0].token_ids), finish=o.outputs[0].finish_reason) for o in outs]


class HFEngine:
    def __init__(self, spec, batch, stops, pad_id, device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["sha"], dtype=getattr(torch, dtype),
                                                          trust_remote_code=spec["trust"]).to(device).eval()
        # a clean generation config: none of the model's shipped sampling defaults leak into greedy runs
        self.model.generation_config = GenerationConfig(eos_token_id=stops, pad_token_id=pad_id)
        self.batch, self.stops, self.pad, self.device = batch, set(stops), pad_id, device

    def generate(self, reqs, one_by_one=False):
        torch = self.torch
        out = [None] * len(reqs)
        groups = {}
        for i, r in enumerate(reqs):   # one task cap and one sampling setting per batch
            groups.setdefault((r["cap"], json.dumps(r.get("sampling"), sort_keys=True)), []).append(i)
        for _, idx in groups.items():
            idx.sort(key=lambda i: len(reqs[i]["prompt_ids"]))
            B = 1 if one_by_one else self.batch
            for k in range(0, len(idx), B):
                part = idx[k:k + B]
                L = max(len(reqs[i]["prompt_ids"]) for i in part)
                ids = torch.full((len(part), L), self.pad, dtype=torch.long)
                att = torch.zeros((len(part), L), dtype=torch.long)
                for j, i in enumerate(part):
                    p = reqs[i]["prompt_ids"]
                    ids[j, L - len(p):] = torch.tensor(p)
                    att[j, L - len(p):] = 1
                # each row is trimmed to its own cap afterwards: rows are causal and masked, so a longer batch cap
                # never changes the tokens a row keeps
                max_new = max(reqs[i]["max_new"] for i in part)
                s = reqs[part[0]].get("sampling")
                kw = dict(do_sample=False)
                if s:
                    torch.manual_seed(reqs[part[0]]["seed"])
                    kw = dict(do_sample=True, temperature=s.get("temperature", 1.0), top_p=s.get("top_p", 1.0),
                              top_k=s.get("top_k", 0), repetition_penalty=s.get("repetition_penalty", 1.0))
                with torch.no_grad():
                    g = self.model.generate(input_ids=ids.to(self.device), attention_mask=att.to(self.device), max_new_tokens=max_new, **kw)
                for j, i in enumerate(part):
                    new = g[j, L:].tolist()
                    cut = next((t for t, x in enumerate(new) if x in self.stops), None)
                    if cut is not None and cut < reqs[i]["max_new"]:
                        out[i] = dict(ids=new[:cut + 1], finish="stop")
                    else:
                        out[i] = dict(ids=new[:reqs[i]["max_new"]], finish="length")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--engine", choices=["vllm", "hf"], required=True)
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json")); ap.add_argument("--items", default=os.path.join(HERE, "items"))
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", default="gsm8k,math,ifeval,mt,loop,rec,ref")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--device", default="cuda", help="HF engine only (cpu/mps for local tests)")
    ap.add_argument("--dry", action="store_true", help="build and print the requests, load no model")
    ap.add_argument("--max-cap", type=int, default=0, help="clamp every cap (local tests; the fp32 reference runs)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="float32 only for the 2.4 references")
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. fp32)")
    ap.add_argument("--order-seed", type=int, default=0, help="shuffle the request order (bf16 batch-noise study)")
    a = ap.parse_args()
    spec = next(m for m in json.load(open(a.manifest))["models"] if m["name"] == a.name)
    tasks = set(a.tasks.split(","))
    lim = (lambda xs: xs[:a.limit]) if a.limit else (lambda xs: xs)
    out_dir = os.path.join(a.out, a.name, a.engine + (f"_{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    from transformers import AutoTokenizer, GenerationConfig
    # tokenizers here need no remote code (cagliostro ships tokenizer.json), so the repo's Python never runs for them
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["sha"], trust_remote_code=False)
    sp = special_ids(tok)
    try:
        gc_eos = GenerationConfig.from_pretrained(spec["repo"], revision=spec["sha"]).eos_token_id
    except Exception:
        gc_eos = None
    stops = set([gc_eos] if isinstance(gc_eos, int) else (gc_eos or []))
    if tok.eos_token_id is not None:
        stops.add(tok.eos_token_id)
    eot = end_of_turn_id(tok, sp)
    if eot is not None:
        stops.add(eot)
    stops = sorted(stops)
    specials = sorted({tok.convert_ids_to_tokens(i) for i in sp})
    stop_strings = spec.get("stop_strings") or []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else stops[0]
    caps = {k: min(v, a.max_cap) if a.max_cap else v for k, v in CAPS.items()}
    rec_cap = min(spec.get("rec_cap") or 0, a.max_cap or 10 ** 9)
    max_len = min(spec["ctx"], max(max(caps.values()), rec_cap) + 1024)

    def render(messages, mode):
        kw = {} if mode == "native" else {"enable_thinking": mode == "on"}
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if spec["bos_check"] and ids[0] != tok.bos_token_id:
            raise SystemExit(f"{a.name}: first prompt token {ids[0]} is not <bos> {tok.bos_token_id}")
        return ids

    def req(task, mode, var, item_id, messages, cap, sampling=None, seed=0, keep_ids=False):
        ids = render(messages, mode)
        return dict(task=task, mode=mode, var=var, id=item_id, prompt_ids=ids, cap=cap, max_new=min(cap, max_len - len(ids)),
                    sampling=sampling, seed=seed, keep_ids=keep_ids)

    gsm = load(os.path.join(a.items, "gsm8k.jsonl"))
    modes = spec["modes"]
    think_mode = "on" if "on" in modes else "native"
    phase1, phase3 = [], []
    for m in modes:
        if "gsm8k" in tasks:
            phase1 += [req("gsm8k", m, "greedy", r["id"], [{"role": "user", "content": r["question"] + SUFFIX}], caps["gsm8k"],
                           keep_ids=k < REF_N) for k, r in enumerate(lim(gsm))]
        if "math" in tasks:
            phase1 += [req("math", m, "greedy", r["id"], [{"role": "user", "content": r["problem"] + MATH_SUFFIX}], caps["math"])
                       for r in lim(load(os.path.join(a.items, "math_l123.jsonl")))]
        if "ifeval" in tasks:
            phase1 += [req("ifeval", m, "greedy", r["id"], [{"role": "user", "content": r["prompt"]}], caps["ifeval"])
                       for r in lim(load(os.path.join(a.items, "ifeval.jsonl")))]
        if "mt" in tasks:
            phase1 += [req("mt1", m, "greedy", r["id"], [{"role": "user", "content": r["q1"]}], caps["mt1"])
                       for r in lim(load(os.path.join(a.items, "mt.jsonl")))]
        if "loop" in tasks:
            for var in ("greedy", "sampled"):
                phase1 += [req("loop", m, var, r["id"], [{"role": "user", "content": r["prompt"]}], caps["loop"],
                               sampling=spec["sampling"] if var == "sampled" else None, seed=1000 + k, keep_ids=True)
                           for k, r in enumerate(lim(load(os.path.join(a.items, "loop.jsonl"))))]
    if "mt" in tasks:
        n = min(E2_N, a.limit) if a.limit else E2_N
        phase1 += [req("e2", think_mode, "greedy", gsm[E2_N + i]["id"],
                       [{"role": "user", "content": gsm[i]["question"] + SUFFIX}, {"role": "assistant", "content": ref_solution(gsm[i]["answer"])},
                        {"role": "user", "content": gsm[E2_N + i]["question"] + SUFFIX}], caps["e2"]) for i in range(n)]
    if "rec" in tasks and rec_cap:
        rec_items = lim(gsm)[:spec.get("rec_limit") or None]   # rec_limit: a long cap on the first N items only (cost)
        phase1 += [req("gsm8k", m, f"cap{rec_cap}", r["id"], [{"role": "user", "content": r["question"] + SUFFIX}], rec_cap)
                   for m in modes for r in rec_items]
    if "ref" in tasks:
        m0 = "off" if "off" in modes else modes[0]
        phase3 = [req("ref", m0, "one_by_one", r["id"], [{"role": "user", "content": r["question"] + SUFFIX}], caps["ref"], keep_ids=True)
                  for r in gsm[:min(REF_N, a.limit) if a.limit else REF_N]]
    skipped = [dict(task=r["task"], mode=r["mode"], id=r["id"], plen=len(r["prompt_ids"])) for r in phase1 + phase3 if r["max_new"] < 1]
    phase1 = [r for r in phase1 if r["max_new"] >= 1]
    phase3 = [r for r in phase3 if r["max_new"] >= 1]

    if a.dry:
        from collections import Counter
        allr = phase1 + phase3
        print(json.dumps(dict(stops=stops, stops_text=[tok.convert_ids_to_tokens(i) for i in stops], end_of_turn=eot, max_len=max_len,
                              pad=pad_id, skipped=len(skipped), counts=Counter(f"{r['task']}/{r['mode']}/{r['var']}" for r in allr)), indent=1))
        seen = set()
        for r in allr:
            k = (r["task"], r["mode"])
            if k not in seen:
                seen.add(k)
                print(f"--- {k} plen {len(r['prompt_ids'])} max_new {r['max_new']}\n{tok.decode(r['prompt_ids'], skip_special_tokens=False)!r}")
        return
    t0 = time.time()
    eng = VllmEngine(spec, max_len, a.gpu_util, stops, a.dtype) if a.engine == "vllm" else HFEngine(spec, a.batch, stops, pad_id, a.device, a.dtype)
    t_load = time.time() - t0
    files, timing = {}, dict(load=round(t_load, 1))

    def record(r, o):
        ids = o["ids"]
        finish = o["finish"]
        if ids and ids[-1] in stops:
            ids = ids[:-1]; finish = "stop"
        rec = dict(id=r["id"], task=r["task"], mode=r["mode"], var=r["var"], plen=len(r["prompt_ids"]), psha=sha1_ids(r["prompt_ids"]),
                   cap=r["cap"], max_new=r["max_new"], n=len(ids), finish=finish, osha=sha1_ids(ids),
                   raw=tok.decode(ids, skip_special_tokens=False))
        if r["keep_ids"]:
            rec["ids"] = ids
        if r["task"] == "loop":
            rec["period"] = periodic_tail(ids)
        files.setdefault(f"{r['task']}__{r['mode']}__{r['var']}", []).append(rec)
        return rec

    def run(reqs, label, one_by_one=False):
        if not reqs:
            return []
        t = time.time()
        outs = eng.generate(reqs, one_by_one=one_by_one)
        recs = [record(r, o) for r, o in zip(reqs, outs)]
        timing[label] = round(time.time() - t, 1)
        print(f"[{a.name}/{a.engine}] {label}: {len(reqs)} requests in {timing[label]} s, "
              f"{sum(x['n'] for x in recs)} tokens", flush=True)
        return recs

    if a.order_seed:   # a different batch composition; greedy outputs differ only through bf16 rounding
        import random
        random.Random(a.order_seed).shuffle(phase1)
    p1 = run(phase1, "phase1")
    vis = {(x["mode"], x["id"]): visible(x["raw"], specials, stop_strings).strip() for x in p1 if x["task"] == "mt1"}
    if "mt" in tasks:
        mt = {r["id"]: r for r in load(os.path.join(a.items, "mt.jsonl"))}
        phase2 = [req("mt2", m, "greedy", i, [{"role": "user", "content": mt[i]["q1"]}, {"role": "assistant", "content": v},
                                             {"role": "user", "content": mt[i]["q2"]}], caps["mt2"]) for (m, i), v in vis.items()]
        run([r for r in phase2 if r["max_new"] >= 1], "phase2")
    run(phase3, "phase3_one_by_one", one_by_one=True)

    for k, recs in files.items():
        with gzip.open(os.path.join(out_dir, f"{k}.jsonl.gz"), "wt") as f:
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import torch, transformers
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else a.device)
    if a.engine == "vllm":
        import vllm
        versions["vllm"] = vllm.__version__
    meta = dict(name=a.name, engine=a.engine, dtype=a.dtype, tag=a.tag, order_seed=a.order_seed, spec=spec, stops=stops, end_of_turn=eot, specials=specials, stop_strings=stop_strings,
                max_len=max_len, caps=caps, limit=a.limit, batch=a.batch if a.engine == "hf" else None,
                gpu_util=getattr(eng, "gpu_util", None),
                skipped=skipped, files={k: len(v) for k, v in files.items()}, timing=timing,
                total_s=round(time.time() - t_start, 1), versions=versions, date=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"[{a.name}/{a.engine}] done in {meta['total_s']} s; skipped {len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
