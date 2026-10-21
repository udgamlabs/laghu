"""Render the Udgam-3 SFT mix v1 with Udgam-3's own exported chat template (the Gemma 4 canonical template) and build
assistant-only loss masks, for nanochat/scripts/udgam_sft.py. Plan: udgam3/SFT_DATA_PLAN.md. $0, Mac CPU.

Sources:
  udgam3/sft/full_v1/<slice>.jsonl   the verified DeepSeek-V4.1-Flash rows (10 slices, ~28.7K)
  data/udgam2_sft_v4/dl/mt_v1/gen_mt.jsonl   multi-turn follow-ups by gpt-oss-120b (Apache-2.0) on CC-BY-4.0 seeds;
                                     Gargi identity rows and rows naming Gargi are dropped (this model is not Gargi)
Thinking: a row with enable_thinking renders thinking-on (reasoning -> thought channel). Maths/logic thinking rows ALSO
get a thinking-off twin (reasoning stripped, the answer kept), so the switch is learned on the same prompts.
Filters: live-looking credentials (2026-10-18: a third-party key in nemo_chat seeds), teacher fingerprints
(DeepSeek / "as an AI language model"), {MODEL_NAME} filled with --model-name, rows over --max-tokens dropped.
Loss mask: tokens inside model turns (after "<|turn>model\\n" up to and including "<turn|>"); tool-response payloads
masked except the "<|tool_response>" hand-over token (same rule as udgam/sft/render_sft.py, reused here).
Split: 2% of ids per source -> val (hash-based, stable).
Output: <out>/{train,val}.parquet (id, src, job, think, n_tok, n_loss, epochs, ids, mask), stats.json, samples.jsonl.

  uv run --project evals/harness python udgam3/sft/render_v1.py --tokenizer-dir HF_EXPORT_DIR --model-name NAME [--out DIR]
"""
import argparse, ast, collections, glob, hashlib, json, os, random, re, sys

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "udgam", "sft"))
from render_sft import mask_spans, encode, strip_reasoning  # noqa: E402  (the reviewed Gargi mask rule)

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer-dir", required=True, help="Udgam-3 HF export (tokenizer.json + chat_template.jinja)")
ap.add_argument("--model-name", required=True); ap.add_argument("--out", default=os.path.join(ROOT, "data", "udgam3_sft_v1"))
ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--limit", type=int, default=0, help="rows per source (tests)")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
tpl = open(os.path.join(a.tokenizer_dir, "chat_template.jinja")).read()
ref = open(os.path.join(ROOT, "tools", "ref", "gemma-4-E2B-it", "chat_template.jinja")).read()
assert tok.chat_template == tpl and tpl == ref, "chat template is not the exported Gemma 4 canonical template"
SECRET = re.compile(r"infura\.io/v3/[0-9a-f]{32}|\bAKIA[0-9A-Z]{16}\b|\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{32,}|\bghp_[A-Za-z0-9]{36}\b|"
                    r"\bgithub_pat_[A-Za-z0-9_]{60,}|\bhf_[A-Za-z0-9]{34}\b|\bxox[abprs]-[A-Za-z0-9-]{20,}|\bsk_live_[A-Za-z0-9]{24,}|"
                    r"\bAIza[0-9A-Za-z_-]{35}\b|-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----|"
                    r"[\"']?infuraKey[\"']?\s*[:=]\s*[\"'][0-9a-f]{32}")
FINGER = re.compile(r"\bDeepSeek\b|as an AI language model|my knowledge cutoff|I was trained by|OpenAI's|\bChatGPT\b", re.I)
NOT_US = re.compile(r"\bGargi\b|384 ?million parameters|\b384M\b")
EPOCHS = {"identity": 2}
rng = random.Random(a.seed)
st, rows, samples = collections.Counter(), {"train": [], "val": []}, []


def fix_msgs(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if isinstance(m.get("tool_calls"), str):
            m["tool_calls"] = ast.literal_eval(m["tool_calls"])
        if isinstance(m.get("content"), str):
            m["content"] = m["content"].replace("{MODEL_NAME}", a.model_name)
        if isinstance(m.get("reasoning_content"), str):
            m["reasoning_content"] = m["reasoning_content"].replace("{MODEL_NAME}", a.model_name)
        out.append(m)
    return out


def emit(rid, src, job, msgs, tools, think):
    text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, enable_thinking=think)
    if SECRET.search(text):
        st[f"drop_secret:{src}"] += 1; return
    body = "".join(text[s:e] for s, e in mask_spans(text))
    if src != "identity" and FINGER.search(body):  # identity rows may say "I'm not ChatGPT"
        st[f"drop_fingerprint:{src}"] += 1; return
    ids, mask = encode(tok, text)
    if len(ids) > a.max_tokens:
        st[f"drop_too_long:{src}"] += 1; return
    if sum(mask) == 0:
        st[f"drop_no_loss:{src}"] += 1; return
    split = "val" if int(hashlib.sha1(rid.split("-nothink")[0].encode()).hexdigest(), 16) % 50 == 0 else "train"
    rows[split].append({"id": rid, "src": src, "job": job, "think": think, "n_tok": len(ids), "n_loss": sum(mask),
                        "epochs": EPOCHS.get(src, 1) if split == "train" else 1, "ids": ids, "mask": mask})
    st[f"{split}:{src}:{'on' if think else 'off'}"] += 1
    if split == "train" and sum(1 for m in msgs if m.get("role") == "user") >= 2:
        st["train_multi_turn"] += 1
    if len(samples) < 300 and rng.random() < 0.01:
        samples.append({"id": rid, "src": src, "think": think, "text": text, "loss_text": body})


for f in sorted(glob.glob(os.path.join(ROOT, "udgam3", "sft", "full_v1", "*.jsonl"))):
    name = os.path.basename(f)[:-6]
    if any(name.startswith(x) for x in ("rejected_", "requests_", "raw_", "run_requests")):
        continue
    for i, line in enumerate(open(f)):
        if a.limit and i >= a.limit:
            break
        r = json.loads(line)
        msgs, think = fix_msgs(r["messages"]), bool(r.get("enable_thinking"))
        emit(r["id"], r["slice"], r.get("category", ""), msgs, r.get("tools"), think)
        if think and r["slice"] == "maths_logic":
            emit(r["id"] + "-nothink", r["slice"], r.get("category", ""), strip_reasoning(msgs), r.get("tools"), False)
gm = os.path.join(ROOT, "data", "udgam2_sft_v4", "dl", "mt_v1", "gen_mt.jsonl")
for i, line in enumerate(open(gm)):
    if a.limit and i >= a.limit:
        break
    r = json.loads(line)
    if r.get("kind") == "identity" or any(NOT_US.search(m.get("content") or "") for m in r["messages"]):
        st["drop_not_us:gen_mt"] += 1; continue
    emit(r["id"], "gen_mt", r.get("ftype", ""), fix_msgs(r["messages"]), None, False)

os.makedirs(a.out, exist_ok=True)
for split, rs in rows.items():
    rng.shuffle(rs)
    pq.write_table(pa.Table.from_pylist(rs), os.path.join(a.out, f"{split}.parquet"))
by = collections.defaultdict(lambda: collections.Counter())
for split, rs in rows.items():
    for r in rs:
        by[r["src"]][f"{split}_rows"] += 1; by[r["src"]][f"{split}_tok"] += r["n_tok"]; by[r["src"]][f"{split}_loss"] += r["n_loss"]
stats = {"args": vars(a), "counts": dict(st), "by_source": {k: dict(v) for k, v in by.items()},
         "train_rows": len(rows["train"]), "val_rows": len(rows["val"]),
         "train_tokens": sum(r["n_tok"] for r in rows["train"]), "train_loss_tokens": sum(r["n_loss"] for r in rows["train"]),
         "multi_turn_share": round(st["train_multi_turn"] / max(1, len(rows["train"])), 3)}
json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
with open(os.path.join(a.out, "samples.jsonl"), "w") as fo:
    for s in samples:
        fo.write(json.dumps(s) + "\n")
print(json.dumps({k: stats[k] for k in ("train_rows", "val_rows", "train_tokens", "train_loss_tokens", "multi_turn_share")}, indent=1))
print({k: v for k, v in st.items() if k.startswith("drop")})
