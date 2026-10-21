"""Validate a tokenizer dir: (1) HF decode(encode(x)) == x, (2) HF ids == llama.cpp ids on N snippets
(via a vocab-only GGUF + llama-tokenize), (3) compression (chars/token) per source vs a reference tokenizer.
Usage: python tokenizer_check.py TOK_DIR [--n 2000] [--ref /path/to/ref_tokenizer_dir]"""
import json, os, sys, glob, random, subprocess, argparse, tempfile, shutil
from transformers import PreTrainedTokenizerFast, AutoTokenizer
ROOT = os.path.join(os.path.dirname(__file__), "..")
LLAMA = os.path.join(ROOT, "tools/llama.cpp")
ap = argparse.ArgumentParser(); ap.add_argument("tok"); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--ref", default=os.path.join(ROOT, "tools/ref/gemma-4-E2B-it")); a = ap.parse_args()
random.seed(0)
tok = PreTrainedTokenizerFast(tokenizer_file=f"{a.tok}/tokenizer.json")
ref = AutoTokenizer.from_pretrained(a.ref)
# snippets: random windows of real docs + adversarial strings
snips = []
for f in sorted(glob.glob(f"{ROOT}/data/samples/*.jsonl")):
    docs = [json.loads(l)["text"] for _, l in zip(range(3000), open(f))]
    for _ in range(a.n // 10):
        d = random.choice(docs); s = random.randrange(max(1, len(d) - 400)); snips.append((os.path.basename(f)[:-6], d[s:s + random.randint(20, 400)]))
adv = ["  leading and trailing  ", "\n\n\nthree newlines", "tabs\there\t\tand\r\nCRLF", "a  b   c    d", "2026-10-06 $4,200.00 #00042",
       "naïve café — 日本語 🙂 🇮🇳", "<|tool_call>call:create_reminder{title:<|\"|>Call Mom<|\"|>}<tool_call|>", "▁literal▁marker", "x"*300, "    def f():\n        return 1\n"]
snips += [("adversarial", s) for s in adv]
# (1) round trip
bad_rt = [s for _, s in snips if tok.decode(tok.encode(s, add_special_tokens=False)) != s]
print(f"round-trip: {len(snips)-len(bad_rt)}/{len(snips)} exact")
for s in bad_rt[:3]: print("   RT FAIL:", repr(s[:80]), "->", repr(tok.decode(tok.encode(s, add_special_tokens=False))[:80]))
# (2) llama.cpp parity via vocab-only gguf
tmp = tempfile.mkdtemp()
for f in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
    if os.path.exists(f"{a.tok}/{f}"): shutil.copy(f"{a.tok}/{f}", tmp)
gg = f"{tmp}/vocab.gguf"
r = subprocess.run([sys.executable, f"{LLAMA}/convert_hf_to_gguf.py", tmp, "--vocab-only", "--outfile", gg], capture_output=True, text=True)
if r.returncode: print("vocab-only GGUF conversion failed:\n", r.stderr[-1500:]); sys.exit(1)
mism = 0
for i, (src, s) in enumerate(snips):
    if s.endswith("\n"): s = s.rstrip("\n") + "."   # llama.cpp strips a trailing newline from prompt files
    pf = f"{tmp}/p.txt"; open(pf, "w").write(s)
    out = subprocess.run([f"{LLAMA}/build/bin/llama-tokenize", "-m", gg, "-f", pf, "--ids", "--log-disable", "--no-bos", "--no-escape"], capture_output=True, text=True).stdout.strip()
    lc = json.loads(out.splitlines()[-1]) if out else None
    hf = tok.encode(s, add_special_tokens=False)
    if lc != hf:
        mism += 1
        if mism <= 3: print(f"   PARITY FAIL [{src}]: {s[:60]!r}\n      HF {hf[:15]}\n      LC {lc[:15] if lc else lc}")
print(f"llama.cpp parity: {len(snips)-mism}/{len(snips)} identical")
# (3) compression
by = {}
for src, s in snips:
    c = by.setdefault(src, [0, 0, 0]); c[0] += len(s); c[1] += len(tok.encode(s, add_special_tokens=False)); c[2] += len(ref.encode(s, add_special_tokens=False))
print(f"{'source':22s} chars/token  ours | gemma4(262k)")
for src, (ch, t1, t2) in sorted(by.items()): print(f"{src:22s} {ch/t1:9.2f} | {ch/t2:6.2f}")
