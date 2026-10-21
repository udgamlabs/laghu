"""Leak check for the public snapshot (run before any public push): builds it from the live state in dry mode and fails if
any sensitive term or value appears. Usage: python3 ops/train-dash/push/check_public.py PUBLIC_JSON_FILE"""
import json, re, sys
BAD = [r"\$", r"cost", r"price", r"spend", r"budget", r"cap\b", r"shadeform", r"massed", r"beltsville", r"provider", r"runpod",
       r"\bTST\b", r"token bag", r"superposition", r"muon", r"warmdown", r"learning rate", r"\blr\b", r"lrm", r"batch", r"(?<![\d.])524288(?![\d.])",
       r"\bsteps?\b", r"(?<![\d.])76294(?![\d.])", r"(?<![\d.])19073(?![\d.])", r"(?<![\d.])61035(?![\d.])", r"wandb", r"w&b", r"gargi", r"smollm", r"target", r"project", r"phase",
       r"mix", r"finephrase", r"fineweb", r"dclm", r"synth", r"pool", r"epoch", r"d28", r"(?<![\d.])640(?![\d.])", r"muon", r"ema\b", r"recipe",
       r"watcher", r"mirror", r"secret", r"token_", r"hf_", r"udgam3-base-ckpt", r"YOUR_HF_NAMESPACE", r"deadline"]


def leaks(text):
    return [b for b in BAD if re.search(b, text, re.I)]


if __name__ == "__main__":
    pub = open(sys.argv[1]).read()
    hits = leaks(pub)
    d = json.loads(pub)
    print("public keys:", sorted(d)); print("size:", len(pub), "bytes")
    print("LEAK CHECK:", "PASS (no sensitive terms)" if not hits else "FAIL: " + ", ".join(hits))
    sys.exit(1 if hits else 0)
