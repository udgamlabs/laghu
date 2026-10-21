"""Chat-template test matrix (Phase 0; release gate item for Udgam-3, PREREGISTRATION 2.3).

Sends the same five checks to each runtime and records the raw replies:
  C1 thinking off  -> answer present, no reasoning text anywhere in the visible reply
  C2 thinking on   -> reasoning separated from the answer (reasoning field, or no raw thought markers in content)
  C3 JSON only     -> the visible reply parses as JSON (strict, and after stripping ``` fences)
  C4 multi-turn    -> turn 2 renders and answers using turn 1; no raw markers
  C5 tool call     -> a tool schema is accepted; records whether a parsed tool call came back

Runtimes (each optional): an OpenAI-compatible server (llama-server with or without --jinja; MLX server),
Ollama /api/chat, and transformers (apply_chat_template + greedy generate).

  python run_matrix.py --label llama-jinja --openai http://127.0.0.1:18766 OUT.json
  python run_matrix.py --label ollama --ollama gargi OUT.json
  python run_matrix.py --label transformers --hf udgamlabs/Gargi OUT.json
"""
import argparse, json, re, urllib.request

MARKERS = ["<|channel>", "<channel|>", "<|think|>", "<|turn>", "<turn|>", "thought\n", "<think>", "</think>"]
TOOL = {"type": "function", "function": {"name": "python", "description": "Run Python code and return what it prints.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}}, "required": ["code"]}}}
CHECKS = [
    ("C1_think_off", False, [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}], None),
    ("C2_think_on", True, [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}], None),
    ("C3_json_only", False, [{"role": "user", "content": "Give the city Paris and its country as JSON with keys \"city\" and \"country\". Output only the JSON, nothing else."}], None),
    ("C4_multi_turn", False, [{"role": "user", "content": "What is 17 + 25? Reply with just the number."},
                              {"role": "assistant", "content": "42"},
                              {"role": "user", "content": "Now multiply that by 2. Reply with just the number."}], None),
    ("C5_tool_call", False, [{"role": "user", "content": "Use the python tool to compute 123 * 456."}], [TOOL]),
]


def post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def via_openai(base, msgs, think, tools):
    body = {"messages": msgs, "temperature": 0, "max_tokens": 384, "chat_template_kwargs": {"enable_thinking": think}}
    if tools:
        body["tools"] = tools
    m = post(base + "/v1/chat/completions", body)["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "tool_calls": m.get("tool_calls") or []}


def via_ollama(model, msgs, think, tools):
    body = {"model": model, "stream": False, "think": think, "messages": msgs, "options": {"temperature": 0, "num_predict": 384}}
    if tools:
        body["tools"] = tools
    m = post("http://localhost:11434/api/chat", body)["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("thinking") or "", "tool_calls": m.get("tool_calls") or []}


def via_hf(state, repo, msgs, think, tools):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if "model" not in state:
        state["tok"] = AutoTokenizer.from_pretrained(repo)
        state["model"] = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    tok, model = state["tok"], state["model"]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)
    v = tok.get_vocab()
    stops = [i for i in (v.get("<eos>"), v.get("<turn|>")) if i is not None]
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=384, do_sample=False, eos_token_id=stops)
    raw = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
    m = re.search(r"<\|channel>thought\n(.*?)<channel\|>", raw, re.S)
    content = raw[m.end():] if m else raw
    content = content.replace("<turn|>", "").replace("<eos>", "").strip()
    return {"content": content, "reasoning": m.group(1) if m else "", "tool_calls": [], "raw": raw, "prompt": text}


def judge(name, r):
    c, reas = r["content"], r["reasoning"]
    leaked = [k for k in MARKERS if k in c]
    res = {"leaked_markers": leaked}
    if name in ("C1_think_off", "C2_think_on"):
        res["answer_ok"] = "42" in c
    if name == "C1_think_off":
        res["no_reasoning"] = not reas.strip() and not leaked
    if name == "C2_think_on":
        res["reasoning_separated"] = bool(reas.strip()) and not leaked
    if name == "C3_json_only":
        def ok(s):
            try:
                d = json.loads(s); return isinstance(d, dict) and "city" in d
            except Exception:
                return False
        res["json_strict"] = ok(c.strip())
        res["json_after_fences"] = ok(re.sub(r"^```(?:json)?\s*|\s*```$", "", c.strip()))
        res["no_reasoning"] = not reas.strip() and not leaked
    if name == "C4_multi_turn":
        res["answer_ok"] = "84" in c
    if name == "C5_tool_call":
        res["tool_call_parsed"] = bool(r["tool_calls"])
        res["tool_call_in_text"] = "<|tool_call>" in c or '"name"' in c
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--label", required=True)
    ap.add_argument("--openai"); ap.add_argument("--ollama"); ap.add_argument("--hf")
    a = ap.parse_args()
    state, rows = {}, []
    for name, think, msgs, tools in CHECKS:
        try:
            if a.openai:
                r = via_openai(a.openai, msgs, think, tools)
            elif a.ollama:
                r = via_ollama(a.ollama, msgs, think, tools)
            else:
                r = via_hf(state, a.hf, msgs, think, tools)
            rows.append({"check": name, "think": think, "reply": r, "verdict": judge(name, r)})
        except Exception as e:
            rows.append({"check": name, "think": think, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(a.label, name, rows[-1].get("verdict", rows[-1].get("error")), flush=True)
    json.dump({"label": a.label, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
