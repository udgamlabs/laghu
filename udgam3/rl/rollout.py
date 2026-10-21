"""Shared by Laghu RL training (laghu_rl.py) and its dev evals (rl_eval.py): render prompts, generate (batched, left
padded, HF transformers), run Python tool calls in the sandbox and continue, and score replies.

Item fields: id, src (gsm8k|math|chat), prompt (user text) or messages, gold (maths), mode: "think" | "plain" | "tool".
A rollout = dict(prompt_ids, gen_ids, gen_mask (1 = model token, 0 = tool output inserted by the environment), text,
stopped, n_model, tool_calls, tool_errors).
"""
import json, os, re, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import sandbox  # noqa: E402
import importlib.util
_s = importlib.util.spec_from_file_location("cs_common", os.path.join(ROOT, "udgam3", "evals", "chat_suite", "common.py"))
cs = importlib.util.module_from_spec(_s); _s.loader.exec_module(cs)   # the chat suite's answer rule, thought split, loop test

TOOLS = [{"type": "function", "function": {"name": "python", "description": "Run Python code and return what it prints.",
          "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}}, "required": ["code"]}}}]
CALL_RX = re.compile(r'<\|tool_call>call:python\{code:<\|"\|>(.*?)<\|"\|>\}<tool_call\|>', re.S)
SPECIALS = ["<|turn>", "<turn|>", "<eos>", "<bos>", "<pad>", "<|channel>", "<channel|>", "<|think|>", "<|tool_call>", "<tool_call|>",
            "<|tool_response>", "<tool_response|>", "<|tool>", "<tool|>", '<|"|>']


def render(tok, it):
    msgs = it.get("messages") or [{"role": "user", "content": it["prompt"]}]
    return tok.apply_chat_template(msgs, tools=TOOLS if it["mode"] == "tool" else it.get("tools"), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=it["mode"] == "think")


def visible(text):
    """Final visible answer: thought channel removed, tool calls and tool outputs removed, special tokens dropped."""
    ans = cs.split_thinking(text)[1]
    ans = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", ans, flags=re.S)
    ans = re.sub(r"<\|tool_response>.*?<tool_response\|>", "", ans, flags=re.S)
    for s in SPECIALS:
        ans = ans.replace(s, "")
    return ans.strip()


def correct(it, text):
    return it.get("gold") is not None and cs.gsm8k_ok(visible(text), it["gold"])


def _gen(model, tok, prefixes, max_new, sample, temperature, stops, pad, budget_tokens):
    """Batched generate for a list of token-id prefixes; returns new token lists (cut at the first stop, stop kept)."""
    out = [None] * len(prefixes)
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * (len(prefixes[order[j]]) + max_new) <= budget_tokens and j - i < 256:
            j += 1
        j = max(j, i + 1); grp = order[i:j]; L = max(len(prefixes[k]) for k in grp)
        ids = torch.full((len(grp), L), pad, dtype=torch.long); att = torch.zeros_like(ids)
        for r, k in enumerate(grp):
            p = prefixes[k]; ids[r, L - len(p):] = torch.tensor(p); att[r, L - len(p):] = 1
        dev = next(model.parameters()).device
        with torch.no_grad():
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
            g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, eos_token_id=stops,
                               pad_token_id=pad, **kw)
        for r, k in enumerate(grp):
            new = g[r, L:].tolist(); cut = None
            for t_, t in enumerate(new):
                if t in stops:
                    cut = t_ + 1; break
            out[k] = new[:cut] if cut else [t for t in new if t != pad]
        i = j
    return out


def rollouts(model, tok, items, k=1, max_new=768, sample=True, temperature=1.0, max_tool_calls=3, budget_tokens=200_000):
    """k rollouts per item (tool items may call python up to max_tool_calls times; the model's own tokens share max_new)."""
    v = tok.get_vocab(); END, EOS, TR = v["<turn|>"], v["<eos>"], v["<|tool_response>"]
    stops = [END, EOS, TR]; pad = tok.pad_token_id if tok.pad_token_id is not None else v["<pad>"]
    R = []
    for it in items:
        p = tok(render(tok, it), add_special_tokens=False)["input_ids"]
        R += [dict(item=it, prompt_ids=p, gen_ids=[], gen_mask=[], stopped=False, n_model=0, tool_calls=0, tool_errors=0, bad_call=0)
              for _ in range(k)]
    active = list(range(len(R)))
    while active:
        news = _gen(model, tok, [R[i]["prompt_ids"] + R[i]["gen_ids"] for i in active],
                    max(1, max(max_new - R[i]["n_model"] for i in active)), sample, temperature, stops, pad, budget_tokens)
        nxt = []
        for i, new in zip(active, news):
            r = R[i]; room = max_new - r["n_model"]; new = new[:room]
            r["gen_ids"] += new; r["gen_mask"] += [1] * len(new); r["n_model"] += len(new)
            last = new[-1] if new else None
            if last in (END, EOS):
                r["stopped"] = True; continue
            if last == TR and r["item"]["mode"] == "tool" and r["tool_calls"] < max_tool_calls and r["n_model"] < max_new:
                seg = tok.decode(r["gen_ids"], skip_special_tokens=False)
                calls = CALL_RX.findall(seg)
                if not calls:
                    r["bad_call"] += 1; continue
                ok, res = sandbox.run(calls[-1]); r["tool_calls"] += 1; r["tool_errors"] += (not ok)
                env = tok(f'response:python{{value:<|"|>{res}<|"|>}}<tool_response|>', add_special_tokens=False)["input_ids"]
                r["gen_ids"] += env; r["gen_mask"] += [0] * len(env); nxt.append(i)
            elif last == TR and r["item"].get("tools") and r["item"]["mode"] != "tool":
                r["stopped"] = True   # a chat item with its own function schemas: handing over to the tool is a proper end
            # otherwise (a tool hand-over without tools, or the cap) the rollout ends without a proper stop
        active = nxt
    for r in R:
        r["text"] = tok.decode(r["gen_ids"], skip_special_tokens=False)
        model_ids = [t for t, m in zip(r["gen_ids"], r["gen_mask"]) if m]
        r["loop"] = (not r["stopped"]) and cs.periodic_tail(model_ids) > 0
    return R


def reward(r, max_new, buf=192):
    """1 for a correct final answer; penalties: no proper stop (cap, loop, stray tool hand-over) -0.5, a periodic loop
    tail -0.25 more, overlong (linear in the last `buf` tokens of the cap, up to -0.5), thinking format -0.2 (thinking
    on without a closed thought, or a thought with thinking off), malformed tool call -0.2."""
    it, text = r["item"], r["text"]
    rew = 1.0 if correct(it, text) else 0.0
    if not r["stopped"]:
        rew -= 0.5
    if r["loop"]:
        rew -= 0.25
    over = r["n_model"] - (max_new - buf)
    if over > 0:
        rew -= 0.5 * min(1.0, over / buf)
    thought = "<|channel>thought" in text
    closed = thought and "<channel|>" in text
    if (it["mode"] == "think" and not closed) or (it["mode"] != "think" and thought):
        rew -= 0.2
    if r["bad_call"]:
        rew -= 0.2
    return rew
