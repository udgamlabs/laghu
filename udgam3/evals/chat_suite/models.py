"""Chat-suite model list and per-model settings; writes manifest.json with every model pinned to a commit sha.

  uv run --project evals/harness python udgam3/evals/chat_suite/models.py

Settings per model:
- modes: "off"/"on" = the chat template's enable_thinking switch; "native" = no switch (the template is used as is).
- sampling: the "shipped sampling defaults" for the loop check (PREREGISTRATION.md 2.3), with where they come from:
  generation_config.json first, then the model card's own example, then vLLM's default (temperature 1.0) when neither
  gives any.
- rec_cap: a reasoning model's recommended output cap, for the secondary GSM8K row (2.3).
- stop_strings: only for templates whose turns end in plain text instead of a special token.
- bos_check: Udgam models must start with <bos> (2.1).
Already-pinned shas are kept, so reruns use the same weights.
"""
import json, os

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
VLLM_DEFAULT = dict(temperature=1.0)
CARD_SMOL = dict(temperature=0.2, top_p=0.9)
MODELS = [
    # size-class rivals (<= 200M), named in the task
    dict(name="smollm2-135m-instruct", repo="HuggingFaceTB/SmolLM2-135M-Instruct", group="rival", sampling=CARD_SMOL,
         sampling_src="model card example (generation_config.json sets none)"),
    dict(name="falcon-h1-tiny-90m-instruct", repo="tiiuae/Falcon-H1-Tiny-90M-Instruct", group="rival", sampling=VLLM_DEFAULT,
         sampling_src="none shipped (generation_config.json and card give none): vLLM default"),
    dict(name="falcon-h1-tiny-r-90m", repo="tiiuae/Falcon-H1-Tiny-R-90M", group="rival", sampling=VLLM_DEFAULT,
         sampling_src="none shipped: vLLM default", reasoning=True, rec_cap=32000, rec_limit=300,
         rec_cap_src="no cap published for the 90M; 32,000 is the generation length its authors used for Falcon-H1-Tiny-R GRPO (blog)"),
    dict(name="supra2-100m-instruct", repo="SupraLabs/Supra2-100M-Instruct", group="rival",
         sampling=dict(temperature=0.2, top_p=0.85, top_k=25),
         sampling_src="model card example (its no_repeat_ngram_size=3 has no vLLM equivalent and is not applied)"),
    dict(name="cagliostro-v3.5-chat", repo="bench-labs/cagliostro-v3.5-chat", group="rival", engine="hf", trust=True,
         sampling=dict(temperature=0.3, repetition_penalty=1.1),
         sampling_src="model card ('greedy or temperature 0.3 with a repetition penalty of 1.1')"),
    dict(name="mobilellm-r1.5-140m", repo="facebook/MobileLLM-R1.5-140M", group="rival", gated=True,
         sampling=dict(temperature=0.6, top_p=0.95),
         sampling_src="model card (the sampling its authors used to regenerate its SFT data); generation_config.json sets none",
         reasoning=True, rec_cap=8192, rec_cap_src="model card example (max_new_tokens=8192)"),
    dict(name="falcon-h1-tiny-tool-calling-90m", repo="tiiuae/Falcon-H1-Tiny-Tool-Calling-90M", group="rival", sampling=VLLM_DEFAULT,
         sampling_src="none shipped: vLLM default"),
    # size-class chat models found by the 2026-10-19 scan of rival orgs (PREREGISTRATION.md 3: a scan can add rivals)
    dict(name="falcon-h1-tiny-multilingual-100m-instruct", repo="tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct",
         group="rival_scan", sampling=VLLM_DEFAULT, sampling_src="none shipped: vLLM default"),
    dict(name="boris-125m-instruct", repo="opencerebral/Boris-125M-Instruct", group="rival_scan",
         sampling=dict(temperature=0.7, top_p=0.95), sampling_src="model card example",
         stop_strings=["### Instruction:"]),
    dict(name="supra2-medium-instruct", repo="SupraLabs/Supra2-Medium-Instruct", group="rival_scan",
         sampling=dict(temperature=0.2, top_p=0.85, top_k=25),
         sampling_src="model card example (no_repeat_ngram_size=3 not applied)"),
    dict(name="surjo-50m-sft-only", repo="SurjoLabs/Surjo-50m-SFT-Only", group="rival", engine="hf", trust=True,
         sampling=dict(temperature=0.7, top_p=0.9), sampling_src="model card example"),
    dict(name="blaze-sft", repo="SurjoLabs/Blaze-SFT", group="rival_scan", engine="hf", trust=True,
         sampling=dict(temperature=0.7, top_p=0.9), sampling_src="model card example"),
    # our own bars
    dict(name="gargi", repo="udgamlabs/Gargi", group="ours", modes=["off", "on"], bos_check=True,
         sampling=dict(temperature=0.7, top_p=0.95, top_k=64), sampling_src="generation_config.json"),
    dict(name="udgam-384m-chat", repo="UdgamLabs/Udgam-384M-Chat", group="ours", modes=["off", "on"], bos_check=True,
         sampling=dict(temperature=0.1, top_p=0.9, top_k=40), sampling_src="generation_config.json"),
    # near-size references, reported separately
    dict(name="gemma-3-270m-it", repo="google/gemma-3-270m-it", group="reference", gated=True,
         sampling=dict(temperature=1.0, top_p=0.95, top_k=64), sampling_src="generation_config.json"),
    dict(name="lfm2.5-230m", repo="LiquidAI/LFM2.5-230M", group="reference",
         sampling=dict(temperature=0.1, top_k=50, repetition_penalty=1.05), sampling_src="generation_config.json"),
    dict(name="smollm2-360m-instruct", repo="HuggingFaceTB/SmolLM2-360M-Instruct", group="reference", sampling=CARD_SMOL,
         sampling_src="model card example (generation_config.json sets none)"),
]


def main():
    api = HfApi()
    path = os.path.join(HERE, "manifest.json")
    old = {m["name"]: m for m in json.load(open(path))["models"]} if os.path.exists(path) else {}
    out = []
    for m in MODELS:
        m = {**dict(engine="vllm", trust=False, modes=["native"], gated=False, reasoning=False, bos_check=False), **m}
        prev = old.get(m["name"], {})
        info = api.model_info(m["repo"], revision=prev.get("sha"))
        m["sha"] = prev.get("sha") or info.sha
        cfg = json.load(open(api.hf_hub_download(m["repo"], "config.json", revision=m["sha"])))
        tc = cfg.get("text_config", {})
        m["ctx"] = cfg.get("max_position_embeddings") or cfg.get("n_positions") or tc.get("max_position_embeddings")
        m["arch"] = (cfg.get("architectures") or [None])[0]
        m["config_dtype"] = cfg.get("torch_dtype") or cfg.get("dtype")
        try:
            st = api.get_safetensors_metadata(m["repo"], revision=m["sha"])
            m["params"] = sum(st.parameter_count.values())
        except Exception as e:
            m["params"] = prev.get("params")
        card = info.card_data or {}
        m["licence"] = card.get("license_name") or card.get("license") if card else None
        out.append(m)
        print(f"{m['name']:44s} {m['sha'][:10]} {m['arch']:24s} ctx {m['ctx']:>7} params {m['params']} {m['licence']}")
    json.dump(dict(note="chat-suite models, pinned 2026-10-19; settings documented in models.py", dtype="bfloat16",
                   models=out), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
