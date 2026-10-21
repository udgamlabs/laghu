#!/usr/bin/env python
"""UdgamLabs pretraining data mix -> nanochat parquet shards (two phases + validation).

    python build_mix.py --out DIR --scale 1.0 --workers 32            # full build (~11B tokens)
    python build_mix.py --out DIR --scale 0.0005 --workers 8 --extra-task-jsonl data/gold/pretrain/gold_rendered.jsonl   # local test
    python build_mix.py --out DIR --dry-run                           # access check + plan only

Output (DIR):
  shard_00000 .. shard_{a-1}   Phase A (first ~80% of tokens), sources shuffled within shards
  shard_{a} .. shard_{a+b-1}   Phase B (last ~20%, the decay/anneal mix)
  shard_{a+b}                  validation (held-out mixed sample, nanochat uses the LAST shard as val)
  manifest.json                per-source licenses + generator provenance / paths / filters / drops / tokens / phases,
                               per-shard stats, licence policy and excluded sources (data/LICENSES_AND_PROVENANCE.md)
  _meta/shard_XXXXX.sources.npy  uint8 source id per row (index into manifest["source_ids"]) for audits;
                                 nanochat only lists *.parquet in DIR, so this does not affect training.

Pipeline: (1) a scheduler hands "units" (a parquet file / every K-th row group of it, or every K-th record
of a small json file) to a process pool; each worker streams its unit with HfFileSystem + pyarrow
iter_batches (column projection, no full downloads), filters + renders docs, drops exact dups and
eval-contaminated docs (decontam.py), counts tokens exactly with our tokenizer and stops at the token
budget it was given. Budgets come from per-(source, phase) quotas, so every source stops at its quota.
(2) assembly: global exact dedup, MinHash near-dup on task docs, validation hold-out, random
doc->shard bucketing per phase, per-shard shuffle, write shards (row groups of 1024 rows, zstd).
"""
import argparse, ast, collections, glob, hashlib, json, math, os, random, re, shutil, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import decontam  # noqa: E402

TOKENIZER = f"{ROOT}/artifacts/tokenizer_v0/tokenizer.json"
CHAT_TEMPLATE = f"{ROOT}/tools/ref/gemma-4-E2B-it/chat_template.jinja"
FULL_TOKENS = 11.0e9
PHASE_FRAC = {"A": 0.8, "B": 0.2}
UNIT_BYTES = 384 << 20          # one unit ~ this many compressed parquet bytes (files are split by row group)
MIN_UNITS = 8                   # spread each (source, phase) quota over >= this many units (diversity)
ROWS_PER_GROUP = 1024
TASK_MAX_TOKENS = 3072          # pretraining seq len is 2048 (a final 4096 stage may follow): no task doc > 3072 tokens
SPECIAL_STRS = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<|turn>", "<turn|>", "<|tool>", "<tool|>",
                "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>",
                "<|channel>", "<channel|>"]
PERMISSIVE_CODE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"}
SHELL_LANGS = {"Shell", "Dockerfile", "Makefile", "PowerShell", "Batchfile"}
GENERAL_LANGS = {"Python", "JavaScript", "TypeScript", "GO", "Go", "Rust", "SQL", "Markdown", "YAML", "JSON"}

# ----------------------------------------------------------------------------------------------- the mix
BUCKETS = {
    # 2026-10-07 (user decision): Nemotron-CC-v2.1 is not used. web_hq = Ultra-FineWeb (en) instead of its High-Quality
    # real-web split; its High-Quality-Synthetic share (hq_synth, A .08) moved to finephrase (A .20 -> .28).
    "A": {"web_hq": .34, "finephrase": .28, "synth": .08, "math": .08, "code": .08, "wiki": .03, "task": .11},
    "B": {"web_hq": .20, "finephrase": .15, "synth": .15, "math": .10, "specialized": .05, "code": .05, "task": .30},
}
# Udgam-2 (2026-10-10, udgam2/PLAN.md P1): general-use + reasoning model, so the task share drops from 11%/30% to
# 4%/10%, and reasoning-dense SYNTH, math and code rise. Selected with --mix v2.
BUCKETS_V2 = {
    "A": {"web_hq": .30, "finephrase": .26, "synth": .14, "math": .12, "code": .10, "wiki": .04, "task": .04},
    "B": {"web_hq": .16, "finephrase": .16, "synth": .24, "math": .16, "specialized": .08, "code": .10, "task": .10},
}
TASK_SUBWEIGHTS_V2 = {"nemo_tool": .15, "xlam": .02, "table_gpt": .03, "nemo_chat": .80}
# Udgam-2 rerun (2026-10-11, user-approved "anneal mix tweak"): the first main run's anneal (phase B, 16% web) improved web
# bpb only 0.8666 -> 0.8505 while task/reasoning improved a lot; v2b keeps phase A = v2 and raises web in the anneal to 25%.
BUCKETS_V2B = {
    "A": dict(BUCKETS_V2["A"]),
    "B": {"web_hq": .25, "finephrase": .14, "synth": .22, "math": .14, "specialized": .07, "code": .09, "task": .09},
}
# Udgam-3 (2026-10-17; udgam3/PLAN_REVIEW.md, PREREGISTRATION.md 6.1): v3 = more real educational web (FineWeb-Edu,
# DCLM-Edu >= 3 next to Ultra-FineWeb), FineMath instead of Nemotron-CC-Math, chat rows as PLAIN TEXT (no chat special
# tokens in pretraining: OLMo 3 saw GSM8K 49 -> 0 with them), tool rows left to SFT. v3s = the same with ~42% structured
# rephrase in the stable phase (G1 arm D2). Phase B is the anneal (last ~20%).
BUCKETS_V3 = {
    "A": {"web_hq": .50, "finephrase": .22, "synth": .06, "code": .09, "math": .07, "wiki": .03, "task": .03},
    "B": {"web_hq": .32, "finephrase": .14, "synth": .14, "math": .12, "code": .10, "specialized": .08, "task": .07, "wiki": .03},
}
BUCKETS_V3S = {
    "A": {"web_hq": .30, "finephrase": .42, "synth": .06, "code": .09, "math": .07, "wiki": .03, "task": .03},
    "B": dict(BUCKETS_V3["B"]),
}
SUBWEIGHTS_V3 = {"web_hq": {"web_hq": .30, "fwedu": .35, "dclm_edu": .35}, "math": {"finemath": 1.0},
                 "task": {"nemo_chat": 1.0}}
TASK_PLAIN = False      # set by --mix v3/v3s: render chat rows as plain "User:/Assistant:" text
# sub-source weights inside each bucket (sum to 1 per bucket)
SUBWEIGHTS = {
    "web_hq": {"web_hq": 1.0},
    "synth": {"synth": 1.0},
    "math": {"math": 1.0},
    "wiki": {"wiki": 1.0},
    "finephrase": {"finephrase_faq": .25, "finephrase_table": .25, "finephrase_tutorial": .25, "finephrase_math": .25},
    "specialized": {"stem_sft": .5, "rqa": .5},
    "code": {"code_shell": .25, "code_general": .75},
    # task: tool-calling / JSON-structured group (.43) + chat / instruction group (.57). License audit 2026-10-06 removed
    # smol_smoltalk (.30), glaive_v2, hermes_fc, hermes_json, when2call and 5 of 6 smoltalk2 subsets (see
    # EXCLUDED_SOURCES); their weight went to the remaining sources of the same group. Availability caps then move any
    # excess to group-mates / the bucket (allocate()), so the task bucket's share of each phase is unchanged.
    # 2026-10-07 (user decision): Tulu-3 Persona IF dropped too (PersonaHub CC-BY-NC-SA seeds); its .05 -> nemo_chat.
    "task": {"nemo_tool": .36, "xlam": .04, "table_gpt": .03, "nemo_chat": .57},
}
NVIDIA_DATA_AGREEMENT = ("NVIDIA Data Agreement for Model Training (v. 2025-08-15; gated; permits training and owning "
                         "the resulting models; no redistribution of the data; no naming/licence flow-down to models)")

# ----------------------------------------------------------------------------------------------- license policy
# User decision 2026-10-06; the model ships under Apache-2.0 as a UdgamLabs model. Excluded: any data generated by a
# model whose licence reaches models trained on its outputs -- Meta Llama (any version: "include 'Llama' at the
# beginning of any such AI model name"), Gemma <=3 (Gemma Terms: models trained on synthetic Gemma outputs are "Model
# Derivatives"), Qwen-licence models (Qwen2.5-72B/3B, Qwen2-72B, Qwen1.x: "Built with Qwen" display duty for models
# trained on outputs), DeepSeek Model License models (DeepSeek-V2/V3-original/Coder/Prover: derivatives incl.
# distillation must carry the use restrictions), BigCode/BigScience OpenRAIL-M (same construction), Mistral non-Apache
# (MRL/MNPL), non-commercial or undisclosed licences/generators. Allowed: Apache/MIT generators (Qwen3, Apache Qwen2.5
# sizes, QwQ, DeepSeek-R1 / V3-0324+, Phi-3/4, Mixtral/Mistral-7B/NeMo, gpt-oss, SmolLM), NVIDIA Open Model License
# Nemotron models (outputs are not Derivative Models, no naming duty), OpenAI and Anthropic API outputs.
# Full audit with sources: data/LICENSES_AND_PROVENANCE.md.
#
# Rules are tried in order on the lower-cased model name (re.search); first match wins; NO match = not allowed.
GENERATOR_RULES = [
    (r"llama", "Llama Community License (naming clause on models trained on outputs)", False),  # incl. Llama-*-Nemotron
    (r"gemma", "Gemma Terms of Use (outputs-trained models are Model Derivatives; Gemma 4 not audited)", False),
    (r"deepseek-(v2|coder|prover|llm|math|moe|vl)|deepseek-v3$|deepseek-v3-base$",
     "DeepSeek License Agreement v1.0 (use restrictions flow to derivatives incl. distillation)", False),
    (r"deepseek-r1|deepseek-v3-0324|deepseek-v3\.\d", "MIT", True),
    (r"qwen2\.5-(math-)?(72b|3b)|qwen2-72b|qwen-(1\.8b|7b|14b|72b)|qwen1\.5",
     "Qwen License / Tongyi Qianwen License / Qwen Research License (3B: non-commercial); output-trained models must show 'Built with Qwen'", False),
    (r"qwen3|qwq-32b|qwen-3-|qwen2\.5-|qwen2-", "Apache-2.0", True),
    (r"phi-[34]", "MIT", True),
    (r"mixtral-8x(7|22)b|mistral-7b|mistral-nemo", "Apache-2.0", True),
    (r"mistral|codestral|ministral|magistral|pixtral", "Mistral Research / Non-Production License (or unverified)", False),
    (r"gpt-oss", "Apache-2.0", True),
    (r"smollm", "Apache-2.0", True),
    (r"starcoder|bloom|santacoder", "BigCode/BigScience OpenRAIL-M (use restrictions flow to derivatives)", False),
    (r"nemotron", "NVIDIA Open Model License (no rights claimed in outputs; outputs are not Derivative Models)", True),
    (r"^(gpt-4|gpt-3\.5|gpt-5|o1|o3|o4)|openai", "OpenAI API output (allowed by user decision)", True),
    (r"claude|anthropic", "Anthropic API output (allowed by user decision)", True),
]
_GEN_CACHE = {}


def generator_verdict(name):
    """(allowed, licence) for one generator string; '+'/','-joined multi-model names must ALL be allowed."""
    key = (name or "").strip()
    if key not in _GEN_CACHE:
        parts = [p.strip().lower() for p in re.split(r"[+,;]", key) if p.strip()]
        verdicts = []
        for p in parts or [""]:
            hit = next(((lic, ok) for rx, lic, ok in GENERATOR_RULES if re.search(rx, p)), ("unknown / undisclosed", False))
            verdicts.append(hit)
        _GEN_CACHE[key] = (all(ok for _, ok in verdicts), "; ".join(lic for lic, _ in verdicts))
    return _GEN_CACHE[key]


def generator_ok(name):
    return generator_verdict(name)[0]


def _P(dataset_license, generators, url, decision="keep", reason="", caveats="", row_filter=""):
    """Provenance record (manifest.json + data/LICENSES_AND_PROVENANCE.md). generators: [(model, licence, role)]."""
    return dict(dataset_license=dataset_license, generators=[dict(model=m, license=l, role=r) for m, l, r in generators],
                url=url, decision=decision, reason=reason, caveats=caveats, row_filter=row_filter)


ORGANIC = []           # no generated text
# Every source: HF glob(s) or files, reader, renderer, license, filters. avail_tokens (task sources) = unique
# tokens after our filters, measured on the 2026-10-06 sample build (tokens/doc x rows x keep-rate). `fallback` is used only with
# --allow-fallbacks when the primary is not accessible (e.g. gated access still pending).
# `gen` = row-level generator filter: `fields` (dotted paths tried in order, value may be 'A+B' / 'A, B'), else `id_ranges`
# [(lo, hi, model)] on rec['id'] (or the record's position), else `default`; every row whose generator is not
# generator_ok() is dropped ("generator_excluded"). `opt_cols` are projected only if the parquet file has them.
SOURCES = {
    # 2026-10-07 (user decision): Ultra-FineWeb English replaces Nemotron-CC-v2.1 High-Quality (no NVIDIA approval wait)
    "web_hq": dict(bucket="web_hq", reader="parquet", hf=["openbmb/Ultra-FineWeb/data/ultrafineweb_en/*.parquet"],
                   cols=["content"], make="text", text_col="content", min_chars=200,
                   license="Apache-2.0 (Ultra-FineWeb; English split built from FineWeb, ODC-By 1.0, Common Crawl)",
                   filters=["len>=200 chars", "mojibake / special-token-literal drops"],
                   prov=_P("Apache-2.0 (dataset); source pages from FineWeb (ODC-By 1.0) / Common Crawl", ORGANIC,
                           "https://huggingface.co/datasets/openbmb/Ultra-FineWeb",
                           reason="organic Common Crawl text (default config, `en` split = data/ultrafineweb_en), "
                                  "selected from FineWeb by OpenBMB's fastText quality classifier; ungated",
                           caveats="pages selected by a fastText classifier (selection only, no generated text); "
                                   "credit OpenBMB / Hugging Face FineWeb; Common Crawl terms of use apply"),
                   fallback=None),
    "math": dict(bucket="math", reader="parquet", hf=["nvidia/Nemotron-CC-Math-v1/4plus/*.parquet"], cols=["text"],
                 opt_cols=["models_used", "metadata"], make="text", min_chars=200,
                 license="NVIDIA Open Data License Agreement (gated; Phi-4 cleaning, MIT)", filters=["len>=200 chars"],
                 gen=dict(fields=["models_used", "metadata.models_used"], default="Phi-4"),
                 prov=_P("NVIDIA Open Data License Agreement (same text as the NVIDIA Data Agreement for Model Training)",
                         [("Phi-4", "MIT", "LLM cleaning / LaTeX normalisation of Common Crawl math pages"),
                          ("Qwen3-30B-A3B-Instruct-2507", "Apache-2.0", "quality scoring only (no text)")],
                         "https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1",
                         reason="organic math pages rewritten by Phi-4 (MIT); card's 'Phi-4 licence flow-down' is MIT, no restriction"),
                 fallback=None),   # gated access works (2026-10-07): no stand-in, a missing source is a hard error
    **{f"finephrase_{k}": dict(bucket="finephrase", reader="parquet", hf=[f"HuggingFaceFW/finephrase/{k}/*.parquet"],
                               cols=["rollout_results"], make="finephrase", min_chars=300, license="ODC-By 1.0",
                               filters=["rollout finish_reason=='stop'", "len>=300", "not startswith('No '|'Document')",
                                        "strip chat preambles ('Here is/Here's ...:' first line)"],
                               prov=_P("ODC-By 1.0", [("SmolLM2-1.7B-Instruct", "Apache-2.0", f"'{k}' rewrite of FineWeb-Edu pages")],
                                       "https://huggingface.co/datasets/HuggingFaceFW/finephrase",
                                       reason="card: generated with HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache-2.0)",
                                       caveats="seed pages are FineWeb-Edu (ODC-By), selected by a classifier trained on "
                                               "Llama-3-70B annotations (selection only)"))
       for k in ["faq", "table", "tutorial", "math"]},
    "synth": dict(bucket="synth", reader="parquet", hf=["PleIAs/SYNTH/synth_*.parquet"], split_files=False,
                  cols=["language", "exercise", "query", "synthetic_reasoning", "synthetic_answer", "constraints", "model"],
                  make="synth", min_chars=200, license="CC-BY-4.0 (seed texts CC-BY-SA-4.0 Wikipedia/Wikibooks, CC0)",
                  gen=dict(fields=["model"]),
                  filters=["language=='en'", "non-empty query/reasoning/answer",
                           "generator allow-list on `model` (drops deepseek-prover rows: DeepSeek Model License)",
                           "math exercises: drop if constraint result:X not present in the reasoning (inconsistent answer)",
                           "rag: sources from `constraints` prepended"],
                  prov=_P("CC-BY-4.0 (seeds: CC-BY-SA-4.0 Wikipedia/Wikibooks, CC0)",
                          [("Pleias fine-tunes of Qwen3-8B (qwen-3-8b-*)", "Apache-2.0 (base model)", "queries, reasoning, answers")],
                          "https://huggingface.co/datasets/PleIAs/SYNTH", decision="keep (row-filtered)",
                          reason="user decision: keep CC-BY-SA seeds with credit; Qwen3-8B fine-tunes are Apache-2.0",
                          caveats="credit Wikipedia/Wikibooks contributors (CC-BY-SA-4.0) and Pleias/AI Alliance in the model card",
                          row_filter="rows whose `model` names deepseek-prover-* (DeepSeek Model License; math exercise / "
                                     "math mcq) are dropped")),
    "code": dict(bucket="code", reader="parquet", hf=["codeparrot/github-code-clean/data/*.parquet"],
                 cols=["code", "language", "license"], make="code", min_chars=100, subs=["code_shell", "code_general"],
                 prefilter="code",
                 license="per-file permissive only: " + ", ".join(sorted(PERMISSIVE_CODE)),
                 filters=["license in permissive set", "len<20000 chars", "len>=100",
                          "languages: shell-family " + "/".join(sorted(SHELL_LANGS)) + " -> code_shell; "
                          "Python/JS/TS/Go/Rust/SQL/Markdown -> code_general (dataset has no YAML/JSON)"],
                 prov=_P("Apache-2.0 (dataset); files kept only if " + "/".join(sorted(PERMISSIVE_CODE)), ORGANIC,
                         "https://huggingface.co/datasets/codeparrot/github-code-clean",
                         reason="organic GitHub code, per-file permissive licence filter",
                         caveats="MIT/BSD/Apache files carry notice requirements; acknowledge in the model card")),
    "wiki": dict(bucket="wiki", reader="parquet", hf=["wikimedia/wikipedia/20231101.en/*.parquet"], cols=["title", "text"],
                 make="wiki", min_chars=500, license="CC-BY-SA-3.0 / GFDL",
                 filters=["len>=500", "drop disambiguation pages", "title prepended"],
                 prov=_P("CC-BY-SA-3.0 / GFDL", ORGANIC, "https://huggingface.co/datasets/wikimedia/wikipedia",
                         reason="user decision: keep with credit",
                         caveats="credit Wikipedia contributors (CC-BY-SA) in the model card")),
    "stem_sft": dict(bucket="specialized", reader="parquet", min_units=2, split_files=False,
                     hf=["nvidia/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-STEM-SFT/*.parquet"],
                     cols=["text", "license", "metadata"], make="specialized", min_chars=200,
                     license="CC-BY-4.0 (per-row license checked)", gen=dict(fields=["metadata.models_used"]),
                     filters=["row license == 'cc-by-4.0'", "generator allow-list (metadata.models_used)"],
                     prov=_P("CC-BY-4.0", [("DeepSeek-R1-0528", "MIT", "responses"), ("Qwen2.5-32B-Instruct", "Apache-2.0", "responses")],
                             "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1",
                             decision="keep (row-filtered)", reason="card + sampled rows: DeepSeek-R1-0528 (MIT), Qwen2.5-32B (Apache-2.0)",
                             caveats="NVIDIA's card mentions Qwen/DeepSeek licence flow-down generically; neither model here carries it",
                             row_filter="metadata.models_used allow-list")),
    "rqa": dict(bucket="specialized", reader="parquet",
                hf=["nvidia/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-RQA/*.parquet"],
                cols=["text", "license", "metadata"], make="specialized", min_chars=200, license="CC-BY-4.0 (per-row license checked)",
                gen=dict(fields=["metadata.models_used"]),
                filters=["row license == 'cc-by-4.0'", "generator allow-list (metadata.models_used)"], min_units=2, split_files=False,
                prov=_P("CC-BY-4.0", [("Qwen3-235B-A22B-Thinking-2507", "Apache-2.0", "questions + answers"),
                                      ("gpt-oss-120b", "Apache-2.0", "answers")],
                        "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1",
                        decision="keep (row-filtered)", reason="card + sampled rows: Qwen3-235B-A22B-Thinking-2507, gpt-oss-120b",
                        row_filter="metadata.models_used allow-list")),
    # ------------------------------------------------------------------ task / chat (rendered in Gemma 4 format)
    "nemo_tool": dict(bucket="task", group="tool", reader="parquet",
                      hf=["nvidia/Nemotron-Post-Training-Dataset-v1/data/tool-*.parquet"],
                      cols=["license", "generator", "messages", "metadata"], make="nemo_tool", task=True, avail_tokens=400e6,
                      license="CC-BY-4.0 (responses: Qwen3-235B-A22B, Apache-2.0)", gen=dict(fields=["generator"]),
                      filters=["generator allow-list (`generator` column)",
                               "<think> traces stripped (kept as Gemma thought channel for ~15% of convs whose traces are "
                               "all <=1500 chars, rendered with enable_thinking)", "non-empty user turns",
                               "tool args must parse to JSON objects"],
                      prov=_P("CC-BY-4.0", [("Qwen3-235B-A22B", "Apache-2.0", "responses / tool calls")],
                              "https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1",
                              decision="keep (row-filtered)",
                              reason="column stats: all tool_calling rows generator == Qwen3-235B-A22B",
                              caveats="dataset was built for Llama-3.3-Nemotron-Super-49B but contains no Llama outputs per "
                                      "the card; prompt generator for synthetic prompts not disclosed",
                              row_filter="`generator` allow-list")),
    "xlam": dict(bucket="task", group="tool", reader="json", hf_files=[("Salesforce/xlam-function-calling-60k", "xlam_function_calling_60k.json")],
                 make="xlam", task=True, avail_tokens=10.5e6, json_units=4, license="CC-BY-4.0 (Mixtral-8x22B rows only)",
                 gen=dict(id_ranges=[(0, 33658, "DeepSeek-V2-Chat"), (33659, None, "Mixtral-8x22B-Instruct-v0.1")]),
                 filters=["ids 0-33658 (DeepSeek-V2-Chat) dropped", "xlam param dicts -> JSON schema",
                          "answers -> assistant tool_calls"],
                 prov=_P("CC-BY-4.0", [("Mixtral-8x22B-Instruct-v0.1", "Apache-2.0", "queries + answers (ids 33659-59999)")],
                         "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
                         decision="keep (row-filtered)",
                         reason="card: ids 0-33658 generated by DeepSeek-V2-Chat (DeepSeek License v1.0 -> excluded), "
                                "the rest by Mixtral-8x22B (Apache-2.0)",
                         row_filter="id >= 33659"),
                 # no fallback: the ungated smoltalk2 re-host (xlam_traces) has no ids and re-generated responses, so the
                 # DeepSeek-V2 rows cannot be separated
                 fallback=None),
    "table_gpt": dict(bucket="task", group="tool", reader="parquet", hf=["HuggingFaceTB/smoltalk2/SFT/table_gpt_no_think-*.parquet"],
                      cols=["messages", "chat_template_kwargs"], make="smoltalk2", task=True, avail_tokens=11e6,
                      license="MIT (Table-GPT)", filters=["no <think>"],
                      prov=_P("MIT", [("OpenAI GPT (reasoning augmentation)", "OpenAI API output", "most completions are programmatic "
                                       "ground truth from real tables / benchmark labels; GPT added reasoning steps")],
                              "https://huggingface.co/datasets/LipengCS/Table-GPT",
                              reason="Table-GPT paper (arXiv 2310.09263): programmatic ground truth + GPT augmentation; "
                                     "smoltalk2 no_think split re-hosts the original",
                              caveats="tables come from web/database corpora and public benchmarks")),
    "nemo_chat": dict(bucket="task", group="chat", reader="parquet", hf=["nvidia/Nemotron-Post-Training-Dataset-v2/data/chat-*.parquet"],
                      # avail: 627,720 chat rows = 81,876 reasoning-off (~1.4K tok) + 545,844 reasoning-on (~1.1K tok after
                      # the trace is stripped), measured 2026-10-07 -> ~640M unique tokens after the 3072-token cap
                      cols=["license", "generator", "reasoning", "messages"], make="nemo_chat", task=True, avail_tokens=640e6,
                      license="CC-BY-4.0 (per-row; CC-BY-SA rows dropped)", gen=dict(fields=["generator"]),
                      filters=["generator allow-list (`generator` column)",
                               "reasoning 'off' rows as is; reasoning 'on' rows: <think> trace stripped, final answer kept "
                               "(trace kept as Gemma thought channel for ~15% of convs whose traces are all <=1500 chars)",
                               "per-row license not CC-BY-SA", "non-empty user turns", "reasoning-control system prompts removed"],
                      prov=_P("CC-BY-4.0 per row (some ODC-By WildChat prompts; CC-BY-SA rows dropped)",
                              [("Qwen3-235B-A22B-Instruct-2507", "Apache-2.0", "chat responses"),
                               ("Qwen3-30B-A3B", "Apache-2.0", "chat responses")],
                              "https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2",
                              decision="keep (row-filtered)",
                              reason="card: chat rows (627,720) generated by Qwen3-235B-A22B / Qwen3-30B-A3B; other v2 "
                                     "generators (DeepSeek-R1-0528, Qwen2.5-14B/32B) are also allowed",
                              caveats="gated; per-row generator values not inspected locally (403) -- the filter enforces them",
                              row_filter="`generator` allow-list"),
                      fallback=None),   # v1 chat has every user prompt blanked (LMSYS) -> unusable; no fallback
}
# Udgam-3 (2026-10-17): v3 adds educational web text and replaces the NVIDIA-licensed maths set with FineMath (ODC-By).
SOURCES_V3 = {
    "fwedu": dict(bucket="web_hq", reader="parquet", hf=["HuggingFaceFW/fineweb-edu/sample/100BT/*.parquet"],
                  cols=["text"], make="text", min_chars=200, license="ODC-By 1.0 (FineWeb-Edu; Common Crawl)",
                  filters=["len>=200 chars", "mojibake / special-token-literal drops"],
                  prov=_P("ODC-By 1.0; source pages from Common Crawl", ORGANIC,
                          "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                          reason="organic Common Crawl pages selected by an educational-quality classifier "
                                 "(sample/100BT, already filtered to edu score >= 3); ungated",
                          caveats="classifier selection only, no generated text; credit Hugging Face FineWeb-Edu; "
                                  "Common Crawl terms of use apply"),
                  fallback=None),
    "dclm_edu": dict(bucket="web_hq", reader="parquet", hf=["HuggingFaceTB/dclm-edu/data/*.parquet"],
                     cols=["text", "edu_int_score"], make="text", min_chars=200,
                     prefilter="edu3", license="CC-BY-4.0 (DCLM-Edu; DCLM-baseline, Common Crawl)",
                     filters=["edu_int_score >= 3", "len>=200 chars", "mojibake / special-token-literal drops"],
                     prov=_P("CC-BY-4.0; source pages from DCLM-baseline / Common Crawl", ORGANIC,
                             "https://huggingface.co/datasets/HuggingFaceTB/dclm-edu",
                             reason="organic DCLM-baseline pages re-scored with the FineWeb-Edu classifier (the SmolLM2 "
                                    "small-model data); we keep edu_int_score >= 3; ungated",
                             caveats="classifier selection only; attribution required (CC-BY-4.0)"),
                     fallback=None),
    "finemath": dict(bucket="math", reader="parquet", hf=["HuggingFaceTB/finemath/finemath-4plus/*.parquet"],
                     cols=["text"], make="text", min_chars=200, license="ODC-By 1.0 (FineMath-4+; Common Crawl)",
                     filters=["len>=200 chars", "mojibake / special-token-literal drops"],
                     prov=_P("ODC-By 1.0; source pages from Common Crawl", ORGANIC,
                             "https://huggingface.co/datasets/HuggingFaceTB/finemath",
                             reason="organic maths pages from Common Crawl selected by a maths-quality classifier "
                                    "(4+ subset); replaces Nemotron-CC-Math (NVIDIA Open Data License) in v3; ungated",
                             caveats="classifier selection only; credit Hugging Face FineMath"),
                     fallback=None),
}
V3_DROP = ["math", "nemo_tool", "xlam", "smoltalk2", "table_gpt"]   # NVIDIA-licensed maths; tool rows go to SFT instead
for _n, _s in list(SOURCES.items()) + list(SOURCES_V3.items()):
    _s.setdefault("subs", [_n]); _s.setdefault("task", False); _s.setdefault("max_epochs", 1); _s.setdefault("group", None)

# Sources (or parts) removed by the 2026-10-06 licence audit. Kept here (not built) so the decision is recorded in
# manifest.json and can be revisited; the makers for them still exist below.
_SMOLTALK = "https://huggingface.co/datasets/HuggingFaceTB/smoltalk"
EXCLUDED_SOURCES = {
    "smol_smoltalk": dict(hf="HuggingFaceTB/smol-smoltalk (train)", prev_weight="task .30", dataset_license="Apache-2.0",
                          url="https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk",
                          reason="no row source is allowed: smol-magpie-ultra-short 270,838 rows (Llama-3.1-405B-Instruct), "
                                 "smol-contraints 34,433 / smollm-rewrite-30k 26,657 / smol-summarize-5k+20k 24,021 "
                                 "(Qwen2.5-72B-Instruct, Qwen licence), everyday-conversations 2,252 (Llama-3.1-70B-Instruct), "
                                 "explore-instruct-rewrite 3,017 (CC-BY-NC-4.0), openhermes-50k 47,492 (no licence, mixed "
                                 "generators), self-oss-instruct 48,071 (StarCoder2-15B, BigCode OpenRAIL-M), longalign 3,560 "
                                 "(Claude 2.1 but LongAlign-10k has no stated licence)"),
    "smoltalk2:smoltalk_smollm3_systemchats_30k_no_think": dict(
        dataset_license="Apache-2.0 (SystemChat-2.0)", url="https://huggingface.co/datasets/cognitivecomputations/SystemChat-2.0",
        reason="generator not disclosed (SmolTalk2 no_think subsets re-host the original responses)"),
    "smoltalk2:smoltalk_smollm3_smol_rewrite_no_think": dict(
        dataset_license="Apache-2.0", url=_SMOLTALK,
        reason="Qwen2.5-72B-Instruct (Qwen licence) per SmolLM2 paper; responses identical to SmolTalk"),
    "smoltalk2:smoltalk_smollm3_smol_summarize_no_think": dict(
        dataset_license="Apache-2.0", url=_SMOLTALK,
        reason="Qwen2.5-72B-Instruct (Qwen licence) per SmolLM2 paper; verified a row is verbatim SmolTalk smol-summarize"),
    "smoltalk2:smoltalk_smollm3_explore_instruct_rewriting_no_think": dict(
        dataset_license="CC-BY-NC-4.0", url="https://huggingface.co/datasets/Wanfq/Explore_Instruct_Rewriting_32k",
        reason="non-commercial licence; generator not disclosed"),
    "smoltalk2:smoltalk_smollm3_everyday_conversations_no_think": dict(
        dataset_license="Apache-2.0", url="https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k",
        reason="generated by Llama-3.1-70B-Instruct"),
    "glaive_v2": dict(hf="glaiveai/glaive-function-calling-v2", prev_weight="task .05", dataset_license="Apache-2.0",
                      url="https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2",
                      reason="generator model not disclosed (Glaive's proprietary pipeline) -> unclear; excluded (user decision 2026-10-07)"),
    "hermes_fc": dict(hf="NousResearch/hermes-function-calling-v1 func-calling + glaive-function-calling-5k",
                      prev_weight="task .04", dataset_license="Apache-2.0",
                      url="https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1",
                      reason="generator model not disclosed (Nous in-house + Glaive) -> unclear; excluded (user decision 2026-10-07)"),
    "hermes_json": dict(hf="NousResearch/hermes-function-calling-v1 json-mode-*", prev_weight="task .03",
                        dataset_license="Apache-2.0", url="https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1",
                        reason="generator model not disclosed -> unclear; excluded (user decision 2026-10-07)"),
    "when2call": dict(hf="nvidia/When2Call train_sft", prev_weight="task .02", dataset_license="CC-BY-4.0",
                      url="https://huggingface.co/datasets/nvidia/When2Call",
                      reason="generated with Mixtral-8x22B (Apache) but built on APIGen/xLAM questions, ~56% of which "
                             "were generated by DeepSeek-V2-Chat (DeepSeek License); no per-row origin -> excluded (user decision 2026-10-07)"),
    "tulu_persona_if": dict(hf="HuggingFaceTB/smoltalk2 SFT tulu_3_sft_personas_instruction_following_no_think",
                            prev_weight="task .05", dataset_license="ODC-By 1.0",
                            url="https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following",
                            reason="user decision 2026-10-07: prompts were generated from PersonaHub personas "
                                   "(CC-BY-NC-SA-4.0, non-commercial) as seeds; response generator (GPT-4o) not stated"),
    "nemotron_cc_v2_1": dict(hf="nvidia/Nemotron-CC-v2.1 High-Quality (was web_hq) + High-Quality-Synthetic (was hq_synth, A .08)",
                             prev_weight="web_hq A .34 / B .20; hq_synth A .08", dataset_license=NVIDIA_DATA_AGREEMENT,
                             url="https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1",
                             reason="user decision 2026-10-07: not used (manual NVIDIA access approval not awaited). web_hq "
                                    "-> openbmb/Ultra-FineWeb en (Apache-2.0); hq_synth share -> HuggingFaceFW/finephrase"),
    "xlam:ids_0-33658": dict(dataset_license="CC-BY-4.0", url="https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
                             reason="generated by DeepSeek-V2-Chat (DeepSeek License Agreement v1.0); dropped row-level"),
    "synth:deepseek-prover rows": dict(dataset_license="CC-BY-4.0", url="https://huggingface.co/datasets/PleIAs/SYNTH",
                                       reason="math exercise / math mcq rows solved by DeepSeek-Prover models (DeepSeek "
                                              "Model License); dropped row-level"),
}


def policy_violations(spec):
    """Generators declared in a spec's provenance that the policy does not allow (must be empty to build)."""
    return [g["model"] for g in spec.get("prov", {}).get("generators", []) if not generator_ok(g["model"])]


# ----------------------------------------------------------------------------------------------- doc builders
def _clean_turn(s):
    return (s or "").strip()


def has_special(s):
    return any(t in s for t in SPECIAL_STRS)


_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_think(content):
    content = content or ""
    thinks = _THINK.findall(content)
    rest = _THINK.sub("", content)
    if "<think>" in rest:             # unterminated trace
        rest = rest.split("<think>")[0]
    return "\n".join(t.strip() for t in thinks if t.strip()), rest.strip()


_TYPE_MAP = {"str": "string", "string": "string", "text": "string", "int": "integer", "integer": "integer",
             "float": "number", "number": "number", "double": "number", "bool": "boolean", "boolean": "boolean",
             "list": "array", "array": "array", "tuple": "array", "set": "array", "dict": "object", "object": "object",
             "map": "object", "any": "string", "null": "string"}


def _norm_type(t):
    """Map loose type strings ('str, optional', 'List[int]', ['string','null']) -> (json type, items, optional)."""
    optional = False
    items = None
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    t = str(t or "string").strip()
    if t.endswith(", optional") or t.startswith("Optional["):
        optional = True
        t = t.replace(", optional", "")
        t = t[len("Optional["):-1] if t.startswith("Optional[") else t
    m = re.match(r"(?i)(list|array|tuple|set)\s*\[\s*(.+?)\s*\]$", t)
    if m:
        it, _, _ = _norm_type(m.group(2))
        return "array", {"type": it}, optional
    m = re.match(r"(?i)(dict|mapping)\s*\[.*\]$", t)
    if m:
        return "object", None, optional
    return _TYPE_MAP.get(t.lower(), "string"), items, optional


def norm_prop(p):
    if not isinstance(p, dict):
        return {"type": "string", "description": str(p)}
    out = {}
    typ, items, _ = _norm_type(p.get("type", "string"))
    if p.get("description"):
        out["description"] = str(p["description"])
    out["type"] = typ
    if typ == "string" and isinstance(p.get("enum"), list) and p["enum"]:
        out["enum"] = [str(x) for x in p["enum"]]
    if typ == "array":
        it = p.get("items") if isinstance(p.get("items"), dict) else (items or {"type": "string"})
        out["items"] = norm_schema(it) if "properties" in it else {"type": _norm_type(it.get("type", "string"))[0]}
    if typ == "object" and isinstance(p.get("properties"), dict):
        out["properties"] = {k: norm_prop(v) for k, v in p["properties"].items()}
        if isinstance(p.get("required"), list):
            out["required"] = [str(r) for r in p["required"] if r in out["properties"]]
    if p.get("nullable") is True:
        out["nullable"] = True
    return out


def norm_schema(params):
    """Any parameters spec (JSON-schema, xLAM-style {param: {type, description, default}}, None) -> JSON schema object."""
    if not params:
        return {"type": "object", "properties": {}}
    if isinstance(params, str):
        params = json.loads(params)
    if "properties" in params and isinstance(params["properties"], dict):
        props = {k: norm_prop(v) for k, v in params["properties"].items()}
        req = params.get("required")
        req = [str(r) for r in req if r in props] if isinstance(req, list) else []
        out = {"type": "object", "properties": props}
        if req:
            out["required"] = req
        return out
    props, req = {}, []
    for k, v in params.items():
        if k in ("type", "required"):
            continue
        props[k] = norm_prop(v)
        opt = isinstance(v, dict) and (_norm_type(v.get("type"))[2] or "default" in v)
        if not opt:
            req.append(k)
    out = {"type": "object", "properties": props}
    if req:
        out["required"] = req
    return out


_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")


def norm_tool(t):
    if isinstance(t, str):
        t = json.loads(t)
    f = t["function"] if isinstance(t, dict) and "function" in t and isinstance(t["function"], dict) else t
    name = str(f["name"]).strip()
    if not _TOOL_NAME.match(name):                  # e.g. glaive's "calculate BMI": not a callable identifier
        raise ValueError("invalid tool name")
    return {"type": "function", "function": {"name": name, "description": str(f.get("description") or "").strip(),
                                             "parameters": norm_schema(f.get("parameters"))}}


def parse_args(a):
    if a is None or a == "":
        return {}
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a)
    except Exception:
        v = ast.literal_eval(a)
    if isinstance(v, str):         # double-encoded
        v = json.loads(v)
    if not isinstance(v, dict):
        raise ValueError("tool arguments are not an object")
    return v


def parse_jsonish(s):
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def tool_content(c):
    """Tool responses are passed as strings, exactly as Ollama / llama.cpp / OpenAI-style clients send them
    (Gemma renders them as response:name{value:<|"|>...<|"|>}). Dict content is NOT usable: the HF template
    treats any mapping as a list of content parts and fails."""
    if isinstance(c, str):
        s = c.strip()
        try:                                    # compact JSON so whitespace-heavy dumps don't waste tokens
            return json.dumps(json.loads(s), ensure_ascii=False)
        except Exception:
            return s
    return json.dumps(c, ensure_ascii=False)


_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TR = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def xml_calls(content, tool_names=()):
    calls = []
    for blob in _TC.findall(content or ""):
        blob = blob.replace("\\n", "\n").strip()        # hermes "ExpertQA" rows carry literal \n escapes
        d = parse_jsonish(blob)
        args = d.get("arguments", d.get("parameters"))
        if "name" not in d and isinstance(args, dict) and args.get("name") in tool_names:
            args = dict(args); d["name"] = args.pop("name")  # name mis-nested inside arguments
        calls.append({"type": "function", "function": {"name": d["name"], "arguments": parse_args(args)}})
    return calls, _TC.sub("", content or "").strip()


def conv_text(msgs):
    """Plain text of a conversation (no tool declarations / system) for MinHash near-dup detection."""
    parts = []
    for m in msgs:
        if m["role"] == "system":
            continue
        c = m.get("content")
        parts.append(c if isinstance(c, str) else json.dumps(c, sort_keys=True, ensure_ascii=False))
        for tc in m.get("tool_calls") or []:
            parts.append(tc["function"]["name"] + " " + json.dumps(tc["function"]["arguments"], sort_keys=True))
    return "\n".join(parts)


class Ctx:
    """Per-worker heavy state: tokenizer, chat renderer, decontam index."""
    def __init__(self, index_path):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(TOKENIZER)
        self.ix = decontam.Index(index_path)
        self._chat = None

    @property
    def chat(self):
        if self._chat is None:
            import transformers
            transformers.logging.set_verbosity_error()
            from transformers import PreTrainedTokenizerFast
            tk = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER, bos_token="<bos>", eos_token="<eos>",
                                         pad_token="<pad>", unk_token="<unk>")
            tk.chat_template = open(CHAT_TEMPLATE).read()
            self._chat = tk
        return self._chat

    def render(self, messages, tools=None, enable_thinking=False):
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and has_special(c):
                raise ValueError("special_token_literal")
            if m.get("reasoning_content") and has_special(m["reasoning_content"]):
                raise ValueError("special_token_literal")
        if TASK_PLAIN:                          # v3: plain text, no chat special tokens (tool rows are not built)
            if tools or any(m.get("tool_calls") or m["role"] == "tool" for m in messages):
                raise ValueError("plain_text_no_tools")
            names = {"system": "System", "user": "User", "assistant": "Assistant"}
            parts = []
            for m in messages:
                c = (m.get("content") or "").strip()
                if m["role"] == "assistant" and enable_thinking and (m.get("reasoning_content") or "").strip():
                    c = "Let me think it through.\n" + m["reasoning_content"].strip() + "\n\nAnswer: " + c
                if c:
                    parts.append(f"{names.get(m['role'], m['role'].title())}: {c}")
            return "\n\n".join(parts) + "\n"
        s = self.chat.apply_chat_template(messages, tools=tools or None, tokenize=False, enable_thinking=enable_thinking)
        assert s.startswith("<bos>"), s[:20]
        return s[len("<bos>"):]                 # the dataloader prepends BOS itself


DROP = "drop"


def chunk_doc(text, tok, max_tokens, n_tokens=None):
    """Split a long non-task doc into pieces of <= max_tokens tokens, cutting at the last paragraph / line /
    sentence break inside each window (nanochat's best-fit packer crops anything longer than the row, so
    without this the tail of every long doc is silently thrown away and long-doc sources are under-trained).
    Returns [(text, n_tokens)]."""
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) <= max_tokens:
        return [(text, len(enc.ids))]
    offs = enc.offsets
    out, start_tok, start_chr = [], 0, 0
    n = len(enc.ids)
    while start_tok < n:
        end_tok = min(n, start_tok + max_tokens)
        if end_tok == n:
            cut = len(text)
        else:
            limit = offs[end_tok - 1][1]
            lo = offs[start_tok + max_tokens // 2][0]
            cut = -1
            for sep in ("\n\n", "\n", ". ", " "):
                j = text.rfind(sep, lo, limit)
                if j > start_chr:
                    cut = j + len(sep)
                    break
            if cut <= start_chr:
                cut = limit
        piece = text[start_chr:cut].strip()
        if len(piece) >= 100:
            ids = tok.encode(piece, add_special_tokens=False).ids
            if len(ids) > max_tokens:                      # rare: boundary effects -> hard cut on token offsets
                o2 = tok.encode(piece, add_special_tokens=False).offsets
                piece = piece[:o2[max_tokens - 1][1]]
                ids = tok.encode(piece, add_special_tokens=False).ids
            out.append((piece, len(ids)))
        start_chr = cut
        while start_tok < n and offs[start_tok][0] < cut:
            start_tok += 1
    return out


def mojibake(t):
    return t.count("\u00e2\u20ac") >= 3 or t.count("\u00c3\u00a9") >= 3


def mk_text(rec, ctx, rng, src):
    t = rec[src.get("text_col", "text")] or ""
    return [(src["subs"][0], t, None)]


def mk_finephrase(rec, ctx, rng, src):
    rr = rec["rollout_results"]
    if isinstance(rr, str):
        rr = ast.literal_eval(rr)
    out = []
    for x in rr or []:
        t = (x.get("text") or "").strip()
        if x.get("finish_reason") != "stop":
            out.append((DROP, "finephrase_not_stop")); continue
        if len(t) < 300:
            out.append((DROP, "finephrase_short")); continue
        if t.startswith(("No ", "Document")):
            out.append((DROP, "finephrase_failed_rollout")); continue
        if t.startswith(("The document", "This document", "The text", "This text", "The passage", "The article")):
            out.append((DROP, "finephrase_meta_summary")); continue
        first, _, rest = t.partition("\n")
        if re.match(r"^(Here is|Here's|Below is|Sure[,!])[^\n]{0,160}:\s*$", first) and rest.strip():
            t = rest.strip()                              # chat-style preamble line
        out.append((src["subs"][0], t, None))
    return out


def mk_synth(rec, ctx, rng, src):
    if rec["language"] != "en":
        return [(DROP, "synth_not_english")]
    q, rs, a = _clean_turn(rec["query"]), _clean_turn(rec["synthetic_reasoning"]), _clean_turn(rec["synthetic_answer"])
    if not (q and rs and a):
        return [(DROP, "synth_empty_field")]
    ex = rec.get("exercise") or ""
    cons = rec.get("constraints") or ""
    if ex.startswith("math") and "result:" in cons:
        want = cons.split("result:", 1)[1].split(",")[0].strip()
        if want and want.replace(",", "") not in rs.replace(",", ""):
            return [(DROP, "synth_math_inconsistent")]
    prefix = ""
    if ex == "rag" and "<source_" in cons:
        prefix = "Sources:\n" + cons.strip() + "\n\n"
    return [(src["subs"][0], f"{prefix}Question: {q}\n\nReasoning:\n{rs}\n\nAnswer:\n{a}", None)]


def mk_code(rec, ctx, rng, src):
    if rec["license"] not in PERMISSIVE_CODE:
        return [(DROP, "code_license")]
    code = rec["code"] or ""
    if len(code) >= 20000:
        return [(DROP, "code_too_long")]
    if code.startswith("version https://git-lfs"):
        return [(DROP, "code_git_lfs_pointer")]
    head = code[:600].lower()
    if any(k in head for k in ("auto-generated", "autogenerated", "generated by", "do not edit", "code generated")):
        return [(DROP, "code_autogenerated")]
    lang = rec["language"]
    if lang in SHELL_LANGS:
        return [("code_shell", code, None)]
    if lang in GENERAL_LANGS:
        return [("code_general", code, None)]
    return [(DROP, "code_language_not_selected")]


def mk_wiki(rec, ctx, rng, src):
    t = (rec["text"] or "").strip()
    if "may refer to:" in t[:400]:
        return [(DROP, "wiki_disambiguation")]
    return [(src["subs"][0], f"{rec['title']}\n\n{t}", None)]


def mk_specialized(rec, ctx, rng, src):
    if (rec.get("license") or "").lower() != "cc-by-4.0":
        return [(DROP, "license_not_cc_by_4")]
    return [(src["subs"][0], rec["text"], None)]


def _task(src, ctx, msgs, tools=None, think=False):
    while msgs and msgs[-1]["role"] == "user":              # trailing "thanks!" with no reply -> trim
        msgs = msgs[:-1]
    if not msgs or not any(m["role"] == "user" and _clean_turn(m.get("content")) for m in msgs):
        return [(DROP, "task_no_user_turn")]
    if any(m["role"] == "user" and not _clean_turn(m.get("content")) for m in msgs):
        return [(DROP, "task_empty_user_turn")]
    if not any(m["role"] == "assistant" and (_clean_turn(m.get("content")) or m.get("tool_calls")) for m in msgs):
        return [(DROP, "task_no_assistant_output")]
    if msgs[-1]["role"] == "tool":
        return [(DROP, "ends_with_tool_response")]       # truncated conversation: no final model turn
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if not _TOOL_NAME.match(tc["function"]["name"]):
                return [(DROP, "invalid_tool_name")]
    try:
        text = ctx.render(msgs, tools=tools, enable_thinking=think)
    except ValueError as e:
        return [(DROP, "special_token_literal" if "special_token_literal" in str(e) else "render_error")]
    except Exception:
        return [(DROP, "render_error")]
    return [(src["subs"][0], text, conv_text(msgs))]


def mk_nemo_tool(rec, ctx, rng, src):
    if "sa" in (rec.get("license") or "").lower().replace("usage", ""):
        return [(DROP, "license_cc_by_sa")]
    try:
        meta = json.loads(rec["metadata"]) if isinstance(rec["metadata"], str) else (rec["metadata"] or {})
        tools = [norm_tool(t) for t in meta.get("tools") or []]
    except Exception:
        return [(DROP, "tool_schema_unparseable")]
    keep_think_cand = rng.random() < 0.15
    msgs, pending, thinks = [], [], []
    try:
        for m in rec["messages"]:
            role = m["role"]
            if role == "assistant":
                th, rest = split_think(m.get("content"))
                thinks.append(th)
                d = {"role": "assistant", "content": rest}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"type": "function", "function": {"name": tc["function"]["name"],
                                                                         "arguments": parse_args(tc["function"]["arguments"])}}
                                       for tc in m["tool_calls"]]
                    pending = [tc["function"]["name"] for tc in d["tool_calls"]]
                if th:
                    d["reasoning_content"] = th
                msgs.append(d)
            elif role == "tool":
                name = pending.pop(0) if pending else "unknown"
                msgs.append({"role": "tool", "name": name, "content": tool_content(m.get("content"))})
            elif role in ("user", "system"):
                msgs.append({"role": role, "content": _clean_turn(m.get("content"))})
    except Exception:
        return [(DROP, "tool_args_unparseable")]
    think = keep_think_cand and thinks and all(len(t) <= 1500 for t in thinks) and any(thinks)
    if not think:
        for d in msgs:
            d.pop("reasoning_content", None)
    return _task(src, ctx, msgs, tools, think=bool(think))


def _xlam_core(query, tools, calls, src, ctx):
    try:
        tools = [norm_tool(t) for t in tools]
        calls = [{"type": "function", "function": {"name": c["name"], "arguments": parse_args(c.get("arguments"))}} for c in calls]
    except Exception:
        return [(DROP, "tool_schema_unparseable")]
    return _task(src, ctx, [{"role": "user", "content": query}, {"role": "assistant", "content": "", "tool_calls": calls}], tools)


def mk_xlam(rec, ctx, rng, src):
    try:
        return _xlam_core(rec["query"], json.loads(rec["tools"]), json.loads(rec["answers"]), src, ctx)
    except Exception:
        return [(DROP, "tool_schema_unparseable")]


def _xml_tools(kw):
    out = []
    for blob in (kw or {}).get("xml_tools") or []:
        blob = blob.strip()
        blob = blob[len("<tools>"):] if blob.startswith("<tools>") else blob
        blob = blob[:-len("</tools>")] if blob.endswith("</tools>") else blob
        v = json.loads(blob)
        out += v if isinstance(v, list) else [v]
    return out


def mk_smoltalk2_xlam(rec, ctx, rng, src):
    try:
        tools = [norm_tool(t) for t in _xml_tools(rec["chat_template_kwargs"])]
        msgs = []
        for m in rec["messages"]:
            if m["role"] == "assistant":
                calls, rest = xml_calls(m["content"])
                d = {"role": "assistant", "content": rest}
                if calls:
                    d["tool_calls"] = calls
                msgs.append(d)
            else:
                msgs.append({"role": m["role"], "content": _clean_turn(m["content"])})
    except Exception:
        return [(DROP, "tool_schema_unparseable")]
    return _task(src, ctx, msgs, tools)


def mk_smoltalk2(rec, ctx, rng, src):
    msgs = []
    kw = rec.get("chat_template_kwargs") or {}
    if _clean_turn(kw.get("custom_instructions")):
        msgs.append({"role": "system", "content": kw["custom_instructions"].strip()})
    for m in rec["messages"]:
        c = m["content"] or ""
        if "<think>" in c:
            th, c = split_think(c)
            if th:
                return [(DROP, "task_has_reasoning_trace")]
        msgs.append({"role": m["role"], "content": c.strip()})
    return _task(src, ctx, msgs)


def mk_chat(rec, ctx, rng, src):
    return _task(src, ctx, [{"role": m["role"], "content": _clean_turn(m["content"])} for m in rec["messages"]])


_REASON_CTRL = re.compile(r"^\s*(/no_think|/think|detailed thinking (on|off))\s*$", re.I)


def mk_nemo_chat(rec, ctx, rng, src):
    """2026-10-07: only ~0.1% of the v2 chat rows are reasoning 'off' (measured on the full split), far too few for
    the nemo_chat share. Reasoning-'on' rows are used too: the <think> trace is stripped and the final answer kept;
    like nemo_tool, ~15% of convs whose traces are all <= 1500 chars keep it as the Gemma thought channel."""
    if "sa" in (rec.get("license") or "").lower():
        return [(DROP, "license_cc_by_sa")]
    keep_think_cand = rec.get("reasoning") == "on" and rng.random() < 0.15
    msgs, thinks = [], []
    for m in rec["messages"]:
        c = m["content"] or ""
        if m["role"] == "system":
            c = _REASON_CTRL.sub("", c).replace("/no_think", "").replace("/think", "").strip()
            if not c:
                continue
        if m["role"] == "assistant":
            th, c = split_think(c)
            thinks.append(th)
            d = {"role": "assistant", "content": c.strip()}
            if th:
                d["reasoning_content"] = th
            msgs.append(d)
            continue
        msgs.append({"role": m["role"], "content": c.strip()})
    think = keep_think_cand and thinks and all(len(t) <= 1500 for t in thinks) and any(thinks)
    if not think:
        for d in msgs:
            d.pop("reasoning_content", None)
    return _task(src, ctx, msgs, think=bool(think))


_HERMES_SYS = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)


def mk_hermes_fc(rec, ctx, rng, src):
    try:
        conv = rec["conversations"]
        tools = rec.get("tools")
        if isinstance(tools, str) and tools.strip():
            tools = json.loads(tools)
        if not tools:
            m = _HERMES_SYS.search(conv[0]["value"]) if conv and conv[0]["from"] == "system" else None
            tools = parse_jsonish(m.group(1)) if m else []
        tools = [norm_tool(t) for t in (tools if isinstance(tools, list) else [tools])]
        if not tools:
            return [(DROP, "no_tools")]                   # glaive-5k plain chats: not tool data
        msgs, pending = [], []
        for c in conv:
            fr, v = c["from"], c["value"] or ""
            if fr == "system":
                continue                                  # hermes boilerplate + tool list -> native tool declarations
            if fr == "human":
                msgs.append({"role": "user", "content": v.strip()})
            elif fr == "gpt":
                calls, rest = xml_calls(v, {t["function"]["name"] for t in tools})
                d = {"role": "assistant", "content": rest}
                if calls:
                    d["tool_calls"] = calls
                    pending = [x["function"]["name"] for x in calls]
                msgs.append(d)
            elif fr == "tool":
                blobs = _TR.findall(v) or [v]
                for b in blobs:
                    try:
                        d = parse_jsonish(b)
                        name, content = d.get("name") or (pending[0] if pending else "unknown"), d.get("content", d)
                    except Exception:
                        name, content = (pending[0] if pending else "unknown"), b
                    if pending:
                        pending.pop(0)
                    msgs.append({"role": "tool", "name": name, "content": tool_content(content)})
    except Exception:
        return [(DROP, "tool_schema_unparseable")]
    return _task(src, ctx, msgs, tools)


def mk_hermes_json(rec, ctx, rng, src):
    msgs = []
    for c in rec["conversations"]:
        role = {"system": "system", "human": "user", "gpt": "assistant"}.get(c["from"])
        if role is None:
            continue
        v = (c["value"] or "").strip()
        if role == "assistant":
            try:
                json.loads(v)
            except Exception:
                return [(DROP, "json_output_invalid")]
        msgs.append({"role": role, "content": v})
    return _task(src, ctx, msgs)


_GL_SPLIT = re.compile(r"(?:^|\n+)\s*(USER|ASSISTANT|FUNCTION RESPONSE):\s?")


def mk_glaive(rec, ctx, rng, src):
    try:
        sysmsg = rec["system"] or ""
        tools = []
        if "-" in sysmsg and "{" in sysmsg:
            body = sysmsg[sysmsg.index("{"):]
            dec = json.JSONDecoder()
            i = 0
            while i < len(body):
                while i < len(body) and body[i] in " \n\r\t,":
                    i += 1
                if i >= len(body):
                    break
                obj, i = dec.raw_decode(body, i)
                tools.append(norm_tool(obj))
        parts = _GL_SPLIT.split(rec["chat"])
        msgs, pending = [], None
        for role, txt in zip(parts[1::2], parts[2::2]):
            txt = txt.replace("<|endoftext|>", "").strip()
            if role == "USER":
                msgs.append({"role": "user", "content": txt})
            elif role == "ASSISTANT":
                if "<functioncall>" in txt:
                    pre, payload = txt.split("<functioncall>", 1)
                    name = re.search(r'"name"\s*:\s*"([^"]+)"', payload).group(1)
                    am = re.search(r'"arguments"\s*:\s*\'(.*)\'\s*}\s*$', payload.strip(), re.S)
                    args = parse_args(am.group(1)) if am else parse_args(json.loads(payload.strip()).get("arguments"))
                    msgs.append({"role": "assistant", "content": pre.strip(),
                                 "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]})
                    pending = name
                else:
                    msgs.append({"role": "assistant", "content": txt})
            else:
                msgs.append({"role": "tool", "name": pending or "unknown", "content": tool_content(txt)})
    except Exception:
        return [(DROP, "tool_schema_unparseable")]
    if not tools:
        return [(DROP, "no_tools")]                       # glaive "no access to functions" chats (GPT-3.5 era)
    last = next((m for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), None)
    if last and len(last["content"]) > 200 and not re.search(r"[.!?)\]}`*:\"'>|]\s*$", last["content"]):
        return [(DROP, "truncated_response")]              # glaive v2 has responses cut mid-sentence
    return _task(src, ctx, msgs, tools)


def mk_when2call(rec, ctx, rng, src):
    try:
        tools = [norm_tool(t) for t in rec.get("tools") or []]
    except Exception:
        return [(DROP, "tool_schema_unparseable")]
    return _task(src, ctx, [{"role": m["role"], "content": _clean_turn(m["content"])} for m in rec["messages"]], tools)


def mk_extra(rec, ctx, rng, src):
    if "text" in rec and "messages" not in rec:
        # pre-rendered (udgam/gold/render_for_pretrain.py): MinHash on the conversation text only, so records that
        # share long tool declarations are not mistaken for near-duplicates
        if "<bos>" in rec["text"]:
            return [(DROP, "bad_prerendered_text")]
        return [(src["subs"][0], rec["text"], rec.get("mh_text") or rec["text"])]
    try:
        tools = [norm_tool(t) for t in rec.get("tools") or []]
        msgs = []
        for m in rec["messages"]:
            m = dict(m)
            if m.get("role") == "tool":
                m["content"] = tool_content(m.get("content"))
            for tc in m.get("tool_calls") or []:
                tc["function"]["arguments"] = parse_args(tc["function"].get("arguments"))
            msgs.append(m)
    except Exception:
        return [(DROP, "tool_schema_unparseable")]
    return _task(src, ctx, msgs, tools, think=bool(rec.get("enable_thinking")))


MAKERS = {"text": mk_text, "finephrase": mk_finephrase, "synth": mk_synth, "code": mk_code, "wiki": mk_wiki,
          "specialized": mk_specialized, "nemo_tool": mk_nemo_tool, "xlam": mk_xlam, "smoltalk2_xlam": mk_smoltalk2_xlam,
          "smoltalk2": mk_smoltalk2, "chat": mk_chat, "nemo_chat": mk_nemo_chat, "hermes_fc": mk_hermes_fc,
          "hermes_json": mk_hermes_json, "glaive": mk_glaive, "when2call": mk_when2call, "extra": mk_extra}

# ----------------------------------------------------------------------------------------------- hashing
_MH_RNG = np.random.default_rng(0xD0D0)
MH_A = _MH_RNG.integers(1, 2**63 - 1, size=128, dtype=np.int64).astype(np.uint64) | np.uint64(1)
MH_B = _MH_RNG.integers(1, 2**63 - 1, size=128, dtype=np.int64).astype(np.uint64)
MH_C = _MH_RNG.integers(1, 2**63 - 1, size=8, dtype=np.int64).astype(np.uint64) | np.uint64(1)
MH_BANDS, MH_ROWS = 16, 8      # P(flag) ~0.95 at Jaccard 0.8, ~0.06 at 0.5 (word 5-gram shingles)


def minhash_bands(text):
    g, _ = decontam.ngram_hashes(text, n=5)
    if g.size == 0:
        return []
    g = np.unique(g)[:20000]
    h = g[None, :] * MH_A[:, None] + MH_B[:, None]
    h ^= h >> np.uint64(31)
    sig = h.min(axis=1).reshape(MH_BANDS, MH_ROWS)
    return (sig * MH_C[None, :]).sum(axis=1).tolist()


def text_hash(text, epoch):
    h = int.from_bytes(hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")
    return h ^ ((epoch - 1) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)


# ----------------------------------------------------------------------------------------------- readers
class CountingFile:
    """Wrap an fsspec file to count bytes actually fetched by pyarrow (download volume per source)."""
    def __init__(self, f):
        self.f, self.nbytes = f, 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.nbytes += len(b)
        return b

    def __getattr__(self, k):
        return getattr(self.f, k)


_FS = None


def hffs():
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem()
    return _FS


def code_prefilter(batch, want_subs):
    """Vectorised pre-filter for github-code-clean (most rows are non-permissive or unselected languages):
    avoids converting ~95% of rows to Python objects."""
    import pyarrow.compute as pc
    langs = set()
    if "code_shell" in want_subs:
        langs |= SHELL_LANGS
    if "code_general" in want_subs:
        langs |= GENERAL_LANGS
    m = pc.and_(pc.is_in(batch.column("license"), pa.array(sorted(PERMISSIVE_CODE))),
                pc.is_in(batch.column("language"), pa.array(sorted(langs))))
    m = pc.and_(m, pc.less(pc.utf8_length(batch.column("code")), 20000))
    return batch.filter(m)


def edu3_prefilter(batch, want_subs):
    """DCLM-Edu rows with edu_int_score >= 3 (Udgam-3 v3)."""
    import pyarrow.compute as pc
    return batch.filter(pc.greater_equal(batch.column("edu_int_score"), 3))


PREFILTERS = {"code": code_prefilter, "edu3": edu3_prefilter}


def iter_parquet_unit(u, cols, st, prefilter=None, want_subs=(), opt_cols=()):
    f = CountingFile(hffs().open("datasets/" + u["path"], block_size=16 << 20))
    try:
        pf = pq.ParquetFile(f, buffer_size=8 << 20, pre_buffer=False)
        have = set(pf.schema_arrow.names)
        cols = list(cols) + [c for c in opt_cols if c in have and c not in cols]
        n = pf.metadata.num_row_groups
        rgs = [i for i in range(n) if i % u["K"] == u["k"]]
        g0, r0 = u["pos"]
        for gi in range(g0, len(rgs)):
            row = 0
            for b in pf.iter_batches(batch_size=256, row_groups=[rgs[gi]], columns=cols, use_threads=False):
                if gi == g0 and row + b.num_rows <= r0:
                    row += b.num_rows
                    continue
                if prefilter is not None:          # resume positions then count post-filter rows (deterministic)
                    nb = b.num_rows
                    b = prefilter(b, want_subs)
                    st["prefiltered"] += nb - b.num_rows
                recs = b.to_pylist()
                for r in recs:
                    row += 1
                    if gi == g0 and row <= r0:
                        continue
                    yield (gi, row), r
    finally:
        st["bytes_read"] += f.nbytes
        f.f.close()


_JSON_CACHE = {}


def load_json_records(files):
    key = tuple(map(tuple, files))
    if key not in _JSON_CACHE:
        from huggingface_hub import hf_hub_download
        recs = []
        for repo, fn in files:
            if repo == "local":
                p = fn
            else:
                p = hf_hub_download(repo, fn, repo_type="dataset")
            if p.endswith(".jsonl"):
                part = [json.loads(l) for l in open(p) if l.strip()]
            else:
                part = json.load(open(p))
            for i, r in enumerate(part):          # position in its file (row-level generator filters by id/position)
                if isinstance(r, dict):
                    r.setdefault("_idx", i)
            recs += part
        _JSON_CACHE.clear()
        _JSON_CACHE[key] = recs
    return _JSON_CACHE[key]


def iter_json_unit(u, st):
    recs = load_json_records(u["files"])
    mine = range(u["k"], len(recs), u["K"])
    start = u["pos"][1]
    for j in range(start, len(mine)):
        yield (0, j + 1), recs[mine[j]]


# ----------------------------------------------------------------------------------------------- worker
_CTX = None


def _init_worker(index_path, short_min_hits=decontam.SHORT_DOC_MIN_HITS, task_plain=False):
    global _CTX, TASK_PLAIN
    TASK_PLAIN = task_plain                  # v3: also correct under the spawn start method (macOS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    decontam.SHORT_DOC_MIN_HITS = short_min_hits
    _CTX = Ctx(index_path)


def row_generator(rec, gen):
    """Generator model name of one record per the spec's `gen` rule (see SOURCES), or None if unknown."""
    for path in gen.get("fields") or []:
        v = rec
        for k in path.split("."):
            if isinstance(v, str) and v.lstrip().startswith("{"):
                try:
                    v = json.loads(v)
                except Exception:
                    v = None
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (list, tuple)):
            v = "+".join(map(str, v))
        if v:
            return str(v)
    if gen.get("id_ranges"):
        i = rec.get("id", rec.get("_idx"))
        try:
            i = int(i)
        except (TypeError, ValueError):
            return None
        for lo, hi, model in gen["id_ranges"]:
            if i >= lo and (hi is None or i <= hi):
                return model
        return None
    return gen.get("default")


def run_unit(task):
    """Stream one unit, emit docs until its per-sub token budgets are met. Returns stats; writes a part file."""
    t0, c0 = time.time(), time.process_time()
    ctx = _CTX
    src = task["src"]
    u = task["unit"]
    budgets = task["budgets"]
    produced = {s: 0 for s in budgets}
    st = dict(records=0, bytes_read=0, prefiltered=0, drops=collections.Counter(), decontam=collections.Counter(), decon_examples=[],
              docs=collections.Counter(), chars=collections.Counter(), over_quota=0, local_dups=0, single_hit_kept=0,
              chunked_docs=0, task_over_2048=0, gen_rows=collections.Counter())
    rng = random.Random(task["uid"])
    maker = MAKERS[src["make"]]
    cols = {k: [] for k in ("text", "ntok", "nchar", "h", "sub", "mh")}
    seen = set()
    ep_salt = (u["epoch"] - 1) * 0xC2B2AE3D27D4EB4F & 0xFFFFFFFFFFFFFFFF   # planned repeats are not near-dups
    exhausted, pos = True, u["pos"]
    if src["reader"] == "parquet":
        it = iter_parquet_unit(u, src["cols"], st, PREFILTERS.get(src.get("prefilter")),
                               {k for k, v in budgets.items() if v > 0}, src.get("opt_cols", ()))
    else:
        it = iter_json_unit(u, st)
    gen = src.get("gen")
    try:
        for pos, rec in it:
            st["records"] += 1
            if gen:                                        # licence policy: row-level generator allow-list
                g = row_generator(rec, gen)
                if g is None or not generator_ok(g):
                    st["drops"]["generator_excluded" if g else "generator_unknown"] += 1
                    st["gen_rows"][f"excluded: {g}"] += 1
                    continue
                st["gen_rows"][g] += 1
            try:
                items = maker(rec, ctx, rng, src)
            except Exception:
                items = [(DROP, "parse_error")]
            for item in items:
                if item[0] == DROP:
                    st["drops"][item[1]] += 1
                    continue
                sub, text, mh_text = item
                try:                                       # lone surrogates crash the Rust tokenizer ("TextInputSequence
                    text.encode("utf-8"); (mh_text or "").encode("utf-8")   # must be str") and would fail the unit
                except UnicodeEncodeError:
                    st["drops"]["invalid_unicode"] += 1
                    continue
                if sub not in budgets or produced[sub] >= budgets[sub]:
                    st["over_quota"] += 1
                    continue
                text = text.strip() if not src["task"] else text
                if len(text) < src.get("min_chars", 1):
                    st["drops"]["too_short"] += 1
                    continue
                if not src["task"] and has_special(text):
                    st["drops"]["special_token_literal"] += 1
                    continue
                if not src["task"] and mojibake(text):
                    st["drops"]["mojibake"] += 1
                    continue
                h = text_hash(text, u["epoch"])
                if h in seen:
                    st["local_dups"] += 1
                    continue
                seen.add(h)
                bad, hits, evals = ctx.ix.is_contaminated(text)
                if hits and not bad:
                    st["single_hit_kept"] += 1
                if bad:
                    st["drops"]["decontam"] += 1
                    for e in evals:
                        st["decontam"][e] += 1
                    if len(st["decon_examples"]) < 3:
                        st["decon_examples"].append({"sub": sub, "hits": hits, "evals": evals, "snippet": text[:240]})
                    continue
                if src["task"]:
                    n = len(ctx.tok.encode(text, add_special_tokens=False).ids)
                    if n > TASK_MAX_TOKENS:
                        st["drops"]["task_over_3072_tokens"] += 1
                        continue
                    pieces = [(text, n)]
                    st["task_over_2048"] += n > 2048
                elif task["max_doc_tokens"]:
                    pieces = chunk_doc(text, ctx.tok, task["max_doc_tokens"])
                    st["chunked_docs"] += len(pieces) > 1
                else:
                    pieces = [(text, len(ctx.tok.encode(text, add_special_tokens=False).ids))]
                mh = [b ^ ep_salt for b in minhash_bands(mh_text)] if (mh_text and task["minhash"]) else []
                for k, (piece, n) in enumerate(pieces):
                    produced[sub] += n
                    st["docs"][sub] += 1
                    st["chars"][sub] += len(piece)
                    cols["text"].append(piece); cols["ntok"].append(n); cols["nchar"].append(len(piece))
                    cols["h"].append(h if k == 0 else text_hash(piece, u["epoch"])); cols["sub"].append(sub)
                    cols["mh"].append(mh)
            if task.get("time_limit") and time.time() - t0 > task["time_limit"]:
                exhausted = False
                break
            if all(produced[s] >= budgets[s] for s in budgets):
                exhausted = False
                break
    finally:
        if hasattr(it, "close"):
            it.close()
    os.makedirs(os.path.dirname(task["out"]), exist_ok=True)
    tbl = pa.table({"text": pa.array(cols["text"], pa.large_string()), "ntok": pa.array(cols["ntok"], pa.int32()),
                    "nchar": pa.array(cols["nchar"], pa.int32()), "h": pa.array(cols["h"], pa.uint64()),
                    "sub": pa.array(cols["sub"], pa.string()), "mh": pa.array(cols["mh"], pa.list_(pa.uint64()))})
    pq.write_table(tbl, task["out"], compression="zstd")
    return dict(uid=task["uid"], source=task["source"], phase=task["phase"], out=task["out"], budgets=budgets,
                produced=produced, exhausted=exhausted, pos=list(pos), seconds=time.time() - t0,
                cpu_seconds=time.process_time() - c0,
                records=st["records"], bytes_read=st["bytes_read"], drops=dict(st["drops"]), decontam=dict(st["decontam"]),
                decon_examples=st["decon_examples"], docs=dict(st["docs"]), chars=dict(st["chars"]),
                over_quota=st["over_quota"], local_dups=st["local_dups"], single_hit_kept=st["single_hit_kept"],
                prefiltered=st["prefiltered"], gen_rows=dict(st["gen_rows"]),
                chunked_docs=st["chunked_docs"], task_over_2048=st["task_over_2048"])


def run_unit_safe(task):
    try:
        return run_unit(task)
    except Exception as e:
        return dict(uid=task["uid"], source=task["source"], phase=task["phase"], error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-2000:], budgets=task["budgets"])


# ----------------------------------------------------------------------------------------------- planning
def check_access(spec, tries=5):
    """True if the first data file of a spec can actually be read (gated repos list fine but 403 on read).
    Transient network errors (timeouts, 5xx) are retried; an auth / gating error is final."""
    for i in range(tries):
        ok, why = _check_access_once(spec)
        if ok or i == tries - 1 or any(k in why for k in ("403", "401", "Gated", "gated", "no files match", "local file")):
            return ok, why
        print(f"  access check {spec.get('hf') or spec.get('hf_files')}: {why} -> retry {i + 1}", flush=True)
        time.sleep(10 * (i + 1))
    return ok, why


def _check_access_once(spec):
    fs = hffs()
    try:
        if spec["reader"] == "parquet":
            files = fs.glob("datasets/" + spec["hf"][0])
            if not files:
                return False, "no files match " + spec["hf"][0]
            with fs.open(files[0], block_size=0) as f:
                f.read(4)
        else:
            repo, fn = spec["hf_files"][0]
            if repo == "local":
                return os.path.exists(fn), "local file"
            with fs.open(f"datasets/{repo}/{fn}", block_size=0) as f:
                f.read(4)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


SKIP_HEAD, SKIP_TAIL = {}, {}


def list_units(name, spec, seed):
    fs = hffs()
    units = []
    if spec["reader"] == "parquet":
        for pat in spec["hf"]:
            for path, info in sorted(fs.glob("datasets/" + pat, detail=True).items()):
                # files are split into K units by row group (unit k reads row groups i % K == k); sources whose
                # files are a single row group must not be split (units k > 0 would be empty)
                K = max(1, round((info.get("size") or 0) / UNIT_BYTES)) if spec.get("split_files", True) else 1
                for k in range(K):
                    units.append({"path": path.split("datasets/", 1)[1], "k": k, "K": K})
    else:
        K = spec.get("json_units", 4)
        units = [{"files": spec["hf_files"], "k": k, "K": K} for k in range(K)]
    random.Random(f"{seed}-{name}").shuffle(units)
    # Udgam-2: skip units an earlier build already consumed (head of the same shuffled list, same seed) and units the
    # 4K long-context stage took from the tail, so the new build is fresh text only.
    head, tail = SKIP_HEAD.get(name, 0), SKIP_TAIL.get(name, 0)
    if head or tail:
        units = units[head:len(units) - tail]
        assert units, f"{name}: no units left after skipping {head} head / {tail} tail units"
    for i, u in enumerate(units):
        u.update(idx=i, pos=(0, 0), epoch=1, resumes=0)
    return units


def allocate(active, scale, extra_epochs=(0.5, 2.5)):
    """quota[sub][phase] (tokens). Shares are applied at FULL scale, availability caps (unique tokens x max_epochs)
    are applied there too, excess goes to uncapped group-mates, then the bucket; finally everything x scale."""
    sub_of = {}
    for name, s in active.items():
        for sub in s["subs"]:
            sub_of[sub] = name
    q = {sub: {"A": 0.0, "B": 0.0} for sub in sub_of}
    for ph, buckets in BUCKETS.items():
        T = FULL_TOKENS * PHASE_FRAC[ph]
        for b, share in buckets.items():
            w = {k: v for k, v in SUBWEIGHTS[b].items() if k in sub_of}
            tot = sum(w.values())
            if tot == 0:
                print(f"WARNING: bucket {b} has no active sources; its {share:.0%} share of phase {ph} is unassigned")
                continue
            for k, v in w.items():
                q[k][ph] += T * share * v / tot
    # availability caps (task sources): water-fill within group, then within bucket
    capped = set()
    for _ in range(20):
        changed = False
        for sub, nm in sub_of.items():
            s = active[nm]
            cap = s.get("avail_tokens")
            if not cap or sub in capped or sub == "extra_task":      # extra task is sized by epochs below
                continue
            cap *= s["max_epochs"]
            tot = q[sub]["A"] + q[sub]["B"]
            if tot <= cap * 1.0001:
                continue
            f = cap / tot
            capped.add(sub)
            changed = True
            for ph in "AB":
                excess = q[sub][ph] * (1 - f)
                q[sub][ph] *= f
                bucket, group = s["bucket"], s.get("group")
                ok = lambda k, n: k not in capped and k != sub and k != "extra_task" and active[n]["bucket"] == bucket
                mates = [k for k, n in sub_of.items() if ok(k, n) and active[n].get("group") == group]
                if not mates:
                    mates = [k for k, n in sub_of.items() if ok(k, n)]
                wsum = sum(SUBWEIGHTS[bucket][k] for k in mates)
                if not mates and excess > 0:
                    print(f"WARNING: bucket {bucket} phase {ph}: every source is at its availability cap; "
                          f"{excess * scale / 1e6:.2f}M tokens of its share are unassigned (inaccessible / excluded sources?)")
                for k in mates:
                    q[k][ph] += excess * SUBWEIGHTS[bucket][k] / wsum
        if not changed:
            break
    # our own gold task data: fixed epochs per phase (default 0.5x in A, 2.5x in B = the anneal), carved out of
    # the task bucket (other task sources shrink proportionally so phase totals are unchanged)
    if "extra_task" in sub_of:
        av = active["extra_task"]["avail_tokens"]
        q["extra_task"] = {"A": 0.0, "B": 0.0}
        for ph, ep in zip("AB", extra_epochs):
            want = av * ep
            others = [k for k, n in sub_of.items() if active[n]["bucket"] == "task" and k != "extra_task"]
            tot = sum(q[k][ph] for k in others)
            want = min(want, tot)
            for k in others:
                q[k][ph] *= (1 - want / tot)
            q["extra_task"][ph] = want
    return {k: {ph: v * scale for ph, v in d.items()} for k, d in q.items()}, sub_of


def estimate_jsonl_tokens(path):
    """Rough unique-token size of a task jsonl (string chars / 3.6 + ~12 template tokens per message); the
    exact count is in manifest.json after the build."""
    exact = 0
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if recs and all("ntok" in r for r in recs):          # pre-rendered records carry their exact token count
        exact = sum(int(r["ntok"]) for r in recs)
        return float(exact)
    chars = msgs = 0
    def walk(x):
        nonlocal chars
        if isinstance(x, str):
            chars += len(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            walk(r)
            msgs += len(r.get("messages") or [1])
    return chars / 3.6 + 12 * msgs


def resolve_sources(args):
    active, notes = {}, {}
    names = list(SOURCES)
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    for name in names:
        spec = dict(SOURCES[name])
        ok, why = check_access(spec)
        if ok:
            spec["fallback_used"] = None
        elif args.allow_fallbacks and spec.get("fallback"):
            fb = spec["fallback"]
            spec2 = dict(spec)
            spec2.update({k: v for k, v in fb.items() if k != "note"})
            ok2, why2 = check_access(spec2)
            if not ok2:
                notes[name] = f"primary inaccessible ({why}); fallback inaccessible ({why2}) -> SKIPPED"
                continue
            spec2["fallback_used"] = {"reason": why, "note": fb.get("note"), "primary_hf": spec.get("hf") or spec.get("hf_files")}
            spec = spec2
        elif args.allow_fallbacks:
            notes[name] = f"primary inaccessible ({why}); no fallback -> SKIPPED, share redistributed"
            continue
        else:
            raise SystemExit(f"ERROR: source {name} is not accessible: {why}\n"
                             f"  (request access on Hugging Face, or pass --allow-fallbacks for a test build)")
        bad = policy_violations(spec)
        if bad or "prov" not in spec:
            raise SystemExit(f"ERROR: source {name} violates the data licence policy (generators {bad}) or has no "
                             f"provenance record; see GENERATOR_RULES / data/LICENSES_AND_PROVENANCE.md")
        if spec["task"]:
            spec["max_epochs"] = args.task_max_epochs
        active[name] = spec
        if spec.get("fallback_used"):
            notes[name] = "FALLBACK: " + (spec["fallback_used"]["note"] or "")
    if args.extra_task_jsonl:
        active["extra_task"] = dict(bucket="task", group="tool", reader="json", hf_files=[("local", os.path.abspath(args.extra_task_jsonl))],
                                    make="extra", task=True, subs=["extra_task"], json_units=8,
                                    max_epochs=int(math.ceil(args.extra_task_epochs_a + args.extra_task_epochs_b)),
                                    avail_tokens=estimate_jsonl_tokens(args.extra_task_jsonl), license="UdgamLabs (own data)",
                                    filters=["rendered with Gemma 4 template unless pre-rendered `text`"], fallback_used=None,
                                    prov=_P("UdgamLabs (own data, released with the model)",
                                            [("Claude (Anthropic API)", "Anthropic API output", "gold examples (data/gold/accepted)")],
                                            "data/gold/", reason="own data (Claude-written gold, rendered by "
                                            "udgam/gold/render_for_pretrain.py exactly as the gold validator renders it); "
                                            "generator allowed by user decision",
                                            caveats="must not contain outputs of excluded models"))
    return active, notes


# ----------------------------------------------------------------------------------------------- scheduler
def schedule(active, quota, sub_of, args, work):
    def _lu(n, s):
        for i in range(5):                                 # transient HF listing errors: retry
            try:
                return list_units(n, s, args.seed)
            except Exception as e:
                if i == 4:
                    raise
                print(f"  listing {n} failed ({type(e).__name__}: {str(e)[:120]}); retry {i + 1}", flush=True)
                hffs().invalidate_cache(); time.sleep(10 * (i + 1))
    units = {n: _lu(n, s) for n, s in active.items()}
    queue = {n: collections.deque(us) for n, us in units.items()}
    epoch = {n: 1 for n in active}
    inflight_src = collections.Counter()
    done = {s: {"A": 0, "B": 0} for s in quota}
    reserved = {s: {"A": 0, "B": 0} for s in quota}
    results, errors = [], []
    # files of these sources are one ~1GB row group each: every unit costs a full-file download, so spread less
    cap = {s: {ph: max(quota[s][ph] / active[sub_of[s]].get("min_units", MIN_UNITS), 20000) for ph in "AB"} for s in quota}
    uid_n = 0

    def needs(name, ph):
        return {s: quota[s][ph] - done[s][ph] - reserved[s][ph] for s in active[name]["subs"]
                if quota[s][ph] - done[s][ph] - reserved[s][ph] > 1}

    def progress(name, ph):
        q = sum(quota[s][ph] for s in active[name]["subs"])
        return sum(done[s][ph] + reserved[s][ph] for s in active[name]["subs"]) / max(q, 1)

    t_start = time.time()
    last_print = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.decontam_index, 1 if args.decontam_strict else decontam.SHORT_DOC_MIN_HITS, TASK_PLAIN)) as ex:
        futs = {}
        while True:
            while len(futs) < args.workers:
                cands = []
                for n in active:
                    if not queue[n] and inflight_src[n] == 0 and epoch[n] < active[n]["max_epochs"] \
                            and any(needs(n, ph) for ph in "AB"):
                        epoch[n] += 1                               # every unit used -> next epoch (repeat data)
                        queue[n] = collections.deque(dict(u, pos=(0, 0), epoch=epoch[n], resumes=0) for u in units[n])
                    if not queue[n]:
                        continue
                    for ph in "AB":
                        if needs(n, ph):
                            cands.append((progress(n, ph), n, ph))
                if not cands:
                    break
                _, n, ph = min(cands)
                u = queue[n].popleft()
                budgets = {s: int(min(v, cap[s][ph])) + 1 for s, v in needs(n, ph).items()}
                for s, b in budgets.items():
                    reserved[s][ph] += b
                uid = f"{n}_{u['idx']:05d}_r{u['resumes']}_e{u['epoch']}"
                uid_n += 1
                task = dict(uid=uid, source=n, phase=ph, src=active[n], unit=u, budgets=budgets, minhash=args.minhash_task,
                            max_doc_tokens=args.max_doc_tokens,
                            out=f"{work}/parts/{ph}/{n}/{uid}.parquet")
                futs[ex.submit(run_unit_safe, task)] = (task, u)
                inflight_src[n] += 1
            if not futs:
                break
            fin, _ = wait(list(futs), return_when=FIRST_COMPLETED)
            for f in fin:
                task, u = futs.pop(f)
                r = f.result()
                n, ph = task["source"], task["phase"]
                inflight_src[n] -= 1
                for s, b in task["budgets"].items():
                    reserved[s][ph] -= b
                if "error" in r:
                    retries = u.get("retries", 0)
                    if retries < 3:                        # transient (network) failure: nothing was written -> retry
                        print(f"  unit {r['uid']} failed ({r['error'][:160]}); retry {retries + 1}/3", flush=True)
                        queue[n].append(dict(u, retries=retries + 1))
                        continue
                    errors.append(r)
                    print(f"  ERROR in {r['uid']}: {r['error']}", flush=True)
                    continue
                for s, v in r["produced"].items():
                    done[s][ph] += v
                results.append(r)
                if not r["exhausted"]:                       # budget hit mid-unit: resume it later from pos
                    queue[n].append(dict(u, pos=tuple(r["pos"]), resumes=u["resumes"] + 1))
            if time.time() - last_print > 15:
                last_print = time.time()
                tot_done = sum(done[s][ph] for s in done for ph in "AB")
                tot_q = sum(quota[s][ph] for s in quota for ph in "AB")
                print(f"  [{time.time()-t_start:6.0f}s] {tot_done/1e6:9.2f}M / {tot_q/1e6:.2f}M tokens, "
                      f"{len(results)} units done, {len(futs)} in flight", flush=True)
    return results, errors, done, time.time() - t_start


# ----------------------------------------------------------------------------------------------- assembly
def _finalize_bucket(args_):
    tmp, final, srcmap_path, seed = args_
    t = pq.read_table(tmp)
    perm = np.random.default_rng(seed).permutation(t.num_rows)
    t = t.take(pa.array(perm))
    pq.write_table(pa.table({"text": t.column("text").cast(pa.string())}), final, row_group_size=ROWS_PER_GROUP,
                   compression="zstd", compression_level=3)
    src = t.column("src").to_numpy().astype(np.uint8)
    np.save(srcmap_path, src)
    ntok = t.column("ntok").to_numpy()
    per = collections.Counter()
    for sid, nt in zip(src.tolist(), ntok.tolist()):
        per[sid] += nt
    os.remove(tmp)
    return dict(file=os.path.basename(final), rows=t.num_rows, tokens=int(ntok.sum()), chars=int(t.column("nchar").to_numpy().sum()),
                bytes=os.path.getsize(final), row_groups=pq.ParquetFile(final).metadata.num_row_groups,
                per_source_tokens={int(k): int(v) for k, v in per.items()})


def assemble(results, active, sub_of, args, work, scale):
    rng = np.random.default_rng(args.seed)
    parts = sorted([r for r in results if os.path.exists(r["out"])], key=lambda r: (r["phase"], r["uid"]))
    sub_names = sorted({s for s in sub_of})
    sub_id = {s: i for i, s in enumerate(sub_names)}
    meta = []
    for pi, r in enumerate(parts):
        t = pq.read_table(r["out"], columns=["ntok", "nchar", "h", "sub", "mh"])
        n = t.num_rows
        meta.append(dict(part=np.full(n, pi, np.int32), row=np.arange(n, dtype=np.int32),
                         phase=np.full(n, 0 if r["phase"] == "A" else 1, np.int8),
                         sub=np.array([sub_id[s] for s in t.column("sub").to_pylist()], np.int16),
                         ntok=t.column("ntok").to_numpy(), nchar=t.column("nchar").to_numpy(), h=t.column("h").to_numpy(),
                         mh=t.column("mh").to_pylist()))
    cat = lambda k: np.concatenate([m[k] for m in meta]) if meta else np.empty(0)
    part, row, phase, sub, ntok, nchar, h = (cat(k) for k in ("part", "row", "phase", "sub", "ntok", "nchar", "h"))
    mh = [x for m in meta for x in m["mh"]]
    N = len(h)
    keep = np.ones(N, bool)
    # global exact dedup (first occurrence wins; epoch is folded into the hash so planned repeats survive)
    _, first = np.unique(h, return_index=True)
    dup = np.ones(N, bool); dup[first] = False
    keep &= ~dup
    exact_dups = collections.Counter(sub_names[s] for s in sub[dup])
    # MinHash near-dup for task docs
    near = collections.Counter()
    if args.minhash_task:
        idx = np.array([i for i in range(N) if mh[i] and keep[i]], dtype=np.int64)
        if idx.size:
            M = np.array([mh[i] for i in idx], dtype=np.uint64)
            nd = np.zeros(idx.size, bool)
            for b in range(M.shape[1]):
                col = M[:, b] ^ np.uint64(b * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)
                _, f = np.unique(col, return_index=True)
                mask = np.ones(idx.size, bool); mask[f] = False
                nd |= mask
            keep[idx[nd]] = False
            near = collections.Counter(sub_names[s] for s in sub[idx[nd]])
    # validation hold-out: mixed random sample across both phases
    total_tok = float(ntok[keep].sum())
    val_tokens = max(20e6 * scale, min(0.01 * total_tok, 2e6))
    p_val = min(1.0, val_tokens / max(total_tok, 1))
    is_val = keep & (rng.random(N) < p_val)
    # shard assignment
    shard_chars = args.shard_chars
    bucket = np.full(N, -1, np.int64)
    n_shards = {}
    offset = 0
    for ph in (0, 1):
        m = keep & ~is_val & (phase == ph)
        k = max(1, int(math.ceil(nchar[m].sum() / shard_chars))) if m.any() else 0
        n_shards["AB"[ph]] = k
        if k:
            bucket[m] = offset + rng.integers(0, k, size=int(m.sum()))
        offset += k
    val_bucket = offset
    bucket[is_val] = val_bucket
    nb = offset + 1
    tmpdir = f"{work}/buckets"
    os.makedirs(tmpdir, exist_ok=True)
    schema = pa.schema([("text", pa.large_string()), ("src", pa.uint8()), ("ntok", pa.int32()), ("nchar", pa.int32())])
    writers = {}
    # sidecar ids are sub-source level (code_shell vs code_general distinguishable in audits)
    all_ids = sorted(set(sub_names))
    sid_of_sub = {s: all_ids.index(s) for s in sub_names}
    for pi, r in enumerate(parts):
        sel = np.nonzero((part == pi) & (bucket >= 0))[0]
        if not sel.size:
            continue
        t = pq.read_table(r["out"], columns=["text", "ntok", "nchar", "sub"])
        rows = row[sel]
        bks = bucket[sel]
        sids = np.array([sid_of_sub[s] for s in t.column("sub").to_pylist()], np.uint8)
        for b in np.unique(bks):
            rr = rows[bks == b]
            tt = t.take(pa.array(rr))
            tb = pa.table({"text": tt.column("text"), "src": pa.array(sids[rr], pa.uint8()), "ntok": tt.column("ntok"),
                           "nchar": tt.column("nchar")}, schema=schema)
            if b not in writers:
                writers[b] = pq.ParquetWriter(f"{tmpdir}/b{b:05d}.parquet", schema, compression="lz4")
            writers[b].write_table(tb)
    for w in writers.values():
        w.close()
    os.makedirs(f"{args.out}/_meta", exist_ok=True)
    jobs = [(f"{tmpdir}/b{b:05d}.parquet", f"{args.out}/shard_{b:05d}.parquet", f"{args.out}/_meta/shard_{b:05d}.sources.npy",
             args.seed * 1000 + b) for b in range(nb) if os.path.exists(f"{tmpdir}/b{b:05d}.parquet")]
    from multiprocessing import Pool
    with Pool(min(args.assemble_workers or min(args.workers, 8), max(1, len(jobs)))) as pool:   # ~2-3 GB RAM each
        shard_stats = pool.map(_finalize_bucket, jobs)
    for s in shard_stats:
        b = int(s["file"][6:11])
        s["phase"] = "val" if b == val_bucket else ("A" if b < n_shards["A"] else "B")
        s["per_source_tokens"] = {all_ids[k]: v for k, v in s["per_source_tokens"].items()}
    dedup = dict(exact_dups=dict(exact_dups), near_dups_task=dict(near))
    kept_sub_tok = collections.defaultdict(lambda: {"A": 0, "B": 0, "val": 0})
    for s_i, ph_i, v_i, nt, k in zip(sub.tolist(), phase.tolist(), is_val.tolist(), ntok.tolist(), keep.tolist()):
        if k:
            kept_sub_tok[sub_names[s_i]]["val" if v_i else "AB"[ph_i]] += nt
    return shard_stats, dedup, dict(kept_sub_tok), all_ids, n_shards


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = full ~11B-token build")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dry-run", action="store_true", help="check access, list units, print the plan; no streaming")
    ap.add_argument("--allow-fallbacks", action="store_true", help="use stand-in datasets for inaccessible gated sources (TEST ONLY)")
    ap.add_argument("--extra-task-jsonl", help="our own gold task data: {messages, tools?, enable_thinking?} or {text} per line")
    ap.add_argument("--extra-task-epochs-a", type=float, default=0.5, help="passes over --extra-task-jsonl in phase A")
    ap.add_argument("--extra-task-epochs-b", type=float, default=2.5, help="passes over --extra-task-jsonl in phase B (anneal)")
    ap.add_argument("--task-max-epochs", type=int, default=3, help="max repeats of small task datasets")
    ap.add_argument("--shard-chars", type=float, default=550e6, help="~chars per shard (550M chars ~ 200MB zstd)")
    ap.add_argument("--decontam-index", default=decontam.DEFAULT_INDEX)
    ap.add_argument("--no-minhash-task", dest="minhash_task", action="store_false")
    ap.add_argument("--max-doc-tokens", type=int, default=2048,
                    help="split non-task docs longer than this at paragraph breaks (0 = off); = pretraining seq len")
    ap.add_argument("--decontam-strict", action="store_true", help="drop short docs on a single shared 13-gram")
    ap.add_argument("--bench-json", help="bench_stream.py report, used for the full-build projection")
    ap.add_argument("--assume-MBps", type=float, default=150.0, help="aggregate HF download MB/s assumed for the projection")
    ap.add_argument("--only", help="comma list of source names (debugging)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mix", choices=["v1", "v2", "v2b", "v3", "v3s"], default="v1", help="v2 = Udgam-2 general/reasoning mix (BUCKETS_V2); v2b = v2 + 25%% web anneal; v3/v3s = Udgam-3 (BUCKETS_V3/V3S)")
    ap.add_argument("--skip-units-from", help="comma-separated manifest.json files of earlier builds made IN ORDER with the SAME seed "
                                               "(each skipping the previous ones): skip the units they used per source (summed)")
    ap.add_argument("--skip-tail", default="", help="name=N,... units to skip at the END of each source's list (e.g. the 4K stage's)")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--assemble-workers", type=int, default=0, help="shard-finalize processes (default min(workers, 8); each holds a "
                    "whole ~550M-char shard, 2-3 GB RAM)")
    args = ap.parse_args()
    if args.mix in ("v2", "v2b"):
        BUCKETS.clear(); BUCKETS.update(BUCKETS_V2 if args.mix == "v2" else BUCKETS_V2B)
        SUBWEIGHTS["task"] = dict(TASK_SUBWEIGHTS_V2)
    if args.mix in ("v3", "v3s"):
        global TASK_PLAIN
        BUCKETS.clear(); BUCKETS.update(BUCKETS_V3 if args.mix == "v3" else BUCKETS_V3S)
        SUBWEIGHTS.update({k: dict(v) for k, v in SUBWEIGHTS_V3.items()})
        for n in V3_DROP:
            SOURCES.pop(n, None)
        SOURCES.update(SOURCES_V3)
        TASK_PLAIN = True
    for mpath in filter(None, (args.skip_units_from or "").split(",")):
        prev = json.load(open(mpath))
        assert prev.get("seed") == args.seed, f"--skip-units-from {mpath}: built with a different --seed"
        # only sources read once (epochs_used == 1); small task sets were cycled through completely and are reused
        for n, v in prev["sources"].items():
            if v.get("units") and v.get("epochs_used", 1) == 1 and SOURCES.get(n, {}).get("bucket") != "task":
                SKIP_HEAD[n] = SKIP_HEAD.get(n, 0) + int(v["units"])
    for kv in filter(None, args.skip_tail.split(",")):
        n, v = kv.split("="); SKIP_TAIL[n] = int(v)
    if SKIP_HEAD or SKIP_TAIL:
        print(f"skipping used units: head {SKIP_HEAD} | tail {SKIP_TAIL}", flush=True)

    t0 = time.time()
    if not os.path.exists(args.decontam_index):
        print("building decontamination index ...")
        decontam.build(args.decontam_index)
    print("checking dataset access ...", flush=True)
    active, notes = resolve_sources(args)
    for k, v in notes.items():
        print(f"  {k}: {v}")
    extra_ep = (args.extra_task_epochs_a, args.extra_task_epochs_b)
    quota, sub_of = allocate(active, args.scale, extra_ep)
    task_subs = [s for s in quota if active[sub_of[s]]["task"]]
    tool_subs = [s for s in task_subs if active[sub_of[s]].get("group") == "tool"]
    tq = sum(quota[s][p] for s in task_subs for p in "AB")
    print(f"\nplan @ scale {args.scale}: {sum(q['A'] for q in quota.values())/1e6:,.1f}M tokens phase A, "
          f"{sum(q['B'] for q in quota.values())/1e6:,.1f}M phase B; tool/JSON share of task tokens = "
          f"{sum(quota[s][p] for s in tool_subs for p in 'AB')/max(tq,1):.1%}")
    for s in sorted(quota, key=lambda s: -(quota[s]["A"] + quota[s]["B"])):
        print(f"  {s:20s} A {quota[s]['A']/1e6:10.2f}M  B {quota[s]['B']/1e6:10.2f}M   [{sub_of[s]}]")
    if args.dry_run:
        for n, s in active.items():
            us = list_units(n, s, args.seed)
            print(f"  units {n:16s} {len(us):6d}  e.g. {us[0].get('path') or us[0].get('files')}")
        return
    os.makedirs(args.out, exist_ok=True)
    if any(f.endswith(".parquet") for f in os.listdir(args.out)):
        raise SystemExit(f"{args.out} already has shards; use a fresh --out")
    work = f"{args.out}/_work"
    print(f"\nstreaming with {args.workers} workers ...", flush=True)
    results, errors, done, t_stream = schedule(active, quota, sub_of, args, work)
    print(f"streamed in {t_stream:.0f}s; assembling ...", flush=True)
    t1 = time.time()
    shards, dedup, kept, all_ids, n_shards = assemble(results, active, sub_of, args, work, args.scale)
    t_asm = time.time() - t1
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------------ manifest
    per_src = {}
    for n, s in active.items():
        rs = [r for r in results if r["source"] == n]
        drops = collections.Counter(); dec = collections.Counter(); ex = []
        for r in rs:
            drops.update(r["drops"]); dec.update(r["decontam"]); ex += r["decon_examples"]
        docs = collections.Counter(); chars = collections.Counter()
        for r in rs:
            docs.update(r["docs"]); chars.update(r["chars"])
        recs = sum(r["records"] for r in rs)
        n_out = sum(docs.values())
        secs = sum(r["seconds"] for r in rs)
        tok = sum(sum(r["produced"].values()) for r in rs)
        bts = sum(r["bytes_read"] for r in rs)
        gen_rows = collections.Counter()
        for r in rs:
            gen_rows.update(r.get("gen_rows", {}))
        prov = dict(s["prov"])
        prov["generators"] = [dict(g, license_verdict=generator_verdict(g["model"])[1]) for g in prov["generators"]]
        per_src[n] = dict(
            license=s["license"], hf=s.get("hf") or [f"{a}/{b}" for a, b in s.get("hf_files", [])],
            provenance=prov, generator_rows_read=dict(gen_rows),
            fallback_used=s.get("fallback_used"), filters=s.get("filters", []), subs=s["subs"],
            phases=[ph for ph in "AB" if any(quota[x][ph] > 0 for x in s["subs"])],
            quota_tokens={x: {ph: round(quota[x][ph]) for ph in "AB"} for x in s["subs"]},
            kept_tokens={x: kept.get(x, {}) for x in s["subs"]},
            docs={x: docs.get(x, 0) for x in s["subs"]}, chars={x: chars.get(x, 0) for x in s["subs"]},
            records_read=recs, drops=dict(drops), rows_prefiltered=sum(r.get("prefiltered", 0) for r in rs), over_quota_skipped=sum(r["over_quota"] for r in rs),
            unit_local_dups=sum(r["local_dups"] for r in rs),
            decontam_single_hit_kept=sum(r.get("single_hit_kept", 0) for r in rs),
            long_docs_chunked=sum(r.get("chunked_docs", 0) for r in rs),
            task_docs_2049_to_3072_tokens=sum(r.get("task_over_2048", 0) for r in rs),
            decontam_drops=dict(dec), decontam_drop_rate=round(drops.get("decontam", 0) / max(1, n_out + drops.get("decontam", 0)), 5),
            decontam_examples=ex[:5], exact_dups_global={x: dedup["exact_dups"].get(x, 0) for x in s["subs"]},
            near_dups_task={x: dedup["near_dups_task"].get(x, 0) for x in s["subs"]},
            chars_per_token=round(sum(chars.values()) / max(tok, 1), 3), est_tokens_chars_div_4_4=round(sum(chars.values()) / 4.4),
            units=len(rs), epochs_used=max([int(r["uid"].rsplit("_e", 1)[1]) for r in rs] or [0]),
            worker_seconds=round(secs, 1), bytes_read=bts,
            tokens_per_worker_second=round(tok / max(secs, 1e-9)),
            tokens_per_cpu_second=round(tok / max(sum(r.get("cpu_seconds", 0) for r in rs), 1e-9)), docs_per_worker_second=round(n_out / max(secs, 1e-9), 1),
            records_per_worker_second=round(recs / max(secs, 1e-9), 1), bytes_read_per_token=round(bts / max(tok, 1), 2))
    # ---- projection of the full build on a 32-vCPU cloud box. Streaming is download- or CPU-bound per source:
    # bytes/token and tokens/s per worker come from bench_stream.py (steady state, --bench-json) when given,
    # else from this run (tiny runs overstate both: per-unit startup dominates). For multi-sub sources (code)
    # the scan is driven by the scarcest sub (shell), so download = max over subs of quota x bytes/sub-token.
    bench = json.load(open(args.bench_json))["sources"] if args.bench_json else {}
    full_q, _ = allocate(active, 1.0, extra_ep)
    proj_cpu_ws, proj_bytes, proj_rows = 0.0, 0.0, {}
    for n, s in active.items():
        d, b = per_src[n], bench.get(n, {})
        prod = b.get("produced") or {x: sum(r["produced"].get(x, 0) for r in results if r["source"] == n) for x in s["subs"]}
        bts = b.get("bytes_read", d["bytes_read"])
        rate = b.get("tokens_per_cpu_s", d["tokens_per_cpu_second"])
        q_full = sum(full_q[x][ph] for x in s["subs"] for ph in "AB")
        dl = max((sum(full_q[x].values()) * bts / prod[x]) for x in s["subs"] if prod.get(x)) if any(prod.values()) else 0
        proj_bytes += dl
        proj_cpu_ws += q_full / max(rate, 1)
        proj_rows[n] = dict(full_tokens_M=round(q_full / 1e6), download_GB=round(dl / 1e9, 2), tok_per_cpu_s=rate)
    out_bytes = sum(x["bytes"] for x in shards)
    out_tok = sum(x["tokens"] for x in shards)
    cpu_h = proj_cpu_ws / 32 / 3600
    dl_h = proj_bytes / (args.assume_MBps * 1e6) / 3600
    stream_h = max(cpu_h, dl_h) * 1.25                           # +25% scheduling tail / stragglers
    asm_h = (t_asm / max(out_tok, 1) * 11e9) / 3600 / 4 + 0.15   # assembly: measured rate, 4x cores, + IO slack
    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), scale=args.scale, seed=args.seed, args=vars(args),
        tokenizer=TOKENIZER, chat_template=CHAT_TEMPLATE, decontam_index=args.decontam_index,
        decontam_rule=dict(n=decontam.N, long_doc_words=decontam.LONG_DOC_WORDS, long_doc_min_hits=decontam.LONG_DOC_MIN_HITS,
                           short_doc_min_hits=1 if args.decontam_strict else decontam.SHORT_DOC_MIN_HITS,
                           template_gram_items=decontam.COMMON_ITEMS),
        mix=dict(buckets=BUCKETS, subweights=SUBWEIGHTS), notes=notes, source_ids=all_ids,
        license_policy=dict(decided="2026-10-06", release="Apache-2.0, UdgamLabs name",
                            audit="data/LICENSES_AND_PROVENANCE.md",
                            generator_rules=[dict(pattern=rx, license=lic, allowed=ok) for rx, lic, ok in GENERATOR_RULES],
                            unmatched_generator="not allowed", excluded_sources=EXCLUDED_SOURCES),
        phase_shards=dict(A=[0, n_shards["A"] - 1], B=[n_shards["A"], n_shards["A"] + n_shards["B"] - 1],
                          val=n_shards["A"] + n_shards["B"]),
        totals=dict(tokens=out_tok, tokens_incl_bos=out_tok + sum(x["rows"] for x in shards), docs=sum(x["rows"] for x in shards),
                    bytes=out_bytes, phase_tokens={ph: sum(x["tokens"] for x in shards if x["phase"] == ph) for ph in ("A", "B", "val")},
                    stream_seconds=round(t_stream), assemble_seconds=round(t_asm), wall_seconds=round(time.time() - t0),
                    workers=args.workers, unit_errors=len(errors)),
        projection_full_build_32vcpu=dict(stream_hours=round(stream_h, 2), cpu_bound_hours=round(cpu_h, 2),
                                          download_bound_hours=round(dl_h, 2), assumed_download_MBps=args.assume_MBps,
                                          assemble_hours=round(asm_h, 2), download_GB=round(proj_bytes / 1e9, 1),
                                          per_source=proj_rows, rates_from="bench_stream.py" if bench else "this run",
                                          output_GB=round(out_bytes / max(out_tok, 1) * 11e9 / 1e9, 1),
                                          cost_usd_at_0_5_to_1_per_h=[round((stream_h + asm_h) * 0.5, 2), round((stream_h + asm_h) * 1.0, 2)],
                                          caveat="per-worker CPU rates measured on the dev Mac (11 workers sharing a ~8 MB/s link, so "
                                                 "conservative); download time assumes the stated aggregate HF throughput"),
        sources=per_src, shards=shards, errors=errors[:20])
    json.dump(manifest, open(f"{args.out}/manifest.json", "w"), indent=1, default=str)
    print(f"\nwrote {len(shards)} shards ({n_shards['A']} A + {n_shards['B']} B + 1 val), {out_tok/1e6:.2f}M tokens, "
          f"{out_bytes/1e6:.1f}MB -> {args.out}  ({time.time()-t0:.0f}s, {len(errors)} unit errors)")


if __name__ == "__main__":
    main()
