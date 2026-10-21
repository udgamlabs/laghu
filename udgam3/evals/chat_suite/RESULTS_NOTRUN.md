## Models that were not run, and why

| Model | Why not |
|---|---|
| Nikity/lille-130m-instruct | custom `LilleForCausalLM` with no modeling code in the repo (same as its base model in Phase 0) |
| PleIAs/Monad, SupraLabs/Supra-50M-Instruct | no chat template in the repo (`tokenizer_config.json` has none, no `chat_template.jinja`) |
| AxiomicLabs/GPT-X3-150M, altslate/JugnuLM-110M-R2plus | no chat template (base-style releases; both are in the base suite) |
| opencerebral/Boris-1.3-125M-Instruct | ships no chat template; its older sibling Boris-125M-Instruct (Alpaca template) was run instead |
| tiiuae/Falcon-H1-Tiny-Coder-90M | no chat template (a code-completion model) |
| tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum, -pre-DPO, -Curriculum-pre-DPO | intermediate checkpoints of the released Instruct model, not separate releases |
| glouriousgautam/LiLM1-230M | 234M (above the 200M class, a near-size model); its own modeling code has no generation-cache path, which would make 4K-token caps very slow on the HF engine; not run this round |
| Cactus-Compute/needle3 | tool-calling model without a chat LM head (Phase 0) |
| SurjoLabs/Surjo-100m, q-project/Q-164M, facebook/MobileLLM-R1-140M (pre-R1.5) | no chat checkpoint of Surjo-100m or Q-164M was found; MobileLLM-R1-140M is superseded by R1.5-140M (run) |
| Madhuram-v1 (MaruthLabs) | closed weights (mentioned, not compared, per PREREGISTRATION.md 3) |
