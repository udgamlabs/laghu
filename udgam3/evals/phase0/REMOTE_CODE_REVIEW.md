# Remote-code review for Phase 0 rivals (2026-10-19)

Ten size-class rivals ship their own modeling code (`trust_remote_code`). Before running them on the disposable Colab
VM, which holds no tokens or secrets, their `.py` files were downloaded **without executing them**, pinned to the commit
in `remote_code_pins.json`, and checked.
- **Searched for:** network access, shell or subprocess calls, `exec`/`eval`/`compile`/`__import__`, base64, pickle
  loads, file writes, downloads, environment-variable reads, and unusual imports.

| Model | Lines | Findings | Verdict |
|---|---|---|---|
| SurjoLabs/Surjo-100m | 1,089 | imports `fla` (flash-linear-attention, Gated DeltaNet kernels) and `flash_attn`; no risky calls | OK; install `flash-linear-attention` on the VM |
| AxiomicLabs/GPT-X3-150M | 428 | only `usage.py` example code (`.to("cuda")`) | OK |
| bench-labs/cagliostro-v3.5 | 518 | `torch.compile` of its own decode function | OK |
| opencerebral/Boris-2-Preview-1 | 322 | `__import__("transformers").__version__` version check | OK |
| altslate/JugnuLM-110M-R2plus | 50 | Qwen3 + value-residual layers | OK |
| glouriousgautam/LiLM1-230M-base | 591 | none | OK (234M: near-size reference, above the 200M class) |
| SLM-Archive/BananaMind-2-Pro | 355 | none | OK |
| User01110/100M-exp | 241 | `torch.compile(flex_attention)` | OK |
| bench-labs/tinctura-v1 | 313 | none | OK |
| q-project/Q-164M | 516 | `qpack.py` reads and writes its own packed weight file locally. `circuits.py` is a **built-in calculator**: an `ast` walker over whitelisted operators, gcd/lcm/reverse, and no `eval` | OK. **Its maths scores are tool-assisted** and are labelled so in every table |

**Cannot run in the harness (recorded, not compared):**
- `Nikity/lille-130m-base`: custom architecture with no modeling code in the repo.
- `Maggio33/GoLLeM-v5-128M`: config has no `model_type`.
- `Cactus-Compute/needle3`: a tool-calling model (`NeedleForToolCalling`), no LM head for chat or base scoring.
