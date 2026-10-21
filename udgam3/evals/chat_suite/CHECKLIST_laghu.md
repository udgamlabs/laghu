# Pre-launch checklist: Laghu on the narrowed chat suite (part 2b), 2026-10-19

**Part 2b was written and committed first** (`PREREGISTRATION.md`, commit bab9436, 18:59 IST); nothing below has been run.

**Exact config:**
- **Model:** `udgamlabs/Laghu-SFT-v1` @ 419343f (private; byte-identical weights to `udgam3-sft` @ c8c36d1 `laghu-sft-v1-ep2`, checked by LFS sha256).
- **Generation:** the rivals' exact pipeline (`colab_chat.py` + `chat_gen.py` + hash-locked vLLM 0.31.0, bf16, same caps, same batch settings), Colab G4, session `chatsuite`.
  - `CHAT_MANIFEST=manifest_laghu.json`: setup downloads only Laghu.
  - Tasks: GSM8K test, MATH-500 L1-3, IFEval, multi-turn E1/E2, loop prompts (greedy + shipped sampling), and the 40-item one-at-a-time check. Thinking off and on.
- **Order:**
  1. `up` (setup; the token is deleted by setup before any job);
  2. lane `laghu_smoke.json` (8 items per task): read stops, `<bos>`, thinking switch;
  3. lane `laghu_full.json`;
  4. `fetch`, then `down`.
- **Scoring on the Mac:**
  - `run_scoring.sh full` (score.py, validate.py, eyeball.py, make_tables.py), giving tables with Laghu's row;
  - `chat_quality_judge.py` (324 prompts × 4 opponents × 2 orders = 2,592 judgments);
  - `claims_2b.py`, which writes `CLAIMS_2B.md` (Holm over 9, own bar, loop threshold, gate).
- **Cost:** about 1-2 Colab units (Gargi's full run took 163 s plus ~10 min setup) of 77.1. The judge costs about $1 (gpt-oss-120b, about 3.1M input + 0.5M output tokens at provider prices) or about $0.40 (DeepSeek fallback).

| # | Check | Status |
|---|---|---|
| 1 | Storage | outputs are text (~50 MB raw, gz). ✅ |
| 2 | Memory | a 152M model; Gargi (384M) ran the same jobs on this GPU. ✅ |
| 3 | Dry run | the pipeline ran 17 models already. Laghu-specific risks (vLLM loading a tied-embedding Gemma-4, stops, template) are checked by the 8-item smoke lane before the full lane. The judge and claims scripts were tested end to end at $0.002 (DeepSeek, 4 prompts, a stand-in model). ✅ |
| 4 | End-to-end save | `fetch` pulls the tarball into `raw/full/laghu/`; per-item scores are written by score.py; `down` removes the token and stops the session. ✅ |
| 5 | Price and capacity | Colab subscription; judge prices as above. ✅ |
| 6 | Time | ~20 min on Colab + ~15 min of judging. ✅ |
| 7 | Budget | $0 Colab; judge ≤ $3 cap. ✅ |
| 9 | Frozen copies | scripts, manifests, jobs and part 2b committed before the run. ✅ |
| 11 | Scorer validated | the chat suite's scorers (validated on the rivals); the 40-item one-at-a-time check is part of the run; 10 outputs per task are read by eye (eyeball.py); 20 judgments are read by hand. ✅ |
| 12 | Approval = exact config | this file. |
| 14 | One run at a time | nothing else running. ✅ |
| 18 | Decision metric validated | the rival tables show the metrics separate models (e.g. GSM8K strict 0.0 to 23.3, multi-turn right 3.5 to 54.9). ✅ |
| 19 | Objective | Laghu's numbers on the three part-2b categories, the claims under the written rules, and the release-gate result. **Expected from dev numbers: maths fails (~8% vs 23.3%), so the gate fails whatever the other two show.** |

**Open: the judge.** The pre-registered judge is gpt-oss-120b. It is live on Hugging Face Inference Providers (Together, Groq, Cerebras and others), but our HF token is not allowed to call Inference Providers (403, tested 2026-10-19). Two ways:
- **(a) Keep the pre-registered judge.** You create a fine-grained HF token with "Make calls to Inference Providers" and save it to a file I name (never pasted in chat). I pin one provider and record it.
- **(b) Amend part 2b (before measuring)** to use DeepSeek-V4.1-Flash (MIT open weights), the judge that already ran the SFT dev pick; disclosed in the amendment log.

**APPROVED by the user 2026-10-19 19:04 IST:** "Approved as in CHECKLIST_laghu.md, with an amendment to part 2b (logged before measuring) to use DeepSeek-V4.1-Flash as the chat-quality judge. Run it all." Amendment logged in PREREGISTRATION.md (v1.8).
