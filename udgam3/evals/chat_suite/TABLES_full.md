### Best size-class rival per task (≤ 200M; groups 'rival' and 'rival_scan'; best mode per model)

| Task (metric) | Direction | Best rival | Mode | Score [95% CI] | Runner-up | Our bars: Gargi / Udgam-384M-Chat (best mode) |
|---|---|---|---|---|---|---|
| GSM8K accuracy | higher | **falcon-h1-tiny-multilingual-100m-instruct** | native | **24.9 [22.7, 27.3]** | falcon-h1-tiny-90m-instruct 23.3 [21.1, 25.6] | 35.6 [33.1, 38.3] (on) / 2.7 [1.9, 3.7] (off) |
| GSM8K strict (truncated = wrong; extra) | higher | **falcon-h1-tiny-multilingual-100m-instruct** | native | **24.8 [22.5, 27.2]** | falcon-h1-tiny-90m-instruct 23.3 [21.1, 25.6] | 35.6 [33.0, 38.2] (on) / 2.7 [1.9, 3.7] (on) |
| MATH-500 L1-3 accuracy | higher | **falcon-h1-tiny-r-90m** | native | **51.3 [44.9, 57.5]** | falcon-h1-tiny-90m-instruct 23.9 [19.0, 29.8] | 19.3 [14.8, 24.8] (on) / 7.1 [4.5, 11.1] (off) |
| MATH-500 L1-3 strict (extra) | higher | **falcon-h1-tiny-90m-instruct** | native | **23.9 [19.0, 29.8]** | falcon-h1-tiny-multilingual-100m-instruct 22.3 [17.4, 28.0] | 18.5 [14.1, 23.9] (on) / 6.7 [4.2, 10.6] (off) |
| IFEval prompt-level strict | higher | **falcon-h1-tiny-90m-instruct** | native | **60.8 [56.6, 64.8]** | falcon-h1-tiny-multilingual-100m-instruct 39.0 [35.0, 43.2] | 26.4 [22.9, 30.3] (off) / 27.2 [23.6, 31.1] (off) |
| IFEval instruction-level strict | higher | **falcon-h1-tiny-90m-instruct** | native | **70.5 [67.3, 73.6]** | falcon-h1-tiny-multilingual-100m-instruct 49.6 [46.3, 53.0] | 38.0 [34.7, 41.2] (off) / 41.4 [38.1, 44.7] (off) |
| IFEval prompt strict, answered only (extra) | higher | **falcon-h1-tiny-90m-instruct** | native | **60.8 [56.6, 64.8]** | falcon-h1-tiny-multilingual-100m-instruct 39.0 [35.0, 43.2] | 26.4 [22.9, 30.3] (off) / 27.2 [23.6, 31.1] (off) |
| Multi-turn E1 right % | higher | **falcon-h1-tiny-90m-instruct** | native | **54.9 [46.7, 62.8]** | cagliostro-v3.5-chat 52.8 [44.7, 60.8] | 49.3 [41.3, 57.4] (off) / 40.3 [32.6, 48.4] (on) |
| Multi-turn E1 near-copy % (lower better) | lower | **mobilellm-r1.5-140m** | native | **2.8 [1.3, 6.0]** | falcon-h1-tiny-r-90m 6.5 [3.9, 10.7] | 1.9 [0.7, 4.7] (on) / 21.5 [16.5, 27.5] (on) |
| E2: GSM8K as 2nd turn | higher | **falcon-h1-tiny-90m-instruct** | native | **27.0 [21.3, 33.5]** | falcon-h1-tiny-multilingual-100m-instruct 24.5 [19.1, 30.9] | 37.0 [30.6, 43.9] (on) / 3.5 [1.7, 7.0] (on) |
| Loop rate, greedy (lower better) | lower | **falcon-h1-tiny-90m-instruct** | native | **0.3 [0.1, 1.7]** | boris-125m-instruct 5.2 [3.3, 8.2] | 11.7 [8.7, 15.7] (on) / 8.6 [6.0, 12.2] (on) |
| Loop rate, shipped sampling (lower better) | lower | **falcon-h1-tiny-90m-instruct** | native | **0.0 [0.0, 1.2]** | falcon-h1-tiny-multilingual-100m-instruct 0.0 [0.0, 1.2] | 2.5 [1.3, 4.8] (off) / 7.4 [5.0, 10.8] (on) |

### GSM8K test (1,319), greedy, own chat template, answer = last number (math_eval.py rule)

| Model | Params | sha | Engine | Thinking | Cap | Accuracy % [95% CI] | Truncated % [95% CI] | Strict: truncated = wrong (extra) | Mean tokens | 'Final answer' % |
|---|---|---|---|---|---|---|---|---|---|---|
| **Laghu (Udgam-3), measured after part 2b was written** | | | | | | | | | | |
| laghu | 152M | `419343f6a2` | vllm | off | 4096 | **4.8 [3.8, 6.1]** | 0.5 [0.3, 1.1] | 4.8 [3.8, 6.1] | 72 | 6 |
| laghu | 152M | `419343f6a2` | vllm | on | 4096 | **8.0 [6.6, 9.5]** | 1.7 [1.2, 2.6] | 7.9 [6.5, 9.5] | 219 | 3 |
| **Size-class rivals (<= 200M)** | | | | | | | | | | |
| smollm2-135m-instruct | 135M | `12fd25f773` | vllm | native | 4096 | **2.1 [1.5, 3.1]** | 32.2 [29.8, 34.8] | 1.8 [1.2, 2.7] | 1468 | 50 |
| falcon-h1-tiny-90m-instruct | 91M | `e6389502a0` | vllm | native | 4096 | **23.3 [21.1, 25.6]** | 0.0 [0.0, 0.3] | 23.3 [21.1, 25.6] | 355 | 22 |
| falcon-h1-tiny-r-90m | 91M | `7385612bf0` | vllm | native | 4096 | **12.0 [10.3, 13.8]** | 85.2 [83.2, 87.0] | 3.7 [2.8, 4.9] | 3705 | 29 |
| supra2-100m-instruct | 101M | `2c90ade6ab` | vllm | native | 4096 (≤ 1989, context 2048) | **0.5 [0.3, 1.1]** | 38.1 [35.5, 40.7] | 0.5 [0.2, 1.0] | 790 | 0 |
| cagliostro-v3.5-chat | 146M | `e10b217173` | hf | native | 4096 (≤ 1992, context 2048) | **16.6 [14.7, 18.7]** | 4.9 [3.9, 6.2] | 16.6 [14.7, 18.7] | 300 | 4 |
| mobilellm-r1.5-140m | 140M | `b3fcd0d553` | vllm | native | 4096 | **3.6 [2.8, 4.8]** | 100.0 [99.7, 100.0] | 0.0 [0.0, 0.3] | 4096 | 5 |
| surjo-50m-sft-only | 54M | `f92fe7c8fe` | hf | native | 4096 (≤ 1991, context 2048) | **1.9 [1.3, 2.8]** | 37.8 [35.3, 40.5] | 0.8 [0.5, 1.5] | 806 | 2 |
| **Size-class chat models added by the 2026-10-19 scan** | | | | | | | | | | |
| falcon-h1-tiny-multilingual-100m-instruct | 108M | `295b1eb969` | vllm | native | 4096 | **24.9 [22.7, 27.3]** | 7.1 [5.8, 8.6] | 24.8 [22.5, 27.2] | 549 | 18 |
| boris-125m-instruct | 124M | `78ed3b5a99` | vllm | native | 4096 (≤ 968, context 1024) | **1.0 [0.6, 1.7]** | 8.0 [6.6, 9.5] | 0.8 [0.4, 1.4] | 105 | 2 |
| supra2-medium-instruct | 25M | `7e22157fd9` | vllm | native | 4096 (≤ 966, context 1024) | **2.0 [1.4, 3.0]** | 26.9 [24.6, 29.4] | 1.5 [1.0, 2.3] | 282 | 4 |
| blaze-sft | 48M | `30351574f9` | hf | native | 4096 (≤ 1987, context 2048) | **1.4 [0.9, 2.1]** | 25.8 [23.5, 28.2] | 1.1 [0.7, 1.9] | 639 | 3 |
| **Our own bars** | | | | | | | | | | |
| gargi | 384M | `7e0e6dd39d` | vllm | off | 4096 | **32.1 [29.7, 34.7]** | 0.1 [0.0, 0.4] | 32.1 [29.7, 34.7] | 220 | 100 |
| gargi | 384M | `7e0e6dd39d` | vllm | on | 4096 | **35.6 [33.1, 38.3]** | 0.7 [0.4, 1.3] | 35.6 [33.0, 38.2] | 314 | 98 |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | off | 4096 (≤ 4041, context 4096) | **2.7 [1.9, 3.7]** | 9.7 [8.2, 11.4] | 2.5 [1.8, 3.5] | 769 | 4 |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | on | 4096 (≤ 4034, context 4096) | **2.7 [1.9, 3.7]** | 8.3 [7.0, 10.0] | 2.7 [1.9, 3.7] | 727 | 4 |
| **Near-size references (230-362M, reported separately)** | | | | | | | | | | |
| gemma-3-270m-it | 268M | `ac82b4e820` | vllm | native | 4096 | **11.2 [9.6, 13.0]** | 15.8 [14.0, 17.9] | 10.8 [9.3, 12.6] | 836 | 73 |

Secondary rows (reasoning models at their recommended cap):

| Model | Thinking | Cap (source) | Items | Accuracy % [95% CI] | Same items at cap 4,096 | Truncated % | Strict (extra) | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| falcon-h1-tiny-r-90m | native | 32000 (no cap published for the 90M; 32,000 is the generation length its authors used for Falcon-H1-Tiny-R GRPO (blog)) | 300 | **15.0 [11.4, 19.5]** | 14.0 [10.5, 18.4] | 82.7 [78.0, 86.5] | 4.3 [2.5, 7.3] | 26716 |
| mobilellm-r1.5-140m | native | 8192 (model card example (max_new_tokens=8192)) | 1319 | **4.4 [3.4, 5.6]** | 3.6 [2.8, 4.8] | 100.0 [99.7, 100.0] | 0.0 [0.0, 0.3] | 8192 |

### MATH-500, levels 1-3 (238 items: 43 / 90 / 105), greedy, cap 4,096, Math-Verify on the final answer

| Model | Params | sha | Engine | Thinking | Cap | Accuracy % [95% CI] | L1 | L2 | L3 | Truncated % | Strict: truncated = wrong (extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Laghu (Udgam-3), measured after part 2b was written** | | | | | | | | | | | |
| laghu | 152M | `419343f6a2` | vllm | off | 4096 | **5.5 [3.2, 9.1]** | 7.0 | 5.6 | 4.8 | 3.8 | 5.0 [2.9, 8.6] |
| laghu | 152M | `419343f6a2` | vllm | on | 4096 | **6.3 [3.9, 10.1]** | 7.0 | 8.9 | 3.8 | 10.9 | 5.5 [3.2, 9.1] |
| **Size-class rivals (<= 200M)** | | | | | | | | | | | |
| smollm2-135m-instruct | 135M | `12fd25f773` | vllm | native | 4096 | **1.7 [0.7, 4.2]** | 4.7 | 2.2 | 0.0 | 52.5 | 0.4 [0.1, 2.3] |
| falcon-h1-tiny-90m-instruct | 91M | `e6389502a0` | vllm | native | 4096 | **23.9 [19.0, 29.8]** | 41.9 | 27.8 | 13.3 | 0.4 | 23.9 [19.0, 29.8] |
| falcon-h1-tiny-r-90m | 91M | `7385612bf0` | vllm | native | 4096 | **51.3 [44.9, 57.5]** | 60.5 | 51.1 | 47.6 | 72.7 | 21.8 [17.1, 27.5] |
| supra2-100m-instruct | 101M | `2c90ade6ab` | vllm | native | 4096 (≤ 2003, context 2048) | **5.5 [3.2, 9.1]** | 4.7 | 5.6 | 5.7 | 55.0 | 3.4 [1.7, 6.5] |
| cagliostro-v3.5-chat | 146M | `e10b217173` | hf | native | 4096 (≤ 2005, context 2048) | **16.8 [12.6, 22.1]** | 27.9 | 18.9 | 10.5 | 20.6 | 15.5 [11.5, 20.7] |
| mobilellm-r1.5-140m | 140M | `b3fcd0d553` | vllm | native | 4096 | **13.9 [10.0, 18.8]** | 27.9 | 16.7 | 5.7 | 100.0 | 0.0 [0.0, 1.6] |
| surjo-50m-sft-only | 54M | `f92fe7c8fe` | hf | native | 4096 (≤ 2007, context 2048) | **8.4 [5.5, 12.6]** | 11.6 | 6.7 | 8.6 | 43.3 | 6.7 [4.2, 10.6] |
| **Size-class chat models added by the 2026-10-19 scan** | | | | | | | | | | | |
| falcon-h1-tiny-multilingual-100m-instruct | 108M | `295b1eb969` | vllm | native | 4096 | **23.5 [18.6, 29.3]** | 44.2 | 23.3 | 15.2 | 15.5 | 22.3 [17.4, 28.0] |
| boris-125m-instruct | 124M | `78ed3b5a99` | vllm | native | 4096 (≤ 983, context 1024) | **2.5 [1.2, 5.4]** | 2.3 | 1.1 | 3.8 | 9.7 | 2.1 [0.9, 4.8] |
| supra2-medium-instruct | 25M | `7e22157fd9` | vllm | native | 4096 (≤ 976, context 1024) | **5.9 [3.5, 9.6]** | 4.7 | 8.9 | 3.8 | 58.0 | 3.4 [1.7, 6.5] |
| blaze-sft | 48M | `30351574f9` | hf | native | 4096 (≤ 2002, context 2048) | **3.4 [1.7, 6.5]** | 9.3 | 2.2 | 1.9 | 36.6 | 1.7 [0.7, 4.2] |
| **Our own bars** | | | | | | | | | | | |
| gargi | 384M | `7e0e6dd39d` | vllm | off | 4096 | **14.3 [10.4, 19.3]** | 30.2 | 13.3 | 8.6 | 5.9 | 14.3 [10.4, 19.3] |
| gargi | 384M | `7e0e6dd39d` | vllm | on | 4096 | **19.3 [14.8, 24.8]** | 37.2 | 20.0 | 11.4 | 15.5 | 18.5 [14.1, 23.9] |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | off | 4096 (≤ 4055, context 4096) | **7.1 [4.5, 11.1]** | 9.3 | 10.0 | 3.8 | 10.5 | 6.7 [4.2, 10.6] |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | on | 4096 (≤ 4048, context 4096) | **6.7 [4.2, 10.6]** | 4.7 | 7.8 | 6.7 | 13.4 | 6.7 [4.2, 10.6] |
| **Near-size references (230-362M, reported separately)** | | | | | | | | | | | |
| gemma-3-270m-it | 268M | `ac82b4e820` | vllm | native | 4096 | **16.4 [12.2, 21.6]** | 23.3 | 15.6 | 14.3 | 42.4 | 11.3 [7.9, 16.0] |

### IFEval (541 prompts), lm-eval 0.4.13 scorer, own chat template, greedy, cap 1,280

| Model | Params | sha | Engine | Thinking | Prompt strict % [95% CI] | Prompt loose % | Instruction strict % | Instruction loose % | Truncated % | Unfinished thinking | Prompt strict, answered only (extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Laghu (Udgam-3), measured after part 2b was written** | | | | | | | | | | | |
| laghu | 152M | `419343f6a2` | vllm | off | **26.1 [22.5, 29.9]** | 28.5 [24.8, 32.4] | 38.1 [34.8, 41.5] | 41.0 [37.7, 44.3] | 37.7 | 0 | 26.1 [22.5, 29.9] |
| laghu | 152M | `419343f6a2` | vllm | on | **17.7 [14.8, 21.2]** | 22.4 [19.1, 26.1] | 29.5 [26.4, 32.6] | 34.3 [31.0, 37.6] | 29.0 | 75 | 15.9 [13.1, 19.2] |
| **Size-class rivals (<= 200M)** | | | | | | | | | | | |
| smollm2-135m-instruct | 135M | `12fd25f773` | vllm | native | **23.5 [20.1, 27.2]** | 24.2 [20.8, 28.0] | 36.2 [33.0, 39.5] | 37.9 [34.7, 41.1] | 42.7 | 0 | 23.5 [20.1, 27.2] |
| falcon-h1-tiny-90m-instruct | 91M | `e6389502a0` | vllm | native | **60.8 [56.6, 64.8]** | 66.2 [62.1, 70.0] | 70.5 [67.3, 73.6] | 75.2 [72.1, 78.1] | 7.4 | 0 | 60.8 [56.6, 64.8] |
| falcon-h1-tiny-r-90m | 91M | `7385612bf0` | vllm | native | **22.9 [19.6, 26.6]** | 26.1 [22.5, 29.9] | 36.9 [33.9, 40.1] | 39.9 [36.8, 43.2] | 92.6 | 496 | 4.8 [3.3, 6.9] |
| supra2-100m-instruct | 101M | `2c90ade6ab` | vllm | native | **8.1 [6.1, 10.7]** | 10.5 [8.2, 13.4] | 18.6 [16.0, 21.2] | 21.9 [19.1, 24.8] | 50.3 | 0 | 8.1 [6.1, 10.7] |
| cagliostro-v3.5-chat | 146M | `e10b217173` | hf | native | **18.5 [15.4, 22.0]** | 21.1 [17.8, 24.7] | 30.9 [27.9, 34.1] | 33.8 [30.7, 37.0] | 45.5 | 0 | 18.5 [15.4, 22.0] |
| mobilellm-r1.5-140m | 140M | `b3fcd0d553` | vllm | native | **17.7 [14.8, 21.2]** | 20.0 [16.8, 23.5] | 29.5 [26.4, 32.7] | 33.0 [29.6, 36.4] | 100.0 | 541 | 0.0 [0.0, 0.7] |
| surjo-50m-sft-only | 54M | `f92fe7c8fe` | hf | native | **14.0 [11.4, 17.2]** | 15.9 [13.1, 19.2] | 24.9 [22.1, 27.8] | 27.5 [24.4, 30.5] | 28.7 | 0 | 14.0 [11.4, 17.2] |
| **Size-class chat models added by the 2026-10-19 scan** | | | | | | | | | | | |
| falcon-h1-tiny-multilingual-100m-instruct | 108M | `295b1eb969` | vllm | native | **39.0 [35.0, 43.2]** | 41.6 [37.5, 45.8] | 49.6 [46.3, 53.0] | 52.8 [49.4, 56.2] | 33.6 | 0 | 39.0 [35.0, 43.2] |
| boris-125m-instruct | 124M | `78ed3b5a99` | vllm | native | **8.7 [6.6, 11.4]** | 9.8 [7.6, 12.6] | 19.2 [16.7, 21.7] | 20.4 [17.8, 23.0] | 7.9 | 0 | 8.7 [6.6, 11.4] |
| supra2-medium-instruct | 25M | `7e22157fd9` | vllm | native | **11.1 [8.7, 14.0]** | 14.0 [11.4, 17.2] | 20.1 [17.4, 22.9] | 24.7 [21.7, 27.7] | 49.2 | 0 | 11.1 [8.7, 14.0] |
| blaze-sft | 48M | `30351574f9` | hf | native | **11.6 [9.2, 14.6]** | 14.0 [11.4, 17.2] | 23.3 [20.5, 26.0] | 26.7 [23.8, 29.7] | 43.3 | 0 | 11.6 [9.2, 14.6] |
| **Our own bars** | | | | | | | | | | | |
| gargi | 384M | `7e0e6dd39d` | vllm | off | **26.4 [22.9, 30.3]** | 29.8 [26.1, 33.7] | 38.0 [34.7, 41.2] | 40.9 [37.5, 44.3] | 42.5 | 0 | 26.4 [22.9, 30.3] |
| gargi | 384M | `7e0e6dd39d` | vllm | on | **26.1 [22.5, 29.9]** | 28.3 [24.6, 32.2] | 38.0 [34.8, 41.2] | 40.4 [37.1, 43.7] | 40.3 | 0 | 26.1 [22.5, 29.9] |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | off | **27.2 [23.6, 31.1]** | 29.8 [26.1, 33.7] | 41.4 [38.1, 44.7] | 44.0 [40.7, 47.4] | 34.2 | 0 | 27.2 [23.6, 31.1] |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | on | **26.1 [22.5, 29.9]** | 28.1 [24.5, 32.0] | 39.1 [35.8, 42.4] | 41.1 [37.8, 44.5] | 33.6 | 0 | 26.1 [22.5, 29.9] |
| **Near-size references (230-362M, reported separately)** | | | | | | | | | | | |
| gemma-3-270m-it | 268M | `ac82b4e820` | vllm | native | **25.0 [21.5, 28.8]** | 26.4 [22.9, 30.3] | 37.9 [34.7, 41.1] | 38.8 [35.6, 42.2] | 6.3 | 0 | 25.0 [21.5, 28.8] |

### Multi-turn (mt_eval.py rules), greedy, same engine for every model

E1: 214 two-turn follow-ups (cap 1,024 per turn). Near-copy = reply 2 nearly repeats reply 1 (lower is better); right = reply 2 contains the expected answer (144 checkable items).

| Model | Params | sha | Engine | Thinking | Near-copy % [95% CI] (lower better) | Right % [95% CI] | Near-copy and wrong % (checkable; extra) | Right, answered only (extra) | Reply 2 truncated % | Reply 2 unfinished thinking |
|---|---|---|---|---|---|---|---|---|---|---|
| **Laghu (Udgam-3), measured after part 2b was written** | | | | | | | | | | |
| laghu | 152M | `419343f6a2` | vllm | off | 20.6 [15.7, 26.5] | **41.7 [33.9, 49.8]** | 10.4 [6.4, 16.5] | 41.7 [33.9, 49.8] | 1.9 | 0 |
| laghu | 152M | `419343f6a2` | vllm | on | 36.0 [29.9, 42.6] | **32.6 [25.5, 40.7]** | 29.2 [22.4, 37.1] | 31.9 [24.9, 39.9] | 4.7 | 18 |
| **Size-class rivals (<= 200M)** | | | | | | | | | | |
| smollm2-135m-instruct | 135M | `12fd25f773` | vllm | native | 51.4 [44.7, 58.0] | **37.5 [30.0, 45.6]** | 43.1 [35.3, 51.2] | 37.5 [30.0, 45.6] | 9.3 | 0 |
| falcon-h1-tiny-90m-instruct | 91M | `e6389502a0` | vllm | native | 14.5 [10.4, 19.8] | **54.9 [46.7, 62.8]** | 7.6 [4.3, 13.2] | 54.9 [46.7, 62.8] | 0.5 | 0 |
| falcon-h1-tiny-r-90m | 91M | `7385612bf0` | vllm | native | 6.5 [3.9, 10.7] | **22.2 [16.2, 29.7]** | 6.9 [3.8, 12.3] | 10.4 [6.4, 16.5] | 64.0 | 129 |
| supra2-100m-instruct | 101M | `2c90ade6ab` | vllm | native | 40.2 [33.8, 46.9] | **18.8 [13.2, 25.9]** | 34.0 [26.8, 42.1] | 18.8 [13.2, 25.9] | 9.8 | 0 |
| cagliostro-v3.5-chat | 146M | `e10b217173` | hf | native | 36.4 [30.3, 43.1] | **52.8 [44.7, 60.8]** | 21.5 [15.6, 28.9] | 52.8 [44.7, 60.8] | 4.7 | 0 |
| mobilellm-r1.5-140m | 140M | `b3fcd0d553` | vllm | native | 2.8 [1.3, 6.0] | **3.5 [1.5, 7.9]** | 3.5 [1.5, 7.9] | 0.7 [0.1, 3.8] | 99.1 | 184 |
| surjo-50m-sft-only | 54M | `f92fe7c8fe` | hf | native | 43.0 [36.5, 49.7] | **12.5 [8.1, 18.9]** | 44.4 [36.6, 52.6] | 12.5 [8.1, 18.9] | 5.6 | 0 |
| **Size-class chat models added by the 2026-10-19 scan** | | | | | | | | | | |
| falcon-h1-tiny-multilingual-100m-instruct | 108M | `295b1eb969` | vllm | native | 42.1 [35.6, 48.8] | **45.8 [37.9, 54.0]** | 27.8 [21.1, 35.6] | 45.8 [37.9, 54.0] | 7.0 | 0 |
| boris-125m-instruct | 124M | `78ed3b5a99` | vllm | native | 75.8 [69.6, 81.1] | **3.5 [1.5, 8.0]** | 79.6 [72.2, 85.4] | 3.5 [1.5, 8.0] | 2.8 | 0 |
| supra2-medium-instruct | 25M | `7e22157fd9` | vllm | native | 50.0 [42.4, 57.6] | **5.0 [2.3, 10.5]** | 47.5 [38.8, 56.4] | 5.0 [2.3, 10.5] | 8.5 | 0 |
| blaze-sft | 48M | `30351574f9` | hf | native | 33.6 [27.7, 40.2] | **13.2 [8.6, 19.7]** | 35.4 [28.1, 43.5] | 13.2 [8.6, 19.7] | 8.9 | 0 |
| **Our own bars** | | | | | | | | | | |
| gargi | 384M | `7e0e6dd39d` | vllm | off | 2.8 [1.3, 6.0] | **49.3 [41.3, 57.4]** | 2.1 [0.7, 5.9] | 49.3 [41.3, 57.4] | 3.3 | 0 |
| gargi | 384M | `7e0e6dd39d` | vllm | on | 1.9 [0.7, 4.7] | **48.6 [40.6, 56.7]** | 1.4 [0.4, 4.9] | 48.6 [40.6, 56.7] | 4.2 | 0 |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | off | 24.8 [19.5, 31.0] | **36.1 [28.7, 44.2]** | 18.8 [13.2, 25.9] | 36.1 [28.7, 44.2] | 6.5 | 0 |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | on | 21.5 [16.5, 27.5] | **40.3 [32.6, 48.4]** | 13.9 [9.2, 20.5] | 40.3 [32.6, 48.4] | 6.1 | 0 |
| **Near-size references (230-362M, reported separately)** | | | | | | | | | | |
| gemma-3-270m-it | 268M | `ac82b4e820` | vllm | native | 45.8 [39.3, 52.5] | **43.1 [35.3, 51.2]** | 35.4 [28.1, 43.5] | 43.1 [35.3, 51.2] | 0.9 | 0 |

E2: GSM8K items 200-399 asked as the second turn (turn 1 = items 0-199 with their reference solutions) vs the same items asked alone (the GSM8K rows above); thinking on where the model has a switch; cap 4,096.

| Model | Thinking | Alone % [95% CI] | As 2nd turn % [95% CI] | Difference, points [95% CI] | McNemar p | 2nd-turn truncated % |
|---|---|---|---|---|---|---|
| laghu | on | 8.0 [5.0, 12.6] | 8.0 [5.0, 12.6] | +0.0 [-5.0, +4.5] | 1.000 | 6.0 |
| smollm2-135m-instruct | native | 2.5 [1.1, 5.7] | 1.0 [0.3, 3.6] | -1.5 [-4.0, +1.0] | 0.453 | 18.0 |
| falcon-h1-tiny-90m-instruct | native | 22.5 [17.3, 28.8] | 27.0 [21.3, 33.5] | +4.5 [-1.5, +11.0] | 0.222 | 0.0 |
| falcon-h1-tiny-r-90m | native | 9.5 [6.2, 14.4] | 12.5 [8.6, 17.8] | +3.0 [-3.0, +8.5] | 0.392 | 72.0 |
| supra2-100m-instruct | native | 0.5 [0.1, 2.8] | 0.5 [0.1, 2.8] | +0.0 [-1.5, +1.5] | 1.000 | 66.5 |
| cagliostro-v3.5-chat | native | 21.0 [15.9, 27.2] | 14.5 [10.3, 20.0] | -6.5 [-12.0, -1.0] | 0.029 | 2.0 |
| mobilellm-r1.5-140m | native | 4.5 [2.4, 8.3] | 2.0 [0.8, 5.0] | -2.5 [-6.0, +1.0] | 0.267 | 100.0 |
| surjo-50m-sft-only | native | 1.5 [0.5, 4.3] | 3.0 [1.4, 6.4] | +1.5 [-0.5, +4.0] | 0.375 | 42.5 |
| falcon-h1-tiny-multilingual-100m-instruct | native | 28.0 [22.2, 34.6] | 24.5 [19.1, 30.9] | -3.5 [-10.0, +3.0] | 0.371 | 5.0 |
| boris-125m-instruct | native | 1.0 [0.3, 3.6] | 2.0 [0.8, 5.0] | +1.0 [+0.0, +2.5] | 0.500 | 14.5 |
| supra2-medium-instruct | native | 1.5 [0.5, 4.3] | 0.0 [0.0, 1.9] | -1.5 [-3.5, +0.0] | 0.250 | 27.0 |
| blaze-sft | native | 0.5 [0.1, 2.8] | 1.0 [0.3, 3.6] | +0.5 [-1.0, +2.5] | 1.000 | 35.5 |
| gargi | on | 33.5 [27.3, 40.3] | 37.0 [30.6, 43.9] | +3.5 [-1.5, +9.0] | 0.265 | 0.5 |
| udgam-384m-chat | on | 2.5 [1.1, 5.7] | 3.5 [1.7, 7.0] | +1.0 [-2.5, +4.5] | 0.774 | 8.0 |
| gemma-3-270m-it | native | 12.5 [8.6, 17.8] | 11.0 [7.4, 16.1] | -1.5 [-7.0, +4.0] | 0.720 | 22.5 |

### Loop rate (324 held-out single-turn chat prompts, cap 1,024): reply hits the cap AND its last 256 tokens repeat

| Model | Params | sha | Engine | Thinking | Greedy loop % [95% CI] | Sampled loop % [95% CI] | Shipped sampling defaults (source) | Cap hit %, greedy / sampled |
|---|---|---|---|---|---|---|---|---|
| **Laghu (Udgam-3), measured after part 2b was written** | | | | | | | | |
| laghu | 152M | `419343f6a2` | vllm | off | **8.3 [5.8, 11.9]** | 1.2 [0.5, 3.1] | {"temperature": 0.7, "top_p": 0.95, "top_k": 64} (generation_config.json) | 11.4 / 3.7 |
| laghu | 152M | `419343f6a2` | vllm | on | **12.0 [8.9, 16.0]** | 1.9 [0.9, 4.0] | {"temperature": 0.7, "top_p": 0.95, "top_k": 64} (generation_config.json) | 12.7 / 2.5 |
| **Size-class rivals (<= 200M)** | | | | | | | | |
| smollm2-135m-instruct | 135M | `12fd25f773` | vllm | native | **17.3 [13.6, 21.8]** | 17.9 [14.1, 22.4] | {"temperature": 0.2, "top_p": 0.9} (model card example (generation_config.json sets none)) | 19.4 / 19.4 |
| falcon-h1-tiny-90m-instruct | 91M | `e6389502a0` | vllm | native | **0.3 [0.1, 1.7]** | 0.0 [0.0, 1.2] | {"temperature": 1.0} (none shipped (generation_config.json and card give none): vLLM default) | 6.8 / 9.0 |
| falcon-h1-tiny-r-90m | 91M | `7385612bf0` | vllm | native | **79.6 [74.9, 83.7]** | 0.0 [0.0, 1.2] | {"temperature": 1.0} (none shipped: vLLM default) | 95.1 / 93.5 |
| supra2-100m-instruct | 101M | `2c90ade6ab` | vllm | native | **26.5 [22.0, 31.6]** | 24.4 [20.0, 29.3] | {"temperature": 0.2, "top_p": 0.85, "top_k": 25} (model card example (its no_repeat_ngram_size=3 has no vLLM equivalent and is not applied)) | 27.5 / 25.3 |
| cagliostro-v3.5-chat | 146M | `e10b217173` | hf | native | **16.4 [12.7, 20.8]** | 2.2 [1.1, 4.4] | {"temperature": 0.3, "repetition_penalty": 1.1} (model card ('greedy or temperature 0.3 with a repetition penalty of 1.1')) | 18.2 / 3.1 |
| mobilellm-r1.5-140m | 140M | `b3fcd0d553` | vllm | native | **61.4 [56.0, 66.6]** | 1.2 [0.5, 3.1] | {"temperature": 0.6, "top_p": 0.95} (model card (the sampling its authors used to regenerate its SFT data); generation_config.json sets none) | 100.0 / 99.7 |
| surjo-50m-sft-only | 54M | `f92fe7c8fe` | hf | native | **20.7 [16.6, 25.4]** | 6.8 [4.5, 10.1] | {"temperature": 0.7, "top_p": 0.9} (model card example) | 22.5 / 7.7 |
| **Size-class chat models added by the 2026-10-19 scan** | | | | | | | | |
| falcon-h1-tiny-multilingual-100m-instruct | 108M | `295b1eb969` | vllm | native | **9.0 [6.3, 12.6]** | 0.0 [0.0, 1.2] | {"temperature": 1.0} (none shipped: vLLM default) | 26.5 / 16.7 |
| boris-125m-instruct | 124M | `78ed3b5a99` | vllm | native | **5.2 [3.3, 8.2]** | 0.9 [0.3, 2.7] | {"temperature": 0.7, "top_p": 0.95} (model card example) | 7.7 / 0.9 |
| supra2-medium-instruct | 25M | `7e22157fd9` | vllm | native | **40.1 [34.9, 45.5]** | 42.0 [36.7, 47.4] | {"temperature": 0.2, "top_p": 0.85, "top_k": 25} (model card example (no_repeat_ngram_size=3 not applied)) | 42.3 / 42.9 |
| blaze-sft | 48M | `30351574f9` | hf | native | **33.3 [28.4, 38.6]** | 9.0 [6.3, 12.6] | {"temperature": 0.7, "top_p": 0.9} (model card example) | 34.9 / 10.5 |
| **Our own bars** | | | | | | | | |
| gargi | 384M | `7e0e6dd39d` | vllm | off | **13.0 [9.7, 17.1]** | 2.5 [1.3, 4.8] | {"temperature": 0.7, "top_p": 0.95, "top_k": 64} (generation_config.json) | 15.7 / 4.6 |
| gargi | 384M | `7e0e6dd39d` | vllm | on | **11.7 [8.7, 15.7]** | 2.8 [1.5, 5.2] | {"temperature": 0.7, "top_p": 0.95, "top_k": 64} (generation_config.json) | 15.4 / 5.9 |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | off | **9.6 [6.8, 13.3]** | 7.7 [5.3, 11.1] | {"temperature": 0.1, "top_p": 0.9, "top_k": 40} (generation_config.json) | 12.0 / 9.0 |
| udgam-384m-chat | 384M | `d3feca0bed` | vllm | on | **8.6 [6.0, 12.2]** | 7.4 [5.0, 10.8] | {"temperature": 0.1, "top_p": 0.9, "top_k": 40} (generation_config.json) | 11.1 / 10.2 |
| **Near-size references (230-362M, reported separately)** | | | | | | | | |
| gemma-3-270m-it | 268M | `ac82b4e820` | vllm | native | **4.6 [2.8, 7.5]** | 0.0 [0.0, 1.2] | {"temperature": 1.0, "top_p": 0.95, "top_k": 64} (generation_config.json) | 13.6 / 10.8 |

### Scorer and generation checks (PREREGISTRATION.md 2.4)

First 40 GSM8K items, first 512 tokens: identical outputs / items compared (median first differing token).

| Model | bf16: batched vs one at a time (same engine) | Production vs HF fp32 one at a time | fp32: batched vs one at a time | Same prompt tokens (vs HF) | bf16 batch-noise rerun (full GSM8K): flips, accuracy change, McNemar p |
|---|---|---|---|---|---|
| smollm2-135m-instruct | 6/40 (50) | 6/40 (41) | — | 40/40 | — |
| falcon-h1-tiny-90m-instruct | 17/40 (179) | 14/40 (195) | — | 40/40 | — |
| falcon-h1-tiny-r-90m | 10/40 (186) | 10/40 (179) | — | 40/40 | — |
| supra2-100m-instruct | 34/40 (26) | 36/40 (55) | — | 40/40 | — |
| cagliostro-v3.5-chat | 11/40 (75) | 10/40 (88) | 40/40 (None) | 40/40 | — |
| mobilellm-r1.5-140m | 2/40 (137) | 2/40 (76) | — | 40/40 | — |
| falcon-h1-tiny-multilingual-100m-instruct | 19/40 (124) | 17/40 (46) | — | 40/40 | — |
| boris-125m-instruct | 37/40 (3) | 38/40 (56) | — | 40/40 | — |
| supra2-medium-instruct | 27/40 (28) | 28/40 (36) | — | 40/40 | — |
| surjo-50m-sft-only | 8/40 (21) | — | — | — | — |
| blaze-sft | 12/40 (39) | — | — | — | — |
| gargi | 26/40 (134) | 23/40 (77) | — | 40/40 | — |
| udgam-384m-chat | 14/40 (76) | 10/40 (88) | — | 40/40 | — |
| gemma-3-270m-it | 13/40 (39) | 15/40 (79) | — | 40/40 | — |
| laghu | 32/40 (45) | — | — | — | — |
