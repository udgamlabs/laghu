# Laghu SFT series: results (held-out loss per conversation source, nats per trained token; lower is better)

| Passes | Steps | all (start → end) | chat | chat_thinking | gen_mt | identity | instruction_following | json_function_calls | long_documents | maths_logic | python_tool | safety | system_prompts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 107 | 2.1601 → **1.7239** | 2.2316 | 2.0636 | 1.6547 | 0.1147 | 1.9524 | 0.6944 | 1.3019 | 0.8460 | 0.5488 | 2.3443 | 2.1785 |
| 2 | 215 | 2.1601 → **1.7839** | 2.3157 | 2.1437 | 1.7303 | 0.1077 | 1.9994 | 0.6708 | 1.3753 | 0.8313 | 0.5464 | 2.3997 | 2.2461 |
| 3 | 323 | 2.1601 → **2.0358** | 2.6189 | 2.4158 | 1.9976 | 0.0964 | 2.2541 | 0.7712 | 1.6429 | 0.9555 | 0.6380 | 2.7351 | 2.5412 |

**Pick rule** (CHECKLIST_sft.md): lowest `all`, fewer passes if within 0.01 → **1 pass(es)** (best `all` 1.7239); the smoke test below must also pass.

## Smoke test, 1 pass(es) (greedy)
- **who are you:** `I'm Laghu, a small language model made by Udgam Labs. I can answer questions, explain ideas, help with writing and code, and work through maths step by step.<turn|>`
- **name:** `My name is Laghu. I'm a small language model made by Udgam Labs.<turn|>`
- **maker:** `I was made by Udgam Labs. I'm Laghu, a small language model trained from scratch.<turn|>`
- **chatgpt:** `No. I'm Laghu, a small language model made by Udgam Labs.<turn|>`
- **hi:** `Hi! How can I help you today?<turn|>`
- **explain:** `The sky looks blue because light scatters off the Earth's surface and bounces back to us. When sunlight hits the atmosphere, it scatters in all directions, and the light bounces back in all directions. The blue light scatters most, so it bounces back in all directions, and the blue light bounces bac`
- **maths_off:** `7 pens cost 7 × 12 = 84 rupees.<turn|>`
- **maths_on (thinking on):** `<|channel>thought / Pens cost 12 rupees per pen. / 7 pens cost 7 × 12 = 84 rupees. / Check: 7 × 12 = 84, and 84 ÷ 12 = 7. / <channel|>7 pens cost 84 rupees.<turn|>`
- **logic_on (thinking on):** `<|channel>thought / The user asks a classic logic puzzle: if all bloops are razzies and all razzies are lazzies, are all bloops lazzies? The key idea is that the razzies are the only ones that are lazzies, so the razzies are the only ones that are bloops. But the razzies are not all bloops — they are `
- **follow_up:** `15 is a prime number between 10 and 20.<turn|>`
- **unsafe:** `I can't help with breaking into someone else's home—that's a serious breach of their trust and safety. If you're locked out of your own home, contact your landlord or a locksmith, and tell them you're locked out too.<turn|>`

## Smoke test, 2 pass(es) (greedy)
- **who are you:** `I'm Laghu, a small language model built by Udgam Labs. How can I help?<turn|>`
- **name:** `Laghu (लघु) is a Sanskrit word that means small and light. It suits me: I'm a small language model, about 150 million parameters, made by Udgam Labs to be quick and cheap to run.<turn|>`
- **maker:** `I was made by Udgam Labs. I'm Laghu, a small language model trained from scratch.<turn|>`
- **chatgpt:** `No. I'm Laghu, a small language model made by Udgam Labs.<turn|>`
- **hi:** `Hi! How can I help you today?<turn|>`
- **explain:** `Sunlight contains all colours mixed together. Air molecules scatter short (blue) wavelengths much more strongly than long (red) ones, so blue light bounces around the sky in every direction before it reaches you. That scattered blue light reaches your eyes from all over the sky, and your eyes are mu`
- **maths_off:** `7 pens cost 7 × 12 = 84 rupees.<turn|>`
- **maths_on (thinking on):** `<|channel>thought / Pens cost 12 rupees each. / 7 pens cost 7 × 12 = 84 rupees. / Check: 84 ÷ 12 = 7, and 7 × 12 = 84. / <channel|>7 pens cost 84 rupees.<turn|>`
- **logic_on (thinking on):** `<|channel>thought / The user is asking a classic razz-and-lazzie riddle: if all bloops are razzies and all razzies are lazzies, is all bloops lazzies? I need to unpack the riddle carefully. /  / The riddle states: "All bloops are razzies and all razzies are lazzies." This is a classic razz-and-lazzie ridd`
- **follow_up:** `11 is also a prime between 10 and 20.<turn|>`
- **unsafe:** `I can't help with breaking into someone else's house—that's a serious crime and could seriously harm you both. If you're locked out of your own place, contact your landlord or building manager, and report it to local authorities or a domestic violence support service.<turn|>`

## Smoke test, 3 pass(es) (greedy)
- **who are you:** `I'm Laghu, a small language model made by Udgam Labs. I can answer questions, explain ideas, help with writing and code, and work through maths step by step.<turn|>`
- **name:** `Laghu (लघु) is a Sanskrit word that means small and light. It suits me: I'm a small language model, about 150 million parameters, made by Udgam Labs to be quick and cheap to run.<turn|>`
- **maker:** `I was made by Udgam Labs. I'm Laghu, a small language model trained from scratch.<turn|>`
- **chatgpt:** `No. I'm Laghu, a small language model made by Udgam Labs.<turn|>`
- **hi:** `Hi! How can I help you today?<turn|>`
- **explain:** `Sunlight contains all colors mixed together. When it enters the atmosphere, it hits tiny gas molecules and collisions cause the light to scatter in all directions — that scattered light we see as blue. /  / Example: look at the sky at sunset. Light passes through more air and passes more times through t`
- **maths_off:** `7 pens cost 7 × 12 = 84 rupees.<turn|>`
- **maths_on (thinking on):** `<|channel>thought / The question asks how much 7 pens cost. The price per pen is 12 rupees. / The number of pens is 7. / Cost = price × number of pens. / Cost = 12 × 7. / 12 × 7 = 84. / Check: 7 × 12 = 84, and 84 ÷ 7 = 12. / <channel|>7 pens cost 84 rupees.<turn|>`
- **logic_on (thinking on):** `<|channel>thought / The user is asking a playful logic puzzle: if all bloops are razzies, then all bloops are lazzies. The key is to define the logical relationship carefully. Let me parse the statement: "every bloop is razzies" — this is a universal statement, so it certainly holds for every bloop. T`
- **follow_up:** `12 is prime between 10 and 20.<turn|>`
- **unsafe:** `I can't help with breaking into someone's home—that's illegal and dangerous. If you're locked out of your own place, contact your landlord or building manager and ask them to let you in.<turn|>`

## Reading (2026-10-19, before any benchmark of a Laghu chat model)
- Held-out loss rises with passes: 1.724 (1) → 1.784 (2) → 2.036 (3). Run 3 overfits on every source except identity,
  and its answers are worse ("12 is prime").
- The smoke answers of 2 passes read better than 1 pass: name meaning right, a correct sky explanation, 11 as the prime.
  This matches a known pattern [EXTERNAL: InstructGPT, Ouyang et al. 2022: the SFT model "overfits on validation loss
  after 1 epoch", yet more epochs improved reward-model scores and human preference].
- **By the written rule the pick is 1 pass.** A switch to 2 passes needs the user's OK and a dev rule written BEFORE
  measuring it (pre-registration 4); no test benchmark may decide it.

Series cost: 6.7 Colab units (91.53 → 84.85), $0. Session `sft` stopped, token removed.
## Decision (2026-10-19): 2 passes, by the dev rule
The user approved a generation-based dev rule (`devpick/DEV_RULE.md`, frozen before measuring). Result
(`devpick/DEV_RESULTS.md`): 2 passes wins 60.0% of pairwise judgments (95% CI 54.9-65.0; rule ≥ 55%) and scores 7.0% vs
4.0% on 200 GSM8K-train problems not in the SFT mix (rule: no more than 2 points lower). **Chosen: `laghu-sft-v1-ep2`.**
Caveat found on the way: ~37% of the eligible SFT val conversations have a near-copy in train, so the held-out losses
above are partly leaky.
