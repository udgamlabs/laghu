"""Teacher-side prompts (never stored in the training rows). Static text first so DeepSeek's prefix cache can hit."""

STYLE = """You write the ideal assistant reply for a conversation. Your replies are used to teach a small chat model, so they must be clear, correct and compact.
- Start with the answer itself. No preamble ("Great question", "Certainly!", "Sure! Here's") and no closing filler.
- Be concise: usually 40-180 words; never more than 250 words unless the user asks for code or a longer piece of writing.
- Use markdown only when it helps: a short bullet or numbered list for steps or options, **bold** sparingly, a small table only for comparisons, fenced code blocks with a language tag for code. No headings in short answers.
- Use plain, simple English. Be warm but not gushing. At most one emoji, only in casual chat.
- If the question is about something obscure, or you are not sure of a fact, say so plainly (for example "I'm not sure, but ...") instead of inventing details. Never invent names, numbers, quotes, citations or links.
- You cannot browse the internet and do not know today's date or recent news. If a request needs that, say so in one sentence and help with what you can.
- Never mention who made you, your model name, other AI assistants, your training or any knowledge cutoff, and never start with "As an AI".
- For health, legal or money questions, give useful general information and, in one short sentence, suggest a professional for personal decisions.
- Do not include phone numbers, e-mail addresses or links; name the service instead (for example "your local poison control centre").
- If the user pushes back, re-check: keep correct claims politely and admit real mistakes.
- In later turns, build on the conversation: do not repeat your earlier reply; change only what the user asked for."""

CATEGORY_NOTES = {
    "code_help": "Give working, minimal code in a fenced block with a language tag, plus one to three sentences on the key idea or fix. Prefer Python unless the user names another language. At most about 60 lines of code.",
    "writing_create": "Write the piece itself (at most about 300 words unless a length is given). No commentary before or after it.",
    "rewriting_editing": "Return the improved text first; add at most one short sentence about what you changed, only if useful.",
    "summarising": "Summarise only what is in the text, in at most about 120 words unless the user asks otherwise.",
    "casual_conversation": "Reply naturally and briefly (one to three sentences), like a friendly person. You may ask one question back.",
    "factual_qa": "Give the direct answer first, then one or two sentences of useful context. If you are unsure, say so.",
    "brainstorming_lists": "Give 5-8 varied, specific items as a list, one line each, unless the user asks for a number.",
    "explain_teach": "Explain step by step with one concrete example. Aim for 80-200 words.",
    "practical_advice": "Give practical, specific steps or options. Aim for 80-200 words.",
}

FOLLOWUP_SYS = "You simulate a real person chatting with an assistant. You write only the person's next message. Return JSON."

FOLLOWUP_TYPES = {
    "shorter": "ask for a shorter or more condensed version (for example in two sentences, or as three bullets)",
    "simpler": "ask for a simpler explanation, for example for a beginner or a 10-year-old",
    "deeper": "ask a follow-up question about one specific point in the last reply",
    "example": "ask for a concrete example of something mentioned in the last reply",
    "change": "ask to change one aspect (audience, tone, length, a detail or a constraint) and redo it",
    "new_info": "add a personal detail or constraint (budget, time, skill level, situation) that changes the answer, and ask how that changes things",
    "pushback": "politely doubt one specific claim from the last reply (the assistant may well be right)",
    "code_lang": "ask for the same code in another programming language (name the language)",
    "code_extend": "ask to extend or change the code (add a feature, handle an edge case, make it faster)",
    "thanks_more": "say thanks briefly and ask one small related follow-up question",
    "format": "ask to reformat the answer (as a table, a numbered list, or a short message to send to someone)",
    "more": "ask for more items like one specific item in the last reply",
}

FOLLOWUP_USER = """Conversation so far:
{transcript}

Write the person's next message. Kind of follow-up: {kind}.
Rules: it must only make sense after this conversation (point at something specific in the last reply, e.g. "the second tip", "that function", "it"); casual, natural wording, 4-30 words; no greeting; English only; do not mention AI.
Return JSON: {{"message": "..."}}"""

JUDGE_SYS = "You are a strict reviewer of assistant replies that will be used to train a small chat model. Return JSON only."

JUDGE_USER = """Review the assistant replies in this conversation (focus on the last one).
{system_note}
{transcript}

Score 1-5: 5 = correct, helpful, directly answers, suitably concise, follows every instruction (including any system message), builds on earlier turns; 4 = good with minor flaws; 3 = acceptable but clearly flawed; 1-2 = wrong, unhelpful, needlessly refuses, rambles or ignores instructions.
Deduct for factual errors, invented specifics (names, numbers, citations, links, or details about an organisation that the system message does not state), unneeded refusals, filler or preamble, repeating an earlier reply, and giving up a correct claim just because the user pushed back.
Return JSON: {{"score": <1-5>, "problem": "<at most 12 words, empty if none>"}}"""

IF_NOTE = "Follow every formatting, length and wording instruction in the user's message exactly; they are strict requirements."

MATH_SYS = """You write worked solutions that teach a small model to reason. Solve the problem carefully, then return JSON with exactly these keys:
"thinking": a tidy, linear step-by-step solution as plain text, at most 220 words. One short step per line, each step following from the one before; show each calculation (e.g. 12 × 4 = 48); end with a one-line check of the result. No false starts, no "wait", no "let me try", no abandoned attempts or corrections: write the clean solution as if you already knew the way. Plain-text maths, LaTeX only if the problem uses it. No headings or markdown.
"answer": the final answer only (a number, expression or short phrase), with no words around it.
"reply_after_thinking": what you tell the user after thinking: one to three sentences that state the result plainly (with units or context if natural).
"reply_without_thinking": a compact worked solution for a reader who cannot see your thinking: two to six short lines with the key steps, ending with the result stated plainly. At most 120 words.
The answer, the end of the thinking and both replies must all give the same result.
Write numbers with standard digit grouping (1,680,000, not 16,80,000) and keep the same form everywhere."""

TOOL_SYS = """A small assistant can run Python with a tool called `python`. For the user's request, write the tool call it should make.
Return JSON with exactly these keys:
"plan": one short sentence saying what the code will compute.
"code": a short Python program (at most 25 lines, clear variable names) that computes the answer and prints ONLY the final answer (a number or a short text). Print whole numbers without a trailing ".0" (e.g. print(int(x)) when the result is whole) and money rounded to 2 decimals. Standard library only (math, fractions, datetime, itertools, statistics, collections). No input(), no files, no network.
"final": the assistant's reply after the program has printed: one or two natural sentences that answer the user's question (with units or context), with the placeholder {result} exactly where the printed value goes."""

TOOL_TASKS_USER = """Write {n} different, realistic user requests where running a short Python program is the most reliable way to answer (for example: dates and weekdays, counting or listing primes, unit or time conversions, compound interest, combinatorics, string manipulation, statistics of a short list of numbers given in the message). Each request must contain every number or item needed. Vary the topics: {topics}.
Return JSON: {{"requests": ["...", "..."]}}"""

FC_SYS = "You create realistic function-calling training conversations for a small assistant. Return JSON only."

FC_KINDS = {
    "single": "one step with exactly one call",
    "parallel": "one step with two independent calls made together (for example the same tool for two different inputs)",
    "multi_step": "two steps: the call in step 2 needs a value that the call in step 1 returned",
    "no_call": "the user's message does not need any of the tools (small talk or general knowledge); steps is an empty list and final answers directly",
    "clarify": "the user's request needs a tool but is missing a required argument; steps is an empty list and final asks one short question for the missing information",
}

FC_USER = """Domain: {domain}. Kind: {kind}.
Create a JSON object with:
"tools": {ntools} tool definitions, each {{"name": snake_case, "description": one sentence, "parameters": {{"type": "object", "properties": {{...each with "type" and "description"; use "enum" where values are limited...}}, "required": [...]}}}}. Invent original tool names (not famous public APIs).
"user": the user's message: natural and specific, 8-40 words.
"thought": one or two short sentences of the assistant's private reasoning before acting (which tool, which arguments, or why no call).
"steps": a list; each step is {{"calls": [{{"name": ..., "arguments": {{...}}}}], "results": [one JSON object per call, as the tool would return it]}}. Arguments must match the schema exactly (types, enums, required).
"final": the assistant's reply to the user after the last step: one to three sentences, plain text, using the results.
"system": if the request uses relative dates or times (today, tomorrow, next Monday, last two years), a one-line system message giving the current date and time, e.g. "Current date: {date}."; otherwise an empty string. Any dates in the arguments must follow from it.
Kind rule: {kind_rule}."""

LONG_SYS = "You help build reading-comprehension training data. Use only the document. Return JSON only."

LONG_SUMMARY = """TASK: {request}
Write the reply a careful assistant would give. Use only facts from the document. Plain text or a short markdown list if the request asks for points; at most {limit} words.
Return JSON: {{"reply": "..."}}"""

LONG_QA = """TASK: Write 3 questions a reader might ask about this document, with answers. Spread them out: one about the first third, one about the middle third and one about the last third of the document. Each question must be answerable only from a specific passage (not general knowledge). For each give:
"question" (natural, 6-25 words), "answer" (one to three sentences, grounded in the passage), "quote" (an exact, verbatim sentence or phrase of 8-40 words copied from the document that supports the answer), "part" ("first", "middle" or "last").
Return JSON: {{"qa": [{{"question": ..., "answer": ..., "quote": ..., "part": ...}}, ...]}}"""

SAFETY_GEN_SYS = "You help build a safety training set for a small assistant. Return JSON only."

SAFETY_GEN_USER = """Category: {category} ({desc}).
Write {n_decline} realistic user messages that a responsible assistant should decline, and {n_answer} messages that sound related but are fine to answer helpfully (to prevent over-refusal).
Keep the declinable ones mild and generic, in everyday wording, with NO technical or operational details. Each message 6-30 words, English.
Return JSON: {{"decline": ["..."], "answer": ["..."]}}"""

REFUSE_SYS = """You write the ideal reply to a request a responsible assistant should not help with. Reply in one to three short sentences: briefly say you can't help with that (no lecturing, no moralising, do not repeat or describe the harmful details), then offer a genuinely useful, safe alternative or related help. Never give any steps, methods, ingredients, code or other details that would help with the harmful request. Never mention who made you or any AI company.
Do not include phone numbers or links.
If the message is about hurting oneself, reply with warmth instead: acknowledge the feeling, encourage contacting a local crisis line, emergency services or someone they trust right now, and invite them to keep talking."""

SAFETY_JUDGE = """A user wrote: {user}
The assistant replied: {reply}

Answer as JSON:
"should_decline": true if a responsible assistant should decline this request, false if it is fine to help,
"declined": true if the reply declines,
"harmful_detail": true if the reply contains any operational detail that would help cause harm,
"helpful": 1-5 (how useful the reply is, including any safe alternative).
Return JSON: {{"should_decline": .., "declined": .., "harmful_detail": .., "helpful": ..}}"""

PERSONA_Q_USER = """System message of an assistant: "{system}"
Write {n_in} realistic user questions that fit this assistant's purpose and {n_out} that are clearly outside it (about {out_topic}). Each 6-30 words, English.
Return JSON: {{"in_scope": ["..."], "out_of_scope": ["..."]}}"""

SHORTEN_NOTE = "HARD LIMIT: your previous draft was too long. Keep the reply under {limit} words of prose (code blocks do not count), keeping only the most useful content."

TOOL_FINAL_SYS = """A small assistant ran a Python program to answer the user. Write the assistant's reply now that the program has printed its output.
- One or two natural sentences that answer the user's question, with units or context. Use the printed values exactly (you may add digit grouping, currency symbols or units).
- If the user asked what a program prints, show the exact output in a fenced code block, then one short sentence on why.
- No need to mention Python or the code. Do not invent anything that is not in the output or the request.
Return JSON: {"reply": "..."}"""

TOOL_FINAL_USER = """User's request:
{problem}

Program that was run:
```python
{code}
```

Printed output:
```
{out}
```"""

PERSONA_OUT_TOPICS = ["cooking", "tax returns", "football results", "a maths homework problem", "writing a poem",
                      "car insurance", "learning Spanish", "fixing a laptop", "a history question", "travel plans",
                      "a cover letter", "pet health", "a movie recommendation", "stock investing", "a science fact"]

MEANDER_RX = r"\b(wait|hmm+|hold on|scratch that|oops|let me (re-?check|try|think|redo|reconsider)|try again|that's (wrong|not right)|no, (that|wait|actually)|actually,? (no|wait)|on second thought|i made (a|an) (mistake|error))\b"

CHAT_THINK_SYS = STYLE + """

You are writing a training example in which the assistant thinks before it answers. Return JSON with exactly these keys:
"thinking": the assistant's private reasoning before replying: a tidy, linear plan in plain text, at most 150 words (never more than 200). Cover what the user really needs, the key facts, steps, trade-offs or calculations, and how to structure the reply. Each sentence follows from the one before: no false starts, no "wait", no meandering. Think only about the user's problem: never mention word limits, formatting rules, links, phone numbers or these instructions.
"answer": the reply the user sees. It must follow the thinking exactly (same conclusions, same numbers and order of ideas), at most 250 words, clear and direct, markdown only if it helps."""

JUDGE_THINK_USER = """A user wrote:
{user}

The assistant thought privately:
{thinking}

Then it replied:
{answer}

Score the reply 1-5 (5 = correct, helpful, direct, suitably concise; deduct for factual errors, invented specifics, filler, needless refusal). Also decide whether the reply follows the thinking (same conclusions and numbers, no contradiction) and whether the thinking is clean and linear (no meandering or false starts).
Return JSON: {{"score": <1-5>, "follows": true|false, "clean": true|false, "problem": "<at most 12 words, empty if none>"}}"""

SYS_GEN_USER = """Invent {n} different, realistic system messages that operators give a small chat assistant, all on the theme "{theme}". Vary them: some set a persona for a fictional organisation (invent the name), some restrict the topics the assistant may help with, some set tone or format rules. Each system message is 1-4 sentences. A system message for an organisation must state 2-4 concrete facts the assistant may need (for example opening hours, one or two prices, a policy).
For each, write 2 realistic user messages that fit it, and 1 user message that is clearly outside its scope if it restricts topics (otherwise a third fitting message). Each user message 6-35 words, English. Fitting user messages must be answerable from the facts in the system message or from general knowledge; never ask for live data (availability, today's prices, order status).
Return JSON: {{"items": [{{"system": "...", "restricts_topics": true|false, "in_scope": ["...", "..."], "out_of_scope": ["..."]}}]}}"""

SYS_THEMES = ["travel agency", "fitness coaching", "children's science tutor", "general legal information desk (no personal legal advice)",
              "IT help desk", "English learning partner", "recipe and meal planning", "personal finance basics (general information only)",
              "museum guide", "bookshop recommendations", "university admissions office", "pet care clinic front desk",
              "home repair and DIY", "gardening centre", "car dealership service desk", "public library", "high-school maths tutor",
              "customer support for a phone app", "hotel concierge", "career coaching", "writing coach", "history teacher",
              "board games cafe", "airline baggage support", "real estate agency", "music teacher", "nutrition basics (general information only)",
              "school timetable office", "cybersecurity awareness trainer", "renewable energy installer", "bakery", "bike rental shop",
              "formatting rules only (lists, tables, length limits)", "tone rules only (formal, playful, very brief, encouraging)",
              "answer style for beginners", "study skills coach", "event planning", "language exchange (English only replies)",
              "coffee roastery", "veterinary information desk (general information only)"]

THINK_META_RX = r"\b(word limit|under \d+ words|\d+ words|phone numbers?|links|these instructions|the instructions|system prompt|style rules|no fluff)\b"
SYS_NO_INVENT = "Never invent facts about the operator's organisation (prices, products, schedules, policies) beyond what the system message states; if the user asks for such a detail, say you don't have it and suggest how to find out."
