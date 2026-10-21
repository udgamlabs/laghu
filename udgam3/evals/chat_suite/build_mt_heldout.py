"""Multi-turn held-out set v2 for the chat suite (PREREGISTRATION.md 2.3: the held-out set grows from 50 to >= 200).

Keeps the 50 hand-written items of udgam2/multiturn/heldout_followups.jsonl unchanged and in order, then adds new
hand-written items (written by Claude on 2026-10-19, before any Udgam-3 chat model exists) in the same style: a short
first question, then a short follow-up that only makes sense with the context. "check" lists strings, one of which
reply 2 must contain (word-boundary match, case-insensitive, as in mt_eval.py); the check never matches the answer
to the first question, so copying reply 1 fails it. Items without "check" measure near-copying only.

Writes mt_heldout_v2.jsonl and mt_heldout_v2_hashes.txt (sha256 of the normalised "q1 || q2" pair) so post-training
data can drop any conversation that contains one of these pairs.

  python3 udgam3/evals/chat_suite/build_mt_heldout.py
"""
import hashlib, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
V1 = os.path.join(ROOT, "udgam2", "multiturn", "heldout_followups.jsonl")

NEW = [
    # arithmetic
    ("What is 14 plus 9?", "now subtract 5", ["18"]),
    ("What is 6 times 7?", "and 6 times 8?", ["48"]),
    ("What is 81 divided by 9?", "and 72 divided by 9?", ["8", "eight"]),
    ("What is 3 cubed?", "and 4 cubed?", ["64"]),
    ("What is 25% of 80?", "and 75% of 80?", ["60"]),
    ("Add 125 and 375.", "now double it", ["1000", "1,000"]),
    ("What is 1000 minus 1?", "and 1000 minus 10?", ["990"]),
    ("What is the square root of 49?", "and of 64?", ["8", "eight"]),
    ("What is 11 times 11?", "and 12 times 12?", ["144"]),
    ("What is 0.5 plus 0.25?", "and 0.5 plus 0.75?", ["1.25"]),
    ("Multiply 15 by 4.", "now divide the result by 3", ["20"]),
    ("What is 3 to the power of 3?", "and 3 to the power of 4?", ["81"]),
    ("What is 17 + 26?", "and 17 + 36?", ["53"]),
    ("How many is a dozen?", "and half a dozen?", ["6", "six"]),
    ("What is 1/2 plus 1/4?", "and 1/2 plus 1/8?", ["5/8", "0.625"]),
    ("What is 30 divided by 6?", "now multiply that by 7", ["35"]),
    ("Round 3.14159 to two decimal places.", "and to one decimal place?", ["3.1"]),
    ("What is the next number in the sequence 2, 4, 6, 8?", "and the one after that?", ["12"]),
    ("What is 100 divided by 4?", "and 100 divided by 5?", ["20"]),
    ("What is 45 minus 18?", "add 3 to that", ["30"]),
    # units and time
    ("How many centimetres are in a metre?", "and in 3 metres?", ["300"]),
    ("How many grams are in a kilogram?", "and in half a kilogram?", ["500"]),
    ("How many seconds are in a minute?", "and in five minutes?", ["300"]),
    ("How many hours are in a day?", "and in a week?", ["168"]),
    ("How many months are in a year?", "and in two and a half years?", ["30"]),
    ("Convert 3 hours to minutes.", "now 4.5 hours", ["270"]),
    ("How many millilitres are in a litre?", "and in 2 litres?", ["2000", "2,000"]),
    ("How many days are in a leap year?", "and in a normal year?", ["365"]),
    ("How many weeks are in a year?", "and in half a year?", ["26"]),
    ("How many inches are in a foot?", "and in 3 feet?", ["36"]),
    # geography
    ("What is the capital of Spain?", "and of Portugal?", ["Lisbon"]),
    ("What is the capital of Canada?", "what about Mexico?", ["Mexico City"]),
    ("What is the capital of Egypt?", "and of Kenya?", ["Nairobi"]),
    ("What is the capital of Russia?", "and of Ukraine?", ["Kyiv", "Kiev"]),
    ("What is the capital of China?", "and of South Korea?", ["Seoul"]),
    ("What is the capital of Brazil?", "and of Argentina?", ["Buenos Aires"]),
    ("What is the capital of Norway?", "and Sweden?", ["Stockholm"]),
    ("Which continent is Nigeria in?", "and Peru?", ["South America"]),
    ("Which country is the city of Mumbai in?", "and the city of Lahore?", ["Pakistan"]),
    ("What is the longest river in Africa?", "and in South America?", ["Amazon"]),
    ("What is the tallest mountain in the world?", "and the tallest in Africa?", ["Kilimanjaro"]),
    ("Which ocean lies between Europe and North America?", "and between Africa and Australia?", ["Indian"]),
    ("What is the largest country by area?", "and the second largest?", ["Canada"]),
    ("Which country is shaped like a boot?", "which city there has a famous leaning tower?", ["Pisa"]),
    # science
    ("What is the chemical symbol for iron?", "and for copper?", ["Cu"]),
    ("What is the chemical formula for water?", "and for carbon dioxide?", ["CO2", "CO₂"]),
    ("How many planets are in our solar system?", "which one is known as the Red Planet?", ["Mars"]),
    ("What is the boiling point of water in Fahrenheit?", "and its freezing point?", ["32"]),
    ("What is the hardest natural substance?", "what element is it made of?", ["carbon"]),
    ("What do bees make?", "and what do silkworms make?", ["silk"]),
    ("How many bones are in the adult human body?", "and how many teeth does an adult usually have?", ["32"]),
    ("What is the largest organ of the human body?", "and the largest internal organ?", ["liver"]),
    ("What gas do humans breathe out?", "and which gas do we need to breathe in?", ["oxygen", "O2"]),
    ("What is H2O commonly called?", "and NaCl?", ["salt"]),
    ("Which planet has the most famous rings?", "and which planet is known for its Great Red Spot?", ["Jupiter"]),
    ("What is the main gas in Earth's atmosphere?", "and the second most common one?", ["oxygen"]),
    ("What is the centre of an atom called?", "and the negatively charged particles around it?", ["electron", "electrons"]),
    ("What is the speed of light, roughly, in kilometres per second?", "and the speed of sound in air, in metres per second?", ["343", "340", "330", "331"]),
    # language
    ("What is the plural of 'child'?", "and of 'person'?", ["people"]),
    ("What is the past tense of 'run'?", "and of 'swim'?", ["swam"]),
    ("What is the past tense of 'go'?", "and of 'eat'?", ["ate"]),
    ("What is the opposite of 'early'?", "and of 'ancient'?", ["modern"]),
    ("Translate 'cat' into French.", "and 'dog'?", ["chien"]),
    ("Translate 'water' into Spanish.", "and 'bread'?", ["pan"]),
    ("How do you say 'good morning' in German?", "and 'good night'?", ["Gute Nacht"]),
    ("Spell 'apple' backwards.", "now 'lemon'", ["nomel"]),
    ("How many letters are in the word 'banana'?", "and in 'kiwi'?", ["4", "four"]),
    ("What is the first letter of the English alphabet?", "and the last?", ["Z"]),
    ("Which letter comes after M in the alphabet?", "and the one after that?", ["O"]),
    ("What is the comparative form of 'good'?", "and the superlative?", ["best"]),
    ("Make the word 'happy' negative by adding a prefix.", "now do the same for 'possible'", ["impossible"]),
    ("What is the plural of 'leaf'?", "and of 'knife'?", ["knives"]),
    ("What is the opposite of 'buy'?", "and of 'borrow'?", ["lend"]),
    ("Give me the feminine form of 'king'.", "and of 'actor'?", ["actress"]),
    ("What is 'thank you' in Japanese?", "and 'hello'?", ["konnichiwa", "konnichi wa"]),
    ("What is the plural of 'cactus'?", "and of 'fungus'?", ["fungi", "funguses"]),
    # general facts
    ("How many players does a football (soccer) team have on the pitch?", "and a basketball team on the court?", ["5", "five"]),
    ("How many strings does a standard guitar have?", "and a violin?", ["4", "four"]),
    ("Who painted the Mona Lisa?", "name another painting by him", ["Last Supper", "Vitruvian", "Annunciation", "Lady with an Ermine", "Salvator Mundi", "Virgin of the Rocks", "Ginevra"]),
    ("Who wrote 'Pride and Prejudice'?", "name another novel by her", ["Emma", "Sense and Sensibility", "Persuasion", "Mansfield Park", "Northanger Abbey"]),
    ("Which animal is the national animal of India?", "and of Australia?", ["kangaroo", "kangaroos"]),
    ("How many continents are there?", "which one is the largest?", ["Asia"]),
    ("In which year did World War II end?", "and in which year did it begin?", ["1939"]),
    ("What is the currency of Japan?", "and of the United Kingdom?", ["pound", "pounds", "sterling", "GBP"]),
    ("What is the currency of the United States?", "and of India?", ["rupee", "rupees"]),
    ("How many days does February have in a normal year?", "and in a leap year?", ["29"]),
    ("What day comes after Monday?", "and two days after that?", ["Thursday"]),
    ("Which month comes after March?", "and three months after that?", ["July"]),
    ("How many sides does a square have?", "and an octagon?", ["8", "eight"]),
    ("How many hearts does an octopus have?", "and how many arms?", ["8", "eight"]),
    ("What colour is a ripe banana?", "and a ripe tomato?", ["red"]),
    ("What is the largest mammal?", "and the largest land mammal?", ["elephant", "elephants"]),
    ("What do caterpillars turn into?", "and what do tadpoles turn into?", ["frog", "frogs"]),
    ("Which animal can change its skin colour to blend in?", "and which bird can fly backwards?", ["hummingbird", "hummingbirds"]),
    ("What is the fastest land animal?", "and the fastest bird?", ["peregrine", "falcon"]),
    ("How many years are in a century?", "and in a millennium?", ["1000", "1,000", "thousand"]),
    # computing
    ("In Python, what does len('hello') return?", "and len('hi')?", ["2", "two"]),
    ("In Python, what is 7 // 2?", "and -7 // 2?", ["-4"]),
    ("What does HTML stand for?", "and CSS?", ["Cascading Style Sheets", "Cascading Style Sheet"]),
    ("What does CPU stand for?", "and RAM?", ["Random Access Memory", "Random-Access Memory"]),
    ("In Python, how do you get the last item of a list called items?", "and the first item?", ["items[0]"]),
    ("What is the binary number 1010 in decimal?", "and 1111?", ["15"]),
    ("What is the decimal number 8 in binary?", "and 5?", ["101"]),
    ("In JavaScript, which keyword declares a constant?", "and a block-scoped variable that can change?", ["let"]),
    ("What does SQL stand for?", "which keyword reads rows from a table?", ["SELECT"]),
    ("In Python, what is 'ab' * 3?", "and 'xy' * 2?", ["xyxy"]),
    # simple logic
    ("A shirt costs 20 dollars and is 10% off. What is the price now?", "and if it were 25% off?", ["15"]),
    ("If today is Wednesday, what day is tomorrow?", "and what day was yesterday?", ["Tuesday"]),
    ("I have 3 apples and buy 4 more. How many do I have?", "now I eat 2. How many are left?", ["5", "five"]),
    ("A train leaves at 9:00 and the trip takes 2 hours. When does it arrive?", "and if it is 30 minutes late?", ["11:30"]),
    ("What is the next letter: A, C, E, G?", "and the one after that?", ["K"]),
    ("Anna has twice as many books as Ben. Ben has 6. How many does Anna have?", "how many do they have together?", ["18"]),
    ("What is 2 + 2 x 3?", "and (2 + 2) x 3?", ["12"]),
    ("If I face north and turn right, which direction am I facing?", "and if I turn right again?", ["south"]),
    # open-ended (near-copy only)
    ("Write a short poem about autumn.", "make it rhyme", None),
    ("Give me a name for a coffee shop.", "something more playful", None),
    ("Explain what a black hole is.", "in two sentences", None),
    ("Suggest a weekend hobby.", "something I can do indoors", None),
    ("Tell me about the Eiffel Tower.", "why was it built?", None),
    ("Write a short thank-you note to a teacher.", "make it more formal", None),
    ("Give me three tips for studying.", "which one is the most important?", None),
    ("What is machine learning?", "give me a real-life example", None),
    ("Describe a sunset.", "now describe a sunrise", None),
    ("Suggest a healthy breakfast.", "something without eggs", None),
    ("Write a tweet about recycling.", "make it funny", None),
    ("Explain how a rainbow forms.", "why is the sky blue then?", None),
    ("Give me a motivational quote.", "who said it?", None),
    ("What is a budget?", "how do I make one?", None),
    ("Write a one-line slogan for a bakery.", "another one", None),
    ("Recommend a movie for a family night.", "something animated", None),
    ("Explain what DNA is.", "simpler please", None),
    ("Tell me a riddle.", "what's the answer?", None),
    ("Name a famous scientist.", "what are they known for?", None),
    ("Write an opening line for a mystery story.", "continue it with one more sentence", None),
    ("Give me a fun fact about the ocean.", "another one", None),
    ("How do I make tea?", "and iced tea?", None),
    ("Suggest a gift for my mother.", "under 20 dollars", None),
    ("Explain what inflation means.", "give an example with prices", None),
    ("Write a short email asking for a day off.", "make it shorter", None),
    ("What are the benefits of walking?", "how long should I walk each day?", None),
    ("Describe a cat in three words.", "now a dog", None),
    ("Give me an idea for a science project.", "something with plants", None),
    ("Explain the difference between weather and climate.", "give me an example of each", None),
    ("Help me name my new goldfish.", "something more serious", None),
    ("What is a haiku?", "write one about winter", None),
    ("Summarise the story of Cinderella.", "now in one sentence", None),
    ("List some ways to save water.", "which is easiest?", None),
    ("Write a birthday message for a friend.", "make it rhyme", None),
    ("What is an adjective?", "give me three examples", None),
    ("Explain how to tie a shoelace.", "is there a faster way?", None),
    ("I feel bored.", "I'm at home", None),
    ("Suggest a board game.", "for two players", None),
    ("What is the internet?", "who invented it?", None),
    ("Give me a word that means 'very big'.", "another one", None),
    ("Plan a simple picnic menu.", "make it vegan", None),
    ("Explain what a volcano is.", "name a famous one", None),
    ("Write a limerick about a dog.", "now about a cat", None),
    ("Tell me something interesting about space.", "explain it more simply", None),
    ("What should I pack for a beach trip?", "it might rain", None),
    ("Give me a short bedtime story.", "make it about a dragon", None),
    ("How can I be more productive?", "I work from home", None),
    ("Explain what an ecosystem is.", "give me an example", None),
    ("Write a sentence using the word 'curious'.", "now use 'brave'", None),
    ("What is a good way to learn a new language?", "for Spanish", None),
]


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def pair_hash(q1, q2):
    return hashlib.sha256(f"{norm(q1)} || {norm(q2)}".encode()).hexdigest()


def main():
    v1 = [json.loads(l) for l in open(V1)]
    rows = [dict(id=f"mt-v1-{i:03d}", q1=r["q1"], q2=r["q2"], check=r["check"], src="udgam2 heldout_followups v1") for i, r in enumerate(v1)]
    rows += [dict(id=f"mt-v2-{i:03d}", q1=q1, q2=q2, check=c, src="written 2026-10-19 (chat suite)") for i, (q1, q2, c) in enumerate(NEW)]
    seen = {}
    for r in rows:   # no duplicate first questions or pairs
        k = norm(r["q1"])
        assert k not in seen, f"duplicate q1: {r['q1']} ({seen[k]})"
        seen[k] = r["id"]
    hit = lambda text, checks: any(re.search(r"(?<![\w.])" + re.escape(c) + r"(?![\w])", text, re.I) for c in checks)
    for r in rows:   # the check never fires on the bare question texts themselves (a reply that echoes them gets no credit)
        if r["check"]:
            assert not hit(r["q1"] + " " + r["q2"], r["check"]), f"check matches the question: {r}"
    with open(os.path.join(HERE, "mt_heldout_v2.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "mt_heldout_v2_hashes.txt"), "w") as f:
        f.write("\n".join(sorted(pair_hash(r["q1"], r["q2"]) for r in rows)) + "\n")
    n_check = sum(1 for r in rows if r["check"])
    print(f"{len(rows)} items ({len(v1)} from v1 + {len(NEW)} new), {n_check} checkable")


if __name__ == "__main__":
    main()
