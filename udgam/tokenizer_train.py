"""Train the UdgamLabs BPE tokenizer in Gemma-4 style (SentencePiece-like, '▁' for spaces, byte fallback)
so that HF transformers and llama.cpp's `gemma4` tokenizer produce identical token ids.

Design (see ops/STATE.md + PROPOSAL.md):
  * normalizer: Replace(' ' -> '▁') exactly like Gemma 4; decoder: '▁'->' ', ByteFallback, Fuse.
  * TRAINING-ONLY pre-split regex keeps merges inside word/number/punct/whitespace classes.
    The exported tokenizer uses Gemma's (effectively no-op) pre-tokenizer, matching llama.cpp,
    and all training data is later tokenized with the exported tokenizer, so train == inference.
  * digits never merge (each digit is its own token); newline runs and space runs are their own classes.
  * 256 byte-fallback tokens <0x00>..<0xFF>; Gemma 4 special tokens (tool calling, turns, thinking).
Usage: python tokenizer_train.py OUT_DIR [--vocab 32768] [--max-chars-per-source N]
"""
import json, os, argparse, random, glob
from tokenizers import Tokenizer, Regex, models, normalizers, pre_tokenizers, decoders, trainers

SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
            "<|turn>", "<turn|>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
            "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>", "<|channel>", "<channel|>"]
BYTES = [f"<0x{i:02X}>" for i in range(256)]
TRAIN_SPLIT = (r"▁+(?=▁)"                                  # a run of spaces minus the last (which joins the next word)
               r"|▁?[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)?"      # words (+ simple contractions)
               r"|▁?\p{N}"                                 # single digits only
               r"|▁?[^\s\p{L}\p{M}\p{N}▁]+"                # punctuation / symbol runs
               r"|▁+|\n+|\r+|\t+|[^\S\n\r\t]+")            # whitespace classes

def corpus(sample_dir, max_chars, seed=0):
    files = sorted(glob.glob(f"{sample_dir}/*.jsonl"))
    for f in files:
        n = 0
        for line in open(f):
            t = json.loads(line)["text"]
            yield t; n += len(t)
            if n >= max_chars: break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--samples", default=os.path.join(os.path.dirname(__file__), "../data/samples"))
    ap.add_argument("--max-chars-per-source", type=float, default=60e6)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    # 1) train core merges with the training-only pre-split
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(TRAIN_SPLIT), behavior="isolated")
    n_learn = a.vocab - len(SPECIALS) - len(BYTES)
    trainer = trainers.BpeTrainer(vocab_size=n_learn, min_frequency=5, limit_alphabet=600,
                                  max_token_length=24, show_progress=False, special_tokens=[])
    tok.train_from_iterator(corpus(a.samples, a.max_chars_per_source), trainer=trainer)
    learned = json.loads(tok.to_str())["model"]

    # 2) assemble final vocab: specials, bytes, learned (ids shifted), Gemma-style tokenizer.json
    vocab = {s: i for i, s in enumerate(SPECIALS + BYTES)}
    for t, _ in sorted(learned["vocab"].items(), key=lambda kv: kv[1]):
        if t not in vocab: vocab[t] = len(vocab)
    assert len(vocab) <= a.vocab, len(vocab)
    merges = learned["merges"]
    tj = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": vocab[s], "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for s in SPECIALS],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False},
        "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                                                      {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    }
    json.dump(tj, open(f"{a.out}/tokenizer.json", "w"), ensure_ascii=False)
    print(f"vocab {len(vocab)} (specials {len(SPECIALS)}, bytes 256, learned {len(vocab)-len(SPECIALS)-256}), merges {len(merges)}")

if __name__ == "__main__":
    main()
