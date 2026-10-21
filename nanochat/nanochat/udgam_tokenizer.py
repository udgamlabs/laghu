"""UdgamLabs tokenizer: a Gemma-4-style HF `tokenizers` BPE (see udgam/tokenizer_train.py) behind
nanochat's tokenizer interface, so the dataloader / evals / engine work unchanged.
Chat formatting follows Gemma 4 (<|turn>role ... <turn|>), so exported models work with
llama.cpp / Ollama's Gemma 4 chat + tool-call parsers."""
import os
from functools import lru_cache
from tokenizers import Tokenizer


class UdgamTokenizer:
    def __init__(self, tokenizer_dir):
        self.dir = tokenizer_dir
        self.tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.bos_token_id = self.tok.token_to_id("<bos>")
        self.eos_token_id = self.tok.token_to_id("<eos>")
        assert self.bos_token_id is not None and self.eos_token_id is not None

    @classmethod
    def from_directory(cls, tokenizer_dir):
        return cls(tokenizer_dir)

    def get_vocab_size(self):
        return self.tok.get_vocab_size(with_added_tokens=True)

    def get_special_tokens(self):
        return {t.content for t in self.tok.get_added_tokens_decoder().values()}

    def id_to_token(self, id):
        return self.tok.id_to_token(id)

    @lru_cache(maxsize=64)
    def encode_special(self, text):
        i = self.tok.token_to_id(text)
        assert i is not None, f"unknown special token {text}"
        return i

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        pre = None if prepend is None else (prepend if isinstance(prepend, int) else self.encode_special(prepend))
        app = None if append is None else (append if isinstance(append, int) else self.encode_special(append))
        def wrap(ids):
            ids = list(ids)
            if pre is not None: ids.insert(0, pre)
            if app is not None: ids.append(app)
            return ids
        if isinstance(text, str):
            return wrap(self.tok.encode(text, add_special_tokens=False).ids)
        if isinstance(text, list):
            return [wrap(e.ids) for e in self.tok.encode_batch(text, add_special_tokens=False)]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def token_bytes(self):
        """Number of UTF-8 bytes each token decodes to (0 for special tokens) - used for bits-per-byte."""
        specials = set(self.tok.get_added_tokens_decoder().keys())
        out = []
        for i in range(self.get_vocab_size()):
            if i in specials:
                out.append(0); continue
            t = self.tok.id_to_token(i)
            if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                out.append(1)
            else:
                out.append(len(t.replace("▁", " ").encode("utf-8")))
        return out
