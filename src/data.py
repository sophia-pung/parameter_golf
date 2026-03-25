import os                           # used to check that the corpus file exists
from pathlib import Path            # cross-platform path handling

import torch                        # for creating integer token-ID tensors
from torch.utils.data import Dataset  # base class that DataLoader expects

import tiktoken                     # OpenAI's fast BPE tokenizer (used to encode raw text)


# ── Tokenizer factory ──────────────────────────────────────────────────────────

def get_tokenizer(vocab_size: int):
    """
    Return a tiktoken encoding whose vocabulary is as close to vocab_size as possible.

    The challenge specifies a 1024-token vocabulary. tiktoken's 'gpt2' encoding has
    50257 tokens but we can use it as-is and simply clamp/remap if needed, or choose
    a smaller byte-level encoding. For now we return the gpt2 encoding and let the
    caller decide; the vocab_size arg is kept for future swap-in of a custom tokenizer.
    """
    if vocab_size <= 256:
        # Pure byte-level: encode each byte as its own token (no learned merges)
        # Useful for very tight vocab budgets; vocab_size argument is ignored here
        enc = tiktoken.get_encoding("gpt2")    # placeholder; byte-level handled below
        return ByteLevelTokenizer()            # custom class defined below
    else:
        enc = tiktoken.get_encoding("gpt2")    # 50k-token BPE; fine for vocab_size ≥ 1024
        return enc


class ByteLevelTokenizer:
    """
    Trivial byte-level tokenizer: each byte in [0, 255] is its own token.
    Produces a vocab of exactly 256 tokens — the minimum possible for any text.
    No external library needed; encoding is just converting the string to bytes.
    """

    vocab_size = 256                           # fixed: one token per possible byte value

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))      # UTF-8 bytes → list of ints in [0, 255]

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")  # ints → bytes → string


# ── Dataset ────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Reads a plain-text corpus, tokenizes it once up front, then slices it into
    fixed-length windows of `seq_len` tokens. Each item returned is a LongTensor
    of shape (seq_len,) that the training loop uses as both input and shifted target.

    Pre-tokenizing everything into one big flat array avoids re-encoding on each
    __getitem__ call and is the standard approach for language-model pre-training.
    """

    def __init__(self, path: str, seq_len: int, vocab_size: int):
        self.seq_len = seq_len                 # number of tokens per training window

        corpus_path = Path(path)               # convert string path to Path object
        if not corpus_path.exists():           # fail early with a clear message if file is missing
            raise FileNotFoundError(
                f"Corpus not found at '{path}'. "
                "Set data.path in configs/base.yaml to a .txt file."
            )

        # Read the entire corpus into memory as a single string
        text = corpus_path.read_text(encoding="utf-8")  # read raw UTF-8 text
        print(f"[data] loaded {len(text):,} characters from {path}")

        # Tokenize once and store as a flat list of integer IDs
        enc = get_tokenizer(vocab_size)        # get the appropriate tokenizer
        self.tokens = enc.encode(text)         # list of ints, one per token
        print(f"[data] {len(self.tokens):,} tokens after encoding")

        # Convert to a PyTorch LongTensor for fast slicing in __getitem__
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

    def __len__(self):
        # Number of non-overlapping windows of length seq_len that fit in the corpus.
        # We subtract seq_len so the last window never runs off the end.
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        # Return a contiguous chunk of seq_len tokens starting at idx.
        # The training loss will shift this by one position internally (ids[:, 1:]).
        return self.tokens[idx : idx + self.seq_len]   # (seq_len,) LongTensor


# ── Utility: download a small sample corpus ────────────────────────────────────

def download_sample_corpus(dest: str = "data/corpus.txt"):
    """
    Downloads the Project Gutenberg 'War and Peace' plain-text file as a default
    training corpus. This is ~3MB of English text — a reasonable smoke-test dataset.
    Only runs if the destination file does not already exist.
    """
    import urllib.request                      # stdlib: no extra dependency for a simple download

    url = (
        "https://www.gutenberg.org/files/2600/2600-0.txt"  # War and Peace, UTF-8
    )
    os.makedirs(os.path.dirname(dest), exist_ok=True)      # create data/ directory if needed
    if not os.path.exists(dest):                           # skip if already downloaded
        print(f"[data] downloading sample corpus → {dest}")
        urllib.request.urlretrieve(url, dest)              # blocking download; fine for a one-off
        print("[data] download complete")
    else:
        print(f"[data] corpus already exists at {dest}, skipping download")


# ── Script entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run `python src/data.py` to download the sample corpus and verify tokenization
    download_sample_corpus()                               # fetch corpus if needed
    ds = TextDataset(
        path="data/corpus.txt",                            # path to downloaded corpus
        seq_len=256,                                       # arbitrary window for this test
        vocab_size=50257,                                  # gpt2 vocab size
    )
    print(f"[data] dataset has {len(ds):,} windows")
    sample = ds[0]                                         # grab the first window
    print(f"[data] sample shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"[data] first 20 token IDs: {sample[:20].tolist()}")
