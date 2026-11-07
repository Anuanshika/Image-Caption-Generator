"""A tiny word-level tokenizer (pure Python, saved as JSON)."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"
SPECIALS = [PAD, UNK, START, END]  # ids 0, 1, 2, 3


def clean_caption(text: str) -> list[str]:
    """Lowercase, keep letters only, drop 1-letter words except 'a'."""
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    return [w for w in words if len(w) > 1 or w == "a"]


class Vocabulary:
    """Maps words <-> integer ids."""

    def __init__(self, words: list[str]) -> None:
        self.itos = list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id = self.stoi[PAD], self.stoi[UNK]
        self.start_id, self.end_id = self.stoi[START], self.stoi[END]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, captions: Iterable[str], min_freq: int = 5) -> "Vocabulary":
        counts = Counter(w for c in captions for w in clean_caption(c))
        words = sorted(w for w, n in counts.items() if n >= min_freq)
        return cls(SPECIALS + words)

    def encode(self, text: str, max_len: int) -> list[int]:
        """'A dog runs' -> [start, a, dog, runs, end, pad, pad, ...] (length max_len)."""
        words = clean_caption(text)[: max_len - 2]
        ids = [self.start_id] + [self.stoi.get(w, self.unk_id) for w in words] + [self.end_id]
        return ids + [self.pad_id] * (max_len - len(ids))

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Ids -> words, stopping at <end> and skipping special tokens."""
        words = []
        for i in ids:
            i = int(i)
            if i == self.end_id:
                break
            if i in (self.pad_id, self.start_id):
                continue
            words.append(self.itos[i])
        return words

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
