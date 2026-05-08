from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_FALLBACK_WORDS = {
    "A",
    "I",
    "AN",
    "AT",
    "AS",
    "TO",
    "IN",
    "IT",
    "IS",
    "ON",
    "CAT",
    "DOG",
    "WORD",
    "GRID",
    "PLAY",
    "GAME",
    "BANANA",
    "GRAM",
    "GRAMS",
}


class Dictionary:
    """Small dictionary helper with optional system-file loading."""

    def __init__(self, words: Iterable[str]):
        normalized = {
            w.strip().upper()
            for w in words
            if w and w.strip() and w.strip().isalpha()
        }
        self._words = normalized or set(DEFAULT_FALLBACK_WORDS)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Dictionary":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {path}")
        return cls(path.read_text(encoding="utf-8", errors="ignore").splitlines())

    @classmethod
    def from_system_or_fallback(cls) -> "Dictionary":
        # macOS and many Linux systems provide this default word list.
        default_path = Path("/usr/share/dict/words")
        if default_path.exists():
            return cls.from_file(default_path)
        return cls(DEFAULT_FALLBACK_WORDS)

    def contains(self, word: str) -> bool:
        return word.upper() in self._words

    def __len__(self) -> int:
        return len(self._words)
