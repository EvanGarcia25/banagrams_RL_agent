from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any

from .dictionary import Dictionary


class InvalidAction(ValueError):
    """Raised when an action violates game or board constraints."""


@dataclass(frozen=True)
class GameConfig:
    grid_size: int = 15
    initial_hand_size: int = 12


class BananagramsGame:
    """
    Minimal single-player Bananagrams logic.

    Core actions:
    - place(letter, row, col)
    - remove(row, col)
    - dump(letter)
    """

    TILE_DISTRIBUTION = {
        "A": 13,
        "B": 3,
        "C": 3,
        "D": 6,
        "E": 18,
        "F": 3,
        "G": 4,
        "H": 3,
        "I": 12,
        "J": 2,
        "K": 2,
        "L": 5,
        "M": 3,
        "N": 8,
        "O": 11,
        "P": 3,
        "Q": 2,
        "R": 9,
        "S": 6,
        "T": 9,
        "U": 6,
        "V": 3,
        "W": 3,
        "X": 2,
        "Y": 3,
        "Z": 2,
    }

    def __init__(
        self,
        dictionary: Dictionary,
        config: GameConfig | None = None,
        seed: int | None = None,
    ):
        self.dictionary = dictionary
        self.config = config or GameConfig()
        self.rng = Random(seed)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.grid = [
            ["" for _ in range(self.config.grid_size)]
            for _ in range(self.config.grid_size)
        ]
        self.tile_bag = [
            letter
            for letter, count in self.TILE_DISTRIBUTION.items()
            for _ in range(count)
        ]
        self.rng.shuffle(self.tile_bag)
        self.hand: list[str] = []
        self.draw_tiles(self.config.initial_hand_size)
        return self.state()

    def draw_tiles(self, count: int) -> list[str]:
        drawn: list[str] = []
        for _ in range(max(0, count)):
            if not self.tile_bag:
                break
            drawn_tile = self.tile_bag.pop()
            self.hand.append(drawn_tile)
            drawn.append(drawn_tile)
        return drawn

    def place(self, letter: str, row: int, col: int) -> dict[str, Any]:
        letter = self._normalize_letter(letter)
        self._validate_in_bounds(row, col)
        if self.grid[row][col]:
            raise InvalidAction("Cell is already occupied.")

        if letter not in self.hand:
            raise InvalidAction(f"Letter '{letter}' is not in hand.")

        self.hand.remove(letter)
        self.grid[row][col] = letter

        if not self._is_connected_board():
            self.grid[row][col] = ""
            self.hand.append(letter)
            raise InvalidAction("All placed tiles must remain connected.")

        valid_words, _invalid_words = self.validate_board_words()

        self._maybe_peel()
        return self.state(last_action="place", valid_words=valid_words)

    def remove(self, row: int, col: int) -> dict[str, Any]:
        self._validate_in_bounds(row, col)
        current = self.grid[row][col]
        if not current:
            raise InvalidAction("Cell is already empty.")

        self.grid[row][col] = ""
        if self._has_tiles() and not self._is_connected_board():
            self.grid[row][col] = current
            raise InvalidAction("Cannot disconnect the board.")

        self.hand.append(current)
        return self.state(last_action="remove")

    def dump(self, letter: str) -> dict[str, Any]:
        """Swap one hand tile for up to three fresh tiles from the bag."""
        letter = self._normalize_letter(letter)
        if letter not in self.hand:
            raise InvalidAction(f"Letter '{letter}' is not in hand.")

        self.hand.remove(letter)
        self.tile_bag.append(letter)
        self.rng.shuffle(self.tile_bag)
        self.draw_tiles(3)
        return self.state(last_action="dump")

    def validate_board_words(self) -> tuple[list[str], list[str]]:
        words = self._extract_words()
        invalid_words: list[str] = []
        valid_words: list[str] = []

        for word in words:
            if self.dictionary.contains(word):
                valid_words.append(word)
            else:
                invalid_words.append(word)

        return sorted(set(valid_words)), sorted(set(invalid_words))

    def is_finished(self) -> bool:
        return not self.hand and not self.tile_bag

    def state(self, last_action: str | None = None, valid_words: list[str] | None = None) -> dict[str, Any]:
        valid_words = valid_words if valid_words is not None else self.validate_board_words()[0]
        return {
            "grid_size": self.config.grid_size,
            "grid": self.grid,
            "hand": sorted(self.hand),
            "hand_count": len(self.hand),
            "tile_bag_count": len(self.tile_bag),
            "valid_words": valid_words,
            "is_finished": self.is_finished(),
            "last_action": last_action,
            "letter_histogram": dict(sorted(Counter(self.hand).items())),
        }

    def _normalize_letter(self, letter: str) -> str:
        if not isinstance(letter, str) or len(letter.strip()) != 1 or not letter.strip().isalpha():
            raise InvalidAction("Letter must be a single alphabetic character.")
        return letter.strip().upper()

    def _validate_in_bounds(self, row: int, col: int) -> None:
        if not (0 <= row < self.config.grid_size and 0 <= col < self.config.grid_size):
            raise InvalidAction("Position is outside grid bounds.")

    def _extract_words(self) -> list[str]:
        words: list[str] = []
        n = self.config.grid_size

        for r in range(n):
            current = ""
            for c in range(n):
                cell = self.grid[r][c]
                if cell:
                    current += cell
                else:
                    if len(current) >= 2:
                        words.append(current)
                    current = ""
            if len(current) >= 2:
                words.append(current)

        for c in range(n):
            current = ""
            for r in range(n):
                cell = self.grid[r][c]
                if cell:
                    current += cell
                else:
                    if len(current) >= 2:
                        words.append(current)
                    current = ""
            if len(current) >= 2:
                words.append(current)

        return words

    def _has_tiles(self) -> bool:
        for row in self.grid:
            for cell in row:
                if cell:
                    return True
        return False

    def _is_connected_board(self) -> bool:
        n = self.config.grid_size
        occupied: list[tuple[int, int]] = []
        for r in range(n):
            for c in range(n):
                if self.grid[r][c]:
                    occupied.append((r, c))

        if len(occupied) <= 1:
            return True

        seen = set()
        stack = [occupied[0]]

        while stack:
            r, c = stack.pop()
            if (r, c) in seen:
                continue
            seen.add((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < n and 0 <= nc < n and self.grid[nr][nc] and (nr, nc) not in seen:
                    stack.append((nr, nc))

        return len(seen) == len(occupied)

    def _maybe_peel(self) -> None:
        # Simplified peel: if hand empties, draw one new tile if available.
        if not self.hand and self.tile_bag:
            self.draw_tiles(1)
