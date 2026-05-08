import random
from collections import deque
from dictionary import Dictionary

TILE_DISTRIBUTION = {
    'A': 13, 'B': 3,  'C': 3,  'D': 6,  'E': 18, 'F': 3,
    'G': 4,  'H': 3,  'I': 12, 'J': 2,  'K': 2,  'L': 5,
    'M': 3,  'N': 8,  'O': 11, 'P': 3,  'Q': 2,  'R': 9,
    'S': 6,  'T': 9,  'U': 6,  'V': 3,  'W': 3,  'X': 2,
    'Y': 3,  'Z': 2,
}  # 144 total

GRID_SIZE = 20
STARTING_HAND = 21


class BananagramsGame:
    def __init__(self):
        self._dict = Dictionary.load()
        self.reset()

    def reset(self):
        self.bag: list[str] = []
        for letter, count in TILE_DISTRIBUTION.items():
            self.bag.extend([letter] * count)
        random.shuffle(self.bag)

        self.hand: list[str] = [self.bag.pop() for _ in range(STARTING_HAND)]
        self.grid: list[list[str | None]] = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.last_action: str = "Game started. 21 tiles drawn."
        self.done: bool = False
        self.won: bool = False

    # ------------------------------------------------------------------ actions

    def place(self, letter: str, row: int, col: int) -> dict:
        letter = letter.upper()
        if self.done:
            return self._err("Game is over.")
        if letter not in self.hand:
            return self._err(f"'{letter}' not in hand.")
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            return self._err(f"Position ({row},{col}) out of bounds.")
        if self.grid[row][col] is not None:
            return self._err(f"Cell ({row},{col}) already occupied by '{self.grid[row][col]}'.")

        self.hand.remove(letter)
        self.grid[row][col] = letter

        if not self.hand and self.bag:
            drawn = self.bag.pop()
            self.hand.append(drawn)
            self.last_action = f"Placed {letter} at ({row},{col}). Hand empty — peel! Drew '{drawn}'."
        else:
            self.last_action = f"Placed {letter} at ({row},{col})."

        self._check_win()
        return {"success": True, "message": self.last_action, "state": self.get_state()}

    def remove(self, row: int, col: int) -> dict:
        if self.done:
            return self._err("Game is over.")
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            return self._err(f"Position ({row},{col}) out of bounds.")
        letter = self.grid[row][col]
        if letter is None:
            return self._err(f"Cell ({row},{col}) is empty.")

        self.grid[row][col] = None
        self.hand.append(letter)
        self.last_action = f"Removed '{letter}' from ({row},{col}) back to hand."
        return {"success": True, "message": self.last_action, "state": self.get_state()}

    def dump(self, letter: str) -> dict:
        letter = letter.upper()
        if self.done:
            return self._err("Game is over.")
        if letter not in self.hand:
            return self._err(f"'{letter}' not in hand.")
        if len(self.bag) < 3:
            return self._err(f"Need ≥3 tiles in bag to dump (bag has {len(self.bag)}).")

        self.hand.remove(letter)
        self.bag.append(letter)
        random.shuffle(self.bag)
        drawn = [self.bag.pop() for _ in range(3)]
        self.hand.extend(drawn)
        self.last_action = f"Dumped '{letter}', drew {drawn}."
        return {"success": True, "message": self.last_action, "state": self.get_state()}

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict:
        words = self._get_words()
        invalid = [w for w in words if not self._dict.is_valid(w)]
        placed = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c]]
        return {
            "grid": self.grid,
            "hand": sorted(self.hand),
            "bag_count": len(self.bag),
            "words": words,
            "invalid_words": invalid,
            "connected": self._is_connected(),
            "tile_count": len(placed),
            "done": self.done,
            "won": self.won,
            "last_action": self.last_action,
        }

    # ------------------------------------------------------------------ internals

    def _get_words(self) -> list[str]:
        words = []
        for r in range(GRID_SIZE):
            run = ""
            for c in range(GRID_SIZE):
                ch = self.grid[r][c]
                if ch:
                    run += ch
                else:
                    if len(run) >= 2:
                        words.append(run)
                    run = ""
            if len(run) >= 2:
                words.append(run)

        for c in range(GRID_SIZE):
            run = ""
            for r in range(GRID_SIZE):
                ch = self.grid[r][c]
                if ch:
                    run += ch
                else:
                    if len(run) >= 2:
                        words.append(run)
                    run = ""
            if len(run) >= 2:
                words.append(run)

        return words

    def _is_connected(self) -> bool:
        placed = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c]]
        if not placed:
            return True
        visited = set()
        queue = deque([placed[0]])
        visited.add(placed[0])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    if self.grid[nr][nc]:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return len(visited) == len(placed)

    def _check_win(self):
        if self.hand or self.bag:
            return
        placed = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c]]
        if not placed:
            return
        words = self._get_words()
        if words and all(self._dict.is_valid(w) for w in words) and self._is_connected():
            self.done = True
            self.won = True
            self.last_action = "You win! All tiles placed, all words valid."

    def _err(self, msg: str) -> dict:
        return {"success": False, "message": msg, "state": self.get_state()}
