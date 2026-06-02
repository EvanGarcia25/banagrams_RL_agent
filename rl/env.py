"""
Gymnasium wrapper for BananagramsGame.

Observation (Dict):
    grid      : (20, 20) int8   - 0 = empty, 1-26 = A-Z
    hand      : (26,)    int8   - count of each letter in hand
    bag_count : (1,)     int16  - tiles remaining in bag

Action space: Discrete(10_826)
    0       .. 10_399  : place(letter, row, col)  - letter*400 + row*20 + col
    10_400  .. 10_799  : remove(row, col)          - row*20 + col
    10_800  .. 10_825  : dump(letter)              - letter index

Action masking:
    Call env.action_masks() to get a (10_826,) bool array of currently-legal
    actions. Plug directly into sb3-contrib MaskablePPO or any masking wrapper.

Reward shaping:
    +1.0  per net-new valid word formed by a placement
    -0.5  per net-new invalid word created
    -0.01 step cost for place / remove
    -0.2  for dump
    -0.5  for attempting an illegal action (the step is a no-op)
    +20.0 on win (done=True, won=True)

Episode termination:
    terminated = True when game.done (win or - currently - only wins, since
    the game has no lose condition beyond max steps).
    truncated  = True when step count exceeds max_steps (default 2000).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import BananagramsGame, GRID_SIZE, TILE_DISTRIBUTION
from dictionary import Dictionary

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_TO_IDX = {ch: i for i, ch in enumerate(LETTERS)}

N_PLACE = 26 * GRID_SIZE * GRID_SIZE   # 10 400
N_REMOVE = GRID_SIZE * GRID_SIZE        # 400
N_DUMP = 26
N_ACTIONS = N_PLACE + N_REMOVE + N_DUMP  # 10 826


def _encode_action(action_type: str, *args) -> int:
    """Encode a (type, *args) tuple back into the flat action integer."""
    if action_type == "place":
        letter, row, col = args
        return LETTER_TO_IDX[letter.upper()] * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col
    if action_type == "remove":
        row, col = args
        return N_PLACE + row * GRID_SIZE + col
    letter = args[0]
    return N_PLACE + N_REMOVE + LETTER_TO_IDX[letter.upper()]


def _decode_action(action: int) -> tuple:
    """Decode flat action integer -> (type, *args)."""
    if action < N_PLACE:
        li = action // (GRID_SIZE * GRID_SIZE)
        rem = action % (GRID_SIZE * GRID_SIZE)
        return ("place", LETTERS[li], rem // GRID_SIZE, rem % GRID_SIZE)
    action -= N_PLACE
    if action < N_REMOVE:
        return ("remove", action // GRID_SIZE, action % GRID_SIZE)
    return ("dump", LETTERS[action - N_REMOVE])


# Cache for letter weights
_LETTER_WEIGHTS = None

def get_letter_weights() -> dict[str, float]:
    global _LETTER_WEIGHTS
    if _LETTER_WEIGHTS is not None:
        return _LETTER_WEIGHTS

    d = Dictionary.load()
    dict_freqs = {ch: 0 for ch in LETTERS}
    for w in d._words:
        for ch in w.upper():
            if ch in dict_freqs:
                dict_freqs[ch] += 1
                
    max_dict_f = max(dict_freqs.values()) if dict_freqs else 1
    dict_weights = {ch: dict_freqs[ch] / max_dict_f for ch in LETTERS}
    
    max_bag_f = max(TILE_DISTRIBUTION.values())
    bag_weights = {ch: count / max_bag_f for ch, count in TILE_DISTRIBUTION.items()}
    
    _LETTER_WEIGHTS = {ch: (dict_weights[ch] + bag_weights[ch]) / 2.0 for ch in LETTERS}
    return _LETTER_WEIGHTS


class BananagramsEnv(gym.Env):
    """Gymnasium environment wrapping BananagramsGame."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None, max_steps: int = 300, alpha: float = 0.05, beta: float = 0.05):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.alpha = alpha
        self.beta = beta
        self.game = BananagramsGame()
        self._step_count = 0

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 26, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8),
            "hand": spaces.Box(0, 18, shape=(26,), dtype=np.int8),
            "bag_count": spaces.Box(0, 72, shape=(1,), dtype=np.int16),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._prev_valid = 0
        self._prev_invalid = 0
        
    def _calc_porosity_and_hook(self, grid: list[list[str | None]]) -> tuple[int, float]:
        porosity = 0
        hook_val = 0.0
        weights = get_letter_weights()
        
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ch = grid[r][c]
                if ch is not None:
                    empty_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                            if grid[nr][nc] is None:
                                empty_neighbors += 1
                    
                    porosity += empty_neighbors
                    hook_val += empty_neighbors * weights[ch.upper()]
                    
        return porosity, hook_val

    def _get_clusters(self, grid: list[list[str | None]]) -> list[set[tuple[int, int]]]:
        placed = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c] is not None:
                    placed.add((r, c))
        
        clusters = []
        unvisited = placed.copy()
        while unvisited:
            start = unvisited.pop()
            cluster = {start}
            queue = [start]
            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in unvisited:
                        unvisited.remove((nr, nc))
                        cluster.add((nr, nc))
                        queue.append((nr, nc))
            clusters.append(cluster)
        return clusters

    def _get_word_coords(self, grid: list[list[str | None]]) -> list[tuple[str, set[tuple[int, int]]]]:
        words_info = []
        for r in range(GRID_SIZE):
            run = ""
            coords = set()
            for c in range(GRID_SIZE):
                ch = grid[r][c]
                if ch:
                    run += ch
                    coords.add((r, c))
                else:
                    if len(run) >= 2:
                        words_info.append((run, coords))
                    run = ""
                    coords = set()
            if len(run) >= 2:
                words_info.append((run, coords))

        for c in range(GRID_SIZE):
            run = ""
            coords = set()
            for r in range(GRID_SIZE):
                ch = grid[r][c]
                if ch:
                    run += ch
                    coords.add((r, c))
                else:
                    if len(run) >= 2:
                        words_info.append((run, coords))
                    run = ""
                    coords = set()
            if len(run) >= 2:
                words_info.append((run, coords))
        return words_info

    def _calc_valid_words_score(self, grid: list[list[str | None]]) -> float:
        clusters = self._get_clusters(grid)
        if not clusters:
            return 0.0
        main_grid = max(clusters, key=len)
        
        words_info = self._get_word_coords(grid)
        score = 0.0
        for w, coords in words_info:
            if self.game._dict.is_valid(w) and not coords.isdisjoint(main_grid):
                score += ((len(w) - 1.5) ** 2) / 2.0
        return score

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._step_count = 0
        state = self.game.get_state()
        self._prev_valid_score = self._calc_valid_words_score(state["grid"])
        return self._encode_obs(state), {}

    def step(self, action: int):
        decoded = _decode_action(int(action))
        state_before = self.game.get_state()
        prev_porosity, prev_hook = self._calc_porosity_and_hook(state_before["grid"])

        if decoded[0] == "place":
            result = self.game.place(decoded[1], decoded[2], decoded[3])
        elif decoded[0] == "remove":
            result = self.game.remove(decoded[1], decoded[2])
        else:
            result = self.game.dump(decoded[1])

        state = result["state"]
        obs = self._encode_obs(state)
        self._step_count += 1
        
        cur_porosity, cur_hook = self._calc_porosity_and_hook(state["grid"])
        delta_p = cur_porosity - prev_porosity
        delta_h = cur_hook - prev_hook

        reward = 0.0
        if not result["success"]:
            reward = -0.5
        else:
            cur_valid_score = self._calc_valid_words_score(state["grid"])
            delta_valid_score = cur_valid_score - self._prev_valid_score
            self._prev_valid_score = cur_valid_score
            
            if decoded[0] == "place":
                reward += delta_valid_score - 0.05
            elif decoded[0] == "remove":
                reward -= 0.05
            else:
                reward -= 0.2
                
            reward += (delta_p * self.alpha) + (delta_h * self.beta)

        reward -= (len(state["invalid_words"]) * 0.1)

        terminated = state["done"]
        clusters = self._get_clusters(state["grid"])
        D = max(0, len(clusters) - 1)
        
        if len(state["invalid_words"]) > 4 or D > 3:
            terminated = True

        truncated = (not terminated) and (self._step_count >= self.max_steps)
        done = terminated or truncated

        if done:
            if terminated and state["won"]:
                reward += 20.0
            else:
                H = len(state["hand"])
                I = len(state["invalid_words"])
                terminal_penalty = - (5.0 * H) - (50.0 * I) - (10.0 * D)
                reward += terminal_penalty

        if self.render_mode == "human":
            self.render()

        info = {
            "success": result["success"],
            "message": result["message"],
            "won": state["won"],
            "words": state["words"],
            "invalid": state["invalid_words"],
            "connected": state["connected"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        state = self.game.get_state()
        grid = state["grid"]
        placed_rows = [r for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c]]
        placed_cols = [c for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c]]
        if placed_rows:
            r0 = max(0, min(placed_rows) - 1)
            r1 = min(GRID_SIZE - 1, max(placed_rows) + 1)
            c0 = max(0, min(placed_cols) - 1)
            c1 = min(GRID_SIZE - 1, max(placed_cols) + 1)
        else:
            mid = GRID_SIZE // 2
            r0, r1, c0, c1 = mid - 2, mid + 2, mid - 2, mid + 2

        lines = [f"Step {self._step_count}  |  Bag: {state['bag_count']}  |  Hand: {' '.join(state['hand'])}"]
        for r in range(r0, r1 + 1):
            lines.append(f"{r:2} | " + " ".join(grid[r][c] or "." for c in range(c0, c1 + 1)))
        lines.append(f"Words: {state['words']}   Invalid: {state['invalid_words']}")
        print("\n".join(lines) + "\n")
        return "\n".join(lines)

    def action_masks(self) -> np.ndarray:
        state = self.game.get_state()
        grid = state["grid"]
        hand_set = set(state["hand"])
        bag_ok = state["bag_count"] >= 3
        mask = np.zeros(N_ACTIONS, dtype=bool)

        if self.game.done:
            return mask

        is_empty_board = state["tile_count"] == 0

        for letter in hand_set:
            li = LETTER_TO_IDX[letter]
            base = li * GRID_SIZE * GRID_SIZE
            
            if is_empty_board:
                r, c = GRID_SIZE // 2, GRID_SIZE // 2
                mask[base + r * GRID_SIZE + c] = True
            else:
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        if grid[r][c] is None:
                            has_adj = False
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                                    if grid[nr][nc] is not None:
                                        has_adj = True
                                        break
                            if has_adj:
                                mask[base + r * GRID_SIZE + c] = True

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c] is not None:
                    mask[N_PLACE + r * GRID_SIZE + c] = True

        if bag_ok:
            for letter in hand_set:
                mask[N_PLACE + N_REMOVE + LETTER_TO_IDX[letter]] = True

        return mask

    def _encode_obs(self, state: dict) -> dict:
        grid_arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ch = state["grid"][r][c]
                if ch:
                    grid_arr[r, c] = LETTER_TO_IDX[ch] + 1

        hand_arr = np.zeros(26, dtype=np.int8)
        for letter in state["hand"]:
            hand_arr[LETTER_TO_IDX[letter]] += 1

        bag_arr = np.array([state["bag_count"]], dtype=np.int16)
        return {"grid": grid_arr, "hand": hand_arr, "bag_count": bag_arr}

    @staticmethod
    def encode_action(action_type: str, *args) -> int:
        return _encode_action(action_type, *args)

    @staticmethod
    def decode_action(action: int) -> tuple:
        return _decode_action(action)
