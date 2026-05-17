"""
Gymnasium wrapper for BananagramsGame.

Observation (Dict):
    grid      : (20, 20) int8   — 0 = empty, 1-26 = A-Z
    hand      : (26,)    int8   — count of each letter in hand
    bag_count : (1,)     int16  — tiles remaining in bag

Action space: Discrete(10_826)
    0       .. 10_399  : place(letter, row, col)  — letter*400 + row*20 + col
    10_400  .. 10_799  : remove(row, col)          — row*20 + col
    10_800  .. 10_825  : dump(letter)              — letter index

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
    terminated = True when game.done (win or — currently — only wins, since
    the game has no lose condition beyond max steps).
    truncated  = True when step count exceeds max_steps (default 2000).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import BananagramsGame, GRID_SIZE

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_TO_IDX = {ch: i for i, ch in enumerate(LETTERS)}

N_PLACE  = 26 * GRID_SIZE * GRID_SIZE   # 10 400
N_REMOVE = GRID_SIZE * GRID_SIZE        # 400
N_DUMP   = 26
N_ACTIONS = N_PLACE + N_REMOVE + N_DUMP  # 10 826


def _encode_action(action_type: str, *args) -> int:
    """Encode a (type, *args) tuple back into the flat action integer."""
    if action_type == "place":
        letter, row, col = args
        return LETTER_TO_IDX[letter.upper()] * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col
    if action_type == "remove":
        row, col = args
        return N_PLACE + row * GRID_SIZE + col
    # dump
    letter = args[0]
    return N_PLACE + N_REMOVE + LETTER_TO_IDX[letter.upper()]


def _decode_action(action: int) -> tuple:
    """Decode flat action integer → (type, *args)."""
    if action < N_PLACE:
        li  = action // (GRID_SIZE * GRID_SIZE)
        rem = action %  (GRID_SIZE * GRID_SIZE)
        return ("place", LETTERS[li], rem // GRID_SIZE, rem % GRID_SIZE)
    action -= N_PLACE
    if action < N_REMOVE:
        return ("remove", action // GRID_SIZE, action % GRID_SIZE)
    return ("dump", LETTERS[action - N_REMOVE])


class BananagramsEnv(gym.Env):
    """Gymnasium environment wrapping BananagramsGame."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None, max_steps: int = 2000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.game        = BananagramsGame()
        self._step_count = 0

        self.observation_space = spaces.Dict({
            "grid":      spaces.Box(0, 26, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8),
            "hand":      spaces.Box(0, 18, shape=(26,),                  dtype=np.int8),
            "bag_count": spaces.Box(0, 144, shape=(1,),                  dtype=np.int16),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

        # cached state to compute reward deltas
        self._prev_valid   = 0
        self._prev_invalid = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self._step_count = 0
        state = self.game.get_state()
        self._prev_valid   = len(state["words"]) - len(state["invalid_words"])
        self._prev_invalid = len(state["invalid_words"])
        return self._encode_obs(state), {}

    def step(self, action: int):
        decoded = _decode_action(int(action))
        state_before = self.game.get_state()
        prev_valid   = len(state_before["words"]) - len(state_before["invalid_words"])
        prev_invalid = len(state_before["invalid_words"])

        # Execute
        if decoded[0] == "place":
            result = self.game.place(decoded[1], decoded[2], decoded[3])
        elif decoded[0] == "remove":
            result = self.game.remove(decoded[1], decoded[2])
        else:
            result = self.game.dump(decoded[1])

        state = result["state"]
        obs   = self._encode_obs(state)
        self._step_count += 1

        # Reward
        reward = 0.0
        if not result["success"]:
            reward = -0.5          # illegal action — game state unchanged
        else:
            cur_valid   = len(state["words"]) - len(state["invalid_words"])
            cur_invalid = len(state["invalid_words"])
            if decoded[0] == "place":
                reward += (cur_valid   - prev_valid)   * 1.0
                reward -= (cur_invalid - prev_invalid) * 0.5
                reward -= 0.01
            elif decoded[0] == "remove":
                reward -= 0.01
            else:  # dump
                reward -= 0.2

        terminated = state["done"]
        if terminated and state["won"]:
            reward += 20.0

        truncated = (not terminated) and (self._step_count >= self.max_steps)

        if self.render_mode == "human":
            self.render()

        info = {
            "success":  result["success"],
            "message":  result["message"],
            "won":      state["won"],
            "words":    state["words"],
            "invalid":  state["invalid_words"],
            "connected": state["connected"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        state = self.game.get_state()
        grid  = state["grid"]
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

    # ------------------------------------------------------------------
    # Action masking (for sb3-contrib MaskablePPO)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean array of shape (N_ACTIONS,).
        True  = action is currently legal.
        False = action would fail (game returns success=False).

        Compatible with sb3_contrib.MaskablePPO which calls env.action_masks().
        """
        state    = self.game.get_state()
        grid     = state["grid"]
        hand_set = set(state["hand"])
        bag_ok   = state["bag_count"] >= 3
        mask     = np.zeros(N_ACTIONS, dtype=bool)

        if self.game.done:
            return mask  # all False — episode over

        # place: letter in hand AND cell empty
        for letter in hand_set:
            li = LETTER_TO_IDX[letter]
            base = li * GRID_SIZE * GRID_SIZE
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if grid[r][c] is None:
                        mask[base + r * GRID_SIZE + c] = True

        # remove: cell occupied
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c] is not None:
                    mask[N_PLACE + r * GRID_SIZE + c] = True

        # dump: letter in hand AND bag has ≥3 tiles
        if bag_ok:
            for letter in hand_set:
                mask[N_PLACE + N_REMOVE + LETTER_TO_IDX[letter]] = True

        return mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, state: dict) -> dict:
        grid_arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ch = state["grid"][r][c]
                if ch:
                    grid_arr[r, c] = LETTER_TO_IDX[ch] + 1  # 1–26; 0 = empty

        hand_arr = np.zeros(26, dtype=np.int8)
        for letter in state["hand"]:
            hand_arr[LETTER_TO_IDX[letter]] += 1

        bag_arr = np.array([state["bag_count"]], dtype=np.int16)
        return {"grid": grid_arr, "hand": hand_arr, "bag_count": bag_arr}

    @staticmethod
    def encode_action(action_type: str, *args) -> int:
        """Utility: convert a human-readable action to its integer index."""
        return _encode_action(action_type, *args)

    @staticmethod
    def decode_action(action: int) -> tuple:
        """Utility: convert an integer action back to a human-readable tuple."""
        return _decode_action(action)
