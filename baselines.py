"""
Random and greedy baselines for Bananagrams solitaire.

Run evaluation:
  python baselines.py --agent random --episodes 50
  python baselines.py --agent greedy --episodes 50 --max-steps 5000
"""

from __future__ import annotations

import argparse
import random
from functools import partial
from typing import Callable

from game import GRID_SIZE, BananagramsGame

# Rough Scrabble-style letter weights: higher = more willing to dump (harder to use).
DUMP_WEIGHT = {
    "Q": 10,
    "Z": 9,
    "X": 8,
    "J": 7,
    "K": 6,
    "V": 5,
    "W": 5,
    "Y": 4,
    "B": 3,
    "C": 3,
    "F": 3,
    "G": 3,
    "H": 3,
    "M": 3,
    "P": 3,
    "D": 2,
    "L": 2,
    "N": 2,
    "R": 2,
    "S": 2,
    "T": 2,
}


def clone_game(game: BananagramsGame) -> BananagramsGame:
    """Shallow structural copy for simulating placements (shares Dictionary)."""
    g = object.__new__(BananagramsGame)
    g._dict = game._dict
    g.bag = game.bag.copy()
    g.hand = game.hand.copy()
    g.grid = [row[:] for row in game.grid]
    g.last_action = game.last_action
    g.done = game.done
    g.won = game.won
    return g


def _center_seed_cells() -> list[tuple[int, int]]:
    mid = GRID_SIZE // 2
    span = 2
    return [(r, c) for r in range(mid - span, mid + span + 1) for c in range(mid - span, mid + span + 1) if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE]


def adjacent_empty_cells(game: BananagramsGame) -> list[tuple[int, int]]:
    placed = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if game.grid[r][c]]
    if not placed:
        return _center_seed_cells()
    opts: set[tuple[int, int]] = set()
    for r, c in placed:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and game.grid[nr][nc] is None:
                opts.add((nr, nc))
    return list(opts)


def legal_placements(game: BananagramsGame) -> list[tuple[str, int, int]]:
    if not game.hand:
        return []
    cells = adjacent_empty_cells(game)
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for letter in set(game.hand):
        for r, c in cells:
            t = (letter, r, c)
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def legal_removes(game: BananagramsGame) -> list[tuple[int, int]]:
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if game.grid[r][c]]


def heuristic(state: dict) -> float:
    """Higher is better: valid words, connectivity, penalize invalid runs."""
    words = state["words"]
    invalid = set(state["invalid_words"])
    n_invalid = len(invalid)
    n_valid = sum(1 for w in words if w not in invalid)
    conn = 5.0 if state["connected"] else -30.0
    hand_pen = 0.02 * len(state["hand"])
    return -12.0 * n_invalid + 3.0 * n_valid + conn - hand_pen


def greedy_pick_placement(game: BananagramsGame, max_eval: int = 48) -> tuple[str, int, int, float] | None:
    moves = legal_placements(game)
    if not moves:
        return None
    if len(moves) > max_eval:
        moves = random.sample(moves, max_eval)
    best: tuple[str, int, int] | None = None
    best_h = float("-inf")
    for letter, r, c in moves:
        g = clone_game(game)
        res = g.place(letter, r, c)
        if not res["success"]:
            continue
        h = heuristic(res["state"])
        if h > best_h or (h == best_h and random.random() < 0.5):
            best_h = h
            best = (letter, r, c)
    if best is None:
        return None
    return (*best, best_h)


def greedy_pick_dump_letter(game: BananagramsGame) -> str:
    hand = list(game.hand)
    hand.sort(key=lambda L: (-DUMP_WEIGHT.get(L, 1), L))
    return hand[0]


def step_random(game: BananagramsGame, dump_prob: float = 0.06) -> dict:
    """One random action: usually random legal place, sometimes dump."""
    moves = legal_placements(game)
    if moves and random.random() >= dump_prob:
        letter, r, c = random.choice(moves)
        return game.place(letter, r, c)
    if len(game.bag) >= 3 and game.hand:
        letter = random.choice(game.hand)
        return game.dump(letter)
    if moves:
        letter, r, c = random.choice(moves)
        return game.place(letter, r, c)
    rems = legal_removes(game)
    if rems:
        r, c = random.choice(rems)
        return game.remove(r, c)
    return {"success": False, "message": "no legal action", "state": game.get_state()}


def step_greedy(game: BananagramsGame, *, max_eval: int = 48) -> dict:
    """Place at best heuristic neighbor; if no improvement, dump a heavy letter."""
    base = heuristic(game.get_state())
    picked = greedy_pick_placement(game, max_eval=max_eval)
    if picked is not None:
        letter, r, c, h_after = picked
        if h_after >= base - 0.01:
            return game.place(letter, r, c)
    if len(game.bag) >= 3 and game.hand:
        return game.dump(greedy_pick_dump_letter(game))
    if picked is not None:
        letter, r, c, _ = picked
        return game.place(letter, r, c)
    rems = legal_removes(game)
    if rems:
        r, c = random.choice(rems)
        return game.remove(r, c)
    return {"success": False, "message": "stuck", "state": game.get_state()}


def run_episode(step_fn: Callable[[BananagramsGame], dict], max_steps: int) -> dict:
    game = BananagramsGame()
    for step in range(max_steps):
        st = game.get_state()
        if st["done"]:
            return {"won": st["won"], "steps": step, "reason": "terminal"}
        out = step_fn(game)
        if not out["success"]:
            return {"won": False, "steps": step + 1, "reason": f"error:{out['message']}"}
    return {"won": False, "steps": max_steps, "reason": "max_steps"}


def main() -> None:
    p = argparse.ArgumentParser(description="Bananagrams baseline evaluation")
    p.add_argument("--agent", choices=("random", "greedy"), default="random")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy-max-eval", type=int, default=48, help="Greedy: max candidate placements sampled per step")
    args = p.parse_args()

    if args.agent == "random":
        step_fn: Callable[[BananagramsGame], dict] = step_random
    else:
        step_fn = partial(step_greedy, max_eval=args.greedy_max_eval)
    wins = 0
    for i in range(args.episodes):
        random.seed(args.seed + i)
        r = run_episode(step_fn, args.max_steps)
        wins += int(r["won"])
        print(f"episode {i + 1:3d}: won={r['won']!s:5} steps={r['steps']:5d}  ({r['reason']})")

    print(f"agent={args.agent}  win_rate={wins}/{args.episodes} = {wins / max(1, args.episodes):.3f}")


if __name__ == "__main__":
    main()
