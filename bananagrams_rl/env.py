from __future__ import annotations

from typing import Any

from .game import BananagramsGame, InvalidAction


class BananagramsEnv:
    """Tiny RL-style wrapper around BananagramsGame."""

    def __init__(self, game: BananagramsGame):
        self.game = game

    def reset(self) -> dict[str, Any]:
        return self.game.reset()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        action format examples:
        {'type': 'place', 'letter': 'A', 'row': 7, 'col': 8}
        {'type': 'remove', 'row': 7, 'col': 8}
        {'type': 'dump', 'letter': 'Q'}
        """
        action_type = (action or {}).get("type", "").lower().strip()
        reward = -0.01

        try:
            if action_type == "place":
                state = self.game.place(action["letter"], int(action["row"]), int(action["col"]))
                reward = 0.1
            elif action_type == "remove":
                state = self.game.remove(int(action["row"]), int(action["col"]))
                reward = -0.05
            elif action_type == "dump":
                state = self.game.dump(action["letter"])
                reward = -0.03
            else:
                raise InvalidAction("Unknown action type.")

            done = state["is_finished"]
            if done:
                reward += 1.0
            return state, reward, done, {"error": None}
        except (KeyError, ValueError, InvalidAction) as exc:
            state = self.game.state(last_action="invalid")
            return state, -0.2, False, {"error": str(exc)}
