from __future__ import annotations

import random

import requests


BASE_URL = "http://127.0.0.1:5000"


def get_state():
    return requests.get(f"{BASE_URL}/api/state", timeout=10).json()


def send_action(action):
    response = requests.post(f"{BASE_URL}/api/action", json=action, timeout=10)
    return response.status_code, response.json()


def pick_random_place_action(state):
    if not state["hand"]:
        return None

    letter = random.choice(state["hand"])
    size = state["grid_size"]
    for _ in range(200):
        row = random.randrange(size)
        col = random.randrange(size)
        if state["grid"][row][col] == "":
            return {"type": "place", "letter": letter, "row": row, "col": col}
    return None


def main():
    state = get_state()

    for _ in range(50):
        action = pick_random_place_action(state)
        if not action:
            break

        status, payload = send_action(action)
        state = payload.get("state", state)
        print("status=", status, "action=", action, "reward=", payload.get("reward"))

        if payload.get("done"):
            print("Game finished")
            break


if __name__ == "__main__":
    main()
