# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
pip install -r requirements.txt   # flask>=3.0.0 only
python3 server.py                  # http://localhost:8080
```

No build step. No test suite. Python 3.12+.

## Architecture

Flat, two-layer design — no packages, no RL wrapper.

**Layer 1 — game logic (`game.py`)**  
`BananagramsGame` holds all mutable state (`bag`, `hand`, `grid`) and exposes:

- `place(letter, row, col)` — moves tile from `hand` onto `grid[row][col]`; auto-peels when hand empties and bag is non-empty
- `remove(row, col)` — returns grid tile to `hand`
- `dump(letter)` — returns 1 tile to bag, draws 3; requires `len(bag) >= 3`
- `reset()` — rebuilds 144-tile bag, draws 21 to hand, clears grid
- `get_state()` — serialisable snapshot: `grid, hand, bag_count, words, invalid_words, connected, done, won, last_action`

Win is detected inside `place()` via `_check_win()`: hand empty + bag empty + all words valid + board connected.

**Layer 2 — server (`server.py`)**  
Flask app on port **8080** with a single module-level `game` instance. Exposes REST endpoints for both display and model interaction (see below). `start_background()` runs Flask in a daemon thread so an in-process agent can share the same `game` object while the browser polls live.

**Dictionary (`dictionary.py`)**  
`Dictionary.load()` tries `/usr/share/dict/words` first, falls back to a hardcoded word set. Used only at `BananagramsGame` construction time.

## API endpoints (`server.py`)

| Method | Route | Body | Description |
|--------|-------|------|-------------|
| GET | `/` | — | Browser UI |
| GET | `/api/state` | — | Full game state JSON |
| POST | `/api/place` | `{"letter","row","col"}` | Place tile |
| POST | `/api/remove` | `{"row","col"}` | Remove tile |
| POST | `/api/dump` | `{"letter"}` | Dump tile (−1, +3) |
| POST | `/api/reset` | — | Reset game |

All action responses: `{"success": bool, "message": str, "state": {...}}`.

## Key constants (`game.py`)

| Name | Value | Meaning |
|---|---|---|
| `GRID_SIZE` | 20 | Grid is 20×20, coordinates 0–19 |
| `STARTING_HAND` | 21 | Tiles drawn at game start |
| `TILE_DISTRIBUTION` | 144 total | Standard Bananagrams distribution |

## Model interaction patterns

**Via HTTP (out-of-process agent):**
```python
import requests
BASE = "http://localhost:8080"
requests.post(f"{BASE}/api/place",  json={"letter":"C","row":10,"col":10})
requests.post(f"{BASE}/api/remove", json={"row":10,"col":10})
requests.post(f"{BASE}/api/dump",   json={"letter":"Q"})
requests.post(f"{BASE}/api/reset")
state = requests.get(f"{BASE}/api/state").json()
```

**Via direct Python (in-process agent):**
```python
from server import start_background, game  # shared instance

start_background()   # browser display on port 8080

while not game.done:
    state = game.get_state()
    result = game.place(state["hand"][0], 10, 10)
    if not result["success"]:
        break
```
