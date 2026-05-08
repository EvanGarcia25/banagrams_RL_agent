# Bananagrams RL — Baseline

Single-player Bananagrams game with a REST API and live browser display. Models can drive the game over HTTP or directly via Python method calls.

---

## Setup & run

```bash
pip install -r requirements.txt
python3 server.py          # http://localhost:8080
```

---

## Project structure

```
banagrams_RL_agent/
├── game.py          # Core game logic — BananagramsGame class
├── dictionary.py    # Word validation (system dict or fallback)
├── server.py        # Flask server — API + browser UI (port 8080)
├── requirements.txt # flask>=3.0.0
├── static/
│   ├── app.js       # Polls /api/state every 500ms, renders display
│   └── style.css    # Dark grid UI
└── templates/
    └── index.html   # Full-page grid layout
```

---

## Game rules (singleplayer)

- **144 tiles**, standard distribution (A×13, E×18, I×12, …)
- **21 tiles** drawn to start; remaining 123 in the bag
- Place tiles on a **20×20 grid** (rows/cols 0–19) to form interlocking words
- **Peel** — hand empties while bag has tiles → 1 tile drawn automatically
- **Dump** — return 1 tile to bag, draw 3 (requires ≥3 in bag)
- **Win** — hand empty, bag empty, all words valid, board connected

---

## API reference

All `POST` action endpoints accept JSON and return `{"success": bool, "message": str, "state": {...}}`.

| Method | Route | Body | Action |
|--------|-------|------|--------|
| GET | `/api/state` | — | Full game state |
| POST | `/api/place` | `{"letter":"A","row":10,"col":10}` | Place tile from hand onto grid |
| POST | `/api/remove` | `{"row":10,"col":10}` | Return grid tile to hand |
| POST | `/api/dump` | `{"letter":"Q"}` | Return 1 tile, draw 3 |
| POST | `/api/reset` | — | Fresh game, new 21-tile hand |

### State object fields

| Key | Type | Description |
|---|---|---|
| `grid` | `list[list[str\|None]]` | 20×20 board; `None` = empty |
| `hand` | `list[str]` | Current tiles (sorted) |
| `bag_count` | `int` | Tiles remaining in bag |
| `words` | `list[str]` | All words (≥2 letters) on board |
| `invalid_words` | `list[str]` | Words failing dictionary check |
| `connected` | `bool` | All placed tiles form one group |
| `tile_count` | `int` | Tiles currently on grid |
| `done` | `bool` | Game ended |
| `won` | `bool` | Game ended by winning |
| `last_action` | `str` | Description of last move |

---

## Driving the game

### Option A — HTTP (out-of-process)

```python
import requests
BASE = "http://localhost:8080"

state = requests.get(f"{BASE}/api/state").json()
requests.post(f"{BASE}/api/place",  json={"letter": "C", "row": 10, "col": 10})
requests.post(f"{BASE}/api/remove", json={"row": 10, "col": 10})
requests.post(f"{BASE}/api/dump",   json={"letter": "Q"})
requests.post(f"{BASE}/api/reset")
```

### Option B — Direct Python (in-process)

```python
from server import start_background, game  # shared instance

start_background()   # browser display on port 8080, daemon thread

while not game.done:
    state = game.get_state()
    result = game.place(state["hand"][0], 10, 10)
    if not result["success"]:
        break

print("Won:", game.won)
```

Both options return the same response format. The browser display updates automatically either way.

---

## Tile distribution (144 total)

| A×13 | B×3 | C×3 | D×6 | E×18 | F×3 | G×4 | H×3 | I×12 | J×2 |
|------|-----|-----|-----|------|-----|-----|-----|------|-----|
| K×2 | L×5 | M×3 | N×8 | O×11 | P×3 | Q×2 | R×9 | S×6 | T×9 |
| U×6 | V×3 | W×3 | X×2 | Y×3 | Z×2 | | | | |
