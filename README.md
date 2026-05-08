# Bananagrams RL Agent Playground

Minimal single-player Bananagrams clone in Python with:
- a browser UI for manual play
- a clean action API for RL/control agents
- a shared game engine used by both

## What Is Implemented

Core gameplay primitives (kept intentionally simple):
- Fixed square grid (`15x15` by default)
- Action: place one letter at a specific `(row, col)`
- Action: remove one letter from `(row, col)` back to hand
- Action: dump one hand letter and draw up to 3 replacements
- Dictionary validation for all contiguous row/column words (length >= 2)
- Connectivity rule: all placed tiles must stay in one connected component

This is designed so a model can call exact low-level actions such as:

```json
{"type": "place", "letter": "A", "row": 7, "col": 8}
```

## Project Structure

- `server.py`: Flask app + API endpoints
- `bananagrams_rl/game.py`: game rules/state/actions
- `bananagrams_rl/env.py`: tiny RL-style `step()` wrapper
- `bananagrams_rl/dictionary.py`: dictionary loading
- `templates/index.html`: browser UI
- `static/app.js`: UI logic/API calls
- `static/style.css`: styling
- `example_agent.py`: simple scripted client example
- `tests/test_game.py`: unit tests for core actions

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Web App

```bash
python server.py
```

Then open:

`http://127.0.0.1:5000`

## Dictionary Source

Priority order:
1. `words.txt` in project root (if present)
2. `/usr/share/dict/words` (default on macOS/Linux)
3. Small built-in fallback word set

To customize the lexicon, create a root-level `words.txt` with one word per line.

## RL / Agent API

### Get Current State

`GET /api/state`

### Reset Game

`POST /api/reset`

### Apply Action

`POST /api/action`

Request body examples:

```json
{"type": "place", "letter": "C", "row": 5, "col": 5}
```

```json
{"type": "remove", "row": 5, "col": 5}
```

```json
{"type": "dump", "letter": "Q"}
```

Response shape:

```json
{
	"state": {"grid": [], "hand": [], "valid_words": []},
	"reward": 0.1,
	"done": false,
	"info": {"error": null}
}
```

### Validate Current Board

`POST /api/validate`

Returns both valid and invalid discovered words from current board.

## Example Programmatic Agent

With server running:

```bash
python example_agent.py
```

This sends random coordinate placement actions to demonstrate machine interaction.

## Run Tests

```bash
python -m unittest discover -s tests
```
