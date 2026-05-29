from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_run_metadata(
    *,
    source: str,
    seed: int | None = None,
    training_step: int | None = None,
    episode_index: int | None = None,
    model_name: str | None = None,
    checkpoint_path: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "source": source,
        "seed": seed,
        "training_step": training_step,
        "episode_index": episode_index,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "note": note,
        "started_at_utc": utc_now_iso(),
    }


def build_episode_record(run_metadata: dict[str, Any], initial_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_metadata,
        "initial_state": deepcopy(initial_state),
        "events": [],
        "summary": {},
    }


def parse_command(raw_command: str) -> dict[str, Any]:
    parts = raw_command.strip().split()
    if not parts:
        return {"kind": "noop"}

    kind = parts[0].lower()
    if kind in {"quit", "exit", "help"}:
        return {"kind": kind}
    if kind == "place":
        if len(parts) != 4:
            raise ValueError("Usage: place <letter> <row> <col>")
        return {
            "kind": "place",
            "letter": parts[1],
            "row": int(parts[2]),
            "col": int(parts[3]),
        }
    if kind == "remove":
        if len(parts) != 3:
            raise ValueError("Usage: remove <row> <col>")
        return {"kind": "remove", "row": int(parts[1]), "col": int(parts[2])}
    if kind == "dump":
        if len(parts) != 2:
            raise ValueError("Usage: dump <letter>")
        return {"kind": "dump", "letter": parts[1]}
    if kind == "reset":
        if len(parts) == 1:
            return {"kind": "reset", "seed": None}
        if len(parts) == 2:
            return {"kind": "reset", "seed": int(parts[1])}
        raise ValueError("Usage: reset [seed]")

    raise ValueError(f"Unknown command '{parts[0]}'.")


def execute_command(game, command: dict[str, Any], *, default_seed: int | None = None) -> dict[str, Any]:
    kind = command["kind"]
    if kind == "place":
        return game.place(command["letter"], command["row"], command["col"])
    if kind == "remove":
        return game.remove(command["row"], command["col"])
    if kind == "dump":
        return game.dump(command["letter"])
    if kind == "reset":
        seed = command.get("seed", default_seed)
        game.reset(seed=seed)
        message = f"Game reset{f' with seed {seed}' if seed is not None else ''}."
        return {"success": True, "message": message, "state": game.get_state()}
    if kind in {"help", "quit", "exit", "noop"}:
        return {"success": True, "message": kind, "state": game.get_state()}
    raise ValueError(f"Unsupported command kind '{kind}'.")


def append_event(
    record: dict[str, Any],
    *,
    step_index: int,
    raw_command: str,
    command: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    event = {
        "step_index": step_index,
        "raw_command": raw_command,
        "command": command,
        "success": result["success"],
        "message": result["message"],
        "state_after": deepcopy(result["state"]),
    }
    record["events"].append(event)
    return event


def finalize_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    initial_state = record["initial_state"]
    final_state = record["events"][-1]["state_after"] if record["events"] else initial_state
    success_count = sum(1 for event in record["events"] if event["success"])
    record["summary"] = {
        "steps": len(record["events"]),
        "successful_steps": success_count,
        "failed_steps": len(record["events"]) - success_count,
        "won": final_state["won"],
        "done": final_state["done"],
        "bag_remaining_final": final_state["bag_count"],
        "hand_size_final": len(final_state["hand"]),
        "tile_count_final": final_state["tile_count"],
        "words_final": len(final_state["words"]),
        "invalid_words_final": len(final_state["invalid_words"]),
        "connected_final": final_state["connected"],
    }
    record["run"]["completed_at_utc"] = utc_now_iso()
    return record


def save_episode(record: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return output_path


def load_episode(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def episode_frames(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [episode["initial_state"], *[event["state_after"] for event in episode["events"]]]


def default_episode_path(*, source: str, seed: int | None = None, suffix: str = ".json") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_part = f"_seed-{seed}" if seed is not None else ""
    return Path("viewer/episodes") / f"{source}_{stamp}{seed_part}{suffix}"
