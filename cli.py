"""Terminal interface for playing Bananagrams solitaire.

Commands:
  place <letter> <row> <col>   place a tile from hand onto the grid
  remove <row> <col>           pick a tile back from the grid into hand
  dump <letter>                return 1 tile, draw 3 (needs >=3 in bag)
  reset [seed]                 start a new game
  quit / exit                  leave

The runner can also replay a scripted list of commands and records every step
to a JSON episode log.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from viewer.episode_log import (
    append_event,
    build_episode_record,
    default_episode_path,
    execute_command,
    finalize_episode_record,
    load_episode,
    make_run_metadata,
    parse_command,
    save_episode,
)
from game import BananagramsGame

PADDING = 2
MIN_VIEW = 5


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def render(state: dict):
    grid = state["grid"]
    hand = state["hand"]
    bag_count = state["bag_count"]
    words = state["words"]
    invalid = set(state["invalid_words"])
    connected = state["connected"]
    tile_count = state["tile_count"]
    last_action = state["last_action"]

    placed_rows = [r for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c]]
    placed_cols = [c for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c]]

    if placed_rows:
        r_min = max(0, min(placed_rows) - PADDING)
        r_max = min(len(grid) - 1, max(placed_rows) + PADDING)
        c_min = max(0, min(placed_cols) - PADDING)
        c_max = min(len(grid[0]) - 1, max(placed_cols) + PADDING)
    else:
        mid = len(grid) // 2
        r_min, r_max = mid - MIN_VIEW, mid + MIN_VIEW
        c_min, c_max = mid - MIN_VIEW, mid + MIN_VIEW

    col_range = range(c_min, c_max + 1)
    header = "     " + "".join(f"{c:2}" for c in col_range)
    print(header)
    print("     " + "--" * len(col_range))

    for r in range(r_min, r_max + 1):
        row_str = f"{r:3} | "
        for c in col_range:
            ch = grid[r][c]
            row_str += f" {ch if ch else '.'}"
        print(row_str)

    print()

    hand_str = " ".join(hand) if hand else "(empty)"
    print(f"Hand [{len(hand)}]: {hand_str}")
    print(f"Bag:  {bag_count} tiles remaining")
    print(f"Tiles on board: {tile_count}")

    if words:
        word_display = []
        for w in words:
            word_display.append(f"\033[91m{w}\033[0m" if w in invalid else w)
        print(f"Words: {', '.join(word_display)}", end="")
        if invalid:
            print(f"  <- {len(invalid)} invalid", end="")
        print()
    else:
        print("Words: (none yet)")

    if tile_count > 0 and not connected:
        print("\033[93mWarning: tiles are not all connected.\033[0m")

    print(f"\n-> {last_action}")
    print("-" * 50)


def print_help() -> None:
    print(
        """
Commands:
  place <letter> <row> <col>   place a tile (e.g. place A 10 10)
  remove <row> <col>           pick tile back to hand
  dump <letter>                return 1 tile, draw 3 (needs >=3 in bag)
  reset [seed]                 start a new game
  help                         show this message
  quit / exit                  leave
"""
    )


def load_commands_file(path: str | Path) -> list[str]:
    commands: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        commands.append(raw)
    return commands


def _initialize_record(game: BananagramsGame, *, source: str, seed: int | None, note: str | None = None):
    run_metadata = make_run_metadata(source=source, seed=seed, note=note)
    return build_episode_record(run_metadata, game.get_state())


def run_scripted_episode(
    commands: list[str],
    *,
    seed: int | None = None,
    record_path: str | Path | None = None,
    pause_seconds: float = 0.0,
    clear_between_steps: bool = True,
    source: str = "scripted",
    note: str | None = None,
):
    game = BananagramsGame()
    game.reset(seed=seed)
    record = _initialize_record(game, source=source, seed=seed, note=note)

    if clear_between_steps:
        clear()
    print(f"=== Bananagrams Episode ({source}) ===")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Commands: {len(commands)}\n")
    render(game.get_state())

    event_index = 0
    for raw in commands:
        try:
            command = parse_command(raw)
        except ValueError as exc:
            print(f"\033[91mError: {exc}\033[0m")
            continue

        if command["kind"] == "help":
            print_help()
            continue
        if command["kind"] in {"quit", "exit"}:
            break

        result = execute_command(game, command, default_seed=seed)
        append_event(record, step_index=event_index, raw_command=raw, command=command, result=result)
        event_index += 1

        if clear_between_steps:
            clear()
        render(result["state"])

        if not result["success"]:
            print(f"\033[91mError: {result['message']}\033[0m")
        if result["state"]["done"]:
            if result["state"]["won"]:
                print("\n\033[92m*** YOU WIN! Congratulations! ***\033[0m\n")
            else:
                print("\nGame over.")
            break

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    finalize_episode_record(record)
    if record_path is None:
        record_path = default_episode_path(source=source, seed=seed)
    output_path = save_episode(record, record_path)
    print(f"Saved episode log to {output_path}")
    return record, output_path


def run_interactive_episode(
    *,
    seed: int | None = None,
    record_path: str | Path | None = None,
):
    game = BananagramsGame()
    game.reset(seed=seed)
    record = _initialize_record(game, source="interactive", seed=seed)

    clear()
    print("=== Bananagrams Solitaire ===")
    print("Type 'help' for commands.\n")
    if seed is not None:
        print(f"Seed: {seed}\n")
    render(game.get_state())

    event_index = 0
    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        try:
            command = parse_command(raw)
        except ValueError as exc:
            print(f"\033[91mError: {exc}\033[0m")
            continue

        if command["kind"] == "help":
            print_help()
            continue
        if command["kind"] in {"quit", "exit"}:
            print("Goodbye!")
            break

        result = execute_command(game, command, default_seed=seed)
        append_event(record, step_index=event_index, raw_command=raw, command=command, result=result)
        event_index += 1

        clear()
        if command["kind"] == "reset":
            print("Game reset.")
        render(result["state"])

        if not result["success"]:
            print(f"\033[91mError: {result['message']}\033[0m")
        if result["state"]["done"]:
            if result["state"]["won"]:
                print("\n\033[92m*** YOU WIN! Congratulations! ***\033[0m\n")
            else:
                print("\nGame over.")

    finalize_episode_record(record)
    if record_path is None:
        record_path = default_episode_path(source="interactive", seed=seed)
    output_path = save_episode(record, record_path)
    print(f"Saved episode log to {output_path}")
    return record, output_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Play or script Bananagrams solitaire")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic shuffle seed")
    parser.add_argument("--commands-file", type=str, default=None, help="Run commands from a text file")
    parser.add_argument("--record-path", type=str, default=None, help="Where to save the episode log")
    parser.add_argument("--pause", type=float, default=0.0, help="Pause between scripted commands")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between scripted steps")
    args = parser.parse_args(argv)

    if args.commands_file:
        commands = load_commands_file(args.commands_file)
        run_scripted_episode(
            commands,
            seed=args.seed,
            record_path=args.record_path,
            pause_seconds=args.pause,
            clear_between_steps=not args.no_clear,
            source=Path(args.commands_file).stem,
            note=f"commands_file={args.commands_file}",
        )
        return

    run_interactive_episode(seed=args.seed, record_path=args.record_path)


if __name__ == "__main__":
    main()