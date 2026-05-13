"""
Terminal interface for playing Bananagrams solitaire.

Commands:
  place <letter> <row> <col>   — place a tile from hand onto the grid
  remove <row> <col>           — pick a tile back from the grid into hand
  dump <letter>                — return 1 tile, draw 3 (needs ≥3 in bag)
  reset                        — start a new game
  quit / exit                  — leave

Example:
  > place A 10 10
  > place T 10 11
  > remove 10 10
  > dump Q
"""

import os
from game import BananagramsGame

PADDING = 2          # empty cells to show around the placed-tile bounding box
MIN_VIEW  = 5        # minimum display size when nothing is placed yet


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def render(state: dict):
    grid        = state["grid"]
    hand        = state["hand"]
    bag_count   = state["bag_count"]
    words       = state["words"]
    invalid     = set(state["invalid_words"])
    connected   = state["connected"]
    tile_count  = state["tile_count"]
    last_action = state["last_action"]

    # --- find bounding box of placed tiles ---
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

    # --- column header ---
    col_range = range(c_min, c_max + 1)
    header = "     " + "".join(f"{c:2}" for c in col_range)
    print(header)
    print("     " + "--" * len(col_range))

    # --- grid rows ---
    for r in range(r_min, r_max + 1):
        row_str = f"{r:3} | "
        for c in col_range:
            ch = grid[r][c]
            row_str += f" {ch if ch else '.'}"
        print(row_str)

    print()

    # --- hand ---
    hand_str = " ".join(hand) if hand else "(empty)"
    print(f"Hand [{len(hand)}]: {hand_str}")
    print(f"Bag:  {bag_count} tiles remaining")
    print(f"Tiles on board: {tile_count}")

    # --- words ---
    if words:
        word_display = []
        for w in words:
            word_display.append(f"\033[91m{w}\033[0m" if w in invalid else w)
        print(f"Words: {', '.join(word_display)}", end="")
        if invalid:
            print(f"  ← {len(invalid)} invalid", end="")
        print()
    else:
        print("Words: (none yet)")

    # --- connectivity ---
    if tile_count > 0 and not connected:
        print("\033[93mWarning: tiles are not all connected.\033[0m")

    # --- last action ---
    print(f"\n→ {last_action}")
    print("-" * 50)


def print_help():
    print("""
Commands:
  place <letter> <row> <col>   place a tile (e.g.  place A 10 10)
  remove <row> <col>           pick tile back to hand
  dump <letter>                return 1 tile, draw 3 (needs ≥3 in bag)
  reset                        start a new game
  help                         show this message
  quit / exit                  leave
""")


def main():
    game = BananagramsGame()
    clear()
    print("=== Bananagrams Solitaire ===")
    print("Type 'help' for commands.\n")
    render(game.get_state())

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "exit"):
            print("Goodbye!")
            break

        elif cmd == "help":
            print_help()
            continue

        elif cmd == "reset":
            game.reset()
            clear()
            print("Game reset.")
            render(game.get_state())

        elif cmd == "place":
            if len(parts) != 4:
                print("Usage: place <letter> <row> <col>")
                continue
            _, letter, row, col = parts
            if not row.lstrip("-").isdigit() or not col.lstrip("-").isdigit():
                print("Row and col must be integers.")
                continue
            result = game.place(letter, int(row), int(col))
            clear()
            render(result["state"])
            if not result["success"]:
                print(f"\033[91mError: {result['message']}\033[0m")
            if result["state"]["done"]:
                if result["state"]["won"]:
                    print("\n\033[92m*** YOU WIN! Congratulations! ***\033[0m\n")
                else:
                    print("\nGame over.")

        elif cmd == "remove":
            if len(parts) != 3:
                print("Usage: remove <row> <col>")
                continue
            _, row, col = parts
            if not row.lstrip("-").isdigit() or not col.lstrip("-").isdigit():
                print("Row and col must be integers.")
                continue
            result = game.remove(int(row), int(col))
            clear()
            render(result["state"])
            if not result["success"]:
                print(f"\033[91mError: {result['message']}\033[0m")

        elif cmd == "dump":
            if len(parts) != 2:
                print("Usage: dump <letter>")
                continue
            result = game.dump(parts[1])
            clear()
            render(result["state"])
            if not result["success"]:
                print(f"\033[91mError: {result['message']}\033[0m")

        else:
            print(f"Unknown command '{cmd}'. Type 'help' for commands.")


if __name__ == "__main__":
    main()
