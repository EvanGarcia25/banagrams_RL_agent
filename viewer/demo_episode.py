from cli import run_scripted_episode


SEED = 7
COMMANDS = [
    "place D 10 10",
    "place A 10 11",
    "place N 10 12",
    "place C 10 13",
    "place E 10 14",
]


def main():
    run_scripted_episode(
        COMMANDS,
        seed=SEED,
        record_path="viewer/episodes/demo_seed_7.json",
        pause_seconds=0.35,
        source="demo_episode",
        note="seeded demo episode",
    )


if __name__ == "__main__":
    main()
